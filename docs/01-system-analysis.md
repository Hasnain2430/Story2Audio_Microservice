# 01 — System Analysis (v1, as-built)

Audit of the existing Story2Audio codebase as it stands on `main`. Every claim below is
traced to a file in the current tree. This document is the baseline the v2 rebuild is
measured against.

---

## 1. What the system does

Takes a one-line story prompt plus voice/emotion/length settings, asks a local LLM to
write a full story, then renders that story to a `.wav` with a cloned voice, and hands
back both the text and the audio bytes.

## 2. Actual runtime topology

```
Streamlit process (streamlit_ms.py)
  └── ThreadPoolExecutor(3)  ──gRPC unary──►  gRPC process (server_ms.py)  :50051
                                                 ├── XTTS v2        (in-process, CUDA, ~3GB VRAM)
                                                 ├── DistilRoBERTa emotion clf (in-process)
                                                 ├── MarianMT opus-mt-en-XX  (lazy, lru_cache 5)
                                                 └── ollama.chat()  ──HTTP──► Ollama daemon :11434
                                                                                 ├── llama3.2:1b
                                                                                 ├── mistral:7b-instruct
                                                                                 └── llama3

Flask process (rest_server.py) :8000  ──gRPC unary──► same gRPC process
```

Both the gRPC server and Streamlit run **inside one container** via `start.sh`
(`python server_ms.py &` then `streamlit run`). Ollama is not containerised at all — it is
assumed to already be running on the host.

## 3. The blocking-call problem (the headline issue)

`StoryServiceServicer.GenerateStory` in `server_ms.py` is a **unary** RPC. The client
thread is held open for the entire duration of LLM generation *plus* full TTS synthesis.

Measured in the v1 README (RTX 3050):

| Paragraph range | Narration only | Narration + dialogue |
|---|---:|---:|
| 1–3  | ~1.8 min | ~2.8 min |
| 4–7  | ~7.0 min | ~9.0 min |
| 8+   | ~8.0 min | ~9.9 min |

So the p95 request holds a socket, a gRPC worker thread, a Streamlit script-run, and the
GPU lock for **up to ten minutes**. Consequences:

- Any proxy, load balancer, or browser in front of this times out long before the response
  arrives. No deadline is set anywhere, so gRPC's own defaults do not rescue it either.
- A page refresh in Streamlit destroys the `ThreadPoolExecutor` and every in-flight future.
  The work continues on the server; the result is unreachable forever.
- No cancellation. Closing the tab does not stop GPU work.
- No retry. A transient Ollama hiccup nine minutes in loses the whole job.
- No progress. The user gets a spinner for ten minutes with zero signal.

## 4. Concurrency is fake

`server_ms.py` creates `futures.ThreadPoolExecutor(max_workers=5)` for gRPC, and
`streamlit_ms.py` creates its own `ThreadPoolExecutor(max_workers=3)`. Both are defeated
by a module-scope mutex:

```python
tts_lock = threading.Lock()
...
with tts_lock:
    tts.tts_to_file(...)
```

Every synthesis call in the process is serialised on one global lock. The lock is
*correct* — a single XTTS instance is not thread-safe — but it means five gRPC workers
queue behind one GPU with no visibility, no fairness, and no bound. Requests six through
N simply sit in the thread pool's unbounded queue.

The README's "⚡ Concurrent Processing" claim does not survive contact with this lock.

## 5. Correctness and security defects

### 5.1 Global shared `chat_history` — the worst bug in the repo

```python
chat_history = []                                   # module scope

def get_llama3_response(user_input, split_voices):
    chat_history.append({"role": "user", "content": user_input})
    ...
    response = ollama.chat(model=model_name, messages=chat_history, ...)
    chat_history.append({"role": "assistant", "content": reply})
```

One list, module-global, shared by **every request from every user**, never trimmed.

- User B's story is generated with User A's prompt and User A's full story in context.
  That is a cross-tenant data leak and a quality bug at the same time.
- It grows without bound. After ~20 requests the context blows past the model window and
  Ollama silently truncates from the front, so behaviour degrades non-deterministically.
- It is mutated from multiple gRPC worker threads with no lock — interleaved appends can
  produce a malformed `messages` array.

There is no reason for conversational history to exist here at all; each generation is
independent.

### 5.2 Filename derived from user input

```python
def sanitize_filename(prompt_text, speaker_display_name, max_length=70):
    base = prompt_text.strip().title()
    base = re.sub(r'[^\w\s-]', '', base)
    ...
    return os.path.join(OUTPUT_DIR, filename)
```

The regex does strip path separators, so traversal is blocked — but two users submitting
the same prompt with the same speaker write to the **same file**, concurrently, from
different threads. `generate_narration_only_audio` writes then immediately re-reads that
path, so a colliding request returns the other user's audio. `output/` also grows forever
with no eviction.

### 5.3 Fixed-path speaker upload in the REST proxy

```python
speaker_path = "uploaded_speaker.wav"
with open(speaker_path, "wb") as f:
    f.write(base64.b64decode(speaker_audio_b64))
```

Hardcoded filename, process-wide. Two concurrent REST calls and the second overwrites the
first's reference voice before the first has read it. The response then returns
`"audio_file": "response_audio.wav"` — a path on the *server's* disk, meaningless to a
remote caller, and a fixed path with the identical race.

### 5.4 Speaker path crosses the wire as a string

`StoryRequest.speaker_audio` is a `string` filesystem path chosen by the client. It works
only because client and server share a filesystem. The moment they are separate containers
this breaks; and the raw path is handed straight to `tts.tts_to_file(speaker_wav=...)`.
A client controlling that string controls which file the server opens.

### 5.5 Structured data smuggled through free text

Story length is transported as a sentinel embedded in the prompt:

```python
if "[PARA_LEVEL:1–3]" in user_input:      # note: en-dash U+2013, not hyphen
```

The frontend inserts it, the server string-matches it, then three separate
`re.sub(r'\[PARA_LEVEL:.*?\]', '', ...)` calls strip it back out in three different files.
A user typing `[PARA_LEVEL:8+]` into their own prompt changes routing. The en-dash means a
hyphen typed by hand silently falls through to the `else` branch and the slowest model.

### 5.6 No input validation, auth, quota, or rate limit

Anything reachable at `:50051` or `:8000` can queue unbounded GPU work. `speed` is cast
with `float()` and passed through unchecked. `language` is unchecked and used to build a
HuggingFace model id: `f"Helsinki-NLP/opus-mt-{src_lang}-{tgt_lang}"` — a remote model
download driven by an unvalidated client string.

### 5.7 Prompt injection is wide open

User text is concatenated directly onto the system instructions
(`full_prompt = f"{prompt}{stripped_input}"`) with no delimiting, no role separation, and
no output filtering.

### 5.8 Silent failure

```python
except:
    return "neutral"
```

A bare `except` in `detect_emotion` swallows everything including `KeyboardInterrupt`.
The gRPC handler's `except Exception as e` returns `message="error"` with the raw
exception string in gRPC details — internal paths and stack info leaked to the client.

## 6. Features that don't actually do anything

- **`emotion=` on XTTS.** `tts.tts_to_file(..., emotion=emotion, ...)` — XTTS v2 does not
  implement emotion conditioning through this argument; Coqui accepts and ignores it. The
  emotion dropdown changes nothing audible. The `emotion_classifier` pipeline (a full
  DistilRoBERTa loaded at boot) feeds only this dead parameter.
- **Translation path.** For `language != "en"` the story is generated in English, then
  each segment is round-tripped through MarianMT sentence-by-sentence, then synthesised.
  Two lossy hops. The LLM can write in the target language directly.
- **Subtitles.** `changes.txt` line 10 admits the feature is broken; there is no subtitle
  code in the tree at all.
- **`xtts_model/` directory.** Present with `config.json`, `vocab.json`,
  `speakers_xtts.pth`, but `model.pth` is gitignored, and the local-path loader is
  commented out in favour of `TTS(model_name="tts_models/...xtts_v2")`, which downloads
  from the internet at boot. The vendored directory is dead weight.
- **Dialogue voice is hardcoded.** `dialogue_voice = "voices/female.wav"` — the "multi
  speaker" feature is one narrator voice plus one fixed female voice.

## 7. Packaging and operations

- **`requirements.txt` is UTF-16-encoded with zero version pins.** An unpinned
  `TTS` + `transformers` + `torch` triple is guaranteed to break on a future resolve.
  `torch==2.5.1+cu121` is pinned but the base image is `python:3.10-slim` with no CUDA
  runtime — the GPU path cannot work in the container as written.
- **Dockerfile layer order is inverted.** `COPY . .` precedes `pip install`, so editing one
  line of Python re-downloads and rebuilds multi-gigabyte torch wheels. No multi-stage
  build, no `--no-cache-dir`, no non-root user.
- **`COPY . .` ships `TestCases.json` (2.3 MB of base64 WAV), `performance_graph.png`, and
  all 16 reference voices** into the image.
- **Two processes, one container, no supervisor.** If `server_ms.py` dies, `start.sh`
  keeps the container alive on Streamlit alone and every request fails. No healthcheck,
  no restart policy, no readiness signal.
- **Ollama is an undeclared host dependency.** Not in the Dockerfile, not in a compose
  file (there is no compose file), just assumed present on `localhost:11434`.
- **Audio returned as protobuf `bytes`.** Forces
  `grpc.max_receive_message_length = 100 MB` on the client. Everything is buffered in RAM
  at least three times (server read → proto → client write).
- **No tests, no CI, no structured logging, no metrics, no tracing.** Observability is
  `print("🚀 Starting gRPC server...")`. `TestCases.json` is a fixture file with no runner.

## 8. Frontend

`streamlit_ms.py` is ~160 lines. Its problems are structural, not cosmetic:

- State lives in `st.session_state` — per-browser-session and in-process. Refresh or
  reconnect and every job and result is gone.
- Progress is a `time.sleep(2); st.rerun()` busy-loop that re-executes the entire script,
  re-opens the gRPC channel, and re-reads `speakers.json` every two seconds.
- Results are written to `generated_audio_{uuid}.wav` in the working directory and never
  cleaned up.
- The whole form is one `st.form` with three submit buttons (`Upload File`, `Record
  Voice`, `Generate Audio`), so any submit re-runs all the branch checks.
- The 15-second minimum for a reference voice is enforced as `len(audio_bytes) < 15000`
  — a byte count, not a duration. 15000 bytes of 44.1 kHz 16-bit mono is ~0.17 seconds.
- No routes, no deep links, no shareable result, no history, no mobile layout.

## 9. Model selection is coupled to the wrong thing

```
1–3 paragraphs → llama3.2:1b
4–7 paragraphs → mistral:7b-instruct
8+  paragraphs → llama3
```

Requested *length* selects model *quality*. A user who wants a short, well-written story
is forced onto a 1B model. `changes.txt` states the reasoning was RAM and speed — a
deployment constraint leaking into product behaviour. All three paths pass
`num_predict: 2000`, which is below what the 800–1200-word prompt asks for, so long
stories are truncated mid-sentence — the exact failure the prompt tries to forbid
("PLEASE MAKE SURE THE STORY HAS A PROPER END").

## 10. What is worth keeping

- The product idea and the prompt library. The six prompt variants encode real iteration
  and should be ported — parameterised rather than copy-pasted six times.
- The reference voice pack (16 voices) and `speakers.json` as a seed catalogue.
- Segment splitting into narration vs. dialogue (`split_into_narration_and_dialogues`) and
  the silence-trim / fade / join logic (`trim_silence`, the 300 ms inter-segment pad).
  That audio post-processing is the right approach and ports directly.
- `TestCases.json` as regression fixtures, once there is a runner.
- The performance table — it is the "before" number that makes the v2 story concrete.

## 11. Defect summary

| # | Defect | Severity | Where |
|---|---|---|---|
| 1 | Global `chat_history` shared across all users | Critical | `server_ms.py` |
| 2 | Up-to-10-minute blocking unary RPC | Critical | `server_ms.py`, proto |
| 3 | Client-supplied filesystem path used as `speaker_wav` | Critical | `server_ms.py` |
| 4 | Fixed-path speaker/response files, concurrent overwrite | High | `rest_server.py` |
| 5 | Output filename derived from prompt → collisions | High | `server_ms.py` |
| 6 | No auth / rate limit / quota on GPU work | High | all entrypoints |
| 7 | Global `tts_lock` makes declared concurrency fictional | High | `server_ms.py` |
| 8 | Unvalidated `language` builds a remote model id | High | `server_ms.py` |
| 9 | Prompt injection, no role separation | High | `server_ms.py` |
| 10 | Two processes, one container, no supervisor/healthcheck | Medium | `Dockerfile`, `start.sh` |
| 11 | `requirements.txt` UTF-16, unpinned | Medium | `requirements.txt` |
| 12 | Docker layer order forces full torch rebuild per edit | Medium | `Dockerfile` |
| 13 | `emotion` parameter is a no-op | Medium | `server_ms.py` |
| 14 | 100 MB audio blobs through protobuf | Medium | `proto/story_service.proto` |
| 15 | Streamlit state lost on refresh; 2 s rerun polling | Medium | `streamlit_ms.py` |
| 16 | Voice length check is a byte count, not a duration | Medium | `streamlit_ms.py` |
| 17 | `[PARA_LEVEL:...]` sentinel in free text, en-dash fragile | Medium | 3 files |
| 18 | `num_predict: 2000` truncates long stories | Medium | `server_ms.py` |
| 19 | Bare `except:`; exception text leaked to client | Low | `server_ms.py` |
| 20 | No tests, CI, logging, metrics, tracing | Low | repo-wide |
