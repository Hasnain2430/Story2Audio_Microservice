# 02 — What Changed, v1 to v2

[01](./01-system-analysis.md) audited v1 and catalogued twenty defects. This is the other
half: what v2 does instead, and what that measures as. The v1 sources are preserved in
[`legacy/`](../legacy/) so the two can be compared directly.

Every number here was measured on the development machine — RTX 3050 4 GB laptop GPU, fp32
XTTS v2, Groq `openai/gpt-oss-120b` — rather than estimated.

---

## The result

| | v1 | v2 |
|---|---|---|
| Request returns | after up to 10 minutes | **~190 ms**, with a job id |
| Story text visible | when everything is done | **~2 s**, streaming token by token |
| First audio playable | when everything is done | **8.3 s** |
| Whole recording | 7–10 minutes | **60 s** for about three minutes of audio |
| Progress | none | real segment counts |
| Page refresh | loses the job | loses nothing |
| Cancel | not possible | stops the GPU mid-job |
| Dialogue in two-voice mode | 1 line of 12 segments | 8 lines of 21, two characters |

The first row is the headline. The third is the one that matters to someone listening: the
opening of the story plays while the rest is still being recorded.

---

## Architecture

v1 was one gRPC process holding XTTS, an emotion classifier and a translator, answering a
single unary RPC that blocked for the whole job, with Streamlit calling it synchronously.

v2 splits along the lines that differ in how they scale and fail:

```
React / TypeScript
        │  REST + WebSocket
        ▼
   FastAPI gateway ──────────► Postgres   (jobs, voices, the timeline)
        │  Celery over Redis
        ├────────────► story-worker ─────► LLM provider (Groq, or Ollama locally)
        │                    │
        │              Redis pub/sub ◄────── events, per-job sequence numbers
        │                    │
        └────────────► tts-worker ──gRPC──► tts-engine (XTTS v2, GPU)
                             │
                             ▼
                      object storage ──► presigned URLs to the browser
```

| Service | Responsibility | Why it is separate |
|---|---|---|
| `gateway` | HTTP and WebSocket API, validation, quotas, signing URLs | Must answer in milliseconds; holds no model |
| `story-worker` | Writes the story, streams tokens | Network-bound, cheap, scales on CPU |
| `tts-worker` | Segments the story, casts voices, assembles and uploads audio | CPU work that must not hold GPU memory |
| `tts-engine` | Holds XTTS in VRAM, streams PCM over gRPC | The one expensive resource, shareable across workers |

The design decisions that were not obvious have their own records in [`adr/`](./adr/):
one shared database and model layer, UUIDv7 identifiers, a discriminated-union event
contract with monotonic sequence numbers, and gRPC kept only on the worker-to-engine link.

---

## The twenty v1 defects

Numbered as in [01 §11](./01-system-analysis.md#11-defect-summary).

### Critical

**1. Global `chat_history` shared across every user.** Removed. Each job carries its own
prompt and story in its own row; there is no process-wide conversational state.

**2. Up-to-ten-minute blocking RPC.** Replaced by a queue. `POST /v1/jobs` returns
`202 Accepted` with a job id in about 190 ms, and the work happens on the far side of Redis.
The client watches over a WebSocket or polls — both write the same cache, so a dropped
socket degrades to polling with nothing downstream able to tell the difference.

**3. Client-supplied filesystem path used as `speaker_wav`.** The client names a voice by
opaque id. The gateway resolves it against the catalogue and the engine receives reference
audio bytes; no client-controlled path reaches the TTS engine.

### High

**4. Fixed-path speaker and response files, overwritten under concurrency.** Voices are
stored under their UUID in object storage. When the engine needs a file on disk it writes a
uniquely named temporary file and removes it immediately.

**5. Output filename derived from the prompt, so identical prompts collided.** Audio is
keyed by job id.

**6. No rate limit or quota on GPU work.** Each session is held to an hourly sliding-window
limit, and the whole service to a daily cap on jobs, both enforced in Redis before anything
is queued.

**7. A global `tts_lock` made the declared concurrency fictional.** Replaced by an explicit
bounded semaphore sized from configuration, so the engine's concurrency is a stated number
rather than five threads queued behind one mutex.

**8. Unvalidated `language` used to build a remote model id.** `language` is an enum. The
translation hop that consumed it is gone entirely — the model writes in the target language
directly.

**9. Prompt injection with no role separation.** Instructions are a system message; the
user's storyline is a separate, delimited user message, sanitised so it cannot close the
delimiter early.

### Medium

**10. Two processes in one container, no supervisor or healthcheck.** One service per
container. The gateway, the TTS engine and every datastore carry healthchecks, and services
start only once what they depend on reports healthy; a one-shot `migrate` job runs Alembic
and seeds the voice catalogue before anything else starts.

**11. `requirements.txt` in UTF-16, unpinned.** A `uv` workspace with a single lockfile.

**12. Docker layer order rebuilt torch on every edit.** Dependencies install before source
is copied. The GPU image is a separate build target, so the default image carries no torch.

**13. `emotion` was a no-op.** XTTS v2 accepts and ignores it. Emotion now steers the
writing instead, where it demonstrably changes what is heard.

**14. 100 MB audio blobs inside protobuf messages.** Audio never travels through the API.
The engine streams PCM in bounded chunks with no message-size overrides, and the browser
fetches finished audio from object storage through a short-lived presigned URL.

**15. Streamlit state lost on refresh, polled every two seconds.** A React frontend. Every
job has its own URL from the moment it exists, and that URL survives a reload, a reconnect,
or coming back the next day.

**16. Voice length checked as a byte count.** v1's "15 second minimum" was
`len(bytes) < 15000` — about a sixth of a second. Uploads are decoded, and the real
duration is checked in the browser before upload and again on the server.

**17. `[PARA_LEVEL:...]` sentinel smuggled through free text.** Length is a structured field
on the request.

**18. `num_predict: 2000` truncated long stories.** Each length has its own token budget,
and a story that stops short gets one bounded continuation pass.

### Low

**19. Bare `except:`, with exception text returned to the client.** Every failure is
classified into an error code with a fixed public message and a retryable flag. Raw
exception text goes to the logs only.

**20. No tests, CI, logging or metrics.** 555 Python unit tests and 17 TypeScript tests. CI
runs lint, strict typechecking and the tests for both halves, a format check on the Python,
and a production build of the frontend. Structured logging in every service; Prometheus
metrics on the gateway.

---

## What v2 adds

### Streaming at every stage

- **Story text** streams token by token over the WebSocket, so writing is visible within
  about two seconds.
- **Audio** is published a segment at a time as it is rendered. Each segment is uploaded
  and announced the moment it exists, and the browser schedules them with the Web Audio
  API so they play back to back while the rest is still being made. First audio at 8.3
  seconds, against 60 for the whole file.
- **Recovery.** The timeline is persisted as each segment is rendered, so a browser that
  reloads mid-job picks up everything already made rather than starting from nothing.

### Dialogue that sounds like a scene

v1 asked the model for *"ONE character dialogue ... from a female character"*, and sent
every quoted line to one hardcoded voice.

- The prompt asks for **two named characters** and at least six alternating lines.
- Each line is **attributed to a speaker**: from explicit tags ("Mara said"), from action
  beats ("Tom set the cloth aside. '…'"), from the name within the same paragraph, from a
  pronoun tag resolved by elimination, and finally by alternation. A line that resolves to
  nothing falls back to the first character's voice — a fallback is unremarkable, where a
  confidently wrong voice is immediately audible.
- Each character is **cast to their own voice**; the narrator has a third.
- **Pauses depend on the boundary.** 80 ms where the text was split only for length, 300 ms
  at a paragraph break, 420 ms when the speaker changes. v1 used one gap everywhere, which
  chopped narration into equal slabs and let a reply land on top of the line it answered.

### Reading along

The worker records where every segment lands in the finished track, measured from the
audio it assembled. The story highlights itself as it is read, dialogue tinted by
character, and any word can be clicked to seek there.

Segment boundaries are measured. Word positions within a segment are interpolated from
character offsets, because XTTS returns no per-word alignment — so the highlight follows
along closely rather than claiming to be a forced alignment.

### The frontend

- **Compose** — a premise, a length, a register, a language, a pace, and a cast: narrator,
  first character, second character.
- **The recording page** — a pipeline showing each stage as it completes, a segment meter,
  a live listening control, and a *tape*: one block per segment, coloured by who is
  speaking, filling from the left as the story is recorded. It doubles as the transport.
- **Voices** — the built-in catalogue with previews, and your own voice added by upload or
  recorded in the browser.
- **History** — every job, paginated.
- Hand-written CSS on design tokens, with full light and dark themes, and reduced-motion
  respected throughout.

### LLM providers

v1 was bound to a local Ollama instance. v2 puts the model behind one interface with an
OpenAI-compatible implementation (Groq by default) and an Ollama implementation, selected by
configuration. Reasoning models are handled explicitly: their reasoning tokens are budgeted
separately from the prose, so a model that thinks at length cannot spend the whole story's
allowance before writing a word.

---

## Audio quality

**Generation settings come from the model's own configuration.** XTTS's high-level wrapper
overrides the raw method defaults with tuned values before generating. Bypassing the wrapper
to cache speaker embeddings would otherwise silently take the untuned defaults — twice the
intended repetition penalty, and a fifth of the speaker conditioning — which rushes and
slurs speech.

**Reference clips are thirty seconds**, matching how much of the reference XTTS actually
reads. Longer uploads are trimmed; high sample-rate uploads are shortened to stay within
the transport's message limit rather than resampled.

**Segments respect the model's per-language input limit** — 250 characters for English, 182
for Russian. Past it XTTS truncates the audio and only logs about it. A sentence longer
than the limit is broken at a clause boundary instead.

**Speech is verified by spectrum, not by loudness.** `infra/scripts/check_audio.py` tells
real speech from a test tone using single-bin energy, spectral flatness and spectral flux.
Peak level and RMS cannot make that distinction, and a sine wave passes both.

---

## Engineering

- **A `uv` workspace**: one shared package (`packages/shared`) for the domain, and one
  package per service, all against a single lockfile.
- **Strict typing throughout**: `mypy --strict` on every Python package; TypeScript with
  `strict`, `exactOptionalPropertyTypes` and `noUncheckedIndexedAccess`.
- **Generated client types.** The gateway's OpenAPI schema is committed, and the frontend's
  types are generated from it, so a breaking API change fails the frontend build.
- **Idempotent workers.** Celery delivers at least once. Each stage checks its own job row
  before acting, status moves by compare-and-set, and a Redis lease stops two deliveries of
  one job both paying for the model.
- **Explicit errors.** One enumeration of error codes, each mapped to an HTTP status, a
  public message and a retryability flag, checked for completeness at import.
- **Object storage lifecycle.** Streamed segments live under their own prefix and expire
  after a day, so early playback does not accumulate storage for every job ever run.

---

## What was kept from v1

The product itself; the prompt library, deduplicated from six near-identical strings into
one template; the reference voice pack; the idea of splitting narration from dialogue so
they can be voiced differently; Docker; and gRPC — no longer the public API, but the
internal streaming contract between the TTS worker and the GPU engine, which is the one
place its strengths actually apply.
