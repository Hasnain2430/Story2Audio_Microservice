# Story2Audio

Give it a premise; it writes a short story and reads it aloud in cloned voices — a
narrator, plus a voice for each character when the story has dialogue.

It is built as a job-based pipeline. The API answers in milliseconds with a job id, and
the work happens behind a queue: one worker writes the story with an LLM, another turns it
into speech with XTTS v2 on a GPU. The browser follows along live, and starts playing the
opening of the story while the rest is still being recorded.

> v1 — a Streamlit client over a gRPC call that blocked for up to ten minutes — is
> preserved in [`legacy/`](./legacy/). It is there for comparison, not to be run.

---

## Measured

On an RTX 3050 4 GB laptop GPU, fp32 XTTS v2, Groq `openai/gpt-oss-120b`:

| | v1 | v2 |
|---|---|---|
| Request returns | after up to 10 minutes | **~190 ms**, with a job id |
| Story text visible | when everything is done | **~2 s**, streaming |
| First audio playable | when everything is done | **8.3 s** |
| Whole recording | 7–10 minutes | **60 s** for about three minutes of audio |
| Page refresh | loses the job | loses nothing |
| Cancel | not possible | stops the GPU mid-job |

---

## Architecture

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

| Service | Responsibility |
|---|---|
| `gateway` | HTTP and WebSocket API, validation, rate limits, signing URLs. Holds no model |
| `story-worker` | Writes the story and streams it token by token |
| `tts-worker` | Splits the story into segments, casts each character to a voice, assembles and uploads the audio |
| `tts-engine` | Holds XTTS in GPU memory and streams PCM over gRPC |

### What it does

- **Streams at every stage.** Story text arrives token by token. Audio is published a
  segment at a time as it is rendered, and played back to back through the Web Audio API
  while the rest is still being made.
- **Casts dialogue.** Each quoted line is attributed to the character who says it — from
  speech tags, action beats, pronouns, and turn-taking — and each character is read in
  their own voice. Pauses follow the text: short where a sentence was split for length,
  longer at a paragraph, longer still when the speaker changes.
- **Reads along.** The story highlights itself as it is narrated, dialogue tinted by
  character, and any word can be clicked to jump there.
- **Clones voices** from an uploaded clip or one recorded in the browser.
- **Survives the browser.** Every job has its own URL from the moment it exists; close the
  tab, come back, and the recording is still going — or finished.

---

## Running it

You need **Docker**, **Node 22+**, and **[uv](https://docs.astral.sh/uv/)**. A GPU is
optional; without one the stack runs end to end, but the voice engine renders a test tone
rather than speech.

**1. Configure.**

```sh
cp .env.example .env
```

Pick a story model in `.env`:

| To use | Set |
|---|---|
| Groq (hosted) | `LLM_PROVIDER=groq`, `LLM_MODEL=openai/gpt-oss-120b`, `GROQ_API_KEY=…` |
| Ollama (local) | `LLM_PROVIDER=ollama`, `LLM_MODEL=llama3`, with Ollama running |
| Nothing at all | `LLM_PROVIDER=fake` — canned text, useful for checking the plumbing |

**2. Start the backend.**

```sh
docker compose --env-file .env -f infra/docker-compose.yml up --build
```

This brings up Postgres, Redis, MinIO, the gateway on `:8000`, both workers, and the voice
engine, after a one-shot job that runs the database migrations and seeds the voice
catalogue. Pass `--env-file` explicitly: Compose otherwise looks for `.env` next to the
compose file, in `infra/`, and silently ignores the one you just made.

**3. Start the frontend.**

```sh
cd web
npm install
npm run dev
```

Open **http://localhost:5173**.

### Real voices

With an NVIDIA GPU and the NVIDIA Container Toolkit, add the GPU overlay:

```sh
docker compose --env-file .env \
  -f infra/docker-compose.yml -f infra/docker-compose.gpu.yml up --build
```

On Windows, where a laptop GPU cannot be passed through to Docker, run the engine natively
and point the worker at it — see [`infra/README.md`](./infra/README.md).

To confirm you are hearing speech and not the test tone:

```sh
uv run python infra/scripts/run_job.py --save out.wav   # submit a job, follow it, save the WAV
uv run python infra/scripts/check_audio.py out.wav      # "SPEECH" or "TONE"
```

### Tests

```sh
uv sync --all-packages --all-groups
uv run python infra/scripts/gen_proto.py
uv run pytest tests/unit                           # 555 tests, no Docker needed
cd web && npm test                                 # 17 tests
```

With `make`: `make setup`, `make check` runs everything CI runs, and `make up` / `make up-gpu`
start the stack.

---

## Layout

```
packages/shared/     the domain: enums, errors, events, schemas, models, prompts, storage
services/gateway/    FastAPI app, Alembic migrations, voice catalogue seeding
services/story_worker/
services/tts_worker/ segmentation, speaker attribution, casting, assembly
services/tts_engine/ XTTS behind gRPC, with its own CUDA Dockerfile
proto/               the gRPC contract between tts-worker and tts-engine
web/                 React + TypeScript frontend
infra/               compose files, the shared Python Dockerfile, operational scripts
tests/               unit and end-to-end tests
docs/                the audit of v1, what changed, and design decisions
legacy/              v1, preserved for comparison
```

## Documentation

- [**What changed from v1**](./docs/02-what-changed.md) — the result, the architecture, and
  each of v1's twenty defects with what replaced it.
- [**The v1 audit**](./docs/01-system-analysis.md) — the evidence behind every claim about
  how v1 behaved.
- [**Design decisions**](./docs/adr/) — what was chosen, what was turned down, and why.
