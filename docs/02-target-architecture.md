# 02 — Target Architecture (v2)

The v1 system is a monolith wearing a microservice costume: one process holds the LLM
client, the TTS model, the emotion classifier and the translator, and answers a single
blocking RPC that can run for ten minutes. v2 keeps the *product* and replaces the
*shape*.

The one-line story: **the original version blocked for up to ten minutes on a single
synchronous RPC, so it was redesigned around a job queue with independently scalable LLM
and TTS workers, progress streamed to the client over WebSocket.**

---

## 1. Design goals

| Goal | v1 | v2 target |
|---|---|---|
| Time to first response | up to 10 min | < 200 ms (`202 Accepted` + job id) |
| Time to first visible output | 10 min | ~2 s (story tokens streaming) |
| Survives page refresh | no | yes (job state is durable) |
| Cancellable | no | yes |
| Retryable | no | yes, per stage, with backoff |
| Independently scalable stages | no | yes (LLM and TTS scale separately) |
| Concurrency | 1 (global lock) | N (bounded by GPU/API concurrency, queued fairly) |
| Multi-tenant safe | no (shared `chat_history`) | yes (no shared mutable state) |
| Cost at idle | a GPU box running 24/7 | ~$0 (scale-to-zero TTS, free tiers elsewhere) |

## 2. Service map

```
                         ┌──────────────────────────────┐
  Browser                │  web  (React + TypeScript)   │   Vercel / Cloudflare Pages
  ───────────────────────┤  Vite · TanStack Query · WS  │
                         └───────────────┬──────────────┘
                              REST + WebSocket (HTTPS/WSS)
                                         │
                         ┌───────────────▼──────────────┐
                         │  gateway  (FastAPI, async)   │   1 small container
                         │  · POST /v1/jobs  → 202 + id │
                         │  · GET  /v1/jobs/{id}        │
                         │  · WS   /v1/jobs/{id}/events │
                         │  · voice catalog + uploads   │
                         │  · auth, rate limit, quota   │
                         └───┬───────────────────────┬──┘
                 enqueue     │                       │  subscribe
                             ▼                       ▲
                  ┌──────────────────────────────────────────┐
                  │  Redis   queue (Celery broker)           │   Upstash free tier
                  │          pub/sub (progress fan-out)      │
                  │          short-lived cache / idempotency │
                  └───┬─────────────────────────┬────────────┘
                      │ story queue             │ tts queue
                      ▼                         ▼
          ┌───────────────────────┐   ┌────────────────────────┐
          │  story-worker         │   │  tts-worker            │
          │  Celery, async        │   │  Celery                │
          │  · prompt assembly    │   │  · segment the story   │
          │  · LLM provider call  │   │  · call tts-engine     │
          │    (streams tokens)   │   │  · trim/fade/join      │
          │  · publish progress   │   │  · encode mp3 + wav    │
          │  · chain → tts queue  │   │  · upload to storage   │
          └──────────┬────────────┘   └───────┬────────────────┘
                     │ HTTPS                  │ gRPC (streamed PCM)
                     ▼                        ▼
          ┌───────────────────────┐   ┌────────────────────────┐
          │  LLM provider         │   │  tts-engine (gRPC, GPU)│  scale-to-zero
          │  Groq / Gemini /      │   │  · XTTS v2 in VRAM     │  serverless GPU
          │  OpenRouter / Ollama  │   │  · speaker embedding   │  (or hosted TTS API
          │  (adapter interface)  │   │    cache               │   behind same iface)
          └───────────────────────┘   └────────────────────────┘

          ┌───────────────────────┐   ┌────────────────────────┐
          │  Postgres             │   │  Object storage (S3)   │
          │  jobs, voices, users  │   │  Cloudflare R2         │
          │  Neon free tier       │   │  audio + reference wav │
          └───────────────────────┘   └────────────────────────┘
```

## 3. Why each boundary exists

A service boundary that isn't justified is just latency. Each one here earns its place:

- **gateway / workers** — the gateway must stay sub-second and horizontally scalable;
  the workers are long-running and resource-bound. Different scaling curve, different
  failure mode, different deploy cadence.
- **story-worker / tts-worker** — the LLM stage is network-bound and cheap; the TTS stage
  is GPU-bound and expensive. Splitting them means a queue of stories waiting on a busy
  GPU doesn't also stall LLM generation, and each scales on its own signal. This is the
  split that makes the queue worth having.
- **tts-worker / tts-engine** — the worker does CPU work (segmentation, `pydub` stitching,
  encoding, upload) and must not occupy a GPU while doing it. The engine holds the model
  in VRAM and does nothing else. Separating them means one GPU serves many workers, and
  the GPU process can be a scale-to-zero serverless container while the workers stay warm.
  **This is where gRPC is kept**: internal, strongly typed, and streaming raw PCM chunks
  back — the exact workload gRPC is good at, and the reason `proto/` survives the rewrite
  rather than being deleted.
- **Object storage** — audio leaves the request path entirely. No more 100 MB protobuf
  messages, no more `max_receive_message_length` hacks; the client gets a presigned URL.

## 4. Request lifecycle

```
1.  POST /v1/jobs {prompt, length, voiceId, mode, language, speed}
    gateway validates → writes jobs row (status=queued) → enqueues story task
    → 202 {jobId, status: "queued"}                                    (< 200 ms)

2.  Client opens WS /v1/jobs/{jobId}/events
    gateway subscribes to Redis channel job:{jobId}

3.  story-worker picks up task
    → status=writing
    → streams LLM tokens, publishes  {type:"token", text:"..."}         (~2 s to first token)
    → story complete, persisted
    → publishes {type:"story_done", text}
    → chains tts task

4.  tts-worker picks up task
    → status=synthesizing
    → splits into N segments
    → per segment: gRPC Synthesize → publishes {type:"progress", done:i, total:N}
    → joins, encodes, uploads to R2
    → status=done, writes audioUrl
    → publishes {type:"done", audioUrl, durationSec}

5.  Client renders player. WS closes.
    Any later GET /v1/jobs/{jobId} returns the same terminal state, forever.
```

The client never waits on a long request. If the socket drops, it falls back to polling
`GET /v1/jobs/{id}` and loses nothing — the WebSocket is an optimisation over a
poll-correct design, not a requirement of it.

## 5. Job state machine

```
queued ──► writing ──► written ──► synthesizing ──► done
   │          │                         │
   │          └────► failed ◄───────────┘
   │                   ▲
   └──► cancelled ─────┘
```

- Each transition is a single Postgres `UPDATE` with the previous status in the `WHERE`
  clause, so a duplicate worker delivery cannot move a job backwards.
- `failed` carries a machine-readable `errorCode` plus a user-safe `errorMessage`. Raw
  exception text is logged, never returned — fixing v1 defect 19.
- `cancelled` is checked by the worker between segments, so cancellation actually frees
  the GPU instead of being cosmetic.
- Retries are per-stage with exponential backoff and a cap. Because the story is persisted
  at `written`, a TTS failure retries **only** the TTS stage — it does not re-run the LLM.
  That alone removes most of the cost of a failure.

## 6. API surface

```
POST   /v1/jobs                    → 202 {jobId, status}
GET    /v1/jobs/{id}               → job record (status, story, audioUrl, timings, error)
GET    /v1/jobs?cursor=&limit=     → paginated history for the caller
DELETE /v1/jobs/{id}               → request cancellation
WS     /v1/jobs/{id}/events        → token / progress / done / failed events

GET    /v1/voices                  → catalogue (built-in + caller's own)
POST   /v1/voices                  → multipart upload, returns voiceId
DELETE /v1/voices/{id}             → caller's own only

GET    /healthz  /readyz           → liveness / readiness
GET    /metrics                    → Prometheus
```

Design notes:

- `POST /v1/jobs` accepts an `Idempotency-Key` header. Same key within the TTL returns the
  same `jobId` instead of queueing duplicate GPU work.
- Length is a first-class enum field (`short` | `medium` | `long`), **not** a
  `[PARA_LEVEL:...]` sentinel spliced into the prompt — fixing v1 defect 17.
- `voiceId` is an opaque identifier resolved server-side against the catalogue. The client
  never sends a filesystem path — fixing v1 defects 3 and 4.
- Every request body is a Pydantic model with bounds (`speed` in `[0.5, 1.5]`, `language`
  from a closed enum, prompt length capped) — fixing v1 defects 6 and 8.

## 7. Data model

```sql
users      (id, email, created_at)                       -- anonymous session id is fine for v1 launch
voices     (id, owner_id, name, storage_key, duration_sec, sample_rate, is_builtin, created_at)
jobs       (id, owner_id, status, prompt, length, language, mode, speed,
            voice_id, dialogue_voice_id,
            story_text, audio_key, audio_duration_sec,
            error_code, error_message,
            queued_at, writing_at, written_at, synth_at, done_at,
            llm_model, llm_tokens, retry_count, idempotency_key)
```

`queued_at → done_at` timestamps are the source of the performance numbers in the README,
so the v1-vs-v2 comparison is measured rather than claimed.

Redis holds only ephemeral state: the Celery queues, the `job:{id}` pub/sub channel, a
rate-limit counter, and the idempotency map. Losing Redis loses in-flight work (which
retries); it never loses history.

## 8. Provider abstraction

Both external dependencies sit behind a narrow interface so deployment target and local
development can differ without a code fork.

```python
class LLMProvider(Protocol):
    async def stream(self, system: str, user: str, *, max_tokens: int) -> AsyncIterator[str]: ...

class TTSProvider(Protocol):
    async def synthesize(self, text: str, voice: VoiceRef, *, language: str,
                         speed: float) -> AsyncIterator[AudioChunk]: ...
```

| Slot | Local dev | Cheap deploy | Notes |
|---|---|---|---|
| LLM | Ollama (existing models) | Groq / Gemini Flash / OpenRouter | Free or near-free tiers; 50–100× faster than a local 7B on a 3050 |
| TTS | self-hosted XTTS v2 on the dev GPU | serverless GPU XTTS, or a hosted cloning API | same gRPC contract either side |

Keeping the Ollama adapter means the project still runs fully offline with no API key — a
genuine feature, not a leftover.

## 9. Fixes carried by the architecture

| v1 defect | How v2 removes it |
|---|---|
| 1 Global `chat_history` | Workers are stateless. Each task builds its prompt from its own row; there is no cross-request conversation at all. |
| 2 10-min blocking RPC | Queue + job id + WS. Nothing blocks. |
| 3 Client-supplied speaker path | `voiceId` → server-side lookup → storage key. Paths never cross the wire. |
| 4 Fixed-path upload race | Uploads go to `voices/{uuid}.wav` in object storage. |
| 5 Prompt-derived filenames | `audio/{jobId}.mp3`. Collision impossible. |
| 6 No auth/limits | Gateway does auth, per-key rate limiting, and a concurrent-job quota before anything is enqueued. |
| 7 Fake concurrency | The queue *is* the concurrency control. Depth is observable; workers scale on it. |
| 8 Unvalidated `language` | Closed enum. Translation path deleted — the LLM writes in the target language directly. |
| 9 Prompt injection | User text is delimited and passed as a separate `user` message; system instructions are never concatenated with it. Output is length- and content-checked. |
| 10 Two processes, one container | One process per container, each with its own healthcheck and restart policy. |
| 11/12 requirements & Docker | `pyproject.toml` + `uv` lockfile per service; multi-stage builds with dependency layers cached ahead of source. |
| 13 Dead `emotion` param | Emotion is dropped from the TTS call and expressed where it actually works — in the LLM prompt, and in per-segment reference-voice selection. |
| 14 100 MB protobuf blobs | Audio goes to object storage; the API returns a presigned URL. gRPC streams PCM in bounded chunks. |
| 15/16 Streamlit state & checks | React client with server-durable state; duration validated by decoding the audio header, not by byte count. |
| 17 `[PARA_LEVEL]` sentinel | Typed enum field. |
| 18 Truncated long stories | `max_tokens` derived from the requested length, plus a completeness check with one bounded continuation pass. |
| 19 Leaked exception text | Error taxonomy: safe code + message to the client, full trace to structured logs. |
| 20 No tests/CI/observability | pytest + Vitest + Playwright in CI; structured JSON logs, Prometheus metrics, OpenTelemetry trace ids propagated through the queue. |

## 10. Frontend architecture

React 19 + TypeScript (strict) + Vite. Hand-written, no scaffolding generator.

```
web/src/
  api/          typed fetch client, generated types from the OpenAPI schema
  hooks/        useJob, useJobEvents (WS + polling fallback), useVoices
  features/
    compose/    prompt form, length/voice/language/mode controls
    job/        live story stream, segment progress, audio player
    voices/     catalogue, upload with client-side duration check, recorder
    history/    past jobs, deep-linkable at /jobs/:id
  components/   design-system primitives
  lib/          formatting, waveform rendering, WS reconnect with backoff
```

Deliberate choices:

- **Server state via TanStack Query; no global store.** Job state lives on the server.
  The client caches and invalidates it. The WS handler writes into the query cache so
  polling and streaming converge on one source of truth.
- **`useJobEvents` degrades cleanly.** WebSocket by default; on failure or three
  reconnect attempts it silently falls back to polling `GET /v1/jobs/{id}`. The UI cannot
  tell the difference — that is the test.
- **Every job is a route.** `/jobs/:id` is shareable and survives refresh. This is the
  visible payoff of the architecture change and should be obvious in ten seconds of demo.
- **Story text streams in as tokens arrive**, so the user is reading within ~2 seconds
  while the GPU work is still queued. Perceived latency drops from 10 minutes to 2 seconds
  even when total wall time is unchanged.
- **Accessibility and mobile are requirements**, not polish: keyboard-reachable controls,
  visible focus, labelled form fields, `prefers-reduced-motion` respected, real dark mode.

## 11. Repository layout

```
.
├── docs/                     this folder
├── proto/
│   └── tts/v1/tts.proto      internal TTS contract (streaming)
├── services/
│   ├── gateway/              FastAPI — REST + WS
│   ├── story-worker/         Celery — LLM stage
│   ├── tts-worker/           Celery — segmentation, stitching, upload
│   └── tts-engine/           gRPC + GPU — XTTS
├── packages/
│   └── shared/               Pydantic schemas, job enums, prompt library, storage client
├── web/                      React + TypeScript
├── infra/
│   ├── docker-compose.yml        full local stack
│   ├── docker-compose.gpu.yml    overlay: local GPU tts-engine
│   └── deploy/                   Fly/Railway/Hetzner manifests
├── tests/
│   ├── unit/  integration/  e2e/
│   └── fixtures/             ported from TestCases.json
└── .github/workflows/
```

Python services share `packages/shared` via a workspace, so job status enums and request
schemas are defined once and imported everywhere — the gateway and the workers cannot
drift.

## 12. Observability

- **Structured JSON logs** with `job_id` and `trace_id` on every line, from the gateway
  through the queue into both workers.
- **Metrics**: queue depth per stage, stage duration histograms, GPU seconds per job,
  failure rate by `errorCode`, tokens per second, cost per job.
- **Tracing**: OpenTelemetry context propagated in the Celery task headers, so a single
  trace spans `POST /v1/jobs` → story-worker → tts-worker → tts-engine.
- **A `/stats` page in the UI** rendering stage timings for the caller's own jobs. This is
  what turns "I redesigned it around a queue" into a number someone can look at.

## 13. Explicitly out of scope for v2.0

Named here so they don't creep in mid-build: user accounts with passwords (anonymous
session ids are enough at launch), payment, multi-character casting beyond
narrator + dialogue, background music, subtitle/SRT export, and a mobile app. The
subtitle feature in particular was already broken in v1; it is a v2.1 candidate, sized
once `tts-engine` returns word timings.
