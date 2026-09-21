# 03 — Implementation Plan

Ten phases. Each one ends at a state that runs, is committed, and is demonstrable — no
phase leaves the repo broken. Phases 0–5 produce a working end-to-end system; 6–9 are
hardening, deployment and polish.

Branch: `v2-rebuild`. One commit series per phase, merged to `main` only when Phase 9
closes.

---

## Phase 0 — Groundwork

**Goal:** a clean skeleton that builds and tests green with zero features in it.

- [ ] Create the `services/ packages/ web/ infra/ tests/` layout from doc 02 §11.
- [ ] Move v1 sources to `legacy/` (kept for reference and for the benchmark comparison,
      deleted in Phase 9).
- [ ] `pyproject.toml` + `uv` lock per Python service; `packages/shared` as a workspace
      dependency. Replaces the UTF-16 unpinned `requirements.txt`.
- [ ] Tooling: `ruff` + `mypy --strict` for Python, `eslint` + `tsc --noEmit` for the web,
      `pre-commit` hooks, `.editorconfig`.
- [ ] `.github/workflows/ci.yml`: lint, typecheck, unit tests on every push.
- [ ] `.env.example` documenting every variable. No secret ever lands in the repo.
- [ ] Fix `.gitignore` / `.dockerignore` for the new layout; keep `voices/` tracked.

**Done when:** `uv run pytest` and `npm run build` both pass on an empty feature set, and
CI is green.

---

## Phase 1 — Shared domain

**Goal:** one definition of the domain, imported by every service.

- [ ] `packages/shared/schemas.py` — Pydantic models: `CreateJobRequest`, `JobRecord`,
      `JobEvent` (tagged union: `token` / `progress` / `story_done` / `done` / `failed`),
      `VoiceRecord`.
- [ ] `JobStatus` and `StoryLength` / `VoiceMode` / `Language` enums. `StoryLength`
      replaces the `[PARA_LEVEL:...]` sentinel outright.
- [ ] `packages/shared/prompts.py` — port the six v1 prompts into **one** parameterised
      template taking `(length, mode, language)`. The six near-identical blocks in
      `server_ms.py` collapse to one template plus a table of word targets.
- [ ] Error taxonomy: `ErrorCode` enum with a user-safe message per code.
- [ ] `packages/shared/storage.py` — S3-compatible client wrapper (`put`, `presign_get`,
      `delete`), backed by MinIO locally and R2 in production.
- [ ] Unit tests for prompt assembly and schema validation bounds.

**Done when:** prompt assembly is covered by tests and no prompt string exists outside
`packages/shared`.

---

## Phase 2 — Gateway (API only, no workers yet)

**Goal:** the full public API, returning job ids, backed by a stub that completes
instantly.

- [ ] FastAPI app with lifespan-managed Postgres and Redis pools.
- [ ] Alembic migrations for `users`, `voices`, `jobs` (doc 02 §7).
- [ ] `POST /v1/jobs` → validate → insert `queued` → enqueue → `202 {jobId}`.
      Honour `Idempotency-Key`.
- [ ] `GET /v1/jobs/{id}`, `GET /v1/jobs`, `DELETE /v1/jobs/{id}`.
- [ ] `GET/POST/DELETE /v1/voices`. Upload validates **decoded duration ≥ 6 s** (via
      `soundfile`/`mutagen`, not a byte count), sample rate, channel count, and MIME —
      fixing v1 defect 16. Store to object storage under `voices/{uuid}.wav`.
- [ ] Seed the 16 built-in voices from `voices/speakers.json` as `is_builtin` rows.
- [ ] `WS /v1/jobs/{id}/events` — subscribe to Redis `job:{id}`, replay current state on
      connect so a late subscriber isn't stuck, heartbeat ping, clean close on terminal
      status.
- [ ] Anonymous session identity (signed cookie) + per-session rate limit and a
      max-concurrent-jobs quota, enforced before enqueue.
- [ ] `/healthz`, `/readyz`, `/metrics`; structured JSON logging with `trace_id`.
- [ ] A `noop` task that marks jobs `done` with canned text, so the API is testable
      before any worker exists.

**Done when:** `curl POST /v1/jobs` returns in under 200 ms and `wscat` on the events
endpoint shows the job reach `done`.

---

## Phase 3 — story-worker (LLM stage)

**Goal:** real stories, streaming, with the local Ollama models still working.

- [ ] Celery app with Redis broker; separate queues `story` and `tts`; task routing;
      `acks_late` + `reject_on_worker_lost` so a killed worker's job is redelivered.
- [ ] `LLMProvider` protocol + adapters: `OllamaProvider` (keeps v1 working offline) and
      `OpenAICompatProvider` (Groq / OpenRouter / any OpenAI-shaped endpoint).
      Selected by `LLM_PROVIDER` env var.
- [ ] Decouple model choice from story length — the v1 coupling is removed. Length sets
      `max_tokens` only; the model is a config value. Fixes v1 defect 18 alongside a
      completeness check (does the text end on terminal punctuation?) with at most one
      bounded continuation pass.
- [ ] Prompt assembly with strict role separation: system instructions as `system`,
      user storyline as `user`, never concatenated — fixing v1 defect 9.
- [ ] Token streaming → `redis.publish("job:{id}", {"type":"token", ...})`, batched at
      ~50 ms so one WS frame per token doesn't flood the socket.
- [ ] Status transitions `queued → writing → written`, conditional on the prior status.
- [ ] Persist `story_text`, `llm_model`, `llm_tokens`, and stage timestamps; chain the TTS
      task.
- [ ] Retries with exponential backoff; classify provider errors into `ErrorCode`.
- [ ] Tests: a fake provider that yields scripted tokens; assert the event sequence and
      every state transition, including the failure and retry paths.

**Done when:** posting a job streams story text into `wscat` within ~2 s and the job lands
on `written`.

---

## Phase 4 — tts-engine (gRPC + GPU)

**Goal:** XTTS behind a clean streaming contract, one process, one job.

- [ ] `proto/tts/v1/tts.proto`:
      ```proto
      service TtsEngine {
        rpc Synthesize (SynthesizeRequest) returns (stream AudioChunk);
        rpc EmbedSpeaker (EmbedSpeakerRequest) returns (SpeakerEmbedding);
        rpc Health (HealthRequest) returns (HealthResponse);
      }
      ```
      Server-streaming PCM in bounded chunks — no 100 MB messages, no
      `max_receive_message_length` override. Fixes v1 defect 14.
- [ ] Load XTTS v2 once at startup; `Health` reports model-loaded and warm.
- [ ] **Speaker-embedding cache.** v1 recomputed the conditioning latents from the
      reference WAV on *every* `tts_to_file` call. Compute once per voice, cache by
      `voiceId` (memory + object storage). This is a large real speedup on multi-segment
      dialogue jobs and is worth measuring before/after.
- [ ] Single-inference-at-a-time semantics made explicit and bounded: a worker semaphore
      plus a `RESOURCE_EXHAUSTED` response when saturated, rather than v1's unbounded
      silent queue behind a module-global lock.
- [ ] Drop the `emotion=` argument — it is a no-op (v1 defect 13). Drop the MarianMT
      translation path entirely; the LLM writes in the target language.
- [ ] CUDA-correct Dockerfile: `nvidia/cuda:*-runtime` base, multi-stage, dependency
      layers before source, non-root user.
- [ ] `infra/docker-compose.gpu.yml` overlay for local GPU runs.

**Done when:** a gRPC client streams a synthesized WAV of a known sentence, and the second
call with the same voice is measurably faster than the first (embedding cache proven).

---

## Phase 5 — tts-worker + end-to-end

**Goal:** full pipeline, audio in the browser.

- [ ] Port from v1, now with tests: `split_into_narration_and_dialogues`, `trim_silence`,
      `detect_leading_silence`, fade-in/out, the 300 ms inter-segment pad. This logic was
      correct in v1 and transfers directly.
- [ ] Per-segment gRPC synthesis with a progress publish after each segment:
      `{"type":"progress","done":i,"total":n}`.
- [ ] Cancellation check between segments — a `DELETE`d job stops consuming GPU.
- [ ] Join, normalise loudness, encode **mp3 (delivery) + wav (download)**, upload to
      object storage at `audio/{jobId}.{ext}`.
- [ ] Status `synthesizing → done`, persist `audio_key` and duration, publish
      `{"type":"done","audioUrl": <presigned>}`.
- [ ] Retry semantics: because the story is already persisted at `written`, a TTS retry
      never re-runs the LLM.
- [ ] `infra/docker-compose.yml` bringing up gateway + both workers + tts-engine + redis +
      postgres + minio with one command.
- [ ] Integration test: post a job, drive it to `done`, assert a decodable audio file of
      plausible duration comes back.

**Done when:** `docker compose up` → post a job → play the audio. First milestone worth
recording as a demo.

---

## Phase 6 — Frontend

**Goal:** the React + TypeScript client. Written by hand, reviewed like production code.

- [ ] Vite + React 19 + TypeScript strict. Routes: `/`, `/jobs/:id`, `/voices`,
      `/history`, `/stats`.
- [ ] Generate TS types from the gateway's OpenAPI schema so client and server cannot
      drift; typed fetch wrapper with an error envelope.
- [ ] `useJobEvents` — WebSocket with exponential-backoff reconnect, **falling back to
      polling** `GET /v1/jobs/{id}` after repeated failure. Both paths write into the same
      TanStack Query cache, so the rest of the UI is transport-agnostic.
- [ ] Compose view: prompt, length, language, voice picker with inline preview playback,
      narration/dialogue mode, speed. Optimistic navigation to `/jobs/:id` on submit.
- [ ] Job view: story text typing in live as tokens arrive, a real segment progress bar
      (`done/total`, not a fake spinner), then a custom audio player with waveform, seek,
      speed, and download.
- [ ] Voices: upload with **client-side duration decode** before the request (instant
      feedback), plus `MediaRecorder` in-browser recording with a live level meter.
- [ ] History: paginated past jobs, each deep-linkable.
- [ ] Stats: stage-timing charts from the caller's job timestamps — the visible proof of
      the architecture change.
- [ ] Design system: type scale, spacing scale, tokens, dark mode, reduced-motion,
      keyboard navigation, visible focus rings, labelled inputs, mobile layouts.
- [ ] Vitest unit tests for hooks and reducers; Playwright e2e for the happy path,
      the refresh-mid-job path, and the WS-fails-fallback-to-polling path.

**Done when:** submitting a job, refreshing the tab mid-generation, and still landing on
the finished audio works — the single behaviour v1 could not do.

---

## Phase 7 — Hardening

- [ ] Security pass: request-size caps, upload MIME/magic-byte sniffing, rate limits on
      upload and job creation, presigned-URL expiry, CORS allowlist, security headers,
      no stack traces in responses (v1 defect 19).
- [ ] Prompt-injection defence: delimited user input, output length and content checks,
      a regression fixture of adversarial prompts.
- [ ] Storage lifecycle: object-expiry rules, a reaper for orphaned uploads, per-session
      storage quota. v1's `output/` grew forever.
- [ ] Graceful shutdown: workers finish or requeue the current task on `SIGTERM`; the
      gateway drains WS connections.
- [ ] Load test with `k6`: 20 concurrent jobs. Confirm queue depth grows and API latency
      does **not**. That graph is the whole argument for the redesign.
- [ ] Benchmark harness: replay `TestCases.json` against `legacy/` and v2, produce the
      new performance table and chart to replace `performance_graph.png`.
- [ ] OpenTelemetry trace propagation through Celery headers; Grafana/Prometheus wired in
      compose.

---

## Phase 8 — Deployment

Detail and costs in `04-deployment-and-cost.md`. Execution order:

- [ ] Provision: Neon Postgres, Upstash Redis, Cloudflare R2, LLM provider key.
- [ ] Build and publish images via GitHub Actions to GHCR, with build cache.
- [ ] Deploy gateway + both workers (Fly.io or a single Hetzner box with compose).
- [ ] Deploy `tts-engine` to a **scale-to-zero serverless GPU**, or point the `TTSProvider`
      at a hosted cloning API — same interface either way (doc 04 §4).
- [ ] Deploy `web` to Vercel or Cloudflare Pages.
- [ ] Custom domain, TLS, staging environment, one-command rollback.
- [ ] Uptime check + alert on queue depth and error rate.
- [ ] Cost guardrail: a hard monthly ceiling on GPU/LLM spend, enforced in code by the
      quota layer, not just by a billing alert.

---

## Phase 9 — Documentation and close-out

- [ ] Rewrite `README.md`: architecture diagram, the before/after latency table with real
      measurements, quickstart, a 30-second demo GIF.
- [ ] `docs/05-api.md` from the OpenAPI schema; `docs/06-operations.md` (runbook: common
      failures, how to drain a queue, how to replay a failed job).
- [ ] An ADR per significant decision — queue over streaming-RPC, Celery over BullMQ,
      keeping gRPC internal only, hosted LLM over local Ollama.
- [ ] Delete `legacy/`, `changes.txt`, `xtts_model/` (dead weight, doc 01 §6), and the
      stale `performance_graph.png`.
- [ ] Merge `v2-rebuild` → `main`.

---

## Sequencing and dependencies

```
0 ──► 1 ──► 2 ──► 3 ──┐
            │         ├──► 5 ──► 6 ──► 7 ──► 8 ──► 9
            └──► 4 ───┘
```

Phases 3 and 4 are independent once Phase 2 lands and can be worked in either order.
Phase 6 can start against the Phase 2 stub API before the workers are real.

## Definition of done, per phase

Every phase must satisfy all four before the next begins:

1. CI green — lint, typecheck, tests.
2. `docker compose up` still brings the stack to a healthy state.
3. New behaviour is covered by a test that would fail without it.
4. A one-paragraph note in the phase's commit message stating what changed and why.

## Risks

| Risk | Mitigation |
|---|---|
| Serverless GPU cold starts (30–90 s for XTTS) | Job is already async — a cold start is queue time, not user-facing latency. Story text is already streaming during it. Optionally keep one warm instance during demos. |
| Hosted LLM cost or rate limits | Provider adapter + a hard per-session quota; Ollama fallback always present. |
| XTTS licensing (Coqui CPML) for a public deploy | Confirm terms before Phase 8. If it blocks, the `TTSProvider` interface lets a permissively licensed or commercial engine drop in without touching the pipeline. |
| Scope creep into subtitles / music / casting | Explicitly deferred in doc 02 §13. |
| Rebuild stalls half-finished | Every phase ends demonstrable; the repo is never left broken. Phase 5 alone is already a complete, shippable product. |
