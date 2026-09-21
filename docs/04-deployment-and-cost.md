# 04 — Deployment and Cost

Constraint: this is a portfolio/demo project that must be publicly reachable and must not
cost meaningful money at idle. The whole plan is built around one number — **idle cost
must approach zero** — because a demo is idle ~99% of the time.

> All prices below are ballpark figures from public pricing pages and **must be
> re-checked before Phase 8**. They are here to shape the decision, not to be quoted.

---

## 1. Where the money actually goes

v1's cost profile is a GPU box running 24/7 so that an occasional visitor can wait ten
minutes. That is the worst possible shape: maximum fixed cost, minimum utilisation.

The only genuinely expensive component in v2 is **GPU time for TTS**. Everything else —
API, workers, queue, database, storage, frontend — fits inside free tiers or a single
small VM. So the deployment strategy is:

1. Put everything cheap on free tiers or one small always-on box.
2. Make the GPU **scale to zero**, so it costs nothing between demos.
3. Make the async architecture absorb the cold start, which it does for free — a 60 s
   cold start is queue time on a job that was never going to be synchronous anyway.

The architecture and the cost strategy are the same decision. That is the point.

## 2. Component-by-component

| Component | Choice | Free tier / cost | Why |
|---|---|---|---|
| Frontend | **Cloudflare Pages** or Vercel Hobby | Free | Static SPA, global CDN, preview deploys per PR |
| Gateway + workers | **Fly.io** (3 small machines) or **Hetzner CX22** | Fly: ~$0–5/mo at this size · Hetzner: ~€4–5/mo | Fly scales machines to zero and has good WS support. Hetzner is one box with compose — cheapest and simplest, but always-on and manual. |
| Queue / pub-sub | **Upstash Redis** | Free tier covers this workload comfortably | Serverless, per-request pricing, no idle cost. Redis Cloud free tier is an alternative. |
| Database | **Neon Postgres** | Free tier, scales to zero | Job history is tiny. Supabase is the alternative if auth is wanted later. |
| Object storage | **Cloudflare R2** | Free tier ~10 GB storage · **zero egress** | Egress is what kills S3 for an audio app. R2 charging nothing for egress is the deciding factor. |
| LLM | **Groq** or **Gemini Flash**, via OpenAI-compatible adapter | Generous free tiers; paid is fractions of a cent per story | A ~700-word story is roughly 1–2k output tokens. At Flash/Groq rates that is well under $0.001 per story. |
| **TTS (GPU)** | see §4 — the real decision | see §4 | |
| CI | GitHub Actions | Free for public repos | |
| Monitoring | Grafana Cloud free / Better Stack free | Free | |
| Domain | any registrar | ~$10/yr | Optional |

## 3. Two deployment shapes

### Shape A — "managed, scale-to-zero" (recommended)

```
Cloudflare Pages   →  web
Fly.io             →  gateway (1 machine, auto-stop) 
                      story-worker (1 machine, auto-stop)
                      tts-worker  (1 machine, auto-stop)
Upstash            →  Redis
Neon               →  Postgres
Cloudflare R2      →  audio + voices
Serverless GPU     →  tts-engine  (scale to zero)
Groq / Gemini      →  LLM
```

Idle cost ≈ $0. Cost is proportional to demo traffic. Best story to tell in an interview,
and the configuration that actually matches the architecture.

### Shape B — "one box" (cheapest fixed, simplest)

```
Hetzner CX22 (2 vCPU, 4 GB, ~€4.5/mo)
  docker compose: gateway, story-worker, tts-worker, redis, postgres, caddy(TLS)
Cloudflare R2   → storage
Groq / Gemini   → LLM
Serverless GPU  → tts-engine   (still remote — a CX22 has no GPU)
```

Flat ~€5/mo regardless of traffic. Fewer moving parts, one `docker compose up`, and it
proves the compose file works. Downside: no scale-to-zero, manual TLS/deploy, and a
single point of failure.

**Recommendation:** build for Shape B first (it is what `infra/docker-compose.yml`
already gives you at the end of Phase 5), then move to Shape A in Phase 8. The service
boundaries make that move a config change, not a rewrite.

## 4. The TTS decision — the only hard one

XTTS v2 needs a GPU to be usable. Four options, in preference order:

### Option 1 — Serverless GPU, scale to zero (recommended)

RunPod Serverless, Modal, or Beam. Deploy `tts-engine` as a container; the platform keeps
it at zero replicas until a request arrives.

- Cost: billed per GPU-second. A T4/A10-class GPU is roughly **$0.0002–0.0006 per second**.
  A 5-minute story is maybe 60–120 s of GPU time → **~$0.02–0.07 per story**.
- Idle cost: **zero**.
- Cold start: 30–90 s to pull the model into VRAM. Mitigated by flash-boot / cached
  volumes, and — crucially — **hidden by the async design**. The user is already reading
  streamed story text while the GPU wakes up.
- Keeps XTTS, keeps voice cloning, keeps the gRPC contract, keeps the code you wrote.

This is the option that best matches the architecture. Pick it unless licensing blocks it.

### Option 2 — Hosted cloning TTS API

Cartesia, PlayHT, ElevenLabs, fal.ai (which hosts XTTS-class models per-second).

- No GPU to operate, no cold start, better latency than Option 1.
- Cost is per character or per second of audio and varies widely by vendor — ElevenLabs
  is the expensive end, Cartesia and fal.ai substantially cheaper. Check current rates.
- Requires an adapter behind `TTSProvider`. The interface in doc 02 §8 exists precisely so
  this is a swap, not a rewrite.
- Trade-off: it removes the self-hosted GPU service, which is part of what makes the
  project interesting. **Keep `tts-engine` in the repo and runnable locally even if
  production points at a hosted API** — the local GPU path is the engineering story.

### Option 3 — CPU TTS fallback for the public demo

Run a small CPU-friendly model (Kokoro-82M class, or Piper) on the always-on box as a
`TTSProvider` implementation used when no GPU budget is available.

- Cost: zero marginal.
- Loses voice cloning — this is the feature the project is built on, so this is a
  **degraded demo mode**, not the default.
- Worth having anyway as a smoke-test provider in CI, where a GPU is unavailable.

### Option 4 — Self-host XTTS on a rented always-on GPU

A dedicated GPU VM is roughly $70–250/mo depending on card and vendor. Rejected: it
reproduces v1's cost shape exactly.

### Recommendation

Implement `TTSProvider` with **three** adapters: local XTTS over gRPC (dev + the real
engineering artifact), serverless-GPU XTTS (production default), and a CPU model (CI and
degraded demo). Choose per environment by env var. The cost strategy then becomes a
deployment concern rather than an architectural one.

## 5. Cost model

Fixed monthly:

| Shape | Fixed |
|---|---|
| A (managed, scale-to-zero) | ~$0 |
| B (Hetzner one box) | ~€5 |
| plus domain | ~$1/mo amortised |

Marginal, per story (~700 words, ~5 min audio):

| Line item | Estimate |
|---|---|
| LLM (Gemini Flash / Groq) | < $0.001 |
| TTS, serverless GPU | ~$0.02–0.07 |
| Storage + egress (R2) | ~$0 (zero egress) |
| Postgres / Redis | ~$0 (free tier) |
| **Total** | **~$0.02–0.07** |

So 100 demo stories a month ≈ **$2–7**, on top of ~$0–5 fixed. That is the entire budget.

## 6. Guardrails (non-optional)

An open, unauthenticated endpoint that spends GPU money per request is a bill waiting to
happen. Enforced in code, not just in a billing dashboard:

- **Per-session quota**: max concurrent jobs, max jobs per day.
- **Global daily cap**: a counter in Redis. When the day's job count or estimated spend
  exceeds the ceiling, `POST /v1/jobs` returns `429` with a clear message, and the app
  offers a pre-rendered sample instead of a live generation.
- **Rate limit** on job creation and voice upload, by session and by IP.
- **Input caps**: prompt length, upload size, reference-audio duration.
- **Storage lifecycle**: audio objects expire after N days; orphaned uploads reaped.
- **Provider-side caps** as a second line of defence: spend limits set on the LLM and GPU
  accounts.
- **Alerts** on queue depth, error rate, and daily spend.

This is also the honest answer to "how would you stop this from being abused?", which is
the question that follows a public demo link.

## 7. Environments

| Env | Web | API/workers | GPU | Data |
|---|---|---|---|---|
| local | Vite dev server | `docker compose up` | local CUDA via `docker-compose.gpu.yml`, or CPU provider | postgres + redis + MinIO in compose |
| staging | Pages preview per PR | Fly staging app | shared serverless GPU endpoint | Neon branch DB |
| production | Pages production | Fly production app | serverless GPU endpoint | Neon main |

Neon's database branching gives a throwaway DB per preview deploy at no cost, which is
worth wiring up in Phase 8.

## 8. Deployment pipeline

```
push to v2-rebuild
  └─ CI: ruff · mypy · pytest · eslint · tsc · vitest
       └─ build images (gateway, story-worker, tts-worker, tts-engine) → GHCR, cached
            └─ deploy preview: Pages preview + Fly staging + Neon branch
                 └─ Playwright e2e against the preview
                      └─ manual promote → production
```

Rollback is redeploying the previous image tag — one command, and it must be tested once
in Phase 8 rather than discovered during an incident.

## 9. What to verify before Phase 8

- [ ] Current pricing for the chosen serverless GPU vendor and LLM provider.
- [ ] Measured cold-start time for `tts-engine` with the model on a cached volume.
- [ ] Coqui XTTS v2 licence terms for a publicly reachable deployment (CPML is
      non-commercial; confirm whether a free public portfolio demo qualifies, and have
      Option 2 ready as the fallback).
- [ ] Upstash and Neon free-tier limits against the expected request shape.
- [ ] R2 free-tier storage ceiling against the audio retention window.
