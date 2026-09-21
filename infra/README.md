# infra

Deployment and local-orchestration assets.

| Path | Phase | What it is |
|---|---|---|
| `docker-compose.yml` | 5 | Full local stack: gateway, story-worker, tts-worker, tts-engine, redis, postgres, minio |
| `docker-compose.gpu.yml` | 4 | Overlay that runs `tts-engine` against a local CUDA device |
| `deploy/` | 8 | Fly.io / Hetzner manifests and the deploy runbook |

Nothing here exists yet beyond this file — the compose stack is built in Phase 4–5, once
there are services worth composing. See `docs/03-implementation-plan.md`.
