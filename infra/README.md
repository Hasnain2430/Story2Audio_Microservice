# infra

Local orchestration and operational scripts.

| Path | What it is |
|---|---|
| `docker-compose.yml` | The full stack: gateway, story-worker, tts-worker, tts-engine, Postgres, Redis, MinIO, and a one-shot `migrate` job that runs Alembic and seeds the voice catalogue |
| `docker-compose.gpu.yml` | Overlay that builds `tts-engine` from its CUDA target and runs real XTTS against a local GPU |
| `Dockerfile.python` | One multi-stage image for the gateway, story-worker and tts-worker, selected by a `PACKAGE` build argument. `tts-engine` has its own Dockerfile because it needs a CUDA base |
| `scripts/run_job.py` | Submits one job and follows it to completion in a single session |
| `scripts/check_audio.py` | Tells real speech apart from a test tone, by spectrum rather than by loudness |
| `scripts/dump_openapi.py` | Writes the gateway's schema to `openapi.json`, which the frontend types are generated from |
| `scripts/gen_proto.py` | Regenerates the gRPC stubs shared by `tts-worker` and `tts-engine` |
| `sql/` | Database roles |

## Two ways to run the voice engine

The default `tts-engine` image carries no torch and synthesises a **tone**, not speech. That
is deliberate — it exercises the whole pipeline on a machine with no GPU without pulling a
multi-gigabyte CUDA base — but it means a stack started with the defaults produces a beep.
`check_audio.py` exists partly because that was once mistaken for working output.

For real narration, either apply the GPU overlay, or run the engine natively and point the
worker at it:

```sh
docker compose -f infra/docker-compose.yml stop tts-engine
TTS_ENGINE_ADDRESS=host.docker.internal:50051 \
  docker compose --env-file .env -f infra/docker-compose.yml up -d tts-worker
make tts-engine-native
```

The native route is how it runs on Windows, where a laptop GPU cannot be passed through to
WSL2 but can be reached over the host network.
