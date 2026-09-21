# tts-engine: XTTS v2 behind a gRPC contract.
#
# Three things v1's Dockerfile got wrong, all fixed here:
#
#   1. It used `python:3.10-slim` while pinning `torch==2.5.1+cu121`. There is no CUDA
#      runtime in that image, so the GPU path could not work in the container at all.
#   2. `COPY . .` came *before* `pip install`, so editing one line of Python re-downloaded
#      and rebuilt multi-gigabyte torch wheels.
#   3. It shipped the whole repository into the image, including 2.3 MB of base64 test
#      fixtures and every reference voice.

# --- Builder -----------------------------------------------------------------------------

FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_INSTALL_DIR=/opt/python

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.9 /uv /usr/local/bin/uv

WORKDIR /build

# Dependency layer first, and only the files that describe dependencies. This layer is
# rebuilt when the lockfile changes and not when application code does -- the inversion
# that made every v1 image rebuild torch from scratch.
COPY pyproject.toml uv.lock ./
COPY packages/shared/pyproject.toml packages/shared/
COPY services/gateway/pyproject.toml services/gateway/
COPY services/story_worker/pyproject.toml services/story_worker/
COPY services/tts_worker/pyproject.toml services/tts_worker/
COPY services/tts_engine/pyproject.toml services/tts_engine/

# `--no-install-project` installs only the dependency tree; the project itself lands in
# the next layer so a code change cannot invalidate this one.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project \
        --package story2audio-tts-engine --extra gpu

# Now the source, plus the contract the stubs are generated from.
COPY proto/ proto/
COPY infra/scripts/gen_proto.py infra/scripts/
COPY services/tts_engine/src/ services/tts_engine/src/

# Stubs are generated at build time. The `.proto` is the source of truth, so a stale
# committed copy cannot drift from it.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv run --no-dev --with grpcio-tools python infra/scripts/gen_proto.py \
    && uv sync --frozen --no-dev --package story2audio-tts-engine --extra gpu

# --- Runtime ------------------------------------------------------------------------------

FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04 AS runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/build/.venv/bin:/opt/python/bin:${PATH}" \
    # Coqui refuses to download XTTS without this. Setting it is an acceptance of the
    # model's licence -- see docs/04 §9 before deploying this publicly.
    COQUI_TOS_AGREED=1 \
    # Model weights are large and must not be baked into the image: mount a cached
    # volume here so a scale-to-zero replica warms from disk rather than the network.
    TTS_HOME=/models \
    HF_HOME=/models/huggingface

RUN apt-get update && apt-get install -y --no-install-recommends \
        libsndfile1 \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Non-root. The engine runs a large model over text supplied by strangers; it has no
# reason to be able to write anywhere but its model cache.
RUN useradd --create-home --uid 10001 engine \
    && mkdir -p /models \
    && chown -R engine:engine /models

COPY --from=builder --chown=engine:engine /opt/python /opt/python
COPY --from=builder --chown=engine:engine /build /build

USER engine
WORKDIR /build

EXPOSE 50051

# Liveness only. Readiness is the standard gRPC health protocol, which reports
# NOT_SERVING until the weights are resident -- an orchestrator should route traffic on
# that, not on the process being alive.
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import socket,sys; s=socket.create_connection(('127.0.0.1',50051),3); s.close()" \
        || exit 1

CMD ["python", "-m", "tts_engine"]
