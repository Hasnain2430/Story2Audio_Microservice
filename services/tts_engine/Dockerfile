# tts-engine: XTTS v2 behind a gRPC contract.
#
# Two runtime targets from one build:
#
#   `runtime` (default)  slim Python, no torch. Serves the stub backend, which renders a
#                        tone rather than speech — enough to exercise the service, the
#                        streaming protocol and the whole worker pipeline on a machine
#                        with no GPU. A CUDA base would be a 2 GB pull for nothing here.
#   `gpu`                CUDA base plus the `gpu` extra. Real synthesis.
#
# Three things v1's Dockerfile got wrong, all fixed:
#
#   1. It used `python:3.10-slim` while pinning `torch==2.5.1+cu121`. There is no CUDA
#      runtime in that image, so the GPU path could not work in the container at all.
#   2. `COPY . .` came *before* `pip install`, so editing one line of Python re-downloaded
#      and rebuilt multi-gigabyte torch wheels.
#   3. It shipped the whole repository into the image, including 2.3 MB of base64 test
#      fixtures and every reference voice.

ARG PYTHON_VERSION=3.11
ARG CUDA_IMAGE=nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

# --- Dependency layers ---------------------------------------------------------------------

FROM python:${PYTHON_VERSION}-slim-bookworm AS builder-base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

COPY --from=ghcr.io/astral-sh/uv:0.9 /uv /usr/local/bin/uv

WORKDIR /app

# Only the files that describe dependencies, so editing source cannot invalidate this.
COPY pyproject.toml uv.lock ./
COPY packages/shared/pyproject.toml packages/shared/
COPY services/gateway/pyproject.toml services/gateway/
COPY services/story_worker/pyproject.toml services/story_worker/
COPY services/tts_worker/pyproject.toml services/tts_worker/
COPY services/tts_engine/pyproject.toml services/tts_engine/


FROM builder-base AS builder-cpu
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project --package story2audio-tts-engine


FROM builder-base AS builder-gpu
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project --package story2audio-tts-engine --extra gpu


# --- Source layer --------------------------------------------------------------------------

FROM builder-cpu AS source-cpu
COPY proto/ proto/
COPY infra/scripts/gen_proto.py infra/scripts/
COPY packages/ packages/
COPY services/ services/
RUN --mount=type=cache,target=/root/.cache/uv \
    uv run --no-dev --with grpcio-tools python infra/scripts/gen_proto.py \
    && uv sync --frozen --no-dev --package story2audio-tts-engine


FROM builder-gpu AS source-gpu
COPY proto/ proto/
COPY infra/scripts/gen_proto.py infra/scripts/
COPY packages/ packages/
COPY services/ services/
RUN --mount=type=cache,target=/root/.cache/uv \
    uv run --no-dev --with grpcio-tools python infra/scripts/gen_proto.py \
    && uv sync --frozen --no-dev --package story2audio-tts-engine --extra gpu


# --- Runtime: CPU / stub (default) -------------------------------------------------------------

FROM python:${PYTHON_VERSION}-slim-bookworm AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:${PATH}" \
    TTS_BACKEND=stub \
    TTS_DEVICE=cpu

RUN apt-get update && apt-get install -y --no-install-recommends \
        libsndfile1 ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN useradd --create-home --uid 10001 engine

COPY --from=source-cpu --chown=engine:engine /app /app

USER engine
WORKDIR /app
EXPOSE 50051

# Liveness only. Readiness is the standard gRPC health protocol, which reports
# NOT_SERVING until the model is resident — an orchestrator should route on that.
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import socket; socket.create_connection(('127.0.0.1',50051),3).close()"

CMD ["python", "-m", "tts_engine"]


# --- Runtime: GPU / XTTS -------------------------------------------------------------------------

FROM ${CUDA_IMAGE} AS gpu

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:${PATH}" \
    TTS_BACKEND=xtts \
    TTS_DEVICE=cuda \
    # Coqui refuses to download XTTS without this. Setting it accepts the model's
    # licence — see docs/04 §9 before deploying this publicly.
    COQUI_TOS_AGREED=1 \
    # Weights are large and must not be baked into the image: mount a cached volume here
    # so a scale-to-zero replica warms from disk rather than the network.
    TTS_HOME=/models \
    HF_HOME=/models/huggingface

RUN apt-get update && apt-get install -y --no-install-recommends \
        libsndfile1 ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Non-root. The engine runs a large model over text supplied by strangers; it has no
# reason to write anywhere but its model cache.
RUN useradd --create-home --uid 10001 engine \
    && mkdir -p /models \
    && chown -R engine:engine /models

COPY --from=source-gpu --chown=engine:engine /app /app

USER engine
WORKDIR /app
EXPOSE 50051

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=5 \
    CMD python -c "import socket; socket.create_connection(('127.0.0.1',50051),3).close()"

CMD ["python", "-m", "tts_engine"]
