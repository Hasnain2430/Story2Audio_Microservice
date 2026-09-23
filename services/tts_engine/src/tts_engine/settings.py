"""tts-engine configuration."""

from __future__ import annotations

from enum import StrEnum
from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Backend(StrEnum):
    """Which synthesis implementation to load.

    ``stub`` produces a deterministic tone rather than speech. It exists so the server,
    the streaming protocol, the embedding cache and the concurrency limit can all be
    tested on a machine with no GPU -- and it is refused in production, because serving
    a tone as a finished story would be worse than failing.
    """

    XTTS = "xtts"
    STUB = "stub"


class TtsEngineSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore", frozen=True
    )

    environment: str = "local"
    log_level: str = "info"
    log_format: str = "console"

    tts_engine_host: str = "0.0.0.0"  # noqa: S104 - binding all interfaces is correct in a container
    tts_engine_port: int = Field(default=50051, ge=1, le=65535)

    tts_backend: Backend = Backend.STUB
    tts_model_name: str = "tts_models/multilingual/multi-dataset/xtts_v2"
    tts_device: str = "cuda"
    #: Load the model in half precision. Roughly halves resident VRAM -- measured at
    #: 0.99 GB against 1.91 GB for XTTS v2.
    #:
    #: EXPERIMENTAL, and off by default for a reason: XTTS's autoregressive stage is
    #: not numerically stable in fp16. It loads and embeds correctly, then fails
    #: during generation with `CUDA error: device-side assert triggered` -- NaNs
    #: producing an out-of-range token index. Enable only if you have measured it
    #: working on your own hardware and model version.
    #:
    #: Ignored on CPU, where fp16 is slower rather than smaller.
    tts_use_half: bool = False

    #: One GPU, one inference. Requests beyond this are refused with RESOURCE_EXHAUSTED
    #: rather than queued. v1 had a module-global mutex that silently serialised every
    #: request behind one lock with no bound and no visibility.
    tts_max_concurrent_inferences: int = Field(default=1, ge=1, le=16)

    #: gRPC handler threads. Comfortably above the inference limit so that GetInfo and
    #: health checks still answer while the model is busy.
    grpc_max_workers: int = Field(default=8, ge=2, le=64)

    #: How many speaker embeddings to keep. v1 recomputed the conditioning latents from
    #: the reference WAV on every single `tts_to_file` call, so a twelve-segment dialogue
    #: job paid for the same computation twelve times.
    speaker_cache_size: int = Field(default=64, ge=1)

    #: Bytes of PCM per streamed chunk. Small enough to stay far below gRPC's default
    #: 4 MB message limit -- no limit override anywhere, which is the point.
    audio_chunk_bytes: int = Field(default=32 * 1024, ge=1024, le=1024 * 1024)

    #: Graceful shutdown budget: let an in-flight segment finish rather than truncating
    #: audio a user is waiting on.
    shutdown_grace_seconds: float = Field(default=30.0, ge=0)

    @property
    def is_production(self) -> bool:
        return self.environment == "production"


@lru_cache(maxsize=1)
def tts_engine_settings() -> TtsEngineSettings:
    return TtsEngineSettings()
