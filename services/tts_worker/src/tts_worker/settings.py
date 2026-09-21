"""tts-worker configuration."""

from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class TtsWorkerSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore", frozen=True
    )

    tts_engine_address: str = "localhost:50051"
    #: Per-segment deadline. Segments are a few hundred characters, so this is generous
    #: even on a cold GPU -- but bounded, so a wedged engine cannot hold a worker forever.
    tts_request_timeout_seconds: float = Field(default=600.0, gt=0)

    #: Retries for a segment the engine refused with RESOURCE_EXHAUSTED. Cheap to retry
    #: and very likely to succeed once a slot frees up.
    tts_capacity_retries: int = Field(default=5, ge=0, le=20)
    tts_capacity_backoff_seconds: float = Field(default=2.0, gt=0)

    #: Audio assembly. v1's values, kept.
    #: Pause at an ordinary boundary between segments of the same speaker — the end of a
    #: paragraph, or a sentence break the story wrote.
    segment_pause_ms: int = Field(default=300, ge=0, le=5_000)
    #: Pause where a segment boundary exists only because the text was too long for one
    #: synthesis request. A reader does not pause there, so neither should the audio. It
    #: is not zero: two independently rendered clips need a seam to butt against, and a
    #: hard cut between them is audible.
    continuation_pause_ms: int = Field(default=80, ge=0, le=5_000)
    #: Pause when the voice changes. A beat longer than a paragraph break, because the
    #: listener has to register that someone else is talking; with the same 300 ms used
    #: everywhere, a reply lands on top of the line it answers.
    speaker_change_pause_ms: int = Field(default=420, ge=0, le=5_000)
    lead_silence_ms: int = Field(default=300, ge=0, le=5_000)
    fade_ms: int = Field(default=20, ge=0, le=500)
    silence_threshold_dbfs: float = Field(default=-40.0, le=0)
    max_segment_chars: int = Field(default=300, ge=50, le=2_000)
    mp3_bitrate_kbps: int = Field(default=128, ge=32, le=320)

    #: Checked between segments, which is the granularity at which cancellation can
    #: actually free the GPU.
    cancel_check_every_segments: int = Field(default=1, ge=1)

    max_retries: int = Field(default=3, ge=0, le=10)
    retry_backoff_seconds: float = Field(default=5.0, gt=0)
    retry_backoff_max_seconds: float = Field(default=120.0, gt=0)

    job_lease_ttl_seconds: int = Field(default=1_800, ge=60)


@lru_cache(maxsize=1)
def tts_worker_settings() -> TtsWorkerSettings:
    return TtsWorkerSettings()
