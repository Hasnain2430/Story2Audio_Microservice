"""Configuration shared by the gateway and both workers.

Settings are grouped by concern rather than by service, and each service composes the
groups it needs. Nothing reads ``os.environ`` directly anywhere else in the codebase, so
`.env.example` stays an exhaustive description of what the stack requires to boot.
"""

from __future__ import annotations

from enum import StrEnum
from functools import lru_cache

from pydantic import Field, SecretStr, ValidationInfo, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Environment(StrEnum):
    LOCAL = "local"
    STAGING = "staging"
    PRODUCTION = "production"


class LogFormat(StrEnum):
    CONSOLE = "console"
    JSON = "json"


_BASE_CONFIG = SettingsConfigDict(
    env_file=".env",
    env_file_encoding="utf-8",
    extra="ignore",
    frozen=True,
)


class CoreSettings(BaseSettings):
    """Process-wide basics."""

    model_config = _BASE_CONFIG

    environment: Environment = Environment.LOCAL
    log_level: str = "info"
    log_format: LogFormat = LogFormat.CONSOLE

    @property
    def is_production(self) -> bool:
        return self.environment is Environment.PRODUCTION


class DatabaseSettings(BaseSettings):
    """Postgres connection strings.

    Two URLs for one database: the gateway is async and uses ``asyncpg``, while Celery
    workers are synchronous and use ``psycopg``. Keeping both explicit is clearer than
    rewriting a driver prefix at runtime.
    """

    model_config = _BASE_CONFIG

    database_url: str = "postgresql+asyncpg://story2audio:story2audio@localhost:5432/story2audio"
    database_url_sync: str = (
        "postgresql+psycopg://story2audio:story2audio@localhost:5432/story2audio"
    )
    db_pool_size: int = Field(default=5, ge=1, le=50)
    db_max_overflow: int = Field(default=5, ge=0, le=50)
    db_echo: bool = False


class RedisSettings(BaseSettings):
    """Redis endpoints.

    Separate logical databases so that flushing the Celery broker during development does
    not also wipe rate-limit counters or in-flight pub/sub channels.
    """

    model_config = _BASE_CONFIG

    redis_url: str = "redis://localhost:6379/0"
    celery_broker_url: str = "redis://localhost:6379/1"
    celery_result_backend: str = "redis://localhost:6379/2"


class StorageSettings(BaseSettings):
    """S3-compatible object storage: MinIO locally, Cloudflare R2 in production."""

    model_config = _BASE_CONFIG

    s3_endpoint_url: str | None = "http://localhost:9000"
    #: Address to sign URLs against, when it differs from the one used to talk to
    #: storage. Presigned URLs are handed to a browser, so they must name a host the
    #: browser can reach: inside docker compose the service connects to `minio:9000`,
    #: which resolves only on the docker network. Unset means "same as the endpoint",
    #: which is correct for R2 and for a local process outside compose.
    s3_public_endpoint_url: str | None = None
    s3_region: str = "auto"
    s3_bucket: str = "story2audio"
    s3_access_key_id: str = "minioadmin"
    s3_secret_access_key: SecretStr = SecretStr("minioadmin")
    #: MinIO needs path-style addressing; R2 and S3 use virtual-host style.
    s3_force_path_style: bool = True
    presigned_url_ttl_seconds: int = Field(default=3600, ge=60, le=604_800)
    audio_retention_days: int = Field(default=30, ge=1)


class LimitSettings(BaseSettings):
    """Quotas and spend guardrails.

    These are enforced in application code before any job is enqueued, not left to a
    billing alert. An open endpoint that spends GPU money per request needs a hard ceiling
    it cannot exceed, not a notification after it already has.
    """

    model_config = _BASE_CONFIG

    rate_limit_jobs_per_hour: int = Field(default=10, ge=1)
    rate_limit_uploads_per_hour: int = Field(default=5, ge=1)
    max_concurrent_jobs_per_session: int = Field(default=2, ge=1)
    global_daily_job_cap: int = Field(default=200, ge=1)

    max_prompt_chars: int = Field(default=2000, ge=50, le=20_000)
    max_voice_upload_bytes: int = Field(default=10 * 1024 * 1024, ge=1024)
    min_voice_duration_seconds: float = Field(default=6.0, gt=0)
    max_voice_duration_seconds: float = Field(default=120.0, gt=0)

    #: Reference samples are stored mono and clipped to this length. Voice cloning
    #: needs a few seconds, not a few minutes, and the stored clip is sent to the TTS
    #: engine over gRPC -- where a 30-second 48 kHz stereo file (5.7 MB) exceeds the
    #: default 4 MB message limit outright. Clipping is the fix; raising the limit
    #: would be v1's hack.
    reference_clip_seconds: float = Field(default=20.0, gt=0, le=60.0)

    @field_validator("max_voice_duration_seconds")
    @classmethod
    def _max_exceeds_min(cls, value: float, info: ValidationInfo) -> float:
        # Validated against the already-parsed minimum; a window where max <= min would
        # reject every possible upload, which is a config bug worth failing fast on.
        minimum = info.data.get("min_voice_duration_seconds")
        if minimum is not None and value <= minimum:
            raise ValueError(
                "max_voice_duration_seconds must be greater than min_voice_duration_seconds"
            )
        return value


@lru_cache(maxsize=1)
def core_settings() -> CoreSettings:
    return CoreSettings()


@lru_cache(maxsize=1)
def database_settings() -> DatabaseSettings:
    return DatabaseSettings()


@lru_cache(maxsize=1)
def redis_settings() -> RedisSettings:
    return RedisSettings()


@lru_cache(maxsize=1)
def storage_settings() -> StorageSettings:
    return StorageSettings()


@lru_cache(maxsize=1)
def limit_settings() -> LimitSettings:
    return LimitSettings()
