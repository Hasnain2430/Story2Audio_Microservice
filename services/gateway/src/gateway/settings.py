"""Gateway configuration.

Composes the shared settings groups and adds what only the public edge needs: bind
address, CORS allowlist, session cookie policy, and how jobs are dispatched.
"""

from __future__ import annotations

from enum import StrEnum
from functools import lru_cache
from typing import Annotated

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict


class JobDispatch(StrEnum):
    """How an accepted job reaches a worker.

    ``celery`` is the real path. ``inline`` runs a canned pipeline in-process, which is
    what makes the whole API — including the WebSocket — testable before either worker
    exists, and is the test double thereafter. It is refused in production.
    """

    CELERY = "celery"
    INLINE = "inline"


class SameSitePolicy(StrEnum):
    LAX = "lax"
    STRICT = "strict"
    NONE = "none"


class GatewaySettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore", frozen=True
    )

    gateway_host: str = "0.0.0.0"  # noqa: S104 - binding all interfaces is correct in a container
    gateway_port: int = Field(default=8000, ge=1, le=65535)

    #: Exact origins allowed to call the API with credentials. Never "*": the API
    #: authenticates with a cookie, and the browser refuses wildcard-with-credentials
    #: anyway. Comma-separated in the environment.
    #:
    #: `NoDecode` is required. Without it pydantic-settings tries to JSON-decode any
    #: complex type straight from the environment variable, which fails outright on
    #: `a,b` before the validator below ever runs.
    cors_allow_origins: Annotated[list[str], NoDecode] = Field(
        default_factory=lambda: ["http://localhost:5173"]
    )

    public_web_origin: str = "http://localhost:5173"

    #: Signs the anonymous session cookie.
    session_secret: SecretStr = SecretStr("dev-only-insecure-secret-change-me")
    session_cookie_name: str = "s2a_session"
    #: Scope the cookie to the registrable domain (".example.com") so that an API on a
    #: sibling subdomain is same-site and `Lax` keeps working. Leave unset for localhost.
    session_cookie_domain: str | None = None
    session_cookie_samesite: SameSitePolicy = SameSitePolicy.LAX
    session_cookie_secure: bool = False
    session_max_age_seconds: int = Field(default=60 * 60 * 24 * 365, ge=3600)

    job_dispatch: JobDispatch = JobDispatch.CELERY

    #: Upper bound on a single request body, enforced before the body is buffered.
    max_request_bytes: int = Field(default=12 * 1024 * 1024, ge=1024)

    @field_validator("cors_allow_origins", mode="before")
    @classmethod
    def _split_origins(cls, value: object) -> object:
        if isinstance(value, str):
            return [origin.strip() for origin in value.split(",") if origin.strip()]
        return value

    @field_validator("cors_allow_origins")
    @classmethod
    def _reject_wildcard(cls, value: list[str]) -> list[str]:
        if "*" in value:
            raise ValueError(
                "cors_allow_origins must list exact origins; '*' is incompatible with "
                "cookie-authenticated requests"
            )
        return value

    @property
    def samesite_value(self) -> str:
        return self.session_cookie_samesite.value


@lru_cache(maxsize=1)
def gateway_settings() -> GatewaySettings:
    return GatewaySettings()
