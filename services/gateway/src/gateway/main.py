"""Application assembly.

`create_app` is a factory rather than a module-level singleton so tests can build the
whole application against SQLite, fakeredis and an in-memory storage stub without
touching a single production code path. That is the difference between testing the API
and testing a parallel implementation of it.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from gateway.db import create_engine, create_session_factory
from gateway.deps import AppState
from gateway.dispatch import CeleryDispatcher, InlineDispatcher, JobDispatcher
from gateway.errors import register_exception_handlers
from gateway.middleware import (
    BodySizeLimitMiddleware,
    RequestContextMiddleware,
    SessionMiddleware,
)
from gateway.publisher import EventPublisher
from gateway.ratelimit import RateLimiter
from gateway.redis_client import create_redis, create_redis_pool
from gateway.routers import health, jobs, voices
from gateway.session import SessionCodec
from gateway.settings import GatewaySettings, JobDispatch, gateway_settings
from story2audio_shared.config import (
    Environment,
    core_settings,
    database_settings,
    limit_settings,
    redis_settings,
    storage_settings,
)
from story2audio_shared.logging import configure_logging, get_logger
from story2audio_shared.storage import ObjectStorage

log = get_logger(__name__)

API_TITLE = "Story2Audio"
API_VERSION = "2.0.0"

API_DESCRIPTION = """\
Asynchronous story-to-audio generation.

`POST /v1/jobs` accepts a request and returns a job id immediately. Progress arrives over
`WS /v1/jobs/{id}/events`, and `GET /v1/jobs/{id}` is authoritative at any time — the
WebSocket is an optimisation over polling it, never a replacement.
"""


def build_state(settings: GatewaySettings | None = None) -> AppState:
    """Construct every long-lived resource the app owns."""
    gateway = settings or gateway_settings()
    storage_config = storage_settings()

    engine = create_engine(database_settings())
    session_factory = create_session_factory(engine)

    redis_pool = create_redis_pool(redis_settings())
    redis = create_redis(redis_pool)

    publisher = EventPublisher(redis)
    dispatcher = _build_dispatcher(gateway, session_factory, publisher)

    return AppState(
        settings=gateway,
        limits=limit_settings(),
        engine=engine,
        session_factory=session_factory,
        redis_pool=redis_pool,
        redis=redis,
        storage=ObjectStorage(storage_config),
        publisher=publisher,
        rate_limiter=RateLimiter(redis),
        dispatcher=dispatcher,
        session_codec=SessionCodec(gateway),
        presigned_ttl_seconds=storage_config.presigned_url_ttl_seconds,
    )


def _build_dispatcher(
    settings: GatewaySettings,
    session_factory: object,
    publisher: EventPublisher,
) -> JobDispatcher:
    if settings.job_dispatch is JobDispatch.INLINE:
        if core_settings().environment is Environment.PRODUCTION:
            # The inline dispatcher returns canned text and never synthesizes audio.
            # Reaching production with it enabled would mean silently serving stubs.
            raise RuntimeError("job_dispatch=inline is not permitted in production")
        log.warning("using_inline_dispatcher", reason="development or test configuration")
        return InlineDispatcher(session_factory, publisher)  # type: ignore[arg-type]
    return CeleryDispatcher(redis_settings().celery_broker_url)


def create_app(state: AppState | None = None) -> FastAPI:
    """Build the ASGI application.

    Passing ``state`` substitutes the whole resource set — used by tests, never in
    production, where the lifespan builds it.
    """
    core = core_settings()
    configure_logging(level=core.log_level, json_output=core.log_format.value == "json")

    settings = state.settings if state is not None else gateway_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.app_state = state if state is not None else build_state(settings)
        log.info(
            "gateway_started",
            environment=core.environment.value,
            dispatch=settings.job_dispatch.value,
        )
        try:
            yield
        finally:
            await _shutdown(app.state.app_state, owns_resources=state is None)
            log.info("gateway_stopped")

    app = FastAPI(
        title=API_TITLE,
        version=API_VERSION,
        description=API_DESCRIPTION,
        lifespan=lifespan,
        # The canonical schema. The frontend generates its TypeScript types from this, so
        # client and server cannot drift.
        openapi_url="/openapi.json",
        docs_url="/docs" if core.environment is not Environment.PRODUCTION else None,
        redoc_url=None,
    )

    register_exception_handlers(app)
    _install_middleware(app, settings)

    app.include_router(health.router)
    app.include_router(jobs.router)
    app.include_router(voices.router)
    return app


def _install_middleware(app: FastAPI, settings: GatewaySettings) -> None:
    """Install middleware.

    Starlette runs middleware in reverse registration order, so the last one added is the
    outermost. Registration here is therefore innermost-first: session, then body limit,
    then request context, then CORS at the edge.
    """
    app.add_middleware(
        SessionMiddleware,
        codec=SessionCodec(settings),
        cookie_name=settings.session_cookie_name,
    )
    app.add_middleware(BodySizeLimitMiddleware, max_bytes=settings.max_request_bytes)
    app.add_middleware(RequestContextMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_allow_origins,
        # Required: the API authenticates with a cookie, so the browser must be told to
        # send it. This is also why the origin list can never be "*".
        allow_credentials=True,
        allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
        allow_headers=["Content-Type", "Idempotency-Key", "X-Trace-Id"],
        expose_headers=["X-Trace-Id", "Retry-After"],
        max_age=600,
    )


async def _shutdown(state: AppState, *, owns_resources: bool) -> None:
    """Release resources on the way out.

    Only what this process created. A test that supplied its own state keeps ownership of
    it, so shutting the app down does not tear down fixtures the test still needs.
    """
    await state.dispatcher.shutdown()
    if not owns_resources:
        return
    await state.redis.aclose()
    await state.redis_pool.aclose()
    await state.engine.dispose()


app = create_app  # uvicorn factory entrypoint: `uvicorn gateway.main:app --factory`
