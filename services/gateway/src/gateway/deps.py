"""Application state and request dependencies.

Long-lived resources — the database engine, the Redis pool, the storage client, the
dispatcher — are built once during startup and hung off ``app.state``. Request handlers
reach them through the dependencies here rather than through module-level globals, which
is what makes the whole app constructible against SQLite and fakeredis in tests without
touching production code paths.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Annotated
from uuid import UUID

from fastapi import Depends, Request
from redis.asyncio import ConnectionPool, Redis
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker
from starlette.websockets import WebSocket

from gateway.dispatch import JobDispatcher
from gateway.publisher import EventPublisher
from gateway.ratelimit import RateLimiter
from gateway.session import SessionCodec, read_session
from gateway.settings import GatewaySettings
from story2audio_shared.config import LimitSettings
from story2audio_shared.storage import ObjectStorage


@dataclass(slots=True)
class AppState:
    """Everything the app owns for its whole lifetime."""

    settings: GatewaySettings
    limits: LimitSettings
    engine: AsyncEngine
    session_factory: async_sessionmaker[AsyncSession]
    redis_pool: ConnectionPool
    redis: Redis
    storage: ObjectStorage
    publisher: EventPublisher
    rate_limiter: RateLimiter
    dispatcher: JobDispatcher
    session_codec: SessionCodec
    #: TTL applied to every presigned URL the API hands out. Short on purpose: a
    #: leaked media URL should stop working quickly.
    presigned_ttl_seconds: int


def get_state(request: Request) -> AppState:
    state: AppState = request.app.state.app_state
    return state


def get_ws_state(websocket: WebSocket) -> AppState:
    """``app.state`` for a WebSocket handshake.

    Separate accessor because Starlette gives WebSocket routes a ``WebSocket`` rather than
    a ``Request``; the state behind it is the same object.
    """
    state: AppState = websocket.app.state.app_state
    return state


async def get_session(
    state: Annotated[AppState, Depends(get_state)],
) -> AsyncIterator[AsyncSession]:
    """One database session per request, committed on success.

    A handler that raises gets a rollback, so a request cannot leave a half-applied
    change behind — which matters most on job creation, where the quota check and the
    insert have to agree.
    """
    async with state.session_factory() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise
        else:
            await session.commit()


def get_redis(state: Annotated[AppState, Depends(get_state)]) -> Redis:
    return state.redis


def get_storage(state: Annotated[AppState, Depends(get_state)]) -> ObjectStorage:
    return state.storage


def get_limits(state: Annotated[AppState, Depends(get_state)]) -> LimitSettings:
    return state.limits


def get_settings(state: Annotated[AppState, Depends(get_state)]) -> GatewaySettings:
    return state.settings


def get_dispatcher(state: Annotated[AppState, Depends(get_state)]) -> JobDispatcher:
    return state.dispatcher


def get_publisher(state: Annotated[AppState, Depends(get_state)]) -> EventPublisher:
    return state.publisher


def get_rate_limiter(state: Annotated[AppState, Depends(get_state)]) -> RateLimiter:
    return state.rate_limiter


def get_optional_user_id(
    request: Request, state: Annotated[AppState, Depends(get_state)]
) -> UUID | None:
    """The caller's user id, if they arrived with a valid session cookie.

    Returns ``None`` rather than raising for a missing or forged cookie. Creating the
    user row is the middleware's job, not a dependency's, because it needs to write a
    cookie onto the response.
    """
    return read_session(request, state.session_codec, state.settings.session_cookie_name)


def require_user_id(request: Request) -> UUID:
    """The caller's user id, guaranteed.

    The session middleware resolves or creates an identity for every request before
    routing, so by the time a handler runs this is always populated. A missing value is a
    wiring bug, not a client error.
    """
    user_id: UUID | None = getattr(request.state, "user_id", None)
    if user_id is None:
        raise RuntimeError("session middleware did not run before the route handler")
    return user_id


def require_ws_user_id(websocket: WebSocket) -> UUID | None:
    """The caller's user id on a WebSocket handshake, or ``None`` if unauthenticated.

    WebSocket routes bypass HTTP middleware in Starlette, so the cookie is read directly
    here. A connection with no valid session is closed rather than silently given a new
    identity — a fresh session could not own the job being watched anyway.
    """
    state = get_ws_state(websocket)
    return read_session(websocket, state.session_codec, state.settings.session_cookie_name)


StateDep = Annotated[AppState, Depends(get_state)]
SessionDep = Annotated[AsyncSession, Depends(get_session)]
RedisDep = Annotated[Redis, Depends(get_redis)]
StorageDep = Annotated[ObjectStorage, Depends(get_storage)]
LimitsDep = Annotated[LimitSettings, Depends(get_limits)]
SettingsDep = Annotated[GatewaySettings, Depends(get_settings)]
DispatcherDep = Annotated[JobDispatcher, Depends(get_dispatcher)]
PublisherDep = Annotated[EventPublisher, Depends(get_publisher)]
RateLimiterDep = Annotated[RateLimiter, Depends(get_rate_limiter)]
UserIdDep = Annotated[UUID, Depends(require_user_id)]
