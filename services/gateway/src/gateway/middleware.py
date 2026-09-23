"""HTTP middleware: request identity, correlation ids, body-size ceiling.

Ordered outermost-first when installed in :mod:`gateway.main`:

1. :class:`RequestContextMiddleware` — assigns a trace id and binds it to the logger.
2. :class:`BodySizeLimitMiddleware` — rejects oversized bodies before they are buffered.
3. :class:`SessionMiddleware` — resolves or creates the anonymous identity.
"""

from __future__ import annotations

import uuid
from collections.abc import Awaitable, Callable

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.types import ASGIApp

from gateway.errors import error_body
from gateway.session import SessionCodec, read_session, write_session
from story2audio_shared.errors import ErrorCode
from story2audio_shared.ids import uuid7
from story2audio_shared.logging import bind_trace_id, get_logger
from story2audio_shared.models import User

log = get_logger(__name__)

TRACE_HEADER = "x-trace-id"

CallNext = Callable[[Request], Awaitable[Response]]


class RequestContextMiddleware(BaseHTTPMiddleware):
    """Assign a trace id to every request and echo it back.

    An inbound ``X-Trace-Id`` is honoured so a trace can span the frontend and the API;
    otherwise one is minted. The id is bound to the logging context, and from Phase 3 it
    travels into the Celery task headers, so one story can be followed from the browser
    through the queue into both workers.
    """

    async def dispatch(self, request: Request, call_next: CallNext) -> Response:
        trace_id = request.headers.get(TRACE_HEADER) or uuid.uuid4().hex
        bind_trace_id(trace_id)
        request.state.trace_id = trace_id

        response = await call_next(request)
        response.headers[TRACE_HEADER] = trace_id
        return response


class BodySizeLimitMiddleware(BaseHTTPMiddleware):
    """Reject bodies above the configured ceiling.

    Checked against ``Content-Length`` before the body is read, so an oversized upload
    costs a header parse rather than the memory to buffer it. A chunked request without
    ``Content-Length`` passes here and is caught by the per-endpoint read limit instead.
    """

    def __init__(self, app: ASGIApp, *, max_bytes: int) -> None:
        super().__init__(app)
        self._max_bytes = max_bytes

    async def dispatch(self, request: Request, call_next: CallNext) -> Response:
        raw_length = request.headers.get("content-length")
        if raw_length is not None:
            try:
                declared = int(raw_length)
            except ValueError:
                declared = 0
            if declared > self._max_bytes:
                return JSONResponse(status_code=413, content=error_body(ErrorCode.VOICE_TOO_LARGE))
        return await call_next(request)


class SessionMiddleware(BaseHTTPMiddleware):
    """Resolve the caller's anonymous identity, creating one on first contact.

    Runs before routing so every handler can rely on ``request.state.user_id``. A cookie
    that is missing, forged or pointing at a deleted user yields a fresh identity rather
    than an error — there is nothing the caller could do to fix a bad cookie, and
    stranding them behind a failure would be worse than issuing a new session.

    The database session factory is read from ``app.state`` per request rather than
    injected at construction time: middleware is installed while the app is being built,
    but the engine is not created until the lifespan starts.
    """

    def __init__(self, app: ASGIApp, *, codec: SessionCodec, cookie_name: str) -> None:
        super().__init__(app)
        self._codec = codec
        self._cookie_name = cookie_name

    async def dispatch(self, request: Request, call_next: CallNext) -> Response:
        factory: async_sessionmaker[AsyncSession] = request.app.state.app_state.session_factory
        claimed = read_session(request, self._codec, self._cookie_name)
        user_id, is_new = await self._resolve(factory, claimed)

        request.state.user_id = user_id
        response = await call_next(request)

        if is_new:
            write_session(response, user_id, self._codec)
        return response

    async def _resolve(
        self, factory: async_sessionmaker[AsyncSession], claimed: uuid.UUID | None
    ) -> tuple[uuid.UUID, bool]:
        async with factory() as session:
            if claimed is not None:
                existing = await session.scalar(select(User.id).where(User.id == claimed))
                if existing is not None:
                    return existing, False

            user = User(id=uuid7())
            session.add(user)
            await session.commit()
            log.info("session_created", user_id=str(user.id))
            return user.id, True
