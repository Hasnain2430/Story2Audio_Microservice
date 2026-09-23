"""Job routes.

``POST /v1/jobs`` is the endpoint the whole rebuild exists for. v1's equivalent held a
socket, a gRPC worker thread and the GPU lock for up to ten minutes. This one validates,
enqueues, and returns a job id — then the client watches progress over a WebSocket, or
polls, or closes the tab and comes back tomorrow to the same URL.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Header, Query, Response, WebSocket, WebSocketDisconnect, status
from fastapi.responses import RedirectResponse
from redis.asyncio.client import PubSub

from gateway.deps import (
    DispatcherDep,
    LimitsDep,
    PublisherDep,
    RateLimiterDep,
    SessionDep,
    StateDep,
    StorageDep,
    UserIdDep,
    get_ws_state,
    require_ws_user_id,
)
from gateway.services import jobs as job_service
from story2audio_shared.enums import TERMINAL_STATUSES
from story2audio_shared.errors import AppError
from story2audio_shared.events import TERMINAL_EVENT_TYPES, job_channel
from story2audio_shared.logging import bind_job_id, get_logger
from story2audio_shared.schemas import (
    CreateJobRequest,
    CreateJobResponse,
    JobResponse,
    Page,
)

log = get_logger(__name__)

router = APIRouter(prefix="/v1/jobs", tags=["jobs"])

#: WebSocket close codes. 4000-4999 is the application-defined range.
WS_UNAUTHENTICATED = 4401
WS_NOT_FOUND = 4404

#: How long the relay waits on Redis before looping. Bounded so that a cancelled task and
#: a client disconnect are both noticed promptly rather than at the next published event.
_PUBSUB_POLL_SECONDS = 1.0

#: Terminal status values as they appear on the wire.
_TERMINAL_STATUS_VALUES = frozenset(status.value for status in TERMINAL_STATUSES)


@router.post(
    "",
    response_model=CreateJobResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Submit a story for generation",
)
async def create_job(
    body: CreateJobRequest,
    user_id: UserIdDep,
    session: SessionDep,
    state: StateDep,
    limits: LimitsDep,
    rate_limiter: RateLimiterDep,
    dispatcher: DispatcherDep,
    response: Response,
    idempotency_key: Annotated[str | None, Header(alias="Idempotency-Key")] = None,
) -> CreateJobResponse:
    """Accept a job and return its id.

    ``202``, not ``200``: the work has been accepted, not performed. A repeated
    ``Idempotency-Key`` returns the original job rather than queueing the same GPU work
    twice, and is answered ``200`` so a client can tell a replay from a fresh submission.
    """
    job, created = await job_service.create_job(
        session,
        owner_id=user_id,
        request=body,
        limits=limits,
        rate_limiter=rate_limiter,
        idempotency_key=idempotency_key,
    )
    bind_job_id(job.id)

    if created:
        # Committed before dispatch: a worker must never pick up a job id that is not yet
        # visible in the database. The cost is that a crash between the two loses the
        # dispatch, which the queued-job sweeper recovers.
        await session.commit()
        await dispatcher.dispatch(job.id)
        log.info("job_accepted", job_id=str(job.id), length=job.length.value)
    else:
        response.status_code = status.HTTP_200_OK

    return CreateJobResponse(
        id=job.id,
        status=job.status,
        events_url=f"{state.settings.public_web_origin.rstrip('/')}/v1/jobs/{job.id}/events",
    )


@router.get("", response_model=Page[JobResponse], summary="List your jobs")
async def list_jobs(
    user_id: UserIdDep,
    session: SessionDep,
    storage: StorageDep,
    state: StateDep,
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
    cursor: Annotated[UUID | None, Query()] = None,
) -> Page[JobResponse]:
    """Newest first, cursor-paginated on the job id (ADR-0002)."""
    rows, has_more = await job_service.list_jobs(
        session, owner_id=user_id, limit=limit, cursor=cursor
    )
    ttl = state.presigned_ttl_seconds
    items = [job_service.to_response(job, storage, ttl_seconds=ttl) for job in rows]
    return Page[JobResponse](
        items=items,
        next_cursor=items[-1].id if items and has_more else None,
        has_more=has_more,
    )


@router.get("/{job_id}", response_model=JobResponse, summary="Fetch one job")
async def get_job(
    job_id: UUID,
    user_id: UserIdDep,
    session: SessionDep,
    storage: StorageDep,
    state: StateDep,
) -> JobResponse:
    """The authoritative state of a job.

    Sufficient on its own to drive the entire UI. The WebSocket is an optimisation over
    polling this endpoint, never a replacement for it — which is what makes the client's
    polling fallback safe.
    """
    job = await job_service.get_job(session, owner_id=user_id, job_id=job_id)
    return job_service.to_response(job, storage, ttl_seconds=state.presigned_ttl_seconds)


@router.get(
    "/{job_id}/segments/{index}/audio",
    summary="Play one rendered segment",
    response_class=RedirectResponse,
    status_code=status.HTTP_307_TEMPORARY_REDIRECT,
)
async def get_segment_audio(
    job_id: UUID,
    index: int,
    user_id: UserIdDep,
    session: SessionDep,
    storage: StorageDep,
    state: StateDep,
) -> RedirectResponse:
    """Redirect to a freshly signed URL for one segment of a job.

    This is what makes playback start before the job finishes: the worker publishes each
    segment as it is rendered, and the client fetches them here by index. A stable URL
    rather than a signed one in the event, so the link cannot expire between being
    announced and being used.

    307 rather than 302: the method must be preserved, and browsers and audio elements
    both follow it to the object store without a second round trip through this service.
    """
    url = await job_service.segment_audio_url(
        session,
        owner_id=user_id,
        job_id=job_id,
        index=index,
        storage=storage,
        ttl_seconds=state.presigned_ttl_seconds,
    )
    return RedirectResponse(url, status_code=status.HTTP_307_TEMPORARY_REDIRECT)


@router.delete("/{job_id}", response_model=JobResponse, summary="Cancel a job")
async def cancel_job(
    job_id: UUID,
    user_id: UserIdDep,
    session: SessionDep,
    storage: StorageDep,
    publisher: PublisherDep,
    state: StateDep,
) -> JobResponse:
    """Request cancellation.

    Workers check between segments, so this actually stops GPU work rather than only
    relabelling the row — the thing v1 could not do at all.
    """
    job = await job_service.cancel_job(session, publisher, owner_id=user_id, job_id=job_id)
    log.info("job_cancelled", job_id=str(job_id))
    return job_service.to_response(job, storage, ttl_seconds=state.presigned_ttl_seconds)


@router.websocket("/{job_id}/events")
async def job_events(websocket: WebSocket, job_id: UUID) -> None:
    """Stream a job's events.

    Ordering matters here and is deliberate:

    1. Authenticate from the session cookie, which the browser sends on the handshake
       exactly as it would on an HTTP request.
    2. **Subscribe to Redis first**, then read the snapshot from Postgres. Doing it the
       other way round leaves a window in which an event published between the read and
       the subscribe is lost forever. Subscribing first can duplicate a frame instead,
       which the client already tolerates.
    3. Relay until a terminal event, then close.

    Keepalives are WebSocket protocol pings from uvicorn (``--ws-ping-interval``), so
    there is no application-level heartbeat to implement or for a client to parse.
    """
    state = get_ws_state(websocket)
    user_id = require_ws_user_id(websocket)
    if user_id is None:
        await websocket.close(code=WS_UNAUTHENTICATED, reason="no session")
        return

    # Authorise before accepting, so an unauthorised peer is refused at the handshake
    # rather than being upgraded and then disconnected.
    async with state.session_factory() as session:
        try:
            await job_service.get_job(session, owner_id=user_id, job_id=job_id)
        except AppError:
            await websocket.close(code=WS_NOT_FOUND, reason="job not found")
            return

    await websocket.accept()
    bind_job_id(job_id)

    pubsub: PubSub = state.redis.pubsub(ignore_subscribe_messages=True)
    try:
        await pubsub.subscribe(job_channel(job_id))

        # Snapshot after subscribing, so nothing published in between is missed.
        async with state.session_factory() as session:
            current = await job_service.get_job(session, owner_id=user_id, job_id=job_id)
        await websocket.send_text(job_service.snapshot_event(current).model_dump_json())

        if current.status in TERMINAL_STATUSES:
            # Already finished. The snapshot is the whole story; nothing more will arrive.
            return

        await _relay_until_disconnect(websocket, pubsub)
    except WebSocketDisconnect:
        log.info("job_events_disconnected", job_id=str(job_id))
    finally:
        with contextlib.suppress(Exception):
            await pubsub.unsubscribe(job_channel(job_id))
        with contextlib.suppress(Exception):
            # redis-py's PubSub.aclose is untyped; the call is correct, the stub is not.
            await pubsub.aclose()  # type: ignore[no-untyped-call]
        with contextlib.suppress(Exception):
            await websocket.close()


async def _relay_until_disconnect(websocket: WebSocket, pubsub: PubSub) -> None:
    """Relay events, stopping as soon as either side is finished.

    Two concurrent tasks, whichever completes first wins:

    - the relay, which ends on a terminal event;
    - a watcher on the inbound half of the socket, which ends when the client goes away.

    The watcher is not optional. The relay only ever *sends*, so without something reading
    the socket a client that closes its tab leaves this coroutine parked on Redis with a
    live subscription, indefinitely. Protocol pings would eventually notice, but a
    connection that is already known to be dead should not wait on a timeout.
    """
    relay = asyncio.create_task(_relay(websocket, pubsub))
    watcher = asyncio.create_task(_watch_for_disconnect(websocket))

    done, pending = await asyncio.wait({relay, watcher}, return_when=asyncio.FIRST_COMPLETED)
    for task in pending:
        task.cancel()
    if pending:
        await asyncio.gather(*pending, return_exceptions=True)
    for task in done:
        # Surface a genuine relay failure; a disconnect is an ordinary ending.
        with contextlib.suppress(WebSocketDisconnect, asyncio.CancelledError):
            task.result()


async def _watch_for_disconnect(websocket: WebSocket) -> None:
    """Resolve when the client disconnects.

    Clients are not expected to send anything on this socket, so any inbound frame other
    than a disconnect is ignored rather than treated as a protocol error.
    """
    while True:
        message = await websocket.receive()
        if message["type"] == "websocket.disconnect":
            return


async def _relay(websocket: WebSocket, pubsub: PubSub) -> None:
    """Forward published frames until the job reaches a terminal event."""
    while True:
        message = await pubsub.get_message(
            ignore_subscribe_messages=True, timeout=_PUBSUB_POLL_SECONDS
        )
        if message is None:
            # Timed out with nothing published. Yield so a cancelled task can unwind.
            await asyncio.sleep(0)
            continue

        payload = message.get("data")
        if not isinstance(payload, str):
            continue

        await websocket.send_text(payload)
        if _is_terminal_payload(payload):
            return


def _is_terminal_payload(payload: str) -> bool:
    """Does this frame end the stream?

    Reads the discriminator out of the JSON rather than validating the whole event
    through the union: token frames are frequent, and a full model construction per frame
    to inspect one field is not worth it. A frame that will not parse is treated as
    non-terminal, so a malformed publish cannot silently close a live connection.
    """
    try:
        decoded = json.loads(payload)
    except json.JSONDecodeError:
        return False
    if not isinstance(decoded, dict):
        return False
    if decoded.get("type") in TERMINAL_EVENT_TYPES:
        return True
    # Belt and braces: a status frame reporting a terminal status also ends the stream.
    # A worker that sets the final status and then dies before emitting its `done` event
    # must not leave every watching client parked on the socket forever.
    return decoded.get("type") == "status" and decoded.get("status") in _TERMINAL_STATUS_VALUES
