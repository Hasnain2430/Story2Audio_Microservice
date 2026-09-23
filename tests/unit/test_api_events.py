"""WebSocket event stream (ADR-0003).

Driven through the real ASGI app on the test's own event loop, so the handshake, cookie
authentication, Redis subscription and relay are all production code paths.
"""

from __future__ import annotations

import contextlib
import json
from collections.abc import AsyncIterator
from contextlib import AbstractAsyncContextManager
from typing import Any
from uuid import uuid4

from httpx import AsyncClient
from httpx_ws import (
    AsyncWebSocketSession,
    WebSocketDisconnect,
    WebSocketUpgradeError,
    aconnect_ws,
)

from gateway.deps import AppState
from gateway.routers.jobs import WS_NOT_FOUND
from story2audio_shared.events import job_event_adapter
from story2audio_shared.models import Voice
from tests.conftest import wait_for_status, websocket_client

TERMINAL_TYPES = {"done", "failed", "cancelled"}
NORMAL_CLOSURE = 1000


# --- Helpers -----------------------------------------------------------------------------------


def _events_socket(
    client: AsyncClient, job_id: str
) -> AbstractAsyncContextManager[AsyncWebSocketSession]:
    """Open the event stream for a job, with the session type made explicit."""
    return aconnect_ws(f"/v1/jobs/{job_id}/events", client)


def _leaves(exc: BaseException) -> list[BaseException]:
    """Flatten an ExceptionGroup, which anyio task groups raise."""
    if isinstance(exc, BaseExceptionGroup):
        return [leaf for sub in exc.exceptions for leaf in _leaves(sub)]
    return [exc]


@contextlib.asynccontextmanager
async def _tolerating_server_close() -> AsyncIterator[None]:
    """Swallow the clean close the server sends after a terminal event.

    The route closes the socket once the job finishes, so the client's context manager
    exits with a normal-closure disconnect. That is the designed behaviour, not a failure
    — but any other close code still propagates.
    """
    try:
        yield
    except BaseException as exc:
        for leaf in _leaves(exc):
            if not (isinstance(leaf, WebSocketDisconnect) and leaf.code == NORMAL_CLOSURE):
                raise


async def _create_job(client: AsyncClient, voice: Voice) -> str:
    response = await client.post(
        "/v1/jobs",
        json={"prompt": "A ferry crossing in heavy fog.", "voice_id": str(voice.id)},
    )
    assert response.status_code == 202
    job_id: str = response.json()["id"]
    return job_id


async def _collect(client: AsyncClient, job_id: str, *, limit: int = 300) -> list[dict[str, Any]]:
    """Open the stream and read frames until it ends."""
    frames: list[dict[str, Any]] = []
    async with _tolerating_server_close(), _events_socket(client, job_id) as ws:
        for _ in range(limit):
            frames.append(json.loads(await ws.receive_text()))
            if frames[-1]["type"] in TERMINAL_TYPES:
                break
    return frames


async def _refusal_code(client: AsyncClient, path: str) -> int:
    """Attempt a connection that should be refused, and report how.

    Over a real network a pre-accept ``close()`` arrives as a failed HTTP upgrade; over
    the in-process ASGI transport it arrives as a close frame carrying the same code.
    Both are accepted here, and the code itself is what gets asserted.
    """
    try:
        async with aconnect_ws(path, client):
            pass
    except BaseException as exc:
        for leaf in _leaves(exc):
            if isinstance(leaf, WebSocketDisconnect):
                return leaf.code
            if isinstance(leaf, WebSocketUpgradeError):
                return leaf.response.status_code
        raise
    raise AssertionError(f"connection to {path} was accepted but should have been refused")


# --- Authentication ---------------------------------------------------------------------------


async def test_watching_someone_elses_job_is_refused(
    app_state: AppState, builtin_voice: Voice
) -> None:
    """Refused at the handshake rather than upgraded and then disconnected."""
    async with websocket_client(app_state) as owner:
        job_id = await _create_job(owner, builtin_voice)

    async with websocket_client(app_state) as stranger:
        # Establish a valid session for the stranger — just not the owner's.
        await stranger.get("/v1/voices")
        code = await _refusal_code(stranger, f"/v1/jobs/{job_id}/events")

    # Not found, not forbidden: a 403 would confirm the job id exists.
    assert code == WS_NOT_FOUND


async def test_watching_an_unknown_job_is_refused(app_state: AppState) -> None:
    async with websocket_client(app_state) as client:
        await client.get("/v1/voices")
        code = await _refusal_code(client, f"/v1/jobs/{uuid4()}/events")

    assert code == WS_NOT_FOUND


# --- Snapshot on connect -------------------------------------------------------------------------


async def test_first_frame_is_a_snapshot_built_from_the_database(
    app_state: AppState, builtin_voice: Voice
) -> None:
    """A late or reconnecting client starts from truth, not from the next publish."""
    async with websocket_client(app_state) as client:
        job_id = await _create_job(client, builtin_voice)
        async with (
            _tolerating_server_close(),
            _events_socket(client, job_id) as websocket,
        ):
            first = json.loads(await websocket.receive_text())

    assert first["type"] == "status"
    assert first["snapshot"] is True
    assert first["job_id"] == job_id


async def test_connecting_after_the_job_finished_still_yields_its_state(
    app_state: AppState, builtin_voice: Voice
) -> None:
    """The case that made v1's results unreachable: arriving after the work is done."""
    async with websocket_client(app_state) as client:
        job_id = await _create_job(client, builtin_voice)
        await wait_for_status(client, job_id, TERMINAL_TYPES)

        async with (
            _tolerating_server_close(),
            _events_socket(client, job_id) as websocket,
        ):
            first = json.loads(await websocket.receive_text())

    assert first["type"] == "status"
    assert first["status"] == "done"
    assert first["snapshot"] is True


# --- Event contract -------------------------------------------------------------------------------


async def test_every_frame_parses_as_a_known_event_variant(
    app_state: AppState, builtin_voice: Voice
) -> None:
    """The discriminated union is what gives the TypeScript client an exhaustive switch."""
    async with websocket_client(app_state) as client:
        job_id = await _create_job(client, builtin_voice)
        frames = await _collect(client, job_id)

    assert frames
    for frame in frames:
        job_event_adapter.validate_python(frame)


async def test_sequence_numbers_are_monotonic_across_the_stream(
    app_state: AppState, builtin_voice: Voice
) -> None:
    """Without this a client cannot tell "quiet" from "I missed four frames"."""
    async with websocket_client(app_state) as client:
        job_id = await _create_job(client, builtin_voice)
        frames = await _collect(client, job_id)

    # The snapshot deliberately reuses the current sequence number, so it is excluded.
    live = [frame for frame in frames if not frame.get("snapshot")]
    sequences = [frame["seq"] for frame in live]
    assert sequences == sorted(sequences)
    assert all(seq >= 1 for seq in sequences)


async def test_every_frame_carries_the_schema_version_and_job_id(
    app_state: AppState, builtin_voice: Voice
) -> None:
    async with websocket_client(app_state) as client:
        job_id = await _create_job(client, builtin_voice)
        frames = await _collect(client, job_id)

    assert frames
    for frame in frames:
        assert frame["v"] == 1
        assert frame["job_id"] == job_id


async def test_story_tokens_arrive_before_synthesis_progress(
    app_state: AppState, builtin_voice: Voice
) -> None:
    """Text reaches the reader while the audio work is still ahead of it.

    This is the perceived-latency win: v1 showed a spinner for the whole ten minutes.
    """
    async with websocket_client(app_state) as client:
        job_id = await _create_job(client, builtin_voice)
        frames = await _collect(client, job_id)

    types = [frame["type"] for frame in frames]
    assert "token" in types
    assert "progress" in types
    assert types.index("token") < types.index("progress")


async def test_progress_reports_real_segment_counts(
    app_state: AppState, builtin_voice: Voice
) -> None:
    """Not an indeterminate spinner: the segment total is known before synthesis starts."""
    async with websocket_client(app_state) as client:
        job_id = await _create_job(client, builtin_voice)
        frames = await _collect(client, job_id)

    progress = [frame for frame in frames if frame["type"] == "progress"]
    assert progress
    assert all(frame["total"] >= 1 for frame in progress)
    assert [frame["done"] for frame in progress] == sorted(frame["done"] for frame in progress)
    assert progress[-1]["done"] == progress[-1]["total"]


async def test_the_stream_ends_on_a_terminal_event(
    app_state: AppState, builtin_voice: Voice
) -> None:
    async with websocket_client(app_state) as client:
        job_id = await _create_job(client, builtin_voice)
        frames = await _collect(client, job_id)

    assert frames[-1]["type"] in TERMINAL_TYPES


# --- The fallback contract -------------------------------------------------------------


async def test_polling_alone_reaches_the_same_final_state(
    client: AsyncClient, builtin_voice: Voice
) -> None:
    """The WebSocket is an optimisation over polling, never a requirement.

    This is the property that makes the client's polling fallback safe: a client that
    never opens a socket sees exactly the same outcome.
    """
    job_id = await _create_job(client, builtin_voice)
    final = await wait_for_status(client, job_id, TERMINAL_TYPES)

    assert final["status"] == "done"
    assert final["story_text"]
