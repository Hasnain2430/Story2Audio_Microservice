"""End-to-end against the real compose stack.

Excluded from the default suite — these need Postgres, Redis, MinIO and all four
services actually running:

    docker compose -f infra/docker-compose.yml up --build -d
    uv run pytest tests/integration -m integration

This is the suite that covers what the fast tests deliberately cannot: real Postgres
semantics (partial indexes, row locking, genuine write concurrency), a real Redis, real
object storage, and a real gRPC hop to the engine. The divergences are listed in
`tests/conftest.py`.

With the stub TTS backend the audio is a tone rather than speech, which is exactly the
point — every structural property below is true regardless of what the model would have
produced.
"""

from __future__ import annotations

import asyncio
import io
import json
import os
import time
import wave
from contextlib import AbstractAsyncContextManager
from typing import Any

import httpx
import pytest
from httpx_ws import AsyncWebSocketSession, aconnect_ws

pytestmark = pytest.mark.integration

API = os.environ.get("STORY2AUDIO_API", "http://localhost:8000")

TERMINAL = {"done", "failed", "cancelled"}

#: A full run is LLM plus synthesis. Generous, because a cold Ollama model or a cold
#: engine both land inside it.
JOB_TIMEOUT_SECONDS = float(os.environ.get("STORY2AUDIO_JOB_TIMEOUT", "600"))


@pytest.fixture
async def client() -> Any:
    async with httpx.AsyncClient(base_url=API, timeout=30.0) as http_client:
        yield http_client


async def wait_for_terminal(client: httpx.AsyncClient, job_id: str) -> dict[str, Any]:
    deadline = time.monotonic() + JOB_TIMEOUT_SECONDS
    payload: dict[str, Any] = {}

    while time.monotonic() < deadline:
        response = await client.get(f"/v1/jobs/{job_id}")
        response.raise_for_status()
        payload = response.json()
        if payload["status"] in TERMINAL:
            return payload
        await asyncio.sleep(1.0)

    raise AssertionError(f"job stayed at {payload.get('status')!r} for {JOB_TIMEOUT_SECONDS}s")


def events_socket(
    client: httpx.AsyncClient, job_id: str
) -> AbstractAsyncContextManager[AsyncWebSocketSession]:
    """Open a job's event stream, with the session type made explicit."""
    return aconnect_ws(f"{API}/v1/jobs/{job_id}/events", client)


async def first_voice(client: httpx.AsyncClient) -> str:
    response = await client.get("/v1/voices")
    response.raise_for_status()
    items = response.json()["items"]
    assert items, "no built-in voices: did the migrate/seed step run?"
    voice_id: str = items[0]["id"]
    return voice_id


# --- Readiness -----------------------------------------------------------------------------


async def test_the_stack_is_up(client: httpx.AsyncClient) -> None:
    response = await client.get("/readyz")

    assert response.status_code == 200
    body = response.json()
    assert body["ready"] is True, body["checks"]


async def test_builtin_voices_were_seeded(client: httpx.AsyncClient) -> None:
    """Seeding reads `assets/voices`, the reference pack carried over from v1."""
    response = await client.get("/v1/voices")

    items = response.json()["items"]
    assert len(items) >= 5
    assert all(item["is_builtin"] for item in items)
    assert all(item["duration_seconds"] > 0 for item in items)


# --- The whole pipeline ----------------------------------------------------------------------


async def test_a_job_runs_end_to_end_and_produces_playable_audio(
    client: httpx.AsyncClient,
) -> None:
    """The one that matters: prompt in, audio out, without ever blocking a request."""
    voice_id = await first_voice(client)

    started = time.monotonic()
    created = await client.post(
        "/v1/jobs",
        json={
            "prompt": "A lighthouse keeper finds the lamp cold on a night with no moon.",
            "length": "short",
            "voice_id": voice_id,
        },
    )
    accepted_in = time.monotonic() - started

    assert created.status_code == 202
    # v1's equivalent call blocked for up to ten minutes. This one returns before the
    # work has even started.
    assert accepted_in < 2.0

    job_id = created.json()["id"]
    final = await wait_for_terminal(client, job_id)

    assert final["status"] == "done", final.get("error")
    assert final["story_text"]
    assert final["segment_count"] and final["segment_count"] >= 1

    formats = {asset["format"]: asset for asset in final["audio"]}
    assert {"mp3", "wav"} <= set(formats)

    # Follow the presigned URL and decode what comes back, rather than trusting that a
    # key exists.
    async with httpx.AsyncClient(timeout=60.0) as raw:
        wav_response = await raw.get(formats["wav"]["url"])
    wav_response.raise_for_status()

    with wave.open(io.BytesIO(wav_response.content), "rb") as handle:
        assert handle.getnchannels() == 1
        assert handle.getsampwidth() == 2
        assert handle.getnframes() > 0


async def test_stage_timings_show_both_stages_ran(client: httpx.AsyncClient) -> None:
    """These timestamps are the measured basis for the v1-versus-v2 comparison."""
    voice_id = await first_voice(client)
    created = await client.post(
        "/v1/jobs",
        json={
            "prompt": "A cartographer finds a road on no map.",
            "length": "short",
            "voice_id": voice_id,
        },
    )
    final = await wait_for_terminal(client, created.json()["id"])

    assert final["status"] == "done"
    timings = final["timings"]
    for field in ("queued_at", "writing_at", "written_at", "synthesizing_at", "finished_at"):
        assert timings[field], f"{field} was not recorded"


async def test_progress_streams_over_the_websocket(client: httpx.AsyncClient) -> None:
    """Tokens while the story is written, then real segment counts while it renders."""
    voice_id = await first_voice(client)
    created = await client.post(
        "/v1/jobs",
        json={"prompt": "A ferry crossing in heavy fog.", "length": "short", "voice_id": voice_id},
    )
    job_id = created.json()["id"]

    frames: list[dict[str, Any]] = []
    async with events_socket(client, job_id) as websocket:
        for _ in range(2_000):
            frames.append(json.loads(await websocket.receive_text()))
            if frames[-1]["type"] in TERMINAL:
                break

    types = [frame["type"] for frame in frames]
    assert types[0] == "status"
    assert frames[0]["snapshot"] is True
    assert "progress" in types
    assert types[-1] in TERMINAL

    progress = [frame for frame in frames if frame["type"] == "progress"]
    assert progress[-1]["done"] == progress[-1]["total"]

    # Sequence numbers are monotonic, so a gap would be detectable (ADR-0003).
    live = [frame["seq"] for frame in frames if not frame.get("snapshot")]
    assert live == sorted(live)

    # Token frames are deliberately NOT asserted here. Whether a client sees them
    # depends on winning a race against the writing stage, and with LLM_PROVIDER=fake
    # that stage finishes in tens of milliseconds. Missing them costs nothing by design:
    # the snapshot conveys state and `GET /v1/jobs/{id}` is authoritative for the text
    # (ADR-0003), which is exactly what makes the client's polling fallback safe.
    #
    # Token publication itself is covered deterministically, without a race, by
    # `tests/unit/test_story_pipeline.py::
    #  test_the_stage_publishes_a_writing_status_then_tokens_then_story_done`.


async def test_a_job_survives_the_client_disconnecting(client: httpx.AsyncClient) -> None:
    """The behaviour v1 could not offer: close the tab, come back to the finished result."""
    voice_id = await first_voice(client)
    created = await client.post(
        "/v1/jobs",
        json={
            "prompt": "A night watchman hears the docks creak.",
            "length": "short",
            "voice_id": voice_id,
        },
    )
    job_id = created.json()["id"]

    # Open the stream, read one frame, then walk away mid-job.
    async with events_socket(client, job_id) as websocket:
        await websocket.receive_text()

    final = await wait_for_terminal(client, job_id)
    assert final["status"] == "done"
    assert final["audio"]


async def test_cancelling_stops_a_running_job(client: httpx.AsyncClient) -> None:
    voice_id = await first_voice(client)
    created = await client.post(
        "/v1/jobs",
        json={
            "prompt": "An archivist opens the wrong drawer.",
            "length": "long",
            "voice_id": voice_id,
        },
    )
    job_id = created.json()["id"]

    cancelled = await client.delete(f"/v1/jobs/{job_id}")
    assert cancelled.status_code in {200, 409}

    final = await wait_for_terminal(client, job_id)
    assert final["status"] in {"cancelled", "done"}


async def test_idempotency_key_does_not_queue_the_work_twice(
    client: httpx.AsyncClient,
) -> None:
    voice_id = await first_voice(client)
    headers = {"Idempotency-Key": f"integration-{time.time()}"}
    body = {"prompt": "A bell rings in an empty tower.", "length": "short", "voice_id": voice_id}

    first = await client.post("/v1/jobs", json=body, headers=headers)
    second = await client.post("/v1/jobs", json=body, headers=headers)

    assert first.status_code == 202
    assert second.status_code == 200
    assert first.json()["id"] == second.json()["id"]


# --- Real Postgres behaviour -------------------------------------------------------------------


async def test_cursor_pagination_walks_the_list_without_gaps(
    client: httpx.AsyncClient,
) -> None:
    """Against real Postgres, where the partial index and ordering actually apply."""
    listed = await client.get("/v1/jobs?limit=2")
    listed.raise_for_status()

    seen: list[str] = []
    page = listed.json()
    for _ in range(20):
        seen.extend(item["id"] for item in page["items"])
        if not page["has_more"]:
            break
        following = await client.get(f"/v1/jobs?limit=2&cursor={page['next_cursor']}")
        page = following.json()

    assert len(seen) == len(set(seen))
