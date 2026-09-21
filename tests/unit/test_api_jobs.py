"""Job API behaviour.

The headline assertion is the first one: submitting a job returns immediately with an id.
Everything else follows from that being true.
"""

from __future__ import annotations

import asyncio
from typing import Any
from uuid import UUID, uuid4

import pytest
from httpx import AsyncClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from story2audio_shared.models import Job, Voice
from tests.conftest import new_session_client, wait_for_status

TERMINAL = {"done", "failed", "cancelled"}


def payload(voice: Voice, **overrides: Any) -> dict[str, Any]:
    body: dict[str, Any] = {
        "prompt": "A lighthouse keeper hears knocking on a night with no boats.",
        "voice_id": str(voice.id),
    }
    body.update(overrides)
    return body


# --- The point of the rebuild -----------------------------------------------------------


async def test_submitting_a_job_returns_immediately_with_an_id(
    client: AsyncClient, builtin_voice: Voice
) -> None:
    """v1's equivalent call blocked for up to ten minutes."""
    started = asyncio.get_event_loop().time()
    response = await client.post("/v1/jobs", json=payload(builtin_voice))
    elapsed = asyncio.get_event_loop().time() - started

    assert response.status_code == 202
    body = response.json()
    assert body["status"] == "queued"
    assert body["id"]
    assert body["events_url"].endswith(f"/v1/jobs/{body['id']}/events")
    # Generous for a CI box; the real budget is ~200 ms. What is being pinned is that the
    # response does not wait on generation.
    assert elapsed < 1.0


async def test_the_job_is_retrievable_forever_after(
    client: AsyncClient, builtin_voice: Voice
) -> None:
    """State is durable, not tied to a connection.

    In v1 a page refresh destroyed the in-process future and the result became
    permanently unreachable even though the work continued.
    """
    created = await client.post("/v1/jobs", json=payload(builtin_voice))
    job_id = created.json()["id"]

    final = await wait_for_status(client, job_id, TERMINAL)
    assert final["status"] == "done"
    assert final["story_text"]

    # Fetched again later: identical.
    again = await client.get(f"/v1/jobs/{job_id}")
    assert again.status_code == 200
    assert again.json()["status"] == "done"


async def test_stage_timings_are_recorded(client: AsyncClient, builtin_voice: Voice) -> None:
    """The measurements behind the v1-versus-v2 comparison."""
    created = await client.post("/v1/jobs", json=payload(builtin_voice))
    final = await wait_for_status(client, created.json()["id"], TERMINAL)

    timings = final["timings"]
    assert timings["queued_at"]
    assert timings["writing_at"]
    assert timings["written_at"]
    assert timings["synthesizing_at"]
    assert timings["finished_at"]


# --- Validation ------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "overrides",
    [
        {"speed": 9.0},
        {"speed": 0.1},
        {"language": "klingon"},
        {"length": "enormous"},
        {"prompt": ""},
        {"emotion": "smug"},
        {"unknown_field": 1},
    ],
)
async def test_invalid_requests_are_rejected(
    client: AsyncClient, builtin_voice: Voice, overrides: dict[str, Any]
) -> None:
    response = await client.post("/v1/jobs", json=payload(builtin_voice, **overrides))
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "validation_failed"


async def test_validation_errors_name_fields_but_do_not_echo_values(
    client: AsyncClient, builtin_voice: Voice
) -> None:
    """Field paths help the client; reflected values are an injection surface."""
    secret = "super-secret-value-that-must-not-come-back"
    response = await client.post("/v1/jobs", json=payload(builtin_voice, language=secret))

    assert response.status_code == 422
    assert secret not in response.text
    assert any("language" in field for field in response.json()["error"]["fields"])


async def test_dialogue_mode_requires_a_second_voice(
    client: AsyncClient, builtin_voice: Voice
) -> None:
    response = await client.post(
        "/v1/jobs", json=payload(builtin_voice, mode="narration_with_dialogue")
    )
    assert response.status_code == 422


async def test_dialogue_mode_with_both_voices_is_accepted(
    client: AsyncClient, builtin_voice: Voice, second_builtin_voice: Voice
) -> None:
    response = await client.post(
        "/v1/jobs",
        json=payload(
            builtin_voice,
            mode="narration_with_dialogue",
            dialogue_voice_id=str(second_builtin_voice.id),
        ),
    )
    assert response.status_code == 202


async def test_unknown_voice_is_rejected(client: AsyncClient) -> None:
    response = await client.post("/v1/jobs", json={"prompt": "A story.", "voice_id": str(uuid4())})
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "voice_not_found"


async def test_para_level_sentinel_has_no_special_meaning(
    client: AsyncClient, builtin_voice: Voice
) -> None:
    """In v1 this string in the prompt body changed server-side model routing."""
    response = await client.post(
        "/v1/jobs", json=payload(builtin_voice, prompt="[PARA_LEVEL:8+] a quiet village")
    )
    assert response.status_code == 202

    fetched = await client.get(f"/v1/jobs/{response.json()['id']}")
    assert fetched.json()["length"] == "medium"


# --- Idempotency -------------------------------------------------------------------------------


async def test_repeating_an_idempotency_key_returns_the_original_job(
    client: AsyncClient, builtin_voice: Voice
) -> None:
    """A retried submission must not queue the same GPU work twice."""
    headers = {"Idempotency-Key": "client-generated-key-1"}

    first = await client.post("/v1/jobs", json=payload(builtin_voice), headers=headers)
    second = await client.post("/v1/jobs", json=payload(builtin_voice), headers=headers)

    assert first.status_code == 202
    # 200 rather than 202, so the client can tell a replay from a fresh submission.
    assert second.status_code == 200
    assert first.json()["id"] == second.json()["id"]


async def test_different_keys_create_different_jobs(
    client: AsyncClient, builtin_voice: Voice
) -> None:
    first = await client.post(
        "/v1/jobs", json=payload(builtin_voice), headers={"Idempotency-Key": "a"}
    )
    second = await client.post(
        "/v1/jobs", json=payload(builtin_voice), headers={"Idempotency-Key": "b"}
    )
    assert first.json()["id"] != second.json()["id"]


# --- Ownership --------------------------------------------------------------------------------


async def test_another_session_cannot_read_your_job(
    client: AsyncClient, app_state: Any, builtin_voice: Voice
) -> None:
    """Reported as 404, not 403: a 403 would confirm the id exists."""
    created = await client.post("/v1/jobs", json=payload(builtin_voice))
    job_id = created.json()["id"]

    async with new_session_client(app_state) as stranger:
        response = await stranger.get(f"/v1/jobs/{job_id}")

    assert response.status_code == 404
    assert response.json()["error"]["code"] == "job_not_found"


# --- Cancellation -------------------------------------------------------------------------------


async def test_cancelling_a_running_job_stops_it(client: AsyncClient, builtin_voice: Voice) -> None:
    """v1 had no cancellation at all: closing the tab left the GPU working."""
    created = await client.post("/v1/jobs", json=payload(builtin_voice))
    job_id = created.json()["id"]

    cancelled = await client.delete(f"/v1/jobs/{job_id}")
    if cancelled.status_code == 200:
        assert cancelled.json()["status"] == "cancelled"
    else:
        # The stub pipeline can finish before the cancel lands; that is a legitimate race
        # and the API must report it as a conflict rather than pretend to cancel.
        assert cancelled.status_code == 409
        assert cancelled.json()["error"]["code"] == "job_not_cancellable"


async def test_cancelling_a_finished_job_is_a_conflict(
    client: AsyncClient, builtin_voice: Voice
) -> None:
    created = await client.post("/v1/jobs", json=payload(builtin_voice))
    job_id = created.json()["id"]
    await wait_for_status(client, job_id, TERMINAL)

    response = await client.delete(f"/v1/jobs/{job_id}")
    assert response.status_code == 409
    assert response.json()["error"]["code"] == "job_not_cancellable"


async def test_cancelling_an_unknown_job_is_not_found(client: AsyncClient) -> None:
    response = await client.delete(f"/v1/jobs/{uuid4()}")
    assert response.status_code == 404


# --- Listing and pagination -----------------------------------------------------------------------


async def test_listing_returns_newest_first(client: AsyncClient, builtin_voice: Voice) -> None:
    created_ids = []
    for index in range(5):
        response = await client.post(
            "/v1/jobs", json=payload(builtin_voice, prompt=f"Story number {index}.")
        )
        created_ids.append(response.json()["id"])

    listed = await client.get("/v1/jobs")
    assert listed.status_code == 200
    returned = [item["id"] for item in listed.json()["items"]]
    assert returned == list(reversed(created_ids))


async def test_cursor_pagination_walks_the_whole_list_without_gaps(
    client: AsyncClient, builtin_voice: Voice
) -> None:
    """Cursor on the UUIDv7 id, so no row is skipped or repeated (ADR-0002)."""
    for index in range(7):
        await client.post("/v1/jobs", json=payload(builtin_voice, prompt=f"Story {index}."))

    seen: list[str] = []
    cursor: str | None = None
    for _ in range(10):
        url = "/v1/jobs?limit=3" + (f"&cursor={cursor}" if cursor else "")
        page = (await client.get(url)).json()
        seen.extend(item["id"] for item in page["items"])
        if not page["has_more"]:
            break
        cursor = page["next_cursor"]

    assert len(seen) == 7
    assert len(set(seen)) == 7


async def test_listing_only_shows_your_own_jobs(
    client: AsyncClient, app_state: Any, builtin_voice: Voice
) -> None:
    await client.post("/v1/jobs", json=payload(builtin_voice))

    async with new_session_client(app_state) as stranger:
        listed = await stranger.get("/v1/jobs")

    assert listed.json()["items"] == []


# --- Persistence ---------------------------------------------------------------------------------


async def test_job_row_records_the_request_faithfully(
    client: AsyncClient,
    builtin_voice: Voice,
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    response = await client.post(
        "/v1/jobs",
        json=payload(builtin_voice, length="long", language="fr", emotion="sad", speed=1.25),
    )
    job_id = response.json()["id"]

    async with session_factory() as session:
        job = await session.scalar(select(Job).where(Job.id == UUID(job_id)))

    assert job is not None
    assert job.length.value == "long"
    assert job.language.value == "fr"
    assert job.emotion.value == "sad"
    assert job.speed == 1.25
