"""Cross-cutting API behaviour: identity, guardrails, error shape, health.

These are the properties v1 had none of. An open endpoint that spends GPU money per
request needs all four.
"""

from __future__ import annotations

from typing import Any

import pytest
from httpx import AsyncClient

from gateway.deps import AppState
from gateway.session import SessionCodec
from gateway.settings import GatewaySettings
from story2audio_shared.config import LimitSettings
from story2audio_shared.ids import uuid7
from story2audio_shared.models import Voice
from tests.conftest import make_wav, new_session_client
from tests.unit.test_api_voices import upload_files

JOB_BODY_PROMPT = "A cartographer finds a road that is not on any map."


def job_body(voice: Voice, **overrides: Any) -> dict[str, Any]:
    body: dict[str, Any] = {"prompt": JOB_BODY_PROMPT, "voice_id": str(voice.id)}
    body.update(overrides)
    return body


# --- Anonymous identity ---------------------------------------------------------------------


async def test_first_request_issues_a_session_cookie(client: AsyncClient) -> None:
    response = await client.get("/v1/voices")
    assert response.status_code == 200
    assert "s2a_session" in response.cookies


async def test_the_session_cookie_is_http_only(client: AsyncClient) -> None:
    """A bearer token in localStorage is readable by any injected script; a cookie is not."""
    response = await client.get("/v1/voices")
    set_cookie = response.headers["set-cookie"].lower()
    assert "httponly" in set_cookie
    assert "samesite=lax" in set_cookie


async def test_identity_persists_across_requests(client: AsyncClient, builtin_voice: Voice) -> None:
    await client.post("/v1/jobs", json=job_body(builtin_voice))
    await client.post("/v1/jobs", json=job_body(builtin_voice, prompt="A second story."))

    listed = await client.get("/v1/jobs")
    assert len(listed.json()["items"]) == 2


async def test_two_clients_get_two_identities(
    client: AsyncClient, app_state: AppState, builtin_voice: Voice
) -> None:
    await client.post("/v1/jobs", json=job_body(builtin_voice))

    async with new_session_client(app_state) as other:
        assert (await other.get("/v1/jobs")).json()["items"] == []


async def test_a_forged_cookie_yields_a_fresh_session_rather_than_an_error(
    client: AsyncClient,
) -> None:
    """There is nothing a caller can do to repair a bad cookie, so failing them is wrong."""
    client.cookies.set("s2a_session", "not-a-valid-signed-value")

    response = await client.get("/v1/voices")
    assert response.status_code == 200
    assert "s2a_session" in response.cookies


def test_a_cookie_signed_with_another_secret_is_rejected() -> None:
    """The signature is what stops one caller adopting another's identity."""
    attacker = SessionCodec(GatewaySettings(session_secret="attacker-secret"))  # type: ignore[arg-type]
    victim = SessionCodec(GatewaySettings(session_secret="real-secret"))  # type: ignore[arg-type]

    forged = attacker.dumps(uuid7())
    assert victim.loads(forged) is None


def test_a_valid_cookie_round_trips() -> None:
    codec = SessionCodec(GatewaySettings(session_secret="a-secret"))  # type: ignore[arg-type]
    user_id = uuid7()
    assert codec.loads(codec.dumps(user_id)) == user_id


def test_cors_origins_may_not_be_a_wildcard() -> None:
    """A wildcard is incompatible with cookie authentication, and browsers reject it."""
    with pytest.raises(ValueError, match="exact origins"):
        GatewaySettings(session_secret="x", cors_allow_origins=["*"])  # type: ignore[arg-type]


# --- Guardrails ------------------------------------------------------------------------


@pytest.fixture
def limits() -> LimitSettings:
    """Tight limits, so the guardrails are actually reached."""
    return LimitSettings(
        rate_limit_jobs_per_hour=3,
        rate_limit_uploads_per_hour=2,
        max_concurrent_jobs_per_session=2,
        global_daily_job_cap=5,
        max_prompt_chars=200,
    )


async def test_hourly_job_limit_is_enforced(client: AsyncClient, builtin_voice: Voice) -> None:
    statuses = []
    for index in range(5):
        response = await client.post(
            "/v1/jobs", json=job_body(builtin_voice, prompt=f"Story {index}.")
        )
        statuses.append(response.status_code)

    assert 429 in statuses
    limited = next(code for code in statuses if code == 429)
    assert limited == 429


async def test_rate_limited_responses_say_when_to_retry(
    client: AsyncClient, builtin_voice: Voice
) -> None:
    last = None
    for index in range(6):
        last = await client.post("/v1/jobs", json=job_body(builtin_voice, prompt=f"Story {index}."))
        if last.status_code == 429:
            break

    assert last is not None
    assert last.status_code == 429
    assert "Retry-After" in last.headers
    assert last.json()["error"]["retryable"] is True


async def test_deployment_prompt_ceiling_is_enforced(
    client: AsyncClient, builtin_voice: Voice
) -> None:
    """Tighter than the schema's absolute maximum, which no deployment may exceed."""
    response = await client.post("/v1/jobs", json=job_body(builtin_voice, prompt="x" * 500))
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "prompt_too_long"


async def test_upload_limit_is_enforced(client: AsyncClient) -> None:
    sample = make_wav(8.0)
    statuses = []
    for index in range(4):
        response = await client.post(
            "/v1/voices", data={"name": f"Voice {index}"}, files=upload_files(sample)
        )
        statuses.append(response.status_code)

    assert 429 in statuses


# --- Error envelope --------------------------------------------------------------------


async def test_every_error_uses_one_envelope(client: AsyncClient) -> None:
    """One shape for the frontend to handle, not three."""
    from uuid import uuid4

    responses = [
        await client.get(f"/v1/jobs/{uuid4()}"),
        await client.post("/v1/jobs", json={"prompt": "", "voice_id": str(uuid4())}),
        await client.delete(f"/v1/voices/{uuid4()}"),
    ]

    for response in responses:
        assert response.status_code >= 400
        body = response.json()
        assert set(body) >= {"error"}
        assert set(body["error"]) >= {"code", "message", "retryable"}
        assert isinstance(body["error"]["retryable"], bool)


async def test_errors_do_not_leak_internal_detail(client: AsyncClient) -> None:
    """v1 returned `str(exception)` verbatim, paths and all."""
    from uuid import uuid4

    response = await client.get(f"/v1/jobs/{uuid4()}")
    text = response.text.lower()

    for leak in ("traceback", "sqlalchemy", "site-packages", "/app/", "select "):
        assert leak not in text


async def test_responses_carry_a_trace_id(client: AsyncClient) -> None:
    """One id ties a user-visible failure to the log line that explains it."""
    response = await client.get("/v1/voices")
    assert response.headers["x-trace-id"]


async def test_an_inbound_trace_id_is_honoured(client: AsyncClient) -> None:
    """So a trace can span the frontend and the API rather than restarting at the edge."""
    response = await client.get("/v1/voices", headers={"X-Trace-Id": "abc123"})
    assert response.headers["x-trace-id"] == "abc123"


# --- Health ----------------------------------------------------------------------------


async def test_liveness_touches_no_dependency(client: AsyncClient) -> None:
    """An orchestrator restarting the API because Redis blipped turns a blip into an outage."""
    response = await client.get("/healthz")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


async def test_readiness_reports_each_dependency(client: AsyncClient) -> None:
    response = await client.get("/readyz")
    assert response.status_code == 200

    body = response.json()
    assert body["ready"] is True
    assert body["checks"] == {"database": "ok", "redis": "ok"}


async def test_metrics_are_exposed_for_prometheus(client: AsyncClient) -> None:
    response = await client.get("/metrics")
    assert response.status_code == 200
    assert "text/plain" in response.headers["content-type"]


async def test_the_openapi_schema_is_published(client: AsyncClient) -> None:
    """The frontend generates its TypeScript types from this, so it must exist."""
    response = await client.get("/openapi.json")
    assert response.status_code == 200

    schema = response.json()
    assert "/v1/jobs" in schema["paths"]
    assert "/v1/voices" in schema["paths"]
