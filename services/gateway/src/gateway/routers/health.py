"""Liveness, readiness and metrics.

Split deliberately. ``/healthz`` answers "is this process alive" and must never touch a
dependency -- an orchestrator that restarts the API because Redis blipped turns a
degradation into an outage. ``/readyz`` answers "should traffic come here", and that one
does check dependencies.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Response, status
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from sqlalchemy import text

from gateway.deps import StateDep
from story2audio_shared.logging import get_logger

log = get_logger(__name__)

router = APIRouter(tags=["health"])


@router.get("/healthz", summary="Liveness")
async def healthz() -> dict[str, str]:
    """Process is up. No dependency is consulted, by design."""
    return {"status": "ok"}


@router.get("/readyz", summary="Readiness")
async def readyz(state: StateDep, response: Response) -> dict[str, Any]:
    """Dependencies are reachable.

    Reports per-dependency status rather than a bare boolean, so a failing readiness check
    says which dependency is at fault without anyone having to read logs.
    """
    checks: dict[str, str] = {}

    try:
        async with state.session_factory() as session:
            await session.execute(text("SELECT 1"))
        checks["database"] = "ok"
    except Exception as exc:  # noqa: BLE001 - any failure means not ready
        checks["database"] = "error"
        log.warning("readiness_database_failed", error=str(exc))

    try:
        await state.redis.ping()
        checks["redis"] = "ok"
    except Exception as exc:  # noqa: BLE001 - any failure means not ready
        checks["redis"] = "error"
        log.warning("readiness_redis_failed", error=str(exc))

    ready = all(value == "ok" for value in checks.values())
    if not ready:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return {"ready": ready, "checks": checks}


@router.get("/metrics", include_in_schema=False)
async def metrics() -> Response:
    """Prometheus exposition.

    Queue depth, stage durations and failure rate by error code are the numbers that make
    the queue architecture observable rather than merely asserted.
    """
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)
