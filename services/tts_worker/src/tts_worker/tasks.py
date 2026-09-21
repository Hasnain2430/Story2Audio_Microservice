"""Celery tasks for the TTS stage."""

from __future__ import annotations

import os
from uuid import UUID

from celery import Task
from sqlalchemy import update

from story2audio_shared.errors import AppError, ErrorCode, is_retryable
from story2audio_shared.logging import bind_job_id, bind_trace_id, get_logger
from story2audio_shared.models import Job
from story2audio_shared.worker import fail_job, session_scope
from tts_worker.app import WorkerResources, celery_app, resources
from tts_worker.engine_client import EngineClient
from tts_worker.pipeline import run_tts_stage
from tts_worker.settings import TtsWorkerSettings

log = get_logger(__name__)


@celery_app.task(
    bind=True,
    name="tts_worker.tasks.synthesize",
    max_retries=None,  # the retry budget is enforced explicitly below
)
def synthesize(self: Task, job_id: str, trace_id: str | None = None) -> str:
    """Render, assemble and store the audio for one job."""
    resource = resources()
    job_uuid = UUID(job_id)

    bind_job_id(job_uuid)
    if trace_id:
        # Continues the trace that started at POST /v1/jobs and passed through the story
        # worker, so one story is followable end to end.
        bind_trace_id(trace_id)

    owner = f"{os.getpid()}:{self.request.id}"
    with resource.lease.acquire(job_uuid, owner=owner) as acquired:
        if not acquired:
            # Another worker holds this job. Status transitions keep state correct
            # regardless; dropping the duplicate stops two GPUs rendering one story.
            log.info("tts_task_duplicate_delivery", job_id=job_id)
            return "duplicate"

        return _run(self, resource, resource.settings, job_uuid)


def _run(
    task: Task,
    resource: WorkerResources,
    settings: TtsWorkerSettings,
    job_id: UUID,
) -> str:
    engine = EngineClient(
        settings.tts_engine_address,
        timeout_seconds=settings.tts_request_timeout_seconds,
        capacity_retries=settings.tts_capacity_retries,
        capacity_backoff_seconds=settings.tts_capacity_backoff_seconds,
    )
    try:
        outcome = run_tts_stage(
            resource.session_factory,
            resource.publisher,
            engine,
            resource.storage,
            settings,
            job_id,
        )
    except AppError as exc:
        return _handle_failure(task, resource, settings, job_id, exc)
    except Exception as exc:  # noqa: BLE001 - classified below, never surfaced raw
        return _handle_failure(
            task,
            resource,
            settings,
            job_id,
            AppError(ErrorCode.INTERNAL, detail=f"{type(exc).__name__}: {exc}"),
        )
    finally:
        engine.close()

    return "done" if outcome.completed else "skipped"


def _handle_failure(
    task: Task,
    resource: WorkerResources,
    settings: TtsWorkerSettings,
    job_id: UUID,
    error: AppError,
) -> str:
    """Retry, or fail the job for good.

    A retry re-runs synthesis only. The story is durable at ``written``, so the model is
    never called again — which is the whole reason ``written`` is a distinct state.
    """
    if error.code is ErrorCode.CANCELLED:
        log.info("tts_task_cancelled", job_id=str(job_id))
        return "cancelled"

    attempt = task.request.retries
    if is_retryable(error.code) and attempt < settings.max_retries:
        delay = min(
            settings.retry_backoff_seconds * (2**attempt),
            settings.retry_backoff_max_seconds,
        )
        log.warning(
            "tts_task_retrying",
            job_id=str(job_id),
            code=error.code.value,
            attempt=attempt + 1,
            max_retries=settings.max_retries,
            delay_seconds=delay,
            detail=error.detail,
        )
        with session_scope(resource.session_factory) as session:
            session.execute(
                update(Job).where(Job.id == job_id).values(retry_count=Job.retry_count + 1)
            )
        raise task.retry(exc=error, countdown=delay)

    with session_scope(resource.session_factory) as session:
        fail_job(session, resource.publisher, job_id, code=error.code, detail=error.detail)
    return "failed"
