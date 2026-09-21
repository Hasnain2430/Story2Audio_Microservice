"""Celery tasks for the LLM stage.

Thin on purpose. Everything that decides what happens lives in `pipeline.py`; this module
owns only the concerns Celery brings with it — the lease, retry classification, terminal
failure handling, and handing off to the TTS queue.
"""

from __future__ import annotations

import os
from uuid import UUID

from celery import Task
from sqlalchemy import update

from story2audio_shared.errors import AppError, ErrorCode, is_retryable
from story2audio_shared.logging import bind_job_id, bind_trace_id, get_logger
from story2audio_shared.models import Job
from story2audio_shared.worker import fail_job, session_scope
from story_worker.app import (
    SYNTHESIZE_TASK,
    TTS_QUEUE,
    WorkerResources,
    celery_app,
    resources,
)
from story_worker.pipeline import run_story_stage
from story_worker.providers import build_provider
from story_worker.settings import StoryWorkerSettings

log = get_logger(__name__)


@celery_app.task(
    bind=True,
    name="story_worker.tasks.write_story",
    max_retries=None,  # the retry budget is enforced explicitly below
)
def write_story(self: Task, job_id: str, trace_id: str | None = None) -> str:
    """Generate the story for one job, then enqueue synthesis.

    Returns a short status string for the result backend and for logs; the durable
    outcome is the job row.
    """
    resource = resources()
    settings = resource.settings
    job_uuid = UUID(job_id)

    bind_job_id(job_uuid)
    if trace_id:
        # Continues the trace that started at POST /v1/jobs, so one story can be followed
        # from the browser through the queue into both workers.
        bind_trace_id(trace_id)

    owner = f"{os.getpid()}:{self.request.id}"
    with resource.lease.acquire(job_uuid, owner=owner) as acquired:
        if not acquired:
            # Another worker holds this job. Status transitions would keep state correct
            # regardless, but both workers would pay the model first. Dropping the
            # duplicate is cheaper than racing it.
            log.info("story_task_duplicate_delivery", job_id=job_id)
            return "duplicate"

        return _run(self, resource, settings, job_uuid, trace_id)


def _run(
    task: Task,
    resource: WorkerResources,
    settings: StoryWorkerSettings,
    job_id: UUID,
    trace_id: str | None,
) -> str:
    provider = None
    try:
        provider = build_provider(settings)
        outcome = run_story_stage(
            resource.session_factory, resource.publisher, provider, settings, job_id
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
        if provider is not None:
            provider.close()

    if not outcome.ready_for_synthesis:
        return "skipped"

    _enqueue_synthesis(job_id, trace_id)
    return "written" if outcome.performed_work else "already_written"


def _enqueue_synthesis(job_id: UUID, trace_id: str | None) -> None:
    """Hand the job to the TTS queue.

    Sent by task *name* rather than by importing the worker's function, so this service
    never has to depend on the TTS stack. The queues are separate deployments precisely
    so a GPU backlog cannot stall story generation.
    """
    celery_app.send_task(
        SYNTHESIZE_TASK,
        args=[str(job_id)],
        kwargs={"trace_id": trace_id},
        queue=TTS_QUEUE,
    )
    log.info("synthesis_enqueued", job_id=str(job_id))


def _handle_failure(
    task: Task,
    resource: WorkerResources,
    settings: StoryWorkerSettings,
    job_id: UUID,
    error: AppError,
) -> str:
    """Retry, or fail the job for good.

    Cancellation is not a failure: a job the user stopped is already terminal, and
    writing an error over it would replace something they asked for with something they
    did not.
    """
    if error.code is ErrorCode.CANCELLED:
        log.info("story_task_cancelled", job_id=str(job_id))
        return "cancelled"

    attempt = task.request.retries
    if is_retryable(error.code) and attempt < settings.max_retries:
        delay = min(
            settings.retry_backoff_seconds * (2**attempt),
            settings.retry_backoff_max_seconds,
        )
        log.warning(
            "story_task_retrying",
            job_id=str(job_id),
            code=error.code.value,
            attempt=attempt + 1,
            max_retries=settings.max_retries,
            delay_seconds=delay,
            detail=error.detail,
        )
        _record_attempt(resource, job_id)
        raise task.retry(exc=error, countdown=delay)

    with session_scope(resource.session_factory) as session:
        fail_job(session, resource.publisher, job_id, code=error.code, detail=error.detail)
    return "failed"


def _record_attempt(resource: WorkerResources, job_id: UUID) -> None:
    """Count the retry on the job row, so a flaky provider is visible in the data."""
    with session_scope(resource.session_factory) as session:
        session.execute(update(Job).where(Job.id == job_id).values(retry_count=Job.retry_count + 1))
