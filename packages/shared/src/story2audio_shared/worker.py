"""Shared worker toolkit.

Both Celery workers need the same three things, and all three are safety-critical under
at-least-once delivery: a synchronous database session, atomic event-sequence allocation,
and a status transition that cannot be applied twice or out of order. Two copies of that
logic would eventually diverge, and the symptom would be corrupted job state under
redelivery — so it lives here and is imported, not reimplemented (ADR-0001).

Everything here is synchronous. Celery runs a pool of synchronous worker processes and
these tasks are minutes long, so concurrency comes from processes rather than from an
event loop.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any
from uuid import UUID

from redis import Redis
from sqlalchemy import Engine, create_engine, select, update
from sqlalchemy.orm import Session, sessionmaker

from story2audio_shared.config import DatabaseSettings, RedisSettings, database_settings
from story2audio_shared.enums import TERMINAL_STATUSES, JobStatus, can_transition
from story2audio_shared.errors import AppError, ErrorCode, spec_for
from story2audio_shared.events import FailedEvent, JobEvent, job_channel
from story2audio_shared.logging import get_logger
from story2audio_shared.models import Job

log = get_logger(__name__)

#: Builds an event once its sequence number and timestamp are known.
EventFactory = Callable[[int, datetime], JobEvent]


# --- Database ---------------------------------------------------------------------------


def create_worker_engine(settings: DatabaseSettings | None = None) -> Engine:
    """Synchronous engine for a Celery worker.

    ``pool_pre_ping`` because managed Postgres drops idle connections and a worker can sit
    idle between jobs; without it the first task after a quiet period fails on a dead
    socket. The pool is small on purpose — a worker process handles one job at a time.
    """
    config = settings or database_settings()
    return create_engine(
        config.database_url_sync,
        echo=config.db_echo,
        pool_size=2,
        max_overflow=2,
        pool_pre_ping=True,
    )


def create_worker_session_factory(engine: Engine) -> sessionmaker[Session]:
    return sessionmaker(engine, expire_on_commit=False, autoflush=False)


@contextmanager
def session_scope(factory: sessionmaker[Session]) -> Iterator[Session]:
    """Run a unit of work: commit on success, roll back on any exception."""
    session = factory()
    try:
        yield session
    except Exception:
        session.rollback()
        raise
    else:
        session.commit()
    finally:
        session.close()


# --- Events -------------------------------------------------------------------------------


class SyncEventPublisher:
    """Allocates sequence numbers and publishes events over Redis pub/sub.

    Publishing is best-effort. Redis pub/sub guarantees nothing, and the durable record is
    the ``jobs`` row — so a failed publish must never fail the work that produced it. The
    client sees the change on its next poll or on reconnect.
    """

    def __init__(self, redis: Redis) -> None:
        self._redis = redis

    def next_seq(self, session: Session, job_id: UUID) -> int:
        """Reserve the next sequence number for a job.

        One atomic statement rather than read-modify-write, so the gateway and a worker
        publishing concurrently cannot take the same number (ADR-0003).
        """
        result = session.execute(
            update(Job)
            .where(Job.id == job_id)
            .values(last_event_seq=Job.last_event_seq + 1)
            .returning(Job.last_event_seq)
        )
        return int(result.scalar_one())

    def publish(self, session: Session, job_id: UUID, factory: EventFactory) -> None:
        """Allocate a sequence number, build the event, and publish it."""
        seq = self.next_seq(session, job_id)
        event = factory(seq, datetime.now(UTC))
        try:
            self._redis.publish(job_channel(job_id), event.model_dump_json())
        except Exception as exc:  # noqa: BLE001 - deliberately swallowed, see class docstring
            log.warning(
                "event_publish_failed",
                job_id=str(job_id),
                event_type=event.type,
                error=str(exc),
            )


class TokenBatcher:
    """Coalesces streamed text into periodic frames.

    One WebSocket frame per token would flood the socket without the text arriving any
    sooner — the bottleneck is the model, not the transport. Flushing on a short interval
    or a size threshold keeps the prose visibly streaming while cutting frame count by
    one to two orders of magnitude.
    """

    def __init__(self, *, interval_seconds: float = 0.05, max_chars: int = 400) -> None:
        self._interval = interval_seconds
        self._max_chars = max_chars
        self._buffer: list[str] = []
        self._length = 0
        self._last_flush = time.monotonic()

    def add(self, text: str) -> str | None:
        """Buffer a chunk, returning a frame's worth of text when one is due."""
        if not text:
            return None
        self._buffer.append(text)
        self._length += len(text)

        due = self._length >= self._max_chars
        elapsed = time.monotonic() - self._last_flush >= self._interval
        if due or elapsed:
            return self.flush()
        return None

    def flush(self) -> str | None:
        """Return whatever is buffered, if anything."""
        if not self._buffer:
            return None
        text = "".join(self._buffer)
        self._buffer.clear()
        self._length = 0
        self._last_flush = time.monotonic()
        return text


# --- Job state ------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class TransitionResult:
    """Outcome of an attempted status change."""

    applied: bool
    #: The status observed before the attempt, or ``None`` if the job is gone.
    previous: JobStatus | None


def load_job(session: Session, job_id: UUID) -> Job | None:
    job: Job | None = session.scalar(select(Job).where(Job.id == job_id))
    return job


def current_status(session: Session, job_id: UUID) -> JobStatus | None:
    status: JobStatus | None = session.scalar(select(Job.status).where(Job.id == job_id))
    return status


def is_cancelled(session: Session, job_id: UUID) -> bool:
    """Has the job reached a terminal state underneath us?

    Checked between steps so that a cancellation actually stops work in progress rather
    than only relabelling the row. In the TTS stage this is the difference between
    releasing the GPU and not.
    """
    status = current_status(session, job_id)
    return status is None or status in TERMINAL_STATUSES


def advance_status(
    session: Session,
    job_id: UUID,
    *,
    expected: JobStatus,
    target: JobStatus,
    **values: Any,
) -> TransitionResult:
    """Move a job from ``expected`` to ``target``, atomically.

    The ``WHERE`` clause asserts the previous status, so this is a compare-and-set rather
    than a blind write. That is what makes the pipeline safe under Celery's at-least-once
    delivery: a duplicate task cannot move a job backwards, re-run a completed stage, or
    drag a cancelled job back into flight. A ``False`` result is the caller's signal to
    stop, not an error.
    """
    if not can_transition(expected, target):
        raise ValueError(f"illegal transition {expected.value} -> {target.value}")

    result = session.execute(
        update(Job)
        .where(Job.id == job_id, Job.status == expected)
        .values(status=target, **values)
        .returning(Job.status)
    )
    if result.scalar_one_or_none() is not None:
        return TransitionResult(applied=True, previous=expected)

    # Did not apply: report what the status actually is, so the caller can tell a
    # cancellation from a duplicate delivery.
    return TransitionResult(applied=False, previous=current_status(session, job_id))


def fail_job(
    session: Session,
    publisher: SyncEventPublisher,
    job_id: UUID,
    *,
    code: ErrorCode,
    detail: str | None = None,
) -> None:
    """Mark a job failed and announce it.

    ``detail`` is logged, never persisted to a field the API returns: the row carries the
    classified code and its fixed public message. v1 handed the caller ``str(exception)``.
    """
    spec = spec_for(code)
    status = current_status(session, job_id)
    if status is None or status in TERMINAL_STATUSES:
        # Already finished, or cancelled while we were failing. Leave it alone: a
        # cancellation the user asked for should not be overwritten by a late error.
        log.info("fail_skipped_terminal", job_id=str(job_id), code=code.value)
        return

    session.execute(
        update(Job)
        .where(Job.id == job_id, Job.status == status)
        .values(
            status=JobStatus.FAILED,
            error_code=code,
            error_message=spec.message,
            finished_at=datetime.now(UTC),
        )
    )

    def build(seq: int, at: datetime) -> JobEvent:
        return FailedEvent(
            job_id=job_id, seq=seq, at=at, code=code, message=spec.message, retryable=spec.retryable
        )

    publisher.publish(session, job_id, build)
    log.warning("job_failed", job_id=str(job_id), code=code.value, detail=detail)


# --- Concurrency lease --------------------------------------------------------------------------


class JobLease:
    """Advisory lock preventing two workers from processing one job at once.

    Status transitions are the real arbiter — a duplicate worker cannot corrupt state
    even without this. What the lease prevents is *paying twice*: two concurrent
    deliveries would both call the model, and only the loser's work would be discarded.

    Deliberately advisory. It expires, and an expired lease must never be treated as
    permission to skip the conditional transition.
    """

    def __init__(self, redis: Redis, *, ttl_seconds: int) -> None:
        self._redis = redis
        self._ttl = ttl_seconds

    @contextmanager
    def acquire(self, job_id: UUID, *, owner: str) -> Iterator[bool]:
        """Try to take the lease, releasing it on the way out.

        Yields whether it was acquired. Release is conditional on still owning it, so a
        worker whose lease expired and was taken by someone else cannot delete theirs.
        """
        key = f"lease:job:{job_id}"
        acquired = bool(self._redis.set(key, owner, nx=True, ex=self._ttl))
        try:
            yield acquired
        finally:
            if acquired:
                try:
                    if self._redis.get(key) == owner:
                        self._redis.delete(key)
                except Exception as exc:  # noqa: BLE001 - expiry is the fallback
                    log.warning("lease_release_failed", job_id=str(job_id), error=str(exc))


def create_worker_redis(settings: RedisSettings) -> Redis:
    """Synchronous Redis client for a worker process."""
    return Redis.from_url(settings.redis_url, decode_responses=True, health_check_interval=30)


def raise_for_terminal(session: Session, job_id: UUID) -> None:
    """Abort the current stage if the job has already finished.

    Raises :class:`AppError` with ``CANCELLED`` so the caller's normal error path unwinds
    cleanly rather than needing a separate control flow for cancellation.
    """
    if is_cancelled(session, job_id):
        raise AppError(ErrorCode.CANCELLED, detail=f"job {job_id} reached a terminal state")
