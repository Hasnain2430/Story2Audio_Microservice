"""Job creation, retrieval and cancellation.

The rule that shapes all of it: nothing in this module may take longer than a database
round trip. Work happens on the other side of the queue.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime
from uuid import UUID, uuid4

from pydantic import ValidationError
from sqlalchemy import Select, func, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.publisher import EventPublisher
from gateway.ratelimit import RateLimiter
from story2audio_shared.config import LimitSettings
from story2audio_shared.enums import (
    CANCELLABLE_STATUSES,
    TERMINAL_STATUSES,
    AudioFormat,
    JobStatus,
    VoiceMode,
)
from story2audio_shared.errors import AppError, ErrorCode, spec_for
from story2audio_shared.events import CancelledEvent, JobEvent, StatusEvent
from story2audio_shared.ids import uuid7
from story2audio_shared.logging import get_logger
from story2audio_shared.models import Job, Voice
from story2audio_shared.schemas import (
    AudioAsset,
    CreateJobRequest,
    ErrorDetail,
    JobResponse,
    JobTimings,
    SpokenSegment,
)
from story2audio_shared.storage import ObjectStorage

#: Hourly job-creation window.
_JOBS_WINDOW_SECONDS = 3600

log = get_logger(__name__)


async def create_job(
    session: AsyncSession,
    *,
    owner_id: UUID,
    request: CreateJobRequest,
    limits: LimitSettings,
    rate_limiter: RateLimiter,
    idempotency_key: str | None,
) -> tuple[Job, bool]:
    """Validate, admit and persist a job.

    Returns the job and whether it was newly created — ``False`` means an idempotency key
    matched an existing one, so the caller gets the original rather than a second run of
    the same GPU work.

    Guardrails are applied cheapest-first, and the expensive global cap is consumed last
    so that a request rejected for a local reason never burns the day's budget.
    """
    if idempotency_key is not None:
        existing = await _find_by_idempotency_key(session, owner_id, idempotency_key)
        if existing is not None:
            return existing, False

    await _assert_prompt_within_deployment_limit(request, limits)
    voice, dialogue_voice = await _resolve_voices(session, owner_id, request)
    await _assert_concurrency_available(session, owner_id, limits)
    await _assert_hourly_quota_available(rate_limiter, owner_id, limits)

    cap = await rate_limiter.check_daily_cap(cap=limits.global_daily_job_cap)
    if not cap.allowed:
        raise AppError(
            ErrorCode.DAILY_CAP_REACHED,
            detail=f"global daily cap of {cap.limit} reached",
        )

    job = Job(
        id=uuid7(),
        owner_id=owner_id,
        status=JobStatus.QUEUED,
        prompt=request.prompt,
        length=request.length,
        mode=request.mode,
        language=request.language,
        emotion=request.emotion,
        speed=request.speed,
        voice_id=voice.id,
        dialogue_voice_id=dialogue_voice.id if dialogue_voice is not None else None,
        idempotency_key=idempotency_key,
        queued_at=datetime.now(UTC),
    )
    session.add(job)

    try:
        await session.flush()
    except IntegrityError:
        # Two requests with the same idempotency key raced past the lookup above. The
        # unique index is the real arbiter; the loser reads the winner's row.
        await session.rollback()
        await rate_limiter.release_daily_cap()
        if idempotency_key is not None:
            existing = await _find_by_idempotency_key(session, owner_id, idempotency_key)
            if existing is not None:
                return existing, False
        raise

    return job, True


async def get_job(session: AsyncSession, *, owner_id: UUID, job_id: UUID) -> Job:
    """Load a job the caller owns.

    A job belonging to someone else is reported as not found rather than forbidden: a 403
    would confirm the id exists, which is more than a stranger needs to know.
    """
    job = await session.scalar(select(Job).where(Job.id == job_id, Job.owner_id == owner_id))
    if job is None:
        raise AppError(ErrorCode.JOB_NOT_FOUND, detail=f"job {job_id} not visible to {owner_id}")
    return job


async def list_jobs(
    session: AsyncSession, *, owner_id: UUID, limit: int, cursor: UUID | None
) -> tuple[Sequence[Job], bool]:
    """Page through the caller's jobs, newest first.

    The cursor is a job id. Because ids are UUIDv7 and therefore time-ordered (ADR-0002),
    ``id < cursor`` is both the filter and the ordering — no ``created_at`` tiebreak, and
    no offset to skip or duplicate rows as new jobs arrive mid-scroll.
    """
    query: Select[tuple[Job]] = select(Job).where(Job.owner_id == owner_id)
    if cursor is not None:
        query = query.where(Job.id < cursor)

    # One extra row tells us whether another page exists without a second COUNT query.
    rows = (await session.scalars(query.order_by(Job.id.desc()).limit(limit + 1))).all()
    has_more = len(rows) > limit
    return rows[:limit], has_more


async def cancel_job(
    session: AsyncSession,
    publisher: EventPublisher,
    *,
    owner_id: UUID,
    job_id: UUID,
) -> Job:
    """Request cancellation.

    The transition is a conditional UPDATE asserting the job is still cancellable, so two
    concurrent cancels cannot both succeed and a job that finished microseconds ago is not
    dragged back out of its terminal state. Workers check for cancellation between
    segments, which is what makes this actually free the GPU rather than just relabel the
    row.
    """
    job = await get_job(session, owner_id=owner_id, job_id=job_id)
    if job.status in TERMINAL_STATUSES:
        raise AppError(ErrorCode.JOB_NOT_CANCELLABLE, detail=f"job is already {job.status.value}")

    now = datetime.now(UTC)
    result = await session.execute(
        update(Job)
        .where(Job.id == job_id, Job.status.in_(tuple(CANCELLABLE_STATUSES)))
        .values(status=JobStatus.CANCELLED, finished_at=now)
        .returning(Job.id)
    )
    if result.scalar_one_or_none() is None:
        raise AppError(ErrorCode.JOB_NOT_CANCELLABLE, detail="job reached a terminal state first")

    def build(seq: int, at: datetime) -> JobEvent:
        return CancelledEvent(job_id=job_id, seq=seq, at=at)

    await publisher.publish(session, job_id, build)
    await session.refresh(job)
    return job


def snapshot_event(job: Job) -> StatusEvent:
    """The first frame sent on a WebSocket connection.

    Built from the database rather than from Redis, so a client that connects late or
    reconnects after a gap starts from truth. It reuses the job's current sequence number
    instead of allocating a new one, and is flagged as a snapshot so the client does not
    mistake the repeat for a dropped frame (ADR-0003).
    """
    return StatusEvent(
        job_id=job.id,
        seq=max(1, job.last_event_seq),
        at=datetime.now(UTC),
        status=job.status,
        snapshot=True,
    )


def to_response(job: Job, storage: ObjectStorage, *, ttl_seconds: int) -> JobResponse:
    """Render a job for the API, presigning any audio it has produced."""
    audio: list[AudioAsset] = []
    for audio_format, key, size in (
        (AudioFormat.MP3, job.audio_key_mp3, job.audio_bytes_mp3),
        (AudioFormat.WAV, job.audio_key_wav, job.audio_bytes_wav),
    ):
        if key is None:
            continue
        presigned = storage.presign_get(key, ttl_seconds=ttl_seconds)
        audio.append(
            AudioAsset(
                format=audio_format,
                url=presigned.url,
                duration_seconds=job.audio_duration_seconds or 0.0,
                size_bytes=size or 0,
                expires_at=presigned.expires_at,
            )
        )

    error: ErrorDetail | None = None
    if job.error_code is not None:
        spec = spec_for(job.error_code)
        error = ErrorDetail(code=job.error_code, message=spec.message, retryable=spec.retryable)

    return JobResponse(
        id=job.id,
        status=job.status,
        prompt=job.prompt,
        length=job.length,
        mode=job.mode,
        language=job.language,
        emotion=job.emotion,
        speed=job.speed,
        voice_id=job.voice_id,
        dialogue_voice_id=job.dialogue_voice_id,
        story_text=job.story_text,
        audio=audio,
        segment_count=job.segment_count,
        segments=_spoken_segments(job),
        error=error,
        timings=JobTimings(
            queued_at=job.queued_at,
            writing_at=job.writing_at,
            written_at=job.written_at,
            synthesizing_at=job.synthesizing_at,
            finished_at=job.finished_at,
        ),
        created_at=job.created_at,
        updated_at=job.updated_at,
    )


def _spoken_segments(job: Job) -> list[SpokenSegment]:
    """Read the worker's timeline, tolerating its absence and its age.

    The column is JSON written by another service, so it is validated rather than
    trusted: a row from before the timeline existed is null, and a row written by an
    older worker may lack fields this version expects. Either way the job is still
    complete — the timeline only drives a playback nicety — so a bad entry is dropped
    and the rest are served, never raised at a caller asking for their story.
    """
    raw = job.segment_timeline
    if not raw:
        return []

    segments: list[SpokenSegment] = []
    for entry in raw:
        try:
            segments.append(SpokenSegment.model_validate(entry))
        except ValidationError:
            log.warning("job_segment_timeline_unreadable", job_id=str(job.id))
            return []
    return segments


# --- Guardrails -------------------------------------------------------------------------


async def _find_by_idempotency_key(session: AsyncSession, owner_id: UUID, key: str) -> Job | None:
    job: Job | None = await session.scalar(
        select(Job).where(Job.owner_id == owner_id, Job.idempotency_key == key)
    )
    return job


async def _assert_prompt_within_deployment_limit(
    request: CreateJobRequest, limits: LimitSettings
) -> None:
    """Apply the deployment's prompt ceiling.

    The schema enforces an absolute maximum that no deployment may exceed; this is the
    per-deployment tightening on top of it.
    """
    if len(request.prompt) > limits.max_prompt_chars:
        raise AppError(
            ErrorCode.PROMPT_TOO_LONG,
            detail=f"{len(request.prompt)} chars exceeds limit {limits.max_prompt_chars}",
        )


async def _resolve_voices(
    session: AsyncSession, owner_id: UUID, request: CreateJobRequest
) -> tuple[Voice, Voice | None]:
    """Resolve voice ids to rows the caller is allowed to use.

    This is the replacement for v1's ``speaker_audio`` string: the client names a voice,
    and the server decides what file that means. A client-supplied path never reaches the
    TTS engine.
    """
    narrator = await _load_voice(session, owner_id, request.voice_id)
    if request.mode is not VoiceMode.NARRATION_WITH_DIALOGUE:
        return narrator, None

    if request.dialogue_voice_id is None:  # pragma: no cover - schema already rejects this
        raise AppError(ErrorCode.DIALOGUE_VOICE_REQUIRED)
    dialogue = await _load_voice(session, owner_id, request.dialogue_voice_id)
    return narrator, dialogue


async def _load_voice(session: AsyncSession, owner_id: UUID, voice_id: UUID) -> Voice:
    voice = await session.scalar(select(Voice).where(Voice.id == voice_id))
    if voice is None:
        raise AppError(ErrorCode.VOICE_NOT_FOUND, detail=f"voice {voice_id} does not exist")
    if not voice.is_builtin and voice.owner_id != owner_id:
        raise AppError(
            ErrorCode.VOICE_FORBIDDEN, detail=f"voice {voice_id} belongs to another user"
        )
    return voice


async def _assert_concurrency_available(
    session: AsyncSession, owner_id: UUID, limits: LimitSettings
) -> None:
    """Cap how many of the caller's jobs may be in flight at once.

    Counted from the rows, in the same transaction as the insert, rather than from a Redis
    counter. A counter drifts permanently the first time a worker dies without
    decrementing it; a count of non-terminal rows is always correct by construction.
    """
    in_flight = await session.scalar(
        select(func.count())
        .select_from(Job)
        .where(Job.owner_id == owner_id, Job.status.notin_(tuple(TERMINAL_STATUSES)))
    )
    if (in_flight or 0) >= limits.max_concurrent_jobs_per_session:
        raise AppError(
            ErrorCode.CONCURRENCY_LIMIT_REACHED,
            detail=f"{in_flight} jobs already in flight for {owner_id}",
        )


async def _assert_hourly_quota_available(
    rate_limiter: RateLimiter, owner_id: UUID, limits: LimitSettings
) -> None:
    decision = await rate_limiter.check_sliding_window(
        bucket="jobs",
        subject=owner_id,
        limit=limits.rate_limit_jobs_per_hour,
        window_seconds=_JOBS_WINDOW_SECONDS,
        token=uuid4().hex,
    )
    if not decision.allowed:
        raise AppError(
            ErrorCode.RATE_LIMITED,
            detail=f"{decision.used}/{decision.limit} jobs in the last hour",
        )
