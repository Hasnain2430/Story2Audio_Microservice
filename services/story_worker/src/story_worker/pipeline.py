"""The LLM stage, independent of Celery.

Kept free of task decorators, retries and broker concerns so it can be driven directly
in tests with a scripted provider. `tasks.py` is the thin Celery wrapper around it.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from uuid import UUID

from sqlalchemy.orm import Session, sessionmaker

from story2audio_shared.enums import Emotion, JobStatus, Language, StoryLength, VoiceMode
from story2audio_shared.errors import AppError, ErrorCode
from story2audio_shared.events import JobEvent, StatusEvent, StoryDoneEvent, TokenEvent
from story2audio_shared.logging import get_logger
from story2audio_shared.models import Job
from story2audio_shared.prompts import (
    StoryPrompt,
    build_continuation_prompt,
    build_story_prompt,
    looks_complete,
)
from story2audio_shared.worker import (
    SyncEventPublisher,
    TokenBatcher,
    advance_status,
    current_status,
    is_cancelled,
    load_job,
    session_scope,
)
from story_worker.providers import LLMProvider, StreamStats
from story_worker.settings import StoryWorkerSettings

log = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class StageOutcome:
    """What the stage did, so the caller knows whether to enqueue synthesis."""

    #: True when the job is at `written` and synthesis should be enqueued. Also true for
    #: a redelivery of an already-written job, which is what makes the task idempotent.
    ready_for_synthesis: bool
    #: False when the job was cancelled or had already finished.
    performed_work: bool = False


def run_story_stage(
    session_factory: sessionmaker[Session],
    publisher: SyncEventPublisher,
    provider: LLMProvider,
    settings: StoryWorkerSettings,
    job_id: UUID,
) -> StageOutcome:
    """Write the story for one job.

    Idempotent against its own row, because Celery delivers at least once:

    - already ``written`` — a previous delivery finished the work; enqueue synthesis and
      return without calling the model again.
    - already terminal — cancelled or failed; do nothing.
    - ``writing`` — a retry or a redelivery of an interrupted attempt; resume.
    - ``queued`` — the normal path.
    """
    with session_scope(session_factory) as session:
        job = load_job(session, job_id)
        if job is None:
            log.warning("story_stage_job_missing", job_id=str(job_id))
            return StageOutcome(ready_for_synthesis=False)

        decision = _decide_entry(job)
        if decision is not None:
            return decision

        request = _snapshot_request(job)

    if not _begin(session_factory, publisher, job_id):
        return StageOutcome(ready_for_synthesis=False)

    story, stats = _generate(session_factory, publisher, provider, settings, job_id, request)

    with session_scope(session_factory) as session:
        _finish(session, publisher, job_id, story=story, stats=stats)

    return StageOutcome(ready_for_synthesis=True, performed_work=True)


# --- Entry decisions --------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _Request:
    """The generation parameters, read once so the session need not stay open."""

    prompt: str
    length: StoryLength
    mode: VoiceMode
    language: Language
    emotion: Emotion


def _snapshot_request(job: Job) -> _Request:
    return _Request(
        prompt=job.prompt,
        length=job.length,
        mode=job.mode,
        language=job.language,
        emotion=job.emotion,
    )


def _decide_entry(job: Job) -> StageOutcome | None:
    """Return an outcome when the job should not be generated, or ``None`` to proceed."""
    if job.status is JobStatus.WRITTEN:
        # A redelivery after the story was already written. Re-running the model here
        # would pay twice for text we already have.
        log.info("story_stage_already_written", job_id=str(job.id))
        return StageOutcome(ready_for_synthesis=True)

    if job.status in {JobStatus.DONE, JobStatus.FAILED, JobStatus.CANCELLED}:
        log.info("story_stage_skipped_terminal", job_id=str(job.id), status=job.status.value)
        return StageOutcome(ready_for_synthesis=False)

    if job.status is JobStatus.SYNTHESIZING:
        # The TTS stage is already running; nothing for this stage to do, and enqueueing
        # again would duplicate synthesis.
        log.info("story_stage_already_synthesizing", job_id=str(job.id))
        return StageOutcome(ready_for_synthesis=False)

    return None


def _begin(
    session_factory: sessionmaker[Session], publisher: SyncEventPublisher, job_id: UUID
) -> bool:
    """Move the job into ``writing``, tolerating a resumed attempt.

    Returns ``False`` when the job was cancelled between the read and the write, which is
    the signal to stop.
    """
    with session_scope(session_factory) as session:
        status = current_status(session, job_id)
        if status is JobStatus.WRITING:
            # A retry, or a redelivery of an attempt that died mid-stream. Resume without
            # a transition: the job is already in the right state.
            log.info("story_stage_resuming", job_id=str(job_id))
            return True

        if status is not JobStatus.QUEUED:
            log.info("story_stage_unexpected_status", job_id=str(job_id), status=str(status))
            return False

        result = advance_status(
            session,
            job_id,
            expected=JobStatus.QUEUED,
            target=JobStatus.WRITING,
            writing_at=datetime.now(UTC),
        )
        if not result.applied:
            log.info("story_stage_lost_transition", job_id=str(job_id), now=str(result.previous))
            return False

        def build(seq: int, at: datetime) -> JobEvent:
            return StatusEvent(job_id=job_id, seq=seq, at=at, status=JobStatus.WRITING)

        publisher.publish(session, job_id, build)
    return True


# --- Generation ----------------------------------------------------------------------------


def _generate(
    session_factory: sessionmaker[Session],
    publisher: SyncEventPublisher,
    provider: LLMProvider,
    settings: StoryWorkerSettings,
    job_id: UUID,
    request: _Request,
) -> tuple[str, StreamStats]:
    """Stream the story, publishing text as it arrives.

    The user is reading within a couple of seconds while the GPU work is still ahead of
    them — the perceived-latency win over v1, which showed a spinner for the whole run.
    """
    prompt = build_story_prompt(
        request.prompt,
        length=request.length,
        mode=request.mode,
        language=request.language,
        emotion=request.emotion,
    )
    stats = StreamStats()
    story = _stream_into_events(
        session_factory, publisher, provider, settings, job_id, prompt, stats
    )

    story = _maybe_continue(
        session_factory, publisher, provider, settings, job_id, request, story, stats
    )

    story = story.strip()
    if not story:
        raise AppError(ErrorCode.STORY_EMPTY, detail="provider returned no text")
    return story, stats


def _stream_into_events(
    session_factory: sessionmaker[Session],
    publisher: SyncEventPublisher,
    provider: LLMProvider,
    settings: StoryWorkerSettings,
    job_id: UUID,
    prompt: StoryPrompt,
    stats: StreamStats,
) -> str:
    """Consume one generation, batching frames and honouring cancellation."""
    batcher = TokenBatcher(
        interval_seconds=settings.token_batch_interval_seconds,
        max_chars=settings.token_batch_max_chars,
    )
    pieces: list[str] = []
    frames = 0

    for chunk in provider.stream(prompt, stats):
        pieces.append(chunk)
        frame = batcher.add(chunk)
        if frame is None:
            continue

        frames += 1
        _publish_tokens(session_factory, publisher, job_id, frame)

        # Checked every N frames rather than per token: often enough to stop promptly,
        # rare enough not to query the database for every word.
        if frames % settings.cancel_check_every_frames == 0:
            with session_scope(session_factory) as session:
                if is_cancelled(session, job_id):
                    raise AppError(ErrorCode.CANCELLED, detail="cancelled mid-generation")

    if tail := batcher.flush():
        _publish_tokens(session_factory, publisher, job_id, tail)

    return "".join(pieces)


def _publish_tokens(
    session_factory: sessionmaker[Session],
    publisher: SyncEventPublisher,
    job_id: UUID,
    text: str,
) -> None:
    with session_scope(session_factory) as session:

        def build(seq: int, at: datetime) -> JobEvent:
            return TokenEvent(job_id=job_id, seq=seq, at=at, text=text)

        publisher.publish(session, job_id, build)


def _maybe_continue(
    session_factory: sessionmaker[Session],
    publisher: SyncEventPublisher,
    provider: LLMProvider,
    settings: StoryWorkerSettings,
    job_id: UUID,
    request: _Request,
    story: str,
    stats: StreamStats,
) -> str:
    """Finish a story that stopped mid-sentence.

    v1's flat 2000-token cap truncated its own 800-1200 word target, and the prompt's
    plea to "MAKE SURE THE STORY HAS A PROPER END" could not fix a hard limit. Budgets
    are now sized per length, and this is the backstop for when a model trails off anyway.

    Bounded to at most ``max_continuation_passes``: a model that never lands an ending
    would otherwise be an open-ended bill.
    """
    for attempt in range(settings.max_continuation_passes):
        if looks_complete(story):
            return story

        log.info(
            "story_continuation_pass",
            job_id=str(job_id),
            attempt=attempt + 1,
            finish_reason=stats.finish_reason,
            words=len(story.split()),
        )
        continuation_prompt = build_continuation_prompt(
            story, length=request.length, language=request.language
        )
        continuation = _stream_into_events(
            session_factory, publisher, provider, settings, job_id, continuation_prompt, stats
        )
        if not continuation.strip():
            break
        story = f"{story.rstrip()} {continuation.lstrip()}"

    return story


# --- Completion ---------------------------------------------------------------------------------


def _finish(
    session: Session,
    publisher: SyncEventPublisher,
    job_id: UUID,
    *,
    story: str,
    stats: StreamStats,
) -> None:
    """Persist the story and hand off to synthesis.

    ``written`` is the retry boundary: the story is durable here, so a later TTS failure
    resumes from this point and never re-runs the model.
    """
    result = advance_status(
        session,
        job_id,
        expected=JobStatus.WRITING,
        target=JobStatus.WRITTEN,
        story_text=story,
        llm_model=stats.model,
        llm_output_tokens=stats.output_tokens,
        written_at=datetime.now(UTC),
    )
    if not result.applied:
        # Cancelled while the last frames were in flight. The story is discarded rather
        # than forced onto a job the user already abandoned.
        raise AppError(
            ErrorCode.CANCELLED,
            detail=f"job left `writing` during generation (now {result.previous})",
        )

    word_count = len(story.split())

    def build(seq: int, at: datetime) -> JobEvent:
        return StoryDoneEvent(job_id=job_id, seq=seq, at=at, text=story, word_count=word_count)

    publisher.publish(session, job_id, build)
    log.info(
        "story_written",
        job_id=str(job_id),
        words=word_count,
        model=stats.model,
        output_tokens=stats.output_tokens,
    )
