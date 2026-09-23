"""Handing an accepted job to a worker.

The gateway's contract is that ``POST /v1/jobs`` returns in well under a second. What
happens after that is the dispatcher's problem, and there are two implementations:

``CeleryDispatcher``
    The real path. Puts a message on the ``story`` queue and returns. Wired up in Phase 3
    when ``story-worker`` exists.

``InlineDispatcher``
    Runs a canned pipeline in-process, emitting the same event sequence a real run would:
    status transitions, streamed tokens, segment progress, then a terminal event. This is
    what makes the entire API — including the WebSocket, the state machine and the
    reconnect path — testable before either worker exists, and it stays useful afterwards
    as a deterministic test double.

The inline dispatcher is refused in production: it would silently return fake audio.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import Protocol
from uuid import UUID

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from gateway.publisher import EventFactory, EventPublisher
from story2audio_shared.enums import TERMINAL_STATUSES, JobStatus, can_transition
from story2audio_shared.events import (
    DoneEvent,
    JobEvent,
    ProgressEvent,
    StatusEvent,
    StoryDoneEvent,
    TokenEvent,
)
from story2audio_shared.logging import get_logger
from story2audio_shared.models import Job

log = get_logger(__name__)

#: Placeholder prose for the inline dispatcher. Obviously synthetic so that a stub result
#: reaching a real environment is unmistakable rather than plausible.
_CANNED_STORY = (
    "This is placeholder narration produced by the inline dispatcher. "
    "No language model was called and no audio was synthesized. "
    "It exists so the job pipeline, the event stream and the client can be exercised "
    "end to end before the workers are connected."
)


class JobDispatcher(Protocol):
    """Hands an accepted job to whatever will actually do the work."""

    async def dispatch(self, job_id: UUID) -> None: ...

    async def shutdown(self) -> None: ...


class CeleryDispatcher:
    """Enqueues onto the Celery ``story`` queue.

    Implemented against the task *name* rather than an imported function, so the gateway
    never has to import worker code — which would drag torch and the TTS stack into the
    API image.
    """

    STORY_TASK_NAME = "story_worker.tasks.write_story"
    STORY_QUEUE = "story"

    def __init__(self, broker_url: str) -> None:
        from celery import Celery

        self._app = Celery(broker=broker_url)

    async def dispatch(self, job_id: UUID) -> None:
        # Kombu's publish is synchronous and does network I/O, so it goes to a thread
        # rather than stalling the event loop that is serving every other request.
        await asyncio.to_thread(
            self._app.send_task,
            self.STORY_TASK_NAME,
            args=[str(job_id)],
            queue=self.STORY_QUEUE,
        )

    async def shutdown(self) -> None:
        await asyncio.to_thread(self._app.close)


class InlineDispatcher:
    """Runs a canned pipeline in-process. Development and tests only."""

    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession],
        publisher: EventPublisher,
        *,
        step_delay_seconds: float = 0.0,
        segment_count: int = 4,
    ) -> None:
        self._session_factory = session_factory
        self._publisher = publisher
        self._delay = step_delay_seconds
        self._segments = segment_count
        self._tasks: set[asyncio.Task[None]] = set()

    async def dispatch(self, job_id: UUID) -> None:
        task = asyncio.create_task(self._run(job_id))
        # Hold a reference: asyncio only keeps a weak one, so an unreferenced task can be
        # garbage-collected mid-flight.
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def shutdown(self) -> None:
        """Cancel in-flight pipelines and wait for them to unwind."""
        for task in list(self._tasks):
            task.cancel()
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)

    async def _run(self, job_id: UUID) -> None:
        try:
            await self._pipeline(job_id)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - a background stub must never take the app down
            log.warning("inline_dispatch_failed", job_id=str(job_id), error=str(exc))

    async def _pipeline(self, job_id: UUID) -> None:
        if not await self._advance(job_id, JobStatus.WRITING, writing_at=datetime.now(UTC)):
            return

        for chunk in _chunks(_CANNED_STORY, size=24):
            await self._sleep()
            if await self._is_cancelled(job_id):
                return
            await self._emit_token(job_id, chunk)

        await self._advance(
            job_id,
            JobStatus.WRITTEN,
            written_at=datetime.now(UTC),
            story_text=_CANNED_STORY,
            llm_model="inline-dispatcher",
        )
        word_count = len(_CANNED_STORY.split())

        def build_story_done(seq: int, at: datetime) -> JobEvent:
            return StoryDoneEvent(
                job_id=job_id, seq=seq, at=at, text=_CANNED_STORY, word_count=word_count
            )

        await self._emit(job_id, build_story_done)

        if not await self._advance(
            job_id,
            JobStatus.SYNTHESIZING,
            synthesizing_at=datetime.now(UTC),
            segment_count=self._segments,
        ):
            return

        for index in range(1, self._segments + 1):
            await self._sleep()
            if await self._is_cancelled(job_id):
                return
            await self._emit_progress(job_id, index)

        if not await self._advance(
            job_id,
            JobStatus.DONE,
            finished_at=datetime.now(UTC),
            segments_done=self._segments,
        ):
            return

        # A real terminal event, so watching clients see the stream end rather than
        # inferring it from a status frame. The audio list is empty on purpose: the
        # inline dispatcher synthesizes nothing and must not fabricate a media URL that
        # would make a stub run look like a real result.
        def build_done(seq: int, at: datetime) -> JobEvent:
            return DoneEvent(job_id=job_id, seq=seq, at=at, audio=[], duration_seconds=0.0)

        await self._emit(job_id, build_done)

    async def _advance(self, job_id: UUID, target: JobStatus, **values: object) -> bool:
        """Move the job to ``target``.

        Returns ``False`` when the transition is not legal from the job's current status —
        which is how a cancellation mid-pipeline stops the rest of it.

        A ``status`` event is published only for non-terminal transitions. Reaching a
        terminal state is announced by the terminal event itself (``done`` / ``failed`` /
        ``cancelled``), which is what closes the stream; publishing a terminal *status*
        frame as well would end the relay one frame early and the client would never see
        the result attached to the real event.
        """
        async with self._session_factory() as session:
            current = await session.scalar(select(Job.status).where(Job.id == job_id))
            if current is None or not can_transition(current, target):
                await session.rollback()
                return False

            # Conditional on the status just read, so a concurrent cancel between the
            # read and the write loses rather than being silently overwritten.
            applied = await session.execute(
                update(Job)
                .where(Job.id == job_id, Job.status == current)
                .values(status=target, **values)
                .returning(Job.id)
            )
            if applied.scalar_one_or_none() is None:
                await session.rollback()
                return False

            if target not in TERMINAL_STATUSES:

                def build_status(seq: int, at: datetime) -> JobEvent:
                    return StatusEvent(job_id=job_id, seq=seq, at=at, status=target)

                await self._publisher.publish(session, job_id, build_status)

            await session.commit()
        return True

    async def _emit_token(self, job_id: UUID, text: str) -> None:
        def build(seq: int, at: datetime) -> JobEvent:
            return TokenEvent(job_id=job_id, seq=seq, at=at, text=text)

        await self._emit(job_id, build)

    async def _emit_progress(self, job_id: UUID, done: int) -> None:
        total = self._segments

        def build(seq: int, at: datetime) -> JobEvent:
            return ProgressEvent(job_id=job_id, seq=seq, at=at, done=done, total=total)

        await self._emit(job_id, build)

    async def _emit(self, job_id: UUID, factory: EventFactory) -> None:
        async with self._session_factory() as session:
            await self._publisher.publish(session, job_id, factory)
            await session.commit()

    async def _is_cancelled(self, job_id: UUID) -> bool:
        """Has the job reached a terminal state underneath us?

        Checked between every step, so a DELETE while the pipeline is running actually
        stops it rather than merely marking the row. In the real TTS worker this is the
        difference between freeing the GPU and not.
        """
        async with self._session_factory() as session:
            status = await session.scalar(select(Job.status).where(Job.id == job_id))
        return status is None or status in TERMINAL_STATUSES

    async def _sleep(self) -> None:
        if self._delay:
            await asyncio.sleep(self._delay)


def _chunks(text: str, *, size: int) -> list[str]:
    """Split text into fixed-size pieces, mimicking batched token frames."""
    return [text[index : index + size] for index in range(0, len(text), size)]
