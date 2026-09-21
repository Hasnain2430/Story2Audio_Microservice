"""Publishing job events.

Each event gets a per-job sequence number allocated by an atomic ``UPDATE ... RETURNING``
on ``jobs.last_event_seq``, so two publishers racing cannot hand out the same number. The
sequence is what lets a reconnecting WebSocket client tell "nothing happened" from "I
missed four frames" — see ADR-0003.

Publishing is best-effort by design. Redis pub/sub has no delivery guarantee, and the
durable record of what happened is the ``jobs`` row, not the event stream. A failed
publish must therefore never fail the operation that produced it: the client will simply
see the change on its next poll or on reconnect.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from uuid import UUID

from redis.asyncio import Redis
from sqlalchemy import update
from sqlalchemy.ext.asyncio import AsyncSession

from story2audio_shared.events import JobEvent, job_channel
from story2audio_shared.logging import get_logger
from story2audio_shared.models import Job

log = get_logger(__name__)

#: Builds an event once its sequence number and timestamp are known.
EventFactory = Callable[[int, datetime], JobEvent]


class EventPublisher:
    """Allocates sequence numbers and fans events out over Redis pub/sub."""

    def __init__(self, redis: Redis) -> None:
        self._redis = redis

    async def next_seq(self, session: AsyncSession, job_id: UUID) -> int:
        """Reserve the next sequence number for a job.

        A single atomic statement rather than read-modify-write, so concurrent publishers
        (the gateway and a worker, say) cannot both take the same number.
        """
        result = await session.execute(
            update(Job)
            .where(Job.id == job_id)
            .values(last_event_seq=Job.last_event_seq + 1)
            .returning(Job.last_event_seq)
        )
        seq = result.scalar_one()
        return int(seq)

    async def publish(
        self, session: AsyncSession, job_id: UUID, factory: EventFactory
    ) -> JobEvent | None:
        """Allocate a sequence number, build the event, and publish it.

        Returns the event that was built, or ``None`` if publishing failed. The caller is
        expected to carry on regardless.
        """
        seq = await self.next_seq(session, job_id)
        event = factory(seq, datetime.now(UTC))
        await self.publish_prebuilt(event)
        return event

    async def publish_prebuilt(self, event: JobEvent) -> None:
        """Publish an already-sequenced event.

        Used for the connect-time snapshot, which reuses the job's current sequence number
        rather than consuming a new one.
        """
        try:
            await self._redis.publish(job_channel(event.job_id), event.model_dump_json())
        except Exception as exc:  # noqa: BLE001 - deliberately swallowed, see module docstring
            log.warning(
                "event_publish_failed",
                job_id=str(event.job_id),
                event_type=event.type,
                error=str(exc),
            )
