"""Job events published over Redis pub/sub and relayed to clients over WebSocket.

Design constraints, settled in ADR-0003:

*Discriminated union.* Every event carries a literal ``type``, so the TypeScript client
gets an exhaustive ``switch`` with no casting and a new event variant becomes a compile
error rather than a silent no-op.

*Per-job monotonic ``seq``.* Redis pub/sub is fire-and-forget: a client that reconnects
mid-job cannot tell "nothing happened" from "I missed four frames". A sequence number
makes the gap detectable. On detecting one, the client refetches ``GET /v1/jobs/{id}``
rather than attempting a replay — Redis holds no history to replay from.

*Snapshot on connect.* The gateway sends a :class:`StatusEvent` built from Postgres as the
first frame on every connection, so a late or reconnecting subscriber starts from truth.
"""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Literal
from uuid import UUID

from pydantic import Field, TypeAdapter

from story2audio_shared.enums import JobStatus, SegmentKind
from story2audio_shared.errors import ErrorCode
from story2audio_shared.schemas import ApiModel, AudioAsset

#: Bumped only on a breaking change to the event contract. The client refuses to
#: interpret a version it does not know rather than mis-rendering it.
EVENT_SCHEMA_VERSION = 1


def job_channel(job_id: UUID) -> str:
    """Redis pub/sub channel carrying one job's events."""
    return f"job:{job_id}"


class BaseEvent(ApiModel):
    """Fields common to every event."""

    job_id: UUID
    #: Monotonic within a job, starting at 1. Gaps mean dropped frames.
    seq: int = Field(ge=1)
    at: datetime
    v: int = EVENT_SCHEMA_VERSION


class StatusEvent(BaseEvent):
    """The job entered a new status.

    Also sent as the first frame of every WebSocket connection, as the snapshot that lets
    a reconnecting client resynchronise.
    """

    type: Literal["status"] = "status"
    status: JobStatus
    #: True when this frame is the connect-time snapshot rather than a live transition.
    #: A snapshot reuses the job's current sequence number instead of allocating a new
    #: one, so the client must not treat its `seq` as evidence of a gap.
    snapshot: bool = False


class TokenEvent(BaseEvent):
    """A chunk of story text from the LLM.

    Batched at roughly 50 ms in the worker rather than published per token: one frame per
    token would flood the socket without making the text arrive any sooner.
    """

    type: Literal["token"] = "token"
    text: str


class StoryDoneEvent(BaseEvent):
    """The story is written and persisted.

    Emitted at the ``written`` boundary, which is also the retry boundary: a later TTS
    failure resumes from here and never re-runs the LLM.
    """

    type: Literal["story_done"] = "story_done"
    text: str
    word_count: int


class ProgressEvent(BaseEvent):
    """Synthesis progress, in completed segments.

    Real counts, not an indeterminate spinner: the story is split into a known number of
    narration and dialogue segments before synthesis starts.
    """

    type: Literal["progress"] = "progress"
    done: int = Field(ge=0)
    total: int = Field(ge=1)

    @property
    def fraction(self) -> float:
        return min(1.0, self.done / self.total)


class SegmentReadyEvent(BaseEvent):
    """One segment has been rendered, uploaded, and can be played.

    This is what makes the audio arrive while it is still being made. The first segment
    of a three-minute story is finished about seven seconds in; without this the listener
    waits for the last one before hearing the first.

    It deliberately carries **no URL**. Signing here would put this worker's clock and a
    fixed TTL inside a durable event, and the event may be replayed to a client that
    connects much later. The client asks the gateway for the audio by index, and the
    gateway signs on read, where the TTL means something (ADR-0003).

    ``start_seconds`` is the segment's position in the finished track, including the
    silence in front of it, so a client can schedule segments against one clock and
    reproduce exactly the file the worker will assemble.

    Adding a variant is backwards compatible: a client that does not know this type
    ignores it and keeps using the terminal event, which is why the schema version does
    not change.
    """

    type: Literal["segment_ready"] = "segment_ready"
    index: int = Field(ge=0)
    total: int = Field(ge=1)
    kind: SegmentKind
    text: str
    speaker: str | None = None
    start_char: int = Field(ge=0)
    end_char: int = Field(ge=0)
    #: Position in the assembled track, silence in front of it included.
    start_seconds: float = Field(ge=0)
    end_seconds: float = Field(ge=0)


class DoneEvent(BaseEvent):
    """Terminal: the audio is rendered, uploaded and addressable."""

    type: Literal["done"] = "done"
    audio: list[AudioAsset]
    duration_seconds: float


class FailedEvent(BaseEvent):
    """Terminal: the job failed.

    Carries the classified code and its fixed public message. Raw exception text stays in
    the logs — v1 returned it to the caller verbatim.
    """

    type: Literal["failed"] = "failed"
    code: ErrorCode
    message: str
    retryable: bool


class CancelledEvent(BaseEvent):
    """Terminal: the job was cancelled by its owner."""

    type: Literal["cancelled"] = "cancelled"


JobEvent = Annotated[
    StatusEvent
    | TokenEvent
    | StoryDoneEvent
    | ProgressEvent
    | SegmentReadyEvent
    | DoneEvent
    | FailedEvent
    | CancelledEvent,
    Field(discriminator="type"),
]

#: Parses any event off the wire into the right concrete model.
job_event_adapter: TypeAdapter[JobEvent] = TypeAdapter(JobEvent)

TERMINAL_EVENT_TYPES = frozenset({"done", "failed", "cancelled"})


def is_terminal_event(event: JobEvent) -> bool:
    """Return whether this event ends the stream, so the socket can be closed."""
    return event.type in TERMINAL_EVENT_TYPES
