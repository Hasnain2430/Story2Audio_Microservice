"""Wire schemas for the public API.

These are the only shapes that cross the gateway boundary. Every field that was an
unvalidated string in v1 is either an enum or a bounded value here, and identifiers are
opaque ids resolved server-side rather than filesystem paths chosen by the client.
"""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Generic, Self, TypeVar
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

from story2audio_shared.enums import (
    AudioFormat,
    Emotion,
    JobStatus,
    Language,
    SegmentKind,
    StoryLength,
    VoiceMode,
)
from story2audio_shared.errors import ErrorCode

#: Upper bound on prompt length at the schema layer. The configurable
#: ``LimitSettings.max_prompt_chars`` may tighten this per deployment; this is the ceiling
#: that no deployment may exceed.
MAX_PROMPT_CHARS = 20_000

PromptText = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=MAX_PROMPT_CHARS),
]
VoiceName = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=60),
]
Speed = Annotated[float, Field(ge=0.5, le=1.5)]

#: Element type of a paginated collection. An explicit TypeVar rather than PEP 695 syntax:
#: the services target Python 3.11, where `class Page[ItemT]` is a syntax error.
ItemT = TypeVar("ItemT")


class ApiModel(BaseModel):
    """Base for every wire model.

    ``extra="forbid"`` means an unknown field is a 422 rather than a silently ignored
    typo — the failure mode that makes a client bug look like a server bug.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, use_enum_values=False)


# --- Requests -------------------------------------------------------------------------


class CreateJobRequest(ApiModel):
    """Body of ``POST /v1/jobs``."""

    prompt: PromptText
    length: StoryLength = StoryLength.MEDIUM
    mode: VoiceMode = VoiceMode.NARRATION
    language: Language = Language.EN
    emotion: Emotion = Emotion.NEUTRAL
    speed: Speed = 1.0

    #: Narrator voice. An opaque id resolved against the voice catalogue server-side; v1
    #: accepted a filesystem path from the client and passed it straight to the TTS engine.
    voice_id: UUID
    #: Voice for the first speaking character. Required in dialogue mode, rejected
    #: otherwise. v1 hardcoded ``voices/female.wav`` and called it multi-speaker support.
    dialogue_voice_id: UUID | None = None
    #: Voice for the second speaking character. Optional: without it both characters are
    #: read by ``dialogue_voice_id``, which is what the mode used to do for every line in
    #: the story. The prompt asks for exactly two named characters, so this is the voice
    #: that makes an exchange sound like two people rather than one person answering
    #: themselves.
    second_dialogue_voice_id: UUID | None = None

    @model_validator(mode="after")
    def _dialogue_voices_match_mode(self) -> Self:
        if self.mode is VoiceMode.NARRATION_WITH_DIALOGUE:
            if self.dialogue_voice_id is None:
                raise ValueError(
                    "dialogue_voice_id is required when mode is narration_with_dialogue"
                )
            if self.second_dialogue_voice_id == self.dialogue_voice_id:
                # Permitted by the data model, but always a mistake by the caller: it asks
                # for two characters and then gives them one voice.
                raise ValueError("second_dialogue_voice_id must differ from dialogue_voice_id")
        elif self.dialogue_voice_id is not None or self.second_dialogue_voice_id is not None:
            raise ValueError("dialogue voices are only valid when mode is narration_with_dialogue")
        return self


class CreateVoiceRequest(ApiModel):
    """Metadata accompanying a voice upload.

    The audio itself arrives as a multipart file part, not base64 in a JSON body. v1's
    REST proxy took a base64 WAV inline, which inflates the payload by a third and forced
    the whole sample through the JSON parser.
    """

    name: VoiceName


# --- Responses ------------------------------------------------------------------------


class ErrorDetail(ApiModel):
    """Public description of a failure. Never carries internal detail."""

    code: ErrorCode
    message: str
    retryable: bool


class JobTimings(ApiModel):
    """Per-stage timestamps.

    The source of the v1-versus-v2 performance comparison, and of the ``/stats`` page, so
    the improvement is measured from real jobs rather than asserted.
    """

    queued_at: datetime
    writing_at: datetime | None = None
    written_at: datetime | None = None
    synthesizing_at: datetime | None = None
    finished_at: datetime | None = None

    @property
    def llm_seconds(self) -> float | None:
        if self.writing_at is None or self.written_at is None:
            return None
        return (self.written_at - self.writing_at).total_seconds()

    @property
    def tts_seconds(self) -> float | None:
        if self.synthesizing_at is None or self.finished_at is None:
            return None
        return (self.finished_at - self.synthesizing_at).total_seconds()

    @property
    def total_seconds(self) -> float | None:
        if self.finished_at is None:
            return None
        return (self.finished_at - self.queued_at).total_seconds()


class AudioAsset(ApiModel):
    """One rendered audio file, addressed by a short-lived presigned URL.

    Audio never travels through the API itself. v1 returned it as protobuf ``bytes``,
    which is why both client and server had to raise their gRPC message limit to 100 MB.
    """

    format: AudioFormat
    url: str
    duration_seconds: float
    size_bytes: int
    expires_at: datetime


class SpokenSegment(ApiModel):
    """One spoken segment, placed in the audio and in the story.

    Two coordinate systems, because they are not the same text. ``start_seconds`` and
    ``end_seconds`` are the segment's position in the rendered track, measured by the
    worker that assembled it. ``start_char`` and ``end_char`` are its span in
    ``story_text`` — needed separately because the spoken form has been cleaned
    (quotes stripped, whitespace collapsed) and so cannot be located by searching.

    Together they are what a player needs to highlight the story as it is read.
    """

    index: int
    kind: SegmentKind
    #: The cleaned text, as actually spoken.
    text: str
    #: Which character speaks this, where the story attributed it. ``None`` for narration,
    #: and for a line whose speaker could not be read from the prose.
    speaker: str | None = None
    start_char: int
    end_char: int
    start_seconds: float
    end_seconds: float


class JobResponse(ApiModel):
    """A job as the API presents it."""

    id: UUID
    status: JobStatus
    prompt: str
    length: StoryLength
    mode: VoiceMode
    language: Language
    emotion: Emotion
    speed: float
    voice_id: UUID
    dialogue_voice_id: UUID | None = None
    second_dialogue_voice_id: UUID | None = None

    story_text: str | None = None
    audio: list[AudioAsset] = Field(default_factory=list)
    segment_count: int | None = None
    #: Empty until the audio exists, and stays empty for jobs finished before the
    #: timeline was recorded. A client must treat it as an enhancement, never as a
    #: requirement: the story and the audio are both complete without it.
    segments: list[SpokenSegment] = Field(default_factory=list)

    error: ErrorDetail | None = None
    timings: JobTimings
    created_at: datetime
    updated_at: datetime

    @property
    def is_terminal(self) -> bool:
        return self.status in {JobStatus.DONE, JobStatus.FAILED, JobStatus.CANCELLED}


class CreateJobResponse(ApiModel):
    """``202 Accepted`` body.

    This is the whole architectural change in one shape: the caller gets an id and a
    place to watch, in under 200 ms, instead of holding a socket open for ten minutes.
    """

    id: UUID
    status: JobStatus
    events_url: str


class VoiceResponse(ApiModel):
    """A voice in the catalogue."""

    id: UUID
    name: str
    is_builtin: bool
    duration_seconds: float
    sample_rate: int
    preview_url: str | None = None
    created_at: datetime


class Page(ApiModel, Generic[ItemT]):
    """Cursor-paginated collection.

    The cursor is the last item's UUIDv7, which is time-ordered — so it doubles as the
    sort key and needs no companion timestamp column or tiebreak (see ADR-0002).
    """

    items: list[ItemT]
    next_cursor: UUID | None = None
    has_more: bool = False
