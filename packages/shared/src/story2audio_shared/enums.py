"""Closed vocabularies shared by the gateway and both workers.

Every one of these was a free-form string in v1. `language` in particular was interpolated
straight into a HuggingFace model id, and story length travelled as a `[PARA_LEVEL:1–3]`
sentinel spliced into the user's prompt text. Making them enums is what closes those holes.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Final


class JobStatus(StrEnum):
    """Lifecycle of a generation job.

    ``written`` is a distinct state rather than an internal detail: the story is durably
    persisted at that point, so a failure in the TTS stage retries synthesis alone and
    never re-runs the LLM.
    """

    QUEUED = "queued"
    WRITING = "writing"
    WRITTEN = "written"
    SYNTHESIZING = "synthesizing"
    DONE = "done"
    FAILED = "failed"
    CANCELLED = "cancelled"


TERMINAL_STATUSES: Final[frozenset[JobStatus]] = frozenset(
    {JobStatus.DONE, JobStatus.FAILED, JobStatus.CANCELLED}
)

#: A job may be cancelled only while work is still outstanding.
CANCELLABLE_STATUSES: Final[frozenset[JobStatus]] = frozenset(
    {
        JobStatus.QUEUED,
        JobStatus.WRITING,
        JobStatus.WRITTEN,
        JobStatus.SYNTHESIZING,
    }
)

#: The only legal status transitions.
#:
#: Every worker update is a conditional UPDATE that asserts the previous status, so a
#: duplicate task delivery (Celery is at-least-once) cannot move a job backwards or
#: resurrect one that has already reached a terminal state.
ALLOWED_TRANSITIONS: Final[dict[JobStatus, frozenset[JobStatus]]] = {
    JobStatus.QUEUED: frozenset({JobStatus.WRITING, JobStatus.FAILED, JobStatus.CANCELLED}),
    JobStatus.WRITING: frozenset({JobStatus.WRITTEN, JobStatus.FAILED, JobStatus.CANCELLED}),
    JobStatus.WRITTEN: frozenset({JobStatus.SYNTHESIZING, JobStatus.FAILED, JobStatus.CANCELLED}),
    JobStatus.SYNTHESIZING: frozenset({JobStatus.DONE, JobStatus.FAILED, JobStatus.CANCELLED}),
    JobStatus.DONE: frozenset(),
    JobStatus.FAILED: frozenset(),
    JobStatus.CANCELLED: frozenset(),
}


def can_transition(current: JobStatus, target: JobStatus) -> bool:
    """Return whether ``current -> target`` is a legal status transition."""
    return target in ALLOWED_TRANSITIONS[current]


class StoryLength(StrEnum):
    """Requested story length.

    A first-class field. In v1 this was a `[PARA_LEVEL:...]` marker embedded in the prompt
    string, parsed by substring match, and stripped back out again in three separate
    files — so a user who typed the marker themselves could change server-side routing.
    """

    SHORT = "short"
    MEDIUM = "medium"
    LONG = "long"


class VoiceMode(StrEnum):
    """How many voices the finished audio uses."""

    NARRATION = "narration"
    NARRATION_WITH_DIALOGUE = "narration_with_dialogue"


class Emotion(StrEnum):
    """Emotional register for the story.

    In v1 this was passed to ``TTS.tts_to_file(emotion=...)``, which XTTS v2 accepts and
    ignores — the control did nothing audible. Here it conditions the *writing* instead,
    where it demonstrably changes the output.
    """

    NEUTRAL = "neutral"
    HAPPY = "happy"
    SAD = "sad"
    ANGRY = "angry"


class Language(StrEnum):
    """Output language.

    Restricted to the intersection of what XTTS v2 synthesises and what v1 offered. The
    story is written directly in this language by the LLM; v1's English-then-MarianMT
    round trip is gone, along with the unvalidated model-id interpolation it required.
    """

    EN = "en"
    ES = "es"
    FR = "fr"
    DE = "de"
    IT = "it"
    RU = "ru"
    HI = "hi"


#: Human-readable names, used in the prompt so the model is told a language rather than a
#: two-letter code, and in the UI language picker.
LANGUAGE_NAMES: Final[dict[Language, str]] = {
    Language.EN: "English",
    Language.ES: "Spanish",
    Language.FR: "French",
    Language.DE: "German",
    Language.IT: "Italian",
    Language.RU: "Russian",
    Language.HI: "Hindi",
}


class SegmentKind(StrEnum):
    """Whether a slice of the story is narration or a character's spoken line."""

    NARRATION = "narration"
    DIALOGUE = "dialogue"


class AudioFormat(StrEnum):
    """Delivery formats. MP3 is streamed to the player; WAV is offered for download."""

    MP3 = "mp3"
    WAV = "wav"
