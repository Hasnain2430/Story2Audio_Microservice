"""Error taxonomy.

v1 returned ``context.set_details(str(e))`` — the raw exception text, including internal
paths, straight to the caller. Here every failure is classified into an :class:`ErrorCode`
that carries a fixed, user-safe message, an HTTP status, and whether a retry could plausibly
succeed. The original exception is logged with full context and never crosses the wire.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Final


class ErrorCode(StrEnum):
    """Machine-readable failure reason.

    Stored on the job row, published in the ``failed`` event, and returned in API error
    bodies, so the frontend can branch on the cause without parsing prose.
    """

    # --- Request problems (4xx) ------------------------------------------------------
    VALIDATION_FAILED = "validation_failed"
    PROMPT_TOO_LONG = "prompt_too_long"
    PROMPT_EMPTY = "prompt_empty"
    JOB_NOT_FOUND = "job_not_found"
    JOB_NOT_CANCELLABLE = "job_not_cancellable"
    SEGMENT_NOT_FOUND = "segment_not_found"
    VOICE_NOT_FOUND = "voice_not_found"
    VOICE_FORBIDDEN = "voice_forbidden"
    VOICE_INVALID_AUDIO = "voice_invalid_audio"
    VOICE_TOO_SHORT = "voice_too_short"
    VOICE_TOO_LONG = "voice_too_long"
    VOICE_TOO_LARGE = "voice_too_large"
    VOICE_NAME_TAKEN = "voice_name_taken"
    DIALOGUE_VOICE_REQUIRED = "dialogue_voice_required"

    # --- Throttling and spend guardrails (429) ---------------------------------------
    RATE_LIMITED = "rate_limited"
    CONCURRENCY_LIMIT_REACHED = "concurrency_limit_reached"
    DAILY_CAP_REACHED = "daily_cap_reached"

    # --- LLM stage -------------------------------------------------------------------
    LLM_UNAVAILABLE = "llm_unavailable"
    LLM_TIMEOUT = "llm_timeout"
    LLM_RATE_LIMITED = "llm_rate_limited"
    LLM_CONTENT_REJECTED = "llm_content_rejected"
    STORY_EMPTY = "story_empty"

    # --- TTS stage -------------------------------------------------------------------
    TTS_UNAVAILABLE = "tts_unavailable"
    TTS_TIMEOUT = "tts_timeout"
    TTS_CAPACITY = "tts_capacity"
    AUDIO_ASSEMBLY_FAILED = "audio_assembly_failed"

    # --- Infrastructure ---------------------------------------------------------------
    STORAGE_UNAVAILABLE = "storage_unavailable"
    CANCELLED = "cancelled"
    INTERNAL = "internal"


@dataclass(frozen=True, slots=True)
class ErrorSpec:
    """Everything the system is allowed to say publicly about one failure mode."""

    http_status: int
    message: str
    #: Whether retrying the *same* request could plausibly succeed. Drives worker retry
    #: policy and whether the UI offers a "try again" button.
    retryable: bool


_SPECS: Final[dict[ErrorCode, ErrorSpec]] = {
    ErrorCode.VALIDATION_FAILED: ErrorSpec(422, "The request could not be validated.", False),
    ErrorCode.PROMPT_TOO_LONG: ErrorSpec(422, "That storyline is too long.", False),
    ErrorCode.PROMPT_EMPTY: ErrorSpec(422, "Please describe the story you want.", False),
    ErrorCode.JOB_NOT_FOUND: ErrorSpec(404, "That story could not be found.", False),
    ErrorCode.JOB_NOT_CANCELLABLE: ErrorSpec(
        409, "That story has already finished and cannot be cancelled.", False
    ),
    # Routine rather than exceptional: a client that heard a segment was ready can ask
    # for it before the object has settled, and a job made before progressive playback
    # existed has no segments stored at all. Retryable, because waiting usually fixes it.
    ErrorCode.SEGMENT_NOT_FOUND: ErrorSpec(404, "That part is not ready yet.", True),
    ErrorCode.VOICE_NOT_FOUND: ErrorSpec(404, "That voice could not be found.", False),
    ErrorCode.VOICE_FORBIDDEN: ErrorSpec(403, "That voice is not available to you.", False),
    ErrorCode.VOICE_INVALID_AUDIO: ErrorSpec(
        422, "That file could not be read as audio. Please upload a WAV or MP3.", False
    ),
    ErrorCode.VOICE_TOO_SHORT: ErrorSpec(
        422, "The voice sample is too short to clone from.", False
    ),
    ErrorCode.VOICE_TOO_LONG: ErrorSpec(422, "The voice sample is too long.", False),
    ErrorCode.VOICE_TOO_LARGE: ErrorSpec(413, "That file is too large.", False),
    ErrorCode.VOICE_NAME_TAKEN: ErrorSpec(409, "You already have a voice with that name.", False),
    ErrorCode.DIALOGUE_VOICE_REQUIRED: ErrorSpec(
        422, "Dialogue mode needs a second voice for the spoken lines.", False
    ),
    ErrorCode.RATE_LIMITED: ErrorSpec(429, "Too many requests. Please slow down.", True),
    ErrorCode.CONCURRENCY_LIMIT_REACHED: ErrorSpec(
        429, "You already have the maximum number of stories generating.", True
    ),
    ErrorCode.DAILY_CAP_REACHED: ErrorSpec(
        429, "The demo has reached its daily limit. Please try again tomorrow.", True
    ),
    ErrorCode.LLM_UNAVAILABLE: ErrorSpec(503, "The story writer is temporarily unavailable.", True),
    ErrorCode.LLM_TIMEOUT: ErrorSpec(504, "Writing the story took too long.", True),
    ErrorCode.LLM_RATE_LIMITED: ErrorSpec(
        503, "The story writer is busy. Please try again shortly.", True
    ),
    ErrorCode.LLM_CONTENT_REJECTED: ErrorSpec(
        422, "That storyline could not be turned into a story. Try rewording it.", False
    ),
    ErrorCode.STORY_EMPTY: ErrorSpec(502, "The story came back empty. Please try again.", True),
    ErrorCode.TTS_UNAVAILABLE: ErrorSpec(503, "The voice engine is temporarily unavailable.", True),
    ErrorCode.TTS_TIMEOUT: ErrorSpec(504, "Generating the audio took too long.", True),
    ErrorCode.TTS_CAPACITY: ErrorSpec(
        503, "The voice engine is at capacity. Please try again shortly.", True
    ),
    ErrorCode.AUDIO_ASSEMBLY_FAILED: ErrorSpec(500, "The audio could not be assembled.", True),
    ErrorCode.STORAGE_UNAVAILABLE: ErrorSpec(503, "Storage is temporarily unavailable.", True),
    ErrorCode.CANCELLED: ErrorSpec(499, "This story was cancelled.", False),
    ErrorCode.INTERNAL: ErrorSpec(500, "Something went wrong on our side.", True),
}

# A missing entry would mean a failure with no safe public message, which is exactly the
# v1 behaviour this module exists to prevent. Catch it at import time, not in production.
# Deliberately not an `assert`: this invariant must survive `python -O`.
_unspecified = set(ErrorCode) - set(_SPECS)
if _unspecified:  # pragma: no cover - import-time invariant
    raise RuntimeError(
        f"ErrorCode members without an ErrorSpec: {sorted(code.value for code in _unspecified)}"
    )


def spec_for(code: ErrorCode) -> ErrorSpec:
    """Return the public specification for ``code``."""
    return _SPECS[code]


def is_retryable(code: ErrorCode) -> bool:
    """Return whether a retry of the same request could plausibly succeed."""
    return _SPECS[code].retryable


class AppError(Exception):
    """An error that is safe to surface to the caller.

    ``detail`` is for logs only and is never serialised into a response. Anything the user
    should see must be expressible through the :class:`ErrorCode`, so that adding a new
    failure mode forces a deliberate decision about what to disclose.
    """

    def __init__(self, code: ErrorCode, *, detail: str | None = None) -> None:
        self.code = code
        self.spec = spec_for(code)
        self.detail = detail
        super().__init__(detail or self.spec.message)

    @property
    def http_status(self) -> int:
        return self.spec.http_status

    @property
    def public_message(self) -> str:
        return self.spec.message

    @property
    def retryable(self) -> bool:
        return self.spec.retryable

    def __repr__(self) -> str:
        return f"AppError(code={self.code.value!r}, detail={self.detail!r})"
