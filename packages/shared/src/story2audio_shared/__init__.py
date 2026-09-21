"""Shared domain for Story2Audio.

Everything defined here is imported by the gateway and by both workers, so that job
statuses, request schemas, prompt templates and error codes have exactly one definition
and cannot drift between services.

Module map:

``enums``      closed vocabularies and the job state machine
``errors``     error taxonomy: code -> HTTP status, public message, retryability
``ids``        UUIDv7 generation (time-ordered ids double as pagination cursors)
``schemas``    Pydantic models for the public REST API
``events``     the discriminated-union event contract for WebSocket progress
``prompts``    the single parameterised story prompt template
``models``     SQLAlchemy ORM models
``storage``    S3-compatible object storage client
``config``     settings groups, read from the environment
``logging``    structured logging with trace/job correlation
"""

from story2audio_shared.enums import (
    ALLOWED_TRANSITIONS,
    CANCELLABLE_STATUSES,
    LANGUAGE_NAMES,
    TERMINAL_STATUSES,
    AudioFormat,
    Emotion,
    JobStatus,
    Language,
    SegmentKind,
    StoryLength,
    VoiceMode,
    can_transition,
)
from story2audio_shared.errors import AppError, ErrorCode, ErrorSpec, is_retryable, spec_for
from story2audio_shared.ids import timestamp_ms_of, uuid7

__version__ = "2.0.0"

__all__ = [
    "ALLOWED_TRANSITIONS",
    "CANCELLABLE_STATUSES",
    "LANGUAGE_NAMES",
    "TERMINAL_STATUSES",
    "AppError",
    "AudioFormat",
    "Emotion",
    "ErrorCode",
    "ErrorSpec",
    "JobStatus",
    "Language",
    "SegmentKind",
    "StoryLength",
    "VoiceMode",
    "__version__",
    "can_transition",
    "is_retryable",
    "spec_for",
    "timestamp_ms_of",
    "uuid7",
]
