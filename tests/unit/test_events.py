"""The WebSocket event contract (ADR-0003).

Two properties matter: the union discriminates correctly so the TypeScript client can
switch exhaustively, and `seq` makes a dropped frame detectable rather than invisible.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from story2audio_shared.enums import AudioFormat, JobStatus
from story2audio_shared.errors import ErrorCode
from story2audio_shared.events import (
    EVENT_SCHEMA_VERSION,
    CancelledEvent,
    DoneEvent,
    FailedEvent,
    ProgressEvent,
    StatusEvent,
    StoryDoneEvent,
    TokenEvent,
    is_terminal_event,
    job_channel,
    job_event_adapter,
)
from story2audio_shared.ids import uuid7
from story2audio_shared.schemas import AudioAsset

JOB_ID = uuid7()
AT = datetime(2026, 1, 1, tzinfo=UTC)


def test_channel_name_is_scoped_to_one_job() -> None:
    other = uuid7()
    assert job_channel(JOB_ID) != job_channel(other)
    assert str(JOB_ID) in job_channel(JOB_ID)


# --- Discrimination -----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        ({"type": "status", "status": "writing"}, StatusEvent),
        ({"type": "token", "text": "Once upon"}, TokenEvent),
        ({"type": "story_done", "text": "...", "word_count": 612}, StoryDoneEvent),
        ({"type": "progress", "done": 3, "total": 11}, ProgressEvent),
        ({"type": "failed", "code": "tts_timeout", "message": "x", "retryable": True}, FailedEvent),
        ({"type": "cancelled"}, CancelledEvent),
    ],
)
def test_wire_payloads_parse_into_the_right_variant(
    payload: dict[str, object], expected: type
) -> None:
    event = job_event_adapter.validate_python(
        {"job_id": str(JOB_ID), "seq": 1, "at": AT.isoformat(), **payload}
    )
    assert isinstance(event, expected)


def test_unknown_event_type_is_rejected() -> None:
    with pytest.raises(ValidationError):
        job_event_adapter.validate_python(
            {"type": "surprise", "job_id": str(JOB_ID), "seq": 1, "at": AT.isoformat()}
        )


def test_round_trip_through_json_preserves_the_variant() -> None:
    original = ProgressEvent(job_id=JOB_ID, seq=7, at=AT, done=4, total=9)
    restored = job_event_adapter.validate_json(original.model_dump_json())
    assert restored == original


# --- Sequencing ---------------------------------------------------------------------------


def test_seq_must_be_positive() -> None:
    with pytest.raises(ValidationError):
        StatusEvent(job_id=JOB_ID, seq=0, at=AT, status=JobStatus.QUEUED)


def test_gap_in_seq_is_detectable_by_a_reconnecting_client() -> None:
    received = [
        TokenEvent(job_id=JOB_ID, seq=1, at=AT, text="a"),
        TokenEvent(job_id=JOB_ID, seq=4, at=AT, text="d"),
    ]
    assert received[1].seq != received[0].seq + 1


def test_schema_version_is_stamped_on_every_event() -> None:
    event = TokenEvent(job_id=JOB_ID, seq=1, at=AT, text="a")
    assert event.v == EVENT_SCHEMA_VERSION


# --- Terminal detection ---------------------------------------------------------------------


def test_terminal_events_close_the_stream() -> None:
    done = DoneEvent(
        job_id=JOB_ID,
        seq=9,
        at=AT,
        duration_seconds=180.0,
        audio=[
            AudioAsset(
                format=AudioFormat.MP3,
                url="https://example.invalid/a.mp3",
                duration_seconds=180.0,
                size_bytes=1,
                expires_at=AT,
            )
        ],
    )
    failed = FailedEvent(
        job_id=JOB_ID, seq=9, at=AT, code=ErrorCode.LLM_TIMEOUT, message="x", retryable=True
    )
    cancelled = CancelledEvent(job_id=JOB_ID, seq=9, at=AT)

    assert is_terminal_event(done)
    assert is_terminal_event(failed)
    assert is_terminal_event(cancelled)


def test_progress_events_do_not_close_the_stream() -> None:
    assert not is_terminal_event(ProgressEvent(job_id=JOB_ID, seq=2, at=AT, done=1, total=5))
    assert not is_terminal_event(TokenEvent(job_id=JOB_ID, seq=2, at=AT, text="x"))
    assert not is_terminal_event(
        StatusEvent(job_id=JOB_ID, seq=2, at=AT, status=JobStatus.SYNTHESIZING)
    )


def test_progress_fraction_is_clamped() -> None:
    assert ProgressEvent(job_id=JOB_ID, seq=1, at=AT, done=0, total=4).fraction == 0.0
    assert ProgressEvent(job_id=JOB_ID, seq=1, at=AT, done=2, total=4).fraction == 0.5
    # A duplicate delivery could push `done` past `total`; the bar must not exceed 100%.
    assert ProgressEvent(job_id=JOB_ID, seq=1, at=AT, done=9, total=4).fraction == 1.0


def test_progress_total_cannot_be_zero() -> None:
    # A zero total would make the fraction a division by zero in the client.
    with pytest.raises(ValidationError):
        ProgressEvent(job_id=JOB_ID, seq=1, at=AT, done=0, total=0)
