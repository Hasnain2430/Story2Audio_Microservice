"""Request and response schemas.

Every field that was an unvalidated string in v1 is bounded here. These tests are the
proof that the bounds are actually enforced rather than merely declared.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest
from pydantic import ValidationError
from story2audio_shared.enums import (
    AudioFormat,
    Emotion,
    JobStatus,
    Language,
    StoryLength,
    VoiceMode,
)
from story2audio_shared.ids import uuid7
from story2audio_shared.schemas import (
    MAX_PROMPT_CHARS,
    AudioAsset,
    CreateJobRequest,
    JobTimings,
    Page,
    VoiceResponse,
)

VOICE = uuid4()
DIALOGUE_VOICE = uuid4()


def _request(**overrides: object) -> CreateJobRequest:
    payload: dict[str, object] = {
        "prompt": "A lighthouse keeper hears knocking.",
        "voice_id": VOICE,
    }
    payload.update(overrides)
    return CreateJobRequest.model_validate(payload)


# --- Defaults and bounds -----------------------------------------------------------------


def test_sensible_defaults() -> None:
    request = _request()
    assert request.length is StoryLength.MEDIUM
    assert request.mode is VoiceMode.NARRATION
    assert request.language is Language.EN
    assert request.emotion is Emotion.NEUTRAL
    assert request.speed == 1.0
    assert request.dialogue_voice_id is None


@pytest.mark.parametrize("speed", [0.49, 1.51, -1.0, 0.0, 100.0])
def test_speed_outside_the_supported_range_is_rejected(speed: float) -> None:
    # v1 did `float(data["speed"])` and passed the result straight through.
    with pytest.raises(ValidationError):
        _request(speed=speed)


@pytest.mark.parametrize("speed", [0.5, 1.0, 1.5])
def test_speed_inside_the_range_is_accepted(speed: float) -> None:
    assert _request(speed=speed).speed == speed


def test_prompt_is_stripped_and_must_be_non_empty() -> None:
    assert _request(prompt="  padded  ").prompt == "padded"
    with pytest.raises(ValidationError):
        _request(prompt="   ")


def test_overlong_prompt_is_rejected() -> None:
    with pytest.raises(ValidationError):
        _request(prompt="x" * (MAX_PROMPT_CHARS + 1))


def test_unknown_language_is_rejected() -> None:
    # v1 interpolated this straight into "Helsinki-NLP/opus-mt-en-{tgt}" and downloaded it.
    with pytest.raises(ValidationError):
        _request(language="../../etc/passwd")


def test_unknown_fields_are_rejected_rather_than_ignored() -> None:
    with pytest.raises(ValidationError):
        _request(speeed=1.0)


def test_para_level_sentinel_is_not_special() -> None:
    # In v1 this string in the prompt body changed server-side model routing.
    request = _request(prompt="[PARA_LEVEL:8+] a quiet village")
    assert request.length is StoryLength.MEDIUM
    assert "[PARA_LEVEL:8+]" in request.prompt


def test_requests_are_immutable() -> None:
    request = _request()
    with pytest.raises(ValidationError):
        request.speed = 1.2


# --- Cross-field rules ---------------------------------------------------------------------


def test_dialogue_mode_requires_a_dialogue_voice() -> None:
    with pytest.raises(ValidationError, match="dialogue_voice_id is required"):
        _request(mode=VoiceMode.NARRATION_WITH_DIALOGUE)


def test_dialogue_voice_is_rejected_in_narration_mode() -> None:
    with pytest.raises(ValidationError, match="only valid when"):
        _request(mode=VoiceMode.NARRATION, dialogue_voice_id=DIALOGUE_VOICE)


def test_dialogue_mode_with_both_voices_is_accepted() -> None:
    request = _request(mode=VoiceMode.NARRATION_WITH_DIALOGUE, dialogue_voice_id=DIALOGUE_VOICE)
    assert request.dialogue_voice_id == DIALOGUE_VOICE


# --- Timings ---------------------------------------------------------------------------------


def test_stage_durations_are_computed_from_timestamps() -> None:
    start = datetime(2026, 1, 1, tzinfo=UTC)
    timings = JobTimings(
        queued_at=start,
        writing_at=start + timedelta(seconds=1),
        written_at=start + timedelta(seconds=9),
        synthesizing_at=start + timedelta(seconds=10),
        finished_at=start + timedelta(seconds=70),
    )

    assert timings.llm_seconds == 8
    assert timings.tts_seconds == 60
    assert timings.total_seconds == 70


def test_durations_are_none_while_stages_are_incomplete() -> None:
    timings = JobTimings(queued_at=datetime(2026, 1, 1, tzinfo=UTC))
    assert timings.llm_seconds is None
    assert timings.tts_seconds is None
    assert timings.total_seconds is None


# --- Responses --------------------------------------------------------------------------------


def test_audio_asset_carries_a_url_not_bytes() -> None:
    # The whole point: audio leaves the request path. v1 returned it as protobuf bytes.
    asset = AudioAsset(
        format=AudioFormat.MP3,
        url="https://example.invalid/audio/abc.mp3?sig=...",
        duration_seconds=312.5,
        size_bytes=5_012_345,
        expires_at=datetime(2026, 1, 1, tzinfo=UTC),
    )
    assert asset.url.startswith("https://")
    assert not hasattr(asset, "data")


def test_page_is_generic_over_its_item_type() -> None:
    cursor = uuid7()
    page: Page[VoiceResponse] = Page(
        items=[
            VoiceResponse(
                id=uuid7(),
                name="Default",
                is_builtin=True,
                duration_seconds=21.4,
                sample_rate=24_000,
                created_at=datetime(2026, 1, 1, tzinfo=UTC),
            )
        ],
        next_cursor=cursor,
        has_more=True,
    )
    assert page.items[0].name == "Default"
    assert page.next_cursor == cursor


def test_job_status_round_trips_as_its_string_value() -> None:
    assert JobStatus("synthesizing") is JobStatus.SYNTHESIZING
    assert JobStatus.SYNTHESIZING.value == "synthesizing"
