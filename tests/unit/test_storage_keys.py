"""Storage key derivation.

v1 built output paths from the prompt text, so two users submitting the same prompt wrote
to -- and read back from -- the same file, and every uploaded voice landed on the single
hardcoded path `uploaded_speaker.wav`. Keys derived from ids make both collisions
impossible.
"""

from __future__ import annotations

import pytest
from story2audio_shared.enums import AudioFormat
from story2audio_shared.ids import uuid7
from story2audio_shared.storage import AUDIO_PREFIX, VOICE_PREFIX, audio_key, voice_key


def test_voice_keys_are_unique_per_voice() -> None:
    keys = {voice_key(uuid7()) for _ in range(1_000)}
    assert len(keys) == 1_000


def test_audio_keys_are_unique_per_job_and_format() -> None:
    job_id = uuid7()
    assert audio_key(job_id, AudioFormat.MP3) != audio_key(job_id, AudioFormat.WAV)


def test_identical_prompts_cannot_collide() -> None:
    # The v1 failure: sanitize_filename(prompt, speaker) produced the same path for two
    # different users with the same prompt.
    first, second = uuid7(), uuid7()
    assert audio_key(first, AudioFormat.MP3) != audio_key(second, AudioFormat.MP3)


@pytest.mark.parametrize("audio_format", sorted(AudioFormat))
def test_keys_are_confined_to_their_prefix(audio_format: AudioFormat) -> None:
    key = audio_key(uuid7(), audio_format)
    assert key.startswith(f"{AUDIO_PREFIX}/")
    assert ".." not in key
    assert not key.startswith("/")
    assert key.endswith(f".{audio_format.value}")


def test_voice_keys_are_confined_to_their_prefix() -> None:
    key = voice_key(uuid7())
    assert key.startswith(f"{VOICE_PREFIX}/")
    assert ".." not in key
    assert not key.startswith("/")
