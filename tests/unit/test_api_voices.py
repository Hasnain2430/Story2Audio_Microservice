"""Voice catalogue API.

The upload path is where v1's worst validation bug lived: a "15 second minimum" enforced
as ``len(audio_bytes) < 15000``, which passes about a sixth of a second of audio.
"""

from __future__ import annotations

import io
from typing import Any
from uuid import uuid4

import pytest
from httpx import AsyncClient

from gateway.deps import AppState
from story2audio_shared.models import Voice
from tests.conftest import StubStorage, make_wav, new_session_client


def upload_files(
    data: bytes, *, filename: str = "voice.wav", content_type: str = "audio/wav"
) -> dict[str, Any]:
    return {"file": (filename, io.BytesIO(data), content_type)}


# --- Listing ------------------------------------------------------------------------------


async def test_builtin_voices_are_visible_to_everyone(
    client: AsyncClient, builtin_voice: Voice
) -> None:
    response = await client.get("/v1/voices")
    assert response.status_code == 200

    names = [item["name"] for item in response.json()["items"]]
    assert builtin_voice.name in names


async def test_each_voice_carries_a_preview_url(client: AsyncClient, builtin_voice: Voice) -> None:
    """So the picker can play a sample before committing to a generation."""
    response = await client.get("/v1/voices")
    item = next(i for i in response.json()["items"] if i["name"] == builtin_voice.name)
    assert item["preview_url"].startswith("https://")


# --- Upload validation ------------------------------------------------------------------------


async def test_uploading_a_valid_sample_succeeds(client: AsyncClient, storage: StubStorage) -> None:
    response = await client.post(
        "/v1/voices", data={"name": "My Voice"}, files=upload_files(make_wav(10.0))
    )

    assert response.status_code == 201
    body = response.json()
    assert body["name"] == "My Voice"
    assert body["is_builtin"] is False
    assert body["duration_seconds"] == pytest.approx(10.0, abs=0.1)
    assert f"voices/{body['id']}.wav" in storage.objects


async def test_the_v1_byte_count_bug_is_fixed(client: AsyncClient) -> None:
    """15000 bytes of 22 kHz PCM16 is about a third of a second, and v1 called it 15."""
    tiny = make_wav(0.34)
    assert len(tiny) > 15_000  # would have sailed past v1's check

    response = await client.post("/v1/voices", data={"name": "Too Short"}, files=upload_files(tiny))

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "voice_too_short"


async def test_overlong_samples_are_rejected(client: AsyncClient) -> None:
    response = await client.post(
        "/v1/voices",
        data={"name": "Too Long"},
        files=upload_files(make_wav(200.0, sample_rate=8_000)),
    )
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "voice_too_long"


async def test_a_renamed_non_audio_file_is_rejected(client: AsyncClient) -> None:
    """Decoding is the real check; the extension and content type are only claims."""
    response = await client.post(
        "/v1/voices",
        data={"name": "Not Audio"},
        files=upload_files(b"MZ\x90\x00" + b"\x00" * 40_000, filename="totally.wav"),
    )
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "voice_invalid_audio"


async def test_an_empty_upload_is_rejected(client: AsyncClient) -> None:
    response = await client.post("/v1/voices", data={"name": "Empty"}, files=upload_files(b""))
    assert response.status_code == 422


async def test_duplicate_names_within_one_session_are_rejected(client: AsyncClient) -> None:
    await client.post("/v1/voices", data={"name": "Same"}, files=upload_files(make_wav(8.0)))
    second = await client.post(
        "/v1/voices", data={"name": "Same"}, files=upload_files(make_wav(8.0))
    )
    assert second.status_code == 409
    assert second.json()["error"]["code"] == "voice_name_taken"


# --- Canonicalisation ----------------------------------------------------------------------------


async def test_uploads_are_stored_as_canonical_wav(
    client: AsyncClient, storage: StubStorage
) -> None:
    """Whatever arrives, one format is stored, so a key's extension never lies."""
    response = await client.post(
        "/v1/voices", data={"name": "Canonical"}, files=upload_files(make_wav(8.0))
    )
    key = f"voices/{response.json()['id']}.wav"

    stored = storage.objects[key]
    assert stored[:4] == b"RIFF"
    assert stored[8:12] == b"WAVE"
    assert storage.content_types[key] == "audio/wav"


async def test_long_uploads_are_clipped_for_storage(
    client: AsyncClient, storage: StubStorage
) -> None:
    """The stored clip travels to the TTS engine over gRPC.

    A 30-second 48 kHz stereo file is 5.7 MB and exceeds gRPC's 4 MB default message
    limit outright — which is exactly how this surfaced, as a RESOURCE_EXHAUSTED from
    the transport. Clipping at upload is the fix; raising the limit would be v1's hack.

    30 seconds is not an arbitrary ceiling: it is XTTS's ``gpt_cond_len``, the amount of
    reference the model actually reads. Clipping shorter than that — this was 20 — hands
    the model less conditioning than it asked for, and the clones get worse.
    """
    response = await client.post(
        "/v1/voices", data={"name": "Long"}, files=upload_files(make_wav(90.0))
    )

    assert response.status_code == 201
    # Reported duration is the stored clip, not the upload.
    assert response.json()["duration_seconds"] == pytest.approx(30.0, abs=0.1)

    stored = storage.objects[f"voices/{response.json()['id']}.wav"]
    assert len(stored) < 4 * 1024 * 1024


async def test_high_sample_rate_uploads_are_shortened_to_fit_the_wire(
    client: AsyncClient, storage: StubStorage
) -> None:
    """The byte budget wins over the clip length, by shortening rather than resampling.

    30 seconds of 96 kHz mono PCM16 is 5.5 MB — over gRPC's limit even after the clip.
    The gateway has no resampler and should not grow one for this, so such an upload is
    stored as fewer seconds instead. A shorter reference is a smaller loss than a voice
    that fails at synthesis time, which is how the same overflow presented before.

    40 seconds, not 90: at this rate a 90-second upload is 17 MB and is refused by the
    upload cap before the budget is ever consulted, which would make this test pass for
    the wrong reason.
    """
    response = await client.post(
        "/v1/voices",
        data={"name": "Hi-res"},
        files=upload_files(make_wav(40.0, sample_rate=96_000)),
    )

    assert response.status_code == 201
    body = response.json()
    assert body["sample_rate"] == 96_000
    # Shortened below the nominal clip, rather than rejected or resampled.
    assert body["duration_seconds"] < 30.0

    stored = storage.objects[f"voices/{body['id']}.wav"]
    assert len(stored) < 4 * 1024 * 1024


async def test_uploads_are_stored_as_mono(client: AsyncClient, storage: StubStorage) -> None:
    """The engine synthesises mono, so a second channel is bytes nothing ever uses."""
    import wave

    response = await client.post(
        "/v1/voices", data={"name": "Stereo"}, files=upload_files(make_wav(8.0, channels=2))
    )
    assert response.status_code == 201

    stored = storage.objects[f"voices/{response.json()['id']}.wav"]
    with wave.open(io.BytesIO(stored), "rb") as handle:
        assert handle.getnchannels() == 1


async def test_stored_key_is_derived_from_the_id_not_the_name(
    client: AsyncClient, storage: StubStorage
) -> None:
    """v1 wrote every upload to the single hardcoded path `uploaded_speaker.wav`."""
    first = await client.post("/v1/voices", data={"name": "A"}, files=upload_files(make_wav(8.0)))
    second = await client.post("/v1/voices", data={"name": "B"}, files=upload_files(make_wav(8.0)))

    assert f"voices/{first.json()['id']}.wav" in storage.objects
    assert f"voices/{second.json()['id']}.wav" in storage.objects
    assert len([key for key in storage.objects if key.startswith("voices/")]) >= 2


# --- Isolation between sessions ----------------------------------------------------------------


async def test_your_uploads_are_invisible_to_other_sessions(
    client: AsyncClient, app_state: AppState
) -> None:
    await client.post("/v1/voices", data={"name": "Private"}, files=upload_files(make_wav(8.0)))

    async with new_session_client(app_state) as stranger:
        names = [item["name"] for item in (await stranger.get("/v1/voices")).json()["items"]]

    assert "Private" not in names


async def test_another_session_cannot_use_your_voice_for_a_job(
    client: AsyncClient, app_state: AppState
) -> None:
    created = await client.post(
        "/v1/voices", data={"name": "Mine"}, files=upload_files(make_wav(8.0))
    )
    voice_id = created.json()["id"]

    async with new_session_client(app_state) as stranger:
        response = await stranger.post(
            "/v1/jobs", json={"prompt": "A story.", "voice_id": voice_id}
        )

    assert response.status_code == 403
    assert response.json()["error"]["code"] == "voice_forbidden"


# --- Deletion ------------------------------------------------------------------------------------


async def test_deleting_your_own_voice_removes_the_object(
    client: AsyncClient, storage: StubStorage
) -> None:
    created = await client.post(
        "/v1/voices", data={"name": "Disposable"}, files=upload_files(make_wav(8.0))
    )
    voice_id = created.json()["id"]

    response = await client.delete(f"/v1/voices/{voice_id}")
    assert response.status_code == 204
    assert f"voices/{voice_id}.wav" in storage.deleted


async def test_builtin_voices_cannot_be_deleted(client: AsyncClient, builtin_voice: Voice) -> None:
    response = await client.delete(f"/v1/voices/{builtin_voice.id}")
    assert response.status_code == 403


async def test_a_voice_in_use_by_a_job_cannot_be_deleted(client: AsyncClient) -> None:
    """Refused explicitly, rather than surfacing the foreign key's RESTRICT as a 500."""
    created = await client.post(
        "/v1/voices", data={"name": "In Use"}, files=upload_files(make_wav(8.0))
    )
    voice_id = created.json()["id"]
    await client.post("/v1/jobs", json={"prompt": "A story.", "voice_id": voice_id})

    response = await client.delete(f"/v1/voices/{voice_id}")
    assert response.status_code == 403


async def test_deleting_an_unknown_voice_is_not_found(client: AsyncClient) -> None:
    assert (await client.delete(f"/v1/voices/{uuid4()}")).status_code == 404
