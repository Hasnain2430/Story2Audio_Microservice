"""Reference-voice validation and canonicalisation.

v1 enforced its "minimum 15 seconds" rule as ``len(audio_bytes) < 15000`` — a byte count.
15000 bytes of 44.1 kHz 16-bit mono is about 0.17 seconds, so the check passed essentially
anything and XTTS was handed samples far too short to clone from.

Here the file is actually decoded. That does three jobs at once: it measures the real
duration, it proves the bytes are audio rather than something renamed to ``.wav``, and it
yields decoded frames that can be written back out in one canonical form.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Final

import soundfile as sf

from story2audio_shared.errors import AppError, ErrorCode

#: Container formats accepted for upload. Deliberately narrow: every one of these is
#: something libsndfile decodes without shelling out to an external binary.
ALLOWED_CONTENT_TYPES: Final[frozenset[str]] = frozenset(
    {
        "audio/wav",
        "audio/wave",
        "audio/x-wav",
        "audio/mpeg",
        "audio/mp3",
        "audio/flac",
        "audio/x-flac",
        "audio/ogg",
        "application/octet-stream",  # what browsers send for a drag-and-dropped file
    }
)


@dataclass(frozen=True, slots=True)
class ValidatedVoice:
    """A decoded, re-encoded reference sample ready to store."""

    #: Canonical WAV bytes. Whatever came in, this is what gets written.
    wav_bytes: bytes
    duration_seconds: float
    sample_rate: int
    channels: int


def validate_voice_upload(
    data: bytes,
    *,
    min_duration_seconds: float,
    max_duration_seconds: float,
    max_bytes: int,
) -> ValidatedVoice:
    """Decode, check and canonicalise an uploaded voice sample.

    Raises:
        AppError: with a classified :class:`ErrorCode` for every rejection path, so the
            caller never has to interpret a libsndfile exception.
    """
    if len(data) > max_bytes:
        raise AppError(
            ErrorCode.VOICE_TOO_LARGE,
            detail=f"upload is {len(data)} bytes, limit is {max_bytes}",
        )
    if not data:
        raise AppError(ErrorCode.VOICE_INVALID_AUDIO, detail="upload was empty")

    try:
        with sf.SoundFile(io.BytesIO(data)) as source:
            sample_rate = int(source.samplerate)
            channels = int(source.channels)
            frames = source.read(dtype="float32", always_2d=True)
    except Exception as exc:
        raise AppError(
            ErrorCode.VOICE_INVALID_AUDIO, detail=f"could not decode upload: {exc}"
        ) from exc

    if sample_rate <= 0:
        raise AppError(ErrorCode.VOICE_INVALID_AUDIO, detail="decoded sample rate was zero")

    duration_seconds = len(frames) / sample_rate
    if duration_seconds < min_duration_seconds:
        raise AppError(
            ErrorCode.VOICE_TOO_SHORT,
            detail=f"{duration_seconds:.2f}s is below the {min_duration_seconds}s minimum",
        )
    if duration_seconds > max_duration_seconds:
        raise AppError(
            ErrorCode.VOICE_TOO_LONG,
            detail=f"{duration_seconds:.2f}s is above the {max_duration_seconds}s maximum",
        )

    # Store one canonical form regardless of what was uploaded, so the TTS engine reads
    # WAV every time and the storage key's extension never lies about its contents.
    buffer = io.BytesIO()
    sf.write(buffer, frames, sample_rate, format="WAV", subtype="PCM_16")

    return ValidatedVoice(
        wav_bytes=buffer.getvalue(),
        duration_seconds=duration_seconds,
        sample_rate=sample_rate,
        channels=channels,
    )


def check_content_type(content_type: str | None) -> None:
    """Reject an obviously wrong content type before spending memory on a decode.

    A cheap first filter only — the decode in :func:`validate_voice_upload` is what
    actually establishes that the bytes are audio. A client can claim any content type it
    likes, so this is a courtesy to honest clients, not a security control.
    """
    if content_type is None:
        return
    base = content_type.split(";", 1)[0].strip().lower()
    if base and base not in ALLOWED_CONTENT_TYPES:
        raise AppError(ErrorCode.VOICE_INVALID_AUDIO, detail=f"unsupported content type {base!r}")
