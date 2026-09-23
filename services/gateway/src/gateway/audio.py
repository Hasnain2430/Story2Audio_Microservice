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

    #: Canonical WAV bytes: mono, clipped. Whatever came in, this is what gets written.
    wav_bytes: bytes
    #: Duration of the *stored* clip, which is what the catalogue reports.
    duration_seconds: float
    #: Duration of the uploaded file, before clipping.
    source_duration_seconds: float
    sample_rate: int
    #: Channel count of the upload. The stored clip is always mono.
    source_channels: int


#: Ceiling on the stored reference clip, in bytes. The clip is sent to the TTS engine in
#: one gRPC message, and the default limit there is 4 MB.
_REFERENCE_BYTE_BUDGET = 3 * 1024 * 1024


def validate_voice_upload(
    data: bytes,
    *,
    min_duration_seconds: float,
    max_duration_seconds: float,
    max_bytes: int,
    clip_seconds: float = 30.0,
) -> ValidatedVoice:
    """Decode, check and canonicalise an uploaded voice sample.

    The stored form is always **mono and clipped**. Voice cloning needs a few seconds of
    reference, not a few minutes, and the stored clip is what later travels to the TTS
    engine over gRPC — where a 30-second 48 kHz stereo file is 5.7 MB and exceeds the
    default 4 MB message limit outright. Clipping at the point of storage is the fix;
    raising the transport limit would be v1's hack, and v1 needed exactly that hack.

    How long to clip to is not a free choice. XTTS reads ``gpt_cond_len`` seconds of the
    reference — 30 in the shipped config — so a clip shorter than that throws away
    conditioning the model would have used, and a longer one stores bytes it will never
    read. 30 seconds is the point where those meet.

    Sample rate is left as uploaded, so the byte budget is enforced by *shortening* rather
    than by resampling: a 96 kHz upload is clipped to fewer seconds instead of pulling a
    resampler into the gateway. Rare, and a shorter reference is a smaller loss than a
    failed upload.

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

    # Mix to mono. The engine synthesises mono, so a second channel is bytes on the wire
    # that no part of the pipeline ever uses.
    mono = frames.mean(axis=1) if frames.shape[1] > 1 else frames[:, 0]

    # Stay inside gRPC's default 4 MB message, with headroom for the WAV header and the
    # rest of the request. PCM_16 mono is two bytes per frame.
    budget_seconds = _REFERENCE_BYTE_BUDGET / (2 * sample_rate)
    clipped = mono[: int(min(clip_seconds, budget_seconds) * sample_rate)]
    stored_duration = len(clipped) / sample_rate

    # One canonical form regardless of what was uploaded, so the engine reads WAV every
    # time and the storage key's extension never lies about its contents.
    buffer = io.BytesIO()
    sf.write(buffer, clipped, sample_rate, format="WAV", subtype="PCM_16")

    return ValidatedVoice(
        wav_bytes=buffer.getvalue(),
        duration_seconds=stored_duration,
        source_duration_seconds=duration_seconds,
        sample_rate=sample_rate,
        source_channels=channels,
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
