"""Deterministic stub backend.

Produces a tone, not speech. It exists so that the gRPC service, the streaming protocol,
the embedding cache, the concurrency limit and the whole `tts-worker` pipeline can be
tested on a machine with no GPU and in CI — which is most of the surface area, none of
which needs a real model to be wrong.

Refused in production by :func:`tts_engine.synthesis.factory.build_backend`: serving a
tone as a finished story would be worse than failing outright.

The output is deterministic — same voice and text give byte-identical audio — so tests
can assert on content rather than merely on length.
"""

from __future__ import annotations

import hashlib
import io
import math
import struct
import wave
from collections.abc import Iterator

from tts_engine.synthesis.base import (
    AudioFormat,
    SampleFormat,
    SpeakerEmbedding,
    SynthesisError,
)

#: Roughly conversational pace, so a stub render takes about as long as real speech and
#: downstream duration assertions stay meaningful.
_CHARS_PER_SECOND = 14.0
_MIN_DURATION_SECONDS = 0.20
_MAX_DURATION_SECONDS = 120.0

_SUPPORTED_LANGUAGES = frozenset({"en", "es", "fr", "de", "it", "ru", "hi"})


class StubBackend:
    """A backend that renders a per-voice tone instead of speech."""

    name = "stub"

    def __init__(self, *, sample_rate: int = 24_000, chunk_bytes: int = 32 * 1024) -> None:
        self.model = "stub-tone"
        self.device = "cpu"
        self.audio_format = AudioFormat(
            sample_rate=sample_rate, channels=1, sample_format=SampleFormat.PCM_S16LE
        )
        self.supported_languages = _SUPPORTED_LANGUAGES
        self._chunk_bytes = chunk_bytes
        self._loaded = False

    @property
    def ready(self) -> bool:
        return self._loaded

    def load(self) -> None:
        self._loaded = True

    def embed(self, voice_id: str, reference_wav: bytes) -> SpeakerEmbedding:
        """Derive a stable pitch from the reference bytes.

        Different voices produce audibly different tones, so a test can tell that the
        narrator and the dialogue voice were actually used for different segments.

        The sample must decode. A stub that accepted any bytes would leave the service's
        rejection path untested while the real backend fails on exactly that input --
        the stub is only useful if it fails where XTTS would.
        """
        if not reference_wav:
            raise SynthesisError("reference audio was empty")

        duration = self._wav_duration_seconds(reference_wav)

        digest = hashlib.sha256(reference_wav).digest()
        # 160-360 Hz: a plausible speech range, and far enough apart between voices to
        # be distinguishable in a spectrum check.
        frequency = 160.0 + (digest[0] / 255.0) * 200.0

        return SpeakerEmbedding(
            voice_id=voice_id,
            payload=frequency,
            reference_duration_seconds=duration,
        )

    def synthesize(
        self,
        text: str,
        embedding: SpeakerEmbedding,
        *,
        language: str,
        speed: float,
    ) -> Iterator[bytes]:
        if language not in self.supported_languages:
            raise SynthesisError(f"unsupported language {language!r}")
        if speed <= 0:
            raise SynthesisError(f"speed must be positive, got {speed}")
        if not text.strip():
            raise SynthesisError("text was empty")

        frequency = float(embedding.payload)
        duration = min(
            _MAX_DURATION_SECONDS,
            max(_MIN_DURATION_SECONDS, len(text) / (_CHARS_PER_SECOND * speed)),
        )
        yield from self._tone(frequency, duration)

    def _tone(self, frequency: float, duration_seconds: float) -> Iterator[bytes]:
        sample_rate = self.audio_format.sample_rate
        total_frames = int(duration_seconds * sample_rate)
        frames_per_chunk = max(1, self._chunk_bytes // self.audio_format.bytes_per_frame)

        emitted = 0
        while emitted < total_frames:
            count = min(frames_per_chunk, total_frames - emitted)
            buffer = bytearray()
            for index in range(emitted, emitted + count):
                # Gentle fade at both ends, so joined segments do not click -- the same
                # reason the worker fades real segments.
                envelope = _envelope(index, total_frames)
                value = int(
                    18_000 * envelope * math.sin(2 * math.pi * frequency * index / sample_rate)
                )
                buffer += struct.pack("<h", value)
            emitted += count
            yield bytes(buffer)

    @staticmethod
    def _wav_duration_seconds(data: bytes) -> float:
        """Read the duration from a WAV header, rejecting anything that is not one."""
        try:
            with io.BytesIO(data) as stream, wave.open(stream, "rb") as wav:
                frame_rate = wav.getframerate()
                if frame_rate <= 0:
                    raise SynthesisError("reference audio declares a zero sample rate")
                return wav.getnframes() / float(frame_rate)
        except SynthesisError:
            raise
        except Exception as exc:
            raise SynthesisError(f"reference audio is not a readable WAV: {exc}") from exc

    def close(self) -> None:
        self._loaded = False


def _envelope(index: int, total: int) -> float:
    """Linear fade over the first and last 5 ms worth of frames."""
    fade = max(1, total // 200)
    if index < fade:
        return index / fade
    if index > total - fade:
        return max(0.0, (total - index) / fade)
    return 1.0
