"""The synthesis backend contract.

Narrow on purpose: embed a speaker, stream PCM for a piece of text. Everything else —
segmentation, stitching, encoding, upload — belongs to the worker, which is CPU work and
must not occupy a GPU while it happens.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Protocol


class SampleFormat(StrEnum):
    PCM_S16LE = "pcm_s16le"
    PCM_F32LE = "pcm_f32le"


@dataclass(frozen=True, slots=True)
class AudioFormat:
    """What the backend emits. Sent once, ahead of any audio."""

    sample_rate: int
    channels: int
    sample_format: SampleFormat

    @property
    def bytes_per_frame(self) -> int:
        width = 2 if self.sample_format is SampleFormat.PCM_S16LE else 4
        return width * self.channels

    def duration_seconds(self, pcm_bytes: int) -> float:
        return pcm_bytes / (self.bytes_per_frame * self.sample_rate)


@dataclass(frozen=True, slots=True)
class SpeakerEmbedding:
    """A speaker's conditioning, computed once and reused.

    ``payload`` is backend-specific and deliberately opaque: for XTTS it is a pair of
    tensors, for the stub it is a seed. The cache stores it without inspecting it.
    """

    voice_id: str
    payload: Any
    reference_duration_seconds: float


class SynthesisError(RuntimeError):
    """A backend failed to produce audio.

    Translated to a gRPC status at the service boundary; the raw exception never crosses
    the wire.
    """


class SynthesisBackend(Protocol):
    """Loads a model and renders text with a cloned voice."""

    name: str
    model: str
    device: str
    audio_format: AudioFormat
    supported_languages: frozenset[str]

    @property
    def ready(self) -> bool:
        """Is the model resident and able to serve? Drives readiness, not liveness."""
        ...

    def load(self) -> None:
        """Bring the model into memory. Called once at startup, may take minutes."""
        ...

    def embed(self, voice_id: str, reference_wav: bytes) -> SpeakerEmbedding:
        """Compute a speaker's conditioning from a reference sample."""
        ...

    def synthesize(
        self,
        text: str,
        embedding: SpeakerEmbedding,
        *,
        language: str,
        speed: float,
    ) -> Iterator[bytes]:
        """Render ``text``, yielding raw PCM in the declared format."""
        ...

    def close(self) -> None:
        """Release the model and any device memory."""
        ...
