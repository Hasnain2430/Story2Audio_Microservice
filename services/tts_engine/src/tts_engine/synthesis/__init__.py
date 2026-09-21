"""Synthesis backends.

One interface, two implementations: XTTS on a GPU, and a deterministic stub for CI and
for machines without one.
"""

from tts_engine.synthesis.base import (
    AudioFormat,
    SpeakerEmbedding,
    SynthesisBackend,
    SynthesisError,
)
from tts_engine.synthesis.factory import build_backend

__all__ = [
    "AudioFormat",
    "SpeakerEmbedding",
    "SynthesisBackend",
    "SynthesisError",
    "build_backend",
]
