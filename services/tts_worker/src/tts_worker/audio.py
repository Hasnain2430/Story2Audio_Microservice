"""Assembling the finished audio.

v1's stitching logic was sound and is ported here: trim the silence the model leaves at
each end, fade in and out by 20 ms so joins do not click, and separate segments with a
short pause. What changes is the implementation — v1 did this with `pydub`, which meant
ffmpeg in the image and a subprocess per export. Every one of those operations is array
slicing and multiplication, so they are numpy here, and MP3 encoding uses `lameenc`
rather than shelling out.

All audio is mono signed 16-bit little-endian PCM, which is what the engine streams.
"""

from __future__ import annotations

import io
import wave
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

#: v1's values, kept: enough to remove the model's lead-in without clipping speech.
DEFAULT_SILENCE_THRESHOLD_DBFS = -40.0
DEFAULT_FADE_MS = 20
DEFAULT_SEGMENT_PAUSE_MS = 300
DEFAULT_LEAD_SILENCE_MS = 300

#: Analysis window for silence detection, matching v1's 10 ms chunks.
_WINDOW_MS = 10

_FULL_SCALE = 32768.0

#: Target peak for the finished mix. Leaves headroom so a player's own processing does
#: not clip -- v1 applied no normalisation at all, so output level varied per voice.
DEFAULT_PEAK_TARGET = 0.89


@dataclass(frozen=True, slots=True)
class PcmAudio:
    """Mono PCM16 samples at a known rate."""

    samples: np.ndarray
    sample_rate: int

    @property
    def duration_seconds(self) -> float:
        return len(self.samples) / float(self.sample_rate)

    @property
    def is_empty(self) -> bool:
        return len(self.samples) == 0


def from_pcm_bytes(pcm: bytes, sample_rate: int) -> PcmAudio:
    """Wrap raw little-endian PCM16 bytes.

    An odd trailing byte means a truncated stream; it is dropped rather than shifting
    every subsequent sample by one byte and turning the whole segment into noise.
    """
    usable = len(pcm) - (len(pcm) % 2)
    samples = np.frombuffer(pcm[:usable], dtype="<i2").astype(np.int16)
    return PcmAudio(samples=samples, sample_rate=sample_rate)


def trim_silence(
    audio: PcmAudio, *, threshold_dbfs: float = DEFAULT_SILENCE_THRESHOLD_DBFS
) -> PcmAudio:
    """Remove leading and trailing silence.

    v1 walked the audio in 10 ms chunks from each end until one exceeded the threshold;
    this computes the per-window level once and takes the first and last window above it.
    Same result, one pass instead of two loops.
    """
    if audio.is_empty:
        return audio

    window = max(1, int(audio.sample_rate * _WINDOW_MS / 1000))
    usable = len(audio.samples) - (len(audio.samples) % window)
    if usable == 0:
        return audio

    frames = audio.samples[:usable].reshape(-1, window).astype(np.float32) / _FULL_SCALE
    # Peak rather than RMS: a brief transient at the start of a word should count as
    # speech, and RMS over a 10 ms window can average it away.
    levels = np.max(np.abs(frames), axis=1)

    threshold = 10.0 ** (threshold_dbfs / 20.0)
    loud = np.flatnonzero(levels > threshold)
    if loud.size == 0:
        # Entirely below the threshold. Returning empty is correct: a silent segment
        # should contribute nothing but its pause.
        return PcmAudio(samples=audio.samples[:0], sample_rate=audio.sample_rate)

    start = int(loud[0]) * window
    end = min(len(audio.samples), (int(loud[-1]) + 1) * window)
    return PcmAudio(samples=audio.samples[start:end], sample_rate=audio.sample_rate)


def apply_fades(audio: PcmAudio, *, fade_ms: int = DEFAULT_FADE_MS) -> PcmAudio:
    """Fade the first and last few milliseconds.

    Without this, joining two segments that each start mid-waveform produces an audible
    click at every boundary — the reason v1 faded too.
    """
    if audio.is_empty or fade_ms <= 0:
        return audio

    fade = min(int(audio.sample_rate * fade_ms / 1000), len(audio.samples) // 2)
    if fade <= 0:
        return audio

    samples = audio.samples.astype(np.float32)
    ramp = np.linspace(0.0, 1.0, fade, dtype=np.float32)
    samples[:fade] *= ramp
    samples[-fade:] *= ramp[::-1]

    return PcmAudio(samples=_to_int16(samples), sample_rate=audio.sample_rate)


def silence(duration_ms: int, sample_rate: int) -> PcmAudio:
    count = max(0, int(sample_rate * duration_ms / 1000))
    return PcmAudio(samples=np.zeros(count, dtype=np.int16), sample_rate=sample_rate)


def join(segments: list[PcmAudio], *, sample_rate: int, gaps_ms: Sequence[int]) -> PcmAudio:
    """Concatenate segments, preceding each one with the silence it was given.

    ``gaps_ms[i]`` is the silence placed *before* segment ``i``, so ``gaps_ms[0]`` is the
    lead-in. The lead-in matters for playback: browsers and podcast players often clip the
    very first moment of a stream, and v1 prefixed 500 ms for the same reason.

    Taking a gap per boundary rather than one pause for all of them is what lets the
    caller distinguish a paragraph break from a change of speaker from a split that only
    happened because a segment hit the size limit. Those are three different silences, and
    using one length for all three is audible: it chops narration into equal slabs and
    lets a reply tread on the line it answers.
    """
    if not segments:
        return silence(0, sample_rate)
    if len(gaps_ms) != len(segments):
        raise ValueError(f"expected {len(segments)} gaps, got {len(gaps_ms)}")

    pieces: list[np.ndarray] = []
    for gap_ms, segment in zip(gaps_ms, segments, strict=True):
        pieces.append(silence(gap_ms, sample_rate).samples)
        pieces.append(segment.samples)

    return PcmAudio(samples=np.concatenate(pieces), sample_rate=sample_rate)


def normalise_peak(audio: PcmAudio, *, target: float = DEFAULT_PEAK_TARGET) -> PcmAudio:
    """Scale to a consistent peak level.

    New in v2. v1 applied none, so output loudness varied with whichever reference voice
    was used and a user switching voices heard the volume jump.

    Only ever attenuates or amplifies toward the target; a silent track is left alone
    rather than being multiplied by infinity.
    """
    if audio.is_empty:
        return audio

    samples = audio.samples.astype(np.float32) / _FULL_SCALE
    peak = float(np.max(np.abs(samples)))
    if peak <= 1e-6:
        return audio

    return PcmAudio(
        samples=_to_int16(samples * (target / peak) * _FULL_SCALE), sample_rate=audio.sample_rate
    )


def encode_wav(audio: PcmAudio) -> bytes:
    """Encode to a WAV container. Offered as the download format."""
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(audio.sample_rate)
        handle.writeframes(audio.samples.tobytes())
    return buffer.getvalue()


def encode_mp3(audio: PcmAudio, *, bitrate_kbps: int = 128) -> bytes:
    """Encode to MP3. Streamed to the player, where size matters more than fidelity.

    Uses `lameenc` rather than an ffmpeg subprocess, so the worker image needs no system
    audio tooling at all.
    """
    import lameenc

    encoder = lameenc.Encoder()
    encoder.set_bit_rate(bitrate_kbps)
    encoder.set_in_sample_rate(audio.sample_rate)
    encoder.set_channels(1)
    encoder.set_quality(2)  # 0 best, 9 worst; 2 is transparent enough for speech
    encoder.silence()

    data = bytes(encoder.encode(audio.samples.tobytes()))
    return data + bytes(encoder.flush())


def _to_int16(samples: np.ndarray) -> np.ndarray:
    """Clip and cast back to PCM16.

    Clipping before the cast matters: an out-of-range float wraps around on conversion,
    turning a loud passage into a burst of noise rather than a clipped peak.
    """
    clipped: np.ndarray = np.clip(samples, -_FULL_SCALE, _FULL_SCALE - 1).astype(np.int16)
    return clipped
