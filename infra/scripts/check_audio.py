"""Tell speech apart from a test tone.

This script exists because of a mistake. The first XTTS verification measured peak
amplitude, RMS and the proportion of non-silent samples, and concluded "that is a human
voice, not a tone" — but **none of those statistics can distinguish a sine wave from
speech**. A 220 Hz tone scores identically on all three. The stub backend renders exactly
such a tone, so the check could not have caught the very thing it was run to rule out.

What actually separates them is the shape of the spectrum and how it moves:

``single_bin_energy``
    Fraction of power in the loudest FFT bin. A pure tone concentrates nearly everything
    there; speech spreads power across a fundamental and its formants.

``spectral_flatness``
    Geometric mean of the power spectrum over its arithmetic mean. Near zero for a tone,
    materially higher for the broadband, noisy content of real speech.

``spectral_flux``
    How much the normalised spectrum changes between frames. A steady tone barely moves;
    speech is in constant motion as the articulators change.

A second lesson from the same episode: verify with text of a realistic *length*. The
first probe used a 52-character sentence, which never reached Coqui's text splitter, so
it missed `enable_text_splitting=True requires Spacy` — a failure that hit every real
segment immediately. A probe short enough to be convenient is short enough to miss things.

Usage::

    uv run python infra/scripts/check_audio.py rendered.wav
    uv run python infra/scripts/check_audio.py --job <job-id>   # pull from the API
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import urllib.request
import wave
from dataclasses import dataclass
from http.cookiejar import CookieJar
from pathlib import Path

import numpy as np

#: Above this share of power in one bin, the signal is effectively a single tone.
TONE_SINGLE_BIN_THRESHOLD = 0.25
#: Below this, the spectrum moves too little to be speech.
SPEECH_MIN_FLUX = 0.15


@dataclass(frozen=True, slots=True)
class Analysis:
    duration_seconds: float
    sample_rate: int
    peak_hz: float
    single_bin_energy: float
    spectral_flatness: float
    spectral_flux: float

    @property
    def verdict(self) -> str:
        if self.single_bin_energy > TONE_SINGLE_BIN_THRESHOLD:
            return "TONE — almost all power sits in one frequency"
        if self.spectral_flux < SPEECH_MIN_FLUX:
            return "TONE or steady noise — the spectrum barely moves"
        return "SPEECH — broadband and time-varying"

    @property
    def is_speech(self) -> bool:
        return self.verdict.startswith("SPEECH")


def analyse(wav_bytes: bytes) -> Analysis:
    with wave.open(io.BytesIO(wav_bytes), "rb") as handle:
        sample_rate = handle.getframerate()
        frames = handle.readframes(handle.getnframes())

    samples = np.frombuffer(frames, dtype="<i2").astype(np.float64) / 32768.0
    if samples.size == 0:
        raise SystemExit("file contains no audio")

    # Analyse a window around the loudest moment, so leading silence cannot dominate.
    window = min(len(samples), sample_rate)
    centre = int(np.argmax(np.abs(samples)))
    start = max(0, min(len(samples) - window, centre - window // 2))
    segment = samples[start : start + window]

    power = np.abs(np.fft.rfft(segment * np.hanning(len(segment)))) ** 2 + 1e-20
    freqs = np.fft.rfftfreq(len(segment), 1 / sample_rate)
    normalised = power / power.sum()

    flatness = float(np.exp(np.mean(np.log(normalised))) / np.mean(normalised))

    frame_size, hop = 1024, 512
    spectra = [
        np.abs(np.fft.rfft(samples[i : i + frame_size] * np.hanning(frame_size)))[:200]
        for i in range(0, max(0, len(samples) - frame_size), hop)
    ]
    if len(spectra) > 1:
        matrix = np.array(spectra)
        matrix = matrix / (matrix.sum(axis=1, keepdims=True) + 1e-12)
        flux = float(np.mean(np.abs(np.diff(matrix, axis=0)).sum(axis=1)))
    else:
        flux = 0.0

    return Analysis(
        duration_seconds=len(samples) / sample_rate,
        sample_rate=sample_rate,
        peak_hz=float(freqs[int(np.argmax(power))]),
        single_bin_energy=float(power.max() / power.sum()),
        spectral_flatness=flatness,
        spectral_flux=flux,
    )


def fetch_job_audio(job_id: str, api: str) -> bytes:
    opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(CookieJar()))
    with opener.open(f"{api}/v1/jobs/{job_id}", timeout=30) as response:
        job = json.load(response)

    assets = job.get("audio") or []
    wav = next((a for a in assets if a["format"] == "wav"), None)
    if wav is None:
        raise SystemExit(f"job {job_id} has no WAV yet (status {job['status']})")

    with opener.open(wav["url"], timeout=120) as response:
        data: bytes = response.read()
    return data


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", nargs="?", type=Path, help="WAV file to analyse")
    parser.add_argument("--job", help="job id to pull from the API instead")
    parser.add_argument("--api", default="http://localhost:8000")
    args = parser.parse_args()

    if args.job:
        data = fetch_job_audio(args.job, args.api)
    elif args.path:
        data = args.path.read_bytes()
    else:
        parser.error("give a WAV path or --job")

    result = analyse(data)
    sys.stdout.write(
        f"  duration          {result.duration_seconds:.2f}s @ {result.sample_rate} Hz\n"
        f"  dominant freq     {result.peak_hz:.0f} Hz\n"
        f"  single-bin energy {result.single_bin_energy * 100:.1f}%"
        f"   (a pure tone is >{TONE_SINGLE_BIN_THRESHOLD * 100:.0f}%)\n"
        f"  spectral flatness {result.spectral_flatness:.5f}\n"
        f"  spectral flux     {result.spectral_flux:.3f}"
        f"   (speech is >{SPEECH_MIN_FLUX})\n"
        f"\n  {result.verdict}\n"
    )
    return 0 if result.is_speech else 1


if __name__ == "__main__":
    raise SystemExit(main())
