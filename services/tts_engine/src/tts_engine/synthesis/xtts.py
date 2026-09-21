"""XTTS v2 backend.

Two things differ from how v1 used the same model.

**Conditioning is computed once per voice.** v1 passed ``speaker_wav=<path>`` to every
``tts_to_file`` call, so the reference sample was re-read and the conditioning latents
recomputed for every segment — twelve times over for a twelve-segment dialogue job. Here
:meth:`embed` computes them once and the service caches them by voice id.

**The dead ``emotion`` argument is gone.** v1 passed ``emotion=`` to ``tts_to_file``;
XTTS v2 accepts and ignores it, so the control changed nothing audible. Emotion now
steers the writing instead, where it demonstrably changes the output.

torch and coqui-tts are imported inside :meth:`load` rather than at module scope, so this
module can be imported — and the rest of the service tested — on a machine that has
neither installed.
"""

from __future__ import annotations

import io
import tempfile
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING, Any

from tts_engine.logging import get_logger
from tts_engine.synthesis.base import (
    AudioFormat,
    SampleFormat,
    SpeakerEmbedding,
    SynthesisError,
)

if TYPE_CHECKING:
    import numpy as np

log = get_logger(__name__)

#: XTTS v2 renders at 24 kHz mono.
_SAMPLE_RATE = 24_000

#: Restricted to what the rest of the system exposes, not to everything XTTS can read.
_SUPPORTED_LANGUAGES = frozenset({"en", "es", "fr", "de", "it", "ru", "hi"})


class XttsBackend:
    """Holds XTTS v2 in VRAM and streams PCM."""

    name = "xtts"

    def __init__(
        self,
        *,
        model_name: str,
        device: str,
        chunk_bytes: int = 32 * 1024,
    ) -> None:
        self.model = model_name
        self.device = device
        self.audio_format = AudioFormat(
            sample_rate=_SAMPLE_RATE, channels=1, sample_format=SampleFormat.PCM_S16LE
        )
        self.supported_languages = _SUPPORTED_LANGUAGES
        self._chunk_bytes = chunk_bytes
        self._model: Any = None
        self._torch: Any = None

    @property
    def ready(self) -> bool:
        return self._model is not None

    def load(self) -> None:
        """Bring the model into memory.

        Slow — tens of seconds on a warm disk, minutes on a cold one — which is why the
        whole architecture is asynchronous: a cold start is queue time on a job that was
        never going to be synchronous, and the user is already reading streamed story
        text while it happens.
        """
        if self._model is not None:
            return

        import torch
        from TTS.api import TTS

        self._torch = torch
        log.info("xtts_loading", model=self.model, device=self.device)

        tts = TTS(model_name=self.model, progress_bar=False)
        tts.to(self.device)
        # Reach through to the underlying model: the high-level `TTS` wrapper exposes
        # only file-in/file-out, which is what forced v1 to recompute conditioning on
        # every call and to write every segment through a temporary file.
        self._model = tts.synthesizer.tts_model
        log.info("xtts_loaded", model=self.model, device=self.device)

    def embed(self, voice_id: str, reference_wav: bytes) -> SpeakerEmbedding:
        """Compute the conditioning latents for one voice.

        This is the expensive part that v1 repeated per segment.
        """
        if self._model is None:
            raise SynthesisError("model is not loaded")
        if not reference_wav:
            raise SynthesisError("reference audio was empty")

        path = self._materialise(reference_wav)
        try:
            gpt_cond_latent, speaker_embedding = self._model.get_conditioning_latents(
                audio_path=[path]
            )
        except Exception as exc:
            raise SynthesisError(f"could not embed voice {voice_id}: {exc}") from exc
        finally:
            Path(path).unlink(missing_ok=True)

        return SpeakerEmbedding(
            voice_id=voice_id,
            payload=(gpt_cond_latent, speaker_embedding),
            reference_duration_seconds=_wav_duration_seconds(reference_wav),
        )

    def synthesize(
        self,
        text: str,
        embedding: SpeakerEmbedding,
        *,
        language: str,
        speed: float,
    ) -> Iterator[bytes]:
        if self._model is None:
            raise SynthesisError("model is not loaded")
        if language not in self.supported_languages:
            raise SynthesisError(f"unsupported language {language!r}")
        if not text.strip():
            raise SynthesisError("text was empty")

        gpt_cond_latent, speaker_embedding = embedding.payload

        try:
            stream = self._model.inference_stream(
                text,
                language,
                gpt_cond_latent,
                speaker_embedding,
                speed=speed,
                enable_text_splitting=True,
            )
            for tensor in stream:
                yield from self._to_pcm_chunks(tensor)
        except Exception as exc:
            raise SynthesisError(f"synthesis failed: {exc}") from exc

    def _to_pcm_chunks(self, tensor: Any) -> Iterator[bytes]:
        """Convert one model output tensor to bounded little-endian PCM16 chunks."""
        import numpy as np

        samples: np.ndarray[Any, Any] = tensor.detach().to("cpu").numpy().reshape(-1)
        # Clip before casting: XTTS occasionally overshoots and a wraparound is an
        # audible crack rather than a clipped peak.
        clipped = np.clip(samples, -1.0, 1.0)
        pcm = (clipped * 32767.0).astype("<i2").tobytes()

        for start in range(0, len(pcm), self._chunk_bytes):
            yield pcm[start : start + self._chunk_bytes]

    def _materialise(self, reference_wav: bytes) -> str:
        """Write the reference sample where the model can read it.

        Coqui's conditioning API takes file paths. The file is created with a unique
        name in the system temporary directory and removed immediately afterwards — v1
        wrote every upload to one hardcoded ``uploaded_speaker.wav``, so two concurrent
        requests overwrote each other's reference voice.
        """

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as handle:
            handle.write(reference_wav)
            return handle.name

    def close(self) -> None:
        if self._model is None:
            return
        self._model = None
        if self._torch is not None and self._torch.cuda.is_available():
            self._torch.cuda.empty_cache()
        log.info("xtts_unloaded")


def _wav_duration_seconds(data: bytes) -> float:
    import wave

    try:
        with io.BytesIO(data) as stream, wave.open(stream, "rb") as wav:
            return wav.getnframes() / float(wav.getframerate())
    except Exception:  # noqa: BLE001 - duration is reported, never depended upon
        return 0.0
