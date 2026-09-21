"""XTTS v2 backend.

Two things differ from how v1 used the same model.

**Conditioning is computed once per voice.** v1 passed ``speaker_wav=<path>`` to every
``tts_to_file`` call, so the reference sample was re-read and the conditioning latents
recomputed for every segment — twelve times over for a twelve-segment dialogue job. Here
:meth:`embed` computes them once and the service caches them by voice id.

**The dead ``emotion`` argument is gone.** v1 passed ``emotion=`` to ``tts_to_file``;
XTTS v2 accepts and ignores it, so the control changed nothing audible. Emotion now
steers the writing instead, where it demonstrably changes the output.

**Generation settings come from the model config, not from the method defaults.** This
was a real regression, and it is worth stating plainly because reaching past the
high-level wrapper is what caused it. ``Xtts.synthesize`` -- the path ``tts_to_file``
takes, and therefore the path v1 took -- does not call ``inference`` with its declared
defaults. It overrides them from ``config`` first, under the comment "Use generally found
best tuning knobs for generation". Calling ``inference`` and ``get_conditioning_latents``
directly skips that layer and silently picks up the raw defaults, which differ sharply:

==========================  ==========  ===========
setting                     config      raw default
==========================  ==========  ===========
``repetition_penalty``      5.0         10.0
``gpt_cond_len``            30          6
``gpt_cond_chunk_len``      4           6
==========================  ==========  ===========

Those are the values in the *downloaded* ``config.json``, which is what matters: it
overrides the defaults declared on ``XttsConfig``, so reading the dataclass is not the
same as reading the model. They are taken from ``model.config`` here for that reason.

``repetition_penalty`` at twice its intended value is the audible one -- it over-penalises
repeated tokens, and XTTS answers by rushing, slurring and dropping words. The
conditioning is the other half: 6 seconds of reference instead of 30, and with
``gpt_cond_len == gpt_cond_chunk_len`` no chunk averaging at all, which Coqui documents as
what makes the latents stable.

Optimising away the wrapper is fine. Optimising away the tuning it applied was not.

torch and coqui-tts are imported inside :meth:`load` rather than at module scope, so this
module can be imported — and the rest of the service tested — on a machine that has
neither installed.
"""

from __future__ import annotations

import io
import tempfile
from collections.abc import Iterator
from contextlib import AbstractContextManager, nullcontext
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
        use_half: bool = False,
    ) -> None:
        self.model = model_name
        self.device = device
        self._use_half = use_half and device.startswith("cuda")
        self.audio_format = AudioFormat(
            sample_rate=_SAMPLE_RATE, channels=1, sample_format=SampleFormat.PCM_S16LE
        )
        self.supported_languages = _SUPPORTED_LANGUAGES
        self._chunk_bytes = chunk_bytes
        self._model: Any = None
        self._torch: Any = None
        # Populated from the model config at load; see the module docstring.
        self._voice_settings: dict[str, Any] = {}
        self._inference_settings: dict[str, Any] = {}

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

        try:
            # Loaded to CPU first. Reaching through to the underlying model is what lets
            # conditioning be computed once and PCM be streamed; the high-level `TTS`
            # wrapper is file-in/file-out, which is what forced v1 to recompute
            # conditioning on every call and write every segment through a temp file.
            tts = TTS(model_name=self.model, progress_bar=False)
            model = tts.synthesizer.tts_model

            if self._use_half:
                # Cast *before* moving to the device. Halving after `.to(cuda)` still
                # puts full fp32 weights in VRAM first, so peak usage is unchanged and a
                # 4 GB card runs out part-way through the load -- which is exactly what
                # happened before this order was fixed.
                model = model.half()

            self._model = model.to(self.device)
            self._read_settings(model.config)
        except torch.cuda.OutOfMemoryError as exc:
            free, total = torch.cuda.mem_get_info()
            raise SynthesisError(
                f"not enough VRAM to load {self.model}: "
                f"{free / 1e9:.1f} GB free of {total / 1e9:.1f} GB. "
                "Set TTS_USE_HALF=true, free GPU memory, or use TTS_DEVICE=cpu."
            ) from exc

        log.info(
            "xtts_loaded",
            model=self.model,
            device=self.device,
            precision="fp16" if self._use_half else "fp32",
            **self._inference_settings,
            **self._voice_settings,
            vram_allocated_gb=(
                round(torch.cuda.memory_allocated() / 1e9, 2)
                if self.device.startswith("cuda")
                else None
            ),
        )

    def _read_settings(self, config: Any) -> None:
        """Mirror the settings ``Xtts.synthesize`` applies before calling ``inference``.

        Same keys, same source, read once. The fallbacks are the shipped ``config.json``
        values, not the raw method defaults, so a config missing a key degrades to the
        tuned number rather than back to the untuned one this method exists to avoid.
        """
        voice_defaults = {
            "gpt_cond_len": 30,
            "gpt_cond_chunk_len": 4,
            "max_ref_len": 30,
            "sound_norm_refs": False,
        }
        settings = {key: getattr(config, key, fallback) for key, fallback in voice_defaults.items()}
        # `get_conditioning_latents` spells this one differently from the config.
        settings["max_ref_length"] = settings.pop("max_ref_len")
        self._voice_settings = settings

        inference_defaults = {
            "temperature": 0.75,
            "length_penalty": 1.0,
            "repetition_penalty": 5.0,
            "top_k": 50,
            "top_p": 0.85,
        }
        self._inference_settings = {
            key: getattr(config, key, fallback) for key, fallback in inference_defaults.items()
        }

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
            with self._inference_context():
                gpt_cond_latent, speaker_embedding = self._model.get_conditioning_latents(
                    audio_path=[path], **self._voice_settings
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
            with self._inference_context():
                # `inference`, not `inference_stream`. Streaming decodes the GPT output in
                # fixed-size chunks and vocodes each one separately, crossfading a 1024
                # sample overlap between them -- cheaper to first byte, but it is a
                # different and worse signal than decoding the segment whole.
                #
                # Nothing here needed the earlier bytes earlier. The worker stitches the
                # segments and uploads a finished file; no one is listening while it
                # renders, and progress is already reported per segment. So streaming
                # bought latency that had no consumer and paid for it in quality.
                result = self._model.inference(
                    text,
                    language,
                    gpt_cond_latent,
                    speaker_embedding,
                    speed=speed,
                    # The worker has already split the story into sentence-bounded
                    # segments of a few hundred characters -- that split is what the
                    # progress meter counts. Splitting again here would fragment the
                    # audio a second time, and Coqui's splitter pulls in Spacy as a
                    # dependency the engine otherwise does not need.
                    enable_text_splitting=False,
                    **self._inference_settings,
                )
            yield from self._to_pcm_chunks(result["wav"])
        except Exception as exc:
            raise SynthesisError(f"synthesis failed: {exc}") from exc

    def _inference_context(self) -> AbstractContextManager[Any]:
        """Run inference under autocast when the weights are half precision.

        `model.half()` alone is not enough: Coqui decodes reference audio to float32 and
        then feeds it to half-precision weights, which torch refuses with
        "Input type (torch.cuda.FloatTensor) and weight type (torch.cuda.HalfTensor)
        should be the same". Autocast casts the eligible operations instead of requiring
        every input to be converted by hand.
        """
        if not self._use_half or self._torch is None:
            return nullcontext()
        autocast: AbstractContextManager[Any] = self._torch.autocast(
            device_type="cuda", dtype=self._torch.float16
        )
        return autocast

    def _to_pcm_chunks(self, waveform: Any) -> Iterator[bytes]:
        """Convert a model output waveform to bounded little-endian PCM16 chunks.

        `inference` hands back a numpy array; accepting a tensor too keeps this usable
        from any other call path without a second conversion helper.
        """
        import numpy as np

        if hasattr(waveform, "detach"):
            # `.float()` before `.numpy()`: a half-precision tensor has no direct numpy
            # dtype on every platform, and the conversion below assumes float32 range.
            waveform = waveform.detach().float().to("cpu").numpy()
        samples: np.ndarray[Any, Any] = np.asarray(waveform, dtype="float32").reshape(-1)
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
