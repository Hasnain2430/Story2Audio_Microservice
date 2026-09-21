"""Backend selection."""

from __future__ import annotations

from tts_engine.logging import get_logger
from tts_engine.settings import Backend, TtsEngineSettings
from tts_engine.synthesis.base import SynthesisBackend
from tts_engine.synthesis.stub import StubBackend
from tts_engine.synthesis.xtts import XttsBackend

log = get_logger(__name__)


def build_backend(settings: TtsEngineSettings) -> SynthesisBackend:
    """Construct the configured backend.

    The stub is refused in production. It renders a tone rather than speech, and
    returning that as a finished story would be worse than failing outright -- the
    failure is at least visible.
    """
    match settings.tts_backend:
        case Backend.XTTS:
            return XttsBackend(
                model_name=settings.tts_model_name,
                device=settings.tts_device,
                chunk_bytes=settings.audio_chunk_bytes,
            )

        case Backend.STUB:
            if settings.is_production:
                raise RuntimeError(
                    "TTS_BACKEND=stub is not permitted in production: it renders a tone, not speech"
                )
            log.warning(
                "using_stub_tts_backend",
                reason="development or test configuration",
                note="renders a tone, not speech",
            )
            return StubBackend(chunk_bytes=settings.audio_chunk_bytes)
