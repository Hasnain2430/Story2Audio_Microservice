"""The gRPC service.

Owns three things the backend deliberately does not: the speaker cache, the concurrency
limit, and the translation of failures into gRPC statuses. The backend renders audio;
this decides who is allowed to ask and what the caller is told when the answer is no.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Iterator
from concurrent import futures

import grpc
from grpc_health.v1 import health, health_pb2, health_pb2_grpc

from tts_engine.cache import SpeakerCache
from tts_engine.logging import get_logger
from tts_engine.pb import tts_pb2, tts_pb2_grpc
from tts_engine.settings import TtsEngineSettings
from tts_engine.synthesis.base import (
    SampleFormat,
    SpeakerEmbedding,
    SynthesisBackend,
    SynthesisError,
)

log = get_logger(__name__)

_SERVICE_NAME = "story2audio.tts.v1.TtsEngine"

_FORMAT_TO_PROTO = {
    SampleFormat.PCM_S16LE: tts_pb2.SAMPLE_FORMAT_PCM_S16LE,
    SampleFormat.PCM_F32LE: tts_pb2.SAMPLE_FORMAT_PCM_F32LE,
}


class InferenceSlots:
    """Bounded admission to the model.

    v1 had a module-global ``threading.Lock`` around every synthesis call. It was
    *correct* — a single XTTS instance is not thread-safe — but it made the declared
    concurrency fictional: five gRPC workers queued behind one lock, unbounded and
    invisible, so the fifth caller simply waited with no way to know it was waiting.

    Here excess demand is refused immediately with ``RESOURCE_EXHAUSTED`` instead of
    silently queued. The caller is a worker that can back off and retry, and refusing is
    what makes saturation observable rather than merely slow.
    """

    def __init__(self, limit: int) -> None:
        self._semaphore = threading.BoundedSemaphore(limit)
        self._limit = limit
        self._in_flight = 0
        self._lock = threading.Lock()

    @property
    def limit(self) -> int:
        return self._limit

    @property
    def in_flight(self) -> int:
        with self._lock:
            return self._in_flight

    def try_acquire(self) -> bool:
        if not self._semaphore.acquire(blocking=False):
            return False
        with self._lock:
            self._in_flight += 1
        return True

    def release(self) -> None:
        with self._lock:
            self._in_flight -= 1
        self._semaphore.release()


class TtsEngineServicer(tts_pb2_grpc.TtsEngineServicer):
    """Implements the TTS contract."""

    def __init__(
        self,
        backend: SynthesisBackend,
        settings: TtsEngineSettings,
    ) -> None:
        self._backend = backend
        self._settings = settings
        self._cache = SpeakerCache(settings.speaker_cache_size)
        self._slots = InferenceSlots(settings.tts_max_concurrent_inferences)

    # --- Synthesize -------------------------------------------------------------------------

    def Synthesize(  # noqa: N802 - the name is fixed by the generated base class
        self,
        request: tts_pb2.SynthesizeRequest,
        context: grpc.ServicerContext,
    ) -> Iterator[tts_pb2.SynthesizeResponse]:
        request_id = request.request_id or "-"

        if not self._backend.ready:
            context.abort(grpc.StatusCode.UNAVAILABLE, "model is still loading")

        if not request.text.strip():
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, "text must not be empty")
        if request.language not in self._backend.supported_languages:
            context.abort(
                grpc.StatusCode.INVALID_ARGUMENT, f"unsupported language {request.language!r}"
            )
        if request.speed <= 0:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, "speed must be positive")

        embedding = self._resolve_speaker(request.speaker, context)

        if not self._slots.try_acquire():
            # Refused, not queued. The caller retries with backoff, and a saturated GPU
            # shows up as a metric instead of as unexplained latency.
            context.abort(
                grpc.StatusCode.RESOURCE_EXHAUSTED,
                f"all {self._slots.limit} inference slots are busy",
            )

        started = time.monotonic()
        total_bytes = 0
        sequence = 0
        try:
            yield tts_pb2.SynthesizeResponse(info=self._audio_info())

            for pcm in self._backend.synthesize(
                request.text,
                embedding,
                language=request.language,
                speed=request.speed,
            ):
                if not context.is_active():
                    # The caller hung up -- a cancelled job, most likely. Stop rendering
                    # rather than finishing work nobody will collect. This is what makes
                    # cancellation actually free the GPU.
                    log.info("synthesize_client_gone", request_id=request_id, sequence=sequence)
                    return

                sequence += 1
                total_bytes += len(pcm)
                yield tts_pb2.SynthesizeResponse(
                    chunk=tts_pb2.AudioChunk(pcm=pcm, sequence=sequence)
                )

        except SynthesisError as exc:
            log.warning("synthesize_failed", request_id=request_id, error=str(exc))
            context.abort(grpc.StatusCode.INTERNAL, "synthesis failed")
        finally:
            self._slots.release()

        log.info(
            "synthesize_complete",
            request_id=request_id,
            voice_id=request.speaker.voice_id,
            chunks=sequence,
            audio_seconds=round(self._backend.audio_format.duration_seconds(total_bytes), 2),
            elapsed_seconds=round(time.monotonic() - started, 2),
        )

    # --- EmbedSpeaker -----------------------------------------------------------------------

    def EmbedSpeaker(  # noqa: N802 - the name is fixed by the generated base class
        self,
        request: tts_pb2.EmbedSpeakerRequest,
        context: grpc.ServicerContext,
    ) -> tts_pb2.EmbedSpeakerResponse:
        if not self._backend.ready:
            context.abort(grpc.StatusCode.UNAVAILABLE, "model is still loading")
        if not request.voice_id:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, "voice_id is required")

        cached = self._cache.get(request.voice_id)
        if cached is not None:
            return tts_pb2.EmbedSpeakerResponse(
                voice_id=request.voice_id,
                computed=False,
                reference_duration_seconds=cached.reference_duration_seconds,
            )

        if not request.reference_audio:
            context.abort(
                grpc.StatusCode.INVALID_ARGUMENT,
                "reference_audio is required to embed an unknown voice",
            )

        embedding = self._embed(request.voice_id, request.reference_audio, context)
        return tts_pb2.EmbedSpeakerResponse(
            voice_id=request.voice_id,
            computed=True,
            reference_duration_seconds=embedding.reference_duration_seconds,
        )

    # --- GetInfo ----------------------------------------------------------------------------

    def GetInfo(  # noqa: N802 - the name is fixed by the generated base class
        self,
        request: tts_pb2.GetInfoRequest,
        context: grpc.ServicerContext,
    ) -> tts_pb2.GetInfoResponse:
        return tts_pb2.GetInfoResponse(
            backend=self._backend.name,
            model=self._backend.model,
            device=self._backend.device,
            ready=self._backend.ready,
            max_concurrent=self._slots.limit,
            in_flight=self._slots.in_flight,
            languages=sorted(self._backend.supported_languages),
            cached_speakers=len(self._cache),
        )

    # --- Internals --------------------------------------------------------------------------

    def _audio_info(self) -> tts_pb2.AudioInfo:
        audio_format = self._backend.audio_format
        return tts_pb2.AudioInfo(
            sample_rate=audio_format.sample_rate,
            channels=audio_format.channels,
            format=_FORMAT_TO_PROTO[audio_format.sample_format],
        )

    def _resolve_speaker(
        self, speaker: tts_pb2.SpeakerRef, context: grpc.ServicerContext
    ) -> SpeakerEmbedding:
        """Find the speaker's conditioning, computing it only on a cache miss.

        A caller that sent no reference audio for an unknown voice gets
        ``FAILED_PRECONDITION``, which is its signal to fetch the sample from object
        storage and retry. The engine holds no storage credentials of its own: it is the
        process running a large model over untrusted text, so its blast radius stays
        small.
        """
        if not speaker.voice_id:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, "speaker.voice_id is required")

        cached = self._cache.get(speaker.voice_id)
        if cached is not None:
            return cached

        if not speaker.reference_audio:
            context.abort(
                grpc.StatusCode.FAILED_PRECONDITION,
                f"voice {speaker.voice_id} is not cached; resend with reference_audio",
            )

        return self._embed(speaker.voice_id, speaker.reference_audio, context)

    def _embed(
        self, voice_id: str, reference_audio: bytes, context: grpc.ServicerContext
    ) -> SpeakerEmbedding:
        started = time.monotonic()
        try:
            embedding = self._backend.embed(voice_id, reference_audio)
        except SynthesisError as exc:
            log.warning("embed_failed", voice_id=voice_id, error=str(exc))
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, "reference audio could not be used")

        self._cache.put(embedding)
        log.info(
            "speaker_embedded",
            voice_id=voice_id,
            elapsed_seconds=round(time.monotonic() - started, 3),
            cached_speakers=len(self._cache),
        )
        return embedding


def create_server(
    backend: SynthesisBackend, settings: TtsEngineSettings
) -> tuple[grpc.Server, TtsEngineServicer]:
    """Build the gRPC server.

    No message-size overrides. v1 shipped whole rendered WAVs inside a single protobuf
    field and had to raise both ends to 100 MB; here audio is streamed in bounded chunks
    and the defaults are correct.
    """
    server = grpc.server(
        futures.ThreadPoolExecutor(
            max_workers=settings.grpc_max_workers, thread_name_prefix="tts-engine"
        ),
        options=[
            # Keepalives so a half-open connection to a scale-to-zero GPU host is
            # detected rather than hanging a worker for the full request timeout.
            ("grpc.keepalive_time_ms", 30_000),
            ("grpc.keepalive_timeout_ms", 10_000),
            ("grpc.keepalive_permit_without_calls", 1),
        ],
    )

    servicer = TtsEngineServicer(backend, settings)
    tts_pb2_grpc.add_TtsEngineServicer_to_server(servicer, server)

    # Standard health protocol, so any orchestrator's probe works without bespoke glue.
    health_servicer = health.HealthServicer()
    health_pb2_grpc.add_HealthServicer_to_server(health_servicer, server)
    health_servicer.set(_SERVICE_NAME, health_pb2.HealthCheckResponse.NOT_SERVING)
    server.add_insecure_port(f"{settings.tts_engine_host}:{settings.tts_engine_port}")

    # Attached so the entrypoint can flip to SERVING once the model is resident.
    server.health_servicer = health_servicer  # type: ignore[attr-defined]
    return server, servicer


def mark_serving(server: grpc.Server, *, serving: bool) -> None:
    """Flip the health status once the model is loaded, or on the way down."""
    status = (
        health_pb2.HealthCheckResponse.SERVING
        if serving
        else health_pb2.HealthCheckResponse.NOT_SERVING
    )
    server.health_servicer.set(_SERVICE_NAME, status)  # type: ignore[attr-defined]
