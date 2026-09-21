"""gRPC client for `tts-engine`.

Owns three things the pipeline should not have to think about: the cache-miss handshake,
retrying a refusal, and mapping gRPC status codes onto the shared error taxonomy.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass

import grpc

from story2audio_shared.errors import AppError, ErrorCode
from story2audio_shared.logging import get_logger
from tts_worker.audio import PcmAudio, from_pcm_bytes
from tts_worker.pb import tts_pb2, tts_pb2_grpc

log = get_logger(__name__)

#: gRPC status -> our taxonomy. The worker's retry policy and the user's error message
#: both derive from this one mapping rather than from scattered status checks.
_STATUS_TO_ERROR = {
    grpc.StatusCode.UNAVAILABLE: ErrorCode.TTS_UNAVAILABLE,
    grpc.StatusCode.DEADLINE_EXCEEDED: ErrorCode.TTS_TIMEOUT,
    grpc.StatusCode.RESOURCE_EXHAUSTED: ErrorCode.TTS_CAPACITY,
    grpc.StatusCode.CANCELLED: ErrorCode.CANCELLED,
}

#: Supplies a voice's reference sample, fetched lazily from object storage. Called only
#: on a cache miss, so the common path sends no audio at all.
ReferenceLoader = Callable[[], bytes]


@dataclass(frozen=True, slots=True)
class SynthesisResult:
    audio: PcmAudio
    chunk_count: int


class EngineClient:
    """Talks to the TTS engine."""

    def __init__(
        self,
        address: str,
        *,
        timeout_seconds: float,
        capacity_retries: int = 5,
        capacity_backoff_seconds: float = 2.0,
    ) -> None:
        self._timeout = timeout_seconds
        self._capacity_retries = capacity_retries
        self._capacity_backoff = capacity_backoff_seconds
        self._channel = grpc.insecure_channel(
            address,
            options=[
                # Matching the server's keepalives, so a half-open connection to a
                # scale-to-zero GPU host is detected rather than hanging for the full
                # request timeout.
                ("grpc.keepalive_time_ms", 30_000),
                ("grpc.keepalive_timeout_ms", 10_000),
                ("grpc.keepalive_permit_without_calls", 1),
            ],
        )
        self._stub = tts_pb2_grpc.TtsEngineStub(self._channel)

    def synthesize(
        self,
        text: str,
        *,
        voice_id: str,
        reference_loader: ReferenceLoader,
        language: str,
        speed: float,
        request_id: str,
    ) -> SynthesisResult:
        """Render one segment.

        The reference sample is sent only when the engine says it needs it. A
        twelve-segment dialogue job therefore transfers each voice once, not twelve
        times, and the engine computes its conditioning once — the work v1 repeated for
        every single segment.
        """
        try:
            return self._attempt(text, voice_id, None, language, speed, request_id)
        except _SpeakerCacheMissError:
            log.info("tts_speaker_cache_miss", voice_id=voice_id, request_id=request_id)

        reference = reference_loader()
        if not reference:
            raise AppError(
                ErrorCode.AUDIO_ASSEMBLY_FAILED,
                detail=f"voice {voice_id} has no retrievable reference audio",
            )

        try:
            return self._attempt(text, voice_id, reference, language, speed, request_id)
        except _SpeakerCacheMissError as exc:  # pragma: no cover - the engine just asked for this
            raise AppError(
                ErrorCode.TTS_UNAVAILABLE,
                detail=f"engine rejected reference audio for {voice_id}",
            ) from exc

    def _attempt(
        self,
        text: str,
        voice_id: str,
        reference: bytes | None,
        language: str,
        speed: float,
        request_id: str,
    ) -> SynthesisResult:
        request = tts_pb2.SynthesizeRequest(
            text=text,
            speaker=tts_pb2.SpeakerRef(voice_id=voice_id, reference_audio=reference or b""),
            language=language,
            speed=speed,
            request_id=request_id,
        )

        for attempt in range(self._capacity_retries + 1):
            try:
                return self._consume(self._stub.Synthesize(request, timeout=self._timeout))
            except grpc.RpcError as exc:
                code = exc.code()

                if code is grpc.StatusCode.FAILED_PRECONDITION:
                    raise _SpeakerCacheMissError from exc

                if _is_message_too_large(exc):
                    # gRPC overloads RESOURCE_EXHAUSTED: it means both "the server is
                    # busy" (our own abort) and "this message exceeds the size limit"
                    # (the transport). Retrying the second one is pure waste -- the
                    # payload will be exactly as large next time.
                    raise AppError(
                        ErrorCode.AUDIO_ASSEMBLY_FAILED,
                        detail=f"message rejected as too large: {exc.details()}",
                    ) from exc

                if code is grpc.StatusCode.RESOURCE_EXHAUSTED and attempt < self._capacity_retries:
                    # The GPU is busy, not broken. Backing off and retrying is right;
                    # failing the job would waste the story that is already written.
                    delay = self._capacity_backoff * (2**attempt)
                    log.info(
                        "tts_engine_busy",
                        request_id=request_id,
                        attempt=attempt + 1,
                        delay_seconds=round(delay, 1),
                    )
                    time.sleep(delay)
                    continue

                raise self._classify(exc) from exc

        raise AppError(
            ErrorCode.TTS_CAPACITY,
            detail=f"engine stayed busy across {self._capacity_retries} retries",
        )

    def _consume(self, stream: object) -> SynthesisResult:
        """Read the header then the chunks.

        The sequence numbers are checked rather than trusted: gRPC orders a stream, but a
        gap means frames were lost, and silently concatenating what arrived would produce
        audio that is quietly too short.
        """
        sample_rate: int | None = None
        chunks: list[bytes] = []
        expected = 0

        for response in stream:  # type: ignore[attr-defined]
            payload = response.WhichOneof("payload")
            if payload == "info":
                sample_rate = int(response.info.sample_rate)
                continue
            if payload != "chunk":
                continue

            expected += 1
            if response.chunk.sequence != expected:
                raise AppError(
                    ErrorCode.AUDIO_ASSEMBLY_FAILED,
                    detail=(
                        f"audio stream gap: expected chunk {expected}, "
                        f"got {response.chunk.sequence}"
                    ),
                )
            chunks.append(response.chunk.pcm)

        if sample_rate is None:
            raise AppError(
                ErrorCode.AUDIO_ASSEMBLY_FAILED, detail="engine sent no audio format header"
            )

        return SynthesisResult(
            audio=from_pcm_bytes(b"".join(chunks), sample_rate), chunk_count=len(chunks)
        )

    @staticmethod
    def _classify(exc: grpc.RpcError) -> AppError:
        code = _STATUS_TO_ERROR.get(exc.code(), ErrorCode.TTS_UNAVAILABLE)
        return AppError(code, detail=f"tts-engine returned {exc.code()}: {exc.details()}")

    def close(self) -> None:
        self._channel.close()


def _is_message_too_large(exc: grpc.RpcError) -> bool:
    """Distinguish a transport size rejection from a genuinely busy server."""
    if exc.code() is not grpc.StatusCode.RESOURCE_EXHAUSTED:
        return False
    return "larger than max" in (exc.details() or "")


class _SpeakerCacheMissError(Exception):
    """The engine does not hold this voice's embedding yet. Internal control flow only."""
