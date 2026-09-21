"""The TTS gRPC service.

Driven over a real gRPC channel against a real server on a loopback port — the
servicer, the streaming protocol, the status codes and the thread pool are all
exercised as deployed. Only the model is substituted, by the stub backend, which is
precisely what that backend exists for.
"""

from __future__ import annotations

import io
import struct
import threading
import wave
from collections.abc import Iterator
from concurrent import futures
from typing import Any

import grpc
import pytest

from tts_engine.cache import SpeakerCache
from tts_engine.pb import tts_pb2, tts_pb2_grpc
from tts_engine.server import InferenceSlots, TtsEngineServicer, create_server
from tts_engine.settings import Backend, TtsEngineSettings
from tts_engine.synthesis import build_backend
from tts_engine.synthesis.base import SpeakerEmbedding, SynthesisError
from tts_engine.synthesis.stub import StubBackend

TEXT = "The lighthouse keeper climbed the stairs in the dark."


def make_wav(duration_seconds: float, *, sample_rate: int = 22_050, tone: int = 220) -> bytes:
    """A real, decodable mono PCM16 WAV."""
    frames = max(1, int(duration_seconds * sample_rate))
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(
            b"".join(
                struct.pack("<h", int(8000 * ((index * tone) % 100) / 100))
                for index in range(frames)
            )
        )
    return buffer.getvalue()


REFERENCE = make_wav(8.0)
OTHER_REFERENCE = make_wav(8.0, tone=330)


# --- Fixtures ------------------------------------------------------------------------------------


@pytest.fixture
def settings() -> TtsEngineSettings:
    return TtsEngineSettings(
        tts_backend=Backend.STUB,
        # The port in settings is unused here: every test binds 127.0.0.1:0 directly
        # so the OS picks a free one and the suite can run in parallel.
        tts_max_concurrent_inferences=1,
        speaker_cache_size=4,
        audio_chunk_bytes=4096,
        environment="local",
    )


@pytest.fixture
def stub(settings: TtsEngineSettings) -> Iterator[tts_pb2_grpc.TtsEngineStub]:
    """A client talking to a real server on an ephemeral loopback port."""
    backend = build_backend(settings)
    backend.load()

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=8))
    tts_pb2_grpc.add_TtsEngineServicer_to_server(TtsEngineServicer(backend, settings), server)
    port = server.add_insecure_port("127.0.0.1:0")
    server.start()

    with grpc.insecure_channel(f"127.0.0.1:{port}") as channel:
        yield tts_pb2_grpc.TtsEngineStub(channel)

    server.stop(0).wait()
    backend.close()


def synthesize(
    stub: tts_pb2_grpc.TtsEngineStub,
    *,
    text: str = TEXT,
    voice_id: str = "voice-1",
    reference: bytes | None = REFERENCE,
    language: str = "en",
    speed: float = 1.0,
) -> list[tts_pb2.SynthesizeResponse]:
    speaker = tts_pb2.SpeakerRef(voice_id=voice_id, reference_audio=reference or b"")
    request = tts_pb2.SynthesizeRequest(
        text=text, speaker=speaker, language=language, speed=speed, request_id="test"
    )
    return list(stub.Synthesize(request))


# --- The stream contract -------------------------------------------------------------------------


def test_the_first_message_describes_the_audio_format(
    stub: tts_pb2_grpc.TtsEngineStub,
) -> None:
    """The caller needs the format before the first sample, to write a WAV header."""
    responses = synthesize(stub)

    assert responses[0].WhichOneof("payload") == "info"
    info = responses[0].info
    assert info.sample_rate == 24_000
    assert info.channels == 1
    assert info.format == tts_pb2.SAMPLE_FORMAT_PCM_S16LE


def test_audio_arrives_as_chunks_after_the_header(stub: tts_pb2_grpc.TtsEngineStub) -> None:
    responses = synthesize(stub)
    chunks = [r.chunk for r in responses if r.WhichOneof("payload") == "chunk"]

    assert chunks
    assert all(chunk.pcm for chunk in chunks)


def test_chunk_sequence_numbers_start_at_one_and_are_contiguous(
    stub: tts_pb2_grpc.TtsEngineStub,
) -> None:
    """Not for reassembly -- so a truncated stream is detectable."""
    chunks = [r.chunk for r in synthesize(stub) if r.WhichOneof("payload") == "chunk"]

    assert [chunk.sequence for chunk in chunks] == list(range(1, len(chunks) + 1))


def test_chunks_stay_well_under_the_default_message_limit(
    stub: tts_pb2_grpc.TtsEngineStub, settings: TtsEngineSettings
) -> None:
    """v1 shipped whole WAVs in one field and had to raise both ends to 100 MB."""
    chunks = [r.chunk for r in synthesize(stub) if r.WhichOneof("payload") == "chunk"]

    assert all(len(chunk.pcm) <= settings.audio_chunk_bytes for chunk in chunks)


def test_the_rendered_audio_is_a_plausible_length(stub: tts_pb2_grpc.TtsEngineStub) -> None:
    chunks = [r.chunk for r in synthesize(stub) if r.WhichOneof("payload") == "chunk"]
    total = sum(len(chunk.pcm) for chunk in chunks)

    # 24 kHz mono PCM16 -> 48000 bytes per second.
    seconds = total / 48_000
    assert 1.0 < seconds < 30.0


def test_pcm_decodes_as_signed_16_bit(stub: tts_pb2_grpc.TtsEngineStub) -> None:
    """Prove the bytes are audio, not an opaque blob that merely has the right length."""
    chunks = [r.chunk for r in synthesize(stub) if r.WhichOneof("payload") == "chunk"]
    pcm = b"".join(chunk.pcm for chunk in chunks)

    assert len(pcm) % 2 == 0
    samples = struct.unpack(f"<{len(pcm) // 2}h", pcm)
    assert any(sample != 0 for sample in samples)
    assert max(abs(sample) for sample in samples) < 32_768


def test_speed_changes_the_rendered_duration(stub: tts_pb2_grpc.TtsEngineStub) -> None:
    def total_bytes(speed: float) -> int:
        responses = synthesize(stub, speed=speed, voice_id=f"voice-{speed}")
        return sum(len(r.chunk.pcm) for r in responses if r.WhichOneof("payload") == "chunk")

    assert total_bytes(2.0) < total_bytes(1.0)


# --- Speaker caching -----------------------------------------------------------------------------


def test_a_known_voice_needs_no_reference_audio(stub: tts_pb2_grpc.TtsEngineStub) -> None:
    """The point of the cache: a twelve-segment job sends the sample once, not twelve times."""
    synthesize(stub, voice_id="cached-voice", reference=REFERENCE)

    responses = synthesize(stub, voice_id="cached-voice", reference=None)

    assert any(r.WhichOneof("payload") == "chunk" for r in responses)


def test_an_unknown_voice_without_audio_is_a_failed_precondition(
    stub: tts_pb2_grpc.TtsEngineStub,
) -> None:
    """The caller's signal to fetch the sample from storage and retry."""
    with pytest.raises(grpc.RpcError) as info:
        synthesize(stub, voice_id="never-seen", reference=None)

    assert info.value.code() is grpc.StatusCode.FAILED_PRECONDITION
    assert "reference_audio" in (info.value.details() or "")


def test_embedding_is_reported_as_cached_on_the_second_call(
    stub: tts_pb2_grpc.TtsEngineStub,
) -> None:
    first = stub.EmbedSpeaker(
        tts_pb2.EmbedSpeakerRequest(voice_id="warm", reference_audio=REFERENCE)
    )
    second = stub.EmbedSpeaker(tts_pb2.EmbedSpeakerRequest(voice_id="warm"))

    assert first.computed is True
    assert second.computed is False
    assert second.reference_duration_seconds == pytest.approx(8.0, abs=0.1)


def test_embedding_an_unknown_voice_without_audio_is_rejected(
    stub: tts_pb2_grpc.TtsEngineStub,
) -> None:
    with pytest.raises(grpc.RpcError) as info:
        stub.EmbedSpeaker(tts_pb2.EmbedSpeakerRequest(voice_id="cold"))

    assert info.value.code() is grpc.StatusCode.INVALID_ARGUMENT


def test_unusable_reference_audio_is_rejected(stub: tts_pb2_grpc.TtsEngineStub) -> None:
    with pytest.raises(grpc.RpcError) as info:
        stub.EmbedSpeaker(tts_pb2.EmbedSpeakerRequest(voice_id="bad", reference_audio=b"\x00" * 16))

    assert info.value.code() is grpc.StatusCode.INVALID_ARGUMENT


def test_different_voices_render_differently(stub: tts_pb2_grpc.TtsEngineStub) -> None:
    """So a dialogue job can be shown to have used two voices, not one twice."""

    def audio(voice_id: str, reference: bytes) -> bytes:
        responses = synthesize(stub, voice_id=voice_id, reference=reference)
        return b"".join(r.chunk.pcm for r in responses if r.WhichOneof("payload") == "chunk")

    assert audio("narrator", REFERENCE) != audio("dialogue", OTHER_REFERENCE)


def test_the_same_voice_and_text_render_identically(
    stub: tts_pb2_grpc.TtsEngineStub,
) -> None:
    """Determinism is what lets the other tests assert on content rather than length."""

    def audio() -> bytes:
        responses = synthesize(stub, voice_id="stable", reference=REFERENCE)
        return b"".join(r.chunk.pcm for r in responses if r.WhichOneof("payload") == "chunk")

    assert audio() == audio()


# --- Validation ----------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("kwargs", "expected_fragment"),
    [
        ({"text": "   "}, "text"),
        ({"language": "klingon"}, "language"),
        ({"speed": 0.0}, "speed"),
        ({"speed": -1.0}, "speed"),
    ],
)
def test_invalid_requests_are_rejected(
    stub: tts_pb2_grpc.TtsEngineStub, kwargs: dict[str, Any], expected_fragment: str
) -> None:
    with pytest.raises(grpc.RpcError) as info:
        synthesize(stub, **kwargs)

    assert info.value.code() is grpc.StatusCode.INVALID_ARGUMENT
    assert expected_fragment in (info.value.details() or "")


def test_a_missing_voice_id_is_rejected(stub: tts_pb2_grpc.TtsEngineStub) -> None:
    with pytest.raises(grpc.RpcError) as info:
        synthesize(stub, voice_id="")

    assert info.value.code() is grpc.StatusCode.INVALID_ARGUMENT


def test_internal_failures_do_not_leak_their_cause(
    settings: TtsEngineSettings,
) -> None:
    """The engine renders untrusted text; its exceptions must not reach the caller."""

    class ExplodingBackend(StubBackend):
        def synthesize(self, text: str, embedding: SpeakerEmbedding, **_: Any) -> Iterator[bytes]:
            raise SynthesisError("CUDA out of memory at /opt/conda/lib/python3.11/xtts.py:214")
            yield b""  # pragma: no cover - unreachable, keeps this a generator

    backend = ExplodingBackend()
    backend.load()
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    tts_pb2_grpc.add_TtsEngineServicer_to_server(TtsEngineServicer(backend, settings), server)
    port = server.add_insecure_port("127.0.0.1:0")
    server.start()

    try:
        with grpc.insecure_channel(f"127.0.0.1:{port}") as channel:
            client = tts_pb2_grpc.TtsEngineStub(channel)
            with pytest.raises(grpc.RpcError) as info:
                synthesize(client)

        assert info.value.code() is grpc.StatusCode.INTERNAL
        details = info.value.details() or ""
        assert "CUDA" not in details
        assert "/opt/conda" not in details
    finally:
        server.stop(0).wait()


# --- Concurrency ---------------------------------------------------------------------------------


def test_excess_demand_is_refused_rather_than_queued(settings: TtsEngineSettings) -> None:
    """v1's global mutex queued callers invisibly and without bound.

    Refusing makes saturation observable: the caller backs off, and the rejection is a
    metric rather than unexplained latency.
    """
    release = threading.Event()
    entered = threading.Event()

    class BlockingBackend(StubBackend):
        def synthesize(self, text: str, embedding: SpeakerEmbedding, **_: Any) -> Iterator[bytes]:
            entered.set()
            release.wait(timeout=10)
            yield b"\x00\x00" * 128

    backend = BlockingBackend()
    backend.load()
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=8))
    tts_pb2_grpc.add_TtsEngineServicer_to_server(TtsEngineServicer(backend, settings), server)
    port = server.add_insecure_port("127.0.0.1:0")
    server.start()

    try:
        with grpc.insecure_channel(f"127.0.0.1:{port}") as channel:
            client = tts_pb2_grpc.TtsEngineStub(channel)

            holder = threading.Thread(target=lambda: _drain(client, "a"), daemon=True)
            holder.start()
            assert entered.wait(timeout=5), "first request never reached the backend"

            with pytest.raises(grpc.RpcError) as info:
                _drain(client, "b")

            assert info.value.code() is grpc.StatusCode.RESOURCE_EXHAUSTED

            release.set()
            holder.join(timeout=10)
    finally:
        release.set()
        server.stop(0).wait()


def _drain(client: tts_pb2_grpc.TtsEngineStub, voice_id: str) -> None:
    list(
        client.Synthesize(
            tts_pb2.SynthesizeRequest(
                text=TEXT,
                speaker=tts_pb2.SpeakerRef(voice_id=voice_id, reference_audio=REFERENCE),
                language="en",
                speed=1.0,
            )
        )
    )


def test_a_slot_is_returned_after_each_request(stub: tts_pb2_grpc.TtsEngineStub) -> None:
    """A leaked slot would wedge a single-slot engine permanently after one request."""
    for index in range(3):
        synthesize(stub, voice_id=f"seq-{index}")

    info = stub.GetInfo(tts_pb2.GetInfoRequest())
    assert info.in_flight == 0


def test_a_slot_is_returned_even_when_synthesis_fails(
    settings: TtsEngineSettings,
) -> None:
    class ExplodingBackend(StubBackend):
        def synthesize(self, text: str, embedding: SpeakerEmbedding, **_: Any) -> Iterator[bytes]:
            raise SynthesisError("boom")
            yield b""  # pragma: no cover

    backend = ExplodingBackend()
    backend.load()
    servicer = TtsEngineServicer(backend, settings)
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    tts_pb2_grpc.add_TtsEngineServicer_to_server(servicer, server)
    port = server.add_insecure_port("127.0.0.1:0")
    server.start()

    try:
        with grpc.insecure_channel(f"127.0.0.1:{port}") as channel:
            client = tts_pb2_grpc.TtsEngineStub(channel)
            for _ in range(3):
                with pytest.raises(grpc.RpcError):
                    synthesize(client)

            assert client.GetInfo(tts_pb2.GetInfoRequest()).in_flight == 0
    finally:
        server.stop(0).wait()


# --- GetInfo -------------------------------------------------------------------------------------


def test_get_info_reports_capacity_and_capability(stub: tts_pb2_grpc.TtsEngineStub) -> None:
    info = stub.GetInfo(tts_pb2.GetInfoRequest())

    assert info.backend == "stub"
    assert info.ready is True
    assert info.max_concurrent == 1
    assert "en" in info.languages


def test_get_info_reports_the_cache_filling(stub: tts_pb2_grpc.TtsEngineStub) -> None:
    assert stub.GetInfo(tts_pb2.GetInfoRequest()).cached_speakers == 0

    stub.EmbedSpeaker(tts_pb2.EmbedSpeakerRequest(voice_id="v1", reference_audio=REFERENCE))
    stub.EmbedSpeaker(tts_pb2.EmbedSpeakerRequest(voice_id="v2", reference_audio=REFERENCE))

    assert stub.GetInfo(tts_pb2.GetInfoRequest()).cached_speakers == 2


def test_requests_are_refused_while_the_model_is_loading(
    settings: TtsEngineSettings,
) -> None:
    """Readiness, not liveness: a cold start answers, it does not refuse the connection."""
    backend = StubBackend()  # deliberately not loaded
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    tts_pb2_grpc.add_TtsEngineServicer_to_server(TtsEngineServicer(backend, settings), server)
    port = server.add_insecure_port("127.0.0.1:0")
    server.start()

    try:
        with grpc.insecure_channel(f"127.0.0.1:{port}") as channel:
            client = tts_pb2_grpc.TtsEngineStub(channel)

            assert client.GetInfo(tts_pb2.GetInfoRequest()).ready is False
            with pytest.raises(grpc.RpcError) as info:
                synthesize(client)
            assert info.value.code() is grpc.StatusCode.UNAVAILABLE
    finally:
        server.stop(0).wait()


# --- Server construction -------------------------------------------------------------------------


def test_the_server_sets_no_message_size_overrides(settings: TtsEngineSettings) -> None:
    """The 100 MB limits v1 needed are exactly what streaming removes."""
    backend = build_backend(settings)
    server, _ = create_server(backend, settings)
    server.stop(0).wait()


def test_the_stub_backend_is_refused_in_production() -> None:
    """Serving a tone as a finished story would be worse than failing."""
    with pytest.raises(RuntimeError, match="not permitted in production"):
        build_backend(TtsEngineSettings(tts_backend=Backend.STUB, environment="production"))


# --- Speaker cache -------------------------------------------------------------------------------


def _embedding(voice_id: str) -> SpeakerEmbedding:
    return SpeakerEmbedding(voice_id=voice_id, payload=220.0, reference_duration_seconds=8.0)


def test_the_cache_evicts_least_recently_used() -> None:
    cache = SpeakerCache(max_entries=2)
    cache.put(_embedding("a"))
    cache.put(_embedding("b"))

    cache.get("a")  # `a` becomes the most recently used
    cache.put(_embedding("c"))

    assert "a" in cache
    assert "c" in cache
    assert "b" not in cache


def test_the_cache_is_bounded() -> None:
    cache = SpeakerCache(max_entries=3)
    for index in range(50):
        cache.put(_embedding(f"voice-{index}"))

    assert len(cache) == 3


def test_the_cache_counts_hits_and_misses() -> None:
    cache = SpeakerCache(max_entries=4)
    cache.put(_embedding("a"))

    cache.get("a")
    cache.get("missing")

    assert cache.hits == 1
    assert cache.misses == 1


def test_the_cache_is_safe_under_concurrent_use() -> None:
    """gRPC dispatches handlers on a thread pool, so this is not hypothetical."""
    cache = SpeakerCache(max_entries=16)
    errors: list[BaseException] = []

    def hammer(worker: int) -> None:
        try:
            for index in range(200):
                cache.put(_embedding(f"w{worker}-{index}"))
                cache.get(f"w{worker}-{index}")
        except BaseException as exc:  # noqa: BLE001 - recorded and re-raised in the main thread
            errors.append(exc)

    threads = [threading.Thread(target=hammer, args=(worker,)) for worker in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)

    assert not errors
    assert len(cache) == 16


# --- Inference slots -----------------------------------------------------------------------------


def test_slots_are_bounded_and_returned() -> None:
    slots = InferenceSlots(2)

    assert slots.try_acquire()
    assert slots.try_acquire()
    assert not slots.try_acquire()
    assert slots.in_flight == 2

    slots.release()
    assert slots.try_acquire()
    assert slots.in_flight == 2
