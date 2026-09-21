# ADR-0004 — gRPC survives, but only between `tts-worker` and `tts-engine`

**Status:** Accepted (Phase 4)

## Context

v1 exposed gRPC as its client-facing API. The Streamlit app and a Flask REST proxy both
called a unary `GenerateStory` RPC that returned the finished WAV as a protobuf `bytes`
field — which is why both ends had to raise `grpc.max_receive_message_length` to 100 MB,
and why everything was buffered in memory at least three times.

The rebuild replaces that public surface with REST plus a WebSocket. The question was
whether gRPC has any remaining place, or whether removing it entirely is simpler.

## Decision

Keep it, in exactly one place: the internal link between `tts-worker` and `tts-engine`.

That boundary exists because the two halves have genuinely different resource profiles.
The worker segments the story, stitches audio with `pydub`, encodes, and uploads — all
CPU and network, and none of it should occupy a GPU. The engine holds XTTS in VRAM and
does nothing else. Separating them means one GPU serves several workers, and the GPU
process can be a scale-to-zero container while the workers stay warm.

Across that boundary the traffic is a high-frequency stream of binary audio frames
between two services in one repository. That is the workload gRPC is actually good at:
a schema both sides compile against, server-streaming, no JSON encoding of PCM, and
backpressure from HTTP/2 flow control.

## Consequences

- `proto/tts/v1/tts.proto` is the contract, and it is the only thing the two services
  share — `tts-engine` imports no Python from the rest of the workspace.
- Stubs are generated at build time rather than committed, so a stale copy cannot drift
  from the `.proto`.
- Audio streams as an `AudioInfo` header followed by bounded `AudioChunk`s. No
  message-size override anywhere; removing that hack is half the point.
- The engine holds **no object-storage credentials**. It caches speaker embeddings by
  voice id and answers `FAILED_PRECONDITION` on a miss; the worker fetches the sample and
  retries. The process running a large model over untrusted text keeps the smallest
  blast radius available.
- Saturation is refused with `RESOURCE_EXHAUSTED`, not queued. v1's module-global mutex
  queued callers invisibly and without bound, which made its "concurrent processing"
  claim untrue and its latency unexplainable.
- Cost: a second serialisation format in the codebase, and a build step. Both are paid
  once and confined to one boundary.

## Rejected

- **REST between worker and engine.** Base64 or multipart for PCM, no streaming without
  hand-rolled chunked encoding, and no shared schema. Strictly worse for this traffic.
- **Removing the boundary — synthesis inside the worker.** This is v1's shape. It ties
  GPU memory to a process doing CPU work, makes the GPU un-shareable, and makes
  scale-to-zero impossible, which is the whole cost strategy in `docs/04`.
- **Keeping gRPC public as well.** Browsers need grpc-web and a proxy, and the API would
  gain a second contract to version for no benefit.

## Revisit when

The engine is replaced by a hosted TTS API for good, with no self-hosted path retained.
At that point the `TTSProvider` interface remains and this contract does not.
