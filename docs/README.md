# Story2Audio — v2 Rebuild Docs

Working documents for the v2 rebuild. Branch: `v2-rebuild`.

| Doc | What it covers |
|---|---|
| [01 — System Analysis](./01-system-analysis.md) | Full audit of the v1 code: topology, the 10-minute blocking RPC, 20 catalogued defects, and what's worth keeping |
| [02 — Target Architecture](./02-target-architecture.md) | v2 service map, why each boundary exists, job lifecycle, API surface, data model, frontend architecture |
| [03 — Implementation Plan](./03-implementation-plan.md) | Ten phases, each ending in a runnable state; sequencing, definition of done, risks |
| [04 — Deployment and Cost](./04-deployment-and-cost.md) | Two deployment shapes, the TTS/GPU cost decision, per-story cost model, spend guardrails |
| [ADRs](./adr/) | One record per non-obvious decision: what was chosen, what was rejected, and what it costs |

## The short version

**Problem.** v1 is a monolith in microservice clothing. One gRPC process holds XTTS, an
emotion classifier and a translator, answers a single unary RPC that blocks for up to ten
minutes, serialises all synthesis on one global mutex, and shares one mutable
`chat_history` list across every user.

**Fix.** Split it along the lines that actually differ in scaling and failure behaviour:

```
React/TS  →  FastAPI gateway  →  Redis queue  →  story-worker (LLM)
                   ↑                                    ↓
              WebSocket ←── Redis pub/sub ──── tts-worker → tts-engine (gRPC, GPU)
                                                              ↓
                                                    object storage → presigned URL
```

The API returns a job id in under 200 ms. Story text streams to the browser in about two
seconds. Segment progress is real, not a spinner. Refreshing the page loses nothing.
Cancellation actually frees the GPU. The queue is the concurrency control, and it is
observable.

**What stays.** The product, the prompt library, the 16 reference voices, the
narration/dialogue segmentation and the audio stitching logic, Docker, and gRPC — kept
where it belongs, as the internal streaming contract between `tts-worker` and the GPU
engine.

**What goes.** Streamlit, the global `chat_history`, the MarianMT translation hop, the
no-op `emotion` argument, the `[PARA_LEVEL:...]` prompt sentinel, filesystem paths on the
wire, and 100 MB audio blobs inside protobuf messages.

**Cost.** Everything except the GPU fits in free tiers. The GPU scales to zero, and the
async design absorbs the cold start for free. Idle ≈ $0; roughly $0.02–0.07 per generated
story.

## Reading order

New to the rebuild: read this page, then 02, then 03. Doc 01 is the evidence base —
read it when you want the receipts for a claim in 02.
