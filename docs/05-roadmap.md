# 05 — Roadmap: from working to shippable

Phases 0–6 produced a system that works. This document is about what comes after, and it
is written against **measured numbers from a real run on the development machine**, not
against estimates.

Baseline, 21 September 2026, RTX 3050 4 GB, fp32 XTTS, Groq `openai/gpt-oss-120b`:

```
  0.1s  queued          job accepted, id returned
  5.1s  synthesizing    story written, 12 segments planned
 85.5s  done            184.2s of audio rendered
```

Real-time factor 0.46 — the GPU renders roughly twice as fast as the audio plays. That
number is the budget everything below is spent against.

---

## 1. The problem worth solving first

**Segment 0 finished at about 7 seconds. The listener waited 85.**

Nothing about that gap is hardware. The segments are rendered one at a time and the
listener is handed the result only once the last one is stitched, so 78 seconds of
finished audio sits in memory waiting for audio that has not been made yet.

This is the same shape of mistake v1 made, one level down. v1 blocked a request until the
whole job was done; v2 blocks *playback* until the whole job is done. The queue fixed the
first. It did not fix the second.

### 1a. Progressive playback

Upload each segment as it finishes, publish its URL on the existing WebSocket, and let the
player schedule segments back to back as they arrive.

Everything needed is already built:

| Piece | Status |
|---|---|
| Segments rendered individually | done — `_synthesize_segments` |
| Where each segment belongs in the track | done — `segment_timeline` (Phase 6.5) |
| Per-segment progress already on the wire | done — `ProgressEvent` |
| Client that follows segment timing | done — `Transcript` / `usePlayback` |
| Per-segment upload + a `segment_ready` event | **missing** |

Scheduling belongs in the Web Audio API rather than in a chain of `<audio>` elements:
`AudioBufferSourceNode.start(when)` is sample-accurate, so consecutive segments join
without the gap that element swapping leaves. The lead-in and inter-segment pauses are
already known from the timeline, so the client reconstructs the exact track the worker
would have assembled.

**Time to first audio: 85 s → ~7 s.**

### 1b. Sentence-level pipelining

Then stop waiting for the writer. Story tokens already stream from the LLM; once a
sentence is complete it can be segmented and synthesised while the rest is still being
written. The two stages overlap instead of queueing.

Worth more than the 5 seconds it saves here — a long story spends 20–30 s writing, and
that whole phase disappears behind synthesis.

**Time to first audio: ~7 s → ~3 s.**

The claim this project makes changes shape as a result. Not "I put a queue in front of a
slow thing", but: *the pipeline is a stream from the first token to the first sound.*

### 1c. What it costs to build

The honest part. This is the largest change in the document:

- `JobStatus` gains no states, but `synthesizing` becomes a state with **partial public
  output**, which the API must expose without implying the job is done.
- A new event variant (`segment_ready`) — a breaking `EVENT_SCHEMA_VERSION` bump, since
  clients currently drop unknown types.
- Storage lifecycle for per-segment objects, which are garbage the moment the stitched
  file exists.
- The stitched file must still be produced: downloads, sharing, and anything that is not
  this one web player depend on it.
- The player gets materially harder — buffering, a segment that fails mid-playback, and
  seeking into a region that has not arrived yet.

Do 1a first and ship it. 1b is a smaller change on top and is not worth coupling to it.

---

## 2. Throughput

`tts-worker` runs `--concurrency=1` and `tts-engine` guards inference with a bounded
semaphore. Both are correct for one GPU: segments are GPU-bound, and oversubscribing one
card trades latency for nothing.

But **segments are independent** — separate `inference()` calls sharing only a cached
speaker embedding. On serverless GPU they fan out across replicas and wall time divides:

| Configuration | 184 s of audio |
|---|---|
| Today: 1×3050, sequential | 85 s |
| 4 replicas, same card class | ~25 s |
| 1×L4, sequential | ~28 s |
| 4×L4 | ~8 s |

The worker needs a bounded pool and ordered reassembly. The timeline arithmetic does not
care what order segments complete in — it is computed from durations after the fact — so
this is contained.

Do this **after** progressive playback. Once the first sentence plays in 3 seconds, total
wall time stops being the number anyone feels, and this becomes a cost decision rather
than a latency one.

---

## 3. Cost and failure

### Resume

`pipeline.py` says it plainly today:

> `synthesizing` — a redelivery of an interrupted attempt. Synthesis restarts from the
> beginning: rendered PCM is not persisted, so there is nothing to resume from.

That was an acceptable limit when the expensive half — the story — was already durable.
It stops being acceptable on a per-second GPU bill: a redelivery at segment 11 of 12
pays for all 12 again.

Per-segment upload (§1a) removes the reason. Resume becomes "skip what already exists".

### Content-addressed segment cache

Key a rendered segment by a hash of `(text, voice_id, speed, language, model_version)`.
Retries, re-runs of the same story, and repeated phrases stop reaching the GPU at all.
The cache is in object storage, so it survives engine restarts — unlike the speaker
embedding cache, which today is per-process and recomputed on every cold start.

Persisting the speaker latents themselves is the same idea and smaller: a cold engine
currently recomputes conditioning for every voice it sees.

---

## 4. What shipping publicly actually requires

### 4a. The voice pack is a legal problem

**The built-in voices clone real, identifiable people** — Morgan Freeman, Cristiano
Ronaldo, Mbappé, Mahira Khan, Wasim Akram. Inherited from v1, where they were WAV files
in a folder.

That is fine for a local project and a portfolio demo that runs on your machine. It is
not fine for something publicly reachable that will synthesise arbitrary text in those
voices. Right of publicity and personality rights attach to a recognisable voice in most
jurisdictions that matter, and the licence on the model has no bearing on it.

Before anything is public-facing:

- ship with **synthetic or properly licensed** built-in voices; keep the current pack for
  local development only
- **consent gate** on upload — an explicit attestation that the speaker is the uploader or
  has given permission
- **disclose** that audio is synthetic, and consider an inaudible watermark
- a **takedown path**, and the ability to delete a voice and everything made with it

This is not a "nice to have" section. It is the difference between a demo and a liability.

### 4b. Accounts

Sessions are anonymous cookies today (`deps.py` returns `None` rather than raising for a
missing or forged cookie, and mints a new session). Clearing cookies destroys a user's
history, and quotas are per-cookie rather than per-person — trivially reset.

Magic-link or OAuth sign-in, with the existing anonymous session upgraded in place so
work done before signing in is not lost.

### 4c. Budgets, not request counts

Quotas today count requests. The cost driver is **GPU-seconds**, and a 2000-character
prompt at 1500 words costs many times what a one-liner does. Meter what is actually spent.

### 4d. Observability

Stage timings are recorded per job, which is enough to say *that* a job was slow and not
*why*. OpenTelemetry traces spanning gateway → queue → worker → engine, plus metrics on
real-time factor, queue depth, and failure rate by `ErrorCode`, turn the next slow job
into something diagnosable rather than reproducible-if-you-are-lucky.

---

## 5. Quality

### Real word alignment

The karaoke highlight interpolates word positions from character offsets inside a
segment, because XTTS returns no per-word timing. Segment boundaries are measured; word
positions are estimated, and a long pause mid-sentence makes them drift.

A forced aligner — a CTC aligner, or Whisper timestamps — run on **CPU** against the
rendered audio while the GPU moves on, replaces the estimate with a measurement. The
`SpokenSegment` contract already carries per-segment spans; this adds a word array inside
each one, and the player's rendering path barely changes.

### Other

- **Deterministic seeds**, so a job can be reproduced exactly when investigating a
  complaint about output.
- **Emotion that does something.** v1 passed `emotion=` to XTTS, which accepts and ignores
  it; v2 steers the writing with it instead, which is honest but indirect. A model with
  real prosody control would make it a property of the voice.
- **Longer-form structure.** Everything is one prompt → one story. Chapters, a
  consistent cast across segments, and narration that remembers what happened are a
  different and more interesting product.

---

## 6. Order

1. **Progressive playback** (§1a) — largest perceived-speed win; unlocks §3 entirely.
2. **Resume + segment cache** (§3) — nearly free once §1a lands.
3. **Voice pack and consent** (§4a) — blocking for any public deployment.
4. **Sentence-level pipelining** (§1b).
5. **Parallel synthesis** (§2) — when GPU cost, not latency, is the constraint.
6. **Accounts and budgets** (§4b, §4c).
7. **Forced alignment** (§5).

Items 1 and 2 are one piece of work. Item 3 gates deployment and nothing else. Everything
below that is genuinely optional, and saying so is more useful than a roadmap that
pretends all of it is planned.
