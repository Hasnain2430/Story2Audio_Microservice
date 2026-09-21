# ADR-0003 — Discriminated union and monotonic `seq` for job events

**Status:** Accepted (Phase 1)

## Context

Job progress reaches the browser over a WebSocket fed by Redis pub/sub. Redis pub/sub is
fire-and-forget: no backlog, no acknowledgement, no replay. A subscriber that connects late
or reconnects after a dropped connection simply misses whatever was published while it was
away.

The client therefore needs to answer two questions it cannot answer from a payload alone:
*what kind of event is this*, and *did I miss any*.

## Decision

**Tagged union on `type`.** Every event model carries a `Literal` discriminator and the
union is parsed through a single `TypeAdapter`. On the TypeScript side this becomes an
exhaustive `switch` with no casting, so adding a variant is a compile error in the client
rather than a silent no-op.

**Per-job monotonic `seq`, starting at 1.** Allocated by an atomic increment of
`jobs.last_event_seq`. A client that sees `seq` jump knows frames were lost.

**Snapshot on connect.** The first frame of every WebSocket connection is a `StatusEvent`
built from Postgres, not from Redis, so a late subscriber starts from truth rather than
from whatever happens to be published next.

**Refetch, never replay, on a gap.** There is no history in Redis to replay from, so the
client responds to a detected gap by refetching `GET /v1/jobs/{id}` and reconciling.

**Schema version `v` on every event.** A client that does not recognise the version refuses
to interpret the frame rather than mis-rendering it.

## Consequences

- The WebSocket is strictly an optimisation over a poll-correct design. `GET /v1/jobs/{id}`
  alone is sufficient to drive the entire UI, which is what makes the polling fallback in
  `useJobEvents` safe — and it is tested as an explicit e2e path.
- Terminal events (`done`, `failed`, `cancelled`) are identified by a shared predicate, so
  server and client agree on when the stream is over and the socket can close.
- `seq` also defends against out-of-order delivery being rendered as garbled story text,
  since token frames are applied in sequence order.
- Cost: one extra integer column and one atomic increment per published event.

## Rejected

- **Redis Streams instead of pub/sub**, which would give a replayable backlog. It addresses
  the gap problem more directly, but adds trimming policy, consumer-group management and
  memory growth to operate — for a system whose durable state already lives in Postgres and
  can simply be refetched.
- **No sequence numbers, reconnect-and-refetch unconditionally.** Simpler, but it cannot
  distinguish "quiet" from "disconnected", so it either refetches constantly or misses
  frames.
