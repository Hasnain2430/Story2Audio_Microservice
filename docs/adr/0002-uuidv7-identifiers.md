# ADR-0002 — UUIDv7 for job and voice ids

**Status:** Accepted (Phase 1)

## Context

`GET /v1/jobs` needs stable pagination over a list that grows while the user is reading it.
Offset pagination skips and repeats rows under concurrent inserts, so the API uses a cursor.
A cursor needs a sort key that is unique, stable and monotonic.

## Options

1. **`uuid4` primary key, cursor on `(created_at, id)`.** Needs a compound index and a
   tiebreak, because two rows can share a millisecond.
2. **Auto-increment integer.** Perfect ordering, but enumerable: `/v1/jobs/41` tells you
   there is a job 40, and roughly how much traffic the service has seen.
3. **UUIDv7.** Time-ordered by construction, so the id *is* the cursor.

## Decision

Option 3, implemented in `packages/shared/src/story2audio_shared/ids.py` rather than taken
as a dependency — RFC 9562 §5.7 is a small, well-specified bit layout, and twenty lines
plus tests are cheaper to own than another package to track.

The implementation uses the 12-bit `rand_a` field as a monotonic sub-millisecond counter
(RFC 9562 §6.2, method 1), seeded randomly in the lower half of its range at each new
millisecond. That gives ordering within a millisecond, headroom before rollover, and no
fully predictable sequence. A backwards clock step reuses the last emitted timestamp rather
than going backwards, and counter exhaustion borrows from the next millisecond.

## Consequences

- One index, `(owner_id, id)`, serves both the filter and the sort. No `created_at`
  tiebreak in the query.
- The string form sorts identically to the numeric form, so a cursor travels in a URL
  unchanged. Covered by a test.
- **A UUIDv7 leaks its creation time.** For job and voice ids this discloses nothing the
  API does not already return in `created_at`. It would be the wrong choice for anything
  user-identifying, and this ADR does not license it there.
- Monotonicity is per-process. Two processes can emit ids within the same millisecond in
  either relative order — acceptable, since what is needed is a total order for pagination,
  not a global happened-before.

## Revisit when

An id is needed for something where disclosing a creation timestamp matters.
