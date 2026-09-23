"""UUIDv7 identifiers (RFC 9562 §5.7).

Job and voice ids are time-ordered so that the id itself is a valid pagination cursor:
sorting by id is sorting by creation time, with no companion timestamp column and no
tiebreak. Python 3.11 ships no ``uuid7``, and the layout is small enough that
implementing it is preferable to taking a dependency for twenty lines.

Layout::

    0                   1                   2                   3
     0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1
    +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
    |                       unix_ts_ms (48 bits)                    |
    +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
    |    unix_ts_ms     |  ver  |  counter (12 bits, "rand_a")      |
    +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
    |var|                    rand_b (62 bits)                       |
    +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+

``rand_a`` is used as a monotonic sub-millisecond counter (RFC 9562 §6.2, method 1)
rather than as random bits, so ids generated within the same millisecond still sort in
creation order. The counter is seeded randomly within the lower half of its range on each
new millisecond, which preserves ordering while leaving headroom before rollover and
avoiding a fully predictable sequence.
"""

from __future__ import annotations

import secrets
import threading
import time
import uuid

_COUNTER_BITS = 12
_COUNTER_MAX = (1 << _COUNTER_BITS) - 1
_RAND_B_BITS = 62

_lock = threading.Lock()
_last_timestamp_ms = -1
_counter = 0


def _new_counter_seed() -> int:
    """Seed the sub-millisecond counter in the lower half of its range.

    Starting at zero would make the first id of every millisecond predictable; starting
    anywhere in the full range would risk rollover after only a few ids. The lower half
    is the usual compromise and still leaves >2000 ids per millisecond.
    """
    return secrets.randbelow(_COUNTER_MAX // 2)


def uuid7() -> uuid.UUID:
    """Return a time-ordered UUIDv7.

    Monotonic within a process even under concurrent calls and across a backwards clock
    step: if the wall clock moves backwards, the previous timestamp is reused and the
    counter continues, so ordering is never violated.
    """
    global _last_timestamp_ms, _counter

    with _lock:
        timestamp_ms = time.time_ns() // 1_000_000

        if timestamp_ms > _last_timestamp_ms:
            _last_timestamp_ms = timestamp_ms
            _counter = _new_counter_seed()
        else:
            # Same millisecond, or the clock went backwards. Either way, stay on the last
            # timestamp we emitted and advance the counter.
            _counter += 1
            if _counter > _COUNTER_MAX:
                # Counter exhausted within one millisecond. Borrow from the next
                # millisecond rather than emitting a non-monotonic id.
                _last_timestamp_ms += 1
                _counter = 0
            timestamp_ms = _last_timestamp_ms

        counter = _counter

    rand_b = secrets.randbits(_RAND_B_BITS)

    value = (timestamp_ms & 0xFFFF_FFFF_FFFF) << 80
    value |= 0x7 << 76  # version
    value |= counter << 64
    value |= 0b10 << 62  # variant
    value |= rand_b

    return uuid.UUID(int=value)


def timestamp_ms_of(value: uuid.UUID) -> int:
    """Extract the embedded creation timestamp, in milliseconds since the Unix epoch.

    Raises:
        ValueError: if ``value`` is not a UUIDv7.
    """
    if value.version != 7:
        raise ValueError(f"expected a UUIDv7, got version {value.version}")
    return value.int >> 80
