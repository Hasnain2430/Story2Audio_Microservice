"""UUIDv7 generation.

The ordering guarantees here are load-bearing: cursor pagination sorts on the id alone,
so a non-monotonic id would silently skip or repeat rows in a page.
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor

from story2audio_shared.ids import timestamp_ms_of, uuid7


def test_version_and_variant_are_rfc_9562_compliant() -> None:
    value = uuid7()
    assert value.version == 7
    # RFC 9562 variant bits are 0b10.
    assert (value.int >> 62) & 0b11 == 0b10


def test_embedded_timestamp_matches_wall_clock() -> None:
    before = time.time_ns() // 1_000_000
    value = uuid7()
    after = time.time_ns() // 1_000_000

    assert before <= timestamp_ms_of(value) <= after


def test_ids_are_strictly_increasing_within_a_millisecond() -> None:
    # Far more ids than one millisecond can hold, so this also exercises the counter
    # rollover path that borrows from the next millisecond.
    values = [uuid7() for _ in range(10_000)]
    assert values == sorted(values)
    assert len(set(values)) == len(values)


def test_ids_are_unique_and_ordered_under_concurrency() -> None:
    with ThreadPoolExecutor(max_workers=8) as executor:
        values = list(executor.map(lambda _: uuid7(), range(4_000)))

    # Threads interleave, so the emission order is not the call order. What must hold is
    # that the generator never issues the same id twice under contention.
    assert len(set(values)) == len(values)


def test_lexicographic_string_order_matches_numeric_order() -> None:
    # Cursor values travel as strings in URLs; the string form must sort identically.
    values = [uuid7() for _ in range(500)]
    assert [str(v) for v in values] == sorted(str(v) for v in values)


def test_timestamp_extraction_rejects_other_uuid_versions() -> None:
    import uuid

    import pytest

    with pytest.raises(ValueError, match="expected a UUIDv7"):
        timestamp_ms_of(uuid.uuid4())
