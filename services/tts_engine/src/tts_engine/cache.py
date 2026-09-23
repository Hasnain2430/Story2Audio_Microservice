"""Speaker-embedding cache.

The single biggest avoidable cost in v1. It passed ``speaker_wav=<path>`` to every
``tts_to_file`` call, so the reference sample was re-read and the conditioning latents
recomputed for each segment -- a twelve-segment dialogue job paid for the same
computation twelve times, on the GPU, while holding the global lock.

Bounded LRU by count rather than by bytes: an XTTS embedding is a fixed-size pair of
tensors, so entries are interchangeable in size and counting them is both simpler and
accurate.
"""

from __future__ import annotations

import threading
from collections import OrderedDict

from tts_engine.synthesis.base import SpeakerEmbedding


class SpeakerCache:
    """Thread-safe LRU of speaker embeddings.

    Locked because gRPC dispatches handlers on a thread pool: two concurrent requests
    for the same cold voice would otherwise race, and two for different voices could
    interleave an eviction with an insert.
    """

    def __init__(self, max_entries: int) -> None:
        if max_entries < 1:
            raise ValueError("max_entries must be at least 1")
        self._max_entries = max_entries
        self._entries: OrderedDict[str, SpeakerEmbedding] = OrderedDict()
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0

    def get(self, voice_id: str) -> SpeakerEmbedding | None:
        with self._lock:
            embedding = self._entries.get(voice_id)
            if embedding is None:
                self._misses += 1
                return None
            self._entries.move_to_end(voice_id)
            self._hits += 1
            return embedding

    def put(self, embedding: SpeakerEmbedding) -> None:
        with self._lock:
            self._entries[embedding.voice_id] = embedding
            self._entries.move_to_end(embedding.voice_id)
            while len(self._entries) > self._max_entries:
                self._entries.popitem(last=False)

    def __contains__(self, voice_id: object) -> bool:
        with self._lock:
            return voice_id in self._entries

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)

    @property
    def hits(self) -> int:
        return self._hits

    @property
    def misses(self) -> int:
        return self._misses

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()
