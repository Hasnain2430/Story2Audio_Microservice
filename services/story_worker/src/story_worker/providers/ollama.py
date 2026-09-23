"""Ollama adapter.

Keeps the project runnable entirely offline with no API key — the same models v1 used,
reached over Ollama's native chat API rather than through the ``ollama`` Python package,
so there is one HTTP client and one error taxonomy across every provider.

Unlike v1 this sends the instructions as a ``system`` message and the storyline as a
separate ``user`` message, and it holds no conversation history: v1 appended every
request and reply to one module-global list shared by all users.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any

import httpx

from story2audio_shared.logging import get_logger
from story2audio_shared.prompts import StoryPrompt
from story_worker.providers.base import StreamStats, classify_http_error

log = get_logger(__name__)


class OllamaProvider:
    """Streams from a local Ollama daemon."""

    name = "ollama"

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        temperature: float,
        top_p: float,
        timeout_seconds: float,
    ) -> None:
        self.model = model
        self._temperature = temperature
        self._top_p = top_p
        self._client = httpx.Client(
            base_url=base_url.rstrip("/"),
            timeout=httpx.Timeout(timeout_seconds, connect=10.0),
        )

    def stream(self, prompt: StoryPrompt, stats: StreamStats) -> Iterator[str]:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": prompt.system},
                {"role": "user", "content": prompt.user},
            ],
            "stream": True,
            "options": {
                "temperature": self._temperature,
                "top_p": self._top_p,
                # Sized from the requested story length. v1 sent a flat 2000 for every
                # length and truncated its own 800-1200 word target mid-sentence.
                "num_predict": prompt.max_output_tokens,
            },
        }

        try:
            with self._client.stream("POST", "/api/chat", json=payload) as response:
                response.raise_for_status()
                yield from self._iter_chunks(response, stats)
        except httpx.HTTPError as exc:
            raise classify_http_error(exc, provider=self.name) from exc

    def _iter_chunks(self, response: httpx.Response, stats: StreamStats) -> Iterator[str]:
        """Parse Ollama's newline-delimited JSON stream."""
        for line in response.iter_lines():
            if not line.strip():
                continue
            try:
                frame = json.loads(line)
            except json.JSONDecodeError:
                # A malformed frame is not worth failing a multi-minute generation over.
                log.warning("ollama_unparseable_frame", preview=line[:120])
                continue

            message = frame.get("message") or {}
            content = message.get("content")
            if content:
                yield str(content)

            if frame.get("done"):
                stats.output_tokens = frame.get("eval_count")
                stats.model = frame.get("model") or self.model
                stats.finish_reason = frame.get("done_reason")
                return

    def close(self) -> None:
        self._client.close()
