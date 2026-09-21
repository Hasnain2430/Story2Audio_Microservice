"""OpenAI-compatible chat adapter.

Written against the wire format rather than against one vendor, so Groq — the deployed
default — OpenRouter, Together, or a self-hosted vLLM all work through the same class
with a different base URL and key. That is most of what makes the provider decision
reversible later.
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

_SSE_DATA_PREFIX = "data:"
_SSE_DONE = "[DONE]"


class OpenAICompatProvider:
    """Streams from any endpoint that speaks the OpenAI chat-completions protocol."""

    def __init__(
        self,
        *,
        name: str,
        base_url: str,
        api_key: str,
        model: str,
        temperature: float,
        top_p: float,
        timeout_seconds: float,
        reasoning_effort: str | None = None,
        reasoning_token_allowance: int = 0,
    ) -> None:
        self.name = name
        self.model = model
        self._temperature = temperature
        self._top_p = top_p
        self._reasoning_effort = reasoning_effort
        self._reasoning_allowance = reasoning_token_allowance
        self._client = httpx.Client(
            base_url=base_url.rstrip("/"),
            timeout=httpx.Timeout(timeout_seconds, connect=10.0),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
        )

    def stream(self, prompt: StoryPrompt, stats: StreamStats) -> Iterator[str]:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": prompt.system},
                {"role": "user", "content": prompt.user},
            ],
            "stream": True,
            # Ask for usage on the final frame. Providers that ignore the option simply
            # omit it, which is why `output_tokens` stays optional.
            "stream_options": {"include_usage": True},
            "temperature": self._temperature,
            "top_p": self._top_p,
            # Reasoning tokens are drawn from the same budget as the prose. A reasoning
            # model handed a prose-sized budget can think until it is exhausted and
            # return an empty completion -- `gpt-oss-120b` spent all 1000 tokens on
            # reasoning and emitted nothing. The allowance is what the model may spend
            # thinking, on top of what the story itself needs.
            "max_tokens": prompt.max_output_tokens + self._reasoning_allowance,
        }
        if self._reasoning_effort is not None:
            # Supported by gpt-oss and Qwen3 on OpenAI-compatible endpoints; ignored by
            # models that do not reason, which is why it is safe to send unconditionally
            # once configured.
            payload["reasoning_effort"] = self._reasoning_effort

        try:
            with self._client.stream("POST", "/chat/completions", json=payload) as response:
                self._raise_for_status(response)
                yield from self._iter_chunks(response, stats)
        except httpx.HTTPError as exc:
            raise classify_http_error(exc, provider=self.name) from exc

    @staticmethod
    def _raise_for_status(response: httpx.Response) -> None:
        """Raise with the body available.

        ``raise_for_status`` on a streaming response cannot read the body, so the error
        detail would be a bare status code. Reading it first means the logs say *why* the
        provider refused.
        """
        if response.is_success:
            return
        response.read()
        response.raise_for_status()

    def _iter_chunks(self, response: httpx.Response, stats: StreamStats) -> Iterator[str]:
        """Parse a server-sent-events stream of chat-completion deltas."""
        for raw_line in response.iter_lines():
            line = raw_line.strip()
            if not line or not line.startswith(_SSE_DATA_PREFIX):
                continue

            data = line[len(_SSE_DATA_PREFIX) :].strip()
            if data == _SSE_DONE:
                return

            try:
                frame = json.loads(data)
            except json.JSONDecodeError:
                log.warning("sse_unparseable_frame", provider=self.name, preview=data[:120])
                continue

            if model := frame.get("model"):
                stats.model = str(model)

            # The usage frame arrives after the last content frame and carries no choices.
            if usage := frame.get("usage"):
                stats.output_tokens = usage.get("completion_tokens")
                details = usage.get("completion_tokens_details") or {}
                stats.reasoning_tokens = details.get("reasoning_tokens")

            for choice in frame.get("choices") or []:
                if reason := choice.get("finish_reason"):
                    stats.finish_reason = str(reason)
                # `reasoning` deltas are the model's private scratchpad. They are
                # deliberately not yielded: they are not the story, and streaming them to
                # the reader would show them the model's working.
                content = (choice.get("delta") or {}).get("content")
                if content:
                    yield str(content)

    def close(self) -> None:
        self._client.close()
