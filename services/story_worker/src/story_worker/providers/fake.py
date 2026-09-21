"""Scripted provider for tests.

Lets the worker's control flow -- transitions, batching, cancellation, retries,
truncation handling -- be tested exhaustively and deterministically, without a model and
without a network. Every branch the real providers can take is reachable from here.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence

from story2audio_shared.errors import AppError, ErrorCode
from story2audio_shared.prompts import StoryPrompt
from story_worker.providers.base import StreamStats


class FakeProvider:
    """Replays a scripted response, optionally failing on the first N calls."""

    name = "fake"

    def __init__(
        self,
        chunks: Sequence[str] | None = None,
        *,
        model: str = "fake-model",
        fail_times: int = 0,
        failure: AppError | None = None,
        finish_reason: str = "stop",
        output_tokens: int | None = 128,
        chunks_per_call: Sequence[Sequence[str]] | None = None,
    ) -> None:
        self.model = model
        self._chunks = list(chunks or ["Once upon a time, ", "the lighthouse went dark."])
        self._chunks_per_call = (
            [list(call) for call in chunks_per_call] if chunks_per_call else None
        )
        self._fail_times = fail_times
        self._failure = failure or AppError(ErrorCode.LLM_UNAVAILABLE, detail="scripted failure")
        self._finish_reason = finish_reason
        self._output_tokens = output_tokens

        #: Every prompt this provider was asked to generate, for assertions about role
        #: separation and prompt assembly.
        self.calls: list[StoryPrompt] = []
        self.closed = False

    def stream(self, prompt: StoryPrompt, stats: StreamStats) -> Iterator[str]:
        self.calls.append(prompt)

        if self._fail_times > 0:
            self._fail_times -= 1
            raise self._failure

        if self._chunks_per_call is not None:
            index = min(len(self.calls) - 1, len(self._chunks_per_call) - 1)
            chunks = self._chunks_per_call[index]
        else:
            chunks = self._chunks

        stats.model = self.model
        stats.output_tokens = self._output_tokens
        stats.finish_reason = self._finish_reason
        yield from chunks

    def close(self) -> None:
        self.closed = True
