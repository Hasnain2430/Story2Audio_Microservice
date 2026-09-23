"""The LLM provider contract.

Synchronous by design. Celery runs a pool of synchronous worker processes, these tasks
run for minutes rather than milliseconds, and concurrency therefore comes from processes
rather than from an event loop. Wrapping each task in ``asyncio.run`` would build and
tear down a loop and a connection pool per job to gain nothing.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Protocol

import httpx

from story2audio_shared.errors import AppError, ErrorCode
from story2audio_shared.prompts import StoryPrompt


@dataclass
class StreamStats:
    """Out-of-band statistics for one generation.

    Passed in and filled by the provider rather than returned, because the text itself is
    yielded lazily and the totals are only known once the stream ends. Explicit and
    stateless, which a property on the provider would not be.
    """

    output_tokens: int | None = None
    #: Tokens the model spent on internal reasoning. These count against the same budget
    #: as the prose, so a reasoning model can exhaust `max_tokens` and return nothing.
    reasoning_tokens: int | None = None
    #: The model that actually served the request, which can differ from the one asked
    #: for when a gateway routes or substitutes.
    model: str | None = None
    finish_reason: str | None = None

    @property
    def truncated_by_token_limit(self) -> bool:
        """Did the model stop because it hit the cap rather than because it finished?"""
        return self.finish_reason in {"length", "max_tokens"}

    @property
    def spent_budget_on_reasoning(self) -> bool:
        """Did a reasoning model think until the budget ran out, producing no prose?"""
        return self.truncated_by_token_limit and bool(self.reasoning_tokens)


class LLMProvider(Protocol):
    """Streams a story, one chunk of text at a time."""

    #: Identifies the backend in logs and on the job row.
    name: str
    model: str

    def stream(self, prompt: StoryPrompt, stats: StreamStats) -> Iterator[str]: ...

    def close(self) -> None: ...


def classify_http_error(exc: Exception, *, provider: str) -> AppError:
    """Map a transport or HTTP failure onto the error taxonomy.

    The point is that the worker's retry policy and the user's error message both derive
    from one classification, instead of the worker guessing from an exception type and
    the API leaking a vendor string.
    """
    if isinstance(exc, httpx.TimeoutException):
        return AppError(ErrorCode.LLM_TIMEOUT, detail=f"{provider} timed out: {exc}")

    if isinstance(exc, httpx.HTTPStatusError):
        status = exc.response.status_code
        if status == httpx.codes.TOO_MANY_REQUESTS:
            return AppError(ErrorCode.LLM_RATE_LIMITED, detail=f"{provider} returned 429")
        if status in {httpx.codes.BAD_REQUEST, httpx.codes.UNPROCESSABLE_ENTITY}:
            # A 400 from a chat endpoint is usually a refused prompt rather than a
            # malformed request: the schemas above this make malformed unlikely.
            return AppError(
                ErrorCode.LLM_CONTENT_REJECTED,
                detail=f"{provider} rejected the request with {status}",
            )
        if status in {httpx.codes.UNAUTHORIZED, httpx.codes.FORBIDDEN}:
            # A misconfigured key is an operator problem. Retrying cannot fix it, but the
            # user-facing message must not say which credential is wrong.
            return AppError(
                ErrorCode.LLM_UNAVAILABLE,
                detail=f"{provider} rejected our credentials with {status}",
            )
        return AppError(ErrorCode.LLM_UNAVAILABLE, detail=f"{provider} returned {status}")

    if isinstance(exc, httpx.HTTPError):
        return AppError(ErrorCode.LLM_UNAVAILABLE, detail=f"{provider} unreachable: {exc}")

    return AppError(ErrorCode.INTERNAL, detail=f"{provider} failed unexpectedly: {exc!r}")
