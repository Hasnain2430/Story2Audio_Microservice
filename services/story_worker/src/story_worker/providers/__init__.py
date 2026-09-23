"""LLM provider adapters.

One narrow interface, three implementations. The deployment target and local development
differ by an environment variable rather than by a code fork, which is what keeps the
offline Ollama path working without it becoming a second codebase.
"""

from story_worker.providers.base import (
    LLMProvider,
    StreamStats,
    classify_http_error,
)
from story_worker.providers.factory import build_provider

__all__ = ["LLMProvider", "StreamStats", "build_provider", "classify_http_error"]
