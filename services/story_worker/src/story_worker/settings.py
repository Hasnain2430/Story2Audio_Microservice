"""story-worker configuration."""

from __future__ import annotations

from enum import StrEnum
from functools import lru_cache

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class LLMProviderName(StrEnum):
    """Which model backend to call.

    ``ollama`` keeps the project fully runnable offline with no API key, which is a real
    feature rather than a leftover from v1. ``groq`` is the deployed default: an
    OpenAI-compatible endpoint, far faster than a 7B model on a consumer GPU, and priced
    in fractions of a cent per story.
    """

    OLLAMA = "ollama"
    GROQ = "groq"
    FAKE = "fake"


class StoryWorkerSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore", frozen=True
    )

    llm_provider: LLMProviderName = LLMProviderName.OLLAMA
    #: One model for every story length. v1 picked the model from the requested length,
    #: so asking for a short story forced a 1B model and capped its quality -- a
    #: deployment constraint leaking into product behaviour.
    llm_model: str = "llama3"
    llm_temperature: float = Field(default=0.9, ge=0.0, le=2.0)
    llm_top_p: float = Field(default=0.95, gt=0.0, le=1.0)
    llm_request_timeout_seconds: float = Field(default=120.0, gt=0)

    #: How hard a reasoning model should think, where the endpoint supports it
    #: (`low` | `medium` | `high`). Unset means the parameter is not sent at all.
    #: Story writing needs very little deliberation, and reasoning is charged from
    #: the same token budget as the prose.
    llm_reasoning_effort: str | None = None
    #: Extra tokens a reasoning model may spend thinking, on top of the budget sized
    #: for the story itself. Zero for a non-reasoning model. Without it, a reasoning
    #: model can exhaust the budget before writing a single word.
    llm_reasoning_token_allowance: int = Field(default=0, ge=0, le=32_000)

    ollama_base_url: str = "http://localhost:11434"

    groq_api_key: SecretStr = SecretStr("")
    groq_base_url: str = "https://api.groq.com/openai/v1"

    #: How often streamed text is forwarded to subscribers.
    token_batch_interval_seconds: float = Field(default=0.05, gt=0)
    token_batch_max_chars: int = Field(default=400, ge=1)

    #: Check for cancellation every N published frames. Frequent enough to stop promptly,
    #: rare enough not to query the database per token.
    cancel_check_every_frames: int = Field(default=10, ge=1)

    #: At most one continuation pass when a story stops mid-sentence. Bounded on purpose:
    #: an unbounded loop against a model that keeps trailing off would spend without end.
    max_continuation_passes: int = Field(default=1, ge=0, le=2)

    #: Retry budget for transient provider failures.
    max_retries: int = Field(default=3, ge=0, le=10)
    retry_backoff_seconds: float = Field(default=5.0, gt=0)
    retry_backoff_max_seconds: float = Field(default=120.0, gt=0)

    #: Advisory lease TTL. Must exceed the longest plausible generation, or a second
    #: delivery could start while the first is still streaming.
    job_lease_ttl_seconds: int = Field(default=900, ge=60)


@lru_cache(maxsize=1)
def story_worker_settings() -> StoryWorkerSettings:
    return StoryWorkerSettings()
