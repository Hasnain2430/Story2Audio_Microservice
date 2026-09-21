"""Provider selection.

One environment variable decides the backend. Nothing above this layer knows which one
is in use, which is what keeps the deployment choice from leaking into the pipeline.
"""

from __future__ import annotations

from story2audio_shared.errors import AppError, ErrorCode
from story2audio_shared.logging import get_logger
from story_worker.providers.base import LLMProvider
from story_worker.providers.fake import FakeProvider
from story_worker.providers.ollama import OllamaProvider
from story_worker.providers.openai_compat import OpenAICompatProvider
from story_worker.settings import LLMProviderName, StoryWorkerSettings

log = get_logger(__name__)


def build_provider(settings: StoryWorkerSettings) -> LLMProvider:
    """Construct the configured provider.

    A new client per task rather than a process-wide singleton: tasks are minutes apart,
    a pooled connection would be stale by the next one anyway, and a per-task client
    cannot leak state between jobs.
    """
    match settings.llm_provider:
        case LLMProviderName.OLLAMA:
            return OllamaProvider(
                base_url=settings.ollama_base_url,
                model=settings.llm_model,
                temperature=settings.llm_temperature,
                top_p=settings.llm_top_p,
                timeout_seconds=settings.llm_request_timeout_seconds,
            )

        case LLMProviderName.GROQ:
            key = settings.groq_api_key.get_secret_value()
            if not key:
                # Fail at construction with a clear operator-facing message rather than
                # letting every job fail with a 401 from the provider.
                raise AppError(
                    ErrorCode.LLM_UNAVAILABLE,
                    detail="GROQ_API_KEY is not set but LLM_PROVIDER=groq",
                )
            return OpenAICompatProvider(
                name="groq",
                base_url=settings.groq_base_url,
                api_key=key,
                model=settings.llm_model,
                temperature=settings.llm_temperature,
                top_p=settings.llm_top_p,
                timeout_seconds=settings.llm_request_timeout_seconds,
            )

        case LLMProviderName.FAKE:
            log.warning("using_fake_llm_provider", reason="test configuration")
            return FakeProvider()
