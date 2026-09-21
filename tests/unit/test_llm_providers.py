"""LLM provider adapters.

Exercised against a stubbed HTTP transport rather than a live model, so the wire-format
parsing and the error classification are pinned exactly — including the malformed and
partial frames a real stream produces and a happy-path test would never see.
"""

from __future__ import annotations

import json

import httpx
import pytest

from story2audio_shared.enums import Emotion, Language, StoryLength, VoiceMode
from story2audio_shared.errors import AppError, ErrorCode
from story2audio_shared.prompts import build_story_prompt
from story_worker.providers.base import StreamStats, classify_http_error
from story_worker.providers.ollama import OllamaProvider
from story_worker.providers.openai_compat import OpenAICompatProvider
from story_worker.settings import LLMProviderName, StoryWorkerSettings

PROMPT = build_story_prompt(
    "A cartographer finds a road that is on no map.",
    length=StoryLength.SHORT,
    mode=VoiceMode.NARRATION,
    language=Language.EN,
    emotion=Emotion.NEUTRAL,
)


def _transport(handler: object) -> httpx.MockTransport:
    return httpx.MockTransport(handler)  # type: ignore[arg-type]


def _ollama_with(body: str, *, status: int = 200) -> OllamaProvider:
    provider = OllamaProvider(
        base_url="http://ollama.test",
        model="llama3",
        temperature=0.9,
        top_p=0.95,
        timeout_seconds=5.0,
    )
    provider._client = httpx.Client(
        base_url="http://ollama.test",
        transport=_transport(lambda request: httpx.Response(status, text=body)),
    )
    return provider


def _openai_with(body: str, *, status: int = 200) -> OpenAICompatProvider:
    provider = OpenAICompatProvider(
        name="groq",
        base_url="http://groq.test/v1",
        api_key="test-key",
        model="llama-3.3-70b",
        temperature=0.9,
        top_p=0.95,
        timeout_seconds=5.0,
    )
    provider._client = httpx.Client(
        base_url="http://groq.test/v1",
        transport=_transport(lambda request: httpx.Response(status, text=body)),
    )
    return provider


# --- Ollama --------------------------------------------------------------------------------------


def test_ollama_streams_content_and_reports_usage() -> None:
    body = "\n".join(
        [
            json.dumps({"message": {"content": "The map "}, "done": False}),
            json.dumps({"message": {"content": "was wrong."}, "done": False}),
            json.dumps(
                {
                    "message": {"content": ""},
                    "done": True,
                    "done_reason": "stop",
                    "eval_count": 42,
                    "model": "llama3:latest",
                }
            ),
        ]
    )
    provider = _ollama_with(body)
    stats = StreamStats()

    text = "".join(provider.stream(PROMPT, stats))

    assert text == "The map was wrong."
    assert stats.output_tokens == 42
    assert stats.model == "llama3:latest"
    assert stats.finish_reason == "stop"
    assert not stats.truncated_by_token_limit


def test_ollama_skips_unparseable_frames_rather_than_failing() -> None:
    """A malformed frame is not worth losing a multi-minute generation over."""
    body = "\n".join(
        [
            json.dumps({"message": {"content": "good "}, "done": False}),
            "{not json at all",
            json.dumps({"message": {"content": "still good"}, "done": True}),
        ]
    )

    assert "".join(_ollama_with(body).stream(PROMPT, StreamStats())) == "good still good"


def test_ollama_stops_at_the_done_frame() -> None:
    """Anything after `done` is not part of this response."""
    body = "\n".join(
        [
            json.dumps({"message": {"content": "first"}, "done": False}),
            json.dumps({"message": {"content": ""}, "done": True}),
            json.dumps({"message": {"content": "LEAKED"}, "done": False}),
        ]
    )

    assert "LEAKED" not in "".join(_ollama_with(body).stream(PROMPT, StreamStats()))


def test_ollama_reports_a_token_limit_stop() -> None:
    body = json.dumps({"message": {"content": "cut off"}, "done": True, "done_reason": "length"})
    stats = StreamStats()

    list(_ollama_with(body).stream(PROMPT, stats))

    assert stats.truncated_by_token_limit


def test_ollama_sends_system_and_user_as_separate_messages() -> None:
    """v1 concatenated the instructions and the user's text into one string."""
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured.update(json.loads(request.content))
        return httpx.Response(200, text=json.dumps({"message": {"content": "x"}, "done": True}))

    provider = OllamaProvider(
        base_url="http://ollama.test",
        model="llama3",
        temperature=0.9,
        top_p=0.95,
        timeout_seconds=5.0,
    )
    provider._client = httpx.Client(base_url="http://ollama.test", transport=_transport(handler))

    list(provider.stream(PROMPT, StreamStats()))

    messages = captured["messages"]
    assert isinstance(messages, list)
    assert [message["role"] for message in messages] == ["system", "user"]
    assert captured["options"]["num_predict"] == PROMPT.max_output_tokens  # type: ignore[index]


# --- OpenAI-compatible ---------------------------------------------------------------------------


def test_openai_compat_streams_sse_deltas() -> None:
    body = "\n".join(
        [
            'data: {"model":"llama-3.3-70b","choices":[{"delta":{"content":"The road "}}]}',
            'data: {"choices":[{"delta":{"content":"led nowhere."},"finish_reason":"stop"}]}',
            'data: {"choices":[],"usage":{"completion_tokens":57}}',
            "data: [DONE]",
        ]
    )
    stats = StreamStats()

    text = "".join(_openai_with(body).stream(PROMPT, stats))

    assert text == "The road led nowhere."
    assert stats.output_tokens == 57
    assert stats.model == "llama-3.3-70b"
    assert stats.finish_reason == "stop"


def test_openai_compat_ignores_keepalives_and_blank_lines() -> None:
    body = "\n".join(
        [
            "",
            ": keepalive",
            'data: {"choices":[{"delta":{"content":"text"}}]}',
            "",
            "data: [DONE]",
        ]
    )

    assert "".join(_openai_with(body).stream(PROMPT, StreamStats())) == "text"


def test_openai_compat_stops_at_done_sentinel() -> None:
    body = "\n".join(
        [
            'data: {"choices":[{"delta":{"content":"kept"}}]}',
            "data: [DONE]",
            'data: {"choices":[{"delta":{"content":"LEAKED"}}]}',
        ]
    )

    assert "".join(_openai_with(body).stream(PROMPT, StreamStats())) == "kept"


def test_openai_compat_survives_a_malformed_frame() -> None:
    body = "\n".join(
        [
            "data: {broken",
            'data: {"choices":[{"delta":{"content":"recovered"}}]}',
            "data: [DONE]",
        ]
    )

    assert "".join(_openai_with(body).stream(PROMPT, StreamStats())) == "recovered"


def test_openai_compat_tolerates_a_provider_that_omits_usage() -> None:
    """Not every OpenAI-compatible endpoint honours `stream_options`."""
    body = "\n".join(['data: {"choices":[{"delta":{"content":"x"}}]}', "data: [DONE]"])
    stats = StreamStats()

    list(_openai_with(body).stream(PROMPT, stats))

    assert stats.output_tokens is None


def test_openai_compat_requests_a_bounded_completion() -> None:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured.update(json.loads(request.content))
        return httpx.Response(200, text="data: [DONE]")

    provider = OpenAICompatProvider(
        name="groq",
        base_url="http://groq.test/v1",
        api_key="k",
        model="m",
        temperature=0.9,
        top_p=0.95,
        timeout_seconds=5.0,
    )
    provider._client = httpx.Client(base_url="http://groq.test/v1", transport=_transport(handler))

    list(provider.stream(PROMPT, StreamStats()))

    assert captured["max_tokens"] == PROMPT.max_output_tokens
    assert captured["stream"] is True


# --- Error classification ------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("status", "expected"),
    [
        (429, ErrorCode.LLM_RATE_LIMITED),
        (400, ErrorCode.LLM_CONTENT_REJECTED),
        (401, ErrorCode.LLM_UNAVAILABLE),
        (403, ErrorCode.LLM_UNAVAILABLE),
        (500, ErrorCode.LLM_UNAVAILABLE),
        (503, ErrorCode.LLM_UNAVAILABLE),
    ],
)
def test_http_status_codes_map_to_error_codes(status: int, expected: ErrorCode) -> None:
    """Retry policy and the user's message both derive from this one classification."""
    with pytest.raises(AppError) as info:
        list(_openai_with("nope", status=status).stream(PROMPT, StreamStats()))

    assert info.value.code is expected


def test_a_timeout_is_classified_as_a_timeout() -> None:
    error = classify_http_error(httpx.ReadTimeout("slow"), provider="groq")
    assert error.code is ErrorCode.LLM_TIMEOUT


def test_an_unreachable_host_is_classified_as_unavailable() -> None:
    error = classify_http_error(httpx.ConnectError("refused"), provider="ollama")
    assert error.code is ErrorCode.LLM_UNAVAILABLE


def test_credential_failures_do_not_say_which_credential() -> None:
    """An operator problem must not become a message that describes our configuration."""
    with pytest.raises(AppError) as info:
        list(_openai_with("unauthorized", status=401).stream(PROMPT, StreamStats()))

    assert "key" not in info.value.public_message.lower()
    assert "401" not in info.value.public_message


def test_the_error_body_reaches_the_logs() -> None:
    """`raise_for_status` on a stream cannot read the body, so it is read first."""
    with pytest.raises(AppError) as info:
        list(_openai_with("model_decommissioned", status=400).stream(PROMPT, StreamStats()))

    assert info.value.detail is not None


# --- Factory -------------------------------------------------------------------------------------


def test_groq_without_a_key_fails_at_construction() -> None:
    """One clear operator-facing error, rather than every job failing with a 401."""
    from story_worker.providers import build_provider

    settings = StoryWorkerSettings(llm_provider=LLMProviderName.GROQ, groq_api_key="")  # type: ignore[arg-type]

    with pytest.raises(AppError) as info:
        build_provider(settings)

    assert info.value.code is ErrorCode.LLM_UNAVAILABLE
    assert "GROQ_API_KEY" in (info.value.detail or "")


def test_the_configured_provider_is_selected() -> None:
    from story_worker.providers import build_provider

    ollama = build_provider(StoryWorkerSettings(llm_provider=LLMProviderName.OLLAMA))
    assert ollama.name == "ollama"
    ollama.close()

    groq = build_provider(
        StoryWorkerSettings(llm_provider=LLMProviderName.GROQ, groq_api_key="k")  # type: ignore[arg-type]
    )
    assert groq.name == "groq"
    groq.close()
