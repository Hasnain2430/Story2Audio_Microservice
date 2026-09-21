"""Error taxonomy.

The property that matters: no failure can reach a client without a deliberately written,
internal-detail-free message. v1 returned `str(exception)` verbatim.
"""

from __future__ import annotations

import pytest
from story2audio_shared.errors import AppError, ErrorCode, is_retryable, spec_for


@pytest.mark.parametrize("code", sorted(ErrorCode))
def test_every_code_has_a_usable_public_spec(code: ErrorCode) -> None:
    spec = spec_for(code)
    assert 400 <= spec.http_status <= 599
    assert spec.message
    assert spec.message[0].isupper()
    assert spec.message.endswith((".", "!", "?"))


@pytest.mark.parametrize("code", sorted(ErrorCode))
def test_public_messages_leak_no_internal_detail(code: ErrorCode) -> None:
    message = spec_for(code).message.lower()
    # Nothing that hints at internals, vendors or infrastructure.
    for forbidden in ("traceback", "exception", "redis", "postgres", "celery", "s3", "grpc"):
        assert forbidden not in message


def test_app_error_keeps_detail_out_of_the_public_message() -> None:
    error = AppError(
        ErrorCode.TTS_UNAVAILABLE,
        detail="grpc channel to tts-engine:50051 refused: /srv/app/tts_worker/client.py:88",
    )

    assert error.public_message == spec_for(ErrorCode.TTS_UNAVAILABLE).message
    assert "50051" not in error.public_message
    assert "50051" in (error.detail or "")
    assert error.http_status == 503
    assert error.retryable is True


def test_app_error_without_detail_falls_back_to_the_public_message() -> None:
    error = AppError(ErrorCode.JOB_NOT_FOUND)
    assert str(error) == error.public_message
    assert error.http_status == 404


def test_client_mistakes_are_not_advertised_as_retryable() -> None:
    # Retrying an identical bad request cannot succeed, and telling the UI otherwise would
    # produce a "try again" button that always fails.
    for code in (
        ErrorCode.PROMPT_TOO_LONG,
        ErrorCode.VOICE_TOO_SHORT,
        ErrorCode.DIALOGUE_VOICE_REQUIRED,
        ErrorCode.JOB_NOT_FOUND,
    ):
        assert not is_retryable(code)


def test_transient_infrastructure_failures_are_retryable() -> None:
    for code in (
        ErrorCode.LLM_UNAVAILABLE,
        ErrorCode.LLM_TIMEOUT,
        ErrorCode.TTS_CAPACITY,
        ErrorCode.STORAGE_UNAVAILABLE,
    ):
        assert is_retryable(code)


def test_repr_is_useful_in_logs_and_carries_the_detail() -> None:
    error = AppError(ErrorCode.INTERNAL, detail="boom")
    assert repr(error) == "AppError(code='internal', detail='boom')"
