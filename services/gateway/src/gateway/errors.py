"""Translating exceptions into responses.

The rule this module exists to enforce: nothing internal reaches the client. v1 called
``context.set_details(str(e))``, so a caller could read exception text and filesystem
paths straight off a failed request.

Every response body here is the same shape, so the frontend has one error type to handle
rather than three:

.. code-block:: json

    {"error": {"code": "voice_too_short", "message": "...", "retryable": false}}
"""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from story2audio_shared.errors import AppError, ErrorCode, spec_for
from story2audio_shared.logging import get_logger

log = get_logger(__name__)

#: Status codes the API returns for conditions that are not application failures, and so
#: have no ErrorCode of their own.
_STATUS_TO_CODE: dict[int, ErrorCode] = {
    status.HTTP_404_NOT_FOUND: ErrorCode.JOB_NOT_FOUND,
    status.HTTP_413_CONTENT_TOO_LARGE: ErrorCode.VOICE_TOO_LARGE,
    status.HTTP_422_UNPROCESSABLE_CONTENT: ErrorCode.VALIDATION_FAILED,
    status.HTTP_429_TOO_MANY_REQUESTS: ErrorCode.RATE_LIMITED,
}


def error_body(code: ErrorCode, message: str | None = None) -> dict[str, Any]:
    """Build the canonical error envelope."""
    spec = spec_for(code)
    return {
        "error": {
            "code": code.value,
            "message": message or spec.message,
            "retryable": spec.retryable,
        }
    }


def error_response(error: AppError) -> JSONResponse:
    """Render an :class:`AppError` as a response, with ``Retry-After`` where it helps."""
    headers: dict[str, str] = {}
    if error.http_status == status.HTTP_429_TOO_MANY_REQUESTS:
        headers["Retry-After"] = "60"
    return JSONResponse(
        status_code=error.http_status,
        content=error_body(error.code),
        headers=headers,
    )


def register_exception_handlers(app: FastAPI) -> None:
    """Install handlers so that no code path can return an unshaped error."""

    @app.exception_handler(AppError)
    async def _handle_app_error(_request: Request, exc: Exception) -> JSONResponse:
        error = exc if isinstance(exc, AppError) else AppError(ErrorCode.INTERNAL)
        # The detail is logged and dropped; only the fixed public message is serialised.
        log.info(
            "request_rejected",
            code=error.code.value,
            http_status=error.http_status,
            detail=error.detail,
        )
        return error_response(error)

    @app.exception_handler(RequestValidationError)
    async def _handle_validation_error(_request: Request, exc: Exception) -> JSONResponse:
        # Pydantic's error list can echo submitted values back, so it is logged rather
        # than returned. The client gets the field paths, which is what it needs to fix
        # the request, without the values being reflected.
        errors = exc.errors() if isinstance(exc, RequestValidationError) else []
        fields = [".".join(str(part) for part in item.get("loc", ())) for item in errors]
        log.info("request_validation_failed", fields=fields)

        body = error_body(ErrorCode.VALIDATION_FAILED)
        body["error"]["fields"] = fields
        return JSONResponse(status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, content=body)

    @app.exception_handler(StarletteHTTPException)
    async def _handle_http_exception(_request: Request, exc: Exception) -> JSONResponse:
        http_exc = (
            exc
            if isinstance(exc, StarletteHTTPException)
            else StarletteHTTPException(status_code=500)
        )
        code = _STATUS_TO_CODE.get(http_exc.status_code, ErrorCode.INTERNAL)
        return JSONResponse(
            status_code=http_exc.status_code,
            content=error_body(code),
            headers=getattr(http_exc, "headers", None),
        )

    @app.exception_handler(Exception)
    async def _handle_unexpected(request: Request, exc: Exception) -> JSONResponse:
        # The catch-all. Full context to the logs, a fixed sentence to the caller.
        log.exception(
            "unhandled_exception",
            path=request.url.path,
            method=request.method,
            error_type=type(exc).__name__,
        )
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=error_body(ErrorCode.INTERNAL),
        )
