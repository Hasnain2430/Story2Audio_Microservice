"""Structured logging.

v1's entire observability story was ``print("\U0001f680 Starting gRPC server...")``. Here every
log line is a structured event carrying ``job_id`` and ``trace_id``, so one job can be
followed from ``POST /v1/jobs`` through the queue into both workers and the TTS engine.

Console rendering locally for readability; JSON in deployed environments so a log shipper
can index the fields.
"""

from __future__ import annotations

import logging
import sys
from collections.abc import MutableMapping
from contextvars import ContextVar
from typing import Any
from uuid import UUID

import structlog

#: Correlation identifiers, set once per request or per task and attached to every log
#: line emitted underneath. Context variables rather than parameters so that call sites
#: deep in the stack do not have to thread them through.
_trace_id: ContextVar[str | None] = ContextVar("trace_id", default=None)
_job_id: ContextVar[str | None] = ContextVar("job_id", default=None)


def bind_trace_id(trace_id: str) -> None:
    """Attach a trace id to every subsequent log line in this context."""
    _trace_id.set(trace_id)


def bind_job_id(job_id: UUID | str) -> None:
    """Attach a job id to every subsequent log line in this context."""
    _job_id.set(str(job_id))


def current_trace_id() -> str | None:
    return _trace_id.get()


def _add_correlation_ids(
    _logger: Any, _method: str, event_dict: MutableMapping[str, Any]
) -> MutableMapping[str, Any]:
    """Processor that copies the context variables onto each event."""
    trace_id = _trace_id.get()
    if trace_id is not None:
        event_dict.setdefault("trace_id", trace_id)
    job_id = _job_id.get()
    if job_id is not None:
        event_dict.setdefault("job_id", job_id)
    return event_dict


def configure_logging(*, level: str = "info", json_output: bool = False) -> None:
    """Configure structlog and the stdlib logging bridge.

    Called once at process startup by each service. Idempotent.
    """
    numeric_level = getattr(logging, level.upper(), logging.INFO)

    shared_processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
        _add_correlation_ids,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
    ]

    renderer: Any
    if json_output:
        # Exceptions become a `exception` string field rather than a multi-line traceback,
        # so one failure stays one log record in whatever is indexing them.
        shared_processors.append(structlog.processors.format_exc_info)
        renderer = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer(colors=sys.stderr.isatty())

    structlog.configure(
        processors=[*shared_processors, renderer],
        wrapper_class=structlog.make_filtering_bound_logger(numeric_level),
        logger_factory=structlog.PrintLoggerFactory(file=sys.stderr),
        cache_logger_on_first_use=True,
    )

    # Route stdlib logging (uvicorn, sqlalchemy, botocore, celery) through the same handler
    # so deployed output is one consistent stream rather than two interleaved formats.
    logging.basicConfig(format="%(message)s", stream=sys.stderr, level=numeric_level, force=True)


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    """Return a bound logger, optionally named after the calling module."""
    logger: structlog.stdlib.BoundLogger = structlog.get_logger(name)
    return logger
