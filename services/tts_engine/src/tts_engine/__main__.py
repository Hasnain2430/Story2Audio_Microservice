"""Entrypoint.

Run with::

    uv run --project services/tts_engine python -m tts_engine
"""

from __future__ import annotations

import signal
import sys
import threading
from types import FrameType

from tts_engine.logging import configure_logging, get_logger
from tts_engine.server import create_server, mark_serving
from tts_engine.settings import tts_engine_settings
from tts_engine.synthesis import build_backend

log = get_logger(__name__)


def main() -> int:
    settings = tts_engine_settings()
    configure_logging(level=settings.log_level, json_output=settings.log_format == "json")

    backend = build_backend(settings)
    server, _ = create_server(backend, settings)

    stopping = threading.Event()

    def handle_signal(signum: int, _frame: FrameType | None) -> None:
        log.info("shutdown_signal", signal=signal.Signals(signum).name)
        stopping.set()

    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)

    server.start()
    log.info(
        "tts_engine_listening",
        address=f"{settings.tts_engine_host}:{settings.tts_engine_port}",
        backend=backend.name,
    )

    # Started before the model loads, so health probes get an answer during a cold start
    # rather than a connection refusal. The status stays NOT_SERVING until the weights
    # are resident: a scale-to-zero GPU can take a minute, and that minute is queue time
    # on an async job, not a user waiting.
    try:
        backend.load()
    except Exception:
        log.exception("model_load_failed", backend=backend.name)
        server.stop(0).wait()
        return 1

    mark_serving(server, serving=True)
    log.info("tts_engine_ready", backend=backend.name, model=backend.model)

    stopping.wait()

    # Refuse new work first, then let in-flight synthesis finish: truncating audio a
    # user is already waiting on is worse than a slightly slower shutdown.
    mark_serving(server, serving=False)
    log.info("tts_engine_draining", grace_seconds=settings.shutdown_grace_seconds)
    server.stop(settings.shutdown_grace_seconds).wait()
    backend.close()
    log.info("tts_engine_stopped")
    return 0


if __name__ == "__main__":
    sys.exit(main())
