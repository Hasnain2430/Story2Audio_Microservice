"""Celery application and worker-process resources.

Two queues, ``story`` and ``tts``, consumed by two different worker deployments. That
split is the point of the architecture: the LLM stage is network-bound and cheap, the TTS
stage is GPU-bound and expensive, so a backlog of stories waiting on a busy GPU must not
also stall story generation. They scale on different signals.
"""

from __future__ import annotations

from functools import lru_cache

from celery import Celery
from redis import Redis
from sqlalchemy.orm import Session, sessionmaker

from story2audio_shared.config import core_settings, redis_settings
from story2audio_shared.logging import configure_logging, get_logger
from story2audio_shared.worker import (
    JobLease,
    SyncEventPublisher,
    create_worker_engine,
    create_worker_redis,
    create_worker_session_factory,
)
from story_worker.settings import story_worker_settings

log = get_logger(__name__)

STORY_QUEUE = "story"
TTS_QUEUE = "tts"

WRITE_STORY_TASK = "story_worker.tasks.write_story"
SYNTHESIZE_TASK = "tts_worker.tasks.synthesize"


def build_celery_app() -> Celery:
    """Configure the Celery application."""
    redis_config = redis_settings()
    worker_config = story_worker_settings()

    app = Celery(
        "story_worker",
        broker=redis_config.celery_broker_url,
        backend=redis_config.celery_result_backend,
        include=["story_worker.tasks"],
    )

    app.conf.update(
        task_default_queue=STORY_QUEUE,
        task_routes={
            WRITE_STORY_TASK: {"queue": STORY_QUEUE},
            SYNTHESIZE_TASK: {"queue": TTS_QUEUE},
        },
        task_serializer="json",
        result_serializer="json",
        accept_content=["json"],
        timezone="UTC",
        enable_utc=True,
        # At-least-once delivery: the message is acknowledged after the task finishes, so
        # a worker killed mid-generation gets it redelivered rather than losing the job.
        # The task is written to be idempotent against its own row, and an advisory Redis
        # lease stops two concurrent deliveries both paying the model.
        task_acks_late=True,
        task_reject_on_worker_lost=True,
        # One job at a time per process. These tasks are minutes long and the useful
        # concurrency knob is the number of worker processes, not prefetch depth. A high
        # prefetch would also let one worker hoard the queue while others idle.
        worker_prefetch_multiplier=1,
        # Hard ceiling well above the longest plausible generation, so a wedged provider
        # connection cannot occupy a worker forever.
        task_time_limit=worker_config.job_lease_ttl_seconds + 120,
        task_soft_time_limit=worker_config.job_lease_ttl_seconds,
        broker_connection_retry_on_startup=True,
        worker_hijack_root_logger=False,
        # Celery otherwise replaces sys.stdout/sys.stderr with a proxy that writes into
        # its own logger. `configure_logging` then installs a stdlib handler pointed at
        # that proxy, the proxy's recursion guard drops the write, and every structured
        # log line from this worker disappears -- silently, with logging that works
        # perfectly when the same code is run outside Celery. Keep the real streams.
        worker_redirect_stdouts=False,
    )
    return app


_core = core_settings()
configure_logging(level=_core.log_level, json_output=_core.log_format.value == "json")

celery_app = build_celery_app()


class WorkerResources:
    """Per-process resources, built once and reused across tasks.

    A Celery worker is a long-lived process handling one job at a time, so the engine and
    the Redis client are worth keeping; the LLM client is not, and is built per task.
    """

    def __init__(self) -> None:
        core = core_settings()
        configure_logging(level=core.log_level, json_output=core.log_format.value == "json")

        self.settings = story_worker_settings()
        self.engine = create_worker_engine()
        self.session_factory: sessionmaker[Session] = create_worker_session_factory(self.engine)
        self.redis: Redis = create_worker_redis(redis_settings())
        self.publisher = SyncEventPublisher(self.redis)
        self.lease = JobLease(self.redis, ttl_seconds=self.settings.job_lease_ttl_seconds)


@lru_cache(maxsize=1)
def resources() -> WorkerResources:
    """Process-wide resources, created on first use."""
    return WorkerResources()
