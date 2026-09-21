"""Celery application and per-process resources for the TTS stage.

A separate deployment from `story-worker` on purpose: the LLM stage is network-bound and
cheap, this one is GPU-bound and expensive, and a backlog here must not stall story
generation. They consume different queues and scale on different signals.
"""

from __future__ import annotations

from functools import lru_cache

from celery import Celery
from redis import Redis
from sqlalchemy.orm import Session, sessionmaker

from story2audio_shared.config import core_settings, redis_settings, storage_settings
from story2audio_shared.logging import configure_logging, get_logger
from story2audio_shared.storage import ObjectStorage
from story2audio_shared.worker import (
    JobLease,
    SyncEventPublisher,
    create_worker_engine,
    create_worker_redis,
    create_worker_session_factory,
)
from tts_worker.settings import tts_worker_settings

log = get_logger(__name__)

TTS_QUEUE = "tts"
SYNTHESIZE_TASK = "tts_worker.tasks.synthesize"


def build_celery_app() -> Celery:
    redis_config = redis_settings()
    worker_config = tts_worker_settings()

    app = Celery(
        "tts_worker",
        broker=redis_config.celery_broker_url,
        backend=redis_config.celery_result_backend,
        include=["tts_worker.tasks"],
    )

    app.conf.update(
        task_default_queue=TTS_QUEUE,
        task_routes={SYNTHESIZE_TASK: {"queue": TTS_QUEUE}},
        task_serializer="json",
        result_serializer="json",
        accept_content=["json"],
        timezone="UTC",
        enable_utc=True,
        # At-least-once: a worker killed mid-synthesis gets the job redelivered rather
        # than losing it. Synthesis restarts from the beginning, but the story is already
        # durable at `written`, so the model is never called again.
        task_acks_late=True,
        task_reject_on_worker_lost=True,
        # One job at a time. These are minutes long and the useful concurrency knob is
        # the number of worker processes, not prefetch depth.
        worker_prefetch_multiplier=1,
        task_time_limit=worker_config.job_lease_ttl_seconds + 120,
        task_soft_time_limit=worker_config.job_lease_ttl_seconds,
        broker_connection_retry_on_startup=True,
        worker_hijack_root_logger=False,
    )
    return app


celery_app = build_celery_app()


class WorkerResources:
    """Per-process resources, built once and reused across tasks.

    The gRPC channel is deliberately *not* here: it is created per task, so a
    scale-to-zero engine that comes back at a new address is picked up on the next job
    rather than needing a worker restart.
    """

    def __init__(self) -> None:
        core = core_settings()
        configure_logging(level=core.log_level, json_output=core.log_format.value == "json")

        self.settings = tts_worker_settings()
        self.engine = create_worker_engine()
        self.session_factory: sessionmaker[Session] = create_worker_session_factory(self.engine)
        self.redis: Redis = create_worker_redis(redis_settings())
        self.publisher = SyncEventPublisher(self.redis)
        self.lease = JobLease(self.redis, ttl_seconds=self.settings.job_lease_ttl_seconds)
        self.storage = ObjectStorage(storage_settings())


@lru_cache(maxsize=1)
def resources() -> WorkerResources:
    return WorkerResources()
