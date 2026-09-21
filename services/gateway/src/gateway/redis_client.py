"""Redis connection management.

Redis carries three things for the gateway: the Celery broker (written to indirectly),
the per-job pub/sub channel that feeds WebSocket clients, and the rate-limit counters.
None of it is durable state -- losing Redis loses in-flight work, never history.
"""

from __future__ import annotations

from redis.asyncio import ConnectionPool, Redis

from story2audio_shared.config import RedisSettings, redis_settings


def create_redis_pool(settings: RedisSettings | None = None) -> ConnectionPool:
    """Build a connection pool.

    ``health_check_interval`` guards against the same idle-drop problem the database
    engine has: a pooled connection that a managed provider closed underneath us should
    be detected and replaced rather than surfacing as a failed request.
    """
    config = settings or redis_settings()
    return ConnectionPool.from_url(
        config.redis_url,
        decode_responses=True,
        health_check_interval=30,
    )


def create_redis(pool: ConnectionPool) -> Redis:
    """Build a client over an existing pool."""
    return Redis(connection_pool=pool)
