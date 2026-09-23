"""Async database engine and session factory.

The gateway is the only service that runs migrations (ADR-0001); the workers use a
synchronous engine against the same database.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from story2audio_shared.config import DatabaseSettings, database_settings


def create_engine(settings: DatabaseSettings | None = None) -> AsyncEngine:
    """Build the async engine.

    ``pool_pre_ping`` matters on managed Postgres: Neon and friends drop idle connections,
    and without it the first request after an idle period fails on a dead socket.
    """
    config = settings or database_settings()
    return create_async_engine(
        config.database_url,
        echo=config.db_echo,
        pool_size=config.db_pool_size,
        max_overflow=config.db_max_overflow,
        pool_pre_ping=True,
    )


def create_session_factory(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    """Build the session factory.

    ``expire_on_commit=False`` so that a response can still read attributes off an object
    after the transaction closes, without triggering a lazy refresh against a session that
    is already gone.
    """
    return async_sessionmaker(engine, expire_on_commit=False, autoflush=False)


@asynccontextmanager
async def session_scope(
    factory: async_sessionmaker[AsyncSession],
) -> AsyncIterator[AsyncSession]:
    """Run a unit of work: commit on success, roll back on any exception."""
    async with factory() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise
        else:
            await session.commit()
