"""Test fixtures for the gateway.

The application is constructed exactly as production constructs it — same
:func:`create_app`, same routers, same middleware, same dependency graph — with three
resources substituted:

=========  ============  =================================================================
Postgres   SQLite        File-backed, one database per test, WAL mode
Redis      fakeredis     Real command semantics including Lua, no server
R2/MinIO   stub          Records puts and hands back deterministic presigned URLs
=========  ============  =================================================================

**Known divergences** from the deployed stack, none of which these tests can catch:

- SQLite ignores ``postgresql_where``, so the idempotency index is created as a full
  unique index. Both engines treat NULLs as distinct, so the observable behaviour matches.
- ``SELECT ... FOR UPDATE`` is a no-op in SQLite, so lock contention is not exercised.
- SQLite is single-writer, so genuine write concurrency is not exercised either.
- The test database is a file rather than ``:memory:``; see the ``engine`` fixture.

Phase 5 adds an ``-m integration`` suite against the real compose stack for exactly these.
"""

from __future__ import annotations

import asyncio
import io
import math
import pathlib
import struct
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import IO, Any, cast
from uuid import UUID

import fakeredis.aioredis
import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from httpx_ws.transport import ASGIWebSocketTransport
from sqlalchemy import event
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from gateway.deps import AppState
from gateway.dispatch import InlineDispatcher
from gateway.main import create_app
from gateway.publisher import EventPublisher
from gateway.ratelimit import RateLimiter
from gateway.session import SessionCodec
from gateway.settings import GatewaySettings, JobDispatch
from story2audio_shared.config import LimitSettings
from story2audio_shared.ids import uuid7
from story2audio_shared.models import Base, Voice
from story2audio_shared.storage import ObjectStorage, PresignedUrl

TEST_SECRET = "test-secret-not-used-anywhere-real"


# --- Storage stub -----------------------------------------------------------------------


@dataclass
class StubStorage:
    """In-memory stand-in for :class:`ObjectStorage`.

    Records every write so tests can assert *what* was stored, not merely that a call was
    made — the difference between proving a voice was canonicalised to WAV and proving a
    method was invoked.
    """

    bucket: str = "test-bucket"
    objects: dict[str, bytes] = field(default_factory=dict)
    content_types: dict[str, str] = field(default_factory=dict)
    deleted: list[str] = field(default_factory=list)

    def put_bytes(self, key: str, data: bytes, *, content_type: str) -> None:
        self.objects[key] = data
        self.content_types[key] = content_type

    def put_stream(self, key: str, stream: IO[bytes], *, content_type: str) -> None:
        self.put_bytes(key, stream.read(), content_type=content_type)

    def put_audio(self, key: str, data: bytes, audio_format: Any) -> None:
        self.put_bytes(key, data, content_type="audio/mpeg")

    def get_bytes(self, key: str) -> bytes:
        return self.objects[key]

    def presign_get(self, key: str, *, ttl_seconds: int | None = None) -> PresignedUrl:
        ttl = ttl_seconds or 3600
        return PresignedUrl(
            url=f"https://storage.test/{key}?ttl={ttl}",
            expires_at=datetime.now(UTC) + timedelta(seconds=ttl),
        )

    def delete(self, key: str) -> None:
        self.objects.pop(key, None)
        self.deleted.append(key)

    def exists(self, key: str) -> bool:
        return key in self.objects

    def ensure_bucket(self) -> None:
        return None


# --- Audio helpers -------------------------------------------------------------------------


def make_wav(duration_seconds: float, *, sample_rate: int = 22_050, channels: int = 1) -> bytes:
    """A real, decodable PCM16 WAV of a given duration.

    Written by hand rather than with soundfile so the fixture cannot be satisfied by the
    same library the code under test uses to decode it.
    """
    frame_count = max(1, int(duration_seconds * sample_rate))
    samples = bytearray()
    for index in range(frame_count):
        value = int(12_000 * math.sin(2 * math.pi * 220 * index / sample_rate))
        for _ in range(channels):
            samples += struct.pack("<h", value)

    data = bytes(samples)
    block_align = 2 * channels
    header = io.BytesIO()
    header.write(b"RIFF")
    header.write(struct.pack("<I", 36 + len(data)))
    header.write(b"WAVEfmt ")
    header.write(
        struct.pack(
            "<IHHIIHH",
            16,
            1,
            channels,
            sample_rate,
            sample_rate * block_align,
            block_align,
            16,
        )
    )
    header.write(b"data")
    header.write(struct.pack("<I", len(data)))
    header.write(data)
    return header.getvalue()


# --- Core fixtures ---------------------------------------------------------------------------


@pytest.fixture
def gateway_settings_fixture() -> GatewaySettings:
    return GatewaySettings(
        session_secret=TEST_SECRET,  # type: ignore[arg-type]
        job_dispatch=JobDispatch.INLINE,
        cors_allow_origins=["http://testserver"],
        public_web_origin="http://testserver",
    )


@pytest.fixture
def limits() -> LimitSettings:
    """Deliberately generous, so a test that hits a limit meant to."""
    return LimitSettings(
        rate_limit_jobs_per_hour=100,
        rate_limit_uploads_per_hour=100,
        max_concurrent_jobs_per_session=50,
        global_daily_job_cap=1_000,
        max_prompt_chars=2_000,
        min_voice_duration_seconds=6.0,
        max_voice_duration_seconds=120.0,
    )


@pytest_asyncio.fixture
async def engine(tmp_path: pathlib.Path) -> AsyncIterator[AsyncEngine]:
    """A file-backed SQLite database, one per test.

    Not ``:memory:``. An in-memory database has to be pinned to a single connection to be
    visible at all, and one connection cannot serve concurrent async sessions — the
    background dispatcher and an in-flight request interleave statements and SQLite
    rejects the commit with "SQL statements in progress". A file lets each session take
    its own connection, which is how Postgres behaves in production.

    WAL mode so a reader does not block the writer; a busy timeout so brief write
    contention waits instead of failing.
    """
    db_path = tmp_path / "test.sqlite3"
    engine = create_async_engine(
        f"sqlite+aiosqlite:///{db_path}",
        connect_args={"check_same_thread": False, "timeout": 30},
    )

    @event.listens_for(engine.sync_engine, "connect")
    def _configure(dbapi_connection: Any, _record: Any) -> None:
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA busy_timeout=30000")
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    yield engine
    await engine.dispose()


@pytest.fixture
def session_factory(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    return async_sessionmaker(engine, expire_on_commit=False, autoflush=False)


@pytest_asyncio.fixture
async def redis() -> AsyncIterator[fakeredis.aioredis.FakeRedis]:
    client = fakeredis.aioredis.FakeRedis(decode_responses=True)
    yield client
    await client.aclose()


@pytest.fixture
def storage() -> StubStorage:
    return StubStorage()


@pytest_asyncio.fixture
async def app_state(
    gateway_settings_fixture: GatewaySettings,
    limits: LimitSettings,
    engine: AsyncEngine,
    session_factory: async_sessionmaker[AsyncSession],
    redis: fakeredis.aioredis.FakeRedis,
    storage: StubStorage,
) -> AppState:
    publisher = EventPublisher(redis)
    return AppState(
        settings=gateway_settings_fixture,
        limits=limits,
        engine=engine,
        session_factory=session_factory,
        redis_pool=redis.connection_pool,
        redis=redis,
        # Structurally compatible with ObjectStorage but not a subclass of it: the
        # stub exists precisely so no boto3 client is constructed.
        storage=cast(ObjectStorage, storage),
        publisher=publisher,
        rate_limiter=RateLimiter(redis),
        dispatcher=InlineDispatcher(session_factory, publisher),
        session_codec=SessionCodec(gateway_settings_fixture),
        presigned_ttl_seconds=3600,
    )


@pytest_asyncio.fixture
async def client(app_state: AppState) -> AsyncIterator[AsyncClient]:
    """A client bound to the real ASGI app, for both HTTP and WebSocket.

    Requests go through routing, middleware and validation exactly as a deployed request
    would; only the backing resources are substituted.

    HTTP only. WebSocket tests use :func:`websocket_client` instead, for the reason
    documented there.
    """
    app = create_app(app_state)
    async with (
        AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://testserver",
            follow_redirects=True,
        ) as http_client,
        app.router.lifespan_context(app),
    ):
        yield http_client


@asynccontextmanager
async def new_session_client(app_state: AppState) -> AsyncIterator[AsyncClient]:
    """A second client with its own cookie jar, i.e. a different anonymous user.

    Used by the isolation tests: two sessions against one application, which is the only
    way to prove that one caller cannot see or use another's jobs and voices.
    """
    app = create_app(app_state)
    async with (
        AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://testserver",
            follow_redirects=True,
        ) as http_client,
        app.router.lifespan_context(app),
    ):
        yield http_client


@asynccontextmanager
async def websocket_client(app_state: AppState) -> AsyncIterator[AsyncClient]:
    """A WebSocket-capable client, entered and exited inside one test.

    Deliberately not a fixture. ``ASGIWebSocketTransport`` holds an anyio task group, and
    pytest-asyncio runs an async fixture's setup and its finalizer in different tasks —
    so a task group opened either side of a `yield` raises "attempted to exit cancel scope
    in a different task". Keeping the whole lifetime inside the test body avoids that.

    Starlette's ``TestClient`` is not an option either: it runs the app on its own loop in
    a worker thread, while the SQLite and fakeredis objects these fixtures create are
    bound to the test's loop, which deadlocks.
    """
    app = create_app(app_state)
    async with (
        AsyncClient(
            transport=ASGIWebSocketTransport(app=app),
            base_url="http://testserver",
            follow_redirects=True,
        ) as http_client,
        app.router.lifespan_context(app),
    ):
        yield http_client


@pytest_asyncio.fixture
async def builtin_voice(
    session_factory: async_sessionmaker[AsyncSession], storage: StubStorage
) -> Voice:
    """A built-in voice, so job tests need not upload one first."""
    voice_id = uuid7()
    key = f"voices/{voice_id}.wav"
    storage.put_bytes(key, make_wav(10.0), content_type="audio/wav")

    voice = Voice(
        id=voice_id,
        owner_id=None,
        name="Test Narrator",
        storage_key=key,
        duration_seconds=10.0,
        sample_rate=22_050,
        is_builtin=True,
    )
    async with session_factory() as session:
        session.add(voice)
        await session.commit()
    return voice


@pytest_asyncio.fixture
async def second_builtin_voice(
    session_factory: async_sessionmaker[AsyncSession], storage: StubStorage
) -> Voice:
    """A second built-in, for dialogue mode."""
    voice_id = uuid7()
    key = f"voices/{voice_id}.wav"
    storage.put_bytes(key, make_wav(10.0), content_type="audio/wav")

    voice = Voice(
        id=voice_id,
        owner_id=None,
        name="Test Dialogue",
        storage_key=key,
        duration_seconds=10.0,
        sample_rate=22_050,
        is_builtin=True,
    )
    async with session_factory() as session:
        session.add(voice)
        await session.commit()
    return voice


async def wait_for_status(
    client: AsyncClient, job_id: UUID | str, target: set[str], *, deadline_seconds: float = 5.0
) -> dict[str, Any]:
    """Poll until the job reaches one of ``target``.

    Polling rather than sleeping a fixed interval: the inline dispatcher runs as fast as
    the event loop allows, and a fixed sleep would be both slower and flakier.
    """
    deadline = asyncio.get_event_loop().time() + deadline_seconds
    payload: dict[str, Any] = {}
    while asyncio.get_event_loop().time() < deadline:
        response = await client.get(f"/v1/jobs/{job_id}")
        payload = response.json()
        if payload.get("status") in target:
            return payload
        await asyncio.sleep(0.01)
    raise AssertionError(f"job never reached {target}; last status {payload.get('status')!r}")
