"""The LLM stage.

Driven directly against a scripted provider, with no Celery and no network, so every
branch the worker can take is reachable and deterministic: retries, cancellation mid
stream, truncation, and — the one that matters most under at-least-once delivery —
redelivery of work that is already done.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import UUID

import fakeredis
import pytest
from redis.client import PubSub
from sqlalchemy import Engine, create_engine, event, select
from sqlalchemy.orm import Session, sessionmaker

from story2audio_shared.enums import Emotion, JobStatus, Language, StoryLength, VoiceMode
from story2audio_shared.errors import AppError, ErrorCode
from story2audio_shared.events import job_channel
from story2audio_shared.ids import uuid7
from story2audio_shared.models import Base, Job, User, Voice
from story2audio_shared.prompts import StoryPrompt
from story2audio_shared.worker import (
    SyncEventPublisher,
    TokenBatcher,
    advance_status,
    create_worker_session_factory,
    fail_job,
)
from story_worker.pipeline import run_story_stage
from story_worker.providers.base import StreamStats
from story_worker.providers.fake import FakeProvider
from story_worker.settings import LLMProviderName, StoryWorkerSettings

COMPLETE_STORY = ["The lighthouse keeper woke. ", "The lamp was dark. ", "He lit it again."]


# --- Fixtures ------------------------------------------------------------------------------------


@pytest.fixture
def sync_engine(tmp_path: Path) -> Iterator[Engine]:
    """A synchronous SQLite database, mirroring the worker's real engine shape."""
    engine = create_engine(f"sqlite:///{tmp_path / 'worker.sqlite3'}", connect_args={"timeout": 30})

    @event.listens_for(engine, "connect")
    def _configure(dbapi_connection: Any, _record: Any) -> None:
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA journal_mode=WAL")
        # Enforced so the tests catch a broken foreign key rather than silently
        # accepting an orphaned row, which is what Postgres would do in production.
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    Base.metadata.create_all(engine)
    yield engine
    engine.dispose()


@pytest.fixture
def sync_sessions(sync_engine: Engine) -> sessionmaker[Session]:
    return create_worker_session_factory(sync_engine)


@pytest.fixture
def sync_redis() -> Iterator[fakeredis.FakeRedis]:
    client = fakeredis.FakeRedis(decode_responses=True)
    yield client
    client.close()


@pytest.fixture
def publisher(sync_redis: fakeredis.FakeRedis) -> SyncEventPublisher:
    return SyncEventPublisher(sync_redis)


@pytest.fixture
def worker_settings() -> StoryWorkerSettings:
    return StoryWorkerSettings(
        llm_provider=LLMProviderName.FAKE,
        llm_model="fake-model",
        # Flush every chunk, so the frame sequence in a test is the chunk sequence.
        token_batch_interval_seconds=0.0001,
        token_batch_max_chars=1,
        cancel_check_every_frames=1,
        max_continuation_passes=1,
    )


@pytest.fixture
def job(sync_sessions: sessionmaker[Session]) -> Job:
    """A queued job with an owner and a voice, ready for the stage to pick up."""
    user = User(id=uuid7())
    voice = Voice(
        id=uuid7(),
        owner_id=None,
        name="Narrator",
        storage_key="voices/narrator.wav",
        duration_seconds=12.0,
        sample_rate=22_050,
        is_builtin=True,
    )
    record = Job(
        id=uuid7(),
        owner_id=user.id,
        status=JobStatus.QUEUED,
        prompt="A lighthouse keeper finds the lamp extinguished.",
        length=StoryLength.SHORT,
        mode=VoiceMode.NARRATION,
        language=Language.EN,
        emotion=Emotion.NEUTRAL,
        speed=1.0,
        voice_id=voice.id,
        queued_at=datetime.now(UTC),
    )
    # Inserted in dependency order with explicit flushes. `add_all` leaves the order to
    # the unit of work, and SQLite with `PRAGMA foreign_keys=ON` rejects a job row whose
    # referenced voice has not been written yet.
    with sync_sessions() as session:
        session.add(user)
        session.flush()
        session.add(voice)
        session.flush()
        session.add(record)
        session.commit()
    return record


def reload_job(sessions: sessionmaker[Session], job_id: UUID) -> Job:
    with sessions() as session:
        loaded = session.scalar(select(Job).where(Job.id == job_id))
    assert loaded is not None
    return loaded


def subscribe(redis: fakeredis.FakeRedis, job_id: UUID) -> PubSub:
    """Attach a subscriber to a job's channel before the stage runs.

    `ignore_subscribe_messages=True` is deliberately not used: fakeredis drops delivered
    messages entirely when it is set. The confirmation frame is filtered out in
    :func:`drain` instead.
    """
    # fakeredis ships no type information, so these two calls are untyped to mypy.
    pubsub: PubSub = redis.pubsub()  # type: ignore[no-untyped-call]
    pubsub.subscribe(job_channel(job_id))  # type: ignore[no-untyped-call]
    return pubsub


def drain(pubsub: PubSub) -> list[dict[str, Any]]:
    """Collect every published event the subscriber has buffered."""
    frames: list[dict[str, Any]] = []
    while (message := pubsub.get_message(timeout=0.01)) is not None:
        if message.get("type") != "message":
            continue
        data = message.get("data")
        if isinstance(data, str):
            frames.append(json.loads(data))
    pubsub.close()
    return frames


# --- The happy path ------------------------------------------------------------------------------


def test_the_story_is_generated_and_persisted(
    sync_sessions: sessionmaker[Session],
    publisher: SyncEventPublisher,
    worker_settings: StoryWorkerSettings,
    job: Job,
) -> None:
    provider = FakeProvider(COMPLETE_STORY)

    outcome = run_story_stage(sync_sessions, publisher, provider, worker_settings, job.id)

    assert outcome.ready_for_synthesis
    assert outcome.performed_work

    written = reload_job(sync_sessions, job.id)
    assert written.status is JobStatus.WRITTEN
    assert written.story_text == "".join(COMPLETE_STORY).strip()
    assert written.llm_model == "fake-model"
    assert written.llm_output_tokens == 128


def test_stage_timestamps_are_recorded(
    sync_sessions: sessionmaker[Session],
    publisher: SyncEventPublisher,
    worker_settings: StoryWorkerSettings,
    job: Job,
) -> None:
    """These are the measurements behind the v1-versus-v2 comparison."""
    run_story_stage(sync_sessions, publisher, FakeProvider(COMPLETE_STORY), worker_settings, job.id)

    written = reload_job(sync_sessions, job.id)
    assert written.writing_at is not None
    assert written.written_at is not None
    assert written.written_at >= written.writing_at


def test_instructions_and_storyline_are_separate_messages(
    sync_sessions: sessionmaker[Session],
    publisher: SyncEventPublisher,
    worker_settings: StoryWorkerSettings,
    job: Job,
) -> None:
    """v1 concatenated them into one string, so the user's text read as instructions."""
    provider = FakeProvider(COMPLETE_STORY)
    run_story_stage(sync_sessions, publisher, provider, worker_settings, job.id)

    prompt = provider.calls[0]
    assert job.prompt in prompt.user
    assert job.prompt not in prompt.system
    assert "storyteller" in prompt.system


def test_the_provider_is_closed_by_the_caller(
    sync_sessions: sessionmaker[Session],
    publisher: SyncEventPublisher,
    worker_settings: StoryWorkerSettings,
    job: Job,
) -> None:
    """The stage itself does not own the client's lifetime; the task does."""
    provider = FakeProvider(COMPLETE_STORY)
    run_story_stage(sync_sessions, publisher, provider, worker_settings, job.id)
    assert provider.closed is False


# --- Idempotency under at-least-once delivery ----------------------------------------------------


def test_redelivery_of_a_written_job_does_not_call_the_model_again(
    sync_sessions: sessionmaker[Session],
    publisher: SyncEventPublisher,
    worker_settings: StoryWorkerSettings,
    job: Job,
) -> None:
    """The row is the dedupe key. Re-running here would pay twice for text we have."""
    first = FakeProvider(COMPLETE_STORY)
    run_story_stage(sync_sessions, publisher, first, worker_settings, job.id)

    second = FakeProvider(["SHOULD NOT BE USED"])
    outcome = run_story_stage(sync_sessions, publisher, second, worker_settings, job.id)

    assert second.calls == []
    # Still ready for synthesis, so a redelivery re-enqueues TTS rather than stranding
    # the job.
    assert outcome.ready_for_synthesis
    assert not outcome.performed_work
    assert reload_job(sync_sessions, job.id).story_text == "".join(COMPLETE_STORY).strip()


@pytest.mark.parametrize(
    "status", [JobStatus.DONE, JobStatus.FAILED, JobStatus.CANCELLED, JobStatus.SYNTHESIZING]
)
def test_a_job_past_this_stage_is_left_alone(
    sync_sessions: sessionmaker[Session],
    publisher: SyncEventPublisher,
    worker_settings: StoryWorkerSettings,
    job: Job,
    status: JobStatus,
) -> None:
    with sync_sessions() as session:
        session.execute(select(Job).where(Job.id == job.id))  # keep the row loaded in this session
        session.query(Job).filter(Job.id == job.id).update({"status": status})
        session.commit()

    provider = FakeProvider(COMPLETE_STORY)
    outcome = run_story_stage(sync_sessions, publisher, provider, worker_settings, job.id)

    assert provider.calls == []
    assert not outcome.ready_for_synthesis
    assert reload_job(sync_sessions, job.id).status is status


def test_a_missing_job_is_not_an_error(
    sync_sessions: sessionmaker[Session],
    publisher: SyncEventPublisher,
    worker_settings: StoryWorkerSettings,
) -> None:
    outcome = run_story_stage(sync_sessions, publisher, FakeProvider(), worker_settings, uuid7())
    assert not outcome.ready_for_synthesis


def test_an_interrupted_attempt_resumes(
    sync_sessions: sessionmaker[Session],
    publisher: SyncEventPublisher,
    worker_settings: StoryWorkerSettings,
    job: Job,
) -> None:
    """A worker killed mid-stream leaves the job at `writing`; the retry picks it up."""
    with sync_sessions() as session:
        advance_status(
            session,
            job.id,
            expected=JobStatus.QUEUED,
            target=JobStatus.WRITING,
            writing_at=datetime.now(UTC),
        )
        session.commit()

    provider = FakeProvider(COMPLETE_STORY)
    outcome = run_story_stage(sync_sessions, publisher, provider, worker_settings, job.id)

    assert outcome.performed_work
    assert len(provider.calls) == 1
    assert reload_job(sync_sessions, job.id).status is JobStatus.WRITTEN


# --- Cancellation --------------------------------------------------------------------------------


def test_cancelling_mid_stream_stops_generation(
    sync_sessions: sessionmaker[Session],
    publisher: SyncEventPublisher,
    worker_settings: StoryWorkerSettings,
    job: Job,
) -> None:
    """v1 had no cancellation at all: closing the tab left the work running."""

    class CancellingProvider(FakeProvider):
        """Cancels the job from underneath itself after the first chunk."""

        def __init__(self) -> None:
            super().__init__(["first chunk ", "second chunk ", "third chunk"])

        def stream(self, prompt: StoryPrompt, stats: StreamStats) -> Iterator[str]:
            for index, chunk in enumerate(super().stream(prompt, stats)):
                if index == 1:
                    with sync_sessions() as session:
                        session.query(Job).filter(Job.id == job.id).update(
                            {"status": JobStatus.CANCELLED}
                        )
                        session.commit()
                yield chunk

    with pytest.raises(AppError) as info:
        run_story_stage(sync_sessions, publisher, CancellingProvider(), worker_settings, job.id)

    assert info.value.code is ErrorCode.CANCELLED
    # The cancellation stands; a partial story is not written over it.
    assert reload_job(sync_sessions, job.id).status is JobStatus.CANCELLED


def test_a_job_cancelled_during_the_final_write_is_not_overwritten(
    sync_sessions: sessionmaker[Session],
    publisher: SyncEventPublisher,
    worker_settings: StoryWorkerSettings,
    job: Job,
) -> None:
    """The conditional UPDATE is what protects a user's cancel from a late completion."""

    class LateCancelProvider(FakeProvider):
        def stream(self, prompt: StoryPrompt, stats: StreamStats) -> Iterator[str]:
            yield from super().stream(prompt, stats)
            with sync_sessions() as session:
                session.query(Job).filter(Job.id == job.id).update({"status": JobStatus.CANCELLED})
                session.commit()

    with pytest.raises(AppError) as info:
        run_story_stage(
            sync_sessions, publisher, LateCancelProvider(COMPLETE_STORY), worker_settings, job.id
        )

    assert info.value.code is ErrorCode.CANCELLED
    final = reload_job(sync_sessions, job.id)
    assert final.status is JobStatus.CANCELLED
    assert final.story_text is None


# --- Truncation and continuation -----------------------------------------------------------------


def test_a_truncated_story_gets_one_continuation_pass(
    sync_sessions: sessionmaker[Session],
    publisher: SyncEventPublisher,
    worker_settings: StoryWorkerSettings,
    job: Job,
) -> None:
    """v1's flat 2000-token cap truncated its own 800-1200 word target mid-sentence."""
    provider = FakeProvider(
        chunks_per_call=[
            ["He climbed the stairs and"],
            [" reached the lamp, and lit it."],
        ],
        finish_reason="length",
    )

    run_story_stage(sync_sessions, publisher, provider, worker_settings, job.id)

    assert len(provider.calls) == 2
    story = reload_job(sync_sessions, job.id).story_text
    assert story is not None
    assert story.endswith(".")
    assert "climbed the stairs" in story
    assert "lit it" in story


def test_a_complete_story_is_not_continued(
    sync_sessions: sessionmaker[Session],
    publisher: SyncEventPublisher,
    worker_settings: StoryWorkerSettings,
    job: Job,
) -> None:
    provider = FakeProvider(COMPLETE_STORY)
    run_story_stage(sync_sessions, publisher, provider, worker_settings, job.id)
    assert len(provider.calls) == 1


def test_continuation_is_bounded(
    sync_sessions: sessionmaker[Session],
    publisher: SyncEventPublisher,
    worker_settings: StoryWorkerSettings,
    job: Job,
) -> None:
    """A model that never lands an ending must not become an open-ended bill."""
    provider = FakeProvider(["it never ends and"], finish_reason="length")

    run_story_stage(sync_sessions, publisher, provider, worker_settings, job.id)

    # One initial call plus max_continuation_passes, and no more.
    assert len(provider.calls) == 1 + worker_settings.max_continuation_passes


def test_an_empty_response_fails_the_stage(
    sync_sessions: sessionmaker[Session],
    publisher: SyncEventPublisher,
    worker_settings: StoryWorkerSettings,
    job: Job,
) -> None:
    with pytest.raises(AppError) as info:
        run_story_stage(sync_sessions, publisher, FakeProvider([""]), worker_settings, job.id)
    assert info.value.code is ErrorCode.STORY_EMPTY


def test_an_empty_generation_is_never_fed_to_the_continuation_prompt(
    sync_sessions: sessionmaker[Session],
    publisher: SyncEventPublisher,
    worker_settings: StoryWorkerSettings,
    job: Job,
) -> None:
    """The worst bug the first real Groq run exposed.

    A reasoning model spent its entire token budget thinking and returned nothing. The
    empty result was handed to the continuation prompt, which asks the model to pick up
    from an excerpt that is not there — so it answered with a refusal, and that refusal
    was persisted as the story. A clean failure must not become corrupt output.
    """
    provider = FakeProvider([""], finish_reason="length")

    with pytest.raises(AppError) as info:
        run_story_stage(sync_sessions, publisher, provider, worker_settings, job.id)

    assert info.value.code is ErrorCode.STORY_EMPTY
    # Exactly one call: no continuation was attempted on an empty story.
    assert len(provider.calls) == 1
    assert reload_job(sync_sessions, job.id).story_text is None


def test_the_empty_story_error_carries_diagnostics(
    sync_sessions: sessionmaker[Session],
    publisher: SyncEventPublisher,
    worker_settings: StoryWorkerSettings,
    job: Job,
) -> None:
    """`finish_reason=length` with no content is the signature of a reasoning overrun."""
    provider = FakeProvider([""], finish_reason="length", output_tokens=1000)

    with pytest.raises(AppError) as info:
        run_story_stage(sync_sessions, publisher, provider, worker_settings, job.id)

    detail = info.value.detail or ""
    assert "finish_reason=length" in detail
    assert "reasoning_tokens" in detail


def test_provider_failures_propagate_classified(
    sync_sessions: sessionmaker[Session],
    publisher: SyncEventPublisher,
    worker_settings: StoryWorkerSettings,
    job: Job,
) -> None:
    """The retry policy and the user's message both derive from one classification."""
    provider = FakeProvider(
        fail_times=1, failure=AppError(ErrorCode.LLM_TIMEOUT, detail="scripted timeout")
    )

    with pytest.raises(AppError) as info:
        run_story_stage(sync_sessions, publisher, provider, worker_settings, job.id)

    assert info.value.code is ErrorCode.LLM_TIMEOUT


# --- Events --------------------------------------------------------------------------------------


def test_the_stage_publishes_a_writing_status_then_tokens_then_story_done(
    sync_sessions: sessionmaker[Session],
    sync_redis: fakeredis.FakeRedis,
    publisher: SyncEventPublisher,
    worker_settings: StoryWorkerSettings,
    job: Job,
) -> None:
    pubsub = subscribe(sync_redis, job.id)

    run_story_stage(sync_sessions, publisher, FakeProvider(COMPLETE_STORY), worker_settings, job.id)

    frames = drain(pubsub)

    types = [frame["type"] for frame in frames]
    assert types[0] == "status"
    assert "token" in types
    assert types[-1] == "story_done"
    assert types.index("token") < types.index("story_done")


def test_event_sequence_numbers_are_strictly_increasing(
    sync_sessions: sessionmaker[Session],
    sync_redis: fakeredis.FakeRedis,
    publisher: SyncEventPublisher,
    worker_settings: StoryWorkerSettings,
    job: Job,
) -> None:
    pubsub = subscribe(sync_redis, job.id)

    run_story_stage(sync_sessions, publisher, FakeProvider(COMPLETE_STORY), worker_settings, job.id)

    sequences = [frame["seq"] for frame in drain(pubsub)]

    assert sequences == sorted(sequences)
    assert len(set(sequences)) == len(sequences)


def test_the_story_done_event_carries_the_word_count(
    sync_sessions: sessionmaker[Session],
    sync_redis: fakeredis.FakeRedis,
    publisher: SyncEventPublisher,
    worker_settings: StoryWorkerSettings,
    job: Job,
) -> None:
    pubsub = subscribe(sync_redis, job.id)

    run_story_stage(sync_sessions, publisher, FakeProvider(COMPLETE_STORY), worker_settings, job.id)

    final = next(frame for frame in drain(pubsub) if frame["type"] == "story_done")

    assert final is not None
    assert final["word_count"] == len("".join(COMPLETE_STORY).split())


# --- fail_job ------------------------------------------------------------------------------------


def test_failing_a_job_records_the_code_and_a_public_message(
    sync_sessions: sessionmaker[Session],
    publisher: SyncEventPublisher,
    job: Job,
) -> None:
    with sync_sessions() as session:
        fail_job(
            session,
            publisher,
            job.id,
            code=ErrorCode.LLM_UNAVAILABLE,
            detail="connection refused to http://internal:11434",
        )
        session.commit()

    failed = reload_job(sync_sessions, job.id)
    assert failed.status is JobStatus.FAILED
    assert failed.error_code is ErrorCode.LLM_UNAVAILABLE
    # The internal detail is logged, never persisted where the API would return it.
    assert failed.error_message is not None
    assert "11434" not in failed.error_message
    assert failed.finished_at is not None


def test_failing_a_cancelled_job_leaves_the_cancellation_intact(
    sync_sessions: sessionmaker[Session],
    publisher: SyncEventPublisher,
    job: Job,
) -> None:
    """A late error must not overwrite an outcome the user asked for."""
    with sync_sessions() as session:
        session.query(Job).filter(Job.id == job.id).update({"status": JobStatus.CANCELLED})
        session.commit()

    with sync_sessions() as session:
        fail_job(session, publisher, job.id, code=ErrorCode.INTERNAL, detail="too late")
        session.commit()

    assert reload_job(sync_sessions, job.id).status is JobStatus.CANCELLED


# --- Transitions ---------------------------------------------------------------------------------


def test_a_transition_from_the_wrong_state_does_not_apply(
    sync_sessions: sessionmaker[Session], job: Job
) -> None:
    """The guard that makes at-least-once delivery safe."""
    with sync_sessions() as session:
        first = advance_status(session, job.id, expected=JobStatus.QUEUED, target=JobStatus.WRITING)
        second = advance_status(
            session, job.id, expected=JobStatus.QUEUED, target=JobStatus.WRITING
        )
        session.commit()

    assert first.applied
    assert not second.applied
    assert second.previous is JobStatus.WRITING


def test_an_illegal_transition_is_a_programming_error(
    sync_sessions: sessionmaker[Session], job: Job
) -> None:
    with sync_sessions() as session, pytest.raises(ValueError, match="illegal transition"):
        advance_status(session, job.id, expected=JobStatus.QUEUED, target=JobStatus.DONE)


# --- Token batching ------------------------------------------------------------------------------


def test_batching_coalesces_chunks_below_the_threshold() -> None:
    """One frame per token would flood the socket without the text arriving sooner."""
    batcher = TokenBatcher(interval_seconds=60.0, max_chars=10)

    assert batcher.add("abc") is None
    assert batcher.add("def") is None
    assert batcher.add("ghij") == "abcdefghij"


def test_batching_flushes_the_remainder() -> None:
    batcher = TokenBatcher(interval_seconds=60.0, max_chars=100)
    batcher.add("trailing")
    assert batcher.flush() == "trailing"
    assert batcher.flush() is None


def test_batching_loses_no_text() -> None:
    batcher = TokenBatcher(interval_seconds=60.0, max_chars=5)
    source = ["one ", "two ", "three ", "four"]

    out = [frame for chunk in source if (frame := batcher.add(chunk)) is not None]
    if tail := batcher.flush():
        out.append(tail)

    assert "".join(out) == "".join(source)
