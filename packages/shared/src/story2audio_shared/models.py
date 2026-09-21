"""Database models.

These live in `shared` and are imported by the gateway and both workers. That is a shared
database across service boundaries, which is a deliberate trade-off rather than an
oversight: the alternative is the gateway exposing an internal write-API that workers call
to report progress, which buys independence these co-deployed services do not need and
costs a network hop, a retry surface and a new failure mode. See ADR-0001, which also
records the mitigation — each service connects with a role scoped to the columns it writes.

Migrations are owned solely by the gateway; workers never issue DDL.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID

from sqlalchemy import (
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    Uuid,
    func,
    text,
)
from sqlalchemy import (
    Enum as SAEnum,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

from story2audio_shared.enums import (
    Emotion,
    JobStatus,
    Language,
    StoryLength,
    VoiceMode,
)
from story2audio_shared.errors import ErrorCode
from story2audio_shared.ids import uuid7


def _enum_column(enum_type: type[Any], name: str) -> SAEnum:
    """Build a SQL enum that stores the member *value*, not the member name.

    Without ``values_callable`` SQLAlchemy persists ``"SHORT"`` for ``StoryLength.SHORT``,
    while every other layer — the API, the events, the frontend — uses ``"short"``. This
    keeps one spelling end to end.
    """
    return SAEnum(
        enum_type,
        name=name,
        values_callable=lambda enum_cls: [member.value for member in enum_cls],
        native_enum=False,
        length=32,
    )


class Base(DeclarativeBase):
    """Declarative base for all models."""


class TimestampMixin:
    """``created_at`` / ``updated_at``, maintained by the database.

    Server-side defaults rather than Python defaults, so a row written by a worker and a
    row written by the gateway carry timestamps from the same clock.
    """

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )


class User(Base, TimestampMixin):
    """An identity.

    Anonymous at launch: the gateway issues a signed session cookie carrying this id, and
    that is enough to scope voices, job history and quotas to one person. ``email`` exists
    so that adding real accounts later is a migration, not a redesign.
    """

    __tablename__ = "users"

    id: Mapped[UUID] = mapped_column(Uuid, primary_key=True, default=uuid7)
    email: Mapped[str | None] = mapped_column(String(320), unique=True, nullable=True)
    last_seen_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    voices: Mapped[list[Voice]] = relationship(back_populates="owner")
    jobs: Mapped[list[Job]] = relationship(back_populates="owner")


class Voice(Base, TimestampMixin):
    """A reference voice available for cloning.

    ``storage_key`` is an object-storage key, never a filesystem path. v1 sent the path
    over the wire from the client and fed it straight to the TTS engine, and wrote every
    upload to a single hardcoded ``uploaded_speaker.wav``.
    """

    __tablename__ = "voices"
    __table_args__ = (
        UniqueConstraint("owner_id", "name", name="uq_voices_owner_name"),
        CheckConstraint("duration_seconds > 0", name="ck_voices_duration_positive"),
        # A built-in voice has no owner; an uploaded one must have exactly one.
        CheckConstraint(
            "(is_builtin AND owner_id IS NULL) OR (NOT is_builtin AND owner_id IS NOT NULL)",
            name="ck_voices_ownership",
        ),
        Index("ix_voices_owner_id", "owner_id"),
    )

    id: Mapped[UUID] = mapped_column(Uuid, primary_key=True, default=uuid7)
    owner_id: Mapped[UUID | None] = mapped_column(
        Uuid, ForeignKey("users.id", ondelete="CASCADE"), nullable=True
    )
    name: Mapped[str] = mapped_column(String(60), nullable=False)
    storage_key: Mapped[str] = mapped_column(String(512), nullable=False)
    duration_seconds: Mapped[float] = mapped_column(Float, nullable=False)
    sample_rate: Mapped[int] = mapped_column(Integer, nullable=False)
    is_builtin: Mapped[bool] = mapped_column(nullable=False, default=False)

    owner: Mapped[User | None] = relationship(back_populates="voices")


class Job(Base, TimestampMixin):
    """One story-to-audio generation.

    The row is the single source of truth for job state. Redis carries the queue and the
    live progress channel, but losing Redis loses only in-flight work, never history.
    """

    __tablename__ = "jobs"
    __table_args__ = (
        # Cursor pagination: ids are UUIDv7 and therefore time-ordered, so `(owner_id, id)`
        # is both the filter and the sort. No companion timestamp column, no tiebreak.
        Index("ix_jobs_owner_id_id", "owner_id", "id"),
        Index("ix_jobs_status", "status"),
        # One row per idempotency key per owner, so a retried POST returns the original
        # job instead of queueing a second run of the same GPU work. Partial, because most
        # jobs carry no key.
        Index(
            "uq_jobs_owner_idempotency_key",
            "owner_id",
            "idempotency_key",
            unique=True,
            postgresql_where=text("idempotency_key IS NOT NULL"),
        ),
        CheckConstraint("speed >= 0.5 AND speed <= 1.5", name="ck_jobs_speed_range"),
        CheckConstraint("retry_count >= 0", name="ck_jobs_retry_count_non_negative"),
    )

    id: Mapped[UUID] = mapped_column(Uuid, primary_key=True, default=uuid7)
    owner_id: Mapped[UUID] = mapped_column(
        Uuid, ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    status: Mapped[JobStatus] = mapped_column(
        _enum_column(JobStatus, "job_status"), nullable=False, default=JobStatus.QUEUED
    )

    # --- Request ---------------------------------------------------------------------
    prompt: Mapped[str] = mapped_column(Text, nullable=False)
    length: Mapped[StoryLength] = mapped_column(
        _enum_column(StoryLength, "story_length"), nullable=False
    )
    mode: Mapped[VoiceMode] = mapped_column(_enum_column(VoiceMode, "voice_mode"), nullable=False)
    language: Mapped[Language] = mapped_column(_enum_column(Language, "language"), nullable=False)
    emotion: Mapped[Emotion] = mapped_column(_enum_column(Emotion, "emotion"), nullable=False)
    speed: Mapped[float] = mapped_column(Float, nullable=False, default=1.0)
    voice_id: Mapped[UUID] = mapped_column(
        Uuid, ForeignKey("voices.id", ondelete="RESTRICT"), nullable=False
    )
    dialogue_voice_id: Mapped[UUID | None] = mapped_column(
        Uuid, ForeignKey("voices.id", ondelete="RESTRICT"), nullable=True
    )

    # --- LLM stage output -------------------------------------------------------------
    story_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    llm_model: Mapped[str | None] = mapped_column(String(120), nullable=True)
    llm_output_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)

    # --- TTS stage output ---------------------------------------------------------------
    audio_key_mp3: Mapped[str | None] = mapped_column(String(512), nullable=True)
    audio_key_wav: Mapped[str | None] = mapped_column(String(512), nullable=True)
    audio_duration_seconds: Mapped[float | None] = mapped_column(Float, nullable=True)
    segment_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    segments_done: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    # --- Failure -------------------------------------------------------------------------
    #: Classified cause. The raw exception goes to the logs only; v1 returned it verbatim.
    error_code: Mapped[ErrorCode | None] = mapped_column(
        _enum_column(ErrorCode, "error_code"), nullable=True
    )
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    retry_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    # --- Stage timings -------------------------------------------------------------------
    # These are the measurements behind the v1-versus-v2 comparison and the /stats page.
    queued_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    writing_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    written_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    synthesizing_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    # --- Delivery -------------------------------------------------------------------------
    idempotency_key: Mapped[str | None] = mapped_column(String(128), nullable=True)
    #: Last published event sequence number. Incremented atomically by the publisher so
    #: every event for a job carries a gap-detectable, monotonic `seq` (ADR-0003).
    last_event_seq: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    owner: Mapped[User] = relationship(back_populates="jobs")

    def __repr__(self) -> str:
        return f"Job(id={self.id!s}, status={self.status.value!r})"
