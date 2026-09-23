"""Initial schema: users, voices, jobs.

Revision ID: 0001
Revises:
Create Date: 2026-09-21
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _enum(*values: str, name: str) -> sa.Enum:
    """A string-valued enum.

    ``native_enum=False`` renders a VARCHAR plus a CHECK constraint rather than a
    Postgres ENUM type. Adding a value to a native ENUM requires ``ALTER TYPE``, which
    cannot run inside a transaction block on older Postgres and makes every future enum
    change a migration hazard. A CHECK constraint is rewritten freely.
    """
    return sa.Enum(*values, name=name, native_enum=False, length=32)


def upgrade() -> None:
    op.create_table(
        "users",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column("email", sa.String(320), nullable=True, unique=True),
        sa.Column("last_seen_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
    )

    op.create_table(
        "voices",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column(
            "owner_id",
            sa.Uuid(),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=True,
        ),
        sa.Column("name", sa.String(60), nullable=False),
        sa.Column("storage_key", sa.String(512), nullable=False),
        sa.Column("duration_seconds", sa.Float(), nullable=False),
        sa.Column("sample_rate", sa.Integer(), nullable=False),
        sa.Column("is_builtin", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.UniqueConstraint("owner_id", "name", name="uq_voices_owner_name"),
        sa.CheckConstraint("duration_seconds > 0", name="ck_voices_duration_positive"),
        # A built-in voice has no owner; an uploaded one must have exactly one. Without
        # this, an upload with a null owner would be visible to every user.
        sa.CheckConstraint(
            "(is_builtin AND owner_id IS NULL) OR (NOT is_builtin AND owner_id IS NOT NULL)",
            name="ck_voices_ownership",
        ),
    )
    op.create_index("ix_voices_owner_id", "voices", ["owner_id"])

    op.create_table(
        "jobs",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column(
            "owner_id",
            sa.Uuid(),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "status",
            _enum(
                "queued",
                "writing",
                "written",
                "synthesizing",
                "done",
                "failed",
                "cancelled",
                name="job_status",
            ),
            nullable=False,
        ),
        sa.Column("prompt", sa.Text(), nullable=False),
        sa.Column("length", _enum("short", "medium", "long", name="story_length"), nullable=False),
        sa.Column(
            "mode",
            _enum("narration", "narration_with_dialogue", name="voice_mode"),
            nullable=False,
        ),
        sa.Column(
            "language",
            _enum("en", "es", "fr", "de", "it", "ru", "hi", name="language"),
            nullable=False,
        ),
        sa.Column(
            "emotion",
            _enum("neutral", "happy", "sad", "angry", name="emotion"),
            nullable=False,
        ),
        sa.Column("speed", sa.Float(), nullable=False, server_default="1.0"),
        sa.Column(
            "voice_id",
            sa.Uuid(),
            sa.ForeignKey("voices.id", ondelete="RESTRICT"),
            nullable=False,
        ),
        sa.Column(
            "dialogue_voice_id",
            sa.Uuid(),
            sa.ForeignKey("voices.id", ondelete="RESTRICT"),
            nullable=True,
        ),
        sa.Column("story_text", sa.Text(), nullable=True),
        sa.Column("llm_model", sa.String(120), nullable=True),
        sa.Column("llm_output_tokens", sa.Integer(), nullable=True),
        sa.Column("audio_key_mp3", sa.String(512), nullable=True),
        sa.Column("audio_key_wav", sa.String(512), nullable=True),
        sa.Column("audio_duration_seconds", sa.Float(), nullable=True),
        sa.Column("segment_count", sa.Integer(), nullable=True),
        sa.Column("segments_done", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("error_code", sa.String(64), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("retry_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column(
            "queued_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.Column("writing_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("written_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("synthesizing_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("idempotency_key", sa.String(128), nullable=True),
        sa.Column("last_event_seq", sa.Integer(), nullable=False, server_default="0"),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.CheckConstraint("speed >= 0.5 AND speed <= 1.5", name="ck_jobs_speed_range"),
        sa.CheckConstraint("retry_count >= 0", name="ck_jobs_retry_count_non_negative"),
    )

    # Cursor pagination: ids are UUIDv7 and therefore time-ordered, so this one index is
    # both the filter and the sort (ADR-0002).
    op.create_index("ix_jobs_owner_id_id", "jobs", ["owner_id", "id"])
    # Queue-depth metrics and the sweeper that re-dispatches stranded `queued` rows.
    op.create_index("ix_jobs_status", "jobs", ["status"])
    # Partial: most jobs carry no idempotency key, and indexing those rows would be pure
    # overhead.
    op.create_index(
        "uq_jobs_owner_idempotency_key",
        "jobs",
        ["owner_id", "idempotency_key"],
        unique=True,
        postgresql_where=sa.text("idempotency_key IS NOT NULL"),
    )


def downgrade() -> None:
    op.drop_index("uq_jobs_owner_idempotency_key", table_name="jobs")
    op.drop_index("ix_jobs_status", table_name="jobs")
    op.drop_index("ix_jobs_owner_id_id", table_name="jobs")
    op.drop_table("jobs")
    op.drop_index("ix_voices_owner_id", table_name="voices")
    op.drop_table("voices")
    op.drop_table("users")
