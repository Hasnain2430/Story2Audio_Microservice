"""Record encoded audio sizes.

The API has always returned a `size_bytes` per audio asset; until now it returned 0,
because the TTS worker computed the size at upload and then discarded it. Persisting it
is cheaper than a HEAD request per asset on every job read.

Revision ID: 0002
Revises: 0001
Create Date: 2026-09-21
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0002"
down_revision: str | None = "0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("jobs", sa.Column("audio_bytes_mp3", sa.Integer(), nullable=True))
    op.add_column("jobs", sa.Column("audio_bytes_wav", sa.Integer(), nullable=True))


def downgrade() -> None:
    op.drop_column("jobs", "audio_bytes_wav")
    op.drop_column("jobs", "audio_bytes_mp3")
