"""Add the per-segment timeline to jobs.

The TTS worker knows exactly where each segment lands in the assembled track, because it
rendered every segment and inserted every pause itself. Persisting that turns the story
into something the player can follow along with, rather than a block of text sitting next
to an audio element.

Nullable, with no backfill. A job finished before this migration has no timeline and
cannot be given one without re-synthesising it; the API reports null and the player falls
back to plain text.
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision: str = "0003"
down_revision: str | None = "0002"
branch_labels: str | None = None
depends_on: str | None = None


def upgrade() -> None:
    op.add_column("jobs", sa.Column("segment_timeline", sa.JSON(), nullable=True))


def downgrade() -> None:
    op.drop_column("jobs", "segment_timeline")
