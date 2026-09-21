"""Add a second dialogue voice to jobs.

Dialogue mode used one voice for every spoken line, so a scene between two characters was
read by the same person twice. The prompt now asks for two named characters and the worker
attributes each line to one of them; this is the voice the second character gets.

Nullable, and nothing is backfilled. A job created before this has one dialogue voice and
still renders exactly as it did.
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision: str = "0004"
down_revision: str | None = "0003"
branch_labels: str | None = None
depends_on: str | None = None


def upgrade() -> None:
    op.add_column("jobs", sa.Column("second_dialogue_voice_id", sa.Uuid(), nullable=True))
    op.create_foreign_key(
        "fk_jobs_second_dialogue_voice_id_voices",
        "jobs",
        "voices",
        ["second_dialogue_voice_id"],
        ["id"],
        ondelete="RESTRICT",
    )


def downgrade() -> None:
    op.drop_constraint("fk_jobs_second_dialogue_voice_id_voices", "jobs", type_="foreignkey")
    op.drop_column("jobs", "second_dialogue_voice_id")
