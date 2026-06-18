"""add prediction_proofs table

Revision ID: d0e1f2a3b4c5
Revises: c9d2e3f4a5b6
Create Date: 2026-06-18

"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "d0e1f2a3b4c5"
down_revision: str | None = "c9d2e3f4a5b6"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "prediction_proofs",
        sa.Column("proof_date", sa.Date(), nullable=False),
        sa.Column("sha256", sa.String(64), nullable=False),
        sa.Column("n_predictions", sa.Integer(), nullable=False),
        sa.Column("payload_json", sa.String(), nullable=True),
        sa.Column("ots_path", sa.String(500), nullable=True),
        sa.Column("git_commit", sa.String(40), nullable=True),
        sa.Column("revealed_at", sa.DateTime(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False, server_default=sa.text("now()")),
        sa.PrimaryKeyConstraint("proof_date"),
    )
    op.create_index("ix_prediction_proofs_date", "prediction_proofs", ["proof_date"])


def downgrade() -> None:
    op.drop_index("ix_prediction_proofs_date", table_name="prediction_proofs")
    op.drop_table("prediction_proofs")
