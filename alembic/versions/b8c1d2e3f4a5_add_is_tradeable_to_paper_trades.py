"""add is_tradeable to paper_trades

Revision ID: b8c1d2e3f4a5
Revises: 6f2d6044726f
Create Date: 2026-06-12 23:05:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "b8c1d2e3f4a5"
down_revision: str | Sequence[str] | None = "6f2d6044726f"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Persist the meta-labeling gate decision with each paper trade.

    Without this column the live track record cannot distinguish "every
    prediction" from "the trades the strategy would actually take" — the
    regime-dependent confidence threshold (0.40-0.55) cannot be
    reconstructed from the stored confidence alone.
    """
    op.add_column("paper_trades", sa.Column("is_tradeable", sa.Boolean(), nullable=True))


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column("paper_trades", "is_tradeable")
