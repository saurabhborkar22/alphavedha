"""add stop_loss_price, take_profit_price, exit_reason to paper_trades

Revision ID: c9d2e3f4a5b6
Revises: b8c1d2e3f4a5
Create Date: 2026-06-18 00:00:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "c9d2e3f4a5b6"
down_revision: str | Sequence[str] | None = "b8c1d2e3f4a5"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("paper_trades", sa.Column("stop_loss_price", sa.Float(), nullable=True))
    op.add_column("paper_trades", sa.Column("take_profit_price", sa.Float(), nullable=True))
    op.add_column("paper_trades", sa.Column("exit_reason", sa.String(20), nullable=True))


def downgrade() -> None:
    op.drop_column("paper_trades", "exit_reason")
    op.drop_column("paper_trades", "take_profit_price")
    op.drop_column("paper_trades", "stop_loss_price")
