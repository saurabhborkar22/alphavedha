"""add shadow_fills table (P4-D3 shadow mode)

Revision ID: a7b8c9d0e1f2
Revises: f2a3b4c5d6e7
Create Date: 2026-07-02 19:40:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "a7b8c9d0e1f2"
down_revision: str | Sequence[str] | None = "f2a3b4c5d6e7"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Create shadow_fills — ghost execution log from the paper broker.

    Plain table (no hypertable): low volume (max ~8 fills/day), queried
    by date range for slippage reports.
    """
    op.create_table(
        "shadow_fills",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("strategy", sa.String(50), nullable=False),
        sa.Column("symbol", sa.String(20), nullable=False),
        sa.Column("fill_date", sa.Date, nullable=False),
        sa.Column("side", sa.String(4), nullable=False),
        sa.Column("decision_price", sa.Float, nullable=False),
        sa.Column("sim_fill_price", sa.Float, nullable=False),
        sa.Column("quantity", sa.Integer, nullable=False),
        sa.Column("slippage_bps", sa.Float, nullable=False),
        sa.Column("created_at", sa.DateTime, nullable=False, server_default=sa.text("now()")),
    )
    op.create_index("ix_shadow_fills_date", "shadow_fills", ["fill_date"])
    op.create_index("ix_shadow_fills_symbol", "shadow_fills", ["symbol"])
    op.create_index("ix_shadow_fills_strategy", "shadow_fills", ["strategy"])


def downgrade() -> None:
    op.drop_index("ix_shadow_fills_strategy", table_name="shadow_fills")
    op.drop_index("ix_shadow_fills_symbol", table_name="shadow_fills")
    op.drop_index("ix_shadow_fills_date", table_name="shadow_fills")
    op.drop_table("shadow_fills")
