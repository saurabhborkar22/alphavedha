"""normalize actual_return to price-return convention

Revision ID: b8c9d0e1f2a3
Revises: a7b8c9d0e1f2
Create Date: 2026-07-16 12:00:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "b8c9d0e1f2a3"
down_revision: str | Sequence[str] | None = "a7b8c9d0e1f2"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_FLIP_STOP_PATH_ROWS = (
    "UPDATE paper_trades "
    "SET actual_return = actual_return * predicted_direction "
    "WHERE exit_reason IN ('stop_loss', 'take_profit') "
    "AND actual_return IS NOT NULL "
    "AND predicted_direction IN (-1, 1)"
)


def upgrade() -> None:
    """Convert stop/target-exited rows from trade-return to price-return.

    stop_evaluation stored actual_return direction-multiplied
    ((exit - entry) / entry * direction) while the scheduler's horizon
    path stored the raw price return — the same column carried two
    meanings. The standard is now the price return everywhere.

    Multiplying by predicted_direction recovers the price return for the
    stop-path rows (direction² == 1). Horizon rows (exit_reason IS NULL)
    already hold price returns and are untouched. Rows closed through the
    manual-close endpoint before this migration are indistinguishable
    from horizon rows and are left as-is; from this release on they are
    tagged exit_reason='manual_close' and stored as price returns.
    """
    conn = op.get_bind()
    conn.execute(sa.text(_FLIP_STOP_PATH_ROWS))


def downgrade() -> None:
    """Self-inverse: multiplying by direction again restores trade returns."""
    conn = op.get_bind()
    conn.execute(sa.text(_FLIP_STOP_PATH_ROWS))
