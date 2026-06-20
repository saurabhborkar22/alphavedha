"""add strategy column to paper_trades

Revision ID: f2a3b4c5d6e7
Revises: e1f2a3b4c5d6
Create Date: 2026-06-19 22:00:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "f2a3b4c5d6e7"
down_revision: str | Sequence[str] | None = "e1f2a3b4c5d6"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Add strategy column to paper_trades with default 'ensemble_v1'.

    TimescaleDB hypertables don't support ALTER ... ADD PRIMARY KEY
    directly, so we:
    1. Add the column with a default
    2. Recreate the table with the new PK via backup + drop + rename
    3. Re-create the hypertable
    """
    conn = op.get_bind()

    conn.execute(
        sa.text(
            "ALTER TABLE paper_trades "
            "ADD COLUMN IF NOT EXISTS strategy VARCHAR(50) "
            "NOT NULL DEFAULT 'ensemble_v1'"
        )
    )

    conn.execute(sa.text("CREATE TABLE paper_trades_backup AS SELECT * FROM paper_trades"))

    conn.execute(sa.text("DROP TABLE paper_trades"))

    conn.execute(
        sa.text(
            "CREATE TABLE paper_trades ("
            "  symbol VARCHAR(20) NOT NULL,"
            "  prediction_date DATE NOT NULL,"
            "  strategy VARCHAR(50) NOT NULL DEFAULT 'ensemble_v1',"
            "  predicted_direction INTEGER NOT NULL,"
            "  predicted_magnitude DOUBLE PRECISION NOT NULL,"
            "  confidence DOUBLE PRECISION NOT NULL,"
            "  model_version VARCHAR(50) NOT NULL,"
            "  regime VARCHAR(20),"
            "  is_tradeable BOOLEAN,"
            "  entry_price DOUBLE PRECISION,"
            "  stop_loss_price DOUBLE PRECISION,"
            "  take_profit_price DOUBLE PRECISION,"
            "  exit_price DOUBLE PRECISION,"
            "  exit_reason VARCHAR(20),"
            "  actual_return DOUBLE PRECISION,"
            "  is_correct BOOLEAN,"
            "  created_at TIMESTAMP NOT NULL DEFAULT now(),"
            "  PRIMARY KEY (symbol, prediction_date, strategy)"
            ")"
        )
    )

    conn.execute(
        sa.text(
            "INSERT INTO paper_trades "
            "SELECT symbol, prediction_date, strategy, predicted_direction, "
            "predicted_magnitude, confidence, model_version, regime, "
            "is_tradeable, entry_price, stop_loss_price, take_profit_price, "
            "exit_price, exit_reason, actual_return, is_correct, created_at "
            "FROM paper_trades_backup"
        )
    )

    conn.execute(sa.text("DROP TABLE paper_trades_backup"))

    conn.execute(
        sa.text(
            "SELECT create_hypertable('paper_trades', 'prediction_date', "
            "chunk_time_interval => INTERVAL '1 month', "
            "migrate_data => TRUE)"
        )
    )

    conn.execute(
        sa.text(
            "CREATE INDEX IF NOT EXISTS ix_paper_trades_date ON paper_trades (prediction_date DESC)"
        )
    )
    conn.execute(
        sa.text("CREATE INDEX IF NOT EXISTS ix_paper_trades_symbol ON paper_trades (symbol)")
    )
    conn.execute(
        sa.text("CREATE INDEX IF NOT EXISTS ix_paper_trades_strategy ON paper_trades (strategy)")
    )


def downgrade() -> None:
    """Remove strategy column — drop and recreate with original PK."""
    conn = op.get_bind()

    conn.execute(sa.text("CREATE TABLE paper_trades_backup AS SELECT * FROM paper_trades"))

    conn.execute(sa.text("DROP TABLE paper_trades"))

    conn.execute(
        sa.text(
            "CREATE TABLE paper_trades ("
            "  symbol VARCHAR(20) NOT NULL,"
            "  prediction_date DATE NOT NULL,"
            "  predicted_direction INTEGER NOT NULL,"
            "  predicted_magnitude DOUBLE PRECISION NOT NULL,"
            "  confidence DOUBLE PRECISION NOT NULL,"
            "  model_version VARCHAR(50) NOT NULL,"
            "  regime VARCHAR(20),"
            "  is_tradeable BOOLEAN,"
            "  entry_price DOUBLE PRECISION,"
            "  stop_loss_price DOUBLE PRECISION,"
            "  take_profit_price DOUBLE PRECISION,"
            "  exit_price DOUBLE PRECISION,"
            "  exit_reason VARCHAR(20),"
            "  actual_return DOUBLE PRECISION,"
            "  is_correct BOOLEAN,"
            "  created_at TIMESTAMP NOT NULL DEFAULT now(),"
            "  PRIMARY KEY (symbol, prediction_date)"
            ")"
        )
    )

    conn.execute(
        sa.text(
            "INSERT INTO paper_trades "
            "SELECT symbol, prediction_date, predicted_direction, "
            "predicted_magnitude, confidence, model_version, regime, "
            "is_tradeable, entry_price, stop_loss_price, take_profit_price, "
            "exit_price, exit_reason, actual_return, is_correct, created_at "
            "FROM paper_trades_backup "
            "WHERE strategy = 'ensemble_v1'"
        )
    )

    conn.execute(sa.text("DROP TABLE paper_trades_backup"))

    conn.execute(
        sa.text(
            "SELECT create_hypertable('paper_trades', 'prediction_date', "
            "chunk_time_interval => INTERVAL '1 month', "
            "migrate_data => TRUE)"
        )
    )
