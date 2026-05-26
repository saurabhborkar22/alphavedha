"""Convert time-series tables to TimescaleDB hypertables.

Drops serial id PKs, sets natural composite PKs, converts to hypertables
with monthly chunks. Migrates existing data in-place.

Revision ID: a1b2c3d4e5f6
Revises: 05c23a1b9653
Create Date: 2026-05-23 10:00:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "a1b2c3d4e5f6"
down_revision: str | Sequence[str] | None = "05c23a1b9653"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

HYPERTABLE_CONFIGS: list[dict[str, str | list[str] | None]] = [
    {
        "table": "daily_ohlcv",
        "time_col": "date",
        "old_pk": "daily_ohlcv_pkey",
        "old_unique": "uq_daily_ohlcv_symbol_date",
        "old_indexes": ["ix_daily_ohlcv_symbol_date"],
        "new_pk_cols": "symbol, date",
        "old_seq": "daily_ohlcv_id_seq",
        "chunk_interval": "1 month",
    },
    {
        "table": "features",
        "time_col": "date",
        "old_pk": "features_pkey",
        "old_unique": "uq_feature",
        "old_indexes": ["ix_features_symbol_date"],
        "new_pk_cols": "symbol, date, feature_version",
        "old_seq": "features_id_seq",
        "chunk_interval": "1 month",
    },
    {
        "table": "derivatives_data",
        "time_col": "date",
        "old_pk": "derivatives_data_pkey",
        "old_unique": "uq_derivatives_data",
        "old_indexes": ["ix_derivatives_data_symbol_date"],
        "new_pk_cols": "symbol, date",
        "old_seq": "derivatives_data_id_seq",
        "chunk_interval": "1 month",
    },
    {
        "table": "institutional_flows",
        "time_col": "date",
        "old_pk": "institutional_flows_pkey",
        "old_unique": "uq_institutional_flow",
        "old_indexes": ["ix_institutional_flows_date"],
        "new_pk_cols": "date, category",
        "old_seq": "institutional_flows_id_seq",
        "chunk_interval": "1 month",
    },
    {
        "table": "daily_pnl",
        "time_col": "date",
        "old_pk": "daily_pnl_pkey",
        "old_unique": "daily_pnl_date_key",
        "old_indexes": ["ix_daily_pnl_date"],
        "new_pk_cols": "date",
        "old_seq": "daily_pnl_id_seq",
        "chunk_interval": "1 month",
    },
    {
        "table": "paper_trades",
        "time_col": "prediction_date",
        "old_pk": "paper_trades_pkey",
        "old_unique": "uq_paper_trade",
        "old_indexes": [],
        "new_pk_cols": "symbol, prediction_date",
        "old_seq": "paper_trades_id_seq",
        "chunk_interval": "1 month",
    },
    {
        "table": "news_articles",
        "time_col": "published_date",
        "old_pk": "news_articles_pkey",
        "old_unique": "uq_news_article_hash",
        "old_indexes": [],
        "new_pk_cols": "content_hash, published_date",
        "old_seq": "news_articles_id_seq",
        "chunk_interval": "1 month",
    },
    {
        "table": "insider_trades",
        "time_col": "trade_date",
        "old_pk": "insider_trades_pkey",
        "old_unique": None,
        "old_indexes": [],
        "new_pk_cols": "symbol, trade_date, person_name",
        "old_seq": "insider_trades_id_seq",
        "chunk_interval": "1 month",
    },
]


def upgrade() -> None:
    conn = op.get_bind()

    conn.execute(sa.text("CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE"))

    for cfg in HYPERTABLE_CONFIGS:
        table = cfg["table"]
        time_col = cfg["time_col"]

        conn.execute(
            sa.text(
                f"ALTER TABLE {table} DROP CONSTRAINT {cfg['old_pk']} CASCADE"
            )
        )

        if cfg["old_unique"]:
            conn.execute(
                sa.text(
                    f"ALTER TABLE {table} DROP CONSTRAINT IF EXISTS"
                    f" {cfg['old_unique']} CASCADE"
                )
            )

        for idx in cfg.get("old_indexes") or []:
            conn.execute(sa.text(f"DROP INDEX IF EXISTS {idx}"))

        conn.execute(
            sa.text(f"ALTER TABLE {table} DROP COLUMN IF EXISTS id")
        )

        conn.execute(
            sa.text(f"DROP SEQUENCE IF EXISTS {cfg['old_seq']}")
        )

        conn.execute(
            sa.text(
                f"ALTER TABLE {table}"
                f" ADD PRIMARY KEY ({cfg['new_pk_cols']})"
            )
        )

        conn.execute(
            sa.text(
                f"SELECT create_hypertable('{table}', '{time_col}',"
                f" chunk_time_interval => INTERVAL '{cfg['chunk_interval']}',"
                f" migrate_data => TRUE)"
            )
        )

    conn.execute(
        sa.text(
            "CREATE INDEX IF NOT EXISTS ix_daily_ohlcv_date"
            " ON daily_ohlcv (date DESC)"
        )
    )
    conn.execute(
        sa.text(
            "CREATE INDEX IF NOT EXISTS ix_paper_trades_date"
            " ON paper_trades (prediction_date DESC)"
        )
    )
    conn.execute(
        sa.text(
            "CREATE INDEX IF NOT EXISTS ix_paper_trades_symbol"
            " ON paper_trades (symbol)"
        )
    )
    conn.execute(
        sa.text(
            "CREATE INDEX IF NOT EXISTS ix_news_articles_date"
            " ON news_articles (published_date DESC)"
        )
    )
    conn.execute(
        sa.text(
            "CREATE INDEX IF NOT EXISTS ix_news_articles_symbol_date"
            " ON news_articles (symbol, published_date DESC)"
        )
    )
    conn.execute(
        sa.text(
            "CREATE INDEX IF NOT EXISTS ix_insider_trades_symbol"
            " ON insider_trades (symbol)"
        )
    )
    conn.execute(
        sa.text(
            "CREATE INDEX IF NOT EXISTS ix_institutional_flows_date"
            " ON institutional_flows (date DESC)"
        )
    )

    conn.execute(
        sa.text(
            "ALTER TABLE daily_ohlcv SET ("
            " timescaledb.compress,"
            " timescaledb.compress_segmentby = 'symbol',"
            " timescaledb.compress_orderby = 'date DESC'"
            ")"
        )
    )
    conn.execute(
        sa.text(
            "SELECT add_compression_policy('daily_ohlcv', INTERVAL '6 months')"
        )
    )

    conn.execute(
        sa.text(
            "ALTER TABLE features SET ("
            " timescaledb.compress,"
            " timescaledb.compress_segmentby = 'symbol',"
            " timescaledb.compress_orderby = 'date DESC'"
            ")"
        )
    )
    conn.execute(
        sa.text(
            "SELECT add_compression_policy('features', INTERVAL '3 months')"
        )
    )


def downgrade() -> None:
    """Downgrade is destructive — converts hypertables back to regular tables.

    TimescaleDB does not support direct conversion back. We recreate
    tables from the hypertable data. This WILL lose chunk metadata
    and compression policies.
    """
    conn = op.get_bind()

    conn.execute(
        sa.text(
            "SELECT remove_compression_policy('daily_ohlcv', if_exists => TRUE)"
        )
    )
    conn.execute(
        sa.text(
            "SELECT remove_compression_policy('features', if_exists => TRUE)"
        )
    )

    for cfg in reversed(HYPERTABLE_CONFIGS):
        table = cfg["table"]
        new_pk_cols = cfg["new_pk_cols"]

        conn.execute(
            sa.text(
                f"CREATE TABLE {table}_backup AS SELECT * FROM {table}"
            )
        )
        conn.execute(sa.text(f"DROP TABLE {table}"))
        conn.execute(
            sa.text(f"ALTER TABLE {table}_backup RENAME TO {table}")
        )

        conn.execute(
            sa.text(
                f"ALTER TABLE {table} ADD COLUMN id SERIAL"
            )
        )
        conn.execute(
            sa.text(
                f"ALTER TABLE {table} ADD PRIMARY KEY (id)"
            )
        )

        if cfg["old_unique"]:
            conn.execute(
                sa.text(
                    f"ALTER TABLE {table} ADD CONSTRAINT {cfg['old_unique']}"
                    f" UNIQUE ({new_pk_cols})"
                )
            )
