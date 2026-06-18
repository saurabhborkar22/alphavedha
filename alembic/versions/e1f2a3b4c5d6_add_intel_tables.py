"""add intel tables for disclosure ingestion

Revision ID: e1f2a3b4c5d6
Revises: d0e1f2a3b4c5
Create Date: 2026-06-18

"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "e1f2a3b4c5d6"
down_revision: str | None = "d0e1f2a3b4c5"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # -- disclosures --
    op.create_table(
        "disclosures",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("symbol", sa.String(20), nullable=False),
        sa.Column("source", sa.String(10), nullable=False),
        sa.Column("category", sa.String(100), nullable=False),
        sa.Column("headline", sa.String(1000), nullable=False),
        sa.Column("filed_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("url", sa.String(1000), nullable=True),
        sa.Column("text", sa.Text(), nullable=True),
        sa.Column("text_hash", sa.String(64), nullable=True),
        sa.Column("processed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False, server_default=sa.text("now()")),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("symbol", "source", "filed_at", "headline", name="uq_disclosure"),
    )
    op.create_index("ix_disclosures_symbol_filed", "disclosures", ["symbol", "filed_at"])
    op.create_index("ix_disclosures_filed", "disclosures", ["filed_at"])
    op.create_index("ix_disclosures_category", "disclosures", ["category"])

    # -- disclosure_events --
    op.create_table(
        "disclosure_events",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column(
            "disclosure_id",
            sa.Integer(),
            sa.ForeignKey("disclosures.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("symbol", sa.String(20), nullable=False),
        sa.Column("event_type", sa.String(50), nullable=False),
        sa.Column("direction", sa.Integer(), nullable=False),
        sa.Column("materiality", sa.Integer(), nullable=False),
        sa.Column("confidence", sa.Float(), nullable=False),
        sa.Column("summary", sa.String(500), nullable=False),
        sa.Column("red_flags", sa.JSON(), nullable=True),
        sa.Column("llm_model", sa.String(50), nullable=False),
        sa.Column("prompt_version", sa.String(20), nullable=False),
        sa.Column("extracted_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False, server_default=sa.text("now()")),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_disclosure_events_symbol", "disclosure_events", ["symbol"])
    op.create_index("ix_disclosure_events_type", "disclosure_events", ["event_type"])
    op.create_index("ix_disclosure_events_disclosure", "disclosure_events", ["disclosure_id"])

    # -- transcripts --
    op.create_table(
        "transcripts",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("symbol", sa.String(20), nullable=False),
        sa.Column("fiscal_quarter", sa.String(10), nullable=False),
        sa.Column("filed_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("text", sa.Text(), nullable=True),
        sa.Column("sections", sa.JSON(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False, server_default=sa.text("now()")),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("symbol", "fiscal_quarter", name="uq_transcript"),
    )
    op.create_index("ix_transcripts_symbol", "transcripts", ["symbol"])
    op.create_index("ix_transcripts_filed", "transcripts", ["filed_at"])

    # -- rating_events --
    op.create_table(
        "rating_events",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("symbol", sa.String(20), nullable=False),
        sa.Column("agency", sa.String(30), nullable=False),
        sa.Column("action", sa.String(30), nullable=False),
        sa.Column("rating_from", sa.String(20), nullable=True),
        sa.Column("rating_to", sa.String(20), nullable=True),
        sa.Column("outlook", sa.String(20), nullable=True),
        sa.Column("rationale_text", sa.Text(), nullable=True),
        sa.Column("filed_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False, server_default=sa.text("now()")),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("symbol", "agency", "filed_at", name="uq_rating_event"),
    )
    op.create_index("ix_rating_events_symbol", "rating_events", ["symbol"])
    op.create_index("ix_rating_events_filed", "rating_events", ["filed_at"])

    # -- pledge_snapshots --
    op.create_table(
        "pledge_snapshots",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("symbol", sa.String(20), nullable=False),
        sa.Column("as_of", sa.Date(), nullable=False),
        sa.Column("promoter_pledge_pct", sa.Float(), nullable=False),
        sa.Column("change_pct", sa.Float(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False, server_default=sa.text("now()")),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("symbol", "as_of", name="uq_pledge_snapshot"),
    )
    op.create_index("ix_pledge_snapshots_symbol", "pledge_snapshots", ["symbol"])

    # -- surveillance_flags --
    op.create_table(
        "surveillance_flags",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("symbol", sa.String(20), nullable=False),
        sa.Column("list_name", sa.String(20), nullable=False),
        sa.Column("added_on", sa.Date(), nullable=False),
        sa.Column("removed_on", sa.Date(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False, server_default=sa.text("now()")),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("symbol", "list_name", "added_on", name="uq_surveillance_flag"),
    )
    op.create_index("ix_surveillance_flags_symbol", "surveillance_flags", ["symbol"])

    # -- bulk_block_deals --
    op.create_table(
        "bulk_block_deals",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("symbol", sa.String(20), nullable=False),
        sa.Column("deal_date", sa.Date(), nullable=False),
        sa.Column("deal_type", sa.String(10), nullable=False),
        sa.Column("client_name", sa.String(200), nullable=False),
        sa.Column("trade_type", sa.String(10), nullable=False),
        sa.Column("quantity", sa.Integer(), nullable=False),
        sa.Column("price", sa.Float(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False, server_default=sa.text("now()")),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_bulk_block_deals_symbol", "bulk_block_deals", ["symbol"])
    op.create_index("ix_bulk_block_deals_date", "bulk_block_deals", ["deal_date"])


def downgrade() -> None:
    op.drop_table("bulk_block_deals")
    op.drop_table("surveillance_flags")
    op.drop_table("pledge_snapshots")
    op.drop_table("rating_events")
    op.drop_table("transcripts")
    op.drop_table("disclosure_events")
    op.drop_table("disclosures")
