"""Store / load functions for intel tables (disclosures, events, etc.)."""

from __future__ import annotations

from datetime import datetime
from typing import Any

import pandas as pd
import structlog
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert

from alphavedha.data.database import get_session_factory
from alphavedha.data.models import (
    BulkBlockDeal,
    Disclosure,
    DisclosureEvent,
    PledgeSnapshot,
    RatingEvent,
    SurveillanceFlag,
    Transcript,
)

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Disclosures
# ---------------------------------------------------------------------------


async def store_disclosures(rows: list[dict[str, Any]]) -> int:
    """Upsert disclosures. Deduplicates on (symbol, source, filed_at, headline)."""
    if not rows:
        return 0

    session_factory = get_session_factory()
    stored = 0

    async with session_factory() as session:
        for batch_start in range(0, len(rows), 100):
            batch = rows[batch_start : batch_start + 100]
            values = [
                {
                    "symbol": r["symbol"],
                    "source": r["source"],
                    "category": r["category"],
                    "headline": r["headline"],
                    "filed_at": r["filed_at"],
                    "url": r.get("url"),
                    "text": r.get("text"),
                    "text_hash": r.get("text_hash"),
                    "processed_at": r.get("processed_at"),
                }
                for r in batch
            ]
            stmt = pg_insert(Disclosure).values(values)
            stmt = stmt.on_conflict_do_update(
                constraint="uq_disclosure",
                set_={
                    "text": stmt.excluded.text,
                    "text_hash": stmt.excluded.text_hash,
                    "processed_at": stmt.excluded.processed_at,
                    "url": stmt.excluded.url,
                },
            )
            await session.execute(stmt)
            stored += len(batch)

        await session.commit()

    logger.info("disclosures_stored", rows=stored)
    return stored


async def load_disclosures(
    symbol: str | None = None,
    since: datetime | None = None,
    until: datetime | None = None,
    category: str | None = None,
    unprocessed_only: bool = False,
    limit: int = 500,
) -> pd.DataFrame:
    """Load disclosures with optional filters."""
    session_factory = get_session_factory()

    async with session_factory() as session:
        stmt = select(Disclosure).order_by(Disclosure.filed_at.desc()).limit(limit)

        if symbol is not None:
            stmt = stmt.where(Disclosure.symbol == symbol)
        if since is not None:
            stmt = stmt.where(Disclosure.filed_at >= since)
        if until is not None:
            stmt = stmt.where(Disclosure.filed_at <= until)
        if category is not None:
            stmt = stmt.where(Disclosure.category == category)
        if unprocessed_only:
            stmt = stmt.where(Disclosure.processed_at.is_(None))

        result = await session.execute(stmt)
        rows = result.scalars().all()

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(
        [
            {
                "id": r.id,
                "symbol": r.symbol,
                "source": r.source,
                "category": r.category,
                "headline": r.headline,
                "filed_at": r.filed_at,
                "url": r.url,
                "text": r.text,
                "text_hash": r.text_hash,
                "processed_at": r.processed_at,
            }
            for r in rows
        ]
    )


# ---------------------------------------------------------------------------
# Disclosure Events
# ---------------------------------------------------------------------------


async def store_disclosure_events(rows: list[dict[str, Any]]) -> int:
    """Insert extracted disclosure events."""
    if not rows:
        return 0

    session_factory = get_session_factory()
    stored = 0

    async with session_factory() as session:
        for r in rows:
            event = DisclosureEvent(
                disclosure_id=r["disclosure_id"],
                symbol=r["symbol"],
                event_type=r["event_type"],
                direction=r["direction"],
                materiality=r["materiality"],
                confidence=r["confidence"],
                summary=r["summary"],
                red_flags=r.get("red_flags"),
                llm_model=r["llm_model"],
                prompt_version=r["prompt_version"],
                extracted_at=r["extracted_at"],
            )
            session.add(event)
            stored += 1

        await session.commit()

    logger.info("disclosure_events_stored", rows=stored)
    return stored


async def load_disclosure_events(
    symbol: str | None = None,
    event_type: str | None = None,
    since: datetime | None = None,
    limit: int = 500,
) -> pd.DataFrame:
    """Load structured disclosure events."""
    session_factory = get_session_factory()

    async with session_factory() as session:
        stmt = select(DisclosureEvent).order_by(DisclosureEvent.extracted_at.desc()).limit(limit)

        if symbol is not None:
            stmt = stmt.where(DisclosureEvent.symbol == symbol)
        if event_type is not None:
            stmt = stmt.where(DisclosureEvent.event_type == event_type)
        if since is not None:
            stmt = stmt.where(DisclosureEvent.extracted_at >= since)

        result = await session.execute(stmt)
        rows = result.scalars().all()

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(
        [
            {
                "id": r.id,
                "disclosure_id": r.disclosure_id,
                "symbol": r.symbol,
                "event_type": r.event_type,
                "direction": r.direction,
                "materiality": r.materiality,
                "confidence": r.confidence,
                "summary": r.summary,
                "red_flags": r.red_flags,
                "llm_model": r.llm_model,
                "prompt_version": r.prompt_version,
                "extracted_at": r.extracted_at,
            }
            for r in rows
        ]
    )


# ---------------------------------------------------------------------------
# Rating Events
# ---------------------------------------------------------------------------


async def store_rating_events(rows: list[dict[str, Any]]) -> int:
    """Upsert rating events. Deduplicates on (symbol, agency, filed_at)."""
    if not rows:
        return 0

    session_factory = get_session_factory()
    stored = 0

    async with session_factory() as session:
        for batch_start in range(0, len(rows), 100):
            batch = rows[batch_start : batch_start + 100]
            values = [
                {
                    "symbol": r["symbol"],
                    "agency": r["agency"],
                    "action": r["action"],
                    "rating_from": r.get("rating_from"),
                    "rating_to": r.get("rating_to"),
                    "outlook": r.get("outlook"),
                    "rationale_text": r.get("rationale_text"),
                    "filed_at": r["filed_at"],
                }
                for r in batch
            ]
            stmt = pg_insert(RatingEvent).values(values)
            stmt = stmt.on_conflict_do_update(
                constraint="uq_rating_event",
                set_={
                    "action": stmt.excluded.action,
                    "rating_from": stmt.excluded.rating_from,
                    "rating_to": stmt.excluded.rating_to,
                    "outlook": stmt.excluded.outlook,
                    "rationale_text": stmt.excluded.rationale_text,
                },
            )
            await session.execute(stmt)
            stored += len(batch)

        await session.commit()

    logger.info("rating_events_stored", rows=stored)
    return stored


# ---------------------------------------------------------------------------
# Pledge Snapshots
# ---------------------------------------------------------------------------


async def store_pledge_snapshots(rows: list[dict[str, Any]]) -> int:
    """Upsert pledge snapshots. Deduplicates on (symbol, as_of)."""
    if not rows:
        return 0

    session_factory = get_session_factory()
    stored = 0

    async with session_factory() as session:
        for batch_start in range(0, len(rows), 100):
            batch = rows[batch_start : batch_start + 100]
            values = [
                {
                    "symbol": r["symbol"],
                    "as_of": r["as_of"],
                    "promoter_pledge_pct": r["promoter_pledge_pct"],
                    "change_pct": r.get("change_pct"),
                }
                for r in batch
            ]
            stmt = pg_insert(PledgeSnapshot).values(values)
            stmt = stmt.on_conflict_do_update(
                constraint="uq_pledge_snapshot",
                set_={
                    "promoter_pledge_pct": stmt.excluded.promoter_pledge_pct,
                    "change_pct": stmt.excluded.change_pct,
                },
            )
            await session.execute(stmt)
            stored += len(batch)

        await session.commit()

    logger.info("pledge_snapshots_stored", rows=stored)
    return stored


# ---------------------------------------------------------------------------
# Surveillance Flags
# ---------------------------------------------------------------------------


async def store_surveillance_flags(rows: list[dict[str, Any]]) -> int:
    """Upsert surveillance flags. Deduplicates on (symbol, list_name, added_on)."""
    if not rows:
        return 0

    session_factory = get_session_factory()
    stored = 0

    async with session_factory() as session:
        for batch_start in range(0, len(rows), 100):
            batch = rows[batch_start : batch_start + 100]
            values = [
                {
                    "symbol": r["symbol"],
                    "list_name": r["list_name"],
                    "added_on": r["added_on"],
                    "removed_on": r.get("removed_on"),
                }
                for r in batch
            ]
            stmt = pg_insert(SurveillanceFlag).values(values)
            stmt = stmt.on_conflict_do_update(
                constraint="uq_surveillance_flag",
                set_={"removed_on": stmt.excluded.removed_on},
            )
            await session.execute(stmt)
            stored += len(batch)

        await session.commit()

    logger.info("surveillance_flags_stored", rows=stored)
    return stored


# ---------------------------------------------------------------------------
# Bulk/Block Deals
# ---------------------------------------------------------------------------


async def store_bulk_block_deals(rows: list[dict[str, Any]]) -> int:
    """Insert bulk/block deals (no natural unique key — append only)."""
    if not rows:
        return 0

    session_factory = get_session_factory()
    stored = 0

    async with session_factory() as session:
        for r in rows:
            deal = BulkBlockDeal(
                symbol=r["symbol"],
                deal_date=r["deal_date"],
                deal_type=r["deal_type"],
                client_name=r["client_name"],
                trade_type=r["trade_type"],
                quantity=r["quantity"],
                price=r["price"],
            )
            session.add(deal)
            stored += 1

        await session.commit()

    logger.info("bulk_block_deals_stored", rows=stored)
    return stored


# ---------------------------------------------------------------------------
# Transcripts
# ---------------------------------------------------------------------------


async def store_transcripts(rows: list[dict[str, Any]]) -> int:
    """Upsert transcripts. Deduplicates on (symbol, fiscal_quarter)."""
    if not rows:
        return 0

    session_factory = get_session_factory()
    stored = 0

    async with session_factory() as session:
        for batch_start in range(0, len(rows), 100):
            batch = rows[batch_start : batch_start + 100]
            values = [
                {
                    "symbol": r["symbol"],
                    "fiscal_quarter": r["fiscal_quarter"],
                    "filed_at": r["filed_at"],
                    "text": r.get("text"),
                    "sections": r.get("sections"),
                }
                for r in batch
            ]
            stmt = pg_insert(Transcript).values(values)
            stmt = stmt.on_conflict_do_update(
                constraint="uq_transcript",
                set_={
                    "filed_at": stmt.excluded.filed_at,
                    "text": stmt.excluded.text,
                    "sections": stmt.excluded.sections,
                },
            )
            await session.execute(stmt)
            stored += len(batch)

        await session.commit()

    logger.info("transcripts_stored", rows=stored)
    return stored


async def load_transcripts(
    symbol: str | None = None,
    limit: int = 100,
) -> pd.DataFrame:
    """Load transcripts, ordered by fiscal_quarter descending."""
    session_factory = get_session_factory()

    async with session_factory() as session:
        stmt = select(Transcript).order_by(Transcript.fiscal_quarter.desc()).limit(limit)

        if symbol is not None:
            stmt = stmt.where(Transcript.symbol == symbol)

        result = await session.execute(stmt)
        rows = result.scalars().all()

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(
        [
            {
                "id": r.id,
                "symbol": r.symbol,
                "fiscal_quarter": r.fiscal_quarter,
                "filed_at": r.filed_at,
                "text": r.text,
                "sections": r.sections,
            }
            for r in rows
        ]
    )


async def load_transcript_pairs(symbol: str) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    """Load consecutive transcript pairs for a symbol (newer, older)."""
    df = await load_transcripts(symbol=symbol)
    if len(df) < 2:
        return []

    records = df.to_dict("records")
    pairs: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for i in range(len(records) - 1):
        pairs.append((records[i], records[i + 1]))
    return pairs


async def load_disclosures_by_ids(ids: list[int]) -> list[Disclosure]:
    """Load disclosure rows by their primary key IDs."""
    if not ids:
        return []

    session_factory = get_session_factory()

    async with session_factory() as session:
        stmt = select(Disclosure).where(Disclosure.id.in_(ids))
        result = await session.execute(stmt)
        return list(result.scalars().all())


async def mark_disclosures_processed(ids: list[int], processed_at: datetime) -> int:
    """Mark disclosures as processed after LLM extraction."""
    if not ids:
        return 0

    from sqlalchemy import update

    session_factory = get_session_factory()

    async with session_factory() as session:
        stmt = update(Disclosure).where(Disclosure.id.in_(ids)).values(processed_at=processed_at)
        result = await session.execute(stmt)
        await session.commit()
        count: int = getattr(result, "rowcount", 0) or 0
        return count
