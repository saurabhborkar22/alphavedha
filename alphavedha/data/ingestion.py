"""Data ingestion — fetch, preprocess, and store OHLCV data end-to-end.

Wires providers → preprocessing → store for single stocks or full universe.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta

import pandas as pd
import structlog
from sqlalchemy.ext.asyncio import AsyncSession

from alphavedha.data.preprocessing.pipeline import run_pipeline
from alphavedha.data.providers.bse_provider import BSEProvider
from alphavedha.data.providers.trends_provider import GoogleTrendsProvider
from alphavedha.data.providers.yfinance_provider import YFinanceProvider
from alphavedha.data.store import store_derivatives, store_earnings, store_fii_dii, store_ohlcv
from alphavedha.data.universe import (
    get_symbols_for_tier,
    refresh_universe,
)

logger = structlog.get_logger(__name__)


async def _write_lineage(
    session: AsyncSession,
    *,
    symbol: str | None,
    record_date: date,
    table_name: str,
    provider: str,
    fetched_at: datetime,
    row_count: int,
) -> None:
    """Record data provenance for auditing."""
    from alphavedha.data.models import DataLineage

    row = DataLineage(
        symbol=symbol,
        date=record_date,
        table_name=table_name,
        provider=provider,
        fetched_at=fetched_at,
        row_count=row_count,
    )
    session.add(row)
    await session.commit()


@dataclass
class IngestionResult:
    """Summary of a data ingestion run."""

    symbols_requested: int = 0
    symbols_succeeded: int = 0
    symbols_failed: int = 0
    total_rows_stored: int = 0
    failed_symbols: list[str] = field(default_factory=list)
    errors: dict[str, str] = field(default_factory=dict)


async def ingest_symbol(
    symbol: str,
    start: date,
    end: date,
    provider: YFinanceProvider | None = None,
) -> int:
    """Fetch, preprocess, and store OHLCV for a single symbol. Returns rows stored."""
    if provider is None:
        provider = YFinanceProvider()

    df = await provider.fetch_ohlcv(symbol, start, end)
    if df.empty:
        logger.warning("ingest_empty", symbol=symbol)
        return 0

    result = run_pipeline(df, symbol, skip_frac_diff=True, skip_outlier=True)
    stored = await store_ohlcv(symbol, result.df)

    logger.info(
        "ingest_complete",
        symbol=symbol,
        rows_fetched=len(df),
        rows_stored=stored,
        circuit_hits=result.circuit_hits,
    )
    return stored


async def ingest_universe(
    tier: str,
    start: date,
    end: date,
    concurrency: int = 3,
) -> IngestionResult:
    """Ingest OHLCV for all symbols in a universe tier."""
    symbols = await get_symbols_for_tier(tier)
    if not symbols:
        logger.warning("ingest_no_symbols", tier=tier)
        return IngestionResult()

    provider = YFinanceProvider()
    sem = asyncio.Semaphore(concurrency)
    result = IngestionResult(symbols_requested=len(symbols))

    async def _ingest_one(sym: str) -> None:
        async with sem:
            try:
                rows = await ingest_symbol(sym, start, end, provider)
                if rows > 0:
                    result.symbols_succeeded += 1
                    result.total_rows_stored += rows
                else:
                    result.symbols_failed += 1
                    result.failed_symbols.append(sym)
                    result.errors[sym] = "empty data"
            except Exception as e:
                result.symbols_failed += 1
                result.failed_symbols.append(sym)
                result.errors[sym] = str(e)
                logger.error("ingest_symbol_error", symbol=sym, error=str(e))

    await asyncio.gather(*[_ingest_one(s) for s in symbols])

    logger.info(
        "ingest_universe_complete",
        tier=tier,
        succeeded=result.symbols_succeeded,
        failed=result.symbols_failed,
        total_rows=result.total_rows_stored,
    )
    return result


async def backfill(
    tier: str = "large",
    start: str = "2020-01-01",
) -> IngestionResult:
    """Full backfill: refresh universe, then ingest all symbols."""
    await refresh_universe()

    start_date = date.fromisoformat(start)
    end_date = date.today()

    logger.info(
        "backfill_starting",
        tier=tier,
        start=str(start_date),
        end=str(end_date),
    )
    return await ingest_universe(tier, start_date, end_date)


async def refresh_latest(
    tier: str = "large",
    lookback_days: int = 5,
) -> IngestionResult:
    """Refresh recent data for all symbols in a tier."""
    start_date = date.today() - timedelta(days=lookback_days)
    end_date = date.today() + timedelta(days=1)  # yfinance end is exclusive

    return await ingest_universe(tier, start_date, end_date)


@dataclass
class FIIDIIResult:
    """Summary of FII/DII ingestion."""

    rows_fetched: int = 0
    rows_stored: int = 0
    categories: list[str] = field(default_factory=list)
    error: str | None = None


async def ingest_fii_dii() -> FIIDIIResult:
    """Fetch today's FII/DII data from NSE and store it."""
    from alphavedha.data.providers.nse_provider import NSEProvider, parse_fii_dii_response

    provider = NSEProvider()
    result = FIIDIIResult()

    try:
        raw = await provider.fetch_fii_dii_today()
        result.rows_fetched = len(raw)

        if not raw:
            logger.warning("fii_dii_empty_response")
            return result

        parsed = parse_fii_dii_response(raw)
        result.rows_stored = await store_fii_dii(parsed)
        result.categories = list({r["category"] for r in parsed})

        logger.info(
            "fii_dii_ingested",
            fetched=result.rows_fetched,
            stored=result.rows_stored,
            categories=result.categories,
        )
    except Exception as e:
        result.error = str(e)
        logger.error("fii_dii_ingestion_failed", error=str(e))

    return result


@dataclass
class DerivativesResult:
    """Summary of derivatives ingestion."""

    symbols_requested: int = 0
    symbols_succeeded: int = 0
    rows_stored: int = 0
    errors: dict[str, str] = field(default_factory=dict)


async def ingest_derivatives(
    symbols: list[str] | None = None,
    tier: str = "large",
) -> DerivativesResult:
    """Fetch F&O data for symbols and store it."""
    from alphavedha.data.providers.nse_provider import NSEProvider, parse_fno_to_derivatives

    if symbols is None:
        symbols = await get_symbols_for_tier(tier)

    provider = NSEProvider()
    result = DerivativesResult(symbols_requested=len(symbols))
    today = date.today()

    for symbol in symbols:
        try:
            fno_data = await provider.fetch_stock_fno_quote(symbol)
            parsed = parse_fno_to_derivatives(fno_data, symbol, today)
            stored = await store_derivatives([parsed])
            result.symbols_succeeded += 1
            result.rows_stored += stored
        except Exception as e:
            result.errors[symbol] = str(e)
            logger.warning("derivatives_ingest_error", symbol=symbol, error=str(e))

    logger.info(
        "derivatives_ingested",
        succeeded=result.symbols_succeeded,
        requested=result.symbols_requested,
        rows=result.rows_stored,
    )
    return result


@dataclass
class EarningsIngestionResult:
    """Summary of earnings ingestion."""

    symbols_requested: int = 0
    symbols_succeeded: int = 0
    total_quarters: int = 0
    errors: dict[str, str] = field(default_factory=dict)


async def ingest_earnings(
    symbols: list[str] | None = None,
    tier: str = "large",
) -> EarningsIngestionResult:
    """Fetch quarterly earnings for symbols and store them."""
    from alphavedha.data.providers.earnings_provider import EarningsProvider

    if symbols is None:
        symbols = await get_symbols_for_tier(tier)

    provider = EarningsProvider()
    result = EarningsIngestionResult(symbols_requested=len(symbols))

    for symbol in symbols:
        try:
            nse_sym = symbol.replace(".NS", "").replace(".BO", "")
            earnings = await provider.fetch_quarterly_results(nse_sym)
            if earnings:
                stored = await store_earnings(earnings)
                result.symbols_succeeded += 1
                result.total_quarters += stored
        except Exception as e:
            result.errors[symbol] = str(e)
            logger.warning("earnings_ingest_error", symbol=symbol, error=str(e))

    logger.info(
        "earnings_ingested",
        succeeded=result.symbols_succeeded,
        requested=result.symbols_requested,
        quarters=result.total_quarters,
    )
    return result


async def ingest_bse_announcements(
    symbols: list[str],
    start: date,
    end: date,
    session: AsyncSession,
) -> int:
    """Fetch BSE corporate announcements for symbols and upsert into DB.

    Returns total number of announcement records processed.
    """
    from sqlalchemy.dialects.postgresql import insert as pg_insert

    from alphavedha.data.models import CorporateAnnouncement

    provider = BSEProvider()
    bulk = await provider.fetch_bulk(symbols, start, end)

    total = 0
    for sym_records in bulk.values():
        for rec in sym_records:
            stmt = (
                pg_insert(CorporateAnnouncement)
                .values(
                    symbol=rec.symbol,
                    announced_date=rec.announced_date,
                    ex_date=rec.ex_date,
                    event_type=rec.event_type,
                    description=rec.description,
                )
                .on_conflict_do_nothing(constraint="uq_corp_announcement")
            )
            await session.execute(stmt)
            total += 1
    await session.commit()
    logger.info(
        "ingest_bse_announcements.complete",
        total=total,
        symbols=len(symbols),
        start=str(start),
        end=str(end),
    )
    return total


async def ingest_trends() -> dict[str, pd.DataFrame]:
    """Fetch Google Trends for all 5 market sectors.

    Returns dict mapping sector name to DataFrame of interest-over-time data.
    This data is held in memory and not persisted to DB (used directly for feature computation).
    """
    provider = GoogleTrendsProvider()
    trends = await provider.fetch_all_sectors()
    logger.info("ingest_trends.complete", sectors=list(trends.keys()))
    return trends
