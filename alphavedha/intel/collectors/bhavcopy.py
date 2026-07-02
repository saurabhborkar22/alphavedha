"""NSE bhavcopy collector — whole-market EOD OHLCV in a single file per day.

Downloads the full bhavcopy CSV from NSE via jugaad-data, filters to equity
series (EQ + BE), normalises column names, and upserts into ``daily_ohlcv``.

One bhavcopy file covers ~2,400 stocks — this replaces 2,400 individual
yfinance calls and keeps the VPS well within NSE rate limits.
"""

from __future__ import annotations

import asyncio
import io
from datetime import date

import pandas as pd
import structlog

from alphavedha.data.providers.base import RateLimiter
from alphavedha.data.store import store_ohlcv

logger = structlog.get_logger(__name__)

_rate_limiter = RateLimiter(requests_per_second=0.5)

EQUITY_SERIES = {"EQ", "BE"}


def _fetch_bhavcopy_raw(dt: date) -> str:
    """Fetch raw bhavcopy CSV string from NSE. Blocking call."""
    from jugaad_data import nse

    return str(nse.full_bhavcopy_raw(dt))


def parse_bhavcopy(raw_csv: str) -> pd.DataFrame:
    """Parse bhavcopy CSV into a normalised DataFrame.

    Returns a DataFrame with columns: symbol, date, open, high, low, close,
    volume, delivery_pct — one row per equity-series stock.
    """
    df = pd.read_csv(io.StringIO(raw_csv))
    df.columns = [c.strip() for c in df.columns]

    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].str.strip()

    df = df[df["SERIES"].isin(EQUITY_SERIES)].copy()

    if df.empty:
        return pd.DataFrame()

    df["date"] = pd.to_datetime(df["DATE1"], format="%d-%b-%Y").dt.date

    result = pd.DataFrame(
        {
            # Bare NSE code is the store's canonical symbol form — the old
            # ".NS" suffix created a split-brain table vs yfinance rows.
            "symbol": df["SYMBOL"],
            "date": df["date"],
            "open": pd.to_numeric(df["OPEN_PRICE"], errors="coerce"),
            "high": pd.to_numeric(df["HIGH_PRICE"], errors="coerce"),
            "low": pd.to_numeric(df["LOW_PRICE"], errors="coerce"),
            "close": pd.to_numeric(df["CLOSE_PRICE"], errors="coerce"),
            "volume": pd.to_numeric(df["TTL_TRD_QNTY"], errors="coerce").fillna(0).astype(int),
            "delivery_pct": pd.to_numeric(df["DELIV_PER"], errors="coerce"),
        }
    )

    result = result.dropna(subset=["open", "high", "low", "close"])
    result["adj_close"] = result["close"]

    return result.reset_index(drop=True)


async def fetch_bhavcopy(dt: date) -> pd.DataFrame:
    """Fetch and parse one day's bhavcopy. Returns normalised DataFrame."""
    await _rate_limiter.acquire()
    raw = await asyncio.to_thread(_fetch_bhavcopy_raw, dt)
    return parse_bhavcopy(raw)


async def ingest_bhavcopy(dt: date) -> int:
    """Fetch bhavcopy for a date, parse, and upsert into daily_ohlcv.

    Returns the number of rows stored.
    """
    logger.info("bhavcopy_ingest_start", date=dt.isoformat())

    try:
        df = await fetch_bhavcopy(dt)
    except Exception as e:
        logger.error("bhavcopy_fetch_failed", date=dt.isoformat(), error=str(e))
        return 0

    if df.empty:
        logger.warning("bhavcopy_empty", date=dt.isoformat())
        return 0

    total = 0
    for symbol, group in df.groupby("symbol"):
        group_indexed = group.set_index("date").drop(columns=["symbol"])
        group_indexed.index = pd.to_datetime(group_indexed.index)
        stored = await store_ohlcv(str(symbol), group_indexed)
        total += stored

    logger.info("bhavcopy_ingest_complete", date=dt.isoformat(), symbols=total)
    return total


async def backfill_bhavcopy(start: date, end: date) -> dict[str, int]:
    """Backfill bhavcopy data for a date range.

    Skips weekends. Returns a dict of {date_iso: rows_stored}.
    """
    results: dict[str, int] = {}
    current = start

    while current <= end:
        if current.weekday() >= 5:
            current = date.fromordinal(current.toordinal() + 1)
            continue

        try:
            rows = await ingest_bhavcopy(current)
            results[current.isoformat()] = rows
        except Exception as e:
            logger.error("bhavcopy_backfill_failed", date=current.isoformat(), error=str(e))
            results[current.isoformat()] = 0

        current = date.fromordinal(current.toordinal() + 1)

    logger.info(
        "bhavcopy_backfill_complete",
        start=start.isoformat(),
        end=end.isoformat(),
        days=len(results),
        total_rows=sum(results.values()),
    )
    return results
