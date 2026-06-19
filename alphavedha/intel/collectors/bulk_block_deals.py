"""Bulk and block deals collector — daily deal reports from NSE.

Fetches bulk deals, block deals, and short-selling data from the NSE
large-deal snapshot API, normalises into ``bulk_block_deals`` rows.
"""

from __future__ import annotations

import asyncio
from datetime import date, datetime
from typing import Any

import structlog

from alphavedha.data.providers.base import RateLimiter
from alphavedha.data.providers.nse_provider import NSESession

logger = structlog.get_logger(__name__)

_DEALS_URL = "https://www.nseindia.com/api/snapshot-capital-market-largedeal"

_rate_limiter = RateLimiter(requests_per_second=0.5)


def _parse_date(dt_str: str) -> date | None:
    """Parse NSE deal date string."""
    for fmt in ("%d-%b-%Y", "%d-%B-%Y", "%d/%m/%Y"):
        try:
            return datetime.strptime(dt_str.strip(), fmt).date()
        except (ValueError, TypeError):
            continue
    return None


def _parse_deal_rows(
    items: list[dict[str, Any]],
    deal_type: str,
) -> list[dict[str, Any]]:
    """Parse a list of NSE deal rows into normalised dicts."""
    rows: list[dict[str, Any]] = []

    for item in items:
        symbol = (item.get("symbol", "") or "").strip()
        if not symbol:
            continue

        date_str = (item.get("date", "") or "").strip()
        deal_date = _parse_date(date_str) if date_str else date.today()
        if deal_date is None:
            deal_date = date.today()

        client = (item.get("clientName", "") or "").strip() or "Unknown"
        buy_sell = (item.get("buySell", "") or "").strip().upper()
        trade_type = buy_sell if buy_sell in ("BUY", "SELL") else "UNKNOWN"

        try:
            qty = int(str(item.get("qty", "0")).replace(",", ""))
        except (ValueError, TypeError):
            qty = 0

        try:
            price = float(str(item.get("watp", "0")).replace(",", ""))
        except (ValueError, TypeError):
            price = 0.0

        if qty <= 0:
            continue

        rows.append(
            {
                "symbol": f"{symbol}.NS",
                "deal_date": deal_date,
                "deal_type": deal_type[:10],
                "client_name": client[:200],
                "trade_type": trade_type[:10],
                "quantity": qty,
                "price": price,
            }
        )

    return rows


def _fetch_deals_sync(session: NSESession) -> dict[str, Any]:
    """Fetch large deals snapshot from NSE. Blocking."""
    resp = session.get(_DEALS_URL)
    data = resp.json()
    if isinstance(data, dict):
        return data
    return {}


async def collect_bulk_block_deals() -> list[dict[str, Any]]:
    """Fetch today's bulk, block, and short-selling deals from NSE."""
    await _rate_limiter.acquire()
    session = NSESession()

    data = await asyncio.to_thread(_fetch_deals_sync, session)

    bulk_rows = _parse_deal_rows(list(data.get("BULK_DEALS_DATA", [])), "BULK")
    block_rows = _parse_deal_rows(list(data.get("BLOCK_DEALS_DATA", [])), "BLOCK")
    short_rows = _parse_deal_rows(list(data.get("SHORT_DEALS_DATA", [])), "SHORT")

    all_rows = bulk_rows + block_rows + short_rows

    logger.info(
        "deals_collect_complete",
        bulk=len(bulk_rows),
        block=len(block_rows),
        short=len(short_rows),
        total=len(all_rows),
    )
    return all_rows


async def ingest_bulk_block_deals_daily() -> int:
    """Daily ingestion: fetch bulk/block deals, store, return count."""
    from alphavedha.intel.store import store_bulk_block_deals

    rows = await collect_bulk_block_deals()
    if not rows:
        logger.info("deals_no_new_deals")
        return 0

    return await store_bulk_block_deals(rows)
