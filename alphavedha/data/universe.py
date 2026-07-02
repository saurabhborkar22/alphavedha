"""Universe manager — manages point-in-time index compositions.

Tracks Nifty 50, Midcap 150, Smallcap 250 compositions.
Stores historical compositions with effective dates to avoid survivorship bias.
"""

from __future__ import annotations

import io
from datetime import date
from typing import Any

import httpx
import pandas as pd
import structlog
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert

from alphavedha.config import get_config
from alphavedha.data.database import get_session_factory
from alphavedha.data.models import IndexConstituent

logger = structlog.get_logger(__name__)

INDEX_URLS: dict[str, str] = {
    "NIFTY 50": "https://www.niftyindices.com/IndexConstituent/ind_nifty50list.csv",
    "NIFTY MIDCAP 150": "https://www.niftyindices.com/IndexConstituent/ind_niftymidcap150list.csv",
    "NIFTY SMALLCAP 250": "https://www.niftyindices.com/IndexConstituent/ind_niftysmallcap250list.csv",
}

_HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36",
    "Accept": "text/csv,text/plain,*/*",
    "Referer": "https://www.niftyindices.com/",
}


async def fetch_index_constituents(index_name: str) -> pd.DataFrame:
    """Fetch current constituents of an index from niftyindices.com."""
    url = INDEX_URLS.get(index_name)
    if not url:
        raise ValueError(f"Unknown index: {index_name}. Known: {list(INDEX_URLS)}")

    async with httpx.AsyncClient(timeout=30, follow_redirects=True) as client:
        resp = await client.get(url, headers=_HEADERS)
        resp.raise_for_status()

    df = pd.read_csv(io.StringIO(resp.text))

    column_map: dict[str, str] = {}
    for col in df.columns:
        lower = col.strip().lower()
        if "symbol" in lower:
            column_map[col] = "symbol"
        elif "company" in lower or "name" in lower:
            column_map[col] = "company_name"
        elif "industry" in lower or "sector" in lower:
            column_map[col] = "sector"

    df = df.rename(columns=column_map)

    if "symbol" in df.columns:
        df["symbol"] = df["symbol"].str.strip()

    logger.info(
        "index_constituents_fetched",
        index=index_name,
        count=len(df),
    )
    return df


async def save_constituents(
    index_name: str,
    constituents: pd.DataFrame,
    effective_date: date | None = None,
) -> int:
    """Save index constituents to the database with point-in-time tracking."""
    if effective_date is None:
        effective_date = date.today()

    session_factory = get_session_factory()
    saved = 0

    async with session_factory() as session:
        for _, row in constituents.iterrows():
            symbol = row.get("symbol", "")
            if not symbol:
                continue

            stmt = (
                pg_insert(IndexConstituent)
                .values(
                    index_name=index_name,
                    symbol=symbol,
                    company_name=row.get("company_name"),
                    sector=row.get("sector"),
                    effective_from=effective_date,
                    effective_to=None,
                )
                .on_conflict_do_nothing()
            )
            await session.execute(stmt)
            saved += 1

        await session.commit()

    logger.info("constituents_saved", index=index_name, count=saved)
    return saved


async def get_constituents(
    index_name: str,
    as_of: date | None = None,
) -> list[dict[str, Any]]:
    """Get index constituents as of a specific date (point-in-time)."""
    if as_of is None:
        as_of = date.today()

    session_factory = get_session_factory()

    async with session_factory() as session:
        stmt = (
            select(IndexConstituent)
            .where(
                IndexConstituent.index_name == index_name,
                IndexConstituent.effective_from <= as_of,
                (IndexConstituent.effective_to.is_(None))
                | (IndexConstituent.effective_to >= as_of),
            )
            .order_by(IndexConstituent.symbol)
        )
        result = await session.execute(stmt)
        rows = result.scalars().all()

    return [
        {
            "symbol": r.symbol,
            "company_name": r.company_name,
            "sector": r.sector,
            "effective_from": r.effective_from,
        }
        for r in rows
    ]


async def get_symbols_for_tier(tier: str, as_of: date | None = None) -> list[str]:
    """Get list of stock symbols for a market cap tier."""
    cfg = get_config()
    tier_cfg = cfg.universe.available_tiers.get(tier)
    if not tier_cfg:
        raise ValueError(f"Unknown tier: {tier}. Available: {list(cfg.universe.available_tiers)}")

    constituents = await get_constituents(tier_cfg.index, as_of=as_of)
    return [c["symbol"] for c in constituents]


async def refresh_universe() -> dict[str, int]:
    """Fetch and save latest compositions for all configured tiers."""
    cfg = get_config()
    counts: dict[str, int] = {}

    for tier_name, tier_cfg in cfg.universe.available_tiers.items():
        try:
            df = await fetch_index_constituents(tier_cfg.index)
            saved = await save_constituents(tier_cfg.index, df)
            counts[tier_name] = saved
        except Exception:
            logger.exception("universe_refresh_failed", tier=tier_name)
            counts[tier_name] = 0

    logger.info("universe_refresh_complete", counts=counts)
    return counts


async def get_strategy_universe(tier: str = "large") -> list[str]:
    """Symbols the intel strategies scan — live index membership.

    The blowup/insider strategies used a hardcoded 20-name list
    (ui_support.NIFTY_50, misnamed) and silently scanned 40% of the
    index. Live constituents are the source of truth; the hardcoded
    list remains only as a fallback when the DB has no composition.
    """
    symbols: list[str] = []
    try:
        symbols = await get_symbols_for_tier(tier)
    except Exception as e:
        logger.warning("strategy_universe_fallback", tier=tier, error=str(e))

    if symbols:
        return [s.strip().upper() for s in symbols]

    from alphavedha.api.routes.ui_support import NIFTY_50

    return [s for s, _n, _sec, _c in NIFTY_50]
