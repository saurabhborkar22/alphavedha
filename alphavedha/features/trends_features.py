"""Google Trends feature computation for market sectors."""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import structlog

logger = structlog.get_logger(__name__)

TRENDS_FEATURE_COUNT = 2


def compute_trends_features(
    symbol: str,
    as_of_date: date,
    trends_df: pd.DataFrame | None,
) -> dict[str, float]:
    """Compute 2 Google Trends features for a symbol.

    Features:
        trends_sector_7d: average sector search interest over last 7 days (0-100 scale)
        trends_sector_change: 7-day avg minus prior 7-day avg (positive = rising interest)

    Args:
        symbol: NSE symbol e.g. "TCS.NS"
        as_of_date: The date we are computing features for (no future data)
        trends_df: DataFrame indexed by date with sector columns (e.g. "it", "banking").
                   Can be None if trends data is unavailable.

    Returns:
        dict with exactly 2 keys. Returns NaN values if data unavailable.
    """
    nan = float("nan")
    defaults: dict[str, float] = {
        "trends_sector_7d": nan,
        "trends_sector_change": nan,
    }

    if trends_df is None or trends_df.empty:
        return defaults

    # Import here to avoid circular imports at module load time
    from alphavedha.data.providers.trends_provider import SYMBOL_TO_SECTOR

    sector = SYMBOL_TO_SECTOR.get(symbol)
    if sector is None or sector not in trends_df.columns:
        return defaults

    try:
        idx_dates = pd.to_datetime(trends_df.index).date
    except Exception:
        return defaults

    window_end = as_of_date
    window_start = as_of_date - timedelta(days=6)  # last 7 days inclusive
    prev_start = as_of_date - timedelta(days=13)  # prior 7 days

    mask_recent = (idx_dates >= window_start) & (idx_dates <= window_end)
    mask_prior = (idx_dates >= prev_start) & (idx_dates < window_start)

    recent_vals = trends_df.loc[mask_recent, sector].dropna()
    prior_vals = trends_df.loc[mask_prior, sector].dropna()

    if recent_vals.empty:
        return defaults

    sector_7d = float(recent_vals.mean())
    sector_change = float(recent_vals.mean() - prior_vals.mean()) if not prior_vals.empty else nan

    logger.debug(
        "trends_features.computed",
        symbol=symbol,
        sector=sector,
        as_of=str(as_of_date),
        sector_7d=sector_7d,
        sector_change=sector_change,
    )

    return {
        "trends_sector_7d": sector_7d,
        "trends_sector_change": sector_change,
    }
