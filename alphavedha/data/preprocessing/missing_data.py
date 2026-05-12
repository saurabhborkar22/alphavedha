"""Missing data handling for market time-series.

Rules:
- Market holidays: forward-fill prices, volume = 0
- Suspended stocks: flag, do NOT interpolate
- Provider outages: flag gaps, forward-fill with is_filled marker
- NEVER interpolate prices
"""

from __future__ import annotations

import pandas as pd
import structlog

from alphavedha.config import get_config

logger = structlog.get_logger(__name__)


def generate_trading_calendar(start: pd.Timestamp, end: pd.Timestamp) -> pd.DatetimeIndex:
    """Generate expected trading days (Mon-Fri, excluding weekends).

    Does not account for Indian market holidays — those are detected as expected gaps.
    """
    return pd.bdate_range(start=start, end=end, freq="B")


def handle_missing_data(
    df: pd.DataFrame,
    expected_dates: pd.DatetimeIndex | None = None,
    max_gap_days: int | None = None,
) -> pd.DataFrame:
    """Fill gaps in OHLCV data with forward-fill and flag filled rows.

    Does NOT interpolate — only forward-fills prices and sets volume to 0.
    """
    if df.empty:
        return df.copy()

    if max_gap_days is None:
        cfg = get_config()
        max_gap_days = cfg.preprocessing.missing_data.max_gap_days

    result = df.copy()

    if "is_filled" not in result.columns:
        result["is_filled"] = False

    if expected_dates is None:
        if not isinstance(result.index, pd.DatetimeIndex):
            return result
        expected_dates = generate_trading_calendar(result.index.min(), result.index.max())

    missing_dates = expected_dates.difference(result.index)

    if missing_dates.empty:
        return result

    gaps = _find_consecutive_gaps(missing_dates, max_gap_days)

    missing_df = pd.DataFrame(index=missing_dates)
    result = pd.concat([result, missing_df]).sort_index()

    price_cols = [c for c in ["open", "high", "low", "close", "adj_close"] if c in result.columns]
    result[price_cols] = result[price_cols].ffill()

    if "volume" in result.columns:
        result.loc[missing_dates, "volume"] = 0

    result.loc[missing_dates, "is_filled"] = True

    n_filled = len(missing_dates)
    logger.info(
        "missing_data_filled",
        filled_rows=n_filled,
        total_rows=len(result),
        long_gaps=len(gaps),
    )

    return result


def _find_consecutive_gaps(
    missing_dates: pd.DatetimeIndex, max_gap_days: int
) -> list[tuple[pd.Timestamp, pd.Timestamp, int]]:
    """Find consecutive gaps that exceed max_gap_days."""
    if missing_dates.empty:
        return []

    gaps: list[tuple[pd.Timestamp, pd.Timestamp, int]] = []
    sorted_dates = missing_dates.sort_values()

    gap_start = sorted_dates[0]
    gap_length = 1

    for i in range(1, len(sorted_dates)):
        delta = (sorted_dates[i] - sorted_dates[i - 1]).days
        if delta <= 3:
            gap_length += 1
        else:
            if gap_length >= max_gap_days:
                gaps.append((gap_start, sorted_dates[i - 1], gap_length))
                logger.warning(
                    "large_data_gap",
                    gap_start=str(gap_start.date()),
                    gap_end=str(sorted_dates[i - 1].date()),
                    business_days=gap_length,
                )
            gap_start = sorted_dates[i]
            gap_length = 1

    if gap_length >= max_gap_days:
        gaps.append((gap_start, sorted_dates[-1], gap_length))

    return gaps


def detect_suspensions(df: pd.DataFrame, min_zero_volume_days: int = 5) -> pd.Series:
    """Flag periods where a stock appears suspended (zero volume for extended period)."""
    if df.empty or "volume" not in df.columns:
        return pd.Series(False, index=df.index)

    is_zero_vol = df["volume"] == 0
    groups = (~is_zero_vol).cumsum()
    consecutive = is_zero_vol.groupby(groups).transform("sum")

    return (consecutive >= min_zero_volume_days) & is_zero_vol
