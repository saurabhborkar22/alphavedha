"""Fundamental features — 5 earnings-based signals for PEAD analysis.

Requires earnings_df DataFrame with quarterly results.
Graceful degradation: returns NaN if no earnings data available.
Column naming: fund_{indicator}.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import structlog

logger = structlog.get_logger(__name__)

FUNDAMENTAL_FEATURE_COUNT = 5


def compute_fundamental_features(
    ohlcv_df: pd.DataFrame,
    earnings_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Compute 5 fundamental features from quarterly earnings data.

    Args:
        ohlcv_df: DataFrame with OHLCV data, DatetimeIndex.
        earnings_df: DataFrame with columns: symbol, quarter, year,
            revenue_actual, profit_actual, announced_date, expenses.
            Sorted by (year, quarter).

    Returns:
        DataFrame with 5 fund_* columns, same index as ohlcv_df.
    """
    result = pd.DataFrame(index=ohlcv_df.index)

    if earnings_df is None or earnings_df.empty:
        logger.warning("fundamental_no_earnings", msg="No earnings data, returning NaN")
        for col in _ALL_COLUMNS:
            result[col] = np.nan
        return result

    earnings = earnings_df.copy()

    if "announced_date" in earnings.columns:
        earnings["announced_date"] = pd.to_datetime(earnings["announced_date"])
    else:
        logger.warning("fundamental_no_announced_date", msg="No announced_date, estimating")
        earnings["announced_date"] = pd.NaT

    earnings = earnings.sort_values("announced_date").reset_index(drop=True)

    revenue = earnings["revenue_actual"].values
    profit = earnings["profit_actual"].values
    expenses = earnings["expenses"].values if "expenses" in earnings.columns else None
    announced = earnings["announced_date"].values

    surprise_pct = _compute_surprise_series(earnings)
    revenue_growth = _compute_qoq_growth(revenue)
    margin_change = _compute_margin_change(revenue, profit, expenses)
    streak = _compute_surprise_streak(surprise_pct)

    for idx in ohlcv_df.index:
        ts = pd.Timestamp(idx)
        mask = earnings["announced_date"] <= ts
        last_idx = mask.sum() - 1

        if last_idx < 0:
            result.loc[idx, "fund_earnings_surprise_pct"] = np.nan
            result.loc[idx, "fund_days_since_earnings"] = np.nan
            result.loc[idx, "fund_earnings_surprise_streak"] = np.nan
            result.loc[idx, "fund_revenue_growth_qoq"] = np.nan
            result.loc[idx, "fund_profit_margin_change"] = np.nan
            continue

        result.loc[idx, "fund_earnings_surprise_pct"] = surprise_pct[last_idx]

        last_date = pd.Timestamp(announced[last_idx])
        if pd.notna(last_date):
            result.loc[idx, "fund_days_since_earnings"] = (ts - last_date).days
        else:
            result.loc[idx, "fund_days_since_earnings"] = np.nan

        result.loc[idx, "fund_earnings_surprise_streak"] = streak[last_idx]
        result.loc[idx, "fund_revenue_growth_qoq"] = revenue_growth[last_idx]
        result.loc[idx, "fund_profit_margin_change"] = margin_change[last_idx]

    logger.info(
        "fundamental_features_computed",
        n_features=FUNDAMENTAL_FEATURE_COUNT,
        n_earnings_quarters=len(earnings),
    )
    return result


_ALL_COLUMNS = [
    "fund_earnings_surprise_pct",
    "fund_days_since_earnings",
    "fund_earnings_surprise_streak",
    "fund_revenue_growth_qoq",
    "fund_profit_margin_change",
]


def _compute_surprise_series(earnings: pd.DataFrame) -> list[float]:
    """Compute earnings surprise % for each quarter.

    Uses sequential QoQ profit growth as proxy when estimates unavailable.
    """
    surprises: list[float] = []
    profit = earnings["profit_actual"].values

    has_estimates = (
        "profit_estimate" in earnings.columns
        and earnings["profit_estimate"].notna().any()
    )

    for i in range(len(earnings)):
        if has_estimates and pd.notna(earnings["profit_estimate"].iloc[i]):
            est = earnings["profit_estimate"].iloc[i]
            act = profit[i]
            if est and est != 0 and pd.notna(act):
                surprises.append(((act - est) / abs(est)) * 100.0)
            else:
                surprises.append(np.nan)
        elif i >= 4 and pd.notna(profit[i]) and pd.notna(profit[i - 4]):
            prev = profit[i - 4]
            if prev != 0:
                surprises.append(((profit[i] - prev) / abs(prev)) * 100.0)
            else:
                surprises.append(np.nan)
        else:
            surprises.append(np.nan)

    return surprises


def _compute_qoq_growth(revenue: np.ndarray) -> list[float]:
    """Compute quarter-over-quarter revenue growth."""
    growth: list[float] = [np.nan]
    for i in range(1, len(revenue)):
        if pd.notna(revenue[i]) and pd.notna(revenue[i - 1]) and revenue[i - 1] != 0:
            growth.append(((revenue[i] - revenue[i - 1]) / abs(revenue[i - 1])) * 100.0)
        else:
            growth.append(np.nan)
    return growth


def _compute_margin_change(
    revenue: np.ndarray,
    profit: np.ndarray,
    expenses: np.ndarray | None,
) -> list[float]:
    """Compute profit margin change vs previous quarter."""
    margins: list[float] = []
    for i in range(len(revenue)):
        if pd.notna(profit[i]) and pd.notna(revenue[i]) and revenue[i] != 0:
            margins.append((profit[i] / revenue[i]) * 100.0)
        else:
            margins.append(np.nan)

    changes: list[float] = [np.nan]
    for i in range(1, len(margins)):
        if pd.notna(margins[i]) and pd.notna(margins[i - 1]):
            changes.append(margins[i] - margins[i - 1])
        else:
            changes.append(np.nan)
    return changes


def _compute_surprise_streak(surprises: list[float]) -> list[float]:
    """Compute consecutive beats (positive) or misses (negative)."""
    streaks: list[float] = []
    current_streak = 0.0

    for s in surprises:
        if pd.isna(s):
            streaks.append(current_streak)
            continue
        if s > 0:
            current_streak = max(0, current_streak) + 1
        elif s < 0:
            current_streak = min(0, current_streak) - 1
        else:
            current_streak = 0
        streaks.append(current_streak)

    return streaks
