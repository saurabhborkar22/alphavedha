"""Corporate action adjustment — splits, bonuses, rights issues, dividends.

Adjusts historical prices BACKWARDS from current price so all prices
are comparable across time. Stores both raw and adjusted values.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd
import structlog

logger = structlog.get_logger(__name__)


@dataclass
class CorporateActionRecord:
    symbol: str
    ex_date: date
    action_type: str
    ratio: float
    details: str = ""


VALID_ACTION_TYPES = {"split", "bonus", "rights", "dividend"}


def compute_adjustment_factors(
    actions: list[CorporateActionRecord],
    price_dates: pd.DatetimeIndex,
) -> pd.Series:
    """Compute cumulative adjustment factors for a series of corporate actions.

    Factor is applied multiplicatively: adj_price = raw_price * factor.
    Actions are applied backwards from most recent to oldest.
    """
    factors = pd.Series(1.0, index=price_dates, dtype=np.float64)

    sorted_actions = sorted(actions, key=lambda a: a.ex_date, reverse=True)

    for action in sorted_actions:
        if action.action_type not in VALID_ACTION_TYPES:
            logger.warning(
                "unknown_corporate_action_type",
                symbol=action.symbol,
                action_type=action.action_type,
            )
            continue

        if action.ratio <= 0 or action.ratio == 1.0:
            continue

        ex_dt = pd.Timestamp(action.ex_date)
        mask = price_dates < ex_dt

        if action.action_type in ("split", "bonus", "rights"):
            factors[mask] /= action.ratio
        elif action.action_type == "dividend":
            continue

        logger.info(
            "adjustment_factor_applied",
            symbol=action.symbol,
            action=action.action_type,
            ratio=action.ratio,
            ex_date=str(action.ex_date),
            affected_rows=int(mask.sum()),
        )

    return factors


def adjust_ohlcv(
    df: pd.DataFrame,
    actions: list[CorporateActionRecord],
) -> pd.DataFrame:
    """Apply corporate action adjustments to OHLCV data.

    Returns a copy with adjusted prices. Volume is inverse-adjusted.
    Original columns are preserved as raw_* columns.
    """
    if not actions or df.empty:
        return df.copy()

    result = df.copy()
    price_cols = ["open", "high", "low", "close"]

    for col in price_cols:
        if col in result.columns:
            result[f"raw_{col}"] = result[col].copy()

    factors = compute_adjustment_factors(actions, result.index)

    for col in price_cols:
        if col in result.columns:
            result[col] = result[col] * factors

    if "adj_close" in result.columns:
        result["adj_close"] = result["close"]

    if "volume" in result.columns:
        result["raw_volume"] = result["volume"].copy()
        safe_factors = factors.replace(0, np.nan)
        result["volume"] = (result["volume"] / safe_factors).round().astype(np.int64)

    result["is_adjusted"] = True

    logger.info(
        "ohlcv_adjusted",
        actions_applied=len(actions),
        rows=len(result),
    )
    return result


def detect_potential_splits(df: pd.DataFrame, threshold: float = 0.4) -> list[date]:
    """Heuristic: detect dates where close dropped >40% overnight (likely unadjusted split)."""
    if df.empty or "close" not in df.columns:
        return []

    returns = df["close"].pct_change()
    split_dates = returns[returns < -threshold].index
    return [d.date() if hasattr(d, "date") else d for d in split_dates]
