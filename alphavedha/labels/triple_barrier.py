"""Triple barrier labeling — generates direction labels from OHLCV + ATR."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import structlog
from ta.volatility import AverageTrueRange

from alphavedha.config import TripleBarrierConfig
from alphavedha.exceptions import InsufficientDataError

logger = structlog.get_logger(__name__)

_MIN_ROWS = 50


@dataclass
class LabelResult:
    df: pd.DataFrame
    symbol: str
    label_counts: dict[int, int] = field(default_factory=dict)
    skipped_low_atr: int = 0
    avg_days_to_hit: float = 0.0


def _compute_atr(df: pd.DataFrame, period: int) -> pd.Series:
    """Compute ATR using only past data at each point (no look-ahead)."""
    return AverageTrueRange(
        high=df["high"], low=df["low"], close=df["close"], window=period
    ).average_true_range()


def compute_triple_barrier_labels(
    ohlcv_df: pd.DataFrame,
    config: TripleBarrierConfig,
    symbol: str = "",
) -> LabelResult:
    if len(ohlcv_df) < _MIN_ROWS:
        raise InsufficientDataError(f"Need >= {_MIN_ROWS} rows for labeling, got {len(ohlcv_df)}")

    df = ohlcv_df.copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    atr = _compute_atr(df, config.atr_period)
    closes = df["close"].values
    highs = df["high"].values
    lows = df["low"].values
    n = len(df)
    hp = config.max_holding_period

    labels = np.full(n, np.nan)
    return_pcts = np.full(n, np.nan)
    barrier_hits = np.full(n, None, dtype=object)
    days_to_hits = np.full(n, np.nan)
    entry_prices = np.full(n, np.nan)
    exit_prices = np.full(n, np.nan)
    atr_at_entries = np.full(n, np.nan)

    skipped_low_atr = 0

    for t in range(n):
        atr_val = atr.iloc[t]

        if pd.isna(atr_val) or atr_val <= 0:
            continue

        if atr_val / closes[t] < config.min_atr_threshold:
            skipped_low_atr += 1
            continue

        if t + hp >= n:
            continue

        entry = closes[t]
        upper = entry + config.multiplier_up * atr_val
        lower = entry - config.multiplier_down * atr_val
        atr_at_entries[t] = atr_val
        entry_prices[t] = entry

        hit_label = None
        hit_day = None
        hit_price = None

        for d in range(1, hp + 1):
            idx = t + d
            h = highs[idx]
            lo = lows[idx]

            upper_touched = h >= upper
            lower_touched = lo <= lower

            if upper_touched and lower_touched:
                hit_label = -1
                hit_day = d
                hit_price = lower
                break
            elif upper_touched:
                hit_label = 1
                hit_day = d
                hit_price = upper
                break
            elif lower_touched:
                hit_label = -1
                hit_day = d
                hit_price = lower
                break

        if hit_label is None:
            hit_label = 0
            hit_day = hp
            hit_price = closes[t + hp]

        labels[t] = hit_label
        days_to_hits[t] = hit_day
        exit_prices[t] = hit_price
        return_pcts[t] = hit_price / entry - 1
        barrier_hits[t] = "upper" if hit_label == 1 else "lower" if hit_label == -1 else "time"

    result_df = pd.DataFrame(
        {
            "label": labels,
            "return_pct": return_pcts,
            "barrier_hit": barrier_hits,
            "days_to_hit": days_to_hits,
            "entry_price": entry_prices,
            "exit_price": exit_prices,
            "atr_at_entry": atr_at_entries,
        },
        index=df.index,
    )

    valid_labels = result_df["label"].dropna()
    label_counts: dict[int, int] = {}
    for val in (-1, 0, 1):
        label_counts[val] = int((valid_labels == val).sum())

    valid_days = result_df["days_to_hit"].dropna()
    avg_days = float(valid_days.mean()) if len(valid_days) > 0 else 0.0

    logger.info(
        "triple_barrier_labels_computed",
        symbol=symbol,
        n_rows=n,
        n_labeled=len(valid_labels),
        label_counts=label_counts,
        skipped_low_atr=skipped_low_atr,
        avg_days_to_hit=round(avg_days, 1),
    )

    return LabelResult(
        df=result_df,
        symbol=symbol,
        label_counts=label_counts,
        skipped_low_atr=skipped_low_atr,
        avg_days_to_hit=avg_days,
    )
