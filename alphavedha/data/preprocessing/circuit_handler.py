"""Circuit limit detection for Indian stock markets.

Individual stocks have circuit limits at 5%, 10%, or 20% bands.
Index-level circuits trigger at 10%, 15%, 20%.
Circuit-hit days are flagged but NOT excluded from data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import structlog

from alphavedha.config import get_config

logger = structlog.get_logger(__name__)


def detect_circuit_hits(
    df: pd.DataFrame,
    thresholds: list[float] | None = None,
) -> pd.DataFrame:
    """Detect circuit limit hits based on daily price range vs previous close.

    Adds `circuit_hit` column: "upper", "lower", or None.
    Adds `circuit_band` column: the matched threshold (e.g., 0.05, 0.10, 0.20).
    """
    if df.empty:
        result = df.copy()
        result["circuit_hit"] = None
        result["circuit_band"] = np.nan
        return result

    if thresholds is None:
        cfg = get_config()
        thresholds = cfg.preprocessing.circuit.thresholds

    result = df.copy()
    result["circuit_hit"] = None
    result["circuit_band"] = np.nan

    if "close" not in result.columns:
        return result

    prev_close = result["close"].shift(1)
    daily_return = (result["close"] - prev_close) / prev_close

    tolerance = 0.002

    for threshold in sorted(thresholds, reverse=True):
        upper_mask = daily_return >= (threshold - tolerance)
        lower_mask = daily_return <= -(threshold - tolerance)

        result.loc[upper_mask, "circuit_hit"] = "upper"
        result.loc[upper_mask, "circuit_band"] = threshold

        result.loc[lower_mask, "circuit_hit"] = "lower"
        result.loc[lower_mask, "circuit_band"] = threshold

    n_upper = (result["circuit_hit"] == "upper").sum()
    n_lower = (result["circuit_hit"] == "lower").sum()

    if n_upper > 0 or n_lower > 0:
        logger.info(
            "circuit_hits_detected",
            upper=int(n_upper),
            lower=int(n_lower),
            total_rows=len(result),
        )

    return result
