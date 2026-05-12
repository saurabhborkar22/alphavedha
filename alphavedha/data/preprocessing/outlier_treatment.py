"""Outlier treatment — winsorization of computed features.

Rules:
- Winsorize features at 1st and 99th percentile
- Do NOT winsorize prices or returns — only computed features
- Log outlier counts per feature for drift monitoring
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import structlog

from alphavedha.config import get_config

logger = structlog.get_logger(__name__)

PRICE_COLUMNS = {
    "open",
    "high",
    "low",
    "close",
    "adj_close",
    "volume",
    "raw_open",
    "raw_high",
    "raw_low",
    "raw_close",
    "raw_volume",
}
RETURN_COLUMNS = {"log_return", "return_1d", "return_5d", "return_10d", "return_20d"}
SKIP_COLUMNS = (
    PRICE_COLUMNS
    | RETURN_COLUMNS
    | {"date", "symbol", "is_filled", "is_adjusted", "circuit_hit", "circuit_band", "is_suspended"}
)


def winsorize_series(
    series: pd.Series,
    lower_pct: float = 0.01,
    upper_pct: float = 0.99,
) -> tuple[pd.Series, int]:
    """Winsorize a single series. Returns (clipped_series, outlier_count)."""
    valid = series.dropna()
    if valid.empty:
        return series.copy(), 0

    lower_bound = valid.quantile(lower_pct)
    upper_bound = valid.quantile(upper_pct)

    outliers = ((series < lower_bound) | (series > upper_bound)).sum()
    clipped = series.clip(lower=lower_bound, upper=upper_bound)

    return clipped, int(outliers)


def winsorize_features(
    df: pd.DataFrame,
    lower_pct: float | None = None,
    upper_pct: float | None = None,
    skip_columns: set[str] | None = None,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Winsorize all feature columns in the DataFrame.

    Returns (winsorized_df, outlier_counts_per_column).
    """
    if lower_pct is None or upper_pct is None:
        cfg = get_config()
        lower_pct = lower_pct or cfg.preprocessing.outlier.winsorize_lower
        upper_pct = upper_pct or cfg.preprocessing.outlier.winsorize_upper

    if skip_columns is None:
        skip_columns = SKIP_COLUMNS

    result = df.copy()
    outlier_counts: dict[str, int] = {}

    numeric_cols = result.select_dtypes(include=[np.number]).columns

    for col in numeric_cols:
        if col in skip_columns:
            continue

        result[col], n_outliers = winsorize_series(result[col], lower_pct, upper_pct)
        if n_outliers > 0:
            outlier_counts[col] = n_outliers

    if outlier_counts:
        logger.info(
            "outliers_winsorized",
            columns_affected=len(outlier_counts),
            total_outliers=sum(outlier_counts.values()),
        )

    return result, outlier_counts
