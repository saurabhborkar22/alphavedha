"""Fractional differentiation — preserve memory while achieving stationarity.

Uses fixed-width window fractional differentiation (Marcos López de Prado).
Computes minimum d per stock that passes ADF test (p < 0.05).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import structlog
from statsmodels.tsa.stattools import adfuller

from alphavedha.config import get_config

logger = structlog.get_logger(__name__)


def _get_weights_ffd(d: float, max_lags: int) -> np.ndarray:
    """Compute weights for fixed-width window fractional differentiation."""
    weights = [1.0]
    for k in range(1, max_lags):
        w = -weights[-1] * (d - k + 1) / k
        weights.append(w)
    return np.array(weights[::-1])


def frac_diff_ffd(
    series: pd.Series,
    d: float,
    max_lags: int | None = None,
) -> pd.Series:
    """Apply fixed-width window fractional differentiation to a series."""
    if max_lags is None:
        cfg = get_config()
        max_lags = cfg.preprocessing.fractional_diff.max_lags

    weights = _get_weights_ffd(d, max_lags)

    width = len(weights)
    result = pd.Series(index=series.index, dtype=np.float64)

    for i in range(width - 1, len(series)):
        window = series.iloc[i - width + 1 : i + 1].values
        if len(window) == width:
            result.iloc[i] = np.dot(weights, window)

    return result


def find_min_d(
    series: pd.Series,
    max_lags: int | None = None,
    adf_pvalue: float | None = None,
    d_range: tuple[float, float] | None = None,
    step: float = 0.05,
) -> float:
    """Find minimum d that makes the series stationary (ADF p-value < threshold).

    Scans d values from min to max in steps, returns first d that passes ADF.
    """
    if max_lags is None or adf_pvalue is None or d_range is None:
        cfg = get_config()
        frac_cfg = cfg.preprocessing.fractional_diff
        max_lags = max_lags or frac_cfg.max_lags
        adf_pvalue = adf_pvalue or frac_cfg.adf_pvalue_threshold
        d_range = d_range or (frac_cfg.min_d, frac_cfg.max_d)

    clean = series.dropna()
    if len(clean) < max_lags + 50:
        logger.warning("insufficient_data_for_frac_diff", length=len(clean))
        return d_range[1]

    d_low, d_high = d_range
    best_d = d_high

    d_values = np.arange(d_low, d_high + step, step)

    for d in d_values:
        diffed = frac_diff_ffd(clean, d, max_lags)
        valid = diffed.dropna()

        if len(valid) < 30:
            continue

        try:
            adf_stat, pval, *_ = adfuller(valid, maxlag=20, autolag="AIC")
        except Exception:
            continue

        if pval < adf_pvalue:
            best_d = round(float(d), 3)
            logger.info(
                "min_d_found",
                d=best_d,
                adf_pvalue=round(pval, 4),
                adf_stat=round(adf_stat, 4),
            )
            break

    return best_d


def frac_diff_dataframe(
    df: pd.DataFrame,
    d_values: dict[str, float] | None = None,
    columns: list[str] | None = None,
    max_lags: int | None = None,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Apply fractional differentiation to multiple columns.

    If d_values not provided, computes optimal d for each column.
    Returns (transformed_df, dict of column -> d value used).
    """
    if columns is None:
        columns = ["close"]

    result = df.copy()
    used_d: dict[str, float] = {}

    for col in columns:
        if col not in df.columns:
            continue

        if d_values and col in d_values:
            d = d_values[col]
        else:
            d = find_min_d(df[col], max_lags=max_lags)

        used_d[col] = d
        result[f"{col}_fracdiff"] = frac_diff_ffd(df[col], d, max_lags)

    if used_d:
        logger.info("frac_diff_applied", d_values=used_d)

    return result, used_d
