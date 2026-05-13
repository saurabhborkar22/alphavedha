"""Return-derived features — 20 features from price data.

Log returns (not simple), rolling statistics, risk metrics, and momentum.
Column naming: ret_{indicator}.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import structlog

logger = structlog.get_logger(__name__)

RETURN_FEATURE_COUNT = 21


def compute_return_features(
    df: pd.DataFrame,
    frac_diff_col: str | None = None,
) -> pd.DataFrame:
    """Compute 20 return-derived features.

    Args:
        df: DataFrame with 'close' column and DatetimeIndex.
        frac_diff_col: Name of fractionally differentiated column in df (from preprocessing).
            If None or not found, ret_frac_diff is filled with 0.

    Returns:
        DataFrame with 20 ret_* columns, same index as input.
    """
    close = df["close"].astype(float)
    result = pd.DataFrame(index=df.index)

    log_ret_1d = np.log(close / close.shift(1))
    result["ret_log_1d"] = log_ret_1d
    result["ret_log_5d"] = np.log(close / close.shift(5))
    result["ret_log_10d"] = np.log(close / close.shift(10))
    result["ret_log_20d"] = np.log(close / close.shift(20))

    result["ret_mean_5d"] = log_ret_1d.rolling(5).mean()
    result["ret_mean_20d"] = log_ret_1d.rolling(20).mean()
    result["ret_std_5d"] = log_ret_1d.rolling(5).std()
    result["ret_std_20d"] = log_ret_1d.rolling(20).std()

    result["ret_skew_20d"] = log_ret_1d.rolling(20).skew()
    result["ret_kurt_20d"] = log_ret_1d.rolling(20).kurt()

    rolling_mean = log_ret_1d.rolling(20).mean()
    rolling_std = log_ret_1d.rolling(20).std()
    result["ret_sharpe_20d"] = (rolling_mean / rolling_std) * np.sqrt(252)

    rolling_cumret = (
        (1 + log_ret_1d.apply(np.exp) - 1)
        .rolling(20)
        .apply(
            lambda x: (
                (x.cumprod().cummax() - x.cumprod()).max() / x.cumprod().cummax().max()
                if x.cumprod().cummax().max() > 0
                else 0
            ),
            raw=False,
        )
    )
    result["ret_max_dd_20d"] = rolling_cumret

    def _up_ratio(x: pd.Series) -> float:
        total = len(x)
        if total == 0:
            return 0.5
        return float((x > 0).sum() / total)

    result["ret_up_ratio_20d"] = log_ret_1d.rolling(20).apply(_up_ratio, raw=False)

    result["ret_mom_5d"] = close / close.shift(5) - 1
    result["ret_mom_20d"] = close / close.shift(20) - 1
    result["ret_mom_60d"] = close / close.shift(60) - 1

    result["ret_zscore_20d"] = (log_ret_1d - rolling_mean) / rolling_std

    if frac_diff_col and frac_diff_col in df.columns:
        result["ret_frac_diff"] = df[frac_diff_col]
    else:
        result["ret_frac_diff"] = 0.0

    rolling_high_252 = close.rolling(252, min_periods=20).max()
    rolling_low_252 = close.rolling(252, min_periods=20).min()
    result["ret_52w_high_dist"] = (close - rolling_high_252) / rolling_high_252
    result["ret_52w_low_dist"] = (close - rolling_low_252) / rolling_low_252

    result["ret_regime"] = 1

    logger.info(
        "return_features_computed",
        n_features=len(result.columns),
        n_rows=len(result),
    )
    return result
