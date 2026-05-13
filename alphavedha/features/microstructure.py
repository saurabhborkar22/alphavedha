"""Microstructure features — 10 India-specific delivery and volume signals.

Requires delivery_pct column from jugaad-data provider.
Graceful degradation: returns zeros if delivery_pct is missing.
Column naming: micro_{indicator}.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import structlog

logger = structlog.get_logger(__name__)

MICROSTRUCTURE_FEATURE_COUNT = 10


def compute_microstructure_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute 10 microstructure features.

    Args:
        df: DataFrame with OHLCV columns and optional delivery_pct.

    Returns:
        DataFrame with 10 micro_* columns, same index as input.
    """
    result = pd.DataFrame(index=df.index)
    close = df["close"].astype(float)
    open_price = df["open"].astype(float)
    volume = df["volume"].astype(float)

    has_delivery = "delivery_pct" in df.columns and df["delivery_pct"].notna().any()

    if has_delivery:
        delivery = df["delivery_pct"].astype(float)
    else:
        logger.warning("microstructure_no_delivery_pct", msg="delivery_pct missing, using zeros")
        delivery = pd.Series(0.0, index=df.index)

    result["micro_delivery_pct"] = delivery

    rolling_mean = delivery.rolling(20, min_periods=5).mean()
    rolling_std = delivery.rolling(20, min_periods=5).std()
    result["micro_delivery_zscore"] = (delivery - rolling_mean) / rolling_std.replace(0, np.nan)

    ma5 = delivery.rolling(5, min_periods=1).mean()
    result["micro_delivery_to_ma5"] = delivery / ma5.replace(0, np.nan)

    result["micro_delivery_trend_5d"] = delivery - delivery.shift(5)

    result["micro_delivery_accel"] = delivery.diff().diff()

    vol_ma_20 = volume.rolling(20, min_periods=5).mean()
    result["micro_vol_anomaly"] = volume / vol_ma_20.replace(0, np.nan)

    up_move = (close > open_price).astype(int)
    down_move = (close < open_price).astype(int)
    high_delivery = (delivery > 0.5).astype(int) if has_delivery else pd.Series(0, index=df.index)
    low_delivery = (delivery < 0.3).astype(int) if has_delivery else pd.Series(0, index=df.index)

    result["micro_hd_up"] = high_delivery * up_move
    result["micro_hd_down"] = high_delivery * down_move
    result["micro_ld_up"] = low_delivery * up_move

    result["micro_delivery_rolling_10d"] = delivery.rolling(10, min_periods=3).mean()

    logger.info(
        "microstructure_features_computed",
        n_features=len(result.columns),
        has_delivery=has_delivery,
    )
    return result
