"""Technical indicators — 40 features from OHLCV data using the `ta` library.

Groups: momentum (12), trend (10), volatility (8), volume (10).
All computed on adjusted close prices. Column naming: {indicator}_{window}.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import structlog
from ta.momentum import (
    ROCIndicator,
    RSIIndicator,
    StochasticOscillator,
    WilliamsRIndicator,
)
from ta.trend import MACD, ADXIndicator, CCIIndicator, EMAIndicator, SMAIndicator
from ta.volatility import AverageTrueRange, BollingerBands
from ta.volume import (
    AccDistIndexIndicator,
    ForceIndexIndicator,
    MFIIndicator,
    OnBalanceVolumeIndicator,
)

logger = structlog.get_logger(__name__)

TECHNICAL_FEATURE_COUNT = 40


def _compute_momentum(close: pd.Series, high: pd.Series, low: pd.Series) -> pd.DataFrame:
    """12 momentum indicators."""
    result = pd.DataFrame(index=close.index)

    for window in (7, 14, 21):
        result[f"rsi_{window}"] = RSIIndicator(close=close, window=window).rsi()

    stoch = StochasticOscillator(high=high, low=low, close=close, window=14, smooth_window=3)
    result["stoch_k_14"] = stoch.stoch()
    result["stoch_d_14"] = stoch.stoch_signal()

    macd = MACD(close=close, window_slow=26, window_fast=12, window_sign=9)
    result["macd_12_26"] = macd.macd()
    result["macd_signal_12_26"] = macd.macd_signal()
    result["macd_hist_12_26"] = macd.macd_diff()

    result["willr_14"] = WilliamsRIndicator(high=high, low=low, close=close, lbp=14).williams_r()

    for window in (10, 20):
        result[f"roc_{window}"] = ROCIndicator(close=close, window=window).roc()

    result["cci_20"] = CCIIndicator(high=high, low=low, close=close, window=20).cci()

    return result


def _compute_trend(close: pd.Series, high: pd.Series, low: pd.Series) -> pd.DataFrame:
    """10 trend indicators."""
    result = pd.DataFrame(index=close.index)

    for window in (20, 50, 200):
        sma = SMAIndicator(close=close, window=window).sma_indicator()
        result[f"sma_{window}"] = sma

    for window in (9, 21):
        result[f"ema_{window}"] = EMAIndicator(close=close, window=window).ema_indicator()

    sma_20 = result["sma_20"]
    sma_50 = result["sma_50"]
    result["price_to_sma_20"] = close / sma_20
    result["price_to_sma_50"] = close / sma_50

    if len(close) > 28:
        adx = ADXIndicator(high=high, low=low, close=close, window=14)
        result["adx_14"] = adx.adx()
        result["dip_14"] = adx.adx_pos()
        result["dim_14"] = adx.adx_neg()
    else:
        result["adx_14"] = np.nan
        result["dip_14"] = np.nan
        result["dim_14"] = np.nan

    return result


def _compute_volatility(
    close: pd.Series,
    high: pd.Series,
    low: pd.Series,
) -> pd.DataFrame:
    """8 volatility indicators."""
    result = pd.DataFrame(index=close.index)

    bb = BollingerBands(close=close, window=20, window_dev=2)
    result["bb_upper_20"] = bb.bollinger_hband()
    result["bb_lower_20"] = bb.bollinger_lband()
    result["bb_width_20"] = bb.bollinger_wband()
    result["bb_pct_20"] = bb.bollinger_pband()

    atr = AverageTrueRange(high=high, low=low, close=close, window=14)
    result["atr_14"] = atr.average_true_range()
    result["natr_14"] = (result["atr_14"] / close) * 100

    log_returns = np.log(close / close.shift(1))
    result["hvol_20"] = log_returns.rolling(20).std() * np.sqrt(252)
    result["hvol_60"] = log_returns.rolling(60).std() * np.sqrt(252)

    return result


def _compute_volume(
    close: pd.Series,
    high: pd.Series,
    low: pd.Series,
    volume: pd.Series,
) -> pd.DataFrame:
    """10 volume indicators."""
    result = pd.DataFrame(index=close.index)

    obv = OnBalanceVolumeIndicator(close=close, volume=volume).on_balance_volume()
    result["obv"] = obv
    result["obv_ema_20"] = EMAIndicator(close=obv, window=20).ema_indicator()

    vol_sma_20 = volume.rolling(20).mean()
    result["vol_sma_20"] = vol_sma_20
    result["vol_ratio_20"] = volume / vol_sma_20

    typical_price = (high + low + close) / 3
    tp_vol = typical_price * volume
    result["vwap_20"] = tp_vol.rolling(20).sum() / volume.rolling(20).sum()
    result["price_to_vwap_20"] = close / result["vwap_20"]

    result["mfi_14"] = MFIIndicator(
        high=high,
        low=low,
        close=close,
        volume=volume,
        window=14,
    ).money_flow_index()

    result["ad"] = AccDistIndexIndicator(
        high=high,
        low=low,
        close=close,
        volume=volume,
    ).acc_dist_index()

    ad_ema_3 = EMAIndicator(close=result["ad"], window=3).ema_indicator()
    ad_ema_10 = EMAIndicator(close=result["ad"], window=10).ema_indicator()
    result["cho_3_10"] = ad_ema_3 - ad_ema_10

    result["fi_13"] = ForceIndexIndicator(close=close, volume=volume, window=13).force_index()

    return result


def compute_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute 40 technical indicators from OHLCV data.

    Args:
        df: DataFrame with columns: open, high, low, close, volume.
            Index must be DatetimeIndex.

    Returns:
        DataFrame with 40 feature columns, same index as input.
    """
    close = df["close"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    volume = df["volume"].astype(float)

    momentum = _compute_momentum(close, high, low)
    trend = _compute_trend(close, high, low)
    volatility = _compute_volatility(close, high, low)
    vol_features = _compute_volume(close, high, low, volume)

    result = pd.concat([momentum, trend, volatility, vol_features], axis=1)

    logger.info(
        "technical_features_computed",
        n_features=len(result.columns),
        n_rows=len(result),
    )
    return result
