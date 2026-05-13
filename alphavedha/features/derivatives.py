"""Derivatives features — 20 F&O features.

Source: derivatives_data table (futures OI, options chain JSON).
Uses Black-Scholes IV via scipy.
Column naming: deriv_{indicator}.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import structlog

logger = structlog.get_logger(__name__)

DERIVATIVES_FEATURE_COUNT = 20

RISK_FREE_RATE = 0.065

_ALL_DERIV_COLUMNS = [
    "deriv_futures_oi",
    "deriv_futures_oi_change",
    "deriv_futures_premium",
    "deriv_atm_iv",
    "deriv_iv_rank",
    "deriv_iv_pctile",
    "deriv_pcr_oi",
    "deriv_pcr_vol",
    "deriv_max_pain",
    "deriv_dist_max_pain",
    "deriv_fii_futures_oi",
    "deriv_fii_options_oi",
    "deriv_pro_futures_net",
    "deriv_retail_futures_net",
    "deriv_oi_buildup",
    "deriv_oi_unwind",
    "deriv_short_cover",
    "deriv_short_build",
    "deriv_gex",
    "deriv_delta_oi",
]


def _bs_price(
    s: float,
    k: float,
    t: float,
    r: float,
    sigma: float,
    option_type: str = "call",
) -> float:
    """Black-Scholes option price."""
    from scipy.stats import norm

    if t <= 0 or sigma <= 0:
        return 0.0
    d1 = (np.log(s / k) + (r + 0.5 * sigma**2) * t) / (sigma * np.sqrt(t))
    d2 = d1 - sigma * np.sqrt(t)
    if option_type == "call":
        return float(s * norm.cdf(d1) - k * np.exp(-r * t) * norm.cdf(d2))
    return float(k * np.exp(-r * t) * norm.cdf(-d2) - s * norm.cdf(-d1))


def implied_volatility(
    market_price: float,
    s: float,
    k: float,
    t: float,
    r: float = RISK_FREE_RATE,
    option_type: str = "call",
) -> float:
    """Compute implied volatility using Brent's method. Returns NaN if no solution."""
    from scipy.optimize import brentq

    if market_price <= 0 or t <= 0:
        return np.nan
    try:
        return float(
            brentq(
                lambda sigma: _bs_price(s, k, t, r, sigma, option_type) - market_price,
                0.001,
                5.0,
                xtol=1e-6,
            )
        )
    except (ValueError, RuntimeError):
        return np.nan


def _extract_options_features(options_json: dict | None, spot: float) -> dict[str, float]:
    """Extract features from options chain JSON snapshot."""
    defaults = {
        "atm_iv": np.nan,
        "pcr_oi": np.nan,
        "pcr_vol": np.nan,
        "max_pain": np.nan,
        "total_call_oi": np.nan,
        "total_put_oi": np.nan,
    }
    if not options_json or not isinstance(options_json, dict):
        return defaults

    chain = options_json.get("chain", [])
    if not chain:
        return defaults

    total_call_oi = 0.0
    total_put_oi = 0.0
    total_call_vol = 0.0
    total_put_vol = 0.0
    atm_iv = np.nan
    min_strike_diff = float("inf")
    pain_strikes: dict[float, float] = {}

    for entry in chain:
        strike = float(entry.get("strike", 0))
        c_oi = float(entry.get("call_oi", 0))
        p_oi = float(entry.get("put_oi", 0))
        total_call_oi += c_oi
        total_put_oi += p_oi
        total_call_vol += float(entry.get("call_vol", 0))
        total_put_vol += float(entry.get("put_vol", 0))

        c_iv = entry.get("call_iv")
        strike_diff = abs(strike - spot)
        if strike_diff < min_strike_diff and c_iv is not None:
            min_strike_diff = strike_diff
            atm_iv = float(c_iv)

        pain_strikes[strike] = c_oi + p_oi

    pcr_oi = total_put_oi / total_call_oi if total_call_oi > 0 else np.nan
    pcr_vol = total_put_vol / total_call_vol if total_call_vol > 0 else np.nan
    max_pain = max(pain_strikes, key=pain_strikes.get) if pain_strikes else np.nan

    return {
        "atm_iv": atm_iv,
        "pcr_oi": pcr_oi,
        "pcr_vol": pcr_vol,
        "max_pain": max_pain,
        "total_call_oi": total_call_oi,
        "total_put_oi": total_put_oi,
    }


def _compute_futures_features(
    result: pd.DataFrame,
    deriv_aligned: pd.DataFrame,
    close: pd.Series,
) -> None:
    """Compute futures OI, OI change, and premium."""
    if "futures_oi" in deriv_aligned.columns:
        fut_oi = deriv_aligned["futures_oi"].astype(float)
        result["deriv_futures_oi"] = fut_oi
        result["deriv_futures_oi_change"] = fut_oi.pct_change()
    else:
        result["deriv_futures_oi"] = np.nan
        result["deriv_futures_oi_change"] = np.nan

    if "futures_price" in deriv_aligned.columns:
        fut_price = deriv_aligned["futures_price"].astype(float)
        result["deriv_futures_premium"] = (fut_price - close) / close * 100
    else:
        result["deriv_futures_premium"] = np.nan


def _compute_oi_interpretation(
    result: pd.DataFrame,
    deriv_aligned: pd.DataFrame,
    close: pd.Series,
) -> None:
    """Compute OI buildup/unwinding/short cover/short build flags."""
    if "futures_oi" not in deriv_aligned.columns:
        for col in (
            "deriv_oi_buildup",
            "deriv_oi_unwind",
            "deriv_short_cover",
            "deriv_short_build",
        ):
            result[col] = np.nan
        return

    oi_change = deriv_aligned["futures_oi"].astype(float).diff()
    price_change = close.diff()
    result["deriv_oi_buildup"] = ((oi_change > 0) & (price_change > 0)).astype(int)
    result["deriv_oi_unwind"] = ((oi_change < 0) & (price_change < 0)).astype(int)
    result["deriv_short_cover"] = ((oi_change < 0) & (price_change > 0)).astype(int)
    result["deriv_short_build"] = ((oi_change > 0) & (price_change < 0)).astype(int)


def compute_derivatives_features(
    stock_df: pd.DataFrame,
    deriv_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Compute 20 derivatives features."""
    result = pd.DataFrame(index=stock_df.index)
    close = stock_df["close"].astype(float)

    if deriv_df is None or deriv_df.empty:
        logger.warning("derivatives_no_data", msg="No derivatives data, returning NaN")
        for col in _ALL_DERIV_COLUMNS:
            result[col] = np.nan
        return result

    deriv_aligned = deriv_df.reindex(stock_df.index, method="ffill")

    _compute_futures_features(result, deriv_aligned, close)

    opt_features_list = [
        _extract_options_features(
            deriv_aligned.loc[idx_val, "options_data_json"]
            if idx_val in deriv_aligned.index and "options_data_json" in deriv_aligned.columns
            else None,
            close.loc[idx_val],
        )
        for idx_val in stock_df.index
    ]
    opt_df = pd.DataFrame(opt_features_list, index=stock_df.index)
    result["deriv_atm_iv"] = opt_df["atm_iv"]
    result["deriv_pcr_oi"] = opt_df["pcr_oi"]
    result["deriv_pcr_vol"] = opt_df["pcr_vol"]
    result["deriv_max_pain"] = opt_df["max_pain"]
    result["deriv_dist_max_pain"] = (close - opt_df["max_pain"]) / close * 100

    iv = result["deriv_atm_iv"]
    iv_252_min = iv.rolling(252, min_periods=20).min()
    iv_252_max = iv.rolling(252, min_periods=20).max()
    result["deriv_iv_rank"] = (iv - iv_252_min) / (iv_252_max - iv_252_min).replace(0, np.nan)
    result["deriv_iv_pctile"] = iv.rolling(252, min_periods=20).apply(
        lambda x: (x < x.iloc[-1]).sum() / len(x),
        raw=False,
    )

    result["deriv_fii_futures_oi"] = np.nan
    result["deriv_fii_options_oi"] = np.nan
    result["deriv_pro_futures_net"] = np.nan
    result["deriv_retail_futures_net"] = np.nan

    _compute_oi_interpretation(result, deriv_aligned, close)

    result["deriv_gex"] = np.nan
    result["deriv_delta_oi"] = np.nan

    for col in _ALL_DERIV_COLUMNS:
        if col not in result.columns:
            result[col] = np.nan

    logger.info("derivatives_features_computed", n_features=len(result.columns))
    return result
