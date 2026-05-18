"""Macro/market-wide features — 25 features.

Sources: yfinance (VIX, FX, commodities), institutional_flows table (FII/DII),
and computed sector-relative metrics.
Column naming: macro_{indicator}.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import structlog

logger = structlog.get_logger(__name__)

MACRO_FEATURE_COUNT = 25

MACRO_TICKERS = {
    "vix": "^INDIAVIX",
    "nifty": "^NSEI",
    "usdinr": "USDINR=X",
    "brent": "BZ=F",
    "gold": "GC=F",
    "us10y": "^TNX",
}

_ALL_MACRO_COLUMNS = [
    "macro_vix",
    "macro_vix_change_1d",
    "macro_vix_zscore_20d",
    "macro_nifty_ret_1d",
    "macro_nifty_ret_5d",
    "macro_usdinr",
    "macro_usdinr_change_1d",
    "macro_brent",
    "macro_brent_change_1d",
    "macro_gold",
    "macro_us10y",
    "macro_gsec_10y",
    "macro_gsec_change_1d",
    "macro_fii_net",
    "macro_fii_cum_5d",
    "macro_dii_net",
    "macro_dii_cum_5d",
    "macro_sector_ret_1d",
    "macro_sector_rel_ret_1d",
    "macro_pmi",
    "macro_pmi_staleness_days",
    "macro_breadth_200sma",
    "macro_adv_dec_ratio",
    "macro_index_cpr",
    "macro_mktcap_flow",
]


async def load_fii_dii_for_features(start: str, end: str) -> pd.DataFrame:
    """Load FII/DII data from DB and pivot into feature-ready format.

    Returns DataFrame indexed by date with columns: fii_net, dii_net.
    """
    from datetime import date as date_type

    from alphavedha.data.store import load_fii_dii

    start_dt = date_type.fromisoformat(start)
    end_dt = date_type.fromisoformat(end)
    raw = await load_fii_dii(start_dt, end_dt)

    if raw.empty:
        return pd.DataFrame()

    pivoted = raw.pivot_table(
        index="date", columns="category", values="net_value", aggfunc="first"
    )

    result = pd.DataFrame(index=pivoted.index)
    result["fii_net"] = pivoted.get("FII", pivoted.get("FPI", np.nan))
    result["dii_net"] = pivoted.get("DII", np.nan)
    result.index = pd.to_datetime(result.index)
    return result


def fetch_macro_data(start: str, end: str) -> pd.DataFrame:
    """Fetch market-wide macro data via yfinance."""
    import yfinance as yf

    result = pd.DataFrame()
    for name, ticker in MACRO_TICKERS.items():
        try:
            data = yf.download(ticker, start=start, end=end, progress=False)
            if not data.empty:
                if isinstance(data.columns, pd.MultiIndex):
                    data.columns = data.columns.get_level_values(0)
                result[name] = data["Close"]
        except (KeyError, ValueError, ConnectionError):
            logger.warning("macro_fetch_failed", ticker=ticker)
    return result


def _compute_market_features(
    result: pd.DataFrame,
    macro_df: pd.DataFrame,
    index: pd.DatetimeIndex,
) -> None:
    """Compute VIX, Nifty, FX, commodity features from aligned macro data."""
    macro_aligned = macro_df.reindex(index, method="ffill")

    if "vix" in macro_aligned.columns:
        vix = macro_aligned["vix"]
        result["macro_vix"] = vix
        result["macro_vix_change_1d"] = vix.pct_change()
        vix_mean = vix.rolling(20, min_periods=5).mean()
        vix_std = vix.rolling(20, min_periods=5).std()
        result["macro_vix_zscore_20d"] = (vix - vix_mean) / vix_std.replace(0, np.nan)

    if "nifty" in macro_aligned.columns:
        nifty = macro_aligned["nifty"]
        result["macro_nifty_ret_1d"] = np.log(nifty / nifty.shift(1))
        result["macro_nifty_ret_5d"] = np.log(nifty / nifty.shift(5))

    if "usdinr" in macro_aligned.columns:
        usdinr = macro_aligned["usdinr"]
        result["macro_usdinr"] = usdinr
        result["macro_usdinr_change_1d"] = usdinr.pct_change()

    if "brent" in macro_aligned.columns:
        brent = macro_aligned["brent"]
        result["macro_brent"] = brent
        result["macro_brent_change_1d"] = brent.pct_change()

    if "gold" in macro_aligned.columns:
        result["macro_gold"] = macro_aligned["gold"]

    if "us10y" in macro_aligned.columns:
        result["macro_us10y"] = macro_aligned["us10y"]

    result["macro_gsec_10y"] = 7.0
    result["macro_gsec_change_1d"] = 0.0


def _compute_flow_features(
    result: pd.DataFrame,
    fii_dii_df: pd.DataFrame,
    index: pd.DatetimeIndex,
) -> None:
    """Compute FII/DII flow features."""
    fii_aligned = fii_dii_df.reindex(index, method="ffill")

    for prefix in ("fii", "dii"):
        col = f"{prefix}_net"
        if col in fii_aligned.columns:
            result[f"macro_{col}"] = fii_aligned[col]
            result[f"macro_{prefix}_cum_5d"] = fii_aligned[col].rolling(5, min_periods=1).sum()
        else:
            result[f"macro_{col}"] = np.nan
            result[f"macro_{prefix}_cum_5d"] = np.nan


def compute_macro_features(
    stock_df: pd.DataFrame,
    macro_df: pd.DataFrame | None = None,
    fii_dii_df: pd.DataFrame | None = None,
    sector_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Compute 25 macro features."""
    result = pd.DataFrame(index=stock_df.index)
    stock_close = stock_df["close"].astype(float)

    if macro_df is not None and not macro_df.empty:
        _compute_market_features(result, macro_df, stock_df.index)

    if fii_dii_df is not None and not fii_dii_df.empty:
        _compute_flow_features(result, fii_dii_df, stock_df.index)

    if sector_df is not None and not sector_df.empty:
        sector_aligned = sector_df.reindex(stock_df.index, method="ffill")
        sector_close = sector_aligned["close"].astype(float)
        sector_ret = np.log(sector_close / sector_close.shift(1))
        stock_ret = np.log(stock_close / stock_close.shift(1))
        result["macro_sector_ret_1d"] = sector_ret
        result["macro_sector_rel_ret_1d"] = stock_ret - sector_ret

    for col in _ALL_MACRO_COLUMNS:
        if col not in result.columns:
            result[col] = np.nan

    logger.info(
        "macro_features_computed",
        n_features=len(result.columns),
        n_rows=len(result),
    )
    return result
