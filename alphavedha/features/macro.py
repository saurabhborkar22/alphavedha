"""Macro/market-wide features — 30 features.

Sources: yfinance (VIX, FX, commodities, crude oil, US futures),
institutional_flows table (FII/DII), universe stock prices (breadth),
and computed sector-relative metrics.
Column naming: macro_{indicator}.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import structlog

logger = structlog.get_logger(__name__)

MACRO_FEATURE_COUNT = 30

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
    "macro_crude_oil",
    "macro_crude_oil_change_5d",
    "macro_us_overnight_return",
    "macro_forex_reserves",
    "macro_forex_reserves_change",
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
    alt_data_df: pd.DataFrame | None = None,
    universe_prices: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Compute 30 macro features."""
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

    if alt_data_df is not None and not alt_data_df.empty:
        _compute_alt_data_features(result, alt_data_df, stock_df.index)
        _compute_energy_features(result, alt_data_df, stock_df.index)

    _compute_breadth_features(result, universe_prices, stock_df.index)

    for col in _ALL_MACRO_COLUMNS:
        if col not in result.columns:
            result[col] = np.nan

    logger.info(
        "macro_features_computed",
        n_features=len(result.columns),
        n_rows=len(result),
    )
    return result


def _compute_alt_data_features(
    result: pd.DataFrame,
    alt_data_df: pd.DataFrame,
    index: pd.DatetimeIndex,
) -> None:
    """Compute macro features from alternative data (PMI, breadth, etc.)."""
    alt = alt_data_df.copy()
    if "period_date" in alt.columns:
        alt["period_date"] = pd.to_datetime(alt["period_date"])

    for data_type, col_name in [
        ("pmi_manufacturing", "macro_pmi"),
    ]:
        subset = alt[alt["data_type"] == data_type] if "data_type" in alt.columns else pd.DataFrame()
        if not subset.empty:
            pmi_series = subset.set_index("period_date")["value"].sort_index()
            pmi_aligned = pmi_series.reindex(index, method="ffill")
            result[col_name] = pmi_aligned

            staleness = pd.Series(np.nan, index=index)
            for i, idx in enumerate(index):
                valid = pmi_series.index[pmi_series.index <= idx]
                if len(valid) > 0:
                    staleness.iloc[i] = (idx - valid[-1]).days
            result["macro_pmi_staleness_days"] = staleness


def _compute_breadth_features(
    result: pd.DataFrame,
    universe_prices: pd.DataFrame | None,
    index: pd.DatetimeIndex,
) -> None:
    """Compute market breadth features from universe stock prices.

    universe_prices: DataFrame with columns = symbols, index = dates, values = close prices

    macro_breadth_200sma: for each date, % of stocks with close > 200-day SMA
    macro_adv_dec_ratio: for each date, count(stocks with positive return) / count(stocks with negative return)
    macro_index_cpr: (high + low + close) / 3 for the index — uses the mean of universe as proxy
    macro_mktcap_flow: placeholder using FII flow differential if available
    """
    if universe_prices is None or universe_prices.empty:
        return

    aligned = universe_prices.reindex(index, method="ffill")

    sma_200 = aligned.rolling(200, min_periods=50).mean()
    above_sma = (aligned > sma_200).sum(axis=1)
    total_stocks = aligned.notna().sum(axis=1).replace(0, np.nan)
    result["macro_breadth_200sma"] = above_sma / total_stocks

    daily_returns = aligned.pct_change()
    advancing = (daily_returns > 0).sum(axis=1)
    declining = (daily_returns < 0).sum(axis=1).replace(0, np.nan)
    result["macro_adv_dec_ratio"] = advancing / declining

    if {"high", "low", "close"}.issubset(universe_prices.columns):
        high = universe_prices["high"].reindex(index, method="ffill")
        low = universe_prices["low"].reindex(index, method="ffill")
        close = universe_prices["close"].reindex(index, method="ffill")
        result["macro_index_cpr"] = (high + low + close) / 3.0
    else:
        mean_price = aligned.mean(axis=1)
        result["macro_index_cpr"] = mean_price

    if "macro_fii_net" in result.columns and "macro_dii_net" in result.columns:
        result["macro_mktcap_flow"] = result["macro_fii_net"] - result["macro_dii_net"]


def _compute_energy_features(
    result: pd.DataFrame,
    alt_data_df: pd.DataFrame,
    index: pd.DatetimeIndex,
) -> None:
    """Compute crude oil, US overnight, and forex reserves macro features."""
    alt = alt_data_df.copy()
    if "period_date" in alt.columns:
        alt["period_date"] = pd.to_datetime(alt["period_date"])

    if "data_type" not in alt.columns:
        return

    crude_subset = alt[alt["data_type"] == "crude_oil"]
    if not crude_subset.empty:
        crude_series = crude_subset.set_index("period_date")["value"].sort_index()
        crude_series = crude_series[~crude_series.index.duplicated(keep="last")]
        crude_aligned = crude_series.reindex(index, method="ffill")
        result["macro_crude_oil"] = crude_aligned
        result["macro_crude_oil_change_5d"] = crude_aligned.pct_change(5)

    us_subset = alt[alt["data_type"] == "us_overnight"]
    if not us_subset.empty:
        us_series = us_subset.set_index("period_date")["value"].sort_index()
        us_series = us_series[~us_series.index.duplicated(keep="last")]
        us_aligned = us_series.reindex(index, method="ffill")
        result["macro_us_overnight_return"] = us_aligned

    fx_subset = alt[alt["data_type"] == "forex_reserves"]
    if not fx_subset.empty:
        fx_series = fx_subset.set_index("period_date")["value"].sort_index()
        fx_series = fx_series[~fx_series.index.duplicated(keep="last")]
        fx_aligned = fx_series.reindex(index, method="ffill")
        result["macro_forex_reserves"] = fx_aligned
        result["macro_forex_reserves_change"] = fx_aligned.pct_change()
