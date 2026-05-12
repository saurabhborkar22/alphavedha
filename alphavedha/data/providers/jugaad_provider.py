"""jugaad-data provider — primary for NSE daily data, includes delivery %."""

from __future__ import annotations

import asyncio
from datetime import date

import pandas as pd
import structlog
from jugaad_data.nse import stock_df

from alphavedha.config import get_config
from alphavedha.data.providers.base import (
    FetchResult,
    RateLimiter,
    fetch_with_retry,
    validate_ohlcv,
)

logger = structlog.get_logger(__name__)


def _strip_suffix(symbol: str) -> str:
    for suffix in (".NS", ".BO"):
        if symbol.endswith(suffix):
            return symbol[: -len(suffix)]
    return symbol


def _download_jugaad(symbol: str, from_date: date, to_date: date) -> pd.DataFrame:
    """Sync download — runs inside asyncio.to_thread."""
    nse_symbol = _strip_suffix(symbol)
    df = stock_df(symbol=nse_symbol, from_date=from_date, to_date=to_date, series="EQ")

    if df.empty:
        return pd.DataFrame()

    rename = {
        "DATE": "date",
        "OPEN": "open",
        "HIGH": "high",
        "LOW": "low",
        "CLOSE": "close",
        "LTP": "ltp",
        "PREV. CLOSE": "prev_close",
        "TOTAL TRADED QUANTITY": "volume",
        "TOTAL TRADED VALUE": "traded_value",
        "52 WEEK HIGH": "high_52w",
        "52 WEEK LOW": "low_52w",
        "DELIVERABLE QTY": "deliverable_qty",
        "% DELIVERBLE": "delivery_pct",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date").sort_index()

    df["adj_close"] = df["close"]

    if "delivery_pct" in df.columns:
        df["delivery_pct"] = pd.to_numeric(df["delivery_pct"], errors="coerce") / 100.0

    return df


class JugaadProvider:
    """Fetch OHLCV + delivery data from NSE via jugaad-data."""

    def __init__(self) -> None:
        cfg = get_config()
        rl_cfg = cfg.data.rate_limits.get("nse")
        rps = rl_cfg.requests_per_second if rl_cfg else 0.5
        self._rate_limiter = RateLimiter(requests_per_second=rps)

    @property
    def name(self) -> str:
        return "jugaad"

    async def fetch_ohlcv(self, symbol: str, start: date, end: date) -> pd.DataFrame:
        await self._rate_limiter.acquire()

        df = await fetch_with_retry(
            _download_jugaad,
            symbol,
            start,
            end,
            provider_name=self.name,
        )

        delivery_pct = df["delivery_pct"].copy() if "delivery_pct" in df.columns else None

        df = validate_ohlcv(df, symbol, self.name)

        if delivery_pct is not None and not df.empty:
            df["delivery_pct"] = delivery_pct.reindex(df.index)

        logger.info(
            "jugaad_fetched",
            symbol=symbol,
            rows=len(df),
            start=str(start),
            end=str(end),
        )
        return df

    async def fetch_bulk(
        self, symbols: list[str], start: date, end: date
    ) -> dict[str, FetchResult]:
        results: dict[str, FetchResult] = {}

        for sym in symbols:
            try:
                df = await self.fetch_ohlcv(sym, start, end)
                results[sym] = FetchResult(symbol=sym, df=df, provider=self.name)
            except Exception as e:
                logger.error("jugaad_bulk_error", symbol=sym, error=str(e))
                results[sym] = FetchResult(
                    symbol=sym,
                    df=pd.DataFrame(),
                    provider=self.name,
                    had_errors=True,
                    error_message=str(e),
                )

        logger.info(
            "jugaad_bulk_complete",
            total=len(symbols),
            ok=sum(1 for r in results.values() if not r.had_errors),
        )
        return results

    async def health_check(self) -> bool:
        try:
            today = date.today()
            start = date(today.year, today.month, 1)
            df = await asyncio.to_thread(
                stock_df, symbol="TCS", from_date=start, to_date=today, series="EQ"
            )
            return not df.empty
        except Exception:
            return False
