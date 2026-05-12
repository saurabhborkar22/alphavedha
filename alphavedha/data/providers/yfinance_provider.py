"""yfinance data provider — primary for historical backfill, fallback for daily data."""

from __future__ import annotations

import asyncio
from datetime import date

import pandas as pd
import structlog
import yfinance as yf

from alphavedha.config import get_config
from alphavedha.data.providers.base import (
    FetchResult,
    RateLimiter,
    fetch_with_retry,
    validate_ohlcv,
)

logger = structlog.get_logger(__name__)


def _ensure_ns_suffix(symbol: str) -> str:
    if not symbol.endswith(".NS") and not symbol.endswith(".BO"):
        return f"{symbol}.NS"
    return symbol


def _download_single(symbol: str, start: str, end: str) -> pd.DataFrame:
    """Sync download — runs inside asyncio.to_thread."""
    ticker = yf.Ticker(symbol)
    df = ticker.history(start=start, end=end, auto_adjust=False)
    if df.empty:
        return pd.DataFrame()

    df.index = pd.to_datetime(df.index)
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    df.index.name = "date"

    rename = {
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Adj Close": "adj_close",
        "Volume": "volume",
    }
    df = df.rename(columns=rename)

    keep = [c for c in ["open", "high", "low", "close", "adj_close", "volume"] if c in df.columns]
    return df[keep]


class YFinanceProvider:
    """Fetch OHLCV data from Yahoo Finance via yfinance."""

    def __init__(self) -> None:
        cfg = get_config()
        rl_cfg = cfg.data.rate_limits.get("yfinance")
        rps = rl_cfg.requests_per_second if rl_cfg else 2.0
        self._rate_limiter = RateLimiter(requests_per_second=rps)

    @property
    def name(self) -> str:
        return "yfinance"

    async def fetch_ohlcv(self, symbol: str, start: date, end: date) -> pd.DataFrame:
        yf_symbol = _ensure_ns_suffix(symbol)
        await self._rate_limiter.acquire()

        df = await fetch_with_retry(
            _download_single,
            yf_symbol,
            start.isoformat(),
            end.isoformat(),
            provider_name=self.name,
        )
        df = validate_ohlcv(df, symbol, self.name)

        logger.info(
            "yfinance_fetched",
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
        sem = asyncio.Semaphore(5)

        async def _fetch_one(sym: str) -> None:
            async with sem:
                try:
                    df = await self.fetch_ohlcv(sym, start, end)
                    results[sym] = FetchResult(symbol=sym, df=df, provider=self.name)
                except Exception as e:
                    logger.error("yfinance_bulk_error", symbol=sym, error=str(e))
                    results[sym] = FetchResult(
                        symbol=sym,
                        df=pd.DataFrame(),
                        provider=self.name,
                        had_errors=True,
                        error_message=str(e),
                    )

        await asyncio.gather(*[_fetch_one(s) for s in symbols])
        logger.info(
            "yfinance_bulk_complete",
            total=len(symbols),
            ok=sum(1 for r in results.values() if not r.had_errors),
        )
        return results

    async def health_check(self) -> bool:
        try:
            df = await asyncio.to_thread(lambda: yf.Ticker("TCS.NS").history(period="1d"))
            return not df.empty
        except Exception:
            return False
