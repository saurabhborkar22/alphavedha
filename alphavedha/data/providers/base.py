"""Data provider protocol and shared utilities (rate limiter, retry, result types)."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from datetime import date
from typing import Protocol, runtime_checkable

import pandas as pd
import structlog

from alphavedha.exceptions import DataProviderError

logger = structlog.get_logger(__name__)

OHLCV_COLUMNS = ["open", "high", "low", "close", "adj_close", "volume"]


@dataclass
class FetchResult:
    """Result of a data fetch operation."""

    symbol: str
    df: pd.DataFrame
    provider: str
    rows_fetched: int = 0
    had_errors: bool = False
    error_message: str | None = None

    def __post_init__(self) -> None:
        self.rows_fetched = len(self.df)


@runtime_checkable
class DataProvider(Protocol):
    """Protocol that all data providers must implement."""

    @property
    def name(self) -> str: ...

    async def fetch_ohlcv(self, symbol: str, start: date, end: date) -> pd.DataFrame: ...

    async def fetch_bulk(
        self, symbols: list[str], start: date, end: date
    ) -> dict[str, FetchResult]: ...

    async def health_check(self) -> bool: ...


class RateLimiter:
    """Token-bucket rate limiter — works in-memory (no Redis required)."""

    def __init__(
        self,
        requests_per_second: float | None = None,
        requests_per_minute: float | None = None,
    ) -> None:
        if requests_per_second:
            self._min_interval = 1.0 / requests_per_second
        elif requests_per_minute:
            self._min_interval = 60.0 / requests_per_minute
        else:
            self._min_interval = 0.0
        self._last_request: float = 0.0
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        if self._min_interval <= 0:
            return
        async with self._lock:
            now = time.monotonic()
            wait = self._min_interval - (now - self._last_request)
            if wait > 0:
                await asyncio.sleep(wait)
            self._last_request = time.monotonic()


async def fetch_with_retry(
    func: object,
    *args: object,
    max_retries: int = 3,
    backoff_factor: float = 2.0,
    provider_name: str = "unknown",
) -> pd.DataFrame:
    """Retry a sync fetch function with exponential backoff, running in a thread."""
    last_error: Exception | None = None
    for attempt in range(max_retries):
        try:
            result = await asyncio.to_thread(func, *args)  # type: ignore[arg-type]
            return result  # type: ignore[return-value]
        except Exception as e:
            last_error = e
            wait = backoff_factor**attempt
            logger.warning(
                "provider_fetch_retry",
                provider=provider_name,
                attempt=attempt + 1,
                max_retries=max_retries,
                wait_seconds=wait,
                error=str(e),
            )
            if attempt < max_retries - 1:
                await asyncio.sleep(wait)

    raise DataProviderError(f"[{provider_name}] Failed after {max_retries} retries: {last_error}")


def validate_ohlcv(df: pd.DataFrame, symbol: str, provider: str) -> pd.DataFrame:
    """Validate and normalize OHLCV DataFrame to standard column names."""
    if df.empty:
        logger.warning("provider_empty_result", symbol=symbol, provider=provider)
        return df

    df = df.copy()

    column_map: dict[str, str] = {}
    for col in df.columns:
        lower = str(col).lower().replace(" ", "_")
        if lower in ("adj_close", "adj close", "adjusted_close", "adjclose"):
            column_map[col] = "adj_close"
        elif lower in OHLCV_COLUMNS:
            column_map[col] = lower

    df = df.rename(columns=column_map)

    if "adj_close" not in df.columns and "close" in df.columns:
        df["adj_close"] = df["close"]

    missing = [c for c in OHLCV_COLUMNS if c not in df.columns]
    if missing:
        raise DataProviderError(
            f"[{provider}] {symbol}: missing columns {missing}. Got: {list(df.columns)}"
        )

    df = df[OHLCV_COLUMNS].copy()
    df = df.apply(pd.to_numeric, errors="coerce")

    bad_rows = df[OHLCV_COLUMNS[:4]].le(0).any(axis=1)
    if bad_rows.any():
        n_bad = int(bad_rows.sum())
        logger.warning(
            "provider_negative_prices_dropped",
            symbol=symbol,
            provider=provider,
            count=n_bad,
        )
        df = df[~bad_rows]

    return df
