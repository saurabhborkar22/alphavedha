"""Live intraday data polling — polls yfinance every 2 minutes during market hours.

Writes to the IntradayOHLCV table (upsert in-place, preserving day high/low) and
invalidates Redis prediction cache keys every 5th tick so predictions stay fresh.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any
from zoneinfo import ZoneInfo

import structlog
import yfinance as yf
from sqlalchemy import func
from sqlalchemy.dialects.postgresql import insert as pg_insert

from alphavedha.data.models import IntradayOHLCV

logger = structlog.get_logger(__name__)

IST = ZoneInfo("Asia/Kolkata")
_MARKET_OPEN_H, _MARKET_OPEN_M = 9, 15
_MARKET_CLOSE_H, _MARKET_CLOSE_M = 15, 30

POLL_INTERVAL_SECONDS: float = 120.0
CACHE_INVALIDATE_EVERY_N_TICKS = 5


def is_market_open() -> bool:
    """Return True if the Indian market is currently open (9:15-15:30 IST, Mon-Fri)."""
    now = datetime.now(IST)
    if now.weekday() >= 5:
        return False
    market_open = now.replace(hour=_MARKET_OPEN_H, minute=_MARKET_OPEN_M, second=0, microsecond=0)
    market_close = now.replace(
        hour=_MARKET_CLOSE_H, minute=_MARKET_CLOSE_M, second=0, microsecond=0
    )
    return market_open <= now <= market_close


@dataclass
class PollResult:
    symbol: str
    polled_at: datetime
    success: bool
    last_price: float | None = None
    error: str | None = None


def _fetch_fast_info(symbol: str) -> dict[str, float | int]:
    """Fetch yfinance fast_info fields synchronously (runs in thread pool)."""
    ticker = yf.Ticker(symbol)
    info = ticker.fast_info
    return {
        "open": float(info.open or 0),
        "high": float(info.day_high or 0),
        "low": float(info.day_low or 0),
        "last_price": float(info.last_price or 0),
        "volume": int(info.day_volume or 0),
    }


class LiveDataPoller:
    """Polls yfinance fast_info for all tracked symbols and upserts IntradayOHLCV.

    Preserves running day high/low across ticks via GREATEST/LEAST on conflict.
    Every ``CACHE_INVALIDATE_EVERY_N_TICKS`` ticks, deletes Redis prediction
    cache keys (pattern ``predict:{symbol}:*``) so stale predictions are evicted.
    """

    def __init__(
        self,
        symbols: list[str],
        session_factory: Any,
        redis_client: Any | None = None,
        poll_interval: float = POLL_INTERVAL_SECONDS,
    ) -> None:
        self._symbols = symbols
        self._session_factory = session_factory
        self._redis_client = redis_client
        self._poll_interval = poll_interval
        self._tick_count = 0

    @property
    def tick_count(self) -> int:
        return self._tick_count

    async def poll_once(self) -> list[PollResult]:
        """Poll all symbols once, upsert to DB, and maybe invalidate cache."""
        now = datetime.now(IST)
        today = now.date()
        results: list[PollResult] = []

        async with self._session_factory() as session:
            for symbol in self._symbols:
                result = await self._poll_symbol(session, symbol, today, now)
                results.append(result)
            await session.commit()

        self._tick_count += 1
        if self._tick_count % CACHE_INVALIDATE_EVERY_N_TICKS == 0:
            await self._invalidate_cache()

        successes = sum(1 for r in results if r.success)
        logger.info(
            "live_poll_complete",
            tick=self._tick_count,
            symbols=len(self._symbols),
            successes=successes,
        )
        return results

    async def _poll_symbol(
        self,
        session: Any,
        symbol: str,
        today: date,
        polled_at: datetime,
    ) -> PollResult:
        try:
            raw = await asyncio.to_thread(_fetch_fast_info, symbol)

            if raw["last_price"] == 0:
                return PollResult(
                    symbol=symbol,
                    polled_at=polled_at,
                    success=False,
                    error="zero last_price from fast_info",
                )

            stmt = pg_insert(IntradayOHLCV).values(
                symbol=symbol,
                date=today,
                open=raw["open"],
                high=raw["high"],
                low=raw["low"],
                last_price=raw["last_price"],
                volume=raw["volume"],
                tick_count=1,
                last_updated=polled_at.replace(tzinfo=None),
            )
            t = IntradayOHLCV.__table__
            stmt = stmt.on_conflict_do_update(
                index_elements=["symbol", "date"],
                set_={
                    "high": func.greatest(t.c.high, stmt.excluded.high),
                    "low": func.least(t.c.low, stmt.excluded.low),
                    "last_price": stmt.excluded.last_price,
                    "volume": stmt.excluded.volume,
                    "tick_count": t.c.tick_count + 1,
                    "last_updated": stmt.excluded.last_updated,
                },
            )
            await session.execute(stmt)

            return PollResult(
                symbol=symbol,
                polled_at=polled_at,
                success=True,
                last_price=raw["last_price"],
            )
        except Exception as exc:
            logger.warning("live_poll_symbol_failed", symbol=symbol, error=str(exc))
            return PollResult(
                symbol=symbol,
                polled_at=polled_at,
                success=False,
                error=str(exc),
            )

    async def _invalidate_cache(self) -> None:
        """Delete Redis prediction cache entries for all polled symbols."""
        if self._redis_client is None:
            return
        try:
            for symbol in self._symbols:
                keys = await self._redis_client.keys(f"predict:{symbol}:*")
                if keys:
                    await self._redis_client.delete(*keys)
            logger.debug("live_cache_invalidated", symbols=len(self._symbols))
        except Exception as exc:
            logger.warning("live_cache_invalidate_failed", error=str(exc))

    async def run_until_close(self) -> None:
        """Poll in a loop until market closes or the task is cancelled."""
        logger.info("live_poller_started", symbols=len(self._symbols))
        while is_market_open():
            await self.poll_once()
            await asyncio.sleep(self._poll_interval)
        logger.info("live_poller_stopped_market_closed")
