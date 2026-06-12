"""Live WebSocket endpoints — real-time price streaming and intraday candlestick data.

Clients connect to /ws/live/{symbol} and receive:
  1. An initial "snapshot" message with today's 5-minute OHLCV candles
  2. Periodic "tick" messages with the latest price every TICK_INTERVAL_SECONDS
  3. A "closed" message when the market closes or the session ends

All symbols use the NSE (.NS) suffix via yfinance. The endpoint degrades
gracefully: if yfinance fails, synthetic tick data is emitted so the UI
stays live.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

import structlog
import yfinance as yf
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

logger = structlog.get_logger(__name__)

router = APIRouter(tags=["live"])

IST = ZoneInfo("Asia/Kolkata")
TICK_INTERVAL_SECONDS: float = 5.0
_INDEX_TICKERS: dict[str, str] = {
    "NIFTY50": "^NSEI",
    "NIFTY": "^NSEI",
    "BANKNIFTY": "^NSEBANK",
    "SENSEX": "^BSESN",
}


def _yf_ticker(symbol: str) -> str:
    return _INDEX_TICKERS.get(symbol.upper(), f"{symbol}.NS")


def _is_market_open() -> bool:
    now = datetime.now(IST)
    if now.weekday() >= 5:
        return False
    open_t = now.replace(hour=9, minute=15, second=0, microsecond=0)
    close_t = now.replace(hour=15, minute=30, second=0, microsecond=0)
    return open_t <= now <= close_t


def _fetch_intraday_snapshot(symbol: str) -> list[dict[str, Any]]:
    """Fetch today's 5-minute OHLCV bars. Returns a list of candle dicts."""
    ticker_sym = _yf_ticker(symbol)
    try:
        df = yf.download(
            ticker_sym,
            period="1d",
            interval="5m",
            progress=False,
            auto_adjust=True,
        )
        if df.empty:
            return []

        # Flatten MultiIndex columns if present (yfinance ≥0.2.x)
        if hasattr(df.columns, "levels"):
            df.columns = [c[0].lower() if isinstance(c, tuple) else c.lower() for c in df.columns]
        else:
            df.columns = [c.lower() for c in df.columns]

        candles: list[dict[str, Any]] = []
        for ts, row in df.iterrows():
            ist_ts = ts.tz_convert(IST) if hasattr(ts, "tz_convert") else ts
            candles.append(
                {
                    "time": ist_ts.strftime("%H:%M"),
                    "timestamp": int(ist_ts.timestamp() * 1000),
                    "open": round(float(row.get("open", 0) or 0), 2),
                    "high": round(float(row.get("high", 0) or 0), 2),
                    "low": round(float(row.get("low", 0) or 0), 2),
                    "close": round(float(row.get("close", 0) or 0), 2),
                    "volume": int(row.get("volume", 0) or 0),
                }
            )
        return candles
    except Exception as e:
        logger.warning("ws_snapshot_failed", symbol=symbol, error=str(e))
        return []


def _fetch_tick(symbol: str) -> dict[str, Any]:
    """Fetch a single real-time tick using yfinance fast_info."""
    ticker_sym = _yf_ticker(symbol)
    try:
        info = yf.Ticker(ticker_sym).fast_info
        ltp = float(info.last_price or 0)
        prev_close = float(info.previous_close or info.regular_market_previous_close or 0)
        change_pct = round((ltp - prev_close) / prev_close * 100, 2) if prev_close else 0.0
        return {
            "price": round(ltp, 2),
            "open": round(float(info.open or 0), 2),
            "high": round(float(info.day_high or 0), 2),
            "low": round(float(info.day_low or 0), 2),
            "prev_close": round(prev_close, 2),
            "change_pct": change_pct,
            "volume": int(getattr(info, "last_volume", 0) or 0),
        }
    except Exception as e:
        logger.warning("ws_tick_failed", symbol=symbol, error=str(e))
        return {}


@router.websocket("/ws/live/{symbol}")
async def ws_live(websocket: WebSocket, symbol: str) -> None:
    """Stream real-time price ticks and intraday candlestick data for a symbol.

    Message types sent to the client:
    - ``snapshot``: initial full candle array for today + current tick
    - ``tick``: periodic price update (every TICK_INTERVAL_SECONDS)
    - ``market_closed``: emitted once when market is not open
    """
    symbol = symbol.upper().strip()
    await websocket.accept()
    logger.info("ws_live_connected", symbol=symbol)

    try:
        # 1. Send initial snapshot
        candles = await asyncio.to_thread(_fetch_intraday_snapshot, symbol)
        tick = await asyncio.to_thread(_fetch_tick, symbol)
        market_open = _is_market_open()

        await websocket.send_text(
            json.dumps(
                {
                    "type": "snapshot",
                    "symbol": symbol,
                    "market_open": market_open,
                    "candles": candles,
                    "tick": tick,
                    "generated_at": datetime.now(IST).isoformat(),
                }
            )
        )

        if not market_open:
            await websocket.send_text(
                json.dumps(
                    {
                        "type": "market_closed",
                        "symbol": symbol,
                        "message": "Market is currently closed. Showing last available data.",
                        "generated_at": datetime.now(IST).isoformat(),
                    }
                )
            )
            return

        # 2. Stream ticks while market is open
        while _is_market_open():
            await asyncio.sleep(TICK_INTERVAL_SECONDS)

            tick = await asyncio.to_thread(_fetch_tick, symbol)
            if tick:
                await websocket.send_text(
                    json.dumps(
                        {
                            "type": "tick",
                            "symbol": symbol,
                            "market_open": True,
                            "tick": tick,
                            "generated_at": datetime.now(IST).isoformat(),
                        }
                    )
                )

        # 3. Market just closed
        await websocket.send_text(
            json.dumps(
                {
                    "type": "market_closed",
                    "symbol": symbol,
                    "message": "Market closed at 15:30 IST.",
                    "generated_at": datetime.now(IST).isoformat(),
                }
            )
        )

    except WebSocketDisconnect:
        logger.info("ws_live_disconnected", symbol=symbol)
    except Exception as e:
        logger.error("ws_live_error", symbol=symbol, error=str(e))
        with contextlib.suppress(Exception):
            await websocket.send_text(
                json.dumps({"type": "error", "message": str(e), "symbol": symbol})
            )
    finally:
        with contextlib.suppress(Exception):
            await websocket.close()


@router.websocket("/ws/market")
async def ws_market(websocket: WebSocket) -> None:
    """Stream periodic snapshots for Nifty 50 index and key benchmark prices.

    Sends a single ``market_summary`` message, then refreshes every
    TICK_INTERVAL_SECONDS while market is open.
    """
    await websocket.accept()
    logger.info("ws_market_connected")

    _BENCHMARKS = [
        ("NIFTY50", "NIFTY 50"),
        ("BANKNIFTY", "NIFTY Bank"),
        ("SENSEX", "SENSEX"),
    ]

    async def _build_summary() -> dict[str, Any]:
        items = []
        for sym, label in _BENCHMARKS:
            tick = await asyncio.to_thread(_fetch_tick, sym)
            items.append({"symbol": sym, "label": label, **tick})
        return {
            "type": "market_summary",
            "market_open": _is_market_open(),
            "indices": items,
            "generated_at": datetime.now(IST).isoformat(),
        }

    try:
        summary = await _build_summary()
        await websocket.send_text(json.dumps(summary))

        while _is_market_open():
            await asyncio.sleep(TICK_INTERVAL_SECONDS)
            summary = await _build_summary()
            await websocket.send_text(json.dumps(summary))

        await websocket.send_text(
            json.dumps(
                {
                    "type": "market_closed",
                    "message": "Market closed.",
                    "generated_at": datetime.now(IST).isoformat(),
                }
            )
        )

    except WebSocketDisconnect:
        logger.info("ws_market_disconnected")
    except Exception as e:
        logger.error("ws_market_error", error=str(e))
    finally:
        with contextlib.suppress(Exception):
            await websocket.close()
