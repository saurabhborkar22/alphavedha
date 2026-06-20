"""Insider cluster signal — detects coordinated insider buying.

Triggers when >= MIN_DISTINCT_INSIDERS distinct insiders net-buy >= MIN_VALUE_LAKHS
within a CLUSTER_WINDOW_DAYS window for the same symbol. One of the most robust
documented effects in market microstructure literature.

Fires as strategy ``insider_cluster_v1`` with long-only signals.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any

import structlog

logger = structlog.get_logger(__name__)

STRATEGY_NAME = "insider_cluster_v1"
MIN_DISTINCT_INSIDERS = 2
MIN_VALUE_LAKHS = 25.0
CLUSTER_WINDOW_DAYS = 14


@dataclass
class InsiderClusterSignal:
    symbol: str
    direction: int
    confidence: float
    distinct_insiders: int
    total_value_lakhs: float
    window_start: date
    window_end: date


def _compute_confidence(distinct_insiders: int, total_value_lakhs: float) -> float:
    """More insiders and higher value → higher confidence."""
    base = 0.55
    insider_bonus = min((distinct_insiders - MIN_DISTINCT_INSIDERS) * 0.05, 0.15)
    value_bonus = min((total_value_lakhs - MIN_VALUE_LAKHS) / 500.0, 0.15)
    return round(min(base + insider_bonus + value_bonus, 0.90), 4)


def generate_insider_cluster_signals(
    trades_by_symbol: dict[str, list[dict[str, Any]]],
    signal_date: date,
    avoid_symbols: frozenset[str] | None = None,
) -> list[InsiderClusterSignal]:
    """Detect insider buying clusters across symbols.

    Args:
        trades_by_symbol: {symbol: [insider_trade_dicts]} — each dict has
            person_name, trade_type ("Buy"/"Sell"), value_lakhs, trade_date.
        signal_date: The date signals are generated for.
        avoid_symbols: Symbols on the blowup avoid list (vetoed).
    """
    if avoid_symbols is None:
        avoid_symbols = frozenset()

    window_start = signal_date - timedelta(days=CLUSTER_WINDOW_DAYS)
    signals: list[InsiderClusterSignal] = []

    for symbol, trades in trades_by_symbol.items():
        if symbol in avoid_symbols:
            continue

        recent = [
            t
            for t in trades
            if _parse_date(t.get("trade_date")) is not None
            and window_start <= _parse_date(t["trade_date"]) <= signal_date  # type: ignore[operator]
        ]

        buyers: dict[str, float] = defaultdict(float)
        for t in recent:
            person = str(t.get("person_name", "unknown"))
            trade_type = str(t.get("trade_type", "")).lower()
            value = float(t.get("value_lakhs", 0))

            if "buy" in trade_type:
                buyers[person] += value
            elif "sell" in trade_type:
                buyers[person] -= value

        net_buyers = {p: v for p, v in buyers.items() if v > 0}
        total_buy_value = sum(net_buyers.values())

        if len(net_buyers) >= MIN_DISTINCT_INSIDERS and total_buy_value >= MIN_VALUE_LAKHS:
            signals.append(
                InsiderClusterSignal(
                    symbol=symbol,
                    direction=1,
                    confidence=_compute_confidence(len(net_buyers), total_buy_value),
                    distinct_insiders=len(net_buyers),
                    total_value_lakhs=round(total_buy_value, 2),
                    window_start=window_start,
                    window_end=signal_date,
                )
            )

    signals.sort(key=lambda s: s.confidence, reverse=True)
    return signals


def _parse_date(val: Any) -> date | None:
    if val is None:
        return None
    if isinstance(val, date):
        return val
    try:
        return date.fromisoformat(str(val)[:10])
    except (ValueError, TypeError):
        return None


async def run_insider_cluster_signals(
    symbols: list[str],
    signal_date: date | None = None,
    avoid_symbols: frozenset[str] | None = None,
) -> list[InsiderClusterSignal]:
    """Load insider trades from DB and generate cluster signals."""
    from datetime import datetime
    from zoneinfo import ZoneInfo

    from alphavedha.data.store import load_insider_trades

    IST = ZoneInfo("Asia/Kolkata")

    if signal_date is None:
        signal_date = datetime.now(IST).date()

    trades_by_symbol: dict[str, list[dict[str, Any]]] = {}
    for symbol in symbols:
        df = await load_insider_trades(symbol, days_back=CLUSTER_WINDOW_DAYS + 7)
        if not df.empty:
            trades_by_symbol[symbol] = df.to_dict("records")

    return generate_insider_cluster_signals(trades_by_symbol, signal_date, avoid_symbols)
