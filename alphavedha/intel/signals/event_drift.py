"""Event drift signal — generates paper-trade signals from disclosure events.

For each symbol with recent material events (filed yesterday, so no look-ahead),
emits a direction + confidence based on event type, materiality, and the
taxonomy's default direction. Positive events → long; negative events → short
with ``short_constrained=True`` (measured, not assumed tradeable).

Wired into the 08:30 scheduler job as strategy ``event_drift_v1``.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo

import structlog

from alphavedha.intel.extraction.taxonomy import EVENT_CATALOG, EventType

logger = structlog.get_logger(__name__)

IST = ZoneInfo("Asia/Kolkata")

STRATEGY_NAME = "event_drift_v1"
MAX_POSITIONS_PER_DAY = 10
MIN_MATERIALITY = 3
MIN_CONFIDENCE = 0.5


@dataclass
class EventDriftSignal:
    symbol: str
    direction: int
    confidence: float
    materiality: int
    event_type: str
    short_constrained: bool
    event_summary: str


def _event_confidence(materiality: int, extraction_confidence: float) -> float:
    """Confidence = materiality (1-5) mapped to 0.4-0.9 range, weighted by LLM confidence."""
    base = 0.3 + materiality * 0.12
    return round(min(base * extraction_confidence * 1.2, 0.95), 4)


def generate_signals(
    events: list[dict[str, Any]],
    signal_date: date,
) -> list[EventDriftSignal]:
    """Generate event drift signals from disclosure events.

    Only events filed BEFORE signal_date are eligible (no look-ahead).
    Events from the previous trading day (signal_date - 1 calendar day as
    approximation) are the primary source.
    """
    cutoff = datetime(signal_date.year, signal_date.month, signal_date.day, tzinfo=IST)
    lookback_start = cutoff - timedelta(days=3)

    eligible: list[dict[str, Any]] = []
    for ev in events:
        filed = ev.get("extracted_at") or ev.get("filed_at")
        if filed is None:
            continue
        if isinstance(filed, str):
            try:
                filed = datetime.fromisoformat(filed)
            except ValueError:
                continue
        if not hasattr(filed, "date"):
            continue
        if filed >= cutoff:
            continue
        if filed < lookback_start:
            continue

        materiality = int(ev.get("materiality", 0))
        if materiality < MIN_MATERIALITY:
            continue

        eligible.append(ev)

    by_symbol: dict[str, list[dict[str, Any]]] = {}
    for ev in eligible:
        sym = str(ev["symbol"])
        by_symbol.setdefault(sym, []).append(ev)

    signals: list[EventDriftSignal] = []
    for symbol, symbol_events in by_symbol.items():
        best = max(symbol_events, key=lambda e: int(e.get("materiality", 0)))
        event_type_str = str(best.get("event_type", "other"))
        direction = int(best.get("direction", 0))

        if direction == 0:
            try:
                et = EventType(event_type_str)
                direction = EVENT_CATALOG[et].default_direction
            except (ValueError, KeyError):
                continue

        if direction == 0:
            continue

        materiality = int(best.get("materiality", 0))
        extraction_conf = float(best.get("confidence", 0.5))
        confidence = _event_confidence(materiality, extraction_conf)

        if confidence < MIN_CONFIDENCE:
            continue

        signals.append(
            EventDriftSignal(
                symbol=symbol,
                direction=direction,
                confidence=confidence,
                materiality=materiality,
                event_type=event_type_str,
                short_constrained=direction == -1,
                event_summary=str(best.get("summary", ""))[:200],
            )
        )

    signals.sort(key=lambda s: s.confidence, reverse=True)
    return signals[:MAX_POSITIONS_PER_DAY]


async def run_event_drift_signals(signal_date: date | None = None) -> list[EventDriftSignal]:
    """Load recent disclosure events and generate signals for the given date."""
    from alphavedha.intel.store import load_disclosure_events

    if signal_date is None:
        signal_date = datetime.now(IST).date()

    since = datetime(signal_date.year, signal_date.month, signal_date.day, tzinfo=IST) - timedelta(
        days=3
    )
    events_df = await load_disclosure_events(since=since)

    if events_df.empty:
        logger.info("event_drift_no_events", signal_date=str(signal_date))
        return []

    events = events_df.to_dict("records")
    signals = generate_signals(events, signal_date)
    logger.info(
        "event_drift_signals_generated",
        signal_date=str(signal_date),
        n_signals=len(signals),
    )
    return signals
