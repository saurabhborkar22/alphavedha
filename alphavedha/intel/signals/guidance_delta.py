"""Guidance delta signal — trades on transcript QoQ changes.

Uses disclosure_events with event_type=results_guidance produced by the
transcript delta extractor. A guidance_delta >= +1 with non-negative tone_delta
fires a long signal; guidance_delta <= -1 fires a flagged-negative signal.

Fires as strategy ``guidance_delta_v1``.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo

import structlog

logger = structlog.get_logger(__name__)

IST = ZoneInfo("Asia/Kolkata")

STRATEGY_NAME = "guidance_delta_v1"
LOOKBACK_DAYS = 30
MIN_MATERIALITY = 2


@dataclass
class GuidanceDeltaSignal:
    symbol: str
    direction: int
    confidence: float
    guidance_delta: int
    tone_delta: int
    materiality: int
    short_constrained: bool
    event_summary: str


def generate_guidance_signals(
    events: list[dict[str, Any]],
    signal_date: date,
) -> list[GuidanceDeltaSignal]:
    """Generate signals from transcript guidance delta events.

    Only events filed BEFORE signal_date are eligible (no look-ahead).
    """
    cutoff = datetime(signal_date.year, signal_date.month, signal_date.day, tzinfo=IST)
    lookback_start = cutoff - timedelta(days=LOOKBACK_DAYS)

    eligible: list[dict[str, Any]] = []
    for ev in events:
        if str(ev.get("event_type", "")) != "results_guidance":
            continue

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
        if filed >= cutoff or filed < lookback_start:
            continue

        materiality = int(ev.get("materiality", 0))
        if materiality < MIN_MATERIALITY:
            continue

        eligible.append(ev)

    by_symbol: dict[str, dict[str, Any]] = {}
    for ev in eligible:
        sym = str(ev["symbol"])
        if sym not in by_symbol or int(ev.get("materiality", 0)) > int(
            by_symbol[sym].get("materiality", 0)
        ):
            by_symbol[sym] = ev

    signals: list[GuidanceDeltaSignal] = []
    for symbol, ev in by_symbol.items():
        direction = int(ev.get("direction", 0))
        if direction == 0:
            continue

        materiality = int(ev.get("materiality", 0))
        summary = str(ev.get("summary", ""))[:200]

        guidance_delta = _extract_guidance_delta(ev)
        tone_delta = _extract_tone_delta(ev)

        confidence = _compute_confidence(materiality, abs(guidance_delta), abs(tone_delta))

        signals.append(
            GuidanceDeltaSignal(
                symbol=symbol,
                direction=direction,
                confidence=confidence,
                guidance_delta=guidance_delta,
                tone_delta=tone_delta,
                materiality=materiality,
                short_constrained=direction == -1,
                event_summary=summary,
            )
        )

    signals.sort(key=lambda s: s.confidence, reverse=True)
    return signals


def _extract_guidance_delta(ev: dict[str, Any]) -> int:
    """Extract guidance_delta from event summary or red_flags metadata."""
    summary = str(ev.get("summary", "")).lower()
    direction = int(ev.get("direction", 0))

    if "significantly better" in summary or "significantly improved" in summary:
        return 2
    if "slightly better" in summary or "improved" in summary:
        return 1
    if "significantly worse" in summary or "significantly deteriorated" in summary:
        return -2
    if "slightly worse" in summary or "deteriorated" in summary:
        return -1

    if direction > 0:
        return 1
    if direction < 0:
        return -1
    return 0


def _extract_tone_delta(ev: dict[str, Any]) -> int:
    """Extract tone_delta from event metadata."""
    red_flags = ev.get("red_flags") or []
    for flag in red_flags:
        flag_str = str(flag)
        if "evasiveness_score=" in flag_str:
            try:
                score = int(flag_str.split("=")[1])
                if score >= 7:
                    return -1
            except (ValueError, IndexError):
                pass

    direction = int(ev.get("direction", 0))
    return 1 if direction > 0 else (-1 if direction < 0 else 0)


def _compute_confidence(materiality: int, abs_guidance: int, abs_tone: int) -> float:
    """Higher materiality and stronger delta → higher confidence."""
    base = 0.45
    mat_bonus = min(materiality * 0.06, 0.3)
    guidance_bonus = min(abs_guidance * 0.05, 0.1)
    tone_bonus = min(abs_tone * 0.03, 0.06)
    return round(min(base + mat_bonus + guidance_bonus + tone_bonus, 0.85), 4)


async def run_guidance_delta_signals(
    signal_date: date | None = None,
) -> list[GuidanceDeltaSignal]:
    """Load transcript delta events and generate guidance signals."""
    from alphavedha.intel.store import load_disclosure_events

    if signal_date is None:
        signal_date = datetime.now(IST).date()

    since = datetime(signal_date.year, signal_date.month, signal_date.day, tzinfo=IST) - timedelta(
        days=LOOKBACK_DAYS
    )

    events_df = await load_disclosure_events(event_type="results_guidance", since=since)

    if events_df.empty:
        logger.info("guidance_delta_no_events", signal_date=str(signal_date))
        return []

    events = events_df.to_dict("records")
    signals = generate_guidance_signals(events, signal_date)
    logger.info(
        "guidance_delta_signals_generated",
        signal_date=str(signal_date),
        n_signals=len(signals),
    )
    return signals
