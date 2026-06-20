"""Blowup detector — composite 0-100 risk score per symbol.

Combines multiple red-flag signals into a single score:
- Pledge increase trend (promoter pledging rising)
- Rating downgrade / negative outlook
- Auditor or KMP resignation
- Default or payment delay
- ASM/GSM surveillance addition
- Beneish M-Score red zone
- Insider sell clusters
- Volume + price pump signature (not yet implemented — placeholder 0)

Symbols scoring >= AVOID_THRESHOLD land on the daily avoid list, exposed
at ``/api/intel/red-flags``. Ensemble and event_drift signals on avoid-listed
symbols are vetoed (recorded as vetoed so the veto itself is measurable).

Paper strategy ``blowup_short_v1`` emits direction=-1 on new avoid-list
entries to measure whether the flag predicts drawdowns.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo

import structlog

logger = structlog.get_logger(__name__)

IST = ZoneInfo("Asia/Kolkata")

STRATEGY_NAME = "blowup_short_v1"
AVOID_THRESHOLD = 70
LOOKBACK_DAYS = 90


@dataclass
class BlowupScore:
    symbol: str
    total_score: int
    pledge_score: int = 0
    rating_score: int = 0
    governance_score: int = 0
    default_score: int = 0
    surveillance_score: int = 0
    beneish_score: int = 0
    insider_sell_score: int = 0
    pump_score: int = 0
    flags: list[str] = field(default_factory=list)
    on_avoid_list: bool = False


def compute_blowup_score(
    symbol: str,
    disclosure_events: list[dict[str, Any]],
    rating_events: list[dict[str, Any]],
    pledge_snapshots: list[dict[str, Any]],
    surveillance_flags: list[dict[str, Any]],
    beneish_result: dict[str, Any] | None = None,
) -> BlowupScore:
    """Compute composite blowup score for a single symbol."""
    flags: list[str] = []

    pledge_score = _score_pledges(pledge_snapshots, flags)
    rating_score = _score_ratings(rating_events, flags)
    governance_score = _score_governance(disclosure_events, flags)
    default_score = _score_defaults(disclosure_events, flags)
    surveillance_score = _score_surveillance(surveillance_flags, flags)
    beneish_score_val = _score_beneish(beneish_result, flags)
    insider_sell_score = _score_insider_sells(disclosure_events, flags)

    total = min(
        100,
        pledge_score
        + rating_score
        + governance_score
        + default_score
        + surveillance_score
        + beneish_score_val
        + insider_sell_score,
    )

    return BlowupScore(
        symbol=symbol,
        total_score=total,
        pledge_score=pledge_score,
        rating_score=rating_score,
        governance_score=governance_score,
        default_score=default_score,
        surveillance_score=surveillance_score,
        beneish_score=beneish_score_val,
        insider_sell_score=insider_sell_score,
        pump_score=0,
        flags=flags,
        on_avoid_list=total >= AVOID_THRESHOLD,
    )


def _score_pledges(snapshots: list[dict[str, Any]], flags: list[str]) -> int:
    """Promoter pledge > 30% or rising trend → 0-25 points."""
    if not snapshots:
        return 0

    latest_pct = float(snapshots[0].get("promoter_pledge_pct", 0))
    if latest_pct >= 50:
        flags.append("pledge_critical_50pct")
        return 25
    if latest_pct >= 30:
        flags.append("pledge_high_30pct")
        return 15

    if len(snapshots) >= 2:
        prev_pct = float(snapshots[-1].get("promoter_pledge_pct", 0))
        if latest_pct > prev_pct + 5:
            flags.append("pledge_rising")
            return 10

    return 0


def _score_ratings(events: list[dict[str, Any]], flags: list[str]) -> int:
    """Rating downgrade or negative outlook → 0-20 points."""
    score = 0
    for ev in events:
        action = str(ev.get("action", "")).lower()
        outlook = str(ev.get("outlook", "")).lower()
        if "downgrade" in action:
            flags.append(f"rating_downgrade_{ev.get('agency', 'unknown')}")
            score = max(score, 20)
        elif "negative" in outlook:
            flags.append(f"outlook_negative_{ev.get('agency', 'unknown')}")
            score = max(score, 10)
    return score


def _score_governance(events: list[dict[str, Any]], flags: list[str]) -> int:
    """Auditor or KMP resignation → 0-20 points."""
    score = 0
    for ev in events:
        et = str(ev.get("event_type", ""))
        if et == "auditor_resignation":
            flags.append("auditor_resignation")
            score = max(score, 20)
        elif et == "kmp_resignation":
            flags.append("kmp_resignation")
            score = max(score, 15)
    return score


def _score_defaults(events: list[dict[str, Any]], flags: list[str]) -> int:
    """Default or payment delay → 0-25 points."""
    for ev in events:
        et = str(ev.get("event_type", ""))
        if et == "default_or_delay":
            flags.append("default_or_delay")
            return 25
    return 0


def _score_surveillance(flags_list: list[dict[str, Any]], flags: list[str]) -> int:
    """ASM/GSM addition → 0-15 points."""
    for sf in flags_list:
        if sf.get("removed_on") is None:
            list_name = str(sf.get("list_name", ""))
            flags.append(f"surveillance_{list_name}")
            return 15
    return 0


def _score_beneish(result: dict[str, Any] | None, flags: list[str]) -> int:
    """Beneish M-Score in manipulator zone → 0-15 points."""
    if result is None:
        return 0
    verdict = str(result.get("verdict", ""))
    if verdict == "manipulator":
        flags.append("beneish_manipulator")
        return 15
    if verdict == "grey_zone":
        flags.append("beneish_grey_zone")
        return 5
    return 0


def _score_insider_sells(events: list[dict[str, Any]], flags: list[str]) -> int:
    """Cluster of insider sells (>= 2 within the lookback) → 0-15 points."""
    sells = [e for e in events if str(e.get("event_type", "")) == "insider_sell"]
    if len(sells) >= 3:
        flags.append("insider_sell_cluster_3plus")
        return 15
    if len(sells) >= 2:
        flags.append("insider_sell_cluster_2")
        return 10
    return 0


def compute_avoid_list(scores: list[BlowupScore]) -> list[BlowupScore]:
    """Filter scores to only those on the avoid list (score >= threshold)."""
    return [s for s in scores if s.on_avoid_list]


def is_vetoed(symbol: str, avoid_list: list[BlowupScore]) -> bool:
    """Check if a symbol is on the avoid list."""
    return any(s.symbol == symbol for s in avoid_list)


async def run_blowup_scores(
    symbols: list[str],
    as_of: date | None = None,
) -> list[BlowupScore]:
    """Compute blowup scores for a list of symbols using stored intel data."""
    from alphavedha.intel.store import (
        load_disclosure_events,
        load_rating_events,
        load_surveillance_flags,
    )

    if as_of is None:
        as_of = datetime.now(IST).date()

    since = datetime(as_of.year, as_of.month, as_of.day, tzinfo=IST) - timedelta(days=LOOKBACK_DAYS)

    scores: list[BlowupScore] = []
    for symbol in symbols:
        try:
            events_df = await load_disclosure_events(symbol=symbol, since=since)
            events = events_df.to_dict("records") if not events_df.empty else []

            rating_df = await load_rating_events(symbol=symbol, since=since)
            rating_events = rating_df.to_dict("records") if not rating_df.empty else []

            surv_df = await load_surveillance_flags(symbol=symbol)
            surveillance = surv_df.to_dict("records") if not surv_df.empty else []

            score = compute_blowup_score(
                symbol=symbol,
                disclosure_events=events,
                rating_events=rating_events,
                pledge_snapshots=[],
                surveillance_flags=surveillance,
            )
            scores.append(score)
        except Exception as e:
            logger.error("blowup_score_failed", symbol=symbol, error=str(e))

    avoid = compute_avoid_list(scores)
    if avoid:
        logger.warning(
            "blowup_avoid_list",
            symbols=[s.symbol for s in avoid],
            scores=[s.total_score for s in avoid],
        )
    return scores
