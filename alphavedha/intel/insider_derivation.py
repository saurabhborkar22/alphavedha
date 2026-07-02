"""Derive insider trades from classified disclosure events.

NSE discontinued the ``corporates-pit`` JSON API (~2026-04-28): it returns
HTTP 200 with an empty dataset for any recent window, which froze the
``insider_trades`` table and starved the insider_cluster_v1 strategy.

The disclosures pipeline already ingests the underlying SEBI PIT/SAST
filings as announcements and the LLM layer classifies them into
``insider_buy`` / ``insider_sell`` events — so the primary source for
insider trades is now our own event stream:

    disclosure_events (insider_buy/sell) + parent disclosure
        → parse person / shares / value from the event summary
        → upsert into insider_trades (same table, same signal code)

Point-in-time rule: the stored ``trade_date`` is never after the filing's
exchange timestamp — a trade only becomes knowable when it is filed.
"""

from __future__ import annotations

import re
from datetime import date, datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo

import structlog

logger = structlog.get_logger(__name__)

IST = ZoneInfo("Asia/Kolkata")

DERIVATION_LOOKBACK_DAYS = 30
_LAKH = 100_000.0
_CRORE_LAKHS = 100.0

_INSIDER_EVENT_TYPES = ("insider_buy", "insider_sell")

# Person = everything before the first action verb in the LLM summary.
_ACTION_VERBS = (
    "acquired",
    "purchased",
    "bought",
    "sold",
    "disposed",
    "tendered",
    "transferred",
    "subscribed",
    "received",
    "allotted",
    "gifted",
    "invoked",
)
_PERSON_RE = re.compile(rf"^(?P<person>.+?)\s+(?:{'|'.join(_ACTION_VERBS)})\b", re.IGNORECASE)
_ROLE_PREFIX_RE = re.compile(
    r"^(?:promoter(?:\s+group)?(?:\s+member|\s+entity)?|director|kmp|designated person|employee|chairman|md|ceo|cfo)\s+",
    re.IGNORECASE,
)

# Share counts: "1,95,000 equity shares", "2.88L shares", "12.17Cr shares",
# "46.83 lakh shares", "3.83Cr" — the multiplier suffix may touch the number.
_SHARES_RE = re.compile(
    r"(?P<num>\d[\d,]*(?:\.\d+)?)\s*(?P<mult>lakh|lacs?|l\b|crores?|cr\b)?\s*(?:equity\s+)?shares",
    re.IGNORECASE,
)

# Explicit rupee values: "worth INR 4.07 crores", "Rs. 55 lakh", "₹12,50,000".
_VALUE_RE = re.compile(
    r"(?:worth\s+|valued\s+at\s+|aggregating\s+(?:to\s+)?)?(?:inr|rs\.?|₹)\s*"
    r"(?P<num>\d[\d,]*(?:\.\d+)?)\s*(?P<mult>lakhs?|lacs?|crores?|cr\b)?",
    re.IGNORECASE,
)

# Trade dates the LLM often echoes: "on 25-Jun-2026" or "on 2026-06-25".
_DATE_RE = re.compile(r"\bon\s+(?P<d>\d{1,2}-[A-Za-z]{3}-\d{4}|\d{4}-\d{2}-\d{2})\b")


def _to_float(num: str) -> float:
    return float(num.replace(",", ""))


def _apply_multiplier(value: float, mult: str | None) -> float:
    if not mult:
        return value
    m = mult.lower().rstrip(".")
    if m.startswith(("lakh", "lac")) or m == "l":
        return value * _LAKH
    if m.startswith("cr"):
        return value * _LAKH * _CRORE_LAKHS
    return value


def parse_person(summary: str, disclosure_id: int) -> str:
    """Extract the acting person/entity from an event summary.

    Falls back to a per-filing synthetic key so each unattributable filing
    still counts as one distinct insider (each PIT filing is per person).
    """
    match = _PERSON_RE.match(summary.strip())
    if match:
        person = _ROLE_PREFIX_RE.sub("", match.group("person").strip())
        person = person.strip(" ,;:-")
        if person:
            return person[:200]
    return f"filing-{disclosure_id}"


def parse_shares(summary: str) -> int:
    """Extract the share count from an event summary (0 when absent)."""
    match = _SHARES_RE.search(summary)
    if not match:
        return 0
    return int(_apply_multiplier(_to_float(match.group("num")), match.group("mult")))


def parse_value_lakhs(summary: str) -> float:
    """Extract an explicit rupee value in lakhs (0.0 when absent).

    Bare rupee amounts (no lakh/crore suffix) are treated as absolute
    rupees and converted.
    """
    match = _VALUE_RE.search(summary)
    if not match:
        return 0.0
    value = _to_float(match.group("num"))
    mult = match.group("mult")
    if mult:
        m = mult.lower().rstrip(".")
        if m.startswith(("lakh", "lac")):
            return round(value, 2)
        if m.startswith("cr"):
            return round(value * _CRORE_LAKHS, 2)
    # Plain rupees only make sense as a value when they are large.
    if value >= _LAKH:
        return round(value / _LAKH, 2)
    return 0.0


def parse_trade_date(summary: str, filed_at: datetime) -> date:
    """Trade date from the summary, clamped to the filing date.

    A trade only becomes knowable when filed — a parsed date after
    ``filed_at`` (LLM echo error) must not leak future information.
    """
    filed_date = filed_at.astimezone(IST).date() if filed_at.tzinfo else filed_at.date()
    match = _DATE_RE.search(summary)
    if match:
        raw = match.group("d")
        for fmt in ("%d-%b-%Y", "%Y-%m-%d"):
            try:
                parsed = datetime.strptime(raw, fmt).date()
            except ValueError:
                continue
            return min(parsed, filed_date)
    return filed_date


async def _estimate_value_lakhs(symbol: str, shares: int, as_of: date) -> float:
    """Estimate trade value from the latest close when the filing has none.

    daily_ohlcv holds both bare (yfinance) and .NS (bhavcopy) symbols —
    try both. Returns 0.0 when no price is available (never fabricate).
    """
    if shares <= 0:
        return 0.0
    from alphavedha.data.store import load_ohlcv

    for candidate in (symbol, f"{symbol}.NS"):
        try:
            df = await load_ohlcv(candidate, as_of - timedelta(days=10), as_of)
        except Exception as e:
            logger.warning("insider_value_estimate_failed", symbol=candidate, error=str(e))
            continue
        if not df.empty:
            close = float(df["close"].iloc[-1])
            return round(shares * close / _LAKH, 2)
    return 0.0


async def _load_recent_insider_events(
    since: datetime,
) -> list[tuple[Any, Any]]:
    """Load (event, parent disclosure) pairs for recent insider events."""
    from sqlalchemy import select

    from alphavedha.data.database import get_session_factory
    from alphavedha.data.models import Disclosure, DisclosureEvent

    session_factory = get_session_factory()
    async with session_factory() as session:
        stmt = (
            select(DisclosureEvent, Disclosure)
            .join(Disclosure, DisclosureEvent.disclosure_id == Disclosure.id)
            .where(
                DisclosureEvent.event_type.in_(_INSIDER_EVENT_TYPES),
                DisclosureEvent.extracted_at >= since,
            )
            .order_by(DisclosureEvent.id)
        )
        result = await session.execute(stmt)
        return [(row[0], row[1]) for row in result.all()]


async def derive_insider_trades(
    lookback_days: int = DERIVATION_LOOKBACK_DAYS,
) -> int:
    """Derive insider_trades rows from recent insider disclosure events.

    Idempotent: rows upsert on (symbol, trade_date, person_name), so
    re-running over an overlapping window never duplicates. Returns the
    number of rows written.
    """
    from alphavedha.data.store import store_insider_trades

    since = datetime.now(IST) - timedelta(days=lookback_days)
    pairs = await _load_recent_insider_events(since)
    if not pairs:
        logger.info("insider_derivation_no_events", since=str(since.date()))
        return 0

    rows: list[dict[str, Any]] = []
    for event, disclosure in pairs:
        summary = str(event.summary or "")
        symbol = str(event.symbol).removesuffix(".NS")
        trade_date = parse_trade_date(summary, disclosure.filed_at)
        shares = parse_shares(summary)
        value_lakhs = parse_value_lakhs(summary)
        if value_lakhs == 0.0:
            value_lakhs = await _estimate_value_lakhs(symbol, shares, trade_date)

        rows.append(
            {
                "symbol": symbol,
                "trade_date": trade_date,
                "person_name": parse_person(summary, int(event.disclosure_id)),
                "person_category": "derived_from_disclosure",
                "trade_type": "buy" if event.event_type == "insider_buy" else "sell",
                "shares": shares,
                "value_lakhs": value_lakhs,
            }
        )

    stored = await store_insider_trades(rows)
    logger.info(
        "insider_derivation_complete",
        events=len(pairs),
        rows_stored=stored,
        since=str(since.date()),
    )
    return stored
