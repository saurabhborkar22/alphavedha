"""Credit rating actions collector — extracts rating events from disclosures.

Primary source: NSE corporate announcements with ``desc`` starting with
"Credit Rating". The headline typically says "X has informed the Exchange
about Credit Rating" — the actual rating details (agency, action, from/to,
outlook) live in the attached PDF.

This collector:
1. Fetches credit-rating announcements from NSE
2. Downloads and extracts PDF text
3. Parses agency name from headline/PDF text via keyword matching
4. Maps NSE ``desc`` subtypes to rating actions
5. Stores in ``rating_events`` with PDF text as ``rationale_text``

Full structured extraction (rating grades, outlook changes) is deferred
to P2 when the LLM extractor is built.
"""

from __future__ import annotations

import asyncio
import re
from datetime import date, datetime
from typing import Any
from zoneinfo import ZoneInfo

import structlog

from alphavedha.data.providers.base import RateLimiter
from alphavedha.data.providers.nse_provider import NSESession
from alphavedha.intel.collectors.bse_announcements import extract_pdf_text

logger = structlog.get_logger(__name__)

IST = ZoneInfo("Asia/Kolkata")

_NSE_ANN_URL = (
    "https://www.nseindia.com/api/corporate-announcements"
    "?index=equities&from_date={from_date}&to_date={to_date}"
)
_NSE_DATE_FMT = "%d-%m-%Y"

_rate_limiter = RateLimiter(requests_per_second=0.5)

_CREDIT_RATING_PREFIXES = ("Credit Rating",)

_ACTION_MAP: dict[str, str] = {
    "Credit Rating": "reaffirmed",
    "Credit Rating- New": "assigned",
    "Credit Rating- Revision": "revised",
    "Credit Rating- Others": "other",
}

AGENCY_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\bCRISIL\b", re.IGNORECASE), "CRISIL"),
    (re.compile(r"\bICRA\b", re.IGNORECASE), "ICRA"),
    (re.compile(r"\bCARE\b", re.IGNORECASE), "CARE"),
    (re.compile(r"\bIndia Ratings?\b", re.IGNORECASE), "India Ratings"),
    (re.compile(r"\bFitch\b", re.IGNORECASE), "Fitch"),
    (re.compile(r"\bBrickwork\b", re.IGNORECASE), "Brickwork"),
    (re.compile(r"\bAcuit[eé]\b", re.IGNORECASE), "Acuite"),
    (re.compile(r"\bInfomerics\b", re.IGNORECASE), "Infomerics"),
]


def detect_agency(text: str) -> str:
    """Detect rating agency from text via regex patterns."""
    for pattern, name in AGENCY_PATTERNS:
        if pattern.search(text):
            return name
    return "Unknown"


def _parse_filed_at(dt_str: str) -> datetime | None:
    for fmt in ("%d-%b-%Y %H:%M:%S", "%d-%b-%Y %H:%M", "%d-%b-%Y"):
        try:
            naive = datetime.strptime(dt_str, fmt)
            return naive.replace(tzinfo=IST)
        except (ValueError, TypeError):
            continue
    return None


def _is_credit_rating(desc: str) -> bool:
    return any(desc.startswith(p) for p in _CREDIT_RATING_PREFIXES)


def _map_action(desc: str) -> str:
    return _ACTION_MAP.get(desc, "reaffirmed")


def row_to_rating_event(row: dict[str, Any]) -> dict[str, Any] | None:
    """Convert an NSE announcement row into a rating_event dict.

    Returns None if the row is not a credit rating announcement or
    lacks required fields.
    """
    desc = (row.get("desc", "") or "").strip()
    if not _is_credit_rating(desc):
        return None

    filed_at = _parse_filed_at(row.get("an_dt", "") or row.get("sort_date", ""))
    if filed_at is None:
        return None

    symbol_raw = (row.get("symbol", "") or "").strip()
    if not symbol_raw:
        return None

    headline = (row.get("attchmntText", "") or row.get("sm_name", "")).strip()
    agency = detect_agency(headline)
    action = _map_action(desc)

    return {
        "symbol": f"{symbol_raw}.NS",
        "agency": agency[:30],
        "action": action[:30],
        "rating_from": None,
        "rating_to": None,
        "outlook": None,
        "rationale_text": None,
        "filed_at": filed_at,
        "_pdf_url": (row.get("attchmntFile", "") or "").strip() or None,
        "_headline": headline,
    }


def _fetch_announcements_sync(
    session: NSESession,
    start: date,
    end: date,
) -> list[dict[str, Any]]:
    url = _NSE_ANN_URL.format(
        from_date=start.strftime(_NSE_DATE_FMT),
        to_date=end.strftime(_NSE_DATE_FMT),
    )
    try:
        resp = session.get(url)
        data = resp.json()
        if isinstance(data, list):
            return data
        return []
    except Exception as e:
        logger.error("credit_rating_fetch_failed", error=str(e))
        return []


async def collect_credit_ratings(
    start: date,
    end: date,
    fetch_pdfs: bool = True,
) -> list[dict[str, Any]]:
    """Fetch credit rating announcements and extract rating events.

    Downloads PDFs to extract rationale text and refine agency detection.
    """
    await _rate_limiter.acquire()
    session = NSESession()

    raw_rows = await asyncio.to_thread(_fetch_announcements_sync, session, start, end)

    events: list[dict[str, Any]] = []
    for row in raw_rows:
        event = row_to_rating_event(row)
        if event is None:
            continue

        if fetch_pdfs and event.get("_pdf_url"):
            await _rate_limiter.acquire()
            try:

                def _download(url: str = event["_pdf_url"]) -> bytes:
                    return bytes(session.get(url).content)

                pdf_bytes = await asyncio.to_thread(_download)
                if pdf_bytes and len(pdf_bytes) > 100:
                    text = extract_pdf_text(pdf_bytes)
                    if text:
                        event["rationale_text"] = text[:10000]
                        if event["agency"] == "Unknown":
                            event["agency"] = detect_agency(text)
            except Exception as e:
                logger.warning(
                    "credit_rating_pdf_failed",
                    url=event["_pdf_url"],
                    error=str(e),
                )

        events.append(event)

    logger.info(
        "credit_rating_collect_complete",
        total=len(events),
        with_pdf=len([e for e in events if e.get("rationale_text")]),
        agencies={e["agency"] for e in events},
    )
    return events


async def ingest_credit_ratings_daily(
    fetch_pdfs: bool = True,
    days_back: int = 1,
) -> int:
    """Daily ingestion: fetch recent credit rating announcements, store events."""
    from alphavedha.intel.store import store_rating_events

    end = date.today()
    start = date.fromordinal(end.toordinal() - days_back)

    events = await collect_credit_ratings(start, end, fetch_pdfs=fetch_pdfs)
    if not events:
        logger.info("credit_rating_no_new_events")
        return 0

    store_rows = [{k: v for k, v in e.items() if not k.startswith("_")} for e in events]
    return await store_rating_events(store_rows)
