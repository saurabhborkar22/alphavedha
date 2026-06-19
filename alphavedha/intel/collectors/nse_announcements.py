"""NSE announcements collector — daily corporate disclosures from NSE India.

Fetches the corporate announcements API (whole-market, no per-symbol calls
needed), normalises into ``disclosures`` rows, and downloads attached PDFs
for text extraction.

Also identifies insider trading (PIT) and pledge (SAST) filings from the
announcement categories and routes them to the appropriate tables.
"""

from __future__ import annotations

import asyncio
import hashlib
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

PIT_CATEGORIES = {
    "insider trading",
    "acquisition",
    "disposal",
    "pit",
    "regulation 7",
}

PLEDGE_CATEGORIES = {
    "pledge",
    "sast",
    "substantial acquisition",
    "encumbrance",
}


def _parse_filed_at(dt_str: str) -> datetime | None:
    """Parse NSE announcement datetime into timezone-aware IST."""
    for fmt in ("%d-%b-%Y %H:%M:%S", "%d-%b-%Y %H:%M", "%d-%b-%Y"):
        try:
            naive = datetime.strptime(dt_str, fmt)
            return naive.replace(tzinfo=IST)
        except (ValueError, TypeError):
            continue
    return None


def _is_pit_filing(desc: str, headline: str) -> bool:
    """Check if this announcement is an insider trading (PIT) disclosure."""
    text = f"{desc} {headline}".lower()
    return any(cat in text for cat in PIT_CATEGORIES)


def _is_pledge_filing(desc: str, headline: str) -> bool:
    """Check if this announcement is a pledge/SAST disclosure."""
    text = f"{desc} {headline}".lower()
    return any(cat in text for cat in PLEDGE_CATEGORIES)


def _row_to_disclosure(row: dict[str, Any]) -> dict[str, Any] | None:
    """Convert an NSE API announcement row into a disclosure dict."""
    filed_at = _parse_filed_at(row.get("an_dt", "") or row.get("sort_date", ""))
    if filed_at is None:
        return None

    symbol_raw = (row.get("symbol", "") or "").strip()
    if not symbol_raw:
        return None
    symbol = f"{symbol_raw}.NS"

    headline = (row.get("attchmntText", "") or row.get("sm_name", "")).strip()
    if not headline:
        return None

    desc = (row.get("desc", "") or "").strip()
    category = desc or "General"

    pdf_url = (row.get("attchmntFile", "") or "").strip() or None

    return {
        "symbol": symbol,
        "source": "NSE",
        "category": category[:100],
        "headline": headline[:1000],
        "filed_at": filed_at,
        "url": pdf_url,
        "_is_pit": _is_pit_filing(desc, headline),
        "_is_pledge": _is_pledge_filing(desc, headline),
    }


def _text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _fetch_announcements_sync(
    session: NSESession,
    start: date,
    end: date,
) -> list[dict[str, Any]]:
    """Fetch all announcements for a date range. Blocking call."""
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
        logger.error("nse_ann_fetch_failed", error=str(e))
        return []


async def collect_nse_announcements(
    start: date,
    end: date,
    fetch_pdfs: bool = True,
) -> list[dict[str, Any]]:
    """Fetch NSE announcements for a date range and normalise.

    Returns list of disclosure dicts ready for store_disclosures().
    The ``_is_pit`` and ``_is_pledge`` flags are included for downstream
    routing but stripped before DB storage.
    """
    await _rate_limiter.acquire()

    session = NSESession()
    raw_rows = await asyncio.to_thread(_fetch_announcements_sync, session, start, end)

    disclosures: list[dict[str, Any]] = []
    for row in raw_rows:
        disc = _row_to_disclosure(row)
        if disc is None:
            continue

        if fetch_pdfs and disc.get("url"):
            await _rate_limiter.acquire()
            try:

                def _download(url: str = disc["url"]) -> bytes:
                    return bytes(session.get(url).content)

                pdf_bytes = await asyncio.to_thread(_download)
                if pdf_bytes and len(pdf_bytes) > 100:
                    text = extract_pdf_text(pdf_bytes)
                    if text:
                        disc["text"] = text
                        disc["text_hash"] = _text_hash(text)
            except Exception as e:
                logger.warning("nse_pdf_download_failed", url=disc["url"], error=str(e))

        disclosures.append(disc)

    logger.info(
        "nse_ann_collect_complete",
        total=len(disclosures),
        pit=len([d for d in disclosures if d.get("_is_pit")]),
        pledge=len([d for d in disclosures if d.get("_is_pledge")]),
        with_text=len([d for d in disclosures if d.get("text")]),
    )
    return disclosures


def _extract_insider_trade(disc: dict[str, Any]) -> dict[str, Any] | None:
    """Extract insider trade fields from a PIT-flagged disclosure.

    Returns a dict suitable for the existing insider_trades store,
    or None if parsing fails. Full extraction happens in P2 via LLM.
    """
    return {
        "symbol": disc["symbol"].replace(".NS", ""),
        "trade_date": disc["filed_at"].date()
        if isinstance(disc["filed_at"], datetime)
        else disc["filed_at"],
        "person_name": "See disclosure",
        "person_category": "PIT",
        "trade_type": "unknown",
        "shares": 0,
        "value_lakhs": 0.0,
    }


async def ingest_nse_announcements_daily(
    fetch_pdfs: bool = True,
    days_back: int = 1,
) -> dict[str, int]:
    """Daily ingestion: fetch recent NSE announcements, store disclosures.

    Returns counts: {'disclosures': N, 'pit_flagged': N, 'pledge_flagged': N}
    """
    from alphavedha.intel.store import store_disclosures

    end = date.today()
    start = date.fromordinal(end.toordinal() - days_back)

    all_discs = await collect_nse_announcements(start, end, fetch_pdfs=fetch_pdfs)

    pit_count = len([d for d in all_discs if d.get("_is_pit")])
    pledge_count = len([d for d in all_discs if d.get("_is_pledge")])

    store_rows = []
    for d in all_discs:
        row = {k: v for k, v in d.items() if not k.startswith("_")}
        store_rows.append(row)

    stored = await store_disclosures(store_rows) if store_rows else 0

    return {
        "disclosures": stored,
        "pit_flagged": pit_count,
        "pledge_flagged": pledge_count,
    }
