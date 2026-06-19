"""Concall transcript collector — earnings call transcripts from NSE/BSE filings.

Companies file analyst-call transcripts within ~5 working days of the call
under LODR Reg 30. The NSE ``desc`` field is typically
"Analysts/Institutional Investor Meet/Con. Call Updates".

This collector:
1. Fetches transcript-related announcements from NSE
2. Filters for actual transcript PDFs (vs. schedules, notices)
3. Downloads and extracts full PDF text
4. Infers fiscal quarter from filing date
5. Splits into management remarks vs Q&A sections via heuristics
6. Stores in ``transcripts`` keyed by (symbol, fiscal_quarter)
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

TRANSCRIPT_DESC = "Analysts/Institutional Investor Meet/Con. Call Updates"

_TRANSCRIPT_KEYWORDS = {"transcript", "concall", "con call", "conference call", "earnings call"}

_SCHEDULE_KEYWORDS = {"schedule", "intimation", "notice", "revised", "postpone"}

_QA_MARKERS = re.compile(
    r"(?:^|\n)\s*(?:"
    r"(?:Q\s*(?:&|and)\s*A|question\s*(?:&|and)\s*answer)\s*(?:session|round|segment)?"
    r"|(?:we\s+(?:will|can)\s+now\s+open|(?:open|begin)\s+(?:the\s+)?(?:floor|session)\s+for\s+question)"
    r"|(?:moderator|operator)\s*:"
    r")",
    re.IGNORECASE,
)

_MGMT_MARKERS = re.compile(
    r"(?:^|\n)\s*(?:"
    r"(?:management|opening)\s+(?:remarks|commentary|discussion|presentation)"
    r"|(?:chairman|ceo|cfo|md|managing\s+director)\s*(?:\'s)?\s*(?:remarks|speech|address)"
    r"|good\s+(?:morning|afternoon|evening).*(?:ladies|shareholders|investors|participants)"
    r")",
    re.IGNORECASE,
)


def infer_fiscal_quarter(filed_at: datetime) -> str:
    """Infer Indian fiscal quarter from filing date.

    Indian fiscal year: Apr-Mar. Earnings calls happen ~1 month after
    quarter end, so a filing in May/Jun/Jul is Q4 results (Jan-Mar),
    filed in Aug/Sep/Oct is Q1 (Apr-Jun), etc.

    Returns format like "Q4FY26", "Q1FY27".
    """
    month = filed_at.month
    year = filed_at.year

    if month in (5, 6, 7):
        quarter = "Q4"
        fy = year
    elif month in (8, 9, 10):
        quarter = "Q1"
        fy = year + 1
    elif month in (11, 12, 1):
        quarter = "Q2"
        fy = year + 1 if month >= 11 else year
    else:
        quarter = "Q3"
        fy = year + 1

    fy_short = fy % 100
    return f"{quarter}FY{fy_short:02d}"


def is_transcript_announcement(desc: str, headline: str) -> bool:
    """Check if this announcement is likely a transcript filing."""
    if desc != TRANSCRIPT_DESC:
        text_lower = f"{desc} {headline}".lower()
        if not any(kw in text_lower for kw in _TRANSCRIPT_KEYWORDS):
            return False

    headline_lower = headline.lower()
    return not (
        any(kw in headline_lower for kw in _SCHEDULE_KEYWORDS)
        and "transcript" not in headline_lower
    )


def split_sections(text: str) -> dict[str, str]:
    """Split transcript text into management remarks and Q&A sections.

    Returns a dict with keys 'management' and/or 'qa'. If splitting
    fails (no markers found), returns {'full': text}.
    """
    qa_match = _QA_MARKERS.search(text)
    mgmt_match = _MGMT_MARKERS.search(text)

    if qa_match:
        qa_start = qa_match.start()
        management = text[:qa_start].strip()
        qa = text[qa_start:].strip()

        if mgmt_match and mgmt_match.start() < qa_start:
            management = text[mgmt_match.start() : qa_start].strip()

        result: dict[str, str] = {}
        if management:
            result["management"] = management
        if qa:
            result["qa"] = qa
        if result:
            return result

    if mgmt_match:
        return {"management": text[mgmt_match.start() :].strip()}

    return {"full": text}


def _parse_filed_at(dt_str: str) -> datetime | None:
    for fmt in ("%d-%b-%Y %H:%M:%S", "%d-%b-%Y %H:%M", "%d-%b-%Y"):
        try:
            naive = datetime.strptime(dt_str, fmt)
            return naive.replace(tzinfo=IST)
        except (ValueError, TypeError):
            continue
    return None


def row_to_transcript(row: dict[str, Any]) -> dict[str, Any] | None:
    """Convert an NSE announcement row into a transcript dict.

    Returns None if this isn't a transcript filing or lacks required fields.
    """
    desc = (row.get("desc", "") or "").strip()
    headline = (row.get("attchmntText", "") or row.get("sm_name", "")).strip()

    if not is_transcript_announcement(desc, headline):
        return None

    filed_at = _parse_filed_at(row.get("an_dt", "") or row.get("sort_date", ""))
    if filed_at is None:
        return None

    symbol_raw = (row.get("symbol", "") or "").strip()
    if not symbol_raw:
        return None

    pdf_url = (row.get("attchmntFile", "") or "").strip() or None
    fiscal_quarter = infer_fiscal_quarter(filed_at)

    return {
        "symbol": f"{symbol_raw}.NS",
        "fiscal_quarter": fiscal_quarter,
        "filed_at": filed_at,
        "text": None,
        "sections": None,
        "_pdf_url": pdf_url,
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
        logger.error("transcript_fetch_failed", error=str(e))
        return []


async def collect_transcripts(
    start: date,
    end: date,
    fetch_pdfs: bool = True,
) -> list[dict[str, Any]]:
    """Fetch transcript announcements, download PDFs, split sections."""
    await _rate_limiter.acquire()
    session = NSESession()

    raw_rows = await asyncio.to_thread(_fetch_announcements_sync, session, start, end)

    transcripts: list[dict[str, Any]] = []
    for row in raw_rows:
        tr = row_to_transcript(row)
        if tr is None:
            continue

        if fetch_pdfs and tr.get("_pdf_url"):
            await _rate_limiter.acquire()
            try:

                def _download(url: str = tr["_pdf_url"]) -> bytes:
                    return bytes(session.get(url).content)

                pdf_bytes = await asyncio.to_thread(_download)
                if pdf_bytes and len(pdf_bytes) > 100:
                    text = extract_pdf_text(pdf_bytes)
                    if text and len(text) > 500:
                        tr["text"] = text
                        tr["sections"] = split_sections(text)
            except Exception as e:
                logger.warning(
                    "transcript_pdf_failed",
                    url=tr["_pdf_url"],
                    error=str(e),
                )

        transcripts.append(tr)

    split_count = sum(
        1 for t in transcripts if t.get("sections") and "full" not in (t.get("sections") or {})
    )

    logger.info(
        "transcript_collect_complete",
        total=len(transcripts),
        with_text=sum(1 for t in transcripts if t.get("text")),
        with_sections=split_count,
    )
    return transcripts


async def ingest_transcripts_daily(
    fetch_pdfs: bool = True,
    days_back: int = 7,
) -> int:
    """Daily ingestion: fetch recent transcript announcements, store.

    Uses 7-day lookback by default since transcripts are filed ~5 days
    after the earnings call.
    """
    from alphavedha.intel.store import store_transcripts

    end = date.today()
    start = date.fromordinal(end.toordinal() - days_back)

    transcripts = await collect_transcripts(start, end, fetch_pdfs=fetch_pdfs)
    if not transcripts:
        logger.info("transcript_no_new_filings")
        return 0

    store_rows = [{k: v for k, v in t.items() if not k.startswith("_")} for t in transcripts]
    return await store_transcripts(store_rows)
