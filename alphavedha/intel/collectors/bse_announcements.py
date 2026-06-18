"""BSE announcements collector — daily disclosure ingestion + PDF text extraction.

Fetches corporate announcements from BSE India's API (per-symbol, with session
cookies), normalises them into ``disclosures`` rows, downloads the attached PDF,
and extracts text via PyMuPDF. Scanned PDFs (no extractable text) are stored
with ``text IS NULL``.
"""

from __future__ import annotations

import hashlib
from datetime import date, datetime
from typing import Any
from zoneinfo import ZoneInfo

import aiohttp
import structlog

from alphavedha.data.providers.base import RateLimiter

logger = structlog.get_logger(__name__)

IST = ZoneInfo("Asia/Kolkata")

_BSE_ANN_URL = (
    "https://api.bseindia.com/BseIndiaAPI/api/AnnSubCategoryGetData/w?"
    "pageno={page}&strCat=-1&strPrevDate={from_date}&strScrip={scrip_code}"
    "&strSearch=P&strToDate={to_date}&strType=C&subcategory=-1"
)
_BSE_PDF_BASE = "https://www.bseindia.com/xml-data/corpfiling/AttachLive"
_BSE_DATE_FMT = "%Y%m%d"
_BSE_ANNOUNCEMENTS_PAGE = "https://www.bseindia.com/corporates/ann.html"
_MAX_PDF_PAGES = 50

_HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36",
    "Referer": "https://www.bseindia.com/corporates/ann.html",
    "Accept": "application/json, text/plain, */*",
}

_rate_limiter = RateLimiter(requests_per_second=0.5)


def _parse_filed_at(dt_str: str) -> datetime | None:
    """Parse BSE datetime string into timezone-aware IST datetime."""
    for fmt in ("%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S"):
        try:
            naive = datetime.strptime(dt_str, fmt)
            return naive.replace(tzinfo=IST)
        except (ValueError, TypeError):
            continue
    return None


def _pdf_url(attachment_name: str) -> str | None:
    if not attachment_name or not attachment_name.strip():
        return None
    return f"{_BSE_PDF_BASE}/{attachment_name.strip()}"


def _row_to_disclosure(nse_symbol: str, row: dict[str, Any]) -> dict[str, Any] | None:
    """Convert a BSE API row into a disclosure dict for store_disclosures."""
    filed_at = _parse_filed_at(row.get("DT_TM", "") or row.get("NEWS_DT", ""))
    if filed_at is None:
        return None

    headline = (row.get("NEWSSUB", "") or row.get("HEADLINE", "")).replace("''", "'").strip()
    if not headline:
        return None

    category = (row.get("CATEGORYNAME", "") or "").strip()
    attachment = (row.get("ATTACHMENTNAME", "") or "").strip()

    return {
        "symbol": nse_symbol,
        "source": "BSE",
        "category": category or "General",
        "headline": headline[:1000],
        "filed_at": filed_at,
        "url": _pdf_url(attachment),
    }


def extract_pdf_text(pdf_bytes: bytes) -> str | None:
    """Extract text from PDF bytes using PyMuPDF. Returns None for scanned PDFs."""
    try:
        import fitz
    except ImportError:
        logger.warning("pymupdf_not_installed")
        return None

    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        pages_to_read = min(len(doc), _MAX_PDF_PAGES)
        text_parts: list[str] = []

        for i in range(pages_to_read):
            page_text = doc[i].get_text()
            if page_text.strip():
                text_parts.append(page_text)

        doc.close()

        full_text = "\n".join(text_parts).strip()
        return full_text if full_text else None
    except Exception as e:
        logger.warning("pdf_extraction_failed", error=str(e))
        return None


def _text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


async def _fetch_announcements_page(
    session: aiohttp.ClientSession,
    scrip_code: str,
    start: date,
    end: date,
    page: int = 1,
) -> list[dict[str, Any]]:
    """Fetch one page of BSE announcements for a scrip code."""
    await _rate_limiter.acquire()

    url = _BSE_ANN_URL.format(
        page=page,
        from_date=start.strftime(_BSE_DATE_FMT),
        scrip_code=scrip_code,
        to_date=end.strftime(_BSE_DATE_FMT),
    )

    try:
        async with session.get(url) as resp:
            if resp.status != 200:
                logger.warning("bse_ann_http_error", status=resp.status, scrip=scrip_code)
                return []
            data = await resp.json(content_type=None)
            if isinstance(data, dict):
                return list(data.get("Table", []))
            return []
    except Exception as e:
        logger.error("bse_ann_fetch_failed", scrip=scrip_code, error=str(e))
        return []


async def _download_pdf(session: aiohttp.ClientSession, url: str) -> bytes | None:
    """Download a PDF from BSE. Returns bytes or None on failure."""
    await _rate_limiter.acquire()

    try:
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=30)) as resp:
            if resp.status != 200:
                return None
            content = await resp.read()
            if len(content) < 100:
                return None
            return content
    except Exception as e:
        logger.warning("pdf_download_failed", url=url, error=str(e))
        return None


async def collect_bse_announcements(
    symbols_to_scrips: dict[str, str],
    start: date,
    end: date,
    fetch_pdfs: bool = True,
) -> list[dict[str, Any]]:
    """Fetch BSE announcements for all mapped symbols, optionally extracting PDF text.

    Args:
        symbols_to_scrips: mapping of NSE symbol → BSE scrip code
        start: start date (inclusive)
        end: end date (inclusive)
        fetch_pdfs: whether to download and extract PDF text

    Returns:
        list of disclosure dicts ready for store_disclosures()
    """
    disclosures: list[dict[str, Any]] = []

    async with aiohttp.ClientSession(headers=_HEADERS) as session:
        async with session.get(_BSE_ANNOUNCEMENTS_PAGE) as _:
            pass

        for nse_symbol, scrip_code in symbols_to_scrips.items():
            rows = await _fetch_announcements_page(session, scrip_code, start, end)

            for row in rows:
                disc = _row_to_disclosure(nse_symbol, row)
                if disc is None:
                    continue

                if fetch_pdfs and disc["url"]:
                    pdf_bytes = await _download_pdf(session, disc["url"])
                    if pdf_bytes:
                        text = extract_pdf_text(pdf_bytes)
                        if text:
                            disc["text"] = text
                            disc["text_hash"] = _text_hash(text)

                disclosures.append(disc)

            logger.info(
                "bse_ann_symbol_done",
                symbol=nse_symbol,
                count=len([d for d in disclosures if d["symbol"] == nse_symbol]),
            )

    logger.info(
        "bse_ann_collect_complete",
        symbols=len(symbols_to_scrips),
        disclosures=len(disclosures),
        with_text=len([d for d in disclosures if d.get("text")]),
    )
    return disclosures


async def ingest_bse_announcements_daily(
    fetch_pdfs: bool = True,
    days_back: int = 1,
) -> int:
    """Daily ingestion: fetch yesterday's (or recent) BSE announcements and store.

    Returns number of disclosures stored.
    """
    from alphavedha.data.providers.sebi_provider import _BSE_SYMBOL_MAP
    from alphavedha.intel.store import store_disclosures

    end = date.today()
    start = date.fromordinal(end.toordinal() - days_back)

    symbols_to_scrips = {sym: code for sym, code in _BSE_SYMBOL_MAP.items() if code}

    if not symbols_to_scrips:
        logger.warning("bse_ann_no_symbols_mapped")
        return 0

    disclosures = await collect_bse_announcements(
        symbols_to_scrips, start, end, fetch_pdfs=fetch_pdfs
    )

    if not disclosures:
        logger.info("bse_ann_no_new_disclosures")
        return 0

    return await store_disclosures(disclosures)
