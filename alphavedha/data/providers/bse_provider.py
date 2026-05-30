"""BSE corporate announcements provider.

Fetches corporate announcements (board meetings, dividends, bonuses, etc.)
from BSE India's public API. No authentication required.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any

import aiohttp
import structlog

# NSE symbol → BSE scrip code map (reuse from sebi_provider to avoid duplication)
from alphavedha.data.providers.sebi_provider import _BSE_SYMBOL_MAP

logger = structlog.get_logger(__name__)

# BSE Corporate Announcements API (no auth required)
_BSE_CORP_URL = (
    "https://api.bseindia.com/BseIndiaAPI/api/AnnSubCategoryGetData/w?"
    "pageno=1&strCat=-1&strPrevDate={from_date}&strScrip={scrip_code}"
    "&strSearch=P&strToDate={to_date}&strType=C&subcategory=-1"
)

_BSE_DATE_FMT = "%Y%m%d"

_CATEGORY_MAP: dict[str, str] = {
    "board meeting": "BOARD_MEETING",
    "dividend": "DIVIDEND",
    "bonus": "BONUS",
    "rights": "RIGHTS",
    "buyback": "BUYBACK",
    "split": "SPLIT",
    "agm": "AGM",
    "egm": "EGM",
}


@dataclass
class CorporateAnnouncementRecord:
    symbol: str
    announced_date: date
    ex_date: date | None
    event_type: str
    description: str

    def __post_init__(self) -> None:
        self.description = self.description[:500]


class BSEProvider:
    """Fetches corporate announcements from BSE India's public API."""

    def __init__(self, timeout_seconds: int = 30) -> None:
        self._timeout = aiohttp.ClientTimeout(total=timeout_seconds)

    def _nse_to_bse_code(self, nse_symbol: str) -> str | None:
        return _BSE_SYMBOL_MAP.get(nse_symbol)

    def _parse_event_type(self, category: str) -> str:
        cat_lower = category.lower()
        for key, val in _CATEGORY_MAP.items():
            if key in cat_lower:
                return val
        return "OTHER"

    def _parse_date(self, dt_str: str) -> date | None:
        for fmt in ("%Y%m%d%H%M%S", "%d-%m-%Y", "%Y-%m-%d"):
            try:
                return datetime.strptime(dt_str, fmt).date()
            except (ValueError, TypeError):
                continue
        return None

    def _parse_row(self, symbol: str, row: dict[str, Any]) -> CorporateAnnouncementRecord | None:
        announced = self._parse_date(row.get("DT_TM", "") or row.get("News_dt", ""))
        if announced is None:
            return None
        return CorporateAnnouncementRecord(
            symbol=symbol,
            announced_date=announced,
            ex_date=None,
            event_type=self._parse_event_type(row.get("CATEGORYNAME", "") or row.get("NSURL", "")),
            description=(row.get("HEADLINE", "") or row.get("Headline", ""))[:500],
        )

    async def fetch_announcements(
        self, symbol: str, start: date, end: date
    ) -> list[CorporateAnnouncementRecord]:
        scrip_code = self._nse_to_bse_code(symbol)
        if scrip_code is None:
            logger.warning("bse_provider.no_scrip_code", symbol=symbol)
            return []

        url = _BSE_CORP_URL.format(
            from_date=start.strftime(_BSE_DATE_FMT),
            scrip_code=scrip_code,
            to_date=end.strftime(_BSE_DATE_FMT),
        )

        try:
            async with (
                aiohttp.ClientSession(timeout=self._timeout) as client,
                client.get(url, headers={"User-Agent": "AlphaVedha/1.0"}) as resp,
            ):
                if resp.status != 200:
                    logger.warning("bse_provider.http_error", status=resp.status, symbol=symbol)
                    return []
                data: dict[str, Any] = await resp.json(content_type=None)
        except Exception as exc:
            logger.error("bse_provider.fetch_error", symbol=symbol, error=str(exc))
            return []

        rows = data.get("Table", [])
        records: list[CorporateAnnouncementRecord] = []
        for row in rows:
            rec = self._parse_row(symbol, row)
            if rec is not None:
                records.append(rec)
        logger.info("bse_provider.fetched", symbol=symbol, count=len(records))
        return records

    async def fetch_bulk(
        self, symbols: list[str], start: date, end: date
    ) -> dict[str, list[CorporateAnnouncementRecord]]:
        tasks = {sym: self.fetch_announcements(sym, start, end) for sym in symbols}
        results = await asyncio.gather(*tasks.values(), return_exceptions=True)
        output: dict[str, list[CorporateAnnouncementRecord]] = {}
        for sym, result in zip(symbols, results, strict=True):
            if isinstance(result, Exception):
                logger.error("bse_provider.bulk_error", symbol=sym, error=str(result))
                output[sym] = []
            else:
                output[sym] = result  # type: ignore[assignment]
        return output

    async def health_check(self) -> bool:
        try:
            url = _BSE_CORP_URL.format(
                from_date="20260101", scrip_code="532540", to_date="20260110"
            )
            async with (
                aiohttp.ClientSession(timeout=self._timeout) as client,
                client.get(url, headers={"User-Agent": "AlphaVedha/1.0"}) as resp,
            ):
                return resp.status == 200
        except Exception:
            return False
