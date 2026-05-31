from __future__ import annotations

import asyncio
from typing import Any

import pandas as pd
import structlog

logger = structlog.get_logger(__name__)

SECTOR_KEYWORDS: dict[str, list[str]] = {
    "banking": ["HDFC Bank", "SBI", "ICICI Bank", "Axis Bank", "bank nifty"],
    "it": ["Infosys", "TCS", "Wipro", "HCL Tech", "IT stocks India"],
    "pharma": ["Sun Pharma", "Dr Reddy", "Cipla", "pharma stocks India"],
    "auto": ["Maruti", "Tata Motors", "Bajaj Auto", "auto stocks India"],
    "fmcg": ["Hindustan Unilever", "ITC", "Nestle India", "FMCG stocks India"],
}

# NSE symbol → sector mapping for the trends features
SYMBOL_TO_SECTOR: dict[str, str] = {
    "HDFCBANK.NS": "banking",
    "SBIN.NS": "banking",
    "ICICIBANK.NS": "banking",
    "AXISBANK.NS": "banking",
    "KOTAKBANK.NS": "banking",
    "TCS.NS": "it",
    "INFY.NS": "it",
    "WIPRO.NS": "it",
    "HCLTECH.NS": "it",
    "TECHM.NS": "it",
    "SUNPHARMA.NS": "pharma",
    "DRREDDY.NS": "pharma",
    "CIPLA.NS": "pharma",
    "MARUTI.NS": "auto",
    "TATAMOTORS.NS": "auto",
    "M&M.NS": "auto",
    "HINDUNILVR.NS": "fmcg",
    "ITC.NS": "fmcg",
    "NESTLEIND.NS": "fmcg",
}


class GoogleTrendsProvider:
    """Fetches Google Trends interest-over-time data for Indian market sectors.

    Uses pytrends in a thread pool executor to avoid blocking the event loop.
    Degrades gracefully on rate limits or network errors (returns empty DataFrame).
    """

    def __init__(self, hl: str = "en-IN", geo: str = "IN") -> None:
        self._hl = hl
        self._geo = geo

    def _build_pytrends(self) -> Any:
        from pytrends.request import TrendReq

        return TrendReq(hl=self._hl, geo=self._geo, timeout=(10, 30))

    def _fetch_sync(self, keywords: list[str], timeframe: str) -> pd.DataFrame:
        pt = self._build_pytrends()
        pt.build_payload(keywords[:5], cat=0, timeframe=timeframe, geo=self._geo)
        return pt.interest_over_time()

    async def fetch_sector_trends(self, sector: str, timeframe: str = "today 3-m") -> pd.DataFrame:
        """Fetch Google Trends for a sector keyword group.

        Returns empty DataFrame on any error (rate limits, network issues).
        """
        keywords = SECTOR_KEYWORDS.get(sector, [])
        if not keywords:
            logger.warning("trends_provider.unknown_sector", sector=sector)
            return pd.DataFrame()

        loop = asyncio.get_event_loop()
        try:
            result = await loop.run_in_executor(None, self._fetch_sync, keywords, timeframe)
            logger.info(
                "trends_provider.fetched",
                sector=sector,
                rows=len(result),
            )
            return result
        except Exception as exc:
            logger.error("trends_provider.fetch_error", sector=sector, error=str(exc))
            return pd.DataFrame()

    async def fetch_all_sectors(self) -> dict[str, pd.DataFrame]:
        """Fetch trends for all 5 sectors with a 1s delay between requests."""
        results: dict[str, pd.DataFrame] = {}
        for sector in SECTOR_KEYWORDS:
            results[sector] = await self.fetch_sector_trends(sector)
            await asyncio.sleep(1.0)
        return results

    def symbol_to_sector(self, symbol: str) -> str | None:
        return SYMBOL_TO_SECTOR.get(symbol)

    async def health_check(self) -> bool:
        try:
            await self.fetch_sector_trends("banking", timeframe="now 1-d")
            return True
        except Exception:
            return False
