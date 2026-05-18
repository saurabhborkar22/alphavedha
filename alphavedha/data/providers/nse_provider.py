"""NSE provider — FII/DII flows, derivatives data, and live market data from NSE India."""

from __future__ import annotations

import asyncio
from datetime import date, timedelta

import pandas as pd
import requests
import structlog

from alphavedha.data.providers.base import RateLimiter

logger = structlog.get_logger(__name__)

_NSE_BASE = "https://www.nseindia.com"
_HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.nseindia.com/",
}


class NSESession:
    """Manages NSE website session with cookie refresh."""

    def __init__(self) -> None:
        self._session = requests.Session()
        self._session.headers.update(_HEADERS)
        self._cookies_valid = False

    def _refresh_cookies(self) -> None:
        try:
            resp = self._session.get(_NSE_BASE, timeout=10)
            resp.raise_for_status()
            self._cookies_valid = True
        except Exception as e:
            logger.warning("nse_cookie_refresh_failed", error=str(e))
            self._cookies_valid = False

    def get(self, url: str, retries: int = 2) -> requests.Response:
        for attempt in range(retries + 1):
            if not self._cookies_valid:
                self._refresh_cookies()
            try:
                resp = self._session.get(url, timeout=15)
                if resp.status_code == 401 or resp.status_code == 403:
                    self._cookies_valid = False
                    continue
                resp.raise_for_status()
                return resp
            except requests.RequestException:
                self._cookies_valid = False
                if attempt == retries:
                    raise
        raise requests.RequestException(f"Failed after {retries + 1} attempts: {url}")


class NSEProvider:
    """Fetch FII/DII flows and derivatives data from NSE India."""

    def __init__(self) -> None:
        self._session = NSESession()
        self._rate_limiter = RateLimiter(requests_per_second=0.5)

    @property
    def name(self) -> str:
        return "nse"

    async def fetch_fii_dii_today(self) -> list[dict]:
        """Fetch today's FII/DII data from NSE API."""
        await self._rate_limiter.acquire()

        def _fetch() -> list[dict]:
            url = f"{_NSE_BASE}/api/fiidiiTradeReact"
            resp = self._session.get(url)
            return resp.json()

        return await asyncio.to_thread(_fetch)

    async def fetch_fii_dii_range(
        self, start: date, end: date
    ) -> pd.DataFrame:
        """Fetch FII/DII data for a date range by iterating daily.

        NSE only provides today's data via API, so for historical data
        we fetch day by day. For bulk historical backfill, use the
        backfill_fii_dii_from_nsdl method instead.
        """
        await self._rate_limiter.acquire()

        def _fetch_range() -> pd.DataFrame:
            rows: list[dict] = []
            current = start
            session = NSESession()

            while current <= end:
                if current.weekday() >= 5:
                    current += timedelta(days=1)
                    continue
                try:
                    url = f"{_NSE_BASE}/api/fiidiiTradeReact"
                    resp = session.get(url)
                    data = resp.json()
                    for item in data:
                        rows.append({
                            "date": pd.to_datetime(item["date"], format="%d-%b-%Y"),
                            "category": item["category"].upper(),
                            "buy_value": float(item["buyValue"].replace(",", "")),
                            "sell_value": float(item["sellValue"].replace(",", "")),
                            "net_value": float(item["netValue"].replace(",", "")),
                        })
                except Exception as e:
                    logger.warning("nse_fii_dii_fetch_error", date=str(current), error=str(e))
                current += timedelta(days=1)

            if not rows:
                return pd.DataFrame()
            return pd.DataFrame(rows)

        return await asyncio.to_thread(_fetch_range)

    async def fetch_stock_fno_quote(self, symbol: str) -> dict:
        """Fetch live F&O quote for a stock (OI, IV, chain data)."""
        await self._rate_limiter.acquire()

        def _fetch() -> dict:
            from jugaad_data.nse import NSELive
            live = NSELive()
            return live.stock_quote_fno(symbol)

        return await asyncio.to_thread(_fetch)

    async def fetch_option_chain(self, symbol: str) -> dict:
        """Fetch options chain for a stock."""
        await self._rate_limiter.acquire()

        def _fetch() -> dict:
            from jugaad_data.nse import NSELive
            live = NSELive()
            return live.equities_option_chain(symbol)

        return await asyncio.to_thread(_fetch)

    async def fetch_trade_info(self, symbol: str) -> dict:
        """Fetch trade info (delivery data) for a stock."""
        await self._rate_limiter.acquire()

        def _fetch() -> dict:
            from jugaad_data.nse import NSELive
            live = NSELive()
            return live.trade_info(symbol)

        return await asyncio.to_thread(_fetch)

    async def health_check(self) -> bool:
        try:
            data = await self.fetch_fii_dii_today()
            return len(data) > 0
        except Exception:
            return False


def _normalize_category(raw: str) -> str:
    """Normalize FII/DII category names from NSE API."""
    upper = raw.upper().strip()
    if "FII" in upper or "FPI" in upper:
        return "FII"
    if "DII" in upper:
        return "DII"
    return upper


def parse_fii_dii_response(data: list[dict]) -> list[dict]:
    """Parse NSE FII/DII API response into DB-ready rows."""
    rows = []
    for item in data:
        try:
            rows.append({
                "date": pd.to_datetime(item["date"], format="%d-%b-%Y").date(),
                "category": _normalize_category(item["category"]),
                "buy_value": float(str(item["buyValue"]).replace(",", "")),
                "sell_value": float(str(item["sellValue"]).replace(",", "")),
                "net_value": float(str(item["netValue"]).replace(",", "")),
            })
        except (KeyError, ValueError) as e:
            logger.warning("nse_parse_fii_dii_error", item=item, error=str(e))
    return rows


def parse_fno_to_derivatives(fno_data: dict, symbol: str, trade_date: date) -> dict:
    """Parse NSE F&O quote into derivatives_data DB row."""
    result: dict = {
        "symbol": symbol,
        "date": trade_date,
        "futures_oi": None,
        "futures_price": None,
        "options_data_json": None,
    }

    if not isinstance(fno_data, dict) or "data" not in fno_data:
        return result

    entries = fno_data["data"]
    if not isinstance(entries, list):
        return result

    for entry in entries:
        inst_type = entry.get("instrumentType", "")
        if inst_type == "FUTSTK":
            result["futures_oi"] = entry.get("openInterest")
            result["futures_price"] = entry.get("lastPrice")
            break

    chain: list[dict] = []
    for entry in entries:
        inst_type = entry.get("instrumentType", "")
        if inst_type in ("OPTSTK", "CE", "PE"):
            chain.append({
                "strike": entry.get("strikePrice"),
                "call_oi": entry.get("openInterest") if "CE" in entry.get("identifier", "") else 0,
                "put_oi": entry.get("openInterest") if "PE" in entry.get("identifier", "") else 0,
                "call_vol": entry.get("totalTradedVolume") if "CE" in entry.get("identifier", "") else 0,
                "put_vol": entry.get("totalTradedVolume") if "PE" in entry.get("identifier", "") else 0,
                "call_iv": entry.get("impliedVolatility"),
            })

    if chain:
        result["options_data_json"] = {"chain": chain}

    return result
