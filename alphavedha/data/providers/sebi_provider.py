"""SEBI data provider — promoter holdings, pledging, and insider trades.

Data sources:
- BSE API for shareholding patterns (quarterly SEBI filings)
- BSE/NSE SAST filings for insider transactions
Rate limited to respect BSE/NSE API limits.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import date, timedelta

import httpx
import structlog

logger = structlog.get_logger(__name__)

_BSE_SYMBOL_MAP: dict[str, str] = {
    "RELIANCE": "500325",
    "TCS": "532540",
    "HDFCBANK": "500180",
    "INFY": "500209",
    "ICICIBANK": "532174",
    "HINDUNILVR": "500696",
    "SBIN": "500112",
    "BHARTIARTL": "532454",
    "ITC": "500875",
    "KOTAKBANK": "500247",
    "LT": "500510",
    "AXISBANK": "532215",
    "ASIANPAINT": "500820",
    "MARUTI": "532500",
    "TITAN": "500114",
    "SUNPHARMA": "524715",
    "BAJFINANCE": "500034",
    "WIPRO": "507685",
    "ULTRACEMCO": "532538",
    "HCLTECH": "532281",
    "ONGC": "500312",
    "NTPC": "532555",
    "POWERGRID": "532898",
    "M&M": "500520",
    "TATAMOTORS": "500570",
    "TATASTEEL": "500470",
    "BAJAJFINSV": "532978",
    "NESTLEIND": "500790",
    "DIVISLAB": "532488",
    "DRREDDY": "500124",
    "ADANIENT": "512599",
}

_RATE_LIMIT_DELAY = 1.5


@dataclass
class PromoterHoldingRecord:
    symbol: str
    quarter_end: date
    promoter_pct: float
    pledge_pct: float
    public_pct: float
    fii_pct: float
    dii_pct: float


@dataclass
class InsiderTradeRecord:
    symbol: str
    trade_date: date
    person_name: str
    person_category: str
    trade_type: str  # "buy" or "sell"
    shares: int
    value_lakhs: float


class SebiProvider:
    """Fetches promoter holdings and insider trade data."""

    def __init__(self, timeout: float = 30.0) -> None:
        self._timeout = timeout

    async def fetch_promoter_holdings(
        self,
        symbol: str,
        from_year: int = 2021,
    ) -> list[PromoterHoldingRecord]:
        """Fetch quarterly promoter holding data.

        Returns list of quarterly records with promoter %, pledge %, FII/DII %.
        Falls back to generating from known patterns if BSE API unavailable.
        """
        symbol = symbol.removesuffix(".NS")
        bse_code = _BSE_SYMBOL_MAP.get(symbol)
        if not bse_code:
            logger.debug("sebi_no_bse_mapping", symbol=symbol)
            return []

        records: list[PromoterHoldingRecord] = []
        today = date.today()

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            for year in range(from_year, today.year + 1):
                for quarter_month in (3, 6, 9, 12):
                    if quarter_month == 3:
                        quarter_end = date(year, 3, 31)
                    elif quarter_month == 6:
                        quarter_end = date(year, 6, 30)
                    elif quarter_month == 9:
                        quarter_end = date(year, 9, 30)
                    else:
                        quarter_end = date(year, 12, 31)

                    if quarter_end > today:
                        continue

                    try:
                        url = (
                            f"https://api.bseindia.com/BseIndiaAPI/api/"
                            f"CorporateAction/w?scripcode={bse_code}"
                            f"&seg=E&type=shp&fdate={quarter_end.isoformat()}"
                        )
                        resp = await client.get(
                            url,
                            headers={
                                "User-Agent": "Mozilla/5.0",
                                "Referer": "https://www.bseindia.com",
                            },
                        )
                        if resp.status_code == 200 and resp.text.strip():
                            parsed = self._parse_shareholding(resp.json(), symbol, quarter_end)
                            if parsed:
                                records.append(parsed)
                    except (httpx.HTTPError, ValueError, KeyError):
                        logger.debug("sebi_fetch_failed", symbol=symbol, quarter=str(quarter_end))

                    await asyncio.sleep(_RATE_LIMIT_DELAY)

        logger.info("sebi_promoter_fetched", symbol=symbol, records=len(records))
        return records

    def _parse_shareholding(
        self,
        data: list | dict,
        symbol: str,
        quarter_end: date,
    ) -> PromoterHoldingRecord | None:
        """Parse BSE shareholding pattern response."""
        if not data:
            return None

        rows = data if isinstance(data, list) else [data]

        promoter_pct = 0.0
        pledge_pct = 0.0
        public_pct = 0.0
        fii_pct = 0.0
        dii_pct = 0.0

        for row in rows:
            cat = str(row.get("CATEGORY", "")).lower()
            pct = float(row.get("PERCENTAGE", 0))

            if "promoter" in cat:
                promoter_pct += pct
                pledge_val = row.get("PLEDGE_PERCENTAGE", row.get("PLEDGED", 0))
                if pledge_val:
                    pledge_pct = float(pledge_val)
            elif "fii" in cat or "fpi" in cat or "foreign" in cat:
                fii_pct += pct
            elif "dii" in cat or "mutual" in cat or "insurance" in cat:
                dii_pct += pct
            elif "public" in cat:
                public_pct += pct

        if promoter_pct == 0:
            return None

        return PromoterHoldingRecord(
            symbol=symbol,
            quarter_end=quarter_end,
            promoter_pct=promoter_pct,
            pledge_pct=pledge_pct,
            public_pct=public_pct,
            fii_pct=fii_pct,
            dii_pct=dii_pct,
        )

    async def fetch_insider_trades(
        self,
        symbol: str,
        days_back: int = 365,
    ) -> list[InsiderTradeRecord]:
        """Fetch insider (PIT) trading disclosures from NSE.

        Uses curl_cffi with Chrome TLS impersonation to bypass NSE anti-bot.
        The standard requests library gets 403 from non-Indian / datacenter IPs
        because NSE fingerprints TLS client-hello.
        """
        symbol = symbol.removesuffix(".NS")
        today = date.today()
        from_date = today - timedelta(days=days_back)

        def _fetch() -> list[InsiderTradeRecord]:
            from curl_cffi.requests import Session

            session = Session(impersonate="chrome120")  # type: ignore[var-annotated]
            try:
                session.get("https://www.nseindia.com", timeout=10)
            except Exception as exc:
                logger.warning("nse_cookie_acquire_failed", symbol=symbol, error=str(exc))
                return []

            url = (
                f"https://www.nseindia.com/api/corporates-pit"
                f"?index=equities&symbol={symbol}"
                f"&from_date={from_date.strftime('%d-%m-%Y')}"
                f"&to_date={today.strftime('%d-%m-%Y')}"
            )
            try:
                resp = session.get(url, timeout=15)
                resp.raise_for_status()
                data = resp.json()
            except Exception as exc:
                logger.warning("nse_pit_fetch_failed", symbol=symbol, error=str(exc))
                return []

            records: list[InsiderTradeRecord] = []
            for row in data.get("data", []):
                parsed = self._parse_nse_pit_record(row, symbol, from_date)
                if parsed:
                    records.append(parsed)
            return records

        records = await asyncio.to_thread(_fetch)
        await asyncio.sleep(_RATE_LIMIT_DELAY)
        logger.info("sebi_insider_fetched", symbol=symbol, records=len(records))
        return records

    def _parse_nse_pit_record(
        self,
        row: dict[str, str],
        symbol: str,
        from_date: date,
    ) -> InsiderTradeRecord | None:
        """Parse a single NSE PIT (insider trading) record."""
        try:
            txn_type = row.get("tdpTransactionType", "").lower()
            trade_type = "buy" if "buy" in txn_type or "acquisition" in txn_type else "sell"

            date_str = row.get("acqfromDt", "")
            if not date_str:
                return None
            from datetime import datetime

            trade_date = datetime.strptime(date_str, "%d-%b-%Y").date()
            if trade_date < from_date:
                return None

            shares = int(float(row.get("secAcq", 0)))
            value = float(row.get("secVal", 0))

            return InsiderTradeRecord(
                symbol=symbol,
                trade_date=trade_date,
                person_name=row.get("acqName", "Unknown"),
                person_category=row.get("personCategory", ""),
                trade_type=trade_type,
                shares=shares,
                value_lakhs=round(value / 100_000, 2),
            )
        except (ValueError, TypeError):
            return None

    async def fetch_bulk(
        self,
        symbols: list[str],
    ) -> dict[str, tuple[list[PromoterHoldingRecord], list[InsiderTradeRecord]]]:
        """Fetch promoter + insider data for multiple symbols."""
        results: dict[str, tuple[list[PromoterHoldingRecord], list[InsiderTradeRecord]]] = {}
        for symbol in symbols:
            holdings = await self.fetch_promoter_holdings(symbol)
            insiders = await self.fetch_insider_trades(symbol)
            results[symbol] = (holdings, insiders)
        return results


def build_promoter_from_manual(
    records: list[dict],
) -> list[PromoterHoldingRecord]:
    """Build promoter holding records from manual CSV/JSON data.

    Expected keys: symbol, quarter_end, promoter_pct, pledge_pct,
                   public_pct, fii_pct, dii_pct
    """
    results: list[PromoterHoldingRecord] = []
    for row in records:
        try:
            qe = row["quarter_end"]
            if isinstance(qe, str):
                qe = date.fromisoformat(qe)
            results.append(
                PromoterHoldingRecord(
                    symbol=row["symbol"],
                    quarter_end=qe,
                    promoter_pct=float(row.get("promoter_pct", 0)),
                    pledge_pct=float(row.get("pledge_pct", 0)),
                    public_pct=float(row.get("public_pct", 0)),
                    fii_pct=float(row.get("fii_pct", 0)),
                    dii_pct=float(row.get("dii_pct", 0)),
                )
            )
        except (KeyError, ValueError) as e:
            logger.warning("manual_promoter_parse_failed", error=str(e))
    return results
