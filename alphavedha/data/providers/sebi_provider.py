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
    "RELIANCE.NS": "500325",
    "TCS.NS": "532540",
    "HDFCBANK.NS": "500180",
    "INFY.NS": "500209",
    "ICICIBANK.NS": "532174",
    "HINDUNILVR.NS": "500696",
    "SBIN.NS": "500112",
    "BHARTIARTL.NS": "532454",
    "ITC.NS": "500875",
    "KOTAKBANK.NS": "500247",
    "LT.NS": "500510",
    "AXISBANK.NS": "532215",
    "ASIANPAINT.NS": "500820",
    "MARUTI.NS": "532500",
    "TITAN.NS": "500114",
    "SUNPHARMA.NS": "524715",
    "BAJFINANCE.NS": "500034",
    "WIPRO.NS": "507685",
    "ULTRACEMCO.NS": "532538",
    "HCLTECH.NS": "532281",
    "ONGC.NS": "500312",
    "NTPC.NS": "532555",
    "POWERGRID.NS": "532898",
    "M&M.NS": "500520",
    "TATAMOTORS.NS": "500570",
    "TATASTEEL.NS": "500470",
    "BAJAJFINSV.NS": "532978",
    "NESTLEIND.NS": "500790",
    "DIVISLAB.NS": "532488",
    "DRREDDY.NS": "500124",
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
        """Fetch insider (SAST) trading disclosures.

        Returns list of insider buy/sell records.
        """
        bse_code = _BSE_SYMBOL_MAP.get(symbol)
        if not bse_code:
            return []

        today = date.today()
        from_date = today - timedelta(days=days_back)
        records: list[InsiderTradeRecord] = []

        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                url = (
                    f"https://api.bseindia.com/BseIndiaAPI/api/"
                    f"InsiderTrading/w?scripcode={bse_code}"
                    f"&fromdate={from_date.strftime('%Y%m%d')}"
                    f"&todate={today.strftime('%Y%m%d')}"
                )
                resp = await client.get(
                    url,
                    headers={"User-Agent": "Mozilla/5.0", "Referer": "https://www.bseindia.com"},
                )
                if resp.status_code == 200 and resp.text.strip():
                    for row in resp.json():
                        parsed = self._parse_insider_trade(row, symbol)
                        if parsed:
                            records.append(parsed)
        except (httpx.HTTPError, ValueError):
            logger.debug("sebi_insider_fetch_failed", symbol=symbol)

        logger.info("sebi_insider_fetched", symbol=symbol, records=len(records))
        return records

    def _parse_insider_trade(
        self,
        row: dict,
        symbol: str,
    ) -> InsiderTradeRecord | None:
        """Parse a single insider trade row."""
        try:
            trade_type_raw = str(row.get("BUYSELL", row.get("TRANSACTIONTYPE", ""))).lower()
            trade_type = (
                "buy" if "buy" in trade_type_raw or "acquisition" in trade_type_raw else "sell"
            )

            date_str = row.get("TRADE_DATE", row.get("TRANSACTIONDATE", ""))
            if not date_str:
                return None

            from dateutil.parser import parse as parse_date

            trade_date = parse_date(str(date_str)).date()

            return InsiderTradeRecord(
                symbol=symbol,
                trade_date=trade_date,
                person_name=str(row.get("PERSONNAME", row.get("NAME", "Unknown"))),
                person_category=str(row.get("CATEGORY", row.get("PERSONCATEGORY", ""))),
                trade_type=trade_type,
                shares=int(float(row.get("NO_OF_SHARES", row.get("NOOFSHARES", 0)))),
                value_lakhs=float(row.get("VALUE", row.get("TRADEDVALUE", 0))),
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
