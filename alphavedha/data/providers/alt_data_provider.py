"""Alternative data provider — sector-level macro indicators.

Sources: SIAM (auto sales), CMA (cement dispatch), IHS Markit (PMI),
RBI (credit growth). Most are monthly releases with known publication dates.
Manual data import supported via CSV/JSON.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import date

import httpx
import structlog

logger = structlog.get_logger(__name__)

ALT_DATA_TYPES = [
    "auto_sales",
    "cement_dispatch",
    "pmi_manufacturing",
    "pmi_services",
    "credit_growth",
    "power_consumption",
    "upi_volume",
    "gst_collections",
]

SECTOR_MAPPING: dict[str, list[str]] = {
    "auto_sales": [
        "MARUTI.NS", "TATAMOTORS.NS", "M&M.NS", "BAJAJ-AUTO.NS",
        "EICHERMOT.NS", "HEROMOTOCO.NS",
    ],
    "cement_dispatch": [
        "ULTRACEMCO.NS", "SHREECEM.NS", "AMBUJACEM.NS", "ACC.NS",
    ],
    "pmi_manufacturing": [],  # market-wide
    "pmi_services": [],
    "credit_growth": [
        "HDFCBANK.NS", "ICICIBANK.NS", "SBIN.NS", "KOTAKBANK.NS",
        "AXISBANK.NS", "BANKBARODA.NS",
    ],
    "power_consumption": [
        "NTPC.NS", "POWERGRID.NS", "TATAPOWER.NS",
    ],
}


@dataclass
class AltDataRecord:
    data_type: str
    period_date: date
    value: float
    yoy_change: float | None = None
    sector: str | None = None
    source: str | None = None


class AltDataProvider:
    """Fetches and manages alternative data indicators."""

    def __init__(self, timeout: float = 30.0) -> None:
        self._timeout = timeout

    async def fetch_auto_sales(self) -> list[AltDataRecord]:
        """Attempt to fetch monthly auto sales from SIAM/public sources.

        In practice this data is released via press releases.
        Falls back to empty list — use build_from_manual() for data entry.
        """
        logger.info("alt_data_auto_sales_fetch", msg="Auto sales requires manual data entry")
        return []

    async def fetch_pmi(self) -> list[AltDataRecord]:
        """Attempt to fetch PMI data.

        PMI is published by S&P Global/IHS Markit. No free API.
        Falls back to empty — use build_from_manual().
        """
        logger.info("alt_data_pmi_fetch", msg="PMI requires manual data entry")
        return []

    async def fetch_crude_oil(self, days: int = 365) -> list[AltDataRecord]:
        """Fetch crude oil prices via yfinance (free, no key needed)."""
        try:
            import yfinance as yf
            from datetime import timedelta

            end = date.today()
            start = end - timedelta(days=days)
            data = yf.download("BZ=F", start=str(start), end=str(end), progress=False)

            if data.empty:
                return []

            if hasattr(data.columns, "levels"):
                data.columns = data.columns.get_level_values(0)

            records: list[AltDataRecord] = []
            for idx, row in data.iterrows():
                dt = idx.date() if hasattr(idx, "date") else idx
                records.append(AltDataRecord(
                    data_type="crude_oil",
                    period_date=dt,
                    value=float(row["Close"]),
                    sector="Energy",
                    source="yfinance",
                ))
            return records
        except Exception as e:
            logger.warning("alt_data_crude_failed", error=str(e))
            return []

    async def fetch_all(self) -> dict[str, list[AltDataRecord]]:
        """Fetch all available alternative data."""
        results: dict[str, list[AltDataRecord]] = {}

        crude = await self.fetch_crude_oil()
        if crude:
            results["crude_oil"] = crude

        return results


def build_from_manual(records: list[dict]) -> list[AltDataRecord]:
    """Build AltDataRecords from manual CSV/JSON input.

    Expected keys: data_type, period_date (YYYY-MM-DD or date),
                   value, yoy_change (optional), sector (optional), source (optional)
    """
    results: list[AltDataRecord] = []
    for row in records:
        try:
            pd = row["period_date"]
            if isinstance(pd, str):
                pd = date.fromisoformat(pd)

            yoy = row.get("yoy_change")
            if yoy is not None:
                yoy = float(yoy)

            results.append(AltDataRecord(
                data_type=row["data_type"],
                period_date=pd,
                value=float(row["value"]),
                yoy_change=yoy,
                sector=row.get("sector"),
                source=row.get("source", "manual"),
            ))
        except (KeyError, ValueError) as e:
            logger.warning("alt_data_manual_parse_failed", error=str(e))

    return results


def get_relevant_symbols(data_type: str) -> list[str]:
    """Get list of symbols affected by a given alternative data type."""
    return SECTOR_MAPPING.get(data_type, [])
