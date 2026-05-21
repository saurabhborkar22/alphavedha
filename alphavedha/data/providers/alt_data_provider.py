"""Alternative data provider — sector-level macro indicators.

Sources: SIAM (auto sales), CMA (cement dispatch), IHS Markit (PMI),
RBI (credit growth), yfinance (crude oil, US futures), manual CSV imports.
Most are monthly releases with known publication dates.
Manual data import supported via CSV/JSON.
"""

from __future__ import annotations

import asyncio
import csv
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

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
    "port_cargo",
    "forex_reserves",
    "crude_oil",
    "us_overnight",
    "job_postings",
]

SECTOR_MAPPING: dict[str, list[str]] = {
    "auto_sales": [
        "MARUTI.NS",
        "TATAMOTORS.NS",
        "M&M.NS",
        "BAJAJ-AUTO.NS",
        "EICHERMOT.NS",
        "HEROMOTOCO.NS",
    ],
    "cement_dispatch": [
        "ULTRACEMCO.NS",
        "SHREECEM.NS",
        "AMBUJACEM.NS",
        "ACC.NS",
    ],
    "pmi_manufacturing": [],  # market-wide
    "pmi_services": [],
    "credit_growth": [
        "HDFCBANK.NS",
        "ICICIBANK.NS",
        "SBIN.NS",
        "KOTAKBANK.NS",
        "AXISBANK.NS",
        "BANKBARODA.NS",
    ],
    "power_consumption": [
        "NTPC.NS",
        "POWERGRID.NS",
        "TATAPOWER.NS",
    ],
    "upi_volume": [],
    "gst_collections": [],
    "port_cargo": ["ADANIPORTS.NS"],
    "forex_reserves": [
        "TCS.NS",
        "INFY.NS",
        "WIPRO.NS",
        "HCLTECH.NS",
        "TECHM.NS",
        "LTIM.NS",
    ],
    "crude_oil": ["RELIANCE.NS", "ONGC.NS", "BPCL.NS", "COALINDIA.NS"],
    "us_overnight": [],  # applies to all
    "job_postings": [
        "TCS.NS",
        "INFY.NS",
        "WIPRO.NS",
        "HCLTECH.NS",
        "HDFCBANK.NS",
        "ICICIBANK.NS",
        "SBIN.NS",
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

    async def fetch_crude_oil(
        self,
        start: date | None = None,
        end: date | None = None,
        days: int = 365,
    ) -> list[AltDataRecord]:
        """Fetch daily crude oil (Brent) prices via yfinance BZ=F."""
        try:
            import yfinance as yf

            if end is None:
                end = date.today()
            if start is None:
                start = end - timedelta(days=days)

            data = yf.download("BZ=F", start=str(start), end=str(end), progress=False)

            if data.empty:
                return []

            if isinstance(data.columns, __import__("pandas").MultiIndex):
                data.columns = data.columns.get_level_values(0)

            records: list[AltDataRecord] = []
            for idx, row in data.iterrows():
                dt = idx.date() if hasattr(idx, "date") else idx
                records.append(
                    AltDataRecord(
                        data_type="crude_oil",
                        period_date=dt,
                        value=float(row["Close"]),
                        sector="Energy",
                        source="yfinance",
                    )
                )
            return records
        except Exception as e:
            logger.warning("alt_data_crude_failed", error=str(e))
            return []

    async def fetch_us_overnight(
        self,
        start: date | None = None,
        end: date | None = None,
        days: int = 365,
    ) -> list[AltDataRecord]:
        """Fetch S&P 500 futures overnight return via yfinance ES=F."""
        try:
            import yfinance as yf

            if end is None:
                end = date.today()
            if start is None:
                start = end - timedelta(days=days)

            data = yf.download("ES=F", start=str(start), end=str(end), progress=False)

            if data.empty:
                return []

            if isinstance(data.columns, __import__("pandas").MultiIndex):
                data.columns = data.columns.get_level_values(0)

            records: list[AltDataRecord] = []
            closes = data["Close"]
            for i, (idx, row) in enumerate(data.iterrows()):
                dt = idx.date() if hasattr(idx, "date") else idx
                current_close = float(row["Close"])
                if i > 0:
                    prev_close = float(closes.iloc[i - 1])
                    overnight_return = (
                        (current_close - prev_close) / prev_close if prev_close != 0 else 0.0
                    )
                else:
                    overnight_return = 0.0
                records.append(
                    AltDataRecord(
                        data_type="us_overnight",
                        period_date=dt,
                        value=overnight_return,
                        sector="Global",
                        source="yfinance",
                    )
                )
            return records
        except Exception as e:
            logger.warning("alt_data_us_overnight_failed", error=str(e))
            return []

    async def fetch_forex_reserves(
        self,
        start: date | None = None,
        end: date | None = None,
    ) -> list[AltDataRecord]:
        """Fetch RBI forex reserves (weekly, manual CSV import supported).

        Returns empty — use load_from_csv() with a CSV containing
        columns: period_date, value, source.
        """
        logger.info(
            "alt_data_forex_reserves_fetch",
            msg="Forex reserves requires manual CSV import via load_from_csv()",
        )
        return []

    async def fetch_port_cargo(
        self,
        start: date | None = None,
        end: date | None = None,
    ) -> list[AltDataRecord]:
        """Fetch port cargo data (monthly, manual CSV import).

        Returns empty — use load_from_csv() with a CSV containing
        columns: period_date, value, source.
        """
        logger.info(
            "alt_data_port_cargo_fetch",
            msg="Port cargo requires manual CSV import via load_from_csv()",
        )
        return []

    async def fetch_job_postings(
        self,
        start: date | None = None,
        end: date | None = None,
    ) -> list[AltDataRecord]:
        """Fetch Naukri job index (monthly, manual CSV import).

        Returns empty — use load_from_csv() with a CSV containing
        columns: period_date, value, source.
        """
        logger.info(
            "alt_data_job_postings_fetch",
            msg="Job postings requires manual CSV import via load_from_csv()",
        )
        return []

    async def fetch_all(
        self,
        start: date | None = None,
        end: date | None = None,
    ) -> dict[str, list[AltDataRecord]]:
        """Fetch all available alternative data sources.

        Gathers results from all fetchable sources concurrently.
        Manual-only sources (port_cargo, forex_reserves, job_postings)
        are included but return empty unless CSVs have been loaded.
        """
        tasks = {
            "crude_oil": self.fetch_crude_oil(start, end),
            "us_overnight": self.fetch_us_overnight(start, end),
            "forex_reserves": self.fetch_forex_reserves(start, end),
            "port_cargo": self.fetch_port_cargo(start, end),
            "job_postings": self.fetch_job_postings(start, end),
            "auto_sales": self.fetch_auto_sales(),
            "pmi": self.fetch_pmi(),
        }

        results: dict[str, list[AltDataRecord]] = {}
        gathered = await asyncio.gather(
            *tasks.values(),
            return_exceptions=True,
        )
        for key, result in zip(tasks.keys(), gathered, strict=False):
            if isinstance(result, Exception):
                logger.warning("alt_data_fetch_error", source=key, error=str(result))
                continue
            if result:
                results[key] = result

        return results


def load_from_csv(path: str | Path, data_type: str) -> list[AltDataRecord]:
    """Read a CSV file with columns: period_date, value, source.

    Useful for manual import of data not available via API
    (forex_reserves, port_cargo, job_postings).
    """
    csv_path = Path(path)
    if not csv_path.exists():
        logger.warning("alt_data_csv_not_found", path=str(csv_path), data_type=data_type)
        return []

    records: list[AltDataRecord] = []
    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                period_date = date.fromisoformat(row["period_date"].strip())
                value = float(row["value"])
                source = row.get("source", "csv").strip() or "csv"
                records.append(
                    AltDataRecord(
                        data_type=data_type,
                        period_date=period_date,
                        value=value,
                        sector=SECTOR_MAPPING.get(data_type, [None])[0]
                        if SECTOR_MAPPING.get(data_type)
                        else None,
                        source=source,
                    )
                )
            except (KeyError, ValueError) as e:
                logger.warning(
                    "alt_data_csv_row_parse_failed",
                    data_type=data_type,
                    error=str(e),
                    row=row,
                )

    logger.info("alt_data_csv_loaded", data_type=data_type, n_records=len(records))
    return records


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

            results.append(
                AltDataRecord(
                    data_type=row["data_type"],
                    period_date=pd,
                    value=float(row["value"]),
                    yoy_change=yoy,
                    sector=row.get("sector"),
                    source=row.get("source", "manual"),
                )
            )
        except (KeyError, ValueError) as e:
            logger.warning("alt_data_manual_parse_failed", error=str(e))

    return results


def get_relevant_symbols(data_type: str) -> list[str]:
    """Get list of symbols affected by a given alternative data type."""
    return SECTOR_MAPPING.get(data_type, [])
