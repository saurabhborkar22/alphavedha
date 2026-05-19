"""Earnings provider — fetch quarterly results from public sources.

Uses screener.in consolidated financials page and BSE corporate announcements.
Graceful degradation: returns empty DataFrame if source unavailable.
"""

from __future__ import annotations

import asyncio
from datetime import date
from typing import Any

import pandas as pd
import requests
import structlog

from alphavedha.data.providers.base import RateLimiter

logger = structlog.get_logger(__name__)

_SCREENER_BASE = "https://www.screener.in"
_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/json",
}

_SYMBOL_TO_SCREENER_SLUG: dict[str, str] = {
    "RELIANCE": "RELIANCE",
    "TCS": "TCS",
    "HDFCBANK": "HDFCBANK",
    "INFY": "INFY",
    "ICICIBANK": "ICICIBANK",
    "HINDUNILVR": "HINDUNILVR",
    "BHARTIARTL": "BHARTIARTL",
    "ITC": "ITC",
    "SBIN": "SBIN",
    "BAJFINANCE": "BAJFINANCE",
    "KOTAKBANK": "KOTAKBANK",
    "LT": "LT",
    "AXISBANK": "AXISBANK",
    "WIPRO": "WIPRO",
    "HCLTECH": "HCLTECH",
    "ASIANPAINT": "ASIANPAINT",
    "MARUTI": "MARUTI",
    "SUNPHARMA": "SUNPHARMA",
    "TITAN": "TITAN",
    "TATAMOTORS": "TATAMOTORS",
    "ULTRACEMCO": "ULTRACEMCO",
    "NTPC": "NTPC",
    "TATASTEEL": "TATASTEEL",
    "POWERGRID": "POWERGRID",
    "ONGC": "ONGC",
    "NESTLEIND": "NESTLEIND",
    "BAJAJFINSV": "BAJAJFINSV",
    "INDUSINDBK": "INDUSINDBK",
    "M&M": "M&M",
    "ADANIENT": "ADANIENT",
    "ADANIPORTS": "ADANIPORTS",
    "COALINDIA": "COALINDIA",
    "JSWSTEEL": "JSWSTEEL",
    "TECHM": "TECHM",
    "BAJAJ-AUTO": "BAJAJ-AUTO",
    "GRASIM": "GRASIM",
    "BPCL": "BPCL",
    "HEROMOTOCO": "HEROMOTOCO",
    "DIVISLAB": "DIVISLAB",
    "DRREDDY": "DRREDDY",
    "BRITANNIA": "BRITANNIA",
    "CIPLA": "CIPLA",
    "EICHERMOT": "EICHERMOT",
    "APOLLOHOSP": "APOLLOHOSP",
    "TATACONSUM": "TATACONSUM",
    "SBILIFE": "SBILIFE",
    "HDFCLIFE": "HDFCLIFE",
    "HINDALCO": "HINDALCO",
    "SHRIRAMFIN": "SHRIRAMFIN",
    "WIPRO": "WIPRO",
}


def _strip_suffix(symbol: str) -> str:
    for suffix in (".NS", ".BO"):
        if symbol.endswith(suffix):
            return symbol[: -len(suffix)]
    return symbol


def _parse_quarter_label(label: str) -> tuple[int, int] | None:
    """Parse screener quarter label like 'Mar 2024' into (quarter, year)."""
    parts = label.strip().split()
    if len(parts) != 2:
        return None
    month_map = {
        "Mar": 4, "Jun": 1, "Sep": 2, "Dec": 3,
    }
    month_str, year_str = parts
    quarter = month_map.get(month_str)
    if quarter is None:
        return None
    try:
        year = int(year_str)
    except ValueError:
        return None
    return quarter, year


def _quarter_end_date(quarter: int, year: int) -> date:
    """Return the last date of the quarter."""
    quarter_ends = {
        1: (6, 30),
        2: (9, 30),
        3: (12, 31),
        4: (3, 31),
    }
    month, day = quarter_ends[quarter]
    if quarter == 4:
        year = year + 1
    return date(year, month, day)


def _announced_date_estimate(quarter: int, year: int) -> date:
    """Estimate announcement date (typically 30-45 days after quarter end)."""
    end = _quarter_end_date(quarter, year)
    return date(end.year, end.month, min(28, end.day)) + pd.Timedelta(days=45)


class EarningsProvider:
    """Fetch quarterly earnings data from screener.in."""

    def __init__(self) -> None:
        self._session = requests.Session()
        self._session.headers.update(_HEADERS)
        self._rate_limiter = RateLimiter(requests_per_second=0.3)

    @property
    def name(self) -> str:
        return "earnings"

    async def fetch_quarterly_results(
        self, symbol: str,
    ) -> list[dict[str, Any]]:
        """Fetch quarterly P&L for a symbol from screener.in."""
        await self._rate_limiter.acquire()

        nse_symbol = _strip_suffix(symbol)
        slug = _SYMBOL_TO_SCREENER_SLUG.get(nse_symbol, nse_symbol)

        def _fetch() -> list[dict[str, Any]]:
            url = f"{_SCREENER_BASE}/api/company/{slug}/quarterly/"
            try:
                resp = self._session.get(url, timeout=15)
                if resp.status_code == 404:
                    logger.warning("earnings_not_found", symbol=nse_symbol)
                    return []
                resp.raise_for_status()
                return _parse_screener_quarterly(resp.json(), nse_symbol)
            except requests.RequestException as e:
                logger.warning("earnings_fetch_failed", symbol=nse_symbol, error=str(e))
                return []

        return await asyncio.to_thread(_fetch)

    async def fetch_bulk(
        self, symbols: list[str],
    ) -> dict[str, list[dict[str, Any]]]:
        """Fetch earnings for multiple symbols sequentially (rate-limited)."""
        results: dict[str, list[dict[str, Any]]] = {}
        for sym in symbols:
            try:
                data = await self.fetch_quarterly_results(sym)
                results[sym] = data
            except Exception as e:
                logger.error("earnings_bulk_error", symbol=sym, error=str(e))
                results[sym] = []
        return results


def _parse_screener_quarterly(
    data: dict | list, symbol: str,
) -> list[dict[str, Any]]:
    """Parse screener.in quarterly API response into earnings rows."""
    results: list[dict[str, Any]] = []

    if isinstance(data, dict):
        quarters = data.get("quarters", [])
        revenue_row = _find_row(data, "Sales")
        expenses_row = _find_row(data, "Expenses")
        profit_row = _find_row(data, "Net Profit")
        opm_row = _find_row(data, "OPM")
    else:
        return results

    if not quarters or not revenue_row:
        return results

    for i, q_label in enumerate(quarters):
        parsed = _parse_quarter_label(q_label)
        if parsed is None:
            continue
        quarter, year = parsed

        revenue = _safe_float(revenue_row, i)
        profit = _safe_float(profit_row, i) if profit_row else None
        expenses = _safe_float(expenses_row, i) if expenses_row else None

        results.append({
            "symbol": symbol,
            "quarter": quarter,
            "year": year,
            "revenue_actual": revenue,
            "revenue_estimate": None,
            "revenue_surprise_pct": None,
            "profit_actual": profit,
            "profit_estimate": None,
            "profit_surprise_pct": None,
            "expenses": expenses,
            "announced_date": _announced_date_estimate(quarter, year),
        })

    return results


def _find_row(data: dict, key: str) -> list | None:
    """Find a row by name in screener quarterly data."""
    for row in data.get("data", []):
        if isinstance(row, dict) and row.get("name") == key:
            return row.get("values", [])
        if isinstance(row, list) and len(row) > 0 and row[0] == key:
            return row[1:]
    return None


def _safe_float(values: list, index: int) -> float | None:
    """Safely extract float from a list at index."""
    if index >= len(values):
        return None
    val = values[index]
    if val is None or val == "":
        return None
    try:
        if isinstance(val, str):
            return float(val.replace(",", ""))
        return float(val)
    except (ValueError, TypeError):
        return None


def build_earnings_from_manual(
    records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Build earnings records from manually curated data (CSV/JSON).

    Expected keys: symbol, quarter, year, revenue_actual, profit_actual,
    revenue_estimate (optional), profit_estimate (optional), announced_date (optional).
    """
    results: list[dict[str, Any]] = []
    for r in records:
        entry = {
            "symbol": r["symbol"],
            "quarter": int(r["quarter"]),
            "year": int(r["year"]),
            "revenue_actual": r.get("revenue_actual"),
            "profit_actual": r.get("profit_actual"),
            "revenue_estimate": r.get("revenue_estimate"),
            "profit_estimate": r.get("profit_estimate"),
            "revenue_surprise_pct": None,
            "profit_surprise_pct": None,
            "expenses": r.get("expenses"),
            "announced_date": r.get("announced_date"),
        }

        if entry["revenue_actual"] and entry["revenue_estimate"]:
            est = entry["revenue_estimate"]
            if est != 0:
                entry["revenue_surprise_pct"] = (
                    (entry["revenue_actual"] - est) / abs(est)
                ) * 100.0

        if entry["profit_actual"] and entry["profit_estimate"]:
            est = entry["profit_estimate"]
            if est != 0:
                entry["profit_surprise_pct"] = (
                    (entry["profit_actual"] - est) / abs(est)
                ) * 100.0

        results.append(entry)
    return results
