"""ASM/GSM surveillance list collector — daily snapshots from NSE.

Fetches the current ASM (Additional Surveillance Measure) and GSM (Graded
Surveillance Measure) lists from NSE, diffs against the DB to detect
add/remove transitions, and stores results in ``surveillance_flags``.
"""

from __future__ import annotations

import asyncio
from datetime import date
from typing import Any

import structlog

from alphavedha.data.providers.base import RateLimiter
from alphavedha.data.providers.nse_provider import NSESession

logger = structlog.get_logger(__name__)

_ASM_URL = "https://www.nseindia.com/api/reportASM"
_GSM_URL = "https://www.nseindia.com/api/reportGSM"

_rate_limiter = RateLimiter(requests_per_second=0.5)


def _parse_asm_response(data: dict[str, Any]) -> list[dict[str, Any]]:
    """Parse ASM API response into surveillance flag rows."""
    rows: list[dict[str, Any]] = []
    today = date.today()

    for term in ("longterm", "shortterm"):
        section = data.get(term, {})
        if not isinstance(section, dict):
            continue
        items = section.get("data", [])
        if not isinstance(items, list):
            continue

        for item in items:
            symbol = (item.get("symbol", "") or "").strip()
            if not symbol:
                continue

            stage = (item.get("asmSurvIndicator", "") or "").strip()
            surv_code = (item.get("survCode", "") or "").strip()
            prefix = "LTASM" if term == "longterm" else "STASM"
            list_name = f"{prefix}-{stage}" if stage else prefix

            rows.append(
                {
                    "symbol": f"{symbol}.NS",
                    "list_name": list_name[:20],
                    "added_on": today,
                    "surv_code": surv_code,
                }
            )

    return rows


def _parse_gsm_response(data: list[dict[str, Any]] | Any) -> list[dict[str, Any]]:
    """Parse GSM API response into surveillance flag rows."""
    rows: list[dict[str, Any]] = []
    today = date.today()

    if not isinstance(data, list):
        return rows

    for item in data:
        symbol = (item.get("symbol", "") or "").strip()
        if not symbol:
            continue

        stage = (item.get("gsmSurvIndicator", "") or "").strip()
        list_name = f"GSM-{stage}" if stage else "GSM"

        rows.append(
            {
                "symbol": f"{symbol}.NS",
                "list_name": list_name[:20],
                "added_on": today,
            }
        )

    return rows


def _fetch_asm_sync(session: NSESession) -> dict[str, Any]:
    """Fetch ASM list from NSE. Blocking."""
    resp = session.get(_ASM_URL)
    data = resp.json()
    if isinstance(data, dict):
        return data
    return {}


def _fetch_gsm_sync(session: NSESession) -> list[dict[str, Any]]:
    """Fetch GSM list from NSE. Blocking."""
    resp = session.get(_GSM_URL)
    data = resp.json()
    if isinstance(data, list):
        return data
    return []


async def collect_surveillance_lists() -> list[dict[str, Any]]:
    """Fetch current ASM + GSM lists from NSE and parse into flag rows."""
    await _rate_limiter.acquire()
    session = NSESession()

    asm_data = await asyncio.to_thread(_fetch_asm_sync, session)
    asm_rows = _parse_asm_response(asm_data)

    await _rate_limiter.acquire()
    gsm_data = await asyncio.to_thread(_fetch_gsm_sync, session)
    gsm_rows = _parse_gsm_response(gsm_data)

    all_rows = asm_rows + gsm_rows

    logger.info(
        "surveillance_collect_complete",
        asm=len(asm_rows),
        gsm=len(gsm_rows),
        total=len(all_rows),
    )
    return all_rows


async def ingest_surveillance_daily() -> int:
    """Daily ingestion: fetch ASM/GSM, store flags, return count stored."""
    from alphavedha.intel.store import store_surveillance_flags

    rows = await collect_surveillance_lists()
    if not rows:
        logger.info("surveillance_no_flags")
        return 0

    store_rows = [{k: v for k, v in r.items() if k != "surv_code"} for r in rows]
    return await store_surveillance_flags(store_rows)
