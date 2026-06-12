"""Fundamental analysis API endpoints — Beneish M-Score and Altman Z'-Score."""

from __future__ import annotations

from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

import structlog
from fastapi import APIRouter, Depends, HTTPException

from alphavedha.api.deps import verify_api_key
from alphavedha.fundamental.analyzer import analyze

logger = structlog.get_logger(__name__)
router = APIRouter(
    prefix="/fundamental",
    tags=["fundamental"],
    dependencies=[Depends(verify_api_key)],
)

IST = ZoneInfo("Asia/Kolkata")


def _beneish_dict(b: Any) -> dict[str, Any]:
    return {
        "m_score": b.m_score,
        "verdict": b.verdict,
        "interpretation": b.interpretation,
        "components": {
            "dsri": b.dsri,
            "gmi": b.gmi,
            "aqi": b.aqi,
            "sgi": b.sgi,
            "depi": b.depi,
            "sgai": b.sgai,
            "tata": b.tata,
            "lvgi": b.lvgi,
        },
    }


def _altman_dict(a: Any) -> dict[str, Any]:
    return {
        "z_score": a.z_score,
        "verdict": a.verdict,
        "interpretation": a.interpretation,
        "components": {
            "x1_working_capital_ratio": a.x1_working_capital_ratio,
            "x2_retained_earnings_ratio": a.x2_retained_earnings_ratio,
            "x3_ebit_ratio": a.x3_ebit_ratio,
            "x4_equity_to_liabilities": a.x4_equity_to_liabilities,
        },
    }


@router.get("/analyze/{symbol}")
async def fundamental_analyze(symbol: str) -> dict[str, Any]:
    """Run Beneish M-Score (earnings manipulation) and Altman Z'-Score (distress) analysis.

    Returns a combined fundamental health report for the given NSE symbol.
    Data is fetched from yfinance annual financial statements.

    Verdicts
    --------
    overall_verdict:
      - ``healthy``          : both scores in safe range
      - ``caution``          : one or both in grey zone
      - ``red_flag``         : manipulation or distress signals present
      - ``insufficient_data``: could not fetch financial statements

    beneish.verdict:
      - ``non_manipulator`` / ``grey_zone`` / ``manipulator``

    altman.verdict:
      - ``safe`` / ``grey_zone`` / ``distress``
    """
    symbol = symbol.upper().strip()
    if not symbol or len(symbol) > 20:
        raise HTTPException(status_code=400, detail=f"Invalid symbol: {symbol}")

    report = await analyze(symbol)
    now = datetime.now(IST)

    return {
        "symbol": symbol,
        "overall_verdict": report.overall_verdict,
        "summary": report.summary,
        "data_quality": report.data_quality,
        "beneish_m_score": _beneish_dict(report.beneish) if report.beneish else None,
        "altman_z_score": _altman_dict(report.altman) if report.altman else None,
        "generated_at": now.isoformat(),
    }
