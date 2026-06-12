"""Sector rotation API endpoints."""

from __future__ import annotations

from typing import Any

import structlog
from fastapi import APIRouter, Depends

from alphavedha.api.deps import verify_api_key
from alphavedha.sectors.rotation import SectorSignal, compute_sector_rotation

logger = structlog.get_logger(__name__)
router = APIRouter(
    prefix="/sectors",
    tags=["sectors"],
    dependencies=[Depends(verify_api_key)],
)


def _signal_dict(s: SectorSignal) -> dict[str, Any]:
    return {
        "sector": s.sector,
        "ticker": s.ticker,
        "phase": s.phase,
        "rank": s.rank,
        "rs_ratio": s.rs_ratio,
        "rs_momentum": s.rs_momentum,
        "ret_1m_pct": s.ret_1m,
        "ret_3m_pct": s.ret_3m,
        "rel_ret_1m_pct": s.rel_ret_1m,
        "interpretation": s.interpretation,
    }


@router.get("/rotation")
async def sector_rotation() -> dict[str, Any]:
    """Return NSE sector rotation signals using RRG (Relative Rotation Graph) methodology.

    Computes RS-Ratio (relative strength vs Nifty50) and RS-Momentum for all major
    NSE sector indices and classifies each into a rotation phase.

    Rotation phases
    ---------------
    leading    — outperforming Nifty with accelerating momentum (overweight)
    improving  — underperforming but gaining momentum (accumulate)
    weakening  — outperforming but momentum slowing (reduce)
    lagging    — underperforming with declining momentum (avoid)

    Sectors are ranked by phase priority then RS-Ratio.
    """
    report = await compute_sector_rotation()

    return {
        "rotation_message": report.rotation_message,
        "top_sectors": report.top_sectors,
        "avoid_sectors": report.avoid_sectors,
        "benchmark": {
            "name": "NIFTY50",
            "ret_1m_pct": report.benchmark_ret_1m,
            "ret_3m_pct": report.benchmark_ret_3m,
        },
        "sectors": [_signal_dict(s) for s in report.sectors],
        "data_quality": report.data_quality,
        "generated_at": report.generated_at,
    }
