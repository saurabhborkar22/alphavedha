"""Red-flag / blowup-score endpoint — exposes the daily avoid list."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

import structlog
from fastapi import APIRouter, Depends, Query

from alphavedha.api.deps import verify_api_key
from alphavedha.intel.signals.blowup_score import (
    AVOID_THRESHOLD,
    BlowupScore,
)

logger = structlog.get_logger(__name__)
router = APIRouter(
    prefix="/intel",
    tags=["intel"],
    dependencies=[Depends(verify_api_key)],
)


def _score_dict(s: BlowupScore) -> dict[str, Any]:
    return asdict(s)


@router.get("/red-flags")
async def get_red_flags(
    symbol: str | None = Query(default=None, description="Single symbol to check"),
    threshold: int = Query(default=AVOID_THRESHOLD, ge=0, le=100),
) -> dict[str, Any]:
    """Return symbols on the blowup avoid list.

    Pass ``symbol`` to check a single stock, or omit for full Nifty 50 scan.
    ``threshold`` overrides the default (70) for exploration.
    """
    try:
        from alphavedha.intel.signals.blowup_score import run_blowup_scores

        if symbol:
            symbols = [symbol]
        else:
            from alphavedha.api.routes.ui_support import NIFTY_50

            symbols = [s for s, _n, _sec, _c in NIFTY_50]

        all_scores = await run_blowup_scores(symbols)
        flagged = [s for s in all_scores if s.total_score >= threshold]
        flagged.sort(key=lambda s: s.total_score, reverse=True)

        return {
            "threshold": threshold,
            "flagged_count": len(flagged),
            "symbols": [_score_dict(s) for s in flagged],
        }
    except Exception as e:
        logger.error("red_flags_endpoint_error", error=str(e))
        return {
            "threshold": threshold,
            "flagged_count": 0,
            "symbols": [],
            "error": str(e),
        }
