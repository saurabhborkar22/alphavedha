"""FundamentalAnalyzer — orchestrate Beneish M-Score + Altman Z'-Score for a symbol."""

from __future__ import annotations

from dataclasses import dataclass

import structlog

from alphavedha.fundamental.altman import AltmanResult, compute_altman
from alphavedha.fundamental.beneish import BeneishResult, compute_beneish
from alphavedha.fundamental.fetcher import FinancialStatements, fetch_financials

logger = structlog.get_logger(__name__)


@dataclass
class FundamentalReport:
    symbol: str
    beneish: BeneishResult | None
    altman: AltmanResult | None
    overall_verdict: str  # "healthy" | "caution" | "red_flag" | "insufficient_data"
    summary: str
    data_quality: float


def _overall_verdict(beneish: BeneishResult | None, altman: AltmanResult | None) -> tuple[str, str]:
    if beneish is None and altman is None:
        return "insufficient_data", "Insufficient financial data to compute scores."

    flags: list[str] = []
    cautions: list[str] = []

    if beneish is not None:
        if beneish.verdict == "manipulator":
            flags.append(f"M-Score {beneish.m_score:.2f} flags earnings manipulation risk")
        elif beneish.verdict == "grey_zone":
            cautions.append(f"M-Score {beneish.m_score:.2f} in grey zone")

    if altman is not None:
        if altman.verdict == "distress":
            flags.append(f"Z'-Score {altman.z_score:.2f} indicates financial distress")
        elif altman.verdict == "grey_zone":
            cautions.append(f"Z'-Score {altman.z_score:.2f} in grey zone")

    if flags:
        verdict = "red_flag"
        summary = "RED FLAG: " + "; ".join(flags) + "."
    elif cautions:
        verdict = "caution"
        summary = "CAUTION: " + "; ".join(cautions) + "."
    else:
        parts = []
        if beneish:
            parts.append(f"M-Score {beneish.m_score:.2f} (non-manipulator)")
        if altman:
            parts.append(f"Z'-Score {altman.z_score:.2f} (safe)")
        verdict = "healthy"
        summary = "HEALTHY: " + "; ".join(parts) + "."

    return verdict, summary


async def analyze(symbol: str) -> FundamentalReport:
    """Fetch financial data and compute Beneish M-Score + Altman Z'-Score."""
    fs: FinancialStatements | None = await fetch_financials(symbol)

    if fs is None:
        logger.warning("fundamental_no_data", symbol=symbol)
        return FundamentalReport(
            symbol=symbol,
            beneish=None,
            altman=None,
            overall_verdict="insufficient_data",
            summary="Could not fetch financial statements for this symbol.",
            data_quality=0.0,
        )

    beneish_result: BeneishResult | None = None
    altman_result: AltmanResult | None = None

    try:
        beneish_result = compute_beneish(fs)
        logger.info(
            "beneish_computed",
            symbol=symbol,
            m_score=beneish_result.m_score,
            verdict=beneish_result.verdict,
        )
    except Exception as e:
        logger.warning("beneish_failed", symbol=symbol, error=str(e))

    try:
        altman_result = compute_altman(fs)
        logger.info(
            "altman_computed",
            symbol=symbol,
            z_score=altman_result.z_score,
            verdict=altman_result.verdict,
        )
    except Exception as e:
        logger.warning("altman_failed", symbol=symbol, error=str(e))

    verdict, summary = _overall_verdict(beneish_result, altman_result)

    return FundamentalReport(
        symbol=symbol,
        beneish=beneish_result,
        altman=altman_result,
        overall_verdict=verdict,
        summary=summary,
        data_quality=fs.data_quality,
    )
