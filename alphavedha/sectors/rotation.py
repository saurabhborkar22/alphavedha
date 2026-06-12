"""Sector Rotation Strategy — RRG-style analysis for NSE sector indices.

Implements a Relative Rotation Graph (RRG) approach:
- JdK RS-Ratio: relative performance of sector vs Nifty50 benchmark
- JdK RS-Momentum: rate-of-change of RS-Ratio

Rotation phases
---------------
Leading     (RS-Ratio > 100, RS-Momentum > 100): outperforming and accelerating
Weakening   (RS-Ratio > 100, RS-Momentum < 100): outperforming but momentum slowing
Lagging     (RS-Ratio < 100, RS-Momentum < 100): underperforming and decelerating
Improving   (RS-Ratio < 100, RS-Momentum > 100): underperforming but gaining momentum

References
----------
de Vries, J. (2004). Relative Rotation Graphs — a new tool for
analyzing relative strength and for portfolio management.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import structlog

logger = structlog.get_logger(__name__)

IST = ZoneInfo("Asia/Kolkata")

# ---------------------------------------------------------------------------
# NSE sector indices available on yfinance
# ---------------------------------------------------------------------------

SECTOR_TICKERS: dict[str, str] = {
    "NIFTY_BANK": "^NSEBANK",
    "NIFTY_IT": "^CNXIT",
    "NIFTY_AUTO": "^CNXAUTO",
    "NIFTY_PHARMA": "^CNXPHARMA",
    "NIFTY_FMCG": "^CNXFMCG",
    "NIFTY_METAL": "^CNXMETAL",
    "NIFTY_ENERGY": "^CNXENERGY",
    "NIFTY_INFRA": "^CNXINFRA",
    "NIFTY_REALTY": "^CNXREALTY",
    "NIFTY_MEDIA": "^CNXMEDIA",
    "NIFTY_PSU_BANK": "^CNXPSUBANK",
    "NIFTY_FIN_SERVICE": "^CNXFINANCE",
}

BENCHMARK_TICKER = "^NSEI"  # Nifty 50

# RRG smoothing window for RS-Ratio (in trading days)
_RS_RATIO_SMOOTHING = 10
# RS-Momentum is the 1d change in RS-Ratio, smoothed over this window
_RS_MOMENTUM_SMOOTHING = 5

# Lookback for fetching price history
_LOOKBACK_DAYS_HISTORY = 60


@dataclass
class SectorSignal:
    sector: str             # e.g. "NIFTY_IT"
    ticker: str             # e.g. "^CNXIT"
    rs_ratio: float         # relative strength vs Nifty; 100 = parity
    rs_momentum: float      # rate of change of RS-Ratio; 100 = flat
    phase: str              # "leading" | "weakening" | "lagging" | "improving"
    rank: int               # 1 = strongest rotation momentum (leading phase, highest rs_ratio)
    ret_1m: float           # 1-month absolute return (%)
    ret_3m: float           # 3-month absolute return (%)
    rel_ret_1m: float       # 1-month return relative to Nifty (%)
    interpretation: str


@dataclass
class SectorRotationReport:
    sectors: list[SectorSignal]
    benchmark_ret_1m: float
    benchmark_ret_3m: float
    top_sectors: list[str]       # top 3 by phase + rs_ratio
    avoid_sectors: list[str]     # lagging sectors
    rotation_message: str
    data_quality: float          # fraction of sectors with valid data
    generated_at: str = ""


def _phase(rs_ratio: float, rs_momentum: float) -> str:
    if rs_ratio >= 100 and rs_momentum >= 100:
        return "leading"
    if rs_ratio >= 100 and rs_momentum < 100:
        return "weakening"
    if rs_ratio < 100 and rs_momentum < 100:
        return "lagging"
    return "improving"


def _phase_rank(phase: str) -> int:
    return {"leading": 0, "improving": 1, "weakening": 2, "lagging": 3}[phase]


def _interpretation(sector: str, phase: str, rs_ratio: float, rel_ret_1m: float) -> str:
    clean = sector.replace("_", " ").title()
    if phase == "leading":
        return (
            f"{clean} is LEADING — outperforming Nifty (RS={rs_ratio:.1f}) "
            f"with accelerating momentum. Overweight."
        )
    if phase == "improving":
        return (
            f"{clean} is IMPROVING — underperforming but gaining momentum (RS={rs_ratio:.1f}). "
            f"Consider accumulating."
        )
    if phase == "weakening":
        return (
            f"{clean} is WEAKENING — still above benchmark (RS={rs_ratio:.1f}) "
            f"but momentum is slowing. Reduce exposure."
        )
    return (
        f"{clean} is LAGGING — underperforming Nifty (RS={rs_ratio:.1f}) "
        f"with declining momentum. Avoid / underweight."
    )


def _compute_rs_ratio(sector_prices: pd.Series, benchmark_prices: pd.Series) -> pd.Series:
    """Compute smoothed RS-Ratio relative to benchmark, normalised to 100."""
    sector_aligned = sector_prices.reindex(benchmark_prices.index, method="ffill")
    ratio = (sector_aligned / benchmark_prices) * 100
    smoothed = ratio.rolling(_RS_RATIO_SMOOTHING, min_periods=1).mean()
    # Normalise so that the last benchmark period = 100
    baseline = smoothed.iloc[0] if len(smoothed) > 0 else 100.0
    return (smoothed / baseline) * 100


def _compute_rs_momentum(rs_ratio: pd.Series) -> pd.Series:
    """Rate of change of RS-Ratio, smoothed; 100 = flat."""
    roc = rs_ratio / rs_ratio.shift(1) * 100
    return roc.rolling(_RS_MOMENTUM_SMOOTHING, min_periods=1).mean()


def _fetch_prices_sync(tickers: list[str], period: str = "3mo") -> dict[str, pd.Series]:
    """Fetch adjusted close prices for a list of tickers via yfinance."""
    import yfinance as yf

    results: dict[str, pd.Series] = {}
    for ticker in tickers:
        try:
            t = yf.Ticker(ticker)
            hist = t.history(period=period, auto_adjust=True)
            if hist.empty or "Close" not in hist.columns:
                logger.debug("sector_no_data", ticker=ticker)
                continue
            s = hist["Close"].dropna()
            s.index = pd.DatetimeIndex(s.index).tz_localize(None)
            results[ticker] = s
        except Exception as exc:
            logger.debug("sector_fetch_failed", ticker=ticker, error=str(exc))
    return results


async def _fetch_prices(tickers: list[str]) -> dict[str, pd.Series]:
    return await asyncio.to_thread(_fetch_prices_sync, tickers)


def _pct_ret(prices: pd.Series, n_days: int) -> float:
    """Compute n-day trailing return in percent."""
    if len(prices) < 2:
        return 0.0
    end = prices.iloc[-1]
    start_idx = max(0, len(prices) - n_days - 1)
    start = prices.iloc[start_idx]
    return round(float((end / start - 1) * 100), 2) if start != 0 else 0.0


async def compute_sector_rotation() -> SectorRotationReport:
    """Fetch sector index prices and compute RRG sector rotation signals."""
    all_tickers = [BENCHMARK_TICKER] + list(SECTOR_TICKERS.values())
    prices = await _fetch_prices(all_tickers)

    now_ist = datetime.now(IST).isoformat()
    benchmark_prices = prices.get(BENCHMARK_TICKER)

    if benchmark_prices is None or len(benchmark_prices) < 10:
        logger.warning("sector_rotation_no_benchmark")
        return SectorRotationReport(
            sectors=[],
            benchmark_ret_1m=0.0,
            benchmark_ret_3m=0.0,
            top_sectors=[],
            avoid_sectors=[],
            rotation_message="Insufficient data to compute sector rotation.",
            data_quality=0.0,
            generated_at=now_ist,
        )

    benchmark_ret_1m = _pct_ret(benchmark_prices, 21)
    benchmark_ret_3m = _pct_ret(benchmark_prices, 63)

    signals: list[SectorSignal] = []
    n_valid = 0

    for sector_name, ticker in SECTOR_TICKERS.items():
        sector_prices = prices.get(ticker)
        if sector_prices is None or len(sector_prices) < 10:
            continue
        n_valid += 1

        # Align to common index
        common_idx = benchmark_prices.index.intersection(sector_prices.index)
        if len(common_idx) < 5:
            continue
        bench_aligned = benchmark_prices.reindex(common_idx)
        sec_aligned = sector_prices.reindex(common_idx)

        rs_ratio_series = _compute_rs_ratio(sec_aligned, bench_aligned)
        rs_mom_series = _compute_rs_momentum(rs_ratio_series)

        rs_ratio = round(float(rs_ratio_series.iloc[-1]), 2)
        rs_momentum = round(float(rs_mom_series.iloc[-1]), 2)

        # Handle NaN (e.g. very short series)
        if np.isnan(rs_ratio):
            rs_ratio = 100.0
        if np.isnan(rs_momentum):
            rs_momentum = 100.0

        ret_1m = _pct_ret(sec_aligned, 21)
        ret_3m = _pct_ret(sec_aligned, 63)
        rel_ret_1m = round(ret_1m - benchmark_ret_1m, 2)

        phase = _phase(rs_ratio, rs_momentum)

        signals.append(
            SectorSignal(
                sector=sector_name,
                ticker=ticker,
                rs_ratio=rs_ratio,
                rs_momentum=rs_momentum,
                phase=phase,
                rank=0,  # filled below
                ret_1m=ret_1m,
                ret_3m=ret_3m,
                rel_ret_1m=rel_ret_1m,
                interpretation=_interpretation(sector_name, phase, rs_ratio, rel_ret_1m),
            )
        )

    # Sort: phase priority (leading > improving > weakening > lagging), then rs_ratio desc
    signals.sort(key=lambda s: (_phase_rank(s.phase), -s.rs_ratio))
    for i, sig in enumerate(signals, 1):
        sig.rank = i

    top_sectors = [s.sector for s in signals if s.phase in ("leading", "improving")][:3]
    avoid_sectors = [s.sector for s in signals if s.phase == "lagging"]

    if signals:
        leading = [s for s in signals if s.phase == "leading"]
        lagging = [s for s in signals if s.phase == "lagging"]
        msg_parts = []
        if leading:
            msg_parts.append(f"Leading: {', '.join(s.sector for s in leading[:3])}")
        if lagging:
            msg_parts.append(f"Lagging: {', '.join(s.sector for s in lagging[:3])}")
        rotation_message = " | ".join(msg_parts) if msg_parts else "All sectors near parity."
    else:
        rotation_message = "Insufficient sector data."

    quality = round(n_valid / len(SECTOR_TICKERS), 2)

    logger.info(
        "sector_rotation_computed",
        n_sectors=len(signals),
        leading=[s.sector for s in signals if s.phase == "leading"],
    )

    return SectorRotationReport(
        sectors=signals,
        benchmark_ret_1m=benchmark_ret_1m,
        benchmark_ret_3m=benchmark_ret_3m,
        top_sectors=top_sectors,
        avoid_sectors=avoid_sectors,
        rotation_message=rotation_message,
        data_quality=quality,
        generated_at=now_ist,
    )
