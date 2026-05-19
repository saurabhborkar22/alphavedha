"""Pair universe identification — find cointegrated stock pairs within sectors.

Uses Engle-Granger cointegration test to validate sector-based pair candidates.
Only pairs that pass statistical tests are included in the trading universe.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import structlog
from statsmodels.tsa.stattools import coint

logger = structlog.get_logger(__name__)

SECTOR_PAIRS: list[tuple[str, str, str]] = [
    ("HDFCBANK.NS", "ICICIBANK.NS", "Banking"),
    ("TCS.NS", "INFY.NS", "IT"),
    ("RELIANCE.NS", "ONGC.NS", "Energy"),
    ("BHARTIARTL.NS", "IDEA.NS", "Telecom"),
    ("TITAN.NS", "ASIANPAINT.NS", "Consumer"),
    ("SBIN.NS", "BANKBARODA.NS", "PSU Banking"),
    ("WIPRO.NS", "HCLTECH.NS", "IT"),
    ("KOTAKBANK.NS", "AXISBANK.NS", "Banking"),
    ("NTPC.NS", "POWERGRID.NS", "Power"),
    ("HINDALCO.NS", "TATASTEEL.NS", "Metals"),
]


@dataclass
class PairCandidate:
    symbol_a: str
    symbol_b: str
    sector: str
    coint_pvalue: float
    correlation: float
    half_life: float
    is_valid: bool


def estimate_half_life(spread: pd.Series) -> float:
    """Estimate mean-reversion half-life via OLS on lagged spread."""
    spread_clean = spread.dropna()
    if len(spread_clean) < 30:
        return float("inf")

    lag = spread_clean.shift(1).dropna()
    delta = spread_clean.diff().dropna()

    common_idx = lag.index.intersection(delta.index)
    lag = lag.loc[common_idx]
    delta = delta.loc[common_idx]

    if lag.std() == 0:
        return float("inf")

    beta = float(np.corrcoef(lag, delta)[0, 1] * delta.std() / lag.std())

    if beta >= 0:
        return float("inf")

    return float(-np.log(2) / beta)


def compute_hedge_ratio(prices_a: pd.Series, prices_b: pd.Series) -> float:
    """OLS hedge ratio: how many units of B to short per unit of A."""
    from numpy.linalg import lstsq

    X = prices_b.values.reshape(-1, 1)
    y = prices_a.values
    beta, _, _, _ = lstsq(X, y, rcond=None)
    return float(beta[0])


def validate_pair(
    prices_a: pd.Series,
    prices_b: pd.Series,
    max_pvalue: float = 0.05,
    min_correlation: float = 0.6,
    max_half_life: float = 60.0,
) -> tuple[bool, float, float, float, float]:
    """Test if a pair is cointegrated and suitable for pairs trading.

    Returns: (is_valid, pvalue, correlation, half_life, hedge_ratio)
    """
    common_idx = prices_a.dropna().index.intersection(prices_b.dropna().index)
    if len(common_idx) < 120:
        return False, 1.0, 0.0, float("inf"), 0.0

    a = prices_a.loc[common_idx]
    b = prices_b.loc[common_idx]

    corr = float(a.corr(b))
    if abs(corr) < min_correlation:
        return False, 1.0, corr, float("inf"), 0.0

    _, pvalue, _ = coint(a.values, b.values)
    pvalue = float(pvalue)

    hedge = compute_hedge_ratio(a, b)
    spread = a - hedge * b
    hl = estimate_half_life(spread)

    is_valid = pvalue < max_pvalue and hl < max_half_life
    return is_valid, pvalue, corr, hl, hedge


def scan_pair_universe(
    price_data: dict[str, pd.Series],
    candidates: list[tuple[str, str, str]] | None = None,
    max_pvalue: float = 0.05,
    min_correlation: float = 0.6,
    max_half_life: float = 60.0,
) -> list[PairCandidate]:
    """Scan candidate pairs and return validated cointegrated pairs.

    Args:
        price_data: Dict mapping symbol -> closing price Series.
        candidates: List of (sym_a, sym_b, sector) tuples. Defaults to SECTOR_PAIRS.
        max_pvalue: Maximum cointegration p-value.
        min_correlation: Minimum price correlation.
        max_half_life: Maximum mean-reversion half-life in days.

    Returns:
        List of PairCandidate results, sorted by p-value (best first).
    """
    if candidates is None:
        candidates = SECTOR_PAIRS

    results: list[PairCandidate] = []

    for sym_a, sym_b, sector in candidates:
        if sym_a not in price_data or sym_b not in price_data:
            logger.debug("pair_skipped_missing_data", sym_a=sym_a, sym_b=sym_b)
            continue

        is_valid, pval, corr, hl, _ = validate_pair(
            price_data[sym_a],
            price_data[sym_b],
            max_pvalue=max_pvalue,
            min_correlation=min_correlation,
            max_half_life=max_half_life,
        )

        results.append(PairCandidate(
            symbol_a=sym_a,
            symbol_b=sym_b,
            sector=sector,
            coint_pvalue=pval,
            correlation=corr,
            half_life=hl,
            is_valid=is_valid,
        ))

    results.sort(key=lambda p: p.coint_pvalue)
    valid_count = sum(1 for p in results if p.is_valid)
    logger.info("pair_universe_scan_complete", total=len(results), valid=valid_count)
    return results
