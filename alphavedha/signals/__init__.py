"""Signals module — trading signal generation."""

from alphavedha.signals.pairs import PairsBacktestResult, PairsTrader, PairSignal
from alphavedha.signals.pairs_universe import (
    PairCandidate,
    scan_pair_universe,
    SECTOR_PAIRS,
)

__all__ = [
    "PairCandidate",
    "PairSignal",
    "PairsBacktestResult",
    "PairsTrader",
    "scan_pair_universe",
    "SECTOR_PAIRS",
]
