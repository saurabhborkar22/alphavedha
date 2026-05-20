"""Signals module — trading signal generation."""

from alphavedha.signals.execution import ExecutionEngine, ExecutionPlan, ExecutionWindow
from alphavedha.signals.pairs import PairsBacktestResult, PairSignal, PairsTrader
from alphavedha.signals.pairs_universe import (
    SECTOR_PAIRS,
    PairCandidate,
    scan_pair_universe,
)

__all__ = [
    "SECTOR_PAIRS",
    "ExecutionEngine",
    "ExecutionPlan",
    "ExecutionWindow",
    "PairCandidate",
    "PairSignal",
    "PairsBacktestResult",
    "PairsTrader",
    "scan_pair_universe",
]
