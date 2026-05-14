"""Backtesting — CPCV validation, cost modeling, and VectorBT engine."""

from alphavedha.backtest.costs import TradeCost, compute_round_trip_cost_pct, compute_trade_cost
from alphavedha.backtest.cpcv import CPCVResult, PathResult, generate_cpcv_splits, run_cpcv
from alphavedha.backtest.engine import BacktestResult, run_backtest

__all__ = [
    "BacktestResult",
    "CPCVResult",
    "PathResult",
    "TradeCost",
    "compute_round_trip_cost_pct",
    "compute_trade_cost",
    "generate_cpcv_splits",
    "run_backtest",
    "run_cpcv",
]
