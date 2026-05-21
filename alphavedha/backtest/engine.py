"""VectorBT backtesting engine — runs predictions through a strategy with Indian market costs."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import structlog

from alphavedha.backtest.costs import compute_round_trip_cost_pct
from alphavedha.config import BacktestConfig

logger = structlog.get_logger(__name__)


@dataclass
class BacktestResult:
    total_return: float
    annualized_return: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    max_drawdown_duration_days: int
    alpha_vs_benchmark: float
    win_rate: float
    profit_factor: float
    n_trades: int
    avg_holding_days: float
    avg_return_per_trade: float
    equity_curve: pd.Series
    drawdown_curve: pd.Series
    trade_log: pd.DataFrame
    benchmark_return: float


def _compute_drawdown(equity: pd.Series) -> tuple[pd.Series, float, int]:
    peak = equity.cummax()
    dd = (equity - peak) / peak
    max_dd = float(dd.min())

    duration = 0
    max_duration = 0
    for val in dd.values:
        if val < 0:
            duration += 1
            max_duration = max(max_duration, duration)
        else:
            duration = 0

    return dd, max_dd, max_duration


def _compute_sharpe(returns: pd.Series) -> float:
    if len(returns) < 2 or returns.std() == 0:
        return 0.0
    return float(returns.mean() / returns.std() * np.sqrt(252))


def _compute_sortino(returns: pd.Series) -> float:
    if len(returns) < 2:
        return 0.0
    downside = returns[returns < 0]
    if len(downside) == 0 or downside.std() == 0:
        return 0.0
    return float(returns.mean() / downside.std() * np.sqrt(252))


def run_backtest(
    predictions_df: pd.DataFrame,
    ohlcv_df: pd.DataFrame,
    config: BacktestConfig,
    market_cap_tier: str = "large",
    min_confidence: float = 0.55,
    max_holding_days: int = 15,
) -> BacktestResult:
    cost_pct = compute_round_trip_cost_pct(market_cap_tier, config)
    closes = ohlcv_df["close"]
    daily_returns = closes.pct_change().fillna(0.0)

    entries = (predictions_df["direction"] == 1) & (predictions_df["confidence"] >= min_confidence)
    exits = predictions_df["direction"] == -1

    position = pd.Series(0, index=closes.index, dtype=int)
    in_position = False
    entry_idx = -1
    trades: list[dict] = []
    holding_period = 0

    for i in range(len(closes)):
        if not in_position and entries.iloc[i]:
            in_position = True
            entry_idx = i
            holding_period = 0
            position.iloc[i] = 1
        elif in_position:
            holding_period += 1
            should_exit = exits.iloc[i] or holding_period >= max_holding_days
            if should_exit:
                position.iloc[i] = 0
                entry_price = closes.iloc[entry_idx]
                exit_price = closes.iloc[i]
                gross_ret = exit_price / entry_price - 1
                net_ret = gross_ret - cost_pct
                trades.append(
                    {
                        "entry_date": closes.index[entry_idx],
                        "exit_date": closes.index[i],
                        "entry_price": entry_price,
                        "exit_price": exit_price,
                        "gross_return": gross_ret,
                        "net_return": net_ret,
                        "holding_days": holding_period,
                    }
                )
                in_position = False
            else:
                position.iloc[i] = 1

    strategy_returns = daily_returns * position.shift(1).fillna(0)

    for trade in trades:
        exit_loc = closes.index.get_loc(trade["exit_date"])
        strategy_returns.iloc[exit_loc] -= cost_pct

    equity = (1 + strategy_returns).cumprod()
    dd_curve, max_dd, max_dd_duration = _compute_drawdown(equity)

    total_ret = float(equity.iloc[-1] - 1) if len(equity) > 0 else 0.0
    n_days = len(equity)
    ann_ret = float((1 + total_ret) ** (252 / max(n_days, 1)) - 1)
    sharpe = _compute_sharpe(strategy_returns)
    sortino = _compute_sortino(strategy_returns)

    benchmark_ret = float(closes.iloc[-1] / closes.iloc[0] - 1) if len(closes) > 1 else 0.0
    alpha = ann_ret - float((1 + benchmark_ret) ** (252 / max(n_days, 1)) - 1)

    trade_log = (
        pd.DataFrame(trades)
        if trades
        else pd.DataFrame(
            columns=[
                "entry_date",
                "exit_date",
                "entry_price",
                "exit_price",
                "gross_return",
                "net_return",
                "holding_days",
            ]
        )
    )

    n_trades = len(trades)
    if n_trades > 0:
        wins = [t for t in trades if t["net_return"] > 0]
        losses = [t for t in trades if t["net_return"] <= 0]
        win_rate = len(wins) / n_trades
        gross_profit = sum(t["net_return"] for t in wins) if wins else 0.0
        gross_loss = abs(sum(t["net_return"] for t in losses)) if losses else 0.0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")
        avg_hold = float(np.mean([t["holding_days"] for t in trades]))
        avg_ret = float(np.mean([t["net_return"] for t in trades]))
    else:
        win_rate = 0.0
        profit_factor = 0.0
        avg_hold = 0.0
        avg_ret = 0.0

    logger.info(
        "backtest_completed",
        n_trades=n_trades,
        total_return=round(total_ret, 4),
        sharpe=round(sharpe, 4),
        max_drawdown=round(max_dd, 4),
        win_rate=round(win_rate, 4),
    )

    return BacktestResult(
        total_return=total_ret,
        annualized_return=ann_ret,
        sharpe_ratio=sharpe,
        sortino_ratio=sortino,
        max_drawdown=max_dd,
        max_drawdown_duration_days=max_dd_duration,
        alpha_vs_benchmark=alpha,
        win_rate=win_rate,
        profit_factor=profit_factor,
        n_trades=n_trades,
        avg_holding_days=avg_hold,
        avg_return_per_trade=avg_ret,
        equity_curve=equity,
        drawdown_curve=dd_curve,
        trade_log=trade_log,
        benchmark_return=benchmark_ret,
    )
