"""Walk-forward backtesting engine.

Simulates real-world usage: for each month, train on data BEFORE that month,
generate predictions, and measure P&L with full Indian market costs.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd
import structlog

from alphavedha.backtest.costs import compute_round_trip_cost_pct
from alphavedha.config import BacktestConfig

logger = structlog.get_logger(__name__)


@dataclass
class WalkForwardFold:
    """Results for a single walk-forward fold (one month)."""

    fold_start: date
    fold_end: date
    train_rows: int
    test_rows: int
    n_trades: int
    gross_return: float
    net_return: float
    win_rate: float
    regime: str = "unknown"


@dataclass
class WalkForwardResult:
    """Aggregate walk-forward backtest results."""

    folds: list[WalkForwardFold]
    total_return: float
    annualized_return: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    n_trades: int
    monthly_returns: pd.Series
    equity_curve: pd.Series
    trade_log: pd.DataFrame
    benchmark_return: float
    alpha_vs_benchmark: float


def _generate_monthly_folds(
    start: date,
    end: date,
) -> list[tuple[date, date]]:
    """Generate (fold_start, fold_end) for each month in range."""
    folds: list[tuple[date, date]] = []
    current = date(start.year, start.month, 1)

    from datetime import timedelta

    while current <= end:
        if current.month == 12:
            next_month = date(current.year + 1, 1, 1)
        else:
            next_month = date(current.year, current.month + 1, 1)

        last_day_of_month = next_month - timedelta(days=1)
        fold_end = min(last_day_of_month, end)

        if current >= start:
            folds.append((current, fold_end))

        current = next_month

    return folds


def run_walk_forward(
    ohlcv_df: pd.DataFrame,
    predictions_fn: callable,
    config: BacktestConfig,
    start: date | None = None,
    end: date | None = None,
    min_train_days: int = 252,
    market_cap_tier: str = "large",
    min_confidence: float = 0.55,
    max_holding_days: int = 15,
) -> WalkForwardResult:
    """Run walk-forward backtest.

    Args:
        ohlcv_df: Full OHLCV DataFrame with DatetimeIndex.
        predictions_fn: Callable(train_df, test_df) -> pd.DataFrame with
            columns: direction, confidence, magnitude.
        config: Backtest cost configuration.
        start: First test fold start date. Defaults to min_train_days into data.
        end: Last test fold end date. Defaults to last date in data.
        min_train_days: Minimum training window size.
        market_cap_tier: For slippage estimation.
        min_confidence: Minimum confidence to enter a trade.
        max_holding_days: Maximum holding period per trade.

    Returns:
        WalkForwardResult with per-fold and aggregate metrics.
    """
    if ohlcv_df.empty:
        raise ValueError("ohlcv_df is empty")

    data_start = ohlcv_df.index[0].date() if hasattr(ohlcv_df.index[0], "date") else ohlcv_df.index[0]
    data_end = ohlcv_df.index[-1].date() if hasattr(ohlcv_df.index[-1], "date") else ohlcv_df.index[-1]

    if start is None:
        start = (pd.Timestamp(data_start) + pd.Timedelta(days=min_train_days + 30)).date()
    if end is None:
        end = data_end

    folds_spec = _generate_monthly_folds(start, end)
    cost_pct = compute_round_trip_cost_pct(market_cap_tier, config)

    folds: list[WalkForwardFold] = []
    all_trades: list[dict] = []
    monthly_rets: list[float] = []
    monthly_dates: list[date] = []

    for fold_start, fold_end in folds_spec:
        train_mask = ohlcv_df.index < pd.Timestamp(fold_start)
        test_mask = (ohlcv_df.index >= pd.Timestamp(fold_start)) & (
            ohlcv_df.index <= pd.Timestamp(fold_end)
        )

        train_df = ohlcv_df[train_mask]
        test_df = ohlcv_df[test_mask]

        if len(train_df) < min_train_days or test_df.empty:
            continue

        try:
            preds = predictions_fn(train_df, test_df)
        except Exception as e:
            logger.warning("walk_forward_fold_failed", fold=str(fold_start), error=str(e))
            folds.append(WalkForwardFold(
                fold_start=fold_start,
                fold_end=fold_end,
                train_rows=len(train_df),
                test_rows=len(test_df),
                n_trades=0,
                gross_return=0.0,
                net_return=0.0,
                win_rate=0.0,
            ))
            monthly_rets.append(0.0)
            monthly_dates.append(fold_start)
            continue

        fold_trades = _execute_trades(
            test_df, preds, cost_pct, min_confidence, max_holding_days,
        )

        gross_ret = sum(t["gross_return"] for t in fold_trades) if fold_trades else 0.0
        net_ret = sum(t["net_return"] for t in fold_trades) if fold_trades else 0.0
        wins = [t for t in fold_trades if t["net_return"] > 0]
        win_rate = len(wins) / len(fold_trades) if fold_trades else 0.0

        folds.append(WalkForwardFold(
            fold_start=fold_start,
            fold_end=fold_end,
            train_rows=len(train_df),
            test_rows=len(test_df),
            n_trades=len(fold_trades),
            gross_return=gross_ret,
            net_return=net_ret,
            win_rate=win_rate,
        ))

        all_trades.extend(fold_trades)
        monthly_rets.append(net_ret)
        monthly_dates.append(fold_start)

    monthly_returns = pd.Series(monthly_rets, index=pd.to_datetime(monthly_dates))
    equity = (1 + monthly_returns).cumprod()

    total_ret = float(equity.iloc[-1] - 1) if len(equity) > 0 else 0.0
    n_months = len(equity)
    ann_ret = float((1 + total_ret) ** (12 / max(n_months, 1)) - 1)

    sharpe = _monthly_sharpe(monthly_returns)
    sortino = _monthly_sortino(monthly_returns)

    peak = equity.cummax()
    dd = (equity - peak) / peak
    max_dd = float(dd.min()) if len(dd) > 0 else 0.0

    n_trades = len(all_trades)
    if n_trades > 0:
        wins = [t for t in all_trades if t["net_return"] > 0]
        losses = [t for t in all_trades if t["net_return"] <= 0]
        overall_win_rate = len(wins) / n_trades
        gross_profit = sum(t["net_return"] for t in wins) if wins else 0.0
        gross_loss = abs(sum(t["net_return"] for t in losses)) if losses else 0.0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")
    else:
        overall_win_rate = 0.0
        profit_factor = 0.0

    trade_log = pd.DataFrame(all_trades) if all_trades else pd.DataFrame(
        columns=["entry_date", "exit_date", "entry_price", "exit_price",
                 "gross_return", "net_return", "holding_days", "fold"]
    )

    closes = ohlcv_df["close"]
    bench_start = closes.loc[closes.index >= pd.Timestamp(start)]
    benchmark_ret = (
        float(bench_start.iloc[-1] / bench_start.iloc[0] - 1)
        if len(bench_start) > 1 else 0.0
    )
    bench_ann = float((1 + benchmark_ret) ** (12 / max(n_months, 1)) - 1)
    alpha = ann_ret - bench_ann

    logger.info(
        "walk_forward_complete",
        n_folds=len(folds),
        n_trades=n_trades,
        total_return=round(total_ret, 4),
        sharpe=round(sharpe, 4),
        max_drawdown=round(max_dd, 4),
    )

    return WalkForwardResult(
        folds=folds,
        total_return=total_ret,
        annualized_return=ann_ret,
        sharpe_ratio=sharpe,
        sortino_ratio=sortino,
        max_drawdown=max_dd,
        win_rate=overall_win_rate,
        profit_factor=profit_factor,
        n_trades=n_trades,
        monthly_returns=monthly_returns,
        equity_curve=equity,
        trade_log=trade_log,
        benchmark_return=benchmark_ret,
        alpha_vs_benchmark=alpha,
    )


def _execute_trades(
    test_df: pd.DataFrame,
    preds: pd.DataFrame,
    cost_pct: float,
    min_confidence: float,
    max_holding_days: int,
) -> list[dict]:
    """Execute trades for a single fold based on predictions."""
    trades: list[dict] = []
    closes = test_df["close"]

    if preds.empty or closes.empty:
        return trades

    preds = preds.reindex(test_df.index)

    in_position = False
    entry_idx = -1
    holding = 0

    for i in range(len(closes)):
        direction = int(preds["direction"].iloc[i]) if pd.notna(preds["direction"].iloc[i]) else 0
        confidence = float(preds["confidence"].iloc[i]) if pd.notna(preds["confidence"].iloc[i]) else 0.0

        if not in_position and direction == 1 and confidence >= min_confidence:
            in_position = True
            entry_idx = i
            holding = 0
        elif in_position:
            holding += 1
            should_exit = direction == -1 or holding >= max_holding_days
            if should_exit:
                entry_price = closes.iloc[entry_idx]
                exit_price = closes.iloc[i]
                gross_ret = exit_price / entry_price - 1
                net_ret = gross_ret - cost_pct
                trades.append({
                    "entry_date": closes.index[entry_idx],
                    "exit_date": closes.index[i],
                    "entry_price": float(entry_price),
                    "exit_price": float(exit_price),
                    "gross_return": float(gross_ret),
                    "net_return": float(net_ret),
                    "holding_days": holding,
                    "fold": str(closes.index[0].date()),
                })
                in_position = False

    return trades


def _monthly_sharpe(monthly_returns: pd.Series) -> float:
    if len(monthly_returns) < 2 or monthly_returns.std() == 0:
        return 0.0
    return float(monthly_returns.mean() / monthly_returns.std() * np.sqrt(12))


def _monthly_sortino(monthly_returns: pd.Series) -> float:
    if len(monthly_returns) < 2:
        return 0.0
    downside = monthly_returns[monthly_returns < 0]
    if len(downside) == 0 or downside.std() == 0:
        return 0.0
    return float(monthly_returns.mean() / downside.std() * np.sqrt(12))
