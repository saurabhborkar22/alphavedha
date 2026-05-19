"""Tests for walk-forward backtesting engine."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from alphavedha.backtest.walk_forward import (
    WalkForwardResult,
    _generate_monthly_folds,
    run_walk_forward,
)
from alphavedha.config import BacktestConfig


def _make_ohlcv(n_days: int = 600) -> pd.DataFrame:
    """Create realistic OHLCV data for testing."""
    rng = np.random.default_rng(42)
    dates = pd.bdate_range("2023-01-01", periods=n_days)
    prices = 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.015, n_days)))

    return pd.DataFrame(
        {
            "open": prices * (1 + rng.uniform(-0.005, 0.005, n_days)),
            "high": prices * (1 + rng.uniform(0.001, 0.02, n_days)),
            "low": prices * (1 - rng.uniform(0.001, 0.02, n_days)),
            "close": prices,
            "volume": rng.integers(100000, 1000000, n_days),
        },
        index=dates,
    )


def _simple_predictions(train_df: pd.DataFrame, test_df: pd.DataFrame) -> pd.DataFrame:
    """Simple momentum strategy: buy if last 5-day return positive."""
    rng = np.random.default_rng(99)
    n = len(test_df)
    return pd.DataFrame(
        {
            "direction": rng.choice([1, -1], size=n),
            "confidence": rng.uniform(0.5, 0.8, size=n),
            "magnitude": rng.uniform(0.01, 0.03, size=n),
        },
        index=test_df.index,
    )


class TestMonthlyFolds:
    def test_generates_monthly_buckets(self) -> None:
        folds = _generate_monthly_folds(date(2024, 1, 1), date(2024, 3, 31))
        assert len(folds) == 3
        assert folds[0][0] == date(2024, 1, 1)
        assert folds[2][0] == date(2024, 3, 1)

    def test_single_month(self) -> None:
        folds = _generate_monthly_folds(date(2024, 6, 1), date(2024, 6, 30))
        assert len(folds) == 1

    def test_cross_year_boundary(self) -> None:
        folds = _generate_monthly_folds(date(2024, 11, 1), date(2025, 2, 28))
        assert len(folds) == 4


class TestWalkForwardBacktest:
    def test_returns_walk_forward_result(self) -> None:
        ohlcv = _make_ohlcv(600)
        config = BacktestConfig()
        result = run_walk_forward(
            ohlcv_df=ohlcv,
            predictions_fn=_simple_predictions,
            config=config,
            min_train_days=252,
        )
        assert isinstance(result, WalkForwardResult)
        assert len(result.folds) > 0
        assert isinstance(result.equity_curve, pd.Series)
        assert isinstance(result.trade_log, pd.DataFrame)

    def test_sharpe_is_finite(self) -> None:
        ohlcv = _make_ohlcv(600)
        config = BacktestConfig()
        result = run_walk_forward(
            ohlcv_df=ohlcv,
            predictions_fn=_simple_predictions,
            config=config,
        )
        assert np.isfinite(result.sharpe_ratio)

    def test_max_drawdown_nonpositive(self) -> None:
        ohlcv = _make_ohlcv(600)
        config = BacktestConfig()
        result = run_walk_forward(
            ohlcv_df=ohlcv,
            predictions_fn=_simple_predictions,
            config=config,
        )
        assert result.max_drawdown <= 0

    def test_empty_data_raises(self) -> None:
        config = BacktestConfig()
        with pytest.raises(ValueError, match="empty"):
            run_walk_forward(
                ohlcv_df=pd.DataFrame(),
                predictions_fn=_simple_predictions,
                config=config,
            )

    def test_trades_have_positive_holding_days(self) -> None:
        ohlcv = _make_ohlcv(600)
        config = BacktestConfig()
        result = run_walk_forward(
            ohlcv_df=ohlcv,
            predictions_fn=_simple_predictions,
            config=config,
        )
        if not result.trade_log.empty:
            assert (result.trade_log["holding_days"] > 0).all()

    def test_fold_prediction_failure_handled(self) -> None:
        ohlcv = _make_ohlcv(600)
        config = BacktestConfig()

        call_count = 0

        def _failing_preds(train_df: pd.DataFrame, test_df: pd.DataFrame) -> pd.DataFrame:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("Simulated failure")
            return _simple_predictions(train_df, test_df)

        result = run_walk_forward(
            ohlcv_df=ohlcv,
            predictions_fn=_failing_preds,
            config=config,
        )
        assert len(result.folds) > 0
