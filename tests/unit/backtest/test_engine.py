"""Tests for VectorBT backtesting engine."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from alphavedha.backtest.engine import BacktestResult, run_backtest
from alphavedha.config import BacktestConfig


@pytest.fixture
def default_config() -> BacktestConfig:
    return BacktestConfig()


@pytest.fixture
def bullish_predictions(sample_ohlcv_500: pd.DataFrame) -> pd.DataFrame:
    """Predictions that always say UP with high confidence."""
    n = len(sample_ohlcv_500)
    return pd.DataFrame(
        {
            "direction": np.ones(n, dtype=int),
            "magnitude": np.full(n, 0.02),
            "confidence": np.full(n, 0.7),
        },
        index=sample_ohlcv_500.index,
    )


@pytest.fixture
def neutral_predictions(sample_ohlcv_500: pd.DataFrame) -> pd.DataFrame:
    """Predictions that always say NEUTRAL (0)."""
    n = len(sample_ohlcv_500)
    return pd.DataFrame(
        {
            "direction": np.zeros(n, dtype=int),
            "magnitude": np.zeros(n),
            "confidence": np.full(n, 0.5),
        },
        index=sample_ohlcv_500.index,
    )


@pytest.fixture
def low_confidence_predictions(sample_ohlcv_500: pd.DataFrame) -> pd.DataFrame:
    """Predictions that say UP but with confidence below threshold."""
    n = len(sample_ohlcv_500)
    return pd.DataFrame(
        {
            "direction": np.ones(n, dtype=int),
            "magnitude": np.full(n, 0.02),
            "confidence": np.full(n, 0.3),
        },
        index=sample_ohlcv_500.index,
    )


class TestBacktestEngine:
    def test_returns_backtest_result(
        self,
        sample_ohlcv_500: pd.DataFrame,
        bullish_predictions: pd.DataFrame,
        default_config: BacktestConfig,
    ) -> None:
        result = run_backtest(
            predictions_df=bullish_predictions,
            ohlcv_df=sample_ohlcv_500,
            config=default_config,
        )
        assert isinstance(result, BacktestResult)

    def test_equity_curve_length(
        self,
        sample_ohlcv_500: pd.DataFrame,
        bullish_predictions: pd.DataFrame,
        default_config: BacktestConfig,
    ) -> None:
        result = run_backtest(bullish_predictions, sample_ohlcv_500, default_config)
        assert len(result.equity_curve) == len(sample_ohlcv_500)

    def test_no_trades_with_neutral(
        self,
        sample_ohlcv_500: pd.DataFrame,
        neutral_predictions: pd.DataFrame,
        default_config: BacktestConfig,
    ) -> None:
        result = run_backtest(neutral_predictions, sample_ohlcv_500, default_config)
        assert result.n_trades == 0

    def test_no_trades_with_low_confidence(
        self,
        sample_ohlcv_500: pd.DataFrame,
        low_confidence_predictions: pd.DataFrame,
        default_config: BacktestConfig,
    ) -> None:
        result = run_backtest(
            low_confidence_predictions,
            sample_ohlcv_500,
            default_config,
            min_confidence=0.55,
        )
        assert result.n_trades == 0

    def test_sharpe_is_float(
        self,
        sample_ohlcv_500: pd.DataFrame,
        bullish_predictions: pd.DataFrame,
        default_config: BacktestConfig,
    ) -> None:
        result = run_backtest(bullish_predictions, sample_ohlcv_500, default_config)
        assert isinstance(result.sharpe_ratio, float)

    def test_max_drawdown_negative_or_zero(
        self,
        sample_ohlcv_500: pd.DataFrame,
        bullish_predictions: pd.DataFrame,
        default_config: BacktestConfig,
    ) -> None:
        result = run_backtest(bullish_predictions, sample_ohlcv_500, default_config)
        assert result.max_drawdown <= 0

    def test_win_rate_bounded(
        self,
        sample_ohlcv_500: pd.DataFrame,
        bullish_predictions: pd.DataFrame,
        default_config: BacktestConfig,
    ) -> None:
        result = run_backtest(bullish_predictions, sample_ohlcv_500, default_config)
        if result.n_trades > 0:
            assert 0 <= result.win_rate <= 1

    def test_costs_reduce_returns(
        self,
        sample_ohlcv_500: pd.DataFrame,
        bullish_predictions: pd.DataFrame,
    ) -> None:
        zero_cost = BacktestConfig()
        zero_cost.costs.stt_delivery = 0
        zero_cost.costs.brokerage_flat = 0
        zero_cost.costs.exchange_txn = 0
        zero_cost.costs.gst = 0
        zero_cost.costs.sebi_turnover = 0
        zero_cost.costs.stamp_duty = 0
        zero_cost.slippage.large_cap = 0
        zero_cost.slippage.mid_cap = 0
        zero_cost.slippage.small_cap = 0

        normal_config = BacktestConfig()

        result_no_cost = run_backtest(bullish_predictions, sample_ohlcv_500, zero_cost)
        result_with_cost = run_backtest(bullish_predictions, sample_ohlcv_500, normal_config)

        if result_no_cost.n_trades > 0:
            assert result_with_cost.total_return <= result_no_cost.total_return

    def test_trade_log_dataframe(
        self,
        sample_ohlcv_500: pd.DataFrame,
        bullish_predictions: pd.DataFrame,
        default_config: BacktestConfig,
    ) -> None:
        result = run_backtest(bullish_predictions, sample_ohlcv_500, default_config)
        assert isinstance(result.trade_log, pd.DataFrame)
