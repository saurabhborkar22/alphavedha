"""Tests for Indian market cost calculator."""

from __future__ import annotations

import pytest

from alphavedha.backtest.costs import TradeCost, compute_round_trip_cost_pct, compute_trade_cost
from alphavedha.config import BacktestConfig


@pytest.fixture
def default_config() -> BacktestConfig:
    return BacktestConfig()


class TestTradeCost:
    def test_buy_side_components(self, default_config: BacktestConfig) -> None:
        cost = compute_trade_cost(
            trade_value=100_000.0,
            side="buy",
            market_cap_tier="large",
            config=default_config.costs,
            slippage_config=default_config.slippage,
        )
        assert isinstance(cost, TradeCost)
        assert cost.stt > 0
        assert cost.stamp_duty > 0
        assert cost.total > 0

    def test_sell_side_no_stamp_duty(self, default_config: BacktestConfig) -> None:
        cost = compute_trade_cost(
            trade_value=100_000.0,
            side="sell",
            market_cap_tier="large",
            config=default_config.costs,
            slippage_config=default_config.slippage,
        )
        assert cost.stamp_duty == 0.0

    def test_stt_calculation(self, default_config: BacktestConfig) -> None:
        cost = compute_trade_cost(
            trade_value=100_000.0,
            side="buy",
            market_cap_tier="large",
            config=default_config.costs,
            slippage_config=default_config.slippage,
        )
        expected_stt = 100_000.0 * 0.001
        assert abs(cost.stt - expected_stt) < 0.01

    def test_brokerage_flat(self, default_config: BacktestConfig) -> None:
        cost = compute_trade_cost(
            trade_value=100_000.0,
            side="buy",
            market_cap_tier="large",
            config=default_config.costs,
            slippage_config=default_config.slippage,
        )
        assert cost.brokerage == 20.0

    def test_gst_on_brokerage_and_exchange(self, default_config: BacktestConfig) -> None:
        cost = compute_trade_cost(
            trade_value=100_000.0,
            side="buy",
            market_cap_tier="large",
            config=default_config.costs,
            slippage_config=default_config.slippage,
        )
        expected_gst = (20.0 + 100_000.0 * 0.0000345) * 0.18
        assert abs(cost.gst - expected_gst) < 0.01

    def test_slippage_varies_by_tier(self, default_config: BacktestConfig) -> None:
        large = compute_trade_cost(
            100_000.0, "buy", "large", default_config.costs, default_config.slippage
        )
        mid = compute_trade_cost(
            100_000.0, "buy", "mid", default_config.costs, default_config.slippage
        )
        small = compute_trade_cost(
            100_000.0, "buy", "small", default_config.costs, default_config.slippage
        )
        assert small.slippage > mid.slippage > large.slippage

    def test_total_is_sum_of_components(self, default_config: BacktestConfig) -> None:
        cost = compute_trade_cost(
            100_000.0, "buy", "large", default_config.costs, default_config.slippage
        )
        component_sum = (
            cost.stt + cost.brokerage + cost.exchange_txn
            + cost.gst + cost.sebi_turnover + cost.stamp_duty + cost.slippage
        )
        assert abs(cost.total - component_sum) < 0.01


class TestRoundTripCost:
    def test_round_trip_positive(self, default_config: BacktestConfig) -> None:
        pct = compute_round_trip_cost_pct("large", default_config)
        assert pct > 0

    def test_round_trip_large_vs_mid(self, default_config: BacktestConfig) -> None:
        large = compute_round_trip_cost_pct("large", default_config)
        mid = compute_round_trip_cost_pct("mid", default_config)
        assert mid > large

    def test_round_trip_includes_all_7_components(self, default_config: BacktestConfig) -> None:
        """Verify the 7 cost types are all nonzero in a round trip."""
        buy = compute_trade_cost(
            100_000.0, "buy", "large", default_config.costs, default_config.slippage
        )
        sell = compute_trade_cost(
            100_000.0, "sell", "large", default_config.costs, default_config.slippage
        )
        assert buy.stt > 0 and sell.stt > 0
        assert buy.brokerage > 0 and sell.brokerage > 0
        assert buy.exchange_txn > 0 and sell.exchange_txn > 0
        assert buy.gst > 0 and sell.gst > 0
        assert buy.sebi_turnover > 0 and sell.sebi_turnover > 0
        assert buy.stamp_duty > 0
        assert sell.stamp_duty == 0
        assert buy.slippage > 0 and sell.slippage > 0
