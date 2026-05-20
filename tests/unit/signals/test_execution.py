"""Tests for execution timing engine — windows, slippage, and expiry detection."""

from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

from alphavedha.signals.execution import ExecutionEngine

IST = ZoneInfo("Asia/Kolkata")


@pytest.fixture
def engine() -> ExecutionEngine:
    return ExecutionEngine()


class TestPlanExecution:
    def test_large_cap_market_order(self, engine: ExecutionEngine) -> None:
        plan = engine.plan_execution(
            symbol="RELIANCE",
            cap_tier="large",
            avg_daily_volume=5_000_000,
            order_size_shares=1000,
        )
        assert plan.order_type == "market"
        assert plan.n_tranches == 1
        assert plan.cap_tier == "large"

    def test_mid_cap_limit_order(self, engine: ExecutionEngine) -> None:
        plan = engine.plan_execution(
            symbol="TRENT",
            cap_tier="mid",
            avg_daily_volume=500_000,
            order_size_shares=1000,
        )
        assert plan.order_type == "limit"
        assert plan.n_tranches >= 2

    def test_small_cap_vwap(self, engine: ExecutionEngine) -> None:
        plan = engine.plan_execution(
            symbol="SMALLSTOCK",
            cap_tier="small",
            avg_daily_volume=50_000,
            order_size_shares=100,
        )
        assert plan.order_type == "vwap"
        assert 3 <= plan.n_tranches <= 5

    def test_expiry_day_warning(self, engine: ExecutionEngine) -> None:
        plan = engine.plan_execution(
            symbol="TCS",
            cap_tier="large",
            avg_daily_volume=2_000_000,
            order_size_shares=500,
            is_expiry_day=True,
        )
        assert any("expiry" in w.lower() for w in plan.warnings)

    def test_high_volume_splits_tranches(self, engine: ExecutionEngine) -> None:
        plan = engine.plan_execution(
            symbol="TCS",
            cap_tier="large",
            avg_daily_volume=100_000,
            order_size_shares=5000,
        )
        assert plan.n_tranches > 1


class TestIsGoodTime:
    def test_optimal_window(self, engine: ExecutionEngine) -> None:
        t = datetime(2026, 5, 20, 10, 45, tzinfo=IST)
        is_good, reason = engine.is_good_time_to_trade(t)
        assert is_good is True
        assert "stability" in reason.lower() or "liquidity" in reason.lower()

    def test_avoid_window_opening(self, engine: ExecutionEngine) -> None:
        t = datetime(2026, 5, 20, 9, 20, tzinfo=IST)
        is_good, _reason = engine.is_good_time_to_trade(t)
        assert is_good is False

    def test_outside_market_hours(self, engine: ExecutionEngine) -> None:
        t = datetime(2026, 5, 20, 8, 0, tzinfo=IST)
        is_good, reason = engine.is_good_time_to_trade(t)
        assert is_good is False
        assert "closed" in reason.lower()


class TestSlippage:
    def test_large_cap_low_slippage(self, engine: ExecutionEngine) -> None:
        slippage = engine.estimate_slippage(
            order_size_shares=1000,
            avg_daily_volume=5_000_000,
            bid_ask_spread_pct=0.001,
            volatility=0.015,
            cap_tier="large",
        )
        assert slippage < 0.05

    def test_small_cap_higher_slippage(self, engine: ExecutionEngine) -> None:
        slippage = engine.estimate_slippage(
            order_size_shares=1000,
            avg_daily_volume=50_000,
            bid_ask_spread_pct=0.005,
            volatility=0.03,
            cap_tier="small",
        )
        assert slippage > 0.01


class TestExpiry:
    def test_is_expiry_day_true(self) -> None:
        last_thu_may_2026 = datetime(2026, 5, 28, tzinfo=IST)
        assert ExecutionEngine.is_expiry_day(last_thu_may_2026) is True

    def test_is_expiry_day_false(self) -> None:
        not_thursday = datetime(2026, 5, 20, tzinfo=IST)
        assert ExecutionEngine.is_expiry_day(not_thursday) is False

    def test_next_expiry(self) -> None:
        dt = datetime(2026, 5, 1, tzinfo=IST)
        nxt = ExecutionEngine.next_expiry(dt)
        assert nxt.month == 5
        assert nxt.weekday() == 3  # Thursday
