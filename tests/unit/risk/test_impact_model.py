"""Tests for Almgren-Chriss market impact model."""

from __future__ import annotations

import pytest

from alphavedha.risk.impact_model import MarketImpactModel


@pytest.fixture
def model() -> MarketImpactModel:
    return MarketImpactModel()


class TestEstimateImpact:
    def test_large_cap_low_impact(self, model: MarketImpactModel) -> None:
        est = model.estimate_impact(
            order_size_shares=1000,
            avg_daily_volume=5_000_000,
            daily_volatility=0.015,
            cap_tier="large",
        )
        assert est.temporary_impact_pct < 0.01
        assert est.permanent_impact_pct < 0.01
        assert est.is_feasible is True

    def test_small_cap_higher_impact(self, model: MarketImpactModel) -> None:
        est = model.estimate_impact(
            order_size_shares=5000,
            avg_daily_volume=50_000,
            daily_volatility=0.03,
            cap_tier="small",
        )
        assert est.total_impact_pct > 0.001
        assert est.temporary_impact_pct > est.permanent_impact_pct

    def test_participation_rate(self, model: MarketImpactModel) -> None:
        est = model.estimate_impact(
            order_size_shares=10_000,
            avg_daily_volume=100_000,
            daily_volatility=0.02,
            cap_tier="mid",
        )
        assert est.participation_rate == pytest.approx(0.1, abs=0.001)

    def test_infeasible_order(self, model: MarketImpactModel) -> None:
        est = model.estimate_impact(
            order_size_shares=50_000,
            avg_daily_volume=100_000,
            daily_volatility=0.02,
            cap_tier="mid",
        )
        assert est.is_feasible is False
        assert "too large" in est.recommendation.lower()

    def test_recommendation_normal(self, model: MarketImpactModel) -> None:
        est = model.estimate_impact(
            order_size_shares=100,
            avg_daily_volume=1_000_000,
            daily_volatility=0.015,
            cap_tier="large",
        )
        assert "normally" in est.recommendation.lower()

    def test_recommendation_split(self, model: MarketImpactModel) -> None:
        est = model.estimate_impact(
            order_size_shares=7000,
            avg_daily_volume=100_000,
            daily_volatility=0.02,
            cap_tier="mid",
        )
        assert "tranches" in est.recommendation.lower()

    def test_recommendation_vwap(self, model: MarketImpactModel) -> None:
        est = model.estimate_impact(
            order_size_shares=15_000,
            avg_daily_volume=100_000,
            daily_volatility=0.02,
            cap_tier="mid",
        )
        assert "vwap" in est.recommendation.lower() or "twap" in est.recommendation.lower()


class TestOptimalHorizon:
    def test_patient_longer_horizon(self, model: MarketImpactModel) -> None:
        patient = model.optimal_execution_horizon(
            order_size_shares=5000,
            avg_daily_volume=100_000,
            urgency=0.0,
        )
        urgent = model.optimal_execution_horizon(
            order_size_shares=5000,
            avg_daily_volume=100_000,
            urgency=1.0,
        )
        assert patient > urgent

    def test_horizon_bounded(self, model: MarketImpactModel) -> None:
        horizon = model.optimal_execution_horizon(
            order_size_shares=100,
            avg_daily_volume=10_000_000,
            urgency=0.5,
        )
        assert 5 <= horizon <= 375
