"""Tests for portfolio-level constraints."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from alphavedha.config import PortfolioConfig
from alphavedha.risk.portfolio import (
    ConstraintResult,
    HoldingInfo,
    PortfolioConstraints,
    PortfolioState,
)


@pytest.fixture
def config() -> PortfolioConfig:
    return PortfolioConfig(
        max_sector_pct=25.0,
        max_correlation=0.7,
        min_holding_days=3,
        min_daily_turnover_cr=5.0,
    )


@pytest.fixture
def empty_portfolio() -> PortfolioState:
    return PortfolioState(holdings={}, total_value=1_000_000.0, peak_value=1_000_000.0)


def _make_holding(
    symbol: str,
    sector: str,
    weight_pct: float,
    days_held: int = 10,
    corr: dict[str, float] | None = None,
    turnover: float = 50.0,
) -> HoldingInfo:
    return HoldingInfo(
        symbol=symbol,
        sector=sector,
        weight_pct=weight_pct,
        entry_date=datetime.now(UTC) - timedelta(days=days_held),
        correlation_60d=corr or {},
        avg_daily_turnover_cr=turnover,
    )


class TestPortfolioConstraints:
    def test_within_all_limits_passes(
        self, config: PortfolioConfig, empty_portfolio: PortfolioState
    ) -> None:
        constraints = PortfolioConstraints(config)
        result = constraints.check(
            symbol="TCS",
            proposed_weight_pct=5.0,
            sector="IT",
            portfolio=empty_portfolio,
        )
        assert isinstance(result, ConstraintResult)
        assert result.passed is True
        assert result.adjusted_weight_pct == 5.0
        assert len(result.violations) == 0

    def test_exceeds_sector_cap_reduced(self, config: PortfolioConfig) -> None:
        portfolio = PortfolioState(
            holdings={
                "INFY": _make_holding("INFY", "IT", 20.0),
            },
            total_value=1_000_000.0,
            peak_value=1_000_000.0,
        )
        constraints = PortfolioConstraints(config)
        result = constraints.check(
            symbol="TCS",
            proposed_weight_pct=8.0,
            sector="IT",
            portfolio=portfolio,
        )
        assert result.adjusted_weight_pct == 5.0
        assert any("sector" in v.lower() for v in result.violations)

    def test_high_correlation_rejected(self, config: PortfolioConfig) -> None:
        portfolio = PortfolioState(
            holdings={
                "INFY": _make_holding("INFY", "IT", 5.0, corr={"TCS": 0.85}),
            },
            total_value=1_000_000.0,
            peak_value=1_000_000.0,
        )
        constraints = PortfolioConstraints(config)
        result = constraints.check(
            symbol="TCS",
            proposed_weight_pct=5.0,
            sector="IT",
            portfolio=portfolio,
        )
        assert result.adjusted_weight_pct == 0.0
        assert result.passed is False
        assert any("correlation" in v.lower() for v in result.violations)

    def test_sell_before_min_holding_rejected(self, config: PortfolioConfig) -> None:
        portfolio = PortfolioState(
            holdings={
                "TCS": _make_holding("TCS", "IT", 5.0, days_held=1),
            },
            total_value=1_000_000.0,
            peak_value=1_000_000.0,
        )
        constraints = PortfolioConstraints(config)
        result = constraints.check(
            symbol="TCS",
            proposed_weight_pct=-5.0,
            sector="IT",
            portfolio=portfolio,
        )
        assert result.adjusted_weight_pct == 0.0
        assert result.passed is False
        assert any("holding period" in v.lower() for v in result.violations)

    def test_low_liquidity_rejected(self, config: PortfolioConfig) -> None:
        portfolio = PortfolioState(
            holdings={},
            total_value=1_000_000.0,
            peak_value=1_000_000.0,
        )
        constraints = PortfolioConstraints(config)
        result = constraints.check(
            symbol="SMALLCAP",
            proposed_weight_pct=5.0,
            sector="Misc",
            portfolio=portfolio,
            avg_daily_turnover_cr=2.0,
        )
        assert result.adjusted_weight_pct == 0.0
        assert result.passed is False
        assert any("liquidity" in v.lower() for v in result.violations)
