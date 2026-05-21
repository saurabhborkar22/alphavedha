"""Tests for RiskManager — orchestrates position sizing, portfolio, and circuit breaker."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from alphavedha.config import CircuitBreakerConfig, PortfolioConfig, PositionSizingConfig
from alphavedha.risk.portfolio import HoldingInfo, PortfolioState
from alphavedha.risk.risk_manager import RiskAssessment, RiskManager


@pytest.fixture
def risk_manager() -> RiskManager:
    return RiskManager(
        position_config=PositionSizingConfig(
            method="half_kelly", max_single_stock_pct=10.0, min_confidence=0.55
        ),
        portfolio_config=PortfolioConfig(
            max_sector_pct=25.0, max_correlation=0.7, min_holding_days=3, min_daily_turnover_cr=5.0
        ),
        circuit_breaker_config=CircuitBreakerConfig(
            level_1_drawdown=10.0,
            level_2_drawdown=15.0,
            level_3_drawdown=20.0,
            recovery_threshold=0.95,
        ),
    )


@pytest.fixture
def healthy_portfolio() -> PortfolioState:
    return PortfolioState(
        holdings={
            "INFY": HoldingInfo(
                symbol="INFY",
                sector="IT",
                weight_pct=5.0,
                entry_date=datetime(2026, 1, 1, tzinfo=UTC),
                correlation_60d={},
                avg_daily_turnover_cr=100.0,
            )
        },
        total_value=1_000_000.0,
        peak_value=1_000_000.0,
    )


class TestRiskManager:
    def test_full_pipeline_returns_assessment(
        self, risk_manager: RiskManager, healthy_portfolio: PortfolioState
    ) -> None:
        result = risk_manager.assess(
            meta_confidence=0.70,
            magnitude=0.05,
            symbol="TCS",
            sector="IT",
            portfolio=healthy_portfolio,
        )
        assert isinstance(result, RiskAssessment)
        assert result.position_size_pct > 0.0
        assert result.kelly_raw > 0.0
        assert result.kelly_half > 0.0
        assert result.circuit_breaker_level == 0

    def test_no_portfolio_only_kelly(self, risk_manager: RiskManager) -> None:
        result = risk_manager.assess(
            meta_confidence=0.70,
            magnitude=0.05,
            symbol="TCS",
            sector="IT",
            portfolio=None,
        )
        assert result.position_size_pct > 0.0
        assert result.circuit_breaker_level == 0
        assert len(result.constraint_violations) == 0

    def test_low_confidence_zero_position(self, risk_manager: RiskManager) -> None:
        result = risk_manager.assess(
            meta_confidence=0.40,
            magnitude=0.05,
            symbol="TCS",
            sector="IT",
            portfolio=None,
        )
        assert result.position_size_pct == 0.0

    def test_circuit_breaker_level_2_blocks_new_entry(self, risk_manager: RiskManager) -> None:
        drawdown_portfolio = PortfolioState(
            holdings={},
            total_value=840_000.0,
            peak_value=1_000_000.0,
        )
        result = risk_manager.assess(
            meta_confidence=0.70,
            magnitude=0.05,
            symbol="TCS",
            sector="IT",
            portfolio=drawdown_portfolio,
        )
        assert result.position_size_pct == 0.0
        assert result.circuit_breaker_level == 2
        assert result.risk_adjusted is True
