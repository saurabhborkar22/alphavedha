"""RiskManager — orchestrates position sizing, portfolio constraints, and circuit breaker."""

from __future__ import annotations

from dataclasses import dataclass, field

import structlog

from alphavedha.config import CircuitBreakerConfig, PortfolioConfig, PositionSizingConfig
from alphavedha.risk.circuit_breaker import CircuitBreaker
from alphavedha.risk.portfolio import PortfolioConstraints, PortfolioState
from alphavedha.risk.position_sizing import compute_position_size

logger = structlog.get_logger(__name__)


@dataclass
class RiskAssessment:
    position_size_pct: float
    kelly_raw: float
    kelly_half: float
    constraint_violations: list[str] = field(default_factory=list)
    circuit_breaker_level: int = 0
    risk_adjusted: bool = False


class RiskManager:
    def __init__(
        self,
        position_config: PositionSizingConfig,
        portfolio_config: PortfolioConfig,
        circuit_breaker_config: CircuitBreakerConfig,
    ) -> None:
        self._position_config = position_config
        self._portfolio_constraints = PortfolioConstraints(portfolio_config)
        self._circuit_breaker = CircuitBreaker(circuit_breaker_config)

    def assess(
        self,
        meta_confidence: float,
        magnitude: float,
        symbol: str,
        sector: str,
        portfolio: PortfolioState | None = None,
    ) -> RiskAssessment:
        # Step 1: Kelly position sizing
        kelly_half = compute_position_size(meta_confidence, magnitude, self._position_config)
        kelly_raw = (
            (2 * meta_confidence - 1) * 100
            if magnitude > 0 and meta_confidence >= self._position_config.min_confidence
            else 0.0
        )
        kelly_raw = max(kelly_raw, 0.0)

        position = kelly_half
        violations: list[str] = []
        cb_level = 0
        adjusted = False

        if portfolio is not None:
            # Step 2: Portfolio constraints
            constraint_result = self._portfolio_constraints.check(
                symbol=symbol,
                proposed_weight_pct=position,
                sector=sector,
                portfolio=portfolio,
            )
            violations = constraint_result.violations
            if constraint_result.adjusted_weight_pct != position:
                adjusted = True
            position = constraint_result.adjusted_weight_pct

            # Step 3: Circuit breaker
            cb_state = self._circuit_breaker.evaluate(
                current_value=portfolio.total_value,
                peak_value=portfolio.peak_value,
            )
            cb_level = cb_state.level
            cb_adjusted = self._circuit_breaker.adjust_position(
                proposed_size_pct=position,
                state=cb_state,
                is_new_entry=symbol not in portfolio.holdings,
            )
            if cb_adjusted != position:
                adjusted = True
            position = cb_adjusted

        if position != kelly_half:
            adjusted = True

        logger.info(
            "risk_assessment",
            symbol=symbol,
            kelly_raw=round(kelly_raw, 4),
            kelly_half=round(kelly_half, 4),
            final_position=round(position, 4),
            cb_level=cb_level,
            violations=violations,
        )

        return RiskAssessment(
            position_size_pct=position,
            kelly_raw=kelly_raw,
            kelly_half=kelly_half,
            constraint_violations=violations,
            circuit_breaker_level=cb_level,
            risk_adjusted=adjusted,
        )
