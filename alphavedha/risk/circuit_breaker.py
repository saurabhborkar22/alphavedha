"""Circuit breaker — drawdown protection with 3 escalation levels."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

import structlog

from alphavedha.config import CircuitBreakerConfig

logger = structlog.get_logger(__name__)


@dataclass
class CircuitBreakerState:
    level: int
    current_drawdown_pct: float
    peak_value: float
    triggered_at: datetime | None


class CircuitBreaker:
    def __init__(self, config: CircuitBreakerConfig) -> None:
        self._config = config

    def evaluate(
        self,
        current_value: float,
        peak_value: float,
    ) -> CircuitBreakerState:
        if peak_value <= 0:
            return CircuitBreakerState(
                level=0, current_drawdown_pct=0.0, peak_value=peak_value, triggered_at=None
            )

        drawdown_pct = (1 - current_value / peak_value) * 100

        recovery_value = peak_value * self._config.recovery_threshold
        if current_value >= recovery_value:
            return CircuitBreakerState(
                level=0,
                current_drawdown_pct=drawdown_pct,
                peak_value=peak_value,
                triggered_at=None,
            )

        now = datetime.now(UTC)

        if drawdown_pct >= self._config.level_3_drawdown:
            level = 3
        elif drawdown_pct >= self._config.level_2_drawdown:
            level = 2
        elif drawdown_pct >= self._config.level_1_drawdown:
            level = 1
        else:
            level = 0

        if level > 0:
            logger.warning(
                "circuit_breaker_triggered",
                level=level,
                drawdown_pct=round(drawdown_pct, 2),
                current_value=current_value,
                peak_value=peak_value,
            )

        return CircuitBreakerState(
            level=level,
            current_drawdown_pct=drawdown_pct,
            peak_value=peak_value,
            triggered_at=now if level > 0 else None,
        )

    def adjust_position(
        self,
        proposed_size_pct: float,
        state: CircuitBreakerState,
        is_new_entry: bool,
    ) -> float:
        if state.level == 0:
            return proposed_size_pct
        if state.level == 3:
            return 0.0
        if state.level == 2:
            if is_new_entry:
                return 0.0
            return proposed_size_pct * 0.5
        if state.level == 1:
            return proposed_size_pct * 0.5
        return proposed_size_pct
