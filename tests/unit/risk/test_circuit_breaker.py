"""Tests for drawdown circuit breaker."""

from __future__ import annotations

import pytest

from alphavedha.config import CircuitBreakerConfig
from alphavedha.risk.circuit_breaker import CircuitBreaker, CircuitBreakerState


@pytest.fixture
def config() -> CircuitBreakerConfig:
    return CircuitBreakerConfig(
        level_1_drawdown=10.0,
        level_2_drawdown=15.0,
        level_3_drawdown=20.0,
        recovery_threshold=0.95,
    )


class TestCircuitBreakerEvaluate:
    def test_normal_no_drawdown(self, config: CircuitBreakerConfig) -> None:
        cb = CircuitBreaker(config)
        state = cb.evaluate(current_value=1_000_000.0, peak_value=1_000_000.0)
        assert isinstance(state, CircuitBreakerState)
        assert state.level == 0
        assert state.current_drawdown_pct == 0.0

    def test_level_1_at_10pct(self, config: CircuitBreakerConfig) -> None:
        cb = CircuitBreaker(config)
        state = cb.evaluate(current_value=890_000.0, peak_value=1_000_000.0)
        assert state.level == 1
        assert abs(state.current_drawdown_pct - 11.0) < 0.1

    def test_level_2_at_15pct(self, config: CircuitBreakerConfig) -> None:
        cb = CircuitBreaker(config)
        state = cb.evaluate(current_value=840_000.0, peak_value=1_000_000.0)
        assert state.level == 2
        assert abs(state.current_drawdown_pct - 16.0) < 0.1

    def test_level_3_at_20pct(self, config: CircuitBreakerConfig) -> None:
        cb = CircuitBreaker(config)
        state = cb.evaluate(current_value=790_000.0, peak_value=1_000_000.0)
        assert state.level == 3
        assert abs(state.current_drawdown_pct - 21.0) < 0.1

    def test_recovery_resets_to_normal(self, config: CircuitBreakerConfig) -> None:
        cb = CircuitBreaker(config)
        state = cb.evaluate(current_value=960_000.0, peak_value=1_000_000.0)
        assert state.level == 0


class TestCircuitBreakerAdjust:
    def test_level_0_no_adjustment(self, config: CircuitBreakerConfig) -> None:
        cb = CircuitBreaker(config)
        state = CircuitBreakerState(
            level=0, current_drawdown_pct=0.0, peak_value=1e6, triggered_at=None
        )
        assert cb.adjust_position(5.0, state, is_new_entry=True) == 5.0

    def test_level_1_halves_position(self, config: CircuitBreakerConfig) -> None:
        cb = CircuitBreaker(config)
        state = CircuitBreakerState(
            level=1, current_drawdown_pct=11.0, peak_value=1e6, triggered_at=None
        )
        assert cb.adjust_position(6.0, state, is_new_entry=True) == 3.0

    def test_level_2_blocks_new_entry(self, config: CircuitBreakerConfig) -> None:
        cb = CircuitBreaker(config)
        state = CircuitBreakerState(
            level=2, current_drawdown_pct=16.0, peak_value=1e6, triggered_at=None
        )
        assert cb.adjust_position(5.0, state, is_new_entry=True) == 0.0

    def test_level_2_halves_existing(self, config: CircuitBreakerConfig) -> None:
        cb = CircuitBreaker(config)
        state = CircuitBreakerState(
            level=2, current_drawdown_pct=16.0, peak_value=1e6, triggered_at=None
        )
        assert cb.adjust_position(5.0, state, is_new_entry=False) == 2.5

    def test_level_3_zeroes_all(self, config: CircuitBreakerConfig) -> None:
        cb = CircuitBreaker(config)
        state = CircuitBreakerState(
            level=3, current_drawdown_pct=22.0, peak_value=1e6, triggered_at=None
        )
        assert cb.adjust_position(5.0, state, is_new_entry=False) == 0.0
