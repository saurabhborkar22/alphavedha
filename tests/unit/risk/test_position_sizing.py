"""Tests for Half-Kelly position sizing."""

from __future__ import annotations

import pytest

from alphavedha.config import PositionSizingConfig
from alphavedha.risk.position_sizing import compute_position_size


@pytest.fixture
def config() -> PositionSizingConfig:
    return PositionSizingConfig(
        method="half_kelly",
        max_single_stock_pct=10.0,
        min_confidence=0.55,
    )


class TestComputePositionSize:
    def test_valid_confidence_returns_positive(self, config: PositionSizingConfig) -> None:
        result = compute_position_size(
            meta_confidence=0.70,
            magnitude=0.05,
            config=config,
        )
        assert result > 0.0
        assert result <= config.max_single_stock_pct

    def test_half_kelly_is_half_of_full(self, config: PositionSizingConfig) -> None:
        meta_confidence = 0.70
        magnitude = 0.05
        full_kelly = 2 * meta_confidence - 1
        half_kelly_expected = full_kelly * 0.5 * 100
        result = compute_position_size(meta_confidence, magnitude, config)
        assert abs(result - min(half_kelly_expected, config.max_single_stock_pct)) < 1e-10

    def test_below_min_confidence_returns_zero(self, config: PositionSizingConfig) -> None:
        result = compute_position_size(
            meta_confidence=0.50,
            magnitude=0.05,
            config=config,
        )
        assert result == 0.0

    def test_zero_magnitude_returns_zero(self, config: PositionSizingConfig) -> None:
        result = compute_position_size(
            meta_confidence=0.70,
            magnitude=0.0,
            config=config,
        )
        assert result == 0.0

    def test_negative_magnitude_returns_zero(self, config: PositionSizingConfig) -> None:
        result = compute_position_size(
            meta_confidence=0.70,
            magnitude=-0.02,
            config=config,
        )
        assert result == 0.0

    def test_caps_at_max_single_stock(self, config: PositionSizingConfig) -> None:
        result = compute_position_size(
            meta_confidence=0.99,
            magnitude=0.10,
            config=config,
        )
        assert result == config.max_single_stock_pct

    def test_negative_kelly_returns_zero(self, config: PositionSizingConfig) -> None:
        result = compute_position_size(
            meta_confidence=0.40,
            magnitude=0.05,
            config=PositionSizingConfig(
                method="half_kelly",
                max_single_stock_pct=10.0,
                min_confidence=0.30,
            ),
        )
        assert result == 0.0
