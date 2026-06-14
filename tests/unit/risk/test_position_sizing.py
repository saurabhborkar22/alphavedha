"""Tests for generalized half-Kelly position sizing."""

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
        magnitude_loss_ref=0.02,
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

    def test_generalized_kelly_formula(self, config: PositionSizingConfig) -> None:
        # p=0.70, magnitude=0.02 → b=1 → kelly = p - q/b = 0.70 - 0.30 = 0.40
        # half-kelly = 0.40 * 0.5 * 100 = 20.0  → capped at 10.0
        result = compute_position_size(meta_confidence=0.70, magnitude=0.02, config=config)
        assert abs(result - 10.0) < 1e-9  # capped

        # p=0.60, magnitude=0.01 → b=0.5 → kelly = 0.60 - 0.40/0.5 = 0.60 - 0.80 = -0.20 → 0
        result_neg = compute_position_size(meta_confidence=0.60, magnitude=0.01, config=config)
        assert result_neg == 0.0

        # p=0.65, magnitude=0.04 → b=2 → kelly = 0.65 - 0.35/2 = 0.65 - 0.175 = 0.475
        # half-kelly = 0.475 * 0.5 * 100 = 23.75 → capped at 10.0
        result_high = compute_position_size(meta_confidence=0.65, magnitude=0.04, config=config)
        assert abs(result_high - 10.0) < 1e-9

    def test_higher_magnitude_gives_larger_position(self, config: PositionSizingConfig) -> None:
        # Same confidence, different magnitudes → higher magnitude = larger (or equal-capped) position
        low = compute_position_size(meta_confidence=0.62, magnitude=0.01, config=config)
        mid = compute_position_size(meta_confidence=0.62, magnitude=0.02, config=config)
        high = compute_position_size(meta_confidence=0.62, magnitude=0.04, config=config)
        assert low <= mid <= high

    def test_magnitude_at_ref_equals_symmetric_kelly(self, config: PositionSizingConfig) -> None:
        # When magnitude == magnitude_loss_ref, b=1, generalized Kelly == symmetric Kelly
        p = 0.65
        q = 1 - p
        symmetric_half_kelly = (p - q) * 0.5 * 100  # = (2p-1)*0.5*100
        result = compute_position_size(
            meta_confidence=p,
            magnitude=config.magnitude_loss_ref,
            config=config,
        )
        expected = min(symmetric_half_kelly, config.max_single_stock_pct)
        assert abs(result - expected) < 1e-9

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

    def test_negative_generalized_kelly_returns_zero(self, config: PositionSizingConfig) -> None:
        # p=0.60, magnitude=0.005 → b=0.25 → kelly = 0.60 - 0.40/0.25 = 0.60 - 1.60 = -1.0
        result = compute_position_size(
            meta_confidence=0.60,
            magnitude=0.005,
            config=PositionSizingConfig(
                method="half_kelly",
                max_single_stock_pct=10.0,
                min_confidence=0.55,
                magnitude_loss_ref=0.02,
            ),
        )
        assert result == 0.0

    def test_caps_at_max_single_stock(self, config: PositionSizingConfig) -> None:
        result = compute_position_size(
            meta_confidence=0.99,
            magnitude=0.10,
            config=config,
        )
        assert result == config.max_single_stock_pct
