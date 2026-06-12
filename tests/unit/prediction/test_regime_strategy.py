"""Tests for regime-conditional strategy selection."""

from __future__ import annotations

from alphavedha.prediction.regime_strategy import (
    RegimeStrategyConfig,
    RegimeStrategySelector,
)


class TestRegimeStrategySelector:
    def test_bull_full_kelly(self) -> None:
        selector = RegimeStrategySelector()
        result = selector.select("bull")
        assert result.regime == "bull"
        assert result.kelly_fraction == 1.0
        assert result.meta_confidence_threshold == 0.40

    def test_bear_quarter_kelly(self) -> None:
        selector = RegimeStrategySelector()
        result = selector.select("bear")
        assert result.regime == "bear"
        assert result.kelly_fraction == 0.25
        assert result.meta_confidence_threshold == 0.45

    def test_high_vol_requires_agreement(self) -> None:
        selector = RegimeStrategySelector()
        result = selector.select("high_volatility")
        assert result.require_all_models_agree is True
        assert result.kelly_fraction == 0.1
        assert result.meta_confidence_threshold == 0.52

    def test_sideways_half_kelly(self) -> None:
        selector = RegimeStrategySelector()
        result = selector.select("sideways")
        assert result.regime == "sideways"
        assert result.kelly_fraction == 0.5

    def test_unknown_regime_falls_back(self) -> None:
        selector = RegimeStrategySelector()
        result = selector.select("unknown")
        assert result.regime == "sideways"
        assert len(result.warnings) == 1
        assert "Unknown regime" in result.warnings[0]

    def test_model_weights_sum_to_one(self) -> None:
        selector = RegimeStrategySelector()
        for regime in ("bull", "bear", "sideways", "high_volatility"):
            result = selector.select(regime)
            total = sum(result.model_weights.values())
            assert abs(total - 1.0) < 0.01, f"{regime}: weights sum to {total}"

    def test_custom_config(self) -> None:
        from alphavedha.prediction.regime_strategy import RegimeStrategyParams

        config = RegimeStrategyConfig(
            bull=RegimeStrategyParams(kelly_fraction=0.8),
        )
        selector = RegimeStrategySelector(config)
        result = selector.select("bull")
        assert result.kelly_fraction == 0.8
