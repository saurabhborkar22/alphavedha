"""Tests for ModelRegistry — demo mode mock models and real mode artifact loading."""

from __future__ import annotations

import pytest

from alphavedha.exceptions import ModelNotFoundError
from alphavedha.prediction.engine import StockPrediction
from alphavedha.services.model_registry import ModelRegistry


class TestModelRegistryDemo:
    """Tests for demo mode — synthetic models, deterministic predictions."""

    def test_demo_mode_returns_prediction_engine(self) -> None:
        registry = ModelRegistry(demo=True)
        engine = registry.get_prediction_engine()
        assert engine is not None

    def test_demo_version_string(self) -> None:
        registry = ModelRegistry(demo=True)
        assert registry.model_version == "demo-v0.1.0"

    def test_demo_engine_produces_valid_stock_prediction(self) -> None:
        registry = ModelRegistry(demo=True)
        engine = registry.get_prediction_engine()
        features = registry.get_demo_features("TCS")
        result = engine.predict(symbol="TCS", features=features)
        assert isinstance(result, StockPrediction)
        assert result.symbol == "TCS"
        assert result.direction in (-1, 0, 1)
        assert 0.0 <= result.composite_score <= 100.0
        assert result.model_version == "demo-v0.1.0"

    def test_demo_predictions_are_deterministic(self) -> None:
        registry = ModelRegistry(demo=True)
        engine = registry.get_prediction_engine()
        features = registry.get_demo_features("TCS")
        r1 = engine.predict(symbol="TCS", features=features)
        r2 = engine.predict(symbol="TCS", features=features)
        assert r1.direction == r2.direction
        assert r1.magnitude == r2.magnitude
        assert r1.composite_score == r2.composite_score

    def test_demo_symbols_list(self) -> None:
        registry = ModelRegistry(demo=True)
        symbols = registry.get_demo_symbols()
        assert len(symbols) == 15
        assert "TCS" in symbols
        assert "RELIANCE" in symbols
        assert "INFY" in symbols

    def test_demo_features_shape(self) -> None:
        registry = ModelRegistry(demo=True)
        features = registry.get_demo_features("TCS")
        assert features.shape[0] == 1
        assert features.shape[1] > 0

    def test_demo_features_are_deterministic(self) -> None:
        registry = ModelRegistry(demo=True)
        f1 = registry.get_demo_features("TCS")
        f2 = registry.get_demo_features("TCS")
        assert f1.equals(f2)

    def test_different_symbols_produce_different_predictions(self) -> None:
        registry = ModelRegistry(demo=True)
        engine = registry.get_prediction_engine()
        f_tcs = registry.get_demo_features("TCS")
        f_infy = registry.get_demo_features("INFY")
        r_tcs = engine.predict(symbol="TCS", features=f_tcs)
        r_infy = engine.predict(symbol="INFY", features=f_infy)
        # At least one field should differ between different symbols
        differs = (
            r_tcs.direction != r_infy.direction
            or r_tcs.magnitude != r_infy.magnitude
            or r_tcs.composite_score != r_infy.composite_score
        )
        assert differs


class TestModelRegistryReal:
    """Tests for real mode — raises ModelNotFoundError when artifacts missing."""

    def test_real_mode_raises_when_no_artifacts(self, tmp_path: object) -> None:
        registry = ModelRegistry(demo=False)
        with pytest.raises(ModelNotFoundError):
            registry.get_prediction_engine()

    def test_real_version_string(self) -> None:
        registry = ModelRegistry(demo=False)
        assert registry.model_version == "v0.1.0"
