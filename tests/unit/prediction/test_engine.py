"""Tests for PredictionEngine — full pipeline orchestration with mocked models."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from alphavedha.exceptions import PredictionError
from alphavedha.models.base import PredictionResult
from alphavedha.models.conformal import ConformalResult
from alphavedha.models.ensemble import EnsembleResult
from alphavedha.models.meta_model import MetaLabelResult
from alphavedha.models.regime import RegimeResult
from alphavedha.prediction.engine import PredictionEngine, StockPrediction
from alphavedha.prediction.scorer import CompositeScorer
from alphavedha.risk.risk_manager import RiskAssessment, RiskManager


def _mock_base_model(name: str) -> MagicMock:
    model = MagicMock()
    model.predict.return_value = PredictionResult(
        direction=np.array([1]),
        magnitude=np.array([0.03]),
        probabilities=np.array([[0.1, 0.2, 0.7]]),
        confidence=np.array([0.7]),
    )
    return model


def _mock_regime() -> MagicMock:
    regime = MagicMock()
    regime.predict.return_value = RegimeResult(
        current_regime="bull",
        regime_id=0,
        state_probabilities=np.array([0.7, 0.1, 0.1, 0.1]),
        regime_history=np.array([0]),
        transition_matrix=np.eye(4),
    )
    return regime


def _mock_ensemble() -> MagicMock:
    ens = MagicMock()
    ens.predict.return_value = EnsembleResult(
        direction=np.array([1]),
        magnitude=np.array([0.03]),
        probabilities=np.array([[0.1, 0.2, 0.7]]),
        confidence=np.array([0.75]),
        model_disagreement=np.array([0.05]),
    )
    return ens


def _mock_meta() -> MagicMock:
    meta = MagicMock()
    meta.predict.return_value = MetaLabelResult(
        meta_confidence=np.array([0.72]),
        is_tradeable=np.array([True]),
    )
    return meta


def _mock_conformal() -> MagicMock:
    conf = MagicMock()
    conf.predict.return_value = ConformalResult(
        price_low=np.array([95.0]),
        price_mid=np.array([100.0]),
        price_high=np.array([105.0]),
        interval_width=np.array([10.0]),
        coverage=0.90,
    )
    return conf


def _mock_risk_manager() -> MagicMock:
    rm = MagicMock(spec=RiskManager)
    rm.assess.return_value = RiskAssessment(
        position_size_pct=5.0,
        kelly_raw=0.40,
        kelly_half=0.20,
        constraint_violations=[],
        circuit_breaker_level=0,
        risk_adjusted=False,
    )
    return rm


@pytest.fixture
def engine() -> PredictionEngine:
    return PredictionEngine(
        xgboost=_mock_base_model("xgboost"),
        lstm=_mock_base_model("lstm"),
        tft=_mock_base_model("tft"),
        regime=_mock_regime(),
        ensemble=_mock_ensemble(),
        meta_model=_mock_meta(),
        conformal=_mock_conformal(),
        scorer=CompositeScorer(),
        risk_manager=_mock_risk_manager(),
        model_version="v0.1.0",
    )


@pytest.fixture
def features() -> pd.DataFrame:
    return pd.DataFrame({"feature1": [1.0], "feature2": [2.0]})


@pytest.fixture
def returns() -> pd.Series:
    return pd.Series([0.01])


class TestPredictionEngine:
    def test_predict_returns_stock_prediction(
        self, engine: PredictionEngine, features: pd.DataFrame, returns: pd.Series
    ) -> None:
        market_features = pd.DataFrame({"returns": [0.01], "volatility": [0.02]})
        result = engine.predict(
            symbol="TCS",
            features=features,
            returns=returns,
            current_price=3500.0,
            market_features=market_features,
        )
        assert isinstance(result, StockPrediction)
        assert result.symbol == "TCS"
        assert result.direction in (-1, 0, 1)
        assert 0.0 <= result.composite_score <= 100.0
        assert result.model_version == "v0.1.0"
        assert len(result.warnings) == 0

    def test_one_model_failure_degrades_gracefully(
        self, features: pd.DataFrame, returns: pd.Series
    ) -> None:
        lstm = _mock_base_model("lstm")
        lstm.predict.side_effect = RuntimeError("LSTM failed")
        engine = PredictionEngine(
            xgboost=_mock_base_model("xgboost"),
            lstm=lstm,
            tft=_mock_base_model("tft"),
            regime=_mock_regime(),
            ensemble=_mock_ensemble(),
            meta_model=_mock_meta(),
            conformal=_mock_conformal(),
            scorer=CompositeScorer(),
            risk_manager=_mock_risk_manager(),
        )
        result = engine.predict("TCS", features, returns, 3500.0)
        assert isinstance(result, StockPrediction)
        assert any("lstm" in w.lower() for w in result.warnings)

    def test_two_model_failures_raises(self, features: pd.DataFrame, returns: pd.Series) -> None:
        lstm = _mock_base_model("lstm")
        lstm.predict.side_effect = RuntimeError("LSTM failed")
        tft = _mock_base_model("tft")
        tft.predict.side_effect = RuntimeError("TFT failed")
        engine = PredictionEngine(
            xgboost=_mock_base_model("xgboost"),
            lstm=lstm,
            tft=tft,
            regime=_mock_regime(),
            ensemble=_mock_ensemble(),
            meta_model=_mock_meta(),
            conformal=_mock_conformal(),
            scorer=CompositeScorer(),
            risk_manager=_mock_risk_manager(),
        )
        with pytest.raises(PredictionError, match="fewer than 2"):
            engine.predict("TCS", features, returns, 3500.0)

    def test_all_models_fail_raises(self, features: pd.DataFrame, returns: pd.Series) -> None:
        xgb = _mock_base_model("xgboost")
        xgb.predict.side_effect = RuntimeError("XGB failed")
        lstm = _mock_base_model("lstm")
        lstm.predict.side_effect = RuntimeError("LSTM failed")
        tft = _mock_base_model("tft")
        tft.predict.side_effect = RuntimeError("TFT failed")
        engine = PredictionEngine(
            xgboost=xgb,
            lstm=lstm,
            tft=tft,
            regime=_mock_regime(),
            ensemble=_mock_ensemble(),
            meta_model=_mock_meta(),
            conformal=_mock_conformal(),
            scorer=CompositeScorer(),
            risk_manager=_mock_risk_manager(),
        )
        with pytest.raises(PredictionError):
            engine.predict("TCS", features, returns, 3500.0)

    def test_regime_failure_uses_default(self, features: pd.DataFrame, returns: pd.Series) -> None:
        regime = _mock_regime()
        regime.predict.side_effect = RuntimeError("Regime failed")
        engine = PredictionEngine(
            xgboost=_mock_base_model("xgboost"),
            lstm=_mock_base_model("lstm"),
            tft=_mock_base_model("tft"),
            regime=regime,
            ensemble=_mock_ensemble(),
            meta_model=_mock_meta(),
            conformal=_mock_conformal(),
            scorer=CompositeScorer(),
            risk_manager=_mock_risk_manager(),
        )
        market_features = pd.DataFrame({"returns": [0.01], "volatility": [0.02]})
        result = engine.predict("TCS", features, returns, 3500.0, market_features=market_features)
        assert any("regime" in w.lower() for w in result.warnings)
        np.testing.assert_allclose(result.regime_probabilities, [0.25, 0.25, 0.25, 0.25])

    def test_meta_model_failure_defaults_not_tradeable(
        self, features: pd.DataFrame, returns: pd.Series
    ) -> None:
        meta = _mock_meta()
        meta.predict.side_effect = RuntimeError("Meta failed")
        rm = _mock_risk_manager()
        rm.assess.return_value = RiskAssessment(
            position_size_pct=0.0,
            kelly_raw=0.0,
            kelly_half=0.0,
            constraint_violations=[],
            circuit_breaker_level=0,
            risk_adjusted=False,
        )
        engine = PredictionEngine(
            xgboost=_mock_base_model("xgboost"),
            lstm=_mock_base_model("lstm"),
            tft=_mock_base_model("tft"),
            regime=_mock_regime(),
            ensemble=_mock_ensemble(),
            meta_model=meta,
            conformal=_mock_conformal(),
            scorer=CompositeScorer(),
            risk_manager=rm,
        )
        result = engine.predict("TCS", features, returns, 3500.0)
        assert result.meta_confidence == 0.0
        assert result.is_tradeable is False
        assert any("meta" in w.lower() for w in result.warnings)

    def test_conformal_failure_uses_nan(self, features: pd.DataFrame, returns: pd.Series) -> None:
        conf = _mock_conformal()
        conf.predict.side_effect = RuntimeError("Conformal failed")
        engine = PredictionEngine(
            xgboost=_mock_base_model("xgboost"),
            lstm=_mock_base_model("lstm"),
            tft=_mock_base_model("tft"),
            regime=_mock_regime(),
            ensemble=_mock_ensemble(),
            meta_model=_mock_meta(),
            conformal=conf,
            scorer=CompositeScorer(),
            risk_manager=_mock_risk_manager(),
        )
        result = engine.predict("TCS", features, returns, 3500.0)
        assert np.isnan(result.price_target_low)
        assert np.isnan(result.price_target_mid)
        assert np.isnan(result.price_target_high)
        assert any("conformal" in w.lower() for w in result.warnings)

    def test_no_market_features_skips_regime(
        self, features: pd.DataFrame, returns: pd.Series
    ) -> None:
        regime = _mock_regime()
        engine = PredictionEngine(
            xgboost=_mock_base_model("xgboost"),
            lstm=_mock_base_model("lstm"),
            tft=_mock_base_model("tft"),
            regime=regime,
            ensemble=_mock_ensemble(),
            meta_model=_mock_meta(),
            conformal=_mock_conformal(),
            scorer=CompositeScorer(),
            risk_manager=_mock_risk_manager(),
        )
        result = engine.predict("TCS", features, returns, 3500.0, market_features=None)
        regime.predict.assert_not_called()
        assert any("regime" in w.lower() or "market_features" in w.lower() for w in result.warnings)

    def test_position_zero_when_not_tradeable(
        self, features: pd.DataFrame, returns: pd.Series
    ) -> None:
        meta = _mock_meta()
        meta.predict.return_value = MetaLabelResult(
            meta_confidence=np.array([0.30]),
            is_tradeable=np.array([False]),
        )
        rm = _mock_risk_manager()
        rm.assess.return_value = RiskAssessment(
            position_size_pct=0.0,
            kelly_raw=0.0,
            kelly_half=0.0,
            constraint_violations=[],
            circuit_breaker_level=0,
            risk_adjusted=False,
        )
        engine = PredictionEngine(
            xgboost=_mock_base_model("xgboost"),
            lstm=_mock_base_model("lstm"),
            tft=_mock_base_model("tft"),
            regime=_mock_regime(),
            ensemble=_mock_ensemble(),
            meta_model=meta,
            conformal=_mock_conformal(),
            scorer=CompositeScorer(),
            risk_manager=rm,
        )
        result = engine.predict("TCS", features, returns, 3500.0)
        assert result.position_size_pct == 0.0
        assert result.is_tradeable is False
