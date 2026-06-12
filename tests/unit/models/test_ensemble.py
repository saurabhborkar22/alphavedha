"""Tests for StackingEnsemble — stacking meta-learner combining base model outputs."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from alphavedha.config import EnsembleConfig
from alphavedha.exceptions import DataQualityError, InsufficientDataError, ModelTrainingError
from alphavedha.models.base import PredictionResult
from alphavedha.models.ensemble import EnsembleResult, StackingEnsemble

_LABEL_REVERSE = {0: -1, 1: 0, 2: 1}


def _make_prediction_result(
    n: int, rng: np.random.Generator, bias_class: int = 2
) -> PredictionResult:
    raw = rng.random((n, 3))
    raw[:, bias_class] += 1.0
    probabilities = raw / raw.sum(axis=1, keepdims=True)
    direction_idx = np.argmax(probabilities, axis=1)
    direction = np.array([_LABEL_REVERSE[d] for d in direction_idx])
    confidence = np.max(probabilities, axis=1)
    magnitude = rng.normal(0.01, 0.005, size=n)
    return PredictionResult(
        direction=direction,
        magnitude=magnitude,
        probabilities=probabilities,
        confidence=confidence,
    )


@pytest.fixture
def ensemble_config() -> EnsembleConfig:
    return EnsembleConfig(meta_learner="ridge", alpha=1.0)


@pytest.fixture
def synthetic_ensemble_data() -> tuple[dict[str, PredictionResult], np.ndarray, pd.Series]:
    rng = np.random.default_rng(42)
    n = 100
    base_preds = {
        "xgboost": _make_prediction_result(n, rng, bias_class=2),
        "lstm": _make_prediction_result(n, rng, bias_class=2),
        "tft": _make_prediction_result(n, rng, bias_class=1),
    }
    regime_raw = rng.random((n, 4))
    regime_probs = regime_raw / regime_raw.sum(axis=1, keepdims=True)
    y_true = pd.Series(rng.choice([-1, 0, 1], size=n), name="direction")
    return base_preds, regime_probs, y_true


class TestStackingEnsembleFit:
    def test_fit_returns_metrics(
        self,
        synthetic_ensemble_data: tuple[dict[str, PredictionResult], np.ndarray, pd.Series],
        ensemble_config: EnsembleConfig,
    ) -> None:
        base_preds, regime_probs, y_true = synthetic_ensemble_data
        ensemble = StackingEnsemble(config=ensemble_config)
        metrics = ensemble.fit(base_preds, regime_probs, y_true)
        assert isinstance(metrics, dict)
        assert "train_accuracy" in metrics
        assert "train_f1_weighted" in metrics

    def test_predict_before_fit_raises(self, ensemble_config: EnsembleConfig) -> None:
        ensemble = StackingEnsemble(config=ensemble_config)
        rng = np.random.default_rng(0)
        preds = {name: _make_prediction_result(10, rng) for name in ["xgboost", "lstm", "tft"]}
        regime = np.ones((10, 4)) / 4
        with pytest.raises(ModelTrainingError):
            ensemble.predict(preds, regime)

    def test_nan_input_raises(self, ensemble_config: EnsembleConfig) -> None:
        rng = np.random.default_rng(0)
        n = 50
        preds = {name: _make_prediction_result(n, rng) for name in ["xgboost", "lstm", "tft"]}
        regime = np.ones((n, 4)) / 4
        preds["xgboost"].probabilities[0, 0] = np.nan
        y = pd.Series(rng.choice([-1, 0, 1], size=n))
        ensemble = StackingEnsemble(config=ensemble_config)
        with pytest.raises(DataQualityError):
            ensemble.fit(preds, regime, y)

    def test_inf_input_raises(self, ensemble_config: EnsembleConfig) -> None:
        rng = np.random.default_rng(0)
        n = 50
        preds = {name: _make_prediction_result(n, rng) for name in ["xgboost", "lstm", "tft"]}
        regime = np.ones((n, 4)) / 4
        regime[0, 0] = np.inf
        y = pd.Series(rng.choice([-1, 0, 1], size=n))
        ensemble = StackingEnsemble(config=ensemble_config)
        with pytest.raises(DataQualityError):
            ensemble.fit(preds, regime, y)

    def test_missing_model_name_raises(self, ensemble_config: EnsembleConfig) -> None:
        rng = np.random.default_rng(0)
        n = 50
        preds = {name: _make_prediction_result(n, rng) for name in ["xgboost", "lstm"]}
        regime = np.ones((n, 4)) / 4
        y = pd.Series(rng.choice([-1, 0, 1], size=n))
        ensemble = StackingEnsemble(config=ensemble_config)
        with pytest.raises(ValueError, match="Expected models"):
            ensemble.fit(preds, regime, y)

    def test_extra_model_name_raises(self, ensemble_config: EnsembleConfig) -> None:
        rng = np.random.default_rng(0)
        n = 50
        preds = {
            name: _make_prediction_result(n, rng) for name in ["xgboost", "lstm", "tft", "extra"]
        }
        regime = np.ones((n, 4)) / 4
        y = pd.Series(rng.choice([-1, 0, 1], size=n))
        ensemble = StackingEnsemble(config=ensemble_config)
        with pytest.raises(ValueError, match="Expected models"):
            ensemble.fit(preds, regime, y)

    def test_empty_predictions_raises(self, ensemble_config: EnsembleConfig) -> None:
        rng = np.random.default_rng(0)
        preds = {name: _make_prediction_result(0, rng) for name in ["xgboost", "lstm", "tft"]}
        regime = np.ones((0, 4))
        y = pd.Series([], dtype=int)
        ensemble = StackingEnsemble(config=ensemble_config)
        with pytest.raises(InsufficientDataError):
            ensemble.fit(preds, regime, y)

    def test_gnn_as_optional_fourth_learner(self, ensemble_config: EnsembleConfig) -> None:
        rng = np.random.default_rng(7)
        n = 50
        preds = {
            name: _make_prediction_result(n, rng) for name in ["xgboost", "lstm", "tft", "gnn"]
        }
        regime = np.ones((n, 4)) / 4
        y = pd.Series(rng.choice([-1, 0, 1], size=n))
        ensemble = StackingEnsemble(config=ensemble_config)
        metrics = ensemble.fit(preds, regime, y)
        assert ensemble._has_gnn is True
        assert "train_accuracy" in metrics
        result = ensemble.predict(preds, regime)
        assert len(result.direction) == n


class TestStackingEnsemblePredict:
    @pytest.fixture(autouse=True)
    def _fitted_ensemble(
        self,
        synthetic_ensemble_data: tuple[dict[str, PredictionResult], np.ndarray, pd.Series],
        ensemble_config: EnsembleConfig,
    ) -> None:
        base_preds, regime_probs, y_true = synthetic_ensemble_data
        self.ensemble = StackingEnsemble(config=ensemble_config)
        self.ensemble.fit(base_preds, regime_probs, y_true)
        self.base_preds = base_preds
        self.regime_probs = regime_probs
        self.result = self.ensemble.predict(base_preds, regime_probs)

    def test_predict_returns_ensemble_result(self) -> None:
        assert isinstance(self.result, EnsembleResult)

    def test_direction_values_valid(self) -> None:
        unique = set(np.unique(self.result.direction))
        assert unique.issubset({-1, 0, 1})

    def test_probabilities_sum_to_one(self) -> None:
        row_sums = self.result.probabilities.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-6)

    def test_confidence_range(self) -> None:
        assert np.all(self.result.confidence >= 0.0)
        assert np.all(self.result.confidence <= 1.0)

    def test_model_disagreement_is_std(self) -> None:
        stacked = np.stack(
            [self.base_preds[m].probabilities for m in ["xgboost", "lstm", "tft"]],
            axis=0,
        )
        mean_probs = stacked.mean(axis=0)
        consensus = np.argmax(mean_probs, axis=1)
        n = len(consensus)
        expected = np.std(stacked[:, np.arange(n), consensus], axis=0)
        np.testing.assert_allclose(self.result.model_disagreement, expected, atol=1e-10)

    def test_magnitude_is_weighted_average(self) -> None:
        confs = np.stack(
            [self.base_preds[m].confidence for m in ["xgboost", "lstm", "tft"]],
            axis=0,
        )
        mags = np.stack(
            [self.base_preds[m].magnitude for m in ["xgboost", "lstm", "tft"]],
            axis=0,
        )
        conf_sum = confs.sum(axis=0, keepdims=True)
        conf_sum = np.where(conf_sum == 0, 1.0, conf_sum)
        weights = confs / conf_sum
        expected = (weights * mags).sum(axis=0)
        np.testing.assert_allclose(self.result.magnitude, expected, atol=1e-10)


class TestStackingEnsemblePersistence:
    def test_save_load_roundtrip(
        self,
        synthetic_ensemble_data: tuple[dict[str, PredictionResult], np.ndarray, pd.Series],
        ensemble_config: EnsembleConfig,
        tmp_path: Path,
    ) -> None:
        base_preds, regime_probs, y_true = synthetic_ensemble_data
        ensemble = StackingEnsemble(config=ensemble_config)
        ensemble.fit(base_preds, regime_probs, y_true)
        result_before = ensemble.predict(base_preds, regime_probs)

        save_dir = tmp_path / "ensemble_test"
        ensemble.save(save_dir)

        loaded = StackingEnsemble.load(save_dir)
        result_after = loaded.predict(base_preds, regime_probs)

        np.testing.assert_array_equal(result_before.direction, result_after.direction)
        np.testing.assert_allclose(
            result_before.probabilities, result_after.probabilities, atol=1e-10
        )
