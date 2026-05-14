"""Tests for XGBoostModel — classifier + regressor."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from alphavedha.config import XGBoostConfig
from alphavedha.exceptions import ModelTrainingError
from alphavedha.models.base import PredictionResult, TrainResult
from alphavedha.models.xgboost_model import XGBoostModel


@pytest.fixture
def xgb_config() -> XGBoostConfig:
    return XGBoostConfig()


@pytest.fixture
def synthetic_data() -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Synthetic training data: 200 samples, 10 features, 3-class labels + returns."""
    rng = np.random.default_rng(42)
    n, f = 200, 10
    X = pd.DataFrame(rng.standard_normal((n, f)), columns=[f"f{i}" for i in range(f)])
    labels = pd.Series(rng.choice([-1, 0, 1], size=n), name="label")
    returns = pd.Series(rng.normal(0, 0.02, size=n), name="return_pct")
    return X, labels, returns


class TestXGBoostModel:
    def test_fit_returns_train_result(
        self,
        synthetic_data: tuple[pd.DataFrame, pd.Series, pd.Series],
        xgb_config: XGBoostConfig,
    ) -> None:
        X, labels, returns = synthetic_data
        model = XGBoostModel(config=xgb_config)
        result = model.fit(
            X_train=X[:160], y_train=labels[:160],
            X_val=X[160:], y_val=labels[160:],
            return_train=returns[:160], return_val=returns[160:],
        )
        assert isinstance(result, TrainResult)
        assert "accuracy" in result.train_metrics
        assert "rmse" in result.train_metrics

    def test_predict_returns_prediction_result(
        self,
        synthetic_data: tuple[pd.DataFrame, pd.Series, pd.Series],
        xgb_config: XGBoostConfig,
    ) -> None:
        X, labels, returns = synthetic_data
        model = XGBoostModel(config=xgb_config)
        model.fit(
            X_train=X[:160], y_train=labels[:160],
            X_val=X[160:], y_val=labels[160:],
            return_train=returns[:160], return_val=returns[160:],
        )
        pred = model.predict(X[160:])
        assert isinstance(pred, PredictionResult)

    def test_direction_values(
        self,
        synthetic_data: tuple[pd.DataFrame, pd.Series, pd.Series],
        xgb_config: XGBoostConfig,
    ) -> None:
        X, labels, returns = synthetic_data
        model = XGBoostModel(config=xgb_config)
        model.fit(
            X_train=X[:160], y_train=labels[:160],
            X_val=X[160:], y_val=labels[160:],
            return_train=returns[:160], return_val=returns[160:],
        )
        pred = model.predict(X[160:])
        assert set(np.unique(pred.direction)).issubset({-1, 0, 1})

    def test_magnitude_shape(
        self,
        synthetic_data: tuple[pd.DataFrame, pd.Series, pd.Series],
        xgb_config: XGBoostConfig,
    ) -> None:
        X, labels, returns = synthetic_data
        model = XGBoostModel(config=xgb_config)
        model.fit(
            X_train=X[:160], y_train=labels[:160],
            X_val=X[160:], y_val=labels[160:],
            return_train=returns[:160], return_val=returns[160:],
        )
        pred = model.predict(X[160:])
        assert len(pred.magnitude) == 40

    def test_probabilities_shape_and_sum(
        self,
        synthetic_data: tuple[pd.DataFrame, pd.Series, pd.Series],
        xgb_config: XGBoostConfig,
    ) -> None:
        X, labels, returns = synthetic_data
        model = XGBoostModel(config=xgb_config)
        model.fit(
            X_train=X[:160], y_train=labels[:160],
            X_val=X[160:], y_val=labels[160:],
            return_train=returns[:160], return_val=returns[160:],
        )
        pred = model.predict(X[160:])
        assert pred.probabilities is not None
        assert pred.probabilities.shape == (40, 3)
        row_sums = pred.probabilities.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-6)

    def test_confidence_range(
        self,
        synthetic_data: tuple[pd.DataFrame, pd.Series, pd.Series],
        xgb_config: XGBoostConfig,
    ) -> None:
        X, labels, returns = synthetic_data
        model = XGBoostModel(config=xgb_config)
        model.fit(
            X_train=X[:160], y_train=labels[:160],
            X_val=X[160:], y_val=labels[160:],
            return_train=returns[:160], return_val=returns[160:],
        )
        pred = model.predict(X[160:])
        assert (pred.confidence >= 0).all()
        assert (pred.confidence <= 1).all()

    def test_feature_importance(
        self,
        synthetic_data: tuple[pd.DataFrame, pd.Series, pd.Series],
        xgb_config: XGBoostConfig,
    ) -> None:
        X, labels, returns = synthetic_data
        model = XGBoostModel(config=xgb_config)
        model.fit(
            X_train=X[:160], y_train=labels[:160],
            X_val=X[160:], y_val=labels[160:],
            return_train=returns[:160], return_val=returns[160:],
        )
        fi = model.get_feature_importance()
        assert fi is not None
        assert len(fi) == 10
        assert (fi >= 0).all()

    def test_predict_before_fit_raises(self, xgb_config: XGBoostConfig) -> None:
        model = XGBoostModel(config=xgb_config)
        X = pd.DataFrame({"a": [1, 2]})
        with pytest.raises(ModelTrainingError):
            model.predict(X)

    def test_save_load_roundtrip(
        self,
        synthetic_data: tuple[pd.DataFrame, pd.Series, pd.Series],
        xgb_config: XGBoostConfig,
        tmp_path: Path,
    ) -> None:
        X, labels, returns = synthetic_data
        model = XGBoostModel(config=xgb_config)
        model.fit(
            X_train=X[:160], y_train=labels[:160],
            X_val=X[160:], y_val=labels[160:],
            return_train=returns[:160], return_val=returns[160:],
        )
        pred_before = model.predict(X[160:])

        save_dir = tmp_path / "xgb_test"
        model.save(save_dir)

        loaded = XGBoostModel.load(save_dir)
        pred_after = loaded.predict(X[160:])

        np.testing.assert_array_equal(pred_before.direction, pred_after.direction)
        np.testing.assert_allclose(pred_before.magnitude, pred_after.magnitude, atol=1e-6)

    def test_sample_weight_changes_result(
        self,
        synthetic_data: tuple[pd.DataFrame, pd.Series, pd.Series],
        xgb_config: XGBoostConfig,
    ) -> None:
        X, labels, returns = synthetic_data
        model1 = XGBoostModel(config=xgb_config)
        model1.fit(
            X_train=X[:160], y_train=labels[:160],
            X_val=X[160:], y_val=labels[160:],
            return_train=returns[:160], return_val=returns[160:],
        )
        pred1 = model1.predict(X[160:])

        rng = np.random.default_rng(99)
        weights = pd.Series(rng.uniform(0.1, 10.0, size=160))
        model2 = XGBoostModel(config=xgb_config)
        model2.fit(
            X_train=X[:160], y_train=labels[:160],
            X_val=X[160:], y_val=labels[160:],
            return_train=returns[:160], return_val=returns[160:],
            sample_weight=weights,
        )
        pred2 = model2.predict(X[160:])

        assert not np.array_equal(pred1.magnitude, pred2.magnitude)

    def test_config_hyperparams_applied(self) -> None:
        config = XGBoostConfig()
        model = XGBoostModel(config=config)
        assert model._xgb_params["learning_rate"] == 0.05
        assert model._xgb_params["max_depth"] == 6
