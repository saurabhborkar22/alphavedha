"""Tests for LSTMModel — dual-head LSTM (classification + regression)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from alphavedha.config import LSTMConfig
from alphavedha.exceptions import ModelTrainingError
from alphavedha.models.base import PredictionResult, TrainResult
from alphavedha.models.lstm_model import LSTMModel


@pytest.fixture
def lstm_config() -> LSTMConfig:
    return LSTMConfig(
        sequence_length=10,
        hidden_size=16,
        num_layers=2,
        dropout=0.1,
        learning_rate=0.01,
        batch_size=32,
        max_epochs=3,
        early_stopping_patience=2,
        top_n_features=10,
    )


@pytest.fixture
def synthetic_data() -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """200 samples, 10 features, 3-class labels + returns."""
    rng = np.random.default_rng(42)
    n, f = 200, 10
    X = pd.DataFrame(rng.standard_normal((n, f)), columns=[f"f{i}" for i in range(f)])
    labels = pd.Series(rng.choice([-1, 0, 1], size=n, p=[0.3, 0.4, 0.3]), name="label")
    returns = pd.Series(rng.normal(0, 0.02, size=n), name="return_pct")
    return X, labels, returns


class TestLSTMModel:
    def test_fit_returns_train_result(
        self,
        synthetic_data: tuple[pd.DataFrame, pd.Series, pd.Series],
        lstm_config: LSTMConfig,
    ) -> None:
        X, labels, returns = synthetic_data
        model = LSTMModel(config=lstm_config)
        result = model.fit(
            X_train=X[:160], y_train=labels[:160],
            X_val=X[160:], y_val=labels[160:],
            return_train=returns[:160], return_val=returns[160:],
        )
        assert isinstance(result, TrainResult)
        assert "accuracy" in result.train_metrics
        assert "f1_weighted" in result.train_metrics
        assert "rmse" in result.train_metrics

    def test_predict_returns_prediction_result(
        self,
        synthetic_data: tuple[pd.DataFrame, pd.Series, pd.Series],
        lstm_config: LSTMConfig,
    ) -> None:
        X, labels, returns = synthetic_data
        model = LSTMModel(config=lstm_config)
        model.fit(
            X_train=X[:160], y_train=labels[:160],
            return_train=returns[:160],
        )
        pred = model.predict(X[160:])
        assert isinstance(pred, PredictionResult)

    def test_predict_output_length_matches_input(
        self,
        synthetic_data: tuple[pd.DataFrame, pd.Series, pd.Series],
        lstm_config: LSTMConfig,
    ) -> None:
        X, labels, returns = synthetic_data
        model = LSTMModel(config=lstm_config)
        model.fit(X_train=X[:160], y_train=labels[:160], return_train=returns[:160])
        pred = model.predict(X[160:])
        assert len(pred.direction) == 40
        assert len(pred.magnitude) == 40
        assert len(pred.confidence) == 40
        assert pred.probabilities is not None
        assert pred.probabilities.shape[0] == 40

    def test_direction_values(
        self,
        synthetic_data: tuple[pd.DataFrame, pd.Series, pd.Series],
        lstm_config: LSTMConfig,
    ) -> None:
        X, labels, returns = synthetic_data
        model = LSTMModel(config=lstm_config)
        model.fit(X_train=X[:160], y_train=labels[:160], return_train=returns[:160])
        pred = model.predict(X[160:])
        assert set(np.unique(pred.direction)).issubset({-1, 0, 1})

    def test_warmup_rows_are_neutral(
        self,
        synthetic_data: tuple[pd.DataFrame, pd.Series, pd.Series],
        lstm_config: LSTMConfig,
    ) -> None:
        X, labels, returns = synthetic_data
        model = LSTMModel(config=lstm_config)
        model.fit(X_train=X[:160], y_train=labels[:160], return_train=returns[:160])
        pred = model.predict(X[160:])
        seq_len = lstm_config.sequence_length
        # First seq_len-1 rows should be neutral padding
        assert all(pred.direction[:seq_len - 1] == 0)
        assert all(pred.confidence[:seq_len - 1] == 0.0)
        np.testing.assert_array_equal(
            pred.probabilities[:seq_len - 1],
            np.tile([0.0, 1.0, 0.0], (seq_len - 1, 1)),
        )

    def test_probabilities_shape_and_sum(
        self,
        synthetic_data: tuple[pd.DataFrame, pd.Series, pd.Series],
        lstm_config: LSTMConfig,
    ) -> None:
        X, labels, returns = synthetic_data
        model = LSTMModel(config=lstm_config)
        model.fit(X_train=X[:160], y_train=labels[:160], return_train=returns[:160])
        pred = model.predict(X[160:])
        assert pred.probabilities is not None
        assert pred.probabilities.shape == (40, 3)
        row_sums = pred.probabilities.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-5)

    def test_confidence_range(
        self,
        synthetic_data: tuple[pd.DataFrame, pd.Series, pd.Series],
        lstm_config: LSTMConfig,
    ) -> None:
        X, labels, returns = synthetic_data
        model = LSTMModel(config=lstm_config)
        model.fit(X_train=X[:160], y_train=labels[:160], return_train=returns[:160])
        pred = model.predict(X[160:])
        assert (pred.confidence >= 0).all()
        assert (pred.confidence <= 1).all()

    def test_magnitude_shape(
        self,
        synthetic_data: tuple[pd.DataFrame, pd.Series, pd.Series],
        lstm_config: LSTMConfig,
    ) -> None:
        X, labels, returns = synthetic_data
        model = LSTMModel(config=lstm_config)
        model.fit(X_train=X[:160], y_train=labels[:160], return_train=returns[:160])
        pred = model.predict(X[160:])
        assert len(pred.magnitude) == 40

    def test_predict_before_fit_raises(self, lstm_config: LSTMConfig) -> None:
        model = LSTMModel(config=lstm_config)
        X = pd.DataFrame({"a": range(20), "b": range(20)})
        with pytest.raises(ModelTrainingError):
            model.predict(X)

    def test_save_load_roundtrip(
        self,
        synthetic_data: tuple[pd.DataFrame, pd.Series, pd.Series],
        lstm_config: LSTMConfig,
        tmp_path: Path,
    ) -> None:
        X, labels, returns = synthetic_data
        model = LSTMModel(config=lstm_config)
        model.fit(X_train=X[:160], y_train=labels[:160], return_train=returns[:160])
        pred_before = model.predict(X[160:])

        save_dir = tmp_path / "lstm_test"
        model.save(save_dir)

        loaded = LSTMModel.load(save_dir)
        pred_after = loaded.predict(X[160:])

        np.testing.assert_array_equal(pred_before.direction, pred_after.direction)
        np.testing.assert_allclose(pred_before.magnitude, pred_after.magnitude, atol=1e-5)

    def test_feature_importance_returns_none(self, lstm_config: LSTMConfig) -> None:
        model = LSTMModel(config=lstm_config)
        assert model.get_feature_importance() is None

    def test_sample_weight_accepted(
        self,
        synthetic_data: tuple[pd.DataFrame, pd.Series, pd.Series],
        lstm_config: LSTMConfig,
    ) -> None:
        X, labels, returns = synthetic_data
        rng = np.random.default_rng(99)
        weights = pd.Series(rng.uniform(0.1, 5.0, size=160))
        model = LSTMModel(config=lstm_config)
        result = model.fit(
            X_train=X[:160], y_train=labels[:160],
            return_train=returns[:160],
            sample_weight=weights,
        )
        assert isinstance(result, TrainResult)
