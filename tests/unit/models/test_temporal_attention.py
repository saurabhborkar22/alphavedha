"""Tests for TemporalAttentionModel — TFT-lite with VSN, GRN, multi-head attention."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from alphavedha.config import TFTConfig
from alphavedha.exceptions import ModelTrainingError
from alphavedha.models.base import PredictionResult, TrainResult
from alphavedha.models.temporal_attention import TemporalAttentionModel


@pytest.fixture
def tft_config() -> TFTConfig:
    return TFTConfig(
        sequence_length=10,
        hidden_size=16,
        attention_head_size=4,
        num_layers=1,
        dropout=0.1,
        learning_rate=0.01,
        batch_size=32,
        max_epochs=3,
        early_stopping_patience=2,
        horizons=[7, 15, 30],
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


class TestTemporalAttentionModel:
    def test_fit_returns_train_result(
        self,
        synthetic_data: tuple[pd.DataFrame, pd.Series, pd.Series],
        tft_config: TFTConfig,
    ) -> None:
        X, labels, returns = synthetic_data
        model = TemporalAttentionModel(config=tft_config)
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
        tft_config: TFTConfig,
    ) -> None:
        X, labels, returns = synthetic_data
        model = TemporalAttentionModel(config=tft_config)
        model.fit(X_train=X[:160], y_train=labels[:160], return_train=returns[:160])
        pred = model.predict(X[160:])
        assert isinstance(pred, PredictionResult)

    def test_predict_output_length_matches_input(
        self,
        synthetic_data: tuple[pd.DataFrame, pd.Series, pd.Series],
        tft_config: TFTConfig,
    ) -> None:
        X, labels, returns = synthetic_data
        model = TemporalAttentionModel(config=tft_config)
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
        tft_config: TFTConfig,
    ) -> None:
        X, labels, returns = synthetic_data
        model = TemporalAttentionModel(config=tft_config)
        model.fit(X_train=X[:160], y_train=labels[:160], return_train=returns[:160])
        pred = model.predict(X[160:])
        assert set(np.unique(pred.direction)).issubset({-1, 0, 1})

    def test_warmup_rows_are_neutral(
        self,
        synthetic_data: tuple[pd.DataFrame, pd.Series, pd.Series],
        tft_config: TFTConfig,
    ) -> None:
        X, labels, returns = synthetic_data
        model = TemporalAttentionModel(config=tft_config)
        model.fit(X_train=X[:160], y_train=labels[:160], return_train=returns[:160])
        pred = model.predict(X[160:])
        seq_len = tft_config.sequence_length
        assert all(pred.direction[:seq_len - 1] == 0)
        assert all(pred.confidence[:seq_len - 1] == 0.0)

    def test_probabilities_shape_and_sum(
        self,
        synthetic_data: tuple[pd.DataFrame, pd.Series, pd.Series],
        tft_config: TFTConfig,
    ) -> None:
        X, labels, returns = synthetic_data
        model = TemporalAttentionModel(config=tft_config)
        model.fit(X_train=X[:160], y_train=labels[:160], return_train=returns[:160])
        pred = model.predict(X[160:])
        assert pred.probabilities is not None
        assert pred.probabilities.shape == (40, 3)
        row_sums = pred.probabilities.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-5)

    def test_confidence_range(
        self,
        synthetic_data: tuple[pd.DataFrame, pd.Series, pd.Series],
        tft_config: TFTConfig,
    ) -> None:
        X, labels, returns = synthetic_data
        model = TemporalAttentionModel(config=tft_config)
        model.fit(X_train=X[:160], y_train=labels[:160], return_train=returns[:160])
        pred = model.predict(X[160:])
        assert (pred.confidence >= 0).all()
        assert (pred.confidence <= 1).all()

    def test_magnitude_shape(
        self,
        synthetic_data: tuple[pd.DataFrame, pd.Series, pd.Series],
        tft_config: TFTConfig,
    ) -> None:
        X, labels, returns = synthetic_data
        model = TemporalAttentionModel(config=tft_config)
        model.fit(X_train=X[:160], y_train=labels[:160], return_train=returns[:160])
        pred = model.predict(X[160:])
        assert len(pred.magnitude) == 40

    def test_predict_before_fit_raises(self, tft_config: TFTConfig) -> None:
        model = TemporalAttentionModel(config=tft_config)
        X = pd.DataFrame({"a": range(20), "b": range(20)})
        with pytest.raises(ModelTrainingError):
            model.predict(X)

    def test_save_load_roundtrip(
        self,
        synthetic_data: tuple[pd.DataFrame, pd.Series, pd.Series],
        tft_config: TFTConfig,
        tmp_path: Path,
    ) -> None:
        X, labels, returns = synthetic_data
        model = TemporalAttentionModel(config=tft_config)
        model.fit(X_train=X[:160], y_train=labels[:160], return_train=returns[:160])
        pred_before = model.predict(X[160:])

        save_dir = tmp_path / "tft_test"
        model.save(save_dir)

        loaded = TemporalAttentionModel.load(save_dir)
        pred_after = loaded.predict(X[160:])

        np.testing.assert_array_equal(pred_before.direction, pred_after.direction)
        np.testing.assert_allclose(pred_before.magnitude, pred_after.magnitude, atol=1e-5)

    def test_feature_importance_returns_series(
        self,
        synthetic_data: tuple[pd.DataFrame, pd.Series, pd.Series],
        tft_config: TFTConfig,
    ) -> None:
        X, labels, returns = synthetic_data
        model = TemporalAttentionModel(config=tft_config)
        model.fit(X_train=X[:160], y_train=labels[:160], return_train=returns[:160])
        model.predict(X[160:])
        fi = model.get_feature_importance()
        assert fi is not None
        assert isinstance(fi, pd.Series)
        assert len(fi) == 10

    def test_feature_importance_sums_to_one(
        self,
        synthetic_data: tuple[pd.DataFrame, pd.Series, pd.Series],
        tft_config: TFTConfig,
    ) -> None:
        X, labels, returns = synthetic_data
        model = TemporalAttentionModel(config=tft_config)
        model.fit(X_train=X[:160], y_train=labels[:160], return_train=returns[:160])
        model.predict(X[160:])
        fi = model.get_feature_importance()
        assert fi is not None
        assert fi.sum() == pytest.approx(1.0, abs=1e-5)

    def test_horizon_predictions_available(
        self,
        synthetic_data: tuple[pd.DataFrame, pd.Series, pd.Series],
        tft_config: TFTConfig,
    ) -> None:
        X, labels, returns = synthetic_data
        model = TemporalAttentionModel(config=tft_config)
        model.fit(X_train=X[:160], y_train=labels[:160], return_train=returns[:160])
        model.predict(X[160:])
        horizons = model.get_horizon_predictions()
        assert set(horizons.keys()) == {7, 15, 30}
        for h, pred in horizons.items():
            assert isinstance(pred, PredictionResult)

    def test_horizon_prediction_shapes(
        self,
        synthetic_data: tuple[pd.DataFrame, pd.Series, pd.Series],
        tft_config: TFTConfig,
    ) -> None:
        X, labels, returns = synthetic_data
        model = TemporalAttentionModel(config=tft_config)
        model.fit(X_train=X[:160], y_train=labels[:160], return_train=returns[:160])
        model.predict(X[160:])
        horizons = model.get_horizon_predictions()
        for h, pred in horizons.items():
            assert len(pred.direction) == 40
            assert len(pred.magnitude) == 40

    def test_attention_weights_available(
        self,
        synthetic_data: tuple[pd.DataFrame, pd.Series, pd.Series],
        tft_config: TFTConfig,
    ) -> None:
        X, labels, returns = synthetic_data
        model = TemporalAttentionModel(config=tft_config)
        model.fit(X_train=X[:160], y_train=labels[:160], return_train=returns[:160])
        model.predict(X[160:])
        attn = model.get_attention_weights()
        assert attn is not None
        assert attn.ndim == 4  # (n_valid, n_heads, seq_len, seq_len)

    def test_sample_weight_accepted(
        self,
        synthetic_data: tuple[pd.DataFrame, pd.Series, pd.Series],
        tft_config: TFTConfig,
    ) -> None:
        X, labels, returns = synthetic_data
        rng = np.random.default_rng(99)
        weights = pd.Series(rng.uniform(0.1, 5.0, size=160))
        model = TemporalAttentionModel(config=tft_config)
        result = model.fit(
            X_train=X[:160], y_train=labels[:160],
            return_train=returns[:160],
            sample_weight=weights,
        )
        assert isinstance(result, TrainResult)
