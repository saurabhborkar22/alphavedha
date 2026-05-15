"""Tests for MetaLabelingModel — binary gate for ensemble signal confidence."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from alphavedha.config import MetaLabelingConfig
from alphavedha.exceptions import DataQualityError, InsufficientDataError, ModelTrainingError
from alphavedha.models.meta_model import MetaLabelingModel, MetaLabelResult


@pytest.fixture
def meta_config() -> MetaLabelingConfig:
    return MetaLabelingConfig(min_confidence=0.55, model="xgboost")


@pytest.fixture
def synthetic_meta_data() -> tuple[pd.DataFrame, np.ndarray, np.ndarray, pd.Series]:
    """Synthetic features + ensemble outputs + binary correctness labels."""
    rng = np.random.default_rng(42)
    n = 200
    X = pd.DataFrame(rng.standard_normal((n, 10)), columns=[f"f{i}" for i in range(10)])
    ensemble_direction = rng.choice([-1, 0, 1], size=n).astype(float)
    ensemble_confidence = rng.uniform(0.3, 0.95, size=n)
    y_correct = pd.Series(rng.integers(0, 2, size=n), name="correct")
    return X, ensemble_direction, ensemble_confidence, y_correct


class TestMetaLabelingModelFit:
    def test_fit_returns_metrics(
        self,
        synthetic_meta_data: tuple[pd.DataFrame, np.ndarray, np.ndarray, pd.Series],
        meta_config: MetaLabelingConfig,
    ) -> None:
        X, ens_dir, ens_conf, y_correct = synthetic_meta_data
        model = MetaLabelingModel(config=meta_config)
        metrics = model.fit(X, ens_dir, ens_conf, y_correct)
        assert isinstance(metrics, dict)
        assert "train_accuracy" in metrics
        assert "train_precision" in metrics
        assert "train_recall" in metrics
        assert "train_f1" in metrics

    def test_predict_before_fit_raises(self, meta_config: MetaLabelingConfig) -> None:
        model = MetaLabelingModel(config=meta_config)
        X = pd.DataFrame({"a": range(10), "b": range(10)})
        direction = np.zeros(10)
        confidence = np.ones(10) * 0.5
        with pytest.raises(ModelTrainingError):
            model.predict(X, direction, confidence)

    def test_nan_input_raises(self, meta_config: MetaLabelingConfig) -> None:
        rng = np.random.default_rng(0)
        n = 50
        X = pd.DataFrame(rng.standard_normal((n, 5)), columns=[f"f{i}" for i in range(5)])
        X.iloc[0, 0] = np.nan
        direction = rng.choice([-1, 0, 1], size=n).astype(float)
        confidence = rng.uniform(0.3, 0.9, size=n)
        y = pd.Series(rng.integers(0, 2, size=n))
        model = MetaLabelingModel(config=meta_config)
        with pytest.raises(DataQualityError):
            model.fit(X, direction, confidence, y)

    def test_empty_input_raises(self, meta_config: MetaLabelingConfig) -> None:
        X = pd.DataFrame(columns=["a", "b"])
        direction = np.array([])
        confidence = np.array([])
        y = pd.Series([], dtype=int)
        model = MetaLabelingModel(config=meta_config)
        with pytest.raises(InsufficientDataError):
            model.fit(X, direction, confidence, y)

    def test_fit_with_validation_set(
        self,
        synthetic_meta_data: tuple[pd.DataFrame, np.ndarray, np.ndarray, pd.Series],
        meta_config: MetaLabelingConfig,
    ) -> None:
        X, ens_dir, ens_conf, y_correct = synthetic_meta_data
        n_train = 150
        model = MetaLabelingModel(config=meta_config)
        metrics = model.fit(
            X[:n_train],
            ens_dir[:n_train],
            ens_conf[:n_train],
            y_correct[:n_train],
            X_val=X[n_train:],
            ensemble_direction_val=ens_dir[n_train:],
            ensemble_confidence_val=ens_conf[n_train:],
            y_correct_val=y_correct[n_train:],
        )
        assert isinstance(metrics, dict)
        assert "train_accuracy" in metrics


class TestMetaLabelingModelPredict:
    @pytest.fixture(autouse=True)
    def _fitted_model(
        self,
        synthetic_meta_data: tuple[pd.DataFrame, np.ndarray, np.ndarray, pd.Series],
        meta_config: MetaLabelingConfig,
    ) -> None:
        X, ens_dir, ens_conf, y_correct = synthetic_meta_data
        self.model = MetaLabelingModel(config=meta_config)
        self.model.fit(X, ens_dir, ens_conf, y_correct)
        self.X = X
        self.ens_dir = ens_dir
        self.ens_conf = ens_conf
        self.result = self.model.predict(X, ens_dir, ens_conf)

    def test_predict_returns_meta_label_result(self) -> None:
        assert isinstance(self.result, MetaLabelResult)

    def test_meta_confidence_range(self) -> None:
        assert np.all(self.result.meta_confidence >= 0.0)
        assert np.all(self.result.meta_confidence <= 1.0)

    def test_is_tradeable_respects_threshold(self) -> None:
        expected = self.result.meta_confidence > 0.55
        np.testing.assert_array_equal(self.result.is_tradeable, expected)

    def test_custom_threshold(self) -> None:
        config = MetaLabelingConfig(min_confidence=0.80)
        model = MetaLabelingModel(config=config)
        X, ens_dir, ens_conf, y_correct = (
            self.X,
            self.ens_dir,
            self.ens_conf,
            pd.Series(np.random.default_rng(42).integers(0, 2, size=len(self.X))),
        )
        model.fit(X, ens_dir, ens_conf, y_correct)
        result = model.predict(X, ens_dir, ens_conf)
        expected = result.meta_confidence > 0.80
        np.testing.assert_array_equal(result.is_tradeable, expected)


class TestMetaLabelingModelPersistence:
    def test_save_load_roundtrip(
        self,
        synthetic_meta_data: tuple[pd.DataFrame, np.ndarray, np.ndarray, pd.Series],
        meta_config: MetaLabelingConfig,
        tmp_path: Path,
    ) -> None:
        X, ens_dir, ens_conf, y_correct = synthetic_meta_data
        model = MetaLabelingModel(config=meta_config)
        model.fit(X, ens_dir, ens_conf, y_correct)
        result_before = model.predict(X, ens_dir, ens_conf)

        save_dir = tmp_path / "meta_model_test"
        model.save(save_dir)

        loaded = MetaLabelingModel.load(save_dir)
        result_after = loaded.predict(X, ens_dir, ens_conf)

        np.testing.assert_allclose(
            result_before.meta_confidence, result_after.meta_confidence, atol=1e-6
        )
        np.testing.assert_array_equal(result_before.is_tradeable, result_after.is_tradeable)
