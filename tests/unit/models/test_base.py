"""Tests for BaseModel ABC and result dataclasses."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from alphavedha.models.base import (
    BaseModel,
    PredictionResult,
    TrainResult,
)


class DummyModel(BaseModel):
    """Minimal BaseModel subclass for testing the ABC."""

    def fit(
        self,
        X_train: pd.DataFrame,
        _y_train: pd.Series,
        X_val: pd.DataFrame | None = None,
        _y_val: pd.Series | None = None,
        _sample_weight: pd.Series | None = None,
    ) -> TrainResult:
        self._is_fitted = True
        self._feature_names = list(X_train.columns)
        self._train_metrics = {"accuracy": 0.75}
        return TrainResult(
            train_metrics={"accuracy": 0.75},
            val_metrics={"accuracy": 0.70},
            feature_importances=pd.Series(
                np.ones(len(X_train.columns)) / len(X_train.columns),
                index=X_train.columns,
            ),
            training_time_seconds=0.1,
            n_train_samples=len(X_train),
            n_val_samples=len(X_val) if X_val is not None else 0,
            hyperparams={"dummy": True},
        )

    def predict(self, X: pd.DataFrame) -> PredictionResult:
        if not self._is_fitted:
            from alphavedha.exceptions import ModelTrainingError

            raise ModelTrainingError("Model not fitted")
        n = len(X)
        return PredictionResult(
            direction=np.ones(n, dtype=int),
            magnitude=np.full(n, 0.02),
            probabilities=np.full((n, 3), 1 / 3),
            confidence=np.full(n, 0.6),
        )

    def get_feature_importance(self) -> pd.Series | None:
        if not self._is_fitted:
            return None
        return pd.Series(
            np.ones(len(self._feature_names)) / len(self._feature_names),
            index=self._feature_names,
        )

    def _save_model_artifacts(self, directory: Path) -> None:
        (directory / "dummy.txt").write_text("dummy")

    @classmethod
    def _load_model_artifacts(cls, _directory: Path, config: dict[str, Any]) -> DummyModel:
        model = cls(name="dummy", config=config)
        model._is_fitted = True
        return model


class TestBaseModelABC:
    def test_cannot_instantiate_abc(self) -> None:
        with pytest.raises(TypeError):
            BaseModel(name="test", config={})  # type: ignore[abstract]

    def test_dummy_model_properties(self) -> None:
        model = DummyModel(name="dummy", config={"x": 1})
        assert model.name == "dummy"
        assert model.version == "0.0.0"
        assert model.is_fitted is False

    def test_fit_sets_fitted(self) -> None:
        model = DummyModel(name="dummy", config={})
        X = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        y = pd.Series([1, 0, 1])
        model.fit(X, y)
        assert model.is_fitted is True

    def test_predict_before_fit_raises(self) -> None:
        model = DummyModel(name="dummy", config={})
        X = pd.DataFrame({"a": [1, 2]})
        from alphavedha.exceptions import ModelTrainingError

        with pytest.raises(ModelTrainingError):
            model.predict(X)

    def test_get_metrics(self) -> None:
        model = DummyModel(name="dummy", config={})
        X = pd.DataFrame({"a": [1, 2, 3]})
        y = pd.Series([1, 0, 1])
        model.fit(X, y)
        metrics = model.get_metrics()
        assert "accuracy" in metrics

    def test_version_increments_on_save(self, tmp_path: Path) -> None:
        model = DummyModel(name="dummy", config={})
        X = pd.DataFrame({"a": [1, 2, 3]})
        y = pd.Series([1, 0, 1])
        model.fit(X, y)
        artifact = model.save(tmp_path / "v1")
        assert artifact.version == "0.0.1"
        artifact2 = model.save(tmp_path / "v2")
        assert artifact2.version == "0.0.2"

    def test_save_creates_metadata_json(self, tmp_path: Path) -> None:
        model = DummyModel(name="dummy", config={"x": 1})
        X = pd.DataFrame({"a": [1, 2, 3]})
        y = pd.Series([1, 0, 1])
        model.fit(X, y)
        model.save(tmp_path / "out")
        metadata_path = tmp_path / "out" / "metadata.json"
        assert metadata_path.exists()
        metadata = json.loads(metadata_path.read_text())
        assert metadata["name"] == "dummy"
        assert "feature_names" in metadata

    def test_save_creates_feature_importance_csv(self, tmp_path: Path) -> None:
        model = DummyModel(name="dummy", config={})
        X = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        y = pd.Series([1, 0, 1])
        model.fit(X, y)
        model.save(tmp_path / "out")
        fi_path = tmp_path / "out" / "feature_importance.csv"
        assert fi_path.exists()


class TestTrainResult:
    def test_fields(self) -> None:
        tr = TrainResult(
            train_metrics={"acc": 0.8},
            val_metrics={"acc": 0.7},
            feature_importances=None,
            training_time_seconds=1.0,
            n_train_samples=100,
            n_val_samples=20,
            hyperparams={"lr": 0.05},
        )
        assert tr.n_train_samples == 100
        assert tr.hyperparams["lr"] == 0.05


class TestPredictionResult:
    def test_fields(self) -> None:
        pr = PredictionResult(
            direction=np.array([1, -1, 0]),
            magnitude=np.array([0.02, -0.01, 0.0]),
            probabilities=np.ones((3, 3)) / 3,
            confidence=np.array([0.7, 0.6, 0.5]),
        )
        assert len(pr.direction) == 3
        assert pr.probabilities.shape == (3, 3)
