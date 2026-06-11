"""BaseModel ABC — lifecycle contract for all ML models."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import structlog

from alphavedha.exceptions import ModelNotFoundError, PredictionError

logger = structlog.get_logger(__name__)


@dataclass
class TrainResult:
    train_metrics: dict[str, float]
    val_metrics: dict[str, float]
    feature_importances: pd.Series | None
    training_time_seconds: float
    n_train_samples: int
    n_val_samples: int
    hyperparams: dict[str, Any]


@dataclass
class PredictionResult:
    direction: np.ndarray
    magnitude: np.ndarray
    probabilities: np.ndarray | None
    confidence: np.ndarray


@dataclass
class ModelArtifact:
    path: Path
    name: str
    version: str
    created_at: str
    feature_names: list[str]
    metrics: dict[str, float]
    config: dict[str, Any]


class BaseModel(ABC):
    def __init__(self, name: str, config: dict[str, Any]) -> None:
        self._name = name
        self._config = config
        self._version_counter = 0
        self._is_fitted = False
        self._train_metrics: dict[str, float] = {}
        self._feature_names: list[str] = []

    @property
    def name(self) -> str:
        return self._name

    @property
    def version(self) -> str:
        return f"0.0.{self._version_counter}"

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    def _align_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Subset and reorder X to the features the model was trained on.

        Models trained on a feature subset (e.g. LSTM/TFT on top-30 by
        importance) receive the full feature matrix at prediction time.
        """
        if not self._feature_names or list(X.columns) == self._feature_names:
            return X
        missing = [f for f in self._feature_names if f not in X.columns]
        if missing:
            raise PredictionError(
                f"{self._name}: input is missing {len(missing)} trained "
                f"feature(s), e.g. {missing[:5]}"
            )
        return X[self._feature_names]

    @abstractmethod
    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame | None = None,
        y_val: pd.Series | None = None,
        sample_weight: pd.Series | None = None,
    ) -> TrainResult: ...

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> PredictionResult: ...

    @abstractmethod
    def get_feature_importance(self) -> pd.Series | None: ...

    @abstractmethod
    def _save_model_artifacts(self, directory: Path) -> None: ...

    @classmethod
    @abstractmethod
    def _load_model_artifacts(cls, directory: Path, config: dict[str, Any]) -> BaseModel: ...

    def get_metrics(self) -> dict[str, float]:
        return dict(self._train_metrics)

    def save(self, directory: Path) -> ModelArtifact:
        self._version_counter += 1
        directory.mkdir(parents=True, exist_ok=True)

        self._save_model_artifacts(directory)

        fi = self.get_feature_importance()
        if fi is not None:
            fi.to_csv(directory / "feature_importance.csv")

        artifact = ModelArtifact(
            path=directory,
            name=self._name,
            version=self.version,
            created_at=datetime.now(UTC).isoformat(),
            feature_names=list(self._feature_names),
            metrics=dict(self._train_metrics),
            config=dict(self._config),
        )

        metadata = {
            "name": artifact.name,
            "version": artifact.version,
            "created_at": artifact.created_at,
            "feature_names": artifact.feature_names,
            "metrics": artifact.metrics,
            "config": artifact.config,
        }
        (directory / "metadata.json").write_text(json.dumps(metadata, indent=2))

        logger.info(
            "model_saved",
            name=self._name,
            version=self.version,
            path=str(directory),
        )

        return artifact

    @classmethod
    def load(cls, directory: Path, config: dict[str, Any] | None = None) -> BaseModel:
        metadata_path = directory / "metadata.json"
        if not metadata_path.exists():
            raise ModelNotFoundError(f"No metadata.json at {directory}")

        metadata = json.loads(metadata_path.read_text())
        load_config = config if config is not None else metadata.get("config", {})

        model = cls._load_model_artifacts(directory, load_config)
        model._feature_names = metadata.get("feature_names", [])
        model._train_metrics = metadata.get("metrics", {})
        model._is_fitted = True

        logger.info(
            "model_loaded",
            name=metadata["name"],
            version=metadata["version"],
            path=str(directory),
        )

        return model
