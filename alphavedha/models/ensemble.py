"""StackingEnsemble — Ridge-based meta-learner combining base model OOF outputs."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, ClassVar

import joblib
import numpy as np
import pandas as pd
import structlog
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import accuracy_score, f1_score

from alphavedha.config import EnsembleConfig
from alphavedha.exceptions import (
    DataQualityError,
    InsufficientDataError,
    ModelNotFoundError,
    ModelTrainingError,
)
from alphavedha.models.base import PredictionResult

logger = structlog.get_logger(__name__)

_MIN_SAMPLES = 10
_LABEL_MAP = {-1: 0, 0: 1, 1: 2}
_LABEL_REVERSE = {0: -1, 1: 0, 2: 1}


@dataclass
class EnsembleResult:
    direction: np.ndarray
    magnitude: np.ndarray
    probabilities: np.ndarray
    confidence: np.ndarray
    model_disagreement: np.ndarray


class StackingEnsemble:
    """Combines base model OOF predictions + regime probs via RidgeClassifier."""

    EXPECTED_MODELS: ClassVar[list[str]] = ["xgboost", "lstm", "tft"]

    def __init__(self, config: EnsembleConfig | None = None) -> None:
        self._config = config or EnsembleConfig()
        self._ridge: RidgeClassifier | None = None
        self._is_fitted = False
        self._training_metrics: dict[str, float] = {}

    def fit(
        self,
        base_oof_predictions: dict[str, PredictionResult],
        regime_probs: np.ndarray,
        y_true: pd.Series,
    ) -> dict[str, float]:
        self._validate_model_names(base_oof_predictions)
        meta_X = self._build_meta_features(base_oof_predictions, regime_probs)
        self._validate_inputs(meta_X)

        n_samples = meta_X.shape[0]
        if n_samples < _MIN_SAMPLES:
            raise InsufficientDataError(
                f"Need at least {_MIN_SAMPLES} OOF samples, got {n_samples}"
            )

        y_cls = np.array([_LABEL_MAP[v] for v in y_true.values])

        self._ridge = RidgeClassifier(alpha=self._config.alpha)
        self._ridge.fit(meta_X, y_cls)
        self._is_fitted = True

        y_pred = self._ridge.predict(meta_X)
        self._training_metrics = {
            "accuracy": float(accuracy_score(y_cls, y_pred)),
            "f1_weighted": float(f1_score(y_cls, y_pred, average="weighted")),
        }

        logger.info(
            "ensemble_fitted",
            n_samples=n_samples,
            metrics=self._training_metrics,
        )
        return dict(self._training_metrics)

    def predict(
        self,
        base_predictions: dict[str, PredictionResult],
        regime_probs: np.ndarray,
    ) -> EnsembleResult:
        if not self._is_fitted or self._ridge is None:
            raise ModelTrainingError("StackingEnsemble is not fitted. Call fit() first.")

        self._validate_model_names(base_predictions)
        meta_X = self._build_meta_features(base_predictions, regime_probs)
        self._validate_inputs(meta_X)

        if meta_X.shape[0] == 0:
            raise InsufficientDataError("Cannot predict with empty input")

        cls_pred = self._ridge.predict(meta_X)
        direction = np.array([_LABEL_REVERSE[c] for c in cls_pred])

        decision = self._ridge.decision_function(meta_X)
        if decision.ndim == 1:
            proba = np.column_stack([1 - decision, decision])
            proba = np.clip(proba, 0, 1)
            proba = proba / proba.sum(axis=1, keepdims=True)
        else:
            proba = self._softmax(decision)

        confidence = np.max(proba, axis=1)
        disagreement = self._compute_disagreement(base_predictions)
        magnitude = self._aggregate_magnitude(base_predictions)

        return EnsembleResult(
            direction=direction,
            magnitude=magnitude,
            probabilities=proba,
            confidence=confidence,
            model_disagreement=disagreement,
        )

    def save(self, directory: Path) -> None:
        if not self._is_fitted or self._ridge is None:
            raise ModelTrainingError("Cannot save unfitted StackingEnsemble.")

        directory.mkdir(parents=True, exist_ok=True)
        joblib.dump(self._ridge, directory / "ridge_model.joblib")

        metadata: dict[str, Any] = {
            "name": "stacking_ensemble",
            "created_at": datetime.now(UTC).isoformat(),
            "config": self._config.model_dump(),
            "metrics": self._training_metrics,
            "expected_models": self.EXPECTED_MODELS,
        }
        (directory / "metadata.json").write_text(json.dumps(metadata, indent=2))
        logger.info("ensemble_saved", path=str(directory))

    @classmethod
    def load(cls, directory: Path) -> StackingEnsemble:
        metadata_path = directory / "metadata.json"
        if not metadata_path.exists():
            raise ModelNotFoundError(f"No metadata.json at {directory}")

        metadata = json.loads(metadata_path.read_text())

        ridge_path = directory / "ridge_model.joblib"
        if not ridge_path.exists():
            raise ModelNotFoundError(f"No ridge_model.joblib at {directory}")

        config = EnsembleConfig.model_validate(metadata["config"])
        ensemble = cls(config=config)
        ensemble._ridge = joblib.load(ridge_path)
        ensemble._training_metrics = metadata.get("metrics", {})
        ensemble._is_fitted = True

        logger.info("ensemble_loaded", path=str(directory))
        return ensemble

    def _build_meta_features(
        self,
        base_predictions: dict[str, PredictionResult],
        regime_probs: np.ndarray,
    ) -> np.ndarray:
        model_probs: list[np.ndarray] = [
            p
            for name in self.EXPECTED_MODELS
            if (p := base_predictions[name].probabilities) is not None
        ]
        disagreement = self._compute_disagreement(base_predictions)
        return np.column_stack([*model_probs, regime_probs, disagreement])

    def _compute_disagreement(self, base_predictions: dict[str, PredictionResult]) -> np.ndarray:
        probs_list: list[np.ndarray] = [
            p for m in self.EXPECTED_MODELS if (p := base_predictions[m].probabilities) is not None
        ]
        stacked = np.stack(probs_list, axis=0)
        mean_probs = stacked.mean(axis=0)
        consensus = np.argmax(mean_probs, axis=1)
        n = len(consensus)
        probs_for_consensus = stacked[:, np.arange(n), consensus]
        return np.std(probs_for_consensus, axis=0)  # type: ignore[no-any-return]

    def _aggregate_magnitude(self, base_predictions: dict[str, PredictionResult]) -> np.ndarray:
        confs = np.stack(
            [base_predictions[m].confidence for m in self.EXPECTED_MODELS],
            axis=0,
        )
        mags = np.stack(
            [base_predictions[m].magnitude for m in self.EXPECTED_MODELS],
            axis=0,
        )
        conf_sum = confs.sum(axis=0, keepdims=True)
        conf_sum = np.where(conf_sum == 0, 1.0, conf_sum)
        weights = confs / conf_sum
        return (weights * mags).sum(axis=0)  # type: ignore[no-any-return]

    def _validate_model_names(self, base_predictions: dict[str, PredictionResult]) -> None:
        expected = set(self.EXPECTED_MODELS)
        actual = set(base_predictions.keys())
        if actual != expected:
            raise ValueError(f"Expected models {sorted(expected)}, got {sorted(actual)}")

    def _validate_inputs(self, meta_X: np.ndarray) -> None:
        if np.any(~np.isfinite(meta_X)):
            raise DataQualityError("Input contains NaN or Inf values")

    @staticmethod
    def _softmax(logits: np.ndarray) -> np.ndarray:
        exp = np.exp(logits - logits.max(axis=1, keepdims=True))
        return exp / exp.sum(axis=1, keepdims=True)  # type: ignore[no-any-return]
