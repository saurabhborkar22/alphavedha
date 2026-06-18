"""MetaLabelingModel — binary gate that predicts P(ensemble prediction is correct)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import structlog
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from xgboost import XGBClassifier

from alphavedha.config import MetaLabelingConfig
from alphavedha.exceptions import (
    DataQualityError,
    InsufficientDataError,
    ModelNotFoundError,
    ModelTrainingError,
    PredictionError,
)

logger = structlog.get_logger(__name__)

_MIN_SAMPLES = 10


@dataclass
class MetaLabelResult:
    meta_confidence: np.ndarray
    is_tradeable: np.ndarray


class MetaLabelingModel:
    """Predicts P(ensemble prediction is correct) to gate low-confidence signals."""

    def __init__(self, config: MetaLabelingConfig | None = None) -> None:
        self._config = config or MetaLabelingConfig()
        self._classifier: XGBClassifier | None = None
        self._is_fitted = False
        self._training_metrics: dict[str, float] = {}
        self._feature_names: list[str] = []

    def fit(
        self,
        X_features: pd.DataFrame,
        ensemble_direction: np.ndarray,
        ensemble_confidence: np.ndarray,
        y_correct: pd.Series,
        X_val: pd.DataFrame | None = None,
        ensemble_direction_val: np.ndarray | None = None,
        ensemble_confidence_val: np.ndarray | None = None,
        y_correct_val: pd.Series | None = None,
    ) -> dict[str, float]:
        X_aug = self._build_features(X_features, ensemble_direction, ensemble_confidence)

        n_samples = len(X_aug)
        if n_samples < _MIN_SAMPLES:
            raise InsufficientDataError(f"Need at least {_MIN_SAMPLES} samples, got {n_samples}")

        self._validate_inputs(X_aug)

        self._feature_names = list(X_aug.columns)

        self._classifier = XGBClassifier(
            objective="binary:logistic",
            learning_rate=0.05,
            max_depth=4,
            n_estimators=200,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric="logloss",
            verbosity=0,
            n_jobs=-1,
        )

        eval_set: list[tuple[np.ndarray, np.ndarray]] = []
        if (
            X_val is not None
            and ensemble_direction_val is not None
            and ensemble_confidence_val is not None
            and y_correct_val is not None
        ):
            X_val_aug = self._build_features(X_val, ensemble_direction_val, ensemble_confidence_val)
            eval_set = [(X_val_aug.values, y_correct_val.values)]
            self._classifier.set_params(early_stopping_rounds=20)

        self._classifier.fit(
            X_aug.values,
            y_correct.values,
            eval_set=eval_set or None,
            verbose=False,
        )
        self._is_fitted = True

        y_pred = self._classifier.predict(X_aug.values)
        self._training_metrics = {
            "train_accuracy": float(accuracy_score(y_correct.values, y_pred)),
            "train_precision": float(precision_score(y_correct.values, y_pred, zero_division=0)),
            "train_recall": float(recall_score(y_correct.values, y_pred, zero_division=0)),
            "train_f1": float(f1_score(y_correct.values, y_pred, zero_division=0)),
        }

        logger.info(
            "meta_labeling_fitted",
            n_samples=n_samples,
            metrics=self._training_metrics,
        )
        return dict(self._training_metrics)

    def predict(
        self,
        X_features: pd.DataFrame,
        ensemble_direction: np.ndarray,
        ensemble_confidence: np.ndarray,
    ) -> MetaLabelResult:
        if not self._is_fitted or self._classifier is None:
            raise ModelTrainingError("MetaLabelingModel is not fitted. Call fit() first.")

        X_aug = self._build_features(X_features, ensemble_direction, ensemble_confidence)
        X_aug = self._align_features(X_aug)
        # Training data was NaN/Inf-filled with 0 (_fill_nan_for_torch) — match it
        X_aug = X_aug.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        self._validate_inputs(X_aug)

        if len(X_aug) == 0:
            raise InsufficientDataError("Cannot predict with empty input")

        proba = self._classifier.predict_proba(X_aug.values)
        meta_confidence = proba[:, 1]

        return MetaLabelResult(
            meta_confidence=meta_confidence,
            is_tradeable=meta_confidence > self._config.min_confidence,
        )

    def save(self, directory: Path) -> None:
        if not self._is_fitted or self._classifier is None:
            raise ModelTrainingError("Cannot save unfitted MetaLabelingModel.")

        directory.mkdir(parents=True, exist_ok=True)
        joblib.dump(self._classifier, directory / "meta_classifier.joblib")

        metadata: dict[str, Any] = {
            "name": "meta_labeling_model",
            "version": "0.0.1",
            "created_at": datetime.now(UTC).isoformat(),
            "config": self._config.model_dump(),
            "metrics": self._training_metrics,
            "feature_names": self._feature_names,
        }
        (directory / "metadata.json").write_text(json.dumps(metadata, indent=2))
        logger.info("meta_labeling_saved", path=str(directory))

    @classmethod
    def load(cls, directory: Path) -> MetaLabelingModel:
        metadata_path = directory / "metadata.json"
        if not metadata_path.exists():
            raise ModelNotFoundError(f"No metadata.json at {directory}")

        metadata = json.loads(metadata_path.read_text())

        classifier_path = directory / "meta_classifier.joblib"
        if not classifier_path.exists():
            raise ModelNotFoundError(f"No meta_classifier.joblib at {directory}")

        config = MetaLabelingConfig.model_validate(metadata["config"])
        model = cls(config=config)
        model._classifier = joblib.load(classifier_path)
        model._training_metrics = metadata.get("metrics", {})
        model._feature_names = metadata.get("feature_names", [])
        model._is_fitted = True

        logger.info("meta_labeling_loaded", path=str(directory))
        return model

    def _align_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Subset and reorder X to the features the model was trained on."""
        if not self._feature_names or list(X.columns) == self._feature_names:
            return X
        missing = [f for f in self._feature_names if f not in X.columns]
        if missing:
            raise PredictionError(
                f"meta_labeling: input is missing {len(missing)} trained "
                f"feature(s), e.g. {missing[:5]}"
            )
        return X[self._feature_names]

    @staticmethod
    def _build_features(
        X_features: pd.DataFrame,
        ensemble_direction: np.ndarray,
        ensemble_confidence: np.ndarray,
    ) -> pd.DataFrame:
        X_aug = X_features.copy()
        X_aug["ensemble_direction"] = ensemble_direction
        X_aug["ensemble_confidence"] = ensemble_confidence
        return X_aug

    def validate_monotonicity(
        self,
        X_features: pd.DataFrame,
        ensemble_direction: np.ndarray,
        ensemble_confidence: np.ndarray,
        y_correct: np.ndarray,
        n_bins: int = 5,
    ) -> dict[str, Any]:
        """Check that higher meta-confidence correlates with higher win rate.

        Bins OOS predictions by meta-confidence and returns per-bin win rates.
        A well-calibrated meta-model should show monotonically increasing
        win rates. Anti-predictive (inverted) or flat win rates indicate the
        model should be bypassed.

        Returns dict with bins, is_monotonic flag, and rank correlation.
        """
        result = self.predict(X_features, ensemble_direction, ensemble_confidence)
        confidences = result.meta_confidence
        n = len(confidences)

        bin_edges = np.quantile(confidences, np.linspace(0.0, 1.0, n_bins + 1))
        bin_edges[0] = confidences.min() - 1e-9
        bin_edges[-1] = confidences.max() + 1e-9
        bins: list[dict[str, Any]] = []
        win_rates: list[float] = []

        for i in range(n_bins):
            lo, hi = bin_edges[i], bin_edges[i + 1]
            mask = (confidences >= lo) & (confidences < hi)
            count = int(mask.sum())
            wr = float(y_correct[mask].mean()) if count > 0 else float("nan")
            bins.append(
                {
                    "bin": f"{lo:.4f}-{hi:.4f}",
                    "count": count,
                    "win_rate": wr,
                }
            )
            if count > 0:
                win_rates.append(wr)

        is_monotonic = (
            all(win_rates[i] <= win_rates[i + 1] for i in range(len(win_rates) - 1))
            if len(win_rates) >= 2
            else False
        )

        from scipy.stats import spearmanr

        if len(win_rates) >= 3:
            rho, p_value = spearmanr(range(len(win_rates)), win_rates)
        else:
            rho, p_value = float("nan"), float("nan")

        return {
            "n_samples": n,
            "n_bins": n_bins,
            "bins": bins,
            "is_monotonic": is_monotonic,
            "spearman_rho": float(rho),
            "spearman_p": float(p_value),
            "overall_win_rate": float(y_correct.mean()),
        }

    @staticmethod
    def _validate_inputs(X: pd.DataFrame) -> None:
        if np.any(~np.isfinite(X.values)):
            raise DataQualityError("Input contains NaN or Inf values")
