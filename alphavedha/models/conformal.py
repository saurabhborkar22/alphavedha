"""Conformal Predictor — MAPIE-based prediction intervals with coverage guarantees."""

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
from mapie.regression import CrossConformalRegressor, SplitConformalRegressor
from sklearn.base import RegressorMixin
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor

from alphavedha.config import ConformalConfig
from alphavedha.exceptions import ModelNotFoundError, ModelTrainingError

logger = structlog.get_logger(__name__)


@dataclass
class ConformalResult:
    price_low: np.ndarray
    price_mid: np.ndarray
    price_high: np.ndarray
    interval_width: np.ndarray
    coverage: float


class ConformalPredictor:
    """Wraps any sklearn-compatible regressor with MAPIE for prediction intervals.

    Uses CrossConformalRegressor (jackknife+) for training and
    SplitConformalRegressor (prefit) for post-hoc calibration.
    """

    def __init__(
        self,
        config: ConformalConfig | None = None,
        base_regressor: Any | None = None,
    ) -> None:
        self._config = config or ConformalConfig()
        self._base_regressor: RegressorMixin = base_regressor or XGBRegressor(
            n_estimators=100, random_state=42
        )
        self._mapie: Any = None
        self._is_fitted = False
        self._training_metrics: dict[str, float] = {}

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> dict[str, float]:
        """Fit the predictor using cross-conformal (jackknife+) approach."""
        mapie = CrossConformalRegressor(
            estimator=self._base_regressor,
            confidence_level=self._config.coverage,
            method=self._config.method,
            random_state=42,
        )
        mapie.fit_conformalize(X_train.values, y_train.values)
        self._mapie = mapie
        self._is_fitted = True

        y_pred = mapie.predict(X_train.values)
        self._training_metrics = {
            "r2": float(r2_score(y_train.values, y_pred)),
            "rmse": float(np.sqrt(mean_squared_error(y_train.values, y_pred))),
        }

        logger.info(
            "conformal_predictor_fitted",
            n_samples=len(X_train),
            metrics=self._training_metrics,
        )
        return dict(self._training_metrics)

    def predict(self, X: pd.DataFrame) -> ConformalResult:
        """Predict point estimates and prediction intervals."""
        if not self._is_fitted or self._mapie is None:
            raise ModelTrainingError("ConformalPredictor is not fitted. Call fit() first.")

        y_pred, y_pis = self._mapie.predict_interval(X.values)

        # y_pis shape: (n_samples, 2, n_confidence_levels)
        price_low = y_pis[:, 0, 0]
        price_high = y_pis[:, 1, 0]

        return ConformalResult(
            price_low=price_low,
            price_mid=y_pred,
            price_high=price_high,
            interval_width=price_high - price_low,
            coverage=self._config.coverage,
        )

    def calibrate(self, X_cal: pd.DataFrame, y_cal: pd.Series) -> None:
        """Re-calibrate intervals using a held-out calibration set.

        Extracts the already-fitted base estimator and wraps it in a
        SplitConformalRegressor (prefit=True) for new conformity score estimation.
        """
        if not self._is_fitted or self._mapie is None:
            raise ModelTrainingError("ConformalPredictor is not fitted. Call fit() first.")

        # Extract the underlying fitted base estimator from the cross-conformal model
        fitted_base: RegressorMixin = self._mapie._mapie_regressor.estimator_.single_estimator_

        split_mapie = SplitConformalRegressor(
            estimator=fitted_base,
            confidence_level=self._config.coverage,
            prefit=True,
        )
        split_mapie.conformalize(X_cal.values, y_cal.values)
        self._mapie = split_mapie

        logger.info("conformal_predictor_calibrated", n_cal_samples=len(X_cal))

    def save(self, directory: Path) -> None:
        """Serialize the fitted predictor to disk."""
        if not self._is_fitted or self._mapie is None:
            raise ModelTrainingError("Cannot save unfitted ConformalPredictor.")

        directory.mkdir(parents=True, exist_ok=True)
        joblib.dump(self._mapie, directory / "mapie_model.joblib")

        metadata: dict[str, Any] = {
            "name": "conformal_predictor",
            "created_at": datetime.now(UTC).isoformat(),
            "config": self._config.model_dump(),
            "metrics": self._training_metrics,
        }
        (directory / "metadata.json").write_text(json.dumps(metadata, indent=2))
        logger.info("conformal_predictor_saved", path=str(directory))

    @classmethod
    def load(cls, directory: Path) -> ConformalPredictor:
        """Load a previously saved ConformalPredictor from disk."""
        metadata_path = directory / "metadata.json"
        if not metadata_path.exists():
            raise ModelNotFoundError(f"No metadata.json at {directory}")

        metadata = json.loads(metadata_path.read_text())

        mapie_path = directory / "mapie_model.joblib"
        if not mapie_path.exists():
            raise ModelNotFoundError(f"No mapie_model.joblib at {directory}")

        config = ConformalConfig.model_validate(metadata["config"])
        predictor = cls(config=config)
        predictor._mapie = joblib.load(mapie_path)
        predictor._training_metrics = metadata.get("metrics", {})
        predictor._is_fitted = True

        logger.info("conformal_predictor_loaded", path=str(directory))
        return predictor
