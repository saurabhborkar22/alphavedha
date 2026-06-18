"""HMM Regime Detector — classifies market regimes using Gaussian HMM."""

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
from hmmlearn.hmm import GaussianHMM

from alphavedha.config import RegimeConfig
from alphavedha.exceptions import (
    DataQualityError,
    InsufficientDataError,
    ModelNotFoundError,
    ModelTrainingError,
)

logger = structlog.get_logger(__name__)

_MIN_SAMPLES = 10


@dataclass
class RegimeResult:
    current_regime: str
    regime_id: int
    state_probabilities: np.ndarray
    regime_history: np.ndarray
    transition_matrix: np.ndarray


class RegimeDetector:
    """Detects market regimes (bull/bear/sideways/high-volatility) via Gaussian HMM."""

    def __init__(self, config: RegimeConfig | None = None) -> None:
        self._config = config or RegimeConfig()
        self._hmm: GaussianHMM | None = None
        self._is_fitted = False
        self._state_mapping: dict[str, int] = {}
        self._reverse_mapping: dict[int, str] = {}
        self._training_metrics: dict[str, float] = {}
        self._last_posteriors: np.ndarray | None = None
        # Normalization params (fit on training data, applied at predict time)
        self._feature_mean: np.ndarray | None = None
        self._feature_std: np.ndarray | None = None
        self._n_features: int = 2

    @property
    def state_mapping(self) -> dict[str, int]:
        return dict(self._state_mapping)

    @property
    def hmm_model(self) -> GaussianHMM:
        if self._hmm is None:
            raise ModelTrainingError("RegimeDetector is not fitted.")
        return self._hmm

    def fit(
        self,
        returns: pd.Series,
        volatility: pd.Series,
        extra_features: pd.DataFrame | None = None,
    ) -> dict[str, float]:
        X = self._prepare_input(returns, volatility, extra_features)
        if np.any(~np.isfinite(X)):
            raise DataQualityError("Input contains NaN or Inf values")
        n_samples = X.shape[0]
        if n_samples < _MIN_SAMPLES:
            raise InsufficientDataError(f"Need at least {_MIN_SAMPLES} samples, got {n_samples}")

        self._n_features = X.shape[1]

        # Standardize features for numerical stability with full covariance
        self._feature_mean = X.mean(axis=0)
        self._feature_std = X.std(axis=0)
        # Avoid division by zero for constant features
        self._feature_std = np.where(self._feature_std == 0, 1.0, self._feature_std)
        X_scaled = (X - self._feature_mean) / self._feature_std

        self._hmm = GaussianHMM(
            n_components=self._config.n_states,
            covariance_type=self._config.covariance_type,
            n_iter=self._config.n_iter,
            random_state=42,
        )
        self._hmm.fit(X_scaled)
        self._label_states()
        self._is_fitted = True

        n_params = self._count_params()
        log_likelihood = float(self._hmm.score(X_scaled))
        self._training_metrics = {
            "log_likelihood": log_likelihood,
            "aic": -2.0 * log_likelihood + 2.0 * n_params,
            "bic": -2.0 * log_likelihood + n_params * np.log(n_samples),
        }

        logger.info(
            "regime_detector_fitted",
            n_samples=n_samples,
            n_features=self._n_features,
            metrics=self._training_metrics,
        )
        return dict(self._training_metrics)

    def predict(
        self,
        returns: pd.Series,
        volatility: pd.Series,
        extra_features: pd.DataFrame | None = None,
    ) -> RegimeResult:
        if not self._is_fitted or self._hmm is None:
            raise ModelTrainingError("RegimeDetector is not fitted. Call fit() first.")

        X = self._prepare_input(returns, volatility, extra_features)
        if X.shape[0] == 0:
            raise InsufficientDataError("Cannot predict with empty input")
        if np.any(~np.isfinite(X)):
            raise DataQualityError("Input contains NaN or Inf values")
        if self._feature_mean is not None and self._feature_std is not None:
            X = (X - self._feature_mean) / self._feature_std

        raw_states = self._hmm.predict(X)
        posteriors = self._hmm.predict_proba(X)
        self._last_posteriors = posteriors

        mapped_history = np.array([self._hmm_id_to_semantic_id(s) for s in raw_states])

        last_posteriors_reordered = self._reorder_probabilities(posteriors[-1])
        last_semantic_id = mapped_history[-1]
        current_name = self._config.state_names[last_semantic_id]

        raw_transmat = self._hmm.transmat_
        mapped_transmat = self._reorder_transition_matrix(raw_transmat)

        return RegimeResult(
            current_regime=current_name,
            regime_id=int(last_semantic_id),
            state_probabilities=last_posteriors_reordered,
            regime_history=mapped_history,
            transition_matrix=mapped_transmat,
        )

    def get_regime_features(self) -> pd.DataFrame:
        if self._last_posteriors is None:
            raise ModelTrainingError("No predictions available. Call predict() first.")

        n_states = self._config.n_states
        columns: list[str] = []
        reordered = np.zeros_like(self._last_posteriors)

        for semantic_id in range(n_states):
            name = self._config.state_names[semantic_id]
            columns.append(f"p_{name}")
            hmm_id = self._semantic_id_to_hmm_id(semantic_id)
            reordered[:, semantic_id] = self._last_posteriors[:, hmm_id]

        return pd.DataFrame(reordered, columns=columns)

    def save(self, directory: Path) -> None:
        if not self._is_fitted or self._hmm is None:
            raise ModelTrainingError("Cannot save unfitted RegimeDetector.")

        directory.mkdir(parents=True, exist_ok=True)
        joblib.dump(self._hmm, directory / "hmm_model.joblib")

        metadata: dict[str, Any] = {
            "name": "regime_detector",
            "version": "0.0.1",
            "created_at": datetime.now(UTC).isoformat(),
            "state_mapping": self._state_mapping,
            "config": self._config.model_dump(),
            "metrics": self._training_metrics,
            "feature_mean": self._feature_mean.tolist() if self._feature_mean is not None else None,
            "feature_std": self._feature_std.tolist() if self._feature_std is not None else None,
            "n_features": self._n_features,
        }
        (directory / "metadata.json").write_text(json.dumps(metadata, indent=2))
        logger.info("regime_detector_saved", path=str(directory))

    @classmethod
    def load(cls, directory: Path) -> RegimeDetector:
        metadata_path = directory / "metadata.json"
        if not metadata_path.exists():
            raise ModelNotFoundError(f"No metadata.json at {directory}")

        metadata = json.loads(metadata_path.read_text())

        hmm_path = directory / "hmm_model.joblib"
        if not hmm_path.exists():
            raise ModelNotFoundError(f"No hmm_model.joblib at {directory}")

        config = RegimeConfig.model_validate(metadata["config"])
        detector = cls(config=config)
        detector._hmm = joblib.load(hmm_path)
        detector._state_mapping = {k: int(v) for k, v in metadata["state_mapping"].items()}
        detector._reverse_mapping = {v: k for k, v in detector._state_mapping.items()}
        detector._training_metrics = metadata.get("metrics", {})
        detector._n_features = metadata.get("n_features", 2)
        if metadata.get("feature_mean") is not None:
            detector._feature_mean = np.array(metadata["feature_mean"])
        if metadata.get("feature_std") is not None:
            detector._feature_std = np.array(metadata["feature_std"])
        detector._is_fitted = True

        logger.info("regime_detector_loaded", path=str(directory))
        return detector

    def _prepare_input(
        self,
        returns: pd.Series,
        volatility: pd.Series,
        extra_features: pd.DataFrame | None = None,
    ) -> np.ndarray:
        base = np.column_stack(
            [returns.values.astype(np.float64), volatility.values.astype(np.float64)]
        )
        if extra_features is not None and not extra_features.empty:
            extra = extra_features.values.astype(np.float64)
            if extra.shape[0] == base.shape[0]:
                return np.column_stack([base, extra])
            logger.warning(
                "regime_extra_features_length_mismatch",
                base=base.shape[0],
                extra=extra.shape[0],
            )
        return base

    def _label_states(self) -> None:
        if self._hmm is None:
            return

        means = self._hmm.means_[:, 0]
        n_states = self._config.n_states
        if self._config.covariance_type == "full":
            covariances = np.array([self._hmm.covars_[i][0, 0] for i in range(n_states)])
        elif self._config.covariance_type == "diag":
            covariances = np.array([self._hmm.covars_[i][0] for i in range(n_states)])
        elif self._config.covariance_type == "spherical":
            covariances = self._hmm.covars_
        else:
            covariances = np.array([self._hmm.covars_[i][0, 0] for i in range(n_states)])

        assigned: set[int] = set()
        self._state_mapping = {}

        bull_id = int(np.argmax(means))
        self._state_mapping["bull"] = bull_id
        assigned.add(bull_id)

        remaining_for_bear = [i for i in range(n_states) if i not in assigned]
        bear_id = min(remaining_for_bear, key=lambda i: means[i])
        self._state_mapping["bear"] = bear_id
        assigned.add(bear_id)

        remaining = [i for i in range(n_states) if i not in assigned]
        highvol_id = max(remaining, key=lambda i: covariances[i])
        self._state_mapping["high_volatility"] = highvol_id
        assigned.add(highvol_id)

        sideways_id = next(i for i in range(n_states) if i not in assigned)
        self._state_mapping["sideways"] = sideways_id

        self._reverse_mapping = {v: k for k, v in self._state_mapping.items()}

    def _hmm_id_to_semantic_id(self, hmm_id: int) -> int:
        name = self._reverse_mapping[hmm_id]
        return self._config.state_names.index(name)

    def _semantic_id_to_hmm_id(self, semantic_id: int) -> int:
        name = self._config.state_names[semantic_id]
        return self._state_mapping[name]

    def _reorder_probabilities(self, raw_probs: np.ndarray) -> np.ndarray:
        n_states = self._config.n_states
        reordered = np.zeros(n_states)
        for semantic_id in range(n_states):
            hmm_id = self._semantic_id_to_hmm_id(semantic_id)
            reordered[semantic_id] = raw_probs[hmm_id]
        return reordered

    def _reorder_transition_matrix(self, raw_transmat: np.ndarray) -> np.ndarray:
        n = self._config.n_states
        reordered = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                hmm_i = self._semantic_id_to_hmm_id(i)
                hmm_j = self._semantic_id_to_hmm_id(j)
                reordered[i, j] = raw_transmat[hmm_i, hmm_j]
        return reordered

    def _count_params(self) -> int:
        n = self._config.n_states
        n_features = self._n_features
        start_probs = n - 1
        transition = n * (n - 1)
        means = n * n_features
        if self._config.covariance_type == "full":
            covars = n * n_features * (n_features + 1) // 2
        elif self._config.covariance_type == "diag":
            covars = n * n_features
        elif self._config.covariance_type == "spherical":
            covars = n
        else:
            covars = n * n_features
        return start_probs + transition + means + covars
