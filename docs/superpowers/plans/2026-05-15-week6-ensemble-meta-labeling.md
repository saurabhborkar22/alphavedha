# Week 6: Ensemble Stacking + Meta-Labeling Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the stacking ensemble meta-learner (Ridge) that combines XGBoost/LSTM/TFT OOF outputs + regime probabilities into a final prediction, plus a meta-labeling model (XGBoost binary classifier) that gates low-confidence signals.

**Architecture:** Two new files in `alphavedha/models/` — `ensemble.py` (StackingEnsemble with RidgeClassifier) and `meta_model.py` (MetaLabelingModel with XGBClassifier). Neither subclasses BaseModel; both follow the save/load pattern of RegimeDetector and ConformalPredictor (joblib + metadata.json). TDD with ~24 tests total.

**Tech Stack:** scikit-learn RidgeClassifier, XGBoost XGBClassifier, joblib, numpy, pandas, structlog

---

## File Structure

| File | Action | Responsibility |
|------|--------|----------------|
| `tests/unit/models/test_ensemble.py` | Create | 14 tests for StackingEnsemble |
| `alphavedha/models/ensemble.py` | Create | StackingEnsemble class |
| `tests/unit/models/test_meta_model.py` | Create | 10 tests for MetaLabelingModel |
| `alphavedha/models/meta_model.py` | Create | MetaLabelingModel class |
| `alphavedha/models/__init__.py` | Modify | Export new classes |
| `alphavedha/models/CLAUDE.md` | Modify | Document implementation details |

---

### Task 1: Write Failing Tests for StackingEnsemble

**Files:**
- Create: `tests/unit/models/test_ensemble.py`

- [ ] **Step 1: Write all 14 failing tests**

```python
"""Tests for StackingEnsemble — stacking meta-learner combining base model outputs."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from alphavedha.config import EnsembleConfig
from alphavedha.exceptions import DataQualityError, InsufficientDataError, ModelTrainingError
from alphavedha.models.base import PredictionResult
from alphavedha.models.ensemble import EnsembleResult, StackingEnsemble

_LABEL_REVERSE = {0: -1, 1: 0, 2: 1}


def _make_prediction_result(
    n: int, rng: np.random.Generator, bias_class: int = 2
) -> PredictionResult:
    raw = rng.random((n, 3))
    raw[:, bias_class] += 1.0
    probabilities = raw / raw.sum(axis=1, keepdims=True)
    direction_idx = np.argmax(probabilities, axis=1)
    direction = np.array([_LABEL_REVERSE[d] for d in direction_idx])
    confidence = np.max(probabilities, axis=1)
    magnitude = rng.normal(0.01, 0.005, size=n)
    return PredictionResult(
        direction=direction,
        magnitude=magnitude,
        probabilities=probabilities,
        confidence=confidence,
    )


@pytest.fixture
def ensemble_config() -> EnsembleConfig:
    return EnsembleConfig(meta_learner="ridge", alpha=1.0)


@pytest.fixture
def synthetic_ensemble_data() -> (
    tuple[dict[str, PredictionResult], np.ndarray, pd.Series]
):
    rng = np.random.default_rng(42)
    n = 100
    base_preds = {
        "xgboost": _make_prediction_result(n, rng, bias_class=2),
        "lstm": _make_prediction_result(n, rng, bias_class=2),
        "tft": _make_prediction_result(n, rng, bias_class=1),
    }
    regime_raw = rng.random((n, 4))
    regime_probs = regime_raw / regime_raw.sum(axis=1, keepdims=True)
    y_true = pd.Series(rng.choice([-1, 0, 1], size=n), name="direction")
    return base_preds, regime_probs, y_true


class TestStackingEnsembleFit:
    def test_fit_returns_metrics(
        self,
        synthetic_ensemble_data: tuple[
            dict[str, PredictionResult], np.ndarray, pd.Series
        ],
        ensemble_config: EnsembleConfig,
    ) -> None:
        base_preds, regime_probs, y_true = synthetic_ensemble_data
        ensemble = StackingEnsemble(config=ensemble_config)
        metrics = ensemble.fit(base_preds, regime_probs, y_true)
        assert isinstance(metrics, dict)
        assert "accuracy" in metrics
        assert "f1_weighted" in metrics

    def test_predict_before_fit_raises(
        self, ensemble_config: EnsembleConfig
    ) -> None:
        ensemble = StackingEnsemble(config=ensemble_config)
        rng = np.random.default_rng(0)
        preds = {
            name: _make_prediction_result(10, rng)
            for name in ["xgboost", "lstm", "tft"]
        }
        regime = np.ones((10, 4)) / 4
        with pytest.raises(ModelTrainingError):
            ensemble.predict(preds, regime)

    def test_nan_input_raises(
        self, ensemble_config: EnsembleConfig
    ) -> None:
        rng = np.random.default_rng(0)
        n = 50
        preds = {
            name: _make_prediction_result(n, rng)
            for name in ["xgboost", "lstm", "tft"]
        }
        regime = np.ones((n, 4)) / 4
        preds["xgboost"].probabilities[0, 0] = np.nan
        y = pd.Series(rng.choice([-1, 0, 1], size=n))
        ensemble = StackingEnsemble(config=ensemble_config)
        with pytest.raises(DataQualityError):
            ensemble.fit(preds, regime, y)

    def test_inf_input_raises(
        self, ensemble_config: EnsembleConfig
    ) -> None:
        rng = np.random.default_rng(0)
        n = 50
        preds = {
            name: _make_prediction_result(n, rng)
            for name in ["xgboost", "lstm", "tft"]
        }
        regime = np.ones((n, 4)) / 4
        regime[0, 0] = np.inf
        y = pd.Series(rng.choice([-1, 0, 1], size=n))
        ensemble = StackingEnsemble(config=ensemble_config)
        with pytest.raises(DataQualityError):
            ensemble.fit(preds, regime, y)

    def test_missing_model_name_raises(
        self, ensemble_config: EnsembleConfig
    ) -> None:
        rng = np.random.default_rng(0)
        n = 50
        preds = {
            name: _make_prediction_result(n, rng)
            for name in ["xgboost", "lstm"]
        }
        regime = np.ones((n, 4)) / 4
        y = pd.Series(rng.choice([-1, 0, 1], size=n))
        ensemble = StackingEnsemble(config=ensemble_config)
        with pytest.raises(ValueError, match="Expected models"):
            ensemble.fit(preds, regime, y)

    def test_extra_model_name_raises(
        self, ensemble_config: EnsembleConfig
    ) -> None:
        rng = np.random.default_rng(0)
        n = 50
        preds = {
            name: _make_prediction_result(n, rng)
            for name in ["xgboost", "lstm", "tft", "extra"]
        }
        regime = np.ones((n, 4)) / 4
        y = pd.Series(rng.choice([-1, 0, 1], size=n))
        ensemble = StackingEnsemble(config=ensemble_config)
        with pytest.raises(ValueError, match="Expected models"):
            ensemble.fit(preds, regime, y)

    def test_empty_predictions_raises(
        self, ensemble_config: EnsembleConfig
    ) -> None:
        rng = np.random.default_rng(0)
        preds = {
            name: _make_prediction_result(0, rng)
            for name in ["xgboost", "lstm", "tft"]
        }
        regime = np.ones((0, 4))
        y = pd.Series([], dtype=int)
        ensemble = StackingEnsemble(config=ensemble_config)
        with pytest.raises(InsufficientDataError):
            ensemble.fit(preds, regime, y)


class TestStackingEnsemblePredict:
    @pytest.fixture(autouse=True)
    def _fitted_ensemble(
        self,
        synthetic_ensemble_data: tuple[
            dict[str, PredictionResult], np.ndarray, pd.Series
        ],
        ensemble_config: EnsembleConfig,
    ) -> None:
        base_preds, regime_probs, y_true = synthetic_ensemble_data
        self.ensemble = StackingEnsemble(config=ensemble_config)
        self.ensemble.fit(base_preds, regime_probs, y_true)
        self.base_preds = base_preds
        self.regime_probs = regime_probs
        self.result = self.ensemble.predict(base_preds, regime_probs)

    def test_predict_returns_ensemble_result(self) -> None:
        assert isinstance(self.result, EnsembleResult)

    def test_direction_values_valid(self) -> None:
        unique = set(np.unique(self.result.direction))
        assert unique.issubset({-1, 0, 1})

    def test_probabilities_sum_to_one(self) -> None:
        row_sums = self.result.probabilities.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-6)

    def test_confidence_range(self) -> None:
        assert np.all(self.result.confidence >= 0.0)
        assert np.all(self.result.confidence <= 1.0)

    def test_model_disagreement_is_std(self) -> None:
        stacked = np.stack(
            [self.base_preds[m].probabilities for m in ["xgboost", "lstm", "tft"]],
            axis=0,
        )
        mean_probs = stacked.mean(axis=0)
        consensus = np.argmax(mean_probs, axis=1)
        n = len(consensus)
        expected = np.std(
            stacked[:, np.arange(n), consensus], axis=0
        )
        np.testing.assert_allclose(
            self.result.model_disagreement, expected, atol=1e-10
        )

    def test_magnitude_is_weighted_average(self) -> None:
        confs = np.stack(
            [self.base_preds[m].confidence for m in ["xgboost", "lstm", "tft"]],
            axis=0,
        )
        mags = np.stack(
            [self.base_preds[m].magnitude for m in ["xgboost", "lstm", "tft"]],
            axis=0,
        )
        conf_sum = confs.sum(axis=0, keepdims=True)
        conf_sum = np.where(conf_sum == 0, 1.0, conf_sum)
        weights = confs / conf_sum
        expected = (weights * mags).sum(axis=0)
        np.testing.assert_allclose(
            self.result.magnitude, expected, atol=1e-10
        )


class TestStackingEnsemblePersistence:
    def test_save_load_roundtrip(
        self,
        synthetic_ensemble_data: tuple[
            dict[str, PredictionResult], np.ndarray, pd.Series
        ],
        ensemble_config: EnsembleConfig,
        tmp_path: Path,
    ) -> None:
        base_preds, regime_probs, y_true = synthetic_ensemble_data
        ensemble = StackingEnsemble(config=ensemble_config)
        ensemble.fit(base_preds, regime_probs, y_true)
        result_before = ensemble.predict(base_preds, regime_probs)

        save_dir = tmp_path / "ensemble_test"
        ensemble.save(save_dir)

        loaded = StackingEnsemble.load(save_dir)
        result_after = loaded.predict(base_preds, regime_probs)

        np.testing.assert_array_equal(
            result_before.direction, result_after.direction
        )
        np.testing.assert_allclose(
            result_before.probabilities, result_after.probabilities, atol=1e-10
        )
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/unit/models/test_ensemble.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'alphavedha.models.ensemble'`

- [ ] **Step 3: Commit**

```bash
git add tests/unit/models/test_ensemble.py
git commit -m "test: add failing tests for StackingEnsemble (14 tests)"
```

---

### Task 2: Implement StackingEnsemble

**Files:**
- Create: `alphavedha/models/ensemble.py`

- [ ] **Step 1: Implement the full StackingEnsemble class**

```python
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
            raise ModelTrainingError(
                "StackingEnsemble is not fitted. Call fit() first."
            )

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
        model_probs = [
            base_predictions[name].probabilities for name in self.EXPECTED_MODELS
        ]
        disagreement = self._compute_disagreement(base_predictions)
        return np.column_stack([*model_probs, regime_probs, disagreement])

    def _compute_disagreement(
        self, base_predictions: dict[str, PredictionResult]
    ) -> np.ndarray:
        stacked = np.stack(
            [base_predictions[m].probabilities for m in self.EXPECTED_MODELS],
            axis=0,
        )
        mean_probs = stacked.mean(axis=0)
        consensus = np.argmax(mean_probs, axis=1)
        n = len(consensus)
        probs_for_consensus = stacked[:, np.arange(n), consensus]
        return np.std(probs_for_consensus, axis=0)

    def _aggregate_magnitude(
        self, base_predictions: dict[str, PredictionResult]
    ) -> np.ndarray:
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
        return (weights * mags).sum(axis=0)

    def _validate_model_names(
        self, base_predictions: dict[str, PredictionResult]
    ) -> None:
        expected = set(self.EXPECTED_MODELS)
        actual = set(base_predictions.keys())
        if actual != expected:
            raise ValueError(
                f"Expected models {sorted(expected)}, got {sorted(actual)}"
            )

    def _validate_inputs(self, meta_X: np.ndarray) -> None:
        if np.any(~np.isfinite(meta_X)):
            raise DataQualityError("Input contains NaN or Inf values")

    @staticmethod
    def _softmax(logits: np.ndarray) -> np.ndarray:
        exp = np.exp(logits - logits.max(axis=1, keepdims=True))
        return exp / exp.sum(axis=1, keepdims=True)
```

- [ ] **Step 2: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/unit/models/test_ensemble.py -v`
Expected: 14 passed

- [ ] **Step 3: Commit**

```bash
git add alphavedha/models/ensemble.py
git commit -m "feat: add StackingEnsemble with RidgeClassifier meta-learner"
```

---

### Task 3: Write Failing Tests for MetaLabelingModel

**Files:**
- Create: `tests/unit/models/test_meta_model.py`

- [ ] **Step 1: Write all 10 failing tests**

```python
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
def synthetic_meta_data() -> (
    tuple[pd.DataFrame, np.ndarray, np.ndarray, pd.Series]
):
    """Synthetic features + ensemble outputs + binary correctness labels."""
    rng = np.random.default_rng(42)
    n = 200
    X = pd.DataFrame(
        rng.standard_normal((n, 10)), columns=[f"f{i}" for i in range(10)]
    )
    ensemble_direction = rng.choice([-1, 0, 1], size=n).astype(float)
    ensemble_confidence = rng.uniform(0.3, 0.95, size=n)
    y_correct = pd.Series(rng.integers(0, 2, size=n), name="correct")
    return X, ensemble_direction, ensemble_confidence, y_correct


class TestMetaLabelingModelFit:
    def test_fit_returns_metrics(
        self,
        synthetic_meta_data: tuple[
            pd.DataFrame, np.ndarray, np.ndarray, pd.Series
        ],
        meta_config: MetaLabelingConfig,
    ) -> None:
        X, ens_dir, ens_conf, y_correct = synthetic_meta_data
        model = MetaLabelingModel(config=meta_config)
        metrics = model.fit(X, ens_dir, ens_conf, y_correct)
        assert isinstance(metrics, dict)
        assert "accuracy" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1" in metrics

    def test_predict_before_fit_raises(
        self, meta_config: MetaLabelingConfig
    ) -> None:
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
        synthetic_meta_data: tuple[
            pd.DataFrame, np.ndarray, np.ndarray, pd.Series
        ],
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
        assert "accuracy" in metrics


class TestMetaLabelingModelPredict:
    @pytest.fixture(autouse=True)
    def _fitted_model(
        self,
        synthetic_meta_data: tuple[
            pd.DataFrame, np.ndarray, np.ndarray, pd.Series
        ],
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
            self.X, self.ens_dir, self.ens_conf,
            pd.Series(np.random.default_rng(42).integers(0, 2, size=len(self.X))),
        )
        model.fit(X, ens_dir, ens_conf, y_correct)
        result = model.predict(X, ens_dir, ens_conf)
        expected = result.meta_confidence > 0.80
        np.testing.assert_array_equal(result.is_tradeable, expected)


class TestMetaLabelingModelPersistence:
    def test_save_load_roundtrip(
        self,
        synthetic_meta_data: tuple[
            pd.DataFrame, np.ndarray, np.ndarray, pd.Series
        ],
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
        np.testing.assert_array_equal(
            result_before.is_tradeable, result_after.is_tradeable
        )
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/unit/models/test_meta_model.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'alphavedha.models.meta_model'`

- [ ] **Step 3: Commit**

```bash
git add tests/unit/models/test_meta_model.py
git commit -m "test: add failing tests for MetaLabelingModel (10 tests)"
```

---

### Task 4: Implement MetaLabelingModel

**Files:**
- Create: `alphavedha/models/meta_model.py`

- [ ] **Step 1: Implement the full MetaLabelingModel class**

```python
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
        self._validate_inputs(X_aug)

        n_samples = len(X_aug)
        if n_samples < _MIN_SAMPLES:
            raise InsufficientDataError(
                f"Need at least {_MIN_SAMPLES} samples, got {n_samples}"
            )

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
            X_val_aug = self._build_features(
                X_val, ensemble_direction_val, ensemble_confidence_val
            )
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
            "accuracy": float(accuracy_score(y_correct.values, y_pred)),
            "precision": float(precision_score(y_correct.values, y_pred, zero_division=0)),
            "recall": float(recall_score(y_correct.values, y_pred, zero_division=0)),
            "f1": float(f1_score(y_correct.values, y_pred, zero_division=0)),
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
            raise ModelTrainingError(
                "MetaLabelingModel is not fitted. Call fit() first."
            )

        X_aug = self._build_features(X_features, ensemble_direction, ensemble_confidence)
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

    @staticmethod
    def _validate_inputs(X: pd.DataFrame) -> None:
        if np.any(~np.isfinite(X.values)):
            raise DataQualityError("Input contains NaN or Inf values")
```

- [ ] **Step 2: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/unit/models/test_meta_model.py -v`
Expected: 10 passed

- [ ] **Step 3: Commit**

```bash
git add alphavedha/models/meta_model.py
git commit -m "feat: add MetaLabelingModel with XGBClassifier binary gate"
```

---

### Task 5: Export New Classes and Update Docs

**Files:**
- Modify: `alphavedha/models/__init__.py`
- Modify: `alphavedha/models/CLAUDE.md`

- [ ] **Step 1: Update `__init__.py` exports**

Add to imports:

```python
from alphavedha.models.ensemble import EnsembleResult, StackingEnsemble
from alphavedha.models.meta_model import MetaLabelingModel, MetaLabelResult
```

Add to `__all__`:

```python
"EnsembleResult",
"MetaLabelingModel",
"MetaLabelResult",
"StackingEnsemble",
```

The full file after modification:

```python
"""Models — BaseModel ABC, model implementations, and utility models."""

from alphavedha.models.base import (
    BaseModel,
    ModelArtifact,
    PredictionResult,
    TrainResult,
)
from alphavedha.models.conformal import ConformalPredictor, ConformalResult
from alphavedha.models.ensemble import EnsembleResult, StackingEnsemble
from alphavedha.models.lstm_model import LSTMModel
from alphavedha.models.meta_model import MetaLabelingModel, MetaLabelResult
from alphavedha.models.regime import RegimeDetector, RegimeResult
from alphavedha.models.temporal_attention import TemporalAttentionModel
from alphavedha.models.xgboost_model import XGBoostModel

__all__ = [
    "BaseModel",
    "ConformalPredictor",
    "ConformalResult",
    "EnsembleResult",
    "LSTMModel",
    "MetaLabelingModel",
    "MetaLabelResult",
    "ModelArtifact",
    "PredictionResult",
    "RegimeDetector",
    "RegimeResult",
    "StackingEnsemble",
    "TemporalAttentionModel",
    "TrainResult",
    "XGBoostModel",
]
```

- [ ] **Step 2: Update CLAUDE.md ensemble and meta_model sections**

Replace the existing `### ensemble.py (Stacking)` section (lines ~57-61) with:

```markdown
### ensemble.py (Stacking)
- `StackingEnsemble` — does NOT subclass BaseModel (custom interface like RegimeDetector)
- Meta-learner input: 14 features = [xgb_proba(3), lstm_proba(3), tft_proba(3), regime_probs(4), disagreement(1)]
- `_build_meta_features()` constructs the 14-column matrix from base model `PredictionResult.probabilities`
- `_compute_disagreement()` — std of each model's probability for the consensus class (argmax of mean probabilities)
- `_aggregate_magnitude()` — confidence-weighted average of base model magnitudes
- Meta-learner: `sklearn.linear_model.RidgeClassifier(alpha=config.alpha)`
- Probabilities: softmax of `decision_function()` output (RidgeClassifier has no native `predict_proba`)
- model_disagreement = std(base_predictions) — high disagreement = low confidence
- Train meta-learner on OUT-OF-FOLD predictions from base models (never on training predictions)
- Direction mapping: {0: -1, 1: 0, 2: 1} — same as all other models
- Serialization: `joblib.dump()` for RidgeClassifier + `metadata.json`
```

Replace the existing `### meta_model.py (Meta-Labeling)` section (lines ~63-69) with:

```markdown
### meta_model.py (Meta-Labeling)
- `MetaLabelingModel` — does NOT subclass BaseModel (custom interface)
- Input: original feature DataFrame + `ensemble_direction` (int) + `ensemble_confidence` (float)
- `_build_features()` appends the two ensemble columns to the feature DataFrame
- Output: `MetaLabelResult(meta_confidence, is_tradeable)` — P(primary prediction is correct)
- `is_tradeable` = meta_confidence > config.min_confidence (default 0.55)
- Use `XGBClassifier` with binary:logistic, max_depth=4, n_estimators=200
- Early stopping via `early_stopping_rounds=20` when validation set provided
- Threshold: only output predictions where meta_confidence > 0.55
- This model answers "should I bet on this signal?" not "what direction?"
- Serialization: `joblib.dump()` for XGBClassifier + `metadata.json`
```

- [ ] **Step 3: Run full test suite**

Run: `.venv/bin/python -m pytest tests/unit/models/ -v`
Expected: All tests pass (previous + 24 new)

- [ ] **Step 4: Commit**

```bash
git add alphavedha/models/__init__.py alphavedha/models/CLAUDE.md
git commit -m "feat: export StackingEnsemble, MetaLabelingModel from models package"
```

---

### Task 6: Lint, Type-Check, and Full Test Validation

**Files:** None (validation only)

- [ ] **Step 1: Run ruff linter**

Run: `.venv/bin/ruff check alphavedha/models/ensemble.py alphavedha/models/meta_model.py tests/unit/models/test_ensemble.py tests/unit/models/test_meta_model.py`
Expected: No errors. If there are errors, fix them.

- [ ] **Step 2: Run ruff formatter**

Run: `.venv/bin/ruff format alphavedha/models/ensemble.py alphavedha/models/meta_model.py tests/unit/models/test_ensemble.py tests/unit/models/test_meta_model.py`
Expected: Files reformatted or already formatted.

- [ ] **Step 3: Run mypy type-check**

Run: `.venv/bin/python -m mypy alphavedha/models/ensemble.py alphavedha/models/meta_model.py --ignore-missing-imports`
Expected: No errors.

- [ ] **Step 4: Run complete test suite**

Run: `.venv/bin/python -m pytest tests/ -v --tb=short`
Expected: ALL tests pass (previous suite + 24 new = all green).

- [ ] **Step 5: Fix any issues and commit**

If any lint/mypy/test issues found, fix them:

```bash
git add -u
git commit -m "fix: resolve lint, mypy, and formatting issues in Week 6 model files"
```
