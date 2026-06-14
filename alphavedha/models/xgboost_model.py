"""XGBoostModel — wraps XGBClassifier (direction) + XGBRegressor (magnitude)."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import structlog
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error
from xgboost import XGBClassifier, XGBRegressor

from alphavedha.config import XGBoostConfig
from alphavedha.exceptions import ModelTrainingError
from alphavedha.models.base import BaseModel, PredictionResult, TrainResult

logger = structlog.get_logger(__name__)

_LABEL_MAP = {-1: 0, 0: 1, 1: 2}
_LABEL_REVERSE = {0: -1, 1: 0, 2: 1}


class XGBoostModel(BaseModel):
    def __init__(self, config: XGBoostConfig | None = None, name: str = "xgboost") -> None:
        cfg = config or XGBoostConfig()
        params = cfg.params
        self._xgb_params: dict[str, Any] = {
            "learning_rate": params.learning_rate,
            "max_depth": params.max_depth,
            "n_estimators": params.n_estimators,
            "subsample": params.subsample,
            "colsample_bytree": params.colsample_bytree,
            "reg_alpha": params.reg_alpha,
            "reg_lambda": params.reg_lambda,
            "random_state": 42,
            "n_jobs": -1,
            "verbosity": 0,
        }
        self._early_stopping_rounds = params.early_stopping_rounds
        self._classifier: XGBClassifier | None = None
        self._regressor: XGBRegressor | None = None
        super().__init__(name=name, config=self._xgb_params)

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame | None = None,
        y_val: pd.Series | None = None,
        sample_weight: pd.Series | None = None,
        return_train: pd.Series | None = None,
        return_val: pd.Series | None = None,
    ) -> TrainResult:
        start = time.perf_counter()
        self._feature_names = list(X_train.columns)

        y_cls_train = y_train.map(_LABEL_MAP).astype(int)
        weight_arr = sample_weight.values if sample_weight is not None else None

        self._classifier = XGBClassifier(
            objective="multi:softprob",
            num_class=3,
            eval_metric="mlogloss",
            early_stopping_rounds=self._early_stopping_rounds,
            **self._xgb_params,
        )

        y_cls_val = None
        eval_set_cls = []
        if X_val is not None and y_val is not None:
            y_cls_val = y_val.map(_LABEL_MAP).astype(int)
            eval_set_cls = [(X_val, y_cls_val)]

        self._classifier.fit(
            X_train,
            y_cls_train,
            eval_set=eval_set_cls or None,
            sample_weight=weight_arr,
            verbose=False,
        )

        train_metrics: dict[str, float] = {}
        cls_train_pred = self._classifier.predict(X_train)
        train_metrics["accuracy"] = float(accuracy_score(y_cls_train, cls_train_pred))
        train_metrics["f1_weighted"] = float(
            f1_score(y_cls_train, cls_train_pred, average="weighted")
        )

        val_metrics: dict[str, float] = {}
        if X_val is not None and y_cls_val is not None:
            cls_val_pred = self._classifier.predict(X_val)
            val_metrics["accuracy"] = float(accuracy_score(y_cls_val, cls_val_pred))
            val_metrics["f1_weighted"] = float(
                f1_score(y_cls_val, cls_val_pred, average="weighted")
            )

        self._regressor = XGBRegressor(
            objective="reg:squarederror",
            eval_metric="rmse",
            early_stopping_rounds=self._early_stopping_rounds,
            **self._xgb_params,
        )

        if return_train is not None:
            eval_set_reg = []
            if X_val is not None and return_val is not None:
                eval_set_reg = [(X_val, return_val)]

            self._regressor.fit(
                X_train,
                return_train,
                eval_set=eval_set_reg or None,
                sample_weight=weight_arr,
                verbose=False,
            )

            reg_train_pred = self._regressor.predict(X_train)
            train_metrics["rmse"] = float(np.sqrt(mean_squared_error(return_train, reg_train_pred)))
            if return_val is not None and X_val is not None:
                reg_val_pred = self._regressor.predict(X_val)
                val_metrics["rmse"] = float(np.sqrt(mean_squared_error(return_val, reg_val_pred)))

        fi_raw = self._classifier.feature_importances_
        fi = pd.Series(fi_raw, index=self._feature_names, name="importance")

        elapsed = time.perf_counter() - start
        self._is_fitted = True
        self._train_metrics = train_metrics

        logger.info(
            "xgboost_trained",
            train_metrics=train_metrics,
            val_metrics=val_metrics,
            training_time_s=round(elapsed, 2),
            n_train=len(X_train),
        )

        return TrainResult(
            train_metrics=train_metrics,
            val_metrics=val_metrics,
            feature_importances=fi,
            training_time_seconds=elapsed,
            n_train_samples=len(X_train),
            n_val_samples=len(X_val) if X_val is not None else 0,
            hyperparams=dict(self._xgb_params),
        )

    def predict(self, X: pd.DataFrame) -> PredictionResult:
        if not self._is_fitted or self._classifier is None:
            raise ModelTrainingError("XGBoostModel is not fitted. Call fit() first.")

        X = self._align_features(X)
        proba = self._classifier.predict_proba(X)
        cls_pred = np.argmax(proba, axis=1)
        direction = np.array([_LABEL_REVERSE[c] for c in cls_pred])
        confidence = np.max(proba, axis=1)

        magnitude = self._regressor.predict(X) if self._regressor is not None else np.zeros(len(X))

        return PredictionResult(
            direction=direction,
            magnitude=magnitude,
            probabilities=proba,
            confidence=confidence,
        )

    def get_feature_importance(self) -> pd.Series | None:
        if self._classifier is None:
            return None
        fi = self._classifier.feature_importances_
        return pd.Series(fi, index=self._feature_names, name="importance")

    def _save_model_artifacts(self, directory: Path) -> None:
        if self._classifier is not None:
            joblib.dump(self._classifier, directory / "classifier.joblib")
        if self._regressor is not None:
            joblib.dump(self._regressor, directory / "regressor.joblib")

    @classmethod
    def _load_model_artifacts(cls, directory: Path, config: dict[str, Any]) -> XGBoostModel:
        model = cls(config=None, name="xgboost")
        model._config = config

        cls_path = directory / "classifier.joblib"
        if cls_path.exists():
            model._classifier = joblib.load(cls_path)

        reg_path = directory / "regressor.joblib"
        if reg_path.exists():
            model._regressor = joblib.load(reg_path)

        model._is_fitted = True
        return model
