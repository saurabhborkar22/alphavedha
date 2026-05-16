"""PredictionEngine — orchestrates the full prediction pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd
import structlog

from alphavedha.exceptions import PredictionError
from alphavedha.models.base import PredictionResult
from alphavedha.models.conformal import ConformalPredictor
from alphavedha.models.ensemble import StackingEnsemble
from alphavedha.models.meta_model import MetaLabelingModel
from alphavedha.models.regime import RegimeDetector
from alphavedha.prediction.scorer import CompositeScorer
from alphavedha.risk.portfolio import PortfolioState
from alphavedha.risk.risk_manager import RiskManager

logger = structlog.get_logger(__name__)

_NEUTRAL_PROBS = np.array([[1 / 3, 1 / 3, 1 / 3]])
_UNIFORM_REGIME = np.array([0.25, 0.25, 0.25, 0.25])
_MIN_SUCCESSFUL_MODELS = 2


@dataclass
class StockPrediction:
    symbol: str
    timestamp: datetime
    direction: int
    magnitude: float
    composite_score: float
    meta_confidence: float
    is_tradeable: bool
    regime: str
    regime_probabilities: np.ndarray
    price_target_low: float
    price_target_mid: float
    price_target_high: float
    model_disagreement: float
    position_size_pct: float
    model_version: str
    warnings: list[str] = field(default_factory=list)


class PredictionEngine:
    def __init__(
        self,
        xgboost: Any,
        lstm: Any,
        tft: Any,
        regime: RegimeDetector,
        ensemble: StackingEnsemble,
        meta_model: MetaLabelingModel,
        conformal: ConformalPredictor,
        scorer: CompositeScorer,
        risk_manager: RiskManager,
        model_version: str = "v0.1.0",
    ) -> None:
        self._models = {"xgboost": xgboost, "lstm": lstm, "tft": tft}
        self._regime = regime
        self._ensemble = ensemble
        self._meta_model = meta_model
        self._conformal = conformal
        self._scorer = scorer
        self._risk_manager = risk_manager
        self._model_version = model_version

    def predict(
        self,
        symbol: str,
        features: pd.DataFrame,
        returns: pd.Series,
        current_price: float,
        market_features: pd.DataFrame | None = None,
        current_portfolio: PortfolioState | None = None,
    ) -> StockPrediction:
        warnings: list[str] = []
        now = datetime.now(UTC)

        # Step 1: Regime detection
        regime_name, regime_probs = self._run_regime(market_features, warnings)

        # Step 2-4: Base models with graceful degradation
        base_predictions = self._run_base_models(features, warnings)

        # Step 5: Ensemble
        ensemble_result = self._ensemble.predict(base_predictions, regime_probs.reshape(1, -1))
        direction = int(ensemble_result.direction[0])
        magnitude = float(ensemble_result.magnitude[0])
        disagreement = float(ensemble_result.model_disagreement[0])

        # Step 6: Meta-labeling
        meta_confidence, is_tradeable = self._run_meta(features, ensemble_result, warnings)

        # Step 7: Conformal prediction
        price_low, price_mid, price_high = self._run_conformal(features, warnings)

        # Step 8: Composite score
        regime_result = self._build_regime_result(regime_name, regime_probs)
        composite_score = self._scorer.score(ensemble_result, regime_result, features)

        # Step 9: Risk assessment
        risk = self._risk_manager.assess(
            meta_confidence=meta_confidence,
            magnitude=magnitude,
            symbol=symbol,
            sector="",
            portfolio=current_portfolio,
        )

        return StockPrediction(
            symbol=symbol,
            timestamp=now,
            direction=direction,
            magnitude=magnitude,
            composite_score=composite_score,
            meta_confidence=meta_confidence,
            is_tradeable=is_tradeable,
            regime=regime_name,
            regime_probabilities=regime_probs,
            price_target_low=price_low,
            price_target_mid=price_mid,
            price_target_high=price_high,
            model_disagreement=disagreement,
            position_size_pct=risk.position_size_pct,
            model_version=self._model_version,
            warnings=warnings,
        )

    def _run_regime(
        self,
        market_features: pd.DataFrame | None,
        warnings: list[str],
    ) -> tuple[str, np.ndarray]:
        if market_features is None:
            warnings.append("No market_features provided; regime detection skipped")
            return "unknown", _UNIFORM_REGIME.copy()
        try:
            result = self._regime.predict(
                returns=market_features.iloc[:, 0],
                volatility=market_features.iloc[:, 1],
            )
            return result.current_regime, result.state_probabilities
        except Exception as e:
            logger.warning("regime_detection_failed", error=str(e))
            warnings.append(f"Regime detection failed: {e}")
            return "unknown", _UNIFORM_REGIME.copy()

    def _run_base_models(
        self,
        features: pd.DataFrame,
        warnings: list[str],
    ) -> dict[str, PredictionResult]:
        results: dict[str, PredictionResult] = {}
        failed: list[str] = []

        for name, model in self._models.items():
            try:
                results[name] = model.predict(features)
            except Exception as e:
                logger.warning("base_model_failed", model=name, error=str(e))
                warnings.append(f"{name} model failed: {e}")
                failed.append(name)

        n_success = len(results)
        if n_success < _MIN_SUCCESSFUL_MODELS:
            raise PredictionError(
                f"Only {n_success} base model(s) succeeded, fewer than 2 required. Failed: {failed}"
            )

        # Fill failed models with neutral predictions
        n = features.shape[0]
        for name in failed:
            results[name] = PredictionResult(
                direction=np.zeros(n, dtype=int),
                magnitude=np.zeros(n),
                probabilities=np.tile([1 / 3, 1 / 3, 1 / 3], (n, 1)),
                confidence=np.zeros(n),
            )

        return results

    def _run_meta(
        self,
        features: pd.DataFrame,
        ensemble_result: Any,
        warnings: list[str],
    ) -> tuple[float, bool]:
        try:
            meta_result = self._meta_model.predict(
                features,
                ensemble_result.direction,
                ensemble_result.confidence,
            )
            return float(meta_result.meta_confidence[0]), bool(meta_result.is_tradeable[0])
        except Exception as e:
            logger.warning("meta_model_failed", error=str(e))
            warnings.append(f"Meta-labeling failed: {e}")
            return 0.0, False

    def _run_conformal(
        self,
        features: pd.DataFrame,
        warnings: list[str],
    ) -> tuple[float, float, float]:
        try:
            result = self._conformal.predict(features)
            return (
                float(result.price_low[0]),
                float(result.price_mid[0]),
                float(result.price_high[0]),
            )
        except Exception as e:
            logger.warning("conformal_failed", error=str(e))
            warnings.append(f"Conformal prediction failed: {e}")
            return float("nan"), float("nan"), float("nan")

    def _build_regime_result(self, regime_name: str, regime_probs: np.ndarray) -> Any:
        from alphavedha.models.regime import RegimeResult

        return RegimeResult(
            current_regime=regime_name,
            regime_id=0,
            state_probabilities=regime_probs,
            regime_history=np.array([0]),
            transition_matrix=np.eye(4),
        )
