"""PredictionEngine — orchestrates the full prediction pipeline."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Protocol

import numpy as np
import pandas as pd
import structlog

from alphavedha.exceptions import PredictionError
from alphavedha.models.base import PredictionResult
from alphavedha.models.conformal import ConformalPredictor
from alphavedha.models.ensemble import EnsembleResult, StackingEnsemble
from alphavedha.models.meta_model import MetaLabelingModel
from alphavedha.models.regime import RegimeDetector, RegimeResult
from alphavedha.prediction.regime_strategy import RegimeStrategySelector
from alphavedha.prediction.scorer import CompositeScorer
from alphavedha.risk.portfolio import PortfolioState
from alphavedha.risk.risk_manager import RiskManager

logger = structlog.get_logger(__name__)

_UNIFORM_REGIME = np.array([0.25, 0.25, 0.25, 0.25])
_MIN_SUCCESSFUL_MODELS = 2


class BaseModelProtocol(Protocol):
    def predict(self, X: pd.DataFrame) -> PredictionResult: ...


_ATR_STOP_MULT: float = 1.5  # mirrors triple_barrier config.multiplier_down
_ATR_TARGET_MULT: float = 2.0  # mirrors triple_barrier config.multiplier_up


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
    entry_price: float | None = None
    stop_loss_price: float | None = None
    take_profit_price: float | None = None
    warnings: list[str] = field(default_factory=list)


@dataclass
class RegimeOverlay:
    """Regime-aware exposure overlay — always active.

    Caps Kelly and, in a market downtrend, cuts position size and suppresses
    new longs. Parameters are tunable via env vars but the overlay itself
    is always on (see docs/prediction_audit.md §8 — the model is long-biased
    and profitable only in bull markets; this overlay prevents giving back
    those gains in downtrends).
    """

    trend_lookback: int = 50
    kelly_cap: float = 0.25
    downtrend_size_mult: float = 0.3
    suppress_longs_in_downtrend: bool = True


def _load_regime_overlay() -> RegimeOverlay:
    """Build the overlay from env vars. Always returns an active overlay."""
    return RegimeOverlay(
        trend_lookback=int(os.environ.get("ALPHAVEDHA_REGIME_OVERLAY_LOOKBACK", "50")),
        kelly_cap=float(os.environ.get("ALPHAVEDHA_REGIME_OVERLAY_KELLY_CAP", "0.25")),
        downtrend_size_mult=float(os.environ.get("ALPHAVEDHA_REGIME_OVERLAY_DOWN_MULT", "0.3")),
        suppress_longs_in_downtrend=os.environ.get(
            "ALPHAVEDHA_REGIME_OVERLAY_SUPPRESS_LONGS", "1"
        ).lower()
        in ("1", "true", "yes"),
    )


def apply_regime_overlay(
    overlay: RegimeOverlay,
    kelly: float,
    direction: int,
    is_tradeable: bool,
    market_features: pd.DataFrame | None,
) -> tuple[float, bool, str | None]:
    """Apply the overlay → (effective_kelly, is_tradeable, warning).

    Kelly cap is always applied. Downtrend suppression requires market
    features; without them only the cap takes effect. Can only further
    restrict ``is_tradeable`` (never enables a trade the gate rejected).
    """
    kelly = min(kelly, overlay.kelly_cap)
    if market_features is None or len(market_features) == 0 or "returns" not in market_features:
        return kelly, is_tradeable, None
    trend = float(market_features["returns"].tail(overlay.trend_lookback).mean())
    if trend < 0:
        kelly *= overlay.downtrend_size_mult
        if overlay.suppress_longs_in_downtrend and direction == 1:
            return kelly, False, "regime_overlay_long_suppressed_downtrend"
    return kelly, is_tradeable, None


class PredictionEngine:
    def __init__(
        self,
        xgboost: BaseModelProtocol,
        lstm: BaseModelProtocol | None = None,
        tft: BaseModelProtocol | None = None,
        regime: RegimeDetector | None = None,
        ensemble: StackingEnsemble | None = None,
        meta_model: MetaLabelingModel | None = None,
        conformal: ConformalPredictor | None = None,
        scorer: CompositeScorer | None = None,
        risk_manager: RiskManager | None = None,
        regime_strategy: RegimeStrategySelector | None = None,
        model_version: str = "v0.1.0",
        conformal_outputs_returns: bool = False,
        gnn: BaseModelProtocol | None = None,
    ) -> None:
        self._models: dict[str, BaseModelProtocol] = {"xgboost": xgboost}
        if lstm is not None:
            self._models["lstm"] = lstm
        if tft is not None:
            self._models["tft"] = tft
        if gnn is not None:
            self._models["gnn"] = gnn
        self._regime = regime
        self._ensemble = ensemble
        self._meta_model = meta_model
        self._conformal = conformal
        self._scorer = scorer or CompositeScorer()
        self._risk_manager = risk_manager
        self._regime_strategy = regime_strategy or RegimeStrategySelector()
        self._model_version = model_version
        # The production conformal model is trained on forward returns, not
        # prices — when set, its intervals are converted to price space
        # using the latest close.
        self._conformal_outputs_returns = conformal_outputs_returns
        self._regime_overlay = _load_regime_overlay()

    def predict(
        self,
        symbol: str,
        features: pd.DataFrame,
        sector: str = "",
        market_features: pd.DataFrame | None = None,
        current_portfolio: PortfolioState | None = None,
        last_close: float | None = None,
    ) -> StockPrediction:
        warnings: list[str] = []
        now = datetime.now(UTC)

        regime_name, regime_probs = self._run_regime(market_features, warnings)

        strategy = self._regime_strategy.select(regime_name)
        warnings.extend(strategy.warnings)

        base_predictions = self._run_base_models(features, warnings)

        # Prediction arrays are row-aligned with `features`; the latest
        # observation is the LAST row (sequence models pad earlier rows).
        if strategy.require_all_models_agree:
            directions = [
                int(p.direction[-1])
                for p in base_predictions.values()
                if hasattr(p, "direction") and len(p.direction) > 0
            ]
            if len(set(directions)) > 1:
                warnings.append(
                    f"High-vol regime: models disagree ({directions}), marking untradeable"
                )

        if self._ensemble is not None:
            ensemble_result = self._ensemble.predict(
                base_predictions,
                np.tile(regime_probs.reshape(1, -1), (len(features), 1)),
            )
        else:
            xgb_pred = base_predictions["xgboost"]
            ensemble_result = EnsembleResult(
                direction=xgb_pred.direction,
                magnitude=xgb_pred.magnitude,
                probabilities=xgb_pred.probabilities,
                confidence=xgb_pred.confidence,
                model_disagreement=np.array([0.0]),
            )
        direction = int(ensemble_result.direction[-1])
        magnitude = float(ensemble_result.magnitude[-1])
        disagreement = float(ensemble_result.model_disagreement[-1])

        meta_confidence, is_tradeable = self._run_meta(features, ensemble_result, warnings)

        if meta_confidence < strategy.meta_confidence_threshold:
            is_tradeable = False

        if strategy.require_all_models_agree:
            directions = [
                int(p.direction[-1])
                for p in base_predictions.values()
                if hasattr(p, "direction") and len(p.direction) > 0
            ]
            if len(set(directions)) > 1:
                is_tradeable = False

        price_low, price_mid, price_high = self._run_conformal(features, warnings)
        if self._conformal_outputs_returns:
            # The feature matrix's close column is NaN in serving (raw prices
            # are not features) — callers pass the latest close explicitly.
            if last_close is not None and np.isfinite(last_close) and last_close > 0:
                price_low = last_close * (1.0 + price_low)
                price_mid = last_close * (1.0 + price_mid)
                price_high = last_close * (1.0 + price_high)
            else:
                warnings.append("No last_close available; price targets are returns")

        regime_result = self._build_regime_result(regime_name, regime_probs)
        composite_score = self._scorer.score(ensemble_result, regime_result, features)

        if self._risk_manager is not None:
            risk = self._risk_manager.assess(
                meta_confidence=meta_confidence,
                magnitude=magnitude,
                symbol=symbol,
                sector=sector,
                portfolio=current_portfolio,
            )
        else:
            from alphavedha.risk.risk_manager import RiskAssessment

            risk = RiskAssessment(
                position_size_pct=5.0,
                kelly_raw=0.5,
                kelly_half=0.25,
                constraint_violations=[],
                circuit_breaker_level=0,
                risk_adjusted=False,
            )

        kelly, is_tradeable, overlay_warning = apply_regime_overlay(
            self._regime_overlay,
            strategy.kelly_fraction,
            direction,
            is_tradeable,
            market_features,
        )
        if overlay_warning:
            warnings.append(overlay_warning)
        position_size = risk.position_size_pct * kelly

        entry_price, stop_loss_price, take_profit_price = self._compute_atr_levels(
            features, last_close, direction
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
            position_size_pct=position_size,
            model_version=self._model_version,
            entry_price=entry_price,
            stop_loss_price=stop_loss_price,
            take_profit_price=take_profit_price,
            warnings=warnings,
        )

    def _compute_atr_levels(
        self,
        features: pd.DataFrame,
        last_close: float | None,
        direction: int,
    ) -> tuple[float | None, float | None, float | None]:
        """Return (entry_price, stop_loss_price, take_profit_price) using ATR14.

        Multipliers match the triple-barrier training labels so the model's
        sense of a "good trade" is consistent with what we show the user.
        Returns (None, None, None) when ATR or last_close is unavailable.
        """
        entry = last_close
        if entry is None or not np.isfinite(entry) or entry <= 0 or direction == 0:
            return entry, None, None
        if "atr_14" not in features.columns:
            return entry, None, None
        atr = float(features["atr_14"].iloc[-1])
        if not np.isfinite(atr) or atr <= 0:
            return entry, None, None
        if direction == 1:
            stop = entry - _ATR_STOP_MULT * atr
            target = entry + _ATR_TARGET_MULT * atr
        else:
            stop = entry + _ATR_STOP_MULT * atr
            target = entry - _ATR_TARGET_MULT * atr
        return entry, round(stop, 2), round(target, 2)

    def _run_regime(
        self,
        market_features: pd.DataFrame | None,
        warnings: list[str],
    ) -> tuple[str, np.ndarray]:
        if self._regime is None or market_features is None:
            warnings.append("No market_features provided; regime detection skipped")
            return "unknown", _UNIFORM_REGIME.copy()
        try:
            extra_cols = [c for c in market_features.columns if c not in ("returns", "volatility")]
            extra = market_features[extra_cols] if extra_cols else None
            result = self._regime.predict(
                returns=market_features["returns"],
                volatility=market_features["volatility"],
                extra_features=extra,
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
        min_required = min(_MIN_SUCCESSFUL_MODELS, len(self._models))
        if n_success < min_required:
            raise PredictionError(
                f"Only {n_success} base model(s) succeeded, fewer than {min_required} required. "
                f"Failed: {failed}"
            )

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
        ensemble_result: EnsembleResult,
        warnings: list[str],
    ) -> tuple[float, bool]:
        if self._meta_model is None:
            return float(ensemble_result.confidence[-1]), True
        try:
            meta_result = self._meta_model.predict(
                features,
                ensemble_result.direction,
                ensemble_result.confidence,
            )
            return float(meta_result.meta_confidence[-1]), bool(meta_result.is_tradeable[-1])
        except Exception as e:
            logger.warning("meta_model_failed", error=str(e))
            warnings.append(f"Meta-labeling failed: {e}")
            return 0.0, False

    def _run_conformal(
        self,
        features: pd.DataFrame,
        warnings: list[str],
    ) -> tuple[float, float, float]:
        if self._conformal is None:
            return float("nan"), float("nan"), float("nan")
        try:
            result = self._conformal.predict(features)
            return (
                float(result.price_low[-1]),
                float(result.price_mid[-1]),
                float(result.price_high[-1]),
            )
        except Exception as e:
            logger.warning("conformal_failed", error=str(e))
            warnings.append(f"Conformal prediction failed: {e}")
            return float("nan"), float("nan"), float("nan")

    def _build_regime_result(self, regime_name: str, regime_probs: np.ndarray) -> RegimeResult:
        return RegimeResult(
            current_regime=regime_name,
            regime_id=0,
            state_probabilities=regime_probs,
            regime_history=np.array([0]),
            transition_matrix=np.eye(4),
        )
