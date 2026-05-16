"""CompositeScorer — convert model outputs + features into a 0-100 human-readable score."""

from __future__ import annotations

import numpy as np
import pandas as pd
import structlog

from alphavedha.config import CompositeScoreWeights
from alphavedha.models.ensemble import EnsembleResult
from alphavedha.models.regime import RegimeResult

logger = structlog.get_logger(__name__)

_DEFAULT_WEIGHTS = CompositeScoreWeights()

_REGIME_ALIGNMENT: dict[tuple[str, int], float] = {
    ("bull", 1): 100.0,
    ("bull", 0): 50.0,
    ("bull", -1): 0.0,
    ("bear", -1): 100.0,
    ("bear", 0): 50.0,
    ("bear", 1): 0.0,
    ("sideways", 0): 70.0,
    ("sideways", 1): 40.0,
    ("sideways", -1): 40.0,
    ("high_volatility", 0): 60.0,
    ("high_volatility", 1): 30.0,
    ("high_volatility", -1): 30.0,
}

_FEATURE_PREFIXES: dict[str, list[str]] = {
    "derivatives_sentiment": ["deriv_"],
    "microstructure_quality": ["micro_"],
    "news_sentiment": ["sent_"],
    "volatility_risk": ["hvol_", "natr_", "atr_", "bb_width_"],
}


class CompositeScorer:
    """Convert ensemble + regime outputs and raw features into a 0-100 composite score.

    Sub-scores:
    - technical_momentum: derived from ensemble confidence (always available)
    - macro_alignment: regime x direction alignment (always available)
    - derivatives_sentiment: from ``deriv_*`` feature columns
    - microstructure_quality: from ``micro_*`` feature columns
    - news_sentiment: from ``sent_*`` feature columns
    - volatility_risk: from ``hvol_*``, ``natr_*``, ``atr_*``, ``bb_width_*`` columns (inverted)

    When a feature group is absent, its weight is redistributed proportionally across
    the remaining available sub-scores so the final score is always on [0, 100].
    """

    def __init__(self, weights: CompositeScoreWeights | None = None) -> None:
        self._weights = weights or _DEFAULT_WEIGHTS

    def score(
        self,
        ensemble_result: EnsembleResult,
        regime_result: RegimeResult,
        features: pd.DataFrame,
    ) -> float:
        """Compute the composite score.

        Args:
            ensemble_result: Output from StackingEnsemble.predict().
            regime_result: Output from RegimeDetector.predict().
            features: Single-row DataFrame with raw feature columns.

        Returns:
            Float in [0.0, 100.0].
        """
        weight_dict: dict[str, float] = {
            "technical_momentum": self._weights.technical_momentum,
            "derivatives_sentiment": self._weights.derivatives_sentiment,
            "macro_alignment": self._weights.macro_alignment,
            "microstructure_quality": self._weights.microstructure_quality,
            "news_sentiment": self._weights.news_sentiment,
            "volatility_risk": self._weights.volatility_risk,
        }

        direction = int(ensemble_result.direction[0])
        confidence = float(ensemble_result.confidence[0])

        sub_scores: dict[str, float | None] = {}

        # Technical momentum: always available from ensemble output
        sub_scores["technical_momentum"] = confidence * 100.0

        # Macro alignment: always available from regime output
        regime = regime_result.current_regime
        sub_scores["macro_alignment"] = _REGIME_ALIGNMENT.get((regime, direction), 50.0)

        # Feature-derived sub-scores
        for score_name, prefixes in _FEATURE_PREFIXES.items():
            cols = [c for c in features.columns if any(c.startswith(p) for p in prefixes)]
            if not cols:
                sub_scores[score_name] = None
            else:
                values = features[cols].iloc[0].values.astype(float)
                finite_vals = values[np.isfinite(values)]
                if len(finite_vals) == 0:
                    sub_scores[score_name] = None
                else:
                    mean_val = float(np.mean(finite_vals))
                    if score_name == "volatility_risk":
                        # High volatility → lower score (inverted via sigmoid)
                        normalized = 1.0 / (1.0 + np.exp(mean_val * 5))
                        sub_scores[score_name] = float(normalized * 100.0)
                    else:
                        # Higher positive values → higher score
                        normalized = 1.0 / (1.0 + np.exp(-mean_val * 2))
                        sub_scores[score_name] = float(normalized * 100.0)

        # Redistribute weight from unavailable sub-scores to available ones
        available_weight = sum(weight_dict[k] for k, v in sub_scores.items() if v is not None)
        if available_weight <= 0:
            return 50.0

        weighted_sum = 0.0
        for name, raw_score in sub_scores.items():
            if raw_score is None:
                continue
            normalized_weight = weight_dict[name] / available_weight
            weighted_sum += raw_score * normalized_weight

        result = max(0.0, min(100.0, weighted_sum))

        logger.debug(
            "composite_score_computed",
            score=round(result, 2),
            sub_scores={k: round(v, 2) if v is not None else None for k, v in sub_scores.items()},
        )

        return result
