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


def _sigmoid_normalize(value: float, scale: float = 2.0, invert: bool = False) -> float:
    """Map a single value to [0, 1] via sigmoid. Invert for "lower is better" metrics."""
    x = -value * scale if invert else value * scale
    return 1.0 / (1.0 + np.exp(-x))


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
            features: DataFrame with raw feature columns; the last row
                (latest observation) is the one scored.

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

        direction = int(ensemble_result.direction[-1])
        confidence = float(ensemble_result.confidence[-1])

        sub_scores: dict[str, float | None] = {}

        # Technical momentum: always available from ensemble output
        sub_scores["technical_momentum"] = confidence * 100.0

        # Macro alignment: always available from regime output
        regime = regime_result.current_regime
        sub_scores["macro_alignment"] = _REGIME_ALIGNMENT.get((regime, direction), 50.0)

        # Feature-derived sub-scores — normalize each feature individually
        # via sigmoid, then average. This avoids scale mismatch when
        # features in the same group span different orders of magnitude
        # (e.g., atr_14 ≈ 50 vs natr_14 ≈ 0.02).
        invert = False
        for score_name, prefixes in _FEATURE_PREFIXES.items():
            cols = [c for c in features.columns if any(c.startswith(p) for p in prefixes)]
            if not cols:
                sub_scores[score_name] = None
            else:
                values = features[cols].iloc[-1].values.astype(float)
                finite_vals = values[np.isfinite(values)]
                if len(finite_vals) == 0:
                    sub_scores[score_name] = None
                else:
                    invert = score_name == "volatility_risk"
                    per_feature = [_sigmoid_normalize(float(v), invert=invert) for v in finite_vals]
                    sub_scores[score_name] = float(np.mean(per_feature)) * 100.0

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
