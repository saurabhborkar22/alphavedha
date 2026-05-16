"""Tests for CompositeScorer — 0-100 weighted scoring."""

from __future__ import annotations

import numpy as np
import pandas as pd

from alphavedha.config import CompositeScoreWeights
from alphavedha.models.ensemble import EnsembleResult
from alphavedha.models.regime import RegimeResult
from alphavedha.prediction.scorer import CompositeScorer

_DEFAULT_WEIGHTS = CompositeScoreWeights()


def _make_regime(regime: str = "bull") -> RegimeResult:
    probs = np.array([0.7, 0.1, 0.1, 0.1]) if regime == "bull" else np.array([0.1, 0.7, 0.1, 0.1])
    return RegimeResult(
        current_regime=regime,
        regime_id=0 if regime == "bull" else 1,
        state_probabilities=probs,
        regime_history=np.array([0]),
        transition_matrix=np.eye(4),
    )


def _make_ensemble(direction: int = 1, confidence: float = 0.8) -> EnsembleResult:
    probs = np.array([[0.1, 0.1, 0.8]]) if direction == 1 else np.array([[0.8, 0.1, 0.1]])
    return EnsembleResult(
        direction=np.array([direction]),
        magnitude=np.array([0.03]),
        probabilities=probs,
        confidence=np.array([confidence]),
        model_disagreement=np.array([0.05]),
    )


def _make_features_full() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "deriv_pcr_oi": [0.8],
            "deriv_futures_oi_change": [1000],
            "micro_delivery_pct": [0.65],
            "micro_vol_anomaly": [0.3],
            "sent_news_score": [0.7],
            "sent_velocity": [0.5],
            "hvol_20": [0.15],
            "natr_14": [0.02],
            "atr_14": [50.0],
        }
    )


class TestCompositeScorer:
    def test_full_features_score_in_range(self) -> None:
        scorer = CompositeScorer()
        score = scorer.score(
            _make_ensemble(direction=1, confidence=0.8),
            _make_regime("bull"),
            _make_features_full(),
        )
        assert 0.0 <= score <= 100.0

    def test_missing_feature_group_uses_neutral(self) -> None:
        scorer = CompositeScorer()
        features = pd.DataFrame({"some_unrelated": [1.0]})
        score = scorer.score(
            _make_ensemble(direction=1, confidence=0.8),
            _make_regime("bull"),
            features,
        )
        assert 0.0 <= score <= 100.0

    def test_all_features_missing_returns_near_neutral(self) -> None:
        scorer = CompositeScorer()
        features = pd.DataFrame({"unrelated": [1.0]})
        score = scorer.score(
            _make_ensemble(direction=0, confidence=0.5),
            _make_regime("sideways"),
            features,
        )
        assert 40.0 <= score <= 60.0

    def test_bull_regime_buy_signal_high_macro(self) -> None:
        scorer = CompositeScorer()
        score_bull_buy = scorer.score(
            _make_ensemble(direction=1, confidence=0.8),
            _make_regime("bull"),
            _make_features_full(),
        )
        score_bull_sell = scorer.score(
            _make_ensemble(direction=-1, confidence=0.8),
            _make_regime("bull"),
            _make_features_full(),
        )
        assert score_bull_buy > score_bull_sell

    def test_bear_regime_sell_signal_high_macro(self) -> None:
        scorer = CompositeScorer()
        score_bear_sell = scorer.score(
            _make_ensemble(direction=-1, confidence=0.8),
            _make_regime("bear"),
            _make_features_full(),
        )
        score_bear_buy = scorer.score(
            _make_ensemble(direction=1, confidence=0.8),
            _make_regime("bear"),
            _make_features_full(),
        )
        assert score_bear_sell > score_bear_buy

    def test_custom_weights(self) -> None:
        weights = CompositeScoreWeights(
            technical_momentum=1.0,
            derivatives_sentiment=0.0,
            macro_alignment=0.0,
            microstructure_quality=0.0,
            news_sentiment=0.0,
            volatility_risk=0.0,
        )
        scorer = CompositeScorer(weights=weights)
        score = scorer.score(
            _make_ensemble(direction=1, confidence=0.9),
            _make_regime("bull"),
            _make_features_full(),
        )
        assert 0.0 <= score <= 100.0
