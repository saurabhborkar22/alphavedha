# Prediction Engine — AlphaVedha

## Responsibility
Orchestrate the full prediction pipeline: data → features → models → ensemble → risk → output.

## Pipeline Flow

```
1. engine.py receives prediction request (symbol + horizon)
2. Fetch latest data from feature store (or compute if stale)
3. Run regime detection (HMM) → get current regime
4. Run all base models (XGBoost, LSTM, TFT) → get raw predictions
5. Compute model disagreement (std of base predictions)
6. Run stacking meta-learner → get calibrated prediction
7. Run meta-labeling model → get confidence score
8. Run conformal prediction → get price target range
9. Run risk management → get position sizing
10. Compute composite score (0-100)
11. Return structured PredictionResult
```

## Composite Score (scorer.py)

Weighted combination of sub-scores, each 0-100:

```python
SCORE_WEIGHTS = {
    "technical_momentum": 0.25,
    "derivatives_sentiment": 0.20,
    "macro_alignment": 0.15,
    "microstructure_quality": 0.15,
    "news_sentiment": 0.10,
    "volatility_risk": 0.15,   # Inverted — low vol = high score
}
```

Each sub-score is a percentile rank within the current universe.

## Ranker (ranker.py)
- Rank all stocks in a tier by composite_score
- Apply filters: min confidence > 0.55, min liquidity, no circuit-hit today
- Return top-N with full prediction details
- Separate rankings for BUY candidates (direction=UP) and SELL candidates (direction=DOWN)

## Rules
- Cache intermediate results — don't recompute features for the same stock on the same day
- If any model fails, return predictions from remaining models with a warning flag
- Never return a prediction without a confidence score and model version
- Log the full pipeline execution time for performance monitoring
