# Prediction Engine — AlphaVedha

## Responsibility
Orchestrate the full prediction pipeline: data → features → models → ensemble → risk → output.

## Pipeline Flow

```
1. engine.py receives prediction request (symbol + features + returns + price)
2. Run regime detection (HMM) → get current regime (or default if no market_features)
3. Run all base models (XGBoost, LSTM, TFT) → get raw predictions
4. Graceful degradation: 2/3 minimum, failed models get neutral PredictionResult
5. Run stacking meta-learner → get calibrated prediction
6. Run meta-labeling model → get confidence score + is_tradeable
7. Run conformal prediction → get price target range
8. Run CompositeScorer → get 0-100 composite score
9. Run RiskManager.assess() → get position sizing
10. Return StockPrediction with all fields populated
```

## Modules

### engine.py — PredictionEngine
- Constructor takes all models + scorer + risk_manager via dependency injection
- `predict()` returns `StockPrediction` dataclass with 16 fields
- Graceful degradation: catches individual model failures, applies defaults, adds warnings
- Minimum 2/3 base models must succeed; otherwise raises `PredictionError`
- `market_features=None` → regime skipped, uniform probs `[0.25, 0.25, 0.25, 0.25]`
- `current_portfolio=None` → portfolio constraints and circuit breaker skipped

### scorer.py — CompositeScorer
- 6 weighted sub-scores: technical_momentum, derivatives_sentiment, macro_alignment, microstructure_quality, news_sentiment, volatility_risk
- Feature columns matched by prefix: `deriv_*`, `micro_*`, `sent_*`, `hvol_*`/`natr_*`/`atr_*`/`bb_width_*`
- Missing feature groups have weight redistributed to available groups
- Configurable weights via `CompositeScoreWeights` Pydantic config

### ranker.py — StockRanker
- Filters: is_tradeable, position_size > 0, no circuit-hit
- Produces `RankingResult` with separate buy/sell candidate lists sorted by composite_score desc
- Respects top_n limit
- Tracks excluded symbols with reasons
