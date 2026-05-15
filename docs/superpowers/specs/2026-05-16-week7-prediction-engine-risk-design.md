# Week 7: Prediction Engine + Risk Management — Design Spec

## Goal

Wire all 7 ML models into a single `predict()` pipeline that returns a fully risk-adjusted, scored, and ranked stock prediction. This transforms the individual model components (Weeks 1-6) into a usable prediction system.

## Architecture

Two new modules — `prediction/` (orchestrator, scorer, ranker) and `risk/` (position sizing, portfolio constraints, circuit breaker) — connected by a `RiskManager` that the prediction engine calls internally. Consumers get one call: `engine.predict(symbol, features, returns, price)` → `StockPrediction`.

All models are injected as constructor arguments (dependency injection), making the engine trivially testable with mocks.

## Tech Stack

- Python 3.12, Pydantic v2 for config (already in project)
- numpy, pandas for computation
- structlog for structured logging
- No new dependencies

---

## Module 1: prediction/engine.py — PredictionEngine

### Responsibility

Orchestrate the full prediction pipeline: features → regime → base models → ensemble → meta-label → conformal → score → risk → output.

### Interface

```python
@dataclass
class StockPrediction:
    symbol: str
    timestamp: datetime
    direction: int                    # -1 (sell), 0 (hold), 1 (buy)
    magnitude: float                  # expected return magnitude
    composite_score: float            # 0-100 human-readable score
    meta_confidence: float            # P(prediction is correct)
    is_tradeable: bool                # meta_confidence > threshold
    regime: str                       # current regime label (e.g. "bull")
    regime_probabilities: np.ndarray  # shape (4,) — [bull, bear, sideways, high_vol]
    price_target_low: float           # conformal lower bound
    price_target_mid: float           # conformal mid (point estimate)
    price_target_high: float          # conformal upper bound
    model_disagreement: float         # std of base model consensus probabilities
    position_size_pct: float          # risk-adjusted position % (0 if not tradeable)
    model_version: str                # version string for tracking
    warnings: list[str]               # degradation warnings (e.g. "lstm model failed")
```

### Constructor

```python
class PredictionEngine:
    def __init__(
        self,
        xgboost: XGBoostModel,
        lstm: LSTMModel,
        tft: TemporalAttentionModel,
        regime: RegimeDetector,
        ensemble: StackingEnsemble,
        meta_model: MetaLabelingModel,
        conformal: ConformalPredictor,
        scorer: CompositeScorer,
        risk_manager: RiskManager,
        model_version: str = "v0.1.0",
    ) -> None:
```

### Pipeline Steps

1. Run `regime.predict(market_features)` → `RegimeResult` (regime label + probabilities). Note: RegimeDetector operates on market-level features (Nifty returns + VIX), not per-stock features. The engine must extract or receive these separately.
2. Run base models in sequence:
   - `xgboost.predict(features)` → `PredictionResult`
   - `lstm.predict(features)` → `PredictionResult`
   - `tft.predict(features)` → `PredictionResult`
3. If any base model raises, catch the exception, log a warning, and proceed with remaining models. If fewer than 2 base models succeed, raise `PredictionError`.
4. Build `base_predictions` dict from successful models. For failed models, synthesize a neutral `PredictionResult` (direction=0, confidence=0, uniform probabilities).
5. Run `ensemble.predict(base_predictions, regime_probs)` → `EnsembleResult`
6. Run `meta_model.predict(features, ensemble_direction, ensemble_confidence)` → `MetaLabelResult`
7. Run `conformal.predict(features)` → `ConformalResult` (arrays; take index [0] for single-stock prediction)
8. Run `scorer.score(ensemble_result, regime_result, features)` → `float` (0-100)
9. Run `risk_manager.assess(meta_confidence, magnitude, current_portfolio)` → `RiskAssessment`
10. Assemble and return `StockPrediction`

### Graceful Degradation

- 3/3 base models succeed: normal operation
- 2/3 succeed: proceed with warning, failed model gets neutral PredictionResult (direction=0, confidence=0, probabilities=[1/3, 1/3, 1/3])
- 1/3 or 0/3 succeed: raise `PredictionError` — insufficient model coverage
- Regime, ensemble, meta-label, conformal failures: each is caught individually, defaults applied, warning added

### predict() Signature

```python
def predict(
    self,
    symbol: str,
    features: pd.DataFrame,
    returns: pd.Series,
    current_price: float,
    market_features: pd.DataFrame | None = None,
    current_portfolio: PortfolioState | None = None,
) -> StockPrediction:
```

`market_features` is optional — if None, regime detection is skipped (default to uniform regime probs `[0.25, 0.25, 0.25, 0.25]` with warning). `current_portfolio` is optional — if None, position sizing runs without portfolio constraints (useful for single-stock predictions and testing).

---

## Module 2: prediction/scorer.py — CompositeScorer

### Responsibility

Convert raw model outputs + feature values into a 0-100 human-readable composite score.

### Sub-scores

Six sub-scores, each normalized to 0-100, with configurable weights:

| Sub-score | Weight | Source | Computation |
|-----------|--------|--------|-------------|
| technical_momentum | 0.25 | ensemble confidence + direction | confidence × 100 if direction matches signal, scaled |
| derivatives_sentiment | 0.20 | feature columns (OI, PCR) | percentile rank of OI change + PCR features |
| macro_alignment | 0.15 | regime + direction | bull+buy=100, bull+sell=0, etc. |
| microstructure_quality | 0.15 | volume, spread features | percentile rank of volume/spread quality |
| news_sentiment | 0.10 | sentiment features | percentile rank of sentiment score |
| volatility_risk | 0.15 | volatility features | inverted: low vol=100, high vol=0 |

### Missing Feature Handling

If a feature group needed for a sub-score is entirely absent from the DataFrame:
- That sub-score defaults to 50.0 (neutral)
- Its weight is redistributed proportionally to the available sub-scores
- A warning is logged

### Interface

```python
class CompositeScorer:
    def __init__(self, weights: dict[str, float] | None = None) -> None:
        """Use default weights if None."""

    def score(
        self,
        ensemble_result: EnsembleResult,
        regime_result: RegimeResult,
        features: pd.DataFrame,
    ) -> float:
        """Return composite score in [0, 100]."""
```

### Feature Column Mapping

The scorer looks for feature columns by prefix convention:
- `derivatives_*` → derivatives_sentiment sub-score
- `microstructure_*` → microstructure_quality sub-score
- `sentiment_*` → news_sentiment sub-score
- `volatility_*` → volatility_risk sub-score

If no columns match a prefix, that sub-score is unavailable.

---

## Module 3: prediction/ranker.py — StockRanker

### Responsibility

Take a list of `StockPrediction` objects and produce ranked buy/sell candidate lists.

### Filters Applied

1. `is_tradeable == True` (meta_confidence above threshold)
2. `position_size_pct > 0` (passed risk checks)
3. No circuit-hit flag in features (optional, passed as parameter)

### Output

```python
@dataclass
class RankingResult:
    buy_candidates: list[StockPrediction]   # direction=1, sorted by composite_score desc
    sell_candidates: list[StockPrediction]   # direction=-1, sorted by composite_score desc
    excluded: list[tuple[str, str]]         # (symbol, reason) for filtered-out stocks
    generated_at: datetime
```

### Interface

```python
class StockRanker:
    def rank(
        self,
        predictions: list[StockPrediction],
        top_n: int = 10,
        circuit_hit_symbols: set[str] | None = None,
    ) -> RankingResult:
```

---

## Module 4: risk/position_sizing.py — Half-Kelly Position Sizing

### Responsibility

Compute optimal position size using fractional Kelly criterion.

### Formula

```
win_prob = meta_confidence
loss_prob = 1 - meta_confidence
avg_win = abs(magnitude) when direction is correct
avg_loss = abs(magnitude) when direction is wrong (estimated as magnitude × loss_ratio)

kelly_fraction = (win_prob × avg_win - loss_prob × avg_loss) / avg_win
position_pct = kelly_fraction × 0.5           # half-Kelly
position_pct = clamp(position_pct, 0.0, max_single_stock_pct)
```

For simplicity at this stage, `avg_win = abs(magnitude)` and `avg_loss = abs(magnitude)` (symmetric assumption). This can be refined later with historical win/loss data.

### Interface

```python
def compute_position_size(
    meta_confidence: float,
    magnitude: float,
    config: PositionSizingConfig,
) -> float:
    """Return position size as percentage of portfolio (0.0-10.0)."""
```

### Edge Cases

- `meta_confidence < config.min_confidence` → return 0.0
- `magnitude <= 0` → return 0.0
- Negative Kelly (expected loss) → return 0.0
- Result capped at `config.max_single_stock_pct` (default 10.0)

---

## Module 5: risk/portfolio.py — Portfolio Constraints

### Responsibility

Check proposed trades against portfolio-level rules.

### Constraints

1. **Single stock cap**: no position > 10% of portfolio value
2. **Sector cap**: total exposure to one sector < 25%
3. **Correlation cap**: reject if 60d rolling correlation with any existing holding > 0.7
4. **Minimum holding period**: reject sell for positions held < 3 days
5. **Liquidity filter**: reject if 20d avg daily turnover < Rs 5 crore

### State

```python
@dataclass
class PortfolioState:
    holdings: dict[str, HoldingInfo]    # symbol → holding details
    total_value: float                   # current portfolio value
    peak_value: float                    # high-water mark for circuit breaker

@dataclass
class HoldingInfo:
    symbol: str
    sector: str
    weight_pct: float                    # current % of portfolio
    entry_date: datetime
    correlation_60d: dict[str, float]    # correlations with other holdings
    avg_daily_turnover_cr: float         # 20d avg turnover in crores
```

### Interface

```python
class PortfolioConstraints:
    def __init__(self, config: PortfolioConfig) -> None: ...

    def check(
        self,
        symbol: str,
        proposed_weight_pct: float,
        sector: str,
        portfolio: PortfolioState,
    ) -> ConstraintResult:
        """Return adjusted weight and list of violated constraints."""

@dataclass
class ConstraintResult:
    adjusted_weight_pct: float
    violations: list[str]        # human-readable constraint violation messages
    passed: bool                 # True if no violations (or all were soft-adjusted)
```

---

## Module 6: risk/circuit_breaker.py — Drawdown Protection

### Responsibility

Track portfolio drawdown from peak and enforce protection levels.

### Levels

| Level | Drawdown | Action |
|-------|----------|--------|
| 1 | 10% | Reduce all positions by 50%, log alert |
| 2 | 15% | Halt new entries (position_size → 0 for new trades) |
| 3 | 20% | Close all positions (full stop) |
| Reset | Recovery to 95% of peak | Return to normal trading |

### State

```python
@dataclass
class CircuitBreakerState:
    level: int                  # 0 (normal), 1, 2, or 3
    current_drawdown_pct: float # current drawdown from peak
    peak_value: float           # high-water mark
    triggered_at: datetime | None
```

### Interface

```python
class CircuitBreaker:
    def __init__(self, config: CircuitBreakerConfig) -> None: ...

    def evaluate(
        self,
        current_value: float,
        peak_value: float,
    ) -> CircuitBreakerState:
        """Compute current drawdown level and return state."""

    def adjust_position(
        self,
        proposed_size_pct: float,
        state: CircuitBreakerState,
        is_new_entry: bool,
    ) -> float:
        """Adjust position size based on circuit breaker level."""
```

### Adjustment Logic

- Level 0: no adjustment
- Level 1: `proposed_size_pct × 0.5`
- Level 2: `0.0` if `is_new_entry`, else `proposed_size_pct × 0.5`
- Level 3: `0.0` (close all)

---

## Module 7: risk/risk_manager.py — RiskManager

### Responsibility

Orchestrate all risk checks in sequence, returning a single assessment.

### Interface

```python
@dataclass
class RiskAssessment:
    position_size_pct: float         # final position size (0 = no trade)
    kelly_raw: float                 # raw Kelly fraction before adjustments
    kelly_half: float                # half-Kelly before portfolio constraints
    constraint_violations: list[str] # any portfolio constraint violations
    circuit_breaker_level: int       # current CB level (0-3)
    risk_adjusted: bool              # True if position was reduced by risk checks

class RiskManager:
    def __init__(
        self,
        position_config: PositionSizingConfig,
        portfolio_config: PortfolioConfig,
        circuit_breaker_config: CircuitBreakerConfig,
    ) -> None: ...

    def assess(
        self,
        meta_confidence: float,
        magnitude: float,
        symbol: str,
        sector: str,
        portfolio: PortfolioState | None = None,
    ) -> RiskAssessment:
```

### Pipeline

1. Compute Kelly position size → `kelly_raw`, `kelly_half`
2. If `portfolio` is provided:
   a. Check portfolio constraints → may reduce position
   b. Evaluate circuit breaker → may reduce further or zero out
3. If `portfolio` is None: skip portfolio constraints and circuit breaker (single-stock mode)
4. Return `RiskAssessment`

---

## Config Additions

No new Pydantic config classes needed. All configs already exist in `config.py`:
- `PositionSizingConfig` (method, max_single_stock_pct, min_confidence)
- `PortfolioConfig` (max_sector_pct, max_correlation, min_holding_days, min_daily_turnover_cr)
- `CircuitBreakerConfig` (level_1/2/3_drawdown, recovery_threshold)
- `RiskConfig` (aggregates all three above)

One addition needed: `CompositeScoreWeights` config for the scorer weights.

```python
class CompositeScoreWeights(BaseModel):
    technical_momentum: float = 0.25
    derivatives_sentiment: float = 0.20
    macro_alignment: float = 0.15
    microstructure_quality: float = 0.15
    news_sentiment: float = 0.10
    volatility_risk: float = 0.15
```

---

## File Map

### New Files

| File | Responsibility |
|------|---------------|
| `alphavedha/prediction/engine.py` | PredictionEngine — main pipeline orchestrator |
| `alphavedha/prediction/scorer.py` | CompositeScorer — 0-100 scoring |
| `alphavedha/prediction/ranker.py` | StockRanker — filter and rank candidates |
| `alphavedha/risk/position_sizing.py` | Half-Kelly position sizing |
| `alphavedha/risk/portfolio.py` | PortfolioConstraints + PortfolioState + HoldingInfo |
| `alphavedha/risk/circuit_breaker.py` | CircuitBreaker + CircuitBreakerState |
| `alphavedha/risk/risk_manager.py` | RiskManager — orchestrates all risk checks |
| `tests/unit/prediction/test_engine.py` | Engine pipeline tests with mocked models |
| `tests/unit/prediction/test_scorer.py` | Composite scoring tests |
| `tests/unit/prediction/test_ranker.py` | Ranking and filtering tests |
| `tests/unit/risk/test_position_sizing.py` | Kelly math tests |
| `tests/unit/risk/test_portfolio.py` | Portfolio constraint tests |
| `tests/unit/risk/test_circuit_breaker.py` | Circuit breaker level tests |
| `tests/unit/risk/test_risk_manager.py` | Risk orchestration tests |

### Modified Files

| File | Change |
|------|--------|
| `alphavedha/prediction/__init__.py` | Export PredictionEngine, CompositeScorer, StockRanker, StockPrediction |
| `alphavedha/risk/__init__.py` | Export RiskManager, RiskAssessment, PortfolioState, etc. |
| `alphavedha/config.py` | Add CompositeScoreWeights |
| `alphavedha/exceptions.py` | Add PredictionError if not already present |
| `alphavedha/prediction/CLAUDE.md` | Update with implementation details |
| `alphavedha/risk/CLAUDE.md` | Update with implementation details |

---

## Testing Strategy

~35-40 tests total across 7 test files.

### Prediction Tests

**test_engine.py** (~10 tests):
- Pipeline produces valid StockPrediction with all fields populated
- Graceful degradation: 1 model failure → warning + valid output
- Graceful degradation: 2 model failures → PredictionError
- All 3 fail → PredictionError
- Regime failure → default regime, warning
- Meta-model failure → default confidence=0, is_tradeable=False
- Conformal failure → default price targets = NaN, warning
- Position size is 0 when is_tradeable is False

**test_scorer.py** (~6 tests):
- Full feature set → score in [0, 100]
- Missing feature group → neutral sub-score, weight redistribution
- All features missing → score = 50 (all neutral)
- Bull regime + buy direction = high macro_alignment
- Bear regime + sell direction = high macro_alignment
- Score weights sum to 1.0

**test_ranker.py** (~5 tests):
- Filters non-tradeable predictions
- Separates buy vs sell candidates
- Sorts by composite_score descending
- Respects top_n limit
- Circuit-hit symbols excluded

### Risk Tests

**test_position_sizing.py** (~6 tests):
- Valid confidence + magnitude → positive position
- Below min_confidence → 0.0
- Negative magnitude → 0.0
- Caps at max_single_stock_pct
- Negative Kelly (low confidence) → 0.0
- Half-Kelly is exactly 0.5 × full Kelly

**test_portfolio.py** (~5 tests):
- Position within all limits → passes unchanged
- Exceeds single stock cap → reduced to cap
- Exceeds sector cap → reduced
- Correlation too high → rejected
- Holding below min period → sell rejected

**test_circuit_breaker.py** (~5 tests):
- Normal (< 10% drawdown) → level 0, no adjustment
- Level 1 (10-15%) → halve positions
- Level 2 (15-20%) → halt new entries
- Level 3 (> 20%) → zero all positions
- Recovery to 95% of peak → reset to level 0

**test_risk_manager.py** (~4 tests):
- Full pipeline: Kelly → constraints → CB → final size
- No portfolio (None) → only Kelly sizing
- CB level 2 blocks new entries
- Low confidence → 0 position (short-circuits risk chain)

---

## Dependencies Between Components

```
PredictionEngine
├── XGBoostModel, LSTMModel, TFTModel (existing)
├── RegimeDetector (existing)
├── StackingEnsemble (existing)
├── MetaLabelingModel (existing)
├── ConformalPredictor (existing)
├── CompositeScorer (new — this week)
└── RiskManager (new — this week)
    ├── position_sizing.compute_position_size()
    ├── PortfolioConstraints
    └── CircuitBreaker
```

All existing model dependencies are injected, not imported. The engine has no knowledge of how models are loaded or trained.
