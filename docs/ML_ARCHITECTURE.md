# AlphaVedha — ML Architecture Reference

## Model Zoo

| Model | Class | Purpose | Input | Output | Serialization | Trains After |
|---|---|---|---|---|---|---|
| XGBoostModel | `XGBoostModel` | Tabular direction (3-class) + magnitude regression | 148 non-stub features | direction(-1/0/1), magnitude, proba[3], confidence | joblib: classifier.joblib + regressor.joblib | Nothing (trains first) |
| LSTMModel | `LSTMModel` | Temporal sequence direction + magnitude | Top-30 features by XGB importance, 60-day windows | direction, magnitude, proba[3], confidence | safetensors: model.safetensors + model_config.json | XGBoost |
| TemporalAttentionModel (TFT) | `TemporalAttentionModel` | Multi-horizon direction + magnitude + attention weights | Same top-30 features as LSTM, 60-day sequences | per-horizon predictions (7d/15d/30d), attention, VSN feature weights | safetensors: model.safetensors + model_config.json | XGBoost |
| GNNModel | `GNNModel` | Graph-based stock relationship modeling | Full features + StockGraph edge_index | direction, magnitude, proba[3], confidence | safetensors: model.safetensors + model_config.json | Nothing (optional 4th learner) |
| RegimeDetector | `RegimeDetector` | Classify market regime via Gaussian HMM | 2 features: portfolio log returns + 20d realized vol | current_regime(str), regime_id, state_proba[4], transition_matrix | joblib: hmm_model.joblib + metadata.json | Nothing (trains independently) |
| StackingEnsemble | `StackingEnsemble` | Combine base model OOF probabilities via Ridge | 14 meta-features (XGB+LSTM+TFT proba×3 each + regime proba×4 + disagreement) | direction, magnitude, proba[3], confidence, model_disagreement | joblib: ridge_model.joblib + metadata.json | XGBoost + LSTM + TFT + Regime |
| MetaLabelingModel | `MetaLabelingModel` | Binary gate: P(ensemble prediction correct) | Full features + ensemble_direction + ensemble_confidence | meta_confidence, is_tradeable (> 0.55) | joblib: meta_classifier.joblib + metadata.json | Ensemble |
| ConformalPredictor | `ConformalPredictor` | Prediction intervals for magnitude, 90% coverage | Feature DataFrame → target = return_pct | price_low, price_mid, price_high (in return% units), interval_width | joblib: mapie_model.joblib + metadata.json | Nothing (uses OOF data only) |
| PPOAgent | `PPOAgent` | Portfolio position sizing via PPO RL | obs=[features+positions+portfolio_value+regime_one_hot+drawdown] | action = position weights per stock [-1,1] | torch.save: ppo_weights.pt + ppo_config.json | TradingEnvironment |

---

## Artifact Directories

All artifacts under `models/artifacts/` (shared Docker volume between api, scheduler, trainer):

```
models/artifacts/
├── xgboost/latest/         classifier.joblib, regressor.joblib, feature_importance.csv, metadata.json
├── lstm/latest/            model.safetensors, model_config.json, metadata.json
├── tft/latest/             model.safetensors, model_config.json, metadata.json
├── gnn/latest/             model.safetensors, model_config.json, metadata.json
├── regime/latest/          hmm_model.joblib, metadata.json
├── ensemble/latest/        ridge_model.joblib, metadata.json
├── meta_labeling/latest/   meta_classifier.joblib, metadata.json
├── conformal/latest/       mapie_model.joblib, metadata.json
├── rl_ppo/                 ppo_weights.pt, ppo_config.json
├── version.json            {version: "ensemble_v1.2", ...}
└── runs/                   experiment tracker JSON per run
```

Models are ready when `{name}/latest/metadata.json` (or `{name}/metadata.json`) exists. `ModelRegistry.models_available()` checks this without loading weights.

---

## Training Pipeline (train_all in training/pipeline.py)

### Step 1 — Load Tier Data
- Fetch symbols for tier (e.g., "large" = Nifty 50) from `index_constituents`
- Historical range: 2020-01-01 to today (or `end_date` for frozen historical simulation)
- Load FII/DII from `institutional_flows` DB, macro from yfinance (VIX, Nifty, USDINR, Brent, Gold, US10Y)
- Per symbol: skip if < 252 rows; `compute_all_features()` → 164 cols; `compute_triple_barrier_labels()` → labels; skip if < 100 valid labeled rows
- **3-way temporal split** (20-day embargo gaps between boundaries):
  - train: first 70% of rows
  - OOF: next 15% (starts after train_end + 20 rows purge)
  - val: final 15% (starts after OOF_end + 20 rows purge)
- Pool all symbols cross-sectionally; drop 16 stub features

### Step 2 — Train XGBoost
- XGBClassifier (3-class) + XGBRegressor (magnitude), early_stopping_rounds=50
- Saves `feature_importance.csv` — critical for next two models
- Artifacts: `xgboost/latest/`

### Step 3 — Select Top-N Features
- Read feature_importance.csv, take top 30 by importance score
- Used identically by LSTM and TFT

### Step 4 — Train LSTM
- `FastSequenceLoader` — 60-day sliding windows via torch.unfold (no-copy, pre-materialized)
- Adam + ReduceLROnPlateau(factor=0.5, patience=3) + EarlyStopping(patience=10)
- Inverse-frequency class weights (handles neutral class ~15% imbalance)
- NaN → 0 imputation on feature subset

### Step 5 — Train TFT
- `FastMultiHorizonLoader` — multi-horizon batches with per-horizon labels + validity masks
- 3 prediction heads (7d/15d/30d); loss weights [0.5, 0.3, 0.2]
- VSN selection weights provide feature importance readout

### Step 6 — Train Regime Detector
- Aggregate portfolio: mean log returns across all symbols per day + 20-day realized volatility
- `GaussianHMM(n_components=4, covariance_type="full", n_iter=100)`
- State assignment post-fit: bull=max-mean, bear=min-mean, high_vol=max-variance, sideways=remainder

### Step 6.5 — Train GNN (optional)
- Full feature set (no selection), NaN → 0
- 2-layer GraphSAGE; fallback to MLP if no graph structure passed
- Graph built via `StockGraph` (SECTOR + CORRELATION + PROMOTER edges)

### Step 7 — Train Stacking Ensemble
- Loads XGBoost + LSTM + TFT from disk
- **Runs OOF data through all 3 base models** (never train-set predictions)
- Gets regime_probs from RegimeDetector on OOF portfolio returns
- Builds 14-column meta-feature matrix; fits `RidgeClassifier(alpha=1.0)`

### Step 8 — Train Meta-Labeling
- Loads Ensemble; runs OOF data through ensemble
- `y_correct = (ensemble_direction == y_oof)` — binary: is the ensemble right?
- Augments OOF features with `ensemble_direction` + `ensemble_confidence`
- Fits `XGBClassifier(binary:logistic, max_depth=4, n_estimators=200, early_stopping=20)`

### Step 9 — Train Conformal Predictor
- OOF data capped at 5000 rows (CX23 RAM constraint — takes most recent)
- `CrossConformalRegressor(XGBRegressor(n_estimators=100), coverage=0.90, method="plus")`
- Jackknife+ → valid marginal coverage without distributional assumptions

### Step 10 — Train RL Agent
- Builds TradingEnvironment (initial_capital=1M, cost=0.3%, max_position=10%)
- `PPOAgent` — 50 episodes, walk-forward (80% train / 20% val)

---

## Label Generation (triple_barrier.py)

**Parameters:** atr_period=14, multiplier_up=2.0, multiplier_down=1.5, max_holding_period=15d, min_atr_threshold=0.005

**Algorithm per row t:**
1. Compute ATR14 using only past data
2. Skip if ATR NaN/zero or ATR/close < 0.5%
3. Skip if t + 15 >= n (insufficient future bars)
4. upper = close[t] + 2.0 × ATR
5. lower = close[t] - 1.5 × ATR
6. Scan forward d=1..15:
   - Both upper AND lower breached same day → label = -1 (conservative)
   - Only upper breached → label = +1, break
   - Only lower breached → label = -1, break
7. No breach after 15 days → label = 0 (time barrier)

**Output:** label (-1/0/1), return_pct, barrier_hit, days_to_hit, entry_price, exit_price, atr_at_entry

**Semantics:** +1 = long signal (take-profit hit), -1 = avoid/short (stop-loss hit or simultaneous breach), 0 = neutral

---

## Sample Weights (sample_weights.py)

Two components multiplied together, normalized so mean weight = 1.0:

**Uniqueness weights** — reduces weight of overlapping labels:
- `concurrency[i]` = count of samples whose barrier windows overlap position i
- `weight[i] = mean(1/concurrency[j] for j in sample i's active window)`
- Active window = [i, i + days_to_hit[i])

**Recency weights** — decays older samples:
- `decay[i] = exp(-log(2) × (last_pos - pos[i]) / halflife)`
- halflife = 252 trading days; newest sample = 1.0, 252-day-old sample = 0.5

---

## Ensemble Logic

**Meta-feature matrix columns** (14 total, 17 with GNN):
```
cols 0-2:   XGBoost proba [P(down), P(neutral), P(up)]
cols 3-5:   LSTM proba
cols 6-8:   TFT proba
cols 9-12:  RegimeDetector state proba [P(bull), P(bear), P(sideways), P(high_vol)]
col 13:     Disagreement score = std of base models' consensus-class probabilities
```

**Disagreement computation:**
1. Stack all base model proba arrays: shape (n_models, n_samples, 3)
2. Mean proba → consensus class = argmax
3. Std of each model's consensus-class prob = disagreement

**Magnitude aggregation:** confidence-weighted mean of base model magnitudes. Zero-sum guard: uniform weights when all confidences are zero.

---

## Meta-Labeling Logic

**Purpose:** Answer "should I trade this signal?" not "which direction?"

**Input:** Full feature DataFrame + `ensemble_direction` + `ensemble_confidence` (appended columns)

**Training target:** `y_correct = (ensemble_direction == y_oof)` — binary

**Output:** `meta_confidence` = P(ensemble correct), `is_tradeable = meta_confidence > 0.55`

**Effect:** Only is_tradeable=True signals surface to API. Regime-dependent threshold overrides 0.55:
- bull: 0.40, bear: 0.45, sideways: 0.42, high_volatility: 0.52

---

## Conformal Prediction

**What:** Guaranteed 90% marginal coverage interval for return magnitude. Despite naming (price_low/price_mid/price_high), these are **return percentages**. The API converts to price levels using `last_close * (1.0 + interval_value)`.

**Method:** CrossConformalRegressor (jackknife+) — valid without distributional assumptions on OOF residuals.

**Recalibration:** `calibrate(X_cal, y_cal)` updates conformity score distribution without retraining base. Used after model promotion to adjust for regime drift.

**Interpretation:** Interval width signals uncertainty — wide interval = high magnitude uncertainty even if direction is clear.

---

## Prediction Engine Flow (engine.py — per-request)

```
1. RegimeDetector.predict(portfolio_returns, portfolio_vol)
   → regime_name, regime_probs[4]

2. RegimeStrategySelector.select(regime_name)
   → kelly_fraction, meta_confidence_threshold, require_all_models_agree

3. Base models inference (XGBoost + LSTM + TFT + optional GNN)
   → base_predictions per model; need >= 2 successful

4. [if high_vol] Check unanimous direction before ensemble

5. StackingEnsemble.predict(base_predictions, regime_probs)
   → direction, magnitude, confidence, model_disagreement

6. MetaLabelingModel.predict(features, ensemble_direction, ensemble_confidence)
   → meta_confidence, is_tradeable

7. Apply regime threshold gate: meta_confidence < threshold → not tradeable

8. ConformalPredictor.predict(features) → price_low, price_mid, price_high

9. CompositeScorer.score(ensemble, regime, features) → composite_score [0-100]

10. RiskManager.assess(meta_confidence, magnitude) → position_size_pct (half Kelly)

11. RegimeOverlay.apply() — optional (ALPHAVEDHA_REGIME_OVERLAY=1 only)

12. _compute_atr_levels() → entry, stop_loss, take_profit prices

13. Return StockPrediction with all fields + warnings[]
```

---

## Regime Strategy Parameters

| Regime | Kelly Fraction | Meta Threshold | Require All Agree |
|---|---|---|---|
| bull | 1.0 | 0.40 | No |
| bear | 0.25 | 0.45 | No |
| sideways | 0.50 | 0.42 | No |
| high_volatility | 0.10 | 0.52 | Yes |

---

## Composite Score Weights

| Component | Weight | Source |
|---|---|---|
| technical_momentum | 0.25 | ensemble confidence × 100 |
| derivatives_sentiment | 0.20 | mean sigmoid of deriv_* columns |
| macro_alignment | 0.15 | regime×direction alignment lookup (0/30/60/100) |
| microstructure_quality | 0.15 | mean sigmoid of micro_* columns |
| volatility_risk (inverted) | 0.15 | mean sigmoid of hvol/natr/atr/bb_width columns |
| news_sentiment | 0.10 | mean sigmoid of sent_* columns |

Missing feature groups: weight redistributed proportionally. Output clipped [0, 100].

---

## Model Hyperparameters (from default.yaml)

### XGBoost
`lr=0.05, max_depth=6, n_estimators=500, subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0, eval_metric=logloss, early_stopping=50`

### LSTM
`sequence_length=60, hidden_size=64, num_layers=2, dropout=0.5, lr=0.001, weight_decay=0.0001, label_smoothing=0.1, batch_size=128, max_epochs=60, early_stopping_patience=10, top_n_features=30`

### TFT (TemporalAttention)
`hidden_size=32, attention_head_size=4, dropout=0.35, lr=0.001, weight_decay=0.0001, label_smoothing=0.1, max_epochs=50, horizons=[7,15,30], horizon_loss_weights={0:0.5, 1:0.3, 2:0.2}`

### Regime (HMM)
`n_states=4, covariance_type="full", n_iter=100, state_names=["bull","bear","sideways","high_volatility"]`

### Conformal
`coverage=0.90, calibration_window=60, base=XGBRegressor(n_estimators=100)`

---

## Stub Features Dropped at Training (16 features)

These are hardcoded zeros/constants in the feature pipeline and excluded by `_STUB_FEATURES` frozenset in `training/pipeline.py` before any model sees data:

`macro_gsec_10y, macro_gsec_change_1d, macro_pmi, macro_pmi_staleness_days, macro_breadth_200sma, macro_adv_dec_ratio, macro_index_cpr, macro_mktcap_flow, deriv_fii_futures_oi, deriv_fii_options_oi, deriv_pro_futures_net, deriv_retail_futures_net, deriv_gex, deriv_delta_oi, ret_regime, trends_sector_7d, trends_sector_change`

**Total features at training time: 148** (164 declared - 16 stubs).
