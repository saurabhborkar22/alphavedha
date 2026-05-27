# Model Training Guide

End-to-end guide for training AlphaVedha's ML models, from data ingestion to deployment.

## Prerequisites

- PostgreSQL + TimescaleDB running (`make docker-up`)
- Redis running (for feature caching)
- Python environment activated (`source .venv/bin/activate`)
- Environment variables configured (copy `.env.prod.example` to `.env` and fill in values)

Verify the setup:

```bash
alphavedha data status
```

This should show the database connection and row counts. If it fails, check `DATABASE_URL` in your environment.

## Step 1: Data Ingestion

### Backfill Historical Data

For initial setup, backfill from 2005 (or 2020 for a quicker start):

```bash
# Nifty 50 (large cap) — ~80K rows per tier
alphavedha data backfill --tier large --start 2005-01-01

# Midcap 150 — ~200K+ rows
alphavedha data backfill --tier mid --start 2005-01-01
```

### Refresh Recent Data

For daily updates (run before market open):

```bash
# Fetch last 5 days of data (default)
alphavedha data refresh --tier large

# Custom lookback
alphavedha data refresh --tier large --days 10
```

### Additional Data Sources

```bash
# FII/DII institutional flow data
alphavedha data fii-refresh

# F&O derivatives data
alphavedha data derivatives-refresh --tier large

# Earnings results
alphavedha data earnings-refresh --tier large
```

### Verify Data Quality

```bash
alphavedha data status
```

Expected output:
```
Database Status
  OHLCV rows:      80,264
  Symbols:         50
  Latest date:     2026-05-27
  Index members:   200
  FII/DII rows:    1,500
  Derivatives:     25,000
```

**Minimum data requirements:**
- At least 252 trading days (1 year) of OHLCV per symbol for meaningful features
- Recommended: 1,000+ days (4 years) for robust model training

## Step 2: Feature Engineering

Features are computed automatically during training. The pipeline generates 159 features per stock per day across 8 categories:

| Category | Count | Examples |
|----------|-------|---------|
| Technical | ~30 | SMA, RSI, MACD, Bollinger Bands, ATR |
| Returns | ~15 | Log returns, rolling volatility, Sharpe, Sortino |
| Macro | ~15 | India VIX, FII/DII flows, Nifty indices, USD/INR |
| Fundamental | ~10 | P/E, P/B, ROE, Debt/Equity |
| Derivatives | ~15 | OI change, PCR, max pain distance, IV percentile |
| Sentiment | ~10 | News sentiment, volume z-score |
| Calendar | ~10 | Day of week, month, earnings proximity |
| Microstructure | ~10 | Delivery %, volume profile, VWAP deviation |

Features are cached in Redis during market hours (5-minute TTL) and until next market open after hours.

## Step 3: Train Models

### Train All at Once (Recommended)

The `train all` command handles dependency order automatically:

```bash
alphavedha train all --tier large
```

Training order (each step depends on the previous):

1. **XGBoost** — trains on all 159 tabular features
2. **Feature selection** — XGBoost importance selects top 30 features for LSTM/TFT
3. **LSTM** — trains on 60-day sequences of top 30 features
4. **TFT** — trains on sequences with multi-horizon output (7d/15d/30d)
5. **Regime Detector** — HMM on Nifty 50 returns + VIX
6. **Stacking Ensemble** — RidgeClassifier on out-of-fold base model predictions + regime probabilities
7. **Meta-Labeling** — XGBClassifier predicting P(ensemble is correct)
8. **Conformal Predictor** — MAPIE jackknife+ for calibrated price intervals
9. **GNN** — GraphSAGE on stock relationship graph (sector, correlation edges)
10. **RL Agent** — PPO on the trading environment

Expected training time on CPU (Nifty 50, 4 years of data):
- XGBoost: ~2-5 minutes
- LSTM: ~10-20 minutes
- TFT: ~15-30 minutes
- Regime: ~1 minute
- Ensemble + Meta: ~2 minutes
- Conformal: ~1 minute
- GNN: ~5-10 minutes
- RL: ~10-20 minutes
- **Total: ~45-90 minutes** (CPU)

### Train Individual Models

```bash
# Base models
alphavedha train xgboost --tier large
alphavedha train lstm --tier large
alphavedha train tft --tier large
alphavedha train regime --tier large
```

**Dependency chain:** LSTM and TFT require XGBoost to be trained first (for feature selection). Ensemble requires all three base models + regime. Meta-labeling requires ensemble.

### Training on Google Colab (GPU)

For GPU-accelerated training, use the Colab notebook:

1. Open `notebooks/colab_training.ipynb` in Google Colab
2. Set your GitHub PAT in Colab secrets (for private repo access)
3. Connect to a GPU runtime (T4 is sufficient)
4. Run all cells — the notebook clones the repo, installs dependencies, and trains all models
5. Download the resulting `models/artifacts/` directory

## Step 4: Model Artifacts

All model artifacts are saved to `models/artifacts/{model_name}/latest/`:

```
models/artifacts/
├── xgboost/latest/
│   ├── metadata.json       # Training params, metrics, data range
│   └── model.joblib         # Serialized XGBClassifier + XGBRegressor
├── lstm/latest/
│   ├── metadata.json
│   ├── model.safetensors    # PyTorch weights
│   └── model_config.json    # Architecture params
├── tft/latest/
│   ├── metadata.json
│   ├── model.safetensors
│   └── model_config.json
├── regime/latest/
│   ├── metadata.json
│   └── model.joblib         # GaussianHMM
├── ensemble/latest/
│   ├── metadata.json
│   └── model.joblib         # RidgeClassifier
├── meta_labeling/latest/
│   ├── metadata.json
│   └── model.joblib         # XGBClassifier
├── conformal/latest/
│   ├── metadata.json
│   └── model.joblib         # MAPIE model + calibration data
└── experiments/
    └── *.json               # One file per training run (for experiment tracking)
```

**Never commit model artifacts to git.** The `models/` directory is in `.gitignore`.

## Step 5: Validate

### Compare Experiments

```bash
# List all training runs
alphavedha experiment list

# Compare two runs side by side
alphavedha experiment compare <run_id_1> <run_id_2>
```

### Run Walk-Forward Backtest

```bash
alphavedha backtest walk-forward --tier large
```

This runs a time-series cross-validation with purge and embargo:
- Expanding training window (minimum 252 days)
- 20-day embargo between train and test sets
- Reports: Sharpe ratio, max drawdown, win rate, return vs Nifty 50

### Target Metrics

| Metric | Target | Action if Below |
|--------|--------|-----------------|
| Walk-forward Sharpe | > 1.5 | Check feature quality, increase training data |
| Max drawdown | < 12% | Tighten position sizing, review risk parameters |
| Win rate | > 55% | Review labeling parameters, check for data issues |
| Profit factor | > 1.8 | Check cost assumptions, review magnitude predictions |
| Meta-label filter rate | 30-40% | Adjust threshold (default 0.55) |

## Step 6: Deploy

### Start the Prediction API

```bash
# Development (with auto-reload)
alphavedha serve

# Production (with gunicorn)
make serve-prod
```

The server loads all model artifacts on startup and runs a warm-up prediction to verify the full pipeline works.

### Verify Predictions

```bash
# Single prediction
alphavedha predict TCS

# Scan top picks
alphavedha scan large --top-n 5
```

### Start the Scheduler

```bash
# Background job scheduler
alphavedha scheduler start
```

Scheduled jobs:
- **Daily 8:30 AM IST** — pre-market predictions for all stocks
- **Daily 3:45 PM IST** — evaluate yesterday's predictions against actual outcomes
- **Weekly (Saturday 8 PM)** — drift detection + performance evaluation
- **Monthly (1st Saturday 10 PM)** — model retraining (if triggered by drift/performance)
- **Quarterly (March/September)** — Nifty index rebalancing check

## Troubleshooting

### "No data available" during training

The database is empty or has insufficient data for the requested tier.

```bash
alphavedha data status     # Check what's in the DB
alphavedha data backfill --tier large --start 2020-01-01
```

### LSTM/TFT training fails with "XGBoost model not found"

Train XGBoost first — it provides feature selection for the sequence models.

```bash
alphavedha train xgboost --tier large
alphavedha train lstm --tier large
```

### Out of memory during training

Reduce batch size in `configs/config.yaml`:

```yaml
models:
  lstm:
    batch_size: 32    # Default: 64
    max_epochs: 20    # Default: 50
  tft:
    batch_size: 32
    max_epochs: 20
```

Or use Google Colab with a GPU runtime for larger datasets.

### Model accuracy below target

1. Check data freshness: `alphavedha data status`
2. Check for feature drift: `alphavedha scheduler run-now weekly_drift_check`
3. Try retraining with more data: extend the backfill start date
4. Review experiment history: `alphavedha experiment list`

### Predictions return None

The model artifacts are missing or corrupted. Check:

```bash
ls models/artifacts/xgboost/latest/
ls models/artifacts/ensemble/latest/
```

If empty, retrain. If files exist but predictions fail, check the logs:

```bash
tail -100 logs/alphavedha.log | grep ERROR
```
