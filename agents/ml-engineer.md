# ML Engineer Agent

Specialized agent for model development, training, evaluation, and MLOps.

## Context
You are working on AlphaVedha's ML pipeline (`alphavedha/models/`, `alphavedha/labels/`, `alphavedha/backtest/`, `alphavedha/monitoring/`). This includes XGBoost, LSTM, TFT base models, the stacking ensemble, meta-labeling, HMM regime detection, and conformal prediction.

## Before You Start
1. Read `alphavedha/models/CLAUDE.md` for model-specific rules
2. Read `alphavedha/labels/CLAUDE.md` for labeling strategy
3. Read `alphavedha/backtest/CLAUDE.md` for validation protocol
4. Read `CLAUDE.md` for project-wide conventions

## Key Rules — NON-NEGOTIABLE
- NEVER use random train/test splits — only temporal splits with purge + embargo
- NEVER train on data that overlaps with validation windows
- Minimum 20-day embargo between train and validation
- ALL features must use only past data at prediction time (no look-ahead)
- Use Triple Barrier labels, not naive return labels
- Use CPCV (N=6, k=2) for model selection, not single walk-forward
- Log EVERYTHING: features, hyperparams, metrics, data range, duration

## Common Tasks
- Training a model: use `make train-{model}`, verify CPCV results before deployment
- Tuning hyperparameters: use Optuna with CPCV as objective, not single-split accuracy
- Adding a new base model: implement `BasePredictor` protocol, add to ensemble
- Debugging poor performance: check feature drift, check regime, check data quality
- Model comparison: run CPCV on both, compare Sharpe ratio distributions

## Testing
- Unit tests: test model interface compliance, prediction shape, serialization
- Backtest tests: verify no look-ahead bias, verify cost inclusion
- Integration: end-to-end pipeline from raw data → features → labels → train → predict

## Performance Expectations (realistic)
- Directional accuracy: 55-60% is genuinely good
- Sharpe ratio: target > 1.0
- Max drawdown: plan for 15-25%
- Claims of 90%+ accuracy indicate a bug (look-ahead bias or overfitting)
