# MLOps & Monitoring — AlphaVedha

## Responsibility
Track model health, detect drift, manage model versions, and trigger retraining.

## Modules

### drift.py — Feature Drift Detection
- Compute PSI (Population Stability Index) per feature group daily
- Alert threshold: PSI > 0.2 (significant drift)
- Warning threshold: PSI > 0.1 (moderate drift)
- KS test as secondary check for distribution shifts
- Log drift scores to database for historical tracking

### performance.py — Prediction Accuracy Tracking
- Track rolling accuracy: 7d, 30d, 90d windows
- Track directional accuracy, magnitude error (MAE), and calibration
- Compare live predictions vs actual outcomes (after the prediction horizon passes)
- Alert if rolling accuracy drops below 52% (barely above random)
- Track alpha vs Nifty 50 buy-and-hold (the benchmark that matters)

### versioning.py — Model Registry
- Each model version stored with metadata:
  - version string, training date, data range used
  - feature set hash, hyperparameters
  - validation metrics (CPCV results)
  - status: "shadow" | "active" | "retired"
- Shadow deployment: new models run in parallel without affecting output
- Promote shadow → active only after 20 trading days of shadow performance

### retrainer.py — Automated Retraining
Triggers:
1. **Calendar**: weekly (Saturday night) for base models, monthly for HMM
2. **Drift**: any feature group PSI > 0.2
3. **Performance**: rolling 30d accuracy < 53%
4. **Manual**: CLI command `alphavedha retrain --model xgboost`

Process:
1. Fetch latest data
2. Recompute features
3. Generate new labels (triple barrier)
4. Train new model version
5. Run CPCV validation
6. If validation passes → deploy as shadow
7. After 20d shadow period → promote or discard

## Rules
- Never auto-promote a model without shadow period
- Keep last 5 model versions (for rollback)
- All monitoring metrics exposed via `/metrics` API endpoint
- Structured logging with structlog — every event is JSON
