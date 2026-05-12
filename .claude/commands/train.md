# Train Models

Train or retrain AlphaVedha ML models with proper validation.

## Usage
- `/train all` — train all models end-to-end
- `/train xgboost` — train only XGBoost
- `/train lstm` — train only LSTM
- `/train tft` — train only TFT
- `/train regime` — train HMM regime detector
- `/train meta` — train meta-labeling model

## Steps
1. Activate venv: `source .venv/bin/activate`
2. Verify data freshness: check latest date in feature store
3. Generate labels: `python -m alphavedha.labels.triple_barrier`
4. Train the specified model(s):
   - Use configs from `configs/models.yaml`
   - Log hyperparameters and metrics
   - Save model artifacts with version tag
5. Run CPCV validation on the new model
6. Report metrics: Sharpe ratio (median + worst path), accuracy, MAE
7. If validation passes, deploy as shadow model
8. If validation fails, report why and keep current active model

## CRITICAL
- Never skip validation
- Never overwrite the active model directly
- Always check for data leakage warnings in output

## Arguments
$ARGUMENTS
