# Models — AlphaVedha

## Responsibility
All ML models: base learners (XGBoost, LSTM, TFT), meta-learner (stacking ensemble), meta-labeling model, regime detector (HMM), and conformal prediction.

## Model Interface

Every model MUST implement:

```python
class BasePredictor(Protocol):
    def train(self, X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.Series) -> TrainResult: ...
    def predict(self, X: pd.DataFrame) -> PredictionResult: ...
    def save(self, path: Path) -> None: ...
    def load(self, path: Path) -> None: ...
    def get_feature_importance(self) -> dict[str, float]: ...
```

## Model-Specific Rules

### xgboost_model.py
- Input: all 141 tabular features (flattened, no sequence)
- Use `xgboost.XGBClassifier` for direction, `XGBRegressor` for magnitude
- Key hyperparameters: learning_rate=0.05, max_depth=6, n_estimators=500, subsample=0.8
- Enable GPU training if available (`tree_method='gpu_hist'`)
- Use built-in feature importance for monitoring

### lstm_model.py (PyTorch)
- Input: 60-day sequence of top-30 features (by XGBoost importance)
- Architecture: 2 LSTM layers, 128 hidden units, dropout=0.3
- Training: Adam optimizer, lr=1e-3, batch_size=64, early stopping patience=10
- Use `torch.nn.utils.clip_grad_norm_` (max_norm=1.0)
- Save with `torch.save(model.state_dict(), path)` — never save full model object

### tft_model.py (PyTorch)
- Use `pytorch-forecasting` TFT implementation OR custom
- Static covariates: sector, market_cap_tier, index_membership
- Known future inputs: calendar features (day_of_week, month, expiry_proximity)
- Unknown future inputs: all price-derived and market features
- Multi-horizon output: 7d, 15d, 30d simultaneously
- Attention weights → save for interpretability dashboard

### ensemble.py (Stacking)
- Meta-learner input: [xgboost_pred, lstm_pred, tft_pred, regime_probs, model_disagreement]
- model_disagreement = std(base_predictions) — high disagreement = low confidence
- Use Ridge regression as default meta-learner (prevents overfitting on small meta-dataset)
- Train meta-learner on OUT-OF-FOLD predictions from base models (never on training predictions)

### meta_model.py (Meta-Labeling)
- Input: primary prediction + all original features
- Output: P(primary prediction is correct)
- Use XGBoost classifier
- Threshold: only output predictions where meta_confidence > 0.55
- This model answers "should I bet on this signal?" not "what direction?"

### regime.py (HMM)
- Trained on: Nifty 50 log returns + India VIX
- 4 states: bull, bear, sideways, high_volatility
- Use `hmmlearn.GaussianHMM`
- Retrain monthly (regime dynamics are slow-moving)
- Output: current regime label + probability per state

### conformal.py
- Use MAPIE `MapieRegressor` with conformalized quantile regression
- Calibration set: most recent 60 trading days (rolling)
- Coverage target: 90%
- Output: [price_low, price_mid, price_high]
- Interval width MUST expand in high-volatility regimes (verify this)

## Training Rules
- NEVER train and evaluate on overlapping data
- Use CPCV (see backtest/CLAUDE.md) for model selection
- Log everything: features, hyperparameters, metrics, duration, data range
- Model artifacts stored in `models/artifacts/{model_name}/{version}/`
- Version format: `v{major}.{minor}.{patch}` — bump minor on retrain, major on architecture change

## Serialization
- XGBoost: `joblib.dump()` / `joblib.load()`
- PyTorch (LSTM, TFT): `torch.save(model.state_dict())` — save state_dict only, not full model
- HMM: `joblib.dump()`
- Meta-learner (Ridge): `joblib.dump()`
- All artifacts include a `metadata.json` with training details
