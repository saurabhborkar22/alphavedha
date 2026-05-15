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

### sequence_utils.py (Shared)
- `SequenceDataset` — sliding-window Dataset for single-horizon models (LSTM)
- `MultiHorizonSequenceDataset` — sliding-window Dataset with per-horizon labels and validity masks (TFT)
- `EarlyStopping` — patience-based training stopper
- `compute_combined_loss` — weighted CrossEntropy (0.7) + MSE (0.3) with per-sample weights
- `create_data_loaders` — convenience factory for train/val DataLoader pairs
- `get_device` — auto-detect CUDA > MPS > CPU
- Label mapping: {-1: 0, 0: 1, 1: 2} for 3-class classification

### lstm_model.py (PyTorch)
- `LSTMNetwork(nn.Module)` with dual heads: classification (3-class) + regression (magnitude)
- Input: 60-day sequence of top-30 features (by XGBoost importance)
- Architecture: 2 LSTM layers, 128 hidden units, dropout=0.3
- Training: Adam optimizer, lr=1e-3, batch_size=64, early stopping patience=10
- Use `torch.nn.utils.clip_grad_norm_` (max_norm=1.0)
- Warmup padding: first `seq_len-1` predictions are neutral (direction=0, confidence=0)
- Save with `safetensors` — never save full model object
- `get_feature_importance()` returns None (LSTM has no intrinsic feature importance)

### temporal_attention.py (PyTorch — TFT-lite)
- Custom TFT-lite architecture (not pytorch-forecasting)
- Components: `GatedResidualNetwork`, `VariableSelectionNetwork`, `InterpretableMultiHeadAttention`, `TemporalAttentionNetwork`
- Architecture: VSN → LSTM encoder → multi-head attention → per-horizon heads
- Multi-horizon output: 7d, 15d, 30d simultaneously via `get_horizon_predictions()`
- Feature importance from VSN selection weights (softmax-normalized, sums to 1.0)
- Attention weights available via `get_attention_weights()` for interpretability
- Horizon loss weights: {7d: 0.5, 15d: 0.3, 30d: 0.2}
- Save with `safetensors`

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
- PyTorch (LSTM, TFT): `safetensors` (`save_file` / `load_file`) + `model_config.json` for architecture params
- HMM: `joblib.dump()`
- Meta-learner (Ridge): `joblib.dump()`
- All artifacts include a `metadata.json` with training details
