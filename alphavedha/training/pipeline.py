"""Training pipeline — end-to-end: load data → features → labels → train → save.

Orchestrates the full training workflow for a given model type and universe tier.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import structlog

from alphavedha.config import get_config
from alphavedha.data.store import load_ohlcv
from alphavedha.data.universe import get_symbols_for_tier
from alphavedha.features.pipeline import compute_all_features
from alphavedha.labels.triple_barrier import compute_triple_barrier_labels
from alphavedha.models.base import PredictionResult, TrainResult
from alphavedha.models.xgboost_model import XGBoostModel

logger = structlog.get_logger(__name__)

ARTIFACTS_DIR = Path("models/artifacts")


@dataclass
class TrainingPipelineResult:
    model_name: str
    artifact_path: Path | None = None
    train_result: TrainResult | None = None
    n_symbols: int = 0
    n_train_rows: int = 0
    n_val_rows: int = 0
    total_time_seconds: float = 0.0
    errors: dict[str, str] = field(default_factory=dict)
    extra_metrics: dict[str, float] = field(default_factory=dict)


@dataclass
class TierData:
    """Holds prepared train/oof/val splits for a tier."""

    X_train: pd.DataFrame
    y_train: pd.Series
    ret_train: pd.Series
    X_oof: pd.DataFrame
    y_oof: pd.Series
    ret_oof: pd.Series
    X_val: pd.DataFrame
    y_val: pd.Series
    ret_val: pd.Series
    ohlcv_by_symbol: dict[str, pd.DataFrame]
    n_symbols: int
    errors: dict[str, str]


def _prepare_symbol_data(
    symbol: str,
    ohlcv_df: pd.DataFrame,
    fii_dii_df: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.Series, pd.Series] | None:
    """Compute features and labels for a single symbol. Returns (X, y_direction, y_return) or None."""
    config = get_config()

    if len(ohlcv_df) < 252:
        logger.warning("train_skip_short", symbol=symbol, rows=len(ohlcv_df))
        return None

    feature_result = compute_all_features(
        symbol=symbol, ohlcv_df=ohlcv_df, fii_dii_df=fii_dii_df,
    )
    features_df = feature_result.df

    label_result = compute_triple_barrier_labels(
        ohlcv_df, config.labels.triple_barrier, symbol=symbol,
    )
    labels_df = label_result.df

    valid_mask = labels_df["label"].notna()
    features_df = features_df.loc[valid_mask]
    labels_df = labels_df.loc[valid_mask]

    if len(features_df) < 100:
        logger.warning("train_skip_few_labels", symbol=symbol, valid=len(features_df))
        return None

    return features_df, labels_df["label"].astype(int), labels_df["return_pct"]


def _temporal_split(
    X: pd.DataFrame,
    y: pd.Series,
    returns: pd.Series,
    val_ratio: float = 0.2,
    embargo_days: int = 20,
) -> tuple[pd.DataFrame, pd.Series, pd.Series, pd.DataFrame, pd.Series, pd.Series]:
    """Split into train/val with temporal ordering and embargo gap."""
    n = len(X)
    split_idx = int(n * (1 - val_ratio))
    train_end = split_idx - embargo_days

    if train_end < 100:
        train_end = split_idx

    X_train = X.iloc[:train_end]
    y_train = y.iloc[:train_end]
    ret_train = returns.iloc[:train_end]

    X_val = X.iloc[split_idx:]
    y_val = y.iloc[split_idx:]
    ret_val = returns.iloc[split_idx:]

    return X_train, y_train, ret_train, X_val, y_val, ret_val


def _temporal_split_3way(
    X: pd.DataFrame,
    y: pd.Series,
    returns: pd.Series,
    oof_ratio: float = 0.15,
    val_ratio: float = 0.15,
    embargo_days: int = 20,
) -> tuple[
    pd.DataFrame, pd.Series, pd.Series,
    pd.DataFrame, pd.Series, pd.Series,
    pd.DataFrame, pd.Series, pd.Series,
]:
    """Split into train/oof/val with temporal ordering and embargo gaps.

    OOF (out-of-fold) portion is used for training the ensemble meta-learner.
    Val portion is the final holdout.
    """
    n = len(X)
    train_end_raw = int(n * (1 - oof_ratio - val_ratio))
    oof_start = train_end_raw + embargo_days
    oof_end_raw = int(n * (1 - val_ratio))
    val_start = oof_end_raw + embargo_days

    train_end = max(train_end_raw, 100)
    oof_start = min(oof_start, oof_end_raw)
    val_start = min(val_start, n)

    return (
        X.iloc[:train_end], y.iloc[:train_end], returns.iloc[:train_end],
        X.iloc[oof_start:oof_end_raw], y.iloc[oof_start:oof_end_raw], returns.iloc[oof_start:oof_end_raw],
        X.iloc[val_start:], y.iloc[val_start:], returns.iloc[val_start:],
    )


async def _load_tier_data(
    tier: str = "large",
    oof_ratio: float = 0.15,
    val_ratio: float = 0.15,
    embargo_days: int = 20,
) -> TierData:
    """Load all symbol data for a tier with 3-way temporal split."""
    symbols = await get_symbols_for_tier(tier)
    if not symbols:
        logger.error("train_no_symbols", tier=tier)
        return TierData(
            X_train=pd.DataFrame(), y_train=pd.Series(dtype=float),
            ret_train=pd.Series(dtype=float),
            X_oof=pd.DataFrame(), y_oof=pd.Series(dtype=float),
            ret_oof=pd.Series(dtype=float),
            X_val=pd.DataFrame(), y_val=pd.Series(dtype=float),
            ret_val=pd.Series(dtype=float),
            ohlcv_by_symbol={}, n_symbols=0, errors={},
        )

    logger.info("train_loading_data", tier=tier, n_symbols=len(symbols))

    all_train: list[tuple[pd.DataFrame, pd.Series, pd.Series]] = []
    all_oof: list[tuple[pd.DataFrame, pd.Series, pd.Series]] = []
    all_val: list[tuple[pd.DataFrame, pd.Series, pd.Series]] = []
    ohlcv_by_symbol: dict[str, pd.DataFrame] = {}
    errors: dict[str, str] = {}
    n_symbols = 0

    start_date = date(2020, 1, 1)
    end_date = date.today()

    fii_dii_df: pd.DataFrame | None = None
    try:
        from alphavedha.features.macro import load_fii_dii_for_features
        fii_dii_df = await load_fii_dii_for_features(str(start_date), str(end_date))
        if fii_dii_df is not None and not fii_dii_df.empty:
            logger.info("train_fii_dii_loaded", rows=len(fii_dii_df))
    except Exception as e:
        logger.warning("train_fii_dii_load_failed", error=str(e))

    for symbol in symbols:
        try:
            ohlcv_df = await load_ohlcv(symbol, start_date, end_date)
            if ohlcv_df.empty:
                errors[symbol] = "no data in DB"
                continue

            ohlcv_by_symbol[symbol] = ohlcv_df
            prepared = _prepare_symbol_data(symbol, ohlcv_df, fii_dii_df)
            if prepared is None:
                continue

            features_df, y_dir, y_ret = prepared

            X_tr, y_tr, ret_tr, X_o, y_o, ret_o, X_v, y_v, ret_v = _temporal_split_3way(
                features_df, y_dir, y_ret, oof_ratio, val_ratio, embargo_days,
            )

            if len(X_tr) > 0:
                all_train.append((X_tr, y_tr, ret_tr))
            if len(X_o) > 0:
                all_oof.append((X_o, y_o, ret_o))
            if len(X_v) > 0:
                all_val.append((X_v, y_v, ret_v))
            n_symbols += 1

            logger.info(
                "train_symbol_prepared",
                symbol=symbol,
                train=len(X_tr), oof=len(X_o), val=len(X_v),
            )
        except Exception as e:
            errors[symbol] = str(e)
            logger.error("train_symbol_error", symbol=symbol, error=str(e))

    def _concat(parts: list[tuple[pd.DataFrame, pd.Series, pd.Series]]) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
        if not parts:
            return pd.DataFrame(), pd.Series(dtype=float), pd.Series(dtype=float)
        Xs, ys, rs = zip(*parts, strict=True)
        return (
            pd.concat(Xs, ignore_index=True),
            pd.concat(ys, ignore_index=True),
            pd.concat(rs, ignore_index=True),
        )

    X_train, y_train, ret_train = _concat(all_train)
    X_oof, y_oof, ret_oof = _concat(all_oof)
    X_val, y_val, ret_val = _concat(all_val)

    logger.info(
        "train_data_ready",
        n_symbols=n_symbols,
        train_rows=len(X_train), oof_rows=len(X_oof), val_rows=len(X_val),
        features=len(X_train.columns) if not X_train.empty else 0,
    )

    return TierData(
        X_train=X_train, y_train=y_train, ret_train=ret_train,
        X_oof=X_oof, y_oof=y_oof, ret_oof=ret_oof,
        X_val=X_val, y_val=y_val, ret_val=ret_val,
        ohlcv_by_symbol=ohlcv_by_symbol, n_symbols=n_symbols, errors=errors,
    )


def _select_top_features(
    feature_importance: pd.Series,
    top_n: int,
    available_columns: list[str],
) -> list[str]:
    """Select top N features by importance, intersected with available columns."""
    sorted_features = feature_importance.sort_values(ascending=False)
    top_features = [f for f in sorted_features.index if f in available_columns]
    return top_features[:top_n]


def _fill_nan_for_torch(df: pd.DataFrame) -> pd.DataFrame:
    """Replace NaN/Inf with 0 for PyTorch models that can't handle missing values."""
    return df.replace([np.inf, -np.inf], np.nan).fillna(0.0)


async def train_xgboost(
    tier: str = "large",
    val_ratio: float = 0.2,
    embargo_days: int = 20,
) -> TrainingPipelineResult:
    """Train XGBoost on all symbols in a tier (standalone, 2-way split)."""
    start_time = time.perf_counter()
    config = get_config()
    result = TrainingPipelineResult(model_name="xgboost")

    symbols = await get_symbols_for_tier(tier)
    if not symbols:
        logger.error("train_no_symbols", tier=tier)
        return result

    logger.info("train_loading_data", tier=tier, n_symbols=len(symbols))

    all_X_train: list[pd.DataFrame] = []
    all_y_train: list[pd.Series] = []
    all_ret_train: list[pd.Series] = []
    all_X_val: list[pd.DataFrame] = []
    all_y_val: list[pd.Series] = []
    all_ret_val: list[pd.Series] = []

    start_date = date(2020, 1, 1)
    end_date = date.today()

    fii_dii_df: pd.DataFrame | None = None
    try:
        from alphavedha.features.macro import load_fii_dii_for_features
        fii_dii_df = await load_fii_dii_for_features(str(start_date), str(end_date))
    except Exception:
        pass

    for symbol in symbols:
        try:
            ohlcv_df = await load_ohlcv(symbol, start_date, end_date)
            if ohlcv_df.empty:
                result.errors[symbol] = "no data in DB"
                continue

            prepared = _prepare_symbol_data(symbol, ohlcv_df, fii_dii_df)
            if prepared is None:
                continue

            features_df, y_dir, y_ret = prepared

            X_tr, y_tr, ret_tr, X_v, y_v, ret_v = _temporal_split(
                features_df, y_dir, y_ret, val_ratio, embargo_days,
            )

            all_X_train.append(X_tr)
            all_y_train.append(y_tr)
            all_ret_train.append(ret_tr)
            all_X_val.append(X_v)
            all_y_val.append(y_v)
            all_ret_val.append(ret_v)
            result.n_symbols += 1

            logger.info(
                "train_symbol_prepared",
                symbol=symbol,
                train=len(X_tr),
                val=len(X_v),
            )
        except Exception as e:
            result.errors[symbol] = str(e)
            logger.error("train_symbol_error", symbol=symbol, error=str(e))

    if not all_X_train:
        logger.error("train_no_data", msg="No symbols produced training data")
        return result

    X_train = pd.concat(all_X_train, ignore_index=True)
    y_train = pd.concat(all_y_train, ignore_index=True)
    ret_train = pd.concat(all_ret_train, ignore_index=True)
    X_val = pd.concat(all_X_val, ignore_index=True)
    y_val = pd.concat(all_y_val, ignore_index=True)
    ret_val = pd.concat(all_ret_val, ignore_index=True)

    result.n_train_rows = len(X_train)
    result.n_val_rows = len(X_val)

    logger.info(
        "train_data_ready",
        n_symbols=result.n_symbols,
        train_rows=result.n_train_rows,
        val_rows=result.n_val_rows,
        features=len(X_train.columns),
    )

    model = XGBoostModel(config=config.models.xgboost)
    train_result = model.fit(
        X_train, y_train,
        X_val=X_val, y_val=y_val,
        return_train=ret_train, return_val=ret_val,
    )
    result.train_result = train_result

    artifact_dir = ARTIFACTS_DIR / "xgboost" / "latest"
    artifact = model.save(artifact_dir)
    result.artifact_path = artifact.path

    result.total_time_seconds = time.perf_counter() - start_time

    logger.info(
        "train_complete",
        model="xgboost",
        train_accuracy=train_result.train_metrics.get("accuracy"),
        val_accuracy=train_result.val_metrics.get("accuracy"),
        val_f1=train_result.val_metrics.get("f1_weighted"),
        n_symbols=result.n_symbols,
        total_time=round(result.total_time_seconds, 1),
        artifact_path=str(artifact_dir),
    )

    return result


def _train_xgb_on_data(
    data: TierData,
) -> tuple[XGBoostModel, TrainResult]:
    """Train XGBoost on pre-loaded data (used by train_all)."""
    config = get_config()
    model = XGBoostModel(config=config.models.xgboost)
    train_result = model.fit(
        data.X_train, data.y_train,
        X_val=data.X_val, y_val=data.y_val,
        return_train=data.ret_train, return_val=data.ret_val,
    )
    artifact_dir = ARTIFACTS_DIR / "xgboost" / "latest"
    model.save(artifact_dir)
    return model, train_result


def _train_lstm_on_data(
    data: TierData,
    top_features: list[str],
) -> tuple[TrainResult, Path]:
    """Train LSTM on pre-loaded data with selected features."""
    from alphavedha.models.lstm_model import LSTMModel

    config = get_config()

    X_train = _fill_nan_for_torch(data.X_train[top_features])
    X_val = _fill_nan_for_torch(data.X_val[top_features])

    model = LSTMModel(config=config.models.lstm)
    train_result = model.fit(
        X_train, data.y_train,
        X_val=X_val, y_val=data.y_val,
        return_train=data.ret_train, return_val=data.ret_val,
    )

    artifact_dir = ARTIFACTS_DIR / "lstm" / "latest"
    model.save(artifact_dir)

    return train_result, artifact_dir


def _train_tft_on_data(
    data: TierData,
    top_features: list[str] | None = None,
) -> tuple[TrainResult, Path]:
    """Train TFT on pre-loaded data."""
    from alphavedha.models.temporal_attention import TemporalAttentionModel

    config = get_config()

    cols = top_features if top_features else list(data.X_train.columns)
    X_train = _fill_nan_for_torch(data.X_train[cols])
    X_val = _fill_nan_for_torch(data.X_val[cols])

    model = TemporalAttentionModel(config=config.models.tft)
    train_result = model.fit(
        X_train, data.y_train,
        X_val=X_val, y_val=data.y_val,
        return_train=data.ret_train, return_val=data.ret_val,
    )

    artifact_dir = ARTIFACTS_DIR / "tft" / "latest"
    model.save(artifact_dir)

    return train_result, artifact_dir


def _train_regime_on_data(
    data: TierData,
) -> tuple[dict[str, float], Path]:
    """Train HMM regime detector using aggregate portfolio returns + realized volatility."""
    from alphavedha.models.regime import RegimeDetector

    config = get_config()

    all_returns: list[pd.Series] = []
    for _symbol, ohlcv_df in data.ohlcv_by_symbol.items():
        if "close" in ohlcv_df.columns and len(ohlcv_df) > 50:
            rets = np.log(ohlcv_df["close"] / ohlcv_df["close"].shift(1)).dropna()
            all_returns.append(rets)

    if not all_returns:
        logger.error("regime_no_return_data")
        return {}, Path()

    combined = pd.concat(all_returns, axis=1)
    portfolio_returns = combined.mean(axis=1).dropna()
    realized_vol = portfolio_returns.rolling(20).std().dropna()
    portfolio_returns = portfolio_returns.loc[realized_vol.index]

    detector = RegimeDetector(config=config.models.regime)
    metrics = detector.fit(portfolio_returns, realized_vol)

    artifact_dir = ARTIFACTS_DIR / "regime" / "latest"
    detector.save(artifact_dir)

    return metrics, artifact_dir


def _get_base_predictions(
    xgb_model: XGBoostModel,
    lstm_model: object,
    tft_model: object,
    X: pd.DataFrame,
    top_features: list[str],
) -> dict[str, PredictionResult]:
    """Get predictions from all 3 base models on the same dataset."""
    xgb_pred = xgb_model.predict(X)

    X_clean = _fill_nan_for_torch(X[top_features])
    lstm_pred = lstm_model.predict(X_clean)  # type: ignore[union-attr]
    tft_pred = tft_model.predict(X_clean)  # type: ignore[union-attr]

    return {"xgboost": xgb_pred, "lstm": lstm_pred, "tft": tft_pred}


def _get_regime_probs(
    data: TierData,
    n_rows: int,
) -> np.ndarray:
    """Get regime probabilities for a dataset. Falls back to uniform if detector fails."""
    from alphavedha.models.regime import RegimeDetector

    artifact_dir = ARTIFACTS_DIR / "regime" / "latest"
    if not (artifact_dir / "metadata.json").exists():
        logger.warning("regime_not_trained", msg="Using uniform regime probs")
        return np.full((n_rows, 4), 0.25)

    try:
        detector = RegimeDetector.load(artifact_dir)
        all_returns: list[pd.Series] = []
        for ohlcv_df in data.ohlcv_by_symbol.values():
            if "close" in ohlcv_df.columns and len(ohlcv_df) > 50:
                rets = np.log(ohlcv_df["close"] / ohlcv_df["close"].shift(1)).dropna()
                all_returns.append(rets)

        if not all_returns:
            return np.full((n_rows, 4), 0.25)

        combined = pd.concat(all_returns, axis=1)
        portfolio_returns = combined.mean(axis=1).dropna()
        realized_vol = portfolio_returns.rolling(20).std().dropna()
        portfolio_returns = portfolio_returns.loc[realized_vol.index]

        detector.predict(portfolio_returns, realized_vol)
        regime_features = detector.get_regime_features()
        probs = regime_features.values

        if len(probs) >= n_rows:
            return probs[-n_rows:]
        tiled = np.tile(probs[-1:], (n_rows, 1))
        tiled[:len(probs)] = probs[-min(len(probs), n_rows):]
        return tiled

    except Exception as e:
        logger.warning("regime_predict_failed", error=str(e))
        return np.full((n_rows, 4), 0.25)


def _train_ensemble_on_data(
    xgb_model: XGBoostModel,
    lstm_model: object,
    tft_model: object,
    data: TierData,
    top_features: list[str],
) -> tuple[dict[str, float], Path]:
    """Train stacking ensemble on OOF predictions from base models."""
    from alphavedha.models.ensemble import StackingEnsemble

    config = get_config()

    if data.X_oof.empty or len(data.X_oof) < 50:
        logger.error("ensemble_insufficient_oof", rows=len(data.X_oof))
        return {}, Path()

    base_preds = _get_base_predictions(
        xgb_model, lstm_model, tft_model, data.X_oof, top_features,
    )
    regime_probs = _get_regime_probs(data, len(data.X_oof))

    ensemble = StackingEnsemble(config=config.models.ensemble)
    metrics = ensemble.fit(base_preds, regime_probs, data.y_oof)

    artifact_dir = ARTIFACTS_DIR / "ensemble" / "latest"
    ensemble.save(artifact_dir)

    return metrics, artifact_dir


def _train_meta_labeling_on_data(
    xgb_model: XGBoostModel,
    lstm_model: object,
    tft_model: object,
    data: TierData,
    top_features: list[str],
) -> tuple[dict[str, float], Path]:
    """Train meta-labeling model on OOF data using ensemble predictions."""
    from alphavedha.models.ensemble import StackingEnsemble
    from alphavedha.models.meta_model import MetaLabelingModel

    if data.X_oof.empty or len(data.X_oof) < 50:
        logger.error("meta_labeling_insufficient_data", rows=len(data.X_oof))
        return {}, Path()

    ensemble_dir = ARTIFACTS_DIR / "ensemble" / "latest"
    if not (ensemble_dir / "metadata.json").exists():
        logger.error("meta_labeling_no_ensemble")
        return {}, Path()

    ensemble = StackingEnsemble.load(ensemble_dir)

    base_preds = _get_base_predictions(
        xgb_model, lstm_model, tft_model, data.X_oof, top_features,
    )
    regime_probs = _get_regime_probs(data, len(data.X_oof))
    ensemble_result = ensemble.predict(base_preds, regime_probs)

    y_correct = pd.Series(
        (ensemble_result.direction == data.y_oof.values).astype(int),
        index=data.y_oof.index,
    )

    X_oof_clean = _fill_nan_for_torch(data.X_oof)

    meta_model = MetaLabelingModel()
    metrics = meta_model.fit(
        X_oof_clean,
        ensemble_result.direction,
        ensemble_result.confidence,
        y_correct,
    )

    artifact_dir = ARTIFACTS_DIR / "meta_labeling" / "latest"
    meta_model.save(artifact_dir)

    return metrics, artifact_dir


def _train_conformal_on_data(
    data: TierData,
) -> tuple[dict[str, float], Path]:
    """Train conformal predictor on OOF data for prediction intervals."""
    from alphavedha.models.conformal import ConformalPredictor

    if data.X_oof.empty or len(data.X_oof) < 50:
        logger.error("conformal_insufficient_data", rows=len(data.X_oof))
        return {}, Path()

    X_oof_clean = _fill_nan_for_torch(data.X_oof)

    predictor = ConformalPredictor()
    metrics = predictor.fit(X_oof_clean, data.ret_oof)

    artifact_dir = ARTIFACTS_DIR / "conformal" / "latest"
    predictor.save(artifact_dir)

    return metrics, artifact_dir


async def train_lstm(
    tier: str = "large",
) -> TrainingPipelineResult:
    """Train LSTM standalone (requires XGBoost artifacts for feature selection)."""
    start_time = time.perf_counter()
    config = get_config()
    result = TrainingPipelineResult(model_name="lstm")

    xgb_artifact = ARTIFACTS_DIR / "xgboost" / "latest"
    if not (xgb_artifact / "feature_importance.csv").exists():
        logger.error("lstm_no_xgb_importance", msg="Train XGBoost first for feature selection")
        return result

    fi = pd.read_csv(xgb_artifact / "feature_importance.csv", index_col=0).squeeze()
    data = await _load_tier_data(tier)

    if data.X_train.empty:
        return result

    top_n = config.models.lstm.top_n_features
    top_features = _select_top_features(fi, top_n, list(data.X_train.columns))
    result.n_symbols = data.n_symbols
    result.n_train_rows = len(data.X_train)
    result.n_val_rows = len(data.X_val)
    result.errors = data.errors

    train_result, artifact_dir = _train_lstm_on_data(data, top_features)
    result.train_result = train_result
    result.artifact_path = artifact_dir
    result.total_time_seconds = time.perf_counter() - start_time

    logger.info("train_complete", model="lstm", total_time=round(result.total_time_seconds, 1))
    return result


async def train_tft(
    tier: str = "large",
) -> TrainingPipelineResult:
    """Train TFT standalone (requires XGBoost artifacts for feature selection)."""
    start_time = time.perf_counter()
    config = get_config()
    result = TrainingPipelineResult(model_name="tft")

    xgb_artifact = ARTIFACTS_DIR / "xgboost" / "latest"
    if not (xgb_artifact / "feature_importance.csv").exists():
        logger.error("tft_no_xgb_importance", msg="Train XGBoost first for feature selection")
        return result

    fi = pd.read_csv(xgb_artifact / "feature_importance.csv", index_col=0).squeeze()
    data = await _load_tier_data(tier)

    if data.X_train.empty:
        return result

    top_n = config.models.lstm.top_n_features
    top_features = _select_top_features(fi, top_n, list(data.X_train.columns))
    result.n_symbols = data.n_symbols
    result.n_train_rows = len(data.X_train)
    result.n_val_rows = len(data.X_val)
    result.errors = data.errors

    train_result, artifact_dir = _train_tft_on_data(data, top_features)
    result.train_result = train_result
    result.artifact_path = artifact_dir
    result.total_time_seconds = time.perf_counter() - start_time

    logger.info("train_complete", model="tft", total_time=round(result.total_time_seconds, 1))
    return result


async def train_regime(
    tier: str = "large",
) -> TrainingPipelineResult:
    """Train regime detector standalone."""
    start_time = time.perf_counter()
    result = TrainingPipelineResult(model_name="regime")

    data = await _load_tier_data(tier)
    if not data.ohlcv_by_symbol:
        return result

    result.n_symbols = data.n_symbols
    result.errors = data.errors

    metrics, artifact_dir = _train_regime_on_data(data)
    result.extra_metrics = metrics
    if artifact_dir != Path():
        result.artifact_path = artifact_dir
    result.total_time_seconds = time.perf_counter() - start_time

    logger.info("train_complete", model="regime", total_time=round(result.total_time_seconds, 1))
    return result


async def train_all(
    tier: str = "large",
) -> dict[str, TrainingPipelineResult]:
    """Train all models in dependency order: base models → ensemble → meta-labeling → conformal."""
    start_time = time.perf_counter()
    config = get_config()
    results: dict[str, TrainingPipelineResult] = {}

    logger.info("train_all_start", tier=tier)

    # Step 1: Load all data with 3-way split
    data = await _load_tier_data(tier)
    if data.X_train.empty:
        logger.error("train_all_no_data")
        return results

    # Step 2: Train XGBoost (needed first for feature selection)
    logger.info("train_all_step", step="xgboost")
    xgb_result = TrainingPipelineResult(model_name="xgboost")
    xgb_result.n_symbols = data.n_symbols
    xgb_result.n_train_rows = len(data.X_train)
    xgb_result.n_val_rows = len(data.X_val)

    xgb_model, xgb_train = _train_xgb_on_data(data)
    xgb_result.train_result = xgb_train
    xgb_result.artifact_path = ARTIFACTS_DIR / "xgboost" / "latest"
    results["xgboost"] = xgb_result

    logger.info(
        "train_all_xgb_done",
        train_acc=xgb_train.train_metrics.get("accuracy"),
        val_acc=xgb_train.val_metrics.get("accuracy"),
    )

    # Step 3: Select top features for LSTM/TFT
    fi = xgb_model.get_feature_importance()
    top_n = config.models.lstm.top_n_features
    top_features = _select_top_features(
        fi, top_n, list(data.X_train.columns),
    ) if fi is not None else list(data.X_train.columns)[:top_n]

    logger.info("train_all_features_selected", top_n=len(top_features))

    # Step 4: Train LSTM
    logger.info("train_all_step", step="lstm")
    lstm_result = TrainingPipelineResult(model_name="lstm")
    lstm_result.n_symbols = data.n_symbols
    lstm_result.n_train_rows = len(data.X_train)
    lstm_result.n_val_rows = len(data.X_val)
    try:
        lstm_train, lstm_dir = _train_lstm_on_data(data, top_features)
        lstm_result.train_result = lstm_train
        lstm_result.artifact_path = lstm_dir
        logger.info(
            "train_all_lstm_done",
            train_acc=lstm_train.train_metrics.get("accuracy"),
            val_acc=lstm_train.val_metrics.get("accuracy"),
        )
    except Exception as e:
        lstm_result.errors["lstm"] = str(e)
        logger.error("train_all_lstm_failed", error=str(e))
    results["lstm"] = lstm_result

    # Step 5: Train TFT
    logger.info("train_all_step", step="tft")
    tft_result = TrainingPipelineResult(model_name="tft")
    tft_result.n_symbols = data.n_symbols
    tft_result.n_train_rows = len(data.X_train)
    tft_result.n_val_rows = len(data.X_val)
    try:
        tft_train, tft_dir = _train_tft_on_data(data, top_features)
        tft_result.train_result = tft_train
        tft_result.artifact_path = tft_dir
        logger.info(
            "train_all_tft_done",
            train_acc=tft_train.train_metrics.get("accuracy"),
            val_acc=tft_train.val_metrics.get("accuracy"),
        )
    except Exception as e:
        tft_result.errors["tft"] = str(e)
        logger.error("train_all_tft_failed", error=str(e))
    results["tft"] = tft_result

    # Step 6: Train Regime Detector
    logger.info("train_all_step", step="regime")
    regime_result = TrainingPipelineResult(model_name="regime")
    regime_result.n_symbols = data.n_symbols
    try:
        regime_metrics, regime_dir = _train_regime_on_data(data)
        regime_result.extra_metrics = regime_metrics
        if regime_dir != Path():
            regime_result.artifact_path = regime_dir
        logger.info("train_all_regime_done", metrics=regime_metrics)
    except Exception as e:
        regime_result.errors["regime"] = str(e)
        logger.error("train_all_regime_failed", error=str(e))
    results["regime"] = regime_result

    # Step 7: Train Ensemble (needs all base models + regime)
    if all(
        results[m].train_result is not None or results[m].artifact_path is not None
        for m in ["xgboost", "lstm", "tft"]
    ):
        logger.info("train_all_step", step="ensemble")
        ensemble_result = TrainingPipelineResult(model_name="ensemble")
        try:
            from alphavedha.models.lstm_model import LSTMModel
            from alphavedha.models.temporal_attention import TemporalAttentionModel

            lstm_model = LSTMModel.load(ARTIFACTS_DIR / "lstm" / "latest")
            tft_model = TemporalAttentionModel.load(ARTIFACTS_DIR / "tft" / "latest")

            ens_metrics, ens_dir = _train_ensemble_on_data(
                xgb_model, lstm_model, tft_model, data, top_features,
            )
            ensemble_result.extra_metrics = ens_metrics
            if ens_dir != Path():
                ensemble_result.artifact_path = ens_dir
            logger.info("train_all_ensemble_done", metrics=ens_metrics)
        except Exception as e:
            ensemble_result.errors["ensemble"] = str(e)
            logger.error("train_all_ensemble_failed", error=str(e))
        results["ensemble"] = ensemble_result

        # Step 8: Train Meta-Labeling (needs ensemble)
        if ensemble_result.artifact_path is not None:
            logger.info("train_all_step", step="meta_labeling")
            meta_result = TrainingPipelineResult(model_name="meta_labeling")
            try:
                meta_metrics, meta_dir = _train_meta_labeling_on_data(
                    xgb_model, lstm_model, tft_model, data, top_features,
                )
                meta_result.extra_metrics = meta_metrics
                if meta_dir != Path():
                    meta_result.artifact_path = meta_dir
                logger.info("train_all_meta_done", metrics=meta_metrics)
            except Exception as e:
                meta_result.errors["meta_labeling"] = str(e)
                logger.error("train_all_meta_failed", error=str(e))
            results["meta_labeling"] = meta_result
    else:
        logger.warning("train_all_skip_ensemble", msg="Not all base models trained successfully")

    # Step 9: Train Conformal Predictor
    logger.info("train_all_step", step="conformal")
    conformal_result = TrainingPipelineResult(model_name="conformal")
    try:
        conf_metrics, conf_dir = _train_conformal_on_data(data)
        conformal_result.extra_metrics = conf_metrics
        if conf_dir != Path():
            conformal_result.artifact_path = conf_dir
        logger.info("train_all_conformal_done", metrics=conf_metrics)
    except Exception as e:
        conformal_result.errors["conformal"] = str(e)
        logger.error("train_all_conformal_failed", error=str(e))
    results["conformal"] = conformal_result

    total_time = time.perf_counter() - start_time
    for r in results.values():
        r.total_time_seconds = total_time

    logger.info(
        "train_all_complete",
        models_trained=[m for m, r in results.items() if r.artifact_path is not None],
        models_failed=[m for m, r in results.items() if r.artifact_path is None],
        total_time=round(total_time, 1),
    )

    return results
