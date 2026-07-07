"""Training pipeline — end-to-end: load data → features → labels → train → save.

Orchestrates the full training workflow for a given model type and universe tier.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import joblib
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
from alphavedha.monitoring.experiment_tracker import ExperimentTracker

logger = structlog.get_logger(__name__)

# Override via ALPHAVEDHA_ARTIFACTS_DIR so a one-off (e.g. the historical-sim
# runner) can train a frozen as-of model into an isolated directory without
# overwriting the live `latest` models or the live tier-data cache. Unset = live.
ARTIFACTS_DIR = Path(os.environ.get("ALPHAVEDHA_ARTIFACTS_DIR", "models/artifacts"))

# Features that always produce constants or NaN because their data sources are not
# wired up yet. Including them wastes XGBoost split budget on zero-variance columns.
_STUB_FEATURES: frozenset[str] = frozenset(
    [
        # macro — hardcoded constants (no data source)
        "macro_gsec_10y",
        "macro_gsec_change_1d",
        "macro_pmi",
        "macro_pmi_staleness_days",
        # macro — require universe_prices arg (not passed per-symbol)
        "macro_breadth_200sma",
        "macro_adv_dec_ratio",
        # derivatives — participant OI not fetched (NSE bulk data not wired)
        "deriv_fii_futures_oi",
        "deriv_fii_options_oi",
        "deriv_pro_futures_net",
        "deriv_retail_futures_net",
        "deriv_gex",
        "deriv_delta_oi",
        # returns — hardcoded constant (HMM not applied per-symbol at feature time)
        "ret_regime",
        # trends — Google Trends API not wired
        "trends_sector_7d",
        "trends_sector_change",
    ]
)

# Raw ordinal date positions let trees memorize WHERE in history a row sits
# instead of market structure. At inference every symbol shares the same
# value (cal_year=2026 for all 50), so the model replays whatever the
# current year's training slice looked like — diagnosed Jul 2026 as
# cal_year/cal_doy top importance on all symbols with a uniform bearish
# cross-section. Cyclical calendar features (cal_month, expiry flags,
# result-season flags, ...) remain — they encode seasonality, not position.
# Still computed at serve time so previously trained artifacts keep working.
_RAW_DATE_FEATURES: frozenset[str] = frozenset(
    [
        "cal_year",
        "cal_doy",
        "cal_week_of_year",
    ]
)


def _drop_stub_features(df: pd.DataFrame) -> pd.DataFrame:
    """Remove known-stub (constant/NaN) and raw-date columns before training."""
    cols_to_drop = [c for c in df.columns if c in _STUB_FEATURES]
    if cols_to_drop:
        logger.info("train_drop_stub_features", dropped=cols_to_drop, n=len(cols_to_drop))
        df = df.drop(columns=cols_to_drop)
    date_cols = [c for c in df.columns if c in _RAW_DATE_FEATURES]
    if date_cols:
        logger.info("train_drop_raw_date_features", dropped=date_cols, n=len(date_cols))
        df = df.drop(columns=date_cols)
    return df


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
    macro_df: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.Series, pd.Series] | None:
    """Compute features and labels for a single symbol. Returns (X, y_direction, y_return) or None."""
    config = get_config()

    if len(ohlcv_df) < 252:
        logger.warning("train_skip_short", symbol=symbol, rows=len(ohlcv_df))
        return None

    feature_result = compute_all_features(
        symbol=symbol,
        ohlcv_df=ohlcv_df,
        fii_dii_df=fii_dii_df,
        macro_df=macro_df,
    )
    features_df = feature_result.df

    label_result = compute_triple_barrier_labels(
        ohlcv_df,
        config.labels.triple_barrier,
        symbol=symbol,
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
    pd.DataFrame,
    pd.Series,
    pd.Series,
    pd.DataFrame,
    pd.Series,
    pd.Series,
    pd.DataFrame,
    pd.Series,
    pd.Series,
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
        X.iloc[:train_end],
        y.iloc[:train_end],
        returns.iloc[:train_end],
        X.iloc[oof_start:oof_end_raw],
        y.iloc[oof_start:oof_end_raw],
        returns.iloc[oof_start:oof_end_raw],
        X.iloc[val_start:],
        y.iloc[val_start:],
        returns.iloc[val_start:],
    )


async def _load_tier_data(
    tier: str = "large",
    oof_ratio: float = 0.15,
    val_ratio: float = 0.15,
    embargo_days: int = 20,
    use_cache: bool = True,
    end_date: date | None = None,
) -> TierData:
    """Load all symbol data for a tier with 3-way temporal split.

    Results are cached to disk keyed by (tier, today, split params) so that
    sequential model trainings on the same day skip the ~5 min feature
    recomputation. The cache is invalidated daily and on split-param changes.
    """
    # None = today (live). A past cutoff trains an out-of-sample frozen model
    # that never sees data after end_date; it also keys the cache so different
    # cutoffs never collide and never clobber the live (today-keyed) cache.
    end_date = end_date or date.today()
    cache_dir = ARTIFACTS_DIR / "tier_data_cache"
    cache_path = (
        cache_dir / f"{tier}_{end_date.isoformat()}_{oof_ratio}_{val_ratio}_{embargo_days}.joblib"
    )
    if use_cache and cache_path.exists():
        try:
            cached: TierData = joblib.load(cache_path)
            logger.info("tier_data_cache_hit", path=str(cache_path))
            return cached
        except Exception as e:
            logger.warning("tier_data_cache_load_failed", error=str(e))

    symbols = await get_symbols_for_tier(tier)
    if not symbols:
        logger.error("train_no_symbols", tier=tier)
        return TierData(
            X_train=pd.DataFrame(),
            y_train=pd.Series(dtype=float),
            ret_train=pd.Series(dtype=float),
            X_oof=pd.DataFrame(),
            y_oof=pd.Series(dtype=float),
            ret_oof=pd.Series(dtype=float),
            X_val=pd.DataFrame(),
            y_val=pd.Series(dtype=float),
            ret_val=pd.Series(dtype=float),
            ohlcv_by_symbol={},
            n_symbols=0,
            errors={},
        )

    logger.info("train_loading_data", tier=tier, n_symbols=len(symbols))

    all_train: list[tuple[pd.DataFrame, pd.Series, pd.Series]] = []
    all_oof: list[tuple[pd.DataFrame, pd.Series, pd.Series]] = []
    all_val: list[tuple[pd.DataFrame, pd.Series, pd.Series]] = []
    ohlcv_by_symbol: dict[str, pd.DataFrame] = {}
    errors: dict[str, str] = {}
    n_symbols = 0

    start_date = date(2020, 1, 1)

    fii_dii_df: pd.DataFrame | None = None
    try:
        from alphavedha.features.macro import load_fii_dii_for_features

        fii_dii_df = await load_fii_dii_for_features(str(start_date), str(end_date))
        if fii_dii_df is not None and not fii_dii_df.empty:
            logger.info("train_fii_dii_loaded", rows=len(fii_dii_df))
    except Exception as e:
        logger.warning("train_fii_dii_load_failed", error=str(e))

    macro_df: pd.DataFrame | None = None
    try:
        import asyncio

        from alphavedha.features.macro import fetch_macro_data

        macro_df = await asyncio.to_thread(fetch_macro_data, str(start_date), str(end_date))
        if macro_df is not None and not macro_df.empty:
            logger.info("train_macro_loaded", rows=len(macro_df), cols=list(macro_df.columns))
    except Exception as e:
        logger.warning("train_macro_load_failed", error=str(e))

    for symbol in symbols:
        try:
            ohlcv_df = await load_ohlcv(symbol, start_date, end_date)
            if ohlcv_df.empty:
                errors[symbol] = "no data in DB"
                continue

            ohlcv_by_symbol[symbol] = ohlcv_df
            prepared = _prepare_symbol_data(symbol, ohlcv_df, fii_dii_df, macro_df)
            if prepared is None:
                continue

            features_df, y_dir, y_ret = prepared

            X_tr, y_tr, ret_tr, X_o, y_o, ret_o, X_v, y_v, ret_v = _temporal_split_3way(
                features_df,
                y_dir,
                y_ret,
                oof_ratio,
                val_ratio,
                embargo_days,
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
                train=len(X_tr),
                oof=len(X_o),
                val=len(X_v),
            )
        except Exception as e:
            errors[symbol] = str(e)
            logger.error("train_symbol_error", symbol=symbol, error=str(e))

    def _concat(
        parts: list[tuple[pd.DataFrame, pd.Series, pd.Series]],
    ) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
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

    if not X_train.empty:
        X_train = _drop_stub_features(X_train)
        live_cols = list(X_train.columns)
        X_oof = X_oof[live_cols] if not X_oof.empty else X_oof
        X_val = X_val[live_cols] if not X_val.empty else X_val

    logger.info(
        "train_data_ready",
        n_symbols=n_symbols,
        train_rows=len(X_train),
        oof_rows=len(X_oof),
        val_rows=len(X_val),
        features=len(X_train.columns) if not X_train.empty else 0,
    )

    result = TierData(
        X_train=X_train,
        y_train=y_train,
        ret_train=ret_train,
        X_oof=X_oof,
        y_oof=y_oof,
        ret_oof=ret_oof,
        X_val=X_val,
        y_val=y_val,
        ret_val=ret_val,
        ohlcv_by_symbol=ohlcv_by_symbol,
        n_symbols=n_symbols,
        errors=errors,
    )

    if use_cache and not X_train.empty:
        try:
            cache_dir.mkdir(parents=True, exist_ok=True)
            for old in cache_dir.glob(f"{tier}_*.joblib"):
                if old != cache_path:
                    old.unlink(missing_ok=True)
            joblib.dump(result, cache_path)
            logger.info("tier_data_cache_saved", path=str(cache_path))
        except Exception as e:
            logger.warning("tier_data_cache_save_failed", error=str(e))

    return result


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


_DEGENERATE_MAX_CLASS_SHARE = 0.90
_DEGENERATE_MIN_VAL_ACCURACY = 0.40


def _check_degenerate_direction_model(
    model: XGBoostModel,
    X_val: pd.DataFrame,
    y_val: pd.Series,
) -> str | None:
    """Return a rejection reason when a direction model is unusable, else None.

    Two failure modes are checked on the validation set:
    - collapse: one class dominates the predictions (> 90%), which is how a
      majority-class model looks — it scores near the class prior but carries
      zero signal;
    - accuracy floor: worse than 0.40 on three classes means the artifact
      would degrade whatever is currently serving.

    An empty validation set skips the gate rather than blocking the train.
    """
    if X_val.empty or y_val.empty:
        return None

    pred = model.predict(X_val)
    directions = pd.Series(pred.direction)
    shares = directions.value_counts(normalize=True)
    top_class = int(shares.index[0])
    top_share = float(shares.iloc[0])
    if top_share > _DEGENERATE_MAX_CLASS_SHARE:
        return f"{top_share:.0%} of val predictions are class {top_class}"

    accuracy = float((directions.to_numpy() == y_val.to_numpy()).mean())
    if accuracy < _DEGENERATE_MIN_VAL_ACCURACY:
        return f"val accuracy {accuracy:.3f} below floor {_DEGENERATE_MIN_VAL_ACCURACY}"

    return None


async def train_xgboost(
    tier: str = "large",
    val_ratio: float = 0.2,
    embargo_days: int = 20,
    end_date: date | None = None,
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
    end_date = end_date or date.today()

    fii_dii_df: pd.DataFrame | None = None
    try:
        from alphavedha.features.macro import load_fii_dii_for_features

        fii_dii_df = await load_fii_dii_for_features(str(start_date), str(end_date))
    except Exception:
        pass

    # Load macro exactly like _load_tier_data does — without it the nightly
    # retrain fits on a different feature distribution than the weekly full
    # pipeline, and serving computes macro features the model never saw.
    macro_df: pd.DataFrame | None = None
    try:
        import asyncio

        from alphavedha.features.macro import fetch_macro_data

        macro_df = await asyncio.to_thread(fetch_macro_data, str(start_date), str(end_date))
        if macro_df is not None and not macro_df.empty:
            logger.info("train_macro_loaded", rows=len(macro_df))
    except Exception as e:
        logger.warning("train_macro_load_failed", error=str(e))

    for symbol in symbols:
        try:
            ohlcv_df = await load_ohlcv(symbol, start_date, end_date)
            if ohlcv_df.empty:
                result.errors[symbol] = "no data in DB"
                continue

            prepared = _prepare_symbol_data(symbol, ohlcv_df, fii_dii_df, macro_df)
            if prepared is None:
                continue

            features_df, y_dir, y_ret = prepared

            X_tr, y_tr, ret_tr, X_v, y_v, ret_v = _temporal_split(
                features_df,
                y_dir,
                y_ret,
                val_ratio,
                embargo_days,
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

    X_train = _drop_stub_features(X_train)
    live_cols = list(X_train.columns)
    X_val = X_val[live_cols]

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
        X_train,
        y_train,
        X_val=X_val,
        y_val=y_val,
        return_train=ret_train,
        return_val=ret_val,
    )
    result.train_result = train_result

    # Promotion gate: this function feeds the unattended nightly retrain,
    # which would otherwise overwrite the production artifact with whatever
    # came out of the fit. A degenerate model (all-short, Jun 2026) must
    # never replace a working one; leaving artifact_path=None makes the
    # scheduler record the run as failed and alert.
    degeneracy = _check_degenerate_direction_model(model, X_val, y_val)
    if degeneracy is not None:
        result.total_time_seconds = time.perf_counter() - start_time
        logger.error(
            "train_rejected_degenerate",
            model="xgboost",
            reason=degeneracy,
            val_accuracy=train_result.val_metrics.get("accuracy"),
            val_f1=train_result.val_metrics.get("f1_weighted"),
        )
        return result

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
        data.X_train,
        data.y_train,
        X_val=data.X_val,
        y_val=data.y_val,
        return_train=data.ret_train,
        return_val=data.ret_val,
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
        X_train,
        data.y_train,
        X_val=X_val,
        y_val=data.y_val,
        return_train=data.ret_train,
        return_val=data.ret_val,
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
        X_train,
        data.y_train,
        X_val=X_val,
        y_val=data.y_val,
        return_train=data.ret_train,
        return_val=data.ret_val,
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


def _train_gnn_on_data(data: TierData) -> tuple[TrainResult, Path]:
    """Train GNN on pre-loaded data (no graph structure — falls back to MLP)."""
    from alphavedha.models.gnn_model import GNNModel

    X_train = _fill_nan_for_torch(data.X_train)
    X_val = _fill_nan_for_torch(data.X_val)

    model = GNNModel()
    train_result = model.fit(
        X_train,
        data.y_train,
        X_val=X_val,
        y_val=data.y_val,
    )

    artifact_dir = ARTIFACTS_DIR / "gnn" / "latest"
    model.save(artifact_dir)
    return train_result, artifact_dir


def _get_base_predictions(
    xgb_model: XGBoostModel,
    lstm_model: object,
    tft_model: object,
    X: pd.DataFrame,
    top_features: list[str],
    gnn_model: object | None = None,
) -> dict[str, PredictionResult]:
    """Get predictions from base models. GNN is an optional 4th learner."""
    xgb_pred = xgb_model.predict(X)

    X_clean = _fill_nan_for_torch(X[top_features])
    lstm_pred = lstm_model.predict(X_clean)  # type: ignore[union-attr]
    tft_pred = tft_model.predict(X_clean)  # type: ignore[union-attr]

    preds: dict[str, PredictionResult] = {"xgboost": xgb_pred, "lstm": lstm_pred, "tft": tft_pred}
    if gnn_model is not None:
        X_gnn = _fill_nan_for_torch(X)
        preds["gnn"] = gnn_model.predict(X_gnn)  # type: ignore[union-attr]
    return preds


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
        tiled[: len(probs)] = probs[-min(len(probs), n_rows) :]
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
    gnn_model: object | None = None,
) -> tuple[dict[str, float], Path]:
    """Train stacking ensemble on OOF predictions from base models."""
    from alphavedha.models.ensemble import StackingEnsemble

    config = get_config()

    if data.X_oof.empty or len(data.X_oof) < 50:
        logger.error("ensemble_insufficient_oof", rows=len(data.X_oof))
        return {}, Path()

    base_preds = _get_base_predictions(
        xgb_model,
        lstm_model,
        tft_model,
        data.X_oof,
        top_features,
        gnn_model=gnn_model,
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
    gnn_model: object | None = None,
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
        xgb_model,
        lstm_model,
        tft_model,
        data.X_oof,
        top_features,
        gnn_model=gnn_model,
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


_CONFORMAL_MAX_OOF_SAMPLES = 5_000  # caps MAPIE memory use on CX23 (4GB)


def _train_conformal_on_data(
    data: TierData,
) -> tuple[dict[str, float], Path]:
    """Train conformal predictor on OOF data for prediction intervals."""
    from alphavedha.models.conformal import ConformalPredictor

    if data.X_oof.empty or len(data.X_oof) < 50:
        logger.error("conformal_insufficient_data", rows=len(data.X_oof))
        return {}, Path()

    X_oof = data.X_oof
    ret_oof = data.ret_oof
    if len(X_oof) > _CONFORMAL_MAX_OOF_SAMPLES:
        # Take the most recent rows — preserves temporal ordering and avoids OOM
        X_oof = X_oof.iloc[-_CONFORMAL_MAX_OOF_SAMPLES:]
        ret_oof = ret_oof.iloc[-_CONFORMAL_MAX_OOF_SAMPLES:]
        logger.info("conformal_oof_subsampled", original=len(data.X_oof), kept=len(X_oof))

    X_oof_clean = _fill_nan_for_torch(X_oof)

    predictor = ConformalPredictor()
    metrics = predictor.fit(X_oof_clean, ret_oof)

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


def _log_experiment(
    tracker: ExperimentTracker,
    result: TrainingPipelineResult,
    data: TierData,
) -> None:
    """Log a training run to the experiment tracker (skip if errors)."""
    if result.errors:
        return

    metrics = {}
    if result.train_result is not None:
        metrics.update(result.train_result.val_metrics)
    elif result.extra_metrics:
        metrics.update(result.extra_metrics)

    try:
        tracker.log_run(
            model_name=result.model_name,
            hyperparams={},
            train_metrics=result.train_result.train_metrics if result.train_result else {},
            val_metrics=metrics,
            data_range=("", ""),
            n_train_rows=result.n_train_rows or len(data.X_train),
            n_val_rows=result.n_val_rows or len(data.X_val),
            n_symbols=result.n_symbols or data.n_symbols,
            feature_count=data.X_train.shape[1] if not data.X_train.empty else 0,
            artifact_path=str(result.artifact_path) if result.artifact_path else "",
            duration_seconds=result.total_time_seconds,
            extra=result.extra_metrics or {},
        )
    except Exception as e:
        logger.warning("experiment_log_failed", model=result.model_name, error=str(e))


def _train_rl_on_data(data: TierData) -> TrainingPipelineResult:
    """Bridge TierData to the RL training pipeline."""
    from alphavedha.training.rl_pipeline import train_rl_agent

    result = TrainingPipelineResult(model_name="rl_agent")
    try:
        price_dfs = []
        for symbol, ohlcv in data.ohlcv_by_symbol.items():
            price_dfs.append(ohlcv[["close"]].rename(columns={"close": symbol}))

        if not price_dfs:
            result.errors["rl_agent"] = "No price data available"
            return result

        price_df = pd.concat(price_dfs, axis=1).dropna()
        symbols = list(data.ohlcv_by_symbol.keys())

        train_len = len(data.X_train)
        rl_result = train_rl_agent(
            feature_df=data.X_train,
            price_df=price_df.iloc[:train_len] if len(price_df) >= train_len else price_df,
            symbols=symbols,
            n_episodes=50,
        )

        result.extra_metrics = {
            "val_return": rl_result.total_return,
            "val_sharpe": rl_result.sharpe_ratio,
            "val_max_dd": rl_result.max_drawdown,
        }
        if rl_result.artifact_path:
            result.artifact_path = Path(rl_result.artifact_path)
    except Exception as e:
        result.errors["rl_agent"] = str(e)
        logger.error("train_rl_failed", error=str(e))

    return result


async def train_ensemble(
    tier: str = "large",
) -> TrainingPipelineResult:
    """Train stacking ensemble (requires XGBoost, LSTM, TFT artifacts on disk)."""
    start_time = time.perf_counter()
    config = get_config()
    result = TrainingPipelineResult(model_name="ensemble")

    xgb_artifact = ARTIFACTS_DIR / "xgboost" / "latest"
    lstm_artifact = ARTIFACTS_DIR / "lstm" / "latest"
    tft_artifact = ARTIFACTS_DIR / "tft" / "latest"

    for name, path in [("xgboost", xgb_artifact), ("lstm", lstm_artifact), ("tft", tft_artifact)]:
        if not (path / "metadata.json").exists():
            logger.error("ensemble_missing_artifact", model=name, path=str(path))
            return result

    if not (xgb_artifact / "feature_importance.csv").exists():
        logger.error("ensemble_no_xgb_importance")
        return result

    data = await _load_tier_data(tier)
    if data.X_train.empty:
        return result

    fi = pd.read_csv(xgb_artifact / "feature_importance.csv", index_col=0).squeeze()
    top_n = config.models.lstm.top_n_features
    top_features = _select_top_features(fi, top_n, list(data.X_train.columns))

    result.n_symbols = data.n_symbols
    result.n_train_rows = len(data.X_train)
    result.n_val_rows = len(data.X_val)
    result.errors = data.errors

    from alphavedha.models.lstm_model import LSTMModel
    from alphavedha.models.temporal_attention import TemporalAttentionModel

    xgb_model = XGBoostModel.load(xgb_artifact)
    lstm_model = LSTMModel.load(lstm_artifact)
    tft_model = TemporalAttentionModel.load(tft_artifact)

    ens_metrics, ens_dir = _train_ensemble_on_data(
        xgb_model, lstm_model, tft_model, data, top_features
    )
    result.extra_metrics = ens_metrics
    if ens_dir != Path():
        result.artifact_path = ens_dir
    result.total_time_seconds = time.perf_counter() - start_time

    logger.info("train_complete", model="ensemble", total_time=round(result.total_time_seconds, 1))
    return result


async def train_meta_labeling(
    tier: str = "large",
) -> TrainingPipelineResult:
    """Train meta-labeling model (requires XGBoost, LSTM, TFT, and Ensemble artifacts on disk)."""
    start_time = time.perf_counter()
    config = get_config()
    result = TrainingPipelineResult(model_name="meta_labeling")

    xgb_artifact = ARTIFACTS_DIR / "xgboost" / "latest"
    lstm_artifact = ARTIFACTS_DIR / "lstm" / "latest"
    tft_artifact = ARTIFACTS_DIR / "tft" / "latest"
    ensemble_artifact = ARTIFACTS_DIR / "ensemble" / "latest"

    for name, path in [
        ("xgboost", xgb_artifact),
        ("lstm", lstm_artifact),
        ("tft", tft_artifact),
        ("ensemble", ensemble_artifact),
    ]:
        if not (path / "metadata.json").exists():
            logger.error("meta_labeling_missing_artifact", model=name, path=str(path))
            return result

    if not (xgb_artifact / "feature_importance.csv").exists():
        logger.error("meta_labeling_no_xgb_importance")
        return result

    data = await _load_tier_data(tier)
    if data.X_train.empty:
        return result

    fi = pd.read_csv(xgb_artifact / "feature_importance.csv", index_col=0).squeeze()
    top_n = config.models.lstm.top_n_features
    top_features = _select_top_features(fi, top_n, list(data.X_train.columns))

    result.n_symbols = data.n_symbols
    result.n_train_rows = len(data.X_train)
    result.n_val_rows = len(data.X_val)
    result.errors = data.errors

    from alphavedha.models.lstm_model import LSTMModel
    from alphavedha.models.temporal_attention import TemporalAttentionModel

    xgb_model = XGBoostModel.load(xgb_artifact)
    lstm_model = LSTMModel.load(lstm_artifact)
    tft_model = TemporalAttentionModel.load(tft_artifact)

    meta_metrics, meta_dir = _train_meta_labeling_on_data(
        xgb_model, lstm_model, tft_model, data, top_features
    )
    result.extra_metrics = meta_metrics
    if meta_dir != Path():
        result.artifact_path = meta_dir
    result.total_time_seconds = time.perf_counter() - start_time

    logger.info(
        "train_complete", model="meta_labeling", total_time=round(result.total_time_seconds, 1)
    )
    return result


async def train_conformal(
    tier: str = "large",
) -> TrainingPipelineResult:
    """Train conformal predictor for prediction intervals (no base model dependencies)."""
    start_time = time.perf_counter()
    result = TrainingPipelineResult(model_name="conformal")

    data = await _load_tier_data(tier)
    if data.X_train.empty:
        return result

    result.n_symbols = data.n_symbols
    result.n_train_rows = len(data.X_train)
    result.n_val_rows = len(data.X_val)
    result.errors = data.errors

    conf_metrics, conf_dir = _train_conformal_on_data(data)
    result.extra_metrics = conf_metrics
    if conf_dir != Path():
        result.artifact_path = conf_dir
    result.total_time_seconds = time.perf_counter() - start_time

    logger.info("train_complete", model="conformal", total_time=round(result.total_time_seconds, 1))
    return result


async def train_all(
    tier: str = "large",
    end_date: date | None = None,
) -> dict[str, TrainingPipelineResult]:
    """Train all models in dependency order: base models → ensemble → meta-labeling → conformal.

    ``end_date`` limits training data to bars on or before that date (None =
    today). Combined with ``ALPHAVEDHA_ARTIFACTS_DIR``, this trains a frozen,
    out-of-sample model for historical simulation without touching live models.
    """
    start_time = time.perf_counter()
    config = get_config()
    results: dict[str, TrainingPipelineResult] = {}

    tracker = ExperimentTracker(base_dir=ARTIFACTS_DIR)
    logger.info("train_all_start", tier=tier)

    # Step 1: Load all data with 3-way split
    data = await _load_tier_data(tier, end_date=end_date)
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
    _log_experiment(tracker, xgb_result, data)
    results["xgboost"] = xgb_result

    logger.info(
        "train_all_xgb_done",
        train_acc=xgb_train.train_metrics.get("accuracy"),
        val_acc=xgb_train.val_metrics.get("accuracy"),
    )

    # Step 3: Select top features for LSTM/TFT
    fi = xgb_model.get_feature_importance()
    top_n = config.models.lstm.top_n_features
    top_features = (
        _select_top_features(
            fi,
            top_n,
            list(data.X_train.columns),
        )
        if fi is not None
        else list(data.X_train.columns)[:top_n]
    )

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
    _log_experiment(tracker, lstm_result, data)
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
    _log_experiment(tracker, tft_result, data)
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
    _log_experiment(tracker, regime_result, data)
    results["regime"] = regime_result

    # Step 6.5: Train GNN (optional 4th base learner — no graph, MLP fallback)
    logger.info("train_all_step", step="gnn")
    gnn_result = TrainingPipelineResult(model_name="gnn")
    gnn_model_loaded: object | None = None
    try:
        from alphavedha.models.gnn_model import GNNModel

        gnn_train, gnn_dir = _train_gnn_on_data(data)
        gnn_result.train_result = gnn_train
        gnn_result.artifact_path = gnn_dir
        gnn_model_loaded = GNNModel.load(gnn_dir)
        logger.info(
            "train_all_gnn_done",
            train_acc=gnn_train.train_metrics.get("accuracy"),
            val_acc=gnn_train.val_metrics.get("accuracy"),
        )
    except Exception as e:
        gnn_result.errors["gnn"] = str(e)
        logger.error("train_all_gnn_failed", error=str(e))
    _log_experiment(tracker, gnn_result, data)
    results["gnn"] = gnn_result

    # Step 7: Train Ensemble (needs all base models + regime + optional GNN)
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
                xgb_model,
                lstm_model,
                tft_model,
                data,
                top_features,
                gnn_model=gnn_model_loaded,
            )
            ensemble_result.extra_metrics = ens_metrics
            if ens_dir != Path():
                ensemble_result.artifact_path = ens_dir
            logger.info("train_all_ensemble_done", metrics=ens_metrics)
        except Exception as e:
            ensemble_result.errors["ensemble"] = str(e)
            logger.error("train_all_ensemble_failed", error=str(e))
        _log_experiment(tracker, ensemble_result, data)
        results["ensemble"] = ensemble_result

        # Step 8: Train Meta-Labeling (needs ensemble)
        if ensemble_result.artifact_path is not None:
            logger.info("train_all_step", step="meta_labeling")
            meta_result = TrainingPipelineResult(model_name="meta_labeling")
            try:
                meta_metrics, meta_dir = _train_meta_labeling_on_data(
                    xgb_model,
                    lstm_model,
                    tft_model,
                    data,
                    top_features,
                    gnn_model=gnn_model_loaded,
                )
                meta_result.extra_metrics = meta_metrics
                if meta_dir != Path():
                    meta_result.artifact_path = meta_dir
                logger.info("train_all_meta_done", metrics=meta_metrics)
            except Exception as e:
                meta_result.errors["meta_labeling"] = str(e)
                logger.error("train_all_meta_failed", error=str(e))
            _log_experiment(tracker, meta_result, data)
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
    _log_experiment(tracker, conformal_result, data)
    results["conformal"] = conformal_result

    # Step 10: RL Agent
    logger.info("train_all_step", step="rl_agent")
    rl_result = _train_rl_on_data(data)
    _log_experiment(tracker, rl_result, data)
    results["rl_agent"] = rl_result

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
