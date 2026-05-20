"""GNN training pipeline — build stock graph, stack features, train with temporal split."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import structlog

from alphavedha.data.stock_graph import build_stock_graph
from alphavedha.exceptions import InsufficientDataError
from alphavedha.models.base import TrainResult
from alphavedha.models.gnn_model import GNNModel

logger = structlog.get_logger(__name__)

ARTIFACTS_DIR = Path("models/artifacts")


def _temporal_split(
    X: pd.DataFrame,
    y: pd.Series,
    val_ratio: float = 0.2,
    embargo_days: int = 20,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Split by time with embargo gap — no random shuffling for time-series."""
    n = len(X)
    split_idx = int(n * (1 - val_ratio))
    train_end = max(split_idx - embargo_days, 50)

    X_train = X.iloc[:train_end]
    y_train = y.iloc[:train_end]
    X_val = X.iloc[split_idx:]
    y_val = y.iloc[split_idx:]

    return X_train, y_train, X_val, y_val


async def train_gnn(
    symbols: list[str],
    feature_df: dict[str, pd.DataFrame],
    labels: dict[str, pd.Series],
    returns_df: pd.DataFrame | None = None,
    config: dict[str, Any] | None = None,
    output_dir: Path | None = None,
) -> TrainResult:
    """Train the GNN model.

    1. Build stock graph from symbols + returns
    2. Stack features into a single node-feature matrix (one row per symbol per date)
    3. Train/val split by time (not random)
    4. Train GNN with early stopping
    5. Save artifacts
    """
    start = time.perf_counter()

    valid_symbols = [s for s in symbols if s in feature_df and s in labels]
    if len(valid_symbols) < 2:
        raise InsufficientDataError(
            f"Need at least 2 symbols with features and labels, got {len(valid_symbols)}",
        )

    logger.info("gnn_pipeline_start", n_symbols=len(valid_symbols))

    graph = build_stock_graph(
        symbols=valid_symbols,
        returns_df=returns_df,
    )

    all_X: list[pd.DataFrame] = []
    all_y: list[pd.Series] = []

    for symbol in valid_symbols:
        feat = feature_df[symbol]
        lab = labels[symbol]

        common_idx = feat.index.intersection(lab.index)
        if len(common_idx) < 10:
            logger.warning("gnn_skip_symbol", symbol=symbol, rows=len(common_idx))
            continue

        feat_aligned = feat.loc[common_idx]
        lab_aligned = lab.loc[common_idx]

        all_X.append(feat_aligned)
        all_y.append(lab_aligned)

    if not all_X:
        raise InsufficientDataError("No symbols produced usable data for GNN training")

    X_combined = pd.concat(all_X, ignore_index=True)
    y_combined = pd.concat(all_y, ignore_index=True)

    X_combined = X_combined.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    X_train, y_train, X_val, y_val = _temporal_split(X_combined, y_combined)

    logger.info(
        "gnn_data_prepared",
        n_train=len(X_train),
        n_val=len(X_val),
        n_features=X_combined.shape[1],
        n_edges=graph.edge_index.shape[1],
    )

    model = GNNModel(config=config)
    train_result = model.fit(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        graph=graph,
    )

    save_dir = output_dir or (ARTIFACTS_DIR / "gnn" / "latest")
    model.save(save_dir)

    elapsed = time.perf_counter() - start
    logger.info(
        "gnn_pipeline_complete",
        training_time_s=round(elapsed, 2),
        train_accuracy=train_result.train_metrics.get("accuracy"),
        val_accuracy=train_result.val_metrics.get("accuracy"),
        artifact_path=str(save_dir),
    )

    return train_result
