"""Sample weighting — uniqueness and recency weights for overlapping barrier labels."""

from __future__ import annotations

import numpy as np
import pandas as pd
import structlog

from alphavedha.config import SampleWeightsConfig

logger = structlog.get_logger(__name__)


def _compute_uniqueness_weights(labels_df: pd.DataFrame) -> np.ndarray:
    """Weight = mean(1 / concurrency) over each sample's active window."""
    n = len(labels_df)
    days_to_hit = labels_df["days_to_hit"].fillna(1).astype(int).values
    label_vals = labels_df["label"].values
    concurrency = np.zeros(n, dtype=float)

    for i in range(n):
        if pd.isna(label_vals[i]):
            continue
        end = min(i + days_to_hit[i], n)
        for j in range(i, end):
            concurrency[j] += 1

    concurrency = np.maximum(concurrency, 1.0)

    weights = np.ones(n, dtype=float)
    for i in range(n):
        if pd.isna(label_vals[i]):
            weights[i] = 1.0
            continue
        end = min(i + days_to_hit[i], n)
        window_concurrency = concurrency[i:end]
        weights[i] = float(np.mean(1.0 / window_concurrency))

    return weights


def _compute_recency_weights(index: pd.DatetimeIndex, halflife: int) -> np.ndarray:
    """Exponential decay from most recent timestamp."""
    positions = np.arange(len(index), dtype=float)
    last_pos = positions[-1]
    decay = np.exp(-np.log(2) * (last_pos - positions) / halflife)
    return decay


def compute_sample_weights(
    labels_df: pd.DataFrame,
    config: SampleWeightsConfig,
) -> pd.Series:
    n = len(labels_df)

    if config.uniqueness:
        uniqueness = _compute_uniqueness_weights(labels_df)
    else:
        uniqueness = np.ones(n, dtype=float)

    recency = _compute_recency_weights(labels_df.index, config.recency_halflife)

    combined = uniqueness * recency
    combined = combined * (n / combined.sum())

    logger.info(
        "sample_weights_computed",
        n_samples=n,
        uniqueness_enabled=config.uniqueness,
        halflife=config.recency_halflife,
        weight_min=round(float(combined.min()), 4),
        weight_max=round(float(combined.max()), 4),
    )

    return pd.Series(combined, index=labels_df.index, name="sample_weight")
