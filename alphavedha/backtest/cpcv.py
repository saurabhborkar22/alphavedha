"""Combinatorial Purged Cross-Validation (CPCV) — rigorous time-series model validation."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd
import structlog
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from alphavedha.config import AcceptanceConfig, CPCVConfig
from alphavedha.models.base import BaseModel

logger = structlog.get_logger(__name__)


@dataclass
class PathResult:
    path_id: int
    test_segments: tuple[int, ...]
    accuracy: float
    precision_weighted: float
    recall_weighted: float
    f1_weighted: float
    sharpe_ratio: float
    total_return: float
    n_test_samples: int
    confusion_matrix: np.ndarray = field(default_factory=lambda: np.array([]))


@dataclass
class CPCVResult:
    path_results: list[PathResult]
    n_paths: int
    median_sharpe: float
    worst_sharpe: float
    best_sharpe: float
    mean_accuracy: float
    std_accuracy: float
    passed: bool


def generate_cpcv_splits(
    n_samples: int,
    config: CPCVConfig,
) -> list[tuple[np.ndarray, np.ndarray, tuple[int, ...]]]:
    seg_size = n_samples // config.n_segments
    segment_ranges: list[tuple[int, int]] = []
    for s in range(config.n_segments):
        start = s * seg_size
        end = (s + 1) * seg_size if s < config.n_segments - 1 else n_samples
        segment_ranges.append((start, end))

    splits: list[tuple[np.ndarray, np.ndarray, tuple[int, ...]]] = []

    for test_combo in combinations(range(config.n_segments), config.k_test_segments):
        test_indices: list[int] = []
        for seg in test_combo:
            s, e = segment_ranges[seg]
            test_indices.extend(range(s, e))

        excluded = set(test_indices)

        for seg in test_combo:
            seg_start, seg_end = segment_ranges[seg]

            purge_start = max(0, seg_start - config.purge_days)
            for i in range(purge_start, seg_start):
                excluded.add(i)

            embargo_end = min(n_samples, seg_end + config.embargo_days)
            for i in range(seg_end, embargo_end):
                excluded.add(i)

        train_indices = [i for i in range(n_samples) if i not in excluded]
        splits.append(
            (
                np.array(train_indices),
                np.array(test_indices),
                test_combo,
            )
        )

    return splits


def _compute_sharpe(returns: np.ndarray) -> float:
    if len(returns) < 2 or np.std(returns) == 0:
        return 0.0
    return float(np.mean(returns) / np.std(returns) * np.sqrt(252))


def run_cpcv(
    X: pd.DataFrame,
    y: pd.Series,
    returns: pd.Series,
    sample_weight: pd.Series | None,
    model_factory: Callable[[], Any],
    config: CPCVConfig,
    acceptance: AcceptanceConfig,
) -> CPCVResult:
    splits = generate_cpcv_splits(len(X), config)
    path_results: list[PathResult] = []

    for path_id, (train_idx, test_idx, test_segs) in enumerate(splits):
        model: BaseModel = model_factory()

        X_train = X.iloc[train_idx]
        y_train = y.iloc[train_idx]
        ret_train = returns.iloc[train_idx]
        X_test = X.iloc[test_idx]
        y_test = y.iloc[test_idx]
        ret_test = returns.iloc[test_idx]

        sw_train = sample_weight.iloc[train_idx] if sample_weight is not None else None

        val_size = max(int(len(X_train) * 0.15), 20)
        X_tr = X_train.iloc[:-val_size]
        y_tr = y_train.iloc[:-val_size]
        ret_tr = ret_train.iloc[:-val_size]
        X_vl = X_train.iloc[-val_size:]
        y_vl = y_train.iloc[-val_size:]
        ret_vl = ret_train.iloc[-val_size:]
        sw_tr = sw_train.iloc[:-val_size] if sw_train is not None else None

        model.fit(
            X_train=X_tr,
            y_train=y_tr,
            X_val=X_vl,
            y_val=y_vl,
            sample_weight=sw_tr,
            return_train=ret_tr,
            return_val=ret_vl,
        )

        pred = model.predict(X_test)
        y_test_mapped = y_test.values
        pred_dir = pred.direction

        y_test_labels = y_test_mapped.astype(int)
        pred_labels = pred_dir.astype(int)

        acc = float(accuracy_score(y_test_labels, pred_labels))
        prec = float(
            precision_score(
                y_test_labels,
                pred_labels,
                average="weighted",
                zero_division=0,
            )
        )
        rec = float(
            recall_score(
                y_test_labels,
                pred_labels,
                average="weighted",
                zero_division=0,
            )
        )
        f1 = float(
            f1_score(
                y_test_labels,
                pred_labels,
                average="weighted",
                zero_division=0,
            )
        )

        strategy_returns = pred_dir * ret_test.values
        sharpe = _compute_sharpe(strategy_returns)
        total_ret = float(np.sum(strategy_returns))

        path_results.append(
            PathResult(
                path_id=path_id,
                test_segments=test_segs,
                accuracy=acc,
                precision_weighted=prec,
                recall_weighted=rec,
                f1_weighted=f1,
                sharpe_ratio=sharpe,
                total_return=total_ret,
                n_test_samples=len(test_idx),
            )
        )

        logger.debug(
            "cpcv_path_completed",
            path_id=path_id,
            test_segments=test_segs,
            accuracy=round(acc, 4),
            sharpe=round(sharpe, 4),
        )

    sharpes = [pr.sharpe_ratio for pr in path_results]
    accuracies = [pr.accuracy for pr in path_results]

    median_sharpe = float(np.median(sharpes))
    worst_sharpe = float(np.min(sharpes))
    best_sharpe = float(np.max(sharpes))
    mean_acc = float(np.mean(accuracies))
    std_acc = float(np.std(accuracies))

    passed = (
        median_sharpe >= acceptance.min_median_sharpe
        and worst_sharpe >= acceptance.min_worst_sharpe
    )

    logger.info(
        "cpcv_completed",
        n_paths=len(path_results),
        median_sharpe=round(median_sharpe, 4),
        worst_sharpe=round(worst_sharpe, 4),
        mean_accuracy=round(mean_acc, 4),
        passed=passed,
    )

    return CPCVResult(
        path_results=path_results,
        n_paths=len(path_results),
        median_sharpe=median_sharpe,
        worst_sharpe=worst_sharpe,
        best_sharpe=best_sharpe,
        mean_accuracy=mean_acc,
        std_accuracy=std_acc,
        passed=passed,
    )
