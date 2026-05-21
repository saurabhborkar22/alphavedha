"""Prediction performance tracking — rolling accuracy, precision, and alpha monitoring."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime

import pandas as pd
import structlog

from alphavedha.config import PerformanceMonitorConfig

logger = structlog.get_logger(__name__)


@dataclass
class PerformanceSnapshot:
    window_days: int
    accuracy: float
    precision_buy: float
    precision_sell: float
    magnitude_mae: float
    n_predictions: int
    profitable_pct: float


@dataclass
class PerformanceReport:
    timestamp: datetime
    model_version: str
    snapshots: dict[int, PerformanceSnapshot] = field(default_factory=dict)
    alpha_vs_benchmark: float = 0.0
    requires_retrain: bool = False


class PerformanceMonitor:
    def __init__(self, config: PerformanceMonitorConfig | None = None) -> None:
        self._config = config or PerformanceMonitorConfig()

    def _compute_snapshot(
        self, merged: pd.DataFrame, window_days: int
    ) -> PerformanceSnapshot | None:
        """Compute a single performance snapshot for a rolling window."""
        if merged.empty:
            return None

        max_date = merged["date"].max()
        cutoff = max_date - pd.Timedelta(days=window_days)
        window_df = merged[merged["date"] > cutoff]

        if window_df.empty:
            return None

        n_predictions = len(window_df)
        accuracy = float((window_df["predicted_direction"] == window_df["actual_direction"]).mean())

        buy_mask = window_df["predicted_direction"] == 1
        if buy_mask.any():
            precision_buy = float((window_df.loc[buy_mask, "actual_direction"] == 1).mean())
        else:
            precision_buy = 0.0

        sell_mask = window_df["predicted_direction"] == -1
        if sell_mask.any():
            precision_sell = float((window_df.loc[sell_mask, "actual_direction"] == -1).mean())
        else:
            precision_sell = 0.0

        magnitude_mae = float(
            (window_df["predicted_magnitude"] - window_df["actual_return"]).abs().mean()
        )

        tradeable = window_df[window_df["predicted_direction"] != 0]
        if not tradeable.empty:
            directional_profit = tradeable["predicted_direction"] * tradeable["actual_return"]
            profitable_pct = float((directional_profit > 0).mean())
        else:
            profitable_pct = 0.0

        return PerformanceSnapshot(
            window_days=window_days,
            accuracy=accuracy,
            precision_buy=precision_buy,
            precision_sell=precision_sell,
            magnitude_mae=magnitude_mae,
            n_predictions=n_predictions,
            profitable_pct=profitable_pct,
        )

    def evaluate(
        self,
        predictions: pd.DataFrame,
        actuals: pd.DataFrame,
        model_version: str = "unknown",
    ) -> PerformanceReport:
        """Evaluate prediction performance over rolling windows.

        predictions columns: date, symbol, predicted_direction, predicted_magnitude, confidence
        actuals columns: date, symbol, actual_direction, actual_return
        """
        if predictions.empty or actuals.empty:
            logger.warning("performance_evaluate_empty_input")
            return PerformanceReport(
                timestamp=datetime.now(UTC),
                model_version=model_version,
            )

        merged = predictions.merge(actuals, on=["date", "symbol"], how="inner")

        if merged.empty:
            logger.warning("performance_evaluate_no_matching_predictions")
            return PerformanceReport(
                timestamp=datetime.now(UTC),
                model_version=model_version,
            )

        snapshots: dict[int, PerformanceSnapshot] = {}
        requires_retrain = False

        for window in self._config.rolling_windows:
            snapshot = self._compute_snapshot(merged, window)
            if snapshot is not None:
                snapshots[window] = snapshot
                if snapshot.accuracy < self._config.min_accuracy:
                    requires_retrain = True
                    logger.warning(
                        "performance_below_threshold",
                        window_days=window,
                        accuracy=snapshot.accuracy,
                        threshold=self._config.min_accuracy,
                    )

        alpha_vs_benchmark = 0.0
        benchmark_window = 30
        if benchmark_window in snapshots:
            benchmark_return = merged.loc[
                merged["date"] > merged["date"].max() - pd.Timedelta(days=benchmark_window),
                "actual_return",
            ].mean()
            strategy_return = (
                merged.loc[
                    merged["date"] > merged["date"].max() - pd.Timedelta(days=benchmark_window),
                ]
                .apply(
                    lambda row: (
                        row["actual_return"]
                        if row["predicted_direction"] == 1
                        else -row["actual_return"]
                        if row["predicted_direction"] == -1
                        else 0.0
                    ),
                    axis=1,
                )
                .mean()
            )
            alpha_vs_benchmark = float(strategy_return - benchmark_return)

        report = PerformanceReport(
            timestamp=datetime.now(UTC),
            model_version=model_version,
            snapshots=snapshots,
            alpha_vs_benchmark=alpha_vs_benchmark,
            requires_retrain=requires_retrain,
        )

        logger.info(
            "performance_evaluated",
            model_version=model_version,
            n_windows=len(snapshots),
            requires_retrain=requires_retrain,
        )

        return report
