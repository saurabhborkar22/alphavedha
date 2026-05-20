"""Feature drift detection — PSI and KS tests for distribution shift monitoring."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime

import numpy as np
import pandas as pd
import structlog
from scipy.stats import ks_2samp

from alphavedha.config import DriftConfig

logger = structlog.get_logger(__name__)

_EPSILON = 1e-6


@dataclass
class DriftResult:
    feature_name: str
    psi_value: float
    ks_statistic: float
    ks_pvalue: float
    is_warning: bool
    is_alert: bool


@dataclass
class DriftReport:
    timestamp: datetime
    features_checked: int
    warnings: list[DriftResult] = field(default_factory=list)
    alerts: list[DriftResult] = field(default_factory=list)
    overall_psi: float = 0.0
    requires_retrain: bool = False


class DriftDetector:
    def __init__(self, config: DriftConfig | None = None) -> None:
        self._config = config or DriftConfig()

    def compute_psi(
        self, reference: np.ndarray, current: np.ndarray, n_bins: int = 10
    ) -> float:
        """Population Stability Index between two distributions.

        PSI = sum((current_pct - reference_pct) * ln(current_pct / reference_pct))
        Bins using reference distribution quantiles.
        """
        reference = reference[~np.isnan(reference)]
        current = current[~np.isnan(current)]

        if len(reference) == 0 or len(current) == 0:
            return 0.0

        quantiles = np.linspace(0, 100, n_bins + 1)
        bin_edges = np.percentile(reference, quantiles)
        bin_edges[0] = -np.inf
        bin_edges[-1] = np.inf
        bin_edges = np.unique(bin_edges)

        ref_counts = np.histogram(reference, bins=bin_edges)[0].astype(float)
        cur_counts = np.histogram(current, bins=bin_edges)[0].astype(float)

        ref_pct = ref_counts / ref_counts.sum()
        cur_pct = cur_counts / cur_counts.sum()

        ref_pct = np.where(ref_pct < _EPSILON, _EPSILON, ref_pct)
        cur_pct = np.where(cur_pct < _EPSILON, _EPSILON, cur_pct)

        psi = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))
        return float(psi)

    def compute_ks(
        self, reference: np.ndarray, current: np.ndarray
    ) -> tuple[float, float]:
        """Kolmogorov-Smirnov test. Returns (statistic, p-value)."""
        reference = reference[~np.isnan(reference)]
        current = current[~np.isnan(current)]

        if len(reference) == 0 or len(current) == 0:
            return 0.0, 1.0

        stat, pvalue = ks_2samp(reference, current)
        return float(stat), float(pvalue)

    def check_drift(
        self,
        reference_df: pd.DataFrame,
        current_df: pd.DataFrame,
    ) -> DriftReport:
        """Check all numeric features for drift."""
        numeric_cols = reference_df.select_dtypes(include=[np.number]).columns
        common_cols = numeric_cols.intersection(
            current_df.select_dtypes(include=[np.number]).columns
        )

        if len(common_cols) == 0:
            logger.warning("drift_check_no_common_columns")
            return DriftReport(
                timestamp=datetime.now(UTC),
                features_checked=0,
            )

        results: list[DriftResult] = []
        for col in common_cols:
            ref_vals = reference_df[col].to_numpy()
            cur_vals = current_df[col].to_numpy()

            psi = self.compute_psi(ref_vals, cur_vals)
            ks_stat, ks_pvalue = self.compute_ks(ref_vals, cur_vals)

            result = DriftResult(
                feature_name=col,
                psi_value=psi,
                ks_statistic=ks_stat,
                ks_pvalue=ks_pvalue,
                is_warning=psi > self._config.psi_warning,
                is_alert=psi > self._config.psi_alert,
            )
            results.append(result)

        warnings = [r for r in results if r.is_warning]
        alerts = [r for r in results if r.is_alert]
        psi_values = [r.psi_value for r in results]
        overall_psi = float(np.mean(psi_values)) if psi_values else 0.0
        requires_retrain = len(alerts) > 0

        if warnings:
            logger.warning(
                "drift_warnings_detected",
                n_warnings=len(warnings),
                features=[w.feature_name for w in warnings],
            )
        if alerts:
            logger.error(
                "drift_alerts_detected",
                n_alerts=len(alerts),
                features=[a.feature_name for a in alerts],
                requires_retrain=True,
            )

        return DriftReport(
            timestamp=datetime.now(UTC),
            features_checked=len(results),
            warnings=warnings,
            alerts=alerts,
            overall_psi=overall_psi,
            requires_retrain=requires_retrain,
        )
