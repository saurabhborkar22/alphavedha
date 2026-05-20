"""Auto retraining manager — version lifecycle, promotion, and retrain decisions."""

from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from pathlib import Path

import structlog

from alphavedha.config import RetrainingConfig
from alphavedha.monitoring.drift import DriftReport
from alphavedha.monitoring.performance import PerformanceReport

logger = structlog.get_logger(__name__)

_SCHEDULE_INTERVALS: dict[str, int] = {
    "daily": 1,
    "weekly": 7,
    "biweekly": 14,
    "monthly": 30,
}


@dataclass
class ModelVersion:
    version: str
    created_at: datetime
    status: str
    metrics: dict[str, float] = field(default_factory=dict)
    data_range: tuple[str, str] = ("", "")
    artifact_path: Path = field(default_factory=lambda: Path("."))


@dataclass
class RetrainDecision:
    should_retrain: bool
    reason: str
    drift_report: DriftReport | None = None
    performance_report: PerformanceReport | None = None


class RetrainingManager:
    def __init__(
        self,
        config: RetrainingConfig | None = None,
        artifact_dir: Path | None = None,
    ) -> None:
        self._config = config or RetrainingConfig()
        self._artifact_dir = artifact_dir or Path("models/artifacts")
        self._versions: list[ModelVersion] = []

    def should_retrain(
        self,
        drift_report: DriftReport | None = None,
        performance_report: PerformanceReport | None = None,
        last_train_date: date | None = None,
    ) -> RetrainDecision:
        """Decide if retraining is needed based on drift, performance, or schedule."""
        if drift_report is not None and drift_report.requires_retrain:
            logger.info(
                "retrain_triggered_by_drift",
                n_alerts=len(drift_report.alerts),
            )
            return RetrainDecision(
                should_retrain=True,
                reason="drift_detected",
                drift_report=drift_report,
                performance_report=performance_report,
            )

        if performance_report is not None and performance_report.requires_retrain:
            logger.info("retrain_triggered_by_performance")
            return RetrainDecision(
                should_retrain=True,
                reason="performance_degraded",
                drift_report=drift_report,
                performance_report=performance_report,
            )

        if last_train_date is not None:
            interval_days = _SCHEDULE_INTERVALS.get(
                self._config.schedule, 7
            )
            days_since = (date.today() - last_train_date).days
            if days_since >= interval_days:
                logger.info(
                    "retrain_triggered_by_schedule",
                    days_since_last_train=days_since,
                    schedule=self._config.schedule,
                )
                return RetrainDecision(
                    should_retrain=True,
                    reason="scheduled",
                    drift_report=drift_report,
                    performance_report=performance_report,
                )

        return RetrainDecision(
            should_retrain=False,
            reason="no_trigger",
            drift_report=drift_report,
            performance_report=performance_report,
        )

    def register_version(
        self,
        version: str,
        metrics: dict[str, float],
        artifact_path: Path,
        data_range: tuple[str, str],
    ) -> ModelVersion:
        """Register a new model version as shadow."""
        model_version = ModelVersion(
            version=version,
            created_at=datetime.now(UTC),
            status="shadow",
            metrics=metrics,
            data_range=data_range,
            artifact_path=artifact_path,
        )
        self._versions.append(model_version)
        logger.info(
            "model_version_registered",
            version=version,
            status="shadow",
            artifact_path=str(artifact_path),
        )
        return model_version

    def promote_version(self, version: str) -> ModelVersion:
        """Promote shadow to active, retire previous active."""
        target: ModelVersion | None = None
        for v in self._versions:
            if v.version == version:
                target = v
                break

        if target is None:
            msg = f"Version {version} not found"
            raise ValueError(msg)

        if target.status != "shadow":
            msg = f"Version {version} is {target.status}, can only promote shadow versions"
            raise ValueError(msg)

        for v in self._versions:
            if v.status == "active":
                v.status = "retired"
                logger.info("model_version_retired", version=v.version)

        target.status = "active"
        logger.info("model_version_promoted", version=version)
        return target

    def get_active_version(self) -> ModelVersion | None:
        """Return the currently active model version."""
        for v in reversed(self._versions):
            if v.status == "active":
                return v
        return None

    def cleanup_old_versions(self) -> list[str]:
        """Remove versions beyond keep_versions limit (keep active + N most recent)."""
        active = [v for v in self._versions if v.status == "active"]
        non_active = [v for v in self._versions if v.status != "active"]

        non_active.sort(key=lambda v: v.created_at, reverse=True)

        keep_count = self._config.keep_versions
        to_keep = non_active[:keep_count]
        to_remove = non_active[keep_count:]

        removed_versions: list[str] = []
        for v in to_remove:
            if v.artifact_path.exists():
                shutil.rmtree(v.artifact_path)
            removed_versions.append(v.version)
            logger.info(
                "model_version_cleaned_up",
                version=v.version,
                path=str(v.artifact_path),
            )

        self._versions = active + to_keep
        return removed_versions

    def get_version_history(self) -> list[ModelVersion]:
        """Return all versions sorted by creation date."""
        return sorted(self._versions, key=lambda v: v.created_at)
