"""Data quality checks: completeness and freshness for AlphaVedha market data."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta, timezone

import structlog
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from alphavedha.data.models import DailyOHLCV
from alphavedha.data.models import DataQualityReport as DQRModel

logger = structlog.get_logger(__name__)

IST = timezone(timedelta(hours=5, minutes=30))
FRESHNESS_CRITICAL_HOURS = 26  # more than 1 trading day stale
COMPLETENESS_WARNING_PCT = 0.90
COMPLETENESS_CRITICAL_PCT = 0.80


@dataclass
class QualityResult:
    check_type: str  # "completeness" | "freshness" | "consistency" | "anomaly"
    passed: bool
    severity: str  # "warning" | "critical"
    detail: str
    symbol: str | None = None


@dataclass
class QualityReport:
    report_date: date
    results: list[QualityResult] = field(default_factory=list)

    @property
    def n_passed(self) -> int:
        return sum(1 for r in self.results if r.passed)

    @property
    def n_warnings(self) -> int:
        return sum(1 for r in self.results if not r.passed and r.severity == "warning")

    @property
    def n_critical(self) -> int:
        return sum(1 for r in self.results if not r.passed and r.severity == "critical")


class QualityChecker:
    def __init__(self, session: AsyncSession, universe_size: int = 200) -> None:
        self._session = session
        self._universe_size = universe_size

    async def check_completeness(self, report_date: date) -> list[QualityResult]:
        """Count symbols with OHLCV data on report_date; warn/critical if below threshold."""
        result = await self._session.execute(
            select(func.count(func.distinct(DailyOHLCV.symbol))).where(
                DailyOHLCV.date == report_date
            )
        )
        count = result.scalar() or 0
        pct = count / self._universe_size if self._universe_size > 0 else 1.0
        passed = pct >= COMPLETENESS_WARNING_PCT
        severity = "critical" if pct < COMPLETENESS_CRITICAL_PCT else "warning"
        detail = f"{count}/{self._universe_size} symbols have data on {report_date} ({pct:.1%})"
        logger.info(
            "quality.completeness_checked",
            date=str(report_date),
            count=count,
            pct=f"{pct:.1%}",
            passed=passed,
        )
        return [
            QualityResult(
                check_type="completeness",
                passed=passed,
                severity=severity if not passed else "ok",
                detail=detail,
            )
        ]

    async def check_freshness(self) -> list[QualityResult]:
        """Check when the most recent DailyOHLCV record was created."""
        result = await self._session.execute(select(func.max(DailyOHLCV.created_at)))
        last_update: datetime | None = result.scalar()
        now_utc = datetime.now(UTC).replace(tzinfo=None)
        if last_update is None:
            logger.warning("quality.freshness_no_data")
            return [
                QualityResult(
                    check_type="freshness",
                    passed=False,
                    severity="critical",
                    detail="No OHLCV data found at all",
                )
            ]
        age_hours = (now_utc - last_update).total_seconds() / 3600
        passed = age_hours <= FRESHNESS_CRITICAL_HOURS
        severity = "critical" if age_hours > FRESHNESS_CRITICAL_HOURS else "warning"
        detail = f"Last OHLCV update {age_hours:.1f}h ago ({last_update.isoformat()})"
        logger.info("quality.freshness_checked", age_hours=f"{age_hours:.1f}", passed=passed)
        return [
            QualityResult(
                check_type="freshness",
                passed=passed,
                severity=severity if not passed else "ok",
                detail=detail,
            )
        ]

    async def check_consistency(self, report_date: date) -> list[QualityResult]:
        """Flag OHLCV rows where high < low."""
        rows = (
            await self._session.execute(
                select(DailyOHLCV.symbol, DailyOHLCV.date).where(
                    DailyOHLCV.date == report_date,
                    DailyOHLCV.high < DailyOHLCV.low,
                )
            )
        ).fetchall()
        if not rows:
            return [
                QualityResult(
                    check_type="consistency",
                    passed=True,
                    severity="ok",
                    detail=f"No OHLCV violations on {report_date}",
                )
            ]
        results = [
            QualityResult(
                check_type="consistency",
                passed=False,
                severity="critical",
                detail=f"OHLCV violation (high < low): {row.symbol} on {row.date}",
                symbol=row.symbol,
            )
            for row in rows
        ]
        logger.info("quality.consistency_checked", date=str(report_date), violations=len(rows))
        return results

    async def check_anomalies(self, report_date: date) -> list[QualityResult]:
        """Flag symbols with zero volume on a trading day."""
        rows = (
            await self._session.execute(
                select(DailyOHLCV.symbol, DailyOHLCV.date).where(
                    DailyOHLCV.date == report_date,
                    DailyOHLCV.volume == 0,
                )
            )
        ).fetchall()
        if not rows:
            return [
                QualityResult(
                    check_type="anomaly",
                    passed=True,
                    severity="ok",
                    detail=f"No zero-volume anomalies on {report_date}",
                )
            ]
        results = [
            QualityResult(
                check_type="anomaly",
                passed=False,
                severity="warning",
                detail=f"Zero volume: {row.symbol} on {row.date}",
                symbol=row.symbol,
            )
            for row in rows
        ]
        logger.info("quality.anomaly_checked", date=str(report_date), anomalies=len(rows))
        return results

    async def run_full_check(self, report_date: date) -> QualityReport:
        completeness = await self.check_completeness(report_date)
        freshness = await self.check_freshness()
        consistency = await self.check_consistency(report_date)
        anomalies = await self.check_anomalies(report_date)
        report = QualityReport(
            report_date=report_date,
            results=completeness + freshness + consistency + anomalies,
        )
        logger.info(
            "quality.full_check_complete",
            date=str(report_date),
            passed=report.n_passed,
            warnings=report.n_warnings,
            critical=report.n_critical,
        )
        return report

    async def persist_report(self, report: QualityReport) -> None:
        """Write each QualityResult to the data_quality_reports table."""
        for r in report.results:
            row = DQRModel(
                symbol=r.symbol,
                report_date=report.report_date,
                check_type=r.check_type,
                passed=r.passed,
                severity=r.severity,
                detail=r.detail,
            )
            self._session.add(row)
        await self._session.commit()
        logger.info(
            "quality.report_persisted",
            date=str(report.report_date),
            rows=len(report.results),
        )
