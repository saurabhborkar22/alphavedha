"""Data quality checks: completeness and freshness for AlphaVedha market data."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone

import structlog
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from alphavedha.data.models import DailyOHLCV

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
        return [
            QualityResult(
                check_type="completeness",
                passed=passed,
                severity=severity if not passed else "warning",
                detail=detail,
            )
        ]

    async def check_freshness(self) -> list[QualityResult]:
        """Check when the most recent DailyOHLCV record was created."""
        result = await self._session.execute(select(func.max(DailyOHLCV.created_at)))
        last_update: datetime | None = result.scalar()
        now_ist = datetime.now(IST).replace(tzinfo=None)
        if last_update is None:
            return [
                QualityResult(
                    check_type="freshness",
                    passed=False,
                    severity="critical",
                    detail="No OHLCV data found at all",
                )
            ]
        age_hours = (now_ist - last_update).total_seconds() / 3600
        passed = age_hours <= FRESHNESS_CRITICAL_HOURS
        severity = "critical" if age_hours > FRESHNESS_CRITICAL_HOURS else "warning"
        detail = f"Last OHLCV update {age_hours:.1f}h ago ({last_update.isoformat()})"
        return [
            QualityResult(
                check_type="freshness",
                passed=passed,
                severity=severity if not passed else "warning",
                detail=detail,
            )
        ]
