"""Intel data quality checks — row-count anomalies and disk monitoring.

Integrates with the existing QualityChecker framework. Checks that each
intel collector produced rows today (weekdays only), and monitors disk
usage on the VPS.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import structlog
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from alphavedha.data.models import (
    BulkBlockDeal,
    Disclosure,
    RatingEvent,
    SurveillanceFlag,
    Transcript,
)

logger = structlog.get_logger(__name__)

IST = ZoneInfo("Asia/Kolkata")

DISK_WARNING_PCT = 0.70
DISK_CRITICAL_PCT = 0.85


@dataclass
class IntelCheckResult:
    table: str
    passed: bool
    severity: str
    detail: str
    row_count: int = 0


@dataclass
class DiskCheckResult:
    passed: bool
    severity: str
    detail: str
    used_pct: float = 0.0


_INTEL_TABLES: list[tuple[str, Any, str]] = [
    ("disclosures", Disclosure, "filed_at"),
    ("surveillance_flags", SurveillanceFlag, "added_on"),
    ("bulk_block_deals", BulkBlockDeal, "deal_date"),
    ("rating_events", RatingEvent, "filed_at"),
    ("transcripts", Transcript, "filed_at"),
]


async def check_intel_row_counts(
    session: AsyncSession,
    check_date: date | None = None,
    lookback_days: int = 1,
) -> list[IntelCheckResult]:
    """Check that each intel table received rows recently.

    For weekdays, expects at least some rows in the lookback window.
    Weekends are skipped (returns all-pass).
    """
    if check_date is None:
        check_date = date.today()

    if check_date.weekday() >= 5:
        return [
            IntelCheckResult(
                table="intel_tables",
                passed=True,
                severity="ok",
                detail="Weekend — intel checks skipped",
            )
        ]

    cutoff = datetime(
        check_date.year,
        check_date.month,
        check_date.day,
        tzinfo=IST,
    ) - timedelta(days=lookback_days)

    results: list[IntelCheckResult] = []

    for table_name, model, date_col in _INTEL_TABLES:
        col = getattr(model, date_col)
        stmt = select(func.count()).where(col >= cutoff)

        try:
            row = await session.execute(stmt)
            count = row.scalar() or 0
        except Exception as e:
            logger.warning("intel_quality_check_error", table=table_name, error=str(e))
            count = 0

        if table_name == "transcripts":
            passed = True
            severity = "ok"
            detail = f"{table_name}: {count} rows (transcripts may be sparse)"
        elif count == 0:
            passed = False
            severity = "warning"
            detail = f"{table_name}: 0 rows since {cutoff.date()}"
        else:
            passed = True
            severity = "ok"
            detail = f"{table_name}: {count} rows since {cutoff.date()}"

        results.append(
            IntelCheckResult(
                table=table_name,
                passed=passed,
                severity=severity,
                row_count=count,
                detail=detail,
            )
        )

    logger.info(
        "intel_quality_check_complete",
        date=str(check_date),
        checks=len(results),
        passed=sum(1 for r in results if r.passed),
        failed=sum(1 for r in results if not r.passed),
    )
    return results


def check_disk_usage(mount_path: str = "/") -> DiskCheckResult:
    """Check disk usage on the given mount point."""
    try:
        usage = shutil.disk_usage(mount_path)
        used_pct = usage.used / usage.total
        free_gb = usage.free / (1024**3)

        if used_pct >= DISK_CRITICAL_PCT:
            return DiskCheckResult(
                passed=False,
                severity="critical",
                detail=f"Disk {used_pct:.0%} full ({free_gb:.1f} GB free)",
                used_pct=used_pct,
            )
        if used_pct >= DISK_WARNING_PCT:
            return DiskCheckResult(
                passed=False,
                severity="warning",
                detail=f"Disk {used_pct:.0%} full ({free_gb:.1f} GB free)",
                used_pct=used_pct,
            )
        return DiskCheckResult(
            passed=True,
            severity="ok",
            detail=f"Disk {used_pct:.0%} full ({free_gb:.1f} GB free)",
            used_pct=used_pct,
        )
    except Exception as e:
        logger.error("disk_check_failed", error=str(e))
        return DiskCheckResult(
            passed=False,
            severity="critical",
            detail=f"Disk check failed: {e}",
        )


def cleanup_old_pdfs(
    base_dir: Path,
    max_age_days: int = 30,
    dry_run: bool = True,
) -> int:
    """Remove PDF files older than max_age_days.

    Returns count of files removed (or that would be removed in dry_run).
    """
    if not base_dir.exists():
        return 0

    cutoff = datetime.now(IST) - timedelta(days=max_age_days)
    removed = 0

    for pdf_path in base_dir.rglob("*.pdf"):
        try:
            mtime = datetime.fromtimestamp(pdf_path.stat().st_mtime, tz=IST)
            if mtime < cutoff:
                if not dry_run:
                    pdf_path.unlink()
                removed += 1
        except OSError:
            continue

    logger.info(
        "pdf_cleanup_complete",
        base_dir=str(base_dir),
        removed=removed,
        dry_run=dry_run,
        max_age_days=max_age_days,
    )
    return removed
