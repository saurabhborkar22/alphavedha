from __future__ import annotations

from datetime import date, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from alphavedha.data.quality import QualityChecker, QualityReport, QualityResult


def test_quality_result_dataclass() -> None:
    r = QualityResult(
        check_type="completeness",
        passed=True,
        severity="warning",
        detail="ok",
        symbol="TCS.NS",
    )
    assert r.check_type == "completeness"
    assert r.passed is True
    assert r.symbol == "TCS.NS"


def test_quality_report_counts() -> None:
    results = [
        QualityResult("completeness", True, "warning", "ok"),
        QualityResult("freshness", False, "warning", "stale"),
        QualityResult("consistency", False, "critical", "gap"),
    ]
    report = QualityReport(report_date=date(2026, 5, 30), results=results)
    assert report.n_passed == 1
    assert report.n_warnings == 1
    assert report.n_critical == 1


def _make_session(scalar_value: object) -> AsyncMock:
    """Build an AsyncSession mock where execute() awaits to a MagicMock with .scalar() returning scalar_value."""
    mock_result = MagicMock()
    mock_result.scalar.return_value = scalar_value
    mock_session = AsyncMock()
    mock_session.execute.return_value = mock_result
    return mock_session


@pytest.mark.asyncio
async def test_check_completeness_returns_results() -> None:
    checker = QualityChecker(session=_make_session(45), universe_size=50)
    results = await checker.check_completeness(date(2026, 5, 30))
    assert len(results) >= 1
    assert all(r.check_type == "completeness" for r in results)


@pytest.mark.asyncio
async def test_check_completeness_warns_below_threshold() -> None:
    checker = QualityChecker(session=_make_session(44), universe_size=50)  # 88% < 90%
    results = await checker.check_completeness(date(2026, 5, 30))
    assert len(results) == 1
    assert results[0].passed is False
    assert results[0].severity == "warning"


@pytest.mark.asyncio
async def test_check_completeness_critical_below_80pct() -> None:
    checker = QualityChecker(session=_make_session(39), universe_size=50)  # 78% < 80%
    results = await checker.check_completeness(date(2026, 5, 30))
    assert results[0].passed is False
    assert results[0].severity == "critical"


@pytest.mark.asyncio
async def test_check_freshness_stale_triggers_critical() -> None:
    stale_dt = datetime(2026, 5, 27, 16, 0, 0)  # 3 days ago
    checker = QualityChecker(session=_make_session(stale_dt), universe_size=50)
    results = await checker.check_freshness()
    critical = [r for r in results if r.severity == "critical"]
    assert len(critical) >= 1


@pytest.mark.asyncio
async def test_check_freshness_no_data_is_critical() -> None:
    checker = QualityChecker(session=_make_session(None), universe_size=50)
    results = await checker.check_freshness()
    assert results[0].passed is False
    assert results[0].severity == "critical"
    assert "No OHLCV data" in results[0].detail


@pytest.mark.asyncio
async def test_check_completeness_passes_at_90pct() -> None:
    checker = QualityChecker(session=_make_session(45), universe_size=50)  # 90%
    results = await checker.check_completeness(date(2026, 5, 30))
    assert results[0].passed is True
    assert results[0].severity == "ok"


@pytest.mark.asyncio
async def test_check_consistency_detects_ohlcv_violation() -> None:
    mock_row = MagicMock()
    mock_row.symbol = "BAD.NS"
    mock_row.date = date(2026, 5, 30)
    session = _make_session(None)
    session.execute.return_value.fetchall.return_value = [mock_row]
    checker = QualityChecker(session=session, universe_size=50)
    results = await checker.check_consistency(date(2026, 5, 30))
    failed = [r for r in results if not r.passed]
    assert len(failed) >= 1
    assert "BAD.NS" in failed[0].detail


@pytest.mark.asyncio
async def test_check_consistency_passes_when_no_violations() -> None:
    session = _make_session(None)
    session.execute.return_value.fetchall.return_value = []
    checker = QualityChecker(session=session, universe_size=50)
    results = await checker.check_consistency(date(2026, 5, 30))
    assert results[0].passed is True
    assert results[0].severity == "ok"


@pytest.mark.asyncio
async def test_check_anomalies_flags_zero_volume() -> None:
    mock_row = MagicMock()
    mock_row.symbol = "ZERO.NS"
    mock_row.date = date(2026, 5, 30)
    session = _make_session(None)
    session.execute.return_value.fetchall.return_value = [mock_row]
    checker = QualityChecker(session=session, universe_size=50)
    results = await checker.check_anomalies(date(2026, 5, 30))
    assert any("ZERO.NS" in r.detail for r in results if not r.passed)


@pytest.mark.asyncio
async def test_run_full_check_aggregates_all() -> None:
    session = _make_session(48)
    session.execute.return_value.fetchall.return_value = []
    checker = QualityChecker(session=session, universe_size=50)
    with patch.object(
        checker, "check_freshness", return_value=[QualityResult("freshness", True, "ok", "fresh")]
    ):
        report = await checker.run_full_check(date(2026, 5, 30))
    assert isinstance(report, QualityReport)
    assert len(report.results) >= 3  # completeness + freshness + consistency + anomalies


@pytest.mark.asyncio
async def test_persist_report_writes_rows() -> None:
    session = AsyncMock()
    results = [
        QualityResult("completeness", True, "ok", "ok"),
        QualityResult("freshness", False, "critical", "stale"),
    ]
    report = QualityReport(report_date=date(2026, 5, 30), results=results)
    checker = QualityChecker(session=session, universe_size=50)
    await checker.persist_report(report)
    assert session.add.call_count == 2
    assert session.commit.called
