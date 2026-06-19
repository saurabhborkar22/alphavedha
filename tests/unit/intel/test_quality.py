"""Tests for intel quality checks and disk monitoring."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from alphavedha.intel.quality import (
    DISK_CRITICAL_PCT,
    DISK_WARNING_PCT,
    DiskCheckResult,
    IntelCheckResult,
    check_disk_usage,
    cleanup_old_pdfs,
)


class TestDiskUsage:
    def test_healthy_disk(self) -> None:
        result = check_disk_usage("/")
        assert isinstance(result, DiskCheckResult)
        assert result.used_pct > 0

    def test_warning_threshold(self) -> None:
        assert DISK_WARNING_PCT == 0.70

    def test_critical_threshold(self) -> None:
        assert DISK_CRITICAL_PCT == 0.85

    def test_returns_detail_string(self) -> None:
        result = check_disk_usage("/")
        assert "Disk" in result.detail
        assert "GB free" in result.detail

    @patch("alphavedha.intel.quality.shutil.disk_usage")
    def test_warning_at_threshold(self, mock_usage: object) -> None:
        from unittest.mock import MagicMock

        mock = MagicMock()
        mock.total = 100
        mock.used = 75
        mock.free = 25
        assert isinstance(mock_usage, MagicMock)
        mock_usage.return_value = mock

        result = check_disk_usage("/")
        assert not result.passed
        assert result.severity == "warning"

    @patch("alphavedha.intel.quality.shutil.disk_usage")
    def test_critical_at_threshold(self, mock_usage: object) -> None:
        from unittest.mock import MagicMock

        mock = MagicMock()
        mock.total = 100
        mock.used = 90
        mock.free = 10
        assert isinstance(mock_usage, MagicMock)
        mock_usage.return_value = mock

        result = check_disk_usage("/")
        assert not result.passed
        assert result.severity == "critical"

    @patch("alphavedha.intel.quality.shutil.disk_usage", side_effect=OSError("fail"))
    def test_handles_os_error(self, _mock: object) -> None:
        result = check_disk_usage("/nonexistent")
        assert not result.passed
        assert result.severity == "critical"


class TestCleanupOldPdfs:
    def test_returns_zero_for_nonexistent_dir(self) -> None:
        assert cleanup_old_pdfs(Path("/nonexistent/path"), dry_run=True) == 0

    def test_dry_run_does_not_delete(self, tmp_path: Path) -> None:
        pdf = tmp_path / "test.pdf"
        pdf.write_bytes(b"%PDF-1.4 test")
        import os
        import time

        old_time = time.time() - 60 * 60 * 24 * 60
        os.utime(pdf, (old_time, old_time))

        removed = cleanup_old_pdfs(tmp_path, max_age_days=30, dry_run=True)
        assert removed == 1
        assert pdf.exists()

    def test_actual_delete(self, tmp_path: Path) -> None:
        pdf = tmp_path / "old.pdf"
        pdf.write_bytes(b"%PDF-1.4 test")
        import os
        import time

        old_time = time.time() - 60 * 60 * 24 * 60
        os.utime(pdf, (old_time, old_time))

        removed = cleanup_old_pdfs(tmp_path, max_age_days=30, dry_run=False)
        assert removed == 1
        assert not pdf.exists()

    def test_keeps_recent_files(self, tmp_path: Path) -> None:
        pdf = tmp_path / "recent.pdf"
        pdf.write_bytes(b"%PDF-1.4 test")

        removed = cleanup_old_pdfs(tmp_path, max_age_days=30, dry_run=True)
        assert removed == 0


class TestIntelCheckResult:
    def test_dataclass_fields(self) -> None:
        r = IntelCheckResult(
            table="disclosures",
            passed=True,
            severity="ok",
            detail="5 rows",
            row_count=5,
        )
        assert r.table == "disclosures"
        assert r.passed is True
        assert r.row_count == 5


class TestJobHealthSummary:
    def test_empty_history(self) -> None:
        from alphavedha.scheduler import AlphaVedhaScheduler

        sched = AlphaVedhaScheduler(demo=True)
        summary = sched.job_health_summary()
        assert summary == {"jobs": []}

    def test_records_last_per_job(self) -> None:
        from alphavedha.scheduler import AlphaVedhaScheduler, JobResult, _now_ist

        sched = AlphaVedhaScheduler(demo=True)
        sched._record_job(JobResult(job_name="test_job", started_at=_now_ist(), success=True))
        sched._record_job(
            JobResult(job_name="test_job", started_at=_now_ist(), success=False, error="fail")
        )
        summary = sched.job_health_summary()
        assert len(summary["jobs"]) == 1
        assert summary["jobs"][0]["name"] == "test_job"
        assert summary["jobs"][0]["success"] is False
        assert summary["jobs"][0]["error"] == "fail"

    def test_multiple_jobs(self) -> None:
        from alphavedha.scheduler import AlphaVedhaScheduler, JobResult, _now_ist

        sched = AlphaVedhaScheduler(demo=True)
        sched._record_job(JobResult(job_name="job_a", started_at=_now_ist(), success=True))
        sched._record_job(JobResult(job_name="job_b", started_at=_now_ist(), success=True))
        summary = sched.job_health_summary()
        assert len(summary["jobs"]) == 2
        names = {j["name"] for j in summary["jobs"]}
        assert names == {"job_a", "job_b"}
