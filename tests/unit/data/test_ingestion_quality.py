from __future__ import annotations

from datetime import date, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


def test_write_lineage_function_exists() -> None:
    import inspect

    from alphavedha.data.ingestion import _write_lineage

    assert inspect.iscoroutinefunction(_write_lineage)


def test_email_alerter_has_data_quality_failed() -> None:
    from alphavedha.monitoring.alerts import EmailAlerter

    assert hasattr(EmailAlerter, "data_quality_failed")
    assert callable(EmailAlerter.data_quality_failed)


@pytest.mark.asyncio
async def test_write_lineage_adds_row() -> None:
    from alphavedha.data.ingestion import _write_lineage

    mock_session = MagicMock()
    mock_session.commit = AsyncMock()
    await _write_lineage(
        mock_session,
        symbol="TCS.NS",
        record_date=date(2026, 5, 30),
        table_name="daily_ohlcv",
        provider="yfinance",
        fetched_at=datetime(2026, 5, 30, 16, 0, 0),
        row_count=100,
    )
    assert mock_session.add.call_count == 1
    assert mock_session.commit.called


def test_data_quality_failed_skips_when_no_critical() -> None:
    from alphavedha.data.quality import QualityReport, QualityResult
    from alphavedha.monitoring.alerts import EmailAlerter

    alerter = EmailAlerter()
    report = QualityReport(
        report_date=date(2026, 5, 30),
        results=[QualityResult("completeness", True, "ok", "all good")],
    )
    result = alerter.data_quality_failed(report)
    assert result is False


def test_data_quality_failed_sends_when_critical() -> None:
    from alphavedha.data.quality import QualityReport, QualityResult
    from alphavedha.monitoring.alerts import EmailAlerter

    alerter = EmailAlerter()
    report = QualityReport(
        report_date=date(2026, 5, 30),
        results=[
            QualityResult("freshness", False, "critical", "data is 30h old"),
            QualityResult("completeness", True, "ok", "all good"),
        ],
    )
    with patch.object(alerter, "send", return_value=True) as mock_send:
        result = alerter.data_quality_failed(report)
    assert result is True
    mock_send.assert_called_once()
    call_kwargs = mock_send.call_args[1]
    from alphavedha.monitoring.alerts import AlertLevel

    assert call_kwargs["level"] == AlertLevel.CRITICAL
