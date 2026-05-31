from __future__ import annotations

from datetime import date
from unittest.mock import AsyncMock, patch

import pytest


@pytest.mark.asyncio
async def test_ingest_bse_announcements_returns_count() -> None:
    from alphavedha.data.ingestion import ingest_bse_announcements
    from alphavedha.data.providers.bse_provider import CorporateAnnouncementRecord

    mock_session = AsyncMock()
    mock_records = [
        CorporateAnnouncementRecord(
            symbol="TCS.NS",
            announced_date=date(2026, 5, 30),
            ex_date=None,
            event_type="BOARD_MEETING",
            description="Board meeting for Q4 results",
        )
    ]

    mock_provider = AsyncMock()
    mock_provider.fetch_bulk.return_value = {"TCS.NS": mock_records}

    with patch("alphavedha.data.ingestion.BSEProvider", return_value=mock_provider):
        count = await ingest_bse_announcements(
            ["TCS.NS"],
            date(2026, 5, 1),
            date(2026, 5, 30),
            session=mock_session,
        )

    assert count == 1
    assert mock_session.commit.called


@pytest.mark.asyncio
async def test_ingest_bse_announcements_empty_symbols() -> None:
    from alphavedha.data.ingestion import ingest_bse_announcements

    mock_session = AsyncMock()
    mock_provider = AsyncMock()
    mock_provider.fetch_bulk.return_value = {}

    with patch("alphavedha.data.ingestion.BSEProvider", return_value=mock_provider):
        count = await ingest_bse_announcements(
            [],
            date(2026, 5, 1),
            date(2026, 5, 30),
            session=mock_session,
        )

    assert count == 0


def test_fetch_bse_cli_command_exists() -> None:
    from typer.testing import CliRunner

    from alphavedha.cli.main import app

    runner = CliRunner()
    result = runner.invoke(app, ["data", "fetch-bse", "--help"])
    assert result.exit_code == 0
