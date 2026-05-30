"""Unit tests for BSEProvider."""

from __future__ import annotations

from datetime import date
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from alphavedha.data.providers.bse_provider import (
    BSEProvider,
    CorporateAnnouncementRecord,
)

BSE_RESPONSE_FIXTURE = {
    "Table": [
        {
            "SCRIP_CD": "532540",
            "DT_TM": "20260530120000",
            "CATEGORYNAME": "Board Meeting",
            "HEADLINE": "Board Meeting for Q4 results",
        }
    ]
}


def test_announcement_record_dataclass() -> None:
    rec = CorporateAnnouncementRecord(
        symbol="TCS.NS",
        announced_date=date(2026, 5, 30),
        ex_date=None,
        event_type="BOARD_MEETING",
        description="Board Meeting for Q4 results",
    )
    assert rec.symbol == "TCS.NS"
    assert rec.event_type == "BOARD_MEETING"
    assert len(rec.description) <= 500


def test_description_truncated_to_500() -> None:
    long_desc = "x" * 600
    rec = CorporateAnnouncementRecord(
        symbol="TCS.NS",
        announced_date=date(2026, 5, 30),
        ex_date=None,
        event_type="OTHER",
        description=long_desc,
    )
    assert len(rec.description) == 500


def test_parse_event_type_board_meeting() -> None:
    provider = BSEProvider()
    assert provider._parse_event_type("Board Meeting for Q4") == "BOARD_MEETING"


def test_parse_event_type_dividend() -> None:
    provider = BSEProvider()
    assert provider._parse_event_type("Cash Dividend") == "DIVIDEND"


def test_parse_event_type_unknown() -> None:
    provider = BSEProvider()
    assert provider._parse_event_type("Some Unknown Category") == "OTHER"


def test_nse_to_bse_code_known_symbol() -> None:
    provider = BSEProvider()
    assert provider._nse_to_bse_code("TCS.NS") == "532540"


def test_nse_to_bse_code_unknown_symbol() -> None:
    provider = BSEProvider()
    assert provider._nse_to_bse_code("UNKNOWN.NS") is None


@pytest.mark.asyncio
async def test_fetch_announcements_parses_response() -> None:
    provider = BSEProvider()

    mock_resp = AsyncMock()
    mock_resp.status = 200
    mock_resp.json = AsyncMock(return_value=BSE_RESPONSE_FIXTURE)
    mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
    mock_resp.__aexit__ = AsyncMock(return_value=False)

    mock_session = MagicMock()
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=False)
    mock_session.get = MagicMock(return_value=mock_resp)

    with patch("aiohttp.ClientSession", return_value=mock_session):
        results = await provider.fetch_announcements("TCS.NS", date(2026, 5, 1), date(2026, 5, 30))

    assert len(results) == 1
    assert results[0].symbol == "TCS.NS"
    assert results[0].event_type == "BOARD_MEETING"
    assert results[0].announced_date == date(2026, 5, 30)


@pytest.mark.asyncio
async def test_fetch_announcements_empty_on_http_error() -> None:
    provider = BSEProvider()

    mock_resp = AsyncMock()
    mock_resp.status = 500
    mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
    mock_resp.__aexit__ = AsyncMock(return_value=False)

    mock_session = MagicMock()
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=False)
    mock_session.get = MagicMock(return_value=mock_resp)

    with patch("aiohttp.ClientSession", return_value=mock_session):
        results = await provider.fetch_announcements("TCS.NS", date(2026, 5, 1), date(2026, 5, 30))
    assert results == []


@pytest.mark.asyncio
async def test_fetch_announcements_empty_for_unknown_symbol() -> None:
    provider = BSEProvider()
    results = await provider.fetch_announcements("UNKNOWN.NS", date(2026, 5, 1), date(2026, 5, 30))
    assert results == []
