from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pandas as pd
import pytest


@pytest.mark.asyncio
async def test_ingest_trends_returns_dict() -> None:
    from alphavedha.data.ingestion import ingest_trends

    mock_df = pd.DataFrame({"it": [50, 60, 70]})
    mock_provider = AsyncMock()
    mock_provider.fetch_all_sectors.return_value = {
        "banking": mock_df,
        "it": mock_df,
        "pharma": mock_df,
        "auto": mock_df,
        "fmcg": mock_df,
    }

    with patch("alphavedha.data.ingestion.GoogleTrendsProvider", return_value=mock_provider):
        result = await ingest_trends()

    assert set(result.keys()) == {"banking", "it", "pharma", "auto", "fmcg"}


def test_fetch_trends_cli_exists() -> None:
    from typer.testing import CliRunner

    from alphavedha.cli.main import app

    runner = CliRunner()
    result = runner.invoke(app, ["data", "fetch-trends", "--help"])
    assert result.exit_code == 0
