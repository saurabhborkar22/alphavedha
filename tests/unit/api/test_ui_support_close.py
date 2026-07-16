"""Manual position close: outcomes must use the price-return convention."""

from __future__ import annotations

import asyncio
from datetime import date
from unittest.mock import AsyncMock, patch

import pandas as pd
import pytest

from alphavedha.api.routes import ui_support


def _open_short() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "symbol": "INFY",
                "prediction_date": date(2026, 7, 1),
                "strategy": "ensemble_v1",
                "predicted_direction": -1,
                "entry_price": 100.0,
                "exit_price": None,
            }
        ]
    )


def test_manual_close_stores_price_return(monkeypatch: pytest.MonkeyPatch) -> None:
    """Regression: a winning short (price fell 4%) must store the raw price
    return (-0.04) with is_correct=True, tagged exit_reason='manual_close'."""
    monkeypatch.delenv("ALPHAVEDHA_DEMO", raising=False)
    closes = pd.DataFrame({"close": [96.0]})

    with (
        patch(
            "alphavedha.data.store.load_paper_trades",
            new_callable=AsyncMock,
            return_value=_open_short(),
        ),
        patch.object(
            ui_support.ui_data,
            "load_recent_closes",
            new_callable=AsyncMock,
            return_value=closes,
        ),
        patch(
            "alphavedha.data.store.update_paper_trade_outcome",
            new_callable=AsyncMock,
        ) as mock_update,
    ):
        result = asyncio.run(ui_support.close_paper_position("INFY:2026-07-01"))

    assert result == {"status": "closed", "id": "INFY:2026-07-01"}
    kwargs = mock_update.await_args.kwargs
    assert kwargs["exit_price"] == 96.0
    assert kwargs["actual_return"] == pytest.approx(-0.04)
    assert kwargs["is_correct"] is True
    assert kwargs["exit_reason"] == "manual_close"
