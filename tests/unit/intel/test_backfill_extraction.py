"""Tests for backfill extraction script."""

from __future__ import annotations

import importlib
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import scripts.backfill_extraction as backfill_mod


def _reload() -> None:
    importlib.reload(backfill_mod)


def _mock_provider() -> MagicMock:
    p = MagicMock()
    p.name = "mock/test"
    return p


class TestBackfillScript:
    """Smoke tests for the backfill script's async logic."""

    @pytest.mark.asyncio
    async def test_dry_run_shows_count(self, capsys: pytest.CaptureFixture[str]) -> None:
        mock_disclosures = [
            {"category": "Board Meeting"},
            {"category": "Board Meeting"},
            {"category": "Financial Results"},
        ]

        with (
            patch.object(
                backfill_mod,
                "get_unprocessed_disclosures",
                new_callable=AsyncMock,
                return_value=mock_disclosures,
            ),
            patch.object(
                backfill_mod,
                "get_provider",
                return_value=_mock_provider(),
            ),
        ):
            await backfill_mod.backfill(dry_run=True)
            captured = capsys.readouterr()
            assert "3 unprocessed" in captured.out
            assert "Board Meeting" in captured.out

    @pytest.mark.asyncio
    async def test_stops_on_empty(self, capsys: pytest.CaptureFixture[str]) -> None:
        with (
            patch.object(
                backfill_mod,
                "run_extraction_batch",
                new_callable=AsyncMock,
                return_value={"status": "empty"},
            ),
            patch.object(
                backfill_mod,
                "get_provider",
                return_value=_mock_provider(),
            ),
        ):
            await backfill_mod.backfill(max_batches=5, delay_seconds=0)
            captured = capsys.readouterr()
            assert "All disclosures processed" in captured.out

    @pytest.mark.asyncio
    async def test_stops_on_budget(self, capsys: pytest.CaptureFixture[str]) -> None:
        with (
            patch.object(
                backfill_mod,
                "run_extraction_batch",
                new_callable=AsyncMock,
                return_value={"status": "budget_exceeded"},
            ),
            patch.object(
                backfill_mod,
                "get_provider",
                return_value=_mock_provider(),
            ),
        ):
            await backfill_mod.backfill(max_batches=5, delay_seconds=0)
            captured = capsys.readouterr()
            assert "Budget exceeded" in captured.out

    @pytest.mark.asyncio
    async def test_accumulates_results(self, capsys: pytest.CaptureFixture[str]) -> None:
        call_count = 0

        async def mock_batch(**_kwargs: object) -> dict[str, object]:
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                return {
                    "status": "ok",
                    "extracted": 10,
                    "skipped_boilerplate": 5,
                    "failed": 1,
                    "estimated_cost_usd": 0.01,
                }
            return {"status": "empty"}

        with (
            patch.object(
                backfill_mod,
                "run_extraction_batch",
                side_effect=mock_batch,
            ),
            patch.object(
                backfill_mod,
                "get_provider",
                return_value=_mock_provider(),
            ),
        ):
            await backfill_mod.backfill(max_batches=10, delay_seconds=0)
            captured = capsys.readouterr()
            assert "Total extracted: 20" in captured.out
            assert "Total skipped:   10" in captured.out
