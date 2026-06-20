"""Tests for batch extraction pipeline and cost ledger."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from alphavedha.intel.extraction.batcher import (
    CostLedger,
    _get_ledger_path,
    run_extraction_batch,
    run_nightly_extraction,
)


class TestCostLedger:
    def test_default_budget(self) -> None:
        ledger = CostLedger()
        assert ledger.monthly_budget_usd == 50.0

    def test_custom_budget_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("INTEL_LLM_BUDGET_USD", "100")
        ledger = CostLedger()
        assert ledger.monthly_budget_usd == 100.0

    def test_estimate_batch_cost_gemini(self) -> None:
        ledger = CostLedger()
        cost = ledger.estimate_batch_cost(10, "gemini/gemini-2.5-flash")
        assert cost == pytest.approx(0.002)

    def test_estimate_batch_cost_groq(self) -> None:
        ledger = CostLedger()
        cost = ledger.estimate_batch_cost(10, "groq/llama-3.3-70b")
        assert cost == pytest.approx(0.003)

    def test_estimate_batch_cost_cerebras(self) -> None:
        ledger = CostLedger()
        cost = ledger.estimate_batch_cost(10, "cerebras/llama-3.3-70b")
        assert cost == pytest.approx(0.001)

    def test_estimate_batch_cost_unknown_provider(self) -> None:
        ledger = CostLedger()
        cost = ledger.estimate_batch_cost(10, "unknown/model")
        assert cost == pytest.approx(0.01)

    def test_is_over_budget_initially_false(self) -> None:
        ledger = CostLedger()
        assert ledger.is_over_budget() is False

    def test_record_and_read_back(self, tmp_path: Path) -> None:
        with patch(
            "alphavedha.intel.extraction.batcher._get_ledger_path",
            return_value=tmp_path / "ledger.csv",
        ):
            ledger = CostLedger()
            ledger._ledger_path = tmp_path / "ledger.csv"
            ledger.record_batch(1.50)
            ledger.record_batch(0.75)
            assert ledger.current_month_usd() == pytest.approx(2.25)

    def test_is_over_budget_when_exceeded(self, tmp_path: Path) -> None:
        ledger = CostLedger()
        ledger._ledger_path = tmp_path / "ledger.csv"
        ledger.monthly_budget_usd = 1.0
        ledger.record_batch(0.60)
        ledger.record_batch(0.50)
        assert ledger.is_over_budget() is True


class TestGetLedgerPath:
    def test_default_path(self) -> None:
        path = _get_ledger_path()
        assert path.name == "intel_cost_ledger.csv"

    def test_custom_data_dir(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ALPHAVEDHA_DATA_DIR", "/tmp/av_test")
        path = _get_ledger_path()
        assert str(path) == "/tmp/av_test/intel_cost_ledger.csv"


class TestRunExtractionBatch:
    @pytest.mark.asyncio
    async def test_empty_batch(self) -> None:
        with patch(
            "alphavedha.intel.extraction.batcher.load_disclosures",
            new_callable=AsyncMock,
        ) as mock_load:
            import pandas as pd

            mock_load.return_value = pd.DataFrame()
            result = await run_extraction_batch()
            assert result["status"] == "empty"
            assert result["processed"] == 0

    @pytest.mark.asyncio
    async def test_budget_exceeded(self, tmp_path: Path) -> None:
        with patch("alphavedha.intel.extraction.batcher.CostLedger") as MockLedger:
            mock_ledger = MagicMock()
            mock_ledger.is_over_budget.return_value = True
            mock_ledger.current_month_usd.return_value = 55.0
            mock_ledger.monthly_budget_usd = 50.0
            MockLedger.return_value = mock_ledger

            result = await run_extraction_batch()
            assert result["status"] == "budget_exceeded"

    @pytest.mark.asyncio
    async def test_processes_disclosures(self) -> None:
        import pandas as pd

        from alphavedha.intel.extraction.schemas import DisclosureExtraction
        from alphavedha.intel.extraction.taxonomy import EventType

        mock_df = pd.DataFrame(
            [
                {
                    "id": 1,
                    "symbol": "TCS.NS",
                    "source": "nse",
                    "category": "Press Release",
                    "headline": "TCS wins deal",
                    "filed_at": "2026-06-19",
                    "url": None,
                    "text": None,
                    "text_hash": None,
                    "processed_at": None,
                },
                {
                    "id": 2,
                    "symbol": "INFY.NS",
                    "source": "nse",
                    "category": "Trading Window",
                    "headline": "Closure",
                    "filed_at": "2026-06-19",
                    "url": None,
                    "text": None,
                    "text_hash": None,
                    "processed_at": None,
                },
            ]
        )

        mock_extraction = DisclosureExtraction(
            event_type=EventType.ORDER_WIN,
            direction=1,
            materiality=7,
            confidence=0.9,
            summary="TCS wins deal",
        )

        with (
            patch(
                "alphavedha.intel.extraction.batcher.load_disclosures",
                new_callable=AsyncMock,
                return_value=mock_df,
            ),
            patch(
                "alphavedha.intel.extraction.batcher.extract_one",
                return_value=mock_extraction,
            ),
            patch(
                "alphavedha.intel.extraction.batcher.store_disclosure_events",
                new_callable=AsyncMock,
                return_value=1,
            ),
            patch(
                "alphavedha.intel.extraction.batcher.mark_disclosures_processed",
                new_callable=AsyncMock,
                return_value=2,
            ),
            patch("alphavedha.intel.extraction.batcher.CostLedger") as MockLedger,
        ):
            mock_ledger = MagicMock()
            mock_ledger.is_over_budget.return_value = False
            mock_ledger.estimate_batch_cost.return_value = 0.001
            mock_ledger.current_month_usd.return_value = 0.001
            MockLedger.return_value = mock_ledger

            result = await run_extraction_batch()
            assert result["status"] == "ok"
            assert result["total"] == 2
            assert result["extracted"] == 1
            assert result["skipped_boilerplate"] == 1

    @pytest.mark.asyncio
    async def test_failed_not_marked_processed(self) -> None:
        """Failed extractions must NOT be marked as processed (so they can be retried)."""
        import pandas as pd

        mock_df = pd.DataFrame(
            [
                {
                    "id": 10,
                    "symbol": "SBIN.NS",
                    "source": "nse",
                    "category": "Press Release",
                    "headline": "Quarterly update",
                    "filed_at": "2026-06-19",
                    "url": None,
                    "text": None,
                    "text_hash": None,
                    "processed_at": None,
                },
            ]
        )

        with (
            patch(
                "alphavedha.intel.extraction.batcher.load_disclosures",
                new_callable=AsyncMock,
                return_value=mock_df,
            ),
            patch(
                "alphavedha.intel.extraction.batcher.extract_one",
                return_value=None,
            ),
            patch(
                "alphavedha.intel.extraction.batcher.store_disclosure_events",
                new_callable=AsyncMock,
                return_value=0,
            ),
            patch(
                "alphavedha.intel.extraction.batcher.mark_disclosures_processed",
                new_callable=AsyncMock,
                return_value=0,
            ) as mock_mark,
            patch("alphavedha.intel.extraction.batcher.CostLedger") as MockLedger,
        ):
            mock_ledger = MagicMock()
            mock_ledger.is_over_budget.return_value = False
            mock_ledger.estimate_batch_cost.return_value = 0.0002
            mock_ledger.current_month_usd.return_value = 0.0002
            MockLedger.return_value = mock_ledger

            result = await run_extraction_batch()
            assert result["failed"] == 1
            assert result["extracted"] == 0
            mock_mark.assert_not_called()

    @pytest.mark.asyncio
    async def test_dedup_skips_duplicate_text_hash(self) -> None:
        """Disclosures with the same text_hash reuse the first extraction."""
        import pandas as pd

        from alphavedha.intel.extraction.schemas import DisclosureExtraction
        from alphavedha.intel.extraction.taxonomy import EventType

        mock_df = pd.DataFrame(
            [
                {
                    "id": 1,
                    "symbol": "TCS.NS",
                    "source": "bse",
                    "category": "Press Release",
                    "headline": "TCS wins deal",
                    "filed_at": "2026-06-19",
                    "url": None,
                    "text": "Full text...",
                    "text_hash": "abc123hash",
                    "processed_at": None,
                },
                {
                    "id": 2,
                    "symbol": "TCS.NS",
                    "source": "nse",
                    "category": "Press Release",
                    "headline": "TCS wins deal",
                    "filed_at": "2026-06-19",
                    "url": None,
                    "text": "Full text...",
                    "text_hash": "abc123hash",
                    "processed_at": None,
                },
            ]
        )

        mock_extraction = DisclosureExtraction(
            event_type=EventType.ORDER_WIN,
            direction=1,
            materiality=7,
            confidence=0.9,
            summary="TCS wins deal",
        )

        with (
            patch(
                "alphavedha.intel.extraction.batcher.load_disclosures",
                new_callable=AsyncMock,
                return_value=mock_df,
            ),
            patch(
                "alphavedha.intel.extraction.batcher.extract_one",
                return_value=mock_extraction,
            ) as mock_extract,
            patch(
                "alphavedha.intel.extraction.batcher.store_disclosure_events",
                new_callable=AsyncMock,
                return_value=2,
            ),
            patch(
                "alphavedha.intel.extraction.batcher.mark_disclosures_processed",
                new_callable=AsyncMock,
                return_value=2,
            ),
            patch("alphavedha.intel.extraction.batcher.CostLedger") as MockLedger,
        ):
            mock_ledger = MagicMock()
            mock_ledger.is_over_budget.return_value = False
            mock_ledger.estimate_batch_cost.return_value = 0.0002
            mock_ledger.current_month_usd.return_value = 0.0002
            MockLedger.return_value = mock_ledger

            result = await run_extraction_batch()
            assert result["status"] == "ok"
            assert result["extracted"] == 2
            assert result["skipped_dedup"] == 1
            assert mock_extract.call_count == 1


class TestRunNightlyExtraction:
    @pytest.mark.asyncio
    async def test_stops_on_empty(self) -> None:
        with (
            patch(
                "alphavedha.intel.extraction.batcher.get_provider",
            ) as mock_get,
            patch(
                "alphavedha.intel.extraction.batcher.run_extraction_batch",
                new_callable=AsyncMock,
                return_value={"status": "empty", "processed": 0},
            ),
        ):
            mock_get.return_value = MagicMock()
            result = await run_nightly_extraction(max_batches=5)
            assert result["batches_run"] == 1
            assert result["status"] == "ok"

    @pytest.mark.asyncio
    async def test_stops_on_budget(self) -> None:
        with (
            patch(
                "alphavedha.intel.extraction.batcher.get_provider",
            ) as mock_get,
            patch(
                "alphavedha.intel.extraction.batcher.run_extraction_batch",
                new_callable=AsyncMock,
                return_value={
                    "status": "budget_exceeded",
                    "spent_usd": 55.0,
                    "budget_usd": 50.0,
                },
            ),
        ):
            mock_get.return_value = MagicMock()
            result = await run_nightly_extraction(max_batches=5)
            assert result["status"] == "budget_exceeded"
            assert result["batches_run"] == 1
