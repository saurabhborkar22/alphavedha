"""Nightly batch extraction pipeline.

Collects unprocessed disclosures, runs LLM extraction, stores events,
and tracks token costs. Respects monthly budget cap.
"""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import Any, ClassVar
from zoneinfo import ZoneInfo

import structlog

from alphavedha.intel.extraction.extractor import (
    CURRENT_PROMPT_VERSION,
    extract_one,
    is_boilerplate,
)
from alphavedha.intel.extraction.llm import LLMProvider, get_provider
from alphavedha.intel.store import (
    load_disclosures,
    mark_disclosures_processed,
    store_disclosure_events,
)

logger = structlog.get_logger(__name__)

IST = ZoneInfo("Asia/Kolkata")

BATCH_SIZE = 50
DEFAULT_MONTHLY_BUDGET_USD = 50.0


async def get_unprocessed_disclosures(limit: int = 500) -> list[dict[str, Any]]:
    """Fetch disclosures that haven't been through LLM extraction."""
    df = await load_disclosures(unprocessed_only=True, limit=limit)
    if df.empty:
        return []
    records: list[dict[str, Any]] = df.to_dict("records")
    return records


async def run_extraction_batch(
    provider: LLMProvider | None = None,
    batch_size: int = BATCH_SIZE,
    prompt_version: str = CURRENT_PROMPT_VERSION,
) -> dict[str, Any]:
    """Run one batch of LLM extraction on unprocessed disclosures.

    Returns a summary dict with counts and cost info.
    """
    if provider is None:
        provider = get_provider()

    cost_ledger = CostLedger()
    if cost_ledger.is_over_budget():
        logger.warning(
            "extraction_budget_exceeded",
            spent=cost_ledger.current_month_usd(),
            budget=cost_ledger.monthly_budget_usd,
        )
        return {
            "status": "budget_exceeded",
            "spent_usd": cost_ledger.current_month_usd(),
            "budget_usd": cost_ledger.monthly_budget_usd,
        }

    disclosures = await get_unprocessed_disclosures(limit=batch_size)
    if not disclosures:
        logger.info("extraction_batch_empty", reason="no unprocessed disclosures")
        return {"status": "empty", "processed": 0}

    events: list[dict[str, Any]] = []
    processed_ids: list[int] = []
    skipped = 0
    failed = 0

    for disc in disclosures:
        disc_id = disc["id"]
        symbol = disc["symbol"]
        category = disc.get("category", "")
        headline = disc.get("headline", "")
        text = disc.get("text")

        if is_boilerplate(category):
            skipped += 1
            processed_ids.append(disc_id)
            continue

        extraction = extract_one(provider, symbol, category, headline, text, prompt_version)

        if extraction is None:
            failed += 1
            processed_ids.append(disc_id)
            continue

        events.append(
            {
                "disclosure_id": disc_id,
                "symbol": symbol,
                "event_type": extraction.event_type.value,
                "direction": extraction.direction,
                "materiality": extraction.materiality,
                "confidence": extraction.confidence,
                "summary": extraction.summary,
                "red_flags": extraction.red_flags if extraction.red_flags else None,
                "llm_model": provider.name,
                "prompt_version": prompt_version,
                "extracted_at": datetime.now(IST),
            }
        )
        processed_ids.append(disc_id)

    stored = 0
    if events:
        stored = await store_disclosure_events(events)

    if processed_ids:
        await mark_disclosures_processed(processed_ids, datetime.now(IST))

    estimated_cost = cost_ledger.estimate_batch_cost(
        llm_calls=len(disclosures) - skipped,
        provider_name=provider.name,
    )
    cost_ledger.record_batch(estimated_cost)

    summary = {
        "status": "ok",
        "total": len(disclosures),
        "extracted": stored,
        "skipped_boilerplate": skipped,
        "failed": failed,
        "estimated_cost_usd": round(estimated_cost, 4),
        "month_total_usd": round(cost_ledger.current_month_usd(), 4),
    }

    logger.info("extraction_batch_complete", **summary)
    return summary


async def run_nightly_extraction(
    max_batches: int = 10,
    batch_size: int = BATCH_SIZE,
) -> dict[str, Any]:
    """Run extraction in batches until all disclosures are processed or budget is hit."""
    provider = get_provider()
    batches_run = 0
    total_extracted = 0
    total_skipped = 0
    total_failed = 0
    total_cost_usd = 0.0
    status = "ok"

    for i in range(max_batches):
        result = await run_extraction_batch(
            provider=provider,
            batch_size=batch_size,
        )

        batches_run = i + 1

        if result["status"] == "budget_exceeded":
            status = "budget_exceeded"
            break

        if result["status"] == "empty":
            break

        total_extracted += int(result["extracted"])
        total_skipped += int(result["skipped_boilerplate"])
        total_failed += int(result["failed"])
        total_cost_usd += float(result["estimated_cost_usd"])

    totals: dict[str, Any] = {
        "batches_run": batches_run,
        "total_extracted": total_extracted,
        "total_skipped": total_skipped,
        "total_failed": total_failed,
        "total_cost_usd": round(total_cost_usd, 4),
        "status": status,
    }

    logger.info("nightly_extraction_complete", **totals)
    return totals


class CostLedger:
    """Track LLM extraction costs against a monthly budget.

    Costs are stored in a simple text file for persistence across runs.
    """

    COST_PER_CALL_USD: ClassVar[dict[str, float]] = {
        "gemini": 0.0002,
        "groq": 0.0003,
        "anthropic": 0.003,
    }

    def __init__(self) -> None:
        self.monthly_budget_usd = float(
            os.environ.get("INTEL_LLM_BUDGET_USD", str(DEFAULT_MONTHLY_BUDGET_USD))
        )
        self._ledger_path = _get_ledger_path()

    def current_month_usd(self) -> float:
        """Get total spend for the current month."""
        current_month = datetime.now(IST).strftime("%Y-%m")
        total = 0.0
        if self._ledger_path.exists():
            for line in self._ledger_path.read_text().strip().splitlines():
                parts = line.split(",")
                if len(parts) >= 2 and parts[0].startswith(current_month):
                    total += float(parts[1])
        return total

    def is_over_budget(self) -> bool:
        return self.current_month_usd() >= self.monthly_budget_usd

    def estimate_batch_cost(
        self,
        llm_calls: int,
        provider_name: str,
    ) -> float:
        provider_key = provider_name.split("/")[0]
        cost_per_call = self.COST_PER_CALL_USD.get(provider_key, 0.001)
        return llm_calls * cost_per_call

    def record_batch(self, cost_usd: float) -> None:
        """Append a cost entry to the ledger file."""
        timestamp = datetime.now(IST).isoformat()
        self._ledger_path.parent.mkdir(parents=True, exist_ok=True)
        with self._ledger_path.open("a") as f:
            f.write(f"{timestamp},{cost_usd:.6f}\n")


def _get_ledger_path() -> Path:
    data_dir = Path(os.environ.get("ALPHAVEDHA_DATA_DIR", "data"))
    return data_dir / "intel_cost_ledger.csv"
