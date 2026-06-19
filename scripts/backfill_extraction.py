"""Backfill LLM extraction over historical disclosures.

Processes unprocessed disclosures in batches, respecting the cost budget.
Designed for one-time historical catch-up after P1 data ingestion.

Usage:
    GEMINI_API_KEY=... python scripts/backfill_extraction.py
    GEMINI_API_KEY=... python scripts/backfill_extraction.py --provider groq --batch-size 25
    GEMINI_API_KEY=... python scripts/backfill_extraction.py --max-batches 5 --dry-run
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from alphavedha.intel.extraction.batcher import (
    CostLedger,
    get_unprocessed_disclosures,
    run_extraction_batch,
)
from alphavedha.intel.extraction.llm import get_provider


async def backfill(
    provider_name: str = "gemini",
    batch_size: int = 50,
    max_batches: int = 100,
    dry_run: bool = False,
    delay_seconds: float = 2.0,
) -> None:
    provider = get_provider(provider_name)
    ledger = CostLedger()

    print(f"Provider: {provider.name}")
    print(f"Batch size: {batch_size}")
    print(f"Max batches: {max_batches}")
    print(f"Monthly budget: ${ledger.monthly_budget_usd}")
    print(f"Month spend so far: ${ledger.current_month_usd():.4f}")
    print()

    if dry_run:
        disclosures = await get_unprocessed_disclosures(limit=1000)
        print(f"Dry run: {len(disclosures)} unprocessed disclosures found")
        if disclosures:
            from collections import Counter

            categories = Counter(d.get("category", "unknown") for d in disclosures)
            print("\nCategory breakdown:")
            for cat, count in categories.most_common(20):
                print(f"  {cat:40s} {count:4d}")
        return

    total_extracted = 0
    total_skipped = 0
    total_failed = 0
    total_cost = 0.0
    start = time.time()

    for i in range(max_batches):
        result = await run_extraction_batch(
            provider=provider,
            batch_size=batch_size,
        )

        if result["status"] == "budget_exceeded":
            print(f"\nBudget exceeded at batch {i + 1}. Stopping.")
            break

        if result["status"] == "empty":
            print(f"\nAll disclosures processed at batch {i + 1}.")
            break

        total_extracted += result["extracted"]
        total_skipped += result["skipped_boilerplate"]
        total_failed += result["failed"]
        total_cost += result["estimated_cost_usd"]

        elapsed = time.time() - start
        print(
            f"  Batch {i + 1}/{max_batches}: "
            f"extracted={result['extracted']} "
            f"skipped={result['skipped_boilerplate']} "
            f"failed={result['failed']} "
            f"cost=${result['estimated_cost_usd']:.4f} "
            f"[{elapsed:.0f}s elapsed]"
        )

        if delay_seconds > 0 and i < max_batches - 1:
            time.sleep(delay_seconds)

    elapsed = time.time() - start
    print(f"\n{'=' * 60}")
    print(f"Backfill complete in {elapsed:.1f}s")
    print(f"  Total extracted: {total_extracted}")
    print(f"  Total skipped:   {total_skipped}")
    print(f"  Total failed:    {total_failed}")
    print(f"  Total cost:      ${total_cost:.4f}")
    print(f"  Month total:     ${ledger.current_month_usd():.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Backfill LLM extraction on historical disclosures"
    )
    parser.add_argument("--provider", default="gemini", help="LLM provider (gemini or groq)")
    parser.add_argument("--batch-size", type=int, default=50, help="Disclosures per batch")
    parser.add_argument("--max-batches", type=int, default=100, help="Max batches to run")
    parser.add_argument("--delay", type=float, default=2.0, help="Seconds between batches")
    parser.add_argument("--dry-run", action="store_true", help="Just show counts, don't extract")
    args = parser.parse_args()

    asyncio.run(
        backfill(
            provider_name=args.provider,
            batch_size=args.batch_size,
            max_batches=args.max_batches,
            dry_run=args.dry_run,
            delay_seconds=args.delay,
        )
    )
