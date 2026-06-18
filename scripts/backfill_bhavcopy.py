#!/usr/bin/env python3
"""Backfill bhavcopy data for a date range.

Usage:
    python scripts/backfill_bhavcopy.py --start 2023-06-01 --end 2026-06-16
    python scripts/backfill_bhavcopy.py --days 30   # last 30 days
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from datetime import date, timedelta


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill NSE bhavcopy data")
    parser.add_argument("--start", type=date.fromisoformat, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=date.fromisoformat, help="End date (YYYY-MM-DD)")
    parser.add_argument(
        "--days", type=int, help="Backfill last N days (alternative to --start/--end)"
    )
    args = parser.parse_args()

    if args.days:
        end = date.today()
        start = end - timedelta(days=args.days)
    elif args.start and args.end:
        start = args.start
        end = args.end
    else:
        print("Provide --start/--end or --days", file=sys.stderr)
        sys.exit(1)

    print(f"Backfilling bhavcopy: {start} to {end}")

    from alphavedha.intel.collectors.bhavcopy import backfill_bhavcopy

    results = asyncio.run(backfill_bhavcopy(start, end))

    successes = sum(1 for v in results.values() if v > 0)
    total_rows = sum(results.values())
    print(f"\nDone: {successes}/{len(results)} days ingested, {total_rows} total rows")


if __name__ == "__main__":
    main()
