"""3-window validation: run the historical sim across bull/bear/crash regimes.

Validates that prediction system changes improve (or at least hold) across
all three regime windows from the OOS audit (docs/prediction_audit.md):

  Window 1: 2023-06 (bull)   — model's best case
  Window 2: 2025-06 (bear)   — structural long-bias test
  Window 3: 2025-12 (crash)  — tail-risk / regime-detection test

Each window trains a frozen model as-of the cutoff and replays forward
predictions using sim_paper_trading's async machinery.  The output is a
side-by-side comparison table (printed + written to JSON) with per-window
net P&L, Sharpe, max DD, win rate, and profit factor.

A change is accepted only when it improves or holds across ALL 3 windows.

Usage:
    python -m scripts.validate_3window --tier large
    python -m scripts.validate_3window --tier large --no-train  # reuse frozen models
    python -m scripts.validate_3window --tier large --max-days 5 --max-symbols 3  # smoke test
"""

from __future__ import annotations

import argparse
import asyncio
import json
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import structlog

logger = structlog.get_logger(__name__)

WINDOWS: list[dict[str, Any]] = [
    {
        "name": "bull_2023",
        "cutoff": date(2023, 6, 1),
        "end": date(2024, 3, 31),
        "label": "Bull (Jun 2023 - Mar 2024)",
    },
    {
        "name": "bear_2025",
        "cutoff": date(2025, 3, 1),
        "end": date(2025, 9, 30),
        "label": "Bear (Mar 2025 - Sep 2025)",
    },
    {
        "name": "crash_2025",
        "cutoff": date(2025, 9, 1),
        "end": date(2026, 3, 31),
        "label": "Crash (Sep 2025 - Mar 2026)",
    },
]

DEFAULT_OUT_DIR = Path("reports/3window")


@dataclass
class WindowResult:
    name: str
    label: str
    cutoff: str
    end: str
    n_trades: int
    n_days: int
    total_return: float
    cagr: float
    sharpe: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    avg_return_net: float


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tier", default="large", help="Universe tier (default: large).")
    p.add_argument(
        "--no-train",
        action="store_true",
        help="Reuse existing frozen models in each window's sim-dir.",
    )
    p.add_argument("--max-days", type=int, default=0, help="Limit simulated days per window.")
    p.add_argument("--max-symbols", type=int, default=0, help="Limit symbols per window.")
    p.add_argument(
        "--out-dir",
        default=str(DEFAULT_OUT_DIR),
        help="Output directory for the report.",
    )
    return p.parse_args()


async def _run_window(
    window: dict[str, Any],
    tier: str,
    no_train: bool,
    max_days: int,
    max_symbols: int,
    out_dir: Path,
) -> WindowResult:
    """Run sim_paper_trading for a single window and extract summary metrics."""
    from scripts.sim_paper_trading import _simulate, _train_frozen

    from alphavedha.backtest.costs import compute_round_trip_cost_pct
    from alphavedha.backtest.sim_views import build_artifact
    from alphavedha.config import get_config

    name = window["name"]
    cutoff = window["cutoff"]
    end = window["end"]
    sim_dir = Path(f"models/artifacts_sim_{name}")

    logger.info("window_start", name=name, cutoff=str(cutoff), end=str(end))

    if not no_train:
        await _train_frozen(tier, cutoff, sim_dir)

    trades = await _simulate(tier, cutoff, end, sim_dir, max_days, max_symbols)
    logger.info("window_trades", name=name, n=len(trades))

    config = get_config()
    cost_pct = compute_round_trip_cost_pct(tier, config.backtest)
    meta = {
        "tier": tier,
        "cutoff": cutoff.isoformat(),
        "end": end.isoformat(),
        "n_trades": len(trades),
        "window_name": name,
        "window_label": window["label"],
    }
    artifact = build_artifact(trades, cost_pct, meta)

    window_path = out_dir / f"{name}.json"
    window_path.write_text(json.dumps(artifact, indent=2))

    bt = artifact["backtest"]["summary"]
    tr = artifact["track_record"]
    all_track = tr.get("tracks", {}).get("all", {})

    return WindowResult(
        name=name,
        label=window["label"],
        cutoff=cutoff.isoformat(),
        end=end.isoformat(),
        n_trades=bt.get("total_trades", 0),
        n_days=tr.get("days_tracked", 0),
        total_return=bt.get("cagr", 0.0),
        cagr=bt.get("cagr", 0.0),
        sharpe=bt.get("sharpe", 0.0),
        max_drawdown=bt.get("max_drawdown", 0.0),
        win_rate=bt.get("win_rate", 0.0),
        profit_factor=bt.get("profit_factor", 0.0),
        avg_return_net=all_track.get("avg_return_net", 0.0) or 0.0,
    )


def _print_report(results: list[WindowResult]) -> None:
    """Print a comparison table to stdout."""
    header = f"{'Metric':<25}"
    for r in results:
        header += f"  {r.name:>16}"
    print("\n" + "=" * len(header))
    print("3-WINDOW VALIDATION REPORT")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    rows: list[tuple[str, list[str | int | float], str]] = [
        ("Label", [r.label for r in results], "s"),
        ("Cutoff", [r.cutoff for r in results], "s"),
        ("End", [r.end for r in results], "s"),
        ("Trades", [r.n_trades for r in results], "d"),
        ("Days", [r.n_days for r in results], "d"),
        ("CAGR", [r.cagr for r in results], ".2%"),
        ("Sharpe", [r.sharpe for r in results], ".3f"),
        ("Max Drawdown", [r.max_drawdown for r in results], ".2%"),
        ("Win Rate", [r.win_rate for r in results], ".2%"),
        ("Profit Factor", [r.profit_factor for r in results], ".3f"),
        ("Avg Return (net)", [r.avg_return_net for r in results], ".4%"),
    ]

    for label, values, fmt in rows:
        line = f"{label:<25}"
        for v in values:
            if fmt == "s":
                line += f"  {v!s:>16}"
            elif fmt == "d":
                line += f"  {int(v):>16d}"
            else:
                line += f"  {float(v):>16{fmt}}"
        print(line)

    print("-" * len(header))

    all_positive_sharpe = all(r.sharpe > 0 for r in results)
    all_positive_pf = all(r.profit_factor >= 1.0 for r in results)
    dd_ok = all(r.max_drawdown > -0.15 for r in results)

    print(f"\nAll windows Sharpe > 0:     {'PASS' if all_positive_sharpe else 'FAIL'}")
    print(f"All windows PF >= 1.0:      {'PASS' if all_positive_pf else 'FAIL'}")
    print(f"All windows DD > -15%:      {'PASS' if dd_ok else 'FAIL'}")

    verdict = all_positive_sharpe and all_positive_pf and dd_ok
    print(f"\nOVERALL: {'PASS' if verdict else 'FAIL'}")
    print("=" * len(header) + "\n")


async def _run(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results: list[WindowResult] = []
    for window in WINDOWS:
        result = await _run_window(
            window,
            tier=args.tier,
            no_train=args.no_train,
            max_days=args.max_days,
            max_symbols=args.max_symbols,
            out_dir=out_dir,
        )
        results.append(result)

    _print_report(results)

    report = {
        "generated_at": datetime.now(UTC).isoformat(),
        "tier": args.tier,
        "windows": [asdict(r) for r in results],
    }
    report_path = out_dir / "3window_report.json"
    report_path.write_text(json.dumps(report, indent=2))
    logger.info("report_written", path=str(report_path))


def main() -> None:
    asyncio.run(_run(_parse_args()))


if __name__ == "__main__":
    main()
