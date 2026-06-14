"""One-time historical paper-trading + walk-forward simulation.

Produces an *honest, out-of-sample* track record without waiting for live
trades to mature, by:

  1. Training a FROZEN model as-of ``--cutoff`` (sees no data after that date),
     into an isolated artifacts dir so the live ``latest`` models are untouched.
  2. Replaying each trading day in (cutoff, end-15d] through the REAL prediction
     engine via the ``as_of`` seam — byte-identical to production serving except
     the reference date — recording every prediction as a simulated paper trade.
  3. Evaluating each against the actual 15-trading-day forward close (the same
     horizon the live evaluator uses).
  4. Writing a committed JSON artifact with the 3-track cost-adjusted record
     (paper page) and walk-forward views (backtest page).

Touches nothing live: no ``paper_trades`` / ``daily_pnl`` writes, trains into
``--sim-dir``, and leaves live serving behaviour unchanged.

Run on a host with the OHLCV store + trained-data access (i.e. the VPS):

    python -m scripts.sim_paper_trading --cutoff 2025-12-12 --tier large \
        --out alphavedha/api/sim_artifact.json

Use ``--no-train`` to reuse an already-trained frozen model, and
``--max-days`` / ``--max-symbols`` for a fast smoke test.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import structlog

logger = structlog.get_logger(__name__)

HORIZON_TRADING_DAYS = 15
OHLCV_HISTORY_START = date(2020, 1, 1)
DEFAULT_SIM_DIR = Path("models/artifacts_sim")
DEFAULT_OUT = Path("alphavedha/api/sim_artifact.json")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--cutoff", required=True, help="Train cutoff (YYYY-MM-DD); model sees no later data."
    )
    p.add_argument("--end", default=None, help="Last simulated day (default: latest OHLCV bar).")
    p.add_argument("--tier", default="large", help="Universe tier (default: large).")
    p.add_argument(
        "--sim-dir",
        default=str(DEFAULT_SIM_DIR),
        help="Isolated artifacts dir for the frozen model.",
    )
    p.add_argument("--out", default=str(DEFAULT_OUT), help="Output JSON artifact path.")
    p.add_argument(
        "--no-train", action="store_true", help="Reuse an existing frozen model in --sim-dir."
    )
    p.add_argument(
        "--max-days", type=int, default=0, help="Limit simulated days (smoke test; 0 = all)."
    )
    p.add_argument(
        "--max-symbols", type=int, default=0, help="Limit symbols (smoke test; 0 = all)."
    )
    p.add_argument(
        "--regime-overlay",
        action="store_true",
        help="Enable the regime-aware exposure overlay (sets ALPHAVEDHA_REGIME_OVERLAY=1).",
    )
    return p.parse_args()


async def _train_frozen(tier: str, cutoff: date, sim_dir: Path) -> None:
    """Train the full ensemble as-of cutoff into the isolated sim dir."""
    import alphavedha.training.pipeline as pipeline

    # Redirect every artifact + tier-data-cache write into the sim dir. The
    # pipeline reads this module global at call time, so reassigning is enough.
    pipeline.ARTIFACTS_DIR = sim_dir
    logger.info("sim_train_start", tier=tier, cutoff=str(cutoff), sim_dir=str(sim_dir))
    results = await pipeline.train_all(tier=tier, end_date=cutoff)
    trained = [name for name, r in results.items() if getattr(r, "artifact_path", None) is not None]
    logger.info("sim_train_done", trained=trained)


async def _load_universe_ohlcv(symbols: list[str], end: date) -> dict[str, pd.DataFrame]:
    """Preload each symbol's OHLCV once for entry/exit price lookups."""
    from alphavedha.data.store import load_ohlcv

    out: dict[str, pd.DataFrame] = {}
    for sym in symbols:
        try:
            df = await load_ohlcv(sym, OHLCV_HISTORY_START, end)
            if not df.empty:
                out[sym] = df
        except Exception as exc:
            logger.warning("sim_ohlcv_load_failed", symbol=sym, error=str(exc))
    return out


def _trading_calendar(ohlcv: dict[str, pd.DataFrame], cutoff: date, end: date) -> list[date]:
    """Union of trading dates across the universe within (cutoff, end]."""
    days: set[date] = set()
    for df in ohlcv.values():
        for ts in df.index:
            d = ts.date() if hasattr(ts, "date") else ts
            if cutoff < d <= end:
                days.add(d)
    return sorted(days)


def _entry_close(df: pd.DataFrame, as_of: date) -> float | None:
    past = df[df.index.date <= as_of]
    return float(past["close"].iloc[-1]) if not past.empty else None


def _exit_close(df: pd.DataFrame, as_of: date, horizon: int = HORIZON_TRADING_DAYS) -> float | None:
    future = df[df.index.date > as_of]
    if future.empty:
        return None
    idx = min(horizon, len(future)) - 1
    return float(future["close"].iloc[idx])


def _sign(x: float) -> int:
    return 1 if x > 0 else (-1 if x < 0 else 0)


async def _simulate(
    tier: str,
    cutoff: date,
    end: date,
    sim_dir: Path,
    max_days: int,
    max_symbols: int,
) -> pd.DataFrame:
    """Replay each trading day through the real engine and evaluate outcomes."""
    from alphavedha.config import get_config
    from alphavedha.data.universe import get_symbols_for_tier
    from alphavedha.services.cache import PredictionCache
    from alphavedha.services.model_registry import ModelRegistry
    from alphavedha.services.prediction_service import PredictionService

    config = get_config()
    symbols = await get_symbols_for_tier(tier)
    if max_symbols:
        symbols = symbols[:max_symbols]

    ohlcv = await _load_universe_ohlcv(symbols, end)
    if not ohlcv:
        raise RuntimeError("No OHLCV data available — run the sim where the store is populated.")

    calendar = _trading_calendar(ohlcv, cutoff, end)
    # Only days that can fully mature (need HORIZON bars after T).
    last = max(df.index[-1].date() for df in ohlcv.values())
    calendar = [d for d in calendar if d <= last - timedelta(days=int(HORIZON_TRADING_DAYS * 1.6))]
    if max_days:
        calendar = calendar[:max_days]
    logger.info(
        "sim_calendar", n_days=len(calendar), first=str(calendar[0]), last=str(calendar[-1])
    )

    # Build the engine once (loads the frozen models from sim_dir); reuse it
    # across days by mutating the reference date and clearing the regime cache.
    registry = ModelRegistry(demo=False, artifact_dir=sim_dir)
    service = PredictionService(
        registry=registry, cache=PredictionCache(redis_client=None), config=config
    )

    rows: list[dict] = []
    for n, t in enumerate(calendar, 1):
        service._as_of = t
        service._market_features_cache = None
        try:
            preds = await service.predict_tier(tier)
        except Exception as exc:
            logger.warning("sim_day_failed", day=str(t), error=str(exc))
            continue

        for p in preds:
            df = ohlcv.get(p.symbol)
            if df is None:
                continue
            entry = _entry_close(df, t)
            exit_price = _exit_close(df, t)
            if entry is None or entry <= 0 or exit_price is None:
                continue
            actual_return = (exit_price - entry) / entry
            direction = int(p.direction)
            rows.append(
                {
                    "symbol": p.symbol,
                    "prediction_date": t,
                    "predicted_direction": direction,
                    "predicted_magnitude": float(p.magnitude),
                    "confidence": float(p.meta_confidence),
                    "model_version": p.model_version,
                    "regime": p.regime,
                    "is_tradeable": bool(p.is_tradeable),
                    "entry_price": entry,
                    "exit_price": exit_price,
                    "actual_return": actual_return,
                    "is_correct": _sign(direction) == _sign(actual_return),
                }
            )
        if n % 10 == 0 or n == len(calendar):
            logger.info("sim_progress", day=f"{n}/{len(calendar)}", trades=len(rows))

    return pd.DataFrame(rows)


async def _run(args: argparse.Namespace) -> None:
    from alphavedha.backtest.costs import compute_round_trip_cost_pct
    from alphavedha.backtest.sim_views import build_artifact
    from alphavedha.config import get_config

    cutoff = date.fromisoformat(args.cutoff)
    end = date.fromisoformat(args.end) if args.end else date.today()
    sim_dir = Path(args.sim_dir)

    # Engine reads this in its constructor, so set it before the service is built.
    if args.regime_overlay:
        os.environ["ALPHAVEDHA_REGIME_OVERLAY"] = "1"
        logger.info("regime_overlay_enabled")

    if not args.no_train:
        await _train_frozen(args.tier, cutoff, sim_dir)

    trades = await _simulate(args.tier, cutoff, end, sim_dir, args.max_days, args.max_symbols)
    logger.info("sim_trades", n=len(trades))

    cost_pct = compute_round_trip_cost_pct(args.tier, get_config().backtest)
    meta = {
        "tier": args.tier,
        "cutoff": cutoff.isoformat(),
        "end": end.isoformat(),
        "n_trades": len(trades),
        "n_days": int(trades["prediction_date"].nunique()) if not trades.empty else 0,
        "model_version": (str(trades["model_version"].iloc[0]) if not trades.empty else "unknown"),
        "horizon_trading_days": HORIZON_TRADING_DAYS,
        "out_of_sample": True,
        "regime_overlay": bool(args.regime_overlay),
        "generated_by": "scripts/sim_paper_trading.py",
    }
    artifact = build_artifact(trades, cost_pct, meta)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(artifact, indent=2))
    logger.info("sim_artifact_written", path=str(out_path), bytes=out_path.stat().st_size)

    # Archive this run alongside past runs (never overwritten by a later
    # window). The API serves these via /paper/simulations + /simulation/{slug}.
    # Overlay runs get a distinct slug so a window's baseline and its overlay
    # A/B coexist in the run-picker instead of overwriting each other.
    slug = f"{args.tier}__{cutoff.isoformat()}__{end.isoformat()}"
    if args.regime_overlay:
        slug += "__overlay"
    archive_dir = out_path.parent / "sim_runs"
    try:
        archive_dir.mkdir(parents=True, exist_ok=True)
        (archive_dir / f"{slug}.json").write_text(json.dumps(artifact, indent=2))
        logger.info("sim_run_archived", slug=slug, dir=str(archive_dir))
    except Exception as exc:
        logger.warning("sim_run_archive_failed", slug=slug, error=str(exc))

    print(f"\n✅ Wrote {out_path}")
    print(
        f"   trades={meta['n_trades']} days={meta['n_days']} cutoff={meta['cutoff']} end={meta['end']}"
    )
    bt = artifact["backtest"]["summary"]
    print(
        f"   strategy: cagr={bt['cagr']:.2%} sharpe={bt['sharpe']} maxDD={bt['max_drawdown']:.2%} win={bt['win_rate']:.2%}"
    )


def main() -> None:
    asyncio.run(_run(_parse_args()))


if __name__ == "__main__":
    main()
