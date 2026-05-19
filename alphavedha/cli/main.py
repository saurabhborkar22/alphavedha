"""AlphaVedha CLI — predict, scan, and serve commands."""

from __future__ import annotations

import asyncio
import sys

import structlog
import typer
from rich.console import Console

from alphavedha.cli.formatters import (
    format_prediction,
    format_ranking,
    prediction_to_json,
    ranking_to_json,
)
from alphavedha.config import get_config
from alphavedha.services.cache import PredictionCache
from alphavedha.services.model_registry import ModelRegistry
from alphavedha.services.prediction_service import PredictionService

# Send structlog output to stderr so stdout stays clean for JSON/piping
structlog.configure(
    logger_factory=structlog.PrintLoggerFactory(file=sys.stderr),
)

console = Console()
app = typer.Typer(
    name="alphavedha",
    help="AlphaVedha — AI-powered Indian stock market prediction engine",
    no_args_is_help=True,
)


def _build_service(demo: bool) -> PredictionService:
    config = get_config()
    registry = ModelRegistry(demo=demo)
    cache = PredictionCache(redis_client=None)
    return PredictionService(registry=registry, cache=cache, config=config)


@app.command()
def predict(
    symbol: str = typer.Argument(..., help="Stock symbol (e.g., TCS)"),
    demo: bool = typer.Option(False, "--demo", help="Use synthetic predictions"),
    output_json: bool = typer.Option(False, "--json", help="Output as JSON"),
) -> None:
    """Run prediction for a single stock."""
    try:
        service = _build_service(demo)
        result = asyncio.run(service.predict_single(symbol.upper()))

        if output_json:
            typer.echo(prediction_to_json(result))
        else:
            console.print(format_prediction(result))
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(code=1) from None


@app.command()
def scan(
    tier: str = typer.Argument("large", help="Market cap tier: large, mid, small"),
    top_n: int = typer.Option(10, "--top-n", help="Number of top candidates"),
    demo: bool = typer.Option(False, "--demo", help="Use synthetic predictions"),
    output_json: bool = typer.Option(False, "--json", help="Output as JSON"),
) -> None:
    """Scan and rank all stocks in a tier."""
    try:
        service = _build_service(demo)

        if not output_json:
            with console.status(f"Scanning {tier} cap stocks..."):
                result = asyncio.run(service.scan_tier(tier, top_n=top_n))
        else:
            result = asyncio.run(service.scan_tier(tier, top_n=top_n))

        if output_json:
            typer.echo(ranking_to_json(result))
        else:
            console.print(format_ranking(result))

            if result.excluded:
                console.print(f"\n[dim]Excluded: {len(result.excluded)} stocks[/dim]")
                for sym, reason in result.excluded[:5]:
                    console.print(f"  [dim]{sym}: {reason}[/dim]")
                if len(result.excluded) > 5:
                    console.print(f"  [dim]... and {len(result.excluded) - 5} more[/dim]")
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(code=1) from None


@app.command()
def serve(
    host: str = typer.Option("0.0.0.0", help="Bind host"),
    port: int = typer.Option(8000, help="Bind port"),
    demo: bool = typer.Option(False, "--demo", help="Start in demo mode"),
    reload: bool = typer.Option(False, "--reload", help="Auto-reload on code changes"),
) -> None:
    """Start the FastAPI prediction server."""
    import os

    import uvicorn

    if demo:
        os.environ["ALPHAVEDHA_DEMO"] = "1"

    console.print(f"Starting AlphaVedha API on {host}:{port}", style="bold green")
    if demo:
        console.print("[yellow]Demo mode enabled — using synthetic predictions[/yellow]")

    uvicorn.run(
        "alphavedha.api.app:create_app",
        host=host,
        port=port,
        reload=reload,
        factory=True,
    )


data_app = typer.Typer(help="Data management commands")


@data_app.command("refresh")
def data_refresh(
    tier: str = typer.Option("large", help="Market cap tier: large, mid, small"),
    days: int = typer.Option(5, help="Lookback days for refresh"),
) -> None:
    """Fetch latest market data for a tier."""
    from alphavedha.data.ingestion import refresh_latest

    with console.status(f"Refreshing {tier} cap data (last {days} days)..."):
        result = asyncio.run(refresh_latest(tier, lookback_days=days))

    console.print(
        f"[green]Done:[/green] {result.symbols_succeeded}/{result.symbols_requested} symbols, "
        f"{result.total_rows_stored} rows stored"
    )
    if result.failed_symbols:
        console.print(f"[yellow]Failed ({result.symbols_failed}):[/yellow] {', '.join(result.failed_symbols[:10])}")


@data_app.command("backfill")
def data_backfill(
    tier: str = typer.Option("large", help="Market cap tier: large, mid, small"),
    start: str = typer.Option("2020-01-01", help="Start date (YYYY-MM-DD)"),
) -> None:
    """Backfill historical market data for a tier."""
    from alphavedha.data.ingestion import backfill

    console.print(f"Backfilling [bold]{tier}[/bold] cap from {start}...")
    result = asyncio.run(backfill(tier, start))

    console.print(
        f"[green]Done:[/green] {result.symbols_succeeded}/{result.symbols_requested} symbols, "
        f"{result.total_rows_stored} rows stored"
    )
    if result.failed_symbols:
        console.print(f"[yellow]Failed ({result.symbols_failed}):[/yellow] {', '.join(result.failed_symbols[:10])}")
        for sym, err in list(result.errors.items())[:5]:
            console.print(f"  [dim]{sym}: {err}[/dim]")


@data_app.command("status")
def data_status() -> None:
    """Show data freshness status."""
    from alphavedha.data.database import get_session_factory
    from alphavedha.data.models import DailyOHLCV, DerivativesData, IndexConstituent, InstitutionalFlow

    from sqlalchemy import func, select

    async def _status() -> None:
        session_factory = get_session_factory()
        async with session_factory() as session:
            ohlcv_count = (await session.execute(select(func.count(DailyOHLCV.id)))).scalar() or 0
            symbol_count = (await session.execute(
                select(func.count(func.distinct(DailyOHLCV.symbol)))
            )).scalar() or 0
            latest_date = (await session.execute(select(func.max(DailyOHLCV.date)))).scalar()
            index_count = (await session.execute(select(func.count(IndexConstituent.id)))).scalar() or 0
            flow_count = (await session.execute(select(func.count(InstitutionalFlow.id)))).scalar() or 0
            flow_latest = (await session.execute(select(func.max(InstitutionalFlow.date)))).scalar()
            deriv_count = (await session.execute(select(func.count(DerivativesData.id)))).scalar() or 0
            deriv_latest = (await session.execute(select(func.max(DerivativesData.date)))).scalar()

        console.print(f"[bold]Database Status[/bold]")
        console.print(f"  OHLCV rows:      {ohlcv_count:,}")
        console.print(f"  Symbols:         {symbol_count}")
        console.print(f"  Latest date:     {latest_date or 'no data'}")
        console.print(f"  Index members:   {index_count}")
        console.print(f"  FII/DII rows:    {flow_count:,} (latest: {flow_latest or 'no data'})")
        console.print(f"  Derivatives:     {deriv_count:,} (latest: {deriv_latest or 'no data'})")

    asyncio.run(_status())


@data_app.command("fii-refresh")
def data_fii_refresh() -> None:
    """Fetch today's FII/DII flow data from NSE."""
    from alphavedha.data.ingestion import ingest_fii_dii

    with console.status("Fetching FII/DII data from NSE..."):
        result = asyncio.run(ingest_fii_dii())

    if result.error:
        console.print(f"[red]Error:[/red] {result.error}")
        raise typer.Exit(code=1)

    console.print(
        f"[green]Done:[/green] {result.rows_stored} rows stored "
        f"(categories: {', '.join(result.categories)})"
    )


@data_app.command("derivatives-refresh")
def data_derivatives_refresh(
    tier: str = typer.Option("large", help="Market cap tier"),
    symbol: str | None = typer.Option(None, help="Single symbol to fetch"),
) -> None:
    """Fetch F&O derivatives data from NSE."""
    from alphavedha.data.ingestion import ingest_derivatives

    symbols = [symbol] if symbol else None
    label = symbol or f"{tier} cap"

    with console.status(f"Fetching derivatives data for {label}..."):
        result = asyncio.run(ingest_derivatives(symbols=symbols, tier=tier))

    console.print(
        f"[green]Done:[/green] {result.symbols_succeeded}/{result.symbols_requested} symbols, "
        f"{result.rows_stored} rows stored"
    )
    if result.errors:
        console.print(f"[yellow]Errors ({len(result.errors)}):[/yellow]")
        for sym, err in list(result.errors.items())[:5]:
            console.print(f"  [dim]{sym}: {err}[/dim]")


@data_app.command("earnings-refresh")
def data_earnings_refresh(
    tier: str = typer.Option("large", help="Market cap tier"),
    symbol: str | None = typer.Option(None, help="Single symbol to fetch"),
) -> None:
    """Fetch quarterly earnings data for stocks."""
    from alphavedha.data.ingestion import ingest_earnings

    symbols = [symbol] if symbol else None
    label = symbol or f"{tier} cap"

    with console.status(f"Fetching earnings data for {label}..."):
        result = asyncio.run(ingest_earnings(symbols=symbols, tier=tier))

    console.print(
        f"[green]Done:[/green] {result.symbols_succeeded}/{result.symbols_requested} symbols, "
        f"{result.total_quarters} quarters stored"
    )
    if result.errors:
        console.print(f"[yellow]Errors ({len(result.errors)}):[/yellow]")
        for sym, err in list(result.errors.items())[:5]:
            console.print(f"  [dim]{sym}: {err}[/dim]")


app.add_typer(data_app, name="data")


# Training subcommands
train_app = typer.Typer(help="Model training commands")


@train_app.command("xgboost")
def train_xgboost_cmd(
    tier: str = typer.Option("large", help="Market cap tier: large, mid, small"),
) -> None:
    """Train XGBoost model on all stocks in a tier."""
    from alphavedha.training.pipeline import train_xgboost

    console.print(f"Training XGBoost on [bold]{tier}[/bold] cap stocks...")
    result = asyncio.run(train_xgboost(tier))

    if result.train_result:
        tr = result.train_result
        console.print(f"\n[bold green]Training complete[/bold green]")
        console.print(f"  Symbols:        {result.n_symbols}")
        console.print(f"  Train rows:     {result.n_train_rows:,}")
        console.print(f"  Val rows:       {result.n_val_rows:,}")
        console.print(f"  Train accuracy: {tr.train_metrics.get('accuracy', 0):.3f}")
        console.print(f"  Val accuracy:   {tr.val_metrics.get('accuracy', 0):.3f}")
        console.print(f"  Val F1:         {tr.val_metrics.get('f1_weighted', 0):.3f}")
        if "rmse" in tr.val_metrics:
            console.print(f"  Val RMSE:       {tr.val_metrics['rmse']:.4f}")
        console.print(f"  Time:           {result.total_time_seconds:.1f}s")
        console.print(f"  Saved to:       {result.artifact_path}")
    else:
        console.print("[red]Training failed — no data available[/red]")

    if result.errors:
        console.print(f"\n[yellow]Errors ({len(result.errors)}):[/yellow]")
        for sym, err in list(result.errors.items())[:5]:
            console.print(f"  [dim]{sym}: {err}[/dim]")


def _print_train_result(result: object) -> None:
    """Print standard training result metrics."""
    from alphavedha.training.pipeline import TrainingPipelineResult

    r: TrainingPipelineResult = result  # type: ignore[assignment]
    if r.train_result:
        tr = r.train_result
        console.print(f"\n[bold green]{r.model_name} training complete[/bold green]")
        if r.n_symbols:
            console.print(f"  Symbols:        {r.n_symbols}")
        if r.n_train_rows:
            console.print(f"  Train rows:     {r.n_train_rows:,}")
        if r.n_val_rows:
            console.print(f"  Val rows:       {r.n_val_rows:,}")
        if "accuracy" in tr.train_metrics:
            console.print(f"  Train accuracy: {tr.train_metrics['accuracy']:.3f}")
        if "accuracy" in tr.val_metrics:
            console.print(f"  Val accuracy:   {tr.val_metrics['accuracy']:.3f}")
        if "f1_weighted" in tr.val_metrics:
            console.print(f"  Val F1:         {tr.val_metrics['f1_weighted']:.3f}")
        if "rmse" in tr.val_metrics:
            console.print(f"  Val RMSE:       {tr.val_metrics['rmse']:.4f}")
        console.print(f"  Time:           {tr.training_time_seconds:.1f}s")
        if r.artifact_path:
            console.print(f"  Saved to:       {r.artifact_path}")
    elif r.extra_metrics:
        console.print(f"\n[bold green]{r.model_name} training complete[/bold green]")
        for k, v in r.extra_metrics.items():
            console.print(f"  {k}: {v:.4f}")
        if r.artifact_path:
            console.print(f"  Saved to:       {r.artifact_path}")
    else:
        console.print(f"[red]{r.model_name} training failed — no data available[/red]")

    if r.errors:
        console.print(f"\n[yellow]Errors ({len(r.errors)}):[/yellow]")
        for sym, err in list(r.errors.items())[:5]:
            console.print(f"  [dim]{sym}: {err}[/dim]")


@train_app.command("lstm")
def train_lstm_cmd(
    tier: str = typer.Option("large", help="Market cap tier: large, mid, small"),
) -> None:
    """Train LSTM model (requires XGBoost trained first for feature selection)."""
    from alphavedha.training.pipeline import train_lstm

    console.print(f"Training LSTM on [bold]{tier}[/bold] cap stocks...")
    result = asyncio.run(train_lstm(tier))
    _print_train_result(result)


@train_app.command("tft")
def train_tft_cmd(
    tier: str = typer.Option("large", help="Market cap tier: large, mid, small"),
) -> None:
    """Train Temporal Fusion Transformer (TFT-lite) model."""
    from alphavedha.training.pipeline import train_tft

    console.print(f"Training TFT on [bold]{tier}[/bold] cap stocks...")
    result = asyncio.run(train_tft(tier))
    _print_train_result(result)


@train_app.command("regime")
def train_regime_cmd(
    tier: str = typer.Option("large", help="Market cap tier: large, mid, small"),
) -> None:
    """Train HMM regime detector on portfolio returns + volatility."""
    from alphavedha.training.pipeline import train_regime

    console.print(f"Training Regime Detector on [bold]{tier}[/bold] cap stocks...")
    result = asyncio.run(train_regime(tier))
    _print_train_result(result)


@train_app.command("all")
def train_all_cmd(
    tier: str = typer.Option("large", help="Market cap tier: large, mid, small"),
) -> None:
    """Train all models in dependency order: XGBoost → LSTM → TFT → Regime → Ensemble → Meta → Conformal."""
    from alphavedha.training.pipeline import train_all

    console.print(f"Training [bold]all models[/bold] on [bold]{tier}[/bold] cap stocks...")
    console.print("Order: XGBoost → LSTM → TFT → Regime → Ensemble → Meta-labeling → Conformal\n")

    results = asyncio.run(train_all(tier))

    trained = [m for m, r in results.items() if r.artifact_path is not None]
    failed = [m for m, r in results.items() if r.artifact_path is None]

    console.print(f"\n[bold]{'='*50}[/bold]")
    console.print(f"[bold]Training Summary[/bold]")
    console.print(f"[bold]{'='*50}[/bold]")

    for name, r in results.items():
        _print_train_result(r)

    console.print(f"\n[bold green]Trained:[/bold green] {', '.join(trained) if trained else 'none'}")
    if failed:
        console.print(f"[bold red]Failed:[/bold red] {', '.join(failed)}")

    if results:
        total = next(iter(results.values())).total_time_seconds
        console.print(f"\n[bold]Total time: {total:.1f}s[/bold]")


app.add_typer(train_app, name="train")


# Backtest subcommands
backtest_app = typer.Typer(help="Backtesting commands")


@backtest_app.command("walk-forward")
def backtest_walk_forward(
    symbol: str = typer.Argument("TCS.NS", help="Stock symbol"),
    start: str = typer.Option("2024-01-01", help="Test period start (YYYY-MM-DD)"),
    end: str = typer.Option("2026-05-01", help="Test period end (YYYY-MM-DD)"),
    tier: str = typer.Option("large", help="Market cap tier for costs"),
) -> None:
    """Run walk-forward backtest for a stock."""
    from datetime import date as date_type

    from alphavedha.backtest.walk_forward import run_walk_forward
    from alphavedha.config import get_config

    config = get_config()

    console.print(f"Walk-forward backtest for [bold]{symbol}[/bold]")
    console.print(f"Test period: {start} to {end}")
    console.print("Loading data...")

    def _run() -> None:
        from alphavedha.data.store import load_ohlcv

        ohlcv = asyncio.run(load_ohlcv(
            symbol,
            date_type.fromisoformat("2020-01-01"),
            date_type.fromisoformat(end),
        ))

        if ohlcv.empty:
            console.print("[red]No OHLCV data found. Run data backfill first.[/red]")
            raise typer.Exit(code=1)

        def dummy_predictions(train_df: object, test_df: object) -> object:
            import pandas as _pd
            import numpy as _np

            idx = test_df.index  # type: ignore[union-attr]
            rng = _np.random.default_rng(42)
            return _pd.DataFrame({
                "direction": rng.choice([-1, 0, 1], size=len(idx)),
                "confidence": rng.uniform(0.4, 0.8, size=len(idx)),
                "magnitude": rng.uniform(0.01, 0.05, size=len(idx)),
            }, index=idx)

        result = run_walk_forward(
            ohlcv_df=ohlcv,
            predictions_fn=dummy_predictions,
            config=config.backtest,
            start=date_type.fromisoformat(start),
            end=date_type.fromisoformat(end),
            market_cap_tier=tier,
        )

        console.print(f"\n[bold]{'='*50}[/bold]")
        console.print(f"[bold]Walk-Forward Results[/bold]")
        console.print(f"[bold]{'='*50}[/bold]")
        console.print(f"  Folds:              {len(result.folds)}")
        console.print(f"  Total trades:       {result.n_trades}")
        console.print(f"  Total return:       {result.total_return:.2%}")
        console.print(f"  Annualized return:  {result.annualized_return:.2%}")
        console.print(f"  Sharpe ratio:       {result.sharpe_ratio:.4f}")
        console.print(f"  Max drawdown:       {result.max_drawdown:.2%}")
        console.print(f"  Win rate:           {result.win_rate:.2%}")
        console.print(f"  Profit factor:      {result.profit_factor:.2f}")
        console.print(f"  Benchmark return:   {result.benchmark_return:.2%}")
        console.print(f"  Alpha:              {result.alpha_vs_benchmark:.2%}")

    _run()


app.add_typer(backtest_app, name="backtest")

if __name__ == "__main__":
    app()
