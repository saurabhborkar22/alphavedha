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


# Data subcommands (stubs — out of scope for Week 8)
data_app = typer.Typer(help="Data management commands")


@data_app.command("refresh")
def data_refresh() -> None:
    """Fetch latest market data."""
    typer.echo("Refreshing data... (not yet wired — requires DB)")


@data_app.command("backfill")
def data_backfill(
    start: str = typer.Option("2005-01-01", help="Start date for backfill (YYYY-MM-DD)"),
) -> None:
    """Backfill historical market data."""
    typer.echo(f"Backfilling from {start}... (not yet wired — requires DB)")


@data_app.command("status")
def data_status() -> None:
    """Show data freshness status."""
    typer.echo("Checking data status... (not yet wired — requires DB)")


app.add_typer(data_app, name="data")

if __name__ == "__main__":
    app()
