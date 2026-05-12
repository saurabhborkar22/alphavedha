"""AlphaVedha CLI — command-line interface for predictions, training, and data management."""

from __future__ import annotations

import typer

app = typer.Typer(
    name="alphavedha",
    help="AlphaVedha — AI-powered Indian stock market prediction engine",
    no_args_is_help=True,
)


@app.command()
def predict(symbol: str = typer.Argument(..., help="Stock symbol (e.g., TCS.NS)")) -> None:
    """Run prediction for a single stock."""
    typer.echo(f"Predicting {symbol}... (not yet implemented)")


@app.command()
def scan(
    tier: str = typer.Argument("large", help="Market cap tier: large, mid, small"),
) -> None:
    """Scan and rank all stocks in a tier."""
    typer.echo(f"Scanning {tier} cap stocks... (not yet implemented)")


@app.command()
def train(
    model: str = typer.Argument("all", help="Model to train: all, xgboost, lstm, tft, regime, meta"),
) -> None:
    """Train ML models."""
    typer.echo(f"Training {model}... (not yet implemented)")


@app.command()
def backtest(
    mode: str = typer.Argument("portfolio", help="Backtest mode: portfolio, or stock symbol"),
) -> None:
    """Run strategy backtest."""
    typer.echo(f"Backtesting {mode}... (not yet implemented)")


@app.command()
def validate(
    target: str = typer.Argument("all", help="Validation target: all, code, tests, model"),
) -> None:
    """Run validation suite."""
    typer.echo(f"Validating {target}... (not yet implemented)")


data_app = typer.Typer(help="Data management commands")


@data_app.command("refresh")
def data_refresh() -> None:
    """Fetch latest market data."""
    typer.echo("Refreshing data... (not yet implemented)")


@data_app.command("backfill")
def data_backfill(
    start: str = typer.Option("2005-01-01", help="Start date for backfill (YYYY-MM-DD)"),
) -> None:
    """Backfill historical market data."""
    typer.echo(f"Backfilling from {start}... (not yet implemented)")


@data_app.command("status")
def data_status() -> None:
    """Show data freshness status."""
    typer.echo("Checking data status... (not yet implemented)")


app.add_typer(data_app, name="data")

if __name__ == "__main__":
    app()
