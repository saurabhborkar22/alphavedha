"""Rich formatters for CLI output — prediction panels and ranking tables."""

from __future__ import annotations

import json
from dataclasses import asdict

from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from alphavedha.prediction.engine import StockPrediction
from alphavedha.prediction.ranker import RankingResult

_DIRECTION_COLORS = {1: "green", -1: "red", 0: "yellow"}
_DIRECTION_LABELS = {1: "BUY", -1: "SELL", 0: "HOLD"}


def format_prediction(pred: StockPrediction) -> Panel:
    """Format a single prediction as a Rich Panel with color-coded direction."""
    direction_label = _DIRECTION_LABELS.get(pred.direction, "?")
    color = _DIRECTION_COLORS.get(pred.direction, "white")

    lines: list[str] = []
    lines.append(f"Direction:      [{color}]{direction_label}[/{color}]")
    lines.append(f"Composite Score: {pred.composite_score:.1f}/100")
    lines.append(f"Meta Confidence: {pred.meta_confidence:.2f}")
    lines.append(f"Regime:          {pred.regime}")
    lines.append(f"Magnitude:       {pred.magnitude:.4f}")
    lines.append("")
    lines.append("[bold]Price Targets[/bold]")
    lines.append(f"  Low:  {pred.price_target_low:.2f}")
    lines.append(f"  Mid:  {pred.price_target_mid:.2f}")
    lines.append(f"  High: {pred.price_target_high:.2f}")
    lines.append("")
    lines.append("[bold]Risk[/bold]")
    lines.append(f"  Position Size: {pred.position_size_pct:.1f}%")
    lines.append(f"  Disagreement:  {pred.model_disagreement:.4f}")
    lines.append(f"  Tradeable:     {'Yes' if pred.is_tradeable else 'No'}")

    if pred.warnings:
        lines.append("")
        lines.append("[bold yellow]Warnings[/bold yellow]")
        for w in pred.warnings:
            lines.append(f"  - {w}")

    lines.append("")
    lines.append(f"[dim]{pred.model_version} | {pred.timestamp.isoformat()}[/dim]")

    body = "\n".join(lines)
    title = Text(f" {pred.symbol} — {direction_label} ", style=f"bold {color}")

    return Panel(body, title=title, border_style=color, expand=False)


def format_ranking(result: RankingResult) -> Table:
    """Format a ranking result as a Rich Table with buy/sell candidates."""
    table = Table(title="Stock Rankings", show_lines=True)
    table.add_column("#", style="dim", width=4)
    table.add_column("Symbol", style="bold")
    table.add_column("Direction")
    table.add_column("Score", justify="right")
    table.add_column("Position %", justify="right")
    table.add_column("Regime")

    rank = 1
    for pred in result.buy_candidates:
        color = _DIRECTION_COLORS.get(pred.direction, "white")
        table.add_row(
            str(rank),
            pred.symbol,
            Text(_DIRECTION_LABELS.get(pred.direction, "?"), style=color),
            f"{pred.composite_score:.1f}",
            f"{pred.position_size_pct:.1f}%",
            pred.regime,
        )
        rank += 1

    for pred in result.sell_candidates:
        color = _DIRECTION_COLORS.get(pred.direction, "white")
        table.add_row(
            str(rank),
            pred.symbol,
            Text(_DIRECTION_LABELS.get(pred.direction, "?"), style=color),
            f"{pred.composite_score:.1f}",
            f"{pred.position_size_pct:.1f}%",
            pred.regime,
        )
        rank += 1

    return table


def prediction_to_json(pred: StockPrediction) -> str:
    """Serialize a prediction to JSON with numpy array handling."""
    data = asdict(pred)
    data["timestamp"] = pred.timestamp.isoformat()
    data["regime_probabilities"] = pred.regime_probabilities.tolist()
    data["direction_label"] = _DIRECTION_LABELS.get(pred.direction, "UNKNOWN")
    return json.dumps(data, indent=2, default=str)


def ranking_to_json(result: RankingResult) -> str:
    """Serialize a ranking result to JSON."""
    data = {
        "buy_candidates": [
            {
                "symbol": p.symbol,
                "direction": p.direction,
                "direction_label": _DIRECTION_LABELS.get(p.direction, "?"),
                "composite_score": p.composite_score,
                "position_size_pct": p.position_size_pct,
                "regime": p.regime,
            }
            for p in result.buy_candidates
        ],
        "sell_candidates": [
            {
                "symbol": p.symbol,
                "direction": p.direction,
                "direction_label": _DIRECTION_LABELS.get(p.direction, "?"),
                "composite_score": p.composite_score,
                "position_size_pct": p.position_size_pct,
                "regime": p.regime,
            }
            for p in result.sell_candidates
        ],
        "excluded": [{"symbol": s, "reason": r} for s, r in result.excluded],
        "generated_at": result.generated_at.isoformat(),
    }
    return json.dumps(data, indent=2, default=str)
