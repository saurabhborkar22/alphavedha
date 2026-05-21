"""Stock relationship graph builder for GNN models."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import structlog
import yaml

logger = structlog.get_logger(__name__)

_CONFIGS_DIR = Path(__file__).resolve().parent.parent.parent / "configs"


def _load_stocks_config() -> dict:
    path = _CONFIGS_DIR / "stocks.yaml"
    with path.open() as f:
        return yaml.safe_load(f)


def get_sector_map() -> dict[str, list[str]]:
    return _load_stocks_config()["sectors"]


def get_promoter_groups() -> dict[str, list[str]]:
    return _load_stocks_config()["promoter_groups"]


SECTOR_MAP = get_sector_map()
PROMOTER_GROUPS = get_promoter_groups()

_EDGE_TYPE_SECTOR = 0
_EDGE_TYPE_CORRELATION = 1
_EDGE_TYPE_PROMOTER = 2


@dataclass
class StockGraph:
    symbols: list[str]
    edge_index: np.ndarray
    edge_type: np.ndarray
    edge_weight: np.ndarray
    symbol_to_idx: dict[str, int]


def _add_sector_edges(
    symbol_set: set[str],
    symbol_to_idx: dict[str, int],
) -> tuple[list[tuple[int, int]], list[int], list[float]]:
    """Create edges between stocks in the same sector."""
    edges: list[tuple[int, int]] = []
    types: list[int] = []
    weights: list[float] = []

    for sector_symbols in SECTOR_MAP.values():
        present = [s for s in sector_symbols if s in symbol_set]
        for i in range(len(present)):
            for j in range(i + 1, len(present)):
                src = symbol_to_idx[present[i]]
                dst = symbol_to_idx[present[j]]
                edges.append((src, dst))
                edges.append((dst, src))
                types.extend([_EDGE_TYPE_SECTOR, _EDGE_TYPE_SECTOR])
                weights.extend([1.0, 1.0])

    return edges, types, weights


def _add_correlation_edges(
    symbols: list[str],
    symbol_to_idx: dict[str, int],
    returns_df: pd.DataFrame,
    threshold: float,
) -> tuple[list[tuple[int, int]], list[int], list[float]]:
    """Create edges between stocks with rolling correlation above threshold."""
    edges: list[tuple[int, int]] = []
    types: list[int] = []
    weights: list[float] = []

    available = [s for s in symbols if s in returns_df.columns]
    if len(available) < 2:
        return edges, types, weights

    corr_matrix = returns_df[available].corr()

    for i in range(len(available)):
        for j in range(i + 1, len(available)):
            corr_val = corr_matrix.iloc[i, j]
            if np.isnan(corr_val):
                continue
            if abs(corr_val) >= threshold:
                src = symbol_to_idx[available[i]]
                dst = symbol_to_idx[available[j]]
                edges.append((src, dst))
                edges.append((dst, src))
                types.extend([_EDGE_TYPE_CORRELATION, _EDGE_TYPE_CORRELATION])
                weights.extend([abs(corr_val), abs(corr_val)])

    return edges, types, weights


def _add_promoter_edges(
    symbol_set: set[str],
    symbol_to_idx: dict[str, int],
) -> tuple[list[tuple[int, int]], list[int], list[float]]:
    """Create edges between stocks owned by the same promoter group."""
    edges: list[tuple[int, int]] = []
    types: list[int] = []
    weights: list[float] = []

    for group_symbols in PROMOTER_GROUPS.values():
        present = [s for s in group_symbols if s in symbol_set]
        for i in range(len(present)):
            for j in range(i + 1, len(present)):
                src = symbol_to_idx[present[i]]
                dst = symbol_to_idx[present[j]]
                edges.append((src, dst))
                edges.append((dst, src))
                types.extend([_EDGE_TYPE_PROMOTER, _EDGE_TYPE_PROMOTER])
                weights.extend([1.0, 1.0])

    return edges, types, weights


def build_stock_graph(
    symbols: list[str],
    returns_df: pd.DataFrame | None = None,
    correlation_threshold: float = 0.6,
) -> StockGraph:
    """Build graph from sector membership, return correlations, and promoter groups."""
    symbol_to_idx = {s: i for i, s in enumerate(symbols)}
    symbol_set = set(symbols)

    all_edges: list[tuple[int, int]] = []
    all_types: list[int] = []
    all_weights: list[float] = []

    sector_edges, sector_types, sector_weights = _add_sector_edges(symbol_set, symbol_to_idx)
    all_edges.extend(sector_edges)
    all_types.extend(sector_types)
    all_weights.extend(sector_weights)

    if returns_df is not None and not returns_df.empty:
        corr_edges, corr_types, corr_weights = _add_correlation_edges(
            symbols,
            symbol_to_idx,
            returns_df,
            correlation_threshold,
        )
        all_edges.extend(corr_edges)
        all_types.extend(corr_types)
        all_weights.extend(corr_weights)

    promoter_edges, promoter_types, promoter_weights = _add_promoter_edges(
        symbol_set,
        symbol_to_idx,
    )
    all_edges.extend(promoter_edges)
    all_types.extend(promoter_types)
    all_weights.extend(promoter_weights)

    seen: set[tuple[int, int, int]] = set()
    deduped_edges: list[tuple[int, int]] = []
    deduped_types: list[int] = []
    deduped_weights: list[float] = []
    for edge, etype, weight in zip(all_edges, all_types, all_weights, strict=True):
        key = (edge[0], edge[1], etype)
        if key not in seen:
            seen.add(key)
            deduped_edges.append(edge)
            deduped_types.append(etype)
            deduped_weights.append(weight)
    all_edges = deduped_edges
    all_types = deduped_types
    all_weights = deduped_weights

    if all_edges:
        edge_index = np.array(all_edges, dtype=np.int64).T
        edge_type = np.array(all_types, dtype=np.int64)
        edge_weight = np.array(all_weights, dtype=np.float64)
    else:
        edge_index = np.zeros((2, 0), dtype=np.int64)
        edge_type = np.zeros(0, dtype=np.int64)
        edge_weight = np.zeros(0, dtype=np.float64)

    logger.info(
        "stock_graph_built",
        n_nodes=len(symbols),
        n_edges=edge_index.shape[1],
        n_sector=sum(1 for t in all_types if t == _EDGE_TYPE_SECTOR),
        n_correlation=sum(1 for t in all_types if t == _EDGE_TYPE_CORRELATION),
        n_promoter=sum(1 for t in all_types if t == _EDGE_TYPE_PROMOTER),
    )

    return StockGraph(
        symbols=symbols,
        edge_index=edge_index,
        edge_type=edge_type,
        edge_weight=edge_weight,
        symbol_to_idx=symbol_to_idx,
    )
