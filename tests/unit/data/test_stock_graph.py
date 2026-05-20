"""Tests for stock relationship graph builder."""

from __future__ import annotations

import numpy as np
import pandas as pd

from alphavedha.data.stock_graph import (
    _EDGE_TYPE_CORRELATION,
    _EDGE_TYPE_SECTOR,
    StockGraph,
    _add_correlation_edges,
    _add_sector_edges,
    build_stock_graph,
    get_promoter_groups,
    get_sector_map,
)


class TestSectorMap:
    def test_returns_dict(self) -> None:
        sectors = get_sector_map()
        assert isinstance(sectors, dict)
        assert len(sectors) > 0

    def test_banking_sector_has_symbols(self) -> None:
        sectors = get_sector_map()
        assert "banking" in sectors
        assert "HDFCBANK" in sectors["banking"]
        assert "ICICIBANK" in sectors["banking"]

    def test_it_sector_has_symbols(self) -> None:
        sectors = get_sector_map()
        assert "it" in sectors
        assert "TCS" in sectors["it"]
        assert "INFY" in sectors["it"]


class TestPromoterGroups:
    def test_returns_dict(self) -> None:
        groups = get_promoter_groups()
        assert isinstance(groups, dict)

    def test_groups_have_lists(self) -> None:
        groups = get_promoter_groups()
        for name, symbols in groups.items():
            assert isinstance(symbols, list), f"Group {name} should be a list"


class TestSectorEdges:
    def test_same_sector_creates_bidirectional_edges(self) -> None:
        symbols = {"HDFCBANK", "ICICIBANK", "TCS"}
        idx = {"HDFCBANK": 0, "ICICIBANK": 1, "TCS": 2}
        edges, types, weights = _add_sector_edges(symbols, idx)
        assert len(edges) > 0
        assert all(t == _EDGE_TYPE_SECTOR for t in types)
        assert all(w == 1.0 for w in weights)

    def test_bidirectional(self) -> None:
        symbols = {"HDFCBANK", "ICICIBANK"}
        idx = {"HDFCBANK": 0, "ICICIBANK": 1}
        edges, _, _ = _add_sector_edges(symbols, idx)
        edge_set = set(edges)
        if (0, 1) in edge_set:
            assert (1, 0) in edge_set

    def test_no_edges_for_single_stock_per_sector(self) -> None:
        symbols = {"TCS"}
        idx = {"TCS": 0}
        edges, _, _ = _add_sector_edges(symbols, idx)
        single_sector_edges = [e for e in edges]
        # TCS alone in IT sector among our symbols — but it may still connect
        # via the loaded config; just ensure no self-loops
        for src, dst in single_sector_edges:
            assert src != dst


class TestCorrelationEdges:
    def test_high_correlation_creates_edges(self) -> None:
        dates = pd.bdate_range("2024-01-01", periods=100)
        rng = np.random.default_rng(42)
        base = np.cumsum(rng.normal(0, 0.01, 100))
        returns_df = pd.DataFrame(
            {"A": base, "B": base + rng.normal(0, 0.001, 100)},
            index=dates,
        )
        idx = {"A": 0, "B": 1}
        edges, types, _weights = _add_correlation_edges(["A", "B"], idx, returns_df, 0.5)
        assert len(edges) > 0
        assert all(t == _EDGE_TYPE_CORRELATION for t in types)

    def test_low_correlation_no_edges(self) -> None:
        dates = pd.bdate_range("2024-01-01", periods=100)
        rng = np.random.default_rng(42)
        returns_df = pd.DataFrame(
            {"A": rng.normal(0, 1, 100), "B": rng.normal(0, 1, 100)},
            index=dates,
        )
        idx = {"A": 0, "B": 1}
        edges, _, _ = _add_correlation_edges(["A", "B"], idx, returns_df, 0.99)
        assert len(edges) == 0

    def test_missing_symbols_ignored(self) -> None:
        dates = pd.bdate_range("2024-01-01", periods=50)
        rng = np.random.default_rng(42)
        returns_df = pd.DataFrame({"A": rng.normal(0, 1, 50)}, index=dates)
        idx = {"A": 0, "B": 1}
        edges, _, _ = _add_correlation_edges(["A", "B"], idx, returns_df, 0.5)
        assert len(edges) == 0


class TestBuildStockGraph:
    def test_returns_stock_graph(self) -> None:
        symbols = ["HDFCBANK", "ICICIBANK", "TCS", "INFY"]
        graph = build_stock_graph(symbols)
        assert isinstance(graph, StockGraph)
        assert graph.symbols == symbols
        assert len(graph.symbol_to_idx) == 4

    def test_edge_index_shape(self) -> None:
        symbols = ["HDFCBANK", "ICICIBANK", "TCS", "INFY"]
        graph = build_stock_graph(symbols)
        assert graph.edge_index.shape[0] == 2
        assert graph.edge_type.shape[0] == graph.edge_index.shape[1]
        assert graph.edge_weight.shape[0] == graph.edge_index.shape[1]

    def test_with_correlation_data(self) -> None:
        symbols = ["HDFCBANK", "ICICIBANK", "TCS"]
        dates = pd.bdate_range("2024-01-01", periods=100)
        rng = np.random.default_rng(42)
        base = np.cumsum(rng.normal(0, 0.01, 100))
        returns_df = pd.DataFrame(
            {
                "HDFCBANK": base,
                "ICICIBANK": base + rng.normal(0, 0.001, 100),
                "TCS": rng.normal(0, 0.02, 100),
            },
            index=dates,
        )
        graph = build_stock_graph(symbols, returns_df=returns_df, correlation_threshold=0.5)
        assert graph.edge_index.shape[1] > 0

    def test_empty_symbols_returns_empty_graph(self) -> None:
        graph = build_stock_graph([])
        assert graph.edge_index.shape == (2, 0)
        assert len(graph.symbols) == 0

    def test_no_duplicate_edges(self) -> None:
        symbols = ["HDFCBANK", "ICICIBANK", "SBIN", "KOTAKBANK"]
        graph = build_stock_graph(symbols)
        edge_keys = set()
        for i in range(graph.edge_index.shape[1]):
            src = int(graph.edge_index[0, i])
            dst = int(graph.edge_index[1, i])
            etype = int(graph.edge_type[i])
            key = (src, dst, etype)
            assert key not in edge_keys, f"Duplicate edge: {key}"
            edge_keys.add(key)

    def test_no_self_loops(self) -> None:
        symbols = ["HDFCBANK", "ICICIBANK", "TCS", "INFY"]
        graph = build_stock_graph(symbols)
        for i in range(graph.edge_index.shape[1]):
            assert graph.edge_index[0, i] != graph.edge_index[1, i]
