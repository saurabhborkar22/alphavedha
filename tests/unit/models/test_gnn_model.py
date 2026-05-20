"""Tests for GNN model — GraphSAGE, stock graph, and full model lifecycle."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from alphavedha.data.stock_graph import (
    StockGraph,
    build_stock_graph,
)
from alphavedha.exceptions import ModelTrainingError
from alphavedha.models.base import PredictionResult, TrainResult
from alphavedha.models.gnn_model import GNNModel, GraphSAGELayer, StockGNN


@pytest.fixture
def small_symbols() -> list[str]:
    return ["TCS", "INFY", "HDFCBANK", "ICICIBANK", "RELIANCE", "TATAMOTORS", "TATASTEEL"]


@pytest.fixture
def returns_df(small_symbols: list[str]) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    n_days = 60
    data: dict[str, np.ndarray] = {}
    base = rng.standard_normal(n_days)
    for sym in small_symbols:
        data[sym] = base + rng.standard_normal(n_days) * 0.3
    return pd.DataFrame(data)


@pytest.fixture
def synthetic_data() -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(42)
    n, f = 100, 10
    X = pd.DataFrame(rng.standard_normal((n, f)), columns=[f"f{i}" for i in range(f)])
    y = pd.Series(rng.choice([-1, 0, 1], size=n, p=[0.3, 0.4, 0.3]), name="label")
    return X, y


@pytest.fixture
def small_graph(small_symbols: list[str]) -> StockGraph:
    return build_stock_graph(small_symbols)


@pytest.fixture
def gnn_config() -> dict[str, object]:
    return {
        "hidden_size": 16,
        "dropout": 0.1,
        "learning_rate": 0.01,
        "batch_size": 32,
        "max_epochs": 3,
        "early_stopping_patience": 2,
    }


class TestStockGraph:
    def test_sector_edges_created(self, small_symbols: list[str]) -> None:
        graph = build_stock_graph(small_symbols)
        assert graph.edge_index.shape[0] == 2
        assert graph.edge_index.shape[1] > 0
        assert (graph.edge_type == 0).any()

    def test_correlation_edges_with_high_corr(
        self, small_symbols: list[str], returns_df: pd.DataFrame,
    ) -> None:
        graph = build_stock_graph(
            small_symbols, returns_df=returns_df, correlation_threshold=0.3,
        )
        corr_mask = graph.edge_type == 1
        assert corr_mask.any()
        assert all(graph.edge_weight[corr_mask] >= 0.3)

    def test_promoter_edges_created(self, small_symbols: list[str]) -> None:
        graph = build_stock_graph(small_symbols)
        promoter_mask = graph.edge_type == 2
        assert promoter_mask.any()

    def test_empty_symbols(self) -> None:
        graph = build_stock_graph([])
        assert graph.edge_index.shape == (2, 0)
        assert len(graph.symbols) == 0

    def test_no_matching_symbols(self) -> None:
        graph = build_stock_graph(["UNKNOWN1", "UNKNOWN2"])
        assert graph.edge_index.shape[1] == 0
        assert len(graph.symbols) == 2

    def test_symbol_to_idx_mapping(self, small_symbols: list[str]) -> None:
        graph = build_stock_graph(small_symbols)
        assert len(graph.symbol_to_idx) == len(small_symbols)
        for sym in small_symbols:
            assert sym in graph.symbol_to_idx
        assert set(graph.symbol_to_idx.values()) == set(range(len(small_symbols)))

    def test_edges_are_bidirectional(self, small_symbols: list[str]) -> None:
        graph = build_stock_graph(small_symbols)
        edge_set = set()
        for i in range(graph.edge_index.shape[1]):
            edge_set.add((int(graph.edge_index[0, i]), int(graph.edge_index[1, i])))
        for src, dst in list(edge_set):
            assert (dst, src) in edge_set

    def test_edge_weight_shape_matches(self, small_symbols: list[str]) -> None:
        graph = build_stock_graph(small_symbols)
        n_edges = graph.edge_index.shape[1]
        assert graph.edge_type.shape == (n_edges,)
        assert graph.edge_weight.shape == (n_edges,)


class TestGraphSAGELayer:
    def test_output_shape(self) -> None:
        layer = GraphSAGELayer(in_features=10, out_features=16)
        x = torch.randn(5, 10)
        edge_index = torch.tensor([[0, 1, 2, 3], [1, 0, 3, 2]], dtype=torch.long)
        out = layer(x, edge_index)
        assert out.shape == (5, 16)

    def test_no_edges_fallback(self) -> None:
        layer = GraphSAGELayer(in_features=10, out_features=16)
        x = torch.randn(5, 10)
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        out = layer(x, edge_index)
        assert out.shape == (5, 16)

    def test_gradient_flows(self) -> None:
        layer = GraphSAGELayer(in_features=8, out_features=8)
        x = torch.randn(4, 8, requires_grad=True)
        edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
        out = layer(x, edge_index)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert x.grad.shape == (4, 8)


class TestStockGNN:
    def test_forward_shapes(self) -> None:
        model = StockGNN(in_features=10, hidden_size=16, num_classes=3, dropout=0.1)
        x = torch.randn(5, 10)
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)
        cls_logits, magnitude = model(x, edge_index)
        assert cls_logits.shape == (5, 3)
        assert magnitude.shape == (5,)

    def test_forward_no_edges(self) -> None:
        model = StockGNN(in_features=10, hidden_size=16)
        x = torch.randn(3, 10)
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        cls_logits, magnitude = model(x, edge_index)
        assert cls_logits.shape == (3, 3)
        assert magnitude.shape == (3,)


class TestGNNModel:
    def test_fit_returns_train_result(
        self,
        synthetic_data: tuple[pd.DataFrame, pd.Series],
        gnn_config: dict[str, object],
    ) -> None:
        X, y = synthetic_data
        model = GNNModel(config=gnn_config)
        result = model.fit(
            X_train=X[:80], y_train=y[:80],
            X_val=X[80:], y_val=y[80:],
        )
        assert isinstance(result, TrainResult)
        assert "accuracy" in result.train_metrics
        assert "f1_weighted" in result.train_metrics
        assert "rmse" in result.train_metrics

    def test_predict_returns_prediction_result(
        self,
        synthetic_data: tuple[pd.DataFrame, pd.Series],
        gnn_config: dict[str, object],
    ) -> None:
        X, y = synthetic_data
        model = GNNModel(config=gnn_config)
        model.fit(X_train=X[:80], y_train=y[:80])
        pred = model.predict(X[80:])
        assert isinstance(pred, PredictionResult)
        assert len(pred.direction) == 20
        assert len(pred.magnitude) == 20
        assert len(pred.confidence) == 20
        assert pred.probabilities is not None
        assert pred.probabilities.shape == (20, 3)

    def test_predict_with_graph(
        self,
        synthetic_data: tuple[pd.DataFrame, pd.Series],
        small_graph: StockGraph,
        gnn_config: dict[str, object],
    ) -> None:
        X, y = synthetic_data
        model = GNNModel(config=gnn_config)
        model.fit(X_train=X[:80], y_train=y[:80], graph=small_graph)
        pred = model.predict_with_graph(X[80:], small_graph)
        assert isinstance(pred, PredictionResult)
        assert len(pred.direction) == 20

    def test_direction_values(
        self,
        synthetic_data: tuple[pd.DataFrame, pd.Series],
        gnn_config: dict[str, object],
    ) -> None:
        X, y = synthetic_data
        model = GNNModel(config=gnn_config)
        model.fit(X_train=X[:80], y_train=y[:80])
        pred = model.predict(X[80:])
        assert set(np.unique(pred.direction)).issubset({-1, 0, 1})

    def test_probabilities_sum_to_one(
        self,
        synthetic_data: tuple[pd.DataFrame, pd.Series],
        gnn_config: dict[str, object],
    ) -> None:
        X, y = synthetic_data
        model = GNNModel(config=gnn_config)
        model.fit(X_train=X[:80], y_train=y[:80])
        pred = model.predict(X[80:])
        assert pred.probabilities is not None
        row_sums = pred.probabilities.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-5)

    def test_confidence_range(
        self,
        synthetic_data: tuple[pd.DataFrame, pd.Series],
        gnn_config: dict[str, object],
    ) -> None:
        X, y = synthetic_data
        model = GNNModel(config=gnn_config)
        model.fit(X_train=X[:80], y_train=y[:80])
        pred = model.predict(X[80:])
        assert (pred.confidence >= 0).all()
        assert (pred.confidence <= 1).all()

    def test_predict_before_fit_raises(self, gnn_config: dict[str, object]) -> None:
        model = GNNModel(config=gnn_config)
        X = pd.DataFrame({"a": range(10), "b": range(10)})
        with pytest.raises(ModelTrainingError):
            model.predict(X)

    def test_save_load_roundtrip(
        self,
        synthetic_data: tuple[pd.DataFrame, pd.Series],
        gnn_config: dict[str, object],
        tmp_path: Path,
    ) -> None:
        X, y = synthetic_data
        model = GNNModel(config=gnn_config)
        model.fit(X_train=X[:80], y_train=y[:80])
        pred_before = model.predict(X[80:])

        save_dir = tmp_path / "gnn_test"
        model.save(save_dir)

        loaded = GNNModel.load(save_dir)
        pred_after = loaded.predict(X[80:])

        np.testing.assert_array_equal(pred_before.direction, pred_after.direction)
        np.testing.assert_allclose(pred_before.magnitude, pred_after.magnitude, atol=1e-5)
        np.testing.assert_allclose(
            pred_before.probabilities, pred_after.probabilities, atol=1e-5,
        )

    def test_feature_importance_returns_none(
        self, gnn_config: dict[str, object],
    ) -> None:
        model = GNNModel(config=gnn_config)
        assert model.get_feature_importance() is None

    def test_sample_weight_accepted(
        self,
        synthetic_data: tuple[pd.DataFrame, pd.Series],
        gnn_config: dict[str, object],
    ) -> None:
        X, y = synthetic_data
        rng = np.random.default_rng(99)
        weights = pd.Series(rng.uniform(0.1, 5.0, size=80))
        model = GNNModel(config=gnn_config)
        result = model.fit(
            X_train=X[:80], y_train=y[:80], sample_weight=weights,
        )
        assert isinstance(result, TrainResult)

    def test_fallback_without_graph(
        self,
        synthetic_data: tuple[pd.DataFrame, pd.Series],
        gnn_config: dict[str, object],
    ) -> None:
        X, y = synthetic_data
        model = GNNModel(config=gnn_config)
        model.fit(X_train=X[:80], y_train=y[:80])
        pred_no_graph = model.predict(X[80:])
        assert pred_no_graph.direction is not None
        assert len(pred_no_graph.direction) == 20
