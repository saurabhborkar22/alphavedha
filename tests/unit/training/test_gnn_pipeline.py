"""Tests for GNN training pipeline."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from alphavedha.exceptions import InsufficientDataError
from alphavedha.training.gnn_pipeline import _temporal_split, train_gnn


class TestTemporalSplit:
    def test_split_sizes(self) -> None:
        n = 200
        X = pd.DataFrame(np.random.randn(n, 10))
        y = pd.Series(np.random.choice([-1, 0, 1], n))
        X_train, _y_train, X_val, _y_val = _temporal_split(X, y, val_ratio=0.2, embargo_days=20)
        assert len(X_train) > 0
        assert len(X_val) > 0
        assert len(X_train) + 20 + len(X_val) <= n

    def test_no_overlap_with_embargo(self) -> None:
        n = 200
        X = pd.DataFrame(np.random.randn(n, 5))
        y = pd.Series(range(n))
        X_train, _, X_val, _ = _temporal_split(X, y, val_ratio=0.2, embargo_days=20)
        train_end = X_train.index[-1]
        val_start = X_val.index[0]
        assert val_start - train_end >= 20

    def test_small_dataset(self) -> None:
        X = pd.DataFrame(np.random.randn(100, 3))
        y = pd.Series(np.random.choice([0, 1], 100))
        X_train, _y_train, X_val, _y_val = _temporal_split(X, y, val_ratio=0.2, embargo_days=20)
        assert len(X_train) >= 50
        assert len(X_val) > 0


class TestTrainGNN:
    @pytest.mark.asyncio
    async def test_insufficient_symbols_raises(self) -> None:
        with pytest.raises(InsufficientDataError, match="at least 2 symbols"):
            await train_gnn(
                symbols=["TCS"],
                feature_df={"TCS": pd.DataFrame(np.random.randn(100, 5))},
                labels={"TCS": pd.Series(np.random.choice([0, 1], 100))},
            )

    @pytest.mark.asyncio
    async def test_missing_feature_data_raises(self) -> None:
        with pytest.raises(InsufficientDataError):
            await train_gnn(
                symbols=["TCS", "INFY"],
                feature_df={},
                labels={},
            )

    @pytest.mark.asyncio
    async def test_successful_training(self, tmp_path) -> None:
        rng = np.random.default_rng(42)
        n = 200
        dates = pd.bdate_range("2024-01-01", periods=n)

        feature_df = {
            "TCS": pd.DataFrame(rng.standard_normal((n, 10)), index=dates),
            "INFY": pd.DataFrame(rng.standard_normal((n, 10)), index=dates),
        }
        labels = {
            "TCS": pd.Series(rng.choice([-1, 0, 1], n), index=dates),
            "INFY": pd.Series(rng.choice([-1, 0, 1], n), index=dates),
        }

        mock_result = MagicMock()
        mock_result.train_metrics = {"accuracy": 0.6}
        mock_result.val_metrics = {"accuracy": 0.55}

        mock_model = MagicMock()
        mock_model.fit.return_value = mock_result

        with (
            patch("alphavedha.training.gnn_pipeline.GNNModel", return_value=mock_model),
            patch("alphavedha.training.gnn_pipeline.build_stock_graph") as mock_graph,
        ):
            mock_graph.return_value = MagicMock(
                edge_index=np.zeros((2, 0), dtype=np.int64),
            )
            result = await train_gnn(
                symbols=["TCS", "INFY"],
                feature_df=feature_df,
                labels=labels,
                output_dir=tmp_path / "gnn",
            )
            assert result.train_metrics["accuracy"] == 0.6
            mock_model.save.assert_called_once()
