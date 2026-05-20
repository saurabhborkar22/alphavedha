"""GNN Model — GraphSAGE for stock relationship modeling (pure PyTorch)."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import structlog
import torch
from safetensors.torch import load_file, save_file
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error
from torch import nn

from alphavedha.data.stock_graph import StockGraph
from alphavedha.exceptions import ModelNotFoundError, ModelTrainingError
from alphavedha.models.base import BaseModel, PredictionResult, TrainResult
from alphavedha.models.sequence_utils import EarlyStopping, get_device

logger = structlog.get_logger(__name__)

_LABEL_MAP = {-1: 0, 0: 1, 1: 2}
_LABEL_REVERSE = {0: -1, 1: 0, 2: 1}

_DEFAULT_CONFIG: dict[str, Any] = {
    "hidden_size": 64,
    "num_layers": 2,
    "dropout": 0.3,
    "learning_rate": 0.001,
    "batch_size": 32,
    "max_epochs": 100,
    "early_stopping_patience": 10,
}


class GraphSAGELayer(nn.Module):
    """Single GraphSAGE convolution: aggregate neighbors -> concat -> transform."""

    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features * 2, out_features)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        n_nodes = x.size(0)
        src, dst = edge_index[0], edge_index[1]

        if edge_index.size(1) == 0:
            neighbor_agg = torch.zeros_like(x)
        else:
            neighbor_sum = torch.zeros(n_nodes, x.size(1), device=x.device, dtype=x.dtype)
            neighbor_count = torch.zeros(n_nodes, 1, device=x.device, dtype=x.dtype)

            neighbor_sum.index_add_(0, dst, x[src])
            ones = torch.ones(src.size(0), 1, device=x.device, dtype=x.dtype)
            neighbor_count.index_add_(0, dst, ones)

            neighbor_count = neighbor_count.clamp(min=1.0)
            neighbor_agg = neighbor_sum / neighbor_count

        combined = torch.cat([x, neighbor_agg], dim=1)
        return self.relu(self.linear(combined))


class StockGNN(nn.Module):
    """2-layer GraphSAGE with classification + regression heads."""

    def __init__(
        self,
        in_features: int,
        hidden_size: int = 64,
        num_classes: int = 3,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.layer1 = GraphSAGELayer(in_features, hidden_size)
        self.layer2 = GraphSAGELayer(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.cls_head = nn.Linear(hidden_size, num_classes)
        self.reg_head = nn.Linear(hidden_size, 1)

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.layer1(x, edge_index)
        h = self.dropout(h)
        h = self.layer2(h, edge_index)
        h = self.dropout(h)
        cls_logits = self.cls_head(h)
        magnitude = self.reg_head(h).squeeze(-1)
        return cls_logits, magnitude


class GNNModel(BaseModel):
    """GNN model following BaseModel interface."""

    def __init__(
        self,
        name: str = "gnn",
        config: dict[str, Any] | None = None,
    ) -> None:
        cfg = {**_DEFAULT_CONFIG, **(config or {})}
        self._gnn_config = cfg
        self._network: StockGNN | None = None
        self._device = get_device()
        super().__init__(name=name, config=cfg)

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame | None = None,
        y_val: pd.Series | None = None,
        sample_weight: pd.Series | None = None,
        graph: StockGraph | None = None,
    ) -> TrainResult:
        start = time.perf_counter()
        cfg = self._gnn_config
        self._feature_names = list(X_train.columns)

        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)

        n_features = X_train.shape[1]
        hidden_size = cfg["hidden_size"]
        dropout = cfg["dropout"]

        self._network = StockGNN(
            in_features=n_features,
            hidden_size=hidden_size,
            num_classes=3,
            dropout=dropout,
        ).to(self._device)

        edge_index = self._prepare_edge_index(graph)

        X_train_t = torch.tensor(
            X_train.values.astype(np.float32), device=self._device,
        )
        y_dir_train = self._map_labels(y_train)
        y_mag_train = torch.tensor(
            np.zeros(len(y_train), dtype=np.float32), device=self._device,
        )
        weights_train = (
            torch.tensor(sample_weight.values.astype(np.float32), device=self._device)
            if sample_weight is not None
            else torch.ones(len(y_train), device=self._device)
        )

        X_val_t: torch.Tensor | None = None
        y_dir_val: torch.Tensor | None = None
        y_mag_val: torch.Tensor | None = None
        if X_val is not None and y_val is not None:
            X_val_t = torch.tensor(X_val.values.astype(np.float32), device=self._device)
            y_dir_val = self._map_labels(y_val)
            y_mag_val = torch.zeros(len(y_val), device=self._device)

        optimizer = torch.optim.Adam(
            self._network.parameters(), lr=cfg["learning_rate"],
        )
        early_stop = EarlyStopping(patience=cfg["early_stopping_patience"])
        best_state: dict[str, torch.Tensor] | None = None
        best_val_loss = float("inf")
        n_train = len(X_train_t)

        for epoch in range(cfg["max_epochs"]):
            self._network.train()

            cls_logits, reg_output = self._network(X_train_t, edge_index)
            ce_loss = nn.functional.cross_entropy(
                cls_logits, y_dir_train, reduction="none",
            )
            mse_loss = nn.functional.mse_loss(
                reg_output, y_mag_train, reduction="none",
            )
            loss = (0.7 * (ce_loss * weights_train).mean()
                    + 0.3 * (mse_loss * weights_train).mean())

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self._network.parameters(), max_norm=1.0)
            optimizer.step()

            if X_val_t is not None and y_dir_val is not None and y_mag_val is not None:
                self._set_inference_mode()
                with torch.no_grad():
                    val_cls, val_reg = self._network(X_val_t, edge_index)
                    val_ce = nn.functional.cross_entropy(val_cls, y_dir_val)
                    val_mse = nn.functional.mse_loss(val_reg, y_mag_val)
                    val_loss = 0.7 * val_ce + 0.3 * val_mse

                current_val_loss = val_loss.item()
                if early_stop.step(current_val_loss):
                    logger.info(
                        "gnn_early_stopping",
                        epoch=epoch,
                        best_loss=best_val_loss,
                    )
                    break
                if current_val_loss < best_val_loss:
                    best_val_loss = current_val_loss
                    best_state = {
                        k: v.cpu().clone()
                        for k, v in self._network.state_dict().items()
                    }

        if best_state is not None:
            self._network.load_state_dict(best_state)
            self._network.to(self._device)

        train_metrics = self._compute_metrics(X_train_t, y_dir_train, edge_index)
        val_metrics = (
            self._compute_metrics(X_val_t, y_dir_val, edge_index)
            if X_val_t is not None and y_dir_val is not None
            else {}
        )

        elapsed = time.perf_counter() - start
        self._is_fitted = True
        self._train_metrics = train_metrics

        logger.info(
            "gnn_trained",
            train_metrics=train_metrics,
            val_metrics=val_metrics,
            training_time_s=round(elapsed, 2),
            n_train=n_train,
        )

        return TrainResult(
            train_metrics=train_metrics,
            val_metrics=val_metrics,
            feature_importances=None,
            training_time_seconds=elapsed,
            n_train_samples=n_train,
            n_val_samples=len(X_val) if X_val is not None else 0,
            hyperparams=dict(cfg),
        )

    def predict(self, X: pd.DataFrame) -> PredictionResult:
        """Predict without graph — falls back to MLP (no neighbor aggregation)."""
        return self._predict_internal(X, graph=None)

    def predict_with_graph(
        self, X: pd.DataFrame, graph: StockGraph,
    ) -> PredictionResult:
        """Predict with graph structure for neighbor aggregation."""
        return self._predict_internal(X, graph=graph)

    def _predict_internal(
        self, X: pd.DataFrame, graph: StockGraph | None,
    ) -> PredictionResult:
        if not self._is_fitted or self._network is None:
            raise ModelTrainingError("GNNModel is not fitted. Call fit() first.")

        self._set_inference_mode()
        edge_index = self._prepare_edge_index(graph)

        X_t = torch.tensor(X.values.astype(np.float32), device=self._device)

        with torch.no_grad():
            cls_logits, reg_output = self._network(X_t, edge_index)
            proba = torch.softmax(cls_logits, dim=1).cpu().numpy()
            magnitude = reg_output.cpu().numpy()

        cls_pred = np.argmax(proba, axis=1)
        direction = np.array([_LABEL_REVERSE[c] for c in cls_pred])
        confidence = np.max(proba, axis=1)

        return PredictionResult(
            direction=direction,
            magnitude=magnitude,
            probabilities=proba,
            confidence=confidence,
        )

    def get_feature_importance(self) -> pd.Series | None:
        return None

    def _save_model_artifacts(self, directory: Path) -> None:
        if self._network is None:
            return

        state_dict = {k: v.cpu() for k, v in self._network.state_dict().items()}
        save_file(state_dict, directory / "model.safetensors")

        model_config = {
            "n_features": len(self._feature_names),
            "hidden_size": self._gnn_config["hidden_size"],
            "dropout": self._gnn_config["dropout"],
        }
        (directory / "model_config.json").write_text(json.dumps(model_config, indent=2))

    @classmethod
    def _load_model_artifacts(
        cls, directory: Path, config: dict[str, Any],
    ) -> GNNModel:
        model = cls(name="gnn", config=config)

        config_path = directory / "model_config.json"
        if not config_path.exists():
            raise ModelNotFoundError(f"No model_config.json at {directory}")

        model_config = json.loads(config_path.read_text())

        safetensors_path = directory / "model.safetensors"
        if not safetensors_path.exists():
            raise ModelNotFoundError(f"No model.safetensors at {directory}")

        network = StockGNN(
            in_features=model_config["n_features"],
            hidden_size=model_config["hidden_size"],
            dropout=model_config["dropout"],
        )

        state_dict = load_file(safetensors_path)
        network.load_state_dict(state_dict)
        network.to(model._device)

        model._network = network
        model._set_inference_mode()
        model._gnn_config = {
            **_DEFAULT_CONFIG,
            "hidden_size": model_config["hidden_size"],
            "dropout": model_config["dropout"],
        }
        model._is_fitted = True

        return model

    def _set_inference_mode(self) -> None:
        """Switch network to inference mode (disables dropout)."""
        if self._network is not None:
            self._network.train(False)

    def _prepare_edge_index(self, graph: StockGraph | None) -> torch.Tensor:
        """Convert graph edge_index to torch tensor, or return empty edges."""
        if graph is not None and graph.edge_index.shape[1] > 0:
            return torch.tensor(graph.edge_index, dtype=torch.long, device=self._device)
        return torch.zeros((2, 0), dtype=torch.long, device=self._device)

    def _map_labels(self, y: pd.Series) -> torch.Tensor:
        """Map direction labels {-1, 0, 1} to class indices {0, 1, 2}."""
        mapped = np.array([_LABEL_MAP.get(int(v), 1) for v in y.values], dtype=np.int64)
        return torch.tensor(mapped, device=self._device)

    def _compute_metrics(
        self,
        X_t: torch.Tensor,
        y_dir: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> dict[str, float]:
        if self._network is None:
            return {}
        self._set_inference_mode()
        with torch.no_grad():
            cls_logits, reg_output = self._network(X_t, edge_index)
            cls_pred = torch.argmax(cls_logits, dim=1).cpu().numpy()
            y_true = y_dir.cpu().numpy()
            reg_pred = reg_output.cpu().numpy()

        return {
            "accuracy": float(accuracy_score(y_true, cls_pred)),
            "f1_weighted": float(f1_score(y_true, cls_pred, average="weighted")),
            "rmse": float(np.sqrt(mean_squared_error(
                np.zeros_like(reg_pred), reg_pred,
            ))),
        }
