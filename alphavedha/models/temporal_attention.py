"""Temporal Attention Model — TFT-lite with Variable Selection, GRN, multi-head attention."""

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
from torch.utils.data import DataLoader

from alphavedha.config import TFTConfig
from alphavedha.exceptions import ModelNotFoundError, ModelTrainingError
from alphavedha.models.base import BaseModel, PredictionResult, TrainResult
from alphavedha.models.sequence_utils import (
    EarlyStopping,
    MultiHorizonSequenceDataset,
    SequenceDataset,
    compute_combined_loss,
    get_device,
)

logger = structlog.get_logger(__name__)

_LABEL_REVERSE = {0: -1, 1: 0, 2: 1}

_HORIZON_LOSS_WEIGHTS = {0: 0.5, 1: 0.3, 2: 0.2}


class GatedResidualNetwork(nn.Module):
    """Gated transformation with skip connection and layer norm."""

    def __init__(
        self, input_size: int, hidden_size: int, output_size: int, dropout: float
    ) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.elu = nn.ELU()
        self.fc2 = nn.Linear(hidden_size, output_size * 2)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(output_size)
        self.skip = (
            nn.Linear(input_size, output_size) if input_size != output_size else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip = self.skip(x) if self.skip is not None else x
        h = self.fc1(x)
        h = self.elu(h)
        h = self.fc2(h)
        h = self.dropout(h)
        h1, h2 = h.chunk(2, dim=-1)
        h = h1 * torch.sigmoid(h2)
        return self.layer_norm(h + skip)


class VariableSelectionNetwork(nn.Module):
    """Learns per-feature importance weights using GRNs."""

    def __init__(self, n_features: int, hidden_size: int, dropout: float) -> None:
        super().__init__()
        self.n_features = n_features
        self.feature_grns = nn.ModuleList(
            [
                GatedResidualNetwork(1, hidden_size, hidden_size, dropout)
                for _ in range(n_features)
            ]
        )
        self.selection_grn = GatedResidualNetwork(
            n_features, hidden_size, n_features, dropout
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        weights = self.selection_grn(x)
        weights = torch.softmax(weights, dim=-1)

        transformed = []
        for i in range(self.n_features):
            feat = x[..., i : i + 1]
            transformed.append(self.feature_grns[i](feat))
        transformed_stack = torch.stack(transformed, dim=-2)

        weights_expanded = weights.unsqueeze(-1)
        output = (transformed_stack * weights_expanded).sum(dim=-2)

        return output, weights


class InterpretableMultiHeadAttention(nn.Module):
    """Multi-head attention returning interpretable attention weights."""

    def __init__(self, hidden_size: int, n_heads: int, dropout: float) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = hidden_size // n_heads

        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, _ = x.shape

        Q = (
            self.q_proj(x)
            .view(batch_size, seq_len, self.n_heads, self.head_dim)
            .transpose(1, 2)
        )
        K = (
            self.k_proj(x)
            .view(batch_size, seq_len, self.n_heads, self.head_dim)
            .transpose(1, 2)
        )
        V = (
            self.v_proj(x)
            .view(batch_size, seq_len, self.n_heads, self.head_dim)
            .transpose(1, 2)
        )

        scale = self.head_dim**0.5
        scores = torch.matmul(Q, K.transpose(-2, -1)) / scale
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context = torch.matmul(attn_weights, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        output = self.out_proj(context)

        return output, attn_weights


class TemporalAttentionNetwork(nn.Module):
    """Full TFT-lite: VSN -> LSTM encoder -> multi-head attention -> per-horizon heads."""

    def __init__(
        self,
        n_features: int,
        hidden_size: int,
        num_layers: int,
        n_heads: int,
        dropout: float,
        horizons: list[int],
    ) -> None:
        super().__init__()
        self.horizons = horizons

        self.vsn = VariableSelectionNetwork(n_features, hidden_size, dropout)
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.attention = InterpretableMultiHeadAttention(hidden_size, n_heads, dropout)
        self.dropout = nn.Dropout(dropout)

        self.cls_heads = nn.ModuleDict(
            {str(h): nn.Linear(hidden_size, 3) for h in horizons}
        )
        self.reg_heads = nn.ModuleDict(
            {str(h): nn.Linear(hidden_size, 1) for h in horizons}
        )

    def forward(
        self, x: torch.Tensor
    ) -> tuple[
        dict[int, torch.Tensor],
        dict[int, torch.Tensor],
        torch.Tensor,
        torch.Tensor,
    ]:
        batch_size, seq_len, _ = x.shape

        x_flat = x.reshape(-1, x.shape[-1])
        selected, feature_weights = self.vsn(x_flat)
        selected = selected.reshape(batch_size, seq_len, -1)
        feature_weights = feature_weights.reshape(batch_size, seq_len, -1)

        lstm_out, _ = self.lstm(selected)

        context, attn_weights = self.attention(lstm_out)
        context = self.dropout(context)

        last_context = context[:, -1, :]

        cls_outputs: dict[int, torch.Tensor] = {}
        reg_outputs: dict[int, torch.Tensor] = {}
        for h in self.horizons:
            cls_outputs[h] = self.cls_heads[str(h)](last_context)
            reg_outputs[h] = self.reg_heads[str(h)](last_context).squeeze(-1)

        avg_feature_weights = feature_weights.mean(dim=1)

        return cls_outputs, reg_outputs, avg_feature_weights, attn_weights


class TemporalAttentionModel(BaseModel):
    """BaseModel wrapper around TemporalAttentionNetwork with multi-horizon output."""

    def __init__(
        self, config: TFTConfig | None = None, name: str = "temporal_attention"
    ) -> None:
        cfg = config or TFTConfig()
        self._tft_config = cfg
        self._network: TemporalAttentionNetwork | None = None
        self._device = get_device()
        self._last_horizons: dict[int, PredictionResult] = {}
        self._last_attention_weights: np.ndarray | None = None
        self._last_feature_weights: np.ndarray | None = None
        super().__init__(
            name=name,
            config={
                "hidden_size": cfg.hidden_size,
                "attention_head_size": cfg.attention_head_size,
                "num_layers": cfg.num_layers,
                "dropout": cfg.dropout,
                "learning_rate": cfg.learning_rate,
                "sequence_length": cfg.sequence_length,
                "batch_size": cfg.batch_size,
                "max_epochs": cfg.max_epochs,
                "early_stopping_patience": cfg.early_stopping_patience,
                "horizons": cfg.horizons,
            },
        )

    def _set_inference_mode(self) -> None:
        """Switch network to inference mode."""
        if self._network is not None:
            self._network.train(False)

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame | None = None,
        y_val: pd.Series | None = None,
        sample_weight: pd.Series | None = None,
        return_train: pd.Series | None = None,
        return_val: pd.Series | None = None,
    ) -> TrainResult:
        start = time.perf_counter()
        cfg = self._tft_config
        self._feature_names = list(X_train.columns)

        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)

        if return_train is None:
            return_train = pd.Series(np.zeros(len(X_train)), index=X_train.index)
        if return_val is None and X_val is not None:
            return_val = pd.Series(np.zeros(len(X_val)), index=X_val.index)

        train_ds = MultiHorizonSequenceDataset(
            X=X_train.values,
            y_direction=y_train.values,
            y_magnitude=return_train.values,
            sequence_length=cfg.sequence_length,
            horizons=cfg.horizons,
            sample_weight=(
                sample_weight.values if sample_weight is not None else None
            ),
        )
        train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)

        val_loader = None
        if X_val is not None and y_val is not None and return_val is not None:
            val_ds = MultiHorizonSequenceDataset(
                X=X_val.values,
                y_direction=y_val.values,
                y_magnitude=return_val.values,
                sequence_length=cfg.sequence_length,
                horizons=cfg.horizons,
            )
            val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False)

        n_features = X_train.shape[1]
        self._network = TemporalAttentionNetwork(
            n_features=n_features,
            hidden_size=cfg.hidden_size,
            num_layers=cfg.num_layers,
            n_heads=cfg.attention_head_size,
            dropout=cfg.dropout,
            horizons=cfg.horizons,
        ).to(self._device)

        optimizer = torch.optim.Adam(
            self._network.parameters(), lr=cfg.learning_rate
        )
        early_stop = EarlyStopping(patience=cfg.early_stopping_patience)
        best_state: dict[str, torch.Tensor] | None = None
        horizons_sorted = sorted(cfg.horizons)

        for epoch in range(cfg.max_epochs):
            self._network.train()
            for X_seq, _y_dir, _y_mag, weights, h_dirs, h_mags, h_masks in train_loader:
                X_seq = X_seq.to(self._device)
                weights = weights.to(self._device)
                h_dirs = h_dirs.to(self._device)
                h_mags = h_mags.to(self._device)
                h_masks = h_masks.to(self._device)

                optimizer.zero_grad()
                cls_outputs, reg_outputs, _, _ = self._network(X_seq)

                total_loss = torch.tensor(0.0, device=self._device)
                for j, h in enumerate(horizons_sorted):
                    mask = h_masks[:, j]
                    if mask.sum() == 0:
                        continue
                    h_cls = cls_outputs[h][mask]
                    h_reg = reg_outputs[h][mask]
                    h_y_dir = h_dirs[:, j][mask]
                    h_y_mag = h_mags[:, j][mask]
                    h_w = weights[mask]
                    h_loss = compute_combined_loss(
                        h_cls, h_reg, h_y_dir, h_y_mag, h_w
                    )
                    total_loss = total_loss + _HORIZON_LOSS_WEIGHTS[j] * h_loss

                total_loss.backward()
                nn.utils.clip_grad_norm_(
                    self._network.parameters(), max_norm=1.0
                )
                optimizer.step()

            if val_loader is not None:
                self._set_inference_mode()
                val_losses: list[float] = []
                with torch.no_grad():
                    for (
                        X_seq,
                        _y_dir,
                        _y_mag,
                        weights,
                        h_dirs,
                        h_mags,
                        h_masks,
                    ) in val_loader:
                        X_seq = X_seq.to(self._device)
                        weights = weights.to(self._device)
                        h_dirs = h_dirs.to(self._device)
                        h_mags = h_mags.to(self._device)
                        h_masks = h_masks.to(self._device)

                        cls_outputs, reg_outputs, _, _ = self._network(X_seq)

                        vloss = torch.tensor(0.0, device=self._device)
                        for j, h in enumerate(horizons_sorted):
                            mask = h_masks[:, j]
                            if mask.sum() == 0:
                                continue
                            h_cls = cls_outputs[h][mask]
                            h_reg = reg_outputs[h][mask]
                            h_y_dir = h_dirs[:, j][mask]
                            h_y_mag = h_mags[:, j][mask]
                            h_w = weights[mask]
                            h_loss = compute_combined_loss(
                                h_cls, h_reg, h_y_dir, h_y_mag, h_w
                            )
                            vloss = vloss + _HORIZON_LOSS_WEIGHTS[j] * h_loss
                        val_losses.append(vloss.item())

                avg_val_loss = float(np.mean(val_losses))
                if early_stop.step(avg_val_loss):
                    logger.info(
                        "early_stopping",
                        epoch=epoch,
                        best_loss=early_stop.best_loss,
                    )
                    break
                if avg_val_loss <= early_stop.best_loss:
                    best_state = {
                        k: v.cpu().clone()
                        for k, v in self._network.state_dict().items()
                    }

        if best_state is not None:
            self._network.load_state_dict(best_state)
            self._network.to(self._device)

        train_metrics = self._compute_metrics(train_loader)
        val_metrics = (
            self._compute_metrics(val_loader) if val_loader is not None else {}
        )

        elapsed = time.perf_counter() - start
        self._is_fitted = True
        self._train_metrics = train_metrics

        logger.info(
            "tft_trained",
            train_metrics=train_metrics,
            val_metrics=val_metrics,
            training_time_s=round(elapsed, 2),
            n_train=len(X_train),
        )

        return TrainResult(
            train_metrics=train_metrics,
            val_metrics=val_metrics,
            feature_importances=self.get_feature_importance(),
            training_time_seconds=elapsed,
            n_train_samples=len(X_train),
            n_val_samples=len(X_val) if X_val is not None else 0,
            hyperparams=dict(self._config),
        )

    def _compute_metrics(self, loader: DataLoader) -> dict[str, float]:  # type: ignore[type-arg]
        if self._network is None:
            return {}
        self._set_inference_mode()
        all_y_dir: list[int] = []
        all_y_mag: list[float] = []
        all_cls_pred: list[int] = []
        all_reg_pred: list[float] = []

        primary_h = min(self._tft_config.horizons)

        with torch.no_grad():
            for batch in loader:
                X_seq = batch[0].to(self._device)
                y_dir = batch[1]
                y_mag = batch[2]

                cls_outputs, reg_outputs, feat_w, _ = self._network(X_seq)
                cls_logits = cls_outputs[primary_h]
                reg_output = reg_outputs[primary_h]
                cls_pred = torch.argmax(cls_logits, dim=1)

                all_y_dir.extend(y_dir.tolist())
                all_y_mag.extend(y_mag.tolist())
                all_cls_pred.extend(cls_pred.cpu().tolist())
                all_reg_pred.extend(reg_output.cpu().tolist())

                self._last_feature_weights = feat_w.cpu().numpy().mean(axis=0)

        y_dir_arr = np.array(all_y_dir)
        cls_pred_arr = np.array(all_cls_pred)
        y_mag_arr = np.array(all_y_mag)
        reg_pred_arr = np.array(all_reg_pred)

        return {
            "accuracy": float(accuracy_score(y_dir_arr, cls_pred_arr)),
            "f1_weighted": float(
                f1_score(y_dir_arr, cls_pred_arr, average="weighted")
            ),
            "rmse": float(np.sqrt(mean_squared_error(y_mag_arr, reg_pred_arr))),
        }

    def predict(self, X: pd.DataFrame) -> PredictionResult:
        if not self._is_fitted or self._network is None:
            raise ModelTrainingError(
                "TemporalAttentionModel is not fitted. Call fit() first."
            )

        cfg = self._tft_config
        seq_len = cfg.sequence_length
        n_total = len(X)
        n_warmup = seq_len - 1

        X_arr = X.values.astype(np.float32)
        dummy_dir = np.zeros(n_total, dtype=float)
        dummy_mag = np.zeros(n_total, dtype=np.float32)

        ds = SequenceDataset(
            X=X_arr,
            y_direction=dummy_dir,
            y_magnitude=dummy_mag,
            sequence_length=seq_len,
        )
        loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=False)

        per_horizon_proba: dict[int, list[np.ndarray]] = {
            h: [] for h in cfg.horizons
        }
        per_horizon_reg: dict[int, list[np.ndarray]] = {
            h: [] for h in cfg.horizons
        }
        all_attn: list[np.ndarray] = []
        all_feat_w: list[np.ndarray] = []

        self._set_inference_mode()
        with torch.no_grad():
            for X_seq, _, _, _ in loader:
                X_seq = X_seq.to(self._device)
                cls_outputs, reg_outputs, feat_w, attn_w = self._network(X_seq)

                for h in cfg.horizons:
                    proba = torch.softmax(cls_outputs[h], dim=1)
                    per_horizon_proba[h].append(proba.cpu().numpy())
                    per_horizon_reg[h].append(reg_outputs[h].cpu().numpy())

                all_attn.append(attn_w.cpu().numpy())
                all_feat_w.append(feat_w.cpu().numpy())

        self._last_attention_weights = np.concatenate(all_attn, axis=0)
        feat_w_all = np.concatenate(all_feat_w, axis=0)
        self._last_feature_weights = feat_w_all.mean(axis=0)

        self._last_horizons = {}
        for h in cfg.horizons:
            proba_arr = np.concatenate(per_horizon_proba[h], axis=0)
            reg_arr = np.concatenate(per_horizon_reg[h], axis=0)

            cls_pred = np.argmax(proba_arr, axis=1)
            dir_valid = np.array([_LABEL_REVERSE[c] for c in cls_pred])
            conf_valid = np.max(proba_arr, axis=1)

            direction = np.zeros(n_total, dtype=int)
            magnitude = np.zeros(n_total, dtype=np.float64)
            confidence = np.zeros(n_total, dtype=np.float64)
            probabilities = np.tile([0.0, 1.0, 0.0], (n_total, 1))

            direction[n_warmup:] = dir_valid
            magnitude[n_warmup:] = reg_arr
            confidence[n_warmup:] = conf_valid
            probabilities[n_warmup:] = proba_arr

            self._last_horizons[h] = PredictionResult(
                direction=direction,
                magnitude=magnitude,
                probabilities=probabilities,
                confidence=confidence,
            )

        primary_h = min(cfg.horizons)
        return self._last_horizons[primary_h]

    def get_feature_importance(self) -> pd.Series | None:
        if self._last_feature_weights is None:
            return None
        return pd.Series(
            self._last_feature_weights,
            index=self._feature_names,
            name="importance",
        )

    def get_horizon_predictions(self) -> dict[int, PredictionResult]:
        return dict(self._last_horizons)

    def get_attention_weights(self) -> np.ndarray | None:
        return self._last_attention_weights

    def _save_model_artifacts(self, directory: Path) -> None:
        if self._network is None:
            return

        state_dict = {
            k: v.cpu() for k, v in self._network.state_dict().items()
        }
        save_file(state_dict, directory / "model.safetensors")

        model_config = {
            "n_features": self._network.vsn.n_features,
            "hidden_size": self._tft_config.hidden_size,
            "num_layers": self._tft_config.num_layers,
            "n_heads": self._tft_config.attention_head_size,
            "dropout": self._tft_config.dropout,
            "sequence_length": self._tft_config.sequence_length,
            "horizons": self._tft_config.horizons,
        }
        (directory / "model_config.json").write_text(
            json.dumps(model_config, indent=2)
        )

    @classmethod
    def _load_model_artifacts(
        cls, directory: Path, config: dict[str, Any]
    ) -> TemporalAttentionModel:
        model = cls(config=None, name="temporal_attention")

        config_path = directory / "model_config.json"
        if not config_path.exists():
            raise ModelNotFoundError(f"No model_config.json at {directory}")

        model_config = json.loads(config_path.read_text())

        safetensors_path = directory / "model.safetensors"
        if not safetensors_path.exists():
            raise ModelNotFoundError(f"No model.safetensors at {directory}")

        network = TemporalAttentionNetwork(
            n_features=model_config["n_features"],
            hidden_size=model_config["hidden_size"],
            num_layers=model_config["num_layers"],
            n_heads=model_config["n_heads"],
            dropout=model_config["dropout"],
            horizons=model_config["horizons"],
        )

        state_dict = load_file(safetensors_path)
        network.load_state_dict(state_dict)
        network.train(False)
        network.to(model._device)

        model._network = network
        model._tft_config = TFTConfig(
            hidden_size=model_config["hidden_size"],
            num_layers=model_config["num_layers"],
            attention_head_size=model_config["n_heads"],
            dropout=model_config["dropout"],
            sequence_length=model_config["sequence_length"],
            horizons=model_config["horizons"],
        )
        model._feature_names = [
            f"f{i}" for i in range(model_config["n_features"])
        ]
        model._is_fitted = True

        return model
