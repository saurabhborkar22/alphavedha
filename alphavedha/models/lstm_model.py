"""LSTMModel — LSTM network with dual heads (classification + regression)."""

from __future__ import annotations

import json
import os
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

from alphavedha.config import LSTMConfig
from alphavedha.exceptions import ModelNotFoundError, ModelTrainingError
from alphavedha.models.base import BaseModel, PredictionResult, TrainResult
from alphavedha.models.sequence_utils import (
    EarlyStopping,
    FastSequenceLoader,
    FeatureScaler,
    SequenceDataset,
    compute_combined_loss,
    create_data_loaders,
    get_device,
)

logger = structlog.get_logger(__name__)

_LABEL_REVERSE = {0: -1, 1: 0, 2: 1}


class LSTMNetwork(nn.Module):
    """2-layer LSTM with classification (3-class) and regression (magnitude) heads."""

    def __init__(
        self,
        n_features: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.cls_head = nn.Linear(hidden_size, 3)
        self.reg_head = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        lstm_out, _ = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]
        dropped = self.dropout(last_hidden)
        cls_logits = self.cls_head(dropped)
        reg_output = self.reg_head(dropped).squeeze(-1)
        return cls_logits, reg_output


class LSTMModel(BaseModel):
    """BaseModel wrapper around LSTMNetwork with safetensors serialization."""

    def __init__(self, config: LSTMConfig | None = None, name: str = "lstm") -> None:
        cfg = config or LSTMConfig()
        self._lstm_config = cfg
        self._network: LSTMNetwork | None = None
        self._scaler: FeatureScaler | None = None
        self._device = get_device()
        super().__init__(
            name=name,
            config={
                "hidden_size": cfg.hidden_size,
                "num_layers": cfg.num_layers,
                "dropout": cfg.dropout,
                "learning_rate": cfg.learning_rate,
                "sequence_length": cfg.sequence_length,
                "batch_size": cfg.batch_size,
                "max_epochs": cfg.max_epochs,
                "early_stopping_patience": cfg.early_stopping_patience,
            },
        )

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
        cfg = self._lstm_config
        self._feature_names = list(X_train.columns)

        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)
        if self._device.type == "cpu":
            torch.set_num_threads(os.cpu_count() or 1)

        if return_train is None:
            return_train = pd.Series(np.zeros(len(X_train)), index=X_train.index)
        if return_val is None and X_val is not None:
            return_val = pd.Series(np.zeros(len(X_val)), index=X_val.index)

        self._scaler = FeatureScaler.fit(X_train.values)
        X_train = pd.DataFrame(
            self._scaler.transform(X_train.values), index=X_train.index, columns=X_train.columns
        )
        if X_val is not None:
            X_val = pd.DataFrame(
                self._scaler.transform(X_val.values), index=X_val.index, columns=X_val.columns
            )

        train_loader, val_loader = create_data_loaders(
            X_train=X_train,
            y_train=y_train,
            return_train=return_train,
            X_val=X_val,
            y_val=y_val,
            return_val=return_val,
            sample_weight=sample_weight,
            sequence_length=cfg.sequence_length,
            batch_size=cfg.batch_size,
        )

        n_features = X_train.shape[1]
        self._network = LSTMNetwork(
            n_features=n_features,
            hidden_size=cfg.hidden_size,
            num_layers=cfg.num_layers,
            dropout=cfg.dropout,
        ).to(self._device)

        optimizer = torch.optim.Adam(self._network.parameters(), lr=cfg.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=3)
        early_stop = EarlyStopping(patience=cfg.early_stopping_patience, min_delta=1e-4)
        best_state: dict[str, torch.Tensor] | None = None

        # Inverse-frequency class weights — the neutral class (~15% of triple
        # barrier labels) is otherwise drowned out by up/down.
        y_arr = y_train.dropna().astype(int).to_numpy() + 1
        counts = np.bincount(np.clip(y_arr, 0, 2), minlength=3).astype(np.float64)
        class_weights = torch.tensor(
            counts.sum() / (3.0 * np.maximum(counts, 1.0)),
            dtype=torch.float32,
            device=self._device,
        )

        for epoch in range(cfg.max_epochs):
            self._network.train()
            train_losses: list[float] = []
            for X_seq, y_dir, y_mag, weights in train_loader:
                X_seq = X_seq.to(self._device)
                y_dir = y_dir.to(self._device)
                y_mag = y_mag.to(self._device)
                weights = weights.to(self._device) * class_weights[y_dir]

                optimizer.zero_grad()
                cls_logits, reg_output = self._network(X_seq)
                loss = compute_combined_loss(cls_logits, reg_output, y_dir, y_mag, weights)
                loss.backward()  # type: ignore[no-untyped-call]
                nn.utils.clip_grad_norm_(self._network.parameters(), max_norm=1.0)
                optimizer.step()
                train_losses.append(loss.item())

            if val_loader is not None:
                self._set_inference_mode()
                val_losses: list[float] = []
                with torch.no_grad():
                    for X_seq, y_dir, y_mag, weights in val_loader:
                        X_seq = X_seq.to(self._device)
                        y_dir = y_dir.to(self._device)
                        y_mag = y_mag.to(self._device)
                        weights = weights.to(self._device) * class_weights[y_dir]
                        cls_logits, reg_output = self._network(X_seq)
                        vloss = compute_combined_loss(cls_logits, reg_output, y_dir, y_mag, weights)
                        val_losses.append(vloss.item())

                avg_val_loss = float(np.mean(val_losses))
                avg_train_loss = float(np.mean(train_losses))
                scheduler.step(avg_val_loss)
                logger.info(
                    "lstm_epoch",
                    epoch=epoch + 1,
                    max_epochs=cfg.max_epochs,
                    train_loss=round(avg_train_loss, 4),
                    val_loss=round(avg_val_loss, 4),
                    best_val_loss=round(early_stop.best_loss, 4),
                    lr=optimizer.param_groups[0]["lr"],
                    elapsed_s=round(time.perf_counter() - start, 1),
                )
                if early_stop.step(avg_val_loss):
                    logger.info("early_stopping", epoch=epoch + 1, best_loss=early_stop.best_loss)
                    break
                if avg_val_loss <= early_stop.best_loss:
                    best_state = {k: v.cpu().clone() for k, v in self._network.state_dict().items()}

        if best_state is not None:
            self._network.load_state_dict(best_state)
            self._network.to(self._device)

        train_metrics = self._compute_metrics(train_loader)
        val_metrics = self._compute_metrics(val_loader) if val_loader is not None else {}

        elapsed = time.perf_counter() - start
        self._is_fitted = True
        self._train_metrics = train_metrics

        logger.info(
            "lstm_trained",
            train_metrics=train_metrics,
            val_metrics=val_metrics,
            training_time_s=round(elapsed, 2),
            n_train=len(X_train),
        )

        return TrainResult(
            train_metrics=train_metrics,
            val_metrics=val_metrics,
            feature_importances=None,
            training_time_seconds=elapsed,
            n_train_samples=len(X_train),
            n_val_samples=len(X_val) if X_val is not None else 0,
            hyperparams=dict(self._config),
        )

    def _set_inference_mode(self) -> None:
        """Switch network to inference mode (disables dropout, batch norm updates)."""
        if self._network is not None:
            self._network.train(False)

    def _compute_metrics(self, loader: FastSequenceLoader) -> dict[str, float]:
        if self._network is None:
            return {}
        self._set_inference_mode()
        all_y_dir: list[int] = []
        all_y_mag: list[float] = []
        all_cls_pred: list[int] = []
        all_reg_pred: list[float] = []

        with torch.no_grad():
            for X_seq, y_dir, y_mag, _ in loader:
                X_seq = X_seq.to(self._device)
                cls_logits, reg_output = self._network(X_seq)
                cls_pred = torch.argmax(cls_logits, dim=1)
                all_y_dir.extend(y_dir.tolist())
                all_y_mag.extend(y_mag.tolist())
                all_cls_pred.extend(cls_pred.cpu().tolist())
                all_reg_pred.extend(reg_output.cpu().tolist())

        y_dir_arr = np.array(all_y_dir)
        cls_pred_arr = np.array(all_cls_pred)
        y_mag_arr = np.array(all_y_mag)
        reg_pred_arr = np.array(all_reg_pred)

        return {
            "accuracy": float(accuracy_score(y_dir_arr, cls_pred_arr)),
            "f1_weighted": float(f1_score(y_dir_arr, cls_pred_arr, average="weighted")),
            "rmse": float(np.sqrt(mean_squared_error(y_mag_arr, reg_pred_arr))),
        }

    def predict(self, X: pd.DataFrame) -> PredictionResult:
        if not self._is_fitted or self._network is None:
            raise ModelTrainingError("LSTMModel is not fitted. Call fit() first.")

        cfg = self._lstm_config
        seq_len = cfg.sequence_length
        n_total = len(X)
        n_warmup = seq_len - 1

        X_arr = X.values.astype(np.float32)
        if self._scaler is not None:
            X_arr = self._scaler.transform(X_arr)
        dummy_dir = np.zeros(n_total, dtype=float)
        dummy_mag = np.zeros(n_total, dtype=np.float32)

        ds = SequenceDataset(
            X=X_arr,
            y_direction=dummy_dir,
            y_magnitude=dummy_mag,
            sequence_length=seq_len,
        )
        loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=False)

        all_proba: list[np.ndarray] = []
        all_reg: list[np.ndarray] = []

        self._set_inference_mode()
        with torch.no_grad():
            for X_seq, _, _, _ in loader:
                X_seq = X_seq.to(self._device)
                cls_logits, reg_output = self._network(X_seq)
                proba = torch.softmax(cls_logits, dim=1)
                all_proba.append(proba.cpu().numpy())
                all_reg.append(reg_output.cpu().numpy())

        proba_arr = np.concatenate(all_proba, axis=0)
        reg_arr = np.concatenate(all_reg, axis=0)

        cls_pred = np.argmax(proba_arr, axis=1)
        direction_valid = np.array([_LABEL_REVERSE[c] for c in cls_pred])
        confidence_valid = np.max(proba_arr, axis=1)

        direction = np.zeros(n_total, dtype=int)
        magnitude = np.zeros(n_total, dtype=np.float64)
        confidence = np.zeros(n_total, dtype=np.float64)
        probabilities = np.tile([0.0, 1.0, 0.0], (n_total, 1))

        direction[n_warmup:] = direction_valid
        magnitude[n_warmup:] = reg_arr
        confidence[n_warmup:] = confidence_valid
        probabilities[n_warmup:] = proba_arr

        return PredictionResult(
            direction=direction,
            magnitude=magnitude,
            probabilities=probabilities,
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
            "n_features": self._network.lstm.input_size,
            "hidden_size": self._lstm_config.hidden_size,
            "num_layers": self._lstm_config.num_layers,
            "dropout": self._lstm_config.dropout,
            "sequence_length": self._lstm_config.sequence_length,
            "scaler": self._scaler.to_dict() if self._scaler is not None else None,
        }
        (directory / "model_config.json").write_text(json.dumps(model_config, indent=2))

    @classmethod
    def _load_model_artifacts(cls, directory: Path, config: dict[str, Any]) -> LSTMModel:
        model = cls(config=None, name="lstm")

        config_path = directory / "model_config.json"
        if not config_path.exists():
            raise ModelNotFoundError(f"No model_config.json at {directory}")

        model_config = json.loads(config_path.read_text())

        safetensors_path = directory / "model.safetensors"
        if not safetensors_path.exists():
            raise ModelNotFoundError(f"No model.safetensors at {directory}")

        network = LSTMNetwork(
            n_features=model_config["n_features"],
            hidden_size=model_config["hidden_size"],
            num_layers=model_config["num_layers"],
            dropout=model_config["dropout"],
        )

        state_dict = load_file(safetensors_path)
        network.load_state_dict(state_dict)
        network.train(False)
        network.to(model._device)

        model._network = network
        scaler_dict = model_config.get("scaler")
        model._scaler = FeatureScaler.from_dict(scaler_dict) if scaler_dict else None
        model._lstm_config = LSTMConfig(
            hidden_size=model_config["hidden_size"],
            num_layers=model_config["num_layers"],
            dropout=model_config["dropout"],
            sequence_length=model_config["sequence_length"],
        )
        model._is_fitted = True

        return model
