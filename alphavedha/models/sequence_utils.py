"""Shared sequence utilities for PyTorch deep learning models."""

from __future__ import annotations

import numpy as np
import pandas as pd
import structlog
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from alphavedha.exceptions import InsufficientDataError

logger = structlog.get_logger(__name__)

_LABEL_MAP = {-1: 0, 0: 1, 1: 2}


class SequenceDataset(Dataset):  # type: ignore[type-arg]
    """Creates fixed-length sliding windows from tabular data for single-horizon models."""

    def __init__(
        self,
        X: np.ndarray,
        y_direction: np.ndarray,
        y_magnitude: np.ndarray,
        sequence_length: int = 60,
        sample_weight: np.ndarray | None = None,
    ) -> None:
        n_samples = X.shape[0]
        if n_samples < sequence_length:
            raise InsufficientDataError(
                f"Need at least {sequence_length} samples, got {n_samples}"
            )

        self._X = X.astype(np.float32)
        self._sequence_length = sequence_length

        y_dir_arr = np.asarray(y_direction, dtype=np.float64)
        nan_mask = np.isnan(y_dir_arr)
        mapped_dir = np.ones(len(y_dir_arr), dtype=np.int64)
        for i in range(len(y_dir_arr)):
            if not nan_mask[i]:
                mapped_dir[i] = _LABEL_MAP.get(int(y_dir_arr[i]), 1)
        self._y_direction = mapped_dir
        self._y_magnitude = np.asarray(y_magnitude, dtype=np.float32)
        self._weights = (
            np.asarray(sample_weight, dtype=np.float32)
            if sample_weight is not None
            else np.ones(n_samples, dtype=np.float32)
        )

        n_windows = n_samples - sequence_length + 1
        self._valid_indices: list[int] = []
        for i in range(n_windows):
            label_idx = i + sequence_length - 1
            if not nan_mask[label_idx]:
                self._valid_indices.append(i)

        if len(self._valid_indices) == 0:
            raise InsufficientDataError("No valid training samples after NaN filtering")

        logger.debug(
            "sequence_dataset_created",
            n_sequences=len(self._valid_indices),
            sequence_length=sequence_length,
            n_features=X.shape[1],
        )

    def __len__(self) -> int:
        return len(self._valid_indices)

    def __getitem__(
        self, idx: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        i = self._valid_indices[idx]
        label_idx = i + self._sequence_length - 1

        X_seq = torch.from_numpy(self._X[i : i + self._sequence_length])
        y_dir = torch.tensor(self._y_direction[label_idx], dtype=torch.long)
        y_mag = torch.tensor(self._y_magnitude[label_idx], dtype=torch.float32)
        weight = torch.tensor(self._weights[label_idx], dtype=torch.float32)

        return X_seq, y_dir, y_mag, weight


class MultiHorizonSequenceDataset(Dataset):  # type: ignore[type-arg]
    """Sliding windows with per-horizon labels and validity masks for TFT training."""

    def __init__(
        self,
        X: np.ndarray,
        y_direction: np.ndarray,
        y_magnitude: np.ndarray,
        sequence_length: int,
        horizons: list[int],
        sample_weight: np.ndarray | None = None,
    ) -> None:
        n_samples = X.shape[0]
        if n_samples < sequence_length:
            raise InsufficientDataError(
                f"Need at least {sequence_length} samples, got {n_samples}"
            )

        self._X = X.astype(np.float32)
        self._seq_len = sequence_length
        self._horizons = sorted(horizons)

        base_horizon = min(horizons)
        self._offsets = [h - base_horizon for h in self._horizons]

        y_dir_arr = np.asarray(y_direction, dtype=np.float64)
        nan_mask = np.isnan(y_dir_arr)
        mapped_dir = np.ones(len(y_dir_arr), dtype=np.int64)
        for i in range(len(y_dir_arr)):
            if not nan_mask[i]:
                mapped_dir[i] = _LABEL_MAP.get(int(y_dir_arr[i]), 1)
        self._y_dir = mapped_dir
        self._y_mag = np.asarray(y_magnitude, dtype=np.float32)
        self._weights = (
            np.asarray(sample_weight, dtype=np.float32)
            if sample_weight is not None
            else np.ones(n_samples, dtype=np.float32)
        )

        n_windows = n_samples - sequence_length + 1
        self._valid_indices: list[int] = []
        for i in range(n_windows):
            label_idx = i + sequence_length - 1
            if not nan_mask[label_idx]:
                self._valid_indices.append(i)

        if len(self._valid_indices) == 0:
            raise InsufficientDataError("No valid training samples after NaN filtering")

    def __len__(self) -> int:
        return len(self._valid_indices)

    def __getitem__(
        self, idx: int
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        i = self._valid_indices[idx]
        base_idx = i + self._seq_len - 1

        X_seq = torch.from_numpy(self._X[i : i + self._seq_len])
        y_dir = torch.tensor(self._y_dir[base_idx], dtype=torch.long)
        y_mag = torch.tensor(self._y_mag[base_idx], dtype=torch.float32)
        weight = torch.tensor(self._weights[base_idx], dtype=torch.float32)

        n_h = len(self._horizons)
        h_dirs = torch.ones(n_h, dtype=torch.long)
        h_mags = torch.zeros(n_h, dtype=torch.float32)
        h_masks = torch.zeros(n_h, dtype=torch.bool)

        for j, offset in enumerate(self._offsets):
            shifted_idx = base_idx + offset
            if 0 <= shifted_idx < len(self._y_dir):
                h_dirs[j] = int(self._y_dir[shifted_idx])
                h_mags[j] = float(self._y_mag[shifted_idx])
                h_masks[j] = True

        return X_seq, y_dir, y_mag, weight, h_dirs, h_mags, h_masks


def create_data_loaders(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    return_train: pd.Series,
    X_val: pd.DataFrame | None,
    y_val: pd.Series | None,
    return_val: pd.Series | None,
    sample_weight: pd.Series | None,
    sequence_length: int,
    batch_size: int,
) -> tuple[DataLoader, DataLoader | None]:  # type: ignore[type-arg]
    """Create train and optional val DataLoaders from tabular data."""
    train_ds = SequenceDataset(
        X=X_train.values,
        y_direction=y_train.values,
        y_magnitude=return_train.values,
        sequence_length=sequence_length,
        sample_weight=sample_weight.values if sample_weight is not None else None,
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    val_loader = None
    if X_val is not None and y_val is not None and return_val is not None:
        val_ds = SequenceDataset(
            X=X_val.values,
            y_direction=y_val.values,
            y_magnitude=return_val.values,
            sequence_length=sequence_length,
        )
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader


class EarlyStopping:
    """Tracks validation loss and signals when to stop training."""

    def __init__(self, patience: int, min_delta: float = 0.0) -> None:
        self._patience = patience
        self._min_delta = min_delta
        self._best_loss = float("inf")
        self._counter = 0

    def step(self, val_loss: float) -> bool:
        """Returns True if training should stop."""
        if val_loss < self._best_loss - self._min_delta:
            self._best_loss = val_loss
            self._counter = 0
            return False
        self._counter += 1
        return self._counter >= self._patience

    @property
    def best_loss(self) -> float:
        return self._best_loss


def compute_combined_loss(
    cls_logits: torch.Tensor,
    reg_output: torch.Tensor,
    y_dir: torch.Tensor,
    y_mag: torch.Tensor,
    weights: torch.Tensor,
    cls_weight: float = 0.7,
    reg_weight: float = 0.3,
) -> torch.Tensor:
    """Weighted combination of CrossEntropy (classification) and MSE (regression)."""
    ce_loss = nn.functional.cross_entropy(cls_logits, y_dir, reduction="none")
    mse_loss = nn.functional.mse_loss(reg_output, y_mag, reduction="none")

    weighted_ce = (ce_loss * weights).mean()
    weighted_mse = (mse_loss * weights).mean()

    return cls_weight * weighted_ce + reg_weight * weighted_mse


def get_device() -> torch.device:
    """Auto-detect best available device: CUDA > MPS > CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
