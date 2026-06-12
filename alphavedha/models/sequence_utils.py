"""Shared sequence utilities for PyTorch deep learning models."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import pandas as pd
import structlog
import torch
from torch import nn
from torch.utils.data import Dataset

from alphavedha.exceptions import InsufficientDataError

logger = structlog.get_logger(__name__)

_LABEL_MAP = {-1: 0, 0: 1, 1: 2}


class FeatureScaler:
    """Per-feature standardization fit on train data only — no leakage into val/test."""

    def __init__(self, mean: np.ndarray, scale: np.ndarray) -> None:
        self.mean = np.asarray(mean, dtype=np.float32)
        self.scale = np.asarray(scale, dtype=np.float32)

    @classmethod
    def fit(cls, X: np.ndarray) -> FeatureScaler:
        with np.errstate(invalid="ignore"):
            mean = np.nanmean(X, axis=0)
            scale = np.nanstd(X, axis=0)
        mean = np.nan_to_num(mean, nan=0.0)
        scale = np.where(np.isnan(scale) | (scale < 1e-8), 1.0, scale)
        return cls(mean, scale)

    def transform(self, X: np.ndarray) -> np.ndarray:
        return ((X - self.mean) / self.scale).astype(np.float32)

    def to_dict(self) -> dict[str, list[float]]:
        return {"mean": self.mean.tolist(), "scale": self.scale.tolist()}

    @classmethod
    def from_dict(cls, d: dict[str, list[float]]) -> FeatureScaler:
        return cls(np.asarray(d["mean"]), np.asarray(d["scale"]))


class FastSequenceLoader:
    """Pre-materialized sliding-window batches for single-horizon training.

    Replaces DataLoader+SequenceDataset in the training loop: windows are built
    once as a zero-copy unfold view, and each epoch slices whole batches by
    index instead of collating tens of thousands of per-sample __getitem__
    calls in Python.
    """

    def __init__(
        self,
        X: np.ndarray,
        y_direction: np.ndarray,
        y_magnitude: np.ndarray,
        sequence_length: int = 60,
        batch_size: int = 64,
        sample_weight: np.ndarray | None = None,
        shuffle: bool = False,
    ) -> None:
        n_samples = X.shape[0]
        if n_samples < sequence_length:
            raise InsufficientDataError(f"Need at least {sequence_length} samples, got {n_samples}")

        X_t = torch.from_numpy(np.ascontiguousarray(X, dtype=np.float32))
        # (n_windows, n_features, seq_len) — view, no copy
        self._windows = X_t.unfold(0, sequence_length, 1)
        self._seq_len = sequence_length
        self._batch_size = batch_size
        self._shuffle = shuffle

        y_dir_arr = np.asarray(y_direction, dtype=np.float64)
        nan_mask = np.isnan(y_dir_arr)
        mapped_dir = np.ones(len(y_dir_arr), dtype=np.int64)
        valid = ~nan_mask
        mapped_dir[valid] = np.clip(y_dir_arr[valid].astype(np.int64) + 1, 0, 2)
        self._y_direction = torch.from_numpy(mapped_dir)
        self._y_magnitude = torch.from_numpy(np.asarray(y_magnitude, dtype=np.float32))
        weights = (
            np.asarray(sample_weight, dtype=np.float32)
            if sample_weight is not None
            else np.ones(n_samples, dtype=np.float32)
        )
        self._weights = torch.from_numpy(weights)

        n_windows = n_samples - sequence_length + 1
        label_idx = np.arange(n_windows) + sequence_length - 1
        valid_starts = np.nonzero(~nan_mask[label_idx])[0]
        if len(valid_starts) == 0:
            raise InsufficientDataError("No valid training samples after NaN filtering")
        self._valid_starts = torch.from_numpy(valid_starts)

        logger.debug(
            "fast_sequence_loader_created",
            n_sequences=len(valid_starts),
            sequence_length=sequence_length,
            n_features=X.shape[1],
            batch_size=batch_size,
        )

    def __len__(self) -> int:
        return (len(self._valid_starts) + self._batch_size - 1) // self._batch_size

    def __iter__(
        self,
    ) -> Iterator[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
        order = self._valid_starts
        if self._shuffle:
            order = order[torch.randperm(len(order))]
        for i in range(0, len(order), self._batch_size):
            idx = order[i : i + self._batch_size]
            X_seq = self._windows[idx].transpose(1, 2).contiguous()
            label_idx = idx + self._seq_len - 1
            yield (
                X_seq,
                self._y_direction[label_idx],
                self._y_magnitude[label_idx],
                self._weights[label_idx],
            )


class FastMultiHorizonLoader:
    """Pre-materialized sliding-window batches with per-horizon labels for TFT.

    Multi-horizon counterpart of FastSequenceLoader: per-horizon label/mask
    arrays are vectorized once at construction, then each epoch slices whole
    batches by index instead of building 7 tensors per sample in Python.
    Yields the same 7-tuples as MultiHorizonSequenceDataset batches.
    """

    def __init__(
        self,
        X: np.ndarray,
        y_direction: np.ndarray,
        y_magnitude: np.ndarray,
        sequence_length: int,
        horizons: list[int],
        batch_size: int = 64,
        sample_weight: np.ndarray | None = None,
        shuffle: bool = False,
    ) -> None:
        n_samples = X.shape[0]
        if n_samples < sequence_length:
            raise InsufficientDataError(f"Need at least {sequence_length} samples, got {n_samples}")

        X_t = torch.from_numpy(np.ascontiguousarray(X, dtype=np.float32))
        # (n_windows, n_features, seq_len) — view, no copy
        self._windows = X_t.unfold(0, sequence_length, 1)
        self._seq_len = sequence_length
        self._batch_size = batch_size
        self._shuffle = shuffle

        sorted_h = sorted(horizons)
        base_horizon = sorted_h[0]
        offsets = np.array([h - base_horizon for h in sorted_h], dtype=np.int64)

        y_dir_arr = np.asarray(y_direction, dtype=np.float64)
        nan_mask = np.isnan(y_dir_arr)
        mapped_dir = np.ones(len(y_dir_arr), dtype=np.int64)
        valid = ~nan_mask
        mapped_dir[valid] = np.clip(y_dir_arr[valid].astype(np.int64) + 1, 0, 2)
        y_mag_arr = np.asarray(y_magnitude, dtype=np.float32)
        weights = (
            np.asarray(sample_weight, dtype=np.float32)
            if sample_weight is not None
            else np.ones(n_samples, dtype=np.float32)
        )

        n_windows = n_samples - sequence_length + 1
        window_label_idx = np.arange(n_windows) + sequence_length - 1
        valid_starts = np.nonzero(~nan_mask[window_label_idx])[0]
        if len(valid_starts) == 0:
            raise InsufficientDataError("No valid training samples after NaN filtering")
        self._valid_starts = torch.from_numpy(valid_starts)

        base_idx = valid_starts + sequence_length - 1
        self._y_dir = torch.from_numpy(mapped_dir[base_idx])
        self._y_mag = torch.from_numpy(y_mag_arr[base_idx])
        self._weights = torch.from_numpy(weights[base_idx])

        # Per-horizon labels: same bounds-only masking as MultiHorizonSequenceDataset
        shifted = base_idx[:, None] + offsets[None, :]  # (n_valid, n_h)
        in_range = (shifted >= 0) & (shifted < n_samples)
        clipped = np.clip(shifted, 0, n_samples - 1)
        self._h_dirs = torch.from_numpy(np.where(in_range, mapped_dir[clipped], 1))
        self._h_mags = torch.from_numpy(
            np.where(in_range, y_mag_arr[clipped], 0.0).astype(np.float32)
        )
        self._h_masks = torch.from_numpy(in_range)

        logger.debug(
            "fast_multihorizon_loader_created",
            n_sequences=len(valid_starts),
            sequence_length=sequence_length,
            n_features=X.shape[1],
            horizons=sorted_h,
            batch_size=batch_size,
        )

    def __len__(self) -> int:
        return (len(self._valid_starts) + self._batch_size - 1) // self._batch_size

    def __iter__(
        self,
    ) -> Iterator[
        tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
        ]
    ]:
        n = len(self._valid_starts)
        order = torch.randperm(n) if self._shuffle else torch.arange(n)
        for i in range(0, n, self._batch_size):
            pos = order[i : i + self._batch_size]
            X_seq = self._windows[self._valid_starts[pos]].transpose(1, 2).contiguous()
            yield (
                X_seq,
                self._y_dir[pos],
                self._y_mag[pos],
                self._weights[pos],
                self._h_dirs[pos],
                self._h_mags[pos],
                self._h_masks[pos],
            )


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
            raise InsufficientDataError(f"Need at least {sequence_length} samples, got {n_samples}")

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
            raise InsufficientDataError(f"Need at least {sequence_length} samples, got {n_samples}")

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
) -> tuple[FastSequenceLoader, FastSequenceLoader | None]:
    """Create train and optional val batch loaders from tabular data."""
    train_loader = FastSequenceLoader(
        X=X_train.values,
        y_direction=y_train.values,
        y_magnitude=return_train.values,
        sequence_length=sequence_length,
        batch_size=batch_size,
        sample_weight=sample_weight.values if sample_weight is not None else None,
        shuffle=True,
    )

    val_loader = None
    if X_val is not None and y_val is not None and return_val is not None:
        val_loader = FastSequenceLoader(
            X=X_val.values,
            y_direction=y_val.values,
            y_magnitude=return_val.values,
            sequence_length=sequence_length,
            batch_size=batch_size,
            shuffle=False,
        )

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
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    """Weighted combination of CrossEntropy (classification) and MSE (regression)."""
    ce_loss = nn.functional.cross_entropy(
        cls_logits, y_dir, reduction="none", label_smoothing=label_smoothing
    )
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
