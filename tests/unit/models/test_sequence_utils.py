"""Tests for shared sequence utilities — datasets, early stopping, loss, device."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from alphavedha.exceptions import InsufficientDataError
from alphavedha.models.sequence_utils import (
    EarlyStopping,
    MultiHorizonSequenceDataset,
    SequenceDataset,
    compute_combined_loss,
    get_device,
)


@pytest.fixture
def seq_data() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """100 samples, 5 features, random labels and returns."""
    rng = np.random.default_rng(42)
    X = rng.standard_normal((100, 5)).astype(np.float32)
    y_dir = rng.choice([-1, 0, 1], size=100).astype(float)
    y_mag = rng.normal(0, 0.02, size=100).astype(np.float32)
    return X, y_dir, y_mag


class TestSequenceDataset:
    def test_dataset_length(self, seq_data: tuple[np.ndarray, np.ndarray, np.ndarray]) -> None:
        X, y_dir, y_mag = seq_data
        ds = SequenceDataset(X, y_dir, y_mag, sequence_length=60)
        assert len(ds) == 41  # 100 - 60 + 1

    def test_sequence_shape(self, seq_data: tuple[np.ndarray, np.ndarray, np.ndarray]) -> None:
        X, y_dir, y_mag = seq_data
        ds = SequenceDataset(X, y_dir, y_mag, sequence_length=10)
        X_seq, y_d, y_m, w = ds[0]
        assert X_seq.shape == (10, 5)
        assert y_d.shape == ()
        assert y_m.shape == ()
        assert w.shape == ()

    def test_label_alignment(self, seq_data: tuple[np.ndarray, np.ndarray, np.ndarray]) -> None:
        X, y_dir, y_mag = seq_data
        ds = SequenceDataset(X, y_dir, y_mag, sequence_length=10)
        _, _y_d, y_m, _ = ds[0]
        # Label should come from index 9 (last of window 0..9)
        expected_mag = np.float32(y_mag[9])
        assert float(y_m) == pytest.approx(expected_mag, abs=1e-6)

    def test_label_mapping(self, seq_data: tuple[np.ndarray, np.ndarray, np.ndarray]) -> None:
        X = np.zeros((20, 3), dtype=np.float32)
        y_dir = np.array([0, 1, -1] * 6 + [0, 1], dtype=float)
        y_mag = np.zeros(20, dtype=np.float32)
        ds = SequenceDataset(X, y_dir, y_mag, sequence_length=5)
        # Index 0 -> label at position 4 -> y_dir[4] = 1 -> mapped to 2
        _, y_d, _, _ = ds[0]
        assert int(y_d) == 2  # 1 maps to 2

    def test_nan_labels_excluded(self) -> None:
        X = np.zeros((20, 3), dtype=np.float32)
        y_dir = np.zeros(20, dtype=float)
        y_dir[9] = np.nan  # position 9 is label for window 0..9 (seq_len=10)
        y_dir[10] = np.nan  # position 10 is label for window 1..10
        y_mag = np.zeros(20, dtype=np.float32)
        ds = SequenceDataset(X, y_dir, y_mag, sequence_length=10)
        # 20 - 10 + 1 = 11 total windows, minus 2 NaN labels = 9
        assert len(ds) == 9

    def test_insufficient_data_raises(self) -> None:
        X = np.zeros((5, 3), dtype=np.float32)
        y_dir = np.zeros(5, dtype=float)
        y_mag = np.zeros(5, dtype=np.float32)
        with pytest.raises(InsufficientDataError):
            SequenceDataset(X, y_dir, y_mag, sequence_length=10)

    def test_sample_weights_default_ones(
        self, seq_data: tuple[np.ndarray, np.ndarray, np.ndarray]
    ) -> None:
        X, y_dir, y_mag = seq_data
        ds = SequenceDataset(X, y_dir, y_mag, sequence_length=10)
        _, _, _, w = ds[0]
        assert float(w) == pytest.approx(1.0)

    def test_sample_weights_custom(
        self, seq_data: tuple[np.ndarray, np.ndarray, np.ndarray]
    ) -> None:
        X, y_dir, y_mag = seq_data
        weights = np.full(100, 2.5, dtype=np.float32)
        ds = SequenceDataset(X, y_dir, y_mag, sequence_length=10, sample_weight=weights)
        _, _, _, w = ds[0]
        assert float(w) == pytest.approx(2.5)


class TestMultiHorizonSequenceDataset:
    def test_multi_horizon_length(
        self, seq_data: tuple[np.ndarray, np.ndarray, np.ndarray]
    ) -> None:
        X, y_dir, y_mag = seq_data
        ds = MultiHorizonSequenceDataset(X, y_dir, y_mag, sequence_length=10, horizons=[7, 15, 30])
        assert len(ds) == 91  # 100 - 10 + 1

    def test_multi_horizon_returns_masks(
        self, seq_data: tuple[np.ndarray, np.ndarray, np.ndarray]
    ) -> None:
        X, y_dir, y_mag = seq_data
        ds = MultiHorizonSequenceDataset(X, y_dir, y_mag, sequence_length=10, horizons=[7, 15, 30])
        _X_seq, _y_d, _y_m, _w, h_dirs, h_mags, h_masks = ds[0]
        assert h_dirs.shape == (3,)
        assert h_mags.shape == (3,)
        assert h_masks.shape == (3,)
        # 7d horizon (offset 0) should always be valid
        assert h_masks[0].item() is True

    def test_multi_horizon_masks_far_horizons(self) -> None:
        X = np.zeros((30, 3), dtype=np.float32)
        y_dir = np.zeros(30, dtype=float)
        y_mag = np.zeros(30, dtype=np.float32)
        ds = MultiHorizonSequenceDataset(X, y_dir, y_mag, sequence_length=10, horizons=[7, 15, 30])
        # Last window: i=20, label_idx=29
        # 7d offset=0 -> idx 29 -> valid (29 < 30)
        # 15d offset=8 -> idx 37 -> invalid (37 >= 30)
        # 30d offset=23 -> idx 52 -> invalid (52 >= 30)
        last_idx = len(ds) - 1
        _, _, _, _, _, _, h_masks = ds[last_idx]
        assert h_masks[0].item() is True  # 7d valid
        assert h_masks[1].item() is False  # 15d out of bounds
        assert h_masks[2].item() is False  # 30d out of bounds


class TestEarlyStopping:
    def test_triggers_after_patience(self) -> None:
        es = EarlyStopping(patience=3)
        es.step(1.0)  # improvement (from inf)
        es.step(1.1)  # no improvement, counter=1
        es.step(1.2)  # no improvement, counter=2
        assert es.step(1.3) is True  # counter=3 >= patience

    def test_resets_on_improvement(self) -> None:
        es = EarlyStopping(patience=3)
        es.step(1.0)
        es.step(1.1)  # counter=1
        es.step(0.5)  # improvement, counter reset
        es.step(0.6)  # counter=1
        assert es.step(0.7) is False  # counter=2 < patience=3

    def test_tracks_best_loss(self) -> None:
        es = EarlyStopping(patience=5)
        es.step(1.0)
        es.step(0.5)
        es.step(0.8)
        assert es.best_loss == pytest.approx(0.5)


class TestUtilities:
    def test_get_device_returns_valid(self) -> None:
        device = get_device()
        assert device.type in ("cuda", "mps", "cpu")

    def test_combined_loss_is_scalar(self) -> None:
        cls_logits = torch.randn(8, 3)
        reg_output = torch.randn(8)
        y_dir = torch.randint(0, 3, (8,))
        y_mag = torch.randn(8)
        weights = torch.ones(8)
        loss = compute_combined_loss(cls_logits, reg_output, y_dir, y_mag, weights)
        assert loss.shape == ()
        assert loss.item() > 0
