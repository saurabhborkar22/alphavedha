"""Tests for training pipeline — temporal splits, feature selection, data prep."""

from __future__ import annotations

import numpy as np
import pandas as pd

from alphavedha.training.pipeline import (
    TierData,
    TrainingPipelineResult,
    _fill_nan_for_torch,
    _select_top_features,
    _temporal_split,
    _temporal_split_3way,
)

# Rotate a class label to a different one (-1 → 0 → 1 → -1): perturbs
# predictions off the truth while keeping the 3-class distribution balanced.
_ROTATE = {-1: 0, 0: 1, 1: -1}


def _make_dataset(n: int = 500) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    rng = np.random.default_rng(42)
    X = pd.DataFrame(rng.standard_normal((n, 10)), columns=[f"f{i}" for i in range(10)])
    y = pd.Series(rng.integers(0, 2, n))
    returns = pd.Series(rng.normal(0.001, 0.02, n))
    return X, y, returns


class TestTemporalSplit:
    def test_basic_split_sizes(self) -> None:
        X, y, ret = _make_dataset(500)
        X_tr, _y_tr, _ret_tr, X_v, _y_v, _ret_v = _temporal_split(X, y, ret, val_ratio=0.2)
        assert len(X_tr) + len(X_v) <= len(X)
        assert len(X_tr) > len(X_v)

    def test_no_overlap(self) -> None:
        X, y, ret = _make_dataset(500)
        X_tr, _, _, X_v, _, _ = _temporal_split(X, y, ret, val_ratio=0.2, embargo_days=20)
        assert X_tr.index[-1] < X_v.index[0]

    def test_embargo_gap(self) -> None:
        X, y, ret = _make_dataset(500)
        X_tr, _, _, X_v, _, _ = _temporal_split(X, y, ret, val_ratio=0.2, embargo_days=20)
        gap = X_v.index[0] - X_tr.index[-1]
        assert gap >= 20

    def test_small_dataset_still_works(self) -> None:
        X, y, ret = _make_dataset(50)
        X_tr, _y_tr, _ret_tr, X_v, _y_v, _ret_v = _temporal_split(X, y, ret, val_ratio=0.2)
        assert len(X_tr) > 0
        assert len(X_v) > 0

    def test_consistent_label_alignment(self) -> None:
        X, y, ret = _make_dataset(500)
        X_tr, y_tr, ret_tr, X_v, y_v, ret_v = _temporal_split(X, y, ret)
        assert len(X_tr) == len(y_tr) == len(ret_tr)
        assert len(X_v) == len(y_v) == len(ret_v)


class TestTemporalSplit3Way:
    def test_three_partitions(self) -> None:
        X, y, ret = _make_dataset(1000)
        result = _temporal_split_3way(X, y, ret, oof_ratio=0.15, val_ratio=0.15)
        X_tr, _y_tr, _ret_tr, X_oof, _y_oof, _ret_oof, X_val, _y_val, _ret_val = result
        assert len(X_tr) > 0
        assert len(X_oof) > 0
        assert len(X_val) > 0

    def test_temporal_ordering(self) -> None:
        X, y, ret = _make_dataset(1000)
        X_tr, _, _, X_oof, _, _, X_val, _, _ = _temporal_split_3way(X, y, ret)
        assert X_tr.index[-1] < X_oof.index[0]
        assert X_oof.index[-1] < X_val.index[0]

    def test_partition_sizes_reasonable(self) -> None:
        X, y, ret = _make_dataset(1000)
        X_tr, _, _, X_oof, _, _, X_val, _, _ = _temporal_split_3way(X, y, ret)
        assert len(X_tr) > len(X_oof)
        assert len(X_tr) > len(X_val)

    def test_alignment(self) -> None:
        X, y, ret = _make_dataset(1000)
        X_tr, y_tr, ret_tr, X_oof, y_oof, ret_oof, X_val, y_val, ret_val = _temporal_split_3way(
            X, y, ret
        )
        assert len(X_tr) == len(y_tr) == len(ret_tr)
        assert len(X_oof) == len(y_oof) == len(ret_oof)
        assert len(X_val) == len(y_val) == len(ret_val)


class TestSelectTopFeatures:
    def test_selects_top_n(self) -> None:
        importance = pd.Series({"f0": 10, "f1": 5, "f2": 8, "f3": 1, "f4": 3})
        result = _select_top_features(importance, 3, ["f0", "f1", "f2", "f3", "f4"])
        assert len(result) == 3
        assert result[0] == "f0"
        assert result[1] == "f2"
        assert result[2] == "f1"

    def test_filters_unavailable_columns(self) -> None:
        importance = pd.Series({"f0": 10, "f1": 5, "f_missing": 8})
        result = _select_top_features(importance, 3, ["f0", "f1"])
        assert "f_missing" not in result
        assert len(result) == 2

    def test_returns_fewer_if_not_enough(self) -> None:
        importance = pd.Series({"f0": 10, "f1": 5})
        result = _select_top_features(importance, 10, ["f0", "f1"])
        assert len(result) == 2


class TestFillNanForTorch:
    def test_replaces_nan(self) -> None:
        df = pd.DataFrame({"a": [1.0, np.nan, 3.0], "b": [np.nan, 2.0, np.nan]})
        result = _fill_nan_for_torch(df)
        assert not result.isna().any().any()

    def test_replaces_inf(self) -> None:
        df = pd.DataFrame({"a": [1.0, np.inf, -np.inf]})
        result = _fill_nan_for_torch(df)
        assert not np.isinf(result.values).any()

    def test_preserves_normal_values(self) -> None:
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
        result = _fill_nan_for_torch(df)
        pd.testing.assert_frame_equal(result, df)


class TestTrainingPipelineResult:
    def test_default_values(self) -> None:
        result = TrainingPipelineResult(model_name="test")
        assert result.model_name == "test"
        assert result.artifact_path is None
        assert result.n_symbols == 0
        assert result.errors == {}

    def test_extra_metrics(self) -> None:
        result = TrainingPipelineResult(
            model_name="xgboost",
            extra_metrics={"accuracy": 0.65, "f1": 0.60},
        )
        assert result.extra_metrics["accuracy"] == 0.65


class TestTierData:
    def test_empty_tier_data(self) -> None:
        data = TierData(
            X_train=pd.DataFrame(),
            y_train=pd.Series(dtype=float),
            ret_train=pd.Series(dtype=float),
            X_oof=pd.DataFrame(),
            y_oof=pd.Series(dtype=float),
            ret_oof=pd.Series(dtype=float),
            X_val=pd.DataFrame(),
            y_val=pd.Series(dtype=float),
            ret_val=pd.Series(dtype=float),
            ohlcv_by_symbol={},
            n_symbols=0,
            errors={},
        )
        assert data.X_train.empty
        assert data.n_symbols == 0


class TestDegeneracyGate:
    """The nightly retrain must never promote a collapsed direction model."""

    @staticmethod
    def _fit_model(labels: pd.Series, X: pd.DataFrame) -> object:
        from alphavedha.models.xgboost_model import XGBoostModel

        rng = np.random.default_rng(0)
        returns = pd.Series(rng.normal(0, 0.02, size=len(X)), name="return_pct")
        split = int(len(X) * 0.8)
        model = XGBoostModel()
        model.fit(
            X_train=X[:split],
            y_train=labels[:split],
            X_val=X[split:],
            y_val=labels[split:],
            return_train=returns[:split],
            return_val=returns[split:],
        )
        return model

    def test_rejects_single_class_model(self) -> None:
        from alphavedha.models.base import PredictionResult
        from alphavedha.training.pipeline import _check_degenerate_direction_model

        class AllShortModel:
            """Mimics the Jun-2026 production failure: -1 for every input."""

            def predict(self, X: pd.DataFrame) -> PredictionResult:
                n = len(X)
                return PredictionResult(
                    direction=np.full(n, -1, dtype=int),
                    magnitude=np.zeros(n),
                    probabilities=np.tile([0.45, 0.30, 0.25], (n, 1)),
                    confidence=np.full(n, 0.45),
                )

        rng = np.random.default_rng(3)
        X = pd.DataFrame(rng.standard_normal((100, 6)), columns=[f"f{i}" for i in range(6)])
        y = pd.Series(rng.choice([-1, 0, 1], size=100), name="label")

        reason = _check_degenerate_direction_model(AllShortModel(), X, y)  # type: ignore[arg-type]
        assert reason is not None
        assert "class -1" in reason

    def test_accepts_healthy_model(self) -> None:
        from alphavedha.training.pipeline import _check_degenerate_direction_model

        rng = np.random.default_rng(5)
        n = 600
        X = pd.DataFrame(rng.standard_normal((n, 6)), columns=[f"f{i}" for i in range(6)])
        # Separable: sign of f0 determines the class.
        labels = pd.Series(
            np.where(X["f0"] > 0.5, 1, np.where(X["f0"] < -0.5, -1, 0)), name="label"
        )
        model = self._fit_model(labels[:450], X[:450])

        reason = _check_degenerate_direction_model(model, X[450:], labels[450:])  # type: ignore[arg-type]
        assert reason is None

    def test_empty_validation_skips_gate(self) -> None:
        from alphavedha.training.pipeline import _check_degenerate_direction_model

        rng = np.random.default_rng(9)
        X = pd.DataFrame(rng.standard_normal((100, 4)), columns=[f"f{i}" for i in range(4)])
        labels = pd.Series([-1] * 100, name="label")
        model = self._fit_model(labels, X)

        empty_X = X.iloc[0:0]
        empty_y = labels.iloc[0:0]
        reason = _check_degenerate_direction_model(model, empty_X, empty_y)  # type: ignore[arg-type]
        assert reason is None

    # The array-based gate below is what train_all runs on the ENSEMBLE's
    # validation directions — the layer that actually decides served direction.

    def test_directions_reject_single_class(self) -> None:
        from alphavedha.training.pipeline import _check_degenerate_directions

        rng = np.random.default_rng(3)
        directions = np.full(100, -1, dtype=int)  # the Jul-2026 all-short ensemble
        y = pd.Series(rng.choice([-1, 0, 1], size=100), name="label")

        reason = _check_degenerate_directions(directions, y)
        assert reason is not None
        assert "class -1" in reason

    def test_directions_accept_mixed_and_accurate(self) -> None:
        from alphavedha.training.pipeline import _check_degenerate_directions

        y = pd.Series(([-1, 0, 1] * 34)[:99], name="label")  # balanced 3-class
        directions = y.to_numpy().copy()
        for i in range(25):  # ~75% accuracy, still class-balanced
            directions[i] = _ROTATE[int(directions[i])]

        assert _check_degenerate_directions(directions, y) is None

    def test_directions_reject_below_accuracy_floor(self) -> None:
        from alphavedha.training.pipeline import _check_degenerate_directions

        y = pd.Series(([-1, 0, 1] * 34)[:99], name="label")
        # Balanced (not collapsed) but every prediction is rotated off → 0% right.
        directions = np.array([_ROTATE[int(v)] for v in y], dtype=int)

        reason = _check_degenerate_directions(directions, y)
        assert reason is not None
        assert "accuracy" in reason

    def test_directions_empty_skips_gate(self) -> None:
        from alphavedha.training.pipeline import _check_degenerate_directions

        reason = _check_degenerate_directions(np.array([], dtype=int), pd.Series([], dtype=int))
        assert reason is None


class TestDropExcludedFeatures:
    def test_drops_raw_date_features(self) -> None:
        from alphavedha.training.pipeline import _drop_stub_features

        X = pd.DataFrame(
            {
                "ret_1d": [0.01, -0.02],
                "cal_year": [2026, 2026],
                "cal_doy": [187, 188],
                "cal_week_of_year": [28, 28],
                "cal_month": [7, 7],
            }
        )
        out = _drop_stub_features(X)
        assert "cal_year" not in out.columns
        assert "cal_doy" not in out.columns
        assert "cal_week_of_year" not in out.columns
        # Cyclical seasonality features survive — only raw positions go.
        assert list(out.columns) == ["ret_1d", "cal_month"]

    def test_drops_stub_and_date_features_together(self) -> None:
        from alphavedha.training.pipeline import _drop_stub_features

        X = pd.DataFrame(
            {
                "macro_gsec_10y": [7.0, 7.0],
                "cal_year": [2026, 2026],
                "mom_rsi_14": [55.0, 60.0],
            }
        )
        out = _drop_stub_features(X)
        assert list(out.columns) == ["mom_rsi_14"]

    def test_noop_when_nothing_excluded(self) -> None:
        from alphavedha.training.pipeline import _drop_stub_features

        X = pd.DataFrame({"ret_1d": [0.01], "mom_rsi_14": [55.0]})
        out = _drop_stub_features(X)
        assert list(out.columns) == ["ret_1d", "mom_rsi_14"]
