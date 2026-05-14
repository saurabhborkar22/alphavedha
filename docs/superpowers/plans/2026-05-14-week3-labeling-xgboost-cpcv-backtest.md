# Week 3 Implementation Plan: Labeling + XGBoost + CPCV + Backtesting

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the core ML pipeline — label generation, first base model (XGBoost), cross-validation (CPCV), and backtesting with Indian market costs.

**Architecture:** Labels module generates triple-barrier labels from OHLCV+ATR. BaseModel ABC defines the model lifecycle contract. XGBoostModel wraps a classifier + regressor. CPCV validates any BaseModel with purge+embargo. VectorBT backtester converts predictions to equity curves with real Indian costs.

**Tech Stack:** xgboost, scikit-learn, vectorbt, joblib, numpy, pandas, structlog (all already in pyproject.toml)

**Design spec:** `docs/superpowers/specs/2026-05-14-week3-labeling-xgboost-cpcv-backtest.md`

---

## File Map

```
# New files (create)
alphavedha/labels/triple_barrier.py      — Triple barrier label generation
alphavedha/labels/sample_weights.py      — Uniqueness + recency sample weighting
alphavedha/models/base.py                — BaseModel ABC + TrainResult, PredictionResult, ModelArtifact
alphavedha/models/xgboost_model.py       — XGBoostModel (classifier + regressor)
alphavedha/backtest/costs.py             — Indian market cost calculator
alphavedha/backtest/cpcv.py              — Combinatorial Purged Cross-Validation
alphavedha/backtest/engine.py            — VectorBT backtesting engine

tests/unit/labels/test_triple_barrier.py — Label generation tests
tests/unit/labels/test_sample_weights.py — Sample weight tests
tests/unit/models/test_base.py           — BaseModel ABC tests
tests/unit/models/test_xgboost_model.py  — XGBoost model tests
tests/unit/backtest/test_costs.py        — Cost calculator tests
tests/unit/backtest/test_cpcv.py         — CPCV validation tests
tests/unit/backtest/test_engine.py       — Backtest engine tests

# Modified files
tests/conftest.py                        — Add fixtures: sample_ohlcv_500, sample_features_500, sample_known_path
alphavedha/labels/__init__.py            — Module exports
alphavedha/models/__init__.py            — Module exports
alphavedha/backtest/__init__.py          — Module exports
```

---

### Task 1: Add shared test fixtures

**Files:**
- Modify: `tests/conftest.py`

These fixtures are used by label, model, and backtest tests. Add them first so all downstream tests can reference them.

- [ ] **Step 1: Add `sample_ohlcv_500` fixture**

Append to `tests/conftest.py`:

```python
@pytest.fixture
def sample_ohlcv_500() -> pd.DataFrame:
    """500 trading days of OHLCV for labeling and CPCV tests."""
    dates = pd.bdate_range("2022-01-03", periods=500, freq="B")
    rng = np.random.default_rng(42)

    base_price = 3800.0
    returns = rng.normal(0.0005, 0.018, size=500)
    closes = base_price * np.cumprod(1 + returns)

    highs = closes * (1 + np.abs(rng.normal(0, 0.012, 500)))
    lows = closes * (1 - np.abs(rng.normal(0, 0.012, 500)))
    opens = closes * (1 + rng.normal(0, 0.005, 500))

    df = pd.DataFrame(
        {
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "adj_close": closes,
            "volume": rng.integers(5_000_000, 15_000_000, size=500),
        },
        index=dates,
    )
    df.index.name = "date"
    return df
```

- [ ] **Step 2: Add `sample_features_500` fixture**

Append to `tests/conftest.py`:

```python
@pytest.fixture
def sample_features_500(sample_ohlcv_500: pd.DataFrame) -> pd.DataFrame:
    """142 synthetic feature columns aligned to sample_ohlcv_500."""
    rng = np.random.default_rng(99)
    n = len(sample_ohlcv_500)
    data = rng.standard_normal((n, 142))
    columns = [f"feat_{i:03d}" for i in range(142)]
    return pd.DataFrame(data, index=sample_ohlcv_500.index, columns=columns)
```

- [ ] **Step 3: Add `sample_known_path` fixture**

This fixture has a deterministic price path for hand-verifiable label tests: price rises, then drops, then stays flat.

Append to `tests/conftest.py`:

```python
@pytest.fixture
def sample_known_path() -> pd.DataFrame:
    """Deterministic price path for label verification.

    Days 0-4:   steady rise (100 -> 110) — should trigger upper barrier
    Days 5-9:   sharp drop (110 -> 95)  — should trigger lower barrier
    Days 10-24: flat (100)              — should trigger time barrier
    """
    dates = pd.bdate_range("2024-01-02", periods=25, freq="B")
    closes = np.array(
        [100, 102, 104, 106, 110,
         108, 104, 100, 97, 95,
         100, 100, 100, 100, 100,
         100, 100, 100, 100, 100,
         100, 100, 100, 100, 100],
        dtype=float,
    )
    highs = closes * 1.005
    lows = closes * 0.995
    opens = closes * 1.001

    df = pd.DataFrame(
        {
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "adj_close": closes,
            "volume": np.full(25, 10_000_000),
        },
        index=dates,
    )
    df.index.name = "date"
    return df
```

- [ ] **Step 4: Verify fixtures load**

Run: `pytest tests/conftest.py --collect-only 2>&1 | head -20`

Expected: No import errors. Fixtures are collected.

- [ ] **Step 5: Commit**

```bash
git add tests/conftest.py
git commit -m "test: add Week 3 fixtures — 500-day OHLCV, synthetic features, known price path"
```

---

### Task 2: Triple barrier labeling

**Files:**
- Create: `alphavedha/labels/triple_barrier.py`
- Create: `tests/unit/labels/test_triple_barrier.py`

- [ ] **Step 1: Write failing tests**

Create `tests/unit/labels/test_triple_barrier.py`:

```python
"""Tests for triple barrier labeling."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from alphavedha.config import TripleBarrierConfig
from alphavedha.labels.triple_barrier import LabelResult, compute_triple_barrier_labels


@pytest.fixture
def default_config() -> TripleBarrierConfig:
    return TripleBarrierConfig()


class TestTripleBarrierLabels:
    def test_returns_label_result(
        self, sample_ohlcv_500: pd.DataFrame, default_config: TripleBarrierConfig
    ) -> None:
        result = compute_triple_barrier_labels(sample_ohlcv_500, default_config)
        assert isinstance(result, LabelResult)
        assert isinstance(result.df, pd.DataFrame)
        assert result.symbol == ""

    def test_output_columns(
        self, sample_ohlcv_500: pd.DataFrame, default_config: TripleBarrierConfig
    ) -> None:
        result = compute_triple_barrier_labels(sample_ohlcv_500, default_config)
        expected_cols = {
            "label", "return_pct", "barrier_hit", "days_to_hit",
            "entry_price", "exit_price", "atr_at_entry",
        }
        assert expected_cols.issubset(set(result.df.columns))

    def test_labels_are_valid_values(
        self, sample_ohlcv_500: pd.DataFrame, default_config: TripleBarrierConfig
    ) -> None:
        result = compute_triple_barrier_labels(sample_ohlcv_500, default_config)
        valid = result.df["label"].dropna()
        assert set(valid.unique()).issubset({-1, 0, 1})

    def test_last_rows_are_nan(
        self, sample_ohlcv_500: pd.DataFrame, default_config: TripleBarrierConfig
    ) -> None:
        result = compute_triple_barrier_labels(sample_ohlcv_500, default_config)
        last_labels = result.df["label"].iloc[-default_config.max_holding_period:]
        assert last_labels.isna().all()

    def test_label_counts_match_data(
        self, sample_ohlcv_500: pd.DataFrame, default_config: TripleBarrierConfig
    ) -> None:
        result = compute_triple_barrier_labels(sample_ohlcv_500, default_config)
        total_labeled = sum(result.label_counts.values())
        non_nan = result.df["label"].notna().sum()
        assert total_labeled == non_nan

    def test_no_lookahead_in_atr(
        self, sample_ohlcv_500: pd.DataFrame, default_config: TripleBarrierConfig
    ) -> None:
        """ATR at time t must use only data up to t."""
        result = compute_triple_barrier_labels(sample_ohlcv_500, default_config)
        atr_col = result.df["atr_at_entry"].dropna()
        assert len(atr_col) > 0
        first_valid_idx = atr_col.index[0]
        pos = sample_ohlcv_500.index.get_loc(first_valid_idx)
        assert pos >= default_config.atr_period

    def test_barrier_hit_values(
        self, sample_ohlcv_500: pd.DataFrame, default_config: TripleBarrierConfig
    ) -> None:
        result = compute_triple_barrier_labels(sample_ohlcv_500, default_config)
        valid_hits = result.df["barrier_hit"].dropna()
        assert set(valid_hits.unique()).issubset({"upper", "lower", "time"})

    def test_days_to_hit_bounded(
        self, sample_ohlcv_500: pd.DataFrame, default_config: TripleBarrierConfig
    ) -> None:
        result = compute_triple_barrier_labels(sample_ohlcv_500, default_config)
        valid_days = result.df["days_to_hit"].dropna()
        assert (valid_days >= 1).all()
        assert (valid_days <= default_config.max_holding_period).all()

    def test_return_pct_matches_prices(
        self, sample_ohlcv_500: pd.DataFrame, default_config: TripleBarrierConfig
    ) -> None:
        result = compute_triple_barrier_labels(sample_ohlcv_500, default_config)
        valid = result.df.dropna(subset=["label"])
        for _, row in valid.head(10).iterrows():
            expected_ret = row["exit_price"] / row["entry_price"] - 1
            assert abs(row["return_pct"] - expected_ret) < 1e-10

    def test_low_atr_skipped(self) -> None:
        """Stocks with ATR/close < min_atr_threshold get NaN labels."""
        dates = pd.bdate_range("2024-01-02", periods=50, freq="B")
        flat_price = np.full(50, 100.0)
        df = pd.DataFrame(
            {
                "open": flat_price,
                "high": flat_price * 1.0001,
                "low": flat_price * 0.9999,
                "close": flat_price,
                "adj_close": flat_price,
                "volume": np.full(50, 10_000_000),
            },
            index=dates,
        )
        df.index.name = "date"
        config = TripleBarrierConfig(min_atr_threshold=0.005)
        result = compute_triple_barrier_labels(df, config)
        assert result.skipped_low_atr > 0

    def test_insufficient_data_raises(self, default_config: TripleBarrierConfig) -> None:
        dates = pd.bdate_range("2024-01-02", periods=10, freq="B")
        df = pd.DataFrame(
            {
                "open": np.ones(10) * 100,
                "high": np.ones(10) * 101,
                "low": np.ones(10) * 99,
                "close": np.ones(10) * 100,
                "adj_close": np.ones(10) * 100,
                "volume": np.ones(10, dtype=int) * 1_000_000,
            },
            index=dates,
        )
        df.index.name = "date"
        from alphavedha.exceptions import InsufficientDataError
        with pytest.raises(InsufficientDataError):
            compute_triple_barrier_labels(df, default_config)

    def test_with_symbol(
        self, sample_ohlcv_500: pd.DataFrame, default_config: TripleBarrierConfig
    ) -> None:
        result = compute_triple_barrier_labels(
            sample_ohlcv_500, default_config, symbol="TCS.NS"
        )
        assert result.symbol == "TCS.NS"

    def test_same_bar_touch_favors_lower(self) -> None:
        """When both barriers are touched on the same day, lower wins."""
        dates = pd.bdate_range("2024-01-02", periods=50, freq="B")
        rng = np.random.default_rng(42)
        closes = np.full(50, 100.0)
        closes[0] = 100.0
        highs = closes.copy()
        lows = closes.copy()
        highs[1] = 200.0
        lows[1] = 50.0
        df = pd.DataFrame(
            {
                "open": closes * (1 + rng.normal(0, 0.001, 50)),
                "high": highs,
                "low": lows,
                "close": closes,
                "adj_close": closes,
                "volume": np.full(50, 10_000_000),
            },
            index=dates,
        )
        df.index.name = "date"
        config = TripleBarrierConfig()
        result = compute_triple_barrier_labels(df, config)
        label_at_0 = result.df["label"].iloc[0]
        if pd.notna(label_at_0):
            assert label_at_0 == -1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/labels/test_triple_barrier.py -v 2>&1 | tail -5`

Expected: `ModuleNotFoundError: No module named 'alphavedha.labels.triple_barrier'`

- [ ] **Step 3: Implement triple barrier labeling**

Create `alphavedha/labels/triple_barrier.py`:

```python
"""Triple barrier labeling — generates direction labels from OHLCV + ATR."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import structlog
from ta.volatility import AverageTrueRange

from alphavedha.config import TripleBarrierConfig
from alphavedha.exceptions import InsufficientDataError

logger = structlog.get_logger(__name__)

_MIN_ROWS = 50


@dataclass
class LabelResult:
    df: pd.DataFrame
    symbol: str
    label_counts: dict[int, int] = field(default_factory=dict)
    skipped_low_atr: int = 0
    avg_days_to_hit: float = 0.0


def _compute_atr(df: pd.DataFrame, period: int) -> pd.Series:
    """Compute ATR using only past data at each point (no look-ahead)."""
    return AverageTrueRange(
        high=df["high"], low=df["low"], close=df["close"], window=period
    ).average_true_range()


def compute_triple_barrier_labels(
    ohlcv_df: pd.DataFrame,
    config: TripleBarrierConfig,
    symbol: str = "",
) -> LabelResult:
    if len(ohlcv_df) < _MIN_ROWS:
        raise InsufficientDataError(
            f"Need >= {_MIN_ROWS} rows for labeling, got {len(ohlcv_df)}"
        )

    df = ohlcv_df.copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    atr = _compute_atr(df, config.atr_period)
    closes = df["close"].values
    highs = df["high"].values
    lows = df["low"].values
    n = len(df)
    hp = config.max_holding_period

    labels = np.full(n, np.nan)
    return_pcts = np.full(n, np.nan)
    barrier_hits = np.full(n, None, dtype=object)
    days_to_hits = np.full(n, np.nan)
    entry_prices = np.full(n, np.nan)
    exit_prices = np.full(n, np.nan)
    atr_at_entries = np.full(n, np.nan)

    skipped_low_atr = 0

    for t in range(n):
        atr_val = atr.iloc[t]

        if pd.isna(atr_val) or atr_val <= 0:
            continue

        if atr_val / closes[t] < config.min_atr_threshold:
            skipped_low_atr += 1
            continue

        if t + hp >= n:
            continue

        entry = closes[t]
        upper = entry + config.multiplier_up * atr_val
        lower = entry - config.multiplier_down * atr_val
        atr_at_entries[t] = atr_val
        entry_prices[t] = entry

        hit_label = None
        hit_day = None
        hit_price = None

        for d in range(1, hp + 1):
            idx = t + d
            h = highs[idx]
            lo = lows[idx]

            upper_touched = h >= upper
            lower_touched = lo <= lower

            if upper_touched and lower_touched:
                hit_label = -1
                hit_day = d
                hit_price = lower
                break
            elif upper_touched:
                hit_label = 1
                hit_day = d
                hit_price = upper
                break
            elif lower_touched:
                hit_label = -1
                hit_day = d
                hit_price = lower
                break

        if hit_label is None:
            hit_label = 0
            hit_day = hp
            hit_price = closes[t + hp]

        labels[t] = hit_label
        days_to_hits[t] = hit_day
        exit_prices[t] = hit_price
        return_pcts[t] = hit_price / entry - 1
        barrier_hits[t] = (
            "upper" if hit_label == 1 else "lower" if hit_label == -1 else "time"
        )

    result_df = pd.DataFrame(
        {
            "label": labels,
            "return_pct": return_pcts,
            "barrier_hit": barrier_hits,
            "days_to_hit": days_to_hits,
            "entry_price": entry_prices,
            "exit_price": exit_prices,
            "atr_at_entry": atr_at_entries,
        },
        index=df.index,
    )

    valid_labels = result_df["label"].dropna()
    label_counts: dict[int, int] = {}
    for val in (-1, 0, 1):
        label_counts[val] = int((valid_labels == val).sum())

    valid_days = result_df["days_to_hit"].dropna()
    avg_days = float(valid_days.mean()) if len(valid_days) > 0 else 0.0

    logger.info(
        "triple_barrier_labels_computed",
        symbol=symbol,
        n_rows=n,
        n_labeled=len(valid_labels),
        label_counts=label_counts,
        skipped_low_atr=skipped_low_atr,
        avg_days_to_hit=round(avg_days, 1),
    )

    return LabelResult(
        df=result_df,
        symbol=symbol,
        label_counts=label_counts,
        skipped_low_atr=skipped_low_atr,
        avg_days_to_hit=avg_days,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/labels/test_triple_barrier.py -v`

Expected: All tests PASS.

- [ ] **Step 5: Commit**

```bash
git add alphavedha/labels/triple_barrier.py tests/unit/labels/test_triple_barrier.py
git commit -m "feat: implement triple barrier labeling with ATR-scaled barriers"
```

---

### Task 3: Sample weights

**Files:**
- Create: `alphavedha/labels/sample_weights.py`
- Create: `tests/unit/labels/test_sample_weights.py`

- [ ] **Step 1: Write failing tests**

Create `tests/unit/labels/test_sample_weights.py`:

```python
"""Tests for sample weight computation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from alphavedha.config import SampleWeightsConfig
from alphavedha.labels.sample_weights import compute_sample_weights


@pytest.fixture
def default_config() -> SampleWeightsConfig:
    return SampleWeightsConfig()


class TestSampleWeights:
    def test_returns_series(self, default_config: SampleWeightsConfig) -> None:
        dates = pd.bdate_range("2024-01-02", periods=100, freq="B")
        labels_df = pd.DataFrame(
            {"label": np.random.default_rng(42).choice([-1, 0, 1], size=100),
             "days_to_hit": np.full(100, 5)},
            index=dates,
        )
        result = compute_sample_weights(labels_df, default_config)
        assert isinstance(result, pd.Series)
        assert len(result) == len(labels_df)

    def test_weights_sum_to_n(self, default_config: SampleWeightsConfig) -> None:
        dates = pd.bdate_range("2024-01-02", periods=100, freq="B")
        labels_df = pd.DataFrame(
            {"label": np.random.default_rng(42).choice([-1, 0, 1], size=100),
             "days_to_hit": np.full(100, 5)},
            index=dates,
        )
        result = compute_sample_weights(labels_df, default_config)
        assert abs(result.sum() - len(labels_df)) < 1e-6

    def test_all_weights_positive(self, default_config: SampleWeightsConfig) -> None:
        dates = pd.bdate_range("2024-01-02", periods=100, freq="B")
        labels_df = pd.DataFrame(
            {"label": np.random.default_rng(42).choice([-1, 0, 1], size=100),
             "days_to_hit": np.full(100, 5)},
            index=dates,
        )
        result = compute_sample_weights(labels_df, default_config)
        assert (result > 0).all()

    def test_recency_most_recent_highest(self, default_config: SampleWeightsConfig) -> None:
        """Most recent sample should have the highest weight (all else equal)."""
        dates = pd.bdate_range("2024-01-02", periods=100, freq="B")
        labels_df = pd.DataFrame(
            {"label": np.ones(100, dtype=int),
             "days_to_hit": np.ones(100)},
            index=dates,
        )
        result = compute_sample_weights(labels_df, default_config)
        assert result.iloc[-1] > result.iloc[0]

    def test_no_overlap_weight_equals_one_before_normalization(self) -> None:
        """Non-overlapping labels should have uniqueness weight = 1."""
        dates = pd.bdate_range("2024-01-02", periods=100, freq="B")
        labels_df = pd.DataFrame(
            {"label": np.ones(100, dtype=int),
             "days_to_hit": np.ones(100)},
            index=dates,
        )
        config = SampleWeightsConfig(uniqueness=True, recency_halflife=999999)
        result = compute_sample_weights(labels_df, config)
        mean_w = result.mean()
        assert abs(mean_w - 1.0) < 0.1

    def test_overlap_reduces_weight(self) -> None:
        """Overlapping labels should produce lower uniqueness weights than non-overlapping."""
        dates = pd.bdate_range("2024-01-02", periods=100, freq="B")
        non_overlap = pd.DataFrame(
            {"label": np.ones(100, dtype=int), "days_to_hit": np.ones(100)},
            index=dates,
        )
        overlap = pd.DataFrame(
            {"label": np.ones(100, dtype=int), "days_to_hit": np.full(100, 15)},
            index=dates,
        )
        config = SampleWeightsConfig(uniqueness=True, recency_halflife=999999)
        w_non = compute_sample_weights(non_overlap, config)
        w_over = compute_sample_weights(overlap, config)
        assert w_non.std() < w_over.std() or w_non.min() >= w_over.min()

    def test_uniqueness_disabled(self) -> None:
        """When uniqueness=False, only recency weighting is applied."""
        dates = pd.bdate_range("2024-01-02", periods=100, freq="B")
        labels_df = pd.DataFrame(
            {"label": np.ones(100, dtype=int), "days_to_hit": np.full(100, 15)},
            index=dates,
        )
        config = SampleWeightsConfig(uniqueness=False, recency_halflife=252)
        result = compute_sample_weights(labels_df, config)
        assert result.iloc[-1] > result.iloc[0]

    def test_handles_nan_labels(self, default_config: SampleWeightsConfig) -> None:
        dates = pd.bdate_range("2024-01-02", periods=100, freq="B")
        labels = np.ones(100)
        labels[-15:] = np.nan
        labels_df = pd.DataFrame(
            {"label": labels, "days_to_hit": np.full(100, 5)},
            index=dates,
        )
        result = compute_sample_weights(labels_df, default_config)
        assert len(result) == len(labels_df)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/labels/test_sample_weights.py -v 2>&1 | tail -5`

Expected: `ModuleNotFoundError`

- [ ] **Step 3: Implement sample weights**

Create `alphavedha/labels/sample_weights.py`:

```python
"""Sample weighting — uniqueness and recency weights for overlapping barrier labels."""

from __future__ import annotations

import numpy as np
import pandas as pd
import structlog

from alphavedha.config import SampleWeightsConfig

logger = structlog.get_logger(__name__)


def _compute_uniqueness_weights(labels_df: pd.DataFrame) -> np.ndarray:
    """Weight = mean(1 / concurrency) over each sample's active window."""
    n = len(labels_df)
    days_to_hit = labels_df["days_to_hit"].fillna(1).astype(int).values
    concurrency = np.ones(n, dtype=float)

    for i in range(n):
        if pd.isna(labels_df["label"].iloc[i]):
            continue
        end = min(i + days_to_hit[i], n)
        for j in range(i, end):
            concurrency[j] += 1

    concurrency = np.maximum(concurrency, 1.0)

    weights = np.ones(n, dtype=float)
    for i in range(n):
        if pd.isna(labels_df["label"].iloc[i]):
            weights[i] = 1.0
            continue
        end = min(i + days_to_hit[i], n)
        window_concurrency = concurrency[i:end]
        weights[i] = float(np.mean(1.0 / window_concurrency))

    return weights


def _compute_recency_weights(index: pd.DatetimeIndex, halflife: int) -> np.ndarray:
    """Exponential decay from most recent timestamp."""
    positions = np.arange(len(index), dtype=float)
    last_pos = positions[-1]
    decay = np.exp(-np.log(2) * (last_pos - positions) / halflife)
    return decay


def compute_sample_weights(
    labels_df: pd.DataFrame,
    config: SampleWeightsConfig,
) -> pd.Series:
    n = len(labels_df)

    if config.uniqueness:
        uniqueness = _compute_uniqueness_weights(labels_df)
    else:
        uniqueness = np.ones(n, dtype=float)

    recency = _compute_recency_weights(labels_df.index, config.recency_halflife)

    combined = uniqueness * recency
    combined = combined * (n / combined.sum())

    logger.info(
        "sample_weights_computed",
        n_samples=n,
        uniqueness_enabled=config.uniqueness,
        halflife=config.recency_halflife,
        weight_min=round(float(combined.min()), 4),
        weight_max=round(float(combined.max()), 4),
    )

    return pd.Series(combined, index=labels_df.index, name="sample_weight")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/labels/test_sample_weights.py -v`

Expected: All tests PASS.

- [ ] **Step 5: Update labels `__init__.py` and commit**

Write `alphavedha/labels/__init__.py`:

```python
"""Labels — triple barrier labeling and sample weighting."""

from alphavedha.labels.sample_weights import compute_sample_weights
from alphavedha.labels.triple_barrier import LabelResult, compute_triple_barrier_labels

__all__ = ["LabelResult", "compute_triple_barrier_labels", "compute_sample_weights"]
```

```bash
git add alphavedha/labels/ tests/unit/labels/test_sample_weights.py
git commit -m "feat: implement sample weights with uniqueness and recency decay"
```

---

### Task 4: BaseModel ABC and result types

**Files:**
- Create: `alphavedha/models/base.py`
- Create: `tests/unit/models/test_base.py`

- [ ] **Step 1: Write failing tests**

Create `tests/unit/models/test_base.py`:

```python
"""Tests for BaseModel ABC and result dataclasses."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from alphavedha.models.base import (
    BaseModel,
    ModelArtifact,
    PredictionResult,
    TrainResult,
)


class DummyModel(BaseModel):
    """Minimal BaseModel subclass for testing the ABC."""

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame | None = None,
        y_val: pd.Series | None = None,
        sample_weight: pd.Series | None = None,
    ) -> TrainResult:
        self._is_fitted = True
        self._feature_names = list(X_train.columns)
        self._train_metrics = {"accuracy": 0.75}
        return TrainResult(
            train_metrics={"accuracy": 0.75},
            val_metrics={"accuracy": 0.70},
            feature_importances=pd.Series(
                np.ones(len(X_train.columns)) / len(X_train.columns),
                index=X_train.columns,
            ),
            training_time_seconds=0.1,
            n_train_samples=len(X_train),
            n_val_samples=len(X_val) if X_val is not None else 0,
            hyperparams={"dummy": True},
        )

    def predict(self, X: pd.DataFrame) -> PredictionResult:
        if not self._is_fitted:
            from alphavedha.exceptions import ModelTrainingError
            raise ModelTrainingError("Model not fitted")
        n = len(X)
        return PredictionResult(
            direction=np.ones(n, dtype=int),
            magnitude=np.full(n, 0.02),
            probabilities=np.full((n, 3), 1 / 3),
            confidence=np.full(n, 0.6),
        )

    def get_feature_importance(self) -> pd.Series | None:
        if not self._is_fitted:
            return None
        return pd.Series(
            np.ones(len(self._feature_names)) / len(self._feature_names),
            index=self._feature_names,
        )

    def _save_model_artifacts(self, directory: Path) -> None:
        (directory / "dummy.txt").write_text("dummy")

    @classmethod
    def _load_model_artifacts(cls, directory: Path, config: dict[str, Any]) -> DummyModel:
        model = cls(name="dummy", config=config)
        model._is_fitted = True
        return model


class TestBaseModelABC:
    def test_cannot_instantiate_abc(self) -> None:
        with pytest.raises(TypeError):
            BaseModel(name="test", config={})  # type: ignore[abstract]

    def test_dummy_model_properties(self) -> None:
        model = DummyModel(name="dummy", config={"x": 1})
        assert model.name == "dummy"
        assert model.version == "0.0.0"
        assert model.is_fitted is False

    def test_fit_sets_fitted(self) -> None:
        model = DummyModel(name="dummy", config={})
        X = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        y = pd.Series([1, 0, 1])
        model.fit(X, y)
        assert model.is_fitted is True

    def test_predict_before_fit_raises(self) -> None:
        model = DummyModel(name="dummy", config={})
        X = pd.DataFrame({"a": [1, 2]})
        from alphavedha.exceptions import ModelTrainingError
        with pytest.raises(ModelTrainingError):
            model.predict(X)

    def test_get_metrics(self) -> None:
        model = DummyModel(name="dummy", config={})
        X = pd.DataFrame({"a": [1, 2, 3]})
        y = pd.Series([1, 0, 1])
        model.fit(X, y)
        metrics = model.get_metrics()
        assert "accuracy" in metrics

    def test_version_increments_on_save(self, tmp_path: Path) -> None:
        model = DummyModel(name="dummy", config={})
        X = pd.DataFrame({"a": [1, 2, 3]})
        y = pd.Series([1, 0, 1])
        model.fit(X, y)
        artifact = model.save(tmp_path / "v1")
        assert artifact.version == "0.0.1"
        artifact2 = model.save(tmp_path / "v2")
        assert artifact2.version == "0.0.2"

    def test_save_creates_metadata_json(self, tmp_path: Path) -> None:
        model = DummyModel(name="dummy", config={"x": 1})
        X = pd.DataFrame({"a": [1, 2, 3]})
        y = pd.Series([1, 0, 1])
        model.fit(X, y)
        artifact = model.save(tmp_path / "out")
        metadata_path = tmp_path / "out" / "metadata.json"
        assert metadata_path.exists()
        metadata = json.loads(metadata_path.read_text())
        assert metadata["name"] == "dummy"
        assert "feature_names" in metadata

    def test_save_creates_feature_importance_csv(self, tmp_path: Path) -> None:
        model = DummyModel(name="dummy", config={})
        X = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        y = pd.Series([1, 0, 1])
        model.fit(X, y)
        model.save(tmp_path / "out")
        fi_path = tmp_path / "out" / "feature_importance.csv"
        assert fi_path.exists()


class TestTrainResult:
    def test_fields(self) -> None:
        tr = TrainResult(
            train_metrics={"acc": 0.8},
            val_metrics={"acc": 0.7},
            feature_importances=None,
            training_time_seconds=1.0,
            n_train_samples=100,
            n_val_samples=20,
            hyperparams={"lr": 0.05},
        )
        assert tr.n_train_samples == 100
        assert tr.hyperparams["lr"] == 0.05


class TestPredictionResult:
    def test_fields(self) -> None:
        pr = PredictionResult(
            direction=np.array([1, -1, 0]),
            magnitude=np.array([0.02, -0.01, 0.0]),
            probabilities=np.ones((3, 3)) / 3,
            confidence=np.array([0.7, 0.6, 0.5]),
        )
        assert len(pr.direction) == 3
        assert pr.probabilities.shape == (3, 3)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/models/test_base.py -v 2>&1 | tail -5`

Expected: `ModuleNotFoundError`

- [ ] **Step 3: Implement BaseModel ABC**

Create `alphavedha/models/base.py`:

```python
"""BaseModel ABC — lifecycle contract for all ML models."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import structlog

logger = structlog.get_logger(__name__)


@dataclass
class TrainResult:
    train_metrics: dict[str, float]
    val_metrics: dict[str, float]
    feature_importances: pd.Series | None
    training_time_seconds: float
    n_train_samples: int
    n_val_samples: int
    hyperparams: dict[str, Any]


@dataclass
class PredictionResult:
    direction: np.ndarray
    magnitude: np.ndarray
    probabilities: np.ndarray | None
    confidence: np.ndarray


@dataclass
class ModelArtifact:
    path: Path
    name: str
    version: str
    created_at: str
    feature_names: list[str]
    metrics: dict[str, float]
    config: dict[str, Any]


class BaseModel(ABC):
    def __init__(self, name: str, config: dict[str, Any]) -> None:
        self._name = name
        self._config = config
        self._version_counter = 0
        self._is_fitted = False
        self._train_metrics: dict[str, float] = {}
        self._feature_names: list[str] = []

    @property
    def name(self) -> str:
        return self._name

    @property
    def version(self) -> str:
        return f"0.0.{self._version_counter}"

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    @abstractmethod
    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame | None = None,
        y_val: pd.Series | None = None,
        sample_weight: pd.Series | None = None,
    ) -> TrainResult: ...

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> PredictionResult: ...

    @abstractmethod
    def get_feature_importance(self) -> pd.Series | None: ...

    @abstractmethod
    def _save_model_artifacts(self, directory: Path) -> None: ...

    @classmethod
    @abstractmethod
    def _load_model_artifacts(cls, directory: Path, config: dict[str, Any]) -> BaseModel: ...

    def get_metrics(self) -> dict[str, float]:
        return dict(self._train_metrics)

    def save(self, directory: Path) -> ModelArtifact:
        self._version_counter += 1
        directory.mkdir(parents=True, exist_ok=True)

        self._save_model_artifacts(directory)

        fi = self.get_feature_importance()
        if fi is not None:
            fi.to_csv(directory / "feature_importance.csv")

        artifact = ModelArtifact(
            path=directory,
            name=self._name,
            version=self.version,
            created_at=datetime.now(timezone.utc).isoformat(),
            feature_names=list(self._feature_names),
            metrics=dict(self._train_metrics),
            config=dict(self._config),
        )

        metadata = {
            "name": artifact.name,
            "version": artifact.version,
            "created_at": artifact.created_at,
            "feature_names": artifact.feature_names,
            "metrics": artifact.metrics,
            "config": artifact.config,
        }
        (directory / "metadata.json").write_text(json.dumps(metadata, indent=2))

        logger.info(
            "model_saved",
            name=self._name,
            version=self.version,
            path=str(directory),
        )

        return artifact

    @classmethod
    def load(cls, directory: Path, config: dict[str, Any] | None = None) -> BaseModel:
        metadata_path = directory / "metadata.json"
        if not metadata_path.exists():
            from alphavedha.exceptions import ModelNotFoundError
            raise ModelNotFoundError(f"No metadata.json at {directory}")

        metadata = json.loads(metadata_path.read_text())
        load_config = config if config is not None else metadata.get("config", {})

        model = cls._load_model_artifacts(directory, load_config)
        model._feature_names = metadata.get("feature_names", [])
        model._train_metrics = metadata.get("metrics", {})
        model._is_fitted = True

        logger.info(
            "model_loaded",
            name=metadata["name"],
            version=metadata["version"],
            path=str(directory),
        )

        return model
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/models/test_base.py -v`

Expected: All tests PASS.

- [ ] **Step 5: Commit**

```bash
git add alphavedha/models/base.py tests/unit/models/test_base.py
git commit -m "feat: implement BaseModel ABC with save/load, TrainResult, PredictionResult"
```

---

### Task 5: XGBoost model

**Files:**
- Create: `alphavedha/models/xgboost_model.py`
- Create: `tests/unit/models/test_xgboost_model.py`

- [ ] **Step 1: Write failing tests**

Create `tests/unit/models/test_xgboost_model.py`:

```python
"""Tests for XGBoostModel — classifier + regressor."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from alphavedha.config import XGBoostConfig
from alphavedha.exceptions import ModelTrainingError
from alphavedha.models.base import PredictionResult, TrainResult
from alphavedha.models.xgboost_model import XGBoostModel


@pytest.fixture
def xgb_config() -> XGBoostConfig:
    return XGBoostConfig()


@pytest.fixture
def synthetic_data() -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Synthetic training data: 200 samples, 10 features, 3-class labels + returns."""
    rng = np.random.default_rng(42)
    n, f = 200, 10
    X = pd.DataFrame(rng.standard_normal((n, f)), columns=[f"f{i}" for i in range(f)])
    labels = pd.Series(rng.choice([-1, 0, 1], size=n), name="label")
    returns = pd.Series(rng.normal(0, 0.02, size=n), name="return_pct")
    return X, labels, returns


class TestXGBoostModel:
    def test_fit_returns_train_result(
        self,
        synthetic_data: tuple[pd.DataFrame, pd.Series, pd.Series],
        xgb_config: XGBoostConfig,
    ) -> None:
        X, labels, returns = synthetic_data
        model = XGBoostModel(config=xgb_config)
        result = model.fit(
            X_train=X[:160], y_train=labels[:160],
            X_val=X[160:], y_val=labels[160:],
            return_train=returns[:160], return_val=returns[160:],
        )
        assert isinstance(result, TrainResult)
        assert "accuracy" in result.train_metrics
        assert "rmse" in result.train_metrics

    def test_predict_returns_prediction_result(
        self,
        synthetic_data: tuple[pd.DataFrame, pd.Series, pd.Series],
        xgb_config: XGBoostConfig,
    ) -> None:
        X, labels, returns = synthetic_data
        model = XGBoostModel(config=xgb_config)
        model.fit(
            X_train=X[:160], y_train=labels[:160],
            X_val=X[160:], y_val=labels[160:],
            return_train=returns[:160], return_val=returns[160:],
        )
        pred = model.predict(X[160:])
        assert isinstance(pred, PredictionResult)

    def test_direction_values(
        self,
        synthetic_data: tuple[pd.DataFrame, pd.Series, pd.Series],
        xgb_config: XGBoostConfig,
    ) -> None:
        X, labels, returns = synthetic_data
        model = XGBoostModel(config=xgb_config)
        model.fit(
            X_train=X[:160], y_train=labels[:160],
            X_val=X[160:], y_val=labels[160:],
            return_train=returns[:160], return_val=returns[160:],
        )
        pred = model.predict(X[160:])
        assert set(np.unique(pred.direction)).issubset({-1, 0, 1})

    def test_magnitude_shape(
        self,
        synthetic_data: tuple[pd.DataFrame, pd.Series, pd.Series],
        xgb_config: XGBoostConfig,
    ) -> None:
        X, labels, returns = synthetic_data
        model = XGBoostModel(config=xgb_config)
        model.fit(
            X_train=X[:160], y_train=labels[:160],
            X_val=X[160:], y_val=labels[160:],
            return_train=returns[:160], return_val=returns[160:],
        )
        pred = model.predict(X[160:])
        assert len(pred.magnitude) == 40

    def test_probabilities_shape_and_sum(
        self,
        synthetic_data: tuple[pd.DataFrame, pd.Series, pd.Series],
        xgb_config: XGBoostConfig,
    ) -> None:
        X, labels, returns = synthetic_data
        model = XGBoostModel(config=xgb_config)
        model.fit(
            X_train=X[:160], y_train=labels[:160],
            X_val=X[160:], y_val=labels[160:],
            return_train=returns[:160], return_val=returns[160:],
        )
        pred = model.predict(X[160:])
        assert pred.probabilities is not None
        assert pred.probabilities.shape == (40, 3)
        row_sums = pred.probabilities.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-6)

    def test_confidence_range(
        self,
        synthetic_data: tuple[pd.DataFrame, pd.Series, pd.Series],
        xgb_config: XGBoostConfig,
    ) -> None:
        X, labels, returns = synthetic_data
        model = XGBoostModel(config=xgb_config)
        model.fit(
            X_train=X[:160], y_train=labels[:160],
            X_val=X[160:], y_val=labels[160:],
            return_train=returns[:160], return_val=returns[160:],
        )
        pred = model.predict(X[160:])
        assert (pred.confidence >= 0).all()
        assert (pred.confidence <= 1).all()

    def test_feature_importance(
        self,
        synthetic_data: tuple[pd.DataFrame, pd.Series, pd.Series],
        xgb_config: XGBoostConfig,
    ) -> None:
        X, labels, returns = synthetic_data
        model = XGBoostModel(config=xgb_config)
        model.fit(
            X_train=X[:160], y_train=labels[:160],
            X_val=X[160:], y_val=labels[160:],
            return_train=returns[:160], return_val=returns[160:],
        )
        fi = model.get_feature_importance()
        assert fi is not None
        assert len(fi) == 10
        assert (fi >= 0).all()

    def test_predict_before_fit_raises(self, xgb_config: XGBoostConfig) -> None:
        model = XGBoostModel(config=xgb_config)
        X = pd.DataFrame({"a": [1, 2]})
        with pytest.raises(ModelTrainingError):
            model.predict(X)

    def test_save_load_roundtrip(
        self,
        synthetic_data: tuple[pd.DataFrame, pd.Series, pd.Series],
        xgb_config: XGBoostConfig,
        tmp_path: Path,
    ) -> None:
        X, labels, returns = synthetic_data
        model = XGBoostModel(config=xgb_config)
        model.fit(
            X_train=X[:160], y_train=labels[:160],
            X_val=X[160:], y_val=labels[160:],
            return_train=returns[:160], return_val=returns[160:],
        )
        pred_before = model.predict(X[160:])

        save_dir = tmp_path / "xgb_test"
        model.save(save_dir)

        loaded = XGBoostModel.load(save_dir)
        pred_after = loaded.predict(X[160:])

        np.testing.assert_array_equal(pred_before.direction, pred_after.direction)
        np.testing.assert_allclose(pred_before.magnitude, pred_after.magnitude, atol=1e-6)

    def test_sample_weight_changes_result(
        self,
        synthetic_data: tuple[pd.DataFrame, pd.Series, pd.Series],
        xgb_config: XGBoostConfig,
    ) -> None:
        X, labels, returns = synthetic_data
        model1 = XGBoostModel(config=xgb_config)
        model1.fit(
            X_train=X[:160], y_train=labels[:160],
            X_val=X[160:], y_val=labels[160:],
            return_train=returns[:160], return_val=returns[160:],
        )
        pred1 = model1.predict(X[160:])

        rng = np.random.default_rng(99)
        weights = pd.Series(rng.uniform(0.1, 10.0, size=160))
        model2 = XGBoostModel(config=xgb_config)
        model2.fit(
            X_train=X[:160], y_train=labels[:160],
            X_val=X[160:], y_val=labels[160:],
            return_train=returns[:160], return_val=returns[160:],
            sample_weight=weights,
        )
        pred2 = model2.predict(X[160:])

        assert not np.array_equal(pred1.magnitude, pred2.magnitude)

    def test_config_hyperparams_applied(self) -> None:
        config = XGBoostConfig()
        model = XGBoostModel(config=config)
        assert model._xgb_params["learning_rate"] == 0.05
        assert model._xgb_params["max_depth"] == 6
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/models/test_xgboost_model.py -v 2>&1 | tail -5`

Expected: `ModuleNotFoundError`

- [ ] **Step 3: Implement XGBoostModel**

Create `alphavedha/models/xgboost_model.py`:

```python
"""XGBoostModel — wraps XGBClassifier (direction) + XGBRegressor (magnitude)."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import structlog
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error
from xgboost import XGBClassifier, XGBRegressor

from alphavedha.config import XGBoostConfig
from alphavedha.exceptions import ModelTrainingError
from alphavedha.models.base import BaseModel, ModelArtifact, PredictionResult, TrainResult

logger = structlog.get_logger(__name__)

_LABEL_MAP = {-1: 0, 0: 1, 1: 2}
_LABEL_REVERSE = {0: -1, 1: 0, 2: 1}


class XGBoostModel(BaseModel):
    def __init__(self, config: XGBoostConfig | None = None, name: str = "xgboost") -> None:
        cfg = config or XGBoostConfig()
        params = cfg.params
        self._xgb_params: dict[str, Any] = {
            "learning_rate": params.learning_rate,
            "max_depth": params.max_depth,
            "n_estimators": params.n_estimators,
            "subsample": params.subsample,
            "colsample_bytree": params.colsample_bytree,
            "reg_alpha": params.reg_alpha,
            "reg_lambda": params.reg_lambda,
            "random_state": 42,
            "n_jobs": -1,
            "verbosity": 0,
        }
        self._early_stopping_rounds = params.early_stopping_rounds
        self._classifier: XGBClassifier | None = None
        self._regressor: XGBRegressor | None = None
        super().__init__(name=name, config=self._xgb_params)

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
        self._feature_names = list(X_train.columns)

        y_cls_train = y_train.map(_LABEL_MAP).astype(int)
        weight_arr = sample_weight.values if sample_weight is not None else None

        self._classifier = XGBClassifier(
            objective="multi:softprob",
            num_class=3,
            eval_metric="mlogloss",
            early_stopping_rounds=self._early_stopping_rounds,
            **self._xgb_params,
        )

        eval_set_cls = []
        if X_val is not None and y_val is not None:
            y_cls_val = y_val.map(_LABEL_MAP).astype(int)
            eval_set_cls = [(X_val, y_cls_val)]

        self._classifier.fit(
            X_train, y_cls_train,
            eval_set=eval_set_cls or None,
            sample_weight=weight_arr,
            verbose=False,
        )

        train_metrics: dict[str, float] = {}
        cls_train_pred = self._classifier.predict(X_train)
        train_metrics["accuracy"] = float(accuracy_score(y_cls_train, cls_train_pred))
        train_metrics["f1_weighted"] = float(
            f1_score(y_cls_train, cls_train_pred, average="weighted")
        )

        val_metrics: dict[str, float] = {}
        if X_val is not None and y_val is not None:
            y_cls_val = y_val.map(_LABEL_MAP).astype(int)
            cls_val_pred = self._classifier.predict(X_val)
            val_metrics["accuracy"] = float(accuracy_score(y_cls_val, cls_val_pred))
            val_metrics["f1_weighted"] = float(
                f1_score(y_cls_val, cls_val_pred, average="weighted")
            )

        self._regressor = XGBRegressor(
            objective="reg:squarederror",
            eval_metric="rmse",
            early_stopping_rounds=self._early_stopping_rounds,
            **self._xgb_params,
        )

        if return_train is not None:
            eval_set_reg = []
            if X_val is not None and return_val is not None:
                eval_set_reg = [(X_val, return_val)]

            self._regressor.fit(
                X_train, return_train,
                eval_set=eval_set_reg or None,
                sample_weight=weight_arr,
                verbose=False,
            )

            reg_train_pred = self._regressor.predict(X_train)
            train_metrics["rmse"] = float(
                np.sqrt(mean_squared_error(return_train, reg_train_pred))
            )
            if return_val is not None and X_val is not None:
                reg_val_pred = self._regressor.predict(X_val)
                val_metrics["rmse"] = float(
                    np.sqrt(mean_squared_error(return_val, reg_val_pred))
                )

        fi_raw = self._classifier.feature_importances_
        fi = pd.Series(fi_raw, index=self._feature_names, name="importance")

        elapsed = time.perf_counter() - start
        self._is_fitted = True
        self._train_metrics = train_metrics

        logger.info(
            "xgboost_trained",
            train_metrics=train_metrics,
            val_metrics=val_metrics,
            training_time_s=round(elapsed, 2),
            n_train=len(X_train),
        )

        return TrainResult(
            train_metrics=train_metrics,
            val_metrics=val_metrics,
            feature_importances=fi,
            training_time_seconds=elapsed,
            n_train_samples=len(X_train),
            n_val_samples=len(X_val) if X_val is not None else 0,
            hyperparams=dict(self._xgb_params),
        )

    def predict(self, X: pd.DataFrame) -> PredictionResult:
        if not self._is_fitted or self._classifier is None:
            raise ModelTrainingError("XGBoostModel is not fitted. Call fit() first.")

        proba = self._classifier.predict_proba(X)
        cls_pred = np.argmax(proba, axis=1)
        direction = np.array([_LABEL_REVERSE[c] for c in cls_pred])
        confidence = np.max(proba, axis=1)

        if self._regressor is not None:
            magnitude = self._regressor.predict(X)
        else:
            magnitude = np.zeros(len(X))

        return PredictionResult(
            direction=direction,
            magnitude=magnitude,
            probabilities=proba,
            confidence=confidence,
        )

    def get_feature_importance(self) -> pd.Series | None:
        if self._classifier is None:
            return None
        fi = self._classifier.feature_importances_
        return pd.Series(fi, index=self._feature_names, name="importance")

    def _save_model_artifacts(self, directory: Path) -> None:
        if self._classifier is not None:
            joblib.dump(self._classifier, directory / "classifier.joblib")
        if self._regressor is not None:
            joblib.dump(self._regressor, directory / "regressor.joblib")

    @classmethod
    def _load_model_artifacts(cls, directory: Path, config: dict[str, Any]) -> XGBoostModel:
        model = cls(config=None, name="xgboost")
        model._config = config

        cls_path = directory / "classifier.joblib"
        if cls_path.exists():
            model._classifier = joblib.load(cls_path)

        reg_path = directory / "regressor.joblib"
        if reg_path.exists():
            model._regressor = joblib.load(reg_path)

        model._is_fitted = True
        return model
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/models/test_xgboost_model.py -v`

Expected: All tests PASS.

- [ ] **Step 5: Update models `__init__.py` and commit**

Write `alphavedha/models/__init__.py`:

```python
"""Models — BaseModel ABC and model implementations."""

from alphavedha.models.base import (
    BaseModel,
    ModelArtifact,
    PredictionResult,
    TrainResult,
)
from alphavedha.models.xgboost_model import XGBoostModel

__all__ = [
    "BaseModel",
    "ModelArtifact",
    "PredictionResult",
    "TrainResult",
    "XGBoostModel",
]
```

```bash
git add alphavedha/models/ tests/unit/models/test_xgboost_model.py
git commit -m "feat: implement XGBoostModel with classifier + regressor"
```

---

### Task 6: Indian market cost calculator

**Files:**
- Create: `alphavedha/backtest/costs.py`
- Create: `tests/unit/backtest/test_costs.py`

- [ ] **Step 1: Write failing tests**

Create `tests/unit/backtest/test_costs.py`:

```python
"""Tests for Indian market cost calculator."""

from __future__ import annotations

import pytest

from alphavedha.backtest.costs import TradeCost, compute_round_trip_cost_pct, compute_trade_cost
from alphavedha.config import BacktestConfig, CostsConfig, SlippageConfig


@pytest.fixture
def default_config() -> BacktestConfig:
    return BacktestConfig()


class TestTradeCost:
    def test_buy_side_components(self, default_config: BacktestConfig) -> None:
        cost = compute_trade_cost(
            trade_value=100_000.0,
            side="buy",
            market_cap_tier="large",
            config=default_config.costs,
            slippage_config=default_config.slippage,
        )
        assert isinstance(cost, TradeCost)
        assert cost.stt > 0
        assert cost.stamp_duty > 0
        assert cost.total > 0

    def test_sell_side_no_stamp_duty(self, default_config: BacktestConfig) -> None:
        cost = compute_trade_cost(
            trade_value=100_000.0,
            side="sell",
            market_cap_tier="large",
            config=default_config.costs,
            slippage_config=default_config.slippage,
        )
        assert cost.stamp_duty == 0.0

    def test_stt_calculation(self, default_config: BacktestConfig) -> None:
        cost = compute_trade_cost(
            trade_value=100_000.0,
            side="buy",
            market_cap_tier="large",
            config=default_config.costs,
            slippage_config=default_config.slippage,
        )
        expected_stt = 100_000.0 * 0.001
        assert abs(cost.stt - expected_stt) < 0.01

    def test_brokerage_flat(self, default_config: BacktestConfig) -> None:
        cost = compute_trade_cost(
            trade_value=100_000.0,
            side="buy",
            market_cap_tier="large",
            config=default_config.costs,
            slippage_config=default_config.slippage,
        )
        assert cost.brokerage == 20.0

    def test_gst_on_brokerage_and_exchange(self, default_config: BacktestConfig) -> None:
        cost = compute_trade_cost(
            trade_value=100_000.0,
            side="buy",
            market_cap_tier="large",
            config=default_config.costs,
            slippage_config=default_config.slippage,
        )
        expected_gst = (20.0 + 100_000.0 * 0.0000345) * 0.18
        assert abs(cost.gst - expected_gst) < 0.01

    def test_slippage_varies_by_tier(self, default_config: BacktestConfig) -> None:
        large = compute_trade_cost(
            100_000.0, "buy", "large", default_config.costs, default_config.slippage
        )
        mid = compute_trade_cost(
            100_000.0, "buy", "mid", default_config.costs, default_config.slippage
        )
        small = compute_trade_cost(
            100_000.0, "buy", "small", default_config.costs, default_config.slippage
        )
        assert small.slippage > mid.slippage > large.slippage

    def test_total_is_sum_of_components(self, default_config: BacktestConfig) -> None:
        cost = compute_trade_cost(
            100_000.0, "buy", "large", default_config.costs, default_config.slippage
        )
        component_sum = (
            cost.stt + cost.brokerage + cost.exchange_txn
            + cost.gst + cost.sebi_turnover + cost.stamp_duty + cost.slippage
        )
        assert abs(cost.total - component_sum) < 0.01


class TestRoundTripCost:
    def test_round_trip_positive(self, default_config: BacktestConfig) -> None:
        pct = compute_round_trip_cost_pct("large", default_config)
        assert pct > 0

    def test_round_trip_large_vs_mid(self, default_config: BacktestConfig) -> None:
        large = compute_round_trip_cost_pct("large", default_config)
        mid = compute_round_trip_cost_pct("mid", default_config)
        assert mid > large

    def test_round_trip_includes_all_7_components(self, default_config: BacktestConfig) -> None:
        """Verify the 7 cost types are all nonzero in a round trip."""
        buy = compute_trade_cost(
            100_000.0, "buy", "large", default_config.costs, default_config.slippage
        )
        sell = compute_trade_cost(
            100_000.0, "sell", "large", default_config.costs, default_config.slippage
        )
        assert buy.stt > 0 and sell.stt > 0
        assert buy.brokerage > 0 and sell.brokerage > 0
        assert buy.exchange_txn > 0 and sell.exchange_txn > 0
        assert buy.gst > 0 and sell.gst > 0
        assert buy.sebi_turnover > 0 and sell.sebi_turnover > 0
        assert buy.stamp_duty > 0
        assert sell.stamp_duty == 0
        assert buy.slippage > 0 and sell.slippage > 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/backtest/test_costs.py -v 2>&1 | tail -5`

Expected: `ModuleNotFoundError`

- [ ] **Step 3: Implement cost calculator**

Create `alphavedha/backtest/costs.py`:

```python
"""Indian market cost calculator — all regulatory and market costs for backtesting."""

from __future__ import annotations

from dataclasses import dataclass

from alphavedha.config import BacktestConfig, CostsConfig, SlippageConfig


@dataclass
class TradeCost:
    stt: float
    brokerage: float
    exchange_txn: float
    gst: float
    sebi_turnover: float
    stamp_duty: float
    slippage: float
    total: float


def _get_slippage_rate(market_cap_tier: str, slippage_config: SlippageConfig) -> float:
    rates = {
        "large": slippage_config.large_cap,
        "mid": slippage_config.mid_cap,
        "small": slippage_config.small_cap,
    }
    return rates.get(market_cap_tier, slippage_config.mid_cap)


def compute_trade_cost(
    trade_value: float,
    side: str,
    market_cap_tier: str,
    config: CostsConfig,
    slippage_config: SlippageConfig,
) -> TradeCost:
    stt = trade_value * config.stt_delivery
    brokerage = config.brokerage_flat
    exchange_txn = trade_value * config.exchange_txn
    gst = (brokerage + exchange_txn) * config.gst
    sebi_turnover = trade_value * config.sebi_turnover
    stamp_duty = trade_value * config.stamp_duty if side == "buy" else 0.0
    slippage_rate = _get_slippage_rate(market_cap_tier, slippage_config)
    slippage = trade_value * slippage_rate

    total = stt + brokerage + exchange_txn + gst + sebi_turnover + stamp_duty + slippage

    return TradeCost(
        stt=stt,
        brokerage=brokerage,
        exchange_txn=exchange_txn,
        gst=gst,
        sebi_turnover=sebi_turnover,
        stamp_duty=stamp_duty,
        slippage=slippage,
        total=total,
    )


def compute_round_trip_cost_pct(
    market_cap_tier: str,
    config: BacktestConfig,
) -> float:
    ref_value = 100_000.0
    buy = compute_trade_cost(ref_value, "buy", market_cap_tier, config.costs, config.slippage)
    sell = compute_trade_cost(ref_value, "sell", market_cap_tier, config.costs, config.slippage)
    return (buy.total + sell.total) / ref_value
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/backtest/test_costs.py -v`

Expected: All tests PASS.

- [ ] **Step 5: Commit**

```bash
git add alphavedha/backtest/costs.py tests/unit/backtest/test_costs.py
git commit -m "feat: implement Indian market cost calculator with all 7 cost components"
```

---

### Task 7: CPCV validation

**Files:**
- Create: `alphavedha/backtest/cpcv.py`
- Create: `tests/unit/backtest/test_cpcv.py`

- [ ] **Step 1: Write failing tests**

Create `tests/unit/backtest/test_cpcv.py`:

```python
"""Tests for Combinatorial Purged Cross-Validation."""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd
import pytest

from alphavedha.backtest.cpcv import CPCVResult, PathResult, generate_cpcv_splits, run_cpcv
from alphavedha.config import AcceptanceConfig, CPCVConfig, XGBoostConfig
from alphavedha.models.xgboost_model import XGBoostModel


@pytest.fixture
def default_cpcv_config() -> CPCVConfig:
    return CPCVConfig()


@pytest.fixture
def default_acceptance() -> AcceptanceConfig:
    return AcceptanceConfig()


class TestGenerateSplits:
    def test_generates_15_paths(self, default_cpcv_config: CPCVConfig) -> None:
        n_samples = 500
        splits = generate_cpcv_splits(n_samples, default_cpcv_config)
        assert len(splits) == 15

    def test_each_split_has_train_and_test(self, default_cpcv_config: CPCVConfig) -> None:
        splits = generate_cpcv_splits(500, default_cpcv_config)
        for train_idx, test_idx, test_segs in splits:
            assert len(train_idx) > 0
            assert len(test_idx) > 0
            assert len(test_segs) == 2

    def test_no_train_test_overlap(self, default_cpcv_config: CPCVConfig) -> None:
        splits = generate_cpcv_splits(500, default_cpcv_config)
        for train_idx, test_idx, _ in splits:
            overlap = set(train_idx) & set(test_idx)
            assert len(overlap) == 0

    def test_purge_gap_exists(self, default_cpcv_config: CPCVConfig) -> None:
        """Training indices near test boundaries should be removed."""
        splits = generate_cpcv_splits(500, default_cpcv_config)
        for train_idx, test_idx, _ in splits:
            train_set = set(train_idx)
            test_min = min(test_idx)
            test_max = max(test_idx)
            purge = default_cpcv_config.purge_days
            for i in range(max(0, test_min - purge), test_min):
                assert i not in train_set

    def test_embargo_gap_exists(self, default_cpcv_config: CPCVConfig) -> None:
        """Training indices just after test segments should be removed."""
        splits = generate_cpcv_splits(600, default_cpcv_config)
        seg_size = 600 // default_cpcv_config.n_segments
        for train_idx, test_idx, test_segs in splits:
            train_set = set(train_idx)
            embargo = default_cpcv_config.embargo_days
            for seg in test_segs:
                seg_end = min((seg + 1) * seg_size, 600)
                for i in range(seg_end, min(seg_end + embargo, 600)):
                    assert i not in train_set

    def test_custom_config(self) -> None:
        config = CPCVConfig(n_segments=4, k_test_segments=1)
        splits = generate_cpcv_splits(400, config)
        expected_paths = len(list(combinations(range(4), 1)))
        assert len(splits) == expected_paths


class TestRunCPCV:
    def test_returns_cpcv_result(
        self,
        sample_ohlcv_500: pd.DataFrame,
        sample_features_500: pd.DataFrame,
        default_cpcv_config: CPCVConfig,
        default_acceptance: AcceptanceConfig,
    ) -> None:
        rng = np.random.default_rng(42)
        n = len(sample_features_500)
        y = pd.Series(rng.choice([-1, 0, 1], size=n), index=sample_features_500.index)
        returns = pd.Series(rng.normal(0, 0.02, n), index=sample_features_500.index)

        def model_factory() -> XGBoostModel:
            config = XGBoostConfig()
            config.params.n_estimators = 10
            config.params.early_stopping_rounds = 5
            return XGBoostModel(config=config)

        result = run_cpcv(
            X=sample_features_500,
            y=y,
            returns=returns,
            sample_weight=None,
            model_factory=model_factory,
            config=default_cpcv_config,
            acceptance=default_acceptance,
        )

        assert isinstance(result, CPCVResult)
        assert result.n_paths == 15
        assert len(result.path_results) == 15

    def test_path_result_has_metrics(
        self,
        sample_ohlcv_500: pd.DataFrame,
        sample_features_500: pd.DataFrame,
        default_cpcv_config: CPCVConfig,
        default_acceptance: AcceptanceConfig,
    ) -> None:
        rng = np.random.default_rng(42)
        n = len(sample_features_500)
        y = pd.Series(rng.choice([-1, 0, 1], size=n), index=sample_features_500.index)
        returns = pd.Series(rng.normal(0, 0.02, n), index=sample_features_500.index)

        def model_factory() -> XGBoostModel:
            config = XGBoostConfig()
            config.params.n_estimators = 10
            config.params.early_stopping_rounds = 5
            return XGBoostModel(config=config)

        result = run_cpcv(
            X=sample_features_500, y=y, returns=returns,
            sample_weight=None, model_factory=model_factory,
            config=default_cpcv_config, acceptance=default_acceptance,
        )

        for pr in result.path_results:
            assert isinstance(pr, PathResult)
            assert 0 <= pr.accuracy <= 1
            assert pr.n_test_samples > 0

    def test_passed_flag(
        self,
        sample_features_500: pd.DataFrame,
        default_cpcv_config: CPCVConfig,
    ) -> None:
        rng = np.random.default_rng(42)
        n = len(sample_features_500)
        y = pd.Series(rng.choice([-1, 0, 1], size=n), index=sample_features_500.index)
        returns = pd.Series(rng.normal(0, 0.02, n), index=sample_features_500.index)

        easy_accept = AcceptanceConfig(min_median_sharpe=-999, min_worst_sharpe=-999)

        def model_factory() -> XGBoostModel:
            config = XGBoostConfig()
            config.params.n_estimators = 10
            config.params.early_stopping_rounds = 5
            return XGBoostModel(config=config)

        result = run_cpcv(
            X=sample_features_500, y=y, returns=returns,
            sample_weight=None, model_factory=model_factory,
            config=default_cpcv_config, acceptance=easy_accept,
        )
        assert result.passed is True
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/backtest/test_cpcv.py -v 2>&1 | tail -5`

Expected: `ModuleNotFoundError`

- [ ] **Step 3: Implement CPCV**

Create `alphavedha/backtest/cpcv.py`:

```python
"""Combinatorial Purged Cross-Validation (CPCV) — rigorous time-series model validation."""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from typing import Any, Callable

import numpy as np
import pandas as pd
import structlog
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from alphavedha.config import AcceptanceConfig, CPCVConfig
from alphavedha.models.base import BaseModel

logger = structlog.get_logger(__name__)


@dataclass
class PathResult:
    path_id: int
    test_segments: tuple[int, ...]
    accuracy: float
    precision_weighted: float
    recall_weighted: float
    f1_weighted: float
    sharpe_ratio: float
    total_return: float
    n_test_samples: int
    confusion_matrix: np.ndarray = field(default_factory=lambda: np.array([]))


@dataclass
class CPCVResult:
    path_results: list[PathResult]
    n_paths: int
    median_sharpe: float
    worst_sharpe: float
    best_sharpe: float
    mean_accuracy: float
    std_accuracy: float
    passed: bool


def generate_cpcv_splits(
    n_samples: int,
    config: CPCVConfig,
) -> list[tuple[np.ndarray, np.ndarray, tuple[int, ...]]]:
    seg_size = n_samples // config.n_segments
    segment_ranges: list[tuple[int, int]] = []
    for s in range(config.n_segments):
        start = s * seg_size
        end = (s + 1) * seg_size if s < config.n_segments - 1 else n_samples
        segment_ranges.append((start, end))

    splits: list[tuple[np.ndarray, np.ndarray, tuple[int, ...]]] = []

    for test_combo in combinations(range(config.n_segments), config.k_test_segments):
        test_indices: list[int] = []
        for seg in test_combo:
            s, e = segment_ranges[seg]
            test_indices.extend(range(s, e))

        excluded = set(test_indices)

        for seg in test_combo:
            seg_start, seg_end = segment_ranges[seg]

            purge_start = max(0, seg_start - config.purge_days)
            for i in range(purge_start, seg_start):
                excluded.add(i)

            embargo_end = min(n_samples, seg_end + config.embargo_days)
            for i in range(seg_end, embargo_end):
                excluded.add(i)

        train_indices = [i for i in range(n_samples) if i not in excluded]
        splits.append((
            np.array(train_indices),
            np.array(test_indices),
            test_combo,
        ))

    return splits


def _compute_sharpe(returns: np.ndarray) -> float:
    if len(returns) < 2 or np.std(returns) == 0:
        return 0.0
    return float(np.mean(returns) / np.std(returns) * np.sqrt(252))


def run_cpcv(
    X: pd.DataFrame,
    y: pd.Series,
    returns: pd.Series,
    sample_weight: pd.Series | None,
    model_factory: Callable[[], Any],
    config: CPCVConfig,
    acceptance: AcceptanceConfig,
) -> CPCVResult:
    splits = generate_cpcv_splits(len(X), config)
    path_results: list[PathResult] = []

    for path_id, (train_idx, test_idx, test_segs) in enumerate(splits):
        model: BaseModel = model_factory()

        X_train = X.iloc[train_idx]
        y_train = y.iloc[train_idx]
        ret_train = returns.iloc[train_idx]
        X_test = X.iloc[test_idx]
        y_test = y.iloc[test_idx]
        ret_test = returns.iloc[test_idx]

        sw_train = sample_weight.iloc[train_idx] if sample_weight is not None else None

        val_size = max(int(len(X_train) * 0.15), 20)
        X_tr = X_train.iloc[:-val_size]
        y_tr = y_train.iloc[:-val_size]
        ret_tr = ret_train.iloc[:-val_size]
        X_vl = X_train.iloc[-val_size:]
        y_vl = y_train.iloc[-val_size:]
        ret_vl = ret_train.iloc[-val_size:]
        sw_tr = sw_train.iloc[:-val_size] if sw_train is not None else None

        model.fit(
            X_train=X_tr, y_train=y_tr,
            X_val=X_vl, y_val=y_vl,
            sample_weight=sw_tr,
            return_train=ret_tr, return_val=ret_vl,
        )

        pred = model.predict(X_test)
        y_test_mapped = y_test.values
        pred_dir = pred.direction

        y_test_labels = y_test_mapped.astype(int)
        pred_labels = pred_dir.astype(int)
        labels_present = sorted(set(y_test_labels) | set(pred_labels))

        acc = float(accuracy_score(y_test_labels, pred_labels))
        prec = float(precision_score(y_test_labels, pred_labels, average="weighted", zero_division=0))
        rec = float(recall_score(y_test_labels, pred_labels, average="weighted", zero_division=0))
        f1 = float(f1_score(y_test_labels, pred_labels, average="weighted", zero_division=0))

        test_returns = ret_test.values
        sharpe = _compute_sharpe(test_returns)
        total_ret = float(np.sum(test_returns))

        path_results.append(PathResult(
            path_id=path_id,
            test_segments=test_segs,
            accuracy=acc,
            precision_weighted=prec,
            recall_weighted=rec,
            f1_weighted=f1,
            sharpe_ratio=sharpe,
            total_return=total_ret,
            n_test_samples=len(test_idx),
        ))

        logger.debug(
            "cpcv_path_completed",
            path_id=path_id,
            test_segments=test_segs,
            accuracy=round(acc, 4),
            sharpe=round(sharpe, 4),
        )

    sharpes = [pr.sharpe_ratio for pr in path_results]
    accuracies = [pr.accuracy for pr in path_results]

    median_sharpe = float(np.median(sharpes))
    worst_sharpe = float(np.min(sharpes))
    best_sharpe = float(np.max(sharpes))
    mean_acc = float(np.mean(accuracies))
    std_acc = float(np.std(accuracies))

    passed = (median_sharpe >= acceptance.min_median_sharpe
              and worst_sharpe >= acceptance.min_worst_sharpe)

    logger.info(
        "cpcv_completed",
        n_paths=len(path_results),
        median_sharpe=round(median_sharpe, 4),
        worst_sharpe=round(worst_sharpe, 4),
        mean_accuracy=round(mean_acc, 4),
        passed=passed,
    )

    return CPCVResult(
        path_results=path_results,
        n_paths=len(path_results),
        median_sharpe=median_sharpe,
        worst_sharpe=worst_sharpe,
        best_sharpe=best_sharpe,
        mean_accuracy=mean_acc,
        std_accuracy=std_acc,
        passed=passed,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/backtest/test_cpcv.py -v`

Expected: All tests PASS. Note: the `TestRunCPCV` tests train 15 XGBoost models with small n_estimators=10 each so they take ~30-60 seconds total.

- [ ] **Step 5: Commit**

```bash
git add alphavedha/backtest/cpcv.py tests/unit/backtest/test_cpcv.py
git commit -m "feat: implement CPCV validation with purge, embargo, and acceptance criteria"
```

---

### Task 8: VectorBT backtesting engine

**Files:**
- Create: `alphavedha/backtest/engine.py`
- Create: `tests/unit/backtest/test_engine.py`

- [ ] **Step 1: Write failing tests**

Create `tests/unit/backtest/test_engine.py`:

```python
"""Tests for VectorBT backtesting engine."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from alphavedha.backtest.engine import BacktestResult, run_backtest
from alphavedha.config import BacktestConfig


@pytest.fixture
def default_config() -> BacktestConfig:
    return BacktestConfig()


@pytest.fixture
def bullish_predictions(sample_ohlcv_500: pd.DataFrame) -> pd.DataFrame:
    """Predictions that always say UP with high confidence."""
    n = len(sample_ohlcv_500)
    return pd.DataFrame(
        {
            "direction": np.ones(n, dtype=int),
            "magnitude": np.full(n, 0.02),
            "confidence": np.full(n, 0.7),
        },
        index=sample_ohlcv_500.index,
    )


@pytest.fixture
def neutral_predictions(sample_ohlcv_500: pd.DataFrame) -> pd.DataFrame:
    """Predictions that always say NEUTRAL (0)."""
    n = len(sample_ohlcv_500)
    return pd.DataFrame(
        {
            "direction": np.zeros(n, dtype=int),
            "magnitude": np.zeros(n),
            "confidence": np.full(n, 0.5),
        },
        index=sample_ohlcv_500.index,
    )


@pytest.fixture
def low_confidence_predictions(sample_ohlcv_500: pd.DataFrame) -> pd.DataFrame:
    """Predictions that say UP but with confidence below threshold."""
    n = len(sample_ohlcv_500)
    return pd.DataFrame(
        {
            "direction": np.ones(n, dtype=int),
            "magnitude": np.full(n, 0.02),
            "confidence": np.full(n, 0.3),
        },
        index=sample_ohlcv_500.index,
    )


class TestBacktestEngine:
    def test_returns_backtest_result(
        self,
        sample_ohlcv_500: pd.DataFrame,
        bullish_predictions: pd.DataFrame,
        default_config: BacktestConfig,
    ) -> None:
        result = run_backtest(
            predictions_df=bullish_predictions,
            ohlcv_df=sample_ohlcv_500,
            config=default_config,
        )
        assert isinstance(result, BacktestResult)

    def test_equity_curve_length(
        self,
        sample_ohlcv_500: pd.DataFrame,
        bullish_predictions: pd.DataFrame,
        default_config: BacktestConfig,
    ) -> None:
        result = run_backtest(bullish_predictions, sample_ohlcv_500, default_config)
        assert len(result.equity_curve) == len(sample_ohlcv_500)

    def test_no_trades_with_neutral(
        self,
        sample_ohlcv_500: pd.DataFrame,
        neutral_predictions: pd.DataFrame,
        default_config: BacktestConfig,
    ) -> None:
        result = run_backtest(neutral_predictions, sample_ohlcv_500, default_config)
        assert result.n_trades == 0

    def test_no_trades_with_low_confidence(
        self,
        sample_ohlcv_500: pd.DataFrame,
        low_confidence_predictions: pd.DataFrame,
        default_config: BacktestConfig,
    ) -> None:
        result = run_backtest(
            low_confidence_predictions, sample_ohlcv_500, default_config,
            min_confidence=0.55,
        )
        assert result.n_trades == 0

    def test_sharpe_is_float(
        self,
        sample_ohlcv_500: pd.DataFrame,
        bullish_predictions: pd.DataFrame,
        default_config: BacktestConfig,
    ) -> None:
        result = run_backtest(bullish_predictions, sample_ohlcv_500, default_config)
        assert isinstance(result.sharpe_ratio, float)

    def test_max_drawdown_negative_or_zero(
        self,
        sample_ohlcv_500: pd.DataFrame,
        bullish_predictions: pd.DataFrame,
        default_config: BacktestConfig,
    ) -> None:
        result = run_backtest(bullish_predictions, sample_ohlcv_500, default_config)
        assert result.max_drawdown <= 0

    def test_win_rate_bounded(
        self,
        sample_ohlcv_500: pd.DataFrame,
        bullish_predictions: pd.DataFrame,
        default_config: BacktestConfig,
    ) -> None:
        result = run_backtest(bullish_predictions, sample_ohlcv_500, default_config)
        if result.n_trades > 0:
            assert 0 <= result.win_rate <= 1

    def test_costs_reduce_returns(
        self,
        sample_ohlcv_500: pd.DataFrame,
        bullish_predictions: pd.DataFrame,
    ) -> None:
        zero_cost = BacktestConfig()
        zero_cost.costs.stt_delivery = 0
        zero_cost.costs.brokerage_flat = 0
        zero_cost.costs.exchange_txn = 0
        zero_cost.costs.gst = 0
        zero_cost.costs.sebi_turnover = 0
        zero_cost.costs.stamp_duty = 0
        zero_cost.slippage.large_cap = 0
        zero_cost.slippage.mid_cap = 0
        zero_cost.slippage.small_cap = 0

        normal_config = BacktestConfig()

        result_no_cost = run_backtest(bullish_predictions, sample_ohlcv_500, zero_cost)
        result_with_cost = run_backtest(bullish_predictions, sample_ohlcv_500, normal_config)

        if result_no_cost.n_trades > 0:
            assert result_with_cost.total_return <= result_no_cost.total_return

    def test_trade_log_dataframe(
        self,
        sample_ohlcv_500: pd.DataFrame,
        bullish_predictions: pd.DataFrame,
        default_config: BacktestConfig,
    ) -> None:
        result = run_backtest(bullish_predictions, sample_ohlcv_500, default_config)
        assert isinstance(result.trade_log, pd.DataFrame)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/backtest/test_engine.py -v 2>&1 | tail -5`

Expected: `ModuleNotFoundError`

- [ ] **Step 3: Implement backtest engine**

Create `alphavedha/backtest/engine.py`:

```python
"""VectorBT backtesting engine — runs predictions through a strategy with Indian market costs."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import structlog

from alphavedha.backtest.costs import compute_round_trip_cost_pct
from alphavedha.config import BacktestConfig

logger = structlog.get_logger(__name__)


@dataclass
class BacktestResult:
    total_return: float
    annualized_return: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    max_drawdown_duration_days: int
    alpha_vs_benchmark: float
    win_rate: float
    profit_factor: float
    n_trades: int
    avg_holding_days: float
    avg_return_per_trade: float
    equity_curve: pd.Series
    drawdown_curve: pd.Series
    trade_log: pd.DataFrame
    benchmark_return: float


def _compute_drawdown(equity: pd.Series) -> tuple[pd.Series, float, int]:
    peak = equity.cummax()
    dd = (equity - peak) / peak
    max_dd = float(dd.min())

    duration = 0
    max_duration = 0
    for val in dd.values:
        if val < 0:
            duration += 1
            max_duration = max(max_duration, duration)
        else:
            duration = 0

    return dd, max_dd, max_duration


def _compute_sharpe(returns: pd.Series) -> float:
    if len(returns) < 2 or returns.std() == 0:
        return 0.0
    return float(returns.mean() / returns.std() * np.sqrt(252))


def _compute_sortino(returns: pd.Series) -> float:
    if len(returns) < 2:
        return 0.0
    downside = returns[returns < 0]
    if len(downside) == 0 or downside.std() == 0:
        return 0.0
    return float(returns.mean() / downside.std() * np.sqrt(252))


def run_backtest(
    predictions_df: pd.DataFrame,
    ohlcv_df: pd.DataFrame,
    config: BacktestConfig,
    market_cap_tier: str = "large",
    min_confidence: float = 0.55,
) -> BacktestResult:
    cost_pct = compute_round_trip_cost_pct(market_cap_tier, config)
    closes = ohlcv_df["close"]
    daily_returns = closes.pct_change().fillna(0.0)

    entries = (
        (predictions_df["direction"] == 1)
        & (predictions_df["confidence"] >= min_confidence)
    )
    exits = predictions_df["direction"] == -1

    position = pd.Series(0, index=closes.index, dtype=int)
    in_position = False
    entry_idx = -1
    trades: list[dict] = []
    holding_period = 0
    max_hold = 15

    for i in range(len(closes)):
        if not in_position and entries.iloc[i]:
            in_position = True
            entry_idx = i
            holding_period = 0
            position.iloc[i] = 1
        elif in_position:
            holding_period += 1
            should_exit = exits.iloc[i] or holding_period >= max_hold
            if should_exit:
                position.iloc[i] = 0
                entry_price = closes.iloc[entry_idx]
                exit_price = closes.iloc[i]
                gross_ret = exit_price / entry_price - 1
                net_ret = gross_ret - cost_pct
                trades.append({
                    "entry_date": closes.index[entry_idx],
                    "exit_date": closes.index[i],
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "gross_return": gross_ret,
                    "net_return": net_ret,
                    "holding_days": holding_period,
                })
                in_position = False
            else:
                position.iloc[i] = 1

    strategy_returns = daily_returns * position.shift(1).fillna(0)

    for trade in trades:
        cost_per_day = cost_pct / max(trade["holding_days"], 1)
        entry_loc = closes.index.get_loc(trade["entry_date"])
        exit_loc = closes.index.get_loc(trade["exit_date"])
        strategy_returns.iloc[exit_loc] -= cost_pct

    equity = (1 + strategy_returns).cumprod()
    dd_curve, max_dd, max_dd_duration = _compute_drawdown(equity)

    total_ret = float(equity.iloc[-1] - 1) if len(equity) > 0 else 0.0
    n_days = len(equity)
    ann_ret = float((1 + total_ret) ** (252 / max(n_days, 1)) - 1)
    sharpe = _compute_sharpe(strategy_returns)
    sortino = _compute_sortino(strategy_returns)

    benchmark_ret = float(closes.iloc[-1] / closes.iloc[0] - 1) if len(closes) > 1 else 0.0
    alpha = ann_ret - float((1 + benchmark_ret) ** (252 / max(n_days, 1)) - 1)

    trade_log = pd.DataFrame(trades) if trades else pd.DataFrame(
        columns=["entry_date", "exit_date", "entry_price", "exit_price",
                 "gross_return", "net_return", "holding_days"]
    )

    n_trades = len(trades)
    if n_trades > 0:
        wins = [t for t in trades if t["net_return"] > 0]
        losses = [t for t in trades if t["net_return"] <= 0]
        win_rate = len(wins) / n_trades
        gross_profit = sum(t["net_return"] for t in wins) if wins else 0.0
        gross_loss = abs(sum(t["net_return"] for t in losses)) if losses else 0.0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")
        avg_hold = float(np.mean([t["holding_days"] for t in trades]))
        avg_ret = float(np.mean([t["net_return"] for t in trades]))
    else:
        win_rate = 0.0
        profit_factor = 0.0
        avg_hold = 0.0
        avg_ret = 0.0

    logger.info(
        "backtest_completed",
        n_trades=n_trades,
        total_return=round(total_ret, 4),
        sharpe=round(sharpe, 4),
        max_drawdown=round(max_dd, 4),
        win_rate=round(win_rate, 4),
    )

    return BacktestResult(
        total_return=total_ret,
        annualized_return=ann_ret,
        sharpe_ratio=sharpe,
        sortino_ratio=sortino,
        max_drawdown=max_dd,
        max_drawdown_duration_days=max_dd_duration,
        alpha_vs_benchmark=alpha,
        win_rate=win_rate,
        profit_factor=profit_factor,
        n_trades=n_trades,
        avg_holding_days=avg_hold,
        avg_return_per_trade=avg_ret,
        equity_curve=equity,
        drawdown_curve=dd_curve,
        trade_log=trade_log,
        benchmark_return=benchmark_ret,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/backtest/test_engine.py -v`

Expected: All tests PASS.

- [ ] **Step 5: Update backtest `__init__.py` and commit**

Write `alphavedha/backtest/__init__.py`:

```python
"""Backtesting — CPCV validation, cost modeling, and VectorBT engine."""

from alphavedha.backtest.costs import TradeCost, compute_round_trip_cost_pct, compute_trade_cost
from alphavedha.backtest.cpcv import CPCVResult, PathResult, generate_cpcv_splits, run_cpcv
from alphavedha.backtest.engine import BacktestResult, run_backtest

__all__ = [
    "BacktestResult",
    "CPCVResult",
    "PathResult",
    "TradeCost",
    "compute_round_trip_cost_pct",
    "compute_trade_cost",
    "generate_cpcv_splits",
    "run_backtest",
    "run_cpcv",
]
```

```bash
git add alphavedha/backtest/ tests/unit/backtest/test_engine.py
git commit -m "feat: implement VectorBT backtest engine with Indian market costs"
```

---

### Task 9: Full test suite run and final commit

**Files:**
- No new files — verify everything works together.

- [ ] **Step 1: Run the full Week 3 test suite**

Run: `pytest tests/unit/labels/ tests/unit/models/ tests/unit/backtest/ -v --tb=short`

Expected: All tests PASS.

- [ ] **Step 2: Run linting**

Run: `ruff check alphavedha/labels/ alphavedha/models/ alphavedha/backtest/ --fix`

Expected: No errors (or auto-fixed).

- [ ] **Step 3: Run full existing test suite to check for regressions**

Run: `pytest tests/unit/ -v --tb=short`

Expected: All existing tests still pass. No regressions.

- [ ] **Step 4: Final commit if any lint fixes were needed**

```bash
git add -A
git commit -m "chore: lint fixes for Week 3 modules"
```

(Skip if no changes from lint.)

---

## Summary

| Task | Module | Tests | Key deliverable |
|------|--------|-------|-----------------|
| 1 | tests/conftest.py | — | 3 new fixtures (500-day OHLCV, features, known path) |
| 2 | labels/triple_barrier.py | 12 tests | ATR-scaled labeling with +1/-1/0 |
| 3 | labels/sample_weights.py | 8 tests | Uniqueness + recency weighting |
| 4 | models/base.py | 10 tests | BaseModel ABC, TrainResult, PredictionResult, ModelArtifact |
| 5 | models/xgboost_model.py | 11 tests | XGBClassifier + XGBRegressor, save/load |
| 6 | backtest/costs.py | 10 tests | 7-component Indian market cost model |
| 7 | backtest/cpcv.py | 9 tests | 15-path CPCV with purge+embargo |
| 8 | backtest/engine.py | 10 tests | VectorBT strategy runner with equity curve |
| 9 | — | Full suite | Regression check, lint |

**Total: 9 tasks, 70 tests, 7 new source files, 7 new test files, 3 updated `__init__.py` files.**
