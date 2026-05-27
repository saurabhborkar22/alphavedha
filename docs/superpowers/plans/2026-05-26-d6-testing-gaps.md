# D6: Testing Gaps Completion — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Complete integration tests (data pipeline, API, feature store, model round-trips), add pre-commit hooks, and add a coverage Makefile target.

**Architecture:** Docker-based test DB via `docker-compose.test.yml` (port 5433), session-scoped fixtures for DB lifecycle, function-scoped table truncation for isolation. Model round-trip tests use `tmp_path` only (no DB). Pre-commit hooks mirror CI checks.

**Tech Stack:** pytest, pytest-asyncio, SQLAlchemy async, Docker, pre-commit, ruff, mypy

---

## File Map

| Action | File | Purpose |
|--------|------|---------|
| Create | `docker-compose.test.yml` | Test PostgreSQL+TimescaleDB on port 5433 |
| Create | `tests/integration/conftest.py` | DB session fixtures, table truncation |
| Create | `tests/integration/data/test_pipeline_e2e.py` | OHLCV store → load round-trip |
| Create | `tests/integration/features/test_store_consistency.py` | Feature store → load round-trip |
| Create | `tests/integration/api/test_api_integration.py` | FastAPI endpoints with real DB |
| Create | `tests/unit/models/test_model_roundtrip.py` | Save/load for all 8 model types |
| Create | `.pre-commit-config.yaml` | Ruff lint + format + mypy hooks |
| Modify | `Makefile` | Add `coverage`, `test-integration-up`, `test-integration-down` targets |
| Modify | `docs/PROGRESS.md` | Mark D6 complete |

---

### Task 1: Docker Test Environment + Integration Conftest

**Files:**
- Create: `docker-compose.test.yml`
- Create: `tests/integration/conftest.py`

- [ ] **Step 1: Create docker-compose.test.yml**

```yaml
# docker-compose.test.yml
services:
  test-postgres:
    image: timescale/timescaledb:latest-pg16
    container_name: alphavedha-test-db
    environment:
      POSTGRES_USER: alphavedha
      POSTGRES_PASSWORD: testpass
      POSTGRES_DB: alphavedha_test
    ports:
      - "5433:5432"
    tmpfs:
      - /var/lib/postgresql/data
```

- [ ] **Step 2: Create integration conftest.py**

```python
# tests/integration/conftest.py
from __future__ import annotations

import asyncio
import os
import subprocess
from collections.abc import AsyncIterator, Iterator

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

TEST_DB_URL = os.environ.get(
    "TEST_DATABASE_URL",
    "postgresql+asyncpg://alphavedha:testpass@localhost:5433/alphavedha_test",
)

_TABLES_TO_TRUNCATE = [
    "daily_ohlcv",
    "corporate_actions",
    "index_constituents",
    "institutional_flows",
    "derivatives_data",
    "earnings_results",
    "promoter_holdings",
    "insider_trades",
    "news_articles",
    "paper_trades",
    "daily_pnl",
    "alternative_data",
    "features",
]


def _db_is_reachable() -> bool:
    """Check if the test DB container is accepting connections."""
    import socket

    try:
        sock = socket.create_connection(("localhost", 5433), timeout=2)
        sock.close()
        return True
    except OSError:
        return False


@pytest.fixture(scope="session")
def event_loop() -> Iterator[asyncio.AbstractEventLoop]:
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session")
def test_engine(event_loop: asyncio.AbstractEventLoop) -> AsyncEngine:
    if not _db_is_reachable():
        pytest.skip("Test database not available on port 5433")
    return create_async_engine(TEST_DB_URL, pool_size=5, max_overflow=0)


@pytest.fixture(scope="session", autouse=True)
def _create_schema(
    test_engine: AsyncEngine, event_loop: asyncio.AbstractEventLoop
) -> Iterator[None]:
    """Create all tables once per session via ORM metadata."""
    from alphavedha.data.models import Base

    async def _setup() -> None:
        async with test_engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    event_loop.run_until_complete(_setup())
    yield

    async def _teardown() -> None:
        async with test_engine.begin() as conn:
            await conn.run_sync(Base.metadata.drop_all)
        await test_engine.dispose()

    event_loop.run_until_complete(_teardown())


@pytest.fixture()
def session_factory(test_engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    return async_sessionmaker(
        test_engine, class_=AsyncSession, expire_on_commit=False
    )


@pytest.fixture(autouse=True)
def _truncate_tables(
    test_engine: AsyncEngine, event_loop: asyncio.AbstractEventLoop
) -> Iterator[None]:
    """Truncate all data tables before each test for isolation."""
    yield

    async def _truncate() -> None:
        async with test_engine.begin() as conn:
            for table in _TABLES_TO_TRUNCATE:
                await conn.execute(text(f"TRUNCATE TABLE {table} CASCADE"))

    event_loop.run_until_complete(_truncate())
```

- [ ] **Step 3: Verify conftest loads without errors**

Run: `docker compose -f docker-compose.test.yml up -d && sleep 3 && .venv/bin/python -m pytest tests/integration/ --co -q`
Expected: Shows collected 0 items (no tests yet, but no import errors)

- [ ] **Step 4: Add Makefile targets**

Add these targets to `Makefile` after the existing `test-backtest` target:

```makefile
test-integration-up:
	docker compose -f docker-compose.test.yml up -d
	@echo "Waiting for test database..." && sleep 3

test-integration-down:
	docker compose -f docker-compose.test.yml down

coverage:
	$(VENV)/bin/pytest tests/unit/ --cov=alphavedha --cov-report=term-missing --cov-report=html
	@echo "HTML report: htmlcov/index.html"
```

Also update the `.PHONY` line at the top of `Makefile` to include `test-integration-up test-integration-down coverage`.

- [ ] **Step 5: Commit**

```bash
git add docker-compose.test.yml tests/integration/conftest.py Makefile
PRE_COMMIT_ALLOW_NO_CONFIG=1 git commit -m "feat(d6): add Docker test environment and integration conftest"
```

---

### Task 2: Data Pipeline End-to-End Tests

**Files:**
- Create: `tests/integration/data/test_pipeline_e2e.py`

- [ ] **Step 1: Write integration tests for OHLCV store/load**

```python
# tests/integration/data/test_pipeline_e2e.py
from __future__ import annotations

from datetime import date
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from alphavedha.data.store import delete_ohlcv, load_ohlcv, store_ohlcv

pytestmark = pytest.mark.integration


def _make_ohlcv(symbol: str, n_days: int, start: str = "2024-01-02") -> pd.DataFrame:
    dates = pd.bdate_range(start, periods=n_days, freq="B")
    rng = np.random.default_rng(42)
    base = 100.0
    closes = base * np.cumprod(1 + rng.normal(0.001, 0.015, n_days))
    return pd.DataFrame(
        {
            "open": closes * (1 + rng.normal(0, 0.005, n_days)),
            "high": closes * (1 + np.abs(rng.normal(0, 0.01, n_days))),
            "low": closes * (1 - np.abs(rng.normal(0, 0.01, n_days))),
            "close": closes,
            "adj_close": closes,
            "volume": rng.integers(1_000_000, 10_000_000, size=n_days),
        },
        index=dates,
    )


def _patch_session_factory(session_factory):
    """Patch get_session_factory to use the test DB session."""
    return patch(
        "alphavedha.data.store.get_session_factory", return_value=session_factory
    )


class TestOHLCVStoreLoad:
    @pytest.mark.asyncio()
    async def test_insert_and_query_ohlcv(self, session_factory) -> None:
        df = _make_ohlcv("TCS.NS", 20)
        with _patch_session_factory(session_factory):
            stored = await store_ohlcv("TCS.NS", df)
            assert stored == 20

            loaded = await load_ohlcv(
                "TCS.NS", date(2024, 1, 1), date(2024, 12, 31)
            )
        assert len(loaded) == 20
        np.testing.assert_allclose(
            loaded["close"].values, df["close"].values, rtol=1e-6
        )

    @pytest.mark.asyncio()
    async def test_upsert_idempotent(self, session_factory) -> None:
        df = _make_ohlcv("INFY.NS", 10)
        with _patch_session_factory(session_factory):
            await store_ohlcv("INFY.NS", df)
            await store_ohlcv("INFY.NS", df)
            loaded = await load_ohlcv(
                "INFY.NS", date(2024, 1, 1), date(2024, 12, 31)
            )
        assert len(loaded) == 10

    @pytest.mark.asyncio()
    async def test_date_range_filtering(self, session_factory) -> None:
        df = _make_ohlcv("RELIANCE.NS", 100)
        with _patch_session_factory(session_factory):
            await store_ohlcv("RELIANCE.NS", df)
            loaded = await load_ohlcv(
                "RELIANCE.NS", date(2024, 1, 1), date(2024, 1, 31)
            )
        assert len(loaded) < 100
        assert all(d.date() <= date(2024, 1, 31) for d in loaded.index)

    @pytest.mark.asyncio()
    async def test_multiple_symbols_no_cross_contamination(
        self, session_factory
    ) -> None:
        df_tcs = _make_ohlcv("TCS.NS", 15)
        df_infy = _make_ohlcv("INFY.NS", 10)
        with _patch_session_factory(session_factory):
            await store_ohlcv("TCS.NS", df_tcs)
            await store_ohlcv("INFY.NS", df_infy)
            loaded_tcs = await load_ohlcv(
                "TCS.NS", date(2024, 1, 1), date(2024, 12, 31)
            )
            loaded_infy = await load_ohlcv(
                "INFY.NS", date(2024, 1, 1), date(2024, 12, 31)
            )
        assert len(loaded_tcs) == 15
        assert len(loaded_infy) == 10

    @pytest.mark.asyncio()
    async def test_delete_ohlcv(self, session_factory) -> None:
        df = _make_ohlcv("HDFC.NS", 5)
        with _patch_session_factory(session_factory):
            await store_ohlcv("HDFC.NS", df)
            deleted = await delete_ohlcv("HDFC.NS")
            assert deleted == 5
            loaded = await load_ohlcv(
                "HDFC.NS", date(2024, 1, 1), date(2024, 12, 31)
            )
        assert len(loaded) == 0
```

- [ ] **Step 2: Run tests against Docker DB**

Run: `make test-integration-up && .venv/bin/python -m pytest tests/integration/data/test_pipeline_e2e.py -v`
Expected: All 5 tests PASS

- [ ] **Step 3: Commit**

```bash
git add tests/integration/data/test_pipeline_e2e.py
PRE_COMMIT_ALLOW_NO_CONFIG=1 git commit -m "test(d6): add data pipeline end-to-end integration tests"
```

---

### Task 3: Feature Store Consistency Tests

**Files:**
- Create: `tests/integration/features/test_store_consistency.py`

- [ ] **Step 1: Write feature store round-trip tests**

```python
# tests/integration/features/test_store_consistency.py
from __future__ import annotations

from datetime import date
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from alphavedha.data.store import load_features, store_features

pytestmark = pytest.mark.integration


def _make_features(n_days: int, n_features: int) -> pd.DataFrame:
    dates = pd.bdate_range("2024-01-02", periods=n_days, freq="B")
    rng = np.random.default_rng(42)
    data = rng.standard_normal((n_days, n_features))
    cols = [f"feat_{i}" for i in range(n_features)]
    return pd.DataFrame(data, index=dates, columns=cols)


def _patch_session_factory(session_factory):
    return patch(
        "alphavedha.data.store.get_session_factory", return_value=session_factory
    )


class TestFeatureStoreConsistency:
    @pytest.mark.asyncio()
    async def test_save_and_load_features(self, session_factory) -> None:
        df = _make_features(10, 5)
        with _patch_session_factory(session_factory):
            stored = await store_features("TCS.NS", df, feature_version="v1")
            assert stored == 10

            loaded = await load_features(
                "TCS.NS", date(2024, 1, 1), date(2024, 12, 31), feature_version="v1"
            )

        assert len(loaded) == 10
        for col in df.columns:
            np.testing.assert_allclose(
                loaded[col].values, df[col].values, rtol=1e-6
            )

    @pytest.mark.asyncio()
    async def test_feature_versioning(self, session_factory) -> None:
        df_v1 = _make_features(5, 3)
        rng = np.random.default_rng(99)
        df_v2 = pd.DataFrame(
            rng.standard_normal((5, 3)),
            index=df_v1.index,
            columns=df_v1.columns,
        )

        with _patch_session_factory(session_factory):
            await store_features("INFY.NS", df_v1, feature_version="v1")
            await store_features("INFY.NS", df_v2, feature_version="v2")

            loaded_v1 = await load_features(
                "INFY.NS", date(2024, 1, 1), date(2024, 12, 31), feature_version="v1"
            )
            loaded_v2 = await load_features(
                "INFY.NS", date(2024, 1, 1), date(2024, 12, 31), feature_version="v2"
            )

        assert len(loaded_v1) == 5
        assert len(loaded_v2) == 5
        assert not np.allclose(loaded_v1.values, loaded_v2.values)

    @pytest.mark.asyncio()
    async def test_load_features_date_range(self, session_factory) -> None:
        df = _make_features(30, 3)
        with _patch_session_factory(session_factory):
            await store_features("TCS.NS", df, feature_version="v1")
            loaded = await load_features(
                "TCS.NS", date(2024, 1, 1), date(2024, 1, 15), feature_version="v1"
            )

        assert len(loaded) < 30
        assert all(d.date() <= date(2024, 1, 15) for d in loaded.index)
```

- [ ] **Step 2: Run tests**

Run: `.venv/bin/python -m pytest tests/integration/features/test_store_consistency.py -v`
Expected: All 3 tests PASS

- [ ] **Step 3: Commit**

```bash
git add tests/integration/features/test_store_consistency.py
PRE_COMMIT_ALLOW_NO_CONFIG=1 git commit -m "test(d6): add feature store consistency integration tests"
```

---

### Task 4: API Integration Tests

**Files:**
- Create: `tests/integration/api/test_api_integration.py`

- [ ] **Step 1: Write API integration tests**

```python
# tests/integration/api/test_api_integration.py
from __future__ import annotations

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from alphavedha.api.app import create_app

pytestmark = pytest.mark.integration


@pytest.fixture()
def client() -> Iterator[TestClient]:
    """TestClient with demo=True — exercises real service wiring without model artifacts."""
    app = create_app(demo=True)
    with TestClient(app) as c:
        yield c


class TestHealthWithDB:
    def test_health_returns_ok(self, client: TestClient) -> None:
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"

    def test_ready_reports_db_status(self, client: TestClient) -> None:
        resp = client.get("/ready")
        assert resp.status_code == 200
        data = resp.json()
        assert "database_available" in data
        assert "models_loaded" in data
        assert data["models_loaded"] is True


class TestPredictEndpoints:
    def test_predict_returns_valid_response(self, client: TestClient) -> None:
        resp = client.get("/predict/TCS.NS")
        assert resp.status_code == 200
        data = resp.json()
        assert "symbol" in data
        assert "direction" in data
        assert "confidence" in data
        assert "generated_at" in data
        assert "model_version" in data

    def test_predict_batch(self, client: TestClient) -> None:
        resp = client.post(
            "/predict/batch",
            json={"symbols": ["TCS.NS", "INFY.NS", "RELIANCE.NS"]},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "predictions" in data
        assert len(data["predictions"]) == 3

    def test_predict_invalid_symbol_format(self, client: TestClient) -> None:
        resp = client.get("/predict/INVALID!!!")
        assert resp.status_code in (400, 422)

    def test_scan_tier(self, client: TestClient) -> None:
        resp = client.get("/scan/large?top_n=3")
        assert resp.status_code == 200
        data = resp.json()
        assert "predictions" in data

    def test_scan_invalid_tier(self, client: TestClient) -> None:
        resp = client.get("/scan/nonexistent")
        assert resp.status_code == 400
```

- [ ] **Step 2: Run tests**

Run: `.venv/bin/python -m pytest tests/integration/api/test_api_integration.py -v`
Expected: All 7 tests PASS

- [ ] **Step 3: Commit**

```bash
git add tests/integration/api/test_api_integration.py
PRE_COMMIT_ALLOW_NO_CONFIG=1 git commit -m "test(d6): add API integration tests with real service wiring"
```

---

### Task 5: Model Save/Load Round-Trip Tests

**Files:**
- Create: `tests/unit/models/test_model_roundtrip.py`

- [ ] **Step 1: Write round-trip tests for all 8 model types**

```python
# tests/unit/models/test_model_roundtrip.py
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def _make_tabular_data(
    n: int = 100, n_features: int = 10
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Synthetic tabular data: features, direction labels, returns."""
    rng = np.random.default_rng(42)
    X = pd.DataFrame(
        rng.standard_normal((n, n_features)),
        columns=[f"f{i}" for i in range(n_features)],
    )
    y_direction = pd.Series(rng.choice([-1, 0, 1], size=n), name="label")
    y_return = pd.Series(rng.normal(0, 0.02, n), name="return")
    return X, y_direction, y_return


class TestXGBoostRoundTrip:
    def test_save_load_predictions_match(self, tmp_path: Path) -> None:
        from alphavedha.models.xgboost_model import XGBoostModel

        X, y_dir, y_ret = _make_tabular_data()
        model = XGBoostModel()
        model.fit(X, y_dir, return_train=y_ret)

        pred_before = model.predict(X[:10])
        model.save(tmp_path / "xgb")
        loaded = XGBoostModel.load(tmp_path / "xgb")
        pred_after = loaded.predict(X[:10])

        np.testing.assert_array_equal(pred_before.direction, pred_after.direction)
        np.testing.assert_allclose(pred_before.confidence, pred_after.confidence)


class TestLSTMRoundTrip:
    def test_save_load_predictions_match(self, tmp_path: Path) -> None:
        from alphavedha.models.lstm_model import LSTMConfig, LSTMModel

        X, y_dir, y_ret = _make_tabular_data(n=120, n_features=10)
        config = LSTMConfig(
            sequence_length=10,
            hidden_size=16,
            num_layers=1,
            max_epochs=2,
            batch_size=16,
        )
        model = LSTMModel(config=config)
        model.fit(X, y_dir, return_train=y_ret)

        pred_before = model.predict(X[:20])
        model.save(tmp_path / "lstm")
        loaded = LSTMModel.load(tmp_path / "lstm")
        pred_after = loaded.predict(X[:20])

        np.testing.assert_array_equal(pred_before.direction, pred_after.direction)
        np.testing.assert_allclose(
            pred_before.confidence, pred_after.confidence, atol=1e-5
        )


class TestTFTRoundTrip:
    def test_save_load_predictions_match(self, tmp_path: Path) -> None:
        from alphavedha.models.temporal_attention import (
            TFTConfig,
            TemporalAttentionModel,
        )

        X, y_dir, y_ret = _make_tabular_data(n=120, n_features=10)
        config = TFTConfig(
            sequence_length=10,
            d_model=16,
            n_heads=2,
            max_epochs=2,
            batch_size=16,
        )
        model = TemporalAttentionModel(config=config)
        model.fit(X, y_dir, return_train=y_ret)

        pred_before = model.predict(X[:20])
        model.save(tmp_path / "tft")
        loaded = TemporalAttentionModel.load(tmp_path / "tft")
        pred_after = loaded.predict(X[:20])

        np.testing.assert_array_equal(pred_before.direction, pred_after.direction)
        np.testing.assert_allclose(
            pred_before.confidence, pred_after.confidence, atol=1e-5
        )


class TestRegimeRoundTrip:
    def test_save_load_predictions_match(self, tmp_path: Path) -> None:
        from alphavedha.models.regime import RegimeDetector

        rng = np.random.default_rng(42)
        returns = pd.Series(rng.normal(0.001, 0.02, 200))
        volatility = pd.Series(np.abs(rng.normal(0.02, 0.005, 200)))

        model = RegimeDetector()
        model.fit(returns, volatility)

        result_before = model.predict(returns[-50:], volatility[-50:])
        model.save(tmp_path / "regime")
        loaded = RegimeDetector.load(tmp_path / "regime")
        result_after = loaded.predict(returns[-50:], volatility[-50:])

        assert result_before.current_regime == result_after.current_regime
        np.testing.assert_allclose(
            result_before.state_probabilities,
            result_after.state_probabilities,
            atol=1e-6,
        )


class TestEnsembleRoundTrip:
    def test_save_load_predictions_match(self, tmp_path: Path) -> None:
        from alphavedha.models.base import PredictionResult
        from alphavedha.models.ensemble import StackingEnsemble

        rng = np.random.default_rng(42)
        n = 100

        def _mock_pred(n: int) -> PredictionResult:
            probs = rng.dirichlet([1, 1, 1], size=n)
            direction = np.argmax(probs, axis=1) - 1
            return PredictionResult(
                direction=direction,
                magnitude=rng.uniform(0, 0.05, n),
                probabilities=probs,
                confidence=np.max(probs, axis=1),
            )

        base_oof = {
            "xgboost": _mock_pred(n),
            "lstm": _mock_pred(n),
            "tft": _mock_pred(n),
        }
        regime_probs = rng.dirichlet([1, 1, 1, 1], size=n)
        y_true = pd.Series(rng.choice([-1, 0, 1], size=n))

        model = StackingEnsemble()
        model.fit(base_oof, regime_probs, y_true)

        result_before = model.predict(base_oof, regime_probs)
        model.save(tmp_path / "ensemble")
        loaded = StackingEnsemble.load(tmp_path / "ensemble")
        result_after = loaded.predict(base_oof, regime_probs)

        np.testing.assert_array_equal(
            result_before.direction, result_after.direction
        )
        np.testing.assert_allclose(
            result_before.confidence, result_after.confidence, atol=1e-6
        )


class TestMetaLabelingRoundTrip:
    def test_save_load_predictions_match(self, tmp_path: Path) -> None:
        from alphavedha.models.meta_model import MetaLabelingModel

        rng = np.random.default_rng(42)
        n = 100
        X = pd.DataFrame(rng.standard_normal((n, 10)), columns=[f"f{i}" for i in range(10)])
        ens_dir = rng.choice([-1, 0, 1], size=n).astype(float)
        ens_conf = rng.uniform(0.3, 0.9, n)
        y_correct = pd.Series(rng.choice([0, 1], size=n))

        model = MetaLabelingModel()
        model.fit(X, ens_dir, ens_conf, y_correct)

        result_before = model.predict(X[:20], ens_dir[:20], ens_conf[:20])
        model.save(tmp_path / "meta")
        loaded = MetaLabelingModel.load(tmp_path / "meta")
        result_after = loaded.predict(X[:20], ens_dir[:20], ens_conf[:20])

        np.testing.assert_allclose(
            result_before.meta_confidence,
            result_after.meta_confidence,
            atol=1e-6,
        )


class TestConformalRoundTrip:
    def test_save_load_predictions_match(self, tmp_path: Path) -> None:
        from alphavedha.models.conformal import ConformalPredictor

        rng = np.random.default_rng(42)
        n = 100
        X = pd.DataFrame(rng.standard_normal((n, 5)), columns=[f"f{i}" for i in range(5)])
        y = pd.Series(rng.normal(0, 0.02, n))

        model = ConformalPredictor()
        model.fit(X, y)

        result_before = model.predict(X[:10])
        model.save(tmp_path / "conformal")
        loaded = ConformalPredictor.load(tmp_path / "conformal")
        result_after = loaded.predict(X[:10])

        np.testing.assert_allclose(
            result_before.point_estimates,
            result_after.point_estimates,
            atol=1e-5,
        )


class TestPPORoundTrip:
    def test_save_load_action_match(self, tmp_path: Path) -> None:
        import torch

        from alphavedha.models.rl_agent import PPOAgent, PPOConfig

        torch.manual_seed(42)
        config = PPOConfig(hidden_size=32)
        agent = PPOAgent(obs_size=10, action_size=3, config=config)

        obs = np.random.default_rng(42).standard_normal(10).astype(np.float32)
        action_before, _, _ = agent.select_action(obs)

        agent.save(tmp_path / "ppo")
        loaded = PPOAgent.load(tmp_path / "ppo")
        action_after, _, _ = loaded.select_action(obs)

        np.testing.assert_allclose(action_before, action_after, atol=1e-5)
```

- [ ] **Step 2: Run round-trip tests**

Run: `.venv/bin/python -m pytest tests/unit/models/test_model_roundtrip.py -v`
Expected: All 8 tests PASS

- [ ] **Step 3: Commit**

```bash
git add tests/unit/models/test_model_roundtrip.py
PRE_COMMIT_ALLOW_NO_CONFIG=1 git commit -m "test(d6): add model save/load round-trip tests for all 8 model types"
```

---

### Task 6: Pre-commit Hooks

**Files:**
- Create: `.pre-commit-config.yaml`

- [ ] **Step 1: Create .pre-commit-config.yaml**

```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.15.12
    hooks:
      - id: ruff
        args: [check, --fix]
      - id: ruff-format

  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.15.0
    hooks:
      - id: mypy
        args: [--ignore-missing-imports]
        additional_dependencies:
          - pydantic>=2.0
          - types-redis
        pass_filenames: false
        entry: mypy alphavedha/
```

- [ ] **Step 2: Install pre-commit hooks**

Run: `.venv/bin/pre-commit install`
Expected: "pre-commit installed at .git/hooks/pre-commit"

- [ ] **Step 3: Verify hooks run**

Run: `.venv/bin/pre-commit run --all-files`
Expected: All hooks pass (ruff check, ruff format, mypy)

- [ ] **Step 4: Commit**

```bash
git add .pre-commit-config.yaml
PRE_COMMIT_ALLOW_NO_CONFIG=1 git commit -m "chore(d6): add pre-commit hooks for ruff lint, format, and mypy"
```

---

### Task 7: Update PROGRESS.md

**Files:**
- Modify: `docs/PROGRESS.md`

- [ ] **Step 1: Update D6 section in PROGRESS.md**

Change the D6 section to mark all items complete. Update test counts. Mark the summary table row for D6 as COMPLETE.

Key changes:
- D6.2 integration tests: all 4 sub-items checked
- D6.3 quality: coverage report, pre-commit hooks, pre-existing failure all checked
- Update total test count (run `pytest tests/ --co -q` to get the number)
- Summary table: `D6: Testing Gaps | COMPLETE | 100%`

- [ ] **Step 2: Commit**

```bash
git add docs/PROGRESS.md
PRE_COMMIT_ALLOW_NO_CONFIG=1 git commit -m "docs: mark D6 testing gaps as complete"
```

---

## Post-Implementation

After all 7 tasks are complete:

1. Run full unit test suite: `.venv/bin/pytest tests/unit/ -v --tb=short`
2. Run integration tests: `make test-integration-up && .venv/bin/pytest tests/integration/ -v -m integration && make test-integration-down`
3. Run lint + format + typecheck: `make lint`
4. Run coverage: `make coverage` (verify ≥80%)
5. Verify pre-commit hooks: `.venv/bin/pre-commit run --all-files`
