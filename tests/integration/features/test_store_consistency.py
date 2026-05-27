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


def _patch_sf(session_factory):
    return patch("alphavedha.data.store.get_session_factory", return_value=session_factory)


class TestFeatureStoreConsistency:
    @pytest.mark.asyncio()
    async def test_save_and_load_features(self, session_factory) -> None:
        df = _make_features(10, 5)
        with _patch_sf(session_factory):
            stored = await store_features("TCS.NS", df, feature_version="v1")
            assert stored == 10

            loaded = await load_features(
                "TCS.NS", date(2024, 1, 1), date(2024, 12, 31), feature_version="v1"
            )

        assert len(loaded) == 10
        for col in df.columns:
            np.testing.assert_allclose(loaded[col].values, df[col].values, rtol=1e-6)

    @pytest.mark.asyncio()
    async def test_feature_versioning(self, session_factory) -> None:
        df_v1 = _make_features(5, 3)
        rng = np.random.default_rng(99)
        df_v2 = pd.DataFrame(
            rng.standard_normal((5, 3)),
            index=df_v1.index,
            columns=df_v1.columns,
        )

        with _patch_sf(session_factory):
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
        with _patch_sf(session_factory):
            await store_features("TCS.NS", df, feature_version="v1")
            loaded = await load_features(
                "TCS.NS", date(2024, 1, 1), date(2024, 1, 15), feature_version="v1"
            )

        assert len(loaded) < 30
        assert all(d.date() <= date(2024, 1, 15) for d in loaded.index)
