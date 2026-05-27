"""Save/load round-trip tests for all 8 model types.

Verifies: train → predict → save → load → predict produces identical results.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def _make_tabular_data(
    n: int = 100, n_features: int = 10
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
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
        X_train, X_val = X[:80], X[80:]
        y_train, y_val = y_dir[:80], y_dir[80:]
        ret_train, ret_val = y_ret[:80], y_ret[80:]

        model = XGBoostModel()
        model.fit(
            X_train,
            y_train,
            X_val=X_val,
            y_val=y_val,
            return_train=ret_train,
            return_val=ret_val,
        )

        pred_before = model.predict(X[:10])
        model.save(tmp_path / "xgb")
        loaded = XGBoostModel.load(tmp_path / "xgb")
        pred_after = loaded.predict(X[:10])

        np.testing.assert_array_equal(pred_before.direction, pred_after.direction)
        np.testing.assert_allclose(pred_before.confidence, pred_after.confidence)


class TestLSTMRoundTrip:
    def test_save_load_predictions_match(self, tmp_path: Path) -> None:
        from alphavedha.config import LSTMConfig
        from alphavedha.models.lstm_model import LSTMModel

        X, y_dir, y_ret = _make_tabular_data(n=120, n_features=10)
        config = LSTMConfig(
            sequence_length=10,
            hidden_size=16,
            num_layers=1,
            max_epochs=2,
            batch_size=16,
            top_n_features=10,
        )
        model = LSTMModel(config=config)
        model.fit(X, y_dir, return_train=y_ret)

        pred_before = model.predict(X[:20])
        model.save(tmp_path / "lstm")
        loaded = LSTMModel.load(tmp_path / "lstm")
        pred_after = loaded.predict(X[:20])

        np.testing.assert_array_equal(pred_before.direction, pred_after.direction)
        np.testing.assert_allclose(pred_before.confidence, pred_after.confidence, atol=1e-5)


class TestTFTRoundTrip:
    def test_save_load_predictions_match(self, tmp_path: Path) -> None:
        from alphavedha.config import TFTConfig
        from alphavedha.models.temporal_attention import TemporalAttentionModel

        X, y_dir, y_ret = _make_tabular_data(n=120, n_features=10)
        config = TFTConfig(
            sequence_length=10,
            hidden_size=16,
            attention_head_size=2,
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
        np.testing.assert_allclose(pred_before.confidence, pred_after.confidence, atol=1e-5)


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

        def _mock_pred(seed: int) -> PredictionResult:
            r = np.random.default_rng(seed)
            probs = r.dirichlet([1, 1, 1], size=n)
            direction = np.argmax(probs, axis=1).astype(int) - 1
            return PredictionResult(
                direction=direction,
                magnitude=r.uniform(0, 0.05, n),
                probabilities=probs,
                confidence=np.max(probs, axis=1),
            )

        base_oof = {
            "xgboost": _mock_pred(10),
            "lstm": _mock_pred(20),
            "tft": _mock_pred(30),
        }
        regime_probs = rng.dirichlet([1, 1, 1, 1], size=n)
        y_true = pd.Series(rng.choice([-1, 0, 1], size=n))

        model = StackingEnsemble()
        model.fit(base_oof, regime_probs, y_true)

        result_before = model.predict(base_oof, regime_probs)
        model.save(tmp_path / "ensemble")
        loaded = StackingEnsemble.load(tmp_path / "ensemble")
        result_after = loaded.predict(base_oof, regime_probs)

        np.testing.assert_array_equal(result_before.direction, result_after.direction)
        np.testing.assert_allclose(result_before.confidence, result_after.confidence, atol=1e-6)


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
            result_before.price_mid,
            result_after.price_mid,
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
        obs_t = torch.FloatTensor(obs).unsqueeze(0)

        agent._network.eval()
        with torch.no_grad():
            dist_before, value_before = agent._network(obs_t)
        mean_before = dist_before.mean.cpu().numpy()

        agent.save(tmp_path / "ppo")
        loaded = PPOAgent.load(tmp_path / "ppo")

        loaded._network.eval()
        with torch.no_grad():
            dist_after, value_after = loaded._network(obs_t)
        mean_after = dist_after.mean.cpu().numpy()

        np.testing.assert_allclose(mean_before, mean_after, atol=1e-5)
        np.testing.assert_allclose(value_before.cpu().numpy(), value_after.cpu().numpy(), atol=1e-5)
