"""ModelRegistry — loads real or demo models into a PredictionEngine."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import structlog

from alphavedha.config import get_config
from alphavedha.exceptions import ModelNotFoundError
from alphavedha.models.base import PredictionResult
from alphavedha.models.conformal import ConformalPredictor, ConformalResult
from alphavedha.models.ensemble import EnsembleResult, StackingEnsemble
from alphavedha.models.gnn_model import GNNModel
from alphavedha.models.lstm_model import LSTMModel
from alphavedha.models.meta_model import MetaLabelingModel, MetaLabelResult
from alphavedha.models.regime import RegimeDetector, RegimeResult
from alphavedha.models.temporal_attention import TemporalAttentionModel
from alphavedha.models.xgboost_model import XGBoostModel
from alphavedha.prediction.engine import PredictionEngine
from alphavedha.prediction.scorer import CompositeScorer
from alphavedha.risk.risk_manager import RiskManager

logger = structlog.get_logger(__name__)

_DEMO_SYMBOLS: list[str] = [
    "TCS",
    "INFY",
    "RELIANCE",
    "HDFCBANK",
    "ICICIBANK",
    "BHARTIARTL",
    "ITC",
    "SBIN",
    "HINDUNILVR",
    "LT",
    "KOTAKBANK",
    "WIPRO",
    "AXISBANK",
    "BAJFINANCE",
    "MARUTI",
]

_DEMO_FEATURE_NAMES: list[str] = [
    "close",
    "volume",
    "rsi_14",
    "macd_12_26",
    "bb_width_20",
    "atr_14",
    "natr_14",
    "hvol_20",
    "obv",
    "vwap",
]


def _symbol_seed(symbol: str) -> int:
    """Deterministic seed from symbol name using MD5 hash."""
    h = hashlib.md5(symbol.encode()).hexdigest()
    return int(h[:8], 16)


class _DemoBaseModel:
    """Mock base model returning deterministic PredictionResult."""

    def __init__(self, name: str, seed_offset: int = 0) -> None:
        self._name = name
        self._seed_offset = seed_offset

    def predict(self, X: pd.DataFrame) -> PredictionResult:
        n = X.shape[0]
        # Use first row's values to derive a deterministic seed
        row_hash = hashlib.md5(X.iloc[0].values.tobytes() + self._name.encode()).hexdigest()
        seed = int(row_hash[:8], 16) + self._seed_offset
        rng = np.random.default_rng(seed)

        direction_val = rng.choice([-1, 0, 1])
        direction = np.full(n, direction_val, dtype=int)
        magnitude = np.full(n, rng.uniform(0.01, 0.05))

        # Generate 3-class probabilities that sum to 1
        raw = rng.dirichlet([1.0, 1.0, 1.0])
        probabilities = np.tile(raw, (n, 1))

        confidence = np.full(n, rng.uniform(0.5, 0.9))

        return PredictionResult(
            direction=direction,
            magnitude=magnitude,
            probabilities=probabilities,
            confidence=confidence,
        )


class _DemoRegime:
    """Mock regime detector returning deterministic RegimeResult."""

    def predict(self, returns: pd.Series, volatility: pd.Series) -> RegimeResult:
        return RegimeResult(
            current_regime="bull",
            regime_id=0,
            state_probabilities=np.array([0.6, 0.1, 0.2, 0.1]),
            regime_history=np.array([0]),
            transition_matrix=np.eye(4),
        )


class _DemoEnsemble:
    """Mock stacking ensemble averaging base predictions into EnsembleResult."""

    def predict(
        self,
        base_predictions: dict[str, PredictionResult],
        regime_probs: np.ndarray,
    ) -> EnsembleResult:
        directions = []
        magnitudes = []
        confidences = []
        all_probs = []

        for pred in base_predictions.values():
            directions.append(pred.direction)
            magnitudes.append(pred.magnitude)
            confidences.append(pred.confidence)
            if pred.probabilities is not None:
                all_probs.append(pred.probabilities)

        n = directions[0].shape[0]

        # Average probabilities across models
        if all_probs:
            stacked = np.stack(all_probs, axis=0)
            avg_probs = stacked.mean(axis=0)
        else:
            avg_probs = np.tile([1 / 3, 1 / 3, 1 / 3], (n, 1))

        # Direction from argmax of averaged probabilities, mapped to {-1, 0, 1}
        direction_map = {0: -1, 1: 0, 2: 1}
        direction = np.array([direction_map[int(np.argmax(avg_probs[i]))] for i in range(n)])

        # Weighted average magnitude
        conf_stack = np.stack(confidences, axis=0)
        mag_stack = np.stack(magnitudes, axis=0)
        conf_sum = conf_stack.sum(axis=0, keepdims=True)
        conf_sum = np.where(conf_sum == 0, 1.0, conf_sum)
        weights = conf_stack / conf_sum
        magnitude = (weights * mag_stack).sum(axis=0)

        confidence = np.max(avg_probs, axis=1)

        # Disagreement: std of each model's prob for consensus class
        if all_probs:
            consensus = np.argmax(avg_probs, axis=1)
            probs_for_consensus = np.stack(all_probs, axis=0)[:, np.arange(n), consensus]
            disagreement = np.std(probs_for_consensus, axis=0)
        else:
            disagreement = np.zeros(n)

        return EnsembleResult(
            direction=direction,
            magnitude=magnitude,
            probabilities=avg_probs,
            confidence=confidence,
            model_disagreement=disagreement,
        )


class _DemoMeta:
    """Mock meta-labeling model returning high confidence for buy/sell, low for hold."""

    def predict(
        self,
        X_features: pd.DataFrame,
        ensemble_direction: np.ndarray,
        ensemble_confidence: np.ndarray,
    ) -> MetaLabelResult:
        n = len(X_features)
        meta_confidence = np.empty(n)
        for i in range(n):
            if ensemble_direction[i] != 0:
                meta_confidence[i] = 0.72
            else:
                meta_confidence[i] = 0.45

        return MetaLabelResult(
            meta_confidence=meta_confidence,
            is_tradeable=meta_confidence > 0.55,
        )


class _DemoConformal:
    """Mock conformal predictor returning fixed price targets around 100."""

    def predict(self, X: pd.DataFrame) -> ConformalResult:
        n = X.shape[0]
        return ConformalResult(
            price_low=np.full(n, 95.0),
            price_mid=np.full(n, 100.0),
            price_high=np.full(n, 105.0),
            interval_width=np.full(n, 10.0),
            coverage=0.90,
        )


class ModelRegistry:
    """Loads real trained models or builds demo mock models into a PredictionEngine.

    Args:
        demo: If True, use synthetic mock models instead of real artifacts.
        artifact_dir: Override path to model artifacts directory.
    """

    def __init__(self, demo: bool = False, artifact_dir: Path | None = None) -> None:
        self._demo = demo
        self._real_version = "v0.1.0"
        if artifact_dir is not None:
            self._artifact_dir = artifact_dir
        else:
            config = get_config()
            self._artifact_dir = Path(config.models.artifact_dir)

    @property
    def is_demo(self) -> bool:
        """Whether the registry is operating in demo mode."""
        return self._demo

    @property
    def model_version(self) -> str:
        """Return the model version string."""
        if self._demo:
            return "demo-v0.1.0"
        return self._real_version

    def get_prediction_engine(self) -> PredictionEngine:
        """Build and return a PredictionEngine with loaded models.

        In demo mode, returns an engine with synthetic mock models.
        In real mode, attempts to load trained model artifacts from disk.

        Raises:
            ModelNotFoundError: When real mode is selected but artifacts are missing.
        """
        if self._demo:
            return self._build_demo_engine()
        return self._build_real_engine()

    def get_demo_symbols(self) -> list[str]:
        """Return list of 15 hardcoded NIFTY symbols for demo mode."""
        return list(_DEMO_SYMBOLS)

    def get_demo_features(self, symbol: str) -> pd.DataFrame:
        """Generate deterministic synthetic features for a demo symbol.

        Uses MD5 hash of symbol name as RNG seed for reproducibility.
        """
        seed = _symbol_seed(symbol)
        rng = np.random.default_rng(seed)

        data: dict[str, list[float]] = {}
        for name in _DEMO_FEATURE_NAMES:
            data[name] = [float(rng.standard_normal())]

        return pd.DataFrame(data)

    def _build_demo_engine(self) -> PredictionEngine:
        """Construct a PredictionEngine with mock models for demo mode."""
        config = get_config()
        risk_manager = RiskManager(
            position_config=config.risk.position_sizing,
            portfolio_config=config.risk.portfolio,
            circuit_breaker_config=config.risk.circuit_breaker,
        )

        return PredictionEngine(
            xgboost=_DemoBaseModel("xgboost", seed_offset=0),
            lstm=_DemoBaseModel("lstm", seed_offset=1),
            tft=_DemoBaseModel("tft", seed_offset=2),
            regime=_DemoRegime(),  # type: ignore[arg-type]
            ensemble=_DemoEnsemble(),  # type: ignore[arg-type]
            meta_model=_DemoMeta(),  # type: ignore[arg-type]
            conformal=_DemoConformal(),  # type: ignore[arg-type]
            scorer=CompositeScorer(),
            risk_manager=risk_manager,
            model_version=self.model_version,
        )

    def _build_real_engine(self) -> PredictionEngine:
        """Load trained model artifacts from disk and build a PredictionEngine.

        Raises:
            ModelNotFoundError: When required model artifacts are not found.
        """
        artifact_dir = self._artifact_dir
        if not artifact_dir.exists():
            raise ModelNotFoundError(
                f"Model artifact directory not found: {artifact_dir}. "
                "Train models first with `make train`, or use demo mode."
            )

        # Training pipeline saves each model under {name}/latest/ — resolve that
        # first, falling back to {name}/ for artifacts laid out flat.
        def _resolve(name: str) -> Path:
            latest = artifact_dir / name / "latest"
            return latest if latest.exists() else artifact_dir / name

        required = ["xgboost", "lstm", "tft", "regime", "ensemble", "meta_labeling", "conformal"]
        missing = [name for name in required if not (_resolve(name) / "metadata.json").exists()]
        if missing:
            raise ModelNotFoundError(
                f"Missing model artifacts: {', '.join(missing)}. "
                "Train models first with `make train`, or use demo mode."
            )

        logger.info("loading_real_models", artifact_dir=str(artifact_dir))

        xgboost = XGBoostModel.load(_resolve("xgboost"))
        lstm = LSTMModel.load(_resolve("lstm"))
        tft = TemporalAttentionModel.load(_resolve("tft"))
        regime = RegimeDetector.load(_resolve("regime"))
        ensemble = StackingEnsemble.load(_resolve("ensemble"))
        meta_model = MetaLabelingModel.load(_resolve("meta_labeling"))
        conformal = ConformalPredictor.load(_resolve("conformal"))

        # GNN is optional — load if trained artifact exists
        gnn: GNNModel | None = None
        gnn_dir = _resolve("gnn")
        if (gnn_dir / "metadata.json").exists():
            try:
                gnn = GNNModel.load(gnn_dir)
                logger.info("gnn_loaded", path=str(gnn_dir))
            except Exception as e:
                logger.warning("gnn_load_failed", error=str(e))

        version_file = artifact_dir / "version.json"
        if version_file.exists():
            self._real_version = json.loads(version_file.read_text()).get("version", "v0.1.0")
        else:
            self._real_version = "v0.1.0"

        config = get_config()
        risk_manager = RiskManager(
            position_config=config.risk.position_sizing,
            portfolio_config=config.risk.portfolio,
            circuit_breaker_config=config.risk.circuit_breaker,
        )

        logger.info("real_models_loaded", model_version=self._real_version, has_gnn=gnn is not None)

        return PredictionEngine(
            xgboost=xgboost,
            lstm=lstm,
            tft=tft,
            gnn=gnn,
            regime=regime,
            ensemble=ensemble,
            meta_model=meta_model,
            conformal=conformal,
            scorer=CompositeScorer(),
            risk_manager=risk_manager,
            model_version=self._real_version,
            # Trained ConformalPredictor predicts forward returns (fit on
            # ret_oof) — convert intervals to prices at serving time.
            conformal_outputs_returns=True,
        )
