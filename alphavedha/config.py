"""Configuration loader — reads configs/default.yaml and validates with Pydantic."""

from __future__ import annotations

import functools
from pathlib import Path
from typing import Any, Self

import yaml
from pydantic import BaseModel, Field, model_validator

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


class ProjectConfig(BaseModel):
    name: str = "alphavedha"
    version: str = "0.1.0"
    timezone: str = "Asia/Kolkata"


class UniverseTierConfig(BaseModel):
    index: str
    count: int


class UniverseConfig(BaseModel):
    default_tiers: list[str] = Field(default_factory=lambda: ["large", "mid"])
    available_tiers: dict[str, UniverseTierConfig] = Field(default_factory=dict)
    rebalance_check: str = "quarterly"


class ProviderListConfig(BaseModel):
    primary: str = "jugaad"
    fallback: str = "yfinance"
    derivatives: str = "nse"
    sentiment: str = "finnhub"


class RateLimitConfig(BaseModel):
    requests_per_second: float | None = None
    requests_per_minute: float | None = None
    backoff_factor: float = 2.0
    user_agent_rotate: bool = False


class DataHistoryConfig(BaseModel):
    backfill_start: str = "2005-01-01"
    min_history_days: int = 252


class DataConfig(BaseModel):
    providers: ProviderListConfig = Field(default_factory=ProviderListConfig)
    rate_limits: dict[str, RateLimitConfig] = Field(default_factory=dict)
    history: DataHistoryConfig = Field(default_factory=DataHistoryConfig)


class FractionalDiffConfig(BaseModel):
    min_d: float = 0.1
    max_d: float = 0.8
    adf_pvalue_threshold: float = 0.05
    max_lags: int = 100
    recompute_interval: str = "monthly"


class OutlierConfig(BaseModel):
    winsorize_lower: float = 0.01
    winsorize_upper: float = 0.99


class CircuitConfig(BaseModel):
    thresholds: list[float] = Field(default_factory=lambda: [0.05, 0.10, 0.20])
    flag_column: str = "circuit_hit"


class MissingDataConfig(BaseModel):
    method: str = "forward_fill"
    add_flag: bool = True
    max_gap_days: int = 10


class PreprocessingConfig(BaseModel):
    fractional_diff: FractionalDiffConfig = Field(default_factory=FractionalDiffConfig)
    outlier: OutlierConfig = Field(default_factory=OutlierConfig)
    circuit: CircuitConfig = Field(default_factory=CircuitConfig)
    missing_data: MissingDataConfig = Field(default_factory=MissingDataConfig)


class TripleBarrierConfig(BaseModel):
    multiplier_up: float = 2.0
    multiplier_down: float = 1.5
    max_holding_period: int = 15
    min_atr_threshold: float = 0.005
    atr_period: int = 14


class MetaLabelingConfig(BaseModel):
    min_confidence: float = 0.55
    model: str = "xgboost"


class SampleWeightsConfig(BaseModel):
    uniqueness: bool = True
    recency_halflife: int = 252


class LabelsConfig(BaseModel):
    triple_barrier: TripleBarrierConfig = Field(default_factory=TripleBarrierConfig)
    meta_labeling: MetaLabelingConfig = Field(default_factory=MetaLabelingConfig)
    sample_weights: SampleWeightsConfig = Field(default_factory=SampleWeightsConfig)


class XGBoostParams(BaseModel):
    learning_rate: float = 0.05
    max_depth: int = 6
    n_estimators: int = 500
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    reg_alpha: float = 0.1
    reg_lambda: float = 1.0
    eval_metric: str = "logloss"
    early_stopping_rounds: int = 50


class XGBoostConfig(BaseModel):
    task: str = "classification"
    params: XGBoostParams = Field(default_factory=XGBoostParams)


class LSTMConfig(BaseModel):
    sequence_length: int = 60
    hidden_size: int = 128
    num_layers: int = 2
    dropout: float = 0.3
    learning_rate: float = 0.001
    batch_size: int = 64
    max_epochs: int = 100
    early_stopping_patience: int = 10
    top_n_features: int = 30


class TFTConfig(BaseModel):
    sequence_length: int = 60
    hidden_size: int = 64
    attention_head_size: int = 4
    num_layers: int = 2
    dropout: float = 0.1
    learning_rate: float = 0.001
    batch_size: int = 64
    max_epochs: int = 50
    early_stopping_patience: int = 10
    horizons: list[int] = Field(default_factory=lambda: [7, 15, 30])


class RegimeConfig(BaseModel):
    n_states: int = 4
    state_names: list[str] = Field(
        default_factory=lambda: ["bull", "bear", "sideways", "high_volatility"]
    )
    n_iter: int = 100
    covariance_type: str = "full"
    retrain_interval: str = "monthly"

    @model_validator(mode="after")
    def _validate_state_names_count(self) -> Self:
        if len(self.state_names) != self.n_states:
            raise ValueError(
                f"n_states ({self.n_states}) != len(state_names) ({len(self.state_names)})"
            )
        return self


class EnsembleConfig(BaseModel):
    meta_learner: str = "ridge"
    alpha: float = 1.0


class ConformalConfig(BaseModel):
    coverage: float = 0.90
    calibration_window: int = 60
    method: str = "plus"


class ModelsConfig(BaseModel):
    artifact_dir: str = "models/artifacts"
    xgboost: XGBoostConfig = Field(default_factory=XGBoostConfig)
    lstm: LSTMConfig = Field(default_factory=LSTMConfig)
    tft: TFTConfig = Field(default_factory=TFTConfig)
    regime: RegimeConfig = Field(default_factory=RegimeConfig)
    ensemble: EnsembleConfig = Field(default_factory=EnsembleConfig)
    conformal: ConformalConfig = Field(default_factory=ConformalConfig)


class CPCVConfig(BaseModel):
    n_segments: int = 6
    k_test_segments: int = 2
    purge_days: int = 20
    embargo_days: int = 20


class AcceptanceConfig(BaseModel):
    min_median_sharpe: float = 0.8
    min_worst_sharpe: float = 0.3


class ValidationConfig(BaseModel):
    cpcv: CPCVConfig = Field(default_factory=CPCVConfig)
    acceptance: AcceptanceConfig = Field(default_factory=AcceptanceConfig)


class PositionSizingConfig(BaseModel):
    method: str = "half_kelly"
    max_single_stock_pct: float = 10.0
    min_confidence: float = 0.55


class PortfolioConfig(BaseModel):
    max_sector_pct: float = 25.0
    max_correlation: float = 0.7
    min_holding_days: int = 3
    min_daily_turnover_cr: float = 5.0


class CircuitBreakerConfig(BaseModel):
    level_1_drawdown: float = 10.0
    level_2_drawdown: float = 15.0
    level_3_drawdown: float = 20.0
    recovery_threshold: float = 0.95


class RiskConfig(BaseModel):
    position_sizing: PositionSizingConfig = Field(default_factory=PositionSizingConfig)
    portfolio: PortfolioConfig = Field(default_factory=PortfolioConfig)
    circuit_breaker: CircuitBreakerConfig = Field(default_factory=CircuitBreakerConfig)


class ApiConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8000
    rate_limit: dict[str, int] = Field(
        default_factory=lambda: {"default_per_minute": 100, "batch_per_minute": 10}
    )
    cache: dict[str, int] = Field(
        default_factory=lambda: {"market_hours_ttl": 300, "after_hours_ttl": 43200}
    )


class CompositeScoreWeights(BaseModel):
    technical_momentum: float = 0.25
    derivatives_sentiment: float = 0.20
    macro_alignment: float = 0.15
    microstructure_quality: float = 0.15
    news_sentiment: float = 0.10
    volatility_risk: float = 0.15


class DriftConfig(BaseModel):
    psi_warning: float = 0.1
    psi_alert: float = 0.2


class PerformanceMonitorConfig(BaseModel):
    min_accuracy: float = 0.52
    rolling_windows: list[int] = Field(default_factory=lambda: [7, 30, 90])


class RetrainingConfig(BaseModel):
    schedule: str = "weekly"
    shadow_period_days: int = 20
    keep_versions: int = 5


class MonitoringConfig(BaseModel):
    drift: DriftConfig = Field(default_factory=DriftConfig)
    performance: PerformanceMonitorConfig = Field(default_factory=PerformanceMonitorConfig)
    retraining: RetrainingConfig = Field(default_factory=RetrainingConfig)


class CostsConfig(BaseModel):
    stt_delivery: float = 0.001
    stt_intraday: float = 0.00025
    brokerage_flat: float = 20.0
    exchange_txn: float = 0.0000345
    gst: float = 0.18
    sebi_turnover: float = 0.000001
    stamp_duty: float = 0.00015


class SlippageConfig(BaseModel):
    large_cap: float = 0.001
    mid_cap: float = 0.003
    small_cap: float = 0.005


class BacktestConfig(BaseModel):
    costs: CostsConfig = Field(default_factory=CostsConfig)
    slippage: SlippageConfig = Field(default_factory=SlippageConfig)
    benchmark: str = "^NSEI"


class AppConfig(BaseModel):
    project: ProjectConfig = Field(default_factory=ProjectConfig)
    universe: UniverseConfig = Field(default_factory=UniverseConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    preprocessing: PreprocessingConfig = Field(default_factory=PreprocessingConfig)
    labels: LabelsConfig = Field(default_factory=LabelsConfig)
    models: ModelsConfig = Field(default_factory=ModelsConfig)
    validation: ValidationConfig = Field(default_factory=ValidationConfig)
    risk: RiskConfig = Field(default_factory=RiskConfig)
    api: ApiConfig = Field(default_factory=ApiConfig)
    monitoring: MonitoringConfig = Field(default_factory=MonitoringConfig)
    backtest: BacktestConfig = Field(default_factory=BacktestConfig)


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open() as f:
        return yaml.safe_load(f)  # type: ignore[no-any-return]


@functools.lru_cache(maxsize=1)
def get_config(config_path: Path | None = None) -> AppConfig:
    """Load and return the validated application config (cached singleton)."""
    if config_path is None:
        config_path = _PROJECT_ROOT / "configs" / "default.yaml"
    raw = _load_yaml(config_path)
    return AppConfig.model_validate(raw)
