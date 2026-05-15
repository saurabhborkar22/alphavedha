# Week 7: Prediction Engine + Risk Management — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wire all 7 ML models into a single `predict()` pipeline with risk-adjusted position sizing, composite scoring, and stock ranking.

**Architecture:** Two new modules — `prediction/` (engine, scorer, ranker) and `risk/` (position sizing, portfolio constraints, circuit breaker, risk manager). All models are injected as constructor args (dependency injection). The engine calls risk internally so consumers get one `predict()` → `StockPrediction`.

**Tech Stack:** Python 3.12, numpy, pandas, structlog, Pydantic v2 (all existing — no new deps)

---

## File Structure

| File | Responsibility |
|------|---------------|
| `alphavedha/risk/position_sizing.py` | Half-Kelly position sizing function |
| `alphavedha/risk/portfolio.py` | PortfolioState, HoldingInfo, PortfolioConstraints, ConstraintResult |
| `alphavedha/risk/circuit_breaker.py` | CircuitBreaker, CircuitBreakerState |
| `alphavedha/risk/risk_manager.py` | RiskManager orchestrating all risk checks, RiskAssessment |
| `alphavedha/prediction/scorer.py` | CompositeScorer — 0-100 weighted scoring |
| `alphavedha/prediction/ranker.py` | StockRanker — filter and rank candidates |
| `alphavedha/prediction/engine.py` | PredictionEngine — main pipeline orchestrator, StockPrediction |
| `alphavedha/config.py` | Add CompositeScoreWeights (line ~236, after ApiConfig) |
| `alphavedha/risk/__init__.py` | Exports for risk module |
| `alphavedha/prediction/__init__.py` | Exports for prediction module |

---

### Task 1: Position Sizing (Half-Kelly)

**Files:**
- Create: `alphavedha/risk/position_sizing.py`
- Test: `tests/unit/risk/test_position_sizing.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/risk/__init__.py` (empty) and `tests/unit/risk/test_position_sizing.py`:

```python
"""Tests for Half-Kelly position sizing."""

from __future__ import annotations

import pytest

from alphavedha.config import PositionSizingConfig
from alphavedha.risk.position_sizing import compute_position_size


@pytest.fixture
def config() -> PositionSizingConfig:
    return PositionSizingConfig(
        method="half_kelly",
        max_single_stock_pct=10.0,
        min_confidence=0.55,
    )


class TestComputePositionSize:
    def test_valid_confidence_returns_positive(self, config: PositionSizingConfig) -> None:
        result = compute_position_size(
            meta_confidence=0.70,
            magnitude=0.05,
            config=config,
        )
        assert result > 0.0
        assert result <= config.max_single_stock_pct

    def test_half_kelly_is_half_of_full(self, config: PositionSizingConfig) -> None:
        meta_confidence = 0.70
        magnitude = 0.05
        # Symmetric Kelly: f = (2p - 1) where avg_win == avg_loss
        full_kelly = 2 * meta_confidence - 1
        half_kelly_expected = full_kelly * 0.5 * 100  # as percentage
        result = compute_position_size(meta_confidence, magnitude, config)
        assert abs(result - min(half_kelly_expected, config.max_single_stock_pct)) < 1e-10

    def test_below_min_confidence_returns_zero(self, config: PositionSizingConfig) -> None:
        result = compute_position_size(
            meta_confidence=0.50,
            magnitude=0.05,
            config=config,
        )
        assert result == 0.0

    def test_zero_magnitude_returns_zero(self, config: PositionSizingConfig) -> None:
        result = compute_position_size(
            meta_confidence=0.70,
            magnitude=0.0,
            config=config,
        )
        assert result == 0.0

    def test_negative_magnitude_returns_zero(self, config: PositionSizingConfig) -> None:
        result = compute_position_size(
            meta_confidence=0.70,
            magnitude=-0.02,
            config=config,
        )
        assert result == 0.0

    def test_caps_at_max_single_stock(self, config: PositionSizingConfig) -> None:
        result = compute_position_size(
            meta_confidence=0.99,
            magnitude=0.10,
            config=config,
        )
        assert result == config.max_single_stock_pct

    def test_negative_kelly_returns_zero(self, config: PositionSizingConfig) -> None:
        result = compute_position_size(
            meta_confidence=0.40,
            magnitude=0.05,
            config=PositionSizingConfig(
                method="half_kelly",
                max_single_stock_pct=10.0,
                min_confidence=0.30,
            ),
        )
        assert result == 0.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/unit/risk/test_position_sizing.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'alphavedha.risk.position_sizing'`

- [ ] **Step 3: Implement position sizing**

Create `alphavedha/risk/position_sizing.py`:

```python
"""Half-Kelly position sizing — compute optimal position % from meta-confidence and magnitude."""

from __future__ import annotations

import structlog

from alphavedha.config import PositionSizingConfig

logger = structlog.get_logger(__name__)


def compute_position_size(
    meta_confidence: float,
    magnitude: float,
    config: PositionSizingConfig,
) -> float:
    if meta_confidence < config.min_confidence:
        return 0.0

    if magnitude <= 0.0:
        return 0.0

    # Symmetric Kelly: avg_win == avg_loss == magnitude
    # kelly = (win_prob * avg_win - loss_prob * avg_loss) / avg_win
    #       = (p * m - (1-p) * m) / m = 2p - 1
    kelly_fraction = 2 * meta_confidence - 1

    if kelly_fraction <= 0.0:
        return 0.0

    half_kelly_pct = kelly_fraction * 0.5 * 100  # convert to percentage

    position_pct = min(half_kelly_pct, config.max_single_stock_pct)

    logger.debug(
        "position_size_computed",
        meta_confidence=round(meta_confidence, 4),
        magnitude=round(magnitude, 6),
        kelly_fraction=round(kelly_fraction, 4),
        position_pct=round(position_pct, 4),
    )

    return position_pct
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/unit/risk/test_position_sizing.py -v`
Expected: 7 PASSED

- [ ] **Step 5: Lint and type-check**

Run: `ruff check alphavedha/risk/position_sizing.py tests/unit/risk/test_position_sizing.py && ruff format alphavedha/risk/position_sizing.py tests/unit/risk/test_position_sizing.py && python -m mypy alphavedha/risk/position_sizing.py`

- [ ] **Step 6: Commit**

```bash
git add alphavedha/risk/position_sizing.py tests/unit/risk/__init__.py tests/unit/risk/test_position_sizing.py
git commit -m "feat: add Half-Kelly position sizing with 7 tests"
```

---

### Task 2: Portfolio Constraints

**Files:**
- Create: `alphavedha/risk/portfolio.py`
- Test: `tests/unit/risk/test_portfolio.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/risk/test_portfolio.py`:

```python
"""Tests for portfolio-level constraints."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from alphavedha.config import PortfolioConfig
from alphavedha.risk.portfolio import (
    ConstraintResult,
    HoldingInfo,
    PortfolioConstraints,
    PortfolioState,
)


@pytest.fixture
def config() -> PortfolioConfig:
    return PortfolioConfig(
        max_sector_pct=25.0,
        max_correlation=0.7,
        min_holding_days=3,
        min_daily_turnover_cr=5.0,
    )


@pytest.fixture
def empty_portfolio() -> PortfolioState:
    return PortfolioState(holdings={}, total_value=1_000_000.0, peak_value=1_000_000.0)


def _make_holding(
    symbol: str,
    sector: str,
    weight_pct: float,
    days_held: int = 10,
    corr: dict[str, float] | None = None,
    turnover: float = 50.0,
) -> HoldingInfo:
    return HoldingInfo(
        symbol=symbol,
        sector=sector,
        weight_pct=weight_pct,
        entry_date=datetime.now(UTC) - timedelta(days=days_held),
        correlation_60d=corr or {},
        avg_daily_turnover_cr=turnover,
    )


class TestPortfolioConstraints:
    def test_within_all_limits_passes(
        self, config: PortfolioConfig, empty_portfolio: PortfolioState
    ) -> None:
        constraints = PortfolioConstraints(config)
        result = constraints.check(
            symbol="TCS",
            proposed_weight_pct=5.0,
            sector="IT",
            portfolio=empty_portfolio,
        )
        assert isinstance(result, ConstraintResult)
        assert result.passed is True
        assert result.adjusted_weight_pct == 5.0
        assert len(result.violations) == 0

    def test_exceeds_sector_cap_reduced(self, config: PortfolioConfig) -> None:
        portfolio = PortfolioState(
            holdings={
                "INFY": _make_holding("INFY", "IT", 20.0),
            },
            total_value=1_000_000.0,
            peak_value=1_000_000.0,
        )
        constraints = PortfolioConstraints(config)
        result = constraints.check(
            symbol="TCS",
            proposed_weight_pct=8.0,
            sector="IT",
            portfolio=portfolio,
        )
        assert result.adjusted_weight_pct == 5.0  # 25% cap - 20% existing = 5%
        assert any("sector" in v.lower() for v in result.violations)

    def test_high_correlation_rejected(self, config: PortfolioConfig) -> None:
        portfolio = PortfolioState(
            holdings={
                "INFY": _make_holding("INFY", "IT", 5.0, corr={"TCS": 0.85}),
            },
            total_value=1_000_000.0,
            peak_value=1_000_000.0,
        )
        constraints = PortfolioConstraints(config)
        result = constraints.check(
            symbol="TCS",
            proposed_weight_pct=5.0,
            sector="IT",
            portfolio=portfolio,
        )
        assert result.adjusted_weight_pct == 0.0
        assert result.passed is False
        assert any("correlation" in v.lower() for v in result.violations)

    def test_sell_before_min_holding_rejected(self, config: PortfolioConfig) -> None:
        portfolio = PortfolioState(
            holdings={
                "TCS": _make_holding("TCS", "IT", 5.0, days_held=1),
            },
            total_value=1_000_000.0,
            peak_value=1_000_000.0,
        )
        constraints = PortfolioConstraints(config)
        result = constraints.check(
            symbol="TCS",
            proposed_weight_pct=-5.0,  # negative = sell/reduce
            sector="IT",
            portfolio=portfolio,
        )
        assert result.adjusted_weight_pct == 0.0
        assert result.passed is False
        assert any("holding period" in v.lower() for v in result.violations)

    def test_low_liquidity_rejected(self, config: PortfolioConfig) -> None:
        portfolio = PortfolioState(
            holdings={},
            total_value=1_000_000.0,
            peak_value=1_000_000.0,
        )
        constraints = PortfolioConstraints(config)
        holding_info = _make_holding("SMALLCAP", "Misc", 0.0, turnover=2.0)
        result = constraints.check(
            symbol="SMALLCAP",
            proposed_weight_pct=5.0,
            sector="Misc",
            portfolio=portfolio,
            avg_daily_turnover_cr=2.0,
        )
        assert result.adjusted_weight_pct == 0.0
        assert result.passed is False
        assert any("liquidity" in v.lower() for v in result.violations)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/unit/risk/test_portfolio.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'alphavedha.risk.portfolio'`

- [ ] **Step 3: Implement portfolio constraints**

Create `alphavedha/risk/portfolio.py`:

```python
"""Portfolio constraints — sector caps, correlation limits, holding periods, liquidity."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime

import structlog

from alphavedha.config import PortfolioConfig

logger = structlog.get_logger(__name__)


@dataclass
class HoldingInfo:
    symbol: str
    sector: str
    weight_pct: float
    entry_date: datetime
    correlation_60d: dict[str, float] = field(default_factory=dict)
    avg_daily_turnover_cr: float = 50.0


@dataclass
class PortfolioState:
    holdings: dict[str, HoldingInfo]
    total_value: float
    peak_value: float


@dataclass
class ConstraintResult:
    adjusted_weight_pct: float
    violations: list[str]
    passed: bool


class PortfolioConstraints:

    def __init__(self, config: PortfolioConfig) -> None:
        self._config = config

    def check(
        self,
        symbol: str,
        proposed_weight_pct: float,
        sector: str,
        portfolio: PortfolioState,
        avg_daily_turnover_cr: float | None = None,
    ) -> ConstraintResult:
        violations: list[str] = []
        weight = proposed_weight_pct

        # Sell / reduce: check minimum holding period
        if weight < 0.0:
            holding = portfolio.holdings.get(symbol)
            if holding is not None:
                days_held = (datetime.now(UTC) - holding.entry_date).days
                if days_held < self._config.min_holding_days:
                    violations.append(
                        f"Holding period violation: {symbol} held {days_held}d, "
                        f"min {self._config.min_holding_days}d"
                    )
                    return ConstraintResult(
                        adjusted_weight_pct=0.0, violations=violations, passed=False
                    )
            return ConstraintResult(
                adjusted_weight_pct=weight, violations=violations, passed=True
            )

        # Buy / add: check liquidity
        turnover = avg_daily_turnover_cr
        if turnover is None:
            existing = portfolio.holdings.get(symbol)
            turnover = existing.avg_daily_turnover_cr if existing else None

        if turnover is not None and turnover < self._config.min_daily_turnover_cr:
            violations.append(
                f"Liquidity violation: {symbol} turnover {turnover:.1f} cr < "
                f"{self._config.min_daily_turnover_cr:.1f} cr min"
            )
            return ConstraintResult(
                adjusted_weight_pct=0.0, violations=violations, passed=False
            )

        # Check correlation with existing holdings
        for held_sym, held_info in portfolio.holdings.items():
            corr = held_info.correlation_60d.get(symbol, 0.0)
            if abs(corr) > self._config.max_correlation:
                violations.append(
                    f"Correlation violation: {symbol} corr with {held_sym} = "
                    f"{corr:.2f} > {self._config.max_correlation}"
                )
                return ConstraintResult(
                    adjusted_weight_pct=0.0, violations=violations, passed=False
                )

        # Check sector exposure cap
        current_sector_weight = sum(
            h.weight_pct
            for h in portfolio.holdings.values()
            if h.sector == sector and h.symbol != symbol
        )
        max_allowed = self._config.max_sector_pct - current_sector_weight
        if weight > max_allowed:
            violations.append(
                f"Sector cap: {sector} at {current_sector_weight:.1f}% + "
                f"{weight:.1f}% > {self._config.max_sector_pct}%, "
                f"reduced to {max_allowed:.1f}%"
            )
            weight = max(max_allowed, 0.0)

        logger.debug(
            "portfolio_constraint_check",
            symbol=symbol,
            proposed=proposed_weight_pct,
            adjusted=weight,
            violations=violations,
        )

        return ConstraintResult(
            adjusted_weight_pct=weight,
            violations=violations,
            passed=len(violations) == 0 or weight > 0.0,
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/unit/risk/test_portfolio.py -v`
Expected: 5 PASSED

- [ ] **Step 5: Lint and type-check**

Run: `ruff check alphavedha/risk/portfolio.py tests/unit/risk/test_portfolio.py && ruff format alphavedha/risk/portfolio.py tests/unit/risk/test_portfolio.py && python -m mypy alphavedha/risk/portfolio.py`

- [ ] **Step 6: Commit**

```bash
git add alphavedha/risk/portfolio.py tests/unit/risk/test_portfolio.py
git commit -m "feat: add portfolio constraints with sector, correlation, liquidity checks"
```

---

### Task 3: Circuit Breaker

**Files:**
- Create: `alphavedha/risk/circuit_breaker.py`
- Test: `tests/unit/risk/test_circuit_breaker.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/risk/test_circuit_breaker.py`:

```python
"""Tests for drawdown circuit breaker."""

from __future__ import annotations

import pytest

from alphavedha.config import CircuitBreakerConfig
from alphavedha.risk.circuit_breaker import CircuitBreaker, CircuitBreakerState


@pytest.fixture
def config() -> CircuitBreakerConfig:
    return CircuitBreakerConfig(
        level_1_drawdown=10.0,
        level_2_drawdown=15.0,
        level_3_drawdown=20.0,
        recovery_threshold=0.95,
    )


class TestCircuitBreakerEvaluate:
    def test_normal_no_drawdown(self, config: CircuitBreakerConfig) -> None:
        cb = CircuitBreaker(config)
        state = cb.evaluate(current_value=1_000_000.0, peak_value=1_000_000.0)
        assert isinstance(state, CircuitBreakerState)
        assert state.level == 0
        assert state.current_drawdown_pct == 0.0

    def test_level_1_at_10pct(self, config: CircuitBreakerConfig) -> None:
        cb = CircuitBreaker(config)
        state = cb.evaluate(current_value=890_000.0, peak_value=1_000_000.0)
        assert state.level == 1
        assert abs(state.current_drawdown_pct - 11.0) < 0.1

    def test_level_2_at_15pct(self, config: CircuitBreakerConfig) -> None:
        cb = CircuitBreaker(config)
        state = cb.evaluate(current_value=840_000.0, peak_value=1_000_000.0)
        assert state.level == 2
        assert abs(state.current_drawdown_pct - 16.0) < 0.1

    def test_level_3_at_20pct(self, config: CircuitBreakerConfig) -> None:
        cb = CircuitBreaker(config)
        state = cb.evaluate(current_value=790_000.0, peak_value=1_000_000.0)
        assert state.level == 3
        assert abs(state.current_drawdown_pct - 21.0) < 0.1

    def test_recovery_resets_to_normal(self, config: CircuitBreakerConfig) -> None:
        cb = CircuitBreaker(config)
        # 95% of peak = 950_000 → level 0
        state = cb.evaluate(current_value=960_000.0, peak_value=1_000_000.0)
        assert state.level == 0


class TestCircuitBreakerAdjust:
    def test_level_0_no_adjustment(self, config: CircuitBreakerConfig) -> None:
        cb = CircuitBreaker(config)
        state = CircuitBreakerState(level=0, current_drawdown_pct=0.0, peak_value=1e6, triggered_at=None)
        assert cb.adjust_position(5.0, state, is_new_entry=True) == 5.0

    def test_level_1_halves_position(self, config: CircuitBreakerConfig) -> None:
        cb = CircuitBreaker(config)
        state = CircuitBreakerState(level=1, current_drawdown_pct=11.0, peak_value=1e6, triggered_at=None)
        assert cb.adjust_position(6.0, state, is_new_entry=True) == 3.0

    def test_level_2_blocks_new_entry(self, config: CircuitBreakerConfig) -> None:
        cb = CircuitBreaker(config)
        state = CircuitBreakerState(level=2, current_drawdown_pct=16.0, peak_value=1e6, triggered_at=None)
        assert cb.adjust_position(5.0, state, is_new_entry=True) == 0.0

    def test_level_2_halves_existing(self, config: CircuitBreakerConfig) -> None:
        cb = CircuitBreaker(config)
        state = CircuitBreakerState(level=2, current_drawdown_pct=16.0, peak_value=1e6, triggered_at=None)
        assert cb.adjust_position(5.0, state, is_new_entry=False) == 2.5

    def test_level_3_zeroes_all(self, config: CircuitBreakerConfig) -> None:
        cb = CircuitBreaker(config)
        state = CircuitBreakerState(level=3, current_drawdown_pct=22.0, peak_value=1e6, triggered_at=None)
        assert cb.adjust_position(5.0, state, is_new_entry=False) == 0.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/unit/risk/test_circuit_breaker.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'alphavedha.risk.circuit_breaker'`

- [ ] **Step 3: Implement circuit breaker**

Create `alphavedha/risk/circuit_breaker.py`:

```python
"""Circuit breaker — drawdown protection with 3 escalation levels."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

import structlog

from alphavedha.config import CircuitBreakerConfig

logger = structlog.get_logger(__name__)


@dataclass
class CircuitBreakerState:
    level: int
    current_drawdown_pct: float
    peak_value: float
    triggered_at: datetime | None


class CircuitBreaker:

    def __init__(self, config: CircuitBreakerConfig) -> None:
        self._config = config

    def evaluate(
        self,
        current_value: float,
        peak_value: float,
    ) -> CircuitBreakerState:
        if peak_value <= 0:
            return CircuitBreakerState(
                level=0, current_drawdown_pct=0.0, peak_value=peak_value, triggered_at=None
            )

        drawdown_pct = (1 - current_value / peak_value) * 100

        # Recovery check: if within recovery threshold, normal
        recovery_value = peak_value * self._config.recovery_threshold
        if current_value >= recovery_value:
            return CircuitBreakerState(
                level=0,
                current_drawdown_pct=drawdown_pct,
                peak_value=peak_value,
                triggered_at=None,
            )

        now = datetime.now(UTC)

        if drawdown_pct >= self._config.level_3_drawdown:
            level = 3
        elif drawdown_pct >= self._config.level_2_drawdown:
            level = 2
        elif drawdown_pct >= self._config.level_1_drawdown:
            level = 1
        else:
            level = 0

        if level > 0:
            logger.warning(
                "circuit_breaker_triggered",
                level=level,
                drawdown_pct=round(drawdown_pct, 2),
                current_value=current_value,
                peak_value=peak_value,
            )

        return CircuitBreakerState(
            level=level,
            current_drawdown_pct=drawdown_pct,
            peak_value=peak_value,
            triggered_at=now if level > 0 else None,
        )

    def adjust_position(
        self,
        proposed_size_pct: float,
        state: CircuitBreakerState,
        is_new_entry: bool,
    ) -> float:
        if state.level == 0:
            return proposed_size_pct
        if state.level == 3:
            return 0.0
        if state.level == 2:
            if is_new_entry:
                return 0.0
            return proposed_size_pct * 0.5
        if state.level == 1:
            return proposed_size_pct * 0.5
        return proposed_size_pct
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/unit/risk/test_circuit_breaker.py -v`
Expected: 10 PASSED

- [ ] **Step 5: Lint and type-check**

Run: `ruff check alphavedha/risk/circuit_breaker.py tests/unit/risk/test_circuit_breaker.py && ruff format alphavedha/risk/circuit_breaker.py tests/unit/risk/test_circuit_breaker.py && python -m mypy alphavedha/risk/circuit_breaker.py`

- [ ] **Step 6: Commit**

```bash
git add alphavedha/risk/circuit_breaker.py tests/unit/risk/test_circuit_breaker.py
git commit -m "feat: add circuit breaker with 3 drawdown levels and position adjustment"
```

---

### Task 4: Risk Manager

**Files:**
- Create: `alphavedha/risk/risk_manager.py`
- Modify: `alphavedha/risk/__init__.py`
- Test: `tests/unit/risk/test_risk_manager.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/risk/test_risk_manager.py`:

```python
"""Tests for RiskManager — orchestrates position sizing, portfolio, and circuit breaker."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from alphavedha.config import CircuitBreakerConfig, PortfolioConfig, PositionSizingConfig
from alphavedha.risk.portfolio import HoldingInfo, PortfolioState
from alphavedha.risk.risk_manager import RiskAssessment, RiskManager


@pytest.fixture
def risk_manager() -> RiskManager:
    return RiskManager(
        position_config=PositionSizingConfig(
            method="half_kelly", max_single_stock_pct=10.0, min_confidence=0.55
        ),
        portfolio_config=PortfolioConfig(
            max_sector_pct=25.0, max_correlation=0.7, min_holding_days=3, min_daily_turnover_cr=5.0
        ),
        circuit_breaker_config=CircuitBreakerConfig(
            level_1_drawdown=10.0, level_2_drawdown=15.0, level_3_drawdown=20.0, recovery_threshold=0.95
        ),
    )


@pytest.fixture
def healthy_portfolio() -> PortfolioState:
    return PortfolioState(
        holdings={
            "INFY": HoldingInfo(
                symbol="INFY",
                sector="IT",
                weight_pct=5.0,
                entry_date=datetime(2026, 1, 1, tzinfo=UTC),
                correlation_60d={},
                avg_daily_turnover_cr=100.0,
            )
        },
        total_value=1_000_000.0,
        peak_value=1_000_000.0,
    )


class TestRiskManager:
    def test_full_pipeline_returns_assessment(
        self, risk_manager: RiskManager, healthy_portfolio: PortfolioState
    ) -> None:
        result = risk_manager.assess(
            meta_confidence=0.70,
            magnitude=0.05,
            symbol="TCS",
            sector="IT",
            portfolio=healthy_portfolio,
        )
        assert isinstance(result, RiskAssessment)
        assert result.position_size_pct > 0.0
        assert result.kelly_raw > 0.0
        assert result.kelly_half > 0.0
        assert result.circuit_breaker_level == 0

    def test_no_portfolio_only_kelly(self, risk_manager: RiskManager) -> None:
        result = risk_manager.assess(
            meta_confidence=0.70,
            magnitude=0.05,
            symbol="TCS",
            sector="IT",
            portfolio=None,
        )
        assert result.position_size_pct > 0.0
        assert result.circuit_breaker_level == 0
        assert len(result.constraint_violations) == 0

    def test_low_confidence_zero_position(self, risk_manager: RiskManager) -> None:
        result = risk_manager.assess(
            meta_confidence=0.40,
            magnitude=0.05,
            symbol="TCS",
            sector="IT",
            portfolio=None,
        )
        assert result.position_size_pct == 0.0

    def test_circuit_breaker_level_2_blocks_new_entry(
        self, risk_manager: RiskManager
    ) -> None:
        drawdown_portfolio = PortfolioState(
            holdings={},
            total_value=840_000.0,
            peak_value=1_000_000.0,
        )
        result = risk_manager.assess(
            meta_confidence=0.70,
            magnitude=0.05,
            symbol="TCS",
            sector="IT",
            portfolio=drawdown_portfolio,
        )
        assert result.position_size_pct == 0.0
        assert result.circuit_breaker_level == 2
        assert result.risk_adjusted is True
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/unit/risk/test_risk_manager.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'alphavedha.risk.risk_manager'`

- [ ] **Step 3: Implement RiskManager**

Create `alphavedha/risk/risk_manager.py`:

```python
"""RiskManager — orchestrates position sizing, portfolio constraints, and circuit breaker."""

from __future__ import annotations

from dataclasses import dataclass, field

import structlog

from alphavedha.config import CircuitBreakerConfig, PortfolioConfig, PositionSizingConfig
from alphavedha.risk.circuit_breaker import CircuitBreaker
from alphavedha.risk.portfolio import PortfolioConstraints, PortfolioState
from alphavedha.risk.position_sizing import compute_position_size

logger = structlog.get_logger(__name__)


@dataclass
class RiskAssessment:
    position_size_pct: float
    kelly_raw: float
    kelly_half: float
    constraint_violations: list[str] = field(default_factory=list)
    circuit_breaker_level: int = 0
    risk_adjusted: bool = False


class RiskManager:

    def __init__(
        self,
        position_config: PositionSizingConfig,
        portfolio_config: PortfolioConfig,
        circuit_breaker_config: CircuitBreakerConfig,
    ) -> None:
        self._position_config = position_config
        self._portfolio_constraints = PortfolioConstraints(portfolio_config)
        self._circuit_breaker = CircuitBreaker(circuit_breaker_config)

    def assess(
        self,
        meta_confidence: float,
        magnitude: float,
        symbol: str,
        sector: str,
        portfolio: PortfolioState | None = None,
    ) -> RiskAssessment:
        # Step 1: Kelly position sizing
        kelly_half = compute_position_size(meta_confidence, magnitude, self._position_config)
        kelly_raw = (2 * meta_confidence - 1) * 100 if magnitude > 0 and meta_confidence >= self._position_config.min_confidence else 0.0
        kelly_raw = max(kelly_raw, 0.0)

        position = kelly_half
        violations: list[str] = []
        cb_level = 0
        adjusted = False

        if portfolio is not None:
            # Step 2: Portfolio constraints
            constraint_result = self._portfolio_constraints.check(
                symbol=symbol,
                proposed_weight_pct=position,
                sector=sector,
                portfolio=portfolio,
            )
            violations = constraint_result.violations
            if constraint_result.adjusted_weight_pct != position:
                adjusted = True
            position = constraint_result.adjusted_weight_pct

            # Step 3: Circuit breaker
            cb_state = self._circuit_breaker.evaluate(
                current_value=portfolio.total_value,
                peak_value=portfolio.peak_value,
            )
            cb_level = cb_state.level
            cb_adjusted = self._circuit_breaker.adjust_position(
                proposed_size_pct=position,
                state=cb_state,
                is_new_entry=symbol not in portfolio.holdings,
            )
            if cb_adjusted != position:
                adjusted = True
            position = cb_adjusted

        if position != kelly_half:
            adjusted = True

        logger.info(
            "risk_assessment",
            symbol=symbol,
            kelly_raw=round(kelly_raw, 4),
            kelly_half=round(kelly_half, 4),
            final_position=round(position, 4),
            cb_level=cb_level,
            violations=violations,
        )

        return RiskAssessment(
            position_size_pct=position,
            kelly_raw=kelly_raw,
            kelly_half=kelly_half,
            constraint_violations=violations,
            circuit_breaker_level=cb_level,
            risk_adjusted=adjusted,
        )
```

- [ ] **Step 4: Update `alphavedha/risk/__init__.py`**

```python
"""Risk management — position sizing, portfolio constraints, circuit breakers."""

from alphavedha.risk.circuit_breaker import CircuitBreaker, CircuitBreakerState
from alphavedha.risk.portfolio import (
    ConstraintResult,
    HoldingInfo,
    PortfolioConstraints,
    PortfolioState,
)
from alphavedha.risk.position_sizing import compute_position_size
from alphavedha.risk.risk_manager import RiskAssessment, RiskManager

__all__ = [
    "CircuitBreaker",
    "CircuitBreakerState",
    "ConstraintResult",
    "HoldingInfo",
    "PortfolioConstraints",
    "PortfolioState",
    "RiskAssessment",
    "RiskManager",
    "compute_position_size",
]
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/unit/risk/ -v`
Expected: 22 PASSED (7 + 5 + 10 from Tasks 1-3, plus 4 new)

- [ ] **Step 6: Lint and type-check**

Run: `ruff check alphavedha/risk/ tests/unit/risk/ && ruff format alphavedha/risk/ tests/unit/risk/ && python -m mypy alphavedha/risk/`

- [ ] **Step 7: Commit**

```bash
git add alphavedha/risk/risk_manager.py alphavedha/risk/__init__.py tests/unit/risk/test_risk_manager.py
git commit -m "feat: add RiskManager orchestrating Kelly, portfolio, and circuit breaker"
```

---

### Task 5: Composite Scorer

**Files:**
- Create: `alphavedha/prediction/scorer.py`
- Modify: `alphavedha/config.py` (add `CompositeScoreWeights`)
- Test: `tests/unit/prediction/test_scorer.py`

- [ ] **Step 1: Add CompositeScoreWeights config**

Add to `alphavedha/config.py` after the `ApiConfig` class (around line 245):

```python
class CompositeScoreWeights(BaseModel):
    technical_momentum: float = 0.25
    derivatives_sentiment: float = 0.20
    macro_alignment: float = 0.15
    microstructure_quality: float = 0.15
    news_sentiment: float = 0.10
    volatility_risk: float = 0.15
```

- [ ] **Step 2: Write the failing tests**

Create `tests/unit/prediction/__init__.py` (empty) and `tests/unit/prediction/test_scorer.py`:

```python
"""Tests for CompositeScorer — 0-100 weighted scoring."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from alphavedha.config import CompositeScoreWeights
from alphavedha.models.ensemble import EnsembleResult
from alphavedha.models.regime import RegimeResult
from alphavedha.prediction.scorer import CompositeScorer

_DEFAULT_WEIGHTS = CompositeScoreWeights()


def _make_regime(regime: str = "bull") -> RegimeResult:
    probs = np.array([0.7, 0.1, 0.1, 0.1]) if regime == "bull" else np.array([0.1, 0.7, 0.1, 0.1])
    return RegimeResult(
        current_regime=regime,
        regime_id=0 if regime == "bull" else 1,
        state_probabilities=probs,
        regime_history=np.array([0]),
        transition_matrix=np.eye(4),
    )


def _make_ensemble(direction: int = 1, confidence: float = 0.8) -> EnsembleResult:
    probs = np.array([[0.1, 0.1, 0.8]]) if direction == 1 else np.array([[0.8, 0.1, 0.1]])
    return EnsembleResult(
        direction=np.array([direction]),
        magnitude=np.array([0.03]),
        probabilities=probs,
        confidence=np.array([confidence]),
        model_disagreement=np.array([0.05]),
    )


def _make_features_full() -> pd.DataFrame:
    return pd.DataFrame({
        "deriv_pcr_oi": [0.8],
        "deriv_futures_oi_change": [1000],
        "micro_delivery_pct": [0.65],
        "micro_vol_anomaly": [0.3],
        "sent_news_score": [0.7],
        "sent_velocity": [0.5],
        "hvol_20": [0.15],
        "natr_14": [0.02],
        "atr_14": [50.0],
    })


class TestCompositeScorer:
    def test_full_features_score_in_range(self) -> None:
        scorer = CompositeScorer()
        score = scorer.score(
            _make_ensemble(direction=1, confidence=0.8),
            _make_regime("bull"),
            _make_features_full(),
        )
        assert 0.0 <= score <= 100.0

    def test_missing_feature_group_uses_neutral(self) -> None:
        scorer = CompositeScorer()
        features = pd.DataFrame({"some_unrelated": [1.0]})
        score = scorer.score(
            _make_ensemble(direction=1, confidence=0.8),
            _make_regime("bull"),
            features,
        )
        assert 0.0 <= score <= 100.0

    def test_all_features_missing_returns_near_neutral(self) -> None:
        scorer = CompositeScorer()
        features = pd.DataFrame({"unrelated": [1.0]})
        score = scorer.score(
            _make_ensemble(direction=0, confidence=0.5),
            _make_regime("sideways"),
            features,
        )
        assert 40.0 <= score <= 60.0

    def test_bull_regime_buy_signal_high_macro(self) -> None:
        scorer = CompositeScorer()
        score_bull_buy = scorer.score(
            _make_ensemble(direction=1, confidence=0.8),
            _make_regime("bull"),
            _make_features_full(),
        )
        score_bull_sell = scorer.score(
            _make_ensemble(direction=-1, confidence=0.8),
            _make_regime("bull"),
            _make_features_full(),
        )
        assert score_bull_buy > score_bull_sell

    def test_bear_regime_sell_signal_high_macro(self) -> None:
        scorer = CompositeScorer()
        score_bear_sell = scorer.score(
            _make_ensemble(direction=-1, confidence=0.8),
            _make_regime("bear"),
            _make_features_full(),
        )
        score_bear_buy = scorer.score(
            _make_ensemble(direction=1, confidence=0.8),
            _make_regime("bear"),
            _make_features_full(),
        )
        assert score_bear_sell > score_bear_buy

    def test_custom_weights(self) -> None:
        weights = CompositeScoreWeights(
            technical_momentum=1.0,
            derivatives_sentiment=0.0,
            macro_alignment=0.0,
            microstructure_quality=0.0,
            news_sentiment=0.0,
            volatility_risk=0.0,
        )
        scorer = CompositeScorer(weights=weights)
        score = scorer.score(
            _make_ensemble(direction=1, confidence=0.9),
            _make_regime("bull"),
            _make_features_full(),
        )
        assert 0.0 <= score <= 100.0
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `python -m pytest tests/unit/prediction/test_scorer.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'alphavedha.prediction.scorer'`

- [ ] **Step 4: Implement CompositeScorer**

Create `alphavedha/prediction/scorer.py`:

```python
"""CompositeScorer — convert model outputs + features into a 0-100 human-readable score."""

from __future__ import annotations

import numpy as np
import pandas as pd
import structlog

from alphavedha.config import CompositeScoreWeights
from alphavedha.models.ensemble import EnsembleResult
from alphavedha.models.regime import RegimeResult

logger = structlog.get_logger(__name__)

_DEFAULT_WEIGHTS = CompositeScoreWeights()

_REGIME_ALIGNMENT = {
    ("bull", 1): 100.0,
    ("bull", 0): 50.0,
    ("bull", -1): 0.0,
    ("bear", -1): 100.0,
    ("bear", 0): 50.0,
    ("bear", 1): 0.0,
    ("sideways", 0): 70.0,
    ("sideways", 1): 40.0,
    ("sideways", -1): 40.0,
    ("high_volatility", 0): 60.0,
    ("high_volatility", 1): 30.0,
    ("high_volatility", -1): 30.0,
}

_FEATURE_PREFIXES = {
    "derivatives_sentiment": ["deriv_"],
    "microstructure_quality": ["micro_"],
    "news_sentiment": ["sent_"],
    "volatility_risk": ["hvol_", "natr_", "atr_", "bb_width_"],
}


class CompositeScorer:

    def __init__(self, weights: CompositeScoreWeights | None = None) -> None:
        self._weights = weights or _DEFAULT_WEIGHTS

    def score(
        self,
        ensemble_result: EnsembleResult,
        regime_result: RegimeResult,
        features: pd.DataFrame,
    ) -> float:
        weight_dict = {
            "technical_momentum": self._weights.technical_momentum,
            "derivatives_sentiment": self._weights.derivatives_sentiment,
            "macro_alignment": self._weights.macro_alignment,
            "microstructure_quality": self._weights.microstructure_quality,
            "news_sentiment": self._weights.news_sentiment,
            "volatility_risk": self._weights.volatility_risk,
        }

        direction = int(ensemble_result.direction[0])
        confidence = float(ensemble_result.confidence[0])

        sub_scores: dict[str, float | None] = {}

        # Technical momentum: always available from ensemble
        sub_scores["technical_momentum"] = confidence * 100.0

        # Macro alignment: always available from regime
        regime = regime_result.current_regime
        sub_scores["macro_alignment"] = _REGIME_ALIGNMENT.get((regime, direction), 50.0)

        # Feature-derived sub-scores
        for score_name, prefixes in _FEATURE_PREFIXES.items():
            cols = [c for c in features.columns if any(c.startswith(p) for p in prefixes)]
            if not cols:
                sub_scores[score_name] = None
            else:
                values = features[cols].iloc[0].values.astype(float)
                finite_vals = values[np.isfinite(values)]
                if len(finite_vals) == 0:
                    sub_scores[score_name] = None
                else:
                    mean_val = float(np.mean(finite_vals))
                    if score_name == "volatility_risk":
                        # Invert: low vol = high score
                        # Normalize via sigmoid-like mapping centered around 0
                        normalized = 1.0 / (1.0 + np.exp(mean_val * 5))
                        sub_scores[score_name] = float(normalized * 100.0)
                    else:
                        # Sigmoid normalization to 0-100
                        normalized = 1.0 / (1.0 + np.exp(-mean_val * 2))
                        sub_scores[score_name] = float(normalized * 100.0)

        # Redistribute weights from unavailable sub-scores
        available_weight = sum(
            weight_dict[k] for k, v in sub_scores.items() if v is not None
        )
        if available_weight <= 0:
            return 50.0

        weighted_sum = 0.0
        for name, raw_score in sub_scores.items():
            if raw_score is None:
                continue
            normalized_weight = weight_dict[name] / available_weight
            weighted_sum += raw_score * normalized_weight

        result = max(0.0, min(100.0, weighted_sum))

        logger.debug(
            "composite_score_computed",
            score=round(result, 2),
            sub_scores={k: round(v, 2) if v is not None else None for k, v in sub_scores.items()},
        )

        return result
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/unit/prediction/test_scorer.py -v`
Expected: 6 PASSED

- [ ] **Step 6: Lint and type-check**

Run: `ruff check alphavedha/prediction/scorer.py alphavedha/config.py tests/unit/prediction/test_scorer.py && ruff format alphavedha/prediction/scorer.py alphavedha/config.py tests/unit/prediction/test_scorer.py && python -m mypy alphavedha/prediction/scorer.py`

- [ ] **Step 7: Commit**

```bash
git add alphavedha/prediction/scorer.py alphavedha/config.py tests/unit/prediction/__init__.py tests/unit/prediction/test_scorer.py
git commit -m "feat: add CompositeScorer with 6 weighted sub-scores and missing-feature handling"
```

---

### Task 6: Stock Ranker

**Files:**
- Create: `alphavedha/prediction/ranker.py`
- Test: `tests/unit/prediction/test_ranker.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/prediction/test_ranker.py`:

```python
"""Tests for StockRanker — filter and rank predictions."""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pytest

from alphavedha.prediction.ranker import RankingResult, StockRanker

# Import StockPrediction from engine once created; for now define a compatible stub
from alphavedha.prediction.engine import StockPrediction


def _make_prediction(
    symbol: str,
    direction: int = 1,
    composite_score: float = 75.0,
    is_tradeable: bool = True,
    position_size_pct: float = 5.0,
) -> StockPrediction:
    return StockPrediction(
        symbol=symbol,
        timestamp=datetime.now(UTC),
        direction=direction,
        magnitude=0.03,
        composite_score=composite_score,
        meta_confidence=0.7,
        is_tradeable=is_tradeable,
        regime="bull",
        regime_probabilities=np.array([0.7, 0.1, 0.1, 0.1]),
        price_target_low=100.0,
        price_target_mid=105.0,
        price_target_high=110.0,
        model_disagreement=0.05,
        position_size_pct=position_size_pct,
        model_version="v0.1.0",
        warnings=[],
    )


class TestStockRanker:
    def test_filters_non_tradeable(self) -> None:
        ranker = StockRanker()
        preds = [
            _make_prediction("TCS", direction=1, is_tradeable=True),
            _make_prediction("INFY", direction=1, is_tradeable=False),
        ]
        result = ranker.rank(preds)
        assert isinstance(result, RankingResult)
        assert len(result.buy_candidates) == 1
        assert result.buy_candidates[0].symbol == "TCS"
        assert any("INFY" == sym for sym, _ in result.excluded)

    def test_separates_buy_and_sell(self) -> None:
        ranker = StockRanker()
        preds = [
            _make_prediction("TCS", direction=1),
            _make_prediction("RELIANCE", direction=-1),
            _make_prediction("HDFC", direction=0, is_tradeable=True),
        ]
        result = ranker.rank(preds)
        assert len(result.buy_candidates) == 1
        assert len(result.sell_candidates) == 1
        assert result.buy_candidates[0].symbol == "TCS"
        assert result.sell_candidates[0].symbol == "RELIANCE"

    def test_sorts_by_composite_score_desc(self) -> None:
        ranker = StockRanker()
        preds = [
            _make_prediction("LOW", direction=1, composite_score=60.0),
            _make_prediction("HIGH", direction=1, composite_score=90.0),
            _make_prediction("MID", direction=1, composite_score=75.0),
        ]
        result = ranker.rank(preds)
        scores = [p.composite_score for p in result.buy_candidates]
        assert scores == sorted(scores, reverse=True)
        assert result.buy_candidates[0].symbol == "HIGH"

    def test_respects_top_n(self) -> None:
        ranker = StockRanker()
        preds = [_make_prediction(f"STOCK{i}", direction=1, composite_score=float(90 - i)) for i in range(20)]
        result = ranker.rank(preds, top_n=5)
        assert len(result.buy_candidates) == 5

    def test_circuit_hit_excluded(self) -> None:
        ranker = StockRanker()
        preds = [
            _make_prediction("TCS", direction=1),
            _make_prediction("INFY", direction=1),
        ]
        result = ranker.rank(preds, circuit_hit_symbols={"INFY"})
        assert len(result.buy_candidates) == 1
        assert result.buy_candidates[0].symbol == "TCS"
        assert any("INFY" == sym for sym, _ in result.excluded)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/unit/prediction/test_ranker.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement StockPrediction dataclass and StockRanker**

First, create `alphavedha/prediction/engine.py` with just the `StockPrediction` dataclass (the full engine comes in Task 7):

```python
"""PredictionEngine — orchestrates the full prediction pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

import numpy as np


@dataclass
class StockPrediction:
    symbol: str
    timestamp: datetime
    direction: int
    magnitude: float
    composite_score: float
    meta_confidence: float
    is_tradeable: bool
    regime: str
    regime_probabilities: np.ndarray
    price_target_low: float
    price_target_mid: float
    price_target_high: float
    model_disagreement: float
    position_size_pct: float
    model_version: str
    warnings: list[str] = field(default_factory=list)
```

Then create `alphavedha/prediction/ranker.py`:

```python
"""StockRanker — filter and rank stock predictions into buy/sell candidate lists."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime

import structlog

from alphavedha.prediction.engine import StockPrediction

logger = structlog.get_logger(__name__)


@dataclass
class RankingResult:
    buy_candidates: list[StockPrediction]
    sell_candidates: list[StockPrediction]
    excluded: list[tuple[str, str]] = field(default_factory=list)
    generated_at: datetime = field(default_factory=lambda: datetime.now(UTC))


class StockRanker:

    def rank(
        self,
        predictions: list[StockPrediction],
        top_n: int = 10,
        circuit_hit_symbols: set[str] | None = None,
    ) -> RankingResult:
        circuit_hits = circuit_hit_symbols or set()
        excluded: list[tuple[str, str]] = []
        candidates: list[StockPrediction] = []

        for pred in predictions:
            if pred.symbol in circuit_hits:
                excluded.append((pred.symbol, "circuit hit"))
                continue
            if not pred.is_tradeable:
                excluded.append((pred.symbol, "not tradeable"))
                continue
            if pred.position_size_pct <= 0:
                excluded.append((pred.symbol, "zero position size"))
                continue
            candidates.append(pred)

        buy = sorted(
            [p for p in candidates if p.direction == 1],
            key=lambda p: p.composite_score,
            reverse=True,
        )[:top_n]

        sell = sorted(
            [p for p in candidates if p.direction == -1],
            key=lambda p: p.composite_score,
            reverse=True,
        )[:top_n]

        logger.info(
            "ranking_completed",
            total=len(predictions),
            buy=len(buy),
            sell=len(sell),
            excluded=len(excluded),
        )

        return RankingResult(
            buy_candidates=buy,
            sell_candidates=sell,
            excluded=excluded,
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/unit/prediction/test_ranker.py -v`
Expected: 5 PASSED

- [ ] **Step 5: Lint and type-check**

Run: `ruff check alphavedha/prediction/engine.py alphavedha/prediction/ranker.py tests/unit/prediction/test_ranker.py && ruff format alphavedha/prediction/engine.py alphavedha/prediction/ranker.py tests/unit/prediction/test_ranker.py && python -m mypy alphavedha/prediction/engine.py alphavedha/prediction/ranker.py`

- [ ] **Step 6: Commit**

```bash
git add alphavedha/prediction/engine.py alphavedha/prediction/ranker.py tests/unit/prediction/test_ranker.py
git commit -m "feat: add StockRanker with filtering, buy/sell separation, and top-N"
```

---

### Task 7: Prediction Engine

**Files:**
- Modify: `alphavedha/prediction/engine.py` (add PredictionEngine class)
- Modify: `alphavedha/prediction/__init__.py`
- Test: `tests/unit/prediction/test_engine.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/prediction/test_engine.py`:

```python
"""Tests for PredictionEngine — full pipeline orchestration with mocked models."""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from alphavedha.exceptions import PredictionError
from alphavedha.models.base import PredictionResult
from alphavedha.models.conformal import ConformalResult
from alphavedha.models.ensemble import EnsembleResult
from alphavedha.models.meta_model import MetaLabelResult
from alphavedha.models.regime import RegimeResult
from alphavedha.prediction.engine import PredictionEngine, StockPrediction
from alphavedha.prediction.scorer import CompositeScorer
from alphavedha.risk.risk_manager import RiskAssessment, RiskManager


def _mock_base_model(name: str) -> MagicMock:
    model = MagicMock()
    model.predict.return_value = PredictionResult(
        direction=np.array([1]),
        magnitude=np.array([0.03]),
        probabilities=np.array([[0.1, 0.2, 0.7]]),
        confidence=np.array([0.7]),
    )
    return model


def _mock_regime() -> MagicMock:
    regime = MagicMock()
    regime.predict.return_value = RegimeResult(
        current_regime="bull",
        regime_id=0,
        state_probabilities=np.array([0.7, 0.1, 0.1, 0.1]),
        regime_history=np.array([0]),
        transition_matrix=np.eye(4),
    )
    return regime


def _mock_ensemble() -> MagicMock:
    ens = MagicMock()
    ens.predict.return_value = EnsembleResult(
        direction=np.array([1]),
        magnitude=np.array([0.03]),
        probabilities=np.array([[0.1, 0.2, 0.7]]),
        confidence=np.array([0.75]),
        model_disagreement=np.array([0.05]),
    )
    return ens


def _mock_meta() -> MagicMock:
    meta = MagicMock()
    meta.predict.return_value = MetaLabelResult(
        meta_confidence=np.array([0.72]),
        is_tradeable=np.array([True]),
    )
    return meta


def _mock_conformal() -> MagicMock:
    conf = MagicMock()
    conf.predict.return_value = ConformalResult(
        price_low=np.array([95.0]),
        price_mid=np.array([100.0]),
        price_high=np.array([105.0]),
        interval_width=np.array([10.0]),
        coverage=0.90,
    )
    return conf


def _mock_risk_manager() -> MagicMock:
    rm = MagicMock(spec=RiskManager)
    rm.assess.return_value = RiskAssessment(
        position_size_pct=5.0,
        kelly_raw=0.40,
        kelly_half=0.20,
        constraint_violations=[],
        circuit_breaker_level=0,
        risk_adjusted=False,
    )
    return rm


@pytest.fixture
def engine() -> PredictionEngine:
    return PredictionEngine(
        xgboost=_mock_base_model("xgboost"),
        lstm=_mock_base_model("lstm"),
        tft=_mock_base_model("tft"),
        regime=_mock_regime(),
        ensemble=_mock_ensemble(),
        meta_model=_mock_meta(),
        conformal=_mock_conformal(),
        scorer=CompositeScorer(),
        risk_manager=_mock_risk_manager(),
        model_version="v0.1.0",
    )


@pytest.fixture
def features() -> pd.DataFrame:
    return pd.DataFrame({"feature1": [1.0], "feature2": [2.0]})


@pytest.fixture
def returns() -> pd.Series:
    return pd.Series([0.01])


class TestPredictionEngine:
    def test_predict_returns_stock_prediction(
        self, engine: PredictionEngine, features: pd.DataFrame, returns: pd.Series
    ) -> None:
        result = engine.predict(
            symbol="TCS",
            features=features,
            returns=returns,
            current_price=3500.0,
        )
        assert isinstance(result, StockPrediction)
        assert result.symbol == "TCS"
        assert result.direction in (-1, 0, 1)
        assert 0.0 <= result.composite_score <= 100.0
        assert result.model_version == "v0.1.0"
        assert len(result.warnings) == 0

    def test_one_model_failure_degrades_gracefully(
        self, features: pd.DataFrame, returns: pd.Series
    ) -> None:
        lstm = _mock_base_model("lstm")
        lstm.predict.side_effect = RuntimeError("LSTM failed")
        engine = PredictionEngine(
            xgboost=_mock_base_model("xgboost"),
            lstm=lstm,
            tft=_mock_base_model("tft"),
            regime=_mock_regime(),
            ensemble=_mock_ensemble(),
            meta_model=_mock_meta(),
            conformal=_mock_conformal(),
            scorer=CompositeScorer(),
            risk_manager=_mock_risk_manager(),
        )
        result = engine.predict("TCS", features, returns, 3500.0)
        assert isinstance(result, StockPrediction)
        assert any("lstm" in w.lower() for w in result.warnings)

    def test_two_model_failures_raises(
        self, features: pd.DataFrame, returns: pd.Series
    ) -> None:
        lstm = _mock_base_model("lstm")
        lstm.predict.side_effect = RuntimeError("LSTM failed")
        tft = _mock_base_model("tft")
        tft.predict.side_effect = RuntimeError("TFT failed")
        engine = PredictionEngine(
            xgboost=_mock_base_model("xgboost"),
            lstm=lstm,
            tft=tft,
            regime=_mock_regime(),
            ensemble=_mock_ensemble(),
            meta_model=_mock_meta(),
            conformal=_mock_conformal(),
            scorer=CompositeScorer(),
            risk_manager=_mock_risk_manager(),
        )
        with pytest.raises(PredictionError, match="fewer than 2"):
            engine.predict("TCS", features, returns, 3500.0)

    def test_all_models_fail_raises(
        self, features: pd.DataFrame, returns: pd.Series
    ) -> None:
        xgb = _mock_base_model("xgboost")
        xgb.predict.side_effect = RuntimeError("XGB failed")
        lstm = _mock_base_model("lstm")
        lstm.predict.side_effect = RuntimeError("LSTM failed")
        tft = _mock_base_model("tft")
        tft.predict.side_effect = RuntimeError("TFT failed")
        engine = PredictionEngine(
            xgboost=xgb, lstm=lstm, tft=tft,
            regime=_mock_regime(),
            ensemble=_mock_ensemble(),
            meta_model=_mock_meta(),
            conformal=_mock_conformal(),
            scorer=CompositeScorer(),
            risk_manager=_mock_risk_manager(),
        )
        with pytest.raises(PredictionError):
            engine.predict("TCS", features, returns, 3500.0)

    def test_regime_failure_uses_default(
        self, features: pd.DataFrame, returns: pd.Series
    ) -> None:
        regime = _mock_regime()
        regime.predict.side_effect = RuntimeError("Regime failed")
        engine = PredictionEngine(
            xgboost=_mock_base_model("xgboost"),
            lstm=_mock_base_model("lstm"),
            tft=_mock_base_model("tft"),
            regime=regime,
            ensemble=_mock_ensemble(),
            meta_model=_mock_meta(),
            conformal=_mock_conformal(),
            scorer=CompositeScorer(),
            risk_manager=_mock_risk_manager(),
        )
        result = engine.predict("TCS", features, returns, 3500.0)
        assert any("regime" in w.lower() for w in result.warnings)
        np.testing.assert_allclose(result.regime_probabilities, [0.25, 0.25, 0.25, 0.25])

    def test_meta_model_failure_defaults_not_tradeable(
        self, features: pd.DataFrame, returns: pd.Series
    ) -> None:
        meta = _mock_meta()
        meta.predict.side_effect = RuntimeError("Meta failed")
        rm = _mock_risk_manager()
        rm.assess.return_value = RiskAssessment(
            position_size_pct=0.0, kelly_raw=0.0, kelly_half=0.0,
            constraint_violations=[], circuit_breaker_level=0, risk_adjusted=False,
        )
        engine = PredictionEngine(
            xgboost=_mock_base_model("xgboost"),
            lstm=_mock_base_model("lstm"),
            tft=_mock_base_model("tft"),
            regime=_mock_regime(),
            ensemble=_mock_ensemble(),
            meta_model=meta,
            conformal=_mock_conformal(),
            scorer=CompositeScorer(),
            risk_manager=rm,
        )
        result = engine.predict("TCS", features, returns, 3500.0)
        assert result.meta_confidence == 0.0
        assert result.is_tradeable is False
        assert any("meta" in w.lower() for w in result.warnings)

    def test_conformal_failure_uses_nan(
        self, features: pd.DataFrame, returns: pd.Series
    ) -> None:
        conf = _mock_conformal()
        conf.predict.side_effect = RuntimeError("Conformal failed")
        engine = PredictionEngine(
            xgboost=_mock_base_model("xgboost"),
            lstm=_mock_base_model("lstm"),
            tft=_mock_base_model("tft"),
            regime=_mock_regime(),
            ensemble=_mock_ensemble(),
            meta_model=_mock_meta(),
            conformal=conf,
            scorer=CompositeScorer(),
            risk_manager=_mock_risk_manager(),
        )
        result = engine.predict("TCS", features, returns, 3500.0)
        assert np.isnan(result.price_target_low)
        assert np.isnan(result.price_target_mid)
        assert np.isnan(result.price_target_high)
        assert any("conformal" in w.lower() for w in result.warnings)

    def test_no_market_features_skips_regime(
        self, features: pd.DataFrame, returns: pd.Series
    ) -> None:
        regime = _mock_regime()
        engine = PredictionEngine(
            xgboost=_mock_base_model("xgboost"),
            lstm=_mock_base_model("lstm"),
            tft=_mock_base_model("tft"),
            regime=regime,
            ensemble=_mock_ensemble(),
            meta_model=_mock_meta(),
            conformal=_mock_conformal(),
            scorer=CompositeScorer(),
            risk_manager=_mock_risk_manager(),
        )
        result = engine.predict("TCS", features, returns, 3500.0, market_features=None)
        regime.predict.assert_not_called()
        assert any("regime" in w.lower() or "market_features" in w.lower() for w in result.warnings)

    def test_position_zero_when_not_tradeable(
        self, features: pd.DataFrame, returns: pd.Series
    ) -> None:
        meta = _mock_meta()
        meta.predict.return_value = MetaLabelResult(
            meta_confidence=np.array([0.30]),
            is_tradeable=np.array([False]),
        )
        rm = _mock_risk_manager()
        rm.assess.return_value = RiskAssessment(
            position_size_pct=0.0, kelly_raw=0.0, kelly_half=0.0,
            constraint_violations=[], circuit_breaker_level=0, risk_adjusted=False,
        )
        engine = PredictionEngine(
            xgboost=_mock_base_model("xgboost"),
            lstm=_mock_base_model("lstm"),
            tft=_mock_base_model("tft"),
            regime=_mock_regime(),
            ensemble=_mock_ensemble(),
            meta_model=meta,
            conformal=_mock_conformal(),
            scorer=CompositeScorer(),
            risk_manager=rm,
        )
        result = engine.predict("TCS", features, returns, 3500.0)
        assert result.position_size_pct == 0.0
        assert result.is_tradeable is False
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/unit/prediction/test_engine.py -v`
Expected: FAIL with `ImportError: cannot import name 'PredictionEngine'`

- [ ] **Step 3: Implement PredictionEngine**

Replace the contents of `alphavedha/prediction/engine.py` with the full implementation (keeping `StockPrediction` from Task 6):

```python
"""PredictionEngine — orchestrates the full prediction pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd
import structlog

from alphavedha.exceptions import PredictionError
from alphavedha.models.base import PredictionResult
from alphavedha.models.conformal import ConformalPredictor
from alphavedha.models.ensemble import StackingEnsemble
from alphavedha.models.meta_model import MetaLabelingModel
from alphavedha.models.regime import RegimeDetector
from alphavedha.prediction.scorer import CompositeScorer
from alphavedha.risk.portfolio import PortfolioState
from alphavedha.risk.risk_manager import RiskManager

logger = structlog.get_logger(__name__)

_NEUTRAL_PROBS = np.array([[1 / 3, 1 / 3, 1 / 3]])
_UNIFORM_REGIME = np.array([0.25, 0.25, 0.25, 0.25])
_MIN_SUCCESSFUL_MODELS = 2


@dataclass
class StockPrediction:
    symbol: str
    timestamp: datetime
    direction: int
    magnitude: float
    composite_score: float
    meta_confidence: float
    is_tradeable: bool
    regime: str
    regime_probabilities: np.ndarray
    price_target_low: float
    price_target_mid: float
    price_target_high: float
    model_disagreement: float
    position_size_pct: float
    model_version: str
    warnings: list[str] = field(default_factory=list)


class PredictionEngine:

    def __init__(
        self,
        xgboost: Any,
        lstm: Any,
        tft: Any,
        regime: RegimeDetector,
        ensemble: StackingEnsemble,
        meta_model: MetaLabelingModel,
        conformal: ConformalPredictor,
        scorer: CompositeScorer,
        risk_manager: RiskManager,
        model_version: str = "v0.1.0",
    ) -> None:
        self._models = {"xgboost": xgboost, "lstm": lstm, "tft": tft}
        self._regime = regime
        self._ensemble = ensemble
        self._meta_model = meta_model
        self._conformal = conformal
        self._scorer = scorer
        self._risk_manager = risk_manager
        self._model_version = model_version

    def predict(
        self,
        symbol: str,
        features: pd.DataFrame,
        returns: pd.Series,
        current_price: float,
        market_features: pd.DataFrame | None = None,
        current_portfolio: PortfolioState | None = None,
    ) -> StockPrediction:
        warnings: list[str] = []
        now = datetime.now(UTC)

        # Step 1: Regime detection
        regime_name, regime_probs = self._run_regime(market_features, warnings)

        # Step 2-4: Base models with graceful degradation
        base_predictions = self._run_base_models(features, warnings)

        # Step 5: Ensemble
        ensemble_result = self._ensemble.predict(
            base_predictions, regime_probs.reshape(1, -1)
        )
        direction = int(ensemble_result.direction[0])
        magnitude = float(ensemble_result.magnitude[0])
        confidence = float(ensemble_result.confidence[0])
        disagreement = float(ensemble_result.model_disagreement[0])

        # Step 6: Meta-labeling
        meta_confidence, is_tradeable = self._run_meta(
            features, ensemble_result, warnings
        )

        # Step 7: Conformal prediction
        price_low, price_mid, price_high = self._run_conformal(features, warnings)

        # Step 8: Composite score
        regime_result = self._build_regime_result(regime_name, regime_probs)
        composite_score = self._scorer.score(ensemble_result, regime_result, features)

        # Step 9: Risk assessment
        risk = self._risk_manager.assess(
            meta_confidence=meta_confidence,
            magnitude=magnitude,
            symbol=symbol,
            sector="",
            portfolio=current_portfolio,
        )

        return StockPrediction(
            symbol=symbol,
            timestamp=now,
            direction=direction,
            magnitude=magnitude,
            composite_score=composite_score,
            meta_confidence=meta_confidence,
            is_tradeable=is_tradeable,
            regime=regime_name,
            regime_probabilities=regime_probs,
            price_target_low=price_low,
            price_target_mid=price_mid,
            price_target_high=price_high,
            model_disagreement=disagreement,
            position_size_pct=risk.position_size_pct,
            model_version=self._model_version,
            warnings=warnings,
        )

    def _run_regime(
        self,
        market_features: pd.DataFrame | None,
        warnings: list[str],
    ) -> tuple[str, np.ndarray]:
        if market_features is None:
            warnings.append("No market_features provided; regime detection skipped")
            return "unknown", _UNIFORM_REGIME.copy()
        try:
            result = self._regime.predict(
                returns=market_features.iloc[:, 0],
                volatility=market_features.iloc[:, 1],
            )
            return result.current_regime, result.state_probabilities
        except Exception as e:
            logger.warning("regime_detection_failed", error=str(e))
            warnings.append(f"Regime detection failed: {e}")
            return "unknown", _UNIFORM_REGIME.copy()

    def _run_base_models(
        self,
        features: pd.DataFrame,
        warnings: list[str],
    ) -> dict[str, PredictionResult]:
        results: dict[str, PredictionResult] = {}
        failed: list[str] = []

        for name, model in self._models.items():
            try:
                results[name] = model.predict(features)
            except Exception as e:
                logger.warning("base_model_failed", model=name, error=str(e))
                warnings.append(f"{name} model failed: {e}")
                failed.append(name)

        n_success = len(results)
        if n_success < _MIN_SUCCESSFUL_MODELS:
            raise PredictionError(
                f"Only {n_success} base model(s) succeeded, fewer than 2 required. "
                f"Failed: {failed}"
            )

        # Fill failed models with neutral predictions
        n = features.shape[0]
        for name in failed:
            results[name] = PredictionResult(
                direction=np.zeros(n, dtype=int),
                magnitude=np.zeros(n),
                probabilities=np.tile([1 / 3, 1 / 3, 1 / 3], (n, 1)),
                confidence=np.zeros(n),
            )

        return results

    def _run_meta(
        self,
        features: pd.DataFrame,
        ensemble_result: Any,
        warnings: list[str],
    ) -> tuple[float, bool]:
        try:
            meta_result = self._meta_model.predict(
                features,
                ensemble_result.direction,
                ensemble_result.confidence,
            )
            return float(meta_result.meta_confidence[0]), bool(meta_result.is_tradeable[0])
        except Exception as e:
            logger.warning("meta_model_failed", error=str(e))
            warnings.append(f"Meta-labeling failed: {e}")
            return 0.0, False

    def _run_conformal(
        self,
        features: pd.DataFrame,
        warnings: list[str],
    ) -> tuple[float, float, float]:
        try:
            result = self._conformal.predict(features)
            return (
                float(result.price_low[0]),
                float(result.price_mid[0]),
                float(result.price_high[0]),
            )
        except Exception as e:
            logger.warning("conformal_failed", error=str(e))
            warnings.append(f"Conformal prediction failed: {e}")
            return float("nan"), float("nan"), float("nan")

    def _build_regime_result(self, regime_name: str, regime_probs: np.ndarray) -> Any:
        from alphavedha.models.regime import RegimeResult
        return RegimeResult(
            current_regime=regime_name,
            regime_id=0,
            state_probabilities=regime_probs,
            regime_history=np.array([0]),
            transition_matrix=np.eye(4),
        )
```

- [ ] **Step 4: Update `alphavedha/prediction/__init__.py`**

```python
"""Prediction engine — pipeline orchestration, scoring, and ranking."""

from alphavedha.prediction.engine import PredictionEngine, StockPrediction
from alphavedha.prediction.ranker import RankingResult, StockRanker
from alphavedha.prediction.scorer import CompositeScorer

__all__ = [
    "CompositeScorer",
    "PredictionEngine",
    "RankingResult",
    "StockPrediction",
    "StockRanker",
]
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/unit/prediction/ -v`
Expected: All tests PASS (9 engine + 6 scorer + 5 ranker = 20)

- [ ] **Step 6: Run full test suite**

Run: `python -m pytest tests/ -v`
Expected: All ~300 tests PASS (278 existing + ~22 new risk + ~20 new prediction)

- [ ] **Step 7: Lint and type-check**

Run: `ruff check alphavedha/prediction/ alphavedha/risk/ tests/unit/prediction/ tests/unit/risk/ && ruff format alphavedha/prediction/ alphavedha/risk/ tests/unit/prediction/ tests/unit/risk/ && python -m mypy alphavedha/prediction/ alphavedha/risk/`

- [ ] **Step 8: Commit**

```bash
git add alphavedha/prediction/engine.py alphavedha/prediction/__init__.py tests/unit/prediction/test_engine.py
git commit -m "feat: add PredictionEngine orchestrating full model pipeline with graceful degradation"
```

---

### Task 8: Documentation + Exports + Final Verification

**Files:**
- Modify: `alphavedha/prediction/CLAUDE.md`
- Modify: `alphavedha/risk/CLAUDE.md`

- [ ] **Step 1: Update prediction CLAUDE.md**

Replace `alphavedha/prediction/CLAUDE.md` with:

```markdown
# Prediction Engine — AlphaVedha

## Responsibility
Orchestrate the full prediction pipeline: data → features → models → ensemble → risk → output.

## Pipeline Flow

```
1. engine.py receives prediction request (symbol + features + returns + price)
2. Run regime detection (HMM) → get current regime (or default if no market_features)
3. Run all base models (XGBoost, LSTM, TFT) → get raw predictions
4. Graceful degradation: 2/3 minimum, failed models get neutral PredictionResult
5. Run stacking meta-learner → get calibrated prediction
6. Run meta-labeling model → get confidence score + is_tradeable
7. Run conformal prediction → get price target range
8. Run CompositeScorer → get 0-100 composite score
9. Run RiskManager.assess() → get position sizing
10. Return StockPrediction with all fields populated
```

## Modules

### engine.py — PredictionEngine
- Constructor takes all models + scorer + risk_manager via dependency injection
- `predict()` returns `StockPrediction` dataclass with 16 fields
- Graceful degradation: catches individual model failures, applies defaults, adds warnings
- Minimum 2/3 base models must succeed; otherwise raises `PredictionError`
- `market_features=None` → regime skipped, uniform probs `[0.25, 0.25, 0.25, 0.25]`
- `current_portfolio=None` → portfolio constraints and circuit breaker skipped

### scorer.py — CompositeScorer
- 6 weighted sub-scores: technical_momentum, derivatives_sentiment, macro_alignment, microstructure_quality, news_sentiment, volatility_risk
- Feature columns matched by prefix: `deriv_*`, `micro_*`, `sent_*`, `hvol_*`/`natr_*`/`atr_*`/`bb_width_*`
- Missing feature groups default to 50 (neutral), weight redistributed to available groups
- Configurable weights via `CompositeScoreWeights` Pydantic config

### ranker.py — StockRanker
- Filters: is_tradeable, position_size > 0, no circuit-hit
- Produces `RankingResult` with separate buy/sell candidate lists sorted by composite_score desc
- Respects top_n limit
- Tracks excluded symbols with reasons
```

- [ ] **Step 2: Update risk CLAUDE.md**

Replace `alphavedha/risk/CLAUDE.md` with:

```markdown
# Risk Management — AlphaVedha

## Responsibility
Position sizing, portfolio constraints, and circuit breakers. Every prediction MUST pass through this layer.

## Modules

### position_sizing.py — Half-Kelly
- `compute_position_size(meta_confidence, magnitude, config) → float`
- Symmetric Kelly: `kelly = 2p - 1`, then half-Kelly `= kelly × 0.5 × 100` (as %)
- Returns 0.0 if: confidence < min_confidence, magnitude ≤ 0, negative Kelly
- Caps at `config.max_single_stock_pct` (default 10%)

### portfolio.py — PortfolioConstraints
- `PortfolioState` — holdings dict, total_value, peak_value
- `HoldingInfo` — symbol, sector, weight_pct, entry_date, correlation_60d, avg_daily_turnover_cr
- `ConstraintResult` — adjusted_weight_pct, violations list, passed bool
- Checks: sector cap (25%), correlation cap (0.7), min holding period (3d), liquidity (5 cr)
- Sells: checks min holding period. Buys: checks liquidity, correlation, sector cap

### circuit_breaker.py — CircuitBreaker
- `evaluate(current_value, peak_value) → CircuitBreakerState`
- Level 0: normal. Level 1 (10%): halve positions. Level 2 (15%): halt new entries. Level 3 (20%): close all
- Recovery: current_value ≥ peak × 0.95 → back to level 0
- `adjust_position(proposed, state, is_new_entry) → float`

### risk_manager.py — RiskManager
- Orchestrates: Kelly → portfolio constraints → circuit breaker
- `assess(meta_confidence, magnitude, symbol, sector, portfolio) → RiskAssessment`
- portfolio=None → Kelly only (single-stock mode, no constraints/CB)
- `RiskAssessment` includes kelly_raw, kelly_half, final position, violations, CB level
```

- [ ] **Step 3: Run full test suite one final time**

Run: `python -m pytest tests/ -v --tb=short`
Expected: All tests PASS

- [ ] **Step 4: Commit documentation**

```bash
git add alphavedha/prediction/CLAUDE.md alphavedha/risk/CLAUDE.md
git commit -m "docs: update prediction and risk CLAUDE.md with implementation details"
```
