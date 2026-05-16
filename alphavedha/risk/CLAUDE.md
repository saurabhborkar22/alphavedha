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
