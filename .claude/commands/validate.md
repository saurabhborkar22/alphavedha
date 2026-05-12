# Validate

Run the full validation suite — code quality, tests, and model validation.

## Usage
- `/validate` — run everything
- `/validate code` — lint + typecheck only
- `/validate tests` — pytest only
- `/validate model` — CPCV model validation only

## Steps

### Code Quality
1. `ruff check alphavedha/ tests/` — linting
2. `ruff format --check alphavedha/ tests/` — format check
3. `mypy alphavedha/ --strict` — type checking
4. Report any issues found

### Tests
1. `pytest tests/unit/ -v --tb=short` — unit tests
2. `pytest tests/integration/ -v --tb=short` — integration tests (requires DB)
3. `pytest tests/backtest/ -v --tb=short` — backtest validation tests
4. Report coverage summary

### Model Validation (if models exist)
1. Run CPCV on current active models
2. Compare live prediction accuracy vs backtest accuracy
3. Check for feature drift (PSI per feature group)
4. Check model freshness (days since last retrain)
5. Report any degradation warnings

### Data Integrity
1. Check for look-ahead bias in features
2. Verify corporate action adjustments are applied
3. Verify point-in-time universe compositions
4. Check for data gaps > 5 days

## Arguments
$ARGUMENTS
