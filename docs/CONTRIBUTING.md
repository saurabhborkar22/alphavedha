# Contributing to AlphaVedha

## Setup

```bash
git clone https://github.com/saurabhborkar22/alphavedha.git
cd alphavedha
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Branch Naming

| Prefix | Use |
|--------|-----|
| `feature/` | New functionality |
| `fix/` | Bug fixes |
| `data/` | Data pipeline changes |
| `model/` | ML model changes |
| `test/` | Test additions/fixes |
| `docs/` | Documentation only |

## Commit Messages

Use [Conventional Commits](https://www.conventionalcommits.org/):

```
feat: add LSTM multi-horizon prediction
fix: handle missing delivery_pct in OHLCV store
data: add BSE corporate actions provider
model: tune XGBoost hyperparameters for Nifty 50
test: add regression test for circuit limit detection
docs: update deployment guide with backup steps
```

## Code Style

- **Python 3.12+** with strict type hints
- **Ruff** for linting and formatting (`ruff check`, `ruff format`)
- **Mypy** for type checking
- Line length: 100 characters
- `from __future__ import annotations` in every file

```bash
# Run linter
ruff check alphavedha/ tests/

# Auto-format
ruff format alphavedha/ tests/

# Type check
mypy alphavedha/ --ignore-missing-imports
```

## Testing

```bash
# Unit tests (fast, no external deps)
pytest tests/unit/ -v

# Full suite with coverage
pytest tests/unit/ -v --cov=alphavedha --cov-report=term-missing
```

### Rules

- Unit tests: mock ALL external APIs (yfinance, NSE, Finnhub)
- Integration tests: use real PostgreSQL/TimescaleDB (never mock the DB)
- Every bug fix must include a regression test
- Use `pytest-asyncio` with mode `auto`
- Target: 80% line coverage minimum

## PR Process

1. Create a feature branch from `main`
2. Make changes, add tests
3. Run `ruff check` + `ruff format --check` + `pytest tests/unit/`
4. Push and create PR against `main`
5. CI must pass (lint + typecheck + tests + security audit)
6. Get review, address feedback
7. Maintainer merges

## Financial Data Rules

These are non-negotiable:

- ALL timestamps must be timezone-aware (`Asia/Kolkata`)
- NEVER use random train/test splits for time-series
- ALWAYS use point-in-time data (no look-ahead bias)
- Corporate actions MUST be adjusted before feature computation
- Circuit-hit days: flag, don't drop
- Include delisted stocks (survivorship bias protection)
- Minimum 20-day embargo between train and validation sets
