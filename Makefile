.PHONY: setup docker-up docker-down lint test test-unit test-integration test-integration-up test-integration-down test-backtest serve train predict scan backtest validate data-refresh data-backfill coverage

VENV := .venv
PYTHON := $(VENV)/bin/python
PIP := $(VENV)/bin/pip

# ─── Setup ───────────────────────────────────────────────────
setup:
	python3.12 -m venv $(VENV)
	$(PIP) install -e ".[dev]"
	$(PYTHON) -m pre_commit install || true
	@echo "Setup complete. Activate venv: source $(VENV)/bin/activate"

docker-up:
	docker compose up -d

docker-down:
	docker compose down

# ─── Development ─────────────────────────────────────────────
lint:
	$(VENV)/bin/ruff check alphavedha/ tests/
	$(VENV)/bin/ruff format --check alphavedha/ tests/
	$(VENV)/bin/mypy alphavedha/

format:
	$(VENV)/bin/ruff check --fix alphavedha/ tests/
	$(VENV)/bin/ruff format alphavedha/ tests/

test:
	$(VENV)/bin/pytest tests/ --cov=alphavedha --cov-report=term-missing

test-unit:
	$(VENV)/bin/pytest tests/unit/ -v

test-integration:
	$(VENV)/bin/pytest tests/integration/ -v -m integration

test-backtest:
	$(VENV)/bin/pytest tests/backtest/ -v -m backtest

test-integration-up:
	docker compose -f docker-compose.test.yml up -d
	@echo "Waiting for test database..." && sleep 3

test-integration-down:
	docker compose -f docker-compose.test.yml down

coverage:
	$(VENV)/bin/pytest tests/unit/ --cov=alphavedha --cov-report=term-missing --cov-report=html
	@echo "HTML report: htmlcov/index.html"

# ─── Data ────────────────────────────────────────────────────
data-refresh:
	$(PYTHON) -m alphavedha.cli.main data refresh

data-backfill:
	$(PYTHON) -m alphavedha.cli.main data backfill --start 2005-01-01

# ─── ML Training ─────────────────────────────────────────────
train:
	$(PYTHON) -m alphavedha.cli.main train all

train-xgboost:
	$(PYTHON) -m alphavedha.cli.main train xgboost

train-lstm:
	$(PYTHON) -m alphavedha.cli.main train lstm

train-tft:
	$(PYTHON) -m alphavedha.cli.main train tft

train-regime:
	$(PYTHON) -m alphavedha.cli.main train regime

train-meta:
	$(PYTHON) -m alphavedha.cli.main train meta

validate:
	$(PYTHON) -m alphavedha.cli.main validate model

# ─── Prediction ──────────────────────────────────────────────
predict:
	$(PYTHON) -m alphavedha.cli.main predict $(SYMBOL)

scan:
	$(PYTHON) -m alphavedha.cli.main scan $(TIER)

# ─── API ─────────────────────────────────────────────────────
serve:
	$(VENV)/bin/uvicorn alphavedha.api.app:app --reload --host 0.0.0.0 --port 8000

serve-prod:
	$(VENV)/bin/gunicorn alphavedha.api.app:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000

# ─── Backtest ────────────────────────────────────────────────
backtest:
	$(PYTHON) -m alphavedha.cli.main backtest portfolio

# ─── VPS Deployment ──────────────────────────────────────────
VPS_COMPOSE = docker compose -f docker-compose.vps.yml --env-file .env.vps

vps-up:
	$(VPS_COMPOSE) up -d --build

vps-down:
	$(VPS_COMPOSE) down

vps-restart:
	$(VPS_COMPOSE) restart

vps-logs:
	$(VPS_COMPOSE) logs -f --tail=100

vps-status:
	$(VPS_COMPOSE) ps

vps-migrate:
	$(VPS_COMPOSE) exec api alembic upgrade head

vps-shell-api:
	$(VPS_COMPOSE) exec api bash

vps-shell-db:
	$(VPS_COMPOSE) exec postgres psql -U alphavedha -d alphavedha
