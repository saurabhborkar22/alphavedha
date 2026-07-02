# Claude Code Routines — AlphaVedha

Cloud-hosted scheduled agents running on Anthropic infrastructure, configured via [claude.ai/code/routines](https://claude.ai/code/routines). These routines monitor the AlphaVedha VPS, process intelligence, produce analyst briefings, and can create fix PRs for code-level issues.

## Schedule Overview

| # | Routine | Schedule | Purpose |
|---|---|---|---|
| 1 | `alphavedha-pre-market-check` | Weekdays 07:30 IST | Verify morning ML pipeline ran |
| 2 | `alphavedha-morning-briefing` | Weekdays 06:30 IST | Senior analyst review of predictions |
| 3 | `alphavedha-intel-processor` | Weekdays 19:30 IST | Classify corporate disclosures |
| 4 | `alphavedha-post-ingestion-check` | Weekdays 21:00 IST | Verify evening data pipeline |
| 5 | `alphavedha-weekly-summary` | Saturday 10:00 IST | Weekly performance report |
| 6 | `alphavedha-weekly-intel` | Saturday 18:00 IST | Sector themes + macro data scraping |

## Daily Timeline (Weekdays)

```
06:00  ML pipeline runs (scheduler: predictions, signals, hash)
06:30  [Routine 2] Morning briefing — analyst reviews predictions
07:30  [Routine 1] Pre-market check — verifies pipeline health
09:15  Market opens
15:30  Market closes
17:00  Data refresh job (scheduler: OHLCV)
18:30  FII/DII ingestion (scheduler)
19:00  Announcements ingestion (scheduler)
19:20  Insider trades ingestion (scheduler)
19:30  [Routine 3] Intel processor — classifies disclosures
       Surveillance, deals, ratings, transcripts (scheduler: 19:30-20:30)
21:00  [Routine 4] Post-ingestion check — verifies data pipeline
```

## VPS Connection Details

- **VPS IP**: `178.105.237.174` (Hetzner CX23, Germany)
- **API Key Header**: `X-API-Key: <ALPHAVEDHA_API_KEY>` — the actual key lives ONLY in
  `.env.vps` on the server and in each routine's configuration on claude.ai. Never
  commit it: this repo is public, and a key that lands in git history stays there
  even after the file is edited (rotate immediately if that happens).

### API URL Prefix Rules

The VPS has nginx proxying `/api` to FastAPI. Different routers have different prefix patterns:

| Router | FastAPI Prefix | Full URL Pattern | Example |
|---|---|---|---|
| Ops | `/api/ops` | `/api/api/ops/...` (double `/api`) | `/api/api/ops/health` |
| Predictions | none | `/api/...` (single `/api`) | `/api/predict/TCS` |
| Scan | none | `/api/...` (single `/api`) | `/api/scan/large` |
| Intel (red-flags) | `/intel` | `/api/intel/...` (single `/api`) | `/api/intel/red-flags` |

## Branch Naming Convention

All routines create fix PRs on branches prefixed with `claude/fix/` (e.g., `claude/fix/regime-shape-mismatch`). This is required because the routines UI only allows pushing to `claude/` prefixed branches by default.

## Fix/PR Capability

All 6 routines share the same fix procedure and guardrails:

### What routines CAN fix
- Python code bugs in `alphavedha/` (logic errors, wrong field names, missing None checks, type errors)
- Parser mismatches when external APIs change field names
- Missing imports, wrong function signatures, off-by-one errors
- Event loop / async session issues in scheduler.py

### What routines CANNOT fix (report only)
- External API deprecations or URL changes
- Database schema or migration issues
- Docker/compose/CI configuration
- Model training code or hyperparameters
- Environment variables or secrets
- Any issue requiring SSH into the VPS

### Fix procedure (shared by all routines)
1. Check for existing fix PRs: `gh pr list --state open --search "fix" --limit 20`
2. Read `CLAUDE.md`, the failing code (full function), and corresponding tests
3. Confirm root cause with evidence: WHAT fails, WHY, EVIDENCE
4. Quality gate: all 5 questions must be YES before writing code
5. Write MINIMUM fix — no refactoring, no new deps, no schema changes
6. Verify: `ruff format` + `ruff check` + `mypy` + `pytest tests/unit/ -x -q`
7. Create PR on `claude/fix/<name>` branch with root cause in body
8. NEVER merge — owner reviews and merges all PRs

### Absolute rules (all routines)
- NEVER SSH into VPS, restart containers, or modify Docker state
- NEVER modify data directly in the database
- NEVER `git push --force` or commit to main
- NEVER merge any PR
- NEVER modify `.env`, `docker-compose`, or CI files
- NEVER modify model training code or hyperparameters
- NEVER suggest going live with real money

---

## Routine Details

### 1. `alphavedha-pre-market-check`

**Schedule**: Weekdays 07:30 IST
**Purpose**: Verify the 06:00 ML prediction pipeline ran correctly before market open (09:15).

**Endpoints used**:
- `GET /api/api/ops/health` — system health + table staleness
- `GET /api/api/ops/predictions/summary` — prediction count, hash status, directions
- `GET /api/api/ops/scheduler/status` — scheduler running, heartbeat age

**Checks**:
- Predictions count > 0 (expect ~50 for Nifty 50)
- Hash published == true
- Health status != critical, database + models loaded
- Scheduler running, heartbeat < 300s
- Problems array entries

**Issue classification**: transient (just restarted), data (weekend/holiday), or code (fixable).

---

### 2. `alphavedha-morning-briefing`

**Schedule**: Weekdays 06:30 IST
**Purpose**: Senior analyst reviews every ML prediction against qualitative context. Produces structured briefing with per-stock assessments, analyst adjustments, cross-stock patterns, and risk flags.

**Endpoints used**:
- `GET /api/scan/large` — all Nifty 50 predictions (single `/api` prefix)
- `GET /api/api/ops/predictions/summary` — prediction stats
- `GET /api/intel/red-flags` — blowup avoid list (single `/api` prefix)
- `GET /api/api/ops/health` — table freshness
- `GET /api/api/ops/tables/deltas` — row count deltas

**Output structure**:
- Top 10 stocks: detailed analyst review with reasoning, adjustment, conviction, key risk
- Disagreements: explicit override with evidence
- Remaining 40: one-liner assessment
- Cross-stock patterns: sector concentration, FII/DII alignment, event clustering, regime context

**Key principle**: The analyst reviews ML predictions — does NOT replace them. Adjustments require qualitative evidence the model cannot see (disclosures, governance red flags, sector patterns).

---

### 3. `alphavedha-intel-processor`

**Schedule**: Weekdays 19:30 IST
**Purpose**: Financial analyst classifies unprocessed corporate disclosures (NSE/BSE filings) into structured events with direction, materiality, and confidence scores.

**Endpoints used**:
- `GET /api/api/ops/intel/pending?limit=100` — unprocessed disclosures
- `POST /api/api/ops/intel/events` — store classified events

**Event types** (21 categories): `order_win`, `capacity_expansion`, `results_guidance`, `guidance_cut`, `fund_raise`, `m_and_a`, `rating_upgrade`, `rating_downgrade`, `outlook_change`, `pledge_increase`, `pledge_release`, `insider_buy`, `insider_sell`, `auditor_resignation`, `kmp_resignation`, `related_party_txn`, `litigation_regulatory`, `default_or_delay`, `surveillance_action`, `dividend_buyback`, `other`.

**Output per disclosure**: event_type, direction (+1/-1/0), materiality (0-10), confidence, summary (<200 chars), red_flags.

**Classification rules**: Conservative materiality (most filings are 2-4, not 7-10). `kmp_resignation` only for CFO/CS/CEO/MD. SEBI Reg 29/31 = insider trades. Boilerplate filings = `other` with materiality 0.

---

### 4. `alphavedha-post-ingestion-check`

**Schedule**: Weekdays 21:00 IST
**Purpose**: Verify the evening data pipeline completed (OHLCV, FII/DII, disclosures, surveillance, deals, ratings, transcripts).

**Endpoints used**:
- `GET /api/api/ops/health` — table staleness
- `GET /api/api/ops/tables/deltas` — today vs yesterday row counts
- `GET /api/api/ops/tables/counts` — total counts
- `GET /api/api/ops/scheduler/status` — scheduler heartbeat
- `POST /api/api/ops/trigger/data_refresh` — re-trigger if OHLCV stale
- `POST /api/api/ops/trigger/fii_dii` — re-trigger if flows stale

**Expected row counts on trading days**:
- `daily_ohlcv`: ~20-50 new rows
- `institutional_flows`: > 0
- Other tables (insider_trades, surveillance, deals, ratings, transcripts): 0 is normal — not every day has events

**Can trigger re-runs**: If `daily_ohlcv` or `institutional_flows` stale on a trading day, attempts `POST /trigger/data_refresh` or `/trigger/fii_dii`.

---

### 5. `alphavedha-weekly-summary`

**Schedule**: Saturday 10:00 IST
**Purpose**: Comprehensive weekly performance report covering predictions accuracy, data ingestion, model health, infrastructure, and recommendations.

**Endpoints used**:
- `GET /api/api/ops/weekly/report` — week's prediction accuracy + P&L
- `GET /api/api/ops/models/status` — model ages and metrics
- `GET /api/api/ops/health` — system health
- `GET /api/api/ops/tables/counts` — total data volumes
- `GET /api/api/ops/predictions/summary` — latest predictions
- `GET /api/api/ops/scheduler/status` — scheduler uptime

**Report sections**: Performance (accuracy, returns, best/worst day), Predictions, Data Ingestion (per-table row counts), Model Health (per-model age/status), Infrastructure (disk, DB, Redis, scheduler), Recommendations (retrain alerts, data gaps, trends).

**Recurring issue detection**: Checks for problems that appeared 3+ trading days in the week — those indicate systematic code bugs worth fixing.

---

### 6. `alphavedha-weekly-intel`

**Schedule**: Saturday 18:00 IST
**Purpose**: Two-part routine: (A) synthesize week's events into sector themes, strategy narratives, blowup watch, and forward watchlist; (B) scrape public data sources to fill stub features.

**Endpoints used**:
- All ops endpoints (weekly/report, health, tables/counts, tables/deltas, models/status, predictions/summary, scheduler/status)
- `GET /api/intel/red-flags` — blowup avoid list (single `/api` prefix)

**Part A — Intelligence Synthesis**:
- Sector theme detection (group by 11 sectors, flag 3+ stocks same direction for 3+ days)
- Strategy performance narratives (ensemble, blowup_short, event_drift, insider_cluster, guidance_delta)
- Forward watchlist (upcoming events, approaching blowup thresholds, model staleness, recurring data gaps)

**Part B — Stub Data Scraping**:
Currently 16 ML features return NaN or hardcoded values. This routine scrapes public sources for:
- **G-Sec 10Y Yield** — currently hardcoded to 7.0 in `features/macro.py`
- **PMI Manufacturing** — currently NaN
- **PMI Services** — currently NaN
- **Auto Sales (monthly)** — currently requires manual entry

Sources: RBI weekly supplement, CCIL, S&P Global PMI press releases, SIAM.

---

## Known Issues Affecting Routines

1. **insider_trades stale since Apr 28** — NSE PIT API returns empty data (likely due to NSE blocking Hetzner's German IP / cookie auth failure). See `alphavedha/data/providers/sebi_provider.py:191-267`. The `NSESession` at `nse_provider.py:25-57` gets 403 on cookie refresh.

2. **`pump_score = 0` permanently** — `alphavedha/intel/signals/blowup_score.py:95` has a hardcoded 0 placeholder, never implemented. Blowup score only uses 7 of 8 intended components.

3. **`dropped_commitments` extracted but unused** — The intel processor extracts this field from transcript deltas but nothing downstream consumes it.

4. **`/api/api/ops/intel/push` endpoint** — Referenced in the weekly-intel routine for storing scraped macro data. May not exist yet; routine will note this and report values in the briefing instead.
