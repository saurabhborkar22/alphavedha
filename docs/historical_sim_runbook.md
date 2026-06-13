# Historical Simulation — Runbook

One-time job that produces an **honest, out-of-sample** 6-month track record +
walk-forward backtest and shows it on the UI (paper page = 3-track record,
backtest page = equity/Sharpe/drawdown/monthly). No live behaviour changes:
endpoints serve zeros until the artifact exists.

## What was built (all on branch `feat/p0-verifiable-record`)

| File | Change | Live impact |
|---|---|---|
| `alphavedha/services/prediction_service.py` | optional `as_of` date (defaults today) | none (no-op unless passed) |
| `alphavedha/training/pipeline.py` | `end_date` cutoff on `train_all`/`train_xgboost`/`_load_tier_data`; `ALPHAVEDHA_ARTIFACTS_DIR` override | none (defaults unchanged) |
| `alphavedha/backtest/sim_views.py` | pure track-record + walk-forward view builders | additive |
| `scripts/sim_paper_trading.py` | the one-shot runner | additive, run manually |
| `alphavedha/api/sim_artifact.py` | mtime-cached artifact loader | additive |
| `alphavedha/api/routes/ui_support.py` | `/backtest/*` serve artifact, else zeros | none until artifact exists |
| `alphavedha/api/routes/paper_trading.py` | new `/paper/simulation` endpoint | additive |
| `alphavedha-ui` (paper page, api types/client) | labeled "Historical Simulation" panel | additive |

## Why it's out-of-sample (the whole point)

The live models trained through 2026-06-10, so replaying them over the last 6
months would be in-sample (the model already saw those outcomes). This job
trains a **frozen model with a cutoff** — it never sees data after `--cutoff` —
then replays each day via the real engine with `as_of=T`. That is a true
forward test, just on historical data.

## Gates (your hands required)

- **Merge** the code PR — CI deploys the backend. *(no-auto-commit / no-pr-merge)*
- **Compute**: full LSTM/TFT training wants the CX43 (16 GB); the CX23 (4 GB)
  may OOM on the deep models. Scaling the VPS needs your OK. *(Hetzner guardrail)*
- **UI deploy** is manual (no CI for `alphavedha-ui`).

---

## Step 1 — Merge the code (safe, no visible change)

Review the PR on `feat/p0-verifiable-record`, merge to `main`. CI deploys.
Verify nothing changed: backtest page still shows zeros, paper page unchanged
(the simulation panel only renders when the artifact exists).

## Step 2 — Generate the artifact on the VPS

Pick a cutoff ~6 months before the data end (data ends 2026-06-10):

```bash
# On the VPS, in /opt/alphavedha. Optional: scale to CX43 first for LSTM/TFT.
# Smoke test the wiring fast (train once, replay 5 days):
docker compose -f docker-compose.vps.yml --env-file .env.vps run --rm trainer \
  python -m scripts.sim_paper_trading --cutoff 2025-12-10 \
  --sim-dir /app/models/artifacts_sim --out /app/alphavedha/api/sim_artifact.json \
  --max-days 5

# Full run (reuse the frozen model from the smoke test with --no-train):
docker compose -f docker-compose.vps.yml --env-file .env.vps run --rm trainer \
  python -m scripts.sim_paper_trading --cutoff 2025-12-10 --no-train \
  --sim-dir /app/models/artifacts_sim --out /app/alphavedha/api/sim_artifact.json
```

Expected tail: `✅ Wrote …/sim_artifact.json  trades=… days=… cutoff=2025-12-10`
plus a one-line strategy summary (cagr / sharpe / maxDD / win).

> The frozen model lands in `--sim-dir` (isolated); the live `latest` models and
> the `paper_trades`/`daily_pnl` tables are never touched.

## Step 3 — Publish the artifact

The API reads `alphavedha/api/sim_artifact.json` from its image. Two options:

- **Commit + redeploy (simplest):** copy the JSON out of the container, commit it
  to the repo, push → CI rebuilds → API serves it.
  ```bash
  docker cp <trainer-container>:/app/alphavedha/api/sim_artifact.json \
    /opt/alphavedha/alphavedha/api/sim_artifact.json
  # commit + push from your machine, then let CI redeploy
  ```
- **Mount + restart (re-runnable):** add a bind mount for the file + set
  `ALPHAVEDHA_SIM_ARTIFACT` in compose, then `restart api` after each run.

## Step 4 — Deploy the UI (manual)

```bash
cd /opt/alphavedha/alphavedha-ui && git pull
docker compose --env-file ../.env.vps -f docker-compose.vps.yml build ui && \
docker compose --env-file ../.env.vps -f docker-compose.vps.yml up -d ui
```

## Step 5 — Verify on production

- `GET /api/backtest/summary` → non-zero `total_trades`, real `date_from/to`.
- `GET /api/paper/simulation` → `available: true`, 3 tracks populated.
- UI: backtest page populated; paper page shows the **"Historical Simulation —
  Out-of-Sample · BACKFILLED · NOT LIVE"** panel.

## Re-running / cleanup

- Re-run with a different `--cutoff` (e.g. to extend the window) and re-publish.
- This feature is fully removable: delete `sim_artifact.json` (endpoints revert
  to zeros), and the seams are no-ops with no caller.

## Reading the numbers honestly

- Trust **gate-passed net** (strategy, after costs) over gross/accuracy.
- Cohorts overlap (15-day holds) → Sharpe is optimistic early; judge on the
  full window.
- Macro/derivatives/delivery features are dead (NaN→0) — same handicap as live.
- Base models are weak (XGBoost ~0.46, ensemble ~0.53 on 3-class) — expect a
  modest result; the value is that it's *honest*.
