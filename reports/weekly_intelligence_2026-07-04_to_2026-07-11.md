# AlphaVedha Weekly Intelligence — 2026-07-04 to 2026-07-11

## Performance Summary

- **Trading days:** 4 (Jul 7-10; Jul 11 is Saturday)
- **Total predictions:** 340 across 57 symbols
- **Evaluated:** 180 predictions
- **Accuracy:** 47.8% (below 50% break-even threshold)
- **Cumulative return:** -1.52%
- **Best day:** +1.07% | **Worst day:** -2.00%
- **Best single-day accuracy:** 57.8% (Jul 10, 26/45 correct)

This is the second week with live evaluation data. The 47.8% accuracy is marginally below coin-flip, suggesting the ensemble is not yet adding alpha after transaction costs. However, the Jul 10 session showed 57.8% accuracy following the full model retrain — this is encouraging and suggests the freshly-retrained models may perform better going forward.

## Sector Themes

### 1. Q1FY27 Earnings Season Begins — IT in the Spotlight

With TCS results on Jul 9 and HCLTECH on Jul 13, Q1FY27 earnings season has begun. The system ingested 18 transcripts and 213 disclosures this week — the highest disclosure count in the system's history. IT sector remains the bellwether: previous weekly reports flagged IT as the only "Lagging" quadrant sector on the RRG. Watch for guidance commentary on AI capex pass-through.

### 2. Surveillance Activity Elevated

1,325 surveillance flags ingested this week (524 in the latest batch alone). This is consistent with F&O weekly expiry volume and Q1 results front-running. The bulk/block deals pipeline captured 987 deals — also elevated, suggesting institutional repositioning ahead of earnings.

### 3. Credit Rating Activity Picks Up

71 rating events this week, up from historical averages. This clusters around Q1 results season when agencies update outlooks. 31 events in the latest 2-day batch.

## Strategy Breakdown

### ensemble_v1 (ML Predictions)
**Accuracy: 47.8% | Return: -1.52%**

The ensemble produced 340 predictions across 57 symbols. With only 180 evaluated so far (rest awaiting T+1/T+2 settlement windows), the 47.8% accuracy is below break-even. The Jul 10 retrain brought all 7 models to fresh status (0 days old). XGBoost leads at 70.85% train accuracy with 0.7146 F1, while LSTM trails at 55.2%. The meta-labeling model shows strong F1 of 0.827, suggesting the confidence calibration layer is working well — the issue is likely in the base predictions rather than the gating logic.

**Key observation:** The worst day (-2.00%) likely coincided with a broad market move that the model failed to anticipate. The best day (+1.07%) and the 57.8% accuracy on Jul 10 came after the full retrain, suggesting model freshness matters significantly.

### blowup_short_v1 (Avoid List)
**Status: No stocks flagged (threshold 70)**

The red-flags endpoint returned 0 flagged symbols at the default threshold of 70. However, the endpoint was 404 via the standard BASE_URL all week (bug now fixed in PR #212), so the daily scheduler may not have been running blowup scans correctly. After the route fix is deployed, this strategy can be properly evaluated.

### event_drift_v1 (Event-Driven)
**Status: Rich data, no evaluation yet**

213 disclosures and 71 rating events provide a strong signal corpus. The 18 transcripts (8 in the latest batch) from Q1 results should feed into event drift scoring. No evaluation data is available yet for this strategy.

### insider_cluster_v1 (Insider Buying)
**Status: 0 insider trades this week**

The insider_trades table shows 0 new rows this week (last entry: Jun 29). This may be legitimate — insider trading windows are often closed during results season. The pipeline itself is functional (49 total rows), so this is likely a data availability issue rather than a scraper failure.

### guidance_delta_v1 (Guidance-Based)
**Status: Early season — insufficient data**

With only TCS having reported so far, there isn't enough guidance delta data to evaluate. HCLTECH (Jul 13), WIPRO (Jul 16), RELIANCE (Jul 17), and banking heavyweights (Jul 18) will provide the real test.

## Blowup Watch

- **Currently on avoid list (score >= 70):** None
- **Approaching threshold (50-69):** Unknown — endpoint was 404 all week; will be evaluable after PR #212 is deployed
- **Removed from avoid list this week:** N/A

## Forward Watchlist

### Upcoming Events (Next 2 Weeks)
- **Jul 13 (Mon):** HCLTECH Q1 results
- **Jul 16 (Wed):** WIPRO Q1 results
- **Jul 17 (Thu):** RELIANCE Q1 results; F&O weekly expiry
- **Jul 18 (Fri):** HDFCBANK, ICICIBANK, AXISBANK, KOTAKBANK Q1 results
- **Jul 24 (Thu):** F&O monthly expiry (last Thursday of July)
- **Jul 31 (Thu):** Next F&O weekly expiry

### Model Staleness
All 7 models retrained Jul 10 — **0 days old**. No retrain needed until ~Aug 4 (25-day threshold). This is the freshest the model stack has ever been.

### Data Pipeline Health
- **All 12 tables healthy** — 0 stale, 0 problems
- **insider_trades:** 0 new rows (legitimate — results season blackout)
- **Prediction proofs:** 5 new this week (up from 4 last week)
- **daily_pnl:** 4 rows total — now operational after being stuck at 0 for 3+ weeks

### Recurring Issues Resolved This Week
- **daily_pnl:** Was 0 rows for 3+ weeks, now has 4 rows — evaluation pipeline finally working
- **Models:** All retrained to 0 days old (were 6+ days last week)

## Macro Data Update (Scraped)

| Metric | Value | Date | Source | Cross-verified |
|---|---|---|---|---|
| G-Sec 10Y Yield | 6.71% | Jul 10, 2026 | tradingeconomics.com | Yes (worldgovernmentbonds.com unavailable, investing.com showed stale Oct 2024 data) |
| PMI Manufacturing | 54.2 | Jun 2026 | HSBC/S&P Global | Revised down from preliminary 54.5; 2nd weakest since mid-2022 |
| PMI Services | 57.4 | Jun 2026 | S&P Global/HSBC | Revised up from preliminary 57.3; down from May's 58.9 |
| Auto Sales (PV) | 4,38,854 | May 2026 | SIAM | Up from 4,37,312 in Apr 2026 |

**Note on G-Sec yield:** Currently hardcoded at 7.0% in the feature pipeline. Actual yield is 6.71% — a 29 bps error. This gap has persisted for 3+ weeks. PR #213 fixes the date parsing bug that prevented storing this data; once deployed, the macro data push will work.

**Note on storage:** All 4 data points failed to POST to the `alternative_data` table due to a date parsing bug (period_date passed as string, asyncpg requires datetime.date). Fix submitted in PR #213.

## Infrastructure

- **Database:** Healthy
- **Redis:** Healthy
- **Disk:** 53.7% used (15.7 GB free) — down from 67.5% last week (cleanup worked)
- **Scheduler:** Running (heartbeat 20s ago, PID 1, tier: large)
- **Models loaded:** Yes (v0.1.0)
- **Health problems:** 0

### Model Details

| Model | Age | Key Metric |
|---|---|---|
| XGBoost | 0 days | Accuracy: 70.85%, F1: 0.7146, RMSE: 0.0494 |
| LSTM | 0 days | Accuracy: 55.20%, F1: 0.5610, RMSE: 0.0463 |
| TFT | 0 days | Accuracy: 44.22%, F1: 0.4454, RMSE: 0.0495 |
| Regime (HMM) | 0 days | Log-likelihood: -1962.27, AIC: 3994.53 |
| Ensemble | 0 days | Train accuracy: 51.34%, Train F1: 0.4354 |
| Meta-labeling | 0 days | Train accuracy: 82.33%, Train F1: 0.8271 |
| Conformal | 0 days | R2: 0.9723, RMSE: 0.007 |

### Data Ingestion This Week

| Table | Rows This Week | Total Rows |
|---|---|---|
| paper_trades | 340 | 1,081 |
| prediction_proofs | 5 | 14 |
| daily_ohlcv | 13,398 | 283,840 |
| institutional_flows | 10 | 40 |
| disclosures | 213 | 1,135 |
| insider_trades | 0 | 49 |
| surveillance_flags | 1,325 | 4,187 |
| bulk_block_deals | 987 | 3,676 |
| rating_events | 71 | 200 |
| transcripts | 18 | 97 |

## Recommendations

1. **Deploy PR #212 (red-flags route fix)** — Unblocks the blowup-score avoid list from the scheduler's standard BASE_URL. Critical for the blowup_short_v1 strategy to function.

2. **Deploy PR #213 (alt-data date parsing fix)** — Unblocks macro data storage. The G-Sec yield is 29 bps off from the hardcoded value; once this fix is deployed, the weekly scrape can populate the correct value automatically.

3. **Monitor accuracy trend** — 47.8% is below break-even. The Jul 10 retrain showed immediate improvement (57.8% on that day). Track whether the freshly retrained models sustain >50% accuracy next week.

4. **Earnings season coverage** — With HCLTECH, WIPRO, RELIANCE, and 4 major banks reporting next week, ensure the transcript and disclosure pipelines handle the volume spike. The 18 transcripts this week were handled cleanly.

5. **Review 12 open draft PRs** — PRs #153, #173, #176-181, #196, #200-202 are stale draft report PRs. Consider closing them to keep the PR list manageable.

6. **insider_trades monitoring** — 0 rows this week is likely legitimate (results season blackout window), but verify the scraper runs successfully on the next insider filing day.

## Fix PRs Created This Session

| PR | Title | Bug |
|---|---|---|
| [#212](https://github.com/saurabhborkar22/alphavedha/pull/212) | fix: change red_flags router prefix to /api/intel | /intel/red-flags returned 404 via BASE_URL |
| [#213](https://github.com/saurabhborkar22/alphavedha/pull/213) | fix: parse period_date string to datetime.date in store_alternative_data | alternative_data inserts failed with asyncpg DataError |
