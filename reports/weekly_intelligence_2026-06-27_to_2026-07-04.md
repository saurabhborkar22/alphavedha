# AlphaVedha Weekly Intelligence — June 27 to July 4, 2026

## Performance Summary

| Metric | Value |
|---|---|
| Trading days | 5 (Jun 27, 30, Jul 1, 2, 3) |
| Total predictions | 305 (67 unique symbols) |
| Evaluated | 0 (daily_pnl pipeline still empty) |
| Accuracy | N/A — no evaluations yet |
| Cumulative return | N/A |
| Best/worst day | N/A |
| Cumulative predictions (all-time) | 741 across 15 tracked days |
| Signal breakdown (all-time) | 64 UP / 677 DOWN / 0 HOLD |

**Critical gap:** `daily_pnl` has been at 0 rows for 2+ weeks. No prediction has been evaluated. All accuracy, return, Sharpe, and drawdown metrics remain null. This is the single highest-priority issue blocking the entire performance measurement pipeline.

---

## Sector Themes

### 1. Pharma Breakout — Leading Sector, Q1 Catalyst Ahead

Pharma was the strongest Nifty sector this week (+3.37% 7d momentum, RS=109.9, RRG phase: Leading). All 5 tracked Pharma stocks posted positive price action: MAXHEALTH +2.31%, APOLLOHOSP +2.27%, DRREDDY +2.11%, SUNPHARMA +1.81%, CIPLA +0.15%. The sector outperformed Nifty 50 by +3.19% on a 1-month basis (+6.89% vs +3.70% benchmark). SUNPHARMA goes ex-dividend July 7, which could create a short-term dip in the stock price but the sector rotation setup remains bullish. Catalyst: SUNPHARMA Q1 earnings on July 30.

### 2. IT Still Lagging but HCLTECH Divergence Signals Potential

IT remains the only sector in the RRG "Lagging" quadrant (RS=84.4, 1M return -6.6%, 3M return -10.4%). However, HCLTECH surged +5.65% on July 3, creating a notable single-stock divergence within an otherwise weak sector. Q1FY27 earnings season kicks off for IT next week: TCS (Jul 9), HCLTECH (Jul 13), WIPRO (Jul 16). If guidance is maintained or raised, the sector could rotate from Lagging to Improving. This week's 7d momentum (-0.40%) was better than last week's, suggesting the worst may be priced in.

### 3. Realty Leads the RRG — Multi-Month Outperformance

Nifty Realty holds the #1 RRG position (RS=113.0, 1M +5.66%, 3M +20.25%). This is a multi-month trend showing sustained relative strength with accelerating momentum. While no Nifty 50 Realty stocks are in our scan universe, the sector rotation signal suggests broader market risk appetite for cyclicals. Banking (#3, RS=105.1) and Infrastructure (#4, RS=104.4) also remain in the Leading quadrant, confirming a pro-cyclical regime despite the model's uniform bearish calls.

### 4. Energy Weakening — Metals Losing Momentum

Energy (-2.89% 1M) and Metals (-7.8% 1M) are both in the RRG Weakening quadrant. Energy's relative momentum (99.86) is decelerating. Within Metals, however, the 7d trend shows a slight bounce (+0.49%), with TATASTEEL +1.13% and ADANIENT +1.09%. This may be a dead-cat bounce or early reversal — watch for confirmation next week.

---

## Strategy Breakdown

### ensemble_v1 (ML Predictions) — 665 selections, 0 evaluated

The ensemble continues to produce uniform bearish signals: all 50 Nifty stocks predicted DOWN, with `cal_year` as the #1 feature for every single stock. Composite scores are compressed into a 0.7-point band (36.1–36.8), offering zero differentiation between stocks. Meta-confidence ranges from 0.17–0.53, with none crossing the 0.55 actionability threshold. **The XGBoost retrain on July 3 did not fix the calendar feature dominance** — this persists despite 5 retrains. The model is effectively outputting a calendar-driven constant signal rather than stock-specific predictions.

### event_drift_v1 — 60 selections, 0 evaluated

749 corporate disclosures and 66 credit rating events were ingested this week, providing rich event data. 27 earnings transcripts were also captured. However, with 0 evaluations, we cannot assess whether high-materiality events correctly predicted direction. The upcoming earnings season (TCS Jul 9, HCLTECH Jul 13, RELIANCE Jul 17, HDFCBANK Jul 18) will be the first real test for this strategy.

### guidance_delta_v1 — 16 selections, 0 evaluated

16 guidance-based signals were generated, likely from the 27 transcripts ingested. Cannot assess accuracy without daily_pnl evaluation data.

### blowup_short_v1 — NOT ACTIVE

This strategy is mentioned in the design but not yet implemented. The red-flags endpoint shows 0 symbols above the blowup threshold (70), so there would be no short candidates even if it were active.

### insider_cluster_v1 — NOT ACTIVE

This strategy is mentioned in the design but not yet implemented. The insider_trades pipeline was restored this week (31 rows ingested, up from 0 for 2+ months), which is a prerequisite for this strategy.

---

## Blowup Watch

| Category | Symbols |
|---|---|
| Currently on avoid list (score >= 70) | None |
| Approaching threshold (score 50-69) | No blowup-scores endpoint available — cannot monitor |
| Removed from avoid list this week | N/A |

The red-flags endpoint reports 0 flagged symbols at threshold 70. Without a `/intel/blowup-scores` endpoint, we cannot identify stocks trending toward the threshold. The 1,361 surveillance flags ingested this week (highest density period) suggest heightened exchange scrutiny, but we lack the endpoint to correlate these with individual stock blowup scores.

---

## Forward Watchlist

### Upcoming Events (Next 2 Weeks)

| Date | Symbol | Event |
|---|---|---|
| Jul 7 | SUNPHARMA | Ex-dividend |
| Jul 9 | TCS | Q1FY27 Earnings |
| Jul 9 | AXISBANK | Ex-dividend |
| Jul 13 | HCLTECH | Q1FY27 Earnings |
| Jul 16 | WIPRO | Q1FY27 Earnings |
| Jul 17 | RELIANCE | Q1FY27 Earnings |
| Jul 18 | HDFCBANK | Q1FY27 Earnings |
| Jul 18 | ICICIBANK | Q1FY27 Earnings |
| Jul 18 | AXISBANK | Q1FY27 Earnings |
| Jul 18 | KOTAKBANK | Q1FY27 Earnings |
| Jul 23 | INFY | Q1FY27 Earnings |
| Jul 23 | BAJFINANCE | Q1FY27 Earnings |
| Jul 28 | LT | Q1FY27 Earnings |
| Jul 29 | TATASTEEL | Q1FY27 Earnings |
| Jul 29 | NTPC | Q1FY27 Earnings |
| Jul 30 | SUNPHARMA | Q1FY27 Earnings |
| Jul 30 | — | F&O Monthly Expiry |

Q1FY27 earnings season starts in earnest next week. IT sector results (TCS, HCLTECH, WIPRO, INFY) will set the tone. Banking results on Jul 18 (4 major banks reporting same day) could drive significant sector moves.

### Model Staleness

| Model | Age (days) | Status |
|---|---|---|
| XGBoost | 0 | Fresh (retrained Jul 3) |
| LSTM | 6 | OK |
| TFT | 6 | OK |
| Regime (HMM) | 6 | OK |
| Ensemble | 6 | OK |
| Meta-labeling | 6 | OK |
| Conformal | 6 | OK |

All models are within acceptable age. LSTM/TFT/Regime/Ensemble should be retrained within the next week to stay under the 14-day freshness guideline.

### Data Pipeline Improvements This Week

- **insider_trades RECOVERED** — 31 rows ingested this week after 2+ months of being stale (was dead since April 28). Pipeline fix from PR #174 is now live.
- **prediction_proofs** — 5 new proofs added (up from 4 last week), pipeline continuing to function.
- **surveillance_flags** — 1,361 rows this week, highest weekly ingestion yet. Likely driven by F&O quarterly expiry + quarter-end rebalancing.
- **bulk_block_deals** — 1,363 rows, also elevated for the same reasons.

### Recurring Issues

| Issue | Weeks Open | Status |
|---|---|---|
| daily_pnl empty (0 rows) | 3+ | **P0 — BLOCKS ALL PERFORMANCE METRICS** |
| cal_year feature dominance | 3+ | P1 — persists despite XGBoost retrains |
| alt_data table missing | 2 | P2 — scraped macro data has nowhere to go |
| red-flags BASE_URL mismatch | 2 | P3 — works at /api/intel/red-flags, not /api/api/intel/red-flags |

---

## Macro Data Update (Scraped)

| Metric | Value | Period | Source | Verified |
|---|---|---|---|---|
| G-Sec 10Y Yield | 6.71% | Jul 3, 2026 | Trading Economics, Investing.com, Countryeconomy.com | Yes (3 sources) |
| PMI Manufacturing | 54.2 | Jun 2026 (final) | Trading Economics, The Federal | Yes (2 sources, revised down from 54.5 flash) |
| PMI Services | 57.4 | Jun 2026 (final) | Trading Economics, The Hans India | Yes (2 sources, revised up from 57.3 flash) |
| Auto Sales (PV) | ~400,000 | Jun 2026 | Tribune India, CarBikeGPT, Autocar India | Partial (exact SIAM total not yet published) |

### Macro Context

- **G-Sec 10Y at 6.71%** — significantly below the 7.0% hardcoded in the feature pipeline. Bond yields have declined from the 6.9-7.1% range due to foreign inflows (INR 324B since June), easing crude oil prices (~$70/bbl Brent), and potential Bloomberg Global Aggregate Index inclusion. The pipeline's hardcoded 7.0% is now 29 bps too high.
- **PMI Manufacturing at 54.2** — 3-month low, down from 55.0 in May. Softer new orders and weakest hiring pace in 6 months. Still expansionary but decelerating.
- **PMI Services at 57.4** — down from 58.9 in May (6-month high). Weakest since January 2025, but firmly expansionary. Services continues to outperform manufacturing.
- **Auto Sales ~400K** — +24.6% YoY, driven by income tax relief, repo rate cuts, and robust rural demand. Maruti leads with 150K domestic units, Mahindra at record 60K+ PV units.

**Storage note:** The `alt_data` table does not exist in the database. The `/ops/intel/push` endpoint only accepts 5 existing tables (disclosures, insider_trades, rating_events, surveillance_flags, transcripts). These scraped values cannot be persisted until an alt_data table is created.

---

## Infrastructure

| Component | Status |
|---|---|
| Database | Healthy |
| Redis | Healthy |
| Models loaded | Yes (v0.1.0) |
| Disk | 67.5% used (10.5 GB free) — severity: OK |
| Scheduler | Running (heartbeat 38s ago, PID 1, tier: large) |
| Data quality score | 100% (5,366 symbols covered) |
| Problems reported | 0 |
| Weekend mode | Active (July 4 is Saturday) |

All 10 core data tables are ingesting normally. No stale tables detected on any trading day this week. This is the first week since launch with zero data pipeline problems.

---

## Recommendations

### P0 — Critical

1. **Fix daily_pnl evaluation pipeline** — 0 rows after 3+ weeks. This blocks accuracy measurement, return calculation, Sharpe ratio, drawdown tracking, and strategy comparison. Without this, we are flying blind on model quality. The 15:45 evaluation job either isn't running or is failing silently. Investigate immediately.

### P1 — High Priority

2. **Address cal_year feature dominance in ensemble** — Despite 5+ XGBoost retrains, `cal_year` remains the #1 feature for all 50 stocks. The model produces identical bearish signals for every stock with no differentiation (0.7-point composite score range). Options: (a) cap cal_year importance, (b) remove calendar features from the feature set, (c) add a feature decorrelation step. This is rendering all ensemble predictions non-actionable.

3. **Update G-Sec 10Y yield from hardcoded 7.0% to live value** — The real yield is 6.71% (29 bps lower). This error propagates to every prediction that uses the G-Sec spread feature. Either create the alt_data table for storage, or update the hardcoded fallback value.

### P2 — Medium Priority

4. **Create alt_data table** — Needed to store scraped macro indicators (G-Sec yield, PMI, auto sales). Without it, macro data scraping is informational only — it doesn't feed the ML pipeline.

5. **Prepare for earnings season** — 14 Nifty 50 companies report Q1FY27 results between Jul 9-30. Ensure event_drift_v1 and guidance_delta_v1 capture and evaluate these events. This is the first major catalyst since system launch.

6. **Schedule LSTM/TFT/Regime retrain** — Currently 6 days old, approaching the 14-day freshness guideline. Retrain before the earnings season begins (before Jul 9).

### P3 — Low Priority

7. **Fix red-flags BASE_URL** — The endpoint works at `/api/intel/red-flags` but the routine's BASE_URL (`/api/api/...`) produces a 404. Same issue noted last week in PR #178. The double `/api` prefix needs correction either in the routine configuration or via a redirect.

8. **Implement blowup_short_v1 and insider_cluster_v1 strategies** — Both are in the design spec but not yet active. Insider trades data is now flowing (31 rows this week), enabling insider_cluster_v1 development.

9. **Monitor disk growth** — 67.5% (+5% from last week's 62.6%). At this rate, disk reaches 80% in ~2.5 weeks and 90% in ~4.5 weeks. Consider data retention policies or disk expansion.

---

## Fix PRs Created

None — no new code-level bugs were discovered this week. The red-flags path mismatch is a configuration issue in the routine prompt, not a code bug.

---

## Week-over-Week Comparison

| Metric | Last Week (Jun 20-27) | This Week (Jun 27-Jul 4) | Delta |
|---|---|---|---|
| Predictions | 256 | 305 | +49 (+19%) |
| Unique symbols | 54 | 67 | +13 (+24%) |
| insider_trades rows | 0 | 31 | +31 (RECOVERED) |
| prediction_proofs | 4 | 5 | +1 |
| daily_pnl rows | 0 | 0 | No change |
| Disk used | 62.6% | 67.5% | +4.9% |
| Model issues | 0 stale | 0 stale | Stable |
| Data pipeline problems | 1 (insider_trades) | 0 | FIXED |
| Surveillance flags | 1,501 | 1,361 | -140 |

---

*Generated: 2026-07-04T18:05:00+05:30*
*System: AlphaVedha v0.1.0 | Models: 7/7 healthy | Scheduler: running*
