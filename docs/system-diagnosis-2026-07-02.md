# System Diagnosis — 2026-07-02

> Full end-to-end audit of the live VPS, database, scheduler, verification
> pipeline, routines, and UI coverage — run after the edge-factory build
> landed but the system felt "not impressive". Every finding below was
> verified against the live system (logs, DB queries, API responses), not
> inferred from code.

## Verdict in one paragraph

The plumbing works — containers healthy, jobs fire on time, data lands —
but the four outcomes that make the system worth running were all silently
broken: (1) the ensemble collapsed to predicting SHORT on everything with
magnitudes far below the cost hurdle, so the paper record accrues nothing
meaningful; (2) the "provably honest record" never published a single
proof; (3) two of five strategies never fire because their upstream data
is dead; (4) most of what was built is invisible in the UI.

## Findings

### F1 — Ensemble degenerate: 100% short since day one · CRITICAL

Every `ensemble_v1` cohort since 2026-06-16 is all-short (45–50 shorts,
0 longs, 0 tradeable). Confidence/|magnitude| degraded with each retrain:
0.52/0.64% (Jun 16) → 0.44/0.19% (Jun 22) → 0.26/0.05% (Jun 29+). Train
accuracy 0.443 equals the −1 label frequency — a majority-class collapse.
Causes: asymmetric triple-barrier labels over-produce −1 (lower barrier
1.5×ATR vs 2.0× upper), no class weights, nightly `xgboost_retrain`
overwrites the artifact with no quality gate, and the nightly path loads
no macro data (feature-distribution skew vs weekly pipeline + serving).
LSTM hits 0.54 val accuracy on the same data — signal exists.
**Fix: PR #185** (class balance + macro parity + promotion gate).

### F2 — Verifiable record never verifiable · CRITICAL

`prediction_hash` fails daily: `Permission denied:
'/app/alphavedha-proofs/proofs'` (root-owned volume, non-root app user).
Hashes land only in the DB; the GitHub proofs repo has one commit (init,
Jun 22); `git_commit`/`ots_path` NULL on every row; OTS stamping and the
P0-D2 reveal job were never implemented; ops reported `published: true`
from the DB row alone, so the pre-market routine saw green.
**Fix: PR #187** (containment, statuses, OTS, reveal job, honest ops) +
ops actions below.

### F3 — Live API key committed to a public repo · CRITICAL (security)

`docs/ROUTINES.md` (merged in PR #175) contains the production
`X-API-Key` and VPS IP. The repo is public. The key authorizes POST
`/trigger/*` and POST `/intel/events` (data poisoning). Git history
retains the key even after this fix.
**Fix: this PR removes it from docs; the key MUST be rotated on the VPS.**

### F4 — Nightly intel extraction crashed since Jun 29 · HIGH

`'float' object is not subscriptable`: disclosures with NULL text
(scanned PDFs) surface as float NaN through pandas; `text[:8000]` threw;
the crash aborted the batch before store/mark, so the job re-billed the
same LLM calls nightly and the backlog froze at ~397 unprocessed. The
Claude intel routine carried classification alone (266 of 334 events).
**Fix: PR #184** (NaN normalization + per-row containment).

### F5 — Insider pipeline dead upstream; two strategies never fire · HIGH

NSE discontinued the `corporates-pit` JSON API (~Apr 28): it returns
HTTP 200 with an empty/frozen dataset for any recent window (verified
live for HCLTECH, RELIANCE, and whole-market queries). PR #174's
curl_cffi fix restored *access* but the data ends Apr 28; the job
"stores" the same 16 stale rows nightly. Consequences: 
`insider_cluster_v1` never fires; the blowup detector is starved
(pledge_snapshots has 0 rows, `pump_score` hardcoded 0), the avoid list
is empty every day, so `blowup_short_v1` never fires either.
**Fix: needs a new collector (BSE insider/SAST feed — dual-filed) —
follow-up PR; pledge collector also needs investigation.**

### F6 — ATR stops never enforced (FIX-08 half-shipped) · HIGH

The stop-hit logic existed only as `POST /paper/evaluate-stops`; nothing
called it. All 677 paper trades ride the full 15-day hold; `exit_reason`
NULL everywhere.
**Fix: PR #186** (shared service + daily 17:30 scheduler job).

### F7 — Universe stale: TATAMOTORS → TMPV; LTIM gone · MEDIUM

Bhavcopy has no TATAMOTORS/LTIM; TMPV exists (post-demerger ticker).
TATAMOTORS has ZERO OHLCV rows yet still got 2 paper trades today with
NULL entry prices (never evaluable). yfinance live-poll fails for both
symbols every 2 minutes. The Monday rebalance check only logs changes;
configs (`configs/stocks.yaml`, `ui_support.NIFTY_50`) were never updated.
**Fix: follow-up PR to update universe configs (verify current Nifty 50
membership first).**

### F8 — Symbol-format split brain · MEDIUM

`daily_ohlcv` holds both conventions: bhavcopy rows are `.NS`-suffixed
(~5,356/wk), yfinance rows bare (~200/wk). `paper_trades` mixes both
(bare ensemble symbols; intel strategies emit whatever the disclosure
had, e.g. `BASML.NS`). All joins are exact-match — evaluation and entry
pricing silently miss whenever formats disagree.
**Fix: follow-up PR — normalize to one convention at the store boundary.**

### F9 — P4 execution layer built but not deployed · MEDIUM

`alphavedha/execution/` (OMS, kill switch, PaperBroker, Telegram,
Kite adapter, shadow mode) is complete code with no `shadow_fills`
migration, no scheduler job, no env wiring. Zero shadow fills logged.
**Fix: follow-up PR (migration + 09:15 shadow job) once F1 lands —
shadow-testing a degenerate model is pointless.**

### F10 — UI hides most of the system · MEDIUM

The V3 UI rewrite covers the old core (predict/scan/backtest/paper-sim)
but calls none of: `/paper/dashboard` (per-strategy cost-adjusted
3-track record), `/verify` + `/proofs`, `/red-flag-radar`, `/strategies`
(lifecycle), `/weekly-digest`, `/analyze/{symbol}` (Beneish/Altman),
`/rotation`, `/timing`, `/features/drift`, `/experiments`. No intel
events feed page exists.
**Fix: alphavedha-ui PRs — strategy dashboard, verify, red-flag radar
first.**

### F11 — Smaller items · LOW

- NSE announcements 3 days stale (2 job failures in 4 days); BSE covers
  dual-filed docs meanwhile.
- Health flags evening-job tables "stale" every morning before their
  jobs run — 5 warnings/day of noise that trains readers to ignore
  `problems[]`.
- Morning-briefing routine produced no PR on Jun 30 / Jul 2; briefing
  drafts pile up unmerged (7 open).
- `institutional_flows` has only 26 rows total (started mid-June) — the
  FII/DII features the regime detector relies on have almost no history.
- 4 intel paper trades have NULL entry price (never evaluable).

## Fix map

| PR | Fixes | Status |
|---|---|---|
| #184 intel NaN crash | F4 | open |
| #185 retrain quality (class balance + macro parity + gate) | F1 | open |
| #186 stop-loss enforcement job | F6 | open |
| #187 verification publish/OTS/reveal/honest ops (stacked on #186) | F2 | open |
| this PR | F3 (docs part) | — |
| follow-ups | F5, F7, F8, F9, F10 | planned |

## Ops actions (owner: Saurabh — cannot be done from code)

1. **Rotate `ALPHAVEDHA_API_KEY`** in `/opt/alphavedha/alphavedha/.env.vps`,
   restart api+scheduler, update every routine's configuration on
   claude.ai. (Dual-key rotation is supported: set the new key as primary,
   keep the old as `ALPHAVEDHA_API_KEY_SECONDARY` for a day, then drop it.)
2. **Fix the proofs volume**: `chown -R <app-uid>:<app-gid>
   /var/lib/docker/volumes/alphavedha_proofs-repo/_data` (check the UID
   with `docker exec alphavedha-scheduler id`).
3. **Proofs remote**: create a deploy key with write access to
   `alphavedha-proofs`, mount it for the scheduler, set
   `ALPHAVEDHA_PROOFS_REMOTE` in `.env.vps`.
4. Optional: add `opentimestamps-client` to the image for OTS stamping.
5. After merging the PRs: rebuild + restart containers, verify the next
   morning run: `prediction_hash` shows `status=published`, retrain either
   promotes a healthy model or refuses loudly.
