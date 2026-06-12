# AlphaVedha Edge Factory — Master Implementation Plan

> Created: 2026-06-13 · Owner: Saurabh · Built daily with Claude Code
> Strategy: turn AlphaVedha from "one prediction model" into an **edge factory** —
> a machine that ingests India's unread disclosure paper trail, manufactures testable
> trading signals, measures every one of them honestly (cost-adjusted paper trading,
> cryptographically verifiable), and converts the proven ones into capital.

---

## 0. How to use this document daily

This is the single source of truth for the build. The workflow, every working day:

1. **Start a session** and say: `EF: next` (or "continue the edge factory plan").
   Claude reads this doc + the Progress Log, picks the first unchecked task in the
   active phase, and you build it together.
2. **One task = one sitting** (sized ~2–4 focused hours with Claude assistance).
   A task is done only when its **"Done when"** criterion passes — not before.
3. **End the session**: check the box(es), append one line to the Progress Log
   (§14), commit the doc update together with the code.
4. **Friday**: do the Weekly Review ritual (§13) — 15 minutes, KPIs + gate check.

**Where progress is monitored:** this file on GitHub (checkboxes + Progress Log),
plus the live system itself (`/api/paper/dashboard` tracks every strategy's
cost-adjusted record). Optionally mirror each phase as a GitHub issue
(`gh issue create` — task P0-D1 sets this up) so the repo's issue board shows
phase status at a glance.

**Rules of engagement (unchanged from current workflow):**
- One branch per phase: `feat/p1-disclosure-spine`, etc. Small PRs, CI green,
  **Saurabh merges** — Claude never merges.
- Every module gets unit tests per repo standards; financial-data rules in
  `CLAUDE.md` apply (point-in-time, timezone-aware, no look-ahead).
- Tasks can compress: if you have a full day, do 2–3 tasks. The plan is ordered
  by dependency, not by calendar — slipping a day breaks nothing.

---

## 1. Strategy recap — the three moves

1. **The Disclosure Machine.** ~4,000 BSE/NSE-listed companies file announcements,
   concall transcripts, credit-rating actions, pledge and insider disclosures daily;
   analysts cover ~400 of them. An LLM pipeline that reads *everything* same-day is
   an edge institutions are structurally absent from (too small for them) and that
   was technically impossible until LLM costs collapsed. Output: event-drift signals,
   language-delta signals, and a blowup detector ("do-not-touch" list).
2. **The provably honest track record.** Hash every morning's predictions before
   market open, timestamp the hash cryptographically (OpenTimestamps → Bitcoin-anchored).
   Mathematically impossible to backfill. Runs **privately from day one** — all
   public-facing publication is deliberately the LAST phase (§11); the proofs are
   independently verifiable whenever we choose to go public, with full history intact.
   No research shop in India has this; it converts into trust → capital → customers.
3. **The factory discipline.** Everything (the existing ensemble, every new signal)
   flows through the SAME harness: paper trades → cost-adjusted 3-track record
   (PR #72) → quantitative gates (§12) → only then real capital. Ideas are cheap;
   the honest tester is the moat.

**What we explicitly do NOT build:** HFT/market-making, crypto bots, more
deep-learning accuracy chasing on OHLCV, unregistered paid signals.

---

## 2. Current state (2026-06-13)

- Live VPS (Hetzner CX23, €4/mo): API + UI + scheduler + TimescaleDB + Redis healthy.
- 7 models trained; ensemble val acc 0.529; meta-gate currently passes ~0 trades.
- PR #72 merged: cost-adjusted 3-track paper record (`all` / `gate_passed` / `top_k`),
  `is_tradeable` persisted, weekend guard. First paper cohort: **Mon 2026-06-15 08:30 IST**.
- First matured evaluations: **~2026-07-06** (15 trading days + weekend buffer).
- Existing ingestion assets to reuse: BSE announcements (weekly job), FII/DII, NSE
  cookie handling, FinBERT sentiment, Beneish/Altman fundamental module, sector
  rotation, index-constituent tracking with rebalance detection.
- Public API at `http://178.105.237.174/api/*` currently has **no API key** — fix
  scheduled in P0.

---

## 3. Target architecture

```
                        ┌─────────────────────────────────────────────┐
   EXISTING             │              NEW: alphavedha/intel/          │
   ───────              │                                             │
   yfinance/jugaad ─┐   │  collectors/        extraction/   signals/  │
   FII/DII          │   │  ├ bse_announce     ├ taxonomy    ├ event_drift
   index const.     │   │  ├ nse_announce     ├ schemas     ├ lang_delta
                    │   │  ├ insider_pit      ├ extractor   ├ blowup_score
                    ▼   │  ├ pledge_sast      ├ batcher     └ (feeds ───┐
   daily_ohlcv ◄── bhavcopy (NEW: whole-market EOD, 1 file/day)        │
                        │  ├ rating_actions   ├ eval/golden            │
                        │  ├ transcripts      └ prompts/ (versioned)   │
                        │  ├ asm_gsm_lists                             │
                        │  └ bulk_block_deals                          │
                        └──────────────────────────────────────────────┘
                                                                       │
                              paper_trades (+ strategy column) ◄───────┘
                                       │                ▲
                              track_record (per-strategy)│
                                       │            ensemble (existing)
                              decision gates (§12)
                                       │
                        ┌──────────────┴──────────────┐
                        ▼                             ▼
            alphavedha/execution/ (NEW)      alphavedha/verification/ (NEW)
            ├ broker adapter (Kite/Dhan)     ├ hasher (daily 08:35)
            ├ order manager + GTT OCO        ├ opentimestamps stamper
            ├ kill switch / risk caps        ├ publisher (git proofs repo)
            ├ shadow mode (ghost fills)      └ reveal job (+21d)
            └ telegram bot (confirm/panic)
```

**New DB tables** (all with `created_at`, indexed on `(symbol, <date>)`):

| Table | Key columns | Purpose |
|---|---|---|
| `disclosures` | id, symbol, source, category, headline, filed_at (tz-aware), url, text, text_hash, processed_at | Raw normalized filings (one row per filing) |
| `disclosure_events` | id, disclosure_id FK, symbol, event_type, direction, materiality (0–10), confidence, summary, red_flags JSON, llm_model, prompt_version, extracted_at | LLM-structured events |
| `transcripts` | symbol, fiscal_quarter, filed_at, text, sections JSON | Concall transcripts (Reg 30 LODR filings) |
| `rating_events` | symbol, agency, action, rating_from, rating_to, outlook, rationale_text, filed_at | Credit rating actions |
| `pledge_snapshots` | symbol, as_of, promoter_pledge_pct, change_pct | Promoter pledge history |
| `surveillance_flags` | symbol, list_name (ASM/GSM/stage), added_on, removed_on | Exchange surveillance membership |
| `shadow_fills` | strategy, symbol, ts, side, decision_price, sim_fill_price, qty, slippage_bps | Phase-4 ghost execution log |
| `prediction_proofs` | proof_date, sha256, ots_path, git_commit, revealed_at | Phase-0 hash registry |

(`insider_trades` and `bulk/block` reuse existing tables where present.)

**CRITICAL point-in-time rule (applies to every intel signal):** a disclosure may
influence a signal only for prediction dates strictly AFTER its exchange `filed_at`
timestamp (IST). Signals fire at the next 08:30 job after filing. Historical
backfills of text are for research only and must never leak into live-dated
features. Every signal module gets a dedicated look-ahead unit test.

---

## 4. Phase plan at a glance

| Phase | Calendar (approx) | Deliverable | Status |
|---|---|---|---|
| P0 Verifiable record (private) + hygiene | Jun 14–17 | Daily prediction hashes stamped privately; API key on | ☐ |
| P1 Disclosure ingestion spine | Jun 18 – Jul 1 | All 8 sources landing in DB daily | ☐ |
| P2 LLM extraction layer | Jul 2 – Jul 15 | Structured events, ≥85% precision on golden set | ☐ |
| P3 Signals + multi-strategy harness | Jul 16 – Jul 29 | 3 intel strategies live in paper trading | ☐ |
| P4 Execution engine (shadow) | Jul 30 – Aug 12 | OMS + kill switch + Telegram, shadow fills logging | ☐ |
| P5 Gates & scale | Aug 13 → | First live capital on gate-passing strategy | ☐ |
| P6 Public launch — **LAST, by design** | When the record earns it (~Sep+) | /verify + public track record + Red Flag Radar | ☐ |

Parallel passive track throughout: the ensemble's paper record accrues on its own
(first gate review possible ~Jul 20 with ~25 cohorts).

---

## 5. PHASE 0 — Verifiable record (PRIVATE) + hygiene (2–3 days)

**Goal:** every prediction this system ever makes from now on is provably
timestamped before market open. Plus close the open-API hole.

**Why this can't wait even though publishing comes last:** the record only counts
from the day hashing starts. OpenTimestamps anchors each hash into Bitcoin, so the
proof of "this prediction existed before market open on date X" is independently
verifiable forever — regardless of whether anyone can see it yet. We run it
private; P6 flips it public with months of history already banked. Nothing in
this phase exposes anything publicly.

### P0-D1 — Hasher + proof registry
- [ ] New module `alphavedha/verification/hasher.py`:
  - `canonical_payload(trades: list[dict]) -> bytes` — the day's paper_trades rows
    (symbol, prediction_date, direction, magnitude, confidence, is_tradeable,
    model_version, strategy) as JSON with `sort_keys=True`, no whitespace drift.
  - `sha256_hex(payload) -> str`.
  - `prediction_proofs` table + alembic migration.
- [ ] Unit tests: deterministic hash across dict orderings; changing any field
  changes the hash; empty day handled.
- [ ] Optional: `gh issue create` one issue per phase (P0…P6) for board visibility.
- **Done when:** `pytest tests/unit/verification/` green; same trades always
  produce the same hex regardless of row/key order.

### P0-D2 — Stamper + scheduler job (private)
- [ ] Create **PRIVATE** GitHub repo `alphavedha-proofs` (README explains the
  scheme; repo flips to public in P6 with history intact — git commit dates +
  OTS proofs make the record retroactively verifiable).
- [ ] `publisher.py`: writes `proofs/YYYY-MM-DD.sha256`, commits + pushes with a
  scoped deploy key; records git commit SHA in `prediction_proofs`.
- [ ] OpenTimestamps: `pip install opentimestamps-client`; stamp the hash file
  (`ots stamp`) → commit the `.ots` proof alongside. (Free; anchors into Bitcoin
  via public calendar servers — this is the part that makes the private record
  publicly provable later; upgrade `.ots` weekly via a small job.)
- [ ] Scheduler: `PREDICTION_HASH_TIME = "08:40"` job (after 08:30 predictions
  persist, before 09:15 open). Weekend-guarded like predictions. Failure → log
  `scheduler_job_failed` (email alert reuses `EmailAlerter`).
- [ ] Reveal job (daily, 16:00): for proof rows ≥ 21 days old and unrevealed,
  write the raw canonical JSON next to its hash + a `verify.py` script in the
  proofs repo so anyone can re-hash and compare (stays private until P6).
- **Done when:** one full cycle works against a synthetic day locally: hash →
  commit → (ots stamped) → reveal → `verify.py` passes.

### P0-D3 — Hygiene
- [ ] Set `ALPHAVEDHA_API_KEY` in `.env.vps` (API currently open at /api/*);
  confirm UI still works (it proxies server-side or add key to UI env).
  Everything goes behind the key for now — any public/private endpoint split is
  a P6 decision, not now.
- [ ] Deploy P0 (merge → CI deploys → verify first real hash Monday 08:40 IST).
- **Done when:** `curl /api/predict/TCS` without key → 401; Monday's hash lands
  in the private `alphavedha-proofs` repo before 09:15 IST.

---

## 6. PHASE 1 — Disclosure ingestion spine (~2 weeks)

**Goal:** eight data sources land in normalized tables daily, whole-market EOD
prices via bhavcopy, all idempotent and survivable on a 4 GB VPS.

**Design rules:** store extracted TEXT in DB, raw PDFs on disk volume with 30-day
retention; every collector is idempotent (upsert on natural key); per-source
failure logs + continues (never crashes the job); all timestamps Asia/Kolkata.

### P1-D1 — Schema + scaffolding
- [ ] `alphavedha/intel/` package skeleton; ORM models for the §3 tables;
  alembic migration; `store_disclosure` / `load_disclosures` with upsert.
- **Done when:** migration applies cleanly to a fresh DB and to a copy of prod
  schema; unit tests for store/load round-trip.

### P1-D2 — Whole-market EOD via NSE bhavcopy
- [ ] Collector for NSE daily bhavcopy (full-market EOD OHLCV + delivery in ONE
  file/day — this is how 2,000-symbol coverage works on a €4 VPS, not 2,000
  yfinance calls). Parse → upsert into `daily_ohlcv`.
- [ ] Backfill script from bhavcopy archives (target: 3 years, smallcap universe).
- [ ] Scheduler job 18:45 IST (after FII/DII), weekend-guarded.
- **Done when:** one day's bhavcopy ingests < 60s; row counts reconcile vs
  yfinance for 5 spot-checked Nifty symbols (±0.5% on close).

### P1-D3 — BSE announcements: weekly → continuous
- [ ] Upgrade existing `ingest_bse_announcements` to a daily (later hourly) job
  writing to `disclosures` with category, PDF URL, `filed_at`.
- [ ] PDF fetch + text extraction (`pymupdf`); cap 50 pages; store text + hash;
  skip scanned/no-text PDFs gracefully (flag `text IS NULL`).
- **Done when:** yesterday's BSE announcements for the full market are in
  `disclosures` with extracted text for ≥80% of non-scanned PDFs.

### P1-D4 — NSE announcements + insider (PIT) + pledge (SAST)
- [ ] NSE corporate announcements collector (reuses existing NSE cookie/session
  infra; expect 403s — retry with backoff, log-and-skip on persistent block;
  BSE remains the fallback source since most companies dual-file).
- [ ] Insider trading (PIT) disclosures CSV → `insider_trades` (acquirer/seller,
  qty, value, mode). Pledge (SAST) disclosures → `pledge_snapshots`.
- **Done when:** 7 days of PIT + pledge data visible in DB; insider-cluster query
  (≥2 distinct insiders buying same symbol in 14d) returns sane results.

### P1-D5 — Surveillance lists + bulk/block deals
- [ ] ASM/GSM list collector (NSE publishes daily; store add/remove transitions
  in `surveillance_flags`).
- [ ] Bulk + block deals daily CSV collector.
- **Done when:** current ASM list matches NSE website spot-check; transitions
  (added_on/removed_on) tracked across two consecutive days.

### P1-D6 — Credit rating actions
- [ ] Primary: filter `disclosures` for the credit-rating announcement category
  (companies must disclose rating actions) → parse into `rating_events`.
- [ ] Secondary collectors (CRISIL/ICRA/CARE/India Ratings press-release pages)
  — build ONE agency first (whichever scrapes cleanest), others later.
- **Done when:** last 30 days of rating actions for Nifty-500 names populated;
  downgrades/outlook-negative queryable.

### P1-D7 — Concall transcripts
- [ ] Collector: BSE/NSE announcement category for transcripts (LODR Reg 30 —
  companies file analyst-call transcripts within ~5 working days of the call).
  Parse PDF → `transcripts` keyed (symbol, fiscal_quarter).
- [ ] Section splitter: management remarks vs Q&A (regex/heuristic on speaker
  markers; LLM-assisted fallback in P2).
- **Done when:** ≥20 transcripts from the current earnings season stored with
  section splits for ≥70%.

### P1-D8 — Ops hardening
- [ ] All collectors registered in scheduler with staggered times (18:30–21:30
  IST window); `JobResult` history; quality-check integration (row-count
  anomalies → email alert).
- [ ] Disk-retention cron for PDFs; DB size monitoring (text tables on 4 GB box —
  alert at 70% disk).
- [ ] Deploy P1; watch 3 consecutive days of clean runs.
- **Done when:** 3 days, 8 sources, zero crashed jobs (skips/partials allowed and
  logged).

---

## 7. PHASE 2 — LLM extraction layer (~2 weeks)

**Goal:** every new disclosure becomes a structured, scored event within hours,
at < ₹4k/month, with measured accuracy.

**Model strategy — pluggable backend, decided by data not loyalty.**
The extractor talks to models through one interface (`intel/extraction/llm.py`),
so any provider can serve any tier. The golden set (P2-D2) is the referee: the
cheapest model that clears the precision bar wins each tier. Our input data is
public exchange filings — zero confidentiality concern, so free tiers (which may
train on inputs) are acceptable here.

| Tier | Baseline (known-good) | Cheap/free candidates to bake off | Mode |
|---|---|---|---|
| T1 Triage: relevant/boilerplate + coarse category (~300/day) | `claude-haiku-4-5` ($1/$5 per MTok; **Batches −50%** → $0.50/$2.50) | Gemini Flash free tier; DeepSeek; Groq/OpenRouter free open models; Mistral free tier | nightly batch |
| T2 Structured extraction on survivors (~100–150/day) | `claude-haiku-4-5` (Batches) | same candidates — only if they clear the golden-set bar (incl. red-flag recall ≥0.9) | nightly batch |
| T3 Deep: transcripts, materiality ≥7, rating rationales (~60/wk) | `claude-sonnet-4-6` ($3/$15, Batches −50%) | don't cheap out here — errors poison signals | nightly batch |
| T4 Weekly synthesis memo for Saurabh | `claude-opus-4-8` ($5/$25) | n/a — one call/week, judgment matters | weekly |

Plus prompt caching on the stable system prompt (taxonomy + rules) — cached
tokens cost ~10% of base. Rule-based pre-filter (below) cuts volume ~50% before
any model is called.

Estimated spend: **₹500–2,000/month if a free/cheap tier wins T1–T2 at bake-off;
₹2,000–4,000/month on the all-Claude baseline.** Hard monthly cap enforced in
code either way. Provider price cards change monthly — re-verify rates at P2-D3,
don't trust this table's absolute numbers.

**NOT worth doing:** self-hosting open-weight models. The VPS (2 vCPU/4 GB) and
the laptop (i3/8 GB) can't run anything useful; GPU rental costs more than API
calls at ~300 docs/day. Revisit only if volume grows 50×.

### P2-D1 — Taxonomy + schemas
- [ ] `intel/extraction/taxonomy.py` — event_type enum:
  `order_win, capacity_expansion, results_guidance, guidance_cut, fund_raise,
  m_and_a, rating_upgrade, rating_downgrade, outlook_change, pledge_increase,
  pledge_release, insider_buy, insider_sell, auditor_resignation,
  kmp_resignation, related_party_txn, litigation_regulatory, default_or_delay,
  surveillance_action, dividend_buyback, other`.
- [ ] Pydantic v2 extraction schema (used with `client.messages.parse()` /
  structured outputs): event_type, direction (-1/0/+1), materiality 0–10,
  confidence 0–1, one-line summary, red_flags list, numbers extracted
  (order value ₹, capacity %, rating notches…).
- **Done when:** schema validates against 10 hand-written example outputs;
  taxonomy reviewed by Saurabh (30 min — this defines the product).

### P2-D2 — Golden set
- [ ] Sample 100 real disclosures from P1 data (stratified: each category, mix of
  large/smallcap). Claude pre-labels; **Saurabh reviews/corrects** (~45 min).
  Stored as `tests/fixtures/intel_golden_set.jsonl`.
- [ ] Eval harness: precision/recall per event_type, materiality MAE,
  red-flag recall (the one that matters most).
- **Done when:** golden set frozen in repo; eval runs in CI (against recorded
  LLM outputs, not live API).

### P2-D3 — Extractor + prompt v1 + model bake-off
- [ ] `intel/extraction/llm.py`: provider-agnostic interface (one method:
  structured extraction against a Pydantic schema). Anthropic backend first
  (SDK, structured outputs, prompt-cached system prompt); thin adapters for
  bake-off candidates (Gemini free tier, DeepSeek, one OpenRouter free model).
- [ ] `intel/extraction/extractor.py` on top of it; versioned prompts in
  `intel/prompts/` (`v1.md`…); `prompt_version` + `llm_model` stored on every
  event row.
- [ ] Rule-based pre-filter: skip boilerplate categories (trading windows, ESOP
  allotments, newspaper-ad copies) before any LLM call — typically cuts volume
  ~50%.
- [ ] **Bake-off:** run golden set through baseline + each candidate; record
  precision/recall/red-flag-recall + effective ₹/1000 filings in a table in
  this doc. Cheapest model clearing the bar wins T1/T2; verify current price
  cards + free-tier rate limits the same day (they change monthly).
- **Done when:** eval ≥ 0.85 precision on event_type, ≥ 0.9 recall on the
  red-flag subset (auditor/KMP resignation, default, pledge spike) for the
  CHOSEN tier assignment; bake-off table pasted below + in Progress Log.

### P2-D4 — Batch pipeline + cost guard
- [ ] `intel/extraction/batcher.py`: nightly job collects unprocessed
  disclosures → Batches API request (custom_id = disclosure id) → poll → store
  `disclosure_events`. Retries, partial-failure tolerance, idempotent.
- [ ] Cost ledger: tokens in/out per batch logged; monthly cap (env
  `INTEL_LLM_BUDGET_USD`, default 50) — pipeline pauses + alerts at cap.
- [ ] `ANTHROPIC_API_KEY` to `.env.vps`.
- **Done when:** two consecutive nights process the real backlog end-to-end on
  VPS; cost ledger shows < $2/night.

### P2-D5 — Transcript deltas (the language signal)
- [ ] Quarter-over-quarter compare: for each symbol with ≥2 transcripts, Sonnet
  compares management sections → guidance_delta (-2…+2), tone_delta, dropped
  commitments list, evasiveness score on Q&A.
- **Done when:** outputs for 10 known companies pass Saurabh's smell test
  (he knows these stocks); stored in `disclosure_events` with
  event_type=`results_guidance`, linked to both quarters.

### P2-D6 — Backfill + first research readout
- [ ] Backfill extraction over P1's historical window (BSE archives go back far
  enough for a few months of events; respect budget — Haiku batches).
- [ ] Research notebook: event study — median abnormal return (vs Nifty) at
  +1d, +5d, +15d per event_type, split largecap/smallcap. **This readout
  decides which P3 signals to build first.**
- **Done when:** readout table exists in `notebooks/event_study.ipynb` and is
  pasted into the Progress Log; we pick the top 2 drift signals from evidence,
  not vibes.

---

## 8. PHASE 3 — Signals + multi-strategy harness (~2 weeks)

**Goal:** intel becomes strategies, measured by the same honest harness as the
ensemble. Nothing trades; everything is paper.

### P3-D1 — Multi-strategy paper trading (the keystone migration)
- [ ] Add `strategy` column to `paper_trades`, default `'ensemble_v1'`;
  PK becomes (symbol, prediction_date, strategy). Alembic migration (verify
  paper_trades is a plain table, not hypertable, before PK change).
- [ ] Thread `strategy` through: `store_paper_trade`, `load_paper_trades`,
  scheduler persist, evaluation job (evaluates all strategies), hasher payload,
  `/paper/dashboard` (per-strategy tracks), `track_record.py` (group by
  strategy).
- [ ] Update all affected tests.
- **Done when:** dashboard shows the ensemble under `ensemble_v1` unchanged
  (numbers identical pre/post migration on a DB snapshot) + an empty second
  strategy renders cleanly.

### P3-D2 — Signal: event drift v1
- [ ] `intel/signals/event_drift.py`: for events from evidence-selected types
  (P2-D6) with materiality ≥ threshold and `filed_at` < today: emit
  direction/confidence per symbol. Confidence = f(materiality, event-study edge,
  coverage_score (smallcap → higher weight)).
- [ ] Wire into 08:30 job as strategy `event_drift_v1` (max 10 positions/day,
  long-only signals from positive events; negative events emit -1 but are
  flagged `short_constrained=true` — measured, not assumed tradeable).
- [ ] Look-ahead unit test (event filed today must NOT fire today).
- **Done when:** first cohort persists + is hashed alongside ensemble rows.

### P3-D3 — Signal: blowup detector v1
- [ ] `intel/signals/blowup_score.py`: composite 0–100 per symbol from:
  pledge_increase trend, rating_downgrade/outlook-negative, auditor/KMP
  resignation, default_or_delay, ASM/GSM addition, Beneish M-Score red zone
  (reuse `fundamental/`), insider_sell clusters, volume+price pump signature.
- [ ] Output: daily `avoid_list` (score ≥ 70) stored + exposed at
  `/api/intel/red-flags`; ensemble + event_drift signals on avoid-listed
  symbols are vetoed (recorded as vetoed, so the veto itself is measurable).
- [ ] Paper strategy `blowup_short_v1` (direction -1 on new avoid-list entries)
  — measures whether flags predict drawdowns even if we never short.
- **Done when:** avoid list renders; veto plumbing tested; known-blowup
  backtest sanity check (does the score fire on 3 historical disasters?).

### P3-D4 — Signal: insider/transcript composite
- [ ] `insider_cluster_v1`: ≥2 distinct insiders net-buying ≥ ₹25L within 14d,
  not on avoid list → long signal (one of the most robust documented effects).
- [ ] `guidance_delta_v1`: transcript guidance_delta ≥ +1 with positive tone
  delta → long; ≤ -1 → flagged negative.
- **Done when:** both persist daily cohorts; per-strategy dashboard shows 4–5
  strategies accruing.
- [ ] **STOP adding strategies.** 5 concurrent paper strategies max until gates
  decide. (Multiple-testing discipline — more strategies = more false
  positives.)

### P3-D5 — Evaluation upgrades
- [ ] Per-strategy evaluation already works via D1; add: per-strategy daily
  email summary (reuse EmailAlerter) — cohort counts, matured results,
  expectancy net, current avoid list.
- [ ] Deploy P3; verify 3 clean mornings (hash includes all strategies).
- **Done when:** Monday morning email arrives with all strategies listed.

---

## 9. PHASE 4 — Execution engine, shadow mode (~2 weeks)

**Goal:** the gun, built and zeroed while paper proves the ammo. No real orders
until §12 gates pass — but every part exercised daily in shadow.

### P4-D1 — Broker decision + adapter interface
- [ ] Decide broker API: **Kite Connect** (free for personal use since late
  2024; historical-data add-on paid) vs **Dhan** (free, algo-friendly). Open
  the account/API app for the chosen one (Saurabh action). Verify SEBI/broker
  current requirements for personal API algos (static IP — VPS has one;
  broker-side algo registration thresholds; 2025 framework).
- [ ] `alphavedha/execution/broker.py`: `BrokerAdapter` protocol — auth,
  place/modify/cancel order, GTT OCO create, positions, holdings, margins,
  order updates. `PaperBroker` implementation first (fills at next open ±
  slippage model) — this is what shadow mode runs.
- **Done when:** protocol + PaperBroker unit-tested; real adapter stubbed.

### P4-D2 — Order manager + risk caps
- [ ] `execution/oms.py`: turns a gate-passed signal into an order plan —
  position size from existing Kelly module (capped 5%/position), entry limit
  at open, GTT OCO exits (target = conformal upper band, stop = lower barrier),
  time-exit at horizon.
- [ ] `execution/kill_switch.py`: hard caps — max positions (8), max daily new
  exposure (25%), daily loss limit (2%) → flatten + halt, drawdown halt (6%),
  env master switch `EXECUTION_ENABLED=0` default, every order logged before
  send.
- **Done when:** OMS unit tests incl. every kill-switch trip path; mutation
  test: no code path can place an order while `EXECUTION_ENABLED=0`.

### P4-D3 — Shadow mode
- [ ] 09:15 job: run the FULL loop (signals → gates → OMS → PaperBroker) daily;
  log `shadow_fills` with decision price vs simulated fill → measured slippage
  distribution feeds back into the cost model (replacing the flat assumption).
- **Done when:** 5 consecutive days of shadow fills; slippage report vs the
  0.1%/side assumption in `backtest/costs.py`.

### P4-D4 — Telegram control plane
- [ ] Bot (free Bot API): 08:35 sends today's gate-passed signals with
  ✅/❌ inline buttons (semi-auto mode: human taps to approve before OMS acts);
  commands: `/status`, `/positions`, `/halt`, `/resume`, `/panic` (flatten).
- [ ] Auth: chat-id allowlist (just Saurabh).
- **Done when:** end-to-end drill on shadow: approve via phone → shadow fill;
  `/panic` halts within one poll cycle.

### P4-D5 — Real adapter (built, not armed)
- [ ] Implement the real broker adapter behind the same protocol; integration
  test against broker sandbox/paper environment if available, else
  read-only calls (margins, positions) on the live account.
- [ ] Runbook page: arming checklist, key rotation, what `/panic` does, daily
  ops. **`EXECUTION_ENABLED` stays 0.**
- **Done when:** adapter passes read-only integration checks; runbook reviewed.

---

## 10. PHASE 5 — Gates & scale (ongoing from ~Aug 13)

- [ ] **P5-D1** Gate review #1 (run as soon as any strategy hits 30 evaluated
  cohorts — ensemble reaches this first, ~Jul 20): apply §12 criteria, write
  verdict per strategy into Progress Log.
- [ ] **P5-D2** For the first gate-passing strategy: arm semi-auto live with
  ₹50,000 (Telegram-approved orders only, kill switch active, position cap
  ₹10k). 4 weeks minimum at this size.
- [ ] **P5-D3** Live-vs-paper tracking report: live fills vs paper assumption;
  divergence > slippage budget → back to shadow, investigate.
- [ ] **P5-D4** Scale decision ladder: ₹50k → ₹2L → ₹5L only on passing live
  reviews (§12 G2). In parallel, pick the capital path: more own capital /
  family capital with a written mandate / SEBI RA registration (NISM-XV exam)
  if productizing research.
- [ ] **P5-D5** Factory loop becomes routine: new hypothesis → research
  notebook → paper strategy (≤5 concurrent) → gates. Retire failures monthly
  (record the kill in the log — dead strategies are also a track record).

---

## 11. PHASE 6 — Public launch (deliberately LAST)

**Entry condition:** at least one G1 gate review completed AND Saurabh decides
the record is worth showing (typically ≥3 months of unbroken hashes). Until
then, everything stays private — the machine makes money first, talks later.

- [ ] **P6-D1** Flip `alphavedha-proofs` repo to public — months of git history
  + Bitcoin-anchored OTS proofs go live in one click, retroactively verifiable.
- [ ] **P6-D2** `/verify` page on the UI: explains the hash scheme, lists daily
  proofs (from `prediction_proofs`), links to the proofs repo +
  OpenTimestamps verification instructions. "The only fully auditable
  prediction record in India" — say it plainly.
- [ ] **P6-D3** Public track-record page: per-strategy cost-adjusted records
  straight from `/paper/dashboard` (it's already honest — gross AND net,
  including losing strategies. Publishing losers is the credibility move).
  Decide the public/private API endpoint split here.
- [ ] **P6-D4** Red Flag Radar public page: the avoid list with *factual,
  cited* flags (pledge %, rating action, ASM stage, resignation filings —
  each linking to the primary disclosure). Prominent disclaimer: factual
  information, not investment advice, not SEBI-registered research.
  **Before launch: re-verify the SEBI RA boundary for this format** (factual
  data with citations vs "recommendations") — if grey, keep it gated/private
  until resolved.
- [ ] **P6-D5** (optional, Saurabh's call) Build-in-public thread cadence:
  weekly post = one chart from the live record + one thing learned. The
  distribution asset.
- **Done when:** /verify and track-record pages live; radar decision made.

---

## 12. Decision gates (quantitative, pre-committed)

Written BEFORE results exist so we can't move the goalposts.

**G1 — paper → small live (per strategy):**
- ≥ 30 evaluated cohorts AND ≥ 60 evaluated trades
- Net expectancy (after the cost model) > 0 with bootstrap 90% CI excluding 0
- Win rate ≥ 45% AND profit factor ≥ 1.2 (net)
- Max drawdown (net cohort equity) > -10%
- No look-ahead test failures; hash record unbroken for the period
- **Any miss → stays paper. Two consecutive failed reviews → retire.**

**G2 — small live → scale (per strategy):**
- 4+ weeks live, ≥ 15 live fills
- Live net return within 1.5× slippage budget of paper counterpart
- Zero kill-switch breaches caused by code defects
- Ops clean: no missed mornings, hashes unbroken

**G3 — productize (sell anything):**
- 6+ months unbroken public verifiable record
- SEBI RA registration completed BEFORE any paid recommendation of any form

---

## 13. KPIs & weekly review (Fridays, 15 min)

| KPI | Target | Source |
|---|---|---|
| Paper cohorts accrued (per strategy) | +5/week each | /paper/dashboard |
| Hash record | 100% trading days, zero gaps | proofs repo |
| Disclosure latency (filed → event scored) | < 24h (P2), < 4h (later) | intel tables |
| Extraction precision on golden set | ≥ 0.85 (red-flag recall ≥ 0.9) | CI eval |
| Ingestion job success rate | > 95% of source-days | scheduler logs |
| LLM spend | < $50/mo hard cap | cost ledger |
| Infra spend | < ₹6k/mo total | invoices |

Ritual: update KPI row in Progress Log → check §12 gate eligibility → pick next
week's tasks → one sentence: "biggest risk right now is ___".

---

## 14. Risk register

| Risk | Mitigation |
|---|---|
| NSE scraping 403s (cookie fragility) | BSE as primary for dual-filed docs; backoff + skip; bhavcopy/ASM have stable endpoints |
| LLM extraction errors poisoning signals | Golden-set CI gate, versioned prompts, materiality threshold, human review of weekly memo |
| Look-ahead bias via text backfills | filed_at point-in-time rule + dedicated unit tests per signal (§3) |
| Multiple-testing false positives | ≤5 concurrent strategies; pre-committed gates (§12); bootstrap CIs |
| 4 GB VPS limits | Text-only DB storage, PDF retention 30d, nightly batches (not realtime), disk alerts; scale-up path exists (CX43 auto-scale already built) |
| Broker/SEBI rule changes for retail algos | Verify at P4-D1; semi-auto (human tap) mode as the conservative default |
| Key/credential leakage | All keys in `.env.vps` only; proofs repo uses a scoped deploy key; never in git |
| Vendor lock/format changes (BSE/NSE pages) | Collectors isolated per source; one breaking ≠ pipeline down; fixtures recorded for tests |
| Motivation/solo-dev fatigue | This doc: small daily tasks, visible checkboxes, weekly ritual, Claude does the heavy lifting |

---

## 15. Budget

| Item | Monthly |
|---|---|
| Hetzner CX23 (+ training hours) | ~₹450 |
| LLM API (₹500–2k if cheap tier wins bake-off; ₹2–4k all-Claude baseline) | hard cap $50 |
| Broker API (Kite personal / Dhan) | ₹0 |
| OpenTimestamps, GitHub, Telegram | ₹0 |
| **Total burn** | **< ₹5,000/mo** |
| First live capital (post-G1 only) | ₹50,000 (risk capital, separate account) |

---

## 16. Progress Log

> Append one line per working session: `YYYY-MM-DD — did X; next Y; blocker Z`

- 2026-06-13 — Plan created. Next: P0-D1 (hasher module). Blocker: none.
