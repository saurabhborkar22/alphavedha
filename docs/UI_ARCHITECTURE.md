# AlphaVedha — UI Architecture Reference

Repo: `/home/saurabh/alphavedha-ui/` (separate repo, no CI — manual deploy via Docker rebuild on VPS)

## Tech Stack

| Layer | Technology | Version |
|---|---|---|
| Framework | Next.js | 16.2.6 (App Router, all client components) |
| Language | TypeScript | 5 (strict mode, bundler module resolution) |
| UI Library | None — all hand-authored | Tailwind CSS v4 (PostCSS plugin) |
| Charts | Custom SVG from scratch | No external chart library |
| State | Zustand + persist | 5.0.14 |
| Server state | TanStack React Query | 5.100.14 (staleTime 30s default, retry 1) |
| Testing | Vitest + @testing-library/react + msw | 4.1.7 |
| Build output | Standalone (Docker-optimized) | `output: "standalone"` in next.config.ts |

No shadcn, Radix, MUI, Recharts, Chart.js, or form libraries.

---

## Routes

| Route | File | Purpose | Auth | Real-time |
|---|---|---|---|---|
| `/` | `app/page.tsx` | redirect('/dashboard') | No | No |
| `/login` | `app/login/page.tsx` | API key entry (12+ char min) | No | No |
| `/track` | `app/track/page.tsx` | Public track record | No | No |
| `/dashboard` | `app/(app)/dashboard/page.tsx` | Top signals, stats, neural viz | Yes | Passive (React Query staleTime 30-60s) |
| `/scanner` | `app/(app)/scanner/page.tsx` | Stock scanner with filter/sort | Yes | No (staleTime 30s) |
| `/live` | `app/(app)/live/page.tsx` | Intraday chart for a chosen symbol | Yes | Poll 10s (market hours only) |
| `/paper` | `app/(app)/paper/page.tsx` | Paper positions + historical simulation | Yes | Poll positions 15s |
| `/backtest` | `app/(app)/backtest/page.tsx` | Strategy backtest results | Yes | No (staleTime 120-600s) |
| `/track-record` | `app/(app)/track-record/page.tsx` | Redirect to `/track` | Yes | No |
| `/mlops` | `app/(app)/mlops/page.tsx` | Model health, drift, pipeline status | Yes | Poll 60s |
| `/data` | `app/(app)/data/page.tsx` | Data quality browser | Yes | No (staleTime 300s) |
| `/events` | `app/(app)/events/page.tsx` | Corporate events browser | Yes | No (staleTime 300s) |
| `/trends` | `app/(app)/trends/page.tsx` | Sector momentum, RRG signals | Yes | No (staleTime 300s) |
| `/notifications` | `app/(app)/notifications/page.tsx` | Notification feed | Yes | No (staleTime 30s) |
| `/settings` | `app/(app)/settings/page.tsx` | API key display, watchlist, sign-out | Yes | No |
| `/stock/[symbol]` | `app/(app)/stock/[symbol]/page.tsx` | Full prediction detail (3 tabs) | Yes | WebSocket `/api/ws/live/{symbol}` + poll 15s fallback |

### Route Group Structure
```
app/
  layout.tsx              -- Root HTML shell, wraps with <Providers>
  providers.tsx           -- QueryClientProvider (staleTime 30s, retry 1)
  page.tsx                -- redirect('/dashboard')
  login/page.tsx          -- Public, no auth guard
  track/page.tsx          -- Public, outside (app) group
  (app)/                  -- Route group (no URL segment)
    layout.tsx            -- Auth guard + NavBar + MobileBottomNav + CommandPalette
    dashboard/...
    scanner/...
    stock/[symbol]/...    -- Dynamic route, param via React 19 use(params)
    ...12 pages total
```

---

## Auth Flow

1. Any `(app)` route → `AppLayout` waits for `_hydrated` (Zustand rehydration from localStorage)
2. If not `isAuthenticated` → `router.replace('/login')`, renders null
3. Login: user enters API key (12+ chars) → `setApiKey(key)` → stored in localStorage, `isAuthenticated=true` → redirect to `/dashboard`
4. Every API request reads key directly from `localStorage.getItem('av-auth')` (not via hook — avoids async state issues)
5. Sign out: `logout()` → clears localStorage → redirect to `/login`
6. No JWT, no refresh tokens, no session cookies — API key is the credential

---

## State Management

### Zustand Stores (all persisted to localStorage)

| Store | Key | State | Actions |
|---|---|---|---|
| `useAuthStore` | `av-auth` | apiKey, isAuthenticated, _hydrated | setApiKey, logout, setHydrated |
| `useWatchlistStore` | `av-watchlist` | symbols: string[] | toggle(symbol), has(symbol) |
| `useNotificationsStore` | `av-notifications` | unread: number, readIds: string[] | setUnread, markRead, markAllRead |

`_hydrated` flag is critical — AppLayout renders null until rehydration completes (prevents flash of redirect).

### React Query
- All reads: `useQuery`, all writes: `useMutation` + `useQueryClient` invalidation
- Query keys: `['scan', tier]`, `['predict', symbol]`, `['paper', 'positions']`, etc.
- staleTime overrides per query: 30s (predictions), 60s (dashboard), 120-600s (backtest/data), 300s (events/trends)

---

## API Integration (lib/api/)

### Client (lib/api/client.ts)
```typescript
apiFetch<T>(path, options)   // authenticated — adds X-API-Key header
publicFetch<T>(path, options) // no auth — for /public/* /events/* /sectors/trends
```
Base URL: `process.env.NEXT_PUBLIC_API_URL ?? 'http://localhost:8000'`
In production Docker build: `NEXT_PUBLIC_API_URL=/api` (nginx proxies `/api/*` to FastAPI)

### API Modules

| Module | Functions | Used By |
|---|---|---|
| `lib/api/predictions.ts` | predict, scan, explainPrediction, portfolioSummary, modelsStatus, searchStocks | Dashboard, Scanner, Stock Detail, CommandPalette |
| `lib/api/paper.ts` | fetchPaperPositions, fetchPaperOrders, fetchPaperHistory, fetchPaperSimulation, fetchSimulationRuns, fetchSimulationRun, openPaperTrade, closePaperPosition | Paper, Backtest, TradeModal |
| `lib/api/system.ts` | fetchModelsStatus, fetchBacktestRange, fetchDataQuality, fetchIntradayLive, fetchNotifications, markAllNotificationsRead, fetchCorporateEvents (public), fetchSectorTrends (public) | Live, Backtest, MLOps, Data, Events, Trends, Notifications, LiveMarketStrip |
| `lib/api/public.ts` | fetchPublicTrackRecord | `/track` page |

### API Endpoints Called by Each Page

| Page | Endpoints |
|---|---|
| Dashboard | `GET /scan/large`, `GET /portfolio/summary`, `GET /models/status` |
| Scanner | `GET /scan/{tier}` |
| Live | `GET /intraday/live?symbol=`, `GET /stocks/search?q=` |
| Paper | `GET /portfolio/summary`, `GET /paper/positions`, `GET /paper/orders`, `GET /paper/equity-history`, `GET /paper/simulation`, `GET /paper/simulations`, `GET /paper/simulation/{slug}`, `POST /paper/positions/{id}/close` |
| Backtest | `GET /backtest/range`, `GET /backtest/monthly`, `GET /backtest/distribution`, `GET /paper/simulation`, `GET /paper/simulations`, `GET /paper/simulation/{slug}` |
| Track Record | `GET /public/track-record` |
| MLOps | `GET /models/status` |
| Data | `GET /system/data-quality` |
| Events | `GET /events/corporate` |
| Trends | `GET /sectors/trends` |
| Notifications | `GET /notifications`, `POST /notifications/read-all` |
| Stock Detail | `GET /predict/{symbol}`, `GET /predict/{symbol}/explain`, `GET /intraday/live?symbol=`, `GET /events/corporate`, `WS /api/ws/live/{symbol}` |

---

## Real-Time Features

| Feature | Mechanism | Frequency | Pages |
|---|---|---|---|
| Stock detail live feed | WebSocket (`useLiveStream`) → `wss://{host}/api/ws/live/{symbol}` | Server push | `/stock/[symbol]` |
| WebSocket fallback | REST poll (`refetchInterval: 15000`) when WS not connected | 15s | `/stock/[symbol]` |
| Live market page | `refetchInterval: 10000` (market open only, `isMarketOpen()` gate) | 10s | `/live` |
| Live market strip (dashboard) | `Promise.all` fetch × 7 symbols, `isMarketOpen()` gate | 15s | Dashboard |
| Paper positions | `refetchInterval: 15000` (always, no market hours gate) | 15s | `/paper` |
| MLOps models status | `refetchInterval: 60000` | 60s | `/mlops` |
| IST clock in NavBar | `setInterval` | 10s | All (app) pages |

### WebSocket Protocol (`lib/use-live-stream.ts`)
- URL: `wss://{window.location.host}/api/ws/live/{symbol}` (nginx-proxied)
- Messages: `{type: 'snapshot', candles: [...], tick: {...}}`, `{type: 'tick', ltp: number, ...}`, `{type: 'market_closed'}`
- No auto-reconnect (intentionally simple)
- Displayed status: "Live · streaming" when connected, "Live · ~15-min delayed" when REST

---

## Components

### Layout Components

| Component | File | Purpose |
|---|---|---|
| NavBar | `components/layout/nav-bar.tsx` | Fixed top nav: logo, tabs, "More" dropdown, IST clock, Cmd+K |
| MobileBottomNav | `components/layout/mobile-bottom-nav.tsx` | Fixed bottom tabs for mobile (5 icons) |
| CommandPalette | `components/layout/command-palette.tsx` | Full-screen overlay: stock search + page list. Trigger: Ctrl/Cmd+K or `open-cmdpal` DOM event |

### Dashboard Components

| Component | Props | Purpose |
|---|---|---|
| SignalCard | `stock: ScanStock, onClick?` | Card per stock: symbol, direction badge, confidence ring, price, sparkline, sector badge, watchlist star |
| LiveMarketStrip | (none) | Horizontal scrolling ticker strip — 7 symbols in parallel |
| CorporateEventsWidget | (none) | Compact list of up to 6 upcoming corporate events |

### Scanner Components
| Component | Props | Purpose |
|---|---|---|
| FilterPanel | tier, direction, minConfidence, onChange handlers | Controlled filter bar: Universe / Signal direction / Min Confidence slider |

### Paper Trading Components
| Component | Props | Purpose |
|---|---|---|
| TradeModal | `onClose` | Full-screen modal form: place new paper trade (symbol, side, qty, price) |

### Neural Visualization
| Component | Props | Purpose |
|---|---|---|
| NeuralViz | `models?: ModelsStatusResponse` | SVG animation: features → 3 model nodes → Vedha Core → Signal |

### Core Components (`components/core/`)

| Component | Props | Purpose |
|---|---|---|
| GlassCard | children, className, glow, hover, style, onClick | Glassmorphism container |
| StatCard | label, value, sub, subColor, accent, className, info | Metric display card with optional InfoTip |
| ConfidenceRing | value(0-100), size, strokeWidth, color, showLabel | SVG circular progress ring |
| DirectionBadge | direction('UP'\|'DOWN'), size | Colored UP/DOWN badge |
| RegimeBadge | regime(string) | Color-coded regime label (bull=green, bear=red, sideways=muted, high_vol=amber) |
| SectorBadge | sector(string) | Colored sector label (11 sectors defined) |
| SectionLabel | label, children, className | Section header: amber dot + uppercase + divider |
| InfoTip | info(MetricInfo), align | Hover tooltip: metric definition, formula, plain-English guide |
| AnimatedNumber | value, duration, format, className | Smooth number animation via requestAnimationFrame (ease-out cubic) |

### Chart Components (`components/charts/`) — all pure SVG, zero external deps

| Component | Props | Purpose |
|---|---|---|
| AreaChart | series(data,color,label,dashed), height, baseline | Multi-series area/line chart |
| MiniSparkline | data, width, height, color, fill | Inline sparkline |
| CandlestickChart | candles, height, overlays | OHLC with entry/stop/target overlay lines |
| FeatureBars | features(name,value,color?) | Horizontal bars for feature importance / PSI drift |
| AttentionBar | data(number[60]), height | Vertical bar chart for TFT temporal attention |
| RadarChart | labels, values, size | Polygon radar for model agreement |
| DonutChart | segments(label,value,color), size, strokeWidth | Segmented ring chart (currently unused in pages) |

---

## Lib Utilities

| File | Key Exports | Purpose |
|---|---|---|
| `lib/api/types.ts` | PredictionResponse, ScanStock, ScanResponse, ExplainResponse, PortfolioSummary, ModelInfo, ModelsStatusResponse, PaperPosition, PaperOrder, PaperTrade, TrackRecord, PaperSimulation, SimRunSummary, BacktestRangeResponse, DataQualityResponse, IntradayLiveResponse, OHLCCandle, Tick, Notification, CorporateEvent, SectorTrendsResponse | All shared TypeScript interfaces |
| `lib/utils.ts` | cn(...), formatINR(value), isMarketOpen(), fmtPct(v, decimals) | Shared utility functions |
| `lib/glossary.ts` | GLOSSARY, MetricInfo | Plain-English metric definitions for InfoTip. Fields: label, what, calc, plain. Covers: dir_accuracy, win_rate, avg_net, total_net, profit_factor, sharpe, max_drawdown, track_all, track_gate, track_topk, round_trip_cost, out_of_sample, portfolio_value, daily_pnl, unrealized_pnl, active_positions |
| `lib/use-live-stream.ts` | useLiveStream(symbol, enabled): LiveStream | WebSocket hook for stock detail page |

---

## Build & Deploy

### Docker Build (two-stage)
```
Stage 1 (builder: node:20-slim):
  npm ci
  NODE_OPTIONS=--max_old_space_size=2048    # prevents OOM on VPS during build
  NEXT_TELEMETRY_DISABLED=1
  ARG NEXT_PUBLIC_API_URL=/api              # baked into client bundle
  npm run build → output: "standalone"

Stage 2 (runner: node:20-slim):
  Creates nextjs user (uid 1001, non-root)
  Copies .next/standalone/, .next/static/, public/
  EXPOSE 3000
  CMD: node server.js
```

### Production URL Mapping
- All API calls use relative path `/api/*` (baked at build time via `NEXT_PUBLIC_API_URL=/api`)
- nginx on VPS: `location /api/ { proxy_pass http://api:8000/; }` (strips /api prefix)
- WebSocket: `location /api/ws/ { proxy_pass http://api:8000/ws/; upgrade; }` (feat/nginx-ws-proxy branch)

### Manual Deploy (no CI for UI)
```bash
# On VPS: rebuild UI container after code changes
docker compose -f docker-compose.vps.yml build ui
docker compose -f docker-compose.vps.yml up -d ui
```

### Dev Setup
```bash
npm run dev   # port 3000, hits http://localhost:8000 for API
npm test      # vitest
npm run build # production build
```
