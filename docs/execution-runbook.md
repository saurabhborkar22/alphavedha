# Execution Engine — Ops Runbook

## Arming Checklist

Before setting `EXECUTION_ENABLED=1`, verify **all** items:

1. **Gate review passed** — at least one strategy passes all §13 criteria
2. **Broker account active** — Kite Connect API key + secret configured
3. **Daily login flow working** — access token refreshed daily (manual or automated)
4. **Kill switch config reviewed** — max positions (8), daily exposure (25%), daily loss (2%), drawdown (6%)
5. **Telegram bot active** — receiving `/status` responses, `/panic` tested in shadow
6. **Shadow mode ran 5+ days** — no errors, slippage within budget
7. **Capital allocated** — initial ₹50,000 in trading account
8. **Position cap set** — max 5% per position (₹2,500 at ₹50k capital)

## Environment Variables

```bash
# Master switch — must be "1" to place real orders
EXECUTION_ENABLED=0

# Broker
KITE_API_KEY=<your-api-key>
KITE_API_SECRET=<your-api-secret>
KITE_ACCESS_TOKEN=<daily-token>

# Telegram
TELEGRAM_BOT_TOKEN=<bot-token>
TELEGRAM_CHAT_ID=<your-chat-id>
```

## Daily Login Flow

Kite Connect requires a daily access token via OAuth redirect:

1. User opens `generate_login_url()` in browser
2. Logs in → redirected with `?request_token=xxx`
3. Call `set_access_token_from_request_token(request_token)` 
4. Token valid until next 6 AM IST

**TODO:** Automate via headless browser or Kite's TOTP flow when scaling.

## Key Rotation

- **API key/secret**: Rotate via Kite developer console. Update env vars, restart.
- **Access token**: Expires daily. Not stored — generated fresh each session.
- **Telegram bot token**: Rotate via @BotFather. Update env var, restart.

## What `/panic` Does

1. Kill switch manually halted → all new orders blocked
2. OMS iterates all open positions → places MARKET SELL for each
3. Telegram notifies: "PANIC executed. N flatten orders placed. Engine HALTED."
4. Engine stays halted until operator runs `/resume`

**Recovery after panic:**
1. Verify all positions actually closed (check broker dashboard)
2. Investigate why panic was triggered
3. When safe: `/resume` to clear manual halt
4. Kill switch re-evaluates all limits on next order attempt

## Kill Switch Limits

| Limit | Default | Trips when | Action |
|-------|---------|------------|--------|
| Master switch | OFF | `EXECUTION_ENABLED != "1"` | Block all orders |
| Max positions | 8 | Open positions ≥ 8 | Block new orders |
| Daily exposure | 25% | Today's new order value ≥ 25% of equity | Block new orders |
| Daily loss | 2% | Day P&L ≤ -2% | **Flatten all** + halt |
| Drawdown | 6% | Equity ≤ 94% of peak | **Flatten all** + halt |
| Manual halt | - | Operator `/halt` or `/panic` | Block all orders |

## Monitoring

- **Telegram `/status`**: Quick check — shows halt state, positions, P&L, drawdown
- **Shadow fills table**: `SELECT * FROM shadow_fills ORDER BY fill_date DESC LIMIT 20;`
- **Slippage check**: Compare `sim_fill_price` vs `decision_price` in shadow_fills

## Scale Ladder

1. **₹50,000** — first 4 weeks, Telegram-approved only, position cap ₹10k
2. **₹2,00,000** — after passing live review (live fills vs paper)
3. **₹5,00,000** — after passing G2 gate review
