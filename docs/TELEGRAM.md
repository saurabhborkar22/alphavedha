# Telegram Integration

Three independent pieces, each usable without the others:

| Piece | Direction | Transport | Needs |
|---|---|---|---|
| News collector (15-min poll) | inbound | `t.me/s/` preview scrape | nothing |
| Live news watcher (real-time) | inbound | MTProto user session (Telethon) | api_id/api_hash + phone |
| Execution bot (alerts, /halt) | outbound | Bot API | BotFather token + chat id |

## 1. News collector — zero setup

`intel/collectors/telegram_news.py` polls the public preview pages of the
channels in `TELEGRAM_NEWS_CHANNELS` (default: `moneycontrolcom`,
`ndtvprofitnews`, `cnbc_tv18`) every 15 minutes between 07:00–22:00 IST
(scheduler job `telegram_news_ingestion`, also triggerable via
`POST /api/ops/trigger/telegram_news`).

Messages that mention a universe symbol (aliases in `configs/stocks.yaml`
→ `news_aliases`) are upserted into `disclosures` with `source="TELEGRAM"`,
`category="news"`. From there they ride the normal intel pipeline: the
20:00 LLM extraction batch classifies them, and the sentiment aggregator's
`TelegramSource` feeds them to FinBERT.

## 2. Live watcher — real-time red-flag alerts

`intel/telegram_live.py` connects as a *user account* (bots cannot read
third-party channels) and processes messages the second they are posted:
stores them like the collector, then pushes an alert through the execution
bot when a red-flag pattern fires — **critical** severity (fraud, raids,
SEBI action, default, plant shutdown, …) always alerts; **notable**
(resignation, downgrade, pledge, …) alerts only for symbols with an open
paper trade.

Setup (once):

1. Create `api_id` / `api_hash` at <https://my.telegram.org> (any Telegram
   account — consider a dedicated number).
2. Set `TELEGRAM_API_ID`, `TELEGRAM_API_HASH` in the env.
3. Interactive login to mint the session file:

   ```bash
   # local
   python -m alphavedha.intel.telegram_live login

   # VPS (session persists on the telegram-session volume)
   docker compose -f docker-compose.vps.yml --profile telegram \
     run --rm news-watch python -m alphavedha.intel.telegram_live login
   ```

4. Run the daemon: `python -m alphavedha.intel.telegram_live watch`, or on
   the VPS enable the opt-in compose profile:

   ```bash
   docker compose -f docker-compose.vps.yml --profile telegram up -d news-watch
   ```

The account only *reads* a handful of public channels — passive, low
volume, no joins/scrapes of private groups. Do not point it at tips/calls
channels: NSE/SEBI have repeatedly flagged those for pump-and-dump; only
factual news sources belong in the pipeline.

## 3. Execution bot — alerts on your phone

Already implemented in `execution/telegram.py`; the live watcher reuses it
for red-flag pushes.

1. Talk to [@BotFather](https://t.me/BotFather) → `/newbot` → copy the token
   into `TELEGRAM_BOT_TOKEN`.
2. Send `/start` to your new bot, then read your chat id from
   `https://api.telegram.org/bot<TOKEN>/getUpdates` → `TELEGRAM_CHAT_ID`.

Commands: `/status`, `/positions`, `/halt`, `/resume`, `/panic`, `/help`.

## Data flow

```
t.me/s/<channel> ──(15 min poll)──┐
                                  ├─→ disclosures (source=TELEGRAM)
Telethon live stream ──(seconds)──┘        │
        │                                  ├─→ 20:00 LLM extraction → events
        └─→ red-flag classifier            └─→ TelegramSource → FinBERT sentiment
                 │
                 └─→ execution bot alert (critical always; notable if open trade)
```
