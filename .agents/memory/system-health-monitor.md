---
name: System health monitor
description: How the hourly platform health monitor probes connections and reports to the owner on Telegram, and the false-alarm/delivery traps.
---

# Hourly system health monitor

An hourly background sweep verifies the website + every data/broker connection +
executor loops and DMs the owner a Telegram summary. Lives in
`app/services/system_health_check.py`, registered in
`strategy_portal_server.py::_start_executor_tasks` via `_resilient_task` (so it
single-fires in the advisory-locked executor worker).

## Durable design rules (why it's built this way)

- **Always send, every hour — success AND failure.** The user explicitly wants a
  recurring confirmation the platform is alive, not just failure alerts. Don't
  "optimize" it to only-DM-on-problems without asking.
- **Market-aware downgrades prevent weekend false alarms.** Forex-dependent
  probes (FMP, cTrader feed, the forex executor + live-manager heartbeats) must
  be downgraded to ℹ️ info (not ❌ critical) when `is_market_open('forex')` is
  False. cTrader feed dormant *when zero accounts are linked* is also expected,
  not a fault. Without this, every weekend pages the owner.
- **Telegram send must validate `ok==True`.** A raw `await client.post(...)` to
  sendMessage silently "succeeds" on 400/403/429 (wrong bot, owner never opened
  the chat, parse error, rate limit). `_send_report` checks the JSON `ok` flag,
  falls back main→forex bot token, and returns a bool the loop logs as
  delivered vs NOT-delivered. **Why:** the report is worthless if its own
  delivery isn't confirmed.
- **In-process singleton guard (`_MONITOR_ACTIVE`).** The executor reclaim path
  does NOT cancel previously-started background tasks on advisory-lock churn, so
  a reclaim in the same process could spawn a second monitor loop → duplicate
  hourly DMs. The guard refuses a second concurrent loop. **How to apply:** any
  *user-visible-side-effect* executor-owned loop you add needs the same guard
  (or a real task-handle-cancel-on-lock-loss fix), or lock churn duplicates it.

## Executor liveness = heartbeat registry

`strategy_executor.py` has `mark_heartbeat(name)`/`get_heartbeats()` (module
dict, wall-clock). Each scan loop marks at the TOP of its cycle:
`crypto_executor`, `forex_executor`, `forex_live_manager`. The health check flags
a loop critical if it hasn't cycled within 10 min (forex loops only checked when
forex market open). Caveat: marking at cycle *start* means a loop that fails
immediately after marking can still look healthy — acceptable, not a true
liveness proof.

## Dev vs prod

Executor + price feeds are prod-only/disabled in dev, so a dev sweep shows
FMP/cTrader/executor as ⚠️/❌ while DB/website/Telegram/MEXC/Bitunix are ✅. This
is EXPECTED in dev — all green in prod. **User must REPUBLISH** for prod to start
the monitor task.

## Config

`EXECUTOR_HEALTH_INTERVAL` (default 3600s), `PUBLIC_SITE_URL` (default
`https://tradehubmarkets.com`), owner from `settings.OWNER_TELEGRAM_ID`.
