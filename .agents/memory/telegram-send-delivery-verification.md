---
name: Telegram send delivery verification
description: Why trade-notification Telegram sends must verify ok==true and retry, not fire-and-forget.
---

# Telegram trade-notification sends must verify delivery + retry

The executor's `_tg_send` (app/services/strategy_executor.py) is the single send
path for ALL live/paper/breakeven trade DMs (crypto close, forex close, paper
close, breakeven alert all route through it via `_send_paper_close_dm` / direct).

**Rule:** a trade notification send must (1) RETRY transient failures and (2)
confirm Telegram returned HTTP 200 with `ok==true`. Only then count it delivered.

**Why:** A real live XAUUSD stop-out was detected & closed correctly by the
FX-reconcile loop, but the owner got no DM — the log showed only
`Telegram DM failed for <id>:` with an EMPTY error. Two traps:
- `httpx.AsyncClient.post` does NOT raise on a non-2xx, so a 400 (HTML parse
  error — see telegram-card-html-escaping.md) or 403 (user blocked / chat not
  found) was silently treated as success with no log at all.
- A single attempt with no retry meant a transient timeout (cTrader feed
  reconnect / FD-pressure window produced an empty-message exception) dropped the
  ONLY send attempt → no notification, no useful log.

**How to apply:**
- Retry ~3x with backoff on transport exceptions; log `type(e).__name__` (empty
  `str(e)` is common and useless on its own).
- Inspect the response: 200+`ok==true` = delivered; 200 with unparseable body is
  NOT confirmed (retry); 4xx short-circuit (retry can't fix a parse error/blocked
  chat) and log status+body snippet.
- Return bool. Most callsites only log on failure (no durable outbox) — acceptable
  for now, but any "guaranteed delivery" requirement needs a DB-backed retry queue.
- The portal-side `_send_report` (health monitor) already does ok==true + main→forex
  token fallback; keep both send paths consistent.
