---
name: cTrader rotating refresh-token cross-process race
description: Why the cTrader link kept dying within hours of re-linking, and the cross-process lock that fixes it.
---

# cTrader refresh-token rotation must be serialised ACROSS processes, not just within one

**Rule:** Any refresh of a cTrader OAuth token must be serialised across ALL gunicorn
worker processes, keyed per user. A per-process `asyncio.Lock` is NOT enough.

**Why:** cTrader rotates the refresh token on every use (single-use chain). The portal
runs `gunicorn -w 4`. The price feed + executor live in one worker (pg advisory lock),
BUT `tradfi_prices.get_price`/`get_klines` are also called from ordinary API request
handlers (charts, backtests, live-forex account) served by ANY of the 4 workers; for
forex/metals those go through the cTrader trendbar fetch, which on auth failure calls
`refresh_user_ctrader_token`. So two processes can refresh the same single-use token
near-simultaneously → one rotation is lost → the chain permanently bricks → every
later refresh returns `ACCESS_DENIED` → feed never reconnects. The tell-tale symptom
was the link dying within HOURS of each re-link (not the ~weeks of normal expiry), and
prices silently falling back to yfinance `GC=F` (gold FUTURES) which differs from
cTrader XAUUSD SPOT → "price doesn't match my chart".

**How to apply:** `refresh_user_ctrader_token` takes a Postgres transaction-scoped
advisory lock with the two-key (namespace, user_id) form
(`pg_try_advisory_xact_lock(_TOKEN_REFRESH_PG_LOCK_NS=42_424_244, user_id)`) so refreshes
serialise per-user across processes; xact locks auto-release on commit/rollback/close
(no leak path). If the lock can't be acquired (~10s of 0.5s retries) the function
re-reads prefs and returns the CURRENT access token rather than racing the peer.
`db.commit()` persists the rotated token AND releases the lock atomically. Keep the
per-process `asyncio.Lock` too (cheap intra-process guard). The fix only PREVENTS future
bricking — a chain already dead needs a one-time re-link + republish.
