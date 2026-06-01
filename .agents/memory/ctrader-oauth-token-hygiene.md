---
name: cTrader OAuth token hygiene
description: How cTrader refresh-token rotation breaks the auth chain, and the secret-leak traps in OAuth logging.
---

# cTrader OAuth token hygiene

## Refresh-token rotation can permanently brick the link
cTrader ROTATES the refresh token on every successful refresh and invalidates the
previous value. If two callers refresh the SAME token concurrently — or a rotated
token isn't persisted — the chain breaks and every later refresh returns
`errorCode == "ACCESS_DENIED"` (HTTP is still `200 OK`; the error is in the JSON
body). Once denied, BOTH the order path and the price feed fail account auth
("account auth failed"), and the only fix is the **user re-linking** cTrader.

**Why:** the module-level `_token_refresh_lock` (asyncio.Lock) only serializes
within one process. Running the feed/order refreshers across multiple gunicorn
workers + dev let them clobber each other's rotated tokens → dead chain. Keeping
the feed/executor in a single prod-only worker concentrates refreshes under one
lock and prevents re-burning after a re-link.

**How to apply:** when "account auth failed" / paper-fallback shows up, check prod
logs for `ACCESS_DENIED — user must re-link`. If present, it is NOT a host-routing
or order-code bug — the token is dead; tell the user to re-link. After ACCESS_DENIED,
a cooldown (`_refresh_denied[user_id] = (denied_token, monotonic+300s)`) stops the
system hammering the OAuth endpoint every signal tick; it is keyed to the EXACT
denied token so a re-link (new token value) auto-clears it, and is cleared on any
successful refresh. Only cooldown on `ACCESS_DENIED`, not any errorCode (others may
be transient).

## NEVER interpolate raw exceptions in OAuth/token paths — they leak secrets
The token-exchange + refresh calls send `code`/`refresh_token`/`client_secret` as
URL **query params**. That means secrets leak through several sinks:
- `httpx` logs every request URL at INFO → silence it: `logging.getLogger("httpx").setLevel(logging.WARNING)`.
- `httpx.HTTPStatusError`'s string includes the full URL → never log `{e}`; log
  `type(e).__name__` + `getattr(getattr(e,"response",None),"status_code",None)`.
- SQLAlchemy/driver errors + `traceback.format_exc()` can echo bound parameter
  values (the token columns) → in token-persist/UPSERT handlers log only `type(e).__name__`.
- Never log the OAuth response body (`exchange_code`) — it contains the tokens; log
  status + `list(data.keys())` only.

**How to apply:** any new code in the cTrader callback / exchange_code /
refresh_user_ctrader_token paths must sanitize every `except` block this way.
