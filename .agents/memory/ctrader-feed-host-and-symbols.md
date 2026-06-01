---
name: cTrader feed — demo/live host, token rotation, symbol-map scoping
description: Non-obvious cTrader Open API gotchas for the price feed & order path — host routing, refresh-token rotation, per-account symbol IDs, required trendbar timestamps.
---

# cTrader Open API — feed & order gotchas

## Demo vs live are SEPARATE hosts — route by the account's isLive flag
A cTrader access token can own BOTH demo and live trading accounts. Each
`ctidTraderAccountId` is reachable ONLY on its matching endpoint:
- live accounts → `live.ctraderapi.com`
- demo accounts → `demo.ctraderapi.com`

**Why:** account auth (and everything after) silently fails if you connect a
demo ctid to the live host or vice-versa — it looks like an auth/token problem
but the token is fine. This caused both stale-price fallbacks AND live-order
failures until host routing was added.

**How to apply:** read `isLive` from the stored `ctrader_accounts` JSON, pick
the host from it, and keep a fallback that tries the OTHER host if the first
auth fails (covers wrong/stale isLive metadata). The order path (`ctrader_client`)
must do the same routing, not just the feed.

## Refresh tokens ROTATE on every refresh — persist the new one or you're locked out
cTrader returns a NEW refresh token on each `refresh_access_token` call and
invalidates the old one. If you don't persist the rotated refresh token, the
next refresh returns ACCESS_DENIED and the account is locked out until
re-linked.

**Why:** a single dropped/duplicated refresh burns the chain.
**How to apply:** do refreshes under a lock, re-read the row, persist BOTH the
new access token and the rotated refresh token atomically. Never log token
values.

## Symbol IDs are per-account — scope the symbol-name→id cache by (host, ctid)
`ProtoOASymbolsListRes` symbolIds are NOT portable across accounts. A globally
cached `name→id` (and reverse `id→canonical`) map reused after a host/account
switch will map spot events / trendbar requests to the WRONG instrument —
worse than stale prices.

**Why:** caught in review; the original cache returned early whenever populated,
regardless of which account it was resolved from.
**How to apply:** tag the cache with the `(host, ctid)` it was resolved from and
force a re-resolve when that context changes.

## ProtoOAGetTrendbarsReq REQUIRES fromTimestamp
Omitting `fromTimestamp` makes the request fail with "missing required fields:
fromTimestamp" and returns zero bars (kline path silently broken).

**How to apply:** set both `fromTimestamp` and `toTimestamp`. Size the window to
`count × timeframe_minutes × 60_000 × ~3` (the ×3 absorbs weekend/closed gaps so
you still get `count` *trading* bars); cTrader caps the returned set at `count`
(max 4096) regardless of window width.

## Latency: bound the cTrader-first attempt before falling back
The feed's `get_klines` does open + app-auth + account-auth + trendbar-read, ×up
to 3 (primary host, token-refresh retry, fallback host). Wrap the cTrader-first
call in `tradfi_prices` with a strict `asyncio.wait_for` budget (~6s) so a
transient cTrader hiccup can't stall before the Binance/FMP/yfinance fallbacks.

## The streaming feed MUST send client heartbeats or the broker drops it
A subscribed spot stream is NOT self-keepalive. cTrader closes any connection
that doesn't send a `ProtoHeartbeatEvent` (payloadType 51) within ~30s — EVEN
while spot events are streaming inbound. Symptom: "stream read timeout /
disconnect" churn every ~30s.

**Why:** the read loop only reads; nothing was writing to the socket, so the
server's client-inactivity timer expired.
**How to apply:** after subscribing, spawn a task that sends a heartbeat every
~10s on the same writer; cancel it on disconnect. Only the heartbeat task writes
post-subscription (read loop only reads) so there's no interleaved-write hazard.

## Run the feed in EXACTLY ONE worker — co-locate with the executor
The portal runs multiple gunicorn workers (`-w 4`) and dev shares the same Neon
DB/account. If the feed starts per-worker (or in dev too), every worker opens its
own connection to the SAME cTrader account → the broker kicks the duplicate
sessions ("app auth failed" + reconnect churn).

**Why:** cTrader limits concurrent sessions per account; the spot cache is
per-process anyway, and its only critical consumer is the executor (which already
runs in a single advisory-lock-winning, prod-only worker).
**How to apply:** start the feed inside the executor-lock-win path (NOT the
per-worker startup hook) so exactly one connection exists, co-located with its
consumer, and disabled in dev. CRITICAL: on advisory-lock LOSS/failover, call a
feed `stop()` in the former holder, else its feed lingers alongside the new
holder's feed and the duplicate-session churn returns.

## Price scaling
Trendbar/spot prices are integer-scaled by 100_000 (divide by 100_000 to get the
real price). See `ctrader-price-scaling.md` for the order-side wire convention.
