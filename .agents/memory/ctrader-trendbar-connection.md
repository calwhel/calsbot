---
name: cTrader trendbar (candle) connection
description: Why candle fetches must reuse one lock-serialized cTrader connection, and treat wait_for cancellation as connection-fatal.
---

# cTrader trendbar/candle fetch reliability

**Rule 1 — reuse ONE persistent, lock-serialized connection for trendbar (candle) fetches.**
Do NOT open a fresh SSL connection per candle request.

**Why:** Opening a short-lived connection per request meant dozens of concurrent
SSL+auth handshakes per scan cycle (many symbols × timeframes). cTrader throttles/
rejects that, so *every* trendbar fetch failed and callers silently fell back to
yfinance — which for metals is the wrong instrument (gold FUTURES GC=F, not spot),
mismatching the broker spot feed used for fills/entry. Symptom in prod logs:
constant `[tradfi] klines ok: GC=F` / `=X` and never a broker candle.

**How to apply:** Auth (app+account) happens once on connect, not per request.
Serialize all fetches with an asyncio.Lock so only one request is in flight.
Symbol IDs are cached per (host,ctid) by the streaming connection, so the lookup
is a no-op in steady state. Proactively reopen if idle past ~25s (cTrader drops
idle sockets ~30s). On any stale-socket/read failure, drop the connection and
retry once.

**Rule 2 — a wait_for-cancelled cTrader read must be CONNECTION-FATAL.**
`asyncio.wait_for(timeout)` raises `CancelledError` (a `BaseException`, NOT caught
by `except Exception`). If you don't drop the socket on cancellation, a late/
pending trendbar response stays buffered and the NEXT request reads it as its own
answer — there is no clientMsgId correlation, so this is silent data corruption
(wrong candles for a symbol).

**How to apply:** Catch `asyncio.CancelledError` BEFORE the generic retry handler,
abandon the connection **synchronously** (a no-await drop — you cannot safely
`await writer.wait_closed()` while a cancel is propagating; just `writer.close()`),
then re-raise. Same in the connect/auth path: catch `BaseException`, sync-close the
writer, re-raise. Keep the inner read deadline < the outer wait_for budget so warm
round-trips finish before any forced cancellation.

**Rule 3 — metals never fall back to yfinance for candles either.**
Mirror the get_price fix: in get_klines, if the symbol is a metal, return [] after
the broker + Binance-spot attempts rather than serving GC=F/SI=F futures candles.
Skipping a bar is better than poisoning signal/paper evaluation with wrong-
instrument data.
