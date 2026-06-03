---
name: cTrader spot feed reconnect resilience
description: Why the cTrader real-time spot feed disconnects periodically and how reconnect must behave to stay effectively always-on.
---

# cTrader spot feed reconnect resilience

cTrader periodically RECYCLES long-lived spot-subscription sockets (observed in
prod as `stream read timeout / disconnect` every few minutes). This is largely
unavoidable, and is made worse because the on-demand trendbar/kline fetch opens
a SECOND authed socket on the SAME `ctidTraderAccountId` as the spot feed
(`tradfi_prices.get_klines` tries cTrader trendbars first for forex/metals) —
two sessions on one account id invites the broker to drop one.

**Rule:** don't try to PREVENT the recycle — make reconnect SEAMLESS so the
sub-second feed is effectively never down.

- After a session that was genuinely alive (received ≥1 inbound frame) drops,
  reconnect almost immediately (~2s), do NOT grow backoff.
- Reserve the growing/capped backoff ONLY for connections that never became
  healthy (connect/auth/subscribe failing) so we don't hammer the broker.
- **`healthy` must flip on the first RECEIVED frame (spot event OR server
  heartbeat), NOT at subscribe time.** If flagged at subscribe, a socket that
  subscribes then receives nothing would be mis-classified as a "healthy drop"
  and reconnect-spin every (read_timeout + fast_delay) during quiet/closed
  markets. Server heartbeats keep frames arriving < read_timeout in normal
  operation, so a real silence past read_timeout genuinely means dead.

**Why:** user requirement was the feed must "run effortlessly"; the old logic
used a flat 30s→300s backoff even after a healthy drop, so gold lost its
sub-second feed for ≥30s after every routine recycle (fell back to the 5s FMP
feed). Single-instancing is already handled (feed starts only in the executor
advisory-lock-winning worker + a done-guard in `start()`); the gap was purely
reconnect cadence.
