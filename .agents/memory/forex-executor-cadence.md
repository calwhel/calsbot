---
name: Forex executor cadence & DB-stress backoff
description: How fast the forex scan loop can safely run, why, and the backoff guard that protects Neon.
---

# Forex executor scan cadence

**Rule:** Lowering the forex scan interval (`FOREX_SCAN_INTERVAL_SECONDS`) is only worth doing if TWO conditions hold, and it must never be done without a DB-stress guard.

1. **Freshness floor is the cache TTL, not the interval.** Forex/index live prices come from the FMP feed (`fmp_price_feed.py`) which refreshes every ~5s, consumed via `tradfi_prices.get_price`. The executor caches computed price/TA in `_PRICE_TA_CACHE`. If that TTL is longer than the scan interval, a faster scan just re-evaluates stale data and gains nothing. Keep a separate TTL for the non-crypto path (`_PRICE_TA_TTL_TRADFI`) aligned to the ~5s feed cadence. Lowering it is DB-free: the slow 15m klines keep their own 20s cache in `tradfi_prices`, so only the in-memory FMP spot price re-reads.

2. **Scan frequency = DB duty cycle, NOT cache TTL.** Each cycle runs per-strategy gating queries (daily/open counts, last-fired, forex guards) on the small shared background pool (`pool_size=3, max_overflow=4`, shared with the crypto executor). Halving the interval doubles query throughput.

**Why the guard matters:** A prior incident at 3s interval + concurrency 5 saturated Neon → `QueryCanceled` → HTTP 500s on `/api/strategies`. That incident is why the interval was raised to 10s and concurrency capped at `MAX_CONCURRENT=3`.

**How to apply (the safe pattern):** keep the loop sequential with a fixed post-work sleep as baseline, AND add adaptive backoff: init a `_cycle_db_skipped` list BEFORE the try (so it's safe on early-throw), append on per-strategy DB errors, and append a sentinel in the OUTER `except` (covers strategy-list/session-open failure — the hot-table query that precedes saturation). After each cycle, if non-empty sleep `max(interval*3, 15)s` + warn, else the fast interval. This auto-throttles the instant DB pressure appears and auto-recovers on the next clean cycle. Don't go below ~5s and don't remove the fixed sleep.

**Note:** executor is prod-only/disabled in dev, so cadence changes can only be verified by compile + logic review, never live firing locally. User must republish for prod to pick up a new default.

## The real cycle-time bottleneck is candle fetches, not the scan interval
A forex cycle was observed in prod taking ~2 min (not the 5s `FOREX_SCAN_INTERVAL` target) for ~40 strategies. The interval was NOT the cause — the cycle *body* was slow. Per-strategy condition evaluation pulls its own OHLC via `tradfi_prices.get_klines` (forex falls through cTrader→yfinance `download()`, ~1-3s each). With ~28 strategies on the SAME symbol (XAUUSD) + several timeframes, and only a 20s `_KLINE_CACHE` (sequential dedup only), every cold/expired-cache moment triggered a **thundering herd**: many concurrent strategies each launched their own duplicate slow download. Logs show the same `SYMBOL 15m → 100 bars` fetched 9× in one cycle.

**Fix:** a single-flight registry (`_KLINE_INFLIGHT` keyed by symbol/asset_class/timeframe/limit) in `tradfi_prices.get_klines` — concurrent identical requests await one in-flight fetch instead of duplicating. Collapsed a 36-call herd to 3 real fetches in test.

**Gotchas when wrapping an async producer in single-flight:** (1) `asyncio.CancelledError` is a `BaseException`, so a bare `except Exception` does NOT catch it — a cancelled producer leaves piggy-backed waiters hanging forever; add `except asyncio.CancelledError: fut.cancel(); raise`. (2) In `finally`, only pop the registry if `_KLINE_INFLIGHT.get(key) is fut` (don't clobber a newer entry). (3) Guard piggy-backing on `fut.get_loop() is running_loop` (per-process single loop holds today, but cheap insurance). Single-flight shares the same list object to all waiters — same aliasing the cache already had, fine for read-only consumers.

**Why this beats just lowering the interval:** the herd was the cost; deduping it shrinks the cycle ~5-10× with zero extra DB or API load (fewer calls, not more), and changes no fired signals.
