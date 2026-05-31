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
