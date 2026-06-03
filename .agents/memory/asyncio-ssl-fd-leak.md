---
name: asyncio SSL writer FD leak
description: Bare writer.close() on asyncio SSL StreamWriters leaks file descriptors; must await wait_closed().
---

# asyncio SSL StreamWriter FD leak

**Rule:** Any `asyncio` SSL `StreamWriter` (from `asyncio.open_connection(..., ssl=ctx)`) must be closed via a helper that does `writer.close()` THEN `await asyncio.wait_for(writer.wait_closed(), timeout=5.0)`. A bare `writer.close()` does NOT promptly release the underlying SSL socket file descriptor.

**Why:** Production hit `[Errno 24] Too many open files`, which then cascaded into `could not translate host name "...neon.tech" to address: System error` (DNS can't open a socket) → DB `OperationalError` on every strategy → forex scanner couldn't fetch prices or fire signals. Root cause was bare `writer.close()` on per-call cTrader TLS sockets. The worst offender was `ctrader_price_feed._fetch_trendbars`, which opens a FRESH TLS connection per kline fetch (every ~5s per forex strategy); during a dead-token window each call also failed account-auth and leaked an FD. Leak accumulated to exhaustion ~15h after a fresh publish.

**How to apply:**
- Use the `_aclose_writer(writer)` helper (defined in both `app/services/ctrader_price_feed.py` and `app/services/ctrader_client.py`) at every SSL writer close site — never a bare `.close()`.
- The `wait_for(..., timeout=5.0)` guard is mandatory so a dead peer can't hang the close (this can run while `_get_lock()` is held → bounded ~5s hold, no deadlock).
- From a SYNC function (e.g. `_invalidate_persistent_connection`), schedule it: `asyncio.ensure_future(_aclose_writer(w))` with a bare-close fallback if no running loop.
- Symptom triad to recognize this class of bug: "Too many open files" + DNS "could not translate host name" + DB OperationalError appearing together and worsening over hours since last restart = FD leak, not a DB/DNS outage. A republish/restart clears leaked FDs temporarily but the leak recurs until the close sites are fixed.
- `_fetch_trendbars` now reuses ONE persistent lock-serialized connection (`_get_tb_conn`) — no longer opens per call. So the recurring leak shifted to the two reopen/cancel paths in `ctrader_price_feed.py`: `_get_tb_conn`'s `except BaseException` auth-failure path (fires every reopen during a dead-token window) and the sync `_drop_tb_conn_sync` (fires on wait_for cancel/timeout storms). Both were doing a BARE `writer.close()`.
- FIX (applied): both now do `asyncio.ensure_future(_aclose_writer(w))` with a bare-close fallback only when no loop — same proven pattern as `ctrader_client.py::_invalidate_persistent_connection`. ensure_future is cancellation-safe (doesn't await) yet still runs close+wait_closed to free the FD. The only acceptable remaining bare `.close()` calls are the no-running-loop fallbacks and the one inside `_aclose_writer` itself.
- Executor/price-feed is prod-only → this fix only takes effect after a REPUBLISH; a republish also clears the already-leaked FDs.
