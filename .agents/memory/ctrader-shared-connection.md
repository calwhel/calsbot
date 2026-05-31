---
name: cTrader shared persistent connection hazards
description: Why outer asyncio.wait_for around any cTrader call can desync the shared socket, and timeout sizing for cold reconnects
---

# cTrader persistent connection — cancellation & cold-start

`app/services/ctrader_client.py` keeps ONE shared TLS socket to
live.ctraderapi.com (singleton `_conn_reader/_conn_writer`, 45s idle limit),
serialized by `_get_lock()`. Every call (balance/reconcile, place_order,
close_position) reuses it. Two non-obvious traps:

1. **Wrapping a cTrader call in `asyncio.wait_for(...)` is dangerous.** On
   timeout, wait_for cancels the inner coroutine mid-`_send_recv`, leaving an
   unconsumed/partial framed response on the shared socket. The NEXT caller
   (including live order placement on FP Markets) then reads a desynced frame.
   **Fix pattern:** the inner cTrader function must catch `asyncio.CancelledError`,
   call `_invalidate_persistent_connection()`, and re-raise — so the socket is
   dropped and the next call reconnects cleanly. Prefer per-phase deadlines
   *inside* the client over outer cancellation when possible.

2. **Cold-socket reconnect is slow.** When the FX executor isn't trading (e.g.
   weekends, market closed), the socket goes idle >45s and the next balance poll
   pays a full reconnect: SSL handshake + app-auth + account-auth + reconcile
   round-trips. This can exceed a few seconds. Outer timeouts on balance must be
   generous (≥10s) or the UI shows a blank balance with an EMPTY-message
   `TimeoutError` (asyncio.TimeoutError str() is ""). Balance is cached ~20s so
   only the first cold poll pays the cost.

**Note:** FP Markets is a cTrader broker — users connect their FP Markets cTrader
account via OAuth; CTRADER_CLIENT_ID/SECRET is the app, the user's token is
per-account. Balance/reconcile works 24/7 regardless of forex market hours.
