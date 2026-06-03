---
name: Ghost-cleanup broker-id trap
description: Why the "ghost execution" auto-cancel sweep must check EVERY broker order-id column, not just Bitunix — or it silently cancels live forex trades and suppresses their close notifications.
---

# Ghost-cleanup broker-id trap

The portal runs a periodic "ghost cleanup" sweep (`_do_cancel_ghost_executions_sync`
in `strategy_portal_server.py`) that flips live (`is_paper=false`) OPEN executions
to `CANCELLED` when they look like they never reached a broker. A "ghost" is an
execution whose broker order placement failed, so it has no broker order id.

## The trap
The sweep originally cancelled any live OPEN execution where `bitunix_order_id IS NULL`.
But Bitunix is the CRYPTO broker only. Forex/index trades are placed via cTrader and
carry `ctrader_order_id` — their `bitunix_order_id` is ALWAYS NULL. So the sweep
silently cancelled REAL live forex trades a few minutes after entry.

**Consequence (real incident):** a live XAUUSD short hit its broker stop-loss but the
user got NO push/Telegram alert — the trade had already been flipped to CANCELLED, so
it was no longer OPEN, nothing watched for the SL fill, and CANCELLED produces no
win/loss notification. The forex close-reconcile also can't help, because it only
inspects OPEN rows.

## The rule
A ghost is an execution with NO broker id of ANY kind. The sweep WHERE clause must
require every broker-id column to be NULL together:
`bitunix_order_id IS NULL AND ctrader_order_id IS NULL`.

**Why:** "no Bitunix id" ≠ "never placed" once a second broker exists. Any time a new
broker/order-id column is added to `strategy_executions`, this sweep (and any similar
"never placed" heuristic) must be extended to include it, or that broker's live trades
get false-cancelled.

**How to apply:** when adding a broker, grep for `bitunix_order_id IS NULL` /
`bitunix_order_id.isnot(None)` and audit each for "is this crypto-only or
broker-agnostic?". Recovery for already-mis-cancelled rows:
`backfill_ghost_cancelled_executions` (re-evaluates ghost-cancelled rows as paper
against historical candles) — note it converts them to `is_paper=True`.
