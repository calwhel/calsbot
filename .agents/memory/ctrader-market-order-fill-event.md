---
name: cTrader market-order fill event (ACCEPTED before FILLED)
description: Why a live cTrader order's entry/SL/TP on the card differed from the broker, and how to capture the real fill.
---

# cTrader market-order fill event

A cTrader MARKET order emits TWO execution events in sequence:
1. `ORDER_ACCEPTED` — carries the `order` block (orderId) but **NO `deal`**.
2. `ORDER_FILLED` — carries the `deal` (`deal.executionPrice`, `deal.positionId`,
   `deal.orderId`) AND an `order` block with the same orderId.

**Trap:** reading only the FIRST execution event (e.g. a `_send_recv_any` that
returns the first matching frame) gets ACCEPTED → `actual_fill=None`,
`position_id=None`. Consequences seen in prod (live XAUUSD): the Telegram card
showed the pre-fill SIGNAL price (~13 pips off the real fill), and the live
SL/TP manager could never find the position (no `pos=<id>` in notes).

**Two further traps it caused:**
- Leaving the FILLED frame UNCONSUMED in the persistent socket buffer means the
  NEXT order's first read could pick up the previous order's stale FILLED →
  misattribution. Consuming through FILLED also cleans the buffer.
- A %-based "did the fill differ enough to overwrite entry?" threshold (e.g.
  0.05%) is WRONG for gold: 13 pips ≈ 0.028% < 0.05% so it never applied, yet
  that's half a 25-pip stop. Use an absolute/near-zero epsilon, not a %.

**Rules:**
- After the first execution event, if it has no `deal`, keep draining the SAME
  lock-held socket until the FILLED `deal` arrives or a terminal
  rejection/error, under ONE absolute time budget (not per-read) — market orders
  fill or reject within seconds.
- Correlate every drained event to the order by `order.orderId` / `deal.orderId`
  (fail-closed when an event can't be correlated) so a concurrent trade's event
  on the shared account socket is never misattributed.
- "Accepted but no fill seen in budget" must NOT be returned as an error — the
  order is genuinely live at the broker; erroring would false-fallback a real
  live position to paper. Return order_id with `actual_fill=None` and log; the
  broker still enforces the SL/TP set at entry.

**Broker SL/TP = relative offsets on the REAL fill.** `place_order` sends
`relativeStopLoss/relativeTakeProfit = abs(signal_entry − sl/tp)×100000`, which
cTrader applies to the actual fill. So when you overwrite `entry_price` with the
fill, you MUST shift `sl_price`/`tp_price`/`tp2_price` by the SAME
`delta = fill − signal_entry` (works for LONG and SHORT) so the card + the
app's paper-style monitoring match what the broker actually holds. Do this at
BOTH the owner and subscriber live-order callsites.
