---
name: cTrader price wire scaling
description: cTrader SL/TP/price fields are sent/read ×100000 despite the proto type being double — match this convention for any new price field.
---

# cTrader price wire scaling (×100000)

In `app/services/ctrader_client.py`, all order/position **price** fields
(`stopLoss`, `takeProfit`, deal `executionPrice`, position `price`) are nominally
protobuf type `double`, BUT the working broker convention for this account is the
**price multiplied by 100000**.

**The rule:** when SENDING a price to cTrader, write `int(price * 100_000)`. When
READING a price back, divide by `100_000.0`.

**Why:** `place_order` empirically writes `int(stop_loss_price * 100_000)` and reads
`actual_fill = ev.deal.executionPrice / 100_000.0`, and that round-trips to correct
live fills in production. A naive read of the proto schema (type=double = absolute
price) would tell you to send the raw price (e.g. 1.08 for EURUSD) — that is WRONG
for this broker and would set a stop ~5 orders of magnitude off. Always mirror the
existing place_order scaling, not the nominal proto type.

**How to apply:** any new cTrader request that carries a price (e.g.
`modify_position_sltp` for breakeven/trailing amends) must use `int(price*100_000)`.

# Live forex broker positionId

cTrader amend/close requests need the broker `positionId`, which is only available
on the `ProtoOAExecutionEvent.deal.positionId` at fill time. `StrategyExecution` has
no column for it, so it is stashed in `notes` as a `pos=<id>` token at the live-order
callsites and parsed back out by the live position manager. If you ever rewrite the
notes field, preserve that token or live SL amends silently stop working.
