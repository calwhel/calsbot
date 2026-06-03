---
name: cTrader price wire scaling
description: cTrader SL/TP/price fields are sent/read ×100000 despite the proto type being double — match this convention for any new price field.
---

# cTrader price wire scaling (×100000)

In `app/services/ctrader_client.py`, all order/position **price** fields
(`stopLoss`, `takeProfit`, deal `executionPrice`, position `price`) are nominally
protobuf type `double`, BUT the working broker convention for this account is the
**price multiplied by 100000**.

**The ×100000 rule holds ONLY for 5-digit FX majors — price scaling is
SYMBOL-DEPENDENT.** This was learned the hard way: a live XAUUSD fill came back as
`deal.executionPrice = 4452.76` (the TRUE price, UNSCALED), so the fixed
`executionPrice / 100_000.0` produced `0.0445276`, and the executor's delta-shift
then turned SL/TP into garbage (`-4.95547` / `10.0445`) on the trade card.

**RELATIVE offsets ARE universal; ABSOLUTE prices are NOT.**
- `place_order` sends `relativeStopLoss/relativeTakeProfit = abs(entry-level)*100_000`.
  Relative offsets are spec'd as "1/100000 of a price unit" and are
  symbol-INDEPENDENT — this is why gold orders place correct stops. Keep using
  relative offsets for orders.
- `deal.executionPrice` (and other ABSOLUTE price fields) arrive scaled by the
  symbol's digits: FX majors ×10^5, metals/indices effectively ×1. Do NOT divide
  by a fixed constant.

**Reading a fill price (robust):** the fill is always near the known signal
`entry_price`, so pick whichever interpretation lands closest:
`min([raw, raw/100, raw/1000, raw/100_000], key=lambda v: abs(v-entry_price))`.
This auto-resolves FX (raw/100000→1.085) and gold (raw→4452.76).

**KNOWN LATENT BUG (not yet fixed):** `modify_position_sltp` (live breakeven/
trailing) still sends ABSOLUTE `stopLoss/takeProfit` as `int(price*100_000)`. That
is correct for FX majors but ~1000× wrong for gold (gold absolute prices are
unscaled on this account). Before relying on live SL amends for metals, migrate it
to per-symbol digit scaling OR to relative-offset amends. Orders are unaffected
(they use relative offsets).

# Live forex broker positionId

cTrader amend/close requests need the broker `positionId`, which is only available
on the `ProtoOAExecutionEvent.deal.positionId` at fill time. `StrategyExecution` has
no column for it, so it is stashed in `notes` as a `pos=<id>` token at the live-order
callsites and parsed back out by the live position manager. If you ever rewrite the
notes field, preserve that token or live SL amends silently stop working.
