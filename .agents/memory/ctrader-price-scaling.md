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

**ABSOLUTE amend prices = REAL price, NO scaling (fixed).** `modify_position_sltp`
(live breakeven/trailing) sends `req.stopLoss/takeProfit = float(real_price)`. The
proto fields are `double` and the broker quotes absolute prices unscaled — PROVEN
live: a gold stop of ~4452 was ACCEPTED, whereas `4452.76*100_000 = 445276000`
would be rejected. The old `int(price*100_000)` meant gold breakeven NEVER reached
the broker; worse, on one trade a corrupted entry `0.0445276*100_000 ≈ 4452` was
accepted by accident and fired a FALSE "risk-free" Telegram alert on a position
whose stop never actually moved (it then ran to a big loss). Do NOT reintroduce
×100_000 on these absolute double fields. (Relative order offsets stay ×100_000 —
different field, different spec.)

# Live forex broker positionId

cTrader amend/close requests need the broker `positionId`, which is only available
on the `ProtoOAExecutionEvent.deal.positionId` at fill time. `StrategyExecution` has
no column for it, so it is stashed in `notes` as a `pos=<id>` token at the live-order
callsites and parsed back out by the live position manager. If you ever rewrite the
notes field, preserve that token or live SL amends silently stop working.
