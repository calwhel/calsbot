---
name: cTrader order symbol resolution + metals price source
description: Why cTrader orders need a numeric symbolId (not name) and why metals must never fall back to gold-futures pricing.
---

# cTrader order placement needs symbolId, not symbolName

`ProtoOANewOrderReq` has NO `symbolName` field — it requires the integer
`symbolId`. Setting `req.symbolName` raises `AttributeError` and the order
silently falls back to paper. symbolIds are per-account/per-host (not portable),
so they must be resolved at runtime via `ProtoOASymbolsListReq` (payload
2114/2115) over an app+account-authed connection and cached per `(host, ctid)`.

**Why:** the order-placement client module historically had no symbol
resolution at all (only the price-feed module did), so live forex/metals orders
never actually placed.

**How to apply:** any new cTrader request that targets an instrument must send
`symbolId` resolved for that exact `(host, ctid)` — never a symbol name.

## Latent pre-existing bugs found nearby (NOT yet fixed — out of scope)
- `_PAYLOAD_TYPES['ctid_profile_req/res']` is mapped to 2114/2115, but the real
  ctid-profile payload types are 2151/2152 (2114/2115 are SYMBOLS_LIST). Fix if
  ctid-profile ever needs to work.
- `place_order_units()` (the index/CFD order path) is CALLED at two callsites but
  is NEVER DEFINED in `ctrader_client.py` → index orders crash with NameError.
  Forex/metals path (`place_order`) is unaffected.

# cTrader MARKET orders need RELATIVE SL/TP, not absolute

`ProtoOANewOrderReq` with `orderType=MARKET` REJECTS absolute `stopLoss`/`takeProfit`
prices — error: "SL/TP in absolute values are allowed only for order types:
[LIMIT, STOP, STOP_LIMIT]". For MARKET orders set `relativeStopLoss` /
`relativeTakeProfit` instead: a POSITIVE distance from entry in 1/100_000 price
units, i.e. `max(1, int(round(abs(entry_price - target_price) * 100_000)))` (same
×100_000 wire scaling as absolute prices). The broker applies the sign by side
(BUY sl=entry-rel, SELL sl=entry+rel) and anchors to the ACTUAL fill, so it also
fixes slippage drift vs a stale signal price.

**Why:** the order was reaching the broker (symbolId fix) but silently dropping to
paper on this rejection. Computing relative needs the entry price, so `place_order`
takes an `entry_price` param threaded from `place_ctrader_order_for_user`.

**How to apply:** never send absolute SL/TP on a MARKET order. Amending an existing
position (`ProtoOAAmendPositionSLTPReq`) is different — that one DOES take absolute
SL/TP prices and works.

# Metals price must never fall back to gold futures

For XAUUSD/XAGUSD, `tradfi_prices.get_price` must NOT fall through to yfinance
`GC=F` — that's gold FUTURES, a different instrument from spot (contango premium
of several dollars). It mismatches the cTrader broker quote and can falsely hit
tight TP/SL. Order of truth: cTrader live spot feed (matches the broker that
fills) → Binance/FMP spot → else return `None`.

**Why:** returning `None` makes the executor skip the tick (callers guard
`if px and px > 0`); a wrong-instrument price would inject false fills.
