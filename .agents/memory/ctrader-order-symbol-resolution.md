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

# Metals price must never fall back to gold futures

For XAUUSD/XAGUSD, `tradfi_prices.get_price` must NOT fall through to yfinance
`GC=F` — that's gold FUTURES, a different instrument from spot (contango premium
of several dollars). It mismatches the cTrader broker quote and can falsely hit
tight TP/SL. Order of truth: cTrader live spot feed (matches the broker that
fills) → Binance/FMP spot → else return `None`.

**Why:** returning `None` makes the executor skip the tick (callers guard
`if px and px > 0`); a wrong-instrument price would inject false fills.
