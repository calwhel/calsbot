---
name: Forex/gold volume & order-flow data feasibility
description: What volume/order-flow data is actually obtainable for forex & metals on this platform, and why Binance is not an option
---

# Forex/gold volume & order-flow — what's actually possible

When asked for volume-based features (Volume Profile, order flow, delta, footprint) on
forex/gold, the data feed — not the algorithm — is the constraint.

- **Binance is unusable here.** `api.binance.com` returns **HTTP 451** (geoblocked on
  Replit infra, dev AND prod region). That's why the rest of the codebase routes crypto
  through MEXC. Any "Binance spot metals" path (e.g. XAUUSD→XAUUSDT in `tradfi_prices.py`)
  silently fails → falls back to yfinance (near-zero forex volume).
- **There is no real crypto gold pair anyway.** Binance doesn't list spot XAUUSD/XAGUSD
  (closest is PAXG, thin/unrepresentative). MEXC returns "Invalid symbol" for
  XAUUSDT/XAGUSDT/PAXGUSDT. So don't promise "real Binance gold volume" — it doesn't exist.
- **The only reliable volume for FX/gold is broker (cTrader) TICK volume** — count of price
  updates, not contracts — which already backs the kline feed. It's broker-matched to what
  the user actually trades, so volume-profile levels (POC/VAH/VAL) line up with real fills.
- **True order flow / delta / footprint is NOT buildable.** No feed we have (cTrader trendbars
  included) provides the bid/ask aggressor split. cTrader trendbars carry volume but no
  buy/sell breakdown. Don't build "forex order flow" — it would be a tick-volume fake.

**Why:** user premise was "pull gold/silver volume + order flow from Binance"; verified by
live tests it's geoblocked + symbol-less. Built Volume Profile off cTrader tick volume instead.

**How to apply:** any volume-by-price / volume gate for forex/gold must (1) source from the
existing kline volume (broker tick volume), and (2) **no-op (not fire) when feed volume is 0**
(yfinance fallback) rather than firing on noise — surface the reason explicitly.
