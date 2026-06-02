---
name: Forex breakeven/exit triggers must be pip/price-based
description: Why forex auto-breakeven (and trailing) cannot use the crypto leveraged-ROI trigger, and how the pip trigger + legacy fallback work.
---

# Forex breakeven must be pip/price-based, not leveraged-ROI

**Rule:** Any forex auto-exit trigger (breakeven, and by the same logic trailing)
must be expressed as an **absolute price level** (pips from entry, or % of the
distance entry→TP), NOT as the crypto-style `price-move% × leverage >= threshold%`.

**Why:** Forex/metals strategies run at **leverage = 1** (sizing is by lots, not a
leverage multiplier). With lev=1 the leveraged-ROI trigger collapses to the raw
price-move %, so a "60%" or even default "70%" breakeven needs the *price itself*
to move 60–70% — impossible for a 50-pip gold/EURUSD trade (fractions of 1%).
Result before the fix: breakeven literally never fired on any forex trade
(0 BREAKEVEN outcomes across 100+ forex execs); the crypto leveraged path was
fine and must stay untouched.

**How to apply:**
- Shared helper `_compute_be_trigger_price(symbol, entry, direction, tp_price, ex_cfg)`
  in `strategy_executor.py` returns the absolute trigger price for forex:
  - primary: `breakeven_at_pips × forex_engine.pip_size(symbol)` from entry in the
    profit direction (this is the user-chosen, broker-style semantic; matches the
    mobile pip control).
  - legacy fallback: `breakeven_pct`/`breakeven_at_pct` treated as **% of distance
    to TP** → `entry + frac*(tp_price - entry)` (works for LONG and SHORT because
    SHORT's tp_price < entry). Keeps old percent-built forex strategies alive
    without silently editing their config.
- Gate on `asset_class == 'forex'` (paper monitor) — crypto keeps leveraged ROI.
  Live forex manager is already forex-only.
- Pip size must come from `forex_engine.pip_size` (see metal-pip-size-drift.md) so
  gold=0.01, silver=0.001 stay consistent.
- Web wizard: forex breakeven control is in **pips** → `exit.breakeven_at_pips`
  (nulls `breakeven_pct`); crypto stays % ROI → `breakeven_pct`. Mobile already
  emits `breakeven_at_pips` for forex.

**Gotcha:** when you delete the per-cycle `roi` calc from the live manager,
also remove its leftover use in the success log line, or it NameErrors inside the
per-position try (swallowed, but every amend logs a spurious error).

**Known not-yet-fixed sibling:** forex *trailing* still reads `trailing_stop_pct`
(price-% only); `trailing_stop_pips` is converted to pct only in the entry path
(`evaluate_and_fire`, not persisted), so pip-based trailing has the same class of
gap. Out of scope for the breakeven task — fix the same way if asked.
