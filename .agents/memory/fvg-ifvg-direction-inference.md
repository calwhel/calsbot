---
name: FVG/iFVG direction inference for BOTH strategies
description: How a "BOTH"-direction strategy picks LONG vs SHORT, and why iFVG inverts the gap bias
---

When a strategy's `direction` is `BOTH`, the executor must infer the trade side from the
directional signal itself — NOT a generic RSI coin-flip. The inference lives in
`evaluate_and_fire` (strategy_executor.py) right before TP/SL pricing.

**Rules:**
- FVG: bullish gap → LONG, bearish gap → SHORT.
- iFVG (inverse FVG): the gap is violated/inverted, so bias flips — bullish gap → SHORT
  (now resistance), bearish gap → LONG (now support). The `ifvg` condition type shares
  `eval_fvg` but the DIRECTION must be inverted relative to FVG.
- order_block / divergence / cod: use the pinned config direction (bullish→LONG, bearish→SHORT).
- RSI (`>50`→LONG else SHORT) is the LAST RESORT only when no passed directional condition
  yields a side.

**Why:** an iFVG strategy with `direction:'any'` + `condition:'price_in_gap'` fired a SHORT
on a "Bullish FVG" purely by RSI coincidence, because the inference loop didn't handle
`ifvg` at all (only `fvg`/`order_block`/`divergence`) and fell through to RSI. The bullish
gap and the short side were decided by two unrelated things.

**How to apply / gotchas:**
- When the condition's `direction` is `any`/`both`/unset, the gap side isn't in config —
  read it from the fire's `details` strings (`eval_fvg` labels gaps "Bullish FVG"/"Bearish FVG").
- `details` is INDEX-ALIGNED with `entry_conditions.conditions` (same list, `asyncio.gather`
  preserves order) and each line is prefixed `✅`/`❌`. ALWAYS bind the gap side to the
  specific condition that PASSED (check `details[i]` starts with `✅`) — never a global scan,
  or an OR strategy's FAILED FVG line ("❌ No bearish FVG") can drive the wrong side.
- Normalize config direction strings to lowercase before comparing (defensive vs BULLISH/Bullish).
- This is string-coupled to `eval_fvg`'s label wording — changing "Bullish/Bearish FVG" text
  there would silently break side inference for `any`-direction FVG/iFVG.
- Executor is prod-only/disabled in dev → republish to ship; can't live-fire in dev.
