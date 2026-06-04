---
name: FVG/iFVG confidence tiers
description: How the low/medium/high quality grade on FVG/iFVG signals is computed and the iFVG fill gotcha.
---

FVG & iFVG signals carry an optional `min_confidence` quality tier (`any`=off/legacy | `low` | `medium` | `high`; alias `confidence`). Grading is **volatility-relative** (ATR multiples), never absolute % — same rule as `signal-quality-thresholds.md`.

**Grade inputs (per gap):** `disp_atr` (formation-candle body ÷ ATR14 — how impulsive the move that left the gap), `size_atr` (gap width ÷ ATR14), `filled`.
- high = disp ≥1.5×ATR AND size ≥0.5×ATR AND untested
- medium = disp ≥0.8×ATR AND size ≥0.25×ATR
- low = any qualifying gap
A tier accepts that grade AND everything above it. Large size alone is NOT enough — displacement is required (a wide doji gap stays low).

**iFVG fill gotcha (the non-obvious bit):** an iFVG is mitigated BY DEFINITION (`filled=True` is the normal case). So the "untested" freshness requirement for `high` must be skipped when grading an iFVG, else a high-confidence iFVG is unreachable and silently never fires. The grader detects iFVG via `cond["type"]=="ifvg"` and only requires freshness for plain FVG.

**Why ATR must be forced:** the detector only computes `disp_atr`/`size_atr` when an ATR filter is active. Grading needs them always, so a `compute_quality` flag forces ATR(14) + a 60-bar lookback whenever a tier is requested.

**Label:** chosen gap's tier is prefixed as `<tier>-conf ` on the trade-card label. The prefix keeps the `Bullish/Bearish FVG` substring intact, so the iFVG dispatch label-inversion and the executor's BOTH-direction inference still match.

**Back-compat:** omit / `any` → no ATR forced, no filter, no label change → identical to pre-tier behaviour for every existing strategy. Invalid tier strings fall back to `any`.
