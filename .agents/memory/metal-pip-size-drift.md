---
name: Metal pip-size canonical source
description: All pip-size lookups for metals must agree with forex_engine.pip_size, or entry vs close cards drift by 10–100×.
---

# Metal pip-size canonical source

`app/services/forex_engine.py::pip_size` (backed by `_METAL_PIP_SIZES`) is the
**single source of truth** for pip size. The DB `pips_pnl` storage, the pips
backfill SQL, and the entry card all derive from it.

Metals are digits-based: gold (XAUUSD) digits=2 → 0.01/pip, platinum (XPTUSD)
digits=2 → 0.01/pip, silver (XAGUSD) digits=3 → 0.001/pip.

**Why:** a private duplicate pip table (e.g. the close/result-card `_pip_size`
helper in `strategy_executor.py`) once returned 0.10 for gold while the engine
used 0.01 — so a 0.3-price SL move showed "-3 pips" on close but "30 pips" on
the entry card (10× drift). Platinum had the inverse latent bug: the engine
fell back to 0.0001 (100× off) because XPTUSD wasn't in `_METAL_PIP_SIZES`.

**How to apply:** when adding/touching any pip calculation, make it agree with
`forex_engine.pip_size`. Prefer importing/calling it over re-deriving. If a new
metal/commodity is introduced, add it to `_METAL_PIP_SIZES` FIRST so every
surface inherits the same value.
