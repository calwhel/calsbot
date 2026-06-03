---
name: Metal pip-size canonical source
description: All pip-size lookups for metals must agree with forex_engine.pip_size, or entry vs close cards + lot sizing drift by 10–100×. Gold uses the $0.10 retail/broker pip convention.
---

# Metal pip-size canonical source

`app/services/forex_engine.py::pip_size` (backed by `_METAL_PIP_SIZES`) is the
**single source of truth** for pip size. The DB `pips_pnl` storage, the entry/
close cards, the historical pips backfill, and live cTrader lot sizing all
derive from it (directly or via a value that must be kept in lockstep).

## Gold (XAUUSD) = $0.10 per pip — the retail/broker convention

Gold pip = **0.10** (NOT 0.01). This matches FP Markets / broker terminals and
how traders actually count pips: a $2.45 move ≈ 25 pips. A 25-pip stop = $2.50.

**Why:** an earlier session set gold to 0.01 ("smallest price increment") to
make the entry card and close card agree. That fixed the *display* mismatch but
in the WRONG direction — at 0.01 a user's "25 pip" gold stop became only $0.25,
≈ one full gold spread, so live gold trades stopped out almost instantly (and
P&L over-reported pips 10×). The correct fix is to align everything UP to the
broker's $0.10, not down to $0.01.

## The lockstep set (change ALL together or it drifts)

- `forex_engine._METAL_PIP_SIZES['XAUUSD'] = 0.10`
- `forex_engine.TYPICAL_SPREADS_PIPS['XAUUSD'] = 2.5`  (real ~$0.25 spread ÷ $0.10)
- `ctrader_client._PIP_SIZES['XAUUSD'] = 0.10`  (was missing → defaulted to 0.0001, 1000× off)
- `ctrader_client` per-symbol `pip_value` for XAU = **10.0** ($10/pip/lot, 100oz×$0.10).
  This MUST move with pip_size: risk-sizing is `lots = risk / (sl_pips × pip_value)`,
  so keeping `sl_pips × pip_value` = real $/lot loss keeps $ risk per trade invariant.
- `strategy_executor.py::_pip_size` (close/result card) XAU = 0.10
- web `strategy_portal.html`: `_pipSz`, the live-forex pair list `pip:0.10`, the
  lot calculator `pipVal`, and the lot/paper-account hints.

Silver (XAGUSD)=0.001 and platinum (XPTUSD)=0.01 were left at the digits-based
value (out of scope — only gold was reported). They have the same latent 10×
retail-vs-point ambiguity if ever traded heavily.

## Historical exception

`strategy_portal_server.py::_backfill_pips` SQL keeps gold at **0.01** on
purpose. It only fills `pips_pnl IS NULL` (pre-column legacy rows), so it keeps
old gold rows consistent with the era they closed in. Do NOT "align" it to 0.10
or historical pip stats shift 10×. (Not date-bounded — relies on the invariant
that NULL pips only exist for legacy rows; live closes always write non-null.)

## How to apply

When adding/touching any pip calculation, make it agree with
`forex_engine.pip_size`. Prefer importing/calling it over re-deriving. If a new
metal/commodity is introduced, add it to `_METAL_PIP_SIZES` FIRST so every
surface inherits the same value, and update `ctrader_client` pip_value in step.
