---
name: Web wizard local-compile must mirror mobile packCondition
description: The web wizard's client-side condition compiler (non-crypto path) must wrap indicator-route signal types and remap RSI/vwap_deviation exactly like mobile, or strategies silently never fire / over-fire.
---

# Web wizard local-compile parity with mobile packCondition

The web wizard (`app/templates/strategy_portal.html`) compiles a strategy
client-side for **non-crypto** asset classes (forex/stock/index) before saving
(crypto goes through the AI/server path instead). That local compile must
produce the SAME executor-facing condition shapes as mobile's
`mobile/lib/conditionPack.ts::packCondition`.

**Rule:** any signal type that is *indicator-route* in the executor
(`strategy_ta.py` dispatch only auto-wraps `sma*`/`supertrend`; everything else
like `ema/macd/bb/ichimoku/donchian/rsi/adx/cci/mfi/roc/keltner/squeeze/...`
must arrive as `{type:'indicator', name:X, ...}`) MUST be wrapped via a name map
mirroring mobile's `INDICATOR_NAME_MAP`. Bare `{type:'macd'}` etc. hit
"Unknown condition type" and **silently never fire**. Direct types
(`vwap_bands`, `pivot_points`, `support_resistance`, `atr_filter`, `rvol`,
`vwap_bias`, `divergence`, `fx_*`, `forex_*`, ...) stay direct — NOT in the map.

**Two cfg-shape remaps that must also be mirrored (executor reads different keys
than the wizard cfg):**
- **RSI** — wizard cfg uses `condition: lt|gt|custom` (+ optional `value`,
  `period`), but `eval_indicator`'s rsi branch reads `operator`/`value` ONLY
  (ignores `condition`). So `lt/gt` → `{operator, value: value ?? (lt?30:70),
  period: period??14}`; `custom` → `{operator:'lt', value: value??50}`. Without
  this the defaults become `op='gt', val=0` → **always true / over-fires**.
  Structured RSI modes (`rising/falling/midline_cross_*`) are kept as-is but the
  executor still ignores them (pre-existing over-fire edge, mirrored in mobile).
- **vwap_deviation** — wizard uses `vwap_pct`/`vwap_side`; executor's `vwap`
  handler expects `deviation_pct`/`side`. Remap to
  `{name:'vwap', deviation_pct, side}`.

**Why:** discovered when 6 new forex presets (donchian/macd/ichimoku/bb/pivot/
vwap primaries) were added — the web path sent raw `{type:primaryType}` so
macd/bb/ichimoku/donchian never fired (also broke pre-existing
fx_trend_continuation=ema, fx_session_scalp=rsi). `breakout` is neither a direct
executor type nor in the indicator map → fires on NO platform (use `donchian`
upper_break/lower_break instead).

**How to apply:** when adding/changing any non-crypto wizard preset, confirm the
primary + each confirm compile to an executor-supported shape. Keep the web
`WZ_INDICATOR_NAME_MAP` + `_wrapCond` in lockstep with mobile
`INDICATOR_NAME_MAP` + `packCondition`. There is NO server-side condition
normalization at save time — whatever the client emits is what the executor sees.
