---
name: Signal quality/strength thresholds must be volatility-relative
description: When adding a "strength"/"significance" filter to any signal type (order_block, etc.), make thresholds relative to recent ATR/avg-body/avg-volume, never absolute %, or it breaks across crypto vs forex.
---

# Signal quality / strength thresholds

When a signal type gains a quality gate (e.g. order_block `strength: any|strong|
institutional`, "big block" filters, displacement requirements), express the
thresholds as **multiples of recent rolling averages**, not absolute percentages
or pip values:
- candle body size ÷ average body (last ~20 closed bars)
- displacement/impulse move ÷ ATR (avg true range proxy)
- candle volume ÷ average volume

**Why:** a raw `min_impulse_pct = 1.5%` is a huge 150-pip move on EURUSD 15m but
trivial on a volatile crypto alt. Absolute thresholds silently never-fire on
forex and over-fire on crypto. Volatility-relative multiples behave the same on
both asset classes. This platform is forex-heavy, so absolute-% gates are a
silent-no-fire trap.

**Volume gate caveat:** some feeds (cTrader/FMP forex trendbars) report 0/empty
volume. A volume threshold MUST no-op when `avg_vol == 0`, or every forex signal
gets silently rejected.

**Backward-compat rule:** default the new strength key to the *permissive* value
(`any` → all multipliers 0 → gates disabled) in the EXECUTOR so existing saved
strategies (which lack the key) keep their old behavior. Set the stronger
default (`strong`) only in the wizard/AI layers so *new* strategies get the
better filtering. Quality gates are additive — keep the original detection logic
intact and only apply the extra gates when their multiplier > 0.

**How to apply:** exclude the forming bar `[-1]` from the reference-average
windows; for an "unmitigated/fresh" flag, check the zone wasn't already tapped
between formation and the prior bar (not including the current touch). Web
wizards store boolean chips as the strings `'true'`/`'false'` — parse booleans
robustly in the executor (`str(v).lower() in ('true','1','yes','on')`), since
`bool('false')` is `True` in Python.
