---
name: Risk % sizing key duality
description: Why writing a strategy's "% risk" sizing must set BOTH risk_pct_per_trade and position_size_pct
---

# "% risk" auto-lot sizing must write two keys

When a strategy uses risk-%-based auto-lot sizing (`risk.use_risk_pct=True`), the
risk percentage is read from DIFFERENT keys depending on execution path:

- **Owner path** reads `risk.risk_pct_per_trade` with a fallback to
  `risk.position_size_pct` (so either works).
- **Subscriber / propagated path** reads ONLY `risk.position_size_pct` (no
  fallback to `risk_pct_per_trade`).

**Why:** the two cTrader callsites in `strategy_executor.py` were written at
different times and never unified. Writing only `risk_pct_per_trade` makes the UI
report success while subscribers keep trading the stale/default
`position_size_pct`.

**How to apply:** any code that sets a strategy's risk-% sizing (wizard save,
inline edit endpoints, optimizer apply, etc.) must set `risk_pct_per_trade` AND
`position_size_pct` to the same value, plus `use_risk_pct=True`. The cleanest long
-term fix would be to give the subscriber path the same
`risk_pct_per_trade or position_size_pct` fallback the owner path has — until then,
mirror both keys on write.
