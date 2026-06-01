---
name: cTrader index order gap
description: Index (CFD) live orders reference an undefined place_order_units and never actually fire — they fall back to paper.
---

# cTrader index order gap

`place_ctrader_order_for_user` (index branch) calls `place_order_units(...)`,
but **no such function is defined** anywhere in `ctrader_client.py`. An index
live order therefore raises `NameError`, which the executor's order try/except
catches and converts to a paper trade. Net effect: index live orders never
reach the broker; they are silently tracked as paper.

**Why not auto-fixed:** index CFD volume scaling (contracts ↔ wire units) is
broker-specific and differs from forex lot scaling (live position management
deliberately skips index for the same reason). Implementing it speculatively,
without testing against the live broker, risks mis-sized real-money index
trades — strictly worse than the current graceful paper fallback.

**How to apply:** if asked to make index live orders work, define
`place_order_units` modeled on `place_order` (same `_send_recv_any` +
ProtoOAOrderErrorEvent 2132 + terminal-execution-type rejection handling), but
first confirm the correct cTrader volume convention for index CFDs against the
live broker before enabling real orders.
