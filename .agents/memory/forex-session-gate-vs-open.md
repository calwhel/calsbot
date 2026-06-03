---
name: Forex session timing gate vs open/breakout
description: "during the whole session" must map to forex_session in_session, NOT killzone/break (a recurring AI-builder confusion)
---

When a user says "only trade/fire DURING London and NY session" they mean a
TIME GATE spanning the ENTIRE session window — map it to
`{"type":"forex_session","condition":"in_session","sessions":["london","ny"]}`.

Do NOT map "during the session" to `fx_killzone` (only the first ~2h opening
window) or `forex_session_break` (a range BREAKOUT signal) — both fire only at/
around the open, so the user reported "it only does the opening of the session,
not all of it." Reserve those two for explicit "killzone"/"open"/"breakout"/
"range break" wording.

**Why:** the AI builder schema (strategy_builder.py CONDITION_SCHEMA + CONDITION
SELECTION) and the portal chat-builder SIGNAL RECOGNITION originally only
documented killzone + session_break for "London/NY session"; the plain
`forex_session` in_session gate (which the executor already dispatches) was never
offered, AND a stray "London → forex_session_break" line actively pulled the
model the wrong way. Both prompts must stay consistent and the breakout line must
NOT shadow the timing-gate line.

**How to apply:** `eval_forex_session` (strategy_ta.py) supports a plural
`sessions` list with OR semantics (fires when in ANY listed session) plus single
`session`; session ids are `london|ny|asian|sydney|overlap` and it normalizes
`new_york→ny`, `tokyo→asian`. `in_session` = whole window; `session_open`/
`session_close` = first/last N minutes; `overlap` = London/NY overlap. This is
the canonical "trade during X" primitive — keep all 3 prompt layers pointing at
it for timing language.
