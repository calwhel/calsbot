---
name: Telegram trade-card HTML escaping
description: Why dynamic text in Telegram alert cards must be HTML-escaped, and how the cards are laid out
---

Telegram messages are sent with `parse_mode="HTML"` (`_tg_send` in strategy_executor.py).
Any dynamic text interpolated into a card — `strategy_name`, condition/detail bullets,
`symbol`, `order_id` — MUST be `html.escape()`d before insertion.

**Why:** condition/detail strings legitimately contain `<`, `>`, `&` (e.g. "RSI < 30",
"Price > EMA200", "EMA cross & RSI"). Unescaped, Telegram's HTML parser returns 400 and
the ENTIRE DM is silently dropped (the send is wrapped in try/except → no error surfaces,
the user just never gets the alert). This is a silent-failure trap, not a cosmetic issue.

**How to apply:** when adding/editing any Telegram card or DM, wrap every non-literal
field in `_html.escape(str(x))`. The fixed-format numeric/label table is built inside a
`<pre>` block for monospace column alignment — keep `<pre>` content to numbers + plain
labels only (no `<>&`), since entities are still parsed inside `<pre>`.

Trade cards live in `_fmt_open_card` / `_fmt_close_card` / `_notify_breakeven_alert`
(Telegram) and `app/services/expo_push.py` (phone push; push is plain text → no escaping
needed there). Executor is prod-only/disabled in dev → republish to ship card changes.
