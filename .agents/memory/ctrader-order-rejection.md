---
name: cTrader order rejection visibility
description: How cTrader signals a rejected order, and why accepting only the execution event silently loses the reason.
---

# cTrader order rejection visibility

When a cTrader (FP Markets) order is rejected, the broker replies with a
**`ProtoOAOrderErrorEvent` (payloadType 2132)**, NOT a `ProtoOAExecutionEvent`
(2126). It can also reply with an ExecutionEvent whose `executionType` is a
terminal failure: `ORDER_REJECTED` / `ORDER_CANCELLED` / `ORDER_EXPIRED`.

**Rule:** any place-order receive in `ctrader_client.py` must accept BOTH
2126 and 2132 (use `_send_recv_any` with the set of expected types), surface
2132's `errorCode`/`description` as the error reason, and reject the terminal
execution types instead of reading their `order` block as a fill.

**Why:** a receive that only accepts 2126 silently drops the 2132 rejection →
15s socket timeout → generic "no order id" message, hiding the real cause
(unknown symbol, bad volume, invalid SL/TP, trading disabled, market closed,
no margin). An ExecutionEvent rejection that still carries an `order` block,
if read as a fill, marks a non-existent live position as live.

**How to apply:** mirror the parse block of `place_order` in any new order
function. `place_ctrader_order_for_user` must return the result dict (with
`order_id=None` + `error`) on rejection — NOT `None` — so the executor can
thread the real reason into the user's Telegram notification. Both executor
callsites guard with `if order_result:` then `if order_id:`, so a dict with
order_id=None is safe (falls through to the paper-fallback path).
