---
name: cTrader amend SL/TP replaces BOTH legs
description: ProtoOAAmendPositionSLTPReq overwrites stopLoss AND takeProfit together — any breakeven/trailing SL amend must re-send the existing TP or the broker clears it.
---

# cTrader amend clears the omitted leg

`ProtoOAAmendPositionSLTPReq` (modify_position_sltp) **replaces both** the
stop-loss and take-profit on the position. If you send only `stopLoss` and omit
`takeProfit`, the broker **clears the existing TP** (and vice-versa). It is NOT a
partial update.

**Why:** live forex auto-breakeven/trailing amended only the SL
(`modify_position_sltp_for_user(..., stop_loss_price=amend_sl)` with no TP). The
first breakeven move silently wiped each position's broker take-profit. Symptom:
price hits the target but the position never closes; in the DB EVERY live forex
trade closes as BREAKEVEN (0 pips) and not one ever records a live TP win, even
though the same strategies' paper trades hit TP fine.

**How to apply:** on EVERY SL amend (breakeven or trailing), also pass the
position's stored `tp_price` so the broker keeps it. Carry `tp_price` through the
worklist alongside `sl_price`. Conversely, if you ever amend only the TP, re-send
the current SL. Beware the inverse trap: amending SL when the DB has `tp_price =
None` will clear a broker TP that does exist — forex execs always store a TP, but
guard/log if a None TP ever reaches the amend path.

**Self-heal caveat:** a position already at breakeven (`be_moved` in notes, no
trailing) triggers no further amend, so the fix does NOT retroactively restore a
TP that was already wiped — those open positions need a one-time manual amend
(send current SL + stored TP).
