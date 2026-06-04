---
name: TP1 partial-profit runner (paper == live)
description: How 2-target strategies take partial profit at TP1, move stop to breakeven, and run the remainder to TP2 — identically in paper and live cTrader forex.
---

# TP1 partial-profit runner

Two-target (TP1 + TP2) strategies bank a fraction at TP1, ratchet the stop to
entry (breakeven), and let the remainder run to TP2. Scoped to **forex/cTrader**
only (crypto/index unchanged → preserves their test==live fidelity). Gold pip=$0.10.

## State lives in execution `notes` (no schema change)
- `partial_close_pct=<n>` — stashed at fire. Default **50** when forex + `tp2_price`
  set + not explicitly configured. Its presence = "this is a partial-runner".
- `vol=<units>` — the broker fill volume, stashed after live placement, so the live
  manager knows how much to split.
- `partial_close_done` — the partial actually executed → **blend P&L** on close.
- `partial_skip` — confirmed un-splittable; full position runs to TP2, **no blend**.
- `be_moved` — stop ratcheted to entry (only set when the broker amend succeeded).

## Broker TP is parked at TP2 (critical)
On a partial-runner live order the broker take-profit is placed at **TP2**, not TP1
(so the broker doesn't fully close at TP1; we close the partial ourselves). Because
`ProtoOAAmendPositionSLTPReq` replaces BOTH legs (see
`ctrader-amend-replaces-both-legs.md`), every SL amend must re-send `broker_tp`
(=TP2 in partial mode), never TP1, or it clobbers the parked TP2.

## P&L blend (both paper and live close paths)
Realised P&L is linear in exit price, so when `partial_close_done`:
`effective_exit = frac*TP1 + (1-frac)*final_exit`, applied to BOTH pnl_pct and
pips_pnl. The **stored** `exit_price` stays the real final exit for the card.
Live close reads TP1 from `ex.tp_price`; reconcile passes the final exit (TP2 fill).
Reconcile worklist therefore classifies a partial-runner close against **TP2**
(`ex.tp2_price`) — else a TP2 fill is mis-measured against TP1 and underreports.

## `close_partial_position_for_user` 3-outcome contract (don't conflate failures)
Returns: `>0` closed units (success), `-1` CONFIRMED un-splittable (grid/min-volume
bounds — both slices must stay ≥ minVolume; details-fetch-OK), `0` TRANSIENT
(proto down / no creds / detail fetch failed / close rejected).
**Why:** an earlier version returned `0` for everything, so a transient broker
hiccup falsely recorded breakeven and permanently disabled the partial (marked
skip). The caller MUST branch on all three:
- `0` → log + return early, change nothing, leave `partial_active` True → retry next cycle.
- `-1` → mark `partial_skip`, move SL→BE, run full position to TP2.
- `>0` → mark `partial_close_done`, move SL→BE, blend on close.
In the two resolved branches, record `be_moved` + persist `sl_price=entry` ONLY when
the amend succeeded (`amend_ok`) — never claim a BE that didn't reach the broker.

**Known non-blocking gap:** a *deterministic* broker close reject (e.g. volume-invalid
from grid drift) is treated as transient (`0`) → retries each cycle while price ≥ TP1.
Bounded in practice: the position eventually hits SL/TP2 and the reconcile loop closes
it. Parsing the close-error reason to map deterministic rejects → `-1` is the only
fix if strict eventual resolution is ever needed.

## Latency: OPEN notifications are fire-and-forget
Both paper and live OPEN Telegram cards send via `asyncio.create_task(_tg_send(...))`
(matching the close path) so a slow/retrying Telegram POST never delays the firing
cycle. Deeper feed-reconnect / cycle-stall latency is a separate reliability area.

## Deploy
Engine + paper + AI/portal ship via redeploy. The live cTrader partial path
(executor) is **prod-only/disabled in dev → REPUBLISH** for live to use it.
