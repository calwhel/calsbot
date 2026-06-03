---
name: Forex live SL management (breakeven/trailing)
description: How live forex breakeven/trailing stop amendments reach the cTrader broker, the price source for low-latency reactions, and the single-owner rule.
---

# Forex live SL management

Live forex breakeven + trailing stops only take effect on real money when the
executor amends the broker's SL after entry (the broker keeps the entry SL/TP
forever otherwise). This management runs in ONE place.

## Price source for low-latency reactions
- A cTrader **streaming spot feed** already exists (`app/services/ctrader_price_feed.py`),
  started in the executor worker only (one worker, co-located with the executor
  advisory lock). It maintains a per-tick in-memory `_spot_cache` and exposes
  `get_price(symbol)` (mid, short TTL → None when stale), `get_bid_ask`, `is_live`.
- **Any time you need sub-second forex/gold prices, read this feed first**, then
  fall back to FMP (`tradfi_prices.get_price`, which only refreshes ~5s). The 5s
  FMP cadence is why fast instruments like gold used to lag on stop moves.
- **Why:** gold can move far inside a 5s window; reacting only on the FMP/scan
  cadence meant breakevens triggered seconds late.

## Single-owner rule (no double-amend)
- Live SL management runs on its OWN fast loop (`run_forex_live_manager_fast`,
  ~1s), NOT inside the 5s `run_forex_executor` scan. When you add/keep a fast
  loop, REMOVE the scan-cycle management calls — two loops amending the same
  position races the broker and double-fires.
- **How to apply:** the fast loop rebuilds its DB worklist only every few seconds
  (`_FX_WORKLIST_TTL`) and reads prices from the spot cache each tick, so a 1s
  cadence is DB-light (writes only on an actual amend). It must mutate the cached
  work item in place (`sl_price`/`be_moved`) after a successful amend+persist, or
  a re-used worklist will refire the same move before the next rebuild.
- The fast loop must be registered under the SAME executor advisory lock worker
  (in `_start_executor_tasks`) so a multi-worker gunicorn deploy single-fires.

## Breakeven notifications (push + Telegram), sync-safe
- `_notify_breakeven_alert` must be callable from BOTH the sync paper monitor and
  the async live manager: push via the already-threaded `notify_breakeven_bg`;
  Telegram scheduled on the running loop (`get_running_loop().create_task`) when
  one exists, else a daemon-thread `asyncio.run` fallback.
- Normalize `telegram_id` like `_telegram_int_id` (skip `WEB-…`/non-numeric) or
  web users generate noisy doomed sendMessage calls.
- Fire ONCE per activation: paper path on first BE activation; live path only
  when `be_moved` is newly persisted to notes.

## Broker close-reconciliation (SL/TP fill detection) — REQUIRED, separate from amending
- Amending the broker SL is NOT enough: when the broker actually FILLS an SL/TP
  and removes the position, nothing in the amend path notices. Without a
  reconcile, a live forex stop-out produced NO push/Telegram alert and the
  execution sat OPEN until the 48h stale-expiry swept it silently (pnl=0).
- Crypto already had this (Bitunix `_sweep_live` reconcile → close+notify); forex
  needed its own. Pattern: poll cTrader open positions per user via
  `ProtoOAReconcileReq` (`get_open_position_ids_for_user` → set of positionIds,
  **returns None on any failure** so an outage is never read as "all closed"),
  compare to each tracked `pos=<id>`; after 2 consecutive misses classify
  WIN/LOSS/BREAKEVEN by price-vs-TP/SL distance and call the close+notify helper.
- Run the reconcile THROTTLED (~15s) inside the same single-owner fast loop, not
  every tick (one broker round-trip per account per interval).
- **False-close trap:** `pos=<id>` is matched against the user's CURRENT prefs
  account. If they relink/switch cTrader accounts, an old still-open position is
  absent from the new account → would false-close. Fix WITHOUT a schema column:
  stamp `acct=<ctid>` into notes at entry (order placement returns `account_id`)
  and only reconcile when the stored acct == current `UserPreference.ctrader_account_id`;
  skip on mismatch/unknown. Legacy execs (no `acct=`) fall back to best-effort.
- **Why:** see the gold-SL-no-notification incident — the broker enforced the SL
  but the app never learned the trade closed.
