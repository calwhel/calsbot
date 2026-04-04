"""
Liquidity Sweep Entry Filter

Instead of entering immediately on a signal, waits for a stop-hunt wick:
  LONG:  price dips 0.8% below entry (sweeping sell stops), then recovers.
         SL is tightened to just below the wick low.
  SHORT: price spikes 0.8% above entry (sweeping buy stops), then falls back.
         SL is tightened to just above the wick high.

If no sweep occurs within SWEEP_TIMEOUT_SECONDS the trade fires at the current
price — UNLESS price has run more than MAX_PUMP_ABORT_PCT from the original
entry, in which case the callback receives ABORTED_ENTRY (-1.0) and the trade
is skipped with a user notification.

Bug fixed: price-fetch failures no longer silently swallow the timeout check.
"""

import asyncio
import logging
import time
from datetime import datetime, timedelta
from typing import Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

SWEEP_TIMEOUT_SECONDS = 600    # 10-minute max wait
SWEEP_ZONE_PCT        = 0.008  # 0.8% from entry triggers the sweep zone
RECOVERY_PCT          = 0.002  # 0.2% from entry = recovery confirmed
SL_BUFFER_PCT         = 0.002  # 0.2% beyond wick = tight SL placement
CHECK_INTERVAL        = 10     # seconds between price polls
MAX_PUMP_ABORT_PCT    = 0.03   # 3% — if price runs this far from entry at timeout, abort

# Sentinel value passed to the callback to signal "price ran too far, skip trade"
ABORTED_ENTRY = -1.0

_enabled: bool = True                  # global on/off switch
_pending: Dict[str, dict] = {}         # key: "{symbol}_{direction}"
_lock = asyncio.Lock()
_total_sweeps_hit: int = 0             # lifetime confirmed sweeps
_total_timeouts:   int = 0             # lifetime timeout fires
_total_aborts:     int = 0             # lifetime pump-abort fires


def is_sweep_enabled() -> bool:
    return _enabled


def set_sweep_enabled(enabled: bool) -> None:
    global _enabled
    _enabled = enabled
    logger.info(f"🎯 Sweep entry {'ENABLED' if enabled else 'DISABLED'}")


def get_sweep_status() -> dict:
    """Return a status snapshot for the social menu."""
    return {
        "enabled":     _enabled,
        "pending":     len(_pending),
        "pending_list": [
            f"{v['symbol']} {v['direction']} (zone={'hit' if v['swept'] else 'waiting'})"
            for v in _pending.values()
        ],
        "sweeps_hit":  _total_sweeps_hit,
        "timeouts":    _total_timeouts,
        "aborts":      _total_aborts,
    }


async def _get_price(symbol: str) -> Optional[float]:
    """Live price from the Binance WebSocket cache, with REST fallback."""
    try:
        from app.services.binance_ws import get_ticker
        ticker = get_ticker(symbol)
        if ticker:
            raw = ticker.get("c") or ticker.get("lastPrice")
            if raw:
                return float(raw)
    except Exception:
        pass

    try:
        import aiohttp
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"https://fapi.binance.com/fapi/v1/ticker/price?symbol={symbol}",
                timeout=aiohttp.ClientTimeout(total=4),
            ) as resp:
                data = await resp.json()
                return float(data.get("price", 0)) or None
    except Exception:
        return None


def is_sweep_pending(symbol: str, direction: str) -> bool:
    return f"{symbol}_{direction}" in _pending


async def queue_sweep_entry(
    symbol: str,
    direction: str,
    entry_price: float,
    original_sl: float,
    fire_callback: Callable,
    immediate: bool = False,
) -> None:
    """
    Queue a signal for sweep-entry watching.

    fire_callback(entry_price: float, stop_loss: float, sweep_hit: bool)
    is called when a sweep confirms or the timeout expires.

    Special value ABORTED_ENTRY (-1.0) is passed as entry_price when the
    price has pumped/dumped too far from the original entry at timeout —
    callers should skip execution and notify the user.

    If sweep entry is globally disabled, or immediate=True, the callback fires
    immediately with the original values so the trade executes right away.
    """
    if not _enabled or immediate:
        logger.info(f"🎯 {'Immediate' if immediate else 'Sweep disabled'} — firing {symbol} {direction} immediately")
        await fire_callback(entry_price, original_sl, False)
        return

    key = f"{symbol}_{direction}"
    async with _lock:
        if key in _pending:
            logger.info(f"🎯 Sweep watch already active for {key} — skipping duplicate")
            return

        sweep_zone = (
            entry_price * (1 - SWEEP_ZONE_PCT)
            if direction == "LONG"
            else entry_price * (1 + SWEEP_ZONE_PCT)
        )
        recovery = (
            entry_price * (1 - RECOVERY_PCT)
            if direction == "LONG"
            else entry_price * (1 + RECOVERY_PCT)
        )

        _pending[key] = {
            "symbol":        symbol,
            "direction":     direction,
            "entry_price":   entry_price,
            "original_sl":   original_sl,
            "sweep_zone":    sweep_zone,
            "recovery":      recovery,
            "wick_extreme":  None,
            "swept":         False,
            "timeout_at":    datetime.utcnow() + timedelta(seconds=SWEEP_TIMEOUT_SECONDS),
            "fire_callback": fire_callback,
            "queued_at":     time.time(),
            "last_price":    entry_price,  # track last known price for timeout abort check
        }

    arrow = "↓" if direction == "LONG" else "↑"
    logger.info(
        f"🎯 Sweep watch queued — {symbol} {direction} | "
        f"entry={entry_price:.6f} | sweep_zone={sweep_zone:.6f} {arrow} | timeout=10min"
    )


async def run_sweep_watcher_loop() -> None:
    """Background loop — runs forever, polling prices for all pending sweep watches."""
    logger.info("🎯 Sweep watcher loop started")
    while True:
        await asyncio.sleep(CHECK_INTERVAL)
        if not _pending:
            continue

        now = datetime.utcnow()
        to_fire: List[Tuple[Callable, float, float, bool]] = []

        async with _lock:
            for key in list(_pending.keys()):
                w = _pending.get(key)
                if not w:
                    continue

                price = await _get_price(w["symbol"])

                # ── Bug fix: price fetch failure no longer blocks the timeout check ──
                if price:
                    w["last_price"] = price  # keep last known price updated

                direction = w["direction"]
                fired = False

                if price:
                    if direction == "LONG":
                        if w["wick_extreme"] is None or price < w["wick_extreme"]:
                            w["wick_extreme"] = price
                        if price <= w["sweep_zone"] and not w["swept"]:
                            logger.info(
                                f"🎯 Sweep zone hit — {w['symbol']} LONG | "
                                f"price={price:.6f} ≤ zone={w['sweep_zone']:.6f}"
                            )
                            w["swept"] = True
                        if w["swept"] and price >= w["recovery"]:
                            wick     = w["wick_extreme"]
                            tight_sl = wick * (1 - SL_BUFFER_PCT)
                            use_sl   = tight_sl if tight_sl > w["original_sl"] else w["original_sl"]
                            logger.info(
                                f"✅ SWEEP ENTRY confirmed — {w['symbol']} LONG | "
                                f"wick={wick:.6f} | SL={use_sl:.6f} | entry={price:.6f}"
                            )
                            to_fire.append((w["fire_callback"], price, use_sl, True))
                            del _pending[key]
                            fired = True

                    else:  # SHORT
                        if w["wick_extreme"] is None or price > w["wick_extreme"]:
                            w["wick_extreme"] = price
                        if price >= w["sweep_zone"] and not w["swept"]:
                            logger.info(
                                f"🎯 Sweep zone hit — {w['symbol']} SHORT | "
                                f"price={price:.6f} ≥ zone={w['sweep_zone']:.6f}"
                            )
                            w["swept"] = True
                        if w["swept"] and price <= w["recovery"]:
                            wick     = w["wick_extreme"]
                            tight_sl = wick * (1 + SL_BUFFER_PCT)
                            use_sl   = tight_sl if tight_sl < w["original_sl"] else w["original_sl"]
                            logger.info(
                                f"✅ SWEEP ENTRY confirmed — {w['symbol']} SHORT | "
                                f"wick={wick:.6f} | SL={use_sl:.6f} | entry={price:.6f}"
                            )
                            to_fire.append((w["fire_callback"], price, use_sl, True))
                            del _pending[key]
                            fired = True

                # ── Timeout check — runs even if price fetch failed ──
                if not fired and now >= w["timeout_at"]:
                    elapsed   = (time.time() - w["queued_at"]) / 60
                    last_px   = w["last_price"]
                    orig_entry = w["entry_price"]

                    # Check if price has pumped/dumped too far to enter safely
                    if direction == "LONG":
                        drift_pct = (last_px - orig_entry) / orig_entry
                    else:
                        drift_pct = (orig_entry - last_px) / orig_entry

                    if drift_pct > MAX_PUMP_ABORT_PCT:
                        logger.info(
                            f"🚫 Sweep ABORTED — {w['symbol']} {direction} | "
                            f"price ran {drift_pct*100:.1f}% from entry (max {MAX_PUMP_ABORT_PCT*100:.0f}%)"
                        )
                        # Pass sentinel so callback can notify user and skip trade
                        to_fire.append(
                            (w["fire_callback"], ABORTED_ENTRY, w["original_sl"], False)
                        )
                    else:
                        logger.info(
                            f"⏱️ Sweep timeout ({elapsed:.1f}min) — "
                            f"firing {w['symbol']} {direction} at {last_px:.6f}"
                        )
                        to_fire.append(
                            (w["fire_callback"], last_px, w["original_sl"], False)
                        )
                    del _pending[key]

        # Update counters
        global _total_sweeps_hit, _total_timeouts, _total_aborts
        for _, entry_px, _, sweep_hit in to_fire:
            if sweep_hit:
                _total_sweeps_hit += 1
            elif entry_px == ABORTED_ENTRY:
                _total_aborts += 1
            else:
                _total_timeouts += 1

        # Fire callbacks outside the lock
        for cb, entry_px, sl, sweep_hit in to_fire:
            try:
                await cb(entry_px, sl, sweep_hit)
            except Exception as exc:
                logger.error(f"❌ Sweep callback error: {exc}", exc_info=True)
