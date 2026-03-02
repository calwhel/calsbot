"""
Liquidity Sweep Entry Filter

Instead of entering immediately on a signal, waits for a stop-hunt wick:
  LONG:  price dips 0.5% below entry (sweeping sell stops), then recovers.
         SL is tightened to just below the wick low.
  SHORT: price spikes 0.5% above entry (sweeping buy stops), then falls back.
         SL is tightened to just above the wick high.

If no sweep occurs within SWEEP_TIMEOUT_SECONDS the original signal fires unchanged,
so no trades are ever missed.
"""

import asyncio
import logging
import time
from datetime import datetime, timedelta
from typing import Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

SWEEP_TIMEOUT_SECONDS = 300    # 5-minute max wait
SWEEP_ZONE_PCT        = 0.005  # 0.5%  from entry triggers the sweep zone
RECOVERY_PCT          = 0.001  # 0.1%  from entry = recovery confirmed
SL_BUFFER_PCT         = 0.002  # 0.2%  beyond wick = tight SL placement
CHECK_INTERVAL        = 10     # seconds between price polls

_enabled: bool = True                  # global on/off switch
_pending: Dict[str, dict] = {}         # key: "{symbol}_{direction}"
_lock = asyncio.Lock()
_total_sweeps_hit: int = 0             # lifetime confirmed sweeps
_total_timeouts:   int = 0             # lifetime timeout fires


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
) -> None:
    """
    Queue a signal for sweep-entry watching.

    fire_callback(entry_price: float, stop_loss: float, sweep_hit: bool)
    is called when a sweep confirms or the timeout expires.

    If sweep entry is globally disabled the callback fires immediately with
    the original values so the trade executes right away.
    """
    if not _enabled:
        logger.info(f"🎯 Sweep entry disabled — firing {symbol} {direction} immediately")
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
        }

    arrow = "↓" if direction == "LONG" else "↑"
    logger.info(
        f"🎯 Sweep watch queued — {symbol} {direction} | "
        f"entry={entry_price:.6f} | sweep_zone={sweep_zone:.6f} {arrow}"
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
                if not price:
                    continue

                direction = w["direction"]
                fired = False

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

                if not fired and now >= w["timeout_at"]:
                    elapsed = (time.time() - w["queued_at"]) / 60
                    logger.info(
                        f"⏱️ Sweep timeout ({elapsed:.1f}min) — "
                        f"firing original {w['symbol']} {direction}"
                    )
                    to_fire.append(
                        (w["fire_callback"], w["entry_price"], w["original_sl"], False)
                    )
                    del _pending[key]

        # Update counters
        global _total_sweeps_hit, _total_timeouts
        for _, _, _, sweep_hit in to_fire:
            if sweep_hit:
                _total_sweeps_hit += 1
            else:
                _total_timeouts += 1

        # Fire callbacks outside the lock
        for cb, entry, sl, sweep_hit in to_fire:
            try:
                await cb(entry, sl, sweep_hit)
            except Exception as exc:
                logger.error(f"❌ Sweep callback error: {exc}", exc_info=True)
