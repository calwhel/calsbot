"""Asian range sweep → displacement → reclaim (gold scanner)."""
from __future__ import annotations

from datetime import datetime
from typing import Dict, Tuple

from app.gold_ai_trader.context_levels import compute_asian_range


async def eval_asian_range_sweep(
    cond: Dict,
    symbol: str,
    current_price: float,
    http_client,
    cache: Dict,
) -> Tuple[bool, str]:
    """
    Sweep Asian session high/low then reclaim with displacement.
    cfg.direction: bullish (sweep Asian low) | bearish (sweep Asian high)
    """
    from app.services.strategy_ta import (
        _get_klines,
        _opens,
        _highs,
        _lows,
        _closes,
        _atr,
        entry_zone_allows_price,
        entry_max_dist_from_zone_atr,
    )

    direction = (cond.get("direction") or "bullish").lower()
    tf = cond.get("timeframe", "5m")
    try:
        disp_window = max(1, min(int(cond.get("disp_window") or 5), 12))
    except (TypeError, ValueError):
        disp_window = 5
    try:
        min_body_ratio = max(1.0, float(cond.get("min_body_ratio") or 2.0))
    except (TypeError, ValueError):
        min_body_ratio = 2.0

    k5 = await _get_klines(symbol, "5m", 80, http_client, cache)
    k1h = await _get_klines(symbol, "1h", 30, http_client, cache)
    klines = await _get_klines(symbol, tf, 80, http_client, cache)
    if not klines or len(klines) < 25:
        return False, "Asian sweep: insufficient data"

    now = datetime.utcnow()
    asian_hi, asian_lo = compute_asian_range(now, k5 or [], k1h or [])
    if asian_hi is None or asian_lo is None:
        return False, "Asian sweep: no Asian range"

    closed = klines[:-1]
    n = len(closed)
    h = _highs(closed)
    l = _lows(closed)
    o = _opens(closed)
    c = _closes(closed)
    bodies = [abs(c[i] - o[i]) for i in range(n)]
    avg_body = sum(bodies[:-1]) / max(1, len(bodies) - 1) if bodies else 0.0
    atr_val = _atr(closed)
    price = current_price if current_price > 0 else c[-1]

    sweep_level = asian_lo if direction == "bullish" else asian_hi
    start = max(5, n - 25)
    for sweep_i in range(n - 2, start - 1, -1):
        swept = False
        if direction == "bullish":
            swept = l[sweep_i] < asian_lo and c[sweep_i] > asian_lo
        else:
            swept = h[sweep_i] > asian_hi and c[sweep_i] < asian_hi
        if not swept:
            continue

        disp_ok = False
        for d in range(sweep_i + 1, min(sweep_i + 1 + disp_window, n)):
            ratio = bodies[d] / avg_body if avg_body else 0.0
            if ratio < min_body_ratio:
                continue
            if direction == "bullish" and c[d] > o[d]:
                disp_ok = True
                break
            if direction == "bearish" and c[d] < o[d]:
                disp_ok = True
                break
        if not disp_ok:
            continue

        tol = (entry_max_dist_from_zone_atr() * atr_val) if atr_val else (asian_hi - asian_lo) * 0.05
        zone_bot = sweep_level - tol
        zone_top = sweep_level + tol
        ok_zone, zone_msg = entry_zone_allows_price(
            price, zone_bot, zone_top, direction, atr_val,
        )
        if ok_zone:
            return True, (
                f"Asian sweep {direction}: level={sweep_level:.2f} "
                f"(range {asian_lo:.2f}–{asian_hi:.2f}) reclaim price={price:.2f} → FIRED"
            )
        return False, (
            f"Asian sweep {direction}: sweep→disp ok but {zone_msg} @ {sweep_level:.2f}"
        )

    return False, f"Asian sweep {direction}: no sweep→displacement→reclaim"
