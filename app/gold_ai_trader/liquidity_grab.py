"""Liquidity-grab detector with overlap skip + MSS confirmation."""
from __future__ import annotations

import logging
import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from app.gold_ai_trader.call_gates import atr_from_klines
from app.gold_ai_trader.context_levels import compute_asian_range
from app.services.forex_engine import previous_day_high_low
from app.services.tradfi_prices import get_klines

logger = logging.getLogger(__name__)


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _equal_cluster(levels: List[float], tol: float, *, min_touches: int = 2) -> Optional[float]:
    if not levels:
        return None
    for i, lv in enumerate(levels):
        touches = [levels[j] for j in range(len(levels)) if abs(levels[j] - lv) <= tol]
        if len(touches) >= min_touches:
            return sum(touches) / float(len(touches))
    return None


def _overlap_matches(
    *,
    sweep_level: float,
    tol: float,
    pdh: Optional[float],
    pdl: Optional[float],
    asian_hi: Optional[float],
    asian_lo: Optional[float],
    eqh: Optional[float],
    eql: Optional[float],
) -> List[str]:
    refs = {
        "PDH": pdh,
        "PDL": pdl,
        "ASIAN_HIGH": asian_hi,
        "ASIAN_LOW": asian_lo,
        "EQH": eqh,
        "EQL": eql,
    }
    hits = []
    for name, value in refs.items():
        if value is None:
            continue
        if abs(float(sweep_level) - float(value)) <= tol:
            hits.append(f"{name}@{value:.2f}")
    return hits


async def eval_liquidity_grab(
    cond: Dict,
    symbol: str,
    current_price: float,
    _http_client,
    cache: Dict,
) -> Tuple[bool, str, Dict]:
    """
    Liquidity grab = sweep + reclaim + displacement/MSS.
    Includes overlap-skip when sweep level is effectively PDH/PDL/Asian/EQH/EQL.
    """
    direction = (cond.get("direction") or "bullish").strip().lower()
    tf = (cond.get("timeframe") or "5m").strip().lower()
    is_long = direction == "bullish"

    lookback = max(90, _env_int("GOLD_AI_LIQ_GRAB_LOOKBACK_BARS", 160))
    swing_lookback = max(8, _env_int("GOLD_AI_LIQ_GRAB_SWING_LOOKBACK", 18))
    max_scan = max(4, _env_int("GOLD_AI_LIQ_GRAB_MAX_SCAN_BARS", 20))
    mss_pivot_lookback = max(3, _env_int("GOLD_AI_LIQ_GRAB_MSS_PIVOT_LOOKBACK", 6))
    mss_window = max(1, _env_int("GOLD_AI_LIQ_GRAB_MSS_WINDOW_BARS", 6))
    mss_body_min_atr = max(0.05, _env_float("GOLD_AI_LIQ_GRAB_MSS_MIN_BODY_ATR", 0.40))
    mss_break_buffer_atr = max(0.01, _env_float("GOLD_AI_LIQ_GRAB_MSS_BREAK_BUFFER_ATR", 0.08))
    overlap_tol_atr = max(0.01, _env_float("GOLD_AI_LIQ_GRAB_OVERLAP_TOL_ATR", 0.20))
    require_retest = _env_bool("GOLD_AI_LIQ_GRAB_REQUIRE_RETEST", True)
    retest_max_bars = max(1, _env_int("GOLD_AI_LIQ_GRAB_RETEST_MAX_BARS", 4))
    retest_tol_atr = max(0.01, _env_float("GOLD_AI_LIQ_GRAB_RETEST_TOL_ATR", 0.12))

    asset_class = (cache or {}).get("__asset_class__", "forex")
    k5 = await get_klines(symbol, asset_class, "5m", 220) or []
    k1h = await get_klines(symbol, asset_class, "1h", 40) or []
    k = await get_klines(symbol, asset_class, tf, lookback) or []
    if not k or len(k) < 50:
        return False, "liquidity_grab:insufficient_data", {}
    rows = k[:-1]
    if len(rows) < 40:
        return False, "liquidity_grab:insufficient_closed_bars", {}

    opens = [float(r[1]) for r in rows]
    highs = [float(r[2]) for r in rows]
    lows = [float(r[3]) for r in rows]
    closes = [float(r[4]) for r in rows]
    atr = atr_from_klines(k[-80:])
    if atr <= 0:
        return False, "liquidity_grab:atr_unavailable", {}

    overlap_tol = overlap_tol_atr * atr
    mss_break_buffer = mss_break_buffer_atr * atr
    retest_tol = retest_tol_atr * atr

    pdh = None
    pdl = None
    pd_levels = previous_day_high_low(rows)
    if pd_levels is not None:
        pdh, pdl = float(pd_levels[0]), float(pd_levels[1])

    asian_hi = None
    asian_lo = None
    ahi, alo = compute_asian_range(datetime.utcnow(), k5, k1h)
    if ahi is not None and alo is not None:
        asian_hi = float(ahi)
        asian_lo = float(alo)

    cluster_slice = max(20, len(rows) - 120)
    eqh = _equal_cluster(highs[cluster_slice:], overlap_tol)
    eql = _equal_cluster(lows[cluster_slice:], overlap_tol)

    n = len(rows)
    start = max(swing_lookback + 2, n - max_scan)
    for sweep_i in range(n - 3, start - 1, -1):
        left_lows = lows[max(0, sweep_i - swing_lookback) : sweep_i]
        left_highs = highs[max(0, sweep_i - swing_lookback) : sweep_i]
        if not left_lows or not left_highs:
            continue

        sweep_level = None
        if is_long:
            local_low = min(left_lows)
            if lows[sweep_i] < local_low and closes[sweep_i] > local_low:
                sweep_level = local_low
        else:
            local_high = max(left_highs)
            if highs[sweep_i] > local_high and closes[sweep_i] < local_high:
                sweep_level = local_high
        if sweep_level is None:
            continue

        overlap_hits = _overlap_matches(
            sweep_level=sweep_level,
            tol=overlap_tol,
            pdh=pdh,
            pdl=pdl,
            asian_hi=asian_hi,
            asian_lo=asian_lo,
            eqh=eqh,
            eql=eql,
        )
        if overlap_hits:
            logger.info(
                "[gold-ai-liquidity-grab] overlap_skip dir=%s sweep=%.2f hits=%s tol=%.2f",
                direction,
                sweep_level,
                ",".join(overlap_hits),
                overlap_tol,
            )
            return (
                False,
                f"Liquidity grab {direction}: overlap-skip {'|'.join(overlap_hits)}",
                {"liq_grab_overlap_skip": True, "liq_grab_overlap_hits": overlap_hits},
            )

        pivot_start = max(1, sweep_i - mss_pivot_lookback)
        if is_long:
            mss_level = max(highs[pivot_start:sweep_i])
        else:
            mss_level = min(lows[pivot_start:sweep_i])
        if mss_level <= 0:
            continue

        mss_idx = None
        mss_body_atr = 0.0
        for j in range(sweep_i + 1, min(sweep_i + 1 + mss_window, n)):
            body = abs(closes[j] - opens[j]) / max(atr, 1e-9)
            if body < mss_body_min_atr:
                continue
            if is_long:
                if closes[j] > (mss_level + mss_break_buffer):
                    mss_idx = j
                    mss_body_atr = body
                    break
            else:
                if closes[j] < (mss_level - mss_break_buffer):
                    mss_idx = j
                    mss_body_atr = body
                    break
        if mss_idx is None:
            continue

        used_retest = False
        if require_retest:
            retest_ok = False
            max_i = min(n - 1, mss_idx + retest_max_bars)
            for ri in range(mss_idx + 1, max_i + 1):
                if is_long:
                    if lows[ri] <= (mss_level + retest_tol) and closes[ri] >= mss_level:
                        retest_ok = True
                        break
                else:
                    if highs[ri] >= (mss_level - retest_tol) and closes[ri] <= mss_level:
                        retest_ok = True
                        break
            if not retest_ok:
                continue
            used_retest = True

        spot = float(current_price or closes[-1])
        meta = {
            "liq_grab_sweep_level": round(sweep_level, 2),
            "liq_grab_mss_level": round(mss_level, 2),
            "liq_grab_mss_body_atr": round(mss_body_atr, 2),
            "liq_grab_used_retest": bool(used_retest),
            "liq_grab_overlap_refs": {
                "pdh": round(pdh, 2) if pdh is not None else None,
                "pdl": round(pdl, 2) if pdl is not None else None,
                "asian_hi": round(asian_hi, 2) if asian_hi is not None else None,
                "asian_lo": round(asian_lo, 2) if asian_lo is not None else None,
                "eqh": round(eqh, 2) if eqh is not None else None,
                "eql": round(eql, 2) if eql is not None else None,
            },
        }
        logger.info(
            "[gold-ai-liquidity-grab] detected dir=%s sweep=%.2f mss=%.2f mss_body_atr=%.2f "
            "require_retest=%s used_retest=%s overlap_refs={PDH:%s PDL:%s AH:%s AL:%s EQH:%s EQL:%s} spot=%.2f",
            direction,
            sweep_level,
            mss_level,
            mss_body_atr,
            require_retest,
            used_retest,
            f"{pdh:.2f}" if pdh is not None else "n/a",
            f"{pdl:.2f}" if pdl is not None else "n/a",
            f"{asian_hi:.2f}" if asian_hi is not None else "n/a",
            f"{asian_lo:.2f}" if asian_lo is not None else "n/a",
            f"{eqh:.2f}" if eqh is not None else "n/a",
            f"{eql:.2f}" if eql is not None else "n/a",
            spot,
        )
        return (
            True,
            (
                f"Liquidity grab {direction}: sweep @ {sweep_level:.2f} reclaim @ {sweep_level:.2f} "
                f"MSS @ {mss_level:.2f} {'with retest' if used_retest else 'direct'} -> FIRED"
            ),
            meta,
        )

    return False, f"Liquidity grab {direction}: no sweep+MSS sequence", {}
