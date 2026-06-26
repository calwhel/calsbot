"""Momentum/trend-continuation detectors for Gold AI scanner."""
from __future__ import annotations

import logging
import os
from typing import Dict, List, Tuple

from app.gold_ai_trader.call_gates import atr_from_klines
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


def _ema_series(values: List[float], period: int) -> List[float]:
    if not values or period <= 1:
        return list(values)
    alpha = 2.0 / (period + 1.0)
    out: List[float] = []
    ema = float(values[0])
    out.append(ema)
    for v in values[1:]:
        ema = alpha * float(v) + (1.0 - alpha) * ema
        out.append(ema)
    return out


def _trend_state(
    closes: List[float],
    *,
    ema_fast: int,
    ema_slow: int,
    slope_bars: int,
) -> Dict[str, float]:
    ema_f = _ema_series(closes, ema_fast)
    ema_s = _ema_series(closes, ema_slow)
    last_i = len(closes) - 1
    prev_i = max(0, last_i - max(1, slope_bars))
    return {
        "price": float(closes[last_i]),
        "ema_fast": float(ema_f[last_i]),
        "ema_slow": float(ema_s[last_i]),
        "ema_fast_slope": float(ema_f[last_i] - ema_f[prev_i]),
        "ema_slow_slope": float(ema_s[last_i] - ema_s[prev_i]),
    }


async def eval_momentum_ema_bounce(
    cond: Dict,
    symbol: str,
    current_price: float,
    _http_client,
    cache: Dict,
) -> Tuple[bool, str, Dict]:
    """EMA stack continuation + pullback rejection."""
    direction = (cond.get("direction") or "bullish").strip().lower()
    trend_tf = (cond.get("trend_timeframe") or "15m").strip().lower()
    trigger_tf = (cond.get("timeframe") or "5m").strip().lower()

    ema_fast = max(5, _env_int("GOLD_AI_MOMENTUM_EMA_FAST", 20))
    ema_slow = max(ema_fast + 2, _env_int("GOLD_AI_MOMENTUM_EMA_SLOW", 50))
    slope_bars = max(1, _env_int("GOLD_AI_MOMENTUM_SLOPE_BARS", 3))
    touch_tol_atr = max(0.01, _env_float("GOLD_AI_MOMENTUM_EMA_TOUCH_TOL_ATR", 0.12))
    bounce_min_body_atr = max(0.05, _env_float("GOLD_AI_MOMENTUM_EMA_BOUNCE_MIN_BODY_ATR", 0.30))
    bounce_min_wick_ratio = max(
        0.2, _env_float("GOLD_AI_MOMENTUM_EMA_BOUNCE_MIN_WICK_TO_BODY", 1.2)
    )
    pullback_lookback = max(2, _env_int("GOLD_AI_MOMENTUM_EMA_PULLBACK_BARS", 4))
    lookback = max(90, _env_int("GOLD_AI_MOMENTUM_LOOKBACK_BARS", 160))

    asset_class = (cache or {}).get("__asset_class__", "forex")
    k_trend = await get_klines(symbol, asset_class, trend_tf, lookback)
    k_trigger = await get_klines(symbol, asset_class, trigger_tf, lookback)
    if not k_trend or not k_trigger:
        return False, "momentum_ema_bounce:missing_klines", {}
    trend_rows = k_trend[:-1]
    rows = k_trigger[:-1]
    if len(trend_rows) < max(ema_slow + slope_bars + 2, 30) or len(rows) < max(
        ema_fast + pullback_lookback + 4, 30
    ):
        return False, "momentum_ema_bounce:insufficient_data", {}

    trend_closes = [float(r[4]) for r in trend_rows]
    trigger_closes = [float(r[4]) for r in rows]
    trigger_opens = [float(r[1]) for r in rows]
    trigger_highs = [float(r[2]) for r in rows]
    trigger_lows = [float(r[3]) for r in rows]
    trigger_ema_fast = _ema_series(trigger_closes, ema_fast)

    trend = _trend_state(
        trend_closes,
        ema_fast=ema_fast,
        ema_slow=ema_slow,
        slope_bars=slope_bars,
    )
    atr = atr_from_klines(k_trigger[-80:])
    if atr <= 0:
        return False, "momentum_ema_bounce:atr_unavailable", {}

    is_long = direction == "bullish"
    if is_long:
        trend_ok = (
            trend["ema_fast"] > trend["ema_slow"]
            and trend["ema_fast_slope"] > 0
            and trend["ema_slow_slope"] >= 0
            and trend["price"] >= trend["ema_slow"]
        )
    else:
        trend_ok = (
            trend["ema_fast"] < trend["ema_slow"]
            and trend["ema_fast_slope"] < 0
            and trend["ema_slow_slope"] <= 0
            and trend["price"] <= trend["ema_slow"]
        )
    if not trend_ok:
        return False, "momentum_ema_bounce:trend_not_aligned", {}

    i = len(rows) - 1
    ema_now = float(trigger_ema_fast[i])
    o = trigger_opens[i]
    h = trigger_highs[i]
    l = trigger_lows[i]
    c = trigger_closes[i]
    body = abs(c - o)
    body_atr = body / max(atr, 1e-9)
    touch_tol = touch_tol_atr * atr

    if is_long:
        touched = l <= (ema_now + touch_tol)
        reclaimed = c > ema_now
        wick = max(0.0, min(o, c) - l)
        rejection = wick / max(body, 1e-9)
        pullback_ok = min(trigger_lows[max(0, i - pullback_lookback) : i + 1]) <= (
            ema_now + touch_tol
        )
    else:
        touched = h >= (ema_now - touch_tol)
        reclaimed = c < ema_now
        wick = max(0.0, h - max(o, c))
        rejection = wick / max(body, 1e-9)
        pullback_ok = max(trigger_highs[max(0, i - pullback_lookback) : i + 1]) >= (
            ema_now - touch_tol
        )

    if not (
        touched
        and reclaimed
        and pullback_ok
        and body_atr >= bounce_min_body_atr
        and rejection >= bounce_min_wick_ratio
    ):
        return False, "momentum_ema_bounce:no_valid_rejection", {}

    entry_anchor = ema_now
    meta = {
        "momentum_mode": "ema_bounce",
        "momentum_ema_fast": ema_fast,
        "momentum_ema_slow": ema_slow,
        "momentum_trend_tf": trend_tf,
        "momentum_trigger_tf": trigger_tf,
        "momentum_anchor_level": round(entry_anchor, 2),
        "momentum_body_atr": round(body_atr, 2),
        "momentum_wick_ratio": round(rejection, 2),
        "momentum_trend_fast": round(trend["ema_fast"], 2),
        "momentum_trend_slow": round(trend["ema_slow"], 2),
        "momentum_fast_slope": round(trend["ema_fast_slope"], 4),
        "momentum_slow_slope": round(trend["ema_slow_slope"], 4),
    }
    logger.info(
        "[gold-ai-momentum] detected type=ema_bounce dir=%s trend_tf=%s trigger_tf=%s "
        "ema_fast=%.2f ema_slow=%.2f slope_fast=%.4f slope_slow=%.4f "
        "touch_tol_atr=%.2f body_atr=%.2f wick_ratio=%.2f anchor=%.2f",
        direction,
        trend_tf,
        trigger_tf,
        trend["ema_fast"],
        trend["ema_slow"],
        trend["ema_fast_slope"],
        trend["ema_slow_slope"],
        touch_tol_atr,
        body_atr,
        rejection,
        entry_anchor,
    )
    return (
        True,
        (
            f"Momentum EMA bounce {direction}: reclaim @ {entry_anchor:.2f} "
            f"body={body_atr:.2f}ATR wick/body={rejection:.2f} -> FIRED"
        ),
        meta,
    )


async def eval_momentum_flag_break(
    cond: Dict,
    symbol: str,
    current_price: float,
    _http_client,
    cache: Dict,
) -> Tuple[bool, str, Dict]:
    """Impulse -> flag -> breakout continuation with optional retest."""
    direction = (cond.get("direction") or "bullish").strip().lower()
    trend_tf = (cond.get("trend_timeframe") or "15m").strip().lower()
    tf = (cond.get("timeframe") or "5m").strip().lower()
    is_long = direction == "bullish"

    lookback = max(100, _env_int("GOLD_AI_MOMENTUM_FLAG_LOOKBACK_BARS", 180))
    impulse_bars = max(3, _env_int("GOLD_AI_MOMENTUM_FLAG_IMPULSE_BARS", 8))
    flag_min_bars = max(2, _env_int("GOLD_AI_MOMENTUM_FLAG_MIN_BARS", 3))
    flag_max_bars = max(flag_min_bars + 1, _env_int("GOLD_AI_MOMENTUM_FLAG_MAX_BARS", 10))
    impulse_min_atr = max(0.3, _env_float("GOLD_AI_MOMENTUM_FLAG_IMPULSE_MIN_ATR", 1.8))
    max_retrace_pct = min(
        0.95, max(0.1, _env_float("GOLD_AI_MOMENTUM_FLAG_MAX_RETRACE_PCT", 0.55))
    )
    break_buffer_atr = max(0.01, _env_float("GOLD_AI_MOMENTUM_FLAG_BREAK_BUFFER_ATR", 0.10))
    break_min_body_atr = max(0.05, _env_float("GOLD_AI_MOMENTUM_FLAG_BREAK_MIN_BODY_ATR", 0.45))
    require_retest = _env_bool("GOLD_AI_MOMENTUM_FLAG_REQUIRE_RETEST", True)
    retest_max_bars = max(1, _env_int("GOLD_AI_MOMENTUM_FLAG_RETEST_MAX_BARS", 4))
    retest_tol_atr = max(0.01, _env_float("GOLD_AI_MOMENTUM_FLAG_RETEST_TOL_ATR", 0.12))

    ema_fast = max(5, _env_int("GOLD_AI_MOMENTUM_EMA_FAST", 20))
    ema_slow = max(ema_fast + 2, _env_int("GOLD_AI_MOMENTUM_EMA_SLOW", 50))
    slope_bars = max(1, _env_int("GOLD_AI_MOMENTUM_SLOPE_BARS", 3))

    asset_class = (cache or {}).get("__asset_class__", "forex")
    k_trend = await get_klines(symbol, asset_class, trend_tf, lookback)
    k = await get_klines(symbol, asset_class, tf, lookback)
    if not k_trend or not k:
        return False, "momentum_flag_break:missing_klines", {}
    trend_rows = k_trend[:-1]
    rows = k[:-1]
    if len(trend_rows) < max(ema_slow + slope_bars + 2, 30) or len(rows) < 40:
        return False, "momentum_flag_break:insufficient_data", {}

    trend = _trend_state(
        [float(r[4]) for r in trend_rows],
        ema_fast=ema_fast,
        ema_slow=ema_slow,
        slope_bars=slope_bars,
    )
    if is_long:
        trend_ok = (
            trend["ema_fast"] > trend["ema_slow"]
            and trend["ema_fast_slope"] > 0
            and trend["ema_slow_slope"] >= 0
        )
    else:
        trend_ok = (
            trend["ema_fast"] < trend["ema_slow"]
            and trend["ema_fast_slope"] < 0
            and trend["ema_slow_slope"] <= 0
        )
    if not trend_ok:
        return False, "momentum_flag_break:trend_not_aligned", {}

    opens = [float(r[1]) for r in rows]
    highs = [float(r[2]) for r in rows]
    lows = [float(r[3]) for r in rows]
    closes = [float(r[4]) for r in rows]
    atr = atr_from_klines(k[-80:])
    if atr <= 0:
        return False, "momentum_flag_break:atr_unavailable", {}

    n = len(rows)
    scan_start = max(20, n - 40)
    break_buffer = break_buffer_atr * atr
    retest_tol = retest_tol_atr * atr

    for break_idx in range(n - 2, scan_start - 1, -1):
        bo = opens[break_idx]
        bc = closes[break_idx]
        body_atr = abs(bc - bo) / max(atr, 1e-9)
        if body_atr < break_min_body_atr:
            continue
        if is_long and bc <= bo:
            continue
        if (not is_long) and bc >= bo:
            continue

        for flag_bars in range(flag_min_bars, flag_max_bars + 1):
            flag_start = break_idx - flag_bars
            impulse_start = flag_start - impulse_bars
            if impulse_start < 1:
                continue

            impulse_high = max(highs[impulse_start:flag_start])
            impulse_low = min(lows[impulse_start:flag_start])
            impulse_size = impulse_high - impulse_low
            if impulse_size <= 0:
                continue
            if (impulse_size / max(atr, 1e-9)) < impulse_min_atr:
                continue

            flag_high = max(highs[flag_start:break_idx])
            flag_low = min(lows[flag_start:break_idx])
            if is_long:
                retrace = impulse_high - flag_low
            else:
                retrace = flag_high - impulse_low
            retrace_pct = retrace / max(impulse_size, 1e-9)
            if retrace_pct > max_retrace_pct:
                continue

            break_level = flag_high if is_long else flag_low
            break_ok = bc > (break_level + break_buffer) if is_long else bc < (
                break_level - break_buffer
            )
            if not break_ok:
                continue

            used_retest = False
            retest_idx = break_idx
            if require_retest:
                max_i = min(n - 1, break_idx + retest_max_bars)
                found_retest = False
                for ri in range(break_idx + 1, max_i + 1):
                    if is_long:
                        if lows[ri] <= (break_level + retest_tol) and closes[ri] >= break_level:
                            found_retest = True
                            retest_idx = ri
                            break
                    else:
                        if highs[ri] >= (break_level - retest_tol) and closes[ri] <= break_level:
                            found_retest = True
                            retest_idx = ri
                            break
                if not found_retest:
                    continue
                used_retest = True

            if is_long and min(lows[break_idx : retest_idx + 1]) < (flag_low - break_buffer):
                continue
            if (not is_long) and max(highs[break_idx : retest_idx + 1]) > (
                flag_high + break_buffer
            ):
                continue

            spot = float(current_price or closes[-1])
            meta = {
                "momentum_mode": "flag_break",
                "momentum_trend_tf": trend_tf,
                "momentum_trigger_tf": tf,
                "momentum_break_level": round(break_level, 2),
                "momentum_flag_high": round(flag_high, 2),
                "momentum_flag_low": round(flag_low, 2),
                "momentum_impulse_atr": round(impulse_size / max(atr, 1e-9), 2),
                "momentum_retrace_pct": round(retrace_pct, 3),
                "momentum_break_body_atr": round(body_atr, 2),
                "momentum_used_retest": bool(used_retest),
            }
            logger.info(
                "[gold-ai-momentum] detected type=flag_break dir=%s trend_tf=%s trigger_tf=%s "
                "impulse_atr=%.2f flag_bars=%s retrace_pct=%.3f break_level=%.2f "
                "body_atr=%.2f require_retest=%s used_retest=%s spot=%.2f",
                direction,
                trend_tf,
                tf,
                impulse_size / max(atr, 1e-9),
                flag_bars,
                retrace_pct,
                break_level,
                body_atr,
                require_retest,
                used_retest,
                spot,
            )
            return (
                True,
                (
                    f"Momentum flag break {direction}: break @ {break_level:.2f} "
                    f"retrace={retrace_pct:.2%} impulse={impulse_size/max(atr,1e-9):.2f}ATR "
                    f"{'with retest' if used_retest else 'direct'} -> FIRED"
                ),
                meta,
            )

    return False, "momentum_flag_break:no_valid_flag_break", {}
