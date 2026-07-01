"""Post-Claude structure validator — SL/TP/R:R/zone checks before broker."""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

from app.gold_ai_trader.context_history import parse_zone_from_detail


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


MIN_RR = _env_float("GOLD_AI_MIN_RR", 1.5)
# 0 = no execution cap (swing/zone invalidation allowed); set e.g. 1.0 to re-enable hard scalp cap
MAX_SL_ATR_MULT = _env_float("GOLD_AI_MAX_SL_ATR", 0.0)
MIN_SL_PIPS = _env_float("GOLD_AI_MIN_SL_PIPS", 60.0)
# Planning horizon for readiness R:R feasibility only (not an execution limit)
READINESS_RR_RISK_ATR = _env_float("GOLD_AI_READINESS_RR_ATR", 3.0)
SL_BUFFER_ATR = _env_float("GOLD_AI_SL_BUFFER_ATR", 0.08)
ORB_ENTRY_MAX_BREAK_ATR = _env_float("GOLD_AI_ORB_ENTRY_MAX_BREAK_ATR", 0.60)
ORB_ENTRY_MAX_BREAK_RANGE_PCT = _env_float("GOLD_AI_ORB_ENTRY_MAX_BREAK_RANGE_PCT", 0.25)
MOMENTUM_FLAG_ENTRY_MAX_ATR = _env_float("GOLD_AI_MOMENTUM_FLAG_ENTRY_MAX_ATR", 1.00)
MOMENTUM_FLAG_RETEST_ENTRY_MAX_ATR = _env_float(
    "GOLD_AI_MOMENTUM_FLAG_RETEST_ENTRY_MAX_ATR", 0.45
)
LIQ_GRAB_ENTRY_MAX_ATR = _env_float("GOLD_AI_LIQ_GRAB_ENTRY_MAX_ATR", 0.50)


def sl_width_cap_enabled() -> bool:
    return MAX_SL_ATR_MULT > 0


def min_rr_enabled() -> bool:
    """True when GOLD_AI_MIN_RR > 0 (otherwise Claude sizes TP/SL freely)."""
    return MIN_RR > 0


def min_sl_pips_enabled() -> bool:
    """True when GOLD_AI_MIN_SL_PIPS > 0 (minimum stop distance floor)."""
    return MIN_SL_PIPS > 0


def _dir_norm(d: str) -> Optional[str]:
    raw = (d or "").strip().lower()
    if raw in ("long", "buy"):
        return "LONG"
    if raw in ("short", "sell"):
        return "SHORT"
    up = raw.upper()
    if up in ("LONG", "SHORT"):
        return up
    return None


def _risk_reward(entry: float, sl: float, tp: float, direction: str) -> Optional[float]:
    if entry <= 0 or sl <= 0 or tp <= 0:
        return None
    if direction == "LONG":
        risk = entry - sl
        reward = tp - entry
    else:
        risk = sl - entry
        reward = entry - tp
    if risk <= 0 or reward <= 0:
        return None
    return reward / risk


def _suggested_sl_from_zone(
    zone: Optional[Tuple[float, float]],
    direction: str,
    atr: float,
) -> Optional[float]:
    if not zone or atr <= 0:
        return None
    z_bot, z_top = zone
    buf = SL_BUFFER_ATR * atr
    if direction == "LONG":
        return z_bot - buf
    return z_top + buf


def _nearest_tp_candidate(
    entry: float,
    direction: str,
    levels: List[float],
    min_rr: float,
    risk: float,
) -> Optional[float]:
    """First key level in trade direction meeting min R:R (min_rr=0 → any level beyond entry)."""
    if risk <= 0:
        return None
    min_dist = risk * min_rr if min_rr > 0 else 0.0
    cands = []
    for lv in levels:
        if direction == "LONG" and lv > entry + min_dist:
            cands.append(lv)
        elif direction == "SHORT" and lv < entry - min_dist:
            cands.append(lv)
    if not cands:
        return None
    if direction == "LONG":
        return min(cands)
    return max(cands)


def validate_take_decision(
    decision: Dict[str, Any],
    *,
    candidate_direction: str,
    spot: float,
    atr: float,
    setup_detail: str,
    key_levels: Optional[List[float]] = None,
) -> Tuple[bool, str, Dict[str, Any]]:
    """
    Validate Claude TAKE prices. Returns (ok, reason, decision_copy).
    On soft TP fix only, may adjust take_profit in returned copy.
    """
    d = decision.copy()
    direction = _dir_norm(d.get("direction") or candidate_direction)
    if not direction:
        return False, "validator:no_direction", d

    try:
        entry = float(d.get("entry") or spot or 0)
        sl = float(d.get("stop_loss") or 0)
        tp = float(d.get("take_profit") or 0)
    except (TypeError, ValueError):
        return False, "validator:bad_prices", d

    if entry <= 0 or sl <= 0 or tp <= 0:
        return False, "validator:missing_prices", d

    if direction == "LONG":
        if not (sl < entry < tp):
            return False, "validator:long_price_order", d
    else:
        if not (tp < entry < sl):
            return False, "validator:short_price_order", d

    risk_dist = abs(entry - sl)
    if min_sl_pips_enabled() and risk_dist > 0:
        symbol = str(d.get("symbol") or "XAUUSD").strip().upper() or "XAUUSD"
        try:
            from app.services.pip_units import platform_pips_from_price_delta

            sl_pips = float(platform_pips_from_price_delta(symbol, risk_dist))
        except Exception:
            sl_pips = 0.0
        d["validator_sl_pips"] = round(sl_pips, 1)
        if sl_pips > 0 and sl_pips < MIN_SL_PIPS:
            return False, f"validator:sl_too_tight({sl_pips:.1f}<{MIN_SL_PIPS:.1f}pips)", d

    if sl_width_cap_enabled() and atr > 0 and risk_dist > MAX_SL_ATR_MULT * atr:
        return False, f"validator:sl_too_wide({risk_dist:.2f}>{MAX_SL_ATR_MULT}×ATR)", d
    if atr > 0 and risk_dist > 0:
        sl_atr = risk_dist / atr
        d["validator_sl_atr"] = round(sl_atr, 2)

    if (d.get("validator_profile") or "").strip().lower() == "orb":
        try:
            break_level = float(d.get("orb_break_level") or 0)
        except (TypeError, ValueError):
            break_level = 0.0
        try:
            range_height = float(d.get("orb_range_height") or 0)
        except (TypeError, ValueError):
            range_height = 0.0
        if break_level > 0:
            break_dist = abs(entry - break_level)
            max_break_dist = 0.0
            if atr > 0:
                max_break_dist = max(max_break_dist, ORB_ENTRY_MAX_BREAK_ATR * atr)
            if range_height > 0:
                max_break_dist = max(
                    max_break_dist,
                    ORB_ENTRY_MAX_BREAK_RANGE_PCT * range_height,
                )
            if max_break_dist > 0 and break_dist > max_break_dist:
                return (
                    False,
                    f"validator:entry_chasing_orb({break_dist:.2f}>{max_break_dist:.2f})",
                    d,
                )
            d["validator_break_dist"] = round(break_dist, 2)

    if (d.get("validator_profile") or "").strip().lower() == "momentum_flag":
        try:
            break_level = float(d.get("momentum_break_level") or 0)
        except (TypeError, ValueError):
            break_level = 0.0
        used_retest = bool(d.get("momentum_used_retest"))
        if break_level > 0 and atr > 0:
            break_dist = abs(entry - break_level)
            max_break_dist = (
                MOMENTUM_FLAG_RETEST_ENTRY_MAX_ATR * atr
                if used_retest
                else MOMENTUM_FLAG_ENTRY_MAX_ATR * atr
            )
            if max_break_dist > 0 and break_dist > max_break_dist:
                return (
                    False,
                    f"validator:entry_chasing_momentum_flag({break_dist:.2f}>{max_break_dist:.2f})",
                    d,
                )
            d["validator_momentum_break_dist"] = round(break_dist, 2)

    if (d.get("validator_profile") or "").strip().lower() == "liquidity_grab":
        try:
            mss_level = float(d.get("liq_grab_mss_level") or 0)
        except (TypeError, ValueError):
            mss_level = 0.0
        if mss_level > 0 and atr > 0:
            mss_dist = abs(entry - mss_level)
            max_mss_dist = LIQ_GRAB_ENTRY_MAX_ATR * atr
            if max_mss_dist > 0 and mss_dist > max_mss_dist:
                return (
                    False,
                    f"validator:entry_chasing_liquidity_grab({mss_dist:.2f}>{max_mss_dist:.2f})",
                    d,
                )
            d["validator_liq_grab_dist"] = round(mss_dist, 2)

    zone = parse_zone_from_detail(setup_detail)
    if zone and atr > 0:
        z_bot, z_top = zone
        from app.services.strategy_ta import entry_zone_allows_price, entry_max_dist_from_zone_atr

        ok_z, zmsg = entry_zone_allows_price(entry, z_bot, z_top, direction.lower(), atr)
        if not ok_z and "chasing" in zmsg:
            return False, f"validator:entry_chasing({zmsg})", d

        struct_sl = _suggested_sl_from_zone(zone, direction, atr)
        if struct_sl is not None and sl_width_cap_enabled():
            if direction == "LONG" and sl < struct_sl - 0.05 * atr:
                return False, "validator:sl_below_structure", d
            if direction == "SHORT" and sl > struct_sl + 0.05 * atr:
                return False, "validator:sl_above_structure", d

    rr = _risk_reward(entry, sl, tp, direction)
    if rr is None:
        return False, "validator:invalid_rr_geometry", d
    if min_rr_enabled() and rr < MIN_RR:
        levels = key_levels or []
        alt_tp = _nearest_tp_candidate(entry, direction, levels, MIN_RR, risk_dist)
        if alt_tp is not None:
            d["take_profit"] = round(alt_tp, 2)
            rr2 = _risk_reward(entry, sl, float(d["take_profit"]), direction)
            if rr2 is not None and rr2 >= MIN_RR:
                d["validator_note"] = f"tp_adjusted_to_key_level_rr={rr2:.2f}"
                return True, "validator:ok_tp_adjusted", d
        return False, f"validator:rr_below_min({rr:.2f}<{MIN_RR})", d

    d["validator_note"] = f"rr={rr:.2f}"
    return True, "validator:ok", d
