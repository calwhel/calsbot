"""Lightweight post-Gemini validation before execution."""
from __future__ import annotations

import os
from typing import Any, Dict, Optional, Tuple

from app.gemini_gold_trader.config import GeminiGoldRuntimeConfig

ORB_ENTRY_MAX_BREAK_ATR = float(os.environ.get("GEMINI_GOLD_ORB_ENTRY_MAX_BREAK_ATR", "0.60"))
ORB_ENTRY_MAX_BREAK_RANGE_PCT = float(os.environ.get("GEMINI_GOLD_ORB_ENTRY_MAX_BREAK_RANGE_PCT", "0.25"))
MOMENTUM_FLAG_ENTRY_MAX_ATR = float(os.environ.get("GEMINI_GOLD_MOMENTUM_FLAG_ENTRY_MAX_ATR", "1.00"))
MOMENTUM_FLAG_RETEST_ENTRY_MAX_ATR = float(os.environ.get("GEMINI_GOLD_MOMENTUM_FLAG_RETEST_ENTRY_MAX_ATR", "0.45"))
LIQ_GRAB_ENTRY_MAX_ATR = float(os.environ.get("GEMINI_GOLD_LIQ_GRAB_ENTRY_MAX_ATR", "0.50"))


def _dir_norm(d: str) -> Optional[str]:
    raw = (d or "").strip().upper()
    if raw in ("LONG", "SHORT"):
        return raw
    if raw in ("BUY",):
        return "LONG"
    if raw in ("SELL",):
        return "SHORT"
    return None


def _platform_pips(distance: float) -> float:
    try:
        from app.services.pip_units import platform_pips_from_price_delta

        return float(platform_pips_from_price_delta("XAUUSD", distance))
    except Exception:
        return 0.0


def apply_validator_profile(decision: Dict[str, Any]) -> Dict[str, Any]:
    d = decision.copy()
    setup = str(d.get("setup_type") or "").lower()
    if setup.startswith("orb_"):
        d["validator_profile"] = "orb"
    elif "momentum" in setup or setup.startswith("disp_") or setup.startswith("sdp_"):
        d["validator_profile"] = "momentum_flag"
    elif (
        "liquidity" in setup
        or "liq" in setup
        or setup.startswith("sweep")
        or setup.startswith("eqh")
        or setup.startswith("eql")
        or setup.startswith("asian_sweep")
    ):
        d["validator_profile"] = "liquidity_grab"
    elif setup.startswith(("fvg", "ifvg", "ob_", "breaker", "order_block")):
        d["validator_profile"] = "zone_retrace"
    return d


def validate_take_decision(
    decision: Dict[str, Any],
    *,
    cfg: GeminiGoldRuntimeConfig,
    spot: float,
    atr: float = 0.0,
) -> Tuple[bool, str, Dict[str, Any]]:
    """Validate TAKE prices. Returns (ok, reason, decision_copy)."""
    d = apply_validator_profile(decision)
    direction = _dir_norm(d.get("direction") or "")
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

    d["entry"] = entry
    d["stop_loss"] = sl
    d["take_profit"] = tp
    d["direction"] = direction

    if direction == "LONG":
        if not (sl < entry < tp):
            return False, "validator:long_price_order", d
    else:
        if not (tp < entry < sl):
            return False, "validator:short_price_order", d

    risk_dist = abs(entry - sl)
    reward_dist = abs(tp - entry)
    sl_pips = _platform_pips(risk_dist) if risk_dist > 0 else 0.0

    if cfg.min_sl_pips > 0 and sl_pips > 0 and sl_pips < cfg.min_sl_pips:
        return False, f"validator:sl_too_tight({sl_pips:.1f}<{cfg.min_sl_pips:.1f}pips)", d

    if cfg.max_sl_pips > 0 and sl_pips > cfg.max_sl_pips:
        return (
            False,
            f"validator:sl_too_wide({sl_pips:.1f}>{cfg.max_sl_pips:.1f}pips)",
            d,
        )

    profile = (d.get("validator_profile") or "").strip().lower()
    try:
        confidence = int(d.get("confidence") or 0)
    except (TypeError, ValueError):
        confidence = 0
    confidence = max(0, min(100, confidence))
    high_conviction = confidence >= int(cfg.confidence_threshold or 85)

    if profile == "orb" and not high_conviction:
        try:
            break_level = float(d.get("orb_break_level") or 0)
            range_height = float(d.get("orb_range_height") or 0)
        except (TypeError, ValueError):
            break_level = range_height = 0.0
        if break_level > 0:
            break_dist = abs(entry - break_level)
            max_break = 0.0
            if atr > 0:
                max_break = max(max_break, ORB_ENTRY_MAX_BREAK_ATR * atr)
            if range_height > 0:
                max_break = max(max_break, ORB_ENTRY_MAX_BREAK_RANGE_PCT * range_height)
            if max_break > 0 and break_dist > max_break:
                return (
                    False,
                    f"validator:entry_chasing_orb({break_dist:.2f}>{max_break:.2f})",
                    d,
                )

    if profile == "momentum_flag" and not high_conviction:
        try:
            break_level = float(d.get("momentum_break_level") or 0)
        except (TypeError, ValueError):
            break_level = 0.0
        used_retest = bool(d.get("momentum_used_retest"))
        if break_level > 0 and atr > 0:
            break_dist = abs(entry - break_level)
            max_break = (
                MOMENTUM_FLAG_RETEST_ENTRY_MAX_ATR * atr
                if used_retest
                else MOMENTUM_FLAG_ENTRY_MAX_ATR * atr
            )
            if max_break > 0 and break_dist > max_break:
                return (
                    False,
                    f"validator:entry_chasing_momentum({break_dist:.2f}>{max_break:.2f})",
                    d,
                )

    if profile == "liquidity_grab" and not high_conviction:
        try:
            mss_level = float(d.get("liq_grab_mss_level") or 0)
        except (TypeError, ValueError):
            mss_level = 0.0
        if mss_level > 0 and atr > 0:
            mss_dist = abs(entry - mss_level)
            max_mss = LIQ_GRAB_ENTRY_MAX_ATR * atr
            if max_mss > 0 and mss_dist > max_mss:
                return (
                    False,
                    f"validator:entry_chasing_liquidity_grab({mss_dist:.2f}>{max_mss:.2f})",
                    d,
                )

    if risk_dist > 0 and reward_dist > 0:
        rr = reward_dist / risk_dist
        min_rr = max(1.0, float(cfg.min_rr or 1.0))
        max_rr = min(2.0, max(min_rr, float(cfg.max_rr or 2.0)))
        if rr < min_rr - 1e-9:
            if direction == "LONG":
                adj_tp = entry + risk_dist * min_rr
            else:
                adj_tp = entry - risk_dist * min_rr
            adj_reward = abs(adj_tp - entry)
            adj_rr = adj_reward / risk_dist
            if adj_rr >= min_rr - 1e-9 and adj_rr <= max_rr + 1e-9:
                d["take_profit"] = round(adj_tp, 2)
                d["validator_note"] = f"tp_adjusted_for_min_rr({rr:.2f}->{adj_rr:.2f})"
                rr = adj_rr
            else:
                return False, f"validator:rr_too_low({rr:.2f}<{min_rr:.1f})", d
        if rr > max_rr + 1e-9:
            return False, f"validator:rr_too_high({rr:.2f}>{max_rr:.1f})", d

    if spot > 0 and cfg.entry_max_drift_pct > 0:
        drift_pct = abs(spot - entry) / spot * 100.0
        if drift_pct > cfg.entry_max_drift_pct:
            if direction == "LONG" and spot > entry * (1 + cfg.entry_max_drift_pct / 100.0):
                return False, f"validator:entry_chasing({drift_pct:.2f}%>{cfg.entry_max_drift_pct}%)", d
            if direction == "SHORT" and spot < entry * (1 - cfg.entry_max_drift_pct / 100.0):
                return False, f"validator:entry_chasing({drift_pct:.2f}%>{cfg.entry_max_drift_pct}%)", d

    return True, "ok", d
