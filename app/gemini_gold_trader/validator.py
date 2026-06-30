"""Lightweight post-Gemini validation before execution."""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

from app.gemini_gold_trader.config import GeminiGoldRuntimeConfig


def _dir_norm(d: str) -> Optional[str]:
    raw = (d or "").strip().upper()
    if raw in ("LONG", "SHORT"):
        return raw
    if raw in ("BUY",):
        return "LONG"
    if raw in ("SELL",):
        return "SHORT"
    return None


def validate_take_decision(
    decision: Dict[str, Any],
    *,
    cfg: GeminiGoldRuntimeConfig,
    spot: float,
) -> Tuple[bool, str, Dict[str, Any]]:
    """Validate TAKE prices. Returns (ok, reason, decision_copy)."""
    d = decision.copy()
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
    if cfg.min_sl_pips > 0 and risk_dist > 0:
        try:
            from app.services.pip_units import platform_pips_from_price_delta

            sl_pips = float(platform_pips_from_price_delta("XAUUSD", risk_dist))
        except Exception:
            sl_pips = 0.0
        if sl_pips > 0 and sl_pips < cfg.min_sl_pips:
            return False, f"validator:sl_too_tight({sl_pips:.1f}<{cfg.min_sl_pips:.1f}pips)", d

    if spot > 0 and cfg.entry_max_drift_pct > 0:
        drift_pct = abs(spot - entry) / spot * 100.0
        if drift_pct > cfg.entry_max_drift_pct:
            if direction == "LONG" and spot > entry * (1 + cfg.entry_max_drift_pct / 100.0):
                return False, f"validator:entry_chasing({drift_pct:.2f}%>{cfg.entry_max_drift_pct}%)", d
            if direction == "SHORT" and spot < entry * (1 - cfg.entry_max_drift_pct / 100.0):
                return False, f"validator:entry_chasing({drift_pct:.2f}%>{cfg.entry_max_drift_pct}%)", d

    return True, "ok", d
