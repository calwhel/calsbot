"""SL/TP band hints for Claude context (structure-derived, not forced)."""
from __future__ import annotations

from typing import List, Optional, Tuple

from app.gold_ai_trader.context_history import parse_zone_from_detail
from app.gold_ai_trader.decision_validator import (
    MIN_RR,
    MAX_SL_ATR_MULT,
    SL_BUFFER_ATR,
    _nearest_tp_candidate,
    _suggested_sl_from_zone,
)


def build_trade_bands_block(
    *,
    spot: float,
    atr: float,
    direction: str,
    setup_detail: str,
    key_levels: List[float],
) -> List[str]:
    """Structured invalidation / target bands for Claude."""
    d = (direction or "").upper()
    zone = parse_zone_from_detail(setup_detail)
    lines = ["=== TRADE BANDS (structure-derived hints) ==="]

    if atr <= 0:
        lines.append("ATR unavailable — size SL/TP from zone and levels carefully.")
        return lines

    max_sl_dist = MAX_SL_ATR_MULT * atr
    lines.append(f"Max SL distance: {max_sl_dist:.2f} ({MAX_SL_ATR_MULT}× 5m ATR)")
    lines.append(f"Min R:R required: {MIN_RR}:1")

    if zone:
        z_bot, z_top = zone
        lines.append(f"Setup zone: {z_bot:.2f} – {z_top:.2f}")
        struct_sl = _suggested_sl_from_zone(zone, d, atr)
        if struct_sl is not None:
            lines.append(
                f"Suggested SL (beyond zone + {SL_BUFFER_ATR}×ATR buffer): {struct_sl:.2f}"
            )
        entry_hint = (z_bot + z_top) / 2.0 if z_bot != z_top else z_bot
        lines.append(f"Entry anchor (zone mid): {entry_hint:.2f} | Spot: {spot:.2f}")
    else:
        entry_hint = spot
        if d == "LONG":
            struct_sl = spot - max_sl_dist
        else:
            struct_sl = spot + max_sl_dist
        lines.append(f"Suggested SL cap: {struct_sl:.2f} (no zone parsed from detail)")

    risk = abs(entry_hint - struct_sl) if struct_sl else max_sl_dist
    if risk > 0 and key_levels:
        tp_c = _nearest_tp_candidate(entry_hint, d, key_levels, MIN_RR, risk)
        if tp_c:
            lines.append(f"TP1 candidate (nearest level ≥{MIN_RR}R): {tp_c:.2f}")
        else:
            min_tp = entry_hint + MIN_RR * risk if d == "LONG" else entry_hint - MIN_RR * risk
            lines.append(f"Min TP for {MIN_RR}R at anchor: {min_tp:.2f}")
        lvl_str = ", ".join(f"{x:.2f}" for x in sorted(set(key_levels))[:8])
        lines.append(f"Key levels for targets: {lvl_str}")

    return lines
