"""Invert Gemini TAKE decisions before execution (fade the model)."""
from __future__ import annotations

from typing import Any, Dict, Optional


def _dir_norm(d: str) -> Optional[str]:
    raw = (d or "").strip().upper()
    if raw in ("LONG", "SHORT"):
        return raw
    if raw in ("BUY",):
        return "LONG"
    if raw in ("SELL",):
        return "SHORT"
    return None


def invert_take_decision(decision: Dict[str, Any]) -> Dict[str, Any]:
    """Flip direction and swap SL/TP. LONG→SHORT, BUY→SELL, etc."""
    d = decision.copy()
    direction = _dir_norm(d.get("direction") or "")
    if not direction:
        return d

    d["direction"] = "SHORT" if direction == "LONG" else "LONG"

    sl = d.get("stop_loss")
    tp = d.get("take_profit")
    d["stop_loss"] = tp
    d["take_profit"] = sl
    return d
