"""Pure TP/SL helpers for cTrader orders (no protobuf imports)."""
from __future__ import annotations

from typing import Optional, Tuple


def compute_sltp_prices(
    direction: str,
    entry_price: float,
    tp_pct: float,
    sl_pct: float,
) -> Tuple[float, float]:
    """Derive absolute TP/SL from entry and strategy percentages."""
    mult = 1.0 if direction == "LONG" else -1.0
    tp_price = round(entry_price * (1 + mult * tp_pct / 100), 6)
    sl_price = round(entry_price * (1 - mult * sl_pct / 100), 6)
    return tp_price, sl_price


def validate_sltp_sanity(
    direction: str,
    entry_price: float,
    sl_price: Optional[float],
    tp_price: Optional[float],
) -> bool:
    """SHORT → SL above entry, TP below; LONG → SL below, TP above."""
    if not entry_price or entry_price <= 0:
        return False
    if sl_price is not None and sl_price > 0:
        if direction == "SHORT" and sl_price <= entry_price:
            return False
        if direction == "LONG" and sl_price >= entry_price:
            return False
    if tp_price is not None and tp_price > 0:
        if direction == "SHORT" and tp_price >= entry_price:
            return False
        if direction == "LONG" and tp_price <= entry_price:
            return False
    return True


def relative_sltp_wire(
    entry_price: float,
    sl_pct: Optional[float],
    tp_pct: Optional[float],
) -> Tuple[Optional[int], Optional[int]]:
    """Relative SL/TP magnitudes for MARKET orders (broker wire units, not platform pips)."""
    from app.services.pip_units import to_broker_relative_wire_units

    rel_sl = rel_tp = None
    if entry_price and entry_price > 0:
        if sl_pct is not None and sl_pct > 0:
            rel_sl = to_broker_relative_wire_units(entry_price * (sl_pct / 100))
        if tp_pct is not None and tp_pct > 0:
            rel_tp = to_broker_relative_wire_units(entry_price * (tp_pct / 100))
    return rel_sl, rel_tp
