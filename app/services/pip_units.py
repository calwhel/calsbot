"""
Platform vs cTrader broker pip units.

Platform convention (forex_engine.pip_size):
  XAUUSD pip = $0.10 price move (retail/broker terminal convention for P/L, TP/SL, guards).

cTrader ProtoOASymbol.pipPosition:
  Minimum tick / broker pip increment (XAUUSD often 0.01 via pipPosition=2).
  Use ONLY for broker-protocol fields that count in broker pip units — never for
  platform guards, P/L, notifications, or strategy math.
"""
from __future__ import annotations

import logging
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

# cTrader relative SL/TP wire scale for MARKET orders (symbol-independent offset spec).
_BROKER_RELATIVE_WIRE_SCALE = 100_000


def platform_pip_size(symbol: str) -> float:
    """Canonical platform pip size — single source for all platform-facing math."""
    from app.services.forex_engine import pip_size

    ps = float(pip_size(symbol))
    if ps <= 0:
        raise ValueError(f"invalid platform pip_size for {symbol!r}: {ps}")
    return ps


def broker_pip_size_from_metadata(
    *,
    pip_position: Optional[int] = None,
    digits: Optional[int] = None,
    symbol: Optional[str] = None,
) -> float:
    """
    cTrader symbol-metadata pip increment (pipPosition), NOT platform pips.
    pipPosition N → increment 10^-N (EURUSD pipPosition=4 → 0.0001).
    """
    if pip_position is not None and int(pip_position) >= 0:
        return 10.0 ** (-int(pip_position))
    if digits is not None and int(digits) > 0:
        return 10.0 ** (-int(digits))
    sym = (symbol or "").upper()
    if sym == "XAUUSD":
        return 0.01
    if sym in ("XAGUSD",):
        return 0.001
    if sym and "JPY" in sym:
        return 0.01
    return 0.0001


def platform_pips_from_price_delta(symbol: str, price_delta: float) -> float:
    """Signed or unsigned platform pips from a raw price distance."""
    pip = platform_pip_size(symbol)
    return abs(float(price_delta)) / max(pip, 1e-12)


def format_platform_pip_move(
    symbol: str,
    signal_price: float,
    now_price: float,
) -> str:
    """Human-readable move for skip logs: dollar distance + platform pips + pip_size."""
    delta = abs(float(now_price) - float(signal_price))
    pip_sz = platform_pip_size(symbol)
    pips = delta / max(pip_sz, 1e-12)
    sign = "+" if float(now_price) >= float(signal_price) else "−"
    return f"moved ${delta:.3f} (={sign}{pips:.1f} pips, pip_size={pip_sz})"


def platform_pips_to_broker_pips(
    symbol: str,
    platform_pips: float,
    *,
    pip_position: Optional[int] = None,
    digits: Optional[int] = None,
) -> float:
    """Convert platform pips → broker pipPosition units for protocol fields."""
    plat = platform_pip_size(symbol)
    brok = broker_pip_size_from_metadata(
        pip_position=pip_position, digits=digits, symbol=symbol,
    )
    return float(platform_pips) * (plat / max(brok, 1e-12))


def broker_pips_to_platform_pips(
    symbol: str,
    broker_pips: float,
    *,
    pip_position: Optional[int] = None,
    digits: Optional[int] = None,
) -> float:
    """Convert broker pipPosition units → platform pips."""
    plat = platform_pip_size(symbol)
    brok = broker_pip_size_from_metadata(
        pip_position=pip_position, digits=digits, symbol=symbol,
    )
    return float(broker_pips) * (brok / max(plat, 1e-12))


def to_broker_relative_wire_units(price_distance: float) -> int:
    """
    Relative SL/TP magnitude for cTrader MARKET orders.
    Spec: positive offset in 1/100_000 price units (NOT platform pips).
    """
    return max(1, int(round(abs(float(price_distance)) * _BROKER_RELATIVE_WIRE_SCALE)))


def from_broker_relative_wire_units(wire_units: int) -> float:
    """Decode relative wire offset back to price distance."""
    return float(wire_units) / _BROKER_RELATIVE_WIRE_SCALE


def platform_usd_per_pip_per_lot(symbol: str) -> float:
    """USD P&L per platform pip per standard lot (for risk-based sizing)."""
    sym = (symbol or "").upper()
    if sym == "XAUUSD":
        return 10.0
    if sym == "XAGUSD":
        return 5.0
    if "JPY" in sym:
        return 9.28
    return 10.0


def sl_pips_platform(
    symbol: str,
    entry_price: float,
    sl_price: float,
    *,
    sl_pips_hint: Optional[float] = None,
) -> float:
    """Effective SL distance in platform pips."""
    if sl_pips_hint is not None and float(sl_pips_hint) > 0:
        return float(sl_pips_hint)
    pip = platform_pip_size(symbol)
    return abs(float(entry_price) - float(sl_price)) / max(pip, 1e-12)
