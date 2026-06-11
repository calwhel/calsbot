"""Abort live cTrader orders when the signal price or age is no longer valid."""

from __future__ import annotations

import logging
import os
import time
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

_DEFAULT_MAX_AGE_S = float(os.environ.get("CTRADER_MAX_SIGNAL_AGE_S", "30"))
_DEFAULT_MAX_SLIPPAGE_PIPS = float(os.environ.get("CTRADER_MAX_SLIPPAGE_PIPS", "15"))


def _max_slippage_pips(symbol: str) -> float:
    sym = (symbol or "").upper()
    env_key = f"CTRADER_MAX_SLIPPAGE_PIPS_{sym}"
    raw = os.environ.get(env_key)
    if raw is not None and str(raw).strip():
        try:
            return float(raw)
        except ValueError:
            pass
    if sym == "XAUUSD":
        return float(os.environ.get("CTRADER_MAX_SLIPPAGE_PIPS_XAUUSD", "15"))
    return _DEFAULT_MAX_SLIPPAGE_PIPS


def _pip_size(symbol: str) -> float:
    try:
        from app.services.forex_engine import pip_size

        ps = pip_size(symbol)
        if ps and ps > 0:
            return float(ps)
    except Exception:
        pass
    from app.services.ctrader_client import _PIP_SIZES

    return float(_PIP_SIZES.get((symbol or "").upper(), 0.0001))


def _live_mid(symbol: str) -> Optional[float]:
    sym = (symbol or "").upper()
    try:
        from app.services.spot_price_store import get_mid

        mid = get_mid(sym, max_age_s=15.0)
        if mid and mid > 0:
            return float(mid)
    except Exception:
        pass
    try:
        from app.services.ctrader_price_feed import get_price

        px = get_price(sym)
        if px and px > 0:
            return float(px)
    except Exception:
        pass
    return None


def check_signal_stale(
    *,
    symbol: str,
    direction: str,
    signal_price: float,
    signal_mono: float,
    max_age_s: Optional[float] = None,
) -> Optional[Tuple[str, float, float]]:
    """
    Return (reason, age_s, slip_pips) when the live order should be aborted.
  """
    max_age = float(max_age_s if max_age_s is not None else _DEFAULT_MAX_AGE_S)
    age_s = max(0.0, time.monotonic() - float(signal_mono))
    if age_s > max_age:
        return (
            f"signal stale (age {age_s:.0f}s > {max_age:.0f}s)",
            age_s,
            0.0,
        )

    live = _live_mid(symbol)
    if live is None or not signal_price or signal_price <= 0:
        return None

    pip = _pip_size(symbol)
    slip_pips = abs(live - float(signal_price)) / max(pip, 1e-12)
    max_slip = _max_slippage_pips(symbol)
    if slip_pips > max_slip:
        return (
            f"price moved {slip_pips:.1f} pips in {age_s:.0f}s (max {max_slip:.0f})",
            age_s,
            slip_pips,
        )
    return None
