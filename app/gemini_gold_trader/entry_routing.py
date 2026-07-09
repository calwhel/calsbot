"""Per-setup limit vs market entry routing for Gemini Gold."""
from __future__ import annotations

import os
from typing import Optional

from app.gemini_gold_trader.config import GeminiGoldRuntimeConfig

_LIMIT_PREFIXES = (
    "fvg_retrace",
    "ifvg",
    "breaker",
    "order_block",
    "ob_",
    "momentum_ema_bounce",
)

_MARKET_PREFIXES = (
    "sweep",
    "liq_sweep",
    "session_liquidity_sweep",
    "liquidity_grab",
    "eqh_sweep",
    "eql_sweep",
    "asian_sweep",
    "momentum_scalp",
    "momentum_flag",
    "disp_",
    "sdp_",
    "orb_",
    "opening_range",
)


def _env_bool(name: str, default: Optional[bool] = None) -> Optional[bool]:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def use_limit_entry_for_setup(setup_type: str, cfg: GeminiGoldRuntimeConfig) -> bool:
    if _env_bool("GEMINI_GOLD_FORCE_MARKET_ENTRY"):
        return False
    if _env_bool("GEMINI_GOLD_FORCE_LIMIT_ENTRY"):
        return True

    t = (setup_type or "").lower().strip()
    for p in _MARKET_PREFIXES:
        if t.startswith(p) or p in t:
            return False
    for p in _LIMIT_PREFIXES:
        if t.startswith(p) or p in t:
            return True
    return bool(getattr(cfg, "use_limit_entry", False))


def is_market_setup_type(setup_type: str) -> bool:
    """True when setup routing prefers immediate market entry."""
    t = (setup_type or "").lower().strip()
    if not t:
        return False
    for p in _MARKET_PREFIXES:
        if t.startswith(p) or p in t:
            return True
    return False


def entry_touch_tolerance(
    entry_price: float,
    setup_type: str = "",
    *,
    wide: bool = False,
) -> float:
    """
    Price band for entry-watch fills. Gold needs wider bands than FX defaults.
    `wide` applies for liquidity grabs at session extremes (shallower pullbacks).
    """
    try:
        entry = abs(float(entry_price))
    except (TypeError, ValueError):
        entry = 0.0
    base = max(0.50, entry * 0.0002)
    if wide or is_market_setup_type(setup_type):
        return max(base, 1.0, entry * 0.00035)
    if any(p in (setup_type or "").lower() for p in ("fvg", "ifvg", "ob_", "order_block")):
        return max(base, 0.75, entry * 0.00025)
    return base


def market_fallback_on_pending_expire(setup_type: str, cfg: GeminiGoldRuntimeConfig) -> bool:
    """When entry-watch expires, allow market entry for momentum / grab setups."""
    raw = os.environ.get("GEMINI_GOLD_EXPIRE_MARKET_FALLBACK")
    if raw is not None and raw.strip().lower() in ("0", "false", "no", "off"):
        return False
    if is_market_setup_type(setup_type):
        return True
    if not bool(getattr(cfg, "use_limit_entry", False)):
        return True
    return False
