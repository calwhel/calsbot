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
    return bool(getattr(cfg, "use_limit_entry", True))
