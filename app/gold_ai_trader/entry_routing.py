"""Per-setup limit vs market entry routing."""
from __future__ import annotations

import os
from typing import Optional

from app.gold_ai_trader.config import GoldAiRuntimeConfig

# Retrace / zone setups — limit at Claude entry.
_LIMIT_PREFIXES = (
    "ob_", "fvg_retrace_", "ifvg_", "breaker_", "momentum_ema_bounce_",
)

# Momentum / reclaim setups — market (or tight limit fallback in executor).
_MARKET_PREFIXES = (
    "sweep_", "liq_sweep_", "sdp_", "disp_", "eqh_sweep_", "eql_sweep_",
    "asian_sweep_", "judas_", "momentum_flag_break_", "liquidity_grab_",
)


def _env_bool(name: str, default: Optional[bool] = None) -> Optional[bool]:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def use_limit_entry_for_setup(setup_type: str, cfg: GoldAiRuntimeConfig) -> bool:
    """
    Per-setup entry mode. Global cfg.use_limit_entry is the default when type unknown.
    Env GOLD_AI_FORCE_MARKET_ENTRY=true forces all market.
    Env GOLD_AI_FORCE_LIMIT_ENTRY=true forces all limit.
    """
    force_m = _env_bool("GOLD_AI_FORCE_MARKET_ENTRY")
    force_l = _env_bool("GOLD_AI_FORCE_LIMIT_ENTRY")
    if force_m:
        return False
    if force_l:
        return True

    t = setup_type or ""
    for p in _MARKET_PREFIXES:
        if t.startswith(p):
            return False
    for p in _LIMIT_PREFIXES:
        if t.startswith(p):
            return True
    return bool(getattr(cfg, "use_limit_entry", True))
