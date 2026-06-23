"""Per-setup enable flags for Gold AI Trader scanner (env-driven)."""
from __future__ import annotations

import os
from typing import Dict, Optional


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


# Existing setups default ON. Tier-1 additions default ON (user requested more flow).
_DEFAULTS: Dict[str, bool] = {
    "sweep_pdh": True,
    "sweep_pdl": True,
    "disp_bull": True,
    "disp_bear": True,
    "fvg_bull": True,
    "fvg_bear": True,
    "fvg_retrace_bull": True,
    "fvg_retrace_bear": True,
    "ifvg_bull": True,
    "ifvg_bear": True,
    "ob_bull": True,
    "ob_bear": True,
    "liq_sweep_bull": True,
    "liq_sweep_bear": True,
    "sdp_bull": True,
    "sdp_bear": True,
    "breaker_bull": False,
    "breaker_bear": False,
    "eqh_sweep_bear": False,
    "eql_sweep_bull": False,
    "judas_bull": False,
    "judas_bear": False,
    "asian_sweep_bull": False,
    "asian_sweep_bear": False,
}

_ENV_KEYS: Dict[str, str] = {
    "sweep_pdh": "GOLD_AI_SETUP_SWEEP_PDH",
    "sweep_pdl": "GOLD_AI_SETUP_SWEEP_PDL",
    "disp_bull": "GOLD_AI_SETUP_DISP_BULL",
    "disp_bear": "GOLD_AI_SETUP_DISP_BEAR",
    "fvg_bull": "GOLD_AI_SETUP_FVG_BULL",
    "fvg_bear": "GOLD_AI_SETUP_FVG_BEAR",
    "fvg_retrace_bull": "GOLD_AI_SETUP_FVG_RETRACE_BULL",
    "fvg_retrace_bear": "GOLD_AI_SETUP_FVG_RETRACE_BEAR",
    "ifvg_bull": "GOLD_AI_SETUP_IFVG_BULL",
    "ifvg_bear": "GOLD_AI_SETUP_IFVG_BEAR",
    "ob_bull": "GOLD_AI_SETUP_OB_BULL",
    "ob_bear": "GOLD_AI_SETUP_OB_BEAR",
    "liq_sweep_bull": "GOLD_AI_SETUP_LIQ_SWEEP_BULL",
    "liq_sweep_bear": "GOLD_AI_SETUP_LIQ_SWEEP_BEAR",
    "sdp_bull": "GOLD_AI_SETUP_SDP_BULL",
    "sdp_bear": "GOLD_AI_SETUP_SDP_BEAR",
    "breaker_bull": "GOLD_AI_SETUP_BREAKER_BULL",
    "breaker_bear": "GOLD_AI_SETUP_BREAKER_BEAR",
    "eqh_sweep_bear": "GOLD_AI_SETUP_EQH_SWEEP",
    "eql_sweep_bull": "GOLD_AI_SETUP_EQL_SWEEP",
    "judas_bull": "GOLD_AI_SETUP_JUDAS_BULL",
    "judas_bear": "GOLD_AI_SETUP_JUDAS_BEAR",
    "asian_sweep_bull": "GOLD_AI_SETUP_ASIAN_SWEEP_BULL",
    "asian_sweep_bear": "GOLD_AI_SETUP_ASIAN_SWEEP_BEAR",
}


def setup_enabled(setup_key: str) -> bool:
    """Return whether a setup type is enabled for Gold AI scanning."""
    env_key = _ENV_KEYS.get(setup_key)
    default = _DEFAULTS.get(setup_key, False)
    if not env_key:
        return default
    return _env_bool(env_key, default)


def cisd_modifier_enabled() -> bool:
    """CISD alignment is a confidence modifier only — never a standalone fire."""
    return _env_bool("GOLD_AI_CISD_MODIFIER", False)


def smt_modifier_enabled() -> bool:
    """SMT divergence is a confidence modifier only — never a standalone fire."""
    return _env_bool("GOLD_AI_SMT_MODIFIER", True)


def htf_is_directional(bias: dict) -> bool:
    """True when 1h+4h consolidated bias is bullish or bearish (not mixed/chop)."""
    return (bias.get("htf_bias") or "mixed").lower() in ("bullish", "bearish")


def setup_scannable(setup_key: str, bias: Optional[dict] = None) -> bool:
    """
    Whether to evaluate a setup this scan.

    breaker_/eqh_/eql_: auto-enabled when HTF is directional unless env explicitly false.
    All others: follow setup_enabled() env toggles.
    """
    bias = bias or {}
    if setup_key.startswith(("breaker_", "eqh_sweep_", "eql_sweep_")):
        if not htf_is_directional(bias):
            return False
        env_key = _ENV_KEYS.get(setup_key)
        if env_key and os.environ.get(env_key) is not None:
            return setup_enabled(setup_key)
        return True
    return setup_enabled(setup_key)


def max_candidates_per_scan() -> int:
    try:
        return max(1, min(int(os.environ.get("GOLD_AI_TRADER_MAX_CANDIDATES_PER_SCAN", "3")), 5))
    except (TypeError, ValueError):
        return 3
