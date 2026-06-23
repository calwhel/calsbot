"""Per-setup enable flags for Gold AI Trader scanner (env-driven)."""
from __future__ import annotations

import os
from typing import Dict


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


# Existing setups default ON (preserve current behaviour).
_DEFAULTS: Dict[str, bool] = {
    "sweep_pdh": True,
    "sweep_pdl": True,
    "disp_bull": True,
    "disp_bear": True,
    "fvg_bull": True,
    "fvg_bear": True,
    "liq_sweep_bull": True,
    "liq_sweep_bear": True,
    "sdp_bull": True,
    "sdp_bear": True,
    # New structure setups default OFF for safe demo rollout.
    "breaker_bull": False,
    "breaker_bear": False,
    "eqh_sweep_bear": False,
    "eql_sweep_bull": False,
}

_ENV_KEYS: Dict[str, str] = {
    "sweep_pdh": "GOLD_AI_SETUP_SWEEP_PDH",
    "sweep_pdl": "GOLD_AI_SETUP_SWEEP_PDL",
    "disp_bull": "GOLD_AI_SETUP_DISP_BULL",
    "disp_bear": "GOLD_AI_SETUP_DISP_BEAR",
    "fvg_bull": "GOLD_AI_SETUP_FVG_BULL",
    "fvg_bear": "GOLD_AI_SETUP_FVG_BEAR",
    "liq_sweep_bull": "GOLD_AI_SETUP_LIQ_SWEEP_BULL",
    "liq_sweep_bear": "GOLD_AI_SETUP_LIQ_SWEEP_BEAR",
    "sdp_bull": "GOLD_AI_SETUP_SDP_BULL",
    "sdp_bear": "GOLD_AI_SETUP_SDP_BEAR",
    "breaker_bull": "GOLD_AI_SETUP_BREAKER_BULL",
    "breaker_bear": "GOLD_AI_SETUP_BREAKER_BEAR",
    "eqh_sweep_bear": "GOLD_AI_SETUP_EQH_SWEEP",
    "eql_sweep_bull": "GOLD_AI_SETUP_EQL_SWEEP",
}


def setup_enabled(setup_key: str) -> bool:
    """Return whether a setup type is enabled for Gold AI scanning."""
    env_key = _ENV_KEYS.get(setup_key)
    default = _DEFAULTS.get(setup_key, False)
    if not env_key:
        return default
    return _env_bool(env_key, default)


def smt_modifier_enabled() -> bool:
    """SMT divergence is a confidence modifier only — never a standalone fire."""
    return _env_bool("GOLD_AI_SMT_MODIFIER", False)
