"""Session-specific setup allowlists — Asia/London/NY playbook."""
from __future__ import annotations

from typing import FrozenSet, Optional

# London killzone: sweeps at range edges, zone retraces, Asia plays.
_LONDON_PRIMARY: FrozenSet[str] = frozenset({
    "sweep_pdh", "sweep_pdl",
    "liq_sweep_bull", "liq_sweep_bear",
    "fvg_retrace_bull", "fvg_retrace_bear",
    "ifvg_bull", "ifvg_bear",
    "ob_bull", "ob_bear",
    "eqh_sweep_bear", "eql_sweep_bull",
    "asian_sweep_bull", "asian_sweep_bear",
    "sdp_bull", "sdp_bear",
    "momentum_ema_bounce_long", "momentum_ema_bounce_short",
    "momentum_flag_break_long", "momentum_flag_break_short",
    "liquidity_grab_long", "liquidity_grab_short",
})

# NY killzone: continuation structures; breakers when HTF directional.
_NY_PRIMARY: FrozenSet[str] = frozenset({
    "ob_bull", "ob_bear",
    "ifvg_bull", "ifvg_bear",
    "fvg_retrace_bull", "fvg_retrace_bear",
    "breaker_bull", "breaker_bear",
    "disp_bull", "disp_bear",
    "liq_sweep_bull", "liq_sweep_bear",
    "sweep_pdh", "sweep_pdl",
    "eqh_sweep_bear", "eql_sweep_bull",
    "sdp_bull", "sdp_bear",
    "momentum_ema_bounce_long", "momentum_ema_bounce_short",
    "momentum_flag_break_long", "momentum_flag_break_short",
    "liquidity_grab_long", "liquidity_grab_short",
})

_ASIA_PRIMARY: FrozenSet[str] = frozenset({
    "asian_sweep_bull", "asian_sweep_bear",
    "sweep_pdh", "sweep_pdl",
    "liq_sweep_bull", "liq_sweep_bear",
    "eqh_sweep_bear", "eql_sweep_bull",
    "sdp_bull", "sdp_bear",
    "momentum_ema_bounce_long", "momentum_ema_bounce_short",
    "momentum_flag_break_long", "momentum_flag_break_short",
    "liquidity_grab_long", "liquidity_grab_short",
})

# Momentum setups allowed in London only when HTF aligns (no counter-trend fade).
_LONDON_HTF_ONLY: FrozenSet[str] = frozenset({
    "disp_bull", "disp_bear",
    "breaker_bull", "breaker_bear",
})


def _htf_aligned(align_reason: str) -> bool:
    r = (align_reason or "").lower()
    return "aligned_bull" in r or "aligned_bear" in r


def session_allows_setup(
    setup_key: str,
    session: str,
    *,
    htf_align_reason: Optional[str] = None,
) -> tuple[bool, str]:
    """
    Return whether a setup type fits the active session playbook.

    Counter-HTF fades are blocked via HTF gate elsewhere; this filters
    session-inappropriate setup *types* (e.g. Asian sweep in NY).
    """
    s = (session or "").lower()
    key = setup_key or ""

    if s == "london":
        if key in _LONDON_PRIMARY:
            return True, "london_primary"
        if key in _LONDON_HTF_ONLY:
            if _htf_aligned(htf_align_reason or ""):
                return True, "london_htf_aligned"
            return False, "london_htf_only_setup"
        return False, "london_not_in_playbook"

    if s == "new_york":
        if key in _NY_PRIMARY:
            return True, "ny_primary"
        return False, "ny_not_in_playbook"

    if s == "asia":
        if key in _ASIA_PRIMARY:
            return True, "asia_primary"
        return False, "asia_not_in_playbook"

    return True, "unknown_session"
