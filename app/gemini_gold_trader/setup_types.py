"""Gemini Gold setup_type vocabulary (aligned with Gold AI scanner ids)."""
from __future__ import annotations

from typing import Optional

# Exact setup_type values Gemini may return on TAKE.
GEMINI_GOLD_SETUP_TYPES: tuple[str, ...] = (
    # Zone / retrace (limit-friendly)
    "fvg_retrace_bull",
    "fvg_retrace_bear",
    "ifvg_bull",
    "ifvg_bear",
    "ob_bull",
    "ob_bear",
    "breaker_bull",
    "breaker_bear",
    "momentum_ema_bounce_long",
    "momentum_ema_bounce_short",
    # Sweeps / liquidity
    "liq_sweep_bull",
    "liq_sweep_bear",
    "sweep_pdh",
    "sweep_pdl",
    "eqh_sweep_bear",
    "eql_sweep_bull",
    "asian_sweep_bull",
    "asian_sweep_bear",
    "liquidity_grab_long",
    "liquidity_grab_short",
    # Momentum / displacement
    "momentum_flag_break_long",
    "momentum_flag_break_short",
    "disp_bull",
    "disp_bear",
    "sdp_bull",
    "sdp_bear",
    # ORB
    "orb_long",
    "orb_short",
    # Legacy aliases (still accepted; normalized on ingest)
    "session_liquidity_sweep",
    "fvg_retrace",
    "order_block",
    "momentum_scalp",
    "liquidity_grab",
    "opening_range",
)

_LEGACY_BY_DIRECTION: dict[str, dict[str, str]] = {
    "fvg_retrace": {"LONG": "fvg_retrace_bull", "SHORT": "fvg_retrace_bear"},
    "order_block": {"LONG": "ob_bull", "SHORT": "ob_bear"},
    "momentum_scalp": {
        "LONG": "momentum_flag_break_long",
        "SHORT": "momentum_flag_break_short",
    },
    "liquidity_grab": {
        "LONG": "liquidity_grab_long",
        "SHORT": "liquidity_grab_short",
    },
    "session_liquidity_sweep": {
        "LONG": "liq_sweep_bull",
        "SHORT": "liq_sweep_bear",
    },
    "opening_range": {"LONG": "orb_long", "SHORT": "orb_short"},
}


_CANONICAL_SETUP_TYPES = frozenset(
    t for t in GEMINI_GOLD_SETUP_TYPES if t not in _LEGACY_BY_DIRECTION
)


def is_approved_setup_type(setup_type: Optional[str]) -> bool:
    """True when setup_type is a known directional id (required for TAKE)."""
    raw = (setup_type or "").strip().lower()
    if not raw or raw in ("unknown", "none", "skip", "skip_no_setup"):
        return False
    return raw in _CANONICAL_SETUP_TYPES


def normalize_setup_type(
    setup_type: Optional[str],
    direction: Optional[str] = None,
) -> Optional[str]:
    """Map legacy/generic setup ids to directional Gold AI-style ids."""
    raw = (setup_type or "").strip().lower()
    if not raw:
        return None
    if raw in GEMINI_GOLD_SETUP_TYPES and raw not in _LEGACY_BY_DIRECTION:
        return raw
    d = (direction or "").strip().upper()
    if raw in _LEGACY_BY_DIRECTION and d in ("LONG", "SHORT"):
        return _LEGACY_BY_DIRECTION[raw][d]
    if raw.endswith("_long") or raw.endswith("_bull"):
        return raw
    if raw.endswith("_short") or raw.endswith("_bear"):
        return raw
    if raw == "orb_long" or raw == "orb_short":
        return raw
    return raw


def setup_vocabulary_prompt_block() -> str:
    """Compact setup_type list for the Gemini vision prompt."""
    return (
        "setup_type (required on TAKE — use exact id):\n"
        "  Zone/retrace: fvg_retrace_bull, fvg_retrace_bear, ifvg_bull, ifvg_bear, "
        "ob_bull, ob_bear, breaker_bull, breaker_bear, "
        "momentum_ema_bounce_long, momentum_ema_bounce_short\n"
        "  Sweeps: liq_sweep_bull, liq_sweep_bear, sweep_pdh, sweep_pdl, "
        "eqh_sweep_bear, eql_sweep_bull, asian_sweep_bull, asian_sweep_bear, "
        "liquidity_grab_long, liquidity_grab_short\n"
        "  Momentum: momentum_flag_break_long, momentum_flag_break_short, "
        "disp_bull, disp_bear, sdp_bull, sdp_bear\n"
        "  ORB: orb_long, orb_short\n"
        "Charts may show shaded FVG/IFVG/OB zones — align setup_type with the "
        "zone you are trading (bull FVG → fvg_retrace_bull, filled inverted gap → ifvg_*)."
    )
