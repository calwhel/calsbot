"""Engine structure quality score (0–100) for Claude context — not a fire threshold."""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

from app.gold_ai_trader.context_history import parse_zone_from_detail


def compute_structure_score(
    *,
    candidate_type: str,
    direction: str,
    detail: str,
    quality_atr: float,
    htf_align: str,
    raw: Optional[Dict[str, Any]] = None,
    in_zone: bool = False,
) -> Tuple[int, str]:
    """Return (score 0–100, one-line breakdown)."""
    score = 45
    parts = []

    align = (htf_align or "").lower()
    if "aligned_bull" in align or "aligned_bear" in align:
        score += 18
        parts.append("HTF+18")
    elif "counter_htf" in align:
        score -= 22
        parts.append("counter-HTF-22")
    elif align == "htf_mixed_allowed":
        score += 4
        parts.append("HTF-mixed+4")

    q = max(0.0, float(quality_atr or 0))
    q_pts = min(22, int(q * 12))
    score += q_pts
    if q_pts:
        parts.append(f"body{q_pts}")

    zone = parse_zone_from_detail(detail)
    if zone or in_zone:
        score += 12
        parts.append("in-zone+12")

    smt = (raw or {}).get("smt") if raw else None
    if isinstance(smt, dict):
        mod = int(smt.get("modifier") or 0)
        if mod:
            score += mod
            parts.append(f"SMT{mod:+d}")

    # Setup-type priors (structure hierarchy, not indicators)
    priors = {
        "sweep_pdh": 8, "sweep_pdl": 8,
        "sdp_bull": 7, "sdp_bear": 7,
        "liq_sweep_bull": 7, "liq_sweep_bear": 7,
        "eqh_sweep_bear": 7, "eql_sweep_bull": 7,
        "ob_bull": 6, "ob_bear": 6,
        "breaker_bull": 6, "breaker_bear": 6,
        "ifvg_bull": 5, "ifvg_bear": 5,
        "fvg_retrace_bull": 4, "fvg_retrace_bear": 4,
        "disp_bull": 4, "disp_bear": 4,
        "judas_bull": 7, "judas_bear": 7,
        "asian_sweep_bull": 7, "asian_sweep_bear": 7,
        "momentum_ema_bounce_long": 6, "momentum_ema_bounce_short": 6,
        "momentum_flag_break_long": 6, "momentum_flag_break_short": 6,
        "liquidity_grab_long": 7, "liquidity_grab_short": 7,
    }
    prior = priors.get(candidate_type, 0)
    if prior:
        score += prior
        parts.append(f"type+{prior}")

    score = max(0, min(100, score))
    breakdown = ", ".join(parts) if parts else "baseline"
    return score, f"Structure score {score}/100 ({breakdown})"
