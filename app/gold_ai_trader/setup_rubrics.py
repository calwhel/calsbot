"""Setup-specific scoring rubrics for Claude context."""
from __future__ import annotations

from typing import List


def setup_rubric_block(setup_type: str) -> List[str]:
    """Short micro-rubric appended to Claude context per setup family."""
    t = (setup_type or "").lower()
    lines = ["=== SETUP RUBRIC (this trigger type) ==="]

    if t.startswith(("liq_sweep_", "sweep_pdh", "sweep_pdl", "asian_sweep_")):
        lines.extend([
            "Liquidity sweep / reclaim setup:",
            "- Floor 50–55%: reclaim held + displacement — one missing HTF factor OK.",
            "- 60%+: reclaim + disp + logical SL below sweep; TP at session edge or opposing liquidity.",
            "- 75%+: HTF aligned + at reclaim + strong disp + correct premium/discount side.",
        ])
    elif t.startswith(("ob_", "fvg_retrace_", "ifvg_", "breaker_")):
        lines.extend([
            "Zone retrace setup (OB / FVG / iFVG / breaker):",
            "- Floor 55%: price IN zone now + SL beyond zone invalidation.",
            "- 60%+: in zone + HTF aligned + TP at nearest session/HTF level.",
            "- 75%+: in zone + HTF + premium/discount correct + momentum not against.",
        ])
    elif t.startswith("disp_"):
        lines.extend([
            "Displacement / momentum setup:",
            "- Floor 50%: clear disp candle + direction with short-term momentum.",
            "- 60%+: HTF not opposing + entry near origin of displacement.",
            "- 75%+: HTF aligned + continuation structure + TP at next liquidity pool.",
        ])
    elif t.startswith(("eqh_sweep_", "eql_sweep_")):
        lines.extend([
            "Equal highs/lows sweep:",
            "- Floor 55%: sweep + reclaim at cluster + displacement confirmation.",
            "- 60%+: HTF aligned + TP at next cluster or session extreme.",
            "- 75%+: cluster reclaim + disp + P/D side correct.",
        ])
    else:
        lines.extend([
            "General ICT trigger:",
            "- 50%+: clear entry, defined stop at structure, TP at sensible level.",
            "- 60%+: 4+ confluence checks passed (see CONFLUENCE block).",
            "- 75%+: 5+ checks with HTF + location quality.",
        ])

    return lines
