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
            "- Floor 50–55%: reclaim held + displacement ≥0.8×ATR — one missing HTF factor OK if R:R works.",
            "- 60%+: reclaim + disp + 2:1 R:R; HTF aligned OR visible structure shift on 5m.",
            "- 75%+: HTF aligned + at reclaim + strong disp + correct premium/discount side.",
        ])
    elif t.startswith(("ob_", "fvg_retrace_", "ifvg_", "breaker_")):
        lines.extend([
            "Zone retrace setup (OB / FVG / iFVG / breaker):",
            "- Floor 55%: price IN zone now + logical invalidation within 1× ATR.",
            "- 60%+: in zone + HTF aligned + 2:1 R:R to nearest level.",
            "- 75%+: in zone + HTF + premium/discount correct + momentum not against.",
        ])
    elif t.startswith("disp_"):
        lines.extend([
            "Displacement / momentum setup:",
            "- Floor 50%: clear disp candle + direction with short-term momentum; do not require every modifier.",
            "- 60%+: HTF not opposing + not chasing — entry near origin of displacement.",
            "- 75%+: HTF aligned + continuation structure + 2:1 R:R.",
        ])
    elif t.startswith(("eqh_sweep_", "eql_sweep_")):
        lines.extend([
            "Equal highs/lows sweep:",
            "- Floor 55%: sweep + reclaim at cluster + displacement confirmation.",
            "- 60%+: HTF aligned + 2:1 R:R.",
            "- 75%+: cluster reclaim + disp + P/D side correct.",
        ])
    else:
        lines.extend([
            "General ICT trigger:",
            "- 50%+: clear entry, defined stop, ≥2:1 R:R, plausible session narrative.",
            "- 60%+: 4+ confluence checks passed (see CONFLUENCE block).",
            "- 75%+: 5+ checks with HTF + location quality.",
        ])

    return lines
