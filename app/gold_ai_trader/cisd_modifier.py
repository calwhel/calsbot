"""CISD alignment as confidence modifier (not standalone trigger)."""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


async def assess_cisd_modifier(
    *,
    direction: str,
    http_client,
    cache: Dict,
    timeframe: str = "5m",
) -> Dict[str, Any]:
    """Return modifier -10..+10 when CISD confirms/opposes trade direction."""
    from app.services.strategy_ta import eval_fx_cisd

    d = (direction or "").upper()
    cisd_dir = "bullish" if d == "LONG" else "bearish"
    try:
        ok, detail = await eval_fx_cisd(
            {"direction": cisd_dir, "timeframe": timeframe},
            "XAUUSD",
            http_client,
            cache,
        )
    except Exception as exc:
        logger.debug("[gold-ai] CISD eval: %s", exc)
        return {"modifier": 0, "detail": f"CISD: eval failed ({exc})", "aligned": None}

    if ok:
        return {
            "modifier": 8,
            "detail": f"CISD aligned ({cisd_dir}): {detail}",
            "aligned": True,
        }
    # Opposing CISD recently fired — check opposite
    opp = "bearish" if cisd_dir == "bullish" else "bullish"
    try:
        ok_opp, detail_opp = await eval_fx_cisd(
            {"direction": opp, "timeframe": timeframe},
            "XAUUSD",
            http_client,
            cache,
        )
    except Exception:
        ok_opp = False
        detail_opp = ""
    if ok_opp:
        return {
            "modifier": -10,
            "detail": f"CISD opposes ({opp} fired): {detail_opp}",
            "aligned": False,
        }
    return {
        "modifier": 0,
        "detail": f"CISD: no confirming {cisd_dir} delivery flip ({detail[:80]})",
        "aligned": None,
    }


def build_cisd_block(cisd: Optional[Dict[str, Any]]) -> list:
    if not cisd:
        return []
    mod = cisd.get("modifier", 0)
    if mod == 0 and not cisd.get("detail"):
        return []
    sign = "+" if mod > 0 else ""
    lines = [
        "=== CISD (confidence modifier — NOT standalone trigger) ===",
        f"Suggested confidence adjustment: {sign}{mod}",
        cisd.get("detail", ""),
    ]
    return lines
