"""Fresh-spot refresh and full validator re-check immediately before broker fire."""
from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional, Tuple

from app.gold_ai_trader.call_gates import atr_from_klines
from app.gold_ai_trader.config import ASSET_CLASS, SYMBOL
from app.gold_ai_trader.decision_validator import validate_take_decision
from app.services.tradfi_prices import confirm_entry_price, get_klines

logger = logging.getLogger(__name__)


def fire_time_revalidate_enabled() -> bool:
    raw = os.environ.get("GOLD_AI_FIRE_TIME_REVALIDATE", "true")
    return raw.strip().lower() in ("1", "true", "yes", "on")


async def refresh_gold_spot(
    spot_hint: float,
    *,
    user_id: Optional[int] = None,
) -> Tuple[Optional[float], str]:
    """Re-fetch live spot; reject if hint drifts beyond metal drift limit."""
    try:
        proposed = float(spot_hint or 0.0)
    except (TypeError, ValueError):
        proposed = 0.0
    if proposed <= 0:
        from app.services.tradfi_prices import get_price_fresh

        live = await get_price_fresh(SYMBOL, ASSET_CLASS, paper_ok=False, user_id=user_id)
        if live and live > 0:
            return float(live), "fresh_spot_no_hint"
        return None, "no_spot_hint"
    return await confirm_entry_price(
        SYMBOL,
        ASSET_CLASS,
        proposed,
        paper_ok=False,
        user_id=user_id,
    )


async def refresh_spot_after_claude(
    spot_hint: float,
    *,
    user_id: Optional[int] = None,
) -> float:
    """Refresh scan-cycle spot after Claude returns (before validator)."""
    try:
        proposed = float(spot_hint or 0.0)
    except (TypeError, ValueError):
        return spot_hint
    if proposed <= 0:
        return spot_hint
    confirmed, reason = await refresh_gold_spot(proposed, user_id=user_id)
    if confirmed is None or confirmed <= 0:
        logger.debug("[gold-ai] post-claude spot refresh skipped: %s", reason)
        return proposed
    if abs(confirmed - proposed) > 0.01:
        logger.info(
            "[gold-ai] post-claude spot refresh %.4f -> %.4f (%s)",
            proposed,
            float(confirmed),
            reason,
        )
    return float(confirmed)


async def revalidate_before_fire(
    *,
    decision: Dict[str, Any],
    setup_type: str,
    candidate_direction: str,
    setup_detail: str = "",
    user_id: Optional[int] = None,
    atr: Optional[float] = None,
    key_levels: Optional[List[float]] = None,
) -> Tuple[bool, str, Dict[str, Any]]:
    """
    At broker submit: refresh spot and re-run validate_take_decision.
    Returns (ok, reason, decision_copy).
    """
    if not fire_time_revalidate_enabled():
        return True, "fire_revalidate_disabled", decision

    try:
        entry_hint = float(decision.get("entry") or 0.0)
    except (TypeError, ValueError):
        entry_hint = 0.0

    spot, spot_reason = await refresh_gold_spot(
        entry_hint if entry_hint > 0 else 0.0,
        user_id=user_id,
    )
    if spot is None or spot <= 0:
        reason = f"fire_time:{spot_reason}"
        logger.warning(
            "[gold-ai] fire_time blocked setup=%s reason=%s",
            setup_type,
            reason,
        )
        return False, reason, decision

    use_atr = float(atr or 0.0)
    if use_atr <= 0:
        k5 = await get_klines(SYMBOL, ASSET_CLASS, "5m", 60) or []
        use_atr = float(atr_from_klines(k5) or 0.0)

    val_ok, val_reason, updated = validate_take_decision(
        decision,
        candidate_direction=candidate_direction,
        spot=float(spot),
        atr=use_atr,
        setup_detail=setup_detail,
        key_levels=key_levels,
    )
    if not val_ok:
        reason = f"fire_time:{val_reason}"
        logger.warning(
            "[gold-ai] fire_time blocked setup=%s reason=%s spot=%.4f",
            setup_type,
            reason,
            float(spot),
        )
        return False, reason, updated

    updated["fire_time_spot"] = round(float(spot), 4)
    updated["fire_time_note"] = f"revalidated({spot_reason})"
    return True, "fire_time:ok", updated
