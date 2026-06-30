"""Pre-fire spot revalidation for Gemini Gold trades."""
from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional, Tuple

from app.gemini_gold_trader.config import ASSET_CLASS, SYMBOL
from app.gemini_gold_trader.validator import validate_take_decision
from app.gemini_gold_trader.config import GeminiGoldRuntimeConfig

logger = logging.getLogger(__name__)


def fire_time_revalidate_enabled() -> bool:
    raw = os.environ.get("GEMINI_GOLD_FIRE_TIME_REVALIDATE", "true")
    return raw.strip().lower() in ("1", "true", "yes", "on")


async def refresh_spot(
    spot_hint: float,
    *,
    user_id: Optional[int] = None,
) -> Tuple[Optional[float], str]:
    from app.services.tradfi_prices import confirm_entry_price, get_price_fresh

    try:
        proposed = float(spot_hint or 0.0)
    except (TypeError, ValueError):
        proposed = 0.0
    if proposed <= 0:
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


async def revalidate_before_fire(
    *,
    decision: Dict[str, Any],
    cfg: GeminiGoldRuntimeConfig,
    user_id: Optional[int] = None,
    spot_hint: float,
    decision_id: Optional[int] = None,
) -> Tuple[bool, str, Dict[str, Any]]:
    if not fire_time_revalidate_enabled():
        ok, reason, d = validate_take_decision(decision, cfg=cfg, spot=spot_hint)
        if not ok:
            return ok, reason, d
    else:
        confirmed, refresh_reason = await refresh_spot(spot_hint, user_id=user_id)
        if confirmed is None or confirmed <= 0:
            logger.debug("[gemini-gold] fire-time spot refresh failed: %s", refresh_reason)
            confirmed = spot_hint

        ok, reason, d = validate_take_decision(decision, cfg=cfg, spot=float(confirmed))
        if not ok:
            return False, reason, d

        if abs(confirmed - spot_hint) > 0.01:
            logger.info(
                "[gemini-gold] fire-time spot refresh %.4f -> %.4f (%s)",
                spot_hint,
                float(confirmed),
                refresh_reason,
            )
        decision = d

    if user_id and cfg is not None:
        from app.gemini_gold_trader.db_thread import run_with_db
        from app.gemini_gold_trader.guardrails import check_can_execute

        can_exec, exec_reason = await run_with_db(check_can_execute, cfg, user_id)
        if not can_exec:
            logger.warning(
                "[gemini-gold] fire-time cap blocked decision_id=%s reason=%s",
                decision_id,
                exec_reason,
            )
            return False, f"fire_time:{exec_reason}", decision

    return True, "ok", decision
