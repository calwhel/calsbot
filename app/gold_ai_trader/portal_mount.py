"""Mount Gold AI Trader into the Strategy Portal (additive only)."""
from __future__ import annotations

import asyncio
import logging

logger = logging.getLogger(__name__)


def _maybe_reset_daily_credits_on_startup() -> None:
    from app.gold_ai_trader.guardrails import maybe_reset_daily_claude_credits

    try:
        if maybe_reset_daily_claude_credits():
            logger.info("[gold-ai-trader] startup daily credits reset applied")
    except Exception as exc:
        logger.warning("[gold-ai-trader] startup credits reset failed: %s", exc)


def mount_gold_ai_trader_portal(app) -> None:
    """Register routes + isolated background task. Safe to call multiple times."""
    from app.gold_ai_trader.routes import router
    from app.gold_ai_trader.schema import ensure_gold_ai_trader_schema

    ensure_gold_ai_trader_schema()
    _maybe_reset_daily_credits_on_startup()
    app.include_router(router)
    logger.info("[gold-ai-trader] routes mounted at /gold-ai-trader")

    @app.on_event("startup")
    async def _gold_ai_trader_startup():
        try:
            from app.gold_ai_trader.loop import maybe_start_background_loop

            asyncio.create_task(maybe_start_background_loop())
        except Exception as e:
            logger.warning("[gold-ai-trader] startup failed (non-fatal): %s", e)
