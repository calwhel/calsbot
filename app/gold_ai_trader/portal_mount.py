"""Mount Gold AI Trader into the Strategy Portal (additive only)."""
from __future__ import annotations

import asyncio
import logging
import os

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

    app.include_router(router)
    logger.info("[gold-ai-trader] routes mounted at /gold-ai-trader")

    try:
        from app.gold_ai_trader.schema import ensure_gold_ai_trader_schema

        ensure_gold_ai_trader_schema()
        _maybe_reset_daily_credits_on_startup()
    except Exception as exc:
        hard_fail = os.environ.get(
            "GOLD_AI_SCHEMA_STARTUP_HARD_FAIL", ""
        ).strip().lower() in ("1", "true", "yes", "on")
        if hard_fail:
            logger.error(
                "[gold-ai-trader] schema startup hard-fail enabled; refusing startup: %s",
                exc,
            )
            raise
        logger.warning(
            "[gold-ai-trader] schema/credits startup skipped (routes still live): %s",
            exc,
        )

    @app.on_event("startup")
    async def _gold_ai_trader_startup():
        try:
            from app.gold_ai_trader.loop import maybe_start_background_loop

            asyncio.create_task(maybe_start_background_loop())
        except Exception as e:
            logger.warning("[gold-ai-trader] startup failed (non-fatal): %s", e)
