"""Mount Gold AI Trader into the Strategy Portal (additive only)."""
from __future__ import annotations

import asyncio
import logging
import os

logger = logging.getLogger(__name__)


def _maybe_reset_daily_credits_on_startup() -> None:
    """One-shot reset when GOLD_AI_TRADER_RESET_DAILY_CREDITS=1 (remove env after deploy)."""
    if os.environ.get("GOLD_AI_TRADER_RESET_DAILY_CREDITS", "").strip().lower() not in (
        "1",
        "true",
        "yes",
        "on",
    ):
        return
    from app.database import SessionLocal
    from app.gold_ai_trader.guardrails import reset_daily_claude_credits

    db = SessionLocal()
    try:
        reset_at = reset_daily_claude_credits(db)
        logger.info(
            "[gold-ai-trader] startup credits reset (GOLD_AI_TRADER_RESET_DAILY_CREDITS): %s",
            reset_at.isoformat(),
        )
    except Exception as exc:
        logger.warning("[gold-ai-trader] startup credits reset failed: %s", exc)
    finally:
        db.close()


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
