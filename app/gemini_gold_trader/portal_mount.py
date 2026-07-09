"""Mount Gemini Gold Trader into the Strategy Portal."""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def mount_gemini_gold_trader_portal(app) -> None:
    from app.gemini_gold_trader.routes import router

    app.include_router(router)
    logger.info("[gemini-gold] routes mounted at /gemini-gold-trader")

    @app.on_event("startup")
    async def _gemini_gold_schema_startup():
        """Run schema in startup background (after _ensure_tables on Railway)."""
        import asyncio

        async def _run():
            await asyncio.sleep(2)
            try:
                from app.gemini_gold_trader.schema import ensure_gemini_gold_trader_schema

                ensure_gemini_gold_trader_schema(force=True)
                logger.info("[gemini-gold] schema startup complete")
            except Exception as exc:
                logger.error("[gemini-gold] schema startup failed: %s", exc, exc_info=True)

        asyncio.create_task(_run())
