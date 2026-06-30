"""Mount Gemini Gold Trader into the Strategy Portal."""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def mount_gemini_gold_trader_portal(app) -> None:
    from app.gemini_gold_trader.routes import router

    app.include_router(router)
    logger.info("[gemini-gold] routes mounted at /gemini-gold-trader")

    try:
        from app.gemini_gold_trader.schema import ensure_gemini_gold_trader_schema

        ensure_gemini_gold_trader_schema()
    except Exception as exc:
        logger.warning("[gemini-gold] schema startup skipped (routes still live): %s", exc)
