"""Pre-scan XAUUSD kline refresh — reduce stale_klines blocks."""
from __future__ import annotations

import logging
import os
from typing import Optional

from app.gold_ai_trader.config import ASSET_CLASS, SYMBOL
from app.gold_ai_trader.klines import is_ctrader_kline_source
from app.services.kline_staleness import newest_bar_age_s
from app.services.tradfi_prices import (
    clear_metal_kline_cache,
    get_klines,
    get_metal_kline_source,
    sweep_stale_metal_klines,
)

logger = logging.getLogger(__name__)

_SCORING_TF = "5m"
_SCORING_LIMIT = 60


def _refresh_age_s() -> float:
    try:
        return max(120.0, float(os.environ.get("GOLD_AI_KLINE_REFRESH_AGE_S", "300")))
    except (TypeError, ValueError):
        return 300.0


async def refresh_gold_scoring_klines(*, user_id: Optional[int] = None) -> dict:
    """
    Sweep stale metal caches and force-refresh 5m when last bar is too old.

    Returns a small summary dict for logging/diagnostics.
    """
    summary = {"swept": 0, "cleared": 0, "refreshed_5m": False, "bar_age_before": None}

    try:
        summary["swept"] = await sweep_stale_metal_klines(
            [SYMBOL], timeframes=["5m", "15m", "1h"],
        )
    except Exception as exc:
        logger.debug("[gold-ai] kline sweep: %s", exc)

    k5 = await get_klines(
        SYMBOL, ASSET_CLASS, _SCORING_TF, _SCORING_LIMIT, ctrader_user_id=user_id,
    ) or []
    bar_age = newest_bar_age_s(k5)
    summary["bar_age_before"] = round(bar_age, 1) if bar_age != float("inf") else None

    if bar_age <= _refresh_age_s():
        return summary

    cleared = clear_metal_kline_cache([SYMBOL])
    summary["cleared"] = cleared
    logger.info(
        "[gold-ai] 5m klines stale (bar_age=%.0fs > %.0fs) — cache cleared (%s keys)",
        bar_age,
        _refresh_age_s(),
        cleared,
    )

    k5 = await get_klines(
        SYMBOL, ASSET_CLASS, _SCORING_TF, _SCORING_LIMIT, ctrader_user_id=user_id,
    ) or []
    summary["refreshed_5m"] = True
    new_age = newest_bar_age_s(k5)
    src = get_metal_kline_source(SYMBOL, _SCORING_TF, _SCORING_LIMIT)
    logger.info(
        "[gold-ai] 5m refresh done bar_age=%.0fs→%.0fs source=%s bars=%s",
        bar_age,
        new_age if new_age != float("inf") else -1,
        src or "unknown",
        len(k5),
    )
    if not is_ctrader_kline_source(src):
        logger.warning(
            "[gold-ai] 5m refresh still non-cTrader source=%s — data gate may block",
            src,
        )
    return summary
