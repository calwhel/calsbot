"""Gold AI kline helpers — cTrader-first, no Coinbase daily fallback."""
from __future__ import annotations

import logging
from typing import List, Optional

from app.gold_ai_trader.config import ASSET_CLASS, SYMBOL
from app.services.kline_staleness import newest_bar_age_s
from app.services.tradfi_prices import get_klines, get_metal_kline_source

logger = logging.getLogger(__name__)

_CTRADER_KLINE_SOURCES = frozenset({"ctrader", "ctrader-user", "ctrader-cache"})
_GOLD_SCORING_K5_LIMIT = 60


def is_ctrader_kline_source(source: Optional[str]) -> bool:
    return (source or "").lower() in _CTRADER_KLINE_SOURCES


def synthesize_gold_scoring_k5(k5: List[list]) -> List[list]:
    """
    Defense-in-depth: roll the 5m forming bar from live spot before staleness gate.

    ctrader get_klines already synthesizes on return; this covers tradfi cache paths.
    """
    if not k5:
        return k5
    try:
        from app.services.ctrader_price_feed import (
            apply_live_spot_to_klines,
            is_live as ct_is_live,
        )

        if not ct_is_live():
            return k5
        before_age = newest_bar_age_s(k5)
        out = apply_live_spot_to_klines(
            k5, SYMBOL, "5m", _GOLD_SCORING_K5_LIMIT, log_remote=False,
        )
        after_age = newest_bar_age_s(out)
        logger.info(
            "[gold-ai] 5m scoring kline synthesis bar_age=%.0fs→%.0fs bars=%d",
            before_age if before_age != float("inf") else -1.0,
            after_age if after_age != float("inf") else -1.0,
            len(out),
        )
        return out
    except Exception as exc:
        logger.debug("[gold-ai] scoring kline synthesis skipped: %s", exc)
        return k5


async def get_gold_ai_klines(
    timeframe: str,
    limit: int,
    *,
    user_id: Optional[int] = None,
) -> List[list]:
    """
    Fetch klines for Gold AI. Daily bars reject non-cTrader sources so PDH/PDL
    fall back to 1h aggregation instead of stale Coinbase PAXG.
    """
    rows = await get_klines(
        SYMBOL,
        ASSET_CLASS,
        timeframe,
        limit,
        ctrader_user_id=user_id,
    ) or []
    if timeframe != "1d" or not rows:
        return rows

    src = get_metal_kline_source(SYMBOL, timeframe, limit)
    if is_ctrader_kline_source(src):
        return rows

    return []
