"""Gold AI kline helpers — cTrader-first, no Coinbase daily fallback."""
from __future__ import annotations

import logging
from typing import List, Optional

from app.gold_ai_trader.config import ASSET_CLASS, SYMBOL
from app.services.kline_staleness import newest_bar_age_s, stale_limit_s
from app.services.tradfi_prices import get_klines, get_metal_kline_source

logger = logging.getLogger(__name__)

_CTRADER_KLINE_SOURCES = frozenset({"ctrader", "ctrader-user", "ctrader-cache"})
_GOLD_SCORING_K5_LIMIT = 60


def is_ctrader_kline_source(source: Optional[str]) -> bool:
    return (source or "").lower() in _CTRADER_KLINE_SOURCES


def synthesize_gold_scoring_klines(
    rows: List[list],
    timeframe: str,
    *,
    limit: int,
    symbol: str = SYMBOL,
) -> List[list]:
    """
    Defense-in-depth: roll the forming bar from fresh cTrader spot before staleness gate.

    ctrader get_klines already synthesizes on return; this covers tradfi cache paths
    and portal gunicorn workers that read Postgres ticks without a local feed socket.
    """
    if not rows:
        return rows
    try:
        from app.services.ctrader_price_feed import (
            apply_live_spot_to_klines,
            ctrader_spot_ready,
        )

        if not ctrader_spot_ready(symbol):
            return rows
        before_age = newest_bar_age_s(rows)
        out = apply_live_spot_to_klines(
            rows, symbol, timeframe, limit, log_remote=False,
        )
        after_age = newest_bar_age_s(out)
        if before_age != after_age or (
            before_age != float("inf") and before_age > stale_limit_s(timeframe)
        ):
            logger.info(
                "[gold-ai] %s scoring kline synthesis bar_age=%.0fs→%.0fs bars=%d",
                timeframe,
                before_age if before_age != float("inf") else -1.0,
                after_age if after_age != float("inf") else -1.0,
                len(out),
            )
        return out
    except Exception as exc:
        logger.debug("[gold-ai] scoring kline synthesis skipped (%s): %s", timeframe, exc)
        return rows


def synthesize_gold_scoring_k5(k5: List[list]) -> List[list]:
    """Roll 5m forming bar from fresh cTrader spot (portal worker defense-in-depth)."""
    return synthesize_gold_scoring_klines(
        k5, "5m", limit=_GOLD_SCORING_K5_LIMIT,
    )


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
    if timeframe in ("5m", "15m") and rows:
        rows = synthesize_gold_scoring_klines(rows, timeframe, limit=limit)
    if timeframe != "1d" or not rows:
        return rows

    src = get_metal_kline_source(SYMBOL, timeframe, limit)
    if is_ctrader_kline_source(src):
        return rows

    return []
