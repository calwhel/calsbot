"""Gold AI kline helpers — cTrader-first, no Coinbase daily fallback."""
from __future__ import annotations

from typing import List, Optional

from app.gold_ai_trader.config import ASSET_CLASS, SYMBOL
from app.services.tradfi_prices import get_klines, get_metal_kline_source

_CTRADER_KLINE_SOURCES = frozenset({"ctrader", "ctrader-user", "ctrader-cache"})


def is_ctrader_kline_source(source: Optional[str]) -> bool:
    return (source or "").lower() in _CTRADER_KLINE_SOURCES


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
