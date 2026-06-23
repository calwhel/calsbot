"""Gold AI market-data quality gate — cTrader-only, no fallback/stale scoring."""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple

from app.gold_ai_trader.config import ASSET_CLASS, SYMBOL
from app.services.kline_staleness import check_cached_klines_stale
from app.services.tradfi_prices import (
    get_klines,
    get_metal_kline_source,
    is_metal_kline_synthetic,
)

logger = logging.getLogger(__name__)

# Primary scoring timeframe — must match loop dedupe / scanner TA.
SCORING_TIMEFRAME = "5m"
SCORING_KLINE_LIMIT = 60
MIN_KLINE_BARS = 20

_CTRADER_PRICE_SOURCES = frozenset({"ctrader"})
_CTRADER_KLINE_SOURCES = frozenset({"ctrader", "ctrader-user", "ctrader-cache"})


async def assess_gold_market_data(
    *,
    user_id: Optional[int] = None,
) -> Dict[str, Any]:
    """Resolve live price + kline provenance for Gold AI Claude gating."""
    sym = SYMBOL.upper()
    live_px: Optional[float] = None
    live_source: Optional[str] = None
    bid = ask = None

    try:
        from app.services.ctrader_price_feed import (
            ctrader_spot_ready,
            get_bid_ask,
            get_price as ct_get_price,
        )

        if ctrader_spot_ready(sym):
            tick = get_bid_ask(sym)
            if tick:
                bid, ask = tick
                live_px = round((bid + ask) / 2.0, 6)
                live_source = "ctrader"
            else:
                px = ct_get_price(sym)
                if px and px > 0:
                    live_px = float(px)
                    live_source = "ctrader"
    except Exception as exc:
        logger.debug("[gold-ai] cTrader spot probe: %s", exc)

    price_source = live_source or "unknown"
    if not live_px:
        try:
            from app.services.realtime_spot import read_fresh_cached

            hit = read_fresh_cached(sym, ASSET_CLASS)
            if hit and hit[0] > 0:
                live_px = float(hit[0])
                price_source = str(hit[1] or "unknown").lower()
        except Exception:
            pass

    k5 = await get_klines(
        SYMBOL, ASSET_CLASS, SCORING_TIMEFRAME, SCORING_KLINE_LIMIT
    ) or []
    kline_source = get_metal_kline_source(sym, SCORING_TIMEFRAME, SCORING_KLINE_LIMIT)
    kline_synthetic = is_metal_kline_synthetic(
        sym, SCORING_TIMEFRAME, SCORING_KLINE_LIMIT
    )

    klines_stale = False
    stale_reason = ""
    if k5:
        klines_stale, stale_reason = await check_cached_klines_stale(
            sym, k5, SCORING_TIMEFRAME
        )

    price = live_px
    if not price and k5:
        try:
            price = float(k5[-1][4])
        except (IndexError, TypeError, ValueError):
            price = None

    return {
        "price": price,
        "price_source": price_source,
        "live_source": live_source,
        "kline_source": kline_source,
        "kline_synthetic": kline_synthetic,
        "klines_stale": klines_stale,
        "stale_reason": stale_reason,
        "kline_bars": len(k5),
        "bid": bid,
        "ask": ask,
        "user_id": user_id,
    }


def gold_data_ok_for_claude(data: Dict[str, Any]) -> Tuple[bool, str]:
    """
    Require clean cTrader spot + cTrader klines before Claude scoring.

    Mirrors the forex Claude confirm stale-data fail-safe: bad data → skip,
    never score or fire on fallback/stale feeds.
    """
    if not data:
        return False, "no_market_data"

    try:
        price = float(data.get("price") or 0)
    except (TypeError, ValueError):
        price = 0.0
    if price <= 0:
        return False, "no_price"

    if data.get("kline_synthetic"):
        return False, "synthetic_klines"

    ks = (data.get("kline_source") or "").lower()
    if ks not in _CTRADER_KLINE_SOURCES:
        return False, f"fallback_klines:{ks or 'missing'}"

    if int(data.get("kline_bars") or 0) < MIN_KLINE_BARS:
        return False, "insufficient_klines"

    if data.get("klines_stale"):
        reason = data.get("stale_reason") or "stale"
        return False, f"stale_klines:{reason}"

    live_src = (data.get("live_source") or "").lower()
    if live_src not in _CTRADER_PRICE_SOURCES:
        ps = (data.get("price_source") or "").lower()
        return False, f"non_ctrader_price:{live_src or ps or 'unknown'}"

    return True, "ok"


def format_data_source(data: Dict[str, Any]) -> str:
    """Compact source tag for decision logs."""
    ps = data.get("live_source") or data.get("price_source") or "unknown"
    ks = data.get("kline_source") or "none"
    return f"price:{ps}/kline:{ks}"
