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


async def _resolve_ctrader_spot(
    sym: str,
    *,
    user_id: Optional[int] = None,
) -> Tuple[Optional[float], Optional[str], Optional[float], Optional[float]]:
    """
    Broker-only XAUUSD spot: in-memory/Postgres cTrader tick, then on-demand
    cTrader trendbar close. Never Coinbase/Kraken fallbacks.
    """
    from app.services.realtime_spot import (
        _effective_max_age,
        _fetch_ctrader_on_demand,
        _read_ctrader,
    )

    age_limit = _effective_max_age(sym, ASSET_CLASS)
    bid = ask = None

    hit = _read_ctrader(sym, age_limit)
    if hit and hit[0] > 0:
        try:
            from app.services.ctrader_price_feed import get_bid_ask

            tick = get_bid_ask(sym)
            if tick:
                bid, ask = tick
        except Exception:
            pass
        return float(hit[0]), "ctrader", bid, ask

    try:
        from app.services.ctrader_price_feed import get_bid_ask, get_price

        tick = get_bid_ask(sym)
        if tick:
            bid, ask = tick
            mid = round((tick[0] + tick[1]) / 2.0, 6)
            return mid, "ctrader", bid, ask
        px = get_price(sym)
        if px and px > 0:
            return float(px), "ctrader", bid, ask
    except Exception as exc:
        logger.debug("[gold-ai] cTrader spot cache probe: %s", exc)

    hit = await _fetch_ctrader_on_demand(
        sym, ASSET_CLASS, age_limit, user_id=user_id,
    )
    if hit and hit[0] > 0 and (hit[1] or "").lower() == "ctrader":
        return float(hit[0]), "ctrader", bid, ask

    return None, None, bid, ask


async def assess_gold_market_data(
    *,
    user_id: Optional[int] = None,
) -> Dict[str, Any]:
    """Resolve live price + kline provenance for Gold AI Claude gating."""
    sym = SYMBOL.upper()
    live_px, live_source, bid, ask = await _resolve_ctrader_spot(sym, user_id=user_id)
    price_source = live_source or "unknown"

    k5 = await get_klines(
        SYMBOL, ASSET_CLASS, SCORING_TIMEFRAME, SCORING_KLINE_LIMIT
    ) or []
    kline_source = get_metal_kline_source(sym, SCORING_TIMEFRAME, SCORING_KLINE_LIMIT)
    kline_synthetic = is_metal_kline_synthetic(
        sym, SCORING_TIMEFRAME, SCORING_KLINE_LIMIT
    )

    # Last resort: cTrader kline close when tick socket is cold but trendbars flow.
    if not live_px and k5 and (kline_source or "").lower() in _CTRADER_KLINE_SOURCES:
        try:
            live_px = float(k5[-1][4])
            live_source = "ctrader"
            price_source = "ctrader"
            logger.info(
                "[gold-ai] spot from cTrader %s close (tick cold) price=%.4f",
                SCORING_TIMEFRAME,
                live_px,
            )
        except (IndexError, TypeError, ValueError):
            pass

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
