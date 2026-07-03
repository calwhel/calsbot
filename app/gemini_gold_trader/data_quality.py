"""cTrader-only market data gate for Gemini Gold vision scans."""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple

from app.gemini_gold_trader.config import ASSET_CLASS, SYMBOL
from app.gemini_gold_trader.klines import get_chart_klines, _is_ctrader_kline_source
from app.services.kline_staleness import newest_bar_age_s, stale_limit_s

logger = logging.getLogger(__name__)

SCORING_TIMEFRAME = "5m"
MIN_KLINE_BARS = 20
_CTRADER_PRICE_SOURCES = frozenset({"ctrader", "ctrader_kline_close"})


async def _resolve_ctrader_spot(
    *,
    user_id: Optional[int] = None,
) -> Tuple[Optional[float], Optional[str]]:
    from app.services.realtime_spot import _effective_max_age, _fetch_ctrader_on_demand, _read_ctrader

    sym = SYMBOL.upper()
    age_limit = _effective_max_age(sym, ASSET_CLASS)
    hit = _read_ctrader(sym, age_limit)
    if hit and hit[0] > 0:
        return float(hit[0]), "ctrader"

    try:
        from app.services.ctrader_price_feed import get_price

        px = get_price(sym)
        if px and px > 0:
            return float(px), "ctrader"
    except Exception:
        pass

    hit = await _fetch_ctrader_on_demand(sym, ASSET_CLASS, age_limit, user_id=user_id)
    if hit and hit[0] > 0 and (hit[1] or "").lower() == "ctrader":
        return float(hit[0]), "ctrader"
    return None, None


async def assess_gemini_market_data(
    *,
    user_id: Optional[int] = None,
) -> Dict[str, Any]:
    bars, meta = await get_chart_klines(SCORING_TIMEFRAME, 60, user_id=user_id)
    kline_source = (meta.get("source") or "").lower()
    kline_bars = len(bars or [])
    bar_age = newest_bar_age_s(bars) if bars else None
    snapshot_fresh = (
        bar_age is not None and float(bar_age) <= stale_limit_s(SCORING_TIMEFRAME)
        if bars
        else False
    )

    live_px, live_source = await _resolve_ctrader_spot(user_id=user_id)
    price_source = live_source or "unknown"
    spot_tick_cold = live_px is None

    if not live_px and bars and snapshot_fresh:
        try:
            live_px = float(bars[-1][4])
            price_source = "ctrader_kline_close"
            spot_tick_cold = True
        except (IndexError, TypeError, ValueError):
            pass

    price = live_px
    if not price and bars:
        try:
            price = float(bars[-1][4])
        except (IndexError, TypeError, ValueError):
            price = None

    klines_stale = not snapshot_fresh if bars else True
    stale_reason = meta.get("status") or "stale" if klines_stale else ""

    return {
        "price": price,
        "price_source": price_source,
        "live_source": live_source,
        "kline_source": kline_source,
        "kline_synthetic": bool(meta.get("synthetic")),
        "klines_stale": klines_stale,
        "stale_reason": stale_reason,
        "kline_bar_age_s": bar_age,
        "kline_bars": kline_bars,
        "spot_tick_cold": spot_tick_cold,
        "snapshot_fresh": snapshot_fresh,
        "chart_meta_5m": meta,
    }


def gemini_data_ok_for_scan(data: Dict[str, Any]) -> Tuple[bool, str]:
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
    if not _is_ctrader_kline_source(ks):
        return False, f"fallback_klines:{ks or 'missing'}"

    if int(data.get("kline_bars") or 0) < MIN_KLINE_BARS:
        return False, "insufficient_klines"

    if data.get("klines_stale"):
        reason = data.get("stale_reason") or "stale"
        return False, f"stale_klines:{reason}"

    snapshot_ok = bool(data.get("snapshot_fresh"))
    if data.get("spot_tick_cold") and not snapshot_ok:
        return False, "tick_cold"

    ps = (data.get("price_source") or "").lower()
    live_src = (data.get("live_source") or "").lower()
    if live_src not in _CTRADER_PRICE_SOURCES and ps != "ctrader_kline_close":
        return False, f"non_ctrader_price:{live_src or ps or 'unknown'}"

    return True, "ok"


def format_data_source(data: Dict[str, Any]) -> str:
    ps = data.get("live_source") or data.get("price_source") or "unknown"
    ks = data.get("kline_source") or "none"
    return f"price:{ps}/kline:{ks}"
