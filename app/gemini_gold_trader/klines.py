"""Kline access for Gemini Vision charts — tradfi chain + postgres snapshot fallback."""
from __future__ import annotations

import logging
import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from app.gemini_gold_trader.config import ASSET_CLASS, SYMBOL
from app.services.kline_staleness import newest_bar_age_s, stale_limit_s

logger = logging.getLogger(__name__)

_CTRADER_KLINE_SOURCES = frozenset({"ctrader", "ctrader-user", "ctrader-cache"})


def _min_bars() -> int:
    try:
        return max(10, int(os.environ.get("GEMINI_GOLD_KLINE_MIN_BARS", "20")))
    except (TypeError, ValueError):
        return 20


def _is_ctrader_kline_source(source: Optional[str]) -> bool:
    return (source or "").lower() in _CTRADER_KLINE_SOURCES


def _synthesize_forming_bar(bars: List[List[float]], timeframe: str, limit: int) -> List[List[float]]:
    """Roll forming 5m/15m bar from live cTrader spot when available."""
    if not bars or timeframe not in ("5m", "15m"):
        return bars
    try:
        from app.services.ctrader_price_feed import apply_live_spot_to_klines, ctrader_spot_ready

        if not ctrader_spot_ready(SYMBOL):
            return bars
        return apply_live_spot_to_klines(bars, SYMBOL, timeframe, limit, log_remote=False)
    except Exception as exc:
        logger.debug("[gemini-gold] kline synthesis skipped (%s): %s", timeframe, exc)
        return bars


def _bars_fresh(bars: List[List[float]], timeframe: str) -> bool:
    if len(bars) < _min_bars():
        return False
    return newest_bar_age_s(bars) <= stale_limit_s(timeframe)


def _try_postgres_ctrader_snapshot(
    timeframe: str,
    limit: int,
    *,
    tradfi_source: str,
    tradfi_bars: int,
) -> Tuple[List[List[float]], bool]:
    from app.services.kline_snapshot_store import get_klines as get_snapshot_klines

    snap_rows = get_snapshot_klines(
        SYMBOL,
        timeframe,
        limit,
        source="ctrader",
    )
    if not snap_rows:
        return [], False
    snap_rows = _synthesize_forming_bar(snap_rows, timeframe, limit)
    if not _bars_fresh(snap_rows, timeframe):
        return snap_rows, False
    logger.info(
        "[gemini-gold] %s from postgres ctrader snapshot "
        "(tradfi had %s, %d bars) → %d bars bar_age=%.0fs",
        timeframe,
        tradfi_source or "missing",
        tradfi_bars,
        len(snap_rows),
        newest_bar_age_s(snap_rows),
    )
    return snap_rows, True


async def get_chart_klines(
    timeframe: str,
    limit: int,
    *,
    user_id: Optional[int] = None,
) -> Tuple[List[List[float]], Dict]:
    """Return (bars, meta) for chart rendering."""
    from app.services.tradfi_prices import get_klines, get_metal_kline_source

    meta: Dict = {
        "timeframe": timeframe,
        "source": None,
        "stale_limit_s": stale_limit_s(timeframe),
        "fetched_at": datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
    }
    bars: List[List[float]] = []
    tradfi_source = ""

    try:
        bars = await get_klines(
            SYMBOL,
            ASSET_CLASS,
            timeframe,
            limit,
            ctrader_user_id=user_id,
        ) or []
        tradfi_source = (get_metal_kline_source(SYMBOL, timeframe, limit) or "tradfi").lower()
        meta["source"] = tradfi_source
    except Exception as exc:
        logger.warning("[gemini-gold] tradfi klines failed %s %s: %s", SYMBOL, timeframe, exc)
        meta["error"] = str(exc)

    if bars:
        bars = _synthesize_forming_bar(bars, timeframe, limit)

    if _is_ctrader_kline_source(tradfi_source) and _bars_fresh(bars, timeframe):
        meta["bars"] = len(bars)
        meta["status"] = "ok"
        meta["bar_age_s"] = newest_bar_age_s(bars)
        meta["last_close"] = float(bars[-1][4]) if len(bars[-1]) > 4 else None
        return bars, meta

    snap_rows, ok = _try_postgres_ctrader_snapshot(
        timeframe,
        limit,
        tradfi_source=tradfi_source,
        tradfi_bars=len(bars),
    )
    if ok:
        meta["source"] = "ctrader"
        meta["bars"] = len(snap_rows)
        meta["status"] = "ok"
        meta["bar_age_s"] = newest_bar_age_s(snap_rows)
        meta["last_close"] = float(snap_rows[-1][4]) if len(snap_rows[-1]) > 4 else None
        return snap_rows, meta

    meta["bars"] = len(bars)
    meta["status"] = "missing_or_stale"
    if bars:
        meta["bar_age_s"] = newest_bar_age_s(bars)
    logger.info(
        "[gemini-gold] klines not ready tf=%s bars=%d source=%s bar_age_s=%s stale_limit_s=%.0f",
        timeframe,
        len(bars),
        meta.get("source"),
        meta.get("bar_age_s"),
        meta["stale_limit_s"],
    )
    return [], meta


def klines_ready(bars_15m: List, bars_1h: List, *, min_bars: Optional[int] = None) -> bool:
    mb = min_bars if min_bars is not None else _min_bars()
    return len(bars_15m) >= mb and len(bars_1h) >= mb
