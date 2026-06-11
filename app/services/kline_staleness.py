"""Source-agnostic kline cache staleness — applies to cTrader, Coinbase, Kraken, FMP, etc."""
from __future__ import annotations

import logging
import os
import time
from datetime import datetime
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)

_TF_MINUTES = {
    "1m": 1, "3m": 3, "5m": 5, "10m": 10, "15m": 15,
    "30m": 30, "1h": 60, "4h": 240, "1d": 1440,
}

_KLINE_STALE_DRIFT_PCT = float(os.environ.get("KLINE_STALE_DRIFT_PCT", "0.75"))


def stale_limit_s(timeframe: str) -> float:
    """Max age before klines are stale while live spot is updating (2× bar width)."""
    return 2.0 * _TF_MINUTES.get(timeframe, 15) * 60.0


def newest_bar_age_s(rows: List[List[float]]) -> float:
    if not rows:
        return float("inf")
    try:
        newest_ts_ms = int(rows[-1][0])
        return max(0.0, time.time() - newest_ts_ms / 1000.0)
    except (IndexError, TypeError, ValueError):
        return float("inf")


def kline_close_drift_pct(live_px: float, rows: List[List[float]]) -> Optional[float]:
    if not rows or not live_px or live_px <= 0:
        return None
    try:
        close = float(rows[-1][4])
    except (IndexError, TypeError, ValueError):
        return None
    if close <= 0:
        return None
    return abs(live_px - close) / live_px * 100.0


async def metal_live_spot_updating(symbol: str, max_age_s: float = 30.0) -> Tuple[bool, Optional[float]]:
    """True when a fresh live metal spot price is available (any source)."""
    sym = symbol.upper()
    try:
        from app.services.tradfi_prices import get_price_fresh
        px = await get_price_fresh(sym, "forex")
        if px and px > 0:
            return True, float(px)
    except Exception:
        pass
    try:
        from app.services.ctrader_price_feed import get_price, is_live
        if is_live():
            px = get_price(sym)
            if px and px > 0:
                return True, float(px)
    except Exception:
        pass
    return False, None


def cached_klines_stale(
    symbol: str,
    rows: List[List[float]],
    timeframe: str,
    *,
    cache_fetched_at: Optional[datetime] = None,
    live_px: Optional[float] = None,
    live_updating: bool = False,
) -> Tuple[bool, str]:
    """Source-agnostic staleness: bar timestamp, fetch age, or live/kline drift."""
    if not live_updating:
        return False, ""
    stale_limit = stale_limit_s(timeframe)
    bar_age = newest_bar_age_s(rows)
    fetch_age: Optional[float] = None
    if cache_fetched_at is not None:
        fetch_age = max(0.0, (datetime.utcnow() - cache_fetched_at).total_seconds())
    if bar_age > stale_limit:
        return True, f"bar_ts={bar_age:.0f}s"
    if fetch_age is not None and fetch_age > stale_limit:
        return True, f"fetch={fetch_age:.0f}s"
    if live_px is not None:
        drift = kline_close_drift_pct(live_px, rows)
        if drift is not None and drift > _KLINE_STALE_DRIFT_PCT:
            return True, f"drift={drift:.2f}%"
    return False, ""


async def check_cached_klines_stale(
    symbol: str,
    rows: List[List[float]],
    timeframe: str,
    *,
    cache_fetched_at: Optional[datetime] = None,
) -> Tuple[bool, str]:
    """Async wrapper — resolves live spot before staleness check."""
    live_updating, live_px = await metal_live_spot_updating(symbol)
    return cached_klines_stale(
        symbol,
        rows,
        timeframe,
        cache_fetched_at=cache_fetched_at,
        live_px=live_px,
        live_updating=live_updating,
    )
