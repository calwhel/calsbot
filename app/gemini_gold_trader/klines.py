"""Read-only kline access from Postgres snapshots."""
from __future__ import annotations

import logging
import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from app.gemini_gold_trader.config import SYMBOL

logger = logging.getLogger(__name__)


def _max_age_s() -> float:
    try:
        return max(30.0, float(os.environ.get("GEMINI_GOLD_KLINE_MAX_AGE_S", "300")))
    except (TypeError, ValueError):
        return 300.0


def get_chart_klines(timeframe: str, limit: int) -> Tuple[List[List[float]], Dict]:
    """Return (bars, meta) from market_kline_snapshots — read-only, no cTrader sockets."""
    from app.services.kline_snapshot_store import get_klines

    bars = get_klines(
        SYMBOL,
        timeframe,
        limit,
        max_age_s=_max_age_s(),
        source="ctrader",
    )
    meta = {
        "timeframe": timeframe,
        "bars": len(bars),
        "source": "ctrader_snapshot",
        "max_age_s": _max_age_s(),
        "fetched_at": datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
    }
    if not bars:
        meta["status"] = "missing_or_stale"
        return [], meta
    meta["status"] = "ok"
    meta["last_close"] = float(bars[-1][4]) if len(bars[-1]) > 4 else None
    return bars, meta


def klines_ready(bars_15m: List, bars_1h: List, *, min_bars: int = 20) -> bool:
    return len(bars_15m) >= min_bars and len(bars_1h) >= min_bars
