"""Pre-scan XAUUSD kline refresh — reduce stale_klines blocks."""
from __future__ import annotations

import asyncio
import logging
import os
import time
from typing import Optional

from app.gold_ai_trader.config import ASSET_CLASS, SYMBOL
from app.gold_ai_trader.klines import (
    fetch_gold_scoring_k5,
    is_ctrader_kline_source,
)
from app.services.kline_staleness import newest_bar_age_s
from app.services.tradfi_prices import (
    clear_metal_kline_cache,
    sweep_stale_metal_klines,
)

logger = logging.getLogger(__name__)

_SCORING_TF = "5m"
_SCORING_LIMIT = 60
_last_ctrader_restart_mono = 0.0


def _refresh_age_s() -> float:
    try:
        return max(120.0, float(os.environ.get("GOLD_AI_KLINE_REFRESH_AGE_S", "300")))
    except (TypeError, ValueError):
        return 300.0


def _restart_min_interval_s() -> float:
    try:
        return max(
            60.0,
            float(os.environ.get("GOLD_AI_KLINE_RESTART_MIN_INTERVAL_S", "180")),
        )
    except (TypeError, ValueError):
        return 180.0


async def _maybe_restart_ctrader_builder(reason: str) -> bool:
    global _last_ctrader_restart_mono
    now = time.monotonic()
    if now - _last_ctrader_restart_mono < _restart_min_interval_s():
        return False
    from app.services.ctrader_price_feed import restart_kline_builder

    await restart_kline_builder(reason)
    _last_ctrader_restart_mono = time.monotonic()
    return True


async def _hard_refresh_scoring_k5(
    *,
    user_id: Optional[int] = None,
    reason: str,
) -> tuple[list, str, float, bool]:
    """Clear caches, optionally restart cTrader builder, refetch scoring 5m."""
    cleared = clear_metal_kline_cache([SYMBOL])
    block_reason = ""
    restarted = False
    try:
        from app.services.ctrader_price_feed import (
            sweep_stale_klines,
            trendbar_fetch_blocked_reason,
        )

        block_reason = trendbar_fetch_blocked_reason() or ""
        restarted = await _maybe_restart_ctrader_builder(reason)
        if restarted:
            await sweep_stale_klines(symbols=[SYMBOL], timeframes=["1m", "5m", "15m", "1h"])
    except Exception as exc:
        logger.warning("[gold-ai] ctrader builder restart failed: %s", exc)

    k5, src = await fetch_gold_scoring_k5(user_id=user_id)
    new_age = newest_bar_age_s(k5)
    logger.warning(
        "[gold-ai] 5m hard refresh (%s); restarted=%s trendbar_block=%s "
        "cleared=%s bar_age=%.0fs source=%s bars=%s",
        reason,
        restarted,
        block_reason or "none",
        cleared,
        new_age if new_age != float("inf") else -1.0,
        src or "unknown",
        len(k5),
    )
    return k5, src or "", float(cleared), restarted


async def refresh_gold_scoring_klines(*, user_id: Optional[int] = None) -> dict:
    """
    Sweep stale metal caches and force-refresh 5m when last bar is too old
    or when scoring klines are on a non-cTrader fallback provider.

    Returns a small summary dict for logging/diagnostics.
    """
    summary = {
        "swept": 0,
        "ctrader_swept": 0,
        "cleared": 0,
        "refreshed_5m": False,
        "bar_age_before": None,
        "bar_age_after": None,
        "kline_source_before": None,
        "kline_source_after": None,
        "ctrader_restarted": False,
        "fallback_recovery": False,
    }

    try:
        from app.services.ctrader_price_feed import sweep_stale_klines

        m_res, c_res = await asyncio.gather(
            sweep_stale_metal_klines([SYMBOL], timeframes=["1m", "5m", "15m", "1h"]),
            sweep_stale_klines(symbols=[SYMBOL], timeframes=["1m", "5m", "15m", "1h"]),
            return_exceptions=True,
        )
        if isinstance(m_res, Exception):
            logger.debug("[gold-ai] metal kline sweep failed: %s", m_res)
        else:
            summary["swept"] = int(m_res or 0)
        if isinstance(c_res, Exception):
            logger.debug("[gold-ai] ctrader kline sweep failed: %s", c_res)
        else:
            summary["ctrader_swept"] = int(c_res or 0)
    except Exception as exc:
        logger.debug("[gold-ai] kline sweep: %s", exc)

    k5, src = await fetch_gold_scoring_k5(user_id=user_id)
    bar_age = newest_bar_age_s(k5)
    summary["bar_age_before"] = round(bar_age, 1) if bar_age != float("inf") else None
    summary["kline_source_before"] = src or None

    stale = bar_age > _refresh_age_s()
    fallback = not is_ctrader_kline_source(src)
    if not stale and not fallback:
        summary["kline_source_after"] = src or None
        return summary

    if fallback:
        summary["fallback_recovery"] = True
        logger.warning(
            "[gold-ai] 5m on fallback source=%s (bar_age=%.0fs) — forcing cTrader recovery",
            src or "missing",
            bar_age if bar_age != float("inf") else -1.0,
        )
    else:
        logger.info(
            "[gold-ai] 5m klines stale (bar_age=%.0fs > %.0fs) — cache clearing",
            bar_age,
            _refresh_age_s(),
        )

    k5, src, cleared, restarted = await _hard_refresh_scoring_k5(
        user_id=user_id,
        reason="gold_ai_fallback_klines" if fallback else "gold_ai_5m_stale_refresh",
    )
    summary["cleared"] = int(cleared)
    summary["refreshed_5m"] = True
    summary["ctrader_restarted"] = restarted
    new_age = newest_bar_age_s(k5)
    summary["bar_age_after"] = round(new_age, 1) if new_age != float("inf") else None
    summary["kline_source_after"] = src or None

    if not is_ctrader_kline_source(src) and fallback:
        k5, src, cleared, restarted = await _hard_refresh_scoring_k5(
            user_id=user_id,
            reason="gold_ai_fallback_persisted",
        )
        summary["cleared"] += int(cleared)
        summary["ctrader_restarted"] = summary["ctrader_restarted"] or restarted
        new_age = newest_bar_age_s(k5)
        summary["bar_age_after"] = round(new_age, 1) if new_age != float("inf") else None
        summary["kline_source_after"] = src or None

    if not is_ctrader_kline_source(src):
        logger.warning(
            "[gold-ai] 5m refresh still non-cTrader source=%s — data gate may block",
            src,
        )
    return summary
