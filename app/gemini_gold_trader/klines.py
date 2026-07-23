"""Gemini Gold kline + scan chart resolution."""
from __future__ import annotations

import logging
import os
import asyncio
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
    """Roll forming 1m/5m/15m bar from live cTrader spot when available."""
    if not bars or timeframe not in ("1m", "5m", "15m"):
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


async def _fetch_ctrader_chart_klines(
    timeframe: str,
    limit: int,
    *,
    user_id: Optional[int] = None,
) -> Tuple[List[List[float]], str]:
    """Direct cTrader trendbars — bypasses tradfi metal cache."""
    attempts = max(1, int(os.environ.get("GEMINI_GOLD_CTRADER_KLINE_RETRIES", "3")))
    delay_s = max(0.1, float(os.environ.get("GEMINI_GOLD_CTRADER_KLINE_RETRY_DELAY_S", "0.4")))
    last_exc: Optional[Exception] = None

    for attempt in range(1, attempts + 1):
        try:
            from app.services import ctrader_price_feed as ctf

            # ctf.get_klines is safe to call even while trendbar fetch is blocked:
            # it returns the in-memory tick-built cache (rolled forward from live
            # spot) or the shared Postgres cTrader snapshot without ever touching
            # external tradfi providers. Short-circuiting on the block reason here
            # (as we used to) skipped that fresh in-process cache and forced a
            # ``ctrader_unavailable`` data-gate block whenever a transient trendbar
            # backoff was active — even though live ticks were flowing.
            rows = await ctf.get_klines(
                SYMBOL,
                ASSET_CLASS,
                timeframe,
                limit,
                user_id=user_id,
            )
            rows = _synthesize_forming_bar(rows or [], timeframe, limit)
            if _bars_fresh(rows, timeframe):
                label = "ctrader-user" if user_id else "ctrader"
                return rows, label
            if attempt < attempts:
                await asyncio.sleep(delay_s)
        except Exception as exc:
            last_exc = exc
            logger.debug(
                "[gemini-gold] direct ctrader %s attempt %s/%s failed: %s",
                timeframe,
                attempt,
                attempts,
                exc,
            )
            if attempt < attempts:
                await asyncio.sleep(delay_s)
    if last_exc:
        logger.debug("[gemini-gold] direct ctrader %s exhausted retries: %s", timeframe, last_exc)
    return [], ""


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

    # Gemini Gold is cTrader-only: when cTrader trendbars are actively blocked,
    # the shared tradfi chain would burn 20-48s on Coinbase/Kraken/FMP timeouts
    # only for us to discard the result. Skip straight to the direct cTrader
    # retry + Postgres cTrader snapshot instead.
    trendbar_block_reason = None
    try:
        from app.services import ctrader_price_feed as _ctf

        trendbar_block_reason = _ctf.trendbar_fetch_blocked_reason()
    except Exception:
        trendbar_block_reason = None
    trendbar_blocked = bool(trendbar_block_reason)

    if not trendbar_blocked:
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
    else:
        meta["source"] = "ctrader_blocked"
        logger.info(
            "[gemini-gold] trendbars blocked (%s) — skipping tradfi externals for %s",
            trendbar_block_reason or "?",
            timeframe,
        )

    if not _is_ctrader_kline_source(tradfi_source):
        ct_rows, ct_src = await _fetch_ctrader_chart_klines(
            timeframe, limit, user_id=user_id,
        )
        if ct_rows:
            meta["source"] = ct_src
            meta["bars"] = len(ct_rows)
            meta["status"] = "ok"
            meta["bar_age_s"] = newest_bar_age_s(ct_rows)
            meta["last_close"] = float(ct_rows[-1][4]) if len(ct_rows[-1]) > 4 else None
            return ct_rows, meta

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
    if not _is_ctrader_kline_source(tradfi_source):
        meta["source"] = "ctrader_unavailable"
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


def klines_ready(
    bars_5m: List,
    bars_15m: List,
    bars_1h: List,
    *,
    min_bars: Optional[int] = None,
) -> bool:
    """Core scalp stack — 5m trigger, 15m structure, 1h bias."""
    mb = min_bars if min_bars is not None else _min_bars()
    return all(len(bars) >= mb for bars in (bars_5m, bars_15m, bars_1h))


def charts_are_ctrader(*metas: Optional[Dict]) -> Tuple[bool, str]:
    """Confirm every rendered chart came from a cTrader source before spending
    a Gemini vision call. Returns (ok, reason) where reason names the offending
    timeframe/source on failure. Empty/absent metas are ignored (bar-count
    readiness is enforced separately by ``klines_ready``)."""
    for meta in metas:
        if not meta:
            continue
        src = (meta.get("source") or "").lower()
        if not src:
            continue
        if int(meta.get("bars") or 0) <= 0:
            continue
        if not _is_ctrader_kline_source(src):
            tf = meta.get("timeframe") or "?"
            return False, f"non_ctrader_{tf}:{src}"
    return True, "ok"


def has_1m_chart(bars_1m: List, *, min_bars: Optional[int] = None) -> bool:
    mb = min_bars if min_bars is not None else _min_bars()
    return len(bars_1m) >= mb


def resolve_entry_chart(
    bars_1m: List[List[float]],
    bars_5m: List[List[float]],
    *,
    min_bars: Optional[int] = None,
) -> Tuple[List[List[float]], str, bool]:
    """
    Return (entry_bars, entry_timeframe_label, used_5m_fallback).

    When broker 1m is unavailable, use recent 5m bars for entry timing.
    """
    mb = min_bars if min_bars is not None else _min_bars()
    if len(bars_1m) >= mb:
        return bars_1m, "1m", False
    if len(bars_5m) >= mb:
        window = min(len(bars_5m), max(mb, 40))
        logger.info(
            "[gemini-gold] 1m unavailable — using recent %d×5m bars for entry timing",
            window,
        )
        return bars_5m[-window:], "5m", True
    return [], "1m", False
