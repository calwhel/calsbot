"""
Dedicated XAUUSD / XAGUSD spot price poller.

cTrader is authoritative when linked, but gold ticks are often missing. This feed
keeps metals spot fresh by polling Coinbase / Kraken in parallel every
few seconds — even when cTrader is connected but its tick is older than the strict
real-time window.
"""
from __future__ import annotations

import asyncio
import logging
import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

_RUNNING = False
_feed_task: Optional[asyncio.Task] = None
_POLL_LOCK_ID = 708_110_006

# Canonical symbol → list of (source_label, fetch_key)
# Binance spot returns HTTP 451 on Railway US — Coinbase/Kraken only.
_METALS: Dict[str, List[Tuple[str, str]]] = {
    "XAUUSD": [
        ("coinbase", "XAU-USD"),
        ("kraken", "PAXGUSD"),
    ],
    "XAGUSD": [
        ("coinbase", "XAG-USD"),
    ],
}

_PRICE_CACHE: Dict[str, Tuple[float, datetime]] = {}
_FRESH_CTRADER_S = float(os.environ.get("REALTIME_SPOT_MAX_AGE_METALS_S", "3"))
_FRESH_ANY_S = _FRESH_CTRADER_S
_POLL_INTERVAL = max(2, int(os.environ.get("METALS_POLL_INTERVAL_SECONDS", "3")))


def is_live() -> bool:
    return _RUNNING


def cached_symbols() -> List[str]:
    return sorted(_PRICE_CACHE.keys())


def symbol_count() -> int:
    return len(_PRICE_CACHE)


def get_price(symbol: str) -> Optional[float]:
    """Mid from in-process cache or shared Postgres store (strict freshness)."""
    sym = symbol.upper()
    entry = _PRICE_CACHE.get(sym)
    if entry:
        age = (datetime.utcnow() - entry[1]).total_seconds()
        if age <= _FRESH_ANY_S:
            return entry[0]
    try:
        from app.services.spot_price_store import get_mid
        px = get_mid(sym, max_age_s=_FRESH_ANY_S)
        if px is not None:
            return px
    except Exception:
        pass
    return None


def _has_fresh_tick(symbol: str) -> bool:
    """True when any source has a tick within the real-time window."""
    sym = symbol.upper()
    try:
        from app.services.realtime_spot import read_fresh_cached
        return read_fresh_cached(sym, "forex", max_age=_FRESH_ANY_S) is not None
    except Exception:
        return get_price(sym) is not None


def _store(symbol: str, mid: float, source: str) -> None:
    if mid <= 0:
        return
    sym = symbol.upper()
    now = datetime.utcnow()
    _PRICE_CACHE[sym] = (mid, now)
    try:
        from app.services.spot_price_store import upsert_tick
        upsert_tick(sym, mid=mid, source=source[:20])
    except Exception:
        pass


async def _fetch_binance(pair: str) -> Optional[float]:
    try:
        import httpx
        async with httpx.AsyncClient(timeout=4.0) as client:
            r = await client.get(
                "https://api.binance.com/api/v3/ticker/price",
                params={"symbol": pair},
            )
        if r.status_code != 200:
            return None
        body = r.json()
        if isinstance(body, dict) and body.get("code") not in (None, 0):
            return None
        px = float(body.get("price", 0))
        return px if px > 0 else None
    except Exception:
        return None


async def _fetch_coinbase(pair: str) -> Optional[float]:
    if not pair:
        return None
    import httpx
    timeout = httpx.Timeout(connect=8.0, read=12.0, write=8.0, pool=8.0)
    for _ in range(2):
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                r = await client.get(f"https://api.coinbase.com/v2/prices/{pair}/spot")
            if r.status_code != 200:
                return None
            amount = ((r.json() or {}).get("data") or {}).get("amount")
            px = float(amount) if amount else 0.0
            return px if px > 0 else None
        except (httpx.ConnectTimeout, httpx.ReadTimeout, httpx.ConnectError):
            await asyncio.sleep(0.5)
        except Exception:
            return None
    return None


async def _fetch_kraken(pair: str) -> Optional[float]:
    try:
        import httpx
        async with httpx.AsyncClient(timeout=4.0) as client:
            r = await client.get(
                "https://api.kraken.com/0/public/Ticker",
                params={"pair": pair},
            )
        if r.status_code != 200:
            return None
        result = (r.json() or {}).get("result") or {}
        for _k, v in result.items():
            if not isinstance(v, dict):
                continue
            c = v.get("c") or []
            if c:
                px = float(c[0])
                return px if px > 0 else None
    except Exception:
        pass
    return None


_FETCHERS = {
    "binance": _fetch_binance,
    "coinbase": _fetch_coinbase,
    "kraken": _fetch_kraken,
}

_COINBASE_PAIR = {"XAUUSDT": "XAU-USD", "XAGUSDT": "XAG-USD"}


async def _fetch_source(source: str, key: str) -> Optional[Tuple[float, str]]:
    fetcher = _FETCHERS.get(source)
    if not fetcher:
        return None
    pair = _COINBASE_PAIR.get(key, key) if source == "coinbase" else key
    px = await fetcher(pair)
    if px is not None and px > 0:
        return (float(px), source)
    return None


async def fetch_now(symbol: str) -> Optional[float]:
    """On-demand metals price — parallel fetch from all externals."""
    sym = symbol.upper()
    if sym not in _METALS:
        return None
    if _has_fresh_tick(sym):
        return get_price(sym)
    if await _poll_symbol(sym):
        return get_price(sym)
    return None


async def _poll_symbol(symbol: str) -> bool:
    sym = symbol.upper()
    if _has_fresh_tick(sym):
        return True

    tasks = [
        _fetch_source(source, key)
        for source, key in _METALS.get(sym, [])
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    candidates: List[Tuple[float, str]] = []
    for r in results:
        if isinstance(r, tuple) and len(r) == 2:
            candidates.append(r)

    if not candidates:
        return False

    # Prefer coinbase > kraken when multiple succeed.
    rank = {"coinbase": 0, "kraken": 1}
    candidates.sort(key=lambda x: rank.get(x[1], 99))
    mid, source = candidates[0]
    _store(sym, mid, source)
    return True


def _acquire_lock():
    try:
        import psycopg2
        from app.config import settings
        conn = psycopg2.connect(settings.get_database_url())
        conn.autocommit = True
        cur = conn.cursor()
        cur.execute("SELECT pg_try_advisory_lock(%s)", (_POLL_LOCK_ID,))
        if not cur.fetchone()[0]:
            conn.close()
            return None
        return conn
    except Exception:
        return None


def _release_lock(conn) -> None:
    if not conn:
        return
    try:
        cur = conn.cursor()
        cur.execute("SELECT pg_advisory_unlock(%s)", (_POLL_LOCK_ID,))
    except Exception:
        pass
    try:
        conn.close()
    except Exception:
        pass


async def _stream() -> None:
    global _RUNNING
    logger.info(
        f"[MetalsFeed] started — XAUUSD/XAGUSD every {_POLL_INTERVAL}s "
        "(parallel coinbase/kraken; always-on real-time spot)"
    )
    _RUNNING = True
    try:
        while True:
            lock_conn = await asyncio.to_thread(_acquire_lock)
            if not lock_conn:
                await asyncio.sleep(_POLL_INTERVAL)
                continue
            try:
                results = await asyncio.gather(
                    *[_poll_symbol(sym) for sym in _METALS],
                    return_exceptions=True,
                )
                got = sum(1 for r in results if r is True)
                if got:
                    logger.debug(f"[MetalsFeed] fresh tick(s) for {got} metal(s)")
            finally:
                await asyncio.to_thread(_release_lock, lock_conn)
            await asyncio.sleep(_POLL_INTERVAL)
    except asyncio.CancelledError:
        logger.info("[MetalsFeed] polling task cancelled")
        raise
    finally:
        _RUNNING = False


def start() -> None:
    """Schedule background metals poller (executor worker only)."""
    global _feed_task
    if _feed_task and not _feed_task.done():
        return
    try:
        loop = asyncio.get_event_loop()
        _feed_task = loop.create_task(_stream())
        logger.info("[MetalsFeed] background task scheduled")
    except Exception as e:
        logger.error(f"[MetalsFeed] failed to start: {e}")


def stop() -> None:
    global _feed_task
    if _feed_task and not _feed_task.done():
        _feed_task.cancel()
        logger.info("[MetalsFeed] background task cancelled")
    _feed_task = None
