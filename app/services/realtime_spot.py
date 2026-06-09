"""
Unified real-time spot price resolver for tradfi (forex / metals / indices).

Trading paths must never use stale ticks. This module:
  • Reads only ticks younger than REALTIME_SPOT_MAX_AGE_* (default 3s metals, 5s forex)
  • Fetches all available sources in parallel when cache is cold
  • Prefers cTrader (broker-matched fills) when fresh; on cache miss actively
    waits for stream ticks or pulls a 1m trendbar before FMP/yfinance
"""
from __future__ import annotations

import asyncio
import logging
import os
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from app.services.asset_classes import (
    ASSET_CLASS_FOREX,
    ASSET_CLASS_INDEX,
    normalize_asset_class,
)

logger = logging.getLogger(__name__)

# Strict max tick age for trading / entry confirmation (seconds).
_MAX_AGE_METALS = float(os.environ.get("REALTIME_SPOT_MAX_AGE_METALS_S", "3"))
_MAX_AGE_FOREX = float(os.environ.get("REALTIME_SPOT_MAX_AGE_FOREX_S", "5"))
_MAX_AGE_INDEX = float(os.environ.get("REALTIME_SPOT_MAX_AGE_INDEX_S", "5"))

_METALS = frozenset({"XAUUSD", "XAGUSD"})
_BINANCE_MAP = {"XAUUSD": "XAUUSDT", "XAGUSD": "XAGUSDT"}
_COINBASE_MAP = {"XAUUSDT": "XAU-USD", "XAGUSDT": "XAG-USD"}
_BINANCE_BASE = "https://api.binance.com/api/v3"

# Source preference when multiple fresh ticks arrive (lower = better).
_SOURCE_RANK = {
    "ctrader": 0,
    "binance": 1,
    "coinbase": 2,
    "fmp": 3,
    "twelvedata": 4,
    "kraken": 5,
    "yfinance": 6,
    "metals_cache": 7,
    "store": 99,
}
_PAPER_MAX_AGE_S = float(os.environ.get("REALTIME_SPOT_MAX_AGE_PAPER_S", "15"))
_CTRADER_ON_DEMAND_WAIT_S = float(os.environ.get("CTRADER_ON_DEMAND_WAIT_S", "2.5"))
_CTRADER_SPOT_BAR_TF = os.environ.get("CTRADER_SPOT_BAR_TF", "1m")
_CTRADER_SPOT_BAR_MINUTES = int(os.environ.get("CTRADER_SPOT_BAR_MINUTES", "1"))


def _norm(symbol: str) -> str:
    return symbol.upper().replace("/", "").replace("-", "")


def is_metal(symbol: str) -> bool:
    return _norm(symbol) in _METALS


def max_age_s(symbol: str, asset_class: str) -> float:
    sym = _norm(symbol)
    if sym in _METALS:
        return _MAX_AGE_METALS
    cls = normalize_asset_class(asset_class)
    if cls == ASSET_CLASS_INDEX:
        return _MAX_AGE_INDEX
    return _MAX_AGE_FOREX


def _rank(source: str) -> int:
    return _SOURCE_RANK.get((source or "").lower(), 99)


def _pick_best(candidates: List[Tuple[float, str]]) -> Optional[Tuple[float, str]]:
    if not candidates:
        return None
    candidates.sort(key=lambda x: (_rank(x[1]), -x[0]))
    return candidates[0]


def _read_ctrader(symbol: str, max_age: float) -> Optional[Tuple[float, str]]:
    sym = _norm(symbol)
    try:
        from app.services import ctrader_price_feed as ctf
        px = ctf.get_price(sym)
        if px is not None and px > 0:
            return (float(px), "ctrader")
    except Exception:
        pass
    try:
        from app.services.spot_price_store import get_tick
        row = get_tick(sym, max_age_s=max_age)
        if row and (row.get("source") or "").lower() == "ctrader":
            mid = float(row["mid"])
            if mid > 0:
                return (mid, "ctrader")
    except Exception:
        pass
    return None


def _read_store(symbol: str, max_age: float) -> Optional[Tuple[float, str]]:
    sym = _norm(symbol)
    try:
        from app.services.spot_price_store import get_tick
        row = get_tick(sym, max_age_s=max_age)
        if row:
            mid = float(row["mid"])
            src = (row.get("source") or "store").lower()
            if mid > 0:
                return (mid, src)
    except Exception:
        pass
    return None


def _read_metals_cache(symbol: str, max_age: float) -> Optional[Tuple[float, str]]:
    sym = _norm(symbol)
    if sym not in _METALS:
        return None
    try:
        from app.services import metals_spot_feed as msf
        entry = msf._PRICE_CACHE.get(sym)  # noqa: SLF001 — shared freshness gate
        if entry:
            px, ts = entry
            age = (datetime.utcnow() - ts).total_seconds()
            if age <= max_age and px > 0:
                return (float(px), "metals_cache")
    except Exception:
        pass
    return None


def _read_twelve_data_fresh(symbol: str, max_age: float) -> Optional[Tuple[float, str]]:
    sym = _norm(symbol)
    try:
        from app.services import twelve_data_feed as td
        if not td.is_active_window():
            return None
        px = td.read_cached_quote(sym, max_age)
        if px is not None and px > 0:
            return (float(px), "twelvedata")
    except Exception:
        pass
    return None


def _read_fmp_fresh(symbol: str, max_age: float) -> Optional[Tuple[float, str]]:
    sym = _norm(symbol)
    try:
        from app.services import fmp_price_feed as fmp
        entry = fmp._PRICE_CACHE.get(sym)  # noqa: SLF001
        if entry:
            px, ts = entry
            age = (datetime.utcnow() - ts).total_seconds()
            if age <= max_age and px > 0:
                return (float(px), "fmp")
    except Exception:
        pass
    return None


def _effective_max_age(
    symbol: str,
    asset_class: str,
    *,
    max_age: Optional[float] = None,
    paper_ok: bool = False,
) -> float:
    base = max_age if max_age is not None else max_age_s(symbol, asset_class)
    if paper_ok:
        return max(base, _PAPER_MAX_AGE_S)
    return base


def read_fresh_cached(
    symbol: str,
    asset_class: str,
    *,
    max_age: Optional[float] = None,
    paper_ok: bool = False,
) -> Optional[Tuple[float, str]]:
    """Return (price, source) from in-memory / Postgres caches only."""
    sym = _norm(symbol)
    age_limit = _effective_max_age(sym, asset_class, max_age=max_age, paper_ok=paper_ok)
    candidates: List[Tuple[float, str]] = []

    hit = _read_ctrader(sym, age_limit)
    if hit:
        candidates.append(hit)

    if sym in _METALS:
        hit = _read_metals_cache(sym, age_limit)
        if hit:
            candidates.append(hit)

    hit = _read_store(sym, age_limit)
    if hit and hit not in candidates:
        candidates.append(hit)

    if sym not in _METALS:
        hit = _read_fmp_fresh(sym, age_limit)
        if hit:
            candidates.append(hit)
        hit = _read_twelve_data_fresh(sym, age_limit)
        if hit:
            candidates.append(hit)

    return _pick_best(candidates)


async def _fetch_binance(pair: str) -> Optional[float]:
    try:
        import httpx
        async with httpx.AsyncClient(timeout=4.0) as client:
            r = await client.get(
                f"{_BINANCE_BASE}/ticker/price",
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
    try:
        import httpx
        async with httpx.AsyncClient(timeout=4.0) as client:
            r = await client.get(f"https://api.coinbase.com/v2/prices/{pair}/spot")
        if r.status_code != 200:
            return None
        amount = ((r.json() or {}).get("data") or {}).get("amount")
        px = float(amount) if amount else 0.0
        return px if px > 0 else None
    except Exception:
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
        for v in result.values():
            if not isinstance(v, dict):
                continue
            c = v.get("c") or []
            if c:
                px = float(c[0])
                return px if px > 0 else None
    except Exception:
        pass
    return None


def _persist_tick(symbol: str, mid: float, source: str) -> None:
    sym = _norm(symbol)
    try:
        from app.services.spot_price_store import upsert_tick
        upsert_tick(sym, mid=mid, source=source[:20])
    except Exception:
        pass
    if sym in _METALS:
        try:
            from app.services import metals_spot_feed as msf
            msf._PRICE_CACHE[sym] = (mid, datetime.utcnow())  # noqa: SLF001
        except Exception:
            pass


async def _fetch_metals_parallel(symbol: str) -> Optional[Tuple[float, str]]:
    sym = _norm(symbol)
    bn = _BINANCE_MAP.get(sym)
    if not bn:
        return None

    async def _try(source: str, coro) -> Optional[Tuple[float, str]]:
        try:
            px = await coro
            if px is not None and px > 0:
                return (float(px), source)
        except Exception:
            pass
        return None

    tasks = [
        _try("binance", _fetch_binance(bn)),
        _try("coinbase", _fetch_coinbase(_COINBASE_MAP.get(bn, ""))),
    ]
    if sym == "XAUUSD":
        tasks.append(_try("kraken", _fetch_kraken("PAXGUSD")))

    results = await asyncio.gather(*tasks, return_exceptions=True)
    candidates: List[Tuple[float, str]] = []
    for r in results:
        if isinstance(r, tuple) and len(r) == 2:
            candidates.append(r)

    best = _pick_best(candidates)
    if best:
        _persist_tick(sym, best[0], best[1])
    return best


async def _fetch_yfinance_spot(
    symbol: str,
    asset_class: str,
) -> Optional[Tuple[float, str]]:
    """yfinance fast_info — forex/index spot when broker feeds are cold."""
    try:
        from app.services.tradfi_prices import _resolve_ticker, _yf_fast_price_blocking

        ticker = _resolve_ticker(asset_class, symbol)
        if not ticker:
            return None
        px = await asyncio.to_thread(_yf_fast_price_blocking, ticker)
        if px is not None and px > 0:
            _persist_tick(symbol, float(px), "yfinance")
            return (float(px), "yfinance")
    except Exception:
        pass
    return None


async def _fetch_twelve_data_on_demand(
    symbol: str,
    asset_class: str,
    max_age: float,
) -> Optional[Tuple[float, str]]:
    """Fire-time only — 09:00–18:00 UTC, rate-capped Twelve Data /price."""
    sym = _norm(symbol)
    hit = _read_twelve_data_fresh(sym, max_age)
    if hit:
        return hit
    try:
        from app.services.twelve_data_feed import fetch_quote

        px = await fetch_quote(sym, asset_class, max_age=max_age)
        if px is not None and px > 0:
            return (float(px), "twelvedata")
    except Exception:
        pass
    return None


async def _fetch_fmp_on_demand(
    symbol: str,
    max_age: float,
) -> Optional[Tuple[float, str]]:
    """Poll cache first, then a single-symbol FMP HTTP quote."""
    sym = _norm(symbol)
    hit = _read_fmp_fresh(sym, max_age)
    if hit:
        return hit
    try:
        from app.services.fmp_price_feed import fetch_quote

        px = await fetch_quote(sym)
        if px is not None and px > 0:
            return (float(px), "fmp")
    except Exception:
        pass
    return None


def _ctrader_fresh(symbol: str, max_age: float) -> Optional[Tuple[float, str]]:
    """Broker-matched tick from cTrader feed or Postgres store."""
    return _read_ctrader(symbol, max_age)


async def _fetch_ctrader_on_demand(
    symbol: str,
    asset_class: str,
    max_age: float,
    *,
    paper_ok: bool = False,
    user_id: Optional[int] = None,
) -> Optional[Tuple[float, str]]:
    """
    Active cTrader path when cache is cold: wait for stream tick, then 1m bar close.
    """
    sym = _norm(symbol)
    try:
        from app.services import ctrader_price_feed as ctf
        if not ctf.broker_session_ready(sym):
            return None
        feed_live = bool(ctf.is_live())
    except Exception:
        return None

    if feed_live and _CTRADER_ON_DEMAND_WAIT_S > 0:
        deadline = time.monotonic() + _CTRADER_ON_DEMAND_WAIT_S
        while time.monotonic() < deadline:
            hit = _read_ctrader(sym, max_age)
            if hit:
                return hit
            await asyncio.sleep(0.25)

    try:
        from app.services import ctrader_price_feed as ctf

        timeout = 12.0 if feed_live else 6.0
        rows = await asyncio.wait_for(
            ctf.get_klines(
                sym, asset_class, _CTRADER_SPOT_BAR_TF, 2, user_id=user_id,
            ),
            timeout=timeout,
        )
        if not rows:
            return None
        bar = rows[-1]
        if not bar or len(bar) < 5:
            return None
        ts_ms, close = float(bar[0]), float(bar[4])
        if close <= 0:
            return None
        bar_end_s = ts_ms / 1000.0 + _CTRADER_SPOT_BAR_MINUTES * 60.0
        bar_age = time.time() - bar_end_s
        if bar_age > max_age:
            return None
        _persist_tick(sym, close, "ctrader")
        return (close, "ctrader")
    except Exception:
        return None


async def fetch_parallel(
    symbol: str,
    asset_class: str,
    *,
    paper_ok: bool = False,
    user_id: Optional[int] = None,
    twelve_data_ok: bool = False,
) -> Optional[Tuple[float, str]]:
    """
    Hit live sources; return best fresh tick.

    Priority: cTrader → FMP → Twelve Data (09–18 UTC, fire-time only) → yfinance.
    cTrader is queried from cache first, then on-demand (stream wait + trendbar).
    """
    sym = _norm(symbol)
    age_limit = _effective_max_age(sym, asset_class, paper_ok=paper_ok)
    candidates: List[Tuple[float, str]] = []

    # 1) cTrader cache — in-memory + shared Postgres store.
    ct = _ctrader_fresh(sym, age_limit)
    if ct:
        candidates.append(ct)

    cached = read_fresh_cached(sym, asset_class, max_age=age_limit, paper_ok=paper_ok)
    if cached and cached not in candidates:
        candidates.append(cached)

    parallel: List = []
    if not ct:
        parallel.append(
            _fetch_ctrader_on_demand(
                sym, asset_class, age_limit, paper_ok=paper_ok, user_id=user_id,
            )
        )
    if sym in _METALS:
        parallel.append(_fetch_metals_parallel(sym))
    else:
        parallel.append(_fetch_fmp_on_demand(sym, age_limit))

    for result in await asyncio.gather(*parallel, return_exceptions=True):
        if isinstance(result, tuple) and len(result) == 2 and result not in candidates:
            candidates.append(result)

    if sym not in _METALS:
        if twelve_data_ok:
            td = await _fetch_twelve_data_on_demand(sym, asset_class, age_limit)
            if td and td not in candidates:
                candidates.append(td)

        need_yf = paper_ok or not any(
            (c[1] or "").lower() in ("ctrader", "twelvedata") for c in candidates
        )
        if need_yf:
            yf = await _fetch_yfinance_spot(sym, asset_class)
            if yf:
                candidates.append(yf)

    return _pick_best(candidates)


async def get_realtime_spot(
    symbol: str,
    asset_class: str,
    *,
    force_fetch: bool = False,
    paper_ok: bool = False,
    user_id: Optional[int] = None,
    twelve_data_ok: bool = False,
) -> Optional[float]:
    """
    Latest real-time spot mid. Returns None when no source has a tick within
    the max-age window (never serves stale / futures fallbacks for metals).
    """
    sym = _norm(symbol)
    age_limit = _effective_max_age(sym, asset_class, paper_ok=paper_ok)

    if not force_fetch:
        cached = read_fresh_cached(
            sym, asset_class, max_age=age_limit, paper_ok=paper_ok,
        )
        if cached:
            return cached[0]

    fetched = await fetch_parallel(
        sym,
        asset_class,
        paper_ok=paper_ok,
        user_id=user_id,
        twelve_data_ok=twelve_data_ok,
    )
    return fetched[0] if fetched else None


async def prime_symbols(symbols: List[Tuple[str, str]]) -> int:
    """Warm spot store for key symbols; returns count refreshed."""
    refreshed = 0
    for sym, ac in symbols:
        try:
            px = await get_realtime_spot(sym, ac, force_fetch=True)
            if px is not None and px > 0:
                refreshed += 1
        except Exception:
            pass
    return refreshed
