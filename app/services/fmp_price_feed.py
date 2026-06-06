"""
Real-time forex + index price feed via FMP REST polling.

Polls FMP forex/index quotes every 5 seconds. Kline fetches use FMP's
`/stable/historical-chart` API first (legacy `/api/v3/` is blocked for
new subscriptions), with symbol aliases for metals (XAUUSD → GCUSD).
"""
from __future__ import annotations

import asyncio
import logging
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ── Price cache ──────────────────────────────────────────────────────────────
_PRICE_CACHE: Dict[str, Tuple[float, datetime]] = {}
_PRICE_TTL = timedelta(seconds=10)

_RUNNING = False

# ── Symbol maps ──────────────────────────────────────────────────────────────
_FOREX_SYMBOLS = [
    "EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD",
    "USDCHF", "NZDUSD", "EURGBP", "EURJPY", "GBPJPY",
    "XAUUSD", "XAGUSD",
]

# Index display → FMP quote ticker (canonical + legacy aliases)
_INDEX_FMP: Dict[str, str] = {
    "NAS100": "%5ENDX",  "NDX": "%5ENDX",  "US100": "%5ENDX",
    "SPX500": "%5EGSPC", "SPX": "%5EGSPC", "US500": "%5EGSPC",
    "US30":   "%5EDJI",  "DJI": "%5EDJI",
    "GER40":  "%5EGDAXI", "DAX": "%5EGDAXI",
    "UK100":  "%5EFTSE", "FTSE": "%5EFTSE",
    "VIX":    "%5EVIX",
}

# Reverse map: FMP ticker → display symbol
_FMP_TO_DISPLAY: Dict[str, str] = {v: k for k, v in _INDEX_FMP.items()}
for _s in _FOREX_SYMBOLS:
    _FMP_TO_DISPLAY[_s] = _s

# Interval map: our TF → FMP REST interval string
_TF_TO_FMP = {
    "1m":  "1min",
    "3m":  "5min",
    "5m":  "5min",
    "15m": "15min",
    "30m": "30min",
    "1h":  "1hour",
    "4h":  "4hour",
    "1d":  "1day",
}

_KLINE_CACHE: Dict[Tuple[str, str, int], Tuple[List[List[float]], datetime]] = {}
_KLINE_TTL = timedelta(seconds=20)

_FMP_STABLE_BASE = "https://financialmodelingprep.com/stable"
_FMP_LEGACY_BASE = "https://financialmodelingprep.com/api/v3"

# FMP lists gold/silver under both forex (XAUUSD) and commodities (GCUSD) tickers.
_METALS_FMP_ALIASES: Dict[str, List[str]] = {
    "XAUUSD": ["XAUUSD", "GCUSD"],
    "XAGUSD": ["XAGUSD", "SIUSD"],
}

_TF_MINUTES: Dict[str, int] = {
    "1min": 1, "5min": 5, "15min": 15, "30min": 30,
    "1hour": 60, "4hour": 240, "1day": 1440,
}

# Poll interval for REST price updates
_POLL_INTERVAL = 5


# ── Public API ───────────────────────────────────────────────────────────────

def get_price(symbol: str) -> Optional[float]:
    """Mid price from live cache, or None if stale / not yet received."""
    entry = _PRICE_CACHE.get(symbol.upper())
    if entry and (datetime.utcnow() - entry[1]) < _PRICE_TTL:
        return entry[0]
    return None


def is_live() -> bool:
    return _RUNNING


def symbol_count() -> int:
    return len(_PRICE_CACHE)


def cached_symbols() -> List[str]:
    return sorted(_PRICE_CACHE.keys())


def _fmp_api_key() -> str:
    for name in (
        "FMP_API_KEY",
        "FMP_KEY",
        "FINANCIAL_MODELING_PREP_API_KEY",
    ):
        val = (os.environ.get(name) or "").strip()
        if val:
            return val
    return ""


def _parse_fmp_bar_date(date_str: str) -> int:
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M:%S.%f"):
        try:
            return int(datetime.strptime(date_str, fmt).timestamp() * 1000)
        except ValueError:
            continue
    return int(datetime.fromisoformat(date_str.replace("Z", "+00:00")).timestamp() * 1000)


def _bars_from_fmp_json(data: object, limit: int) -> List[List[float]]:
    if not isinstance(data, list) or not data:
        return []
    rows: List[List[float]] = []
    for bar in reversed(data):
        try:
            rows.append([
                _parse_fmp_bar_date(bar["date"]),
                float(bar["open"]),
                float(bar["high"]),
                float(bar["low"]),
                float(bar["close"]),
                float(bar.get("volume") or 0),
            ])
        except Exception:
            continue
    return rows[-limit:]


def _fmp_symbol_candidates(sym: str) -> List[str]:
    sym_up = sym.upper()
    if sym_up in _METALS_FMP_ALIASES:
        return _METALS_FMP_ALIASES[sym_up]
    fmp_sym = _INDEX_FMP.get(sym_up, sym_up).replace("%5E", "^")
    return [fmp_sym]


def _resample_klines(
    rows: List[List[float]],
    src_minutes: int,
    dst_minutes: int,
    limit: int,
) -> List[List[float]]:
    if not rows or dst_minutes <= src_minutes or dst_minutes % src_minutes != 0:
        return rows[-limit:] if rows else []
    bucket_ms = dst_minutes * 60 * 1000
    buckets: Dict[int, List[List[float]]] = {}
    for bar in rows:
        key = int(bar[0]) // bucket_ms
        buckets.setdefault(key, []).append(bar)
    out: List[List[float]] = []
    for key in sorted(buckets):
        chunk = buckets[key]
        out.append([
            key * bucket_ms,
            float(chunk[0][1]),
            max(float(b[2]) for b in chunk),
            min(float(b[3]) for b in chunk),
            float(chunk[-1][4]),
            sum(float(b[5]) for b in chunk),
        ])
    return out[-limit:]


async def _fetch_fmp_chart_once(
    fmp_sym: str,
    fmp_interval: str,
    limit: int,
    api_key: str,
) -> List[List[float]]:
    minutes = _TF_MINUTES.get(fmp_interval, 15)
    days_back = max(7, int(limit * minutes / 1440) + 7)
    from_d = (datetime.utcnow() - timedelta(days=days_back)).strftime("%Y-%m-%d")
    to_d = datetime.utcnow().strftime("%Y-%m-%d")
    fmp_sym_url = fmp_sym.replace("^", "%5E")

    attempts = (
        (
            f"{_FMP_STABLE_BASE}/historical-chart/{fmp_interval}",
            {"symbol": fmp_sym, "apikey": api_key, "from": from_d, "to": to_d},
        ),
        (
            f"{_FMP_LEGACY_BASE}/historical-chart/{fmp_interval}/{fmp_sym_url}",
            {"apikey": api_key, "limit": str(limit)},
        ),
    )
    for url, params in attempts:
        data = None
        try:
            import httpx
            async with httpx.AsyncClient(timeout=15.0) as client:
                resp = await client.get(url, params=params)
                if resp.status_code != 200:
                    logger.debug(
                        f"[FMPFeed] klines {fmp_sym} {fmp_interval} HTTP {resp.status_code} ({url})"
                    )
                    continue
                data = resp.json()
        except Exception as e:
            logger.debug(f"[FMPFeed] klines httpx error {fmp_sym} {fmp_interval}: {e}")
            try:
                import aiohttp
                async with aiohttp.ClientSession() as session:
                    async with session.get(
                        url, params=params, timeout=aiohttp.ClientTimeout(total=15)
                    ) as resp:
                        if resp.status != 200:
                            continue
                        data = await resp.json(content_type=None)
            except Exception as e2:
                logger.debug(f"[FMPFeed] klines fetch error {fmp_sym} {fmp_interval}: {e2}")
                continue
        if data is None:
            continue
        if isinstance(data, dict) and data.get("Error Message"):
            logger.warning(
                f"[FMPFeed] klines {fmp_sym} {fmp_interval}: {data.get('Error Message')}"
            )
            continue
        rows = _bars_from_fmp_json(data, limit)
        if rows:
            return rows
    return []


async def get_klines(
    symbol: str,
    asset_class: str,
    timeframe: str = "15m",
    limit: int = 200,
) -> List[List[float]]:
    """
    OHLC bars from FMP REST historical-chart (stable API first, legacy v3 fallback).
    Returns [[ts_ms, o, h, l, c, v], ...] oldest-first, up to `limit` rows.
    """
    api_key = _fmp_api_key()
    if not api_key:
        return []

    sym = symbol.upper()
    fmp_interval = _TF_TO_FMP.get(timeframe, "15min")
    cache_key = (sym, fmp_interval, limit)
    now = datetime.utcnow()
    cached = _KLINE_CACHE.get(cache_key)
    if cached and (now - cached[1]) < _KLINE_TTL:
        return cached[0]

    rows: List[List[float]] = []
    for fmp_sym in _fmp_symbol_candidates(sym):
        rows = await _fetch_fmp_chart_once(fmp_sym, fmp_interval, limit, api_key)
        if rows:
            break

    # If the target interval is unavailable on this plan, build it from 1m bars.
    if not rows and fmp_interval != "1min":
        need_1m = min(max(limit * _TF_MINUTES.get(fmp_interval, 15), limit), 5000)
        one_min: List[List[float]] = []
        for fmp_sym in _fmp_symbol_candidates(sym):
            one_min = await _fetch_fmp_chart_once(fmp_sym, "1min", need_1m, api_key)
            if one_min:
                break
        if one_min:
            rows = _resample_klines(
                one_min,
                1,
                _TF_MINUTES.get(fmp_interval, 15),
                limit,
            )

    if rows:
        _KLINE_CACHE[cache_key] = (rows, now)
        logger.info(f"[FMPFeed] klines ok: {sym} {timeframe} → {len(rows)} bars")
    else:
        logger.warning(f"[FMPFeed] klines empty for {sym} {timeframe} (limit={limit})")
    return rows


# ── REST polling loop ─────────────────────────────────────────────────────────

async def _poll_forex():
    """Poll FMP /api/v3/fx for all forex + metals symbols."""
    api_key = _fmp_api_key()
    if not api_key:
        return

    symbols_csv = ",".join(_FOREX_SYMBOLS)
    url = f"https://financialmodelingprep.com/api/v3/fx?apikey={api_key}"
    try:
        import aiohttp
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=6)) as resp:
                if resp.status != 200:
                    return
                data = await resp.json(content_type=None)
    except Exception as e:
        logger.debug(f"[FMPFeed] forex poll error: {e}")
        return

    if not isinstance(data, list):
        return

    now = datetime.utcnow()
    for item in data:
        try:
            ticker = (item.get("ticker") or "").replace("/", "").upper()
            if ticker not in _FOREX_SYMBOLS:
                continue
            bid = item.get("bid")
            ask = item.get("ask")
            price = item.get("price") or item.get("changes")
            if bid and ask:
                mid = (float(bid) + float(ask)) / 2.0
            elif price:
                mid = float(price)
            else:
                continue
            _PRICE_CACHE[ticker] = (mid, now)
        except Exception:
            continue


async def _poll_indices():
    """Poll FMP /api/v3/quote for index symbols."""
    api_key = _fmp_api_key()
    if not api_key:
        return

    index_tickers = [v.replace("%5E", "^") for v in _INDEX_FMP.values()]
    symbols_csv = ",".join(t.replace("^", "%5E") for t in index_tickers)
    url = f"https://financialmodelingprep.com/api/v3/quote/{symbols_csv}?apikey={api_key}"
    try:
        import aiohttp
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=6)) as resp:
                if resp.status != 200:
                    return
                data = await resp.json(content_type=None)
    except Exception as e:
        logger.debug(f"[FMPFeed] index poll error: {e}")
        return

    if not isinstance(data, list):
        return

    now = datetime.utcnow()
    for item in data:
        try:
            fmp_sym = (item.get("symbol") or "").upper()
            price = item.get("price")
            if not price:
                continue
            # Map encoded key back to display name
            encoded = fmp_sym.replace("^", "%5E")
            display = _FMP_TO_DISPLAY.get(encoded) or _FMP_TO_DISPLAY.get(fmp_sym)
            if display:
                _PRICE_CACHE[display] = (float(price), now)
        except Exception:
            continue


async def _stream():
    global _RUNNING
    api_key = _fmp_api_key()
    if not api_key:
        logger.warning("[FMPFeed] FMP API key not set — real-time feed disabled")
        return

    logger.info(f"[FMPFeed] REST polling started — {len(_FOREX_SYMBOLS)} forex + {len(_INDEX_FMP)} indices every {_POLL_INTERVAL}s")
    _RUNNING = True
    errors = 0

    while True:
        try:
            await asyncio.gather(
                _poll_forex(),
                _poll_indices(),
                return_exceptions=True,
            )
            errors = 0
        except Exception as e:
            errors += 1
            logger.debug(f"[FMPFeed] poll cycle error ({errors}): {e}")

        if errors > 10:
            logger.warning("[FMPFeed] too many consecutive errors — pausing 60s")
            await asyncio.sleep(60)
            errors = 0
        else:
            await asyncio.sleep(_POLL_INTERVAL)


def start():
    """Schedule the polling loop as a background asyncio task."""
    asyncio.create_task(_stream())
    logger.info("[FMPFeed] background streaming task started")
