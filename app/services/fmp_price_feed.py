"""
Real-time forex + index price feed via FMP REST polling.

Polls FMP's /api/v3/quote endpoint every 5 seconds for all tracked forex
and index symbols. Results are cached with a 10s TTL.

tradfi_prices.py checks this cache first (TTL=10s), then falls back to
yfinance.  Kline (OHLC) fetches also route through FMP's REST
historical-chart endpoint for up-to-the-minute candles.
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

# Index display → FMP quote ticker
_INDEX_FMP: Dict[str, str] = {
    "SPX":  "%5EGSPC",
    "NDX":  "%5ENDX",
    "DJI":  "%5EDJI",
    "VIX":  "%5EVIX",
    "DAX":  "%5EGDAXI",
    "FTSE": "%5EFTSE",
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


async def get_klines(
    symbol: str,
    asset_class: str,
    timeframe: str = "15m",
    limit: int = 200,
) -> List[List[float]]:
    """
    OHLC bars from FMP REST historical-chart.
    Returns [[ts_ms, o, h, l, c, v], ...] oldest-first, up to `limit` rows.
    """
    api_key = os.environ.get("FMP_API_KEY", "")
    if not api_key:
        return []

    sym = symbol.upper()
    fmp_sym = _INDEX_FMP.get(sym, sym).replace("%5E", "^")
    fmp_interval = _TF_TO_FMP.get(timeframe, "15min")
    cache_key = (sym, fmp_interval, limit)
    now = datetime.utcnow()
    cached = _KLINE_CACHE.get(cache_key)
    if cached and (now - cached[1]) < _KLINE_TTL:
        return cached[0]

    fmp_sym_url = fmp_sym.replace("^", "%5E")
    url = (
        f"https://financialmodelingprep.com/api/v3/historical-chart"
        f"/{fmp_interval}/{fmp_sym_url}?apikey={api_key}&limit={limit}"
    )
    try:
        import aiohttp
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=8)) as resp:
                if resp.status != 200:
                    logger.debug(f"[FMPFeed] klines {sym} HTTP {resp.status}")
                    return []
                data = await resp.json(content_type=None)
    except Exception as e:
        logger.debug(f"[FMPFeed] klines fetch error {sym}: {e}")
        return []

    if not isinstance(data, list) or not data:
        return []

    rows: List[List[float]] = []
    for bar in reversed(data):
        try:
            ts = int(
                datetime.strptime(bar["date"], "%Y-%m-%d %H:%M:%S").timestamp() * 1000
            )
            rows.append([
                ts,
                float(bar["open"]),
                float(bar["high"]),
                float(bar["low"]),
                float(bar["close"]),
                float(bar.get("volume") or 0),
            ])
        except Exception:
            continue

    rows = rows[-limit:]
    _KLINE_CACHE[cache_key] = (rows, now)
    return rows


# ── REST polling loop ─────────────────────────────────────────────────────────

async def _poll_forex():
    """Poll FMP /api/v3/fx for all forex + metals symbols."""
    api_key = os.environ.get("FMP_API_KEY", "")
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
    api_key = os.environ.get("FMP_API_KEY", "")
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
    api_key = os.environ.get("FMP_API_KEY", "")
    if not api_key:
        logger.warning("[FMPFeed] FMP_API_KEY not set — real-time feed disabled")
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
