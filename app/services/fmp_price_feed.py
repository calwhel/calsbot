"""
Real-time forex + index price feed via FMP WebSocket.

No account linking required — uses the platform FMP_API_KEY environment
variable.  Streams live bid/ask ticks into a module-level cache.
tradfi_prices.py checks this cache first (TTL=10s), then falls back to
yfinance.  Kline (OHLC) fetches also route through FMP's REST
historical-chart endpoint for up-to-the-minute candles.
"""
from __future__ import annotations

import asyncio
import json
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

# Index display → FMP WebSocket ticker
_INDEX_FMP: Dict[str, str] = {
    "SPX":  "^GSPC",
    "NDX":  "^NDX",
    "DJI":  "^DJI",
    "VIX":  "^VIX",
    "DAX":  "^GDAXI",
    "FTSE": "^FTSE",
}

# Reverse map: FMP ticker → display symbol
_FMP_TO_DISPLAY: Dict[str, str] = {v: k for k, v in _INDEX_FMP.items()}
for _s in _FOREX_SYMBOLS:
    _FMP_TO_DISPLAY[_s] = _s

_ALL_SUBSCRIBE = _FOREX_SYMBOLS + list(_INDEX_FMP.values())

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
_KLINE_TTL = timedelta(seconds=60)


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
    fmp_sym = _INDEX_FMP.get(sym, sym)
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


# ── WebSocket streaming loop ─────────────────────────────────────────────────

async def _stream():
    global _RUNNING
    import ssl
    import websockets

    api_key = os.environ.get("FMP_API_KEY", "")
    if not api_key:
        logger.warning("[FMPFeed] FMP_API_KEY not set — real-time feed disabled")
        return

    # Replit sandbox lacks a full CA bundle; disable cert verification for this
    # read-only market-data connection (no user credentials in transit).
    _ssl_ctx = ssl.create_default_context()
    _ssl_ctx.check_hostname = False
    _ssl_ctx.verify_mode = ssl.CERT_NONE

    url = "wss://websockets.financialmodelingprep.com"
    retry_delay = 5

    while True:
        try:
            async with websockets.connect(
                url, ssl=_ssl_ctx, ping_interval=20, ping_timeout=15
            ) as ws:
                # Authenticate
                await ws.send(json.dumps({"event": "login", "data": {"apiKey": api_key}}))
                try:
                    resp = await asyncio.wait_for(ws.recv(), timeout=10)
                    logger.info(f"[FMPFeed] login response: {str(resp)[:120]}")
                    try:
                        resp_data = json.loads(resp)
                        if resp_data.get("status") == 401:
                            logger.warning(
                                "[FMPFeed] API key unauthorised for WebSocket "
                                "(plan upgrade required) — feed disabled"
                            )
                            return   # permanent exit; no retry
                    except Exception:
                        pass
                except asyncio.TimeoutError:
                    logger.warning("[FMPFeed] login timeout — retrying")
                    continue

                # Subscribe
                for sym in _ALL_SUBSCRIBE:
                    await ws.send(
                        json.dumps({"event": "subscribe", "data": {"ticker": sym}})
                    )

                logger.info(
                    f"[FMPFeed] subscribed to {len(_ALL_SUBSCRIBE)} symbols — streaming"
                )
                _RUNNING = True
                retry_delay = 5

                async for raw in ws:
                    try:
                        msg = json.loads(raw)
                        fmp_sym = msg.get("s", "")
                        if not fmp_sym:
                            continue
                        ap = msg.get("ap") or msg.get("askPrice")
                        bp = msg.get("bp") or msg.get("bidPrice")
                        lp = msg.get("lp") or msg.get("lastPrice")
                        if ap and bp:
                            price = (float(ap) + float(bp)) / 2.0
                        elif lp:
                            price = float(lp)
                        else:
                            continue
                        display = _FMP_TO_DISPLAY.get(fmp_sym, fmp_sym)
                        _PRICE_CACHE[display] = (price, datetime.utcnow())
                    except Exception:
                        pass

        except Exception as e:
            _RUNNING = False
            logger.warning(f"[FMPFeed] disconnected: {e} — retry in {retry_delay}s")
            await asyncio.sleep(retry_delay)
            retry_delay = min(retry_delay * 2, 300)


def start():
    """Schedule the streaming loop as a background asyncio task."""
    asyncio.create_task(_stream())
    logger.info("[FMPFeed] background streaming task started")
