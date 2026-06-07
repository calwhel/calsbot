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
# httpx logs every request URL at INFO — floods logs during 429 storms.
logging.getLogger("httpx").setLevel(logging.WARNING)

# ── Price cache ──────────────────────────────────────────────────────────────
_PRICE_CACHE: Dict[str, Tuple[float, datetime]] = {}
_PRICE_TTL = timedelta(seconds=10)
_PRICE_STALE_TTL = timedelta(seconds=120)  # serve last-good during rate limits

_RUNNING = False
_feed_task: Optional[asyncio.Task] = None

# ── Rate-limit circuit breaker (FMP free/starter tiers are tight) ─────────────
_FMP_BACKOFF_UNTIL: Optional[datetime] = None
_FMP_RATE_LIMIT_STREAK = 0
_FMP_LAST_RATE_LIMIT_LOG: Optional[datetime] = None

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
_KLINE_TTL = timedelta(seconds=60)
_KLINE_STALE_TTL = timedelta(seconds=300)

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

# Poll interval for REST price updates (env: FMP_POLL_INTERVAL_SECONDS)
_POLL_INTERVAL = max(8, int(os.environ.get("FMP_POLL_INTERVAL_SECONDS", "15")))
_FMP_POLL_LOCK_ID = 708_110_005


# ── Public API ───────────────────────────────────────────────────────────────

def fmp_in_backoff() -> bool:
    """True when FMP returned 429 recently — callers should use fallbacks."""
    return _FMP_BACKOFF_UNTIL is not None and datetime.utcnow() < _FMP_BACKOFF_UNTIL


def fmp_backoff_remaining_seconds() -> int:
    if not fmp_in_backoff() or _FMP_BACKOFF_UNTIL is None:
        return 0
    return max(0, int((_FMP_BACKOFF_UNTIL - datetime.utcnow()).total_seconds()))


def _fmp_note_rate_limit() -> None:
    global _FMP_BACKOFF_UNTIL, _FMP_RATE_LIMIT_STREAK, _FMP_LAST_RATE_LIMIT_LOG
    _FMP_RATE_LIMIT_STREAK = min(_FMP_RATE_LIMIT_STREAK + 1, 5)
    wait = min(60 * (2 ** (_FMP_RATE_LIMIT_STREAK - 1)), 300)
    _FMP_BACKOFF_UNTIL = datetime.utcnow() + timedelta(seconds=wait)
    now = datetime.utcnow()
    if (
        _FMP_LAST_RATE_LIMIT_LOG is None
        or (now - _FMP_LAST_RATE_LIMIT_LOG).total_seconds() > 60
    ):
        _FMP_LAST_RATE_LIMIT_LOG = now
        logger.warning(
            f"[FMPFeed] rate limited (429) — backing off {wait}s; "
            "serving stale cache; cTrader/yfinance fallbacks stay active"
        )


def _fmp_note_success() -> None:
    global _FMP_RATE_LIMIT_STREAK, _FMP_BACKOFF_UNTIL
    _FMP_RATE_LIMIT_STREAK = 0
    _FMP_BACKOFF_UNTIL = None


def get_price(symbol: str) -> Optional[float]:
    """Mid price from live cache; shared DB store; stale up to _PRICE_STALE_TTL."""
    sym = symbol.upper()
    entry = _PRICE_CACHE.get(sym)
    if entry:
        age = datetime.utcnow() - entry[1]
        if age < _PRICE_TTL:
            return entry[0]
        if age < _PRICE_STALE_TTL:
            return entry[0]
    try:
        from app.services.spot_price_store import get_mid
        px = get_mid(sym, max_age_s=_PRICE_TTL.total_seconds())
        if px is not None:
            return px
    except Exception:
        pass
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


async def _fmp_http_get(url: str, params: dict, timeout: float = 8.0) -> Tuple[int, Optional[object]]:
    """Return (status_code, json_or_none)."""
    if fmp_in_backoff():
        return 0, None
    try:
        import httpx
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.get(url, params=params)
            if resp.status_code == 429:
                _fmp_note_rate_limit()
                return 429, None
            if resp.status_code != 200:
                return resp.status_code, None
            return 200, resp.json()
    except Exception as e:
        logger.debug(f"[FMPFeed] httpx GET failed {url}: {e}")
    try:
        import aiohttp
        async with aiohttp.ClientSession() as session:
            async with session.get(
                url, params=params, timeout=aiohttp.ClientTimeout(total=timeout)
            ) as resp:
                if resp.status == 429:
                    _fmp_note_rate_limit()
                    return 429, None
                if resp.status != 200:
                    return resp.status, None
                return 200, await resp.json(content_type=None)
    except Exception as e:
        logger.debug(f"[FMPFeed] aiohttp GET failed {url}: {e}")
    return 0, None


async def _fetch_fmp_chart_once(
    fmp_sym: str,
    fmp_interval: str,
    limit: int,
    api_key: str,
) -> List[List[float]]:
    if fmp_in_backoff():
        return []
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
        status, data = await _fmp_http_get(url, params, timeout=15.0)
        if status == 429:
            return []
        if status != 200 or data is None:
            logger.debug(
                f"[FMPFeed] klines {fmp_sym} {fmp_interval} HTTP {status} ({url})"
            )
            continue
        if isinstance(data, dict) and data.get("Error Message"):
            logger.warning(
                f"[FMPFeed] klines {fmp_sym} {fmp_interval}: {data.get('Error Message')}"
            )
            continue
        rows = _bars_from_fmp_json(data, limit)
        if rows:
            _fmp_note_success()
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
    if cached:
        age = now - cached[1]
        if age < _KLINE_TTL:
            return cached[0]
        if fmp_in_backoff() and age < _KLINE_STALE_TTL:
            return cached[0]

    if fmp_in_backoff():
        return cached[0] if cached else []

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

async def _fmp_get_json(url: str, params: dict) -> Optional[object]:
    """GET JSON from FMP — respects circuit breaker on 429."""
    status, data = await _fmp_http_get(url, params)
    if status == 200 and data is not None:
        return data
    return None


def _mid_from_quote_item(item: dict) -> Optional[float]:
    """Extract mid price from FMP stable or legacy quote objects."""
    bid = item.get("bid")
    ask = item.get("ask")
    if bid is not None and ask is not None:
        try:
            return (float(bid) + float(ask)) / 2.0
        except (TypeError, ValueError):
            pass
    for key in ("price", "previousClose", "open"):
        val = item.get(key)
        if val is not None:
            try:
                return float(val)
            except (TypeError, ValueError):
                continue
    return None


def _store_fmp_price(display_sym: str, mid: float, now: datetime) -> None:
    if mid > 0:
        sym = display_sym.upper()
        _PRICE_CACHE[sym] = (mid, now)
        try:
            from app.services.spot_price_store import upsert_tick
            upsert_tick(sym, mid=mid, source="fmp")
        except Exception:
            pass


async def _poll_forex():
    """One batched forex quote request per poll cycle."""
    api_key = _fmp_api_key()
    if not api_key or fmp_in_backoff():
        return

    now = datetime.utcnow()
    status, data = await _fmp_http_get(
        f"{_FMP_STABLE_BASE}/batch-forex-quotes",
        {"apikey": api_key},
    )
    if status == 429:
        return
    if data is None and not fmp_in_backoff():
        data = await _fmp_get_json(
            f"{_FMP_LEGACY_BASE}/fx",
            {"apikey": api_key},
        )
    if not isinstance(data, list):
        return

    got = 0
    for item in data:
        if not isinstance(item, dict):
            continue
        try:
            ticker = (
                (item.get("symbol") or item.get("ticker") or "")
                .replace("/", "")
                .upper()
            )
            if ticker not in _FOREX_SYMBOLS:
                continue
            mid = _mid_from_quote_item(item)
            if mid is not None:
                _store_fmp_price(ticker, mid, now)
                got += 1
        except Exception:
            continue
    if got:
        _fmp_note_success()
        return

    # Fallback: stable per-pair quotes when batch endpoint is empty/unavailable.
    for sym in ("EURUSD", "GBPUSD", "USDJPY", "XAUUSD"):
        status, data = await _fmp_http_get(
            f"{_FMP_STABLE_BASE}/quote",
            {"symbol": sym, "apikey": api_key},
        )
        if status != 200 or not isinstance(data, list) or not data:
            continue
        item = data[0] if isinstance(data[0], dict) else None
        if not item:
            continue
        mid = _mid_from_quote_item(item)
        if mid is not None:
            _store_fmp_price(sym, mid, now)
            got += 1
    if got:
        _fmp_note_success()
        logger.info(f"[FMPFeed] forex fallback quotes stored: {got} pairs")


async def _poll_indices():
    """One batched index quote request per poll cycle (not 6× per-symbol)."""
    api_key = _fmp_api_key()
    if not api_key or fmp_in_backoff():
        return

    now = datetime.utcnow()
    index_tickers = sorted({v.replace("%5E", "^") for v in _INDEX_FMP.values()})
    symbols_csv = ",".join(t.replace("^", "%5E") for t in index_tickers)
    status, data = await _fmp_http_get(
        f"{_FMP_LEGACY_BASE}/quote/{symbols_csv}",
        {"apikey": api_key},
    )
    if status == 429:
        return
    if not isinstance(data, list):
        return

    got = 0
    for item in data:
        if not isinstance(item, dict):
            continue
        try:
            fmp_sym = (item.get("symbol") or "").upper()
            mid = _mid_from_quote_item(item)
            if mid is None:
                continue
            encoded = fmp_sym.replace("^", "%5E")
            display = _FMP_TO_DISPLAY.get(encoded) or _FMP_TO_DISPLAY.get(fmp_sym)
            if display:
                _store_fmp_price(display, mid, now)
                got += 1
        except Exception:
            continue
    if got:
        _fmp_note_success()


def _ctrader_feed_active() -> bool:
    """True only when cTrader has fresh ticks — not merely 'subscribed'."""
    try:
        from app.services import ctrader_price_feed as _ctf
        syms = set(_ctf.cached_symbols())
        if len(syms) >= 8:
            return True
        return len(syms & {"EURUSD", "NAS100", "XAUUSD", "GBPUSD", "US30"}) >= 2
    except Exception:
        return False


def _acquire_fmp_poll_lock():
    """Hold a dedicated DB session for the poll cycle (session advisory lock)."""
    try:
        import psycopg2
        from app.config import settings
        conn = psycopg2.connect(settings.get_database_url())
        conn.autocommit = True
        cur = conn.cursor()
        cur.execute("SELECT pg_try_advisory_lock(%s)", (_FMP_POLL_LOCK_ID,))
        if not cur.fetchone()[0]:
            conn.close()
            return None
        return conn
    except Exception:
        return None


def _release_fmp_poll_lock(conn) -> None:
    if not conn:
        return
    try:
        cur = conn.cursor()
        cur.execute("SELECT pg_advisory_unlock(%s)", (_FMP_POLL_LOCK_ID,))
    except Exception:
        pass
    try:
        conn.close()
    except Exception:
        pass


async def _stream():
    global _RUNNING
    api_key = _fmp_api_key()
    if not api_key:
        logger.warning("[FMPFeed] FMP API key not set — real-time feed disabled")
        return

    logger.info(
        f"[FMPFeed] REST polling task — {len(_FOREX_SYMBOLS)} forex + "
        f"{len(_INDEX_FMP)} indices every {_POLL_INTERVAL}s (single global poller)"
    )
    _RUNNING = True
    errors = 0
    _ctrader_skip_logged = False

    try:
        while True:
            if not await asyncio.to_thread(_try_fmp_poll_lock):
                await asyncio.sleep(_POLL_INTERVAL)
                continue
            try:
                if _ctrader_feed_active():
                    if not _ctrader_skip_logged:
                        _ctrader_skip_logged = True
                        logger.info(
                            "[FMPFeed] cTrader live ticks active — pausing FMP polls "
                            "(FMP remains fallback if cTrader drops)"
                        )
                    await asyncio.sleep(max(120, _POLL_INTERVAL))
                    continue
                _ctrader_skip_logged = False

                if fmp_in_backoff():
                    await asyncio.sleep(max(_POLL_INTERVAL, fmp_backoff_remaining_seconds()))
                    continue
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
            finally:
                await asyncio.to_thread(_release_fmp_poll_lock)
    except asyncio.CancelledError:
        logger.info("[FMPFeed] polling task cancelled")
        raise
    finally:
        _RUNNING = False


def start():
    """Schedule the polling loop as a background asyncio task (executor worker only)."""
    global _feed_task
    if _feed_task and not _feed_task.done():
        return
    try:
        loop = asyncio.get_event_loop()
        _feed_task = loop.create_task(_stream())
        logger.info("[FMPFeed] background streaming task started")
    except Exception as e:
        logger.error(f"[FMPFeed] failed to start task: {e}")


def stop() -> None:
    """Cancel the background polling task when the executor lock is lost."""
    global _feed_task
    if _feed_task and not _feed_task.done():
        _feed_task.cancel()
        logger.info("[FMPFeed] background streaming task cancelled (lock lost)")
    _feed_task = None
