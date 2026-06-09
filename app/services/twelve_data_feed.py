"""
Twelve Data REST spot quotes — narrow fallback for fire-time confirmation only.

Free Basic tier: 8 requests/minute, 800/day. We only call during the London+US
overlap window (default 09:00–18:00 UTC) and only when explicitly allowed by
the caller (confirm_entry_price), never from the spot primer loop.
"""
from __future__ import annotations

import logging
import os
import time
from collections import deque
from datetime import date, datetime, timedelta
from typing import Deque, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

_BASE = "https://api.twelvedata.com"
_PRICE_CACHE: Dict[str, Tuple[float, datetime]] = {}
_PRICE_TTL = timedelta(seconds=30)

_WINDOW_START_UTC = int(os.environ.get("TWELVE_DATA_WINDOW_START_UTC", "9"))
_WINDOW_END_UTC = int(os.environ.get("TWELVE_DATA_WINDOW_END_UTC", "18"))
_MAX_PER_MINUTE = int(os.environ.get("TWELVE_DATA_MAX_PER_MINUTE", "8"))
_MAX_PER_DAY = int(os.environ.get("TWELVE_DATA_MAX_PER_DAY", "800"))

_minute_calls: Deque[float] = deque()
_day_count: int = 0
_day_key: Optional[date] = None
_backoff_until: float = 0.0

# canonical symbol → Twelve Data ticker
_SYMBOL_MAP: Dict[str, str] = {
    "EURUSD": "EUR/USD", "GBPUSD": "GBP/USD", "USDJPY": "USD/JPY",
    "AUDUSD": "AUD/USD", "USDCAD": "USD/CAD", "USDCHF": "USD/CHF",
    "NZDUSD": "NZD/USD", "EURJPY": "EUR/JPY", "GBPJPY": "GBP/JPY",
    "EURGBP": "EUR/GBP",
    "XAUUSD": "XAU/USD", "XAGUSD": "XAG/USD",
    "NAS100": "NDX", "NDX": "NDX", "US100": "NDX",
    "SPX500": "GSPC", "SPX": "GSPC", "US500": "GSPC",
    "US30": "DJI", "DJI": "DJI",
    "GER40": "GDAXI", "DAX": "GDAXI", "DE40": "GDAXI",
    "UK100": "FTSE", "FTSE": "FTSE",
}


def _api_key() -> str:
    for name in ("TWELVE_DATA_API_KEY", "TWELVEDATA_API_KEY", "TWELVE_DATA_KEY"):
        val = (os.environ.get(name) or "").strip()
        if val:
            return val
    return ""


def is_enabled() -> bool:
    return (
        bool(_api_key())
        and os.environ.get("TWELVE_DATA_ENABLED", "1").strip() not in ("0", "false", "no")
    )


def is_active_window(now_utc: Optional[datetime] = None) -> bool:
    """True during London + US session overlap (default 09:00–18:00 UTC)."""
    now = now_utc or datetime.utcnow()
    hour = now.hour
    if _WINDOW_START_UTC <= _WINDOW_END_UTC:
        return _WINDOW_START_UTC <= hour < _WINDOW_END_UTC
    # Wraps midnight, e.g. 22→6
    return hour >= _WINDOW_START_UTC or hour < _WINDOW_END_UTC


def _reset_day_if_needed(now: datetime) -> None:
    global _day_count, _day_key
    today = now.date()
    if _day_key != today:
        _day_key = today
        _day_count = 0


def _prune_minute_window() -> None:
    cutoff = time.monotonic() - 60.0
    while _minute_calls and _minute_calls[0] < cutoff:
        _minute_calls.popleft()


def can_request(now_utc: Optional[datetime] = None) -> bool:
    if not is_enabled() or not is_active_window(now_utc):
        return False
    if time.monotonic() < _backoff_until:
        return False
    _reset_day_if_needed(now_utc or datetime.utcnow())
    _prune_minute_window()
    if _day_count >= _MAX_PER_DAY:
        return False
    if len(_minute_calls) >= _MAX_PER_MINUTE:
        return False
    return True


def _note_request() -> None:
    global _day_count
    _minute_calls.append(time.monotonic())
    _day_count += 1


def _note_rate_limit() -> None:
    global _backoff_until
    _backoff_until = time.monotonic() + 60.0
    logger.warning("[TwelveData] rate limited — pausing 60s")


def to_twelve_data_symbol(symbol: str, asset_class: str = "forex") -> Optional[str]:
    sym = symbol.upper().replace("/", "").replace("-", "")
    if sym in _SYMBOL_MAP:
        return _SYMBOL_MAP[sym]
    try:
        from app.services.index_symbols import normalize_index_symbol, yf_ticker_for_index
        from app.services.asset_classes import normalize_asset_class, ASSET_CLASS_INDEX

        if normalize_asset_class(asset_class) == ASSET_CLASS_INDEX or sym.endswith("100"):
            canon = normalize_index_symbol(sym)
            yf = yf_ticker_for_index(canon)
            if yf:
                return yf.lstrip("^")
    except Exception:
        pass
    if len(sym) == 6 and sym.isalpha():
        return f"{sym[:3]}/{sym[3:]}"
    return None


def read_cached_quote(symbol: str, max_age: float) -> Optional[float]:
    sym = symbol.upper().replace("/", "").replace("-", "")
    entry = _PRICE_CACHE.get(sym)
    if not entry:
        return None
    px, ts = entry
    if datetime.utcnow() - ts > timedelta(seconds=max_age):
        return None
    return px if px > 0 else None


def _store_price(symbol: str, mid: float) -> None:
    sym = symbol.upper().replace("/", "").replace("-", "")
    now = datetime.utcnow()
    _PRICE_CACHE[sym] = (mid, now)
    try:
        from app.services.spot_price_store import upsert_tick
        upsert_tick(sym, mid=mid, source="twelvedata")
    except Exception:
        pass


async def fetch_quote(
    symbol: str,
    asset_class: str = "forex",
    *,
    max_age: float = 15.0,
) -> Optional[float]:
    """
    On-demand /price quote. Returns None outside 09:00–18:00 UTC or when capped.
    """
    sym = symbol.upper().replace("/", "").replace("-", "")
    if not can_request():
        return read_cached_quote(sym, max_age)

    cached = read_cached_quote(sym, max_age)
    if cached is not None:
        return cached

    td_sym = to_twelve_data_symbol(sym, asset_class)
    if not td_sym:
        return None

    api_key = _api_key()
    if not api_key:
        return None

    url = f"{_BASE}/price"
    params = {"symbol": td_sym, "apikey": api_key}

    try:
        import httpx
        async with httpx.AsyncClient(timeout=6.0) as client:
            resp = await client.get(url, params=params)
        if resp.status_code == 429:
            _note_rate_limit()
            return None
        if resp.status_code != 200:
            logger.debug(f"[TwelveData] HTTP {resp.status_code} for {td_sym}")
            return None
        body = resp.json()
        if not isinstance(body, dict) or body.get("status") == "error":
            msg = (body or {}).get("message") if isinstance(body, dict) else None
            if msg:
                logger.debug(f"[TwelveData] error for {td_sym}: {msg}")
            return None
        price = body.get("price")
        if price is None:
            return None
        mid = float(price)
        if mid <= 0:
            return None
        _note_request()
        _store_price(sym, mid)
        return mid
    except Exception as exc:
        logger.debug(f"[TwelveData] fetch failed {td_sym}: {exc}")
        return None


def status() -> Dict[str, object]:
    _prune_minute_window()
    _reset_day_if_needed(datetime.utcnow())
    return {
        "enabled": is_enabled(),
        "active_window": is_active_window(),
        "window_utc": f"{_WINDOW_START_UTC:02d}:00-{_WINDOW_END_UTC:02d}:00",
        "calls_last_minute": len(_minute_calls),
        "calls_today": _day_count,
        "max_per_minute": _MAX_PER_MINUTE,
        "max_per_day": _MAX_PER_DAY,
    }
