"""Defensive CoinGecko JSON parsing — global rate limiter + safe parsing."""
from __future__ import annotations

import logging
import os
import time
from typing import Any, Callable, Dict, List, Optional, Set, TypeVar

import httpx

logger = logging.getLogger(__name__)

_MARKETS_URL = "https://api.coingecko.com/api/v3/coins/markets"

_T = TypeVar("_T")

# Free tier ~10–30 calls/min — default 20/min with instant skip while paused.
_MAX_PER_MIN = max(5, int(os.environ.get("COINGECKO_MAX_PER_MIN", "20")))
_PAUSE_DEFAULT_S = float(os.environ.get("COINGECKO_PAUSE_S", "90"))

_req_times: List[float] = []
_paused_until: float = 0.0
_last_pause_log: float = 0.0


def coingecko_paused() -> bool:
    return time.monotonic() < _paused_until


def _record_request() -> bool:
    """Return False when over budget (caller should skip without HTTP)."""
    global _paused_until, _last_pause_log
    now = time.monotonic()
    if now < _paused_until:
        return False
    window = [t for t in _req_times if now - t < 60.0]
    _req_times.clear()
    _req_times.extend(window)
    if len(window) >= _MAX_PER_MIN:
        _paused_until = now + _PAUSE_DEFAULT_S
        if now - _last_pause_log >= _PAUSE_DEFAULT_S:
            _last_pause_log = now
            logger.warning(
                "[CoinGecko] rate budget exhausted (%s/min) — pausing %.0fs",
                _MAX_PER_MIN,
                _PAUSE_DEFAULT_S,
            )
        return False
    _req_times.append(now)
    return True


def _pause_on_429(retry_after_s: float = _PAUSE_DEFAULT_S) -> None:
    global _paused_until, _last_pause_log
    now = time.monotonic()
    _paused_until = max(_paused_until, now + retry_after_s)
    if now - _last_pause_log >= retry_after_s:
        _last_pause_log = now
        logger.warning(
            "[CoinGecko] rate limited (429) — pausing %.0fs", retry_after_s,
        )


async def coingecko_limited(
    fn: Callable[[], Any],
    *,
    skip_label: str = "request",
) -> Optional[_T]:
    """Run *fn* only when global budget allows; instant-skip while paused."""
    if not _record_request():
        return None
    try:
        return await fn()
    except Exception:
        raise


def parse_coin_list(data: Any) -> List[Dict[str, Any]]:
    """Return a list of coin dicts; empty on None, dict, or non-list."""
    if data is None:
        return []
    if isinstance(data, dict):
        err = data.get("status") or data.get("error")
        if err:
            logger.debug("[CoinGecko] API error payload: %s", str(err)[:120])
        return []
    if not isinstance(data, list):
        logger.debug("[CoinGecko] expected list, got %s", type(data).__name__)
        return []
    return [c for c in data if isinstance(c, dict)]


def symbols_from_markets(data: Any) -> Set[str]:
    out: Set[str] = set()
    for coin in parse_coin_list(data):
        sym = coin.get("symbol")
        if isinstance(sym, str) and sym.strip():
            out.add(sym.upper())
    return out


async def fetch_markets(
    client: httpx.AsyncClient,
    *,
    params: Optional[Dict[str, Any]] = None,
    timeout: float = 15.0,
) -> List[Dict[str, Any]]:
    """Fetch /coins/markets with safe parsing. Never raises on bad JSON."""
    if coingecko_paused():
        return []

    default_params = {
        "vs_currency": "usd",
        "order": "market_cap_desc",
        "per_page": 200,
        "page": 1,
        "sparkline": "false",
    }
    if params:
        default_params.update(params)

    if not _record_request():
        return []

    try:
        resp = await client.get(_MARKETS_URL, params=default_params, timeout=timeout)
        if resp.status_code == 429:
            _pause_on_429()
            return []
        if resp.status_code != 200:
            logger.debug("[CoinGecko] HTTP %s", resp.status_code)
            return []
        return parse_coin_list(resp.json())
    except Exception as exc:
        logger.debug("[CoinGecko] fetch failed: %s", type(exc).__name__)
        return []


async def fetch_simple_price(
    client: httpx.AsyncClient,
    coin_id: str,
    *,
    timeout: float = 8.0,
) -> Optional[float]:
    """Spot USD price for a CoinGecko coin id (e.g. bitcoin)."""
    if not coin_id or coingecko_paused():
        return None
    if not _record_request():
        return None
    try:
        resp = await client.get(
            "https://api.coingecko.com/api/v3/simple/price",
            params={"ids": coin_id, "vs_currencies": "usd"},
            timeout=timeout,
        )
        if resp.status_code == 429:
            _pause_on_429()
            return None
        if resp.status_code != 200:
            return None
        body = resp.json()
        if not isinstance(body, dict):
            return None
        entry = body.get(coin_id)
        if not isinstance(entry, dict):
            return None
        px = entry.get("usd")
        return float(px) if px is not None and float(px) > 0 else None
    except Exception:
        return None
