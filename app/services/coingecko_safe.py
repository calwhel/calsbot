"""Defensive CoinGecko JSON parsing — never crash on None / malformed bodies."""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Set

import httpx

logger = logging.getLogger(__name__)

_MARKETS_URL = "https://api.coingecko.com/api/v3/coins/markets"


def parse_coin_list(data: Any) -> List[Dict[str, Any]]:
    """Return a list of coin dicts; empty on None, dict, or non-list."""
    if data is None:
        return []
    if isinstance(data, dict):
        # Rate-limit / error payloads: {"status": {"error_code": 429, ...}}
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
    default_params = {
        "vs_currency": "usd",
        "order": "market_cap_desc",
        "per_page": 200,
        "page": 1,
        "sparkline": "false",
    }
    if params:
        default_params.update(params)
    try:
        resp = await client.get(_MARKETS_URL, params=default_params, timeout=timeout)
        if resp.status_code == 429:
            logger.warning("[CoinGecko] rate limited (429)")
            return []
        if resp.status_code != 200:
            logger.warning("[CoinGecko] HTTP %s", resp.status_code)
            return []
        return parse_coin_list(resp.json())
    except Exception as exc:
        logger.warning("[CoinGecko] fetch failed: %s", type(exc).__name__)
        return []


async def fetch_simple_price(
    client: httpx.AsyncClient,
    coin_id: str,
    *,
    timeout: float = 8.0,
) -> Optional[float]:
    """Spot USD price for a CoinGecko coin id (e.g. bitcoin)."""
    if not coin_id:
        return None
    try:
        resp = await client.get(
            "https://api.coingecko.com/api/v3/simple/price",
            params={"ids": coin_id, "vs_currencies": "usd"},
            timeout=timeout,
        )
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
