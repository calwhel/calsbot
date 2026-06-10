"""Shared Binance HTTP helpers — geo-block circuit breaker and rate-limit backoff."""
from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

_BINANCE_GEO_BLOCKED_UNTIL: Optional[datetime] = None
_BINANCE_GEO_WARNED = False
_BINANCE_ENABLED_LOGGED = False
_BINANCE_WEIGHT_BACKOFF_UNTIL: Optional[datetime] = None
_BINANCE_TIMEOUT_S = 5.0

_SPOT_BASE = "https://api.binance.com/api/v3"
_FUTURES_BASE = "https://fapi.binance.com/fapi/v1"


def binance_disabled() -> bool:
    """True when Binance must be skipped (env flag or recent 451/403)."""
    if os.environ.get("DISABLE_BINANCE", "").lower() in ("1", "true", "yes"):
        return True
    if _BINANCE_GEO_BLOCKED_UNTIL and datetime.utcnow() < _BINANCE_GEO_BLOCKED_UNTIL:
        return True
    return False


def _note_geo_block(status: int) -> None:
    global _BINANCE_GEO_BLOCKED_UNTIL, _BINANCE_GEO_WARNED
    _BINANCE_GEO_BLOCKED_UNTIL = datetime.utcnow() + timedelta(hours=6)
    if not _BINANCE_GEO_WARNED:
        _BINANCE_GEO_WARNED = True
        logger.warning(
            "[prices] Binance geo-blocked (HTTP %s) — skipping for 6h", status,
        )


def _note_weight(header_val: Optional[str]) -> None:
    global _BINANCE_WEIGHT_BACKOFF_UNTIL
    if not header_val:
        return
    try:
        w = int(str(header_val).split("/")[0].strip())
        if w > 1000:
            _BINANCE_WEIGHT_BACKOFF_UNTIL = datetime.utcnow() + timedelta(seconds=60)
            logger.debug("[binance] weight %d/min — backing off 60s", w)
    except Exception:
        pass


def _weight_backoff_active() -> bool:
    return (
        _BINANCE_WEIGHT_BACKOFF_UNTIL is not None
        and datetime.utcnow() < _BINANCE_WEIGHT_BACKOFF_UNTIL
    )


async def binance_get(
    http_client,
    url: str,
    params: Optional[Dict[str, Any]] = None,
    *,
    timeout_s: float = _BINANCE_TIMEOUT_S,
) -> Tuple[int, Optional[Any], Dict[str, str]]:
    """GET Binance endpoint. Returns (status, json, headers). Instant skip when blocked."""
    global _BINANCE_ENABLED_LOGGED
    if binance_disabled() or _weight_backoff_active():
        return 0, None, {}
    from app.services.prefetch_fast import provider_timeout_s

    timeout = provider_timeout_s(timeout_s)
    try:
        resp = await http_client.get(url, params=params or {}, timeout=timeout)
        headers = {k.lower(): v for k, v in (resp.headers or {}).items()}
        _note_weight(headers.get("x-mbx-used-weight"))
        if resp.status_code in (451, 403):
            _note_geo_block(resp.status_code)
            return resp.status_code, None, headers
        if resp.status_code == 200:
            if not _BINANCE_ENABLED_LOGGED:
                _BINANCE_ENABLED_LOGGED = True
                logger.info("[prices] Binance enabled, region check OK")
            try:
                return 200, resp.json(), headers
            except Exception:
                return 200, None, headers
        return resp.status_code, None, headers
    except Exception as exc:
        logger.debug("[binance] GET %s failed: %s", url, exc)
        return 0, None, {}


async def binance_spot_price(http_client, pair: str) -> Optional[float]:
    status, body, _ = await binance_get(
        http_client,
        f"{_SPOT_BASE}/ticker/price",
        {"symbol": pair},
    )
    if status == 200 and isinstance(body, dict):
        px = float(body.get("price") or 0)
        return px if px > 0 else None
    return None


async def binance_futures_klines(
    http_client, symbol: str, interval: str, limit: int,
) -> list:
    iv = interval.replace("60m", "1h").replace("120m", "2h").replace("240m", "4h")
    status, body, _ = await binance_get(
        http_client,
        f"{_FUTURES_BASE}/klines",
        {"symbol": symbol, "interval": iv, "limit": max(int(limit), 1)},
    )
    if status == 200 and isinstance(body, list) and body:
        return body
    return []


async def binance_spot_klines(
    http_client, symbol: str, interval: str, limit: int,
) -> list:
    iv = interval.replace("60m", "1h")
    status, body, _ = await binance_get(
        http_client,
        f"{_SPOT_BASE}/klines",
        {"symbol": symbol, "interval": iv, "limit": max(int(limit), 1) + 1},
    )
    if status == 200 and isinstance(body, list) and body:
        return body
    return []
