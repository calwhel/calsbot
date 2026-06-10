"""Alpha Vantage FX klines — scanner fallback after cTrader / Twelve Data / Yahoo."""
from __future__ import annotations

import logging
import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

_BASE = "https://www.alphavantage.co/query"

_TF_MAP: Dict[str, Tuple[str, str]] = {
    "1m": ("1min", "compact"),
    "5m": ("5min", "compact"),
    "15m": ("15min", "compact"),
    "30m": ("30min", "compact"),
    "1h": ("60min", "compact"),
    "60m": ("60min", "compact"),
    "4h": ("60min", "full"),
    "1d": ("daily", "compact"),
}


def _api_key() -> str:
    for name in ("ALPHA_VANTAGE_API_KEY", "ALPHAVANTAGE_API_KEY", "AV_API_KEY"):
        val = (os.environ.get(name) or "").strip()
        if val:
            return val
    return ""


def is_enabled() -> bool:
    return (
        bool(_api_key())
        and os.environ.get("ALPHA_VANTAGE_ENABLED", "1").strip() not in ("0", "false", "no")
    )


def _split_forex_pair(symbol: str) -> Optional[Tuple[str, str]]:
    sym = symbol.upper().replace("/", "").replace("-", "")
    if len(sym) == 6 and sym.isalpha():
        return sym[:3], sym[3:]
    return None


def _parse_time_series(body: dict, limit: int) -> List[List[float]]:
    if not isinstance(body, dict):
        return []
    if body.get("Note") or body.get("Information"):
        logger.debug("[AlphaVantage] rate/info: %s", str(body.get("Note") or body.get("Information"))[:80])
        return []
    if "Error Message" in body:
        logger.debug("[AlphaVantage] error: %s", str(body["Error Message"])[:80])
        return []

    series_key = next((k for k in body if "Time Series" in k), None)
    if not series_key:
        return []
    series = body.get(series_key)
    if not isinstance(series, dict):
        return []

    rows: List[List[float]] = []
    for ts_str, bar in sorted(series.items()):
        if not isinstance(bar, dict):
            continue
        try:
            dt = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
            ts_ms = int(dt.timestamp() * 1000)
            o = float(bar.get("1. open") or bar.get("open") or 0)
            h = float(bar.get("2. high") or bar.get("high") or 0)
            l = float(bar.get("3. low") or bar.get("low") or 0)
            c = float(bar.get("4. close") or bar.get("close") or 0)
            if c <= 0:
                continue
            rows.append([ts_ms, o, h, l, c, 0.0])
        except Exception:
            continue
    rows.sort(key=lambda r: r[0])
    return rows[-limit:] if rows else []


async def fetch_klines(
    symbol: str,
    timeframe: str,
    limit: int,
) -> List[List[float]]:
    """FX intraday/daily candles for major pairs."""
    if not is_enabled():
        return []
    pair = _split_forex_pair(symbol)
    if not pair:
        return []
    from_sym, to_sym = pair
    interval, outputsize = _TF_MAP.get(timeframe, ("15min", "compact"))

    params = {
        "function": "FX_INTRADAY" if interval != "daily" else "FX_DAILY",
        "from_symbol": from_sym,
        "to_symbol": to_sym,
        "apikey": _api_key(),
        "outputsize": outputsize,
    }
    if interval != "daily":
        params["interval"] = interval

    try:
        import httpx
        async with httpx.AsyncClient(timeout=12.0) as client:
            resp = await client.get(_BASE, params=params)
        if resp.status_code != 200:
            return []
        rows = _parse_time_series(resp.json(), limit)
        if rows:
            logger.info(
                "[AlphaVantage] klines ok %s %s → %d bars",
                symbol, timeframe, len(rows),
            )
        return rows
    except Exception as exc:
        logger.debug("[AlphaVantage] fetch failed %s %s: %s", symbol, timeframe, exc)
        return []
