"""Bitunix futures PUBLIC market data — a reachable fallback for crypto price /
ticker / kline data when MEXC and Binance are unreachable.

Railway's datacenter egress is geo-blocked from Binance (HTTP 451) and times out
to MEXC, which left the crypto strategy executor with an empty symbol universe
and no klines — so crypto trades stopped firing after the Replit→Railway move.
Bitunix (the crypto execution venue) IS reachable from Railway, so we read its
public market-data endpoints and reshape them to the MEXC payload format every
existing crypto consumer already understands.

IMPORTANT: this module reads PUBLIC market data only. It does NOT touch any
Bitunix order/execution code.
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional

import httpx

logger = logging.getLogger(__name__)

_BASE = "https://fapi.bitunix.com/api/v1/futures/market"

# Bitunix accepts these interval tokens directly (verified). Map the few extra
# tokens our strategies may use to the nearest supported one so a kline fetch
# never fails on an unknown interval.
_SUPPORTED = {"1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "8h", "12h", "1d", "1w"}
_INTERVAL_ALIASES = {
    "60m": "1h", "120m": "2h", "180m": "4h", "240m": "4h",
    "1H": "1h", "4H": "4h", "1D": "1d", "1W": "1w",
}


def _map_interval(interval: str) -> str:
    iv = (interval or "").strip()
    if iv in _SUPPORTED:
        return iv
    return _INTERVAL_ALIASES.get(iv, "15m")


async def fetch_tickers(http_client: httpx.AsyncClient) -> List[Dict]:
    """All USDT-perp tickers in MEXC 24hr-ticker shape.

    Returns a list of dicts with the keys MEXC/Binance consumers expect:
    ``symbol``, ``lastPrice``, ``priceChangePercent``, ``quoteVolume``,
    ``highPrice``, ``lowPrice``, ``openPrice``. Bitunix has no 24h-change field,
    so we derive ``priceChangePercent`` from ``open`` vs ``last``.
    """
    try:
        resp = await http_client.get(f"{_BASE}/tickers", timeout=10)
        if resp.status_code != 200:
            return []
        payload = resp.json()
    except Exception as e:
        logger.debug(f"[bitunix-md] tickers fetch failed: {e}")
        return []

    rows = (payload or {}).get("data") or []
    out: List[Dict] = []
    for t in rows:
        try:
            sym = t.get("symbol", "")
            if not sym:
                continue
            last = float(t.get("lastPrice") or t.get("last") or 0) or 0.0
            open_ = float(t.get("open") or 0) or 0.0
            chg_pct = ((last - open_) / open_ * 100.0) if open_ > 0 else 0.0
            out.append({
                "symbol": sym,
                "lastPrice": str(last),
                "priceChangePercent": str(chg_pct),
                "quoteVolume": str(float(t.get("quoteVol") or 0) or 0.0),
                "highPrice": str(float(t.get("high") or 0) or 0.0),
                "lowPrice": str(float(t.get("low") or 0) or 0.0),
                "openPrice": str(open_),
            })
        except Exception:
            continue
    return out


async def fetch_ticker(http_client: httpx.AsyncClient, symbol: str) -> Optional[Dict]:
    """Single-symbol 24hr ticker (MEXC shape), or None."""
    try:
        resp = await http_client.get(
            f"{_BASE}/tickers", params={"symbols": symbol}, timeout=8
        )
        if resp.status_code != 200:
            return None
        rows = (resp.json() or {}).get("data") or []
    except Exception as e:
        logger.debug(f"[bitunix-md] ticker {symbol} failed: {e}")
        return None
    for t in rows:
        if t.get("symbol") == symbol:
            last = float(t.get("lastPrice") or t.get("last") or 0) or 0.0
            open_ = float(t.get("open") or 0) or 0.0
            if last <= 0:
                return None
            chg_pct = ((last - open_) / open_ * 100.0) if open_ > 0 else 0.0
            return {
                "symbol": symbol,
                "lastPrice": str(last),
                "priceChangePercent": str(chg_pct),
                "quoteVolume": str(float(t.get("quoteVol") or 0) or 0.0),
                "highPrice": str(float(t.get("high") or 0) or 0.0),
                "lowPrice": str(float(t.get("low") or 0) or 0.0),
                "openPrice": str(open_),
            }
    return None


async def fetch_klines(
    http_client: httpx.AsyncClient, symbol: str, interval: str, limit: int
) -> List[list]:
    """OHLCV bars in MEXC list shape: ``[openTime_ms, o, h, l, c, vol, closeTime, quoteVol]``.

    Bitunix returns newest-first dicts; we reshape to oldest-first lists so every
    consumer that does ``klines[-limit:]`` / ``float(row[4])`` keeps working.
    """
    iv = _map_interval(interval)
    try:
        resp = await http_client.get(
            f"{_BASE}/kline",
            params={"symbol": symbol, "interval": iv, "limit": max(int(limit), 1)},
            timeout=8,
        )
        if resp.status_code != 200:
            return []
        rows = (resp.json() or {}).get("data") or []
    except Exception as e:
        logger.debug(f"[bitunix-md] klines {symbol} {interval} failed: {e}")
        return []

    out: List[list] = []
    for k in rows:
        try:
            ts = int(k.get("time") or 0)
            out.append([
                ts,
                str(k.get("open")),
                str(k.get("high")),
                str(k.get("low")),
                str(k.get("close")),
                str(k.get("baseVol") or 0),
                ts,
                str(k.get("quoteVol") or 0),
            ])
        except Exception:
            continue
    # Bitunix returns newest-first; consumers expect oldest-first (MEXC order).
    out.sort(key=lambda r: r[0])
    return out
