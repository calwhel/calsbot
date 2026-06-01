"""
Stocks / forex / indices price + kline fetcher.

Price path:
  Metals (XAUUSD/XAGUSD): Binance spot API (XAUUSDT/XAGUSDT) — real-time,
  no futures contango premium, same format as crypto pairs.
  Others: FMP real-time WebSocket → yfinance fast_info fallback.

Kline path:
  Metals (XAUUSD/XAGUSD): Binance spot API — returns closed candles only
  (we fetch limit+1 bars and drop the still-forming last bar).
  Others: yfinance download() — intraday OHLC for forex/indices.

Returned shapes mirror the crypto helpers so the strategy executor can
consume them without branching:
- get_price(symbol, asset_class) -> Optional[float]
- get_klines(symbol, asset_class, interval, limit) -> List[[ts, o, h, l, c, v]]
"""
from __future__ import annotations

import asyncio
import logging
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

_FMP_API_KEY: str = os.environ.get("FMP_API_KEY", "")

from app.services.asset_classes import (
    ASSET_CLASS_CRYPTO,
    normalize_asset_class,
    yf_ticker,
)

logger = logging.getLogger(__name__)

# yfinance logs at ERROR level for every missing ticker / temporarily-delisted
# symbol even though we already handle the None return gracefully.  Suppress
# to CRITICAL so these don't flood production logs.
logging.getLogger("yfinance").setLevel(logging.CRITICAL)

_PRICE_CACHE: Dict[str, Tuple[float, datetime]] = {}
_PRICE_TTL = timedelta(seconds=20)

_KLINE_CACHE: Dict[Tuple[str, str, int], Tuple[List[List[float]], datetime]] = {}
_KLINE_TTL = timedelta(seconds=20)

# Map our wizard timeframes to yfinance intervals + a sensible history window.
# yfinance restricts intraday lookback: 1m → 7d max, 5m/15m → 60d, 1h → 730d.
_TF_MAP: Dict[str, Tuple[str, str]] = {
    "1m":  ("1m",  "5d"),
    "3m":  ("5m",  "30d"),    # 3m not supported — bucket to 5m
    "5m":  ("5m",  "30d"),
    "15m": ("15m", "60d"),
    "30m": ("30m", "60d"),
    "1h":  ("60m", "180d"),
    "4h":  ("60m", "365d"),   # 4h not supported — re-aggregate from 60m
    "1d":  ("1d",  "5y"),
}

# ── Binance spot metals ──────────────────────────────────────────────────────
# XAUUSD and XAGUSD trade on Binance spot as XAUUSDT / XAGUSDT.
# Benefits over yfinance GC=F (gold futures):
#   • No contango premium — spot price matches forex broker quotes
#   • Same kline format as crypto (direct list, no pandas wrangling)
#   • Fetching limit+1 bars and dropping the last gives only CLOSED candles,
#     preventing signals from firing on a still-forming bar.
_METALS_BINANCE_MAP: Dict[str, str] = {
    "XAUUSD": "XAUUSDT",
    "XAGUSD": "XAGUSDT",
}
_BINANCE_SPOT_BASE = "https://api.binance.com/api/v3"

# Binance interval names match our internal convention directly.
_BINANCE_INTERVAL_MAP: Dict[str, str] = {
    "1m": "1m", "3m": "3m", "5m": "5m", "15m": "15m",
    "30m": "30m", "1h": "1h", "4h": "4h", "1d": "1d",
}


def _resolve_ticker(asset_class: str, symbol: str) -> Optional[str]:
    if normalize_asset_class(asset_class) == ASSET_CLASS_CRYPTO:
        return None
    return yf_ticker(asset_class, symbol)


def _yf_fast_price_blocking(ticker: str) -> Optional[float]:
    """
    yfinance Ticker.fast_info gives real-time mid prices for forex and indices
    (no 15-min exchange delay — that only applies to the download() path for
    US equities).  Run in a thread so we don't block the event loop.
    """
    import yfinance as yf
    try:
        fi = yf.Ticker(ticker).fast_info
        px = fi.get("last_price") or fi.get("lastPrice")
        return float(px) if px else None
    except Exception:
        return None


def _yf_download_blocking(ticker: str, interval: str, period: str):
    """yfinance is sync — run inside a thread executor."""
    import yfinance as yf
    return yf.download(
        tickers=ticker,
        interval=interval,
        period=period,
        progress=False,
        auto_adjust=False,
        prepost=False,
        threads=False,
    )


async def get_price(symbol: str, asset_class: str) -> Optional[float]:
    """
    Latest trade price.
    Metals: Binance spot (XAUUSDT/XAGUSDT) → FMP → yfinance fallback.
    Others: FMP real-time feed → yfinance fast_info fallback.
    """
    cls = normalize_asset_class(asset_class)
    if cls == ASSET_CLASS_CRYPTO:
        return None

    # ── 0a. cTrader live spot feed — matches the broker that fills orders ─────
    # FP Markets / cTrader is where forex/metal/index orders actually execute,
    # so its spot feed is the ONLY source that matches the user's chart and
    # fills in real time. (FMP's legacy endpoints are dead and Binance metals
    # are geo-restricted / a different instrument, so both lag or mismatch.)
    # Returns None when there's no fresh tick (cold feed / no connected
    # account) → falls through to the legacy sources below.
    try:
        from app.services import ctrader_price_feed as _ctf
        _cpx = _ctf.get_price(symbol)
        if _cpx is not None:
            return _cpx
    except Exception:
        pass

    # ── 0. Binance spot — metals only (XAUUSDT / XAGUSDT) ────────────────────
    _bn_sym = _METALS_BINANCE_MAP.get(symbol.upper())
    if _bn_sym:
        _cache_key = f"binance:{_bn_sym}"
        now = datetime.utcnow()
        cached = _PRICE_CACHE.get(_cache_key)
        if cached and (now - cached[1]) < _PRICE_TTL:
            return cached[0]
        try:
            import httpx
            async with httpx.AsyncClient(timeout=3.0) as _c:
                r = await _c.get(
                    f"{_BINANCE_SPOT_BASE}/ticker/price",
                    params={"symbol": _bn_sym},
                )
            if r.status_code == 200:
                px = float(r.json().get("price", 0))
                if px:
                    _PRICE_CACHE[_cache_key] = (px, now)
                    return px
        except Exception as _be:
            logger.debug(f"[tradfi] Binance spot price failed for {_bn_sym}: {_be}")
        # fall through to FMP / yfinance below

    # ── 1. FMP real-time WebSocket feed (by-the-second, always active) ────────
    try:
        from app.services.fmp_price_feed import get_price as _fmp_price
        px = _fmp_price(symbol)
        if px is not None:
            return px
    except Exception:
        pass

    # ── 2. yfinance fast_info — real-time mid price (no exchange delay) ────────
    ticker = _resolve_ticker(cls, symbol)
    if not ticker:
        return None

    now = datetime.utcnow()
    cached = _PRICE_CACHE.get(ticker)
    if cached and (now - cached[1]) < _PRICE_TTL:
        return cached[0]

    try:
        price = await asyncio.to_thread(_yf_fast_price_blocking, ticker)
        if price:
            _PRICE_CACHE[ticker] = (price, now)
            return price
        logger.warning(f"[tradfi] fast_info returned no price for {ticker}")
    except Exception as e:
        logger.warning(f"[tradfi] fast_info failed for {ticker}: {e}")

    return None


async def get_klines(
    symbol: str,
    asset_class: str,
    timeframe: str = "15m",
    limit: int = 200,
) -> List[List[float]]:
    """
    Return up to `limit` OHLC bars in MEXC-shape: [[ts_ms, o, h, l, c, v], ...]
    Metals: Binance spot API (closed candles only — last forming bar stripped).
    Others: yfinance download().
    """
    cls = normalize_asset_class(asset_class)
    if cls == ASSET_CLASS_CRYPTO:
        return []

    now = datetime.utcnow()

    # ── cTrader trendbars — same OHLC the broker's chart draws ───────────────
    # Primary candle source for forex/metals/indices so paper evaluation and
    # signal candles match the broker that fills orders. Returns [] when the
    # symbol isn't broker-tracked or no account is connected → falls through.
    try:
        from app.services import ctrader_price_feed as _ctf
        # Bound the cTrader-first attempt (open+app auth+account auth+trendbar
        # read, ×up to 3 host/refresh tries) so a transient cTrader hiccup
        # can't stall us before the Binance/yfinance fallbacks below.
        _crows = await asyncio.wait_for(
            _ctf.get_klines(symbol, asset_class, timeframe, limit),
            timeout=6.0,
        )
        if _crows:
            return _crows
    except Exception:
        pass

    # ── Binance spot metals path (XAUUSDT / XAGUSDT) ─────────────────────────
    _bn_sym = _METALS_BINANCE_MAP.get(symbol.upper())
    if _bn_sym:
        _bn_interval = _BINANCE_INTERVAL_MAP.get(timeframe, timeframe)
        key = (_bn_sym, _bn_interval, limit)
        cached = _KLINE_CACHE.get(key)
        if cached and (now - cached[1]) < _KLINE_TTL:
            return cached[0]
        try:
            import httpx
            # Fetch limit+1 bars so after dropping the still-forming last bar
            # we still have `limit` complete, closed candles.
            async with httpx.AsyncClient(timeout=5.0) as _c:
                r = await _c.get(
                    f"{_BINANCE_SPOT_BASE}/klines",
                    params={
                        "symbol":   _bn_sym,
                        "interval": _bn_interval,
                        "limit":    limit + 1,
                    },
                )
            if r.status_code == 200:
                raw = r.json()
                if raw and isinstance(raw, list):
                    # Drop the last bar — it is the still-forming current candle.
                    # All preceding bars are fully closed.
                    complete = raw[:-1] if len(raw) > 1 else raw
                    rows: List[List[float]] = []
                    for bar in complete[-limit:]:
                        try:
                            rows.append([
                                int(bar[0]),      # open_time ms
                                float(bar[1]),    # open
                                float(bar[2]),    # high
                                float(bar[3]),    # low
                                float(bar[4]),    # close
                                float(bar[5]),    # volume
                            ])
                        except Exception:
                            continue
                    if rows:
                        _KLINE_CACHE[key] = (rows, now)
                        logger.info(
                            f"[tradfi] klines ok (Binance spot): "
                            f"{_bn_sym} {_bn_interval} → {len(rows)} bars"
                        )
                        return rows
        except Exception as _be:
            logger.debug(
                f"[tradfi] Binance spot klines failed for {_bn_sym} "
                f"({_bn_interval}): {_be}"
            )
        # fall through to yfinance below

    # ── FMP 1-minute REST — primary source for forex/metals paper evaluation ─────
    # FMP provides real broker mid-prices for forex with no 7-day cap, making it
    # better than yfinance for paper trade OHLC evaluation. Used only for 1m
    # requests (paper position monitor). Falls through to yfinance on failure.
    if timeframe == "1m" and _FMP_API_KEY:
        try:
            import httpx as _httpx
            async with _httpx.AsyncClient(timeout=10.0) as _fc:
                _fr = await _fc.get(
                    f"https://financialmodelingprep.com/api/v3/historical-chart/1min/{symbol.upper()}",
                    params={"apikey": _FMP_API_KEY, "limit": limit + 5},
                )
            if _fr.status_code == 200:
                _fmp_data = _fr.json()
                if _fmp_data and isinstance(_fmp_data, list):
                    # FMP returns newest-first — reverse to oldest-first
                    rows: List[List[float]] = []
                    for bar in reversed(_fmp_data[-(limit + 5):]):
                        try:
                            ts_ms = int(
                                datetime.fromisoformat(bar["date"]).timestamp() * 1000
                            )
                            rows.append([
                                ts_ms,
                                float(bar["open"]), float(bar["high"]),
                                float(bar["low"]),  float(bar["close"]),
                                float(bar.get("volume", 0) or 0),
                            ])
                        except Exception:
                            continue
                    if rows:
                        rows = rows[-limit:]
                        _KLINE_CACHE[(symbol.upper(), "1m_fmp", limit)] = (rows, now)
                        logger.info(
                            f"[tradfi] klines ok (FMP 1min): {symbol.upper()} → {len(rows)} bars"
                        )
                        return rows
        except Exception as _fe:
            logger.debug(f"[tradfi] FMP 1min failed {symbol}: {_fe}")
        # fall through to yfinance

    # ── yfinance download() — forex/indices/stocks ────────────────────────────
    ticker = _resolve_ticker(cls, symbol)
    if not ticker:
        return []

    yf_interval, period = _TF_MAP.get(timeframe, ("15m", "60d"))
    key = (ticker, yf_interval, limit)
    cached = _KLINE_CACHE.get(key)
    if cached and (now - cached[1]) < _KLINE_TTL:
        return cached[0]

    try:
        df = await asyncio.to_thread(_yf_download_blocking, ticker, yf_interval, period)
        if df is None or df.empty:
            return []
        if hasattr(df.columns, "nlevels") and df.columns.nlevels > 1:
            df.columns = df.columns.get_level_values(0)
        df = df.tail(limit)
        rows: List[List[float]] = []
        for ts, row in df.iterrows():
            try:
                rows.append([
                    int(ts.timestamp() * 1000),
                    float(row["Open"]),
                    float(row["High"]),
                    float(row["Low"]),
                    float(row["Close"]),
                    float(row.get("Volume", 0) or 0),
                ])
            except Exception:
                continue
        _KLINE_CACHE[key] = (rows, now)
        logger.info(f"[tradfi] klines ok: {ticker} {yf_interval} → {len(rows)} bars")
        return rows
    except Exception as e:
        logger.warning(f"[tradfi] klines failed for {ticker} ({yf_interval}): {e}")
        return []


async def warm_cache(symbols: List[Tuple[str, str]]) -> None:
    """Best-effort prefetch — kicks off price fetches in parallel."""
    await asyncio.gather(
        *[get_price(sym, cls) for sym, cls in symbols],
        return_exceptions=True,
    )
