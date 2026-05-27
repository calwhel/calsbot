"""
Stocks / forex / indices price + kline fetcher.

Price path:
  yfinance Ticker.fast_info — real-time mid price, no exchange delay for
  forex/indices. Cached 20 s. (FMP WebSocket attempted first but exits on
  plan-level 401 so effectively always falls straight to fast_info.)

Kline path:
  yfinance download() — intraday OHLC up to the current minute for forex/
  metals/indices (no 15-min delay for these asset classes). Cached 60 s.

Returned shapes mirror the crypto helpers so the strategy executor can
consume them without branching:
- get_price(symbol, asset_class) -> Optional[float]
- get_klines(symbol, asset_class, interval, limit) -> List[[ts, o, h, l, c, v]]
"""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

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
    Priority: FMP real-time feed → yfinance (15-min delayed fallback).
    """
    cls = normalize_asset_class(asset_class)
    if cls == ASSET_CLASS_CRYPTO:
        return None

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
    Priority: FMP REST historical-chart → yfinance.
    """
    cls = normalize_asset_class(asset_class)
    if cls == ASSET_CLASS_CRYPTO:
        return []

    # ── yfinance download() — intraday OHLC, near-real-time for forex/indices ──
    ticker = _resolve_ticker(cls, symbol)
    if not ticker:
        return []

    yf_interval, period = _TF_MAP.get(timeframe, ("15m", "60d"))
    key = (ticker, yf_interval, limit)
    now = datetime.utcnow()
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
