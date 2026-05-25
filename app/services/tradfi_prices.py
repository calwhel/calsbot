"""
Stocks / forex / indices price + kline fetcher backed by yfinance.

Returned shapes mirror the existing crypto helpers so the strategy executor
can consume them without branching every TA call:
- get_price(symbol, asset_class) -> Optional[float]
- get_klines(symbol, asset_class, interval, limit) -> List[[ts, o, h, l, c, v]]

We cache aggressively (price 20s, klines 60s) — yfinance scrapes Yahoo, so
rate-limit etiquette matters even though there's no documented quota.
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

_PRICE_CACHE: Dict[str, Tuple[float, datetime]] = {}
_PRICE_TTL = timedelta(seconds=20)

_KLINE_CACHE: Dict[Tuple[str, str, int], Tuple[List[List[float]], datetime]] = {}
_KLINE_TTL = timedelta(seconds=60)

# Map our wizard timeframes to yfinance intervals + a sensible history window.
# yfinance restricts intraday lookback: 1m → 7d max, 5m/15m → 60d, 1h → 730d.
_TF_MAP: Dict[str, Tuple[str, str]] = {
    "1m":  ("1m",  "5d"),
    "3m":  ("5m",  "30d"),    # 3m not supported — bucket to 5m
    "5m":  ("5m",  "30d"),
    "15m": ("15m", "60d"),
    "30m": ("30m", "60d"),
    "1h":  ("60m", "180d"),
    "4h":  ("60m", "365d"),   # 4h not supported — re-aggregate from 60m client-side if ever needed
    "1d":  ("1d",  "5y"),
}


def _resolve_ticker(asset_class: str, symbol: str) -> Optional[str]:
    if normalize_asset_class(asset_class) == ASSET_CLASS_CRYPTO:
        return None  # caller must use crypto path
    return yf_ticker(asset_class, symbol)


def _yf_download_blocking(ticker: str, interval: str, period: str):
    """yfinance is sync — run inside a thread executor."""
    import yfinance as yf  # imported here to keep cold-start light
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
    Priority: cTrader live feed → yfinance (15-min delayed fallback).
    """
    cls = normalize_asset_class(asset_class)
    if cls == ASSET_CLASS_CRYPTO:
        return None

    # ── 1. Try cTrader live feed (real-time, same source as FP Markets) ──────
    try:
        from app.services.ctrader_price_feed import get_price as _ct_price
        ct_px = _ct_price(symbol)
        if ct_px is not None:
            return ct_px
    except Exception:
        pass

    # ── 2. Fallback: yfinance ─────────────────────────────────────────────────
    ticker = _resolve_ticker(cls, symbol)
    if not ticker:
        return None

    now = datetime.utcnow()
    cached = _PRICE_CACHE.get(ticker)
    if cached and (now - cached[1]) < _PRICE_TTL:
        return cached[0]

    try:
        df = await asyncio.to_thread(_yf_download_blocking, ticker, "1m", "1d")
        if df is None or df.empty:
            return None
        price = float(df["Close"].iloc[-1])
        _PRICE_CACHE[ticker] = (price, now)
        return price
    except Exception as e:
        logger.debug(f"tradfi price fetch failed for {ticker}: {e}")
        return None


async def get_klines(
    symbol: str,
    asset_class: str,
    timeframe: str = "15m",
    limit: int = 200,
) -> List[List[float]]:
    """
    Return up to `limit` OHLC bars in MEXC-shape: [[ts_ms, o, h, l, c, v], ...]
    Priority: cTrader trendbar feed → yfinance (15-min delayed fallback).
    """
    cls = normalize_asset_class(asset_class)
    if cls == ASSET_CLASS_CRYPTO:
        return []

    # ── 1. Try cTrader trendbar feed (FP Markets exact data) ─────────────────
    try:
        from app.services.ctrader_price_feed import get_klines as _ct_klines
        ct_rows = await _ct_klines(symbol, asset_class, timeframe, limit)
        if ct_rows:
            return ct_rows
    except Exception:
        pass

    # ── 2. Fallback: yfinance ─────────────────────────────────────────────────
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
        # yfinance returns MultiIndex columns if multiple tickers — we only ever
        # pass one, but defensively normalize.
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
        return rows
    except Exception as e:
        logger.debug(f"tradfi klines fetch failed for {ticker} ({yf_interval}): {e}")
        return []


async def warm_cache(symbols: List[Tuple[str, str]]) -> None:
    """Best-effort prefetch — kicks off price fetches in parallel."""
    await asyncio.gather(
        *[get_price(sym, cls) for sym, cls in symbols],
        return_exceptions=True,
    )
