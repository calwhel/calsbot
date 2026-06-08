"""
Stocks / forex / indices price + kline fetcher.

Price path:
  Metals (XAUUSD/XAGUSD): Binance spot API (XAUUSDT/XAGUSDT) — real-time,
  no futures contango premium, same format as crypto pairs.
  Others: FMP real-time WebSocket → yfinance fast_info fallback.

Kline path:
  Metals (XAUUSD/XAGUSD): cTrader trendbars → FMP → Binance spot → Yahoo.
  Forex live: cTrader → Yahoo chart → FMP → yfinance.
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

from app.services.asset_classes import (
    ASSET_CLASS_CRYPTO,
    ASSET_CLASS_FOREX,
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

# Single-flight registry — collapses concurrent identical candle requests into
# ONE underlying fetch. The forex executor evaluates up to ~40 strategies per
# cycle, many on the same symbol+timeframe (e.g. ~28 on XAUUSD); without this,
# each strategy independently fired its own (slow, ~1-3s) yfinance download when
# the 20s cache was cold/expired, stretching a forex scan cycle to ~2 minutes.
# Keyed by the function inputs so any caller asking for the same bars while a
# fetch is in flight awaits that fetch instead of starting a duplicate one.
_KLINE_INFLIGHT: Dict[Tuple[str, str, str, int], "asyncio.Future"] = {}

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

# Yahoo v8 chart API — reliable server-side fallback (no yfinance lib required).
_YAHOO_CHART_TF: Dict[str, Tuple[str, str]] = {
    "1m":  ("1m",  "5d"),
    "5m":  ("5m",  "60d"),
    "15m": ("15m", "60d"),
    "30m": ("30m", "60d"),
    "1h":  ("60m", "730d"),
    "4h":  ("60m", "730d"),
    "1d":  ("1d",  "5y"),
}
_METALS_YAHOO_TICKER: Dict[str, str] = {
    "XAUUSD": "GC=F",
    "XAGUSD": "SI=F",
}


def _resolve_ticker(asset_class: str, symbol: str) -> Optional[str]:
    cls = normalize_asset_class(asset_class)
    if cls == ASSET_CLASS_CRYPTO:
        return None
    if cls == "index":
        try:
            from app.services.index_symbols import normalize_index_symbol, yf_ticker_for_index
            return yf_ticker_for_index(normalize_index_symbol(symbol))
        except Exception:
            pass
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


def _env_fmp_api_key() -> str:
    try:
        from app.services.fmp_price_feed import _fmp_api_key
        return _fmp_api_key()
    except Exception:
        for name in ("FMP_API_KEY", "FMP_KEY", "FINANCIAL_MODELING_PREP_API_KEY"):
            val = (os.environ.get(name) or "").strip()
            if val:
                return val
        return ""


async def _fetch_yahoo_chart_klines(
    yahoo_ticker: str,
    timeframe: str,
    limit: int,
    *,
    http_timeout_s: float = 20.0,
) -> List[List[float]]:
    """OHLC from Yahoo Finance chart API — works on Railway without yfinance."""
    interval, range_ = _YAHOO_CHART_TF.get(timeframe, ("15m", "60d"))
    cache_key = ("yahoo", yahoo_ticker, interval, range_, limit)
    now = datetime.utcnow()
    cached = _KLINE_CACHE.get(cache_key)
    if cached and (now - cached[1]) < _KLINE_TTL:
        return cached[0]
    try:
        import httpx
        async with httpx.AsyncClient(
            timeout=http_timeout_s,
            headers={"User-Agent": "Mozilla/5.0 (compatible; TradeHub/1.0)"},
        ) as client:
            resp = await client.get(
                f"https://query1.finance.yahoo.com/v8/finance/chart/{yahoo_ticker}",
                params={"interval": interval, "range": range_},
            )
        if resp.status_code != 200:
            logger.warning(
                f"[tradfi] Yahoo chart {yahoo_ticker} {timeframe} HTTP {resp.status_code}"
            )
            return []
        payload = resp.json()
        result = (payload.get("chart") or {}).get("result") or []
        if not result:
            return []
        block = result[0]
        timestamps = block.get("timestamp") or []
        quotes = ((block.get("indicators") or {}).get("quote") or [{}])[0]
        opens = quotes.get("open") or []
        highs = quotes.get("high") or []
        lows = quotes.get("low") or []
        closes = quotes.get("close") or []
        vols = quotes.get("volume") or []
        rows: List[List[float]] = []
        for i, ts in enumerate(timestamps):
            try:
                if ts is None:
                    continue
                o, h, l, c = opens[i], highs[i], lows[i], closes[i]
                if o is None or h is None or l is None or c is None:
                    continue
                rows.append([
                    int(ts) * 1000,
                    float(o), float(h), float(l), float(c),
                    float(vols[i] or 0),
                ])
            except Exception:
                continue
        rows = rows[-limit:]
        if rows:
            _KLINE_CACHE[cache_key] = (rows, now)
            logger.info(
                f"[tradfi] klines ok (Yahoo chart): {yahoo_ticker} {timeframe} → {len(rows)} bars"
            )
        return rows
    except Exception as e:
        logger.warning(f"[tradfi] Yahoo chart failed {yahoo_ticker} {timeframe}: {e}")
        return []


async def fetch_index_scan_candles(
    symbol: str,
    timeframe: str,
    limit: int,
    user_id: Optional[int] = None,
) -> List[List[float]]:
    """
    Multi-source fetch for index backtest scanners (NASDAQ, S&P, etc.).
    Prefers linked cTrader demo/live candles when available.
    """
    try:
        from app.services.index_symbols import normalize_index_symbol, yf_ticker_for_index
        sym = normalize_index_symbol(symbol)
        yahoo_ticker = yf_ticker_for_index(sym)
    except Exception:
        sym = symbol.upper()
        yahoo_ticker = None

    best: List[List[float]] = []

    async def _keep(rows: List[List[float]], label: str) -> None:
        nonlocal best
        if rows and len(rows) > len(best):
            best = rows
            logger.info(f"[tradfi] index-scan best={label} {sym} {timeframe} → {len(rows)} bars")

    if user_id:
        await _keep(
            await _fetch_ctrader_klines(sym, "index", timeframe, min(limit, 500), user_id=user_id),
            "ctrader-user",
        )
        if len(best) >= 120:
            return best[-limit:]

    if yahoo_ticker:
        await _keep(await _fetch_yahoo_chart_klines(yahoo_ticker, timeframe, limit), "yahoo")
        if len(best) >= 120:
            return best[-limit:]

    if _env_fmp_api_key():
        try:
            from app.services.fmp_price_feed import get_klines as _fmp_klines
            await _keep(await _fmp_klines(sym, "index", timeframe, limit), "fmp")
        except Exception:
            pass
        if len(best) >= 120:
            return best[-limit:]

    await _keep(
        await _fetch_ctrader_klines(sym, "index", timeframe, min(limit, 500), user_id=user_id),
        "ctrader",
    )
    if len(best) >= 120:
        return best[-limit:]

    tradfi_rows = await _get_klines_impl(
        sym, "index", timeframe, limit, for_backtest=True, ctrader_user_id=user_id,
    )
    await _keep(tradfi_rows, "tradfi-chain")
    return best[-limit:] if best else []


async def fetch_metal_scan_candles(
    symbol: str,
    timeframe: str,
    limit: int,
    user_id: Optional[int] = None,
) -> List[List[float]]:
    """
    Aggressive multi-source fetch for gold/silver backtest scanners.
    When a user has cTrader linked, broker trendbars are preferred (demo/live
    matched). Otherwise Yahoo → FMP → cTrader (app creds) → tradfi chain.
    """
    sym = symbol.upper()
    yahoo_ticker = _METALS_YAHOO_TICKER.get(sym)
    best: List[List[float]] = []

    async def _keep(rows: List[List[float]], label: str) -> None:
        nonlocal best
        if rows and len(rows) > len(best):
            best = rows
            logger.info(f"[tradfi] metal-scan best={label} {sym} {timeframe} → {len(rows)} bars")

    # Linked cTrader account → broker OHLC first (matches demo/live charts).
    if user_id:
        await _keep(
            await _fetch_ctrader_klines(sym, "forex", timeframe, min(limit, 500), user_id=user_id),
            "ctrader-user",
        )
        if len(best) >= 120:
            return best[-limit:]

    if yahoo_ticker:
        await _keep(await _fetch_yahoo_chart_klines(yahoo_ticker, timeframe, limit), "yahoo")
        if len(best) >= 120:
            return best[-limit:]

    if _env_fmp_api_key():
        await _keep(
            await _fetch_fmp_metals_klines(sym, "forex", timeframe, limit),
            "fmp",
        )
        if len(best) >= 120:
            return best[-limit:]

    await _keep(
        await _fetch_ctrader_klines(sym, "forex", timeframe, min(limit, 500), user_id=user_id),
        "ctrader",
    )
    if len(best) >= 120:
        return best[-limit:]

    tradfi_rows = await _get_klines_impl(
        sym, "forex", timeframe, limit, for_backtest=True, ctrader_user_id=user_id,
    )
    await _keep(tradfi_rows, "tradfi-chain")
    return best[-limit:] if best else []


async def fetch_forex_scan_candles(
    symbol: str,
    timeframe: str,
    limit: int,
    user_id: Optional[int] = None,
) -> List[List[float]]:
    """
    Multi-source fetch for major FX pair backtest scanners (EURUSD, GBPUSD, …).
    Prefers linked cTrader demo/live candles when available.
    """
    sym = symbol.upper().replace("/", "").replace("-", "")
    yahoo_ticker = yf_ticker("forex", sym)
    best: List[List[float]] = []

    async def _keep(rows: List[List[float]], label: str) -> None:
        nonlocal best
        if rows and len(rows) > len(best):
            best = rows
            logger.info(f"[tradfi] forex-scan best={label} {sym} {timeframe} → {len(rows)} bars")

    if user_id:
        await _keep(
            await _fetch_ctrader_klines(sym, "forex", timeframe, min(limit, 500), user_id=user_id),
            "ctrader-user",
        )
        if len(best) >= 120:
            return best[-limit:]

    if yahoo_ticker:
        await _keep(await _fetch_yahoo_chart_klines(yahoo_ticker, timeframe, limit), "yahoo")
        if len(best) >= 120:
            return best[-limit:]

    if _env_fmp_api_key():
        try:
            from app.services.fmp_price_feed import get_klines as _fmp_klines
            await _keep(await _fmp_klines(sym, "forex", timeframe, limit), "fmp")
        except Exception:
            pass
        if len(best) >= 120:
            return best[-limit:]

    await _keep(
        await _fetch_ctrader_klines(sym, "forex", timeframe, min(limit, 500), user_id=user_id),
        "ctrader",
    )
    if len(best) >= 120:
        return best[-limit:]

    tradfi_rows = await _get_klines_impl(
        sym, "forex", timeframe, limit, for_backtest=True, ctrader_user_id=user_id,
    )
    await _keep(tradfi_rows, "tradfi-chain")
    return best[-limit:] if best else []


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
    # fills in real time. Returns None when there's no fresh tick (cold feed /
    # no connected account) → falls through to the legacy sources below.
    try:
        from app.services import ctrader_price_feed as _ctf
        _cpx = _ctf.get_price(symbol)
        if _cpx is not None:
            return _cpx
    except Exception:
        pass

    # ── 0b. Shared metals ticks (dedicated poller: binance / coinbase / kraken) ─
    if symbol.upper() in _METALS_BINANCE_MAP:
        try:
            from app.services.spot_price_store import get_mid as _store_mid
            _spx = _store_mid(symbol.upper(), max_age_s=20.0)
            if _spx is not None:
                return _spx
        except Exception:
            pass
        try:
            from app.services.metals_spot_feed import get_price as _metals_px
            _mpx = _metals_px(symbol)
            if _mpx is not None:
                return _mpx
        except Exception:
            pass
        try:
            from app.services.metals_spot_feed import fetch_now as _metals_fetch
            _mpx = await _metals_fetch(symbol)
            if _mpx is not None:
                return _mpx
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
                body = r.json()
                if isinstance(body, dict) and body.get("code") not in (None, 0):
                    raise ValueError(body.get("msg") or "binance error")
                px = float(body.get("price", 0))
                if px:
                    _PRICE_CACHE[_cache_key] = (px, now)
                    try:
                        from app.services.spot_price_store import upsert_tick
                        upsert_tick(symbol.upper(), mid=px, source="binance")
                    except Exception:
                        pass
                    return px
        except Exception as _be:
            logger.debug(f"[tradfi] Binance spot price failed for {_bn_sym}: {_be}")
        # Coinbase works from US/Railway where Binance spot is geo-blocked.
        _cb_pair = {"XAUUSDT": "XAU-USD", "XAGUSDT": "XAG-USD"}.get(_bn_sym)
        if _cb_pair:
            try:
                import httpx
                async with httpx.AsyncClient(timeout=4.0) as _c:
                    r = await _c.get(f"https://api.coinbase.com/v2/prices/{_cb_pair}/spot")
                if r.status_code == 200:
                    amount = ((r.json() or {}).get("data") or {}).get("amount")
                    px = float(amount) if amount else 0.0
                    if px > 0:
                        _PRICE_CACHE[_cache_key] = (px, now)
                        try:
                            from app.services.spot_price_store import upsert_tick
                            upsert_tick(symbol.upper(), mid=px, source="coinbase")
                        except Exception:
                            pass
                        return px
            except Exception as _ce:
                logger.debug(f"[tradfi] Coinbase spot price failed for {_cb_pair}: {_ce}")
        # fall through to FMP / yfinance below

    # ── 1. FMP real-time WebSocket feed (by-the-second, always active) ────────
    try:
        from app.services.fmp_price_feed import get_price as _fmp_price
        px = _fmp_price(symbol)
        if px is not None:
            return px
    except Exception:
        pass

    # Metals must NOT fall through to yfinance GC=F — that's gold FUTURES, a
    # DIFFERENT instrument from XAUUSD spot (carries a contango premium of
    # several dollars), so it mismatches the cTrader broker quote and can
    # falsely trigger tight TP/SL. Better to return None (skip the tick) than
    # feed a wrong-instrument price into live/paper trade evaluation.
    if symbol.upper() in _METALS_BINANCE_MAP:
        return None

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
            try:
                from app.services.spot_price_store import upsert_tick
                upsert_tick(symbol.upper(), mid=float(price), source="yfinance")
            except Exception:
                pass
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
    for_backtest: bool = False,
    ctrader_user_id: Optional[int] = None,
    max_wait_s: Optional[float] = None,
) -> List[List[float]]:
    """
    Return up to `limit` OHLC bars in MEXC-shape: [[ts_ms, o, h, l, c, v], ...]

    Single-flight wrapper around `_get_klines_impl`: concurrent callers asking
    for the same (symbol, asset_class, timeframe, limit) while a fetch is in
    flight await that one fetch instead of each launching a duplicate (slow)
    download. The 20s `_KLINE_CACHE` already dedupes *sequential* requests; this
    dedupes the *simultaneous* ones (the executor's per-cycle thundering herd).
    """
    key = (
        symbol.upper(),
        normalize_asset_class(asset_class),
        timeframe,
        int(limit),
        bool(for_backtest),
        int(ctrader_user_id) if ctrader_user_id is not None else -1,
    )
    try:
        running = asyncio.get_running_loop()
    except RuntimeError:
        running = None

    if running is not None:
        inflight = _KLINE_INFLIGHT.get(key)
        # Only piggy-back on an in-flight fetch that belongs to THIS loop and is
        # still pending — guards against the (per-process single-loop) invariant
        # ever being violated.
        if (
            inflight is not None
            and not inflight.done()
            and inflight.get_loop() is running
        ):
            try:
                return await inflight
            except Exception:
                return []

        fut = running.create_future()
        _KLINE_INFLIGHT[key] = fut
        try:
            async def _run():
                return await _get_klines_impl(
                    symbol,
                    asset_class,
                    timeframe,
                    limit,
                    for_backtest=for_backtest,
                    ctrader_user_id=ctrader_user_id,
                )

            if max_wait_s and max_wait_s > 0:
                result = await asyncio.wait_for(_run(), timeout=max_wait_s)
            else:
                result = await _run()
            if not fut.done():
                fut.set_result(result)
            return result
        except asyncio.CancelledError:
            # Producer was cancelled mid-fetch — never leave piggy-backed waiters
            # hanging on an unresolved future. Cancel it so they wake promptly.
            if not fut.done():
                fut.cancel()
            raise
        except asyncio.TimeoutError:
            logger.warning(
                f"[tradfi] get_klines budget hit for {symbol.upper()} "
                f"{timeframe} (max_wait_s={max_wait_s})"
            )
            if not fut.done():
                fut.set_result([])
            return []
        except Exception:
            if not fut.done():
                fut.set_result([])
            return []
        finally:
            if _KLINE_INFLIGHT.get(key) is fut:
                _KLINE_INFLIGHT.pop(key, None)

    # No running loop (shouldn't happen for an async caller) — fetch directly.
    coro = _get_klines_impl(
        symbol,
        asset_class,
        timeframe,
        limit,
        for_backtest=for_backtest,
        ctrader_user_id=ctrader_user_id,
    )
    if max_wait_s and max_wait_s > 0:
        return await asyncio.wait_for(coro, timeout=max_wait_s)
    return await coro


async def _fetch_fmp_metals_klines(
    symbol: str,
    asset_class: str,
    timeframe: str,
    limit: int,
) -> List[List[float]]:
    if not _env_fmp_api_key():
        return []
    try:
        from app.services.fmp_price_feed import get_klines as _fmp_klines
        rows = await _fmp_klines(symbol, asset_class, timeframe, limit)
        if rows:
            logger.info(
                f"[tradfi] klines ok (FMP): {symbol.upper()} {timeframe} → {len(rows)} bars"
            )
        return rows
    except Exception as fe:
        logger.warning(f"[tradfi] FMP klines failed {symbol} {timeframe}: {fe}")
        return []


async def _fetch_ctrader_klines(
    symbol: str,
    asset_class: str,
    timeframe: str,
    limit: int,
    user_id: Optional[int] = None,
) -> List[List[float]]:
    try:
        from app.services import ctrader_price_feed as _ctf
        # Keep broker trendbar waits short — a slow miss + Yahoo fallback must
        # stay under gunicorn/Railway ~30s worker budgets on HTTP workers.
        timeout = min(10.0, 5.0 + limit / 200.0)
        return await asyncio.wait_for(
            _ctf.get_klines(symbol, asset_class, timeframe, limit, user_id=user_id),
            timeout=timeout,
        )
    except asyncio.TimeoutError:
        logger.info(
            f"[tradfi] cTrader klines timeout {symbol.upper()} {timeframe} "
            f"(limit={limit})"
        )
        return []
    except Exception as exc:
        logger.debug(
            f"[tradfi] cTrader klines failed {symbol.upper()} {timeframe}: {exc}"
        )
        return []


async def _fetch_binance_metals_klines(
    symbol: str,
    timeframe: str,
    limit: int,
) -> List[List[float]]:
    """Closed OHLC from Binance spot XAUUSDT / XAGUSDT (forming bar stripped)."""
    _bn_sym = _METALS_BINANCE_MAP.get(symbol.upper())
    if not _bn_sym:
        return []

    _bn_interval = _BINANCE_INTERVAL_MAP.get(timeframe, timeframe)
    now = datetime.utcnow()
    key = (_bn_sym, _bn_interval, limit)
    cached = _KLINE_CACHE.get(key)
    if cached and (now - cached[1]) < _KLINE_TTL:
        return cached[0]

    try:
        import httpx
        async with httpx.AsyncClient(timeout=5.0) as _c:
            r = await _c.get(
                f"{_BINANCE_SPOT_BASE}/klines",
                params={
                    "symbol": _bn_sym,
                    "interval": _bn_interval,
                    "limit": limit + 1,
                },
            )
        if r.status_code == 200:
            raw = r.json()
            if raw and isinstance(raw, list):
                complete = raw[:-1] if len(raw) > 1 else raw
                rows: List[List[float]] = []
                for bar in complete[-limit:]:
                    try:
                        rows.append([
                            int(bar[0]),
                            float(bar[1]),
                            float(bar[2]),
                            float(bar[3]),
                            float(bar[4]),
                            float(bar[5]),
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
    return []


async def _get_klines_impl(
    symbol: str,
    asset_class: str,
    timeframe: str = "15m",
    limit: int = 200,
    for_backtest: bool = False,
    ctrader_user_id: Optional[int] = None,
) -> List[List[float]]:
    """
    Return up to `limit` OHLC bars in MEXC-shape: [[ts_ms, o, h, l, c, v], ...]
    Metals: Binance spot API (closed candles only — last forming bar stripped).
    Others: yfinance download().
    """
    cls = normalize_asset_class(asset_class)
    if cls == ASSET_CLASS_CRYPTO:
        return []

    if cls == "index":
        try:
            from app.services.index_symbols import normalize_index_symbol
            symbol = normalize_index_symbol(symbol)
        except Exception:
            symbol = symbol.upper()

    now = datetime.utcnow()
    sym_norm = symbol.upper().replace("/", "").replace("-", "")
    is_metal = sym_norm in _METALS_BINANCE_MAP

    # Live forex majors: same multi-source chain as the UI scanner/backtest tools.
    # A thin cTrader response (e.g. 3 bars after a broker hiccup) used to return
    # immediately and skip Yahoo/FMP — every strategy then hit blk_no_price_data /
    # blk_ta_conditions even though the scanner still found signals.
    if cls == ASSET_CLASS_FOREX and not is_metal and not for_backtest:
        try:
            scan_rows = await fetch_forex_scan_candles(
                sym_norm, timeframe, limit, user_id=ctrader_user_id,
            )
            if scan_rows:
                return scan_rows[-limit:] if len(scan_rows) > limit else scan_rows
        except Exception as _fse:
            logger.debug(f"[tradfi] fetch_forex_scan_candles failed {sym_norm}: {_fse}")

    # Live/paper paths prefer broker-matched cTrader OHLC. Backtest discovery
    # (Gold Strategy Finder) prefers FMP first — faster, no protobuf socket auth,
    # and avoids cTrader timing out on multi-thousand-bar requests.
    is_index = cls == "index"
    if is_metal and for_backtest:
        _sources = ("fmp", "ctrader")
    elif is_index:
        _sources = ("ctrader", "fmp")
    else:
        _sources = ("ctrader", "fmp") if is_metal else ("ctrader",)

    for src in _sources:
        if src == "fmp":
            try:
                from app.services.fmp_price_feed import fmp_in_backoff
                if fmp_in_backoff():
                    continue
            except Exception:
                pass
            if is_metal:
                _frows = await _fetch_fmp_metals_klines(symbol, asset_class, timeframe, limit)
            elif is_index:
                try:
                    from app.services.fmp_price_feed import get_klines as _fmp_klines
                    _frows = await _fmp_klines(symbol, asset_class, timeframe, limit)
                except Exception:
                    _frows = []
            else:
                _frows = []
            if _frows:
                return _frows
        elif src == "ctrader":
            _crows = await _fetch_ctrader_klines(
                symbol, asset_class, timeframe, limit, user_id=ctrader_user_id
            )
            _min_bars = min(limit, max(25, limit // 3))
            if _crows and len(_crows) >= _min_bars:
                return _crows
            if _crows:
                logger.info(
                    f"[tradfi] cTrader partial {sym_norm} {timeframe}: "
                    f"{len(_crows)} bars < {_min_bars} — trying other sources"
                )

    # ── Live forex: Yahoo chart + FMP (same chain as fetch_forex_scan_candles) ─
    # cTrader alone often returns [] on Railway (no linked account / cold feed);
    # yfinance download() is also unreliable server-side. Yahoo chart + FMP are
    # what keep the UI scanner finding signals — the executor must use them too.
    if cls == ASSET_CLASS_FOREX and not is_metal and not for_backtest:
        yahoo_ticker = yf_ticker(cls, symbol)
        if yahoo_ticker:
            _yrows = await _fetch_yahoo_chart_klines(
                yahoo_ticker, timeframe, limit, http_timeout_s=10.0,
            )
            if _yrows:
                logger.info(
                    f"[tradfi] klines ok (Yahoo fallback): {symbol.upper()} {timeframe} "
                    f"→ {len(_yrows)} bars"
                )
                return _yrows
        if _env_fmp_api_key():
            try:
                from app.services.fmp_price_feed import fmp_in_backoff, get_klines as _fmp_klines
                if not fmp_in_backoff():
                    _frows = await _fmp_klines(symbol, asset_class, timeframe, limit)
                    if _frows:
                        return _frows
            except Exception as _fe:
                logger.debug(
                    f"[tradfi] FMP forex klines failed {symbol} {timeframe}: {_fe}"
                )

    # ── Binance spot metals fallback (backtest / after cTrader+FMP miss) ───────
    if is_metal:
        _brows = await _fetch_binance_metals_klines(symbol, timeframe, limit)
        if _brows:
            return _brows

    # ── FMP 1-minute REST — primary source for forex paper evaluation ─────────
    if timeframe == "1m" and _env_fmp_api_key():
        try:
            from app.services.fmp_price_feed import fmp_in_backoff, get_klines as _fmp_klines
            if not fmp_in_backoff():
                _frows = await _fmp_klines(symbol, asset_class, timeframe, limit)
                if _frows:
                    return _frows
        except Exception as _fe:
            logger.debug(f"[tradfi] FMP 1min failed {symbol}: {_fe}")

    # ── Yahoo chart API — metals futures (GC=F / SI=F) before yfinance lib ─────
    if is_metal:
        _yt = _METALS_YAHOO_TICKER.get(symbol.upper())
        if _yt:
            _yrows = await _fetch_yahoo_chart_klines(_yt, timeframe, limit)
            if _yrows:
                return _yrows
        logger.warning(
            f"[tradfi] metals spot klines unavailable for {symbol.upper()} {timeframe} "
            f"— trying yfinance futures as last resort"
        )

    # ── yfinance download() ───────────────────────────────────────────────────
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
