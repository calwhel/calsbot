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

import asyncio
import logging
import os
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import httpx

logger = logging.getLogger(__name__)


def _binance_disabled() -> bool:
    from app.services.binance_feed import binance_disabled
    return binance_disabled()

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
            timeout=_http_timeout_s(8.0),
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


# Symbols MEXC returns 400 for — skip without retrying every cycle.
_mexc_missing: set = set()
_btc_closes_cache: Optional[list] = None
_btc_closes_at: Optional[datetime] = None
# In-memory crypto kline cache — prefetch timeout fallback (mirrors tradfi peek).
_CRYPTO_KLINE_CACHE: Dict[Tuple[str, str, int], Tuple[List[list], str, float]] = {}
_CRYPTO_KLINE_CACHE_TTL_S = float(os.environ.get("CRYPTO_KLINE_CACHE_TTL_S", "120"))


def _crypto_kline_cache_key(symbol: str, interval: str, limit: int) -> Tuple[str, str, int]:
    return (symbol.upper(), interval, int(limit))


def _store_crypto_kline_cache(
    symbol: str, interval: str, limit: int, rows: List[list], source: str,
) -> None:
    if not rows:
        return
    _CRYPTO_KLINE_CACHE[_crypto_kline_cache_key(symbol, interval, limit)] = (
        rows, source, datetime.utcnow().timestamp(),
    )


def peek_cached_crypto_klines(
    symbol: str, interval: str, limit: int,
) -> Tuple[List[list], str]:
    """Return cached crypto OHLC rows without network I/O (stale OK on timeout)."""
    key = _crypto_kline_cache_key(symbol, interval, limit)
    hit = _CRYPTO_KLINE_CACHE.get(key)
    if not hit:
        return [], ""
    rows, src, _ts = hit
    return list(rows), src


def _http_timeout_s(default: float) -> float:
    try:
        from app.services.prefetch_fast import provider_timeout_s
        return provider_timeout_s(default)
    except Exception:
        return default


async def _fetch_mexc_klines(
    http_client: httpx.AsyncClient, symbol: str, interval: str, limit: int,
) -> List[list]:
    if symbol in _mexc_missing:
        return []
    mexc_interval = interval.replace("1h", "60m").replace("2h", "120m")
    url = (
        f"https://api.mexc.com/api/v3/klines"
        f"?symbol={symbol}&interval={mexc_interval}&limit={limit}"
    )
    try:
        resp = await http_client.get(url, timeout=_http_timeout_s(5.0))
        if resp.status_code == 200:
            data = resp.json()
            if data and isinstance(data, list):
                return data
        elif resp.status_code == 400:
            _mexc_missing.add(symbol)
    except Exception:
        pass
    return []


async def _fetch_mexc_ticker(
    http_client: httpx.AsyncClient, symbol: str,
) -> Optional[Dict]:
    if symbol in _mexc_missing:
        return None
    try:
        resp = await http_client.get(
            f"https://api.mexc.com/api/v3/ticker/24hr?symbol={symbol}",
            timeout=_http_timeout_s(5.0),
        )
        if resp.status_code == 200:
            data = resp.json()
            if isinstance(data, dict) and float(data.get("lastPrice", 0) or 0) > 0:
                return data
        elif resp.status_code == 400:
            _mexc_missing.add(symbol)
    except Exception:
        pass
    return None


async def _fetch_binance_futures_klines(
    http_client: httpx.AsyncClient, symbol: str, interval: str, limit: int,
) -> List[list]:
    if _binance_disabled():
        return []
    from app.services.binance_feed import binance_futures_klines
    return await binance_futures_klines(http_client, symbol, interval, limit)


async def _fetch_bybit_klines(
    http_client: httpx.AsyncClient, symbol: str, interval: str, limit: int,
) -> List[list]:
    _bybit_iv = {
        "1m": "1", "5m": "5", "15m": "15", "30m": "30",
        "1h": "60", "60m": "60", "2h": "120", "4h": "240",
        "1d": "D", "1w": "W",
    }
    iv = _bybit_iv.get(interval, "15")
    try:
        resp = await http_client.get(
            "https://api.bybit.com/v5/market/kline",
            params={
                "category": "linear",
                "symbol": symbol,
                "interval": iv,
                "limit": max(int(limit), 1),
            },
            timeout=_http_timeout_s(8.0),
        )
        if resp.status_code != 200:
            return []
        payload = resp.json()
        if not isinstance(payload, dict):
            return []
        result = payload.get("result")
        if not isinstance(result, dict):
            return []
        rows_raw = result.get("list") or []
        if not isinstance(rows_raw, list):
            return []
        out: List[list] = []
        for k in rows_raw:
            if not isinstance(k, (list, tuple)) or len(k) < 5:
                continue
            try:
                ts = int(k[0])
                out.append([
                    ts,
                    str(k[1]), str(k[2]), str(k[3]), str(k[4]),
                    str(k[5] if len(k) > 5 else 0),
                    ts,
                    "0",
                ])
            except Exception:
                continue
        out.sort(key=lambda r: r[0])
        return out
    except Exception as e:
        logger.debug("[crypto-md] Bybit klines %s %s: %s", symbol, interval, e)
    return []


async def _fetch_coingecko_ticker(
    http_client: httpx.AsyncClient, symbol: str,
) -> Optional[Dict]:
    """Last-resort spot price when exchange tickers fail (no klines)."""
    base = symbol.upper().replace("USDT", "").replace("PERP", "")
    if not base:
        return None
    try:
        from app.services.coingecko_safe import fetch_markets
        coins = await fetch_markets(
            http_client,
            params={
                "vs_currency": "usd",
                "ids": "",
                "order": "market_cap_desc",
                "per_page": 250,
                "page": 1,
            },
        )
        match = next(
            (c for c in coins if (c.get("symbol") or "").upper() == base),
            None,
        )
        if not match:
            return None
        px = float(match.get("current_price") or 0)
        if px <= 0:
            return None
        chg = float(match.get("price_change_percentage_24h") or 0)
        vol = float(match.get("total_volume") or 0)
        return {
            "symbol": symbol,
            "lastPrice": str(px),
            "priceChangePercent": str(chg),
            "quoteVolume": str(vol),
            "highPrice": str(px),
            "lowPrice": str(px),
            "openPrice": str(px),
        }
    except Exception:
        return None


async def _fetch_klines_chain(
    http_client: httpx.AsyncClient, symbol: str, interval: str, limit: int,
    *,
    deadline_mono: Optional[float] = None,
) -> List[list]:
    # Binance primary (EU region) → Bybit → MEXC → Bitunix.
    chain: List[Tuple] = [
        (_fetch_bybit_klines, "bybit"),
        (_fetch_mexc_klines, "mexc"),
    ]
    if not _binance_disabled():
        chain.insert(0, (_fetch_binance_futures_klines, "binance"))

    try:
        from app.services.prefetch_fast import prefetch_fast_active, provider_timeout_s
        _fast = prefetch_fast_active()
    except Exception:
        _fast = False
        provider_timeout_s = lambda d: d  # type: ignore

    if _fast and len(chain) > 1:
        from app.services.prefetch_provider_limits import (
            PrefetchSlotUnavailable,
            prefetch_provider_slot,
        )

        def _slot_wait_budget_s() -> Optional[float]:
            if deadline_mono is None:
                return None
            remaining = deadline_mono - time.monotonic()
            if remaining <= 0.0:
                return 0.0
            # Leave a small tail so slot timeouts surface as clean skips before
            # the outer symbol budget hard-cancels the whole fetch.
            return max(0.0, remaining - 0.02)

        async def _attempt(fetcher, label: str):
            try:
                rows = await asyncio.wait_for(
                    fetcher(http_client, symbol, interval, limit),
                    timeout=provider_timeout_s(2.0),
                )
                return rows, label
            except Exception:
                return [], label

        try:
            async with prefetch_provider_slot(
                "crypto",
                max_wait_s=_slot_wait_budget_s(),
            ):
                tasks = [asyncio.create_task(_attempt(f, lbl)) for f, lbl in chain]
                try:
                    while tasks:
                        done, pending = await asyncio.wait(
                            tasks,
                            return_when=asyncio.FIRST_COMPLETED,
                            timeout=provider_timeout_s(2.0),
                        )
                        if not done:
                            break
                        for task in done:
                            tasks.remove(task)
                            try:
                                rows, label = task.result()
                                if rows:
                                    for pt in pending:
                                        pt.cancel()
                                    _store_crypto_kline_cache(symbol, interval, limit, rows, label)
                                    return rows
                            except Exception:
                                pass
                finally:
                    for task in tasks:
                        task.cancel()
        except PrefetchSlotUnavailable:
            logger.info(
                "[crypto-md] %s %s skipped: no fetch slot available in time",
                symbol,
                interval,
            )
            return []

    for fetcher, label in chain:
        rows = await fetcher(http_client, symbol, interval, limit)
        if rows:
            logger.debug("[crypto-md] klines %s %s via %s (%d bars)", symbol, interval, label, len(rows))
            _store_crypto_kline_cache(symbol, interval, limit, rows, label)
            return rows
    rows = await fetch_klines(http_client, symbol, interval, limit)
    if rows:
        _store_crypto_kline_cache(symbol, interval, limit, rows, "bitunix")
    return rows


async def _fetch_binance_futures_ticker(
    http_client: httpx.AsyncClient, symbol: str,
) -> Optional[Dict]:
    if _binance_disabled():
        return None
    from app.services.binance_feed import binance_get
    status, data, _ = await binance_get(
        http_client,
        "https://fapi.binance.com/fapi/v1/ticker/24hr",
        {"symbol": symbol},
    )
    if status == 200 and isinstance(data, dict) and float(data.get("lastPrice", 0) or 0) > 0:
        return data
    return None


async def _fetch_bybit_ticker(
    http_client: httpx.AsyncClient, symbol: str,
) -> Optional[Dict]:
    try:
        resp = await http_client.get(
            "https://api.bybit.com/v5/market/tickers",
            params={"category": "linear", "symbol": symbol},
            timeout=_http_timeout_s(8.0),
        )
        if resp.status_code != 200:
            return None
        payload = resp.json()
        if not isinstance(payload, dict):
            return None
        result = payload.get("result")
        if not isinstance(result, dict):
            return None
        items = result.get("list") or []
        if not isinstance(items, list) or not items:
            return None
        t = items[0] if isinstance(items[0], dict) else None
        if not t:
            return None
        last = float(t.get("lastPrice") or 0)
        if last <= 0:
            return None
        open_ = float(t.get("prevPrice24h") or t.get("openPrice") or last)
        chg = ((last - open_) / open_ * 100.0) if open_ > 0 else 0.0
        return {
            "symbol": symbol,
            "lastPrice": str(last),
            "priceChangePercent": str(chg),
            "quoteVolume": str(float(t.get("turnover24h") or 0)),
            "highPrice": str(float(t.get("highPrice24h") or last)),
            "lowPrice": str(float(t.get("lowPrice24h") or last)),
            "openPrice": str(open_),
        }
    except Exception:
        return None


async def _fetch_ticker_chain(
    http_client: httpx.AsyncClient, symbol: str,
) -> Tuple[Optional[Dict], str]:
    fetchers = [
        (_fetch_bybit_ticker, "bybit"),
        (_fetch_mexc_ticker, "mexc"),
        (fetch_ticker, "bitunix"),
    ]
    if not _binance_disabled():
        fetchers.insert(0, (_fetch_binance_futures_ticker, "binance"))

    try:
        from app.services.prefetch_fast import prefetch_fast_active, provider_timeout_s
        _fast = prefetch_fast_active()
    except Exception:
        _fast = False
        provider_timeout_s = lambda d: d  # type: ignore

    if _fast and len(fetchers) > 1:
        async def _attempt(fetcher, label: str):
            try:
                t = await asyncio.wait_for(
                    fetcher(http_client, symbol),
                    timeout=provider_timeout_s(2.0),
                )
                return t, label
            except Exception:
                return None, label

        tasks = [asyncio.create_task(_attempt(f, lbl)) for f, lbl in fetchers]
        try:
            while tasks:
                done, pending = await asyncio.wait(
                    tasks,
                    return_when=asyncio.FIRST_COMPLETED,
                    timeout=provider_timeout_s(2.0),
                )
                if not done:
                    break
                for task in done:
                    tasks.remove(task)
                    try:
                        ticker, label = task.result()
                        if ticker:
                            for pt in pending:
                                pt.cancel()
                            return ticker, label
                    except Exception:
                        pass
        finally:
            for task in tasks:
                task.cancel()

    for fetcher, label in fetchers:
        t = await fetcher(http_client, symbol)
        if t:
            return t, label
    return None, ""


def _calc_rsi(closes: list) -> float:
    if len(closes) < 14:
        return 50.0
    deltas = [closes[i] - closes[i - 1] for i in range(1, len(closes))]
    gains = [d if d > 0 else 0.0 for d in deltas]
    losses = [-d if d < 0 else 0.0 for d in deltas]
    avg_gain = sum(gains[-14:]) / 14
    avg_loss = sum(losses[-14:]) / 14
    if avg_loss > 0:
        return 100.0 - (100.0 / (1.0 + avg_gain / avg_loss))
    return 100.0


def _calc_volume_ratio(volumes: list) -> float:
    if len(volumes) < 5:
        return 1.0
    avg_vol = sum(volumes[:-1]) / len(volumes[:-1])
    return volumes[-1] / avg_vol if avg_vol > 0 else 1.0


async def _get_btc_closes(http_client: httpx.AsyncClient) -> list:
    global _btc_closes_cache, _btc_closes_at
    now = datetime.utcnow()
    if (
        _btc_closes_cache
        and _btc_closes_at
        and (now - _btc_closes_at).total_seconds() < 300
    ):
        return _btc_closes_cache
    try:
        klines = await _fetch_klines_chain(http_client, "BTCUSDT", "15m", 20)
        if klines:
            closes = [float(k[4]) for k in klines]
            _btc_closes_cache = closes
            _btc_closes_at = now
            return closes
    except Exception:
        pass
    return _btc_closes_cache or []


def _calc_correlation(coin_closes: list, btc_closes: list) -> float:
    n = min(len(coin_closes), len(btc_closes))
    if n < 6:
        return 0.0
    coin_closes = coin_closes[-n:]
    btc_closes = btc_closes[-n:]
    coin_returns = [
        (coin_closes[i] - coin_closes[i - 1]) / coin_closes[i - 1]
        for i in range(1, n)
        if coin_closes[i - 1] != 0
    ]
    btc_returns = [
        (btc_closes[i] - btc_closes[i - 1]) / btc_closes[i - 1]
        for i in range(1, n)
        if btc_closes[i - 1] != 0
    ]
    m = min(len(coin_returns), len(btc_returns))
    if m < 5:
        return 0.0
    coin_returns = coin_returns[-m:]
    btc_returns = btc_returns[-m:]
    mean_c = sum(coin_returns) / m
    mean_b = sum(btc_returns) / m
    num = sum((coin_returns[i] - mean_c) * (btc_returns[i] - mean_b) for i in range(m))
    den_c = sum((coin_returns[i] - mean_c) ** 2 for i in range(m)) ** 0.5
    den_b = sum((btc_returns[i] - mean_b) ** 2 for i in range(m)) ** 0.5
    if den_c > 0 and den_b > 0:
        return num / (den_c * den_b)
    return 0.0


async def fetch_crypto_price_and_ta(
    http_client: httpx.AsyncClient,
    symbol: str,
    *,
    prefetch: bool = False,
) -> Optional[Dict]:
    """
    Price + TA for portal crypto strategy evaluation.

    Uses MEXC → Bitunix public market data only. Intentionally NOT the legacy
    SocialSignalService (that path logged 📱 social-scanner lines and confused
    portal strategy scans with the old Telegram social-signals bot).
    """
    try:
        from app.services.enhanced_ta import analyze_klines
        from app.services.prefetch_fast import SYMBOL_BUDGET_S, prefetch_fast_active

        async def _run(deadline_mono: Optional[float] = None):
            ticker_task = asyncio.create_task(_fetch_ticker_chain(http_client, symbol))
            k15_task = asyncio.create_task(
                _fetch_klines_chain(
                    http_client,
                    symbol,
                    "15m",
                    50,
                    deadline_mono=deadline_mono,
                )
            )
            k1h_task = asyncio.create_task(
                _fetch_klines_chain(
                    http_client,
                    symbol,
                    "1h",
                    30,
                    deadline_mono=deadline_mono,
                )
            )
            (ticker, ticker_src), klines_15m, klines_1h = await asyncio.gather(
                ticker_task, k15_task, k1h_task,
            )
            return ticker, ticker_src, klines_15m, klines_1h

        if prefetch or prefetch_fast_active():
            deadline_mono = time.monotonic() + float(SYMBOL_BUDGET_S)
            ticker, ticker_src, klines_15m, klines_1h = await asyncio.wait_for(
                _run(deadline_mono=deadline_mono),
                timeout=SYMBOL_BUDGET_S,
            )
        else:
            ticker, ticker_src, klines_15m, klines_1h = await _run()
        if not ticker:
            return None
        if not klines_15m:
            logger.info(
                "[crypto-md] %s skipped: no 15m klines available within budget",
                symbol,
            )
            return None

        kline_src = ""
        if klines_15m:
            _, kline_src = peek_cached_crypto_klines(symbol, "15m", 50)
        if not kline_src:
            kline_src = ticker_src or "unknown"

        closes = [float(k[4]) for k in klines_15m] if klines_15m else []
        volumes = [float(k[5]) for k in klines_15m] if klines_15m else []
        rsi = _calc_rsi(closes)
        volume_ratio = _calc_volume_ratio(volumes)
        btc_closes = await _get_btc_closes(http_client)
        btc_corr = _calc_correlation(closes, btc_closes) if closes and btc_closes else 0.0
        enhanced_ta = analyze_klines(klines_15m, klines_1h)

        return {
            "price": float(ticker.get("lastPrice", 0)),
            "change_24h": float(ticker.get("priceChangePercent", 0)),
            "volume_24h": float(ticker.get("quoteVolume", 0)),
            "high_24h": float(ticker.get("highPrice", 0)),
            "low_24h": float(ticker.get("lowPrice", 0)),
            "rsi": rsi,
            "volume_ratio": volume_ratio,
            "btc_correlation": btc_corr,
            "enhanced_ta": enhanced_ta,
            "candles_loaded": max(len(klines_15m or []), len(klines_1h or [])),
            "kline_source": kline_src,
            "price_source": ticker_src or "spot_live",
            "live_source": ticker_src,
        }
    except asyncio.TimeoutError:
        logger.info(
            "[crypto-md] %s skipped: price/TA fetch budget exceeded",
            symbol,
        )
        return None
    except Exception as exc:
        logger.debug("[crypto-md] price/TA failed %s: %s", symbol, exc)
        return None
