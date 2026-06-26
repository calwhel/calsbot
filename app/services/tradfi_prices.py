"""
Stocks / forex / indices price + kline fetcher.

Price path:
  Metals (XAUUSD/XAGUSD): Binance spot API (XAUUSDT/XAGUSDT) — real-time,
  no futures contango premium, same format as crypto pairs.
  Others: FMP real-time WebSocket → yfinance fast_info fallback.

Kline path:
  Metals live: cTrader → FMP → Coinbase → Kraken → GC=F → synthetic (no Binance).
  Metals backtest: Binance → cTrader → FMP → Yahoo (GC=F only if spot-aligned).
  Forex live: cTrader → Yahoo chart → FMP → yfinance.
  Index live: cTrader → Yahoo chart → FMP (same chain as forex scanner).
  Others: yfinance download() — intraday OHLC fallback.

Returned shapes mirror the crypto helpers so the strategy executor can
consume them without branching:
- get_price(symbol, asset_class) -> Optional[float]
- get_klines(symbol, asset_class, interval, limit) -> List[[ts, o, h, l, c, v]]
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

from app.services.asset_classes import (
    ASSET_CLASS_CRYPTO,
    ASSET_CLASS_FOREX,
    ASSET_CLASS_INDEX,
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
_METAL_KLINE_TTL = timedelta(
    seconds=max(15, int(os.environ.get("METAL_KLINE_CACHE_SECONDS", "30")))
)
_KRAKEN_KLINE_TIMEOUT_S = float(os.environ.get("KRAKEN_KLINE_TIMEOUT_SECONDS", "20"))
_BINANCE_KLINE_TIMEOUT_S = float(os.environ.get("BINANCE_KLINE_TIMEOUT_SECONDS", "15"))
_FMP_KLINE_TIMEOUT_S = float(os.environ.get("FMP_KLINE_TIMEOUT_SECONDS", "15"))

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

# Kraken public OHLC — works from US/Railway when Binance spot is geo-blocked.
_KRAKEN_METAL_MAP: Dict[str, str] = {
    "XAUUSD": "PAXGUSD",
    "XAGUSD": "XAGUSD",
}
_KRAKEN_INTERVAL: Dict[str, int] = {
    "1m": 1, "5m": 5, "15m": 15, "30m": 30, "1h": 60, "4h": 240, "1d": 1440,
}
# Coinbase Exchange public candles — no US geo-block (Binance returns HTTP 451).
_COINBASE_METAL_PRODUCT: Dict[str, str] = {
    "XAUUSD": "PAXG-USD",
    "XAGUSD": "XAG-USD",
}
_COINBASE_GRANULARITY: Dict[str, int] = {
    "1m": 60, "3m": 180, "5m": 300, "15m": 900, "30m": 1800,
    "1h": 3600, "4h": 14400, "1d": 86400,
}
_COINBASE_KLINE_TIMEOUT_S = float(os.environ.get("COINBASE_KLINE_TIMEOUT_SECONDS", "25"))
_COINBASE_CONNECT_TIMEOUT_S = float(os.environ.get("COINBASE_CONNECT_TIMEOUT_SECONDS", "20"))
_COINBASE_STALE_KLINE_TTL = timedelta(
    seconds=max(60, int(os.environ.get("COINBASE_KLINE_STALE_SECONDS", "300")))
)
_COINBASE_EXCHANGE_BASE = "https://api.exchange.coinbase.com"

_METAL_KLINE_SOURCE_RANK = {
    "ctrader-user": 0,
    "ctrader": 1,
    "coinbase": 2,
    "binance": 3,
    "kraken": 4,
    "yahoo_gc": 5,
    "synthetic": 6,
    "fmp": 7,
}
_INTERVAL_MS: Dict[str, int] = {
    "1m": 60_000,
    "3m": 180_000,
    "5m": 300_000,
    "15m": 900_000,
    "30m": 1_800_000,
    "1h": 3_600_000,
    "4h": 14_400_000,
    "1d": 86_400_000,
}

# Gold futures (GC=F) can sit $5–$50 away from XAUUSD spot — never fire trades
# on futures kline closes when live spot disagrees by more than this %.
METAL_KLINE_LIVE_MAX_DRIFT_PCT = float(
    os.environ.get("METAL_KLINE_LIVE_MAX_DRIFT_PCT", "0.4")
)
# Spot-aligned kline sources (Binance/Kraken PAXG/cTrader) naturally diverge
# ~0.25–0.35% from XAUUSD spot — use a looser cap than the GC=F guard.
METAL_SPOT_KLINE_MAX_DRIFT_PCT = float(
    os.environ.get("METAL_SPOT_KLINE_MAX_DRIFT_PCT", "0.5")
)
# PAXG proxies (Coinbase/Kraken klines) vs live tick — closed H1 bar can diverge ~1%.
METAL_PAXG_KLINE_MAX_DRIFT_PCT = float(
    os.environ.get("METAL_PAXG_KLINE_MAX_DRIFT_PCT", "1.0")
)
_PAXG_KLINE_SOURCES = frozenset({"coinbase", "kraken"})
_SPOT_ALIGNED_KLINE_SOURCES = frozenset({
    "binance", "coinbase", "ctrader-user", "ctrader", "fmp", "kraken", "synthetic",
})


@dataclass
class MetalProviderDiagnostic:
    """Per-provider outcome for XAUUSD/XAGUSD kline failover tracing."""
    provider: str
    url: str = ""
    response_bytes: int = 0
    candle_count: int = 0
    failure: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)


_metal_kline_trace_counter = 0


def _log_metal_kline_trace(
    symbol: str,
    timeframe: str,
    diagnostics: List[MetalProviderDiagnostic],
) -> None:
    """Structured trace — sampled DEBUG (fallback line covers ops visibility)."""
    global _metal_kline_trace_counter
    _metal_kline_trace_counter += 1
    if _metal_kline_trace_counter % 20 != 1:
        return
    for d in diagnostics:
        logger.debug(
            "[metal-kline-trace] %s",
            json.dumps(
                {"symbol": symbol, "timeframe": timeframe, **asdict(d)},
                separators=(",", ":"),
                default=str,
            ),
        )
# Winning metal kline provider per (symbol, timeframe, limit) — for drift rules.
_METAL_KLINE_SOURCE_CACHE: Dict[Tuple[str, str, int], Tuple[str, datetime]] = {}
_METAL_LIVE_FETCH_SEM: Optional[asyncio.Semaphore] = None
_CTRADER_KLINE_TIMEOUT_LIVE_S = float(
    os.environ.get("CTRADER_KLINE_TIMEOUT_LIVE_S", "15")
)


def is_metal_symbol(symbol: str) -> bool:
    return symbol.upper().replace("/", "").replace("-", "") in _METALS_BINANCE_MAP


def metal_kline_drift_limit(source: Optional[str] = None) -> float:
    """Source-aware drift cap — spot proxies (PAXG) need more headroom than GC=F."""
    src = str(source or "").lower()
    if src in _PAXG_KLINE_SOURCES:
        return METAL_PAXG_KLINE_MAX_DRIFT_PCT
    if src in _SPOT_ALIGNED_KLINE_SOURCES:
        return METAL_SPOT_KLINE_MAX_DRIFT_PCT
    return METAL_KLINE_LIVE_MAX_DRIFT_PCT


def _ctrader_unavailable_reason(symbol: str) -> str:
    """Human-readable reason cTrader did not serve metal klines this cycle."""
    sym = symbol.upper()
    try:
        from app.services import ctrader_price_feed as _ctf
        if getattr(_ctf, "_is_terminal_auth_error", lambda _e: False)(
            getattr(_ctf, "_last_auth_error", None)
        ):
            return "oauth expired — relink cTrader"
        st = _ctf.feed_status()
        if not st.get("subscribed") and not st.get("remote_live"):
            return "feed task not running"
        if st.get("needs_relink"):
            return "needs relink"
        if not _ctf.ctrader_spot_ready(sym) and not _ctf.broker_session_ready(sym):
            tick_age = st.get("last_tick_age_s")
            if tick_age is not None:
                return f"no spot ticks (last_tick_age_s={tick_age})"
            return "no broker session and no spot ticks"
        blocked = getattr(_ctf, "trendbar_fetch_blocked_reason", None)
        if callable(blocked):
            reason = blocked()
            if reason:
                return reason
        elif not _ctf._trendbar_fetch_allowed():
            return "trendbar fetch blocked"
        return "empty or timed-out trendbars"
    except Exception as exc:
        return type(exc).__name__


async def sweep_stale_metal_klines(
    symbols: Optional[List[str]] = None,
    timeframes: Optional[List[str]] = None,
) -> int:
    """Per-cycle sweep across ALL tradfi kline caches (coinbase, kraken, fmp, …)."""
    from app.services.kline_staleness import check_cached_klines_stale

    targets = {s.upper() for s in symbols} if symbols else {"XAUUSD", "XAGUSD"}
    tfs = timeframes or ["5m", "15m", "1h"]
    rebuilt = 0
    for key in list(_KLINE_CACHE):
        key0 = str(key[0]).upper()
        if not any(t in key0 for t in targets):
            continue
        tf = key[1] if len(key) > 1 else "15m"
        if tf not in tfs:
            continue
        rows, fetched_at = _KLINE_CACHE.get(key, ([], datetime.utcnow()))
        if not rows:
            continue
        sym_guess = next((t for t in targets if t in key0), "XAUUSD")
        stale, detail = await check_cached_klines_stale(
            sym_guess, rows, tf, cache_fetched_at=fetched_at,
        )
        if not stale:
            continue
        source = key0.split(":")[0] if ":" in key0 else "cache"
        logger.warning(
            "[tradfi] kline builder stale for %s — rebuilt (%s, source=%s, tf=%s)",
            sym_guess,
            detail,
            source,
            tf,
        )
        _KLINE_CACHE.pop(key, None)
        rebuilt += 1
    if rebuilt:
        for sym in targets:
            for tf in tfs:
                _METAL_KLINE_SOURCE_CACHE.pop((sym, tf, 80), None)
    return rebuilt


def clear_metal_kline_cache(symbols: Optional[List[str]] = None) -> int:
    """Invalidate metal/forex kline caches after cTrader feed reconnect."""
    targets = None
    if symbols:
        targets = {
            s.upper().replace("/", "").replace("-", "")
            for s in symbols
        }
    removed = 0
    for key in list(_KLINE_CACHE):
        sym_part = str(key[0]).upper().replace(":", "")
        if targets is None or sym_part in targets or any(t in sym_part for t in targets):
            _KLINE_CACHE.pop(key, None)
            removed += 1
    for key in list(_METAL_KLINE_SOURCE_CACHE):
        if targets is None or key[0] in targets:
            _METAL_KLINE_SOURCE_CACHE.pop(key, None)
    return removed


def get_metal_kline_source(
    symbol: str,
    timeframe: str,
    limit: int,
) -> Optional[str]:
    """Return the provider label from the most recent metal-live kline fetch."""
    sym = symbol.upper().replace("/", "").replace("-", "")
    entry = _METAL_KLINE_SOURCE_CACHE.get((sym, timeframe, int(limit)))
    if not entry:
        return None
    label, fetched_at = entry
    if (datetime.utcnow() - fetched_at) > _METAL_KLINE_TTL:
        return None
    return label


def get_metal_kline_fetch_age_s(
    symbol: str,
    timeframe: str,
    limit: int,
) -> Optional[float]:
    """Age in seconds of last metal kline provider fetch for this key."""
    sym = symbol.upper().replace("/", "").replace("-", "")
    entry = _METAL_KLINE_SOURCE_CACHE.get((sym, timeframe, int(limit)))
    if not entry:
        return None
    _label, fetched_at = entry
    return max(0.0, (datetime.utcnow() - fetched_at).total_seconds())


def get_metal_kline_fetched_at(
    symbol: str,
    timeframe: str,
    limit: int,
) -> Optional[datetime]:
    """Timestamp of last metal kline provider fetch for this key."""
    sym = symbol.upper().replace("/", "").replace("-", "")
    entry = _METAL_KLINE_SOURCE_CACHE.get((sym, timeframe, int(limit)))
    if not entry:
        return None
    return entry[1]


def is_metal_kline_synthetic(
    symbol: str,
    timeframe: str,
    limit: int,
) -> bool:
    """True when the latest metal kline batch was built from flat synthetic spot bars."""
    return (get_metal_kline_source(symbol, timeframe, limit) or "").lower() == "synthetic"


def peek_cached_klines(
    symbol: str,
    asset_class: str,
    timeframe: str = "15m",
    limit: int = 200,
    ctrader_user_id: Optional[int] = None,
) -> Tuple[List[List[float]], str]:
    """
    Return last-known OHLC bars without network I/O.
    Prefers cTrader tick-updated cache over stale external-provider bars.
  """
    sym = symbol.upper().replace("/", "").replace("-", "")
    cls = normalize_asset_class(asset_class)
    best_rows: List[List[float]] = []
    best_src = ""

    # 1. cTrader inline tick→bar cache (live forex/metals — freshest on hot paths).
    try:
        from app.services.ctrader_price_feed import _kline_cache as _ct_cache
        for key, (rows, _ts) in _ct_cache.items():
            if not isinstance(key, tuple) or len(key) < 2:
                continue
            if str(key[0]).upper() != sym or str(key[1]) != timeframe:
                continue
            if rows and len(rows) > len(best_rows):
                best_rows = rows
                best_src = "ctrader-cache"
    except Exception:
        pass

    # 2. tradfi _KLINE_CACHE — yahoo/coinbase/kraken/metal keys (symbol in key[0] or key[1]).
    for key, (rows, _fetched_at) in _KLINE_CACHE.items():
        if not rows:
            continue
        key_sym = ""
        if isinstance(key, tuple) and key:
            key_sym = str(key[0]).replace("kraken:", "").upper()
            if key_sym in ("yahoo", "binance", "coinbase", "kraken") and len(key) > 1:
                key_sym = str(key[1]).upper()
        if key_sym and sym not in (key_sym, key_sym.replace("=X", "")):
            if sym not in _METALS_BINANCE_MAP or _METALS_BINANCE_MAP.get(sym) != key_sym:
                continue
        if len(rows) > len(best_rows):
            best_rows = rows
            label = get_metal_kline_source(sym, timeframe, limit)
            best_src = label or "tradfi-cache"

    if best_rows:
        out = best_rows[-limit:] if len(best_rows) > limit else best_rows
        return out, best_src or "cache"
    return [], ""


def _metal_live_fetch_sem() -> asyncio.Semaphore:
    global _METAL_LIVE_FETCH_SEM
    if _METAL_LIVE_FETCH_SEM is None:
        n = max(1, int(os.environ.get("METAL_KLINE_FETCH_CONCURRENT", "2")))
        _METAL_LIVE_FETCH_SEM = asyncio.Semaphore(n)
    return _METAL_LIVE_FETCH_SEM


def _metal_kline_cache_ttl(cache_key: Tuple) -> timedelta:
    """Metal spot kline keys use a longer TTL to survive multi-shard scan cycles."""
    key0 = str(cache_key[0]) if cache_key else ""
    if key0.startswith("kraken:") or key0 in _METALS_BINANCE_MAP.values():
        return _METAL_KLINE_TTL
    return _KLINE_TTL


async def metal_klines_match_live_spot(
    symbol: str,
    rows: List[List[float]],
    *,
    asset_class: str = ASSET_CLASS_FOREX,
) -> bool:
    """False when OHLC last close diverges from live spot (futures vs spot mismatch)."""
    if not rows:
        return False
    try:
        close = float(rows[-1][4])
    except (IndexError, TypeError, ValueError):
        return False
    if close <= 0:
        return False
    live = await get_price_fresh(symbol, asset_class)
    if not live or live <= 0:
        return False
    drift_pct = abs(live - close) / live * 100.0
    if drift_pct > METAL_KLINE_LIVE_MAX_DRIFT_PCT:
        logger.warning(
            "[tradfi] %s klines rejected — last close %.2f vs live spot %.2f "
            "(%.2f%% > %.2f%% max; likely GC=F futures vs XAUUSD spot)",
            symbol.upper(),
            close,
            live,
            drift_pct,
            METAL_KLINE_LIVE_MAX_DRIFT_PCT,
        )
        return False
    return True


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
    from app.services.prefetch_provider_limits import (
        is_rate_limit_http,
        prefetch_http_get,
    )

    interval, range_ = _YAHOO_CHART_TF.get(timeframe, ("15m", "60d"))
    cache_key = ("yahoo", yahoo_ticker, interval, range_, limit)
    now = datetime.utcnow()
    cached = _KLINE_CACHE.get(cache_key)
    if cached and (now - cached[1]) < _KLINE_TTL:
        return cached[0]
    try:
        import httpx
        _headers = {"User-Agent": "Mozilla/5.0 (compatible; TradeHub/1.0)"}
        async with httpx.AsyncClient(
            timeout=http_timeout_s,
            headers=_headers,
        ) as client:
            resp = await prefetch_http_get(
                "yahoo",
                client,
                f"https://query1.finance.yahoo.com/v8/finance/chart/{yahoo_ticker}",
                params={"interval": interval, "range": range_},
                headers=_headers,
            )
        if resp.status_code != 200:
            if is_rate_limit_http(resp.status_code) and cached and cached[0]:
                logger.warning(
                    "[tradfi] Yahoo chart 429 → cache %s %s (%d bars)",
                    yahoo_ticker,
                    timeframe,
                    len(cached[0]),
                )
                return cached[0][-limit:] if len(cached[0]) > limit else cached[0]
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
    Multi-source fetch for index scanners and executor (NASDAQ, S&P, etc.).
    Prefers linked cTrader demo/live candles when broker session is up.
    """
    try:
        from app.services.index_symbols import normalize_index_symbol, yf_ticker_for_index
        sym = normalize_index_symbol(symbol)
        yahoo_ticker = yf_ticker_for_index(sym)
    except Exception:
        sym = symbol.upper()
        yahoo_ticker = None

    best: List[List[float]] = []
    _min_bars = _scan_min_bars(limit)

    try:
        from app.services.prefetch_fast import prefetch_fast_active
        if prefetch_fast_active():
            cached_rows, cached_src = peek_cached_klines(
                sym, "index", timeframe, limit, user_id,
            )
            _src = str(cached_src or "").lower()
            _accept_cached = (
                len(cached_rows) >= _min_bars
                or (_src.startswith("ctrader") and len(cached_rows) >= 3)
            )
            if cached_rows and _accept_cached:
                logger.info(
                    f"[tradfi] prefetch-fast cache hit {sym} {timeframe} "
                    f"src={cached_src} → {len(cached_rows)} bars"
                )
                return cached_rows[-limit:] if len(cached_rows) > limit else cached_rows
    except Exception:
        pass

    async def _keep(rows: List[List[float]], label: str) -> None:
        nonlocal best
        if rows and len(rows) > len(best):
            best = rows
            logger.info(f"[tradfi] index-scan best={label} {sym} {timeframe} → {len(rows)} bars")

    _broker_ready = False
    try:
        from app.services.ctrader_price_feed import broker_session_ready as _ct_ready
        _broker_ready = bool(_ct_ready(sym))
    except Exception:
        pass

    async def _try_ctrader(label: str) -> bool:
        rows = await _fetch_ctrader_klines(
            sym, "index", timeframe, min(limit, 500), user_id=user_id,
        )
        await _keep(rows, label)
        return len(best) >= _min_bars

    if _broker_ready:
        if await _try_ctrader("ctrader-user" if user_id else "ctrader"):
            return best[-limit:]

    if yahoo_ticker:
        await _keep(await _fetch_yahoo_chart_klines(yahoo_ticker, timeframe, limit), "yahoo")
        if len(best) >= _min_bars:
            return best[-limit:]

    if _env_fmp_api_key():
        try:
            from app.services.fmp_price_feed import get_klines as _fmp_klines
            await _keep(await _fmp_klines(sym, "index", timeframe, limit), "fmp")
        except Exception:
            pass
        if len(best) >= _min_bars:
            return best[-limit:]

    if _broker_ready and user_id:
        if await _try_ctrader("ctrader"):
            return best[-limit:]

    tradfi_rows = await _get_klines_impl(
        sym, "index", timeframe, limit, for_backtest=True, ctrader_user_id=user_id,
    )
    await _keep(tradfi_rows, "tradfi-chain")
    return best[-limit:] if best else []


def _metal_kline_min_bars(limit: int) -> int:
    return min(limit, max(25, limit // 3))


def _scan_min_bars(limit: int) -> int:
    """Min bars to accept from a scan source (executor uses ~80 bars)."""
    return max(15, min(limit, 80))


async def _fetch_with_kline_timeout(
    coro,
    *,
    timeout_s: float,
    label: str,
    symbol: str,
    timeframe: str,
) -> List[List[float]]:
    from app.services.prefetch_fast import provider_timeout_s

    timeout_s = provider_timeout_s(timeout_s)
    try:
        rows = await asyncio.wait_for(coro, timeout=timeout_s)
        return rows if isinstance(rows, list) else []
    except asyncio.TimeoutError:
        logger.info(
            f"[tradfi] {label} klines timeout {symbol} {timeframe} "
            f"(>{timeout_s:.0f}s)"
        )
        return []
    except Exception as exc:
        logger.debug(f"[tradfi] {label} klines failed {symbol} {timeframe}: {exc}")
        return []


async def _resolve_metal_spot_price(symbol: str) -> Tuple[Optional[float], str]:
    """Best-effort spot for synthetic candles — never requires cTrader."""
    sym = symbol.upper()
    try:
        from app.services.metals_spot_feed import fetch_now, get_price
        px = get_price(sym)
        if px and px > 0:
            return float(px), "metals_spot_feed"
        px = await fetch_now(sym)
        if px and px > 0:
            return float(px), "metals_spot_feed_fetch"
    except Exception:
        pass
    try:
        from app.services.realtime_spot import get_realtime_spot
        px = await get_realtime_spot(sym, "forex", force_fetch=True, paper_ok=True)
        if px and px > 0:
            return float(px), "realtime_spot"
    except Exception:
        pass
    try:
        from app.services.spot_price_store import get_mid
        px = get_mid(sym, max_age_s=300.0)
        if px and px > 0:
            return float(px), "spot_store_stale"
    except Exception:
        pass
    # Last closed bar from any cached metal klines
    now = datetime.utcnow()
    for key, (rows, fetched_at) in _KLINE_CACHE.items():
        if (now - fetched_at).total_seconds() > 3600:
            continue
        key_sym = ""
        if isinstance(key, tuple) and key:
            key_sym = str(key[0]).replace("kraken:", "")
        if sym in (key_sym, _METALS_BINANCE_MAP.get(sym, "")):
            if rows:
                try:
                    return float(rows[-1][4]), "kline_cache"
                except (IndexError, TypeError, ValueError):
                    pass
    return None, ""


async def build_synthetic_metal_candles(
    symbol: str,
    timeframe: str,
    limit: int,
) -> List[List[float]]:
    """
    Emergency OHLC when cTrader/Binance/Kraken/FMP all miss.
    Built from live metals spot so strategy evaluation never sees zero bars.
    """
    sym = symbol.upper()
    if sym not in _METALS_BINANCE_MAP:
        return []

    spot, spot_src = await _resolve_metal_spot_price(sym)
    if not spot or spot <= 0:
        logger.error(
            "[tradfi] synthetic metal candles impossible for %s %s — no spot anchor",
            sym, timeframe,
        )
        return []

    step_ms = _INTERVAL_MS.get(timeframe, 900_000)
    now_ms = int(datetime.utcnow().timestamp() * 1000)
    # Align to bar boundary
    now_ms = (now_ms // step_ms) * step_ms
    rows: List[List[float]] = []
    for i in range(limit):
        ts = now_ms - (limit - 1 - i) * step_ms
        # Tiny deterministic wiggle so RSI/volatility indicators are not stuck at 50/0
        wiggle = 1.0 + 0.00015 * ((i % 7) - 3)
        c = spot * wiggle
        h = c * 1.00025
        l = c * 0.99975
        o = c * (1.0 + 0.00005 * ((i % 3) - 1))
        rows.append([ts, o, h, l, c, 0.0])

    logger.warning(
        "[tradfi] metal-live SYNTHETIC %s %s → %d bars from spot=%.2f (%s) "
        "(price reference only — TA/signal eval blocked)",
        sym, timeframe, len(rows), spot, spot_src,
    )
    _METAL_KLINE_SOURCE_CACHE[(sym, timeframe, int(limit))] = (
        "synthetic", datetime.utcnow(),
    )
    return rows


def _coinbase_candle_window(granularity_s: int, limit: int) -> Tuple[str, str]:
    """Bounded start/end for Coinbase candles — smaller payloads, faster on Railway."""
    end_dt = datetime.utcnow()
    span_s = granularity_s * min(max(limit + 5, 30), 300)
    start_dt = end_dt - timedelta(seconds=span_s)
    fmt = "%Y-%m-%dT%H:%M:%SZ"
    return start_dt.strftime(fmt), end_dt.strftime(fmt)


def _coinbase_httpx_timeout():
    import httpx
    return httpx.Timeout(
        connect=_COINBASE_CONNECT_TIMEOUT_S,
        read=_COINBASE_KLINE_TIMEOUT_S,
        write=10.0,
        pool=10.0,
    )


async def _fetch_coinbase_metals_klines(
    symbol: str,
    timeframe: str,
    limit: int,
    *,
    diag: Optional[MetalProviderDiagnostic] = None,
) -> List[List[float]]:
    """Closed OHLC from Coinbase Exchange (PAXG-USD / XAG-USD) — US-accessible."""
    sym = symbol.upper()
    product = _COINBASE_METAL_PRODUCT.get(sym)
    gran = _COINBASE_GRANULARITY.get(timeframe)
    start_s, end_s = ("", "")
    if gran:
        start_s, end_s = _coinbase_candle_window(gran, limit)
    url = (
        f"{_COINBASE_EXCHANGE_BASE}/products/{product}/candles"
        f"?granularity={gran}&start={start_s}&end={end_s}"
    )
    if diag is not None:
        diag.provider = "coinbase"
        diag.url = url
    if not product or not gran:
        if diag is not None:
            diag.failure = "unsupported_pair_or_timeframe"
        return []

    now = datetime.utcnow()
    key = (f"coinbase:{product}", timeframe, limit)
    cached = _KLINE_CACHE.get(key)
    if cached and (now - cached[1]) < _metal_kline_cache_ttl(key):
        rows = cached[0]
        from app.services.kline_staleness import check_cached_klines_stale
        stale, detail = await check_cached_klines_stale(
            sym, rows, timeframe, cache_fetched_at=cached[1],
        )
        if stale:
            logger.warning(
                "[tradfi] kline builder stale for %s — rebuilt (%s, source=coinbase)",
                sym,
                detail,
            )
            _KLINE_CACHE.pop(key, None)
        else:
            if diag is not None:
                diag.candle_count = len(rows)
                diag.extra["cache_hit"] = True
            return rows

    def _stale_fallback(reason: str) -> List[List[float]]:
        if not cached:
            return []
        age = now - cached[1]
        if age < _COINBASE_STALE_KLINE_TTL:
            if diag is not None:
                diag.candle_count = len(cached[0])
                diag.failure = reason
                diag.extra["stale_fallback"] = True
            logger.info(
                f"[tradfi] Coinbase stale klines for {product} {timeframe} "
                f"({reason}, age={age.total_seconds():.0f}s)"
            )
            return cached[0]
        return []

    import httpx
    params = {"granularity": gran, "start": start_s, "end": end_s}
    headers = {"User-Agent": "TradeHub/1.0 (metal-klines)"}
    last_exc: Optional[Exception] = None

    from app.services.prefetch_provider_limits import (
        is_rate_limit_http,
        prefetch_http_get,
    )

    try:
        from app.services.prefetch_fast import prefetch_fast_active as _pfa_coinbase
        _fast_coinbase = bool(_pfa_coinbase())
    except Exception:
        _fast_coinbase = False
    _attempts = 1 if _fast_coinbase else 2

    for attempt in range(_attempts):
        try:
            async with httpx.AsyncClient(
                timeout=_coinbase_httpx_timeout(),
                headers=headers,
            ) as client:
                r = await prefetch_http_get(
                    "coinbase",
                    client,
                    f"{_COINBASE_EXCHANGE_BASE}/products/{product}/candles",
                    params=params,
                    headers=headers,
                )
            if diag is not None:
                diag.response_bytes = len(r.content or b"")
            if r.status_code != 200:
                if diag is not None:
                    diag.failure = f"http_{r.status_code}"
                if is_rate_limit_http(r.status_code):
                    stale = _stale_fallback(f"http_{r.status_code}")
                    if stale:
                        logger.info(
                            "[tradfi] Coinbase 429 → cache %s %s (%d bars)",
                            product,
                            timeframe,
                            len(stale),
                        )
                    return stale
                stale = _stale_fallback(f"http_{r.status_code}")
                return stale
            raw = r.json()
            if not isinstance(raw, list) or not raw:
                if diag is not None:
                    diag.failure = "empty_ohlc"
                return _stale_fallback("empty_ohlc")

            complete = raw[1:] if len(raw) > 1 else raw
            rows: List[List[float]] = []
            for bar in reversed(complete[-(limit + 1):]):
                try:
                    rows.append([
                        int(bar[0]) * 1000,
                        float(bar[3]),
                        float(bar[2]),
                        float(bar[1]),
                        float(bar[4]),
                        float(bar[5]) if len(bar) > 5 else 0.0,
                    ])
                except Exception:
                    continue
            rows = rows[-limit:]
            if diag is not None:
                diag.candle_count = len(rows)
            if rows:
                _KLINE_CACHE[key] = (rows, now)
                logger.info(
                    f"[tradfi] klines ok (Coinbase): {product} {timeframe} "
                    f"→ {len(rows)} bars"
                )
                return rows
            if diag is not None:
                diag.failure = "parse_empty"
            return _stale_fallback("parse_empty")
        except (httpx.ConnectTimeout, httpx.ReadTimeout, httpx.ConnectError) as exc:
            last_exc = exc
            from app.services.prefetch_fast import prefetch_fast_active
            if attempt == 0 and not prefetch_fast_active():
                await asyncio.sleep(0.75)
                continue
            if diag is not None:
                diag.failure = type(exc).__name__
            stale = _stale_fallback(type(exc).__name__)
            if stale:
                return stale
            logger.warning(
                f"[tradfi] Coinbase klines failed for {product} {timeframe}: {exc}"
            )
            return []
        except Exception as exc:
            last_exc = exc
            if diag is not None:
                diag.failure = type(exc).__name__
            stale = _stale_fallback(type(exc).__name__)
            if stale:
                return stale
            logger.warning(
                f"[tradfi] Coinbase klines failed for {product} {timeframe}: {exc}"
            )
            return []

    if last_exc and diag is not None:
        diag.failure = type(last_exc).__name__
    return _stale_fallback(diag.failure if diag and diag.failure else "unknown")


async def _fetch_kraken_metals_klines(
    symbol: str,
    timeframe: str,
    limit: int,
    *,
    diag: Optional[MetalProviderDiagnostic] = None,
) -> List[List[float]]:
    """Closed OHLC from Kraken PAXGUSD / XAGUSD (US-accessible spot proxy)."""
    pair = _KRAKEN_METAL_MAP.get(symbol.upper())
    interval = _KRAKEN_INTERVAL.get(timeframe)
    url = f"https://api.kraken.com/0/public/OHLC?pair={pair}&interval={interval}"
    if diag is not None:
        diag.provider = "kraken"
        diag.url = url
    if not pair or not interval:
        if diag is not None:
            diag.failure = "unsupported_pair_or_timeframe"
        return []

    now = datetime.utcnow()
    key = (f"kraken:{pair}", timeframe, limit)
    cached = _KLINE_CACHE.get(key)
    if cached and (now - cached[1]) < _metal_kline_cache_ttl(key):
        rows = cached[0]
        if diag is not None:
            diag.candle_count = len(rows)
            diag.extra["cache_hit"] = True
        return rows

    try:
        from app.services.prefetch_provider_limits import prefetch_http_get

        import httpx
        async with httpx.AsyncClient(timeout=_KRAKEN_KLINE_TIMEOUT_S) as client:
            r = await prefetch_http_get(
                "kraken",
                client,
                "https://api.kraken.com/0/public/OHLC",
                params={"pair": pair, "interval": interval},
            )
        if diag is not None:
            diag.response_bytes = len(r.content or b"")
        if r.status_code != 200:
            if diag is not None:
                diag.failure = f"http_{r.status_code}"
            stale = _KLINE_CACHE.get(key)
            if stale and (now - stale[1]) < _metal_kline_cache_ttl(key):
                from app.services.prefetch_provider_limits import consume_prefetch_429
                if consume_prefetch_429():
                    logger.warning(
                        "[tradfi] Kraken HTTP %s for %s → stale cache (%d bars)",
                        r.status_code, pair, len(stale[0]),
                    )
                    return stale[0]
            return []
        result = (r.json() or {}).get("result") or {}
        raw_bars = None
        for k, v in result.items():
            if k != "last" and isinstance(v, list):
                raw_bars = v
                break
        if not raw_bars:
            if diag is not None:
                diag.failure = "empty_ohlc"
            return []

        complete = raw_bars[:-1] if len(raw_bars) > 1 else raw_bars
        rows: List[List[float]] = []
        for bar in complete[-limit:]:
            try:
                rows.append([
                    int(bar[0]) * 1000,
                    float(bar[1]),
                    float(bar[2]),
                    float(bar[3]),
                    float(bar[4]),
                    float(bar[6]) if len(bar) > 6 else 0.0,
                ])
            except Exception:
                continue
        if diag is not None:
            diag.candle_count = len(rows)
        if rows:
            _KLINE_CACHE[key] = (rows, now)
            logger.info(
                f"[tradfi] klines ok (Kraken): {pair} {timeframe} → {len(rows)} bars"
            )
            return rows
        if diag is not None:
            diag.failure = "parse_empty"
    except Exception as exc:
        if diag is not None:
            diag.failure = type(exc).__name__
        logger.warning(
            f"[tradfi] Kraken klines failed for {pair} {timeframe}: {exc}"
        )
    return []


async def fetch_metal_live_candles(
    symbol: str,
    timeframe: str,
    limit: int,
    user_id: Optional[int] = None,
) -> List[List[float]]:
    """
    Parallel spot-metal OHLC for executor live/paper scans.
    Never returns GC=F futures — only spot-aligned sources.
    Falls back to synthetic spot-built candles when all providers miss.
    """
    from app.services.prefetch_fast import provider_timeout_s

    sym = symbol.upper()
    min_bars = _metal_kline_min_bars(limit)

    try:
        from app.services.prefetch_fast import prefetch_fast_active
        if prefetch_fast_active():
            cached_rows, cached_src = peek_cached_klines(
                sym, "forex", timeframe, limit, user_id,
            )
            _src = str(cached_src or "").lower()
            _accept_cached = (
                len(cached_rows) >= min(min_bars, 15)
                or (_src.startswith("ctrader") and len(cached_rows) >= 3)
            )
            if cached_rows and _accept_cached:
                from app.services.kline_staleness import check_cached_klines_stale

                stale, detail = await check_cached_klines_stale(
                    sym, cached_rows, timeframe,
                )
                if not stale:
                    logger.info(
                        f"[tradfi] prefetch-fast cache hit {sym} {timeframe} "
                        f"src={cached_src} → {len(cached_rows)} bars"
                    )
                    return cached_rows[-limit:] if len(cached_rows) > limit else cached_rows
                logger.warning(
                    "[tradfi] prefetch-fast stale cache bypass %s %s (src=%s, %s)",
                    sym,
                    timeframe,
                    cached_src or "unknown",
                    detail or "stale",
                )
    except Exception:
        pass

    _kraken_timeout = provider_timeout_s(max(8.0, _KRAKEN_KLINE_TIMEOUT_S))
    _fmp_timeout = provider_timeout_s(max(8.0, _FMP_KLINE_TIMEOUT_S))
    _ct_timeout = provider_timeout_s(_CTRADER_KLINE_TIMEOUT_LIVE_S)
    trace: List[MetalProviderDiagnostic] = []

    async with _metal_live_fetch_sem():
        broker_ready = False
        ct_spot_ready = False
        try:
            from app.services.ctrader_price_feed import (
                broker_session_ready as _ct_ready,
                ctrader_spot_ready as _ct_spot,
                is_live as _ct_live,
            )
            broker_ready = bool(_ct_ready(sym))
            ct_spot_ready = bool(_ct_spot(sym) or _ct_live())
        except Exception:
            pass

        # cTrader first when broker session is up or spot ticks are streaming
        ctrader_tried = False
        if broker_ready or ct_spot_ready:
            ctrader_tried = True
            ct_diag = MetalProviderDiagnostic(
                provider="ctrader-user" if user_id else "ctrader",
                url=f"ctrader://trendbars/{sym}/{timeframe}",
            )
            ct_rows = await _fetch_with_kline_timeout(
                _fetch_ctrader_klines(
                    sym, "forex", timeframe, limit, user_id=user_id, diag=ct_diag,
                ),
                timeout_s=_ct_timeout,
                label=ct_diag.provider,
                symbol=sym,
                timeframe=timeframe,
            )
            ct_diag.candle_count = len(ct_rows) if ct_rows else 0
            if not ct_rows and not ct_diag.failure:
                ct_diag.failure = "empty_or_timeout"
            trace.append(ct_diag)
            if ct_rows and (len(ct_rows) >= min_bars or len(ct_rows) >= 15):
                label = ct_diag.provider
                logger.info(
                    f"[prices] {sym} kline_source=ctrader {timeframe} "
                    f"→ {len(ct_rows)} bars (broker_session={broker_ready})"
                )
                out = ct_rows[-limit:] if len(ct_rows) > limit else ct_rows
                _METAL_KLINE_SOURCE_CACHE[(sym, timeframe, int(limit))] = (
                    label, datetime.utcnow(),
                )
                _log_metal_kline_trace(sym, timeframe, trace)
                return out
            if ct_rows:
                logger.debug(
                    f"[tradfi] metal-live cTrader thin {sym} {timeframe} "
                    f"({len(ct_rows)} bars) — trying externals"
                )
            else:
                logger.info(
                    f"[tradfi] metal-live cTrader empty {sym} {timeframe} "
                    f"— falling back to externals"
                )

        _kraken_pair = _KRAKEN_METAL_MAP.get(sym, "")
        _kraken_iv = _KRAKEN_INTERVAL.get(timeframe, "")
        _cb_product = _COINBASE_METAL_PRODUCT.get(sym, "")

        async def _probe_seq(
            label: str, url: str, timeout_s: float, fetch_coro,
        ) -> Tuple[str, List[List[float]], MetalProviderDiagnostic]:
            diag = MetalProviderDiagnostic(provider=label, url=url)
            rows = await _fetch_with_kline_timeout(
                fetch_coro(diag),
                timeout_s=timeout_s,
                label=label,
                symbol=sym,
                timeframe=timeframe,
            )
            if diag.candle_count == 0:
                diag.candle_count = len(rows) if rows else 0
            if not rows and not diag.failure:
                diag.failure = "empty_or_timeout"
            return label, rows, diag

        def _accept(label: str, rows: List[List[float]]) -> Optional[List[List[float]]]:
            if not rows:
                return None
            if len(rows) >= min_bars or len(rows) >= 15:
                out = rows[-limit:] if len(rows) > limit else rows
                _METAL_KLINE_SOURCE_CACHE[(sym, timeframe, int(limit))] = (
                    label, datetime.utcnow(),
                )
                if label == "coinbase" and ctrader_tried:
                    logger.warning(
                        "[prices] %s fell back to coinbase (ctrader feed down: %s)",
                        sym,
                        _ctrader_unavailable_reason(sym),
                    )
                logger.info(
                    f"[prices] {sym} kline_source={label} {timeframe} → {len(out)} bars"
                )
                return out
            logger.info(
                f"[tradfi] metal-live partial {label} {sym} {timeframe}: "
                f"{len(rows)} bars < {min_bars}"
            )
            return None

        # Sequential failover — sources after FMP ordered by Railway reliability.
        _cb_probe_timeout = provider_timeout_s(
            _COINBASE_CONNECT_TIMEOUT_S + _COINBASE_KLINE_TIMEOUT_S + 3.0
        )
        _kraken_entry = (
            "kraken",
            f"https://api.kraken.com/0/public/OHLC?pair={_kraken_pair}&interval={_kraken_iv}",
            _kraken_timeout,
            lambda d: _fetch_kraken_metals_klines(sym, timeframe, limit, diag=d),
        )
        _coinbase_entry = (
            "coinbase",
            f"{_COINBASE_EXCHANGE_BASE}/products/{_cb_product}/candles",
            _cb_probe_timeout,
            lambda d: _fetch_coinbase_metals_klines(sym, timeframe, limit, diag=d),
        )
        _binance_entry = (
            "binance",
            f"{_BINANCE_SPOT_BASE}/klines?symbol={_METALS_BINANCE_MAP.get(sym, '')}",
            provider_timeout_s(5.0),
            lambda d: _fetch_binance_metals_klines(sym, timeframe, limit, diag=d),
        )
        external_chain: List[Tuple[str, str, float, object]] = []
        if _cb_product:
            external_chain.append(_coinbase_entry)
        if sym in _METALS_BINANCE_MAP:
            external_chain.append(_binance_entry)
        if sym in _KRAKEN_METAL_MAP:
            external_chain.append(_kraken_entry)
        # FMP last — often rate-limited on free tier; Coinbase/cTrader are reliable on Railway.
        if _env_fmp_api_key():
            try:
                from app.services.fmp_price_feed import fmp_in_backoff
                if not fmp_in_backoff():
                    external_chain.append((
                        "fmp",
                        f"fmp://historical-chart/{sym}/{timeframe}",
                        _fmp_timeout,
                        lambda d: _fetch_fmp_metals_klines(sym, "forex", timeframe, limit, diag=d),
                    ))
            except Exception:
                external_chain.append((
                    "fmp",
                    f"fmp://historical-chart/{sym}/{timeframe}",
                    _fmp_timeout,
                    lambda d: _fetch_fmp_metals_klines(sym, "forex", timeframe, limit, diag=d),
                ))

        thin_best: Tuple[str, List[List[float]]] = ("", [])
        for label, url, timeout_s, fetch_fn in external_chain:
            lbl, rows, diag = await _probe_seq(label, url, timeout_s, fetch_fn)
            trace.append(diag)
            accepted = _accept(lbl, rows)
            if accepted is not None:
                _log_metal_kline_trace(sym, timeframe, trace)
                return accepted
            if rows and len(rows) > len(thin_best[1]):
                thin_best = (lbl, rows)

        if thin_best[1]:
            label, rows = thin_best
            logger.info(
                f"[tradfi] metal-live thin fallback {label} {sym} {timeframe} "
                f"→ {len(rows)} bars"
            )
            out = rows[-limit:] if len(rows) > limit else rows
            _METAL_KLINE_SOURCE_CACHE[(sym, timeframe, int(limit))] = (
                label, datetime.utcnow(),
            )
            _log_metal_kline_trace(sym, timeframe, trace)
            return out

        # Last resort: Yahoo GC=F futures when all spot sources miss (Railway geo-block).
        _yt = _METALS_YAHOO_TICKER.get(sym)
        if _yt:
            y_diag = MetalProviderDiagnostic(
                provider="yahoo_gc",
                url=f"https://query1.finance.yahoo.com/v8/finance/chart/{_yt}",
            )
            yrows = await _fetch_with_kline_timeout(
                _fetch_yahoo_chart_klines(_yt, timeframe, limit),
                timeout_s=20.0,
                label="yahoo_gc",
                symbol=sym,
                timeframe=timeframe,
            )
            y_diag.candle_count = len(yrows) if yrows else 0
            if not yrows:
                y_diag.failure = "empty_or_timeout"
            trace.append(y_diag)
            accepted = _accept("yahoo_gc", yrows)
            if accepted is not None:
                logger.warning(
                    f"[tradfi] metal-live GC=F fallback {sym} {timeframe} → "
                    f"{len(accepted)} bars (all spot sources missed)"
                )
                _log_metal_kline_trace(sym, timeframe, trace)
                return accepted

        _log_metal_kline_trace(sym, timeframe, trace)
        logger.warning(
            f"[tradfi] metal-live no spot klines for {sym} {timeframe} "
            f"(ctrader/coinbase/kraken/fmp all missed) — building synthetic"
        )
        synth = await build_synthetic_metal_candles(sym, timeframe, limit)
        if synth:
            synth_diag = MetalProviderDiagnostic(
                provider="synthetic",
                url="metals_spot_feed://synthetic",
                candle_count=len(synth),
            )
            trace.append(synth_diag)
            _log_metal_kline_trace(sym, timeframe, trace)
            _METAL_KLINE_SOURCE_CACHE[(sym, timeframe, int(limit))] = (
                "synthetic", datetime.utcnow(),
            )
            return synth[-limit:] if len(synth) > limit else synth

        synth_diag = MetalProviderDiagnostic(
            provider="synthetic",
            url="metals_spot_feed://synthetic",
            failure="no_spot_anchor",
        )
        trace.append(synth_diag)
        _log_metal_kline_trace(sym, timeframe, trace)
        return []


async def fetch_metal_scan_candles(
    symbol: str,
    timeframe: str,
    limit: int,
    user_id: Optional[int] = None,
) -> List[List[float]]:
    """
    Multi-source fetch for gold/silver backtest scanners.
    Spot sources first; Yahoo GC=F only as a late backtest fallback.
    """
    sym = symbol.upper()
    yahoo_ticker = _METALS_YAHOO_TICKER.get(sym)
    best: List[List[float]] = []
    best_label = ""

    async def _keep(rows: List[List[float]], label: str) -> None:
        nonlocal best, best_label
        if rows and len(rows) > len(best):
            best = rows
            best_label = label
            logger.info(
                f"[tradfi] metal-scan best={label} {sym} {timeframe} → {len(rows)} bars"
            )

    live_rows = await fetch_metal_live_candles(sym, timeframe, limit, user_id=user_id)
    if live_rows:
        return live_rows[-limit:] if len(live_rows) > limit else live_rows

    if yahoo_ticker:
        yrows = await _fetch_yahoo_chart_klines(yahoo_ticker, timeframe, limit)
        if yrows and await metal_klines_match_live_spot(sym, yrows, asset_class="forex"):
            await _keep(yrows, "yahoo-aligned")
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

    try:
        from app.services.prefetch_fast import prefetch_fast_active
        if prefetch_fast_active():
            cached_rows, cached_src = peek_cached_klines(
                sym, "forex", timeframe, limit, user_id,
            )
            _min_early = _scan_min_bars(limit)
            _src = str(cached_src or "").lower()
            _accept_cached = (
                len(cached_rows) >= _min_early
                or (_src.startswith("ctrader") and len(cached_rows) >= 3)
            )
            if cached_rows and _accept_cached:
                logger.info(
                    f"[tradfi] prefetch-fast cache hit {sym} {timeframe} "
                    f"src={cached_src} → {len(cached_rows)} bars"
                )
                return cached_rows[-limit:] if len(cached_rows) > limit else cached_rows
    except Exception:
        pass

    async def _keep(rows: List[List[float]], label: str) -> None:
        nonlocal best
        if rows and len(rows) > len(best):
            best = rows
            logger.info(f"[tradfi] forex-scan best={label} {sym} {timeframe} → {len(rows)} bars")

    _ctrader_live = False
    _broker_ready = False
    try:
        from app.services.ctrader_price_feed import (
            broker_session_ready as _ct_ready,
            is_live as _ct_live,
        )
        _ctrader_live = bool(_ct_live())
        _broker_ready = bool(_ct_ready(sym))
    except Exception:
        pass

    _min_bars = _scan_min_bars(limit)

    async def _try_ctrader(label: str) -> bool:
        rows = await _fetch_ctrader_klines(
            sym, "forex", timeframe, min(limit, 500), user_id=user_id,
        )
        await _keep(rows, label)
        return len(best) >= _min_bars

    # Priority: cTrader → Yahoo → AlphaVantage → tradfi-chain → FMP → TwelveData (last)
    if _broker_ready:
        if await _try_ctrader("ctrader-user" if user_id else "ctrader"):
            return best[-limit:]

    if yahoo_ticker:
        await _keep(await _fetch_yahoo_chart_klines(yahoo_ticker, timeframe, limit), "yahoo")
        if len(best) >= _min_bars:
            return best[-limit:]

    try:
        from app.services.alpha_vantage_feed import fetch_klines as _av_klines
        await _keep(await _av_klines(sym, timeframe, limit), "alphavantage")
        if len(best) >= _min_bars:
            return best[-limit:]
    except Exception:
        pass

    if _broker_ready and user_id:
        if await _try_ctrader("ctrader"):
            return best[-limit:]

    tradfi_rows = await _get_klines_impl(
        sym, "forex", timeframe, limit, for_backtest=True, ctrader_user_id=user_id,
    )
    await _keep(tradfi_rows, "tradfi-chain")
    if len(best) >= _min_bars:
        return best[-limit:]

    if _env_fmp_api_key():
        try:
            from app.services.fmp_price_feed import fmp_in_backoff, get_klines as _fmp_klines
            if not fmp_in_backoff():
                await _keep(await _fmp_klines(sym, "forex", timeframe, limit), "fmp")
        except Exception:
            pass

    try:
        from app.services.twelve_data_feed import can_request, fetch_klines as _td_klines
        if can_request(scanner_ok=True):
            await _keep(
                await _td_klines(sym, timeframe, limit, "forex", scanner_ok=True),
                "twelvedata",
            )
    except Exception:
        pass

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


async def get_metal_live_for_source(
    symbol: str,
    source: Optional[str],
    *,
    user_id: Optional[int] = None,
) -> Optional[Tuple[float, str]]:
    """
    Live tick from the same provider as metal klines (for drift guard).
    Coinbase klines use PAXG-USD — never compare to XAU-USD spot.
    """
    sym = symbol.upper().replace("/", "").replace("-", "")
    src = (source or "").lower()
    if src in ("ctrader", "ctrader-user"):
        try:
            from app.services.ctrader_price_feed import get_bid_ask, get_price
            tick = get_bid_ask(sym)
            if tick:
                mid = round((tick[0] + tick[1]) / 2.0, 6)
                if mid > 0:
                    return mid, "ctrader"
            px = get_price(sym)
            if px and px > 0:
                return float(px), "ctrader"
        except Exception:
            pass
        try:
            from app.services.realtime_spot import _read_ctrader
            hit = _read_ctrader(sym, max_age=30.0)
            if hit:
                return hit
        except Exception:
            pass
        return None
    if src == "coinbase":
        product = _COINBASE_METAL_PRODUCT.get(sym)
        if product:
            try:
                import httpx
                async with httpx.AsyncClient(timeout=4.0) as client:
                    r = await client.get(
                        f"https://api.coinbase.com/v2/prices/{product}/spot"
                    )
                if r.status_code == 200:
                    amount = ((r.json() or {}).get("data") or {}).get("amount")
                    px = float(amount) if amount else 0.0
                    if px > 0:
                        return px, "coinbase"
            except Exception:
                pass
        return None
    if src == "kraken":
        pair = _KRAKEN_METAL_MAP.get(sym)
        if pair:
            try:
                import httpx
                async with httpx.AsyncClient(timeout=4.0) as client:
                    r = await client.get(
                        "https://api.kraken.com/0/public/Ticker",
                        params={"pair": pair},
                    )
                if r.status_code == 200:
                    result = (r.json() or {}).get("result") or {}
                    for v in result.values():
                        if not isinstance(v, dict):
                            continue
                        c = v.get("c") or []
                        if c:
                            px = float(c[0])
                            if px > 0:
                                return px, "kraken"
            except Exception:
                pass
        return None
    if src == "binance":
        from app.services.binance_feed import binance_disabled, binance_spot_price
        bn_sym = _METALS_BINANCE_MAP.get(sym)
        if bn_sym and not binance_disabled():
            try:
                import httpx
                async with httpx.AsyncClient(timeout=5.0) as client:
                    px = await binance_spot_price(client, bn_sym)
                if px and px > 0:
                    return float(px), "binance"
            except Exception:
                pass
        return None
    return None


async def get_price_fresh(
    symbol: str,
    asset_class: str,
    *,
    paper_ok: bool = False,
    user_id: Optional[int] = None,
) -> Optional[float]:
    """Bypass caches — cTrader first, then FMP/yfinance before entry_price."""
    try:
        from app.services.realtime_spot import get_realtime_spot
        px = await get_realtime_spot(
            symbol,
            asset_class,
            force_fetch=True,
            paper_ok=paper_ok,
            user_id=user_id,
            twelve_data_ok=True,
        )
        if px is not None and px > 0:
            return px
    except Exception:
        pass
    return None


async def confirm_entry_price(
    symbol: str,
    asset_class: str,
    proposed: float,
    *,
    paper_ok: bool = False,
    user_id: Optional[int] = None,
) -> Tuple[Optional[float], str]:
    """
    Re-fetch live spot at fire time; reject stale proposed prices.
    Returns (confirmed_entry, reason) or (None, reason).
    """
    if not proposed or proposed <= 0:
        return None, "invalid_proposed"
    live = await get_price_fresh(
        symbol, asset_class, paper_ok=paper_ok, user_id=user_id,
    )
    if not live or live <= 0:
        try:
            from app.services.realtime_spot import read_fresh_cached
            hit = read_fresh_cached(symbol, asset_class, paper_ok=paper_ok)
            if hit and hit[0] > 0:
                live = float(hit[0])
        except Exception:
            pass
    if not live or live <= 0:
        try:
            from app.services.ctrader_price_feed import get_bid_ask as _ba, get_price as _ctp
            sym = symbol.upper().replace("/", "").replace("-", "")
            live = _ctp(sym)
            if not live or live <= 0:
                tick = _ba(sym)
                if tick:
                    live = round((tick[0] + tick[1]) / 2.0, 6)
        except Exception:
            pass
    if not live or live <= 0:
        if paper_ok:
            return proposed, "paper_proposed_at_fire"
        return None, "no_live_spot_at_fire"

    max_drift = (
        METAL_SPOT_KLINE_MAX_DRIFT_PCT
        if is_metal_symbol(symbol)
        else float(os.environ.get("ENTRY_PRICE_MAX_DRIFT_PCT", "0.15"))
    )
    drift_pct = abs(live - proposed) / live * 100.0
    if drift_pct > max_drift:
        return None, (
            f"proposed_stale drift={drift_pct:.2f}% "
            f"proposed={proposed:.4f} live={live:.4f}"
        )
    return live, "fresh_spot_confirmed"


async def get_price(symbol: str, asset_class: str) -> Optional[float]:
    """
    Latest trade price.
    Metals/forex/index: strict real-time spot resolver (never stale / futures).
    Others: FMP real-time feed → yfinance fast_info fallback.
    """
    cls = normalize_asset_class(asset_class)
    if cls == ASSET_CLASS_CRYPTO:
        return None

    # ── Real-time spot (cTrader → shared store → parallel externals) ─────────
    try:
        from app.services.realtime_spot import get_realtime_spot
        px = await get_realtime_spot(symbol, asset_class, force_fetch=False)
        if px is not None and px > 0:
            return px
    except Exception:
        pass

    # Metals must NOT fall through to yfinance GC=F — that's gold FUTURES.
    if symbol.upper() in _METALS_BINANCE_MAP:
        return None

    # ── FMP (forex/index only — skip when rate-limited) ─
    try:
        from app.services.fmp_price_feed import fmp_in_backoff, get_price as _fmp_price
        if not fmp_in_backoff():
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
            cached_rows, cached_src = peek_cached_klines(
                symbol, asset_class, timeframe, limit, ctrader_user_id,
            )
            if cached_rows:
                logger.warning(
                    f"[tradfi] get_klines TIMEOUT → cached {symbol.upper()} "
                    f"{timeframe} src={cached_src} ({len(cached_rows)} bars)"
                )
                if not fut.done():
                    fut.set_result(cached_rows)
                return cached_rows
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
    *,
    diag: Optional[MetalProviderDiagnostic] = None,
) -> List[List[float]]:
    if diag is not None:
        diag.provider = "fmp"
        diag.url = diag.url or f"fmp://historical-chart/{symbol.upper()}/{timeframe}"
    if not _env_fmp_api_key():
        if diag is not None:
            diag.failure = "no_api_key"
        return []
    try:
        from app.services.fmp_price_feed import fmp_in_backoff, get_klines as _fmp_klines
        if fmp_in_backoff():
            if diag is not None:
                diag.failure = "rate_limit_backoff"
            return []
        rows = await _fmp_klines(symbol, asset_class, timeframe, limit)
        if diag is not None:
            diag.candle_count = len(rows) if rows else 0
            if not rows:
                diag.failure = diag.failure or "empty_response"
        if rows:
            logger.info(
                f"[tradfi] klines ok (FMP): {symbol.upper()} {timeframe} → {len(rows)} bars"
            )
        return rows
    except Exception as fe:
        if diag is not None:
            diag.failure = type(fe).__name__
        logger.warning(f"[tradfi] FMP klines failed {symbol} {timeframe}: {fe}")
        return []


async def _fetch_ctrader_klines(
    symbol: str,
    asset_class: str,
    timeframe: str,
    limit: int,
    user_id: Optional[int] = None,
    *,
    diag: Optional[MetalProviderDiagnostic] = None,
) -> List[List[float]]:
    if diag is not None:
        diag.url = diag.url or f"ctrader://trendbars/{symbol.upper()}/{timeframe}"
    try:
        from app.services import ctrader_price_feed as _ctf
        # Short timeout when feed is cold — Yahoo fallback is the scan workhorse.
        try:
            _live = bool(_ctf.is_live())
        except Exception:
            _live = False
        timeout = (
            min(_CTRADER_KLINE_TIMEOUT_LIVE_S, 8.0)
            if _live
            else min(6.0, 3.0 + limit / 200.0)
        )
        rows = await asyncio.wait_for(
            _ctf.get_klines(symbol, asset_class, timeframe, limit, user_id=user_id),
            timeout=timeout,
        )
        if diag is not None:
            diag.candle_count = len(rows) if rows else 0
            if not rows:
                diag.failure = diag.failure or "empty_response"
        return rows
    except asyncio.TimeoutError:
        if diag is not None:
            diag.failure = "timeout"
        logger.info(
            f"[tradfi] cTrader klines timeout {symbol.upper()} {timeframe} "
            f"(limit={limit})"
        )
        return []
    except Exception as exc:
        if diag is not None:
            diag.failure = type(exc).__name__
        logger.debug(
            f"[tradfi] cTrader klines failed {symbol.upper()} {timeframe}: {exc}"
        )
        return []


async def _fetch_binance_metals_klines(
    symbol: str,
    timeframe: str,
    limit: int,
    *,
    diag: Optional[MetalProviderDiagnostic] = None,
) -> List[List[float]]:
    """Closed OHLC from Binance spot XAUUSDT / XAGUSDT (forming bar stripped)."""
    from app.services.binance_feed import binance_disabled, binance_spot_klines

    sym = symbol.upper()
    bn_sym = _METALS_BINANCE_MAP.get(sym)
    if diag is not None:
        diag.provider = "binance"
        diag.url = f"{_BINANCE_SPOT_BASE}/klines?symbol={bn_sym}"
    if not bn_sym or binance_disabled():
        if diag is not None:
            diag.failure = "disabled" if binance_disabled() else "unsupported_pair"
        return []

    bn_interval = _BINANCE_INTERVAL_MAP.get(timeframe, timeframe)
    now = datetime.utcnow()
    key = (f"binance:{bn_sym}", timeframe, limit)
    cached = _KLINE_CACHE.get(key)
    if cached and (now - cached[1]) < _metal_kline_cache_ttl(key):
        rows = cached[0]
        if diag is not None:
            diag.candle_count = len(rows)
            diag.extra["cache_hit"] = True
        return rows

    try:
        import httpx
        async with httpx.AsyncClient(timeout=_BINANCE_KLINE_TIMEOUT_S) as client:
            raw = await binance_spot_klines(client, bn_sym, bn_interval, limit)
        if not raw:
            if diag is not None:
                diag.failure = "empty_or_timeout"
            return []
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
        if diag is not None:
            diag.candle_count = len(rows)
        if rows:
            _KLINE_CACHE[key] = (rows, now)
            logger.info(
                f"[tradfi] klines ok (Binance spot): {bn_sym} {bn_interval} → {len(rows)} bars"
            )
            return rows
        if diag is not None:
            diag.failure = "parse_empty"
    except Exception as exc:
        if diag is not None:
            diag.failure = type(exc).__name__
        logger.debug(f"[tradfi] Binance spot klines failed {bn_sym}: {exc}")
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

    # Live/paper metals: parallel spot OHLC — never GC=F futures on this path.
    if is_metal and not for_backtest:
        try:
            scan_rows = await fetch_metal_live_candles(
                sym_norm, timeframe, limit, user_id=ctrader_user_id,
            )
            if scan_rows:
                return scan_rows[-limit:] if len(scan_rows) > limit else scan_rows
        except Exception as _mse:
            logger.warning(
                f"[tradfi] fetch_metal_live_candles failed {sym_norm} "
                f"{timeframe}: {_mse}"
            )
        synth = await build_synthetic_metal_candles(sym_norm, timeframe, limit)
        if synth:
            _METAL_KLINE_SOURCE_CACHE[(sym_norm, timeframe, int(limit))] = (
                "synthetic", datetime.utcnow(),
            )
            return synth[-limit:] if len(synth) > limit else synth
        logger.warning(
            f"[tradfi] metals spot klines unavailable for {sym_norm} {timeframe} "
            f"— refusing GC=F futures (spot feed required for live/paper fires)"
        )
        return []

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

    # Live/paper indices: Yahoo chart chain (executor used to skip this → yfinance-only).
    if cls == ASSET_CLASS_INDEX and not for_backtest:
        try:
            scan_rows = await fetch_index_scan_candles(
                sym_norm, timeframe, limit, user_id=ctrader_user_id,
            )
            if scan_rows:
                return scan_rows[-limit:] if len(scan_rows) > limit else scan_rows
        except Exception as _ise:
            logger.debug(f"[tradfi] fetch_index_scan_candles failed {sym_norm}: {_ise}")

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
            try:
                from app.services.ctrader_price_feed import is_live as _ct_live
                if not _ct_live():
                    continue
            except Exception:
                continue
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

    # ── Coinbase spot metals fallback (backtest / after cTrader+FMP miss) ──────
    if is_metal:
        _cbrows = await _fetch_coinbase_metals_klines(symbol, timeframe, limit)
        if _cbrows:
            return _cbrows
        _krows = await _fetch_kraken_metals_klines(symbol, timeframe, limit)
        if _krows:
            return _krows

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

    # ── Yahoo chart API — metals futures (GC=F / SI=F) ONLY if aligned with spot ─
    if is_metal:
        _yt = _METALS_YAHOO_TICKER.get(symbol.upper())
        if _yt:
            _yrows = await _fetch_yahoo_chart_klines(_yt, timeframe, limit)
            if _yrows and not for_backtest:
                if await metal_klines_match_live_spot(symbol, _yrows, asset_class=cls):
                    return _yrows
                _yrows = []
            elif _yrows:
                return _yrows
        if not for_backtest:
            logger.warning(
                f"[tradfi] metals spot klines unavailable for {symbol.upper()} {timeframe} "
                f"— refusing GC=F futures (spot feed required for live/paper fires)"
            )
            return []
        logger.warning(
            f"[tradfi] metals spot klines unavailable for {symbol.upper()} {timeframe} "
            f"— trying yfinance futures as last resort (backtest only)"
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
        if is_metal and not for_backtest:
            if not await metal_klines_match_live_spot(symbol, rows, asset_class=cls):
                return []
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
