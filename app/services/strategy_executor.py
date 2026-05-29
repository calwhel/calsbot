"""
Strategy Executor — Build Your Own Strategy Portal

Background worker that continuously evaluates active + paper user strategies
and fires trades. Paper trades are tracked with 1m OHLC accuracy — candle
high/low is used to detect TP/SL hits so scalp results are realistic.
"""
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import httpx

logger = logging.getLogger(__name__)

import os as _os_env

# Env-overridable so we can dial pressure on prod without a redeploy.
# Defaults raised from 3→10s and 5→3 because Neon's compute was getting
# saturated — every QueryCanceled in the executor cascaded into HTTP 500s
# on /api/strategies because all pool connections were held by hung scans.
SCAN_INTERVAL_SECONDS       = int(_os_env.environ.get("EXECUTOR_SCAN_INTERVAL", "10"))
FOREX_SCAN_INTERVAL_SECONDS = int(_os_env.environ.get("EXECUTOR_FOREX_SCAN_INTERVAL", "10"))
PAPER_MONITOR_INTERVAL      = int(_os_env.environ.get("EXECUTOR_MONITOR_INTERVAL", "20"))
MAX_CONCURRENT              = int(_os_env.environ.get("EXECUTOR_MAX_CONCURRENT", "3"))
PAPER_MAX_HOLD_HOURS        = 168   # auto-expire paper positions after this many hours (7 days)

# ─── Shared API caches ───────────────────────────────────────────────────────
# Raw ticker list — fetched once per scan cycle, shared by ALL strategies.
# Eliminates N×ticker-fetch where N = number of active strategies.
_RAW_TICKERS_CACHE: Optional[list] = None
_RAW_TICKERS_AT:    Optional[datetime] = None
_RAW_TICKERS_TTL    = 60  # seconds

# Price / TA data — cached per symbol, TTL 30 s.
# When multiple strategies scan the same symbol in the same cycle they all
# hit the cache instead of each making an independent candle + indicator call.
_PRICE_TA_CACHE: Dict[str, tuple] = {}  # symbol -> (data_dict, fetched_at)
_PRICE_TA_TTL    = 15  # seconds — fresher data for faster signal detection

# ─── Helpers ─────────────────────────────────────────────────────────────────

def _telegram_int_id(user) -> Optional[int]:
    """
    Return the integer Telegram chat_id for a user.
    Returns None for web-registered users (telegram_id starts with 'WEB-')
    or any ID that cannot be converted to int, so callers can safely skip DMs.
    """
    tid = getattr(user, "telegram_id", None)
    if not tid or str(tid).startswith("WEB-"):
        return None
    try:
        return int(tid)
    except (ValueError, TypeError):
        return None


def _user_can_live_trade(user, db) -> bool:
    """
    Pre-flight check: return True only if the user has Bitunix auto-trading
    enabled AND API keys saved.  Live strategies silently downgrade to paper
    for this signal when this returns False — so no signal is ever dropped.

    NOTE: This is the LOCAL check only (keys + opt-in). It does NOT enforce
    the Bitunix affiliate-roster gate. For live-trade decisions inside async
    code, prefer `_user_can_live_trade_async()` which adds the affiliate
    check on top.

    Retries once with a fresh SessionLocal on any DB/SSL error so that a
    transient Neon connection drop doesn't silently send a subscriber to paper.
    """
    from app.models import UserPreference

    def _check(session) -> bool:
        prefs = session.query(UserPreference).filter_by(user_id=user.id).first()
        if not prefs:
            return False
        if not getattr(prefs, "auto_trading_enabled", False):
            return False
        if not getattr(prefs, "bitunix_api_key", None):
            return False
        if not getattr(prefs, "bitunix_api_secret", None):
            return False
        return True

    try:
        return _check(db)
    except Exception as _e1:
        logger.warning(
            f"[_user_can_live_trade] DB error for user {user.id}, retrying with fresh session: {_e1}"
        )
        try:
            db.rollback()
        except Exception:
            pass
        try:
            from app.database import BgSessionLocal as _SL
            _fresh = _SL()
            try:
                return _check(_fresh)
            finally:
                _fresh.close()
        except Exception as _e2:
            logger.error(
                f"[_user_can_live_trade] Retry also failed for user {user.id}: {_e2} — defaulting to paper"
            )
            return False


async def _user_can_live_trade_async(user, db) -> Tuple[bool, str]:
    """
    Async live-trade gate: combines the local sync check (auto_trading_enabled
    + Bitunix keys present) with the Bitunix Partner / Affiliate roster check
    ("they have to be under me to trade").

    Returns (allowed, reason). Reason values:
      ok | gate_off | local_check_failed | no_bitunix_uid_on_user
      | partner_api_not_configured | uid_not_under_master
      | partner_api_error:<...> | affiliate_check_error:<...>

    Live strategies silently downgrade to paper when this returns False — no
    signal is ever dropped, just tracked as paper for ROI accounting.
    """
    if not _user_can_live_trade(user, db):
        return False, "local_check_failed"
    try:
        from app.services.bitunix_partner import is_uid_affiliated
        uid = getattr(user, "bitunix_uid", None)
        return await is_uid_affiliated(uid)
    except Exception as _e:
        logger.warning(
            f"[_user_can_live_trade_async] Affiliate check threw for user {user.id}: "
            f"{_e} — defaulting to paper (fail-closed)."
        )
        return False, f"affiliate_check_error:{_e}"


# ─── Bitunix symbol cache ────────────────────────────────────────────────────
_BITUNIX_SYMBOLS: set = set()
_BITUNIX_SYMBOLS_FETCHED_AT: Optional[datetime] = None
_BITUNIX_CACHE_TTL = 300  # refresh every 5 minutes
# Single-flight lock — when the cache expires and many strategies hit the
# refresh path simultaneously, we don't want N concurrent identical requests
# hammering Bitunix (which causes rate-limit failures and the "5 warnings in
# 5ms" log bursts). The lock ensures only ONE refresh runs at a time; other
# callers wait briefly and then read the just-refreshed cache.
_BITUNIX_FETCH_LOCK: Optional[asyncio.Lock] = None
# How recently a failed refresh ran — suppresses thundering retries when
# Bitunix is down (only one attempt per CACHE_TTL window after a failure).
_BITUNIX_LAST_FAIL_AT: Optional[datetime] = None
_BITUNIX_FAIL_BACKOFF = 30  # seconds — don't re-attempt for this long after a failure

# Diagnostic log throttle for strategies whose universe resolves to no symbols.
# Keyed by (strategy_id, minute_bucket) so we get at most one warning per
# strategy per minute — enough to identify misconfigured strategies without
# spamming the log when an outage affects many strategies at once.
_EMPTY_UNI_LOGGED: set = set()

# Same throttle pattern for Bitunix-fetch warnings — prevents the 5-warnings-
# in-5ms bursts when concurrent callers all hit the same outage.
_BITUNIX_WARN_LAST: Optional[datetime] = None
_BITUNIX_WARN_INTERVAL = 60  # seconds between warnings

async def _get_bitunix_symbols(http_client: httpx.AsyncClient) -> set:
    """Return the set of USDT-margined perpetual symbols available on Bitunix.

    STICKY CACHE: once populated, the previously cached set is returned even
    when a refresh attempt fails — this prevents transient Bitunix outages
    (rate limits, timeouts, 5xx) from emptying the strategy universe and
    halting all trades. Caller should derive from tickers only if the cache
    has *never* been populated."""
    global _BITUNIX_SYMBOLS, _BITUNIX_SYMBOLS_FETCHED_AT, _BITUNIX_FETCH_LOCK
    global _BITUNIX_LAST_FAIL_AT, _BITUNIX_WARN_LAST
    now = datetime.utcnow()

    # Fast path: cache fresh.
    if (
        _BITUNIX_SYMBOLS
        and _BITUNIX_SYMBOLS_FETCHED_AT
        and (now - _BITUNIX_SYMBOLS_FETCHED_AT).total_seconds() < _BITUNIX_CACHE_TTL
    ):
        return _BITUNIX_SYMBOLS

    # Recently failed — short-circuit to avoid thundering retries.
    if (
        _BITUNIX_LAST_FAIL_AT
        and (now - _BITUNIX_LAST_FAIL_AT).total_seconds() < _BITUNIX_FAIL_BACKOFF
    ):
        return _BITUNIX_SYMBOLS  # sticky: empty if never warmed, populated otherwise

    # Lazy-init the lock (must happen inside an event loop).
    if _BITUNIX_FETCH_LOCK is None:
        _BITUNIX_FETCH_LOCK = asyncio.Lock()

    async with _BITUNIX_FETCH_LOCK:
        # Re-check after acquiring the lock — a concurrent caller may have
        # already refreshed while we were waiting.
        now = datetime.utcnow()
        if (
            _BITUNIX_SYMBOLS
            and _BITUNIX_SYMBOLS_FETCHED_AT
            and (now - _BITUNIX_SYMBOLS_FETCHED_AT).total_seconds() < _BITUNIX_CACHE_TTL
        ):
            return _BITUNIX_SYMBOLS
        if (
            _BITUNIX_LAST_FAIL_AT
            and (now - _BITUNIX_LAST_FAIL_AT).total_seconds() < _BITUNIX_FAIL_BACKOFF
        ):
            return _BITUNIX_SYMBOLS

        _fail_reason: Optional[str] = None
        try:
            resp = await http_client.get(
                "https://fapi.bitunix.com/api/v1/futures/market/tickers", timeout=8
            )
            if resp.status_code == 200:
                try:
                    data = resp.json()
                except Exception as _je:
                    _fail_reason = f"invalid JSON ({type(_je).__name__})"
                    data = None
                if data is not None:
                    syms = set()
                    for t in (data.get("data") or []):   # guard against null data field
                        sym = t.get("symbol", "")
                        if sym.endswith("USDT"):
                            syms.add(sym)
                    if syms:
                        _BITUNIX_SYMBOLS = syms
                        _BITUNIX_SYMBOLS_FETCHED_AT = now
                        _BITUNIX_LAST_FAIL_AT = None  # clear failure backoff on success
                        logger.info(f"Bitunix symbol list refreshed: {len(syms)} USDT perps")
                        return _BITUNIX_SYMBOLS
                    else:
                        _data_len = len(data.get("data") or []) if isinstance(data.get("data"), list) else 0
                        _fail_reason = (
                            f"empty/no-USDT response (data items={_data_len}, "
                            f"code={data.get('code')}, msg={data.get('msg')!r})"
                        )
            else:
                _body_snip = (resp.text or "")[:200] if hasattr(resp, "text") else ""
                _fail_reason = f"HTTP {resp.status_code} ({_body_snip!r})"
        except Exception as e:
            _fail_reason = f"{type(e).__name__}: {e or '(no message)'}"

        # Mark failure for backoff and emit a throttled, descriptive warning.
        _BITUNIX_LAST_FAIL_AT = now
        if _BITUNIX_WARN_LAST is None or (now - _BITUNIX_WARN_LAST).total_seconds() >= _BITUNIX_WARN_INTERVAL:
            _BITUNIX_WARN_LAST = now
            _have = len(_BITUNIX_SYMBOLS)
            logger.warning(
                f"Could not refresh Bitunix symbol list: {_fail_reason} "
                f"— continuing with {_have} cached symbols "
                f"({'sticky cache' if _have else 'EMPTY — falling back to MEXC tickers'})"
            )

    # Sticky: returns last good cache if one exists, else empty (caller falls
    # back to MEXC ticker derivation in _get_eligible_symbols).
    return _BITUNIX_SYMBOLS


# ─── Symbol eligibility ─────────────────────────────────────────────────────

# Fiat currencies and stablecoins — these trade against USDT on MEXC/Binance
# but are NOT crypto altcoins and should never be included in altcoin strategies.
FIAT_STABLE_BLOCKED = {
    # Fiat currencies
    "EUR", "GBP", "JPY", "AUD", "CHF", "CAD", "SGD", "TRY", "BRL", "ZAR",
    "ARS", "MXN", "HKD", "KRW", "INR", "NOK", "SEK", "DKK", "PLN", "CZK",
    # Stablecoins (USD-pegged)
    "USDC", "USDE", "USDS", "TUSD", "BUSD", "GUSD", "USDP", "DAI", "FRAX",
    "LUSD", "CRVUSD", "PAX", "SUSD", "OUSD", "USDD", "PYUSD", "FDUSD",
    "USDJ", "HUSD", "EURS", "EURT", "AEUR",
    # Tokenised commodities / metals
    "XAUT", "PAXG", "WBTC",
}

# ── Globally banned symbols — never traded by any strategy ───────────────────
SYMBOL_BLACKLIST = {"META", "ONT"}

_RAW_TICKERS_LAST_FAIL_AT: Optional[datetime] = None
_RAW_TICKERS_WARN_LAST: Optional[datetime] = None
_RAW_TICKERS_FAIL_BACKOFF = 15  # short backoff — ticker data goes stale fast

async def _get_raw_tickers(http_client: httpx.AsyncClient) -> Optional[list]:
    """
    Fetch the full MEXC/Binance ticker list once per TTL window.
    All strategies in the same scan cycle share this cached response —
    eliminates N parallel ticker fetches where N = number of active strategies.

    STICKY CACHE: once populated, the previous ticker list is returned even
    when a refresh attempt fails (httpx PoolTimeout, MEXC outage, etc.).
    Without this, a single failed refresh makes _get_eligible_symbols return
    [] for every strategy in the cycle, halting all trades.
    """
    global _RAW_TICKERS_CACHE, _RAW_TICKERS_AT
    global _RAW_TICKERS_LAST_FAIL_AT, _RAW_TICKERS_WARN_LAST
    now = datetime.utcnow()
    if (
        _RAW_TICKERS_CACHE is not None
        and _RAW_TICKERS_AT is not None
        and (now - _RAW_TICKERS_AT).total_seconds() < _RAW_TICKERS_TTL
    ):
        return _RAW_TICKERS_CACHE

    # Recently failed → return whatever we have (sticky cache).
    if (
        _RAW_TICKERS_LAST_FAIL_AT
        and (now - _RAW_TICKERS_LAST_FAIL_AT).total_seconds() < _RAW_TICKERS_FAIL_BACKOFF
    ):
        return _RAW_TICKERS_CACHE

    _last_err: Optional[str] = None
    for url in [
        "https://api.mexc.com/api/v3/ticker/24hr",
        "https://fapi.binance.com/fapi/v1/ticker/24hr",
    ]:
        try:
            resp = await http_client.get(url, timeout=10)
            if resp.status_code == 200:
                _RAW_TICKERS_CACHE = resp.json()
                _RAW_TICKERS_AT    = now
                _RAW_TICKERS_LAST_FAIL_AT = None
                logger.debug(f"Ticker cache refreshed ({len(_RAW_TICKERS_CACHE)} symbols)")
                return _RAW_TICKERS_CACHE
            else:
                _last_err = f"{url.split('//')[1].split('/')[0]} HTTP {resp.status_code}"
        except Exception as e:
            _last_err = f"{url.split('//')[1].split('/')[0]} {type(e).__name__}: {e or '(no msg)'}"
            continue

    # All sources failed — record + throttled warning + sticky fallback.
    _RAW_TICKERS_LAST_FAIL_AT = now
    if _RAW_TICKERS_WARN_LAST is None or (now - _RAW_TICKERS_WARN_LAST).total_seconds() >= 60:
        _RAW_TICKERS_WARN_LAST = now
        _have = len(_RAW_TICKERS_CACHE) if _RAW_TICKERS_CACHE else 0
        logger.warning(
            f"Could not refresh MEXC/Binance ticker list: {_last_err or 'all sources failed'}"
            f" — continuing with {_have} cached symbols "
            f"({'sticky cache' if _have else 'NO cache — strategies will be skipped'})"
        )
    return _RAW_TICKERS_CACHE  # sticky: last good cache or None if never warmed


async def _get_eligible_symbols(
    universe: Dict,
    http_client: httpx.AsyncClient,
    raw_tickers: Optional[list] = None,
) -> List[str]:
    from app.services.social_signals import SLOW_HIGHCAP_BLOCKED

    # Fetch Bitunix-available symbols in parallel with the price tickers
    bitunix_task = asyncio.create_task(_get_bitunix_symbols(http_client))

    tickers = raw_tickers
    if tickers is None:
        tickers = await _get_raw_tickers(http_client)
    if not tickers:
        bitunix_task.cancel()
        return []

    bitunix_symbols = await bitunix_task  # wait for Bitunix list

    sym_type  = universe.get("type", "all")
    specific  = {s.upper() for s in universe.get("symbols", [])}
    is_pinned = sym_type == "specific"   # user explicitly named these coins
    excl_slow = universe.get("exclude_slow_highcap", True)
    min_vol   = float(universe.get("min_volume_usd", 500_000))
    min_chg   = universe.get("min_24h_change")
    max_chg   = universe.get("max_24h_change")

    # ── Bitunix availability guard ────────────────────────────────────────────
    # If the Bitunix API is unreachable, derive the symbol set from the MEXC
    # tickers that were already fetched successfully. Both exchanges list the
    # same major USDT perpetuals, so this is a safe proxy. We filter to coins
    # with ≥$500k volume to avoid including illiquid long-tail tokens.
    if not bitunix_symbols:
        derived = set()
        for t in tickers:
            sym = t.get("symbol", "")
            vol = float(t.get("quoteVolume", 0) or 0)
            if sym.endswith("USDT") and vol >= 500_000:
                derived.add(sym)
        if derived:
            logger.info(
                f"[eligible-symbols] Bitunix API unavailable — using {len(derived)} "
                f"liquid MEXC symbols as proxy (strategies will run normally)"
            )
            bitunix_symbols = derived
        else:
            logger.warning(
                "[eligible-symbols] Bitunix symbol list empty and no ticker fallback — "
                "skipping cycle"
            )
            return []

    # For pinned coins, enforce the Bitunix list strictly.
    if is_pinned and specific:
        ticker_map = {t.get("symbol", ""): t for t in tickers if t.get("symbol", "").endswith("USDT")}
        pinned_found = []
        for sym in specific:
            if not sym.endswith("USDT"):
                sym += "USDT"
            if sym not in ticker_map:
                continue  # not in MEXC price feed
            if sym not in bitunix_symbols:
                logger.debug(f"[eligible-symbols] Skipping pinned {sym} — not on Bitunix futures")
                continue
            pinned_found.append(sym)
        return pinned_found

    # ── top_gainers: return top-N Bitunix coins ranked by 24h % gain ─────────
    # This is a dynamic ranking — always the highest movers of the day,
    # regardless of whether any cross a fixed percentage threshold.
    if sym_type == "top_gainers":
        top_n = int(universe.get("top_n", 30))
        ranked = []
        for t in tickers:
            sym = t.get("symbol", "")
            if not sym.endswith("USDT"):
                continue
            if sym not in bitunix_symbols:
                continue
            base = sym.replace("USDT", "")
            if base in FIAT_STABLE_BLOCKED:
                continue
            if base in SYMBOL_BLACKLIST:
                continue
            if excl_slow and base in SLOW_HIGHCAP_BLOCKED:
                continue
            vol = float(t.get("quoteVolume", 0))
            if vol < min_vol:
                continue
            chg = float(t.get("priceChangePercent", 0))
            ranked.append((sym, chg))
        ranked.sort(key=lambda x: x[1], reverse=True)
        return [sym for sym, _ in ranked[:top_n]]

    symbols = []
    for t in tickers:
        sym = t.get("symbol", "")
        if not sym.endswith("USDT"):
            continue
        # Only coins listed on Bitunix perpetuals — enforced unconditionally
        if sym not in bitunix_symbols:
            continue
        if sym_type == "specific" and sym not in specific:
            continue
        base = sym.replace("USDT", "")
        if base in FIAT_STABLE_BLOCKED:
            continue
        if base in SYMBOL_BLACKLIST:
            continue
        if excl_slow and base in SLOW_HIGHCAP_BLOCKED:
            continue
        vol = float(t.get("quoteVolume", 0))
        if vol < min_vol:
            continue
        chg = float(t.get("priceChangePercent", 0))
        if min_chg is not None and chg < float(min_chg):
            continue
        if max_chg is not None and chg > float(max_chg):
            continue
        symbols.append(sym)
    return symbols


async def _fetch_price_and_ta(
    symbol: str,
    http_client: httpx.AsyncClient,
    asset_class: str = "crypto",
) -> Optional[Dict]:
    """
    Fetch price + TA indicators for a symbol. Results are cached per symbol for
    _PRICE_TA_TTL seconds so multiple strategies checking the same coin in one
    scan cycle share a single API call instead of making duplicate requests.

    For stocks / forex / indices (asset_class != 'crypto'), price + RSI come
    from yfinance via tradfi_prices. Other indicators are computed on demand
    inside the condition evaluator (which reads asset_class from the cache).
    """
    global _PRICE_TA_CACHE
    now = datetime.utcnow()
    cache_key = f"{asset_class}:{symbol}" if asset_class != "crypto" else symbol
    cached = _PRICE_TA_CACHE.get(cache_key)
    if cached:
        data, fetched_at = cached
        if (now - fetched_at).total_seconds() < _PRICE_TA_TTL:
            return data

    if asset_class != "crypto":
        try:
            from app.services.tradfi_prices import (
                get_klines as _tradfi_klines,
                get_price as _tradfi_live_price,
            )
            kl = await _tradfi_klines(symbol, asset_class, "15m", 100)
            if not kl:
                return None
            closes = [float(row[4]) for row in kl if row and len(row) >= 5]
            if len(closes) < 2:
                return None

            # ── Live spot price (FMP WebSocket → yfinance fast_info) ──────────
            # Using closes[-1] (last 15m kline close) as the entry price causes
            # stale entries — that candle could have closed up to 14 minutes ago
            # and price may have moved significantly since then, especially for
            # fast-moving assets like XAUUSD.  Fetch the live spot price so the
            # entry stamped on the StrategyExecution reflects where price actually
            # is right now, not where it was at the last 15m close.
            live_px = await _tradfi_live_price(symbol, asset_class)
            price = live_px if live_px else closes[-1]

            # Inline RSI(14) — same Wilder's smoothing the social_signals impl uses
            rsi = 50.0
            if len(closes) >= 15:
                gains, losses = [], []
                for i in range(1, len(closes)):
                    d = closes[i] - closes[i - 1]
                    gains.append(max(d, 0.0))
                    losses.append(max(-d, 0.0))
                ag = sum(gains[-14:]) / 14
                al = sum(losses[-14:]) / 14
                rsi = 100.0 if al == 0 else 100.0 - (100.0 / (1.0 + (ag / al)))
            result = {
                "price": price,
                "change_24h": ((price - closes[-min(96, len(closes))]) / closes[-min(96, len(closes))] * 100) if closes[-min(96, len(closes))] else 0.0,
                "volume_24h": 0.0,
                "high_24h": max(closes[-min(96, len(closes)):]),
                "low_24h":  min(closes[-min(96, len(closes)):]),
                "rsi": rsi,
                "volume_ratio": 1.0,
                "btc_correlation": 0.0,
                "enhanced_ta": {},
                "_asset_class": asset_class,
            }
            _PRICE_TA_CACHE[cache_key] = (result, now)
            return result
        except Exception as e:
            logger.warning(f"[tradfi] price/TA fetch failed for {symbol} ({asset_class}): {e}")
            return None

    try:
        from app.services.social_signals import SocialSignalService
        svc = SocialSignalService()
        svc.http_client = http_client
        result = await svc.fetch_price_data(symbol)
        if result:
            _PRICE_TA_CACHE[cache_key] = (result, now)
        return result
    except Exception as e:
        logger.debug(f"Price/TA fetch failed for {symbol}: {e}")
        return None


# ─── Guard helpers ───────────────────────────────────────────────────────────

_SESSION_HOURS = {
    "asian":    (0, 8),   "tokyo":    (0, 8),   "asia":    (0, 8),
    "sydney":   (22, 7),  # Sydney wraps midnight UTC — handled specially below
    "london":   (7, 16),  "europe":   (7, 16),
    "new_york": (13, 22), "ny":       (13, 22),
    "overlap":  (13, 16),
}

# ── Session alert dedup + windows ─────────────────────────────────────────────
# Keyed as (user_id, session_id, "YYYY-MM-DD") so each alert fires at most once
# per user per session per calendar day (UTC).  Stale entries pruned each loop.
_SESSION_ALERT_SENT: set = set()

# (session_id_as_stored_in_config, display_label, open_hour_utc, open_min_utc)
_SESSION_ALERT_WINDOWS = [
    ("london_kz",  "London Killzone",  7,  0),
    ("ny_kz",      "NY Killzone",     12,  0),
    ("asian_kz",   "Asian Killzone",  20,  0),
    ("london",     "London Session",   8,  0),
    ("europe",     "Europe Session",   8,  0),
    ("new_york",   "NY Session",      13, 30),
    ("ny",         "NY Session",      13, 30),
    ("asian",      "Asian Session",    0,  0),
    ("tokyo",      "Tokyo Session",    0,  0),
]


def _check_trading_days(filters: Dict) -> bool:
    """Return True if today is an allowed trading day (Mon=0 … Sun=6)."""
    allowed = filters.get("trading_days")
    if not allowed:
        return True
    # Accept both long names ("monday") and short IDs ("mon") — the mobile
    # wizard sends short, the web wizard sends long, and either should work.
    day_map = {
        "monday": 0, "tuesday": 1, "wednesday": 2,
        "thursday": 3, "friday": 4, "saturday": 5, "sunday": 6,
        "mon": 0, "tue": 1, "wed": 2, "thu": 3, "fri": 4, "sat": 5, "sun": 6,
    }
    today = datetime.utcnow().weekday()
    allowed_nums = {day_map[d.lower()] for d in allowed if d.lower() in day_map}
    return today in allowed_nums


def _check_time_filter(filters: Dict) -> bool:
    hour = datetime.utcnow().hour

    # 1. Explicit hour-range filter
    tf = filters.get("time_filter")
    if tf:
        if not (tf.get("start_hour", 0) <= hour < tf.get("end_hour", 24)):
            return False

    # 2. Named session filter  {"type":"session","sessions":["new_york"]}
    sf = filters.get("session")
    if sf:
        wanted = [s.lower() for s in sf.get("sessions", [])]
        if wanted:
            active = []
            for name, (s, e) in _SESSION_HOURS.items():
                # Sessions that wrap midnight UTC (start > end) are active
                # when hour >= start OR hour < end.
                if s > e:
                    if hour >= s or hour < e:
                        active.append(name)
                elif s <= hour < e:
                    active.append(name)
            if not any(s in active for s in wanted):
                return False

    return True


def _check_btc_regime(filters: Dict) -> bool:
    required = filters.get("btc_regime")
    if not required:
        return True
    try:
        from app.services.ai_market_intelligence import get_cached_regime
        regime = get_cached_regime()
        return regime and required.lower() in regime.lower()
    except Exception:
        return True


# ─── HTF (1H) trend filter ────────────────────────────────────────────────────
_HTF_CACHE: Dict[str, tuple] = {}   # symbol -> (is_bullish: bool, fetched_at)
_HTF_CACHE_TTL = 300                 # 5 minutes — 1H candles change slowly


async def _check_htf_trend(
    symbol: str,
    direction: str,
    http_client: httpx.AsyncClient,
    asset_class: str = "crypto",
) -> bool:
    """
    Return True if the 1H trend aligns with `direction`.
    Uses a simple EMA-20 check on the last 30 hourly closes.
    BOTH direction always passes.  Cache results for 5 min.
    """
    if direction == "BOTH":
        return True

    now = datetime.utcnow()
    cache_key = f"{asset_class}:{symbol}" if asset_class != "crypto" else symbol
    cached = _HTF_CACHE.get(cache_key)
    if cached:
        is_bullish, fetched_at = cached
        if (now - fetched_at).total_seconds() < _HTF_CACHE_TTL:
            return is_bullish if direction == "LONG" else not is_bullish

    # Fetch last 30 × 1H candles — route tradfi through yfinance.
    closes: list = []
    if asset_class != "crypto":
        try:
            from app.services.tradfi_prices import get_klines as _tradfi_klines
            kl = await _tradfi_klines(symbol, asset_class, "1h", 60)
            if kl and len(kl) >= 10:
                closes = [float(row[4]) for row in kl if row and len(row) >= 5]
        except Exception as e:
            logger.debug(f"tradfi HTF fetch failed for {symbol} ({asset_class}): {e}")
    else:
        sources = [
            ("https://api.mexc.com/api/v3/klines",     {"symbol": symbol, "interval": "1h", "limit": 30}),
            ("https://fapi.binance.com/fapi/v1/klines", {"symbol": symbol, "interval": "1h", "limit": 30}),
            ("https://api.binance.com/api/v3/klines",   {"symbol": symbol, "interval": "1h", "limit": 30}),
        ]
        for url, params in sources:
            try:
                resp = await http_client.get(url, params=params, timeout=6)
                if resp.status_code == 200:
                    data = resp.json()
                    if data and len(data) >= 10:
                        closes = [float(k[4]) for k in data]
                        break
            except Exception:
                continue

    if len(closes) < 10:
        return True   # can't determine — allow through

    # Simple EMA-20 (or length of data)
    period = min(20, len(closes))
    k = 2 / (period + 1)
    ema = closes[0]
    for c in closes[1:]:
        ema = c * k + ema * (1 - k)

    is_bullish = closes[-1] > ema
    _HTF_CACHE[cache_key] = (is_bullish, now)
    return is_bullish if direction == "LONG" else not is_bullish


def _daily_execution_count(strategy_id: int, db) -> int:
    from app.strategy_models import StrategyExecution
    today = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
    return db.query(StrategyExecution).filter(
        StrategyExecution.strategy_id == strategy_id,
        StrategyExecution.fired_at >= today,
    ).count()


def _open_execution_count(strategy_id: int, db) -> int:
    from app.strategy_models import StrategyExecution
    return db.query(StrategyExecution).filter(
        StrategyExecution.strategy_id == strategy_id,
        StrategyExecution.outcome == "OPEN",
    ).count()


def _fired_today_for_symbol(strategy_id: int, symbol: str, db) -> bool:
    from app.strategy_models import StrategyExecution
    today = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
    return db.query(StrategyExecution).filter(
        StrategyExecution.strategy_id == strategy_id,
        StrategyExecution.symbol == symbol,
        StrategyExecution.fired_at >= today,
    ).first() is not None


def _last_any_fired_time(strategy_id: int, db) -> Optional[datetime]:
    """Global: when did this strategy last fire on ANY symbol."""
    from app.strategy_models import StrategyExecution
    last = (
        db.query(StrategyExecution)
        .filter(StrategyExecution.strategy_id == strategy_id)
        .order_by(StrategyExecution.fired_at.desc())
        .first()
    )
    return last.fired_at if last else None


def _last_fired_time(strategy_id: int, symbol: str, db) -> Optional[datetime]:
    from app.strategy_models import StrategyExecution
    last = (
        db.query(StrategyExecution)
        .filter(
            StrategyExecution.strategy_id == strategy_id,
            StrategyExecution.symbol == symbol,
        )
        .order_by(StrategyExecution.fired_at.desc())
        .first()
    )
    return last.fired_at if last else None


# ─── Performance update ──────────────────────────────────────────────────────

def _update_performance(strategy_id: int, db):
    """Recompute StrategyPerformance from all closed executions."""
    from app.strategy_models import StrategyExecution, StrategyPerformance
    execs = db.query(StrategyExecution).filter(
        StrategyExecution.strategy_id == strategy_id
    ).all()

    closed    = [e for e in execs if e.outcome in ("WIN","LOSS","BREAKEVEN") and e.pnl_pct is not None]
    open_cnt  = sum(1 for e in execs if e.outcome == "OPEN")
    wins      = [e.pnl_pct for e in closed if e.outcome == "WIN"]
    losses    = [e.pnl_pct for e in closed if e.outcome == "LOSS"]
    total_pnl = sum(e.pnl_pct for e in closed)

    perf = db.query(StrategyPerformance).filter(StrategyPerformance.strategy_id == strategy_id).first()
    if not perf:
        perf = StrategyPerformance(strategy_id=strategy_id)
        db.add(perf)

    perf.total_trades  = len(closed)
    perf.open_trades   = open_cnt
    perf.wins          = len(wins)
    perf.losses        = len(losses)
    perf.breakevens    = sum(1 for e in closed if e.outcome == "BREAKEVEN")
    # Win rate excludes BREAKEVEN from the denominator — breakevens are
    # zero-PnL neutral outcomes and shouldn't pull the win rate down.
    decisive = len(wins) + len(losses)
    perf.win_rate      = round(len(wins) / decisive * 100, 1) if decisive else 0.0
    perf.total_pnl_pct = round(total_pnl, 2)
    perf.avg_win_pct   = round(sum(wins)   / len(wins),   2) if wins   else 0.0
    perf.avg_loss_pct  = round(sum(losses) / len(losses), 2) if losses else 0.0
    perf.best_trade    = round(max(wins),   2) if wins   else 0.0
    perf.worst_trade   = round(min(losses), 2) if losses else 0.0
    db.commit()


# ─── Paper position monitor ──────────────────────────────────────────────────

async def _fetch_candles_since_entry(
    symbol: str,
    fired_at: datetime,
    http_client: httpx.AsyncClient,
    asset_class: str = "crypto",
) -> list:
    """
    Fetch all 1m candles from `fired_at` to now so no TP/SL hit is ever missed.
    Returns a list of (open_ts, open, high, low, close) tuples sorted oldest-first.
    For crypto: falls back through MEXC → Binance spot → Binance futures.
    For stocks/forex/indices: routes through yfinance (1m data capped at 5d
    upstream — trades older than that return an empty list and the caller's
    graceful fallback handles it).
    """
    asset_class = (asset_class or "crypto").lower().strip()
    if asset_class and asset_class != "crypto":
        try:
            from app.services.tradfi_prices import get_klines as _tradfi_klines
            fired_ms = int(fired_at.timestamp() * 1000)
            # Dynamically size the fetch window based on minutes elapsed since
            # fired_at, with a 60-candle prologue + buffer. yfinance 1m data
            # is upstream-capped at 5d (7200 candles) so we hard-cap at 7200
            # — older trades will return an empty list and the caller's
            # graceful fallback handles it.
            elapsed_min = max(1, int((datetime.utcnow() - fired_at).total_seconds() / 60))
            tf_limit = min(7200, elapsed_min + 60)
            raw = await _tradfi_klines(symbol, asset_class, "1m", tf_limit)
            out = []
            for k in raw:
                try:
                    ts = int(k[0])
                    if ts < fired_ms:
                        continue
                    out.append((ts, float(k[1]), float(k[2]),
                                float(k[3]), float(k[4])))
                except (TypeError, ValueError, IndexError):
                    continue
            return out
        except Exception as e:
            logger.debug(f"tradfi candle fetch failed {symbol} ({asset_class}): {e}")
            return []

    now        = datetime.utcnow()
    minutes    = max(int((now - fired_at).total_seconds() / 60) + 2, 3)
    all_candles: list = []

    # Page through in 900-candle chunks (MEXC safe limit)
    chunk = 900
    start = fired_at
    while start < now:
        needed = min(chunk, int((now - start).total_seconds() / 60) + 2)
        # All three sources support startTime (ms epoch)
        start_ms = int(start.timestamp() * 1000)

        sources = [
            ("https://api.mexc.com/api/v3/klines",
             {"symbol": symbol, "interval": "1m", "startTime": start_ms, "limit": needed}),
            ("https://api.binance.com/api/v3/klines",
             {"symbol": symbol, "interval": "1m", "startTime": start_ms, "limit": needed}),
            ("https://fapi.binance.com/fapi/v1/klines",
             {"symbol": symbol, "interval": "1m", "startTime": start_ms, "limit": needed}),
        ]
        fetched = False
        for url, params in sources:
            try:
                resp = await http_client.get(url, params=params, timeout=8)
                if resp.status_code != 200:
                    continue
                klines = resp.json()
                if not klines:
                    continue
                for k in klines:
                    ts    = int(k[0])
                    open_ = float(k[1])
                    high  = float(k[2])
                    low   = float(k[3])
                    close = float(k[4])
                    all_candles.append((ts, open_, high, low, close))
                fetched = True
                break
            except Exception as e:
                logger.debug(f"Candle fetch failed {url} {symbol}: {e}")
                continue

        if not fetched or needed < chunk:
            break
        # Advance start to the last fetched candle's open + 1m
        if all_candles:
            last_ts = all_candles[-1][0]
            start = datetime.utcfromtimestamp(last_ts / 1000) + timedelta(minutes=1)
        else:
            break

    # Deduplicate and sort by timestamp
    seen: set = set()
    unique = []
    for c in sorted(all_candles, key=lambda x: x[0]):
        if c[0] not in seen:
            seen.add(c[0])
            unique.append(c)
    return unique


def _paper_cost_basis_pct(asset_class: str, symbol: str = "") -> float:
    """Realistic spread/execution cost as a % of position (deducted silently from raw_pnl).

    Forex (FP Markets cTrader Raw): tight spreads on majors, wider on metals.
    Crypto: ~0.05% entry + 0.05% exit taker fee = 0.1% round-trip.
    Stock/Index CFD: ~0.05% spread equivalent.
    """
    sym = (symbol or "").upper()
    if asset_class == "forex":
        if sym in ("XAUUSD", "XAGUSD", "XPTUSD"):
            return 0.008   # Gold/Silver: ~$0.30 spread on Raw ÷ ~$3500 price
        return 0.002       # FX majors: ~0.2 pip spread on Raw (EURUSD ~0.00020/1.08)
    if asset_class == "crypto":
        return 0.10        # 0.05% entry + 0.05% exit taker
    if asset_class in ("stock", "index"):
        return 0.05        # CFD spread equivalent
    return 0.05            # conservative default


def _close_paper_execution(ex, outcome: str, exit_price: float, db):
    """Mark a paper execution as closed and update performance.

    Uses an atomic UPDATE WHERE outcome='OPEN' so that when multiple uvicorn
    workers race to close the same position, exactly one wins and sends the
    Telegram notification — preventing duplicate messages.
    """
    if ex.direction == "LONG":
        raw_pnl = (exit_price - ex.entry_price) / ex.entry_price * 100
    else:
        raw_pnl = (ex.entry_price - exit_price) / ex.entry_price * 100

    # Silently deduct realistic spread/execution cost so paper P&L reflects
    # what the user would actually see on their broker — no line item shown.
    if outcome != "CANCELLED":
        raw_pnl -= _paper_cost_basis_pct(
            getattr(ex, "asset_class", "crypto"),
            getattr(ex, "symbol", ""),
        )

    pnl_pct   = round(raw_pnl * ex.leverage, 2)
    closed_at = datetime.utcnow()

    # Atomic close — only the first worker to execute this UPDATE wins.
    from sqlalchemy import text as _text
    result = db.execute(
        _text(
            "UPDATE strategy_executions "
            "SET outcome=:outcome, exit_price=:exit_price, pnl_pct=:pnl, closed_at=:closed_at "
            "WHERE id=:id AND outcome='OPEN'"
        ),
        {"outcome": outcome, "exit_price": exit_price, "pnl": pnl_pct,
         "closed_at": closed_at, "id": ex.id},
    )
    db.commit()

    if result.rowcount == 0:
        # Another worker already closed this execution — skip notification.
        return

    # Sync in-memory object so callers that still hold a reference see the
    # updated state (prevents a second evaluation loop from re-closing it).
    ex.outcome    = outcome
    ex.exit_price = exit_price
    ex.pnl_pct    = pnl_pct
    ex.closed_at  = closed_at

    # Append close note — preserve any original open-time context (e.g. Live→Paper
    # fallback reason) so the history is auditable after close.
    pnl_sign   = "+" if pnl_pct >= 0 else ""
    if outcome == "WIN":
        close_note = f"TP hit · {pnl_sign}{pnl_pct}% · exit {exit_price:.6g}"
    elif outcome == "LOSS":
        close_note = f"SL hit · {pnl_sign}{pnl_pct}% · exit {exit_price:.6g}"
    elif outcome == "CANCELLED":
        close_note = "Expired · no TP/SL hit within hold period"
    else:
        close_note = f"Closed · {pnl_sign}{pnl_pct}% · exit {exit_price:.6g}"
    # If there's an existing note that isn't just the live-monitor "open · unrealised" noise,
    # keep it prepended so we can always see why a trade went paper even after it closes.
    existing = (ex.notes or "").strip()
    if existing and not existing.startswith("open ·"):
        close_note = f"{existing} | {close_note}"
    try:
        db.execute(
            _text("UPDATE strategy_executions SET notes=:n WHERE id=:id"),
            {"n": close_note, "id": ex.id},
        )
        db.commit()
        ex.notes = close_note
    except Exception:
        try:
            db.rollback()
        except Exception:
            pass

    _update_performance(ex.strategy_id, db)

    # Telegram DM notification for paper closes
    try:
        from app.strategy_models import UserStrategy, StrategyPortalSettings
        from app.models import User
        user = db.query(User).filter(User.id == ex.user_id).first()
        if not user:
            return
        settings = db.query(StrategyPortalSettings).filter(StrategyPortalSettings.user_id == user.id).first()
        if settings and not settings.dm_paper_alerts:
            return
        strat = db.query(UserStrategy).filter(UserStrategy.id == ex.strategy_id).first()
        strat_name = strat.name if strat else "Your Strategy"
        tg_id = _telegram_int_id(user)
        if tg_id:
            asyncio.create_task(_send_paper_close_dm(
                tg_id,
                _fmt_close_card(
                    strategy_name = strat_name,
                    symbol        = ex.symbol,
                    direction     = ex.direction,
                    entry         = ex.entry_price,
                    exit_price    = exit_price,
                    outcome       = outcome,
                    pnl_pct       = pnl_pct,
                    leverage      = ex.leverage,
                    fired_at      = ex.fired_at,
                    closed_at     = closed_at,
                    conditions    = ex.conditions_met,
                    is_paper      = True,
                ),
            ))
        # Mobile push — trade close (paper)
        from app.services.expo_push import notify_trade_close_bg
        dur_mins = int((closed_at - ex.fired_at).total_seconds() / 60) if ex.fired_at else 0
        notify_trade_close_bg(
            user_id=ex.user_id,
            strategy_name=strat_name,
            symbol=ex.symbol,
            direction=ex.direction,
            outcome=outcome,
            pnl_pct=pnl_pct,
            leverage=ex.leverage or 10,
            entry_price=ex.entry_price,
            exit_price=exit_price,
            strategy_id=ex.strategy_id,
            execution_id=ex.id,
            is_paper=True,
            duration_mins=dur_mins,
            kind="paper",
            position_usd=float(ex.position_size) if ex.position_size else None,
        )
    except Exception:
        pass


def _fmt_open_card(
    strategy_name: str, symbol: str, direction: str,
    entry: float, tp_price: float, tp_pct: float,
    sl_price: float, sl_pct: float, leverage: int,
    conditions: list, is_paper: bool,
    tp2_price: float = None, tp2_pct: float = None,
    order_id: str = None,
    asset_class: str = "crypto",
) -> str:
    dir_icon = "🟢" if direction == "LONG" else "🔴"
    header   = "🧪 <b>YOUR STRATEGY FIRED (PAPER)</b>" if is_paper else "🚀 <b>YOUR STRATEGY IS LIVE</b>"
    bar      = "━━━━━━━━━━━━━━━━━━━━"

    # For forex: show pips (what traders actually think in) instead of %.
    # pips = price_distance / pip_size  e.g. XAUUSD: $36 / $1.00 = 36 pips.
    _is_forex = (asset_class == "forex")
    if _is_forex and entry and entry > 0:
        from app.services.forex_engine import pip_size as _pip_size
        _ps = _pip_size(symbol)
        _tp_pips = round(abs(tp_price - entry) / _ps) if _ps else None
        _sl_pips = round(abs(sl_price - entry) / _ps) if _ps else None
        tp_label  = f"{_tp_pips} pips" if _tp_pips is not None else f"{tp_pct:.1f}%"
        sl_label  = f"{_sl_pips} pips" if _sl_pips is not None else f"{sl_pct:.1f}%"
    else:
        sign_tp  = "+" if direction == "LONG" else "-"
        sign_sl  = "-" if direction == "LONG" else "+"
        tp_label = f"{sign_tp}{tp_pct:.1f}%"
        sl_label = f"{sign_sl}{sl_pct:.1f}%"

    tp2_line = ""
    if tp2_price and tp2_pct:
        if _is_forex and entry and entry > 0:
            _tp2_pips = round(abs(tp2_price - entry) / _ps) if _ps else None
            tp2_val = f"{_tp2_pips} pips" if _tp2_pips is not None else f"+{tp2_pct:.1f}%"
        else:
            sign = "+" if direction == "LONG" else "-"
            tp2_val = f"{sign}{tp2_pct:.1f}%"
        tp2_line = f"\nTP₂      <code>{tp2_price:.6g}</code>  ({tp2_val})"

    cond_lines = ""
    if conditions:
        passed = [c for c in conditions if c.startswith("✅")]
        if passed:
            cond_lines = "\n\n<b>Why it triggered:</b>\n" + "\n".join(f"  {c}" for c in passed[:5])

    order_line = f"\n<i>Order ID: #{order_id}</i>" if order_id else ""
    footer     = "<i>📄 Paper trade · no real funds used</i>" if is_paper else "<i>✅ Live strategy trade executed</i>"

    return (
        f"{header}\n{bar}\n"
        f"📋 <b>{strategy_name}</b>\n"
        f"{dir_icon} <b>{symbol}</b>  ·  {direction}  ·  {leverage}×\n"
        f"{bar}\n"
        f"Entry    <code>{entry:.6g}</code>\n"
        f"TP₁      <code>{tp_price:.6g}</code>  ({tp_label}){tp2_line}\n"
        f"SL       <code>{sl_price:.6g}</code>  ({sl_label})"
        f"{cond_lines}\n"
        f"{bar}\n"
        f"{footer}{order_line}"
    )


def _fmt_close_card(
    strategy_name: str, symbol: str, direction: str,
    entry: float, exit_price: float, outcome: str,
    pnl_pct: float, leverage: int,
    fired_at: datetime = None, closed_at: datetime = None,
    conditions: list = None, is_paper: bool = False,
) -> str:
    dir_icon  = "🟢" if direction == "LONG" else "🔴"
    pnl_sign  = "+" if pnl_pct >= 0 else ""
    bar       = "━━━━━━━━━━━━━━━━━━━━"
    coin      = symbol.replace("USDT", "")

    if outcome == "WIN":
        icon      = "✅"
        result    = "WIN"
        hit_label = "TP hit 🎯"
    elif outcome == "LOSS":
        icon      = "🛑"
        result    = "LOSS"
        hit_label = "SL hit 🛑"
    elif outcome == "BREAKEVEN":
        icon      = "⚖️"
        result    = "BREAKEVEN"
        hit_label = "Breakeven ⚖️"
    else:
        icon      = "📊"
        result    = outcome
        hit_label = f"{outcome}"

    duration_line = ""
    if fired_at and closed_at:
        secs  = int((closed_at - fired_at).total_seconds())
        days, rem = divmod(secs, 86400)
        hours, rem = divmod(rem, 3600)
        mins       = rem // 60
        if days:
            duration_line = f"\n⏱ Duration  <b>{days}d {hours}h {mins}m</b>"
        elif hours:
            duration_line = f"\n⏱ Duration  <b>{hours}h {mins}m</b>"
        else:
            duration_line = f"\n⏱ Duration  <b>{mins}m</b>"

    def _pip_size(sym: str) -> float | None:
        """Return pip size for the symbol, or None for crypto (no pip convention)."""
        s = sym.upper().replace("/", "").replace("=F", "").replace("=X", "")
        # Metals
        if s in ("XAUUSD", "GOLD", "GC", "XAUUSDT"):   return 0.10   # gold: $0.10/pip
        if s in ("XAGUSD", "SILVER", "SI", "XAGUSDT"):  return 0.001  # silver
        if s in ("XPTUSD", "PLATINUM", "PL"):            return 0.10
        # JPY pairs (2 decimal places)
        if "JPY" in s:                                   return 0.01
        # Standard 4-decimal forex pairs
        _FX = ("USD","EUR","GBP","AUD","NZD","CAD","CHF","SGD","HKD","NOK","SEK","DKK","PLN","CZK","HUF","MXN","ZAR","TRY","INR")
        if any(s.startswith(c) or s.endswith(c) for c in _FX) and len(s) == 6:
            return 0.0001
        # Indices — 1 point per pip
        _IDX = ("US30","US500","SPX","NAS","NDX","DAX","DE40","UK100","FTSE","JP225","HK50","ASX","IT40","FR40","ES35")
        if any(s.startswith(i) for i in _IDX):          return 1.0
        # Oil / commodities
        if s in ("USOIL","WTIUSD","BRENTUSD","UKOIL","CL","NG"):
            return 0.01
        return None  # crypto — no pip convention

    pip_sz  = _pip_size(symbol)
    # Signed raw price move (positive = favourable for the trade direction)
    raw_move = (exit_price - entry) if direction == "LONG" else (entry - exit_price)

    if pip_sz is not None:
        pips = raw_move / pip_sz
        sign = "+" if pips >= 0 else ""
        # Whole pips for large values, 1dp for fractional
        pip_str = f"{sign}{pips:.0f}" if abs(pips) >= 1 else f"{sign}{pips:.1f}"
        pnl_display = f"<b>{pip_str} pips</b>"
        move_line   = ""
    else:
        # Crypto: keep % with adaptive precision
        a = abs(pnl_pct)
        if a < 0.1:   pnl_display = f"<b>{pnl_pct:+.3f}%</b>"
        elif a < 10:  pnl_display = f"<b>{pnl_pct:+.2f}%</b>"
        else:         pnl_display = f"<b>{pnl_pct:+.1f}%</b>"
        move_line = ""

    cond_lines = ""
    if conditions:
        passed = [c for c in conditions if c.startswith("✅")]
        if passed:
            cond_lines = "\n<b>Triggered by:</b>\n" + "\n".join(f"  {c}" for c in passed[:3]) + "\n"

    paper_tag = "📄 Paper trade" if is_paper else "✅ Live trade"

    return (
        f"{icon} <b>STRATEGY {result}: {strategy_name}</b>\n{bar}\n"
        f"{dir_icon} <b>${coin}</b>  ·  {direction}  ·  {leverage}×\n"
        f"{bar}\n"
        f"Entry    <code>{entry:.6g}</code>\n"
        f"Exit     <code>{exit_price:.6g}</code>  ({hit_label})\n"
        f"P&L      {pnl_display}"
        f"{move_line}"
        f"{duration_line}\n"
        f"{bar}\n"
        f"{cond_lines}"
        f"<i>{paper_tag} result</i>"
    )


async def _tg_send(telegram_id: int, text: str):
    """Send a Telegram message via direct Bot API HTTP call — works from any process."""
    try:
        from app.config import settings
        token = settings.TELEGRAM_BOT_TOKEN
        async with httpx.AsyncClient(timeout=10) as client:
            await client.post(
                f"https://api.telegram.org/bot{token}/sendMessage",
                json={"chat_id": telegram_id, "text": text, "parse_mode": "HTML"},
            )
    except Exception as e:
        logger.warning(f"Telegram DM failed for {telegram_id}: {e}")


async def _send_paper_close_dm(telegram_id: int, text: str):
    await _tg_send(telegram_id, text)


def _evaluate_paper_position_against_candles(ex, candles: list, db) -> bool:
    """
    Apply a pre-fetched candle list to one open paper position.
    Returns True if the position was closed (TP/SL hit or expired).

    Candle tuples: (open_ts_ms, open, high, low, close)

    Same-candle TP+SL resolution — when both TP and SL are hit inside the same
    1-minute candle, we use the candle's open→close direction to infer which
    level was reached first (standard OHLC backtesting heuristic):
      LONG:  bullish candle (close >= open) → price rose first → TP hit first (WIN)
             bearish candle (close <  open) → price fell first → SL hit first (LOSS)
      SHORT: bearish candle (close <= open) → price fell first → TP hit first (WIN)
             bullish candle (close >  open) → price rose first → SL hit first (LOSS)
    """
    if not ex.entry_price or not ex.tp_price or not ex.sl_price:
        return False

    fired_at = ex.fired_at or datetime.utcnow()
    elapsed_hours = (datetime.utcnow() - fired_at).total_seconds() / 3600
    elapsed_mins  = elapsed_hours * 60

    # ── Candle evaluation FIRST ───────────────────────────────────────────────
    # Always scan for a TP/SL hit before considering expiry.  This prevents
    # incorrectly CANCELLING a trade whose TP was already hit but whose
    # fired_at is older than PAPER_MAX_HOLD_HOURS.
    be_pct = None
    partial_close_pct = None
    be_timer_minutes = None
    try:
        from app.strategy_models import UserStrategy
        strat = db.query(UserStrategy).filter(UserStrategy.id == ex.strategy_id).first()
        if strat and strat.config:
            ex_cfg  = (strat.config or {}).get("exit", {})
            risk_cfg = (strat.config or {}).get("risk", {})
            be_pct = ex_cfg.get("breakeven_pct") or ex_cfg.get("breakeven_at_pct")
            if be_pct is not None:
                be_pct = float(be_pct)
            _pcp = ex_cfg.get("partial_close_pct")
            if _pcp:
                partial_close_pct = float(_pcp)
            _bet = risk_cfg.get("be_timer_minutes")
            if _bet:
                be_timer_minutes = float(_bet)
    except Exception:
        pass

    # ── BE timer: force-close if price hasn't reached breakeven within N mins ─
    # Check BEFORE candle evaluation — if the timer has expired and SL hasn't
    # moved to entry yet, close at the last known price (market close).
    if be_timer_minutes is not None and elapsed_mins >= be_timer_minutes:
        # Only fire if the SL hasn't already been moved to entry (be not activated)
        be_already_at_entry = (
            ex.sl_price is not None and
            ex.entry_price is not None and
            abs(float(ex.sl_price) - float(ex.entry_price)) < float(ex.entry_price) * 0.0001
        )
        if not be_already_at_entry:
            logger.info(
                f"[PaperMonitor] BE TIMER expired: exec #{ex.id} {ex.symbol} "
                f"{elapsed_mins:.0f}m elapsed >= {be_timer_minutes:.0f}m limit — "
                f"closing at entry (CANCELLED). SL never reached entry."
            )
            _close_paper_execution(ex, "CANCELLED", ex.entry_price, db)
            return True

    # Also read partial_close_pct from the execution notes (set at fire time)
    # as a fallback when the strategy config lookup failed.
    if partial_close_pct is None and ex.notes:
        import re as _re
        _m = _re.search(r"partial_close_pct=(\d+(?:\.\d+)?)", ex.notes or "")
        if _m:
            partial_close_pct = float(_m.group(1))

    be_activated = False
    # Track whether we've already performed a partial close for this execution
    # (stored in notes to survive across monitor cycles).
    partial_close_done = bool(ex.notes and "partial_close_done" in ex.notes)

    if candles:
        entry_ms = int(fired_at.timestamp() * 1000)
        relevant = [c for c in candles if c[0] >= entry_ms - 60_000]
        if relevant:
            for _ts, open_, high, low, close in relevant:
                if ex.direction == "LONG":
                    tp_hit = high >= ex.tp_price
                    sl_hit = low  <= ex.sl_price
                    if tp_hit and sl_hit:
                        if close >= open_:
                            _close_paper_execution(ex, "WIN", ex.tp_price, db)
                        else:
                            outcome = "BREAKEVEN" if be_activated and ex.sl_price == ex.entry_price else "LOSS"
                            _close_paper_execution(ex, outcome, ex.sl_price, db)
                        return True
                else:
                    tp_hit = low  <= ex.tp_price
                    sl_hit = high >= ex.sl_price
                    if tp_hit and sl_hit:
                        if close <= open_:
                            _close_paper_execution(ex, "WIN", ex.tp_price, db)
                        else:
                            outcome = "BREAKEVEN" if be_activated and ex.sl_price == ex.entry_price else "LOSS"
                            _close_paper_execution(ex, outcome, ex.sl_price, db)
                        return True

                # ── Partial close at TP1 (paper simulation) ───────────────
                # When partial_close_pct is set and TP1 is hit (but not yet
                # done), we simulate closing partial_close_pct% of the
                # position and moving the SL to breakeven for the remainder.
                # The execution record stays OPEN so TP2 / final SL can still
                # close it — we just record that partial already happened.
                if tp_hit and partial_close_pct and not partial_close_done and ex.tp2_price:
                    partial_close_done = True
                    # Move SL to entry price (protect the partial profit)
                    ex.sl_price = ex.entry_price
                    # Mark partial close in notes (survives across monitor cycles)
                    _old_notes = ex.notes or ""
                    # Preserve any existing metadata before the live-pnl suffix
                    _base_notes = _old_notes.split(" | open")[0].split(" | unrealised")[0].strip(" |")
                    ex.notes = (_base_notes + " | partial_close_done").strip(" |")
                    try:
                        db.commit()
                    except Exception:
                        db.rollback()
                    logger.info(
                        f"[PaperMonitor] PARTIAL CLOSE: exec #{ex.id} {ex.symbol} "
                        f"— {partial_close_pct:.0f}% closed at TP1 {ex.tp_price}, SL→entry. "
                        f"Remainder runs to TP2 {ex.tp2_price}."
                    )
                    # Remainder continues — skip the full-close below
                    continue

                if tp_hit:
                    _close_paper_execution(ex, "WIN", ex.tp_price, db)
                    return True
                if sl_hit:
                    outcome = "BREAKEVEN" if be_activated and ex.sl_price == ex.entry_price else "LOSS"
                    _close_paper_execution(ex, outcome, ex.sl_price, db)
                    return True

                if be_pct is not None and not be_activated and ex.sl_price != ex.entry_price:
                    if ex.direction == "LONG":
                        candle_roi = ((high - ex.entry_price) / ex.entry_price) * 100 * ex.leverage
                    else:
                        candle_roi = ((ex.entry_price - low) / ex.entry_price) * 100 * ex.leverage
                    if candle_roi >= be_pct:
                        ex.sl_price = ex.entry_price
                        be_activated = True
                        logger.info(f"[PaperMonitor] AUTO-BREAKEVEN: exec #{ex.id} {ex.symbol} ROI {candle_roi:.1f}% >= {be_pct}% → SL @ entry")

            last_close = relevant[-1][4]
            if ex.direction == "LONG":
                unreal = (last_close - ex.entry_price) / ex.entry_price * 100
            else:
                unreal = (ex.entry_price - last_close) / ex.entry_price * 100
            pnl_note = f"open · unrealised {'+' if unreal >= 0 else ''}{unreal:.2f}% · last {last_close:.6g}"
            orig = ex.notes or ""
            if any(kw in orig for kw in ["Live→Paper", "fallback", "Bitunix error", "error"]):
                base = orig.split(" | open")[0].split(" | unrealised")[0].strip(" |")
                ex.notes = base + " | " + pnl_note
            else:
                ex.notes = pnl_note
            try:
                db.commit()
            except Exception:
                db.rollback()

    # ── Auto-expire ONLY if no TP/SL was found ───────────────────────────────
    if elapsed_hours > PAPER_MAX_HOLD_HOURS:
        logger.info(
            f"[PaperMonitor] Expiring execution #{ex.id} ({ex.symbol} {ex.direction}) "
            f"after {elapsed_hours:.1f}h — no TP/SL hit found in candle history."
        )
        _close_paper_execution(ex, "CANCELLED", ex.entry_price, db)
        return True

    return False


async def _check_paper_position(ex, db, http_client: httpx.AsyncClient):
    """
    Fetch candles for a single position and evaluate it.
    Used when checking positions individually; prefer the batched path in _sweep.
    """
    if not ex.entry_price or not ex.tp_price or not ex.sl_price:
        return
    fired_at = ex.fired_at or datetime.utcnow()
    _ac = (getattr(ex, "asset_class", None) or "crypto")
    candles  = await _fetch_candles_since_entry(ex.symbol, fired_at, http_client, _ac)
    _evaluate_paper_position_against_candles(ex, candles, db)


async def run_paper_position_monitor():
    """
    Background loop — monitors all open paper positions every 30s using
    1-minute Binance Futures OHLC data for maximum accuracy.
    """
    from app.database import BgSessionLocal as SessionLocal
    from app.strategy_models import StrategyExecution

    logger.info("🧪 Paper position monitor started (30s interval, full-history candle scan)")

    async def _sweep(http_client):
        """
        One full sweep of all open paper positions.

        Connection-safe three-phase design:
          Phase 1  Read   — brief session: load positions, expunge, close.
          Phase 2  Fetch  — async HTTP candle fetches with NO DB connection held.
          Phase 3  Write  — brief per-position session: evaluate + commit.

        This prevents the pool from exhausting when multiple workers each run
        the paper monitor concurrently and hold a connection across awaits.
        """
        # ── Phase 1: Read ─────────────────────────────────────────────────────
        db = SessionLocal()
        try:
            open_papers = (
                db.query(StrategyExecution)
                .filter(
                    StrategyExecution.outcome == "OPEN",
                    StrategyExecution.is_paper == True,
                )
                .all()
            )
            # Detach objects so they remain readable after the session closes.
            # All scalar columns are already loaded by .all(); lazy refs are gone.
            db.expunge_all()
        finally:
            db.close()  # ← released BEFORE any async I/O

        if not open_papers:
            return

        # ── Phase 2: Group + fetch candles (no DB connection held) ────────────
        # Bucket by (symbol, asset_class) so a ticker that exists in multiple
        # classes (e.g. someone's crypto BTC vs a "BTC" stock ticker) never
        # shares candle data between classes.
        from collections import defaultdict
        by_key: dict = defaultdict(list)
        for ex in open_papers:
            _ac = (getattr(ex, "asset_class", None) or "crypto")
            by_key[(ex.symbol, _ac)].append(ex)

        logger.info(
            f"🧪 Sweeping {len(open_papers)} open paper position(s) "
            f"across {len(by_key)} (symbol, asset_class) bucket(s)"
        )

        async def _fetch_for_bucket(symbol: str, asset_class: str, positions: list):
            earliest = min((ex.fired_at or datetime.utcnow()) for ex in positions)
            candles  = await _fetch_candles_since_entry(symbol, earliest, http_client, asset_class)
            return symbol, positions, candles

        fetch_results = await asyncio.gather(
            *[_fetch_for_bucket(sym, ac, pos) for (sym, ac), pos in by_key.items()],
            return_exceptions=True,
        )

        # ── Phase 3: Evaluate + write (brief session per position) ────────────
        _now_sweep = datetime.utcnow()
        _sweep_wd  = _now_sweep.weekday()   # Mon=0 … Sun=6
        _sweep_h   = _now_sweep.hour
        _sweep_m   = _now_sweep.minute

        for result in fetch_results:
            if isinstance(result, Exception):
                logger.warning(f"Candle fetch error in paper sweep: {result}")
                continue
            symbol, positions, candles = result
            for ex in positions:
                write_db = SessionLocal()
                try:
                    # Re-attach the detached object into the fresh session so
                    # any attribute mutations are tracked for commit.
                    managed = write_db.merge(ex)

                    # ── Forex day-trading force-close guards ──────────────────
                    # These run BEFORE TP/SL candle evaluation so the position
                    # is closed at market price rather than waiting for a
                    # TP/SL hit that may never come before the weekend gap.
                    _ex_ac = getattr(managed, "asset_class", None) or "crypto"
                    if _ex_ac == "forex":
                        _forced = False
                        _force_reason = None
                        try:
                            from app.strategy_models import UserStrategy as _US2
                            _strat2 = write_db.query(_US2).filter(
                                _US2.id == managed.strategy_id
                            ).first()
                            if _strat2 and _strat2.config:
                                _risk2 = (_strat2.config or {}).get("risk", {})

                                # Friday close: force-close Fri from 21:00 UTC
                                if _risk2.get("friday_close_protection"):
                                    if (_sweep_wd == 4 and _sweep_h >= 21) or _sweep_wd >= 5:
                                        _forced = True
                                        _force_reason = "friday_close_protection"

                                # No overnight: force-close Mon-Thu from 22:00 UTC
                                if not _forced and _risk2.get("no_overnight_positions"):
                                    if _sweep_wd < 4 and _sweep_h >= 22:
                                        _forced = True
                                        _force_reason = "no_overnight_positions"
                        except Exception as _fce:
                            logger.debug(f"[PaperMonitor] Force-close config read error ex#{ex.id}: {_fce}")

                        if _forced:
                            # Close at the last known candle close price (best
                            # available market price without a live price call)
                            _fc_price = managed.entry_price
                            if candles:
                                try:
                                    _fc_price = float(candles[-1][4])  # last close
                                except Exception:
                                    pass
                            logger.info(
                                f"[PaperMonitor] FORCE-CLOSE ({_force_reason}): "
                                f"exec #{managed.id} {managed.symbol} {managed.direction} "
                                f"@ {_fc_price} — day-trading rule triggered"
                            )
                            _close_paper_execution(managed, "CANCELLED", _fc_price, write_db)
                            continue  # skip normal TP/SL evaluation

                    _evaluate_paper_position_against_candles(managed, candles, write_db)
                except Exception as e:
                    logger.warning(f"Position {ex.id} eval error: {e}")
                    try:
                        write_db.rollback()
                    except Exception:
                        pass
                finally:
                    write_db.close()

    async with httpx.AsyncClient() as http_client:
        # ── Startup catch-up: immediately resolve any positions missed while down ──
        try:
            await _sweep(http_client)
        except Exception as e:
            logger.error(f"Startup catch-up sweep error: {e}", exc_info=True)

        while True:
            await asyncio.sleep(PAPER_MONITOR_INTERVAL)
            try:
                await _sweep(http_client)
            except Exception as e:
                logger.error(f"Paper monitor loop error: {e}", exc_info=True)


# ─── Live position monitor ───────────────────────────────────────────────────

# Duplicate-alert protection is now handled at the DB layer via atomic
# UPDATE WHERE outcome='OPEN', so no in-memory sets are needed.
# Tracks how many consecutive sweeps each execution_id has been absent from
# Bitunix open positions.  We require 2 consecutive misses before closing,
# so a single API blip can't falsely end a live trade.
_reconcile_missing: dict = {}  # ex_id → consecutive_missing_count

async def _fetch_live_price_batch_tradfi(symbols: list) -> dict:
    """Route a batch of stock/forex/index symbols through yfinance."""
    from app.services.asset_classes import get_symbol as _ac_get
    from app.services.tradfi_prices import get_price as _tradfi_price
    out: dict = {}
    async def _one(sym: str):
        for cls in ("stock", "forex", "index"):
            if _ac_get(cls, sym):
                try:
                    px = await _tradfi_price(sym, cls)
                    if px and px > 0:
                        out[sym] = px
                except Exception as e:
                    logger.debug(f"tradfi batch price failed {sym}/{cls}: {e}")
                return
    await asyncio.gather(*[_one(s) for s in symbols])
    return out


async def _fetch_live_price_batch(symbols: list, http_client: httpx.AsyncClient) -> dict:
    """
    Fetch live perpetual-futures prices for a batch of symbols.
    Priority: MEXC Futures → Binance Futures.
    Returns {symbol: price}.
    """
    cache: dict = {}

    async def _mexc_futures():
        try:
            resp = await http_client.get(
                "https://contract.mexc.com/api/v1/contract/ticker", timeout=8
            )
            if resp.status_code == 200:
                data  = resp.json()
                items = data.get("data", data) if isinstance(data, dict) else data
                for item in items:
                    raw = item.get("symbol", "")          # e.g. ME_USDT
                    px  = float(item.get("lastPrice") or 0)
                    if px > 0:
                        plain = raw.replace("_", "")      # ME_USDT → MEUSDT
                        cache[plain] = px
        except Exception as e:
            logger.debug(f"[live-monitor] MEXC futures fetch err: {e}")

    async def _binance_futures():
        try:
            resp = await http_client.get(
                "https://fapi.binance.com/fapi/v1/ticker/price", timeout=8
            )
            if resp.status_code == 200:
                for item in resp.json():
                    sym = item.get("symbol", "")
                    px  = float(item.get("price") or 0)
                    if px > 0 and sym not in cache:
                        cache[sym] = px
        except Exception as e:
            logger.debug(f"[live-monitor] Binance futures fetch err: {e}")

    await asyncio.gather(_mexc_futures(), _binance_futures())

    result = {}
    for sym in symbols:
        plain = sym.replace("/", "").replace(":USDT", "").replace("-USDT", "").upper()
        if not plain.endswith("USDT"):
            plain += "USDT"
        if plain in cache:
            result[sym] = cache[plain]
    return result


async def run_live_position_monitor():
    """
    Background loop — monitors all OPEN live (is_paper=False) strategy executions.
    Checks SL/TP price levels every 30 s and closes + notifies on breach.
    """
    from app.database import BgSessionLocal as SessionLocal
    from app.strategy_models import StrategyExecution, UserStrategy
    from app.models import User

    logger.info("🔴 Live position monitor started (10s interval)")

    LIVE_MONITOR_INTERVAL = 10

    async def _close_live_execution_and_notify(ex, outcome, exit_price, db, source="price"):
        """Shared helper: commit a live execution close and fire the Telegram DM.

        Uses an atomic UPDATE WHERE outcome='OPEN' so that when multiple uvicorn
        workers race to close the same execution, exactly one sends the notification.
        """
        leverage = ex.leverage or 10
        if ex.direction == "LONG":
            raw_pnl = (exit_price - ex.entry_price) / ex.entry_price * 100
        else:
            raw_pnl = (ex.entry_price - exit_price) / ex.entry_price * 100

        pnl_pct   = round(raw_pnl * leverage, 2)
        closed_at = datetime.utcnow()

        # Atomic close — only the first worker wins.
        from sqlalchemy import text as _text
        try:
            result = db.execute(
                _text(
                    "UPDATE strategy_executions "
                    "SET outcome=:outcome, exit_price=:exit_price, pnl_pct=:pnl, closed_at=:closed_at "
                    "WHERE id=:id AND outcome='OPEN'"
                ),
                {"outcome": outcome, "exit_price": exit_price, "pnl": pnl_pct,
                 "closed_at": closed_at, "id": ex.id},
            )
            db.commit()
        except Exception as ce:
            logger.error(f"[live-monitor] DB commit error {ex.id}: {ce}")
            db.rollback()
            return False

        if result.rowcount == 0:
            # Another worker already closed this execution — skip notification.
            return True

        # Sync in-memory object
        ex.outcome    = outcome
        ex.exit_price = exit_price
        ex.pnl_pct    = pnl_pct
        ex.closed_at  = closed_at

        _update_performance(ex.strategy_id, db)

        logger.info(
            f"[live-monitor] {'TP' if outcome == 'WIN' else 'SL'} HIT ({source}): "
            f"{ex.symbol} {ex.direction} entry={ex.entry_price} "
            f"exit={exit_price} pnl={pnl_pct:+.1f}%"
        )

        try:
            user  = db.query(User).filter(User.id == ex.user_id).first()
            strat = db.query(UserStrategy).filter(UserStrategy.id == ex.strategy_id).first()
            strat_name = strat.name if strat else "Unknown"
            if user and user.telegram_id:
                tg_id = _telegram_int_id(user)
                if tg_id:
                    await _send_paper_close_dm(
                        tg_id,
                        _fmt_close_card(
                            strategy_name=strat_name,
                            symbol=ex.symbol,
                            direction=ex.direction,
                            entry=ex.entry_price,
                            exit_price=exit_price,
                            outcome=outcome,
                            pnl_pct=pnl_pct,
                            leverage=leverage,
                            fired_at=ex.fired_at,
                            closed_at=closed_at,
                            conditions=ex.conditions_met,
                            is_paper=False,
                        ),
                    )
            # Mobile push — trade close (live)
            from app.services.expo_push import notify_trade_close_bg
            dur_mins = int((closed_at - ex.fired_at).total_seconds() / 60) if ex.fired_at else 0
            notify_trade_close_bg(
                user_id=ex.user_id,
                strategy_name=strat_name,
                symbol=ex.symbol,
                direction=ex.direction,
                outcome=outcome,
                pnl_pct=pnl_pct,
                leverage=leverage,
                entry_price=ex.entry_price,
                exit_price=exit_price,
                strategy_id=ex.strategy_id,
                execution_id=ex.id,
                is_paper=False,
                duration_mins=dur_mins,
                kind="live",
                position_usd=float(ex.position_size) if ex.position_size else None,
            )
        except Exception as ne:
            logger.warning(f"[live-monitor] Notification error: {ne}")
        return True

    async def _sweep_live(http_client):
        """
        Connection-safe three-phase design:
          Phase 1  Read    — brief session: load positions + prefs, expunge, close.
          Phase 2  Compute — price evaluation + async Bitunix HTTP, NO DB held.
          Phase 3  Write   — brief per-position session for every close/note update.
        """
        # ── Phase 1: Read ─────────────────────────────────────────────────────
        from app.models import UserPreference
        db = SessionLocal()
        try:
            open_lives = (
                db.query(StrategyExecution)
                .filter(
                    StrategyExecution.outcome == "OPEN",
                    StrategyExecution.is_paper == False,
                    StrategyExecution.bitunix_order_id.isnot(None),
                )
                .all()
            )
            # Pre-fetch user API credentials while the session is still open.
            # We do this unconditionally so the finally block can close cleanly.
            user_ids   = list({ex.user_id for ex in open_lives}) if open_lives else []
            prefs_list = (
                db.query(UserPreference)
                .filter(UserPreference.user_id.in_(user_ids))
                .all()
            ) if user_ids else []
            # Detach everything so objects remain readable after session close.
            db.expunge_all()
        finally:
            db.close()  # ← released BEFORE any async I/O

        if not open_lives:
            return

        # Build a quick lookup: user_id → prefs (detached, attributes readable)
        prefs_by_user = {p.user_id: p for p in prefs_list}

        symbols     = list({ex.symbol for ex in open_lives})
        live_prices = await _fetch_live_price_batch(symbols, http_client)  # async HTTP, no DB

        logger.info(
            f"[live-monitor] Checking {len(open_lives)} live position(s) "
            f"across {len(symbols)} symbol(s)"
        )

        # ── Phase 2: Unrealised P&L notes (spot price, no close decisions) ────
        # For live trades Bitunix is the source of truth — actual closes are
        # detected by the Bitunix reconcile in Phase 3b below.
        # We never force-close from MEXC spot data: spot ≠ futures price and
        # can trigger false TP/SL hits on coins like ALICE.
        price_closed_ids: set = set()

        for ex in open_lives:
            live_px = live_prices.get(ex.symbol)
            if not live_px or not ex.entry_price:
                continue
            leverage = ex.leverage or 10
            if ex.direction == "LONG":
                pnl_pct = (live_px - ex.entry_price) / ex.entry_price * 100 * leverage
            else:
                pnl_pct = (ex.entry_price - live_px) / ex.entry_price * 100 * leverage
            note = (
                f"open · live={live_px:.6g} · "
                f"unrealised {'+' if pnl_pct >= 0 else ''}{pnl_pct:.1f}%"
            )
            note_db = SessionLocal()
            try:
                from sqlalchemy import text as _text2
                note_db.execute(
                    _text2("UPDATE strategy_executions SET notes=:n WHERE id=:id"),
                    {"n": note, "id": ex.id},
                )
                note_db.commit()
            except Exception:
                note_db.rollback()
            finally:
                note_db.close()

        # ── Bitunix reconciliation (async HTTP per user, no global DB held) ────
        # Accuracy rules:
        #   1. Require 2 consecutive misses before closing (API blip protection).
        #   2. Validate history entry_price matches ours (stale-history protection).
        #   3. Prefer realized_pnl sign → TP/SL distance → direction.
        #   4. Never close from missing/invalid history — skip and retry.
        #   5. Clear miss counter when position reappears.

        MIN_AGE_SECS = 120
        now = datetime.utcnow()
        still_open = [
            ex for ex in open_lives
            if ex.id not in price_closed_ids
            and ex.fired_at
            and (now - ex.fired_at).total_seconds() >= MIN_AGE_SECS
        ]

        if still_open:
            from app.services.bitunix_trader import BitunixTrader

            by_user: dict = {}
            for ex in still_open:
                by_user.setdefault(ex.user_id, []).append(ex)

            for user_id, execs in by_user.items():
                try:
                    prefs = prefs_by_user.get(user_id)
                    if not prefs or not getattr(prefs, "bitunix_api_key", None) \
                            or not getattr(prefs, "bitunix_api_secret", None):
                        continue

                    from app.utils.encryption import decrypt_api_key
                    try:
                        _raw_key = decrypt_api_key(prefs.bitunix_api_key)
                        _raw_sec = decrypt_api_key(prefs.bitunix_api_secret)
                    except Exception:
                        logger.warning(f"[live-monitor] Could not decrypt API keys for user {user_id} — skipping reconcile")
                        continue
                    trader = BitunixTrader(
                        api_key    = _raw_key,
                        api_secret = _raw_sec,
                    )
                    try:
                        bitunix_positions = await trader.get_open_positions()
                    except Exception as be:
                        logger.warning(f"[live-monitor] Bitunix reconcile fetch failed user {user_id}: {be}")
                        continue

                    bitunix_open: dict = {}
                    for p in bitunix_positions:
                        key = (p["symbol"], p["hold_side"].upper())
                        bitunix_open[key] = float(p.get("entryPrice") or p.get("entry_price") or 0)

                    for ex in execs:
                        key = (ex.symbol, ex.direction)
                        if key in bitunix_open:
                            _reconcile_missing.pop(ex.id, None)
                            continue

                        miss_count = _reconcile_missing.get(ex.id, 0) + 1
                        _reconcile_missing[ex.id] = miss_count

                        if miss_count < 2:
                            logger.info(
                                f"[live-monitor] {ex.symbol} id={ex.id} absent from "
                                f"Bitunix (miss {miss_count}/2) — waiting for confirmation"
                            )
                            continue

                        logger.info(
                            f"[live-monitor] {ex.symbol} {ex.direction} id={ex.id} "
                            f"confirmed absent (2 sweeps) — fetching close history"
                        )
                        close_hist = None
                        try:
                            close_hist = await trader.get_closed_position_history(ex.symbol)
                        except Exception as he:
                            logger.warning(f"[live-monitor] Close history error {ex.symbol}: {he}")

                        if not close_hist or close_hist.get("close_price", 0) <= 0:
                            logger.info(f"[live-monitor] {ex.symbol} id={ex.id} — no close history yet, retrying")
                            continue

                        exit_price   = float(close_hist["close_price"])
                        realized_pnl = float(close_hist.get("realized_pnl", 0))
                        hist_entry   = float(close_hist.get("entry_price", 0))
                        close_type   = close_hist.get("close_type", "")

                        forced_outcome = None  # set if stale-history force-close path fires
                        if hist_entry > 0 and ex.entry_price and ex.entry_price > 0:
                            entry_diff_pct = abs(hist_entry - ex.entry_price) / ex.entry_price
                            if entry_diff_pct > 0.02:
                                # Close history belongs to a different position (different entry).
                                # Keep the miss counter running — do NOT reset it to 0.
                                # After 8+ consecutive misses with stale history, force-close
                                # via TP/SL distance as best effort, since Bitunix clearly
                                # no longer holds this position open.
                                if miss_count >= 8 and ex.tp_price and ex.sl_price:
                                    logger.warning(
                                        f"[live-monitor] {ex.symbol} id={ex.id} — stale history "
                                        f"for {miss_count} sweeps, force-closing via TP/SL distance "
                                        f"(stale entry={hist_entry}, ours={ex.entry_price})"
                                    )
                                    # Use our own TP/SL prices as the exit — NOT the stale
                                    # close_price from Bitunix which belongs to a different
                                    # position and would produce wildly wrong PnL (e.g. -601%).
                                    dist_tp = abs(exit_price - ex.tp_price)
                                    dist_sl = abs(exit_price - ex.sl_price)
                                    if dist_tp <= dist_sl:
                                        forced_outcome = "WIN"
                                        exit_price = ex.tp_price   # use our TP as exit
                                    else:
                                        forced_outcome = "LOSS"
                                        exit_price = ex.sl_price   # use our SL as exit
                                    _reconcile_missing.pop(ex.id, None)
                                else:
                                    logger.warning(
                                        f"[live-monitor] {ex.symbol} id={ex.id} — history entry "
                                        f"{hist_entry} differs from ours {ex.entry_price} by "
                                        f"{entry_diff_pct*100:.2f}% — stale (miss {miss_count}), skipping"
                                    )
                                    continue

                        if forced_outcome is None:
                            _reconcile_missing.pop(ex.id, None)

                        # Determine outcome — priority order:
                        # 1. forced_outcome already set by stale force-close path above
                        # 2. close_type from Bitunix (most reliable when set)
                        # 3. TP/SL distance (reliable; accounts for SHORT direction)
                        # 4. Direction-adjusted realized_pnl (can be negative for SHORT wins
                        #    if Bitunix reports raw price-diff × size without direction flip)
                        # 5. direction vs entry/exit price fallback
                        if forced_outcome is not None:
                            outcome = forced_outcome
                        elif "TAKE" in close_type or close_type in ("TP", "TAKE_PROFIT"):
                            outcome = "WIN"
                        elif "STOP" in close_type or "LIQUID" in close_type or close_type in ("SL", "STOP_LOSS"):
                            outcome = "LOSS"
                        elif ex.tp_price and ex.sl_price:
                            dist_tp = abs(exit_price - ex.tp_price)
                            dist_sl = abs(exit_price - ex.sl_price)
                            outcome = "WIN" if dist_tp <= dist_sl else "LOSS"
                        else:
                            # Adjust realized_pnl for SHORT direction before using its sign
                            if ex.direction == "SHORT":
                                dir_pnl = -realized_pnl
                            else:
                                dir_pnl = realized_pnl
                            if dir_pnl > 0:
                                outcome = "WIN"
                            elif dir_pnl < 0:
                                outcome = "LOSS"
                            elif ex.direction == "LONG":
                                outcome = "WIN" if exit_price >= ex.entry_price else "LOSS"
                            else:
                                outcome = "WIN" if exit_price <= ex.entry_price else "LOSS"

                        logger.info(
                            f"[live-monitor] {ex.symbol} id={ex.id} → {outcome} "
                            f"exit={exit_price} realized_pnl={realized_pnl:+.4f} "
                            f"close_type='{close_type}' source=bitunix-reconcile"
                        )
                        # Brief write session for each reconcile close
                        rec_db = SessionLocal()
                        try:
                            await _close_live_execution_and_notify(
                                ex, outcome, exit_price, rec_db, source="bitunix-reconcile"
                            )
                        finally:
                            rec_db.close()

                except Exception as ue:
                    logger.error(f"[live-monitor] Reconcile error user {user_id}: {ue}", exc_info=True)

        # Trim the missing-counter dict to prevent unbounded growth
        if len(_reconcile_missing) > 500:
            _reconcile_missing.clear()

    async with httpx.AsyncClient() as http_client:
        # Startup catch-up
        try:
            await _sweep_live(http_client)
        except Exception as e:
            logger.error(f"[live-monitor] startup sweep error: {e}", exc_info=True)
        while True:
            await asyncio.sleep(LIVE_MONITOR_INTERVAL)
            try:
                await _sweep_live(http_client)
            except Exception as e:
                logger.error(f"[live-monitor] loop error: {e}", exc_info=True)


# ─── Strategy evaluation & firing ───────────────────────────────────────────

async def evaluate_and_fire(
    strategy,
    user,
    db,
    http_client: httpx.AsyncClient,
    raw_tickers: Optional[list] = None,
    gate_stats: Optional[Dict[str, int]] = None,
):
    """
    Evaluate one strategy. Fires a trade if conditions are met.
    paper=True strategies fire but skip Bitunix order placement.
    raw_tickers — pass the pre-fetched ticker list from the main loop so all
    strategies in one cycle share a single MEXC/Binance fetch instead of each
    making their own request.
    gate_stats — optional shared counter dict for per-cycle diagnostics.
    """
    from app.services.strategy_ta import evaluate_strategy_conditions
    from app.strategy_models import StrategyExecution, StrategyPortalSettings

    def _bump(key: str):
        if gate_stats is not None:
            gate_stats[key] = gate_stats.get(key, 0) + 1

    # Determine paper/live upfront.
    # Live strategies automatically downgrade to paper if the user doesn't have
    # Bitunix auto-trading set up — so every signal is always tracked.
    _wants_live = (strategy.status == "active")
    config   = dict(strategy.config or {})

    # ── Asset class (crypto | stock | forex | index) ────────────────────────
    # Determine broker BEFORE gating live so we don't run the Bitunix affiliate
    # check on a forex strategy (which routes through OANDA).
    from app.services.asset_classes import (
        normalize_asset_class, is_market_open, PAPER_ONLY_CLASSES,
    )
    _col_ac  = getattr(strategy, "asset_class", None) or ""
    # Mobile wizard saves asset_class as "_asset_class"; web portal uses "asset_class".
    # Check both so mobile-built forex/index strategies aren't silently treated as crypto.
    _cfg_ac  = config.get("asset_class") or config.get("_asset_class") or ""
    # If the DB column says 'crypto' but the config explicitly says something
    # else (e.g. 'forex'), trust the config — the column may have been set by
    # the DEFAULT before the backfill migration ran.
    if _col_ac == "crypto" and _cfg_ac and _cfg_ac != "crypto":
        asset_class = normalize_asset_class(_cfg_ac)
        logger.info(
            f"[Strategy {strategy.id}] asset_class mismatch: column='crypto' config='{_cfg_ac}' "
            f"→ using '{asset_class}' (run migration to fix)"
        )
    else:
        asset_class = normalize_asset_class(_col_ac or _cfg_ac)
    config["asset_class"] = asset_class

    # Per-asset broker gate. Forex → cTrader (FP Markets), crypto → Bitunix,
    # stocks/indices → paper-only (no broker integration yet).
    if asset_class in ("forex", "index"):
        _ctrader_live_ok = False
        try:
            from app.models import UserPreference as _UP
            _prefs = db.query(_UP).filter(_UP.user_id == user.id).first()
            _ctrader_live_ok = bool(
                _prefs
                and _prefs.ctrader_access_token
                and _prefs.ctrader_account_id
            )
        except Exception:
            _ctrader_live_ok = False
        is_paper = not (_wants_live and _ctrader_live_ok)
        if _wants_live and is_paper:
            logger.debug(
                f"[Strategy {strategy.id}] {asset_class.title()} live strategy downgraded to paper "
                f"(no cTrader credentials) — signal will still be tracked."
            )
    elif asset_class in PAPER_ONLY_CLASSES:
        # Stocks/indices: no broker yet, always paper.
        is_paper = True
    else:
        # Crypto: Bitunix gate (auto-trading + keys + affiliate roster).
        if _wants_live:
            _live_ok, _live_reason = await _user_can_live_trade_async(user, db)
            is_paper = not _live_ok
            if is_paper:
                logger.debug(
                    f"[Strategy {strategy.id}] Live strategy downgraded to paper "
                    f"(reason={_live_reason}) — signal will still be tracked."
                )
        else:
            is_paper = True

    if asset_class in PAPER_ONLY_CLASSES:
        if not is_market_open(asset_class):
            _bump(f"blk_market_closed_{asset_class}")
            return

    # ── EOD / close_before entry guard ─────────────────────────────────────
    # If the strategy has an intraday close-before time set, stop opening new
    # positions once that UTC time has passed today (Mon-Fri only).  Open trades
    # are handled independently by the trade_tracker EOD force-close loop.
    _close_before_cfg = config.get("exit", {}).get("close_before")
    if _close_before_cfg:
        try:
            from datetime import datetime as _dtnow
            _now_utc = _dtnow.utcnow()
            _h_eod, _m_eod = map(int, str(_close_before_cfg).split(":"))
            _eod_cut = _now_utc.replace(hour=_h_eod, minute=_m_eod, second=0, microsecond=0)
            if _now_utc >= _eod_cut and _now_utc.weekday() < 5:
                _bump("blk_eod_cutoff")
                return
        except Exception:
            pass

    # ── Forex day-trading guards (entry only) ────────────────────────────────
    if asset_class == "forex":
        _risk_cfg = config.get("risk") or {}
        _now_dt = datetime.utcnow()
        _wd = _now_dt.weekday()   # Mon=0 … Sun=6
        _h  = _now_dt.hour
        _m  = _now_dt.minute

        # 1. Friday close protection: no new trades Thu 21:00+ or Fri/Sat (weekend gap risk)
        if _risk_cfg.get("friday_close_protection"):
            if (_wd == 3 and (_h > 21 or (_h == 21 and _m >= 0))) or _wd >= 4:
                _bump("blk_friday_close")
                return

        # 2. No overnight positions: no new entries after 21:45 UTC Mon-Thu
        #    (positions must be closeable before NY close at 22:00)
        if _risk_cfg.get("no_overnight_positions"):
            if _wd < 4 and _h == 21 and _m >= 45:
                _bump("blk_no_overnight")
                return
            if _wd < 4 and _h >= 22:
                _bump("blk_no_overnight")
                return

        # 3. Daily pip target: sum today's WIN pip gains; block if hit
        _pip_target = _risk_cfg.get("daily_pip_target")
        if _pip_target:
            try:
                from app.services.forex_engine import pip_size as _psz
                _today_start = _now_dt.replace(hour=0, minute=0, second=0, microsecond=0)
                _today_wins = db.query(StrategyExecution).filter(
                    StrategyExecution.strategy_id == strategy.id,
                    StrategyExecution.outcome == "WIN",
                    StrategyExecution.fired_at >= _today_start,
                ).all()
                _total_pip_gain = 0.0
                for _wx in _today_wins:
                    if _wx.entry_price and _wx.close_price and _wx.symbol:
                        _ps = _psz(_wx.symbol)
                        if _ps > 0:
                            _entry = float(_wx.entry_price)
                            _close = float(_wx.close_price)
                            if _wx.direction == "LONG" and _close > _entry:
                                _total_pip_gain += (_close - _entry) / _ps
                            elif _wx.direction == "SHORT" and _close < _entry:
                                _total_pip_gain += (_entry - _close) / _ps
                if _total_pip_gain >= float(_pip_target):
                    _bump("blk_daily_pip_target")
                    logger.info(
                        f"[Strategy {strategy.id}] Daily pip target reached: "
                        f"{_total_pip_gain:.1f} pips >= target {_pip_target} — pausing for the day"
                    )
                    return
            except Exception as _pte:
                logger.debug(f"[Strategy {strategy.id}] daily pip target check error: {_pte}")

        # 4. Max trades per session: count fires that happened inside the same session window
        _max_per_sess = _risk_cfg.get("max_trades_per_session")
        if _max_per_sess:
            try:
                from app.services.forex_engine import current_sessions as _cur_sess
                _active_sessions = _cur_sess(_now_dt)
                if _active_sessions:
                    # Determine window start for the current active session(s)
                    from app.services.forex_engine import SESSIONS as _FX_SESSIONS
                    _sess_start = None
                    for _sid in _active_sessions:
                        _sw = _FX_SESSIONS.get(_sid)
                        if _sw:
                            _candidate = _now_dt.replace(
                                hour=_sw.open_h, minute=_sw.open_m, second=0, microsecond=0
                            )
                            # Handle sessions that started yesterday (e.g. Sydney)
                            if _candidate > _now_dt:
                                _candidate -= timedelta(days=1)
                            if _sess_start is None or _candidate > _sess_start:
                                _sess_start = _candidate
                    if _sess_start:
                        _sess_count = db.query(StrategyExecution).filter(
                            StrategyExecution.strategy_id == strategy.id,
                            StrategyExecution.fired_at >= _sess_start,
                        ).count()
                        if _sess_count >= int(_max_per_sess):
                            _bump("blk_session_cap")
                            logger.debug(
                                f"[Strategy {strategy.id}] Session cap: "
                                f"{_sess_count}/{_max_per_sess} trades this session"
                            )
                            return
            except Exception as _mps:
                logger.debug(f"[Strategy {strategy.id}] max_per_session check error: {_mps}")

    # Locked strategy — fetch live entry_conditions from the original source strategy
    if config.get("_locked") and config.get("_source_strategy_id"):
        try:
            from app.strategy_models import UserStrategy as _US
            _src = db.query(_US).filter(_US.id == config["_source_strategy_id"]).first()
            if _src and _src.config:
                config["entry_conditions"] = _src.config.get("entry_conditions", {})
        except Exception as _e:
            logger.warning(f"Locked strategy {strategy.id}: could not fetch source conditions: {_e}")

    risk     = config.get("risk") or {}
    filters  = config.get("filters") or {}
    universe = config.get("universe") or {}
    direction_pref = config.get("direction") or "LONG"

    if not _check_trading_days(filters):
        _bump("blk_trading_days")
        return
    if not _check_time_filter(filters):
        _bump("blk_time_filter")
        return
    if not _check_btc_regime(filters):
        _bump("blk_btc_regime")
        return

    max_per_day   = int(risk.get("max_trades_per_day") or 3)
    max_open      = int(risk.get("max_open_positions") or 1)
    cooldown_mins = int(risk.get("cooldown_minutes") or 30)

    if _daily_execution_count(strategy.id, db) >= max_per_day:
        _bump("blk_daily_cap")
        return
    if _open_execution_count(strategy.id, db) >= max_open:
        _bump("blk_max_open")
        return

    # ── Forex: daily pip loss cap ─────────────────────────────────────────────
    # If daily_loss_limit_pips is set, check the sum of pip losses today.
    # We convert to % using the current price so we can reuse the existing
    # closed-execution query.  Block the whole strategy if exceeded.
    if asset_class == "forex":
        _max_pip_loss = risk.get("daily_loss_limit_pips")
        if _max_pip_loss:
            try:
                from app.services.forex_engine import pip_size as _pip_size
                # Representative price from any symbol in universe to convert pips→%
                _rep_sym = (universe.get("symbols") or ["EURUSD"])[0]
                _rep_price = 1.08  # sensible default; will be overwritten below
                # Sum today's closed pip-losses from StrategyExecution
                _today_start = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
                _today_losses = db.query(StrategyExecution).filter(
                    StrategyExecution.strategy_id == strategy.id,
                    StrategyExecution.outcome.in_(["LOSS", "BREAKEVEN"]),
                    StrategyExecution.fired_at >= _today_start,
                ).all()
                _total_pip_loss = 0.0
                for _ex in _today_losses:
                    if _ex.entry_price and _ex.close_price and _ex.symbol:
                        _ps = _pip_size(_ex.symbol)
                        if _ps > 0:
                            _entry = float(_ex.entry_price)
                            _close = float(_ex.close_price)
                            _loss_price = abs(_entry - _close)
                            if _ex.direction == "LONG" and _close < _entry:
                                _total_pip_loss += _loss_price / _ps
                            elif _ex.direction == "SHORT" and _close > _entry:
                                _total_pip_loss += _loss_price / _ps
                if _total_pip_loss >= float(_max_pip_loss):
                    _bump("blk_daily_pip_loss")
                    logger.info(
                        f"[Strategy {strategy.id}] Daily pip loss cap hit: "
                        f"{_total_pip_loss:.1f} pips >= limit {_max_pip_loss} pips"
                    )
                    return
            except Exception as _dle:
                logger.debug(f"[Strategy {strategy.id}] daily pip loss check error: {_dle}")

    # Global cooldown: if ANY symbol fired within cooldown window, pause entire strategy
    last_global = _last_any_fired_time(strategy.id, db)
    if last_global:
        elapsed_global = (datetime.utcnow() - last_global).total_seconds() / 60
        if elapsed_global < cooldown_mins:
            _bump("blk_cooldown")
            logger.debug(
                f"[Strategy {strategy.id}] Global cooldown active — "
                f"{elapsed_global:.1f}/{cooldown_mins} min elapsed"
            )
            return

    # Strictness level based on max trades/day
    # 1-2/day = sniper (all conditions must pass + 80% pass-rate gate)
    # 3-5/day = selective (all conditions must pass)
    # 6+/day  = standard (configured AND/OR logic)
    if max_per_day <= 2:
        strictness_level = 2
    elif max_per_day <= 5:
        strictness_level = 1
    else:
        strictness_level = 0

    # User-set risk profile (Low / Medium / High) lifts the strictness floor.
    # The wizard surfaces this on Step 6 — Low = "only fire when every
    # confirmation aligns AND the bulk of conditions pass" (sniper), Medium =
    # "all conditions must pass" (selective), High = legacy behaviour. Always
    # max() so the existing max_per_day-derived strictness is never lowered.
    _profile_floor = {
        "low":    2,
        "medium": 1,
        "high":   0,
    }.get(str(risk.get("risk_profile") or "").lower().strip(), None)
    if _profile_floor is not None:
        strictness_level = max(strictness_level, _profile_floor)

    if asset_class != "crypto":
        # Stocks/forex/indices: universe.symbols is a curated list from the
        # mobile/web wizard (validated against the catalog at save time). No
        # Bitunix/MEXC filtering applies — just trust the configured symbols.
        symbols = [
            s.upper() for s in (universe.get("symbols") or [])
            if isinstance(s, str) and s.strip()
        ]
    else:
        symbols = await _get_eligible_symbols(universe, http_client, raw_tickers=raw_tickers)
    if not symbols:
        _bump("blk_empty_universe")
        # Diagnostic: dump the offending universe spec at most once per
        # strategy per minute. This makes it trivial to spot strategies whose
        # universe is misconfigured (e.g. type='specific' with no symbols, or
        # over-restrictive min_24h_change/min_volume_usd) without log spam.
        try:
            _now_min = int(datetime.utcnow().timestamp() // 60)
            _key = (strategy.id, _now_min)
            if _key not in _EMPTY_UNI_LOGGED:
                _EMPTY_UNI_LOGGED.add(_key)
                logger.warning(
                    f"[Strategy {strategy.id}] '{strategy.name}' empty_universe — "
                    f"no eligible symbols matched. universe={universe!r}"
                )
                if len(_EMPTY_UNI_LOGGED) > 1000:
                    _EMPTY_UNI_LOGGED.clear()
        except Exception:
            pass
        return

    no_duplicate_symbol = bool(risk.get("no_duplicate_symbol", False))

    # ── Step 1: Fast sync pre-filter (no awaits) ─────────────────────────────
    # Exclude symbols already fired today or still in cooldown window.
    candidate_symbols: List[str] = []
    for symbol in symbols[:50]:
        if no_duplicate_symbol and _fired_today_for_symbol(strategy.id, symbol, db):
            continue
        last_fired = _last_fired_time(strategy.id, symbol, db)
        if last_fired:
            elapsed_mins = (datetime.utcnow() - last_fired).total_seconds() / 60
            if elapsed_mins < cooldown_mins:
                continue
        candidate_symbols.append(symbol)

    if not candidate_symbols:
        _bump("blk_all_in_cooldown")
        return

    # ── Step 2: Parallel price + TA fetch for ALL candidates at once ──────────
    # Turns O(n × fetch_time) → O(fetch_time).  Results are also stored in the
    # shared _PRICE_TA_CACHE so subsequent calls inside evaluate_strategy_conditions
    # (which fetches its own klines) benefit from the same warm cache.
    _price_results = await asyncio.gather(
        *[_fetch_price_and_ta(sym, http_client, asset_class) for sym in candidate_symbols],
        return_exceptions=True,
    )
    price_map: Dict[str, Dict] = {
        sym: res
        for sym, res in zip(candidate_symbols, _price_results)
        if isinstance(res, dict) and res
    }

    # ── Step 3: Parallel HTF trend checks (if filter active) ─────────────────
    htf_pass: Dict[str, bool] = {}
    if filters.get("htf_trend") and price_map:
        _htf_syms    = [s for s in candidate_symbols if s in price_map]
        _htf_results = await asyncio.gather(
            *[_check_htf_trend(s, direction_pref, http_client, asset_class) for s in _htf_syms],
            return_exceptions=True,
        )
        htf_pass = {
            sym: (res is True)
            for sym, res in zip(_htf_syms, _htf_results)
        }

    # ── Step 4: Sequential condition evaluation — price data already cached ───
    _had_any_candidate = False
    _conditions_failed_for_all = True
    for symbol in candidate_symbols:
        price_data = price_map.get(symbol)
        if not price_data:
            continue
        _had_any_candidate = True

        if filters.get("htf_trend") and not htf_pass.get(symbol):
            logger.debug(f"[Strategy {strategy.id}] HTF trend filter blocked {symbol}")
            _bump("blk_htf_trend_sym")
            continue

        enhanced_ta   = price_data.get("enhanced_ta", {})
        current_price = price_data.get("price", 0)
        if not current_price:
            continue

        # ── Forex: spread filter ──────────────────────────────────────────
        # Block this symbol if the live bid/ask spread exceeds max_spread_pips.
        # We estimate spread from the price data bid/ask if available,
        # otherwise skip the check (fail-open so we don't false-block on stale data).
        if asset_class == "forex":
            _max_sp = risk.get("max_spread_pips")
            if _max_sp:
                try:
                    from app.services.forex_engine import pip_size as _psz
                    _bid = price_data.get("bid") or price_data.get("bid_price")
                    _ask = price_data.get("ask") or price_data.get("ask_price")
                    if _bid and _ask and _ask > _bid:
                        _spread_price = _ask - _bid
                        _spread_pips  = _spread_price / _psz(symbol)
                        if _spread_pips > float(_max_sp):
                            _bump("blk_spread_filter")
                            logger.info(
                                f"[Strategy {strategy.id}] {symbol} spread "
                                f"{_spread_pips:.1f} pips > max {_max_sp} — skipping"
                            )
                            continue
                except Exception as _sfe:
                    logger.debug(f"[Strategy {strategy.id}] spread filter error: {_sfe}")

        passed, details = await evaluate_strategy_conditions(
            config, symbol, price_data, enhanced_ta, http_client,
            strictness_level=strictness_level
        )
        if not passed:
            continue
        _conditions_failed_for_all = False
        # break out of "failure tracking" — we have a real fire below

        mode_tag = "🧪 [PAPER]" if is_paper else "🎯"
        logger.info(
            f"{mode_tag} [Strategy {strategy.id}] {strategy.name} — "
            f"{symbol} conditions met! {direction_pref}"
        )

        ex_config = config.get("exit") or {}
        tp_pct    = float(ex_config.get("take_profit_pct")  or 3.0)
        tp2_pct   = ex_config.get("take_profit2_pct") or None
        sl_pct    = float(ex_config.get("stop_loss_pct")   or 1.5)
        leverage  = int(risk.get("leverage") or 10)

        # Defense-in-depth: stocks have no realistic leverage in paper testing —
        # clamp to 1× so P&L isn't artificially inflated. Forex and indices
        # DO use leverage in real trading (100:1 is standard on XAUUSD/EUR pairs),
        # so their configured leverage is honoured even in paper mode so that
        # paper performance reflects what a live trade would actually make.
        if asset_class == "stock":
            leverage = 1

        # ── Forex pip→% conversion ────────────────────────────────────────
        # Forex strategies set TP/SL in pips (which is how traders actually
        # think). We convert to the % move the existing exit engine
        # understands using the live price + the pair's pip size. This means
        # a "20 pip SL on EURUSD at 1.0850" produces the same TP/SL price
        # whether the user typed it as pips or as a %.
        if asset_class == "forex":
            from app.services.forex_engine import pips_to_pct as _p2p
            tp_pips = ex_config.get("take_profit_pips")
            sl_pips = ex_config.get("stop_loss_pips")
            tp2_pips = ex_config.get("take_profit2_pips")
            if tp_pips:
                tp_pct = _p2p(symbol, current_price, float(tp_pips))
            if sl_pips:
                sl_pct = _p2p(symbol, current_price, float(sl_pips))
            if tp2_pips:
                tp2_pct = _p2p(symbol, current_price, float(tp2_pips))
            # Pip-based trailing stop → convert to % so paper monitor + cTrader
            # trailing logic both use the same % field.
            _trail_pips = ex_config.get("trailing_stop_pips")
            if _trail_pips and float(_trail_pips) > 0:
                ex_config = dict(ex_config)  # don't mutate config in-place
                ex_config["trailing_stop"] = True
                ex_config["trailing_stop_pct"] = _p2p(symbol, current_price, float(_trail_pips))
            # Pip-based breakeven trigger → convert to % ROI threshold
            _be_pips = ex_config.get("breakeven_at_pips")
            if _be_pips and float(_be_pips) > 0:
                ex_config = dict(ex_config) if not isinstance(ex_config, dict) or id(ex_config) == id(config.get("exit")) else ex_config
                _be_pct = _p2p(symbol, current_price, float(_be_pips))
                # breakeven_at_pct is % leveraged ROI, so multiply by leverage
                ex_config["breakeven_at_pct"] = _be_pct * max(1, leverage)

        direction = direction_pref
        if direction == "BOTH":
            # Infer direction from directional conditions (FVG, order block, divergence, COD)
            # before falling back to RSI — a bullish FVG must never produce a SHORT.
            inferred_dir = None
            for _cond in config.get("entry_conditions", {}).get("conditions", []):
                _ct = _cond.get("type", "")
                _d = None
                if _ct == "fvg":
                    _d = _cond.get("direction") or _cond.get("fvg_dir")
                elif _ct in ("order_block", "ob"):
                    _d = _cond.get("ob_type") or _cond.get("direction")
                elif _ct in ("divergence", "cod", "change_of_direction"):
                    _d = _cond.get("direction")
                if _d and _d not in ("any", "both"):
                    inferred_dir = "LONG" if _d == "bullish" else "SHORT"
                    break
            if inferred_dir:
                direction = inferred_dir
            else:
                rsi = price_data.get("rsi", 50)
                direction = "LONG" if rsi > 50 else "SHORT"

        if direction == "LONG":
            tp_price  = current_price * (1 + tp_pct  / 100)
            sl_price  = current_price * (1 - sl_pct  / 100)
            tp2_price = current_price * (1 + float(tp2_pct) / 100) if tp2_pct else None
        else:
            tp_price  = current_price * (1 - tp_pct  / 100)
            sl_price  = current_price * (1 + sl_pct  / 100)
            tp2_price = current_price * (1 - float(tp2_pct) / 100) if tp2_pct else None

        # Stash partial_close_pct in notes so the paper monitor can act on it.
        _partial_close_pct = ex_config.get("partial_close_pct")
        _exec_notes = None
        if _partial_close_pct and float(_partial_close_pct) > 0:
            _exec_notes = f"partial_close_pct={float(_partial_close_pct):.0f}"

        execution = StrategyExecution(
            strategy_id    = strategy.id,
            user_id        = user.id,
            symbol         = symbol,
            direction      = direction,
            entry_price    = current_price,
            tp_price       = tp_price,
            tp2_price      = tp2_price,
            sl_price       = sl_price,
            leverage       = leverage,
            outcome        = "OPEN",
            conditions_met = details,
            fired_at       = datetime.utcnow(),
            is_paper       = is_paper,
            asset_class    = asset_class,
            notes          = _exec_notes,
        )
        db.add(execution)
        db.commit()
        db.refresh(execution)

        # Only increment open_trades counter — do NOT recompute from scratch
        # (full recompute wipes historical performance data if no closed trades exist yet)
        from app.strategy_models import StrategyPerformance
        _perf = db.query(StrategyPerformance).filter(
            StrategyPerformance.strategy_id == strategy.id
        ).first()
        if _perf:
            _perf.open_trades = (_perf.open_trades or 0) + 1
        else:
            _perf = StrategyPerformance(strategy_id=strategy.id, open_trades=1)
            db.add(_perf)
        db.commit()

        if is_paper:
            # Paper trade: notify user and skip Bitunix
            try:
                portal_settings = db.query(StrategyPortalSettings).filter(StrategyPortalSettings.user_id == user.id).first()
                tg_id = _telegram_int_id(user)
                if tg_id and (not portal_settings or portal_settings.dm_paper_alerts):
                    await _tg_send(
                        tg_id,
                        _fmt_open_card(
                            strategy_name = strategy.name,
                            symbol        = symbol,
                            direction     = direction,
                            entry         = current_price,
                            tp_price      = tp_price,
                            tp_pct        = tp_pct,
                            tp2_price     = tp2_price,
                            tp2_pct       = float(tp2_pct) if tp2_pct else None,
                            sl_price      = sl_price,
                            sl_pct        = sl_pct,
                            leverage      = leverage,
                            conditions    = details,
                            is_paper      = True,
                            asset_class   = asset_class,
                        ),
                    )
                # Mobile push (fire-and-forget; never raises)
                from app.services.expo_push import notify_user_bg
                _coin = symbol.replace("USDT", "")
                notify_user_bg(
                    user.id,
                    title=f"📝 {strategy.name}",
                    body=f"Paper trade: {_coin} {direction} {leverage}× @ ${current_price:,.4f}",
                    data={"type": "trade_open", "strategy_id": strategy.id, "kind": "paper"},
                    kind="paper",
                    position_usd=float(risk.get("position_size_usd") or 0) or None,
                )
            except Exception as e:
                logger.warning(f"Paper DM failed: {e}")
        else:
            # Live trade: route by asset class. Forex + indices → cTrader (FP Markets),
            # crypto → Bitunix. Stocks can't reach this branch (paper-lock above).
            order_id    = None
            actual_fill = None
            _broker     = "ctrader" if asset_class in ("forex", "index") else "bitunix"
            try:
                ps_type      = risk.get("position_size_type", "pct")
                _risk_usd    = float(risk["position_size_usd"]) if ps_type == "fixed" and risk.get("position_size_usd") else None
                if _broker == "ctrader":
                    from app.services.ctrader_client import place_ctrader_order_for_user
                    # Risk % auto lot sizing: when use_risk_pct=True, the wizard
                    # stored risk_pct_per_trade (% of account to risk).  We pass
                    # it to the cTrader helper which fetches the account balance
                    # and computes lots = risk% × balance / (sl_pips × pip_value).
                    _use_risk_pct = bool(risk.get("use_risk_pct"))
                    _risk_pct_per = float(risk.get("risk_pct_per_trade") or risk.get("position_size_pct") or 1.0)
                    order_result = await place_ctrader_order_for_user(
                        user           = user,
                        symbol         = symbol,
                        direction      = direction,
                        entry_price    = current_price,
                        tp_pct         = tp_pct,
                        sl_pct         = sl_pct,
                        risk_pct       = _risk_pct_per,
                        risk_usd       = _risk_usd,
                        use_risk_pct   = _use_risk_pct,
                        sl_pips        = float(ex_config.get("stop_loss_pips") or 0) or None,
                    )
                else:
                    from app.services.strategy_trader import place_bitunix_order_for_user
                    order_result = await place_bitunix_order_for_user(
                        user        = user,
                        symbol      = symbol,
                        direction   = direction,
                        leverage    = leverage,
                        entry_price = current_price,
                        tp_pct      = tp_pct,
                        sl_pct      = sl_pct,
                        risk_pct    = float(risk.get("position_size_pct") or 5),
                        risk_usd    = _risk_usd,
                    )
                if order_result:
                    order_id    = order_result.get("order_id")
                    actual_fill = order_result.get("actual_fill")
            except Exception as e:
                logger.error(f"[Strategy {strategy.id}] Order error: {e}")

                # Price-past-TP: market moved through TP before order placed.
                # Cancel entirely — no paper fallback, opportunity is gone.
                if "PRICE_PAST_TP" in str(e):
                    execution.outcome = "CANCELLED"
                    execution.notes   = f"Cancelled: {str(e)[:200]}"
                    db.commit()
                    logger.warning(
                        f"[Strategy {strategy.id}] {symbol} signal cancelled — "
                        f"live price already past TP before order placed."
                    )
                    tg_id_ex = _telegram_int_id(user)
                    if tg_id_ex:
                        try:
                            coin = symbol.replace("USDT", "")
                            await _tg_send(
                                tg_id_ex,
                                f"🚫 <b>Signal cancelled — price already at TP</b>\n"
                                f"Strategy: <b>{strategy.name}</b>\n"
                                f"Signal: {coin} {direction} {leverage}×\n"
                                f"TP: <code>${tp_price:,.4f}</code>\n\n"
                                f"<i>By the time the order reached Bitunix, the market had "
                                f"already moved to/past your take-profit. The trade was "
                                f"cancelled — no position opened.</i>"
                            )
                        except Exception:
                            pass
                    break  # execution cancelled — stop processing matches

                # All other errors: flip to paper so the signal's ROI is still tracked.
                execution.is_paper = True
                execution.notes    = f"Live→Paper fallback (order exception): {str(e)[:200]}"
                db.commit()
                logger.warning(
                    f"[Strategy {strategy.id}] Live order threw an exception for {symbol} — "
                    f"converting execution #{execution.id} to paper trade for ROI tracking."
                )
                tg_id_ex = _telegram_int_id(user)
                if tg_id_ex:
                    try:
                        await _tg_send(
                            tg_id_ex,
                            f"⚠️ <b>{_broker.title()} error — paper trade started</b>\n"
                            f"Strategy: <b>{strategy.name}</b>\n"
                            f"Signal: {symbol.replace('USDT','')} {direction} {leverage}×\n"
                            f"Entry: <code>${current_price:,.4f}</code>  "
                            f"TP: <code>${tp_price:,.4f}</code>  SL: <code>${sl_price:,.4f}</code>\n\n"
                            f"<i>The live order could not be placed (see error below). "
                            f"This trade is being tracked as a 🧪 paper position so your "
                            f"strategy's performance is still recorded.</i>\n"
                            f"Error: <code>{str(e)[:120]}</code>"
                        )
                    except Exception:
                        pass
                # Mobile push (fire-and-forget)
                try:
                    from app.services.expo_push import notify_user_bg
                    _coin = symbol.replace("USDT", "")
                    notify_user_bg(
                        user.id,
                        title=f"⚠️ {strategy.name}",
                        body=f"Live order failed → paper: {_coin} {direction} {leverage}×",
                        data={"type": "trade_open", "strategy_id": strategy.id, "kind": "paper_fallback"},
                        kind="paper",
                        position_usd=float(risk.get("position_size_usd") or 0) or None,
                    )
                except Exception:
                    pass
                # Still propagate to subscriber copies even when the owner's
                # live order failed — subscribers should receive a paper signal.
                if not config.get("_locked"):
                    asyncio.create_task(_propagate_to_subscribers(
                        source_strategy_id=strategy.id,
                        source_execution_id=execution.id,
                        http_client=http_client,
                    ))
                break  # paper execution is now open — stop processing matches

            if order_id:
                if _broker == "ctrader":
                    execution.ctrader_order_id = str(order_id)
                else:
                    execution.bitunix_order_id = str(order_id)
                if (
                    actual_fill
                    and actual_fill > 0
                    and execution.entry_price
                    and abs(actual_fill - execution.entry_price) / execution.entry_price > 0.0005
                ):
                    logger.info(
                        f"[Strategy {strategy.id}] entry_price updated: "
                        f"signal={execution.entry_price:.6g} → fill={actual_fill:.6g}"
                    )
                    execution.entry_price = actual_fill
                db.commit()
                display_entry = actual_fill if actual_fill else current_price
                tg_id_live = _telegram_int_id(user)
                if tg_id_live:
                    try:
                        await _tg_send(
                            tg_id_live,
                            _fmt_open_card(
                                strategy_name = strategy.name,
                                symbol        = symbol,
                                direction     = direction,
                                entry         = display_entry,
                                tp_price      = tp_price,
                                tp_pct        = tp_pct,
                                tp2_price     = tp2_price,
                                tp2_pct       = float(tp2_pct) if tp2_pct else None,
                                sl_price      = sl_price,
                                sl_pct        = sl_pct,
                                leverage      = leverage,
                                conditions    = details,
                                is_paper      = False,
                                order_id      = str(order_id),
                                asset_class   = asset_class,
                            ),
                        )
                        # Mobile push (fire-and-forget) — fires for live trades.
                        from app.services.expo_push import notify_user_bg
                        _coin = symbol.replace("USDT", "")
                        notify_user_bg(
                            user.id,
                            title=f"🚀 {strategy.name}",
                            body=f"Live trade: {_coin} {direction} {leverage}× @ ${display_entry:,.4f}",
                            data={"type": "trade_open", "strategy_id": strategy.id, "kind": "live"},
                            kind="live",
                            position_usd=float(risk.get("position_size_usd") or 0) or None,
                        )
                    except Exception as e:
                        logger.warning(f"Live DM failed: {e}")
            else:
                # Fallback: order_id was None (shouldn't normally reach here post-refactor)
                execution.is_paper = True
                execution.notes    = f"Live→Paper fallback: {_broker} returned no order_id"
                db.commit()
                logger.warning(
                    f"[Strategy {strategy.id}] Live order for {symbol} returned no order_id "
                    f"— converting execution #{execution.id} to paper trade for ROI tracking."
                )
                tg_id_live = _telegram_int_id(user)
                if tg_id_live:
                    try:
                        await _tg_send(
                            tg_id_live,
                            f"⚠️ <b>{_broker.title()} order not confirmed — paper trade started</b>\n"
                            f"Strategy: <b>{strategy.name}</b>\n"
                            f"Signal: {symbol.replace('USDT','')} {direction} {leverage}× lev\n"
                            f"Entry: <code>${current_price:,.4f}</code>\n"
                            f"TP: <code>${tp_price:,.4f}</code> (+{tp_pct}%)  "
                            f"SL: <code>${sl_price:,.4f}</code> (-{sl_pct}%)\n\n"
                            f"<i>{_broker.title()} did not return an order ID. The signal is being tracked "
                            f"as a 🧪 paper position. Check your API key has trading permission.</i>"
                        )
                    except Exception:
                        pass

        # Propagate this trade to all active subscriber copies so they enter at
        # the exact same price simultaneously, instead of evaluating independently
        # (which can cause early fills and divergent SL hits).
        if not config.get("_locked"):
            asyncio.create_task(_propagate_to_subscribers(
                source_strategy_id=strategy.id,
                source_execution_id=execution.id,
                http_client=http_client,
            ))

        break  # one trade per strategy per scan cycle
    else:
        # for-loop completed without break → no fire happened.
        # Bump a gate so we know WHY the strategy didn't fire.
        if not _had_any_candidate:
            _bump("blk_no_price_data")
        elif _conditions_failed_for_all:
            _bump("blk_ta_conditions")


# ─── Subscriber copy-trade propagation ───────────────────────────────────────

async def _propagate_to_subscribers(
    source_strategy_id: int,
    source_execution_id: int,
    http_client,
):
    """
    After a source (non-locked) strategy fires a trade, immediately replicate
    it for every active subscriber copy at the SAME entry/TP/SL prices.

    This guarantees all copies enter simultaneously, preventing the scenario
    where independent condition evaluation fires them minutes earlier/later
    at a worse price and different SL level.
    """
    from app.database import BgSessionLocal as SessionLocal
    from app.models import User
    from app.strategy_models import (
        UserStrategy, StrategyExecution, StrategyPerformance,
        PortalSubscription, StrategyPortalSettings,
    )

    # Re-fetch source execution in a fresh session
    _src_db = SessionLocal()
    try:
        src_exec = _src_db.query(StrategyExecution).filter(
            StrategyExecution.id == source_execution_id
        ).first()
        if not src_exec:
            return

        entry     = src_exec.entry_price
        tp_price  = src_exec.tp_price
        tp2_price = src_exec.tp2_price
        sl_price  = src_exec.sl_price
        symbol    = src_exec.symbol
        direction = src_exec.direction

        # Find all active/paper locked subscriber copies of this source
        sub_strategies = _src_db.query(UserStrategy).filter(
            UserStrategy.status.in_(["active", "paper"])
        ).all()
        copies = [
            s for s in sub_strategies
            if (s.config or {}).get("_locked")
            and (s.config or {}).get("_source_strategy_id") == source_strategy_id
        ]
    finally:
        _src_db.close()

    if not copies:
        return

    logger.info(
        f"[Propagate] Source strategy {source_strategy_id} fired {symbol} "
        f"{direction} @ {entry} — propagating to {len(copies)} subscriber copies"
    )

    # Raw TP/SL percentages from entry (not leveraged) — used by Bitunix order
    tp_pct_raw = abs(tp_price - entry) / entry * 100 if tp_price else 0
    sl_pct_raw = abs(sl_price - entry) / entry * 100 if sl_price else 0

    for sub_strategy in copies:
        _sub_db = SessionLocal()
        try:
            sub_user = _sub_db.query(User).filter(User.id == sub_strategy.user_id).first()
            if not sub_user or sub_user.banned:
                continue

            # Pro / grandfathered check
            _now = datetime.utcnow()
            _psub = _sub_db.query(PortalSubscription).filter_by(user_id=sub_user.id).first()
            _has_pro = (
                _psub and _psub.tier == "pro"
                and _psub.subscription_end
                and _psub.subscription_end > _now
            )
            if not (sub_user.is_admin or sub_user.grandfathered or _has_pro):
                continue

            sub_config  = dict(sub_strategy.config or {})
            sub_risk    = sub_config.get("risk", {})
            sub_filters = sub_config.get("filters", {})

            # Session / time-of-day filter — subscriber can restrict when copies fire
            if not _check_time_filter(sub_filters):
                logger.debug(
                    f"[Propagate] Strategy {sub_strategy.id} blocked by session/time filter"
                )
                continue
            if not _check_trading_days(sub_filters):
                logger.debug(
                    f"[Propagate] Strategy {sub_strategy.id} blocked by trading-day filter"
                )
                continue

            # Daily limit
            if _daily_execution_count(sub_strategy.id, _sub_db) >= int(sub_risk.get("max_trades_per_day", 3)):
                logger.debug(f"[Propagate] Strategy {sub_strategy.id} at daily limit — skip")
                continue

            # Open position limit
            if _open_execution_count(sub_strategy.id, _sub_db) >= int(sub_risk.get("max_open_positions", 1)):
                logger.debug(f"[Propagate] Strategy {sub_strategy.id} max open positions — skip")
                continue

            # Cooldown
            last_fired = _last_any_fired_time(sub_strategy.id, _sub_db)
            if last_fired:
                elapsed = (_now - last_fired).total_seconds() / 60
                cooldown = int(sub_risk.get("cooldown_minutes", 30))
                if elapsed < cooldown:
                    logger.debug(f"[Propagate] Strategy {sub_strategy.id} in cooldown — skip")
                    continue

            # Use subscriber's own leverage & size but source's price/symbol/direction
            leverage   = int(sub_risk.get("leverage", 10))
            _wants_live = sub_strategy.status == "active"
            _live_reason = "ok"
            # P5e-2: broker-aware live gate. Forex copies route through OANDA,
            # crypto through Bitunix. Stocks/indices stay paper.
            try:
                from app.services.asset_classes import normalize_asset_class as _norm_ac
                _sub_asset_class = _norm_ac(
                    getattr(sub_strategy, "asset_class", None) or sub_config.get("asset_class")
                )
            except Exception:
                _sub_asset_class = getattr(sub_strategy, "asset_class", None) or sub_config.get("asset_class") or "crypto"
            if _wants_live:
                if _sub_asset_class in ("forex", "index"):
                    try:
                        from app.models import UserPreference as _UP_sub
                        _sub_prefs = _sub_db.query(_UP_sub).filter(_UP_sub.user_id == sub_user.id).first()
                        _can_live = bool(_sub_prefs and _sub_prefs.ctrader_access_token and _sub_prefs.ctrader_account_id)
                        _live_reason = "ok" if _can_live else "no_ctrader_credentials"
                    except Exception as _e:
                        _can_live = False
                        _live_reason = f"ctrader_check_error:{_e}"
                elif _sub_asset_class == "stock":
                    _can_live = False
                    _live_reason = "paper_only_asset_class"
                else:
                    _can_live, _live_reason = await _user_can_live_trade_async(sub_user, _sub_db)
                is_paper  = not _can_live
                if is_paper:
                    logger.info(
                        f"[Propagate] Strategy {sub_strategy.id} (user {sub_user.username}) "
                        f"downgraded to PAPER — reason={_live_reason}"
                    )
            else:
                is_paper = True

            _open_note = None
            if is_paper and _wants_live:
                _open_note = f"Live→Paper: {_live_reason}"

            sub_exec = StrategyExecution(
                strategy_id    = sub_strategy.id,
                user_id        = sub_user.id,
                symbol         = symbol,
                direction      = direction,
                entry_price    = entry,
                tp_price       = tp_price,
                tp2_price      = tp2_price,
                sl_price       = sl_price,
                leverage       = leverage,
                outcome        = "OPEN",
                conditions_met = [f"✅ Copied from source strategy #{source_strategy_id} @ {entry:.6g}"],
                fired_at       = _now,
                is_paper       = is_paper,
                notes          = _open_note,
                asset_class    = getattr(sub_strategy, "asset_class", None) or "crypto",
            )
            _sub_db.add(sub_exec)
            _sub_db.commit()
            _sub_db.refresh(sub_exec)

            # Performance counter
            _perf = _sub_db.query(StrategyPerformance).filter(
                StrategyPerformance.strategy_id == sub_strategy.id
            ).first()
            if _perf:
                _perf.open_trades = (_perf.open_trades or 0) + 1
            else:
                _perf = StrategyPerformance(strategy_id=sub_strategy.id, open_trades=1)
                _sub_db.add(_perf)
            _sub_db.commit()

            portal_settings = _sub_db.query(StrategyPortalSettings).filter(
                StrategyPortalSettings.user_id == sub_user.id
            ).first()
            tg_id = _telegram_int_id(sub_user)

            if is_paper:
                if tg_id and (not portal_settings or portal_settings.dm_paper_alerts):
                    try:
                        await _tg_send(
                            tg_id,
                            _fmt_open_card(
                                strategy_name = sub_strategy.name,
                                symbol        = symbol,
                                direction     = direction,
                                entry         = entry,
                                tp_price      = tp_price,
                                tp_pct        = round(tp_pct_raw, 2),
                                tp2_price     = tp2_price,
                                tp2_pct       = round(abs(tp2_price - entry) / entry * 100, 2) if tp2_price else None,
                                sl_price      = sl_price,
                                sl_pct        = round(sl_pct_raw, 2),
                                leverage      = leverage,
                                conditions    = sub_exec.conditions_met,
                                is_paper      = True,
                                asset_class   = asset_class,
                            ),
                        )
                    except Exception as _e:
                        logger.warning(f"[Propagate] Paper DM failed for strategy {sub_strategy.id}: {_e}")
            else:
                # Live — route by asset class: forex/index → cTrader, else → Bitunix.
                order_id    = None
                actual_fill = None
                _sub_broker = "ctrader" if _sub_asset_class in ("forex", "index") else "bitunix"
                try:
                    ps_type      = sub_risk.get("position_size_type", "pct")
                    _sub_risk_usd = float(sub_risk["position_size_usd"]) if ps_type == "fixed" and sub_risk.get("position_size_usd") else None
                    if _sub_broker == "ctrader":
                        from app.services.ctrader_client import place_ctrader_order_for_user
                        order_result = await place_ctrader_order_for_user(
                            user        = sub_user,
                            symbol      = symbol,
                            direction   = direction,
                            entry_price = entry,
                            tp_pct      = tp_pct_raw,
                            sl_pct      = sl_pct_raw,
                            risk_pct    = float(sub_risk.get("position_size_pct", 5)),
                            risk_usd    = _sub_risk_usd,
                        )
                    else:
                        from app.services.strategy_trader import place_bitunix_order_for_user
                        order_result = await place_bitunix_order_for_user(
                            user        = sub_user,
                            symbol      = symbol,
                            direction   = direction,
                            leverage    = leverage,
                            entry_price = entry,
                            tp_pct      = tp_pct_raw,
                            sl_pct      = sl_pct_raw,
                            risk_pct    = float(sub_risk.get("position_size_pct", 5)),
                            risk_usd    = _sub_risk_usd,
                        )
                    if order_result:
                        order_id    = order_result.get("order_id")
                        actual_fill = order_result.get("actual_fill")
                except Exception as _e:
                    logger.error(f"[Propagate] Order error for strategy {sub_strategy.id}: {_e}")
                    sub_exec.is_paper = True
                    sub_exec.notes = f"Live→Paper fallback (propagate order error): {str(_e)[:200]}"
                    _sub_db.commit()
                    if tg_id:
                        try:
                            await _tg_send(
                                tg_id,
                                f"⚠️ <b>{_sub_broker.title()} error — paper trade started</b>\n"
                                f"Strategy: <b>{sub_strategy.name}</b>\n"
                                f"Signal: {symbol.replace('USDT','')} {direction} {leverage}×\n"
                                f"Entry: <code>${entry:,.4f}</code>  "
                                f"TP: <code>${tp_price:,.4f}</code>  SL: <code>${sl_price:,.4f}</code>\n\n"
                                f"<i>Live order could not be placed. Tracked as 🧪 paper.</i>\n"
                                f"Error: <code>{str(_e)[:120]}</code>"
                            )
                        except Exception:
                            pass
                    continue

                if order_id:
                    if _sub_broker == "ctrader":
                        sub_exec.ctrader_order_id = str(order_id)
                    else:
                        sub_exec.bitunix_order_id = str(order_id)
                    if (
                        actual_fill
                        and actual_fill > 0
                        and sub_exec.entry_price
                        and abs(actual_fill - sub_exec.entry_price) / sub_exec.entry_price > 0.0005
                    ):
                        logger.info(
                            f"[Propagate] Strategy {sub_strategy.id} entry_price updated: "
                            f"signal={sub_exec.entry_price:.6g} → fill={actual_fill:.6g}"
                        )
                        sub_exec.entry_price = actual_fill
                    _sub_db.commit()
                    display_entry = actual_fill if actual_fill else entry
                    if tg_id:
                        try:
                            await _tg_send(
                                tg_id,
                                _fmt_open_card(
                                    strategy_name = sub_strategy.name,
                                    symbol        = symbol,
                                    direction     = direction,
                                    entry         = display_entry,
                                    tp_price      = tp_price,
                                    tp_pct        = round(tp_pct_raw, 2),
                                    tp2_price     = tp2_price,
                                    tp2_pct       = round(abs(tp2_price - entry) / entry * 100, 2) if tp2_price else None,
                                    sl_price      = sl_price,
                                    sl_pct        = round(sl_pct_raw, 2),
                                    leverage      = leverage,
                                    conditions    = sub_exec.conditions_met,
                                    is_paper      = False,
                                    order_id      = str(order_id),
                                    asset_class   = asset_class,
                                ),
                            )
                        except Exception as _e:
                            logger.warning(f"[Propagate] Live DM failed for strategy {sub_strategy.id}: {_e}")
                else:
                    sub_exec.is_paper = True
                    sub_exec.notes = f"Live→Paper fallback: {_sub_broker} returned no order_id"
                    _sub_db.commit()

            logger.info(
                f"[Propagate] ✅ Strategy {sub_strategy.id} (user {sub_user.username}) "
                f"{symbol} {direction} @ {entry} — "
                f"{'paper' if sub_exec.is_paper else 'live #' + str(sub_exec.bitunix_order_id)}"
            )

        except Exception as _e:
            logger.error(f"[Propagate] Error for subscriber strategy {sub_strategy.id}: {_e}", exc_info=True)
        finally:
            _sub_db.close()


# ─── Asset-class helper (shared by both executor loops) ──────────────────────

def _snap_asset_class(snap: dict) -> str:
    """Return the normalised asset_class for a strategy snapshot dict.
    Mobile wizard saves asset_class as '_asset_class' in config; web portal
    uses 'asset_class'. Both are checked so mobile-built forex/index strategies
    are routed to the correct executor."""
    _obj = snap.get("_obj")
    _cfg = snap.get("config") or {}
    return (
        (getattr(_obj, "asset_class", None) or "").strip()
        or _cfg.get("asset_class")
        or _cfg.get("_asset_class")
        or "crypto"
    )


# ─── Main executor loop (crypto only) ────────────────────────────────────────

async def run_session_alert_loop():
    """
    Background loop — every 60 s fires a push notification to users who have
    an active/paper forex strategy with a session filter matching a session that
    opens within the next ALERT_MINUTES_BEFORE minutes (UTC).

    Dedup: at most one alert per (user_id, session_id, UTC date).
    """
    from app.database import BgSessionLocal
    from app.strategy_models import UserStrategy

    ALERT_MINUTES_BEFORE = 10
    logger.info("⏰ Session alert loop started (60s interval, %d-min lead time)", ALERT_MINUTES_BEFORE)

    while True:
        await asyncio.sleep(60)
        try:
            now       = datetime.utcnow()
            today_str = now.strftime("%Y-%m-%d")

            # Find sessions opening in the next ALERT_MINUTES_BEFORE minutes
            upcoming: list = []
            for sess_id, label, h, m in _SESSION_ALERT_WINDOWS:
                session_start = now.replace(hour=h, minute=m, second=0, microsecond=0)
                diff_min = (session_start - now).total_seconds() / 60
                if 0 < diff_min <= ALERT_MINUTES_BEFORE:
                    upcoming.append((sess_id, label, int(diff_min)))

            # Prune old-date entries so the set doesn't grow unbounded
            stale = {k for k in _SESSION_ALERT_SENT if k[2] != today_str}
            _SESSION_ALERT_SENT.difference_update(stale)

            if not upcoming:
                continue

            db = BgSessionLocal()
            try:
                strategies = (
                    db.query(UserStrategy)
                    .filter(UserStrategy.status.in_(["active", "paper"]))
                    .all()
                )

                for strat in strategies:
                    cfg = strat.config or {}
                    # Mobile wizard saves "_asset_class"; web portal saves "asset_class" — check both
                    _strat_ac = cfg.get("asset_class") or cfg.get("_asset_class") or ""
                    if _strat_ac not in ("forex",):
                        continue

                    filters         = cfg.get("filters", {})
                    sess_filter     = filters.get("session", {})
                    selected_sids   = [s.lower() for s in sess_filter.get("sessions", [])] if sess_filter else []
                    if not selected_sids:
                        continue

                    uni      = cfg.get("universe", {})
                    syms     = uni.get("symbols", [])
                    pair_str = ", ".join(syms[:2]) if syms else "your pairs"

                    for sess_id, label, mins_left in upcoming:
                        if sess_id not in selected_sids:
                            continue
                        alert_key = (strat.user_id, sess_id, today_str)
                        if alert_key in _SESSION_ALERT_SENT:
                            continue
                        _SESSION_ALERT_SENT.add(alert_key)

                        # Push notification (mobile)
                        from app.services.expo_push import notify_session_alert_bg
                        notify_session_alert_bg(
                            strat.user_id,
                            label,
                            mins_left,
                            strat.name,
                            pair_str,
                            strategy_id=strat.id,
                        )

                        # Telegram DM — fetch user and send if they have a telegram_id
                        try:
                            from app.models import User as _AlertUser
                            _alert_user = db.query(_AlertUser).filter(
                                _AlertUser.id == strat.user_id
                            ).first()
                            _tg_alert_id = _telegram_int_id(_alert_user) if _alert_user else None
                            if _tg_alert_id:
                                asyncio.create_task(_tg_send(
                                    _tg_alert_id,
                                    f"⏰ <b>{label}</b> opens in {mins_left} min\n"
                                    f"📊 <b>{strat.name}</b> is active on {pair_str}\n"
                                    f"<i>Your strategy is live and watching for entries.</i>",
                                ))
                        except Exception as _tg_err:
                            logger.debug("Session alert Telegram DM failed: %s", _tg_err)

                        logger.debug(
                            "Session alert → user=%s strat=%s session=%s",
                            strat.user_id, strat.id, sess_id,
                        )
            finally:
                db.close()

        except Exception as exc:
            logger.warning("Session alert loop error: %s", exc)


async def run_strategy_executor():
    """
    Main background loop. Evaluates all active + paper strategies for all users.
    Also spawns the paper position monitor as a sibling task.
    """
    from app.database import BgSessionLocal as SessionLocal, bg_engine as engine
    from app.models import User
    from app.strategy_models import UserStrategy, init_strategy_tables

    init_strategy_tables(engine)
    logger.info("🤖 Strategy executor started (active + paper modes)")

    # Spawn paper monitor and session alert loop as concurrent sibling tasks
    asyncio.create_task(run_paper_position_monitor())
    asyncio.create_task(run_session_alert_loop())

    sem = asyncio.Semaphore(MAX_CONCURRENT)

    # Explicit timeouts: 15s total, 5s connect, 8s pool-wait — prevents hung
    # requests from blocking semaphore slots indefinitely. Pool wait is
    # explicit so PoolTimeout fails fast rather than blocking the full 15s.
    _timeout = httpx.Timeout(15.0, connect=5.0, pool=8.0)

    # Connection pool limits — defaults (max=100, keepalive=20) were
    # exhausted under load (47 strategies × 5-concurrent × per-symbol
    # kline + ticker queries), causing PoolTimeout cascades that knocked
    # out the MEXC ticker refresh and cleared every strategy's universe.
    # 500 connections + 100 keepalive comfortably handles peak bursts.
    _limits = httpx.Limits(
        max_connections=500,
        max_keepalive_connections=100,
        keepalive_expiry=30.0,
    )

    async with httpx.AsyncClient(timeout=_timeout, limits=_limits) as http_client:
        # ── Pre-warm the Bitunix symbol cache ─────────────────────────────────
        # Fetch the tradeable symbol list before the first evaluation cycle so
        # the fail-closed guard never blocks trades on a valid coin due to a
        # cold-start race condition. Retry up to 5 times with a 3s backoff.
        for _attempt in range(5):
            _syms = await _get_bitunix_symbols(http_client)
            if _syms:
                logger.info(
                    f"✅ Bitunix symbol cache warmed: {len(_syms)} USDT perps "
                    f"available for strategy evaluation"
                )
                break
            logger.warning(
                f"⚠️  Bitunix symbol cache empty (attempt {_attempt + 1}/5) — "
                f"retrying in 3s..."
            )
            await asyncio.sleep(3)
        else:
            logger.error(
                "❌ Could not load Bitunix symbol list after 5 attempts — "
                "executor will retry each cycle; trades will not fire until "
                "the list is available (fail-closed)"
            )

        while True:
            try:
                # Load strategy list with a short-lived session — close it
                # immediately so no stale transaction lingers during evaluation.
                _list_db = SessionLocal()
                try:
                    strategies = (
                        _list_db.query(UserStrategy)
                        .filter(UserStrategy.status.in_(["active", "paper"]))
                        .all()
                    )
                    # Detach objects so they're accessible after the session closes
                    strategy_snapshots = [
                        {
                            "id":          s.id,
                            "name":        s.name,
                            "status":      s.status,
                            "config":      s.config,
                            "user_id":     s.user_id,
                            "_obj":        s,
                        }
                        for s in strategies
                    ]
                finally:
                    _list_db.close()

                if not strategy_snapshots:
                    await asyncio.sleep(SCAN_INTERVAL_SECONDS)
                    continue

                active_count = sum(1 for s in strategy_snapshots if s["status"] == "active")
                paper_count  = sum(1 for s in strategy_snapshots if s["status"] == "paper")
                # Asset-class breakdown so we can confirm forex/index strategies are loaded
                _ac_counts: Dict[str, int] = {}
                for _s in strategy_snapshots:
                    _obj = _s.get("_obj")
                    _ac = (
                        (getattr(_obj, "asset_class", None) or "").strip()
                        or (_s.get("config") or {}).get("asset_class")
                        or "crypto"
                    )
                    _ac_counts[_ac] = _ac_counts.get(_ac, 0) + 1
                _ac_str = " ".join(f"{k}={v}" for k, v in sorted(_ac_counts.items()))
                logger.info(
                    f"🤖 Strategy executor: {active_count} live · {paper_count} paper [{_ac_str}]"
                )

                # Locked subscriber copies MUST NOT evaluate independently —
                # they are triggered by _propagate_to_subscribers when the source fires,
                # guaranteeing identical entry/TP/SL prices for all copyholders.
                # Exception: if the source strategy is NOT in the active/paper pool
                # (deleted, paused, owner lost Pro, etc.) we fall back to independent
                # evaluation so subscribers still receive signals.
                active_source_ids = {
                    s["id"] for s in strategy_snapshots
                    if not (s["config"] or {}).get("_locked")
                }
                eval_snapshots = [
                    s for s in strategy_snapshots
                    if not (
                        (s["config"] or {}).get("_locked")
                        and (s["config"] or {}).get("_source_strategy_id") in active_source_ids
                    )
                    and (s["config"] or {}).get("entry_conditions", {}).get("entry_type") != "tradingview_webhook"
                    # Forex / index / stock handled by run_forex_executor (dedicated loop)
                    and _snap_asset_class(s) not in ("forex", "index", "stock")
                ]
                skipped = len(strategy_snapshots) - len(eval_snapshots)
                if skipped:
                    logger.debug(
                        f"[Executor] Skipping {skipped} locked subscriber copies "
                        f"(will be triggered by source propagation)"
                    )

                # Pre-fetch tickers ONCE for the entire cycle.
                shared_tickers = await _get_raw_tickers(http_client)

                # Per-cycle gate diagnostics — counts which gate blocks each strategy.
                # Logged at end of cycle so we can see WHY strategies aren't firing.
                cycle_gate_stats: Dict[str, int] = {}

                async def _run_one(snap, _tickers=shared_tickers):
                    """Each strategy evaluation runs in its own isolated DB session.

                    Retries once with a fresh session on transient Neon errors
                    (SSL connection drops or PK-lookup statement timeouts that
                    occasionally occur during Neon compute scaling/autovacuum).
                    Permanent errors are logged with traceback as before.
                    """
                    from sqlalchemy.exc import OperationalError as _SAOperationalError
                    async with sem:
                        for _attempt in (1, 2):
                            db_one = SessionLocal()
                            try:
                                strategy = db_one.query(UserStrategy).filter(
                                    UserStrategy.id == snap["id"]
                                ).first()
                                if not strategy:
                                    return
                                user = db_one.query(User).filter(
                                    User.id == snap["user_id"]
                                ).first()
                                if not user or user.banned:
                                    return
                                # Only Pro portal subscribers (or admin/grandfathered) get trades
                                _now = datetime.utcnow()
                                _has_portal_pro = False
                                try:
                                    from app.strategy_models import PortalSubscription
                                    _psub = db_one.query(PortalSubscription).filter_by(
                                        user_id=user.id
                                    ).first()
                                    _has_portal_pro = (
                                        _psub and _psub.tier == "pro"
                                        and _psub.subscription_end
                                        and _psub.subscription_end > _now
                                    )
                                except Exception:
                                    pass
                                if not (
                                    user.is_admin
                                    or user.grandfathered
                                    or _has_portal_pro
                                ):
                                    return
                                await evaluate_and_fire(
                                    strategy, user, db_one, http_client,
                                    raw_tickers=_tickers,
                                    gate_stats=cycle_gate_stats,
                                )
                                return  # success
                            except _SAOperationalError as _db_err:
                                # Most likely a transient Neon SSL drop or a
                                # PK-lookup statement timeout. Discard this
                                # session and retry once with a fresh one.
                                _err_name = type(_db_err.orig).__name__ if getattr(_db_err, "orig", None) else type(_db_err).__name__
                                if _attempt == 1:
                                    logger.warning(
                                        f"[Strategy {snap['id']}] Transient DB error "
                                        f"({_err_name}) — retrying with fresh session"
                                    )
                                    try:
                                        db_one.rollback()
                                    except Exception:
                                        pass
                                    continue
                                else:
                                    logger.warning(
                                        f"[Strategy {snap['id']}] Skipping cycle — "
                                        f"DB error persisted after retry ({_err_name})"
                                    )
                                    return
                            except Exception as e:
                                logger.error(
                                    f"[Strategy {snap['id']}] Error: {e}", exc_info=True
                                )
                                return
                            finally:
                                try:
                                    db_one.close()
                                except Exception:
                                    pass

                await asyncio.gather(*[_run_one(s) for s in eval_snapshots])

                # Cycle gate diagnostics — shows exactly which gate blocked each strategy.
                # Helps diagnose "why aren't trades firing?" without spelunking logs.
                if cycle_gate_stats:
                    _gate_summary = " ".join(
                        f"{k.replace('blk_', '')}={v}"
                        for k, v in sorted(cycle_gate_stats.items(), key=lambda kv: -kv[1])
                    )
                    logger.info(f"[Executor] cycle gates → {_gate_summary}")

            except Exception as e:
                logger.error(f"Strategy executor loop error: {e}", exc_info=True)

            await asyncio.sleep(SCAN_INTERVAL_SECONDS)


# ─── Dedicated forex / index / stock executor loop ───────────────────────────

async def run_forex_executor():
    """
    Dedicated scan loop for forex, index, and stock strategies.

    Runs on a shorter independent cycle (EXECUTOR_FOREX_SCAN_INTERVAL, default
    30 s) so these strategies are not forced to wait for the full crypto cycle
    (which can take 30-60 s due to MEXC/Bitunix API calls).  Forex price data
    comes from yfinance, so no MEXC ticker pre-fetch is needed and no Bitunix
    symbol pre-warm is required.
    """
    from app.database import BgSessionLocal as SessionLocal, bg_engine as engine
    from app.models import User
    from app.strategy_models import UserStrategy, init_strategy_tables

    init_strategy_tables(engine)
    logger.info(
        f"📈 Forex/index executor started (cycle={FOREX_SCAN_INTERVAL_SECONDS}s)"
    )

    _TRADFI_CLASSES = {"forex", "index", "stock"}

    sem = asyncio.Semaphore(MAX_CONCURRENT)

    # Lighter pool — yfinance calls are outbound Python HTTP, not MEXC REST;
    # a handful of concurrent connections is plenty.
    _timeout = httpx.Timeout(20.0, connect=5.0, pool=8.0)
    _limits  = httpx.Limits(
        max_connections=50,
        max_keepalive_connections=10,
        keepalive_expiry=30.0,
    )

    async with httpx.AsyncClient(timeout=_timeout, limits=_limits) as http_client:
        while True:
            try:
                _list_db = SessionLocal()
                try:
                    strategies = (
                        _list_db.query(UserStrategy)
                        .filter(UserStrategy.status.in_(["active", "paper"]))
                        .all()
                    )
                    strategy_snapshots = [
                        {
                            "id":      s.id,
                            "name":    s.name,
                            "status":  s.status,
                            "config":  s.config,
                            "user_id": s.user_id,
                            "_obj":    s,
                        }
                        for s in strategies
                        if _snap_asset_class({
                            "_obj":   s,
                            "config": s.config,
                        }) in _TRADFI_CLASSES
                    ]
                finally:
                    _list_db.close()

                if not strategy_snapshots:
                    await asyncio.sleep(FOREX_SCAN_INTERVAL_SECONDS)
                    continue

                active_source_ids = {
                    s["id"] for s in strategy_snapshots
                    if not (s["config"] or {}).get("_locked")
                }
                eval_snapshots = [
                    s for s in strategy_snapshots
                    if not (
                        (s["config"] or {}).get("_locked")
                        and (s["config"] or {}).get("_source_strategy_id") in active_source_ids
                    )
                    and (s["config"] or {}).get("entry_conditions", {}).get("entry_type") != "tradingview_webhook"
                ]

                logger.info(
                    f"📈 Forex executor: scanning {len(eval_snapshots)} tradfi strateg"
                    f"{'y' if len(eval_snapshots) == 1 else 'ies'}"
                )

                # No MEXC ticker prefetch — forex uses yfinance; pass empty list.
                cycle_gate_stats: Dict[str, int] = {}

                # Shared counters for cycle-level DB health reporting.
                _cycle_db_skipped: list = []   # (strategy_id, err_name) tuples

                async def _run_one_fx(snap, _http=http_client):
                    from sqlalchemy.exc import OperationalError as _SAOperationalError
                    async with sem:
                        for _attempt in (1, 2):
                            db_one = SessionLocal()
                            try:
                                strategy = db_one.query(UserStrategy).filter(
                                    UserStrategy.id == snap["id"]
                                ).first()
                                if not strategy:
                                    return
                                user = db_one.query(User).filter(
                                    User.id == snap["user_id"]
                                ).first()
                                if not user or user.banned:
                                    return
                                _now = datetime.utcnow()
                                _has_portal_pro = False
                                try:
                                    from app.strategy_models import PortalSubscription
                                    _psub = db_one.query(PortalSubscription).filter_by(
                                        user_id=user.id
                                    ).first()
                                    _has_portal_pro = (
                                        _psub and _psub.tier == "pro"
                                        and _psub.subscription_end
                                        and _psub.subscription_end > _now
                                    )
                                except Exception:
                                    pass
                                if not (
                                    user.is_admin
                                    or user.grandfathered
                                    or _has_portal_pro
                                ):
                                    return
                                await evaluate_and_fire(
                                    strategy, user, db_one, _http,
                                    raw_tickers=[],
                                    gate_stats=cycle_gate_stats,
                                )
                                return
                            except _SAOperationalError as _db_err:
                                _err_name = type(_db_err.orig).__name__ if getattr(_db_err, "orig", None) else type(_db_err).__name__
                                if _attempt == 1:
                                    # Quiet retry — cycle summary logged after gather
                                    logger.debug(
                                        f"[FX Strategy {snap['id']}] DB error "
                                        f"({_err_name}) — retrying"
                                    )
                                    try:
                                        db_one.rollback()
                                    except Exception:
                                        pass
                                    continue
                                else:
                                    # Both attempts failed — record for cycle summary
                                    _cycle_db_skipped.append((snap["id"], _err_name))
                                    logger.debug(
                                        f"[FX Strategy {snap['id']}] Skipping — "
                                        f"DB error persisted ({_err_name})"
                                    )
                                    return
                            except Exception as e:
                                logger.error(
                                    f"[FX Strategy {snap['id']}] Error: {e}", exc_info=True
                                )
                                return
                            finally:
                                try:
                                    db_one.close()
                                except Exception:
                                    pass

                await asyncio.gather(*[_run_one_fx(s) for s in eval_snapshots])

                # Emit one consolidated warning per cycle instead of per-strategy spam
                if _cycle_db_skipped:
                    _total = len(eval_snapshots)
                    _skipped = len(_cycle_db_skipped)
                    _err_types = ", ".join(sorted({e for _, e in _cycle_db_skipped}))
                    logger.warning(
                        f"[FX Executor] DB unreachable — skipped {_skipped}/{_total} "
                        f"strategies this cycle ({_err_types}). Will retry next cycle."
                    )

                if cycle_gate_stats:
                    _gate_summary = " ".join(
                        f"{k.replace('blk_', '')}={v}"
                        for k, v in sorted(cycle_gate_stats.items(), key=lambda kv: -kv[1])
                    )
                    logger.info(f"[FX Executor] cycle gates → {_gate_summary}")

            except Exception as e:
                logger.error(f"Forex executor loop error: {e}", exc_info=True)

            await asyncio.sleep(FOREX_SCAN_INTERVAL_SECONDS)


async def backfill_cancelled_paper_trades(lookback_days: int = 30) -> int:
    """
    One-time fix: re-evaluate paper trades incorrectly marked CANCELLED because
    MEXC candle fetches were missing the startTime parameter.
    Returns the number of trades corrected.
    """
    from app.database import BgSessionLocal as SessionLocal
    from app.strategy_models import StrategyExecution
    cutoff = datetime.utcnow() - timedelta(days=lookback_days)

    db = SessionLocal()
    try:
        cancelled = (
            db.query(StrategyExecution)
            .filter(
                StrategyExecution.outcome == "CANCELLED",
                StrategyExecution.is_paper == True,
                StrategyExecution.fired_at >= cutoff,
                StrategyExecution.entry_price.isnot(None),
                StrategyExecution.tp_price.isnot(None),
                StrategyExecution.sl_price.isnot(None),
            )
            .all()
        )
    except Exception as e:
        logger.error(f"backfill_cancelled: query failed: {e}")
        db.close()
        return 0

    if not cancelled:
        logger.info("backfill_cancelled: no CANCELLED paper trades found to review")
        db.close()
        return 0

    logger.info(f"backfill_cancelled: reviewing {len(cancelled)} CANCELLED paper trade(s)")
    corrected = 0

    async with httpx.AsyncClient() as client:
        for ex in cancelled:
            try:
                # Temporarily reset outcome to OPEN so _evaluate can close it properly
                ex.outcome = "OPEN"
                ex.closed_at = None
                _ac = (getattr(ex, "asset_class", None) or "crypto")
                candles = await _fetch_candles_since_entry(ex.symbol, ex.fired_at, client, _ac)
                if not candles:
                    ex.outcome = "CANCELLED"  # restore
                    db.commit()
                    continue
                before = ex.outcome
                result = _evaluate_paper_position_against_candles(ex, candles, db)
                if result and ex.outcome != "CANCELLED":
                    logger.info(
                        f"backfill_cancelled: #{ex.id} {ex.symbol} {ex.direction} "
                        f"corrected → {ex.outcome} ({ex.pnl_pct:+.2f}%)"
                    )
                    corrected += 1
                else:
                    # No TP/SL found even with correct candles — restore CANCELLED
                    ex.outcome = "CANCELLED"
                    ex.closed_at = ex.closed_at or datetime.utcnow()
                    db.commit()
            except Exception as e:
                logger.warning(f"backfill_cancelled: error on #{ex.id}: {e}")
                try:
                    ex.outcome = "CANCELLED"
                    db.commit()
                except Exception:
                    db.rollback()
                continue

    db.close()
    logger.info(f"backfill_cancelled: done — {corrected}/{len(cancelled)} trade(s) corrected")
    return corrected


async def backfill_ghost_cancelled_executions(lookback_days: int = 7) -> int:
    """
    Recover live executions that were incorrectly cancelled by the ghost-cleanup
    job due to a race condition (bitunix_order_id not yet written when cleanup ran).

    Affected records have:
      is_paper = false, outcome = 'CANCELLED',
      notes like '%ghost execution%'

    Fix: convert to is_paper=True, re-evaluate against historical candles.
    If the position already closed (TP/SL hit), record the correct outcome.
    If still within hold window (trade might still be open), mark OPEN.
    If candles unavailable, mark BREAKEVEN so it doesn't pollute win-rate.
    """
    from app.database import BgSessionLocal as SessionLocal
    from app.strategy_models import StrategyExecution
    cutoff = datetime.utcnow() - timedelta(days=lookback_days)

    db = SessionLocal()
    try:
        ghosts = (
            db.query(StrategyExecution)
            .filter(
                StrategyExecution.outcome == "CANCELLED",
                StrategyExecution.is_paper == False,
                StrategyExecution.fired_at >= cutoff,
                StrategyExecution.entry_price.isnot(None),
                StrategyExecution.tp_price.isnot(None),
                StrategyExecution.sl_price.isnot(None),
                StrategyExecution.notes.like("%ghost execution%"),
            )
            .all()
        )
    except Exception as e:
        logger.error(f"backfill_ghosts: query failed: {e}")
        db.close()
        return 0

    if not ghosts:
        logger.info("backfill_ghosts: no ghost-cancelled live executions found")
        db.close()
        return 0

    logger.info(f"backfill_ghosts: recovering {len(ghosts)} ghost-cancelled execution(s)")
    corrected = 0

    async with httpx.AsyncClient() as client:
        for ex in ghosts:
            try:
                # Convert to paper so performance is still tracked
                ex.is_paper = True
                ex.outcome = "OPEN"
                ex.closed_at = None
                ex.notes = (ex.notes or "") + " | recovered-as-paper"
                db.commit()

                _ac = (getattr(ex, "asset_class", None) or "crypto")
                candles = await _fetch_candles_since_entry(ex.symbol, ex.fired_at, client, _ac)
                if not candles:
                    # No candle data — record as BREAKEVEN so it doesn't skew win rate
                    ex.outcome = "BREAKEVEN"
                    ex.pnl_pct = 0.0
                    ex.closed_at = datetime.utcnow()
                    db.commit()
                    logger.info(f"backfill_ghosts: #{ex.id} {ex.symbol} — no candles, marked BREAKEVEN")
                    corrected += 1
                    continue

                result = _evaluate_paper_position_against_candles(ex, candles, db)
                if result and ex.outcome not in ("CANCELLED", "OPEN"):
                    logger.info(
                        f"backfill_ghosts: #{ex.id} {ex.symbol} {ex.direction} "
                        f"→ {ex.outcome} ({ex.pnl_pct:+.2f}%)"
                    )
                    corrected += 1
                elif ex.outcome == "OPEN":
                    # Trade would still be running — leave as OPEN paper trade
                    logger.info(f"backfill_ghosts: #{ex.id} {ex.symbol} — still open, left as paper OPEN")
                    corrected += 1
                else:
                    ex.outcome = "BREAKEVEN"
                    ex.pnl_pct = 0.0
                    ex.closed_at = datetime.utcnow()
                    db.commit()
                    logger.info(f"backfill_ghosts: #{ex.id} {ex.symbol} — undetermined, marked BREAKEVEN")
                    corrected += 1

            except Exception as e:
                logger.warning(f"backfill_ghosts: error on #{ex.id}: {e}")
                try:
                    ex.outcome = "BREAKEVEN"
                    ex.pnl_pct = 0.0
                    ex.is_paper = True
                    ex.closed_at = datetime.utcnow()
                    db.commit()
                except Exception:
                    db.rollback()
                continue

    db.close()
    logger.info(f"backfill_ghosts: done — {corrected}/{len(ghosts)} execution(s) recovered")
    return corrected


async def close_stale_open_executions(stale_after_hours: int = 48) -> int:
    """
    Close any strategy_executions rows that have been stuck OPEN for longer than
    `stale_after_hours` hours.  These accumulate when a position is opened but
    the monitor loop never sees it (server restart, DB hiccup, price feed gap).
    Leaving them as OPEN permanently blocks the max_open_positions gate, so all
    strategies with a stale ghost trade silently stop firing.

    Action: mark outcome='EXPIRED', set closed_at = fired_at + stale_after_hours,
    exit_price = entry_price, pnl_pct = 0 (no win/loss attribution).
    Runs once at startup before the scan loops begin.
    """
    from app.database import BgSessionLocal as SessionLocal
    from app.strategy_models import StrategyExecution
    cutoff = datetime.utcnow() - timedelta(hours=stale_after_hours)
    db = SessionLocal()
    try:
        stale = (
            db.query(StrategyExecution)
            .filter(
                StrategyExecution.outcome == "OPEN",
                StrategyExecution.closed_at.is_(None),
                StrategyExecution.fired_at <= cutoff,
            )
            .all()
        )
        if not stale:
            logger.info("close_stale_open_executions: no stale positions found")
            return 0
        logger.warning(
            f"close_stale_open_executions: closing {len(stale)} position(s) "
            f"stuck OPEN >{stale_after_hours}h — these were blocking the max_open gate"
        )
        count = 0
        for ex in stale:
            try:
                ex.outcome    = "EXPIRED"
                ex.closed_at  = ex.fired_at + timedelta(hours=stale_after_hours)
                ex.exit_price = ex.entry_price
                ex.pnl_pct    = 0.0
                ex.notes      = (ex.notes or "") + " | auto-expired: stuck open > 48h"
                db.commit()
                count += 1
            except Exception as e:
                db.rollback()
                logger.error(f"close_stale_open_executions: failed for exec {ex.id}: {e}")
        logger.info(f"close_stale_open_executions: expired {count}/{len(stale)} stale position(s)")
        return count
    except Exception as e:
        logger.error(f"close_stale_open_executions: query failed: {e}")
        return 0
    finally:
        db.close()
