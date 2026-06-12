"""
Strategy Executor — Build Your Own Strategy Portal

Background worker that continuously evaluates active + paper user strategies
and fires trades. Paper trades are tracked with 1m OHLC accuracy — candle
high/low is used to detect TP/SL hits so scalp results are realistic.
"""
import asyncio
import html as _html
import logging
import math
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import httpx

logger = logging.getLogger(__name__)

import os as _os_env


def _log_ts() -> str:
    """UTC timestamp prefix for trade / executor logs (Railway-visible)."""
    return datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")

# Env-overridable so we can dial pressure on prod without a redeploy.
# Defaults raised from 3→10s and 5→3 because Neon's compute was getting
# saturated — every QueryCanceled in the executor cascaded into HTTP 500s
# on /api/strategies because all pool connections were held by hung scans.
SCAN_INTERVAL_SECONDS       = int(_os_env.environ.get("EXECUTOR_SCAN_INTERVAL", "10"))
# Crypto scan cadence — decouple from forex so portal can scan crypto every ~10 min.
CRYPTO_SCAN_INTERVAL_SECONDS = int(
    _os_env.environ.get(
        "EXECUTOR_CRYPTO_SCAN_INTERVAL",
        _os_env.environ.get("EXECUTOR_SCAN_INTERVAL", "10"),
    )
)
# Forex runs at 5s (vs crypto 10s): the forex path fetches fresh OHLC every cycle
# (it does NOT use the 45s cross-cycle kline cache), so a tighter loop directly
# lowers signal-to-order latency. Safe because peak DB concurrency stays capped at
# MAX_CONCURRENT and the post-work fixed sleep self-throttles when scans are slow.
# Do NOT drop below ~5s or remove the fixed post-work sleep — at 3s the executor
# previously saturated Neon and cascaded QueryCanceled → HTTP 500s on /api/strategies.
FOREX_SCAN_INTERVAL_SECONDS = int(_os_env.environ.get("EXECUTOR_FOREX_SCAN_INTERVAL", "5"))
# Live forex SL management (breakeven/trailing) runs on its OWN fast loop so fast
# instruments like gold react in well under a second. It reads prices from the
# cTrader real-time spot feed (per-tick) — NOT the DB — so a 1s cadence is cheap:
# the open-position worklist is rebuilt from the DB only every _FX_WORKLIST_TTL.
FOREX_MANAGE_INTERVAL_SECONDS = float(_os_env.environ.get("EXECUTOR_FOREX_MANAGE_INTERVAL", "1"))
PAPER_MONITOR_INTERVAL      = int(_os_env.environ.get("EXECUTOR_MONITOR_INTERVAL", "10"))
LIVE_MONITOR_INTERVAL       = int(_os_env.environ.get("EXECUTOR_LIVE_MONITOR_INTERVAL", "8"))
MAX_CONCURRENT              = int(_os_env.environ.get("EXECUTOR_MAX_CONCURRENT", "2"))
# Each forex eval holds bg_engine across async kline/TA fetches. Total checkout
# slots are capped by app.database.bg_db_slot() (pool hard limit − reserve).
FOREX_MAX_CONCURRENT        = int(_os_env.environ.get("EXECUTOR_FOREX_MAX_CONCURRENT", "3"))
# Evaluate strategies in batches so Railway logs show progress during long first cycles
# (90 forex + 168 crypto can run 5–15 min with no other INFO lines).
EXECUTOR_SCAN_BATCH_SIZE    = int(_os_env.environ.get("EXECUTOR_SCAN_BATCH_SIZE", "20"))
# Let forex finish its first batch before crypto hammers the same Neon pool + APIs.
EXECUTOR_CRYPTO_START_DELAY = int(_os_env.environ.get("EXECUTOR_CRYPTO_START_DELAY", "45"))
# Klines fetched per symbol during scans — must match strategy_ta max(limit, 200).
EXECUTOR_KLINE_BARS           = int(_os_env.environ.get("EXECUTOR_KLINE_BARS", "200"))
# Parallel Yahoo/Bitunix prefetches at cycle start (unique symbols across all strategies).
EXECUTOR_PREFETCH_CONCURRENT  = int(_os_env.environ.get("EXECUTOR_PREFETCH_CONCURRENT", "10"))
# Split strategies across N parallel scan loops in one executor process (id % N).
# Each shard prefetches + evaluates its slice concurrently — ~Nx throughput vs one loop.
EXECUTOR_SHARD_COUNT          = max(1, int(_os_env.environ.get("EXECUTOR_SHARD_COUNT", "1")))
EXECUTOR_SHARD_STAGGER_SECONDS = int(_os_env.environ.get("EXECUTOR_SHARD_STAGGER_SECONDS", "2"))
PAPER_MAX_HOLD_HOURS        = 168   # auto-expire paper positions after this many hours (7 days)


def _env_flag(name: str) -> bool:
    return _os_env.environ.get(name, "").lower() in ("1", "true", "yes")


def crypto_executor_disabled() -> bool:
    return _env_flag("DISABLE_CRYPTO_EXECUTOR")


def forex_executor_disabled() -> bool:
    return _env_flag("DISABLE_FOREX_EXECUTOR")


def executor_runtime_profile() -> Dict[str, object]:
    """Summarise split-executor flags for logs and /health/deep."""
    return {
        "executor_only": _env_flag("EXECUTOR_ONLY"),
        "crypto_disabled": crypto_executor_disabled(),
        "forex_disabled": forex_executor_disabled(),
        "crypto_scan_interval_s": CRYPTO_SCAN_INTERVAL_SECONDS,
        "forex_scan_interval_s": FOREX_SCAN_INTERVAL_SECONDS,
        "forex_manage_interval_s": FOREX_MANAGE_INTERVAL_SECONDS,
        "forex_worklist_ttl_s": float(
            _os_env.environ.get("EXECUTOR_FX_WORKLIST_TTL", "1")
        ),
        "forex_reconcile_interval_s": float(
            _os_env.environ.get("EXECUTOR_FX_RECONCILE_INTERVAL", "5")
        ),
    }


def strategy_shard_index(strategy_id: int, shard_count: int = EXECUTOR_SHARD_COUNT) -> int:
    return int(strategy_id) % max(1, shard_count)


def strategy_on_shard(
    strategy_id: int,
    shard_index: int,
    shard_count: int = EXECUTOR_SHARD_COUNT,
) -> bool:
    return strategy_shard_index(strategy_id, shard_count) == shard_index


def _executor_shard_label(base: str, shard_index: int, shard_count: int) -> str:
    if shard_count <= 1:
        return base
    return f"{base} S{shard_index}/{shard_count}"

async def _gather_eval_batches(label: str, snapshots: list, run_one, batch_size: int = 0) -> None:
    """Run evaluate tasks in chunks and log progress (visible in Railway during long scans)."""
    size = batch_size or EXECUTOR_SCAN_BATCH_SIZE
    total = len(snapshots)
    if not total:
        return
    n_batches = (total + size - 1) // max(1, size)
    done = 0
    done_lock = asyncio.Lock()
    # Log every N completions so long first cycles (78 forex × Yahoo klines)
    # don't look stuck for 5–10 min before the first batch finishes.
    _PROGRESS_EVERY = max(5, min(10, size // 2 or 5))

    async def _one(snap):
        nonlocal done
        await run_one(snap)
        async with done_lock:
            done += 1
            if done == total or done % _PROGRESS_EVERY == 0:
                logger.info(f"[{label}] progress {done}/{total} strategies evaluated")

    for bi, i in enumerate(range(0, total, max(1, size))):
        batch = snapshots[i : i + size]
        logger.info(
            f"[{label}] batch {bi + 1}/{n_batches} starting — "
            f"{len(batch)} strategies ({done}/{total} done so far)"
        )
        await asyncio.gather(*[_one(s) for s in batch])


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
_METAL_WARM_LOCK: Optional[asyncio.Lock] = None  # one metal warm at a time / process
_METAL_WARM_GLOBAL_AT: float = 0.0  # skip repeat warm across FX shards
_METAL_DRIFT_SKIP_STREAK: Dict[str, int] = {}
_METAL_DRIFT_SKIP_WARN_CYCLES = int(
    _os_env.environ.get("METAL_DRIFT_SKIP_WARN_CYCLES", "5")
)
# Active strategy rows — refreshed at most once per TTL (configs change rarely).
_STRATEGY_SNAPSHOT_CACHE: List[dict] = []
_STRATEGY_SNAPSHOT_AT: float = 0.0
_STRATEGY_SNAPSHOT_TTL_S = float(_os_env.environ.get("EXECUTOR_STRATEGY_CACHE_TTL_S", "60"))
_PRICE_TA_TTL    = 15  # seconds — fresher data for faster signal detection
# Forex/index: the FMP price feed refreshes every 5s, so a 15s cache would make
# a faster scan loop pointless (it'd re-evaluate stale prices). Match the cache
# to the feed cadence. Cheap because the slow-moving 15m klines have their own
# 20s cache in tradfi_prices — only the in-memory FMP spot price is re-read.
_PRICE_TA_TTL_TRADFI = 5

# Opt-in verbose per-strategy TA logging. Set EXECUTOR_VERBOSE_TA=1 (Railway env)
# to watch each saved strategy being evaluated — symbol + which entry conditions
# pass/fail each scan — without ever spamming logs by default. Throttled per
# (strategy, symbol) so a 90-strategy forex cycle stays readable; a real fire is
# always logged regardless of the throttle.
_TA_VERBOSE_LAST: Dict[tuple, float] = {}
_TA_VERBOSE_THROTTLE_S = 30.0


def _maybe_log_ta_eval(strategy, symbol, direction, passed, details) -> None:
    if _os_env.environ.get("EXECUTOR_VERBOSE_TA", "").lower() not in ("1", "true", "yes"):
        return
    key = (getattr(strategy, "id", 0), symbol)
    now = time.monotonic()
    if not passed and (now - _TA_VERBOSE_LAST.get(key, 0.0)) < _TA_VERBOSE_THROTTLE_S:
        return  # throttle repeated "not met" lines; always log a fire
    _TA_VERBOSE_LAST[key] = now
    try:
        _det = details or []
        n_pass = sum(1 for d in _det if str(d).lstrip().startswith("✅"))
        summary = " | ".join(str(d) for d in _det[:6])
        logger.info(
            f"[TA] Strategy {getattr(strategy, 'id', '?')} "
            f"'{getattr(strategy, 'name', '') or ''}' {symbol} {direction}: "
            f"{n_pass}/{len(_det)} conditions "
            f"{'✅ MET → firing' if passed else 'not met'} :: {summary}"
        )
    except Exception:
        pass


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _portal_trade_entitled(
    user,
    has_pro_by_user: Optional[Dict[int, bool]] = None,
    db=None,
) -> bool:
    """Whether a user may receive executor-fired trades (crypto or forex)."""
    from app.deployment import portal_features_free

    if portal_features_free():
        return True
    if user.is_admin or getattr(user, "grandfathered", False):
        return True
    if has_pro_by_user is not None:
        return has_pro_by_user.get(user.id, False)
    if db is not None:
        from app.strategy_models import PortalSubscription

        _now = datetime.utcnow()
        _psub = db.query(PortalSubscription).filter_by(user_id=user.id).first()
        return bool(
            _psub
            and _psub.tier == "pro"
            and _psub.subscription_end
            and _psub.subscription_end > _now
        )
    return False


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
        from app.models import UserPreference
        from app.services.bitunix_partner import is_uid_affiliated

        prefs = db.query(UserPreference).filter_by(user_id=user.id).first()
        uid = getattr(prefs, "bitunix_uid", None) if prefs else None
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

    from app.services.binance_feed import binance_disabled, binance_http_get

    _last_err: Optional[str] = None
    from app.services.geo_block import is_domain_blocked

    ticker_urls = []
    mexc_url = "https://api.mexc.com/api/v3/ticker/24hr"
    if not is_domain_blocked(mexc_url):
        ticker_urls.append(
            (mexc_url, None, "strategy_executor._get_raw_tickers.mexc"),
        )
    if not binance_disabled():
        ticker_urls.append(
            ("https://fapi.binance.com/fapi/v1/ticker/24hr", None, "strategy_executor._get_raw_tickers.binance"),
        )
    for url, params, caller in ticker_urls:
        try:
            status, data = await binance_http_get(
                http_client, url, params, timeout_s=10, caller=caller,
            )
            if status in (451, 403):
                from app.services.geo_block import note_geo_block
                note_geo_block(url, status, caller)
                continue
            if status == 200 and data is not None:
                _RAW_TICKERS_CACHE = data
                _RAW_TICKERS_AT    = now
                _RAW_TICKERS_LAST_FAIL_AT = None
                logger.debug(f"Ticker cache refreshed ({len(_RAW_TICKERS_CACHE)} symbols)")
                return _RAW_TICKERS_CACHE
            if status:
                _last_err = f"{url.split('//')[1].split('/')[0]} HTTP {status}"
        except Exception as e:
            _last_err = f"{url.split('//')[1].split('/')[0]} {type(e).__name__}: {e or '(no msg)'}"
            continue

    # MEXC + Binance unreachable (e.g. geo-blocked / ConnectTimeout on Railway's
    # datacenter egress). Fall back to Bitunix — the crypto execution venue,
    # always reachable from Railway — reshaped to MEXC 24hr-ticker format so the
    # eligibility filter and price path are unchanged. Without this the universe
    # is empty and EVERY crypto strategy is skipped.
    try:
        from app.services.bitunix_market_data import fetch_tickers as _bx_tickers
        _bx = await _bx_tickers(http_client)
        if _bx:
            _RAW_TICKERS_CACHE = _bx
            _RAW_TICKERS_AT = now
            _RAW_TICKERS_LAST_FAIL_AT = None
            if _RAW_TICKERS_WARN_LAST is None or (now - _RAW_TICKERS_WARN_LAST).total_seconds() >= 60:
                _RAW_TICKERS_WARN_LAST = now
                logger.info(
                    f"Ticker list via Bitunix fallback ({len(_bx)} symbols) — "
                    f"MEXC/Binance unreachable ({_last_err or 'all sources failed'})"
                )
            return _RAW_TICKERS_CACHE
    except Exception as _bxe:
        _last_err = f"{_last_err or 'mexc/binance failed'}; bitunix: {type(_bxe).__name__}"

    # All sources failed — record + throttled warning + sticky fallback.
    _RAW_TICKERS_LAST_FAIL_AT = now
    if _RAW_TICKERS_WARN_LAST is None or (now - _RAW_TICKERS_WARN_LAST).total_seconds() >= 60:
        _RAW_TICKERS_WARN_LAST = now
        _have = len(_RAW_TICKERS_CACHE) if _RAW_TICKERS_CACHE else 0
        logger.warning(
            f"Could not refresh MEXC/Binance/Bitunix ticker list: {_last_err or 'all sources failed'}"
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


def _primary_timeframe(config: dict) -> str:
    conds = (config.get("entry_conditions") or {}).get("conditions") or []
    if conds and isinstance(conds[0], dict) and conds[0].get("timeframe"):
        return str(conds[0]["timeframe"])
    return str(config.get("timeframe") or config.get("_timeframe") or "15m")


async def _fetch_price_and_ta(
    symbol: str,
    http_client: httpx.AsyncClient,
    asset_class: str = "crypto",
    *,
    user_id: Optional[int] = None,
    timeframe: Optional[str] = None,
    metal_paper_ok: bool = False,
    prefetch: bool = False,
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
    tf = timeframe or "15m"
    cache_key = (
        f"{asset_class}:{symbol}:{tf}" if asset_class != "crypto" else symbol
    )
    if metal_paper_ok and asset_class != "crypto":
        from app.services.tradfi_prices import is_metal_symbol as _metal_ck
        if _metal_ck(symbol):
            cache_key = f"{cache_key}:paper"
    cached = _PRICE_TA_CACHE.get(cache_key)
    if cached:
        data, fetched_at = cached
        _ttl = _PRICE_TA_TTL if asset_class == "crypto" else _PRICE_TA_TTL_TRADFI
        if (now - fetched_at).total_seconds() < _ttl:
            return data

    if asset_class != "crypto":
        try:
            from app.services.prefetch_fast import SYMBOL_BUDGET_S
            from app.services.tradfi_prices import (
                get_klines as _tradfi_klines,
                get_price as _tradfi_live_price,
            )
            kl = await _tradfi_klines(
                symbol, asset_class, tf, EXECUTOR_KLINE_BARS,
                ctrader_user_id=user_id,
                max_wait_s=SYMBOL_BUDGET_S if prefetch else None,
            )
            if not kl:
                return None
            # Drop forming bar for signal evaluation (closed-candle only).
            if len(kl) > 1:
                kl = kl[:-1]
            closes = [float(row[4]) for row in kl if row and len(row) >= 5]
            if len(closes) < 2:
                return None

            from app.services.tradfi_prices import (
                get_metal_kline_source as _metal_kline_src,
                get_metal_live_for_source as _metal_live_for_src,
                is_metal_symbol as _is_metal_sym,
                metal_kline_drift_limit as _metal_drift_limit,
            )

            from app.services.tradfi_prices import get_price_fresh as _tradfi_price_fresh

            _is_metal = _is_metal_sym(symbol)
            live_px = None
            live_source = None
            bid = ask = None
            kline_close = closes[-1]
            kline_source = None
            kline_synthetic = False
            if _is_metal:
                kline_source = _metal_kline_src(symbol, tf, EXECUTOR_KLINE_BARS)
                kline_synthetic = (kline_source or "").lower() == "synthetic"
                try:
                    from app.services.ctrader_price_feed import (
                        ctrader_spot_ready as _ct_spot_ready,
                        get_bid_ask as _ba,
                    )
                    if _ct_spot_ready(symbol.upper()):
                        _tick = _ba(symbol.upper())
                        if _tick:
                            bid, ask = _tick
                            live_px = round((bid + ask) / 2.0, 6)
                            live_source = "ctrader"
                except Exception:
                    pass
                if not metal_paper_ok:
                    if not live_px or live_px <= 0:
                        try:
                            from app.services.metals_spot_feed import (
                                fetch_now as _msf_fetch,
                                get_price as _msf_price,
                            )
                            live_px = _msf_price(symbol.upper())
                            if not live_px or live_px <= 0:
                                live_px = await _msf_fetch(symbol.upper())
                            if live_px and live_px > 0:
                                live_source = kline_source or "metals_cache"
                        except Exception:
                            pass
                    if not live_px or live_px <= 0:
                        _matched = await _metal_live_for_src(
                            symbol, kline_source, user_id=user_id,
                        )
                        if _matched:
                            live_px, live_source = _matched
                    if not live_px or live_px <= 0:
                        live_px = await _tradfi_price_fresh(
                            symbol, asset_class, user_id=user_id,
                        )
                        if live_px and live_px > 0:
                            live_source = kline_source or "spot"
            else:
                try:
                    from app.services.ctrader_price_feed import get_bid_ask as _ba
                    _tick = _ba(symbol.upper())
                    if _tick:
                        bid, ask = _tick
                        live_px = round((bid + ask) / 2.0, 6)
                except Exception:
                    pass
                if not live_px:
                    live_px = await _tradfi_live_price(symbol, asset_class)

            if _is_metal:
                _spot_sources = frozenset({
                    "coinbase", "ctrader", "ctrader-user", "fmp", "kraken",
                    "yahoo_gc", "synthetic",
                })
                if not live_px or live_px <= 0:
                    _ks = (kline_source or "").lower()
                    if (
                        metal_paper_ok
                        and kline_close > 0
                        and (not _ks or _ks in _spot_sources)
                    ):
                        logger.info(
                            f"[executor] {symbol.upper()}: paper eval using kline "
                            f"close {kline_close:.2f} (src={kline_source or 'spot'}, "
                            f"no live tick)"
                        )
                        price = kline_close
                        price_source = "kline_close_paper"
                    else:
                        logger.warning(
                            f"[executor] {symbol.upper()}: no live spot price — "
                            f"skip eval (refusing kline close {kline_close:.2f})"
                        )
                        return None
                else:
                    _drift_pct = (
                        abs(live_px - kline_close) / live_px * 100.0
                        if kline_close > 0
                        else 0.0
                    )
                    _max_drift = _metal_drift_limit(kline_source)
                    _sym_u = symbol.upper()
                    _ct_live_proxy_klines = (
                        live_source == "ctrader"
                        and kline_source not in ("ctrader", "ctrader-user")
                    )
                    if _ct_live_proxy_klines:
                        _METAL_DRIFT_SKIP_STREAK[_sym_u] = 0
                        logger.info(
                            f"[prices] {_sym_u} priced_by=ctrader "
                            f"(live cTrader, klines from {kline_source or 'fallback'} "
                            f"for TA only — drift guard skipped)"
                        )
                    elif _drift_pct > _max_drift:
                        _streak = _METAL_DRIFT_SKIP_STREAK.get(_sym_u, 0) + 1
                        _METAL_DRIFT_SKIP_STREAK[_sym_u] = _streak
                        if _streak >= _METAL_DRIFT_SKIP_WARN_CYCLES:
                            logger.warning(
                                f"[executor] {_sym_u}: drift skip persisted "
                                f"{_streak} cycles — proceeding with live "
                                f"(kline={kline_close:.2f} src={kline_source or 'unknown'} "
                                f"live={live_px:.2f} src={live_source or 'unknown'} "
                                f"drift={_drift_pct:.2f}% max={_max_drift:.2f}%)"
                            )
                            _METAL_DRIFT_SKIP_STREAK[_sym_u] = 0
                        else:
                            logger.warning(
                                f"[executor] {_sym_u}: kline/live drift "
                                f"{_drift_pct:.2f}% (kline={kline_close:.2f} "
                                f"kline_src={kline_source or 'unknown'} "
                                f"live={live_px:.2f} live_src={live_source or 'unknown'} "
                                f"max={_max_drift:.2f}%) — skip eval "
                                f"({_streak}/{_METAL_DRIFT_SKIP_WARN_CYCLES})"
                            )
                            return None
                    elif not _ct_live_proxy_klines:
                        _METAL_DRIFT_SKIP_STREAK[_sym_u] = 0
                        _price_tag = kline_source or live_source or "fallback"
                        logger.info(
                            f"[prices] {_sym_u} priced_by={_price_tag} "
                            f"kline_src={kline_source or 'unknown'} "
                            f"live_src={live_source or 'unknown'}"
                        )
                    price = live_px
                    price_source = "spot_live"
            else:
                price = live_px if live_px else kline_close
                price_source = "spot_live" if live_px else "kline_close"

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
                "price_source": price_source,
                "kline_close": kline_close,
                "kline_source": kline_source,
                "kline_synthetic": kline_synthetic,
                "candles_loaded": len(kl),
                "bid": bid,
                "ask": ask,
                "bid_price": bid,
                "ask_price": ask,
                "change_24h": ((price - closes[-min(96, len(closes))]) / closes[-min(96, len(closes))] * 100) if closes[-min(96, len(closes))] else 0.0,
                "volume_24h": 0.0,
                "high_24h": max(closes[-min(96, len(closes)):]),
                "low_24h":  min(closes[-min(96, len(closes)):]),
                "rsi": rsi,
                "volume_ratio": 1.0,
                "btc_correlation": 0.0,
                "enhanced_ta": {},
                "_asset_class": asset_class,
                "_timeframe": tf,
            }
            _PRICE_TA_CACHE[cache_key] = (result, now)
            return result
        except Exception as e:
            logger.warning(f"[tradfi] price/TA fetch failed for {symbol} ({asset_class}): {e}")
            return None

    try:
        from app.services.bitunix_market_data import fetch_crypto_price_and_ta
        result = await fetch_crypto_price_and_ta(
            http_client, symbol, prefetch=prefetch,
        )
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
    # ICT killzones — mirror strategy_ta.eval_fx_killzone / backtest_engine
    "london_kz": (7, 9),
    "ny_kz":     (12, 14),
    "asian_kz":  (20, 23),
}
_KZ_SESSION_IDS = frozenset({"london_kz", "ny_kz", "asian_kz", "any_kz"})

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
    # Defensive: accept only a list/tuple/set of day tokens.  A malformed value
    # (e.g. a raw string) would otherwise iterate character-by-character and
    # silently block every day — fail OPEN (allow trading) on bad shapes.
    if isinstance(allowed, str):
        allowed = [allowed]
    elif not isinstance(allowed, (list, tuple, set)):
        return True
    # Accept both long names ("monday") and short IDs ("mon") — the mobile
    # wizard sends short, the web wizard sends long, and either should work.
    day_map = {
        "monday": 0, "tuesday": 1, "wednesday": 2,
        "thursday": 3, "friday": 4, "saturday": 5, "sunday": 6,
        "mon": 0, "tue": 1, "wed": 2, "thu": 3, "fri": 4, "sat": 5, "sun": 6,
    }
    today = datetime.utcnow().weekday()
    allowed_nums = {day_map[str(d).lower()] for d in allowed if str(d).lower() in day_map}
    if not allowed_nums:
        return True  # nothing parseable → don't block
    return today in allowed_nums


def _check_time_filter(filters: Dict) -> bool:
    hour = datetime.utcnow().hour

    # 1. Explicit hour-range filter
    tf = filters.get("time_filter")
    if tf:
        if not (tf.get("start_hour", 0) <= hour < tf.get("end_hour", 24)):
            return False

    # 2. Named session filter  {"type":"session","sessions":["new_york"]}
    # Defensive: tolerate a bare list of ids or a raw string instead of the
    # canonical dict — a malformed shape would otherwise raise and block firing.
    sf = filters.get("session")
    if sf:
        if isinstance(sf, dict):
            raw_sessions = sf.get("sessions", [])
        elif isinstance(sf, (list, tuple)):
            raw_sessions = sf
        elif isinstance(sf, str):
            raw_sessions = [sf]
        else:
            raw_sessions = []
        wanted = [str(s).lower() for s in raw_sessions]
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
            if "any_kz" in wanted:
                wanted = [w for w in wanted if w != "any_kz"] + [
                    k for k in _KZ_SESSION_IDS if k != "any_kz"
                ]
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
        from app.services.binance_feed import binance_disabled, binance_http_get

        sources = [
            ("https://api.mexc.com/api/v3/klines", {"symbol": symbol, "interval": "1h", "limit": 30},
             "strategy_executor._htf_trend.mexc"),
        ]
        if not binance_disabled():
            sources.extend([
                ("https://fapi.binance.com/fapi/v1/klines", {"symbol": symbol, "interval": "1h", "limit": 30},
                 "strategy_executor._htf_trend.binance_futures"),
                ("https://api.binance.com/api/v3/klines", {"symbol": symbol, "interval": "1h", "limit": 30},
                 "strategy_executor._htf_trend.binance_spot"),
            ])
        for url, params, caller in sources:
            try:
                status, data = await binance_http_get(
                    http_client, url, params, timeout_s=6, caller=caller,
                )
                if status == 200 and data and len(data) >= 10:
                    closes = [float(k[4]) for k in data]
                    break
            except Exception:
                continue
        if len(closes) < 10:
            # MEXC/Binance unreachable (Railway geo-block) → Bitunix fallback.
            try:
                from app.services.bitunix_market_data import fetch_klines as _bx_klines
                _bk = await _bx_klines(http_client, symbol, "1h", 30)
                if _bk and len(_bk) >= 10:
                    closes = [float(k[4]) for k in _bk]
            except Exception:
                pass

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


def _db_rollback_safe(db) -> None:
    """Clear a poisoned SQLAlchemy session after Neon SSL drops."""
    try:
        db.rollback()
    except Exception:
        pass


def _rebind_session(db, *instances) -> None:
    """Re-attach ORM rows after rollback so a retry commit can persist them."""
    from sqlalchemy.orm import object_session

    for inst in instances:
        if inst is None:
            continue
        try:
            if object_session(inst) is db:
                continue
        except Exception:
            pass
        try:
            db.add(inst)
        except Exception:
            try:
                db.merge(inst)
            except Exception:
                pass


def _refresh_db_row(db, instance):
    """Refresh a row or reload by PK after commit/rollback churn (Neon SSL blips)."""
    from sqlalchemy.exc import InvalidRequestError
    from sqlalchemy.orm import object_session

    if instance is None:
        return None
    pk = getattr(instance, "id", None)
    if pk is not None:
        fresh = db.get(type(instance), pk)
        if fresh is not None:
            return fresh
    try:
        if object_session(instance) is db:
            db.refresh(instance)
    except InvalidRequestError:
        if pk is not None:
            return db.get(type(instance), pk)
    except Exception:
        pass
    return instance


def _db_count_safe(db, query_callable):
    """Run a DB count/first query; rollback session on transient errors."""
    from sqlalchemy.exc import OperationalError, PendingRollbackError

    try:
        return query_callable()
    except (OperationalError, PendingRollbackError):
        _db_rollback_safe(db)
        raise


async def _commit_db_with_retry(
    db,
    *,
    attempts: int = 3,
    label: str = "",
    instances: Optional[tuple] = None,
) -> None:
    """Commit with rollback + retry — critical after conditions-met fires."""
    from sqlalchemy.exc import OperationalError, PendingRollbackError

    _tag = f" ({label})" if label else ""
    for attempt in range(1, attempts + 1):
        try:
            db.commit()
            return
        except (OperationalError, PendingRollbackError) as exc:
            _db_rollback_safe(db)
            if instances:
                _rebind_session(db, *instances)
            if attempt >= attempts:
                logger.warning(
                    f"[{_log_ts()}] DB commit failed{_tag} after {attempts} attempts: {exc}"
                )
                raise
            logger.warning(
                f"[{_log_ts()}] DB commit transient error{_tag} — "
                f"retry {attempt}/{attempts}"
            )
            await asyncio.sleep(0.5 * attempt)


def _daily_execution_count(strategy_id: int, db) -> int:
    from app.strategy_models import StrategyExecution
    today = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
    return _db_count_safe(
        db,
        lambda: db.query(StrategyExecution).filter(
            StrategyExecution.strategy_id == strategy_id,
            StrategyExecution.fired_at >= today,
            StrategyExecution.outcome.notin_(["CANCELLED", "EXPIRED"]),
        ).count(),
    )


def _open_execution_count(strategy_id: int, db) -> int:
    from app.strategy_models import StrategyExecution
    return _db_count_safe(
        db,
        lambda: db.query(StrategyExecution).filter(
            StrategyExecution.strategy_id == strategy_id,
            StrategyExecution.outcome == "OPEN",
        ).count(),
    )


def _fired_today_for_symbol(strategy_id: int, symbol: str, db) -> bool:
    from app.strategy_models import StrategyExecution
    today = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
    return _db_count_safe(
        db,
        lambda: db.query(StrategyExecution).filter(
            StrategyExecution.strategy_id == strategy_id,
            StrategyExecution.symbol == symbol,
            StrategyExecution.fired_at >= today,
        ).first()
        is not None,
    )


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


def _prefetch_symbol_cooldowns(
    strategy_id: int,
    symbols: List[str],
    db,
    *,
    need_today: bool = False,
) -> Tuple[set, Dict[str, datetime]]:
    """
    One GROUP BY query for last-fired times across all candidate symbols.
    Replaces up to 2×N per-symbol queries in evaluate_and_fire's pre-filter.
    """
    from app.strategy_models import StrategyExecution
    from sqlalchemy import func

    if not symbols:
        return set(), {}

    rows = (
        db.query(
            StrategyExecution.symbol,
            func.max(StrategyExecution.fired_at).label("last_at"),
        )
        .filter(
            StrategyExecution.strategy_id == strategy_id,
            StrategyExecution.symbol.in_(symbols),
        )
        .group_by(StrategyExecution.symbol)
        .all()
    )
    last_fired = {r.symbol: r.last_at for r in rows if r.last_at}

    fired_today: set = set()
    if need_today and last_fired:
        today = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
        fired_today = {sym for sym, ts in last_fired.items() if ts >= today}

    return fired_today, last_fired


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

    # Pip-based aggregation for forex/metals strategies
    pips_closed = [e for e in closed if getattr(e, "pips_pnl", None) is not None]
    if pips_closed:
        _total_pips = sum(e.pips_pnl for e in pips_closed)
        perf.total_pips_pnl    = round(_total_pips, 1)
        perf.avg_pips_per_trade = round(_total_pips / len(pips_closed), 1)
    else:
        perf.total_pips_pnl    = None
        perf.avg_pips_per_trade = None

    db.commit()

    # Pool health monitor — warn when bg connections are nearly exhausted.
    try:
        from app.database import bg_engine as _pool_eng
        _pool = _pool_eng.pool
        _co = _pool.checkedout()
        _limit = _pool.size() + _pool._max_overflow  # type: ignore[attr-defined]
        if _co > max(6, _limit - 4):
            logger.warning(
                f"[PoolMonitor] bg pool {_co}/{_limit} connections checked out — "
                "consider lowering EXECUTOR_*_MAX_CONCURRENT"
            )
    except Exception:
        pass


# ─── Paper position monitor ──────────────────────────────────────────────────

# Incremental 1m candle tail per (symbol, asset_class) — avoids re-fetching full
# history every 20s sweep when positions are already open.
_PAPER_CANDLE_CACHE: Dict[Tuple[str, str], Tuple[list, int, int]] = {}


def _merge_candle_series(existing: list, new_rows: list) -> list:
    """Merge two OHLC tuples sorted by open_ts, deduplicating on timestamp."""
    if not existing:
        return list(new_rows)
    if not new_rows:
        return list(existing)
    seen: set = set()
    merged: list = []
    for c in sorted(existing + new_rows, key=lambda x: x[0]):
        if c[0] not in seen:
            seen.add(c[0])
            merged.append(c)
    return merged


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

        from app.services.binance_feed import binance_disabled, binance_http_get

        sources = [
            ("https://api.mexc.com/api/v3/klines",
             {"symbol": symbol, "interval": "1m", "startTime": start_ms, "limit": needed},
             "strategy_executor._fetch_candles.mexc"),
        ]
        if not binance_disabled():
            sources.extend([
                ("https://api.binance.com/api/v3/klines",
                 {"symbol": symbol, "interval": "1m", "startTime": start_ms, "limit": needed},
                 "strategy_executor._fetch_candles.binance_spot"),
                ("https://fapi.binance.com/fapi/v1/klines",
                 {"symbol": symbol, "interval": "1m", "startTime": start_ms, "limit": needed},
                 "strategy_executor._fetch_candles.binance_futures"),
            ])
        fetched = False
        for url, params, caller in sources:
            try:
                status, klines = await binance_http_get(
                    http_client, url, params, timeout_s=8, caller=caller,
                )
                if status != 200 or not klines:
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

        if not fetched:
            # MEXC/Binance unreachable (Railway geo-block) → one-shot Bitunix
            # recent 1m candles (not startTime-paged) so open crypto paper
            # positions still get TP/SL detection. Best-effort: covers the most
            # recent ~200 minutes, enough for recently-fired positions.
            try:
                from app.services.bitunix_market_data import fetch_klines as _bx_klines
                _bk = await _bx_klines(http_client, symbol, "1m", min(max(needed, 60), 200))
                for k in _bk:
                    _ts = int(k[0])
                    if _ts >= start_ms:
                        all_candles.append(
                            (_ts, float(k[1]), float(k[2]), float(k[3]), float(k[4]))
                        )
            except Exception:
                pass
            break
        if needed < chunk:
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


async def _fetch_paper_candles_cached(
    symbol: str,
    earliest_fired_at: datetime,
    http_client: httpx.AsyncClient,
    asset_class: str,
) -> list:
    """
    Paper-monitor candle fetch with incremental tail cache.
    First sweep loads from earliest_fired_at; later sweeps only append new 1m bars.
    """
    key = (symbol.upper(), (asset_class or "crypto").lower())
    earliest_ms = int(earliest_fired_at.timestamp() * 1000)
    cached = _PAPER_CANDLE_CACHE.get(key)

    if cached:
        old_candles, cache_earliest_ms, last_ts = cached
        if cache_earliest_ms <= earliest_ms and old_candles and last_ts:
            # Overlap one minute so we never miss a partial bar at the boundary.
            overlap_from = datetime.utcfromtimestamp(max(0, last_ts - 60_000) / 1000)
            delta = await _fetch_candles_since_entry(
                symbol, overlap_from, http_client, asset_class,
            )
            candles = _merge_candle_series(old_candles, delta)
            candles = [c for c in candles if c[0] >= earliest_ms]
        else:
            candles = await _fetch_candles_since_entry(
                symbol, earliest_fired_at, http_client, asset_class,
            )
    else:
        candles = await _fetch_candles_since_entry(
            symbol, earliest_fired_at, http_client, asset_class,
        )

    if candles:
        _PAPER_CANDLE_CACHE[key] = (candles, earliest_ms, candles[-1][0])
    elif key in _PAPER_CANDLE_CACHE and cached:
        # Keep prior tail if this sweep's fetch failed transiently.
        return cached[0]

    return candles


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


def _tp1_blend_exit(ex, exit_price: float) -> float:
    """Effective exit for P&L when TP1 partial close was taken."""
    if not getattr(ex, "tp1_done", False):
        return exit_price
    try:
        from app.services.forex_engine import pip_size as _pip_sz
        ps = _pip_sz(getattr(ex, "symbol", ""))
        entry = float(ex.entry_price)
        tp1_pips = float(ex.tp1_realized_pips or 0)
        if ex.direction == "LONG":
            tp1_px = entry + tp1_pips * ps
        else:
            tp1_px = entry - tp1_pips * ps
        closed = float(ex.tp1_closed_volume or 0)
        remain = float(ex.remaining_volume or 0)
        total = closed + remain
        frac = (closed / total) if total > 0 else 0.5
        return frac * tp1_px + (1.0 - frac) * float(exit_price)
    except Exception:
        return exit_price


def _tp1_blend_pips_line(ex, exit_price: float) -> Optional[str]:
    """Human-readable blended pips line for close notifications."""
    if not getattr(ex, "tp1_done", False):
        return None
    try:
        from app.services.forex_engine import pip_size as _pip_sz
        ps = _pip_sz(getattr(ex, "symbol", ""))
        if ps <= 0:
            return None
        entry = float(ex.entry_price)
        tp1_pips = float(ex.tp1_realized_pips or 0)
        if ex.direction == "LONG":
            runner_pips = (float(exit_price) - entry) / ps
        else:
            runner_pips = (entry - float(exit_price)) / ps
        closed = float(ex.tp1_closed_volume or 0)
        remain = float(ex.remaining_volume or 0)
        total = closed + remain
        pct = round(closed / total * 100) if total > 0 else 50
        rest_pct = 100 - pct
        return (
            f"TP1 +{tp1_pips:.0f}p on {pct}% + runner {runner_pips:+.0f}p on {rest_pct}%"
        )
    except Exception:
        return None


def _close_paper_execution(
    ex,
    outcome: str,
    exit_price: float,
    db,
    *,
    close_label: Optional[str] = None,
):
    """Mark a paper execution as closed and update performance.

    Uses an atomic UPDATE WHERE outcome='OPEN' so that when multiple uvicorn
    workers race to close the same position, exactly one wins and sends the
    Telegram notification — preventing duplicate messages.
    """
    _pnl_exit = _tp1_blend_exit(ex, exit_price)

    if ex.direction == "LONG":
        raw_pnl = (_pnl_exit - ex.entry_price) / ex.entry_price * 100
    else:
        raw_pnl = (ex.entry_price - _pnl_exit) / ex.entry_price * 100

    # Silently deduct realistic spread/execution cost so paper P&L reflects
    # what the user would actually see on their broker — no line item shown.
    if outcome != "CANCELLED":
        raw_pnl -= _paper_cost_basis_pct(
            getattr(ex, "asset_class", "crypto"),
            getattr(ex, "symbol", ""),
        )

    pnl_pct   = round(raw_pnl * ex.leverage, 2)
    closed_at = datetime.utcnow()

    # ── Pips P&L and spread audit (forex/metals only) ─────────────────────────
    pips_pnl:           float | None = None
    spread_pips_stored: float | None = None
    _ex_ac = getattr(ex, "asset_class", "crypto") or "crypto"
    if _ex_ac in ("forex", "metals", "commodity"):
        try:
            from app.services.forex_engine import (
                pip_size as _pip_sz,
                get_spread_pips as _gsp,
            )
            _ps = _pip_sz(getattr(ex, "symbol", ""))
            if _ps > 0 and ex.entry_price and _pnl_exit:
                if ex.direction == "LONG":
                    pips_pnl = round((_pnl_exit - ex.entry_price) / _ps, 1)
                else:
                    pips_pnl = round((ex.entry_price - _pnl_exit) / _ps, 1)
            spread_pips_stored = _gsp(getattr(ex, "symbol", ""))
        except Exception:
            pass

    # Atomic close — only the first worker to execute this UPDATE wins.
    from sqlalchemy import text as _text
    result = db.execute(
        _text(
            "UPDATE strategy_executions "
            "SET outcome=:outcome, exit_price=:exit_price, pnl_pct=:pnl, "
            "    closed_at=:closed_at, pips_pnl=:pips_pnl, "
            "    spread_pips_applied=:spread_pips "
            "WHERE id=:id AND outcome='OPEN'"
        ),
        {
            "outcome":    outcome,    "exit_price":  exit_price,
            "pnl":        pnl_pct,   "closed_at":   closed_at,
            "pips_pnl":   pips_pnl,  "spread_pips": spread_pips_stored,
            "id":         ex.id,
        },
    )
    db.commit()

    if result.rowcount == 0:
        # Another worker already closed this execution — skip notification.
        return

    from app.services.close_notify_dedupe import claim_close_notification
    if not claim_close_notification(db, ex.id):
        return

    _close_ac = (getattr(ex, "asset_class", "crypto") or "crypto").upper()
    logger.info(
        "[%s] [PaperMonitor] %s (%s): %s %s entry=%s exit=%s pnl=%+.1f%%",
        _log_ts(),
        outcome,
        _close_ac,
        ex.symbol,
        ex.direction,
        ex.entry_price,
        exit_price,
        pnl_pct,
    )

    # Sync in-memory object so callers that still hold a reference see the
    # updated state (prevents a second evaluation loop from re-closing it).
    ex.outcome    = outcome
    ex.exit_price = exit_price
    ex.pnl_pct    = pnl_pct
    ex.closed_at  = closed_at

    # Append close note — preserve any original open-time context (e.g. Live→Paper
    # fallback reason) so the history is auditable after close.
    from app.services.trade_management import close_note_label

    pnl_sign   = "+" if pnl_pct >= 0 else ""
    if outcome == "CANCELLED":
        close_note = "Expired · no TP/SL hit within hold period"
    else:
        label = close_label or close_note_label(ex, outcome, float(exit_price))
        close_note = f"{label} · {pnl_sign}{pnl_pct}% · exit {exit_price:.6g}"
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
        strat_name = (strat.name if strat else None) or "Your Strategy"
        tg_id = _telegram_int_id(user)
        if tg_id:
            _enqueue_close_telegram(
                tg_id,
                execution_id=ex.id,
                symbol=ex.symbol,
                outcome=outcome,
                asset_class=getattr(ex, "asset_class", "crypto") or "crypto",
                strategy_name=strat_name,
                direction=ex.direction,
                entry=ex.entry_price,
                exit_price=exit_price,
                pnl_pct=pnl_pct,
                leverage=ex.leverage,
                fired_at=ex.fired_at,
                closed_at=closed_at,
                conditions=ex.conditions_met,
                is_paper=True,
                tp1_blend_line=_tp1_blend_pips_line(ex, exit_price),
                mfe_pips=getattr(ex, "mfe_pips", None),
                mae_pips=getattr(ex, "mae_pips", None),
            )
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


_TG_OPEN_SENT = "tg_open_sent"


def _fmt_queued_open_notice(
    strategy_name: str, symbol: str, direction: str, *, leverage: int = 1
) -> str:
    """Brief Telegram while a live cTrader order is queued — not the full open card."""
    coin = symbol.upper().replace("USDT", "")
    return (
        f"⏳ <b>Placing live order on cTrader</b>\n"
        f"📋 <b>{_html.escape(str(strategy_name))}</b>\n\n"
        f"{'🟢' if direction == 'LONG' else '🔴'} <b>${_html.escape(coin)}</b> · "
        f"{direction} · {leverage}×\n\n"
        f"<i>Full trade details will arrive once the broker confirms the fill.</i>"
    )


def _claim_tg_note_flag(db, execution_id: int, flag: str) -> bool:
    """Atomically append a notes flag once (prevents duplicate Telegram alerts)."""
    from sqlalchemy import text as _text

    if not flag or not flag.replace("_", "").isalnum():
        return False
    result = db.execute(
        _text(
            "UPDATE strategy_executions "
            "SET notes = TRIM(BOTH ' |' FROM COALESCE(notes, '') || ' | ' || :flag) "
            "WHERE id = :id AND COALESCE(notes, '') NOT LIKE :pat"
        ),
        {"id": execution_id, "flag": flag, "pat": f"%{flag}%"},
    )
    db.commit()
    return result.rowcount > 0


def _claim_tg_open_notify(db, execution_id: int) -> bool:
    """Atomically claim the open Telegram card for one execution (prevents duplicates)."""
    return _claim_tg_note_flag(db, execution_id, "tg_open_sent")


def _should_dm_trade_alerts(portal_settings, is_paper: bool) -> bool:
    if not portal_settings:
        return True
    return bool(
        portal_settings.dm_paper_alerts if is_paper else portal_settings.dm_live_alerts
    )


def _ensure_open_notify_for_execution(
    db,
    ex,
    *,
    user,
    strategy,
) -> None:
    """Backfill the OPEN Telegram card when a trade exists but open DM was never sent."""
    from app.strategy_models import StrategyPortalSettings

    if _TG_OPEN_SENT in (ex.notes or ""):
        return
    # Live cTrader orders must not backfill until the broker confirms a fill.
    if (
        not ex.is_paper
        and (ex.asset_class or "") in ("forex", "index")
        and "order_queued" in (ex.notes or "")
        and not ex.ctrader_order_id
    ):
        return
    if not ex.entry_price or not ex.tp_price or not ex.sl_price:
        return
    tg_id = _telegram_int_id(user)
    if not tg_id:
        return
    try:
        portal_settings = db.query(StrategyPortalSettings).filter(
            StrategyPortalSettings.user_id == ex.user_id
        ).first()
    except Exception:
        portal_settings = None
    is_paper = bool(ex.is_paper)
    if not _should_dm_trade_alerts(portal_settings, is_paper):
        return
    entry = float(ex.entry_price)
    tp = float(ex.tp_price)
    sl = float(ex.sl_price)
    tp2 = float(ex.tp2_price) if ex.tp2_price else None

    def _dist_pct(target: float) -> float:
        return abs(target - entry) / max(entry, 1e-9) * 100.0

    tp_pct = _dist_pct(tp)
    sl_pct = _dist_pct(sl)
    tp2_pct = _dist_pct(tp2) if tp2 else None
    conds = ex.conditions_met if isinstance(ex.conditions_met, list) else []
    ac = ex.asset_class or "crypto"
    if not _claim_tg_open_notify(db, ex.id):
        return
    _schedule_tg_open_notify(
        ex.id,
        tg_id,
        _fmt_open_card(
            strategy_name=strategy.name if strategy else "Your Strategy",
            symbol=ex.symbol,
            direction=ex.direction,
            entry=entry,
            tp_price=tp,
            tp_pct=tp_pct,
            tp2_price=tp2,
            tp2_pct=tp2_pct,
            sl_price=sl,
            sl_pct=sl_pct,
            leverage=int(ex.leverage or 1),
            conditions=conds,
            is_paper=is_paper,
            asset_class=ac,
        ),
        asset_class=ac,
        symbol=ex.symbol or "",
    )
    logger.info(
        "[%s] [TG] backfilled open notify exec#%s %s",
        _log_ts(),
        ex.id,
        ex.symbol,
    )


def _release_tg_open_notify(db, execution_id: int) -> None:
    """Allow a retry if Telegram delivery failed after claim."""
    from sqlalchemy import text as _text

    try:
        db.execute(
            _text(
                "UPDATE strategy_executions "
                "SET notes = TRIM(BOTH ' |' FROM REPLACE(COALESCE(notes, ''), :flag, '')) "
                "WHERE id = :id"
            ),
            {"id": execution_id, "flag": _TG_OPEN_SENT},
        )
        db.commit()
    except Exception:
        try:
            db.rollback()
        except Exception:
            pass


async def _tg_send(
    telegram_id: int,
    text: str,
    *,
    asset_class: str = "crypto",
    msg_type: str = "dm",
    symbol: str = "",
    exec_id: int = 0,
) -> bool:
    """Send a trade-notification Telegram DM (forex-aware token order)."""
    from app.services.telegram_dm import send_dm, bot_tokens_for_asset
    return await send_dm(
        telegram_id,
        text,
        tokens=bot_tokens_for_asset(asset_class),
        msg_type=msg_type,
        symbol=symbol,
        exec_id=exec_id,
    )


async def _deliver_tg_open_notify(
    execution_id: int,
    telegram_id: int,
    text: str,
    *,
    asset_class: str = "crypto",
) -> None:
    """Send open card; release dedup claim if Telegram rejects delivery."""
    ok = await _tg_send(telegram_id, text, asset_class=asset_class)
    if ok:
        logger.info(
            "[%s] [TG] open notify sent exec#%s chat=%s asset=%s",
            _log_ts(),
            execution_id,
            telegram_id,
            asset_class,
        )
        return
    from app.database import BgSessionLocal as SessionLocal
    db = SessionLocal()
    try:
        _release_tg_open_notify(db, execution_id)
        logger.warning(
            "[%s] [TG] open notify not delivered exec#%s chat=%s asset=%s",
            _log_ts(),
            execution_id,
            telegram_id,
            asset_class,
        )
    finally:
        db.close()


def _schedule_tg_open_notify(
    execution_id: int,
    telegram_id: int,
    text: str,
    *,
    asset_class: str = "crypto",
    symbol: str = "",
) -> None:
    """Queue open alert for durable delivery (same path as close/breakeven)."""
    from app.services.telegram_dm import enqueue_trade_telegram

    try:
        enqueue_trade_telegram(
            telegram_id,
            text,
            msg_type="open",
            symbol=symbol,
            exec_id=execution_id,
            asset_class=asset_class,
        )
    except Exception as exc:
        logger.warning(
            "[%s] [TG] schedule open notify failed exec#%s: %s",
            _log_ts(),
            execution_id,
            exc,
        )


def _claim_tg_be_notify(db, execution_id: int) -> bool:
    """Atomically claim the breakeven Telegram alert for one execution."""
    return _claim_tg_note_flag(db, execution_id, "tg_be_sent")


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
    header   = "🧪 <b>PAPER TRADE OPENED</b>" if is_paper else "🚀 <b>LIVE TRADE OPENED</b>"

    # For forex: show pips (what traders actually think in) instead of %.
    # pips = price_distance / pip_size  e.g. XAUUSD: $2.50 / $0.10 = 25 pips.
    _is_forex = (asset_class == "forex")
    _ps = 0.0
    if _is_forex and entry and entry > 0:
        from app.services.forex_engine import pip_size as _pip_size
        _ps = _pip_size(symbol) or 0.0

    def _dist_label(px: float, pct: float) -> str:
        if _ps:
            return f"{round(abs(px - entry) / _ps)} pips"
        return f"{pct:.1f}%"

    tp_extra = _dist_label(tp_price, tp_pct)
    sl_extra = _dist_label(sl_price, sl_pct)

    # Risk : reward from raw price distances — works for both forex & crypto.
    rr = None
    try:
        _risk = abs(sl_price - entry)
        if _risk > 0:
            rr = abs(tp_price - entry) / _risk
    except Exception:
        rr = None

    # Price block as full-size bold lines. The previous design crammed these
    # into a monospace <pre> table, which Telegram renders in a small, cramped
    # font — users found it too small. Full-size <b> lines read much larger.
    lines = [f"💵 Entry    <b>{entry:.6g}</b>"]
    lines.append(f"🎯 Target   <b>{tp_price:.6g}</b>  <i>(+{tp_extra})</i>")
    if tp2_price and tp2_pct:
        lines.append(f"🎯 Target 2 <b>{tp2_price:.6g}</b>  <i>(+{_dist_label(tp2_price, tp2_pct)})</i>")
    lines.append(f"🛑 Stop     <b>{sl_price:.6g}</b>  <i>(−{sl_extra})</i>")
    if rr:
        lines.append(f"⚖️ R : R    <b>{rr:.1f} : 1</b>")
    price_block = "\n".join(lines)

    why = ""
    if conditions:
        passed = [c for c in conditions if c.startswith("✅")]
        if passed:
            why = "\n\n✅ <b>Why it fired</b>\n" + "\n".join(
                f"• {_html.escape(c[1:].strip())}" for c in passed[:5]
            )

    order_line = f"\n<i>Order #{_html.escape(str(order_id))}</i>" if order_id else ""
    footer     = "📄 <i>Paper trade · no real funds used</i>" if is_paper else "✅ <i>Live trade executed</i>"
    fired_line = f"\n🕐 <i>{_log_ts()}</i>"

    return (
        f"{header}\n"
        f"📋 <b>{_html.escape(str(strategy_name))}</b>\n\n"
        f"{dir_icon} <b>{_html.escape(str(symbol))}</b> · {direction} · {leverage}×\n\n"
        f"{price_block}"
        f"{why}\n\n"
        f"{footer}{fired_line}{order_line}"
    )


def _coerce_price(value, default: float = 0.0) -> float:
    try:
        v = float(value)
        if math.isnan(v) or math.isinf(v):
            return default
        return v
    except (TypeError, ValueError):
        return default


def _coerce_pct(value, default: float = 0.0) -> float:
    return _coerce_price(value, default)


def _close_telegram_type(outcome: Optional[str]) -> str:
    o = (outcome or "").upper()
    if o == "WIN":
        return "win"
    if o == "LOSS":
        return "loss"
    if o == "BREAKEVEN":
        return "breakeven"
    return "close"


def _fmt_excursion_telemetry_line(
    mfe_pips: Optional[float] = None,
    mae_pips: Optional[float] = None,
) -> str:
    if mfe_pips is None and mae_pips is None:
        return ""
    mfe = float(mfe_pips or 0.0)
    mae = float(mae_pips or 0.0)
    return f"📊 Peak +{mfe:.0f} pips · Worst −{mae:.0f} pips"


def _fmt_close_card(
    strategy_name: str, symbol: str, direction: str,
    entry: float, exit_price: float, outcome: str,
    pnl_pct: float, leverage: int,
    fired_at: datetime = None, closed_at: datetime = None,
    conditions: list = None, is_paper: bool = False,
    tp1_blend_line: Optional[str] = None,
    mfe_pips: Optional[float] = None,
    mae_pips: Optional[float] = None,
) -> str:
    entry = _coerce_price(entry)
    exit_price = _coerce_price(exit_price)
    pnl_pct = _coerce_pct(pnl_pct)
    leverage = int(leverage or 1)
    outcome = (outcome or "CLOSED").upper()
    direction = (direction or "LONG").upper()
    symbol = str(symbol or "???")
    strategy_name = str(strategy_name or "Strategy")

    dir_icon  = "🟢" if direction == "LONG" else "🔴"
    coin      = symbol.replace("USDT", "")

    if outcome == "WIN":
        icon      = "✅"
        result    = "WIN"
        hit_label = "TP hit"
    elif outcome == "LOSS":
        icon      = "🛑"
        result    = "LOSS"
        hit_label = "SL hit"
    elif outcome == "BREAKEVEN":
        icon      = "⚖️"
        result    = "BREAKEVEN"
        hit_label = "breakeven stop"
    else:
        icon      = "📊"
        result    = outcome
        hit_label = f"{outcome}"

    dur = ""
    if fired_at and closed_at:
        secs  = int((closed_at - fired_at).total_seconds())
        days, rem = divmod(secs, 86400)
        hours, rem = divmod(rem, 3600)
        mins       = rem // 60
        if days:
            dur = f"{days}d {hours}h {mins}m"
        elif hours:
            dur = f"{hours}h {mins}m"
        else:
            dur = f"{mins}m"

    def _pip_size(sym: str) -> float | None:
        """Return pip size for the symbol, or None for crypto (no pip convention)."""
        s = sym.upper().replace("/", "").replace("=F", "").replace("=X", "")
        # Metals — pip size MUST match app.services.forex_engine.pip_size (the
        # canonical source used for the stored pips_pnl and the entry card).
        # Gold uses the retail/broker pip convention: 1 pip = $0.10 (e.g. a $2.45
        # price move ≈ 25 pips, matching FP Markets / broker terminals). Kept in
        # lockstep with forex_engine._METAL_PIP_SIZES (0.10) and ctrader_client
        # pip_value ($10/pip/lot) — change all three together or sizing drifts.
        if s in ("XAUUSD", "GOLD", "GC", "XAUUSDT"):   return 0.10   # gold: retail pip = $0.10
        if s in ("XAGUSD", "SILVER", "SI", "XAGUSDT"):  return 0.001  # silver: digits=3
        if s in ("XPTUSD", "PLATINUM", "PL"):            return 0.01   # platinum: digits=2
        # JPY pairs (2 decimal places)
        if "JPY" in s:                                   return 0.01
        # Standard 4-decimal forex pairs
        _FX = ("USD","EUR","GBP","AUD","NZD","CAD","CHF","SGD","HKD","NOK","SEK","DKK","PLN","CZK","HUF","MXN","ZAR","TRY","INR")
        if any(s.startswith(c) or s.endswith(c) for c in _FX) and len(s) == 6:
            return 0.0001
        # Indices — 1 point per pip
        try:
            from app.services.index_symbols import is_index_symbol, index_pip_size
            if is_index_symbol(s):
                return index_pip_size(s)
        except Exception:
            pass
        _IDX = ("US30","US500","SPX","SPX500","NAS","NAS100","NDX","DAX","DE40","GER40","UK100","FTSE","JP225","HK50","ASX","IT40","FR40","ES35")
        if any(s.startswith(i) for i in _IDX):          return 1.0
        # Oil / commodities
        if s in ("USOIL","WTIUSD","BRENTUSD","UKOIL","CL","NG"):
            return 0.01
        return None  # crypto — no pip convention

    pip_sz  = _pip_size(symbol)
    # Signed raw price move (positive = favourable for the trade direction)
    if entry <= 0 or exit_price <= 0:
        raw_move = 0.0
    elif abs(exit_price - entry) <= max(abs(entry) * 1e-12, 1e-9):
        raw_move = 0.0
    else:
        raw_move = (exit_price - entry) if direction == "LONG" else (entry - exit_price)

    if tp1_blend_line:
        pnl_display = str(tp1_blend_line)
    elif pip_sz is not None and pip_sz > 0:
        pips = raw_move / pip_sz
        sign = "+" if pips >= 0 else "−"
        # Whole pips for large values, 1dp for fractional
        pnl_display = f"{sign}{abs(pips):.0f} pips" if abs(pips) >= 1 else f"{sign}{abs(pips):.1f} pips"
    else:
        # Crypto: keep % with adaptive precision
        a = abs(pnl_pct)
        if a < 0.1:   pnl_display = f"{pnl_pct:+.3f}%"
        elif a < 10:  pnl_display = f"{pnl_pct:+.2f}%"
        else:         pnl_display = f"{pnl_pct:+.1f}%"

    # Full-size bold lines (see _fmt_open_card — replaces the small <pre> table).
    pl_emoji = "📈" if raw_move >= 0 else "📉"
    lines = [
        f"💵 Entry  <b>{entry:.6g}</b>",
        f"🏁 Exit   <b>{exit_price:.6g}</b>  <i>({hit_label})</i>",
        f"{pl_emoji} P/L    <b>{pnl_display}</b>",
    ]
    _excursion = _fmt_excursion_telemetry_line(mfe_pips, mae_pips)
    if _excursion:
        lines.append(_excursion)
    if dur:
        lines.append(f"⏱ Time   <b>{dur}</b>")
    price_block = "\n".join(lines)

    why = ""
    if conditions:
        passed = [c for c in conditions if c.startswith("✅")]
        if passed:
            why = "\n\n✅ <b>Triggered by</b>\n" + "\n".join(
                f"• {_html.escape(c[1:].strip())}" for c in passed[:3]
            )

    paper_tag = "📄 <i>Paper trade result</i>" if is_paper else "✅ <i>Live trade result</i>"
    if closed_at and hasattr(closed_at, "strftime"):
        closed_line = f"\n🕐 <i>{closed_at.strftime('%Y-%m-%d %H:%M:%S UTC')}</i>"
    else:
        closed_line = f"\n🕐 <i>{_log_ts()}</i>"

    return (
        f"{icon} <b>{result} · {_html.escape(strategy_name)}</b>\n\n"
        f"{dir_icon} <b>${_html.escape(str(coin))}</b> · {direction} · {leverage}×\n\n"
        f"{price_block}"
        f"{why}\n\n"
        f"{paper_tag}{closed_line}"
    )


def _fmt_close_fallback(
    symbol: str,
    direction: str,
    outcome: str,
    *,
    pnl_display: str = "",
) -> str:
    """Minimal close card when the full formatter throws."""
    o = (outcome or "CLOSED").upper()
    icon = {"WIN": "✅", "LOSS": "🛑", "BREAKEVEN": "⚖️"}.get(o, "📊")
    sym = (symbol or "???").upper().replace("USDT", "")
    dir_ = (direction or "").upper()
    pl = (pnl_display or "").strip()
    line = f"{icon} {o} {sym}"
    if dir_:
        line += f" {dir_}"
    if pl:
        line += f" {pl}"
    return line


def _build_close_telegram_safe(
    *,
    execution_id: int = 0,
    strategy_name: str = "Strategy",
    symbol: str = "",
    direction: str = "LONG",
    entry: float = 0.0,
    exit_price: float = 0.0,
    outcome: str = "CLOSED",
    pnl_pct: float = 0.0,
    leverage: int = 1,
    fired_at: datetime = None,
    closed_at: datetime = None,
    conditions: list = None,
    is_paper: bool = False,
    tp1_blend_line: Optional[str] = None,
    mfe_pips: Optional[float] = None,
    mae_pips: Optional[float] = None,
) -> str:
    fields = {
        "execution_id": execution_id,
        "strategy_name": strategy_name,
        "symbol": symbol,
        "direction": direction,
        "entry": entry,
        "exit_price": exit_price,
        "outcome": outcome,
        "pnl_pct": pnl_pct,
        "leverage": leverage,
        "tp1_blend_line": tp1_blend_line,
    }
    try:
        return _fmt_close_card(
            strategy_name=strategy_name,
            symbol=symbol,
            direction=direction,
            entry=entry,
            exit_price=exit_price,
            outcome=outcome,
            pnl_pct=pnl_pct,
            leverage=leverage,
            fired_at=fired_at,
            closed_at=closed_at,
            conditions=conditions,
            is_paper=is_paper,
            tp1_blend_line=tp1_blend_line,
            mfe_pips=mfe_pips,
            mae_pips=mae_pips,
        )
    except Exception as exc:
        logger.exception(
            "[telegram] close message build failed exec=%s fields=%s: %s",
            execution_id,
            fields,
            exc,
        )
        pl = ""
        try:
            ps_fn = None
            from app.services.forex_engine import pip_size as ps_fn
            ps = ps_fn(symbol) if ps_fn else 0.0
            ent = _coerce_price(entry)
            ex = _coerce_price(exit_price)
            if ps and ps > 0 and ent > 0 and ex > 0:
                mv = (ex - ent) if (direction or "").upper() == "LONG" else (ent - ex)
                pips = mv / ps
                pl = f"{'+' if pips >= 0 else '−'}{abs(pips):.0f} pips"
            elif pnl_pct is not None:
                pl = f"{_coerce_pct(pnl_pct):+.2f}%"
        except Exception:
            pl = f"{_coerce_pct(pnl_pct):+.2f}%" if pnl_pct is not None else ""
        return _fmt_close_fallback(symbol, direction, outcome, pnl_display=pl)


def _fmt_breakeven_card(
    strategy_name: str,
    symbol: str,
    direction: str,
    *,
    kind: str = "live",
) -> str:
    sym = str(symbol or "???").upper().replace("USDT", "")
    dir_ = (direction or "LONG").upper()
    _mode = "🧪 PAPER" if (kind or "").lower() == "paper" else "🎯 LIVE"
    text = (
        f"🛡️ <b>Breakeven · {sym}</b> · <i>{_mode}</i>\n"
        f"📋 <b>{_html.escape(str(strategy_name or 'Strategy'))}</b>\n\n"
        f"Stop moved to entry — this {dir_} trade is now risk-free. ✅"
    )
    if (kind or "").lower() == "paper":
        text += (
            "\n\n📄 <i>Paper/simulated — not on your cTrader account. "
            "Switch strategy to Live + link cTrader for real orders.</i>"
        )
    return text


def _fmt_breakeven_fallback(symbol: str, direction: str = "") -> str:
    sym = (symbol or "???").upper().replace("USDT", "")
    dir_ = (direction or "").upper()
    return f"🔒 Breakeven {sym}" + (f" {dir_}" if dir_ else "")


def _build_breakeven_telegram_safe(
    *,
    execution_id: int = 0,
    strategy_name: str = "Strategy",
    symbol: str = "",
    direction: str = "LONG",
    kind: str = "live",
) -> str:
    fields = {
        "execution_id": execution_id,
        "strategy_name": strategy_name,
        "symbol": symbol,
        "direction": direction,
        "kind": kind,
    }
    try:
        return _fmt_breakeven_card(
            strategy_name=strategy_name,
            symbol=symbol,
            direction=direction,
            kind=kind,
        )
    except Exception as exc:
        logger.exception(
            "[telegram] breakeven message build failed exec=%s fields=%s: %s",
            execution_id,
            fields,
            exc,
        )
        return _fmt_breakeven_fallback(symbol, direction)


def _enqueue_close_telegram(
    telegram_id,
    *,
    execution_id: int,
    symbol: str,
    outcome: str,
    asset_class: str = "forex",
    **card_kwargs,
) -> None:
    from app.services.telegram_dm import enqueue_trade_telegram

    text = _build_close_telegram_safe(
        execution_id=execution_id,
        symbol=symbol,
        outcome=outcome,
        **card_kwargs,
    )
    enqueue_trade_telegram(
        telegram_id,
        text,
        msg_type=_close_telegram_type(outcome),
        symbol=symbol,
        exec_id=execution_id,
        asset_class=asset_class,
    )


async def _deliver_close_telegram(
    telegram_id,
    *,
    execution_id: int,
    symbol: str,
    outcome: str,
    asset_class: str = "forex",
    **card_kwargs,
) -> bool:
    from app.services.telegram_dm import deliver_trade_telegram

    text = _build_close_telegram_safe(
        execution_id=execution_id,
        symbol=symbol,
        outcome=outcome,
        **card_kwargs,
    )
    return await deliver_trade_telegram(
        telegram_id,
        text,
        msg_type=_close_telegram_type(outcome),
        symbol=symbol,
        exec_id=execution_id,
        asset_class=asset_class,
    )


def _enqueue_breakeven_telegram(
    telegram_id,
    *,
    execution_id: int = 0,
    strategy_name: str = "Strategy",
    symbol: str = "",
    direction: str = "LONG",
    kind: str = "live",
    asset_class: str = "forex",
) -> None:
    from app.services.telegram_dm import enqueue_trade_telegram

    text = _build_breakeven_telegram_safe(
        execution_id=execution_id,
        strategy_name=strategy_name,
        symbol=symbol,
        direction=direction,
        kind=kind,
    )
    enqueue_trade_telegram(
        telegram_id,
        text,
        msg_type="breakeven",
        symbol=symbol,
        exec_id=execution_id,
        asset_class=asset_class,
    )


async def _send_paper_close_dm(
    telegram_id: int,
    text: str,
    *,
    asset_class: str = "crypto",
    execution_id: int = 0,
    symbol: str = "",
    outcome: str = "CLOSED",
):
    from app.services.telegram_dm import deliver_trade_telegram

    await deliver_trade_telegram(
        telegram_id,
        text,
        msg_type=_close_telegram_type(outcome),
        symbol=symbol,
        exec_id=execution_id,
        asset_class=asset_class,
    )


def _notify_breakeven_alert(
    *,
    user_id: int,
    telegram_id,
    strategy_name: str,
    symbol: str,
    direction: str,
    leverage: int,
    move_pct: float,
    strategy_id: int = 0,
    execution_id: int = 0,
    kind: str = "live",
) -> None:
    """Fire push + Telegram alerts when a stop is moved to breakeven.

    Sync-safe so it can be called from BOTH the sync paper monitor and the async
    live forex manager: push goes via the already-threaded notify_breakeven_bg,
    and the Telegram DM is scheduled on the running loop when one exists, else
    dispatched on a throwaway thread.
    """
    try:
        from app.services.expo_push import notify_breakeven_bg
        notify_breakeven_bg(
            user_id, strategy_name, symbol, direction, int(leverage or 1),
            float(move_pct), strategy_id=strategy_id,
            execution_id=execution_id, kind=kind,
        )
    except Exception as _pe:
        logger.debug(f"[BE-notify] push failed exec#{execution_id}: {_pe}")

    # Skip web-registered users (telegram_id "WEB-…") and any non-numeric id —
    # mirrors _telegram_int_id so we never make a noisy doomed sendMessage call.
    if not telegram_id or str(telegram_id).startswith("WEB-"):
        return
    try:
        _tid = int(telegram_id)
    except (TypeError, ValueError):
        return
    try:
        _enqueue_breakeven_telegram(
            _tid,
            execution_id=execution_id,
            strategy_name=strategy_name,
            symbol=symbol,
            direction=direction,
            kind=kind,
            asset_class="forex",
        )
    except Exception as _te:
        logger.warning(
            "[telegram] breakeven enqueue failed exec=%s: %s",
            execution_id,
            _te,
        )


def _compute_be_trigger_price(symbol, entry, direction, tp_price, ex_cfg):
    """Price at which the stop should jump to entry (auto-breakeven) for tradfi.

    Forex/index strategies run at 1x leverage, so the crypto leveraged-ROI breakeven
    trigger (price-move% x leverage >= threshold) is mathematically unreachable
    (a 50-pip gold trade is a fraction of 1%). This returns a reachable,
    broker-style price level instead:

      * primary  — ``breakeven_at_pips``: move SL to entry once price is N pips
        in profit (matches the wizard's pip control).
      * legacy   — ``breakeven_pct``/``breakeven_at_pct``: treated as % of the
        distance from entry to TP, so older percent-based forex strategies still
        trigger instead of staying dead.

    Returns the trigger price, or None when breakeven isn't configured.
    """
    try:
        be_pips = ex_cfg.get("breakeven_at_pips")
        if be_pips and float(be_pips) > 0:
            from app.services.forex_engine import pip_size as _pipsz
            dist = float(be_pips) * _pipsz(symbol)
            return (entry + dist) if direction == "LONG" else (entry - dist)
        be_pct = ex_cfg.get("breakeven_pct") or ex_cfg.get("breakeven_at_pct")
        if be_pct and float(be_pct) > 0 and tp_price:
            frac = float(be_pct) / 100.0
            return entry + frac * (float(tp_price) - entry)
    except Exception:
        return None
    return None


def _classify_sl_outcome(sl: float, entry: float, direction: str) -> str:
    """Delegate to trade_management — single source of truth for SL labelling."""
    from app.services.trade_management import _classify_sl_outcome as _tm_classify

    return _tm_classify(sl, entry, direction)


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
    from app.services.trade_management import (
        active_sl,
        breakeven_sl_price,
        breakeven_was_claimed,
        check_directional_exit_hit,
        classify_sl_close_outcome,
        effective_tp,
        effective_trade_mgmt_cfg,
        persist_paper_stop_level,
        validate_close_sanity,
    )
    from app.services.forex_tick_manager import unregister_position

    if not ex.entry_price:
        return False
    if effective_tp(ex) is None and active_sl(ex) is None and ex.sl_price is None:
        return False

    fired_at = ex.fired_at or datetime.utcnow()
    elapsed_hours = (datetime.utcnow() - fired_at).total_seconds() / 3600
    elapsed_mins  = elapsed_hours * 60

    # ── Candle evaluation FIRST ───────────────────────────────────────────────
    # Always scan for a TP/SL hit before considering expiry.  This prevents
    # incorrectly CANCELLING a trade whose TP was already hit but whose
    # fired_at is older than PAPER_MAX_HOLD_HOURS.
    be_pct = None
    be_trigger_price = None  # forex: absolute price at which SL jumps to entry
    partial_close_pct = None
    be_timer_minutes = None
    trail_enabled = False
    trail_pct = None
    try:
        from app.strategy_models import UserStrategy
        strat = db.query(UserStrategy).filter(UserStrategy.id == ex.strategy_id).first()
        if strat and strat.config:
            ex_cfg  = (strat.config or {}).get("exit", {})
            risk_cfg = (strat.config or {}).get("risk", {})
            be_pct = ex_cfg.get("breakeven_pct") or ex_cfg.get("breakeven_at_pct")
            if be_pct is not None:
                be_pct = float(be_pct)
            # Forex/index/stock at 1x leverage → leveraged-ROI trigger is unreachable;
            # use pip/point distance or % of entry→TP instead.
            _ex_ac = getattr(ex, "asset_class", None) or "crypto"
            if _ex_ac in ("forex", "index", "stock"):
                be_trigger_price = _compute_be_trigger_price(
                    ex.symbol, float(ex.entry_price), ex.direction,
                    ex.tp_price, ex_cfg,
                )
            _pcp = ex_cfg.get("partial_close_pct")
            if _pcp:
                partial_close_pct = float(_pcp)
            _bet = risk_cfg.get("be_timer_minutes")
            if _bet:
                be_timer_minutes = float(_bet)
            # Trailing stop — price-% distance behind the best price reached.
            trail_enabled = bool(ex_cfg.get("trailing_stop"))
            _tsp = ex_cfg.get("trailing_stop_pct")
            if trail_enabled and _tsp:
                trail_pct = float(_tsp)
            if trail_enabled and (trail_pct is None or trail_pct <= 0):
                # Sensible default: half the stop distance (mirrors the builder).
                _slp = ex_cfg.get("stop_loss_pct")
                trail_pct = (float(_slp) / 2.0) if _slp else None
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

    be_activated = breakeven_was_claimed(ex)

    def _outcome_for_sl(exit_px: float) -> str:
        return classify_sl_close_outcome(ex, exit_px, log_prefix="[trade-mgmt]")

    # Track whether we've already performed a partial close for this execution
    # (stored in notes to survive across monitor cycles).
    partial_close_done = bool(ex.notes and "partial_close_done" in ex.notes)

    # ── Moved-stop chronology guard ───────────────────────────────────────────
    # When the stop is moved tighter (auto-breakeven, partial-close-to-entry or
    # trailing) it must only apply to price action AT/AFTER the move. The sweep
    # re-scans every cycle from entry, so without this guard a candle that dipped
    # to the (later) breakeven/trailing level BEFORE the stop ever moved there is
    # wrongly read as a stop-out on the next scan — closing winners out flat.
    # `sl_eff_ms` = timestamp at which the CURRENT stop level became effective
    # (last move + 1ms). Candles older than it skip the stop-hit test: the trade
    # was open through them under a looser stop, so no real hit occurred. Stored
    # in notes so it survives across monitor cycles.
    sl_eff_ms = None
    if ex.notes:
        import re as _re_se
        _mse = _re_se.search(r"sleff=(\d+)", ex.notes)
        if _mse:
            try:
                sl_eff_ms = int(_mse.group(1))
            except Exception:
                sl_eff_ms = None

    if candles:
        entry_ms = int(fired_at.timestamp() * 1000)
        relevant = [c for c in candles if c[0] >= entry_ms - 60_000]
        if relevant:
            for _ts, open_, high, low, close in relevant:
                # Stop only "live" for candles at/after its last move (chronology
                # guard — a moved stop must never retro-apply to earlier price).
                _sl_active = not (sl_eff_ms is not None and _ts < sl_eff_ms)
                _eff_sl = active_sl(ex)
                if _eff_sl is None:
                    _eff_sl = ex.sl_price
                _tp_lvl = effective_tp(ex)
                if ex.direction == "LONG":
                    tp_hit = _tp_lvl is not None and high >= float(_tp_lvl)
                    sl_hit = (_eff_sl is not None and low <= float(_eff_sl)) and _sl_active
                    if tp_hit and sl_hit:
                        if close >= open_:
                            outcome, label = validate_close_sanity(
                                ex, "WIN", float(_tp_lvl), "tp",
                            )
                            _close_paper_execution(
                                ex, outcome, float(_tp_lvl), db, close_label=label,
                            )
                        else:
                            outcome = _outcome_for_sl(_eff_sl)
                            outcome, label = validate_close_sanity(
                                ex, outcome, float(_eff_sl), "sl",
                            )
                            _close_paper_execution(
                                ex, outcome, float(_eff_sl), db, close_label=label,
                            )
                        unregister_position(ex.id, ex.symbol or "")
                        return True
                else:
                    tp_hit = _tp_lvl is not None and low <= float(_tp_lvl)
                    sl_hit = (_eff_sl is not None and high >= float(_eff_sl)) and _sl_active
                    if tp_hit and sl_hit:
                        if close <= open_:
                            outcome, label = validate_close_sanity(
                                ex, "WIN", float(_tp_lvl), "tp",
                            )
                            _close_paper_execution(
                                ex, outcome, float(_tp_lvl), db, close_label=label,
                            )
                        else:
                            outcome = _outcome_for_sl(_eff_sl)
                            outcome, label = validate_close_sanity(
                                ex, outcome, float(_eff_sl), "sl",
                            )
                            _close_paper_execution(
                                ex, outcome, float(_eff_sl), db, close_label=label,
                            )
                        unregister_position(ex.id, ex.symbol or "")
                        return True

                # ── Partial close at TP1 (paper simulation) ───────────────
                # When partial_close_pct is set and TP1 is hit (but not yet
                # done), we simulate closing partial_close_pct% of the
                # position and moving the SL to breakeven for the remainder.
                # The execution record stays OPEN so TP2 / final SL can still
                # close it — we just record that partial already happened.
                if tp_hit and partial_close_pct and not partial_close_done and ex.tp2_price and _tp_lvl:
                    partial_close_done = True
                    # Move SL to entry price (protect the partial profit). The
                    # entry-stop only applies to candles AFTER this one (chronology
                    # guard) so an earlier dip can't retro-close the remainder.
                    _be_sl = breakeven_sl_price(
                        ex.symbol, float(ex.entry_price), ex.direction, 0.0,
                    )
                    if not persist_paper_stop_level(ex, _be_sl, db, mark_breakeven=True):
                        continue
                    sl_eff_ms = _ts + 1
                    # Mark partial close in notes (survives across monitor cycles)
                    _old_notes = ex.notes or ""
                    # Preserve any existing metadata before the live-pnl suffix
                    import re as _re_pc
                    _base_notes = _old_notes.split(" | open")[0].split(" | unrealised")[0].strip(" |")
                    _base_notes = _re_pc.sub(r"\s*\|?\s*sleff=\d+", "", _base_notes).strip(" |")
                    ex.notes = (_base_notes + f" | partial_close_done | sleff={sl_eff_ms}").strip(" |")
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

                if tp_hit and _tp_lvl is not None:
                    outcome, label = validate_close_sanity(
                        ex, "WIN", float(_tp_lvl), "tp",
                    )
                    _close_paper_execution(
                        ex, outcome, float(_tp_lvl), db, close_label=label,
                    )
                    unregister_position(ex.id, ex.symbol or "")
                    return True
                if sl_hit and _eff_sl is not None:
                    outcome = _outcome_for_sl(_eff_sl)
                    outcome, label = validate_close_sanity(
                        ex, outcome, float(_eff_sl), "sl",
                    )
                    _close_paper_execution(
                        ex, outcome, float(_eff_sl), db, close_label=label,
                    )
                    unregister_position(ex.id, ex.symbol or "")
                    return True

                if not be_activated:
                    be_reached = False
                    be_log = ""
                    if be_trigger_price is not None:
                        # Forex: pip/distance price trigger (reachable at 1x lev).
                        if ex.direction == "LONG":
                            be_reached = high >= be_trigger_price
                        else:
                            be_reached = low <= be_trigger_price
                        be_log = f"price reached {be_trigger_price:.5f}"
                    elif be_pct is not None:
                        # Crypto: leveraged-ROI trigger.
                        if ex.direction == "LONG":
                            candle_roi = ((high - ex.entry_price) / ex.entry_price) * 100 * ex.leverage
                        else:
                            candle_roi = ((ex.entry_price - low) / ex.entry_price) * 100 * ex.leverage
                        be_reached = candle_roi >= be_pct
                        be_log = f"ROI {candle_roi:.1f}% >= {be_pct}%"
                    if be_reached and not breakeven_was_claimed(ex):
                        _tm_cfg = effective_trade_mgmt_cfg(
                            (strat.config or {}) if strat else {}
                        )
                        _new_sl = breakeven_sl_price(
                            ex.symbol,
                            float(ex.entry_price),
                            ex.direction,
                            _tm_cfg.get("breakeven_offset_pips", 0.0),
                        )
                        if persist_paper_stop_level(
                            ex, _new_sl, db, mark_breakeven=True,
                        ):
                            be_activated = True
                            sl_eff_ms = _ts + 1
                            _bn = (ex.notes or "").strip()
                            if "be_moved" not in _bn:
                                ex.notes = (
                                    (f"{_bn} | be_moved".strip(" |")) if _bn else "be_moved"
                                )
                            try:
                                db.commit()
                            except Exception:
                                db.rollback()
                            logger.info(
                                f"[PaperMonitor] AUTO-BREAKEVEN: exec #{ex.id} "
                                f"{ex.symbol} {be_log} → SL @ {_new_sl:.5f} "
                                f"(current_sl persisted)"
                            )
                            try:
                                from app.models import User as _UserM
                                from app.strategy_models import UserStrategy as _StratM
                                _u = db.query(_UserM).filter(_UserM.id == ex.user_id).first()
                                _s = db.query(_StratM).filter(_StratM.id == ex.strategy_id).first()
                                if _u and _s:
                                    _ensure_open_notify_for_execution(
                                        db, ex, user=_u, strategy=_s,
                                    )
                                if _claim_tg_be_notify(db, ex.id):
                                    if ex.direction == "LONG":
                                        _mv = (high - ex.entry_price) / ex.entry_price * 100
                                    else:
                                        _mv = (ex.entry_price - low) / ex.entry_price * 100
                                    _notify_breakeven_alert(
                                        user_id=ex.user_id,
                                        telegram_id=(_u.telegram_id if _u else None),
                                        strategy_name=(_s.name if _s else "Strategy"),
                                        symbol=ex.symbol, direction=ex.direction,
                                        leverage=(ex.leverage or 1),
                                        move_pct=_mv * max(1, (ex.leverage or 1)),
                                        strategy_id=ex.strategy_id, execution_id=ex.id,
                                        kind=("paper" if ex.is_paper else "live"),
                                    )
                            except Exception as _ne:
                                logger.debug(f"[BE-notify] paper exec#{ex.id}: {_ne}")
                        else:
                            logger.warning(
                                "[PaperMonitor] AUTO-BREAKEVEN persist failed "
                                "exec #%s %s — notification suppressed",
                                ex.id,
                                ex.symbol,
                            )

                # ── Trailing stop ─────────────────────────────────────────────
                # Ratchet the SL behind the best price reached so far by
                # trail_pct (price %). Only ever TIGHTENS (never loosens), and
                # uses this candle's extreme so it applies to SUBSEQUENT candles
                # only — no same-candle look-ahead, same discipline as breakeven.
                if trail_enabled and trail_pct and trail_pct > 0:
                    _cur = active_sl(ex) or ex.sl_price
                    if ex.direction == "LONG":
                        cand_sl = high * (1 - trail_pct / 100.0)
                        if _cur is not None and cand_sl > _cur:
                            if persist_paper_stop_level(ex, cand_sl, db):
                                sl_eff_ms = _ts + 1
                    else:
                        cand_sl = low * (1 + trail_pct / 100.0)
                        if _cur is not None and cand_sl < _cur:
                            if persist_paper_stop_level(ex, cand_sl, db):
                                sl_eff_ms = _ts + 1

            last_close = relevant[-1][4]
            if ex.direction == "LONG":
                unreal = (last_close - ex.entry_price) / ex.entry_price * 100
            else:
                unreal = (ex.entry_price - last_close) / ex.entry_price * 100
            pnl_note = f"open · unrealised {'+' if unreal >= 0 else ''}{unreal:.2f}% · last {last_close:.6g}"
            orig = ex.notes or ""
            # Preserve metadata tokens (broker ids, partial/breakeven state and
            # the moved-stop effective timestamp) — only the live-pnl suffix is
            # rebuilt each cycle.
            base = orig.split(" | open")[0].split(" | unrealised")[0].strip(" |")
            import re as _re_notes
            base = _re_notes.sub(r"\s*\|?\s*sleff=\d+", "", base).strip(" |")
            if sl_eff_ms is not None:
                base = (base + f" | sleff={sl_eff_ms}").strip(" |")
            if partial_close_done and "partial_close_done" not in base:
                base = (base + " | partial_close_done").strip(" |")
            ex.notes = (base + " | " + pnl_note).strip(" |") if base else pnl_note
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

    logger.info(
        f"🧪 Paper position monitor started ({PAPER_MONITOR_INTERVAL}s interval, "
        "full-history candle scan)"
    )
    try:
        _audit_db = SessionLocal()
        try:
            from app.services.trade_management import (
                audit_mislabeled_breakeven_paper_closes,
                correct_mislabeled_breakeven_closes,
            )

            _n = audit_mislabeled_breakeven_paper_closes(_audit_db, limit=20)
            _fixed = correct_mislabeled_breakeven_closes(_audit_db, limit=20)
            if _n or _fixed:
                logger.warning(
                    "[trade-mgmt] breakeven audit on monitor start: "
                    "%s mislabeled found, %s corrected",
                    _n,
                    _fixed,
                )
        finally:
            _audit_db.close()
    except Exception as _aud_exc:
        logger.debug("[trade-mgmt] breakeven audit skipped: %s", _aud_exc)

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
        from app.database import bg_db_slot
        from sqlalchemy.exc import TimeoutError as _SATimeoutError

        try:
            async with bg_db_slot():
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
                    db.expunge_all()
                finally:
                    db.close()
        except _SATimeoutError:
            logger.warning("[PaperMonitor] DB pool busy — skipping sweep this cycle")
            return

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
            candles  = await _fetch_paper_candles_cached(
                symbol, earliest, http_client, asset_class,
            )
            # ── Live intra-candle hit detection (forex/index) ──────────────────
            # Paper SL/TP is normally only seen when the 1m candle CLOSES (and the
            # broker candle feed can fall back to a delayed source), which made
            # gold alerts minutes late. Append a synthetic "now" candle built from
            # the real-time broker spot price (same cTrader feed the live path
            # uses) so a stop/target/breakeven is detected within the sweep cycle
            # instead of waiting for the candle to close. Crypto already gets
            # frequent candles, so this targets forex/index where the lag was worst.
            if asset_class in ("forex", "index"):
                try:
                    from app.services.tradfi_prices import get_price as _spot_px
                    _px = await _spot_px(symbol, asset_class)
                    if _px and _px > 0:
                        _now_ms = int(datetime.utcnow().timestamp() * 1000)
                        if (not candles) or _now_ms > candles[-1][0]:
                            candles = list(candles) + [[_now_ms, _px, _px, _px, _px]]
                except Exception as _spe:
                    logger.debug(f"[PaperMonitor] live spot append failed {symbol}: {_spe}")
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
            if isinstance(result, asyncio.CancelledError):
                logger.info("Paper sweep candle fetch cancelled — skipping bucket")
                continue
            if isinstance(result, Exception):
                logger.warning(f"Candle fetch error in paper sweep: {result}")
                continue
            symbol, positions, candles = result
            if not candles:
                # Warn when data is missing for trades that are old enough to evaluate.
                # An empty candle list is normal for very recent trades (< 2 min old)
                # but for older forex/stock trades it means FMP and yfinance both failed.
                for _ex in positions:
                    _elapsed_h = (
                        datetime.utcnow() - (_ex.fired_at or datetime.utcnow())
                    ).total_seconds() / 3600
                    if _elapsed_h >= 0.5:  # only warn if trade is ≥30 min old
                        _ac_label = getattr(_ex, "asset_class", "?")
                        logger.warning(
                            f"[PaperMonitor] EVALUATION_SKIP: exec #{_ex.id} "
                            f"{symbol} ({_ac_label}) — no 1m candle data after "
                            f"{_elapsed_h:.1f}h open. Check FMP/yfinance connectivity."
                        )
            # One DB session per (symbol, asset_class) bucket — not per position.
            from app.database import bg_db_slot as _bg_slot
            try:
                async with _bg_slot():
                    write_db = SessionLocal()
                    try:
                        for ex in positions:
                            try:
                                managed = write_db.merge(ex)

                                # ── Forex day-trading force-close guards ──────────
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

                                            if _risk2.get("friday_close_protection"):
                                                if (_sweep_wd == 4 and _sweep_h >= 21) or _sweep_wd >= 5:
                                                    _forced = True
                                                    _force_reason = "friday_close_protection"

                                            if not _forced and _risk2.get("no_overnight_positions"):
                                                if _sweep_wd < 4 and _sweep_h >= 22:
                                                    _forced = True
                                                    _force_reason = "no_overnight_positions"
                                    except Exception as _fce:
                                        logger.debug(
                                            f"[PaperMonitor] Force-close config read ex#{ex.id}: {_fce}"
                                        )

                                    if _forced:
                                        _fc_price = managed.entry_price
                                        if candles:
                                            try:
                                                _fc_price = float(candles[-1][4])
                                            except Exception:
                                                pass
                                        logger.info(
                                            f"[PaperMonitor] FORCE-CLOSE ({_force_reason}): "
                                            f"exec #{managed.id} {managed.symbol} {managed.direction} "
                                            f"@ {_fc_price} — day-trading rule triggered"
                                        )
                                        _close_paper_execution(managed, "CANCELLED", _fc_price, write_db)
                                        continue

                                if _ex_ac in ("forex", "index"):
                                    try:
                                        from app.strategy_models import UserStrategy as _US_tm
                                        from app.services.trade_management import manage_open_position
                                        _strat_tm = write_db.query(_US_tm).filter(
                                            _US_tm.id == managed.strategy_id
                                        ).first()
                                        _live_px = float(candles[-1][4]) if candles else None
                                        if _strat_tm and _live_px and _live_px > 0:
                                            await manage_open_position(
                                                managed,
                                                _strat_tm.config or {},
                                                _live_px,
                                                write_db,
                                            )
                                    except Exception as _tme:
                                        logger.warning(
                                            "[PaperMonitor] trade-mgmt exec#%s: %s",
                                            managed.id, _tme,
                                        )
                                _evaluate_paper_position_against_candles(managed, candles, write_db)
                            except Exception as e:
                                logger.warning(f"Position {ex.id} eval error: {e}")
                                try:
                                    write_db.rollback()
                                except Exception:
                                    pass
                    finally:
                        write_db.close()
            except _SATimeoutError:
                logger.debug("[PaperMonitor] skipped bucket write — pool busy")
                continue

    async with httpx.AsyncClient() as http_client:
        # ── Startup catch-up: immediately resolve any positions missed while down ──
        try:
            mark_heartbeat("paper_monitor")
            await _sweep(http_client)
        except Exception as e:
            logger.error(f"Startup catch-up sweep error: {e}", exc_info=True)

        while True:
            await asyncio.sleep(PAPER_MONITOR_INTERVAL)
            try:
                mark_heartbeat("paper_monitor")
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
            from app.services.binance_feed import binance_disabled, binance_http_get
            if binance_disabled():
                return
            status, data = await binance_http_get(
                http_client,
                "https://fapi.binance.com/fapi/v1/ticker/price",
                None,
                timeout_s=8,
                caller="strategy_executor._live_monitor_prices.binance",
            )
            if status == 200 and isinstance(data, list):
                for item in data:
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

    logger.info(f"🔴 Live position monitor started ({LIVE_MONITOR_INTERVAL}s interval)")

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

        from app.services.close_notify_dedupe import claim_close_notification
        if not claim_close_notification(db, ex.id):
            return True

        _live_ac = (getattr(ex, "asset_class", "crypto") or "crypto").upper()
        logger.info(
            f"[{_log_ts()}] [live-monitor] "
            f"{'TP' if outcome == 'WIN' else 'SL'} HIT ({source}) [{_live_ac}]: "
            f"{ex.symbol} {ex.direction} entry={ex.entry_price} "
            f"exit={exit_price} pnl={pnl_pct:+.1f}%"
        )

        try:
            user  = db.query(User).filter(User.id == ex.user_id).first()
            strat = db.query(UserStrategy).filter(UserStrategy.id == ex.strategy_id).first()
            strat_name = (strat.name if strat else None) or "Unknown Strategy"
            if user and user.telegram_id:
                tg_id = _telegram_int_id(user)
                if tg_id:
                    await _deliver_close_telegram(
                        tg_id,
                        execution_id=ex.id,
                        symbol=ex.symbol,
                        outcome=outcome,
                        asset_class=getattr(ex, "asset_class", "forex") or "forex",
                        strategy_name=strat_name,
                        direction=ex.direction,
                        entry=ex.entry_price,
                        exit_price=exit_price,
                        pnl_pct=pnl_pct,
                        leverage=leverage,
                        fired_at=ex.fired_at,
                        closed_at=closed_at,
                        conditions=ex.conditions_met,
                        is_paper=False,
                        mfe_pips=getattr(ex, "mfe_pips", None),
                        mae_pips=getattr(ex, "mae_pips", None),
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
        from app.database import bg_db_slot
        from sqlalchemy.exc import TimeoutError as _SATimeoutError

        try:
            async with bg_db_slot():
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
                    user_ids = list({ex.user_id for ex in open_lives}) if open_lives else []
                    prefs_list = (
                        db.query(UserPreference)
                        .filter(UserPreference.user_id.in_(user_ids))
                        .all()
                    ) if user_ids else []
                    db.expunge_all()
                finally:
                    db.close()
        except _SATimeoutError:
            logger.warning("[live-monitor] DB pool busy — skipping sweep this cycle")
            return

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

        note_updates: list = []
        for ex in open_lives:
            live_px = live_prices.get(ex.symbol)
            if not live_px or not ex.entry_price:
                continue
            leverage = ex.leverage or 10
            if ex.direction == "LONG":
                pnl_pct = (live_px - ex.entry_price) / ex.entry_price * 100 * leverage
            else:
                pnl_pct = (ex.entry_price - live_px) / ex.entry_price * 100 * leverage
            note_updates.append((
                ex.id,
                f"open · live={live_px:.6g} · "
                f"unrealised {'+' if pnl_pct >= 0 else ''}{pnl_pct:.1f}%",
            ))
        if note_updates:
            try:
                async with bg_db_slot():
                    note_db = SessionLocal()
                    try:
                        from sqlalchemy import text as _text2
                        for ex_id, note in note_updates:
                            note_db.execute(
                                _text2("UPDATE strategy_executions SET notes=:n WHERE id=:id"),
                                {"n": note, "id": ex_id},
                            )
                        note_db.commit()
                    except Exception:
                        note_db.rollback()
                    finally:
                        note_db.close()
            except _SATimeoutError:
                logger.debug("[live-monitor] skipped unrealised P&L notes — pool busy")

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
                        try:
                            async with bg_db_slot():
                                rec_db = SessionLocal()
                                try:
                                    await _close_live_execution_and_notify(
                                        ex, outcome, exit_price, rec_db, source="bitunix-reconcile"
                                    )
                                finally:
                                    rec_db.close()
                        except _SATimeoutError:
                            logger.debug(
                                f"[live-monitor] skipped reconcile close exec#{ex.id} — pool busy"
                            )

                except Exception as ue:
                    logger.error(f"[live-monitor] Reconcile error user {user_id}: {ue}", exc_info=True)

        # Trim the missing-counter dict to prevent unbounded growth
        if len(_reconcile_missing) > 500:
            _reconcile_missing.clear()

    async with httpx.AsyncClient() as http_client:
        # Startup catch-up
        try:
            mark_heartbeat("live_monitor")
            await _sweep_live(http_client)
        except Exception as e:
            logger.error(f"[live-monitor] startup sweep error: {e}", exc_info=True)
        while True:
            await asyncio.sleep(LIVE_MONITOR_INTERVAL)
            try:
                mark_heartbeat("live_monitor")
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
    prefetched_ctrader_ok: Optional[bool] = None,
):
    """
    Evaluate one strategy. Fires a trade if conditions are met.
    paper=True strategies fire but skip Bitunix order placement.
    raw_tickers — pass the pre-fetched ticker list from the main loop so all
    strategies in one cycle share a single MEXC/Binance fetch instead of each
    making their own request.
    gate_stats — optional shared counter dict for per-cycle diagnostics.
    prefetched_ctrader_ok — for forex/index, the caller may pass the cTrader
    live-eligibility boolean it already resolved (batched once per cycle) so we
    skip a per-strategy UserPreference query. None → resolve it here as before.
    """
    from app.services.strategy_ta import (
        StrategyEvalCancelled,
        evaluate_strategy_conditions,
    )
    from app.strategy_models import StrategyExecution, StrategyPortalSettings

    _strategy_gates: Dict[str, int] = {}

    def _bump(key: str):
        _strategy_gates[key] = _strategy_gates.get(key, 0) + 1
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
        if prefetched_ctrader_ok is not None:
            # Caller already resolved live-eligibility (batched once per cycle).
            _ctrader_live_ok = bool(prefetched_ctrader_ok)
        else:
            _ctrader_live_ok = False
            try:
                from app.models import UserPreference as _UP
                _prefs = db.query(_UP).filter(_UP.user_id == user.id).first()
                _ctrader_live_ok = bool(
                    _prefs
                    and _prefs.ctrader_access_token
                    and _prefs.ctrader_account_id
                    and getattr(_prefs, "forex_approved", False)
                )
            except Exception:
                _ctrader_live_ok = False
        is_paper = not (_wants_live and _ctrader_live_ok)
        if _wants_live and is_paper:
            _ctrader_blockers = []
            try:
                from app.models import UserPreference as _UPb
                _pb = db.query(_UPb).filter(_UPb.user_id == user.id).first()
                if not _pb or not (_pb.ctrader_access_token or "").strip():
                    _ctrader_blockers.append("cTrader not linked")
                elif not (_pb.ctrader_account_id or "").strip():
                    _ctrader_blockers.append("no cTrader account selected")
                elif not getattr(_pb, "forex_approved", False):
                    _ctrader_blockers.append("forex live not approved")
            except Exception:
                _ctrader_blockers.append("cTrader prefs unreadable")
            logger.warning(
                f"[Strategy {strategy.id}] LIVE {asset_class} downgraded to PAPER — "
                f"{'; '.join(_ctrader_blockers) or 'cTrader not ready'} "
                f"(no broker order will be sent; tracked in TradeHub only)"
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
            _db_rollback_safe(db)
            _rebind_session(db, strategy, user)

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

    # ── Forex: weekend gap buffer ─────────────────────────────────────────────
    # Block signal evaluation in the first 90 minutes after the Sunday-night
    # market reopen (22:00–23:30 UTC) and the first 60 minutes of Monday
    # (00:00–00:59 UTC).  Gaps can be large and mislead breakout/momentum
    # signals, so we simply suppress new entries during this window.
    if asset_class == "forex":
        try:
            from app.services.forex_engine import is_weekend_gap_window as _is_gap
            if _is_gap():
                _bump("blk_weekend_gap")
                logger.info(
                    f"[Strategy {strategy.id}] Weekend gap buffer active "
                    f"(Sunday reopen window) — skipping signal evaluation"
                )
                return
        except Exception:
            pass

    # Cooldown is now per-symbol only (handled in the candidate-symbol loop
    # below). The old global cooldown blocked ALL symbols whenever ANY symbol
    # fired — this prevented multi-pair strategies (e.g. EURUSD + GBPUSD)
    # from firing on a second pair while the first was in cooldown.

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
    # One GROUP BY query replaces up to 2×N per-symbol round-trips.
    _sym_slice = symbols[:50]
    _fired_today_set, _last_fired_map = _prefetch_symbol_cooldowns(
        strategy.id, _sym_slice, db, need_today=no_duplicate_symbol,
    )
    candidate_symbols: List[str] = []
    for symbol in _sym_slice:
        if no_duplicate_symbol and symbol in _fired_today_set:
            continue
        last_fired = _last_fired_map.get(symbol)
        if last_fired:
            elapsed_mins = (datetime.utcnow() - last_fired).total_seconds() / 60
            if elapsed_mins < cooldown_mins:
                continue
        candidate_symbols.append(symbol)

    if not candidate_symbols:
        _bump("blk_all_in_cooldown")
        return

    # ── Session filter (forex/tradfi entry only) ─────────────────────────────
    if asset_class in ("forex", "index", "stock"):
        from app.services.session_filter import is_in_allowed_session

        _sess_ok, _ = is_in_allowed_session(config, datetime.utcnow())
        if not _sess_ok:
            _bump("blk_session_filter")
            from app.services.feed_diagnostics import log_scan_metric as _lsm_sess
            _eval_tf_sess = _primary_timeframe(config)
            for _sym_blk in candidate_symbols:
                _lsm_sess(
                    symbol=_sym_blk,
                    timeframe=_eval_tf_sess,
                    candles_loaded=0,
                    strategy_evaluated=True,
                    setup_detected=False,
                    signal_generated=False,
                    signal_sent=False,
                    strategy_id=strategy.id,
                    block_reason="session_filter",
                )
            return

    # ── Step 2: Parallel price + TA fetch for ALL candidates at once ──────────
    # Turns O(n × fetch_time) → O(fetch_time).  Results are also stored in the
    # shared _PRICE_TA_CACHE so subsequent calls inside evaluate_strategy_conditions
    # (which fetches its own klines) benefit from the same warm cache.
    _eval_tf = _primary_timeframe(config)
    _uid = user.id if user else None
    _price_results = await asyncio.gather(
        *[
            _fetch_price_and_ta(
                sym, http_client, asset_class,
                user_id=_uid, timeframe=_eval_tf,
                metal_paper_ok=is_paper and asset_class == "forex",
            )
            for sym in candidate_symbols
        ],
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
    from app.services.feed_diagnostics import log_scan_metric

    _had_any_candidate = False
    _conditions_failed_for_all = True
    for symbol in candidate_symbols:
        price_data = price_map.get(symbol)
        if not price_data:
            log_scan_metric(
                symbol=symbol,
                timeframe=_eval_tf,
                candles_loaded=0,
                strategy_evaluated=True,
                setup_detected=False,
                signal_generated=False,
                signal_sent=False,
                strategy_id=strategy.id,
                block_reason="no_price_data",
            )
            continue
        _had_any_candidate = True
        _candles_n = int(price_data.get("candles_loaded") or 0)

        if asset_class in ("forex", "index", "stock"):
            from app.services.economic_calendar import blocking_event as _news_block
            _news_evt = _news_block(symbol, config, datetime.utcnow())
            if _news_evt:
                _bump("blk_news_filter")
                log_scan_metric(
                    symbol=symbol,
                    timeframe=_eval_tf,
                    candles_loaded=_candles_n,
                    strategy_evaluated=True,
                    setup_detected=False,
                    signal_generated=False,
                    signal_sent=False,
                    strategy_id=strategy.id,
                    block_reason="news_filter",
                    extra={"event_name": _news_evt},
                )
                continue

        if filters.get("htf_trend") and not htf_pass.get(symbol):
            logger.debug(f"[Strategy {strategy.id}] HTF trend filter blocked {symbol}")
            _bump("blk_htf_trend_sym")
            log_scan_metric(
                symbol=symbol,
                timeframe=_eval_tf,
                candles_loaded=_candles_n,
                strategy_evaluated=True,
                setup_detected=False,
                signal_generated=False,
                signal_sent=False,
                strategy_id=strategy.id,
                block_reason="htf_trend",
            )
            continue

        enhanced_ta   = price_data.get("enhanced_ta", {})
        current_price = price_data.get("price", 0)
        if not current_price:
            continue

        # Directional entry: LONG at ask, SHORT at bid when ticks available.
        _side_price = current_price
        if direction_pref == "LONG" and price_data.get("ask"):
            _side_price = price_data["ask"]
        elif direction_pref == "SHORT" and price_data.get("bid"):
            _side_price = price_data["bid"]
        current_price = _side_price

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
                                f"[{_log_ts()}] [Strategy {strategy.id}] {symbol} spread "
                                f"{_spread_pips:.1f} pips > max {_max_sp} — skipping"
                            )
                            continue
                except Exception as _sfe:
                    logger.debug(f"[Strategy {strategy.id}] spread filter error: {_sfe}")

        if price_data.get("kline_synthetic") or (
            (price_data.get("kline_source") or "").lower() == "synthetic"
        ):
            _bump("blk_synthetic_bars")
            logger.info(
                f"[Strategy {strategy.id}] {symbol}: synthetic kline bars — "
                f"blocking TA/signal eval (price reference only)"
            )
            log_scan_metric(
                symbol=symbol,
                timeframe=_eval_tf,
                candles_loaded=_candles_n,
                strategy_evaluated=True,
                setup_detected=False,
                signal_generated=False,
                signal_sent=False,
                strategy_id=strategy.id,
                block_reason="synthetic_bars",
            )
            continue

        try:
            passed, details = await evaluate_strategy_conditions(
                config, symbol, price_data, enhanced_ta, http_client,
                strictness_level=strictness_level,
                ctrader_user_id=_uid,
            )
        except StrategyEvalCancelled:
            logger.info(
                f"[Strategy {strategy.id}] eval cancelled — skipping cycle"
            )
            return
        _maybe_log_ta_eval(strategy, symbol, direction_pref, passed, details)
        if not passed:
            log_scan_metric(
                symbol=symbol,
                timeframe=_eval_tf,
                candles_loaded=_candles_n,
                strategy_evaluated=True,
                setup_detected=False,
                signal_generated=False,
                signal_sent=False,
                strategy_id=strategy.id,
                block_reason="ta_conditions",
            )
            continue
        _conditions_failed_for_all = False
        log_scan_metric(
            symbol=symbol,
            timeframe=_eval_tf,
            candles_loaded=_candles_n,
            strategy_evaluated=True,
            setup_detected=True,
            signal_generated=False,
            signal_sent=False,
            strategy_id=strategy.id,
        )
        # break out of "failure tracking" — we have a real fire below

        # ── Pre-fire entry price confirmation (metals + tradfi) ─────────────
        if asset_class in ("forex", "index", "stock"):
            from app.services.tradfi_prices import confirm_entry_price as _confirm_px
            _confirmed_px, _confirm_reason = await _confirm_px(
                symbol,
                asset_class,
                current_price,
                paper_ok=is_paper,
                user_id=_uid,
            )
            if _confirmed_px is None:
                _bump("blk_entry_price_stale")
                logger.warning(
                    f"[{_log_ts()}] [Strategy {strategy.id}] {symbol} fire blocked — "
                    f"{_confirm_reason} (scan_price={current_price:.4f})"
                )
                log_scan_metric(
                    symbol=symbol,
                    timeframe=_eval_tf,
                    candles_loaded=_candles_n,
                    strategy_evaluated=True,
                    setup_detected=True,
                    signal_generated=False,
                    signal_sent=False,
                    strategy_id=strategy.id,
                    block_reason="entry_price_stale",
                )
                continue
            if abs(_confirmed_px - current_price) / max(_confirmed_px, 1e-9) > 0.00005:
                logger.info(
                    f"[{_log_ts()}] [Strategy {strategy.id}] {symbol} entry adjusted "
                    f"{current_price:.4f} → {_confirmed_px:.4f} ({_confirm_reason})"
                )
            current_price = _confirmed_px

        mode_tag = "🧪 [PAPER]" if is_paper else "🎯"
        _ac_tag = asset_class.upper()
        logger.info(
            f"[{_log_ts()}] {mode_tag} [{_ac_tag}] [Strategy {strategy.id}] "
            f"{strategy.name} — {symbol} @ {current_price:.4f} "
            f"conditions met! {direction_pref}"
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
        if asset_class in ("forex", "index"):
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
            # Infer the trade side from the directional signal itself — FVG/iFVG,
            # order block, divergence/COD — and only fall back to RSI when the
            # strategy carries no directional signal at all.
            #
            #   FVG : bullish gap → LONG,  bearish gap → SHORT.
            #   iFVG: an inverted gap flips the bias — a violated bullish gap
            #         becomes resistance → SHORT, a violated bearish gap becomes
            #         support → LONG.
            #
            # When the condition's direction is "any" the gap side isn't pinned in
            # config, so we read the side actually detected on THIS fire from the
            # matching evaluation detail (eval_fvg labels them "Bullish/Bearish FVG").
            #
            # `details` is index-aligned with entry_conditions.conditions and each
            # line is prefixed "✅"/"❌". We only ever infer direction from a
            # condition that PASSED — critical for OR strategies, where a failed
            # FVG line must never drive the trade side.
            _entry_conds = config.get("entry_conditions", {}).get("conditions", [])

            def _passed_at(_i):
                return _i < len(details) and str(details[_i]).lstrip().startswith("✅")

            def _gap_bias_at(_i):
                if _i >= len(details):
                    return None
                _dl = str(details[_i]).lower()
                if "bullish fvg" in _dl:
                    return "bullish"
                if "bearish fvg" in _dl:
                    return "bearish"
                return None

            def _norm(_v):
                return _v.strip().lower() if isinstance(_v, str) else _v

            inferred_dir = None
            for _i, _cond in enumerate(_entry_conds):
                if not _passed_at(_i):
                    continue
                _ct = _cond.get("type", "")
                if _ct in ("fvg", "ifvg"):
                    _bias = _norm(_cond.get("direction") or _cond.get("fvg_dir"))
                    if not _bias or _bias in ("any", "both"):
                        _bias = _gap_bias_at(_i)
                    if _bias in ("bullish", "bearish"):
                        if _ct == "ifvg":
                            inferred_dir = "SHORT" if _bias == "bullish" else "LONG"
                        else:
                            inferred_dir = "LONG" if _bias == "bullish" else "SHORT"
                        break
                elif _ct in ("order_block", "ob"):
                    _d = _norm(_cond.get("ob_type") or _cond.get("direction"))
                    if _d and _d not in ("any", "both"):
                        inferred_dir = "LONG" if _d == "bullish" else "SHORT"
                        break
                elif _ct in ("divergence", "cod", "change_of_direction"):
                    _d = _norm(_cond.get("direction"))
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

        # Stash partial_close_pct in notes so the paper monitor + live manager can
        # act on it. When a FOREX strategy defines a second target (TP2) but the
        # user never set an explicit partial size, DEFAULT to closing half at TP1
        # → move the stop to breakeven → run the remainder to TP2 (the standard
        # "TP1 + runner"). Scoped to forex because the live partial-close + BE flow
        # is only wired for cTrader forex; crypto keeps full-close-at-TP1 so paper
        # == live there.
        _partial_close_pct = ex_config.get("partial_close_pct")
        if (
            (not _partial_close_pct or float(_partial_close_pct) <= 0)
            and tp2_price
            and asset_class == "forex"
        ):
            _partial_close_pct = 50.0
        _exec_notes_parts: List[str] = []
        if _partial_close_pct and float(_partial_close_pct) > 0:
            _exec_notes_parts.append(f"partial_close_pct={float(_partial_close_pct):.0f}")
        _ps = price_data.get("price_source") or "unknown"
        _exec_notes_parts.append(f"entry_src={_ps}")
        _kc = price_data.get("kline_close")
        if _kc is not None:
            try:
                _exec_notes_parts.append(f"kline_close={float(_kc):.4f}")
            except (TypeError, ValueError):
                pass
        _ks = price_data.get("kline_source")
        if _ks:
            _exec_notes_parts.append(f"kline_src={_ks}")
        if sl_price is not None:
            _exec_notes_parts.append(f"orig_sl={float(sl_price):.6g}")
        _exec_notes = " | ".join(_exec_notes_parts) if _exec_notes_parts else None

        execution = StrategyExecution(
            strategy_id    = strategy.id,
            user_id        = user.id,
            symbol         = symbol,
            direction      = direction,
            entry_price    = current_price,
            tp_price       = tp_price,
            tp2_price      = tp2_price,
            sl_price       = sl_price,
            current_sl     = sl_price,
            leverage       = leverage,
            outcome        = "OPEN",
            conditions_met = details,
            fired_at       = datetime.utcnow(),
            is_paper       = is_paper,
            asset_class    = asset_class,
            notes          = _exec_notes,
        )
        db.add(execution)
        await _commit_db_with_retry(
            db, label=f"exec#{strategy.id}", instances=(execution,)
        )
        execution = _refresh_db_row(db, execution)
        if execution is None or not getattr(execution, "id", None):
            logger.error(
                f"[Strategy {strategy.id}] execution row missing after commit — "
                f"aborting fire for {symbol}"
            )
            return

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
        await _commit_db_with_retry(
            db, label=f"perf#{strategy.id}", instances=(_perf,)
        )

        if is_paper:
            # Paper trade: notify user and skip Bitunix
            try:
                portal_settings = db.query(StrategyPortalSettings).filter(StrategyPortalSettings.user_id == user.id).first()
                tg_id = _telegram_int_id(user)
                if tg_id and _should_dm_trade_alerts(portal_settings, True) \
                        and _claim_tg_open_notify(db, execution.id):
                    _schedule_tg_open_notify(
                        execution.id,
                        tg_id,
                        _fmt_open_card(
                            strategy_name = strategy.name or "Your Strategy",
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
                        asset_class=asset_class,
                        symbol=symbol,
                    )
                elif not tg_id:
                    logger.info(
                        "[%s] [Strategy %s] Paper fire %s — no Telegram "
                        "(link Telegram in Settings)",
                        _log_ts(),
                        strategy.id,
                        symbol,
                    )
                # Mobile push (fire-and-forget; never raises)
                from app.services.expo_push import notify_user_bg
                _coin = symbol.replace("USDT", "")
                notify_user_bg(
                    user.id,
                    title=f"📝 Paper · {_coin} {direction} {leverage}×",
                    body=f"{strategy.name} @ ${current_price:,.4f}",
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
            _position_id = None
            _acct_id    = None
            _order_err  = None
            _broker     = "ctrader" if asset_class in ("forex", "index") else "bitunix"
            # Partial-runner mode (forex only): the broker holds the position to TP2
            # (final target) and the live manager closes half at TP1 + moves the
            # stop to breakeven. Placing the broker TP at TP1 would fully close the
            # position there and kill the runner, so we send TP2 as the broker TP.
            _partial_mode = bool(
                asset_class == "forex"
                and tp2_price
                and _partial_close_pct and float(_partial_close_pct) > 0
            )
            try:
                ps_type      = risk.get("position_size_type", "pct")
                _risk_usd    = float(risk["position_size_usd"]) if ps_type == "fixed" and risk.get("position_size_usd") else None
                if _broker == "ctrader":
                    # Pre-fire drift guard — skip stale scan/signal prices.
                    try:
                        from app.services.ctrader_price_feed import get_price as _feed_px
                        from app.services.forex_engine import pip_size as _psz_drift
                        _fresh = _feed_px(symbol)
                        _max_drift = float(risk.get("max_entry_drift_pips") or 15)
                        if _fresh and _max_drift > 0:
                            _drift_pips = abs(_fresh - current_price) / max(_psz_drift(symbol), 1e-10)
                            if _drift_pips > _max_drift:
                                _bump("blk_entry_drift")
                                execution.outcome = "CANCELLED"
                                execution.notes = f"Cancelled: price drift {_drift_pips:.1f} pips"
                                db.commit()
                                logger.info(
                                    f"[{_log_ts()}] [Strategy {strategy.id}] {symbol} drift "
                                    f"{_drift_pips:.1f}p > {_max_drift} — skipped"
                                )
                                break
                    except Exception:
                        pass

                    from app.services.ctrader_order_queue import (
                        CtraderOrderJob, enqueue_ctrader_order, start_ctrader_order_worker,
                    )
                    from app.services.order_latency import new_order_latency
                    start_ctrader_order_worker()
                    _signal_mono = time.monotonic()
                    # Risk % auto lot sizing: when use_risk_pct=True, the wizard
                    # stored risk_pct_per_trade (% of account to risk).  We pass
                    # it to the cTrader helper which fetches the account balance
                    # and computes lots = risk% × balance / (sl_pips × pip_value).
                    # When the user picked an explicit lot size (position_size_type
                    # == 'lots'), fixed_lots takes priority over every other mode.
                    _use_risk_pct = bool(risk.get("use_risk_pct"))
                    _risk_pct_per = float(risk.get("risk_pct_per_trade") or risk.get("position_size_pct") or 1.0)
                    _fixed_lots   = float(risk.get("position_size_lots") or 0) if ps_type == "lots" else 0.0
                    # In partial-runner mode the broker TP must be TP2 (final target);
                    # otherwise it stays at the strategy's TP1.
                    _live_tp_pct  = float(tp2_pct) if (_partial_mode and tp2_pct) else tp_pct
                    _job = CtraderOrderJob(
                        user_id=user.id,
                        strategy_id=strategy.id,
                        execution_id=execution.id,
                        symbol=symbol,
                        direction=direction,
                        entry_price=current_price,
                        tp_pct=_live_tp_pct,
                        sl_pct=sl_pct,
                        risk_pct=_risk_pct_per,
                        risk_usd=_risk_usd,
                        use_risk_pct=_use_risk_pct,
                        sl_pips=float(ex_config.get("stop_loss_pips") or 0) or None,
                        fixed_lots=_fixed_lots or None,
                        asset_class=asset_class,
                        partial_close_pct=float(_partial_close_pct) if _partial_close_pct else None,
                        signal_mono=_signal_mono,
                        latency=new_order_latency(execution.id, _signal_mono),
                    )
                    _queued = await enqueue_ctrader_order(_job)
                    if _queued:
                        logger.info(
                            f"[{_log_ts()}] [Strategy {strategy.id}] cTrader order queued "
                            f"exec#{execution.id} {symbol} {direction}"
                        )
                        order_result = {"order_id": "queued", "queued": True}
                    else:
                        logger.error(
                            f"[{_log_ts()}] [Strategy {strategy.id}] cTrader order queue "
                            f"FULL — exec#{execution.id} {symbol} not sent to broker"
                        )
                        order_result = None
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
                    _position_id = order_result.get("position_id")
                    _acct_id    = order_result.get("account_id")
                    _order_err  = order_result.get("error")
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
                        title=f"⚠️ Live failed → Paper · {_coin} {direction}",
                        body=f"{strategy.name} · {leverage}×",
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

            if (
                not is_paper
                and _broker == "ctrader"
                and not order_result
            ):
                execution.is_paper = True
                execution.notes = (
                    (execution.notes or "")
                    + " | Live→Paper fallback (cTrader order not queued)"
                ).strip(" |")
                await _commit_db_with_retry(
                    db,
                    label=f"ctrader-queue-fail#{execution.id}",
                    instances=(execution,),
                )
                execution = _refresh_db_row(db, execution)
                logger.warning(
                    f"[Strategy {strategy.id}] {symbol} live exec#{execution.id} — "
                    f"cTrader order never queued; tracking as paper"
                )
                tg_id_ex = _telegram_int_id(user)
                if tg_id_ex:
                    try:
                        await _tg_send(
                            tg_id_ex,
                            f"⚠️ <b>cTrader order not placed — paper trade started</b>\n"
                            f"Strategy: <b>{strategy.name}</b>\n"
                            f"Signal: {symbol} {direction}\n\n"
                            f"<i>The live order could not be queued for cTrader "
                            f"(broker busy or queue full). This position is tracked "
                            f"as 🧪 paper only — nothing was sent to your broker.</i>",
                            asset_class=asset_class,
                        )
                    except Exception:
                        pass
                break

            if order_result and order_result.get("queued"):
                execution.notes = ((execution.notes or "") + " | order_queued").strip(" |")
                await _commit_db_with_retry(
                    db,
                    label=f"ctrader-queued#{execution.id}",
                    instances=(execution,),
                )
                execution = _refresh_db_row(db, execution)
                tg_id_live = _telegram_int_id(user)
                _portal_live = db.query(StrategyPortalSettings).filter(
                    StrategyPortalSettings.user_id == user.id
                ).first()
                # Attempt only — full LIVE card is sent after broker fill confirmation
                # in ctrader_order_queue (never claim tg_open_sent here).
                if tg_id_live and _should_dm_trade_alerts(_portal_live, False):
                    asyncio.create_task(_tg_send(
                        tg_id_live,
                        _fmt_queued_open_notice(
                            strategy.name or "Your Strategy",
                            symbol,
                            direction,
                            leverage=leverage,
                        ),
                        asset_class=asset_class,
                    ))
                try:
                    from app.services.expo_push import notify_user_bg
                    notify_user_bg(
                        user.id,
                        title=f"🎯 Live · {symbol.replace('USDT','')} {direction}",
                        body=f"{strategy.name} — placing on cTrader",
                        data={"type": "trade_open", "strategy_id": strategy.id, "kind": "live"},
                        kind="live",
                    )
                except Exception:
                    pass
                if not config.get("_locked"):
                    asyncio.create_task(_propagate_to_subscribers(
                        source_strategy_id=strategy.id,
                        source_execution_id=execution.id,
                        http_client=http_client,
                    ))
                break

            if order_id:
                if _broker == "ctrader":
                    execution.ctrader_order_id = str(order_id)
                    # Persist the broker positionId so the live forex manager can
                    # later amend SL/TP (auto-breakeven + trailing). No dedicated
                    # column exists → stash it as a "pos=<id>" token in notes.
                    if _position_id:
                        execution.ctrader_position_id = str(_position_id)
                        if _acct_id:
                            execution.ctrader_account_id = str(_acct_id)
                        try:
                            _v = int(order_result.get("volume") or 0)
                            if _v > 0:
                                execution.broker_volume_units = _v
                        except Exception:
                            pass
                        _n = (execution.notes or "").strip()
                        _acct_tok = f" | acct={_acct_id}" if _acct_id else ""
                        _vol_tok = ""
                        try:
                            _v = int(order_result.get("volume") or 0)
                            if _v > 0:
                                _vol_tok = f" | vol={_v}"
                        except Exception:
                            _vol_tok = ""
                        execution.notes = (f"{_n} | pos={_position_id}{_acct_tok}{_vol_tok}".strip(" |"))
                else:
                    execution.bitunix_order_id = str(order_id)
                if (
                    actual_fill
                    and actual_fill > 0
                    and execution.entry_price
                    and abs(actual_fill - execution.entry_price) > execution.entry_price * 1e-7
                ):
                    # The broker filled at actual_fill (not our pre-fill signal
                    # price), and it enforces SL/TP as RELATIVE offsets applied to
                    # that real fill. Shift entry AND SL/TP/TP2 by the same delta
                    # so the card the user sees + our paper-style monitoring match
                    # what the broker is actually holding (otherwise the card shows
                    # the signal price and SL/TP that are ~slippage pips off).
                    _delta = actual_fill - execution.entry_price
                    logger.info(
                        f"[{_log_ts()}] [Strategy {strategy.id}] entry/SL/TP shifted to fill: "
                        f"signal={execution.entry_price:.6g} → fill={actual_fill:.6g} "
                        f"(Δ={_delta:+.6g})"
                    )
                    execution.entry_price = actual_fill
                    if execution.sl_price:
                        execution.sl_price += _delta
                        sl_price = execution.sl_price
                    if execution.tp_price:
                        execution.tp_price += _delta
                        tp_price = execution.tp_price
                    if execution.tp2_price:
                        execution.tp2_price += _delta
                        tp2_price = execution.tp2_price
                db.commit()
                display_entry = actual_fill if actual_fill else current_price
                tg_id_live = _telegram_int_id(user)
                if tg_id_live and _claim_tg_open_notify(db, execution.id):
                    try:
                        # Fire-and-forget so a slow/retrying Telegram send never
                        # delays the firing cycle (a latency source).
                        asyncio.create_task(_tg_send(
                            tg_id_live,
                            _fmt_open_card(
                                strategy_name = strategy.name or "Your Strategy",
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
                        ))
                        # Mobile push (fire-and-forget) — fires for live trades.
                        from app.services.expo_push import notify_user_bg
                        _coin = symbol.replace("USDT", "")
                        notify_user_bg(
                            user.id,
                            title=f"🚀 Live · {_coin} {direction} {leverage}×",
                            body=f"{strategy.name} @ ${display_entry:,.4f}",
                            data={"type": "trade_open", "strategy_id": strategy.id, "kind": "live"},
                            kind="live",
                            position_usd=float(risk.get("position_size_usd") or 0) or None,
                        )
                    except Exception as e:
                        logger.warning(f"Live DM failed: {e}")
            else:
                # Fallback: order_id was None (shouldn't normally reach here post-refactor)
                execution.is_paper = True
                _reason_note = f" ({_order_err})" if _order_err else ""
                execution.notes    = f"Live→Paper fallback: {_broker} returned no order_id{_reason_note}"
                db.commit()
                logger.warning(
                    f"[{_log_ts()}] [Strategy {strategy.id}] Live order for {symbol} "
                    f"returned no order_id ({_order_err or 'no reason'}) — "
                    f"converting execution #{execution.id} to paper trade for ROI tracking."
                )
                tg_id_live = _telegram_int_id(user)
                if tg_id_live:
                    # Surface the broker's actual rejection reason when we have it,
                    # so the user can fix it (bad symbol, trading disabled, etc.)
                    # instead of guessing about API permissions.
                    _reason_line = (
                        f"<i>Reason: {_order_err}</i>\n\n"
                        if _order_err else ""
                    )
                    try:
                        await _tg_send(
                            tg_id_live,
                            f"⚠️ <b>{_broker.title()} order not confirmed — paper trade started</b>\n"
                            f"Strategy: <b>{strategy.name}</b>\n"
                            f"Signal: {symbol.replace('USDT','')} {direction} {leverage}× lev\n"
                            f"Entry: <code>${current_price:,.4f}</code>\n"
                            f"TP: <code>${tp_price:,.4f}</code> (+{tp_pct}%)  "
                            f"SL: <code>${sl_price:,.4f}</code> (-{sl_pct}%)\n\n"
                            f"{_reason_line}"
                            f"<i>The signal is being tracked as a 🧪 paper position. "
                            f"If this keeps happening, check the symbol is tradable on your "
                            f"account and that API trading is enabled.</i>"
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

        log_scan_metric(
            symbol=symbol,
            timeframe=_eval_tf,
            candles_loaded=_candles_n,
            strategy_evaluated=True,
            setup_detected=True,
            signal_generated=True,
            signal_sent=bool(_telegram_int_id(user)),
            strategy_id=strategy.id,
        )
        break  # one trade per strategy per scan cycle
    else:
        # for-loop completed without break → no fire happened.
        # Bump a gate so we know WHY the strategy didn't fire.
        if not _had_any_candidate:
            _bump("blk_no_price_data")
        elif _conditions_failed_for_all:
            _bump("blk_ta_conditions")

    try:
        from app.services.ctrader_order_queue import record_gate_stats
        record_gate_stats(strategy.id, _strategy_gates, persist_db=False)
    except Exception:
        pass


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

            if not _portal_trade_entitled(sub_user, db=_sub_db):
                continue
            _now = datetime.utcnow()

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
                        _can_live = bool(
                            _sub_prefs
                            and _sub_prefs.ctrader_access_token
                            and _sub_prefs.ctrader_account_id
                            and getattr(_sub_prefs, "forex_approved", False)
                        )
                        _live_reason = "ok" if _can_live else "no_ctrader_credentials_or_not_approved"
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
                if tg_id and (not portal_settings or portal_settings.dm_paper_alerts) \
                        and _claim_tg_open_notify(_sub_db, sub_exec.id):
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
                                asset_class   = _sub_asset_class,
                            ),
                        )
                    except Exception as _e:
                        logger.warning(f"[Propagate] Paper DM failed for strategy {sub_strategy.id}: {_e}")
            else:
                # Live — route by asset class: forex/index → cTrader, else → Bitunix.
                order_id    = None
                actual_fill = None
                _position_id = None
                _acct_id    = None
                _sub_broker = "ctrader" if _sub_asset_class in ("forex", "index") else "bitunix"
                try:
                    ps_type      = sub_risk.get("position_size_type", "pct")
                    _sub_risk_usd = float(sub_risk["position_size_usd"]) if ps_type == "fixed" and sub_risk.get("position_size_usd") else None
                    _sub_fixed_lots = float(sub_risk.get("position_size_lots") or 0) if ps_type == "lots" else 0.0
                    if _sub_broker == "ctrader":
                        from app.services.ctrader_order_queue import (
                            CtraderOrderJob, enqueue_ctrader_order, start_ctrader_order_worker,
                        )
                        from app.services.order_latency import new_order_latency
                        start_ctrader_order_worker()
                        _sub_signal_mono = time.monotonic()
                        _sub_use_risk_pct = bool(sub_risk.get("use_risk_pct"))
                        _sub_risk_pct = float(
                            sub_risk.get("risk_pct_per_trade")
                            or sub_risk.get("position_size_pct")
                            or 1.0
                        )
                        _job = CtraderOrderJob(
                            user_id=sub_user.id,
                            strategy_id=sub_strategy.id,
                            execution_id=sub_exec.id,
                            symbol=symbol,
                            direction=direction,
                            entry_price=entry,
                            tp_pct=tp_pct_raw,
                            sl_pct=sl_pct_raw,
                            risk_pct=_sub_risk_pct,
                            risk_usd=_sub_risk_usd,
                            use_risk_pct=_sub_use_risk_pct,
                            sl_pips=float((sub_config.get("exit") or {}).get("stop_loss_pips") or 0) or None,
                            fixed_lots=_sub_fixed_lots or None,
                            asset_class=_sub_asset_class,
                            signal_mono=_sub_signal_mono,
                            latency=new_order_latency(sub_exec.id, _sub_signal_mono),
                        )
                        _queued = await enqueue_ctrader_order(_job)
                        order_result = {"order_id": "queued", "queued": True} if _queued else None
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
                        _position_id = order_result.get("position_id")
                        _acct_id    = order_result.get("account_id")
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

                if order_result and order_result.get("queued"):
                    sub_exec.notes = ((sub_exec.notes or "") + " | order_queued").strip(" |")
                    _sub_db.commit()
                    if tg_id:
                        asyncio.create_task(_tg_send(
                            tg_id,
                            _fmt_queued_open_notice(
                                sub_strategy.name,
                                symbol,
                                direction,
                                leverage=leverage,
                            ),
                            asset_class=_sub_asset_class,
                        ))
                    logger.info(
                        f"[Propagate] Strategy {sub_strategy.id} (user {sub_user.username}) "
                        f"{symbol} {direction} @ {entry} — queued on cTrader"
                    )
                    continue

                if order_id:
                    if _sub_broker == "ctrader":
                        sub_exec.ctrader_order_id = str(order_id)
                        if _position_id:
                            sub_exec.ctrader_position_id = str(_position_id)
                            if _acct_id:
                                sub_exec.ctrader_account_id = str(_acct_id)
                            _sn = (sub_exec.notes or "").strip()
                            _acct_tok = f" | acct={_acct_id}" if _acct_id else ""
                            sub_exec.notes = (f"{_sn} | pos={_position_id}{_acct_tok}".strip(" |"))
                    else:
                        sub_exec.bitunix_order_id = str(order_id)
                    if (
                        actual_fill
                        and actual_fill > 0
                        and sub_exec.entry_price
                        and abs(actual_fill - sub_exec.entry_price) > sub_exec.entry_price * 1e-7
                    ):
                        # Broker filled at actual_fill and enforces SL/TP as relative
                        # offsets from it — shift entry AND SL/TP/TP2 by the same
                        # delta so the subscriber's card + monitoring match the broker.
                        _delta = actual_fill - sub_exec.entry_price
                        logger.info(
                            f"[Propagate] Strategy {sub_strategy.id} entry/SL/TP shifted to fill: "
                            f"signal={sub_exec.entry_price:.6g} → fill={actual_fill:.6g} (Δ={_delta:+.6g})"
                        )
                        sub_exec.entry_price = actual_fill
                        if sub_exec.sl_price:
                            sub_exec.sl_price += _delta
                            sl_price = sub_exec.sl_price
                        if sub_exec.tp_price:
                            sub_exec.tp_price += _delta
                            tp_price = sub_exec.tp_price
                        if sub_exec.tp2_price:
                            sub_exec.tp2_price += _delta
                            tp2_price = sub_exec.tp2_price
                    _sub_db.commit()
                    display_entry = actual_fill if actual_fill else entry
                    if tg_id and _claim_tg_open_notify(_sub_db, sub_exec.id):
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

def _symbols_for_snapshot(snap: dict, max_symbols: int = 20) -> List[str]:
    """Universe symbols for prefetch — capped so wide rosters don't explode HTTP."""
    cfg = snap.get("config") or {}
    raw = (cfg.get("universe") or {}).get("symbols") or []
    out: List[str] = []
    for s in raw:
        if not isinstance(s, str) or not s.strip():
            continue
        sym = s.upper().replace("/", "").replace("-", "")
        if sym and sym not in out:
            out.append(sym)
        if len(out) >= max_symbols:
            break
    return out


def _load_strategy_snapshots_cached(SessionLocal, UserStrategy) -> List[dict]:
    """Return active+paper strategy snapshots; DB hit at most once per TTL."""
    global _STRATEGY_SNAPSHOT_CACHE, _STRATEGY_SNAPSHOT_AT
    now = time.monotonic()
    if (
        _STRATEGY_SNAPSHOT_CACHE
        and (now - _STRATEGY_SNAPSHOT_AT) < _STRATEGY_SNAPSHOT_TTL_S
    ):
        return _STRATEGY_SNAPSHOT_CACHE
    db = SessionLocal()
    try:
        strategies = (
            db.query(UserStrategy)
            .filter(UserStrategy.status.in_(["active", "paper"]))
            .all()
        )
        snapshots = [
            {
                "id": s.id,
                "name": s.name,
                "status": s.status,
                "config": s.config,
                "user_id": s.user_id,
                "_obj": s,
            }
            for s in strategies
        ]
    finally:
        db.close()
    _STRATEGY_SNAPSHOT_CACHE = snapshots
    _STRATEGY_SNAPSHOT_AT = now
    return snapshots


def _preload_strategy_users(
    snap_ids: List[int],
    SessionLocal,
    UserStrategy,
    User,
) -> Dict[int, Tuple]:
    """Batch-load strategy+user rows for a scan cycle (one query)."""
    if not snap_ids:
        return {}
    db = SessionLocal()
    try:
        rows = (
            db.query(UserStrategy, User)
            .join(User, User.id == UserStrategy.user_id)
            .filter(UserStrategy.id.in_(snap_ids))
            .all()
        )
        return {strategy.id: (strategy, user) for strategy, user in rows}
    finally:
        db.close()


async def _prefetch_price_ta_for_cycle(
    snapshots: list,
    http_client: httpx.AsyncClient,
    allowed_asset_classes: set,
    label: str = "Executor",
    ctrader_user_id: Optional[int] = None,
) -> int:
    """
    Warm _PRICE_TA_CACHE + tradfi kline cache for every unique symbol in this
    cycle BEFORE per-strategy evaluate_and_fire. All symbols fetch concurrently
    (semaphore ~10) with a hard per-symbol budget — dead providers never block.
    """
    from app.services.prefetch_fast import (
        SYMBOL_BUDGET_S,
        prefetch_fast_context,
    )

    jobs: List[Tuple[str, str, str]] = []
    seen: set = set()
    for snap in snapshots:
        ac = _snap_asset_class(snap)
        if ac not in allowed_asset_classes:
            continue
        tf = _primary_timeframe(snap.get("config") or {})
        for sym in _symbols_for_snapshot(snap):
            key = (sym, ac, tf)
            if key in seen:
                continue
            seen.add(key)
            jobs.append(key)
    if not jobs:
        return 0

    sem = asyncio.Semaphore(EXECUTOR_PREFETCH_CONCURRENT)
    t0 = time.monotonic()
    stats = {"cache": 0, "fetched": 0, "failed": 0}

    def _cache_fresh(sym: str, ac: str, tf: str) -> bool:
        cache_key = f"{ac}:{sym}:{tf}" if ac != "crypto" else sym
        cached = _PRICE_TA_CACHE.get(cache_key)
        if not cached:
            return False
        _data, fetched_at = cached
        _ttl = _PRICE_TA_TTL if ac == "crypto" else _PRICE_TA_TTL_TRADFI
        return (datetime.utcnow() - fetched_at).total_seconds() < _ttl

    async def _warm(sym: str, ac: str, tf: str) -> None:
        if _cache_fresh(sym, ac, tf):
            stats["cache"] += 1
            return
        async with sem:
            try:
                async with prefetch_fast_context():
                    result = await asyncio.wait_for(
                        _fetch_price_and_ta(
                            sym, http_client, ac,
                            user_id=ctrader_user_id,
                            timeframe=tf,
                            prefetch=True,
                        ),
                        timeout=SYMBOL_BUDGET_S,
                    )
                if result:
                    stats["fetched"] += 1
                else:
                    stats["failed"] += 1
            except Exception:
                stats["failed"] += 1

    await asyncio.gather(*[_warm(s, a, t) for s, a, t in jobs], return_exceptions=True)
    elapsed = time.monotonic() - t0
    logger.info(
        f"[{_log_ts()}] [{label}] prefetch {len(jobs)} symbols in {elapsed:.1f}s "
        f"({stats['cache']} from cache, {stats['fetched']} fetched, "
        f"{stats['failed']} failed)"
    )
    return len(jobs)


def _snap_asset_class(snap: dict) -> str:
    """Return the normalised asset_class for a strategy snapshot dict.
    Mobile wizard saves asset_class as '_asset_class' in config; web portal
    uses 'asset_class'. Both are checked so mobile-built forex/index strategies
    are routed to the correct executor."""
    _obj = snap.get("_obj")
    _cfg = snap.get("config") or {}
    _col_ac = (getattr(_obj, "asset_class", None) or "").strip()
    _cfg_ac = (_cfg.get("asset_class") or _cfg.get("_asset_class") or "").strip()
    # DB column may still be the default 'crypto' before backfill — trust config.
    if _col_ac == "crypto" and _cfg_ac and _cfg_ac != "crypto":
        return _cfg_ac.lower()
    return _col_ac or _cfg_ac or "crypto"


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


_EXECUTOR_HEARTBEATS: Dict[str, float] = {}


def mark_heartbeat(name: str) -> None:
    """Record that an executor loop just started a cycle (monotonic-free wall clock).

    Read by the hourly system health monitor to verify the scan loops are alive.
    """
    try:
        _EXECUTOR_HEARTBEATS[name] = time.time()
    except Exception:
        pass


def get_heartbeats() -> Dict[str, float]:
    """Return a copy of the last-cycle wall-clock timestamps for each loop."""
    return dict(_EXECUTOR_HEARTBEATS)


async def persist_executor_process_heartbeat() -> None:
    """Persist standalone executor liveness to DB (30s cadence from executor_runner)."""
    import os as _os

    mark_heartbeat("executor_process")
    from sqlalchemy import text as _sql_text
    from app.database import BgSessionLocal as SessionLocal
    try:
        from app.executor_lock import get_executor_lock_id
        _lock_id = get_executor_lock_id()
    except Exception:
        _lock_id = None
    db = SessionLocal()
    try:
        db.execute(_sql_text("""
            CREATE TABLE IF NOT EXISTS executor_runtime_heartbeat (
                process_key VARCHAR(32) PRIMARY KEY,
                last_seen_at TIMESTAMP NOT NULL DEFAULT (NOW() AT TIME ZONE 'utc'),
                lock_id INTEGER,
                host_pid INTEGER
            )
        """))
        db.execute(
            _sql_text("""
                INSERT INTO executor_runtime_heartbeat
                    (process_key, last_seen_at, lock_id, host_pid)
                VALUES
                    (:pk, NOW() AT TIME ZONE 'utc', :lid, :pid)
                ON CONFLICT (process_key) DO UPDATE SET
                    last_seen_at = EXCLUDED.last_seen_at,
                    lock_id = EXCLUDED.lock_id,
                    host_pid = EXCLUDED.host_pid
            """),
            {
                "pk": "standalone" if _os.environ.get("EXECUTOR_STANDALONE") else "portal",
                "lid": _lock_id,
                "pid": _os.getpid(),
            },
        )
        db.commit()
    except Exception:
        try:
            db.rollback()
        except Exception:
            pass
    finally:
        db.close()


async def run_strategy_executor():
    """
    Main background loop. Evaluates all active + paper strategies for all users.
  Paper monitor is started from _start_executor_tasks (not here).
    """
    if crypto_executor_disabled():
        logger.info(
            "🤖 Crypto executor SKIPPED (DISABLE_CRYPTO_EXECUTOR=1) — "
            "forex-only or dedicated tradfi replica"
        )
        return

    from app.database import BgSessionLocal as SessionLocal, bg_engine as engine
    from app.models import User
    from app.strategy_models import UserStrategy, init_strategy_tables

    init_strategy_tables(engine)
    logger.info(
        "🤖 Crypto executor started (active + paper modes, "
        f"cycle={CRYPTO_SCAN_INTERVAL_SECONDS}s)"
    )
    if EXECUTOR_CRYPTO_START_DELAY > 0:
        logger.info(
            f"🤖 Crypto executor: waiting {EXECUTOR_CRYPTO_START_DELAY}s before first scan "
            "(forex warm-up — avoids silent log gap on deploy)"
        )
        await asyncio.sleep(EXECUTOR_CRYPTO_START_DELAY)

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

        count = EXECUTOR_SHARD_COUNT
        if count <= 1:
            await _run_crypto_executor_shard(0, 1, http_client)
            return
        logger.info(
            f"🤖 Crypto executor launching {count} parallel scan shards "
            f"(~{MAX_CONCURRENT} concurrent evals per shard)"
        )
        await asyncio.gather(
            *(_run_crypto_executor_shard(i, count, http_client) for i in range(count))
        )


async def _run_crypto_executor_shard(
    shard_index: int,
    shard_count: int,
    http_client: httpx.AsyncClient,
):
    """Crypto scan loop for strategies where ``strategy_id % shard_count == shard_index``."""
    from app.database import BgSessionLocal as SessionLocal
    from app.models import User
    from app.strategy_models import UserStrategy

    _crypto_lbl = _executor_shard_label("Crypto Executor", shard_index, shard_count)
    if shard_index > 0 and EXECUTOR_SHARD_STAGGER_SECONDS > 0:
        await asyncio.sleep(shard_index * EXECUTOR_SHARD_STAGGER_SECONDS)

    sem = asyncio.Semaphore(MAX_CONCURRENT)
    _hb_name = "crypto_executor" if shard_count <= 1 else f"crypto_executor_s{shard_index}"

    while True:
            _cycle_t0 = datetime.utcnow()
            try:
                mark_heartbeat(_hb_name)
                strategy_snapshots = _load_strategy_snapshots_cached(
                    SessionLocal, UserStrategy,
                )

                if not strategy_snapshots:
                    await asyncio.sleep(CRYPTO_SCAN_INTERVAL_SECONDS)
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
                eval_snapshots = [
                    s for s in eval_snapshots
                    if strategy_on_shard(s["id"], shard_index, shard_count)
                ]
                skipped = len(strategy_snapshots) - len(eval_snapshots)
                if skipped:
                    logger.debug(
                        f"[Executor] Skipping {skipped} locked subscriber copies "
                        f"(will be triggered by source propagation)"
                    )
                if eval_snapshots or shard_index == 0:
                    logger.info(
                        f"🤖 {_crypto_lbl}: {len(eval_snapshots)} crypto strateg"
                        f"{'y' if len(eval_snapshots) == 1 else 'ies'} on this shard "
                        f"({active_count} live · {paper_count} paper in pool) [{_ac_str}]"
                    )

                # Pre-fetch tickers ONCE for the entire cycle.
                shared_tickers = await _get_raw_tickers(http_client)

                # Batch-load Pro status ONCE per cycle (was a per-strategy query)
                # so each strategy task only does a single JOINed strategy+user
                # lookup — lets the loop scale to ~100 strategies cheaply.
                _now_cycle = datetime.utcnow()
                _crypto_uids = {s["user_id"] for s in eval_snapshots}
                has_pro_by_user: Dict[int, bool] = {}
                if _crypto_uids:
                    from app.strategy_models import PortalSubscription as _PSub
                    _ref_db = SessionLocal()
                    try:
                        for _ps in _ref_db.query(_PSub).filter(_PSub.user_id.in_(_crypto_uids)).all():
                            has_pro_by_user[_ps.user_id] = bool(
                                _ps.tier == "pro"
                                and _ps.subscription_end
                                and _ps.subscription_end > _now_cycle
                            )
                    except Exception as _ref_err:
                        logger.debug(f"[Executor] ref preload failed: {_ref_err}")
                    finally:
                        _ref_db.close()

                # Per-cycle gate diagnostics — counts which gate blocks each strategy.
                # Logged at end of cycle so we can see WHY strategies aren't firing.
                cycle_gate_stats: Dict[str, int] = {}
                _preloaded_users = _preload_strategy_users(
                    [s["id"] for s in eval_snapshots],
                    SessionLocal,
                    UserStrategy,
                    User,
                )

                async def _run_one(snap, _tickers=shared_tickers):
                    """Each strategy evaluation runs in its own isolated DB session.

                    Retries with a fresh session on transient Neon / pool errors.
                    """
                    from sqlalchemy.exc import OperationalError as _SAOperationalError
                    from sqlalchemy.exc import PendingRollbackError as _SAPendingRollbackError
                    from sqlalchemy.exc import TimeoutError as _SATimeoutError
                    from app.database import bg_db_slot
                    async with sem:
                        async with bg_db_slot():
                            row = _preloaded_users.get(snap["id"])
                            if not row:
                                return
                            _strategy_row, _user_row = row
                            if not _user_row or _user_row.banned:
                                return
                            if not _portal_trade_entitled(_user_row, has_pro_by_user):
                                return
                            for _attempt in (1, 3):
                                db_one = SessionLocal()
                                try:
                                    strategy = db_one.merge(_strategy_row)
                                    user = db_one.merge(_user_row)
                                    await evaluate_and_fire(
                                        strategy, user, db_one, http_client,
                                        raw_tickers=_tickers,
                                        gate_stats=cycle_gate_stats,
                                    )
                                    return
                                except (
                                    _SAOperationalError,
                                    _SAPendingRollbackError,
                                    _SATimeoutError,
                                ) as _db_err:
                                    _err_name = type(_db_err).__name__
                                    if getattr(_db_err, "orig", None) is not None:
                                        _err_name = type(_db_err.orig).__name__
                                    if _attempt < 3:
                                        logger.warning(
                                            f"[Strategy {snap['id']}] Transient DB error "
                                            f"({_err_name}) — retry {_attempt}/3"
                                        )
                                        _db_rollback_safe(db_one)
                                        await asyncio.sleep(0.5 * _attempt)
                                        continue
                                    logger.warning(
                                        f"[Strategy {snap['id']}] Skipping cycle — "
                                        f"DB error persisted ({_err_name})"
                                    )
                                    return
                                except Exception as e:
                                    logger.error(
                                        f"[Strategy {snap['id']}] Error: {e}", exc_info=True
                                    )
                                    _db_rollback_safe(db_one)
                                    return
                                finally:
                                    try:
                                        db_one.close()
                                    except Exception:
                                        pass

                await _prefetch_price_ta_for_cycle(
                    eval_snapshots,
                    http_client,
                    {"crypto"},
                    label=_crypto_lbl,
                )

                await _gather_eval_batches(
                    _crypto_lbl, eval_snapshots, _run_one,
                )

                try:
                    from app.services.ctrader_order_queue import flush_gate_stats_to_db
                    flush_gate_stats_to_db([s["id"] for s in eval_snapshots])
                except Exception:
                    pass

                # Cycle gate diagnostics — shows exactly which gate blocked each strategy.
                # Helps diagnose "why aren't trades firing?" without spelunking logs.
                if cycle_gate_stats:
                    _gate_summary = " ".join(
                        f"{k.replace('blk_', '')}={v}"
                        for k, v in sorted(cycle_gate_stats.items(), key=lambda kv: -kv[1])
                    )
                    logger.info(
                        f"[{_log_ts()}] [{_crypto_lbl}] cycle gates → {_gate_summary}"
                    )
                    try:
                        from app.services.feed_diagnostics import (
                            GATE_BLOCK_LOCATIONS,
                            format_final_report,
                        )
                        _locs = [
                            f"{k}: {GATE_BLOCK_LOCATIONS.get(k, 'strategy_executor.evaluate_and_fire')}"
                            for k in cycle_gate_stats if k.startswith("blk_")
                        ]
                        logger.info(
                            format_final_report(
                                scans_running=1,
                                symbols_scanned=len(eval_snapshots),
                                candles_loaded=0,
                                setups_detected=0,
                                signals_blocked=cycle_gate_stats,
                                block_locations=_locs,
                            )
                        )
                    except Exception:
                        pass

                try:
                    from app.services.feed_diagnostics import log_blocker_rollup
                    log_blocker_rollup(100)
                except Exception:
                    pass

                _cycle_s = (datetime.utcnow() - _cycle_t0).total_seconds()
                if eval_snapshots:
                    logger.info(
                        f"[{_log_ts()}] [{_crypto_lbl}] cycle done in {_cycle_s:.1f}s "
                        f"({len(eval_snapshots)} strategies)"
                    )

            except Exception as e:
                logger.error(f"[{_log_ts()}] {_crypto_lbl} loop error: {e}", exc_info=True)

            await asyncio.sleep(CRYPTO_SCAN_INTERVAL_SECONDS)


_FX_MIN_TRAIL_STEP_FRAC = 0.001  # 0.1% of price — don't hammer the broker every tick
# How often the fast loop reloads open positions from DB (new trades pick up BE/trail).
_FX_WORKLIST_TTL = float(_os_env.environ.get("EXECUTOR_FX_WORKLIST_TTL", "1"))


def _ctrader_position_id_from_execution(ex) -> Optional[int]:
    """Broker position id from column or pos= token in notes."""
    import re as _re
    if ex.ctrader_position_id:
        try:
            return int(ex.ctrader_position_id)
        except Exception:
            pass
    m = _re.search(r"pos=(\d+)", ex.notes or "")
    return int(m.group(1)) if m else None


def _build_forex_worklist() -> list:
    """Build the list of open LIVE forex positions needing breakeven/trailing SL
    management. Pure synchronous DB read so the fast management loop can run it in
    a thread. Returns a list of work dicts (one per managed position).

    On missing trade-mgmt columns, runs schema migration once then retries."""
    try:
        return _build_forex_worklist_impl()
    except Exception as exc:
        from app.trade_mgmt_schema import (
            ensure_trade_mgmt_columns,
            is_trade_mgmt_schema_error,
        )
        if not is_trade_mgmt_schema_error(exc):
            raise
        logger.error(
            "[FX-fast] trade_mgmt columns missing (%s) — running schema migration",
            exc,
        )
        from app.database import bg_engine as _bg_eng
        if ensure_trade_mgmt_columns(_bg_eng, wait_seconds=15.0):
            return _build_forex_worklist_impl()
        logger.error(
            "[FX-fast] trade_mgmt migration failed — worklist empty this cycle"
        )
        return []


def _build_forex_worklist_impl() -> list:
    """Inner worklist query — requires trade_mgmt columns on strategy_executions."""
    import re as _re
    from app.database import BgSessionLocal as SessionLocal
    from app.strategy_models import StrategyExecution, UserStrategy
    from app.models import User

    db = SessionLocal()
    try:
        open_execs = (
            db.query(StrategyExecution)
            .filter(
                StrategyExecution.is_paper == False,          # noqa: E712
                StrategyExecution.outcome == "OPEN",
                StrategyExecution.asset_class.in_(("forex", "index")),
                StrategyExecution.ctrader_order_id.isnot(None),
                StrategyExecution.entry_price.isnot(None),
            )
            .all()
        )
        work = []
        strat_cache: Dict[int, tuple] = {}
        user_cache: Dict[int, object] = {}
        for ex in open_execs:
            notes = ex.notes or ""
            position_id = _ctrader_position_id_from_execution(ex)
            if position_id is None:
                continue

            cached = strat_cache.get(ex.strategy_id)
            if cached is None:
                strat = db.query(UserStrategy).filter(
                    UserStrategy.id == ex.strategy_id
                ).first()
                cfg = (strat.config if strat else {}) or {}
                sname = strat.name if strat else "Strategy"
                strat_cache[ex.strategy_id] = (cfg, sname)
            else:
                cfg, sname = cached
            ex_cfg = cfg.get("exit", {}) or {}

            # ── Partial-runner (TP1 partial profit) state, parsed from notes ──
            _pc_m = _re.search(r"partial_close_pct=(\d+(?:\.\d+)?)", notes)
            partial_pct  = float(_pc_m.group(1)) if _pc_m else 0.0
            partial_done = ("partial_close_done" in notes) or ("partial_skip" in notes)
            _vol_m   = _re.search(r"vol=(\d+)", notes)
            pos_vol  = int(_vol_m.group(1)) if _vol_m else 0
            tp1_price = float(ex.tp_price)  if ex.tp_price  is not None else None
            tp2_price = float(ex.tp2_price) if ex.tp2_price is not None else None
            # Active while a partial close is still pending (have a 2nd target, a
            # captured volume to split, and haven't acted yet).
            partial_active = bool(
                partial_pct > 0 and tp2_price and pos_vol > 0 and not partial_done
            )
            # The broker holds TP at TP2 for any partial-runner position (set at
            # placement); re-sending TP1 on an SL amend would clobber it.
            broker_tp = tp2_price if (partial_pct > 0 and tp2_price) else tp1_price

            # Forex is 1x leverage → the leveraged-ROI breakeven trigger is
            # unreachable. Use a pip/distance-based absolute price level instead.
            be_trigger = _compute_be_trigger_price(
                ex.symbol, float(ex.entry_price), ex.direction, ex.tp_price, ex_cfg,
            )
            trail_enabled = bool(ex_cfg.get("trailing_stop"))
            trail_pct = ex_cfg.get("trailing_stop_pct")
            if trail_enabled and (not trail_pct or float(trail_pct) <= 0):
                _slp = ex_cfg.get("stop_loss_pct")
                trail_pct = (float(_slp) / 2.0) if _slp else None
            from app.services.trade_management import effective_trade_mgmt_cfg
            _tm = effective_trade_mgmt_cfg(cfg)
            if (
                not any((_tm["breakeven_enabled"], _tm["trailing_enabled"], _tm["partial_tp_enabled"]))
                and be_trigger is None
                and not (trail_enabled and trail_pct)
                and not partial_active
            ):
                continue

            user = user_cache.get(ex.user_id)
            if user is None:
                user = db.query(User).filter(User.id == ex.user_id).first()
                user_cache[ex.user_id] = user
            if not user:
                continue

            work.append({
                "exec_id":       ex.id,
                "position_id":   position_id,
                "symbol":        ex.symbol,
                "direction":     ex.direction,
                "entry_price":   float(ex.entry_price),
                "sl_price":      float(ex.sl_price) if ex.sl_price is not None else None,
                "tp_price":      float(ex.tp_price) if ex.tp_price is not None else None,
                "leverage":      float(ex.leverage or 1) or 1.0,
                "be_trigger":    be_trigger,
                "trail_enabled": bool(trail_enabled and trail_pct),
                "trail_pct":     float(trail_pct) if trail_pct else None,
                "be_moved":      ("be_moved" in notes),
                "strategy_id":   ex.strategy_id,
                "strategy_name": sname,
                "user":          user,
                # Partial-runner state (TP1 partial profit → BE → run to TP2).
                "partial_pct":    partial_pct,
                "partial_active": partial_active,
                "partial_done":   partial_done,
                "pos_vol":        pos_vol,
                "tp1_price":      tp1_price,
                "tp2_price":      tp2_price,
                "broker_tp":      broker_tp,
            })
    finally:
        db.close()
    return work


async def _do_forex_partial_close(w: dict, price: float) -> None:
    """TP1 reached on a partial-runner position: close ``partial_pct``% via the
    broker, move the stop to breakeven (keeping the broker TP parked at TP2), mark
    the execution, and notify. Falls back to a stop-to-breakeven-only move when the
    position is too small to split on the broker volume grid (the full position then
    rides to TP2 — and we DON'T mark partial_close_done, so the close-out P&L is the
    full position, not a blend)."""
    from app.services.ctrader_client import (
        close_partial_position_for_user, modify_position_sltp_for_user,
    )
    from app.database import BgSessionLocal as SessionLocal
    from app.strategy_models import StrategyExecution

    entry     = w["entry_price"]
    direction = w["direction"]
    tp2       = w.get("tp2_price")
    frac      = float(w.get("partial_pct") or 0) / 100.0

    # close_partial returns: >0 closed units (success), -1 confirmed un-splittable,
    # 0 transient failure (retry next cycle — DON'T touch SL/notes/state).
    try:
        closed = await close_partial_position_for_user(
            w["user"], w["symbol"], w["position_id"], int(w.get("pos_vol") or 0), frac,
        )
    except Exception as e:
        logger.warning(f"[FX-partial] close failed exec#{w['exec_id']}: {e}")
        closed = 0  # transient

    if closed == 0:
        # Transient: leave the position eligible so the next cycle retries. Do not
        # amend the stop, mark notes, or claim a breakeven move that never happened.
        logger.info(
            f"[FX-partial] exec#{w['exec_id']} {w['symbol']} {direction} TP1 partial "
            f"transient failure — will retry next cycle"
        )
        return

    closed_units = closed if closed > 0 else 0  # closed == -1 → confirmed un-splittable

    # Move the (remaining, or full-if-unsplittable) position's stop to breakeven.
    # Keep the broker TP at TP2 — an SL-only amend would clear it
    # (ProtoOAAmendPositionSLTPReq replaces both legs).
    amend_ok = await modify_position_sltp_for_user(
        w["user"], w["position_id"], stop_loss_price=round(entry, 6),
        take_profit_price=tp2,
    )

    # partial_close_done (success → blend P&L) vs partial_skip (confirmed
    # un-splittable → run the FULL position to TP2, no blend). Either way the
    # partial is now resolved and must not re-fire.
    tok = "partial_close_done" if closed_units > 0 else "partial_skip"
    _db = SessionLocal()
    try:
        ex = _db.query(StrategyExecution).filter(
            StrategyExecution.id == w["exec_id"]
        ).first()
        if ex and ex.outcome == "OPEN":
            n = ex.notes or ""
            if tok not in n:
                n = (n + f" | {tok}").strip(" |")
            # Only record the breakeven move (notes flag + persisted SL) when the
            # broker actually accepted the amend — otherwise we'd claim a BE that
            # never reached the broker.
            if amend_ok:
                if "be_moved" not in n:
                    n = (n + " | be_moved").strip(" |")
                ex.sl_price = round(entry, 6)
            ex.notes = n
            _db.commit()
    finally:
        _db.close()

    # Update the cached work item so the fast loop won't re-fire the partial.
    w["partial_active"] = False
    w["partial_done"]   = True
    if amend_ok:
        w["sl_price"] = round(entry, 6)
        w["be_moved"] = True

    logger.info(
        f"[FX-partial] exec#{w['exec_id']} {w['symbol']} {direction} TP1 {w.get('tp1_price')} → "
        f"{'closed ' + str(closed_units) + 'u' if closed_units > 0 else 'too-small-to-split (full runner)'}, "
        f"SL→entry {entry:.6g} ({'amended' if amend_ok else 'AMEND FAILED'}), runner targets TP2 {tp2}"
    )

    # Breakeven / partial alert (best-effort, once per execution).
    try:
        _dbn = SessionLocal()
        try:
            from app.strategy_models import UserStrategy as _US_partial
            _u = w.get("user")
            ex_p = _dbn.query(StrategyExecution).filter(
                StrategyExecution.id == w["exec_id"]
            ).first()
            strat_p = _dbn.query(_US_partial).filter(
                _US_partial.id == w.get("strategy_id")
            ).first()
            if ex_p and _u:
                _ensure_open_notify_for_execution(
                    _dbn, ex_p, user=_u, strategy=strat_p,
                )
            if _claim_tg_be_notify(_dbn, w["exec_id"]):
                _notify_breakeven_alert(
                    user_id=getattr(_u, "id", 0) or 0,
                    telegram_id=getattr(_u, "telegram_id", None),
                    strategy_name=w.get("strategy_name", "Strategy"),
                    symbol=w["symbol"], direction=direction,
                    leverage=int(w.get("leverage") or 1), move_pct=0.0,
                    strategy_id=w.get("strategy_id", 0), execution_id=w["exec_id"],
                    kind="live",
                )
        finally:
            _dbn.close()
    except Exception:
        pass


async def _amend_forex_position(w: dict) -> None:
    """Live forex position management — delegates to trade_management module."""
    await _amend_forex_position_tick(w)


async def _amend_forex_position_tick(
    w: dict,
    *,
    tick_mono: Optional[float] = None,
    mid: Optional[float] = None,
) -> None:
    """Tick-driven or poll-driven live forex BE/trail management."""
    import time as _time

    from app.services.tradfi_prices import get_price as _tradfi_get_price
    from app.database import BgSessionLocal as SessionLocal
    from app.strategy_models import StrategyExecution, UserStrategy
    from app.services.trade_management import manage_open_position

    t0 = tick_mono or _time.monotonic()
    price = mid
    if not price or price <= 0:
        try:
            from app.services import ctrader_price_feed as _ctf
            price = _ctf.get_price(w["symbol"])
        except Exception:
            price = None
    if not price or price <= 0:
        price = await _tradfi_get_price(w["symbol"], "forex")
    if not price or price <= 0:
        return

    _db2 = SessionLocal()
    try:
        ex2 = _db2.query(StrategyExecution).filter(
            StrategyExecution.id == w["exec_id"]
        ).first()
        strat2 = _db2.query(UserStrategy).filter(
            UserStrategy.id == w.get("strategy_id")
        ).first()
        if ex2 and ex2.outcome == "OPEN" and strat2:
            await manage_open_position(
                ex2, strat2.config or {}, float(price), _db2,
            )
            if ex2.sl_price is not None:
                w["sl_price"] = float(ex2.sl_price)
            if getattr(ex2, "breakeven_applied", False):
                w["be_moved"] = True
            dispatch_ms = (_time.monotonic() - t0) * 1000.0
            if tick_mono is not None and dispatch_ms < 2000:
                logger.info(
                    "[trade-mgmt] breakeven via tick (+%.0fms from threshold cross) "
                    "exec=%s %s",
                    dispatch_ms,
                    w.get("exec_id"),
                    w.get("symbol"),
                )
    finally:
        _db2.close()


_FX_RECONCILE_MISSING: Dict[int, int] = {}   # exec_id → consecutive missing-sweep count
# Broker open-position poll — detects SL/TP fills and fires close alerts.
_FX_RECONCILE_INTERVAL = float(
    _os_env.environ.get("EXECUTOR_FX_RECONCILE_INTERVAL", "5")
)


async def _close_live_forex_execution_with_db_retry(
    ex_id: int,
    outcome: str,
    exit_price: float,
    source: str = "ctrader-reconcile",
) -> bool:
    """Close+notify with fresh-session retry on Neon SSL blips."""
    from app.db_resilience import is_transient_db_error

    for attempt in range(3):
        try:
            return await _close_live_forex_execution_and_notify(
                ex_id, outcome, exit_price, source=source,
            )
        except Exception as exc:
            if is_transient_db_error(exc) and attempt < 2:
                logger.warning(
                    "[FX-reconcile] close transient DB error exec#%s (retry %s/3): %s",
                    ex_id,
                    attempt + 1,
                    exc,
                )
                await asyncio.sleep(1.0 * (attempt + 1))
                continue
            logger.warning(f"[FX-reconcile] close error exec#{ex_id}: {exc}")
            return False
    return False


async def _close_live_forex_execution_and_notify(
    ex_id: int, outcome: str, exit_price: float, source: str = "ctrader-reconcile"
) -> bool:
    """Atomically close a LIVE forex execution whose broker position was detected
    closed by reconciliation, then fire the Telegram DM + mobile push.

    Mirrors the crypto `_close_live_execution_and_notify` closure but is module-
    level so the forex loops can call it, and adds pips_pnl (forex perf tracks it).
    The atomic UPDATE ... WHERE outcome='OPEN' guarantees exactly one notification
    even if loops/workers race to close the same execution.
    """
    from app.database import BgSessionLocal as SessionLocal
    from app.strategy_models import StrategyExecution, UserStrategy
    from app.models import User
    from sqlalchemy import text as _text
    from sqlalchemy.exc import OperationalError as _SAOperationalError
    from app.db_resilience import is_transient_db_error

    db = None
    ex = None
    for _db_attempt in range(3):
        db = SessionLocal()
        try:
            ex = db.query(StrategyExecution).filter(StrategyExecution.id == ex_id).first()
            break
        except _SAOperationalError as _dbe:
            try:
                db.close()
            except Exception:
                pass
            db = None
            if is_transient_db_error(_dbe) and _db_attempt < 2:
                logger.warning(
                    "[FX-reconcile] close notify DB session retry %s/3 exec#%s: %s",
                    _db_attempt + 1,
                    ex_id,
                    _dbe,
                )
                await asyncio.sleep(1.0 * (_db_attempt + 1))
                continue
            raise
    if db is None:
        return False
    try:
        if not ex or ex.outcome != "OPEN":
            return False

        leverage = ex.leverage or 1
        entry    = float(ex.entry_price or 0)
        if entry <= 0:
            return False

        _pnl_exit = _tp1_blend_exit(ex, exit_price)
        _blend_line = _tp1_blend_pips_line(ex, exit_price)

        if ex.direction == "LONG":
            raw_pnl = (_pnl_exit - entry) / entry * 100
        else:
            raw_pnl = (entry - _pnl_exit) / entry * 100
        pnl_pct   = round(raw_pnl * leverage, 2)
        closed_at = datetime.utcnow()

        # Pips P&L (forex/metals) — broker fill, no synthetic spread deduction.
        pips_pnl: Optional[float] = None
        try:
            from app.services.forex_engine import pip_size as _pip_sz
            _ps = _pip_sz(ex.symbol or "")
            if _ps and _ps > 0:
                if ex.direction == "LONG":
                    pips_pnl = round((_pnl_exit - entry) / _ps, 1)
                else:
                    pips_pnl = round((entry - _pnl_exit) / _ps, 1)
        except Exception:
            pips_pnl = None

        result = db.execute(
            _text(
                "UPDATE strategy_executions "
                "SET outcome=:o, exit_price=:xp, pnl_pct=:p, closed_at=:ca, pips_pnl=:pp "
                "WHERE id=:id AND outcome='OPEN'"
            ),
            {"o": outcome, "xp": exit_price, "p": pnl_pct, "ca": closed_at,
             "pp": pips_pnl, "id": ex.id},
        )
        db.commit()
        if result.rowcount == 0:
            return True  # another worker/loop already closed it — no double notify

        ex.outcome = outcome; ex.exit_price = exit_price
        ex.pnl_pct = pnl_pct; ex.closed_at  = closed_at

        from app.services.close_notify_dedupe import claim_close_notification
        if not claim_close_notification(db, ex.id):
            return True

        from app.services.trade_management import close_note_label
        from app.services.forex_tick_manager import unregister_position

        # Append an auditable close note (preserve prior open-time context).
        pnl_sign = "+" if pnl_pct >= 0 else ""
        label = close_note_label(ex, outcome, float(exit_price))
        cn = f"{label} · {pnl_sign}{pnl_pct}% · exit {exit_price:.6g}"
        unregister_position(ex.id, ex.symbol or "")
        existing = (ex.notes or "").strip()
        if existing and not existing.startswith("open ·"):
            cn = f"{existing} | {cn}"
        try:
            db.execute(
                _text("UPDATE strategy_executions SET notes=:n WHERE id=:id"),
                {"n": cn, "id": ex.id},
            )
            db.commit()
            ex.notes = cn
        except Exception:
            try: db.rollback()
            except Exception: pass

        _update_performance(ex.strategy_id, db)

        logger.info(
            f"[FX-reconcile] {('TP' if outcome=='WIN' else 'BE' if outcome=='BREAKEVEN' else 'SL')} "
            f"CLOSE ({source}): {ex.symbol} {ex.direction} entry={entry} "
            f"exit={exit_price} pnl={pnl_pct:+.1f}%"
        )

        try:
            user  = db.query(User).filter(User.id == ex.user_id).first()
            strat = db.query(UserStrategy).filter(UserStrategy.id == ex.strategy_id).first()
            strat_name = (strat.name if strat else None) or "Unknown Strategy"
            if user and user.telegram_id:
                tg_id = _telegram_int_id(user)
                if tg_id:
                    await _deliver_close_telegram(
                        tg_id,
                        execution_id=ex.id,
                        symbol=ex.symbol,
                        outcome=outcome,
                        asset_class=getattr(ex, "asset_class", "forex") or "forex",
                        strategy_name=strat_name,
                        direction=ex.direction,
                        entry=entry,
                        exit_price=exit_price,
                        pnl_pct=pnl_pct,
                        leverage=leverage,
                        fired_at=ex.fired_at,
                        closed_at=closed_at,
                        conditions=ex.conditions_met,
                        is_paper=False,
                        tp1_blend_line=_blend_line,
                        mfe_pips=getattr(ex, "mfe_pips", None),
                        mae_pips=getattr(ex, "mae_pips", None),
                    )
            from app.services.expo_push import notify_trade_close_bg
            dur_mins = int((closed_at - ex.fired_at).total_seconds() / 60) if ex.fired_at else 0
            notify_trade_close_bg(
                user_id=ex.user_id, strategy_name=strat_name, symbol=ex.symbol,
                direction=ex.direction, outcome=outcome, pnl_pct=pnl_pct,
                leverage=leverage, entry_price=entry, exit_price=exit_price,
                strategy_id=ex.strategy_id, execution_id=ex.id, is_paper=False,
                duration_mins=dur_mins, kind="live",
                position_usd=float(ex.position_size) if ex.position_size else None,
            )
        except Exception as ne:
            logger.warning(f"[FX-reconcile] notify error exec#{ex.id}: {ne}")
        return True
    except Exception as e:
        logger.error(f"[FX-reconcile] close error exec#{ex_id}: {e}")
        try: db.rollback()
        except Exception: pass
        return False
    finally:
        db.close()


def _build_forex_reconcile_worklist() -> list:
    """Pure-sync DB read of open LIVE forex execs for broker close-reconciliation.
    Returns one dict per tracked position (carrying tp/sl/entry/be flag and a
    detached User for the per-user broker reconcile fetch)."""
    import re as _re
    from app.database import BgSessionLocal as SessionLocal
    from app.strategy_models import StrategyExecution
    from app.models import User, UserPreference

    db = SessionLocal()
    try:
        open_execs = (
            db.query(StrategyExecution)
            .filter(
                StrategyExecution.is_paper == False,          # noqa: E712
                StrategyExecution.outcome == "OPEN",
                StrategyExecution.asset_class.in_(("forex", "index")),
                StrategyExecution.ctrader_order_id.isnot(None),
                StrategyExecution.entry_price.isnot(None),
            )
            .all()
        )
        user_cache: Dict[int, object] = {}
        acct_cache: Dict[int, Optional[int]] = {}   # user_id → current cTrader account id
        work = []
        for ex in open_execs:
            notes = ex.notes or ""
            position_id = _ctrader_position_id_from_execution(ex)
            if position_id is None:
                continue  # no broker positionId captured → can't reconcile
            uid = ex.user_id
            user = user_cache.get(uid)
            if user is None:
                user = db.query(User).filter(User.id == uid).first()
                if user is not None:
                    db.expunge(user)
                user_cache[uid] = user
            if user is None:
                continue

            # Account binding: if this execution recorded the broker account it was
            # opened on (acct=<ctid>), only reconcile when that matches the user's
            # CURRENT cTrader account. A relink to a different account must NOT
            # false-close a position still open on the original account. Legacy
            # executions without an acct token fall back to current-account
            # reconciliation (best-effort, the prior behaviour).
            am = _re.search(r"acct=(\d+)", notes)
            if am:
                if uid not in acct_cache:
                    _pf = db.query(UserPreference).filter(
                        UserPreference.user_id == uid
                    ).first()
                    acct_cache[uid] = (
                        int(_pf.ctrader_account_id)
                        if _pf and _pf.ctrader_account_id else None
                    )
                cur_acct = acct_cache[uid]
                if cur_acct is None or int(am.group(1)) != cur_acct:
                    continue  # different/unknown account → skip (never false-close)

            # Partial-runner positions park the broker TP at TP2 (the remainder
            # exits there, not at TP1), so reconcile must classify the close against
            # TP2 — otherwise a TP2 fill is mis-measured against TP1 and the
            # close-out P&L blend (which reads ex.tp_price=TP1) underreports.
            _rec_tp = float(ex.tp_price) if ex.tp_price else None
            if ("partial_close_pct=" in notes) and ex.tp2_price:
                _rec_tp = float(ex.tp2_price)

            work.append({
                "exec_id":     ex.id,
                "user_id":     uid,
                "user":        user,
                "position_id": position_id,
                "symbol":      ex.symbol,
                "direction":   ex.direction,
                "entry":       float(ex.entry_price),
                "tp_price":    _rec_tp,
                "sl_price":    float(ex.sl_price) if ex.sl_price else None,
                "be_moved":    ("be_moved" in notes),
            })
    finally:
        db.close()
    return work


async def _reconcile_forex_closes() -> None:
    """Detect live forex positions closed broker-side (SL/TP fill) and close+notify.

    The forex live loop only AMENDS broker SL/TP; nothing else notices when the
    broker actually fills an SL/TP and removes the position. Without this, a live
    forex SL hit produced NO push/Telegram alert and the execution sat OPEN until
    the 48h stale-expiry swept it (silently, pnl=0). This polls cTrader open
    positions per user; when a tracked positionId is gone for 2 consecutive
    sweeps it classifies WIN/LOSS/BREAKEVEN via price-vs-TP/SL distance and fires
    `_close_live_forex_execution_and_notify`.
    """
    from app.db_resilience import is_transient_db_error, run_with_db_retry
    from app.services.ctrader_client import (
        get_open_position_ids_for_user,
        get_position_close_detail_for_user,
    )

    work = None
    for attempt in range(3):
        try:
            work = await asyncio.to_thread(
                lambda: run_with_db_retry(
                    _build_forex_reconcile_worklist,
                    label="FX-reconcile-worklist",
                )
            )
            break
        except Exception as e:
            if is_transient_db_error(e) and attempt < 2:
                logger.warning(
                    "[FX-reconcile] worklist transient DB error (retry %s/3): %s",
                    attempt + 1,
                    e,
                )
                await asyncio.sleep(1.0 * (attempt + 1))
                continue
            logger.warning(f"[FX-reconcile] worklist build failed: {e}")
            return
    if work is None:
        return
    if not work:
        return

    # Group by user so we hit the broker once per account.
    by_user: Dict[int, list] = {}
    user_obj: Dict[int, object] = {}
    for w in work:
        by_user.setdefault(w["user_id"], []).append(w)
        user_obj[w["user_id"]] = w["user"]

    for uid, items in by_user.items():
        try:
            open_ids = await get_open_position_ids_for_user(user_obj[uid])
        except Exception as e:
            logger.warning(f"[FX-reconcile] open-positions fetch failed user {uid}: {e}")
            continue
        if open_ids is None:
            # Broker unreachable / no creds — never false-close.
            continue

        for w in items:
            ex_id = w["exec_id"]
            if w["position_id"] in open_ids:
                _FX_RECONCILE_MISSING.pop(ex_id, None)
                continue

            # Broker truth: deal history gives actual close price (catches downtime misses).
            broker_close = await get_position_close_detail_for_user(
                user_obj[uid],
                int(w["position_id"]),
                entry_hint=w["entry"],
                direction=w["direction"],
            )
            if broker_close and broker_close.get("exit_price"):
                _FX_RECONCILE_MISSING.pop(ex_id, None)
                outcome = broker_close.get("outcome") or "LOSS"
                exit_price = float(broker_close["exit_price"])
                logger.info(
                    f"[FX-reconcile] {w['symbol']} exec#{ex_id} broker closed "
                    f"pos={w['position_id']} @ {exit_price} → {outcome}"
                )
                await _close_live_forex_execution_with_db_retry(
                    ex_id, outcome, exit_price, source="ctrader-reconcile-broker"
                )
                continue

            miss = _FX_RECONCILE_MISSING.get(ex_id, 0) + 1
            _FX_RECONCILE_MISSING[ex_id] = miss
            if miss < 2:
                logger.info(
                    f"[FX-reconcile] {w['symbol']} exec#{ex_id} pos={w['position_id']} "
                    f"absent from broker (miss {miss}/2) — awaiting broker deal"
                )
                continue

            # Never fabricate exit from our SL/TP or spot — mark pending reconcile.
            from app.database import BgSessionLocal as SessionLocal
            from app.strategy_models import StrategyExecution
            from app.services.trade_management import mark_pending_reconcile

            _pdb = SessionLocal()
            try:
                _pex = _pdb.query(StrategyExecution).filter(
                    StrategyExecution.id == ex_id,
                ).first()
                if _pex and _pex.outcome == "OPEN":
                    mark_pending_reconcile(
                        _pex,
                        _pdb,
                        reason="broker deal unavailable",
                    )
                    logger.warning(
                        "[FX-reconcile] exec#%s pos=%s pending_reconcile — "
                        "no broker deal (will not use synthetic SL/TP)",
                        ex_id,
                        w["position_id"],
                    )
            finally:
                _pdb.close()
            _FX_RECONCILE_MISSING.pop(ex_id, None)

    # Bound the missing-counter dict.
    if len(_FX_RECONCILE_MISSING) > 500:
        _FX_RECONCILE_MISSING.clear()

    # Audit recent closes vs broker deal P/L (corrects mis-recorded exits like #45288097).
    try:
        from app.services.trade_management import reconcile_broker_pnl_for_recent_closes

        audit = await reconcile_broker_pnl_for_recent_closes(hours=48)
        if audit.get("corrected") or audit.get("phantom_paper"):
            logger.info("[FX-reconcile] 48h close audit: %s", audit)
    except Exception as _aud:
        logger.debug("[FX-reconcile] broker P/L audit failed: %s", _aud)

    # Untracked OPEN rows (no pos= in notes) cannot enter the reconcile worklist
    # but still trip max_open_positions — expire when broker is flat.
    try:
        from app.services.strategy_heal import expire_untracked_forex_opens_when_broker_empty
        await expire_untracked_forex_opens_when_broker_empty(min_age_minutes=30)
    except Exception as _orph:
        logger.debug(f"[FX-reconcile] orphan untracked OPEN sweep failed: {_orph}")


async def _manage_live_forex_positions():
    """One-shot pass over all live forex positions (back-compat / manual use).
    Normal operation runs through run_forex_live_manager_fast instead."""
    work = await asyncio.to_thread(_build_forex_worklist)
    for w in work:
        try:
            await _amend_forex_position(w)
        except Exception as e:
            logger.warning(f"[FX-manage] error exec#{w['exec_id']}: {e}")


async def run_forex_live_manager_fast():
    """Fast (~1s) loop that amends live forex SL (breakeven + trailing) using the
    cTrader real-time spot feed, so fast instruments like gold move to breakeven
    in well under a second instead of waiting for the 5s strategy-scan cycle.

    DB-light: the open-position worklist is rebuilt only every _FX_WORKLIST_TTL;
    each tick reads prices from the in-memory spot cache and only touches the DB
    when a stop is actually amended. Shares the executor advisory lock (started in
    the same worker as run_forex_executor) so amendments never double-fire."""
    from app.services.asset_classes import is_market_open as _is_mkt_open

    logger.info(
        f"⚡ Forex live-manager started (cycle={FOREX_MANAGE_INTERVAL_SECONDS}s)"
    )
    work: list = []
    last_build = 0.0
    last_reconcile = 0.0
    while True:
        try:
            mark_heartbeat("forex_live_manager")
            if not _is_mkt_open("forex", datetime.utcnow()):
                work = []
                await asyncio.sleep(5)
                continue
            now_m = time.monotonic()
            if not work or (now_m - last_build) >= _FX_WORKLIST_TTL:
                try:
                    from app.db_resilience import run_with_db_retry
                    work = await asyncio.to_thread(
                        lambda: run_with_db_retry(
                            _build_forex_worklist,
                            label="FX-fast-worklist",
                        )
                    )
                except Exception as _be:
                    from app.trade_mgmt_schema import is_trade_mgmt_schema_error
                    if is_trade_mgmt_schema_error(_be):
                        logger.error(
                            "[FX-fast] worklist build failed (schema): %s", _be,
                        )
                    else:
                        logger.warning(f"[FX-fast] worklist build failed: {_be}")
                    work = []
                last_build = now_m
            for w in work:
                try:
                    await _amend_forex_position(w)
                except Exception as _ae:
                    logger.warning(
                        f"[FX-fast] amend error exec#{w.get('exec_id')}: {_ae}"
                    )
                try:
                    from app.services.economic_calendar import maybe_warn_open_position
                    await maybe_warn_open_position(
                        w.get("exec_id", 0),
                        w.get("symbol", ""),
                        w.get("user_id", 0) or getattr(w.get("user"), "id", 0),
                        asset_class="forex",
                    )
                except Exception as _nwe:
                    logger.debug("[FX-fast] news warn error: %s", _nwe)
            # Periodically reconcile broker-side closes (SL/TP fills) so a live
            # forex stop-out actually notifies the user instead of sitting OPEN.
            if (now_m - last_reconcile) >= _FX_RECONCILE_INTERVAL:
                last_reconcile = now_m
                try:
                    await _reconcile_forex_closes()
                except Exception as _re:
                    logger.warning(f"[FX-fast] reconcile error: {_re}")
        except Exception as _ce:
            logger.warning(f"[FX-fast] cycle error: {_ce}")
        await asyncio.sleep(FOREX_MANAGE_INTERVAL_SECONDS)


# ─── Dedicated forex / index / stock executor loop ───────────────────────────

async def run_forex_executor():
    """Launch one or more parallel forex/index scan shards."""
    if forex_executor_disabled():
        logger.info(
            "📈 Forex executor SKIPPED (DISABLE_FOREX_EXECUTOR=1) — "
            "crypto-only portal replica"
        )
        return

    count = EXECUTOR_SHARD_COUNT
    if count <= 1:
        await _run_forex_executor_shard(0, 1)
        return
    logger.info(
        f"📈 Forex executor launching {count} parallel scan shards "
        f"(~{FOREX_MAX_CONCURRENT} concurrent evals per shard)"
    )
    await asyncio.gather(
        *(_run_forex_executor_shard(i, count) for i in range(count))
    )


async def _run_forex_executor_shard(shard_index: int, shard_count: int):
    """
    Dedicated scan loop for forex, index, and stock strategies (one shard).

    Strategies are partitioned by ``strategy_id % shard_count`` so multiple
    shards evaluate disjoint subsets in parallel without double-firing.
    """
    from app.database import BgSessionLocal as SessionLocal, bg_engine as engine
    from app.models import User
    from app.strategy_models import UserStrategy, init_strategy_tables

    init_strategy_tables(engine)
    _fx_lbl = _executor_shard_label("FX Executor", shard_index, shard_count)
    logger.info(
        f"📈 {_fx_lbl} started (cycle={FOREX_SCAN_INTERVAL_SECONDS}s)"
    )
    if shard_index > 0 and EXECUTOR_SHARD_STAGGER_SECONDS > 0:
        await asyncio.sleep(shard_index * EXECUTOR_SHARD_STAGGER_SECONDS)

    _TRADFI_CLASSES = {"forex", "index", "stock"}

    sem = asyncio.Semaphore(FOREX_MAX_CONCURRENT)
    _fx_hb = "forex_executor" if shard_count <= 1 else f"forex_executor_s{shard_index}"

    # Lighter pool — yfinance calls are outbound Python HTTP, not MEXC REST;
    # a handful of concurrent connections is plenty.
    _timeout = httpx.Timeout(20.0, connect=5.0, pool=8.0)
    _limits  = httpx.Limits(
        max_connections=50,
        max_keepalive_connections=10,
        keepalive_expiry=30.0,
    )

    async with httpx.AsyncClient(timeout=_timeout, limits=_limits) as http_client:
        _fx_empty_cycles = 0
        _fx_scan_cycle = 0
        while True:
            _cycle_db_skipped = []  # initialised before try so the adaptive
                                    # backoff below is safe even if the cycle
                                    # throws before its own assignment.
            _cycle_t0 = datetime.utcnow()
            try:
                from app.services.feed_diagnostics import begin_scan_metric_batch
                begin_scan_metric_batch()
                mark_heartbeat(_fx_hb)
                _all_snaps = _load_strategy_snapshots_cached(
                    SessionLocal, UserStrategy,
                )
                strategy_snapshots = [
                    s for s in _all_snaps
                    if _snap_asset_class(s) in _TRADFI_CLASSES
                ]
                strategies = _all_snaps

                if not strategy_snapshots:
                    _fx_empty_cycles += 1
                    if _fx_empty_cycles == 1 or _fx_empty_cycles % 12 == 0:
                        logger.info(
                            f"📈 Forex executor: 0 tradfi strategies in scan pool "
                            f"({len(strategies)} paper/active rows total — "
                            "check asset_class is forex/index/stock, not crypto)"
                        )
                    await asyncio.sleep(FOREX_SCAN_INTERVAL_SECONDS)
                    continue
                _fx_empty_cycles = 0

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

                # ── Per-cycle market-open gate ──────────────────────────────
                # Forex/index/stock strategies early-return inside
                # evaluate_and_fire when their market is closed. Doing that gate
                # HERE — once per asset class instead of after several
                # per-strategy DB round-trips — makes closed-market cycles (e.g.
                # weekends) cost almost nothing instead of ~30s of wasted churn.
                from app.services.asset_classes import is_market_open as _is_mkt_open
                _now_cycle = datetime.utcnow()
                _mkt_open_cache: Dict[str, bool] = {}

                def _ac_open(_ac: str) -> bool:
                    v = _mkt_open_cache.get(_ac)
                    if v is None:
                        v = _is_mkt_open(_ac, _now_cycle)
                        _mkt_open_cache[_ac] = v
                    return v

                open_snaps = []
                _closed_by_class: Dict[str, int] = {}
                for s in eval_snapshots:
                    _ac = _snap_asset_class(s)
                    if _ac_open(_ac):
                        open_snaps.append(s)
                    else:
                        _closed_by_class[_ac] = _closed_by_class.get(_ac, 0) + 1

                open_snaps = [
                    s for s in open_snaps
                    if strategy_on_shard(s["id"], shard_index, shard_count)
                ]

                _forex_open = _ac_open("forex")

                if _forex_open:
                    try:
                        from app.services.ctrader_price_feed import sweep_stale_klines
                        from app.services.tradfi_prices import sweep_stale_metal_klines
                        _stale_syms = {
                            (s.get("symbol") or "").upper()
                            for s in open_snaps
                            if (s.get("symbol") or "").upper()
                        }
                        _sym_list = list(_stale_syms) if _stale_syms else None
                        await sweep_stale_klines(symbols=_sym_list)
                        await sweep_stale_metal_klines(symbols=_sym_list)
                    except Exception as _ks_err:
                        logger.debug("[FX Executor] kline staleness sweep: %s", _ks_err)

                if _closed_by_class and shard_index == 0:
                    _closed_summary = " ".join(
                        f"{k}={v}" for k, v in sorted(_closed_by_class.items())
                    )
                    logger.info(
                        f"📈 Forex executor: shard {shard_index}/{shard_count} — "
                        f"{len(open_snaps)} strateg"
                        f"{'y' if len(open_snaps) == 1 else 'ies'} on this shard "
                        f"(market-closed skipped: {_closed_summary})"
                    )
                elif open_snaps or shard_index == 0:
                    logger.info(
                        f"📈 Forex executor: shard {shard_index}/{shard_count} — "
                        f"{len(open_snaps)} tradfi strateg"
                        f"{'y' if len(open_snaps) == 1 else 'ies'} to scan"
                    )

                _fx_scan_cycle += 1
                if open_snaps and _fx_scan_cycle % 12 == 1:
                    _sample = ", ".join(
                        f"#{s['id']} {((s.get('name') or '')[:24]).strip()}"
                        for s in open_snaps[:4]
                    )
                    _more = max(0, len(open_snaps) - 4)
                    logger.info(
                        f"📈 {_fx_lbl} sample: {_sample}"
                        + (f" (+{_more} more)" if _more else "")
                    )

                if not open_snaps:
                    # No tradfi strategies to evaluate this cycle. Live SL
                    # amendments (breakeven/trailing) run on the dedicated
                    # run_forex_live_manager_fast loop, so nothing to do here.
                    await asyncio.sleep(FOREX_SCAN_INTERVAL_SECONDS)
                    continue

                # ── Batch-load reference data ONCE per cycle ────────────────
                # Collapses the former per-strategy PortalSubscription /
                # UserPreference round-trips into 2 set-based queries, so each
                # strategy task only does a single JOINed strategy+user lookup.
                from app.strategy_models import PortalSubscription as _PSub
                from app.models import UserPreference as _UPref
                _uids = {s["user_id"] for s in open_snaps}
                has_pro_by_user: Dict[int, bool] = {}
                ctrader_ok_by_user: Dict[int, bool] = {}
                if _uids:
                    _ref_db = SessionLocal()
                    try:
                        for _ps in _ref_db.query(_PSub).filter(_PSub.user_id.in_(_uids)).all():
                            has_pro_by_user[_ps.user_id] = bool(
                                _ps.tier == "pro"
                                and _ps.subscription_end
                                and _ps.subscription_end > _now_cycle
                            )
                        for _pf in _ref_db.query(_UPref).filter(_UPref.user_id.in_(_uids)).all():
                            ctrader_ok_by_user[_pf.user_id] = bool(
                                _pf.ctrader_access_token
                                and _pf.ctrader_account_id
                                and getattr(_pf, "forex_approved", False)
                            )
                    except Exception as _ref_err:
                        logger.debug(f"[FX Executor] ref preload failed: {_ref_err}")
                    finally:
                        _ref_db.close()

                # No MEXC ticker prefetch — forex uses yfinance; pass empty list.
                cycle_gate_stats: Dict[str, int] = {}

                # Shared counters for cycle-level DB health reporting.
                _cycle_db_skipped: list = []   # (strategy_id, err_name) tuples
                _preloaded_fx = _preload_strategy_users(
                    [s["id"] for s in open_snaps],
                    SessionLocal,
                    UserStrategy,
                    User,
                )

                async def _run_one_fx(snap, _http=http_client):
                    from sqlalchemy.exc import OperationalError as _SAOperationalError
                    from sqlalchemy.exc import PendingRollbackError as _SAPendingRollbackError
                    from sqlalchemy.exc import TimeoutError as _SATimeoutError
                    from app.database import bg_db_slot
                    from app.services.strategy_ta import StrategyEvalCancelled
                    async with sem:
                        async with bg_db_slot():
                            row = _preloaded_fx.get(snap["id"])
                            if not row:
                                return
                            _strategy_row, _user_row = row
                            if not _user_row or _user_row.banned:
                                return
                            if not _portal_trade_entitled(_user_row, has_pro_by_user):
                                cycle_gate_stats["blk_not_entitled"] = (
                                    cycle_gate_stats.get("blk_not_entitled", 0) + 1
                                )
                                return
                            for _attempt in (1, 3):
                                db_one = SessionLocal()
                                try:
                                    strategy = db_one.merge(_strategy_row)
                                    user = db_one.merge(_user_row)
                                    await evaluate_and_fire(
                                        strategy, user, db_one, _http,
                                        raw_tickers=[],
                                        gate_stats=cycle_gate_stats,
                                        prefetched_ctrader_ok=ctrader_ok_by_user.get(user.id, False),
                                    )
                                    return
                                except (
                                    _SAOperationalError,
                                    _SAPendingRollbackError,
                                    _SATimeoutError,
                                ) as _db_err:
                                    _err_name = type(_db_err).__name__
                                    if getattr(_db_err, "orig", None) is not None:
                                        _err_name = type(_db_err.orig).__name__
                                    if _attempt < 3:
                                        logger.warning(
                                            f"[FX Strategy {snap['id']}] Transient DB error "
                                            f"({_err_name}) — retry {_attempt}/3"
                                        )
                                        _db_rollback_safe(db_one)
                                        await asyncio.sleep(0.5 * _attempt)
                                        continue
                                    _cycle_db_skipped.append((snap["id"], _err_name))
                                    logger.warning(
                                        f"[FX Strategy {snap['id']}] Skipping cycle — "
                                        f"DB error persisted ({_err_name})"
                                    )
                                    return
                                except StrategyEvalCancelled:
                                    logger.info(
                                        f"[FX Strategy {snap['id']}] eval cancelled — "
                                        f"skipping cycle"
                                    )
                                    _db_rollback_safe(db_one)
                                    return
                                except Exception as e:
                                    logger.error(
                                        f"[FX Strategy {snap['id']}] Error: {e}", exc_info=True
                                    )
                                    _db_rollback_safe(db_one)
                                    return
                                finally:
                                    try:
                                        db_one.close()
                                    except Exception:
                                        pass

                _prefetch_ct_uid = next(
                    (uid for uid, ok in ctrader_ok_by_user.items() if ok),
                    None,
                )
                await _prefetch_price_ta_for_cycle(
                    open_snaps,
                    http_client,
                    {"forex", "index", "stock"},
                    label=_fx_lbl,
                    ctrader_user_id=_prefetch_ct_uid,
                )

                logger.info(
                    f"[{_log_ts()}] [{_fx_lbl}] evaluating {len(open_snaps)} strategies "
                    f"(batch size {EXECUTOR_SCAN_BATCH_SIZE})…"
                )
                await _gather_eval_batches(
                    _fx_lbl, open_snaps, _run_one_fx,
                )

                try:
                    from app.services.ctrader_order_queue import flush_gate_stats_to_db
                    flush_gate_stats_to_db([s["id"] for s in open_snaps])
                except Exception:
                    pass

                # Live SL amendments (auto-breakeven + trailing) for open LIVE
                # forex positions run on the dedicated run_forex_live_manager_fast
                # loop (sub-second cadence via the cTrader spot feed), so the scan
                # cycle no longer manages them here.

                # Emit one consolidated warning per cycle instead of per-strategy spam
                if _cycle_db_skipped:
                    _total = len(open_snaps)
                    _skipped = len(_cycle_db_skipped)
                    _err_types = ", ".join(sorted({e for _, e in _cycle_db_skipped}))
                    logger.warning(
                        f"[{_fx_lbl}] DB unreachable — skipped {_skipped}/{_total} "
                        f"strategies this cycle ({_err_types}). Will retry next cycle."
                    )

                if cycle_gate_stats:
                    _gate_summary = " ".join(
                        f"{k.replace('blk_', '')}={v}"
                        for k, v in sorted(cycle_gate_stats.items(), key=lambda kv: -kv[1])
                    )
                    logger.info(
                        f"[{_log_ts()}] [{_fx_lbl}] cycle gates → {_gate_summary}"
                    )
                    try:
                        from app.services.feed_diagnostics import (
                            GATE_BLOCK_LOCATIONS,
                            format_final_report,
                        )
                        _locs = [
                            f"{k}: {GATE_BLOCK_LOCATIONS.get(k, 'strategy_executor.evaluate_and_fire')}"
                            for k in cycle_gate_stats if k.startswith("blk_")
                        ]
                        logger.info(
                            format_final_report(
                                scans_running=1,
                                symbols_scanned=len(open_snaps),
                                candles_loaded=0,
                                setups_detected=0,
                                signals_blocked=cycle_gate_stats,
                                block_locations=_locs,
                            )
                        )
                    except Exception:
                        pass

                try:
                    from app.services.feed_diagnostics import (
                        flush_scan_metric_batch,
                        log_blocker_rollup,
                    )
                    flush_scan_metric_batch(_fx_lbl)
                    log_blocker_rollup(100)
                except Exception:
                    pass

                _cycle_s = (datetime.utcnow() - _cycle_t0).total_seconds()
                if open_snaps:
                    logger.info(
                        f"[{_log_ts()}] [{_fx_lbl}] cycle done in {_cycle_s:.1f}s "
                        f"({len(open_snaps)} strategies)"
                    )

            except Exception as e:
                logger.error(f"Forex executor loop error: {e}", exc_info=True)
                try:
                    from app.services.feed_diagnostics import flush_scan_metric_batch
                    flush_scan_metric_batch(_fx_lbl)
                except Exception:
                    pass
                # An outer-cycle failure (e.g. the strategy-list query or session
                # open failing — the hot-table query that preceded the prior
                # saturation incident) also counts as DB stress, so trip the
                # backoff below conservatively.
                _cycle_db_skipped.append(("__cycle__", type(e).__name__))

            # Adaptive backoff: if this cycle hit DB errors (the precursor to the
            # historical Neon-saturation cascade), space out the next scan to
            # relieve pressure; recover to the fast cadence as soon as a cycle is
            # clean. Fixed sleep otherwise preserves baseline backpressure.
            if _cycle_db_skipped:
                _sleep_s = max(FOREX_SCAN_INTERVAL_SECONDS * 3, 15)
                logger.warning(
                    f"[{_fx_lbl}] DB stress detected — backing off to {_sleep_s}s "
                    f"this cycle (normal cadence {FOREX_SCAN_INTERVAL_SECONDS}s)"
                )
            else:
                _sleep_s = FOREX_SCAN_INTERVAL_SECONDS
            await asyncio.sleep(_sleep_s)


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
    from app.db_resilience import is_transient_db_error
    from app.strategy_models import StrategyExecution
    cutoff = datetime.utcnow() - timedelta(hours=stale_after_hours)
    for attempt in range(1, 4):
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
            if is_transient_db_error(e) and attempt < 3:
                logger.warning(
                    f"close_stale_open_executions: transient DB error "
                    f"(attempt {attempt}/3) — retrying: {e}"
                )
                await asyncio.sleep(0.5 * attempt)
                continue
            logger.error(f"close_stale_open_executions: query failed: {e}")
            return 0
        finally:
            db.close()
    return 0
