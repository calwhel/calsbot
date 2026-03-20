"""
Strategy Executor — Build Your Own Strategy Portal

Background worker that continuously evaluates active + paper user strategies
and fires trades. Paper trades are tracked with 1m OHLC accuracy — candle
high/low is used to detect TP/SL hits so scalp results are realistic.
"""
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import httpx

logger = logging.getLogger(__name__)

SCAN_INTERVAL_SECONDS  = 45    # how often to evaluate strategies
PAPER_MONITOR_INTERVAL = 30    # how often to check open paper positions (seconds)
MAX_CONCURRENT         = 5     # parallel strategy evaluations (stay within DB pool)
PAPER_MAX_HOLD_HOURS   = 168   # auto-expire paper positions after this many hours (7 days)

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
_PRICE_TA_TTL    = 30  # seconds

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
    """
    try:
        from app.models import UserPreference
        prefs = db.query(UserPreference).filter_by(user_id=user.id).first()
        if not prefs:
            return False
        if not getattr(prefs, "auto_trading_enabled", False):
            return False
        if not getattr(prefs, "bitunix_api_key", None):
            return False
        if not getattr(prefs, "bitunix_api_secret", None):
            return False
        return True
    except Exception:
        return False  # safe default — track as paper


# ─── Bitunix symbol cache ────────────────────────────────────────────────────
_BITUNIX_SYMBOLS: set = set()
_BITUNIX_SYMBOLS_FETCHED_AT: Optional[datetime] = None
_BITUNIX_CACHE_TTL = 300  # refresh every 5 minutes

async def _get_bitunix_symbols(http_client: httpx.AsyncClient) -> set:
    """Return the set of USDT-margined perpetual symbols available on Bitunix."""
    global _BITUNIX_SYMBOLS, _BITUNIX_SYMBOLS_FETCHED_AT
    now = datetime.utcnow()
    if _BITUNIX_SYMBOLS and _BITUNIX_SYMBOLS_FETCHED_AT and (now - _BITUNIX_SYMBOLS_FETCHED_AT).seconds < _BITUNIX_CACHE_TTL:
        return _BITUNIX_SYMBOLS
    try:
        resp = await http_client.get(
            "https://fapi.bitunix.com/api/v1/futures/market/tickers", timeout=8
        )
        if resp.status_code == 200:
            data = resp.json()
            syms = set()
            for t in data.get("data", []):
                sym = t.get("symbol", "")
                if sym.endswith("USDT"):
                    syms.add(sym)
            if syms:
                _BITUNIX_SYMBOLS = syms
                _BITUNIX_SYMBOLS_FETCHED_AT = now
                logger.info(f"Bitunix symbol list refreshed: {len(syms)} USDT perps")
    except Exception as e:
        logger.warning(f"Could not fetch Bitunix symbol list: {e}")
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

async def _get_raw_tickers(http_client: httpx.AsyncClient) -> Optional[list]:
    """
    Fetch the full MEXC/Binance ticker list once per TTL window.
    All strategies in the same scan cycle share this cached response —
    eliminates N parallel ticker fetches where N = number of active strategies.
    """
    global _RAW_TICKERS_CACHE, _RAW_TICKERS_AT
    now = datetime.utcnow()
    if (
        _RAW_TICKERS_CACHE is not None
        and _RAW_TICKERS_AT is not None
        and (now - _RAW_TICKERS_AT).total_seconds() < _RAW_TICKERS_TTL
    ):
        return _RAW_TICKERS_CACHE

    for url in [
        "https://api.mexc.com/api/v3/ticker/24hr",
        "https://fapi.binance.com/fapi/v1/ticker/24hr",
    ]:
        try:
            resp = await http_client.get(url, timeout=10)
            if resp.status_code == 200:
                _RAW_TICKERS_CACHE = resp.json()
                _RAW_TICKERS_AT    = now
                logger.debug(f"Ticker cache refreshed ({len(_RAW_TICKERS_CACHE)} symbols)")
                return _RAW_TICKERS_CACHE
        except Exception:
            continue
    return None


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
    # FAIL-CLOSED: if the Bitunix symbol list is empty (API failed to load),
    # we cannot verify which coins are tradeable, so we block all coins rather
    # than letting non-Bitunix coins through.  This prevents phantom alerts on
    # coins that can never be executed on Bitunix (paper or live).
    if not bitunix_symbols:
        logger.warning(
            "[eligible-symbols] Bitunix symbol list is empty — "
            "skipping evaluation cycle until list is available"
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


async def _fetch_price_and_ta(symbol: str, http_client: httpx.AsyncClient) -> Optional[Dict]:
    """
    Fetch price + TA indicators for a symbol. Results are cached per symbol for
    _PRICE_TA_TTL seconds so multiple strategies checking the same coin in one
    scan cycle share a single API call instead of making duplicate requests.
    """
    global _PRICE_TA_CACHE
    now = datetime.utcnow()
    cached = _PRICE_TA_CACHE.get(symbol)
    if cached:
        data, fetched_at = cached
        if (now - fetched_at).total_seconds() < _PRICE_TA_TTL:
            return data

    try:
        from app.services.social_signals import SocialSignalService
        svc = SocialSignalService()
        svc.http_client = http_client
        result = await svc.fetch_price_data(symbol)
        if result:
            _PRICE_TA_CACHE[symbol] = (result, now)
        return result
    except Exception as e:
        logger.debug(f"Price/TA fetch failed for {symbol}: {e}")
        return None


# ─── Guard helpers ───────────────────────────────────────────────────────────

_SESSION_HOURS = {
    "asian":    (0, 8),   "tokyo":    (0, 8),
    "london":   (7, 16),  "europe":   (7, 16),
    "new_york": (13, 22), "ny":       (13, 22),
    "overlap":  (13, 16),
}


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
            active = [name for name, (s, e) in _SESSION_HOURS.items() if s <= hour < e]
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
) -> bool:
    """
    Return True if the 1H trend aligns with `direction`.
    Uses a simple EMA-20 check on the last 30 hourly closes.
    BOTH direction always passes.  Cache results for 5 min.
    """
    if direction == "BOTH":
        return True

    now = datetime.utcnow()
    cached = _HTF_CACHE.get(symbol)
    if cached:
        is_bullish, fetched_at = cached
        if (now - fetched_at).total_seconds() < _HTF_CACHE_TTL:
            return is_bullish if direction == "LONG" else not is_bullish

    # Fetch last 30 × 1H candles
    closes: list = []
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
                    closes = [float(k[4]) for k in data]  # index 4 = close
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
    _HTF_CACHE[symbol] = (is_bullish, now)
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
    perf.win_rate      = round(len(wins) / len(closed) * 100, 1) if closed else 0.0
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
) -> list:
    """
    Fetch all 1m candles from `fired_at` to now so no TP/SL hit is ever missed.
    Returns a list of (open_ts, open, high, low, close) tuples sorted oldest-first.
    Falls back through MEXC → Binance spot → Binance futures.
    MEXC allows up to 1000 candles; we page if the window exceeds that.
    """
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
        tg_id = _telegram_int_id(user)
        if not tg_id:
            return
        asyncio.create_task(_send_paper_close_dm(
            tg_id,
            _fmt_close_card(
                strategy_name = strat.name if strat else "Your Strategy",
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
            ),
        ))
    except Exception:
        pass


def _fmt_open_card(
    strategy_name: str, symbol: str, direction: str,
    entry: float, tp_price: float, tp_pct: float,
    sl_price: float, sl_pct: float, leverage: int,
    conditions: list, is_paper: bool,
    tp2_price: float = None, tp2_pct: float = None,
    order_id: str = None,
) -> str:
    dir_icon = "🟢" if direction == "LONG" else "🔴"
    header   = "🧪 <b>YOUR STRATEGY FIRED (PAPER)</b>" if is_paper else "🚀 <b>YOUR STRATEGY IS LIVE</b>"
    bar      = "━━━━━━━━━━━━━━━━━━━━"

    tp2_line = ""
    if tp2_price and tp2_pct:
        sign = "+" if direction == "LONG" else "-"
        tp2_line = f"\nTP₂      <code>{tp2_price:.6g}</code>  ({sign}{tp2_pct:.1f}%)"

    cond_lines = ""
    if conditions:
        passed = [c for c in conditions if c.startswith("✅")]
        if passed:
            cond_lines = "\n\n<b>Why it triggered:</b>\n" + "\n".join(f"  {c}" for c in passed[:5])

    order_line = f"\n<i>Order ID: #{order_id}</i>" if order_id else ""
    footer     = "<i>📄 Paper trade · no real funds used</i>" if is_paper else "<i>✅ Live strategy trade executed</i>"

    sign_tp = "+" if direction == "LONG" else "-"
    sign_sl = "-" if direction == "LONG" else "+"

    return (
        f"{header}\n{bar}\n"
        f"📋 <b>{strategy_name}</b>\n"
        f"{dir_icon} <b>{symbol}</b>  ·  {direction}  ·  {leverage}×\n"
        f"{bar}\n"
        f"Entry    <code>{entry:.6g}</code>\n"
        f"TP₁      <code>{tp_price:.6g}</code>  ({sign_tp}{tp_pct:.1f}%){tp2_line}\n"
        f"SL       <code>{sl_price:.6g}</code>  ({sign_sl}{sl_pct:.1f}%)"
        f"{cond_lines}\n"
        f"{bar}\n"
        f"{footer}{order_line}"
    )


def _fmt_close_card(
    strategy_name: str, symbol: str, direction: str,
    entry: float, exit_price: float, outcome: str,
    pnl_pct: float, leverage: int,
    fired_at: datetime = None, closed_at: datetime = None,
    conditions: list = None,
) -> str:
    dir_icon  = "🟢" if direction == "LONG" else "🔴"
    icon      = "✅" if outcome == "WIN" else "❌"
    result    = "WIN" if outcome == "WIN" else "LOSS"
    pnl_sign  = "+" if pnl_pct >= 0 else ""
    bar       = "━━━━━━━━━━━━━━━━━━━━"

    hit_label = "TP hit 🎯" if outcome == "WIN" else "SL hit 🛑"

    duration_line = ""
    if fired_at and closed_at:
        secs  = int((closed_at - fired_at).total_seconds())
        hours, rem = divmod(secs, 3600)
        mins       = rem // 60
        if hours:
            duration_line = f"\nDuration {hours}h {mins}m"
        else:
            duration_line = f"\nDuration {mins}m"

    cond_lines = ""
    if conditions:
        passed = [c for c in conditions if c.startswith("✅")]
        if passed:
            cond_lines = "\n<b>Triggered by:</b>\n" + "\n".join(f"  {c}" for c in passed[:3]) + "\n"

    return (
        f"{icon} <b>STRATEGY {result}: {strategy_name}</b>\n{bar}\n"
        f"{dir_icon} <b>{symbol}</b>  ·  {direction}\n"
        f"{bar}\n"
        f"Entry    <code>{entry:.6g}</code>\n"
        f"Exit     <code>{exit_price:.6g}</code>  ({hit_label})\n"
        f"P&L      <b>{pnl_sign}{pnl_pct}%</b>  ({leverage}× leverage)"
        f"{duration_line}\n"
        f"{bar}\n"
        f"{cond_lines}"
        f"<i>📄 Paper trade result</i>"
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

    # ── Candle evaluation FIRST ───────────────────────────────────────────────
    # Always scan for a TP/SL hit before considering expiry.  This prevents
    # incorrectly CANCELLING a trade whose TP was already hit but whose
    # fired_at is older than PAPER_MAX_HOLD_HOURS.
    if candles:
        entry_ms = int(fired_at.timestamp() * 1000)
        relevant = [c for c in candles if c[0] >= entry_ms - 60_000]
        if relevant:
            for _ts, open_, high, low, close in relevant:
                if ex.direction == "LONG":
                    tp_hit = high >= ex.tp_price
                    sl_hit = low  <= ex.sl_price
                    if tp_hit and sl_hit:
                        # Same-candle: use direction to infer order
                        if close >= open_:   # bullish → rose first → TP first
                            _close_paper_execution(ex, "WIN", ex.tp_price, db)
                        else:                # bearish → fell first → SL first
                            _close_paper_execution(ex, "LOSS", ex.sl_price, db)
                        return True
                else:  # SHORT
                    tp_hit = low  <= ex.tp_price
                    sl_hit = high >= ex.sl_price
                    if tp_hit and sl_hit:
                        # Same-candle: use direction to infer order
                        if close <= open_:   # bearish → fell first → TP first
                            _close_paper_execution(ex, "WIN", ex.tp_price, db)
                        else:                # bullish → rose first → SL first
                            _close_paper_execution(ex, "LOSS", ex.sl_price, db)
                        return True

                if tp_hit:
                    _close_paper_execution(ex, "WIN", ex.tp_price, db)
                    return True
                if sl_hit:
                    _close_paper_execution(ex, "LOSS", ex.sl_price, db)
                    return True

            # No TP/SL hit — update unrealised notes
            last_close = relevant[-1][4]
            if ex.direction == "LONG":
                unreal = (last_close - ex.entry_price) / ex.entry_price * 100
            else:
                unreal = (ex.entry_price - last_close) / ex.entry_price * 100
            ex.notes = f"open · unrealised {'+' if unreal >= 0 else ''}{unreal:.2f}% · last {last_close:.6g}"
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
    candles  = await _fetch_candles_since_entry(ex.symbol, fired_at, http_client)
    _evaluate_paper_position_against_candles(ex, candles, db)


async def run_paper_position_monitor():
    """
    Background loop — monitors all open paper positions every 30s using
    1-minute Binance Futures OHLC data for maximum accuracy.
    """
    from app.database import SessionLocal
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
        from collections import defaultdict
        by_symbol: dict = defaultdict(list)
        for ex in open_papers:
            by_symbol[ex.symbol].append(ex)

        logger.info(
            f"🧪 Sweeping {len(open_papers)} open paper position(s) "
            f"across {len(by_symbol)} symbol(s)"
        )

        async def _fetch_for_symbol(symbol: str, positions: list):
            earliest = min((ex.fired_at or datetime.utcnow()) for ex in positions)
            candles  = await _fetch_candles_since_entry(symbol, earliest, http_client)
            return symbol, positions, candles

        fetch_results = await asyncio.gather(
            *[_fetch_for_symbol(sym, pos) for sym, pos in by_symbol.items()],
            return_exceptions=True,
        )

        # ── Phase 3: Evaluate + write (brief session per position) ────────────
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
    from app.database import SessionLocal
    from app.strategy_models import StrategyExecution, UserStrategy
    from app.models import User

    logger.info("🔴 Live position monitor started (30s interval)")

    LIVE_MONITOR_INTERVAL = 30

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
            if user and user.telegram_id:
                emoji   = "✅" if outcome == "WIN" else "🛑"
                label   = "TP HIT" if outcome == "WIN" else "SL HIT"
                ticker  = ex.symbol.replace("USDT", "")
                elapsed = ""
                if ex.fired_at:
                    mins = int((datetime.utcnow() - ex.fired_at).total_seconds() / 60)
                    elapsed = f"\nDuration: {mins}m"
                msg = (
                    f"{emoji} <b>{label} — ${ticker} {ex.direction}</b>\n"
                    f"Strategy: {strat.name if strat else 'Unknown'}\n"
                    f"Entry: <code>{ex.entry_price}</code> → Exit: <code>{exit_price}</code>\n"
                    f"Result: <b>{pnl_pct:+.1f}%</b> (leverage {leverage}×)"
                    f"{elapsed}"
                )
                tg_id = _telegram_int_id(user)
                if tg_id:
                    await _send_paper_close_dm(tg_id, msg)
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

        # ── Phase 2: Candle-based TP/SL evaluation (same accuracy as paper mode) ─
        # Fetch 1m OHLCV candles per symbol since the earliest open position.
        # Using candle HIGH/LOW means any wick that touched TP or SL is detected,
        # even if it happened between two polling cycles.
        price_closed_ids: set = set()

        from collections import defaultdict as _dd
        by_symbol_live: dict = _dd(list)
        for ex in open_lives:
            if ex.entry_price and ex.tp_price and ex.sl_price:
                by_symbol_live[ex.symbol].append(ex)

        async def _fetch_live_candles(symbol: str, positions: list):
            earliest = min((ex.fired_at or datetime.utcnow()) for ex in positions)
            candles  = await _fetch_candles_since_entry(symbol, earliest, http_client)
            return symbol, positions, candles

        candle_results = await asyncio.gather(
            *[_fetch_live_candles(sym, pos) for sym, pos in by_symbol_live.items()],
            return_exceptions=True,
        )

        for result in candle_results:
            if isinstance(result, Exception):
                logger.warning(f"[live-monitor] Candle fetch error: {result}")
                continue
            symbol, positions, candles = result
            live_px = live_prices.get(symbol)

            for ex in positions:
                outcome    = None
                exit_price = None

                # ── Candle HIGH/LOW check (catches wicks between polls) ──────
                if candles and ex.fired_at:
                    _entry_ms = int(ex.fired_at.timestamp() * 1000)
                    relevant  = [c for c in candles if c[0] >= _entry_ms - 60_000]
                    for _ts, open_, high, low, close in relevant:
                        if ex.direction == "LONG":
                            tp_hit = high >= ex.tp_price
                            sl_hit = low  <= ex.sl_price
                        else:
                            tp_hit = low  <= ex.tp_price
                            sl_hit = high >= ex.sl_price

                        if tp_hit and sl_hit:
                            # Same-candle: infer order from candle direction
                            if ex.direction == "LONG":
                                outcome, exit_price = ("WIN", ex.tp_price) if close >= open_ else ("LOSS", ex.sl_price)
                            else:
                                outcome, exit_price = ("WIN", ex.tp_price) if close <= open_ else ("LOSS", ex.sl_price)
                        elif tp_hit:
                            outcome, exit_price = "WIN", ex.tp_price
                        elif sl_hit:
                            outcome, exit_price = "LOSS", ex.sl_price

                        if outcome:
                            break

                # ── Phase 3a: Write close (brief session per position) ───────
                if outcome:
                    write_db = SessionLocal()
                    try:
                        closed = await _close_live_execution_and_notify(
                            ex, outcome, exit_price, write_db, source="candle"
                        )
                        if closed:
                            price_closed_ids.add(ex.id)
                    finally:
                        write_db.close()
                elif live_px and ex.entry_price:
                    # Update unrealised note using live spot price
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

                        if hist_entry > 0 and ex.entry_price and ex.entry_price > 0:
                            entry_diff_pct = abs(hist_entry - ex.entry_price) / ex.entry_price
                            if entry_diff_pct > 0.02:
                                logger.warning(
                                    f"[live-monitor] {ex.symbol} id={ex.id} — history entry "
                                    f"{hist_entry} differs from ours {ex.entry_price} by "
                                    f"{entry_diff_pct*100:.2f}% — stale, skipping"
                                )
                                _reconcile_missing.pop(ex.id, None)
                                continue

                        _reconcile_missing.pop(ex.id, None)

                        if realized_pnl > 0:
                            outcome = "WIN"
                        elif realized_pnl < 0:
                            outcome = "LOSS"
                        elif ex.tp_price and ex.sl_price:
                            dist_tp = abs(exit_price - ex.tp_price)
                            dist_sl = abs(exit_price - ex.sl_price)
                            outcome = "WIN" if dist_tp <= dist_sl else "LOSS"
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
):
    """
    Evaluate one strategy. Fires a trade if conditions are met.
    paper=True strategies fire but skip Bitunix order placement.
    raw_tickers — pass the pre-fetched ticker list from the main loop so all
    strategies in one cycle share a single MEXC/Binance fetch instead of each
    making their own request.
    """
    from app.services.strategy_ta import evaluate_strategy_conditions
    from app.strategy_models import StrategyExecution, StrategyPortalSettings

    # Determine paper/live upfront.
    # Live strategies automatically downgrade to paper if the user doesn't have
    # Bitunix auto-trading set up — so every signal is always tracked.
    _wants_live = (strategy.status == "active")
    if _wants_live:
        is_paper = not _user_can_live_trade(user, db)
        if is_paper:
            logger.debug(
                f"[Strategy {strategy.id}] Live strategy downgraded to paper "
                f"(no Bitunix API keys / auto-trading disabled) — signal will still be tracked."
            )
    else:
        is_paper = True   # paper / draft / paused all track as paper

    config   = dict(strategy.config or {})

    # Locked strategy — fetch live entry_conditions from the original source strategy
    if config.get("_locked") and config.get("_source_strategy_id"):
        try:
            from app.strategy_models import UserStrategy as _US
            _src = db.query(_US).filter(_US.id == config["_source_strategy_id"]).first()
            if _src and _src.config:
                config["entry_conditions"] = _src.config.get("entry_conditions", {})
        except Exception as _e:
            logger.warning(f"Locked strategy {strategy.id}: could not fetch source conditions: {_e}")

    risk     = config.get("risk", {})
    filters  = config.get("filters", {})
    universe = config.get("universe", {})
    direction_pref = config.get("direction", "LONG")

    if not _check_time_filter(filters):
        return
    if not _check_btc_regime(filters):
        return

    max_per_day   = int(risk.get("max_trades_per_day", 3))
    max_open      = int(risk.get("max_open_positions", 1))
    cooldown_mins = int(risk.get("cooldown_minutes", 30))

    if _daily_execution_count(strategy.id, db) >= max_per_day:
        return
    if _open_execution_count(strategy.id, db) >= max_open:
        return

    # Global cooldown: if ANY symbol fired within cooldown window, pause entire strategy
    last_global = _last_any_fired_time(strategy.id, db)
    if last_global:
        elapsed_global = (datetime.utcnow() - last_global).total_seconds() / 60
        if elapsed_global < cooldown_mins:
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

    symbols = await _get_eligible_symbols(universe, http_client, raw_tickers=raw_tickers)
    if not symbols:
        return

    no_duplicate_symbol = bool(risk.get("no_duplicate_symbol", False))

    for symbol in symbols[:50]:
        if no_duplicate_symbol and _fired_today_for_symbol(strategy.id, symbol, db):
            continue

        last_fired = _last_fired_time(strategy.id, symbol, db)
        if last_fired:
            elapsed_mins = (datetime.utcnow() - last_fired).total_seconds() / 60
            if elapsed_mins < cooldown_mins:
                continue

        price_data = await _fetch_price_and_ta(symbol, http_client)
        if not price_data:
            continue

        # HTF (1H) trend filter — skip if symbol is trending against entry direction
        if filters.get("htf_trend"):
            if not await _check_htf_trend(symbol, direction_pref, http_client):
                logger.debug(f"[Strategy {strategy.id}] HTF trend filter blocked {symbol}")
                continue

        enhanced_ta   = price_data.get("enhanced_ta", {})
        current_price = price_data.get("price", 0)
        if not current_price:
            continue

        passed, details = await evaluate_strategy_conditions(
            config, symbol, price_data, enhanced_ta, http_client,
            strictness_level=strictness_level
        )
        if not passed:
            continue

        mode_tag = "🧪 [PAPER]" if is_paper else "🎯"
        logger.info(
            f"{mode_tag} [Strategy {strategy.id}] {strategy.name} — "
            f"{symbol} conditions met! {direction_pref}"
        )

        ex_config = config.get("exit", {})
        tp_pct    = float(ex_config.get("take_profit_pct",  3.0))
        tp2_pct   = ex_config.get("take_profit2_pct")
        sl_pct    = float(ex_config.get("stop_loss_pct",    1.5))
        leverage  = int(risk.get("leverage", 10))

        direction = direction_pref
        if direction == "BOTH":
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
                        ),
                    )
            except Exception as e:
                logger.warning(f"Paper DM failed: {e}")
        else:
            # Live trade: place on Bitunix
            order_id = None
            try:
                from app.services.strategy_trader import place_bitunix_order_for_user
                ps_type = risk.get("position_size_type", "pct")
                order_id = await place_bitunix_order_for_user(
                    user        = user,
                    symbol      = symbol,
                    direction   = direction,
                    leverage    = leverage,
                    entry_price = current_price,
                    tp_pct      = tp_pct,
                    sl_pct      = sl_pct,
                    risk_pct    = float(risk.get("position_size_pct", 5)),
                    risk_usd    = float(risk["position_size_usd"]) if ps_type == "fixed" and risk.get("position_size_usd") else None,
                )
            except Exception as e:
                logger.error(f"[Strategy {strategy.id}] Order error: {e}")
                # Don't cancel — flip to paper so the signal's ROI is still tracked.
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
                            f"⚠️ <b>Bitunix error — paper trade started</b>\n"
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
                break  # paper execution is now open — stop processing matches

            if order_id:
                execution.bitunix_order_id = str(order_id)
                db.commit()
                tg_id_live = _telegram_int_id(user)
                if tg_id_live:
                    try:
                        await _tg_send(
                            tg_id_live,
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
                                is_paper      = False,
                                order_id      = str(order_id),
                            ),
                        )
                    except Exception as e:
                        logger.warning(f"Live DM failed: {e}")
            else:
                # Fallback: order_id was None (shouldn't normally reach here post-refactor)
                execution.is_paper = True
                execution.notes    = "Live→Paper fallback: Bitunix returned no order_id"
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
                            f"⚠️ <b>Bitunix order not confirmed — paper trade started</b>\n"
                            f"Strategy: <b>{strategy.name}</b>\n"
                            f"Signal: {symbol.replace('USDT','')} {direction} {leverage}× lev\n"
                            f"Entry: <code>${current_price:,.4f}</code>\n"
                            f"TP: <code>${tp_price:,.4f}</code> (+{tp_pct}%)  "
                            f"SL: <code>${sl_price:,.4f}</code> (-{sl_pct}%)\n\n"
                            f"<i>Bitunix did not return an order ID. The signal is being tracked "
                            f"as a 🧪 paper position. Check your API key has Futures trading permission.</i>"
                        )
                    except Exception:
                        pass

        break  # one trade per strategy per scan cycle


# ─── Main executor loop ──────────────────────────────────────────────────────

async def run_strategy_executor():
    """
    Main background loop. Evaluates all active + paper strategies for all users.
    Also spawns the paper position monitor as a sibling task.
    """
    from app.database import SessionLocal, engine
    from app.models import User
    from app.strategy_models import UserStrategy, init_strategy_tables

    init_strategy_tables(engine)
    logger.info("🤖 Strategy executor started (active + paper modes)")

    # Spawn paper monitor as a concurrent task
    asyncio.create_task(run_paper_position_monitor())

    sem = asyncio.Semaphore(MAX_CONCURRENT)

    # Explicit timeouts: 15s total, 5s connect — prevents hung requests
    # from blocking semaphore slots indefinitely.
    _timeout = httpx.Timeout(15.0, connect=5.0)

    async with httpx.AsyncClient(timeout=_timeout) as http_client:
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
                logger.info(
                    f"🤖 Strategy executor: {active_count} live · {paper_count} paper"
                )

                # Pre-fetch tickers ONCE for the entire cycle.
                shared_tickers = await _get_raw_tickers(http_client)

                async def _run_one(snap, _tickers=shared_tickers):
                    """Each strategy evaluation runs in its own isolated DB session."""
                    async with sem:
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
                                raw_tickers=_tickers
                            )
                        except Exception as e:
                            logger.error(
                                f"[Strategy {snap['id']}] Error: {e}", exc_info=True
                            )
                        finally:
                            db_one.close()

                await asyncio.gather(*[_run_one(s) for s in strategy_snapshots])

            except Exception as e:
                logger.error(f"Strategy executor loop error: {e}", exc_info=True)

            await asyncio.sleep(SCAN_INTERVAL_SECONDS)


async def backfill_cancelled_paper_trades(lookback_days: int = 30) -> int:
    """
    One-time fix: re-evaluate paper trades incorrectly marked CANCELLED because
    MEXC candle fetches were missing the startTime parameter.
    Returns the number of trades corrected.
    """
    from app.database import SessionLocal
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
                candles = await _fetch_candles_since_entry(ex.symbol, ex.fired_at, client)
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
    from app.database import SessionLocal
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

                candles = await _fetch_candles_since_entry(ex.symbol, ex.fired_at, client)
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
