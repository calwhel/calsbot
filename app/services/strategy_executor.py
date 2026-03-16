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
MAX_CONCURRENT         = 5     # parallel strategy evaluations
PAPER_MAX_HOLD_HOURS   = 48    # auto-expire paper positions after this many hours

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

    # For pinned coins that exist in the MEXC ticker but aren't on Bitunix yet,
    # still include them so paper testing works and the executor can attempt the
    # order (live orders will simply fail gracefully if the market is unavailable).
    if is_pinned and specific:
        ticker_map = {t.get("symbol", ""): t for t in tickers if t.get("symbol", "").endswith("USDT")}
        pinned_found = []
        for sym in specific:
            if not sym.endswith("USDT"):
                sym += "USDT"
            if sym in ticker_map:
                pinned_found.append(sym)
            # If not in MEXC tickers at all, skip silently
        return pinned_found

    symbols = []
    for t in tickers:
        sym = t.get("symbol", "")
        if not sym.endswith("USDT"):
            continue
        # Only include coins tradeable on Bitunix perpetuals
        if bitunix_symbols and sym not in bitunix_symbols:
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
    Returns a list of (open_ts, high, low, close) tuples sorted oldest-first.
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
        # Convert to ms epoch for Binance; MEXC uses limit only
        start_ms = int(start.timestamp() * 1000)

        sources = [
            ("https://api.mexc.com/api/v3/klines",
             {"symbol": symbol, "interval": "1m", "limit": needed}),
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
                    high  = float(k[2])
                    low   = float(k[3])
                    close = float(k[4])
                    all_candles.append((ts, high, low, close))
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
    """Mark a paper execution as closed and update performance."""
    tp_pct = ex.tp_price and ex.entry_price and ex.tp_price != ex.entry_price
    sl_pct = ex.sl_price and ex.entry_price and ex.sl_price != ex.entry_price

    if ex.direction == "LONG":
        raw_pnl = (exit_price - ex.entry_price) / ex.entry_price * 100
    else:
        raw_pnl = (ex.entry_price - exit_price) / ex.entry_price * 100

    ex.outcome    = outcome
    ex.exit_price = exit_price
    ex.pnl_pct    = round(raw_pnl * ex.leverage, 2)
    ex.closed_at  = datetime.utcnow()
    db.commit()
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
                pnl_pct       = ex.pnl_pct,
                leverage      = ex.leverage,
                fired_at      = ex.fired_at,
                closed_at     = ex.closed_at,
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
    footer     = "<i>📄 Paper trade · no real funds used</i>" if is_paper else "<i>✅ Trade placed on Bitunix</i>"

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
    """
    if not ex.entry_price or not ex.tp_price or not ex.sl_price:
        return False

    fired_at = ex.fired_at or datetime.utcnow()

    # Auto-expire very old paper positions
    if (datetime.utcnow() - fired_at).total_seconds() > PAPER_MAX_HOLD_HOURS * 3600:
        _close_paper_execution(ex, "CANCELLED", ex.entry_price, db)
        return True

    if not candles:
        return False

    # Only consider candles from this position's entry onwards
    entry_ms = int(fired_at.timestamp() * 1000)
    relevant = [c for c in candles if c[0] >= entry_ms]
    if not relevant:
        relevant = candles  # fallback — use all if timestamps uncertain

    for _ts, high, low, close in relevant:
        if ex.direction == "LONG":
            tp_hit = high >= ex.tp_price
            sl_hit = low  <= ex.sl_price
        else:
            tp_hit = low  <= ex.tp_price
            sl_hit = high >= ex.sl_price

        if tp_hit and sl_hit:
            _close_paper_execution(ex, "WIN", ex.tp_price, db)
            return True
        if tp_hit:
            _close_paper_execution(ex, "WIN", ex.tp_price, db)
            return True
        if sl_hit:
            _close_paper_execution(ex, "LOSS", ex.sl_price, db)
            return True

    last_close = relevant[-1][3]
    if ex.direction == "LONG":
        unreal = (last_close - ex.entry_price) / ex.entry_price * 100
    else:
        unreal = (ex.entry_price - last_close) / ex.entry_price * 100
    ex.notes = f"open · unrealised {'+' if unreal >= 0 else ''}{unreal:.2f}% · last {last_close:.6g}"
    try:
        db.commit()
    except Exception:
        db.rollback()
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
        Positions are grouped by symbol so candles are fetched once per unique
        coin instead of once per position — at 20 users with 5 strategies each
        this typically cuts candle API calls by 5–10× vs the naïve approach.
        """
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
            if not open_papers:
                return

            # Group positions by symbol
            from collections import defaultdict
            by_symbol: dict = defaultdict(list)
            for ex in open_papers:
                by_symbol[ex.symbol].append(ex)

            logger.info(
                f"🧪 Sweeping {len(open_papers)} open paper position(s) "
                f"across {len(by_symbol)} symbol(s)"
            )

            async def _check_symbol_group(symbol: str, positions: list):
                # Find the earliest entry so we fetch all relevant candles
                earliest = min(
                    (ex.fired_at or datetime.utcnow()) for ex in positions
                )
                candles = await _fetch_candles_since_entry(symbol, earliest, http_client)
                for ex in positions:
                    try:
                        _evaluate_paper_position_against_candles(ex, candles, db)
                    except Exception as e:
                        logger.warning(f"Position {ex.id} eval error: {e}")

            tasks = [
                _check_symbol_group(sym, positions)
                for sym, positions in by_symbol.items()
            ]
            await asyncio.gather(*tasks, return_exceptions=True)
        finally:
            db.close()

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

_live_notified: set = set()   # guard against double alerts for live positions
_NOTIFIED_MAX   = 20_000      # cap both sets so they don't grow unbounded

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

    async def _sweep_live(http_client):
        db = SessionLocal()
        try:
            open_lives = (
                db.query(StrategyExecution)
                .filter(
                    StrategyExecution.outcome == "OPEN",
                    StrategyExecution.is_paper == False,
                )
                .all()
            )
            if not open_lives:
                return

            symbols     = list({ex.symbol for ex in open_lives})
            live_prices = await _fetch_live_price_batch(symbols, http_client)

            logger.info(
                f"[live-monitor] Checking {len(open_lives)} live position(s) "
                f"across {len(symbols)} symbol(s)"
            )

            for ex in open_lives:
                live_px = live_prices.get(ex.symbol)
                if not live_px or not ex.entry_price or not ex.sl_price:
                    continue

                leverage = ex.leverage or 10

                # SL uses a 0.2 % confirmation buffer to avoid firing on brief
                # last-price wicks that never trigger Bitunix's mark-price SL.
                SL_BUFFER = 0.002
                if ex.direction == "LONG":
                    pnl_pct = (live_px - ex.entry_price) / ex.entry_price * 100 * leverage
                    sl_hit  = live_px <= ex.sl_price * (1 - SL_BUFFER)
                    tp_hit  = ex.tp_price and live_px >= ex.tp_price
                else:
                    pnl_pct = (ex.entry_price - live_px) / ex.entry_price * 100 * leverage
                    sl_hit  = live_px >= ex.sl_price * (1 + SL_BUFFER)
                    tp_hit  = ex.tp_price and live_px <= ex.tp_price

                outcome    = None
                exit_price = None

                if tp_hit and ex.tp_price:
                    outcome    = "WIN"
                    exit_price = ex.tp_price
                elif sl_hit:
                    outcome    = "LOSS"
                    exit_price = ex.sl_price

                if outcome:
                    raw_pnl = (
                        (exit_price - ex.entry_price) / ex.entry_price * 100
                        if ex.direction == "LONG"
                        else (ex.entry_price - exit_price) / ex.entry_price * 100
                    )
                    ex.outcome    = outcome
                    ex.exit_price = exit_price
                    ex.pnl_pct    = round(raw_pnl * leverage, 2)
                    ex.closed_at  = datetime.utcnow()
                    try:
                        db.commit()
                        _update_performance(ex.strategy_id, db)
                    except Exception as ce:
                        logger.error(f"[live-monitor] DB commit error {ex.id}: {ce}")
                        db.rollback()
                        continue

                    logger.info(
                        f"[live-monitor] {'TP' if outcome == 'WIN' else 'SL'} HIT: "
                        f"{ex.symbol} {ex.direction} entry={ex.entry_price} "
                        f"exit={exit_price} pnl={ex.pnl_pct:+.1f}%"
                    )

                    # Telegram DM to user
                    alert_key = f"live-{ex.id}-{outcome}"
                    if alert_key not in _live_notified:
                        if len(_live_notified) >= _NOTIFIED_MAX:
                            _live_notified.clear()  # reset rather than grow unbounded
                        _live_notified.add(alert_key)
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
                                    f"Result: <b>{ex.pnl_pct:+.1f}%</b> (leverage {leverage}×)"
                                    f"{elapsed}"
                                )
                                tg_id = _telegram_int_id(user)
                                if tg_id:
                                    await _send_paper_close_dm(tg_id, msg)
                        except Exception as ne:
                            logger.warning(f"[live-monitor] Notification error: {ne}")

                else:
                    # Update unrealised P&L note
                    ex.notes = (
                        f"open · live={live_px:.6g} · "
                        f"unrealised {'+' if pnl_pct >= 0 else ''}{pnl_pct:.1f}%"
                    )
                    try:
                        db.commit()
                    except Exception:
                        db.rollback()

        except Exception as e:
            logger.error(f"[live-monitor] sweep error: {e}", exc_info=True)
        finally:
            db.close()

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

    is_paper = (strategy.status == "paper")
    config   = strategy.config
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
                order_id = await place_bitunix_order_for_user(
                    user        = user,
                    symbol      = symbol,
                    direction   = direction,
                    leverage    = leverage,
                    entry_price = current_price,
                    tp_pct      = tp_pct,
                    sl_pct      = sl_pct,
                    risk_pct    = float(risk.get("position_size_pct", 5)),
                )
            except Exception as e:
                logger.error(f"[Strategy {strategy.id}] Order error: {e}")
                execution.outcome = "CANCELLED"
                execution.notes   = str(e)
                db.commit()
                continue

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
                            if not (
                                user.is_admin
                                or user.grandfathered
                                or (user.subscription_end and user.subscription_end > datetime.utcnow())
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
