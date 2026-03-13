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

async def _get_eligible_symbols(universe: Dict, http_client: httpx.AsyncClient) -> List[str]:
    from app.services.social_signals import SLOW_HIGHCAP_BLOCKED

    # Fetch Bitunix-available symbols in parallel with the price tickers
    bitunix_task = asyncio.create_task(_get_bitunix_symbols(http_client))

    tickers = None
    # Try MEXC spot first (not geo-blocked on Replit), fall back to Binance futures
    for url in [
        "https://api.mexc.com/api/v3/ticker/24hr",
        "https://fapi.binance.com/fapi/v1/ticker/24hr",
    ]:
        try:
            resp = await http_client.get(url, timeout=10)
            if resp.status_code == 200:
                tickers = resp.json()
                break
        except Exception:
            continue
    if not tickers:
        bitunix_task.cancel()
        return []

    bitunix_symbols = await bitunix_task  # wait for Bitunix list

    sym_type  = universe.get("type", "all")
    specific  = {s.upper() for s in universe.get("symbols", [])}
    excl_slow = universe.get("exclude_slow_highcap", True)
    min_vol   = float(universe.get("min_volume_usd", 500_000))
    min_chg   = universe.get("min_24h_change")
    max_chg   = universe.get("max_24h_change")

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
    try:
        from app.services.social_signals import SocialSignalService
        svc = SocialSignalService()
        svc.http_client = http_client
        return await svc.fetch_price_data(symbol)
    except Exception as e:
        logger.debug(f"Price/TA fetch failed for {symbol}: {e}")
        return None


# ─── Guard helpers ───────────────────────────────────────────────────────────

def _check_time_filter(filters: Dict) -> bool:
    tf = filters.get("time_filter")
    if not tf:
        return True
    hour = datetime.utcnow().hour
    return tf.get("start_hour", 0) <= hour < tf.get("end_hour", 24)


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
        asyncio.create_task(_send_paper_close_dm(
            int(user.telegram_id),
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


async def _check_paper_position(ex, db, http_client: httpx.AsyncClient):
    """
    Check one open paper execution against ALL 1m candles since entry.
    Scans chronologically so we correctly identify which candle first hit TP or SL,
    and assign the right exit price. No hit is ever missed due to restarts or gaps.
    """
    if not ex.entry_price or not ex.tp_price or not ex.sl_price:
        return

    fired_at = ex.fired_at or datetime.utcnow()

    # Auto-expire very old paper positions
    if (datetime.utcnow() - fired_at).total_seconds() > PAPER_MAX_HOLD_HOURS * 3600:
        _close_paper_execution(ex, "CANCELLED", ex.entry_price, db)
        return

    candles = await _fetch_candles_since_entry(ex.symbol, fired_at, http_client)
    if not candles:
        return

    # Scan candles chronologically — find the FIRST candle that hit TP or SL
    for _ts, high, low, close in candles:
        if ex.direction == "LONG":
            tp_hit = high >= ex.tp_price
            sl_hit = low  <= ex.sl_price
        else:  # SHORT
            tp_hit = low  <= ex.tp_price
            sl_hit = high >= ex.sl_price

        if tp_hit and sl_hit:
            # Both in same candle — TP wins (optimistic, assume price ran up first)
            _close_paper_execution(ex, "WIN", ex.tp_price, db)
            return
        if tp_hit:
            _close_paper_execution(ex, "WIN", ex.tp_price, db)
            return
        if sl_hit:
            _close_paper_execution(ex, "LOSS", ex.sl_price, db)
            return

    # No TP/SL hit yet — update notes with current unrealised P&L
    last_close = candles[-1][3]
    if ex.direction == "LONG":
        unreal = (last_close - ex.entry_price) / ex.entry_price * 100
    else:
        unreal = (ex.entry_price - last_close) / ex.entry_price * 100
    ex.notes = f"open · unrealised {'+' if unreal >= 0 else ''}{unreal:.2f}% · last {last_close:.6g}"
    try:
        db.commit()
    except Exception:
        db.rollback()


async def run_paper_position_monitor():
    """
    Background loop — monitors all open paper positions every 30s using
    1-minute Binance Futures OHLC data for maximum accuracy.
    """
    from app.database import SessionLocal
    from app.strategy_models import StrategyExecution

    logger.info("🧪 Paper position monitor started (30s interval, full-history candle scan)")

    async def _sweep(http_client):
        """One full sweep of all open paper positions."""
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
            if open_papers:
                logger.info(f"🧪 Sweeping {len(open_papers)} open paper position(s)")
                tasks = [_check_paper_position(ex, db, http_client) for ex in open_papers]
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


# ─── Strategy evaluation & firing ───────────────────────────────────────────

async def evaluate_and_fire(strategy, user, db, http_client: httpx.AsyncClient):
    """
    Evaluate one strategy. Fires a trade if conditions are met.
    paper=True strategies fire but skip Bitunix order placement.
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

    symbols = await _get_eligible_symbols(universe, http_client)
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
                if not portal_settings or portal_settings.dm_paper_alerts:
                    await _tg_send(
                        int(user.telegram_id),
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
                    tp_price    = tp_price,
                    sl_price    = sl_price,
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
                try:
                    await _tg_send(
                        int(user.telegram_id),
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

    async with httpx.AsyncClient() as http_client:
        while True:
            try:
                db = SessionLocal()
                try:
                    strategies = (
                        db.query(UserStrategy)
                        .filter(UserStrategy.status.in_(["active", "paper"]))
                        .all()
                    )

                    if not strategies:
                        await asyncio.sleep(SCAN_INTERVAL_SECONDS)
                        continue

                    active_count = sum(1 for s in strategies if s.status == "active")
                    paper_count  = sum(1 for s in strategies if s.status == "paper")
                    logger.info(
                        f"🤖 Strategy executor: {active_count} live · {paper_count} paper"
                    )

                    async def _run_one(strategy):
                        async with sem:
                            user = db.query(User).filter(User.id == strategy.user_id).first()
                            if not user or user.banned:
                                return
                            if not (
                                user.is_admin
                                or user.grandfathered
                                or (user.subscription_end and user.subscription_end > datetime.utcnow())
                            ):
                                return
                            try:
                                await evaluate_and_fire(strategy, user, db, http_client)
                            except Exception as e:
                                logger.error(f"[Strategy {strategy.id}] Error: {e}", exc_info=True)

                    await asyncio.gather(*[_run_one(s) for s in strategies])
                finally:
                    db.close()

            except Exception as e:
                logger.error(f"Strategy executor loop error: {e}", exc_info=True)

            await asyncio.sleep(SCAN_INTERVAL_SECONDS)
