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
        return []

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

async def _fetch_1m_ohlc(symbol: str, http_client: httpx.AsyncClient):
    """Fetch the last 2 completed 1m candles. Tries MEXC first, falls back to Binance."""
    sources = [
        ("https://api.mexc.com/api/v3/klines",   {"symbol": symbol, "interval": "1m", "limit": 3}),
        ("https://fapi.binance.com/fapi/v1/klines", {"symbol": symbol, "interval": "1m", "limit": 3}),
    ]
    for url, params in sources:
        try:
            resp = await http_client.get(url, params=params, timeout=5)
            if resp.status_code != 200:
                continue
            klines = resp.json()
            if not klines or len(klines) < 2:
                continue
            return {
                "high":  max(float(klines[-2][2]), float(klines[-1][2])),
                "low":   min(float(klines[-2][3]), float(klines[-1][3])),
                "close": float(klines[-1][4]),
            }
        except Exception as e:
            logger.debug(f"OHLC fetch failed ({url}) for {symbol}: {e}")
            continue
    return None


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
    Check one open paper execution against latest 1m OHLC.
    Uses candle high/low — not just spot — so scalp TP/SL hits are detected
    even if price only touched intra-candle.
    """
    if not ex.entry_price or not ex.tp_price or not ex.sl_price:
        return

    # Auto-expire very old paper positions
    if ex.fired_at and (datetime.utcnow() - ex.fired_at).total_seconds() > PAPER_MAX_HOLD_HOURS * 3600:
        _close_paper_execution(ex, "CANCELLED", ex.entry_price, db)
        return

    ohlc = await _fetch_1m_ohlc(ex.symbol, http_client)
    if not ohlc:
        return

    high  = ohlc["high"]
    low   = ohlc["low"]
    close = ohlc["close"]

    if ex.direction == "LONG":
        # Check TP first (optimistic — price may have swept TP before SL)
        if high >= ex.tp_price:
            _close_paper_execution(ex, "WIN", ex.tp_price, db)
        elif low <= ex.sl_price:
            _close_paper_execution(ex, "LOSS", ex.sl_price, db)
    else:  # SHORT
        if low <= ex.tp_price:
            _close_paper_execution(ex, "WIN", ex.tp_price, db)
        elif high >= ex.sl_price:
            _close_paper_execution(ex, "LOSS", ex.sl_price, db)


async def run_paper_position_monitor():
    """
    Background loop — monitors all open paper positions every 30s using
    1-minute Binance Futures OHLC data for maximum accuracy.
    """
    from app.database import SessionLocal
    from app.strategy_models import StrategyExecution

    logger.info("🧪 Paper position monitor started (30s interval, 1m OHLC)")
    async with httpx.AsyncClient() as http_client:
        while True:
            try:
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
                        logger.debug(f"🧪 Monitoring {len(open_papers)} open paper positions")
                        tasks = [_check_paper_position(ex, db, http_client) for ex in open_papers]
                        await asyncio.gather(*tasks, return_exceptions=True)
                finally:
                    db.close()
            except Exception as e:
                logger.error(f"Paper monitor loop error: {e}", exc_info=True)
            await asyncio.sleep(PAPER_MONITOR_INTERVAL)


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

    symbols = await _get_eligible_symbols(universe, http_client)
    if not symbols:
        return

    for symbol in symbols[:50]:
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
            config, symbol, price_data, enhanced_ta, http_client
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
