"""
Strategy Executor — Build Your Own Strategy Portal

Background worker that continuously evaluates active user strategies
and fires trades to each user's Bitunix account when conditions are met.
Runs independently of the broadcast scanner — no shared state.
"""
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import httpx

logger = logging.getLogger(__name__)

SCAN_INTERVAL_SECONDS = 45   # how often to poll per strategy
MAX_CONCURRENT        = 5    # parallel strategy evaluations


async def _get_eligible_symbols(universe: Dict, http_client: httpx.AsyncClient) -> List[str]:
    """Return tradeable symbols matching the strategy universe config."""
    from app.services.social_signals import SLOW_HIGHCAP_BLOCKED

    try:
        resp = await http_client.get(
            "https://fapi.binance.com/fapi/v1/ticker/24hr",
            timeout=10,
        )
        if resp.status_code != 200:
            return []
        tickers = resp.json()
    except Exception:
        return []

    sym_type     = universe.get("type", "all")
    specific     = {s.upper() for s in universe.get("symbols", [])}
    excl_slow    = universe.get("exclude_slow_highcap", True)
    min_vol      = float(universe.get("min_volume_usd", 500_000))
    min_chg      = universe.get("min_24h_change")
    max_chg      = universe.get("max_24h_change")

    symbols = []
    for t in tickers:
        sym = t.get("symbol", "")
        if not sym.endswith("USDT"):
            continue
        if sym_type == "specific" and sym not in specific:
            continue
        base = sym.replace("USDT", "")
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
    """Fetch price_data + enhanced_ta for a symbol (minimal, fast version)."""
    try:
        from app.services.social_signals import SocialSignalService
        svc = SocialSignalService()
        svc.http_client = http_client
        return await svc.fetch_price_data(symbol)
    except Exception as e:
        logger.debug(f"Price/TA fetch failed for {symbol}: {e}")
        return None


def _check_time_filter(filters: Dict) -> bool:
    """Return True if current UTC hour is within allowed window."""
    tf = filters.get("time_filter")
    if not tf:
        return True
    hour = datetime.utcnow().hour
    return tf.get("start_hour", 0) <= hour < tf.get("end_hour", 24)


def _check_btc_regime(filters: Dict) -> bool:
    """Return True if BTC regime matches strategy requirement."""
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
    """Count how many trades this strategy fired today."""
    from app.strategy_models import StrategyExecution
    today = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
    return db.query(StrategyExecution).filter(
        StrategyExecution.strategy_id == strategy_id,
        StrategyExecution.fired_at >= today,
    ).count()


def _open_execution_count(strategy_id: int, db) -> int:
    """Count currently open trades for this strategy."""
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


async def evaluate_and_fire(strategy, user, db, http_client: httpx.AsyncClient):
    """
    Evaluate one strategy for one user. Fires a trade if conditions are met.
    """
    from app.services.strategy_ta import evaluate_strategy_conditions
    from app.strategy_models import StrategyExecution, StrategyPerformance

    config   = strategy.config
    risk     = config.get("risk", {})
    filters  = config.get("filters", {})
    universe = config.get("universe", {})
    direction_pref = config.get("direction", "LONG")  # LONG | SHORT | BOTH

    # ── Global filters ──────────────────────────────────────────────────────
    if not _check_time_filter(filters):
        return
    if not _check_btc_regime(filters):
        logger.debug(f"[Strategy {strategy.id}] BTC regime filter blocked")
        return

    max_per_day    = int(risk.get("max_trades_per_day", 3))
    max_open       = int(risk.get("max_open_positions", 1))
    cooldown_mins  = int(risk.get("cooldown_minutes", 30))

    if _daily_execution_count(strategy.id, db) >= max_per_day:
        return
    if _open_execution_count(strategy.id, db) >= max_open:
        return

    # ── Symbol scan ─────────────────────────────────────────────────────────
    symbols = await _get_eligible_symbols(universe, http_client)
    if not symbols:
        return

    for symbol in symbols[:50]:  # cap at 50 to control latency
        # Per-symbol cooldown
        last_fired = _last_fired_time(strategy.id, symbol, db)
        if last_fired:
            elapsed_mins = (datetime.utcnow() - last_fired).total_seconds() / 60
            if elapsed_mins < cooldown_mins:
                continue

        price_data = await _fetch_price_and_ta(symbol, http_client)
        if not price_data:
            continue

        enhanced_ta = price_data.get("enhanced_ta", {})
        current_price = price_data.get("price", 0)
        if not current_price:
            continue

        # Evaluate entry conditions
        passed, details = await evaluate_strategy_conditions(
            config, symbol, price_data, enhanced_ta, http_client
        )

        if not passed:
            continue

        logger.info(
            f"🎯 [Strategy {strategy.id}] {strategy.name} — "
            f"{symbol} conditions met! Firing {direction_pref} trade."
        )
        for d in details:
            logger.info(f"   {d}")

        # ── Fire trade ──────────────────────────────────────────────────────
        ex_config = config.get("exit", {})
        tp_pct    = float(ex_config.get("take_profit_pct", 3.0))
        sl_pct    = float(ex_config.get("stop_loss_pct", 1.5))
        leverage  = int(risk.get("leverage", 10))

        # Determine direction for BOTH strategies
        direction = direction_pref
        if direction == "BOTH":
            # Use RSI to determine: >50 lean long, <50 lean short
            rsi = price_data.get("rsi", 50)
            direction = "LONG" if rsi > 50 else "SHORT"

        if direction == "LONG":
            tp_price = current_price * (1 + tp_pct / 100)
            sl_price = current_price * (1 - sl_pct / 100)
        else:
            tp_price = current_price * (1 - tp_pct / 100)
            sl_price = current_price * (1 + sl_pct / 100)

        # Record execution BEFORE placing order (captures intent)
        execution = StrategyExecution(
            strategy_id      = strategy.id,
            user_id          = user.id,
            symbol           = symbol,
            direction        = direction,
            entry_price      = current_price,
            tp_price         = tp_price,
            sl_price         = sl_price,
            leverage         = leverage,
            outcome          = "OPEN",
            conditions_met   = details,
            fired_at         = datetime.utcnow(),
        )
        db.add(execution)
        db.commit()
        db.refresh(execution)

        # Place order on Bitunix
        order_id = None
        try:
            from app.services.strategy_trader import place_bitunix_order_for_user
            order_id = await place_bitunix_order_for_user(
                user       = user,
                symbol     = symbol,
                direction  = direction,
                leverage   = leverage,
                entry_price = current_price,
                tp_price   = tp_price,
                sl_price   = sl_price,
                risk_pct   = float(risk.get("position_size_pct", 5)),
            )
        except Exception as e:
            logger.error(f"[Strategy {strategy.id}] Order placement error: {e}")
            execution.outcome = "CANCELLED"
            execution.notes   = str(e)
            db.commit()
            continue

        if order_id:
            execution.bitunix_order_id = str(order_id)
            db.commit()

            # Notify user via Telegram
            try:
                from app.services.bot import bot as tg_bot
                dir_icon = "🟢" if direction == "LONG" else "🔴"
                await tg_bot.send_message(
                    int(user.telegram_id),
                    f"{dir_icon} <b>Strategy Fired: {strategy.name}</b>\n"
                    f"<b>{symbol}</b> {direction}\n"
                    f"Entry  {current_price:.6g}\n"
                    f"TP  {tp_price:.6g}  (+{tp_pct}%)\n"
                    f"SL  {sl_price:.6g}  (-{sl_pct}%)\n"
                    f"Leverage  {leverage}x",
                    parse_mode="HTML",
                )
            except Exception as e:
                logger.warning(f"Telegram notify failed: {e}")

        # Only fire one trade per strategy per scan cycle
        break


async def run_strategy_executor():
    """
    Main background loop. Runs forever, evaluating all active strategies.
    One iteration = evaluate all active strategies for all users.
    """
    from app.database import SessionLocal
    from app.models import User
    from app.strategy_models import UserStrategy, init_strategy_tables
    from app.database import engine

    # Ensure tables exist
    init_strategy_tables(engine)

    logger.info("🤖 Strategy executor started")
    sem = asyncio.Semaphore(MAX_CONCURRENT)

    async with httpx.AsyncClient() as http_client:
        while True:
            try:
                db = SessionLocal()
                try:
                    strategies = (
                        db.query(UserStrategy)
                        .filter(UserStrategy.status == "active")
                        .all()
                    )

                    if not strategies:
                        await asyncio.sleep(SCAN_INTERVAL_SECONDS)
                        continue

                    logger.info(f"🤖 Strategy executor: {len(strategies)} active strategies")

                    async def _run_one(strategy):
                        async with sem:
                            user = db.query(User).filter(User.id == strategy.user_id).first()
                            if not user or user.banned:
                                return
                            # Check subscription is still valid
                            if not (
                                user.is_admin
                                or user.grandfathered
                                or (user.subscription_end and user.subscription_end > datetime.utcnow())
                            ):
                                return
                            try:
                                await evaluate_and_fire(strategy, user, db, http_client)
                            except Exception as e:
                                logger.error(f"[Strategy {strategy.id}] Executor error: {e}", exc_info=True)

                    await asyncio.gather(*[_run_one(s) for s in strategies])

                finally:
                    db.close()

            except Exception as e:
                logger.error(f"Strategy executor loop error: {e}", exc_info=True)

            await asyncio.sleep(SCAN_INTERVAL_SECONDS)
