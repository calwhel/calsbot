"""
Strategy Trader — standalone order placement for user strategies.
Wraps the existing BitunixTrader without modifying bitunix_trader.py.
"""
import logging
from typing import Optional

logger = logging.getLogger(__name__)


async def place_bitunix_order_for_user(
    user,
    symbol: str,
    direction: str,
    leverage: int,
    entry_price: float,
    tp_pct: float,
    sl_pct: float,
    risk_pct: float = 5.0,
) -> Optional[str]:
    """
    Place a single market order on Bitunix for a user running a custom strategy.
    Accepts TP/SL as percentages and fetches a fresh price right before ordering
    so the calculated TP/SL levels are always valid (not stale from signal-fire time).
    Returns the order ID string on success, None on failure.
    """
    try:
        from app.database import SessionLocal
        from app.models import UserPreference
        from app.services.bitunix_trader import BitunixTrader

        db = SessionLocal()
        try:
            prefs = db.query(UserPreference).filter_by(user_id=user.id).first()
            if not prefs:
                logger.warning(f"[StrategyTrader] No prefs for user {user.id}")
                return None

            if not prefs.auto_trading_enabled:
                logger.info(f"[StrategyTrader] Auto-trading disabled for user {user.id}")
                return None

            api_key    = getattr(prefs, "bitunix_api_key", None)
            api_secret = getattr(prefs, "bitunix_api_secret", None)
            if not api_key or not api_secret:
                logger.warning(f"[StrategyTrader] No Bitunix keys for user {user.id}")
                return None

            trader = BitunixTrader(api_key=api_key, api_secret=api_secret)

            balance = await trader.get_account_balance()
            if not balance or balance <= 0:
                logger.warning(f"[StrategyTrader] Could not fetch balance for user {user.id}")
                return None

            # Fetch a fresh live price right before placing — avoids stale TP/SL
            # that can be rejected if price moved since the signal fired.
            live_price = await trader.get_current_price(symbol)
            if not live_price or live_price <= 0:
                logger.warning(f"[StrategyTrader] Could not fetch live price for {symbol}, falling back to entry_price")
                live_price = entry_price

            # Compute TP/SL from the live price with a small buffer to guarantee validity
            BUFFER = 0.001  # 0.1% buffer beyond the required pct
            if direction.upper() == "LONG":
                tp_price = live_price * (1 + (tp_pct + BUFFER) / 100)
                sl_price = live_price * (1 - sl_pct / 100)
            else:
                tp_price = live_price * (1 - (tp_pct + BUFFER) / 100)
                sl_price = live_price * (1 + sl_pct / 100)

            logger.info(
                f"[StrategyTrader] {symbol} {direction} | signal_price={entry_price:.6g} "
                f"live_price={live_price:.6g} | TP={tp_price:.6g} ({tp_pct}%) "
                f"SL={sl_price:.6g} ({sl_pct}%)"
            )

            position_size_usd = balance * (risk_pct / 100)
            position_size_usd = max(position_size_usd, 5.0)

            result = await trader.place_trade(
                symbol             = symbol,
                direction          = direction,
                entry_price        = live_price,
                stop_loss          = sl_price,
                take_profit        = tp_price,
                position_size_usdt = position_size_usd,
                leverage           = leverage,
            )

            if result and result.get("success"):
                order_id = str(result.get("orderId") or result.get("order_id") or "ok")
                logger.info(
                    f"[StrategyTrader] ✅ Order placed for user {user.id}: "
                    f"{symbol} {direction} {leverage}x — {order_id}"
                )
                return order_id
            else:
                logger.warning(f"[StrategyTrader] Order failed for user {user.id}: {result}")
                return None

        finally:
            db.close()

    except Exception as e:
        logger.error(f"[StrategyTrader] Exception for user {user.id}: {e}", exc_info=True)
        return None
