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
    tp_price: float,
    sl_price: float,
    risk_pct: float = 5.0,
) -> Optional[str]:
    """
    Place a single market order on Bitunix for a user running a custom strategy.
    Returns the order ID string on success, None on failure.

    Uses the existing BitunixTrader class — no changes to bitunix_trader.py.
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

            # Require explicit opt-in (auto_trading_enabled covers strategy trades)
            if not prefs.auto_trading_enabled:
                logger.info(f"[StrategyTrader] Auto-trading disabled for user {user.id}")
                return None

            api_key    = getattr(prefs, "bitunix_api_key", None)
            api_secret = getattr(prefs, "bitunix_api_secret", None)
            if not api_key or not api_secret:
                logger.warning(f"[StrategyTrader] No Bitunix keys for user {user.id}")
                return None

            # Estimate position size from account balance
            trader = BitunixTrader(api_key=api_key, api_secret=api_secret)
            balance = await trader.get_account_balance()
            if not balance or balance <= 0:
                logger.warning(f"[StrategyTrader] Could not fetch balance for user {user.id}")
                return None

            position_size_usd = balance * (risk_pct / 100)
            position_size_usd = max(position_size_usd, 5.0)   # minimum $5

            result = await trader.place_trade(
                symbol             = symbol,
                direction          = direction,
                entry_price        = entry_price,
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
