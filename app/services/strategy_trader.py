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
    risk_usd: Optional[float] = None,
) -> Optional[str]:
    """
    Place a single market order on Bitunix for a user running a custom strategy.
    Accepts TP/SL as percentages and fetches a fresh price right before ordering
    so the calculated TP/SL levels are always valid (not stale from signal-fire time).

    Position sizing:
      risk_usd  — fixed dollar amount (overrides risk_pct when provided)
      risk_pct  — percentage of account balance (default 5 %)

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

            enc_key    = getattr(prefs, "bitunix_api_key", None)
            enc_secret = getattr(prefs, "bitunix_api_secret", None)
            if not enc_key or not enc_secret:
                logger.warning(f"[StrategyTrader] No Bitunix keys for user {user.id}")
                return None

            from app.utils.encryption import decrypt_api_key
            try:
                api_key    = decrypt_api_key(enc_key)
                api_secret = decrypt_api_key(enc_secret)
            except Exception as dec_err:
                logger.error(f"[StrategyTrader] Key decryption failed for user {user.id}: {dec_err}")
                return None

            if not api_key or len(api_key) < 10:
                logger.warning(f"[StrategyTrader] Decrypted key too short for user {user.id}")
                return None

            trader = BitunixTrader(api_key=api_key, api_secret=api_secret)

            # Fixed-size mode skips the balance fetch entirely
            if risk_usd and risk_usd > 0:
                position_size_usd = float(risk_usd)
                logger.info(f"[StrategyTrader] {symbol} fixed size ${position_size_usd}")
            else:
                balance = await trader.get_account_balance()
                if not balance or balance <= 0:
                    logger.warning(f"[StrategyTrader] Could not fetch balance for user {user.id}")
                    return None
                position_size_usd = max(balance * (risk_pct / 100), 5.0)
                logger.info(f"[StrategyTrader] {symbol} pct size {risk_pct}% of ${balance:.2f} = ${position_size_usd:.2f}")

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
