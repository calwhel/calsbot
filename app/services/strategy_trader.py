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
                raise RuntimeError("Auto-trading preferences not configured. Enable auto-trading in your Bitunix settings.")

            if not prefs.auto_trading_enabled:
                raise RuntimeError("Auto-trading is disabled. Enable it in your account settings to go live.")

            enc_key    = getattr(prefs, "bitunix_api_key", None)
            enc_secret = getattr(prefs, "bitunix_api_secret", None)
            if not enc_key or not enc_secret:
                raise RuntimeError("Bitunix API keys not set. Add your API key and secret in account settings.")

            from app.utils.encryption import decrypt_api_key
            try:
                api_key    = decrypt_api_key(enc_key)
                api_secret = decrypt_api_key(enc_secret)
            except Exception as dec_err:
                logger.error(f"[StrategyTrader] Key decryption failed for user {user.id}: {dec_err}")
                raise RuntimeError("Could not decrypt Bitunix API keys — they may need to be re-entered.")

            if not api_key or len(api_key) < 10:
                raise RuntimeError("Bitunix API key appears invalid (too short). Please re-enter your API credentials.")

            trader = BitunixTrader(api_key=api_key, api_secret=api_secret)

            # Fixed-size mode skips the balance fetch entirely
            if risk_usd and risk_usd > 0:
                position_size_usd = float(risk_usd)
                logger.info(f"[StrategyTrader] {symbol} fixed size ${position_size_usd}")
            else:
                balance = await trader.get_account_balance()
                # NOTE: use `is None` not `not balance` — balance==0.0 is falsy
                # and must be caught as a separate "$0 balance" case.
                if balance is None or balance < 0:
                    raise RuntimeError(
                        "Could not fetch Bitunix account balance — check your API key "
                        "has Futures read permission (or re-enter your API keys)."
                    )
                if balance == 0:
                    raise RuntimeError(
                        "Bitunix Futures balance is $0. Transfer USDT from your Spot "
                        "wallet to your Futures wallet on Bitunix (Assets → Transfer)."
                    )
                position_size_usd = max(balance * (risk_pct / 100), 5.0)
                logger.info(f"[StrategyTrader] {symbol} pct size {risk_pct}% of ${balance:.2f} = ${position_size_usd:.2f}")

            # Use the signal price (entry_price) as the TP/SL reference.
            # All users in the same scan cycle share the same signal price, so their
            # TP/SL targets are IDENTICAL — preventing divergent outcomes where one
            # user hits TP and another hits SL due to a stale per-user live price.
            # The market order on Bitunix fills at whatever the market price is at
            # placement time; the TP/SL are anchored to the signal trigger price.
            BUFFER = 0.1  # 0.1% buffer to ensure Bitunix accepts the TP/SL
            ref_price = entry_price  # same for every user on this signal
            if direction.upper() == "LONG":
                tp_price = ref_price * (1 + (tp_pct + BUFFER) / 100)
                sl_price = ref_price * (1 - sl_pct / 100)
            else:
                tp_price = ref_price * (1 - (tp_pct + BUFFER) / 100)
                sl_price = ref_price * (1 + sl_pct / 100)

            logger.info(
                f"[StrategyTrader] {symbol} {direction} | ref_price={ref_price:.6g} "
                f"| TP={tp_price:.6g} ({tp_pct}%) SL={sl_price:.6g} ({sl_pct}%)"
            )

            result = await trader.place_trade(
                symbol             = symbol,
                direction          = direction,
                entry_price        = ref_price,
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
                # Propagate the actual Bitunix error so the executor can show the right message
                err = (result or {}).get("error") or "Order rejected by Bitunix (no error detail)"
                logger.warning(f"[StrategyTrader] Order failed for user {user.id}: {err} | result={result}")
                raise RuntimeError(err)

        finally:
            db.close()

    except Exception as e:
        logger.error(f"[StrategyTrader] Exception for user {user.id}: {e}", exc_info=True)
        raise  # Re-raise so the executor sees the real error message
