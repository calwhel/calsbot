"""
Strategy Trader — standalone order placement for user strategies.
Wraps the existing BitunixTrader without modifying bitunix_trader.py.
"""
import asyncio
import logging
from typing import Optional, Dict

logger = logging.getLogger(__name__)

BUFFER_PCT = 0.1   # 0.1% clearance added to TP so Bitunix accepts it
SLIP_PCT   = 0.10  # conservative 0.10% slippage estimate used in fallback only


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
) -> Optional[Dict]:
    """
    Place a single market order on Bitunix for a user running a custom strategy.

    Strategy:
      1. Place a bare MARKET order (no TP/SL embedded in the order).
      2. Wait briefly for the fill, then fetch the position's actual avgOpenPrice.
      3. Calculate TP/SL from the REAL fill price — not the pre-order signal price.
      4. Set TP/SL on the position via a separate API call.
      5. If position fetch fails, fall back to signal price + slippage estimate.

    This eliminates the SL > TP asymmetry that occurs when TP/SL are anchored to
    the pre-order last-trade price while the MARKET order fills at the ask (LONG)
    or bid (SHORT).

    Returns a dict:
      { "order_id": str, "fill_price": float, "tp_price": float, "sl_price": float }
    or raises RuntimeError on failure.
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

            # ── Position size ─────────────────────────────────────────────────
            if risk_usd and risk_usd > 0:
                position_size_usd = float(risk_usd)
                logger.info(f"[StrategyTrader] {symbol} fixed size ${position_size_usd}")
            else:
                balance = await trader.get_account_balance()
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

            logger.info(
                f"[StrategyTrader] {symbol} {direction} {leverage}x | signal_price={entry_price:.6g} "
                f"| tp={tp_pct}% sl={sl_pct}% | placing bare market order..."
            )

            # ── Step 1: Place bare MARKET order — no TP/SL ───────────────────
            # TP/SL set AFTER we know the actual fill price so they are
            # perfectly symmetric relative to where we actually entered.
            result = await trader.place_trade(
                symbol             = symbol,
                direction          = direction,
                entry_price        = entry_price,
                stop_loss          = None,
                take_profit        = None,
                position_size_usdt = position_size_usd,
                leverage           = leverage,
            )

            if not result or not result.get("success"):
                err = (result or {}).get("error") or "Order rejected by Bitunix (no error detail)"
                logger.warning(f"[StrategyTrader] Order failed for user {user.id}: {err} | result={result}")
                raise RuntimeError(err)

            order_id = str(result.get("orderId") or result.get("order_id") or "ok")
            logger.info(f"[StrategyTrader] ✅ Market order placed for user {user.id}: {symbol} {direction} {leverage}x — {order_id}")

            # ── Step 2: Wait for fill then fetch actual entry price ───────────
            await asyncio.sleep(2.0)

            bitunix_sym   = symbol.replace("/", "").upper()
            actual_fill   = None
            position_id   = None

            for attempt in range(3):
                try:
                    positions = await trader.get_open_positions()
                    pos = next(
                        (
                            p for p in positions
                            if str(p.get("symbol", "")).upper().replace("/", "") == bitunix_sym
                            and p.get("hold_side", "").upper() == direction.upper()
                            and p.get("entry_price", 0) > 0
                        ),
                        None,
                    )
                    if pos:
                        actual_fill = pos["entry_price"]
                        position_id = pos.get("position_id", "")
                        break
                except Exception as fe:
                    logger.warning(f"[StrategyTrader] Position fetch attempt {attempt + 1}/3 failed: {fe}")
                await asyncio.sleep(1.0)

            # ── Step 3: Calculate TP/SL from actual fill price ───────────────
            if actual_fill and actual_fill > 0:
                fill_price = actual_fill
                logger.info(
                    f"[StrategyTrader] {symbol} fill confirmed: ${fill_price:.6g} "
                    f"(signal was ${entry_price:.6g}, diff={100 * (fill_price / entry_price - 1):+.3f}%)"
                )
            else:
                # Fallback: signal price + conservative slippage estimate
                slip = SLIP_PCT / 100
                fill_price = entry_price * (1 + slip) if direction.upper() == "LONG" else entry_price * (1 - slip)
                logger.warning(
                    f"[StrategyTrader] {symbol} position not found after fill — "
                    f"using estimated fill ${fill_price:.6g} (signal ${entry_price:.6g} ± {SLIP_PCT}%)"
                )

            if direction.upper() == "LONG":
                tp_price = fill_price * (1 + (tp_pct + BUFFER_PCT) / 100)
                sl_price = fill_price * (1 - sl_pct / 100)
            else:
                tp_price = fill_price * (1 - (tp_pct + BUFFER_PCT) / 100)
                sl_price = fill_price * (1 + sl_pct / 100)

            logger.info(
                f"[StrategyTrader] {symbol} TP/SL from fill: "
                f"TP={tp_price:.6g} (+{tp_pct}%) SL={sl_price:.6g} (-{sl_pct}%)"
            )

            # ── Step 4: Set TP/SL on the position ────────────────────────────
            try:
                await trader.place_position_tpsl(
                    symbol      = symbol,
                    position_id = position_id or "",
                    sl_price    = sl_price,
                    tp_price    = tp_price,
                )
                logger.info(f"[StrategyTrader] ✅ TP/SL set on position for {symbol}")
            except Exception as tpsl_err:
                logger.warning(f"[StrategyTrader] TP/SL placement failed for {symbol}: {tpsl_err}")

            return {
                "order_id":   order_id,
                "fill_price": fill_price,
                "tp_price":   tp_price,
                "sl_price":   sl_price,
            }

        finally:
            db.close()

    except Exception as e:
        logger.error(f"[StrategyTrader] Exception for user {user.id}: {e}", exc_info=True)
        raise
