"""
Strategy Trader — Bitunix order placement wrapper for the strategy executor.
API keys live on UserPreference, not the User model directly.
"""
import logging
from typing import Dict, Optional

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
) -> Optional[Dict]:
    """
    Place a Bitunix futures order for a strategy user.
    Returns {"order_id": ..., "actual_fill": ...} or None on failure.
    Raises on hard errors (e.g. PRICE_PAST_TP) so the caller can cancel cleanly.
    Returns None (no exception) when keys are missing or the order itself fails.
    """
    from app.models import UserPreference
    from app.database import SessionLocal
    from app.services.bitunix_trader import BitunixTrader
    from app.utils.encryption import decrypt_api_key

    # ── Load UserPreference (where keys actually live) ────────────────────────
    db = SessionLocal()
    try:
        prefs = db.query(UserPreference).filter_by(user_id=user.id).first()
    finally:
        db.close()

    if not prefs:
        logger.warning(f"[strategy_trader] No UserPreference for user {user.id} — cannot place live order")
        return None

    raw_key    = getattr(prefs, "bitunix_api_key",    None)
    raw_secret = getattr(prefs, "bitunix_api_secret", None)
    if not raw_key or not raw_secret:
        logger.warning(f"[strategy_trader] User {user.id} has no Bitunix API keys — cannot place live order")
        return None

    # ── Decrypt keys ──────────────────────────────────────────────────────────
    try:
        api_key    = decrypt_api_key(raw_key)
        api_secret = decrypt_api_key(raw_secret)
    except Exception:
        api_key    = raw_key
        api_secret = raw_secret

    # ── Build symbol and prices ───────────────────────────────────────────────
    sym = symbol.upper()
    trade_sym = sym.replace("USDT", "/USDT") if sym.endswith("USDT") else f"{sym}/USDT"

    ep = float(entry_price)
    if direction.upper() == "LONG":
        tp_price = ep * (1 + tp_pct / 100)
        sl_price = ep * (1 - sl_pct / 100)
    else:
        tp_price = ep * (1 - tp_pct / 100)
        sl_price = ep * (1 + sl_pct / 100)

    # ── Size the position ─────────────────────────────────────────────────────
    trader = BitunixTrader(api_key=api_key, api_secret=api_secret)
    try:
        if risk_usd:
            position_size_usdt = float(risk_usd)
        else:
            balance = await trader.get_account_balance()
            if not balance or balance <= 0:
                logger.warning(f"[strategy_trader] Could not get balance for user {user.id}")
                return None
            position_size_usdt = balance * (risk_pct / 100)

        # ── Place the order ───────────────────────────────────────────────────
        result = await trader.place_trade(
            symbol=trade_sym,
            direction=direction.upper(),
            entry_price=ep,
            stop_loss=sl_price,
            take_profit=tp_price,
            position_size_usdt=position_size_usdt,
            leverage=int(leverage),
            live_price_hint=ep,
        )
    finally:
        try:
            await trader.close()
        except Exception:
            pass

    if not result:
        logger.warning(f"[strategy_trader] place_trade returned None for {sym} — order not placed")
        return None

    # ── Propagate hard errors (e.g. PRICE_PAST_TP) — caller handles these ────
    if not result.get("success", False):
        err = result.get("error", "unknown error")
        if "PRICE_PAST_TP" in err:
            raise RuntimeError(err)
        logger.error(f"[strategy_trader] Bitunix order failed for {sym}: {err}")
        return None

    order_id = result.get("order_id")
    if not order_id:
        logger.warning(f"[strategy_trader] Bitunix returned success but no orderId for {sym}: {result}")
        return None

    return {
        "order_id":    order_id,
        "actual_fill": result.get("entry_price") or ep,
    }
