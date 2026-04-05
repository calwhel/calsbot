"""
Strategy Trader — Bitunix order placement wrapper for the strategy executor.
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
    Raises on hard errors so the caller can fall back to paper.
    """
    from app.services.bitunix_trader import BitunixTrader
    from app.utils.encryption import decrypt_api_key

    raw_key    = getattr(user, "bitunix_api_key", None)
    raw_secret = getattr(user, "bitunix_api_secret", None)
    if not raw_key or not raw_secret:
        logger.warning(f"[strategy_trader] User {user.id} has no Bitunix API keys — cannot place live order")
        return None

    try:
        api_key    = decrypt_api_key(raw_key)
        api_secret = decrypt_api_key(raw_secret)
    except Exception:
        api_key    = raw_key
        api_secret = raw_secret

    trader = BitunixTrader(api_key=api_key, api_secret=api_secret)

    sym = symbol.upper()
    if sym.endswith("USDT"):
        trade_sym = sym.replace("USDT", "/USDT")
    else:
        trade_sym = f"{sym}/USDT"

    ep = float(entry_price)
    if direction.upper() == "LONG":
        tp_price = ep * (1 + tp_pct / 100)
        sl_price = ep * (1 - sl_pct / 100)
    else:
        tp_price = ep * (1 - tp_pct / 100)
        sl_price = ep * (1 + sl_pct / 100)

    if risk_usd:
        position_size_usdt = float(risk_usd)
    else:
        balance = await trader.get_account_balance()
        if not balance or balance <= 0:
            logger.warning(f"[strategy_trader] Could not get balance for user {user.id}")
            return None
        position_size_usdt = balance * (risk_pct / 100)

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

    if result:
        return {
            "order_id":   result.get("orderId") or result.get("order_id"),
            "actual_fill": result.get("price") or ep,
        }
    return None
