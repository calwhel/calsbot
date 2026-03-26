"""
Strategy Trader — standalone order placement for user strategies.
Wraps the existing BitunixTrader without modifying bitunix_trader.py.
"""
import hashlib
import logging
import time
from typing import Optional

logger = logging.getLogger(__name__)

# ── Balance cache ──────────────────────────────────────────────────────────────
# Avoids a fresh Bitunix API call on every signal when balance changes rarely.
_balance_cache: dict = {}   # {user_id: (balance_float, fetched_at_timestamp)}
_BALANCE_TTL = 60           # seconds before re-fetching

# ── Trader instance cache ──────────────────────────────────────────────────────
# BitunixTrader holds an httpx.AsyncClient with a connection pool.
# Re-using the same instance across trades means TCP connections to Bitunix
# are kept alive and reused, eliminating the TLS handshake overhead on every trade.
# Key: sha256(api_key + api_secret)[:16]   Value: BitunixTrader instance
_trader_cache: dict = {}


def _trader_cache_key(api_key: str, api_secret: str) -> str:
    return hashlib.sha256(f"{api_key}:{api_secret}".encode()).hexdigest()[:16]


def _get_or_create_trader(api_key: str, api_secret: str):
    from app.services.bitunix_trader import BitunixTrader
    key = _trader_cache_key(api_key, api_secret)
    trader = _trader_cache.get(key)
    if trader is None:
        trader = BitunixTrader(api_key=api_key, api_secret=api_secret)
        _trader_cache[key] = trader
        logger.debug(f"[StrategyTrader] Created new BitunixTrader (cache miss, key={key})")
    return trader


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
    Accepts TP/SL as percentages. The entry_price (already fetched by the executor
    scan cycle) is passed as a live_price_hint to place_trade() so a redundant
    Bitunix get_current_price() call is skipped.

    Position sizing:
      risk_usd  — fixed dollar amount (overrides risk_pct when provided)
      risk_pct  — percentage of account balance (default 5 %)

    Returns the order ID string on success, None on failure.
    """
    try:
        from app.database import SessionLocal
        from app.models import UserPreference

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

            # Reuse the cached trader (and its underlying HTTP connection pool)
            trader = _get_or_create_trader(api_key, api_secret)

            import asyncio as _aio_st

            # ── Fetch balance + live price concurrently (zero extra latency) ────────
            # Live price is fetched alongside balance to recalculate TP/SL from the
            # actual market price at order time — not the scan price from seconds ago.
            # This prevents "SL price must be less than last price" Bitunix rejections
            # that occur when price drifts between signal detection and order placement.
            async def _get_balance_and_price():
                """Returns (balance_or_None, live_price_or_None) in parallel."""
                cached = _balance_cache.get(user.id)
                if cached and (time.time() - cached[1]) < _BALANCE_TTL:
                    # Balance is cached — only need to fetch live price
                    bal, lp = await _aio_st.gather(
                        _aio_st.sleep(0),                    # noop placeholder
                        trader.get_current_price(symbol),
                        return_exceptions=True,
                    )
                    return cached[0], (lp if not isinstance(lp, Exception) else None)
                else:
                    bal, lp = await _aio_st.gather(
                        trader.get_account_balance(),
                        trader.get_current_price(symbol),
                        return_exceptions=True,
                    )
                    return (
                        (bal if not isinstance(bal, Exception) else None),
                        (lp  if not isinstance(lp, Exception)  else None),
                    )

            balance, live_price_now = await _get_balance_and_price()

            # Fixed-size mode skips the balance entirely
            if risk_usd and risk_usd > 0:
                position_size_usd = float(risk_usd)
                logger.info(f"[StrategyTrader] {symbol} fixed size ${position_size_usd}")
            else:
                # Validate balance
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
                _balance_cache[user.id] = (balance, time.time())
                position_size_usd = max(balance * (risk_pct / 100), 5.0)
                logger.info(f"[StrategyTrader] {symbol} pct size {risk_pct}% of ${balance:.2f} = ${position_size_usd:.2f}")

            # ── Compute TP/SL from the freshest available price ──────────────────────
            # If the live price has drifted from the signal scan price, recalculate
            # TP/SL from the actual market price so Bitunix accepts the order.
            BUFFER = 0.1  # 0.1% buffer to ensure Bitunix accepts the TP/SL
            ref_price = entry_price  # default: signal scan price
            if live_price_now and live_price_now > 0:
                drift_pct = abs(live_price_now - entry_price) / entry_price * 100
                if drift_pct >= 0.05:  # >0.05% drift — recalculate from live price
                    logger.info(
                        f"[StrategyTrader] {symbol} price drifted {drift_pct:.2f}% since scan "
                        f"(signal=${entry_price:.6g} → live=${live_price_now:.6g}) — "
                        f"recalculating TP/SL from live price"
                    )
                    ref_price = live_price_now

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
                live_price_hint    = live_price_now or ref_price,
            )

            if result and result.get("success"):
                order_id = str(result.get("orderId") or result.get("order_id") or "ok")
                logger.info(
                    f"[StrategyTrader] ✅ Order placed for user {user.id}: "
                    f"{symbol} {direction} {leverage}x — {order_id}"
                )

                # ── Post-fill: fetch actual fill price and correct TP/SL on Bitunix ──
                # Market orders fill at market price, which can differ from signal price.
                # For SHORTs on rapidly-moving coins the price may already be pulling back
                # by 0.5–1% by the time the order hits, making TP only 2% gain instead of
                # 3% from the actual fill — costing ~20% ROI at 20x leverage.
                # Fix: wait for fill, read avgOpenPrice, and push corrected TP/SL.
                actual_fill = None
                try:
                    import asyncio as _aio
                    await _aio.sleep(1.2)  # let the market order settle
                    open_positions = await trader.get_open_positions()
                    bitunix_sym = symbol.replace("/", "")
                    matching = [
                        p for p in open_positions
                        if str(p.get("symbol", "")) == bitunix_sym
                        and p.get("hold_side", "").upper() == direction.upper()
                    ]
                    if matching:
                        fill_price = float(matching[0].get("entry_price") or 0)
                        if fill_price > 0:
                            drift = abs(fill_price - ref_price) / ref_price
                            actual_fill = fill_price
                            # Always push corrected TP/SL anchored to actual fill price.
                            # This is especially important when the SL was temporarily
                            # clamped by the SL validity guard in place_trade(), since
                            # the clamped SL (e.g. 0.1% below live) is much tighter
                            # than the strategy's intended stop (e.g. 3.1% below fill).
                            if direction.upper() == "LONG":
                                new_tp = fill_price * (1 + (tp_pct + BUFFER) / 100)
                                new_sl = fill_price * (1 - sl_pct / 100)
                            else:
                                new_tp = fill_price * (1 - (tp_pct + BUFFER) / 100)
                                new_sl = fill_price * (1 + sl_pct / 100)
                            position_id = str(matching[0].get("position_id") or "")
                            if position_id:
                                ok = await trader.modify_position_sl(
                                    symbol=symbol,
                                    position_id=position_id,
                                    new_sl_price=new_sl,
                                    existing_tp_price=new_tp,
                                )
                                if ok:
                                    logger.info(
                                        f"[StrategyTrader] ✅ TP/SL set from fill for {symbol} {direction}"
                                        f" | ref={ref_price:.6g} fill={fill_price:.6g}"
                                        f" drift={drift*100:.3f}%"
                                        f" | TP={new_tp:.6g} SL={new_sl:.6g}"
                                    )
                                else:
                                    logger.warning(
                                        f"[StrategyTrader] TP/SL correction API failed for {symbol} "
                                        f"(fill={fill_price:.6g})"
                                    )
                            else:
                                logger.warning(
                                    f"[StrategyTrader] No position_id for {symbol} — "
                                    f"cannot push corrected TP/SL (fill={fill_price:.6g})"
                                )
                    else:
                        logger.warning(
                            f"[StrategyTrader] No open {direction} {symbol} position found after fill"
                            f" wait — skipping TP/SL correction"
                        )
                except Exception as _fe:
                    logger.warning(f"[StrategyTrader] Post-fill TP/SL correction error for {symbol}: {_fe}")

                return {"order_id": order_id, "actual_fill": actual_fill}
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
