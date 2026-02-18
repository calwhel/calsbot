"""
Position Monitor - Monitors open Bitunix positions for TP/SL and smart exits
"""
import logging
from datetime import datetime
from sqlalchemy.orm import Session
from app.database import SessionLocal
from app.models import Trade, User, UserPreference
from app.services.bitunix_trader import BitunixTrader
from app.utils.encryption import decrypt_api_key
from app.services.analytics import AnalyticsService
from app.services.trade_screenshot import screenshot_generator

logger = logging.getLogger(__name__)

BINANCE_FUTURES_URL = "https://fapi.binance.com"

async def _get_binance_price(symbol: str):
    """Fetch live price from Binance Futures as independent verification source."""
    import httpx
    try:
        binance_symbol = symbol.replace('/', '').replace(':USDT', '').replace('-USDT', '').upper()
        async with httpx.AsyncClient(timeout=5) as client:
            response = await client.get(
                f"{BINANCE_FUTURES_URL}/fapi/v1/ticker/price",
                params={'symbol': binance_symbol}
            )
            if response.status_code == 200:
                data = response.json()
                price = float(data.get('price', 0))
                if price > 0:
                    return price
    except Exception as e:
        logger.debug(f"Could not fetch Binance price for {symbol}: {e}")
    return None


async def monitor_positions(bot):
    """Monitor open Bitunix positions and handle TP/SL hits + smart exits"""
    db = SessionLocal()
    
    try:
        from datetime import timedelta
        
        # Get ALL open trades (manual + auto) with Bitunix API keys configured
        # Skip trades opened in last 5 minutes to prevent false "position closed" notifications
        # Bitunix API can be slow to show new positions, so generous grace period needed
        grace_period = datetime.utcnow() - timedelta(minutes=5)
        
        open_trades = db.query(Trade).join(User).join(UserPreference).filter(
            Trade.status == 'open',
            Trade.opened_at < grace_period,  # Only check trades older than 5 minutes
            UserPreference.bitunix_api_key != None  # Just need API keys, not auto-trading enabled
        ).all()
        
        if not open_trades:
            logger.debug("No open Bitunix trades to monitor (or all trades < 5 min old)")
            return
        
        logger.info(f"📊 MONITOR: Checking {len(open_trades)} open positions...")
        
        for trade in open_trades:
            trader = None
            try:
                user = trade.user
                prefs = user.preferences
                
                logger.info(f"👁️ Checking trade #{trade.id}: {trade.symbol} {trade.direction} | TP1={trade.take_profit_1} TP2={trade.take_profit_2} | tp1_hit={trade.tp1_hit}")
                
                # Decrypt Bitunix credentials
                api_key = decrypt_api_key(prefs.bitunix_api_key)
                api_secret = decrypt_api_key(prefs.bitunix_api_secret)
                trader = BitunixTrader(api_key, api_secret)
                
                # Check if position is still open on Bitunix
                bitunix_positions = await trader.get_open_positions()
                bitunix_symbol = trade.symbol.replace('/', '')
                
                # SAFETY: If API returns empty list, it might be an API failure
                api_returned_empty = len(bitunix_positions) == 0
                
                # Log all positions returned for debugging
                if bitunix_positions:
                    bitunix_syms = [f"{p['symbol']}({p['hold_side']})" for p in bitunix_positions]
                    logger.info(f"   📡 Bitunix positions returned: {bitunix_syms} | Looking for: {bitunix_symbol} {trade.direction}")
                else:
                    logger.warning(f"   📡 Bitunix returned 0 positions | Looking for: {bitunix_symbol} {trade.direction}")
                
                position_exists = False
                # Normalize our symbol for matching: strip /, :USDT, -USDT, uppercase
                normalized_trade_sym = bitunix_symbol.replace('/', '').replace(':USDT', '').replace('-USDT', '').upper()
                
                for pos in bitunix_positions:
                    pos_symbol = pos.get('symbol', '')
                    normalized_pos_sym = pos_symbol.replace('/', '').replace(':USDT', '').replace('-USDT', '').upper()
                    
                    if normalized_pos_sym == normalized_trade_sym:
                        expected_side = 'long' if trade.direction == 'LONG' else 'short'
                        hold_side = pos.get('hold_side', '').lower()
                        if hold_side == expected_side:
                            position_exists = True
                            logger.info(f"   ✅ Position OPEN on Bitunix: {pos.get('total', 'N/A')} contracts")
                            break
                        else:
                            logger.info(f"   ⚠️ Found {pos_symbol} but side mismatch: exchange={hold_side}, expected={expected_side}")
                
                if position_exists:
                    trade.not_found_count = 0
                    db.commit()
                
                # If position not found, verify with external price source before closing
                if not position_exists:
                    trade.not_found_count = (trade.not_found_count or 0) + 1
                    required_checks = 3
                    
                    if api_returned_empty:
                        required_checks = 5
                        logger.warning(f"   ⚠️ API returned 0 positions - possible API failure. Not-found count: {trade.not_found_count}/{required_checks}")
                    else:
                        logger.info(f"   ⚠️ Position not found on Bitunix. Not-found count: {trade.not_found_count}/{required_checks}")
                    
                    # CROSS-CHECK: Fetch live price from Binance Futures to verify trade status
                    # Only do this check if we have BOTH SL and at least one TP to verify against
                    binance_price = await _get_binance_price(bitunix_symbol)
                    active_sl = trade.stop_loss
                    # Use breakeven SL if TP1 was already hit
                    if trade.tp1_hit and trade.breakeven_moved:
                        active_sl = trade.entry_price
                    active_tp = trade.take_profit_2 if trade.tp1_hit else (trade.take_profit_1 or trade.take_profit)
                    
                    if binance_price and active_sl and active_tp:
                        trade_still_open = False
                        
                        if trade.direction == 'LONG':
                            trade_still_open = binance_price > active_sl and binance_price < active_tp
                        else:
                            trade_still_open = binance_price < active_sl and binance_price > active_tp
                        
                        if trade_still_open:
                            logger.warning(
                                f"   🔍 BINANCE PRICE CHECK: {bitunix_symbol} = ${binance_price:.6f} | "
                                f"Entry=${trade.entry_price:.6f} | SL=${active_sl:.6f} | TP=${active_tp:.6f} | "
                                f"Price is BETWEEN SL and TP - trade likely STILL OPEN (Bitunix API issue)"
                            )
                            trade.not_found_count = max(0, trade.not_found_count - 1)
                            db.commit()
                            continue
                        else:
                            logger.info(
                                f"   🔍 BINANCE PRICE CHECK: {bitunix_symbol} = ${binance_price:.6f} | "
                                f"SL=${active_sl:.6f} | TP=${active_tp:.6f} | Price has passed SL or TP - trade may be legitimately closed"
                            )
                    elif binance_price:
                        logger.info(f"   🔍 BINANCE PRICE: {bitunix_symbol} = ${binance_price:.6f} (missing SL or TP, cannot verify - relying on multi-check)")
                    
                    if trade.not_found_count < required_checks:
                        db.commit()
                        continue
                    
                    logger.info(f"   ❌ Position CONFIRMED CLOSED after {trade.not_found_count} checks - syncing database...")
                    trade_age_minutes = (datetime.utcnow() - trade.opened_at).total_seconds() / 60
                    logger.info(f"🔄 SYNC: Position {trade.id} ({trade.symbol}) closed on Bitunix but open in DB - Trade age: {trade_age_minutes:.1f} minutes")
                    
                    current_price = await trader.get_current_price(trade.symbol)
                    if not current_price:
                        logger.warning(f"Could not fetch price for {trade.symbol} during sync")
                        continue
                    
                    actual_exit_price = current_price
                    pnl_usd = None
                    pnl_percent = None
                    used_exchange_data = False
                    
                    try:
                        closed_history = await trader.get_closed_position_history(bitunix_symbol)
                        if closed_history and closed_history.get('close_price', 0) > 0:
                            hist_entry = closed_history['entry_price']
                            if abs(hist_entry - trade.entry_price) / trade.entry_price < 0.02:
                                actual_exit_price = closed_history['close_price']
                                pnl_usd = closed_history['realized_pnl']
                                pnl_percent = (pnl_usd / trade.position_size) * 100 if trade.position_size > 0 else 0
                                used_exchange_data = True
                                logger.info(f"📜 EXCHANGE DATA: {trade.symbol} actual close=${actual_exit_price:.6f}, realized PnL=${pnl_usd:.4f} ({pnl_percent:.2f}%)")
                            else:
                                logger.warning(f"📜 History entry price ${hist_entry:.6f} doesn't match trade entry ${trade.entry_price:.6f} - different position, skipping")
                    except Exception as e:
                        logger.warning(f"Could not fetch closed position history: {e}")
                    
                    if not used_exchange_data:
                        if trade.exchange_unrealized_pnl is not None:
                            pnl_usd = trade.exchange_unrealized_pnl
                            pnl_percent = (pnl_usd / trade.position_size) * 100 if trade.position_size > 0 else 0
                            logger.info(f"📊 Using last exchange-reported PnL: ${pnl_usd:.2f} ({pnl_percent:.1f}%)")
                        else:
                            trade_leverage = trade.leverage if trade.leverage else (prefs.top_gainers_leverage if trade.trade_type == 'TOP_GAINER' else (prefs.user_leverage or 5))
                            price_change = actual_exit_price - trade.entry_price if trade.direction == 'LONG' else trade.entry_price - actual_exit_price
                            price_change_percent = price_change / trade.entry_price
                            pnl_usd = price_change_percent * trade.remaining_size * trade_leverage
                            pnl_percent = price_change_percent * trade_leverage * 100
                            logger.warning(f"⚠️ No exchange PnL data, using manual calc with {trade_leverage}x leverage: ${pnl_usd:.2f} ({pnl_percent:.1f}%)")
                    
                    if trade.exchange_unrealized_pnl is not None:
                        trade.exchange_realized_pnl = trade.exchange_unrealized_pnl
                    if used_exchange_data:
                        trade.exchange_realized_pnl = pnl_usd
                    
                    be_threshold = max(0.01, trade.position_size * 0.001) if trade.position_size else 0.01
                    if abs(pnl_percent) < 0.1 and abs(pnl_usd) < be_threshold:
                        pnl_usd = 0.0
                        pnl_percent = 0.0
                        logger.info(f"📊 BREAKEVEN: {trade.symbol} P&L within tolerance (<0.1%), setting to 0%")
                    
                    tp_price = trade.take_profit_1 if trade.take_profit_1 else trade.take_profit
                    tp_hit = False
                    sl_hit = False
                    
                    if trade.direction == 'LONG':
                        if tp_price and actual_exit_price >= tp_price * 0.998:
                            tp_hit = True
                            trade.tp1_hit = True
                            logger.info(f"✅ TP HIT: {trade.symbol} LONG - Exit ${actual_exit_price:.4f} >= TP ${tp_price:.4f}")
                        elif trade.stop_loss and actual_exit_price <= trade.stop_loss * 1.002:
                            sl_hit = True
                            logger.info(f"⛔ SL HIT: {trade.symbol} LONG - Exit ${actual_exit_price:.4f} <= SL ${trade.stop_loss:.4f}")
                    else:
                        if tp_price and actual_exit_price <= tp_price * 1.002:
                            tp_hit = True
                            trade.tp1_hit = True
                            logger.info(f"✅ TP HIT: {trade.symbol} SHORT - Exit ${actual_exit_price:.4f} <= TP ${tp_price:.4f}")
                        elif trade.stop_loss and actual_exit_price >= trade.stop_loss * 0.998:
                            sl_hit = True
                            logger.info(f"⛔ SL HIT: {trade.symbol} SHORT - Exit ${actual_exit_price:.4f} >= SL ${trade.stop_loss:.4f}")
                    
                    if not tp_hit and not sl_hit and used_exchange_data and pnl_usd > 0:
                        tp_hit = True
                        trade.tp1_hit = True
                        logger.info(f"✅ TP HIT (by PnL): {trade.symbol} - Exchange reported positive PnL ${pnl_usd:.2f}")
                    
                    if not tp_hit and not sl_hit and trade.tp1_hit:
                        tp_hit = True
                        logger.info(f"✅ TP2 BREAKEVEN: {trade.symbol} - TP1 already hit, BE close = TP")
                    
                    if not tp_hit and not sl_hit:
                        logger.info(f"📤 MANUAL CLOSE: {trade.symbol} P&L ${pnl_usd:.2f} ({pnl_percent:.1f}%) - price didn't hit TP/SL")
                    
                    trade.status = 'tp_hit' if tp_hit else ('sl_hit' if sl_hit else 'closed')
                    trade.exit_price = actual_exit_price
                    trade.closed_at = datetime.utcnow()
                    trade.pnl = float(pnl_usd)
                    trade.pnl_percent = pnl_percent
                    trade.remaining_size = 0
                    
                    from app.services.risk_controls import record_trade_result
                    record_trade_result(won=(trade.pnl > 0))
                    
                    if trade.pnl > 0:
                        prefs.consecutive_losses = 0
                        
                        # AUTO-COMPOUND: Update Top Gainer win streak
                        if trade.trade_type == 'TOP_GAINER' and prefs.top_gainers_auto_compound:
                            prefs.top_gainers_win_streak += 1
                            if prefs.top_gainers_win_streak >= 3:
                                prefs.top_gainers_position_multiplier = 1.2
                                logger.info(f"🔥 TOP GAINER AUTO-COMPOUND (Sync): User {user.id} hit 3-win streak! Position size +20%")
                    else:
                        prefs.consecutive_losses += 1
                        
                        # AUTO-COMPOUND: Reset Top Gainer streak on loss
                        if trade.trade_type == 'TOP_GAINER' and prefs.top_gainers_auto_compound:
                            if prefs.top_gainers_win_streak > 0 or prefs.top_gainers_position_multiplier > 1.0:
                                logger.info(f"🔄 TOP GAINER LOSS (Sync): User {user.id} - Resetting position multiplier")
                            prefs.top_gainers_win_streak = 0
                            prefs.top_gainers_position_multiplier = 1.0
                    
                    db.commit()
                    
                    # Update signal analytics
                    if trade.signal_id:
                        AnalyticsService.update_signal_outcome(db, trade.signal_id)
                    
                    # Log social/news trade closure
                    if trade.trade_type in ['SOCIAL_SIGNAL', 'SOCIAL_SHORT', 'NEWS_SIGNAL']:
                        try:
                            from app.services.social_trade_logger import log_social_trade_closed
                            await log_social_trade_closed(db, trade)
                        except Exception as log_err:
                            logger.warning(f"Failed to log social trade: {log_err}")
                    
                    # ONLY send notification if TP or SL hit (not for generic closures)
                    if tp_hit or sl_hit:
                        exit_type = "🎯 TAKE PROFIT HIT!" if tp_hit else "⛔ STOP LOSS HIT!"
                        logger.info(f"🔔 Sending {exit_type} notification for trade {trade.id} to user {user.telegram_id}")
                        
                        try:
                            from aiogram.types import InlineKeyboardMarkup, InlineKeyboardButton
                            share_keyboard = None
                            if trade.pnl > 0:
                                share_keyboard = InlineKeyboardMarkup(inline_keyboard=[
                                    [InlineKeyboardButton(text="📸 Share This Win", callback_data=f"share_trade_{trade.id}")]
                                ])
                            
                            await bot.send_message(
                                user.telegram_id,
                                f"{exit_type}\n\n"
                                f"Symbol: {trade.symbol} {trade.direction}\n"
                                f"Entry: ${trade.entry_price:.4f}\n"
                                f"Exit: ${actual_exit_price:.4f}\n\n"
                                f"💰 PnL: ${trade.pnl:.2f} ({trade.pnl_percent:+.1f}%)\n"
                                f"Position Size: ${trade.position_size:.2f}\n\n"
                                f"{'🔥 Great trade!' if trade.pnl > 0 else '📊 On to the next one'}",
                                reply_markup=share_keyboard
                            )
                            logger.info(f"✅ TP/SL notification sent successfully for trade {trade.id}")
                            
                            # Generate and send trade screenshot
                            await send_trade_screenshot(bot, trade, user, db)
                        except Exception as notif_error:
                            logger.error(f"❌ Failed to send TP/SL notification for trade {trade.id}: {notif_error}", exc_info=True)
                    else:
                        # Just log, no notification for generic closures
                        logger.info(f"Position closed (no TP/SL): Trade {trade.id} - PnL ${trade.pnl:.2f}, Entry: ${trade.entry_price:.4f}, Exit: ${actual_exit_price:.4f}, TP: ${tp_price}, SL: ${trade.stop_loss}")
                    
                    logger.info(f"✅ Synced closed position: Trade {trade.id} - PnL ${trade.pnl:.2f}")
                    continue  # Skip rest of checks for this trade
                
                # Position is still open on Bitunix - continue with normal monitoring
                
                # NOTE: Old position-count based TP1 detection REMOVED
                # Bitunix aggregates positions, so position count doesn't work
                # Price-based detection is used below instead (lines 380+)
                
                # 🔥 CRITICAL: Fetch live position data from Bitunix API
                logger.info(f"🔄 Fetching position for {trade.symbol} (Trade ID: {trade.id}) | original_contracts={trade.original_contracts} | tp1={trade.take_profit_1} | tp2={trade.take_profit_2} | tp1_hit={trade.tp1_hit}")
                position_data = await trader.get_position_detail(trade.symbol)
                logger.info(f"📦 Position data for {trade.symbol}: {position_data}")
                
                if position_data:
                    # Update trade with exchange-reported PnL (THE FIX!)
                    trade.exchange_unrealized_pnl = position_data['unrealized_pl']
                    trade.last_sync_at = datetime.utcnow()
                    
                    # Calculate PnL percentage from exchange-reported unrealized_pl
                    exchange_pnl_percent = (position_data['unrealized_pl'] / trade.position_size) * 100 if trade.position_size > 0 else 0
                    
                    # Update trade.pnl with live exchange data
                    trade.pnl = position_data['unrealized_pl']
                    trade.pnl_percent = exchange_pnl_percent
                    
                    db.commit()
                    logger.info(f"📊 LIVE SYNC: {trade.symbol} - Exchange PnL: ${position_data['unrealized_pl']:.2f} ({exchange_pnl_percent:+.1f}%)")
                    
                    # Use exchange mark price instead of ticker price (more accurate)
                    current_price = position_data['mark_price']
                    
                    BREAKEVEN_ROI_THRESHOLD = 40.0
                    
                    should_breakeven = False
                    be_reason = ""
                    
                    if trade.take_profit_1 and trade.entry_price:
                        halfway_to_tp1 = (trade.entry_price + trade.take_profit_1) / 2
                        if trade.direction == 'LONG' and current_price >= halfway_to_tp1:
                            should_breakeven = True
                            be_reason = f"LONG halfway to TP1 (price ${current_price:.6f} >= halfway ${halfway_to_tp1:.6f})"
                        elif trade.direction == 'SHORT' and current_price <= halfway_to_tp1:
                            should_breakeven = True
                            be_reason = f"SHORT halfway to TP1 (price ${current_price:.6f} <= halfway ${halfway_to_tp1:.6f})"
                    
                    if not should_breakeven and exchange_pnl_percent >= BREAKEVEN_ROI_THRESHOLD:
                        should_breakeven = True
                        be_reason = f"ROI {exchange_pnl_percent:.1f}% >= {BREAKEVEN_ROI_THRESHOLD}%"
                    
                    if should_breakeven and not trade.breakeven_moved:
                        logger.info(f"🛡️ AUTO-BREAKEVEN TRIGGER: {trade.symbol} - {be_reason} - Moving SL to entry ${trade.entry_price:.6f}")
                        
                        be_sl_modified = await trader.modify_tpsl_order_sl(
                            symbol=trade.symbol,
                            new_sl_price=trade.entry_price
                        )
                        
                        if not be_sl_modified:
                            logger.info(f"⚠️ Order-level SL failed for breakeven, trying position-level...")
                            position_id = await trader.get_position_id(trade.symbol)
                            if position_id:
                                be_sl_modified = await trader.modify_position_sl(
                                    symbol=trade.symbol,
                                    position_id=position_id,
                                    new_sl_price=trade.entry_price
                                )
                        
                        if not be_sl_modified:
                            logger.info(f"⚠️ Position-level SL failed for breakeven, trying holdSide method...")
                            be_sl_modified = await trader.update_position_stop_loss(
                                symbol=trade.symbol,
                                new_stop_loss=trade.entry_price,
                                direction=trade.direction
                            )
                        
                        if be_sl_modified:
                            old_sl = trade.stop_loss
                            trade.stop_loss = trade.entry_price
                            trade.breakeven_moved = True
                            db.commit()
                            
                            logger.info(f"✅ AUTO-BREAKEVEN ACTIVATED: Trade {trade.id} ({trade.symbol}) - SL moved from ${old_sl:.6f} to ${trade.entry_price:.6f} - Reason: {be_reason}")
                            
                            trigger_line = f"📍 Trigger: Halfway to TP1" if "halfway" in be_reason else f"📍 Trigger: ROI {exchange_pnl_percent:.1f}%"
                            
                            await bot.send_message(
                                user.telegram_id,
                                f"🛡️ <b>AUTO-BREAKEVEN ACTIVATED!</b>\n\n"
                                f"<b>{trade.symbol}</b> {trade.direction}\n"
                                f"Entry: ${trade.entry_price:.6f}\n"
                                f"Current ROI: <b>+{exchange_pnl_percent:.1f}%</b>\n"
                                f"{trigger_line}\n\n"
                                f"🔒 Stop Loss moved to ENTRY (breakeven)\n"
                                f"✅ This trade is now RISK-FREE!\n\n"
                                f"💰 Unrealized: ${position_data['unrealized_pl']:+.2f}",
                                parse_mode='HTML'
                            )
                        else:
                            logger.warning(f"⚠️ AUTO-BREAKEVEN FAILED: Could not modify SL on Bitunix for {trade.symbol} - will retry next cycle")
                    
                    # 🔥 DUAL TP FIX: Detect TP1 hit via position size reduction
                    # For dual TP trades, Bitunix has 2 orders (50% each). When TP1 hits, one order closes.
                    # Detect this by checking if position size dropped to ~50% of original
                    logger.info(f"🔍 DUAL TP CHECK: {trade.symbol} | TP1={trade.take_profit_1} | TP2={trade.take_profit_2} | tp1_hit={trade.tp1_hit}")
                    
                    if trade.take_profit_1 and trade.take_profit_2 and not trade.tp1_hit:
                        # Get current position qty from Bitunix
                        current_qty = position_data['total']
                        
                        # 🔥 CRITICAL FIX: Store original_contracts on FIRST poll if not set
                        # This ensures we capture actual filled quantity before any TP hits
                        if not trade.original_contracts or trade.original_contracts <= 0:
                            trade.original_contracts = current_qty
                            db.commit()
                            logger.info(f"📦 CAPTURED original_contracts on first poll: {current_qty} for {trade.symbol}")
                            # Skip TP1 detection on first poll (need baseline first)
                            continue
                        
                        expected_original_qty = trade.original_contracts
                        logger.info(f"✅ Using stored original_contracts: {expected_original_qty}")
                        
                        qty_ratio = current_qty / expected_original_qty if expected_original_qty > 0 else 1.0
                        
                        # ALWAYS log this for debugging (not just debug level)
                        logger.info(f"📊 TP1 SIZE CHECK: {trade.symbol} | Expected: {expected_original_qty:.4f} | Current: {current_qty:.4f} | Ratio: {qty_ratio:.2%}")
                        
                        # If position is 40-60% of original, TP1 order closed
                        if 0.35 < qty_ratio < 0.65:
                            logger.info(f"🎯 TP1 HIT DETECTED via position size: {trade.symbol} - Expected: {expected_original_qty:.4f}, Current: {current_qty:.4f} ({qty_ratio*100:.0f}%)")
                            
                            # 🔥 BREAKEVEN: Try ORDER-level SL modification FIRST
                            # We use order-attached TP/SL (tpPrice/slPrice), not Position TP/SL
                            logger.info(f"🔧 BREAKEVEN: Attempting to modify SL to entry ${trade.entry_price:.6f}")
                            
                            # METHOD 1 (PRIMARY): Order-attached TP/SL modification
                            sl_modified = await trader.modify_tpsl_order_sl(
                                symbol=trade.symbol,
                                new_sl_price=trade.entry_price
                            )
                            
                            # METHOD 2 (FALLBACK): Position-level SL modification
                            if not sl_modified:
                                logger.info(f"⚠️ Order-level SL failed, trying position-level...")
                                position_id = await trader.get_position_id(trade.symbol)
                                logger.info(f"🔑 Got positionId for {trade.symbol}: {position_id}")
                                
                                if position_id:
                                    sl_modified = await trader.modify_position_sl(
                                        symbol=trade.symbol,
                                        position_id=position_id,
                                        new_sl_price=trade.entry_price
                                    )
                            
                            # METHOD 3 (LAST RESORT): Position holdSide method
                            if not sl_modified:
                                logger.info(f"⚠️ Position-level SL failed, trying holdSide method...")
                                sl_modified = await trader.update_position_stop_loss(
                                    symbol=trade.symbol,
                                    new_stop_loss=trade.entry_price,
                                    direction=trade.direction
                                )
                            
                            if sl_modified:
                                old_sl = trade.stop_loss
                                trade.stop_loss = trade.entry_price
                                trade.tp1_hit = True
                                trade.remaining_size = trade.position_size / 2
                                db.commit()
                                
                                logger.info(f"✅ BREAKEVEN ACTIVATED (via size detection): Trade {trade.id} ({trade.symbol}) - SL moved from ${old_sl:.6f} to ${trade.entry_price:.6f}")
                                
                                # Notify user
                                tp1_profit_usd = (trade.position_size / 2) * 0.5  # ~50% profit on 50% position
                                await bot.send_message(
                                    user.telegram_id,
                                    f"✅ <b>TP1 HIT - BREAKEVEN ACTIVATED!</b>\n\n"
                                    f"<b>{trade.symbol}</b> {trade.direction}\n"
                                    f"Entry: ${trade.entry_price:.6f}\n"
                                    f"TP1: ${trade.take_profit_1:.6f}\n\n"
                                    f"💰 50% closed at TP1 (+50% profit = ~${tp1_profit_usd:.2f})\n"
                                    f"🔒 Stop Loss moved to ENTRY (breakeven)\n"
                                    f"🎯 Remaining 50% now RISK-FREE!\n\n"
                                    f"Targeting TP2 @ ${trade.take_profit_2:.6f} (+100%) 🚀",
                                    parse_mode='HTML'
                                )
                            else:
                                logger.warning(f"⚠️ Failed to update SL on Bitunix for {trade.symbol} - will retry")
                else:
                    # Fallback: Use ticker price if position detail unavailable
                    current_price = await trader.get_current_price(trade.symbol)
                    if not current_price:
                        logger.warning(f"Could not fetch price for {trade.symbol}")
                        continue
                    logger.warning(f"⚠️ No position detail from API, using ticker price for {trade.symbol}")
                
                # Initialize remaining_size if not set
                if trade.remaining_size == 0:
                    trade.remaining_size = trade.position_size
                    db.commit()
                
                # Calculate position amounts
                position_amount = trade.position_size / trade.entry_price
                remaining_amount = trade.remaining_size / trade.entry_price
                
                # ====================
                # 🧠 SMART EXIT: Check for market reversal
                # SKIP for dual TP trades - let them run to TP1/TP2 with breakeven
                # ====================
                should_exit = False
                exit_reason = None
                
                # Only check Smart Exit for trades WITHOUT dual TP system
                # Dual TP trades should follow the TP1 breakeven -> TP2 flow
                if not (trade.take_profit_1 and trade.take_profit_2):
                    from app.services.smart_exit import check_smart_exit
                    should_exit, exit_reason = await check_smart_exit(
                        trade.symbol,
                        trade.direction,
                        trade.entry_price,
                        current_price,
                        'bitunix'
                    )
                
                if should_exit:
                    logger.info(f"Smart exit triggered for trade {trade.id}: {exit_reason}")
                    
                    # Close position on Bitunix
                    close_result = await trader.close_position(trade.symbol, trade.direction)
                    
                    if close_result:
                        # Prioritize exchange-reported PnL when available
                        if trade.exchange_unrealized_pnl is not None:
                            pnl_usd = trade.exchange_unrealized_pnl
                            final_pnl_percent = (pnl_usd / trade.position_size) * 100 if trade.position_size > 0 else 0
                            logger.info(f"📊 SMART EXIT: Using exchange-reported PnL: ${pnl_usd:.2f} ({final_pnl_percent:.1f}%)")
                        else:
                            trade_leverage = trade.leverage if trade.leverage else (prefs.top_gainers_leverage if trade.trade_type == 'TOP_GAINER' else (prefs.user_leverage or 5))
                            price_change = current_price - trade.entry_price if trade.direction == 'LONG' else trade.entry_price - current_price
                            price_change_percent = price_change / trade.entry_price
                            pnl_usd = price_change_percent * trade.remaining_size * trade_leverage
                            final_pnl_percent = price_change_percent * trade_leverage * 100
                            logger.warning(f"⚠️ SMART EXIT: No exchange PnL, using manual calc with {trade_leverage}x: ${pnl_usd:.2f} ({final_pnl_percent:.1f}%)")
                        
                        be_threshold = max(0.01, trade.position_size * 0.001) if trade.position_size else 0.01
                        if abs(final_pnl_percent) < 0.1 and abs(pnl_usd) < be_threshold:
                            pnl_usd = 0.0
                            final_pnl_percent = 0.0
                        
                        # Set status based on P&L
                        if final_pnl_percent > 0:
                            trade.status = 'tp_hit'
                            trade.tp1_hit = True
                        elif final_pnl_percent < -1.0:
                            trade.status = 'sl_hit'
                        else:
                            trade.status = 'closed'  # Breakeven
                        
                        trade.exit_price = current_price
                        trade.closed_at = datetime.utcnow()
                        trade.pnl = float(pnl_usd)  # 🔥 FIX: Set directly, don't accumulate (prevents double-counting)
                        trade.pnl_percent = final_pnl_percent  # Use calculated pnl_percent directly
                        trade.remaining_size = 0
                        
                        from app.services.risk_controls import record_trade_result
                        record_trade_result(won=(trade.pnl > 0))
                        
                        if trade.pnl > 0:
                            prefs.consecutive_losses = 0
                            
                            # AUTO-COMPOUND: Update Top Gainer win streak on smart exit WIN (Upgrade #7)
                            if trade.trade_type == 'TOP_GAINER' and prefs.top_gainers_auto_compound:
                                prefs.top_gainers_win_streak += 1
                                
                                # After 3 wins in a row, increase position size by 20%
                                if prefs.top_gainers_win_streak >= 3:
                                    prefs.top_gainers_position_multiplier = 1.2
                                    logger.info(f"🔥 TOP GAINER AUTO-COMPOUND (Smart Exit Win): User {user.id} hit 3-win streak! Position size +20%")
                                else:
                                    logger.info(f"TOP GAINER WIN STREAK (Smart Exit): User {user.id} - {prefs.top_gainers_win_streak}/3 wins")
                        else:
                            prefs.consecutive_losses += 1
                            
                            # AUTO-COMPOUND: Reset Top Gainer streak on smart exit LOSS (Upgrade #7)
                            if trade.trade_type == 'TOP_GAINER' and prefs.top_gainers_auto_compound:
                                if prefs.top_gainers_win_streak > 0 or prefs.top_gainers_position_multiplier > 1.0:
                                    logger.info(f"🔄 TOP GAINER LOSS (Smart Exit): User {user.id} - Resetting position multiplier from {prefs.top_gainers_position_multiplier}x to 1.0x")
                                prefs.top_gainers_win_streak = 0
                                prefs.top_gainers_position_multiplier = 1.0
                        
                        db.commit()
                        
                        # Update signal analytics
                        if trade.signal_id:
                            AnalyticsService.update_signal_outcome(db, trade.signal_id)
                        
                        # Log social/news trade closure (Smart Exit)
                        if trade.trade_type in ['SOCIAL_SIGNAL', 'SOCIAL_SHORT', 'NEWS_SIGNAL']:
                            try:
                                from app.services.social_trade_logger import log_social_trade_closed
                                await log_social_trade_closed(db, trade)
                            except Exception as log_err:
                                logger.warning(f"Failed to log social trade: {log_err}")
                        
                        # NO NOTIFICATION for smart exits - user only wants TP/SL notifications
                        # Just log it silently
                        logger.info(f"✅ SMART EXIT (Silent): Trade {trade.id} - {exit_reason} - PnL ${trade.pnl:.2f} ({trade.pnl_percent:+.1f}%)")
                        
                        logger.info(f"Smart exit completed for trade {trade.id}: PnL ${trade.pnl:.2f}")
                        continue  # Skip normal TP/SL checks
                
                # ====================
                # BREAKEVEN STOP LOSS: Move SL to entry after TP1 hit
                # ====================
                # Only for dual TP trades (take_profit_2 set) that haven't hit TP1 yet
                if trade.take_profit_2 and trade.take_profit_1 and not trade.tp1_hit and trade.stop_loss != trade.entry_price:
                    # Log that we're checking for TP1
                    logger.debug(f"🔍 Checking TP1 for {trade.symbol}: Price ${current_price:.6f} vs TP1 ${trade.take_profit_1:.6f} ({trade.direction})")
                    
                    # Check if TP1 has been reached using price OR recent candle high/low
                    # This catches TP1 touches that happened between monitor cycles
                    tp1_reached = False
                    
                    # First check current price
                    if trade.direction == 'LONG':
                        tp1_reached = current_price >= trade.take_profit_1
                    else:  # SHORT
                        tp1_reached = current_price <= trade.take_profit_1
                    
                    # If not hit by current price, check recent 1m candle highs/lows
                    # This catches brief TP1 touches that pulled back
                    if not tp1_reached:
                        try:
                            recent_candles = await trader.fetch_candles(trade.symbol, '1m', limit=3)
                            if recent_candles and len(recent_candles) >= 2:
                                if trade.direction == 'LONG':
                                    # Check if any recent candle HIGH touched TP1
                                    recent_high = max(c[2] for c in recent_candles[-2:])  # Last 2 candles
                                    if recent_high >= trade.take_profit_1:
                                        tp1_reached = True
                                        logger.info(f"🎯 TP1 TOUCHED via candle high: ${recent_high:.6f} >= TP1 ${trade.take_profit_1:.6f}")
                                else:  # SHORT
                                    # Check if any recent candle LOW touched TP1
                                    recent_low = min(c[3] for c in recent_candles[-2:])
                                    if recent_low <= trade.take_profit_1:
                                        tp1_reached = True
                                        logger.info(f"🎯 TP1 TOUCHED via candle low: ${recent_low:.6f} <= TP1 ${trade.take_profit_1:.6f}")
                        except Exception as e:
                            logger.warning(f"Could not fetch candles for TP1 check: {e}")
                    
                    if tp1_reached:
                        logger.info(f"🎯 TP1 REACHED: {trade.symbol} {trade.direction} - Price ${current_price:.6f} hit TP1 ${trade.take_profit_1:.6f}")
                        
                        # 🔥 BREAKEVEN: Modify TP/SL orders to move SL to entry price
                        # This keeps TP2 intact while changing just the SL component
                        sl_modified = await trader.modify_tpsl_order_sl(
                            symbol=trade.symbol,
                            new_sl_price=trade.entry_price
                        )
                        
                        # Also update position-level SL as backup
                        sl_updated = await trader.update_position_stop_loss(
                            symbol=trade.symbol,
                            new_stop_loss=trade.entry_price,
                            direction=trade.direction
                        )
                        
                        if sl_modified or sl_updated:
                            # Move SL to entry (breakeven) in database
                            old_sl = trade.stop_loss
                            trade.stop_loss = trade.entry_price
                            trade.tp1_hit = True  # Mark TP1 as hit
                            trade.remaining_size = trade.position_size / 2  # 50% remaining after TP1
                            db.commit()
                            
                            logger.info(f"✅ BREAKEVEN ACTIVATED: Trade {trade.id} ({trade.symbol}) - SL moved from ${old_sl:.6f} to entry ${trade.entry_price:.6f} on Bitunix")
                            
                            # Calculate TP1 profit (50% of position at 50% profit = 25% total)
                            tp1_profit_pct = 50.0  # 50% profit on the 50% that closed
                            tp1_profit_usd = (trade.position_size / 2) * (tp1_profit_pct / 100)
                            
                            # Notify user
                            await bot.send_message(
                                user.telegram_id,
                                f"✅ <b>TP1 HIT - BREAKEVEN ACTIVATED!</b>\n\n"
                                f"<b>{trade.symbol}</b> {trade.direction}\n"
                                f"Entry: ${trade.entry_price:.6f}\n"
                                f"TP1: ${trade.take_profit_1:.6f}\n\n"
                                f"💰 50% closed at TP1 (+50% profit = ~${tp1_profit_usd:.2f})\n"
                                f"🔒 Stop Loss moved to ENTRY (breakeven)\n"
                                f"🎯 Remaining 50% now RISK-FREE!\n\n"
                                f"Targeting TP2 @ ${trade.take_profit_2:.6f} (+100%) 🚀",
                                parse_mode='HTML'
                            )
                        else:
                            logger.warning(f"⚠️ Failed to update SL to breakeven on Bitunix for {trade.symbol} - will retry next cycle")
                
                # ====================
                # Check TP/SL hits by comparing ACTUAL PRICE LEVELS
                # NOT position P&L percentage (which includes leverage)
                # ====================
                tp_hit = False
                sl_hit = False
                
                # Use highest TP available (TP3 > TP2 > TP1 > TP)
                if trade.take_profit_3:
                    tp_price = trade.take_profit_3  # Target TP3 for parabolic dumps
                elif trade.take_profit_2:
                    tp_price = trade.take_profit_2  # Target TP2 for dual TP trades
                elif trade.take_profit_1:
                    tp_price = trade.take_profit_1  # Target TP1 for single TP trades
                else:
                    tp_price = trade.take_profit  # Fallback to legacy take_profit
                
                # ✅ FIX: Check if ACTUAL PRICE hit the TP/SL levels (not P&L %)
                # LONG: TP hit if price >= TP, SL hit if price <= SL
                # SHORT: TP hit if price <= TP, SL hit if price >= SL
                if trade.direction == 'LONG':
                    if tp_price and current_price >= tp_price:
                        tp_hit = True
                        logger.info(f"✅ TP HIT: {trade.symbol} LONG - Price ${current_price:.4f} >= TP ${tp_price:.4f}")
                    elif trade.stop_loss and current_price <= trade.stop_loss:
                        sl_hit = True
                        logger.info(f"⛔ SL HIT: {trade.symbol} LONG - Price ${current_price:.4f} <= SL ${trade.stop_loss:.4f}")
                else:  # SHORT
                    if tp_price and current_price <= tp_price:
                        tp_hit = True
                        logger.info(f"✅ TP HIT: {trade.symbol} SHORT - Price ${current_price:.4f} <= TP ${tp_price:.4f}")
                    elif trade.stop_loss and current_price >= trade.stop_loss:
                        sl_hit = True
                        logger.info(f"⛔ SL HIT: {trade.symbol} SHORT - Price ${current_price:.4f} >= SL ${trade.stop_loss:.4f}")
                
                # Handle TP hit (Single TP - close entire position)
                if tp_hit:
                    result = await trader.close_position(trade.symbol, trade.direction)
                    
                    if result:
                        # Prioritize exchange-reported PnL when available
                        if trade.exchange_unrealized_pnl is not None:
                            pnl_usd = trade.exchange_unrealized_pnl
                            pnl_percent = (pnl_usd / trade.position_size) * 100 if trade.position_size > 0 else 0
                            logger.info(f"📊 TP: Using exchange-reported PnL: ${pnl_usd:.2f} ({pnl_percent:.1f}%)")
                        else:
                            trade_leverage = trade.leverage if trade.leverage else (prefs.top_gainers_leverage if trade.trade_type == 'TOP_GAINER' else (prefs.user_leverage or 5))
                            price_change = current_price - trade.entry_price if trade.direction == 'LONG' else trade.entry_price - current_price
                            price_change_percent = price_change / trade.entry_price
                            pnl_usd = price_change_percent * trade.remaining_size * trade_leverage
                            pnl_percent = price_change_percent * trade_leverage * 100
                            logger.warning(f"⚠️ TP: No exchange PnL, using manual calc with {trade_leverage}x: ${pnl_usd:.2f} ({pnl_percent:.1f}%)")
                        
                        trade.status = 'tp_hit'
                        trade.exit_price = current_price
                        trade.closed_at = datetime.utcnow()
                        trade.pnl = float(pnl_usd)  # 🔥 FIX: Set directly, don't accumulate (prevents double-counting)
                        trade.pnl_percent = pnl_percent  # Use calculated pnl_percent directly
                        trade.remaining_size = 0
                        trade.tp1_hit = True  # Mark TP1 as hit
                        
                        # Reset consecutive losses on win
                        prefs.consecutive_losses = 0
                        
                        # AUTO-COMPOUND: Update Top Gainer win streak (Upgrade #7)
                        if trade.trade_type == 'TOP_GAINER' and prefs.top_gainers_auto_compound:
                            prefs.top_gainers_win_streak += 1
                            
                            # After 3 wins in a row, increase position size by 20%
                            if prefs.top_gainers_win_streak >= 3:
                                prefs.top_gainers_position_multiplier = 1.2
                                logger.info(f"🔥 TOP GAINER AUTO-COMPOUND: User {user.id} hit 3-win streak! Position size +20%")
                            else:
                                logger.info(f"TOP GAINER WIN STREAK: User {user.id} - {prefs.top_gainers_win_streak}/3 wins")
                        
                        db.commit()
                        
                        # Update signal analytics
                        if trade.signal_id:
                            AnalyticsService.update_signal_outcome(db, trade.signal_id)
                        
                        # Log social/news trade closure (TP Hit)
                        if trade.trade_type in ['SOCIAL_SIGNAL', 'SOCIAL_SHORT', 'NEWS_SIGNAL']:
                            try:
                                from app.services.social_trade_logger import log_social_trade_closed
                                await log_social_trade_closed(db, trade)
                            except Exception as log_err:
                                logger.warning(f"Failed to log social trade: {log_err}")
                        
                        # Create share button for winning trade
                        from aiogram.types import InlineKeyboardMarkup, InlineKeyboardButton
                        share_keyboard = InlineKeyboardMarkup(inline_keyboard=[
                            [InlineKeyboardButton(text="📸 Share This Win", callback_data=f"share_trade_{trade.id}")]
                        ])
                        
                        await bot.send_message(
                            user.telegram_id,
                            f"🎯 TAKE PROFIT HIT! Position CLOSED 🎯\n\n"
                            f"Symbol: {trade.symbol}\n"
                            f"Entry: ${trade.entry_price:.4f}\n"
                            f"TP: ${tp_price:.4f}\n\n"
                            f"💰 Total PnL: ${trade.pnl:.2f} ({trade.pnl_percent:+.1f}%)\n"
                            f"Position Size: ${trade.position_size:.2f}",
                            reply_markup=share_keyboard
                        )
                        
                        # Generate and send trade screenshot automatically
                        await send_trade_screenshot(bot, trade, user, db)
                
                # Handle SL hit
                elif sl_hit:
                    result = await trader.close_position(trade.symbol, trade.direction)
                    
                    if result:
                        # Prioritize exchange-reported PnL when available
                        if trade.exchange_unrealized_pnl is not None:
                            pnl_usd = trade.exchange_unrealized_pnl
                            pnl_percent = (pnl_usd / trade.position_size) * 100 if trade.position_size > 0 else 0
                            logger.info(f"📊 SL: Using exchange-reported PnL: ${pnl_usd:.2f} ({pnl_percent:.1f}%)")
                        else:
                            trade_leverage = trade.leverage if trade.leverage else (prefs.top_gainers_leverage if trade.trade_type == 'TOP_GAINER' else (prefs.user_leverage or 5))
                            price_change = current_price - trade.entry_price if trade.direction == 'LONG' else trade.entry_price - current_price
                            price_change_percent = price_change / trade.entry_price
                            pnl_usd = price_change_percent * trade.remaining_size * trade_leverage
                            pnl_percent = price_change_percent * trade_leverage * 100
                            logger.warning(f"⚠️ SL: No exchange PnL, using manual calc with {trade_leverage}x: ${pnl_usd:.2f} ({pnl_percent:.1f}%)")
                        
                        trade.status = 'sl_hit'
                        trade.exit_price = current_price
                        trade.closed_at = datetime.utcnow()
                        trade.pnl = float(pnl_usd)  # 🔥 FIX: Set directly, don't accumulate (prevents double-counting)
                        trade.pnl_percent = pnl_percent  # Use calculated pnl_percent directly
                        trade.remaining_size = 0
                        
                        # Add SHORT cooldown if this was a losing SHORT (prevents re-shorting strong pumps)
                        if trade.direction == 'SHORT':
                            from app.services.top_gainers_signals import add_short_cooldown
                            add_short_cooldown(trade.symbol, cooldown_minutes=30)
                        
                        # Update consecutive losses
                        prefs.consecutive_losses += 1
                        
                        # AUTO-COMPOUND: Reset Top Gainer streak on loss (Upgrade #7)
                        if trade.trade_type == 'TOP_GAINER' and prefs.top_gainers_auto_compound:
                            if prefs.top_gainers_win_streak > 0 or prefs.top_gainers_position_multiplier > 1.0:
                                logger.info(f"🔄 TOP GAINER LOSS: User {user.id} - Resetting position multiplier from {prefs.top_gainers_position_multiplier}x to 1.0x")
                            prefs.top_gainers_win_streak = 0
                            prefs.top_gainers_position_multiplier = 1.0
                        
                        db.commit()
                        
                        # Update signal analytics
                        if trade.signal_id:
                            AnalyticsService.update_signal_outcome(db, trade.signal_id)
                        
                        # Log social/news trade closure (SL Hit)
                        if trade.trade_type in ['SOCIAL_SIGNAL', 'SOCIAL_SHORT', 'NEWS_SIGNAL']:
                            try:
                                from app.services.social_trade_logger import log_social_trade_closed
                                await log_social_trade_closed(db, trade)
                            except Exception as log_err:
                                logger.warning(f"Failed to log social trade: {log_err}")
                        
                        # No share button for stop losses
                        await bot.send_message(
                            user.telegram_id,
                            f"🛑 STOP LOSS HIT!\n\n"
                            f"Symbol: {trade.symbol}\n"
                            f"Entry: ${trade.entry_price:.4f}\n"
                            f"SL: ${trade.stop_loss:.4f}\n\n"
                            f"💰 PnL: ${trade.pnl:.2f} ({trade.pnl_percent:+.1f}%)\n"
                            f"Position Size: ${trade.position_size:.2f}"
                        )
                        
                        # Generate and send trade screenshot automatically
                        await send_trade_screenshot(bot, trade, user, db)
                
            except Exception as e:
                logger.error(f"Error monitoring trade {trade.id}: {e}", exc_info=True)
            finally:
                if trader:
                    await trader.close()
    
    except Exception as e:
        logger.error(f"Error in position monitor: {e}", exc_info=True)
    finally:
        db.close()


async def send_trade_screenshot(bot, trade: Trade, user: User, db: Session):
    """Generate and send trade screenshot after close"""
    try:
        # Calculate duration
        duration_hours = None
        if trade.created_at and trade.closed_at:
            duration_hours = (trade.closed_at - trade.created_at).total_seconds() / 3600
        
        # Calculate win streak
        recent_trades = db.query(Trade).filter(
            Trade.user_id == user.id,
            Trade.status.in_(['closed', 'stopped']),
            Trade.id <= trade.id
        ).order_by(Trade.id.desc()).limit(10).all()
        
        win_streak = 0
        for t in recent_trades:
            if t.pnl and t.pnl > 0:
                win_streak += 1
            else:
                break
        
        # Get trade type label
        trade_type = trade.trade_type or "DAY_TRADING"
        strategy = "Top Gainer" if trade_type == "TOP_GAINER" else "Day Trading"
        
        # Generate screenshot
        img_bytes = screenshot_generator.generate_trade_card(
            symbol=trade.symbol,
            direction=trade.direction,
            entry_price=trade.entry_price,
            exit_price=trade.exit_price or trade.entry_price,
            pnl_percentage=trade.pnl_percent or 0,
            pnl_amount=trade.pnl or 0,
            trade_type=trade_type,
            duration_hours=duration_hours,
            win_streak=win_streak,
            strategy=strategy
        )
        
        # Send to user
        result_emoji = "✅" if trade.pnl and trade.pnl > 0 else "❌"
        caption = f"{result_emoji} <b>Trade Closed</b> | {trade.symbol} {trade.direction}\n\n📊 Share your results!"
        
        from aiogram.types import BufferedInputFile
        photo = BufferedInputFile(img_bytes.read(), filename=f"trade_{trade.id}.png")
        
        await bot.send_photo(
            chat_id=user.telegram_id,
            photo=photo,
            caption=caption,
            parse_mode="HTML"
        )
        
        logger.info(f"✅ Trade screenshot sent to user {user.id} for trade {trade.id}")
        
    except Exception as e:
        logger.error(f"Error sending trade screenshot: {e}", exc_info=True)
