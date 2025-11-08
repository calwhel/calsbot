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


async def monitor_positions(bot):
    """Monitor open Bitunix positions and handle TP/SL hits + smart exits"""
    db = SessionLocal()
    
    try:
        from datetime import timedelta
        
        # Get ALL open trades (manual + auto) with Bitunix API keys configured
        # Skip trades opened in last 2 minutes to prevent false "position closed" notifications
        # Bitunix API can be slow to show new positions, so brief grace period needed
        grace_period = datetime.utcnow() - timedelta(minutes=2)
        
        open_trades = db.query(Trade).join(User).join(UserPreference).filter(
            Trade.status == 'open',
            Trade.opened_at < grace_period,  # Only check trades older than 2 minutes
            UserPreference.bitunix_api_key != None  # Just need API keys, not auto-trading enabled
        ).all()
        
        if not open_trades:
            return
        
        logger.info(f"Monitoring {len(open_trades)} open Bitunix positions (skipping trades opened in last 2 minutes)...")
        
        for trade in open_trades:
            trader = None
            try:
                user = trade.user
                prefs = user.preferences
                
                # Decrypt Bitunix credentials
                api_key = decrypt_api_key(prefs.bitunix_api_key)
                api_secret = decrypt_api_key(prefs.bitunix_api_secret)
                trader = BitunixTrader(api_key, api_secret)
                
                # 🔥 CRITICAL FIX: Check if position is still open on Bitunix first
                # This catches positions that Bitunix closed automatically (TP/SL hit on exchange)
                bitunix_positions = await trader.get_open_positions()
                bitunix_symbol = trade.symbol.replace('/', '')
                
                position_exists = False
                for pos in bitunix_positions:
                    if pos['symbol'] == bitunix_symbol:
                        expected_side = 'long' if trade.direction == 'LONG' else 'short'
                        if pos['hold_side'].lower() == expected_side:
                            position_exists = True
                            break
                
                # If position is closed on Bitunix but open in DB, sync it
                if not position_exists:
                    # Log trade age for debugging
                    trade_age_minutes = (datetime.utcnow() - trade.opened_at).total_seconds() / 60
                    logger.info(f"🔄 SYNC: Position {trade.id} ({trade.symbol}) closed on Bitunix but open in DB - Trade age: {trade_age_minutes:.1f} minutes")
                    
                    # 🔥 FIX #1: Always fetch current_price to avoid UnboundLocalError
                    current_price = await trader.get_current_price(trade.symbol)
                    if not current_price:
                        logger.warning(f"Could not fetch price for {trade.symbol} during sync")
                        continue
                    
                    # 🔥 USE LIVE API DATA: Get last known exchange PnL before closure
                    # If exchange reported PnL exists, use it; otherwise fallback to manual calculation
                    if trade.exchange_unrealized_pnl is not None:
                        # Use exchange-reported PnL directly (already includes fees, funding, etc.)
                        pnl_usd = trade.exchange_unrealized_pnl
                        pnl_percent = (pnl_usd / trade.position_size) * 100 if trade.position_size > 0 else 0
                        logger.info(f"📊 Using exchange-reported PnL: ${pnl_usd:.2f} ({pnl_percent:.1f}%)")
                    else:
                        # Fallback: Calculate manually if exchange data missing
                        leverage = prefs.top_gainers_leverage if trade.trade_type == 'TOP_GAINER' else (prefs.user_leverage or 5)
                        price_change = current_price - trade.entry_price if trade.direction == 'LONG' else trade.entry_price - current_price
                        price_change_percent = price_change / trade.entry_price
                        pnl_usd = price_change_percent * trade.remaining_size * leverage
                        pnl_percent = (pnl_usd / trade.remaining_size) * 100 if trade.remaining_size > 0 else 0
                        logger.warning(f"⚠️ No exchange PnL data, using manual calculation: ${pnl_usd:.2f}")
                        
                    # Store final realized PnL from exchange
                    if trade.exchange_unrealized_pnl is not None:
                        trade.exchange_realized_pnl = trade.exchange_unrealized_pnl
                    
                    # 🔥 BREAKEVEN TOLERANCE: If abs(PnL) < 1%, mark as 0% (prevents price drift fake losses)
                    # This fixes bug where breakeven closes at 0% on exchange but ticker drift creates -0.7% to -2.7%
                    if abs(pnl_percent) < 1.0 and abs(pnl_usd) < 0.5:
                        pnl_usd = 0.0
                        pnl_percent = 0.0
                        logger.info(f"📊 BREAKEVEN: {trade.symbol} P&L within tolerance (<1%), setting to 0%")
                    
                    # ✅ FIX: Check if ACTUAL PRICE hit TP/SL (not P&L %)
                    # This prevents false SL triggers at high leverage (e.g., 20x leverage: -1% P&L = only -0.05% price move)
                    tp_price = trade.take_profit_1 if trade.take_profit_1 else trade.take_profit
                    tp_hit = False
                    sl_hit = False
                    
                    # Check TP/SL based on ACTUAL PRICE vs target levels
                    if trade.direction == 'LONG':
                        # LONG: TP hit if price >= TP, SL hit if price <= SL
                        if tp_price and current_price >= tp_price:
                            tp_hit = True
                            trade.tp1_hit = True
                            logger.info(f"✅ TP HIT: {trade.symbol} LONG - Price ${current_price:.6f} >= TP ${tp_price:.6f}")
                        elif trade.stop_loss and current_price <= trade.stop_loss:
                            sl_hit = True
                            logger.info(f"⛔ SL HIT: {trade.symbol} LONG - Price ${current_price:.6f} <= SL ${trade.stop_loss:.6f}")
                        else:
                            logger.info(f"⚪ Position closed but no TP/SL hit: {trade.symbol} P&L {pnl_percent:.1f}%")
                    else:  # SHORT
                        # SHORT: TP hit if price <= TP, SL hit if price >= SL
                        if tp_price and current_price <= tp_price:
                            tp_hit = True
                            trade.tp1_hit = True
                            logger.info(f"✅ TP HIT: {trade.symbol} SHORT - Price ${current_price:.6f} <= TP ${tp_price:.6f}")
                        elif trade.stop_loss and current_price >= trade.stop_loss:
                            sl_hit = True
                            logger.info(f"⛔ SL HIT: {trade.symbol} SHORT - Price ${current_price:.6f} >= SL ${trade.stop_loss:.6f}")
                        else:
                            logger.info(f"⚪ Position closed but no TP/SL hit: {trade.symbol} P&L {pnl_percent:.1f}%")
                    
                    # 🔥 FIX #2: Set PnL directly, don't accumulate (prevents double-counting)
                    trade.status = 'tp_hit' if tp_hit else ('sl_hit' if sl_hit else 'closed')
                    trade.exit_price = current_price
                    trade.closed_at = datetime.utcnow()
                    trade.pnl = float(pnl_usd)  # Set directly from final PnL (exchange or calculated)
                    trade.pnl_percent = pnl_percent  # Use calculated pnl_percent directly
                    trade.remaining_size = 0
                    
                    # Update consecutive losses
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
                                f"Exit: ${current_price:.4f}\n\n"
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
                        logger.info(f"Position closed (no TP/SL): Trade {trade.id} - PnL ${trade.pnl:.2f}, Entry: ${trade.entry_price:.4f}, Exit: ${current_price:.4f}, TP: ${tp_price}, SL: ${trade.stop_loss}")
                    
                    logger.info(f"✅ Synced closed position: Trade {trade.id} - PnL ${trade.pnl:.2f}")
                    continue  # Skip rest of checks for this trade
                
                # Position is still open on Bitunix - continue with normal monitoring
                # 🔥 CRITICAL: Fetch live position data from Bitunix API
                position_data = await trader.get_position_detail(trade.symbol)
                
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
                # ====================
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
                        # Calculate final PnL with leverage
                        leverage = prefs.top_gainers_leverage if trade.trade_type == 'TOP_GAINER' else (prefs.user_leverage or 5)
                        price_change = current_price - trade.entry_price if trade.direction == 'LONG' else trade.entry_price - current_price
                        price_change_percent = price_change / trade.entry_price
                        pnl_usd = price_change_percent * trade.remaining_size * leverage
                        
                        # Apply breakeven tolerance
                        final_pnl_percent = (pnl_usd / trade.remaining_size) * 100 if trade.remaining_size > 0 else 0
                        if abs(final_pnl_percent) < 1.0 and abs(pnl_usd) < 0.5:
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
                        trade.pnl += float(pnl_usd)
                        trade.pnl_percent = (trade.pnl / trade.position_size) * 100
                        trade.remaining_size = 0
                        
                        # Update consecutive losses
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
                        
                        # NO NOTIFICATION for smart exits - user only wants TP/SL notifications
                        # Just log it silently
                        logger.info(f"✅ SMART EXIT (Silent): Trade {trade.id} - {exit_reason} - PnL ${trade.pnl:.2f} ({trade.pnl_percent:+.1f}%)")
                        
                        logger.info(f"Smart exit completed for trade {trade.id}: PnL ${trade.pnl:.2f}")
                        continue  # Skip normal TP/SL checks
                
                # ====================
                # BREAKEVEN STOP LOSS: Move SL to entry after TP1 hit
                # ====================
                if trade.take_profit_1 and trade.stop_loss != trade.entry_price:
                    # Check if TP1 has been reached
                    tp1_reached = False
                    if trade.direction == 'LONG':
                        tp1_reached = current_price >= trade.take_profit_1
                    else:  # SHORT
                        tp1_reached = current_price <= trade.take_profit_1
                    
                    if tp1_reached:
                        # Move SL to entry (breakeven)
                        old_sl = trade.stop_loss
                        trade.stop_loss = trade.entry_price
                        trade.tp1_hit = True  # Mark TP1 as hit
                        db.commit()
                        
                        logger.info(f"✅ BREAKEVEN: Trade {trade.id} ({trade.symbol}) - TP1 hit! Moving SL from ${old_sl:.6f} to entry ${trade.entry_price:.6f}")
                        
                        # Notify user
                        await bot.send_message(
                            user.telegram_id,
                            f"✅ <b>TP1 HIT - BREAKEVEN ACTIVATED!</b>\n\n"
                            f"<b>{trade.symbol}</b> {trade.direction}\n"
                            f"Entry: ${trade.entry_price:.6f}\n"
                            f"Current Price: ${current_price:.6f}\n\n"
                            f"🔒 Stop Loss moved to ENTRY (breakeven)\n"
                            f"🎯 Position now risk-free!\n\n"
                            f"Waiting for TP2" + (f" and TP3 🚀" if trade.take_profit_3 else "") + "...",
                            parse_mode='HTML'
                        )
                
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
                        # Calculate PnL with leverage
                        leverage = prefs.top_gainers_leverage if trade.trade_type == 'TOP_GAINER' else (prefs.user_leverage or 5)
                        price_change = current_price - trade.entry_price if trade.direction == 'LONG' else trade.entry_price - current_price
                        price_change_percent = price_change / trade.entry_price
                        pnl_usd = price_change_percent * trade.remaining_size * leverage
                        
                        trade.status = 'tp_hit'
                        trade.exit_price = current_price
                        trade.closed_at = datetime.utcnow()
                        trade.pnl += float(pnl_usd)
                        trade.pnl_percent = (trade.pnl / trade.position_size) * 100
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
                        # Calculate PnL with leverage
                        leverage = prefs.top_gainers_leverage if trade.trade_type == 'TOP_GAINER' else (prefs.user_leverage or 5)
                        price_change = current_price - trade.entry_price if trade.direction == 'LONG' else trade.entry_price - current_price
                        price_change_percent = price_change / trade.entry_price
                        pnl_usd = price_change_percent * trade.remaining_size * leverage
                        
                        trade.status = 'sl_hit'
                        trade.exit_price = current_price
                        trade.closed_at = datetime.utcnow()
                        trade.pnl += float(pnl_usd)
                        trade.pnl_percent = (trade.pnl / trade.position_size) * 100
                        trade.remaining_size = 0
                        
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
