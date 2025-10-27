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
        # Get all open trades with users who have auto-trading enabled
        open_trades = db.query(Trade).join(User).join(UserPreference).filter(
            Trade.status == 'open',
            UserPreference.auto_trading_enabled == True,
            UserPreference.bitunix_api_key != None
        ).all()
        
        if not open_trades:
            return
        
        logger.info(f"Monitoring {len(open_trades)} open Bitunix positions...")
        
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
                    logger.info(f"🔄 SYNC: Position {trade.id} ({trade.symbol}) closed on Bitunix but open in DB - syncing now")
                    
                    # Get current price for PnL calculation
                    current_price = await trader.get_current_price(trade.symbol)
                    if not current_price:
                        logger.warning(f"Could not fetch price for {trade.symbol} during sync")
                        continue
                    
                    # Calculate PnL with leverage
                    leverage = prefs.top_gainers_leverage if trade.trade_type == 'TOP_GAINER' else (prefs.user_leverage or 5)
                    price_change = current_price - trade.entry_price if trade.direction == 'LONG' else trade.entry_price - current_price
                    price_change_percent = price_change / trade.entry_price
                    pnl_usd = price_change_percent * trade.position_size * leverage
                    
                    # Determine if TP or SL hit
                    tp_hit = False
                    sl_hit = False
                    
                    if trade.direction == 'LONG':
                        if trade.take_profit and current_price >= trade.take_profit:
                            tp_hit = True
                        elif current_price <= trade.stop_loss:
                            sl_hit = True
                    else:  # SHORT
                        if trade.take_profit and current_price <= trade.take_profit:
                            tp_hit = True
                        elif current_price >= trade.stop_loss:
                            sl_hit = True
                    
                    # Update database
                    trade.status = 'closed'
                    trade.exit_price = current_price
                    trade.closed_at = datetime.utcnow()
                    trade.pnl = float(pnl_usd)
                    trade.pnl_percent = (trade.pnl / trade.position_size) * 100
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
                    
                    # Send notification
                    exit_type = "🎯 TAKE PROFIT HIT!" if tp_hit else "⛔ STOP LOSS HIT!" if sl_hit else "✅ POSITION CLOSED"
                    
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
                    
                    # Generate and send trade screenshot
                    await send_trade_screenshot(bot, trade, user, db)
                    
                    logger.info(f"✅ Synced closed position: Trade {trade.id} - PnL ${trade.pnl:.2f}")
                    continue  # Skip rest of checks for this trade
                
                # Position is still open on Bitunix - continue with normal monitoring
                # Get current price
                current_price = await trader.get_current_price(trade.symbol)
                
                if not current_price:
                    logger.warning(f"Could not fetch price for {trade.symbol}")
                    continue
                
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
                        
                        trade.status = 'closed'
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
                        
                        # Only add share button if it's a win
                        from aiogram.types import InlineKeyboardMarkup, InlineKeyboardButton
                        share_keyboard = None
                        if trade.pnl > 0:
                            share_keyboard = InlineKeyboardMarkup(inline_keyboard=[
                                [InlineKeyboardButton(text="📸 Share This Win", callback_data=f"share_trade_{trade.id}")]
                            ])
                        
                        await bot.send_message(
                            user.telegram_id,
                            f"🧠 SMART EXIT - Reversal Detected!\n\n"
                            f"Symbol: {trade.symbol} {trade.direction}\n"
                            f"Entry: ${trade.entry_price:.4f}\n"
                            f"Exit: ${current_price:.4f}\n\n"
                            f"{exit_reason}\n\n"
                            f"💰 PnL: ${trade.pnl:.2f} ({trade.pnl_percent:+.1f}%)\n"
                            f"Position Size: ${trade.position_size:.2f}\n\n"
                            f"✅ Position closed early to protect capital",
                            reply_markup=share_keyboard
                        )
                        
                        # Generate and send trade screenshot automatically
                        await send_trade_screenshot(bot, trade, user, db)
                        
                        logger.info(f"Smart exit completed for trade {trade.id}: PnL ${trade.pnl:.2f}")
                        continue  # Skip normal TP/SL checks
                
                # ====================
                # Check TP/SL hits (Single TP strategy - 15% TP)
                # ====================
                tp_hit = False
                sl_hit = False
                
                # Check single TP (use take_profit field)
                if trade.direction == 'LONG':
                    if trade.take_profit and current_price >= trade.take_profit:
                        tp_hit = True
                    elif current_price <= trade.stop_loss:
                        sl_hit = True
                else:  # SHORT
                    if trade.take_profit and current_price <= trade.take_profit:
                        tp_hit = True
                    elif current_price >= trade.stop_loss:
                        sl_hit = True
                
                # Handle TP hit (Single TP - close entire position)
                if tp_hit:
                    result = await trader.close_position(trade.symbol, trade.direction)
                    
                    if result:
                        # Calculate PnL with leverage
                        leverage = prefs.top_gainers_leverage if trade.trade_type == 'TOP_GAINER' else (prefs.user_leverage or 5)
                        price_change = current_price - trade.entry_price if trade.direction == 'LONG' else trade.entry_price - current_price
                        price_change_percent = price_change / trade.entry_price
                        pnl_usd = price_change_percent * trade.remaining_size * leverage
                        
                        trade.status = 'closed'
                        trade.exit_price = current_price
                        trade.closed_at = datetime.utcnow()
                        trade.pnl += float(pnl_usd)
                        trade.pnl_percent = (trade.pnl / trade.position_size) * 100
                        trade.remaining_size = 0
                        
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
                            f"TP: ${trade.take_profit:.4f}\n\n"
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
                        
                        trade.status = 'closed'
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
