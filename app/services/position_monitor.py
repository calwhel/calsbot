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
                        # Calculate final PnL
                        price_change = current_price - trade.entry_price if trade.direction == 'LONG' else trade.entry_price - current_price
                        pnl_usd = (price_change / trade.entry_price) * trade.remaining_size
                        
                        trade.status = 'closed'
                        trade.exit_price = current_price
                        trade.closed_at = datetime.utcnow()
                        trade.pnl += float(pnl_usd)
                        trade.pnl_percent = (trade.pnl / trade.position_size) * 100
                        trade.remaining_size = 0
                        
                        # Update consecutive losses
                        if trade.pnl > 0:
                            prefs.consecutive_losses = 0
                        else:
                            prefs.consecutive_losses += 1
                        
                        db.commit()
                        
                        # Update signal analytics
                        if trade.signal_id:
                            AnalyticsService.update_signal_outcome(db, trade.signal_id)
                        
                        await bot.send_message(
                            user.telegram_id,
                            f"🧠 SMART EXIT - Reversal Detected!\n\n"
                            f"Symbol: {trade.symbol} {trade.direction}\n"
                            f"Entry: ${trade.entry_price:.4f}\n"
                            f"Exit: ${current_price:.4f}\n\n"
                            f"{exit_reason}\n\n"
                            f"💰 PnL: ${trade.pnl:.2f} ({trade.pnl_percent:+.1f}%)\n"
                            f"Position Size: ${trade.position_size:.2f}\n\n"
                            f"✅ Position closed early to protect capital"
                        )
                        
                        logger.info(f"Smart exit completed for trade {trade.id}: PnL ${trade.pnl:.2f}")
                        continue  # Skip normal TP/SL checks
                
                # ====================
                # Check TP/SL hits
                # ====================
                tp1_hit = False
                tp2_hit = False
                tp3_hit = False
                sl_hit = False
                
                if trade.direction == 'LONG':
                    if not trade.tp1_hit and trade.take_profit_1 and current_price >= trade.take_profit_1:
                        tp1_hit = True
                    elif not trade.tp2_hit and trade.take_profit_2 and current_price >= trade.take_profit_2:
                        tp2_hit = True
                    elif not trade.tp3_hit and trade.take_profit_3 and current_price >= trade.take_profit_3:
                        tp3_hit = True
                    elif current_price <= trade.stop_loss:
                        sl_hit = True
                else:  # SHORT
                    if not trade.tp1_hit and trade.take_profit_1 and current_price <= trade.take_profit_1:
                        tp1_hit = True
                    elif not trade.tp2_hit and trade.take_profit_2 and current_price <= trade.take_profit_2:
                        tp2_hit = True
                    elif not trade.tp3_hit and trade.take_profit_3 and current_price <= trade.take_profit_3:
                        tp3_hit = True
                    elif current_price >= trade.stop_loss:
                        sl_hit = True
                
                # Handle TP1 hit (partial close + move SL to breakeven)
                if tp1_hit:
                    close_percent = prefs.tp1_percent / 100
                    amount_to_close = remaining_amount * close_percent
                    
                    result = await trader.close_partial_position(
                        symbol=trade.symbol,
                        direction=trade.direction,
                        amount_to_close=amount_to_close,
                        close_price=current_price
                    )
                    
                    if result:
                        price_change = current_price - trade.entry_price if trade.direction == 'LONG' else trade.entry_price - current_price
                        pnl_usd = (price_change / trade.entry_price) * (amount_to_close * current_price)
                        
                        trade.tp1_hit = True
                        trade.remaining_size = trade.remaining_size - (amount_to_close * current_price)
                        trade.pnl += float(pnl_usd)
                        
                        # Move SL to breakeven
                        old_sl = trade.stop_loss
                        trade.stop_loss = trade.entry_price
                        trade.breakeven_moved = True
                        
                        db.commit()
                        
                        await bot.send_message(
                            user.telegram_id,
                            f"🎯 TP1 HIT! ({prefs.tp1_percent}% closed)\n\n"
                            f"Symbol: {trade.symbol}\n"
                            f"TP1: ${trade.take_profit_1:.4f}\n"
                            f"💰 Partial PnL: ${pnl_usd:.2f}\n\n"
                            f"🔒 Stop Loss moved to BREAKEVEN\n"
                            f"Old SL: ${old_sl:.4f} → Entry: ${trade.entry_price:.4f}\n"
                            f"Risk eliminated! 🎯"
                        )
                
                # Handle TP2 hit
                elif tp2_hit:
                    close_percent = prefs.tp2_percent / 100
                    amount_to_close = remaining_amount * close_percent
                    
                    result = await trader.close_partial_position(
                        symbol=trade.symbol,
                        direction=trade.direction,
                        amount_to_close=amount_to_close,
                        close_price=current_price
                    )
                    
                    if result:
                        price_change = current_price - trade.entry_price if trade.direction == 'LONG' else trade.entry_price - current_price
                        pnl_usd = (price_change / trade.entry_price) * (amount_to_close * current_price)
                        
                        trade.tp2_hit = True
                        trade.remaining_size = trade.remaining_size - (amount_to_close * current_price)
                        trade.pnl += float(pnl_usd)
                        
                        db.commit()
                        
                        await bot.send_message(
                            user.telegram_id,
                            f"🎯 TP2 HIT! ({prefs.tp2_percent}% closed)\n\n"
                            f"Symbol: {trade.symbol}\n"
                            f"TP2: ${trade.take_profit_2:.4f}\n"
                            f"💰 Partial PnL: ${pnl_usd:.2f}"
                        )
                
                # Handle TP3 hit (close remaining position)
                elif tp3_hit:
                    result = await trader.close_position(trade.symbol, trade.direction)
                    
                    if result:
                        price_change = current_price - trade.entry_price if trade.direction == 'LONG' else trade.entry_price - current_price
                        pnl_usd = (price_change / trade.entry_price) * (remaining_amount * current_price)
                        
                        trade.tp3_hit = True
                        trade.status = 'closed'
                        trade.exit_price = current_price
                        trade.closed_at = datetime.utcnow()
                        trade.pnl += float(pnl_usd)
                        trade.pnl_percent = (trade.pnl / trade.position_size) * 100
                        trade.remaining_size = 0
                        
                        # Reset consecutive losses on win
                        prefs.consecutive_losses = 0
                        
                        db.commit()
                        
                        # Update signal analytics
                        if trade.signal_id:
                            AnalyticsService.update_signal_outcome(db, trade.signal_id)
                        
                        await bot.send_message(
                            user.telegram_id,
                            f"🎯 TP3 HIT! Position CLOSED 🎯\n\n"
                            f"Symbol: {trade.symbol}\n"
                            f"Entry: ${trade.entry_price:.4f}\n"
                            f"TP3: ${trade.take_profit_3:.4f}\n\n"
                            f"💰 Total PnL: ${trade.pnl:.2f} ({trade.pnl_percent:+.1f}%)\n"
                            f"Position Size: ${trade.position_size:.2f}"
                        )
                
                # Handle SL hit
                elif sl_hit:
                    result = await trader.close_position(trade.symbol, trade.direction)
                    
                    if result:
                        price_change = current_price - trade.entry_price if trade.direction == 'LONG' else trade.entry_price - current_price
                        pnl_usd = (price_change / trade.entry_price) * (remaining_amount * current_price)
                        
                        trade.status = 'closed'
                        trade.exit_price = current_price
                        trade.closed_at = datetime.utcnow()
                        trade.pnl += float(pnl_usd)
                        trade.pnl_percent = (trade.pnl / trade.position_size) * 100
                        trade.remaining_size = 0
                        
                        # Update consecutive losses
                        prefs.consecutive_losses += 1
                        
                        db.commit()
                        
                        # Update signal analytics
                        if trade.signal_id:
                            AnalyticsService.update_signal_outcome(db, trade.signal_id)
                        
                        await bot.send_message(
                            user.telegram_id,
                            f"🛑 STOP LOSS HIT!\n\n"
                            f"Symbol: {trade.symbol}\n"
                            f"Entry: ${trade.entry_price:.4f}\n"
                            f"SL: ${trade.stop_loss:.4f}\n\n"
                            f"💰 PnL: ${trade.pnl:.2f} ({trade.pnl_percent:+.1f}%)\n"
                            f"Position Size: ${trade.position_size:.2f}"
                        )
                
            except Exception as e:
                logger.error(f"Error monitoring trade {trade.id}: {e}", exc_info=True)
            finally:
                if trader:
                    await trader.close()
    
    except Exception as e:
        logger.error(f"Error in position monitor: {e}", exc_info=True)
    finally:
        db.close()
