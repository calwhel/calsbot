import ccxt.async_support as ccxt
import logging
from typing import Optional
from sqlalchemy.orm import Session
from datetime import datetime
from app.models import User, UserPreference, PaperTrade, Signal
from app.database import SessionLocal
from app.services.price_cache import get_multiple_cached_prices
from app.services.multi_analysis import validate_trade_signal

logger = logging.getLogger(__name__)


class PaperTrader:
    """Handles paper trading (virtual trades without real money)"""
    
    @staticmethod
    async def execute_paper_trade(user_id: int, signal: Signal, db: Session) -> Optional[PaperTrade]:
        """Execute a paper trade from a signal with multi-analysis confirmation"""
        try:
            user = db.query(User).filter(User.id == user_id).first()
            if not user or not user.preferences:
                return None
            
            prefs = user.preferences
            
            if not prefs.paper_trading_mode:
                logger.info(f"Paper trading disabled for user {user_id}")
                return None
            
            # Skip validation for TEST signals (admin testing)
            if signal.signal_type != 'TEST':
                # MULTI-ANALYSIS CONFIRMATION CHECK
                # Validate signal against higher timeframe and multiple indicators
                is_valid, reason, analysis_data = await validate_trade_signal(
                    symbol=signal.symbol,
                    direction=signal.direction,
                    entry_price=signal.entry_price,
                    exchange_name='kucoin'
                )
                
                if not is_valid:
                    logger.info(f"Paper trade REJECTED for {signal.symbol} {signal.direction}: {reason}")
                    return None
                
                logger.info(f"Paper trade APPROVED for {signal.symbol} {signal.direction}: {reason}")
            else:
                logger.info(f"Paper TEST signal for user {user_id} - skipping multi-analysis validation")
            
            # Check if enough virtual balance
            if prefs.paper_balance < 10:
                logger.warning(f"Insufficient paper balance for user {user_id}: ${prefs.paper_balance}")
                return None
            
            # Calculate position size based on virtual balance
            position_size_percent = prefs.position_size_percent
            position_size_usdt = (prefs.paper_balance * position_size_percent) / 100
            
            # No max positions limit for paper trading - take all signals
            
            # Create paper trade
            paper_trade = PaperTrade(
                user_id=user_id,
                signal_id=signal.id,
                symbol=signal.symbol,
                direction=signal.direction,
                entry_price=signal.entry_price,
                stop_loss=signal.stop_loss,
                take_profit=signal.take_profit,
                take_profit_1=signal.take_profit_1,
                take_profit_2=signal.take_profit_2,
                take_profit_3=signal.take_profit_3,
                position_size=float(position_size_usdt),
                remaining_size=float(position_size_usdt),
                status="open"
            )
            
            db.add(paper_trade)
            
            # Deduct from virtual balance
            prefs.paper_balance -= position_size_usdt
            
            db.commit()
            db.refresh(paper_trade)
            
            logger.info(f"Paper trade executed for user {user_id}: {signal.symbol} {signal.direction}, size: ${position_size_usdt:.2f}")
            return paper_trade
            
        except Exception as e:
            logger.error(f"Error executing paper trade: {e}", exc_info=True)
            db.rollback()
            return None
    
    @staticmethod
    async def monitor_paper_positions(bot):
        """Monitor and update paper trade positions based on market prices"""
        db = SessionLocal()
        
        try:
            # Get all open paper trades
            open_trades = db.query(PaperTrade).filter(PaperTrade.status == "open").all()
            
            if not open_trades:
                return
            
            # Get all unique symbols for batch price fetching
            symbols = list(set([trade.symbol for trade in open_trades]))
            cached_prices = await get_multiple_cached_prices(symbols, 'kucoin')
            
            try:
                for trade in open_trades:
                    try:
                        # Get current price from cache (reduces API calls)
                        current_price = cached_prices.get(trade.symbol)
                        
                        if not current_price:
                            continue
                        
                        # Get user preferences
                        user = db.query(User).filter(User.id == trade.user_id).first()
                        if not user or not user.preferences:
                            continue
                        
                        prefs = user.preferences
                        remaining_amount = trade.remaining_size / current_price
                        
                        # Check TP/SL hits (same logic as real trading)
                        tp1_hit = False
                        tp2_hit = False
                        tp3_hit = False
                        sl_hit = False
                        
                        if trade.direction == 'LONG':
                            if trade.take_profit_1 and current_price >= trade.take_profit_1 and not trade.tp1_hit:
                                tp1_hit = True
                            elif trade.take_profit_2 and current_price >= trade.take_profit_2 and not trade.tp2_hit:
                                tp2_hit = True
                            elif trade.take_profit_3 and current_price >= trade.take_profit_3 and not trade.tp3_hit:
                                tp3_hit = True
                            elif trade.stop_loss and current_price <= trade.stop_loss:
                                sl_hit = True
                        else:  # SHORT
                            if trade.take_profit_1 and current_price <= trade.take_profit_1 and not trade.tp1_hit:
                                tp1_hit = True
                            elif trade.take_profit_2 and current_price <= trade.take_profit_2 and not trade.tp2_hit:
                                tp2_hit = True
                            elif trade.take_profit_3 and current_price <= trade.take_profit_3 and not trade.tp3_hit:
                                tp3_hit = True
                            elif trade.stop_loss and current_price >= trade.stop_loss:
                                sl_hit = True
                        
                        # Handle partial TP closes
                        if tp1_hit and not trade.tp1_hit:
                            amount_to_close = remaining_amount * (prefs.tp1_percent / 100)
                            price_change = current_price - trade.entry_price if trade.direction == 'LONG' else trade.entry_price - current_price
                            pnl_usd = (price_change / trade.entry_price) * (amount_to_close * current_price) * 10  # 10x leverage
                            
                            trade.tp1_hit = True
                            trade.remaining_size = trade.remaining_size - (amount_to_close * current_price)
                            trade.pnl += float(pnl_usd)
                            
                            # Return virtual balance
                            prefs.paper_balance += (amount_to_close * current_price) + pnl_usd
                            
                            db.commit()
                            
                            await bot.send_message(
                                user.telegram_id,
                                f"📝 PAPER TP1 HIT! ({prefs.tp1_percent}% closed)\n\n"
                                f"Symbol: {trade.symbol}\n"
                                f"TP1: ${trade.take_profit_1:.4f}\n"
                                f"Paper PnL: ${pnl_usd:.2f}\n"
                                f"Paper Balance: ${prefs.paper_balance:.2f}"
                            )
                        
                        elif tp2_hit and not trade.tp2_hit:
                            amount_to_close = remaining_amount * (prefs.tp2_percent / 100)
                            price_change = current_price - trade.entry_price if trade.direction == 'LONG' else trade.entry_price - current_price
                            pnl_usd = (price_change / trade.entry_price) * (amount_to_close * current_price) * 10  # 10x leverage
                            
                            trade.tp2_hit = True
                            trade.remaining_size = trade.remaining_size - (amount_to_close * current_price)
                            trade.pnl += float(pnl_usd)
                            
                            prefs.paper_balance += (amount_to_close * current_price) + pnl_usd
                            
                            db.commit()
                            
                            await bot.send_message(
                                user.telegram_id,
                                f"📝 PAPER TP2 HIT! ({prefs.tp2_percent}% closed)\n\n"
                                f"Symbol: {trade.symbol}\n"
                                f"TP2: ${trade.take_profit_2:.4f}\n"
                                f"Paper PnL: ${pnl_usd:.2f}\n"
                                f"Paper Balance: ${prefs.paper_balance:.2f}"
                            )
                        
                        elif tp3_hit and not trade.tp3_hit:
                            price_change = current_price - trade.entry_price if trade.direction == 'LONG' else trade.entry_price - current_price
                            pnl_usd = (price_change / trade.entry_price) * (remaining_amount * current_price) * 10  # 10x leverage
                            
                            trade.tp3_hit = True
                            trade.status = 'closed'
                            trade.exit_price = current_price
                            trade.closed_at = datetime.utcnow()
                            trade.remaining_size = 0
                            trade.pnl += float(pnl_usd)
                            # Calculate PnL percent (PnL already includes 10x leverage from line 196)
                            trade.pnl_percent = (trade.pnl / trade.position_size) * 100
                            
                            prefs.paper_balance += (remaining_amount * current_price) + pnl_usd
                            
                            db.commit()
                            
                            await bot.send_message(
                                user.telegram_id,
                                f"📝 PAPER TP3 HIT! Position CLOSED\n\n"
                                f"Symbol: {trade.symbol}\n"
                                f"Total Paper PnL: ${trade.pnl:.2f} ({trade.pnl_percent:+.1f}%)\n"
                                f"Paper Balance: ${prefs.paper_balance:.2f}"
                            )
                        
                        elif sl_hit:
                            price_change = current_price - trade.entry_price if trade.direction == 'LONG' else trade.entry_price - current_price
                            pnl_usd = (price_change / trade.entry_price) * (remaining_amount * current_price) * 10  # 10x leverage
                            
                            trade.status = 'closed'
                            trade.exit_price = current_price
                            trade.closed_at = datetime.utcnow()
                            trade.remaining_size = 0
                            trade.pnl += float(pnl_usd)
                            # Calculate PnL percent (PnL already includes 10x leverage from line 220)
                            trade.pnl_percent = (trade.pnl / trade.position_size) * 100
                            
                            prefs.paper_balance += (remaining_amount * current_price) + pnl_usd
                            
                            db.commit()
                            
                            await bot.send_message(
                                user.telegram_id,
                                f"📝 PAPER SL HIT!\n\n"
                                f"Symbol: {trade.symbol}\n"
                                f"Paper PnL: ${trade.pnl:.2f} ({trade.pnl_percent:+.1f}%)\n"
                                f"Paper Balance: ${prefs.paper_balance:.2f}"
                            )
                    
                    except Exception as e:
                        logger.error(f"Error monitoring paper trade {trade.id}: {e}")
                        continue
            except Exception as e:
                logger.error(f"Error in batch price monitoring: {e}")
            
        except Exception as e:
            logger.error(f"Error in paper trading monitor: {e}", exc_info=True)
        finally:
            db.close()
    
    @staticmethod
    async def close_paper_position(position_id: int, reason: str, db: Session):
        """
        Manually close a paper trading position
        
        Args:
            position_id: ID of the PaperTrade to close
            reason: Reason for closing (e.g., "Auto-closed by opposite spot flow signal")
            db: Database session
        """
        try:
            position = db.query(PaperTrade).filter(PaperTrade.id == position_id).first()
            if not position or position.status != 'open':
                logger.warning(f"Position {position_id} not found or already closed")
                return
            
            # Get current price
            from app.services.price_cache import get_cached_price
            current_price = await get_cached_price(position.symbol, 'kucoin')
            
            if not current_price:
                logger.error(f"Failed to get current price for {position.symbol}")
                return
            
            # Get user preferences for leverage
            user = db.query(User).filter(User.id == position.user_id).first()
            if not user or not user.preferences:
                logger.error(f"User {position.user_id} not found")
                return
            
            prefs = user.preferences
            leverage = prefs.user_leverage or 10
            
            # Calculate final PnL
            remaining_size = position.remaining_size if position.remaining_size > 0 else position.position_size
            remaining_amount = remaining_size / current_price
            
            price_change = current_price - position.entry_price if position.direction == 'LONG' else position.entry_price - current_price
            pnl_usd = (price_change / position.entry_price) * (remaining_amount * current_price) * leverage
            
            # Close the position
            position.status = 'closed'
            position.exit_price = current_price
            position.closed_at = datetime.utcnow()
            position.remaining_size = 0
            position.pnl += float(pnl_usd)
            position.pnl_percent = (position.pnl / (position.position_size / leverage)) * 100
            
            # Return remaining capital + PnL to paper balance
            prefs.paper_balance += (remaining_amount * current_price) + pnl_usd
            
            db.commit()
            
            logger.info(f"✅ Closed paper position {position_id}: {position.symbol} {position.direction} at ${current_price:.4f} | PnL: ${pnl_usd:+.2f} | Reason: {reason}")
            
            # Notify user
            from app.services.bot_instance_manager import bot
            if bot:
                try:
                    await bot.send_message(
                        user.telegram_id,
                        f"📝 PAPER POSITION CLOSED\n\n"
                        f"Symbol: {position.symbol} {position.direction}\n"
                        f"Entry: ${position.entry_price:.4f}\n"
                        f"Exit: ${current_price:.4f}\n"
                        f"PnL: ${pnl_usd:+.2f}\n"
                        f"Reason: {reason}\n\n"
                        f"Paper Balance: ${prefs.paper_balance:.2f}"
                    )
                except Exception as e:
                    logger.error(f"Failed to send close notification: {e}")
                    
        except Exception as e:
            logger.error(f"Error closing paper position {position_id}: {e}", exc_info=True)
            raise
