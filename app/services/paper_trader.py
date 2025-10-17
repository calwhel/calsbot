import ccxt.async_support as ccxt
import logging
from typing import Optional
from sqlalchemy.orm import Session
from datetime import datetime
from app.models import User, UserPreference, PaperTrade, Signal
from app.database import SessionLocal

logger = logging.getLogger(__name__)


class PaperTrader:
    """Handles paper trading (virtual trades without real money)"""
    
    @staticmethod
    def execute_paper_trade(user_id: int, signal: Signal, db: Session) -> Optional[PaperTrade]:
        """Execute a paper trade from a signal"""
        try:
            user = db.query(User).filter(User.id == user_id).first()
            if not user or not user.preferences:
                return None
            
            prefs = user.preferences
            
            if not prefs.paper_trading_mode:
                logger.info(f"Paper trading disabled for user {user_id}")
                return None
            
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
            
            # Use ccxt to get current prices
            exchange = ccxt.kucoin()
            
            try:
                for trade in open_trades:
                    try:
                        # Get current price
                        ticker = await exchange.fetch_ticker(trade.symbol)
                        current_price = ticker['last']
                        
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
                            pnl_usd = (price_change / trade.entry_price) * (amount_to_close * current_price)
                            
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
                            pnl_usd = (price_change / trade.entry_price) * (amount_to_close * current_price)
                            
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
                            pnl_usd = (price_change / trade.entry_price) * (remaining_amount * current_price)
                            
                            trade.tp3_hit = True
                            trade.status = 'closed'
                            trade.exit_price = current_price
                            trade.closed_at = datetime.utcnow()
                            trade.remaining_size = 0
                            trade.pnl += float(pnl_usd)
                            trade.pnl_percent = (trade.pnl / (trade.position_size / 10)) * 100
                            
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
                            pnl_usd = (price_change / trade.entry_price) * (remaining_amount * current_price)
                            
                            trade.status = 'closed'
                            trade.exit_price = current_price
                            trade.closed_at = datetime.utcnow()
                            trade.remaining_size = 0
                            trade.pnl += float(pnl_usd)
                            trade.pnl_percent = (trade.pnl / (trade.position_size / 10)) * 100
                            
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
            finally:
                if 'exchange' in locals():
                    await exchange.close()
            
        except Exception as e:
            logger.error(f"Error in paper trading monitor: {e}", exc_info=True)
        finally:
            db.close()
