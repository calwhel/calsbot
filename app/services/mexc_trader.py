import ccxt.async_support as ccxt
import logging
from typing import Optional, Dict
from sqlalchemy.orm import Session
from app.models import User, UserPreference, Trade, Signal
from app.database import SessionLocal
from app.utils.encryption import decrypt_api_key

logger = logging.getLogger(__name__)


class MEXCTrader:
    """Handles automated trading on MEXC exchange"""
    
    def __init__(self, api_key: str, api_secret: str):
        self.exchange = ccxt.mexc({
            'apiKey': api_key,
            'secret': api_secret,
            'options': {
                'defaultType': 'swap',
            }
        })
    
    async def get_account_balance(self) -> float:
        """Get available USDT balance"""
        try:
            balance = await self.exchange.fetch_balance()
            return balance.get('USDT', {}).get('free', 0.0)
        except Exception as e:
            logger.error(f"Error fetching balance: {e}")
            return 0.0
    
    async def calculate_position_size(self, balance: float, position_size_percent: float) -> float:
        """Calculate position size based on account balance and percentage"""
        return (balance * position_size_percent) / 100
    
    async def place_trade(
        self, 
        symbol: str, 
        direction: str, 
        entry_price: float,
        stop_loss: float,
        take_profit: float,
        position_size_usdt: float,
        leverage: int = 10
    ) -> Optional[Dict]:
        """
        Place a leveraged futures trade on MEXC
        
        Args:
            symbol: Trading pair (e.g., 'BTC/USDT:USDT')
            direction: 'LONG' or 'SHORT'
            entry_price: Entry price
            stop_loss: Stop loss price
            take_profit: Take profit price
            position_size_usdt: Position size in USDT
            leverage: Leverage multiplier
        """
        try:
            # Set leverage
            await self.exchange.set_leverage(leverage, symbol)
            
            # Calculate amount to buy/sell
            amount = position_size_usdt / entry_price
            
            # Place market order
            side = 'buy' if direction == 'LONG' else 'sell'
            order = await self.exchange.create_market_order(
                symbol=symbol,
                side=side,
                amount=amount,
                params={'positionSide': direction.lower()}
            )
            
            logger.info(f"Order placed: {order}")
            
            # Place stop loss order
            sl_side = 'sell' if direction == 'LONG' else 'buy'
            stop_order = await self.exchange.create_order(
                symbol=symbol,
                type='STOP_MARKET',
                side=sl_side,
                amount=amount,
                params={
                    'stopPrice': stop_loss,
                    'positionSide': direction.lower(),
                    'reduceOnly': True
                }
            )
            
            logger.info(f"Stop loss placed: {stop_order}")
            
            # Place take profit order
            tp_order = await self.exchange.create_order(
                symbol=symbol,
                type='TAKE_PROFIT_MARKET',
                side=sl_side,
                amount=amount,
                params={
                    'stopPrice': take_profit,
                    'positionSide': direction.lower(),
                    'reduceOnly': True
                }
            )
            
            logger.info(f"Take profit placed: {tp_order}")
            
            return {
                'order': order,
                'stop_loss': stop_order,
                'take_profit': tp_order
            }
            
        except Exception as e:
            logger.error(f"Error placing trade: {e}", exc_info=True)
            return None
    
    async def get_current_price(self, symbol: str) -> Optional[float]:
        """Get current market price for a symbol"""
        try:
            ticker = await self.exchange.fetch_ticker(symbol)
            return ticker.get('last')
        except Exception as e:
            logger.error(f"Error fetching price for {symbol}: {e}")
            return None
    
    async def close(self):
        """Close the exchange connection"""
        await self.exchange.close()


async def check_security_limits(prefs: "UserPreference", balance: float, db: Session, user: User) -> tuple[bool, str]:
    """
    Check all security limits before trading
    Returns: (allowed, reason)
    """
    from datetime import datetime, timedelta
    from app.services.bot import bot
    
    # Emergency stop check
    if prefs.emergency_stop:
        return False, "Emergency stop is active"
    
    # Minimum balance check
    if balance < prefs.min_balance:
        await bot.send_message(user.telegram_id, f"⚠️ Balance ${balance:.2f} below minimum ${prefs.min_balance:.2f}. Auto-trading paused.")
        return False, f"Balance below minimum (${prefs.min_balance})"
    
    # Update peak balance
    if balance > prefs.peak_balance:
        prefs.peak_balance = balance
        db.commit()
    
    # Maximum drawdown check
    if prefs.peak_balance > 0:
        drawdown_percent = ((prefs.peak_balance - balance) / prefs.peak_balance) * 100
        if drawdown_percent > prefs.max_drawdown_percent:
            if not prefs.safety_paused:
                prefs.safety_paused = True
                db.commit()
                await bot.send_message(user.telegram_id, f"🚨 DRAWDOWN LIMIT HIT!\n\nDrawdown: {drawdown_percent:.1f}%\nLimit: {prefs.max_drawdown_percent}%\n\nAuto-trading PAUSED for safety.\n\nTo resume: /security_settings → Toggle Emergency Stop OFF")
            return False, f"Max drawdown exceeded ({drawdown_percent:.1f}%)"
        elif prefs.safety_paused and drawdown_percent <= (prefs.max_drawdown_percent * 0.8):
            # Auto-resume if drawdown recovers to 80% of limit
            prefs.safety_paused = False
            db.commit()
            await bot.send_message(user.telegram_id, f"✅ Drawdown recovered to {drawdown_percent:.1f}%\n\nSafety pause lifted. Auto-trading can resume.")
    
    # Daily loss limit check
    now = datetime.utcnow()
    if not prefs.daily_loss_reset_date or prefs.daily_loss_reset_date.date() < now.date():
        # New day - reset daily tracking and safety pause if it was due to daily loss
        prefs.daily_loss_reset_date = now
        if prefs.safety_paused:
            prefs.safety_paused = False
            await bot.send_message(user.telegram_id, f"✅ Daily loss limit reset!\n\nNew trading day started. Safety pause lifted.")
        db.commit()
    
    # Calculate today's losses
    today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    today_trades = db.query(Trade).filter(
        Trade.user_id == user.id,
        Trade.closed_at >= today_start,
        Trade.status == 'closed'
    ).all()
    
    daily_pnl = sum(t.pnl for t in today_trades)
    if daily_pnl < 0 and abs(daily_pnl) >= prefs.daily_loss_limit:
        if not prefs.safety_paused:
            prefs.safety_paused = True
            db.commit()
            await bot.send_message(user.telegram_id, f"🚨 DAILY LOSS LIMIT HIT!\n\nLoss: ${abs(daily_pnl):.2f}\nLimit: ${prefs.daily_loss_limit:.2f}\n\nAuto-trading PAUSED until tomorrow.")
        return False, f"Daily loss limit exceeded (${abs(daily_pnl):.2f})"
    
    # Consecutive losses check
    if prefs.consecutive_losses >= prefs.max_consecutive_losses:
        # Check if cooldown period has passed
        if prefs.last_loss_time:
            cooldown_end = prefs.last_loss_time + timedelta(minutes=prefs.cooldown_after_loss)
            if now < cooldown_end:
                minutes_left = int((cooldown_end - now).total_seconds() / 60)
                return False, f"Cooldown active ({minutes_left} min left after {prefs.max_consecutive_losses} losses)"
            else:
                # Cooldown passed, reset counter
                prefs.consecutive_losses = 0
                prefs.safety_paused = False
                db.commit()
                await bot.send_message(user.telegram_id, f"✅ Cooldown period ended!\n\nLoss streak reset. Auto-trading can resume.")
    
    # Check safety pause from any source
    if prefs.safety_paused:
        return False, "Trading paused by safety limits"
    
    return True, "All security checks passed"


async def execute_auto_trade(signal_data: dict, user: User, db: Session):
    """Execute auto-trade for a user based on signal"""
    
    prefs = user.preferences
    if not prefs or not prefs.auto_trading_enabled:
        return
    
    if not prefs.mexc_api_key or not prefs.mexc_api_secret:
        logger.warning(f"User {user.telegram_id} has auto-trading enabled but no API keys")
        return
    
    # Check if signal risk level is accepted by user
    signal_risk = signal_data.get('risk_level', 'MEDIUM')
    accepted_risks = [r.strip() for r in prefs.accepted_risk_levels.split(',')]
    
    if signal_risk not in accepted_risks:
        logger.info(f"User {user.telegram_id} skipping {signal_risk} risk signal (only accepts: {accepted_risks})")
        return
    
    # Check max positions limit
    open_positions = db.query(Trade).filter(
        Trade.user_id == user.id,
        Trade.status == 'open'
    ).count()
    
    if open_positions >= prefs.max_positions:
        logger.info(f"User {user.telegram_id} has reached max positions ({prefs.max_positions})")
        return
    
    # Decrypt API keys for use
    api_key = decrypt_api_key(prefs.mexc_api_key)
    api_secret = decrypt_api_key(prefs.mexc_api_secret)
    
    trader = MEXCTrader(api_key, api_secret)
    
    try:
        # Get account balance
        balance = await trader.get_account_balance()
        
        if balance <= 0:
            logger.warning(f"User {user.telegram_id} has no USDT balance")
            return
        
        # Security checks
        allowed, reason = await check_security_limits(prefs, balance, db, user)
        if not allowed:
            logger.info(f"Security check failed for user {user.telegram_id}: {reason}")
            return
        
        # Calculate position size with risk adjustment
        base_position_percent = prefs.position_size_percent
        
        # Risk-based sizing: reduce position size for higher risk signals
        if prefs.risk_based_sizing:
            if signal_risk == 'MEDIUM':
                base_position_percent *= 0.7  # 70% of normal size for medium risk
            # LOW risk uses full position size
        
        position_size = await trader.calculate_position_size(
            balance,
            base_position_percent
        )
        
        logger.info(f"Position size for {signal_risk} risk: {base_position_percent:.1f}% = ${position_size:.2f}")
        
        # Place trade
        result = await trader.place_trade(
            symbol=signal_data['symbol'],
            direction=signal_data['direction'],
            entry_price=signal_data['entry_price'],
            stop_loss=signal_data['stop_loss'],
            take_profit=signal_data['take_profit'],
            position_size_usdt=position_size,
            leverage=10
        )
        
        if result:
            # Create trade record
            trade = Trade(
                user_id=user.id,
                signal_id=None,  # Will be set when signal is saved
                symbol=signal_data['symbol'],
                direction=signal_data['direction'],
                entry_price=signal_data['entry_price'],
                stop_loss=signal_data['stop_loss'],
                take_profit=signal_data['take_profit'],
                position_size=position_size,
                status='open'
            )
            db.add(trade)
            db.commit()
            
            logger.info(f"Auto-trade executed for user {user.telegram_id}: {signal_data['symbol']} {signal_data['direction']}, size: ${position_size:.2f}")
        
    except Exception as e:
        logger.error(f"Error executing auto-trade for user {user.telegram_id}: {e}", exc_info=True)
    
    finally:
        await trader.close()


async def monitor_positions():
    """Monitor open positions and send notifications when TP/SL is hit"""
    from datetime import datetime
    from app.services.bot import bot
    
    db = SessionLocal()
    try:
        # Get all open trades with users who have auto-trading enabled
        open_trades = db.query(Trade).join(User).join(UserPreference).filter(
            Trade.status == 'open',
            UserPreference.auto_trading_enabled == True,
            UserPreference.mexc_api_key.isnot(None)
        ).all()
        
        for trade in open_trades:
            trader = None
            try:
                user = trade.user
                prefs = user.preferences
                
                # Decrypt API keys
                api_key = decrypt_api_key(prefs.mexc_api_key)
                api_secret = decrypt_api_key(prefs.mexc_api_secret)
                
                trader = MEXCTrader(api_key, api_secret)
                
                # Get current price
                current_price = await trader.get_current_price(trade.symbol)
                
                if not current_price:
                    continue
                
                # Check if TP or SL hit
                tp_hit = False
                sl_hit = False
                
                if trade.direction == 'LONG':
                    if current_price >= trade.take_profit:
                        tp_hit = True
                    elif current_price <= trade.stop_loss:
                        sl_hit = True
                else:  # SHORT
                    if current_price <= trade.take_profit:
                        tp_hit = True
                    elif current_price >= trade.stop_loss:
                        sl_hit = True
                
                # Handle TP hit
                if tp_hit:
                    # Calculate PnL - position_size is notional value, leverage already applied
                    price_change = trade.take_profit - trade.entry_price if trade.direction == 'LONG' else trade.entry_price - trade.take_profit
                    price_move_percent = (price_change / trade.entry_price) * 100  # Raw price move
                    pnl_usd = (price_move_percent / 100) * trade.position_size  # Apply to notional
                    # For display: show leveraged return on margin (position_size / leverage)
                    pnl_percent = (pnl_usd / (trade.position_size / 10)) * 100
                    
                    trade.status = 'closed'
                    trade.exit_price = trade.take_profit
                    trade.closed_at = datetime.utcnow()
                    trade.pnl = float(pnl_usd)
                    trade.pnl_percent = float(pnl_percent)
                    
                    # Reset consecutive losses on win
                    prefs.consecutive_losses = 0
                    
                    db.commit()
                    
                    # Send notification
                    await bot.send_message(
                        user.telegram_id,
                        f"🎯 TAKE PROFIT HIT! 🎯\n\n"
                        f"Symbol: {trade.symbol}\n"
                        f"Direction: {trade.direction}\n"
                        f"Entry: ${trade.entry_price:.4f}\n"
                        f"Exit: ${trade.take_profit:.4f}\n"
                        f"Current: ${current_price:.4f}\n\n"
                        f"💰 PnL: ${pnl_usd:.2f} ({pnl_percent:+.1f}%)\n"
                        f"Position Size: ${trade.position_size:.2f}\n"
                        f"Leverage: 10x"
                    )
                    
                    logger.info(f"TP hit for trade {trade.id}: {trade.symbol} {trade.direction}, PnL: ${pnl_usd:.2f}")
                
                # Handle SL hit
                elif sl_hit:
                    # Calculate PnL - position_size is notional value, leverage already applied
                    price_change = trade.stop_loss - trade.entry_price if trade.direction == 'LONG' else trade.entry_price - trade.stop_loss
                    price_move_percent = (price_change / trade.entry_price) * 100  # Raw price move
                    pnl_usd = (price_move_percent / 100) * trade.position_size  # Apply to notional
                    # For display: show leveraged return on margin (position_size / leverage)
                    pnl_percent = (pnl_usd / (trade.position_size / 10)) * 100
                    
                    trade.status = 'closed'
                    trade.exit_price = trade.stop_loss
                    trade.closed_at = datetime.utcnow()
                    trade.pnl = float(pnl_usd)
                    trade.pnl_percent = float(pnl_percent)
                    
                    # Increment consecutive losses
                    prefs.consecutive_losses += 1
                    prefs.last_loss_time = datetime.utcnow()
                    
                    # Check if consecutive loss limit hit
                    if prefs.consecutive_losses >= prefs.max_consecutive_losses:
                        prefs.safety_paused = True
                        await bot.send_message(
                            user.telegram_id,
                            f"🚨 CONSECUTIVE LOSS LIMIT HIT!\n\n"
                            f"Losses in a row: {prefs.consecutive_losses}\n"
                            f"Limit: {prefs.max_consecutive_losses}\n\n"
                            f"Auto-trading PAUSED for {prefs.cooldown_after_loss} minutes."
                        )
                    
                    db.commit()
                    
                    # Send notification
                    await bot.send_message(
                        user.telegram_id,
                        f"🛑 STOP LOSS HIT 🛑\n\n"
                        f"Symbol: {trade.symbol}\n"
                        f"Direction: {trade.direction}\n"
                        f"Entry: ${trade.entry_price:.4f}\n"
                        f"Exit: ${trade.stop_loss:.4f}\n"
                        f"Current: ${current_price:.4f}\n\n"
                        f"💸 PnL: ${pnl_usd:.2f} ({pnl_percent:+.1f}%)\n"
                        f"Position Size: ${trade.position_size:.2f}\n"
                        f"Leverage: 10x\n\n"
                        f"Consecutive losses: {prefs.consecutive_losses}"
                    )
                    
                    logger.info(f"SL hit for trade {trade.id}: {trade.symbol} {trade.direction}, PnL: ${pnl_usd:.2f}")
                
            except Exception as e:
                logger.error(f"Error monitoring trade {trade.id}: {e}", exc_info=True)
            finally:
                if trader:
                    await trader.close()
    
    except Exception as e:
        logger.error(f"Error in position monitor: {e}", exc_info=True)
    
    finally:
        db.close()
