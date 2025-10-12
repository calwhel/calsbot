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
    
    async def close(self):
        """Close the exchange connection"""
        await self.exchange.close()


async def execute_auto_trade(signal_data: dict, user: User, db: Session):
    """Execute auto-trade for a user based on signal"""
    
    prefs = user.preferences
    if not prefs or not prefs.auto_trading_enabled:
        return
    
    if not prefs.mexc_api_key or not prefs.mexc_api_secret:
        logger.warning(f"User {user.telegram_id} has auto-trading enabled but no API keys")
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
        
        # Calculate position size
        position_size = await trader.calculate_position_size(
            balance,
            prefs.position_size_percent
        )
        
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
                status='open'
            )
            db.add(trade)
            db.commit()
            
            logger.info(f"Auto-trade executed for user {user.telegram_id}: {signal_data['symbol']} {signal_data['direction']}")
        
    except Exception as e:
        logger.error(f"Error executing auto-trade for user {user.telegram_id}: {e}", exc_info=True)
    
    finally:
        await trader.close()
