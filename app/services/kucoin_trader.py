import ccxt.async_support as ccxt
import logging
from typing import Optional, Dict
from sqlalchemy.orm import Session
from app.models import User, UserPreference, Trade, Signal
from app.database import SessionLocal
from app.utils.encryption import decrypt_api_key
from app.services.analytics import AnalyticsService
from app.services.multi_analysis import validate_trade_signal

logger = logging.getLogger(__name__)


def calculate_pnl(trade, leverage: int = 10) -> float:
    """
    Calculate PnL in USD for a closed trade
    Args:
        trade: Trade object with entry_price, exit_price, direction, and position_size
        leverage: Leverage used for the trade (default 10x)
    Returns:
        PnL in USD (includes leverage effect)
    """
    if not trade.exit_price or not trade.entry_price:
        return 0.0
    
    # Calculate price change percentage
    if trade.direction == 'LONG':
        price_change_pct = (trade.exit_price - trade.entry_price) / trade.entry_price
    else:  # SHORT
        price_change_pct = (trade.entry_price - trade.exit_price) / trade.entry_price
    
    # PnL = price change % * notional position size
    # Notional = capital * leverage
    pnl_usd = price_change_pct * trade.position_size * leverage
    
    # Calculate PnL percentage
    trade.pnl_percent = (pnl_usd / trade.position_size) * 100
    
    return pnl_usd


class KuCoinTrader:
    """Handles automated trading on KuCoin Futures exchange"""
    
    def __init__(self, api_key: str, api_secret: str, passphrase: str):
        self.exchange = ccxt.kucoinfutures({
            'apiKey': api_key,
            'secret': api_secret,
            'password': passphrase,
            'options': {
                'defaultType': 'swap',
            },
            'timeout': 30000,
            'enableRateLimit': True,
            'adjustForTimeDifference': True
        })
        self.markets_loaded = False
    
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
        Place a leveraged futures trade on KuCoin
        
        Args:
            symbol: Trading pair (e.g., 'BTC/USDT')
            direction: 'LONG' or 'SHORT'
            entry_price: Entry price
            stop_loss: Stop loss price
            take_profit: Take profit price
            position_size_usdt: Position size in USDT
            leverage: Leverage multiplier
        """
        try:
            if not self.markets_loaded:
                await self.exchange.load_markets()
                self.markets_loaded = True
            
            kucoin_symbol = f"{symbol}:USDT"
            
            if kucoin_symbol not in self.exchange.markets:
                logger.error(f"Symbol {kucoin_symbol} not found in KuCoin markets")
                return None
            
            logger.info(f"Trading {kucoin_symbol} (from {symbol})")
            
            amount = (position_size_usdt * leverage) / entry_price
            
            logger.info(f"Position sizing: ${position_size_usdt:.2f} USDT @ {leverage}x = {amount:.4f} contracts")
            
            if direction.upper() == 'LONG':
                order = await self.exchange.create_market_buy_order(
                    kucoin_symbol, 
                    amount,
                    params={'leverage': leverage}
                )
                logger.info(f"Market BUY order placed: {amount:.4f} @ ${entry_price:.2f}")
            else:
                order = await self.exchange.create_market_sell_order(
                    kucoin_symbol, 
                    amount,
                    params={'leverage': leverage}
                )
                logger.info(f"Market SELL order placed: {amount:.4f} @ ${entry_price:.2f}")
            
            try:
                sl_side = 'sell' if direction.upper() == 'LONG' else 'buy'
                sl_order = await self.exchange.create_order(
                    kucoin_symbol,
                    'stop',
                    sl_side,
                    amount,
                    params={
                        'stopPrice': stop_loss,
                        'type': 'market'
                    }
                )
                logger.info(f"Stop loss set at ${stop_loss:.2f}")
            except Exception as e:
                logger.warning(f"Could not set stop loss: {e}")
            
            try:
                tp_side = 'sell' if direction.upper() == 'LONG' else 'buy'
                tp_order = await self.exchange.create_order(
                    kucoin_symbol,
                    'limit',
                    tp_side,
                    amount,
                    take_profit,
                    params={'timeInForce': 'GTC'}
                )
                logger.info(f"Take profit set at ${take_profit:.2f}")
            except Exception as e:
                logger.warning(f"Could not set take profit: {e}")
            
            return {
                'order_id': order['id'],
                'symbol': kucoin_symbol,
                'side': direction,
                'amount': amount,
                'price': entry_price,
                'stop_loss': stop_loss,
                'take_profit': take_profit
            }
            
        except Exception as e:
            logger.error(f"Error placing trade: {e}")
            return None
    
    async def get_open_positions(self) -> list:
        """Get all open positions"""
        try:
            positions = await self.exchange.fetch_positions()
            return [p for p in positions if float(p.get('contracts', 0)) > 0]
        except Exception as e:
            logger.error(f"Error fetching positions: {e}")
            return []
    
    async def close_position(self, symbol: str, side: str, amount: float) -> bool:
        """Close a position"""
        try:
            kucoin_symbol = f"{symbol}:USDT"
            
            if side.upper() == 'LONG':
                await self.exchange.create_market_sell_order(
                    kucoin_symbol, 
                    amount,
                    params={'reduceOnly': True}
                )
            else:
                await self.exchange.create_market_buy_order(
                    kucoin_symbol, 
                    amount,
                    params={'reduceOnly': True}
                )
            
            logger.info(f"Closed {side} position for {kucoin_symbol}")
            return True
        except Exception as e:
            logger.error(f"Error closing position: {e}")
            return False
    
    async def close(self):
        """Close exchange connection"""
        await self.exchange.close()


async def execute_kucoin_trade(signal: Signal, user: User, db: Session):
    """Execute trade on KuCoin for a user based on signal with multi-analysis confirmation"""
    try:
        # MULTI-ANALYSIS CONFIRMATION CHECK
        # Validate signal against higher timeframe and multiple indicators
        is_valid, reason, analysis_data = await validate_trade_signal(
            symbol=signal.symbol,
            direction=signal.direction,
            entry_price=signal.entry_price,
            exchange_name='kucoin'
        )
        
        if not is_valid:
            logger.info(f"KuCoin trade REJECTED for user {user.id} - {signal.symbol} {signal.direction}: {reason}")
            return None
        
        logger.info(f"KuCoin trade APPROVED for user {user.id} - {signal.symbol} {signal.direction}: {reason}")
        
        prefs = db.query(UserPreference).filter_by(user_id=user.id).first()
        
        if not prefs:
            logger.error(f"No preferences found for user {user.id}")
            return None
        
        if not prefs.kucoin_api_key or not prefs.kucoin_api_secret or not prefs.kucoin_passphrase:
            logger.info(f"User {user.id} has no KuCoin API configured")
            return None
        
        api_key = decrypt_api_key(prefs.kucoin_api_key)
        api_secret = decrypt_api_key(prefs.kucoin_api_secret)
        passphrase = decrypt_api_key(prefs.kucoin_passphrase)
        
        trader = KuCoinTrader(api_key, api_secret, passphrase)
        
        try:
            balance = await trader.get_account_balance()
            logger.info(f"User {user.id} KuCoin balance: ${balance:.2f}")
            
            if balance <= 0:
                logger.warning(f"Insufficient balance for user {user.id}")
                return None
            
            position_size = await trader.calculate_position_size(
                balance, 
                prefs.position_size_percent or 7.0
            )
            
            volatility_factor = 1.0
            adjusted_size = position_size * volatility_factor
            
            logger.info(f"Advanced position sizing: base=${position_size:.2f}, volatility_adj={volatility_factor}, final=${adjusted_size:.2f}")
            
            result = await trader.place_trade(
                symbol=signal.symbol,
                direction=signal.direction,
                entry_price=signal.entry_price,
                stop_loss=signal.stop_loss,
                take_profit=signal.take_profit,
                position_size_usdt=adjusted_size,
                leverage=prefs.user_leverage or 10
            )
            
            if result:
                trade = Trade(
                    user_id=user.id,
                    signal_id=signal.id,
                    symbol=signal.symbol,
                    direction=signal.direction,
                    entry_price=result['price'],
                    stop_loss=signal.stop_loss,
                    take_profit=signal.take_profit,
                    position_size=adjusted_size,
                    leverage=prefs.user_leverage or 10,
                    exchange='KuCoin',
                    order_id=result['order_id']
                )
                db.add(trade)
                db.commit()
                
                analytics = AnalyticsService(db)
                await analytics.record_signal_outcome(
                    signal.id,
                    'executed',
                    entry_price=result['price']
                )
                
                logger.info(f"✅ KuCoin trade executed for user {user.id}: {signal.symbol} {signal.direction}")
                return trade
            
        finally:
            await trader.close()
            
    except Exception as e:
        logger.error(f"Error executing KuCoin trade: {e}")
        return None


async def close_position_by_symbol(user: User, symbol: str, direction: str, db: Session) -> int:
    """Close KuCoin positions for a specific symbol and direction. Returns count of successfully closed positions."""
    closed_count = 0
    try:
        # Get user preferences
        prefs = db.query(UserPreference).filter_by(user_id=user.id).first()
        if not prefs or not prefs.kucoin_api_key:
            return 0
        
        # Get open trades for this symbol and direction
        open_trades = db.query(Trade).filter(
            Trade.user_id == user.id,
            Trade.symbol == symbol,
            Trade.direction == direction,
            Trade.status == 'open',
            Trade.exchange == 'KuCoin'
        ).all()
        
        if not open_trades:
            return 0
        
        # Decrypt API credentials
        api_key = decrypt_api_key(prefs.kucoin_api_key)
        api_secret = decrypt_api_key(prefs.kucoin_api_secret)
        passphrase = decrypt_api_key(prefs.kucoin_passphrase)
        
        trader = KuCoinTrader(api_key, api_secret, passphrase)
        
        try:
            # Close each open trade
            for trade in open_trades:
                try:
                    success = await trader.close_position(symbol, direction, trade.position_size)
                    if success:
                        trade.status = 'closed'
                        trade.exit_price = await get_current_price(symbol)
                        trade.pnl = calculate_pnl(trade)
                        db.commit()
                        closed_count += 1
                        logger.info(f"Closed KuCoin {direction} position for {symbol} (trade ID: {trade.id})")
                    else:
                        logger.warning(f"Failed to close KuCoin position {trade.id} - exchange returned False")
                except Exception as e:
                    logger.error(f"Error closing individual KuCoin trade {trade.id}: {e}")
                    db.rollback()
        finally:
            await trader.close()
            
    except Exception as e:
        logger.error(f"Error closing KuCoin position by symbol: {e}")
        db.rollback()
    
    return closed_count
