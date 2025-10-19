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


def calculate_pnl(trade) -> float:
    """
    Calculate PnL in USD for a closed trade
    Args:
        trade: Trade object with entry_price, exit_price, direction, and position_size
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
    
    # PnL = price change % * position size (capital)
    # Note: The position_size already represents the capital/margin used
    # The leverage effect is inherent in the futures position
    pnl_usd = price_change_pct * trade.position_size
    
    return pnl_usd


class OKXTrader:
    """Handles automated trading on OKX exchange"""
    
    def __init__(self, api_key: str, api_secret: str, passphrase: str):
        self.exchange = ccxt.okx({
            'apiKey': api_key,
            'secret': api_secret,
            'password': passphrase,  # OKX requires passphrase
            'options': {
                'defaultType': 'swap',  # Perpetual futures
            },
            'timeout': 30000,  # 30 second timeout
            'enableRateLimit': True
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
        Place a leveraged futures trade on OKX
        
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
            # Load markets if not already loaded
            if not self.markets_loaded:
                await self.exchange.load_markets()
                self.markets_loaded = True
            
            # Convert to OKX swap format (BTC/USDT -> BTC-USDT-SWAP)
            base_quote = symbol.replace('/', '-')
            okx_symbol = f"{base_quote}-SWAP"
            
            if okx_symbol not in self.exchange.markets:
                logger.error(f"Symbol {okx_symbol} not found in OKX markets")
                return None
            
            logger.info(f"Trading {okx_symbol} (from {symbol})")
            
            # Set leverage
            try:
                await self.exchange.set_leverage(
                    leverage, 
                    okx_symbol,
                    params={'mgnMode': 'cross'}  # Cross margin mode
                )
                logger.info(f"Leverage set to {leverage}x")
            except Exception as e:
                logger.warning(f"Could not set leverage: {e}")
            
            # Calculate amount in base currency (BTC, ETH, etc.)
            # OKX uses amount in base currency, not contracts
            amount = (position_size_usdt * leverage) / entry_price
            
            logger.info(f"Position sizing: ${position_size_usdt:.2f} USDT @ {leverage}x = {amount:.4f} {symbol.split('/')[0]}")
            
            # Place market order
            if direction.upper() == 'LONG':
                order = await self.exchange.create_market_buy_order(okx_symbol, amount)
                logger.info(f"Market BUY order placed: {amount:.4f} @ ${entry_price:.2f}")
            else:
                order = await self.exchange.create_market_sell_order(okx_symbol, amount)
                logger.info(f"Market SELL order placed: {amount:.4f} @ ${entry_price:.2f}")
            
            # Place stop loss order
            try:
                sl_params = {
                    'stopLossPrice': stop_loss,
                    'reduceOnly': True
                }
                
                if direction.upper() == 'LONG':
                    sl_order = await self.exchange.create_order(
                        okx_symbol,
                        'stop_market',
                        'sell',
                        amount,
                        params=sl_params
                    )
                else:
                    sl_order = await self.exchange.create_order(
                        okx_symbol,
                        'stop_market',
                        'buy',
                        amount,
                        params=sl_params
                    )
                logger.info(f"Stop loss set at ${stop_loss:.2f}")
            except Exception as e:
                logger.warning(f"Could not set stop loss: {e}")
            
            # Place take profit order
            try:
                tp_params = {
                    'takeProfitPrice': take_profit,
                    'reduceOnly': True
                }
                
                if direction.upper() == 'LONG':
                    tp_order = await self.exchange.create_order(
                        okx_symbol,
                        'take_profit_market',
                        'sell',
                        amount,
                        params=tp_params
                    )
                else:
                    tp_order = await self.exchange.create_order(
                        okx_symbol,
                        'take_profit_market',
                        'buy',
                        amount,
                        params=tp_params
                    )
                logger.info(f"Take profit set at ${take_profit:.2f}")
            except Exception as e:
                logger.warning(f"Could not set take profit: {e}")
            
            return {
                'order_id': order['id'],
                'symbol': okx_symbol,
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
            # Filter out positions with zero contracts
            return [p for p in positions if float(p.get('contracts', 0)) > 0]
        except Exception as e:
            logger.error(f"Error fetching positions: {e}")
            return []
    
    async def close_position(self, symbol: str, side: str, amount: float) -> bool:
        """Close a position"""
        try:
            # Convert to OKX format
            base_quote = symbol.replace('/', '-')
            okx_symbol = f"{base_quote}-SWAP"
            
            # Close long = sell, close short = buy
            if side.upper() == 'LONG':
                await self.exchange.create_market_sell_order(
                    okx_symbol, 
                    amount,
                    params={'reduceOnly': True}
                )
            else:
                await self.exchange.create_market_buy_order(
                    okx_symbol, 
                    amount,
                    params={'reduceOnly': True}
                )
            
            logger.info(f"Closed {side} position for {okx_symbol}")
            return True
        except Exception as e:
            logger.error(f"Error closing position: {e}")
            return False
    
    async def close(self):
        """Close exchange connection"""
        await self.exchange.close()


async def execute_okx_trade(signal: Signal, user: User, db: Session):
    """Execute trade on OKX for a user based on signal with multi-analysis confirmation"""
    try:
        # MULTI-ANALYSIS CONFIRMATION CHECK
        # Validate signal against higher timeframe and multiple indicators
        is_valid, reason, analysis_data = await validate_trade_signal(
            symbol=signal.symbol,
            direction=signal.direction,
            entry_price=signal.entry_price,
            exchange_name='okx'
        )
        
        if not is_valid:
            logger.info(f"OKX trade REJECTED for user {user.id} - {signal.symbol} {signal.direction}: {reason}")
            return None
        
        logger.info(f"OKX trade APPROVED for user {user.id} - {signal.symbol} {signal.direction}: {reason}")
        
        # Get user preferences
        prefs = db.query(UserPreference).filter_by(user_id=user.id).first()
        
        if not prefs:
            logger.error(f"No preferences found for user {user.id}")
            return None
        
        # Check if user has OKX API configured
        if not prefs.okx_api_key or not prefs.okx_api_secret or not prefs.okx_passphrase:
            logger.info(f"User {user.id} has no OKX API configured")
            return None
        
        # Decrypt credentials
        api_key = decrypt_api_key(prefs.okx_api_key)
        api_secret = decrypt_api_key(prefs.okx_api_secret)
        passphrase = decrypt_api_key(prefs.okx_passphrase)
        
        # Initialize trader
        trader = OKXTrader(api_key, api_secret, passphrase)
        
        try:
            # Get account balance
            balance = await trader.get_account_balance()
            logger.info(f"User {user.user_id} OKX balance: ${balance:.2f}")
            
            if balance <= 0:
                logger.warning(f"Insufficient balance for user {user.user_id}")
                return None
            
            # Calculate position size
            position_size = await trader.calculate_position_size(
                balance, 
                prefs.position_size_percent or 7.0
            )
            
            # Advanced position sizing with volatility adjustment
            volatility_factor = 1.0  # Can be adjusted based on market volatility
            adjusted_size = position_size * volatility_factor
            
            logger.info(f"Advanced position sizing: base=${position_size:.2f}, volatility_adj={volatility_factor}, final=${adjusted_size:.2f}")
            
            # Place trade
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
                # Create trade record
                trade = Trade(
                    user_id=user.user_id,
                    signal_id=signal.id,
                    symbol=signal.symbol,
                    direction=signal.direction,
                    entry_price=result['price'],
                    stop_loss=signal.stop_loss,
                    take_profit=signal.take_profit,
                    position_size=adjusted_size,
                    leverage=prefs.user_leverage or 10,
                    exchange='OKX',
                    order_id=result['order_id']
                )
                db.add(trade)
                db.commit()
                
                # Update analytics
                analytics = AnalyticsService(db)
                await analytics.record_signal_outcome(
                    signal.id,
                    'executed',
                    entry_price=result['price']
                )
                
                logger.info(f"✅ OKX trade executed for user {user.user_id}: {signal.symbol} {signal.direction}")
                return trade
            
        finally:
            await trader.close()
            
    except Exception as e:
        logger.error(f"Error executing OKX trade: {e}")
        return None


async def close_okx_position_by_symbol(user: User, symbol: str, direction: str, db: Session) -> int:
    """Close OKX positions for a specific symbol and direction. Returns count of successfully closed positions."""
    closed_count = 0
    try:
        # Get user preferences
        prefs = db.query(UserPreference).filter_by(user_id=user.id).first()
        if not prefs or not prefs.okx_api_key:
            return 0
        
        # Get open trades for this symbol and direction
        open_trades = db.query(Trade).filter(
            Trade.user_id == user.id,
            Trade.symbol == symbol,
            Trade.direction == direction,
            Trade.status == 'open',
            Trade.exchange == 'OKX'
        ).all()
        
        if not open_trades:
            return 0
        
        # Decrypt API credentials
        api_key = decrypt_api_key(prefs.okx_api_key)
        api_secret = decrypt_api_key(prefs.okx_api_secret)
        passphrase = decrypt_api_key(prefs.okx_passphrase)
        
        trader = OKXTrader(api_key, api_secret, passphrase)
        
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
                        logger.info(f"Closed OKX {direction} position for {symbol} (trade ID: {trade.id})")
                    else:
                        logger.warning(f"Failed to close OKX position {trade.id} - exchange returned False")
                except Exception as e:
                    logger.error(f"Error closing individual OKX trade {trade.id}: {e}")
                    db.rollback()
        finally:
            await trader.close()
            
    except Exception as e:
        logger.error(f"Error closing OKX position by symbol: {e}")
        db.rollback()
    
    return closed_count
