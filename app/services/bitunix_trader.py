import hashlib
import time
import os
import logging
import httpx
from typing import Optional, Dict
from sqlalchemy.orm import Session
from app.models import User, UserPreference, Trade, Signal
from app.utils.encryption import decrypt_api_key
from app.services.analytics import AnalyticsService
from app.services.multi_analysis import validate_trade_signal

logger = logging.getLogger(__name__)


class BitunixTrader:
    """Handles automated trading on Bitunix Futures exchange"""
    
    def __init__(self, api_key: str, api_secret: str):
        self.api_key = api_key
        self.api_secret = api_secret
        self.base_url = "https://fapi.bitunix.com"
        self.client = httpx.AsyncClient(timeout=30.0)
    
    def _generate_signature(self, nonce: str, timestamp: str, query_params: str = "", body: str = "") -> str:
        """Generate Bitunix double SHA256 signature
        
        Bitunix uses: SHA256(SHA256(nonce + timestamp + api-key + queryParams + body) + secretKey)
        """
        # First SHA256 hash
        digest_input = nonce + timestamp + self.api_key + query_params + body
        first_hash = hashlib.sha256(digest_input.encode()).hexdigest()
        
        # Second SHA256 hash with secret key
        signature = hashlib.sha256((first_hash + self.api_secret).encode()).hexdigest()
        
        return signature
    
    async def get_account_balance(self) -> float:
        """Get available USDT balance"""
        try:
            # Generate 32-character hex nonce (required by Bitunix)
            nonce = os.urandom(16).hex()
            
            # Bitunix requires YmdHis format timestamp (e.g., "20241120123045"), NOT milliseconds
            from datetime import datetime
            timestamp = datetime.utcnow().strftime('%Y%m%d%H%M%S')
            
            # Query params must be formatted as "name1value1name2value2" for signature
            margin_coin = "USDT"
            query_params_for_signature = f"marginCoin{margin_coin}"
            body = ""
            
            signature = self._generate_signature(nonce, timestamp, query_params_for_signature, body)
            
            headers = {
                'api-key': self.api_key,
                'nonce': nonce,
                'timestamp': timestamp,
                'sign': signature,
                'Content-Type': 'application/json'
            }
            
            # Actual URL uses standard query param format
            response = await self.client.get(
                f"{self.base_url}/api/v1/futures/account",
                headers=headers,
                params={'marginCoin': margin_coin}
            )
            
            if response.status_code == 200:
                data = response.json()
                logger.info(f"Bitunix API response: {data}")
                
                # Bitunix uses integer code 0 for success (not string '0')
                if data.get('code') == 0:
                    balances = data.get('data', {}).get('balances', [])
                    logger.info(f"Bitunix balances array: {balances}")
                    
                    for balance in balances:
                        logger.info(f"Checking balance: {balance}")
                        # Try multiple possible field names for currency and balance
                        currency = balance.get('currency') or balance.get('marginCoin')
                        available = balance.get('availableBalance') or balance.get('availableMargin') or balance.get('availableAmount') or balance.get('totalAvailableBalance') or 0
                        
                        if currency == 'USDT':
                            balance_value = float(available)
                            logger.info(f"Found USDT balance: {balance_value}")
                            return balance_value
                    
                    logger.warning("No USDT balance found in response")
                else:
                    logger.error(f"Bitunix API returned error code: {data.get('code')}, message: {data.get('msg')}")
            else:
                logger.error(f"Bitunix API returned status {response.status_code}: {response.text}")
            
            return 0.0
        except Exception as e:
            logger.error(f"Error fetching Bitunix balance: {e}", exc_info=True)
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
        Place a leveraged futures trade on Bitunix
        
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
            bitunix_symbol = symbol.replace('/', '')
            
            quantity = (position_size_usdt * leverage) / entry_price
            
            logger.info(f"Bitunix position sizing: ${position_size_usdt:.2f} USDT @ {leverage}x = {quantity:.4f} qty")
            
            order_params = {
                'symbol': bitunix_symbol,
                'side': 'BUY' if direction.upper() == 'LONG' else 'SELL',
                'orderType': 'MARKET',
                'qty': str(quantity),
                'tradeSide': 'OPEN',
                'effect': 'GTC',
                'clientId': f"bot_{int(time.time() * 1000)}"
            }
            
            if take_profit:
                order_params.update({
                    'tpPrice': str(take_profit),
                    'tpStopType': 'MARK',
                    'tpOrderType': 'MARKET'
                })
            
            if stop_loss:
                order_params.update({
                    'slPrice': str(stop_loss),
                    'slStopType': 'MARK',
                    'slOrderType': 'MARKET'
                })
            
            # Generate signature for POST with JSON body
            import json
            from datetime import datetime
            nonce = os.urandom(16).hex()
            timestamp = datetime.utcnow().strftime('%Y%m%d%H%M%S')  # YmdHis format
            body = json.dumps(order_params, separators=(',', ':'))  # No spaces
            
            signature = self._generate_signature(nonce, timestamp, "", body)
            
            headers = {
                'api-key': self.api_key,
                'nonce': nonce,
                'timestamp': timestamp,
                'sign': signature,
                'Content-Type': 'application/json'
            }
            
            response = await self.client.post(
                f"{self.base_url}/api/v1/futures/trade/place_order",
                headers=headers,
                data=body  # Use the same body string
            )
            
            if response.status_code == 200:
                data = response.json()
                if data.get('code') == 0:
                    order_data = data.get('data', {})
                    logger.info(f"Bitunix {direction} order placed: {quantity:.4f} @ ${entry_price:.2f}")
                    return {
                        'success': True,
                        'order_id': order_data.get('orderId'),
                        'symbol': symbol,
                        'direction': direction,
                        'entry_price': entry_price,
                        'position_size': position_size_usdt,
                        'leverage': leverage
                    }
                else:
                    logger.error(f"Bitunix API error: {data.get('msg')}")
                    return None
            else:
                logger.error(f"Bitunix HTTP error: {response.status_code}")
                return None
                
        except Exception as e:
            logger.error(f"Error placing Bitunix trade: {e}")
            return None
    
    async def close_position(self, symbol: str, position_id: str = None) -> bool:
        """Flash close position at market price"""
        try:
            bitunix_symbol = symbol.replace('/', '')
            
            params = {
                'symbol': bitunix_symbol
            }
            
            if position_id:
                params['positionId'] = position_id
            
            headers = self._get_headers(params)
            
            response = await self.client.post(
                f"{self.base_url}/api/v1/futures/position/flash_close",
                headers=headers,
                json=params
            )
            
            if response.status_code == 200:
                data = response.json()
                if data.get('code') == 0:
                    logger.info(f"Bitunix position closed for {symbol}")
                    return True
            
            logger.error(f"Error closing Bitunix position: {response.text}")
            return False
            
        except Exception as e:
            logger.error(f"Error closing Bitunix position: {e}")
            return False
    
    async def close(self):
        """Close HTTP client"""
        await self.client.aclose()


async def execute_bitunix_trade(signal: Signal, user: User, db: Session):
    """Execute trade on Bitunix for a user based on signal with multi-analysis confirmation"""
    try:
        # Skip validation for TEST signals (admin testing)
        if signal.signal_type != 'TEST':
            is_valid, reason, analysis_data = await validate_trade_signal(
                symbol=signal.symbol,
                direction=signal.direction,
                entry_price=signal.entry_price,
                exchange_name='binance'
            )
            
            if not is_valid:
                logger.info(f"Bitunix trade REJECTED for user {user.id} - {signal.symbol} {signal.direction}: {reason}")
                return None
            
            logger.info(f"Bitunix trade APPROVED for user {user.id} - {signal.symbol} {signal.direction}: {reason}")
        else:
            logger.info(f"Bitunix TEST signal for user {user.id} - skipping multi-analysis validation")
        
        prefs = db.query(UserPreference).filter_by(user_id=user.id).first()
        
        if not prefs:
            logger.error(f"No preferences found for user {user.id}")
            return None
        
        if not prefs.bitunix_api_key or not prefs.bitunix_api_secret:
            logger.info(f"User {user.id} has no Bitunix API configured")
            return None
        
        api_key = decrypt_api_key(prefs.bitunix_api_key)
        api_secret = decrypt_api_key(prefs.bitunix_api_secret)
        
        trader = BitunixTrader(api_key, api_secret)
        
        try:
            balance = await trader.get_account_balance()
            logger.info(f"User {user.id} Bitunix balance: ${balance:.2f}")
            
            if balance <= 0:
                logger.warning(f"Insufficient balance for user {user.id}")
                return None
            
            position_size = await trader.calculate_position_size(
                balance, 
                prefs.position_size_percent or 10.0
            )
            
            result = await trader.place_trade(
                symbol=signal.symbol,
                direction=signal.direction,
                entry_price=signal.entry_price,
                stop_loss=signal.stop_loss,
                take_profit=signal.take_profit,
                position_size_usdt=position_size,
                leverage=prefs.user_leverage or 10
            )
            
            if result and result.get('success'):
                trade = Trade(
                    user_id=user.id,
                    signal_id=signal.id,
                    symbol=signal.symbol,
                    direction=signal.direction,
                    entry_price=signal.entry_price,
                    stop_loss=signal.stop_loss,
                    take_profit=signal.take_profit,
                    position_size=position_size,
                    status='open',
                    exchange='bitunix',
                    leverage=prefs.user_leverage or 10
                )
                db.add(trade)
                db.commit()
                
                AnalyticsService.track_trade_opened(db, user.id, signal.id, position_size)
                
                logger.info(f"Bitunix trade recorded for user {user.id}: {signal.symbol} {signal.direction}")
                return trade
            
            return None
            
        finally:
            await trader.close()
            
    except Exception as e:
        logger.error(f"Error executing Bitunix trade for user {user.id}: {e}", exc_info=True)
        return None
