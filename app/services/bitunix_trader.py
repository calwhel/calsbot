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
    
    def _get_headers(self, params: dict = None) -> dict:
        """Generate authenticated headers for Bitunix API requests"""
        from datetime import datetime
        import json
        
        # Generate 32-character hex nonce
        nonce = os.urandom(16).hex()
        
        # Bitunix requires YmdHis format timestamp
        timestamp = datetime.utcnow().strftime('%Y%m%d%H%M%S')
        
        # Format params for signature (name1value1name2value2)
        query_params_for_signature = ""
        if params:
            for key, value in sorted(params.items()):
                query_params_for_signature += f"{key}{value}"
        
        # Generate signature
        signature = self._generate_signature(nonce, timestamp, query_params_for_signature, "")
        
        return {
            'api-key': self.api_key,
            'nonce': nonce,
            'timestamp': timestamp,
            'sign': signature,
            'Content-Type': 'application/json'
        }
    
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
            
            # 🔍 Log FULL response for debugging
            logger.info(f"🔍 Bitunix balance API: status={response.status_code}")
            logger.info(f"🔍 Bitunix balance FULL response: {response.text[:500]}")
            
            if response.status_code == 200:
                data = response.json()
                
                # Bitunix uses integer code 0 for success (not string '0')
                if data.get('code') == 0:
                    account_data = data.get('data', {})
                    logger.info(f"Bitunix account data: {account_data}")
                    
                    # The response is a single object, not an array
                    if account_data.get('marginCoin') == 'USDT':
                        # "available" field contains the available balance
                        # Try both "available" and "availableBalance" fields
                        available = float(account_data.get('available') or account_data.get('availableBalance') or 0)
                        logger.info(f"✅ Bitunix USDT balance: ${available:.2f}")
                        logger.info(f"Full balance data: available={account_data.get('available')}, availableBalance={account_data.get('availableBalance')}, total={account_data.get('total')}")
                        return available
                    else:
                        logger.warning(f"Expected USDT but got {account_data.get('marginCoin')}")
                else:
                    error_msg = data.get('msg', 'Unknown error')
                    error_code = data.get('code')
                    # Common Bitunix error codes
                    if 'IP' in str(error_msg).upper() or error_code == 40018:
                        logger.error(f"🚫 BITUNIX IP ERROR: User needs to clear 'Bind IP Address' in API settings! Error: {error_msg}")
                    elif 'sign' in str(error_msg).lower() or 'signature' in str(error_msg).lower():
                        logger.error(f"🔐 BITUNIX SIGNATURE ERROR: API key/secret mismatch. Error: {error_msg}")
                    else:
                        logger.error(f"❌ Bitunix API error code={error_code}: {error_msg}")
            else:
                logger.error(f"❌ Bitunix API HTTP {response.status_code}: {response.text[:200]}")
            
            return 0.0
        except Exception as e:
            logger.error(f"❌ Error fetching Bitunix balance: {e}", exc_info=True)
            return 0.0
    
    async def get_current_price(self, symbol: str) -> Optional[float]:
        """Get current market price for a symbol"""
        try:
            # Bitunix uses symbol without slash (e.g., BTCUSDT)
            bitunix_symbol = symbol.replace('/', '')
            
            # Public endpoint - no authentication required
            # Correct endpoint: /api/v1/futures/market/tickers with symbols parameter
            response = await self.client.get(
                f"{self.base_url}/api/v1/futures/market/tickers",
                params={'symbols': bitunix_symbol}
            )
            
            if response.status_code == 200:
                data = response.json()
                
                if data.get('code') == 0:
                    # Response is a list of tickers
                    ticker_list = data.get('data', [])
                    
                    if ticker_list and len(ticker_list) > 0:
                        ticker_data = ticker_list[0]
                        # Bitunix returns 'lastPrice' as the current price (not 'last')
                        last_price = float(ticker_data.get('lastPrice', 0))
                        
                        if last_price > 0:
                            logger.info(f"Bitunix price for {symbol}: ${last_price}")
                            return last_price
                        else:
                            logger.error(f"Invalid price for {symbol}: {last_price}, ticker_data: {ticker_data}")
                    else:
                        logger.error(f"No ticker data returned for {symbol}")
                else:
                    logger.error(f"Bitunix price API error: {data.get('msg')}")
            else:
                logger.error(f"Bitunix price API returned status {response.status_code}: {response.text}")
            
            return None
        except Exception as e:
            logger.error(f"Error fetching Bitunix price for {symbol}: {e}")
            return None
    
    async def get_open_positions(self) -> list:
        """Get all open positions from Bitunix with detailed PnL data"""
        try:
            nonce = os.urandom(16).hex()
            from datetime import datetime
            timestamp = datetime.utcnow().strftime('%Y%m%d%H%M%S')
            
            query_params = "marginCoinUSDT"
            signature = self._generate_signature(nonce, timestamp, query_params, "")
            
            headers = {
                'api-key': self.api_key,
                'nonce': nonce,
                'timestamp': timestamp,
                'sign': signature,
                'Content-Type': 'application/json'
            }
            
            response = await self.client.get(
                f"{self.base_url}/api/v1/futures/position/all_position",
                headers=headers,
                params={'marginCoin': 'USDT'}
            )
            
            if response.status_code == 200:
                data = response.json()
                
                if data.get('code') == 0:
                    positions = data.get('data', [])
                    
                    open_positions = []
                    for pos in positions:
                        if float(pos.get('total', 0)) > 0:
                            open_positions.append({
                                'symbol': pos.get('symbol'),
                                'hold_side': pos.get('holdSide'),
                                'total': float(pos.get('total', 0)),
                                'available': float(pos.get('available', 0)),
                                'unrealized_pl': float(pos.get('unrealizedPL', 0)),
                                'realized_pl': float(pos.get('realizedPL', 0) if pos.get('realizedPL') else 0),
                                'entry_price': float(pos.get('openPriceAvg', 0)),
                                'mark_price': float(pos.get('markPrice', 0)),
                                'leverage': float(pos.get('leverage', 1))
                            })
                    
                    logger.info(f"Bitunix has {len(open_positions)} open positions")
                    return open_positions
                else:
                    logger.error(f"Bitunix position fetch error: {data.get('msg')}")
            else:
                logger.error(f"Bitunix position API returned status {response.status_code}: {response.text}")
            
            return []
        except Exception as e:
            logger.error(f"Error fetching Bitunix positions: {e}", exc_info=True)
            return []
    
    async def get_position_detail(self, symbol: str) -> Optional[Dict]:
        """
        Get detailed position information for a specific symbol from Bitunix API.
        Returns exchange-reported PnL, entry price, and other real-time data.
        """
        try:
            all_positions = await self.get_open_positions()
            
            # Format symbol for Bitunix (no slash)
            bitunix_symbol = symbol.replace('/', '')
            
            # Find matching position
            for pos in all_positions:
                if pos['symbol'] == bitunix_symbol:
                    return pos
            
            logger.warning(f"No open position found for {symbol} on Bitunix")
            return None
            
        except Exception as e:
            logger.error(f"Error fetching position detail for {symbol}: {e}", exc_info=True)
            return None
    
    async def calculate_position_size(self, balance: float, position_size_percent: float) -> float:
        """Calculate position size based on account balance and percentage"""
        return (balance * position_size_percent) / 100
    
    async def set_leverage(self, symbol: str, leverage: int) -> bool:
        """Set leverage for a symbol before trading"""
        try:
            from datetime import datetime
            nonce = os.urandom(16).hex()
            timestamp = datetime.utcnow().strftime('%Y%m%d%H%M%S')
            
            # Remove slash from symbol for Bitunix format
            bitunix_symbol = symbol.replace('/', '')
            
            payload = {
                'symbol': bitunix_symbol,
                'leverage': leverage,
                'marginCoin': 'USDT'
            }
            
            import json
            body = json.dumps(payload, separators=(',', ':'))
            
            signature = self._generate_signature(nonce, timestamp, "", body)
            
            headers = {
                'api-key': self.api_key,
                'nonce': nonce,
                'timestamp': timestamp,
                'sign': signature,
                'Content-Type': 'application/json'
            }
            
            response = await self.client.post(
                f"{self.base_url}/api/v1/futures/account/change_leverage",
                headers=headers,
                data=body
            )
            
            if response.status_code == 200:
                data = response.json()
                if data.get('code') == 0:
                    logger.info(f"Bitunix leverage set to {leverage}x for {symbol}")
                    return True
                else:
                    logger.error(f"Failed to set Bitunix leverage: {data.get('msg')}")
                    return False
            else:
                logger.error(f"Bitunix leverage API returned {response.status_code}: {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"Error setting Bitunix leverage: {e}", exc_info=True)
            return False
    
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
            # CRITICAL: Set leverage BEFORE placing order
            leverage_set = await self.set_leverage(symbol, leverage)
            if not leverage_set:
                logger.error(f"Failed to set leverage to {leverage}x for {symbol} - aborting trade")
                return None
            
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
    
    async def update_position_stop_loss(self, symbol: str, new_stop_loss: float, direction: str) -> bool:
        """Update stop loss on an open position"""
        try:
            bitunix_symbol = symbol.replace('/', '')
            
            # Bitunix API expects specific hold_side format
            hold_side = 'long' if direction.upper() == 'LONG' else 'short'
            
            order_params = {
                'symbol': bitunix_symbol,
                'holdSide': hold_side,
                'slPrice': str(new_stop_loss),
                'slStopType': 'MARK',
                'slOrderType': 'MARKET'
            }
            
            # Generate signature for POST with JSON body
            import json
            from datetime import datetime
            nonce = os.urandom(16).hex()
            timestamp = datetime.utcnow().strftime('%Y%m%d%H%M%S')
            body = json.dumps(order_params, separators=(',', ':'))
            
            signature = self._generate_signature(nonce, timestamp, "", body)
            
            headers = {
                'api-key': self.api_key,
                'nonce': nonce,
                'timestamp': timestamp,
                'sign': signature,
                'Content-Type': 'application/json'
            }
            
            response = await self.client.post(
                f"{self.base_url}/api/v1/futures/position/modify_sl_tp",
                headers=headers,
                data=body
            )
            
            if response.status_code == 200:
                data = response.json()
                if data.get('code') == 0:
                    logger.info(f"✅ Bitunix SL updated for {symbol} {direction}: ${new_stop_loss:.6f}")
                    return True
                else:
                    logger.error(f"Bitunix SL update error: {data.get('msg')}")
                    return False
            else:
                logger.error(f"Bitunix SL update HTTP error: {response.status_code}")
                return False
            
        except Exception as e:
            logger.error(f"Error updating Bitunix SL: {e}", exc_info=True)
            return False
    
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


async def execute_bitunix_trade(signal: Signal, user: User, db: Session, trade_type: str = 'STANDARD', leverage_override: Optional[int] = None):
    """Execute trade on Bitunix for a user based on signal
    
    Args:
        signal: Trading signal
        user: User object
        db: Database session
        trade_type: Type of trade ('STANDARD', 'TOP_GAINER', 'NEWS')
        leverage_override: Override user leverage (e.g., 5 for top gainers)
    
    For TOP_GAINER trades with high leverage (>10x), TP/SL are automatically capped
    at 80% profit/loss to prevent excessive risk.
    
    🎯 MASTER TRADER INTEGRATION:
    Also executes signal on master Copy Trading account in parallel (transparent to user).
    """
    try:
        # 🎯 EXECUTE ON MASTER ACCOUNT (PARALLEL - doesn't block user trades)
        from app.services.master_trader import get_master_trader
        import asyncio
        
        async def execute_master_trade():
            """Execute trade on master Copy Trading account"""
            try:
                master = await get_master_trader()
                if master.enabled:
                    # Use same leverage as signal or override
                    master_leverage = leverage_override if leverage_override else 5
                    
                    await master.execute_signal_on_master(
                        symbol=signal.symbol,
                        direction=signal.direction,
                        entry_price=signal.entry_price,
                        stop_loss=signal.stop_loss,
                        take_profit_1=signal.take_profit_1 if hasattr(signal, 'take_profit_1') else signal.take_profit,
                        take_profit_2=signal.take_profit_2 if hasattr(signal, 'take_profit_2') else None,
                        take_profit_3=signal.take_profit_3 if hasattr(signal, 'take_profit_3') else None,
                        leverage=master_leverage,
                        position_size_percent=10.0  # 10% of master account balance
                    )
            except Exception as e:
                logger.error(f"Master trade execution failed (non-blocking): {e}")
        
        # Start master trade execution (non-blocking)
        asyncio.create_task(execute_master_trade())
    except Exception as e:
        logger.error(f"Failed to start master trade task: {e}")
        
    try:
        # Skip validation for signals already validated during generation
        # - TEST: Admin test signals
        # - technical: Swing strategy (already validated with multi-timeframe confirmation)
        # - REVERSAL: Reversal patterns (already validated during pattern detection)
        # - DAY_TRADE: Day trading signals (already validated with 5-point confirmation)
        # - TOP_GAINER: Top gainer signals (already validated during generation)
        pre_validated_types = ['TEST', 'technical', 'REVERSAL', 'DAY_TRADE', 'TOP_GAINER']
        
        if signal.signal_type not in pre_validated_types:
            logger.info(f"Running validation for {signal.signal_type} signal")
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
            logger.info(f"Bitunix {signal.signal_type} signal for user {user.id} - skipping validation (pre-validated)")
        
        prefs = db.query(UserPreference).filter_by(user_id=user.id).first()
        
        if not prefs:
            logger.error(f"No preferences found for user {user.id}")
            return None
        
        if not prefs.bitunix_api_key or not prefs.bitunix_api_secret:
            logger.info(f"User {user.id} has no Bitunix API configured")
            return None
        
        try:
            api_key = decrypt_api_key(prefs.bitunix_api_key)
            api_secret = decrypt_api_key(prefs.bitunix_api_secret)
            # Debug: Show key preview (first 8 + last 4 chars only)
            key_preview = f"{api_key[:8]}...{api_key[-4:]}" if len(api_key) > 12 else "TOO_SHORT"
            logger.info(f"🔑 User {user.id} API key preview: {key_preview} (len={len(api_key)})")
        except Exception as decrypt_err:
            logger.error(f"❌ DECRYPTION FAILED for user {user.id}: {decrypt_err} - Check ENCRYPTION_KEY matches!")
            return None
        
        if not api_key or not api_secret or len(api_key) < 10:
            logger.error(f"❌ Invalid decrypted keys for user {user.id} (key_len={len(api_key) if api_key else 0})")
            return None
        
        trader = BitunixTrader(api_key, api_secret)
        
        try:
            balance = await trader.get_account_balance()
            logger.info(f"User {user.id} Bitunix balance: ${balance:.2f}")
            
            if balance <= 0:
                logger.warning(f"Insufficient balance for user {user.id}")
                # Track failed trade
                failed_trade = Trade(
                    user_id=user.id,
                    signal_id=signal.id,
                    symbol=signal.symbol,
                    direction=signal.direction,
                    entry_price=signal.entry_price,
                    stop_loss=signal.stop_loss,
                    take_profit=signal.take_profit,
                    status='failed',
                    position_size=0,
                    remaining_size=0,
                    pnl=0,
                    pnl_percent=0,
                    trade_type=trade_type,
                    opened_at=datetime.utcnow()
                )
                db.add(failed_trade)
                db.commit()
                logger.info(f"Failed trade tracked for user {user.id}: Insufficient balance")
                return None
            
            # 🔥 POSITION SIZING: Fixed $ takes priority over percentage
            fixed_dollars = getattr(prefs, 'position_size_dollars', None)
            
            if fixed_dollars and fixed_dollars > 0:
                # User set a fixed dollar amount - use it directly
                position_size = fixed_dollars
                logger.info(f"💵 Using FIXED position size: ${position_size:.2f} (user configured)")
                
                # Verify they have enough balance
                if position_size > balance:
                    logger.warning(f"⚠️ Fixed position ${position_size:.2f} exceeds balance ${balance:.2f} - reducing to 90% of balance")
                    position_size = balance * 0.9
            else:
                # Use percentage-based sizing (original logic)
                position_size = await trader.calculate_position_size(
                    balance, 
                    prefs.position_size_percent or 10.0
                )
            
            # Check minimum position size for Bitunix ($10 USDT minimum)
            BITUNIX_MIN_POSITION = 10.0
            if position_size < BITUNIX_MIN_POSITION:
                logger.warning(f"⚠️ Position size ${position_size:.2f} below Bitunix minimum ${BITUNIX_MIN_POSITION:.2f}")
                return None
            
            # AUTO-COMPOUND: Apply position multiplier for Top Gainer trades (Upgrade #7)
            if trade_type == 'TOP_GAINER' and prefs.top_gainers_auto_compound:
                multiplier = prefs.top_gainers_position_multiplier or 1.0
                if multiplier > 1.0:
                    position_size = position_size * multiplier
                    logger.info(f"🔥 TOP GAINER AUTO-COMPOUND: User {user.id} - Position size multiplied by {multiplier}x (${position_size:.2f})")
            
            # Use leverage override if provided (e.g., 5x for top gainers), otherwise use user preference
            leverage = leverage_override if leverage_override is not None else (prefs.user_leverage or 10)
            
            # For TOP_GAINER trades: Apply 80% profit/loss cap for high leverage
            # This ensures consistent risk management regardless of user leverage
            final_tp1 = signal.take_profit_1 if hasattr(signal, 'take_profit_1') else signal.take_profit
            final_tp2 = signal.take_profit_2 if hasattr(signal, 'take_profit_2') else None
            final_sl = signal.stop_loss
            
            if trade_type == 'TOP_GAINER' and leverage > 10:
                # Import the helper function
                from app.services.top_gainers_signals import calculate_leverage_capped_targets
                
                # Prepare TP list and base SL
                if signal.direction == 'LONG':
                    tp_pcts = [5.0, 10.0] if final_tp2 else [5.0]  # Dual or single TP
                    base_sl_pct = 4.0
                else:  # SHORT
                    tp_pcts = [8.0]  # SHORTS: Single TP
                    base_sl_pct = 4.0
                
                # Calculate capped targets (scales entire ladder proportionally)
                targets = calculate_leverage_capped_targets(
                    entry_price=signal.entry_price,
                    direction=signal.direction,
                    tp_pcts=tp_pcts,
                    base_sl_pct=base_sl_pct,
                    leverage=leverage,
                    max_profit_cap=80.0,
                    max_loss_cap=80.0
                )
                
                # Override TP/SL with capped values
                final_tp1 = targets['tp_prices'][0]
                if len(targets['tp_prices']) > 1:
                    final_tp2 = targets['tp_prices'][1]
                final_sl = targets['sl_price']
                
                logger.info(f"🔒 TOP GAINER leverage cap applied for user {user.id} ({leverage}x): "
                           f"TPs: {targets['tp_profit_pcts']} (scaling: {targets['scaling_factor']:.2f}x), "
                           f"SL: {targets['sl_loss_pct']:.1f}%")
            
            # For signals with dual TPs (LONGS), split into 2 orders: 50% at TP1, 50% at TP2
            has_dual_tp = final_tp2 is not None
            
            if has_dual_tp:
                # DUAL TP: Place 2 separate orders (50% each)
                half_position = position_size / 2
                
                # Order 1: 50% position at TP1 (leverage-capped if applicable)
                result1 = await trader.place_trade(
                    symbol=signal.symbol,
                    direction=signal.direction,
                    entry_price=signal.entry_price,
                    stop_loss=final_sl,
                    take_profit=final_tp1,
                    position_size_usdt=half_position,
                    leverage=leverage
                )
                
                # Order 2: 50% position at TP2 (leverage-capped if applicable)
                result2 = await trader.place_trade(
                    symbol=signal.symbol,
                    direction=signal.direction,
                    entry_price=signal.entry_price,
                    stop_loss=final_sl,
                    take_profit=final_tp2,
                    position_size_usdt=half_position,
                    leverage=leverage
                )
                
                if result1 and result1.get('success') and result2 and result2.get('success'):
                    # Track as single trade with dual TPs (use leverage-capped values!)
                    trade = Trade(
                        user_id=user.id,
                        signal_id=signal.id,
                        symbol=signal.symbol,
                        direction=signal.direction,
                        entry_price=signal.entry_price,
                        stop_loss=final_sl,  # Leverage-capped SL
                        take_profit=final_tp1,  # Backward compatible
                        take_profit_1=final_tp1,  # Leverage-capped TP1
                        take_profit_2=final_tp2,  # Leverage-capped TP2
                        position_size=position_size,
                        remaining_size=position_size,
                        status='open',
                        trade_type=trade_type
                    )
                    db.add(trade)
                    db.commit()
                    
                    logger.info(f"✅ Bitunix DUAL TP trade recorded for user {user.id}: {signal.symbol} {signal.direction} - 2 orders (50% @ TP1: ${signal.take_profit_1:.6f}, 50% @ TP2: ${signal.take_profit_2:.6f})")
                    return trade
                else:
                    logger.error(f"Failed to place dual TP orders for user {user.id}: Order1: {result1}, Order2: {result2}")
                    return None
            else:
                # SINGLE TP: Standard single order (leverage-capped if applicable)
                result = await trader.place_trade(
                    symbol=signal.symbol,
                    direction=signal.direction,
                    entry_price=signal.entry_price,
                    stop_loss=final_sl,
                    take_profit=final_tp1,
                    position_size_usdt=position_size,
                    leverage=leverage
                )
                
                if result and result.get('success'):
                    trade = Trade(
                        user_id=user.id,
                        signal_id=signal.id,
                        symbol=signal.symbol,
                        direction=signal.direction,
                        entry_price=signal.entry_price,
                        stop_loss=final_sl,  # Leverage-capped SL
                        take_profit=final_tp1,  # Leverage-capped TP
                        take_profit_1=final_tp1,  # Leverage-capped TP1
                        take_profit_2=final_tp2 if final_tp2 else None,
                        position_size=position_size,
                        remaining_size=position_size,
                        status='open',
                        trade_type=trade_type
                    )
                    db.add(trade)
                    db.commit()
                    
                    logger.info(f"Bitunix trade recorded for user {user.id}: {signal.symbol} {signal.direction}")
                    return trade
                else:
                    # Track failed trade (margin error, API error, etc.)
                    failed_trade = Trade(
                        user_id=user.id,
                        signal_id=signal.id,
                        symbol=signal.symbol,
                        direction=signal.direction,
                        entry_price=signal.entry_price,
                        stop_loss=signal.stop_loss,
                        take_profit=signal.take_profit,
                        status='failed',
                        position_size=0,
                        remaining_size=0,
                        pnl=0,
                        pnl_percent=0,
                        trade_type=trade_type,
                        opened_at=datetime.utcnow()
                    )
                    db.add(failed_trade)
                    db.commit()
                    logger.info(f"Failed trade tracked for user {user.id}: Bitunix execution failed")
                    return None
            
        finally:
            await trader.close()
            
    except Exception as e:
        logger.error(f"Error executing Bitunix trade for user {user.id}: {e}", exc_info=True)
        return None
