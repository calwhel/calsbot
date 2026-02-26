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

# Track users who have been notified about expired subscriptions (avoid spam)
_subscription_expiry_notified = set()

# Track recent trade failures to avoid spamming admin (symbol -> last_notified_time)
_trade_failure_notified = {}

class RetryableError(Exception):
    """Error that can be retried (API timeout, connection error, rate limit)"""
    pass

class PermanentError(Exception):
    """Error that should not be retried (invalid keys, no subscription, etc)"""
    pass

async def execute_trade_with_retry(trader, signal, user, position_size, leverage, final_sl, final_tp1, final_tp2, max_retries=3):
    """Execute trade with retry logic for temporary failures
    
    SAFETY: Only retries if NO orders succeeded. If any order succeeds (even partial),
    we return immediately to avoid duplicate positions.
    """
    import asyncio
    
    delays = [5, 15, 30]  # Exponential backoff delays in seconds
    last_error = None
    
    for attempt in range(max_retries):
        try:
            has_dual_tp = final_tp2 is not None
            
            if has_dual_tp:
                half_position = position_size / 2
                
                # Try order 1
                result1 = None
                try:
                    result1 = await trader.place_trade(
                        symbol=signal.symbol,
                        direction=signal.direction,
                        entry_price=signal.entry_price,
                        stop_loss=final_sl,
                        take_profit=final_tp1,
                        position_size_usdt=half_position,
                        leverage=leverage
                    )
                except Exception as e:
                    logger.warning(f"Order 1 failed: {e}")
                
                # If order 1 succeeded, try order 2 but don't retry if it fails
                # (to avoid duplicate order 1)
                result2 = None
                if result1 and result1.get('success'):
                    try:
                        result2 = await trader.place_trade(
                            symbol=signal.symbol,
                            direction=signal.direction,
                            entry_price=signal.entry_price,
                            stop_loss=final_sl,
                            take_profit=final_tp2,
                            position_size_usdt=half_position,
                            leverage=leverage
                        )
                    except Exception as e:
                        logger.warning(f"Order 2 failed (order 1 succeeded): {e}")
                    
                    # Return even if order 2 failed - we have a partial position
                    return {'result1': result1, 'result2': result2, 'dual_tp': True}
                
                # Neither order succeeded - retry
                if not result1 or not result1.get('success'):
                    raise RetryableError("Both orders failed - retrying")
                    
            else:
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
                    return {'result': result, 'dual_tp': False}
                else:
                    raise RetryableError("Order placement failed")
                    
        except (httpx.TimeoutException, httpx.ConnectError, httpx.ReadError) as e:
            last_error = f"Connection error: {e}"
            logger.warning(f"🔄 Retry {attempt + 1}/{max_retries} for user {user.id} {signal.symbol}: {last_error}")
        except RetryableError as e:
            last_error = str(e)
            logger.warning(f"🔄 Retry {attempt + 1}/{max_retries} for user {user.id} {signal.symbol}: {last_error}")
        except Exception as e:
            # Check if it's a retryable API error
            error_str = str(e).lower()
            if any(x in error_str for x in ['timeout', 'connection', 'rate limit', '429', '503', '502']):
                last_error = f"API error: {e}"
                logger.warning(f"🔄 Retry {attempt + 1}/{max_retries} for user {user.id} {signal.symbol}: {last_error}")
            else:
                # Non-retryable error
                raise
        
        # Wait before next retry (except on last attempt)
        if attempt < max_retries - 1:
            delay = delays[min(attempt, len(delays) - 1)]
            logger.info(f"⏳ Waiting {delay}s before retry for user {user.id} {signal.symbol}")
            await asyncio.sleep(delay)
    
    # All retries exhausted
    raise RetryableError(f"Failed after {max_retries} attempts: {last_error}")

async def notify_admin_trade_failure(user: User, signal_symbol: str, reason: str):
    """Send admin notification when a user's trade fails to execute"""
    import asyncio
    from datetime import datetime, timedelta
    
    # Rate limit: Don't spam same failure more than once per 5 minutes
    cache_key = f"{user.id}_{signal_symbol}_{reason}"
    now = datetime.utcnow()
    if cache_key in _trade_failure_notified:
        if now - _trade_failure_notified[cache_key] < timedelta(minutes=5):
            return  # Already notified recently
    _trade_failure_notified[cache_key] = now
    
    try:
        from app.services.bot import bot
        from app.database import SessionLocal
        
        db = SessionLocal()
        try:
            admins = db.query(User).filter(User.is_admin == True).all()
            
            message = (
                f"⚠️ <b>Trade Execution Failed</b>\n\n"
                f"👤 User: @{user.username or user.first_name or user.id}\n"
                f"🪙 Symbol: {signal_symbol}\n"
                f"❌ Reason: {reason}\n"
                f"🕐 Time: {now.strftime('%H:%M:%S UTC')}"
            )
            
            for admin in admins:
                try:
                    await bot.send_message(
                        chat_id=int(admin.telegram_id),
                        text=message,
                        parse_mode="HTML"
                    )
                except Exception as e:
                    logger.error(f"Failed to notify admin {admin.telegram_id}: {e}")
        finally:
            db.close()
    except Exception as e:
        logger.error(f"Failed to send admin trade failure notification: {e}")


class BitunixTrader:
    """Handles automated trading on Bitunix Futures exchange"""
    
    def __init__(self, api_key: str, api_secret: str):
        self.api_key = api_key
        self.api_secret = api_secret
        self.base_url = "https://fapi.bitunix.com"
        self.client = httpx.AsyncClient(timeout=30.0)
        
        # 🔍 DEBUG: Log both key lengths
        key_len = len(api_key) if api_key else 0
        secret_len = len(api_secret) if api_secret else 0
        
        # Bitunix: API Key = 32 chars (hex), Secret = 32 chars
        if key_len != 32:
            logger.warning(f"⚠️ API key unexpected length: {key_len} chars (expected 32)")
        
        if api_key:
            key_preview = f"{api_key[:6]}...{api_key[-4:]}" if len(api_key) > 10 else f"SHORT({len(api_key)})"
            logger.info(f"🔧 BitunixTrader: api_key={key_preview} (len={key_len}), secret_len={secret_len}")
        else:
            logger.error(f"⚠️ BitunixTrader CREATED with EMPTY api_key!")
    
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
    
    def _get_headers(self, params: dict = None, is_post: bool = False) -> dict:
        """Generate authenticated headers for Bitunix API requests
        
        Args:
            params: Query parameters for GET or body for POST
            is_post: If True, use milliseconds timestamp (POST), else YmdHis (GET)
        """
        from datetime import datetime
        import json
        
        # Generate 32-character hex nonce
        nonce = os.urandom(16).hex()
        
        # Bitunix: GET uses YmdHis format, POST uses milliseconds
        if is_post:
            timestamp = str(int(time.time() * 1000))
        else:
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
            from datetime import datetime
            # Generate 32-character hex nonce (required by Bitunix)
            nonce = os.urandom(16).hex()
            
            # Bitunix GET requests use YmdHis format timestamp
            timestamp = datetime.utcnow().strftime('%Y%m%d%H%M%S')
            
            # Query params must be formatted as "name1value1name2value2" for signature
            margin_coin = "USDT"
            query_params_for_signature = f"marginCoin{margin_coin}"
            body = ""
            
            # 🔍 DEBUG: Log signature inputs for verification
            logger.info(f"🔐 SIGN INPUT: nonce={nonce[:8]}..., ts={timestamp}, params='{query_params_for_signature}'")
            
            signature = self._generate_signature(nonce, timestamp, query_params_for_signature, body)
            
            logger.info(f"🔐 SIGNATURE: {signature[:16]}... (len={len(signature)})")
            
            headers = {
                'api-key': self.api_key,
                'nonce': nonce,
                'timestamp': timestamp,
                'sign': signature,
                'Content-Type': 'application/json'
            }
            
            # 🔍 DEBUG: Log exact request details
            logger.info(f"🔍 REQUEST: api-key={self.api_key[:10]}..., ts={timestamp}, nonce={nonce[:8]}...")
            
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
        """Get all open positions from Bitunix with detailed PnL data
        
        Uses get_pending_positions endpoint (new Bitunix API).
        Calls get_position_id() separately when positionId is needed for SL modification.
        """
        try:
            headers = self._get_headers()
            headers['language'] = 'en-US'
            
            response = await self.client.get(
                f"{self.base_url}/api/v1/futures/position/get_pending_positions",
                headers=headers
            )
            
            if response.status_code == 200:
                data = response.json()
                if data is None:
                    logger.error("Bitunix returned null response for positions")
                    return []
                logger.info(f"📡 RAW get_pending_positions response: code={data.get('code')}, positions_count={len(data.get('data') or [])}")
                
                if data.get('code') == 0:
                    positions = data.get('data', [])
                    
                    for p in positions:
                        logger.info(f"   📊 RAW: {p.get('symbol')} | qty={p.get('qty')} | side={p.get('side')} | markPrice={p.get('markPrice')} | avgOpenPrice={p.get('avgOpenPrice')} | unrealizedPNL={p.get('unrealizedPNL')}")
                    
                    open_positions = []
                    for pos in positions:
                        qty = float(pos.get('qty', 0) or pos.get('total', 0))
                        if qty > 0:
                            raw_side = str(pos.get('side', pos.get('holdSide', ''))).upper()
                            if raw_side == 'BUY':
                                raw_side = 'LONG'
                            elif raw_side == 'SELL':
                                raw_side = 'SHORT'
                            
                            def _safe_float(val, default=0.0):
                                try:
                                    return float(val) if val is not None else default
                                except (ValueError, TypeError):
                                    return default
                            
                            entry_px = _safe_float(pos.get('avgOpenPrice') or pos.get('openPriceAvg'))
                            raw_mark = _safe_float(pos.get('markPrice'))
                            unrealized = _safe_float(pos.get('unrealizedPNL') or pos.get('unrealizedPL'))
                            
                            if raw_mark > 0 and raw_mark != entry_px:
                                mark_px = raw_mark
                            elif unrealized != 0 and entry_px > 0 and qty > 0:
                                pnl_per_unit = unrealized / qty
                                if raw_side.lower() == 'short':
                                    mark_px = entry_px - pnl_per_unit
                                else:
                                    mark_px = entry_px + pnl_per_unit
                                logger.info(f"   💡 Calculated mark_price from PnL: ${mark_px:.8f} (entry=${entry_px:.6f}, pnl=${unrealized:.4f}, qty={qty})")
                            else:
                                mark_px = entry_px
                            
                            open_positions.append({
                                'symbol': pos.get('symbol'),
                                'hold_side': raw_side.lower(),
                                'total': qty,
                                'available': _safe_float(pos.get('qty') or pos.get('available')),
                                'unrealized_pl': unrealized,
                                'realized_pl': _safe_float(pos.get('realizedPNL') or pos.get('realizedPL')),
                                'entry_price': entry_px,
                                'mark_price': mark_px,
                                'leverage': _safe_float(pos.get('leverage'), 1),
                                'position_id': pos.get('positionId', ''),
                                'margin': _safe_float(pos.get('margin')),
                                'liq_price': _safe_float(pos.get('liqPrice')),
                            })
                    
                    logger.info(f"Bitunix has {len(open_positions)} open positions")
                    return open_positions
                else:
                    logger.error(f"Bitunix position fetch error: {data.get('msg')}")
                    
                    if data.get('code') == 2 and 'all_position' not in str(response.url):
                        logger.warning("Bitunix API parameter error - check endpoint/signing")
            else:
                logger.error(f"Bitunix position API returned status {response.status_code}: {response.text}")
            
            return []
        except Exception as e:
            logger.error(f"Error fetching Bitunix positions: {e}", exc_info=True)
            return []
    
    async def get_closed_position_history(self, symbol: str) -> Optional[dict]:
        """Fetch closed position history from Bitunix to get actual close price and realized PnL.
        
        Returns the most recent closed position for the symbol, or None if not found.
        Response includes: closePrice, realizedPNL, entryPrice, side, leverage, fee, funding
        """
        try:
            from datetime import datetime
            nonce = os.urandom(16).hex()
            timestamp = datetime.utcnow().strftime('%Y%m%d%H%M%S')
            
            query_params = f"symbol{symbol}"
            signature = self._generate_signature(nonce, timestamp, query_params, "")
            
            headers = {
                'api-key': self.api_key,
                'nonce': nonce,
                'timestamp': timestamp,
                'sign': signature,
                'Content-Type': 'application/json'
            }
            
            response = await self.client.get(
                f"{self.base_url}/api/v1/futures/position/get_history_positions",
                headers=headers,
                params={'symbol': symbol}
            )
            
            if response.status_code == 200:
                data = response.json()
                if data and data.get('code') == 0:
                    positions = data.get('data', {}).get('positionList', [])
                    if positions:
                        positions.sort(key=lambda p: int(p.get('mtime', 0) or p.get('ctime', 0)), reverse=True)
                        most_recent = positions[0]
                        logger.info(f"📜 Bitunix closed position history for {symbol}: closePrice={most_recent.get('closePrice')}, realizedPNL={most_recent.get('realizedPNL')}")
                        return {
                            'close_price': float(most_recent.get('closePrice', 0)),
                            'realized_pnl': float(most_recent.get('realizedPNL', 0)),
                            'entry_price': float(most_recent.get('entryPrice', 0)),
                            'side': most_recent.get('side'),
                            'leverage': float(most_recent.get('leverage', 1)),
                            'fee': float(most_recent.get('fee', 0)),
                            'funding': float(most_recent.get('funding', 0)),
                            'position_id': most_recent.get('positionId'),
                        }
                    else:
                        logger.info(f"No closed position history found for {symbol}")
                else:
                    logger.warning(f"Bitunix history positions error: {data.get('msg') if data else 'null response'}")
            else:
                logger.warning(f"Bitunix history positions API returned {response.status_code}")
            
            return None
        except Exception as e:
            logger.error(f"Error fetching closed position history for {symbol}: {e}")
            return None

    async def get_position_id(self, symbol: str) -> Optional[str]:
        """Get positionId for a symbol using get_pending_positions endpoint
        
        This endpoint returns positionId which is required for modifying position SL.
        """
        try:
            bitunix_symbol = symbol.replace('/', '')
            params = {'symbol': bitunix_symbol}
            headers = self._get_headers(params)
            headers['language'] = 'en-US'
            
            response = await self.client.get(
                f"{self.base_url}/api/v1/futures/position/get_pending_positions",
                headers=headers,
                params=params
            )
            
            if response.status_code == 200:
                data = response.json()
                logger.info(f"📋 get_pending_positions response for {symbol}: {data}")
                
                if data.get('code') == 0:
                    positions = data.get('data', [])
                    if positions:
                        position_id = positions[0].get('positionId')
                        logger.info(f"✅ Found positionId for {symbol}: {position_id}")
                        return position_id
                    else:
                        logger.warning(f"No pending positions found for {symbol}")
                else:
                    logger.error(f"get_pending_positions error: {data.get('msg')}")
            else:
                logger.error(f"get_pending_positions HTTP error: {response.status_code}")
            
            return None
        except Exception as e:
            logger.error(f"Error getting positionId for {symbol}: {e}", exc_info=True)
            return None
    
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
            nonce = os.urandom(16).hex()
            timestamp = str(int(time.time() * 1000))
            
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
    
    async def set_margin_mode(self, symbol: str, margin_mode: str = "ISOLATION") -> bool:
        """Set margin mode for a symbol before trading (ISOLATION or CROSS)"""
        try:
            nonce = os.urandom(16).hex()
            timestamp = str(int(time.time() * 1000))
            
            bitunix_symbol = symbol.replace('/', '')
            
            payload = {
                'symbol': bitunix_symbol,
                'marginMode': margin_mode,
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
                f"{self.base_url}/api/v1/futures/account/change_margin_mode",
                headers=headers,
                data=body
            )
            
            if response.status_code == 200:
                data = response.json()
                if data.get('code') == 0:
                    logger.info(f"Bitunix margin mode set to {margin_mode} for {symbol}")
                    return True
                elif 'already' in str(data.get('msg', '')).lower():
                    logger.info(f"Bitunix margin mode already {margin_mode} for {symbol}")
                    return True
                else:
                    logger.error(f"Failed to set Bitunix margin mode: {data.get('msg')}")
                    return False
            else:
                logger.error(f"Bitunix margin mode API returned {response.status_code}: {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"Error setting Bitunix margin mode: {e}", exc_info=True)
            return False
    
    async def get_max_position_size(self, symbol: str) -> Optional[Dict]:
        """Get max position size limits for a symbol from Bitunix API
        
        Returns dict with maxMarketOrderVolume, minTradeVolume, etc.
        """
        try:
            bitunix_symbol = symbol.replace('/', '')
            
            url = f"{self.base_url}/api/v1/futures/market/trading_pairs"
            params = {"symbols": bitunix_symbol}
            
            response = await self.client.get(url, params=params)
            
            if response.status_code == 200:
                data = response.json()
                if data.get('code') == 0 and data.get('data'):
                    pair_info = data['data'][0]
                    max_market = float(pair_info.get('maxMarketOrderVolume', 100000))
                    min_trade = float(pair_info.get('minTradeVolume', 0.0001))
                    
                    logger.info(f"📊 {symbol} position limits: max={max_market}, min={min_trade}")
                    
                    return {
                        'max_market_order': max_market,
                        'min_trade_volume': min_trade,
                        'max_leverage': int(pair_info.get('maxLeverage', 125))
                    }
            
            logger.warning(f"⚠️ Could not get position limits for {symbol}, using defaults")
            return None
            
        except Exception as e:
            logger.error(f"Error fetching max position size for {symbol}: {e}")
            return None
    
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
            # CRITICAL: Set margin mode to CROSS before placing order
            await self.set_margin_mode(symbol, "CROSS")
            
            # CRITICAL: Set leverage BEFORE placing order
            leverage_set = await self.set_leverage(symbol, leverage)
            if not leverage_set:
                logger.error(f"Failed to set leverage to {leverage}x for {symbol} - aborting trade")
                return None
            
            bitunix_symbol = symbol.replace('/', '')
            
            quantity = (position_size_usdt * leverage) / entry_price
            
            # Check max position size from Bitunix and cap if needed
            limits = await self.get_max_position_size(symbol)
            if limits:
                max_qty = limits['max_market_order']
                min_qty = limits['min_trade_volume']
                
                if quantity > max_qty:
                    logger.warning(f"⚠️ {symbol} quantity {quantity:.4f} exceeds max {max_qty} - CAPPING to max")
                    quantity = max_qty * 0.95  # Use 95% of max to be safe
                
                if quantity < min_qty:
                    logger.error(f"❌ {symbol} quantity {quantity:.8f} below min {min_qty} - trade too small")
                    return None
            
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
            
            # Note: TP is set via separate reduce orders for dual TP support
            # Single TP trades still use tpPrice parameter
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
            nonce = os.urandom(16).hex()
            timestamp = str(int(time.time() * 1000))
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
        """Update stop loss on an open position using holdSide endpoint"""
        try:
            bitunix_symbol = symbol.replace('/', '')
            
            hold_side = 'long' if direction.upper() == 'LONG' else 'short'
            
            order_params = {
                'symbol': bitunix_symbol,
                'holdSide': hold_side,
                'slPrice': str(new_stop_loss),
                'slStopType': 'MARK_PRICE',
                'slOrderType': 'MARKET'
            }
            
            import json
            nonce = os.urandom(16).hex()
            timestamp = str(int(time.time() * 1000))
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
            
            logger.info(f"🔧 Position-level SL update for {symbol}: ${new_stop_loss:.8f}")
            if response.status_code == 200:
                data = response.json()
                logger.info(f"   Position SL API response: {data}")
                if data.get('code') == 0:
                    logger.info(f"✅ Position SL updated for {symbol} {direction}: ${new_stop_loss:.6f}")
                    return True
                else:
                    logger.error(f"❌ Position SL update FAILED for {symbol}: code={data.get('code')}, msg={data.get('msg')}")
                    return False
            else:
                logger.error(f"❌ Position SL update HTTP error: {response.status_code} - {response.text}")
                return False
            
        except Exception as e:
            logger.error(f"Error updating Bitunix SL: {e}", exc_info=True)
            return False
    
    async def modify_position_sl(self, symbol: str, position_id: str, new_sl_price: float, existing_tp_price: float = None) -> bool:
        """Modify position-level SL using the official Bitunix TP/SL API endpoint.
        
        Endpoint: POST /api/v1/futures/tpsl/position/modify_order
        Docs: https://openapidoc.bitunix.com/doc/tp_sl/modify_position_tp_sl_order.html
        """
        try:
            import json
            bitunix_symbol = symbol.replace('/', '')
            
            logger.info(f"🔧 POSITION SL MODIFY: {symbol} | positionId={position_id} | SL=${new_sl_price:.8f} | existingTP={existing_tp_price}")
            
            modify_params = {
                'symbol': bitunix_symbol,
                'positionId': str(position_id),
                'slPrice': f"{new_sl_price:.8f}",
                'slStopType': 'MARK_PRICE'
            }
            
            if existing_tp_price and existing_tp_price > 0:
                modify_params['tpPrice'] = f"{existing_tp_price:.8f}"
                modify_params['tpStopType'] = 'MARK_PRICE'
            
            nonce = os.urandom(16).hex()
            timestamp = str(int(time.time() * 1000))
            body = json.dumps(modify_params, separators=(',', ':'))
            
            signature = self._generate_signature(nonce, timestamp, "", body)
            
            headers = {
                'api-key': self.api_key,
                'nonce': nonce,
                'timestamp': timestamp,
                'sign': signature,
                'Content-Type': 'application/json'
            }
            
            response = await self.client.post(
                f"{self.base_url}/api/v1/futures/tpsl/position/modify_order",
                headers=headers,
                data=body
            )
            
            if response.status_code == 200:
                result = response.json()
                logger.info(f"   Position SL modify response: {result}")
                if result.get('code') == 0:
                    logger.info(f"✅ BREAKEVEN SET: {symbol} SL moved to ${new_sl_price:.6f}")
                    return True
                else:
                    logger.error(f"❌ Position SL modify FAILED: code={result.get('code')}, msg={result.get('msg')}")
                    return False
            else:
                logger.error(f"❌ Position SL modify HTTP error: {response.status_code} - {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"Error modifying position SL: {e}", exc_info=True)
            return False

    async def modify_tpsl_order_sl(self, symbol: str, new_sl_price: float) -> bool:
        """Modify pending TP/SL orders to change the SL price (e.g., to breakeven)
        
        This is used after TP1 hits to move SL to entry price while keeping TP2 intact.
        
        Args:
            symbol: Trading pair (e.g., 'BTC/USDT')
            new_sl_price: New stop loss price (typically entry price for breakeven)
        """
        try:
            import json
            bitunix_symbol = symbol.replace('/', '')
            
            # Step 1: Get pending TP/SL orders
            params = {
                'symbol': bitunix_symbol,
                'limit': '100'
            }
            
            headers = self._get_headers(params)
            response = await self.client.get(
                f"{self.base_url}/api/v1/futures/tpsl/get_pending_orders",
                headers=headers,
                params=params
            )
            
            if response.status_code != 200:
                logger.error(f"Failed to get TP/SL orders: {response.status_code} - {response.text}")
                return False
            
            data = response.json()
            if data.get('code') != 0:
                logger.error(f"TP/SL orders API error: {data.get('msg')}")
                return False
            
            orders = data.get('data', [])
            logger.info(f"📋 Found {len(orders)} pending TP/SL orders for {symbol}")
            
            # Log all orders for debugging
            for i, order in enumerate(orders):
                logger.info(f"   Order {i+1}: id={order.get('id')}, TP=${order.get('tpPrice')}, SL=${order.get('slPrice')}, qty={order.get('slQty')}")
            
            if not orders:
                logger.warning(f"⚠️ No pending TP/SL orders found for {symbol} - trying position-level SL instead")
                return False  # Return False so position-level SL gets used as fallback
            
            # Step 2: Modify each order's SL to the new price
            modified_count = 0
            for order in orders:
                order_id = order.get('id')
                current_sl = order.get('slPrice')
                current_tp = order.get('tpPrice')
                sl_qty = order.get('slQty')
                tp_qty = order.get('tpQty')
                # Preserve existing stop types (don't hardcode - use what exchange already has)
                current_sl_stop_type = order.get('slStopType', 'MARK_PRICE')
                current_tp_stop_type = order.get('tpStopType', 'MARK_PRICE')
                current_sl_order_type = order.get('slOrderType', 'MARKET')
                current_tp_order_type = order.get('tpOrderType', 'MARKET')
                
                if not order_id:
                    continue
                
                logger.info(f"🔧 Modifying TP/SL order {order_id}: SL ${current_sl} → ${new_sl_price:.8f}")
                logger.info(f"   Order details: slStopType={current_sl_stop_type}, tpStopType={current_tp_stop_type}, slQty={sl_qty}, tpQty={tp_qty}")
                
                # Modify the order to update SL while keeping TP intact
                # Use existing stop types from the order (don't override with hardcoded values)
                modify_params = {
                    'orderId': order_id,
                    'slPrice': f"{new_sl_price:.8f}",
                    'slStopType': current_sl_stop_type,
                    'slOrderType': current_sl_order_type
                }
                
                # Keep TP if it exists - preserve all original values
                if current_tp and float(current_tp) > 0:
                    modify_params['tpPrice'] = current_tp
                    modify_params['tpStopType'] = current_tp_stop_type
                    modify_params['tpOrderType'] = current_tp_order_type
                
                # Keep quantities if they exist
                if sl_qty:
                    modify_params['slQty'] = sl_qty
                if tp_qty:
                    modify_params['tpQty'] = tp_qty
                
                nonce = os.urandom(16).hex()
                timestamp = str(int(time.time() * 1000))
                body = json.dumps(modify_params, separators=(',', ':'))
                
                signature = self._generate_signature(nonce, timestamp, "", body)
                
                headers = {
                    'api-key': self.api_key,
                    'nonce': nonce,
                    'timestamp': timestamp,
                    'sign': signature,
                    'Content-Type': 'application/json'
                }
                
                response = await self.client.post(
                    f"{self.base_url}/api/v1/futures/tpsl/modify_order",
                    headers=headers,
                    data=body
                )
                
                if response.status_code == 200:
                    result = response.json()
                    logger.info(f"   Modify API response: {result}")
                    if result.get('code') == 0:
                        logger.info(f"✅ Modified TP/SL order {order_id}: SL now at ${new_sl_price:.6f}")
                        modified_count += 1
                    else:
                        logger.error(f"❌ Modify TP/SL FAILED for {order_id}: code={result.get('code')}, msg={result.get('msg')}")
                else:
                    logger.error(f"❌ Modify TP/SL HTTP error for {order_id}: {response.status_code} - {response.text}")
            
            logger.info(f"✅ Modified {modified_count}/{len(orders)} TP/SL orders for {symbol} - SL moved to ${new_sl_price:.6f}")
            return modified_count > 0 or len(orders) == 0
            
        except Exception as e:
            logger.error(f"Error modifying TP/SL orders: {e}", exc_info=True)
            return False

    async def cancel_and_replace_sl(self, symbol: str, new_sl_price: float, position_id: str = None) -> bool:
        """Place new TP/SL orders with updated SL, then cancel old ones.
        
        Safe order: place FIRST, cancel SECOND - position is never unprotected.
        Uses official documented endpoints:
        - GET /api/v1/futures/tpsl/get_pending_orders
        - POST /api/v1/futures/tpsl/place_order
        - POST /api/v1/futures/tpsl/cancel_order
        """
        try:
            import json
            import asyncio
            bitunix_symbol = symbol.replace('/', '')
            
            logger.info(f"🔄 CANCEL-AND-REPLACE SL: {symbol} | new SL=${new_sl_price:.8f}")
            
            params = {'symbol': bitunix_symbol, 'limit': '100'}
            headers = self._get_headers(params)
            response = await self.client.get(
                f"{self.base_url}/api/v1/futures/tpsl/get_pending_orders",
                headers=headers,
                params=params
            )
            
            if response.status_code != 200:
                logger.error(f"Failed to get TP/SL orders for cancel-replace: {response.status_code}")
                return False
            
            data = response.json()
            if data.get('code') != 0:
                logger.error(f"TP/SL orders API error: {data.get('msg')}")
                return False
            
            orders = data.get('data', [])
            if not orders:
                logger.warning(f"No pending TP/SL orders found for {symbol} to cancel-replace")
                return False
            
            logger.info(f"📋 Found {len(orders)} TP/SL orders to cancel-replace for {symbol}")
            for i, o in enumerate(orders):
                logger.info(f"   Order {i+1}: id={o.get('id')}, positionId={o.get('positionId')}, TP={o.get('tpPrice')}, SL={o.get('slPrice')}, tpQty={o.get('tpQty')}, slQty={o.get('slQty')}")
            
            success_count = 0
            old_order_ids_to_cancel = []
            
            for order in orders:
                order_id = order.get('id')
                current_tp = order.get('tpPrice')
                current_tp_stop_type = order.get('tpStopType', 'MARK_PRICE')
                current_tp_order_type = order.get('tpOrderType', 'MARKET')
                tp_qty = order.get('tpQty')
                sl_qty = order.get('slQty')
                order_position_id = order.get('positionId') or position_id
                
                if not order_id:
                    continue
                
                if not order_position_id:
                    logger.warning(f"   No positionId for order {order_id} - cannot place replacement")
                    continue
                
                place_params = {
                    'symbol': bitunix_symbol,
                    'positionId': str(order_position_id),
                    'slPrice': f"{new_sl_price:.8f}",
                    'slStopType': 'MARK_PRICE',
                    'slOrderType': 'MARKET'
                }
                
                if current_tp and float(current_tp) > 0:
                    place_params['tpPrice'] = str(current_tp)
                    place_params['tpStopType'] = current_tp_stop_type
                    place_params['tpOrderType'] = current_tp_order_type
                
                if sl_qty:
                    place_params['slQty'] = str(sl_qty)
                if tp_qty:
                    place_params['tpQty'] = str(tp_qty)
                
                placed = False
                for attempt in range(3):
                    nonce = os.urandom(16).hex()
                    timestamp = str(int(time.time() * 1000))
                    body = json.dumps(place_params, separators=(',', ':'))
                    signature = self._generate_signature(nonce, timestamp, "", body)
                    
                    place_headers = {
                        'api-key': self.api_key,
                        'nonce': nonce,
                        'timestamp': timestamp,
                        'sign': signature,
                        'Content-Type': 'application/json'
                    }
                    
                    logger.info(f"   Placing new TP/SL (attempt {attempt+1}): SL=${new_sl_price:.8f}, TP={current_tp}, positionId={order_position_id}")
                    place_resp = await self.client.post(
                        f"{self.base_url}/api/v1/futures/tpsl/place_order",
                        headers=place_headers,
                        data=body
                    )
                    
                    if place_resp.status_code == 200:
                        place_result = place_resp.json()
                        logger.info(f"   Place response: {place_result}")
                        if place_result.get('code') == 0:
                            logger.info(f"   ✅ New TP/SL placed: SL=${new_sl_price:.6f}, TP={current_tp}")
                            old_order_ids_to_cancel.append(order_id)
                            success_count += 1
                            placed = True
                            break
                        else:
                            logger.error(f"   ❌ Place attempt {attempt+1} FAILED: code={place_result.get('code')}, msg={place_result.get('msg')}")
                    else:
                        logger.error(f"   ❌ Place HTTP error attempt {attempt+1}: {place_resp.status_code}")
                    
                    if attempt < 2:
                        await asyncio.sleep(0.5)
                
                if not placed:
                    logger.error(f"   ❌ Could not place replacement for order {order_id} - keeping original (position stays protected)")
            
            if old_order_ids_to_cancel:
                logger.info(f"🗑️ Cancelling {len(old_order_ids_to_cancel)} old TP/SL orders after successful placement...")
                for old_id in old_order_ids_to_cancel:
                    cancel_params = {
                        'symbol': bitunix_symbol,
                        'orderId': str(old_id)
                    }
                    nonce = os.urandom(16).hex()
                    timestamp = str(int(time.time() * 1000))
                    body = json.dumps(cancel_params, separators=(',', ':'))
                    signature = self._generate_signature(nonce, timestamp, "", body)
                    
                    cancel_headers = {
                        'api-key': self.api_key,
                        'nonce': nonce,
                        'timestamp': timestamp,
                        'sign': signature,
                        'Content-Type': 'application/json'
                    }
                    
                    cancel_resp = await self.client.post(
                        f"{self.base_url}/api/v1/futures/tpsl/cancel_order",
                        headers=cancel_headers,
                        data=body
                    )
                    
                    if cancel_resp.status_code == 200:
                        cancel_result = cancel_resp.json()
                        if cancel_result.get('code') == 0:
                            logger.info(f"   ✅ Cancelled old order {old_id}")
                        else:
                            logger.warning(f"   ⚠️ Cancel old order {old_id} returned: {cancel_result.get('msg')} (new order already in place)")
                    else:
                        logger.warning(f"   ⚠️ Cancel old order {old_id} HTTP error: {cancel_resp.status_code} (new order already in place)")
                    
                    await asyncio.sleep(0.2)
            
            logger.info(f"✅ Cancel-and-replace complete: {success_count}/{len(orders)} orders replaced for {symbol}")
            return success_count > 0
            
        except Exception as e:
            logger.error(f"Error in cancel-and-replace SL: {e}", exc_info=True)
            return False

    async def place_position_tpsl(self, symbol: str, position_id: str, sl_price: float, tp_price: float = None) -> bool:
        """Place a new position-level TP/SL order (replaces existing one).
        
        Endpoint: POST /api/v1/futures/tpsl/position/place_order
        Each position can only have one Position TP/SL Order - new one replaces old.
        """
        try:
            import json
            bitunix_symbol = symbol.replace('/', '')
            
            logger.info(f"🆕 PLACE POSITION TP/SL: {symbol} | positionId={position_id} | SL=${sl_price:.8f} | TP={tp_price}")
            
            place_params = {
                'symbol': bitunix_symbol,
                'positionId': str(position_id),
                'slPrice': f"{sl_price:.8f}",
                'slStopType': 'MARK_PRICE'
            }
            
            if tp_price and tp_price > 0:
                place_params['tpPrice'] = f"{tp_price:.8f}"
                place_params['tpStopType'] = 'MARK_PRICE'
            
            nonce = os.urandom(16).hex()
            timestamp = str(int(time.time() * 1000))
            body = json.dumps(place_params, separators=(',', ':'))
            
            signature = self._generate_signature(nonce, timestamp, "", body)
            
            headers = {
                'api-key': self.api_key,
                'nonce': nonce,
                'timestamp': timestamp,
                'sign': signature,
                'Content-Type': 'application/json'
            }
            
            response = await self.client.post(
                f"{self.base_url}/api/v1/futures/tpsl/position/place_order",
                headers=headers,
                data=body
            )
            
            if response.status_code == 200:
                result = response.json()
                logger.info(f"   Place position TP/SL response: {result}")
                if result.get('code') == 0:
                    logger.info(f"✅ POSITION TP/SL PLACED: {symbol} SL=${sl_price:.6f}, TP={tp_price}")
                    return True
                else:
                    logger.error(f"❌ Place position TP/SL FAILED: code={result.get('code')}, msg={result.get('msg')}")
                    return False
            else:
                logger.error(f"❌ Place position TP/SL HTTP error: {response.status_code} - {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"Error placing position TP/SL: {e}", exc_info=True)
            return False

    async def close_position(self, symbol: str, position_id: str = None) -> bool:
        """Flash close position at market price"""
        try:
            import json
            bitunix_symbol = symbol.replace('/', '')
            
            params = {
                'symbol': bitunix_symbol
            }
            
            if position_id:
                params['positionId'] = position_id
            
            nonce = os.urandom(16).hex()
            timestamp = str(int(time.time() * 1000))
            body = json.dumps(params, separators=(',', ':'))
            
            signature = self._generate_signature(nonce, timestamp, "", body)
            
            headers = {
                'api-key': self.api_key,
                'nonce': nonce,
                'timestamp': timestamp,
                'sign': signature,
                'Content-Type': 'application/json'
            }
            
            logger.info(f"🔄 FLASH CLOSE: {symbol} | positionId={position_id} | body={body}")
            
            response = await self.client.post(
                f"{self.base_url}/api/v1/futures/position/flash_close",
                headers=headers,
                data=body
            )
            
            if response.status_code == 200:
                data = response.json()
                logger.info(f"   Flash close response: {data}")
                if data.get('code') == 0:
                    logger.info(f"✅ Bitunix position closed for {symbol}")
                    return True
                else:
                    logger.error(f"❌ Flash close FAILED: code={data.get('code')}, msg={data.get('msg')}")
            else:
                logger.error(f"❌ Flash close HTTP error: {response.status_code} - {response.text}")
            
            return False
            
        except Exception as e:
            logger.error(f"Error closing Bitunix position: {e}", exc_info=True)
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
    logger.info(f"🚀 ═══ TRADE EXECUTION START ═══")
    logger.info(f"🚀 User: {user.id} ({user.username}) | Symbol: {signal.symbol} | Direction: {signal.direction}")
    logger.info(f"🚀 Signal Type: {signal.signal_type} | Trade Type: {trade_type} | Entry: ${signal.entry_price}")
    logger.info(f"🚀 Grandfathered: {user.grandfathered} | Subscribed: {user.is_subscribed}")
    
    try:
        # 🛡️ GLOBAL AUTO-TRADING CHECK: Must be enabled to execute ANY trade
        prefs_check = db.query(UserPreference).filter_by(user_id=user.id).first()
        if not prefs_check:
            reason = "No preferences found - auto-trading cannot proceed"
            logger.warning(f"🚫 NO PREFS: User {user.id} has no preferences configured, blocking trade")
            return None
        if not prefs_check.auto_trading_enabled:
            reason = "Auto-trading is DISABLED"
            logger.warning(f"🚫 AUTO-TRADING OFF: User {user.id} has auto_trading_enabled=False, blocking trade")
            return None
        
        # 🛡️ GLOBAL CONFIDENCE GATE: Minimum 7/10 AI confidence for ALL signals
        signal_confidence = getattr(signal, 'confidence', None)
        if signal_confidence is not None:
            conf_value = signal_confidence
            if conf_value > 10:
                conf_value = conf_value / 10
            if conf_value < 7:
                reason = f"AI confidence too low ({signal_confidence} → {conf_value}/10, minimum 7/10)"
                logger.warning(f"🚫 LOW CONFIDENCE BLOCKED: User {user.id} {signal.symbol} - confidence {conf_value}/10 below minimum 7/10")
                return None
        
        # 🛡️ CRITICAL DUPLICATE CHECK: Prevent duplicate trades at execution level
        existing_position = db.query(Trade).filter(
            Trade.user_id == user.id,
            Trade.symbol == signal.symbol,
            Trade.status == 'open'
        ).first()
        
        if existing_position:
            reason = f"Already has open {signal.symbol} position"
            logger.warning(f"🚫 DUPLICATE BLOCKED: User {user.id} {reason} (Trade #{existing_position.id})")
            await notify_admin_trade_failure(user, signal.symbol, reason)
            return None
        
        # 🛡️ MAX POSITIONS CHECK: Enforce position limit
        open_count = db.query(Trade).filter(
            Trade.user_id == user.id,
            Trade.status == 'open'
        ).count()
        
        prefs_check = db.query(UserPreference).filter_by(user_id=user.id).first()
        max_pos = prefs_check.max_positions if prefs_check and prefs_check.max_positions else 3
        
        if open_count >= max_pos:
            reason = f"Max positions reached ({open_count}/{max_pos})"
            logger.warning(f"🚫 MAX POSITIONS BLOCKED: User {user.id} has {open_count} open trades (limit {max_pos})")
            await notify_admin_trade_failure(user, signal.symbol, reason)
            return None
        
        # 🛡️ SUBSCRIPTION CHECK: Block trades if subscription expired
        logger.info(f"🔍 Subscription check for user {user.id}: is_subscribed={user.is_subscribed}, is_admin={user.is_admin}, grandfathered={user.grandfathered}")
        if not user.is_subscribed and not user.is_admin:
            reason = "Subscription expired"
            logger.warning(f"🚫 SUBSCRIPTION EXPIRED: User {user.id} subscription ended, blocking trade execution")
            await notify_admin_trade_failure(user, signal.symbol, reason)
            return None

        # Load user preferences early (needed for scalp mode and trade limit checks)
        prefs = db.query(UserPreference).filter_by(user_id=user.id).first()
        if not prefs:
            reason = "No preferences configured"
            logger.warning(f"🚫 NO PREFERENCES: User {user.id} has no preferences configured")
            await notify_admin_trade_failure(user, signal.symbol, reason)
            return None

        # 🛡️ SCALP MODE CHECK: Block scalp trades if disabled in preferences
        if trade_type == 'SCALP' and not prefs.scalp_mode_enabled:
            reason = "Scalp mode disabled"
            logger.warning(f"🚫 SCALP MODE DISABLED: User {user.id} has scalp mode off, blocking trade")
            await notify_admin_trade_failure(user, signal.symbol, reason)
            return None

        # 🛡️ TRADE COUNTER (no daily limit - window system handles frequency)
        if trade_type != 'SCALP':
            from datetime import datetime, date
            if not prefs.trades_reset_date or prefs.trades_reset_date.date() != date.today():
                prefs.trades_today = 0
                prefs.trades_reset_date = datetime.utcnow()
                db.commit()
            
            # Just track count, no limit (window system controls frequency)
            prefs.trades_today += 1
            db.commit()
            logger.info(f"📈 User {user.id} trade count today: {prefs.trades_today}")
        
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
        # - PARABOLIC_REVERSAL: Parabolic dump signals (validated during 50%+ exhaustion detection)
        pre_validated_types = ['TEST', 'technical', 'REVERSAL', 'DAY_TRADE', 'TOP_GAINER', 'PARABOLIC_REVERSAL', 'SOCIAL_SIGNAL', 'NEWS_SIGNAL', 'BTC_ORB_SCALP']
        
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

        # ⏱️ ENTRY FRESHNESS GUARD — re-fetch live price and reject if we're chasing
        # Social/news/scalp signals are generated 15-40s before execution.
        # If price has already run 0.75%+ in the trade direction, the move is done — skip.
        try:
            import httpx as _httpx
            _sym = signal.symbol.replace('/', '').replace(':USDT', '').replace('-USDT', '').upper()
            async with _httpx.AsyncClient(timeout=4) as _cl:
                _r = await _cl.get(
                    "https://fapi.binance.com/fapi/v1/ticker/price",
                    params={"symbol": _sym}
                )
                if _r.status_code == 200:
                    live_price = float(_r.json().get("price", 0))
                    entry_price = float(signal.entry_price or 0)
                    if live_price > 0 and entry_price > 0:
                        drift_pct = (live_price - entry_price) / entry_price * 100
                        chasing = (
                            (signal.direction == "LONG" and drift_pct > 0.75) or
                            (signal.direction == "SHORT" and drift_pct < -0.75)
                        )
                        setup_failed = (
                            (signal.direction == "LONG" and drift_pct < -1.0) or
                            (signal.direction == "SHORT" and drift_pct > 1.0)
                        )
                        if chasing:
                            logger.warning(
                                f"⏱️ ENTRY STALE — {signal.symbol} {signal.direction}: "
                                f"signal entry ${entry_price:.4f}, live ${live_price:.4f} "
                                f"({drift_pct:+.2f}%) — price already ran, skipping trade"
                            )
                            return None
                        if setup_failed:
                            logger.warning(
                                f"⏱️ SETUP FAILED — {signal.symbol} {signal.direction}: "
                                f"signal entry ${entry_price:.4f}, live ${live_price:.4f} "
                                f"({drift_pct:+.2f}%) — moved against direction, skipping trade"
                            )
                            return None
                        logger.info(
                            f"✅ Entry fresh — {signal.symbol}: signal ${entry_price:.4f}, "
                            f"live ${live_price:.4f} ({drift_pct:+.2f}%)"
                        )
        except Exception as _fresh_err:
            logger.debug(f"Entry freshness check skipped ({_fresh_err}) — proceeding with trade")

        prefs = db.query(UserPreference).filter_by(user_id=user.id).first()
        
        if not prefs:
            logger.error(f"No preferences found for user {user.id}")
            return None
        
        if not prefs.bitunix_api_key or not prefs.bitunix_api_secret:
            reason = "No Bitunix API keys configured"
            logger.info(f"User {user.id} has no Bitunix API configured")
            await notify_admin_trade_failure(user, signal.symbol, reason)
            return None
        
        # 🔍 DEBUG: Log encrypted key from DB (first 20 chars)
        enc_key_preview = prefs.bitunix_api_key[:30] if prefs.bitunix_api_key else "EMPTY"
        logger.info(f"🗄️ User {user.id} ENCRYPTED key from DB: {enc_key_preview}... (len={len(prefs.bitunix_api_key) if prefs.bitunix_api_key else 0})")
        
        try:
            api_key = decrypt_api_key(prefs.bitunix_api_key)
            api_secret = decrypt_api_key(prefs.bitunix_api_secret)
            # Debug: Show key preview (first 8 + last 4 chars only)
            key_preview = f"{api_key[:8]}...{api_key[-4:]}" if len(api_key) > 12 else "TOO_SHORT"
            logger.info(f"🔑 User {user.id} DECRYPTED key preview: {key_preview} (len={len(api_key)})")
        except Exception as decrypt_err:
            reason = f"API key decryption failed: {decrypt_err}"
            logger.error(f"❌ DECRYPTION FAILED for user {user.id}: {decrypt_err} - Check ENCRYPTION_KEY matches!")
            await notify_admin_trade_failure(user, signal.symbol, reason)
            return None
        
        if not api_key or not api_secret or len(api_key) < 10:
            reason = f"Invalid API keys (key_len={len(api_key) if api_key else 0})"
            logger.error(f"❌ Invalid decrypted keys for user {user.id} (key_len={len(api_key) if api_key else 0})")
            await notify_admin_trade_failure(user, signal.symbol, reason)
            return None
        
        trader = BitunixTrader(api_key, api_secret)
        
        try:
            balance = await trader.get_account_balance()
            logger.info(f"User {user.id} Bitunix balance: ${balance:.2f}")
            
            if balance <= 0:
                reason = "Insufficient balance ($0)"
                logger.warning(f"Insufficient balance for user {user.id}")
                await notify_admin_trade_failure(user, signal.symbol, reason)
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
            using_fixed_amount = False
            
            # For social/news trades, use risk-based sizing based on signal score
            if trade_type in ['SOCIAL_SIGNAL', 'SOCIAL_SHORT', 'NEWS_SIGNAL']:
                signal_score = signal.confidence or 60
                
                # News signals use their own position size settings
                if trade_type == 'NEWS_SIGNAL':
                    news_size = getattr(prefs, 'news_position_size_percent', 3.0) or 3.0
                    position_size = await trader.calculate_position_size(balance, news_size)
                    logger.info(f"📰 NEWS position: ${position_size:.2f} ({news_size}% of ${balance:.2f})")
                else:
                    # Minimum score 70 to trade - reject weak signals
                    if signal_score < 70:
                        reason = f"Signal score {signal_score} below minimum 70"
                        logger.warning(f"🚫 WEAK SIGNAL BLOCKED: {signal.symbol} score {signal_score} < 70")
                        return None
                    
                    # Get risk-based sizes from preferences
                    size_low = getattr(prefs, 'social_size_low', 5.0) or 5.0
                    size_med = getattr(prefs, 'social_size_medium', 3.0) or 3.0
                    size_high = getattr(prefs, 'social_size_high', 2.0) or 2.0
                    
                    # Determine size based on signal score (min 70)
                    if signal_score >= 85:
                        size_percent = size_low
                        risk_label = "LOW"
                    elif signal_score >= 75:
                        size_percent = size_med
                        risk_label = "MEDIUM"
                    else:  # 70-74
                        size_percent = size_high
                        risk_label = "HIGH"
                    
                    position_size = await trader.calculate_position_size(balance, size_percent)
                    logger.info(f"📊 SOCIAL {risk_label} risk position: ${position_size:.2f} ({size_percent}% - score {signal_score})")
            elif fixed_dollars and fixed_dollars > 0:
                # User set a fixed dollar amount - use it directly
                position_size = fixed_dollars
                using_fixed_amount = True
                logger.info(f"💵 Using FIXED position size: ${position_size:.2f} (user configured)")
                
                # Only cap to balance - respect user's explicit choice
                if position_size > balance * 0.95:
                    logger.warning(f"⚠️ Fixed position ${position_size:.2f} exceeds 95% of balance ${balance:.2f} - reducing to 90% of balance")
                    position_size = balance * 0.9
            else:
                # Use percentage-based sizing (default 5% for safety)
                position_size = await trader.calculate_position_size(
                    balance, 
                    prefs.position_size_percent or 5.0
                )
                logger.info(f"📊 Using PERCENTAGE position size: ${position_size:.2f} ({prefs.position_size_percent or 5.0}% of ${balance:.2f})")
            
            # Check minimum position size for Bitunix ($3 USDT minimum - lowered to allow all users)
            BITUNIX_MIN_POSITION = 3.0
            if position_size < BITUNIX_MIN_POSITION:
                logger.warning(f"⚠️ Position size ${position_size:.2f} below Bitunix minimum ${BITUNIX_MIN_POSITION:.2f} - continuing anyway")
            
            # AUTO-COMPOUND: Apply position multiplier for Top Gainer trades (Upgrade #7)
            # Only applies to percentage-based sizing, not fixed amounts
            if trade_type == 'TOP_GAINER' and prefs.top_gainers_auto_compound and not using_fixed_amount:
                multiplier = prefs.top_gainers_position_multiplier or 1.0
                if multiplier > 1.0:
                    position_size = position_size * multiplier
                    logger.info(f"🔥 TOP GAINER AUTO-COMPOUND: User {user.id} - Position size multiplied by {multiplier}x (${position_size:.2f})")
            
            # 🛡️ RISK CAP: Only applies to PERCENTAGE sizing (not fixed amounts)
            # Users who set fixed $ explicitly know what they want
            if not using_fixed_amount:
                MAX_POSITION_PERCENT = 0.20  # 20% of balance max for % sizing
                max_allowed = balance * MAX_POSITION_PERCENT
                
                if position_size > max_allowed:
                    original_size = position_size
                    position_size = max_allowed
                    logger.warning(f"🛡️ RISK CAP APPLIED: User {user.id} position reduced ${original_size:.2f} → ${position_size:.2f} (max 20% of balance)")
            
            # Use leverage override if provided (e.g., 5x for top gainers), otherwise use user preference
            leverage = leverage_override if leverage_override is not None else (prefs.user_leverage or 10)
            
            # For TOP_GAINER trades: Apply 80% profit/loss cap for high leverage
            # This ensures consistent risk management regardless of user leverage
            final_tp1 = signal.take_profit_1 if hasattr(signal, 'take_profit_1') else signal.take_profit
            final_tp2 = signal.take_profit_2 if hasattr(signal, 'take_profit_2') else None
            final_sl = signal.stop_loss
            
            # Use signal's actual TP/SL values (set by AI) - no overrides
            # Just log what we're using
            if trade_type == 'TOP_GAINER':
                entry = signal.entry_price
                if signal.direction == 'LONG':
                    tp_pct = ((final_tp1 - entry) / entry) * 100 if entry > 0 else 0
                    sl_pct = ((entry - final_sl) / entry) * 100 if entry > 0 else 0
                else:
                    tp_pct = ((entry - final_tp1) / entry) * 100 if entry > 0 else 0
                    sl_pct = ((final_sl - entry) / entry) * 100 if entry > 0 else 0
                
                logger.info(f"📊 TOP GAINER using AI levels for user {user.id} ({leverage}x): "
                           f"TP: {tp_pct:.2f}% ({tp_pct * leverage:.0f}% profit), "
                           f"SL: {sl_pct:.2f}% ({sl_pct * leverage:.0f}% loss)")
            
            # For signals with dual TPs (LONGS), split into 2 orders: 50% at TP1, 50% at TP2
            has_dual_tp = final_tp2 is not None
            
            # 🔄 RETRY LOGIC: Try up to 3 times for temporary API failures
            try:
                trade_result = await execute_trade_with_retry(
                    trader=trader,
                    signal=signal,
                    user=user,
                    position_size=position_size,
                    leverage=leverage,
                    final_sl=final_sl,
                    final_tp1=final_tp1,
                    final_tp2=final_tp2,
                    max_retries=3
                )
                logger.info(f"✅ Trade executed successfully for user {user.id} on {signal.symbol}")
            except RetryableError as e:
                reason = f"Trade failed after 3 retries: {e}"
                logger.error(f"❌ {reason}")
                await notify_admin_trade_failure(user, signal.symbol, reason)
                return None
            except Exception as e:
                reason = f"Trade execution error: {e}"
                logger.error(f"❌ {reason}")
                await notify_admin_trade_failure(user, signal.symbol, reason)
                return None
            
            if has_dual_tp:
                result1 = trade_result.get('result1')
                result2 = trade_result.get('result2')
                
                # Success if at least ONE order succeeded
                if (result1 and result1.get('success')) or (result2 and result2.get('success')):
                    # CRITICAL: Fetch actual filled contracts from Bitunix immediately
                    # This is required for accurate TP1 detection (50% size reduction)
                    actual_contracts = None
                    try:
                        await asyncio.sleep(0.5)  # Brief delay for order to settle
                        positions = await trader.get_open_positions()
                        bitunix_symbol = signal.symbol.replace('/', '')
                        for pos in positions:
                            if pos.get('symbol') == bitunix_symbol:
                                actual_contracts = pos.get('total', 0)
                                logger.info(f"📦 Captured actual filled contracts for {signal.symbol}: {actual_contracts}")
                                break
                    except Exception as e:
                        logger.warning(f"Could not fetch filled contracts: {e}")
                    
                    trade = Trade(
                        user_id=user.id,
                        signal_id=signal.id,
                        symbol=signal.symbol,
                        direction=signal.direction,
                        entry_price=signal.entry_price,
                        stop_loss=final_sl,
                        take_profit=final_tp1,
                        take_profit_1=final_tp1,
                        take_profit_2=final_tp2,
                        position_size=position_size,
                        remaining_size=position_size,
                        original_contracts=actual_contracts,
                        status='open',
                        trade_type=trade_type,
                        leverage=leverage
                    )
                    db.add(trade)
                    db.commit()
                    
                    # Log social/news trade opening
                    if trade_type in ['SOCIAL_SIGNAL', 'SOCIAL_SHORT', 'NEWS_SIGNAL']:
                        try:
                            from app.services.social_trade_logger import log_social_signal_generated
                            await log_social_signal_generated(
                                db, user.id, 
                                {
                                    'symbol': signal.symbol,
                                    'direction': signal.direction,
                                    'trade_type': trade_type,
                                    'strategy': getattr(signal, 'strategy', trade_type),
                                    'confidence': signal.confidence,
                                    'reasoning': signal.reasoning,
                                    'entry_price': signal.entry_price,
                                    'stop_loss': final_sl,
                                    'take_profit': final_tp1,
                                    'tp_percent': getattr(signal, 'tp_percent', None),
                                    'sl_percent': getattr(signal, 'sl_percent', None),
                                    'rsi': getattr(signal, 'rsi', None),
                                },
                                risk_level=None
                            )
                            from app.services.social_trade_logger import log_social_trade_opened
                            from app.models import SocialTradeLog
                            latest_log = db.query(SocialTradeLog).filter(
                                SocialTradeLog.user_id == user.id,
                                SocialTradeLog.symbol == signal.symbol,
                                SocialTradeLog.status == 'pending'
                            ).order_by(SocialTradeLog.signal_time.desc()).first()
                            if latest_log:
                                await log_social_trade_opened(db, latest_log.id, trade, position_size, leverage)
                        except Exception as log_err:
                            logger.warning(f"Failed to log social trade open: {log_err}")
                    
                    order1_status = "✅" if result1 and result1.get('success') else "❌"
                    order2_status = "✅" if result2 and result2.get('success') else "❌"
                    logger.info(f"✅ Bitunix DUAL TP trade for user {user.id}: {signal.symbol} {signal.direction} | TP1 {order1_status} @ ${final_tp1:.6f} | TP2 {order2_status} @ ${final_tp2:.6f}")
                    return trade
                else:
                    logger.error(f"Failed to place dual TP orders for user {user.id}: Order1: {result1}, Order2: {result2}")
                    return None
            else:
                # SINGLE TP: Use result from retry logic
                result = trade_result.get('result')
                
                if result and result.get('success'):
                    trade = Trade(
                        user_id=user.id,
                        signal_id=signal.id,
                        symbol=signal.symbol,
                        direction=signal.direction,
                        entry_price=signal.entry_price,
                        stop_loss=final_sl,
                        take_profit=final_tp1,
                        take_profit_1=final_tp1,
                        take_profit_2=None,
                        position_size=position_size,
                        remaining_size=position_size,
                        status='open',
                        trade_type=trade_type,
                        leverage=leverage
                    )
                    db.add(trade)
                    db.commit()
                    
                    # Log social/news trade opening
                    if trade_type in ['SOCIAL_SIGNAL', 'SOCIAL_SHORT', 'NEWS_SIGNAL']:
                        try:
                            from app.services.social_trade_logger import log_social_signal_generated
                            await log_social_signal_generated(
                                db, user.id, 
                                {
                                    'symbol': signal.symbol,
                                    'direction': signal.direction,
                                    'trade_type': trade_type,
                                    'strategy': getattr(signal, 'strategy', trade_type),
                                    'confidence': signal.confidence,
                                    'reasoning': signal.reasoning,
                                    'entry_price': signal.entry_price,
                                    'stop_loss': final_sl,
                                    'take_profit': final_tp1,
                                    'tp_percent': getattr(signal, 'tp_percent', None),
                                    'sl_percent': getattr(signal, 'sl_percent', None),
                                    'rsi': getattr(signal, 'rsi', None),
                                },
                                risk_level=None
                            )
                            from app.services.social_trade_logger import log_social_trade_opened
                            from app.models import SocialTradeLog
                            latest_log = db.query(SocialTradeLog).filter(
                                SocialTradeLog.user_id == user.id,
                                SocialTradeLog.symbol == signal.symbol,
                                SocialTradeLog.status == 'pending'
                            ).order_by(SocialTradeLog.signal_time.desc()).first()
                            if latest_log:
                                await log_social_trade_opened(db, latest_log.id, trade, position_size, leverage)
                        except Exception as log_err:
                            logger.warning(f"Failed to log social trade open: {log_err}")
                    
                    logger.info(f"✅ Bitunix trade recorded for user {user.id}: {signal.symbol} {signal.direction}")
                    return trade
                else:
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
