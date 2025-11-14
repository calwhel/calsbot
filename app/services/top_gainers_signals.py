"""
Top Gainers Trading Mode - Generates signals from Bitunix top movers
Focuses on momentum plays with 5x leverage and 15% TP/SL
Includes 48h watchlist to catch delayed reversals
"""
import asyncio
import logging
import httpx
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)

# Track SHORTS that lost to prevent re-shorting the same pump
# Format: {symbol: datetime_when_cooldown_expires}
shorts_cooldown = {}


def calculate_leverage_capped_targets(
    entry_price: float,
    direction: str,
    tp_pcts: List[float],  # List of TP percentages [TP1, TP2, ...] or single value
    base_sl_pct: float,
    leverage: int,
    max_profit_cap: float = 80.0,
    max_loss_cap: float = 80.0
) -> Dict:
    """
    Calculate TP/SL prices with profit/loss cap based on leverage
    
    Maintains proportional spacing between multiple TPs when capping is applied.
    Scales the entire TP ladder uniformly to preserve strategy integrity.
    
    Args:
        entry_price: Entry price
        direction: 'LONG' or 'SHORT'
        tp_pcts: List of TP percentages (e.g., [5.0, 10.0] for dual TP LONGS)
        base_sl_pct: Base stop loss percentage (price move, e.g., 4%)
        leverage: User's leverage (5x, 10x, 20x, etc.)
        max_profit_cap: Maximum profit percentage allowed (default: 80%)
        max_loss_cap: Maximum loss percentage allowed (default: 80%)
    
    Returns:
        Dict with tp_prices[], sl_price, scaling_factor, tp_profit_pcts[], sl_loss_pct
    
    Examples:
        LONG with 20x leverage, TPs [5%, 10%]:
        - Max TP (10%) would be 200% profit → exceeds 80% cap
        - Scaling factor: 80% / 200% = 0.4
        - Scaled TPs: [2%, 4%] → profits: [40%, 80%] ✅
        - TP spacing preserved: TP1 = 40%, TP2 = 80% (still 2:1 ratio)
    """
    # Ensure tp_pcts is a list
    if not isinstance(tp_pcts, list):
        tp_pcts = [tp_pcts]
    
    # Find the maximum TP (furthest profit target)
    max_tp_pct = max(tp_pcts)
    max_tp_profit = max_tp_pct * leverage
    
    # Calculate scaling factor if max TP exceeds cap
    if max_tp_profit > max_profit_cap:
        scaling_factor = max_profit_cap / max_tp_profit
    else:
        scaling_factor = 1.0
    
    # Scale all TPs proportionally
    effective_tp_pcts = [tp * scaling_factor for tp in tp_pcts]
    effective_sl_pct = min(base_sl_pct * scaling_factor, max_loss_cap / leverage)
    
    # Calculate actual profit/loss percentages with leverage
    tp_profit_pcts = [tp * leverage for tp in effective_tp_pcts]
    sl_loss_pct = effective_sl_pct * leverage
    
    # Calculate price targets
    tp_prices = []
    for effective_tp in effective_tp_pcts:
        if direction == 'LONG':
            tp_price = entry_price * (1 + effective_tp / 100)
        else:  # SHORT
            tp_price = entry_price * (1 - effective_tp / 100)
        tp_prices.append(tp_price)
    
    # Calculate SL price
    if direction == 'LONG':
        sl_price = entry_price * (1 - effective_sl_pct / 100)
    else:  # SHORT
        sl_price = entry_price * (1 + effective_sl_pct / 100)
    
    return {
        'tp_prices': tp_prices,  # List of TP prices
        'sl_price': sl_price,
        'effective_tp_pcts': effective_tp_pcts,  # List of effective price move %
        'effective_sl_pct': effective_sl_pct,  # Effective SL price move %
        'tp_profit_pcts': tp_profit_pcts,  # List of profit % with leverage
        'sl_loss_pct': sl_loss_pct,  # Loss % with leverage
        'scaling_factor': scaling_factor,  # How much we scaled (1.0 = no cap)
        'is_capped': scaling_factor < 1.0  # True if cap was applied
    }


def add_short_cooldown(symbol: str, cooldown_minutes: int = 30):
    """
    Add a symbol to SHORT cooldown after a losing trade
    
    Args:
        symbol: Trading pair (e.g., 'LSK/USDT')
        cooldown_minutes: How long to block SHORTS (default: 30 min)
    """
    cooldown_until = datetime.utcnow() + timedelta(minutes=cooldown_minutes)
    shorts_cooldown[symbol] = cooldown_until
    logger.info(f"🚫 {symbol} added to SHORT cooldown for {cooldown_minutes} minutes (prevents re-shorting strong pump)")
    return cooldown_until


class TopGainersSignalService:
    """Service to fetch and analyze top gainers from Bitunix using direct API"""
    
    def __init__(self):
        self.base_url = "https://fapi.bitunix.com"  # For tickers and trading
        self.binance_url = "https://fapi.binance.com"  # For candle data (Binance Futures public API)
        self.client = httpx.AsyncClient(timeout=30.0)
        self.min_volume_usdt = 400000  # $400k minimum 24h volume for liquidity (catches more pumps!)
        self.max_spread_percent = 0.5  # Max 0.5% bid-ask spread for good execution
        self.min_depth_usdt = 50000  # Min $50k liquidity at ±1% price levels
        
    async def initialize(self):
        """Initialize Bitunix API client"""
        try:
            logger.info("TopGainersSignalService initialized with Bitunix direct API")
        except Exception as e:
            logger.error(f"Failed to initialize TopGainersSignalService: {e}")
            raise
    
    async def fetch_candles(self, symbol: str, interval: str, limit: int = 100) -> List:
        """
        Fetch OHLCV candles from Binance Futures (Bitunix klines API is broken)
        
        Uses Binance Futures public API for candle data analysis.
        Bitunix is still used for tickers (finding pumps) and trade execution.
        
        Args:
            symbol: Trading pair (e.g., 'BTC/USDT')
            interval: Timeframe ('5m', '15m', '1h', etc.)
            limit: Number of candles to fetch
            
        Returns:
            List of candles [[timestamp, open, high, low, close, volume], ...]
        """
        try:
            # Convert symbol format: BTC/USDT → BTCUSDT (Binance format)
            binance_symbol = symbol.replace('/', '')
            
            # Use Binance Futures public API (no auth needed)
            url = f"{self.binance_url}/fapi/v1/klines"
            params = {
                'symbol': binance_symbol,
                'interval': interval,
                'limit': limit
            }
            
            response = await self.client.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            # Binance returns direct array of candles (no wrapper object)
            if not isinstance(data, list) or not data:
                logger.warning(f"No candle data returned for {symbol} from Binance")
                return []
            
            candles = data
            
            # Convert Binance format to standardized format: [timestamp, open, high, low, close, volume]
            # Binance candles: [open_time, open, high, low, close, volume, close_time, quote_volume, trades, ...]
            formatted_candles = []
            for candle in candles:
                if isinstance(candle, list) and len(candle) >= 6:
                    formatted_candles.append([
                        int(candle[0]),      # open time (timestamp in milliseconds)
                        float(candle[1]),    # open
                        float(candle[2]),    # high
                        float(candle[3]),    # low
                        float(candle[4]),    # close
                        float(candle[5])     # volume
                    ])
                else:
                    logger.warning(f"Unexpected candle format for {symbol}: {type(candle)}")
                    continue
            
            # Binance returns candles in chronological order (oldest first)
            # No need to reverse - already in correct order for technical indicators
            
            return formatted_candles
            
        except Exception as e:
            logger.error(f"Error fetching candles for {symbol}: {e}")
            return []
    
    async def check_liquidity(self, symbol: str) -> Dict:
        """
        Check execution quality via spread and order book depth
        
        Returns quality metrics:
        - is_liquid: bool (passes all checks)
        - spread_percent: float (bid-ask spread %)
        - depth_1pct: float (total liquidity at ±1% levels in USDT)
        - reason: str (failure reason if not liquid)
        """
        try:
            # Convert symbol format: BTC/USDT → BTCUSDT
            bitunix_symbol = symbol.replace('/', '')
            
            # Fetch ticker for current bid/ask
            ticker_url = f"{self.base_url}/api/v1/futures/market/tickers"
            ticker_response = await self.client.get(ticker_url, params={'symbols': bitunix_symbol})
            ticker_data = ticker_response.json()
            
            # Parse ticker response
            tickers = ticker_data.get('data', []) if isinstance(ticker_data, dict) else ticker_data
            if not tickers or not isinstance(tickers, list):
                return {'is_liquid': False, 'reason': 'No ticker data'}
            
            ticker = tickers[0] if isinstance(tickers, list) else tickers
            
            # Bitunix API doesn't return bid/ask, only markPrice and lastPrice
            # Use 24h volume as primary liquidity indicator
            volume_24h = float(ticker.get('quoteVol', 0) or ticker.get('volume24h', 0))
            last_price = float(ticker.get('lastPrice', 0) or ticker.get('last', 0))
            
            if last_price <= 0:
                return {'is_liquid': False, 'reason': 'Invalid price data'}
            
            # Note: No spread check since Bitunix API doesn't provide bid/ask
            # Volume is more important for momentum trades anyway
            
            # If volume is high, assume decent depth
            if volume_24h < self.min_volume_usdt:
                return {
                    'is_liquid': False,
                    'spread_percent': 0,
                    'reason': f'Low 24h volume: ${volume_24h:,.0f} (need ${self.min_volume_usdt:,.0f}+)'
                }
            
            # Passed all checks
            return {
                'is_liquid': True,
                'spread_percent': 0,  # Not available from Bitunix API
                'volume_24h': volume_24h,
                'reason': 'Good liquidity'
            }
            
        except Exception as e:
            logger.error(f"Error checking liquidity for {symbol}: {e}")
            return {'is_liquid': False, 'reason': f'Error: {str(e)}'}
    
    async def check_manipulation_risk(self, symbol: str, candles_5m: List) -> Dict:
        """
        Detect pump & dump manipulation patterns
        
        Checks:
        1. Volume distribution (not just 1 giant candle)
        2. Wick-to-body ratio (avoid fake pumps with huge wicks)
        3. Listing age (skip coins <48h old)
        
        Returns:
        - is_safe: bool (passes all checks)
        - risk_score: int (0-100, higher = more risky)
        - flags: List[str] (specific red flags)
        """
        try:
            flags = []
            risk_score = 0
            
            if not candles_5m or len(candles_5m) < 10:
                return {'is_safe': False, 'risk_score': 100, 'flags': ['Insufficient candle data']}
            
            # Extract last 10 candles
            recent_candles = candles_5m[-10:]
            volumes = [c[5] for c in recent_candles]
            
            # Check 1: Volume Distribution (not just 1 whale candle)
            max_volume = max(volumes)
            avg_volume = sum(volumes) / len(volumes)
            
            if max_volume > avg_volume * 5:
                flags.append('Single whale candle detected')
                risk_score += 30
            
            # Count elevated volume candles (>1.5x average)
            elevated_count = sum(1 for v in volumes if v > avg_volume * 1.5)
            if elevated_count < 3:
                flags.append('Volume not sustained (need 3+ elevated candles)')
                risk_score += 20
            
            # Check 2: Wick-to-Body Ratio (avoid fake pumps)
            current_candle = candles_5m[-1]
            open_price = current_candle[1]
            high = current_candle[2]
            low = current_candle[3]
            close = current_candle[4]
            
            body_size = abs(close - open_price)
            upper_wick = high - max(close, open_price)
            lower_wick = min(close, open_price) - low
            
            # If wick is >2x body size, it's a fake pump
            if body_size > 0:
                wick_to_body_ratio = max(upper_wick, lower_wick) / body_size
                if wick_to_body_ratio > 2.0:
                    flags.append(f'Excessive wick (fake pump): wick/body={wick_to_body_ratio:.1f}x')
                    risk_score += 40
            
            # Check 3: Listing Age (skip very new coins < 10 candles of history)
            # If we have less than 30 candles of 5m data, coin might be too new
            if len(candles_5m) < 30:
                flags.append('Coin too new (<2.5 hours of data)')
                risk_score += 50
            
            # Determine if safe
            is_safe = risk_score < 50 and len(flags) < 2
            
            return {
                'is_safe': is_safe,
                'risk_score': risk_score,
                'flags': flags if flags else ['No red flags'],
                'elevated_volume_candles': elevated_count
            }
            
        except Exception as e:
            logger.error(f"Error checking manipulation risk for {symbol}: {e}")
            return {'is_safe': False, 'risk_score': 100, 'flags': [f'Error: {str(e)}']}
    
    async def get_top_gainers(self, limit: int = 10, min_change_percent: float = 10.0) -> List[Dict]:
        """
        Fetch top gainers from Bitunix based on 24h price change using direct API
        
        OPTIMIZED FOR SHORTS: Higher min_change (10%+) = better reversal candidates
        
        Args:
            limit: Number of top gainers to return
            min_change_percent: Minimum 24h change % to qualify (default 10% for shorts)
            
        Returns:
            List of {symbol, change_percent, volume, price} sorted by change %
        """
        try:
            # Fetch 24h ticker statistics from Bitunix public API
            # Correct endpoint: /api/v1/futures/market/tickers (returns all tickers if no symbols param)
            url = f"{self.base_url}/api/v1/futures/market/tickers"
            response = await self.client.get(url)
            response.raise_for_status()
            tickers_data = response.json()
            
            # Debug: Log response structure
            logger.info(f"Bitunix ticker API response: code={tickers_data.get('code')}, msg={tickers_data.get('msg')}, data_type={type(tickers_data.get('data'))}, data_length={len(tickers_data.get('data')) if isinstance(tickers_data.get('data'), (list, dict)) else 'N/A'}")
            
            # Handle different possible response formats
            if isinstance(tickers_data, list):
                # Direct list of tickers
                tickers = tickers_data
            elif isinstance(tickers_data, dict):
                # Check for common keys: 'data', 'result', or direct ticker data
                tickers = tickers_data.get('data') or tickers_data.get('result') or tickers_data.get('tickers', [])
                
                # If data is a dict (not a list), it might contain the tickers differently
                if isinstance(tickers, dict) and not isinstance(tickers, list):
                    logger.info(f"Data is a dict with keys: {tickers.keys()}")
                    # Try common nested patterns
                    tickers = tickers.get('tickers') or tickers.get('list') or []
            else:
                logger.error(f"Unexpected ticker response type: {type(tickers_data)}")
                return []
            
            if not tickers:
                logger.warning(f"No tickers returned from Bitunix API. Full response: {tickers_data}")
                return []
            
            gainers = []
            for ticker in tickers:
                symbol = ticker.get('symbol', '')
                
                # Only consider USDT perpetuals
                if not symbol.endswith('USDT'):
                    continue
                
                # Calculate 24h percentage change from open to last price
                try:
                    open_price = float(ticker.get('open', 0))
                    last_price = float(ticker.get('lastPrice') or ticker.get('last', 0))
                    
                    if open_price > 0 and last_price > 0:
                        change_percent = ((last_price - open_price) / open_price) * 100
                    else:
                        continue
                except (ValueError, TypeError):
                    continue
                
                # Volume in USDT (quoteVol field)
                volume_usdt = float(ticker.get('quoteVol', 0))
                
                # Filter criteria
                if (change_percent >= min_change_percent and 
                    volume_usdt >= self.min_volume_usdt):
                    
                    gainers.append({
                        'symbol': symbol.replace('USDT', '/USDT'),  # Format as BTC/USDT
                        'change_percent': round(change_percent, 2),
                        'volume_24h': round(volume_usdt, 0),
                        'price': last_price,
                        'high_24h': float(ticker.get('high', 0)),
                        'low_24h': float(ticker.get('low', 0))
                    })
            
            # Sort by change % descending
            gainers.sort(key=lambda x: x['change_percent'], reverse=True)
            
            return gainers[:limit]
            
        except Exception as e:
            logger.error(f"Error fetching top gainers: {e}", exc_info=True)
            return []
    
    async def validate_fresh_5m_pump(self, symbol: str) -> Optional[Dict]:
        """
        Validate ULTRA-EARLY 5-minute pump (5%+ green candle in last 5min)
        
        ⚡ TIER 1 - ULTRA-EARLY DETECTION ⚡
        Catches pumps in the first 5-10 minutes!
        
        Requirements:
        - Most recent 5m candle is 5%+ green (close > open)
        - Volume 2.5x+ average of previous 3 candles (relaxed from 3.0x)
        - Candle is fresh (within 10 minutes, relaxed from 7)
        
        Returns:
            {
                'is_fresh_pump': bool,
                'tier': '5m',
                'candle_change_percent': float,
                'volume_ratio': float,
                'candle_close_time': int
            }
        """
        try:
            from datetime import datetime
            
            # Fetch last 5 5m candles (need 4 for volume average)
            candles_5m = await self.fetch_candles(symbol, '5m', limit=6)
            
            if len(candles_5m) < 4:
                return None
            
            # Most recent candle
            latest_candle = candles_5m[-1]
            timestamp, open_price, high, low, close_price, volume = latest_candle
            
            # Check 1: Is candle fresh? (within 7 minutes)
            # CRITICAL FIX: timestamp is open_time, add 5min interval to get close_time
            candle_close_time = datetime.fromtimestamp((timestamp + 300000) / 1000)  # +5min in ms
            now = datetime.utcnow()
            age_minutes = (now - candle_close_time).total_seconds() / 60
            
            if age_minutes > 10:  # Candle older than 10 minutes = stale (relaxed from 7)
                return {'is_fresh_pump': False, 'reason': 'stale_candle'}
            
            # Check 2: Is it a green candle AND 5%+ gain?
            if close_price <= open_price:
                return {'is_fresh_pump': False, 'reason': 'not_green_candle'}
            
            candle_change_percent = ((close_price - open_price) / open_price) * 100
            
            if candle_change_percent < 5.0:
                return {'is_fresh_pump': False, 'reason': 'insufficient_pump', 'change': candle_change_percent}
            
            # Check 3: Volume 3.0x+ average of previous 3 candles (high bar for 5m!)
            prev_volumes = [candles_5m[-4][5], candles_5m[-3][5], candles_5m[-2][5]]
            avg_volume = sum(prev_volumes) / len(prev_volumes)
            
            volume_ratio = volume / avg_volume if avg_volume > 0 else 0
            
            if volume_ratio < 2.5:  # Relaxed from 3.0x
                return {'is_fresh_pump': False, 'reason': 'low_volume', 'volume_ratio': volume_ratio}
            
            # ✅ ULTRA-EARLY PUMP DETECTED!
            logger.info(f"⚡ {symbol} ULTRA-EARLY 5m PUMP: +{candle_change_percent:.1f}% with {volume_ratio:.1f}x volume")
            
            return {
                'is_fresh_pump': True,
                'tier': '5m',
                'candle_change_percent': round(candle_change_percent, 2),
                'volume_ratio': round(volume_ratio, 2),
                'candle_close_time': timestamp,
                'candle_age_minutes': round(age_minutes, 1)
            }
            
        except Exception as e:
            logger.error(f"Error validating 5m pump for {symbol}: {e}")
            return None
    
    async def validate_fresh_15m_pump(self, symbol: str) -> Optional[Dict]:
        """
        Validate EARLY 15-minute pump (7%+ green candle in last 15min)
        
        🔥 TIER 2 - EARLY DETECTION 🔥
        Catches pumps in the first 15-20 minutes!
        
        Requirements:
        - Most recent 15m candle is 7%+ green (close > open)
        - Volume 2.5x+ average of previous 2-3 candles
        - Candle is fresh (within 20 minutes)
        
        Returns:
            {
                'is_fresh_pump': bool,
                'tier': '15m',
                'candle_change_percent': float,
                'volume_ratio': float,
                'candle_close_time': int
            }
        """
        try:
            from datetime import datetime
            
            # Fetch last 4 15m candles (need 3 for volume average)
            candles_15m = await self.fetch_candles(symbol, '15m', limit=5)
            
            if len(candles_15m) < 3:
                return None
            
            # Most recent candle
            latest_candle = candles_15m[-1]
            timestamp, open_price, high, low, close_price, volume = latest_candle
            
            # Check 1: Is candle fresh? (within 20 minutes)
            # CRITICAL FIX: timestamp is open_time, add 15min interval to get close_time
            candle_close_time = datetime.fromtimestamp((timestamp + 900000) / 1000)  # +15min in ms
            now = datetime.utcnow()
            age_minutes = (now - candle_close_time).total_seconds() / 60
            
            if age_minutes > 20:  # Candle older than 20 minutes = stale
                return {'is_fresh_pump': False, 'reason': 'stale_candle'}
            
            # Check 2: Is it a green candle AND 7%+ gain?
            if close_price <= open_price:
                return {'is_fresh_pump': False, 'reason': 'not_green_candle'}
            
            candle_change_percent = ((close_price - open_price) / open_price) * 100
            
            if candle_change_percent < 7.0:
                return {'is_fresh_pump': False, 'reason': 'insufficient_pump', 'change': candle_change_percent}
            
            # Check 3: Volume 2.5x+ average of previous 2 candles
            prev_volumes = [candles_15m[-3][5], candles_15m[-2][5]]
            avg_volume = sum(prev_volumes) / len(prev_volumes)
            
            volume_ratio = volume / avg_volume if avg_volume > 0 else 0
            
            if volume_ratio < 2.5:
                return {'is_fresh_pump': False, 'reason': 'low_volume', 'volume_ratio': volume_ratio}
            
            # ✅ EARLY PUMP DETECTED!
            logger.info(f"🔥 {symbol} EARLY 15m PUMP: +{candle_change_percent:.1f}% with {volume_ratio:.1f}x volume")
            
            return {
                'is_fresh_pump': True,
                'tier': '15m',
                'candle_change_percent': round(candle_change_percent, 2),
                'volume_ratio': round(volume_ratio, 2),
                'candle_close_time': timestamp,
                'candle_age_minutes': round(age_minutes, 1)
            }
            
        except Exception as e:
            logger.error(f"Error validating 15m pump for {symbol}: {e}")
            return None
    
    async def validate_fresh_30m_pump(self, symbol: str) -> Optional[Dict]:
        """
        Validate FRESH 30-minute pump (10%+ green candle in last 30min)
        
        Requirements:
        - Most recent 30m candle is 10%+ green (close > open)
        - Volume 2.0x+ average of previous 2-3 candles
        - Candle is fresh (within 35 minutes)
        
        Returns:
            {
                'is_fresh_pump': bool,
                'candle_change_percent': float,
                'volume_ratio': float,
                'candle_close_time': int
            }
        """
        try:
            from datetime import datetime, timedelta
            
            # Fetch last 3 30m candles (need 3 for volume average)
            candles_30m = await self.fetch_candles(symbol, '30m', limit=5)
            
            if len(candles_30m) < 3:
                return None
            
            # Most recent candle
            latest_candle = candles_30m[-1]
            timestamp, open_price, high, low, close_price, volume = latest_candle
            
            # Check 1: Is candle fresh? (within 35 minutes)
            # CRITICAL FIX: timestamp is open_time, add 30min interval to get close_time
            candle_close_time = datetime.fromtimestamp((timestamp + 1800000) / 1000)  # +30min in ms
            now = datetime.utcnow()
            age_minutes = (now - candle_close_time).total_seconds() / 60
            
            if age_minutes > 35:  # Candle older than 35 minutes = stale
                logger.debug(f"{symbol} 30m candle too old: {age_minutes:.1f} min")
                return {'is_fresh_pump': False, 'reason': 'stale_candle'}
            
            # Check 2: Is it a green candle AND 10%+ gain?
            if close_price <= open_price:
                return {'is_fresh_pump': False, 'reason': 'not_green_candle'}
            
            candle_change_percent = ((close_price - open_price) / open_price) * 100
            
            if candle_change_percent < 10.0:
                logger.debug(f"{symbol} 30m candle only +{candle_change_percent:.1f}% (need 10%+)")
                return {'is_fresh_pump': False, 'reason': 'insufficient_pump', 'change': candle_change_percent}
            
            # Check 3: Volume 2.0x+ average of previous 2 candles
            prev_volumes = [candles_30m[-3][5], candles_30m[-2][5]]  # Previous 2 candles' volume
            avg_volume = sum(prev_volumes) / len(prev_volumes)
            
            volume_ratio = volume / avg_volume if avg_volume > 0 else 0
            
            if volume_ratio < 2.0:
                logger.debug(f"{symbol} 30m volume only {volume_ratio:.1f}x (need 2.0x+)")
                return {'is_fresh_pump': False, 'reason': 'low_volume', 'volume_ratio': volume_ratio}
            
            # ✅ All checks passed - FRESH PUMP!
            logger.info(f"✅ {symbol} FRESH 30m PUMP: +{candle_change_percent:.1f}% with {volume_ratio:.1f}x volume")
            
            return {
                'is_fresh_pump': True,
                'tier': '30m',
                'candle_change_percent': round(candle_change_percent, 2),
                'volume_ratio': round(volume_ratio, 2),
                'candle_close_time': timestamp,
                'candle_age_minutes': round(age_minutes, 1)
            }
            
        except Exception as e:
            logger.error(f"Error validating 30m pump for {symbol}: {e}")
            return None
    
    async def get_early_pumpers(self, limit: int = 10, min_change: float = 5.0, max_change: float = 200.0) -> List[Dict]:
        """
        Fetch FRESH PUMP candidates for LONG entries
        
        ⚡ NEW: 3-TIER ULTRA-EARLY DETECTION ⚡
        1. Quick filter: 24h movers with 5%+ gain (fast ticker scan)
        2. Multi-tier validation (checks earliest to latest):
           - TIER 1 (5m):  5%+ pump, 3x volume   → Ultra-early (5-10 min)
           - TIER 2 (15m): 7%+ pump, 2.5x volume → Early (15-20 min)
           - TIER 3 (30m): 10%+ pump, 2x volume  → Fresh (25-30 min)
        
        Returns ONLY fresh pumps (not stale 24h gains!) with tier priority!
        
        Args:
            limit: Number of fresh pumpers to return
            min_change: Minimum 24h change % for pre-filter (default 5%)
            max_change: Maximum 24h change % (default 200% - no cap)
            
        Returns:
            List of {symbol, change_percent, volume_24h, tier, fresh_pump_data} sorted by tier priority then pump %
        """
        try:
            url = f"{self.base_url}/api/v1/futures/market/tickers"
            response = await self.client.get(url)
            response.raise_for_status()
            tickers_data = response.json()
            
            # Handle response format
            if isinstance(tickers_data, list):
                tickers = tickers_data
            elif isinstance(tickers_data, dict):
                tickers = tickers_data.get('data') or tickers_data.get('result') or tickers_data.get('tickers', [])
            else:
                logger.error(f"Unexpected ticker response type: {type(tickers_data)}")
                return []
            
            if not tickers:
                return []
            
            # STAGE 1: Quick pre-filter on 24h movers
            candidates = []
            for ticker in tickers:
                symbol = ticker.get('symbol', '')
                
                if not symbol.endswith('USDT'):
                    continue
                
                try:
                    open_price = float(ticker.get('open', 0))
                    last_price = float(ticker.get('lastPrice') or ticker.get('last', 0))
                    
                    if open_price > 0 and last_price > 0:
                        change_percent = ((last_price - open_price) / open_price) * 100
                    else:
                        continue
                except (ValueError, TypeError):
                    continue
                
                volume_usdt = float(ticker.get('quoteVol', 0))
                
                # Pre-filter: 5%+ 24h gain with volume (quick check)
                if (min_change <= change_percent <= max_change and 
                    volume_usdt >= self.min_volume_usdt):
                    
                    candidates.append({
                        'symbol': symbol.replace('USDT', '/USDT'),
                        'change_percent_24h': round(change_percent, 2),
                        'volume_24h': round(volume_usdt, 0),
                        'price': last_price,
                        'high_24h': float(ticker.get('high', 0)),
                        'low_24h': float(ticker.get('low', 0))
                    })
            
            logger.info(f"Stage 1: {len(candidates)} candidates (24h movers with {min_change}%+ gain)")
            
            # STAGE 2: Multi-tier validation (check all 3 tiers, prioritize earliest)
            fresh_pumpers = []
            tier_counts = {'5m': 0, '15m': 0, '30m': 0}
            
            for candidate in candidates:
                symbol = candidate['symbol']
                pump_data = None
                
                # Check TIER 1 (5m - Ultra-Early) first
                pump_data = await self.validate_fresh_5m_pump(symbol)
                if pump_data and pump_data.get('is_fresh_pump'):
                    tier_counts['5m'] += 1
                    candidate['fresh_pump_data'] = pump_data
                    candidate['tier'] = '5m'
                    candidate['change_percent'] = pump_data['candle_change_percent']
                    fresh_pumpers.append(candidate)
                    logger.info(f"⚡ ULTRA-EARLY (5m): {symbol} → +{pump_data['candle_change_percent']}% (Vol: {pump_data['volume_ratio']}x)")
                    continue
                
                # Check TIER 2 (15m - Early) if no 5m pump
                pump_data = await self.validate_fresh_15m_pump(symbol)
                if pump_data and pump_data.get('is_fresh_pump'):
                    tier_counts['15m'] += 1
                    candidate['fresh_pump_data'] = pump_data
                    candidate['tier'] = '15m'
                    candidate['change_percent'] = pump_data['candle_change_percent']
                    fresh_pumpers.append(candidate)
                    logger.info(f"🔥 EARLY (15m): {symbol} → +{pump_data['candle_change_percent']}% (Vol: {pump_data['volume_ratio']}x)")
                    continue
                
                # Check TIER 3 (30m - Fresh) if no 15m pump
                pump_data = await self.validate_fresh_30m_pump(symbol)
                if pump_data and pump_data.get('is_fresh_pump'):
                    tier_counts['30m'] += 1
                    candidate['fresh_pump_data'] = pump_data
                    candidate['tier'] = '30m'
                    candidate['change_percent'] = pump_data['candle_change_percent']
                    fresh_pumpers.append(candidate)
                    logger.info(f"✅ FRESH (30m): {symbol} → +{pump_data['candle_change_percent']}% (Vol: {pump_data['volume_ratio']}x)")
            
            # Sort by tier priority (5m > 15m > 30m) then by pump % within each tier
            tier_priority = {'5m': 1, '15m': 2, '30m': 3}
            fresh_pumpers.sort(key=lambda x: (tier_priority[x['tier']], -x['change_percent']))
            
            logger.info(f"Stage 2: {len(fresh_pumpers)} FRESH pumps - 5m:{tier_counts['5m']}, 15m:{tier_counts['15m']}, 30m:{tier_counts['30m']}")
            return fresh_pumpers[:limit]
            
        except Exception as e:
            logger.error(f"Error fetching fresh pumpers: {e}")
            return []
    
    async def analyze_momentum(self, symbol: str) -> Optional[Dict]:
        """
        PROFESSIONAL-GRADE ENTRY ANALYSIS for Top Gainers
        
        Filters:
        1. ✅ Pullback entries (price near EMA9, not chasing tops)
        2. ✅ Volume confirmation (>1.3x average = real money flowing)
        3. ✅ Overextension checks (avoid entries >2.5% from EMA)
        4. ✅ RSI confirmation (30-70 range, no extreme overbought/oversold)
        5. ✅ Recent momentum (last 3 candles direction)
        
        Returns:
            {
                'direction': 'LONG' or 'SHORT',
                'confidence': 0-100,
                'entry_price': float,
                'reason': str (detailed entry justification)
            }
        """
        try:
            # 🔥 QUALITY CHECK #1: Liquidity Validation
            liquidity_check = await self.check_liquidity(symbol)
            if not liquidity_check['is_liquid']:
                logger.info(f"{symbol} SHORTS SKIPPED - {liquidity_check['reason']}")
                return None
            
            # 🔥 FRESHNESS CHECK: Skip fresh pumps for SHORTS (let LONGS handle them!)
            # Check all 3 tiers: 5m, 15m, 30m - if ANY are fresh, this is a LONG opportunity
            is_fresh_5m = await self.validate_fresh_5m_pump(symbol)
            is_fresh_15m = await self.validate_fresh_15m_pump(symbol)
            is_fresh_30m = await self.validate_fresh_30m_pump(symbol)
            
            # Check if ANY tier confirms this is a fresh pump (not just truthy dict)
            fresh_5m_confirmed = is_fresh_5m and is_fresh_5m.get('is_fresh_pump') == True
            fresh_15m_confirmed = is_fresh_15m and is_fresh_15m.get('is_fresh_pump') == True
            fresh_30m_confirmed = is_fresh_30m and is_fresh_30m.get('is_fresh_pump') == True
            
            if fresh_5m_confirmed or fresh_15m_confirmed or fresh_30m_confirmed:
                tier = "5m" if fresh_5m_confirmed else ("15m" if fresh_15m_confirmed else "30m")
                logger.info(f"🟢 {symbol} is FRESH PUMP ({tier}) - Skipping SHORTS, will generate LONG signal instead!")
                return None
            
            # Fetch candles with sufficient history for accurate analysis
            candles_5m = await self.fetch_candles(symbol, '5m', limit=50)
            candles_15m = await self.fetch_candles(symbol, '15m', limit=50)
            
            if len(candles_5m) < 30 or len(candles_15m) < 30:
                return None
            
            # 🔥 QUALITY CHECK #2: Anti-Manipulation Filter
            manipulation_check = await self.check_manipulation_risk(symbol, candles_5m)
            if not manipulation_check['is_safe']:
                logger.info(f"{symbol} SHORTS SKIPPED - Manipulation risk: {', '.join(manipulation_check['flags'])}")
                return None
            
            # Convert to DataFrame for candle size analysis
            import pandas as pd
            df_5m = pd.DataFrame(candles_5m, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            
            # 🚨 CRITICAL CHECK: Skip oversized candles (RELAXED: 5% threshold for parabolic moves)
            if self._is_candle_oversized(df_5m, max_body_percent=5.0):
                logger.info(f"{symbol} SKIPPED - Current candle is oversized (prevents poor entries)")
                return None
            
            # Extract price and volume data
            closes_5m = [c[4] for c in candles_5m]
            volumes_5m = [c[5] for c in candles_5m]
            closes_15m = [c[4] for c in candles_15m]
            
            # Current price and previous prices for momentum
            current_price = closes_5m[-1]
            prev_close = closes_5m[-2]
            prev_prev_close = closes_5m[-3] if len(closes_5m) >= 3 else prev_close
            high_5m = candles_5m[-1][2]
            low_5m = candles_5m[-1][3]
            
            # Extract previous candle data for pullback detection
            prev_open = candles_5m[-2][1]
            prev_high = candles_5m[-2][2]
            prev_low = candles_5m[-2][3]
            
            # Calculate previous candle direction
            prev_candle_bullish = prev_close > prev_open
            prev_candle_bearish = prev_close < prev_open
            
            # Current candle direction
            current_open = candles_5m[-1][1]
            current_candle_bullish = current_price > current_open
            current_candle_bearish = current_price < current_open
            
            # Calculate EMAs (trend identification)
            ema9_5m = self._calculate_ema(closes_5m, 9)
            ema21_5m = self._calculate_ema(closes_5m, 21)
            ema9_15m = self._calculate_ema(closes_15m, 9)
            ema21_15m = self._calculate_ema(closes_15m, 21)
            
            # Calculate RSI (momentum strength)
            rsi_5m = self._calculate_rsi(closes_5m, 14)
            
            # ===== VOLUME ANALYSIS (Critical for top gainers) =====
            avg_volume = sum(volumes_5m[-20:-1]) / 19  # Last 20 candles excluding current
            current_volume = volumes_5m[-1]
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0
            
            # ===== PRICE TO EMA DISTANCE (Pullback detection) =====
            price_to_ema9_dist = ((current_price - ema9_5m) / ema9_5m) * 100
            price_to_ema21_dist = ((current_price - ema21_5m) / ema21_5m) * 100
            
            # Is price near EMA9? (STRICT pullback entry zone)
            is_near_ema9 = abs(price_to_ema9_dist) < 1.0  # Within 1% of EMA9 (TIGHTENED)
            is_near_ema21 = abs(price_to_ema21_dist) < 1.5  # Within 1.5% of EMA21
            
            # ===== TREND ALIGNMENT (Multi-timeframe confirmation) =====
            bullish_5m = ema9_5m > ema21_5m
            bullish_15m = ema9_15m > ema21_15m
            
            # ===== RECENT MOMENTUM (Last 3 candles direction) =====
            recent_candles = closes_5m[-4:]
            bullish_momentum = recent_candles[-1] > recent_candles[-3]  # Higher highs
            bearish_momentum = recent_candles[-1] < recent_candles[-3]  # Lower lows
            
            # ===== OVEREXTENSION CHECK (Avoid buying tops / selling bottoms) =====
            is_overextended_up = price_to_ema9_dist > 2.5  # >2.5% above EMA9
            is_overextended_down = price_to_ema9_dist < -2.5  # >2.5% below EMA9
            
            
            # ═══════════════════════════════════════════════════════
            # STRATEGY 1: OVEREXTENDED SHORT - Catch coins STILL PUMPING but ready to dump
            # ═══════════════════════════════════════════════════════
            # For coins at 25%+ that are STILL bullish but extremely overbought
            # This catches the TOP before the dump starts (aggressive mean reversion)
            # ═══════════════════════════════════════════════════════
            if bullish_5m and bullish_15m:
                # Both timeframes STILL bullish but coin may be dangerously overextended
                # Instead of WAITING for dump to start, we SHORT THE TOP aggressively
                
                # OVEREXTENDED SHORT CONDITIONS (RELAXED for real market conditions):
                # 1. RSI 60+ (catch overbought earlier, before extreme peak)
                # 2. Volume 1.0x+ (normal volume acceptable - coins pump on avg volume)
                # 3. Price 3%+ above EMA9 (catch extended moves earlier)
                is_overextended_short = (
                    rsi_5m >= 60 and  # Overbought early (was 70 - too strict!)
                    volume_ratio >= 1.0 and  # Normal volume OK (was 2.0x - unrealistic!)
                    price_to_ema9_dist >= 3.0  # Extended above EMA9 (was 10% - too extreme!)
                )
                
                if is_overextended_short:
                    logger.info(f"{symbol} ✅ OVEREXTENDED SHORT: RSI {rsi_5m:.0f} (overbought!) | Vol {volume_ratio:.1f}x | +{price_to_ema9_dist:.1f}% above EMA9 | Shorting the top!")
                    return {
                        'direction': 'SHORT',
                        'confidence': 88,
                        'entry_price': current_price,
                        'reason': f'🎯 OVEREXTENDED TOP | RSI {rsi_5m:.0f} overbought | Vol {volume_ratio:.1f}x | +{price_to_ema9_dist:.1f}% extended | Mean reversion play!'
                    }
                
                # RARE LONG EXCEPTION: Massive volume breakout (3.5x+) with perfect setup
                # This is "out of the ordinary" - institutional-level buying pressure
                elif volume_ratio >= 3.5 and rsi_5m > 50 and rsi_5m < 65 and not is_overextended_up and is_near_ema9:
                    logger.info(f"{symbol} ✅ EXCEPTIONAL LONG: Massive volume {volume_ratio:.1f}x + perfect EMA9 entry")
                    return {
                        'direction': 'LONG',
                        'confidence': 95,
                        'entry_price': current_price,
                        'reason': f'🚀 EXCEPTIONAL VOLUME {volume_ratio:.1f}x | RSI: {rsi_5m:.0f} | Perfect EMA9 pullback - RARE LONG!'
                    }
                
                # SKIP: Not overextended enough for SHORT, not exceptional enough for LONG
                else:
                    logger.info(f"{symbol} Still pumping but NOT overextended yet: Vol {volume_ratio:.1f}x, RSI {rsi_5m:.0f}, Distance {price_to_ema9_dist:+.1f}% (need RSI 60+, Vol 1.0x+, Distance 3%+)")
                    return None
            
            
            # ═══════════════════════════════════════════════════════
            # STRATEGY 2: SHORT - Mean reversion on failed pumps (TWO ENTRY PATHS)
            # ═══════════════════════════════════════════════════════
            elif not bullish_5m and not bullish_15m:
                # Calculate current candle size for strong dump detection
                current_candle_size = abs((current_price - current_open) / current_open * 100)
                
                # ═════ ENTRY PATH 1: STRONG DUMP (Direct Entry - No Pullback Needed) ═════
                # For violent dumps with high volume, enter immediately
                is_strong_dump = (
                    current_candle_bearish and 
                    current_candle_size >= 1.0 and  # At least 1% dump candle
                    volume_ratio >= 1.5 and  # Strong volume
                    40 <= rsi_5m <= 65  # RSI range for strong dumps (wider)
                )
                
                if is_strong_dump:
                    logger.info(f"{symbol} ✅ STRONG DUMP DETECTED: {current_candle_size:.2f}% red candle | Vol: {volume_ratio:.1f}x | RSI: {rsi_5m:.0f}")
                    return {
                        'direction': 'SHORT',
                        'confidence': 90,
                        'entry_price': current_price,
                        'reason': f'🔥 STRONG DUMP | {current_candle_size:.1f}% red candle | Vol: {volume_ratio:.1f}x | RSI: {rsi_5m:.0f} | Direct entry'
                    }
                
                # ═════ ENTRY PATH 2: RESUMPTION PATTERN (Safer, After Pullback) ═════
                has_resumption_pattern = False
                
                # Check if we have: prev-prev RED (dump) → prev GREEN (pullback) → current RED (resumption)
                if len(closes_5m) >= 3 and len(candles_5m) >= 3:
                    prev_prev_open = candles_5m[-3][1]
                    prev_prev_close = closes_5m[-3]
                    
                    # Calculate candle sizes
                    prev_prev_bearish = prev_prev_close < prev_prev_open
                    prev_candle_size = abs((prev_close - prev_open) / prev_open * 100)
                    
                    # PERFECT PATTERN: Red dump → Green pullback → Red resumption
                    if prev_prev_bearish and prev_candle_bullish and current_candle_bearish:
                        prev_prev_size = abs((prev_prev_close - prev_prev_open) / prev_prev_open * 100)
                        
                        # Pullback must be smaller than dump, and current is resuming down
                        if prev_prev_size > prev_candle_size * 1.5 and current_candle_bearish:
                            has_resumption_pattern = True
                            logger.info(f"{symbol} ✅ RESUMPTION PATTERN: Dump {prev_prev_size:.2f}% → Pullback {prev_candle_size:.2f}% → Resuming down")
                
                # Resumption entry: More relaxed than before
                if has_resumption_pattern and rsi_5m >= 45 and rsi_5m <= 70 and volume_ratio >= 1.0:
                    return {
                        'direction': 'SHORT',
                        'confidence': 95,
                        'entry_price': current_price,
                        'reason': f'🎯 RESUMPTION SHORT | Entered AFTER pullback | Vol: {volume_ratio:.1f}x | RSI: {rsi_5m:.0f} | Clean entry'
                    }
                
                # SKIP - no valid entry pattern
                else:
                    skip_reason = []
                    if not has_resumption_pattern and not is_strong_dump:
                        skip_reason.append("No entry pattern (need: strong dump OR resumption)")
                    if not current_candle_bearish:
                        skip_reason.append("Current candle not red")
                    if rsi_5m < 40 or rsi_5m > 70:
                        skip_reason.append(f"RSI {rsi_5m:.0f} out of range (need 40-70)")
                    if volume_ratio < 1.0:
                        skip_reason.append(f"Low volume {volume_ratio:.1f}x (need 1.0x+)")
                    
                    logger.info(f"{symbol} SHORT SKIPPED: {', '.join(skip_reason)}")
                    return None
            
            
            # ═══════════════════════════════════════════════════════
            # SPECIAL CASE: Mean Reversion SHORT on Parabolic Pumps (ULTRA PRECISION)
            # Even if 5m is still bullish, if price is EXTREMELY overextended
            # and showing reversal signs, take the SHORT (top gainer specialty!)
            # ═══════════════════════════════════════════════════════
            elif bullish_5m and not bullish_15m:
                # 15m already bearish but 5m lagging - this is the reversal point
                # Perfect for catching top gainer dumps (like PIPPIN +120% starting to roll over)
                price_extension = price_to_ema9_dist
                
                # 🎯 ULTRA PRECISION CHECK: Look for topping pattern (previous green, current doji/red)
                has_topping_pattern = False
                if prev_candle_bullish and (current_candle_bearish or abs(current_price - current_open) / current_open * 100 < 0.5):
                    # Previous was green pump, current is red or small doji - reversal starting!
                    has_topping_pattern = True
                    logger.info(f"{symbol} ✅ TOPPING PATTERN: Pump slowing, reversal forming")
                
                # ONLY TAKE PARABOLIC SHORTS WITH: Topping pattern + STRICT RSI + VOLUME
                if price_extension > 2.0 and rsi_5m >= 60 and rsi_5m <= 75 and has_topping_pattern and volume_ratio >= 1.5:
                    # ULTRA STRICT: Need >2% extension, topping pattern, RSI 60-75, 1.5x+ volume
                    return {
                        'direction': 'SHORT',
                        'confidence': 90,
                        'entry_price': current_price,
                        'reason': f'🎯 ULTRA PRECISION PARABOLIC | {price_extension:+.1f}% overextended | Topping confirmed | Vol: {volume_ratio:.1f}x | RSI: {rsi_5m:.0f}'
                    }
                else:
                    skip_reason = []
                    if not has_topping_pattern:
                        skip_reason.append("No topping pattern")
                    if price_extension <= 2.0:
                        skip_reason.append(f"Not extended enough ({price_extension:+.1f}%, need >2%)")
                    if not (60 <= rsi_5m <= 75):
                        skip_reason.append(f"RSI {rsi_5m:.0f} out of range (need 60-75)")
                    if volume_ratio < 1.5:
                        skip_reason.append(f"Low volume {volume_ratio:.1f}x (need 1.5x+)")
                    
                    logger.info(f"{symbol} PARABOLIC SKIPPED: {', '.join(skip_reason)}")
                    return None
            
            
            # ═══════════════════════════════════════════════════════
            # STRATEGY 3: MIXED SIGNALS - Early Reversal Catch (NEW!)
            # 5m bearish + 15m bullish = Coin starting to dump, 15m hasn't caught up yet
            # Perfect for catching reversals EARLY before everyone sees it
            # ═══════════════════════════════════════════════════════
            elif not bullish_5m and bullish_15m:
                # 5m turned bearish but 15m still bullish = Early reversal signal!
                # This catches dumps BEFORE the 15m confirms (super early entry)
                
                # Check for early reversal pattern
                is_early_reversal = (
                    current_candle_bearish and  # Current candle is red
                    bearish_momentum and  # Recent momentum turning down
                    rsi_5m >= 50 and rsi_5m <= 70 and  # RSI showing weakness but not oversold
                    volume_ratio >= 1.2  # Volume confirming the move
                )
                
                if is_early_reversal:
                    logger.info(f"{symbol} ✅ EARLY REVERSAL: 5m bearish, 15m still bullish | Vol: {volume_ratio:.1f}x | RSI: {rsi_5m:.0f}")
                    return {
                        'direction': 'SHORT',
                        'confidence': 85,
                        'entry_price': current_price,
                        'reason': f'⚡ EARLY REVERSAL | 5m turning bearish | Vol: {volume_ratio:.1f}x | RSI: {rsi_5m:.0f} | Caught early!'
                    }
                else:
                    logger.info(f"{symbol} MIXED SIGNALS - not clear enough (5m bearish: {not bullish_5m}, 15m bullish: {bullish_15m}, RSI: {rsi_5m:.0f}, Vol: {volume_ratio:.1f}x)")
                    return None
            
            # Other mixed signals - skip (e.g., 5m bullish + 15m bearish = lagging, not early)
            else:
                logger.info(f"{symbol} MIXED SIGNALS - skipping (5m bullish: {bullish_5m}, 15m bullish: {bullish_15m})")
                return None
                
        except Exception as e:
            logger.error(f"Error analyzing momentum for {symbol}: {e}")
            return None
    
    async def analyze_early_pump_long(self, symbol: str) -> Optional[Dict]:
        """
        Analyze PUMPING coins for LONG entries (5%+ gains, NO MAX)
        
        KEY: Wait for RETRACEMENT before entering (like shorts wait for pullback)
        
        Entry criteria:
        1. ✅ Strong volume surge (2x+ average)
        2. ✅ Bullish momentum building (EMA9 > EMA21, both timeframes)
        3. ✅ MUST have retracement - price near/below EMA9 (NO CHASING!)
        4. ✅ RSI 45-70 (momentum without overbought)
        5. ✅ Resumption pattern (red pullback → green continuation)
        
        Returns signal for LONG entry or None if criteria not met
        """
        try:
            logger.info(f"🟢 ANALYZING {symbol} FOR LONGS...")
            
            # 🔥 QUALITY CHECK #1: Liquidity Validation
            liquidity_check = await self.check_liquidity(symbol)
            if not liquidity_check['is_liquid']:
                logger.info(f"  ❌ {symbol} REJECTED - {liquidity_check['reason']}")
                return None
            logger.info(f"  ✅ {symbol} - Liquidity OK (spread: {liquidity_check.get('spread_percent', 0):.2f}%)")
            
            # Fetch candles
            candles_5m = await self.fetch_candles(symbol, '5m', limit=50)
            candles_15m = await self.fetch_candles(symbol, '15m', limit=50)
            
            if len(candles_5m) < 30 or len(candles_15m) < 30:
                return None
            
            # 🔥 QUALITY CHECK #2: Anti-Manipulation Filter
            manipulation_check = await self.check_manipulation_risk(symbol, candles_5m)
            if not manipulation_check['is_safe']:
                logger.info(f"  ❌ {symbol} REJECTED - Manipulation risk: {', '.join(manipulation_check['flags'])}")
                return None
            logger.info(f"  ✅ {symbol} - Anti-manipulation OK")
            
            # 🔥 FRESH CHECK: Pump within last 2 hours (relaxed from 60 min)
            # Use 5m candles for tracking (24 candles = 120 minutes)
            if len(candles_5m) >= 25:
                price_120m_ago = candles_5m[-25][4]  # Close price 120 minutes ago (24 candles back + current)
                current_price_check = candles_5m[-1][4]  # Current price
                
                if price_120m_ago > 0:
                    pump_120m_percent = ((current_price_check - price_120m_ago) / price_120m_ago) * 100
                    
                    if pump_120m_percent < 5.0:
                        logger.info(f"  ❌ {symbol} REJECTED - Only +{pump_120m_percent:.1f}% in last 2h (need 5%+ for FRESH)")
                        return None
                    logger.info(f"  ✅ {symbol} - FRESH PUMP: +{pump_120m_percent:.1f}% within 2 hours!")
            
            import pandas as pd
            df_5m = pd.DataFrame(candles_5m, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            
            # Note: Removed oversized candle check for LONGS - big green candles are GOOD for momentum!
            
            # Extract data
            closes_5m = [c[4] for c in candles_5m]
            volumes_5m = [c[5] for c in candles_5m]
            closes_15m = [c[4] for c in candles_15m]
            
            current_price = closes_5m[-1]
            current_open = candles_5m[-1][1]
            current_candle_bullish = current_price > current_open
            
            # Calculate EMAs
            ema9_5m = self._calculate_ema(closes_5m, 9)
            ema21_5m = self._calculate_ema(closes_5m, 21)
            ema9_15m = self._calculate_ema(closes_15m, 9)
            ema21_15m = self._calculate_ema(closes_15m, 21)
            
            # Calculate RSI
            rsi_5m = self._calculate_rsi(closes_5m, 14)
            
            # Volume analysis
            avg_volume = sum(volumes_5m[-20:-1]) / 19
            current_volume = volumes_5m[-1]
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0
            
            # Price to EMA distance
            price_to_ema9_dist = ((current_price - ema9_5m) / ema9_5m) * 100
            
            # Trend alignment (MUST be bullish on both timeframes)
            bullish_5m = ema9_5m > ema21_5m
            bullish_15m = ema9_15m > ema21_15m
            
            # Recent momentum
            recent_candles = closes_5m[-4:]
            bullish_momentum = recent_candles[-1] > recent_candles[-3]
            
            
            # ═══════════════════════════════════════════════════════
            # LONG STRATEGY: Wait for RETRACEMENT (like shorts wait for pullback!)
            # ═══════════════════════════════════════════════════════
            
            if not (bullish_5m and bullish_15m):
                logger.info(f"  ❌ {symbol} REJECTED - Trend not aligned (5m bullish: {bullish_5m}, 15m bullish: {bullish_15m})")
                return None
            logger.info(f"  ✅ {symbol} - Bullish trend on BOTH timeframes")
            
            # Calculate candle sizes for retracement detection
            current_candle_size = abs((current_price - current_open) / current_open * 100)
            prev_close = closes_5m[-2]
            prev_open = candles_5m[-2][1]
            prev_candle_bullish = prev_close > prev_open
            prev_candle_bearish = prev_close < prev_open
            
            # ENTRY CONDITION 1: EMA9 PULLBACK LONG (BEST - wait for retracement!)
            # Price pulled back to/below EMA9, ready to resume UP
            is_at_or_below_ema9 = price_to_ema9_dist <= 3.0  # 🔥 ULTRA RELAXED: ±3% from EMA9 (catch more entries)
            
            logger.info(f"  📊 {symbol} - Price to EMA9: {price_to_ema9_dist:+.2f}%, Vol: {volume_ratio:.2f}x, RSI: {rsi_5m:.0f}")
            if (is_at_or_below_ema9 and
                volume_ratio >= 1.3 and  # 🔥 ULTRA RELAXED from 1.8x to 1.3x (more realistic volume)
                rsi_5m >= 40 and rsi_5m <= 75 and  # 🔥 ULTRA RELAXED: wider RSI range
                bullish_momentum and
                current_candle_bullish):  # Green candle resuming
                
                logger.info(f"{symbol} ✅ EMA9 PULLBACK LONG: Near EMA9 ({price_to_ema9_dist:+.1f}%) | Vol {volume_ratio:.1f}x | RSI {rsi_5m:.0f}")
                return {
                    'direction': 'LONG',
                    'confidence': 95,
                    'entry_price': current_price,
                    'reason': f'📈 EMA9 PULLBACK | Retracement entry | Vol {volume_ratio:.1f}x | RSI {rsi_5m:.0f}'
                }
            
            # ENTRY CONDITION 2: RESUMPTION PATTERN (Safer - after pullback)
            # Like shorts: green pump → red pullback → green resumption
            has_resumption_pattern = False
            if len(closes_5m) >= 3 and len(candles_5m) >= 3:
                prev_prev_open = candles_5m[-3][1]
                prev_prev_close = closes_5m[-3]
                prev_prev_bullish = prev_prev_close > prev_prev_open
                
                # PERFECT PATTERN: Green pump → Red pullback → Green resumption
                if prev_prev_bullish and prev_candle_bearish and current_candle_bullish:
                    prev_prev_size = abs((prev_prev_close - prev_prev_open) / prev_prev_open * 100)
                    prev_candle_size = abs((prev_close - prev_open) / prev_open * 100)
                    
                    # Pullback must be smaller than pump, and current is resuming UP
                    if prev_prev_size > prev_candle_size * 1.5:
                        has_resumption_pattern = True
                        logger.info(f"{symbol} ✅ RESUMPTION PATTERN: Pump {prev_prev_size:.2f}% → Pullback {prev_candle_size:.2f}% → Resuming UP")
            
            if (has_resumption_pattern and 
                rsi_5m >= 40 and rsi_5m <= 75 and  # 🔥 ULTRA RELAXED RSI
                volume_ratio >= 1.3 and  # 🔥 ULTRA RELAXED volume
                price_to_ema9_dist >= -2.0 and price_to_ema9_dist <= 5.0):  # 🔥 Wider EMA9 range
                
                return {
                    'direction': 'LONG',
                    'confidence': 90,
                    'entry_price': current_price,
                    'reason': f'🎯 RESUMPTION LONG | Entered AFTER pullback | Vol {volume_ratio:.1f}x | RSI {rsi_5m:.0f}'
                }
            
            # ENTRY CONDITION 3: STRONG PUMP (Direct Entry - catch early momentum)
            # For violent pumps with huge volume, enter immediately (like shorts enter on strong dump)
            is_strong_pump = (
                current_candle_bullish and 
                current_candle_size >= 1.0 and  # 🔥 RELAXED from 1.5% to 1.0% (catch smaller pumps)
                volume_ratio >= 2.0 and  # 🔥 RELAXED from 3.0x to 2.0x (more realistic)
                35 <= rsi_5m <= 85 and  # 🔥 WIDENED RSI range for momentum trades
                price_to_ema9_dist >= -2.0 and price_to_ema9_dist <= 4.0  # 🔥 Wider range
            )
            
            if is_strong_pump:
                logger.info(f"{symbol} ✅ STRONG PUMP DETECTED: {current_candle_size:.2f}% green candle | Vol: {volume_ratio:.1f}x | RSI: {rsi_5m:.0f}")
                return {
                    'direction': 'LONG',
                    'confidence': 88,
                    'entry_price': current_price,
                    'reason': f'🔥 STRONG PUMP | {current_candle_size:.1f}% green candle | Vol: {volume_ratio:.1f}x | RSI: {rsi_5m:.0f}'
                }
            
            # SKIP - no valid LONG entry (relaxed filters for momentum trades)
            skip_reason = []
            if price_to_ema9_dist > 8.0:
                skip_reason.append(f"Too far from EMA9 ({price_to_ema9_dist:+.1f}%, need ≤8%)")
            # Pullback pattern is now OPTIONAL - strong pumps are valid too!
            if volume_ratio < 1.0:  # 🔥 BALANCED: 1.0x minimum (must have average volume)
                skip_reason.append(f"Low volume {volume_ratio:.1f}x (need 1.0x+)")
            if not (35 <= rsi_5m <= 85):  # 🔥 WIDENED: 35-85 range for momentum
                skip_reason.append(f"RSI {rsi_5m:.0f} out of range (need 35-85)")
            
            logger.info(f"{symbol} LONG SKIPPED: {', '.join(skip_reason)}")
            return None
            
        except Exception as e:
            logger.error(f"Error analyzing early pump for {symbol}: {e}")
            return None
    
    def _calculate_rsi(self, prices: List[float], period: int = 14) -> float:
        """Calculate Relative Strength Index (RSI)"""
        if len(prices) < period + 1:
            return 50.0  # Neutral if not enough data
        
        deltas = [prices[i] - prices[i-1] for i in range(1, len(prices))]
        gains = [d if d > 0 else 0 for d in deltas]
        losses = [-d if d < 0 else 0 for d in deltas]
        
        avg_gain = sum(gains[-period:]) / period
        avg_loss = sum(losses[-period:]) / period
        
        if avg_loss == 0:
            return 100.0
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def _calculate_ema(self, prices: List[float], period: int) -> float:
        """Calculate Exponential Moving Average"""
        if len(prices) < period:
            return sum(prices) / len(prices)
        
        multiplier = 2 / (period + 1)
        ema = sum(prices[:period]) / period
        
        for price in prices[period:]:
            ema = (price - ema) * multiplier + ema
        
        return ema
    
    def _is_candle_oversized(self, df, max_body_percent: float = 2.5) -> bool:
        """
        Check if current candle is oversized (prevents entering on parabolic dumps/pumps)
        
        Args:
            df: DataFrame with OHLC data
            max_body_percent: Maximum allowed candle body size (default 2.5%)
        
        Returns:
            True if candle is too big (should skip entry), False if safe to enter
        """
        if len(df) < 2:
            return False
        
        current_candle = df.iloc[-1]
        open_price = current_candle['open']
        close_price = current_candle['close']
        
        # Calculate candle body size as percentage
        body_percent = abs((close_price - open_price) / open_price * 100)
        
        # Also check against average candle size (last 10 candles)
        if len(df) >= 10:
            avg_body_sizes = []
            for i in range(-10, 0):
                candle = df.iloc[i]
                candle_body = abs((candle['close'] - candle['open']) / candle['open'] * 100)
                avg_body_sizes.append(candle_body)
            
            avg_body = sum(avg_body_sizes) / len(avg_body_sizes)
            
            # If current candle is > 2.5x average, it's oversized
            if body_percent > avg_body * 2.5:
                logger.info(f"⚠️ Oversized candle: {body_percent:.2f}% vs avg {avg_body:.2f}% (2.5x threshold)")
                return True
        
        # Absolute threshold: reject if candle body > max_body_percent
        if body_percent > max_body_percent:
            logger.info(f"⚠️ Candle too large: {body_percent:.2f}% body (max: {max_body_percent}%)")
            return True
        
        return False
    
    async def generate_top_gainer_signal(
        self, 
        min_change_percent: float = 10.0,
        max_symbols: int = 5
    ) -> Optional[Dict]:
        """
        Generate trading signal from top gainers
        
        OPTIMIZED FOR SHORTS: Prioritizes mean reversion on big pumps (10%+ gains)
        - Scans more symbols (5 vs 3) to find best short setups
        - Higher min_change (10% vs 5%) = better reversal candidates
        - Parabolic reversal detection prioritized for 50%+ pumps
        
        Returns:
            {
                'symbol': str,
                'direction': 'LONG' or 'SHORT',
                'entry_price': float,
                'stop_loss': float,
                'take_profit': float,
                'confidence': int,
                'reasoning': str,
                'trade_type': 'TOP_GAINER'
            }
        """
        try:
            # Get top gainers (optimized for shorts - higher % gains = better reversal candidates)
            gainers = await self.get_top_gainers(limit=max_symbols, min_change_percent=min_change_percent)
            
            if not gainers:
                logger.info("No top gainers found meeting criteria")
                return None
            
            # Clean up expired cooldowns
            now = datetime.utcnow()
            expired = [sym for sym, expires in shorts_cooldown.items() if expires <= now]
            for sym in expired:
                del shorts_cooldown[sym]
            
            # PRIORITY 1: Look for parabolic reversal shorts (biggest pumps first)
            # These are the BEST opportunities - coins that pumped 50%+ and are rolling over
            for gainer in gainers:
                if gainer['change_percent'] >= 50.0:  # Extreme pumps (50%+)
                    symbol = gainer['symbol']
                    
                    # Check if symbol is in cooldown (lost SHORT recently)
                    if symbol in shorts_cooldown:
                        remaining_min = (shorts_cooldown[symbol] - now).total_seconds() / 60
                        logger.info(f"⏰ {symbol} SKIPPED - SHORT cooldown active ({remaining_min:.0f} min left)")
                        continue
                    
                    logger.info(f"🎯 Analyzing PARABOLIC candidate: {symbol} @ +{gainer['change_percent']}%")
                    
                    momentum = await self.analyze_momentum(symbol)
                    
                    if momentum and momentum['direction'] == 'SHORT' and 'PARABOLIC REVERSAL' in momentum['reason']:
                        logger.info(f"✅ PARABOLIC REVERSAL SHORT found: {symbol}")
                        # Build signal and return immediately (highest priority!)
                        entry_price = momentum['entry_price']
                        
                        # Triple TPs for parabolic reversals (1:1, 1:2, and 1:3 R:R) - These dump HARD!
                        stop_loss = entry_price * (1 + 4.0 / 100)  # 20% loss at 5x
                        take_profit_1 = entry_price * (1 - 4.0 / 100)  # TP1: 20% profit at 5x (1:1 R:R)
                        take_profit_2 = entry_price * (1 - 8.0 / 100)  # TP2: 40% profit at 5x (1:2 R:R)
                        take_profit_3 = entry_price * (1 - 12.0 / 100)  # TP3: 60% profit at 5x (1:3 R:R) 🚀
                        
                        return {
                            'symbol': symbol,
                            'direction': 'SHORT',
                            'entry_price': entry_price,
                            'stop_loss': stop_loss,
                            'take_profit': take_profit_1,
                            'take_profit_1': take_profit_1,
                            'take_profit_2': take_profit_2,
                            'take_profit_3': take_profit_3,
                            'confidence': momentum['confidence'],
                            'reasoning': f"Top Gainer: {gainer['change_percent']}% in 24h | {momentum['reason']}",
                            'trade_type': 'TOP_GAINER',
                            'leverage': 5,
                            '24h_change': gainer['change_percent'],
                            '24h_volume': gainer['volume_24h'],
                            'is_parabolic_reversal': True
                        }
            
            # PRIORITY 2: Regular analysis (shorts preferred, then longs)
            for gainer in gainers:
                symbol = gainer['symbol']
                
                # Check if symbol is in cooldown (lost SHORT recently)
                if symbol in shorts_cooldown:
                    remaining_min = (shorts_cooldown[symbol] - now).total_seconds() / 60
                    logger.info(f"⏰ {symbol} SKIPPED - SHORT cooldown active ({remaining_min:.0f} min left)")
                    continue
                
                # Analyze momentum
                momentum = await self.analyze_momentum(symbol)
                
                if not momentum:
                    continue
                
                entry_price = momentum['entry_price']
                
                # Check if this is a parabolic reversal SHORT
                is_parabolic_reversal = (
                    momentum['direction'] == 'SHORT' and 
                    'PARABOLIC REVERSAL' in momentum['reason']
                )
                
                # Calculate TP/SL with 1:1, 1:2, and 1:3 Risk-to-Reward
                # Parabolic reversals get 3 TPs (they dump HARD!)
                
                if momentum['direction'] == 'LONG':
                    # LONG: Dual TPs (5% and 10% price targets)
                    stop_loss = entry_price * (1 - 4.0 / 100)  # 20% loss at 5x
                    take_profit_1 = entry_price * (1 + 5.0 / 100)  # TP1: 5% price move (25% profit at 5x)
                    take_profit_2 = entry_price * (1 + 10.0 / 100)  # TP2: 10% price move (50% profit at 5x)
                    take_profit_3 = None  # No TP3 for longs
                    
                else:  # SHORT
                    # SHORT: Single TP at 8% price move (40% profit at 5x)
                    stop_loss = entry_price * (1 + 4.0 / 100)  # 20% loss at 5x
                    take_profit_1 = entry_price * (1 - 8.0 / 100)  # TP: 8% price move (40% profit at 5x)
                    take_profit_2 = None  # No TP2 for shorts
                    take_profit_3 = None  # No TP3 for shorts
                
                return {
                    'symbol': symbol,
                    'direction': momentum['direction'],
                    'entry_price': entry_price,
                    'stop_loss': stop_loss,
                    'take_profit': take_profit_1,  # Backward compatible
                    'take_profit_1': take_profit_1,
                    'take_profit_2': take_profit_2,
                    'take_profit_3': take_profit_3,
                    'confidence': momentum['confidence'],
                    'reasoning': f"Top Gainer: {gainer['change_percent']}% in 24h | {momentum['reason']}",
                    'trade_type': 'TOP_GAINER',
                    'leverage': 5,  # Fixed 5x leverage for top gainers
                    '24h_change': gainer['change_percent'],
                    '24h_volume': gainer['volume_24h'],
                    'is_parabolic_reversal': is_parabolic_reversal
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Error generating top gainer signal: {e}")
            return None
    
    async def generate_parabolic_dump_signal(
        self,
        min_change_percent: float = 50.0,
        max_symbols: int = 10
    ) -> Optional[Dict]:
        """
        🚀 DEDICATED PARABOLIC DUMP SCANNER 🚀
        
        Scans for EXHAUSTED parabolic pumps (50%+) ready to reverse.
        Separate from regular SHORTS - focuses on EXTREME moves only.
        
        Strategy:
        - Scans multiple 50%+ gainers
        - Scores each by overextension (RSI, EMA distance, momentum)
        - Returns strongest parabolic reversal candidate
        - Triple TPs: 4%, 8%, 12% (20%, 40%, 60% at 5x leverage)
        
        Returns:
            Signal dict with PARABOLIC_REVERSAL signal type
        """
        try:
            logger.info("🔥 ═══════════════════════════════════════════════════════")
            logger.info(f"🔥 PARABOLIC DUMP SCANNER - Looking for 50%+ exhausted pumps")
            logger.info("🔥 ═══════════════════════════════════════════════════════")
            
            # Get extreme gainers (50%+)
            gainers = await self.get_top_gainers(limit=max_symbols, min_change_percent=min_change_percent)
            
            if not gainers:
                logger.info("No 50%+ gainers found")
                return None
            
            logger.info(f"📊 Found {len(gainers)} coins with 50%+ pumps")
            
            # Clean up expired cooldowns
            now = datetime.utcnow()
            expired = [sym for sym, expires in shorts_cooldown.items() if expires <= now]
            for sym in expired:
                del shorts_cooldown[sym]
            
            # Evaluate ALL candidates and score them
            candidates = []
            
            for gainer in gainers:
                symbol = gainer['symbol']
                change_pct = gainer['change_percent']
                
                # Check if symbol is in cooldown
                if symbol in shorts_cooldown:
                    remaining_min = (shorts_cooldown[symbol] - now).total_seconds() / 60
                    logger.info(f"⏰ {symbol} SKIPPED - Cooldown active ({remaining_min:.0f} min left)")
                    continue
                
                logger.info(f"🎯 Analyzing: {symbol} @ +{change_pct:.1f}%")
                
                # Analyze momentum
                momentum = await self.analyze_momentum(symbol)
                
                if not momentum:
                    logger.info(f"  ❌ {symbol} - No momentum data")
                    continue
                
                # Only interested in SHORT signals with parabolic reversal
                if momentum['direction'] != 'SHORT':
                    logger.info(f"  ❌ {symbol} - Direction is {momentum['direction']} (need SHORT)")
                    continue
                
                if 'PARABOLIC REVERSAL' not in momentum['reason']:
                    logger.info(f"  ❌ {symbol} - Not a parabolic reversal")
                    continue
                
                # Calculate parabolic score (higher = better reversal candidate)
                # Factors: pump size, confidence, overextension signals
                score = (
                    change_pct * 0.4 +  # Bigger pump = more overextended
                    momentum['confidence'] * 0.6  # Higher confidence = better
                )
                
                candidates.append({
                    'symbol': symbol,
                    'gainer': gainer,
                    'momentum': momentum,
                    'score': score
                })
                
                logger.info(f"  ✅ {symbol} - PARABOLIC candidate (score: {score:.1f})")
            
            if not candidates:
                logger.info("No valid parabolic reversal candidates found")
                return None
            
            # Sort by score (highest first) and take best
            candidates.sort(key=lambda x: x['score'], reverse=True)
            best = candidates[0]
            
            logger.info(f"🏆 BEST PARABOLIC: {best['symbol']} (score: {best['score']:.1f})")
            
            # Build signal with TRIPLE TPs (parabolic dumps crash HARD!)
            entry_price = best['momentum']['entry_price']
            
            # Triple TPs: 1:1, 1:2, 1:3 R:R
            stop_loss = entry_price * (1 + 4.0 / 100)  # 4% SL (20% loss at 5x)
            take_profit_1 = entry_price * (1 - 4.0 / 100)  # TP1: 4% (20% profit at 5x)
            take_profit_2 = entry_price * (1 - 8.0 / 100)  # TP2: 8% (40% profit at 5x)
            take_profit_3 = entry_price * (1 - 12.0 / 100)  # TP3: 12% (60% profit at 5x)
            
            return {
                'symbol': best['symbol'],
                'direction': 'SHORT',
                'entry_price': entry_price,
                'stop_loss': stop_loss,
                'take_profit': take_profit_1,
                'take_profit_1': take_profit_1,
                'take_profit_2': take_profit_2,
                'take_profit_3': take_profit_3,
                'confidence': best['momentum']['confidence'],
                'reasoning': f"PARABOLIC DUMP: +{best['gainer']['change_percent']:.1f}% in 24h | {best['momentum']['reason']}",
                'trade_type': 'PARABOLIC_REVERSAL',  # NEW signal type
                'leverage': 5,
                '24h_change': best['gainer']['change_percent'],
                '24h_volume': best['gainer']['volume_24h'],
                'is_parabolic_reversal': True,
                'parabolic_score': best['score']
            }
            
        except Exception as e:
            logger.error(f"Error in parabolic dump scanner: {e}", exc_info=True)
            return None
    
    async def generate_early_pump_long_signal(
        self,
        min_change: float = 5.0,
        max_change: float = 20.0,
        max_symbols: int = 10
    ) -> Optional[Dict]:
        """
        Generate LONG signals from EARLY PUMP candidates (5-20% gains)
        
        Catches coins BEFORE they become top gainers at 25%+
        Perfect for riding the pump from early stage!
        
        Returns:
            Same signal format as generate_top_gainer_signal but for LONGS
        """
        try:
            # Get early pump candidates (5-20% range with high volume)
            pumpers = await self.get_early_pumpers(limit=max_symbols, min_change=min_change, max_change=max_change)
            
            if not pumpers:
                logger.info(f"❌ No coins found pumping {min_change}-{max_change}% in the last 24h")
                return None
            
            logger.info(f"📈 Found {len(pumpers)} pumping coins to analyze:")
            
            # Analyze each early pumper for LONG entry
            for idx, pumper in enumerate(pumpers, 1):
                symbol = pumper['symbol']
                logger.info(f"  [{idx}/{len(pumpers)}] {symbol}: +{pumper['change_percent']:.2f}%")
                
                # Analyze for LONG entry
                momentum = await self.analyze_early_pump_long(symbol)
                
                if not momentum or momentum['direction'] != 'LONG':
                    continue
                
                # Found a valid LONG signal!
                entry_price = momentum['entry_price']
                
                # LONG TPs: Dual targets (5% and 10% price moves)
                stop_loss = entry_price * (1 - 4.0 / 100)  # 20% loss at 5x
                take_profit_1 = entry_price * (1 + 5.0 / 100)  # TP1: 5% price move (25% profit at 5x)
                take_profit_2 = entry_price * (1 + 10.0 / 100)  # TP2: 10% price move (50% profit at 5x)
                take_profit_3 = None  # No TP3 for early pump longs
                
                # Get tier and pump data
                tier = pumper.get('tier', '30m')  # Default to 30m if not set
                pump_data = pumper.get('fresh_pump_data', {})
                tier_change = pump_data.get('candle_change_percent', pumper['change_percent'])
                volume_ratio = pump_data.get('volume_ratio', 0)
                
                # Tier-specific labels
                tier_labels = {
                    '5m': '⚡ ULTRA-EARLY',
                    '15m': '🔥 EARLY',
                    '30m': '✅ FRESH'
                }
                tier_label = tier_labels.get(tier, '✅ FRESH')
                
                logger.info(f"✅ {tier_label} LONG found: {symbol} @ +{tier_change}% ({tier})")
                
                return {
                    'symbol': symbol,
                    'direction': 'LONG',
                    'entry_price': entry_price,
                    'stop_loss': stop_loss,
                    'take_profit': take_profit_1,
                    'take_profit_1': take_profit_1,
                    'take_profit_2': take_profit_2,
                    'take_profit_3': take_profit_3,
                    'confidence': momentum['confidence'],
                    'reasoning': f"{tier_label} ({tier}): +{tier_change}% pump, {volume_ratio:.1f}x vol | {momentum['reason']}",
                    'trade_type': 'TOP_GAINER',
                    'leverage': 5,  # Fixed 5x leverage
                    '24h_change': pumper['change_percent_24h'],
                    '24h_volume': pumper['volume_24h'],
                    'is_parabolic_reversal': False,
                    'tier': tier,  # Add tier to signal data
                    'tier_label': tier_label,
                    'tier_change': tier_change,
                    'volume_ratio': volume_ratio
                }
            
            logger.info("No valid LONG entries found in early pumpers")
            return None
            
        except Exception as e:
            logger.error(f"Error generating early pump LONG signal: {e}")
            return None
    
    async def add_to_watchlist(self, db_session: Session, symbol: str, price: float, change_percent: float):
        """
        Add a symbol to the 48-hour watchlist for delayed reversal monitoring.
        
        Args:
            db_session: Database session
            symbol: Trading pair (e.g., 'AIXBT/USDT')
            price: Current price
            change_percent: Current 24h% change
        """
        from app.models import TopGainerWatchlist
        
        try:
            # Check if already in watchlist
            existing = db_session.query(TopGainerWatchlist).filter(
                TopGainerWatchlist.symbol == symbol
            ).first()
            
            if existing:
                # Update if new peak
                if change_percent > existing.peak_change_percent:
                    existing.peak_price = price
                    existing.peak_change_percent = change_percent
                existing.last_checked = datetime.utcnow()
                db_session.commit()
                logger.info(f"Updated watchlist for {symbol}: {change_percent}% (peak: {existing.peak_change_percent}%)")
            else:
                # Add new entry
                watchlist_entry = TopGainerWatchlist(
                    symbol=symbol,
                    peak_price=price,
                    peak_change_percent=change_percent,
                    first_seen=datetime.utcnow(),
                    last_checked=datetime.utcnow()
                )
                db_session.add(watchlist_entry)
                db_session.commit()
                logger.info(f"Added {symbol} to watchlist: {change_percent}% | Will monitor for 48h")
                
        except Exception as e:
            logger.error(f"Error adding {symbol} to watchlist: {e}")
            db_session.rollback()
    
    async def cleanup_old_watchlist(self, db_session: Session):
        """Remove watchlist entries older than 48 hours"""
        from app.models import TopGainerWatchlist
        
        try:
            cutoff_time = datetime.utcnow() - timedelta(hours=48)
            deleted = db_session.query(TopGainerWatchlist).filter(
                TopGainerWatchlist.first_seen < cutoff_time
            ).delete()
            
            if deleted > 0:
                db_session.commit()
                logger.info(f"Cleaned up {deleted} expired watchlist entries (>48h old)")
                
        except Exception as e:
            logger.error(f"Error cleaning up watchlist: {e}")
            db_session.rollback()
    
    async def get_watchlist_symbols(self, db_session: Session) -> List[Dict]:
        """
        Get all symbols currently on the watchlist.
        
        Returns:
            List of dicts with symbol, peak_change_percent, hours_tracked
        """
        from app.models import TopGainerWatchlist
        
        try:
            watchlist = db_session.query(TopGainerWatchlist).filter(
                TopGainerWatchlist.still_monitoring == True
            ).all()
            
            return [
                {
                    'symbol': entry.symbol,
                    'peak_change_percent': entry.peak_change_percent,
                    'hours_tracked': entry.hours_tracked,
                    'first_seen': entry.first_seen
                }
                for entry in watchlist
            ]
        except Exception as e:
            logger.error(f"Error getting watchlist: {e}")
            return []
    
    async def mark_watchlist_signal_sent(self, db_session: Session, symbol: str):
        """Mark a watchlist symbol as having sent a reversal signal"""
        from app.models import TopGainerWatchlist
        
        try:
            entry = db_session.query(TopGainerWatchlist).filter(
                TopGainerWatchlist.symbol == symbol
            ).first()
            
            if entry:
                entry.still_monitoring = False
                db_session.commit()
                logger.info(f"Marked {symbol} as signal sent (will stop monitoring)")
        except Exception as e:
            logger.error(f"Error marking watchlist entry: {e}")
            db_session.rollback()
    
    async def close(self):
        """Close HTTP client connection"""
        if self.client:
            await self.client.aclose()
            self.client = None


async def broadcast_top_gainer_signal(bot, db_session):
    """
    Scan for signals and broadcast to users with top_gainers_mode_enabled
    Supports 3 modes: SHORTS_ONLY, LONGS_ONLY, or BOTH
    
    - SHORTS: Scan 35%+ gainers for mean reversion (wait for exhausted pumps!)
    - LONGS: Scan 5-20% early pumps for momentum entries (catch pumps early!)
    - BOTH: Try both scans (shorts first, then longs)
    """
    from app.models import User, UserPreference, Signal, Trade
    from app.services.bitunix_trader import execute_bitunix_trade
    from datetime import datetime
    import logging
    
    logger = logging.getLogger(__name__)
    
    try:
        service = TopGainersSignalService()
        await service.initialize()
        
        # Get all users with top gainers mode enabled (regardless of auto-trading status)
        users_with_mode = db_session.query(User).join(UserPreference).filter(
            UserPreference.top_gainers_mode_enabled == True
        ).all()
        
        if not users_with_mode:
            logger.info("No users with top gainers mode enabled")
            await service.close()
            return
        
        # Count users with auto-trading enabled
        auto_traders = [u for u in users_with_mode if u.preferences and u.preferences.auto_trading_enabled]
        manual_traders = [u for u in users_with_mode if u not in auto_traders]
        
        logger.info(f"Scanning for signals: {len(users_with_mode)} total ({len(auto_traders)} auto, {len(manual_traders)} manual)")
        
        # 🔥 CRITICAL FIX: Check if ANY user wants SHORTS or LONGS
        # Don't just check first user - check ALL users' preferences!
        wants_shorts = False
        wants_longs = False
        
        for user in users_with_mode:
            prefs = user.preferences
            if prefs:
                user_mode = getattr(prefs, 'top_gainers_trade_mode', 'shorts_only')
                if user_mode in ['shorts_only', 'both']:
                    wants_shorts = True
                if user_mode in ['longs_only', 'both']:
                    wants_longs = True
        
        logger.info(f"User Preferences: SHORTS={wants_shorts}, LONGS={wants_longs}")
        
        # 🔥 CRITICAL: Generate ALL signal types if wanted (don't exit early!)
        parabolic_signal = None
        short_signal = None
        long_signal = None
        
        # ═══════════════════════════════════════════════════════
        # PARABOLIC MODE: Scan 50%+ exhausted pumps (HIGHEST PRIORITY!)
        # ═══════════════════════════════════════════════════════
        # Auto-enabled when SHORTS mode is on (best reversal opportunities)
        if wants_shorts:
            logger.info("🔥 ═══════════════════════════════════════════════════════")
            logger.info("🔥 PARABOLIC SCANNER - Priority #1 (50%+ exhausted dumps)")
            logger.info("🔥 ═══════════════════════════════════════════════════════")
            parabolic_signal = await service.generate_parabolic_dump_signal(min_change_percent=50.0, max_symbols=10)
            
            if parabolic_signal and parabolic_signal['direction'] == 'SHORT':
                logger.info(f"✅ PARABOLIC signal found: {parabolic_signal['symbol']} @ +{parabolic_signal.get('24h_change')}%")
        
        # ═══════════════════════════════════════════════════════
        # SHORTS MODE: Scan 35%+ gainers for mean reversion
        # ═══════════════════════════════════════════════════════
        # Only run if no parabolic signal found (avoid duplicate SHORTS)
        if wants_shorts and not parabolic_signal:
            logger.info("🔴 Scanning for SHORT signals (35%+ mean reversion)...")
            short_signal = await service.generate_top_gainer_signal(min_change_percent=35.0, max_symbols=5)
            
            if short_signal and short_signal['direction'] == 'SHORT':
                logger.info(f"✅ SHORT signal found: {short_signal['symbol']} @ +{short_signal.get('24h_change')}%")
        
        # ═══════════════════════════════════════════════════════
        # LONGS MODE: Scan FRESH pumps (5-50% gains with extended freshness windows)
        # ═══════════════════════════════════════════════════════
        # 🔥 REMOVED "if not signal_data" - ALWAYS scan if users want LONGS!
        if wants_longs:
            logger.info("🟢 ═══════════════════════════════════════════════════════")
            logger.info("🟢 LONGS SCANNER - Analyzing FRESH pumps (5-50% range, extended freshness)")
            logger.info("🟢 ═══════════════════════════════════════════════════════")
            long_signal = await service.generate_early_pump_long_signal(min_change=5.0, max_change=50.0, max_symbols=20)
            
            if long_signal and long_signal['direction'] == 'LONG':
                logger.info(f"✅ LONG signal found: {long_signal['symbol']} @ +{long_signal.get('24h_change')}%")
        
        # If no signals at all, exit
        if not parabolic_signal and not short_signal and not long_signal:
            mode_str = []
            if wants_shorts:
                mode_str.append("SHORTS/PARABOLIC")
            if wants_longs:
                mode_str.append("LONGS")
            logger.info(f"No signals found for {' and '.join(mode_str) if mode_str else 'any mode'}")
            await service.close()
            return
        
        # Process PARABOLIC signal first (HIGHEST PRIORITY - best opportunities!)
        if parabolic_signal:
            await process_and_broadcast_signal(parabolic_signal, users_with_mode, db_session, bot, service)
        
        # Process regular SHORT signal (if found and no parabolic)
        elif short_signal:
            await process_and_broadcast_signal(short_signal, users_with_mode, db_session, bot, service)
        
        # Process LONG signal (if found) - runs independently
        if long_signal:
            await process_and_broadcast_signal(long_signal, users_with_mode, db_session, bot, service)
        
        await service.close()
    
    except Exception as e:
        logger.error(f"Error in broadcast_top_gainer_signal: {e}", exc_info=True)


async def process_and_broadcast_signal(signal_data, users_with_mode, db_session, bot, service):
    """Helper function to process and broadcast a single signal with parallel execution"""
    from app.models import Signal, Trade
    from app.services.bitunix_trader import execute_bitunix_trade
    from datetime import datetime, timedelta
    import logging
    import asyncio
    import random
    
    logger = logging.getLogger(__name__)
    
    try:
        # 🔒 DUPLICATE PREVENTION: Check if signal already exists (last 5 minutes)
        # Prevents parallel execution race conditions
        # Accepts both TOP_GAINER and PARABOLIC_REVERSAL signal types
        recent_cutoff = datetime.utcnow() - timedelta(minutes=5)
        existing_signal = db_session.query(Signal).filter(
            Signal.symbol == signal_data['symbol'],
            Signal.direction == signal_data['direction'],
            Signal.signal_type.in_(['TOP_GAINER', 'PARABOLIC_REVERSAL']),
            Signal.created_at >= recent_cutoff
        ).first()
        
        if existing_signal:
            logger.warning(f"🚫 DUPLICATE SIGNAL PREVENTED: {signal_data['symbol']} {signal_data['direction']} already exists (Signal ID: {existing_signal.id})")
            return
        
        # Determine signal type from trade_type field (PARABOLIC_REVERSAL or TOP_GAINER)
        signal_type = signal_data.get('trade_type', 'TOP_GAINER')
        
        # Create signal record
        signal = Signal(
            symbol=signal_data['symbol'],
            direction=signal_data['direction'],
            entry_price=signal_data['entry_price'],
            stop_loss=signal_data['stop_loss'],
            take_profit=signal_data.get('take_profit'),
            take_profit_1=signal_data.get('take_profit_1'),
            take_profit_2=signal_data.get('take_profit_2'),
            take_profit_3=signal_data.get('take_profit_3'),  # 60% profit for parabolic dumps
            confidence=signal_data['confidence'],
            reasoning=signal_data['reasoning'],
            signal_type=signal_type,  # Use trade_type from signal data
            timeframe='5m',
            created_at=datetime.utcnow()
        )
        db_session.add(signal)
        db_session.commit()
        db_session.refresh(signal)
        
        logger.info(f"🚀 TOP GAINER SIGNAL: {signal.symbol} {signal.direction} @ ${signal.entry_price} (24h: {signal_data.get('24h_change')}%)")
        
        # Check if parabolic reversal (dual TPs)
        is_parabolic = signal_data.get('is_parabolic_reversal', False)
        
        # Build TP text - Direction-specific profits
        if signal.direction == 'LONG' and signal.take_profit_2:
            # LONGS: Dual TPs (5% and 10%)
            tp_text = f"""<b>TP1:</b> ${signal.take_profit_1:.6f} (+25% @ 5x) 
<b>TP2:</b> ${signal.take_profit_2:.6f} (+50% @ 5x) 🎯"""
            sl_text = "(-20% @ 5x)"  # LONGS: 4% SL * 5x = 20%
            rr_text = "25% and 50% profit targets"
        elif signal.direction == 'SHORT':
            # SHORTS: Single TP at 8% (CAPPED at 80% max for display clarity)
            tp_text = f"<b>TP:</b> ${signal.take_profit_1:.6f} (up to +80% max) 🎯"
            sl_text = "(up to -80% max)"  # SHORTS: Display capped at 80%
            rr_text = "1:2 risk-to-reward"
        else:
            # Fallback
            profit_pct = 25 if signal.direction == 'LONG' else 40
            tp_text = f"<b>TP:</b> ${signal.take_profit:.6f} (+{profit_pct}% @ 5x)"
            sl_text = f"(-{profit_pct}% @ 5x)"
            rr_text = "Single target"
        
        # Broadcast to users
        direction_emoji = "🟢 LONG" if signal.direction == 'LONG' else "🔴 SHORT"
        
        # Add tier badge for LONGS
        tier_badge = ""
        if signal.direction == 'LONG' and signal_data.get('tier'):
            tier_label = signal_data.get('tier_label', '')
            tier = signal_data.get('tier', '')
            tier_change = signal_data.get('tier_change', 0)
            tier_badge = f"\n🎯 <b>{tier_label}</b> detection ({tier} pump: +{tier_change}%)\n"
        
        signal_text = f"""
┏━━━━━━━━━━━━━━━━━━━━━━━┓
  🔥 <b>TOP GAINER ALERT</b> 🔥
┗━━━━━━━━━━━━━━━━━━━━━━━┛
{tier_badge}
{direction_emoji} <b>{signal.symbol}</b>
━━━━━━━━━━━━━━━━━━━━━━━━━━

<b>📊 Market Data</b>
├ 24h Change: <b>+{signal_data.get('24h_change')}%</b>
└ Volume: ${signal_data.get('24h_volume'):,.0f}

<b>🎯 Trade Setup</b>
├ Entry: <b>${signal.entry_price:.6f}</b>
├ {tp_text.replace(chr(10), chr(10) + '├ ')}
└ SL: ${signal.stop_loss:.6f} {sl_text}

<b>⚡ Risk Management</b>
├ Leverage: <b>5x</b> (Fixed)
└ Risk/Reward: <b>{rr_text}</b>

<b>💡 Analysis</b>
{signal.reasoning}

⚠️ <b>HIGH VOLATILITY MODE</b>
<i>Auto-executing for enabled users...</i>
"""
        
        # 🚀 PARALLEL EXECUTION with controlled concurrency
        # Use Semaphore to limit concurrent trades (prevents API rate limit issues)
        semaphore = asyncio.Semaphore(3)  # Max 3 concurrent trades
        
        async def execute_user_trade(user, user_idx):
            """Execute trade for a single user with controlled concurrency"""
            from app.database import SessionLocal
            from app.models import UserPreference, Trade
            
            # Each task gets its own DB session (sessions are not thread-safe)
            user_db = SessionLocal()
            start_time = asyncio.get_event_loop().time()
            executed = False
            
            try:
                async with semaphore:
                    # Add 200-400ms jitter to smooth API burst requests
                    jitter = random.uniform(0.2, 0.4)
                    await asyncio.sleep(jitter)
                    
                    logger.info(f"⚡ Starting trade execution for user {user.id} ({user_idx+1}/{len(users_with_mode)})")
                    
                    prefs = user_db.query(UserPreference).filter_by(user_id=user.id).first()
                    
                    # 🔥 Filter signals by user's trade mode preference
                    user_mode = getattr(prefs, 'top_gainers_trade_mode', 'shorts_only') if prefs else 'shorts_only'
                    
                    # Skip if user doesn't want this signal type
                    if signal.direction == 'SHORT' and user_mode not in ['shorts_only', 'both']:
                        logger.info(f"Skipping SHORT signal for user {user.id} (mode: {user_mode})")
                        return executed
                    if signal.direction == 'LONG' and user_mode not in ['longs_only', 'both']:
                        logger.info(f"Skipping LONG signal for user {user.id} (mode: {user_mode})")
                        return executed
                    
                    # Check if user has auto-trading enabled
                    has_auto_trading = prefs and prefs.auto_trading_enabled
                    
                    # Send manual signal notification for users without auto-trading
                    if not has_auto_trading:
                        try:
                            direction_emoji = "🟢 LONG" if signal.direction == 'LONG' else "🔴 SHORT"
                    
                            # Add tier badge for LONGS
                            tier_badge_manual = ""
                            if signal.direction == 'LONG' and signal_data.get('tier'):
                                tier_label = signal_data.get('tier_label', '')
                                tier = signal_data.get('tier', '')
                                tier_change = signal_data.get('tier_change', 0)
                                tier_badge_manual = f"\n🎯 <b>{tier_label}</b> detection ({tier} pump: +{tier_change}%)\n"
                            
                            # Calculate TP text - Direction-specific
                            if signal.direction == 'LONG' and signal.take_profit_2:
                                # LONGS: Dual TPs (5% and 10%)
                                tp_manual = f"""<b>TP1:</b> ${signal.take_profit_1:.6f} (+25% @ 5x)
<b>TP2:</b> ${signal.take_profit_2:.6f} (+50% @ 5x) 🎯"""
                                sl_manual = "(-20% @ 5x)"  # LONGS: 4% SL * 5x = 20%
                            elif signal.direction == 'SHORT':
                                # SHORTS: Single TP at 8% (CAPPED at 80% max for display)
                                tp_manual = f"<b>TP:</b> ${signal.take_profit_1:.6f} (up to +80% max) 🎯"
                                sl_manual = "(up to -80% max)"  # SHORTS: Display capped at 80%
                            else:
                                # Fallback
                                profit_pct_manual = 25 if signal.direction == 'LONG' else 40
                                tp_manual = f"<b>TP:</b> ${signal.take_profit:.6f} (+{profit_pct_manual}% @ 5x)"
                                sl_manual = f"(-{profit_pct_manual}% @ 5x)"
                            
                            manual_signal_msg = f"""
┏━━━━━━━━━━━━━━━━━━━━━━━┓
  🔥 <b>TOP GAINER SIGNAL</b> 🔥
┗━━━━━━━━━━━━━━━━━━━━━━━┛
{tier_badge_manual}
{direction_emoji} <b>{signal.symbol}</b>
━━━━━━━━━━━━━━━━━━━━━━━━━━

<b>📊 Market Data</b>
├ 24h Change: <b>+{signal_data.get('24h_change')}%</b>
└ Volume: ${signal_data.get('24h_volume'):,.0f}

<b>🎯 Trade Setup</b>
├ Entry: <b>${signal.entry_price:.6f}</b>
├ {tp_manual.replace(chr(10), chr(10) + '├ ')}
└ SL: ${signal.stop_loss:.6f} {sl_manual}

<b>⚡ Recommended Settings</b>
├ Leverage: <b>5x</b>
└ Risk/Reward: <b>1:2</b>

<b>💡 Analysis</b>
{signal.reasoning}

⚠️ <b>MANUAL SIGNAL</b>
<i>Enable auto-trading to execute automatically!</i>
"""
                            
                            await bot.send_message(
                                user.telegram_id,
                                manual_signal_msg,
                                parse_mode='HTML'
                            )
                            logger.info(f"✅ Sent manual signal to user {user.id}")
                            
                        except Exception as e:
                            logger.error(f"Error sending manual signal to user {user.id}: {e}")
                        
                        return executed  # Skip auto-execution for manual traders
            
                    # 🔥 CRITICAL FIX: Check if user already has position in this SPECIFIC symbol
                    existing_symbol_position = user_db.query(Trade).filter(
                        Trade.user_id == user.id,
                        Trade.status == 'open',
                        Trade.symbol == signal.symbol  # Same symbol check
                    ).first()
                    
                    if existing_symbol_position:
                        logger.info(f"⚠️ DUPLICATE PREVENTED: User {user.id} already has open position in {signal.symbol} (Trade ID: {existing_symbol_position.id})")
                        return executed
                    
                    # Check if user has space for more top gainer positions
                    current_top_gainer_positions = user_db.query(Trade).filter(
                        Trade.user_id == user.id,
                        Trade.status == 'open',
                        Trade.trade_type == 'TOP_GAINER'
                    ).count()
                    
                    max_allowed = prefs.top_gainers_max_symbols if prefs else 3
                    
                    if current_top_gainer_positions >= max_allowed:
                        logger.info(f"User {user.id} already has {current_top_gainer_positions} top gainer positions (max: {max_allowed})")
                        return executed
            
                    # Execute trade with user's custom leverage for top gainers
                    user_leverage = prefs.top_gainers_leverage if prefs and prefs.top_gainers_leverage else 5
                    trade = await execute_bitunix_trade(
                        signal=signal,
                        user=user,
                        db=user_db,
                        trade_type='TOP_GAINER',
                        leverage_override=user_leverage  # Use user's custom top gainer leverage
                    )
                    
                    if trade:
                        executed = True
                
                        # Send personalized notification with user's actual leverage
                        try:
                            user_leverage = prefs.top_gainers_leverage if prefs and prefs.top_gainers_leverage else 5
                            
                            # Calculate profit percentages with 80% cap (proportional scaling)
                            if signal.direction == 'LONG':
                                # LONGS: Dual TPs [5%, 10%] scaled proportionally
                                targets = calculate_leverage_capped_targets(
                                    entry_price=signal.entry_price,
                                    direction='LONG',
                                    tp_pcts=[5.0, 10.0],  # TP1, TP2
                                    base_sl_pct=4.0,
                                    leverage=user_leverage,
                                    max_profit_cap=80.0,
                                    max_loss_cap=80.0
                                )
                                tp1_profit_pct = targets['tp_profit_pcts'][0]
                                tp2_profit_pct = targets['tp_profit_pcts'][1]
                                sl_loss_pct = targets['sl_loss_pct']
                                display_leverage = user_leverage
                            else:  # SHORT
                                # SHORTS: Single TP [8%] capped at 80% profit
                                targets = calculate_leverage_capped_targets(
                                    entry_price=signal.entry_price,
                                    direction='SHORT',
                                    tp_pcts=[8.0],
                                    base_sl_pct=4.0,
                                    leverage=user_leverage,
                                    max_profit_cap=80.0,
                                    max_loss_cap=80.0
                                )
                                tp1_profit_pct = targets['tp_profit_pcts'][0]
                                sl_loss_pct = targets['sl_loss_pct']
                                display_leverage = user_leverage
                    
                            # Rebuild TP/SL text with leverage-capped percentages
                            # Calculate R:R based on capped values
                            if signal.direction == 'LONG' and signal.take_profit_2:
                                # LONGS: Dual TPs with capped percentages
                                user_tp_text = f"""<b>TP1:</b> ${targets['tp_prices'][0]:.6f} (+{tp1_profit_pct:.0f}% @ {display_leverage}x)
<b>TP2:</b> ${targets['tp_prices'][1]:.6f} (+{tp2_profit_pct:.0f}% @ {display_leverage}x) 🎯"""
                                user_sl_text = f"${targets['sl_price']:.6f} (-{sl_loss_pct:.0f}% @ {display_leverage}x)"
                                # Calculate actual R:R
                                rr_ratio = tp2_profit_pct / sl_loss_pct if sl_loss_pct > 0 else 0
                                rr_text = f"1:{rr_ratio:.1f} risk-to-reward ({tp1_profit_pct:.0f}% and {tp2_profit_pct:.0f}% targets)"
                            elif signal.direction == 'SHORT':
                                # SHORTS: Show capped profit percentage
                                user_tp_text = f"<b>TP:</b> ${targets['tp_prices'][0]:.6f} (+{tp1_profit_pct:.0f}% @ {display_leverage}x) 🎯"
                                user_sl_text = f"${targets['sl_price']:.6f} (-{sl_loss_pct:.0f}% @ {display_leverage}x)"
                                # Calculate actual R:R
                                rr_ratio = tp1_profit_pct / sl_loss_pct if sl_loss_pct > 0 else 0
                                rr_text = f"1:{rr_ratio:.1f} risk-to-reward"
                            else:
                                # Fallback
                                user_tp_text = f"<b>TP:</b> ${signal.take_profit:.6f} (+{tp1_profit_pct:.0f}% @ {display_leverage}x)"
                                user_sl_text = f"${signal.stop_loss:.6f} (-{sl_loss_pct:.0f}% @ {display_leverage}x)"
                                rr_ratio = tp1_profit_pct / sl_loss_pct if sl_loss_pct > 0 else 0
                                rr_text = f"1:{rr_ratio:.1f} risk-to-reward"
                            
                            # Personalized signal message
                            direction_emoji = "🟢 LONG" if signal.direction == 'LONG' else "🔴 SHORT"
                            
                            # Add tier badge for LONGS (same as broadcast)
                            tier_badge_personalized = ""
                            if signal.direction == 'LONG' and signal_data.get('tier'):
                                tier_label = signal_data.get('tier_label', '')
                                tier = signal_data.get('tier', '')
                                tier_change = signal_data.get('tier_change', 0)
                                tier_badge_personalized = f"\n🎯 <b>{tier_label}</b> detection ({tier} pump: +{tier_change}%)\n"
                            
                            personalized_signal = f"""
┏━━━━━━━━━━━━━━━━━━━━━━━┓
  🔥 <b>TOP GAINER ALERT</b> 🔥
┗━━━━━━━━━━━━━━━━━━━━━━━┛
{tier_badge_personalized}
{direction_emoji} <b>{signal.symbol}</b>
━━━━━━━━━━━━━━━━━━━━━━━━━━

<b>📊 Market Data</b>
├ 24h Change: <b>+{signal_data.get('24h_change')}%</b>
└ Volume: ${signal_data.get('24h_volume'):,.0f}

<b>🎯 Your Trade</b>
├ Entry: <b>${signal.entry_price:.6f}</b>
├ {user_tp_text.replace(chr(10), chr(10) + '├ ')}
└ SL: {user_sl_text}

<b>⚡ Your Settings</b>
├ Leverage: <b>{user_leverage}x</b>
└ Risk/Reward: <b>{rr_text}</b>

<b>💡 Analysis</b>
{signal.reasoning}

⚠️ <b>HIGH VOLATILITY MODE</b>
"""
                            
                            await bot.send_message(
                                user.telegram_id,
                                f"{personalized_signal}\n✅ <b>Trade Executed!</b>\n"
                                f"Position Size: ${trade.position_size:.2f}",
                                parse_mode="HTML"
                            )
                        except Exception as e:
                            logger.error(f"Failed to send notification to user {user.id}: {e}")
                
                return executed
            except Exception as e:
                logger.exception(f"Error executing trade for user {user.id}: {e}")
                return False
            finally:
                user_db.close()
        
        # Execute trades in parallel with controlled concurrency
        tasks = [execute_user_trade(user, idx) for idx, user in enumerate(users_with_mode)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        executed_count = sum(1 for r in results if r is True)
        
        logger.info(f"Top gainer signal executed for {executed_count}/{len(users_with_mode)} users")
        
    except Exception as e:
        logger.error(f"Error processing signal: {e}", exc_info=True)
