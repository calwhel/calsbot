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

# Separate cooldown for SCALP trades (don't interfere with TOP_GAINER SHORTS)
scalp_cooldown = {}


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
    # SL is NOT scaled - it caps directly at max_loss_cap/leverage (e.g., 80%/20x = 4% price move = 80% loss)
    effective_sl_pct = min(base_sl_pct, max_loss_cap / leverage)
    
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
        self.min_volume_usdt = 50000  # $50k minimum 24h volume - AGGRESSIVE for Bitunix thin books! (was $150k)
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
        Fetch OHLCV candles from Binance Futures with Bitunix fallback
        
        First tries Binance Futures public API (most reliable).
        Falls back to Bitunix API for Bitunix-exclusive coins.
        
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
            
            # TRY 1: Use Binance Futures public API (no auth needed, most reliable)
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
            if isinstance(data, list) and data:
                # Convert Binance format to standardized format: [timestamp, open, high, low, close, volume]
                # Binance candles: [open_time, open, high, low, close, volume, close_time, quote_volume, trades, ...]
                formatted_candles = []
                for candle in data:
                    if isinstance(candle, list) and len(candle) >= 6:
                        formatted_candles.append([
                            int(candle[0]),      # open time (timestamp in milliseconds)
                            float(candle[1]),    # open
                            float(candle[2]),    # high
                            float(candle[3]),    # low
                            float(candle[4]),    # close
                            float(candle[5])     # volume
                        ])
                
                # SUCCESS: If we got what we asked for (or close to it), use it!
                # For small requests (≤10 candles), need at least limit-1 to avoid breaking quick checks
                # For large requests (>10 candles), need at least 10 for technical analysis
                required_candles = max(1, limit - 1) if limit <= 10 else 10
                
                if len(formatted_candles) >= required_candles:
                    return formatted_candles
                else:
                    logger.warning(f"Binance returned only {len(formatted_candles)} candles for {symbol} (requested {limit}, need {required_candles}), trying Bitunix...")
            else:
                logger.warning(f"No candle data from Binance for {symbol}, trying Bitunix fallback...")
            
        except Exception as e:
            logger.warning(f"Binance candles failed for {symbol}: {e}, trying Bitunix...")
        
        # TRY 2: Fallback to Bitunix API for Bitunix-exclusive coins
        try:
            bitunix_symbol = symbol.replace('/', '')
            
            # Map interval format (Bitunix uses same format: '5m', '15m', '1h', etc.)
            url = f"{self.base_url}/api/v1/futures/market/klines"
            params = {
                'symbol': bitunix_symbol,
                'interval': interval,
                'limit': limit
            }
            
            response = await self.client.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            # Check for Bitunix API error codes
            if isinstance(data, dict):
                error_code = data.get('code')
                if error_code and error_code != 0:
                    error_msg = data.get('msg', 'Unknown error')
                    logger.error(f"❌ Bitunix API error for {symbol}: code={error_code}, msg={error_msg}")
                    logger.info(f"💡 Symbol {symbol} might not exist on Bitunix Futures (only Spot), or API issue")
                    return []
            
            # Parse Bitunix response (may be wrapped in 'data' key)
            candles = data.get('data', data) if isinstance(data, dict) else data
            
            if not isinstance(candles, list) or not candles:
                logger.error(f"❌ No candles from Bitunix either for {symbol}")
                return []
            
            # Convert Bitunix format to standardized format
            # Bitunix format: [timestamp, open, high, low, close, volume]
            formatted_candles = []
            for candle in candles:
                if isinstance(candle, list) and len(candle) >= 6:
                    formatted_candles.append([
                        int(candle[0]),      # timestamp
                        float(candle[1]),    # open
                        float(candle[2]),    # high
                        float(candle[3]),    # low
                        float(candle[4]),    # close
                        float(candle[5])     # volume
                    ])
            
            logger.info(f"✅ Bitunix fallback successful: {len(formatted_candles)} candles for {symbol}")
            return formatted_candles
            
        except Exception as e:
            logger.error(f"❌ Both Binance and Bitunix failed for {symbol}: {e}")
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
    
    async def get_funding_rate(self, symbol: str) -> Dict:
        """
        Fetch current funding rate from Bitunix Futures API
        
        Funding rate indicates market sentiment:
        - Positive (>0.01%) = Longs paying shorts = Bullish/greedy market
        - Negative (<-0.01%) = Shorts paying longs = Bearish market
        - High positive (>0.1%) = Extremely greedy = Good for SHORTS
        - Negative (<-0.05%) = Shorts underwater = Good for LONGS
        
        Returns:
        - funding_rate: float (e.g., 0.0015 = 0.15%)
        - next_funding_time: int (timestamp)
        - is_extreme: bool (funding > 0.1% or < -0.05%)
        """
        try:
            # Convert symbol format: BTC/USDT → BTCUSDT
            bitunix_symbol = symbol.replace('/', '')
            
            url = f"{self.base_url}/api/v1/futures/market/funding_rate"
            response = await self.client.get(url, params={'symbol': bitunix_symbol})
            response.raise_for_status()
            data = response.json()
            
            # Parse response
            if isinstance(data, dict) and 'data' in data:
                funding_data = data['data']
            else:
                funding_data = data
            
            funding_rate = float(funding_data.get('fundingRate', 0))
            next_funding_time = int(funding_data.get('nextFundingTime', 0))
            
            # Classify funding rate
            is_extreme_positive = funding_rate > 0.001  # >0.1% = very greedy
            is_extreme_negative = funding_rate < -0.0005  # <-0.05% = shorts underwater
            
            return {
                'funding_rate': funding_rate,
                'funding_rate_percent': funding_rate * 100,  # Convert to percentage
                'next_funding_time': next_funding_time,
                'is_extreme_positive': is_extreme_positive,
                'is_extreme_negative': is_extreme_negative,
                'sentiment': 'greedy' if funding_rate > 0.001 else 'fearful' if funding_rate < -0.0005 else 'neutral'
            }
            
        except Exception as e:
            logger.warning(f"Error fetching funding rate for {symbol}: {e}")
            return {
                'funding_rate': 0,
                'funding_rate_percent': 0,
                'next_funding_time': 0,
                'is_extreme_positive': False,
                'is_extreme_negative': False,
                'sentiment': 'unknown'
            }
    
    async def get_order_book_walls(self, symbol: str, entry_price: float, direction: str = 'LONG') -> Dict:
        """
        Analyze order book for massive buy/sell walls that could block price movement
        
        For LONGS:
        - Check for sell walls ABOVE entry price (resistance)
        - Check for buy walls BELOW entry price (support)
        
        For SHORTS:
        - Check for buy walls BELOW entry price (support that prevents dump)
        - Check for sell walls ABOVE entry price (resistance)
        
        Returns:
        - has_blocking_wall: bool (True if massive wall detected in path)
        - wall_price: float (price level of the wall)
        - wall_size_usdt: float (total USDT value of the wall)
        - support_below: float (total buy wall support in USDT)
        - resistance_above: float (total sell wall resistance in USDT)
        """
        try:
            # Convert symbol format: BTC/USDT → BTCUSDT
            bitunix_symbol = symbol.replace('/', '')
            
            # Fetch order book depth (limit=20 gives ±2% price range typically)
            url = f"{self.base_url}/api/v1/futures/market/depth"
            response = await self.client.get(url, params={'symbol': bitunix_symbol, 'limit': 20})
            response.raise_for_status()
            data = response.json()
            
            # Parse response
            if isinstance(data, dict) and 'data' in data:
                book = data['data']
            else:
                book = data
            
            # Null check - API may return empty/null data
            if not book or not isinstance(book, dict):
                return {'has_blocking_wall': False, 'reason': f'Invalid order book response: {type(book).__name__}'}
            
            asks = book.get('asks', [])  # [[price, quantity], ...]
            bids = book.get('bids', [])  # [[price, quantity], ...]
            
            if not asks or not bids:
                return {'has_blocking_wall': False, 'reason': 'Empty order book'}
            
            # Calculate wall detection threshold (dynamic based on book depth)
            # A "wall" is an order 5x larger than average order size
            avg_ask_size = sum(float(a[1]) * float(a[0]) for a in asks[:10]) / 10 if asks else 0
            avg_bid_size = sum(float(b[1]) * float(b[0]) for b in bids[:10]) / 10 if bids else 0
            wall_threshold = max(avg_ask_size, avg_bid_size) * 5
            
            # For LONGS: Check for sell walls above entry (resistance)
            if direction == 'LONG':
                resistance_walls = []
                support_walls = []
                
                for ask in asks:
                    price = float(ask[0])
                    quantity = float(ask[1])
                    size_usdt = price * quantity
                    
                    # Only check walls above entry price
                    if price > entry_price and size_usdt > wall_threshold:
                        resistance_walls.append({
                            'price': price,
                            'size_usdt': size_usdt,
                            'distance_percent': ((price - entry_price) / entry_price) * 100
                        })
                
                # Check buy walls below entry (support)
                for bid in bids:
                    price = float(bid[0])
                    quantity = float(bid[1])
                    size_usdt = price * quantity
                    
                    if price < entry_price and size_usdt > wall_threshold:
                        support_walls.append({
                            'price': price,
                            'size_usdt': size_usdt,
                            'distance_percent': ((entry_price - price) / entry_price) * 100
                        })
                
                # Find nearest resistance wall
                if resistance_walls:
                    nearest_wall = min(resistance_walls, key=lambda x: x['distance_percent'])
                    return {
                        'has_blocking_wall': True,
                        'wall_type': 'resistance',
                        'wall_price': nearest_wall['price'],
                        'wall_size_usdt': nearest_wall['size_usdt'],
                        'wall_distance_percent': nearest_wall['distance_percent'],
                        'total_resistance': sum(w['size_usdt'] for w in resistance_walls),
                        'total_support': sum(w['size_usdt'] for w in support_walls)
                    }
                
                return {
                    'has_blocking_wall': False,
                    'total_resistance': sum(float(a[1]) * float(a[0]) for a in asks[:5]),
                    'total_support': sum(float(b[1]) * float(b[0]) for b in bids[:5])
                }
            
            # For SHORTS: Check for buy walls below entry (support that prevents dump)
            else:  # direction == 'SHORT'
                support_walls = []
                resistance_walls = []
                
                for bid in bids:
                    price = float(bid[0])
                    quantity = float(bid[1])
                    size_usdt = price * quantity
                    
                    # Only check walls below entry price (support)
                    if price < entry_price and size_usdt > wall_threshold:
                        support_walls.append({
                            'price': price,
                            'size_usdt': size_usdt,
                            'distance_percent': ((entry_price - price) / entry_price) * 100
                        })
                
                # Check sell walls above entry
                for ask in asks:
                    price = float(ask[0])
                    quantity = float(ask[1])
                    size_usdt = price * quantity
                    
                    if price > entry_price and size_usdt > wall_threshold:
                        resistance_walls.append({
                            'price': price,
                            'size_usdt': size_usdt,
                            'distance_percent': ((price - entry_price) / entry_price) * 100
                        })
                
                # Find nearest support wall (blocks dump)
                if support_walls:
                    nearest_wall = min(support_walls, key=lambda x: x['distance_percent'])
                    return {
                        'has_blocking_wall': True,
                        'wall_type': 'support',
                        'wall_price': nearest_wall['price'],
                        'wall_size_usdt': nearest_wall['size_usdt'],
                        'wall_distance_percent': nearest_wall['distance_percent'],
                        'total_support': sum(w['size_usdt'] for w in support_walls),
                        'total_resistance': sum(w['size_usdt'] for w in resistance_walls)
                    }
                
                return {
                    'has_blocking_wall': False,
                    'total_support': sum(float(b[1]) * float(b[0]) for b in bids[:5]),
                    'total_resistance': sum(float(a[1]) * float(a[0]) for a in asks[:5])
                }
            
        except Exception as e:
            logger.warning(f"Error fetching order book for {symbol}: {e}")
            return {'has_blocking_wall': False, 'reason': f'Error: {str(e)}'}
    
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
            # Reduced to 15 candles minimum (from 30) for resilience during Bitunix API outages
            # 15 candles = 75 min of 5m data - sufficient for basic validation
            if len(candles_5m) < 15:
                flags.append('Coin too new (<75 min of data)')
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
            rejected_by_change = 0
            rejected_by_volume = 0
            top_pumpers = []  # Track top % pumpers for debugging
            
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
                
                # Track top pumpers for debugging (show why they're rejected)
                if change_percent >= 20.0:
                    top_pumpers.append({
                        'symbol': symbol,
                        'change': change_percent,
                        'volume': volume_usdt,
                        'passed_change': change_percent >= min_change_percent,
                        'passed_volume': volume_usdt >= self.min_volume_usdt
                    })
                
                # Filter criteria
                if change_percent < min_change_percent:
                    rejected_by_change += 1
                    continue
                    
                if volume_usdt < self.min_volume_usdt:
                    rejected_by_volume += 1
                    continue
                    
                gainers.append({
                    'symbol': symbol.replace('USDT', '/USDT'),  # Format as BTC/USDT
                    'change_percent': round(change_percent, 2),
                    'volume_24h': round(volume_usdt, 0),
                    'price': last_price,
                    'high_24h': float(ticker.get('high', 0)),
                    'low_24h': float(ticker.get('low', 0))
                })
            
            # 🔍 DEBUG: Log filtering stats
            logger.info(f"📊 TOP GAINERS FILTER STATS: {len(gainers)} passed | {rejected_by_change} rejected (change <{min_change_percent}%) | {rejected_by_volume} rejected (volume <${self.min_volume_usdt:,.0f})")
            
            # 🔍 DEBUG: Show top pumpers that got rejected (valuable insight!)
            if top_pumpers:
                top_pumpers.sort(key=lambda x: x['change'], reverse=True)
                logger.info(f"🔥 TOP 20%+ PUMPERS (showing why rejected):")
                for p in top_pumpers[:10]:  # Show top 10
                    status = "✅ PASSED" if (p['passed_change'] and p['passed_volume']) else f"❌ REJECTED: {'low volume' if not p['passed_volume'] else 'low change'}"
                    logger.info(f"   {p['symbol']}: +{p['change']:.1f}% | ${p['volume']:,.0f} vol | {status}")
            
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
            
            if age_minutes > 15:  # Candle older than 15 minutes = stale (RELAXED from 10 for Bitunix!)
                return {'is_fresh_pump': False, 'reason': 'stale_candle'}
            
            # Check 2: Is it a green candle AND 5%+ gain?
            if close_price <= open_price:
                return {'is_fresh_pump': False, 'reason': 'not_green_candle'}
            
            candle_change_percent = ((close_price - open_price) / open_price) * 100
            
            if candle_change_percent < 3.0:  # 🔥 RELAXED: 3%+ (was 5%+)
                return {'is_fresh_pump': False, 'reason': 'insufficient_pump', 'change': candle_change_percent}
            
            # Check 3: Volume 1.2x+ average of previous 3 candles (RELAXED for Bitunix thin books!)
            prev_volumes = [candles_5m[-4][5], candles_5m[-3][5], candles_5m[-2][5]]
            avg_volume = sum(prev_volumes) / len(prev_volumes)
            
            volume_ratio = volume / avg_volume if avg_volume > 0 else 0
            
            if volume_ratio < 1.2:  # 🔥 RELAXED: 1.2x (was 1.5x) for Bitunix thin books!
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
            
            if age_minutes > 30:  # Candle older than 30 minutes = stale (RELAXED from 20 for Bitunix!)
                return {'is_fresh_pump': False, 'reason': 'stale_candle'}
            
            # Check 2: Is it a green candle AND 5%+ gain? (RELAXED from 7%)
            if close_price <= open_price:
                return {'is_fresh_pump': False, 'reason': 'not_green_candle'}
            
            candle_change_percent = ((close_price - open_price) / open_price) * 100
            
            if candle_change_percent < 5.0:  # 🔥 RELAXED: 5%+ (was 7%+)
                return {'is_fresh_pump': False, 'reason': 'insufficient_pump', 'change': candle_change_percent}
            
            # Check 3: Volume 1.2x+ average of previous 2 candles (RELAXED for Bitunix thin books!)
            prev_volumes = [candles_15m[-3][5], candles_15m[-2][5]]
            avg_volume = sum(prev_volumes) / len(prev_volumes)
            
            volume_ratio = volume / avg_volume if avg_volume > 0 else 0
            
            if volume_ratio < 1.2:  # 🔥 RELAXED: 1.2x (was 1.5x) for Bitunix thin books!
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
            
            if age_minutes > 45:  # Candle older than 45 minutes = stale (RELAXED from 35 for Bitunix!)
                logger.debug(f"{symbol} 30m candle too old: {age_minutes:.1f} min")
                return {'is_fresh_pump': False, 'reason': 'stale_candle'}
            
            # Check 2: Is it a green candle AND 7%+ gain? (RELAXED from 10%)
            if close_price <= open_price:
                return {'is_fresh_pump': False, 'reason': 'not_green_candle'}
            
            candle_change_percent = ((close_price - open_price) / open_price) * 100
            
            if candle_change_percent < 7.0:  # 🔥 RELAXED: 7%+ (was 10%+)
                logger.debug(f"{symbol} 30m candle only +{candle_change_percent:.1f}% (need 7%+)")
                return {'is_fresh_pump': False, 'reason': 'insufficient_pump', 'change': candle_change_percent}
            
            # Check 3: Volume 1.2x+ average of previous 2 candles (RELAXED for Bitunix thin books!)
            prev_volumes = [candles_30m[-3][5], candles_30m[-2][5]]  # Previous 2 candles' volume
            avg_volume = sum(prev_volumes) / len(prev_volumes)
            
            volume_ratio = volume / avg_volume if avg_volume > 0 else 0
            
            if volume_ratio < 1.2:  # 🔥 RELAXED: 1.2x (was 1.5x) for Bitunix thin books!
                logger.debug(f"{symbol} 30m volume only {volume_ratio:.1f}x (need 1.2x+)")
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
            
            # 🔥 FRESHNESS CHECK REMOVED FOR SHORTS: Allow shorting fresh pumps!
            # Fresh pumps are PERFECT for SHORTS (mean reversion on momentum)
            # Only very very recent 5m pumps (< 5 min old) should be considered "too early" for shorts
            is_fresh_5m = await self.validate_fresh_5m_pump(symbol)
            fresh_5m_confirmed = is_fresh_5m and is_fresh_5m.get('is_fresh_pump') == True
            
            # ONLY skip if it's a VERY RECENT 5m pump AND we're on the first candle of the pump
            # This prevents entering literally right at the pump start
            if fresh_5m_confirmed and is_fresh_5m.get('age_minutes', 10) < 2:
                logger.info(f"{symbol} - TOO EARLY: 5m pump just started (<2 min). Wait for consolidation.")
                return None
            
            # 15m and 30m pumps are fair game for SHORTS (these are established moves ready to reverse)
            
            # Fetch candles with sufficient history for accurate analysis
            candles_5m = await self.fetch_candles(symbol, '5m', limit=50)
            candles_15m = await self.fetch_candles(symbol, '15m', limit=50)
            
            # Reduced to 15 candles minimum (from 30) for resilience during Bitunix API outages
            # 15 candles = 75 min (5m) or 3.75 hours (15m) of history - enough for RSI/EMA/volume analysis
            if len(candles_5m) < 15 or len(candles_15m) < 15:
                logger.warning(f"❌ {symbol} - INSUFFICIENT CANDLES: 5m={len(candles_5m)}/15, 15m={len(candles_15m)}/15 (need 15+ each)")
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
                
                # 🔥 NEW: Check funding rate for confirmation
                funding = await self.get_funding_rate(symbol)
                funding_pct = funding['funding_rate_percent']
                is_greedy = funding['is_extreme_positive']
                
                # 🔥 NEW: Check for buy walls that might prevent dump
                orderbook = await self.get_order_book_walls(symbol, current_price, direction='SHORT')
                has_wall = orderbook.get('has_blocking_wall', False)
                
                # OVEREXTENDED SHORT CONDITIONS (AGGRESSIVE - catch tops earlier!):
                # 1. RSI 55+ (catch overbought VERY early, before peak)
                # 2. Volume 1.0x+ (normal volume acceptable - coins pump on avg volume)
                # 3. Price 2%+ above EMA9 (catch extended moves EARLIER)
                # 4. 🔥 NEW: Funding rate positive (longs paying shorts = greedy market)
                # 5. 🔥 NEW: No massive buy wall blocking the dump
                is_overextended_short = (
                    rsi_5m >= 55 and  # AGGRESSIVE: Catch overbought earlier! (was 60)
                    volume_ratio >= 1.0 and  # Normal volume OK (was 2.0x - unrealistic!)
                    price_to_ema9_dist >= 2.0  # AGGRESSIVE: Catch extended moves earlier! (was 3.0%)
                )
                
                if is_overextended_short:
                    # 🔥 FUNDING RATE ANALYSIS
                    if is_greedy:
                        logger.info(f"  ✅ {symbol} - EXTREMELY GREEDY MARKET: Funding {funding_pct:.2f}% (longs paying shorts!) - STRONG SHORT SIGNAL!")
                        confidence_boost = 10
                    elif funding_pct > 0:
                        logger.info(f"  ✅ {symbol} - Funding {funding_pct:.2f}% (slightly greedy) - Good SHORT")
                        confidence_boost = 5
                    else:
                        logger.info(f"  ⚠️ {symbol} - Funding {funding_pct:.2f}% (neutral/bearish) - Weaker SHORT signal")
                        confidence_boost = 0
                    
                    # 🔥 ORDER BOOK WALL ANALYSIS
                    if has_wall:
                        wall_price = orderbook['wall_price']
                        wall_distance = orderbook['wall_distance_percent']
                        wall_size = orderbook['wall_size_usdt']
                        logger.info(f"  ⚠️ {symbol} - BUY WALL DETECTED at ${wall_price:.4f} ({wall_distance:.1f}% below entry) - ${wall_size:,.0f} USDT")
                        logger.info(f"  ⚠️ {symbol} - Skipping SHORT - whale defending ${wall_price:.4f}")
                        return None  # Skip - dump will likely bounce at wall
                    logger.info(f"{symbol} ✅ OVEREXTENDED SHORT: RSI {rsi_5m:.0f} (overbought!) | Vol {volume_ratio:.1f}x | +{price_to_ema9_dist:.1f}% above EMA9 | Funding {funding_pct:.2f}% | Shorting the top!")
                    return {
                        'direction': 'SHORT',
                        'confidence': 88 + confidence_boost,  # 🔥 Boosted by funding rate!
                        'entry_price': current_price,
                        'reason': f'🎯 OVEREXTENDED TOP | RSI {rsi_5m:.0f} overbought | Vol {volume_ratio:.1f}x | +{price_to_ema9_dist:.1f}% extended | Funding {funding_pct:.2f}% | Mean reversion play!'
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
                    current_candle_size >= 0.8 and  # AGGRESSIVE: 0.8% dump candle (was 1.0%)
                    volume_ratio >= 1.2 and  # AGGRESSIVE: 1.2x volume (was 1.5x)
                    35 <= rsi_5m <= 70  # WIDER RSI range (was 40-65)
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
            # SPECIAL CASE: Mean Reversion SHORT on Parabolic Pumps (RELAXED FOR 48%+ DUMPS!)
            # Even if 5m is still bullish, if price is EXTREMELY overextended
            # and showing reversal signs, take the SHORT (top gainer specialty!)
            # ═══════════════════════════════════════════════════════
            elif bullish_5m and not bullish_15m:
                # 15m already bearish but 5m lagging - this is the reversal point
                # Perfect for catching top gainer dumps (like CROSS +48% starting to roll over)
                price_extension = price_to_ema9_dist
                
                # 🔥 RELAXED PARABOLIC DETECTION (Catch exhausted pumps earlier!)
                # Check last 6 candles (30 min) instead of just 2 (10 min)
                has_exhaustion_signs = False
                exhaustion_reason = ""
                
                # Check 1: Topping pattern in last 6 candles (30 min window)
                if len(closes_5m) >= 6 and len(candles_5m) >= 6:
                    recent_closes = closes_5m[-6:]
                    recent_candles = candles_5m[-6:]
                    
                    # Look for reversal signs: lower highs in last 3 candles
                    highs = [c[2] for c in recent_candles[-3:]]  # Last 3 highs
                    has_lower_highs = highs[-1] < highs[0]  # Current high < first high
                    
                    # OR: Climax candle with big upper wick (10%+ wick = rejection)
                    current_high = candles_5m[-1][2]
                    current_wick_size = ((current_high - current_price) / current_price) * 100
                    has_big_wick = current_wick_size >= 1.0  # 🔥 RELAXED: 1%+ wick (was 10%+)
                    
                    if has_lower_highs or has_big_wick:
                        has_exhaustion_signs = True
                        if has_lower_highs:
                            exhaustion_reason = "Lower highs forming"
                        else:
                            exhaustion_reason = f"{current_wick_size:.1f}% upper wick rejection"
                        logger.info(f"{symbol} ✅ EXHAUSTION: {exhaustion_reason}")
                
                # 🔥 RELAXED ENTRY LOGIC: Accept EITHER good RSI OR exhaustion signs
                # Don't require BOTH - just need ONE strong signal
                good_rsi = rsi_5m >= 55  # 🔥 RELAXED: 55+ (was 60-75 strict range)
                good_volume = volume_ratio >= 1.2  # 🔥 RELAXED: 1.2x (was 1.5x)
                good_extension = price_extension > 2.0
                
                # Entry if: (RSI good OR exhaustion signs) AND volume + extension
                if good_extension and good_volume and (good_rsi or has_exhaustion_signs):
                    confidence = 92 if (good_rsi and has_exhaustion_signs) else 88
                    logger.info(f"{symbol} ✅ PARABOLIC REVERSAL: Extension {price_extension:+.1f}% | RSI {rsi_5m:.0f} | Vol {volume_ratio:.1f}x | {exhaustion_reason}")
                    return {
                        'direction': 'SHORT',
                        'confidence': confidence,
                        'entry_price': current_price,
                        'reason': f'🎯 PARABOLIC REVERSAL | {price_extension:+.1f}% overextended | {exhaustion_reason if exhaustion_reason else f"RSI {rsi_5m:.0f}"} | Vol: {volume_ratio:.1f}x'
                    }
                else:
                    skip_reason = []
                    if not good_extension:
                        skip_reason.append(f"Not extended enough ({price_extension:+.1f}%, need >2%)")
                    if not good_rsi and not has_exhaustion_signs:
                        skip_reason.append(f"RSI {rsi_5m:.0f} (need 55+) AND no exhaustion signs")
                    if not good_volume:
                        skip_reason.append(f"Low volume {volume_ratio:.1f}x (need 1.2x+)")
                    
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
    
    async def analyze_scalp_momentum(self, symbol: str) -> Optional[Dict]:
        """
        ⚡ SCALP MOMENTUM (LONG) - IMPROVED ENTRY QUALITY
        
        High-quality momentum entries with better confirmation:
        - Bullish EMA structure (EMA9 > EMA21)
        - Price at or below EMA9 (pullback = safer entry)
        - RSI 50-70 (building momentum, not overbought)
        - Current candle is GREEN (bullish confirmation)
        - Last 2+ candles bullish (trend strength)
        - Volume acceleration (current > previous vol)
        - Volume 1.5x+ average (decent buying pressure)
        
        Expected: ~15-25% of total scalp signals (better entries, less rare)
        
        Returns scalp LONG signal or None
        """
        try:
            candles_5m = await self.fetch_candles(symbol, '5m', limit=30)
            
            if len(candles_5m) < 20:
                return None
            
            opens_5m = [c[1] for c in candles_5m]
            closes_5m = [c[4] for c in candles_5m]
            volumes_5m = [c[5] for c in candles_5m]
            
            current_price = closes_5m[-1]
            current_open = opens_5m[-1]
            current_close = closes_5m[-1]
            
            # Check if current candle is GREEN (bullish)
            current_candle_bullish = current_close > current_open
            if not current_candle_bullish:
                return None
            
            ema9_5m = self._calculate_ema(closes_5m, 9)
            ema21_5m = self._calculate_ema(closes_5m, 21)
            rsi_5m = self._calculate_rsi(closes_5m, 14)
            
            # Check EMA structure (bullish)
            if ema9_5m <= ema21_5m:
                return None
            
            # Price should be AT or BELOW EMA9 (pullback entry is better than chasing)
            price_to_ema9_dist = ((current_price - ema9_5m) / ema9_5m) * 100
            if price_to_ema9_dist > 0.5:  # Slightly above EMA9 is okay, but prefer at/below
                return None
            
            # RSI in momentum zone (50-70), avoid <50 (weak) and >70 (overbought)
            if not (50 <= rsi_5m <= 70):
                return None
            
            # Check trend strength: last 2+ candles must be bullish
            bullish_count = sum(1 for i in range(-2, 0) if closes_5m[i] > opens_5m[i])
            if bullish_count < 2:
                return None
            
            # Check volume: current > previous AND >= 1.5x average
            avg_volume = sum(volumes_5m[-19:-1]) / 18 if len(volumes_5m) >= 19 else volumes_5m[-5:] and sum(volumes_5m[-5:]) / 5
            volume_ratio = volumes_5m[-1] / avg_volume if avg_volume > 0 else 1.0
            
            if volumes_5m[-1] <= volumes_5m[-2]:  # Volume not accelerating
                return None
            if volume_ratio < 1.5:  # Not enough buying pressure
                return None
            
            confidence = 85 + min(10, int(volume_ratio))  # Boost confidence with volume
            logger.info(f"⚡ SCALP LONG IMPROVED: {symbol} | Price: {price_to_ema9_dist:+.2f}% from EMA9 | RSI: {rsi_5m:.0f} | Vol: {volume_ratio:.1f}x | Trend: {bullish_count}/2 bullish")
            
            return {
                'direction': 'LONG',
                'confidence': min(95, confidence),
                'entry_price': current_price,
                'reason': f'⚡ MOMENTUM PULLBACK | EMA9 entry | RSI {rsi_5m:.0f} | Vol {volume_ratio:.1f}x | {bullish_count} bullish candles'
            }
            
        except Exception as e:
            logger.debug(f"Error in scalp momentum check for {symbol}: {e}")
            return None
    
    async def analyze_scalp_reversal(self, symbol: str) -> Optional[Dict]:
        """
        📉 SCALP REVERSAL (SHORT) - EXTREMELY AGGRESSIVE (fire constantly!)
        
        Detects coins showing ANY reversal/dump signs (super easy to trigger):
        - Price >3.5% above EMA9 (ULTRA LOOSE - catch early)
        - RSI >48 (ULTRA LOOSE - catch before peak)
        - Red candle OR ANY upper wick >0.05% (ULTRA LOOSE - any rejection)
        - Volume 0.5x+ (ULTRA LOOSE - ignore volume almost entirely)
        
        Expected: 85-90% of all scalp signals should be SHORTS
        
        Returns scalp SHORT signal or None
        """
        try:
            candles_5m = await self.fetch_candles(symbol, '5m', limit=30)
            
            if len(candles_5m) < 20:
                return None
            
            closes_5m = [c[4] for c in candles_5m]
            opens_5m = [c[1] for c in candles_5m]
            highs_5m = [c[2] for c in candles_5m]
            volumes_5m = [c[5] for c in candles_5m]
            
            current_price = closes_5m[-1]
            current_open = opens_5m[-1]
            current_high = highs_5m[-1]
            current_candle_bearish = current_price < current_open
            
            ema9_5m = self._calculate_ema(closes_5m, 9)
            ema21_5m = self._calculate_ema(closes_5m, 21)
            rsi_5m = self._calculate_rsi(closes_5m, 14)
            
            avg_volume = sum(volumes_5m[-19:-1]) / 18
            volume_ratio = volumes_5m[-1] / avg_volume if avg_volume > 0 else 1.0
            
            price_to_ema9_dist = ((current_price - ema9_5m) / ema9_5m) * 100
            
            # ⚡ SCALP SHORT REQUIREMENTS (ULTRA AGGRESSIVE - FIRE CONSTANTLY):
            # 1. Overextended pump (>3.5% above EMA9, ULTRA LOOSE)
            if price_to_ema9_dist <= 3.5:
                return None
            
            # 2. Overbought RSI (>48, ULTRA LOOSE - catch before peak)
            if rsi_5m <= 48:
                return None
            
            # 3. Reversal sign (red candle OR ANY tiny wick >0.05%)
            upper_wick = ((current_high - current_price) / current_price) * 100 if current_price > 0 else 0
            is_red = current_candle_bearish
            has_rejection_wick = upper_wick >= 0.05  # 0.05%+ upper wick (ULTRA LOOSE - any rejection)
            
            if not (is_red or has_rejection_wick):
                return None
            
            # 4. Volume confirmation (0.5x+, ULTRA LOOSE - almost ignore volume)
            if volume_ratio < 0.5:
                return None
            
            confidence = 80
            reason = "red candle" if is_red else f"wick rejection ({upper_wick:.1f}%)"
            
            logger.info(f"📉 SCALP SHORT: {symbol} | Overextended: {price_to_ema9_dist:+.1f}% | RSI: {rsi_5m:.0f} | {reason}")
            
            return {
                'direction': 'SHORT',
                'confidence': confidence,
                'entry_price': current_price,
                'reason': f'📉 REVERSAL SHORT | +{price_to_ema9_dist:.1f}% overextended | RSI {rsi_5m:.0f} ({reason})'
            }
            
        except Exception as e:
            logger.debug(f"Error in scalp reversal check for {symbol}: {e}")
            return None
    
    async def analyze_momentum_long(self, symbol: str) -> Optional[Dict]:
        """
        🚀 AGGRESSIVE MOMENTUM LONG - Catch strong pumps WITHOUT waiting for retracement!
        
        This is the "aggressive" version for coins showing STRONG momentum (10-30% pumps).
        Allows entries during the pump (no pullback required) to catch big runners.
        
        Entry criteria:
        1. ✅ Strong volume surge (1.5x+ average)
        2. ✅ Bullish momentum (EMA9 > EMA21, both timeframes)
        3. ✅ NOT OVEREXTENDED - within 10% of EMA9 (relaxed from 5%)
        4. ✅ RSI 45-78 (momentum allowed, not screaming overbought)
        5. ✅ Bullish candle (green, not red)
        6. ✅ No retracement required - direct momentum entry!
        
        Returns signal for AGGRESSIVE LONG entry or None if criteria not met
        """
        try:
            logger.info(f"🚀 MOMENTUM LONG ANALYSIS: {symbol}...")
            
            # Quality checks
            liquidity_check = await self.check_liquidity(symbol)
            if not liquidity_check['is_liquid']:
                logger.info(f"  ❌ {symbol} - {liquidity_check['reason']}")
                return None
            
            candles_5m = await self.fetch_candles(symbol, '5m', limit=50)
            candles_15m = await self.fetch_candles(symbol, '15m', limit=50)
            
            # Reduced to 15 candles minimum (from 30) for resilience during Bitunix API outages
            if len(candles_5m) < 15 or len(candles_15m) < 15:
                logger.warning(f"  ❌ {symbol} - INSUFFICIENT CANDLES: 5m={len(candles_5m)}/15, 15m={len(candles_15m)}/15")
                return None
            
            manipulation_check = await self.check_manipulation_risk(symbol, candles_5m)
            if not manipulation_check['is_safe']:
                logger.info(f"  ❌ {symbol} - Manipulation risk")
                return None
            
            # Technical analysis
            closes_5m = [c[4] for c in candles_5m]
            volumes_5m = [c[5] for c in candles_5m]
            closes_15m = [c[4] for c in candles_15m]
            
            current_price = closes_5m[-1]
            current_open = candles_5m[-1][1]
            current_candle_bullish = current_price > current_open
            
            # EMAs
            ema9_5m = self._calculate_ema(closes_5m, 9)
            ema21_5m = self._calculate_ema(closes_5m, 21)
            ema9_15m = self._calculate_ema(closes_15m, 9)
            ema21_15m = self._calculate_ema(closes_15m, 21)
            
            # RSI
            rsi_5m = self._calculate_rsi(closes_5m, 14)
            
            # Volume
            avg_volume = sum(volumes_5m[-20:-1]) / 19
            current_volume = volumes_5m[-1]
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0
            
            # Price to EMA distance
            price_to_ema9_dist = ((current_price - ema9_5m) / ema9_5m) * 100
            
            # Trend alignment (bullish required)
            bullish_5m = ema9_5m > ema21_5m
            bullish_15m = ema9_15m > ema21_15m
            
            # 🔥 AGGRESSIVE MOMENTUM ENTRY
            # Allows entries during strong pumps WITHOUT waiting for pullback
            
            # Filter 1: Must be in uptrend (both timeframes)
            if not (bullish_5m and bullish_15m):
                logger.info(f"  ❌ {symbol} - Not bullish on both timeframes")
                return None
            
            # Filter 2: Not TOO overextended (within 10% of EMA9)
            if price_to_ema9_dist > 10.0:
                logger.info(f"  ❌ {symbol} - Too extended ({price_to_ema9_dist:+.1f}% from EMA9, need ≤10%)")
                return None
            
            # Filter 3: RSI not screaming overbought (45-78)
            if not (45 <= rsi_5m <= 78):
                logger.info(f"  ❌ {symbol} - RSI {rsi_5m:.0f} out of range (need 45-78)")
                return None
            
            # Filter 4: Volume confirmation (1.5x minimum)
            if volume_ratio < 1.5:
                logger.info(f"  ❌ {symbol} - Low volume {volume_ratio:.1f}x (need 1.5x+)")
                return None
            
            # Filter 5: Current candle must be bullish (green)
            if not current_candle_bullish:
                logger.info(f"  ❌ {symbol} - Current candle is bearish (need green)")
                return None
            
            # ✅ ALL CHECKS PASSED - Aggressive momentum entry!
            confidence = 88  # Aggressive = slightly lower confidence than safe pullback entries
            
            logger.info(f"  ✅ {symbol} - MOMENTUM LONG entry!")
            logger.info(f"     • Price: {price_to_ema9_dist:+.1f}% from EMA9")
            logger.info(f"     • RSI: {rsi_5m:.0f}")
            logger.info(f"     • Volume: {volume_ratio:.1f}x")
            
            return {
                'direction': 'LONG',
                'confidence': confidence,
                'entry_price': current_price,
                'reason': f'🚀 MOMENTUM ENTRY | {price_to_ema9_dist:+.1f}% from EMA9 | RSI {rsi_5m:.0f} | Vol {volume_ratio:.1f}x | Riding strong pump!'
            }
            
        except Exception as e:
            logger.error(f"Error in momentum LONG analysis for {symbol}: {e}")
            return None
    
    async def analyze_early_pump_long(self, symbol: str) -> Optional[Dict]:
        """
        ✅ SAFE PULLBACK LONG - Wait for retracement before entering (conservative)
        
        This is the "safe" version that requires pullback/retracement.
        Lower risk but may miss fast-moving pumps.
        
        Entry criteria:
        1. ✅ Strong volume surge (1.0x+ average)
        2. ✅ Bullish momentum building (EMA9 > EMA21, both timeframes)
        3. ✅ MUST have retracement - price near/below EMA9 (NO CHASING!)
        4. ✅ RSI 40-70 (momentum without overbought)
        5. ✅ Resumption pattern (red pullback → green continuation)
        
        Returns signal for SAFE LONG entry or None if criteria not met
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
            
            # Reduced to 15 candles minimum (from 30) for resilience during Bitunix API outages
            if len(candles_5m) < 15 or len(candles_15m) < 15:
                logger.warning(f"  ❌ {symbol} - INSUFFICIENT CANDLES: 5m={len(candles_5m)}/15, 15m={len(candles_15m)}/15")
                return None
            
            # 🔥 QUALITY CHECK #2: Anti-Manipulation Filter
            manipulation_check = await self.check_manipulation_risk(symbol, candles_5m)
            if not manipulation_check['is_safe']:
                logger.info(f"  ❌ {symbol} REJECTED - Manipulation risk: {', '.join(manipulation_check['flags'])}")
                return None
            logger.info(f"  ✅ {symbol} - Anti-manipulation OK")
            
            # 🔥 FRESHNESS VALIDATION: Handled by tier-based system (5m/15m/30m)
            # The get_early_pumpers() function already validates freshness via validate_fresh_Xm_pump()
            # No need for redundant 3-hour check that was blocking valid signals!
            # 
            # OLD ISSUE: Coin pumps 10% from 4-6h ago → Still fresh and valid
            #            But only +1% in last 3h → Was incorrectly REJECTED
            # 
            # FIX: Trust the tier-based freshness validation (already proven to work)
            if len(candles_5m) >= 37:
                price_180m_ago = candles_5m[-37][4]
                current_price_check = candles_5m[-1][4]
                
                if price_180m_ago > 0:
                    pump_180m_percent = ((current_price_check - price_180m_ago) / price_180m_ago) * 100
                    logger.info(f"  📊 {symbol} - Pump momentum: +{pump_180m_percent:.1f}% in last 3h")
            
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
            
            # 🔥 NEW: Check funding rate for momentum confirmation
            funding = await self.get_funding_rate(symbol)
            funding_pct = funding['funding_rate_percent']
            is_shorts_underwater = funding['is_extreme_negative']
            
            # 🔥 NEW: Check for sell walls that might prevent rally
            orderbook = await self.get_order_book_walls(symbol, current_price, direction='LONG')
            has_wall = orderbook.get('has_blocking_wall', False)
            
            # 🔥 FUNDING RATE ANALYSIS FOR LONGS
            if is_shorts_underwater:
                logger.info(f"  ✅ {symbol} - SHORTS UNDERWATER: Funding {funding_pct:.2f}% (shorts paying longs!) - STRONG LONG SIGNAL!")
                confidence_boost = 10
            elif funding_pct < 0:
                logger.info(f"  ✅ {symbol} - Funding {funding_pct:.2f}% (slightly negative) - Good LONG")
                confidence_boost = 5
            else:
                logger.info(f"  ⚠️ {symbol} - Funding {funding_pct:.2f}% (positive/neutral) - Market already long")
                confidence_boost = 0
            
            # 🔥 ORDER BOOK WALL ANALYSIS
            if has_wall:
                wall_price = orderbook['wall_price']
                wall_distance = orderbook['wall_distance_percent']
                wall_size = orderbook['wall_size_usdt']
                logger.info(f"  ⚠️ {symbol} - SELL WALL DETECTED at ${wall_price:.4f} ({wall_distance:.1f}% above entry) - ${wall_size:,.0f} USDT")
                logger.info(f"  ⚠️ {symbol} - Skipping LONG - whale dumping at ${wall_price:.4f}")
                return None  # Skip - pump will likely get rejected at wall
            
            # Calculate candle sizes for retracement detection
            current_candle_size = abs((current_price - current_open) / current_open * 100)
            prev_close = closes_5m[-2]
            prev_open = candles_5m[-2][1]
            prev_candle_bullish = prev_close > prev_open
            prev_candle_bearish = prev_close < prev_open
            
            # ENTRY CONDITION 1: EMA9 PULLBACK LONG (BEST - wait for retracement!)
            # Price pulled back to/below EMA9, ready to resume UP
            # 🔥 STRICT: Price must be AT or BELOW EMA9 (not above) - prevents top entries!
            is_at_or_below_ema9 = price_to_ema9_dist <= 0.0  # 🔥 STRICT: Price AT or BELOW EMA9 (true pullback!)
            
            logger.info(f"  📊 {symbol} - Price to EMA9: {price_to_ema9_dist:+.2f}%, Vol: {volume_ratio:.2f}x, RSI: {rsi_5m:.0f}")
            if (is_at_or_below_ema9 and
                volume_ratio >= 1.5 and  # 🔥 STRICT: 1.5x minimum volume for confirmation
                rsi_5m >= 45 and rsi_5m <= 70 and  # 🔥 STRICT: Avoid overbought (70+) for true pullbacks
                bullish_momentum and
                current_candle_bullish):  # Green candle resuming
                
                logger.info(f"{symbol} ✅ EMA9 PULLBACK LONG: AT/BELOW EMA9 ({price_to_ema9_dist:+.1f}%) | Vol {volume_ratio:.1f}x | RSI {rsi_5m:.0f} | Funding {funding_pct:.2f}%")
                return {
                    'direction': 'LONG',
                    'confidence': 95 + confidence_boost,  # 🔥 Boosted by funding rate!
                    'entry_price': current_price,
                    'reason': f'📈 EMA9 PULLBACK | Clear retracement entry | Vol {volume_ratio:.1f}x | RSI {rsi_5m:.0f} | Funding {funding_pct:.2f}%'
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
                rsi_5m >= 45 and rsi_5m <= 70 and  # 🔥 STRICT: Avoid overbought (70+)
                volume_ratio >= 1.5 and  # 🔥 STRICT: 1.5x minimum volume for confirmation
                price_to_ema9_dist >= -2.0 and price_to_ema9_dist <= 2.0):  # 🔥 STRICT: Max 2% above EMA9 (true pullback!)
                
                return {
                    'direction': 'LONG',
                    'confidence': 90 + confidence_boost,  # 🔥 Boosted by funding rate!
                    'entry_price': current_price,
                    'reason': f'🎯 RESUMPTION LONG | Clear pullback continuation | Vol {volume_ratio:.1f}x | RSI {rsi_5m:.0f} | Funding {funding_pct:.2f}%'
                }
            
            # ENTRY CONDITION 3: REMOVED - Was causing entries at tops of green candles!
            # All LONGS now REQUIRE retracement (either EMA9 pullback or resumption pattern)
            # This prevents buying exhausted pumps and ensures safer entries
            
            # SKIP - no valid LONG entry (MUST have retracement to avoid buying tops!)
            skip_reason = []
            if not has_ema9_pullback and not has_resumption_pattern:
                skip_reason.append("No retracement (need EMA9 pullback or resumption pattern)")
            if price_to_ema9_dist > 2.0:
                skip_reason.append(f"Too far from EMA9 ({price_to_ema9_dist:+.1f}%, need ≤2% for true pullback)")
            if volume_ratio < 1.5:
                skip_reason.append(f"Low volume {volume_ratio:.1f}x (need 1.5x+ minimum)")
            if not (45 <= rsi_5m <= 70):
                skip_reason.append(f"RSI {rsi_5m:.0f} out of range (need 45-70, avoid overbought)")
            
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
                logger.info("❌ No top gainers found meeting criteria (all rejected by volume/change filters)")
                return None
            
            # 🔍 DEBUG: Show which gainers passed filters and will be analyzed
            logger.info(f"✅ TOP GAINERS PASSED: {len(gainers)} coins ready for momentum analysis:")
            for g in gainers:
                logger.info(f"   → {g['symbol']}: +{g['change_percent']}% | ${g['volume_24h']:,.0f} vol")
            
            # Clean up expired cooldowns
            now = datetime.utcnow()
            expired = [sym for sym, expires in shorts_cooldown.items() if expires <= now]
            for sym in expired:
                del shorts_cooldown[sym]
            
            # Track rejection reasons for debugging
            momentum_rejections = []
            
            # PRIORITY 1: Look for parabolic reversal shorts (biggest pumps first)
            # These are the BEST opportunities - coins that pumped 50%+ and are rolling over
            for gainer in gainers:
                if gainer['change_percent'] >= 50.0:  # Extreme pumps (50%+)
                    symbol = gainer['symbol']
                    
                    # 🔥 PARABOLIC SHORTS BYPASS COOLDOWN!
                    # Cooldown only applies to normal shorts (28%+), not parabolic (50%+)
                    # This ensures we catch BANANA-style exhausted dumps even if a normal short failed earlier
                    logger.info(f"🎯 Analyzing PARABOLIC candidate: {symbol} @ +{gainer['change_percent']}% (bypassing cooldown!)")
                    
                    momentum = await self.analyze_momentum(symbol)
                    
                    if not momentum:
                        momentum_rejections.append(f"{symbol} (PARABOLIC): momentum analysis returned None")
                        logger.info(f"   ❌ {symbol} PARABOLIC: momentum analysis failed")
                        continue
                    
                    if momentum['direction'] != 'SHORT':
                        momentum_rejections.append(f"{symbol} (PARABOLIC): direction is {momentum['direction']}, not SHORT")
                        logger.info(f"   ❌ {symbol} PARABOLIC: direction is {momentum['direction']}, need SHORT")
                        continue
                    
                    if 'PARABOLIC REVERSAL' not in momentum['reason']:
                        momentum_rejections.append(f"{symbol} (PARABOLIC): not a parabolic reversal pattern")
                        logger.info(f"   ❌ {symbol} PARABOLIC: not parabolic reversal pattern")
                        continue
                    
                    # ✅ PARABOLIC REVERSAL SHORT FOUND!
                    logger.info(f"✅ PARABOLIC REVERSAL SHORT found: {symbol}")
                    # Build signal and return immediately (highest priority!)
                    entry_price = momentum['entry_price']
                    
                    # 🔥 AGGRESSIVE PARABOLIC TP/SL - These exhausted pumps dump HARD!
                    # TP: 10% price move = 200% profit @ 20x leverage (2:1 R:R)
                    # SL: 5% price move = 100% loss @ 20x leverage
                    stop_loss = entry_price * (1 + 5.0 / 100)  # 5% SL = 100% loss at 20x
                    take_profit_1 = entry_price * (1 - 10.0 / 100)  # 10% TP = 200% profit at 20x 🚀
                    take_profit_2 = None  # Single aggressive TP for parabolic dumps
                    take_profit_3 = None
                    
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
                        'leverage': 20,  # 20x leverage for AGGRESSIVE parabolic shorts
                        '24h_change': gainer['change_percent'],
                        '24h_volume': gainer['volume_24h'],
                        'is_parabolic_reversal': True
                    }
            
            # PRIORITY 2: Regular analysis (shorts preferred, then longs)
            # 🔥 REMOVED COOLDOWN CHECK: TOP_GAINER SHORTS scan independently from SCALP activity
            # Cooldowns are now tracked per-symbol in database, not globally
            logger.info(f"📋 PRIORITY 2: Analyzing {len(gainers)} gainers for REGULAR signals...")
            
            for gainer in gainers:
                symbol = gainer['symbol']
                logger.info(f"   🔍 Analyzing {symbol} @ +{gainer['change_percent']}%...")
                
                # Analyze momentum
                momentum = await self.analyze_momentum(symbol)
                
                if not momentum:
                    momentum_rejections.append(f"{symbol}: momentum analysis returned None")
                    logger.info(f"   ❌ {symbol}: momentum analysis failed")
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
                
                # 🔥 PARABOLIC SHORTS BYPASS COOLDOWN!
                # Dedicated parabolic scanner doesn't check cooldown - we want ALL 50%+ reversals
                logger.info(f"🎯 Analyzing: {symbol} @ +{change_pct:.1f}% (parabolic scanner - no cooldown)")
                
                # 🚀 PARABOLIC EXHAUSTION LOGIC (balanced - not too strict, not too loose)
                # For 50%+ pumps: check overextension + exhaustion signs + trend context
                try:
                    # Get 5m candles for analysis
                    candles_5m = await self.fetch_candles(symbol, '5m', 20)
                    
                    if not candles_5m or len(candles_5m) < 14:
                        logger.info(f"  ❌ {symbol} - Insufficient 5m candle data")
                        continue
                    
                    # Calculate basic indicators
                    closes_5m = [float(c[4]) for c in candles_5m]
                    current_price = closes_5m[-1]
                    rsi_5m = self._calculate_rsi(closes_5m, 14)
                    
                    # Volume check (proper averaging with minimum sample guard)
                    volumes = [float(c[5]) for c in candles_5m]
                    current_volume = volumes[-1]
                    
                    # Need minimum 5 historical candles for volume context
                    if len(volumes) < 6:
                        logger.info(f"  ❌ {symbol} - Insufficient volume data ({len(volumes)} candles, need 6+)")
                        continue
                    
                    # Average last N candles (excluding current)
                    if len(volumes) >= 10:
                        vol_samples = volumes[-10:-1]  # Last 9 candles
                    else:
                        vol_samples = volumes[:-1]  # All but current
                    avg_volume = sum(vol_samples) / len(vol_samples)
                    volume_ratio = current_volume / avg_volume
                    
                    # Calculate EMA9 for overextension check
                    ema9 = self._calculate_ema(closes_5m, 9)  # Returns float, not list
                    price_to_ema9_dist = ((current_price - ema9) / ema9) * 100
                    
                    # 🔥 TREND VALIDATION: Must be overextended above EMA9
                    is_overextended = price_to_ema9_dist > 1.5  # >1.5% above EMA9 (relaxed from 2%)
                    
                    if not is_overextended:
                        logger.info(f"  ❌ {symbol} - Not overextended (only {price_to_ema9_dist:+.1f}% above EMA9, need >1.5%)")
                        continue
                    
                    # 🔥 EXHAUSTION DETECTION (3 signals - need any 2)
                    # Signal 1: High RSI (overbought)
                    high_rsi = rsi_5m >= 65  # Overbought territory
                    
                    # Signal 2: Upper wick rejection (selling pressure at top)
                    current_candle = candles_5m[-1]
                    current_open = float(current_candle[1])
                    current_high = float(current_candle[2])
                    current_low = float(current_candle[3])
                    wick_size = ((current_high - current_price) / current_price) * 100
                    has_rejection = wick_size >= 0.5  # 0.5%+ upper wick
                    
                    # Signal 3: Bearish candle or slowing bullish momentum
                    is_bearish_candle = current_price < current_open
                    prev_candle_open = float(candles_5m[-2][1])
                    prev_candle_close = float(candles_5m[-2][4])
                    prev_was_bullish = prev_candle_close > prev_candle_open
                    
                    # Only check slowing if previous was bullish (exhausting upward momentum)
                    if prev_was_bullish:
                        prev_candle_size = abs((prev_candle_close - prev_candle_open) / prev_candle_open) * 100
                        current_candle_size = abs((current_price - current_open) / current_open) * 100
                        slowing_bullish_momentum = current_candle_size < (prev_candle_size * 0.5)
                    else:
                        slowing_bullish_momentum = False
                    
                    slowing_momentum = is_bearish_candle or slowing_bullish_momentum
                    
                    # 🎯 RELAXED ENTRY CRITERIA (catch 50%+ exhausted pumps!)
                    # Path 1: 2/3 exhaustion signs (reasonable confirmation for 50%+ pumps)
                    # Path 2: Extreme RSI ≥70 (very overbought - likely to reverse)
                    exhaustion_signals = [high_rsi, has_rejection, slowing_momentum]
                    exhaustion_count = sum(exhaustion_signals)
                    good_volume = volume_ratio >= 1.0
                    extreme_rsi = rsi_5m >= 70  # Very overbought (relaxed from 75)
                    
                    # Entry if: (2/3 exhaustion OR RSI ≥70) + volume + overextension
                    has_strong_signal = exhaustion_count >= 2 or extreme_rsi
                    
                    if has_strong_signal and good_volume:
                        # Build exhaustion reason
                        reasons = []
                        if high_rsi:
                            reasons.append(f"RSI {rsi_5m:.0f} (overbought)")
                        if has_rejection:
                            reasons.append(f"{wick_size:.1f}% wick rejection")
                        if slowing_momentum:
                            reasons.append("Momentum slowing")
                        
                        exhaustion_reason = " + ".join(reasons)
                        # Confidence based on exhaustion count and RSI level
                        if exhaustion_count == 3:
                            confidence = 95  # All exhaustion signs = highest confidence
                        elif extreme_rsi and rsi_5m >= 75:
                            confidence = 93  # Extreme RSI ≥75 = very high confidence
                        elif extreme_rsi:
                            confidence = 90  # RSI ≥70 = high confidence
                        elif exhaustion_count == 2:
                            confidence = 88  # 2/3 exhaustion = good confidence
                        else:
                            confidence = 85  # Fallback
                        
                        logger.info(f"  ✅ {symbol} - PARABOLIC EXHAUSTION: {exhaustion_reason} | Vol: {volume_ratio:.1f}x")
                        
                        # Store as candidate (don't use analyze_momentum)
                        candidates.append({
                            'symbol': symbol,
                            'gainer': gainer,
                            'momentum': {
                                'direction': 'SHORT',
                                'confidence': confidence,
                                'entry_price': current_price,
                                'reason': f'🎯 PARABOLIC REVERSAL | +{change_pct:.1f}% exhausted | {exhaustion_reason} | Vol: {volume_ratio:.1f}x'
                            },
                            'score': change_pct * 0.4 + confidence * 0.6
                        })
                        continue
                    else:
                        skip_reasons = []
                        if not has_strong_signal:
                            if exhaustion_count < 2 and not extreme_rsi:
                                skip_reasons.append(f"Only {exhaustion_count}/3 exhaustion signs (need 2/3 OR RSI ≥70, currently {rsi_5m:.0f})")
                        if not good_volume:
                            skip_reasons.append(f"Low volume {volume_ratio:.1f}x")
                        logger.info(f"  ❌ {symbol} - {', '.join(skip_reasons)}")
                        continue
                        
                except Exception as e:
                    logger.error(f"  ❌ {symbol} - Error analyzing: {e}")
                    continue
            
            if not candidates:
                logger.info("No valid parabolic reversal candidates found")
                return None
            
            # Sort by score (highest first) and take best
            candidates.sort(key=lambda x: x['score'], reverse=True)
            best = candidates[0]
            
            logger.info(f"🏆 BEST PARABOLIC: {best['symbol']} (score: {best['score']:.1f})")
            
            # Build signal with AGGRESSIVE TP/SL (parabolic dumps crash HARD!)
            entry_price = best['momentum']['entry_price']
            
            # 🔥 AGGRESSIVE PARABOLIC TP/SL - 50%+ exhausted pumps dump violently!
            # TP: 10% price move = 200% profit @ 20x leverage (2:1 R:R)
            # SL: 5% price move = 100% loss @ 20x leverage
            stop_loss = entry_price * (1 + 5.0 / 100)  # 5% SL = 100% loss at 20x
            take_profit_1 = entry_price * (1 - 10.0 / 100)  # 10% TP = 200% profit at 20x 🚀
            take_profit_2 = None  # Single aggressive TP for parabolic dumps
            take_profit_3 = None
            
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
                'leverage': 20,  # 20x leverage for AGGRESSIVE parabolic shorts
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
        min_change: float = 8.0,
        max_change: float = 120.0,
        max_symbols: int = 10
    ) -> Optional[Dict]:
        """
        Generate LONG signals from EARLY-to-MID PUMP candidates (8-120% gains)
        
        Catches coins with substantial pumps (8%+) but ONLY with clear pullback confirmation
        Avoids exhausted pumps >120% which are more risky for reversals
        
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
                
                # 🔥 STRICT MODE: ONLY use safe pullback entry (no aggressive momentum)
                # This requires clear retracement confirmation before entering
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
    
    async def generate_scalp_signal(self, max_symbols: int = 100) -> Optional[Dict]:
        """
        🚀 SCALP MODE - Scans for BOTH LONG momentum and SHORT reversals
        
        LONG SCALPS: Momentum entries on bullish coins
        SHORT SCALPS: Exhausted pump reversals
        
        Returns best signal of either type
        """
        try:
            logger.info("⚡ SCALP MODE - Scanning for LONG momentum AND SHORT reversals...")
            
            # Get all top gainers - no filtering
            all_symbols = await self.get_top_gainers(limit=max_symbols, min_change_percent=0.0)
            
            if not all_symbols:
                logger.info("No symbols available for scalp scanning")
                return None
            
            logger.info(f"⚡ Scanning {len(all_symbols)} symbols for momentum/reversal scalps...")
            
            best_scalp = None
            best_confidence = 0
            
            for symbol_dict in all_symbols[:max_symbols]:
                symbol = symbol_dict['symbol']
                try:
                    # Check both LONG and SHORT opportunities
                    long_signal = await self.analyze_scalp_momentum(symbol)
                    short_signal = await self.analyze_scalp_reversal(symbol)
                    
                    # Pick the best opportunity (either LONG or SHORT)
                    signals = [s for s in [long_signal, short_signal] if s]
                    if not signals:
                        continue
                    
                    best_for_symbol = max(signals, key=lambda x: x['confidence'])
                    confidence = best_for_symbol.get('confidence', 82)
                    direction = best_for_symbol['direction']
                    entry_price = best_for_symbol['entry_price']
                    
                    # Calculate TP/SL for 40% profit/loss @ 20x (1:1 ratio)
                    if direction == 'LONG':
                        tp_price = entry_price * (1 + 0.020)  # 2.0% = 40% profit
                        sl_price = entry_price * (1 - 0.020)  # 2.0% = 40% loss
                    else:  # SHORT
                        tp_price = entry_price * (1 - 0.020)  # 2.0% down = 40% profit
                        sl_price = entry_price * (1 + 0.020)  # 2.0% up = 40% loss
                    
                    logger.info(f"⚡ {symbol} ({direction}): {best_for_symbol['reason']} | Confidence: {confidence}")
                    
                    if confidence > best_confidence:
                        best_confidence = confidence
                        best_scalp = {
                            'symbol': symbol,
                            'direction': direction,
                            'entry_price': entry_price,
                            'stop_loss': sl_price,
                            'take_profit': tp_price,
                            'take_profit_1': tp_price,
                            'confidence': confidence,
                            'reasoning': f'{best_for_symbol["reason"]} | 1:1 (40%/40%) @ 20x',
                            'trade_type': 'SCALP',
                            'leverage': 20,
                            'is_scalp': True,
                            'entry_pattern': 'Momentum/Reversal Entry'
                        }
                
                except Exception as e:
                    logger.debug(f"Error analyzing {symbol} for scalp: {e}")
                    continue
            
            if best_scalp:
                logger.info(f"✅ BEST SCALP: {best_scalp['symbol']} ({best_scalp['direction']}) @ {best_scalp['entry_price']:.8f} | Confidence: {best_confidence}%")
                return best_scalp
            
            logger.info(f"⚡ No scalp candidates found (scanned {len(all_symbols)} symbols)")
            return None
            
        except Exception as e:
            logger.error(f"Error in scalp signal generation: {e}")
            return None
    
    async def close(self):
        """Close HTTP client connection"""
        if self.client:
            await self.client.aclose()
            self.client = None


async def broadcast_scalp_signal_simple(signal_data):
    """
    Fire-and-forget scalp signal broadcast + auto-execution on Bitunix
    - Uses PostgreSQL advisory lock (ATOMIC duplicate prevention)
    - Prevents race condition where multiple concurrent scans signal same symbol
    - Direct httpx call (non-blocking)
    - Auto-executes on Bitunix with owner's position size
    """
    from app.database import SessionLocal
    from app.models import Signal, User, UserPreference, Trade
    from app.services.bitunix_trader import execute_bitunix_trade
    from datetime import datetime, timedelta
    from sqlalchemy import text
    import logging
    import os
    import httpx
    import asyncio
    
    logger = logging.getLogger(__name__)
    
    # Use separate DB session (fire-and-forget)
    db = SessionLocal()
    
    # 🔒 CREATE ADVISORY LOCK based on symbol+direction (prevents concurrent signals)
    # Hash symbol+direction to integer for advisory lock ID (PostgreSQL requires int)
    lock_key = f"SCALP:{signal_data['symbol']}:{signal_data['direction']}"
    lock_id = abs(hash(lock_key)) % (2**31 - 1)
    
    try:
        # 🔒 TRY ACQUIRE LOCK - non-blocking version to prevent freezes
        # Returns true if lock acquired, false if already locked
        result = db.execute(text(f"SELECT pg_try_advisory_lock({lock_id})")).scalar()
        if not result:
            logger.info(f"⏭️ SCALP SKIPPED: {signal_data['symbol']} ({signal_data['direction']}) - Lock already held (another process is handling this)")
            return
        logger.debug(f"🔒 SCALP LOCK ACQUIRED for {signal_data['symbol']} ({signal_data['direction']})")
        
        # 🛡️ CHECK PERSISTENT DUPLICATE (3-minute window) - AGGRESSIVE for high-frequency scalps!
        from app.models import ScalpSignal
        three_minutes_ago = datetime.utcnow() - timedelta(minutes=3)  # AGGRESSIVE: 3min (was 10min!)
        recent_scalp_signal = db.query(ScalpSignal).filter(
            ScalpSignal.symbol == signal_data['symbol'],
            ScalpSignal.direction == signal_data['direction'],
            ScalpSignal.created_at >= three_minutes_ago
        ).first()
        
        if recent_scalp_signal:
            minutes_ago = (datetime.utcnow() - recent_scalp_signal.created_at).total_seconds() / 60
            logger.info(f"⏭️ SCALP DUPLICATE BLOCKED: {signal_data['symbol']} ({signal_data['direction']}) - Already signaled {minutes_ago:.1f}m ago")
            return
        
        # Get owner FIRST (before any checks)
        owner = db.query(User).filter(User.telegram_id == "5603353066").first()
        if not owner:
            logger.warning("Owner not found")
            return
        
        # 🔒 PROTECTED: All checks happen WHILE holding lock (NO race condition!)
        # 1. Check for ANY open/pending positions for OWNER on this symbol
        from app.models import Trade
        
        open_position = db.query(Trade).filter(
            Trade.user_id == owner.id,
            Trade.symbol == signal_data['symbol'],
            Trade.status.in_(['open', 'pending', 'in_progress'])  # Any active status
        ).first()
        
        if open_position:
            logger.info(f"⏭️ SCALP SKIPPED: {signal_data['symbol']} - Owner already has {open_position.status} position (Trade ID: {open_position.id})")
            return
        
        # 2. Check for recent SCALP signal (30-MIN cooldown) - AGGRESSIVE for high-frequency scalps!
        thirty_minutes_ago = datetime.utcnow() - timedelta(minutes=30)  # AGGRESSIVE: 30min (was 2 hours!)
        recent_scalp = db.query(Signal).filter(
            Signal.symbol == signal_data['symbol'],
            Signal.signal_type == 'SCALP',
            Signal.created_at >= thirty_minutes_ago
        ).first()
        
        if recent_scalp:
            # Check if the previous signal's trade is still open
            previous_trade = db.query(Trade).filter(
                Trade.signal_id == recent_scalp.id,
                Trade.user_id == owner.id
            ).first()
            
            # If previous trade exists AND is closed (TP/SL hit), allow re-entry immediately
            if previous_trade and previous_trade.status in ['closed', 'tp_hit', 'sl_hit']:
                minutes_ago = (datetime.utcnow() - recent_scalp.created_at).total_seconds() / 60
                logger.info(f"✅ SCALP ALLOWED: {signal_data['symbol']} - Previous trade closed ({previous_trade.status}), re-entry permitted after {minutes_ago:.0f}m")
            else:
                # Trade doesn't exist or still active - apply cooldown
                minutes_ago = (datetime.utcnow() - recent_scalp.created_at).total_seconds() / 60
                logger.info(f"⏭️ SCALP SKIPPED (30min cooldown): {signal_data['symbol']} - Signal sent {minutes_ago:.0f}m ago (trade status: {previous_trade.status if previous_trade else 'not found'})")
                return
        
        # 🛡️ CREATE PERSISTENT DUPLICATE GUARD (BEFORE releasing lock!)
        from app.models import ScalpSignal
        scalp_guard = ScalpSignal(
            symbol=signal_data['symbol'],
            direction=signal_data['direction'],
            entry_price=signal_data['entry_price'],
            created_at=datetime.utcnow()
        )
        db.add(scalp_guard)
        
        # Save signal
        signal = Signal(
            symbol=signal_data['symbol'],
            direction=signal_data['direction'],
            entry_price=signal_data['entry_price'],
            stop_loss=signal_data['stop_loss'],
            take_profit=signal_data.get('take_profit'),
            take_profit_1=signal_data.get('take_profit_1'),
            confidence=signal_data['confidence'],
            reasoning=signal_data['reasoning'],
            signal_type='SCALP',
            pattern='Altcoin Support Bounce',
            timeframe='5m',
            created_at=datetime.utcnow()
        )
        db.add(signal)
        db.commit()
        db.refresh(signal)
        logger.info(f"✅ SCALP SAVED: {signal.symbol} ({signal.direction}) - Duplicate guard active for 10min")
        
        # 📢 BROADCAST to all users with scalp_mode_enabled (not just owner!)
        from app.models import User
        users_with_scalp = db.query(User).join(UserPreference).filter(
            UserPreference.scalp_mode_enabled == True
        ).all()
        
        if not users_with_scalp:
            logger.info("No users with SCALP mode enabled")
            return
        
        logger.info(f"📢 Broadcasting SCALP to {len(users_with_scalp)} users with scalp_mode_enabled")
        
        # Calculate PnL percentages for message
        tp_pnl = ((signal_data['take_profit'] - signal_data['entry_price']) / signal_data['entry_price']) * 100 * 20
        sl_pnl = ((signal_data['stop_loss'] - signal_data['entry_price']) / signal_data['entry_price']) * 100 * 20
        
        # Send signal to all users with SCALP enabled (parallel)
        token = os.getenv('TELEGRAM_BOT_TOKEN', '')
        async with httpx.AsyncClient(timeout=5) as client:
            send_tasks = []
            for user in users_with_scalp:
                prefs = user.preferences
                position_size_pct = getattr(prefs, 'scalp_position_size_percent', 1.0) if prefs else 1.0
                
                message = f"""⚡ <b>SCALP TRADE SIGNAL</b> ⚡

🟢 <b>{signal.symbol}</b>
Entry: ${signal.entry_price:.8f}
TP: ${signal.take_profit:.8f} ({tp_pnl:+.1f}% @ 20x)
SL: ${signal.stop_loss:.8f} ({sl_pnl:+.1f}% @ 20x)

💡 <b>Analysis:</b>
{signal_data['reasoning']}

🎯 <b>Confidence:</b> {signal_data['confidence']}%
📊 <b>Position Size:</b> {position_size_pct}% of account"""
                
                url = f"https://api.telegram.org/bot{token}/sendMessage"
                payload = {
                    "chat_id": int(user.telegram_id),
                    "text": message,
                    "parse_mode": "HTML"
                }
                send_tasks.append(client.post(url, json=payload))
            
            # Send all messages in parallel
            try:
                results = await asyncio.gather(*send_tasks, return_exceptions=True)
                success_count = sum(1 for r in results if not isinstance(r, Exception) and r.status_code == 200)
                logger.info(f"✅ SCALP signal sent to {success_count}/{len(users_with_scalp)} users")
            except Exception as e:
                logger.error(f"Error broadcasting SCALP: {e}")
        
        # 🚀 AUTO-EXECUTE on Bitunix for ALL users with scalp_mode_enabled (parallel execution)
        logger.info(f"🚀 Auto-executing SCALP on Bitunix for {len(users_with_scalp)} users")
        
        # Use semaphore for controlled parallel execution (max 5 concurrent)
        semaphore = asyncio.Semaphore(5)
        
        async def execute_for_user(user):
            async with semaphore:
                # 🔒 CRITICAL: Create separate DB session for each user to avoid session sharing across parallel tasks
                from app.database import get_db
                from app.models import User, Signal
                user_db = next(get_db())
                pending_trade = None
                
                try:
                    logger.info(f"🚀 Auto-executing SCALP for user {user.id}: {signal.symbol}")
                    
                    # 🔒 CRITICAL: Re-query fresh ORM objects from user's session to avoid cross-session errors
                    fresh_user = user_db.query(User).filter(User.id == user.id).first()
                    fresh_signal = user_db.query(Signal).filter(Signal.id == signal.id).first()
                    
                    if not fresh_user or not fresh_signal:
                        logger.error(f"Failed to re-query user {user.id} or signal {signal.id} from user session")
                        return
                    
                    # 🔒 Create PENDING trade to prevent duplicates
                    pending_trade = Trade(
                        user_id=fresh_user.id,
                        signal_id=fresh_signal.id,
                        symbol=fresh_signal.symbol,
                        direction=fresh_signal.direction,
                        entry_price=fresh_signal.entry_price,
                        stop_loss=fresh_signal.stop_loss,
                        take_profit=fresh_signal.take_profit,
                        status='pending',
                        position_size=0.0,
                        remaining_size=0.0,
                        trade_type='SCALP',
                        opened_at=datetime.utcnow()
                    )
                    user_db.add(pending_trade)
                    user_db.commit()
                    user_db.refresh(pending_trade)
                    logger.info(f"🔒 PENDING trade created for user {fresh_user.id} (ID: {pending_trade.id})")
                    
                    # Execute trade with timeout (using user's dedicated session and fresh ORM objects)
                    trade = await asyncio.wait_for(
                        execute_bitunix_trade(
                            signal=fresh_signal,  # Use fresh signal from user_db
                            user=fresh_user,  # Use fresh user from user_db
                            db=user_db,  # Use user-specific session
                            trade_type='SCALP',
                            leverage_override=20  # 20x for scalps
                        ),
                        timeout=30
                    )
                    logger.info(f"📞 execute_bitunix_trade returned: {trade} (type: {type(trade).__name__ if trade else 'None'})")
                    
                    # Delete the pending trade (execute_bitunix_trade creates the real one)
                    if pending_trade:
                        try:
                            user_db.delete(pending_trade)
                            user_db.commit()
                            logger.info(f"🔓 PENDING trade deleted (replaced with real trade)")
                        except Exception as del_err:
                            logger.error(f"Failed to delete pending trade: {del_err}")
                    
                    if trade:
                        logger.info(f"✅ SCALP EXECUTED for user {fresh_user.id}: Trade ID {trade.id} @ ${fresh_signal.entry_price:.8f}")
                        
                        # Send execution confirmation
                        prefs = fresh_user.preferences
                        position_size_pct = getattr(prefs, 'scalp_position_size_percent', 1.0) if prefs else 1.0
                        exec_msg = f"✅ <b>SCALP EXECUTED</b>\n{fresh_signal.symbol} @ ${fresh_signal.entry_price:.8f}\nPosition Size: {position_size_pct}% @ 20x"
                        
                        url = f"https://api.telegram.org/bot{token}/sendMessage"
                        payload = {
                            "chat_id": int(fresh_user.telegram_id),
                            "text": exec_msg,
                            "parse_mode": "HTML"
                        }
                        try:
                            async with httpx.AsyncClient(timeout=5) as client:
                                await client.post(url, json=payload)
                        except:
                            pass
                    else:
                        logger.warning(f"Failed to execute scalp trade for user {fresh_user.id}: {fresh_signal.symbol}")
                        # Send failure notification
                        fail_msg = f"❌ <b>SCALP EXECUTION FAILED</b>\n{fresh_signal.symbol}\nReason: Trade execution returned None"
                        
                        url = f"https://api.telegram.org/bot{token}/sendMessage"
                        payload = {
                            "chat_id": int(fresh_user.telegram_id),
                            "text": fail_msg,
                            "parse_mode": "HTML"
                        }
                        try:
                            async with httpx.AsyncClient(timeout=5) as client:
                                await client.post(url, json=payload)
                        except:
                            pass
                        
                except asyncio.TimeoutError:
                    logger.error(f"SCALP execution timeout (>30s) for user {user.id}: {signal.symbol}")
                    # Cleanup pending trade on timeout
                    if pending_trade:
                        try:
                            user_db.delete(pending_trade)
                            user_db.commit()
                            logger.info(f"🔓 PENDING trade deleted (timeout)")
                        except:
                            pass
                    
                    # Retrieve fresh objects for notification (in case they weren't set before timeout)
                    fresh_signal_ref = user_db.query(Signal).filter(Signal.id == signal.id).first()
                    fresh_user_ref = user_db.query(User).filter(User.id == user.id).first()
                    fail_msg = f"❌ <b>SCALP EXECUTION TIMEOUT</b>\n{fresh_signal_ref.symbol if fresh_signal_ref else signal.symbol}\nExecution took >30 seconds"
                    
                    url = f"https://api.telegram.org/bot{token}/sendMessage"
                    payload = {
                        "chat_id": int(fresh_user_ref.telegram_id if fresh_user_ref else user.telegram_id),
                        "text": fail_msg,
                        "parse_mode": "HTML"
                    }
                    try:
                        async with httpx.AsyncClient(timeout=5) as client:
                            await client.post(url, json=payload)
                    except:
                        pass
                        
                except Exception as e:
                    logger.error(f"❌ Scalp execution error for user {user.id}, {signal.symbol}: {e}", exc_info=True)
                    # Cleanup pending trade on error
                    if pending_trade:
                        try:
                            user_db.delete(pending_trade)
                            user_db.commit()
                            logger.info(f"🔓 PENDING trade deleted (error)")
                        except Exception as del_err:
                            logger.error(f"Failed to delete pending trade after error: {del_err}")
                    
                    # Retrieve fresh objects for notification (in case they weren't set before error)
                    fresh_signal_ref = user_db.query(Signal).filter(Signal.id == signal.id).first() if user_db else None
                    fresh_user_ref = user_db.query(User).filter(User.id == user.id).first() if user_db else None
                    # Send error notification to user
                    fail_msg = f"❌ <b>SCALP EXECUTION ERROR</b>\n{fresh_signal_ref.symbol if fresh_signal_ref else signal.symbol}\n\n<code>{str(e)[:200]}</code>"
                    
                    url = f"https://api.telegram.org/bot{token}/sendMessage"
                    payload = {
                        "chat_id": int(fresh_user_ref.telegram_id if fresh_user_ref else user.telegram_id),
                        "text": fail_msg,
                        "parse_mode": "HTML"
                    }
                    try:
                        async with httpx.AsyncClient(timeout=5) as client:
                            await client.post(url, json=payload)
                    except Exception as notify_err:
                        logger.error(f"Failed to send error notification: {notify_err}")
                
                finally:
                    # 🔒 CRITICAL: Always close user-specific database session
                    try:
                        user_db.close()
                        logger.debug(f"🔐 Closed DB session for user {user.id}")
                    except Exception as close_err:
                        logger.error(f"Failed to close user DB session: {close_err}")
        
        # Execute trades for all users in parallel
        await asyncio.gather(*[execute_for_user(user) for user in users_with_scalp], return_exceptions=True)
        logger.info(f"✅ SCALP broadcast and execution complete for {len(users_with_scalp)} users")
        
    except Exception as e:
        logger.error(f"Scalp broadcast error: {e}", exc_info=True)
    finally:
        # 🔓 ALWAYS RELEASE LOCK (even on error!)
        try:
            db.execute(text(f"SELECT pg_advisory_unlock({lock_id})"))
            logger.debug(f"🔓 SCALP LOCK RELEASED for {signal_data['symbol']}")
        except Exception as e:
            logger.error(f"Failed to release scalp lock: {e}")
        db.close()


async def broadcast_top_gainer_signal(bot, db_session):
    """
    Scan for signals and broadcast to users with top_gainers_mode_enabled
    Supports 3 modes: SHORTS_ONLY, LONGS_ONLY, or BOTH
    
    - SHORTS: Scan 28%+ gainers for mean reversion (wait for exhausted pumps!)
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
        scalp_signal = None
        
        # ═══════════════════════════════════════════════════════
        # SHORTS MODE: Scan 28%+ gainers for mean reversion - PRIORITY #1!
        # ═══════════════════════════════════════════════════════
        # 🔥 MOST PROFITABLE: Mean reversion on 28%+ pumps (80% profit @ 10x leverage)
        if wants_shorts:
            logger.info("🔴 ═══════════════════════════════════════════════════════")
            logger.info("🔴 SHORTS SCANNER - Priority #1 (20%+ mean reversion - AGGRESSIVE MODE!)")
            logger.info("🔴 ═══════════════════════════════════════════════════════")
            short_signal = await service.generate_top_gainer_signal(min_change_percent=20.0, max_symbols=10)
            
            if short_signal and short_signal['direction'] == 'SHORT':
                logger.info(f"✅ SHORT signal found: {short_signal['symbol']} @ +{short_signal.get('24h_change')}%")
        
        # ═══════════════════════════════════════════════════════
        # PARABOLIC MODE: Scan 50%+ exhausted pumps (Priority #2)
        # ═══════════════════════════════════════════════════════
        # Only if no regular SHORT found (PARABOLIC is riskier, SHORTS are more consistent)
        if wants_shorts and not short_signal:
            logger.info("🔥 ═══════════════════════════════════════════════════════")
            logger.info("🔥 PARABOLIC SCANNER - Priority #2 (50%+ exhausted dumps)")
            logger.info("🔥 ═══════════════════════════════════════════════════════")
            parabolic_signal = await service.generate_parabolic_dump_signal(min_change_percent=50.0, max_symbols=10)
            
            if parabolic_signal and parabolic_signal['direction'] == 'SHORT':
                logger.info(f"✅ PARABOLIC signal found: {parabolic_signal['symbol']} @ +{parabolic_signal.get('24h_change')}%")
        
        # ═══════════════════════════════════════════════════════
        # LONGS MODE: Scan FRESH pumps (5-50% gains with extended freshness windows)
        # ═══════════════════════════════════════════════════════
        # Priority #3 - Only if no SHORTS found
        if wants_longs and not short_signal and not parabolic_signal:
            logger.info("🟢 ═══════════════════════════════════════════════════════")
            logger.info("🟢 LONGS SCANNER - Analyzing EARLY pumps (5-20% range, avoid exhausted pumps!)")
            logger.info("🟢 ═══════════════════════════════════════════════════════")
            long_signal = await service.generate_early_pump_long_signal(min_change=5.0, max_change=50.0, max_symbols=20)
            
            if long_signal and long_signal['direction'] == 'LONG':
                logger.info(f"✅ LONG signal found: {long_signal['symbol']} @ +{long_signal.get('24h_change')}%")
        
        # ═══════════════════════════════════════════════════════
        # SCALP MODE: Quick 1-3% moves (Priority #4 - Lowest)
        # ═══════════════════════════════════════════════════════
        # Only if no other signals found (SCALP is lowest profit, highest frequency)
        if not short_signal and not parabolic_signal and not long_signal:
            logger.info("⚡ ═══════════════════════════════════════════════════════")
            logger.info("⚡ SCALP SCANNER - Priority #4 (Quick 1-3% micro-moves)")
            logger.info("⚡ ═══════════════════════════════════════════════════════")
            scalp_signal = await service.generate_scalp_signal(max_symbols=50)
            
            if scalp_signal:
                logger.info(f"✅ SCALP signal found: {scalp_signal['symbol']} @ {scalp_signal['entry_price']:.8f}")
        
        # If no signals at all, exit
        if not short_signal and not parabolic_signal and not long_signal and not scalp_signal:
            mode_str = []
            if wants_shorts:
                mode_str.append("SHORTS/PARABOLIC")
            if wants_longs:
                mode_str.append("LONGS")
            mode_str.append("SCALP")
            logger.info(f"No signals found for {' and '.join(mode_str) if mode_str else 'any mode'}")
            await service.close()
            return
        
        # Process SHORTS signal first (HIGHEST PRIORITY - most profitable!)
        if short_signal:
            await process_and_broadcast_signal(short_signal, users_with_mode, db_session, bot, service)
        
        # Process PARABOLIC signal next (if no regular SHORT found)
        elif parabolic_signal:
            await process_and_broadcast_signal(parabolic_signal, users_with_mode, db_session, bot, service)
        
        # Process LONG signal (Priority #3)
        elif long_signal:
            await process_and_broadcast_signal(long_signal, users_with_mode, db_session, bot, service)
        
        # Process SCALP signal last (LOWEST PRIORITY - quick but small profits)
        # Fire-and-forget with own DB session (no blocking)
        elif scalp_signal:
            logger.info(f"⚡ Scalp broadcast (async): {scalp_signal['symbol']}")
            asyncio.create_task(broadcast_scalp_signal_simple(scalp_signal))
        
        await service.close()
    
    except Exception as e:
        logger.error(f"Error in broadcast_top_gainer_signal: {e}", exc_info=True)


async def process_and_broadcast_signal(signal_data, users_with_mode, db_session, bot, service):
    """Helper function to process and broadcast a single signal with parallel execution
    
    🔒 CRITICAL: Uses PostgreSQL Advisory Locks to prevent duplicate signals
    - Normalizes symbol format before hashing (XAN/USDT → XAN)
    - Acquires advisory lock on {symbol}:{direction} key
    - Checks for duplicates within 5-minute window
    - Creates and broadcasts signal if no duplicate exists
    - Always releases lock in finally block
    """
    from app.models import Signal, Trade
    from app.services.bitunix_trader import execute_bitunix_trade
    from datetime import datetime, timedelta
    import logging
    import asyncio
    import random
    import hashlib
    from sqlalchemy import text
    
    logger = logging.getLogger(__name__)
    
    # 🔒 CRITICAL: PostgreSQL Advisory Lock for Duplicate Prevention
    lock_acquired = False
    lock_id = None
    signal = None
    
    try:
        # Normalize symbol (XAN/USDT → XAN, BTCUSDT → BTC) before hashing
        # This ensures "XAN/USDT" and "XANUSDT" map to the same lock
        normalized_symbol = signal_data['symbol'].replace('/USDT', '').replace('USDT', '')
        lock_key = f"{normalized_symbol}:{signal_data['direction']}"
        lock_id = int(hashlib.md5(lock_key.encode()).hexdigest()[:16], 16) % (2**63 - 1)
        
        # Try to acquire PostgreSQL advisory lock (NON-BLOCKING to prevent freezes)
        result = db_session.execute(text(f"SELECT pg_try_advisory_lock({lock_id})")).scalar()
        if not result:
            logger.info(f"⏭️ SIGNAL SKIPPED: {lock_key} - Lock already held (another process is handling this)")
            return
        lock_acquired = True
        logger.info(f"🔒 Advisory lock acquired: {lock_key} (ID: {lock_id})")
        
        # Check for duplicates (safe - we hold the lock!)
        recent_cutoff = datetime.utcnow() - timedelta(minutes=5)
        existing_signal = db_session.query(Signal).filter(
            Signal.symbol == signal_data['symbol'],
            Signal.direction == signal_data['direction'],
            Signal.signal_type.in_(['TOP_GAINER', 'PARABOLIC_REVERSAL']),
            Signal.created_at >= recent_cutoff
        ).first()
        
        if existing_signal:
            logger.warning(f"🚫 DUPLICATE PREVENTED: {signal_data['symbol']} {signal_data['direction']} (Signal #{existing_signal.id})")
            return
        
        # Create signal (protected by advisory lock - NO race condition!)
        signal_type = signal_data.get('trade_type', 'TOP_GAINER')
        signal = Signal(
            symbol=signal_data['symbol'],
            direction=signal_data['direction'],
            entry_price=signal_data['entry_price'],
            stop_loss=signal_data['stop_loss'],
            take_profit=signal_data.get('take_profit'),
            take_profit_1=signal_data.get('take_profit_1'),
            take_profit_2=signal_data.get('take_profit_2'),
            take_profit_3=signal_data.get('take_profit_3'),
            confidence=signal_data['confidence'],
            reasoning=signal_data['reasoning'],
            signal_type=signal_type,
            timeframe='5m',
            created_at=datetime.utcnow()
        )
        db_session.add(signal)
        db_session.flush()
        db_session.commit()
        db_session.refresh(signal)
        
        logger.info(f"✅ SIGNAL CREATED: {signal.symbol} {signal.direction} @ ${signal.entry_price} (24h: {signal_data.get('24h_change')}%)")
        
        # 📣 BROADCAST & EXECUTE SIGNAL (lock is still held throughout)
        # Check if parabolic reversal (aggressive 20x leverage)
        is_parabolic = signal_data.get('is_parabolic_reversal', False)
        
        # Build TP text - Direction-specific profits
        if is_parabolic and signal.direction == 'SHORT':
            # 🔥 PARABOLIC REVERSALS: Aggressive 20x leverage for exhausted 50%+ pumps
            tp_text = f"<b>TP:</b> ${signal.take_profit_1:.6f} (+200% @ 20x) 🚀💥"
            sl_text = "(-100% @ 20x)"  # All-in on exhausted pumps!
            rr_text = "2:1 risk-to-reward (AGGRESSIVE PARABOLIC DUMP!)"
        elif signal.direction == 'LONG' and signal.take_profit_2:
            # LONGS: Dual TPs (5% and 10%)
            tp_text = f"""<b>TP1:</b> ${signal.take_profit_1:.6f} (+25% @ 5x) 
<b>TP2:</b> ${signal.take_profit_2:.6f} (+50% @ 5x) 🎯"""
            sl_text = "(-20% @ 5x)"  # LONGS: 4% SL * 5x = 20%
            rr_text = "25% and 50% profit targets"
        elif signal.direction == 'SHORT':
            # SHORTS: Single TP at 80% (normal mean reversion)
            tp_text = f"<b>TP:</b> ${signal.take_profit_1:.6f} (up to +80% max) 🎯"
            sl_text = "(up to -80% max)"  # SHORTS: Display capped at 80%
            rr_text = "1:1 risk-to-reward"
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
                                # SHORTS: Single TP at 80% profit (16% price move ensures cap at all leverages)
                                targets = calculate_leverage_capped_targets(
                                    entry_price=signal.entry_price,
                                    direction='SHORT',
                                    tp_pcts=[16.0],
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
        db_session.rollback()
        logger.error(f"❌ Error in process_and_broadcast_signal: {e}", exc_info=True)
    
    finally:
        # 🔓 CRITICAL: Always release advisory lock (even on error/early return!)
        # This ensures the lock is ALWAYS released, preventing deadlocks
        if lock_acquired and lock_id is not None:
            try:
                # Normalize symbol again for logging (in case early return happened)
                norm_sym = signal_data['symbol'].replace('/USDT', '').replace('USDT', '')
                db_session.execute(text(f"SELECT pg_advisory_unlock({lock_id})"))
                logger.info(f"🔓 Lock released: {norm_sym}:{signal_data['direction']}")
            except Exception as unlock_error:
                logger.error(f"⚠️ Failed to release lock: {unlock_error}")
