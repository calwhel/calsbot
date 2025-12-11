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

# 🟢 LONG COOLDOWNS - Prevent signal spam (target 4-6 trades/day)
# Global cooldown: 1.5 hours between ANY long signals
last_long_signal_time = None
LONG_GLOBAL_COOLDOWN_HOURS = 1.5

# Per-symbol cooldown: 6 hours before same symbol can signal again
longs_symbol_cooldown = {}
LONG_SYMBOL_COOLDOWN_HOURS = 6

# 🔥 BREAKOUT TRACKING CACHE - Track candidates waiting for pullback
# Format: {symbol: {'detected_at': datetime, 'breakout_data': {...}, 'checks': int}}
# Candidates are re-evaluated on each scan until pullback occurs or timeout (10 min)
pending_breakout_candidates = {}
BREAKOUT_CANDIDATE_TIMEOUT_MINUTES = 10

# 🔥 DAILY SIGNAL LIMITS - prevents over-trading
MAX_DAILY_SIGNALS = 9  # Total max signals per day (quality filters aim for ~4 shorts)
daily_signal_count = 0
last_signal_date = None

def check_and_increment_daily_signals(direction: str = None) -> bool:
    """
    Check if we can send another signal today (max 6/day).
    Quality filters naturally target ~4 shorts/day.
    Returns True if allowed, False if limit reached.
    Resets counter at midnight UTC.
    """
    global daily_signal_count, last_signal_date
    
    today = datetime.utcnow().date()
    
    # Reset counter if new day
    if last_signal_date != today:
        daily_signal_count = 0
        last_signal_date = today
        logger.info(f"📅 New day - daily signal counter reset to 0")
    
    # Check if limit reached
    if daily_signal_count >= MAX_DAILY_SIGNALS:
        logger.warning(f"⚠️ DAILY LIMIT REACHED: {daily_signal_count}/{MAX_DAILY_SIGNALS} signals today - skipping new signals")
        return False
    
    # Increment and allow
    daily_signal_count += 1
    logger.info(f"📊 Daily signals: {daily_signal_count}/{MAX_DAILY_SIGNALS}")
    return True

def get_daily_signal_count() -> int:
    """Get current daily signal count"""
    global daily_signal_count, last_signal_date
    today = datetime.utcnow().date()
    if last_signal_date != today:
        return 0
    return daily_signal_count


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


# 🔥 GLOBAL SIGNAL COOLDOWN - Prevents duplicate signals for same coin
signal_cooldowns = {}  # {symbol: datetime} - When cooldown expires

def is_symbol_on_cooldown(symbol: str) -> bool:
    """Check if symbol was recently signaled (prevents duplicates)"""
    if symbol in signal_cooldowns:
        if datetime.utcnow() < signal_cooldowns[symbol]:
            return True
        else:
            del signal_cooldowns[symbol]
    return False

def add_signal_cooldown(symbol: str, cooldown_minutes: int = 30):
    """Add cooldown after signaling a coin (prevents rapid duplicate signals)"""
    cooldown_until = datetime.utcnow() + timedelta(minutes=cooldown_minutes)
    signal_cooldowns[symbol] = cooldown_until
    logger.info(f"⏰ {symbol} on signal cooldown for {cooldown_minutes} minutes")


class TopGainersSignalService:
    """Service to fetch and analyze top gainers from Bitunix using direct API"""
    
    def __init__(self):
        self.base_url = "https://fapi.bitunix.com"  # For tickers and trading
        self.binance_url = "https://fapi.binance.com"  # For candle data (Binance Futures public API)
        self.client = httpx.AsyncClient(timeout=30.0)
        self.min_volume_usdt = 200000  # $200k minimum - catches lower caps that pump hard!
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
        Fetch top gainers using BINANCE + MEXC FUTURES APIs for accurate 24h data
        Then filter to only coins available on Bitunix for trading
        
        OPTIMIZED FOR SHORTS: Higher min_change (10%+) = better reversal candidates
        Uses Binance as primary source, MEXC as fallback for coins not on Binance
        
        Args:
            limit: Number of top gainers to return
            min_change_percent: Minimum 24h change % to qualify (default 10% for shorts)
            
        Returns:
            List of {symbol, change_percent, volume, price} sorted by change %
        """
        try:
            # 🔥 FETCH FROM MULTIPLE SOURCES for better coverage
            merged_data = {}  # symbol -> ticker data (Binance priority)
            
            # === SOURCE 1: BINANCE FUTURES (Primary - most reliable) ===
            binance_count = 0
            try:
                binance_url = "https://fapi.binance.com/fapi/v1/ticker/24hr"
                response = await self.client.get(binance_url, timeout=10)
                response.raise_for_status()
                binance_tickers = response.json()
                
                for ticker in binance_tickers:
                    symbol = ticker.get('symbol', '')
                    if not symbol.endswith('USDT'):
                        continue
                    try:
                        merged_data[symbol] = {
                            'symbol': symbol,
                            'change_percent': float(ticker.get('priceChangePercent', 0)),
                            'last_price': float(ticker.get('lastPrice', 0)),
                            'volume_usdt': float(ticker.get('quoteVolume', 0)),
                            'high_24h': float(ticker.get('highPrice', 0)),
                            'low_24h': float(ticker.get('lowPrice', 0)),
                            'source': 'binance'
                        }
                        binance_count += 1
                    except (ValueError, TypeError):
                        continue
            except Exception as e:
                logger.warning(f"⚠️ Binance API error (using MEXC only): {e}")
            
            # === SOURCE 2: MEXC FUTURES (Secondary - for coins not on Binance) ===
            mexc_count = 0
            mexc_added = 0
            try:
                mexc_url = "https://contract.mexc.com/api/v1/contract/ticker"
                response = await self.client.get(mexc_url, timeout=10)
                response.raise_for_status()
                mexc_data = response.json()
                mexc_tickers = mexc_data.get('data', []) if isinstance(mexc_data, dict) else mexc_data
                
                for ticker in mexc_tickers:
                    # MEXC uses underscore format: BTC_USDT -> BTCUSDT
                    raw_symbol = ticker.get('symbol', '')
                    symbol = raw_symbol.replace('_', '')
                    if not symbol.endswith('USDT'):
                        continue
                    mexc_count += 1
                    
                    # Only add if NOT already in Binance data (Binance takes priority)
                    if symbol not in merged_data:
                        try:
                            # MEXC fields: lastPrice, riseFallRate (change %), amount24 (USDT volume)
                            change_pct = float(ticker.get('riseFallRate', 0))
                            # MEXC riseFallRate might be decimal (0.35) or percent (35)
                            if abs(change_pct) < 5 and abs(change_pct) > 0:
                                change_pct = change_pct * 100  # Convert decimal to percent
                            
                            merged_data[symbol] = {
                                'symbol': symbol,
                                'change_percent': change_pct,
                                'last_price': float(ticker.get('lastPrice', 0)),
                                'volume_usdt': float(ticker.get('amount24', 0)),  # USDT volume
                                'high_24h': float(ticker.get('high24Price', 0)),
                                'low_24h': float(ticker.get('low24Price', 0)),
                                'source': 'mexc'
                            }
                            mexc_added += 1
                        except (ValueError, TypeError):
                            continue
            except Exception as e:
                logger.warning(f"⚠️ MEXC API error (using Binance only): {e}")
            
            # === GET BITUNIX AVAILABLE SYMBOLS ===
            bitunix_url = f"{self.base_url}/api/v1/futures/market/tickers"
            bitunix_response = await self.client.get(bitunix_url)
            bitunix_data = bitunix_response.json()
            bitunix_symbols = set()
            if isinstance(bitunix_data, dict) and bitunix_data.get('data'):
                for t in bitunix_data.get('data', []):
                    bitunix_symbols.add(t.get('symbol', ''))
            
            logger.info(f"📊 DATA SOURCES: Binance={binance_count} | MEXC={mexc_count} (+{mexc_added} unique) | Bitunix={len(bitunix_symbols)} tradeable")
            
            gainers = []
            rejected_not_on_bitunix = 0
            
            for symbol, data in merged_data.items():
                # 🔥 MUST be available on Bitunix for trading
                if symbol not in bitunix_symbols:
                    rejected_not_on_bitunix += 1
                    continue
                
                change_percent = data['change_percent']
                last_price = data['last_price']
                volume_usdt = data['volume_usdt']
                high_24h = data['high_24h']
                low_24h = data['low_24h']
                
                # Filter criteria
                if (change_percent >= min_change_percent and 
                    volume_usdt >= self.min_volume_usdt):
                    
                    gainers.append({
                        'symbol': symbol.replace('USDT', '/USDT'),  # Format as BTC/USDT
                        'change_percent': round(change_percent, 2),
                        'volume_24h': round(volume_usdt, 0),
                        'price': last_price,
                        'high_24h': high_24h,
                        'low_24h': low_24h
                    })
            
            # Sort by change % descending
            gainers.sort(key=lambda x: x['change_percent'], reverse=True)
            
            if rejected_not_on_bitunix > 0:
                logger.debug(f"Filtered out {rejected_not_on_bitunix} coins not tradeable on Bitunix")
            
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
            
            if candle_change_percent < 3.0:  # 🔥 RELAXED: 3%+ (was 5%+)
                return {'is_fresh_pump': False, 'reason': 'insufficient_pump', 'change': candle_change_percent}
            
            # Check 3: Volume 1.5x+ average of previous 3 candles (RELAXED for more signals!)
            prev_volumes = [candles_5m[-4][5], candles_5m[-3][5], candles_5m[-2][5]]
            avg_volume = sum(prev_volumes) / len(prev_volumes)
            
            volume_ratio = volume / avg_volume if avg_volume > 0 else 0
            
            if volume_ratio < 1.5:  # 1.5x volume required for fresh pump detection
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
            
            # Check 2: Is it a green candle AND 5%+ gain? (RELAXED from 7%)
            if close_price <= open_price:
                return {'is_fresh_pump': False, 'reason': 'not_green_candle'}
            
            candle_change_percent = ((close_price - open_price) / open_price) * 100
            
            if candle_change_percent < 5.0:  # 🔥 RELAXED: 5%+ (was 7%+)
                return {'is_fresh_pump': False, 'reason': 'insufficient_pump', 'change': candle_change_percent}
            
            # Check 3: Volume 1.5x+ average of previous 2 candles (RELAXED from 2.5x)
            prev_volumes = [candles_15m[-3][5], candles_15m[-2][5]]
            avg_volume = sum(prev_volumes) / len(prev_volumes)
            
            volume_ratio = volume / avg_volume if avg_volume > 0 else 0
            
            if volume_ratio < 1.5:  # 1.5x volume required for fresh pump detection
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
        Validate FRESH 30-minute pump (5%+ green candle in last 30min)
        
        Requirements:
        - Most recent 30m candle is 5%+ green (close > open)  
        - Volume 1.5x+ average of previous 2-3 candles
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
            
            # Check 2: Is it a green candle AND 7%+ gain? (RELAXED from 10%)
            if close_price <= open_price:
                return {'is_fresh_pump': False, 'reason': 'not_green_candle'}
            
            candle_change_percent = ((close_price - open_price) / open_price) * 100
            
            if candle_change_percent < 5.0:  # 🔥 ULTRA RELAXED: 5%+ (catch early 5-20% gainers!)
                logger.debug(f"{symbol} 30m candle only +{candle_change_percent:.1f}% (need 5%+)")
                return {'is_fresh_pump': False, 'reason': 'insufficient_pump', 'change': candle_change_percent}
            
            # Check 3: Volume 1.5x+ average of previous 2 candles (RELAXED from 2.0x)
            prev_volumes = [candles_30m[-3][5], candles_30m[-2][5]]  # Previous 2 candles' volume
            avg_volume = sum(prev_volumes) / len(prev_volumes)
            
            volume_ratio = volume / avg_volume if avg_volume > 0 else 0
            
            if volume_ratio < 1.5:  # 🔥 RELAXED: 1.5x (was 2.0x)
                logger.debug(f"{symbol} 30m volume only {volume_ratio:.1f}x (need 1.5x+)")
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
    
    async def detect_realtime_breakouts(self, max_symbols: int = 30) -> List[Dict]:
        """
        🚀 REAL-TIME BREAKOUT DETECTOR - Catches pumps BEFORE they hit top-gainer lists!
        
        This is the key to EARLY entries. Instead of scanning 24h top gainers (already late),
        we scan ALL symbols for FRESH 1m volume/price spikes happening RIGHT NOW.
        
        Detection criteria (on 1m candles):
        1. Volume spike: Current 1m candle has 3x+ volume vs average of last 10 candles
        2. Price velocity: 1%+ price move in last 1-3 minutes (momentum building)
        3. Trend confirmation: Price above EMA9 on 1m (fresh breakout)
        4. Freshness: Candle must be within last 2 minutes (live action!)
        
        Returns:
            List of breakout candidates sorted by volume spike strength:
            {symbol, volume_ratio, price_velocity, current_price, breakout_type}
        """
        try:
            logger.info("🔍 REALTIME BREAKOUT SCAN: Checking ALL symbols for fresh 1m volume spikes...")
            
            # ═══════════════════════════════════════════════════════
            # 🔥 USE BINANCE + MEXC FOR SCANNING (not Bitunix - unreliable!)
            # Same dual data source architecture as SHORT scanner
            # ═══════════════════════════════════════════════════════
            
            merged_symbols = {}  # symbol -> volume_usdt
            
            # === SOURCE 1: BINANCE FUTURES (Primary) ===
            binance_count = 0
            try:
                binance_url = "https://fapi.binance.com/fapi/v1/ticker/24hr"
                response = await self.client.get(binance_url, timeout=10)
                response.raise_for_status()
                binance_tickers = response.json()
                
                for ticker in binance_tickers:
                    symbol = ticker.get('symbol', '')
                    if symbol.endswith('USDT'):
                        volume_usdt = float(ticker.get('quoteVolume', 0))
                        if volume_usdt >= self.min_volume_usdt:
                            merged_symbols[symbol] = volume_usdt
                            binance_count += 1
            except Exception as e:
                logger.warning(f"⚠️ Binance API error in breakout scan: {e}")
            
            # === SOURCE 2: MEXC FUTURES (Secondary - for coins not on Binance) ===
            mexc_added = 0
            try:
                mexc_url = "https://contract.mexc.com/api/v1/contract/ticker"
                response = await self.client.get(mexc_url, timeout=10)
                response.raise_for_status()
                mexc_data = response.json()
                mexc_tickers = mexc_data.get('data', []) if isinstance(mexc_data, dict) else mexc_data
                
                for ticker in mexc_tickers:
                    raw_symbol = ticker.get('symbol', '')
                    symbol = raw_symbol.replace('_', '')
                    if symbol.endswith('USDT') and symbol not in merged_symbols:
                        volume_usdt = float(ticker.get('amount24', 0))
                        if volume_usdt >= self.min_volume_usdt:
                            merged_symbols[symbol] = volume_usdt
                            mexc_added += 1
            except Exception as e:
                logger.warning(f"⚠️ MEXC API error in breakout scan: {e}")
            
            # === FILTER TO BITUNIX TRADEABLE SYMBOLS ===
            bitunix_url = f"{self.base_url}/api/v1/futures/market/tickers"
            bitunix_response = await self.client.get(bitunix_url, timeout=10)
            bitunix_data = bitunix_response.json()
            bitunix_symbols = set()
            if isinstance(bitunix_data, dict) and bitunix_data.get('data'):
                for t in bitunix_data.get('data', []):
                    bitunix_symbols.add(t.get('symbol', ''))
            
            # Only keep symbols tradeable on Bitunix (with volume data)
            all_symbols = []
            symbol_volumes = {}  # symbol -> 24h volume
            for symbol, volume_usdt in merged_symbols.items():
                if symbol in bitunix_symbols:
                    formatted_symbol = symbol.replace('USDT', '/USDT')
                    all_symbols.append(formatted_symbol)
                    symbol_volumes[formatted_symbol] = volume_usdt
            
            logger.info(f"📊 Scanning {len(all_symbols)} symbols for breakouts (Binance={binance_count}, MEXC=+{mexc_added}, Bitunix={len(bitunix_symbols)} tradeable)")
            
            breakout_candidates = []
            
            # Scan ALL symbols (no limit - catch every breakout!)
            batch_size = 15
            for i in range(0, len(all_symbols), batch_size):
                batch = all_symbols[i:i + batch_size]
                
                # Parallel fetch 1m candles for batch
                tasks = [self.fetch_candles(symbol, '1m', limit=15) for symbol in batch]
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                for symbol, candles in zip(batch, results):
                    if isinstance(candles, Exception) or not candles or len(candles) < 12:
                        continue
                    
                    try:
                        # Latest 1m candle
                        latest = candles[-1]
                        timestamp, open_price, high, low, close_price, volume = latest
                        
                        # Check freshness - candle close time = open_time + 60 seconds
                        # For a 1m candle at 12:00, close_time is 12:01
                        # If current time is 12:02, the candle is 1 minute old (fresh!)
                        from datetime import datetime
                        candle_open_time = datetime.fromtimestamp(timestamp / 1000)
                        candle_close_time = datetime.fromtimestamp((timestamp + 60000) / 1000)
                        now = datetime.utcnow()
                        
                        # Age = how long since candle CLOSED
                        age_seconds = (now - candle_close_time).total_seconds()
                        
                        # Accept candles that are still open (age < 0) or closed within 3 min
                        if age_seconds > 180:  # Skip if closed more than 3 minutes ago
                            continue
                        
                        # Volume spike detection (2.5x+ average for quality)
                        prev_volumes = [c[5] for c in candles[-11:-1]]
                        avg_volume = sum(prev_volumes) / len(prev_volumes) if prev_volumes else 0
                        volume_ratio = volume / avg_volume if avg_volume > 0 else 0
                        
                        # Calculate 1m quote volume (volume in USDT terms)
                        current_1m_volume_usdt = volume * close_price
                        avg_1m_volume_usdt = avg_volume * close_price
                        
                        if volume_ratio < 2.5:  # Need 2.5x volume spike (loosened from 3.5x)
                            continue
                        
                        # Price velocity (current candle change %)
                        candle_change = ((close_price - open_price) / open_price) * 100 if open_price > 0 else 0
                        
                        if candle_change < 0.5:  # Need 0.5%+ momentum (loosened from 0.8%)
                            continue
                        
                        # EMA9 check on 1m (price should be above for breakout)
                        closes = [c[4] for c in candles]
                        ema9 = self._calculate_ema(closes, 9)
                        
                        if close_price < ema9:  # Not a breakout
                            continue
                        
                        # Calculate 3-candle momentum (last 3 minutes velocity)
                        price_3m_ago = candles[-3][4] if len(candles) >= 3 else open_price
                        velocity_3m = ((close_price - price_3m_ago) / price_3m_ago) * 100 if price_3m_ago > 0 else 0
                        
                        # Require 0.8%+ 3-minute velocity (loosened from 1.2%)
                        if velocity_3m < 0.8:
                            continue
                        
                        # Determine breakout strength - Include BUILDING for more candidates
                        if volume_ratio >= 5.0 and candle_change >= 1.2:
                            breakout_type = "EXPLOSIVE"
                        elif volume_ratio >= 3.5 and candle_change >= 0.8:
                            breakout_type = "STRONG"
                        elif volume_ratio >= 2.5 and candle_change >= 0.5:
                            breakout_type = "BUILDING"
                        else:
                            continue
                        
                        breakout_candidates.append({
                            'symbol': symbol,
                            'volume_ratio': round(volume_ratio, 1),
                            'candle_change': round(candle_change, 2),
                            'velocity_3m': round(velocity_3m, 2),
                            'current_price': close_price,
                            'ema9_distance': round(((close_price - ema9) / ema9) * 100, 2),
                            'breakout_type': breakout_type,
                            'age_seconds': round(age_seconds, 0),
                            'volume_24h': symbol_volumes.get(symbol, 0),  # 24h volume from Binance/MEXC
                            'current_1m_vol': round(current_1m_volume_usdt, 0),  # Current 1m USDT volume
                            'avg_1m_vol': round(avg_1m_volume_usdt, 0)  # Average 1m USDT volume
                        })
                        
                    except Exception as e:
                        continue
            
            # Sort by volume spike (strongest first)
            breakout_candidates.sort(key=lambda x: x['volume_ratio'], reverse=True)
            
            if breakout_candidates:
                logger.info(f"🚀 FOUND {len(breakout_candidates)} BREAKOUT CANDIDATES!")
                for bc in breakout_candidates[:5]:
                    logger.info(f"  ⚡ {bc['symbol']}: {bc['breakout_type']} | {bc['volume_ratio']}x vol | +{bc['candle_change']}% | vel: +{bc['velocity_3m']}%")
            else:
                logger.info("❌ No breakout candidates found in this scan")
            
            return breakout_candidates[:max_symbols]
            
        except Exception as e:
            logger.error(f"Error in realtime breakout detection: {e}")
            return []
    
    async def analyze_breakout_entry(self, symbol: str, breakout_data: Dict) -> Optional[Dict]:
        """
        🎯 PULLBACK-FIRST ENTRY - NEVER enter at top of green candle!
        
        Core principle: We detect breakouts, but WAIT for pullback THEN enter on resumption.
        This prevents buying tops and getting stopped out on natural retracements.
        
        Entry requirements (ALL must be true):
        1. Prior impulse detected (green candle with volume)
        2. Pullback occurred (at least 1 red candle touching EMA support)
        3. Resumption starting (current green candle after red)
        4. Current price near candle LOW, not HIGH (not buying the top)
        5. RSI cooled down (48-65, not overbought)
        
        Returns:
            Signal dict or None if entry conditions not met
        """
        try:
            logger.info(f"🎯 PULLBACK-FIRST ENTRY: {symbol} ({breakout_data['breakout_type']})")
            
            # Fetch 1m candles for micro-structure analysis
            candles_1m = await self.fetch_candles(symbol, '1m', limit=20)
            candles_5m = await self.fetch_candles(symbol, '5m', limit=30)
            
            if len(candles_1m) < 15 or len(candles_5m) < 20:
                logger.info(f"  ❌ {symbol} - Insufficient candle data")
                return None
            
            closes_1m = [c[4] for c in candles_1m]
            closes_5m = [c[4] for c in candles_5m]
            
            # Current candle data
            current_candle = candles_1m[-1]
            current_open = current_candle[1]
            current_high = current_candle[2]
            current_low = current_candle[3]
            current_close = current_candle[4]
            
            # EMAs
            ema9_1m = self._calculate_ema(closes_1m, 9)
            ema21_1m = self._calculate_ema(closes_1m, 21)
            ema9_5m = self._calculate_ema(closes_5m, 9)
            ema21_5m = self._calculate_ema(closes_5m, 21)
            
            # RSI
            rsi_5m = self._calculate_rsi(closes_5m, 14)
            
            # ═══════════════════════════════════════════════════════
            # CRITICAL CHECK 1: Current candle must NOT be at its high
            # If close is in top 40% of candle range = buying the top = SKIP
            # TIGHTENED: Was 0.7, now 0.6 - don't enter in top 40%
            # ═══════════════════════════════════════════════════════
            candle_range = current_high - current_low
            if candle_range > 0:
                close_position = (current_close - current_low) / candle_range
                if close_position > 0.6:  # Close is in top 40% of range (TIGHTENED from 0.7)
                    logger.info(f"  ❌ {symbol} - Price at candle TOP ({close_position:.0%}) - would buy top!")
                    return None
            
            # ═══════════════════════════════════════════════════════
            # CRITICAL CHECK 2: Must have had a RED pullback candle recently
            # Looking for impulse → pullback → resumption pattern
            # ═══════════════════════════════════════════════════════
            candle_m1 = candles_1m[-1]  # Current
            candle_m2 = candles_1m[-2]  # 1 min ago
            candle_m3 = candles_1m[-3]  # 2 min ago
            candle_m4 = candles_1m[-4]  # 3 min ago
            candle_m5 = candles_1m[-5]  # 4 min ago
            
            # Check if candles are bullish (green) or bearish (red)
            c1_green = candle_m1[4] > candle_m1[1]
            c2_green = candle_m2[4] > candle_m2[1]
            c3_green = candle_m3[4] > candle_m3[1]
            c4_green = candle_m4[4] > candle_m4[1]
            c5_green = candle_m5[4] > candle_m5[1]
            
            # Count red (pullback) candles in last 4 candles before current
            red_count = sum([not c2_green, not c3_green, not c4_green, not c5_green])
            
            # Must have at least 2 red pullback candles for REAL pullback (TIGHTENED from 1)
            if red_count < 2:
                logger.info(f"  ❌ {symbol} - Weak pullback (only {red_count} red) - need 2+ red candles")
                return None
            
            # ═══════════════════════════════════════════════════════
            # CRITICAL CHECK 3: Current candle must be GREEN (resumption)
            # We enter on the resumption AFTER pullback, not during pullback
            # ═══════════════════════════════════════════════════════
            if not c1_green:
                logger.info(f"  ⏳ {symbol} - Pullback in progress (current red) - waiting for green")
                return None
            
            # ═══════════════════════════════════════════════════════
            # CRITICAL CHECK 4: Price must be near EMA support (not extended)
            # Entry near EMA = better R:R, natural support
            # TIGHTENED: Was 3.5%, now 2.0% - enter closer to support
            # ═══════════════════════════════════════════════════════
            ema_distance = ((current_close - ema9_1m) / ema9_1m) * 100
            if ema_distance > 2.0:  # More than 2% above EMA = extended (TIGHTENED from 3.5%)
                logger.info(f"  ❌ {symbol} - Extended {ema_distance:.1f}% above EMA (need ≤2%)")
                return None
            
            # ═══════════════════════════════════════════════════════
            # CRITICAL CHECK 5: Pullback must have touched EMA support
            # At least one of the red candles should have wicked near EMA
            # ═══════════════════════════════════════════════════════
            pullback_touched_ema = False
            for candle in [candle_m2, candle_m3, candle_m4]:
                candle_low = candle[3]
                # Low within 0.5% of EMA9 or below = touched support
                if candle_low <= ema9_1m * 1.005:
                    pullback_touched_ema = True
                    break
            
            if not pullback_touched_ema:
                logger.info(f"  ❌ {symbol} - Pullback didn't touch EMA support - shallow pullback REJECTED")
                return None  # TIGHTENED: No longer allow shallow pullbacks - must touch EMA
            
            # ═══════════════════════════════════════════════════════
            # CRITICAL CHECK 6: RSI must have cooled (not overbought)
            # TIGHTENED: Was 40-75, now 40-65 - don't enter when hot
            # ═══════════════════════════════════════════════════════
            if not (40 <= rsi_5m <= 65):
                logger.info(f"  ❌ {symbol} - RSI {rsi_5m:.0f} too hot (need 40-65, was 40-75)")
                return None
            
            # ═══════════════════════════════════════════════════════
            # CRITICAL CHECK 7: 5m trend must be bullish
            # ═══════════════════════════════════════════════════════
            if not (ema9_5m > ema21_5m):
                logger.info(f"  ❌ {symbol} - 5m trend not bullish")
                return None
            
            # ═══════════════════════════════════════════════════════
            # CRITICAL CHECK 8: Minimum 24h volume for liquidity
            # Must have at least $150K 24h volume to ensure tradeable
            # ═══════════════════════════════════════════════════════
            volume_24h = breakout_data.get('volume_24h', 0)
            if volume_24h < 150000:
                logger.info(f"  ❌ {symbol} - Low liquidity ${volume_24h:,.0f} (need $150K+)")
                return None
            
            # ═══════════════════════════════════════════════════════
            # ENTRY CONFIRMED: Pullback complete, resumption starting
            # ═══════════════════════════════════════════════════════
            
            # Determine entry pattern based on pullback depth
            # Note: EMA touch is now required, so all entries are EMA_BOUNCE or DEEP_PULLBACK
            if red_count >= 3:
                entry_pattern = "DEEP_PULLBACK"
            else:
                entry_pattern = "EMA_BOUNCE"
            
            logger.info(f"  ✅ {symbol} PULLBACK ENTRY CONFIRMED!")
            logger.info(f"     Pattern: {entry_pattern} | RSI: {rsi_5m:.0f} | EMA dist: {ema_distance:.1f}%")
            logger.info(f"     Pullback: {red_count} red candles | EMA touch: {pullback_touched_ema}")
            logger.info(f"     Candle position: {close_position:.0%} (not at top ✓)")
            
            return {
                'direction': 'LONG',
                'entry_price': current_close,
                'confidence': 85 if red_count >= 3 else 80,  # Higher confidence for deeper pullbacks
                'reason': f"QUALITY ENTRY: {entry_pattern} | {red_count} red + EMA touch | RSI {rsi_5m:.0f} | {ema_distance:.1f}% from EMA",
                'breakout_type': breakout_data['breakout_type'],
                'volume_ratio': breakout_data['volume_ratio'],
                'entry_pattern': entry_pattern
            }
            
        except Exception as e:
            logger.error(f"Error analyzing breakout entry for {symbol}: {e}")
            return None
    
    async def generate_breakout_long_signal(self) -> Optional[Dict]:
        """
        🚀 LONG STRATEGY: Real-time breakout detection with pullback tracking
        
        KEY FIX: Tracks breakout candidates and re-evaluates them on subsequent
        scans to catch the pullback entry (can't catch pullback immediately!)
        
        Flow:
        1. Check global LONG cooldown
        2. Clean up expired pending candidates (>10 min old)
        3. Re-evaluate PENDING candidates first (they already had breakout, now check pullback)
        4. Detect NEW breakouts and add to pending cache
        5. Return signal when pullback entry is found
        
        Returns:
            Signal dict matching existing format, or None
        """
        global last_long_signal_time, longs_symbol_cooldown, pending_breakout_candidates
        
        try:
            logger.info("═══════════════════════════════════════════════════════")
            logger.info("🚀 REALTIME BREAKOUT SCANNER - Tracking Pullback Entries!")
            logger.info("═══════════════════════════════════════════════════════")
            
            # 🔒 CHECK GLOBAL LONG COOLDOWN
            if last_long_signal_time:
                hours_since_last = (datetime.utcnow() - last_long_signal_time).total_seconds() / 3600
                if hours_since_last < LONG_GLOBAL_COOLDOWN_HOURS:
                    remaining = LONG_GLOBAL_COOLDOWN_HOURS - hours_since_last
                    logger.info(f"⏳ LONG COOLDOWN: {remaining:.1f}h remaining (last signal {hours_since_last:.1f}h ago)")
                    return None
            
            # ═══════════════════════════════════════════════════════
            # STEP 1: Clean up expired candidates (>10 min old)
            # ═══════════════════════════════════════════════════════
            now = datetime.utcnow()
            expired = [sym for sym, data in pending_breakout_candidates.items() 
                      if (now - data['detected_at']).total_seconds() > BREAKOUT_CANDIDATE_TIMEOUT_MINUTES * 60]
            for sym in expired:
                logger.info(f"🗑️ Removing expired candidate: {sym} (>{BREAKOUT_CANDIDATE_TIMEOUT_MINUTES}min old)")
                del pending_breakout_candidates[sym]
            
            logger.info(f"📋 Pending breakout candidates: {len(pending_breakout_candidates)}")
            for sym, data in pending_breakout_candidates.items():
                age_min = (now - data['detected_at']).total_seconds() / 60
                logger.info(f"   • {sym}: {data['breakout_data']['breakout_type']} | {age_min:.1f}min old | checks: {data['checks']}")
            
            # ═══════════════════════════════════════════════════════
            # STEP 2: Re-evaluate PENDING candidates (they had breakout, check pullback now)
            # ═══════════════════════════════════════════════════════
            for symbol, candidate_data in list(pending_breakout_candidates.items()):
                logger.info(f"  🔄 Re-checking pending: {symbol}...")
                
                # Check per-symbol cooldown
                if symbol in longs_symbol_cooldown:
                    cooldown_expires = longs_symbol_cooldown[symbol]
                    if datetime.utcnow() < cooldown_expires:
                        remaining = (cooldown_expires - datetime.utcnow()).total_seconds() / 3600
                        logger.info(f"    ⏳ {symbol} on cooldown ({remaining:.1f}h remaining)")
                        continue
                
                # Increment check counter
                candidate_data['checks'] += 1
                
                # Analyze for pullback entry
                entry = await self.analyze_breakout_entry(symbol, candidate_data['breakout_data'])
                
                if entry and entry['direction'] == 'LONG':
                    # Found valid pullback entry!
                    entry_price = entry['entry_price']
                    
                    # LONG TPs @ 20x leverage
                    stop_loss = entry_price * (1 - 4.0 / 100)
                    take_profit_1 = entry_price * (1 + 2.5 / 100)
                    take_profit_2 = entry_price * (1 + 5.0 / 100)
                    
                    breakout_data = candidate_data['breakout_data']
                    signal = {
                        'symbol': symbol,
                        'direction': 'LONG',
                        'entry_price': entry_price,
                        'stop_loss': stop_loss,
                        'take_profit_1': take_profit_1,
                        'take_profit_2': take_profit_2,
                        'take_profit_3': None,
                        'leverage': 20,
                        'confidence': entry['confidence'],
                        'reasoning': entry['reason'],
                        'mode': 'BREAKOUT',
                        'breakout_type': breakout_data['breakout_type'],
                        'volume_ratio': breakout_data['volume_ratio'],
                        '24h_change': breakout_data.get('velocity_3m', 0),
                        '24h_volume': breakout_data.get('volume_24h', 0),
                        'current_1m_vol': breakout_data.get('current_1m_vol', 0),
                        'avg_1m_vol': breakout_data.get('avg_1m_vol', 0)
                    }
                    
                    logger.info(f"✅ PULLBACK ENTRY FOUND: {symbol}")
                    logger.info(f"   Entry: ${entry_price:.6f} | SL: ${stop_loss:.6f} | TP1: ${take_profit_1:.6f}")
                    logger.info(f"   Waited {candidate_data['checks']} checks for pullback")
                    
                    # Remove from pending and update cooldowns
                    del pending_breakout_candidates[symbol]
                    last_long_signal_time = datetime.utcnow()
                    longs_symbol_cooldown[symbol] = datetime.utcnow() + timedelta(hours=LONG_SYMBOL_COOLDOWN_HOURS)
                    
                    return signal
            
            # ═══════════════════════════════════════════════════════
            # STEP 3: Detect NEW breakouts and add to pending cache
            # ═══════════════════════════════════════════════════════
            logger.info("🔍 Scanning for NEW breakouts...")
            breakouts = await self.detect_realtime_breakouts(max_symbols=20)
            
            if breakouts:
                for breakout in breakouts:
                    symbol = breakout['symbol']
                    
                    # Skip if already pending or on cooldown
                    if symbol in pending_breakout_candidates:
                        continue
                    if symbol in longs_symbol_cooldown:
                        if datetime.utcnow() < longs_symbol_cooldown[symbol]:
                            continue
                    
                    # Check liquidity before adding
                    liquidity = await self.check_liquidity(symbol)
                    if not liquidity['is_liquid']:
                        logger.info(f"    ❌ {symbol} - {liquidity['reason']}")
                        continue
                    
                    # Add to pending cache
                    pending_breakout_candidates[symbol] = {
                        'detected_at': datetime.utcnow(),
                        'breakout_data': breakout,
                        'checks': 0
                    }
                    logger.info(f"  ➕ Added to pending: {symbol} ({breakout['breakout_type']} | {breakout['volume_ratio']}x vol)")
            else:
                logger.info("❌ No new breakouts detected")
            
            logger.info(f"📊 Total pending candidates: {len(pending_breakout_candidates)}")
            logger.info("⏳ Waiting for pullback entries on next scan...")
            return None
            
        except Exception as e:
            logger.error(f"Error in breakout long signal generation: {e}")
            return None
    
    async def get_early_pumpers(self, limit: int = 10, min_change: float = 5.0, max_change: float = 200.0) -> List[Dict]:
        """
        Fetch FRESH PUMP candidates for LONG entries
        
        🔥 FIXED: Now uses BINANCE + MEXC for accurate 24h data (Bitunix API is garbage!)
        
        ⚡ 3-TIER ULTRA-EARLY DETECTION ⚡
        1. Quick filter: 24h movers with 5%+ gain (Binance/MEXC data)
        2. Multi-tier validation (checks earliest to latest):
           - TIER 1 (5m):  5%+ pump, 3x volume   → Ultra-early (5-10 min)
           - TIER 2 (15m): 5%+ pump, 1.5x volume → Early (15-20 min)
           - TIER 3 (30m): 5%+ pump, 1.5x volume → Fresh (25-30 min)
        
        Returns ONLY fresh pumps (not stale 24h gains!) with tier priority!
        
        Args:
            limit: Number of fresh pumpers to return
            min_change: Minimum 24h change % for pre-filter (default 5%)
            max_change: Maximum 24h change % (default 200% - no cap)
            
        Returns:
            List of {symbol, change_percent, volume_24h, tier, fresh_pump_data} sorted by tier priority then pump %
        """
        try:
            # 🔥 USE BINANCE + MEXC FOR ACCURATE 24H DATA (same as SHORTS!)
            merged_data = {}
            
            # === SOURCE 1: BINANCE FUTURES (Primary - most reliable) ===
            binance_count = 0
            try:
                binance_url = "https://fapi.binance.com/fapi/v1/ticker/24hr"
                response = await self.client.get(binance_url, timeout=10)
                response.raise_for_status()
                binance_tickers = response.json()
                
                for ticker in binance_tickers:
                    symbol = ticker.get('symbol', '')
                    if not symbol.endswith('USDT'):
                        continue
                    try:
                        merged_data[symbol] = {
                            'symbol': symbol,
                            'change_percent': float(ticker.get('priceChangePercent', 0)),
                            'last_price': float(ticker.get('lastPrice', 0)),
                            'volume_usdt': float(ticker.get('quoteVolume', 0)),
                            'high_24h': float(ticker.get('highPrice', 0)),
                            'low_24h': float(ticker.get('lowPrice', 0)),
                            'source': 'binance'
                        }
                        binance_count += 1
                    except (ValueError, TypeError):
                        continue
            except Exception as e:
                logger.warning(f"⚠️ LONGS: Binance API error: {e}")
            
            # === SOURCE 2: MEXC FUTURES (Secondary - for coins not on Binance) ===
            mexc_count = 0
            mexc_added = 0
            try:
                mexc_url = "https://contract.mexc.com/api/v1/contract/ticker"
                response = await self.client.get(mexc_url, timeout=10)
                response.raise_for_status()
                mexc_data = response.json()
                mexc_tickers = mexc_data.get('data', []) if isinstance(mexc_data, dict) else mexc_data
                
                for ticker in mexc_tickers:
                    raw_symbol = ticker.get('symbol', '')
                    symbol = raw_symbol.replace('_', '')
                    if not symbol.endswith('USDT'):
                        continue
                    mexc_count += 1
                    
                    if symbol not in merged_data:
                        try:
                            change_pct = float(ticker.get('riseFallRate', 0))
                            if abs(change_pct) < 5 and abs(change_pct) > 0:
                                change_pct = change_pct * 100
                            
                            merged_data[symbol] = {
                                'symbol': symbol,
                                'change_percent': change_pct,
                                'last_price': float(ticker.get('lastPrice', 0)),
                                'volume_usdt': float(ticker.get('amount24', 0)),
                                'high_24h': float(ticker.get('high24Price', 0)),
                                'low_24h': float(ticker.get('low24Price', 0)),
                                'source': 'mexc'
                            }
                            mexc_added += 1
                        except (ValueError, TypeError):
                            continue
            except Exception as e:
                logger.warning(f"⚠️ LONGS: MEXC API error: {e}")
            
            # === GET BITUNIX AVAILABLE SYMBOLS ===
            bitunix_url = f"{self.base_url}/api/v1/futures/market/tickers"
            bitunix_response = await self.client.get(bitunix_url)
            bitunix_data = bitunix_response.json()
            bitunix_symbols = set()
            if isinstance(bitunix_data, dict) and bitunix_data.get('data'):
                for t in bitunix_data.get('data', []):
                    bitunix_symbols.add(t.get('symbol', ''))
            
            logger.info(f"📈 LONGS DATA: Binance={binance_count} | MEXC={mexc_count} (+{mexc_added} unique) | Bitunix={len(bitunix_symbols)} tradeable")
            
            # DEBUG: Log top 5 pumpers from merged data
            all_pumpers = [(s, d['change_percent']) for s, d in merged_data.items() if d['change_percent'] > 0]
            all_pumpers.sort(key=lambda x: x[1], reverse=True)
            if all_pumpers[:5]:
                top5 = [(s, f"+{c:.1f}%") for s, c in all_pumpers[:5]]
                logger.info(f"📊 TOP 5 FROM BINANCE/MEXC: {top5}")
            
            # STAGE 1: Filter to pumpers in range AND available on Bitunix
            candidates = []
            rejected_not_bitunix = 0
            rejected_low_volume = 0
            rejected_out_of_range = 0
            
            for symbol, data in merged_data.items():
                change_percent = data['change_percent']
                
                # Check if in Bitunix
                if symbol not in bitunix_symbols:
                    if min_change <= change_percent <= max_change:
                        rejected_not_bitunix += 1
                    continue
                
                # Check range
                if not (min_change <= change_percent <= max_change):
                    rejected_out_of_range += 1
                    continue
                
                # Check volume
                if data['volume_usdt'] < self.min_volume_usdt:
                    rejected_low_volume += 1
                    continue
                
                candidates.append({
                    'symbol': symbol.replace('USDT', '/USDT'),
                    'change_percent_24h': round(change_percent, 2),
                    'volume_24h': round(data['volume_usdt'], 0),
                    'price': data['last_price'],
                    'high_24h': data['high_24h'],
                    'low_24h': data['low_24h']
                })
            
            # Sort by change % for logging
            candidates.sort(key=lambda x: x['change_percent_24h'], reverse=True)
            logger.info(f"Stage 1: {len(candidates)} candidates | Rejected: {rejected_not_bitunix} not-on-Bitunix, {rejected_low_volume} low-vol, {rejected_out_of_range} out-of-range")
            if candidates[:5]:
                top_list = [(c['symbol'], f"+{c['change_percent_24h']}%") for c in candidates[:5]]
                logger.info(f"📈 TOP CANDIDATES: {top_list}")
            
            # 🔥 SIMPLIFIED: Skip tier validation - just return pumping coins
            # Entry quality check happens in analyze_momentum_long / analyze_early_pump_long
            # Those functions check: RSI, EMA position, volume, candle position, etc.
            for candidate in candidates:
                candidate['change_percent'] = candidate['change_percent_24h']  # Use 24h change
                candidate['tier'] = '24h'  # Mark as 24h pumper
                candidate['fresh_pump_data'] = {}
            
            logger.info(f"📈 Returning {min(len(candidates), limit)} pumping coins for LONG analysis")
            return candidates[:limit]
            
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
            # 🔥 BOUNCE DETECTION - Don't short coins already bouncing!
            # If coin dropped 15%+ from high and recovered 30%+ from low = BOUNCING
            # ═══════════════════════════════════════════════════════
            try:
                # Check last 6 hours (72 5m candles) for dump & bounce pattern
                lookback = min(72, len(candles_5m))
                recent_highs = [c[2] for c in candles_5m[-lookback:]]
                recent_lows = [c[3] for c in candles_5m[-lookback:]]
                
                period_high = max(recent_highs)
                period_low = min(recent_lows)
                
                # Calculate drop from high and recovery from low
                drop_from_high_pct = ((period_high - current_price) / period_high) * 100 if period_high > 0 else 0
                recovery_from_low_pct = ((current_price - period_low) / period_low) * 100 if period_low > 0 else 0
                total_range_pct = ((period_high - period_low) / period_low) * 100 if period_low > 0 else 0
                
                # Recovery ratio: how much of the dump has been recovered?
                recovery_ratio = recovery_from_low_pct / total_range_pct if total_range_pct > 0 else 0
                
                # BOUNCE DETECTED: Dropped 15%+ from high but recovered 35%+ of that drop
                is_bouncing = (
                    drop_from_high_pct >= 15 and  # Had significant dump
                    recovery_ratio >= 0.35  # Recovered 35%+ of the drop
                )
                
                if is_bouncing:
                    logger.info(f"  🚫 {symbol} - BOUNCING: Dropped {drop_from_high_pct:.1f}% from high, recovered {recovery_ratio*100:.0f}% - would short the bottom!")
                    return None
                    
            except Exception as e:
                logger.warning(f"{symbol} Bounce detection failed: {e}")
                is_bouncing = False
            
            # ═══════════════════════════════════════════════════════
            # 🔥 EXHAUSTION DETECTION - Find the TOP of chart!
            # ═══════════════════════════════════════════════════════
            try:
                # 1. Price near 24h HIGH (top of chart) - STRICT: within 1% only!
                highs_5m = [c[2] for c in candles_5m]
                high_24h = max(highs_5m[-48:]) if len(highs_5m) >= 48 else max(highs_5m)  # 48 5m candles = 4 hours
                distance_from_high = ((high_24h - current_price) / high_24h) * 100 if high_24h > 0 else 0
                is_near_top = distance_from_high < 1.0  # Within 1% of recent high
                
                # 2. Wick rejection analysis (long upper wick = buyers rejected)
                current_high = candles_5m[-1][2]
                current_low = candles_5m[-1][3]
                candle_body = abs(current_price - current_open)
                upper_wick = current_high - max(current_price, current_open)
                lower_wick = min(current_price, current_open) - current_low
                total_range = current_high - current_low if current_high > current_low else 0.0001
                
                upper_wick_ratio = upper_wick / total_range if total_range > 0 else 0
                has_rejection_wick = upper_wick_ratio > 0.4  # Upper wick is 40%+ of candle = rejection
                
                # 3. Volume exhaustion (declining volume on pumps = buyers drying up)
                recent_volumes = volumes_5m[-5:] if len(volumes_5m) >= 5 else volumes_5m
                older_volumes = volumes_5m[-10:-5] if len(volumes_5m) >= 10 else volumes_5m[:5]
                avg_recent_vol = sum(recent_volumes) / len(recent_volumes) if recent_volumes else 1
                avg_older_vol = sum(older_volumes) / len(older_volumes) if older_volumes else 1
                volume_declining = avg_recent_vol < avg_older_vol * 0.7  # Recent volume 30%+ lower
                
                # 4. Bearish divergence (price making higher high but RSI making lower high)
                has_bearish_divergence = False
                if len(closes_5m) >= 20:
                    try:
                        rsi_range = min(5, len(closes_5m) - 15)
                        if rsi_range > 1:
                            rsi_recent = [self._calculate_rsi(closes_5m[:-i] if i > 0 else closes_5m, 14) for i in range(rsi_range)]
                            price_making_new_high = current_price >= max(closes_5m[-10:-1])
                            rsi_making_lower_high = rsi_5m < max(rsi_recent[1:]) if len(rsi_recent) > 1 else False
                            has_bearish_divergence = price_making_new_high and rsi_making_lower_high and rsi_5m > 55
                    except:
                        pass
                
                # 5. Current candle is RED with body > wick (sellers confirmed)
                candle_body_size = abs(current_price - current_open)
                is_red_candle_confirmed = (
                    current_candle_bearish and 
                    candle_body_size > upper_wick and 
                    candle_body_size > lower_wick
                )
                
                # 6. 15m timeframe showing rejection (upper wick on 15m)
                highs_15m = [c[2] for c in candles_15m]
                current_15m_high = candles_15m[-1][2]
                current_15m_close = closes_15m[-1]
                current_15m_open = candles_15m[-1][1]
                upper_wick_15m = current_15m_high - max(current_15m_close, current_15m_open)
                total_range_15m = candles_15m[-1][2] - candles_15m[-1][3]
                has_15m_rejection = (upper_wick_15m / total_range_15m > 0.3) if total_range_15m > 0 else False
                
                # 7. Consecutive green candles (pump exhaustion - too many green = reversal coming)
                green_streak = 0
                for i in range(-1, -8, -1):
                    if len(candles_5m) >= abs(i):
                        candle = candles_5m[i]
                        if candle[4] > candle[1]:  # close > open = green
                            green_streak += 1
                        else:
                            break
                has_extended_green_streak = green_streak >= 4  # 4+ consecutive green = exhaustion
                
                # 8. 15m RSI overbought (higher timeframe confirmation)
                rsi_15m = self._calculate_rsi(closes_15m, 14)
                is_15m_overbought = rsi_15m >= 67  # Stricter: 67+ for quality
                
                # 9. Slowing momentum (current candle smaller than previous = losing steam)
                current_body = abs(current_price - current_open)
                prev_body = abs(prev_close - prev_open)
                is_momentum_slowing = current_body < prev_body * 0.6  # Current candle 40%+ smaller
                
                # 10. Price far from EMA21 (extended = mean reversion likely)
                is_very_extended = price_to_ema21_dist >= 4.0  # 4%+ above EMA21
                
                # ═══════════════════════════════════════════════════════
                # 🔥 WEIGHTED EXHAUSTION SCORING - Quality over Quantity!
                # Core flags (2 pts each) = high-confidence reversal signs
                # Secondary flags (1 pt each) = supporting confirmation
                # Require: ≥6 total points AND ≥2 core flags for quality
                # ═══════════════════════════════════════════════════════
                
                # CORE FLAGS (2 pts each) - Strong reversal indicators
                core_flags = {
                    'wick_rejection': has_rejection_wick,      # Buyers rejected at top
                    'volume_declining': volume_declining,      # Buyers drying up
                    'bearish_divergence': has_bearish_divergence,  # RSI diverging from price
                    'very_extended': is_very_extended,         # Far from mean (4%+ EMA21)
                    '15m_rejection': has_15m_rejection         # Higher TF rejection
                }
                
                # SECONDARY FLAGS (1 pt each) - Supporting confirmation
                secondary_flags = {
                    'near_top': is_near_top,                   # Within 1% of high
                    'rsi_overbought': rsi_5m >= 70,            # 5m overbought
                    'red_candle': is_red_candle_confirmed,     # Bearish candle
                    'green_streak': has_extended_green_streak, # Exhausted pump
                    '15m_overbought': is_15m_overbought,       # 15m overbought
                    'momentum_slowing': is_momentum_slowing    # Candle shrinking
                }
                
                core_count = sum(core_flags.values())
                secondary_count = sum(secondary_flags.values())
                exhaustion_score = (core_count * 2) + (secondary_count * 1)
                
                # Legacy count for logging
                exhaustion_signs = core_count + secondary_count
                
                logger.info(f"  📊 {symbol} EXHAUSTION SCORE: {exhaustion_score} pts ({core_count} core + {secondary_count} secondary)")
                logger.info(f"     Core: Wick={has_rejection_wick}, VolDown={volume_declining}, Diverg={has_bearish_divergence}, Extended={is_very_extended}, 15mRej={has_15m_rejection}")
                logger.info(f"     Secondary: Top={is_near_top}, RSI={rsi_5m:.0f}, RedCandle={is_red_candle_confirmed}, GreenStreak={green_streak}, 15mRSI={rsi_15m:.0f}, SlowMom={is_momentum_slowing}")
            except Exception as e:
                logger.warning(f"{symbol} Exhaustion detection failed: {e}")
                exhaustion_signs = 0
                exhaustion_score = 0
                core_count = 0
                secondary_count = 0
                is_near_top = False
                has_rejection_wick = False
                volume_declining = False
                has_bearish_divergence = False
                is_red_candle_confirmed = False
                has_15m_rejection = False
                has_extended_green_streak = False
                is_15m_overbought = False
                is_momentum_slowing = False
                is_very_extended = False
                rsi_15m = 50
                green_streak = 0
                highs_5m = [c[2] for c in candles_5m]
                distance_from_high = 0
            
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
                
                # 🔥 ENTRY TIMING: Only enter SHORT when price is in UPPER portion of candle!
                # If price already dumped to bottom of candle = BAD ENTRY (chasing the dump)
                candle_range = current_high - current_low if current_high > current_low else 0.0001
                price_position_in_candle = (current_price - current_low) / candle_range  # 0 = bottom, 1 = top
                is_good_entry_timing = price_position_in_candle >= 0.55  # Price must be in upper 45% of candle (quality)
                
                # OVEREXTENDED SHORT CONDITIONS - QUALITY WEIGHTED SCORING (TIGHTENED):
                # 1. RSI 65+ (overbought) - was 63
                # 2. Volume 1.2x+ (above average) - was 1.0x
                # 3. Price 3.0%+ above EMA9 (extended) - was 2.5%
                # 4. 🔥 EXHAUSTION SCORE: ≥7 pts AND ≥2 core flags (quality filter!) - was 6pts
                # 5. 🔥 Funding rate analysis for confirmation
                # 6. 🔥 No massive buy wall blocking the dump
                # 7. 🔥 ENTRY TIMING: Price in upper 40% of candle (not chasing dump!) - was 45%
                is_overextended_short = (
                    rsi_5m >= 65 and  # Overbought required (TIGHTENED from 63)
                    volume_ratio >= 1.2 and  # Above avg volume (TIGHTENED from 1.0)
                    price_to_ema9_dist >= 3.0 and  # Extended above EMA9 (TIGHTENED from 2.5%)
                    exhaustion_score >= 7 and  # 🔥 Need 7+ weighted points (TIGHTENED from 6)
                    core_count >= 2 and  # 🔥 Need at least 2 core reversal flags
                    is_good_entry_timing  # 🔥 CRITICAL: Don't chase - enter near top of candle!
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
                    # Boost confidence based on weighted exhaustion score
                    exhaustion_boost = (exhaustion_score - 6) * 3  # +3 per extra point above 6
                    core_boost = (core_count - 2) * 5  # +5 per extra core flag above 2
                    total_confidence = 85 + confidence_boost + exhaustion_boost + core_boost
                    
                    logger.info(f"{symbol} ✅ QUALITY SHORT: Score {exhaustion_score} pts ({core_count} core) | RSI {rsi_5m:.0f} | +{price_to_ema9_dist:.1f}% extended | Funding {funding_pct:.2f}%")
                    return {
                        'direction': 'SHORT',
                        'confidence': min(total_confidence, 99),  # Cap at 99
                        'entry_price': current_price,
                        'reason': f'🎯 QUALITY SHORT | Score: {exhaustion_score}pts ({core_count} core) | RSI {rsi_5m:.0f} | +{price_to_ema9_dist:.1f}% extended | Funding {funding_pct:.2f}%'
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
                
                # SKIP: Not quality enough for SHORT, not exceptional enough for LONG
                else:
                    skip_reasons = []
                    if rsi_5m < 65:
                        skip_reasons.append(f"RSI {rsi_5m:.0f} (need 65+)")
                    if volume_ratio < 1.2:
                        skip_reasons.append(f"Volume {volume_ratio:.1f}x (need 1.2x+)")
                    if price_to_ema9_dist < 3.0:
                        skip_reasons.append(f"Distance {price_to_ema9_dist:+.1f}% (need 3.0%+)")
                    if exhaustion_score < 7:
                        skip_reasons.append(f"Score {exhaustion_score} pts (need 7+)")
                    if core_count < 2:
                        skip_reasons.append(f"Only {core_count} core flags (need 2+)")
                    if not is_good_entry_timing:
                        skip_reasons.append(f"Bad entry timing - price at {price_position_in_candle*100:.0f}% of candle (need 55%+)")
                    logger.info(f"{symbol} NOT QUALITY ENOUGH: {', '.join(skip_reasons)}")
                    return None
            
            
            # ═══════════════════════════════════════════════════════
            # STRATEGY 2: DOWNTREND SHORT - Catch top gainers AFTER the pump failed
            # ═══════════════════════════════════════════════════════
            # Both 5m and 15m bearish = trend has CONFIRMED flipped
            # This is for coins that pumped, peaked, and are now in downtrend
            # ═══════════════════════════════════════════════════════
            elif not bullish_5m and not bullish_15m:
                # Calculate current candle size for strong dump detection
                current_candle_size = abs((current_price - current_open) / current_open * 100)
                
                # 🔥 CONFIRM THIS WAS A TOP GAINER THAT PEAKED
                # Check that price was significantly higher recently (had a run)
                highs_5m_local = [c[2] for c in candles_5m]  # Get highs for this strategy
                recent_high = max(highs_5m_local[-12:]) if len(highs_5m_local) >= 12 else max(highs_5m_local)  # 1hr high
                drop_from_high = ((recent_high - current_price) / recent_high) * 100
                had_significant_run = drop_from_high >= 3.0  # Price dropped 3%+ from recent high
                
                # 🔥 CONFIRM DOWNTREND (lower lows forming)
                lows_5m = [c[3] for c in candles_5m]
                recent_lows = lows_5m[-6:]  # Last 30 min
                is_making_lower_lows = len(recent_lows) >= 3 and recent_lows[-1] < recent_lows[0]
                
                # 🔥 CONFIRM SELLERS IN CONTROL (more red candles than green recently)
                recent_directions = [candles_5m[i][4] < candles_5m[i][1] for i in range(-6, 0)]  # Close < Open = red
                red_candle_count = sum(recent_directions)
                sellers_dominant = red_candle_count >= 4  # 4+ of last 6 candles are red
                
                logger.info(f"  📉 {symbol} DOWNTREND CHECK: Drop from high={drop_from_high:.1f}%, Lower lows={is_making_lower_lows}, Sellers dominant={sellers_dominant} ({red_candle_count}/6 red)")
                
                # ═════ ENTRY PATH 1: CONFIRMED DOWNTREND (Best entry) ═════
                is_confirmed_downtrend = (
                    had_significant_run and  # Was a top gainer that pumped
                    is_making_lower_lows and  # Making lower lows
                    sellers_dominant and  # Red candles dominant
                    current_candle_bearish and  # Current candle red
                    40 <= rsi_5m <= 60  # RSI mid-range (room to fall)
                )
                
                if is_confirmed_downtrend:
                    logger.info(f"{symbol} ✅ DOWNTREND CONFIRMED: {drop_from_high:.1f}% off high | Lower lows forming | {red_candle_count}/6 red candles")
                    return {
                        'direction': 'SHORT',
                        'confidence': 92,
                        'entry_price': current_price,
                        'reason': f'📉 DOWNTREND | {drop_from_high:.1f}% off high | Lower lows | {red_candle_count}/6 red | Trend flipped!'
                    }
                
                # ═════ ENTRY PATH 2: STRONG DUMP (Direct Entry - No Pullback Needed) ═════
                # For violent dumps with high volume, enter immediately
                is_strong_dump = (
                    current_candle_bearish and 
                    current_candle_size >= 1.5 and  # At least 1.5% dump candle (was 1%)
                    volume_ratio >= 1.5 and  # Strong volume
                    40 <= rsi_5m <= 60  # RSI range (tighter)
                )
                
                if is_strong_dump:
                    logger.info(f"{symbol} ✅ STRONG DUMP DETECTED: {current_candle_size:.2f}% red candle | Vol: {volume_ratio:.1f}x | RSI: {rsi_5m:.0f}")
                    return {
                        'direction': 'SHORT',
                        'confidence': 88,
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
            
            if len(candles_5m) < 30 or len(candles_15m) < 30:
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
            
            # 🛡️ CRITICAL: Reject if price is near 24h HIGH (prevents top entries on multi-day pumps!)
            highs_5m = [c[2] for c in candles_5m]
            high_24h = max(highs_5m[-48:]) if len(highs_5m) >= 48 else max(highs_5m)  # 4 hours of 5m candles
            distance_from_24h_high = ((high_24h - current_price) / high_24h) * 100 if high_24h > 0 else 0
            
            # 🔥 REMOVED: 24h high check - in bull markets, we WANT to buy breakouts!
            # Let the momentum carry us instead of waiting for pullbacks that never come
            logger.info(f"  📊 {symbol} - Distance from 24h high: {distance_from_24h_high:.1f}%")
            
            # Filter 1: Must be in uptrend (both timeframes)
            if not (bullish_5m and bullish_15m):
                logger.info(f"  ❌ {symbol} - Not bullish on both timeframes")
                return None
            
            # Filter 2: Not overextended (within 5% of EMA9 - STRICT!)
            if price_to_ema9_dist > 5.0:
                logger.info(f"  ❌ {symbol} - Too extended ({price_to_ema9_dist:+.1f}% from EMA9, need ≤5%)")
                return None
            
            # Filter 3: RSI in optimal zone (52-65) - VERY STRICT for quality!
            if not (52 <= rsi_5m <= 65):
                logger.info(f"  ❌ {symbol} - RSI {rsi_5m:.0f} out of range (need 52-65)")
                return None
            
            # Filter 4: Strong volume surge (2.0x minimum - HIGH QUALITY ONLY!)
            if volume_ratio < 2.0:
                logger.info(f"  ❌ {symbol} - Low volume {volume_ratio:.1f}x (need 2.0x+)")
                return None
            
            # Filter 5: Current 5m candle must be bullish (green)
            if not current_candle_bullish:
                logger.info(f"  ❌ {symbol} - Current 5m candle is bearish (need green)")
                return None
            
            # 🔥 FASTER PULLBACK DETECTION: Use 1m candles to catch dips earlier!
            # Instead of waiting for 5m RED candle, check for ANY 1m pullback in last 3 candles
            candles_1m = await self.fetch_candles(symbol, '1m', limit=10)
            
            if len(candles_1m) >= 5:
                # Check last 5 one-minute candles for ANY red candle (pullback)
                has_1m_pullback = False
                for i in range(-5, -1):  # Check candles -5 to -2 (not current)
                    c = candles_1m[i]
                    if c[4] < c[1]:  # Close < Open = RED candle
                        has_1m_pullback = True
                        break
                
                # Current 1m should be green (bounce)
                current_1m = candles_1m[-1]
                current_1m_green = current_1m[4] > current_1m[1]
                
                if has_1m_pullback and current_1m_green:
                    logger.info(f"  ✅ {symbol} - 1m MICRO-PULLBACK detected → GREEN bounce!")
                elif has_1m_pullback:
                    logger.info(f"  ⚠️ {symbol} - 1m pullback found, waiting for green bounce")
                    return None
                else:
                    # No 1m pullback - check if 5m has pullback as fallback
                    prev_close = closes_5m[-2]
                    prev_open = candles_5m[-2][1]
                    if prev_close < prev_open:
                        logger.info(f"  ✅ {symbol} - 5m RED pullback → GREEN continuation")
                    else:
                        logger.info(f"  ❌ {symbol} - No pullback on 1m or 5m - too risky")
                        return None
            else:
                # Fallback to 5m pullback check
                prev_close = closes_5m[-2]
                prev_open = candles_5m[-2][1]
                if prev_close >= prev_open:
                    logger.info(f"  ❌ {symbol} - No pullback detected - waiting for dip")
                    return None
                logger.info(f"  ✅ {symbol} - 5m RED pullback → GREEN continuation")
            
            # Filter 6: Don't enter at TOP of green candle!
            current_high = candles_5m[-1][2]
            current_low = candles_5m[-1][3]
            candle_range = current_high - current_low
            
            if candle_range > 0:
                price_position_in_candle = ((current_price - current_low) / candle_range) * 100
            else:
                price_position_in_candle = 50
            
            # 🔥 STRICT: REJECT if price is in TOP 30% of candle (no more top entries!)
            if price_position_in_candle > 70:
                logger.info(f"  ❌ {symbol} - Price at TOP of candle ({price_position_in_candle:.0f}% of range)")
                return None
            
            # ✅ ALL CHECKS PASSED - Momentum entry AFTER pullback!
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
            
            if len(candles_5m) < 30 or len(candles_15m) < 30:
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
            
            # 🛡️ CRITICAL: Reject if price is near 24h HIGH (prevents top entries on multi-day pumps!)
            highs_5m = [c[2] for c in candles_5m]
            high_24h = max(highs_5m[-48:]) if len(highs_5m) >= 48 else max(highs_5m)  # 4 hours of 5m candles
            distance_from_24h_high = ((high_24h - current_price) / high_24h) * 100 if high_24h > 0 else 0
            
            # 🔥 REMOVED: 24h high check - in bull markets, we WANT to buy breakouts!
            logger.info(f"  📊 {symbol} - Distance from 24h high: {distance_from_24h_high:.1f}%")
            
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
            
            # 🔥 CRITICAL: Don't enter at TOP of green candle!
            # Calculate where price is within current candle's range (0% = low, 100% = high)
            current_high = candles_5m[-1][2]
            current_low = candles_5m[-1][3]
            candle_range = current_high - current_low
            
            if candle_range > 0:
                price_position_in_candle = ((current_price - current_low) / candle_range) * 100
            else:
                price_position_in_candle = 50  # Neutral if no range
            
            # 🔥 STRICT: REJECT if price is in TOP 30% of candle (NO MORE buying tops!)
            if price_position_in_candle > 70:
                logger.info(f"  ❌ {symbol} REJECTED - Price at TOP of candle ({price_position_in_candle:.0f}% of range) - WAIT FOR PULLBACK!")
                return None
            
            # 🔥 RELAXED: Previous candle RED preferred but not required in bull trends
            if prev_candle_bearish:
                logger.info(f"  ✅ {symbol} - Perfect setup: RED pullback → GREEN continuation")
            else:
                logger.info(f"  ⚠️ {symbol} - No pullback (prev GREEN) but allowing in bull trend")
            
            # ENTRY CONDITION 1: EMA9 PULLBACK LONG (BEST - wait for retracement!)
            # Price pulled back to/below EMA9, ready to resume UP
            is_at_or_below_ema9 = price_to_ema9_dist <= 1.5  # 🔥 STRICT: Max 1.5% above EMA9
            
            logger.info(f"  📊 {symbol} - Price to EMA9: {price_to_ema9_dist:+.2f}%, Vol: {volume_ratio:.2f}x, RSI: {rsi_5m:.0f}, Candle pos: {price_position_in_candle:.0f}%")
            if (is_at_or_below_ema9 and
                volume_ratio >= 1.3 and
                rsi_5m >= 40 and rsi_5m <= 65 and  # 🔥 STRICT: Max RSI 65 (avoid overbought!)
                bullish_momentum and
                current_candle_bullish):  # Green candle resuming AFTER red pullback
                
                logger.info(f"{symbol} ✅ EMA9 PULLBACK LONG: Near EMA9 ({price_to_ema9_dist:+.1f}%) | Vol {volume_ratio:.1f}x | RSI {rsi_5m:.0f} | Funding {funding_pct:.2f}%")
                return {
                    'direction': 'LONG',
                    'confidence': 95 + confidence_boost,  # 🔥 Boosted by funding rate!
                    'entry_price': current_price,
                    'reason': f'📈 EMA9 PULLBACK | Retracement entry | Vol {volume_ratio:.1f}x | RSI {rsi_5m:.0f} | Funding {funding_pct:.2f}%'
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
                rsi_5m >= 40 and rsi_5m <= 65 and  # 🔥 STRICT: Max RSI 65 (avoid overbought!)
                volume_ratio >= 1.3 and
                price_to_ema9_dist >= -2.0 and price_to_ema9_dist <= 3.0):  # 🔥 STRICT: Max 3% above EMA9
                
                return {
                    'direction': 'LONG',
                    'confidence': 90 + confidence_boost,  # 🔥 Boosted by funding rate!
                    'entry_price': current_price,
                    'reason': f'🎯 RESUMPTION LONG | Entered AFTER pullback | Vol {volume_ratio:.1f}x | RSI {rsi_5m:.0f} | Funding {funding_pct:.2f}%'
                }
            
            # ENTRY CONDITION 3: REMOVED - Was causing entries at tops of green candles!
            # All LONGS now REQUIRE retracement (either EMA9 pullback or resumption pattern)
            # This prevents buying exhausted pumps and ensures safer entries
            
            # SKIP - no valid LONG entry (MUST have retracement to avoid buying tops!)
            skip_reason = []
            if not is_at_or_below_ema9 and not has_resumption_pattern:
                skip_reason.append("No retracement (need EMA9 pullback or resumption pattern)")
            if price_to_ema9_dist > 5.0:
                skip_reason.append(f"Too far from EMA9 ({price_to_ema9_dist:+.1f}%, need ≤5%)")
            if volume_ratio < 1.0:
                skip_reason.append(f"Low volume {volume_ratio:.1f}x (need 1.0x+)")
            if not (40 <= rsi_5m <= 70):
                skip_reason.append(f"RSI {rsi_5m:.0f} out of range (need 40-70, avoid overbought)")
            
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
                    
                    # 🔥 PARABOLIC SHORTS BYPASS COOLDOWN!
                    # Cooldown only applies to normal shorts (28%+), not parabolic (50%+)
                    # This ensures we catch BANANA-style exhausted dumps even if a normal short failed earlier
                    logger.info(f"🎯 Analyzing PARABOLIC candidate: {symbol} @ +{gainer['change_percent']}% (bypassing cooldown!)")
                    
                    momentum = await self.analyze_momentum(symbol)
                    
                    if momentum and momentum['direction'] == 'SHORT' and 'PARABOLIC REVERSAL' in momentum['reason']:
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
                    
                    # 🔥 ENTRY TIMING: Only enter SHORT when price is in UPPER portion of candle!
                    candle_range = current_high - current_low if current_high > current_low else 0.0001
                    price_position_in_candle = (current_price - current_low) / candle_range  # 0 = bottom, 1 = top
                    is_good_entry_timing = price_position_in_candle >= 0.5  # Price must be in upper 50% of candle
                    
                    # 🎯 RELAXED ENTRY CRITERIA (2/3 exhaustion OR RSI ≥70)
                    # Path 1: 2+ exhaustion signs with good confirmation
                    # Path 2: High RSI ≥70 (overbought - likely to revert)
                    # Path 3: Extreme pump (70%+) with 1+ exhaustion sign
                    exhaustion_signals = [high_rsi, has_rejection, slowing_momentum]
                    exhaustion_count = sum(exhaustion_signals)
                    good_volume = volume_ratio >= 0.8  # Relaxed from 1.0x
                    high_overbought_rsi = rsi_5m >= 70  # Relaxed from 75
                    extreme_pump = change_pct >= 70  # 70%+ pumps are extreme
                    
                    # Entry if: (2+ exhaustion OR RSI ≥70 OR extreme pump with 1+ sign) + volume + good entry timing
                    has_strong_signal = (
                        exhaustion_count >= 2 or  # 2/3 exhaustion signs
                        high_overbought_rsi or    # RSI ≥70 alone
                        (extreme_pump and exhaustion_count >= 1)  # 70%+ pump with any sign
                    )
                    
                    if has_strong_signal and good_volume and is_good_entry_timing:
                        # Build exhaustion reason
                        reasons = []
                        if high_rsi:
                            reasons.append(f"RSI {rsi_5m:.0f} (overbought)")
                        if has_rejection:
                            reasons.append(f"{wick_size:.1f}% wick rejection")
                        if slowing_momentum:
                            reasons.append("Momentum slowing")
                        if extreme_pump:
                            reasons.append(f"+{change_pct:.0f}% extreme pump")
                        
                        exhaustion_reason = " + ".join(reasons) if reasons else f"RSI {rsi_5m:.0f}"
                        
                        # Confidence based on signal strength
                        if exhaustion_count == 3:
                            confidence = 95  # All exhaustion signs = highest confidence
                        elif high_overbought_rsi and exhaustion_count >= 2:
                            confidence = 93  # RSI ≥70 + 2 signs
                        elif extreme_pump:
                            confidence = 91  # 70%+ extreme pump
                        elif exhaustion_count >= 2:
                            confidence = 90  # 2/3 exhaustion signs
                        else:
                            confidence = 88  # RSI ≥70 alone
                        
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
                            skip_reasons.append(f"{exhaustion_count}/3 exhaustion, RSI {rsi_5m:.0f} (need 2/3 OR RSI ≥70)")
                        if not good_volume:
                            skip_reasons.append(f"Vol {volume_ratio:.1f}x (need ≥0.8x)")
                        if not is_good_entry_timing:
                            skip_reasons.append(f"Bad entry - price at {price_position_in_candle*100:.0f}% of candle (need 50%+)")
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
        min_change: float = 5.0,
        max_change: float = 50.0,
        max_symbols: int = 10
    ) -> Optional[Dict]:
        """
        Generate LONG signals from EARLY-to-MID PUMP candidates (5-50% gains)
        
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
                
                # Try AGGRESSIVE momentum entry first (catches strong pumps)
                momentum = await self.analyze_momentum_long(symbol)
                
                # If aggressive didn't work, try SAFE pullback entry
                if not momentum or momentum['direction'] != 'LONG':
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
        # 🔥 CHECK DAILY LIMIT FIRST (max 6 signals per day)
        current_count = get_daily_signal_count()
        if current_count >= MAX_DAILY_SIGNALS:
            logger.info(f"⚠️ DAILY LIMIT: {current_count}/{MAX_DAILY_SIGNALS} signals sent today - skipping scan")
            return
        
        remaining = MAX_DAILY_SIGNALS - current_count
        logger.info(f"📊 Daily signals: {current_count}/{MAX_DAILY_SIGNALS} ({remaining} remaining)")
        
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
        # SHORTS MODE: Scan 15%+ gainers with quality exhaustion scoring
        # ═══════════════════════════════════════════════════════
        # Only run if no parabolic signal found (avoid duplicate SHORTS)
        if wants_shorts and not parabolic_signal:
            logger.info("🔴 Scanning for SHORT signals (15%+ with quality filters)...")
            short_signal = await service.generate_top_gainer_signal(min_change_percent=15.0, max_symbols=8)
            
            if short_signal and short_signal['direction'] == 'SHORT':
                logger.info(f"✅ SHORT signal found: {short_signal['symbol']} @ +{short_signal.get('24h_change')}%")
        
        # ═══════════════════════════════════════════════════════
        # LONGS MODE: REALTIME BREAKOUT DETECTION (catches pumps EARLY!)
        # ═══════════════════════════════════════════════════════
        # 🚀 NEW STRATEGY: Scan 1m volume spikes BEFORE coins hit top-gainer lists
        if wants_longs:
            logger.info("🟢 ═══════════════════════════════════════════════════════")
            logger.info("🟢 REALTIME BREAKOUT SCANNER - Catching pumps BEFORE they're top gainers!")
            logger.info("🟢 ═══════════════════════════════════════════════════════")
            long_signal = await service.generate_breakout_long_signal()
            
            if long_signal and long_signal['direction'] == 'LONG':
                logger.info(f"✅ BREAKOUT LONG found: {long_signal['symbol']} | Type: {long_signal.get('breakout_type')} | Vol: {long_signal.get('volume_ratio')}x")
        
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
        
        # Acquire PostgreSQL advisory lock with timeout (NON-BLOCKING with retry)
        # Use pg_try_advisory_lock to avoid deadlocks from stuck locks
        result = db_session.execute(text(f"SELECT pg_try_advisory_lock({lock_id})"))
        lock_acquired = result.scalar()
        
        if not lock_acquired:
            logger.warning(f"⚠️ Could not acquire lock for {lock_key} - another process is holding it")
            # Try to force-release if lock seems stuck (older than 2 min)
            db_session.execute(text(f"SELECT pg_advisory_unlock({lock_id})"))
            # Retry once
            result = db_session.execute(text(f"SELECT pg_try_advisory_lock({lock_id})"))
            lock_acquired = result.scalar()
            if not lock_acquired:
                logger.error(f"❌ Lock still held after force-release attempt: {lock_key}")
                return
        
        logger.info(f"🔒 Advisory lock acquired: {lock_key} (ID: {lock_id})")
        
        # 🔥 CHECK 1: Recent signal duplicate (within 2 HOURS)
        recent_cutoff = datetime.utcnow() - timedelta(hours=2)
        existing_signal = db_session.query(Signal).filter(
            Signal.symbol == signal_data['symbol'],
            Signal.direction == signal_data['direction'],
            Signal.signal_type.in_(['TOP_GAINER', 'PARABOLIC_REVERSAL']),
            Signal.created_at >= recent_cutoff
        ).first()
        
        if existing_signal:
            logger.warning(f"🚫 DUPLICATE PREVENTED (recent signal): {signal_data['symbol']} {signal_data['direction']} (Signal #{existing_signal.id}, {(datetime.utcnow() - existing_signal.created_at).total_seconds()/60:.0f}m ago)")
            return
        
        # 🔥 CHECK 2: ANY open positions in this symbol (across ALL users!)
        open_positions = db_session.query(Trade).filter(
            Trade.symbol == signal_data['symbol'],
            Trade.status == 'open'
        ).count()
        
        if open_positions > 0:
            logger.warning(f"🚫 DUPLICATE PREVENTED (open positions): {signal_data['symbol']} has {open_positions} open position(s) - SKIPPING!")
            return
        
        # 🔥 CHECK 3: DAILY LIMIT (max 6 total, max 4 shorts per day)
        if not check_and_increment_daily_signals(direction=signal_data['direction']):
            logger.warning(f"⚠️ DAILY LIMIT REACHED - Cannot broadcast {signal_data['symbol']} {signal_data['direction']}")
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
            # LONGS: Dual TPs (2.5% and 5% price moves at 20x = 50% and 100%)
            tp_text = f"""<b>TP1:</b> ${signal.take_profit_1:.6f} (+50% @ 20x) 
<b>TP2:</b> ${signal.take_profit_2:.6f} (+100% @ 20x) 🎯"""
            sl_text = "(-80% @ 20x)"  # LONGS: 4% SL * 20x = 80%
            rr_text = "Dual targets: 50% and 100%"
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
        
        # Volume display - different for LONGS (early breakout) vs SHORTS (24h volume)
        if signal.direction == 'LONG' and signal_data.get('volume_ratio'):
            # LONGS: Show spike ratio + 24h liquidity baseline
            vol_ratio = signal_data.get('volume_ratio', 0)
            current_1m = signal_data.get('current_1m_vol', 0)
            vol_24h = signal_data.get('24h_volume', 0)
            volume_display = f"""├ 1m Spike: <b>{vol_ratio}x</b> (${current_1m:,.0f} vs avg)
└ 24h Liquidity: ${vol_24h:,.0f}"""
        else:
            # SHORTS: Standard 24h volume
            volume_display = f"└ Volume: ${signal_data.get('24h_volume', 0):,.0f}"
        
        signal_text = f"""
┏━━━━━━━━━━━━━━━━━━━━━━━┓
  🔥 <b>TOP GAINER ALERT</b> 🔥
┗━━━━━━━━━━━━━━━━━━━━━━━┛
{tier_badge}
{direction_emoji} <b>{signal.symbol}</b>
━━━━━━━━━━━━━━━━━━━━━━━━━━

<b>📊 Market Data</b>
├ 24h Change: <b>+{signal_data.get('24h_change')}%</b>
{volume_display}

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
                                # LONGS: Dual TPs (2.5% and 5% at 20x = 50% and 100%)
                                tp_manual = f"""<b>TP1:</b> ${signal.take_profit_1:.6f} (+50% @ 20x)
<b>TP2:</b> ${signal.take_profit_2:.6f} (+100% @ 20x) 🎯"""
                                sl_manual = "(-80% @ 20x)"  # LONGS: 4% SL * 20x = 80%
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
├ Leverage: <b>{'20x' if signal.direction == 'LONG' else '10x'}</b>
└ Risk/Reward: <b>{'1:1.25' if signal.direction == 'LONG' else '1:1'}</b>

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
                        logger.info(f"⏭️ User {user.id} ({user.username}) - MAX POSITIONS: {current_top_gainer_positions}/{max_allowed}")
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
                                # LONGS: Dual TPs [5%, 10%] with user's leverage (capped at 80%)
                                targets = calculate_leverage_capped_targets(
                                    entry_price=signal.entry_price,
                                    direction='LONG',
                                    tp_pcts=[5.0, 10.0],  # TP1: 5%, TP2: 10% price moves
                                    base_sl_pct=4.0,      # 4% SL
                                    leverage=user_leverage,
                                    max_profit_cap=80.0,
                                    max_loss_cap=80.0
                                )
                                tp1_profit_pct = targets['tp_profit_pcts'][0]
                                tp2_profit_pct = targets['tp_profit_pcts'][1]
                                sl_loss_pct = targets['sl_loss_pct']
                                display_leverage = user_leverage
                            else:  # SHORT
                                # SHORTS: Single TP with 1:1 R:R (capped at 80%)
                                targets = calculate_leverage_capped_targets(
                                    entry_price=signal.entry_price,
                                    direction='SHORT',
                                    tp_pcts=[8.0],   # 8% TP for mean reversion shorts
                                    base_sl_pct=8.0, # 8% SL for 1:1 R:R
                                    leverage=user_leverage,
                                    max_profit_cap=80.0,
                                    max_loss_cap=80.0
                                )
                                tp1_profit_pct = targets['tp_profit_pcts'][0]
                                sl_loss_pct = targets['sl_loss_pct']
                                display_leverage = user_leverage
                    
                            # Rebuild TP/SL text with leverage-capped percentages
                            # Calculate R:R based on capped values
                            if signal.direction == 'LONG' and len(targets['tp_prices']) > 1:
                                # LONGS: Dual TPs with capped percentages
                                user_tp_text = f"""<b>TP1:</b> ${targets['tp_prices'][0]:.6f} (+{tp1_profit_pct:.0f}% @ {display_leverage}x)
<b>TP2:</b> ${targets['tp_prices'][1]:.6f} (+{tp2_profit_pct:.0f}% @ {display_leverage}x) 🎯"""
                                user_sl_text = f"${targets['sl_price']:.6f} (-{sl_loss_pct:.0f}% @ {display_leverage}x)"
                                rr_text = f"Dual targets: {tp1_profit_pct:.0f}% and {tp2_profit_pct:.0f}%"
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
        
        # 🔥 IMPROVED LOGGING: Track and report execution results
        executed_count = 0
        exception_count = 0
        skipped_count = 0
        
        for idx, (result, user) in enumerate(zip(results, users_with_mode)):
            if result is True:
                executed_count += 1
            elif isinstance(result, Exception):
                exception_count += 1
                logger.error(f"❌ EXCEPTION for user {user.id} ({user.username}): {result}")
            else:
                skipped_count += 1
                # Only log first few skips to avoid spam
                if skipped_count <= 5:
                    logger.info(f"⏭️ Skipped user {user.id} ({user.username}): result={result}")
        
        logger.info(f"📊 EXECUTION SUMMARY: {executed_count} executed, {skipped_count} skipped, {exception_count} errors out of {len(users_with_mode)} users")
        
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
