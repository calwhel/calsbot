"""
Top Gainers Trading Mode - Generates signals from Bitunix top movers
Focuses on momentum plays with 5x leverage and 15% TP/SL
Includes 48h watchlist to catch delayed reversals
"""
import logging
import httpx
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)


class TopGainersSignalService:
    """Service to fetch and analyze top gainers from Bitunix using direct API"""
    
    def __init__(self):
        self.base_url = "https://fapi.bitunix.com"  # For tickers and trading
        self.binance_url = "https://fapi.binance.com"  # For candle data (Binance Futures public API)
        self.client = httpx.AsyncClient(timeout=30.0)
        self.min_volume_usdt = 1000000  # $1M minimum 24h volume for liquidity
        
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
    
    async def get_top_losers(self, limit: int = 10, min_change_percent: float = -5.0) -> List[Dict]:
        """
        Fetch top losers from Bitunix based on 24h price change using direct API
        Used for potential SHORT opportunities on mean reversion
        
        Args:
            limit: Number of top losers to return
            min_change_percent: Minimum negative change % to qualify (e.g., -5.0)
            
        Returns:
            List of {symbol, change_percent, volume, price} sorted by change % ascending
        """
        try:
            # Fetch 24h ticker statistics from Bitunix public API
            # Correct endpoint: /api/v1/futures/market/tickers (returns all tickers if no symbols param)
            url = f"{self.base_url}/api/v1/futures/market/tickers"
            response = await self.client.get(url)
            response.raise_for_status()
            tickers_data = response.json()
            
            # Handle different possible response formats
            if isinstance(tickers_data, list):
                tickers = tickers_data
            elif isinstance(tickers_data, dict):
                tickers = tickers_data.get('data') or tickers_data.get('result') or tickers_data.get('tickers', [])
            else:
                logger.error(f"Unexpected ticker response type in get_top_losers: {type(tickers_data)}")
                return []
            
            if not tickers:
                logger.warning("No tickers returned from Bitunix API in get_top_losers")
                return []
            
            losers = []
            for ticker in tickers:
                symbol = ticker.get('symbol', '')
                
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
                
                # Filter for losers
                if (change_percent <= min_change_percent and 
                    volume_usdt >= self.min_volume_usdt):
                    
                    losers.append({
                        'symbol': symbol.replace('USDT', '/USDT'),
                        'change_percent': round(change_percent, 2),
                        'volume_24h': round(volume_usdt, 0),
                        'price': last_price,
                        'high_24h': float(ticker.get('high', 0)),
                        'low_24h': float(ticker.get('low', 0))
                    })
            
            # Sort by change % ascending (most negative first)
            losers.sort(key=lambda x: x['change_percent'])
            
            return losers[:limit]
            
        except Exception as e:
            logger.error(f"Error fetching top losers: {e}")
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
            # Fetch candles with sufficient history for accurate analysis
            candles_5m = await self.fetch_candles(symbol, '5m', limit=50)
            candles_15m = await self.fetch_candles(symbol, '15m', limit=50)
            
            if len(candles_5m) < 30 or len(candles_15m) < 30:
                return None
            
            # Extract price and volume data
            closes_5m = [c[4] for c in candles_5m]
            volumes_5m = [c[5] for c in candles_5m]
            closes_15m = [c[4] for c in candles_15m]
            
            # Current price and previous prices for momentum
            current_price = closes_5m[-1]
            prev_close = closes_5m[-2]
            high_5m = candles_5m[-1][2]
            low_5m = candles_5m[-1][3]
            
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
            
            # Is price near EMA9? (pullback entry zone)
            is_near_ema9 = abs(price_to_ema9_dist) < 1.5  # Within 1.5% of EMA9
            is_near_ema21 = abs(price_to_ema21_dist) < 2.0  # Within 2% of EMA21
            
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
            # STRATEGY 1: LONG - ONLY FOR EXCEPTIONAL CIRCUMSTANCES
            # ═══════════════════════════════════════════════════════
            # NOTE: LONGs are RARE - only triggered with extreme volume/momentum
            # Top gainers are primarily for SHORTING (mean reversion strategy)
            # ═══════════════════════════════════════════════════════
            if bullish_5m and bullish_15m:
                # EXCEPTIONAL ONLY: Massive volume breakout (3x+) with perfect setup
                # This is "out of the ordinary" - institutional-level buying pressure
                if volume_ratio >= 3.0 and rsi_5m > 50 and rsi_5m < 65 and not is_overextended_up and is_near_ema9:
                    return {
                        'direction': 'LONG',
                        'confidence': 95,
                        'entry_price': current_price,
                        'reason': f'🚀 EXCEPTIONAL VOLUME {volume_ratio:.1f}x | RSI: {rsi_5m:.0f} | Perfect EMA9 pullback - RARE LONG!'
                    }
                
                # SKIP: Normal bullish conditions are NOT enough for LONGs on top gainers
                # We want SHORTs (mean reversion), not continuation longs
                else:
                    logger.info(f"{symbol} LONG trend but SKIPPED (not exceptional): Vol {volume_ratio:.1f}x (need 3.0x+), RSI {rsi_5m:.0f}, Distance {price_to_ema9_dist:+.1f}%")
                    return None
            
            
            # ═══════════════════════════════════════════════════════
            # STRATEGY 2: SHORT - Mean reversion on failed pumps
            # ═══════════════════════════════════════════════════════
            elif not bullish_5m and not bullish_15m:
                # BEST ENTRY: Overextended pump rejection (mean reversion for top gainers!)
                # This catches coins like GIGGLE +150% when they start to dump
                if is_overextended_up and volume_ratio >= 1.4 and rsi_5m > 60:
                    overextension_pct = price_to_ema9_dist
                    return {
                        'direction': 'SHORT',
                        'confidence': 90,
                        'entry_price': current_price,
                        'reason': f'🔻 OVEREXTENDED PUMP {overextension_pct:+.1f}% above EMA | Vol: {volume_ratio:.1f}x | RSI: {rsi_5m:.0f} | Mean reversion SHORT'
                    }
                
                # GOOD ENTRY: Pullback in downtrend with volume
                elif is_near_ema9 and volume_ratio >= 1.3 and rsi_5m < 60 and rsi_5m > 30:
                    return {
                        'direction': 'SHORT',
                        'confidence': 85,
                        'entry_price': current_price,
                        'reason': f'📉 PULLBACK SHORT @ EMA9 | Vol: {volume_ratio:.1f}x | RSI: {rsi_5m:.0f} | Bearish 5m+15m'
                    }
                
                # ACCEPTABLE ENTRY: Strong volume dump continuation (catching the cascade)
                elif volume_ratio >= 1.8 and rsi_5m < 55 and bearish_momentum:
                    return {
                        'direction': 'SHORT',
                        'confidence': 80,
                        'entry_price': current_price,
                        'reason': f'⚡ VOLUME DUMP {volume_ratio:.1f}x | RSI: {rsi_5m:.0f} | Bearish momentum - Dump continuation'
                    }
                
                # SKIP: Bearish but no ideal entry
                else:
                    logger.info(f"{symbol} SHORT trend but NO ENTRY: Vol {volume_ratio:.1f}x, RSI {rsi_5m:.0f}, Distance {price_to_ema9_dist:+.1f}%")
                    return None
            
            
            # ═══════════════════════════════════════════════════════
            # SPECIAL CASE: Mean Reversion SHORT on Parabolic Pumps
            # Even if 5m is still bullish, if price is EXTREMELY overextended
            # and showing reversal signs, take the SHORT (top gainer specialty!)
            # ═══════════════════════════════════════════════════════
            elif bullish_5m and not bullish_15m:
                # 15m already bearish but 5m lagging - this is the reversal point
                # Perfect for catching top gainer dumps (like PIPPIN +120% starting to roll over)
                price_extension = price_to_ema9_dist
                
                # OPTIMIZED THRESHOLDS: Catch reversals EARLIER (3% vs 4%, RSI 60 vs 65)
                if price_extension > 3.0 and volume_ratio >= 1.5 and rsi_5m > 60:
                    # Overextended (>3% above EMA9) with weakening - catch BEFORE full reversal
                    return {
                        'direction': 'SHORT',
                        'confidence': 85,
                        'entry_price': current_price,
                        'reason': f'🎯 PARABOLIC REVERSAL SHORT | {price_extension:+.1f}% overextended | Vol: {volume_ratio:.1f}x | RSI: {rsi_5m:.0f} | 15m bearish divergence'
                    }
                else:
                    logger.info(f"{symbol} MIXED (5m bull, 15m bear) but not extreme enough for reversal SHORT: Distance {price_extension:+.1f}%, RSI {rsi_5m:.0f}")
                    return None
            
            
            # ═══════════════════════════════════════════════════════
            # MIXED SIGNALS - Skip choppy/uncertain conditions
            # ═══════════════════════════════════════════════════════
            else:
                logger.info(f"{symbol} MIXED SIGNALS - skipping (5m bullish: {bullish_5m}, 15m bullish: {bullish_15m})")
                return None
                
        except Exception as e:
            logger.error(f"Error analyzing momentum for {symbol}: {e}")
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
            
            # PRIORITY 1: Look for parabolic reversal shorts (biggest pumps first)
            # These are the BEST opportunities - coins that pumped 50%+ and are rolling over
            for gainer in gainers:
                if gainer['change_percent'] >= 50.0:  # Extreme pumps (50%+)
                    symbol = gainer['symbol']
                    logger.info(f"🎯 Analyzing PARABOLIC candidate: {symbol} @ +{gainer['change_percent']}%")
                    
                    momentum = await self.analyze_momentum(symbol)
                    
                    if momentum and momentum['direction'] == 'SHORT' and 'PARABOLIC REVERSAL' in momentum['reason']:
                        logger.info(f"✅ PARABOLIC REVERSAL SHORT found: {symbol}")
                        # Build signal and return immediately (highest priority!)
                        entry_price = momentum['entry_price']
                        
                        # Dual TPs for parabolic reversals
                        stop_loss = entry_price * (1 + 4.0 / 100)  # 20% loss
                        take_profit_1 = entry_price * (1 - 4.0 / 100)  # TP1: 20% profit
                        take_profit_2 = entry_price * (1 - 7.0 / 100)  # TP2: 35% profit
                        
                        return {
                            'symbol': symbol,
                            'direction': 'SHORT',
                            'entry_price': entry_price,
                            'stop_loss': stop_loss,
                            'take_profit': take_profit_1,
                            'take_profit_1': take_profit_1,
                            'take_profit_2': take_profit_2,
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
                
                # Calculate TP/SL
                # Standard: 20% TP/SL (4% price move @ 5x)
                # Parabolic Reversal SHORT: Dual TPs at 20% + 35% (coins crash hard!)
                
                if momentum['direction'] == 'LONG':
                    # LONG: Single TP at 20%
                    stop_loss = entry_price * (1 - 4.0 / 100)
                    take_profit_1 = entry_price * (1 + 4.0 / 100)  # 20% profit
                    take_profit_2 = None
                    
                else:  # SHORT
                    if is_parabolic_reversal:
                        # PARABOLIC REVERSAL: Dual TPs for big dumps
                        stop_loss = entry_price * (1 + 4.0 / 100)  # 20% loss
                        take_profit_1 = entry_price * (1 - 4.0 / 100)  # TP1: 20% profit (4% dump)
                        take_profit_2 = entry_price * (1 - 7.0 / 100)  # TP2: 35% profit (7% dump)
                    else:
                        # Regular SHORT: Single TP at 20%
                        stop_loss = entry_price * (1 + 4.0 / 100)
                        take_profit_1 = entry_price * (1 - 4.0 / 100)  # 20% profit
                        take_profit_2 = None
                
                return {
                    'symbol': symbol,
                    'direction': momentum['direction'],
                    'entry_price': entry_price,
                    'stop_loss': stop_loss,
                    'take_profit': take_profit_1,  # Backward compatible
                    'take_profit_1': take_profit_1,
                    'take_profit_2': take_profit_2,
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
    Scan for top gainers and broadcast signals to users with top_gainers_mode_enabled
    Called periodically by scheduler
    """
    from app.models import User, UserPreference, Signal, Trade
    from app.services.bitunix_trader import execute_bitunix_trade
    from datetime import datetime
    import logging
    
    logger = logging.getLogger(__name__)
    
    try:
        service = TopGainersSignalService()
        await service.initialize()
        
        # Get all users with top gainers mode enabled
        users_with_mode = db_session.query(User).join(UserPreference).filter(
            UserPreference.top_gainers_mode_enabled == True,
            UserPreference.auto_trading_enabled == True
        ).all()
        
        if not users_with_mode:
            logger.info("No users with top gainers mode enabled")
            await service.close()
            return
        
        logger.info(f"Scanning top gainers for {len(users_with_mode)} users")
        
        # Step 1: Cleanup old watchlist entries (>48 hours)
        await service.cleanup_old_watchlist(db_session)
        
        # Step 2: Fetch current top gainers and add to watchlist
        first_prefs = users_with_mode[0].preferences
        min_change = first_prefs.top_gainers_min_change if first_prefs else 5.0
        max_symbols = first_prefs.top_gainers_max_symbols if first_prefs else 3
        
        top_gainers = await service.get_top_gainers(limit=50, min_change_percent=min_change)
        
        # Add current top gainers to watchlist for 48h monitoring
        for gainer in top_gainers:
            await service.add_to_watchlist(
                db_session,
                gainer['symbol'],
                gainer['price'],
                gainer['change_percent']
            )
        
        # Step 3: Get watchlist symbols (includes both new and previous days' pumps)
        watchlist = await service.get_watchlist_symbols(db_session)
        logger.info(f"Monitoring {len(watchlist)} symbols (current + watchlist)")
        
        # Step 4: Check ALL symbols (current gainers + watchlist) for reversal signals
        combined_symbols = top_gainers + [
            {'symbol': w['symbol'], 'change_percent': w['peak_change_percent']}
            for w in watchlist
        ]
        
        # Remove duplicates (keep latest data)
        seen_symbols = set()
        unique_symbols = []
        for item in combined_symbols:
            if item['symbol'] not in seen_symbols:
                seen_symbols.add(item['symbol'])
                unique_symbols.append(item)
        
        logger.info(f"Analyzing {len(unique_symbols)} unique symbols for reversal signals...")
        
        # Step 5: Check each symbol (current + watchlist) for reversal signals
        signal_data = None
        for item in unique_symbols:
            symbol = item['symbol']
            change_percent = item['change_percent']
            
            logger.info(f"Checking {symbol} (+{change_percent:.1f}%) for reversal...")
            
            # Analyze momentum for this specific symbol
            momentum = await service.analyze_momentum(symbol)
            
            if momentum and momentum['direction'] in ['SHORT', 'LONG']:
                # Found a signal! Build it
                entry_price = momentum['entry_price']
                is_parabolic = change_percent >= 50.0 and 'PARABOLIC' in momentum.get('reason', '')
                
                # Calculate SL/TP
                if momentum['direction'] == 'SHORT':
                    stop_loss = entry_price * (1 + 4.0 / 100)  # 20% loss @ 5x
                    take_profit_1 = entry_price * (1 - 4.0 / 100)  # TP1: 20% profit
                    take_profit_2 = entry_price * (1 - 7.0 / 100) if is_parabolic else None  # TP2: 35% for parabolic
                else:  # LONG
                    stop_loss = entry_price * (1 - 4.0 / 100)
                    take_profit_1 = entry_price * (1 + 4.0 / 100)
                    take_profit_2 = None
                
                # Get current ticker data for volume
                gainers_data = await service.get_top_gainers(limit=50, min_change_percent=0)
                volume_24h = next((g['volume_24h'] for g in gainers_data if g['symbol'] == symbol), 0)
                
                signal_data = {
                    'symbol': symbol,
                    'direction': momentum['direction'],
                    'entry_price': entry_price,
                    'stop_loss': stop_loss,
                    'take_profit': take_profit_1,
                    'take_profit_1': take_profit_1,
                    'take_profit_2': take_profit_2,
                    'confidence': momentum['confidence'],
                    'reasoning': f"Top Gainer: {change_percent}% in 24h | {momentum['reason']}",
                    'trade_type': 'TOP_GAINER',
                    'leverage': 5,
                    '24h_change': change_percent,
                    '24h_volume': volume_24h,
                    'is_parabolic_reversal': is_parabolic
                }
                logger.info(f"✅ Signal found from {symbol} ({change_percent}%): {momentum['direction']}")
                break  # Take first signal found
        
        if not signal_data:
            logger.info("No top gainer signals found")
            await service.close()
            return
        
        # Create signal record
        signal = Signal(
            symbol=signal_data['symbol'],
            direction=signal_data['direction'],
            entry_price=signal_data['entry_price'],
            stop_loss=signal_data['stop_loss'],
            take_profit=signal_data.get('take_profit'),
            take_profit_1=signal_data.get('take_profit_1'),
            take_profit_2=signal_data.get('take_profit_2'),
            confidence=signal_data['confidence'],
            reasoning=signal_data['reasoning'],
            signal_type='TOP_GAINER',
            timeframe='5m',
            created_at=datetime.utcnow()
        )
        db_session.add(signal)
        db_session.commit()
        db_session.refresh(signal)
        
        # Mark watchlist entry as "signal sent" to avoid duplicate signals
        await service.mark_watchlist_signal_sent(db_session, signal.symbol)
        
        logger.info(f"🚀 TOP GAINER SIGNAL: {signal.symbol} {signal.direction} @ ${signal.entry_price} (24h: {signal_data.get('24h_change')}%)")
        
        # Check if parabolic reversal (dual TPs)
        is_parabolic = signal_data.get('is_parabolic_reversal', False)
        
        # Build TP text
        if is_parabolic and signal.take_profit_2:
            tp_text = f"""<b>TP1:</b> ${signal.take_profit_1:.6f} (+20% @ 5x)
<b>TP2:</b> ${signal.take_profit_2:.6f} (+35% @ 5x) 🎯"""
            rr_text = "1:1 + Extended Target"
        else:
            tp_text = f"<b>TP:</b> ${signal.take_profit:.6f} (+20% @ 5x)"
            rr_text = "1:1"
        
        # Broadcast to users
        signal_text = f"""
🔥 <b>TOP GAINER ALERT</b> 🔥

<b>{signal.symbol}</b> {signal.direction}
━━━━━━━━━━━━━━━━━━━━━━━━━━

📊 <b>24h Change:</b> +{signal_data.get('24h_change')}%
💰 <b>24h Volume:</b> ${signal_data.get('24h_volume'):,.0f}

<b>Entry:</b> ${signal.entry_price:.6f}
{tp_text}
<b>SL:</b> ${signal.stop_loss:.6f} (-20% @ 5x)

⚡ <b>Leverage:</b> 5x (Fixed for volatility)
🎯 <b>Risk/Reward:</b> {rr_text}

<b>Reasoning:</b>
{signal.reasoning}

⚠️ <b>HIGH VOLATILITY - TOP GAINER MODE</b>
<i>Auto-executing for users with mode enabled...</i>
"""
        
        # Execute trades for users with top gainers mode + auto-trading
        executed_count = 0
        for user in users_with_mode:
            prefs = user.preferences
            
            # Check if user has space for more top gainer positions
            current_top_gainer_positions = db_session.query(Trade).filter(
                Trade.user_id == user.id,
                Trade.status == 'open',
                Trade.trade_type == 'TOP_GAINER'
            ).count()
            
            max_allowed = prefs.top_gainers_max_symbols if prefs else 3
            
            if current_top_gainer_positions >= max_allowed:
                logger.info(f"User {user.id} already has {current_top_gainer_positions} top gainer positions (max: {max_allowed})")
                continue
            
            # Execute trade with 5x leverage override and TOP_GAINER trade_type
            trade = await execute_bitunix_trade(
                signal=signal,
                user=user,
                db=db_session,
                trade_type='TOP_GAINER',
                leverage_override=5  # Force 5x leverage for top gainers
            )
            
            if trade:
                executed_count += 1
                
                # Send notification
                try:
                    await bot.send_message(
                        user.telegram_id,
                        f"{signal_text}\n\n✅ <b>Trade Executed!</b>\n"
                        f"Position Size: ${trade.position_size:.2f}\n"
                        f"Leverage: 5x (Top Gainer Mode)",
                        parse_mode="HTML"
                    )
                except Exception as e:
                    logger.error(f"Failed to send notification to user {user.id}: {e}")
        
        logger.info(f"Top gainer signal executed for {executed_count}/{len(users_with_mode)} users")
        
        await service.close()
        
    except Exception as e:
        logger.error(f"Error in broadcast_top_gainer_signal: {e}", exc_info=True)
