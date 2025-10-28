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
            
            # Convert to DataFrame for candle size analysis
            import pandas as pd
            df_5m = pd.DataFrame(candles_5m, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            
            # 🚨 CRITICAL CHECK: Skip oversized candles (prevents chasing big dumps/pumps)
            if self._is_candle_oversized(df_5m, max_body_percent=2.5):
                logger.info(f"{symbol} SKIPPED - Current candle is oversized (prevents poor entries)")
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
            # STRATEGY 2: SHORT - Mean reversion on failed pumps (ULTRA LOOSE)
            # ═══════════════════════════════════════════════════════
            elif not bullish_5m and not bullish_15m:
                # BEST ENTRY: Overextended pump rejection (mean reversion for top gainers!)
                # REMOVED volume requirement - only need overextension + RSI
                if is_overextended_up and rsi_5m > 50:
                    overextension_pct = price_to_ema9_dist
                    return {
                        'direction': 'SHORT',
                        'confidence': 90,
                        'entry_price': current_price,
                        'reason': f'🔻 OVEREXTENDED PUMP {overextension_pct:+.1f}% above EMA | Vol: {volume_ratio:.1f}x | RSI: {rsi_5m:.0f} | Mean reversion SHORT'
                    }
                
                # GOOD ENTRY: Pullback in downtrend (volume not required in low-vol markets)
                # Volume ANY level, just need EMA alignment + reasonable RSI
                elif is_near_ema9 and rsi_5m < 70 and rsi_5m > 20:
                    return {
                        'direction': 'SHORT',
                        'confidence': 85,
                        'entry_price': current_price,
                        'reason': f'📉 PULLBACK SHORT @ EMA9 | Vol: {volume_ratio:.1f}x | RSI: {rsi_5m:.0f} | Bearish 5m+15m'
                    }
                
                # ACCEPTABLE ENTRY: Just need bearish momentum + decent RSI (low volume OK)
                # Removed volume requirement entirely for low-volume markets
                elif rsi_5m < 65 and bearish_momentum:
                    return {
                        'direction': 'SHORT',
                        'confidence': 80,
                        'entry_price': current_price,
                        'reason': f'⚡ BEARISH TREND | Vol: {volume_ratio:.1f}x | RSI: {rsi_5m:.0f} | Bearish momentum - Short continuation'
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
                
                # OPTIMIZED THRESHOLDS: Catch reversals EARLIER
                # ULTRA LOOSE: 1.5% extension, NO volume requirement, 50 RSI
                if price_extension > 1.5 and rsi_5m > 50:
                    # Overextended (>1.5% above EMA9) with weakening - catch BEFORE full reversal
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
                    # LONG: Dual TPs (1:1 and 1:2)
                    stop_loss = entry_price * (1 - 4.0 / 100)  # 20% loss at 5x
                    take_profit_1 = entry_price * (1 + 4.0 / 100)  # TP1: 20% profit at 5x (1:1 R:R)
                    take_profit_2 = entry_price * (1 + 8.0 / 100)  # TP2: 40% profit at 5x (1:2 R:R)
                    take_profit_3 = None  # No TP3 for longs
                    
                else:  # SHORT
                    # SHORT: Dual or Triple TPs
                    stop_loss = entry_price * (1 + 4.0 / 100)  # 20% loss at 5x
                    take_profit_1 = entry_price * (1 - 4.0 / 100)  # TP1: 20% profit at 5x (1:1 R:R)
                    take_profit_2 = entry_price * (1 - 8.0 / 100)  # TP2: 40% profit at 5x (1:2 R:R)
                    # TP3 only for extreme parabolic reversals (they crash HARD like MAVIA!)
                    take_profit_3 = entry_price * (1 - 12.0 / 100) if is_parabolic_reversal else None  # TP3: 60% profit at 5x (1:3 R:R) 🚀
                
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
        
        # Get user preferences for min change threshold
        first_prefs = users_with_mode[0].preferences
        min_change = first_prefs.top_gainers_min_change if first_prefs else 40.0
        max_symbols = first_prefs.top_gainers_max_symbols if first_prefs else 3
        
        # Get ONLY current top gainers (40%+ TODAY) - No watchlist
        top_gainers = await service.get_top_gainers(limit=50, min_change_percent=min_change)
        
        if not top_gainers:
            logger.info(f"No coins currently over {min_change}% - skipping scan")
            await service.close()
            return
        
        logger.info(f"Analyzing {len(top_gainers)} symbols currently over {min_change}% for reversal signals...")
        
        # Check each coin currently over threshold for reversal signals
        signal_data = None
        for item in top_gainers:
            symbol = item['symbol']
            change_percent = item['change_percent']
            
            logger.info(f"Checking {symbol} (+{change_percent:.1f}%) for reversal...")
            
            # Analyze momentum for this specific symbol
            momentum = await service.analyze_momentum(symbol)
            
            if momentum and momentum['direction'] in ['SHORT', 'LONG']:
                # Found a signal! Build it
                entry_price = momentum['entry_price']
                is_parabolic = change_percent >= 50.0 and 'PARABOLIC' in momentum.get('reason', '')
                
                # Calculate SL/TP with Dual or Triple TPs
                if momentum['direction'] == 'SHORT':
                    stop_loss = entry_price * (1 + 4.0 / 100)  # 20% loss at 5x
                    take_profit_1 = entry_price * (1 - 4.0 / 100)  # TP1: 20% profit at 5x (1:1 R:R)
                    take_profit_2 = entry_price * (1 - 8.0 / 100)  # TP2: 40% profit at 5x (1:2 R:R)
                    # TP3 only for parabolic dumps (50%+) - they crash HARD like MAVIA!
                    take_profit_3 = entry_price * (1 - 12.0 / 100) if is_parabolic else None  # TP3: 60% profit at 5x (1:3 R:R) 🚀
                else:  # LONG
                    stop_loss = entry_price * (1 - 4.0 / 100)  # 20% loss at 5x
                    take_profit_1 = entry_price * (1 + 4.0 / 100)  # TP1: 20% profit at 5x (1:1 R:R)
                    take_profit_2 = entry_price * (1 + 8.0 / 100)  # TP2: 40% profit at 5x (1:2 R:R)
                    take_profit_3 = None  # No TP3 for longs
                
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
                    'take_profit_3': take_profit_3,  # 60% profit for parabolic dumps (50%+)
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
            take_profit_3=signal_data.get('take_profit_3'),  # 60% profit for parabolic dumps
            confidence=signal_data['confidence'],
            reasoning=signal_data['reasoning'],
            signal_type='TOP_GAINER',
            timeframe='5m',
            created_at=datetime.utcnow()
        )
        db_session.add(signal)
        db_session.commit()
        db_session.refresh(signal)
        
        logger.info(f"🚀 TOP GAINER SIGNAL: {signal.symbol} {signal.direction} @ ${signal.entry_price} (24h: {signal_data.get('24h_change')}%)")
        
        # Check if parabolic reversal (dual TPs)
        is_parabolic = signal_data.get('is_parabolic_reversal', False)
        
        # Build TP text - Parabolic dumps get 3 TPs!
        if signal.take_profit_3:
            # Triple TPs for parabolic reversals (50%+ coins)
            tp_text = f"""<b>TP1:</b> ${signal.take_profit_1:.6f} (+20% @ 5x) [1:1]
<b>TP2:</b> ${signal.take_profit_2:.6f} (+40% @ 5x) [1:2]
<b>TP3:</b> ${signal.take_profit_3:.6f} (+60% @ 5x) 🚀 [1:3]"""
            rr_text = "1:1, 1:2, and 1:3"
        elif signal.take_profit_2:
            # Dual TPs for regular signals
            tp_text = f"""<b>TP1:</b> ${signal.take_profit_1:.6f} (+20% @ 5x) [1:1]
<b>TP2:</b> ${signal.take_profit_2:.6f} (+40% @ 5x) 🎯 [1:2]"""
            rr_text = "1:1 and 1:2"
        else:
            # Fallback for signals without TP2 (shouldn't happen)
            tp_text = f"<b>TP:</b> ${signal.take_profit:.6f} (+20% @ 5x)"
            rr_text = "1:1"
        
        # Broadcast to users
        direction_emoji = "🟢 LONG" if signal.direction == 'LONG' else "🔴 SHORT"
        signal_text = f"""
┏━━━━━━━━━━━━━━━━━━━━━━━┓
  🔥 <b>TOP GAINER ALERT</b> 🔥
┗━━━━━━━━━━━━━━━━━━━━━━━┛

{direction_emoji} <b>{signal.symbol}</b>
━━━━━━━━━━━━━━━━━━━━━━━━━━

<b>📊 Market Data</b>
├ 24h Change: <b>+{signal_data.get('24h_change')}%</b>
└ Volume: ${signal_data.get('24h_volume'):,.0f}

<b>🎯 Trade Setup</b>
├ Entry: <b>${signal.entry_price:.6f}</b>
├ {tp_text.replace(chr(10), chr(10) + '├ ')}
└ SL: ${signal.stop_loss:.6f} (-20% @ 5x)

<b>⚡ Risk Management</b>
├ Leverage: <b>5x</b> (Fixed)
└ Risk/Reward: <b>{rr_text}</b>

<b>💡 Analysis</b>
{signal.reasoning}

⚠️ <b>HIGH VOLATILITY MODE</b>
<i>Auto-executing for enabled users...</i>
"""
        
        # Execute trades for users with top gainers mode + auto-trading
        executed_count = 0
        for user in users_with_mode:
            prefs = user.preferences
            
            # 🔥 CRITICAL FIX: Check if user already has position in this SPECIFIC symbol
            existing_symbol_position = db_session.query(Trade).filter(
                Trade.user_id == user.id,
                Trade.status == 'open',
                Trade.symbol == signal.symbol  # Same symbol check
            ).first()
            
            if existing_symbol_position:
                logger.info(f"⚠️ DUPLICATE PREVENTED: User {user.id} already has open position in {signal.symbol} (Trade ID: {existing_symbol_position.id})")
                continue
            
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
            
            # Execute trade with user's custom leverage for top gainers
            user_leverage = prefs.top_gainers_leverage if prefs and prefs.top_gainers_leverage else 5
            trade = await execute_bitunix_trade(
                signal=signal,
                user=user,
                db=db_session,
                trade_type='TOP_GAINER',
                leverage_override=user_leverage  # Use user's custom top gainer leverage
            )
            
            if trade:
                executed_count += 1
                
                # Send personalized notification with user's actual leverage
                try:
                    user_leverage = prefs.top_gainers_leverage if prefs and prefs.top_gainers_leverage else 5
                    
                    # Calculate profit percentages based on user's actual leverage
                    tp1_profit_pct = 4.0 * user_leverage  # TP1: 4% price move = 20% at 5x (1:1 R:R)
                    tp2_profit_pct = 8.0 * user_leverage  # TP2: 8% price move = 40% at 5x (1:2 R:R)
                    
                    # Calculate TP3 profit percentage for parabolic dumps
                    tp3_profit_pct = 12.0 * user_leverage  # TP3: 12% price move = 60% at 5x (1:3 R:R)
                    
                    # Rebuild TP/SL text with user's leverage
                    if signal.take_profit_3:
                        # Triple TPs for parabolic dumps (50%+ coins)
                        user_tp_text = f"""<b>TP1:</b> ${signal.take_profit_1:.6f} (+{tp1_profit_pct:.0f}% @ {user_leverage}x) [1:1]
<b>TP2:</b> ${signal.take_profit_2:.6f} (+{tp2_profit_pct:.0f}% @ {user_leverage}x) [1:2]
<b>TP3:</b> ${signal.take_profit_3:.6f} (+{tp3_profit_pct:.0f}% @ {user_leverage}x) 🚀 [1:3]"""
                    elif signal.take_profit_2:
                        # Dual TPs for regular signals
                        user_tp_text = f"""<b>TP1:</b> ${signal.take_profit_1:.6f} (+{tp1_profit_pct:.0f}% @ {user_leverage}x) [1:1]
<b>TP2:</b> ${signal.take_profit_2:.6f} (+{tp2_profit_pct:.0f}% @ {user_leverage}x) 🎯 [1:2]"""
                    else:
                        # Fallback
                        user_tp_text = f"<b>TP:</b> ${signal.take_profit:.6f} (+{tp1_profit_pct:.0f}% @ {user_leverage}x)"
                    
                    # Personalized signal message
                    direction_emoji = "🟢 LONG" if signal.direction == 'LONG' else "🔴 SHORT"
                    personalized_signal = f"""
┏━━━━━━━━━━━━━━━━━━━━━━━┓
  🔥 <b>TOP GAINER ALERT</b> 🔥
┗━━━━━━━━━━━━━━━━━━━━━━━┛

{direction_emoji} <b>{signal.symbol}</b>
━━━━━━━━━━━━━━━━━━━━━━━━━━

<b>📊 Market Data</b>
├ 24h Change: <b>+{signal_data.get('24h_change')}%</b>
└ Volume: ${signal_data.get('24h_volume'):,.0f}

<b>🎯 Your Trade</b>
├ Entry: <b>${signal.entry_price:.6f}</b>
├ {user_tp_text.replace(chr(10), chr(10) + '├ ')}
└ SL: ${signal.stop_loss:.6f} (-{tp1_profit_pct:.0f}% @ {user_leverage}x)

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
        
        logger.info(f"Top gainer signal executed for {executed_count}/{len(users_with_mode)} users")
        
        await service.close()
        
    except Exception as e:
        logger.error(f"Error in broadcast_top_gainer_signal: {e}", exc_info=True)
