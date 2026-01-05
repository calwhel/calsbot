"""
On-demand coin analysis service - provides market intelligence without generating signals
"""
import logging
import time
import os
import json
from datetime import datetime, timedelta
from typing import Dict, Optional
import ccxt.async_support as ccxt
from app.services.spot_monitor import SpotMarketMonitor

logger = logging.getLogger(__name__)

# Cache for storing scan analysis data (for "More Details" button)
_scan_cache: Dict[str, Dict] = {}


async def get_ai_trade_idea(
    symbol: str,
    current_price: float,
    market_data: Dict
) -> Optional[Dict]:
    """
    Use OpenAI to generate a smart trade idea based on market data.
    Returns entry, SL, TP levels with reasoning.
    """
    try:
        from openai import OpenAI
        
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            logger.warning("OPENAI_API_KEY not set, skipping AI trade idea")
            return None
        
        client = OpenAI(api_key=api_key)
        
        # Extract key data
        trend = market_data.get('trend', {})
        momentum = market_data.get('momentum', {})
        volume = market_data.get('volume', {})
        spot_flow = market_data.get('spot_flow', {})
        
        base_symbol = symbol.replace('/USDT', '').replace('USDT', '').upper()
        is_major = base_symbol in ['BTC', 'ETH', 'SOL', 'XRP', 'BNB']
        
        prompt = f"""You are an expert crypto trader. Analyze this market data and provide a trade idea.

COIN: {symbol}
CURRENT PRICE: ${current_price:.6f}
COIN TYPE: {"Major (moves slower)" if is_major else "Altcoin (moves faster)"}

MARKET DATA:
- 24h Change: {trend.get('change_24h', 0):.2f}%
- 5m Trend: {trend.get('timeframe_5m', 'unknown')}
- 15m Trend: {trend.get('timeframe_15m', 'unknown')}
- 1h Trend: {trend.get('timeframe_1h', 'unknown')}
- RSI (15m): {momentum.get('rsi', 50):.1f}
- MACD Signal: {momentum.get('macd_signal', 'neutral')}
- Volume Ratio: {volume.get('ratio', 1):.2f}x average
- Spot Flow: {spot_flow.get('signal', 'neutral')} ({spot_flow.get('buy_pressure', 50):.0f}% buy)
- Support: ${trend.get('support', current_price * 0.98):.6f}
- Resistance: ${trend.get('resistance', current_price * 1.02):.6f}

RULES:
1. For {base_symbol}, consider realistic timeframes:
   - SCALP: {"<1% move, 15-60 min" if is_major else "<2.5% move, 5-30 min"}
   - DAY TRADE: {"1-3% move, 4-12 hours" if is_major else "2.5-5% move, 1-4 hours"}  
   - SWING: {"3%+ move, multi-day" if is_major else "5%+ move, multi-day"}
2. SL should be below support for LONG, above resistance for SHORT
3. Minimum R:R ratio of 1.5:1
4. Consider current momentum and trend alignment

Respond in JSON format:
{{
    "direction": "LONG" or "SHORT",
    "trade_type": "SCALP" or "DAY TRADE" or "SWING",
    "trade_type_desc": "Brief timeframe description",
    "quality": "HIGH" or "MEDIUM" or "LOW",
    "entry": {current_price},
    "stop_loss": <price>,
    "sl_pct": <percentage from entry>,
    "tp1": <first target price>,
    "tp1_pct": <percentage profit>,
    "tp2": <second target price>,
    "tp2_pct": <percentage profit>,
    "rr_ratio": <risk reward ratio>,
    "confidence": 1-10,
    "reasoning": "2-3 sentence explanation of why this trade makes sense",
    "key_levels": "Brief note on key support/resistance"
}}"""

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are an expert crypto trader. Always respond with valid JSON. Be realistic about timeframes for different coins."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"},
            max_completion_tokens=600,
            timeout=20.0
        )
        
        result = json.loads(response.choices[0].message.content or "{}")
        
        # Validate required fields
        if not result.get('direction') or not result.get('entry'):
            return None
        
        # Add quality emoji
        quality = result.get('quality', 'MEDIUM')
        result['quality_emoji'] = {'HIGH': '🟢', 'MEDIUM': '🟡', 'LOW': '🔴'}.get(quality, '⚪')
        
        # Add trade type emoji
        trade_type = result.get('trade_type', 'DAY TRADE')
        result['trade_type_emoji'] = {'SCALP': '⚡', 'DAY TRADE': '📊', 'SWING': '🌊'}.get(trade_type, '📊')
        
        logger.info(f"🤖 AI Trade Idea for {symbol}: {result['direction']} {trade_type} - {quality} quality")
        
        return result
        
    except Exception as e:
        logger.error(f"AI trade idea error: {e}")
        return None

# Global cache for scan data (symbol -> {data, timestamp})
_scan_cache = {}
CACHE_TTL_SECONDS = 60  # Cache valid for 60 seconds


class CoinScanService:
    """Provides on-demand analysis of cryptocurrency price action"""
    
    def __init__(self):
        self.exchange = None
        self.spot_monitor = SpotMarketMonitor()
    
    def _get_cached(self, cache_key: str) -> Optional[Dict]:
        """Get cached data if still valid"""
        if cache_key in _scan_cache:
            cached = _scan_cache[cache_key]
            if time.time() - cached['timestamp'] < CACHE_TTL_SECONDS:
                return cached['data']
        return None
    
    def _set_cache(self, cache_key: str, data: Dict):
        """Store data in cache"""
        _scan_cache[cache_key] = {
            'data': data,
            'timestamp': time.time()
        }
    
    async def _fetch_candles_cached(self, symbol: str, timeframe: str, limit: int = 50):
        """Fetch candles with caching"""
        cache_key = f"candles_{symbol}_{timeframe}_{limit}"
        cached = self._get_cached(cache_key)
        if cached:
            return cached
        
        data = await self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        self._set_cache(cache_key, data)
        return data
    
    async def _fetch_ticker_cached(self, symbol: str):
        """Fetch ticker with caching"""
        cache_key = f"ticker_{symbol}"
        cached = self._get_cached(cache_key)
        if cached:
            return cached
        
        data = await self.exchange.fetch_ticker(symbol)
        self._set_cache(cache_key, data)
        return data
    
    async def initialize(self):
        """Initialize exchange connection"""
        if not self.exchange:
            self.exchange = ccxt.binance({
                'enableRateLimit': True,
                'options': {'defaultType': 'future'}
            })
    
    async def scan_coin(self, symbol: str) -> Dict:
        """
        Analyze a coin's market conditions
        
        Returns analysis including:
        - Trend direction and strength
        - Volume analysis
        - Momentum indicators
        - Institutional spot flow
        - Overall bias (bullish/bearish/neutral)
        """
        try:
            await self.initialize()
            
            # Normalize symbol
            if not symbol.endswith('/USDT'):
                symbol = f"{symbol.upper()}/USDT"
            
            # Get current price
            ticker = await self._fetch_ticker_cached(symbol)
            current_price = ticker['last']
            
            # Analyze different components
            trend_analysis = await self._analyze_trend(symbol)
            volume_analysis = await self._analyze_volume(symbol)
            momentum_analysis = await self._analyze_momentum(symbol)
            spot_flow_analysis = await self._analyze_spot_flow(symbol)
            session_analysis = self._analyze_session()
            volatility_analysis = await self._analyze_volatility(symbol)
            btc_correlation = await self._analyze_btc_correlation(symbol)
            
            # Calculate overall bias
            overall_bias = self._calculate_bias(
                trend_analysis,
                volume_analysis,
                momentum_analysis,
                spot_flow_analysis
            )
            
            # Generate trade idea - Try AI first, fallback to technical analysis
            market_data = {
                'trend': trend_analysis,
                'momentum': momentum_analysis,
                'volume': volume_analysis,
                'spot_flow': spot_flow_analysis
            }
            
            # Try AI-powered trade idea first
            trade_idea = await get_ai_trade_idea(symbol, current_price, market_data)
            
            # Fallback to technical analysis if AI fails
            if not trade_idea:
                trade_idea = await self._generate_trade_idea(
                    symbol,
                    trend_analysis,
                    volume_analysis,
                    momentum_analysis,
                    spot_flow_analysis,
                    current_price
                )
            else:
                # AI succeeded - add alternatives from technical analysis
                tech_idea = await self._generate_trade_idea(
                    symbol,
                    trend_analysis,
                    volume_analysis,
                    momentum_analysis,
                    spot_flow_analysis,
                    current_price
                )
                if tech_idea and tech_idea.get('alternatives'):
                    trade_idea['alternatives'] = tech_idea['alternatives']
                trade_idea['ai_generated'] = True
            
            # ═══════════════════════════════════════════════════════════════
            # ADVANCED FEATURES
            # ═══════════════════════════════════════════════════════════════
            
            # Get trade direction for entry timing
            direction = trade_idea.get('direction', 'LONG') if trade_idea else 'LONG'
            
            # Entry Timing Analysis
            entry_timing = await self._analyze_entry_timing(symbol, direction, current_price)
            
            # Sector Strength Analysis
            sector_analysis = await self._analyze_sector_strength(symbol)
            
            # Liquidation Zone Mapping
            liquidation_zones = await self._analyze_liquidation_zones(symbol, current_price)
            
            # News Sentiment Analysis
            news_sentiment = await self._analyze_news_sentiment(symbol)
            
            # Historical Context Analysis
            historical_context = await self._analyze_historical_context(symbol, current_price)
            
            # NEW: Funding Rate Analysis
            funding_rate = await self._analyze_funding_rate(symbol)
            
            # NEW: Open Interest Analysis
            open_interest = await self._analyze_open_interest(symbol)
            
            # NEW: Order Book / Whale Walls
            order_book = await self._analyze_order_book(symbol, current_price)
            
            # NEW: Multi-timeframe trend view
            mtf_trend = await self._analyze_mtf_trend(symbol)
            
            # NEW: Session Performance Patterns
            session_patterns = await self._analyze_session_patterns(symbol)
            
            # NEW: Long/Short Ratio
            long_short_ratio = await self._analyze_long_short_ratio(symbol)
            
            # NEW: RSI Divergence Detection
            rsi_divergence = await self._analyze_rsi_divergence(symbol)
            
            # NEW: Overall Conviction Score
            conviction_score = self._calculate_conviction_score(
                trend_analysis, volume_analysis, momentum_analysis, spot_flow_analysis,
                funding_rate, open_interest, order_book, long_short_ratio, mtf_trend
            )
            
            return {
                'success': True,
                'symbol': symbol,
                'price': current_price,
                'trend': trend_analysis,
                'volume': volume_analysis,
                'momentum': momentum_analysis,
                'spot_flow': spot_flow_analysis,
                'session': session_analysis,
                'volatility': volatility_analysis,
                'btc_correlation': btc_correlation,
                'overall_bias': overall_bias,
                'trade_idea': trade_idea,
                'entry_timing': entry_timing,
                'sector_analysis': sector_analysis,
                'liquidation_zones': liquidation_zones,
                'news_sentiment': news_sentiment,
                'historical_context': historical_context,
                'funding_rate': funding_rate,
                'open_interest': open_interest,
                'order_book': order_book,
                'mtf_trend': mtf_trend,
                'session_patterns': session_patterns,
                'long_short_ratio': long_short_ratio,
                'rsi_divergence': rsi_divergence,
                'conviction_score': conviction_score,
                'timestamp': datetime.utcnow()
            }
            
        except Exception as e:
            logger.error(f"Error scanning {symbol}: {e}", exc_info=True)
            return {
                'success': False,
                'error': str(e),
                'symbol': symbol
            }
    
    async def _analyze_trend(self, symbol: str) -> Dict:
        """Analyze trend using EMA 9/21 on 5m and 15m timeframes with support/resistance"""
        try:
            # Get 5m candles
            candles_5m = await self._fetch_candles_cached(symbol, '5m', limit=100)
            # Get 15m candles
            candles_15m = await self._fetch_candles_cached(symbol, '15m', limit=100)
            # Get 1H candles for support/resistance
            candles_1h = await self._fetch_candles_cached(symbol, '1h', limit=50)
            
            # Calculate EMAs
            ema9_5m = self._calculate_ema([c[4] for c in candles_5m], 9)
            ema21_5m = self._calculate_ema([c[4] for c in candles_5m], 21)
            ema9_15m = self._calculate_ema([c[4] for c in candles_15m], 9)
            ema21_15m = self._calculate_ema([c[4] for c in candles_15m], 21)
            
            # Determine trend
            trend_5m = "bullish" if ema9_5m > ema21_5m else "bearish"
            trend_15m = "bullish" if ema9_15m > ema21_15m else "bearish"
            
            # Calculate trend strength (% difference)
            strength_5m = abs((ema9_5m - ema21_5m) / ema21_5m * 100)
            strength_15m = abs((ema9_15m - ema21_15m) / ema21_15m * 100)
            
            aligned = trend_5m == trend_15m
            
            # Calculate support and resistance from 1H candles
            highs = [c[2] for c in candles_1h[-20:]]  # Last 20 candles
            lows = [c[3] for c in candles_1h[-20:]]
            
            resistance = max(highs)
            support = min(lows)
            current_price = candles_5m[-1][4]
            
            # Distance to levels
            to_resistance = ((resistance - current_price) / current_price) * 100
            to_support = ((current_price - support) / current_price) * 100
            
            return {
                'timeframe_5m': trend_5m,
                'timeframe_15m': trend_15m,
                'strength_5m': round(strength_5m, 2),
                'strength_15m': round(strength_15m, 2),
                'aligned': aligned,
                'current_ema9_5m': round(ema9_5m, 8),
                'current_ema21_5m': round(ema21_5m, 8),
                'support': round(support, 8),
                'resistance': round(resistance, 8),
                'to_support_pct': round(to_support, 2),
                'to_resistance_pct': round(to_resistance, 2)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing trend: {e}")
            return {'error': str(e)}
    
    async def _analyze_volume(self, symbol: str) -> Dict:
        """Analyze volume patterns"""
        try:
            candles = await self._fetch_candles_cached(symbol, '5m', limit=50)
            
            # Get recent volumes
            volumes = [c[5] for c in candles]
            current_volume = volumes[-1]
            avg_volume = sum(volumes[:-1]) / len(volumes[:-1])
            
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 0
            
            # Classify volume
            if volume_ratio > 2.0:
                status = "extreme"
            elif volume_ratio > 1.5:
                status = "high"
            elif volume_ratio > 1.3:
                status = "building"
            elif volume_ratio > 0.7:
                status = "normal"
            else:
                status = "low"
            
            return {
                'current': round(current_volume, 2),
                'average': round(avg_volume, 2),
                'ratio': round(volume_ratio, 2),
                'status': status
            }
            
        except Exception as e:
            logger.error(f"Error analyzing volume: {e}")
            return {'error': str(e)}
    
    async def _analyze_momentum(self, symbol: str) -> Dict:
        """Analyze momentum using MACD and RSI"""
        try:
            candles = await self._fetch_candles_cached(symbol, '5m', limit=100)
            closes = [c[4] for c in candles]
            
            # Calculate MACD
            ema12 = self._calculate_ema(closes, 12)
            ema26 = self._calculate_ema(closes, 26)
            macd = ema12 - ema26
            
            # Calculate RSI
            rsi = self._calculate_rsi(closes, 14)
            
            # Determine momentum
            macd_signal = "bullish" if macd > 0 else "bearish"
            
            if rsi > 70:
                rsi_status = "overbought"
            elif rsi > 65:
                rsi_status = "strong"
            elif rsi > 35:
                rsi_status = "neutral"
            elif rsi > 30:
                rsi_status = "weak"
            else:
                rsi_status = "oversold"
            
            return {
                'macd': round(macd, 8),
                'macd_signal': macd_signal,
                'rsi': round(rsi, 2),
                'rsi_status': rsi_status
            }
            
        except Exception as e:
            logger.error(f"Error analyzing momentum: {e}")
            return {'error': str(e)}
    
    async def _analyze_spot_flow(self, symbol: str) -> Dict:
        """Analyze institutional spot flow across exchanges"""
        try:
            # Remove /USDT suffix if present (spot monitor expects just 'BTC', 'ETH', etc.)
            clean_symbol = symbol.replace('/USDT', '').replace('USDT', '')
            
            # Use analyze_exchange_flow method from SpotMarketMonitor
            flow_data = await self.spot_monitor.analyze_exchange_flow(f"{clean_symbol}/USDT")
            
            if flow_data and not flow_data.get('error'):
                # Convert avg_pressure to buy/sell percentage
                # avg_pressure ranges from -1 (all selling) to +1 (all buying)
                avg_pressure = flow_data.get('avg_pressure', 0)
                
                # Convert to 0-100 scale: -1 = 0%, 0 = 50%, +1 = 100%
                buy_pressure = ((avg_pressure + 1) / 2) * 100
                sell_pressure = 100 - buy_pressure
                
                # Determine flow strength based on flow_signal
                flow_signal = flow_data.get('flow_signal', 'NEUTRAL')
                
                if flow_signal == 'HEAVY_BUYING':
                    signal = "strong_buying"
                    confidence = "institutional"
                elif flow_signal in ['VOLUME_SPIKE_BUY', 'MODERATE_BUYING']:
                    signal = "moderate_buying"
                    confidence = "high"
                elif flow_signal == 'HEAVY_SELLING':
                    signal = "strong_selling"
                    confidence = "institutional"
                elif flow_signal in ['VOLUME_SPIKE_SELL', 'MODERATE_SELLING']:
                    signal = "moderate_selling"
                    confidence = "high"
                else:  # NEUTRAL
                    signal = "neutral"
                    confidence = "low"
                
                return {
                    'buy_pressure': round(buy_pressure, 1),
                    'sell_pressure': round(sell_pressure, 1),
                    'signal': signal,
                    'confidence': confidence,
                    'exchanges_analyzed': flow_data.get('exchanges_analyzed', 0)
                }
            else:
                logger.warning(f"No spot flow data for {symbol}: {flow_data}")
                return {'error': 'No spot flow data available'}
                
        except Exception as e:
            logger.error(f"Error analyzing spot flow for {symbol}: {e}", exc_info=True)
            return {'error': str(e)}
    
    def _analyze_session(self) -> Dict:
        """Analyze current trading session quality"""
        current_hour = datetime.utcnow().hour
        
        # High liquidity: 8am-11pm UTC
        if 8 <= current_hour < 23:
            quality = "high"
            description = "Prime trading hours"
        else:
            quality = "low"
            description = "Low liquidity hours"
        
        return {
            'quality': quality,
            'description': description,
            'utc_hour': current_hour
        }
    
    async def _analyze_volatility(self, symbol: str) -> Dict:
        """Analyze volatility using ATR (Average True Range)"""
        try:
            candles_15m = await self._fetch_candles_cached(symbol, '15m', limit=20)
            candles_1h = await self._fetch_candles_cached(symbol, '1h', limit=20)
            
            # Calculate ATR for 15m
            atr_15m = self._calculate_atr(candles_15m, 14)
            current_price = candles_15m[-1][4]
            atr_pct_15m = (atr_15m / current_price) * 100
            
            # Calculate ATR for 1h
            atr_1h = self._calculate_atr(candles_1h, 14)
            atr_pct_1h = (atr_1h / current_price) * 100
            
            # Determine volatility regime
            if atr_pct_15m > 1.5:
                regime = "extreme"
                description = "Very high volatility - wider stops needed"
            elif atr_pct_15m > 0.8:
                regime = "high"
                description = "Elevated volatility - good for breakouts"
            elif atr_pct_15m > 0.4:
                regime = "normal"
                description = "Standard conditions"
            else:
                regime = "low"
                description = "Compressed - expect breakout soon"
            
            return {
                'atr_15m': round(atr_15m, 8),
                'atr_pct_15m': round(atr_pct_15m, 2),
                'atr_1h': round(atr_1h, 8),
                'atr_pct_1h': round(atr_pct_1h, 2),
                'regime': regime,
                'description': description,
                'suggested_sl_pct': round(atr_pct_15m * 2, 2)  # 2x ATR for SL
            }
            
        except Exception as e:
            logger.error(f"Error analyzing volatility: {e}")
            return {'error': str(e)}
    
    async def _analyze_btc_correlation(self, symbol: str) -> Dict:
        """Analyze correlation with BTC movement"""
        try:
            if 'BTC' in symbol:
                return {'correlation': 1.0, 'status': 'This is BTC', 'risk': 'N/A'}
            
            # Get BTC and symbol candles
            btc_candles = await self._fetch_candles_cached('BTC/USDT', '15m', limit=30)
            symbol_candles = await self._fetch_candles_cached(symbol, '15m', limit=30)
            
            if len(btc_candles) < 20 or len(symbol_candles) < 20:
                return {'error': 'Insufficient data'}
            
            # Calculate returns
            btc_returns = [(btc_candles[i][4] - btc_candles[i-1][4]) / btc_candles[i-1][4] 
                          for i in range(1, len(btc_candles))]
            symbol_returns = [(symbol_candles[i][4] - symbol_candles[i-1][4]) / symbol_candles[i-1][4] 
                             for i in range(1, len(symbol_candles))]
            
            # Simple correlation calculation
            n = min(len(btc_returns), len(symbol_returns))
            btc_returns = btc_returns[:n]
            symbol_returns = symbol_returns[:n]
            
            mean_btc = sum(btc_returns) / n
            mean_symbol = sum(symbol_returns) / n
            
            covariance = sum((btc_returns[i] - mean_btc) * (symbol_returns[i] - mean_symbol) for i in range(n)) / n
            std_btc = (sum((r - mean_btc) ** 2 for r in btc_returns) / n) ** 0.5
            std_symbol = (sum((r - mean_symbol) ** 2 for r in symbol_returns) / n) ** 0.5
            
            if std_btc > 0 and std_symbol > 0:
                correlation = covariance / (std_btc * std_symbol)
            else:
                correlation = 0
            
            # BTC current trend
            btc_change = ((btc_candles[-1][4] - btc_candles[-5][4]) / btc_candles[-5][4]) * 100
            btc_trend = "bullish" if btc_change > 0.3 else "bearish" if btc_change < -0.3 else "sideways"
            
            # Risk assessment
            if correlation > 0.7:
                if btc_trend == "bearish":
                    risk = "HIGH - BTC dumping, alt will follow"
                elif btc_trend == "bullish":
                    risk = "LOW - BTC pumping, alt should follow"
                else:
                    risk = "MEDIUM - BTC choppy"
            elif correlation > 0.4:
                risk = "MEDIUM - Moderate BTC dependency"
            else:
                risk = "LOW - Trades independently of BTC"
            
            return {
                'correlation': round(correlation, 2),
                'btc_trend': btc_trend,
                'btc_change_1h': round(btc_change, 2),
                'risk': risk
            }
            
        except Exception as e:
            logger.error(f"Error analyzing BTC correlation: {e}")
            return {'error': str(e)}
    
    def _calculate_atr(self, candles: list, period: int = 14) -> float:
        """Calculate Average True Range"""
        if len(candles) < period + 1:
            return 0
        
        true_ranges = []
        for i in range(1, len(candles)):
            high = candles[i][2]
            low = candles[i][3]
            prev_close = candles[i-1][4]
            
            tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
            true_ranges.append(tr)
        
        return sum(true_ranges[-period:]) / period
    
    def _calculate_bias(
        self,
        trend: Dict,
        volume: Dict,
        momentum: Dict,
        spot_flow: Dict
    ) -> Dict:
        """Calculate overall market bias"""
        score = 0
        max_score = 0
        reasons = []
        
        # Trend (weight: 2)
        if not trend.get('error'):
            max_score += 2
            if trend.get('aligned'):
                if trend.get('timeframe_5m') == 'bullish':
                    score += 2
                    reasons.append("✅ Bullish trend on both timeframes")
                else:
                    score -= 2
                    reasons.append("⛔ Bearish trend on both timeframes")
            else:
                reasons.append("⚠️ Trend not aligned across timeframes")
        
        # Spot flow (weight: 3 - highest priority)
        if not spot_flow.get('error'):
            max_score += 3
            if spot_flow.get('signal') == 'strong_buying':
                score += 3
                reasons.append("✅ Strong institutional buying")
            elif spot_flow.get('signal') == 'moderate_buying':
                score += 1.5
                reasons.append("✅ Moderate buying pressure")
            elif spot_flow.get('signal') == 'strong_selling':
                score -= 3
                reasons.append("⛔ Strong institutional selling")
            elif spot_flow.get('signal') == 'moderate_selling':
                score -= 1.5
                reasons.append("⛔ Moderate selling pressure")
        
        # Volume (weight: 1)
        if not volume.get('error'):
            max_score += 1
            if volume.get('status') in ['high', 'building']:
                score += 1
                reasons.append(f"✅ {volume.get('status').title()} volume")
            elif volume.get('status') == 'low':
                score -= 1
                reasons.append("⚠️ Low volume")
        
        # Momentum (weight: 1)
        if not momentum.get('error'):
            max_score += 1
            if momentum.get('macd_signal') == 'bullish' and momentum.get('rsi_status') not in ['overbought', 'oversold']:
                score += 1
                reasons.append("✅ Bullish momentum")
            elif momentum.get('macd_signal') == 'bearish':
                score -= 1
                reasons.append("⛔ Bearish momentum")
        
        # Determine overall bias
        if max_score > 0:
            percentage = (score / max_score) * 100
        else:
            percentage = 0
        
        if percentage >= 60:
            bias = "BULLISH"
            emoji = "🟢"
        elif percentage <= -60:
            bias = "BEARISH"
            emoji = "🔴"
        else:
            bias = "NEUTRAL"
            emoji = "⚪"
        
        return {
            'direction': bias,
            'strength': round(abs(percentage), 1),
            'emoji': emoji,
            'reasons': reasons
        }
    
    def _calculate_ema(self, prices: list, period: int) -> float:
        """Calculate Exponential Moving Average"""
        if len(prices) < period:
            return sum(prices) / len(prices)
        
        multiplier = 2 / (period + 1)
        ema = sum(prices[:period]) / period
        
        for price in prices[period:]:
            ema = (price * multiplier) + (ema * (1 - multiplier))
        
        return ema
    
    def _calculate_rsi(self, prices: list, period: int = 14) -> float:
        """Calculate Relative Strength Index"""
        if len(prices) < period + 1:
            return 50.0
        
        gains = []
        losses = []
        
        for i in range(1, len(prices)):
            change = prices[i] - prices[i-1]
            if change > 0:
                gains.append(change)
                losses.append(0)
            else:
                gains.append(0)
                losses.append(abs(change))
        
        avg_gain = sum(gains[-period:]) / period
        avg_loss = sum(losses[-period:]) / period
        
        if avg_loss == 0:
            return 100.0
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    # ═══════════════════════════════════════════════════════════════════════════
    # ADVANCED FEATURE 1: ENTRY TIMING SCORE
    # ═══════════════════════════════════════════════════════════════════════════
    async def _analyze_entry_timing(self, symbol: str, direction: str, current_price: float) -> Dict:
        """
        Advanced entry timing analysis - tells you exactly WHEN and WHERE to enter.
        Analyzes momentum acceleration, pullback depth, and optimal entry zones.
        """
        try:
            candles_1m = await self._fetch_candles_cached(symbol, '1m', limit=30)
            candles_5m = await self._fetch_candles_cached(symbol, '5m', limit=50)
            candles_15m = await self._fetch_candles_cached(symbol, '15m', limit=30)
            
            closes_1m = [c[4] for c in candles_1m]
            closes_5m = [c[4] for c in candles_5m]
            closes_15m = [c[4] for c in candles_15m]
            
            # Calculate key technical levels
            ema9_5m = self._calculate_ema(closes_5m, 9)
            ema21_5m = self._calculate_ema(closes_5m, 21)
            ema9_15m = self._calculate_ema(closes_15m, 9)
            ema21_15m = self._calculate_ema(closes_15m, 21)
            
            # RSI momentum and slope
            rsi_5m = self._calculate_rsi(closes_5m, 14)
            rsi_1m = self._calculate_rsi(closes_1m, 14)
            
            # Calculate RSI slope (momentum acceleration)
            rsi_values = []
            for i in range(len(closes_5m) - 5, len(closes_5m)):
                if i >= 14:
                    rsi_values.append(self._calculate_rsi(closes_5m[:i+1], 14))
            rsi_slope = (rsi_values[-1] - rsi_values[0]) / len(rsi_values) if len(rsi_values) >= 2 else 0
            
            # Recent swing levels for entry zones
            recent_lows = [c[3] for c in candles_5m[-10:]]
            recent_highs = [c[2] for c in candles_5m[-10:]]
            swing_low = min(recent_lows)
            swing_high = max(recent_highs)
            
            # Price position in range
            price_range = swing_high - swing_low
            if price_range > 0:
                position_in_range = (current_price - swing_low) / price_range * 100
            else:
                position_in_range = 50
            
            # Calculate pullback depth from recent high/low
            if direction == 'LONG':
                # For LONG: measure how far price has pulled back from swing high
                pullback_pct = ((swing_high - current_price) / swing_high) * 100
                distance_to_ema = ((current_price - ema21_5m) / ema21_5m) * 100
            else:
                # For SHORT: measure how close price is to swing high (resistance)
                # Small value = near resistance = ideal short entry
                pullback_pct = ((swing_high - current_price) / swing_high) * 100
                distance_to_ema = ((current_price - ema21_5m) / ema21_5m) * 100
            
            # Candle pattern analysis (last 3 candles)
            last_candles = candles_1m[-5:]
            green_count = sum(1 for c in last_candles if c[4] > c[1])
            red_count = len(last_candles) - green_count
            
            # Volume trend
            volumes_5m = [c[5] for c in candles_5m[-10:]]
            avg_volume = sum(volumes_5m[:-3]) / len(volumes_5m[:-3]) if len(volumes_5m) > 3 else volumes_5m[0]
            recent_volume = sum(volumes_5m[-3:]) / 3
            volume_surge = (recent_volume / avg_volume) if avg_volume > 0 else 1
            
            # ═══════════════════════════════════════════════════════════════
            # ENTRY TIMING DECISION LOGIC
            # ═══════════════════════════════════════════════════════════════
            timing_score = 0
            timing_signals = []
            
            if direction == 'LONG':
                # LONG ENTRY TIMING
                
                # 1. Pullback to support (ideal entry zone)
                if 0.3 <= pullback_pct <= 1.5:
                    timing_score += 3
                    timing_signals.append(f"✓ Healthy pullback ({pullback_pct:.1f}%) - ideal entry zone")
                elif pullback_pct < 0.3:
                    timing_score -= 1
                    timing_signals.append(f"⚠ Chasing ({pullback_pct:.1f}% pullback) - wait for dip")
                elif pullback_pct > 2.5:
                    timing_score += 1
                    timing_signals.append(f"⚠ Deep pullback ({pullback_pct:.1f}%) - confirm support holds")
                
                # 2. Near EMA support
                if -0.5 <= distance_to_ema <= 0.8:
                    timing_score += 2
                    timing_signals.append("✓ Price at EMA21 support zone")
                elif distance_to_ema > 2:
                    timing_score -= 2
                    timing_signals.append(f"⚠ Extended {distance_to_ema:.1f}% from EMA - wait for pullback")
                
                # 3. RSI momentum building
                if rsi_slope > 1.5:
                    timing_score += 2
                    timing_signals.append("✓ RSI accelerating upward - momentum building")
                elif rsi_slope < -1:
                    timing_score -= 1
                    timing_signals.append("⚠ RSI declining - wait for reversal")
                
                # 4. Bullish candle confirmation
                if green_count >= 3:
                    timing_score += 1
                    timing_signals.append("✓ Bullish candles forming")
                elif red_count >= 4:
                    timing_score -= 1
                    timing_signals.append("⚠ Still printing red candles - wait for green")
                
                # 5. Volume confirmation
                if volume_surge > 1.3:
                    timing_score += 1
                    timing_signals.append(f"✓ Volume surge {volume_surge:.1f}x - buyers active")
                
                # Calculate optimal entry zone
                optimal_entry = ema21_5m * 1.002  # Slightly above EMA21
                aggressive_entry = current_price
                conservative_entry = ema21_5m * 0.998  # At EMA21
                
            else:
                # SHORT ENTRY TIMING
                
                # 1. Bounce to resistance (ideal short entry)
                if 0.3 <= pullback_pct <= 1.5:
                    timing_score += 3
                    timing_signals.append(f"✓ Relief bounce ({pullback_pct:.1f}%) - ideal short zone")
                elif pullback_pct < 0.3:
                    timing_score -= 1
                    timing_signals.append(f"⚠ Weak bounce - wait for higher short")
                elif pullback_pct > 2.5:
                    timing_score += 1
                    timing_signals.append(f"⚠ Strong bounce ({pullback_pct:.1f}%) - risk of squeeze")
                
                # 2. Near EMA resistance
                if -0.5 <= distance_to_ema <= 0.8:
                    timing_score += 2
                    timing_signals.append("✓ Price at EMA21 resistance zone")
                elif distance_to_ema < -2:
                    timing_score -= 2
                    timing_signals.append(f"⚠ Extended {abs(distance_to_ema):.1f}% below EMA - bounce likely")
                
                # 3. RSI momentum fading
                if rsi_slope < -1.5:
                    timing_score += 2
                    timing_signals.append("✓ RSI rolling over - momentum fading")
                elif rsi_slope > 1:
                    timing_score -= 1
                    timing_signals.append("⚠ RSI still rising - wait for rejection")
                
                # 4. Bearish candle confirmation
                if red_count >= 3:
                    timing_score += 1
                    timing_signals.append("✓ Bearish candles forming")
                elif green_count >= 4:
                    timing_score -= 1
                    timing_signals.append("⚠ Still printing green - wait for red")
                
                # 5. Volume on rejection
                if volume_surge > 1.3:
                    timing_score += 1
                    timing_signals.append(f"✓ Volume on rejection {volume_surge:.1f}x")
                
                # Calculate optimal entry zone
                optimal_entry = ema21_5m * 0.998
                aggressive_entry = current_price
                conservative_entry = ema21_5m * 1.002
            
            # ═══════════════════════════════════════════════════════════════
            # GENERATE ENTRY RECOMMENDATION
            # ═══════════════════════════════════════════════════════════════
            if timing_score >= 5:
                urgency = "🟢 ENTER NOW"
                urgency_desc = "Strong entry timing - multiple confluences align. Execute on next candle close."
            elif timing_score >= 3:
                urgency = "🟡 GOOD ENTRY"
                urgency_desc = f"Solid entry zone. Consider scaling in at ${optimal_entry:,.4f} for better R:R."
            elif timing_score >= 1:
                urgency = "🟠 WAIT FOR PULLBACK"
                if direction == 'LONG':
                    urgency_desc = f"Entry is possible but not optimal. Better entry at ${conservative_entry:,.4f} (EMA21 zone)."
                else:
                    urgency_desc = f"Wait for bounce to ${conservative_entry:,.4f} for optimal short entry."
            else:
                urgency = "🔴 NOT YET"
                if direction == 'LONG':
                    urgency_desc = f"Poor timing - price extended or momentum weak. Wait for pullback to ${ema21_5m:,.4f}."
                else:
                    urgency_desc = f"Poor timing - wait for bounce to resistance around ${ema21_5m:,.4f}."
            
            return {
                'urgency': urgency,
                'urgency_desc': urgency_desc,
                'timing_score': timing_score,
                'signals': timing_signals,
                'optimal_entry': round(optimal_entry, 8),
                'aggressive_entry': round(aggressive_entry, 8),
                'conservative_entry': round(conservative_entry, 8),
                'pullback_pct': round(pullback_pct, 2),
                'ema_distance': round(distance_to_ema, 2),
                'rsi_slope': round(rsi_slope, 2),
                'position_in_range': round(position_in_range, 1)
            }
            
        except Exception as e:
            logger.error(f"Entry timing analysis error: {e}")
            return {'urgency': '⚪ UNKNOWN', 'urgency_desc': 'Could not analyze entry timing', 'timing_score': 0, 'signals': []}
    
    # ═══════════════════════════════════════════════════════════════════════════
    # ADVANCED FEATURE 2: SECTOR STRENGTH ANALYSIS
    # ═══════════════════════════════════════════════════════════════════════════
    SECTOR_MAPPING = {
        'L1': ['BTC', 'ETH', 'SOL', 'AVAX', 'ADA', 'DOT', 'ATOM', 'NEAR', 'APT', 'SUI', 'SEI', 'INJ', 'TIA'],
        'L2': ['ARB', 'OP', 'MATIC', 'IMX', 'STRK', 'ZK', 'MANTA', 'METIS', 'BOBA'],
        'AI': ['FET', 'AGIX', 'OCEAN', 'RNDR', 'TAO', 'ARKM', 'WLD', 'AI', 'NMR', 'CTXC'],
        'MEME': ['DOGE', 'SHIB', 'PEPE', 'FLOKI', 'BONK', 'WIF', 'BOME', 'MEME', 'TURBO', 'MYRO'],
        'DEFI': ['UNI', 'AAVE', 'LINK', 'MKR', 'SNX', 'CRV', 'COMP', 'SUSHI', 'YFI', 'LDO', 'RPL', 'GMX', 'DYDX', 'JUP', 'RAY'],
        'GAMING': ['AXS', 'SAND', 'MANA', 'GALA', 'ENJ', 'IMX', 'ALICE', 'ILV', 'PRIME', 'PIXEL', 'PORTAL'],
        'INFRA': ['FIL', 'AR', 'STORJ', 'GRT', 'THETA', 'HNT', 'RNDR', 'ANKR', 'LPT', 'PYTH'],
        'RWA': ['ONDO', 'POLYX', 'MKR', 'CFG', 'RIO', 'PROPC']
    }
    
    async def _analyze_sector_strength(self, symbol: str) -> Dict:
        """
        Analyze sector performance and rotation.
        Shows what's hot, what's lagging, and how the scanned coin compares.
        """
        try:
            base_symbol = symbol.replace('/USDT:USDT', '').replace('/USDT', '')
            
            # Find which sector this coin belongs to
            coin_sector = None
            for sector, coins in self.SECTOR_MAPPING.items():
                if base_symbol in coins:
                    coin_sector = sector
                    break
            
            # Fetch 24h performance for sector leaders
            sector_performance = {}
            top_sectors = []
            bottom_sectors = []
            
            for sector, coins in self.SECTOR_MAPPING.items():
                sector_changes = []
                for coin in coins[:5]:  # Top 5 per sector for speed
                    try:
                        ticker = await self._fetch_ticker_cached(f"{coin}/USDT")
                        if ticker and ticker.get('percentage'):
                            sector_changes.append(ticker['percentage'])
                    except:
                        continue
                
                if sector_changes:
                    avg_change = sum(sector_changes) / len(sector_changes)
                    sector_performance[sector] = {
                        'avg_change': round(avg_change, 2),
                        'coins_sampled': len(sector_changes)
                    }
            
            # Sort sectors by performance
            sorted_sectors = sorted(sector_performance.items(), key=lambda x: x[1]['avg_change'], reverse=True)
            
            # Get top 3 and bottom 3
            for i, (sector, data) in enumerate(sorted_sectors[:3]):
                emoji = ['🥇', '🥈', '🥉'][i]
                top_sectors.append(f"{emoji} {sector}: {data['avg_change']:+.1f}%")
            
            for sector, data in sorted_sectors[-3:][::-1]:
                bottom_sectors.append(f"📉 {sector}: {data['avg_change']:+.1f}%")
            
            # Coin's relative strength vs sector
            coin_ticker = await self._fetch_ticker_cached(symbol)
            coin_change = coin_ticker.get('percentage', 0) if coin_ticker else 0
            
            relative_strength = None
            sector_context = None
            if coin_sector and coin_sector in sector_performance:
                sector_avg = sector_performance[coin_sector]['avg_change']
                relative_strength = coin_change - sector_avg
                
                if relative_strength > 3:
                    sector_context = f"🚀 {base_symbol} is OUTPERFORMING {coin_sector} sector by {relative_strength:+.1f}% - sector leader"
                elif relative_strength > 0:
                    sector_context = f"💪 {base_symbol} is slightly STRONGER than {coin_sector} average ({relative_strength:+.1f}%)"
                elif relative_strength > -3:
                    sector_context = f"📊 {base_symbol} is tracking {coin_sector} sector average ({relative_strength:+.1f}%)"
                else:
                    sector_context = f"⚠️ {base_symbol} is UNDERPERFORMING {coin_sector} sector by {relative_strength:.1f}% - laggard"
            
            # Sector rotation insight
            rotation_insight = ""
            if sorted_sectors:
                top_sector = sorted_sectors[0][0]
                bottom_sector = sorted_sectors[-1][0]
                spread = sorted_sectors[0][1]['avg_change'] - sorted_sectors[-1][1]['avg_change']
                
                if spread > 10:
                    rotation_insight = f"🔄 Strong rotation into {top_sector}, away from {bottom_sector}. Consider sector alignment."
                elif spread > 5:
                    rotation_insight = f"📊 Moderate sector divergence. {top_sector} leading today."
                else:
                    rotation_insight = "⚖️ Balanced market - no strong sector rotation today."
            
            return {
                'coin_sector': coin_sector or 'Unknown',
                'coin_change': round(coin_change, 2),
                'sector_performance': sector_performance,
                'top_sectors': top_sectors,
                'bottom_sectors': bottom_sectors,
                'relative_strength': round(relative_strength, 2) if relative_strength else None,
                'sector_context': sector_context,
                'rotation_insight': rotation_insight
            }
            
        except Exception as e:
            logger.error(f"Sector analysis error: {e}")
            return {'coin_sector': 'Unknown', 'top_sectors': [], 'bottom_sectors': [], 'sector_context': 'Could not analyze sectors'}
    
    # ═══════════════════════════════════════════════════════════════════════════
    # ADVANCED FEATURE 3: LIQUIDATION ZONE MAPPING
    # ═══════════════════════════════════════════════════════════════════════════
    async def _analyze_liquidation_zones(self, symbol: str, current_price: float) -> Dict:
        """
        Estimate liquidation clusters based on:
        - Common leverage levels (5x, 10x, 20x, 50x, 100x)
        - Recent swing highs/lows (where traders likely entered)
        - Funding rate (positive = longs crowded, negative = shorts crowded)
        - Price levels where leveraged positions would get liquidated
        """
        try:
            candles_1h = await self._fetch_candles_cached(symbol, '1h', limit=48)
            candles_4h = await self._fetch_candles_cached(symbol, '4h', limit=30)
            
            # Find significant swing levels (potential entry points)
            highs_1h = [c[2] for c in candles_1h]
            lows_1h = [c[3] for c in candles_1h]
            
            # Recent swing high (where shorts may have entered)
            recent_swing_high = max(highs_1h[-12:])
            swing_high_48h = max(highs_1h)
            
            # Recent swing low (where longs may have entered)
            recent_swing_low = min(lows_1h[-12:])
            swing_low_48h = min(lows_1h)
            
            # Calculate liquidation zones for common leverage levels
            # Liquidation price = Entry * (1 - 1/leverage) for LONG
            # Liquidation price = Entry * (1 + 1/leverage) for SHORT
            
            liq_zones_above = []  # Short liquidations (above current price)
            liq_zones_below = []  # Long liquidations (below current price)
            
            leverage_levels = [5, 10, 20, 50, 100]
            
            # For LONGS entered at recent lows - where would they liquidate?
            for lev in leverage_levels:
                liq_price = recent_swing_low * (1 - 1/lev)
                distance_pct = ((current_price - liq_price) / current_price) * 100
                if distance_pct > 0 and distance_pct < 20:  # Within 20%
                    liq_zones_below.append({
                        'price': round(liq_price, 8),
                        'leverage': lev,
                        'entry_assumed': round(recent_swing_low, 8),
                        'distance_pct': round(distance_pct, 2),
                        'type': 'LONG_LIQ'
                    })
            
            # For SHORTS entered at recent highs - where would they liquidate?
            for lev in leverage_levels:
                liq_price = recent_swing_high * (1 + 1/lev)
                distance_pct = ((liq_price - current_price) / current_price) * 100
                if distance_pct > 0 and distance_pct < 20:
                    liq_zones_above.append({
                        'price': round(liq_price, 8),
                        'leverage': lev,
                        'entry_assumed': round(recent_swing_high, 8),
                        'distance_pct': round(distance_pct, 2),
                        'type': 'SHORT_LIQ'
                    })
            
            # Calculate liquidation density
            # More positions at lower leverage = more significant cluster
            density_above = sum(1 / z['leverage'] for z in liq_zones_above) if liq_zones_above else 0
            density_below = sum(1 / z['leverage'] for z in liq_zones_below) if liq_zones_below else 0
            
            # Find most significant liquidation level
            closest_above = min(liq_zones_above, key=lambda x: x['distance_pct']) if liq_zones_above else None
            closest_below = min(liq_zones_below, key=lambda x: x['distance_pct']) if liq_zones_below else None
            
            # Magnetism score - which direction has more liquidation fuel?
            if density_above > density_below * 1.5:
                magnet = "⬆️ UPSIDE MAGNET"
                magnet_desc = f"Heavy short liquidations above at ${closest_above['price']:,.4f} ({closest_above['distance_pct']:.1f}% away). Price may squeeze higher to hunt stops."
            elif density_below > density_above * 1.5:
                magnet = "⬇️ DOWNSIDE MAGNET"
                magnet_desc = f"Heavy long liquidations below at ${closest_below['price']:,.4f} ({closest_below['distance_pct']:.1f}% away). Price may sweep lows to hunt stops."
            else:
                magnet = "⚖️ BALANCED"
                magnet_desc = "Liquidation density balanced on both sides. No strong magnetic pull."
            
            # Build liquidation summary
            liq_summary = []
            if closest_above:
                liq_summary.append(f"🔴 Short liqs: ${closest_above['price']:,.4f} ({closest_above['distance_pct']:.1f}% above)")
            if closest_below:
                liq_summary.append(f"🟢 Long liqs: ${closest_below['price']:,.4f} ({closest_below['distance_pct']:.1f}% below)")
            
            # High-impact zone (where cascading liquidations likely)
            cascade_zone_up = recent_swing_high * 1.02  # 2% above recent high
            cascade_zone_down = recent_swing_low * 0.98  # 2% below recent low
            
            return {
                'magnet': magnet,
                'magnet_desc': magnet_desc,
                'liq_zones_above': liq_zones_above[:3],  # Top 3
                'liq_zones_below': liq_zones_below[:3],
                'closest_above': closest_above,
                'closest_below': closest_below,
                'density_above': round(density_above, 2),
                'density_below': round(density_below, 2),
                'liq_summary': liq_summary,
                'cascade_zone_up': round(cascade_zone_up, 8),
                'cascade_zone_down': round(cascade_zone_down, 8),
                'swing_high': round(recent_swing_high, 8),
                'swing_low': round(recent_swing_low, 8)
            }
            
        except Exception as e:
            logger.error(f"Liquidation zone analysis error: {e}")
            return {'magnet': '⚪ UNKNOWN', 'magnet_desc': 'Could not analyze liquidation zones', 'liq_summary': []}

    async def _generate_trade_idea(self, symbol: str, trend: Dict, volume: Dict, momentum: Dict, spot_flow: Dict, current_price: float) -> Dict:
        """
        Generate detailed LONG and SHORT day trade ideas for major alts.
        Analyzes multiple factors and returns the better setup.
        """
        try:
            # Fetch additional data for trade idea
            candles_15m = await self._fetch_candles_cached(symbol, '15m', limit=50)
            candles_1h = await self._fetch_candles_cached(symbol, '1h', limit=30)
            candles_4h = await self._fetch_candles_cached(symbol, '4h', limit=20)
            
            closes_15m = [c[4] for c in candles_15m]
            closes_1h = [c[4] for c in candles_1h]
            closes_4h = [c[4] for c in candles_4h]
            
            # Calculate key levels
            highs_1h = [c[2] for c in candles_1h[-24:]]
            lows_1h = [c[3] for c in candles_1h[-24:]]
            high_24h = max(highs_1h)
            low_24h = min(lows_1h)
            
            # Calculate EMAs
            ema9_15m = self._calculate_ema(closes_15m, 9)
            ema21_15m = self._calculate_ema(closes_15m, 21)
            ema21_1h = self._calculate_ema(closes_1h, 21)
            ema21_4h = self._calculate_ema(closes_4h, 21) if len(closes_4h) >= 21 else closes_4h[-1]
            
            # RSI values
            rsi_15m = self._calculate_rsi(closes_15m, 14)
            rsi_1h = self._calculate_rsi(closes_1h, 14)
            
            # Extension from EMAs
            extension_1h = ((current_price - ema21_1h) / ema21_1h) * 100
            extension_4h = ((current_price - ema21_4h) / ema21_4h) * 100
            
            # Distance from 24h high/low
            dist_from_high = ((high_24h - current_price) / current_price) * 100
            dist_from_low = ((current_price - low_24h) / current_price) * 100
            
            # ═══════════════════════════════════════════════════════
            # LONG SCORING
            # ═══════════════════════════════════════════════════════
            long_score = 0
            long_signals = []
            
            # Bullish trend alignment
            if trend.get('aligned') and trend.get('timeframe_15m') == 'bullish':
                long_score += 2
                long_signals.append("Bullish trend aligned (5m + 15m)")
            elif trend.get('timeframe_15m') == 'bullish':
                long_score += 1
                long_signals.append("15m trend bullish")
            
            # RSI in momentum zone (not overbought, not oversold)
            if 45 <= rsi_15m <= 65:
                long_score += 2
                long_signals.append(f"RSI(15m) in momentum zone ({rsi_15m:.0f})")
            elif 35 <= rsi_15m < 45:
                long_score += 1.5
                long_signals.append(f"RSI(15m) recovering from oversold ({rsi_15m:.0f})")
            
            # Price near EMA support (pullback entry)
            ema_distance = ((current_price - ema21_15m) / ema21_15m) * 100
            if 0 < ema_distance <= 1.5:
                long_score += 2.5
                long_signals.append(f"Near 15m EMA21 support ({ema_distance:.1f}% above)")
            elif -0.5 <= ema_distance <= 0:
                long_score += 2
                long_signals.append(f"Testing 15m EMA21 from above")
            
            # Strong spot buying
            if spot_flow.get('signal') == 'strong_buying':
                long_score += 3
                long_signals.append("Strong institutional buying")
            elif spot_flow.get('signal') == 'moderate_buying':
                long_score += 1.5
                long_signals.append("Moderate buying pressure")
            
            # Volume confirmation
            if volume.get('status') in ['high', 'building']:
                long_score += 1
                long_signals.append(f"{volume.get('status').title()} volume")
            
            # Higher low structure (bullish)
            recent_lows = [c[3] for c in candles_15m[-6:]]
            if len(recent_lows) >= 4 and recent_lows[-1] > min(recent_lows[:-1]):
                long_score += 1.5
                long_signals.append("Higher low forming on 15m")
            
            # ═══════════════════════════════════════════════════════
            # SHORT SCORING
            # ═══════════════════════════════════════════════════════
            short_score = 0
            short_signals = []
            
            # Check if trend is bullish - penalize shorts in uptrends
            is_bullish_trend = (ema9_15m > ema21_15m and current_price > ema21_15m)
            is_bearish_trend = (ema9_15m < ema21_15m and current_price < ema21_15m)
            
            # Only give overbought points if NOT in a strong uptrend, OR if extremely overbought
            if rsi_15m > 80 and rsi_1h > 75:
                # Extreme overbought - potential short even in uptrend
                short_score += 3
                short_signals.append(f"RSI extremely overbought ({rsi_15m:.0f}/{rsi_1h:.0f})")
            elif rsi_15m > 70 and not is_bullish_trend:
                short_score += 2
                short_signals.append(f"RSI(15m) overbought ({rsi_15m:.0f})")
            
            # Bearish trend - most important for shorts
            if is_bearish_trend:
                short_score += 3
                short_signals.append("Bearish trend confirmed (EMA + price)")
            elif trend.get('timeframe_15m') == 'bearish':
                short_score += 1.5
                short_signals.append("15m trend bearish")
            
            # Penalize shorts in clear uptrends
            if is_bullish_trend:
                short_score -= 2
                short_signals.append("⚠️ Bullish trend active (risky short)")
            
            # Extended from EMAs - only count if not in strong uptrend
            if extension_1h > 8 and extension_4h > 10:
                short_score += 2
                short_signals.append(f"Extremely extended from EMAs ({extension_1h:.1f}%)")
            elif extension_1h > 5 and not is_bullish_trend:
                short_score += 1
                short_signals.append(f"Extended {extension_1h:.1f}% above 1h EMA21")
            
            # Near 24h high (resistance) - only meaningful if showing weakness
            if dist_from_high < 1 and rsi_15m > 70:
                short_score += 1
                short_signals.append(f"Testing resistance with overbought RSI")
            
            # Selling pressure
            if spot_flow.get('signal') in ['strong_selling', 'moderate_selling']:
                short_score += 2
                short_signals.append(f"Institutional {spot_flow.get('signal').replace('_', ' ')}")
            
            # Volume at highs with bearish candles (distribution)
            last_candle = candles_15m[-1]
            is_red_candle = last_candle[4] < last_candle[1]
            if volume.get('status') in ['high', 'extreme'] and dist_from_high < 2 and is_red_candle:
                short_score += 1.5
                short_signals.append("High volume rejection at highs")
            
            # ═══════════════════════════════════════════════════════
            # DETERMINE BEST DIRECTION
            # ═══════════════════════════════════════════════════════
            
            # Calculate recent swing points for tighter SL (last 6 candles on 15m)
            recent_lows_15m = [c[3] for c in candles_15m[-6:]]
            recent_highs_15m = [c[2] for c in candles_15m[-6:]]
            swing_low = min(recent_lows_15m)
            swing_high = max(recent_highs_15m)
            
            if long_score >= short_score:
                direction = 'LONG'
                score = long_score
                signals = long_signals
                
                # LONG trade levels - flexible based on swing structure
                entry = current_price
                # SL below swing low with small buffer
                swing_sl = swing_low * 0.995  # 0.5% below swing low
                sl_distance_raw = ((entry - swing_sl) / entry) * 100
                
                # Clamp SL between 1.5% minimum and 6% maximum
                sl_distance = max(1.5, min(sl_distance_raw, 6.0))
                stop_loss = entry * (1 - sl_distance / 100)
                
                # TP levels: ensure minimum 1.2:1 R:R, scale with volatility
                tp1_mult = max(1.2, min(sl_distance_raw / 2 + 1, 2.0))  # 1.2x to 2x R:R
                tp2_mult = tp1_mult + 0.8  # TP2 is ~0.8 R:R higher
                
                tp1_target = entry * (1 + (sl_distance * tp1_mult / 100))
                tp2_target = entry * (1 + (sl_distance * tp2_mult / 100))
                tp1_profit = ((tp1_target - entry) / entry) * 100
                tp2_profit = ((tp2_target - entry) / entry) * 100
            else:
                direction = 'SHORT'
                score = short_score
                signals = short_signals
                
                # SHORT trade levels - flexible based on swing structure
                entry = current_price
                # SL above swing high with small buffer
                swing_sl = swing_high * 1.005  # 0.5% above swing high
                sl_distance_raw = ((swing_sl - entry) / entry) * 100
                
                # Clamp SL between 1.5% minimum and 6% maximum
                sl_distance = max(1.5, min(sl_distance_raw, 6.0))
                stop_loss = entry * (1 + sl_distance / 100)
                
                # TP levels: ensure minimum 1.2:1 R:R, scale with volatility
                tp1_mult = max(1.2, min(sl_distance_raw / 2 + 1, 2.0))  # 1.2x to 2x R:R
                tp2_mult = tp1_mult + 0.8  # TP2 is ~0.8 R:R higher
                
                tp1_target = entry * (1 - (sl_distance * tp1_mult / 100))
                tp2_target = entry * (1 - (sl_distance * tp2_mult / 100))
                tp1_profit = ((entry - tp1_target) / entry) * 100
                tp2_profit = ((entry - tp2_target) / entry) * 100
            
            # R:R calculation
            rr_ratio = tp1_profit / sl_distance if sl_distance > 0 else 1.2
            
            # ═══════════════════════════════════════════════════════
            # TRADE TYPE CLASSIFICATION (Scalp vs Day Trade vs Swing)
            # ═══════════════════════════════════════════════════════
            # Major coins (BTC, ETH) move slower - adjust thresholds
            base_symbol = symbol.replace('/USDT', '').replace('USDT', '').upper()
            is_major_coin = base_symbol in ['BTC', 'ETH', 'SOL', 'XRP', 'BNB']
            
            # For major coins: same % move = longer timeframe
            if is_major_coin:
                # BTC/ETH: 2% move is a day trade, not scalp
                if tp1_profit <= 1.0 and sl_distance <= 0.8:
                    trade_type = "SCALP"
                    trade_type_emoji = "⚡"
                    trade_type_desc = "Quick 15-60 min trade"
                elif tp1_profit <= 3.0 and sl_distance <= 2.5:
                    trade_type = "DAY TRADE"
                    trade_type_emoji = "📊"
                    trade_type_desc = "4-12 hour hold"
                else:
                    trade_type = "SWING"
                    trade_type_emoji = "🌊"
                    trade_type_desc = "Multi-day hold"
            else:
                # Altcoins: original thresholds
                if tp1_profit <= 2.5 and sl_distance <= 2:
                    trade_type = "SCALP"
                    trade_type_emoji = "⚡"
                    trade_type_desc = "Quick 5-30 min trade"
                elif tp1_profit <= 5 and sl_distance <= 4:
                    trade_type = "DAY TRADE"
                    trade_type_emoji = "📊"
                    trade_type_desc = "1-4 hour hold"
                else:
                    trade_type = "SWING"
                    trade_type_emoji = "🌊"
                    trade_type_desc = "Multi-hour to multi-day"
            
            # ═══════════════════════════════════════════════════════
            # GENERATE ALTERNATIVE TRADE OPTIONS (All 3 timeframes)
            # ═══════════════════════════════════════════════════════
            alternatives = []
            
            # SCALP option: Tight SL/TP for quick trades
            if direction == 'LONG':
                scalp_sl = 1.5
                scalp_tp1 = 2.0
                scalp_tp2 = 3.0
                scalp_sl_price = entry * (1 - scalp_sl / 100)
                scalp_tp1_price = entry * (1 + scalp_tp1 / 100)
                scalp_tp2_price = entry * (1 + scalp_tp2 / 100)
            else:
                scalp_sl = 1.5
                scalp_tp1 = 2.0
                scalp_tp2 = 3.0
                scalp_sl_price = entry * (1 + scalp_sl / 100)
                scalp_tp1_price = entry * (1 - scalp_tp1 / 100)
                scalp_tp2_price = entry * (1 - scalp_tp2 / 100)
            
            alternatives.append({
                'type': 'SCALP',
                'emoji': '⚡',
                'desc': '5-30 min',
                'sl_pct': scalp_sl,
                'tp1_pct': scalp_tp1,
                'tp2_pct': scalp_tp2,
                'sl_price': round(scalp_sl_price, 8),
                'tp1_price': round(scalp_tp1_price, 8),
                'tp2_price': round(scalp_tp2_price, 8)
            })
            
            # DAY TRADE option: Medium SL/TP
            if direction == 'LONG':
                day_sl = 3.0
                day_tp1 = 4.5
                day_tp2 = 7.0
                day_sl_price = entry * (1 - day_sl / 100)
                day_tp1_price = entry * (1 + day_tp1 / 100)
                day_tp2_price = entry * (1 + day_tp2 / 100)
            else:
                day_sl = 3.0
                day_tp1 = 4.5
                day_tp2 = 7.0
                day_sl_price = entry * (1 + day_sl / 100)
                day_tp1_price = entry * (1 - day_tp1 / 100)
                day_tp2_price = entry * (1 - day_tp2 / 100)
            
            alternatives.append({
                'type': 'DAY TRADE',
                'emoji': '📊',
                'desc': '1-4 hours',
                'sl_pct': day_sl,
                'tp1_pct': day_tp1,
                'tp2_pct': day_tp2,
                'sl_price': round(day_sl_price, 8),
                'tp1_price': round(day_tp1_price, 8),
                'tp2_price': round(day_tp2_price, 8)
            })
            
            # SWING option: Wider SL/TP for longer holds
            if direction == 'LONG':
                swing_sl = 5.0
                swing_tp1 = 8.0
                swing_tp2 = 12.0
                swing_sl_price = entry * (1 - swing_sl / 100)
                swing_tp1_price = entry * (1 + swing_tp1 / 100)
                swing_tp2_price = entry * (1 + swing_tp2 / 100)
            else:
                swing_sl = 5.0
                swing_tp1 = 8.0
                swing_tp2 = 12.0
                swing_sl_price = entry * (1 + swing_sl / 100)
                swing_tp1_price = entry * (1 - swing_tp1 / 100)
                swing_tp2_price = entry * (1 - swing_tp2 / 100)
            
            alternatives.append({
                'type': 'SWING',
                'emoji': '🌊',
                'desc': 'Multi-day',
                'sl_pct': swing_sl,
                'tp1_pct': swing_tp1,
                'tp2_pct': swing_tp2,
                'sl_price': round(swing_sl_price, 8),
                'tp1_price': round(swing_tp1_price, 8),
                'tp2_price': round(swing_tp2_price, 8)
            })
            
            # Quality rating
            if score >= 8:
                quality = "HIGH"
                quality_emoji = "🟢"
            elif score >= 5:
                quality = "MEDIUM"
                quality_emoji = "🟡"
            elif score >= 3:
                quality = "LOW"
                quality_emoji = "🟠"
            else:
                quality = "NO TRADE"
                quality_emoji = "🔴"
            
            # ═══════════════════════════════════════════════════════
            # GENERATE AI-STYLE ANALYSIS AND REASONING
            # ═══════════════════════════════════════════════════════
            base_symbol = symbol.replace('/USDT:USDT', '').replace('/USDT', '')
            
            # Build natural language analysis
            trend_desc = ""
            if ema9_15m > ema21_15m and current_price > ema21_15m:
                trend_desc = "maintaining bullish structure with price holding above key moving averages"
            elif ema9_15m < ema21_15m and current_price < ema21_15m:
                trend_desc = "showing bearish momentum with price trading below key moving averages"
            elif ema9_15m > ema21_15m:
                trend_desc = "in an uptrend but currently testing support levels"
            else:
                trend_desc = "consolidating near key levels awaiting a breakout"
            
            rsi_desc = ""
            if rsi_15m > 70:
                rsi_desc = f"RSI at {rsi_15m:.0f} indicates overbought conditions, suggesting a potential pullback."
            elif rsi_15m < 30:
                rsi_desc = f"RSI at {rsi_15m:.0f} shows oversold conditions, signaling a possible bounce."
            elif 50 <= rsi_15m <= 65:
                rsi_desc = f"RSI at {rsi_15m:.0f} shows healthy momentum without being extended."
            elif 35 <= rsi_15m < 50:
                rsi_desc = f"RSI at {rsi_15m:.0f} suggests the asset is recovering from recent weakness."
            else:
                rsi_desc = f"RSI at {rsi_15m:.0f} is neutral, indicating no extreme momentum."
            
            # Volume and flow description
            flow_desc = ""
            if spot_flow.get('signal') == 'strong_buying':
                flow_desc = "Strong institutional buying pressure detected, supporting upside potential."
            elif spot_flow.get('signal') == 'moderate_buying':
                flow_desc = "Moderate buying activity suggests accumulation at current levels."
            elif spot_flow.get('signal') == 'strong_selling':
                flow_desc = "Heavy selling pressure observed, indicating distribution phase."
            elif spot_flow.get('signal') == 'moderate_selling':
                flow_desc = "Moderate selling activity suggests weakness in demand."
            else:
                flow_desc = "Order flow appears balanced with no dominant directional bias."
            
            # Build the recommendation
            if direction == 'LONG':
                if quality == "HIGH":
                    recommendation = f"Strong long setup on {base_symbol}. Multiple bullish confluences align with price near support and healthy momentum. Consider entering on the next green candle with stop below recent swing low."
                elif quality == "MEDIUM":
                    recommendation = f"Moderate long opportunity on {base_symbol}. Wait for a confirmation candle or a pullback to EMA support before entering. Manage risk tightly with the suggested stop loss."
                elif quality == "LOW":
                    recommendation = f"Weak long setup on {base_symbol}. The risk-reward is not ideal at current levels. Consider waiting for a deeper pullback or clearer momentum signals."
                else:
                    recommendation = f"No clear long setup on {base_symbol}. Price action lacks direction - patience is recommended until a stronger opportunity develops."
            else:
                if quality == "HIGH":
                    recommendation = f"Strong short setup on {base_symbol}. Price is extended and showing signs of exhaustion near resistance. Look for bearish confirmation before entering."
                elif quality == "MEDIUM":
                    recommendation = f"Moderate short opportunity on {base_symbol}. Wait for rejection candles at resistance before shorting. Keep stop tight above recent highs."
                elif quality == "LOW":
                    recommendation = f"Weak short setup on {base_symbol}. Shorting here carries higher risk - wait for clearer distribution signals or momentum breakdown."
                else:
                    recommendation = f"No clear short setup on {base_symbol}. Current structure doesn't favor shorting - wait for price to reach resistance or show clearer weakness."
            
            # Build AI-style reasoning paragraph
            analysis_text = f"{base_symbol} is currently {trend_desc}. {rsi_desc} {flow_desc}"
            
            if direction == 'LONG':
                sl_context = f"The stop loss at ${stop_loss:,.4f} ({sl_distance:.1f}% risk) is placed below the recent swing low for protection."
                tp_context = f"Target 1 at ${tp1_target:,.4f} (+{tp1_profit:.1f}%) offers a {rr_ratio:.1f}:1 reward-to-risk ratio."
            else:
                sl_context = f"The stop loss at ${stop_loss:,.4f} ({sl_distance:.1f}% risk) is placed above the recent swing high for protection."
                tp_context = f"Target 1 at ${tp1_target:,.4f} (+{tp1_profit:.1f}%) offers a {rr_ratio:.1f}:1 reward-to-risk ratio."
            
            reasoning = f"{analysis_text}\n\n{sl_context} {tp_context}"
            
            return {
                'has_setup': score >= 3,
                'direction': direction,
                'quality': quality,
                'quality_emoji': quality_emoji,
                'score': round(score, 1),
                'entry': round(entry, 8),
                'stop_loss': round(stop_loss, 8),
                'sl_distance_pct': round(sl_distance, 2),
                'tp1': round(tp1_target, 8),
                'tp1_profit_pct': round(tp1_profit, 2),
                'tp2': round(tp2_target, 8),
                'tp2_profit_pct': round(tp2_profit, 2),
                'rr_ratio': round(rr_ratio, 2),
                'trade_type': trade_type,
                'trade_type_emoji': trade_type_emoji,
                'trade_type_desc': trade_type_desc,
                'alternatives': alternatives,
                'recommendation': recommendation,
                'reasoning': reasoning,
                'signals': signals,
                'long_score': round(long_score, 1),
                'short_score': round(short_score, 1)
            }
            
        except Exception as e:
            logger.error(f"Error generating trade idea: {e}", exc_info=True)
            return {'error': str(e)}
    
    async def _analyze_news_sentiment(self, symbol: str) -> Dict:
        """
        Quick news sentiment check for a coin using AI analysis
        Uses free CryptoPanic API or falls back to AI-based title analysis
        """
        import os
        import httpx
        from openai import OpenAI
        
        try:
            base_symbol = symbol.replace('/USDT', '').replace('USDT', '')
            
            # Try CryptoPanic free API (no key needed for basic access)
            news_items = []
            try:
                async with httpx.AsyncClient(timeout=10.0) as client:
                    # CryptoPanic free API endpoint
                    url = f"https://cryptopanic.com/api/free/v1/posts/?currencies={base_symbol}&public=true"
                    response = await client.get(url)
                    if response.status_code == 200:
                        data = response.json()
                        news_items = data.get('results', [])[:5]  # Latest 5 news
            except Exception as e:
                logger.debug(f"CryptoPanic API unavailable: {e}")
            
            # If no news from API, try AI-based general market sentiment
            if not news_items:
                try:
                    client = OpenAI(
                        api_key=os.environ.get("AI_INTEGRATIONS_OPENAI_API_KEY"),
                        base_url=os.environ.get("AI_INTEGRATIONS_OPENAI_BASE_URL")
                    )
                    
                    prompt = f"""As a crypto trading analyst, provide a brief sentiment assessment for {base_symbol} based on general market conditions and recent trends.

Respond in JSON format:
{{
    "sentiment": "bullish" | "bearish" | "neutral",
    "impact_score": 1-5 (general market conditions - keep low without specific news),
    "summary": "One sentence about current market sentiment for this coin"
}}"""
                    
                    response = client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[
                            {"role": "system", "content": "You are a crypto trading sentiment analyst. Respond only with valid JSON."},
                            {"role": "user", "content": prompt}
                        ],
                        max_tokens=150,
                        response_format={"type": "json_object"}
                    )
                    
                    import json
                    analysis = json.loads(response.choices[0].message.content)
                    
                    sentiment = analysis.get('sentiment', 'neutral').lower()
                    emoji_map = {'bullish': '🟢', 'bearish': '🔴', 'neutral': '⚪'}
                    
                    return {
                        'has_news': False,
                        'sentiment': sentiment,
                        'sentiment_emoji': emoji_map.get(sentiment, '⚪'),
                        'summary': analysis.get('summary', 'General market sentiment assessment'),
                        'headlines': [],
                        'impact_score': min(5, max(1, int(analysis.get('impact_score', 2))))
                    }
                except Exception as e:
                    logger.debug(f"AI general sentiment unavailable: {e}")
                    return {
                        'has_news': False,
                        'sentiment': 'neutral',
                        'sentiment_emoji': '⚪',
                        'summary': 'No recent news found for this coin',
                        'headlines': [],
                        'impact_score': 0
                    }
            
            # Extract headlines
            headlines = [item.get('title', '')[:100] for item in news_items[:5]]
            
            # Use OpenAI to analyze sentiment
            try:
                client = OpenAI(
                    api_key=os.environ.get("AI_INTEGRATIONS_OPENAI_API_KEY"),
                    base_url=os.environ.get("AI_INTEGRATIONS_OPENAI_BASE_URL")
                )
                
                prompt = f"""Analyze these recent {base_symbol} crypto news headlines for trading sentiment:

{chr(10).join([f'- {h}' for h in headlines])}

Respond in JSON format:
{{
    "sentiment": "bullish" | "bearish" | "neutral",
    "impact_score": 1-10 (how likely to move price),
    "summary": "One sentence trading-relevant summary"
}}"""
                
                # the newest OpenAI model is "gpt-5" which was released August 7, 2025
                # do not change this unless explicitly requested by the user
                response = client.chat.completions.create(
                    model="gpt-4o-mini",  # Using mini for speed/cost
                    messages=[
                        {"role": "system", "content": "You are a crypto trading sentiment analyst. Respond only with valid JSON."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=200,
                    response_format={"type": "json_object"}
                )
                
                import json
                analysis = json.loads(response.choices[0].message.content)
                
                sentiment = analysis.get('sentiment', 'neutral').lower()
                emoji_map = {'bullish': '🟢', 'bearish': '🔴', 'neutral': '⚪'}
                
                return {
                    'has_news': True,
                    'sentiment': sentiment,
                    'sentiment_emoji': emoji_map.get(sentiment, '⚪'),
                    'summary': analysis.get('summary', 'Recent news activity detected'),
                    'headlines': headlines[:3],
                    'impact_score': min(10, max(1, analysis.get('impact_score', 3)))
                }
                
            except Exception as e:
                logger.debug(f"AI sentiment analysis unavailable: {e}")
                # Fallback: just report news exists
                return {
                    'has_news': True,
                    'sentiment': 'neutral',
                    'sentiment_emoji': '⚪',
                    'summary': f"Found {len(headlines)} recent news items",
                    'headlines': headlines[:3],
                    'impact_score': 3
                }
                
        except Exception as e:
            logger.error(f"Error analyzing news sentiment: {e}")
            return {
                'has_news': False,
                'sentiment': 'neutral',
                'sentiment_emoji': '⚪',
                'summary': 'Unable to check news',
                'headlines': [],
                'impact_score': 0
            }
    
    async def _analyze_historical_context(self, symbol: str, current_price: float) -> Dict:
        """
        Analyze historical price behavior at current levels
        - Previous touches/reactions at this price zone
        - Time since last major move
        - Win rate for bounces/rejections
        """
        try:
            # Get daily candles for historical context
            candles_1d = await self._fetch_candles_cached(symbol, '1d', limit=90)  # 3 months
            candles_4h = await self._fetch_candles_cached(symbol, '4h', limit=168)  # 1 week
            
            if len(candles_1d) < 30:
                return {'error': 'Insufficient historical data'}
            
            # Define price zone (±1.5% of current price)
            zone_pct = 0.015
            zone_high = current_price * (1 + zone_pct)
            zone_low = current_price * (1 - zone_pct)
            
            # Count touches in this zone on daily
            touches = []
            for i, candle in enumerate(candles_1d[:-1]):  # Exclude current
                high = candle[2]
                low = candle[3]
                close = candle[4]
                timestamp = candle[0]
                
                # Check if price touched this zone
                if low <= zone_high and high >= zone_low:
                    # Determine reaction
                    next_candle = candles_1d[i + 1] if i + 1 < len(candles_1d) else None
                    if next_candle:
                        reaction = 'bounce' if next_candle[4] > close else 'rejection'
                        move_pct = abs((next_candle[4] - close) / close) * 100
                        touches.append({
                            'date': datetime.fromtimestamp(timestamp / 1000),
                            'reaction': reaction,
                            'move_pct': move_pct
                        })
            
            # Calculate bounce/rejection stats
            bounces = [t for t in touches if t['reaction'] == 'bounce']
            rejections = [t for t in touches if t['reaction'] == 'rejection']
            
            total_touches = len(touches)
            bounce_rate = (len(bounces) / total_touches * 100) if total_touches > 0 else 50
            
            # Find last major move (>10% in either direction)
            last_major_pump = None
            last_major_dump = None
            
            for i in range(len(candles_1d) - 1, -1, -1):
                candle = candles_1d[i]
                change = ((candle[4] - candle[1]) / candle[1]) * 100
                date = datetime.fromtimestamp(candle[0] / 1000)
                
                if change >= 10 and not last_major_pump:
                    last_major_pump = {
                        'date': date,
                        'change': change,
                        'days_ago': (datetime.utcnow() - date).days
                    }
                elif change <= -10 and not last_major_dump:
                    last_major_dump = {
                        'date': date,
                        'change': abs(change),
                        'days_ago': (datetime.utcnow() - date).days
                    }
                
                if last_major_pump and last_major_dump:
                    break
            
            # Find recent swing highs and lows for context
            highs_4h = [c[2] for c in candles_4h]
            lows_4h = [c[3] for c in candles_4h]
            
            recent_high = max(highs_4h[-42:])  # 1 week
            recent_low = min(lows_4h[-42:])
            
            # Position in range
            range_size = recent_high - recent_low
            position_in_range = ((current_price - recent_low) / range_size * 100) if range_size > 0 else 50
            
            # Generate insight
            if bounce_rate >= 65 and total_touches >= 2:
                zone_behavior = f"🟢 STRONG SUPPORT ZONE - Price has bounced {len(bounces)}/{total_touches} times ({bounce_rate:.0f}% bounce rate)"
            elif bounce_rate <= 35 and total_touches >= 2:
                zone_behavior = f"🔴 STRONG RESISTANCE ZONE - Price rejected {len(rejections)}/{total_touches} times ({100-bounce_rate:.0f}% rejection rate)"
            elif total_touches >= 2:
                zone_behavior = f"⚪ CONTESTED ZONE - Mixed reactions: {len(bounces)} bounces, {len(rejections)} rejections"
            else:
                zone_behavior = "⚪ FRESH PRICE ZONE - Limited historical data at this level"
            
            # Time context
            time_context = ""
            if last_major_pump:
                time_context += f"📈 Last 10%+ pump: {last_major_pump['days_ago']} days ago (+{last_major_pump['change']:.1f}%)\n"
            if last_major_dump:
                time_context += f"📉 Last 10%+ dump: {last_major_dump['days_ago']} days ago (-{last_major_dump['change']:.1f}%)"
            
            if not time_context:
                time_context = "No major moves (10%+) in the last 90 days"
            
            # Range position insight
            if position_in_range >= 80:
                range_insight = f"🔺 Near weekly high - {position_in_range:.0f}% of range"
            elif position_in_range <= 20:
                range_insight = f"🔻 Near weekly low - {position_in_range:.0f}% of range"
            else:
                range_insight = f"↔️ Mid-range - {position_in_range:.0f}% of weekly range"
            
            return {
                'zone_behavior': zone_behavior,
                'total_touches': total_touches,
                'bounce_rate': round(bounce_rate, 1),
                'bounces': len(bounces),
                'rejections': len(rejections),
                'last_major_pump': last_major_pump,
                'last_major_dump': last_major_dump,
                'time_context': time_context,
                'range_insight': range_insight,
                'position_in_range': round(position_in_range, 1),
                'recent_high': round(recent_high, 8),
                'recent_low': round(recent_low, 8)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing historical context: {e}")
            return {
                'zone_behavior': '⚪ Unable to analyze historical context',
                'total_touches': 0,
                'bounce_rate': 50,
                'time_context': 'Data unavailable',
                'range_insight': 'Data unavailable'
            }
    
    async def _analyze_funding_rate(self, symbol: str) -> Dict:
        """Analyze perpetual funding rate - indicates market sentiment"""
        try:
            # Binance funding rate endpoint
            import httpx
            base_symbol = symbol.replace('/USDT', '').replace(':USDT', '')
            
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"https://fapi.binance.com/fapi/v1/fundingRate",
                    params={'symbol': f'{base_symbol}USDT', 'limit': 10},
                    timeout=10
                )
                
                if response.status_code == 200:
                    data = response.json()
                    if data:
                        current_rate = float(data[-1]['fundingRate']) * 100  # Convert to %
                        
                        # Calculate average of last 10
                        rates = [float(d['fundingRate']) * 100 for d in data]
                        avg_rate = sum(rates) / len(rates)
                        
                        # Determine sentiment
                        if current_rate > 0.1:
                            sentiment = "🔴 LONGS PAYING"
                            bias = "Crowded long - potential squeeze down"
                        elif current_rate < -0.1:
                            sentiment = "🟢 SHORTS PAYING"
                            bias = "Crowded short - potential squeeze up"
                        elif current_rate > 0.03:
                            sentiment = "🟡 SLIGHTLY BULLISH"
                            bias = "Mild long bias"
                        elif current_rate < -0.03:
                            sentiment = "🟡 SLIGHTLY BEARISH"
                            bias = "Mild short bias"
                        else:
                            sentiment = "⚪ NEUTRAL"
                            bias = "Balanced market"
                        
                        return {
                            'current_rate': round(current_rate, 4),
                            'avg_rate': round(avg_rate, 4),
                            'sentiment': sentiment,
                            'bias': bias,
                            'annualized': round(current_rate * 3 * 365, 2)  # 8h funding * 3 * 365
                        }
            
            return {'sentiment': '⚪ N/A', 'bias': 'Funding data unavailable'}
            
        except Exception as e:
            logger.debug(f"Funding rate error: {e}")
            return {'sentiment': '⚪ N/A', 'bias': 'Funding data unavailable'}
    
    async def _analyze_open_interest(self, symbol: str) -> Dict:
        """Analyze open interest changes - indicates market conviction"""
        try:
            import httpx
            base_symbol = symbol.replace('/USDT', '').replace(':USDT', '')
            
            async with httpx.AsyncClient() as client:
                # Get OI history
                response = await client.get(
                    f"https://fapi.binance.com/futures/data/openInterestHist",
                    params={'symbol': f'{base_symbol}USDT', 'period': '1h', 'limit': 24},
                    timeout=10
                )
                
                if response.status_code == 200:
                    data = response.json()
                    if data and len(data) >= 2:
                        current_oi = float(data[-1]['sumOpenInterest'])
                        oi_1h_ago = float(data[-2]['sumOpenInterest'])
                        oi_24h_ago = float(data[0]['sumOpenInterest'])
                        
                        change_1h = ((current_oi - oi_1h_ago) / oi_1h_ago) * 100
                        change_24h = ((current_oi - oi_24h_ago) / oi_24h_ago) * 100
                        
                        # Interpret OI changes
                        if change_1h > 5:
                            signal = "🔥 RAPID BUILD"
                            desc = "New positions opening fast - momentum building"
                        elif change_1h > 2:
                            signal = "📈 INCREASING"
                            desc = "New positions entering"
                        elif change_1h < -5:
                            signal = "🚨 RAPID UNWIND"
                            desc = "Mass position closing - volatility ahead"
                        elif change_1h < -2:
                            signal = "📉 DECREASING"
                            desc = "Positions being closed"
                        else:
                            signal = "➡️ STABLE"
                            desc = "OI relatively unchanged"
                        
                        return {
                            'current_oi': round(current_oi / 1_000_000, 2),  # In millions
                            'change_1h': round(change_1h, 2),
                            'change_24h': round(change_24h, 2),
                            'signal': signal,
                            'description': desc
                        }
            
            return {'signal': '⚪ N/A', 'description': 'OI data unavailable'}
            
        except Exception as e:
            logger.debug(f"OI analysis error: {e}")
            return {'signal': '⚪ N/A', 'description': 'OI data unavailable'}
    
    async def _analyze_order_book(self, symbol: str, current_price: float) -> Dict:
        """Analyze order book for whale walls and imbalances - ENHANCED"""
        try:
            # Fetch deeper order book (100 levels)
            order_book = await self.exchange.fetch_order_book(symbol, limit=100)
            
            bids = order_book['bids']  # [[price, amount], ...]
            asks = order_book['asks']
            
            if not bids or not asks:
                return {'imbalance': '⚪ N/A'}
            
            # Calculate bid/ask spread
            best_bid = bids[0][0]
            best_ask = asks[0][0]
            spread = ((best_ask - best_bid) / best_bid) * 100
            
            # Analyze multiple depth levels
            def calc_volume_at_depth(orders, price, depth_pct, is_bid=True):
                if is_bid:
                    return sum([o[1] * o[0] for o in orders if o[0] >= price * (1 - depth_pct/100)])
                else:
                    return sum([o[1] * o[0] for o in orders if o[0] <= price * (1 + depth_pct/100)])
            
            # Volume at different depths
            bid_1pct = calc_volume_at_depth(bids, current_price, 1, True)
            ask_1pct = calc_volume_at_depth(asks, current_price, 1, False)
            bid_2pct = calc_volume_at_depth(bids, current_price, 2, True)
            ask_2pct = calc_volume_at_depth(asks, current_price, 2, False)
            bid_5pct = calc_volume_at_depth(bids, current_price, 5, True)
            ask_5pct = calc_volume_at_depth(asks, current_price, 5, False)
            
            # Total book depth
            total_bid = sum([b[1] * b[0] for b in bids])
            total_ask = sum([a[1] * a[0] for a in asks])
            
            # Calculate imbalances at different levels
            def imbalance_ratio(bid_vol, ask_vol):
                total = bid_vol + ask_vol
                return (bid_vol / total * 100) if total > 0 else 50
            
            imb_1pct = imbalance_ratio(bid_1pct, ask_1pct)
            imb_2pct = imbalance_ratio(bid_2pct, ask_2pct)
            imb_5pct = imbalance_ratio(bid_5pct, ask_5pct)
            total_imb = imbalance_ratio(total_bid, total_ask)
            
            # Weighted imbalance (closer levels matter more)
            weighted_imb = (imb_1pct * 0.5 + imb_2pct * 0.3 + imb_5pct * 0.2)
            
            # Find whale walls (orders > 5x average)
            avg_bid_size = total_bid / len(bids) if bids else 0
            avg_ask_size = total_ask / len(asks) if asks else 0
            
            whale_bids = [(b[0], b[1] * b[0]) for b in bids if b[1] * b[0] > avg_bid_size * 5]
            whale_asks = [(a[0], a[1] * a[0]) for a in asks if a[1] * a[0] > avg_ask_size * 5]
            
            whale_bids.sort(key=lambda x: x[1], reverse=True)
            whale_asks.sort(key=lambda x: x[1], reverse=True)
            
            # Determine imbalance with confidence
            if weighted_imb > 70:
                imbalance = "🟢 STRONG BID SUPPORT"
                imbalance_desc = f"Heavy buying pressure"
                confidence = "high"
            elif weighted_imb > 60:
                imbalance = "🟢 BID BIAS"
                imbalance_desc = f"Buyers have edge"
                confidence = "medium"
            elif weighted_imb < 30:
                imbalance = "🔴 HEAVY SELLING"
                imbalance_desc = f"Strong selling pressure"
                confidence = "high"
            elif weighted_imb < 40:
                imbalance = "🔴 ASK BIAS"
                imbalance_desc = f"Sellers have edge"
                confidence = "medium"
            else:
                imbalance = "⚖️ BALANCED"
                imbalance_desc = f"No clear edge"
                confidence = "low"
            
            # Spread analysis
            if spread < 0.02:
                spread_status = "Tight (high liquidity)"
            elif spread < 0.05:
                spread_status = "Normal"
            elif spread < 0.1:
                spread_status = "Wide (low liquidity)"
            else:
                spread_status = "Very wide (caution)"
            
            # Format whale walls
            walls = []
            if whale_bids:
                best_bid_wall = whale_bids[0]
                dist = ((current_price - best_bid_wall[0]) / current_price) * 100
                walls.append(f"🟢 ${best_bid_wall[1]/1000:.0f}K @ ${best_bid_wall[0]:,.4f} ({dist:.1f}% below)")
            if whale_asks:
                best_ask_wall = whale_asks[0]
                dist = ((best_ask_wall[0] - current_price) / current_price) * 100
                walls.append(f"🔴 ${best_ask_wall[1]/1000:.0f}K @ ${best_ask_wall[0]:,.4f} ({dist:.1f}% above)")
            
            return {
                'imbalance': imbalance,
                'imbalance_desc': imbalance_desc,
                'confidence': confidence,
                'bid_pct': round(weighted_imb, 1),
                'ask_pct': round(100 - weighted_imb, 1),
                'spread': round(spread, 4),
                'spread_status': spread_status,
                'depth_1pct': f"{imb_1pct:.0f}%B / {100-imb_1pct:.0f}%A",
                'depth_2pct': f"{imb_2pct:.0f}%B / {100-imb_2pct:.0f}%A",
                'depth_5pct': f"{imb_5pct:.0f}%B / {100-imb_5pct:.0f}%A",
                'whale_walls': walls[:2],
                'total_bid_usdt': round(total_bid, 0),
                'total_ask_usdt': round(total_ask, 0)
            }
            
        except Exception as e:
            logger.debug(f"Order book analysis error: {e}")
            return {'imbalance': '⚪ N/A', 'imbalance_desc': 'Order book unavailable'}
    
    async def _analyze_mtf_trend(self, symbol: str) -> Dict:
        """Multi-timeframe trend analysis - 5m, 15m, 1H, 4H alignment"""
        try:
            # Fetch candles for each timeframe
            candles_5m = await self._fetch_candles_cached(symbol, '5m', limit=50)
            candles_15m = await self._fetch_candles_cached(symbol, '15m', limit=50)
            candles_1h = await self._fetch_candles_cached(symbol, '1h', limit=50)
            candles_4h = await self._fetch_candles_cached(symbol, '4h', limit=30)
            
            def get_trend(candles):
                if len(candles) < 21:
                    return 'N/A', 0
                closes = [c[4] for c in candles]
                ema9 = self._calculate_ema(closes, 9)
                ema21 = self._calculate_ema(closes, 21)
                strength = abs((ema9 - ema21) / ema21 * 100)
                return ('bullish' if ema9 > ema21 else 'bearish'), strength
            
            trend_5m, str_5m = get_trend(candles_5m)
            trend_15m, str_15m = get_trend(candles_15m)
            trend_1h, str_1h = get_trend(candles_1h)
            trend_4h, str_4h = get_trend(candles_4h)
            
            # Create visual
            def emoji(t):
                if t == 'bullish':
                    return '🟢'
                elif t == 'bearish':
                    return '🔴'
                return '⚪'
            
            visual = f"{emoji(trend_5m)} {emoji(trend_15m)} {emoji(trend_1h)} {emoji(trend_4h)}"
            labels = "5m  15m  1H   4H"
            
            # Count alignment
            trends = [trend_5m, trend_15m, trend_1h, trend_4h]
            bullish_count = trends.count('bullish')
            bearish_count = trends.count('bearish')
            
            if bullish_count == 4:
                alignment = "🟢 FULL BULLISH ALIGNMENT"
                strength = "Maximum - all timeframes bullish"
            elif bearish_count == 4:
                alignment = "🔴 FULL BEARISH ALIGNMENT"
                strength = "Maximum - all timeframes bearish"
            elif bullish_count >= 3:
                alignment = "🟢 MOSTLY BULLISH"
                strength = f"{bullish_count}/4 timeframes bullish"
            elif bearish_count >= 3:
                alignment = "🔴 MOSTLY BEARISH"
                strength = f"{bearish_count}/4 timeframes bearish"
            else:
                alignment = "⚠️ MIXED/CHOPPY"
                strength = "No clear trend alignment - caution"
            
            return {
                'visual': visual,
                'labels': labels,
                'alignment': alignment,
                'strength': strength,
                'trend_5m': trend_5m,
                'trend_15m': trend_15m,
                'trend_1h': trend_1h,
                'trend_4h': trend_4h,
                'bullish_count': bullish_count,
                'bearish_count': bearish_count
            }
            
        except Exception as e:
            logger.debug(f"MTF trend error: {e}")
            return {'alignment': '⚪ N/A', 'strength': 'MTF analysis unavailable'}
    
    async def _analyze_session_patterns(self, symbol: str) -> Dict:
        """Analyze historical performance during different trading sessions"""
        try:
            from datetime import timezone
            
            # Fetch 7 days of hourly data
            candles_1h = await self._fetch_candles_cached(symbol, '1h', limit=168)
            
            if len(candles_1h) < 48:
                return {'current_session': '⚪ N/A'}
            
            # Define sessions (UTC times)
            # Asia: 00:00-08:00 UTC (Tokyo/Singapore open)
            # Europe: 08:00-16:00 UTC (London open)
            # US: 13:00-21:00 UTC (NY open, overlaps with EU close)
            
            asia_moves = []
            eu_moves = []
            us_moves = []
            
            for candle in candles_1h:
                timestamp = candle[0] / 1000  # Convert ms to seconds
                dt = datetime.fromtimestamp(timestamp, tz=timezone.utc)
                hour = dt.hour
                
                open_price = candle[1]
                close_price = candle[4]
                move_pct = ((close_price - open_price) / open_price) * 100
                
                # Classify by session
                if 0 <= hour < 8:
                    asia_moves.append(move_pct)
                elif 8 <= hour < 16:
                    eu_moves.append(move_pct)
                elif 13 <= hour < 21:
                    us_moves.append(move_pct)
            
            def session_stats(moves):
                if not moves:
                    return {'avg': 0, 'win_rate': 0, 'volatility': 0}
                avg = sum(moves) / len(moves)
                wins = sum(1 for m in moves if m > 0)
                win_rate = (wins / len(moves)) * 100
                volatility = sum(abs(m) for m in moves) / len(moves)
                return {'avg': avg, 'win_rate': win_rate, 'volatility': volatility}
            
            asia_stats = session_stats(asia_moves)
            eu_stats = session_stats(eu_moves)
            us_stats = session_stats(us_moves)
            
            # Determine current session
            now = datetime.now(timezone.utc)
            current_hour = now.hour
            
            if 0 <= current_hour < 8:
                current_session = "🌏 ASIA"
                current_stats = asia_stats
            elif 8 <= current_hour < 13:
                current_session = "🇪🇺 EUROPE"
                current_stats = eu_stats
            elif 13 <= current_hour < 16:
                current_session = "🔄 EU/US OVERLAP"
                current_stats = {'avg': (eu_stats['avg'] + us_stats['avg']) / 2, 
                                 'win_rate': (eu_stats['win_rate'] + us_stats['win_rate']) / 2,
                                 'volatility': max(eu_stats['volatility'], us_stats['volatility'])}
            elif 16 <= current_hour < 21:
                current_session = "🇺🇸 US"
                current_stats = us_stats
            else:
                current_session = "🌙 OVERNIGHT"
                current_stats = asia_stats
            
            # Find best session for this coin
            sessions = [
                ('🌏 ASIA', asia_stats),
                ('🇪🇺 EUROPE', eu_stats),
                ('🇺🇸 US', us_stats)
            ]
            
            # Best for longs (highest avg positive move)
            best_long = max(sessions, key=lambda x: x[1]['avg'])
            # Best for shorts (lowest/most negative avg move)
            best_short = min(sessions, key=lambda x: x[1]['avg'])
            # Most volatile (best for scalping)
            most_volatile = max(sessions, key=lambda x: x[1]['volatility'])
            
            # Generate insight
            if current_stats['avg'] > 0.05:
                session_bias = f"🟢 Historically bullish during {current_session}"
            elif current_stats['avg'] < -0.05:
                session_bias = f"🔴 Historically bearish during {current_session}"
            else:
                session_bias = f"⚪ No strong bias during {current_session}"
            
            return {
                'current_session': current_session,
                'session_bias': session_bias,
                'current_win_rate': round(current_stats['win_rate'], 1),
                'current_avg_move': round(current_stats['avg'], 3),
                'asia': {'win_rate': round(asia_stats['win_rate'], 1), 'avg': round(asia_stats['avg'], 3)},
                'europe': {'win_rate': round(eu_stats['win_rate'], 1), 'avg': round(eu_stats['avg'], 3)},
                'us': {'win_rate': round(us_stats['win_rate'], 1), 'avg': round(us_stats['avg'], 3)},
                'best_long_session': f"{best_long[0]} ({best_long[1]['avg']:+.3f}% avg)",
                'best_short_session': f"{best_short[0]} ({best_short[1]['avg']:+.3f}% avg)",
                'most_volatile': f"{most_volatile[0]} ({most_volatile[1]['volatility']:.2f}% avg range)"
            }
            
        except Exception as e:
            logger.debug(f"Session pattern error: {e}")
            return {'current_session': '⚪ N/A', 'session_bias': 'Session data unavailable'}
    
    async def _analyze_long_short_ratio(self, symbol: str) -> Dict:
        """Analyze Long/Short ratio from Binance Futures - shows trader positioning"""
        try:
            import httpx
            base_symbol = symbol.replace('/USDT', '').replace(':USDT', '')
            
            async with httpx.AsyncClient() as client:
                # Get global long/short account ratio
                response_global = await client.get(
                    "https://fapi.binance.com/futures/data/globalLongShortAccountRatio",
                    params={'symbol': f'{base_symbol}USDT', 'period': '1h', 'limit': 24},
                    timeout=10
                )
                
                # Get top trader long/short ratio (positions)
                response_top = await client.get(
                    "https://fapi.binance.com/futures/data/topLongShortPositionRatio",
                    params={'symbol': f'{base_symbol}USDT', 'period': '1h', 'limit': 24},
                    timeout=10
                )
                
                # Get taker buy/sell volume
                response_taker = await client.get(
                    "https://fapi.binance.com/futures/data/takerlongshortRatio",
                    params={'symbol': f'{base_symbol}USDT', 'period': '1h', 'limit': 24},
                    timeout=10
                )
                
                result = {}
                
                # Process global ratio
                if response_global.status_code == 200:
                    data = response_global.json()
                    if data:
                        current = data[-1]
                        long_pct = float(current['longAccount']) * 100
                        short_pct = float(current['shortAccount']) * 100
                        ratio = float(current['longShortRatio'])
                        
                        # Calculate change
                        if len(data) >= 2:
                            prev_ratio = float(data[-2]['longShortRatio'])
                            ratio_change = ((ratio - prev_ratio) / prev_ratio) * 100
                        else:
                            ratio_change = 0
                        
                        # Determine sentiment
                        if ratio > 2.0:
                            sentiment = "🔴 EXTREMELY LONG"
                            warning = "Crowded long - high squeeze risk"
                        elif ratio > 1.5:
                            sentiment = "🟡 MOSTLY LONG"
                            warning = "Leaning long - watch for reversal"
                        elif ratio < 0.5:
                            sentiment = "🟢 EXTREMELY SHORT"
                            warning = "Crowded short - squeeze potential"
                        elif ratio < 0.67:
                            sentiment = "🟡 MOSTLY SHORT"
                            warning = "Leaning short - watch for bounce"
                        else:
                            sentiment = "⚖️ BALANCED"
                            warning = "No extreme positioning"
                        
                        result['global'] = {
                            'long_pct': round(long_pct, 1),
                            'short_pct': round(short_pct, 1),
                            'ratio': round(ratio, 2),
                            'ratio_change': round(ratio_change, 1),
                            'sentiment': sentiment,
                            'warning': warning
                        }
                
                # Process top trader ratio
                if response_top.status_code == 200:
                    data = response_top.json()
                    if data:
                        current = data[-1]
                        long_pct = float(current['longAccount']) * 100
                        short_pct = float(current['shortAccount']) * 100
                        ratio = float(current['longShortRatio'])
                        
                        if ratio > 1.2:
                            top_sentiment = "🐋 Whales LONG"
                        elif ratio < 0.8:
                            top_sentiment = "🐋 Whales SHORT"
                        else:
                            top_sentiment = "🐋 Whales neutral"
                        
                        result['top_traders'] = {
                            'long_pct': round(long_pct, 1),
                            'short_pct': round(short_pct, 1),
                            'ratio': round(ratio, 2),
                            'sentiment': top_sentiment
                        }
                
                # Process taker buy/sell
                if response_taker.status_code == 200:
                    data = response_taker.json()
                    if data:
                        current = data[-1]
                        buy_vol = float(current['buyVol'])
                        sell_vol = float(current['sellVol'])
                        ratio = float(current['buySellRatio'])
                        
                        if ratio > 1.3:
                            taker_sentiment = "🟢 Aggressive buying"
                        elif ratio < 0.7:
                            taker_sentiment = "🔴 Aggressive selling"
                        else:
                            taker_sentiment = "⚪ Balanced flow"
                        
                        result['taker'] = {
                            'buy_sell_ratio': round(ratio, 2),
                            'sentiment': taker_sentiment
                        }
                
                if not result:
                    return {'sentiment': '⚪ N/A'}
                
                # Create visual bar for long/short
                if 'global' in result:
                    g = result['global']
                    bar_len = 10
                    long_bars = int((g['long_pct'] / 100) * bar_len)
                    visual_bar = "🟢" * long_bars + "🔴" * (bar_len - long_bars)
                    result['visual_bar'] = visual_bar
                
                return result
                
        except Exception as e:
            logger.debug(f"Long/short ratio error: {e}")
            return {'sentiment': '⚪ N/A'}
    
    async def _analyze_rsi_divergence(self, symbol: str) -> Dict:
        """Detect RSI divergence - powerful reversal signal"""
        try:
            # Fetch 1H candles for better divergence detection
            candles = await self._fetch_candles_cached(symbol, '1h', limit=50)
            
            if len(candles) < 30:
                return {'divergence': '⚪ N/A'}
            
            closes = [c[4] for c in candles]
            highs = [c[2] for c in candles]
            lows = [c[3] for c in candles]
            
            # Calculate RSI
            rsi_values = []
            for i in range(14, len(closes)):
                rsi = self._calculate_rsi(closes[:i+1], 14)
                rsi_values.append(rsi)
            
            if len(rsi_values) < 10:
                return {'divergence': '⚪ N/A'}
            
            # Find recent swing highs and lows in price
            price_swing_highs = []
            price_swing_lows = []
            
            for i in range(2, len(closes) - 2):
                # Swing high: higher than neighbors
                if highs[i] > highs[i-1] and highs[i] > highs[i-2] and highs[i] > highs[i+1] and highs[i] > highs[i+2]:
                    price_swing_highs.append((i, highs[i]))
                # Swing low: lower than neighbors
                if lows[i] < lows[i-1] and lows[i] < lows[i-2] and lows[i] < lows[i+1] and lows[i] < lows[i+2]:
                    price_swing_lows.append((i, lows[i]))
            
            divergence = None
            divergence_type = None
            strength = "weak"
            
            # Check for bearish divergence (price higher high, RSI lower high)
            if len(price_swing_highs) >= 2:
                latest = price_swing_highs[-1]
                prev = price_swing_highs[-2]
                
                # Map to RSI index (offset by 14)
                rsi_idx_latest = latest[0] - 14
                rsi_idx_prev = prev[0] - 14
                
                if 0 <= rsi_idx_latest < len(rsi_values) and 0 <= rsi_idx_prev < len(rsi_values):
                    if latest[1] > prev[1] and rsi_values[rsi_idx_latest] < rsi_values[rsi_idx_prev]:
                        divergence = "🔴 BEARISH DIVERGENCE"
                        divergence_type = "bearish"
                        price_diff = ((latest[1] - prev[1]) / prev[1]) * 100
                        rsi_diff = rsi_values[rsi_idx_prev] - rsi_values[rsi_idx_latest]
                        if price_diff > 2 and rsi_diff > 5:
                            strength = "strong"
                        elif price_diff > 1 or rsi_diff > 3:
                            strength = "moderate"
            
            # Check for bullish divergence (price lower low, RSI higher low)
            if not divergence and len(price_swing_lows) >= 2:
                latest = price_swing_lows[-1]
                prev = price_swing_lows[-2]
                
                rsi_idx_latest = latest[0] - 14
                rsi_idx_prev = prev[0] - 14
                
                if 0 <= rsi_idx_latest < len(rsi_values) and 0 <= rsi_idx_prev < len(rsi_values):
                    if latest[1] < prev[1] and rsi_values[rsi_idx_latest] > rsi_values[rsi_idx_prev]:
                        divergence = "🟢 BULLISH DIVERGENCE"
                        divergence_type = "bullish"
                        price_diff = ((prev[1] - latest[1]) / prev[1]) * 100
                        rsi_diff = rsi_values[rsi_idx_latest] - rsi_values[rsi_idx_prev]
                        if price_diff > 2 and rsi_diff > 5:
                            strength = "strong"
                        elif price_diff > 1 or rsi_diff > 3:
                            strength = "moderate"
            
            if divergence:
                if divergence_type == "bearish":
                    action = "Price made higher high but momentum weakening - potential reversal DOWN"
                else:
                    action = "Price made lower low but momentum building - potential reversal UP"
                
                return {
                    'divergence': divergence,
                    'type': divergence_type,
                    'strength': strength,
                    'action': action,
                    'timeframe': '1H'
                }
            
            return {
                'divergence': '⚪ NO DIVERGENCE',
                'type': None,
                'strength': None,
                'action': 'No RSI divergence detected on 1H'
            }
            
        except Exception as e:
            logger.debug(f"RSI divergence error: {e}")
            return {'divergence': '⚪ N/A'}
    
    def _calculate_conviction_score(self, trend, volume, momentum, spot_flow, 
                                    funding, oi, order_book, ls_ratio, mtf) -> Dict:
        """Calculate overall conviction score combining all factors"""
        try:
            score = 50  # Start neutral
            bullish_factors = []
            bearish_factors = []
            
            # Trend alignment (+/-15)
            if mtf.get('bullish_count', 0) == 4:
                score += 15
                bullish_factors.append("Full MTF alignment")
            elif mtf.get('bullish_count', 0) >= 3:
                score += 10
                bullish_factors.append("3/4 MTF bullish")
            elif mtf.get('bearish_count', 0) == 4:
                score -= 15
                bearish_factors.append("Full MTF bearish")
            elif mtf.get('bearish_count', 0) >= 3:
                score -= 10
                bearish_factors.append("3/4 MTF bearish")
            
            # Volume (+/-5)
            vol_status = volume.get('status', '')
            if vol_status in ['high', 'extreme']:
                score += 5
                bullish_factors.append("High volume")
            elif vol_status == 'low':
                score -= 3
                bearish_factors.append("Low volume")
            
            # Spot flow (+/-10)
            spot_signal = spot_flow.get('signal', '')
            if 'buying' in spot_signal:
                score += 10 if 'strong' in spot_signal else 5
                bullish_factors.append("Institutional buying")
            elif 'selling' in spot_signal:
                score -= 10 if 'strong' in spot_signal else 5
                bearish_factors.append("Institutional selling")
            
            # Funding rate (+/-5)
            funding_rate = funding.get('current_rate', 0)
            if funding_rate > 0.1:
                score -= 5  # Crowded long = bearish
                bearish_factors.append("Crowded longs")
            elif funding_rate < -0.1:
                score += 5  # Crowded short = bullish squeeze
                bullish_factors.append("Short squeeze potential")
            
            # OI signal (+/-5)
            oi_signal = oi.get('signal', '')
            if 'BUILD' in oi_signal:
                score += 5
                bullish_factors.append("OI building")
            elif 'UNWIND' in oi_signal:
                score -= 5
                bearish_factors.append("OI unwinding")
            
            # Order book (+/-8)
            ob_imb = order_book.get('bid_pct', 50)
            if ob_imb > 65:
                score += 8
                bullish_factors.append("Strong bid support")
            elif ob_imb < 35:
                score -= 8
                bearish_factors.append("Heavy ask pressure")
            
            # Long/short ratio (+/-5) - contrarian
            ls_global = ls_ratio.get('global', {})
            ls_ratio_val = ls_global.get('ratio', 1)
            if ls_ratio_val > 2:
                score -= 5  # Too many longs = bearish
                bearish_factors.append("Extreme long positioning")
            elif ls_ratio_val < 0.5:
                score += 5  # Too many shorts = bullish squeeze
                bullish_factors.append("Short squeeze setup")
            
            # Cap score
            score = max(0, min(100, score))
            
            # Determine direction and confidence
            if score >= 70:
                direction = "STRONG BULLISH"
                emoji = "🟢🟢"
                confidence = "high"
            elif score >= 60:
                direction = "BULLISH"
                emoji = "🟢"
                confidence = "medium"
            elif score <= 30:
                direction = "STRONG BEARISH"
                emoji = "🔴🔴"
                confidence = "high"
            elif score <= 40:
                direction = "BEARISH"
                emoji = "🔴"
                confidence = "medium"
            else:
                direction = "NEUTRAL"
                emoji = "⚪"
                confidence = "low"
            
            return {
                'score': score,
                'direction': direction,
                'emoji': emoji,
                'confidence': confidence,
                'bullish_factors': bullish_factors[:3],
                'bearish_factors': bearish_factors[:3]
            }
            
        except Exception as e:
            logger.debug(f"Conviction score error: {e}")
            return {'score': 50, 'direction': 'NEUTRAL', 'emoji': '⚪', 'confidence': 'low'}
    
    async def close(self):
        """Close exchange connection"""
        if self.exchange:
            await self.exchange.close()
