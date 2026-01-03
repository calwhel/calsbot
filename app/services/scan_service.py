"""
On-demand coin analysis service - provides market intelligence without generating signals
"""
import logging
from datetime import datetime, timedelta
from typing import Dict, Optional
import ccxt.async_support as ccxt
from app.services.spot_monitor import SpotMarketMonitor

logger = logging.getLogger(__name__)


class CoinScanService:
    """Provides on-demand analysis of cryptocurrency price action"""
    
    def __init__(self):
        self.exchange = None
        self.spot_monitor = SpotMarketMonitor()
    
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
            ticker = await self.exchange.fetch_ticker(symbol)
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
            
            # Generate SHORT day trade idea
            trade_idea = await self._generate_trade_idea(
                symbol,
                trend_analysis,
                volume_analysis,
                momentum_analysis,
                spot_flow_analysis,
                current_price
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
            candles_5m = await self.exchange.fetch_ohlcv(symbol, '5m', limit=100)
            # Get 15m candles
            candles_15m = await self.exchange.fetch_ohlcv(symbol, '15m', limit=100)
            # Get 1H candles for support/resistance
            candles_1h = await self.exchange.fetch_ohlcv(symbol, '1h', limit=50)
            
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
            candles = await self.exchange.fetch_ohlcv(symbol, '5m', limit=50)
            
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
            candles = await self.exchange.fetch_ohlcv(symbol, '5m', limit=100)
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
            candles_15m = await self.exchange.fetch_ohlcv(symbol, '15m', limit=20)
            candles_1h = await self.exchange.fetch_ohlcv(symbol, '1h', limit=20)
            
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
            btc_candles = await self.exchange.fetch_ohlcv('BTC/USDT', '15m', limit=30)
            symbol_candles = await self.exchange.fetch_ohlcv(symbol, '15m', limit=30)
            
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
    
    async def _generate_trade_idea(self, symbol: str, trend: Dict, volume: Dict, momentum: Dict, spot_flow: Dict, current_price: float) -> Dict:
        """
        Generate detailed LONG and SHORT day trade ideas for major alts.
        Analyzes multiple factors and returns the better setup.
        """
        try:
            # Fetch additional data for trade idea
            candles_15m = await self.exchange.fetch_ohlcv(symbol, '15m', limit=50)
            candles_1h = await self.exchange.fetch_ohlcv(symbol, '1h', limit=30)
            candles_4h = await self.exchange.fetch_ohlcv(symbol, '4h', limit=20)
            
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
            
            # Overbought RSI
            if rsi_15m > 70:
                short_score += 3
                short_signals.append(f"RSI(15m) overbought ({rsi_15m:.0f})")
            elif rsi_15m > 65:
                short_score += 1.5
                short_signals.append(f"RSI(15m) elevated ({rsi_15m:.0f})")
            
            if rsi_1h > 70:
                short_score += 2
                short_signals.append(f"RSI(1h) overbought ({rsi_1h:.0f})")
            
            # Extended from EMAs
            if extension_1h > 4:
                short_score += 2
                short_signals.append(f"Extended {extension_1h:.1f}% above 1h EMA21")
            
            if extension_4h > 6:
                short_score += 2
                short_signals.append(f"Extended {extension_4h:.1f}% above 4h EMA21")
            
            # Near 24h high (resistance)
            if dist_from_high < 1.5:
                short_score += 2
                short_signals.append(f"At 24h high resistance ({dist_from_high:.1f}% away)")
            
            # Bearish trend
            if trend.get('timeframe_15m') == 'bearish':
                short_score += 1
                short_signals.append("15m trend bearish")
            
            # Selling pressure
            if spot_flow.get('signal') in ['strong_selling', 'moderate_selling']:
                short_score += 2
                short_signals.append(f"Institutional {spot_flow.get('signal').replace('_', ' ')}")
            
            # Volume at highs (distribution)
            if volume.get('status') in ['high', 'extreme'] and dist_from_high < 2:
                short_score += 1.5
                short_signals.append("High volume at highs (distribution)")
            
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
                
                # LONG trade levels - Use recent swing low for tighter SL
                entry = current_price
                # SL below recent swing low (max 3% away)
                stop_loss = max(swing_low * 0.997, entry * 0.97)  # Tighter: 3% max SL
                sl_distance = ((entry - stop_loss) / entry) * 100
                tp1_target = entry * 1.025  # +2.5%
                tp2_target = entry * 1.05   # +5%
                tp1_profit = 2.5
                tp2_profit = 5.0
            else:
                direction = 'SHORT'
                score = short_score
                signals = short_signals
                
                # SHORT trade levels - Use recent swing high for tighter SL
                entry = current_price
                # SL above recent swing high (max 3% away)
                stop_loss = min(swing_high * 1.003, entry * 1.03)  # Tighter: 3% max SL
                sl_distance = ((stop_loss - entry) / entry) * 100
                tp1_target = ema21_1h if ema21_1h < current_price else entry * 0.975
                tp2_target = entry * 0.95
                tp1_profit = ((entry - tp1_target) / entry) * 100
                tp2_profit = 5.0
            
            # R:R calculation
            rr_ratio = tp1_profit / sl_distance if sl_distance > 0 else 0
            
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
                'recommendation': recommendation,
                'reasoning': reasoning,
                'signals': signals,
                'long_score': round(long_score, 1),
                'short_score': round(short_score, 1)
            }
            
        except Exception as e:
            logger.error(f"Error generating trade idea: {e}", exc_info=True)
            return {'error': str(e)}
    
    async def close(self):
        """Close exchange connection"""
        if self.exchange:
            await self.exchange.close()
