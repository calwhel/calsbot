"""
On-demand coin analysis service - provides market intelligence without generating signals
"""
import logging
from datetime import datetime, timedelta
from typing import Dict, Optional
import ccxt.async_support as ccxt
from app.services.spot_monitor import SpotMonitor

logger = logging.getLogger(__name__)


class CoinScanService:
    """Provides on-demand analysis of cryptocurrency price action"""
    
    def __init__(self):
        self.exchange = None
        self.spot_monitor = SpotMonitor()
    
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
            
            # Calculate overall bias
            overall_bias = self._calculate_bias(
                trend_analysis,
                volume_analysis,
                momentum_analysis,
                spot_flow_analysis
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
                'overall_bias': overall_bias,
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
        """Analyze trend using EMA 9/21 on 5m and 15m timeframes"""
        try:
            # Get 5m candles
            candles_5m = await self.exchange.fetch_ohlcv(symbol, '5m', limit=50)
            # Get 15m candles
            candles_15m = await self.exchange.fetch_ohlcv(symbol, '15m', limit=50)
            
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
            
            return {
                'timeframe_5m': trend_5m,
                'timeframe_15m': trend_15m,
                'strength_5m': round(strength_5m, 2),
                'strength_15m': round(strength_15m, 2),
                'aligned': aligned,
                'current_ema9_5m': round(ema9_5m, 8),
                'current_ema21_5m': round(ema21_5m, 8)
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
            flow_data = await self.spot_monitor.get_spot_flow(symbol.replace('/USDT', ''))
            
            if flow_data:
                buy_pressure = flow_data.get('buy_percentage', 0)
                sell_pressure = 100 - buy_pressure
                
                # Determine flow strength
                if buy_pressure >= 75:
                    signal = "strong_buying"
                    confidence = "institutional"
                elif buy_pressure >= 60:
                    signal = "moderate_buying"
                    confidence = "high"
                elif buy_pressure <= 25:
                    signal = "strong_selling"
                    confidence = "institutional"
                elif buy_pressure <= 40:
                    signal = "moderate_selling"
                    confidence = "high"
                else:
                    signal = "neutral"
                    confidence = "low"
                
                return {
                    'buy_pressure': round(buy_pressure, 1),
                    'sell_pressure': round(sell_pressure, 1),
                    'signal': signal,
                    'confidence': confidence,
                    'exchanges_analyzed': len(flow_data.get('exchanges', []))
                }
            else:
                return {'error': 'No spot flow data available'}
                
        except Exception as e:
            logger.error(f"Error analyzing spot flow: {e}")
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
    
    async def close(self):
        """Close exchange connection"""
        if self.exchange:
            await self.exchange.close()
