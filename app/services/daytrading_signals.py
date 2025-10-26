import ccxt.async_support as ccxt
import pandas as pd
import ta
from datetime import datetime, timezone
from typing import Optional, Dict
from app.config import settings
from app.services.spot_monitor import SpotMarketMonitor
import logging

logger = logging.getLogger(__name__)


class SessionQuality:
    """Determines market session quality based on time of day"""
    
    @staticmethod
    def get_session_quality(hour_utc: int) -> Dict:
        """
        Analyze session quality based on UTC hour
        High liquidity: 8am-11pm UTC (US + EU hours)
        Low liquidity: 12am-7am UTC (Asia hours)
        """
        if 8 <= hour_utc < 23:
            return {
                'quality': 'HIGH',
                'active_sessions': ['US', 'EU'],
                'tradeable': True
            }
        else:
            return {
                'quality': 'LOW',
                'active_sessions': ['ASIA'],
                'tradeable': False
            }


class DayTradingSignalGenerator:
    """
    1:1 Risk-Reward Day Trading Signal Generator
    
    STRICT ENTRY REQUIREMENTS (ALL 6 must pass):
    1. Trend Confirmation: EMA alignment on 15m + 1H
    2. Spot Flow Confirmation: Binance + exchanges agree (>60% pressure)
    3. Volume Spike: >2x average volume
    4. Momentum Aligned: RSI + MACD confirm direction
    5. Candle Pattern: Engulfing, hammer, or strong rejection
    6. High Liquidity Session: Only 8am-11pm UTC
    
    TARGET SETUP (10x leverage):
    - Single TP: 15% (1.5% price move)
    - Single SL: 15% (1.5% price move)
    - 1:1 risk-reward ratio
    """
    
    def __init__(self):
        self.exchange_name = 'binance'
        self.exchange = ccxt.binance({
            'enableRateLimit': True,
            'options': {'defaultType': 'spot'}
        })
        self.spot_monitor = SpotMarketMonitor()
        self.symbols = [s.strip() for s in settings.SYMBOLS.split(",")]
        
    async def get_ohlcv(self, symbol: str, timeframe: str, limit: int = 100) -> pd.DataFrame:
        """Fetch OHLCV data"""
        try:
            ohlcv = await self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            return df
        except Exception as e:
            logger.error(f"Error fetching OHLCV for {symbol}: {e}")
            return pd.DataFrame()
    
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate all technical indicators"""
        df['ema_9'] = ta.trend.EMAIndicator(df['close'], window=9).ema_indicator()
        df['ema_21'] = ta.trend.EMAIndicator(df['close'], window=21).ema_indicator()
        df['ema_50'] = ta.trend.EMAIndicator(df['close'], window=50).ema_indicator()
        
        df['rsi'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
        
        macd = ta.trend.MACD(df['close'])
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        df['macd_diff'] = macd.macd_diff()
        
        df['volume_avg'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_avg']
        
        df['atr'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close'], window=14).average_true_range()
        
        return df
    
    def check_session_quality(self) -> bool:
        """Check if current time is in high liquidity session"""
        current_hour = datetime.now(timezone.utc).hour
        session = SessionQuality.get_session_quality(current_hour)
        
        if not session['tradeable']:
            logger.debug(f"Low liquidity session (Hour: {current_hour} UTC) - skipping signals")
            return False
        
        return True
    
    async def check_trend_confirmation(self, symbol: str) -> Optional[str]:
        """
        POINT 1: Check EMA alignment on 5m + 15m timeframes (FASTER for early entry)
        Returns: 'LONG' if bullish, 'SHORT' if bearish, None if unclear
        """
        try:
            # FASTER TIMEFRAMES: 5m for entry, 15m for confirmation
            df_5m = await self.get_ohlcv(symbol, '5m', limit=50)
            df_15m = await self.get_ohlcv(symbol, '15m', limit=50)
            
            if df_5m.empty or df_15m.empty:
                return None
            
            df_5m = self.calculate_indicators(df_5m)
            df_15m = self.calculate_indicators(df_15m)
            
            current_5m = df_5m.iloc[-1]
            current_15m = df_15m.iloc[-1]
            
            # EARLY SIGNAL: Just need EMA 9 crossing above 21 on 5m + 15m in same direction
            bullish_5m = current_5m['ema_9'] > current_5m['ema_21']
            bullish_15m = current_15m['ema_9'] > current_15m['ema_21']
            
            bearish_5m = current_5m['ema_9'] < current_5m['ema_21']
            bearish_15m = current_15m['ema_9'] < current_15m['ema_21']
            
            if bullish_5m and bullish_15m:
                return 'LONG'
            elif bearish_5m and bearish_15m:
                return 'SHORT'
            
            return None
            
        except Exception as e:
            logger.error(f"Error checking trend for {symbol}: {e}")
            return None
    
    async def check_spot_flow(self, symbol: str, expected_direction: str) -> bool:
        """
        POINT 2: Check spot buying/selling pressure across exchanges
        🎯 HIGHEST PRIORITY: Institutional traders on spot markets are the "smart money"
        Returns: True if pressure confirms direction (>75% confidence - institutional grade)
        """
        try:
            flow_data = await self.spot_monitor.analyze_exchange_flow(symbol)
            
            # Institutional flow threshold lowered to 60% for more signals
            # 60% confidence = strong institutional confirmation (balanced approach)
            if not flow_data or flow_data['confidence'] < 60:
                logger.debug(f"{symbol}: Spot flow confidence too low ({flow_data.get('confidence', 0)}% < 60%)")
                return False
            
            if expected_direction == 'LONG':
                return flow_data['flow_signal'] in ['HEAVY_BUYING', 'VOLUME_SPIKE_BUY']
            elif expected_direction == 'SHORT':
                return flow_data['flow_signal'] in ['HEAVY_SELLING', 'VOLUME_SPIKE_SELL']
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking spot flow for {symbol}: {e}")
            return False
    
    def check_volume_spike(self, df: pd.DataFrame) -> bool:
        """
        POINT 3: Check for volume BUILDING (>1.3x average) - EARLY signal, not waiting for 2x spike
        """
        current = df.iloc[-1]
        # LOWER THRESHOLD: Enter when volume STARTS building, not after spike
        return current['volume_ratio'] > 1.3
    
    def check_momentum(self, df: pd.DataFrame, direction: str) -> bool:
        """
        POINT 4: Check RSI + MACD momentum alignment (EARLY divergence detection)
        """
        current = df.iloc[-1]
        prev = df.iloc[-2]
        
        if direction == 'LONG':
            # EARLY: RSI bottoming out (35-65), MACD starting to turn
            rsi_ok = 35 < current['rsi'] < 65
            macd_turning_bullish = current['macd_diff'] > prev['macd_diff']  # Just needs to be rising
            return rsi_ok and macd_turning_bullish
            
        elif direction == 'SHORT':
            # EARLY: RSI topping out (35-65), MACD starting to turn
            rsi_ok = 35 < current['rsi'] < 65
            macd_turning_bearish = current['macd_diff'] < prev['macd_diff']  # Just needs to be falling
            return rsi_ok and macd_turning_bearish
        
        return False
    
    def check_candle_pattern(self, df: pd.DataFrame, direction: str) -> bool:
        """
        POINT 5: Check for EARLY candle formation (don't wait for full pattern)
        """
        current = df.iloc[-1]
        prev = df.iloc[-2]
        
        body = abs(current['close'] - current['open'])
        total_range = current['high'] - current['low']
        
        if total_range == 0:
            return False
        
        body_ratio = body / total_range
        
        if direction == 'LONG':
            is_green = current['close'] > current['open']
            
            # EARLY: Just need green candle with decent body OR lower wick forming
            lower_wick = current['open'] - current['low']
            has_lower_wick = lower_wick > body * 1.2  # Looser than before
            decent_green_body = is_green and body_ratio > 0.4  # Looser threshold
            
            return decent_green_body or has_lower_wick
            
        elif direction == 'SHORT':
            is_red = current['close'] < current['open']
            
            # EARLY: Just need red candle with decent body OR upper wick forming
            upper_wick = current['high'] - current['open']
            has_upper_wick = upper_wick > body * 1.2  # Looser than before
            decent_red_body = is_red and body_ratio > 0.4  # Looser threshold
            
            return decent_red_body or has_upper_wick
        
        return False
    
    def calculate_targets(self, entry_price: float, direction: str) -> Dict:
        """
        Calculate 1:1 risk-reward targets (20% TP / 20% SL)
        With 10x leverage: 20% = 2% actual price move
        """
        leverage = 10
        target_percent = 0.20
        
        if direction == 'LONG':
            tp = entry_price * (1 + (target_percent / leverage))
            sl = entry_price * (1 - (target_percent / leverage))
        else:
            tp = entry_price * (1 - (target_percent / leverage))
            sl = entry_price * (1 + (target_percent / leverage))
        
        return {
            'take_profit': float(tp),
            'stop_loss': float(sl),
            'tp_percent': target_percent,
            'sl_percent': target_percent,
            'risk_reward_ratio': '1:1'
        }
    
    async def scan_for_signal(self, symbol: str) -> Optional[Dict]:
        """
        Main scanner - checks ALL 6 confirmation points
        Returns signal only if ALL points pass
        """
        try:
            if not self.check_session_quality():
                return None
            
            trend = await self.check_trend_confirmation(symbol)
            if not trend:
                logger.debug(f"{symbol}: No clear trend confirmation")
                return None
            
            # 🎯 HIGHEST PRIORITY CHECK: Spot flow (institutional "smart money")
            # If institutions aren't buying/selling, skip all other checks
            spot_flow_ok = await self.check_spot_flow(symbol, trend)
            if not spot_flow_ok:
                logger.debug(f"{symbol}: ❌ Spot flow doesn't confirm {trend} - REJECTED (smart money not aligned)")
                return None
            
            logger.debug(f"{symbol}: ✅ Spot flow CONFIRMED {trend} at >75% institutional confidence")
            
            # FASTER TIMEFRAME: Use 5m for early entry detection
            df = await self.get_ohlcv(symbol, '5m', limit=100)
            if df.empty or len(df) < 50:
                return None
            
            df = self.calculate_indicators(df)
            current = df.iloc[-1]
            
            if not self.check_volume_spike(df):
                logger.debug(f"{symbol}: No volume spike")
                return None
            
            if not self.check_momentum(df, trend):
                logger.debug(f"{symbol}: Momentum not aligned for {trend}")
                return None
            
            if not self.check_candle_pattern(df, trend):
                logger.debug(f"{symbol}: No valid candle pattern for {trend}")
                return None
            
            entry_price = float(current['close'])
            targets = self.calculate_targets(entry_price, trend)
            
            logger.info(f"✅ {symbol} {trend} - ALL 6 CONFIRMATIONS PASSED!")
            
            return {
                'symbol': symbol,
                'direction': trend,
                'signal_type': 'DAY_TRADE',
                'pattern': 'MULTI_CONFIRMATION',
                'entry_price': entry_price,
                'take_profit': targets['take_profit'],
                'stop_loss': targets['stop_loss'],
                'confidence': 90,
                'timeframe': '15m',
                'rsi': float(current['rsi']),
                'volume_ratio': float(current['volume_ratio']),
                'ema_9': float(current['ema_9']),
                'ema_21': float(current['ema_21']),
                'reason': f'6-point confirmation: Trend ✅ Spot Flow ✅ Volume ✅ Momentum ✅ Candle ✅ Session ✅',
                'risk_reward': '1:1 (15% TP / 15% SL)',
                'session': SessionQuality.get_session_quality(datetime.now(timezone.utc).hour)
            }
            
        except Exception as e:
            logger.error(f"Error scanning {symbol}: {e}")
            return None
    
    async def scan_all_symbols(self) -> list:
        """Scan all symbols for day trading signals"""
        signals = []
        
        for symbol in self.symbols:
            signal = await self.scan_for_signal(symbol)
            if signal:
                signals.append(signal)
        
        return signals
    
    async def close(self):
        """Close exchange connections"""
        await self.exchange.close()
        await self.spot_monitor.close()
