import ccxt.async_support as ccxt
import pandas as pd
import ta
from datetime import datetime
from typing import List, Dict, Optional
from app.config import settings


class SignalGenerator:
    def __init__(self):
        self.exchange_name = settings.EXCHANGE
        self.exchange = getattr(ccxt, self.exchange_name)()
        self.timeframes = ['1h', '4h']  # Multi-timeframe analysis
        self.ema_fast = settings.EMA_FAST
        self.ema_slow = settings.EMA_SLOW
        self.ema_trend = settings.EMA_TREND
        self.trail_pct = settings.TRAIL_PCT
        self.symbols = [s.strip() for s in settings.SYMBOLS.split(",")]
    
    async def get_ohlcv(self, symbol: str, timeframe: str, limit: int = 100) -> pd.DataFrame:
        try:
            ohlcv = await self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            return df
        except Exception as e:
            print(f"Error fetching OHLCV for {symbol}: {e}")
            return pd.DataFrame()
    
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        # EMA Indicators
        df['ema_fast'] = ta.trend.EMAIndicator(df['close'], window=self.ema_fast).ema_indicator()
        df['ema_slow'] = ta.trend.EMAIndicator(df['close'], window=self.ema_slow).ema_indicator()
        df['ema_trend'] = ta.trend.EMAIndicator(df['close'], window=self.ema_trend).ema_indicator()
        
        # RSI Indicator (14-period default)
        df['rsi'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
        
        # ATR Indicator (14-period default for volatility)
        df['atr'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close'], window=14).average_true_range()
        
        # Volume Average (20-period)
        df['volume_avg'] = df['volume'].rolling(window=20).mean()
        
        return df
    
    def find_support_resistance(self, df: pd.DataFrame, lookback: int = 20) -> Dict[str, float]:
        recent_df = df.tail(lookback)
        support = recent_df['low'].min()
        resistance = recent_df['high'].max()
        return {'support': support, 'resistance': resistance}
    
    def check_ema_cross(self, df: pd.DataFrame) -> Optional[Dict]:
        if len(df) < 2:
            return None
        
        current = df.iloc[-1]
        previous = df.iloc[-2]
        
        # Guard against NaN values - all indicators must be valid
        required_fields = ['volume_avg', 'rsi', 'atr', 'ema_fast', 'ema_slow', 'ema_trend']
        if any(pd.isna(current[field]) for field in required_fields):
            return None
        
        # Volume Confirmation - relaxed to 80% of average (was 100%)
        if current['volume_avg'] == 0:
            return None
        volume_confirmed = current['volume'] >= (current['volume_avg'] * 0.8)
        
        # RSI Filter - avoid extreme overbought (>75) and oversold (<25)
        rsi_ok_for_long = current['rsi'] < 75  # Relaxed from 70
        rsi_ok_for_short = current['rsi'] > 25  # Relaxed from 30
        
        # Classic EMA crossover signals
        bullish_cross = (
            previous['ema_fast'] <= previous['ema_slow'] and
            current['ema_fast'] > current['ema_slow'] and
            current['close'] > current['ema_trend'] and
            volume_confirmed and
            rsi_ok_for_long
        )
        
        bearish_cross = (
            previous['ema_fast'] >= previous['ema_slow'] and
            current['ema_fast'] < current['ema_slow'] and
            current['close'] < current['ema_trend'] and
            volume_confirmed and
            rsi_ok_for_short
        )
        
        # NEW: Trend-following signals (no crossover needed for strong trends)
        ema_separation = abs(current['ema_fast'] - current['ema_slow']) / current['close'] * 100
        strong_trend_threshold = 0.5  # 0.5% EMA separation indicates strong trend
        
        bullish_trend = (
            current['ema_fast'] > current['ema_slow'] and
            current['close'] > current['ema_fast'] and
            current['close'] > current['ema_trend'] and
            ema_separation > strong_trend_threshold and
            volume_confirmed and
            rsi_ok_for_long and
            current['rsi'] > 50  # Bullish momentum
        )
        
        bearish_trend = (
            current['ema_fast'] < current['ema_slow'] and
            current['close'] < current['ema_fast'] and
            current['close'] < current['ema_trend'] and
            ema_separation > strong_trend_threshold and
            volume_confirmed and
            rsi_ok_for_short and
            current['rsi'] < 50  # Bearish momentum
        )
        
        if bullish_cross or bullish_trend:
            return {
                'direction': 'LONG',
                'entry_price': current['close'],
                'ema_fast': current['ema_fast'],
                'ema_slow': current['ema_slow'],
                'ema_trend': current['ema_trend'],
                'rsi': current['rsi'],
                'atr': current['atr'],
                'volume': current['volume'],
                'volume_avg': current['volume_avg']
            }
        elif bearish_cross or bearish_trend:
            return {
                'direction': 'SHORT',
                'entry_price': current['close'],
                'ema_fast': current['ema_fast'],
                'ema_slow': current['ema_slow'],
                'ema_trend': current['ema_trend'],
                'rsi': current['rsi'],
                'atr': current['atr'],
                'volume': current['volume'],
                'volume_avg': current['volume_avg']
            }
        
        return None
    
    def calculate_atr_stop_take(self, entry_price: float, direction: str, atr: float, atr_sl_multiplier: float = 2.0) -> Dict:
        """
        ATR-based stop loss and 3 take profit levels (REALISTIC)
        - Stop Loss: 2x ATR from entry (adapts to volatility)
        - TP1: 1x risk (30% close) - Quick profit
        - TP2: 1.5x risk (30% close) - Good profit
        - TP3: 2x risk (40% close) - Maximum profit
        """
        if direction == 'LONG':
            stop_loss = entry_price - (atr * atr_sl_multiplier)
            risk_amount = atr * atr_sl_multiplier
            
            # Calculate 3 TP levels based on risk multiples (REDUCED for realistic targets)
            take_profit_1 = entry_price + (risk_amount * 1.0)  # 1R
            take_profit_2 = entry_price + (risk_amount * 1.5)  # 1.5R
            take_profit_3 = entry_price + (risk_amount * 2.0)  # 2R
        else:
            stop_loss = entry_price + (atr * atr_sl_multiplier)
            risk_amount = atr * atr_sl_multiplier
            
            # Calculate 3 TP levels based on risk multiples (SHORT)
            take_profit_1 = entry_price - (risk_amount * 1.0)  # 1R
            take_profit_2 = entry_price - (risk_amount * 1.5)  # 1.5R
            take_profit_3 = entry_price - (risk_amount * 2.0)  # 2R
        
        return {
            'stop_loss': round(stop_loss, 8),
            'take_profit_1': round(take_profit_1, 8),
            'take_profit_2': round(take_profit_2, 8),
            'take_profit_3': round(take_profit_3, 8),
            # Keep backward compatibility
            'take_profit': round(take_profit_3, 8)
        }
    
    def assess_risk(self, entry_price: float, stop_loss: float, take_profit: float, atr: float, rsi: float) -> str:
        """
        Assess signal risk based on volatility, RSI, and risk/reward ratio
        Returns: 'LOW', 'MEDIUM', or 'HIGH'
        """
        # Calculate ATR as percentage of price (volatility measure)
        atr_pct = (atr / entry_price) * 100
        
        # Calculate risk/reward ratio
        risk = abs(entry_price - stop_loss)
        reward = abs(take_profit - entry_price)
        rr_ratio = reward / risk if risk > 0 else 0
        
        # Risk scoring
        risk_score = 0
        
        # Volatility check (relaxed thresholds)
        if atr_pct > 6:
            risk_score += 2  # Very high volatility
        elif atr_pct > 3:
            risk_score += 1  # High volatility
        
        # RSI extremes check (more lenient)
        if rsi > 75 or rsi < 25:
            risk_score += 1  # Extreme RSI only
        
        # Risk/Reward check (accept lower RR)
        if rr_ratio < 1.2:
            risk_score += 2  # Very poor risk/reward
        elif rr_ratio < 1.5:
            risk_score += 1  # Low risk/reward
        
        # Classify risk (only reject if score >= 4)
        if risk_score >= 4:
            return 'HIGH'
        elif risk_score >= 2:
            return 'MEDIUM'
        else:
            return 'LOW'
    
    async def generate_signal(self, symbol: str, timeframe: str) -> Optional[Dict]:
        df = await self.get_ohlcv(symbol, timeframe)
        if df.empty:
            return None
        
        df = self.calculate_indicators(df)
        cross = self.check_ema_cross(df)
        
        if not cross:
            return None
        
        sr_levels = self.find_support_resistance(df)
        
        # Use ATR-based stop loss and take profit
        stop_take = self.calculate_atr_stop_take(
            cross['entry_price'],
            cross['direction'],
            cross['atr']
        )
        
        # Assess risk level
        risk_level = self.assess_risk(
            cross['entry_price'],
            stop_take['stop_loss'],
            stop_take['take_profit'],
            cross['atr'],
            cross['rsi']
        )
        
        # Only return medium and low risk signals
        if risk_level == 'HIGH':
            return None
        
        return {
            'symbol': symbol,
            'direction': cross['direction'],
            'entry_price': float(round(cross['entry_price'], 8)),
            'stop_loss': float(stop_take['stop_loss']),
            'take_profit': float(stop_take['take_profit']),
            'support_level': float(round(sr_levels['support'], 8)),
            'resistance_level': float(round(sr_levels['resistance'], 8)),
            'ema_fast': float(round(cross['ema_fast'], 8)),
            'ema_slow': float(round(cross['ema_slow'], 8)),
            'ema_trend': float(round(cross['ema_trend'], 8)),
            'rsi': float(round(cross['rsi'], 2)),
            'atr': float(round(cross['atr'], 8)),
            'volume': float(round(cross['volume'], 2)),
            'volume_avg': float(round(cross['volume_avg'], 2)),
            'timeframe': timeframe,
            'risk_level': risk_level
        }
    
    async def scan_all_symbols(self) -> List[Dict]:
        signals = []
        # Scan all symbols across all timeframes (1h and 4h)
        for timeframe in self.timeframes:
            for symbol in self.symbols:
                signal = await self.generate_signal(symbol, timeframe)
                if signal:
                    signals.append(signal)
        return signals
    
    async def close(self):
        await self.exchange.close()
