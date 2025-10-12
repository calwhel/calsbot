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
        self.timeframe = settings.TIMEFRAME
        self.ema_fast = settings.EMA_FAST
        self.ema_slow = settings.EMA_SLOW
        self.ema_trend = settings.EMA_TREND
        self.trail_pct = settings.TRAIL_PCT
        self.symbols = [s.strip() for s in settings.SYMBOLS.split(",")]
    
    async def get_ohlcv(self, symbol: str, limit: int = 100) -> pd.DataFrame:
        try:
            ohlcv = await self.exchange.fetch_ohlcv(symbol, self.timeframe, limit=limit)
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
        
        # Volume Confirmation - current volume must be above average
        if current['volume_avg'] == 0:
            return None
        volume_confirmed = current['volume'] > current['volume_avg']
        
        # RSI Filter - avoid overbought (>70) and oversold (<30)
        rsi_ok_for_long = current['rsi'] < 70  # Not overbought
        rsi_ok_for_short = current['rsi'] > 30  # Not oversold
        
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
        
        if bullish_cross:
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
        elif bearish_cross:
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
    
    def calculate_atr_stop_take(self, entry_price: float, direction: str, atr: float, atr_sl_multiplier: float = 2.0, atr_tp_multiplier: float = 3.0) -> Dict:
        """
        ATR-based stop loss and take profit
        - Stop Loss: 2x ATR from entry (adapts to volatility)
        - Take Profit: 3x ATR from entry (1.5:1 risk/reward)
        """
        if direction == 'LONG':
            stop_loss = entry_price - (atr * atr_sl_multiplier)
            take_profit = entry_price + (atr * atr_tp_multiplier)
        else:
            stop_loss = entry_price + (atr * atr_sl_multiplier)
            take_profit = entry_price - (atr * atr_tp_multiplier)
        
        return {
            'stop_loss': round(stop_loss, 8),
            'take_profit': round(take_profit, 8)
        }
    
    async def generate_signal(self, symbol: str) -> Optional[Dict]:
        df = await self.get_ohlcv(symbol)
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
            'timeframe': self.timeframe
        }
    
    async def scan_all_symbols(self) -> List[Dict]:
        signals = []
        for symbol in self.symbols:
            signal = await self.generate_signal(symbol)
            if signal:
                signals.append(signal)
        return signals
    
    async def close(self):
        await self.exchange.close()
