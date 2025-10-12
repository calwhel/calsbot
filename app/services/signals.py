import ccxt
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
    
    def get_ohlcv(self, symbol: str, limit: int = 100) -> pd.DataFrame:
        try:
            ohlcv = self.exchange.fetch_ohlcv(symbol, self.timeframe, limit=limit)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            return df
        except Exception as e:
            print(f"Error fetching OHLCV for {symbol}: {e}")
            return pd.DataFrame()
    
    def calculate_ema(self, df: pd.DataFrame) -> pd.DataFrame:
        df['ema_fast'] = ta.trend.EMAIndicator(df['close'], window=self.ema_fast).ema_indicator()
        df['ema_slow'] = ta.trend.EMAIndicator(df['close'], window=self.ema_slow).ema_indicator()
        df['ema_trend'] = ta.trend.EMAIndicator(df['close'], window=self.ema_trend).ema_indicator()
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
        
        bullish_cross = (
            previous['ema_fast'] <= previous['ema_slow'] and
            current['ema_fast'] > current['ema_slow'] and
            current['close'] > current['ema_trend']
        )
        
        bearish_cross = (
            previous['ema_fast'] >= previous['ema_slow'] and
            current['ema_fast'] < current['ema_slow'] and
            current['close'] < current['ema_trend']
        )
        
        if bullish_cross:
            return {
                'direction': 'LONG',
                'entry_price': current['close'],
                'ema_fast': current['ema_fast'],
                'ema_slow': current['ema_slow'],
                'ema_trend': current['ema_trend']
            }
        elif bearish_cross:
            return {
                'direction': 'SHORT',
                'entry_price': current['close'],
                'ema_fast': current['ema_fast'],
                'ema_slow': current['ema_slow'],
                'ema_trend': current['ema_trend']
            }
        
        return None
    
    def calculate_stop_take(self, entry_price: float, direction: str, support: float, resistance: float) -> Dict:
        if direction == 'LONG':
            stop_loss = support
            take_profit = entry_price + (entry_price - support) * 2
        else:
            stop_loss = resistance
            take_profit = entry_price - (resistance - entry_price) * 2
        
        return {
            'stop_loss': round(stop_loss, 8),
            'take_profit': round(take_profit, 8)
        }
    
    def generate_signal(self, symbol: str) -> Optional[Dict]:
        df = self.get_ohlcv(symbol)
        if df.empty:
            return None
        
        df = self.calculate_ema(df)
        cross = self.check_ema_cross(df)
        
        if not cross:
            return None
        
        sr_levels = self.find_support_resistance(df)
        stop_take = self.calculate_stop_take(
            cross['entry_price'],
            cross['direction'],
            sr_levels['support'],
            sr_levels['resistance']
        )
        
        return {
            'symbol': symbol,
            'direction': cross['direction'],
            'entry_price': round(cross['entry_price'], 8),
            'stop_loss': stop_take['stop_loss'],
            'take_profit': stop_take['take_profit'],
            'support_level': round(sr_levels['support'], 8),
            'resistance_level': round(sr_levels['resistance'], 8),
            'ema_fast': round(cross['ema_fast'], 8),
            'ema_slow': round(cross['ema_slow'], 8),
            'ema_trend': round(cross['ema_trend'], 8),
            'timeframe': self.timeframe,
            'timestamp': datetime.utcnow()
        }
    
    def scan_all_symbols(self) -> List[Dict]:
        signals = []
        for symbol in self.symbols:
            signal = self.generate_signal(symbol)
            if signal:
                signals.append(signal)
        return signals
