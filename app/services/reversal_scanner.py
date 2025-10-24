import ccxt.async_support as ccxt
import pandas as pd
import ta
from typing import Optional, Dict, List
from app.config import settings
import logging

logger = logging.getLogger(__name__)


class ReversalScanner:
    """
    Multi-Pattern Reversal Bounce Catcher
    Scans for 5 types of reversal patterns to catch breakouts early:
    1. Support/Resistance Bounce
    2. Bollinger Band Squeeze Breakout
    3. Double Bottom/Top Reversal
    4. RSI Divergence Reversal
    5. Volume Spike Reversal
    """
    
    def __init__(self):
        self.exchange_name = settings.EXCHANGE
        self.exchange = getattr(ccxt, self.exchange_name)()
        self.symbols = [s.strip() for s in settings.SYMBOLS.split(",")]
    
    async def get_ohlcv(self, symbol: str, timeframe: str, limit: int = 100) -> pd.DataFrame:
        try:
            ohlcv = await self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            return df
        except Exception as e:
            logger.error(f"Error fetching OHLCV for {symbol}: {e}")
            return pd.DataFrame()
    
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate all indicators needed for reversal patterns"""
        # Bollinger Bands
        bollinger = ta.volatility.BollingerBands(df['close'], window=20, window_dev=2)
        df['bb_upper'] = bollinger.bollinger_hband()
        df['bb_lower'] = bollinger.bollinger_lband()
        df['bb_middle'] = bollinger.bollinger_mavg()
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle'] * 100
        
        # RSI
        df['rsi'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
        
        # ATR for stop loss
        df['atr'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close'], window=14).average_true_range()
        
        # Volume
        df['volume_avg'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_avg']
        
        return df
    
    def detect_support_resistance_bounce(self, df: pd.DataFrame) -> Optional[Dict]:
        """
        PATTERN 1: Support/Resistance Bounce
        - Price hits key support/resistance level
        - Strong bounce with volume confirmation
        """
        if len(df) < 50:
            return None
        
        current = df.iloc[-1]
        prev = df.iloc[-2]
        
        # Find support and resistance levels (last 50 candles)
        lookback = df.tail(50)
        support = lookback['low'].min()
        resistance = lookback['high'].max()
        
        # Calculate proximity to levels (within 1%)
        support_proximity = abs(current['close'] - support) / support * 100
        resistance_proximity = abs(current['close'] - resistance) / resistance * 100
        
        # Bullish bounce from support
        if support_proximity < 1.0 and current['close'] > prev['close']:
            # Confirm with volume
            if current['volume_ratio'] > 1.3:
                return {
                    'pattern': 'SUPPORT_BOUNCE',
                    'direction': 'LONG',
                    'entry_price': current['close'],
                    'level': support,
                    'confidence': min(current['volume_ratio'] * 30, 100)
                }
        
        # Bearish bounce from resistance
        elif resistance_proximity < 1.0 and current['close'] < prev['close']:
            if current['volume_ratio'] > 1.3:
                return {
                    'pattern': 'RESISTANCE_BOUNCE',
                    'direction': 'SHORT',
                    'entry_price': current['close'],
                    'level': resistance,
                    'confidence': min(current['volume_ratio'] * 30, 100)
                }
        
        return None
    
    def detect_bollinger_squeeze(self, df: pd.DataFrame) -> Optional[Dict]:
        """
        PATTERN 2: Bollinger Band Squeeze Breakout
        - Low volatility (tight bands) in PREVIOUS candles
        - Breakout expansion from consolidation on CURRENT candle
        """
        if len(df) < 30:
            return None
        
        current = df.iloc[-1]
        prev = df.iloc[-2]
        
        # Check if PREVIOUS 10 candles showed a squeeze (tight bands)
        # Exclude current candle from average calculation
        recent_widths = df['bb_width'].iloc[-11:-1]  # Last 10 candles before current
        avg_width = df['bb_width'].iloc[-21:-1].mean()  # 20 candles before current
        
        # Squeeze detected if recent average width was < 70% of longer-term average
        recent_avg_width = recent_widths.mean()
        was_squeezed = recent_avg_width < (avg_width * 0.7)
        
        if not was_squeezed:
            return None
        
        # Now check if current candle breaks out from the squeeze
        # Bullish breakout (close above upper band)
        if current['close'] > current['bb_upper'] and prev['close'] <= prev['bb_upper']:
            if current['volume_ratio'] > 1.2:
                return {
                    'pattern': 'BB_SQUEEZE_BREAKOUT',
                    'direction': 'LONG',
                    'entry_price': current['close'],
                    'squeeze_width': recent_avg_width,
                    'confidence': 75
                }
        
        # Bearish breakout (close below lower band)
        elif current['close'] < current['bb_lower'] and prev['close'] >= prev['bb_lower']:
            if current['volume_ratio'] > 1.2:
                return {
                    'pattern': 'BB_SQUEEZE_BREAKOUT',
                    'direction': 'SHORT',
                    'entry_price': current['close'],
                    'squeeze_width': recent_avg_width,
                    'confidence': 75
                }
        
        return None
    
    def detect_double_bottom_top(self, df: pd.DataFrame) -> Optional[Dict]:
        """
        PATTERN 3: Double Bottom/Top Reversal
        - W pattern (double bottom) or M pattern (double top)
        - Second bottom/top confirms reversal
        """
        if len(df) < 30:
            return None
        
        recent = df.tail(20)
        current = df.iloc[-1]
        
        # Find local minima and maxima
        lows = []
        highs = []
        
        for i in range(2, len(recent) - 2):
            # Local minimum
            if (recent.iloc[i]['low'] < recent.iloc[i-1]['low'] and 
                recent.iloc[i]['low'] < recent.iloc[i-2]['low'] and
                recent.iloc[i]['low'] < recent.iloc[i+1]['low'] and
                recent.iloc[i]['low'] < recent.iloc[i+2]['low']):
                lows.append(recent.iloc[i]['low'])
            
            # Local maximum
            if (recent.iloc[i]['high'] > recent.iloc[i-1]['high'] and 
                recent.iloc[i]['high'] > recent.iloc[i-2]['high'] and
                recent.iloc[i]['high'] > recent.iloc[i+1]['high'] and
                recent.iloc[i]['high'] > recent.iloc[i+2]['high']):
                highs.append(recent.iloc[i]['high'])
        
        # Double bottom (bullish reversal)
        if len(lows) >= 2:
            last_two_lows = lows[-2:]
            # Check if they're within 2% of each other
            if abs(last_two_lows[0] - last_two_lows[1]) / last_two_lows[0] * 100 < 2.0:
                # Confirm bounce happening
                if current['close'] > min(last_two_lows) * 1.02:
                    return {
                        'pattern': 'DOUBLE_BOTTOM',
                        'direction': 'LONG',
                        'entry_price': current['close'],
                        'support_level': min(last_two_lows),
                        'confidence': 80
                    }
        
        # Double top (bearish reversal)
        if len(highs) >= 2:
            last_two_highs = highs[-2:]
            if abs(last_two_highs[0] - last_two_highs[1]) / last_two_highs[0] * 100 < 2.0:
                if current['close'] < max(last_two_highs) * 0.98:
                    return {
                        'pattern': 'DOUBLE_TOP',
                        'direction': 'SHORT',
                        'entry_price': current['close'],
                        'resistance_level': max(last_two_highs),
                        'confidence': 80
                    }
        
        return None
    
    def detect_rsi_divergence(self, df: pd.DataFrame) -> Optional[Dict]:
        """
        PATTERN 4: RSI Divergence Reversal
        - Price makes new low, RSI doesn't (bullish divergence)
        - Price makes new high, RSI doesn't (bearish divergence)
        """
        if len(df) < 30:
            return None
        
        recent = df.tail(20)
        current = df.iloc[-1]
        
        # Find price and RSI extremes
        price_lows = []
        rsi_at_price_lows = []
        price_highs = []
        rsi_at_price_highs = []
        
        for i in range(2, len(recent) - 2):
            # Price low
            if (recent.iloc[i]['low'] < recent.iloc[i-1]['low'] and 
                recent.iloc[i]['low'] < recent.iloc[i+1]['low']):
                price_lows.append(recent.iloc[i]['low'])
                rsi_at_price_lows.append(recent.iloc[i]['rsi'])
            
            # Price high
            if (recent.iloc[i]['high'] > recent.iloc[i-1]['high'] and 
                recent.iloc[i]['high'] > recent.iloc[i+1]['high']):
                price_highs.append(recent.iloc[i]['high'])
                rsi_at_price_highs.append(recent.iloc[i]['rsi'])
        
        # Bullish divergence (price lower low, RSI higher low)
        if len(price_lows) >= 2 and len(rsi_at_price_lows) >= 2:
            if price_lows[-1] < price_lows[-2] and rsi_at_price_lows[-1] > rsi_at_price_lows[-2]:
                # RSI must be oversold area
                if current['rsi'] < 40:
                    return {
                        'pattern': 'BULLISH_DIVERGENCE',
                        'direction': 'LONG',
                        'entry_price': current['close'],
                        'rsi': current['rsi'],
                        'confidence': 70
                    }
        
        # Bearish divergence (price higher high, RSI lower high)
        if len(price_highs) >= 2 and len(rsi_at_price_highs) >= 2:
            if price_highs[-1] > price_highs[-2] and rsi_at_price_highs[-1] < rsi_at_price_highs[-2]:
                # RSI must be overbought area
                if current['rsi'] > 60:
                    return {
                        'pattern': 'BEARISH_DIVERGENCE',
                        'direction': 'SHORT',
                        'entry_price': current['close'],
                        'rsi': current['rsi'],
                        'confidence': 70
                    }
        
        return None
    
    def detect_volume_spike_reversal(self, df: pd.DataFrame) -> Optional[Dict]:
        """
        PATTERN 5: Volume Spike Reversal
        - Massive volume spike (capitulation)
        - Strong reversal candle
        """
        if len(df) < 30:
            return None
        
        current = df.iloc[-1]
        prev = df.iloc[-2]
        
        # Detect volume spike (3x+ average)
        if current['volume_ratio'] < 3.0:
            return None
        
        # Calculate candle body and wick
        body_size = abs(current['close'] - current['open']) / current['open'] * 100
        
        # Bullish reversal (hammer at bottom with volume)
        if current['close'] > current['open']:  # Green candle
            lower_wick = (min(current['open'], current['close']) - current['low']) / current['open'] * 100
            
            # Long lower wick (2x body) indicates buying pressure
            if lower_wick > body_size * 2 and lower_wick > 1.5:
                # Check if at potential bottom (RSI oversold)
                if current['rsi'] < 40:
                    return {
                        'pattern': 'VOLUME_SPIKE_REVERSAL',
                        'direction': 'LONG',
                        'entry_price': current['close'],
                        'volume_ratio': current['volume_ratio'],
                        'confidence': min(current['volume_ratio'] * 20, 95)
                    }
        
        # Bearish reversal (shooting star at top with volume)
        elif current['close'] < current['open']:  # Red candle
            upper_wick = (current['high'] - max(current['open'], current['close'])) / current['open'] * 100
            
            if upper_wick > body_size * 2 and upper_wick > 1.5:
                if current['rsi'] > 60:
                    return {
                        'pattern': 'VOLUME_SPIKE_REVERSAL',
                        'direction': 'SHORT',
                        'entry_price': current['close'],
                        'volume_ratio': current['volume_ratio'],
                        'confidence': min(current['volume_ratio'] * 20, 95)
                    }
        
        return None
    
    def calculate_percentage_targets(self, entry_price: float, direction: str) -> Dict:
        """
        Calculate percentage-based SL/TP targets (SWING category)
        With 10x leverage, percentages are LEVERAGED GAINS:
        - Stop Loss: 20% leveraged loss = 2% price movement
        - TP1: 15% leveraged gain = 1.5% price movement
        - TP2: 30% leveraged gain = 3% price movement
        - TP3: 50% leveraged gain = 5% price movement
        """
        leverage = 10
        
        # Convert leveraged % to price movement %
        sl_pct = 20 / leverage  # 2%
        tp1_pct = 15 / leverage  # 1.5%
        tp2_pct = 30 / leverage  # 3%
        tp3_pct = 50 / leverage  # 5%
        
        if direction == 'LONG':
            stop_loss = entry_price * (1 - sl_pct / 100)
            take_profit_1 = entry_price * (1 + tp1_pct / 100)
            take_profit_2 = entry_price * (1 + tp2_pct / 100)
            take_profit_3 = entry_price * (1 + tp3_pct / 100)
        else:  # SHORT
            stop_loss = entry_price * (1 + sl_pct / 100)
            take_profit_1 = entry_price * (1 - tp1_pct / 100)
            take_profit_2 = entry_price * (1 - tp2_pct / 100)
            take_profit_3 = entry_price * (1 - tp3_pct / 100)
        
        return {
            'stop_loss': round(stop_loss, 8),
            'take_profit_1': round(take_profit_1, 8),
            'take_profit_2': round(take_profit_2, 8),
            'take_profit_3': round(take_profit_3, 8),
            'take_profit': round(take_profit_3, 8)
        }
    
    async def scan_symbol_for_reversals(self, symbol: str, timeframe: str = '15m') -> Optional[Dict]:
        """
        Scan a symbol for ALL reversal patterns
        Returns the first pattern detected (highest priority)
        """
        try:
            df = await self.get_ohlcv(symbol, timeframe, limit=100)
            if df.empty:
                return None
            
            df = self.calculate_indicators(df)
            current = df.iloc[-1]
            
            # Check all patterns (in priority order)
            patterns = [
                self.detect_volume_spike_reversal(df),
                self.detect_double_bottom_top(df),
                self.detect_rsi_divergence(df),
                self.detect_bollinger_squeeze(df),
                self.detect_support_resistance_bounce(df)
            ]
            
            # Return first detected pattern
            for pattern in patterns:
                if pattern:
                    # Calculate SL/TP targets using percentage-based swing strategy
                    targets = self.calculate_percentage_targets(pattern['entry_price'], pattern['direction'])
                    
                    # Add common signal fields
                    pattern.update({
                        'symbol': symbol,
                        'timeframe': timeframe,
                        'rsi': float(current['rsi']),
                        'atr': float(current['atr']),
                        'volume': float(current['volume']),
                        'volume_avg': float(current['volume_avg']),
                        'signal_type': 'REVERSAL',
                        'stop_loss': targets['stop_loss'],
                        'take_profit': targets['take_profit'],
                        'take_profit_1': targets['take_profit_1'],
                        'take_profit_2': targets['take_profit_2'],
                        'take_profit_3': targets['take_profit_3'],
                        'risk_level': 'MEDIUM'  # Reversal signals are medium risk by default
                    })
                    
                    logger.info(f"🎯 REVERSAL DETECTED: {pattern['pattern']} - {symbol} {pattern['direction']}")
                    return pattern
            
            return None
            
        except Exception as e:
            logger.error(f"Error scanning {symbol} for reversals: {e}")
            return None
    
    async def scan_all_symbols(self) -> List[Dict]:
        """Scan all symbols for reversal patterns"""
        signals = []
        for symbol in self.symbols:
            signal = await self.scan_symbol_for_reversals(symbol)
            if signal:
                signals.append(signal)
        return signals
    
    async def close(self):
        await self.exchange.close()
