"""
Smart Exit System - Detects market reversals and closes trades early
Protects profits and minimizes losses by exiting when momentum shifts
"""
import ccxt.async_support as ccxt
import pandas as pd
import ta
from typing import Dict, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class SmartExitDetector:
    """
    Detects early exit opportunities when market reverses against your position
    """
    
    def __init__(self, exchange_name: str = 'kucoin'):
        # Bitunix is not supported by CCXT, use Binance as fallback for market data
        if exchange_name.lower() == 'bitunix':
            logger.info("Using Binance as fallback for Bitunix market data")
            self.exchange = ccxt.binance()
        else:
            self.exchange = getattr(ccxt, exchange_name)()
    
    async def should_exit_early(
        self,
        symbol: str,
        direction: str,
        entry_price: float,
        current_price: float
    ) -> tuple[bool, Optional[str]]:
        """
        Analyze if position should exit early due to reversal
        
        Returns:
            (should_exit: bool, reason: str)
        """
        try:
            # Fetch recent candles for analysis
            ohlcv = await self.exchange.fetch_ohlcv(symbol, '15m', limit=50)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            
            # Calculate indicators
            df['ema_9'] = ta.trend.EMAIndicator(df['close'], window=9).ema_indicator()
            df['ema_21'] = ta.trend.EMAIndicator(df['close'], window=21).ema_indicator()
            df['rsi'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
            
            # Get latest values
            latest = df.iloc[-1]
            prev = df.iloc[-2]
            
            # 1. EMA CROSSOVER REVERSAL
            if direction == 'LONG':
                # LONG position: Exit if EMA9 crosses below EMA21 (downtrend starting)
                if prev['ema_9'] > prev['ema_21'] and latest['ema_9'] < latest['ema_21']:
                    return True, "🔴 EMA Death Cross detected - Downtrend forming"
            else:  # SHORT
                # SHORT position: Exit if EMA9 crosses above EMA21 (uptrend starting)
                if prev['ema_9'] < prev['ema_21'] and latest['ema_9'] > latest['ema_21']:
                    return True, "🟢 EMA Golden Cross detected - Uptrend forming"
            
            # 2. STRONG REVERSAL CANDLE PATTERN
            reversal_candle = self._detect_reversal_candle(df, direction)
            if reversal_candle:
                return True, reversal_candle
            
            # 3. RSI DIVERGENCE (Price moving against RSI)
            if direction == 'LONG':
                # LONG: Price making higher highs but RSI making lower highs = bearish divergence
                price_higher_high = latest['high'] > df['high'].iloc[-10:-1].max()
                rsi_lower_high = latest['rsi'] < df['rsi'].iloc[-10:-1].max()
                
                if price_higher_high and rsi_lower_high and latest['rsi'] > 70:
                    return True, "📉 Bearish RSI divergence - Momentum weakening"
            else:  # SHORT
                # SHORT: Price making lower lows but RSI making higher lows = bullish divergence
                price_lower_low = latest['low'] < df['low'].iloc[-10:-1].min()
                rsi_higher_low = latest['rsi'] > df['rsi'].iloc[-10:-1].min()
                
                if price_lower_low and rsi_higher_low and latest['rsi'] < 30:
                    return True, "📈 Bullish RSI divergence - Momentum weakening"
            
            # 4. EXTREME RSI REVERSAL ZONES
            if direction == 'LONG' and latest['rsi'] > 75:
                # LONG in extreme overbought - high risk of reversal
                body_pct = abs(latest['close'] - latest['open']) / latest['open'] * 100
                if body_pct > 2 and latest['close'] < latest['open']:  # Strong red candle
                    return True, "⚠️ Extreme overbought + rejection candle"
            
            elif direction == 'SHORT' and latest['rsi'] < 25:
                # SHORT in extreme oversold - high risk of reversal
                body_pct = abs(latest['close'] - latest['open']) / latest['open'] * 100
                if body_pct > 2 and latest['close'] > latest['open']:  # Strong green candle
                    return True, "⚠️ Extreme oversold + bounce candle"
            
            # 5. VOLUME SPIKE REVERSAL
            avg_volume = df['volume'].iloc[-20:-1].mean()
            if latest['volume'] > avg_volume * 2:  # 2x volume spike
                if direction == 'LONG' and latest['close'] < latest['open']:
                    # LONG: Volume spike with red candle = distribution
                    wick_size = (latest['high'] - latest['close']) / latest['close'] * 100
                    if wick_size > 1.5:  # Rejection wick
                        return True, "🔴 Volume spike distribution - Sellers taking over"
                
                elif direction == 'SHORT' and latest['close'] > latest['open']:
                    # SHORT: Volume spike with green candle = accumulation
                    wick_size = (latest['close'] - latest['low']) / latest['close'] * 100
                    if wick_size > 1.5:  # Support wick
                        return True, "🟢 Volume spike accumulation - Buyers taking over"
            
            # 6. PROFIT PROTECTION: Lock in gains ONLY if showing STRONG reversal signs
            # With 10x leverage: 15% leveraged = 1.5% price movement (more reasonable threshold)
            pnl_pct = ((current_price - entry_price) / entry_price * 100) if direction == 'LONG' else ((entry_price - current_price) / entry_price * 100)
            
            if pnl_pct > 15:  # Need significant profit (1.5% price move with 10x leverage)
                # Only exit if BOTH conditions met + sustained trend reversal:
                if direction == 'LONG':
                    # Require SUSTAINED downtrend: EMA9 declining for 3+ candles
                    ema_declining = (
                        latest['ema_9'] < df['ema_9'].iloc[-2] and
                        df['ema_9'].iloc[-2] < df['ema_9'].iloc[-3] and
                        df['ema_9'].iloc[-3] < df['ema_9'].iloc[-4]
                    )
                    # AND EMA9 below EMA21 (death cross confirmed)
                    death_cross = latest['ema_9'] < latest['ema_21']
                    # AND RSI in bearish territory
                    rsi_bearish = latest['rsi'] < 45
                    
                    if ema_declining and death_cross and rsi_bearish:
                        return True, f"💰 Profit protection: +{pnl_pct:.1f}% with sustained reversal"
                else:
                    # Require SUSTAINED uptrend: EMA9 rising for 3+ candles
                    ema_rising = (
                        latest['ema_9'] > df['ema_9'].iloc[-2] and
                        df['ema_9'].iloc[-2] > df['ema_9'].iloc[-3] and
                        df['ema_9'].iloc[-3] > df['ema_9'].iloc[-4]
                    )
                    # AND EMA9 above EMA21 (golden cross confirmed)
                    golden_cross = latest['ema_9'] > latest['ema_21']
                    # AND RSI in bullish territory
                    rsi_bullish = latest['rsi'] > 55
                    
                    if ema_rising and golden_cross and rsi_bullish:
                        return True, f"💰 Profit protection: +{pnl_pct:.1f}% with sustained reversal"
            
            return False, None
            
        except Exception as e:
            logger.error(f"Error in smart exit detection for {symbol}: {e}")
            return False, None
        finally:
            await self.exchange.close()
    
    def _detect_reversal_candle(self, df: pd.DataFrame, direction: str) -> Optional[str]:
        """
        Detect strong reversal candle patterns
        """
        latest = df.iloc[-1]
        prev = df.iloc[-2]
        
        body = abs(latest['close'] - latest['open'])
        prev_body = abs(prev['close'] - prev['open'])
        range_size = latest['high'] - latest['low']
        
        if range_size == 0:
            return None
        
        body_pct = (body / latest['open']) * 100
        
        if direction == 'LONG':
            # BEARISH ENGULFING: Large red candle engulfing previous green
            if (latest['close'] < latest['open'] and  # Red candle
                prev['close'] > prev['open'] and  # Previous was green
                latest['open'] > prev['close'] and  # Opened above previous close
                latest['close'] < prev['open'] and  # Closed below previous open
                body > prev_body * 1.5):  # 1.5x larger
                return "🔴 Bearish Engulfing - Strong reversal signal"
            
            # SHOOTING STAR: Long upper wick in uptrend
            upper_wick = latest['high'] - max(latest['open'], latest['close'])
            lower_wick = min(latest['open'], latest['close']) - latest['low']
            
            if (upper_wick > body * 2 and  # Upper wick 2x body
                lower_wick < body * 0.3 and  # Small lower wick
                latest['close'] < latest['open']):  # Red candle
                return "🎯 Shooting Star - Top reversal pattern"
        
        else:  # SHORT position
            # BULLISH ENGULFING: Large green candle engulfing previous red
            if (latest['close'] > latest['open'] and  # Green candle
                prev['close'] < prev['open'] and  # Previous was red
                latest['open'] < prev['close'] and  # Opened below previous close
                latest['close'] > prev['open'] and  # Closed above previous open
                body > prev_body * 1.5):  # 1.5x larger
                return "🟢 Bullish Engulfing - Strong reversal signal"
            
            # HAMMER: Long lower wick in downtrend
            upper_wick = latest['high'] - max(latest['open'], latest['close'])
            lower_wick = min(latest['open'], latest['close']) - latest['low']
            
            if (lower_wick > body * 2 and  # Lower wick 2x body
                upper_wick < body * 0.3 and  # Small upper wick
                latest['close'] > latest['open']):  # Green candle
                return "🔨 Hammer - Bottom reversal pattern"
        
        return None


async def check_smart_exit(
    symbol: str,
    direction: str,
    entry_price: float,
    current_price: float,
    exchange: str = 'kucoin'
) -> tuple[bool, Optional[str]]:
    """
    Convenience function to check if trade should exit early
    """
    detector = SmartExitDetector(exchange)
    return await detector.should_exit_early(symbol, direction, entry_price, current_price)
