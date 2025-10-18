import ccxt.async_support as ccxt
import logging
import pandas as pd
import ta
from typing import Dict, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class MultiAnalysisConfirmation:
    """
    Multi-timeframe and multi-indicator confirmation system
    Validates signals against higher timeframes and additional analysis
    before allowing trade execution
    """
    
    def __init__(self):
        self.exchanges = {}  # Cache exchanges by name
    
    async def validate_signal(
        self, 
        symbol: str, 
        direction: str, 
        entry_price: float,
        exchange_name: str = 'kucoin'
    ) -> tuple[bool, str, dict]:
        """
        Validate a signal against multiple analysis layers
        
        Args:
            symbol: Trading pair (e.g., 'ETH/USDT')
            direction: 'LONG' or 'SHORT'
            entry_price: Proposed entry price
            exchange_name: Exchange to fetch data from
        
        Returns:
            (is_valid, reason, analysis_data)
        """
        try:
            # Get or create exchange instance for this exchange
            if exchange_name not in self.exchanges:
                if exchange_name == 'kucoin':
                    self.exchanges[exchange_name] = ccxt.kucoin()
                elif exchange_name == 'okx':
                    self.exchanges[exchange_name] = ccxt.okx()
                elif exchange_name == 'binance':
                    self.exchanges[exchange_name] = ccxt.binance()
                else:
                    # Default to kucoin
                    self.exchanges[exchange_name] = ccxt.kucoin()
            
            exchange = self.exchanges[exchange_name]
            
            # Fetch 1-hour timeframe for higher timeframe confirmation
            ohlcv_1h = await exchange.fetch_ohlcv(symbol, '1h', limit=50)
            df_1h = pd.DataFrame(ohlcv_1h, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            
            # Calculate indicators on 1H timeframe
            df_1h = self._calculate_indicators(df_1h)
            
            # Run confirmation checks
            confirmations = {
                'higher_tf_trend': self._check_higher_timeframe_trend(df_1h, direction),
                'volume_profile': self._check_volume_profile(df_1h, direction),
                'momentum_alignment': self._check_momentum_alignment(df_1h, direction),
                'price_structure': self._check_price_structure(df_1h, direction)
            }
            
            # Count confirmations
            confirmed_count = sum(1 for v in confirmations.values() if v)
            total_checks = len(confirmations)
            
            # Require at least 3 out of 4 confirmations (75% agreement)
            min_confirmations = 3
            is_valid = confirmed_count >= min_confirmations
            
            # Build reason string
            if is_valid:
                reason = f"✅ Multi-analysis CONFIRMED ({confirmed_count}/{total_checks})"
            else:
                reason = f"❌ Multi-analysis REJECTED ({confirmed_count}/{total_checks} - need {min_confirmations})"
                failed_checks = [k for k, v in confirmations.items() if not v]
                reason += f" | Failed: {', '.join(failed_checks)}"
            
            analysis_data = {
                'confirmations': confirmations,
                'confirmed_count': confirmed_count,
                'total_checks': total_checks,
                'timestamp': datetime.utcnow().isoformat()
            }
            
            logger.info(f"Multi-analysis for {symbol} {direction}: {reason}")
            
            return is_valid, reason, analysis_data
            
        except Exception as e:
            logger.error(f"Error in multi-analysis validation for {symbol} on {exchange_name}: {e}")
            # CRITICAL FIX: Do NOT fail open - BLOCK trade on validation error
            # This ensures we never execute trades without proper confirmation
            return False, f"❌ Multi-analysis validation ERROR - trade blocked: {str(e)}", {}
    
    def _calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators for analysis"""
        df['ema_20'] = ta.trend.EMAIndicator(df['close'], window=20).ema_indicator()
        df['ema_50'] = ta.trend.EMAIndicator(df['close'], window=50).ema_indicator()
        df['rsi'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
        df['macd'] = ta.trend.MACD(df['close']).macd()
        df['macd_signal'] = ta.trend.MACD(df['close']).macd_signal()
        df['volume_sma'] = df['volume'].rolling(window=20).mean()
        return df
    
    def _check_higher_timeframe_trend(self, df: pd.DataFrame, direction: str) -> bool:
        """
        Check if 1H timeframe aligns with trade direction
        LONG: Price above EMA20 and EMA50, both trending up
        SHORT: Price below EMA20 and EMA50, both trending down
        """
        if len(df) < 5:
            return False
        
        current = df.iloc[-1]
        prev = df.iloc[-2]
        
        if pd.isna(current['ema_20']) or pd.isna(current['ema_50']):
            return False
        
        if direction == 'LONG':
            # Bullish on higher timeframe
            return (
                current['close'] > current['ema_20'] and
                current['close'] > current['ema_50'] and
                current['ema_20'] > prev['ema_20']  # EMA trending up
            )
        else:  # SHORT
            # Bearish on higher timeframe
            return (
                current['close'] < current['ema_20'] and
                current['close'] < current['ema_50'] and
                current['ema_20'] < prev['ema_20']  # EMA trending down
            )
    
    def _check_volume_profile(self, df: pd.DataFrame, direction: str) -> bool:
        """
        Check volume profile for trend confirmation
        Increasing volume on trend = strong confirmation
        """
        if len(df) < 3:
            return False
        
        current = df.iloc[-1]
        avg_volume = df.iloc[-10:]['volume'].mean() if len(df) >= 10 else current['volume_sma']
        
        # Volume should be above average for strong moves
        return current['volume'] > avg_volume * 0.8
    
    def _check_momentum_alignment(self, df: pd.DataFrame, direction: str) -> bool:
        """
        Check MACD and RSI alignment with direction
        """
        if len(df) < 3:
            return False
        
        current = df.iloc[-1]
        
        if pd.isna(current['macd']) or pd.isna(current['rsi']):
            return False
        
        if direction == 'LONG':
            # Bullish momentum: MACD above signal, RSI > 50
            return current['macd'] > current['macd_signal'] and current['rsi'] > 50
        else:  # SHORT
            # Bearish momentum: MACD below signal, RSI < 50
            return current['macd'] < current['macd_signal'] and current['rsi'] < 50
    
    def _check_price_structure(self, df: pd.DataFrame, direction: str) -> bool:
        """
        Check price structure for trend confirmation
        LONG: Higher highs and higher lows
        SHORT: Lower highs and lower lows
        """
        if len(df) < 3:
            return False
        
        # Check last 2 candles
        curr = df.iloc[-1]
        prev = df.iloc[-2]
        prev2 = df.iloc[-3]
        
        if direction == 'LONG':
            # Higher highs and higher lows
            return curr['high'] > prev['high'] and curr['low'] > prev2['low']
        else:  # SHORT
            # Lower highs and lower lows
            return curr['high'] < prev['high'] and curr['low'] < prev2['low']
    
    async def close(self):
        """Close all exchange connections"""
        for exchange in self.exchanges.values():
            try:
                await exchange.close()
            except Exception as e:
                logger.error(f"Error closing exchange: {e}")


# Global singleton instance
_global_multi_analysis = MultiAnalysisConfirmation()


async def validate_trade_signal(
    symbol: str,
    direction: str,
    entry_price: float,
    exchange_name: str = 'kucoin'
) -> tuple[bool, str, dict]:
    """
    Global function to validate a trade signal before execution
    
    Returns:
        (is_valid, reason, analysis_data)
    """
    return await _global_multi_analysis.validate_signal(
        symbol, direction, entry_price, exchange_name
    )
