"""
Hybrid Signal System - Combines scalp and swing signals
- SCALP: Funding extremes (10%/15%/20% TPs, 12% SL)
- SWING: Divergence, Technical, Reversal (15%/30%/50% TPs, 20% SL)
- Session filters for quality control
"""
import ccxt.async_support as ccxt
import pandas as pd
import ta
from datetime import datetime, timezone
from typing import Dict, Optional, List
from app.config import settings
import logging

logger = logging.getLogger(__name__)


class SignalCategory:
    """Signal category definitions with their target profiles"""
    
    SWING = {
        'name': 'SWING',
        'tp1_pct': 15,
        'tp2_pct': 30,
        'tp3_pct': 50,
        'sl_pct': 20,
        'description': 'Multi-day swing trade'
    }
    
    SCALP = {
        'name': 'SCALP',
        'tp1_pct': 10,
        'tp2_pct': 15,
        'tp3_pct': 20,
        'sl_pct': 12,
        'description': 'Quick mean reversion (1-6 hours)'
    }
    
    AGGRESSIVE_SWING = {
        'name': 'AGGRESSIVE_SWING',
        'tp1_pct': 25,
        'tp2_pct': 50,
        'tp3_pct': 75,
        'sl_pct': 25,
        'description': 'High-conviction institutional play'
    }


class SessionQualityFilter:
    """Time-of-day session quality analysis"""
    
    @staticmethod
    def get_session_quality() -> Dict:
        """
        Returns current session quality and multipliers
        - BEST (1.2x): 12-16 UTC (EU/US overlap)
        - GOOD (1.0x): 6-12 UTC (Asian), 16-22 UTC (US)
        - MEDIUM (0.8x): 22-2 UTC (Late US/Early Asian)
        - POOR (0.5x): 2-6 UTC (Dead zone)
        """
        current_hour = datetime.now(timezone.utc).hour
        
        if 12 <= current_hour < 16:
            return {
                'quality': 'BEST',
                'multiplier': 1.2,
                'description': 'EU/US overlap - highest volume',
                'emoji': '🟢'
            }
        elif (6 <= current_hour < 12) or (16 <= current_hour < 22):
            return {
                'quality': 'GOOD',
                'multiplier': 1.0,
                'description': 'Active session - good liquidity',
                'emoji': '🟡'
            }
        elif (22 <= current_hour < 24) or (0 <= current_hour < 2):
            return {
                'quality': 'MEDIUM',
                'multiplier': 0.8,
                'description': 'Moderate liquidity',
                'emoji': '🟠'
            }
        else:  # 2-6 UTC
            return {
                'quality': 'POOR',
                'multiplier': 0.5,
                'description': 'Low liquidity - avoid if possible',
                'emoji': '🔴'
            }


class FundingRateDetector:
    """Detects funding rate extremes for mean reversion scalps"""
    
    def __init__(self):
        # Use Binance for funding rate data (most reliable and supports it)
        self.exchange_name = 'binance'
        self.exchange = ccxt.binance({
            'enableRateLimit': True,
            'options': {'defaultType': 'swap'}
        })
        self.extreme_threshold = 0.0010  # 0.1% is extreme
    
    async def check_funding_extreme(self, symbol: str) -> Optional[Dict]:
        """
        Detect funding rate extremes
        - Funding > 0.1% = Longs overheated → SHORT signal
        - Funding < -0.1% = Shorts overheated → LONG signal
        """
        try:
            # Convert to perpetual futures format (BTC/USDT -> BTC/USDT:USDT)
            perp_symbol = symbol if ':' in symbol else f"{symbol}:USDT"
            funding = await self.exchange.fetch_funding_rate(perp_symbol)
            
            if not funding or 'fundingRate' not in funding:
                return None
            
            rate = funding['fundingRate']
            
            # Extreme positive funding = too many longs → SHORT
            if rate > self.extreme_threshold:
                confidence = min(70 + int((rate - self.extreme_threshold) * 10000), 95)
                
                return {
                    'signal_type': 'FUNDING_EXTREME',
                    'signal_category': SignalCategory.SCALP,
                    'direction': 'SHORT',
                    'entry_price': funding.get('markPrice', 0),
                    'funding_rate': rate * 100,  # Convert to percentage
                    'confidence': confidence,
                    'reason': f'Extreme long funding ({rate*100:.3f}%) - mean reversion expected',
                    'symbol': symbol
                }
            
            # Extreme negative funding = too many shorts → LONG
            elif rate < -self.extreme_threshold:
                confidence = min(70 + int(abs(rate + self.extreme_threshold) * 10000), 95)
                
                return {
                    'signal_type': 'FUNDING_EXTREME',
                    'signal_category': SignalCategory.SCALP,
                    'direction': 'LONG',
                    'entry_price': funding.get('markPrice', 0),
                    'funding_rate': rate * 100,
                    'confidence': confidence,
                    'reason': f'Extreme short funding ({rate*100:.3f}%) - mean reversion expected',
                    'symbol': symbol
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Error checking funding for {symbol}: {e}")
            return None
    
    async def close(self):
        await self.exchange.close()


class DivergenceDetector:
    """Detects MACD and RSI divergence for swing trades"""
    
    def __init__(self):
        self.exchange_name = settings.EXCHANGE
        self.exchange = getattr(ccxt, self.exchange_name)()
    
    async def get_ohlcv(self, symbol: str, timeframe: str = '1h', limit: int = 100) -> pd.DataFrame:
        """Fetch OHLCV data"""
        try:
            ohlcv = await self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            return df
        except Exception as e:
            logger.error(f"Error fetching OHLCV for {symbol}: {e}")
            return pd.DataFrame()
    
    def calculate_divergence_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate MACD and RSI for divergence detection"""
        # MACD
        macd = ta.trend.MACD(df['close'])
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        df['macd_diff'] = macd.macd_diff()
        
        # RSI
        df['rsi'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
        
        return df
    
    def detect_divergence(self, df: pd.DataFrame) -> Optional[Dict]:
        """
        Detect bullish and bearish divergence
        - Bullish: Price lower low + RSI/MACD higher low
        - Bearish: Price higher high + RSI/MACD lower high
        """
        if len(df) < 20:
            return None
        
        df = self.calculate_divergence_indicators(df)
        
        # Check last 20 candles for divergence
        recent = df.tail(20)
        current = df.iloc[-1]
        
        # Find price highs and lows in recent period
        price_high_idx = recent['high'].idxmax()
        price_low_idx = recent['low'].idxmin()
        
        # BULLISH DIVERGENCE: Price lower low, RSI higher low
        if price_low_idx < len(df) - 5:  # Low happened at least 5 candles ago
            price_at_low = recent.loc[price_low_idx, 'low']
            rsi_at_low = recent.loc[price_low_idx, 'rsi']
            
            # Current price lower than previous low, but RSI higher
            if (current['low'] < price_at_low and 
                current['rsi'] > rsi_at_low and
                current['rsi'] < 40):  # Still oversold
                
                return {
                    'signal_type': 'BULLISH_DIVERGENCE',
                    'signal_category': SignalCategory.SWING,
                    'direction': 'LONG',
                    'entry_price': current['close'],
                    'rsi': current['rsi'],
                    'confidence': 80,
                    'reason': f'Bullish divergence: Price lower low but RSI higher low (RSI: {current["rsi"]:.1f})',
                    'pattern': 'RSI_DIVERGENCE'
                }
        
        # BEARISH DIVERGENCE: Price higher high, RSI lower high
        if price_high_idx < len(df) - 5:
            price_at_high = recent.loc[price_high_idx, 'high']
            rsi_at_high = recent.loc[price_high_idx, 'rsi']
            
            # Current price higher than previous high, but RSI lower
            if (current['high'] > price_at_high and 
                current['rsi'] < rsi_at_high and
                current['rsi'] > 60):  # Still overbought
                
                return {
                    'signal_type': 'BEARISH_DIVERGENCE',
                    'signal_category': SignalCategory.SWING,
                    'direction': 'SHORT',
                    'entry_price': current['close'],
                    'rsi': current['rsi'],
                    'confidence': 80,
                    'reason': f'Bearish divergence: Price higher high but RSI lower high (RSI: {current["rsi"]:.1f})',
                    'pattern': 'RSI_DIVERGENCE'
                }
        
        return None
    
    async def scan_for_divergence(self, symbol: str) -> Optional[Dict]:
        """Scan a symbol for divergence signals"""
        try:
            df = await self.get_ohlcv(symbol, '1h', 100)
            if df.empty:
                return None
            
            divergence = self.detect_divergence(df)
            if divergence:
                divergence['symbol'] = symbol
                divergence['timeframe'] = '1h'
                return divergence
            
            return None
            
        except Exception as e:
            logger.error(f"Error scanning divergence for {symbol}: {e}")
            return None
    
    async def close(self):
        await self.exchange.close()


def calculate_dynamic_targets(entry_price: float, direction: str, category: Dict) -> Dict:
    """
    Calculate TP/SL levels based on signal category
    Category percentages are LEVERAGED GAINS, need to convert to price movements
    With 10x leverage: 20% leveraged gain = 2% price movement
    """
    leverage = 10  # Default leverage
    
    # Convert leveraged % to actual price movement % (divide by leverage)
    sl_pct = category['sl_pct'] / leverage
    tp1_pct = category['tp1_pct'] / leverage
    tp2_pct = category['tp2_pct'] / leverage
    tp3_pct = category['tp3_pct'] / leverage
    
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
        'take_profit': round(take_profit_3, 8),
        'tp1_pct': category['tp1_pct'],
        'tp2_pct': category['tp2_pct'],
        'tp3_pct': category['tp3_pct'],
        'sl_pct': category['sl_pct'],
        'category_name': category['name'],
        'category_desc': category['description']
    }


async def scan_hybrid_signals(symbols: List[str]) -> List[Dict]:
    """
    Main function to scan for all hybrid signals
    Returns list of signals with proper categorization
    """
    signals = []
    
    # Get session quality
    session = SessionQualityFilter.get_session_quality()
    
    # Initialize detectors
    funding_detector = FundingRateDetector()
    divergence_detector = DivergenceDetector()
    
    try:
        for symbol in symbols:
            # Check funding extremes (SCALP category)
            funding_signal = await funding_detector.check_funding_extreme(symbol)
            if funding_signal:
                # Calculate dynamic targets
                targets = calculate_dynamic_targets(
                    funding_signal['entry_price'],
                    funding_signal['direction'],
                    funding_signal['signal_category']
                )
                funding_signal.update(targets)
                funding_signal['session_quality'] = session
                
                # Adjust confidence based on session quality
                funding_signal['confidence'] = int(funding_signal['confidence'] * session['multiplier'])
                
                signals.append(funding_signal)
                logger.info(f"✨ SCALP: Funding extreme on {symbol} - {funding_signal['direction']}")
            
            # Check for divergence (SWING category)
            divergence_signal = await divergence_detector.scan_for_divergence(symbol)
            if divergence_signal:
                # Calculate dynamic targets
                targets = calculate_dynamic_targets(
                    divergence_signal['entry_price'],
                    divergence_signal['direction'],
                    divergence_signal['signal_category']
                )
                divergence_signal.update(targets)
                divergence_signal['session_quality'] = session
                
                # Adjust confidence based on session quality
                divergence_signal['confidence'] = int(divergence_signal['confidence'] * session['multiplier'])
                
                signals.append(divergence_signal)
                logger.info(f"✨ SWING: Divergence on {symbol} - {divergence_signal['direction']}")
        
    finally:
        await funding_detector.close()
        await divergence_detector.close()
    
    return signals
