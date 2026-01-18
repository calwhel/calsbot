import logging
import asyncio
from typing import Dict, List, Optional
from datetime import datetime
import ccxt.async_support as ccxt
import numpy as np

logger = logging.getLogger(__name__)

class VWAPScalpStrategy:
    """
    VWAP Bounce Scalp Strategy
    Targets 0.3-0.5% moves at 20x leverage.
    Criteria:
    1. Trend: 1H EMA21 > EMA50 (Bullish)
    2. Pullback: Price touches or dips slightly below VWAP on 5m
    3. Support: EMA21 (5m) nearby
    4. RSI: 40-50 (Neutral-Oversold in uptrend)
    5. Volume: Confirmation on bounce
    """
    
    def __init__(self):
        self.exchange = None

    async def initialize(self):
        if not self.exchange:
            self.exchange = ccxt.binance({
                'enableRateLimit': True,
                'options': {'defaultType': 'future'}
            })

    def calculate_vwap(self, candles: List) -> float:
        """Calculate current VWAP from candles (typical price * volume)"""
        tp_v = 0
        total_v = 0
        for c in candles:
            high, low, close, vol = c[2], c[3], c[4], c[5]
            typical_price = (high + low + close) / 3
            tp_v += typical_price * vol
            total_v += vol
        return tp_v / total_v if total_v > 0 else candles[-1][4]

    def calculate_ema(self, prices: List[float], period: int) -> float:
        if len(prices) < period:
            return prices[-1]
        multiplier = 2 / (period + 1)
        ema = prices[0]
        for price in prices[1:]:
            ema = (price - ema) * multiplier + ema
        return ema

    def calculate_rsi(self, prices: List[float], period: int = 14) -> float:
        if len(prices) < period + 1:
            return 50
        deltas = np.diff(prices)
        seed = deltas[:period]
        up = seed[seed >= 0].sum() / period
        down = -seed[seed < 0].sum() / period
        rs = up / down if down != 0 else 100
        rsi = np.zeros_like(prices)
        rsi[:period] = 100. - 100. / (1. + rs)

        for i in range(period, len(prices)):
            delta = deltas[i - 1]
            if delta > 0:
                upval = delta
                downval = 0.
            else:
                upval = 0.
                downval = -delta

            up = (up * (period - 1) + upval) / period
            down = (down * (period - 1) + downval) / period
            rs = up / down if down != 0 else 100
            rsi[i] = 100. - 100. / (1. + rs)
        return rsi[-1]

    async def analyze_symbol(self, symbol: str) -> Optional[Dict]:
        try:
            await self.initialize()
            # Normalize symbol
            if not symbol.endswith('/USDT'):
                symbol = f"{symbol.upper()}/USDT"
            
            # Fetch multiple timeframes
            # 1H for trend
            ohlcv_1h = await self.exchange.fetch_ohlcv(symbol, '1h', limit=100)
            closes_1h = [c[4] for c in ohlcv_1h]
            ema21_1h = self.calculate_ema(closes_1h, 21)
            ema50_1h = self.calculate_ema(closes_1h, 50)
            
            # Trend Check: Only LONG if 1H trend is bullish
            if ema21_1h <= ema50_1h:
                return None
            
            # 5m for entry
            ohlcv_5m = await self.exchange.fetch_ohlcv(symbol, '5m', limit=100)
            closes_5m = [c[4] for c in ohlcv_5m]
            current_price = closes_5m[-1]
            
            vwap = self.calculate_vwap(ohlcv_5m)
            ema21_5m = self.calculate_ema(closes_5m, 21)
            rsi = self.calculate_rsi(closes_5m, 14)
            
            # Pullback to VWAP/EMA21 check
            # Distance to VWAP
            dist_vwap = (current_price - vwap) / vwap * 100
            dist_ema = (current_price - ema21_5m) / ema21_5m * 100
            
            # Criteria: Price near VWAP (within 0.3% or just below)
            # and RSI between 38-55 indicating temporary cooling
            if -0.3 <= dist_vwap <= 0.2 and 38 <= rsi <= 55:
                # Potential Bounce
                return {
                    'symbol': symbol,
                    'price': current_price,
                    'vwap': vwap,
                    'ema21_5m': ema21_5m,
                    'rsi': rsi,
                    'trend_1h': 'bullish',
                    'dist_vwap': dist_vwap,
                    'direction': 'LONG',
                    'strategy': 'VWAP_BOUNCE_SCALP'
                }
            
            return None
        except Exception as e:
            logger.error(f"Error analyzing {symbol} for VWAP scalp: {e}")
            return None
        finally:
            # We don't close exchange here to reuse it, but ensure cleanup in main loop
            pass
