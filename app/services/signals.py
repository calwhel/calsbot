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
        self.timeframes = ['15m']  # 15-minute scalping for quick in/out trades
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
    
    def check_trend_strength(self, df: pd.DataFrame, direction: str) -> bool:
        """
        Check for consecutive higher highs (bullish) or lower lows (bearish)
        Ensures we're trading with clear trend direction, not choppy markets
        """
        if len(df) < 4:
            return False
        
        # Check last 3 candles for trend continuation
        if direction == 'LONG':
            # Require consecutive higher highs for bullish trend
            return (df.iloc[-1]['high'] > df.iloc[-2]['high'] and 
                    df.iloc[-2]['high'] > df.iloc[-3]['high'])
        else:  # SHORT
            # Require consecutive lower lows for bearish trend
            return (df.iloc[-1]['low'] < df.iloc[-2]['low'] and 
                    df.iloc[-2]['low'] < df.iloc[-3]['low'])
    
    def check_ema_cross(self, df: pd.DataFrame) -> Optional[Dict]:
        if len(df) < 4:  # Need more candles for trend strength check
            return None
        
        current = df.iloc[-1]
        previous = df.iloc[-2]
        
        # Guard against NaN values - all indicators must be valid
        required_fields = ['volume_avg', 'rsi', 'atr', 'ema_fast', 'ema_slow', 'ema_trend']
        if any(pd.isna(current[field]) for field in required_fields):
            return None
        
        # Volume Confirmation - RELAXED from 120% to 110% for more signals
        if current['volume_avg'] == 0:
            return None
        volume_confirmed = current['volume'] >= (current['volume_avg'] * 1.1)
        
        # RSI Filter - RELAXED from 60/40 to 55/45 for higher signal throughput
        rsi_ok_for_long = current['rsi'] > 55  # Bullish momentum
        rsi_ok_for_short = current['rsi'] < 45  # Bearish momentum
        
        # Classic EMA crossover signals WITH trend strength confirmation
        bullish_cross = (
            previous['ema_fast'] <= previous['ema_slow'] and
            current['ema_fast'] > current['ema_slow'] and
            current['close'] > current['ema_trend'] and
            volume_confirmed and
            rsi_ok_for_long and
            self.check_trend_strength(df, 'LONG')  # NEW: Trend strength filter
        )
        
        bearish_cross = (
            previous['ema_fast'] >= previous['ema_slow'] and
            current['ema_fast'] < current['ema_slow'] and
            current['close'] < current['ema_trend'] and
            volume_confirmed and
            rsi_ok_for_short and
            self.check_trend_strength(df, 'SHORT')  # NEW: Trend strength filter
        )
        
        # Trend-following signals - STRICTER with enhanced confluence
        ema_separation = abs(current['ema_fast'] - current['ema_slow']) / current['close'] * 100
        strong_trend_threshold = 0.8  # 0.8% EMA separation
        
        bullish_trend = (
            current['ema_fast'] > current['ema_slow'] and
            current['close'] > current['ema_fast'] and
            current['close'] > current['ema_trend'] and
            ema_separation > strong_trend_threshold and
            volume_confirmed and
            rsi_ok_for_long and  # Already requires > 60 (strengthened)
            self.check_trend_strength(df, 'LONG')  # NEW: Trend strength filter
        )
        
        bearish_trend = (
            current['ema_fast'] < current['ema_slow'] and
            current['close'] < current['ema_fast'] and
            current['close'] < current['ema_trend'] and
            ema_separation > strong_trend_threshold and
            volume_confirmed and
            rsi_ok_for_short and  # Already requires < 40 (strengthened)
            self.check_trend_strength(df, 'SHORT')  # NEW: Trend strength filter
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
    
    def get_adaptive_atr_multiplier(self, atr: float, entry_price: float) -> float:
        """
        Adaptive ATR multiplier based on volatility for scalping
        - Low volatility (<2%): 1.2x ATR (tighter stops)
        - Medium volatility (2-4%): 1.5x ATR (normal scalping)
        - High volatility (>4%): 2.0x ATR (wider stops for room)
        """
        atr_pct = (atr / entry_price) * 100
        
        if atr_pct < 2.0:
            return 1.2  # Low vol = tighter stops
        elif atr_pct > 4.0:
            return 2.0  # High vol = wider stops
        else:
            return 1.5  # Normal scalping stops
    
    def calculate_atr_stop_take(self, entry_price: float, direction: str, atr: float, atr_sl_multiplier: float = None) -> Dict:
        """
        PERCENTAGE-BASED stop loss and 3 take profit levels (SWING TRADING - Bigger Moves)
        - Stop Loss: 15% from entry (proportional to large targets)
        - TP1: 20% from entry (40% position close)
        - TP2: 40% from entry (30% position close)
        - TP3: 60% from entry (30% position close)
        """
        if direction == 'LONG':
            stop_loss = entry_price * 0.85
            take_profit_1 = entry_price * 1.20
            take_profit_2 = entry_price * 1.40
            take_profit_3 = entry_price * 1.60
        else:
            stop_loss = entry_price * 1.15
            take_profit_1 = entry_price * 0.80
            take_profit_2 = entry_price * 0.60
            take_profit_3 = entry_price * 0.40
        
        return {
            'stop_loss': round(stop_loss, 8),
            'take_profit_1': round(take_profit_1, 8),
            'take_profit_2': round(take_profit_2, 8),
            'take_profit_3': round(take_profit_3, 8),
            'take_profit': round(take_profit_3, 8),
            'atr_multiplier': 0
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
    
    async def check_order_flow_confirmation(self, symbol: str, direction: str) -> bool:
        """
        Check if order flow confirms the signal direction (scalping filter)
        Returns True if order flow aligns with signal direction
        """
        try:
            from app.services.spot_monitor import spot_monitor
            
            # Get recent order flow data
            flow_data = await spot_monitor.analyze_symbol_flow(symbol)
            
            if not flow_data:
                return True  # Allow trade if no flow data available
            
            flow_signal = flow_data.get('flow_signal', 'NEUTRAL')
            confidence = flow_data.get('confidence', 0)
            
            # Require 60%+ confidence for flow confirmation
            if confidence < 60:
                return True  # Allow if low confidence (not contradictory)
            
            # Check if order flow aligns with signal
            if direction == 'LONG' and flow_signal in ['HEAVY_BUYING', 'VOLUME_SPIKE_BUY']:
                return True  # Strong buy flow confirms long
            elif direction == 'SHORT' and flow_signal in ['HEAVY_SELLING', 'VOLUME_SPIKE_SELL']:
                return True  # Strong sell flow confirms short
            elif flow_signal == 'NEUTRAL':
                return True  # Neutral flow doesn't contradict
            else:
                return False  # Contradictory flow - reject signal
                
        except Exception as e:
            return True  # Allow trade if flow check fails
    
    async def check_higher_timeframe_confirmation(self, symbol: str, direction: str) -> bool:
        """
        MULTI-TIMEFRAME CONFIRMATION (Swing Trading)
        Check 1H timeframe for swing direction alignment before taking 15m entry
        
        Returns True if 1H trend aligns with 15m signal direction
        Returns False if 1H data unavailable or trend contradicts (strict gating)
        """
        try:
            # Fetch 1H data
            df_1h = await self.get_ohlcv(symbol, '1h', limit=100)
            if df_1h.empty:
                print(f"❌ 1H Confirmation FAILED for {symbol}: No 1H data available")
                return False
            
            # Calculate indicators on 1H
            df_1h = self.calculate_indicators(df_1h)
            current = df_1h.iloc[-1]
            
            # Guard against NaN values - STRICT: reject if indicators unavailable
            required_fields = ['ema_fast', 'ema_slow', 'ema_trend', 'rsi']
            if any(pd.isna(current[field]) for field in required_fields):
                print(f"❌ 1H Confirmation FAILED for {symbol}: Invalid indicators (NaN values)")
                return False
            
            # Check 1H trend alignment
            if direction == 'LONG':
                # For LONG: 1H must show bullish structure
                ema_aligned = (current['ema_fast'] > current['ema_slow'] and 
                              current['close'] > current['ema_trend'])
                rsi_bullish = current['rsi'] > 50  # Above midpoint
                trend_strength = self.check_trend_strength(df_1h, 'LONG')
                
                # Require at least 2 out of 3 confirmations
                confirmations = sum([ema_aligned, rsi_bullish, trend_strength])
                passed = confirmations >= 2
                
                if passed:
                    print(f"✅ 1H Confirmation PASSED for {symbol} LONG ({confirmations}/3)")
                else:
                    print(f"❌ 1H Confirmation FAILED for {symbol} LONG ({confirmations}/3 - need 2)")
                
                return passed
                
            else:  # SHORT
                # For SHORT: 1H must show bearish structure
                ema_aligned = (current['ema_fast'] < current['ema_slow'] and 
                              current['close'] < current['ema_trend'])
                rsi_bearish = current['rsi'] < 50  # Below midpoint
                trend_strength = self.check_trend_strength(df_1h, 'SHORT')
                
                # Require at least 2 out of 3 confirmations
                confirmations = sum([ema_aligned, rsi_bearish, trend_strength])
                passed = confirmations >= 2
                
                if passed:
                    print(f"✅ 1H Confirmation PASSED for {symbol} SHORT ({confirmations}/3)")
                else:
                    print(f"❌ 1H Confirmation FAILED for {symbol} SHORT ({confirmations}/3 - need 2)")
                
                return passed
                
        except Exception as e:
            print(f"❌ 1H Confirmation FAILED for {symbol}: Exception - {e}")
            return False
    
    async def generate_signal(self, symbol: str, timeframe: str) -> Optional[Dict]:
        df = await self.get_ohlcv(symbol, timeframe)
        if df.empty:
            return None
        
        df = self.calculate_indicators(df)
        cross = self.check_ema_cross(df)
        
        if not cross:
            return None
        
        # MULTI-TIMEFRAME CONFIRMATION: Check 1H trend before taking 15m entry
        htf_confirmed = await self.check_higher_timeframe_confirmation(symbol, cross['direction'])
        if not htf_confirmed:
            return None  # Skip signal if 1H trend doesn't align
        
        # Order flow confirmation for scalping (reject contradictory flow)
        flow_confirmed = await self.check_order_flow_confirmation(symbol, cross['direction'])
        if not flow_confirmed:
            return None  # Skip signal if order flow contradicts
        
        sr_levels = self.find_support_resistance(df)
        
        # Use ADAPTIVE ATR-based stop loss and take profit
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
            'take_profit_1': float(stop_take['take_profit_1']),
            'take_profit_2': float(stop_take['take_profit_2']),
            'take_profit_3': float(stop_take['take_profit_3']),
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
        # SWING TRADING: 15m entries confirmed by 1H trend
        # Multi-timeframe hybrid: 15m for timing, 1H for direction
        for timeframe in self.timeframes:
            for symbol in self.symbols:
                signal = await self.generate_signal(symbol, timeframe)
                if signal:
                    signals.append(signal)
        return signals
    
    async def close(self):
        await self.exchange.close()
