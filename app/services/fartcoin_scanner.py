"""
FARTCOIN Scanner - Dedicated scanner for FARTCOIN/USDT with SOL correlation tracking.

Strategy (SOL BETA AMPLIFICATION + LATENCY):
- FARTCOIN amplifies SOL moves: when SOL pumps, FART pumps HARDER
- FARTCOIN lags SOL moves: when SOL dumps, FART dumps with DELAY
- LONG: Detect SOL pump starting → enter FART before the amplified catch-up
- SHORT: Detect SOL dump starting → enter FART before the delayed dump hits
- Compares SOL momentum across 1m/5m/15m windows to catch early moves
- Uses Binance Futures for price/candle data
- AI-validated via Gemini (scanning) + Claude (final approval)
- Separate alert channel with own cooldowns and limits
"""

import os
import logging
import asyncio
import httpx
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

FARTCOIN_SYMBOL = "FARTCOINUSDT"
SOL_SYMBOL = "SOLUSDT"
FARTCOIN_LEVERAGE = 50
FARTCOIN_SCAN_INTERVAL = 90
FARTCOIN_COOLDOWN_MINUTES = 360
MAX_FARTCOIN_DAILY_SIGNALS = 3

BINANCE_FUTURES_URL = "https://fapi.binance.com"

API_SEMAPHORE = asyncio.Semaphore(5)

_fartcoin_enabled = False
_fartcoin_last_signal_time = None
_fartcoin_daily_count = 0
_fartcoin_daily_reset = None

_correlation_history: List[Dict] = []
MAX_CORRELATION_HISTORY = 120


def is_fartcoin_enabled() -> bool:
    return _fartcoin_enabled


def set_fartcoin_enabled(enabled: bool):
    global _fartcoin_enabled
    _fartcoin_enabled = enabled
    logger.info(f"🐸 FARTCOIN scanner {'ENABLED' if enabled else 'DISABLED'}")


def check_fartcoin_cooldown() -> bool:
    global _fartcoin_last_signal_time
    if _fartcoin_last_signal_time is None:
        return True
    elapsed = (datetime.utcnow() - _fartcoin_last_signal_time).total_seconds()
    return elapsed >= (FARTCOIN_COOLDOWN_MINUTES * 60)


def check_fartcoin_daily_limit() -> bool:
    global _fartcoin_daily_count, _fartcoin_daily_reset
    now = datetime.utcnow()
    if _fartcoin_daily_reset is None or now.date() > _fartcoin_daily_reset.date():
        _fartcoin_daily_count = 0
        _fartcoin_daily_reset = now
    return _fartcoin_daily_count < MAX_FARTCOIN_DAILY_SIGNALS


def record_fartcoin_signal():
    global _fartcoin_last_signal_time, _fartcoin_daily_count
    _fartcoin_last_signal_time = datetime.utcnow()
    _fartcoin_daily_count += 1


class FartcoinScanner:
    def __init__(self):
        self.client = None
        self.binance_url = BINANCE_FUTURES_URL

    async def init(self):
        if self.client is None:
            self.client = httpx.AsyncClient(
                timeout=15,
                limits=httpx.Limits(max_connections=10, max_keepalive_connections=5)
            )

    async def close(self):
        if self.client:
            await self.client.aclose()
            self.client = None

    async def fetch_candles(self, symbol: str, interval: str, limit: int = 100) -> List:
        async with API_SEMAPHORE:
            try:
                url = f"{self.binance_url}/fapi/v1/klines"
                params = {
                    'symbol': symbol,
                    'interval': interval,
                    'limit': limit
                }
                response = await self.client.get(url, params=params, timeout=10)
                response.raise_for_status()
                data = response.json()

                if isinstance(data, list) and data:
                    formatted = []
                    for candle in data:
                        if isinstance(candle, list) and len(candle) >= 6:
                            formatted.append([
                                int(candle[0]),
                                float(candle[1]),
                                float(candle[2]),
                                float(candle[3]),
                                float(candle[4]),
                                float(candle[5])
                            ])
                    return formatted
            except Exception as e:
                logger.warning(f"Failed to fetch candles for {symbol} {interval}: {e}")
            return []

    async def get_ticker_price(self, symbol: str) -> Optional[float]:
        try:
            url = f"{self.binance_url}/fapi/v1/ticker/price"
            params = {'symbol': symbol}
            response = await self.client.get(url, params=params, timeout=5)
            response.raise_for_status()
            data = response.json()
            price = float(data.get('price', 0))
            return price if price > 0 else None
        except Exception as e:
            logger.warning(f"Failed to get ticker price for {symbol}: {e}")
            return None

    async def get_24h_ticker(self, symbol: str) -> Optional[Dict]:
        try:
            url = f"{self.binance_url}/fapi/v1/ticker/24hr"
            params = {'symbol': symbol}
            response = await self.client.get(url, params=params, timeout=5)
            response.raise_for_status()
            data = response.json()
            return {
                'price': float(data.get('lastPrice', 0)),
                'change_24h': float(data.get('priceChangePercent', 0)),
                'volume_24h': float(data.get('quoteVolume', 0)),
                'high_24h': float(data.get('highPrice', 0)),
                'low_24h': float(data.get('lowPrice', 0)),
            }
        except Exception as e:
            logger.warning(f"Failed to get 24h ticker for {symbol}: {e}")
            return None

    def calculate_rsi(self, prices: List[float], period: int = 14) -> float:
        if len(prices) < period + 1:
            return 50.0
        deltas = [prices[i] - prices[i-1] for i in range(1, len(prices))]
        gains = [d if d > 0 else 0 for d in deltas]
        losses = [-d if d < 0 else 0 for d in deltas]
        avg_gain = sum(gains[-period:]) / period
        avg_loss = sum(losses[-period:]) / period
        if avg_loss == 0:
            return 100.0
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))

    def calculate_ema(self, prices: List[float], period: int) -> float:
        if len(prices) < period:
            return sum(prices) / len(prices) if prices else 0
        multiplier = 2 / (period + 1)
        ema = sum(prices[:period]) / period
        for price in prices[period:]:
            ema = (price - ema) * multiplier + ema
        return ema

    def calculate_vwap(self, candles: List) -> float:
        if not candles:
            return 0
        total_pv = 0
        total_vol = 0
        for c in candles:
            typical_price = (c[2] + c[3] + c[4]) / 3
            vol = c[5]
            total_pv += typical_price * vol
            total_vol += vol
        return total_pv / total_vol if total_vol > 0 else 0

    def calculate_correlation(self, prices_a: List[float], prices_b: List[float]) -> float:
        n = min(len(prices_a), len(prices_b))
        if n < 5:
            return 0
        a = prices_a[-n:]
        b = prices_b[-n:]

        returns_a = [(a[i] - a[i-1]) / a[i-1] * 100 for i in range(1, len(a)) if a[i-1] != 0]
        returns_b = [(b[i] - b[i-1]) / b[i-1] * 100 for i in range(1, len(b)) if b[i-1] != 0]

        n_r = min(len(returns_a), len(returns_b))
        if n_r < 3:
            return 0

        ra = returns_a[-n_r:]
        rb = returns_b[-n_r:]

        mean_a = sum(ra) / n_r
        mean_b = sum(rb) / n_r

        cov = sum((ra[i] - mean_a) * (rb[i] - mean_b) for i in range(n_r)) / n_r
        std_a = (sum((x - mean_a) ** 2 for x in ra) / n_r) ** 0.5
        std_b = (sum((x - mean_b) ** 2 for x in rb) / n_r) ** 0.5

        if std_a == 0 or std_b == 0:
            return 0

        return cov / (std_a * std_b)

    def _calc_change(self, closes: List[float], lookback: int) -> float:
        if len(closes) < lookback + 1:
            return 0
        start = closes[-(lookback + 1)]
        end = closes[-1]
        return ((end - start) / start) * 100 if start != 0 else 0

    def _calc_momentum_acceleration(self, closes: List[float]) -> float:
        if len(closes) < 6:
            return 0
        recent_3 = self._calc_change(closes, 3)
        prior_3 = 0
        if len(closes) >= 7:
            prior_closes = closes[:-3]
            prior_3 = self._calc_change(prior_closes, 3) if len(prior_closes) >= 4 else 0
        return recent_3 - prior_3

    def _count_green_candles(self, candles: List, lookback: int = 5) -> int:
        recent = candles[-lookback:] if len(candles) >= lookback else candles
        return sum(1 for c in recent if c[4] > c[1])

    def _count_red_candles(self, candles: List, lookback: int = 5) -> int:
        recent = candles[-lookback:] if len(candles) >= lookback else candles
        return sum(1 for c in recent if c[4] < c[1])

    def detect_sol_momentum(self, fart_candles: List, sol_candles: List, sol_1m: List = None) -> Optional[Dict]:
        if len(fart_candles) < 20 or len(sol_candles) < 20:
            return None

        fart_closes = [c[4] for c in fart_candles]
        sol_closes = [c[4] for c in sol_candles]

        sol_3c = self._calc_change(sol_closes, 3)
        sol_6c = self._calc_change(sol_closes, 6)
        sol_10c = self._calc_change(sol_closes, 10)
        sol_20c = self._calc_change(sol_closes, 20)

        fart_3c = self._calc_change(fart_closes, 3)
        fart_6c = self._calc_change(fart_closes, 6)
        fart_10c = self._calc_change(fart_closes, 10)
        fart_20c = self._calc_change(fart_closes, 20)

        sol_1m_change = 0
        sol_1m_accel = 0
        sol_1m_green = 0
        if sol_1m and len(sol_1m) >= 10:
            sol_1m_closes = [c[4] for c in sol_1m]
            sol_1m_change = self._calc_change(sol_1m_closes, 5)
            sol_1m_accel = self._calc_momentum_acceleration(sol_1m_closes)
            sol_1m_green = self._count_green_candles(sol_1m, 5)

        correlation = self.calculate_correlation(fart_closes, sol_closes)
        sol_accel = self._calc_momentum_acceleration(sol_closes)
        fart_rsi = self.calculate_rsi(fart_closes)

        signal_type = None
        score = 0
        reasoning_parts = []
        direction = None

        sol_pumping = sol_3c > 0.3 or sol_6c > 0.5
        sol_accelerating_up = sol_accel > 0.1
        fart_lagging_up = fart_3c < sol_3c * 0.6
        fart_not_pumped_yet = fart_3c < 0.3 and fart_6c < sol_6c * 0.8

        if sol_pumping and (fart_lagging_up or fart_not_pumped_yet):
            signal_type = "SOL_PUMP_FART_LAG"
            direction = "LONG"
            score = 40

            lag_ratio = (sol_3c - fart_3c) / sol_3c if sol_3c > 0.1 else 0
            score += min(lag_ratio * 30, 30)
            reasoning_parts.append(f"$SOL pumping +{sol_3c:.2f}% (3c) / +{sol_6c:.2f}% (6c) but $FARTCOIN only +{fart_3c:.2f}% - catch-up entry")

            if sol_accelerating_up:
                score += 10
                reasoning_parts.append(f"$SOL momentum accelerating (+{sol_accel:.2f}%)")

            if sol_1m_change > 0.15:
                score += 10
                reasoning_parts.append(f"$SOL 1m surge: +{sol_1m_change:.2f}% (fresh move)")

            if sol_1m_green >= 4:
                score += 5
                reasoning_parts.append(f"$SOL {sol_1m_green}/5 green 1m candles")

            if sol_10c > 0.5 and fart_10c < sol_10c * 0.5:
                score += 10
                reasoning_parts.append(f"$FARTCOIN severely lagging over 10 candles ({fart_10c:.2f}% vs SOL {sol_10c:.2f}%)")

            if correlation > 0.5:
                score += 5
                reasoning_parts.append(f"High correlation ({correlation:.2f}) = FART will follow")

            green_count = self._count_green_candles(sol_candles, 5)
            if green_count >= 3:
                score += 5
                reasoning_parts.append(f"$SOL {green_count}/5 green candles (sustained)")

        sol_dumping = sol_3c < -0.3 or sol_6c < -0.5
        sol_accelerating_down = sol_accel < -0.1
        fart_hasnt_dumped = fart_3c > sol_3c * 0.4
        fart_still_holding = fart_3c > -0.2

        if sol_dumping and (fart_hasnt_dumped or fart_still_holding):
            signal_type = "SOL_DUMP_FART_DELAY"
            direction = "SHORT"
            score = 40

            delay_ratio = abs(sol_3c - fart_3c) / abs(sol_3c) if abs(sol_3c) > 0.1 else 0
            score += min(delay_ratio * 30, 30)
            reasoning_parts.append(f"$SOL dumping {sol_3c:.2f}% (3c) / {sol_6c:.2f}% (6c) but $FARTCOIN only {fart_3c:.2f}% - delayed dump coming")

            if sol_accelerating_down:
                score += 10
                reasoning_parts.append(f"$SOL dump accelerating ({sol_accel:.2f}%)")

            if sol_1m_change < -0.15:
                score += 10
                reasoning_parts.append(f"$SOL 1m drop: {sol_1m_change:.2f}% (fresh dump)")

            if sol_10c < -0.5 and fart_10c > sol_10c * 0.5:
                score += 10
                reasoning_parts.append(f"$FARTCOIN hasn't caught up to SOL 10c dump ({fart_10c:.2f}% vs SOL {sol_10c:.2f}%)")

            if correlation > 0.5:
                score += 5
                reasoning_parts.append(f"High correlation ({correlation:.2f}) = FART will follow down")

            red_count = self._count_red_candles(sol_candles, 5)
            if red_count >= 3:
                score += 5
                reasoning_parts.append(f"$SOL {red_count}/5 red candles (sustained selling)")

            if fart_rsi > 60:
                score += 5
                reasoning_parts.append(f"$FARTCOIN RSI still high ({fart_rsi:.0f}) - room to drop")

        if signal_type is None or score < 50:
            if sol_6c > 0.8 and fart_6c > sol_6c * 1.5 and self._count_green_candles(fart_candles, 3) >= 2:
                signal_type = "FART_AMPLIFIED_MOMENTUM"
                direction = "LONG"
                score = max(score, 55)
                reasoning_parts = [f"$FARTCOIN amplifying $SOL pump: FART +{fart_6c:.2f}% vs SOL +{sol_6c:.2f}% (beta > 1.5x) - riding the amplification"]

        if signal_type is None or score < 50:
            return None

        return {
            'signal_type': signal_type,
            'direction': direction,
            'score': score,
            'sol_3c': sol_3c,
            'sol_6c': sol_6c,
            'sol_10c': sol_10c,
            'sol_1m_change': sol_1m_change,
            'fart_3c': fart_3c,
            'fart_6c': fart_6c,
            'fart_10c': fart_10c,
            'correlation': correlation,
            'sol_acceleration': sol_accel,
            'fart_rsi': fart_rsi,
            'reasoning': " | ".join(reasoning_parts)
        }

    async def analyze_fartcoin(self) -> Optional[Dict]:
        await self.init()

        fart_5m, sol_5m, sol_1m, fart_15m, fart_1h = await asyncio.gather(
            self.fetch_candles(FARTCOIN_SYMBOL, '5m', 100),
            self.fetch_candles(SOL_SYMBOL, '5m', 100),
            self.fetch_candles(SOL_SYMBOL, '1m', 30),
            self.fetch_candles(FARTCOIN_SYMBOL, '15m', 50),
            self.fetch_candles(FARTCOIN_SYMBOL, '1h', 24),
        )

        if not fart_5m or not sol_5m:
            logger.warning("🐸 Failed to fetch FARTCOIN or SOL candle data")
            return None

        fart_ticker, sol_ticker = await asyncio.gather(
            self.get_24h_ticker(FARTCOIN_SYMBOL),
            self.get_24h_ticker(SOL_SYMBOL),
        )

        if not fart_ticker or not sol_ticker:
            logger.warning("🐸 Failed to fetch ticker data")
            return None

        current_price = fart_ticker['price']
        if current_price <= 0:
            return None

        fart_closes_5m = [c[4] for c in fart_5m]
        fart_closes_15m = [c[4] for c in fart_15m] if fart_15m else fart_closes_5m
        fart_closes_1h = [c[4] for c in fart_1h] if fart_1h else fart_closes_5m

        rsi_5m = self.calculate_rsi(fart_closes_5m)
        rsi_15m = self.calculate_rsi(fart_closes_15m)
        rsi_1h = self.calculate_rsi(fart_closes_1h)

        ema9 = self.calculate_ema(fart_closes_5m, 9)
        ema21 = self.calculate_ema(fart_closes_5m, 21)
        ema9_15m = self.calculate_ema(fart_closes_15m, 9)
        ema21_15m = self.calculate_ema(fart_closes_15m, 21)

        vwap = self.calculate_vwap(fart_5m[-20:])

        momentum = self.detect_sol_momentum(fart_5m, sol_5m, sol_1m)

        vol_avg = sum(c[5] for c in fart_5m[-20:]) / 20 if len(fart_5m) >= 20 else 0
        vol_current = fart_5m[-1][5] if fart_5m else 0
        vol_ratio = vol_current / vol_avg if vol_avg > 0 else 1.0

        fart_change_24h = fart_ticker['change_24h']
        sol_change_24h = sol_ticker['change_24h']
        volume_24h = fart_ticker['volume_24h']

        if volume_24h < 1_000_000:
            logger.info(f"🐸 $FARTCOIN volume too low: ${volume_24h:,.0f} (need $1M+)")
            return None

        signal = None

        if momentum:
            direction = momentum['direction']
            score = momentum['score']

            confirmations = []
            rejection_reasons = []

            if direction == "LONG":
                if rsi_5m > 80:
                    rejection_reasons.append(f"5m RSI overbought: {rsi_5m:.0f}")
                elif rsi_5m < 75:
                    confirmations.append(f"5m RSI {rsi_5m:.0f} has room to run")

                if rsi_15m > 82:
                    rejection_reasons.append(f"15m RSI overbought: {rsi_15m:.0f}")

                ema_dist = ((current_price - ema9) / ema9) * 100 if ema9 > 0 else 0
                if ema_dist > 5.0:
                    rejection_reasons.append(f"Price too far from EMA9: {ema_dist:.1f}%")
                elif ema_dist < 3.0:
                    confirmations.append(f"Price near EMA9 ({ema_dist:.1f}% away)")

                if ema9 > ema21:
                    confirmations.append("Bullish EMA alignment (9 > 21)")

                if vol_ratio > 1.3:
                    confirmations.append(f"Volume surge {vol_ratio:.1f}x")

                if current_price > vwap:
                    confirmations.append("Above VWAP")

                if len(rejection_reasons) > 0:
                    logger.info(f"🐸 LONG rejected: {', '.join(rejection_reasons)}")
                elif score >= 50:
                    tp_pct = 1.2
                    sl_pct = 0.5

                    if score >= 75:
                        tp_pct = 1.5
                    elif score >= 60:
                        tp_pct = 1.3

                    signal = {
                        'symbol': 'FARTCOIN/USDT',
                        'direction': 'LONG',
                        'entry_price': current_price,
                        'stop_loss': current_price * (1 - sl_pct / 100),
                        'take_profit': current_price * (1 + tp_pct / 100),
                        'take_profit_1': current_price * (1 + tp_pct / 100),
                        'take_profit_2': current_price * (1 + tp_pct * 1.5 / 100),
                        'take_profit_3': current_price * (1 + tp_pct * 2 / 100),
                        'confidence': min(int(score + len(confirmations) * 3), 95),
                        'leverage': FARTCOIN_LEVERAGE,
                        'trade_type': 'FARTCOIN_SIGNAL',
                        'signal_type': 'FARTCOIN',
                        'momentum_data': momentum,
                        'confirmations': confirmations,
                        'rsi_5m': rsi_5m,
                        'rsi_15m': rsi_15m,
                        'rsi_1h': rsi_1h,
                        'ema9': ema9,
                        'ema21': ema21,
                        'vwap': vwap,
                        'vol_ratio': vol_ratio,
                        '24h_change': fart_change_24h,
                        'sol_24h_change': sol_change_24h,
                        '24h_volume': volume_24h,
                        'reasoning': f"SOL BETA PLAY: {momentum['reasoning']} | {', '.join(confirmations)}"
                    }

            elif direction == "SHORT":
                if rsi_5m < 20:
                    rejection_reasons.append(f"5m RSI oversold: {rsi_5m:.0f}")
                elif rsi_5m > 25:
                    confirmations.append(f"5m RSI {rsi_5m:.0f} has room to drop")

                if rsi_15m < 18:
                    rejection_reasons.append(f"15m RSI oversold: {rsi_15m:.0f}")

                ema_dist = ((ema9 - current_price) / ema9) * 100 if ema9 > 0 else 0
                if ema_dist > 5.0:
                    rejection_reasons.append(f"Price already dumped too far from EMA: {ema_dist:.1f}%")

                if ema9 < ema21:
                    confirmations.append("Bearish EMA alignment (9 < 21)")

                if vol_ratio > 1.3:
                    confirmations.append(f"Selling volume {vol_ratio:.1f}x")

                if current_price < vwap:
                    confirmations.append("Below VWAP")

                if len(rejection_reasons) > 0:
                    logger.info(f"🐸 SHORT rejected: {', '.join(rejection_reasons)}")
                elif score >= 50:
                    tp_pct = 1.0
                    sl_pct = 0.5

                    if score >= 75:
                        tp_pct = 1.3
                    elif score >= 60:
                        tp_pct = 1.1

                    signal = {
                        'symbol': 'FARTCOIN/USDT',
                        'direction': 'SHORT',
                        'entry_price': current_price,
                        'stop_loss': current_price * (1 + sl_pct / 100),
                        'take_profit': current_price * (1 - tp_pct / 100),
                        'take_profit_1': current_price * (1 - tp_pct / 100),
                        'take_profit_2': current_price * (1 - tp_pct * 1.5 / 100),
                        'take_profit_3': current_price * (1 - tp_pct * 2 / 100),
                        'confidence': min(int(score + len(confirmations) * 3), 95),
                        'leverage': FARTCOIN_LEVERAGE,
                        'trade_type': 'FARTCOIN_SIGNAL',
                        'signal_type': 'FARTCOIN',
                        'momentum_data': momentum,
                        'confirmations': confirmations,
                        'rsi_5m': rsi_5m,
                        'rsi_15m': rsi_15m,
                        'rsi_1h': rsi_1h,
                        'ema9': ema9,
                        'ema21': ema21,
                        'vwap': vwap,
                        'vol_ratio': vol_ratio,
                        '24h_change': fart_change_24h,
                        'sol_24h_change': sol_change_24h,
                        '24h_volume': volume_24h,
                        'reasoning': f"SOL BETA PLAY: {momentum['reasoning']} | {', '.join(confirmations)}"
                    }

        if signal is None:
            signal = await self._check_standalone_ta(
                current_price, fart_closes_5m, fart_closes_15m, fart_closes_1h,
                fart_5m, rsi_5m, rsi_15m, rsi_1h, ema9, ema21, ema9_15m, ema21_15m,
                vwap, vol_ratio, fart_change_24h, sol_change_24h, volume_24h
            )

        if signal:
            signal = await self._ai_validate_fartcoin(signal)

        return signal

    async def _check_standalone_ta(
        self, current_price, closes_5m, closes_15m, closes_1h,
        candles_5m, rsi_5m, rsi_15m, rsi_1h, ema9, ema21, ema9_15m, ema21_15m,
        vwap, vol_ratio, fart_change_24h, sol_change_24h, volume_24h
    ) -> Optional[Dict]:
        confirmations = []

        bullish_trend = ema9 > ema21 and ema9_15m > ema21_15m
        bearish_trend = ema9 < ema21 and ema9_15m < ema21_15m

        if bullish_trend and rsi_5m < 65 and rsi_15m < 68:
            ema_dist = ((current_price - ema9) / ema9) * 100 if ema9 > 0 else 0
            if ema_dist < 2.0 and ema_dist > -1.0:
                confirmations.append(f"Bullish trend with pullback to EMA ({ema_dist:.1f}%)")
                if vol_ratio > 1.3:
                    confirmations.append(f"Volume confirmation {vol_ratio:.1f}x")
                if current_price > vwap:
                    confirmations.append("Above VWAP")
                if rsi_1h > 45 and rsi_1h < 65:
                    confirmations.append(f"1H RSI healthy: {rsi_1h:.0f}")

                if len(confirmations) >= 3:
                    tp_pct = 0.8
                    sl_pct = 0.4
                    return {
                        'symbol': 'FARTCOIN/USDT',
                        'direction': 'LONG',
                        'entry_price': current_price,
                        'stop_loss': current_price * (1 - sl_pct / 100),
                        'take_profit': current_price * (1 + tp_pct / 100),
                        'take_profit_1': current_price * (1 + tp_pct / 100),
                        'take_profit_2': current_price * (1 + tp_pct * 1.5 / 100),
                        'take_profit_3': current_price * (1 + tp_pct * 2 / 100),
                        'confidence': 65 + len(confirmations) * 5,
                        'leverage': FARTCOIN_LEVERAGE,
                        'trade_type': 'FARTCOIN_SIGNAL',
                        'signal_type': 'FARTCOIN',
                        'momentum_data': None,
                        'confirmations': confirmations,
                        'rsi_5m': rsi_5m,
                        'rsi_15m': rsi_15m,
                        'rsi_1h': rsi_1h,
                        'ema9': ema9,
                        'ema21': ema21,
                        'vwap': vwap,
                        'vol_ratio': vol_ratio,
                        '24h_change': fart_change_24h,
                        'sol_24h_change': sol_change_24h,
                        '24h_volume': volume_24h,
                        'reasoning': f"TREND PULLBACK LONG: {' | '.join(confirmations)}"
                    }

        confirmations = []
        if bearish_trend and rsi_5m > 35 and rsi_15m > 32:
            ema_dist = ((ema9 - current_price) / ema9) * 100 if ema9 > 0 else 0
            if ema_dist < 2.0 and ema_dist > -1.0:
                confirmations.append(f"Bearish trend with bounce to EMA ({ema_dist:.1f}%)")
                if vol_ratio > 1.3:
                    confirmations.append(f"Volume confirmation {vol_ratio:.1f}x")
                if current_price < vwap:
                    confirmations.append("Below VWAP")
                if rsi_1h > 35 and rsi_1h < 55:
                    confirmations.append(f"1H RSI confirming weakness: {rsi_1h:.0f}")

                if len(confirmations) >= 3:
                    tp_pct = 0.8
                    sl_pct = 0.4
                    return {
                        'symbol': 'FARTCOIN/USDT',
                        'direction': 'SHORT',
                        'entry_price': current_price,
                        'stop_loss': current_price * (1 + sl_pct / 100),
                        'take_profit': current_price * (1 - tp_pct / 100),
                        'take_profit_1': current_price * (1 - tp_pct / 100),
                        'take_profit_2': current_price * (1 - tp_pct * 1.5 / 100),
                        'take_profit_3': current_price * (1 - tp_pct * 2 / 100),
                        'confidence': 65 + len(confirmations) * 5,
                        'leverage': FARTCOIN_LEVERAGE,
                        'trade_type': 'FARTCOIN_SIGNAL',
                        'signal_type': 'FARTCOIN',
                        'momentum_data': None,
                        'confirmations': confirmations,
                        'rsi_5m': rsi_5m,
                        'rsi_15m': rsi_15m,
                        'rsi_1h': rsi_1h,
                        'ema9': ema9,
                        'ema21': ema21,
                        'vwap': vwap,
                        'vol_ratio': vol_ratio,
                        '24h_change': fart_change_24h,
                        'sol_24h_change': sol_change_24h,
                        '24h_volume': volume_24h,
                        'reasoning': f"TREND BOUNCE SHORT: {' | '.join(confirmations)}"
                    }

        return None

    async def _ai_validate_fartcoin(self, signal_data: Dict) -> Optional[Dict]:
        try:
            from app.services.top_gainers_signals import call_gemini_signal
        except ImportError:
            logger.warning("🐸 Cannot import Gemini - skipping AI validation")
            return signal_data

        try:
            direction = signal_data['direction']
            entry = signal_data['entry_price']
            sl = signal_data['stop_loss']
            tp = signal_data['take_profit_1']
            rsi = signal_data['rsi_5m']
            momentum = signal_data.get('momentum_data')

            momentum_context = ""
            if momentum:
                momentum_context = f"""
SOL MOMENTUM DATA (KEY EDGE):
- Signal type: {momentum.get('signal_type', 'N/A')}
- SOL 3-candle change: {momentum.get('sol_3c', 0):.2f}%
- SOL 6-candle change: {momentum.get('sol_6c', 0):.2f}%
- SOL 10-candle change: {momentum.get('sol_10c', 0):.2f}%
- SOL 1m surge: {momentum.get('sol_1m_change', 0):.2f}%
- SOL acceleration: {momentum.get('sol_acceleration', 0):.3f}%
- FART 3-candle change: {momentum.get('fart_3c', 0):.2f}%
- FART 6-candle change: {momentum.get('fart_6c', 0):.2f}%
- FART 10-candle change: {momentum.get('fart_10c', 0):.2f}%
- Correlation: {momentum.get('correlation', 0):.3f}
- Momentum score: {momentum.get('score', 0)}/100

STRATEGY CONTEXT:
- FARTCOIN amplifies SOL moves (pumps harder when SOL pumps)
- FARTCOIN lags SOL dumps (dumps with delay after SOL drops)
- For LONGS: We are entering FART because SOL is pumping and FART hasn't caught up yet
- For SHORTS: We are shorting FART because SOL is dumping and FART hasn't reacted yet
"""

            prompt = f"""You are a FARTCOIN specialist analyst. FARTCOIN is a Solana memecoin with a KEY behavioral pattern:
1. When SOL pumps, FARTCOIN pumps HARDER (amplified beta)
2. When SOL dumps, FARTCOIN dumps with LATENCY (delayed reaction)

We exploit this by entering FARTCOIN trades based on SOL's momentum BEFORE FART catches up.

SIGNAL: {direction} $FARTCOIN at ${entry:.8f}
Stop Loss: ${sl:.8f}
Take Profit: ${tp:.8f}
Leverage: {FARTCOIN_LEVERAGE}x

TECHNICAL DATA:
- RSI (5m): {signal_data['rsi_5m']:.0f}
- RSI (15m): {signal_data['rsi_15m']:.0f}
- RSI (1h): {signal_data['rsi_1h']:.0f}
- EMA9: ${signal_data['ema9']:.8f}
- EMA21: ${signal_data['ema21']:.8f}
- VWAP: ${signal_data['vwap']:.8f}
- Volume ratio: {signal_data['vol_ratio']:.1f}x
- 24h change: {signal_data['24h_change']:.2f}%
- SOL 24h change: {signal_data['sol_24h_change']:.2f}%
- 24h volume: ${signal_data['24h_volume']:,.0f}
{momentum_context}
CONFIRMATIONS: {', '.join(signal_data.get('confirmations', []))}

At 50x leverage:
- TP hit = +{abs((tp - entry) / entry) * 100 * FARTCOIN_LEVERAGE:.0f}% ROI
- SL hit = -{abs((sl - entry) / entry) * 100 * FARTCOIN_LEVERAGE:.0f}% ROI

Should this trade be executed? Consider:
1. Is SOL's momentum strong enough to pull FART along?
2. Has FART genuinely not reacted yet (is the lag real)?
3. Are the TA confirmations strong enough for 50x leverage?
4. Is the risk/reward acceptable?
5. Any red flags (FART already moved, SOL reversing, etc)?

Respond in JSON:
{{"approved": true/false, "confidence": 1-10, "reasoning": "brief explanation", "adjusted_tp": null_or_price, "adjusted_sl": null_or_price}}"""

            result = await call_gemini_signal(prompt)

            if result and isinstance(result, str):
                try:
                    cleaned = result.strip()
                    if '```json' in cleaned:
                        cleaned = cleaned.split('```json')[1].split('```')[0].strip()
                    elif '```' in cleaned:
                        cleaned = cleaned.split('```')[1].split('```')[0].strip()

                    ai_result = json.loads(cleaned)

                    if not ai_result.get('approved', False):
                        logger.info(f"🐸 AI REJECTED $FARTCOIN {direction}: {ai_result.get('reasoning', 'No reason')}")
                        return None

                    ai_confidence = ai_result.get('confidence', 5)
                    if ai_confidence < 5:
                        logger.info(f"🐸 AI confidence too low ({ai_confidence}/10) - rejecting")
                        return None

                    if ai_result.get('adjusted_tp') and ai_result['adjusted_tp'] > 0:
                        signal_data['take_profit_1'] = ai_result['adjusted_tp']
                        signal_data['take_profit'] = ai_result['adjusted_tp']
                    if ai_result.get('adjusted_sl') and ai_result['adjusted_sl'] > 0:
                        signal_data['stop_loss'] = ai_result['adjusted_sl']

                    signal_data['ai_confidence'] = ai_confidence
                    signal_data['ai_reasoning'] = ai_result.get('reasoning', '')
                    signal_data['confidence'] = min(signal_data.get('confidence', 50), ai_confidence * 10)

                    logger.info(f"🐸 AI APPROVED $FARTCOIN {direction} (confidence: {ai_confidence}/10)")
                    return signal_data

                except json.JSONDecodeError:
                    logger.warning(f"🐸 AI returned non-JSON response, proceeding with signal")
                    return signal_data

            return signal_data

        except Exception as e:
            logger.error(f"🐸 AI validation error: {e}")
            return signal_data

    async def get_status(self) -> Dict:
        await self.init()

        fart_ticker = await self.get_24h_ticker(FARTCOIN_SYMBOL)
        sol_ticker = await self.get_24h_ticker(SOL_SYMBOL)

        fart_5m = await self.fetch_candles(FARTCOIN_SYMBOL, '5m', 50)
        sol_5m = await self.fetch_candles(SOL_SYMBOL, '5m', 50)

        correlation = 0
        if fart_5m and sol_5m:
            fart_closes = [c[4] for c in fart_5m]
            sol_closes = [c[4] for c in sol_5m]
            correlation = self.calculate_correlation(fart_closes, sol_closes)

        fart_closes = [c[4] for c in fart_5m] if fart_5m else []
        rsi = self.calculate_rsi(fart_closes) if len(fart_closes) > 14 else 50
        ema9 = self.calculate_ema(fart_closes, 9) if fart_closes else 0
        ema21 = self.calculate_ema(fart_closes, 21) if fart_closes else 0

        trend = "BULLISH" if ema9 > ema21 else ("BEARISH" if ema9 < ema21 else "NEUTRAL")

        return {
            'enabled': _fartcoin_enabled,
            'fart_price': fart_ticker['price'] if fart_ticker else 0,
            'fart_change_24h': fart_ticker['change_24h'] if fart_ticker else 0,
            'fart_volume_24h': fart_ticker['volume_24h'] if fart_ticker else 0,
            'sol_price': sol_ticker['price'] if sol_ticker else 0,
            'sol_change_24h': sol_ticker['change_24h'] if sol_ticker else 0,
            'correlation': correlation,
            'rsi': rsi,
            'trend': trend,
            'daily_signals': _fartcoin_daily_count,
            'max_daily': MAX_FARTCOIN_DAILY_SIGNALS,
            'cooldown_active': not check_fartcoin_cooldown(),
            'leverage': FARTCOIN_LEVERAGE,
        }


def format_fartcoin_signal_message(signal_data: Dict) -> str:
    direction = signal_data['direction']
    entry = signal_data['entry_price']
    sl = signal_data['stop_loss']
    tp1 = signal_data['take_profit_1']
    leverage = FARTCOIN_LEVERAGE

    ticker = "$FARTCOIN"

    if direction == 'LONG':
        tp_pct = ((tp1 - entry) / entry) * 100 * leverage
        sl_pct = ((entry - sl) / entry) * 100 * leverage
        direction_emoji = "🟢 LONG"
    else:
        tp_pct = ((entry - tp1) / entry) * 100 * leverage
        sl_pct = ((sl - entry) / entry) * 100 * leverage
        direction_emoji = "🔴 SHORT"

    rr_ratio = tp_pct / sl_pct if sl_pct > 0 else 1.0

    momentum_text = ""
    momentum = signal_data.get('momentum_data')
    if momentum:
        sig_type = momentum.get('signal_type', '')
        if 'PUMP' in sig_type or 'AMPLIFIED' in sig_type:
            edge_label = "SOL PUMP → FART CATCH-UP"
        elif 'DUMP' in sig_type:
            edge_label = "SOL DUMP → FART DELAYED DROP"
        else:
            edge_label = sig_type

        momentum_text = f"""
<b>🔗 SOL Beta Edge</b>
├ Type: <b>{edge_label}</b>
├ $SOL move: <b>{momentum.get('sol_3c', 0):+.2f}%</b> (3c) / <b>{momentum.get('sol_6c', 0):+.2f}%</b> (6c)
├ $FART lag: <b>{momentum.get('fart_3c', 0):+.2f}%</b> (3c) / <b>{momentum.get('fart_6c', 0):+.2f}%</b> (6c)
├ Correlation: <b>{momentum.get('correlation', 0):.2f}</b>
└ Score: <b>{momentum.get('score', 0)}/100</b>
"""

    ai_text = ""
    if signal_data.get('ai_reasoning'):
        ai_text = f"\n🤖 <b>AI:</b> {signal_data['ai_reasoning']}"

    confirmations_text = ""
    confs = signal_data.get('confirmations', [])
    if confs:
        confirmations_text = "\n<b>✅ Confirmations</b>\n" + "\n".join(f"├ {c}" for c in confs)

    msg = f"""
┏━━━━━━━━━━━━━━━━━━━━━━━┓
  🐸 <b>FARTCOIN ALERT</b> 🐸
┗━━━━━━━━━━━━━━━━━━━━━━━┛

{direction_emoji} <b>{ticker}</b>
━━━━━━━━━━━━━━━━━━━━━━━━━━

<b>📊 Market Data</b>
├ 24h Change: <b>{signal_data.get('24h_change', 0):+.2f}%</b>
├ SOL 24h: <b>{signal_data.get('sol_24h_change', 0):+.2f}%</b>
└ Volume: <b>${signal_data.get('24h_volume', 0):,.0f}</b>
{momentum_text}
<b>🎯 Trade Setup</b>
├ Entry: <b>${entry:.8f}</b>
├ TP: <b>${tp1:.8f}</b> (+{tp_pct:.0f}% @ {leverage}x) 🎯
└ SL: <b>${sl:.8f}</b> (-{sl_pct:.0f}% @ {leverage}x)

<b>⚡ Risk Management</b>
├ Leverage: <b>{leverage}x</b>
└ R/R: <b>{rr_ratio:.1f}:1</b>
{confirmations_text}
<b>💡 Analysis</b>
{signal_data.get('reasoning', 'N/A')}
{ai_text}

⚠️ <b>50x LEVERAGE - HIGH RISK</b>
<i>Auto-executing for enabled users...</i>
"""
    return msg.strip()


def format_fartcoin_status_message(status: Dict) -> str:
    enabled_text = "🟢 ENABLED" if status['enabled'] else "🔴 DISABLED"
    corr_emoji = "🔗" if abs(status['correlation']) > 0.5 else "🔓"

    msg = f"""
🐸 <b>FARTCOIN SCANNER STATUS</b>

<b>Scanner:</b> {enabled_text}
<b>Leverage:</b> {status['leverage']}x

<b>📊 $FARTCOIN</b>
├ Price: <b>${status['fart_price']:.8f}</b>
├ 24h: <b>{status['fart_change_24h']:+.2f}%</b>
├ Volume: <b>${status['fart_volume_24h']:,.0f}</b>
├ RSI: <b>{status['rsi']:.0f}</b>
└ Trend: <b>{status['trend']}</b>

<b>📊 $SOL</b>
├ Price: <b>${status['sol_price']:.2f}</b>
└ 24h: <b>{status['sol_change_24h']:+.2f}%</b>

{corr_emoji} <b>Correlation:</b> {status['correlation']:.3f}

<b>📈 Signals Today:</b> {status['daily_signals']}/{status['max_daily']}
<b>⏳ Cooldown:</b> {"Active" if status['cooldown_active'] else "Ready"}
"""
    return msg.strip()


async def broadcast_fartcoin_signal(db_session, bot):
    if not is_fartcoin_enabled():
        return

    if not check_fartcoin_cooldown():
        logger.debug("🐸 FARTCOIN on cooldown")
        return

    if not check_fartcoin_daily_limit():
        logger.info("🐸 FARTCOIN daily signal limit reached")
        return

    try:
        from app.models import User, UserPreference, Signal, Trade
        from app.services.bitunix_trader import execute_bitunix_trade
        from app.services.ai_signal_filter import should_broadcast_signal
        from app.services.social_signals import check_global_signal_limit, increment_global_signal_count
        import hashlib
        from sqlalchemy import text

        FARTCOIN_ALLOWED_IDS = {1, 6}

        users = db_session.query(User).join(UserPreference).filter(
            User.id.in_(FARTCOIN_ALLOWED_IDS)
        ).all()

        if not users:
            logger.debug("🐸 No authorized users for FARTCOIN signals")
            return

        logger.info("🐸 ═══════════════════════════════════════════")
        logger.info("🐸 FARTCOIN SCANNER - Analyzing SOL correlation")
        logger.info("🐸 ═══════════════════════════════════════════")

        scanner = FartcoinScanner()
        await scanner.init()

        try:
            signal_data = await scanner.analyze_fartcoin()

            if not signal_data:
                logger.info("🐸 No FARTCOIN signal found this cycle")
                return

            if not check_global_signal_limit():
                logger.warning("🐸 Global daily signal limit reached")
                return

            ai_approved, ai_analysis_text = await should_broadcast_signal(signal_data)
            if not ai_approved:
                logger.warning(f"🐸 AI FINAL FILTER REJECTED $FARTCOIN {signal_data['direction']}")
                return

            lock_key = f"FARTCOIN:{signal_data['direction']}"
            lock_id = int(hashlib.md5(lock_key.encode()).hexdigest()[:16], 16) % (2**63 - 1)

            result = db_session.execute(text(f"SELECT pg_try_advisory_lock({lock_id})"))
            lock_acquired = result.scalar()

            if not lock_acquired:
                logger.warning("🐸 Could not acquire lock for FARTCOIN signal")
                return

            try:
                from datetime import timedelta
                recent_cutoff = datetime.utcnow() - timedelta(hours=6)
                existing = db_session.query(Signal).filter(
                    Signal.symbol == 'FARTCOIN/USDT',
                    Signal.direction == signal_data['direction'],
                    Signal.created_at >= recent_cutoff
                ).first()

                if existing:
                    logger.warning(f"🐸 DUPLICATE PREVENTED: FARTCOIN {signal_data['direction']} within 6h")
                    return

                open_positions = db_session.query(Trade).filter(
                    Trade.symbol == 'FARTCOIN/USDT',
                    Trade.status == 'open'
                ).count()

                if open_positions > 0:
                    logger.warning(f"🐸 FARTCOIN already has {open_positions} open position(s)")
                    return

                signal = Signal(
                    symbol='FARTCOIN/USDT',
                    direction=signal_data['direction'],
                    entry_price=signal_data['entry_price'],
                    stop_loss=signal_data['stop_loss'],
                    take_profit=signal_data.get('take_profit'),
                    take_profit_1=signal_data.get('take_profit_1'),
                    take_profit_2=signal_data.get('take_profit_2'),
                    take_profit_3=signal_data.get('take_profit_3'),
                    confidence=signal_data['confidence'],
                    reasoning=signal_data['reasoning'],
                    signal_type='FARTCOIN',
                    timeframe='5m',
                    created_at=datetime.utcnow()
                )
                db_session.add(signal)
                db_session.flush()
                db_session.commit()
                db_session.refresh(signal)

                record_fartcoin_signal()
                increment_global_signal_count()

                logger.info(f"🐸 FARTCOIN SIGNAL CREATED: {signal.direction} @ ${signal.entry_price}")

                signal_text = format_fartcoin_signal_message(signal_data)
                if ai_analysis_text:
                    signal_text += f"\n\n{ai_analysis_text}"

                semaphore = asyncio.Semaphore(5)

                async def execute_for_user(user):
                    from app.database import SessionLocal
                    user_db = SessionLocal()
                    try:
                        async with semaphore:
                            await bot.send_message(
                                user.telegram_id,
                                signal_text,
                                parse_mode="HTML"
                            )

                            fresh_user = user_db.query(User).filter(User.id == user.id).first()
                            if fresh_user and fresh_user.preferences:
                                prefs = fresh_user.preferences
                                if prefs.bitunix_api_key and prefs.bitunix_api_secret:
                                    trade_result = await execute_bitunix_trade(
                                        signal, fresh_user, user_db,
                                        trade_type='FARTCOIN_SIGNAL',
                                        leverage_override=FARTCOIN_LEVERAGE
                                    )
                                    if trade_result:
                                        logger.info(f"🐸 Trade executed for user {user.id}")
                                    else:
                                        logger.warning(f"🐸 Trade execution failed for user {user.id}")
                    except Exception as e:
                        logger.error(f"🐸 Error executing for user {user.id}: {e}")
                    finally:
                        user_db.close()

                tasks = [execute_for_user(user) for user in users]
                await asyncio.gather(*tasks, return_exceptions=True)

                logger.info(f"🐸 FARTCOIN signal broadcast complete - {len(users)} users notified")

            finally:
                db_session.execute(text(f"SELECT pg_advisory_unlock({lock_id})"))

        finally:
            await scanner.close()

    except Exception as e:
        logger.error(f"🐸 FARTCOIN broadcast error: {e}", exc_info=True)
