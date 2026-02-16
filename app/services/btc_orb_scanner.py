"""
BTC ORB + FVG Scalper - Opening Range Breakout with Fair Value Gap Strategy

Strategy:
1. Detect 15min Opening Range at Asia (00:00 UTC) and New York (13:30 UTC) sessions
2. Use 1m candles to determine if price formed a HIGH or LOW first (direction bias)
3. Apply Fibonacci retracement: top-down if high first, bottom-up if low first
4. Detect Fair Value Gaps (FVGs) within the Fibonacci zone
5. Wait for price to retest the Fib level or FVG to execute the trade
6. Execute as BTC scalp with tight TP/SL based on ORB range
"""

import logging
import asyncio
import httpx
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

BTC_SYMBOL = "BTCUSDT"
BINANCE_FUTURES_URL = "https://fapi.binance.com"
API_SEMAPHORE = asyncio.Semaphore(5)

ASIA_OPEN_HOUR = 0
ASIA_OPEN_MINUTE = 0
NY_OPEN_HOUR = 13
NY_OPEN_MINUTE = 30

ORB_MINUTES = 15
RETEST_WINDOW_MINUTES = 90
FIB_LEVELS = [0.382, 0.5, 0.618, 0.786]
ENTRY_FIB_MIN = 0.5
ENTRY_FIB_MAX = 0.786

BTC_ORB_LEVERAGE = 25
BTC_ORB_COOLDOWN_MINUTES = 240
MAX_BTC_ORB_DAILY_SIGNALS = 2
BTC_ORB_SESSIONS_ENABLED = {"ASIA": True, "NY": True}

_btc_orb_enabled = False
_btc_orb_last_signal_time = None
_btc_orb_daily_count = 0
_btc_orb_daily_reset = None

_active_orb_setup: Optional[Dict] = None
_last_orb_session: Optional[str] = None


def is_btc_orb_enabled() -> bool:
    return _btc_orb_enabled


def set_btc_orb_enabled(enabled: bool):
    global _btc_orb_enabled
    _btc_orb_enabled = enabled
    logger.info(f"📊 BTC ORB scanner {'ENABLED' if enabled else 'DISABLED'}")


def get_btc_orb_leverage() -> int:
    return BTC_ORB_LEVERAGE


def set_btc_orb_leverage(leverage: int):
    global BTC_ORB_LEVERAGE
    BTC_ORB_LEVERAGE = max(5, min(100, leverage))
    logger.info(f"📊 BTC ORB leverage set to {BTC_ORB_LEVERAGE}x")


def get_btc_orb_max_daily() -> int:
    return MAX_BTC_ORB_DAILY_SIGNALS


def set_btc_orb_max_daily(limit: int):
    global MAX_BTC_ORB_DAILY_SIGNALS
    MAX_BTC_ORB_DAILY_SIGNALS = max(1, min(10, limit))
    logger.info(f"📊 BTC ORB max daily signals set to {MAX_BTC_ORB_DAILY_SIGNALS}")


def get_btc_orb_sessions() -> dict:
    return BTC_ORB_SESSIONS_ENABLED.copy()


def toggle_btc_orb_session(session: str) -> bool:
    global BTC_ORB_SESSIONS_ENABLED
    if session in BTC_ORB_SESSIONS_ENABLED:
        BTC_ORB_SESSIONS_ENABLED[session] = not BTC_ORB_SESSIONS_ENABLED[session]
        logger.info(f"📊 BTC ORB {session} session {'ENABLED' if BTC_ORB_SESSIONS_ENABLED[session] else 'DISABLED'}")
        return BTC_ORB_SESSIONS_ENABLED[session]
    return False


def get_btc_orb_cooldown() -> int:
    return BTC_ORB_COOLDOWN_MINUTES


def set_btc_orb_cooldown(minutes: int):
    global BTC_ORB_COOLDOWN_MINUTES
    BTC_ORB_COOLDOWN_MINUTES = max(30, min(480, minutes))
    logger.info(f"📊 BTC ORB cooldown set to {BTC_ORB_COOLDOWN_MINUTES}min")


def get_btc_orb_daily_count() -> int:
    return _btc_orb_daily_count


def check_btc_orb_cooldown() -> bool:
    global _btc_orb_last_signal_time
    if _btc_orb_last_signal_time is None:
        return True
    elapsed = (datetime.utcnow() - _btc_orb_last_signal_time).total_seconds()
    return elapsed >= (BTC_ORB_COOLDOWN_MINUTES * 60)


def check_btc_orb_daily_limit() -> bool:
    global _btc_orb_daily_count, _btc_orb_daily_reset
    now = datetime.utcnow()
    if _btc_orb_daily_reset is None or now.date() > _btc_orb_daily_reset.date():
        _btc_orb_daily_count = 0
        _btc_orb_daily_reset = now
    return _btc_orb_daily_count < MAX_BTC_ORB_DAILY_SIGNALS


def record_btc_orb_signal():
    global _btc_orb_last_signal_time, _btc_orb_daily_count
    _btc_orb_last_signal_time = datetime.utcnow()
    _btc_orb_daily_count += 1


class BTCOrbScanner:
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

    async def fetch_candles(self, symbol: str, interval: str, limit: int = 100,
                            start_time: Optional[int] = None, end_time: Optional[int] = None) -> List:
        async with API_SEMAPHORE:
            try:
                url = f"{self.binance_url}/fapi/v1/klines"
                params = {
                    'symbol': symbol,
                    'interval': interval,
                    'limit': limit
                }
                if start_time:
                    params['startTime'] = start_time
                if end_time:
                    params['endTime'] = end_time
                response = await self.client.get(url, params=params, timeout=10)
                response.raise_for_status()
                data = response.json()

                if isinstance(data, list) and data:
                    formatted = []
                    for candle in data:
                        if isinstance(candle, list) and len(candle) >= 6:
                            formatted.append({
                                'timestamp': int(candle[0]),
                                'open': float(candle[1]),
                                'high': float(candle[2]),
                                'low': float(candle[3]),
                                'close': float(candle[4]),
                                'volume': float(candle[5])
                            })
                    return formatted
            except Exception as e:
                logger.warning(f"Failed to fetch candles for {symbol} {interval}: {e}")
            return []

    async def get_current_price(self) -> Optional[float]:
        try:
            url = f"{self.binance_url}/fapi/v1/ticker/price"
            params = {'symbol': BTC_SYMBOL}
            response = await self.client.get(url, params=params, timeout=5)
            response.raise_for_status()
            data = response.json()
            price = float(data.get('price', 0))
            return price if price > 0 else None
        except Exception as e:
            logger.warning(f"Failed to get BTC price: {e}")
            return None

    def get_current_session(self) -> Optional[str]:
        now = datetime.utcnow()
        asia_open = now.replace(hour=ASIA_OPEN_HOUR, minute=ASIA_OPEN_MINUTE, second=0, microsecond=0)
        ny_open = now.replace(hour=NY_OPEN_HOUR, minute=NY_OPEN_MINUTE, second=0, microsecond=0)

        asia_window_end = asia_open + timedelta(minutes=ORB_MINUTES + RETEST_WINDOW_MINUTES)
        ny_window_end = ny_open + timedelta(minutes=ORB_MINUTES + RETEST_WINDOW_MINUTES)

        if asia_open <= now <= asia_window_end:
            if BTC_ORB_SESSIONS_ENABLED.get("ASIA", True):
                return "ASIA"
            return None
        elif ny_open <= now <= ny_window_end:
            if BTC_ORB_SESSIONS_ENABLED.get("NY", True):
                return "NY"
            return None
        return None

    def is_in_orb_formation(self) -> bool:
        now = datetime.utcnow()
        asia_open = now.replace(hour=ASIA_OPEN_HOUR, minute=ASIA_OPEN_MINUTE, second=0, microsecond=0)
        ny_open = now.replace(hour=NY_OPEN_HOUR, minute=NY_OPEN_MINUTE, second=0, microsecond=0)

        asia_orb_end = asia_open + timedelta(minutes=ORB_MINUTES)
        ny_orb_end = ny_open + timedelta(minutes=ORB_MINUTES)

        in_asia = (asia_open <= now <= asia_orb_end) and BTC_ORB_SESSIONS_ENABLED.get("ASIA", True)
        in_ny = (ny_open <= now <= ny_orb_end) and BTC_ORB_SESSIONS_ENABLED.get("NY", True)

        return in_asia or in_ny

    def get_session_open_time(self, session: str) -> datetime:
        now = datetime.utcnow()
        if session == "ASIA":
            return now.replace(hour=ASIA_OPEN_HOUR, minute=ASIA_OPEN_MINUTE, second=0, microsecond=0)
        else:
            return now.replace(hour=NY_OPEN_HOUR, minute=NY_OPEN_MINUTE, second=0, microsecond=0)

    async def build_opening_range(self, session: str) -> Optional[Dict]:
        session_open = self.get_session_open_time(session)
        orb_end = session_open + timedelta(minutes=ORB_MINUTES)
        now = datetime.utcnow()

        if now < orb_end:
            logger.debug(f"📊 ORB still forming for {session} session, waiting...")
            return None

        start_ms = int(session_open.timestamp() * 1000)
        end_ms = int(orb_end.timestamp() * 1000)

        candles_1m = await self.fetch_candles(BTC_SYMBOL, '1m', limit=20,
                                              start_time=start_ms, end_time=end_ms)

        if not candles_1m or len(candles_1m) < 5:
            logger.warning(f"📊 Not enough 1m candles for ORB: got {len(candles_1m)}")
            return None

        orb_high = max(c['high'] for c in candles_1m)
        orb_low = min(c['low'] for c in candles_1m)
        orb_range = orb_high - orb_low

        if orb_range <= 0:
            logger.warning("📊 ORB range is zero")
            return None

        orb_range_pct = (orb_range / orb_low) * 100
        if orb_range_pct < 0.05:
            logger.info(f"📊 ORB range too tight ({orb_range_pct:.3f}%) - no clear setup")
            return None

        if orb_range_pct > 1.5:
            logger.info(f"📊 ORB range too wide ({orb_range_pct:.3f}%) - too volatile for scalp")
            return None

        high_first = self._detect_direction_bias(candles_1m, orb_high, orb_low)

        fib_levels = self._calculate_fib_levels(orb_high, orb_low, high_first)

        logger.info(f"📊 ORB BUILT | {session} | High: ${orb_high:.2f} | Low: ${orb_low:.2f} | "
                    f"Range: ${orb_range:.2f} ({orb_range_pct:.3f}%) | "
                    f"{'HIGH first → BEARISH bias (top-down fib)' if high_first else 'LOW first → BULLISH bias (bottom-up fib)'}")

        return {
            'session': session,
            'orb_high': orb_high,
            'orb_low': orb_low,
            'orb_range': orb_range,
            'orb_range_pct': orb_range_pct,
            'high_first': high_first,
            'direction': 'SHORT' if high_first else 'LONG',
            'fib_levels': fib_levels,
            'orb_formed_at': orb_end,
            'retest_deadline': orb_end + timedelta(minutes=RETEST_WINDOW_MINUTES),
            'candles_1m': candles_1m
        }

    def _detect_direction_bias(self, candles: List[Dict], orb_high: float, orb_low: float) -> bool:
        high_threshold = orb_high - (orb_high - orb_low) * 0.15
        low_threshold = orb_low + (orb_high - orb_low) * 0.15

        first_high_time = None
        first_low_time = None

        for c in candles:
            if first_high_time is None and c['high'] >= high_threshold:
                first_high_time = c['timestamp']
            if first_low_time is None and c['low'] <= low_threshold:
                first_low_time = c['timestamp']

        if first_high_time is not None and first_low_time is not None:
            return first_high_time < first_low_time

        if first_high_time is not None:
            return True

        return False

    def _calculate_fib_levels(self, orb_high: float, orb_low: float, high_first: bool) -> Dict:
        orb_range = orb_high - orb_low

        if high_first:
            levels = {}
            for fib in FIB_LEVELS:
                price = orb_high - (orb_range * fib)
                levels[fib] = price
            levels['direction'] = 'SHORT'
            levels['entry_zone_top'] = orb_high - (orb_range * ENTRY_FIB_MIN)
            levels['entry_zone_bottom'] = orb_high - (orb_range * ENTRY_FIB_MAX)
        else:
            levels = {}
            for fib in FIB_LEVELS:
                price = orb_low + (orb_range * fib)
                levels[fib] = price
            levels['direction'] = 'LONG'
            levels['entry_zone_bottom'] = orb_low + (orb_range * ENTRY_FIB_MIN)
            levels['entry_zone_top'] = orb_low + (orb_range * ENTRY_FIB_MAX)

        return levels

    async def detect_fvgs(self, session: str, orb_data: Dict) -> List[Dict]:
        orb_end = orb_data['orb_formed_at']
        now = datetime.utcnow()

        start_ms = int(orb_end.timestamp() * 1000)
        end_ms = int(now.timestamp() * 1000)

        candles = await self.fetch_candles(BTC_SYMBOL, '1m', limit=100,
                                           start_time=start_ms, end_time=end_ms)

        if not candles or len(candles) < 3:
            return []

        fvgs = []
        entry_top = orb_data['fib_levels']['entry_zone_top']
        entry_bottom = orb_data['fib_levels']['entry_zone_bottom']

        for i in range(1, len(candles) - 1):
            prev = candles[i - 1]
            curr = candles[i]
            next_c = candles[i + 1]

            if curr['low'] > prev['high'] and next_c['low'] > prev['high']:
                fvg_top = curr['low']
                fvg_bottom = prev['high']
                fvg_mid = (fvg_top + fvg_bottom) / 2

                if entry_bottom <= fvg_mid <= entry_top:
                    fvgs.append({
                        'type': 'BULLISH',
                        'top': fvg_top,
                        'bottom': fvg_bottom,
                        'mid': fvg_mid,
                        'timestamp': curr['timestamp'],
                        'in_fib_zone': True
                    })

            if curr['high'] < prev['low'] and next_c['high'] < prev['low']:
                fvg_top = prev['low']
                fvg_bottom = curr['high']
                fvg_mid = (fvg_top + fvg_bottom) / 2

                if entry_bottom <= fvg_mid <= entry_top:
                    fvgs.append({
                        'type': 'BEARISH',
                        'top': fvg_top,
                        'bottom': fvg_bottom,
                        'mid': fvg_mid,
                        'timestamp': curr['timestamp'],
                        'in_fib_zone': True
                    })

        logger.info(f"📊 Found {len(fvgs)} FVGs in fib zone ({entry_bottom:.2f} - {entry_top:.2f})")
        return fvgs

    async def check_retest(self, orb_data: Dict, fvgs: List[Dict]) -> Optional[Dict]:
        now = datetime.utcnow()
        if now > orb_data['retest_deadline']:
            logger.info("📊 ORB retest window expired - no setup")
            return None

        current_price = await self.get_current_price()
        if not current_price:
            return None

        direction = orb_data['direction']
        fib_levels = orb_data['fib_levels']
        entry_top = fib_levels['entry_zone_top']
        entry_bottom = fib_levels['entry_zone_bottom']
        orb_high = orb_data['orb_high']
        orb_low = orb_data['orb_low']
        orb_range = orb_data['orb_range']

        in_fib_zone = entry_bottom <= current_price <= entry_top

        in_fvg = False
        matched_fvg = None
        for fvg in fvgs:
            if fvg['bottom'] <= current_price <= fvg['top']:
                in_fvg = True
                matched_fvg = fvg
                break

        if not in_fib_zone and not in_fvg:
            if orb_data.get('high_first'):
                distance_from_zone = current_price - entry_top if current_price > entry_top else entry_bottom - current_price
            else:
                distance_from_zone = entry_bottom - current_price if current_price < entry_bottom else current_price - entry_top

            if distance_from_zone > orb_range * 1.2:
                logger.info(f"📊 Price ${current_price:.2f} too far from fib zone - cancelling ORB setup")
                return {'cancel': True}

            return None

        if direction == 'LONG':
            entry = current_price
            sl = orb_low - (orb_range * 0.3)
            tp1 = orb_high
            tp2 = orb_high + (orb_range * 0.5)
            sl_pct = ((entry - sl) / entry) * 100
            tp1_pct = ((tp1 - entry) / entry) * 100
        else:
            entry = current_price
            sl = orb_high + (orb_range * 0.3)
            tp1 = orb_low
            tp2 = orb_low - (orb_range * 0.5)
            sl_pct = ((sl - entry) / entry) * 100
            tp1_pct = ((entry - tp1) / entry) * 100

        rr_ratio = tp1_pct / sl_pct if sl_pct > 0 else 0

        if rr_ratio < 1.2:
            logger.info(f"📊 R:R too low ({rr_ratio:.2f}) - skipping")
            return None

        fib_618 = fib_levels.get(0.618, 0)
        near_618 = abs(current_price - fib_618) / orb_range < 0.1

        entry_quality = "PREMIUM"
        if in_fvg and near_618:
            entry_quality = "PREMIUM"
        elif in_fvg:
            entry_quality = "STRONG"
        elif near_618:
            entry_quality = "STRONG"
        elif in_fib_zone:
            entry_quality = "GOOD"

        logger.info(f"📊 RETEST DETECTED | {direction} | Price: ${current_price:.2f} | "
                    f"In FVG: {in_fvg} | In Fib: {in_fib_zone} | Quality: {entry_quality} | R:R {rr_ratio:.2f}")

        return {
            'direction': direction,
            'entry': entry,
            'stop_loss': sl,
            'take_profit_1': tp1,
            'take_profit_2': tp2,
            'sl_pct': sl_pct,
            'tp1_pct': tp1_pct,
            'rr_ratio': rr_ratio,
            'in_fvg': in_fvg,
            'in_fib_zone': in_fib_zone,
            'near_618': near_618,
            'entry_quality': entry_quality,
            'matched_fvg': matched_fvg,
            'fib_618': fib_618,
            'session': orb_data['session'],
            'orb_high': orb_high,
            'orb_low': orb_low,
            'orb_range_pct': orb_data['orb_range_pct'],
            'high_first': orb_data['high_first']
        }

    async def scan(self) -> Optional[Dict]:
        global _active_orb_setup, _last_orb_session

        session = self.get_current_session()
        if not session:
            _active_orb_setup = None
            return None

        if self.is_in_orb_formation():
            logger.debug(f"📊 ORB forming for {session} session...")
            return None

        if _active_orb_setup and _active_orb_setup.get('session') == session:
            fvgs = await self.detect_fvgs(session, _active_orb_setup)
            retest = await self.check_retest(_active_orb_setup, fvgs)

            if retest and retest.get('cancel'):
                logger.info("📊 ORB setup cancelled - price moved too far")
                _active_orb_setup = None
                return None

            if retest:
                _active_orb_setup = None
                return retest

            return None

        if _last_orb_session == f"{session}_{datetime.utcnow().date()}":
            return None

        orb_data = await self.build_opening_range(session)
        if orb_data:
            _active_orb_setup = orb_data
            _last_orb_session = f"{session}_{datetime.utcnow().date()}"
            logger.info(f"📊 ORB setup active for {session} | Direction: {orb_data['direction']} | "
                        f"Range: ${orb_data['orb_range']:.2f} ({orb_data['orb_range_pct']:.3f}%)")

            fvgs = await self.detect_fvgs(session, orb_data)
            retest = await self.check_retest(orb_data, fvgs)

            if retest and not retest.get('cancel'):
                _active_orb_setup = None
                return retest

        return None


def format_btc_orb_message(signal_data: Dict) -> str:
    direction = signal_data['direction']
    session = signal_data['session']
    entry = signal_data['entry']
    tp1 = signal_data['take_profit_1']
    tp2 = signal_data['take_profit_2']
    sl = signal_data['stop_loss']
    sl_pct = signal_data['sl_pct']
    tp1_pct = signal_data['tp1_pct']
    rr = signal_data['rr_ratio']
    quality = signal_data['entry_quality']
    leverage = BTC_ORB_LEVERAGE

    direction_emoji = "🟢 LONG" if direction == "LONG" else "🔴 SHORT"
    session_emoji = "🌏" if session == "ASIA" else "🗽"
    quality_stars = {"PREMIUM": "⭐⭐⭐", "STRONG": "⭐⭐", "GOOD": "⭐"}.get(quality, "⭐")

    fib_text = ""
    if signal_data.get('near_618'):
        fib_text = "\n├ 📐 Near 0.618 Fib level"
    if signal_data.get('in_fvg'):
        fvg = signal_data.get('matched_fvg', {})
        fvg_type = fvg.get('type', 'N/A')
        fib_text += f"\n├ 🔲 {fvg_type} FVG (${fvg.get('bottom', 0):.2f} - ${fvg.get('top', 0):.2f})"

    bias_text = "High formed first → Bearish retracement" if signal_data.get('high_first') else "Low formed first → Bullish retracement"

    msg = f"""
┏━━━━━━━━━━━━━━━━━━━━━━━┓
  📊 <b>BTC ORB SCALP</b> 📊
┗━━━━━━━━━━━━━━━━━━━━━━━┛

{direction_emoji} <b>$BTC</b>
━━━━━━━━━━━━━━━━━━━━━━━━━━

{session_emoji} <b>{session} Session ORB</b>
├ Range: ${signal_data['orb_high']:.2f} - ${signal_data['orb_low']:.2f}
├ ORB Width: {signal_data['orb_range_pct']:.3f}%
├ Bias: {bias_text}
└ Quality: {quality} {quality_stars}

<b>📐 Fibonacci + FVG</b>
├ 0.618 Fib: ${signal_data.get('fib_618', 0):.2f}{fib_text}
└ Entry Zone: Retest confirmed

<b>🎯 Trade Setup</b>
├ Entry: <b>${entry:.2f}</b>
├ TP1: <b>${tp1:.2f}</b> (+{tp1_pct:.2f}% / +{tp1_pct * leverage:.0f}% @ {leverage}x) 🎯
├ TP2: <b>${tp2:.2f}</b> 🎯🎯
└ SL: <b>${sl:.2f}</b> (-{sl_pct:.2f}% / -{sl_pct * leverage:.0f}% @ {leverage}x)

<b>⚡ Risk Management</b>
├ Leverage: <b>{leverage}x</b>
└ R/R: <b>{rr:.2f}:1</b>

⚠️ <b>{leverage}x LEVERAGE - MANAGE RISK</b>
<i>Auto-executing for enabled users...</i>
"""
    return msg.strip()


async def broadcast_btc_orb_signal(db_session, bot):
    if not is_btc_orb_enabled():
        return

    if not check_btc_orb_cooldown():
        logger.debug("📊 BTC ORB on cooldown")
        return

    if not check_btc_orb_daily_limit():
        logger.info("📊 BTC ORB daily signal limit reached")
        return

    try:
        from app.models import User, UserPreference, Signal, Trade
        from app.services.bitunix_trader import execute_bitunix_trade
        from app.services.social_signals import check_global_signal_limit, increment_global_signal_count
        from app.database import SessionLocal
        from sqlalchemy import or_

        if not check_global_signal_limit():
            logger.info("📊 Global daily signal limit reached - skipping BTC ORB scan")
            return

        users = db_session.query(User).join(UserPreference).filter(
            UserPreference.social_mode_enabled == True,
            or_(
                User.is_admin == True,
                User.grandfathered == True,
                (User.subscription_end != None) & (User.subscription_end > datetime.utcnow())
            )
        ).all()

        if not users:
            logger.debug("📊 No authorized users for BTC ORB signals")
            return

        scanner = BTCOrbScanner()
        await scanner.init()

        try:
            signal_data = await scanner.scan()

            if not signal_data:
                return

            if signal_data.get('cancel'):
                return

            logger.info(f"📊 ═══════════════════════════════════════════")
            logger.info(f"📊 BTC ORB SIGNAL: {signal_data['direction']} | {signal_data['session']}")
            logger.info(f"📊 Entry: ${signal_data['entry']:.2f} | Quality: {signal_data['entry_quality']}")
            logger.info(f"📊 ═══════════════════════════════════════════")

            message = format_btc_orb_message(signal_data)

            new_signal = Signal(
                symbol='BTCUSDT',
                direction=signal_data['direction'],
                entry_price=signal_data['entry'],
                stop_loss=signal_data['stop_loss'],
                take_profit=signal_data['take_profit_1'],
                take_profit_1=signal_data['take_profit_1'],
                take_profit_2=signal_data['take_profit_2'],
                confidence=80,
                signal_type='BTC_ORB_SCALP',
                timeframe='15m',
                reasoning=f"BTC ORB+FVG Scalp - {signal_data['session']} session | "
                          f"Quality: {signal_data['entry_quality']} | R:R {signal_data['rr_ratio']:.2f}:1 | "
                          f"{'FVG retest' if signal_data.get('in_fvg') else 'Fib retest'}"
            )
            db_session.add(new_signal)
            db_session.commit()
            db_session.refresh(new_signal)

            record_btc_orb_signal()
            increment_global_signal_count()

            for user in users:
                try:
                    prefs = user.preferences
                    lev_line = f"\n\n📊 {BTC_ORB_LEVERAGE}x"
                    user_message = message + lev_line

                    await bot.send_message(
                        user.telegram_id,
                        user_message,
                        parse_mode="HTML"
                    )
                    logger.info(f"📊 Sent BTC ORB signal to user {user.telegram_id} (ID {user.id})")

                    has_keys = prefs and prefs.bitunix_api_key and prefs.bitunix_api_secret
                    if has_keys and prefs.auto_trading_enabled:
                        trade_db = SessionLocal()
                        try:
                            from app.models import User as UserModel, Signal as SignalModel
                            trade_user = trade_db.query(UserModel).filter(UserModel.id == user.id).first()
                            trade_signal = trade_db.query(SignalModel).filter(SignalModel.id == new_signal.id).first()
                            if trade_user and trade_signal:
                                logger.info(f"🔄 EXECUTING BTC ORB: {signal_data['direction']} for user {user.telegram_id} (ID {user.id})")
                                trade_result = await execute_bitunix_trade(
                                    signal=trade_signal,
                                    user=trade_user,
                                    db=trade_db,
                                    trade_type='BTC_ORB_SCALP',
                                    leverage_override=BTC_ORB_LEVERAGE
                                )
                                if trade_result:
                                    logger.info(f"✅ BTC ORB trade executed for user {user.telegram_id}")
                                    await bot.send_message(
                                        user.telegram_id,
                                        f"✅ <b>Trade Executed on Bitunix</b>\n"
                                        f"<b>$BTC</b> {signal_data['direction']} @ {BTC_ORB_LEVERAGE}x",
                                        parse_mode="HTML"
                                    )
                                else:
                                    logger.warning(f"⚠️ BTC ORB trade blocked for user {user.telegram_id}")
                        finally:
                            trade_db.close()
                    elif has_keys and not prefs.auto_trading_enabled:
                        logger.info(f"📊 User {user.telegram_id} - auto-trading disabled, signal only")
                    else:
                        logger.info(f"📊 User {user.telegram_id} - no API keys, signal only")
                except Exception as e:
                    logger.error(f"Failed to send/execute BTC ORB signal for {user.telegram_id}: {e}")

        finally:
            await scanner.close()

    except Exception as e:
        logger.error(f"Error in BTC ORB broadcast: {e}", exc_info=True)


def format_btc_orb_status() -> str:
    global _active_orb_setup
    enabled_text = "🟢 ENABLED" if _btc_orb_enabled else "🔴 DISABLED"
    setup_text = "No active setup"

    if _active_orb_setup:
        setup = _active_orb_setup
        remaining = (setup['retest_deadline'] - datetime.utcnow()).total_seconds() / 60
        setup_text = (f"🔍 Active {setup['session']} ORB\n"
                      f"├ Direction: {setup['direction']}\n"
                      f"├ Range: ${setup['orb_low']:.2f} - ${setup['orb_high']:.2f}\n"
                      f"└ Retest window: {remaining:.0f}min remaining")

    msg = f"""
📊 <b>BTC ORB+FVG SCALPER STATUS</b>

<b>Scanner:</b> {enabled_text}
<b>Leverage:</b> {BTC_ORB_LEVERAGE}x
<b>Sessions:</b> Asia (00:00 UTC) & NY (13:30 UTC)

<b>📈 Current Setup:</b>
{setup_text}

<b>📊 Signals Today:</b> {_btc_orb_daily_count}/{MAX_BTC_ORB_DAILY_SIGNALS}
<b>⏳ Cooldown:</b> {"Active" if not check_btc_orb_cooldown() else "Ready"}
"""
    return msg.strip()
