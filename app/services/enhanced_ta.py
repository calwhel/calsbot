import logging
import math
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


def calc_ema(closes: List[float], period: int) -> List[float]:
    if len(closes) < period:
        return []
    multiplier = 2 / (period + 1)
    ema = [sum(closes[:period]) / period]
    for price in closes[period:]:
        ema.append((price - ema[-1]) * multiplier + ema[-1])
    return ema


def calc_sma(closes: List[float], period: int) -> List[float]:
    if len(closes) < period:
        return []
    return [sum(closes[i:i+period]) / period for i in range(len(closes) - period + 1)]


def calc_macd(closes: List[float], fast: int = 12, slow: int = 26, signal: int = 9) -> Optional[Dict]:
    if len(closes) < slow + signal:
        return None

    ema_fast = calc_ema(closes, fast)
    ema_slow = calc_ema(closes, slow)

    offset = len(ema_fast) - len(ema_slow)
    macd_line = [ema_fast[offset + i] - ema_slow[i] for i in range(len(ema_slow))]

    if len(macd_line) < signal:
        return None

    signal_line = calc_ema(macd_line, signal)
    offset2 = len(macd_line) - len(signal_line)
    histogram = [macd_line[offset2 + i] - signal_line[i] for i in range(len(signal_line))]

    current_macd = macd_line[-1]
    current_signal = signal_line[-1]
    current_hist = histogram[-1]
    prev_hist = histogram[-2] if len(histogram) >= 2 else 0

    if current_macd > current_signal and (len(histogram) >= 2 and histogram[-2] <= 0 and histogram[-1] > 0):
        crossover = "BULLISH_CROSS"
    elif current_macd < current_signal and (len(histogram) >= 2 and histogram[-2] >= 0 and histogram[-1] < 0):
        crossover = "BEARISH_CROSS"
    elif current_macd > current_signal:
        crossover = "BULLISH"
    elif current_macd < current_signal:
        crossover = "BEARISH"
    else:
        crossover = "NEUTRAL"

    momentum = "INCREASING" if abs(current_hist) > abs(prev_hist) else "DECREASING"

    return {
        'macd': round(current_macd, 6),
        'signal': round(current_signal, 6),
        'histogram': round(current_hist, 6),
        'crossover': crossover,
        'momentum': momentum,
    }


def calc_ema_crossover(closes: List[float], fast: int = 9, slow: int = 21) -> Optional[Dict]:
    if len(closes) < slow + 2:
        return None

    ema_fast = calc_ema(closes, fast)
    ema_slow = calc_ema(closes, slow)

    offset = len(ema_fast) - len(ema_slow)
    if offset < 0 or len(ema_slow) < 2:
        return None

    curr_fast = ema_fast[offset + len(ema_slow) - 1]
    curr_slow = ema_slow[-1]
    prev_fast = ema_fast[offset + len(ema_slow) - 2]
    prev_slow = ema_slow[-2]

    if prev_fast <= prev_slow and curr_fast > curr_slow:
        signal = "GOLDEN_CROSS"
    elif prev_fast >= prev_slow and curr_fast < curr_slow:
        signal = "DEATH_CROSS"
    elif curr_fast > curr_slow:
        signal = "BULLISH"
    else:
        signal = "BEARISH"

    spread_pct = ((curr_fast - curr_slow) / curr_slow) * 100 if curr_slow > 0 else 0

    return {
        'ema_fast': round(curr_fast, 8),
        'ema_slow': round(curr_slow, 8),
        'signal': signal,
        'spread_pct': round(spread_pct, 3),
    }


def calc_bollinger_bands(closes: List[float], period: int = 20, std_dev: float = 2.0) -> Optional[Dict]:
    if len(closes) < period:
        return None

    sma = sum(closes[-period:]) / period
    variance = sum((c - sma) ** 2 for c in closes[-period:]) / period
    std = math.sqrt(variance)

    upper = sma + std_dev * std
    lower = sma - std_dev * std
    current = closes[-1]

    bandwidth = ((upper - lower) / sma) * 100 if sma > 0 else 0

    if bandwidth < 3.0:
        squeeze = "TIGHT"
    elif bandwidth < 6.0:
        squeeze = "NORMAL"
    else:
        squeeze = "WIDE"

    if current > upper:
        position = "ABOVE_UPPER"
    elif current < lower:
        position = "BELOW_LOWER"
    else:
        bb_pct = ((current - lower) / (upper - lower)) * 100 if (upper - lower) > 0 else 50
        if bb_pct > 80:
            position = "NEAR_UPPER"
        elif bb_pct < 20:
            position = "NEAR_LOWER"
        else:
            position = "MIDDLE"

    bb_pct_val = ((current - lower) / (upper - lower)) * 100 if (upper - lower) > 0 else 50

    return {
        'upper': round(upper, 8),
        'middle': round(sma, 8),
        'lower': round(lower, 8),
        'bandwidth': round(bandwidth, 2),
        'squeeze': squeeze,
        'position': position,
        'percent_b': round(bb_pct_val, 1),
    }


def calc_atr(highs: List[float], lows: List[float], closes: List[float], period: int = 14) -> Optional[Dict]:
    if len(closes) < period + 1 or len(highs) < period + 1 or len(lows) < period + 1:
        return None

    true_ranges = []
    for i in range(1, len(closes)):
        h = highs[i]
        l = lows[i]
        pc = closes[i - 1]
        tr = max(h - l, abs(h - pc), abs(l - pc))
        true_ranges.append(tr)

    if len(true_ranges) < period:
        return None

    atr = sum(true_ranges[-period:]) / period
    current_price = closes[-1]
    atr_pct = (atr / current_price) * 100 if current_price > 0 else 0

    if atr_pct > 5:
        volatility = "EXTREME"
    elif atr_pct > 3:
        volatility = "HIGH"
    elif atr_pct > 1.5:
        volatility = "MODERATE"
    else:
        volatility = "LOW"

    return {
        'atr': round(atr, 8),
        'atr_percent': round(atr_pct, 2),
        'volatility': volatility,
    }


def calc_vwap(highs: List[float], lows: List[float], closes: List[float], volumes: List[float]) -> Optional[Dict]:
    n = min(len(highs), len(lows), len(closes), len(volumes))
    if n < 5:
        return None

    cum_tp_vol = 0
    cum_vol = 0
    for i in range(n):
        typical_price = (highs[i] + lows[i] + closes[i]) / 3
        cum_tp_vol += typical_price * volumes[i]
        cum_vol += volumes[i]

    vwap = cum_tp_vol / cum_vol if cum_vol > 0 else closes[-1]
    current = closes[-1]
    deviation_pct = ((current - vwap) / vwap) * 100 if vwap > 0 else 0

    if deviation_pct > 2:
        position = "ABOVE"
    elif deviation_pct < -2:
        position = "BELOW"
    else:
        position = "AT_VWAP"

    return {
        'vwap': round(vwap, 8),
        'deviation_pct': round(deviation_pct, 2),
        'position': position,
    }


def calc_rsi(closes: List[float], period: int = 14) -> float:
    if len(closes) < period + 1:
        return 50.0
    deltas = [closes[i] - closes[i - 1] for i in range(1, len(closes))]
    gains = [d if d > 0 else 0 for d in deltas]
    losses = [-d if d < 0 else 0 for d in deltas]
    avg_gain = sum(gains[-period:]) / period
    avg_loss = sum(losses[-period:]) / period
    if avg_loss > 0:
        rs = avg_gain / avg_loss
        return round(100 - (100 / (1 + rs)), 1)
    return 100.0 if avg_gain > 0 else 50.0


def analyze_klines(klines_15m: List, klines_1h: List = None) -> Dict:
    result = {}

    if not klines_15m or len(klines_15m) < 30:
        return result

    closes = [float(k[4]) for k in klines_15m]
    highs = [float(k[2]) for k in klines_15m]
    lows = [float(k[3]) for k in klines_15m]
    volumes = [float(k[5]) for k in klines_15m]

    macd = calc_macd(closes)
    if macd:
        result['macd'] = macd

    ema_cross = calc_ema_crossover(closes)
    if ema_cross:
        result['ema_cross'] = ema_cross

    bb = calc_bollinger_bands(closes)
    if bb:
        result['bollinger'] = bb

    atr = calc_atr(highs, lows, closes)
    if atr:
        result['atr'] = atr

    vwap = calc_vwap(highs, lows, closes, volumes)
    if vwap:
        result['vwap'] = vwap

    result['rsi_15m'] = calc_rsi(closes)

    if klines_1h and len(klines_1h) >= 15:
        closes_1h = [float(k[4]) for k in klines_1h]
        result['rsi_1h'] = calc_rsi(closes_1h)

        macd_1h = calc_macd(closes_1h)
        if macd_1h:
            result['macd_1h'] = macd_1h

    if 'rsi_15m' in result and 'rsi_1h' in result:
        rsi_15 = result['rsi_15m']
        rsi_1h = result['rsi_1h']
        if rsi_15 > 50 and rsi_1h > 50:
            result['trend_alignment'] = 'BULLISH_ALIGNED'
        elif rsi_15 < 50 and rsi_1h < 50:
            result['trend_alignment'] = 'BEARISH_ALIGNED'
        elif rsi_15 > 50 and rsi_1h < 50:
            result['trend_alignment'] = 'MIXED_BULLISH_15M'
        else:
            result['trend_alignment'] = 'MIXED_BEARISH_15M'

    return result


def format_ta_for_ai(ta: Dict) -> str:
    lines = []

    macd = ta.get('macd')
    if macd:
        lines.append(f"MACD: {macd['crossover']} (histogram momentum: {macd['momentum']})")

    ema = ta.get('ema_cross')
    if ema:
        lines.append(f"EMA 9/21: {ema['signal']} (spread: {ema['spread_pct']:+.2f}%)")

    bb = ta.get('bollinger')
    if bb:
        lines.append(f"Bollinger Bands: {bb['position']} | Squeeze: {bb['squeeze']} | Bandwidth: {bb['bandwidth']:.1f}% | %B: {bb['percent_b']:.0f}")

    atr = ta.get('atr')
    if atr:
        lines.append(f"ATR: {atr['atr_percent']:.2f}% ({atr['volatility']} volatility)")

    vwap = ta.get('vwap')
    if vwap:
        lines.append(f"VWAP: {vwap['position']} ({vwap['deviation_pct']:+.2f}% from VWAP)")

    rsi_15 = ta.get('rsi_15m')
    rsi_1h = ta.get('rsi_1h')
    if rsi_15 is not None:
        rsi_line = f"RSI 15m: {rsi_15:.0f}"
        if rsi_1h is not None:
            rsi_line += f" | RSI 1H: {rsi_1h:.0f}"
        lines.append(rsi_line)

    alignment = ta.get('trend_alignment')
    if alignment:
        lines.append(f"Multi-TF Trend: {alignment}")

    macd_1h = ta.get('macd_1h')
    if macd_1h:
        lines.append(f"MACD 1H: {macd_1h['crossover']} ({macd_1h['momentum']})")

    return "\n".join(lines)


def format_ta_for_message(ta: Dict) -> str:
    parts = []

    ema = ta.get('ema_cross')
    if ema:
        ema_icons = {
            'GOLDEN_CROSS': '🟢',
            'DEATH_CROSS': '🔴',
            'BULLISH': '🟢',
            'BEARISH': '🔴',
        }
        parts.append(f"{ema_icons.get(ema['signal'], '⚪')} EMA <b>{ema['signal'].replace('_', ' ')}</b>")

    macd = ta.get('macd')
    if macd:
        m_icon = '📈' if 'BULLISH' in macd['crossover'] else '📉'
        parts.append(f"{m_icon} MACD <b>{macd['crossover'].replace('_', ' ')}</b>")

    bb = ta.get('bollinger')
    if bb:
        sq_icon = '🔥' if bb['squeeze'] == 'TIGHT' else '📊'
        parts.append(f"{sq_icon} BB <b>{bb['squeeze']}</b> squeeze · %B <b>{bb['percent_b']:.0f}</b>")

    atr = ta.get('atr')
    if atr:
        parts.append(f"⚡ ATR <b>{atr['atr_percent']:.1f}%</b> ({atr['volatility']})")

    vwap = ta.get('vwap')
    if vwap:
        v_icon = '🔼' if vwap['deviation_pct'] > 0 else '🔽'
        parts.append(f"{v_icon} VWAP <b>{vwap['deviation_pct']:+.1f}%</b>")

    rsi_15 = ta.get('rsi_15m')
    rsi_1h = ta.get('rsi_1h')
    if rsi_15 is not None:
        r_line = f"RSI <b>{rsi_15:.0f}</b>"
        if rsi_1h is not None:
            r_line += f" (1H: <b>{rsi_1h:.0f}</b>)"
        parts.append(r_line)

    alignment = ta.get('trend_alignment')
    if alignment:
        a_icon = '✅' if 'ALIGNED' in alignment and 'BULLISH' in alignment else ('❌' if 'ALIGNED' in alignment and 'BEARISH' in alignment else '⚠️')
        parts.append(f"{a_icon} Trend <b>{alignment.replace('_', ' ')}</b>")

    return "\n".join(parts)


def get_atr_based_tp_sl(ta: Dict, direction: str, base_tp: float, base_sl: float) -> Tuple[float, float]:
    atr = ta.get('atr')
    if not atr:
        return base_tp, base_sl

    atr_pct = atr['atr_percent']

    if atr_pct > 5:
        tp_mult = 1.8
        sl_mult = 1.5
    elif atr_pct > 3:
        tp_mult = 1.4
        sl_mult = 1.2
    elif atr_pct > 1.5:
        tp_mult = 1.0
        sl_mult = 1.0
    else:
        tp_mult = 0.8
        sl_mult = 0.8

    bb = ta.get('bollinger')
    if bb and bb['squeeze'] == 'TIGHT':
        tp_mult *= 1.3

    macd = ta.get('macd')
    if macd:
        if 'BULLISH_CROSS' in macd['crossover'] and direction == 'LONG':
            tp_mult *= 1.2
        elif 'BEARISH_CROSS' in macd['crossover'] and direction == 'SHORT':
            tp_mult *= 1.2

    adjusted_tp = round(base_tp * tp_mult, 2)
    adjusted_sl = round(base_sl * sl_mult, 2)

    adjusted_tp = max(adjusted_tp, 0.5)
    adjusted_sl = max(adjusted_sl, 0.3)
    adjusted_tp = min(adjusted_tp, 15.0)
    adjusted_sl = min(adjusted_sl, 8.0)

    return adjusted_tp, adjusted_sl
