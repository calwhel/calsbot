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


def calc_keltner_channels(highs: List[float], lows: List[float], closes: List[float], 
                          ema_period: int = 20, atr_period: int = 10, multiplier: float = 1.8) -> Optional[Dict]:
    if len(closes) < max(ema_period, atr_period + 1):
        return None
    
    ema_vals = calc_ema(closes, ema_period)
    if not ema_vals:
        return None
    middle = ema_vals[-1]
    
    atr_data = calc_atr(highs, lows, closes, atr_period)
    if not atr_data:
        return None
    atr = atr_data['atr']
    
    upper = middle + multiplier * atr
    lower = middle - multiplier * atr
    current = closes[-1]
    
    return {
        'upper': round(upper, 8),
        'middle': round(middle, 8),
        'lower': round(lower, 8),
        'atr': round(atr, 8),
    }


def calc_squeeze(closes: List[float], highs: List[float], lows: List[float],
                 bb_period: int = 20, bb_mult: float = 1.5, 
                 kc_ema: int = 20, kc_atr: int = 10, kc_mult: float = 1.8) -> Optional[Dict]:
    bb = calc_bollinger_bands(closes, bb_period, bb_mult)
    kc = calc_keltner_channels(highs, lows, closes, kc_ema, kc_atr, kc_mult)
    if not bb or not kc:
        return None
    
    is_squeeze = bb['upper'] < kc['upper'] and bb['lower'] > kc['lower']
    
    prev_closes = closes[:-1]
    prev_highs = highs[:-1]
    prev_lows = lows[:-1]
    prev_bb = calc_bollinger_bands(prev_closes, bb_period, bb_mult)
    prev_kc = calc_keltner_channels(prev_highs, prev_lows, prev_closes, kc_ema, kc_atr, kc_mult)
    
    was_squeeze = False
    if prev_bb and prev_kc:
        was_squeeze = prev_bb['upper'] < prev_kc['upper'] and prev_bb['lower'] > prev_kc['lower']
    
    squeeze_release = was_squeeze and not is_squeeze
    
    current = closes[-1]
    direction = 'NEUTRAL'
    if squeeze_release:
        if current > bb['upper']:
            direction = 'BULLISH'
        elif current < bb['lower']:
            direction = 'BEARISH'
        elif current > bb['middle']:
            direction = 'BULLISH'
        else:
            direction = 'BEARISH'
    
    return {
        'is_squeeze': is_squeeze,
        'was_squeeze': was_squeeze,
        'squeeze_release': squeeze_release,
        'direction': direction,
        'bb_bandwidth': bb['bandwidth'],
        'bb_position': bb['position'],
    }


def calc_supertrend(highs: List[float], lows: List[float], closes: List[float],
                    atr_period: int = 10, factor: float = 3.0) -> Optional[Dict]:
    if len(closes) < atr_period + 2:
        return None
    
    true_ranges = []
    for i in range(1, len(closes)):
        tr = max(highs[i] - lows[i], abs(highs[i] - closes[i-1]), abs(lows[i] - closes[i-1]))
        true_ranges.append(tr)
    
    atr_values = []
    atr = sum(true_ranges[:atr_period]) / atr_period
    atr_values.append(atr)
    for i in range(atr_period, len(true_ranges)):
        atr = (atr * (atr_period - 1) + true_ranges[i]) / atr_period
        atr_values.append(atr)
    
    n = min(len(closes) - 1, len(atr_values))
    
    upper_bands = []
    lower_bands = []
    supertrend = []
    direction_list = []
    
    for i in range(n):
        ci = len(closes) - n + i
        ai = len(atr_values) - n + i
        
        hl2 = (highs[ci] + lows[ci]) / 2
        basic_upper = hl2 + factor * atr_values[ai]
        basic_lower = hl2 - factor * atr_values[ai]
        
        if i == 0:
            upper_bands.append(basic_upper)
            lower_bands.append(basic_lower)
            direction_list.append(1 if closes[ci] > basic_upper else -1)
            supertrend.append(basic_lower if direction_list[-1] == 1 else basic_upper)
        else:
            final_upper = basic_upper if basic_upper < upper_bands[-1] or closes[ci - 1] > upper_bands[-1] else upper_bands[-1]
            final_lower = basic_lower if basic_lower > lower_bands[-1] or closes[ci - 1] < lower_bands[-1] else lower_bands[-1]
            upper_bands.append(final_upper)
            lower_bands.append(final_lower)
            
            prev_dir = direction_list[-1]
            if prev_dir == 1:
                if closes[ci] < final_lower:
                    direction_list.append(-1)
                else:
                    direction_list.append(1)
            else:
                if closes[ci] > final_upper:
                    direction_list.append(1)
                else:
                    direction_list.append(-1)
            
            supertrend.append(final_lower if direction_list[-1] == 1 else final_upper)
    
    if len(direction_list) < 2:
        return None
    
    curr_dir = direction_list[-1]
    prev_dir = direction_list[-2]
    trend_flip = curr_dir != prev_dir
    
    signal = 'NEUTRAL'
    if trend_flip and curr_dir == 1:
        signal = 'BUY'
    elif trend_flip and curr_dir == -1:
        signal = 'SELL'
    elif curr_dir == 1:
        signal = 'BULLISH'
    elif curr_dir == -1:
        signal = 'BEARISH'
    
    return {
        'direction': curr_dir,
        'supertrend_value': round(supertrend[-1], 8),
        'signal': signal,
        'trend_flip': trend_flip,
        'trend_strength': sum(1 for d in direction_list[-5:] if d == curr_dir),
    }


def calc_ema_ribbon(closes: List[float], periods: List[int] = None) -> Optional[Dict]:
    if periods is None:
        periods = [8, 21, 34]
    
    if len(closes) < max(periods):
        return None
    
    emas = {}
    for p in periods:
        vals = calc_ema(closes, p)
        if not vals:
            return None
        emas[p] = vals[-1]
    
    sorted_periods = sorted(periods)
    bullish_aligned = all(emas[sorted_periods[i]] > emas[sorted_periods[i+1]] for i in range(len(sorted_periods)-1))
    bearish_aligned = all(emas[sorted_periods[i]] < emas[sorted_periods[i+1]] for i in range(len(sorted_periods)-1))
    
    current = closes[-1]
    above_all = all(current > emas[p] for p in periods)
    below_all = all(current < emas[p] for p in periods)
    
    signal = 'NEUTRAL'
    if bullish_aligned and above_all:
        signal = 'STRONG_BULLISH'
    elif bullish_aligned:
        signal = 'BULLISH'
    elif bearish_aligned and below_all:
        signal = 'STRONG_BEARISH'
    elif bearish_aligned:
        signal = 'BEARISH'
    
    return {
        'emas': {str(p): round(v, 8) for p, v in emas.items()},
        'bullish_aligned': bullish_aligned,
        'bearish_aligned': bearish_aligned,
        'above_all': above_all,
        'below_all': below_all,
        'signal': signal,
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


def calc_volume_profile(highs: List[float], lows: List[float], closes: List[float], volumes: List[float], num_bins: int = 24) -> Optional[Dict]:
    """Compute volume profile: volume distribution across price levels.
    Identifies High Volume Nodes (HVN = strong S/R) and Low Volume Nodes (LVN = price voids/targets)."""
    n = min(len(highs), len(lows), len(closes), len(volumes))
    if n < 20:
        return None

    price_min = min(lows[-n:])
    price_max = max(highs[-n:])
    price_range = price_max - price_min
    if price_range <= 0:
        return None

    bin_size = price_range / num_bins
    bins = [0.0] * num_bins
    bin_prices = [price_min + (i + 0.5) * bin_size for i in range(num_bins)]

    for i in range(n):
        candle_low = lows[i]
        candle_high = highs[i]
        candle_vol = volumes[i]
        if candle_vol <= 0:
            continue

        low_bin = max(0, int((candle_low - price_min) / bin_size))
        high_bin = min(num_bins - 1, int((candle_high - price_min) / bin_size))
        num_candle_bins = high_bin - low_bin + 1
        if num_candle_bins <= 0:
            continue
        vol_per_bin = candle_vol / num_candle_bins

        for b in range(low_bin, high_bin + 1):
            if 0 <= b < num_bins:
                bins[b] += vol_per_bin

    total_vol = sum(bins)
    if total_vol <= 0:
        return None

    avg_vol = total_vol / num_bins
    current_price = closes[-1]

    hvn_levels = []
    lvn_levels = []
    poc_idx = 0
    poc_vol = 0

    for i in range(num_bins):
        if bins[i] > poc_vol:
            poc_vol = bins[i]
            poc_idx = i

        ratio = bins[i] / avg_vol if avg_vol > 0 else 0
        if ratio >= 1.5:
            hvn_levels.append({
                'price': round(bin_prices[i], 8),
                'volume_ratio': round(ratio, 2),
            })
        elif ratio <= 0.4:
            lvn_levels.append({
                'price': round(bin_prices[i], 8),
                'volume_ratio': round(ratio, 2),
            })

    poc_price = bin_prices[poc_idx]

    va_target = total_vol * 0.7
    va_vol = poc_vol
    va_low_idx = poc_idx
    va_high_idx = poc_idx
    while va_vol < va_target and (va_low_idx > 0 or va_high_idx < num_bins - 1):
        add_low = bins[va_low_idx - 1] if va_low_idx > 0 else 0
        add_high = bins[va_high_idx + 1] if va_high_idx < num_bins - 1 else 0
        if add_low >= add_high and va_low_idx > 0:
            va_low_idx -= 1
            va_vol += add_low
        elif va_high_idx < num_bins - 1:
            va_high_idx += 1
            va_vol += add_high
        else:
            va_low_idx -= 1
            va_vol += add_low

    hvn_supports = sorted([h for h in hvn_levels if h['price'] < current_price], key=lambda x: x['price'], reverse=True)[:3]
    hvn_resistances = sorted([h for h in hvn_levels if h['price'] > current_price], key=lambda x: x['price'])[:3]
    lvn_above = sorted([l for l in lvn_levels if l['price'] > current_price], key=lambda x: x['price'])[:2]
    lvn_below = sorted([l for l in lvn_levels if l['price'] < current_price], key=lambda x: x['price'], reverse=True)[:2]

    poc_deviation_pct = ((current_price - poc_price) / poc_price) * 100 if poc_price > 0 else 0

    return {
        'poc': round(poc_price, 8),
        'poc_deviation_pct': round(poc_deviation_pct, 2),
        'value_area_high': round(bin_prices[va_high_idx], 8),
        'value_area_low': round(bin_prices[va_low_idx], 8),
        'hvn_supports': hvn_supports,
        'hvn_resistances': hvn_resistances,
        'lvn_above': lvn_above,
        'lvn_below': lvn_below,
        'total_hvn': len(hvn_levels),
        'total_lvn': len(lvn_levels),
    }


def find_support_resistance(highs: List[float], lows: List[float], closes: List[float], current_price: float) -> Dict:
    """Find key support and resistance levels from price action for optimal TP/SL placement."""
    if len(highs) < 10:
        return {}

    price_range = max(highs) - min(lows)
    if price_range <= 0:
        return {}

    cluster_threshold = price_range * 0.008

    pivot_highs = []
    pivot_lows = []

    for i in range(2, len(highs) - 2):
        if highs[i] >= highs[i-1] and highs[i] >= highs[i-2] and highs[i] >= highs[i+1] and highs[i] >= highs[i+2]:
            pivot_highs.append(highs[i])
        if lows[i] <= lows[i-1] and lows[i] <= lows[i-2] and lows[i] <= lows[i+1] and lows[i] <= lows[i+2]:
            pivot_lows.append(lows[i])

    all_levels = pivot_highs + pivot_lows
    if not all_levels:
        return {}

    all_levels.sort()
    clusters = []
    current_cluster = [all_levels[0]]

    for level in all_levels[1:]:
        if level - current_cluster[-1] <= cluster_threshold:
            current_cluster.append(level)
        else:
            clusters.append(sum(current_cluster) / len(current_cluster))
            current_cluster = [level]
    clusters.append(sum(current_cluster) / len(current_cluster))

    supports = sorted([l for l in clusters if l < current_price], reverse=True)
    resistances = sorted([l for l in clusters if l > current_price])

    recent_high = max(highs[-10:])
    recent_low = min(lows[-10:])

    result = {
        'supports': supports[:3],
        'resistances': resistances[:3],
        'recent_high': recent_high,
        'recent_low': recent_low,
    }

    if supports:
        result['nearest_support'] = supports[0]
        result['support_distance_pct'] = ((current_price - supports[0]) / current_price) * 100
    if resistances:
        result['nearest_resistance'] = resistances[0]
        result['resistance_distance_pct'] = ((resistances[0] - current_price) / current_price) * 100

    return result


def optimize_tp_sl_from_chart_levels(enhanced_ta: Dict, direction: str, current_price: float, tp_percent: float, sl_percent: float) -> tuple:
    """Adjust TP/SL to align with chart support/resistance + volume profile levels."""
    sr_data = enhanced_ta.get('support_resistance', {})
    vp_data = enhanced_ta.get('volume_profile', {})

    nearest_resistance = sr_data.get('nearest_resistance')
    nearest_support = sr_data.get('nearest_support')
    resistance_dist = sr_data.get('resistance_distance_pct', 0)
    support_dist = sr_data.get('support_distance_pct', 0)

    if vp_data:
        hvn_resistances = vp_data.get('hvn_resistances', [])
        hvn_supports = vp_data.get('hvn_supports', [])
        lvn_above = vp_data.get('lvn_above', [])
        lvn_below = vp_data.get('lvn_below', [])

        if hvn_resistances and current_price > 0:
            hvn_r = hvn_resistances[0]['price']
            hvn_r_dist = ((hvn_r - current_price) / current_price) * 100
            if not nearest_resistance or hvn_r_dist < resistance_dist:
                nearest_resistance = hvn_r
                resistance_dist = hvn_r_dist
        if hvn_supports and current_price > 0:
            hvn_s = hvn_supports[0]['price']
            hvn_s_dist = ((current_price - hvn_s) / current_price) * 100
            if not nearest_support or hvn_s_dist < support_dist:
                nearest_support = hvn_s
                support_dist = hvn_s_dist

        if direction == 'LONG' and lvn_above:
            lvn_target = lvn_above[0]['price']
            lvn_dist = ((lvn_target - current_price) / current_price) * 100
            if 0.5 <= lvn_dist <= tp_percent * 1.3:
                tp_percent = round(max(lvn_dist * 0.9, tp_percent * 0.8), 2)
        elif direction == 'SHORT' and lvn_below:
            lvn_target = lvn_below[0]['price']
            lvn_dist = ((current_price - lvn_target) / current_price) * 100
            if 0.5 <= lvn_dist <= tp_percent * 1.3:
                tp_percent = round(max(lvn_dist * 0.9, tp_percent * 0.8), 2)

    if not sr_data and not vp_data:
        return tp_percent, sl_percent

    if direction == 'LONG':
        if nearest_resistance and resistance_dist > 0.3:
            chart_tp = resistance_dist * 0.95
            if 0.5 <= chart_tp <= tp_percent * 1.5:
                tp_percent = round(max(chart_tp, tp_percent * 0.7), 2)
        if nearest_support and support_dist > 0.2:
            chart_sl = support_dist * 1.05
            if 0.3 <= chart_sl <= sl_percent * 1.5:
                sl_percent = round(max(chart_sl, sl_percent * 0.7), 2)
    elif direction == 'SHORT':
        if nearest_support and support_dist > 0.3:
            chart_tp = support_dist * 0.95
            if 0.5 <= chart_tp <= tp_percent * 1.5:
                tp_percent = round(max(chart_tp, tp_percent * 0.7), 2)
        if nearest_resistance and resistance_dist > 0.2:
            chart_sl = resistance_dist * 1.05
            if 0.3 <= chart_sl <= sl_percent * 1.5:
                sl_percent = round(max(chart_sl, sl_percent * 0.7), 2)

    return tp_percent, sl_percent


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

    current_price = closes[-1] if closes else 0
    if current_price > 0:
        sr_levels = find_support_resistance(highs, lows, closes, current_price)
        if sr_levels:
            result['support_resistance'] = sr_levels

    vp = calc_volume_profile(highs, lows, closes, volumes)
    if vp:
        result['volume_profile'] = vp

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

    sr = ta.get('support_resistance')
    if sr:
        sr_parts = []
        if sr.get('nearest_support'):
            sr_parts.append(f"Nearest Support: ${sr['nearest_support']:.6f} ({sr.get('support_distance_pct', 0):.2f}% below)")
        if sr.get('nearest_resistance'):
            sr_parts.append(f"Nearest Resistance: ${sr['nearest_resistance']:.6f} ({sr.get('resistance_distance_pct', 0):.2f}% above)")
        supports = sr.get('supports', [])
        resistances = sr.get('resistances', [])
        if len(supports) > 1:
            sr_parts.append(f"Support Levels: {', '.join([f'${s:.6f}' for s in supports[:3]])}")
        if len(resistances) > 1:
            sr_parts.append(f"Resistance Levels: {', '.join([f'${r:.6f}' for r in resistances[:3]])}")
        if sr.get('recent_high'):
            sr_parts.append(f"Recent High: ${sr['recent_high']:.6f} | Recent Low: ${sr['recent_low']:.6f}")
        if sr_parts:
            lines.append("KEY LEVELS: " + " | ".join(sr_parts))

    vp = ta.get('volume_profile')
    if vp:
        vp_parts = [f"POC: ${vp['poc']:.6f} ({vp['poc_deviation_pct']:+.2f}% from price)"]
        vp_parts.append(f"Value Area: ${vp['value_area_low']:.6f} - ${vp['value_area_high']:.6f}")
        hvn_r = vp.get('hvn_resistances', [])
        hvn_s = vp.get('hvn_supports', [])
        if hvn_r:
            hvn_r_strs = [f"${h['price']:.6f} ({h['volume_ratio']:.1f}x vol)" for h in hvn_r[:2]]
            vp_parts.append("HVN Resistance: " + ", ".join(hvn_r_strs))
        if hvn_s:
            hvn_s_strs = [f"${h['price']:.6f} ({h['volume_ratio']:.1f}x vol)" for h in hvn_s[:2]]
            vp_parts.append("HVN Support: " + ", ".join(hvn_s_strs))
        lvn_a = vp.get('lvn_above', [])
        lvn_b = vp.get('lvn_below', [])
        if lvn_a:
            lvn_a_strs = [f"${lv['price']:.6f}" for lv in lvn_a]
            vp_parts.append("LVN Above (price void): " + ", ".join(lvn_a_strs))
        if lvn_b:
            lvn_b_strs = [f"${lv['price']:.6f}" for lv in lvn_b]
            vp_parts.append("LVN Below (price void): " + ", ".join(lvn_b_strs))
        lines.append("VOLUME PROFILE: " + " | ".join(vp_parts))

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

    vp = ta.get('volume_profile')
    if vp:
        poc_dev = vp.get('poc_deviation_pct', 0)
        poc_icon = '🔼' if poc_dev > 0 else '🔽'
        parts.append(f"{poc_icon} POC <b>{poc_dev:+.1f}%</b> | VA <b>${vp['value_area_low']:.4f}-${vp['value_area_high']:.4f}</b>")
        hvn_count = vp.get('total_hvn', 0)
        lvn_count = vp.get('total_lvn', 0)
        if hvn_count or lvn_count:
            parts.append(f"📊 Vol Profile: <b>{hvn_count}</b> HVN · <b>{lvn_count}</b> LVN")

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
