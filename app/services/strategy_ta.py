"""
Strategy Condition Evaluators — Build Your Own Strategy Portal

Evaluates user-defined strategy conditions against live market data.
All calculations are done from raw OHLCV data (Binance Futures klines).
A shared klines cache is passed through each evaluation to avoid duplicate fetches.

Supported condition types:
  indicator           — RSI, MACD, EMA, EMA ribbon, BB, VWAP, Volume, StochRSI,
                        SuperTrend, ADX, ATR expansion, Williams %R, CCI, OBV,
                        Heikin Ashi, Ichimoku, Squeeze Momentum
  price_momentum      — X% move in Y minutes
  volume_spike        — Volume is N× its average
  support_resistance  — At support/resistance, breakout/breakdown
  fvg                 — Fair Value Gap detection
  candlestick         — Engulfing, hammer, pin bar, doji, morning/evening star, etc.
  consecutive_candles — N consecutive green/red candles
  market_structure    — BOS bullish/bearish, CHoCH bullish/bearish
  order_block         — Bullish/bearish order block touch
  fibonacci           — Price at Fib retracement level
  divergence          — RSI/MACD divergence vs price
  funding_rate        — Funding rate threshold (Binance Futures)
  open_interest       — Open interest % change
  session             — Trading session filter (Asian/London/NY)
  price_relative      — Price vs daily open / session high/low
  sentiment           — Social sentiment score (LunarCrush)
  liquidation         — Price near liquidation cluster
"""
import asyncio
import logging
import math
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ─── Math helpers ──────────────────────────────────────────────────────────────

def _cmp(actual: float, operator: str, threshold: float) -> bool:
    return {"gt": actual > threshold, "gte": actual >= threshold,
            "lt": actual < threshold,  "lte": actual <= threshold,
            "eq": abs(actual - threshold) < 0.001}.get(operator, False)

def _closes(k): return [float(x[4]) for x in k]
def _highs(k):  return [float(x[2]) for x in k]
def _lows(k):   return [float(x[3]) for x in k]
def _opens(k):  return [float(x[1]) for x in k]
def _vols(k):   return [float(x[5]) for x in k]

def _sma(data: List[float], period: int) -> Optional[float]:
    if len(data) < period: return None
    return sum(data[-period:]) / period

def _ema_list(data: List[float], period: int) -> List[float]:
    """Return full EMA series for a data list."""
    if not data: return []
    k = 2 / (period + 1)
    ema = [data[0]]
    for v in data[1:]:
        ema.append(v * k + ema[-1] * (1 - k))
    return ema

def _ema(data: List[float], period: int) -> Optional[float]:
    if len(data) < period: return None
    return _ema_list(data, period)[-1]

def _wma_list(data: List[float], period: int) -> List[float]:
    """Weighted Moving Average — linearly weighted, most recent bar has highest weight."""
    if len(data) < period: return []
    weights = list(range(1, period + 1))
    w_sum = sum(weights)
    result = []
    for i in range(period - 1, len(data)):
        wma = sum(data[i - period + 1 + j] * weights[j] for j in range(period)) / w_sum
        result.append(wma)
    return result

def _smma_list(data: List[float], period: int) -> List[float]:
    """Smoothed MA / Wilder's MA (SMMA/RMA) — alpha = 1/period."""
    if len(data) < period: return []
    first = sum(data[:period]) / period
    result = [first]
    for v in data[period:]:
        result.append((result[-1] * (period - 1) + v) / period)
    return result

def _vwma_list(data: List[float], vols: List[float], period: int) -> List[float]:
    """Volume Weighted Moving Average."""
    if len(data) < period or len(vols) < period: return []
    result = []
    for i in range(period - 1, len(data)):
        d_sl = data[i - period + 1:i + 1]
        v_sl = vols[i - period + 1:i + 1]
        v_sum = sum(v_sl)
        result.append(sum(d * v for d, v in zip(d_sl, v_sl)) / v_sum if v_sum else sum(d_sl) / period)
    return result

def _apply_ma_smooth(
    series: List[float], ma_type: str, period: int,
    vols: Optional[List[float]] = None
) -> List[float]:
    """
    Apply MA smoothing to a series. Used for indicators like CCI where
    PineScript applies an optional MA (SMA/EMA/SMMA/WMA/VWMA) to the raw
    indicator values before threshold comparison (e.g. Trend Magic smoothing).
    Returns the smoothed series; falls back to the original if unsupported.
    """
    mt = ma_type.lower()
    if mt == "sma":
        out = []
        for i in range(len(series)):
            win = series[max(0, i - period + 1):i + 1]
            out.append(sum(win) / len(win))
        return out
    if mt == "ema":
        return _ema_list(series, period)
    if mt in ("smma", "rma", "wilder"):
        return _smma_list(series, period)
    if mt == "wma":
        raw = _wma_list(series, period)
        return [series[0]] * (len(series) - len(raw)) + raw
    if mt == "vwma" and vols:
        raw = _vwma_list(series, vols, period)
        return [series[0]] * (len(series) - len(raw)) + raw
    return series  # no-op fallback

def _true_range(klines: List) -> List[float]:
    tr = []
    for i, k in enumerate(klines):
        h, l, c = float(k[2]), float(k[3]), float(k[4])
        if i == 0:
            tr.append(h - l)
        else:
            pc = float(klines[i-1][4])
            tr.append(max(h - l, abs(h - pc), abs(l - pc)))
    return tr

def _atr_values(klines: List, period: int = 14) -> List[float]:
    tr = _true_range(klines)
    return _ema_list(tr, period)

def _atr(klines: List, period: int = 14) -> Optional[float]:
    v = _atr_values(klines, period)
    return v[-1] if v else None

def _rsi_values(closes: List[float], period: int = 14) -> List[float]:
    """Return RSI series for full closes list."""
    if len(closes) < period + 1:
        return []
    gains, losses = [], []
    for i in range(1, len(closes)):
        d = closes[i] - closes[i-1]
        gains.append(max(d, 0))
        losses.append(max(-d, 0))
    rsi = []
    avg_g = sum(gains[:period]) / period
    avg_l = sum(losses[:period]) / period
    for i in range(period, len(gains)):
        if avg_l == 0:
            rsi.append(100.0)
        else:
            rs = avg_g / avg_l
            rsi.append(100 - 100 / (1 + rs))
        avg_g = (avg_g * (period - 1) + gains[i]) / period
        avg_l = (avg_l * (period - 1) + losses[i]) / period
    return rsi

def _stdev(data: List[float]) -> float:
    if len(data) < 2: return 0.0
    m = sum(data) / len(data)
    return math.sqrt(sum((x - m)**2 for x in data) / len(data))

def _swing_highs(highs: List[float], lows: List[float], left=2, right=2) -> List[int]:
    idx = []
    for i in range(left, len(highs) - right):
        if all(highs[i] >= highs[i-j] for j in range(1, left+1)) and \
           all(highs[i] >= highs[i+j] for j in range(1, right+1)):
            idx.append(i)
    return idx

def _swing_lows(highs: List[float], lows: List[float], left=2, right=2) -> List[int]:
    idx = []
    for i in range(left, len(lows) - right):
        if all(lows[i] <= lows[i-j] for j in range(1, left+1)) and \
           all(lows[i] <= lows[i+j] for j in range(1, right+1)):
            idx.append(i)
    return idx


# ─── Klines cache ──────────────────────────────────────────────────────────────

_KLINE_PERSIST_CACHE: Dict[tuple, tuple] = {}
_KLINE_PERSIST_TTL = 45

async def _get_klines(
    symbol: str, interval: str, limit: int,
    http_client, cache: Optional[Dict] = None
) -> Optional[List]:
    import time as _time
    # Route stocks/forex/indices through yfinance — cache stashes the strategy's
    # asset_class under '__asset_class__' (set by evaluate_strategy_conditions).
    _asset_class = (cache or {}).get("__asset_class__") if isinstance(cache, dict) else None
    if _asset_class and _asset_class != "crypto":
        _ckey = (symbol, interval, limit, _asset_class)
        if cache is not None and _ckey in cache:
            return cache[_ckey]
        try:
            from app.services.tradfi_prices import get_klines as _tradfi_klines
            kl = await _tradfi_klines(symbol, _asset_class, interval, max(limit, 200))
        except Exception as _e:
            logger.debug(f"tradfi klines fetch failed for {symbol} ({_asset_class}): {_e}")
            kl = []
        if not kl:
            return None
        sliced = kl[-limit:] if len(kl) > limit else kl
        if cache is not None:
            cache[_ckey] = sliced
        return sliced

    key = (symbol, interval, limit)
    if cache is not None and key in cache:
        return cache[key]
    if cache is not None:
        for _ck, cv in list(cache.items()):
            # Skip non-tuple keys ('__asset_class__' marker) and any tuple shape
            # other than the legacy 3-tuple (symbol, interval, limit).
            if not isinstance(_ck, tuple) or len(_ck) != 3:
                continue
            cs, ci, cl = _ck
            if not isinstance(cl, int):
                continue
            if cs == symbol and ci == interval and cl >= limit and cv:
                sliced = cv[-limit:]
                cache[key] = sliced
                return sliced
    _now = _time.monotonic()
    _pkey = (symbol, interval)
    _pcached = _KLINE_PERSIST_CACHE.get(_pkey)
    if _pcached:
        _pdata, _pfetched, _plimit = _pcached
        if _now - _pfetched < _KLINE_PERSIST_TTL and _plimit >= limit and _pdata:
            sliced = _pdata[-limit:]
            if cache is not None:
                cache[key] = sliced
            return sliced
    mexc_interval = interval.replace("1h", "60m").replace("2h", "120m").replace("3h", "180m")
    fetch_limit = max(limit, 200)
    sources = [
        ("https://api.mexc.com/api/v3/klines",     {"symbol": symbol, "interval": mexc_interval, "limit": fetch_limit}),
        ("https://api.binance.com/api/v3/klines",   {"symbol": symbol, "interval": interval,      "limit": fetch_limit}),
        ("https://fapi.binance.com/fapi/v1/klines", {"symbol": symbol, "interval": interval,      "limit": fetch_limit}),
    ]
    for url, params in sources:
        try:
            resp = await http_client.get(url, params=params, timeout=2)
            if resp.status_code == 200:
                klines = resp.json()
                if klines and isinstance(klines, list):
                    _KLINE_PERSIST_CACHE[_pkey] = (klines, _now, len(klines))
                    sliced = klines[-limit:] if len(klines) > limit else klines
                    if cache is not None:
                        cache[key] = sliced
                        cache[(symbol, interval, len(klines))] = klines
                    return sliced
        except Exception as e:
            logger.debug(f"Klines fetch failed ({url}) {symbol} {interval}: {e}")
            continue
    return None


# ─── 1. INDICATOR evaluator ────────────────────────────────────────────────────

async def eval_indicator(
    cond: Dict, price_data: Dict, enhanced_ta: Dict,
    symbol: str, http_client, cache: Dict
) -> Tuple[bool, str]:
    name = cond.get("name", "").lower()
    op   = cond.get("operator", "gt")
    val  = float(cond.get("value", 0))
    tf   = cond.get("timeframe", "15m")

    # ── RSI ──────────────────────────────────────────────────────────────────
    if name == "rsi":
        key = "rsi_1h" if "1h" in tf else "rsi_15m"
        rsi = enhanced_ta.get(key) or price_data.get("rsi", 50)
        if rsi is None: return False, "RSI unavailable"
        return _cmp(rsi, op, val), f"RSI({tf})={rsi:.1f}"

    # ── MACD ─────────────────────────────────────────────────────────────────
    if name in ("macd", "macd_hist"):
        macd = enhanced_ta.get("macd", {})
        if not macd: return False, "MACD unavailable"
        cross = macd.get("crossover", "")
        sub   = cond.get("condition", op)
        if name == "macd_hist":
            hist = macd.get("histogram", 0)
            return _cmp(hist, op, val), f"MACD hist={hist:.4f}"
        if sub in ("bullish", "bullish_cross"):   return cross in ("BULLISH","BULLISH_CROSS"), f"MACD={cross}"
        if sub in ("bearish", "bearish_cross"):   return cross in ("BEARISH","BEARISH_CROSS"), f"MACD={cross}"
        if sub == "crosses_above":                return cross == "BULLISH_CROSS", f"MACD={cross}"
        if sub == "crosses_below":                return cross == "BEARISH_CROSS", f"MACD={cross}"
        return _cmp(macd.get("histogram", 0), op, val), f"MACD={cross}"

    # ── EMA basic ────────────────────────────────────────────────────────────
    if name in ("ema", "ema_cross"):
        ema_d = enhanced_ta.get("ema_cross", {})
        if not ema_d: return False, "EMA unavailable"
        sig = ema_d.get("signal", "")
        sub = cond.get("condition", op)
        if sub in ("bullish", "golden_cross"):  return sig in ("BULLISH","GOLDEN_CROSS"), f"EMA={sig}"
        if sub in ("bearish", "death_cross"):   return sig in ("BEARISH","DEATH_CROSS"), f"EMA={sig}"
        return _cmp(ema_d.get("spread_pct", 0), op, val), f"EMA={sig}"

    # ── EMA ribbon ───────────────────────────────────────────────────────────
    if name in ("ema_ribbon", "ribbon"):
        klines = await _get_klines(symbol, tf, 210, http_client, cache)
        if not klines or len(klines) < 200: return False, "Not enough data for EMA ribbon"
        closes = _closes(klines)
        periods = [int(p) for p in cond.get("periods", [9, 21, 55, 100, 200])]
        emas = [_ema(closes, p) for p in periods if len(closes) >= p]
        if len(emas) < 2: return False, "EMA ribbon insufficient data"
        sub = cond.get("condition", "aligned_bullish")
        if sub == "aligned_bullish":
            result = all(emas[i] > emas[i+1] for i in range(len(emas)-1))
        elif sub == "aligned_bearish":
            result = all(emas[i] < emas[i+1] for i in range(len(emas)-1))
        else:
            result = False
        return result, f"EMA ribbon ({','.join(str(p) for p in periods)}) {'aligned' if result else 'not aligned'}"

    # ── SMA (Simple Moving Average) ──────────────────────────────────────────
    # Supports price_above / price_below / above_ribbon / below_ribbon /
    # inside_ribbon / bullish_cross / bearish_cross.
    # source= "close" (default) | "high" | "low"
    if name in ("sma", "sma_cross", "sma_ribbon"):
        period  = int(cond.get("period", 200))
        period2 = int(cond.get("period2", 0))
        source  = cond.get("source", "close").lower()
        need    = max(period + 10, period2 + 10 if period2 else 0, 210)
        klines  = await _get_klines(symbol, tf, need, http_client, cache)
        if not klines or len(klines) < period:
            return False, f"SMA({period}) insufficient data ({len(klines) if klines else 0} bars)"

        def _src(kl, s):
            if s == "high":  return [float(k[2]) for k in kl]
            if s == "low":   return [float(k[3]) for k in kl]
            return _closes(kl)

        src_data  = _src(klines, source)
        sma_val   = _sma(src_data, period)
        curr      = float(klines[-1][4])
        sub_c     = cond.get("condition", "price_above")

        def _fmtprice(p):
            return f"{p:.6f}" if p < 0.01 else f"{p:.4f}" if p < 1 else f"{p:,.3f}"

        if sub_c in ("above", "price_above"):
            ok = sma_val is not None and curr > sma_val
            return ok, f"Price {'>' if ok else '<'} SMA({period},{source}) {_fmtprice(curr)} vs {_fmtprice(sma_val or 0)}"

        if sub_c in ("below", "price_below"):
            ok = sma_val is not None and curr < sma_val
            return ok, f"Price {'<' if ok else '>'} SMA({period},{source}) {_fmtprice(curr)} vs {_fmtprice(sma_val or 0)}"

        # Ribbon-specific: use SMA(high) and SMA(low) to define the band
        if sub_c in ("above_ribbon", "above_high"):
            sma_high = _sma(_src(klines, "high"), period)
            ok = sma_high is not None and curr > sma_high
            return ok, f"Price {'above' if ok else 'inside/below'} SMA({period}) ribbon high"

        if sub_c in ("below_ribbon", "below_low"):
            sma_low = _sma(_src(klines, "low"), period)
            ok = sma_low is not None and curr < sma_low
            return ok, f"Price {'below' if ok else 'inside/above'} SMA({period}) ribbon low"

        if sub_c == "inside_ribbon":
            sma_high = _sma(_src(klines, "high"), period)
            sma_low  = _sma(_src(klines, "low"),  period)
            ok = (sma_high is not None and sma_low is not None
                  and sma_low <= curr <= sma_high)
            return ok, f"Price {'inside' if ok else 'outside'} SMA({period}) ribbon"

        if sub_c in ("bullish_cross", "crosses_above") and period2:
            sma_fast = sma_val
            sma_slow = _sma(src_data, period2)
            prev_f   = _sma(src_data[:-1], period)  if len(src_data) > period  else None
            prev_s   = _sma(src_data[:-1], period2) if len(src_data) > period2 else None
            if all(v is not None for v in [sma_fast, sma_slow, prev_f, prev_s]):
                ok = prev_f <= prev_s and sma_fast > sma_slow
                return ok, f"SMA({period}) {'crossed above' if ok else 'below'} SMA({period2})"
            return False, f"SMA cross insufficient data"

        if sub_c in ("bearish_cross", "crosses_below") and period2:
            sma_fast = sma_val
            sma_slow = _sma(src_data, period2)
            prev_f   = _sma(src_data[:-1], period)  if len(src_data) > period  else None
            prev_s   = _sma(src_data[:-1], period2) if len(src_data) > period2 else None
            if all(v is not None for v in [sma_fast, sma_slow, prev_f, prev_s]):
                ok = prev_f >= prev_s and sma_fast < sma_slow
                return ok, f"SMA({period}) {'crossed below' if ok else 'above'} SMA({period2})"
            return False, f"SMA cross insufficient data"

        return _cmp(sma_val or 0, op, val), (f"SMA({period})={_fmtprice(sma_val)}" if sma_val else "SMA unavailable")

    # ── Bollinger Bands ──────────────────────────────────────────────────────
    if name in ("bb", "bollinger"):
        bb = enhanced_ta.get("bollinger", {})
        if not bb: return False, "BB unavailable"
        sub = cond.get("condition", "")
        pb  = bb.get("percent_b", 50)
        if sub == "squeeze":       return bb.get("squeeze") in ("SQUEEZE","TIGHT"), "BB Squeeze"
        if sub == "above_upper":   return pb > 100, f"BB %B={pb:.0f}"
        if sub == "below_lower":   return pb < 0,   f"BB %B={pb:.0f}"
        if sub == "upper_touch":   return pb >= 95,  f"BB %B={pb:.0f}"
        if sub == "lower_touch":   return pb <= 5,   f"BB %B={pb:.0f}"
        if sub == "overbought":    return pb >= val, f"BB %B={pb:.0f}"
        if sub == "oversold":      return pb <= val, f"BB %B={pb:.0f}"
        if sub == "mean_reversion":return 40 <= pb <= 60, f"BB %B={pb:.0f}"
        return _cmp(pb, op, val), f"BB %B={pb:.0f}"

    # ── VWAP ─────────────────────────────────────────────────────────────────
    if name == "vwap":
        vwap = enhanced_ta.get("vwap", {})
        if not vwap: return False, "VWAP unavailable"
        dev = vwap.get("deviation_pct", 0)
        sub = cond.get("condition", "")
        if sub == "above": return dev > 0, f"VWAP dev={dev:+.2f}%"
        if sub == "below": return dev < 0, f"VWAP dev={dev:+.2f}%"
        return _cmp(dev, op, val), f"VWAP dev={dev:+.2f}%"

    # ── Volume ───────────────────────────────────────────────────────────────
    if name in ("volume", "volume_ratio"):
        vr = price_data.get("volume_ratio", 1.0)
        return _cmp(vr, op, val), f"Vol ratio={vr:.2f}×"

    # ── Stochastic RSI ───────────────────────────────────────────────────────
    if name in ("stoch_rsi", "stochrsi"):
        klines = await _get_klines(symbol, tf, 200, http_client, cache)
        if not klines or len(klines) < 30: return False, "StochRSI insufficient data"
        closes = _closes(klines)
        rsi_vals = _rsi_values(closes, 14)
        if len(rsi_vals) < 14: return False, "StochRSI insufficient RSI"
        period = 14
        stoch = []
        for i in range(period, len(rsi_vals)):
            window = rsi_vals[i-period:i]
            mn, mx = min(window), max(window)
            stoch.append((rsi_vals[i] - mn) / (mx - mn) * 100 if mx != mn else 50)
        if not stoch: return False, "StochRSI failed"
        k_line = _ema_list(stoch, 3)[-1]
        sub = cond.get("condition", "")
        if sub == "oversold":        return k_line < 20, f"StochRSI K={k_line:.1f}"
        if sub == "overbought":      return k_line > 80, f"StochRSI K={k_line:.1f}"
        if sub == "bullish_cross":   # K crossed above D
            k_list = _ema_list(stoch, 3)
            d_list = _ema_list(k_list, 3)
            crossed = len(k_list) >= 2 and k_list[-2] < d_list[-2] and k_list[-1] > d_list[-1]
            return crossed, f"StochRSI bullish cross K={k_line:.1f}"
        if sub == "bearish_cross":
            k_list = _ema_list(stoch, 3)
            d_list = _ema_list(k_list, 3)
            crossed = len(k_list) >= 2 and k_list[-2] > d_list[-2] and k_list[-1] < d_list[-1]
            return crossed, f"StochRSI bearish cross K={k_line:.1f}"
        return _cmp(k_line, op, val), f"StochRSI K={k_line:.1f}"

    # ── SuperTrend ───────────────────────────────────────────────────────────
    if name in ("supertrend",):
        period = int(cond.get("period", 10))
        mult   = float(cond.get("multiplier", 3.0))
        klines = await _get_klines(symbol, tf, period * 6 + 20, http_client, cache)
        if not klines or len(klines) < period + 2: return False, "SuperTrend insufficient data"
        closes = _closes(klines)
        highs  = _highs(klines)
        lows   = _lows(klines)
        atrs   = _atr_values(klines, period)
        direction = 1  # 1=bullish, -1=bearish
        prev_upper = prev_lower = None
        st_dir = []
        for i in range(len(closes)):
            if i >= len(atrs): break
            hl2 = (highs[i] + lows[i]) / 2
            upper = hl2 + mult * atrs[i]
            lower = hl2 - mult * atrs[i]
            if prev_upper is not None:
                upper = min(upper, prev_upper) if closes[i-1] <= prev_upper else upper
                lower = max(lower, prev_lower) if closes[i-1] >= prev_lower else lower
            if closes[i] > (prev_upper or upper):  direction = 1
            elif closes[i] < (prev_lower or lower): direction = -1
            st_dir.append(direction)
            prev_upper, prev_lower = upper, lower
        if not st_dir: return False, "SuperTrend calc failed"
        cur = st_dir[-1]
        prev = st_dir[-2] if len(st_dir) >= 2 else cur
        sub = cond.get("condition", "bullish")
        if sub == "bullish":       return cur == 1,  f"SuperTrend={'BULL' if cur==1 else 'BEAR'}"
        if sub == "bearish":       return cur == -1, f"SuperTrend={'BULL' if cur==1 else 'BEAR'}"
        if sub == "bullish_flip":  return cur == 1 and prev == -1, f"SuperTrend flipped BULL"
        if sub == "bearish_flip":  return cur == -1 and prev == 1, f"SuperTrend flipped BEAR"
        return cur == 1, f"SuperTrend={'BULL' if cur==1 else 'BEAR'}"

    # ── ADX ──────────────────────────────────────────────────────────────────
    if name == "adx":
        klines = await _get_klines(symbol, tf, 120, http_client, cache)
        if not klines or len(klines) < 28: return False, "ADX insufficient data"
        highs = _highs(klines); lows = _lows(klines)
        atrs  = _atr_values(klines, 14)
        dm_pos, dm_neg = [], []
        for i in range(1, len(klines)):
            up   = highs[i] - highs[i-1]
            down = lows[i-1] - lows[i]
            dm_pos.append(up   if up > down and up > 0   else 0)
            dm_neg.append(down if down > up and down > 0 else 0)
        if len(atrs) < 14: return False, "ADX ATR error"
        di_pos = [100 * _ema(dm_pos[:i+14], 14) / atrs[i+13] for i in range(len(dm_pos)-13) if atrs[i+13] > 0]
        di_neg = [100 * _ema(dm_neg[:i+14], 14) / atrs[i+13] for i in range(len(dm_neg)-13) if atrs[i+13] > 0]
        if not di_pos or not di_neg: return False, "ADX di error"
        dx = [abs(p-n)/(p+n)*100 if p+n > 0 else 0 for p,n in zip(di_pos, di_neg)]
        adx = _ema(dx, 14) if len(dx) >= 14 else _sma(dx, len(dx))
        if adx is None: return False, "ADX none"
        sub = cond.get("condition", "")
        if sub == "trending":      return adx > 25, f"ADX={adx:.1f}"
        if sub == "strong_trend":  return adx > 40, f"ADX={adx:.1f}"
        if sub == "weak":          return adx < 20, f"ADX={adx:.1f}"
        if sub == "ranging":       return adx < 25, f"ADX={adx:.1f} (ranging)"
        return _cmp(adx, op, val), f"ADX={adx:.1f}"

    # ── ATR expansion ────────────────────────────────────────────────────────
    if name in ("atr", "atr_expansion"):
        klines = await _get_klines(symbol, tf, 80, http_client, cache)
        if not klines or len(klines) < 50: return False, "ATR insufficient data"
        atrs = _atr_values(klines, 14)
        if len(atrs) < 30: return False, "ATR too short"
        cur = atrs[-1]
        avg = _sma(atrs[:-1], min(50, len(atrs)-1))
        if not avg: return False, "ATR avg error"
        sub  = cond.get("condition", "expanding")
        mult = float(cond.get("multiplier", 1.2))
        if sub == "expanding":   return cur > avg * mult, f"ATR={cur:.6f} avg={avg:.6f}"
        if sub == "contracting": return cur < avg / mult, f"ATR contracting"
        return _cmp(cur / avg, op, val), f"ATR ratio={cur/avg:.2f}"

    # ── Williams %R ──────────────────────────────────────────────────────────
    if name in ("williams_r", "williamsr", "wr"):
        period = int(cond.get("period", 14))
        klines = await _get_klines(symbol, tf, period + 5, http_client, cache)
        if not klines or len(klines) < period: return False, "Williams %R insufficient data"
        highs = _highs(klines[-period:])
        lows  = _lows(klines[-period:])
        close = _closes(klines)[-1]
        hh, ll = max(highs), min(lows)
        wr = (hh - close) / (hh - ll) * -100 if hh != ll else -50
        sub = cond.get("condition", "")
        if sub == "oversold":   return wr < -80, f"W%R={wr:.1f}"
        if sub == "overbought": return wr > -20, f"W%R={wr:.1f}"
        return _cmp(wr, op, val), f"W%R={wr:.1f}"

    # ── CCI ──────────────────────────────────────────────────────────────────
    if name == "cci":
        period   = int(cond.get("period", 20))
        ma_type  = cond.get("ma_type", "").strip().lower()
        ma_period = int(cond.get("ma_period", 3))
        # Fetch enough candles for CCI series + MA smoothing window
        need = period + ma_period + 10
        klines = await _get_klines(symbol, tf, need, http_client, cache)
        if not klines or len(klines) < period: return False, "CCI insufficient data"

        # Build full CCI series across all available bars
        tps = [(float(k[2]) + float(k[3]) + float(k[4])) / 3 for k in klines]
        cci_series: List[float] = []
        for i in range(period - 1, len(tps)):
            window = tps[i - period + 1:i + 1]
            sma_tp   = sum(window) / period
            mean_dev = sum(abs(t - sma_tp) for t in window) / period
            cci_series.append((window[-1] - sma_tp) / (0.015 * mean_dev) if mean_dev > 0 else 0)

        if not cci_series: return False, "CCI insufficient data"

        # Optional MA smoothing (SMA/EMA/SMMA/WMA/VWMA) applied to the CCI series
        if ma_type and ma_type not in ("", "none"):
            vols = _vols(klines[period - 1:]) if ma_type == "vwma" else None
            smoothed = _apply_ma_smooth(cci_series, ma_type, ma_period, vols)
            cci   = smoothed[-1] if smoothed else cci_series[-1]
            label = f"CCI({ma_type.upper()}{ma_period})={cci:.1f}"
        else:
            cci   = cci_series[-1]
            label = f"CCI={cci:.1f}"

        sub = cond.get("condition", "")
        if sub == "overbought": return cci > 100,  label
        if sub == "oversold":   return cci < -100, label
        if sub == "bullish":    return cci > 0,    label   # used by Trend Magic / zero-cross
        if sub == "bearish":    return cci < 0,    label
        return _cmp(cci, op, val), label

    # ── OBV ──────────────────────────────────────────────────────────────────
    if name == "obv":
        klines = await _get_klines(symbol, tf, 50, http_client, cache)
        if not klines or len(klines) < 10: return False, "OBV insufficient data"
        closes = _closes(klines); vols = _vols(klines)
        obv = [0.0]
        for i in range(1, len(closes)):
            if closes[i] > closes[i-1]:   obv.append(obv[-1] + vols[i])
            elif closes[i] < closes[i-1]: obv.append(obv[-1] - vols[i])
            else:                         obv.append(obv[-1])
        sub = cond.get("condition", "bullish")
        look = int(cond.get("lookback", 5))
        recent = obv[-look:]
        if sub in ("bullish","rising"):  return obv[-1] > obv[-2], f"OBV rising"
        if sub in ("bearish","falling"): return obv[-1] < obv[-2], f"OBV falling"
        if sub in ("cross_up","signal_cross_up"):   # OBV vs its own EMA
            signal = sum(obv[-5:]) / 5
            return obv[-1] > signal and obv[-2] <= signal, f"OBV crossed above signal"
        if sub in ("cross_down","signal_cross_down"):
            signal = sum(obv[-5:]) / 5
            return obv[-1] < signal and obv[-2] >= signal, f"OBV crossed below signal"
        if sub == "divergence_bullish":  # price down, OBV up
            p_down = closes[-1] < closes[-look]
            o_up   = obv[-1]   > obv[-look]
            return p_down and o_up, f"OBV bull divergence"
        if sub == "divergence_bearish":  # price up, OBV down
            p_up   = closes[-1] > closes[-look]
            o_down = obv[-1]   < obv[-look]
            return p_up and o_down, f"OBV bear divergence"
        return obv[-1] > obv[-2], f"OBV={obv[-1]:.0f}"

    # ── Heikin Ashi ──────────────────────────────────────────────────────────
    if name in ("heikin_ashi", "ha"):
        klines = await _get_klines(symbol, tf, 20, http_client, cache)
        if not klines or len(klines) < 5: return False, "HA insufficient data"
        ha_o, ha_c, ha_h, ha_l = [], [], [], []
        for i, k in enumerate(klines):
            o, h, l, c = float(k[1]), float(k[2]), float(k[3]), float(k[4])
            hc = (o + h + l + c) / 4
            ho = (ha_o[-1] + ha_c[-1]) / 2 if i > 0 else (o + c) / 2
            ha_c.append(hc); ha_o.append(ho)
            ha_h.append(max(h, ho, hc)); ha_l.append(min(l, ho, hc))
        sub = cond.get("condition", "bullish")
        cur_bull  = ha_c[-1] > ha_o[-1]
        prev_bull = ha_c[-2] > ha_o[-2]
        no_lower_wick = ha_l[-1] >= ha_o[-1] * 0.9999
        no_upper_wick = ha_h[-1] <= ha_c[-1] * 1.0001
        if sub == "bullish":       return cur_bull, f"HA={'bullish' if cur_bull else 'bearish'}"
        if sub == "bearish":       return not cur_bull, f"HA={'bullish' if cur_bull else 'bearish'}"
        if sub == "bullish_flip":  return cur_bull and not prev_bull, f"HA flipped bullish"
        if sub == "bearish_flip":  return not cur_bull and prev_bull, f"HA flipped bearish"
        if sub in ("strong_bull","no_lower_shadow"):  return cur_bull and no_lower_wick, f"HA strong bull (no lower wick)"
        if sub in ("strong_bear","no_upper_shadow"):  return not cur_bull and no_upper_wick, f"HA strong bear"
        if sub == "doji":
            body = abs(ha_c[-1] - ha_o[-1])
            candle_range = ha_h[-1] - ha_l[-1]
            return (body / candle_range < 0.2) if candle_range > 0 else False, "HA doji"
        return cur_bull, f"HA={'bull' if cur_bull else 'bear'}"

    # ── Ichimoku ─────────────────────────────────────────────────────────────
    if name in ("ichimoku", "ichi"):
        klines = await _get_klines(symbol, tf, 200, http_client, cache)
        if not klines or len(klines) < 52: return False, "Ichimoku insufficient data"
        def midpoint(ks, p): return (max(_highs(ks[-p:])) + min(_lows(ks[-p:]))) / 2
        tenkan  = midpoint(klines, 9)
        kijun   = midpoint(klines, 26)
        span_a  = (tenkan + kijun) / 2
        span_b  = midpoint(klines, 52)
        close   = float(klines[-1][4])
        sub     = cond.get("condition", "above_cloud")
        cloud_top    = max(span_a, span_b)
        cloud_bottom = min(span_a, span_b)
        if sub == "above_cloud":      return close > cloud_top,    f"Ichimoku above cloud"
        if sub == "below_cloud":      return close < cloud_bottom, f"Ichimoku below cloud"
        if sub == "in_cloud":         return cloud_bottom <= close <= cloud_top, f"Ichimoku in cloud"
        if sub in ("tk_cross_bullish","tk_cross_up"):   return tenkan > kijun, f"T={tenkan:.4f} K={kijun:.4f}"
        if sub in ("tk_cross_bearish","tk_cross_down"): return tenkan < kijun, f"T={tenkan:.4f} K={kijun:.4f}"
        if sub in ("bullish_cloud","kumo_bullish"):   return span_a > span_b, f"Cloud bullish"
        if sub in ("bearish_cloud","kumo_bearish"):   return span_a < span_b, f"Cloud bearish"
        if sub == "kumo_breakout_up":    # price just crossed above cloud
            prev_close = float(klines[-2][4])
            return close > cloud_top and prev_close <= cloud_top, f"Kumo breakout up"
        if sub == "kumo_breakout_down":  # price just crossed below cloud
            prev_close = float(klines[-2][4])
            return close < cloud_bottom and prev_close >= cloud_bottom, f"Kumo breakout down"
        return close > cloud_top, f"Ichimoku above cloud"


    # ── MFI (Money Flow Index) ────────────────────────────────────────────────
    if name == "mfi":
        klines = await _get_klines(symbol, tf, 100, http_client, cache)
        if not klines or len(klines) < 15: return False, "MFI insufficient data"
        highs = _highs(klines); lows = _lows(klines)
        closes = _closes(klines); vols = _vols(klines)
        period = int(cond.get("period", 14))
        tps = [(highs[i] + lows[i] + closes[i]) / 3 for i in range(len(closes))]
        pos_flow = neg_flow = 0.0
        for i in range(1, min(period + 1, len(tps))):
            raw = tps[i] * vols[i]
            if tps[i] > tps[i-1]: pos_flow += raw
            elif tps[i] < tps[i-1]: neg_flow += raw
        mfi_ratio = (pos_flow / neg_flow) if neg_flow > 0 else 100
        mfi = 100 - (100 / (1 + mfi_ratio))
        sub = cond.get("condition", "oversold")
        if sub == "oversold":   return mfi < 20,  f"MFI={mfi:.1f} (oversold)"
        if sub == "overbought": return mfi > 80,  f"MFI={mfi:.1f} (overbought)"
        if sub == "rising":     return mfi > 50,  f"MFI={mfi:.1f} rising"
        if sub == "falling":    return mfi < 50,  f"MFI={mfi:.1f} falling"
        return mfi < 20, f"MFI={mfi:.1f}"

    # ── Donchian Channel ─────────────────────────────────────────────────────
    if name == "donchian":
        period = int(cond.get("period", 20))
        klines = await _get_klines(symbol, tf, period + 5, http_client, cache)
        if not klines or len(klines) < period: return False, "Donchian insufficient data"
        highs = _highs(klines); lows = _lows(klines); closes = _closes(klines)
        upper = max(highs[-period:])
        lower = min(lows[-period:])
        prev_upper = max(highs[-(period+1):-1])
        prev_lower = min(lows[-(period+1):-1])
        price = closes[-1]
        sub = cond.get("condition", "upper_break")
        if sub == "upper_break":  return price >= upper, f"Donchian upper break {upper:.4f}"
        if sub == "lower_break":  return price <= lower, f"Donchian lower break {lower:.4f}"
        channel_range = upper - lower
        near_pct = 0.05  # within 5% of range
        if sub == "near_upper":   return (upper - price) <= channel_range * near_pct, f"Donchian near upper"
        if sub == "near_lower":   return (price - lower) <= channel_range * near_pct, f"Donchian near lower"
        return price >= upper, f"Donchian upper={upper:.4f}"

    # ── Aroon ────────────────────────────────────────────────────────────────
    if name == "aroon":
        period = int(cond.get("period", 25))
        klines = await _get_klines(symbol, tf, period + 5, http_client, cache)
        if not klines or len(klines) < period: return False, "Aroon insufficient data"
        highs = _highs(klines); lows = _lows(klines)
        recent_h = highs[-period:]; recent_l = lows[-period:]
        bars_since_high = period - 1 - recent_h.index(max(recent_h))
        bars_since_low  = period - 1 - recent_l.index(min(recent_l))
        aroon_up   = ((period - bars_since_high) / period) * 100
        aroon_down = ((period - bars_since_low)  / period) * 100
        sub = cond.get("condition", "bullish")
        if sub == "bullish":    return aroon_up > 70,   f"Aroon Up={aroon_up:.0f}"
        if sub == "bearish":    return aroon_down > 70, f"Aroon Down={aroon_down:.0f}"
        if sub == "cross_up":   return aroon_up > aroon_down, f"Aroon Up({aroon_up:.0f}) > Down({aroon_down:.0f})"
        if sub == "cross_down": return aroon_down > aroon_up, f"Aroon Down({aroon_down:.0f}) > Up({aroon_up:.0f})"
        return aroon_up > 70, f"Aroon Up={aroon_up:.0f}"

    # ── ROC (Rate of Change) ─────────────────────────────────────────────────
    if name == "roc":
        period = int(cond.get("period", 10))
        klines = await _get_klines(symbol, tf, period + 10, http_client, cache)
        if not klines or len(klines) < period + 2: return False, "ROC insufficient data"
        closes = _closes(klines)
        if closes[-(period+1)] == 0: return False, "ROC division by zero"
        roc = ((closes[-1] - closes[-(period+1)]) / closes[-(period+1)]) * 100
        roc_prev = ((closes[-2] - closes[-(period+2)]) / closes[-(period+2)]) * 100
        sub = cond.get("condition", "positive")
        if sub == "positive":     return roc > 0,  f"ROC={roc:.2f}%"
        if sub == "negative":     return roc < 0,  f"ROC={roc:.2f}%"
        if sub == "cross_up":     return roc > 0 and roc_prev <= 0, f"ROC crossed above 0"
        if sub == "cross_down":   return roc < 0 and roc_prev >= 0, f"ROC crossed below 0"
        if sub == "accelerating": return roc > roc_prev and roc > 0, f"ROC accelerating ({roc:.2f}% vs {roc_prev:.2f}%)"
        return roc > 0, f"ROC={roc:.2f}%"

    # ── Keltner Channel ──────────────────────────────────────────────────────
    if name == "keltner":
        klines = await _get_klines(symbol, tf, 30, http_client, cache)
        if not klines or len(klines) < 20: return False, "Keltner insufficient data"
        closes = _closes(klines); highs = _highs(klines); lows = _lows(klines)
        period = int(cond.get("period", 20))
        mult   = float(cond.get("multiplier", 1.5))
        sma    = _sma(closes, period)
        std    = _stdev(closes[-period:])
        atr_v  = _atr(klines, period) or 0
        bb_upper = sma + 2.0 * std; bb_lower = sma - 2.0 * std
        kc_upper = sma + mult * atr_v; kc_lower = sma - mult * atr_v
        price  = closes[-1]
        sub    = cond.get("condition", "squeeze")
        if sub == "squeeze":      # BB inside KC = low-volatility compression
            return (bb_upper < kc_upper and bb_lower > kc_lower), f"Keltner squeeze={'ON' if bb_upper < kc_upper else 'OFF'}"
        if sub == "above_upper":  return price > kc_upper, f"Price {price:.4f} > KC upper {kc_upper:.4f}"
        if sub == "below_lower":  return price < kc_lower, f"Price {price:.4f} < KC lower {kc_lower:.4f}"
        if sub == "inside_bands": return kc_lower <= price <= kc_upper, f"Price inside KC"
        return price > kc_upper, f"Keltner upper={kc_upper:.4f}"

    # ── Squeeze Momentum ─────────────────────────────────────────────────────
    if name in ("squeeze", "squeeze_momentum"):
        klines = await _get_klines(symbol, tf, 30, http_client, cache)
        if not klines or len(klines) < 20: return False, "Squeeze insufficient data"
        closes = _closes(klines); highs = _highs(klines); lows = _lows(klines)
        period = 20
        # BB
        sma = _sma(closes, period)
        std = _stdev(closes[-period:])
        bb_upper = sma + 2.0 * std; bb_lower = sma - 2.0 * std
        # Keltner
        atr_v = _atr(klines, period) or 0
        kc_upper = sma + 1.5 * atr_v; kc_lower = sma - 1.5 * atr_v
        squeeze_on = bb_upper < kc_upper and bb_lower > kc_lower
        # Momentum
        highest_h = max(highs[-period:]); lowest_l = min(lows[-period:])
        mid = (highest_h + lowest_l) / 2
        delta = closes[-1] - (mid + (sma or closes[-1])) / 2
        sub = cond.get("condition", "firing")
        if sub == "firing":   return not squeeze_on and delta > 0, f"Squeeze fired bull"
        if sub == "on":       return squeeze_on, f"Squeeze is ON"
        if sub == "off":      return not squeeze_on, f"Squeeze released"
        if sub == "bull_mom": return not squeeze_on and delta > 0, f"Squeeze bullish momentum"
        if sub == "bear_mom": return not squeeze_on and delta < 0, f"Squeeze bearish momentum"
        return squeeze_on, f"Squeeze={'ON' if squeeze_on else 'OFF'}"

    return False, f"Unknown indicator: {name}"


# ─── 2. PRICE MOMENTUM ────────────────────────────────────────────────────────

async def eval_price_momentum(
    cond: Dict, symbol: str, http_client, cache: Dict
) -> Tuple[bool, str]:
    window       = int(cond.get("window_minutes", 10))
    op           = cond.get("operator", "gt")
    val          = float(cond.get("value", 5))
    req_dir      = cond.get("direction", "any")
    interval     = "1m" if window <= 15 else "5m"
    interval_mins = 1 if window <= 15 else 5
    # Number of candles needed to cover exactly `window` minutes
    # e.g. window=30, interval=5m → ceil(30/5)+2 = 8 candles = ~40 min
    n_candles    = max((window + interval_mins - 1) // interval_mins + 2, 4)
    klines       = await _get_klines(symbol, interval, n_candles, http_client, cache)
    if not klines or len(klines) < 2: return False, "Momentum: not enough candles"
    open_p  = float(klines[0][1])
    close_p = float(klines[-1][4])
    pct     = (close_p - open_p) / open_p * 100
    if req_dir == "up"   and pct < 0: return False, f"Price Δ={pct:+.2f}% (need up)"
    if req_dir == "down" and pct > 0: return False, f"Price Δ={pct:+.2f}% (need down)"
    return _cmp(abs(pct), op, val), f"Price Δ({window}min)={pct:+.2f}%"


# ─── 3. VOLUME SPIKE ──────────────────────────────────────────────────────────

def eval_volume_spike(cond: Dict, price_data: Dict) -> Tuple[bool, str]:
    mult = float(cond.get("multiplier", 1.5))
    vr   = price_data.get("volume_ratio", 1.0)
    return vr >= mult, f"Vol={vr:.2f}× (need {mult}×)"


# ─── 4. SUPPORT / RESISTANCE ──────────────────────────────────────────────────

def eval_support_resistance(
    cond: Dict, enhanced_ta: Dict, current_price: float
) -> Tuple[bool, str]:
    sub  = cond.get("condition", "at_support")
    tol  = float(cond.get("tolerance_pct", 1.0)) / 100
    sr   = enhanced_ta.get("support_resistance", {})
    if not sr: return False, "S/R unavailable"
    supports    = sr.get("supports", [])
    resistances = sr.get("resistances", [])
    if sub == "at_support":
        for s in supports:
            if abs(current_price - s) / s <= tol:
                return True, f"At support {s:.6g}"
        return False, "Not at support"
    if sub == "at_resistance":
        for r in resistances:
            if abs(current_price - r) / r <= tol:
                return True, f"At resistance {r:.6g}"
        return False, "Not at resistance"
    if sub == "breakout_above":
        for r in resistances:
            if current_price > r * (1 + tol * 0.3):
                return True, f"Broke above {r:.6g}"
        return False, "No breakout above"
    if sub == "breakout_below":
        for s in supports:
            if current_price < s * (1 - tol * 0.3):
                return True, f"Broke below {s:.6g}"
        return False, "No breakout below"
    if sub == "between":
        top    = resistances[0] if resistances else None
        bottom = supports[0]   if supports    else None
        if top and bottom and bottom <= current_price <= top:
            return True, f"In range {bottom:.6g}–{top:.6g}"
        return False, "Not in range"
    return False, f"Unknown S/R: {sub}"


# ─── 5. FVG ───────────────────────────────────────────────────────────────────

def _detect_fvg(
    klines: List,
    *,
    min_gap_pct:      float = 0.0,
    min_gap_atr_mult: float = 0.0,
    disp_atr_mult:    float = 0.0,
    min_gap_usd:      float = 0.0,
    only_unfilled:    bool  = False,
    atr_period:       int   = 14,
) -> List[Dict]:
    """3-candle FVG detector with ICT-style quality filters.

    All filters default to 0 / False so existing wizard strategies that only
    pass {direction, condition} keep their original "any 3-candle gap counts"
    behaviour. Passing any non-zero filter activates ATR-aware gating.

    Each gap dict contains:
        type       : "bullish" | "bearish"
        top, bottom: float — price boundaries
        idx        : int   — index of the formation (middle) bar
        age        : int   — bars between formation and the latest bar
        size_pct   : float — width / mid_price * 100
        size_atr   : float — width / ATR(period); 0 if ATR n/a
        disp_atr   : float — formation-bar body / ATR; 0 if ATR n/a
        filled     : bool  — True if a later bar's range overlaps the gap
    """
    n = len(klines)
    if n < 3:
        return []

    needs_atr = (min_gap_atr_mult > 0) or (disp_atr_mult > 0)
    atr = _atr(klines, atr_period) if needs_atr else None
    if needs_atr and (atr is None or atr <= 0):
        atr = None  # gracefully disable ATR filters when not enough bars

    gaps: List[Dict] = []
    last_idx = n - 1
    for i in range(1, n - 1):
        try:
            c1h = float(klines[i-1][2]); c1l = float(klines[i-1][3])
            c3h = float(klines[i+1][2]); c3l = float(klines[i+1][3])
            mo  = float(klines[i][1]);   mc  = float(klines[i][4])
        except (IndexError, TypeError, ValueError):
            continue

        side = None
        top = bottom = 0.0
        if c1h < c3l:
            side, top, bottom = "bullish", c3l, c1h
        elif c1l > c3h:
            side, top, bottom = "bearish", c1l, c3h
        if side is None:
            continue

        gap_mid = (top + bottom) / 2.0
        if gap_mid <= 0:
            continue
        width    = top - bottom
        size_pct = width / gap_mid * 100.0

        # Width filter (% of price)
        if min_gap_pct > 0 and size_pct < min_gap_pct:
            continue
        # Width filter (× ATR) — volatility-aware
        size_atr = (width / atr) if atr else 0.0
        if atr and min_gap_atr_mult > 0 and size_atr < min_gap_atr_mult:
            continue
        # Width filter (absolute USD floor)
        if min_gap_usd > 0 and width < min_gap_usd:
            continue
        # ICT displacement filter — formation candle body must be a real
        # expansion bar, not a doji that happened to leave a gap.
        disp_atr = 0.0
        if atr and disp_atr_mult > 0:
            disp_atr = abs(mc - mo) / atr
            if disp_atr < disp_atr_mult:
                continue

        # Walk forward to determine fill status.
        filled = False
        for j in range(i + 2, n):
            try:
                hj = float(klines[j][2]); lj = float(klines[j][3])
            except (IndexError, TypeError, ValueError):
                continue
            if lj <= top and hj >= bottom:
                filled = True
                break
        if only_unfilled and filled:
            continue

        gaps.append({
            "type":     side,
            "top":      top,
            "bottom":   bottom,
            "idx":      i,
            "age":      last_idx - i,
            "size_pct": size_pct,
            "size_atr": size_atr,
            "disp_atr": disp_atr,
            "filled":   filled,
        })
    return gaps


async def eval_fvg(
    cond: Dict, symbol: str, current_price: float, http_client, cache: Dict
) -> Tuple[bool, str]:
    """FVG condition evaluator — ICT-style with optional ATR/displacement filters.

    Sub-conditions (`condition` field):
        gap_exists      → any qualifying FVG present (default)
        just_formed     → FVG completed in the last 1-2 bars (creation signal —
                          fires on the strong-displacement bar that left the gap)
        price_in_gap    → current price is inside the gap zone (mitigation/retest)
        tap_and_reject  → last bar wicked into the gap and closed outside in
                          the FVG's bias direction (bull FVG = closed above top,
                          bear FVG = closed below bottom)
        approaching     → price within 0.5% of the gap edge
        gap_filled      → price has fully traded through the gap

    Optional quality filters (zero/unset = disabled, preserves legacy behaviour):
        min_gap_pct       — minimum width as % of mid price
        min_gap_atr_mult  — minimum width × ATR(14)  ← e.g. 2.5 = "FVG ≥ 2.5×ATR"
        disp_atr_mult     — formation-bar body must be ≥ N×ATR (displacement)
        min_gap_usd       — absolute USD floor on width
        only_unfilled     — drop FVGs already touched
        max_age_bars      — only consider FVGs formed within the last N bars
    """
    sub                = cond.get("condition", "price_in_gap")
    direction          = cond.get("direction", "any")
    tf                 = cond.get("timeframe", "15m")
    # Legacy default kept at 20 for back-compat with existing wizard
    # strategies; we only widen the window when ATR-based filters are on.
    lookback_extra     = int(cond.get("lookback", 20)) + 2
    min_gap_pct        = float(cond.get("min_gap_pct",      0.0) or 0.0)
    min_gap_atr_mult   = float(cond.get("min_gap_atr_mult", 0.0) or 0.0)
    disp_atr_mult      = float(cond.get("disp_atr_mult",    0.0) or 0.0)
    min_gap_usd        = float(cond.get("min_gap_usd",      0.0) or 0.0)
    # ``only_unfilled`` may arrive as a real bool (JSON), or as a string
    # ('true'/'false'/'on'/'off'/'yes'/'no') from the wizard chip helper —
    # which serialises everything as a quoted string. Parse defensively so
    # the toggle works end-to-end.
    _ouf_raw           = cond.get("only_unfilled", False)
    if isinstance(_ouf_raw, str):
        only_unfilled  = _ouf_raw.strip().lower() in ("true", "1", "yes", "on")
    else:
        only_unfilled  = bool(_ouf_raw)
    max_age_bars       = int(cond.get("max_age_bars",       0)   or 0)

    # Ensure enough bars for ATR(14) when ATR-based filters are on.
    needs_atr = (min_gap_atr_mult > 0) or (disp_atr_mult > 0)
    lookback = max(lookback_extra, 60) if needs_atr else lookback_extra

    klines = await _get_klines(symbol, tf, lookback, http_client, cache)
    if not klines or len(klines) < 3:
        return False, "FVG insufficient data"

    gaps = _detect_fvg(
        klines,
        min_gap_pct      = min_gap_pct,
        min_gap_atr_mult = min_gap_atr_mult,
        disp_atr_mult    = disp_atr_mult,
        min_gap_usd      = min_gap_usd,
        only_unfilled    = only_unfilled,
    )
    if not gaps:
        msg = "No FVG"
        if min_gap_atr_mult or disp_atr_mult or min_gap_pct:
            msg += " (after quality filters)"
        return False, msg
    if direction != "any":
        gaps = [g for g in gaps if g["type"] == direction]
        if not gaps:
            return False, f"No {direction} FVG"
    if max_age_bars > 0:
        gaps = [g for g in gaps if g["age"] <= max_age_bars]
        if not gaps:
            return False, f"No FVG within last {max_age_bars} bars"

    # Pick the most recent qualifying gap (highest idx).
    gap = max(gaps, key=lambda g: g["idx"])
    label_size = f"{gap['size_pct']:.2f}%"
    if gap["size_atr"]:
        label_size += f"/{gap['size_atr']:.2f}×ATR"
    label = (
        f"{gap['type'].title()} FVG {gap['bottom']:.6g}–{gap['top']:.6g}"
        f" ({label_size}, {gap['age']}b ago)"
    )

    if sub == "gap_exists":
        return True, label
    if sub == "just_formed":
        # FVG completed on the most recent 3-candle pattern. The middle bar
        # (formation) is the second-to-last bar at minimum, so age 0-1.
        ok = gap["age"] <= 1
        return ok, (f"✓ Fresh {label}" if ok else f"✗ Stale FVG ({gap['age']}b old)")
    if sub == "price_in_gap":
        ok = gap["bottom"] <= current_price <= gap["top"]
        return ok, f"Price {'inside' if ok else 'not in'} {label}"
    if sub == "tap_and_reject":
        # Last bar wicked into the gap zone but closed outside in the FVG's
        # directional bias. Bull FVG below price = bullish reclaim → long;
        # Bear FVG above price = bearish rejection → short.
        try:
            last_h = float(klines[-1][2])
            last_l = float(klines[-1][3])
            last_c = float(klines[-1][4])
        except (IndexError, TypeError, ValueError):
            return False, "FVG: bad last candle"
        wicked_in = (last_l <= gap["top"]) and (last_h >= gap["bottom"])
        if gap["type"] == "bullish":
            ok = wicked_in and last_c >= gap["top"]
        else:
            ok = wicked_in and last_c <= gap["bottom"]
        return ok, f"{'✓ Rejection from' if ok else '✗ No rejection at'} {label}"
    if sub == "approaching":
        dist = min(abs(current_price - gap["top"]), abs(current_price - gap["bottom"]))
        near = dist / current_price < 0.005
        return near, f"Price {'approaching' if near else 'not near'} {label}"
    if sub == "gap_filled":
        if gap["type"] == "bullish":
            r = current_price <= gap["bottom"]
        else:
            r = current_price >= gap["top"]
        return r, f"FVG {'filled ✓' if r else 'unfilled'} ({label})"
    return False, f"Unknown FVG sub: {sub}"


# ─── 6. CANDLESTICK PATTERNS ──────────────────────────────────────────────────

async def eval_candlestick(
    cond: Dict, symbol: str, http_client, cache: Dict
) -> Tuple[bool, str]:
    tf      = cond.get("timeframe", "15m")
    pattern = cond.get("pattern", "bullish_engulfing").lower()
    klines  = await _get_klines(symbol, tf, 10, http_client, cache)
    if not klines or len(klines) < 3: return False, "Candlestick: not enough candles"
    o = _opens(klines);  h = _highs(klines)
    l = _lows(klines);   c = _closes(klines)
    body = [abs(c[i] - o[i]) for i in range(len(c))]
    rng  = [max(h[i] - l[i], 0.000001) for i in range(len(c))]

    def bull(i): return c[i] > o[i]
    def bear(i): return c[i] < o[i]
    def body_pct(i): return body[i] / rng[i]
    def upper_wick(i): return (h[i] - max(c[i], o[i])) / rng[i]
    def lower_wick(i): return (min(c[i], o[i]) - l[i]) / rng[i]

    n = len(c) - 1  # last candle index

    if pattern == "bullish_engulfing":
        r = n >= 1 and bear(n-1) and bull(n) and o[n] <= c[n-1] and c[n] >= o[n-1]
        return r, f"{'✓' if r else '✗'} Bullish Engulfing"

    if pattern == "bearish_engulfing":
        r = n >= 1 and bull(n-1) and bear(n) and o[n] >= c[n-1] and c[n] <= o[n-1]
        return r, f"{'✓' if r else '✗'} Bearish Engulfing"

    if pattern in ("hammer", "inverted_hammer"):
        small_body = body_pct(n) < 0.35
        if pattern == "hammer":
            r = small_body and lower_wick(n) > 0.55 and upper_wick(n) < 0.15
        else:
            r = small_body and upper_wick(n) > 0.55 and lower_wick(n) < 0.15
        return r, f"{'✓' if r else '✗'} {pattern.replace('_',' ').title()}"

    if pattern in ("shooting_star", "pin_bar"):
        small_body = body_pct(n) < 0.35
        r = small_body and upper_wick(n) > 0.55 and lower_wick(n) < 0.15
        return r, f"{'✓' if r else '✗'} {pattern.replace('_',' ').title()}"

    if pattern == "doji":
        r = body_pct(n) < 0.1
        return r, f"{'✓' if r else '✗'} Doji"

    if pattern == "dragonfly_doji":
        r = body_pct(n) < 0.1 and lower_wick(n) > 0.6 and upper_wick(n) < 0.1
        return r, f"{'✓' if r else '✗'} Dragonfly Doji"

    if pattern == "gravestone_doji":
        r = body_pct(n) < 0.1 and upper_wick(n) > 0.6 and lower_wick(n) < 0.1
        return r, f"{'✓' if r else '✗'} Gravestone Doji"

    if pattern == "morning_star":
        r = (n >= 2 and bear(n-2) and body_pct(n-1) < 0.3
             and bull(n) and c[n] > (o[n-2] + c[n-2]) / 2)
        return r, f"{'✓' if r else '✗'} Morning Star"

    if pattern == "evening_star":
        r = (n >= 2 and bull(n-2) and body_pct(n-1) < 0.3
             and bear(n) and c[n] < (o[n-2] + c[n-2]) / 2)
        return r, f"{'✓' if r else '✗'} Evening Star"

    if pattern in ("three_white_soldiers", "three_soldiers"):
        r = (n >= 2 and bull(n-2) and bull(n-1) and bull(n)
             and body_pct(n-2) > 0.5 and body_pct(n-1) > 0.5 and body_pct(n) > 0.5
             and c[n-1] > c[n-2] and c[n] > c[n-1])
        return r, f"{'✓' if r else '✗'} Three White Soldiers"

    if pattern in ("three_black_crows", "three_crows"):
        r = (n >= 2 and bear(n-2) and bear(n-1) and bear(n)
             and body_pct(n-2) > 0.5 and body_pct(n-1) > 0.5 and body_pct(n) > 0.5
             and c[n-1] < c[n-2] and c[n] < c[n-1])
        return r, f"{'✓' if r else '✗'} Three Black Crows"

    if pattern in ("tweezer_bottom", "tweezer_top"):
        if pattern == "tweezer_bottom":
            r = n >= 1 and bear(n-1) and bull(n) and abs(l[n] - l[n-1]) / rng[n] < 0.1
        else:
            r = n >= 1 and bull(n-1) and bear(n) and abs(h[n] - h[n-1]) / rng[n] < 0.1
        return r, f"{'✓' if r else '✗'} {pattern.replace('_',' ').title()}"

    if pattern == "inside_bar":
        r = n >= 1 and h[n] < h[n-1] and l[n] > l[n-1]
        return r, f"{'✓' if r else '✗'} Inside Bar"

    if pattern == "outside_bar":
        r = n >= 1 and h[n] > h[n-1] and l[n] < l[n-1]
        return r, f"{'✓' if r else '✗'} Outside Bar"

    if pattern == "marubozu":
        r = body_pct(n) > 0.9 and upper_wick(n) < 0.05 and lower_wick(n) < 0.05
        return r, f"{'✓' if r else '✗'} Marubozu"

    return False, f"Unknown pattern: {pattern}"


# ─── 7. CONSECUTIVE CANDLES ───────────────────────────────────────────────────

async def eval_consecutive_candles(
    cond: Dict, symbol: str, http_client, cache: Dict
) -> Tuple[bool, str]:
    count  = int(cond.get("count", 3))
    dirn   = cond.get("direction", "red").lower()
    tf     = cond.get("timeframe", "15m")
    klines = await _get_klines(symbol, tf, count + 3, http_client, cache)
    if not klines or len(klines) < count: return False, "Consecutive: not enough candles"
    recent = klines[-count:]
    if dirn == "red":
        r = all(float(k[4]) < float(k[1]) for k in recent)
    elif dirn == "green":
        r = all(float(k[4]) > float(k[1]) for k in recent)
    else:
        r = False
    return r, f"{count} consecutive {dirn} candles: {'✓' if r else '✗'}"


# ─── 8. MARKET STRUCTURE (BOS / CHoCH) ───────────────────────────────────────

async def eval_market_structure(
    cond: Dict, symbol: str, current_price: float, http_client, cache: Dict
) -> Tuple[bool, str]:
    sub    = cond.get("condition", "bos_bullish")
    tf     = cond.get("timeframe", "15m")
    klines = await _get_klines(symbol, tf, 120, http_client, cache)
    if not klines or len(klines) < 20: return False, "Market structure: not enough data"
    highs = _highs(klines[:-3]); lows = _lows(klines[:-3])
    sh_idx = _swing_highs(highs, lows)
    sl_idx = _swing_lows(highs, lows)
    if not sh_idx or not sl_idx: return False, "No swing structure found"
    last_sh = highs[sh_idx[-1]]
    last_sl = lows[sl_idx[-1]]
    if sub == "bos_bullish":
        r = current_price > last_sh
        return r, f"BOS bull: price={current_price:.6g} > swing_high={last_sh:.6g}"
    if sub == "bos_bearish":
        r = current_price < last_sl
        return r, f"BOS bear: price={current_price:.6g} < swing_low={last_sl:.6g}"
    if sub == "choch_bullish":
        # Was making lower highs, now broke previous high
        if len(sh_idx) < 2: return False, "CHoCH: need 2+ swing highs"
        prev_sh = highs[sh_idx[-2]]
        was_lower = last_sh < prev_sh
        r = was_lower and current_price > last_sh
        return r, f"CHoCH bull: prev_SH={prev_sh:.6g} last_SH={last_sh:.6g}"
    if sub == "choch_bearish":
        if len(sl_idx) < 2: return False, "CHoCH: need 2+ swing lows"
        prev_sl = lows[sl_idx[-2]]
        was_higher = last_sl > prev_sl
        r = was_higher and current_price < last_sl
        return r, f"CHoCH bear: prev_SL={prev_sl:.6g} last_SL={last_sl:.6g}"
    return False, f"Unknown MS condition: {sub}"


# ─── 9. ORDER BLOCKS ──────────────────────────────────────────────────────────

async def eval_order_block(
    cond: Dict, symbol: str, current_price: float, http_client, cache: Dict
) -> Tuple[bool, str]:
    ob_type  = cond.get("ob_type", cond.get("direction", "bullish"))
    tf       = cond.get("timeframe", "15m")
    tol      = float(cond.get("tolerance_pct", 1.0)) / 100
    # ── Quality filter: how "big"/significant the order block must be ──────────
    # strength presets map to thresholds measured RELATIVE to recent volatility
    # (so they work the same on crypto and forex). Default "any" = legacy
    # behaviour (no quality gate) so existing saved strategies are unchanged.
    strength = str(cond.get("strength", "any")).lower()
    _um      = cond.get("unmitigated_only", cond.get("fresh_only", False))
    unmit    = _um if isinstance(_um, bool) else str(_um).lower() in ("true", "1", "yes", "on")
    #                       (body_mult, impulse_atr, volume_mult)
    _PRESETS = {
        "any":           (0.0, 0.0, 0.0),
        "strong":        (1.2, 1.0, 1.2),
        "institutional": (1.8, 1.8, 1.5),
    }
    _b = _PRESETS.get(strength, _PRESETS["any"])
    min_body_mult   = float(cond.get("min_body_mult",   _b[0]))   # OB candle body ÷ avg body
    min_impulse_atr = float(cond.get("min_impulse_atr", _b[1]))   # displacement ÷ ATR
    min_volume_mult = float(cond.get("min_volume_mult", _b[2]))   # OB candle vol ÷ avg vol
    klines   = await _get_klines(symbol, tf, 120, http_client, cache)
    if not klines or len(klines) < 12: return False, "OB: insufficient data"
    closes = _closes(klines); opens = _opens(klines)
    highs  = _highs(klines);  lows  = _lows(klines); vols = _vols(klines)
    n = len(klines)
    # Reference averages over the last ~20 closed bars (exclude the forming bar)
    _ref     = slice(max(0, n - 21), n - 1)
    _bodies  = [abs(closes[k] - opens[k]) for k in range(n)]
    _ranges  = [highs[k] - lows[k] for k in range(n)]
    avg_body = (sum(_bodies[_ref]) / max(1, len(_bodies[_ref]))) if _bodies[_ref] else 0.0
    atr      = (sum(_ranges[_ref]) / max(1, len(_ranges[_ref]))) if _ranges[_ref] else 0.0
    _vref    = [v for v in vols[_ref] if v > 0]
    avg_vol  = (sum(_vref) / len(_vref)) if _vref else 0.0

    def _quality_ok(i: int, impulse_move: float) -> bool:
        if min_body_mult > 0 and avg_body > 0 and _bodies[i] < min_body_mult * avg_body:
            return False
        if min_impulse_atr > 0 and atr > 0 and impulse_move < min_impulse_atr * atr:
            return False
        # volume gate only when the feed actually provides volume (forex may not)
        if min_volume_mult > 0 and avg_vol > 0 and vols[i] < min_volume_mult * avg_vol:
            return False
        return True

    _tag = strength if strength != "any" else "OB"
    # Find order blocks: last significant candle before a strong opposite move
    # Requires 2 consecutive candles (not 3) so OBs are found in real markets
    if ob_type == "bullish":
        # Last bearish candle before 2+ bullish candles (impulse move up)
        for i in range(n-3, 5, -1):
            bear_c = opens[i] > closes[i]
            next_range = range(i+1, min(i+3, n))
            next_bull = len(next_range) >= 2 and all(closes[j] > opens[j] for j in next_range)
            # Also accept 1 strong bullish candle (body ≥ 1% move)
            if not next_bull and i+1 < n:
                body_pct = abs(closes[i+1] - opens[i+1]) / (opens[i+1] or 1) * 100
                next_bull = closes[i+1] > opens[i+1] and body_pct >= 1.0
            if not (bear_c and next_bull):
                continue
            peak = max(highs[i+1:min(i+4, n)] or [closes[i]])
            if not _quality_ok(i, peak - closes[i]):
                continue
            ob_high = highs[i]; ob_low = lows[i]
            # unmitigated = price has NOT already returned into the zone since it formed
            if unmit and any(lows[j] <= ob_high * (1 + tol) for j in range(i+1, n-1)):
                continue
            in_ob = ob_low * (1 - tol) <= current_price <= ob_high * (1 + tol)
            return in_ob, f"Bullish {_tag} {ob_low:.6g}–{ob_high:.6g} {'HIT' if in_ob else 'miss'}"
        return False, f"No qualifying bullish OB ({strength})"
    else:
        # Last bullish candle before 2+ bearish candles (impulse move down)
        for i in range(n-3, 5, -1):
            bull_c = closes[i] > opens[i]
            next_range = range(i+1, min(i+3, n))
            next_bear = len(next_range) >= 2 and all(closes[j] < opens[j] for j in next_range)
            # Also accept 1 strong bearish candle (body ≥ 1% move)
            if not next_bear and i+1 < n:
                body_pct = abs(closes[i+1] - opens[i+1]) / (opens[i+1] or 1) * 100
                next_bear = closes[i+1] < opens[i+1] and body_pct >= 1.0
            if not (bull_c and next_bear):
                continue
            trough = min(lows[i+1:min(i+4, n)] or [closes[i]])
            if not _quality_ok(i, closes[i] - trough):
                continue
            ob_high = highs[i]; ob_low = lows[i]
            if unmit and any(highs[j] >= ob_low * (1 - tol) for j in range(i+1, n-1)):
                continue
            in_ob = ob_low * (1 - tol) <= current_price <= ob_high * (1 + tol)
            return in_ob, f"Bearish {_tag} {ob_low:.6g}–{ob_high:.6g} {'HIT' if in_ob else 'miss'}"
        return False, f"No qualifying bearish OB ({strength})"


# ─── 10. FIBONACCI ────────────────────────────────────────────────────────────

async def eval_fibonacci(
    cond: Dict, symbol: str, current_price: float, http_client, cache: Dict
) -> Tuple[bool, str]:
    level    = float(cond.get("level", 0.618))
    sub      = cond.get("condition", "at_retracement")
    tf       = cond.get("timeframe", "4h")
    tol      = float(cond.get("tolerance_pct", 1.0)) / 100
    lookback = int(cond.get("lookback", 50))
    klines   = await _get_klines(symbol, tf, lookback, http_client, cache)
    if not klines or len(klines) < 10: return False, "Fib: insufficient data"
    highs = _highs(klines[:-1]); lows = _lows(klines[:-1])
    swing_h = max(highs); swing_l = min(lows)
    if swing_h == swing_l: return False, "Fib: flat range"
    if sub == "at_retracement":
        fib_price = swing_h - (swing_h - swing_l) * level
        near = abs(current_price - fib_price) / fib_price <= tol
        return near, f"Fib {level*100:.1f}% retrace={fib_price:.6g} {'HIT' if near else 'miss'}"
    if sub == "at_extension":
        fib_price = swing_h + (swing_h - swing_l) * level
        near = abs(current_price - fib_price) / fib_price <= tol
        return near, f"Fib ext {level*100:.1f}%={fib_price:.6g} {'HIT' if near else 'miss'}"
    return False, f"Unknown Fib sub: {sub}"


# ─── 11. DIVERGENCE ───────────────────────────────────────────────────────────

async def eval_divergence(
    cond: Dict, symbol: str, http_client, cache: Dict
) -> Tuple[bool, str]:
    indicator = cond.get("indicator", "rsi").lower()
    direction = cond.get("direction", "bullish")
    tf        = cond.get("timeframe", "15m")
    lookback  = int(cond.get("lookback", 20))
    klines    = await _get_klines(symbol, tf, lookback + 50, http_client, cache)
    if not klines or len(klines) < lookback: return False, "Divergence: insufficient data"
    closes = _closes(klines)
    if indicator == "rsi":
        ind_vals = _rsi_values(closes, 14)
    elif indicator == "macd":
        fast = _ema_list(closes, 12); slow = _ema_list(closes, 26)
        ind_vals = [f - s for f, s in zip(fast, slow)]
    else:
        return False, f"Divergence: unknown indicator {indicator}"
    if len(ind_vals) < lookback // 2: return False, "Divergence: not enough indicator values"
    price_slice = closes[-len(ind_vals):]
    p_start, p_end = price_slice[0], price_slice[-1]
    i_start, i_end = ind_vals[0], ind_vals[-1]
    if direction == "bullish":
        # Price makes lower low, indicator makes higher low → bullish divergence
        r = p_end < p_start and i_end > i_start
    else:
        # Price makes higher high, indicator makes lower high → bearish divergence
        r = p_end > p_start and i_end < i_start
    return r, f"{direction.title()} {indicator.upper()} divergence: {'✓' if r else '✗'}"


# ─── 12. FUNDING RATE ─────────────────────────────────────────────────────────

def _mexc_contract_symbol(symbol: str) -> str:
    """Convert BTCUSDT → BTC_USDT for MEXC contract API."""
    if symbol.upper().endswith("USDT"):
        return symbol[:-4].upper() + "_USDT"
    return symbol.upper()


async def eval_funding_rate(
    cond: Dict, symbol: str, http_client
) -> Tuple[bool, str]:
    op  = cond.get("operator", "lt")
    val = float(cond.get("value", -0.05))
    mexc_sym = _mexc_contract_symbol(symbol)
    try:
        resp = await http_client.get(
            f"https://contract.mexc.com/api/v1/contract/funding_rate/{mexc_sym}",
            timeout=6
        )
        data = resp.json()
        fr_raw = (data.get("data") or {}).get("fundingRate")
        if fr_raw is None:
            return False, "Funding rate: no data from MEXC"
        fr = float(fr_raw) * 100  # to %
        r  = _cmp(fr, op, val)
        return r, f"Funding={fr:+.4f}%"
    except Exception as e:
        return False, f"Funding error: {e}"


# ─── 13. OPEN INTEREST ────────────────────────────────────────────────────────

# Module-level OI snapshot cache: symbol → (timestamp, holdVol)
_OI_PREV: Dict[str, Tuple[float, float]] = {}

async def eval_open_interest(
    cond: Dict, symbol: str, http_client
) -> Tuple[bool, str]:
    change_pct  = float(cond.get("change_pct", 5.0))
    op          = cond.get("operator", "gt")
    sub         = cond.get("condition", "")
    mexc_sym    = _mexc_contract_symbol(symbol)
    import time as _time
    try:
        resp = await http_client.get(
            f"https://contract.mexc.com/api/v1/contract/open_interest/{mexc_sym}",
            timeout=6
        )
        data = resp.json()
        oi_now = float((data.get("data") or {}).get("openInterest", 0))
        if oi_now == 0:
            return False, "OI: no data from MEXC"
        now_ts = _time.time()
        prev = _OI_PREV.get(symbol)
        if prev is None or (now_ts - prev[0]) > 3600:
            # No previous value yet — store and return neutral
            _OI_PREV[symbol] = (now_ts, oi_now)
            return False, f"OI={oi_now:.0f} (initialising baseline)"
        _, oi_prev = prev
        pct = (oi_now - oi_prev) / oi_prev * 100 if oi_prev else 0
        # Refresh stored value every 5 min
        if now_ts - prev[0] > 300:
            _OI_PREV[symbol] = (now_ts, oi_now)
        if sub == "rising":   return pct > 2,  f"OI={pct:+.2f}%"
        if sub == "falling":  return pct < -2, f"OI={pct:+.2f}%"
        return _cmp(abs(pct), op, change_pct), f"OI change={pct:+.2f}%"
    except Exception as e:
        return False, f"OI error: {e}"


# ─── 14. SESSION FILTER ───────────────────────────────────────────────────────

def eval_session(cond: Dict) -> Tuple[bool, str]:
    sessions = [s.lower() for s in cond.get("sessions", ["london", "new_york"])]
    hour = datetime.now(timezone.utc).hour
    SESSION_HOURS = {
        "asian":    (0, 8),   "tokyo":    (0, 8),
        "london":   (7, 16),  "europe":   (7, 16),
        "new_york": (13, 22), "ny":       (13, 22),
        "overlap":  (13, 16),
    }
    active = []
    for name, (start, end) in SESSION_HOURS.items():
        if start <= hour < end:
            active.append(name)
    r = any(s in active for s in sessions)
    return r, f"Session: hour={hour}UTC active={active}"


# ─── 15. PRICE RELATIVE ───────────────────────────────────────────────────────

async def eval_price_relative(
    cond: Dict, symbol: str, current_price: float, http_client, cache: Dict
) -> Tuple[bool, str]:
    reference = cond.get("reference", "daily_open")
    op        = cond.get("operator", "gt")
    val       = float(cond.get("value", 0))
    try:
        if reference == "daily_open":
            klines = await _get_klines(symbol, "1d", 2, http_client, cache)
            if not klines: return False, "Price relative: no daily klines"
            ref_price = float(klines[-1][1])
        elif reference in ("session_high", "session_low"):
            klines = await _get_klines(symbol, "1h", 12, http_client, cache)
            if not klines: return False, "Price relative: no session klines"
            ref_price = max(_highs(klines)) if reference == "session_high" else min(_lows(klines))
        elif reference == "weekly_open":
            klines = await _get_klines(symbol, "1w", 2, http_client, cache)
            if not klines: return False, "Price relative: no weekly klines"
            ref_price = float(klines[-1][1])
        else:
            return False, f"Unknown reference: {reference}"
        pct = (current_price - ref_price) / ref_price * 100
        sub = cond.get("condition", "")
        if sub == "above": return current_price > ref_price, f"Price {'above' if current_price > ref_price else 'below'} {reference} ({pct:+.2f}%)"
        if sub == "below": return current_price < ref_price, f"Price {'above' if current_price > ref_price else 'below'} {reference} ({pct:+.2f}%)"
        if sub == "near":
            threshold = float(cond.get("threshold_pct", 2.0))
            return abs(pct) <= threshold, f"Price {pct:+.2f}% from {reference} (threshold ±{threshold}%)"
        return _cmp(pct, op, val), f"Price vs {reference}={pct:+.2f}%"
    except Exception as e:
        return False, f"Price relative error: {e}"


# ─── 16. SENTIMENT (via CryptoNews API) ──────────────────────────────────────

async def eval_sentiment(
    cond: Dict, symbol: str, http_client
) -> Tuple[bool, str]:
    """
    Derives a 0–100 sentiment score from recent CryptoNews headlines.
    Positive articles score 100, negative score 0, neutral score 50.
    The final score is the average across the last ~10 matching articles.
    Requires CRYPTONEWS_API_KEY env var.
    """
    import os
    op    = cond.get("operator", "gt")
    val   = float(cond.get("value", 60))
    coin  = symbol.replace("USDT", "").upper()
    api_key = os.environ.get("CRYPTONEWS_API_KEY", "")
    if not api_key:
        return False, "Sentiment: CRYPTONEWS_API_KEY not set"
    try:
        resp = await http_client.get(
            "https://cryptonews-api.com/api/v1",
            params={
                "tickers": coin,
                "items":   10,
                "token":   api_key,
            },
            timeout=8,
        )
        data = resp.json()
        articles = data.get("data", [])
        if not articles:
            return False, f"Sentiment: no news found for {coin}"
        scores = []
        for a in articles:
            sent = (a.get("sentiment") or "").lower()
            if sent == "positive":
                scores.append(100)
            elif sent == "negative":
                scores.append(0)
            else:
                scores.append(50)
        score = sum(scores) / len(scores)
        r = _cmp(score, op, val)
        return r, f"News sentiment={score:.0f}/100 ({len(articles)} articles)"
    except Exception as e:
        return False, f"Sentiment error: {e}"


# ─── 17. LIQUIDATION PROXIMITY (math-based, no API needed) ────────────────────

async def eval_liquidation(
    cond: Dict, symbol: str, current_price: float, http_client, cache: Dict = None
) -> Tuple[bool, str]:
    """
    Estimates liquidation clusters mathematically from recent candle data.
    Finds recent swing highs (potential short liquidations) and swing lows
    (potential long liquidations) and checks if price is within tolerance.

    Logic: A 10× long opened at a swing low gets liquidated ~9% below entry.
    A 20× long gets liquidated ~4.8% below entry, etc. We check if the current
    price is within `tolerance_pct` of any of these computed liquidation levels.

    Leverages checked: 10×, 20×, 25×, 50×, 100×
    """
    direction   = cond.get("direction", "below")   # "below" = long liq, "above" = short liq
    tol         = float(cond.get("tolerance_pct", 2.0)) / 100
    tf          = cond.get("timeframe", "15m")
    LEVERAGES   = [10, 20, 25, 50, 100]
    # Maintenance margin ≈ 0.5% for high-leverage perps; liquidation at (1/lev - 0.005)
    def liq_dist(lev): return (1.0 / lev) - 0.005

    try:
        klines = await _get_klines(symbol, tf, 60, http_client, cache or {})
        if not klines or len(klines) < 10:
            return False, "Liquidation: insufficient candle data"

        highs  = _highs(klines[:-2])
        lows   = _lows(klines[:-2])

        if direction == "below":
            # Long liquidation levels: lev × longs opened near recent swing lows
            swing_lows = [lows[i] for i in _swing_lows(highs, lows)][-5:]
            liq_levels = [
                entry * (1 - liq_dist(lev))
                for entry in swing_lows
                for lev in LEVERAGES
            ]
        else:
            # Short liquidation levels: lev × shorts opened near recent swing highs
            swing_highs = [highs[i] for i in _swing_highs(highs, lows)][-5:]
            liq_levels = [
                entry * (1 + liq_dist(lev))
                for entry in swing_highs
                for lev in LEVERAGES
            ]

        nearby = [
            lvl for lvl in liq_levels
            if abs(current_price - lvl) / current_price <= tol
        ]
        r = len(nearby) > 0
        closest = min(nearby, key=lambda x: abs(x - current_price)) if nearby else None
        msg = (
            f"Liq cluster {'found' if r else 'none'} {direction} "
            + (f"@ {closest:.4g}" if closest else f"(tol ±{tol*100:.1f}%)")
        )
        return r, msg
    except Exception as e:
        return False, f"Liquidation error: {e}"


# ─── 18. TREND REVERSAL ───────────────────────────────────────────────────────

async def eval_trend_reversal(
    cond: Dict, symbol: str, current_price: float, http_client, cache: Dict
) -> Tuple[bool, str]:
    """
    Detects change-of-direction / trend reversal signals.
    Bullish reversal: price was in a downtrend, now showing reversal signals.
    Bearish reversal: price was in an uptrend, now showing reversal signals.
    Uses EMA21 trend direction + RSI + candle confirmation.
    """
    direction = cond.get("condition", cond.get("direction", "bullish"))
    tf        = cond.get("timeframe", "15m")

    klines = await _get_klines(symbol, tf, 50, http_client, cache)
    if not klines or len(klines) < 25:
        return False, "Reversal: insufficient data"

    closes = _closes(klines)
    opens  = _opens(klines)

    # RSI
    rsi_vals = _rsi_values(closes, 14)
    if len(rsi_vals) < 2:
        return False, "Reversal: RSI insufficient"
    rsi      = rsi_vals[-1]
    rsi_prev = rsi_vals[-2]

    # EMA21 — measures current trend
    ema21 = _ema_list(closes, 21)
    if len(ema21) < 6:
        return False, "Reversal: EMA insufficient"
    ema_now   = ema21[-1]
    ema_5ago  = ema21[-6]

    cur_close  = closes[-1]
    prev_close = closes[-2]
    cur_open   = opens[-1]

    if direction == "bullish":
        # Prior downtrend: EMA was declining
        prior_downtrend = ema_5ago > ema_now
        # Price near or below EMA (not already far above it)
        price_near_ema  = cur_close <= ema_now * 1.03
        # Current candle is green
        bullish_candle  = cur_close > cur_open
        # RSI was oversold and now turning up
        rsi_was_low     = rsi < 50
        rsi_rising      = rsi > rsi_prev
        # Price is rising from the last candle
        price_bouncing  = cur_close > prev_close

        passed = prior_downtrend and price_near_ema and bullish_candle and rsi_was_low and rsi_rising and price_bouncing
        return passed, (
            f"Bullish reversal: EMA{'↓' if prior_downtrend else '↑'} "
            f"RSI={rsi:.1f}{'↑' if rsi_rising else '↓'} "
            f"candle={'🟢' if bullish_candle else '🔴'} "
            f"{'HIT' if passed else 'miss'}"
        )
    else:
        # Prior uptrend: EMA was rising
        prior_uptrend  = ema_5ago < ema_now
        # Price near or above EMA (not already far below)
        price_near_ema = cur_close >= ema_now * 0.97
        # Current candle is red
        bearish_candle = cur_close < cur_open
        # RSI was overbought and now turning down
        rsi_was_high   = rsi > 50
        rsi_falling    = rsi < rsi_prev
        # Price is dropping from the last candle
        price_dropping = cur_close < prev_close

        passed = prior_uptrend and price_near_ema and bearish_candle and rsi_was_high and rsi_falling and price_dropping
        return passed, (
            f"Bearish reversal: EMA{'↑' if prior_uptrend else '↓'} "
            f"RSI={rsi:.1f}{'↓' if rsi_falling else '↑'} "
            f"candle={'🔴' if bearish_candle else '🟢'} "
            f"{'HIT' if passed else 'miss'}"
        )


# ─── 19. SUSTAINED TREND ──────────────────────────────────────────────────────

async def eval_sustained_trend(
    cond: Dict, symbol: str, http_client, cache: Dict
) -> Tuple[bool, str]:
    """
    Detects coins that have been consistently pumping or dumping over multiple
    candle periods (days, 4h, 1h, etc.).

    Useful for:
      - Continuation longs/shorts — coin already in a strong multi-day trend
      - Exhaustion plays          — coin has been pumping/dumping for many days → fade
      - Relief bounce setups      — coin dumped hard for days → look for a bounce

    Config keys:
      trend_dir      : "pump" | "dump"
      timeframe      : "1h" | "2h" | "4h" | "1d"  (default "1d")
      periods        : int  — number of closed periods to examine (default 3)
      min_total_pct  : float — min total % move across the window (default 10.0)
      min_consistent : float — min % of periods in trend direction (0–100, default 65)
      require_active : 0|1  — current candle must also be in trend direction (default 1)
    """
    trend_dir      = cond.get("trend_dir", "pump")
    tf_raw         = str(cond.get("timeframe", "1d"))
    periods        = max(2, int(cond.get("periods", 3)))
    min_total_pct  = float(cond.get("min_total_pct", 10.0))
    min_consistent = float(cond.get("min_consistent", 65.0)) / 100.0
    require_active = str(cond.get("require_active", 1)) not in ("0", "false", "False")

    # Normalise timeframe string
    tf_map = {"1h": "1h", "2h": "2h", "4h": "4h", "1d": "1d", "day": "1d", "daily": "1d"}
    interval = tf_map.get(tf_raw, "1d")

    # Fetch enough candles: periods for analysis + 2 buffer
    n = periods + 3
    klines = await _get_klines(symbol, interval, n, http_client, cache)
    if not klines or len(klines) < periods + 1:
        return False, f"SustainedTrend: not enough {interval} candles (got {len(klines) if klines else 0})"

    # Exclude the current (still-forming) candle — use the last `periods` closed candles
    closed = klines[-(periods + 1):-1]   # periods fully closed candles
    if len(closed) < periods:
        return False, "SustainedTrend: insufficient closed candles"

    # ── Total % move across the whole window ─────────────────────────────────
    window_open  = float(closed[0][1])   # open of oldest candle
    window_close = float(closed[-1][4])  # close of newest closed candle
    if window_open <= 0:
        return False, "SustainedTrend: invalid open price"
    total_pct = (window_close - window_open) / window_open * 100

    # ── Consistency: fraction of periods that went in the trend direction ─────
    in_direction = sum(
        1 for k in closed
        if (trend_dir == "pump" and float(k[4]) > float(k[1]))
        or (trend_dir == "dump" and float(k[4]) < float(k[1]))
    )
    consistent = in_direction / periods

    # ── Higher-highs / lower-lows structural check ────────────────────────────
    highs = [float(k[2]) for k in closed]
    lows  = [float(k[3]) for k in closed]
    if trend_dir == "pump":
        structure_ok = highs[-1] > highs[0]  # at least overall higher high
    else:
        structure_ok = lows[-1] < lows[0]    # at least overall lower low

    # ── Current candle still-active check ────────────────────────────────────
    cur = klines[-1]
    cur_o, cur_c = float(cur[1]), float(cur[4])
    if trend_dir == "pump":
        active = cur_c >= cur_o
    else:
        active = cur_c <= cur_o

    # ── Decision ─────────────────────────────────────────────────────────────
    pass_direction  = (total_pct > 0) if trend_dir == "pump" else (total_pct < 0)
    pass_total      = abs(total_pct) >= min_total_pct
    pass_consistent = consistent >= min_consistent
    pass_active     = (not require_active) or active

    passed = pass_direction and pass_total and pass_consistent and pass_active

    label = "📈 PUMP" if trend_dir == "pump" else "📉 DUMP"
    detail = (
        f"SustainedTrend({label}) {periods}×{interval}: "
        f"total={total_pct:+.1f}% (need {min_total_pct:.0f}%) "
        f"consistent={consistent*100:.0f}% (need {min_consistent*100:.0f}%) "
        f"struct={'✓' if structure_ok else '✗'} "
        f"active={'✓' if active else '✗'} "
        f"{'HIT' if passed else 'miss'}"
    )
    return passed, detail


# ─── Master evaluator ─────────────────────────────────────────────────────────

# ─── Forex evaluators (P1: sessions, ORB, prev day/week levels) ─────────────
# Pure-Python evaluators — no external HTTP calls beyond the same _get_klines
# the rest of the TA module uses. Asset_class is set in cache by the dispatch.

def eval_forex_session(cond: Dict) -> Tuple[bool, str]:
    """`forex_session` — gate firing on session windows.

    cfg.condition:
      'in_session'     — currently inside the session window (WHOLE session)
      'session_open'   — within first N minutes of session open (default 30)
      'session_close'  — within last N minutes of session close (default 30)
      'overlap'        — currently in the London/NY overlap
    cfg.session:  single session 'london' | 'ny' | 'asian' | 'sydney' | 'overlap'
    cfg.sessions: OPTIONAL list of sessions — fires when in ANY of them (OR).
                  Use this for "trade during London AND New York session" so the
                  gate stays open across BOTH whole windows, not just one.
    cfg.within_minutes (optional, default 30): window for session_open/close.
    """
    from app.services.forex_engine import (
        in_session, session_just_opened, session_about_to_close, SESSIONS,
    )

    def _norm_sid(s: str) -> str:
        s = (s or "").strip().lower()
        # AI/wizard sometimes emit "new_york"/"ny "/"tokyo" — map to canonical ids.
        return {"new_york": "ny", "newyork": "ny", "tokyo": "asian",
                "us": "ny", "uk": "london"}.get(s, s)

    sub = (cond.get("condition") or "in_session").lower()
    within = int(cond.get("within_minutes") or 30)

    # Collect one-or-many sessions (plural `sessions` wins when present).
    raw = cond.get("sessions")
    if raw:
        sess_list = [_norm_sid(s) for s in raw if str(s).strip()]
    else:
        sess_list = [_norm_sid(cond.get("session") or "london")]
    sess_list = [s for s in sess_list if s in SESSIONS] or ["london"]

    labels = " / ".join(SESSIONS[s].label for s in sess_list)

    if sub == "overlap":
        ok = in_session("overlap")
        return ok, f"London/NY Overlap session {'active' if ok else 'inactive'}"
    if sub == "session_open":
        ok = any(session_just_opened(s, within_minutes=within) for s in sess_list)
        return ok, f"{labels} open (first {within} min) {'✓' if ok else '✗'}"
    if sub == "session_close":
        ok = any(session_about_to_close(s, within_minutes=within) for s in sess_list)
        return ok, f"{labels} close (last {within} min) {'✓' if ok else '✗'}"
    # default: in_session — true for the FULL duration of any listed session
    ok = any(in_session(s) for s in sess_list)
    return ok, f"{labels} session {'active' if ok else 'inactive'}"


async def eval_forex_session_break(
    cond: Dict, symbol: str, price: float, http_client, cache: Dict,
) -> Tuple[bool, str]:
    """`forex_session_break` — price breaks the H/L of a named session.

    Foundation of "London Breakout" and "Asian Range Breakout" templates.

    cfg.condition:
      'high_break' — price > session_high
      'low_break'  — price < session_low
      'orb_high'   — price > opening-range high (first N min of session)
      'orb_low'    — price < opening-range low
    cfg.session: which session's range to use
    cfg.range_minutes (default 60 for ORB conditions)
    cfg.timeframe (default '15m')
    """
    from app.services.forex_engine import compute_session_range
    sub = (cond.get("condition") or "high_break").lower()
    session = (cond.get("session") or "asian").lower()
    tf = cond.get("timeframe") or "15m"
    range_min = int(cond.get("range_minutes") or 60)

    klines = await _get_klines(symbol, tf, 200, http_client, cache)
    if not klines:
        return False, f"{symbol}: no klines for session-break eval"

    is_orb = sub in ("orb_high", "orb_low")
    rng = compute_session_range(
        klines, session,
        first_n_minutes=range_min if is_orb else None,
    )
    if rng is None:
        return False, f"{symbol}: no candles in {session} session window today"

    if sub in ("high_break", "orb_high"):
        ok = price > rng.high
        return ok, (
            f"{symbol}: price {price:.5f} {'>' if ok else '≤'} "
            f"{session} {'ORB' if is_orb else ''} high {rng.high:.5f}"
        )
    # low_break / orb_low
    ok = price < rng.low
    return ok, (
        f"{symbol}: price {price:.5f} {'<' if ok else '≥'} "
        f"{session} {'ORB' if is_orb else ''} low {rng.low:.5f}"
    )


async def eval_forex_prev_level(
    cond: Dict, symbol: str, price: float, http_client, cache: Dict,
) -> Tuple[bool, str]:
    """`forex_prev_level` — previous-day or previous-week H/L breaks/sweeps.

    cfg.condition:
      'above_pdh' | 'below_pdh' | 'above_pdl' | 'below_pdl'
      'above_wh'  | 'below_wh'  | 'above_wl'  | 'below_wl'
      'sweep_pdh' — price wicked above PDH then closed back below (liquidity grab)
      'sweep_pdl' — opposite
    """
    from app.services.forex_engine import (
        previous_day_high_low, previous_week_high_low,
    )
    sub = (cond.get("condition") or "above_pdh").lower()
    # Need ≥2 calendar days of 15m candles for PDH/PDL, more for weekly.
    tf = "15m" if "pd" in sub else "1h"
    need = 220 if "pd" in sub else 220
    klines = await _get_klines(symbol, tf, need, http_client, cache)
    if not klines:
        return False, f"{symbol}: no klines for prev-level eval"

    if "w" in sub.split("_")[-1]:
        levels = previous_week_high_low(klines)
        period_label = "PWeek"
    else:
        levels = previous_day_high_low(klines)
        period_label = "PDay"
    if levels is None:
        return False, f"{symbol}: not enough history for {period_label} levels"
    high, low = levels

    if sub.endswith("h"):  # ...pdh / ...wh
        if sub.startswith("above"):
            return price > high, f"{symbol}: {price:.5f} {'>' if price > high else '≤'} {period_label} high {high:.5f}"
        if sub.startswith("below"):
            return price < high, f"{symbol}: {price:.5f} {'<' if price < high else '≥'} {period_label} high {high:.5f}"
        if sub.startswith("sweep"):
            # Last candle wicked above `high` but closed back below it.
            last = klines[-1]
            try:
                last_high = float(last[2]); last_close = float(last[4])
                ok = last_high > high and last_close < high
                return ok, f"{symbol}: {'swept' if ok else 'did not sweep'} {period_label} high {high:.5f}"
            except (IndexError, ValueError):
                return False, f"{symbol}: malformed last kline"
    # ...pdl / ...wl
    if sub.startswith("above"):
        return price > low, f"{symbol}: {price:.5f} {'>' if price > low else '≤'} {period_label} low {low:.5f}"
    if sub.startswith("below"):
        return price < low, f"{symbol}: {price:.5f} {'<' if price < low else '≥'} {period_label} low {low:.5f}"
    if sub.startswith("sweep"):
        last = klines[-1]
        try:
            last_low = float(last[3]); last_close = float(last[4])
            ok = last_low < low and last_close > low
            return ok, f"{symbol}: {'swept' if ok else 'did not sweep'} {period_label} low {low:.5f}"
        except (IndexError, ValueError):
            return False, f"{symbol}: malformed last kline"
    return False, f"{symbol}: unknown prev-level condition {sub}"


async def eval_forex_news_avoidance(
    cond: Dict, symbol: str,
) -> Tuple[bool, str]:
    """`forex_news_avoidance` — TRUE when SAFE to trade (i.e. no high-impact
    event affects either currency within the configured window).

    cfg.minutes_before (default 30) — blackout starts N min before event
    cfg.minutes_after  (default 30) — blackout ends N min after event
    cfg.min_impact     (default 'high') — 'low' | 'medium' | 'high'

    Falls open (returns True) when FMP_API_KEY is missing or the upstream
    is down — strategy keeps trading; it just loses the news filter.
    """
    from app.services.fmp_calendar import is_news_blackout
    mb = int(cond.get("minutes_before") or 30)
    ma = int(cond.get("minutes_after") or 30)
    imp = (cond.get("min_impact") or "high").lower()
    in_blackout, ev = await is_news_blackout(
        symbol, minutes_before=mb, minutes_after=ma, min_impact=imp,
    )
    if in_blackout and ev:
        return False, (
            f"{symbol}: BLOCKED by {ev['impact']} {ev['currency']} "
            f"news '{ev['event']}' ({ev['minutes_from_now']:+.0f} min)"
        )
    return True, f"{symbol}: clear of {imp}-impact news (±{mb}/{ma} min window)"


async def eval_pivot_points(
    cond: Dict, symbol: str, current_price: float, http_client, cache: Dict
) -> Tuple[bool, str]:
    """Daily Pivot Points (PP / R1-R3 / S1-S3) from previous day's OHLC.
    Works on any asset class — crypto, forex/gold, index, stock.

    Config: level='pp'|'r1'|'r2'|'r3'|'s1'|'s2'|'s3'
            condition='above'|'below'|'near'
            tolerance_pct=0.3
    """
    level     = cond.get("level", "r1").lower()
    condition = cond.get("condition", "above")
    tol       = float(cond.get("tolerance_pct", 0.3))
    try:
        klines = await _get_klines(symbol, "1d", 3, http_client, cache)
        if len(klines) < 2:
            return False, "Pivot points: not enough daily klines"
        prev = klines[-2]
        h, l, c = float(prev[2]), float(prev[3]), float(prev[4])
        pp = (h + l + c) / 3
        r1 = 2 * pp - l;  r2 = pp + (h - l);  r3 = h + 2 * (pp - l)
        s1 = 2 * pp - h;  s2 = pp - (h - l);  s3 = l - 2 * (h - pp)
        levels = {"pp": pp, "r1": r1, "r2": r2, "r3": r3,
                  "s1": s1, "s2": s2, "s3": s3}
        ref = levels.get(level)
        if ref is None or ref <= 0:
            return False, f"Pivot: unknown level {level!r}"
        pct = (current_price - ref) / ref * 100
        label = f"{level.upper()}={ref:.5g} price={current_price:.5g} ({pct:+.2f}%)"
        if condition == "above":
            return current_price > ref, f"Pivot {label}"
        if condition == "below":
            return current_price < ref, f"Pivot {label}"
        if condition == "near":
            return abs(pct) <= tol, f"Pivot near {label} (tol ±{tol}%)"
        return False, f"Pivot: unknown condition {condition!r}"
    except Exception as e:
        return False, f"Pivot points error: {e}"


async def eval_session_level(
    cond: Dict, symbol: str, current_price: float, http_client, cache: Dict
) -> Tuple[bool, str]:
    """Session Level — price relative to current-session high, low, or open.
    Wraps eval_price_relative and adds 'session_open' reference.

    Config: reference='session_high'|'session_low'|'session_open'|'daily_open'
            condition='above'|'below'|'near'
            threshold_pct=1.5
    """
    reference = cond.get("reference", "session_low")
    condition = cond.get("condition", "near")
    tol       = float(cond.get("threshold_pct", 1.5))
    try:
        if reference == "session_open":
            klines = await _get_klines(symbol, "1h", 12, http_client, cache)
            if not klines:
                return False, "Session level: no 1h klines"
            ref_price = float(klines[0][1])  # open of first candle in window
            pct = (current_price - ref_price) / ref_price * 100
            lbl = f"session_open={ref_price:.5g} ({pct:+.2f}%)"
            if condition == "above":
                return current_price > ref_price, f"Price above {lbl}"
            if condition == "below":
                return current_price < ref_price, f"Price below {lbl}"
            return abs(pct) <= tol, f"Price near {lbl} (tol ±{tol}%)"
        # Delegate standard references (session_high/low, daily_open) to existing fn
        return await eval_price_relative(
            {**cond, "threshold_pct": tol}, symbol, current_price, http_client, cache)
    except Exception as e:
        return False, f"Session level error: {e}"


async def eval_stock_earnings_avoidance(
    cond: Dict, symbol: str,
) -> Tuple[bool, str]:
    """`stock_earnings_avoidance` — TRUE when SAFE to trade (no upcoming
    or just-released earnings for the ticker within the blackout window).

    cfg.days_before (default 2) — blackout starts N days before report
    cfg.days_after  (default 1) — blackout extends N days after report
    cfg.mode        (default 'both') — 'before' | 'after' | 'both' to
                     restrict the blackout to just the pre- or post-
                     earnings side (some swing strategies want to BUY
                     post-earnings momentum so they only block pre-).

    Falls open (returns True) when FMP_API_KEY is missing or FMP is down
    — strategy keeps trading; it just loses the earnings filter.
    """
    from app.services.earnings_calendar import is_earnings_blackout
    try:
        db_ = int(cond.get("days_before") or 2)
    except (TypeError, ValueError):
        db_ = 2
    try:
        da_ = int(cond.get("days_after") or 1)
    except (TypeError, ValueError):
        da_ = 1
    db_ = max(0, min(14, db_))
    da_ = max(0, min(14, da_))
    mode = (cond.get("mode") or "both").lower()
    if mode == "before":
        da_eff = 0; db_eff = db_
    elif mode == "after":
        da_eff = da_; db_eff = 0
    else:
        da_eff = da_; db_eff = db_

    blocked, ev, ok = await is_earnings_blackout(
        symbol, days_before=db_eff, days_after=da_eff,
    )
    if not ok:
        return True, f"{symbol}: yfinance upstream unavailable — earnings filter passing open"
    if blocked and ev:
        when = ev.get("_dt_utc")
        delta_d = ((when - __import__("datetime").datetime.now(when.tzinfo)).total_seconds() / 86400.0) if when else 0
        side = "upcoming" if delta_d >= 0 else "just released"
        t = (ev.get("time") or "").upper() or "TBA"
        return False, (
            f"{symbol}: BLOCKED — {side} earnings on "
            f"{(ev.get('date') or '?')[:10]} ({t}, {delta_d:+.1f}d)"
        )
    return True, f"{symbol}: clear of earnings (±{db_eff}d before / {da_eff}d after, mode={mode})"


async def eval_forex_currency_strength(
    cond: Dict, symbol: str,
) -> Tuple[bool, str]:
    """`forex_currency_strength` — fires when the base/quote strength
    differential breaches the configured threshold in the wanted direction.

    cfg.window      — '1h' | '4h' | '1d' (default '4h')
    cfg.min_diff    — minimum |base − quote| score required (default 0.5)
    cfg.direction   — 'base_strong' (favours LONG), 'quote_strong' (favours
                      SHORT), 'either' (any side with |diff|≥min)

    Falls open if the pair isn't a major cross or yfinance is down.
    """
    from app.services.currency_strength import pair_strength_diff
    win = (cond.get("window") or "4h").lower()
    try:
        min_diff = float(cond.get("min_diff") or 0.5)
    except (TypeError, ValueError):
        min_diff = 0.5
    direction = (cond.get("direction") or "either").lower()

    res = await pair_strength_diff(symbol, window=win)
    if res is None:
        return True, f"{symbol}: currency-strength filter inactive (not a major cross)"
    diff, base, quote, _, ok = res
    if not ok:
        # Upstream (yfinance) is fully unreachable — fail OPEN so trading
        # continues; the filter just goes dormant for this cycle.
        return True, f"{symbol}: currency-strength upstream unavailable — filter passing open"
    abs_diff = abs(diff)
    if direction == "base_strong":
        passed = diff >= min_diff
    elif direction == "quote_strong":
        passed = diff <= -min_diff
    else:  # either
        passed = abs_diff >= min_diff
    side = "base" if diff >= 0 else "quote"
    return passed, (
        f"{symbol}: {base} {diff:+.2f} vs {quote} (Δ {diff:+.2f}, "
        f"threshold {min_diff:.2f}, {side} stronger) on {win}"
    )


async def eval_forex_cot(
    cond: Dict, symbol: str,
) -> Tuple[bool, str]:
    """`forex_cot` — weekly CFTC Commitment of Traders sentiment filter.

    Uses the Traders-in-Financial-Futures (TFF) report for the non-USD leg
    of a USD pair. Cross pairs (no USD side) → filter inactive, passes open.

    cfg.condition:
      'specs_extreme_long'   — leveraged-fund net long in top X percentile
      'specs_extreme_short'  — leveraged-fund net short in top X percentile
      'specs_flipped_long'   — leveraged funds flipped to net long this wk
      'specs_flipped_short'  — leveraged funds flipped to net short
      'comm_extreme_long'    — dealer/commercial net long extreme
      'comm_extreme_short'   — dealer/commercial net short extreme
    cfg.extreme_pct       — percentile threshold (default 70, range 50–95)
    cfg.lookback_weeks    — history window for the percentile rank (52)
    cfg.respect_pair_inversion — when True (default), USDJPY-style pairs
       flip extreme_long↔extreme_short meaning so the filter still maps
       cleanly to "fire LONG on this pair".

    Always fails OPEN if CFTC Socrata is unreachable or the pair is
    unsupported, so a CFTC outage cannot block live trading.
    """
    from app.services.cot_data import cot_sentiment
    condition = (cond.get("condition") or "specs_extreme_long").lower()
    try:
        extreme_pct = float(cond.get("extreme_pct") or 70)
    except (TypeError, ValueError):
        extreme_pct = 70.0
    extreme_pct = max(50.0, min(95.0, extreme_pct))
    try:
        lookback = int(cond.get("lookback_weeks") or 52)
    except (TypeError, ValueError):
        lookback = 52
    lookback = max(8, min(156, lookback))
    # Tolerant boolean parse — web wizard chips serialize as strings
    # ('true'/'false'), JSON booleans, ints (0/1) all need to behave.
    _ri = cond.get("respect_pair_inversion", True)
    if isinstance(_ri, str):
        respect_invert = _ri.strip().lower() not in ("false", "0", "no", "off", "")
    else:
        respect_invert = bool(_ri)

    res = await cot_sentiment(symbol, lookback_weeks=lookback)
    if res is None:
        return True, f"{symbol}: COT filter inactive (no USD leg)"
    data, ok = res
    if not ok:
        return True, f"{symbol}: CFTC upstream unavailable — COT filter passing open"
    if data.get("weeks_observed", 0) < 2:
        return True, f"{symbol}: COT history too thin ({data.get('weeks_observed', 0)} wks) — passing open"

    ccy = data["ccy"]
    invert = bool(data.get("invert_for_pair")) and respect_invert
    sp = float(data.get("specs_pct", 50))
    cp = float(data.get("comm_pct", 50))
    spf = int(data.get("specs_flipped", 0))
    cmf = int(data.get("comm_flipped", 0))

    # When the non-USD leg is the QUOTE (e.g. USDJPY → JPY), a bullish
    # reading on JPY = bearish for the pair, so we flip the meaning.
    eff = condition
    if invert:
        flip_map = {
            "specs_extreme_long":  "specs_extreme_short",
            "specs_extreme_short": "specs_extreme_long",
            "specs_flipped_long":  "specs_flipped_short",
            "specs_flipped_short": "specs_flipped_long",
            "comm_extreme_long":   "comm_extreme_short",
            "comm_extreme_short":  "comm_extreme_long",
        }
        eff = flip_map.get(condition, condition)

    if eff == "specs_extreme_long":
        passed = sp >= extreme_pct
        why = f"leveraged-fund net-long at {sp:.0f}th pct (≥{extreme_pct:.0f})"
    elif eff == "specs_extreme_short":
        passed = sp <= (100.0 - extreme_pct)
        why = f"leveraged-fund net-short at {100-sp:.0f}th pct (≥{extreme_pct:.0f})"
    elif eff == "specs_flipped_long":
        passed = spf > 0
        why = f"leveraged funds flipped net-long this week" if passed else "no spec flip-long this wk"
    elif eff == "specs_flipped_short":
        passed = spf < 0
        why = f"leveraged funds flipped net-short this week" if passed else "no spec flip-short this wk"
    elif eff == "comm_extreme_long":
        passed = cp >= extreme_pct
        why = f"dealer/commercial net-long at {cp:.0f}th pct (≥{extreme_pct:.0f})"
    elif eff == "comm_extreme_short":
        passed = cp <= (100.0 - extreme_pct)
        why = f"dealer/commercial net-short at {100-cp:.0f}th pct (≥{extreme_pct:.0f})"
    else:
        return True, f"{symbol}: unknown COT condition '{condition}' — passing open"

    inv_note = " (inverted for USD-base pair)" if invert else ""
    return passed, f"{symbol}: COT/{ccy} {why}{inv_note}"


async def eval_forex_liquidity_pa(
    cond: Dict, symbol: str, http_client, cache: Dict,
) -> Tuple[bool, str]:
    """`forex_liquidity_pa` — intraday liquidity sweeps + classic
    price-action candle patterns on the strategy's working timeframe.

    cfg.pattern:
      'sweep_eqh'    — last candle wicked through equal-highs cluster then
                       closed back below (long-bias liquidity grab)
      'sweep_eql'    — opposite (short-bias liquidity grab)
      'pin_bar_bull' — small body near top + lower wick ≥ 2× body
      'pin_bar_bear' — small body near bottom + upper wick ≥ 2× body
      'engulf_bull'  — current green candle body engulfs prior red body
      'engulf_bear'  — current red candle body engulfs prior green body
      'inside_bar'   — current high<prev_high and current low>prev_low
    cfg.timeframe   — kline timeframe (default '15m')
    cfg.lookback    — bars scanned for equal H/L cluster (default 20)
    cfg.tolerance_pips — max distance between highs/lows to count as
                         "equal" (default 3 pips; converted to price via
                         0.01 for JPY pairs, 0.0001 otherwise)

    Returns False (don't fire) when not enough klines; never blocks on
    upstream error since _get_klines already returns None gracefully.
    """
    pattern = (cond.get("pattern") or "sweep_eqh").lower()
    tf = (cond.get("timeframe") or "15m").lower()
    try:
        lookback = max(5, min(int(cond.get("lookback") or 20), 100))
    except (TypeError, ValueError):
        lookback = 20
    try:
        tol_pips = max(0.5, float(cond.get("tolerance_pips") or 3))
    except (TypeError, ValueError):
        tol_pips = 3.0
    pip = 0.01 if "JPY" in symbol.upper() else 0.0001
    tol = tol_pips * pip

    need = max(lookback + 5, 30)
    klines = await _get_klines(symbol, tf, need, http_client, cache)
    if not klines or len(klines) < 3:
        return False, f"{symbol}: no klines for liquidity/PA eval"

    try:
        last_o = float(klines[-1][1]); last_h = float(klines[-1][2])
        last_l = float(klines[-1][3]); last_c = float(klines[-1][4])
        prev_o = float(klines[-2][1]); prev_h = float(klines[-2][2])
        prev_l = float(klines[-2][3]); prev_c = float(klines[-2][4])
    except (IndexError, ValueError, TypeError):
        return False, f"{symbol}: malformed kline data"

    body = abs(last_c - last_o)
    rng  = max(last_h - last_l, 1e-12)
    upper_wick = last_h - max(last_c, last_o)
    lower_wick = min(last_c, last_o) - last_l

    if pattern == "pin_bar_bull":
        ok = (lower_wick >= 2 * body) and (body <= 0.4 * rng) and (last_c > last_o or upper_wick <= body)
        return ok, (
            f"{symbol}: bullish pin {'detected' if ok else 'not detected'} "
            f"(body {body:.5f}, lower_wick {lower_wick:.5f}, range {rng:.5f})"
        )
    if pattern == "pin_bar_bear":
        ok = (upper_wick >= 2 * body) and (body <= 0.4 * rng) and (last_c < last_o or lower_wick <= body)
        return ok, (
            f"{symbol}: bearish pin {'detected' if ok else 'not detected'} "
            f"(body {body:.5f}, upper_wick {upper_wick:.5f}, range {rng:.5f})"
        )
    if pattern == "engulf_bull":
        prev_body_top = max(prev_o, prev_c); prev_body_bot = min(prev_o, prev_c)
        ok = (prev_c < prev_o) and (last_c > last_o) and (last_c >= prev_body_top) and (last_o <= prev_body_bot)
        return ok, f"{symbol}: bullish engulfing {'fired' if ok else 'no'} (prev red, last green & engulfs body)"
    if pattern == "engulf_bear":
        prev_body_top = max(prev_o, prev_c); prev_body_bot = min(prev_o, prev_c)
        ok = (prev_c > prev_o) and (last_c < last_o) and (last_o >= prev_body_top) and (last_c <= prev_body_bot)
        return ok, f"{symbol}: bearish engulfing {'fired' if ok else 'no'} (prev green, last red & engulfs body)"
    if pattern == "inside_bar":
        ok = (last_h < prev_h) and (last_l > prev_l)
        return ok, f"{symbol}: inside bar {'fired' if ok else 'no'} (last H/L inside prev H/L)"

    # Liquidity-sweep modes: scan the lookback window for equal H/L clusters,
    # then check the latest candle for the wick-through-and-close-back pattern.
    window = klines[-(lookback + 1):-1]  # exclude the current candle
    if not window:
        return False, f"{symbol}: insufficient window for sweep eval"
    try:
        highs = [float(k[2]) for k in window]
        lows  = [float(k[3]) for k in window]
    except (IndexError, ValueError, TypeError):
        return False, f"{symbol}: malformed window kline data"

    if pattern == "sweep_eqh":
        ref = max(highs)
        cluster = [h for h in highs if abs(h - ref) <= tol]
        if len(cluster) < 2:
            return False, f"{symbol}: no equal-highs cluster within ±{tol_pips:.1f} pips"
        ok = last_h > ref and last_c < ref
        return ok, (
            f"{symbol}: {'SWEPT' if ok else 'no sweep of'} equal-highs cluster @ {ref:.5f} "
            f"({len(cluster)} touches, ±{tol_pips:.1f} pips tol)"
        )
    if pattern == "sweep_eql":
        ref = min(lows)
        cluster = [l for l in lows if abs(l - ref) <= tol]
        if len(cluster) < 2:
            return False, f"{symbol}: no equal-lows cluster within ±{tol_pips:.1f} pips"
        ok = last_l < ref and last_c > ref
        return ok, (
            f"{symbol}: {'SWEPT' if ok else 'no sweep of'} equal-lows cluster @ {ref:.5f} "
            f"({len(cluster)} touches, ±{tol_pips:.1f} pips tol)"
        )

    return False, f"{symbol}: unknown liquidity_pa pattern '{pattern}'"


# ─── ICT / DAY-TRADING SIGNALS ────────────────────────────────────────────────

def eval_fx_killzone(cond: Dict) -> Tuple[bool, str]:
    """`fx_killzone` — ICT high-probability time windows.

    cfg.killzone:
      'london_kz'  — 07:00–09:00 UTC  (London open)
      'ny_kz'      — 12:00–14:00 UTC  (NY open)
      'asian_kz'   — 20:00–23:00 UTC  (Tokyo/Sydney open)
      'any_kz'     — any of the above
    """
    kz = (cond.get("killzone") or "london_kz").lower()
    now = datetime.utcnow()
    cur = now.hour * 60 + now.minute

    WINDOWS = {
        "london_kz": [(7 * 60, 9 * 60)],
        "ny_kz":     [(12 * 60, 14 * 60)],
        "asian_kz":  [(20 * 60, 23 * 60)],
    }
    if kz == "any_kz":
        windows = [w for ws in WINDOWS.values() for w in ws]
    else:
        windows = WINDOWS.get(kz, WINDOWS["london_kz"])

    inside = any(s <= cur < e for s, e in windows)
    label_map = {
        "london_kz": "London KZ 07-09 UTC",
        "ny_kz":     "NY KZ 12-14 UTC",
        "asian_kz":  "Asian KZ 20-23 UTC",
        "any_kz":    "any KZ",
    }
    label = label_map.get(kz, kz)
    return inside, f"{label}: {'INSIDE' if inside else f'outside (cur={now.hour:02d}:{now.minute:02d} UTC)'}"


async def eval_fx_ote(
    cond: Dict, symbol: str, current_price: float, http_client, cache: Dict
) -> Tuple[bool, str]:
    """`fx_ote` — Optimal Trade Entry (ICT golden zone).

    Finds the most recent significant swing high/low over `swing_lookback` bars,
    then checks whether current price is inside the Fibonacci retracement zone
    [fib_low%, fib_high%] — typically 61.8–78.6% (the OTE / golden pocket).

    cfg.direction:     'bullish' (demand) | 'bearish' (supply)
    cfg.swing_lookback: bars to scan for swing H/L (default 20)
    cfg.fib_low:        lower Fib level % (default 61.8)
    cfg.fib_high:       upper Fib level % (default 78.6)
    cfg.timeframe:      kline TF (default '15m')
    """
    direction    = (cond.get("direction") or "bullish").lower()
    tf           = cond.get("timeframe", "15m")
    lookback     = max(10, int(cond.get("swing_lookback") or 20))
    fib_lo       = float(cond.get("fib_low")  or 61.8) / 100
    fib_hi       = float(cond.get("fib_high") or 78.6) / 100

    klines = await _get_klines(symbol, tf, lookback + 5, http_client, cache)
    if not klines or len(klines) < 10:
        return False, "OTE: insufficient data"

    highs = _highs(klines[:-1])   # exclude current forming candle
    lows  = _lows(klines[:-1])
    swing_h = max(highs)
    swing_l = min(lows)
    rng = swing_h - swing_l
    if rng <= 0:
        return False, "OTE: flat range — no swing defined"

    if direction == "bullish":
        # Bullish OTE: price pulled back from swing_h toward swing_l
        # OTE zone: swing_h - fib_hi*rng  to  swing_h - fib_lo*rng
        zone_top = swing_h - fib_lo * rng
        zone_bot = swing_h - fib_hi * rng
    else:
        # Bearish OTE: price pulled back up from swing_l toward swing_h
        # OTE zone: swing_l + fib_lo*rng  to  swing_l + fib_hi*rng
        zone_bot = swing_l + fib_lo * rng
        zone_top = swing_l + fib_hi * rng

    inside = zone_bot <= current_price <= zone_top
    return inside, (
        f"OTE {direction}: zone {zone_bot:.6g}–{zone_top:.6g} "
        f"(swing {swing_l:.6g}–{swing_h:.6g}, fib {fib_lo*100:.1f}–{fib_hi*100:.1f}%) "
        f"price={current_price:.6g} → {'IN ZONE' if inside else 'outside'}"
    )


async def eval_fx_displacement(
    cond: Dict, symbol: str, http_client, cache: Dict
) -> Tuple[bool, str]:
    """`fx_displacement` — institutional momentum candle.

    A displacement candle has a body significantly larger than the recent
    average — signalling institutional order flow entering the market.

    cfg.direction:     'bullish' | 'bearish' | 'any' (default 'any')
    cfg.min_body_ratio: body must be >= N × 14-period average body (default 3)
    cfg.timeframe:      kline TF (default '15m')
    """
    direction  = (cond.get("direction") or "any").lower()
    min_ratio  = float(cond.get("min_body_ratio") or 3.0)
    tf         = cond.get("timeframe", "15m")

    klines = await _get_klines(symbol, tf, 30, http_client, cache)
    if not klines or len(klines) < 5:
        return False, "DISPLACE: insufficient data"

    closes = _closes(klines)
    opens  = _opens(klines)

    # Average body size over the lookback (exclude last candle)
    bodies = [abs(closes[i] - opens[i]) for i in range(len(klines) - 1)]
    avg_body = sum(bodies) / len(bodies) if bodies else 0
    if avg_body == 0:
        return False, "DISPLACE: zero avg body"

    # Last closed candle (index -2 = last complete; -1 may be forming)
    last_body    = abs(closes[-2] - opens[-2])
    is_bullish_c = closes[-2] > opens[-2]
    is_bearish_c = closes[-2] < opens[-2]

    dir_ok = (
        direction == "any" or
        (direction == "bullish" and is_bullish_c) or
        (direction == "bearish" and is_bearish_c)
    )
    ratio = last_body / avg_body
    fired = dir_ok and ratio >= min_ratio
    dir_lbl = "bullish" if is_bullish_c else "bearish"
    return fired, (
        f"Displacement ({dir_lbl}): body={last_body:.6g} ratio={ratio:.2f}× avg "
        f"(need ≥{min_ratio}×) → {'FIRED' if fired else 'miss'}"
    )


async def eval_fx_cisd(
    cond: Dict, symbol: str, http_client, cache: Dict
) -> Tuple[bool, str]:
    """`fx_cisd` — Change in State of Delivery (ICT).

    CISD marks the moment institutional delivery flips direction: price closes
    back through the *open* of the most recent opposing run of candles (the
    origin of the last delivery leg).

    Bullish CISD: after a run of consecutive bearish (close<open) candles, the
      latest closed candle is bullish AND closes ABOVE the open of the FIRST
      candle of that bearish run → selling delivery has ended, buying begins.
    Bearish CISD: mirror — after a run of consecutive bullish candles, the
      latest closed candle is bearish AND closes BELOW the open of the FIRST
      candle of that bullish run.

    cfg.direction: 'bullish' | 'bearish' (default 'bullish')
    cfg.max_run:   max length of the opposing delivery run to scan (default 10)
    cfg.timeframe: kline TF (default '5m')
    """
    direction = (cond.get("direction") or "bullish").lower()
    tf        = cond.get("timeframe", "5m")
    try:
        max_run = max(1, min(int(cond.get("max_run") or 10), 50))
    except (TypeError, ValueError):
        max_run = 10

    klines = await _get_klines(symbol, tf, max_run + 10, http_client, cache)
    if not klines or len(klines) < 4:
        return False, "CISD: insufficient data"

    opens  = _opens(klines)
    closes = _closes(klines)

    # Exclude the still-forming candle (-1); the last CLOSED candle is the
    # confirmation candle.
    o = opens[:-1]
    c = closes[:-1]
    if len(c) < 3:
        return False, "CISD: insufficient closed candles"

    ci = len(c) - 1  # confirmation candle index (last closed)

    if direction == "bullish":
        if not (c[ci] > o[ci]):
            return False, "CISD bull: confirmation candle not bullish"
        # walk back over consecutive bearish candles immediately before confirm
        run_start = None
        j = ci - 1
        while j >= 0 and (ci - 1 - j) < max_run and c[j] < o[j]:
            run_start = j
            j -= 1
        if run_start is None:
            return False, "CISD bull: no preceding bearish delivery run"
        level = o[run_start]  # open of the first candle in the bearish run
        fired = c[ci] > level
        return fired, (
            f"CISD bull: close={c[ci]:.6g} vs level={level:.6g} "
            f"({ci - run_start} bear candles) → {'FIRED' if fired else 'miss'}"
        )
    else:
        if not (c[ci] < o[ci]):
            return False, "CISD bear: confirmation candle not bearish"
        run_start = None
        j = ci - 1
        while j >= 0 and (ci - 1 - j) < max_run and c[j] > o[j]:
            run_start = j
            j -= 1
        if run_start is None:
            return False, "CISD bear: no preceding bullish delivery run"
        level = o[run_start]
        fired = c[ci] < level
        return fired, (
            f"CISD bear: close={c[ci]:.6g} vs level={level:.6g} "
            f"({ci - run_start} bull candles) → {'FIRED' if fired else 'miss'}"
        )


async def eval_fx_equal_hl(
    cond: Dict, symbol: str, current_price: float, http_client, cache: Dict
) -> Tuple[bool, str]:
    """`fx_equal_hl` — Equal Highs / Equal Lows liquidity pools.

    Scans the last `lookback` bars for at least two highs (EQH) or lows (EQL)
    within `tolerance_pips` of each other.  Fires when the cluster exists AND
    current price is near it — price is approaching or has just swept the pool.

    cfg.type:           'eqh' | 'eql' (default 'eqh')
    cfg.lookback:       bars to scan (default 30)
    cfg.tolerance_pips: max pip gap between equal levels (default 3)
    cfg.timeframe:      kline TF (default '15m')
    """
    eq_type = (cond.get("type") or "eqh").lower()
    tf      = cond.get("timeframe", "15m")
    try:
        lookback = max(5, min(int(cond.get("lookback") or 30), 200))
    except (TypeError, ValueError):
        lookback = 30
    try:
        tol_pips = max(0.5, float(cond.get("tolerance_pips") or 3))
    except (TypeError, ValueError):
        tol_pips = 3.0

    pip  = 0.01 if "JPY" in symbol.upper() else 0.0001
    tol  = tol_pips * pip

    klines = await _get_klines(symbol, tf, lookback + 5, http_client, cache)
    if not klines or len(klines) < 5:
        return False, f"EQ{eq_type.upper()}: insufficient data"

    highs = _highs(klines[:-1])
    lows  = _lows(klines[:-1])
    levels = highs if eq_type == "eqh" else lows

    # Find clusters: at least 2 levels within tolerance
    cluster_level = None
    for i in range(len(levels)):
        matches = [levels[j] for j in range(len(levels)) if abs(levels[j] - levels[i]) <= tol]
        if len(matches) >= 2:
            cluster_level = sum(matches) / len(matches)
            break

    if cluster_level is None:
        return False, f"EQ{eq_type.upper()}: no cluster found in {lookback} bars (tol={tol_pips}pips)"

    # Price within 2× tolerance of the cluster = approaching / sweeping
    near = abs(current_price - cluster_level) <= tol * 2
    return near, (
        f"EQ{eq_type.upper()} cluster @ {cluster_level:.6g} "
        f"(tol={tol_pips}pips, {lookback}bars) "
        f"price={current_price:.6g} → {'NEAR' if near else 'away'}"
    )


async def eval_fx_breaker(
    cond: Dict, symbol: str, current_price: float, http_client, cache: Dict
) -> Tuple[bool, str]:
    """`fx_breaker` — Breaker block (failed order block).

    A breaker block is an OB that price has broken through — the failed
    supply becomes support, the failed demand becomes resistance.

    Algorithm:
    1. Find the most recent OB of the OPPOSITE type (bullish breaker = former
       bearish/supply OB; bearish breaker = former bullish/demand OB).
    2. Confirm price has since broken THROUGH it (closed beyond the OB).
    3. Check price has returned to within `tolerance_pct` of that zone.

    cfg.direction:     'bullish' (old supply→support) | 'bearish' (old demand→resistance)
    cfg.lookback:      bars to scan (default 50)
    cfg.tolerance_pct: % tolerance to count as 'at the zone' (default 0.5)
    cfg.timeframe:     kline TF (default '15m')
    """
    direction   = (cond.get("direction") or "bullish").lower()
    tf          = cond.get("timeframe", "15m")
    lookback    = max(20, min(int(cond.get("lookback") or 50), 200))
    tol         = float(cond.get("tolerance_pct") or 0.5) / 100

    klines = await _get_klines(symbol, tf, lookback + 10, http_client, cache)
    if not klines or len(klines) < 15:
        return False, "BREAKER: insufficient data"

    closes = _closes(klines)
    opens  = _opens(klines)
    highs  = _highs(klines)
    lows   = _lows(klines)
    n = len(klines)

    # Walk backward to find a former OB that price has since broken through
    # Bullish breaker: look for a former bearish OB (supply) that bulls broke above
    # Bearish breaker: look for a former bullish OB (demand) that bears broke below
    for i in range(n - 5, 3, -1):
        if direction == "bullish":
            # Former supply OB: bearish candle before impulse down
            is_ob = opens[i] > closes[i]  # bearish candle
            # Subsequent impulse confirmed by next candle lower
            if not is_ob or not (closes[i+1] < closes[i] if i+1 < n else False):
                continue
            ob_high = highs[i]; ob_low = lows[i]
            # Price must have since broken ABOVE ob_high (bullish break through supply)
            broken = any(closes[j] > ob_high for j in range(i + 1, n - 1))
            if not broken:
                continue
            # Price has returned to the zone (now acting as support)
            in_zone = ob_low * (1 - tol) <= current_price <= ob_high * (1 + tol)
            return in_zone, (
                f"Bullish breaker {ob_low:.6g}–{ob_high:.6g} "
                f"(broke above, returned) price={current_price:.6g} → "
                f"{'IN ZONE' if in_zone else 'away'}"
            )
        else:
            # Former demand OB: bullish candle before impulse up
            is_ob = closes[i] > opens[i]  # bullish candle
            if not is_ob or not (closes[i+1] > closes[i] if i+1 < n else False):
                continue
            ob_high = highs[i]; ob_low = lows[i]
            # Price must have since broken BELOW ob_low (bearish break through demand)
            broken = any(closes[j] < ob_low for j in range(i + 1, n - 1))
            if not broken:
                continue
            # Price has returned to the zone (now acting as resistance)
            in_zone = ob_low * (1 - tol) <= current_price <= ob_high * (1 + tol)
            return in_zone, (
                f"Bearish breaker {ob_low:.6g}–{ob_high:.6g} "
                f"(broke below, returned) price={current_price:.6g} → "
                f"{'IN ZONE' if in_zone else 'away'}"
            )

    return False, f"BREAKER ({direction}): no qualifying breaker block found in {lookback} bars"


async def eval_fx_pd_array(
    cond: Dict, symbol: str, current_price: float, http_client, cache: Dict
) -> Tuple[bool, str]:
    """`fx_pd_array` — Premium / Discount array (ICT equilibrium filter).

    Divides the recent swing range at its 50% midpoint:
      discount zone  → price < midpoint  → institutional buy bias
      premium zone   → price > midpoint  → institutional sell bias

    cfg.bias:    'discount' | 'premium' (default 'discount')
    cfg.lookback: bars to find swing H/L (default 50)
    cfg.timeframe: kline TF (default '1h')
    """
    bias     = (cond.get("bias") or "discount").lower()
    tf       = cond.get("timeframe", "1h")
    lookback = max(10, min(int(cond.get("lookback") or 50), 300))

    klines = await _get_klines(symbol, tf, lookback + 5, http_client, cache)
    if not klines or len(klines) < 5:
        return False, "PD Array: insufficient data"

    highs = _highs(klines[:-1])
    lows  = _lows(klines[:-1])
    swing_h = max(highs)
    swing_l = min(lows)
    mid = (swing_h + swing_l) / 2

    in_discount = current_price < mid
    in_premium  = current_price > mid
    hit = (bias == "discount" and in_discount) or (bias == "premium" and in_premium)
    zone_lbl = "discount" if in_discount else "premium"
    return hit, (
        f"PD Array: swing {swing_l:.6g}–{swing_h:.6g}, mid={mid:.6g} "
        f"price={current_price:.6g} → {zone_lbl} "
        f"({'✓' if hit else '✗'} for bias={bias})"
    )


async def eval_fx_judas_swing(
    cond: Dict, symbol: str, current_price: float, http_client, cache: Dict
) -> Tuple[bool, str]:
    """`fx_judas_swing` — Fake sweep at session open then reversal.

    Algorithm:
    1. Find the prior session's range (high and low before the current session open).
    2. Within the first `window_minutes` of the new session, check if price
       extended BEYOND the prior range (the fake / manipulation leg).
    3. Confirm that price has since reversed back INSIDE the prior range by
       at least `reversal_pips` — the real move beginning.

    cfg.session:        'london' (08:00 UTC) | 'ny' (13:30 UTC, default 'london')
    cfg.swing_pips:     min pip extension beyond prior range (default 10)
    cfg.reversal_pips:  min pip reversal from the fake extreme (default 5)
    cfg.timeframe:      kline TF (default '15m')
    """
    session       = (cond.get("session") or "london").lower()
    swing_pips    = max(1.0, float(cond.get("swing_pips")   or 10))
    reversal_pips = max(1.0, float(cond.get("reversal_pips") or 5))
    tf            = cond.get("timeframe", "15m")

    pip = 0.01 if "JPY" in symbol.upper() else 0.0001
    swing_dist    = swing_pips    * pip
    reversal_dist = reversal_pips * pip

    SESSION_OPEN_UTC = {"london": (8, 0), "ny": (13, 30)}
    open_h, open_m = SESSION_OPEN_UTC.get(session, (8, 0))

    klines = await _get_klines(symbol, tf, 60, http_client, cache)
    if not klines or len(klines) < 10:
        return False, "Judas Swing: insufficient data"

    # Identify session-open index by candle timestamps
    def _candle_utc(k) -> datetime:
        ts = int(k[0])
        return datetime.utcfromtimestamp(ts / 1000 if ts > 1e10 else ts)

    open_idx = None
    for i, k in enumerate(klines):
        ct = _candle_utc(k)
        if ct.hour == open_h and ct.minute >= open_m:
            open_idx = i
            break

    if open_idx is None or open_idx < 5:
        return False, f"Judas Swing: session open ({session}) candles not in window"

    # Prior-session range (candles before session open)
    prior_candles = klines[:open_idx]
    prior_h = max(_highs(prior_candles))
    prior_l = min(_lows(prior_candles))

    # Session candles (after open)
    sess_candles = klines[open_idx:]
    if not sess_candles:
        return False, "Judas Swing: no session candles yet"

    sess_highs = _highs(sess_candles)
    sess_lows  = _lows(sess_candles)
    sess_h = max(sess_highs)
    sess_l = min(sess_lows)

    # Detect bullish Judas: swept below prior_l then reversed up
    bearish_sweep = sess_l < prior_l - swing_dist
    bullish_reversal = current_price > sess_l + reversal_dist and current_price > prior_l

    # Detect bearish Judas: swept above prior_h then reversed down
    bullish_sweep = sess_h > prior_h + swing_dist
    bearish_reversal = current_price < sess_h - reversal_dist and current_price < prior_h

    fired = (bearish_sweep and bullish_reversal) or (bullish_sweep and bearish_reversal)
    if fired:
        direction = "bullish (swept lows)" if bearish_sweep else "bearish (swept highs)"
        return True, (
            f"Judas Swing {direction}: prior range {prior_l:.6g}–{prior_h:.6g} "
            f"sess extremes {sess_l:.6g}–{sess_h:.6g} price={current_price:.6g}"
        )
    return False, (
        f"Judas Swing: no confirmed sweep+reversal "
        f"(prior {prior_l:.6g}–{prior_h:.6g}, sess {sess_l:.6g}–{sess_h:.6g})"
    )


def eval_fx_silver_bullet(cond: Dict) -> Tuple[bool, str]:
    """`fx_silver_bullet` — ICT Silver Bullet time windows.

    Three ultra-precise entry windows in NY local time (EST = UTC-5):
      early_am:  03:00–04:00 NY
      am:        10:00–11:00 NY
      pm:        15:00–16:00 NY

    cfg.window: 'early_am' | 'am' | 'pm' | 'any' (default 'any')

    Note: uses a fixed UTC-5 offset (EST).  Summer EDT (UTC-4) shifts the
    windows by 1 hour — acceptable tolerance for a mechanical time filter.
    """
    window = (cond.get("window") or "any").lower()
    now_utc = datetime.utcnow()
    ny_hour = (now_utc.hour - 5) % 24   # approximate NY time (EST)
    cur_min = now_utc.minute

    WINDOWS = {
        "early_am": (3, 4),
        "am":       (10, 11),
        "pm":       (15, 16),
    }
    LABELS = {
        "early_am": "03:00–04:00 NY",
        "am":       "10:00–11:00 NY",
        "pm":       "15:00–16:00 NY",
        "any":      "any Silver Bullet window",
    }

    if window == "any":
        inside = any(s <= ny_hour < e for s, e in WINDOWS.values())
        active = next((k for k, (s, e) in WINDOWS.items() if s <= ny_hour < e), None)
        lbl = f"in {LABELS[active]}" if active else f"outside all SB windows (NY={ny_hour:02d}:{cur_min:02d})"
    else:
        s, e = WINDOWS.get(window, (3, 4))
        inside = s <= ny_hour < e
        lbl = f"{LABELS.get(window, window)}: {'INSIDE' if inside else f'outside (NY={ny_hour:02d}:{cur_min:02d})'}"

    return inside, f"Silver Bullet — {lbl}"


async def eval_opening_range_break(
    cond: Dict, symbol: str, current_price: float, http_client, cache: Dict
) -> Tuple[bool, str]:
    """`opening_range_break` — Opening Range Breakout (ORB).

    Calculates the high/low of the first `orb_minutes` of a trading session,
    then fires when the current price has broken above (up) or below (down)
    that range.

    cfg.session_start: 'london' (08:00 UTC) | 'ny' (13:30 UTC) |
                       'asia' (00:00 UTC) | 'midnight' (00:00 UTC, default 'london')
    cfg.orb_minutes:   size of the opening range in minutes (default 30)
    cfg.direction:     'up' | 'down' | 'both' (default 'both')
    cfg.timeframe:     kline TF for ORB bars (default '5m')
    """
    session_start = (cond.get("session_start") or "london").lower()
    orb_minutes   = max(5, int(cond.get("orb_minutes") or 30))
    direction     = (cond.get("direction") or "both").lower()
    tf            = cond.get("timeframe", "5m")

    SESSION_UTC = {
        "london":   (8,  0),
        "ny":       (13, 30),
        "asia":     (0,  0),
        "midnight": (0,  0),
    }
    open_h, open_m = SESSION_UTC.get(session_start, (8, 0))

    # TF → minutes
    TF_MIN = {"1m": 1, "3m": 3, "5m": 5, "15m": 15, "30m": 30, "1h": 60, "4h": 240}
    tf_mins = TF_MIN.get(tf, 5)
    orb_bars = max(1, orb_minutes // tf_mins)

    # Fetch enough history to cover session start + ORB window
    klines = await _get_klines(symbol, tf, orb_bars + 80, http_client, cache)
    if not klines or len(klines) < orb_bars + 3:
        return False, "ORB: insufficient data"

    def _candle_utc(k) -> datetime:
        ts = int(k[0])
        return datetime.utcfromtimestamp(ts / 1000 if ts > 1e10 else ts)

    # Find session-open candle
    open_idx = None
    for i, k in enumerate(klines):
        ct = _candle_utc(k)
        # Match candle at or just after session open
        if (ct.hour > open_h) or (ct.hour == open_h and ct.minute >= open_m):
            open_idx = i
            break

    if open_idx is None:
        return False, f"ORB: session open ({session_start} {open_h:02d}:{open_m:02d} UTC) not in fetched window"

    # ORB candles = first `orb_bars` after session open
    orb_end = open_idx + orb_bars
    if orb_end > len(klines) - 1:
        return False, f"ORB: opening range still forming ({orb_end - len(klines) + 1} bars remaining)"

    orb_candles = klines[open_idx:orb_end]
    orb_high = max(_highs(orb_candles))
    orb_low  = min(_lows(orb_candles))

    broke_up   = current_price > orb_high
    broke_down = current_price < orb_low
    fired = (direction == "up"   and broke_up)   or \
            (direction == "down" and broke_down) or \
            (direction == "both" and (broke_up or broke_down))

    dir_lbl = ("UP ✓" if broke_up else "DOWN ✓") if (broke_up or broke_down) else "inside range"
    return fired, (
        f"ORB ({session_start}, {orb_minutes}min): range {orb_low:.6g}–{orb_high:.6g} "
        f"price={current_price:.6g} → {dir_lbl}"
    )


async def eval_vwap_cross(
    cond: Dict, symbol: str, current_price: float, http_client, cache: Dict
) -> Tuple[bool, str]:
    """`vwap_cross` — Price crosses the session VWAP.

    Computes VWAP from the start of the current UTC calendar day (midnight)
    using typical-price × volume.  Fires when the most recent closed candle
    crossed above (cross_above) or below (cross_below) the VWAP line.

    cfg.direction: 'cross_above' | 'cross_below' (default 'cross_above')
    cfg.timeframe: kline TF (default '5m')
    """
    direction = (cond.get("direction") or "cross_above").lower()
    tf        = cond.get("timeframe", "5m")

    TF_MIN = {"1m": 1, "3m": 3, "5m": 5, "15m": 15, "30m": 30, "1h": 60, "4h": 240}
    tf_mins = TF_MIN.get(tf, 5)
    # Fetch enough bars to cover one full trading day (1440 min) plus buffer
    bars_per_day = max(50, 1440 // tf_mins + 20)

    klines = await _get_klines(symbol, tf, bars_per_day, http_client, cache)
    if not klines or len(klines) < 5:
        return False, "VWAP Cross: insufficient data"

    # Slice to today's session (UTC midnight onwards)
    today_utc = datetime.utcnow().date()
    def _candle_utc(k) -> datetime:
        ts = int(k[0])
        return datetime.utcfromtimestamp(ts / 1000 if ts > 1e10 else ts)

    session_klines = [k for k in klines if _candle_utc(k).date() >= today_utc]
    if len(session_klines) < 3:
        session_klines = klines[-min(len(klines), 60):]  # fallback: last 60 bars

    # VWAP = Σ(typical_price × volume) / Σ(volume)
    cum_tpv = 0.0; cum_vol = 0.0
    for k in session_klines[:-1]:   # exclude forming candle
        h, l, c, v = float(k[2]), float(k[3]), float(k[4]), float(k[5])
        tp = (h + l + c) / 3
        cum_tpv += tp * v
        cum_vol  += v

    if cum_vol <= 0:
        return False, "VWAP Cross: zero volume in session"

    vwap = cum_tpv / cum_vol

    # Cross detection: compare last two closed candles to vwap
    if len(session_klines) < 3:
        return False, "VWAP Cross: not enough session bars for cross detection"

    prev_close = float(session_klines[-3][4])
    last_close = float(session_klines[-2][4])

    cross_above = prev_close <= vwap < last_close
    cross_below = prev_close >= vwap > last_close

    fired = (direction == "cross_above" and cross_above) or \
            (direction == "cross_below" and cross_below)

    cross_lbl = "crossed ABOVE" if cross_above else ("crossed BELOW" if cross_below else "no cross")
    return fired, (
        f"VWAP Cross: VWAP={vwap:.6g} prev={prev_close:.6g} last={last_close:.6g} "
        f"→ {cross_lbl} ({'✓' if fired else '✗'} for {direction})"
    )


async def _session_vwap_stats(
    symbol: str, tf: str, http_client, cache: Dict
) -> Optional[Tuple[float, float, float]]:
    """Compute session VWAP, its volume-weighted standard deviation, and the
    last closed price for `symbol` on timeframe `tf`.

    Returns (vwap, sd, last_close) or None when data is insufficient.  Mirrors
    the session-slicing logic in `eval_vwap_cross` (UTC-midnight anchored,
    excludes the forming candle).
    """
    TF_MIN = {"1m": 1, "3m": 3, "5m": 5, "15m": 15, "30m": 30, "1h": 60, "4h": 240}
    tf_mins = TF_MIN.get(tf, 5)
    bars_per_day = max(50, 1440 // tf_mins + 20)

    klines = await _get_klines(symbol, tf, bars_per_day, http_client, cache)
    if not klines or len(klines) < 5:
        return None

    today_utc = datetime.utcnow().date()
    def _candle_utc(k) -> datetime:
        ts = int(k[0])
        return datetime.utcfromtimestamp(ts / 1000 if ts > 1e10 else ts)

    session = [k for k in klines if _candle_utc(k).date() >= today_utc]
    if len(session) < 3:
        session = klines[-min(len(klines), 60):]

    closed = session[:-1]  # exclude forming candle
    if len(closed) < 2:
        return None

    cum_tpv = 0.0; cum_vol = 0.0
    tps = []
    for k in closed:
        h, l, c, v = float(k[2]), float(k[3]), float(k[4]), float(k[5])
        tp = (h + l + c) / 3
        cum_tpv += tp * v
        cum_vol += v
        tps.append((tp, v))
    if cum_vol <= 0:
        return None
    vwap = cum_tpv / cum_vol

    # Volume-weighted variance of typical price around VWAP.
    var = sum(v * (tp - vwap) ** 2 for tp, v in tps) / cum_vol
    sd = math.sqrt(var) if var > 0 else 0.0
    last_close = float(closed[-1][4])
    return vwap, sd, last_close


async def eval_atr_filter(
    cond: Dict, symbol: str, http_client, cache: Dict
) -> Tuple[bool, str]:
    """`atr_filter` — Volatility gate using Average True Range.

    cfg.condition  : 'volatile' (ATR% ≥ min_atr_pct) | 'expanding'
                     (ATR rising vs `lookback` bars ago).  Default 'volatile'.
    cfg.min_atr_pct: minimum ATR as a percentage of price (default 0.3).
    cfg.period     : ATR period (default 14).
    cfg.lookback   : bars back for the 'expanding' comparison (default 5).
    cfg.timeframe  : kline TF (default '5m').
    """
    condition   = (cond.get("condition") or "volatile").lower()
    min_atr_pct = float(cond.get("min_atr_pct") or 0.3)
    period      = int(cond.get("period") or 14)
    lookback    = int(cond.get("lookback") or 5)
    tf          = cond.get("timeframe", "5m")

    need = period + lookback + 10
    klines = await _get_klines(symbol, tf, need, http_client, cache)
    if not klines or len(klines) < period + 2:
        return False, "ATR Filter: insufficient data"

    closed = klines[:-1]  # exclude forming candle
    atrs = _atr_values(closed, period)
    if not atrs:
        return False, "ATR Filter: ATR unavailable"

    last_close = float(closed[-1][4])
    atr_now = atrs[-1]
    atr_pct = (atr_now / last_close * 100) if last_close else 0.0

    if condition == "expanding":
        if len(atrs) <= lookback:
            return False, "ATR Filter: not enough ATR history"
        atr_prev = atrs[-1 - lookback]
        fired = atr_now > atr_prev
        return fired, (
            f"ATR Filter: ATR {atr_now:.6g} vs {lookback} bars ago {atr_prev:.6g} "
            f"→ {'expanding' if fired else 'flat/contracting'}"
        )

    # default: 'volatile'
    fired = atr_pct >= min_atr_pct
    return fired, (
        f"ATR Filter: ATR={atr_pct:.2f}% (need ≥ {min_atr_pct}%) "
        f"→ {'volatile' if fired else 'too quiet'}"
    )


async def eval_rvol(
    cond: Dict, symbol: str, http_client, cache: Dict
) -> Tuple[bool, str]:
    """`rvol` — Relative Volume: latest closed bar volume vs its recent average.

    cfg.condition : 'high' (RVOL ≥ threshold) | 'low' (RVOL < threshold).
                    Default 'high'.
    cfg.threshold : RVOL multiple (default 1.5).
    cfg.period    : bars to average for the baseline (default 20).
    cfg.timeframe : kline TF (default '5m').
    """
    condition = (cond.get("condition") or "high").lower()
    threshold = float(cond.get("threshold") or 1.5)
    period    = int(cond.get("period") or 20)
    tf        = cond.get("timeframe", "5m")

    need = period + 5
    klines = await _get_klines(symbol, tf, need, http_client, cache)
    if not klines or len(klines) < period + 2:
        return False, "RVOL: insufficient data"

    closed = klines[:-1]  # exclude forming candle
    vols = [float(k[5]) for k in closed]
    last_vol = vols[-1]
    baseline = vols[-period - 1:-1] if len(vols) > period else vols[:-1]
    avg = sum(baseline) / len(baseline) if baseline else 0.0
    if avg <= 0:
        return False, "RVOL: zero baseline volume"

    rvol = last_vol / avg
    if condition == "low":
        fired = rvol < threshold
    else:
        fired = rvol >= threshold
    return fired, (
        f"RVOL={rvol:.2f}× ({'≥' if condition != 'low' else '<'} {threshold}×) "
        f"→ {'✓' if fired else '✗'}"
    )


async def eval_vwap_bands(
    cond: Dict, symbol: str, current_price: float, http_client, cache: Dict
) -> Tuple[bool, str]:
    """`vwap_bands` — Session VWAP ± standard-deviation bands.

    cfg.condition : 'below_lower' (price ≤ VWAP − N·SD, oversold/long bias)
                  | 'above_upper' (price ≥ VWAP + N·SD, overbought/short bias)
                  | 'inside'      (price between the bands).  Default 'below_lower'.
    cfg.num_std   : band width in standard deviations (default 2.0).
    cfg.timeframe : kline TF (default '5m').
    """
    condition = (cond.get("condition") or "below_lower").lower()
    num_std   = float(cond.get("num_std") or 2.0)
    tf        = cond.get("timeframe", "5m")

    stats = await _session_vwap_stats(symbol, tf, http_client, cache)
    if stats is None:
        return False, "VWAP Bands: insufficient data"
    vwap, sd, last_close = stats
    price = current_price or last_close
    upper = vwap + num_std * sd
    lower = vwap - num_std * sd

    if condition == "above_upper":
        fired = price >= upper
    elif condition == "inside":
        fired = lower < price < upper
    else:  # below_lower
        fired = price <= lower
    return fired, (
        f"VWAP Bands: price={price:.6g} VWAP={vwap:.6g} "
        f"[{lower:.6g} … {upper:.6g}] ±{num_std}σ → {'✓' if fired else '✗'} ({condition})"
    )


async def eval_vwap_bias(
    cond: Dict, symbol: str, current_price: float, http_client, cache: Dict
) -> Tuple[bool, str]:
    """`vwap_bias` — Directional filter: price above/below session VWAP.

    cfg.condition : 'above' (price > VWAP, long bias) | 'below' (short bias).
                    Default 'above'.
    cfg.timeframe : kline TF (default '5m').
    """
    condition = (cond.get("condition") or "above").lower()
    tf        = cond.get("timeframe", "5m")

    stats = await _session_vwap_stats(symbol, tf, http_client, cache)
    if stats is None:
        return False, "VWAP Bias: insufficient data"
    vwap, _sd, last_close = stats
    price = current_price or last_close

    fired = price > vwap if condition == "above" else price < vwap
    return fired, (
        f"VWAP Bias: price={price:.6g} {'>' if condition == 'above' else '<'} "
        f"VWAP={vwap:.6g} → {'✓' if fired else '✗'} ({condition} bias)"
    )


async def eval_stochastic(
    cond: Dict, symbol: str, http_client, cache: Dict
) -> Tuple[bool, str]:
    """`stochastic` — Classic Stochastic Oscillator (%K/%D).

    Computes %K = (close - lowest_low) / (highest_high - lowest_low) × 100
    over k_period bars, then smooths to %D via a d_period SMA.

    cfg.condition : 'oversold' (<20) | 'overbought' (>80) |
                    'bullish_cross' (%K crosses above %D) |
                    'bearish_cross' (%K crosses below %D)
    cfg.k_period  : lookback for %K (default 14)
    cfg.d_period  : smoothing period for %D (default 3)
    cfg.timeframe : kline TF (default '15m')
    """
    condition = (cond.get("condition") or "bullish_cross").lower()
    k_period  = int(cond.get("k_period") or 14)
    d_period  = int(cond.get("d_period") or 3)
    tf        = cond.get("timeframe", "15m")
    need      = k_period + d_period + 10

    klines = await _get_klines(symbol, tf, need, http_client, cache)
    if not klines or len(klines) < k_period + d_period:
        return False, "Stochastic: insufficient data"

    highs  = [float(k[2]) for k in klines]
    lows   = [float(k[3]) for k in klines]
    closes = [float(k[4]) for k in klines]

    k_values: list = []
    for i in range(k_period - 1, len(closes)):
        hh = max(highs[i - k_period + 1: i + 1])
        ll = min(lows[i  - k_period + 1: i + 1])
        k_values.append(50.0 if hh == ll else (closes[i] - ll) / (hh - ll) * 100.0)

    if len(k_values) < d_period + 1:
        return False, "Stochastic: insufficient %K series"

    def _sma(arr: list, n: int) -> list:
        return [sum(arr[i: i + n]) / n for i in range(len(arr) - n + 1)]

    d_values = _sma(k_values, d_period)

    k_now  = k_values[-1]
    d_now  = d_values[-1]
    k_prev = k_values[-2]
    d_prev = d_values[-2] if len(d_values) >= 2 else d_now

    if condition == "oversold":
        fired = k_now < 20
        return fired, f"Stoch %K={k_now:.1f} {'< 20 ✓' if fired else '≥ 20 ✗'}"
    elif condition == "overbought":
        fired = k_now > 80
        return fired, f"Stoch %K={k_now:.1f} {'> 80 ✓' if fired else '≤ 80 ✗'}"
    elif condition == "bullish_cross":
        fired = (k_prev < d_prev) and (k_now >= d_now)
        return fired, f"Stoch %K={k_now:.1f} %D={d_now:.1f} {'bullish cross ✓' if fired else 'no cross ✗'}"
    elif condition == "bearish_cross":
        fired = (k_prev > d_prev) and (k_now <= d_now)
        return fired, f"Stoch %K={k_now:.1f} %D={d_now:.1f} {'bearish cross ✓' if fired else 'no cross ✗'}"
    else:
        return False, f"Stochastic: unknown condition '{condition}'"


async def eval_fx_po3(
    cond: Dict, symbol: str, price: float, http_client, cache: Dict
) -> Tuple[bool, str]:
    """`fx_po3` — ICT Power of 3 (Accumulation / Manipulation / Distribution).

    Detects the classic 3-phase intraday structure:
      1. Accumulation  — price ranges during Asian session (00:00–07:00 UTC)
      2. Manipulation  — London open sweeps above or below the range (stop hunt)
      3. Distribution  — price reverses and moves in the true direction

    cfg.direction  : 'bullish' (swept lows → distributes up) |
                     'bearish' (swept highs → distributes down)
    cfg.sweep_pips : minimum manipulation distance beyond range in pips (default 5)
    cfg.timeframe  : kline TF (default '15m')
    """
    direction  = (cond.get("direction") or "bullish").lower()
    tf         = cond.get("timeframe", "15m")
    sweep_pips = float(cond.get("sweep_pips") or 5)

    klines = await _get_klines(symbol, tf, 80, http_client, cache)
    if not klines or len(klines) < 20:
        return False, "PO3: insufficient data"

    from app.services.forex_engine import pip_size as _pip_size
    ps = _pip_size(symbol)

    def _candle_hour(k) -> int:
        ts = int(k[0])
        return datetime.utcfromtimestamp(ts / 1000 if ts > 1e10 else ts).hour

    asian_candles  = [k for k in klines if 0  <= _candle_hour(k) < 7]
    london_candles = [k for k in klines if 7  <= _candle_hour(k) < 11]

    if len(asian_candles) < 4:
        return False, "PO3: insufficient Asian session bars (need 4+)"
    if not london_candles:
        return False, "PO3: no London/manipulation bars yet"

    accum_high = max(float(k[2]) for k in asian_candles)
    accum_low  = min(float(k[3]) for k in asian_candles)
    accum_pips = (accum_high - accum_low) / ps

    if accum_pips < 5:
        return False, f"PO3: accumulation range too narrow ({accum_pips:.1f} pips)"

    manip_high = max(float(k[2]) for k in london_candles)
    manip_low  = min(float(k[3]) for k in london_candles)

    swept_above = (manip_high - accum_high) / ps >= sweep_pips
    swept_below = (accum_low  - manip_low)  / ps >= sweep_pips
    mid         = (accum_high + accum_low) / 2

    if direction == "bullish":
        manipulation_ok  = swept_below
        distribution_ok  = price > mid
        fired = manipulation_ok and distribution_ok
        return fired, (
            f"PO3 bullish: Asian {accum_low:.5g}–{accum_high:.5g} ({accum_pips:.0f}pips) "
            f"manip_low={manip_low:.5g} swept_below={swept_below} price>mid={distribution_ok} "
            f"→ {'✓' if fired else '✗'}"
        )
    else:
        manipulation_ok  = swept_above
        distribution_ok  = price < mid
        fired = manipulation_ok and distribution_ok
        return fired, (
            f"PO3 bearish: Asian {accum_low:.5g}–{accum_high:.5g} ({accum_pips:.0f}pips) "
            f"manip_high={manip_high:.5g} swept_above={swept_above} price<mid={distribution_ok} "
            f"→ {'✓' if fired else '✗'}"
        )


async def eval_wyckoff(
    cond: Dict, symbol: str, price: float, http_client, cache: Dict
) -> Tuple[bool, str]:
    """`wyckoff` — Wyckoff accumulation/distribution event detection.

    Identifies classic Wyckoff events within a rolling window:
      spring    — candle wicks below support, closes back inside range (bullish)
      upthrust  — candle wicks above resistance, closes back inside range (bearish)
      test      — low-volume re-test of a spring/upthrust level (neutral confirmation)
      markup    — strong bullish close above midpoint with expanding volume
      markdown  — strong bearish close below midpoint with expanding volume

    cfg.phase    : 'spring' | 'upthrust' | 'test' | 'markup' | 'markdown'
    cfg.lookback : bars to define the trading range (default 30)
    cfg.timeframe: kline TF (default '1h')
    """
    phase    = (cond.get("phase") or "spring").lower()
    tf       = cond.get("timeframe", "1h")
    lookback = int(cond.get("lookback") or 30)

    klines = await _get_klines(symbol, tf, lookback + 10, http_client, cache)
    if not klines or len(klines) < 10:
        return False, "Wyckoff: insufficient data"

    closes  = [float(k[4]) for k in klines]
    highs   = [float(k[2]) for k in klines]
    lows    = [float(k[3]) for k in klines]
    volumes = [float(k[5]) for k in klines]

    window       = min(lookback, len(closes) - 2)
    range_high   = max(highs[-window - 1: -1])
    range_low    = min(lows[-window - 1: -1])
    avg_vol      = sum(volumes[-window:]) / window if window else 1.0
    midpoint     = (range_high + range_low) / 2

    last_h   = highs[-1];   last_l  = lows[-1]
    last_c   = closes[-1];  prev_c  = closes[-2]
    last_vol = volumes[-1]

    if phase == "spring":
        fired = last_l < range_low and last_c > range_low
        return fired, (
            f"Wyckoff Spring: support={range_low:.6g} low={last_l:.6g} close={last_c:.6g} "
            f"→ {'spring ✓' if fired else 'no spring ✗'}"
        )

    elif phase == "upthrust":
        fired = last_h > range_high and last_c < range_high
        return fired, (
            f"Wyckoff Upthrust: resist={range_high:.6g} high={last_h:.6g} close={last_c:.6g} "
            f"→ {'upthrust ✓' if fired else 'no upthrust ✗'}"
        )

    elif phase == "test":
        near_support = abs(last_c - range_low)  / max(range_low,  1e-9) < 0.005
        near_resist  = abs(last_c - range_high) / max(range_high, 1e-9) < 0.005
        low_vol      = last_vol < avg_vol * 0.7
        fired        = (near_support or near_resist) and low_vol
        return fired, (
            f"Wyckoff Test: near_s={near_support} near_r={near_resist} "
            f"vol={last_vol:.0f} avg={avg_vol:.0f} → {'test ✓' if fired else '✗'}"
        )

    elif phase == "markup":
        fired = last_c > midpoint and last_c > prev_c and last_vol > avg_vol * 1.2
        return fired, (
            f"Wyckoff Markup: mid={midpoint:.6g} close={last_c:.6g} "
            f"vol_ratio={last_vol / avg_vol:.1f} → {'markup ✓' if fired else '✗'}"
        )

    elif phase == "markdown":
        fired = last_c < midpoint and last_c < prev_c and last_vol > avg_vol * 1.2
        return fired, (
            f"Wyckoff Markdown: mid={midpoint:.6g} close={last_c:.6g} "
            f"vol_ratio={last_vol / avg_vol:.1f} → {'markdown ✓' if fired else '✗'}"
        )

    else:
        return False, f"Wyckoff: unknown phase '{phase}'"


async def evaluate_strategy_conditions(
    strategy_config: Dict,
    symbol: str,
    price_data: Dict,
    enhanced_ta: Dict,
    http_client,
    strictness_level: int = 0,
) -> Tuple[bool, List[str]]:
    """
    Evaluate all entry_conditions in a strategy config.
    strictness_level:
      0 = standard  — use configured AND/OR operator
      1 = selective  — force AND (all conditions must pass), 3-5 trades/day
      2 = sniper     — force AND + require 90% pass rate, 1-2 trades/day
    Returns (all_passed: bool, detail_lines: list[str])
    """
    entry   = strategy_config.get("entry_conditions", {})
    op      = entry.get("operator", "AND").upper()
    conds   = entry.get("conditions", [])
    price   = price_data.get("price", 0)
    cache: Dict = {}  # shared klines cache for this evaluation pass
    # Asset-class hint — read by _get_klines to route non-crypto fetches
    # through the yfinance-backed tradfi provider instead of MEXC/Binance.
    _ac = strategy_config.get("asset_class") or price_data.get("_asset_class")
    if _ac and _ac != "crypto":
        cache["__asset_class__"] = _ac

    # Override operator based on strictness
    if strictness_level >= 1:
        op = "AND"

    results, details = [], []

    async def _eval_one(cond) -> Tuple[bool, str]:
        ctype = cond.get("type", "")
        try:
            if ctype == "indicator":
                return await eval_indicator(
                    cond, price_data, enhanced_ta, symbol, http_client, cache)
            elif ctype == "price_momentum":
                return await eval_price_momentum(cond, symbol, http_client, cache)
            elif ctype == "volume_spike":
                return eval_volume_spike(cond, price_data)
            elif ctype == "support_resistance":
                return eval_support_resistance(cond, enhanced_ta, price)
            elif ctype == "fvg":
                return await eval_fvg(cond, symbol, price, http_client, cache)
            elif ctype == "ifvg":
                # Inverse FVG — price re-enters a mitigated gap.
                # Shares the same evaluator as FVG; the config supplies
                # condition='price_in_gap' and only_unfilled=False by default,
                # which naturally selects already-filled (mitigated) gaps.
                _ifvg_ok, _ifvg_det = await eval_fvg(
                    cond, symbol, price, http_client, cache)
                # eval_fvg labels the gap "Bullish/Bearish FVG", but an iFVG
                # INVERTS the bias: a mitigated bearish gap becomes support
                # (LONG) and a bullish gap becomes resistance (SHORT). Annotate
                # the detail so the trade card explains why a "Bearish FVG"
                # produces a LONG. Keep the original "Bullish/Bearish FVG"
                # substring intact — the BOTH-direction inference in
                # strategy_executor parses it (case-insensitive).
                if "Bearish FVG" in _ifvg_det:
                    _ifvg_det = _ifvg_det.replace(
                        "Bearish FVG", "Bearish FVG (inverted → LONG)", 1)
                elif "Bullish FVG" in _ifvg_det:
                    _ifvg_det = _ifvg_det.replace(
                        "Bullish FVG", "Bullish FVG (inverted → SHORT)", 1)
                return _ifvg_ok, _ifvg_det
            elif ctype == "candlestick":
                return await eval_candlestick(cond, symbol, http_client, cache)
            elif ctype == "consecutive_candles":
                return await eval_consecutive_candles(cond, symbol, http_client, cache)
            elif ctype == "market_structure":
                return await eval_market_structure(cond, symbol, price, http_client, cache)
            elif ctype == "order_block":
                return await eval_order_block(cond, symbol, price, http_client, cache)
            elif ctype == "fibonacci":
                return await eval_fibonacci(cond, symbol, price, http_client, cache)
            elif ctype == "divergence":
                return await eval_divergence(cond, symbol, http_client, cache)
            elif ctype == "funding_rate":
                return await eval_funding_rate(cond, symbol, http_client)
            elif ctype == "open_interest":
                return await eval_open_interest(cond, symbol, http_client)
            elif ctype == "session":
                return eval_session(cond)
            elif ctype == "price_relative":
                return await eval_price_relative(cond, symbol, price, http_client, cache)
            elif ctype == "sentiment":
                return await eval_sentiment(cond, symbol, http_client)
            elif ctype == "liquidation":
                return await eval_liquidation(cond, symbol, price, http_client, cache)
            elif ctype in ("sma", "sma_cross", "sma_ribbon"):
                return await eval_indicator(
                    {**cond, "type": "indicator", "name": ctype},
                    price_data, enhanced_ta, symbol, http_client, cache)
            elif ctype == "supertrend":
                return await eval_indicator(
                    {**cond, "name": "supertrend"}, price_data, enhanced_ta, symbol, http_client, cache)
            elif ctype == "trend_reversal":
                return await eval_trend_reversal(cond, symbol, price, http_client, cache)
            elif ctype == "sustained_trend":
                return await eval_sustained_trend(cond, symbol, http_client, cache)
            # ── Forex-specific blocks ───────────────────────────────────────
            # These only make sense for asset_class=forex but are evaluable on
            # any OHLC series; the wizard restricts them to forex strategies.
            elif ctype == "forex_session":
                return eval_forex_session(cond)
            elif ctype == "forex_session_break":
                return await eval_forex_session_break(
                    cond, symbol, price, http_client, cache)
            elif ctype == "forex_prev_level":
                return await eval_forex_prev_level(
                    cond, symbol, price, http_client, cache)
            elif ctype == "forex_news_avoidance":
                return await eval_forex_news_avoidance(cond, symbol)
            elif ctype == "forex_currency_strength":
                return await eval_forex_currency_strength(cond, symbol)
            elif ctype == "forex_liquidity_pa":
                return await eval_forex_liquidity_pa(
                    cond, symbol, http_client, cache)
            elif ctype == "forex_cot":
                return await eval_forex_cot(cond, symbol)
            # ── ICT / day-trading blocks ────────────────────────────────────
            elif ctype == "fx_killzone":
                return eval_fx_killzone(cond)
            elif ctype == "fx_ote":
                return await eval_fx_ote(cond, symbol, price, http_client, cache)
            elif ctype == "fx_displacement":
                return await eval_fx_displacement(cond, symbol, http_client, cache)
            elif ctype == "fx_equal_hl":
                return await eval_fx_equal_hl(cond, symbol, price, http_client, cache)
            elif ctype == "fx_cisd":
                return await eval_fx_cisd(cond, symbol, http_client, cache)
            elif ctype == "fx_breaker":
                return await eval_fx_breaker(cond, symbol, price, http_client, cache)
            elif ctype == "fx_pd_array":
                return await eval_fx_pd_array(cond, symbol, price, http_client, cache)
            elif ctype == "fx_judas_swing":
                return await eval_fx_judas_swing(cond, symbol, price, http_client, cache)
            elif ctype == "fx_silver_bullet":
                return eval_fx_silver_bullet(cond)
            elif ctype == "opening_range_break":
                return await eval_opening_range_break(cond, symbol, price, http_client, cache)
            elif ctype == "vwap_cross":
                return await eval_vwap_cross(cond, symbol, price, http_client, cache)
            elif ctype == "atr_filter":
                return await eval_atr_filter(cond, symbol, http_client, cache)
            elif ctype == "rvol":
                return await eval_rvol(cond, symbol, http_client, cache)
            elif ctype == "vwap_bands":
                return await eval_vwap_bands(cond, symbol, price, http_client, cache)
            elif ctype == "vwap_bias":
                return await eval_vwap_bias(cond, symbol, price, http_client, cache)
            elif ctype == "stochastic":
                return await eval_stochastic(cond, symbol, http_client, cache)
            elif ctype == "fx_po3":
                return await eval_fx_po3(cond, symbol, price, http_client, cache)
            elif ctype == "wyckoff":
                return await eval_wyckoff(cond, symbol, price, http_client, cache)
            # ── Stock-specific blocks ───────────────────────────────────────
            elif ctype == "stock_earnings_avoidance":
                return await eval_stock_earnings_avoidance(cond, symbol)
            # ── Day-trading blocks (all asset classes) ──────────────────────
            elif ctype == "pivot_points":
                return await eval_pivot_points(cond, symbol, price, http_client, cache)
            elif ctype == "session_level":
                return await eval_session_level(cond, symbol, price, http_client, cache)
            else:
                return False, f"Unknown condition type: {ctype}"
        except Exception as e:
            logger.warning(f"Condition eval error {symbol} {ctype}: {e}")
            return False, f"[ERROR] {ctype}: {e}"

    # Evaluate all conditions in parallel — they are independent of each other
    raw_results = await asyncio.gather(*[_eval_one(c) for c in conds], return_exceptions=True)
    for r in raw_results:
        if isinstance(r, Exception):
            results.append(False)
            details.append(f"❌ [ERROR] {r}")
        else:
            passed, detail = r
            results.append(passed)
            details.append(f"{'✅' if passed else '❌'} {detail}")

    if not results:
        return False, ["No conditions defined"]

    base_passed = all(results) if op == "AND" else any(results)

    # Sniper mode: additionally gate on 90% pass rate so borderline sets don't fire
    if strictness_level >= 2 and base_passed:
        pass_rate = sum(results) / len(results)
        if pass_rate < 0.90:
            details.append(f"❌ Sniper gate: only {pass_rate*100:.0f}% conditions passed (need 90%)")
            return False, details

    return base_passed, details
