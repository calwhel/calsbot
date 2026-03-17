"""
Backtest Engine — TradeHub Strategy Portal

Replays a wizard strategy config against historical OHLCV data.
Uses Binance Futures (falling back to Binance spot) for candle history.
All indicator math is synchronous and self-contained; no live API calls during replay.
"""
import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional

import httpx

logger = logging.getLogger(__name__)

# ── Candle field helpers ────────────────────────────────────────────────────────
def _closes(k): return [float(x[4]) for x in k]
def _highs(k):  return [float(x[2]) for x in k]
def _lows(k):   return [float(x[3]) for x in k]
def _opens(k):  return [float(x[1]) for x in k]
def _vols(k):   return [float(x[5]) for x in k]

# ── Indicator math (synchronous, no HTTP) ───────────────────────────────────────
def _ema_list(data: List[float], period: int) -> List[float]:
    if not data: return []
    k = 2 / (period + 1)
    ema = [data[0]]
    for v in data[1:]:
        ema.append(v * k + ema[-1] * (1 - k))
    return ema

def _ema(data: List[float], period: int) -> Optional[float]:
    if len(data) < period: return None
    return _ema_list(data, period)[-1]

def _rsi_values(closes: List[float], period: int = 14) -> List[float]:
    if len(closes) < period + 1:
        return []
    gains, losses = [], []
    for i in range(1, len(closes)):
        d = closes[i] - closes[i - 1]
        gains.append(max(d, 0.0))
        losses.append(max(-d, 0.0))
    ag = sum(gains[:period]) / period
    al = sum(losses[:period]) / period
    rsi = []
    for i in range(period, len(closes)):
        if i > period:
            ag = (ag * (period - 1) + gains[i - 1]) / period
            al = (al * (period - 1) + losses[i - 1]) / period
        rsi.append(100 - 100 / (1 + ag / (al or 1e-10)))
    return rsi

def _rsi(closes: List[float], period: int = 14) -> Optional[float]:
    v = _rsi_values(closes, period)
    return v[-1] if v else None

def _macd(closes: List[float], fast: int = 12, slow: int = 26, signal: int = 9) -> Optional[Dict]:
    if len(closes) < slow + signal:
        return None
    fast_ema = _ema_list(closes, fast)
    slow_ema = _ema_list(closes, slow)
    macd_line = [f - s for f, s in zip(fast_ema[slow - 1:], slow_ema[slow - 1:])]
    if len(macd_line) < signal:
        return None
    sig_line = _ema_list(macd_line, signal)
    histogram = macd_line[-1] - sig_line[-1]
    prev_hist = (macd_line[-2] - sig_line[-2]) if len(macd_line) > signal else None
    cross = "NONE"
    if prev_hist is not None:
        if prev_hist < 0 and histogram > 0:
            cross = "BULLISH_CROSS"
        elif prev_hist > 0 and histogram < 0:
            cross = "BEARISH_CROSS"
        elif histogram > 0:
            cross = "BULLISH"
        else:
            cross = "BEARISH"
    return {"histogram": histogram, "cross": cross}

def _bb(closes: List[float], period: int = 20, std_mult: float = 2.0) -> Optional[Dict]:
    if len(closes) < period:
        return None
    w = closes[-period:]
    mid = sum(w) / period
    variance = sum((x - mid) ** 2 for x in w) / period
    std = variance ** 0.5
    upper = mid + std_mult * std
    lower = mid - std_mult * std
    width = (upper - lower) / mid * 100 if mid else 0
    return {"upper": upper, "lower": lower, "mid": mid, "width": width, "squeeze": width < 3.0}

def _stochrsi(closes: List[float], rsi_period: int = 14, stoch_period: int = 14) -> Optional[Dict]:
    rsi_vals = _rsi_values(closes, rsi_period)
    if len(rsi_vals) < stoch_period:
        return None
    window = rsi_vals[-stoch_period:]
    lo, hi = min(window), max(window)
    if hi == lo:
        return {"k": 50.0}
    k = (rsi_vals[-1] - lo) / (hi - lo) * 100
    prev_k = (rsi_vals[-2] - lo) / (hi - lo) * 100 if len(rsi_vals) > stoch_period else k
    return {"k": k, "prev_k": prev_k}

def _supertrend(klines: List, period: int = 10, multiplier: float = 3.0) -> str:
    if len(klines) < period + 1:
        return "UNKNOWN"
    tr_list = []
    for i in range(1, len(klines)):
        h, l = float(klines[i][2]), float(klines[i][3])
        pc = float(klines[i - 1][4])
        tr_list.append(max(h - l, abs(h - pc), abs(l - pc)))
    atr_vals = _ema_list(tr_list, period)
    atr = atr_vals[-1]
    hl2 = (float(klines[-1][2]) + float(klines[-1][3])) / 2
    basic_lower = hl2 - multiplier * atr
    curr_close = float(klines[-1][4])
    return "BULLISH" if curr_close > basic_lower else "BEARISH"

def _vol_ratio(klines: List, lookback: int = 20) -> float:
    vols = _vols(klines)
    if len(vols) < lookback + 1:
        return 1.0
    avg = sum(vols[-lookback - 1:-1]) / lookback
    return vols[-1] / avg if avg > 0 else 1.0

def _price_momentum_pct(klines: List, window_candles: int) -> Optional[float]:
    if len(klines) < window_candles + 1:
        return None
    ref_open = float(klines[-window_candles][1])
    curr_close = float(klines[-1][4])
    if ref_open == 0:
        return None
    return (curr_close - ref_open) / ref_open * 100

def _cmp(actual: float, operator: str, threshold: float) -> bool:
    return {
        "gt":  actual > threshold,
        "gte": actual >= threshold,
        "lt":  actual < threshold,
        "lte": actual <= threshold,
        "eq":  abs(actual - threshold) < 0.001,
    }.get(operator, False)

# ── Condition evaluator (no HTTP, uses pre-fetched klines) ─────────────────────
def eval_condition_bt(cond: Dict, klines: List, interval_min: int = 5) -> bool:
    """
    Evaluate a single condition against a historical candle slice.
    Returns True for unsupported condition types (pass-through) so they
    don't falsely block signals during replay.
    """
    ctype = cond.get("type", "")

    # ── Price Momentum ──────────────────────────────────────────────────────────
    if ctype == "price_momentum":
        window_min = int(cond.get("window_minutes", 10))
        op         = cond.get("operator", "gt")
        val        = float(cond.get("value", 5))
        req_dir    = cond.get("direction", "any")
        window_c   = max(1, round(window_min / interval_min))
        pct = _price_momentum_pct(klines, window_c)
        if pct is None: return False
        if req_dir == "up"   and pct < 0: return False
        if req_dir == "down" and pct > 0: return False
        return _cmp(abs(pct), op, val)

    # ── Volume Spike ────────────────────────────────────────────────────────────
    if ctype == "volume_spike":
        mult = float(cond.get("multiplier", 1.5))
        return _vol_ratio(klines, lookback=20) >= mult

    # ── Indicator ──────────────────────────────────────────────────────────────
    if ctype == "indicator":
        name    = (cond.get("name") or "").lower()
        op      = cond.get("operator", "gt")
        val     = float(cond.get("value", 50))
        sub     = cond.get("condition", "")
        closes  = _closes(klines)

        if name == "rsi":
            period = int(cond.get("period", 14))
            rsi = _rsi(closes, period)
            if rsi is None: return False
            return _cmp(rsi, op, val)

        if name == "macd":
            m = _macd(closes)
            if not m: return False
            if sub in ("bullish", "bullish_cross"):   return m["cross"] in ("BULLISH", "BULLISH_CROSS")
            if sub in ("bearish", "bearish_cross"):   return m["cross"] in ("BEARISH", "BEARISH_CROSS")
            if sub == "crosses_above":                return m["cross"] == "BULLISH_CROSS"
            if sub == "crosses_below":                return m["cross"] == "BEARISH_CROSS"
            return _cmp(m["histogram"], op, val)

        if name == "ema":
            period  = int(cond.get("period", 20))
            period2 = int(cond.get("period2", 50))
            ema_fast = _ema(closes, period)
            curr     = closes[-1]
            if sub in ("above", "price_above"):
                return ema_fast is not None and curr > ema_fast
            if sub in ("below", "price_below"):
                return ema_fast is not None and curr < ema_fast
            if sub in ("bullish_cross", "crosses_above"):
                ema_slow = _ema(closes, period2)
                pf = _ema(closes[:-1], period)  if len(closes) > period  else None
                ps = _ema(closes[:-1], period2) if len(closes) > period2 else None
                if all(v is not None for v in [ema_fast, ema_slow, pf, ps]):
                    return pf <= ps and ema_fast > ema_slow
            if sub in ("bearish_cross", "crosses_below"):
                ema_slow = _ema(closes, period2)
                pf = _ema(closes[:-1], period)  if len(closes) > period  else None
                ps = _ema(closes[:-1], period2) if len(closes) > period2 else None
                if all(v is not None for v in [ema_fast, ema_slow, pf, ps]):
                    return pf >= ps and ema_fast < ema_slow
            return False

        if name == "bb":
            bb = _bb(closes)
            if not bb: return False
            curr = closes[-1]
            if sub == "squeeze":             return bb["squeeze"]
            if sub == "price_above_upper":   return curr > bb["upper"]
            if sub == "price_below_lower":   return curr < bb["lower"]
            if sub == "price_near_upper":    return curr > bb["mid"] and curr <= bb["upper"]
            if sub == "price_near_lower":    return curr < bb["mid"] and curr >= bb["lower"]
            return False

        if name == "stochrsi":
            sr = _stochrsi(closes)
            if not sr: return False
            k = sr["k"]
            if sub == "oversold":      return k < 20
            if sub == "overbought":    return k > 80
            if sub == "bullish_cross": return k < 20 and sr.get("prev_k", k) > k
            if sub == "bearish_cross": return k > 80 and sr.get("prev_k", k) < k
            return _cmp(k, op, val)

        if name == "supertrend":
            st = _supertrend(klines)
            if sub == "bearish": return st == "BEARISH"
            return st == "BULLISH"

        if name in ("volume", "volume_spike"):
            return _cmp(_vol_ratio(klines), op, val)

        return True  # unsupported indicator — pass through

    # ── Candlestick ─────────────────────────────────────────────────────────────
    if ctype == "candlestick":
        if len(klines) < 2: return False
        sub = cond.get("condition", "")
        o  = float(klines[-1][1]); h  = float(klines[-1][2])
        l  = float(klines[-1][3]); c  = float(klines[-1][4])
        po = float(klines[-2][1]); pc = float(klines[-2][4])
        body = abs(c - o)
        rng  = h - l
        if not rng: return False
        if sub == "bullish_engulfing": return c > o and pc > po and c > po and o < pc
        if sub == "bearish_engulfing": return c < o and pc < po and c < po and o > pc
        if sub in ("hammer", "pin_bar"):
            lw = min(o, c) - l
            uw = h - max(o, c)
            return lw > body * 2 and uw < body * 0.5
        if sub == "shooting_star":
            uw = h - max(o, c)
            lw = min(o, c) - l
            return uw > body * 2 and lw < body * 0.5
        if sub == "doji": return body / rng < 0.1
        return True

    # ── Consecutive Candles ─────────────────────────────────────────────────────
    if ctype == "consecutive_candles":
        n   = int(cond.get("count", 3))
        req = cond.get("direction", "green")
        if len(klines) < n: return False
        for k in klines[-n:]:
            if req == "green" and float(k[4]) <= float(k[1]): return False
            if req == "red"   and float(k[4]) >= float(k[1]): return False
        return True

    return True  # all other types pass through


# ── Candle fetching ──────────────────────────────────────────────────────────────
#
# Priority:
#   1. Gate.io Futures (gate.io/api/v4) — covers all USDT perp pairs including
#      low caps (PIPPIN, FARTCOIN, WIF, BONK…). Returns full 30d in one call.
#   2. Kraken (api.kraken.com) — fallback for coins not on Gate.io. Covers
#      BTC, ETH, SOL and major alts with excellent historical depth.
#
# Both sources return 1h OHLCV with proper startTime pagination.
# No proxy is used — if a coin is on neither exchange, we surface an error.


def _to_gateio_pair(symbol: str) -> str:
    """Convert PIPPINUSDT → PIPPIN_USDT for Gate.io futures."""
    base = symbol.upper().replace("USDT", "").replace("BUSD", "").replace("_USDT", "")
    return f"{base}_USDT"


# Kraken uses non-standard names for a handful of coins
_KRAKEN_MAP: Dict[str, str] = {
    "BTC": "XBTUSD", "DOGE": "XDGUSD",
}

def _to_kraken_pair(symbol: str) -> str:
    base = symbol.upper().replace("USDT", "").replace("BUSD", "").replace("USD", "")
    return _KRAKEN_MAP.get(base, f"{base}USD")


async def _fetch_gateio(
    symbol: str,
    days: int,
    http_client: httpx.AsyncClient,
) -> List:
    """
    Fetch 1h candles from Gate.io Futures for the given USDT symbol.
    Gate.io returns up to 999 candles per request and honours the `from`
    parameter, making pagination straightforward.
    Returns [] if the symbol isn't listed on Gate.io.
    """
    pair    = _to_gateio_pair(symbol)
    now_s   = int(datetime.now(timezone.utc).timestamp())
    since_s = now_s - days * 86400
    candles: List = []

    for _ in range(10):
        try:
            resp = await http_client.get(
                "https://api.gateio.ws/api/v4/futures/usdt/candlesticks",
                params={"contract": pair, "interval": "1h", "limit": 999, "from": since_s},
                timeout=15,
            )
        except Exception as exc:
            logger.debug(f"[Backtest] Gate.io fetch error {pair}: {exc}")
            break

        if resp.status_code != 200:
            break

        data = resp.json()
        if not isinstance(data, list) or not data:
            break

        for k in data:
            candles.append([
                int(k["t"]) * 1000,   # ts → ms
                float(k["o"]),         # open
                float(k["h"]),         # high
                float(k["l"]),         # low
                float(k["c"]),         # close
                float(k.get("v", 0)), # volume
            ])

        last_ts = int(data[-1]["t"])
        if last_ts >= now_s - 3600:
            break   # reached current time
        since_s = last_ts + 3600

    return candles


async def _fetch_kraken(
    symbol: str,
    days: int,
    http_client: httpx.AsyncClient,
) -> List:
    """
    Fetch 1h candles from Kraken for the given symbol.
    Returns [] if the symbol isn't listed on Kraken.
    """
    pair  = _to_kraken_pair(symbol)
    now_s = int(datetime.now(timezone.utc).timestamp())
    since = now_s - days * 86400
    candles: List = []

    for _ in range(15):
        try:
            resp = await http_client.get(
                "https://api.kraken.com/0/public/OHLC",
                params={"pair": pair, "interval": 60, "since": since},
                timeout=15,
            )
            d = resp.json()
        except Exception as exc:
            logger.debug(f"[Backtest] Kraken fetch error {pair}: {exc}")
            break

        if d.get("error"):
            break

        result     = d.get("result", {})
        candle_key = next((k for k in result if k != "last"), None)
        if not candle_key:
            break

        rows = result[candle_key]
        if not rows:
            break

        for c in rows:
            candles.append([
                int(c[0]) * 1000,
                float(c[1]), float(c[2]), float(c[3]), float(c[4]),
                float(c[6]),  # volume is index 6 in Kraken
            ])

        last_ts = result.get("last", 0)
        if not last_ts or last_ts >= now_s - 3600:
            break
        since = last_ts

    return candles


async def _fetch_historical(
    symbol: str,
    days: int,
    http_client: httpx.AsyncClient,
) -> tuple:
    """
    Fetch 1h OHLCV candles for symbol over the past `days` days.

    Returns (candles, source_label, proxy_used):
      candles      — list of [ts_ms, open, high, low, close, volume]
      source_label — human-readable exchange name shown in results
      proxy_used   — always False (we no longer use a BTC proxy)
    """
    # 1. Try Gate.io first — covers virtually all USDT perp pairs
    candles = await _fetch_gateio(symbol, days, http_client)
    if len(candles) >= 60:
        base = symbol.upper().replace("USDT", "")
        return candles, f"{base} · Gate.io", False

    # 2. Fall back to Kraken for coins not listed on Gate.io
    candles = await _fetch_kraken(symbol, days, http_client)
    if len(candles) >= 60:
        base = symbol.upper().replace("USDT", "")
        return candles, f"{base} · Kraken", False

    return [], symbol, False


# ── Primary condition builder ───────────────────────────────────────────────────
def _build_primary_cond(primary_type: str, primary_cfg: Dict, direction: str) -> Dict:
    """Convert wizard primaryType + primaryCfg into a backtest condition dict."""
    dir_map = {"LONG": "up", "SHORT": "down"}
    pm_dir  = dir_map.get(direction, "any")

    if primary_type == "price_momentum":
        return {
            "type":           "price_momentum",
            "window_minutes": int(primary_cfg.get("pm_window", 15)),
            "operator":       "gt",
            "value":          float(primary_cfg.get("pm_pct", 5)),
            "direction":      pm_dir,
        }
    if primary_type == "volume_spike":
        return {
            "type":       "volume_spike",
            "multiplier": float(primary_cfg.get("multiplier", 2.0)),
        }
    if primary_type in ("rsi", "macd", "ema", "bb", "stochrsi", "supertrend", "volume"):
        return {"type": "indicator", "name": primary_type, **primary_cfg}

    return {"type": primary_type, **primary_cfg}


def _build_confirm_cond(conf: Dict) -> Dict:
    """Normalise a wizard confirmation dict into a backtest condition dict."""
    ctype = conf.get("type", "")
    if ctype in ("rsi", "macd", "ema", "bb", "stochrsi", "supertrend", "volume"):
        return {"type": "indicator", "name": ctype, **conf}
    return conf


# ── Stats helpers ───────────────────────────────────────────────────────────────
TAKER_FEE_PCT = 0.05   # 0.05 % per side (Bitunix taker rate)
ROUND_TRIP_FEE = TAKER_FEE_PCT * 2  # 0.10 % on notional per round trip

def _compute_pnl(direction: str, entry: float, exit_price: float,
                 leverage: int, include_fees: bool = True) -> float:
    """
    Returns P&L as % of margin.
    Fee drag = ROUND_TRIP_FEE * leverage  (fees are on notional, so they scale with leverage).
    Example: 5× leverage, 0.10 % round-trip → 0.50 % margin drag per trade.
    """
    if direction == "LONG":
        raw = (exit_price - entry) / entry * 100 * leverage
    else:
        raw = (entry - exit_price) / entry * 100 * leverage
    if include_fees:
        raw -= ROUND_TRIP_FEE * leverage
    return raw


def _compute_stats(trades: List[Dict], interval_min: int) -> Dict:
    closed = [t for t in trades if t["outcome"] in ("WIN", "LOSS")]
    wins   = [t for t in closed if t["outcome"] == "WIN"]
    losses = [t for t in closed if t["outcome"] == "LOSS"]

    if not closed:
        return {
            "total_signals": len(trades), "closed_trades": 0,
            "wins": 0, "losses": 0, "win_rate": 0,
            "total_pnl": 0, "avg_win": 0, "avg_loss": 0,
            "max_drawdown": 0, "avg_hold_minutes": 0, "profit_factor": 0,
        }

    total_pnl = sum(t["pnl_pct"] for t in closed)
    avg_win   = sum(t["pnl_pct"] for t in wins)   / len(wins)   if wins   else 0
    avg_loss  = sum(t["pnl_pct"] for t in losses) / len(losses) if losses else 0

    equity, peak, max_dd = 100.0, 100.0, 0.0
    for t in closed:
        equity += t["pnl_pct"]
        if equity > peak: peak = equity
        dd = (peak - equity) / peak * 100
        if dd > max_dd: max_dd = dd

    avg_hold = sum(t["hold_candles"] for t in closed) / len(closed) * interval_min

    gross_profit = sum(t["pnl_pct"] for t in wins)
    gross_loss   = abs(sum(t["pnl_pct"] for t in losses))
    pf = round(gross_profit / gross_loss, 2) if gross_loss > 0 else (99.0 if gross_profit > 0 else 0.0)

    return {
        "total_signals":     len(trades),
        "closed_trades":     len(closed),
        "wins":              len(wins),
        "losses":            len(losses),
        "win_rate":          round(len(wins) / len(closed) * 100, 1),
        "total_pnl":         round(total_pnl, 2),
        "avg_win":           round(avg_win, 2),
        "avg_loss":          round(avg_loss, 2),
        "max_drawdown":      round(max_dd, 2),
        "avg_hold_minutes":  round(avg_hold),
        "profit_factor":     pf,
    }


def _build_equity_curve(trades: List[Dict]) -> List[Dict]:
    equity = 100.0
    points = [{"x": 0, "y": round(equity, 2)}]
    for i, t in enumerate(trades):
        if t["outcome"] == "OPEN":
            continue
        equity += t["pnl_pct"]
        points.append({"x": i + 1, "y": round(equity, 2)})
    return points


# ── Main entry point ────────────────────────────────────────────────────────────
async def run_backtest(config: Dict, days: int = 30) -> Dict:
    """
    Run a full strategy backtest against historical OHLCV data.

    config keys (wizard state):
        direction     LONG | SHORT
        primaryType   price_momentum | volume_spike | rsi | macd | ema | ...
        primaryCfg    dict of config for primary signal
        confirms      list of confirmation condition dicts
        tp1           take-profit % (float)
        sl            stop-loss % (float)
        leverage      leverage multiplier (int)
        timeframe     5m | 15m | 1h | ...
        singleCoin    symbol to test on (e.g. "BTCUSDT"); defaults to BTCUSDT

    Returns dict with keys: symbol, days, interval, total_candles,
                            trades, stats, equity_curve, [error]
    """
    direction    = config.get("direction", "LONG")
    tp_pct       = float(config.get("tp1", 3)) / 100
    sl_pct       = float(config.get("sl", 1.5)) / 100
    leverage     = int(config.get("leverage", 10))
    timeframe    = config.get("timeframe", "5m")
    primary_type = config.get("primaryType", "price_momentum")
    primary_cfg  = config.get("primaryCfg") or {}
    confirms     = config.get("confirms") or []

    # Backtest always uses 1h candles via Kraken (proper historical pagination)
    bt_interval  = "1h"
    interval_min = 60  # minutes per candle

    raw_symbol = (config.get("singleCoin") or "BTCUSDT").upper().strip()
    symbol = raw_symbol if raw_symbol.endswith("USDT") else raw_symbol + "USDT"

    try:
        async with httpx.AsyncClient() as client:
            candles, source_label, proxy_used = await _fetch_historical(symbol, days, client)
    except Exception as exc:
        return {"error": f"Failed to fetch candle data: {exc}"}

    if len(candles) < 60:
        base = symbol.replace("USDT", "")
        return {"error": f"No historical data found for {base}. Check the symbol name or try a different coin."}

    logger.info(f"[Backtest] {source_label} 1h {days}d: {len(candles)} candles")

    warmup = 60
    primary_cond = _build_primary_cond(primary_type, primary_cfg, direction)
    # Max hold: 48 h by default, configurable via config["maxHoldHours"].
    max_hold_h = int(config.get("maxHoldHours") or 48)
    max_hold_c = max(1, max_hold_h)

    trades: List[Dict] = []
    open_trade    = None
    prev_cond_met = False  # used for edge detection on the primary condition

    for i in range(warmup, len(candles)):
        candle     = candles[i]
        curr_ts    = int(candle[0])
        curr_high  = float(candle[2])
        curr_low   = float(candle[3])
        curr_close = float(candle[4])

        # ── Always evaluate primary condition for edge tracking ──────────────────
        # We do this even while a trade is open so that when the trade closes,
        # the edge-detection state is up to date and won't re-fire immediately.
        slice_start   = max(0, i + 1 - 200)
        kslice        = candles[slice_start: i + 1]
        curr_cond_met = eval_condition_bt(primary_cond, kslice, interval_min)

        # ── Check open trade: timeout / TP / SL ─────────────────────────────────
        if open_trade:
            held     = i - open_trade["entry_idx"]
            tp_price = open_trade["tp_price"]
            sl_price = open_trade["sl_price"]

            # Max hold time: force-close at current close after N candles
            if held >= max_hold_c:
                pnl     = _compute_pnl(direction, open_trade["entry_price"], curr_close, leverage)
                outcome = "WIN" if pnl >= 0 else "LOSS"
                trades.append({
                    "entry_ts":     open_trade["entry_ts"],
                    "exit_ts":      curr_ts,
                    "entry_price":  open_trade["entry_price"],
                    "exit_price":   curr_close,
                    "outcome":      outcome,
                    "pnl_pct":      round(pnl, 2),
                    "hold_candles": held,
                    "exit_reason":  "TIMEOUT",
                })
                open_trade    = None
                prev_cond_met = curr_cond_met  # keep state current
                continue

            if direction == "LONG":
                tp_hit = curr_high >= tp_price
                sl_hit = curr_low  <= sl_price
            else:
                tp_hit = curr_low  <= tp_price
                sl_hit = curr_high >= sl_price

            # Same-candle TP+SL: always assume SL hit first (worst-case / conservative)
            if tp_hit and sl_hit:
                outcome, exit_price = "LOSS", sl_price
            elif tp_hit:
                outcome, exit_price = "WIN",  tp_price
            elif sl_hit:
                outcome, exit_price = "LOSS", sl_price
            else:
                outcome = None

            if outcome:
                pnl = _compute_pnl(direction, open_trade["entry_price"], exit_price, leverage)
                trades.append({
                    "entry_ts":     open_trade["entry_ts"],
                    "exit_ts":      curr_ts,
                    "entry_price":  open_trade["entry_price"],
                    "exit_price":   exit_price,
                    "outcome":      outcome,
                    "pnl_pct":      round(pnl, 2),
                    "hold_candles": held,
                    "exit_reason":  "TP" if outcome == "WIN" else "SL",
                })
                open_trade = None

            prev_cond_met = curr_cond_met
            continue  # still managing trade or just closed — don't open another this candle

        # ── Edge detection: only fire on FALSE → TRUE transition ─────────────────
        # This is what your real bot does — a signal fires when the condition
        # is freshly met, not while it stays persistently met (e.g. RSI stuck below 40).
        signal_fires  = curr_cond_met and not prev_cond_met
        prev_cond_met = curr_cond_met

        if not signal_fires:
            continue

        # ── Check confirmations (must be currently met) ──────────────────────────
        all_confirm = all(
            eval_condition_bt(_build_confirm_cond(c), kslice, interval_min)
            for c in confirms
        )
        if not all_confirm:
            continue

        # Enter at NEXT candle's open — signal confirmed at close of candle i,
        # earliest realistic fill is open of candle i+1.
        if i + 1 >= len(candles):
            continue

        next_c    = candles[i + 1]
        entry     = float(next_c[1])
        entry_ts  = int(next_c[0])
        entry_idx = i + 1

        if direction == "LONG":
            tp_price = entry * (1 + tp_pct)
            sl_price = entry * (1 - sl_pct)
        else:
            tp_price = entry * (1 - tp_pct)
            sl_price = entry * (1 + sl_pct)

        open_trade = {
            "entry_ts":    entry_ts,
            "entry_idx":   entry_idx,
            "entry_price": entry,
            "tp_price":    tp_price,
            "sl_price":    sl_price,
        }

    # Close any still-open trade at end of data
    if open_trade and candles:
        last_c     = candles[-1]
        last_close = float(last_c[4])
        held       = len(candles) - 1 - open_trade["entry_idx"]
        pnl        = _compute_pnl(direction, open_trade["entry_price"], last_close, leverage)
        trades.append({
            "entry_ts":     open_trade["entry_ts"],
            "exit_ts":      int(last_c[0]),
            "entry_price":  open_trade["entry_price"],
            "exit_price":   last_close,
            "outcome":      "OPEN",
            "pnl_pct":      round(pnl, 2),
            "hold_candles": held,
            "exit_reason":  "END_OF_DATA",
        })

    stats        = _compute_stats(trades, interval_min)
    equity_curve = _build_equity_curve(trades)

    # Format trade timestamps for display
    display_trades = []
    for t in trades:
        entry_dt = datetime.fromtimestamp(t["entry_ts"] / 1000, tz=timezone.utc)
        exit_dt  = datetime.fromtimestamp(t["exit_ts"]  / 1000, tz=timezone.utc)
        display_trades.append({
            **t,
            "entry_date": entry_dt.strftime("%b %d %H:%M"),
            "exit_date":  exit_dt.strftime("%b %d %H:%M"),
        })

    return {
        "symbol":        source_label,
        "days":          days,
        "interval":      "1h",
        "total_candles": len(candles),
        "trades":        display_trades,
        "stats":         stats,
        "equity_curve":  equity_curve,
        "fees_included": True,
        "fee_pct":       ROUND_TRIP_FEE,
        "max_hold_h":    max_hold_h,
    }
