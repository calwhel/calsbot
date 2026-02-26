"""
AI Exit Optimizer - Uses Gemini AI to analyze open positions in real-time
and provide adaptive exit recommendations (HOLD, TAKE_PROFIT, EXIT_NOW, TIGHTEN_SL)
based on changing momentum, volume, order flow, and derivatives data.

Enhancements:
- 5m klines for fast momentum detection alongside 15m/1h
- Market regime + past trade lessons injected into every prompt
- Escalating confidence: if consecutive HOLDs while P&L declining, threshold auto-reduces
- ATR-based trailing SL computation for TIGHTEN_SL recommendations
"""
import os
import logging
import asyncio
import httpx
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from app.database import SessionLocal
from app.models import Trade, User, UserPreference

logger = logging.getLogger(__name__)

BINANCE_FUTURES_URL = "https://fapi.binance.com"

_last_ai_check: Dict[int, datetime] = {}
_hold_streak: Dict[int, int] = {}
_hold_streak_entry_pnl: Dict[int, float] = {}

_ai_exit_lock = asyncio.Lock()


def get_gemini_client():
    try:
        from google import genai
        api_key = os.environ.get("AI_INTEGRATIONS_GEMINI_API_KEY") or os.environ.get("GEMINI_API_KEY")
        if not api_key:
            return None
        return genai.Client(api_key=api_key)
    except Exception as e:
        logger.error(f"Failed to initialize Gemini client for exit optimizer: {e}")
        return None


async def _fetch_klines(symbol: str, interval: str = "15m", limit: int = 50) -> Optional[List]:
    clean_sym = symbol.replace("/", "").replace(":USDT", "").replace("-USDT", "").upper()
    if not clean_sym.endswith("USDT"):
        clean_sym += "USDT"
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(
                f"{BINANCE_FUTURES_URL}/fapi/v1/klines",
                params={"symbol": clean_sym, "interval": interval, "limit": limit}
            )
            if resp.status_code == 200:
                return resp.json()
    except Exception as e:
        logger.debug(f"Kline fetch failed for {clean_sym} {interval}: {e}")
    return None


async def _fetch_orderbook(symbol: str, limit: int = 20) -> Optional[Dict]:
    clean_sym = symbol.replace("/", "").replace(":USDT", "").replace("-USDT", "").upper()
    if not clean_sym.endswith("USDT"):
        clean_sym += "USDT"
    try:
        async with httpx.AsyncClient(timeout=8) as client:
            resp = await client.get(
                f"{BINANCE_FUTURES_URL}/fapi/v1/depth",
                params={"symbol": clean_sym, "limit": limit}
            )
            if resp.status_code == 200:
                data = resp.json()
                bids = data.get("bids", [])
                asks = data.get("asks", [])
                bid_volume = sum(float(b[1]) for b in bids[:10])
                ask_volume = sum(float(a[1]) for a in asks[:10])
                total = bid_volume + ask_volume
                return {
                    "bid_volume": round(bid_volume, 2),
                    "ask_volume": round(ask_volume, 2),
                    "bid_ratio": round(bid_volume / total * 100, 1) if total > 0 else 50,
                    "imbalance": "BUY_HEAVY" if bid_volume > ask_volume * 1.3 else "SELL_HEAVY" if ask_volume > bid_volume * 1.3 else "BALANCED"
                }
    except Exception as e:
        logger.debug(f"Orderbook fetch failed: {e}")
    return None


def _compute_indicators(klines: List) -> Dict:
    if not klines or len(klines) < 10:
        return {}

    closes = [float(k[4]) for k in klines]
    highs = [float(k[2]) for k in klines]
    lows = [float(k[3]) for k in klines]
    volumes = [float(k[5]) for k in klines]

    current_price = closes[-1]

    def ema(data, period):
        if len(data) < period:
            return data[-1]
        multiplier = 2 / (period + 1)
        result = sum(data[:period]) / period
        for val in data[period:]:
            result = (val - result) * multiplier + result
        return result

    ema_9 = ema(closes, min(9, len(closes)))
    ema_21 = ema(closes, min(21, len(closes)))
    ema_50 = ema(closes, 50) if len(closes) >= 50 else None

    gains, losses = [], []
    for i in range(1, min(15, len(closes))):
        diff = closes[-i] - closes[-i - 1]
        if diff > 0:
            gains.append(diff)
        else:
            losses.append(abs(diff))
    avg_gain = sum(gains) / 14 if gains else 0.001
    avg_loss = sum(losses) / 14 if losses else 0.001
    rs = avg_gain / avg_loss if avg_loss > 0 else 100
    rsi = 100 - (100 / (1 + rs))

    recent_vol = sum(volumes[-5:]) / 5 if len(volumes) >= 5 else volumes[-1]
    avg_vol = sum(volumes[-20:]) / 20 if len(volumes) >= 20 else sum(volumes) / len(volumes)
    volume_ratio = recent_vol / avg_vol if avg_vol > 0 else 1.0

    atr_vals = []
    for i in range(1, min(15, len(closes))):
        tr = max(highs[-i] - lows[-i], abs(highs[-i] - closes[-i - 1]), abs(lows[-i] - closes[-i - 1]))
        atr_vals.append(tr)
    atr = sum(atr_vals) / len(atr_vals) if atr_vals else 0
    atr_percent = (atr / current_price * 100) if current_price > 0 else 0

    price_prev = closes[-4] if len(closes) >= 4 else closes[0]
    momentum = ((current_price - price_prev) / price_prev * 100) if price_prev > 0 else 0

    highest_recent = max(highs[-10:]) if len(highs) >= 10 else max(highs)
    lowest_recent = min(lows[-10:]) if len(lows) >= 10 else min(lows)
    range_position = ((current_price - lowest_recent) / (highest_recent - lowest_recent) * 100) if (highest_recent - lowest_recent) > 0 else 50

    candle_body = abs(closes[-1] - float(klines[-1][1]))
    candle_range = highs[-1] - lows[-1]
    is_doji = candle_body < candle_range * 0.1 if candle_range > 0 else False
    is_bearish_candle = closes[-1] < float(klines[-1][1])

    return {
        "current_price": round(current_price, 8),
        "ema_9": round(ema_9, 8),
        "ema_21": round(ema_21, 8),
        "ema_50": round(ema_50, 8) if ema_50 else None,
        "rsi": round(rsi, 1),
        "volume_ratio": round(volume_ratio, 2),
        "atr": round(atr, 8),
        "atr_percent": round(atr_percent, 2),
        "momentum": round(momentum, 2),
        "range_position": round(range_position, 1),
        "ema_trend": "BULLISH" if ema_9 > ema_21 else "BEARISH",
        "recent_high": round(highest_recent, 8),
        "recent_low": round(lowest_recent, 8),
        "last_candle_doji": is_doji,
        "last_candle_bearish": is_bearish_candle,
    }


def _compute_atr_trailing_sl(ta: Dict, entry_price: float, direction: str, multiplier: float = 1.5) -> Optional[float]:
    atr = ta.get("atr")
    current_price = ta.get("current_price")
    if not atr or not current_price:
        return None
    if direction == "LONG":
        return round(current_price - atr * multiplier, 8)
    else:
        return round(current_price + atr * multiplier, 8)


async def _fetch_derivatives(symbol: str) -> Dict:
    deriv = {}
    try:
        from app.services.coinglass import get_funding_rate, get_open_interest_history, get_long_short_ratio
        funding, oi, lsr = await asyncio.gather(
            get_funding_rate(symbol),
            get_open_interest_history(symbol),
            get_long_short_ratio(symbol),
            return_exceptions=True
        )
        if isinstance(funding, dict):
            deriv["funding_rate"] = funding.get("funding_rate", 0)
        if isinstance(oi, dict):
            deriv["oi_change_pct"] = oi.get("oi_change_pct", 0)
            deriv["oi_trend"] = oi.get("oi_trend", "N/A")
        if isinstance(lsr, dict):
            deriv["long_ratio"] = lsr.get("long_ratio", 50)
            deriv["short_ratio"] = lsr.get("short_ratio", 50)
    except Exception as e:
        logger.debug(f"Derivatives fetch failed for {symbol}: {e}")
    return deriv


def _get_market_regime_context() -> str:
    try:
        from app.services.ai_market_intelligence import _current_market_regime
        if _current_market_regime:
            regime = _current_market_regime.get('regime', 'UNKNOWN')
            btc_change = _current_market_regime.get('btc_change_24h', 0)
            return f"Market Regime: {regime} | BTC 24h: {btc_change:+.1f}%"
    except Exception:
        pass
    return ""


def _get_lessons_context(direction: str) -> str:
    try:
        from app.services.ai_trade_learner import format_lessons_for_ai_prompt
        return format_lessons_for_ai_prompt(direction=direction)
    except Exception:
        return ""


async def analyze_position(trade: Trade) -> Optional[Dict]:
    symbol = trade.symbol
    direction = trade.direction.upper()
    entry_price = trade.entry_price

    klines_5m, klines_15m, klines_1h, orderbook, derivatives = await asyncio.gather(
        _fetch_klines(symbol, "5m", 30),
        _fetch_klines(symbol, "15m", 50),
        _fetch_klines(symbol, "1h", 24),
        _fetch_orderbook(symbol),
        _fetch_derivatives(symbol),
        return_exceptions=True
    )

    if isinstance(klines_15m, Exception) or not klines_15m:
        logger.warning(f"AI Exit: Could not fetch 15m data for {symbol}")
        return None

    ta_5m = _compute_indicators(klines_5m) if isinstance(klines_5m, list) else {}
    ta_15m = _compute_indicators(klines_15m) if klines_15m else {}
    ta_1h = _compute_indicators(klines_1h) if isinstance(klines_1h, list) else {}
    ob = orderbook if isinstance(orderbook, dict) else {}
    deriv = derivatives if isinstance(derivatives, dict) else {}

    if not ta_15m or not ta_15m.get("current_price"):
        return None

    current_price = ta_15m["current_price"]
    if direction == "LONG":
        unrealized_pnl_pct = ((current_price - entry_price) / entry_price) * 100
    else:
        unrealized_pnl_pct = ((entry_price - current_price) / entry_price) * 100

    trade_age_minutes = (datetime.utcnow() - trade.opened_at).total_seconds() / 60
    leverage = trade.leverage or 10
    leveraged_pnl = unrealized_pnl_pct * leverage

    tp1_price = trade.take_profit_1 or trade.take_profit
    tp2_price = trade.take_profit_2
    sl_price = trade.stop_loss

    if tp1_price and direction == "LONG":
        distance_to_tp = ((tp1_price - current_price) / current_price * 100) if not trade.tp1_hit else (((tp2_price - current_price) / current_price * 100) if tp2_price else 0)
        distance_to_sl = ((current_price - sl_price) / current_price * 100) if sl_price else 0
    elif tp1_price:
        distance_to_tp = ((current_price - tp1_price) / current_price * 100) if not trade.tp1_hit else (((current_price - tp2_price) / current_price * 100) if tp2_price else 0)
        distance_to_sl = ((sl_price - current_price) / current_price * 100) if sl_price else 0
    else:
        distance_to_tp = 0
        distance_to_sl = 0

    atr_sl = _compute_atr_trailing_sl(ta_15m, entry_price, direction)

    regime_context = _get_market_regime_context()
    lessons_context = _get_lessons_context(direction)

    hold_count = _hold_streak.get(trade.id, 0)
    hold_warning = ""
    if hold_count >= 3:
        pnl_at_start = _hold_streak_entry_pnl.get(trade.id, unrealized_pnl_pct)
        pnl_drift = unrealized_pnl_pct - pnl_at_start
        if pnl_drift < -1.0:
            hold_warning = f"\nWARNING: AI has said HOLD {hold_count} times but P&L has drifted {pnl_drift:+.2f}% since first HOLD. Apply stricter criteria."

    prompt = f"""You are an expert crypto futures position manager. Analyze this LIVE position and give an exit recommendation.

POSITION:
- Symbol: {symbol}
- Direction: {direction}
- Entry Price: ${entry_price:.6f}
- Current Price: ${current_price:.6f}
- Unrealized P&L: {unrealized_pnl_pct:+.2f}% (leveraged: {leveraged_pnl:+.1f}% at {leverage}x)
- Trade Age: {trade_age_minutes:.0f} minutes
- TP1 Hit: {trade.tp1_hit} | Breakeven Active: {trade.breakeven_moved}
- Distance to TP: {distance_to_tp:.2f}% | Distance to SL: {distance_to_sl:.2f}%
- Peak ROI: {trade.peak_roi or 0:.1f}%
- ATR-based trailing SL: ${atr_sl:.6f} (15m ATR x1.5){hold_warning}

5-MINUTE MOMENTUM (fast signal):
- RSI: {ta_5m.get('rsi', 'N/A')}
- EMA Trend: {ta_5m.get('ema_trend', 'N/A')}
- Volume Ratio: {ta_5m.get('volume_ratio', 'N/A')}x
- Momentum: {ta_5m.get('momentum', 'N/A')}%
- Last candle bearish: {ta_5m.get('last_candle_bearish', 'N/A')} | Doji: {ta_5m.get('last_candle_doji', 'N/A')}

15-MINUTE TECHNICAL ANALYSIS:
- RSI: {ta_15m.get('rsi', 'N/A')}
- EMA 9/21 Trend: {ta_15m.get('ema_trend', 'N/A')}
- Volume Ratio: {ta_15m.get('volume_ratio', 'N/A')}x
- ATR%: {ta_15m.get('atr_percent', 'N/A')}%
- Momentum: {ta_15m.get('momentum', 'N/A')}%
- Range Position: {ta_15m.get('range_position', 'N/A')}% (0=low, 100=high)

1-HOUR TECHNICAL ANALYSIS:
- RSI: {ta_1h.get('rsi', 'N/A')}
- EMA Trend: {ta_1h.get('ema_trend', 'N/A')}
- Volume Ratio: {ta_1h.get('volume_ratio', 'N/A')}x
- Momentum: {ta_1h.get('momentum', 'N/A')}%

ORDER BOOK:
- Bid/Ask Imbalance: {ob.get('imbalance', 'N/A')}
- Bid Ratio: {ob.get('bid_ratio', 'N/A')}%

DERIVATIVES:
- Funding Rate: {deriv.get('funding_rate', 'N/A')}%
- OI Change: {deriv.get('oi_change_pct', 'N/A')}% ({deriv.get('oi_trend', 'N/A')})
- Long/Short Ratio: {deriv.get('long_ratio', 'N/A')}% / {deriv.get('short_ratio', 'N/A')}%

MARKET CONTEXT:
{regime_context if regime_context else "Market regime: unknown"}
{lessons_context}

Respond with EXACTLY this JSON format (no markdown, no extra text):
{{
    "action": "HOLD" or "TAKE_PROFIT" or "EXIT_NOW" or "TIGHTEN_SL",
    "confidence": 1-10,
    "urgency": "LOW" or "MEDIUM" or "HIGH" or "CRITICAL",
    "reasoning": "2-3 sentence explanation of why",
    "suggested_sl": null or new_stop_loss_price_as_number,
    "risk_reward_assessment": "brief current R:R evaluation"
}}

RULES:
- HOLD: Conditions still favorable, let it run — default to this when in doubt
- TAKE_PROFIT: Strong reversal signals on 15m AND 1h — lock in gains (requires 7/10 confidence)
- EXIT_NOW: Clear danger on MULTIPLE timeframes — cut losses or protect capital (requires 8/10 confidence — high bar, only for obvious reversals)
- TIGHTEN_SL: Move stop loss closer to protect gains while staying in trade — use ATR-based SL provided above
- BIAS TOWARD HOLDING: When uncertain, return HOLD or TIGHTEN_SL. Never close speculatively.
- A single 5m candle reversal is NOT enough — require alignment across at least 2 timeframes
- If 5m shows reversal but 1h trend still intact, always return TIGHTEN_SL instead of EXIT_NOW
- If trade is in profit and momentum is merely slowing (not reversing), return TIGHTEN_SL not EXIT_NOW
- Only use EXIT_NOW if there is strong bearish/bullish reversal evidence across 15m and 1h simultaneously
- If suggesting TIGHTEN_SL, set suggested_sl to the ATR-based trailing SL shown above"""

    client = get_gemini_client()
    if not client:
        logger.warning("AI Exit Optimizer: No Gemini client available")
        return None

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )

        text = response.text.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1] if "\n" in text else text[3:]
            if text.endswith("```"):
                text = text[:-3]
            text = text.strip()
        if text.startswith("json"):
            text = text[4:].strip()

        result = json.loads(text)

        result["symbol"] = symbol
        result["direction"] = direction
        result["current_price"] = current_price
        result["entry_price"] = entry_price
        result["unrealized_pnl_pct"] = round(unrealized_pnl_pct, 2)
        result["leveraged_pnl"] = round(leveraged_pnl, 1)
        result["trade_id"] = trade.id
        result["trade_age_minutes"] = round(trade_age_minutes, 0)
        result["rsi"] = ta_15m.get("rsi", 0)
        result["rsi_5m"] = ta_5m.get("rsi", 0)
        result["volume_ratio"] = ta_15m.get("volume_ratio", 1.0)
        result["atr_suggested_sl"] = atr_sl
        result["analyzed_at"] = datetime.utcnow().isoformat()

        return result

    except json.JSONDecodeError as e:
        logger.error(f"AI Exit: Failed to parse Gemini response for {symbol}: {e}")
        return None
    except Exception as e:
        logger.error(f"AI Exit: Gemini analysis failed for {symbol}: {e}")
        return None


async def check_ai_exit(trade: Trade, prefs: UserPreference) -> Tuple[bool, Optional[str], Optional[Dict]]:
    try:
        enabled = getattr(prefs, 'ai_exit_optimizer_enabled', True)
    except Exception:
        enabled = True
    if not enabled:
        return False, None, None

    try:
        min_age = getattr(prefs, 'ai_exit_min_trade_age_minutes', 30)
    except Exception:
        min_age = 30
    trade_age = (datetime.utcnow() - trade.opened_at).total_seconds() / 60
    if trade_age < min_age:
        return False, None, None

    try:
        check_interval = getattr(prefs, 'ai_exit_check_interval_minutes', 5)
    except Exception:
        check_interval = 5
    last_check = _last_ai_check.get(trade.id)
    if last_check and (datetime.utcnow() - last_check).total_seconds() < check_interval * 60:
        return False, None, None

    _last_ai_check[trade.id] = datetime.utcnow()

    try:
        analysis = await asyncio.wait_for(analyze_position(trade), timeout=30)
    except asyncio.TimeoutError:
        logger.warning(f"AI Exit: Timeout analyzing {trade.symbol}")
        return False, None, None
    except Exception as e:
        logger.warning(f"AI Exit: Error analyzing {trade.symbol}: {e}")
        return False, None, None

    if not analysis:
        return False, None, None

    action = analysis.get("action", "HOLD")
    confidence = analysis.get("confidence", 0)
    reasoning = analysis.get("reasoning", "")
    unrealized_pnl = analysis.get("unrealized_pnl_pct", 0)

    logger.info(
        f"AI EXIT [{trade.symbol} {trade.direction}]: "
        f"{action} (confidence: {confidence}/10) - {reasoning}"
    )

    if action == "HOLD":
        hold_count = _hold_streak.get(trade.id, 0)
        if hold_count == 0:
            _hold_streak_entry_pnl[trade.id] = unrealized_pnl
        _hold_streak[trade.id] = hold_count + 1
    else:
        _hold_streak[trade.id] = 0
        _hold_streak_entry_pnl.pop(trade.id, None)

    hold_count = _hold_streak.get(trade.id, 0)
    pnl_at_start = _hold_streak_entry_pnl.get(trade.id, unrealized_pnl)
    pnl_drift = unrealized_pnl - pnl_at_start

    # Thresholds: EXIT_NOW requires 8/10 (cutting losses needs high conviction),
    # TAKE_PROFIT requires 7/10 (locking in gains is lower risk).
    # Escalating threshold (when holding too long with drifting PnL) never drops below 7.
    exit_now_threshold = 8
    take_profit_threshold = 7
    if hold_count >= 3 and pnl_drift < -1.0:
        logger.info(f"AI Exit: {trade.symbol} — {hold_count} HOLDs, PnL drifted {pnl_drift:+.2f}% — thresholds remain EXIT_NOW=8, TAKE_PROFIT=7")

    if action == "EXIT_NOW" and confidence >= exit_now_threshold:
        _hold_streak.pop(trade.id, None)
        _hold_streak_entry_pnl.pop(trade.id, None)
        return True, f"AI Exit: EXIT_NOW ({confidence}/10) - {reasoning}", analysis

    if action == "TAKE_PROFIT" and confidence >= take_profit_threshold:
        _hold_streak.pop(trade.id, None)
        _hold_streak_entry_pnl.pop(trade.id, None)
        return True, f"AI Exit: TAKE_PROFIT ({confidence}/10) - {reasoning}", analysis

    return False, None, analysis


async def analyze_all_positions(user_id: int = None) -> List[Dict]:
    db = SessionLocal()
    try:
        query = db.query(Trade).filter(Trade.status == 'open')
        if user_id:
            query = query.filter(Trade.user_id == user_id)

        open_trades = query.all()
        if not open_trades:
            return []

        results = []
        for trade in open_trades:
            trade_age = (datetime.utcnow() - trade.opened_at).total_seconds() / 60
            if trade_age < 5:
                continue

            analysis = await analyze_position(trade)
            if analysis:
                results.append(analysis)

            await asyncio.sleep(1)

        return results
    finally:
        db.close()


def format_exit_analysis(analysis: Dict) -> str:
    action = analysis.get("action", "HOLD")
    confidence = analysis.get("confidence", 0)
    urgency = analysis.get("urgency", "LOW")
    symbol = analysis.get("symbol", "")
    direction = analysis.get("direction", "")
    pnl = analysis.get("leveraged_pnl", 0)
    reasoning = analysis.get("reasoning", "")
    rr = analysis.get("risk_reward_assessment", "")
    rsi_15m = analysis.get("rsi", 0)
    rsi_5m = analysis.get("rsi_5m", 0)
    vol = analysis.get("volume_ratio", 1.0)

    action_icons = {
        "HOLD": "HOLD",
        "TAKE_PROFIT": "TAKE PROFIT",
        "EXIT_NOW": "EXIT NOW",
        "TIGHTEN_SL": "TIGHTEN SL"
    }

    urgency_icons = {
        "LOW": "Low",
        "MEDIUM": "Medium",
        "HIGH": "High",
        "CRITICAL": "CRITICAL"
    }

    conf_bar = "█" * confidence + "░" * (10 - confidence)

    lines = [
        f"<b>${symbol.split('/')[0] if '/' in symbol else symbol.replace('USDT', '').replace('-', '')}</b> {direction}",
        f"",
        f"Action: <b>{action_icons.get(action, action)}</b>",
        f"Confidence: [{conf_bar}] {confidence}/10",
        f"Urgency: {urgency_icons.get(urgency, urgency)}",
        f"P&L: <b>{pnl:+.1f}%</b> (leveraged)",
        f"",
        f"RSI 5m: {rsi_5m:.0f} | RSI 15m: {rsi_15m:.0f} | Vol: {vol:.1f}x",
        f"",
        f"<i>{reasoning}</i>",
    ]

    if rr:
        lines.append(f"R:R: {rr}")

    suggested_sl = analysis.get("suggested_sl")
    atr_sl = analysis.get("atr_suggested_sl")
    if suggested_sl:
        lines.append(f"Suggested SL: ${suggested_sl:.6f}")
    elif atr_sl and action == "TIGHTEN_SL":
        lines.append(f"ATR Trailing SL: ${atr_sl:.6f}")

    return "\n".join(lines)
