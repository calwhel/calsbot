"""
Social & News Signals Trading Mode - AI-powered trading
Completely separate from Top Gainers mode
"""
import asyncio
import json
import logging
import os
import time
import httpx
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from sqlalchemy.orm import Session

def interpret_signal_score(score):
    return "N/A"

def calculate_signal_strength(signal):
    """
    Weighted composite signal strength score (1-10).
      55% — AI confidence (primary quality gate, already validated ≥6)
      15% — Volume conviction (ratio vs average, saturates at 3×)
      10% — RSI quality (mid-range RSI rewarded, exhaustion penalised)
      10% — Risk/reward ratio (TP% / SL%)
      10% — Order flow alignment (directional flow score vs trade direction)
    """
    direction  = signal.get('direction', 'LONG')
    ai_conf    = signal.get('ai_confidence', 5)
    vol_ratio  = signal.get('volume_ratio', 1.0)
    rsi        = signal.get('rsi', 50)
    tp_pct     = signal.get('tp_percent', 2.0) or 2.0
    sl_pct     = signal.get('sl_percent', 2.0) or 2.0

    # 1. AI confidence (55%) — already 1-10
    ai_score = float(ai_conf)

    # 2. Volume conviction (15%) — saturates at 3× surge
    vol_score = min(vol_ratio / 3.0, 1.0) * 10.0

    # 3. RSI quality (10%) — reward mid-range, penalise extremes
    if direction == 'LONG':
        if   40 <= rsi <= 60: rsi_score = 10.0
        elif 35 <= rsi <= 70: rsi_score = 7.0
        elif 30 <= rsi <= 75: rsi_score = 5.0
        else:                  rsi_score = 2.0
    else:
        if   40 <= rsi <= 60: rsi_score = 10.0
        elif 30 <= rsi <= 65: rsi_score = 7.0
        elif 25 <= rsi <= 70: rsi_score = 5.0
        else:                  rsi_score = 2.0

    # 4. Risk/reward quality (10%)
    rr = tp_pct / sl_pct if sl_pct > 0 else 1.0
    if   rr >= 2.0: rr_score = 10.0
    elif rr >= 1.5: rr_score = 8.0
    elif rr >= 1.0: rr_score = 6.0
    else:           rr_score = 3.0

    # 5. Order flow alignment (10%) — neutral (6) when data absent
    flow_score = 6.0
    order_flow = signal.get('order_flow') or {}
    if order_flow:
        fv = order_flow.get('flow_score', 0)   # -100 to +100
        if direction == 'LONG':
            flow_score = min(10.0, max(1.0, 5.0 + fv / 20.0))
        else:
            flow_score = min(10.0, max(1.0, 5.0 - fv / 20.0))

    raw = (
        0.55 * ai_score  +
        0.15 * vol_score +
        0.10 * rsi_score +
        0.10 * rr_score  +
        0.10 * flow_score
    )
    score = max(1, min(10, round(raw)))
    return {
        'score':        score,
        'ai_confidence': ai_conf,
        'volume_ratio':  vol_ratio,
        'rsi_score':     rsi_score,
        'rr_score':      rr_score,
        'flow_score':    flow_score,
        'raw':           round(raw, 2),
    }


def format_signal_strength_detail(strength):
    if not strength:
        return ""
    score = strength.get('score', 5)
    bars  = '█' * score + '░' * (10 - score)
    ai    = strength.get('ai_confidence', '?')
    rr    = strength.get('rr_score', '?')
    return f"[{bars}] {score}/10  (AI:{ai} R/R:{rr:.0f} raw:{strength.get('raw','?')})"

def format_derivatives_for_ai(deriv_data):
    return ""

def format_derivatives_for_message(deriv_data):
    return ""
from app.services.top_coins import is_top_coin_sync, refresh_top_coins
from app.services.ai_signal_filter import get_btc_state

logger = logging.getLogger(__name__)

MAJOR_COINS = {'BTC', 'ETH', 'SOL', 'BNB', 'XRP', 'ADA', 'DOGE', 'AVAX', 'DOT', 'LINK'}


async def ai_analyze_social_signal(signal_data: Dict) -> Dict:
    """
    Use Gemini for fast scan + Claude for final approval of social signals.
    Returns dict with 'approved', 'reasoning', 'ai_confidence', 'recommendation'.
    """
    symbol = signal_data['symbol']
    direction = signal_data.get('direction', 'LONG')
    
    deriv_data = signal_data.get('derivatives', {})
    deriv_summary = format_derivatives_for_ai(deriv_data) if deriv_data else ""
    
    deriv_adj = signal_data.get('deriv_adjustments', [])
    adj_note = ""
    if deriv_adj:
        adj_note = "\nTP/SL Adjusted by Derivatives:\n" + "\n".join(f"  • {a}" for a in deriv_adj)
    
    is_news = signal_data.get('trade_type') == 'NEWS_SIGNAL'
    is_relief = signal_data.get('trade_type') == 'RELIEF_BOUNCE'
    news_context = signal_data.get('news_context', '')
    
    volume_raw = signal_data.get('24h_volume', 0)
    vol_str = f"${volume_raw/1e6:.1f}M" if volume_raw >= 1e6 else f"${volume_raw/1e3:.0f}K"
    
    data_summary = (
        f"Coin: {symbol}\n"
        f"Direction: {direction}\n"
        f"Entry: ${signal_data['entry_price']}\n"
    )
    
    if is_relief:
        bounce_pct = signal_data.get('bounce_from_low', 0)
        drop_pct = signal_data.get('drop_from_high', 0)
        data_summary += (
            f"Signal Type: RELIEF BOUNCE (contrarian reversal play)\n"
            f"24h Dump: {signal_data.get('24h_change', 0):.1f}%\n"
            f"Bounce from Low: +{bounce_pct:.1f}%\n"
            f"Drop from High: -{drop_pct:.1f}%\n"
        )
    elif is_news and news_context:
        data_summary += f"Signal Type: NEWS-DRIVEN TRADE\n{news_context}\n"
    else:
        vol_ratio = signal_data.get('volume_ratio', 1.0)
        data_summary += f"Volume Ratio (vs avg): {vol_ratio:.1f}x\n"
    
    btc_corr = signal_data.get('btc_correlation', 0)
    vol_ratio = signal_data.get('volume_ratio', 1.0)
    
    data_summary += (
        f"24h Change: {signal_data.get('24h_change', 0):+.1f}%\n"
        f"24h Volume: {vol_str}\n"
        f"BTC Correlation: {btc_corr:.0%}\n"
        f"TP: +{signal_data['tp_percent']:.1f}%\n"
        f"SL: -{signal_data['sl_percent']:.1f}%"
        f"{adj_note}"
    )

    enhanced_ta = signal_data.get('enhanced_ta', {})
    if enhanced_ta:
        from app.services.enhanced_ta import format_ta_for_ai
        ta_section = format_ta_for_ai(enhanced_ta)
        if ta_section:
            data_summary += f"\n\n--- TECHNICAL ANALYSIS ---\n{ta_section}"
    else:
        data_summary += f"\nRSI (15m): {signal_data.get('rsi', 50):.0f}"
    
    
    if deriv_summary:
        data_summary += f"\n\n{deriv_summary}"
    
    order_flow_data = signal_data.get('order_flow')
    if order_flow_data:
        from app.services.order_flow import format_order_flow_for_ai
        flow_section = format_order_flow_for_ai(order_flow_data)
        if flow_section:
            data_summary += f"\n{flow_section}"

    lessons_context = ""
    try:
        from app.services.ai_trade_learner import format_lessons_for_ai_prompt
        trade_type = signal_data.get('trade_type', 'SOCIAL_SIGNAL')
        lessons_context = format_lessons_for_ai_prompt(trade_type=trade_type, direction=direction)
    except Exception as le:
        logger.debug(f"Lessons context failed: {le}")

    # Fetch Grok macro context (cached, refreshed every 20 min via live web+X search)
    macro_section = ""
    try:
        from app.services.ai_signal_filter import get_cached_grok_macro
        grok_macro = await get_cached_grok_macro()
        if grok_macro and grok_macro.get('bias'):
            macro_section = (
                f"\n--- GROK MACRO CONTEXT (live web+X briefing) ---\n"
                f"Global Bias: {grok_macro['bias']}\n"
                f"{grok_macro.get('summary', '')}\n"
                f"RULE: BEARISH macro = extra skepticism on LONGs. BULLISH macro = extra skepticism on SHORTs. NEUTRAL = trade technicals.\n"
            )
    except Exception as me:
        logger.debug(f"Macro context fetch failed: {me}")

    # STEP 1: Gemini fast scan
    gemini_reasoning = None
    _gemini_fallback = None  # stored so Grok failure can fall back to Gemini's verdict
    try:
        from app.services.ai_market_intelligence import get_gemini_client
        gemini = get_gemini_client()
        if gemini:
            if is_relief:
                signal_type_desc = "a RELIEF BOUNCE reversal signal on a coin that dumped hard and is now bouncing"
            elif is_news:
                signal_type_desc = "a NEWS-DRIVEN trading signal based on breaking crypto news"
            else:
                signal_type_desc = "a top-gainer momentum signal on a coin already moving with elevated volume"
            if is_relief:
                trade_instruction = """3. RELIEF BOUNCE ANALYSIS (critical):
   - Has the coin dumped enough (-20%+) to create a genuine reversal opportunity?
   - Is the bounce from the low meaningful (2%+) or just noise?
   - Is RSI oversold enough to suggest a reversal is likely?
   - Is this a legitimate project or a rug pull / dead coin?
   - Are derivatives (funding rate, open interest) supportive of a bounce?
   - Is the TP realistic given the bounce momentum, and is the SL tight enough for a risky reversal play?"""
            elif is_news:
                trade_instruction = """3. NEWS HEADLINE ANALYSIS (critical — headline sentiment overrides macro):
   - Read the exact headline carefully. If it contains cautionary language ("not the time", "wait", "but warns", "caution"), REJECT regardless of macro context.
   - Does the headline describe a concrete event (listing, partnership, ETF approval, hack) or just an opinion/prediction?
   - Concrete events = higher confidence. Analyst opinion/price target alone = lower confidence.
   - Is the impact immediate (happening now) or speculative (months away)? Speculative news should be rejected."""
            else:
                trade_instruction = """3. MOMENTUM ANALYSIS (critical):
   - Is volume elevated (volume ratio vs average)? High volume = conviction behind the move.
   - Is the BTC correlation low enough that this coin can move independently?
   - Do technicals (RSI, EMA, VWAP) confirm continuation rather than exhaustion?
   - Is the coin still near its session high, or has it already faded significantly?"""

            prompt = f"""You are an aggressive crypto perps scalp trader. Your job is to FIND TRADES, not avoid them. You make money by taking quick positions with tight stops.

{data_summary}
{macro_section}
{lessons_context}

Analyze this {direction} signal critically:
- For LONGS: REJECT only if RSI >80 (extreme overbought) or 24h change >30% (parabolic blowoff). Momentum coins with RSI 65-75 and 10-25% moves are NORMAL for this strategy.
- For SHORTS: REJECT only if RSI <20 (extreme oversold) or 24h change <-30% (already dumped hard).
- APPROVE if volume is elevated, technicals confirm continuation, and the move has NOT exhausted.
- The goal is to catch CONTINUING moves, not just early ones. A coin up 15% with strong volume and bullish technicals is a valid entry.

{trade_instruction}

NOTE: These coins are selected because they are ALREADY moving. Do not penalise a signal purely for having positive 24h change. Focus on whether the momentum has more to go, not whether it has started.
If past trade lessons are provided above, use them to avoid repeating losing patterns.

Respond in JSON:
{{
    "scan_pass": true/false,
    "reasoning": "2-3 sentence sharp analysis. Focus on the OPPORTUNITY not the risks.",
    "confidence": 1-10,
    "key_risk": "one sentence main risk"
}}"""
            
            def _gemini_call():
                return gemini.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=prompt,
                    config={"temperature": 0.3, "max_output_tokens": 600}
                )
            
            response = await asyncio.get_event_loop().run_in_executor(None, _gemini_call)
            result_text = (response.text or "").strip()
            if not result_text:
                raise ValueError("Gemini returned empty response (safety filter or content block)")
            
            import re
            if "```json" in result_text:
                match = re.search(r'```json\s*(.*?)\s*```', result_text, re.DOTALL)
                if match:
                    result_text = match.group(1)
            
            first_brace = result_text.find("{")
            last_brace = result_text.rfind("}")
            gemini_result = None
            if first_brace >= 0 and last_brace > first_brace:
                try:
                    gemini_result = json.loads(result_text[first_brace:last_brace + 1])
                except json.JSONDecodeError:
                    pass

            if gemini_result is None:
                # Truncated or malformed — extract key fields directly via regex
                sp_match = re.search(r'"scan_pass"\s*:\s*(true|false)', result_text, re.IGNORECASE)
                conf_match = re.search(r'"confidence"\s*:\s*(\d+)', result_text)
                reasoning_match = re.search(r'"reasoning"\s*:\s*"([^"]+)"', result_text)
                risk_match = re.search(r'"key_risk"\s*:\s*"([^"]+)"', result_text)
                if sp_match:
                    logger.debug(f"Gemini truncated response for {symbol} — extracted scan_pass via regex")
                    gemini_result = {
                        'scan_pass': sp_match.group(1).lower() == 'true',
                        'confidence': int(conf_match.group(1)) if conf_match else 5,
                        'reasoning': reasoning_match.group(1) if reasoning_match else 'Partial Gemini response',
                        'key_risk': risk_match.group(1) if risk_match else '',
                    }
                else:
                    raise ValueError(f"Gemini response contained no JSON object: {result_text[:80]!r}")
            gemini_reasoning = gemini_result.get('reasoning', '')
            
            if not gemini_result.get('scan_pass', True):
                logger.info(f"🤖 Gemini flagged {symbol} as risky: {gemini_reasoning} — passing to Grok-4 for final call")
            else:
                logger.info(f"🤖 Gemini PASSED {symbol}: confidence {gemini_result.get('confidence', 5)}")
            rec_fallback = 'SELL' if direction == 'SHORT' else 'BUY'
            _gemini_fallback = {
                'approved': True,
                'reasoning': gemini_reasoning or '',
                'ai_confidence': gemini_result.get('confidence', 5),
                'recommendation': rec_fallback,
                'entry_quality': 'FAIR',
                'trade_explainer': '',
                'key_risk': gemini_result.get('key_risk', '')
            }
    except Exception as e:
        logger.warning(f"Gemini analysis failed for {symbol}: {e}")

    # STEP 2: Grok-4 final approval
    try:
        xai_key = os.environ.get("XAI_API_KEY")
        if xai_key:
            from openai import AsyncOpenAI
            grok_client = AsyncOpenAI(api_key=xai_key, base_url="https://api.x.ai/v1")

            signal_desc = "news-driven trading signal" if is_news else "top-gainer momentum signal"
            news_extra = "\n- News catalyst assessment: Is this significant enough to move price?" if is_news else ""
            rec_options = '"STRONG SELL" or "SELL" or "HOLD" or "AVOID"' if direction == 'SHORT' else '"STRONG BUY" or "BUY" or "HOLD" or "AVOID"'
            gemini_context = f"\nGemini Initial Scan: {gemini_reasoning}" if gemini_reasoning else ""

            grok_prompt = f"""You are an aggressive crypto perpetual futures scalp trader reviewing a {signal_desc}. You WANT to take trades. Tight stop losses protect your downside.

{data_summary}
{macro_section}
{gemini_context}
{lessons_context}

TRADING RULES:
- This strategy trades TOP GAINERS — coins with elevated volume and continuing price momentum. They will often have RSI 65-75 and 10-25% 24h moves. That is expected and normal.
- For LONGS: REJECT only if RSI >80 (extreme blowoff) or 24h change >30% (parabolic exhaustion). A 15% move with high volume is NOT a reason to reject.
- For SHORTS: REJECT only if RSI <20 or 24h change <-30% (already capitulated).
- APPROVE if: momentum indicators confirm continuation, volume is elevated, and the setup has a logical TP target within reach.
- REJECT if: volume is clearly declining, derivatives strongly oppose direction, or the move looks exhausted (RSI divergence, price fading from high).
- Derivatives data is supplementary context. Extreme funding (>0.05%) against your direction = cautious.{news_extra}
- CRITICAL: This is a {direction} signal. Your recommendation MUST match the direction.
  For SHORT signals use STRONG SELL/SELL. For LONG signals use STRONG BUY/BUY. Never say BUY on a SHORT.

Respond in JSON only:
{{
    "approved": true/false,
    "confidence": 1-10,
    "recommendation": {rec_options},
    "reasoning": "2-3 sentence concise analysis. Focus on opportunity. Be direct and actionable.",
    "entry_quality": "EXCELLENT" or "GOOD" or "FAIR" or "POOR",
    "trade_explainer": "2-3 punchy sentences selling this trade. Cover: (1) what the setup is in plain English, (2) why RIGHT NOW is the timing, (3) what the key risk is. Write for someone who does not read charts. Make it compelling and direct — no filler words."
}}"""

            grok_resp = None
            for _attempt in range(2):
                try:
                    grok_resp = await asyncio.wait_for(
                        grok_client.chat.completions.create(
                            model="grok-4",
                            messages=[{"role": "user", "content": grok_prompt}],
                            max_tokens=500,
                            temperature=0.2,
                        ),
                        timeout=90.0,
                    )
                    break
                except asyncio.TimeoutError:
                    if _attempt == 0:
                        logger.warning(f"Grok-4 timeout for {symbol} (attempt 1/2), retrying…")
                        await asyncio.sleep(3)
                    else:
                        raise
            if grok_resp is None:
                raise ValueError("Grok-4 returned no response after retries")
            result_text = (grok_resp.choices[0].message.content or "").strip()
            if not result_text:
                raise ValueError("Grok returned empty response")
            if "```json" in result_text:
                import re as _re
                m = _re.search(r'```json\s*(.*?)\s*```', result_text, _re.DOTALL)
                if m:
                    result_text = m.group(1)
            f1 = result_text.find("{")
            f2 = result_text.rfind("}")
            if f1 >= 0 and f2 > f1:
                result_text = result_text[f1:f2 + 1]
            grok_result = json.loads(result_text)
            logger.info(f"🤖 Grok verdict {symbol}: {grok_result.get('recommendation')} (conf: {grok_result.get('confidence')})")
            rec = grok_result.get('recommendation', '')
            if direction == 'SHORT' and rec in ('STRONG BUY', 'BUY'):
                rec = rec.replace('BUY', 'SELL')
            elif direction == 'LONG' and rec in ('STRONG SELL', 'SELL'):
                rec = rec.replace('SELL', 'BUY')
            if not rec:
                rec = 'SELL' if direction == 'SHORT' else 'BUY'
            return {
                'approved': grok_result.get('approved', True),
                'reasoning': grok_result.get('reasoning', ''),
                'ai_confidence': grok_result.get('confidence', 5),
                'recommendation': rec,
                'entry_quality': grok_result.get('entry_quality', 'FAIR'),
                'trade_explainer': grok_result.get('trade_explainer', ''),
                'key_risk': ''
            }
    except Exception as e:
        logger.warning(f"Grok analysis failed for {symbol}: {e}")

    if _gemini_fallback:
        logger.info(f"⚠️ {symbol} — Grok unavailable, passing on Gemini approval (conf: {_gemini_fallback.get('ai_confidence', 5)})")
        return _gemini_fallback

    logger.info(f"🚫 {symbol} blocked — both AI validators unavailable")
    return {
        'approved': False,
        'reasoning': 'AI validation failed — signal blocked',
        'ai_confidence': 0,
        'recommendation': 'AVOID',
        'key_risk': 'AI unavailable'
    }

# Top 10 coins get higher leverage (more stable)
TOP_10_COINS = {'BTC', 'ETH', 'SOL', 'XRP', 'DOGE', 'ADA', 'AVAX', 'DOT', 'LINK', 'LTC'}

def is_top_coin(symbol: str) -> bool:
    """Check if a symbol is a top 10 coin"""
    base = symbol.replace('USDT', '').replace('PERP', '').upper()
    return base in TOP_10_COINS

# Scanning control
SOCIAL_SCANNING_ENABLED = True
_social_scanning_active = False

# Cooldowns to prevent over-trading
_symbol_cooldowns: Dict[str, datetime] = {}
SYMBOL_COOLDOWN_MINUTES = 30

_last_signal_broadcast_time: Optional[datetime] = None
MIN_SIGNAL_GAP_MINUTES = 30

# AI rejection cooldown before re-analyzing a rejected coin
_ai_rejection_cache: Dict[str, datetime] = {}
AI_REJECTION_COOLDOWN_MINUTES = 5

_signalled_today: Dict[str, datetime] = {}
_signalled_today_date: Optional[datetime] = None


def is_coin_in_signalled_cooldown(symbol: str) -> bool:
    global _signalled_today, _signalled_today_date
    today = datetime.utcnow().date()
    if _signalled_today_date != today:
        _signalled_today = {}
        _signalled_today_date = today
    if symbol in _signalled_today:
        logger.info(f"🔇 {symbol} already signalled today (memory) - one signal per coin per day")
        return True
    try:
        from app.database import SessionLocal
        from app.models import Signal
        db = SessionLocal()
        try:
            start_of_day = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
            existing = db.query(Signal).filter(
                Signal.symbol == symbol,
                Signal.created_at >= start_of_day
            ).first()
            if existing:
                _signalled_today[symbol] = existing.created_at
                logger.info(f"🔇 {symbol} already signalled today (database, sent at {existing.created_at}) - one signal per coin per day")
                return True
        finally:
            db.close()
    except Exception as e:
        logger.warning(f"DB cooldown check failed for {symbol}: {e}")
    return False


def add_to_signalled_cooldown(symbol: str):
    global _signalled_today, _signalled_today_date
    today = datetime.utcnow().date()
    if _signalled_today_date != today:
        _signalled_today = {}
        _signalled_today_date = today
    _signalled_today[symbol] = datetime.now()
    logger.info(f"⏰ {symbol} signalled - won't signal again today")


def is_coin_in_ai_rejection_cooldown(symbol: str, direction: str) -> bool:
    cache_key = f"{symbol}_{direction}"
    if cache_key in _ai_rejection_cache:
        rejected_at = _ai_rejection_cache[cache_key]
        if datetime.now() - rejected_at < timedelta(minutes=AI_REJECTION_COOLDOWN_MINUTES):
            remaining = AI_REJECTION_COOLDOWN_MINUTES - (datetime.now() - rejected_at).total_seconds() / 60
            logger.debug(f"⏳ {symbol} {direction} in AI rejection cooldown ({remaining:.0f}min left)")
            return True
        else:
            del _ai_rejection_cache[cache_key]
    return False


def add_to_ai_rejection_cooldown(symbol: str, direction: str):
    cache_key = f"{symbol}_{direction}"
    _ai_rejection_cache[cache_key] = datetime.now()
    logger.info(f"📝 {symbol} {direction} added to AI rejection cooldown for {AI_REJECTION_COOLDOWN_MINUTES}min")


# ── Rejection log ─────────────────────────────────────────────────────────────
_rejection_log: list = []           # recent rejections, FIFO, max 80
_rejection_stats_today: Dict[str, int] = {}   # "SCANNER:reason" → count
_rejection_stats_date = None
MAX_REJECTION_LOG = 80


def log_rejection(symbol: str, scanner: str, reason: str,
                  direction: str = 'LONG', confidence: Optional[int] = None,
                  ai_reason: str = ''):
    """Record a rejected signal for admin diagnostics."""
    global _rejection_log, _rejection_stats_today, _rejection_stats_date
    today = datetime.now().date()
    if _rejection_stats_date != today:
        _rejection_stats_today = {}
        _rejection_stats_date = today
    _rejection_log.append({
        'time': datetime.now(),
        'symbol': symbol,
        'scanner': scanner,
        'reason': reason,
        'direction': direction,
        'confidence': confidence,
        'ai_reason': ai_reason,
    })
    if len(_rejection_log) > MAX_REJECTION_LOG:
        _rejection_log.pop(0)
    key = f"{scanner}:{reason}"
    _rejection_stats_today[key] = _rejection_stats_today.get(key, 0) + 1

# Signal tracking
_daily_social_signals = 0
_daily_reset_date: Optional[datetime] = None
MAX_DAILY_SOCIAL_SIGNALS = 999

_daily_scalp_signals = 0
MAX_DAILY_SCALP_SIGNALS = 5
_last_scalp_time: Optional[datetime] = None
MIN_SCALP_GAP_MINUTES = 45

_daily_squeeze_signals = 0
MAX_DAILY_SQUEEZE_SIGNALS = 4
_last_squeeze_time: Optional[datetime] = None
MIN_SQUEEZE_GAP_MINUTES = 45

_daily_supertrend_signals = 0
MAX_DAILY_SUPERTREND_SIGNALS = 4
_last_supertrend_time: Optional[datetime] = None
MIN_SUPERTREND_GAP_MINUTES = 45

_daily_macd_signals = 0
MAX_DAILY_MACD_SIGNALS = 4
_last_macd_time: Optional[datetime] = None
MIN_MACD_GAP_MINUTES = 45

_daily_range_breakout_signals = 0
MAX_DAILY_RANGE_BREAKOUT_SIGNALS = 3
_last_range_breakout_time: Optional[datetime] = None
MIN_RANGE_BREAKOUT_GAP_MINUTES = 60

_daily_ema_pullback_signals = 0
MAX_DAILY_EMA_PULLBACK_SIGNALS = 4
_last_ema_pullback_time: Optional[datetime] = None
MIN_EMA_PULLBACK_GAP_MINUTES = 45

_daily_half_back_signals = 0
MAX_DAILY_HALF_BACK_SIGNALS = 3
_last_half_back_time: Optional[datetime] = None
MIN_HALF_BACK_GAP_MINUTES = 60

_daily_oversold_reversal_signals = 0
MAX_DAILY_OVERSOLD_REVERSAL_SIGNALS = 3
_last_oversold_reversal_time: Optional[datetime] = None
MIN_OVERSOLD_REVERSAL_GAP_MINUTES = 60

_global_daily_signals = 0
_global_daily_reset_date: Optional[datetime] = None
MAX_GLOBAL_DAILY_SIGNALS = 999

from app.services.risk_controls import record_trade_result, is_circuit_breaker_active

def check_global_signal_limit() -> bool:
    global _global_daily_signals, _global_daily_reset_date
    today = datetime.now().date()
    if _global_daily_reset_date != today:
        _global_daily_signals = 0
        _global_daily_reset_date = today
    return _global_daily_signals < MAX_GLOBAL_DAILY_SIGNALS

def increment_global_signal_count():
    global _global_daily_signals, _global_daily_reset_date
    today = datetime.now().date()
    if _global_daily_reset_date != today:
        _global_daily_signals = 0
        _global_daily_reset_date = today
    _global_daily_signals += 1
    logger.info(f"📊 Global daily signals: {_global_daily_signals}/{MAX_GLOBAL_DAILY_SIGNALS}")

def get_global_signal_count() -> int:
    global _global_daily_signals, _global_daily_reset_date
    today = datetime.now().date()
    if _global_daily_reset_date != today:
        return 0
    return _global_daily_signals


def check_signal_gap() -> bool:
    global _last_signal_broadcast_time
    if _last_signal_broadcast_time is None:
        return True
    elapsed = (datetime.now() - _last_signal_broadcast_time).total_seconds() / 60
    if elapsed < MIN_SIGNAL_GAP_MINUTES:
        logger.info(f"📱 Signal gap: {elapsed:.0f}min since last signal (need {MIN_SIGNAL_GAP_MINUTES}min) - waiting")
        return False
    return True


def record_signal_broadcast():
    global _last_signal_broadcast_time
    _last_signal_broadcast_time = datetime.now()
    logger.info(f"📱 Signal broadcast recorded - next signal allowed after {MIN_SIGNAL_GAP_MINUTES}min gap")


def is_social_scanning_enabled() -> bool:
    return SOCIAL_SCANNING_ENABLED


def enable_social_scanning():
    global SOCIAL_SCANNING_ENABLED
    SOCIAL_SCANNING_ENABLED = True
    logger.info("📱 Social scanning ENABLED")


def disable_social_scanning():
    global SOCIAL_SCANNING_ENABLED
    SOCIAL_SCANNING_ENABLED = False
    logger.info("📱 Social scanning DISABLED")


def is_symbol_on_cooldown(symbol: str) -> bool:
    """Check if symbol is on cooldown."""
    if symbol in _symbol_cooldowns:
        cooldown_end = _symbol_cooldowns[symbol]
        if datetime.now() < cooldown_end:
            return True
        del _symbol_cooldowns[symbol]
    return False


def add_symbol_cooldown(symbol: str):
    """Add symbol to cooldown."""
    _symbol_cooldowns[symbol] = datetime.now() + timedelta(minutes=SYMBOL_COOLDOWN_MINUTES)


def get_scanner_status() -> dict:
    """Return live state of every scanner for admin diagnostics."""
    now = datetime.now()

    def _scanner(label, daily_count, daily_max, last_time, gap_min):
        used_up = daily_count >= daily_max
        if last_time:
            elapsed = (now - last_time).total_seconds() / 60
            in_gap = elapsed < gap_min
            since = f"{elapsed:.0f}m ago"
        else:
            in_gap = False
            since = "never"
        if used_up:
            status = "🚫 LIMIT"
        elif in_gap:
            remaining = gap_min - (now - last_time).total_seconds() / 60
            status = f"⏳ GAP ({remaining:.0f}m left)"
        else:
            status = "✅ READY"
        return {
            "label": label,
            "status": status,
            "count": f"{daily_count}/{daily_max}",
            "last": since,
        }

    scanners = [
        _scanner("VOLUME SCALP",       _daily_scalp_signals,            MAX_DAILY_SCALP_SIGNALS,            _last_scalp_time,            MIN_SCALP_GAP_MINUTES),
        _scanner("SQUEEZE BREAKOUT",   _daily_squeeze_signals,          MAX_DAILY_SQUEEZE_SIGNALS,          _last_squeeze_time,          MIN_SQUEEZE_GAP_MINUTES),
        _scanner("SUPERTREND",         _daily_supertrend_signals,       MAX_DAILY_SUPERTREND_SIGNALS,       _last_supertrend_time,       MIN_SUPERTREND_GAP_MINUTES),
        _scanner("MACD MOMENTUM",      _daily_macd_signals,             MAX_DAILY_MACD_SIGNALS,             _last_macd_time,             MIN_MACD_GAP_MINUTES),
        _scanner("RANGE BREAKOUT",     _daily_range_breakout_signals,   MAX_DAILY_RANGE_BREAKOUT_SIGNALS,   _last_range_breakout_time,   MIN_RANGE_BREAKOUT_GAP_MINUTES),
        _scanner("EMA PULLBACK",       _daily_ema_pullback_signals,     MAX_DAILY_EMA_PULLBACK_SIGNALS,     _last_ema_pullback_time,     MIN_EMA_PULLBACK_GAP_MINUTES),
        _scanner("HALF BACK",          _daily_half_back_signals,        MAX_DAILY_HALF_BACK_SIGNALS,        _last_half_back_time,        MIN_HALF_BACK_GAP_MINUTES),
        _scanner("OVERSOLD REVERSAL",  _daily_oversold_reversal_signals,MAX_DAILY_OVERSOLD_REVERSAL_SIGNALS,_last_oversold_reversal_time,MIN_OVERSOLD_REVERSAL_GAP_MINUTES),
    ]

    if _last_signal_broadcast_time:
        elapsed_bc = (now - _last_signal_broadcast_time).total_seconds() / 60
        if elapsed_bc < MIN_SIGNAL_GAP_MINUTES:
            remaining_bc = MIN_SIGNAL_GAP_MINUTES - elapsed_bc
            broadcast_status = f"⏳ {remaining_bc:.0f}m until next allowed"
        else:
            broadcast_status = f"✅ Open (last {elapsed_bc:.0f}m ago)"
    else:
        broadcast_status = "✅ Open (no signal yet today)"

    cooldown_count = len([s for s, t in _symbol_cooldowns.items() if t > now])
    signalled_count = len(_signalled_today) if _signalled_today else 0

    # Build rejection summary grouped by scanner
    rejection_by_scanner: Dict[str, Dict[str, int]] = {}
    for key, count in _rejection_stats_today.items():
        parts = key.split(':', 1)
        sc = parts[0] if len(parts) == 2 else key
        reason = parts[1] if len(parts) == 2 else 'UNKNOWN'
        if sc not in rejection_by_scanner:
            rejection_by_scanner[sc] = {}
        rejection_by_scanner[sc][reason] = count

    total_rejected_today = sum(_rejection_stats_today.values())

    # Recent rejections (last 20, newest first)
    recent = list(reversed(_rejection_log[-20:])) if _rejection_log else []

    return {
        "scanners": scanners,
        "broadcast_gap": broadcast_status,
        "global_today": get_global_signal_count(),
        "symbol_cooldowns": cooldown_count,
        "signalled_coins": signalled_count,
        "reset_date": str(_daily_reset_date),
        "total_rejected_today": total_rejected_today,
        "rejection_by_scanner": rejection_by_scanner,
        "recent_rejections": recent,
    }


def reset_daily_counters_if_needed():
    """Reset daily counters at midnight UTC."""
    global _daily_social_signals, _daily_reset_date, _daily_scalp_signals
    global _daily_squeeze_signals, _daily_supertrend_signals, _daily_macd_signals
    global _daily_range_breakout_signals, _daily_ema_pullback_signals
    global _daily_half_back_signals, _daily_oversold_reversal_signals

    today = datetime.utcnow().date()
    if _daily_reset_date != today:
        _daily_social_signals = 0
        _daily_scalp_signals = 0
        _daily_squeeze_signals = 0
        _daily_supertrend_signals = 0
        _daily_macd_signals = 0
        _daily_range_breakout_signals = 0
        _daily_ema_pullback_signals = 0
        _daily_half_back_signals = 0
        _daily_oversold_reversal_signals = 0
        _daily_reset_date = today
        logger.info("📱 Daily signal counters reset (all scanner types)")


class SocialSignalService:
    """Service for generating trading signals from social data."""
    
    def __init__(self):
        self.http_client: Optional[httpx.AsyncClient] = None
        
    async def init(self):
        """Initialize HTTP client."""
        if not self.http_client:
            self.http_client = httpx.AsyncClient(timeout=15)
    
    async def close(self):
        """Close HTTP client."""
        if self.http_client:
            await self.http_client.aclose()
            self.http_client = None
    
    async def _get_binance_tickers(self) -> Optional[List[Dict]]:
        """Get all futures tickers — MEXC primary (wider low-cap coverage), Binance fallback."""
        # PRIMARY: MEXC Futures
        try:
            resp = await self.http_client.get("https://contract.mexc.com/api/v1/contract/ticker", timeout=8)
            if resp.status_code == 200:
                mexc_data = resp.json()
                raw = mexc_data.get('data', [])
                if raw:
                    tickers = []
                    for t in raw:
                        sym = t.get('symbol', '')
                        if not sym.endswith('_USDT'):
                            continue
                        tickers.append({
                            'symbol': sym.replace('_USDT', 'USDT'),
                            'priceChangePercent': float(t.get('riseFallRate', 0)) * 100,
                            'quoteVolume': float(t.get('amount24', 0) or 0),
                            'lastPrice': float(t.get('lastPrice', 0)),
                            'highPrice': float(t.get('high24Price', 0) or 0),
                            'lowPrice': float(t.get('low24Price', 0) or 0),
                            'openPrice': float(t.get('openPrice', 0) or 0),
                        })
                    logger.info(f"📡 Tickers: MEXC primary ({len(tickers)} futures pairs)")
                    return tickers
        except Exception as e:
            logger.debug(f"MEXC ticker fetch failed: {e}")

        # FALLBACK: Binance Futures WebSocket cache → REST
        try:
            from app.services.binance_ws import get_all_tickers_with_fallback
            tickers = await get_all_tickers_with_fallback(self.http_client)
            if tickers:
                logger.info(f"📡 Tickers: Binance WS fallback ({len(tickers)} pairs)")
                return tickers
        except Exception as e:
            logger.debug(f"Binance WS ticker fetch failed: {e}")

        try:
            resp = await self.http_client.get("https://fapi.binance.com/fapi/v1/ticker/24hr", timeout=8)
            if resp.status_code == 200:
                data = resp.json()
                logger.info(f"📡 Tickers: Binance REST fallback ({len(data)} pairs)")
                return data
        except Exception:
            pass
        return None

    async def fetch_price_data(self, symbol: str) -> Optional[Dict]:
        """Fetch current price and technical data from Binance, fallback to Bitunix."""
        try:
            await self.init()
            
            result = await self._fetch_binance_price(symbol)
            if result:
                return result
            
            result = await self._fetch_bitunix_price(symbol)
            if result:
                return result
            
            return None
            
        except Exception as e:
            logger.error(f"Error fetching price data for {symbol}: {e}")
            return None
    
    async def _fetch_binance_price(self, symbol: str) -> Optional[Dict]:
        """Try fetching from Binance Futures with enhanced technical analysis."""
        try:
            from app.services.enhanced_ta import analyze_klines

            ticker_url = f"https://fapi.binance.com/fapi/v1/ticker/24hr?symbol={symbol}"
            klines_15m_url = f"https://fapi.binance.com/fapi/v1/klines?symbol={symbol}&interval=15m&limit=50"
            klines_1h_url = f"https://fapi.binance.com/fapi/v1/klines?symbol={symbol}&interval=1h&limit=30"

            import asyncio as _aio
            ticker_task = self.http_client.get(ticker_url, timeout=5)
            k15_task = self.http_client.get(klines_15m_url, timeout=5)
            k1h_task = self.http_client.get(klines_1h_url, timeout=5)
            resp, k15_resp, k1h_resp = await _aio.gather(ticker_task, k15_task, k1h_task, return_exceptions=True)

            if isinstance(resp, Exception) or resp.status_code != 200:
                return None

            ticker = resp.json()

            klines_15m = []
            closes = []
            volumes = []
            if not isinstance(k15_resp, Exception) and k15_resp.status_code == 200:
                klines_15m = k15_resp.json()
                closes = [float(k[4]) for k in klines_15m]
                volumes = [float(k[5]) for k in klines_15m]

            klines_1h = []
            if not isinstance(k1h_resp, Exception) and k1h_resp.status_code == 200:
                klines_1h = k1h_resp.json()

            rsi = self._calc_rsi(closes)
            volume_ratio = self._calc_volume_ratio(volumes)

            btc_closes = await self._get_btc_closes()
            btc_corr = self._calc_correlation(closes, btc_closes) if closes and btc_closes else 0.0

            enhanced_ta = analyze_klines(klines_15m, klines_1h)

            return {
                'price': float(ticker.get('lastPrice', 0)),
                'change_24h': float(ticker.get('priceChangePercent', 0)),
                'volume_24h': float(ticker.get('quoteVolume', 0)),
                'high_24h': float(ticker.get('highPrice', 0)),
                'low_24h': float(ticker.get('lowPrice', 0)),
                'rsi': rsi,
                'volume_ratio': volume_ratio,
                'btc_correlation': btc_corr,
                'enhanced_ta': enhanced_ta,
            }
        except Exception:
            return None
    
    async def _fetch_bitunix_price(self, symbol: str) -> Optional[Dict]:
        """Fallback: fetch from Bitunix tickers + Binance spot klines for RSI + enhanced TA."""
        try:
            from app.services.enhanced_ta import analyze_klines

            url = f"https://fapi.bitunix.com/api/v1/futures/market/tickers?symbols={symbol}"
            resp = await self.http_client.get(url, timeout=5)

            if resp.status_code != 200:
                return None

            data = resp.json()
            tickers = data.get('data', [])
            if not tickers or not isinstance(tickers, list):
                return None

            ticker = tickers[0]
            price = float(ticker.get('lastPrice', 0) or ticker.get('last', 0) or 0)
            if price <= 0:
                return None

            open_price = float(ticker.get('open', 0) or 0)
            volume_24h = float(ticker.get('quoteVol', 0) or 0)
            change_24h = ((price - open_price) / open_price * 100) if open_price > 0 else 0

            rsi = 50
            volume_ratio = 1.0
            btc_corr = 0.0
            enhanced_ta = {}
            try:
                import asyncio as _aio
                k15_url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval=15m&limit=50"
                k1h_url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval=1h&limit=30"
                k15_task = self.http_client.get(k15_url, timeout=5)
                k1h_task = self.http_client.get(k1h_url, timeout=5)
                k15_resp, k1h_resp = await _aio.gather(k15_task, k1h_task, return_exceptions=True)

                klines_15m = []
                klines_1h = []
                if not isinstance(k15_resp, Exception) and k15_resp.status_code == 200:
                    klines_15m = k15_resp.json()
                    closes = [float(k[4]) for k in klines_15m]
                    volumes = [float(k[5]) for k in klines_15m]
                    rsi = self._calc_rsi(closes)
                    volume_ratio = self._calc_volume_ratio(volumes)
                    btc_closes = await self._get_btc_closes()
                    btc_corr = self._calc_correlation(closes, btc_closes) if closes and btc_closes else 0.0
                    logger.info(f"  📱 {symbol} - Bitunix price + Binance spot RSI={rsi:.0f}")
                else:
                    logger.info(f"  📱 {symbol} - Bitunix only, RSI unavailable (defaulting 50)")

                if not isinstance(k1h_resp, Exception) and k1h_resp.status_code == 200:
                    klines_1h = k1h_resp.json()

                enhanced_ta = analyze_klines(klines_15m, klines_1h)
            except Exception:
                pass

            return {
                'price': price,
                'change_24h': change_24h,
                'volume_24h': volume_24h,
                'high_24h': float(ticker.get('high', 0) or 0),
                'low_24h': float(ticker.get('low', 0) or 0),
                'rsi': rsi,
                'volume_ratio': volume_ratio,
                'btc_correlation': btc_corr,
                'enhanced_ta': enhanced_ta,
            }
        except Exception:
            return None
    
    def _calc_rsi(self, closes: list) -> float:
        """Calculate RSI from close prices."""
        if len(closes) < 14:
            return 50
        deltas = [closes[i] - closes[i-1] for i in range(1, len(closes))]
        gains = [d if d > 0 else 0 for d in deltas]
        losses = [-d if d < 0 else 0 for d in deltas]
        avg_gain = sum(gains[-14:]) / 14
        avg_loss = sum(losses[-14:]) / 14
        if avg_loss > 0:
            rs = avg_gain / avg_loss
            return 100 - (100 / (1 + rs))
        return 50
    
    def _calc_volume_ratio(self, volumes: list) -> float:
        """Calculate volume ratio."""
        if len(volumes) < 5:
            return 1.0
        avg_vol = sum(volumes[:-1]) / len(volumes[:-1])
        return volumes[-1] / avg_vol if avg_vol > 0 else 1.0
    
    def _calc_social_strength(self, galaxy_score: float, sentiment: float,
                              social_volume: int, social_interactions: int,
                              social_dominance: float, alt_rank: int,
                              social_vol_change: float = 0, is_spike: bool = False) -> float:
        """
        Composite social strength score (0-100) using multiple social metrics.
        Weights: Galaxy Score 25%, Sentiment 15%, Social Volume 15%, 
                 Interactions 15%, Dominance 10%, AltRank 10%, Spike Bonus 10%
        """
        gs_score = min(galaxy_score / 16, 1.0) * 25
        
        sent_score = min(max(sentiment, 0), 1.0) * 15
        
        if social_volume >= 1000:
            vol_score = 15
        elif social_volume >= 500:
            vol_score = 12
        elif social_volume >= 100:
            vol_score = 8
        elif social_volume >= 20:
            vol_score = 5
        else:
            vol_score = 2
        
        if social_interactions >= 100000:
            int_score = 15
        elif social_interactions >= 50000:
            int_score = 12
        elif social_interactions >= 10000:
            int_score = 9
        elif social_interactions >= 1000:
            int_score = 6
        else:
            int_score = 2
        
        if social_dominance >= 1.0:
            dom_score = 10
        elif social_dominance >= 0.5:
            dom_score = 8
        elif social_dominance >= 0.1:
            dom_score = 5
        else:
            dom_score = 1
        
        if alt_rank <= 10:
            rank_score = 10
        elif alt_rank <= 50:
            rank_score = 8
        elif alt_rank <= 100:
            rank_score = 6
        elif alt_rank <= 300:
            rank_score = 3
        else:
            rank_score = 1
        
        spike_score = 0
        if is_spike:
            if social_vol_change >= 200:
                spike_score = 10
            elif social_vol_change >= 100:
                spike_score = 8
            elif social_vol_change >= 50:
                spike_score = 5
            else:
                spike_score = 3
        
        total = gs_score + sent_score + vol_score + int_score + dom_score + rank_score + spike_score
        return min(total, 100)
    
    _btc_klines_cache: Optional[list] = None
    _btc_klines_time: Optional[datetime] = None
    
    async def _get_btc_closes(self) -> list:
        """Get cached BTC 15m closes for correlation calc. Cache 5 min."""
        now = datetime.utcnow()
        if self._btc_klines_cache and self._btc_klines_time and (now - self._btc_klines_time).seconds < 300:
            return self._btc_klines_cache
        try:
            url = "https://fapi.binance.com/fapi/v1/klines?symbol=BTCUSDT&interval=15m&limit=20"
            resp = await self.http_client.get(url, timeout=5)
            if resp.status_code == 200:
                klines = resp.json()
                closes = [float(k[4]) for k in klines]
                self._btc_klines_cache = closes
                self._btc_klines_time = now
                return closes
        except Exception:
            pass
        return self._btc_klines_cache or []
    
    def _calc_correlation(self, coin_closes: list, btc_closes: list) -> float:
        """Calculate Pearson correlation of returns between coin and BTC."""
        n = min(len(coin_closes), len(btc_closes))
        if n < 6:
            return 0.0
        coin_closes = coin_closes[-n:]
        btc_closes = btc_closes[-n:]
        coin_returns = [(coin_closes[i] - coin_closes[i-1]) / coin_closes[i-1] for i in range(1, n) if coin_closes[i-1] != 0]
        btc_returns = [(btc_closes[i] - btc_closes[i-1]) / btc_closes[i-1] for i in range(1, n) if btc_closes[i-1] != 0]
        m = min(len(coin_returns), len(btc_returns))
        if m < 5:
            return 0.0
        coin_returns = coin_returns[-m:]
        btc_returns = btc_returns[-m:]
        mean_c = sum(coin_returns) / m
        mean_b = sum(btc_returns) / m
        cov = sum((coin_returns[i] - mean_c) * (btc_returns[i] - mean_b) for i in range(m))
        var_c = sum((x - mean_c) ** 2 for x in coin_returns)
        var_b = sum((x - mean_b) ** 2 for x in btc_returns)
        if var_c == 0 or var_b == 0:
            return 0.0
        return cov / (var_c ** 0.5 * var_b ** 0.5)
    
    _bitunix_symbols_cache: Optional[set] = None
    _bitunix_cache_time: Optional[datetime] = None
    
    async def _get_bitunix_symbols(self) -> set:
        """Get cached set of Bitunix symbols."""
        now = datetime.now()
        if self._bitunix_symbols_cache and self._bitunix_cache_time and (now - self._bitunix_cache_time).seconds < 300:
            return self._bitunix_symbols_cache
        
        try:
            await self.init()
            url = "https://fapi.bitunix.com/api/v1/futures/market/trading_pairs"
            resp = await self.http_client.get(url, timeout=10)
            
            if resp.status_code == 200:
                data = resp.json()
                contracts = data.get('data', [])
                symbols = {c.get('symbol', '').upper() for c in contracts if c.get('symbol')}
                SocialSignalService._bitunix_symbols_cache = symbols
                SocialSignalService._bitunix_cache_time = now
                logger.info(f"📱 Cached {len(symbols)} Bitunix symbols")
                return symbols
        except Exception as e:
            logger.error(f"Error fetching Bitunix symbols: {e}")
        
        return self._bitunix_symbols_cache or set()
    
    async def check_bitunix_availability(self, symbol: str) -> bool:
        """Check if symbol is tradeable on Bitunix."""
        try:
            symbols = await self._get_bitunix_symbols()
            return symbol.upper() in symbols
            
        except Exception as e:
            logger.debug(f"Bitunix check failed for {symbol}: {e}")
            return False
    
    async def generate_social_signal(
        self,
        risk_level: str = "MEDIUM",
        min_galaxy_score: int = 8
    ) -> Optional[Dict]:
        """
        Generate a trading signal based on social metrics.
        
        Risk levels affect filters:
        - SAFE: Signal Score ≥70, RSI 40-65, bullish price action only
        - BALANCED: Signal Score ≥60, RSI 35-70, some flexibility
        - AGGRESSIVE: Signal Score ≥50, RSI 30-75, more aggressive
        - NEWS RUNNER: Signal Score ≥80, catch big pumps (+15-30%)
        
        Returns signal dict or None.
        """
        global _daily_social_signals
        
        reset_daily_counters_if_needed()
        
        if _daily_social_signals >= MAX_DAILY_SOCIAL_SIGNALS:
            logger.info(f"📱 Daily social signal limit reached ({MAX_DAILY_SOCIAL_SIGNALS})")
            return None
        
        # Get risk-based filters
        # ALL mode: Smart mode - accepts more signals, TP/SL adapts dynamically
        # RISK = CONFIDENCE FILTER (how strong must the signal be?)
        # TP/SL = ALWAYS DYNAMIC based on actual signal strength
        # 
        # LOW = Only high-confidence signals (can still get big TPs on strong signals!)
        # MEDIUM = Moderate confidence signals
        # HIGH = Accept lower confidence signals
        # ALL = Accept everything, dynamic TPs
        
        if risk_level == "LOW":
            min_score = 20
            rsi_range = (38, 75)
            min_change = 1.0
            min_sentiment = 0.6
        elif risk_level == "MEDIUM":
            min_score = 18
            rsi_range = (35, 75)
            min_change = 0.0
            min_sentiment = 0.45
        elif risk_level == "HIGH":
            min_score = 15
            rsi_range = (32, 78)
            min_change = -3.0
            min_sentiment = 0.3
        else:  # ALL or MOMENTUM
            min_score = 12
            rsi_range = (28, 80)
            min_change = -5.0
            min_sentiment = 0.2
        
        logger.info(f"📱 SOCIAL SCANNER | Risk: {risk_level} | Min Score: {min_score}")
        
        bitunix_symbols = await self._get_bitunix_symbols()
        logger.info(f"📱 Pre-loaded {len(bitunix_symbols)} Bitunix symbols for filtering")
        
        raw_tickers = await self._get_binance_tickers()
        if not raw_tickers:
            logger.warning("📱 Momentum scan: no Binance ticker data available")
            return None

        import math
        combined = []
        for t in raw_tickers:
            sym = t.get('symbol', '')
            if not sym.endswith('USDT'):
                continue
            if sym.upper() not in bitunix_symbols:
                continue
            chg = float(t.get('priceChangePercent', 0))
            vol = float(t.get('quoteVolume', 0))
            last = float(t.get('lastPrice', 0))
            high = float(t.get('highPrice', 0))
            low = float(t.get('lowPrice', 0))
            if chg < min_change or chg > 35.0:
                continue
            if vol < 100_000:
                continue
            # Composite top-gainer score:
            #   - change_24h: raw momentum
            #   - vol_score: log-normalised volume conviction
            #   - near_high: price proximity to 24h high (sustained move vs spike that faded)
            vol_score = math.log10(max(vol, 100_000)) - 5.0
            near_high = (last / high) if high > 0 else 0.5
            gainer_score = chg * 2.0 + vol_score * 3.0 + near_high * 15.0
            combined.append({
                'symbol': sym,
                'change_24h': chg,
                'volume_24h': vol,
                'high_24h': high,
                'low_24h': low,
                'last_price': last,
                'gainer_score': gainer_score,
            })

        import random
        # Split into top 30 (primary) and the rest (fallback)
        # Each half is shuffled independently so within each group any coin can be picked first
        combined.sort(key=lambda x: x['gainer_score'], reverse=True)
        top_pool = combined[:30]
        rest_pool = combined[30:]
        random.shuffle(top_pool)
        random.shuffle(rest_pool)
        combined = top_pool + rest_pool
        logger.info(f"🏆 Top Gainers scan: {len(raw_tickers)} tickers → {len(top_pool)} top gainers + {len(rest_pool)} fallback on Bitunix")

        if not combined:
            logger.info("📱 No momentum LONG candidates found this cycle")
            return None

        btc_state = await get_btc_state()
        logger.info(f"🌙 SOCIAL LONG BTC → {btc_state['summary']}")

        rejected_reasons = {'cooldown': 0, 'signal_cooldown': 0, 'no_price_data': 0, 'low_volume': 0, 'btc_corr': 0, 'rsi_range': 0, 'ai_cooldown': 0, 'ai_rejected': 0}

        passed_filters = 0
        for coin in combined:
            symbol = coin['symbol']
            price_change = coin['change_24h']
            galaxy_score = 0
            sentiment = 0.5
            social_volume = 0
            social_interactions = 0
            social_dominance = 0
            alt_rank = 9999
            coin_name = symbol.replace('USDT', '')
            social_vol_change = 0
            is_spike = False

            if is_symbol_on_cooldown(symbol):
                logger.debug(f"  📱 {symbol} - On cooldown, skipping")
                rejected_reasons['cooldown'] += 1
                continue

            if is_coin_in_signalled_cooldown(symbol):
                rejected_reasons['signal_cooldown'] += 1
                continue

            passed_filters += 1
            logger.info(f"  📱 ✅ {symbol} - chg={price_change:.1f}% - checking price data...")
            
            price_data = await self.fetch_price_data(symbol)
            if not price_data:
                logger.info(f"  📱 {symbol} - ❌ No price data from any source")
                rejected_reasons['no_price_data'] += 1
                continue
            
            current_price = price_data['price']
            rsi = price_data['rsi']
            volume_24h = price_data['volume_24h']
            volume_ratio = price_data.get('volume_ratio', 1.0)
            btc_corr = price_data.get('btc_correlation', 0.0)

            high_24h = coin.get('high_24h', 0)
            if high_24h > 0 and current_price > 0:
                pullback_from_high = (high_24h - current_price) / high_24h * 100
                if price_change >= 8:
                    if pullback_from_high > 20:
                        logger.info(f"  📱 {symbol} - ❌ Continuation runner -{pullback_from_high:.1f}% from 24h high — move fading ({price_change:.1f}% 24h)")
                        rejected_reasons.setdefault('chart_position', 0)
                        rejected_reasons['chart_position'] += 1
                        continue
                else:
                    if pullback_from_high > 30:
                        logger.info(f"  📱 {symbol} - ❌ Early mover spike fully reversed (-{pullback_from_high:.1f}% from high, only {price_change:.1f}% up on day)")
                        rejected_reasons.setdefault('chart_position', 0)
                        rejected_reasons['chart_position'] += 1
                        continue
                if pullback_from_high < 2 and rsi > 72:
                    if volume_ratio < 1.2:
                        logger.info(f"  📱 {symbol} - ❌ Right at 24h high ({pullback_from_high:.1f}% from top) + RSI {rsi:.0f} — longing the top without volume")
                        rejected_reasons.setdefault('chart_position', 0)
                        rejected_reasons['chart_position'] += 1
                        continue
                    logger.info(f"  📱 {symbol} - ✅ At 24h high but vol {volume_ratio:.1f}x confirms fresh breakout — allowing LONG")

            min_vol = 100_000
            if volume_24h < min_vol:
                logger.info(f"  📱 {symbol} - ❌ Low volume ${volume_24h/1e6:.3f}M (need $100K+)")
                rejected_reasons['low_volume'] += 1
                continue
            
            
            if btc_corr > 0.90:
                logger.info(f"  📱 {symbol} - ❌ Moves identical to BTC ({btc_corr:.2f})")
                rejected_reasons['btc_corr'] += 1
                continue
            
            if not (rsi_range[0] <= rsi <= rsi_range[1]):
                logger.info(f"  📱 {symbol} - ❌ RSI {rsi:.0f} outside range {rsi_range}")
                rejected_reasons['rsi_range'] += 1
                continue
            
            social_strength = min(100, price_change * 3.0 + volume_ratio * 15.0)

            logger.info(f"✅ MOMENTUM LONG: {symbol} | Chg: {price_change:.1f}% | Strength: {social_strength:.0f}/100 | RSI: {rsi:.0f} | Vol: {volume_ratio:.1f}x | BTC corr: {btc_corr:.2f}")

            if price_change >= 25:
                base_tp = 12.0
                base_sl = 6.0
            elif price_change >= 15:
                base_tp = 10.0
                base_sl = 5.0
            elif price_change >= 8:
                base_tp = 7.0
                base_sl = 4.0
            elif price_change >= 3:
                base_tp = 5.0
                base_sl = 3.0
            else:
                base_tp = 3.0
                base_sl = 2.0

            enhanced_ta = price_data.get('enhanced_ta', {})
            tp_percent = base_tp
            sl_percent = base_sl
            deriv_adjustments = []
            derivatives = {}

            if enhanced_ta:
                from app.services.enhanced_ta import get_atr_based_tp_sl, optimize_tp_sl_from_chart_levels
                tp_percent, sl_percent = get_atr_based_tp_sl(enhanced_ta, 'LONG', tp_percent, sl_percent)
                logger.info(f"📊 {symbol} ATR-adjusted TP/SL: TP {tp_percent:.1f}% | SL {sl_percent:.1f}%")
                old_tp, old_sl = tp_percent, sl_percent
                tp_percent, sl_percent = optimize_tp_sl_from_chart_levels(enhanced_ta, 'LONG', current_price, tp_percent, sl_percent)
                if tp_percent != old_tp or sl_percent != old_sl:
                    logger.info(f"📊 {symbol} Chart-optimized TP/SL: TP {old_tp:.1f}%→{tp_percent:.1f}% | SL {old_sl:.1f}%→{sl_percent:.1f}%")

            sl_cap = tp_percent * 0.70
            if sl_percent > sl_cap:
                logger.info(f"📊 {symbol} SL capped at 70% of TP: {sl_percent:.1f}%→{sl_cap:.1f}% (TP={tp_percent:.1f}%)")
                sl_percent = sl_cap

            take_profit = current_price * (1 + tp_percent / 100)
            stop_loss = current_price * (1 - sl_percent / 100)
            tp2 = current_price * (1 + (tp_percent * 1.5) / 100)
            tp3 = current_price * (1 + (tp_percent * 2.0) / 100)

            influencer_data = None
            buzz_momentum = None
            
            order_flow_result = None
            try:
                from app.services.order_flow import analyze_order_flow
                order_flow_result = await analyze_order_flow(
                    symbol=symbol,
                    price_change_1h=price_data.get('price_change_1h', 0),
                    price_change_24h=price_change,
                    current_price=current_price,
                    volume_24h=volume_24h,
                    ta_data=enhanced_ta,
                )
                if order_flow_result and abs(order_flow_result.get('flow_score', 0)) >= 15:
                    logger.info(f"  📊 {symbol} Order Flow: {order_flow_result['flow_direction']} (score: {order_flow_result['flow_score']:+d})")
            except Exception as e:
                logger.debug(f"Order flow analysis failed for {symbol}: {e}")

            signal_candidate = {
                'symbol': symbol,
                'direction': 'LONG',
                'entry_price': current_price,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'take_profit_1': take_profit,
                'take_profit_2': tp2,
                'take_profit_3': tp3,
                'tp_percent': tp_percent,
                'sl_percent': sl_percent,
                'confidence': min(10, max(5, int(price_change / 3))),
                'galaxy_score': 0,
                'sentiment': 0.5,
                'social_volume': 0,
                'social_strength': social_strength,
                'social_vol_change': 0,
                'is_social_spike': False,
                'rsi': rsi,
                'volume_ratio': volume_ratio,
                'btc_correlation': btc_corr,
                '24h_change': price_change,
                '24h_volume': volume_24h,
                'derivatives': {},
                'deriv_adjustments': [],
                'influencer_consensus': None,
                'buzz_momentum': None,
                'enhanced_ta': enhanced_ta,
                'order_flow': order_flow_result,
            }

            if is_coin_in_ai_rejection_cooldown(symbol, 'LONG'):
                logger.info(f"⏳ Skipping AI for {symbol} LONG - in 15min rejection cooldown")
                rejected_reasons['ai_cooldown'] += 1
                continue
            
            ai_result = await ai_analyze_social_signal(signal_candidate)
            
            if not ai_result.get('approved', True) or ai_result.get('ai_confidence', 5) < 6:
                _conf = ai_result.get('ai_confidence', 5)
                _rsn = ai_result.get('reasoning', 'No reason')
                logger.info(f"🤖 AI REJECTED {symbol} LONG: conf={_conf}/10 | {_rsn}")
                log_rejection(symbol, 'SOCIAL', 'AI_REJECTED', 'LONG', _conf, _rsn)
                add_to_ai_rejection_cooldown(symbol, 'LONG')
                rejected_reasons['ai_rejected'] += 1
                continue
            
            ai_reasoning = ai_result.get('reasoning', '')
            ai_confidence = ai_result.get('ai_confidence', 5)
            ai_recommendation = ai_result.get('recommendation', 'BUY')
            trade_explainer = ai_result.get('trade_explainer', '')
            
            add_symbol_cooldown(symbol)
            
            return {
                'symbol': symbol,
                'direction': 'LONG',
                'entry_price': current_price,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'take_profit_1': take_profit,
                'take_profit_2': tp2,
                'take_profit_3': tp3,
                'tp_percent': tp_percent,
                'sl_percent': sl_percent,
                'confidence': min(10, max(5, int(price_change / 3))),
                'reasoning': ai_reasoning,
                'ai_confidence': ai_confidence,
                'ai_recommendation': ai_recommendation,
                'trade_explainer': trade_explainer,
                'trade_type': 'SOCIAL_SIGNAL',
                'strategy': 'SOCIAL_SIGNAL',
                'risk_level': risk_level,
                'galaxy_score': 0,
                'sentiment': 0.5,
                'social_volume': 0,
                'social_interactions': 0,
                'social_dominance': 0,
                'alt_rank': 9999,
                'coin_name': coin_name,
                'rsi': rsi,
                '24h_change': price_change,
                '24h_volume': volume_24h,
                'derivatives': {},
                'deriv_adjustments': [],
                'social_strength': social_strength,
                'social_vol_change': 0,
                'is_social_spike': False,
                'volume_ratio': volume_ratio,
                'btc_correlation': btc_corr,
                'influencer_consensus': None,
                'buzz_momentum': None,
                'enhanced_ta': enhanced_ta,
            }
        
        active_rejections = {k: v for k, v in rejected_reasons.items() if v > 0}
        logger.info(f"📱 No social LONG signals found | {len(combined)} scanned | {passed_filters} passed filters | Rejections: {active_rejections}")
        return None
    
    async def scan_for_momentum_runners(self) -> Optional[Dict]:
        """
        Scan Binance Futures for coins with big moves RIGHT NOW.
        Catches runners that social/news scanners might miss.
        Looks for: ±3% to ±50% 24h change with $500K+ volume.
        Widened filters to catch more PIPPIN-style runners early.
        """
        global _daily_social_signals
        
        reset_daily_counters_if_needed()
        if _daily_social_signals >= MAX_DAILY_SOCIAL_SIGNALS:
            return None
        
        await self.init()
        
        try:
            tickers = None
            data_source = "UNKNOWN"

            try:
                mexc_url = "https://contract.mexc.com/api/v1/contract/ticker"
                resp = await self.http_client.get(mexc_url, timeout=8)
                if resp.status_code == 200:
                    mexc_data = resp.json()
                    mexc_tickers = mexc_data.get('data', [])
                    if mexc_tickers:
                        tickers = []
                        for t in mexc_tickers:
                            sym = t.get('symbol', '')
                            if not sym.endswith('_USDT'):
                                continue
                            normalized = sym.replace('_USDT', 'USDT')
                            change = float(t.get('riseFallRate', 0)) * 100
                            vol = float(t.get('amount24', 0) or 0)
                            last_price = float(t.get('lastPrice', 0))
                            high_price = float(t.get('high24Price', 0) or t.get('maxBidPrice', 0) or 0)
                            low_price = float(t.get('low24Price', 0) or t.get('minAskPrice', 0) or 0)
                            open_price = float(t.get('openPrice', 0) or (last_price / (1 + change / 100) if change != -100 else 0))
                            tickers.append({
                                'symbol': normalized,
                                'priceChangePercent': change,
                                'quoteVolume': vol,
                                'lastPrice': last_price,
                                'highPrice': high_price,
                                'lowPrice': low_price,
                                'openPrice': open_price,
                            })
                        data_source = "MEXC"
                        logger.info(f"🚀 MOMENTUM: Using MEXC data ({len(tickers)} futures tickers)")
            except Exception as mexc_err:
                logger.warning(f"MEXC ticker fetch failed: {mexc_err}")

            if not tickers:
                try:
                    raw = await self._get_binance_tickers()
                    if raw:
                        tickers = []
                        for t in raw:
                            tickers.append({
                                'symbol': t.get('symbol', ''),
                                'priceChangePercent': float(t.get('priceChangePercent', 0)),
                                'quoteVolume': float(t.get('quoteVolume', 0)),
                                'lastPrice': float(t.get('lastPrice', 0)),
                                'highPrice': float(t.get('highPrice', 0)),
                                'lowPrice': float(t.get('lowPrice', 0)),
                                'openPrice': float(t.get('openPrice', 0)),
                                'weightedAvgPrice': float(t.get('weightedAvgPrice', 0)),
                            })
                        data_source = "Binance"
                        logger.info(f"🚀 MOMENTUM: Using Binance fallback ({len(tickers)} tickers)")
                except Exception as bn_err:
                    logger.warning(f"Binance ticker fetch also failed: {bn_err}")

            if not tickers:
                logger.warning("🚀 MOMENTUM: No data source available (MEXC + Binance both failed)")
                return None
            
            runners = []
            for t in tickers:
                sym = t.get('symbol', '')
                if not sym.endswith('USDT') or sym in ('BTCUSDT', 'ETHUSDT', 'USDCUSDT'):
                    continue
                change = float(t.get('priceChangePercent', 0))
                vol = float(t.get('quoteVolume', 0))
                
                if (change >= 12 or change <= -12) and vol >= 10_000_000:
                    runners.append({
                        'symbol': sym,
                        'change_24h': change,
                        'volume_24h': vol,
                        'price': float(t.get('lastPrice', 0)),
                        'high': float(t.get('highPrice', 0)),
                        'low': float(t.get('lowPrice', 0)),
                    })
            
            runners.sort(key=lambda x: abs(x['change_24h']), reverse=True)
            runners = runners[:25]
            
            all_candidates = runners

            if not all_candidates:
                logger.info("🚀 MOMENTUM: No runners found")
                return None

            logger.info(f"🚀 MOMENTUM SCANNER: {len(all_candidates)} candidates")
            
            for r in all_candidates:
                symbol = r['symbol']

                if is_symbol_on_cooldown(symbol) or is_coin_in_signalled_cooldown(symbol):
                    continue
                
                is_available = await self.check_bitunix_availability(symbol)
                if not is_available:
                    continue
                
                price_data = await self.fetch_price_data(symbol)
                if not price_data:
                    continue
                
                rsi = price_data.get('rsi', 50)
                current_price = price_data['price']
                change = r['change_24h']
                vol = r['volume_24h']
                
                abs_change = abs(change)
                vwap_dev = r.get('vwap_deviation', 0)
                
                if change >= 12:
                    direction = 'LONG'
                    if rsi > 60:
                        logger.info(f"  🚀 {symbol} +{change:.1f}% - RSI {rsi:.0f} overbought, skip long")
                        continue
                    if rsi < 38:
                        logger.info(f"  🚀 {symbol} +{change:.1f}% - RSI {rsi:.0f} too weak, skip long")
                        continue
                    if change > 15:
                        logger.info(f"  🚀 {symbol} +{change:.1f}% - Already up >15% on day, skip long")
                        continue
                elif change <= -12:
                    direction = 'SHORT'
                    if rsi < 30:
                        logger.info(f"  🚀 {symbol} {change:.1f}% - RSI {rsi:.0f} oversold, skip short")
                        continue
                    if change < -18:
                        logger.info(f"  🚀 {symbol} {change:.1f}% - Already dumped too much (>18%), skip short")
                        continue
                else:
                    continue
                
                if abs_change >= 20:
                    base_tp = 8.0 + min(abs_change * 0.2, 12.0)
                    base_sl = 3.5
                elif abs_change >= 15:
                    base_tp = 6.0 + min(abs_change * 0.15, 8.0)
                    base_sl = 3.0
                elif abs_change >= 10:
                    base_tp = 4.0 + min(abs_change * 0.1, 4.0)
                    base_sl = 2.0
                elif abs_change >= 5:
                    base_tp = 2.5 + min(abs_change * 0.08, 3.0)
                    base_sl = 1.5
                else:
                    base_tp = 1.5 + min(abs_change * 0.06, 1.5)
                    base_sl = 1.0
                
                enhanced_ta = price_data.get('enhanced_ta', {})

                tp_percent = base_tp
                sl_percent = base_sl
                deriv_adjustments = []
                derivatives = {}

                if enhanced_ta:
                    from app.services.enhanced_ta import get_atr_based_tp_sl, optimize_tp_sl_from_chart_levels
                    tp_percent, sl_percent = get_atr_based_tp_sl(enhanced_ta, direction, tp_percent, sl_percent)
                    old_tp, old_sl = tp_percent, sl_percent
                    tp_percent, sl_percent = optimize_tp_sl_from_chart_levels(enhanced_ta, direction, current_price, tp_percent, sl_percent)
                    if tp_percent != old_tp or sl_percent != old_sl:
                        logger.info(f"📊 {symbol} Chart-optimized TP/SL: TP {old_tp:.1f}%→{tp_percent:.1f}% | SL {old_sl:.1f}%→{sl_percent:.1f}%")
                
                if direction == 'LONG':
                    take_profit = current_price * (1 + tp_percent / 100)
                    stop_loss = current_price * (1 - sl_percent / 100)
                    tp2 = current_price * (1 + (tp_percent * 1.5) / 100)
                    tp3 = current_price * (1 + (tp_percent * 2.0) / 100)
                else:
                    take_profit = current_price * (1 - tp_percent / 100)
                    stop_loss = current_price * (1 + sl_percent / 100)
                    tp2 = current_price * (1 - (tp_percent * 1.5) / 100)
                    tp3 = current_price * (1 - (tp_percent * 2.0) / 100)
                
                if derivatives and derivatives.get('has_data'):
                    funding = derivatives.get('funding_rate', 0) or 0
                    if direction == 'LONG' and funding > 0.05:
                        logger.info(f"  🚀 {symbol} - Extreme positive funding {funding:.4f}% (longs paying heavily), skip long")
                        continue
                    if direction == 'SHORT' and funding < -0.05:
                        logger.info(f"  🚀 {symbol} - Extreme negative funding {funding:.4f}% (shorts paying heavily), skip short")
                        continue
                
                logger.info(f"🚀 RUNNER: {symbol} +{change:.1f}% | Vol ${vol/1e6:.1f}M | RSI {rsi:.0f} | TP {tp_percent:.1f}% SL {sl_percent:.1f}%")
                
                lunar_galaxy = 0
                lunar_sentiment = 0.5
                lunar_social_vol = 0
                lunar_interactions = 0
                lunar_dominance = 0
                lunar_alt_rank = 9999
                lunar_social_vol_change = 0
                influencer_data = None
                buzz_momentum = None
                
                social_strength = self._calc_social_strength(
                    galaxy_score=lunar_galaxy,
                    sentiment=lunar_sentiment,
                    social_volume=lunar_social_vol,
                    social_interactions=lunar_interactions,
                    social_dominance=lunar_dominance,
                    alt_rank=lunar_alt_rank,
                    social_vol_change=lunar_social_vol_change,
                    is_spike=lunar_social_vol_change > 30
                )
                
                signal_candidate = {
                    'symbol': symbol,
                    'direction': direction,
                    'entry_price': current_price,
                    'stop_loss': stop_loss,
                    'take_profit': take_profit,
                    'take_profit_1': take_profit,
                    'take_profit_2': tp2,
                    'take_profit_3': tp3,
                    'tp_percent': tp_percent,
                    'sl_percent': sl_percent,
                    'rsi': rsi,
                    '24h_change': change,
                    '24h_volume': vol,
                    'volume_ratio': price_data.get('volume_ratio', 1.0),
                    'btc_correlation': price_data.get('btc_correlation', 0.0),
                    'derivatives': derivatives,
                    'deriv_adjustments': deriv_adjustments,
                    'galaxy_score': lunar_galaxy,
                    'sentiment': lunar_sentiment,
                    'social_strength': social_strength,
                    'social_vol_change': lunar_social_vol_change,
                    'is_social_spike': lunar_social_vol_change > 30,
                    'influencer_consensus': influencer_data,
                    'buzz_momentum': buzz_momentum,
                    'enhanced_ta': enhanced_ta,
                }

                try:
                    from app.services.order_flow import analyze_order_flow
                    of = await analyze_order_flow(symbol=symbol, price_change_1h=price_data.get('price_change_1h', 0), price_change_24h=change, current_price=current_price, volume_24h=vol, ta_data=enhanced_ta)
                    if of and abs(of.get('flow_score', 0)) >= 15:
                        signal_candidate['order_flow'] = of
                except Exception as e:
                    logger.debug(f"Order flow failed for runner {symbol}: {e}")

                if is_coin_in_ai_rejection_cooldown(symbol, direction):
                    continue
                
                ai_result = await ai_analyze_social_signal(signal_candidate)
                
                if not ai_result.get('approved', True) or ai_result.get('ai_confidence', 5) < 6:
                    _conf = ai_result.get('ai_confidence', 5)
                    _rsn = ai_result.get('reasoning', '')
                    logger.info(f"🤖 AI REJECTED runner {symbol}: conf={_conf}/10 | {_rsn}")
                    log_rejection(symbol, 'MOMENTUM', 'AI_REJECTED', direction, _conf, _rsn)
                    add_to_ai_rejection_cooldown(symbol, direction)
                    continue
                
                add_symbol_cooldown(symbol)
                
                effective_change = abs(change)
                
                return {
                    'symbol': symbol,
                    'direction': direction,
                    'entry_price': current_price,
                    'stop_loss': stop_loss,
                    'take_profit': take_profit,
                    'take_profit_1': take_profit,
                    'take_profit_2': tp2,
                    'take_profit_3': tp3,
                    'tp_percent': tp_percent,
                    'sl_percent': sl_percent,
                    'confidence': min(int(effective_change), 10),
                    'reasoning': ai_result.get('reasoning', ''),
                    'ai_confidence': ai_result.get('ai_confidence', 5),
                    'ai_recommendation': ai_result.get('recommendation', 'BUY'),
                    'trade_explainer': ai_result.get('trade_explainer', ''),
                    'trade_type': 'MOMENTUM_RUNNER',
                    'strategy': 'MOMENTUM_RUNNER',
                    'risk_level': 'MOMENTUM',
                    'galaxy_score': lunar_galaxy,
                    'sentiment': lunar_sentiment,
                    'social_volume': lunar_social_vol,
                    'social_interactions': lunar_interactions,
                    'social_dominance': lunar_dominance,
                    'alt_rank': lunar_alt_rank,
                    'coin_name': symbol.replace('USDT', ''),
                    'rsi': rsi,
                    '24h_change': change,
                    '24h_volume': vol,
                    'derivatives': derivatives,
                    'deriv_adjustments': deriv_adjustments,
                    'social_strength': social_strength,
                    'social_vol_change': lunar_social_vol_change,
                    'volume_ratio': price_data.get('volume_ratio', 1.0),
                    'btc_correlation': price_data.get('btc_correlation', 0.0),
                    'influencer_consensus': influencer_data,
                    'buzz_momentum': buzz_momentum,
                    'vwap_deviation': vwap_dev,
                    'enhanced_ta': enhanced_ta,
                }
            
            logger.info("🚀 No momentum runners passed all checks")
            return None
            
        except Exception as e:
            logger.error(f"Momentum scanner error: {e}")
            return None

    async def scan_for_volume_scalps(self) -> Optional[Dict]:
        """
        Scan for quick volume surge scalp trades.
        Detects coins with sudden volume spikes (2x+ normal) and short-term momentum.
        Tight 1:1 R:R with ~2-3% TP/SL for quick in-and-out trades.
        Target: 4-5 scalps per day.
        """
        global _daily_scalp_signals, _last_scalp_time
        
        reset_daily_counters_if_needed()
        if _daily_scalp_signals >= MAX_DAILY_SCALP_SIGNALS:
            logger.info(f"⚡ Daily scalp limit reached ({MAX_DAILY_SCALP_SIGNALS})")
            return None
        
        if _last_scalp_time:
            elapsed = (datetime.now() - _last_scalp_time).total_seconds() / 60
            if elapsed < MIN_SCALP_GAP_MINUTES:
                logger.info(f"⚡ Scalp gap: {elapsed:.0f}min since last scalp (need {MIN_SCALP_GAP_MINUTES}min)")
                return None
        
        await self.init()
        
        try:
            tickers = None

            # PRIMARY: MEXC (wider low-cap coverage)
            try:
                tickers = await self._get_binance_tickers()
            except Exception:
                pass
            
            if not tickers:
                return None
            
            candidates = []
            for t in tickers:
                sym = t.get('symbol', '')
                if not sym.endswith('USDT') or sym in ('BTCUSDT', 'ETHUSDT', 'USDCUSDT', 'BNBUSDT'):
                    continue
                
                change = float(t.get('priceChangePercent', 0))
                vol = float(t.get('quoteVolume', 0))
                last_price = float(t.get('lastPrice', 0))
                high = float(t.get('highPrice', 0))
                low = float(t.get('lowPrice', 0))
                
                if vol < 3_000_000:
                    continue
                
                if abs(change) > 15 or abs(change) < 1:
                    continue
                
                if last_price <= 0:
                    continue
                
                candidates.append({
                    'symbol': sym,
                    'change_24h': change,
                    'volume_24h': vol,
                    'price': last_price,
                    'high': high,
                    'low': low,
                })
            
            if not candidates:
                logger.debug("⚡ SCALP: No candidates with sufficient volume")
                return None
            
            candidates.sort(key=lambda x: x['volume_24h'], reverse=True)
            candidates = candidates[:30]
            
            logger.info(f"⚡ SCALP SCANNER: {len(candidates)} volume candidates")
            btc_state = await get_btc_state()
            logger.info(f"⚡ SCALP BTC → {btc_state['summary']}")

            for c in candidates:
                symbol = c['symbol']


                if is_symbol_on_cooldown(symbol) or is_coin_in_signalled_cooldown(symbol):
                    continue
                
                is_available = await self.check_bitunix_availability(symbol)
                if not is_available:
                    continue
                
                price_data = await self.fetch_price_data(symbol)
                if not price_data:
                    continue
                
                rsi = price_data.get('rsi', 50)
                current_price = price_data['price']
                volume_ratio = price_data.get('volume_ratio', 1.0)
                volume_24h = price_data.get('volume_24h', 0)
                enhanced_ta = price_data.get('enhanced_ta', {})
                change = c['change_24h']
                high_24h = c.get('high', 0)
                low_24h = c.get('low', 0)
                day_range = high_24h - low_24h if high_24h > low_24h else 0
                day_range_position = (current_price - low_24h) / day_range if day_range > 0 else 0.5

                if volume_ratio < 1.2:
                    logger.info(f"⚡ SCALP SKIP {symbol}: volume ratio {volume_ratio:.2f}x < 1.2x required")
                    continue

                try:
                    _recency_url = f"https://fapi.binance.com/fapi/v1/klines?symbol={symbol}&interval=1h&limit=4"
                    _recency_resp = await self.http_client.get(_recency_url, timeout=5)
                    if _recency_resp.status_code == 200:
                        _1h_candles = _recency_resp.json()
                        if len(_1h_candles) >= 4:
                            _1h_vols = [float(k[5]) for k in _1h_candles]
                            # Annualise the current (in-progress) candle so a candle that
                            # is only 10 min complete isn't unfairly compared to full candles.
                            _open_time_ms  = int(_1h_candles[-1][0])
                            _close_time_ms = int(_1h_candles[-1][6])
                            _candle_span   = _close_time_ms - _open_time_ms  # ~3 600 000 ms
                            _elapsed_ms    = (time.time() * 1000) - _open_time_ms
                            _fill_ratio    = min(_elapsed_ms / _candle_span, 1.0) if _candle_span > 0 else 1.0
                            # Only extrapolate if the candle is at least 10 % complete to avoid div-by-zero
                            if _fill_ratio >= 0.10:
                                _annualised_vol = _1h_vols[-1] / _fill_ratio
                            else:
                                _annualised_vol = _1h_vols[-1]
                            _recent_avg   = sum(_1h_vols[:3]) / 3
                            _recent_ratio = _annualised_vol / _recent_avg if _recent_avg > 0 else 1.0
                            if _recent_ratio < 1.5:
                                logger.info(f"⚡ SCALP RECENCY FAIL: {symbol} 1h vol ratio {_recent_ratio:.2f}x (annualised, {_fill_ratio:.0%} elapsed) — surge not current")
                                continue
                            logger.info(f"⚡ SCALP RECENCY OK: {symbol} 1h vol ratio {_recent_ratio:.2f}x ({_fill_ratio:.0%} of candle elapsed)")
                except Exception:
                    pass

                if change > 1 and change < 8 and rsi < 65:
                    direction = 'LONG'
                elif rsi > 58 and volume_ratio >= 1.2 and day_range_position > 0.55 and change > -5:
                    # SHORT only when price is in upper 45% of today's range AND not already dumped
                    direction = 'SHORT'
                elif volume_ratio >= 2.5 and change > 0.5 and change < 8 and rsi < 60:
                    direction = 'LONG'
                else:
                    continue

                if direction == 'LONG' and rsi > 75:
                    continue
                if direction == 'SHORT' and rsi < 25:
                    continue
                if direction == 'SHORT' and day_range_position < 0.55:
                    logger.info(f"⚡ SCALP SHORT SKIP {symbol}: price at {day_range_position:.0%} of day range (need >55% for SHORT)")
                    continue
                base_tp = 2.5
                base_sl = 2.5
                
                if volume_ratio >= 3.0:
                    base_tp = 3.0
                    base_sl = 3.0
                elif volume_ratio >= 2.5:
                    base_tp = 2.8
                    base_sl = 2.8
                
                tp_percent = base_tp
                sl_percent = base_sl
                deriv_adjustments = []
                derivatives = {}
                
                tp_percent = min(tp_percent, 4.0)
                sl_percent = min(sl_percent, 4.0)
                
                max_diff = max(tp_percent, sl_percent)
                tp_percent = max_diff
                sl_percent = max_diff
                
                if enhanced_ta:
                    from app.services.enhanced_ta import get_atr_based_tp_sl
                    atr_tp, atr_sl = get_atr_based_tp_sl(enhanced_ta, direction, tp_percent, sl_percent)
                    avg_target = (atr_tp + atr_sl) / 2
                    tp_percent = min(avg_target, 4.0)
                    sl_percent = tp_percent
                
                if direction == 'LONG':
                    take_profit = current_price * (1 + tp_percent / 100)
                    stop_loss = current_price * (1 - sl_percent / 100)
                else:
                    take_profit = current_price * (1 - tp_percent / 100)
                    stop_loss = current_price * (1 + sl_percent / 100)
                
                logger.info(f"⚡ SCALP: {symbol} {direction} | Vol {volume_ratio:.1f}x | 24h {change:+.1f}% | RSI {rsi:.0f} | TP/SL {tp_percent:.1f}%/{sl_percent:.1f}%")
                
                signal_candidate = {
                    'symbol': symbol,
                    'direction': direction,
                    'entry_price': current_price,
                    'stop_loss': stop_loss,
                    'take_profit': take_profit,
                    'take_profit_1': take_profit,
                    'take_profit_2': None,
                    'take_profit_3': None,
                    'tp_percent': tp_percent,
                    'sl_percent': sl_percent,
                    'rsi': rsi,
                    '24h_change': change,
                    '24h_volume': volume_24h,
                    'volume_ratio': volume_ratio,
                    'btc_correlation': price_data.get('btc_correlation', 0.0),
                    'derivatives': derivatives,
                    'deriv_adjustments': deriv_adjustments,
                    'enhanced_ta': enhanced_ta,
                    'trade_type': 'VOLUME_SCALP',
                }
                
                if is_coin_in_ai_rejection_cooldown(symbol, direction):
                    continue
                
                ai_result = await ai_analyze_social_signal(signal_candidate)
                
                if not ai_result.get('approved', True) or ai_result.get('ai_confidence', 5) < 6:
                    _conf = ai_result.get('ai_confidence', 5)
                    _rsn = ai_result.get('reasoning', '')
                    logger.info(f"🤖 AI REJECTED scalp {symbol}: conf={_conf}/10 | {_rsn}")
                    log_rejection(symbol, 'VOLUME_SCALP', 'AI_REJECTED', direction, _conf, _rsn)
                    add_to_ai_rejection_cooldown(symbol, direction)
                    continue
                
                add_symbol_cooldown(symbol)
                
                return {
                    'symbol': symbol,
                    'direction': direction,
                    'entry_price': current_price,
                    'stop_loss': stop_loss,
                    'take_profit': take_profit,
                    'take_profit_1': take_profit,
                    'take_profit_2': None,
                    'take_profit_3': None,
                    'tp_percent': tp_percent,
                    'sl_percent': sl_percent,
                    'confidence': min(int(volume_ratio * 3), 10),
                    'reasoning': ai_result.get('reasoning', ''),
                    'ai_confidence': ai_result.get('ai_confidence', 5),
                    'ai_recommendation': ai_result.get('recommendation', 'BUY'),
                    'trade_explainer': ai_result.get('trade_explainer', ''),
                    'trade_type': 'VOLUME_SCALP',
                    'strategy': 'VOLUME_SCALP',
                    'risk_level': 'SCALP',
                    'galaxy_score': 0,
                    'sentiment': 0.5,
                    'social_volume': 0,
                    'social_interactions': 0,
                    'social_dominance': 0,
                    'alt_rank': 9999,
                    'coin_name': symbol.replace('USDT', ''),
                    'rsi': rsi,
                    '24h_change': change,
                    '24h_volume': volume_24h,
                    'derivatives': derivatives,
                    'deriv_adjustments': deriv_adjustments,
                    'social_strength': 0,
                    'social_vol_change': 0,
                    'volume_ratio': volume_ratio,
                    'btc_correlation': price_data.get('btc_correlation', 0.0),
                    'influencer_consensus': None,
                    'buzz_momentum': None,
                    'enhanced_ta': enhanced_ta,
                    'is_scalp': True,
                }
            
            logger.info("⚡ No volume scalps passed all checks")
            return None
            
        except Exception as e:
            logger.error(f"Volume scalp scanner error: {e}")
            return None

    async def scan_for_relief_bounce(self) -> Optional[Dict]:
        """
        Scan for TOP LOSER RELIEF BOUNCE longs.
        Finds coins down -10% or more on 24h that show signs of bouncing:
        - RSI not overbought (<50)
        - Price bouncing off daily low (current price > low by 1%+)
        - Volume still present (not dead coins)
        - AI approval required
        
        These are contrarian LONG plays catching the dead cat bounce / relief rally.
        Tighter TP (2.5-5%) and tight SL since these are risky reversal plays.
        """
        global _daily_social_signals
        
        reset_daily_counters_if_needed()
        if _daily_social_signals >= MAX_DAILY_SOCIAL_SIGNALS:
            return None
        
        await self.init()
        
        try:
            losers = []
            bitunix_symbols = set()
            binance_data = {}
            
            try:
                tickers = await self._get_binance_tickers()
                if tickers:
                    for t in tickers:
                        sym = t.get('symbol', '')
                        if not sym.endswith('USDT') or sym in ('BTCUSDT', 'ETHUSDT', 'USDCUSDT'):
                            continue
                        try:
                            binance_data[sym] = {
                                'change': float(t.get('priceChangePercent', 0)),
                                'vol': float(t.get('quoteVolume', 0)),
                                'price': float(t.get('lastPrice', 0)),
                                'low': float(t.get('lowPrice', 0)),
                                'high': float(t.get('highPrice', 0)),
                            }
                        except (ValueError, TypeError):
                            continue
                    logger.info(f"📉 Binance 24h data loaded: {len(binance_data)} pairs")
            except Exception as e:
                logger.debug(f"Binance relief bounce fetch failed: {e}")
            
            try:
                bitunix_url = "https://fapi.bitunix.com/api/v1/futures/market/tickers"
                resp = await self.http_client.get(bitunix_url, timeout=8)
                if resp.status_code == 200:
                    for t in resp.json().get('data', []):
                        sym = t.get('symbol', '')
                        if sym.endswith('USDT'):
                            bitunix_symbols.add(sym)
                    logger.info(f"📉 Bitunix tradeable symbols: {len(bitunix_symbols)}")
            except Exception as e:
                logger.debug(f"Bitunix relief bounce fetch failed: {e}")
            
            for sym, bd in binance_data.items():
                if sym not in bitunix_symbols:
                    continue
                
                change = bd['change']
                vol = bd['vol']
                last_price = bd['price']
                low_price = bd['low']
                high_price = bd['high']
                
                if change > -10 or vol < 300_000 or last_price <= 0 or low_price <= 0:
                    continue
                
                bounce_from_low = ((last_price - low_price) / low_price * 100) if low_price > 0 else 0
                drop_from_high = ((high_price - last_price) / high_price * 100) if high_price > 0 else 0
                
                losers.append({
                    'symbol': sym,
                    'change_24h': change,
                    'volume_24h': vol,
                    'price': last_price,
                    'high': high_price,
                    'low': low_price,
                    'bounce_from_low': bounce_from_low,
                    'drop_from_high': drop_from_high,
                })
            
            losers.sort(key=lambda x: x['bounce_from_low'], reverse=True)
            losers = losers[:20]
            
            if not losers:
                logger.info("📉 RELIEF BOUNCE: No top losers (-10%+) found on Binance that are tradeable on Bitunix")
                return None
            
            logger.info(f"📉 RELIEF BOUNCE SCANNER: {len(losers)} coins down -10%+ with volume (Binance 24h, Bitunix tradeable)")
            btc_state = await get_btc_state()
            logger.info(f"📉 RELIEF BOUNCE BTC → {btc_state['summary']}")

            for loser in losers:
                symbol = loser['symbol']
                change = loser['change_24h']
                vol = loser['volume_24h']
                bounce_pct = loser['bounce_from_low']


                if is_symbol_on_cooldown(symbol) or is_coin_in_signalled_cooldown(symbol):
                    continue
                
                is_available = await self.check_bitunix_availability(symbol)
                if not is_available:
                    continue
                
                price_data = await self.fetch_price_data(symbol)
                if not price_data:
                    continue
                
                rsi = price_data.get('rsi', 50)
                current_price = price_data['price']
                
                if rsi > 50:
                    logger.info(f"  📉 {symbol} {change:.1f}% - RSI {rsi:.0f} not oversold enough for relief bounce (need <50)")
                    continue
                
                if bounce_pct < 1.0:
                    logger.info(f"  📉 {symbol} {change:.1f}% - Only {bounce_pct:.1f}% off low, no bounce yet")
                    continue
                
                logger.info(f"  📉 RELIEF CANDIDATE: {symbol} | 24h {change:.1f}% | RSI {rsi:.0f} | Bounce {bounce_pct:.1f}% from low | Vol ${vol/1e6:.1f}M")
                
                direction = 'LONG'
                abs_change = abs(change)
                
                if abs_change >= 40:
                    base_tp = 5.0
                    base_sl = 2.5
                elif abs_change >= 30:
                    base_tp = 4.0
                    base_sl = 2.0
                elif abs_change >= 20:
                    base_tp = 3.0
                    base_sl = 1.5
                else:
                    base_tp = 2.5
                    base_sl = 1.2
                
                enhanced_ta = price_data.get('enhanced_ta', {})
                
                tp_percent = base_tp
                sl_percent = base_sl
                deriv_adjustments = []
                derivatives = {}
                
                if enhanced_ta:
                    from app.services.enhanced_ta import get_atr_based_tp_sl, optimize_tp_sl_from_chart_levels
                    tp_percent, sl_percent = get_atr_based_tp_sl(enhanced_ta, direction, tp_percent, sl_percent)
                    old_tp, old_sl = tp_percent, sl_percent
                    tp_percent, sl_percent = optimize_tp_sl_from_chart_levels(enhanced_ta, direction, current_price, tp_percent, sl_percent)
                    if tp_percent != old_tp or sl_percent != old_sl:
                        logger.info(f"📊 {symbol} Chart-optimized TP/SL: TP {old_tp:.1f}%→{tp_percent:.1f}% | SL {old_sl:.1f}%→{sl_percent:.1f}%")
                
                take_profit = current_price * (1 + tp_percent / 100)
                stop_loss = current_price * (1 - sl_percent / 100)
                tp2 = current_price * (1 + (tp_percent * 1.5) / 100)
                tp3 = current_price * (1 + (tp_percent * 2.0) / 100)
                
                if derivatives and derivatives.get('has_data'):
                    funding = derivatives.get('funding_rate', 0) or 0
                    if funding > 0.05:
                        logger.info(f"  📉 {symbol} - Extreme positive funding {funding:.4f}%, skip relief long")
                        continue
                
                signal_candidate = {
                    'symbol': symbol,
                    'direction': direction,
                    'entry_price': current_price,
                    'stop_loss': stop_loss,
                    'take_profit': take_profit,
                    'take_profit_1': take_profit,
                    'take_profit_2': tp2,
                    'take_profit_3': tp3,
                    'tp_percent': tp_percent,
                    'sl_percent': sl_percent,
                    'rsi': rsi,
                    '24h_change': change,
                    '24h_volume': vol,
                    'volume_ratio': price_data.get('volume_ratio', 1.0),
                    'btc_correlation': price_data.get('btc_correlation', 0.0),
                    'derivatives': derivatives,
                    'deriv_adjustments': deriv_adjustments,
                    'enhanced_ta': enhanced_ta,
                    'bounce_from_low': bounce_pct,
                    'drop_from_high': loser['drop_from_high'],
                }

                try:
                    from app.services.order_flow import analyze_order_flow
                    of = await analyze_order_flow(symbol=symbol, price_change_1h=0, price_change_24h=change, current_price=current_price, volume_24h=vol, ta_data=enhanced_ta)
                    if of and abs(of.get('flow_score', 0)) >= 15:
                        signal_candidate['order_flow'] = of
                except Exception as e:
                    logger.debug(f"Order flow failed for relief {symbol}: {e}")

                if is_coin_in_ai_rejection_cooldown(symbol, direction):
                    continue
                
                ai_result = await ai_analyze_social_signal(signal_candidate)
                
                if not ai_result.get('approved', True) or ai_result.get('ai_confidence', 5) < 6:
                    _conf = ai_result.get('ai_confidence', 5)
                    _rsn = ai_result.get('reasoning', '')
                    logger.info(f"🤖 AI REJECTED relief bounce {symbol}: conf={_conf}/10 | {_rsn}")
                    log_rejection(symbol, 'RELIEF_BOUNCE', 'AI_REJECTED', direction, _conf, _rsn)
                    add_to_ai_rejection_cooldown(symbol, direction)
                    continue
                
                add_symbol_cooldown(symbol)
                
                logger.info(f"📉 RELIEF BOUNCE APPROVED: {symbol} | {change:.1f}% | RSI {rsi:.0f} | Bounce {bounce_pct:.1f}% | TP {tp_percent:.1f}% SL {sl_percent:.1f}%")
                
                return {
                    'symbol': symbol,
                    'direction': direction,
                    'entry_price': current_price,
                    'stop_loss': stop_loss,
                    'take_profit': take_profit,
                    'take_profit_1': take_profit,
                    'take_profit_2': tp2,
                    'take_profit_3': tp3,
                    'tp_percent': tp_percent,
                    'sl_percent': sl_percent,
                    'confidence': min(int(abs_change / 5), 10),
                    'reasoning': ai_result.get('reasoning', ''),
                    'ai_confidence': ai_result.get('ai_confidence', 5),
                    'ai_recommendation': ai_result.get('recommendation', 'BUY'),
                    'trade_explainer': ai_result.get('trade_explainer', ''),
                    'trade_type': 'RELIEF_BOUNCE',
                    'strategy': 'RELIEF_BOUNCE',
                    'risk_level': 'RELIEF',
                    'coin_name': symbol.replace('USDT', ''),
                    'rsi': rsi,
                    '24h_change': change,
                    '24h_volume': vol,
                    'derivatives': derivatives,
                    'deriv_adjustments': deriv_adjustments,
                    'volume_ratio': price_data.get('volume_ratio', 1.0),
                    'btc_correlation': price_data.get('btc_correlation', 0.0),
                    'bounce_from_low': bounce_pct,
                    'enhanced_ta': enhanced_ta,
                }
            
            logger.info("📉 No relief bounce candidates passed all checks")
            return None
            
        except Exception as e:
            logger.error(f"Relief bounce scanner error: {e}")
            return None

    async def scan_for_squeeze_breakout(self) -> Optional[Dict]:
        """
        Scan for BB/Keltner squeeze breakout trades.
        Detects squeeze release (BB inside KC) with volume confirmation.
        ATR-based TP/SL, 2-3% targets capped at 4%.
        Target: 4 signals per day.
        """
        global _daily_squeeze_signals, _last_squeeze_time

        reset_daily_counters_if_needed()
        if _daily_squeeze_signals >= MAX_DAILY_SQUEEZE_SIGNALS:
            logger.debug(f"🔥 Daily squeeze limit reached ({MAX_DAILY_SQUEEZE_SIGNALS})")
            return None

        if _last_squeeze_time:
            elapsed = (datetime.now() - _last_squeeze_time).total_seconds() / 60
            if elapsed < MIN_SQUEEZE_GAP_MINUTES:
                logger.debug(f"🔥 Squeeze gap: {elapsed:.0f}min (need {MIN_SQUEEZE_GAP_MINUTES}min)")
                return None

        await self.init()

        try:
            tickers = None

            try:
                tickers = await self._get_binance_tickers()
            except Exception:
                pass

            if not tickers:
                try:
                    mexc_url = "https://contract.mexc.com/api/v1/contract/ticker"
                    resp = await self.http_client.get(mexc_url, timeout=8)
                    if resp.status_code == 200:
                        mexc_data = resp.json()
                        raw_tickers = mexc_data.get('data', [])
                        tickers = []
                        for t in raw_tickers:
                            sym = t.get('symbol', '')
                            if not sym.endswith('_USDT'):
                                continue
                            tickers.append({
                                'symbol': sym.replace('_USDT', 'USDT'),
                                'priceChangePercent': str(float(t.get('riseFallRate', 0)) * 100),
                                'quoteVolume': str(t.get('amount24', 0) or 0),
                                'lastPrice': str(t.get('lastPrice', 0)),
                                'highPrice': str(t.get('high24Price', 0) or 0),
                                'lowPrice': str(t.get('low24Price', 0) or 0),
                            })
                except Exception:
                    pass

            if not tickers:
                return None

            candidates = []
            for t in tickers:
                sym = t.get('symbol', '')
                if not sym.endswith('USDT') or sym in ('BTCUSDT', 'ETHUSDT', 'USDCUSDT', 'BNBUSDT'):
                    continue

                change = float(t.get('priceChangePercent', 0))
                vol = float(t.get('quoteVolume', 0))
                last_price = float(t.get('lastPrice', 0))

                if vol < 3_000_000:
                    continue
                if abs(change) > 15 or abs(change) < 1:
                    continue
                if last_price <= 0:
                    continue

                candidates.append({
                    'symbol': sym,
                    'change_24h': change,
                    'volume_24h': vol,
                    'price': last_price,
                })

            if not candidates:
                logger.debug("🔥 SQUEEZE: No candidates with sufficient volume")
                return None

            candidates.sort(key=lambda x: x['volume_24h'], reverse=True)
            candidates = candidates[:30]

            logger.info(f"🔥 SQUEEZE SCANNER: {len(candidates)} candidates")
            btc_state = await get_btc_state()
            logger.info(f"🔥 SQUEEZE BTC → {btc_state['summary']}")

            for c in candidates:
                symbol = c['symbol']


                if is_symbol_on_cooldown(symbol) or is_coin_in_signalled_cooldown(symbol):
                    continue

                is_available = await self.check_bitunix_availability(symbol)
                if not is_available:
                    continue

                price_data = await self.fetch_price_data(symbol)
                if not price_data:
                    continue

                rsi = price_data.get('rsi', 50)
                current_price = price_data['price']
                volume_ratio = price_data.get('volume_ratio', 1.0)
                volume_24h = price_data.get('volume_24h', 0)
                enhanced_ta = price_data.get('enhanced_ta', {})
                change = c['change_24h']

                if volume_ratio < 1.5:
                    continue

                try:
                    klines_url = f"https://fapi.binance.com/fapi/v1/klines?symbol={symbol}&interval=15m&limit=50"
                    kl_resp = await self.http_client.get(klines_url, timeout=8)
                    if kl_resp.status_code != 200:
                        continue
                    klines = kl_resp.json()
                    if len(klines) < 30:
                        continue
                    kl_closes = [float(k[4]) for k in klines]
                    kl_highs = [float(k[2]) for k in klines]
                    kl_lows = [float(k[3]) for k in klines]
                except Exception:
                    continue

                from app.services.enhanced_ta import calc_squeeze
                squeeze = calc_squeeze(kl_closes, kl_highs, kl_lows,
                                       bb_period=20, bb_mult=1.5,
                                       kc_ema=20, kc_atr=10, kc_mult=1.8)
                if not squeeze or not squeeze.get('squeeze_release'):
                    continue

                sq_direction = squeeze.get('direction', 'NEUTRAL')
                if sq_direction == 'BULLISH' and 48 <= rsi <= 80:
                    direction = 'LONG'
                elif sq_direction == 'BEARISH' and 20 <= rsi <= 52:
                    direction = 'SHORT'
                else:
                    continue

                base_tp = 2.5
                base_sl = 2.5

                if volume_ratio >= 3.0:
                    base_tp = 3.0
                    base_sl = 3.0
                elif volume_ratio >= 2.5:
                    base_tp = 2.8
                    base_sl = 2.8

                tp_percent = base_tp
                sl_percent = base_sl
                deriv_adjustments = []
                derivatives = {}

                tp_percent = min(tp_percent, 4.0)
                sl_percent = min(sl_percent, 4.0)

                max_diff = max(tp_percent, sl_percent)
                tp_percent = max_diff
                sl_percent = max_diff

                if enhanced_ta:
                    from app.services.enhanced_ta import get_atr_based_tp_sl
                    atr_tp, atr_sl = get_atr_based_tp_sl(enhanced_ta, direction, tp_percent, sl_percent)
                    avg_target = (atr_tp + atr_sl) / 2
                    tp_percent = min(avg_target, 4.0)
                    sl_percent = tp_percent

                if direction == 'LONG':
                    take_profit = current_price * (1 + tp_percent / 100)
                    stop_loss = current_price * (1 - sl_percent / 100)
                else:
                    take_profit = current_price * (1 - tp_percent / 100)
                    stop_loss = current_price * (1 + sl_percent / 100)

                logger.info(f"🔥 SQUEEZE: {symbol} {direction} | Vol {volume_ratio:.1f}x | RSI {rsi:.0f} | TP/SL {tp_percent:.1f}%/{sl_percent:.1f}%")

                signal_candidate = {
                    'symbol': symbol,
                    'direction': direction,
                    'entry_price': current_price,
                    'stop_loss': stop_loss,
                    'take_profit': take_profit,
                    'take_profit_1': take_profit,
                    'take_profit_2': None,
                    'take_profit_3': None,
                    'tp_percent': tp_percent,
                    'sl_percent': sl_percent,
                    'rsi': rsi,
                    '24h_change': change,
                    '24h_volume': volume_24h,
                    'volume_ratio': volume_ratio,
                    'btc_correlation': price_data.get('btc_correlation', 0.0),
                    'derivatives': derivatives,
                    'deriv_adjustments': deriv_adjustments,
                    'enhanced_ta': enhanced_ta,
                    'trade_type': 'SQUEEZE_BREAKOUT',
                }

                if is_coin_in_ai_rejection_cooldown(symbol, direction):
                    continue

                ai_result = await ai_analyze_social_signal(signal_candidate)

                if not ai_result.get('approved', True) or ai_result.get('ai_confidence', 5) < 6:
                    _conf = ai_result.get('ai_confidence', 5)
                    _rsn = ai_result.get('reasoning', '')
                    logger.info(f"🤖 AI REJECTED squeeze {symbol}: conf={_conf}/10 | {_rsn}")
                    log_rejection(symbol, 'SQUEEZE', 'AI_REJECTED', direction, _conf, _rsn)
                    add_to_ai_rejection_cooldown(symbol, direction)
                    continue

                add_symbol_cooldown(symbol)

                return {
                    'symbol': symbol,
                    'direction': direction,
                    'entry_price': current_price,
                    'stop_loss': stop_loss,
                    'take_profit': take_profit,
                    'take_profit_1': take_profit,
                    'take_profit_2': None,
                    'take_profit_3': None,
                    'tp_percent': tp_percent,
                    'sl_percent': sl_percent,
                    'confidence': min(int(volume_ratio * 3), 10),
                    'reasoning': ai_result.get('reasoning', ''),
                    'ai_confidence': ai_result.get('ai_confidence', 5),
                    'ai_recommendation': ai_result.get('recommendation', 'BUY'),
                    'trade_explainer': ai_result.get('trade_explainer', ''),
                    'trade_type': 'SQUEEZE_BREAKOUT',
                    'strategy': 'SQUEEZE_BREAKOUT',
                    'risk_level': 'SCALP',
                    'galaxy_score': 0,
                    'sentiment': 0.5,
                    'social_volume': 0,
                    'social_interactions': 0,
                    'social_dominance': 0,
                    'alt_rank': 9999,
                    'coin_name': symbol.replace('USDT', ''),
                    'rsi': rsi,
                    '24h_change': change,
                    '24h_volume': volume_24h,
                    'derivatives': derivatives,
                    'deriv_adjustments': deriv_adjustments,
                    'social_strength': 0,
                    'social_vol_change': 0,
                    'volume_ratio': volume_ratio,
                    'btc_correlation': price_data.get('btc_correlation', 0.0),
                    'influencer_consensus': None,
                    'buzz_momentum': None,
                    'enhanced_ta': enhanced_ta,
                    'is_scalp': True,
                }

            logger.info("🔥 No squeeze breakouts passed all checks")
            return None

        except Exception as e:
            logger.error(f"Squeeze breakout scanner error: {e}")
            return None

    async def scan_for_supertrend(self) -> Optional[Dict]:
        """
        Scan for SuperTrend trend flip trades on 15m candles.
        Fires on BUY/SELL trend flips with EMA ribbon alignment and RVOL confirmation.
        ATR-based TP/SL with 1:1.5 R:R, 2-4% targets.
        Target: 4 signals per day.
        """
        global _daily_supertrend_signals, _last_supertrend_time

        reset_daily_counters_if_needed()
        if _daily_supertrend_signals >= MAX_DAILY_SUPERTREND_SIGNALS:
            logger.debug(f"📈 Daily supertrend limit reached ({MAX_DAILY_SUPERTREND_SIGNALS})")
            return None

        if _last_supertrend_time:
            elapsed = (datetime.now() - _last_supertrend_time).total_seconds() / 60
            if elapsed < MIN_SUPERTREND_GAP_MINUTES:
                logger.debug(f"📈 Supertrend gap: {elapsed:.0f}min (need {MIN_SUPERTREND_GAP_MINUTES}min)")
                return None

        await self.init()

        try:
            tickers = None

            try:
                tickers = await self._get_binance_tickers()
            except Exception:
                pass

            if not tickers:
                try:
                    mexc_url = "https://contract.mexc.com/api/v1/contract/ticker"
                    resp = await self.http_client.get(mexc_url, timeout=8)
                    if resp.status_code == 200:
                        mexc_data = resp.json()
                        raw_tickers = mexc_data.get('data', [])
                        tickers = []
                        for t in raw_tickers:
                            sym = t.get('symbol', '')
                            if not sym.endswith('_USDT'):
                                continue
                            tickers.append({
                                'symbol': sym.replace('_USDT', 'USDT'),
                                'priceChangePercent': str(float(t.get('riseFallRate', 0)) * 100),
                                'quoteVolume': str(t.get('amount24', 0) or 0),
                                'lastPrice': str(t.get('lastPrice', 0)),
                                'highPrice': str(t.get('high24Price', 0) or 0),
                                'lowPrice': str(t.get('low24Price', 0) or 0),
                            })
                except Exception:
                    pass

            if not tickers:
                return None

            candidates = []
            for t in tickers:
                sym = t.get('symbol', '')
                if not sym.endswith('USDT') or sym in ('BTCUSDT', 'ETHUSDT', 'USDCUSDT', 'BNBUSDT'):
                    continue

                change = float(t.get('priceChangePercent', 0))
                vol = float(t.get('quoteVolume', 0))
                last_price = float(t.get('lastPrice', 0))

                if vol < 3_000_000:
                    continue
                if abs(change) > 15 or abs(change) < 1:
                    continue
                if last_price <= 0:
                    continue

                candidates.append({
                    'symbol': sym,
                    'change_24h': change,
                    'volume_24h': vol,
                    'price': last_price,
                })

            if not candidates:
                logger.debug("📈 SUPERTREND: No candidates with sufficient volume")
                return None

            candidates.sort(key=lambda x: x['volume_24h'], reverse=True)
            candidates = candidates[:30]

            logger.info(f"📈 SUPERTREND SCANNER: {len(candidates)} candidates")
            btc_state = await get_btc_state()
            logger.info(f"📈 SUPERTREND BTC → {btc_state['summary']}")

            for c in candidates:
                symbol = c['symbol']


                if is_symbol_on_cooldown(symbol) or is_coin_in_signalled_cooldown(symbol):
                    continue

                is_available = await self.check_bitunix_availability(symbol)
                if not is_available:
                    continue

                price_data = await self.fetch_price_data(symbol)
                if not price_data:
                    continue

                rsi = price_data.get('rsi', 50)
                current_price = price_data['price']
                volume_ratio = price_data.get('volume_ratio', 1.0)
                volume_24h = price_data.get('volume_24h', 0)
                enhanced_ta = price_data.get('enhanced_ta', {})
                change = c['change_24h']

                if volume_ratio < 1.1:
                    continue

                try:
                    klines_url = f"https://fapi.binance.com/fapi/v1/klines?symbol={symbol}&interval=15m&limit=50"
                    kl_resp = await self.http_client.get(klines_url, timeout=8)
                    if kl_resp.status_code != 200:
                        continue
                    klines = kl_resp.json()
                    if len(klines) < 30:
                        continue
                    kl_closes = [float(k[4]) for k in klines]
                    kl_highs = [float(k[2]) for k in klines]
                    kl_lows = [float(k[3]) for k in klines]
                except Exception:
                    continue

                from app.services.enhanced_ta import calc_supertrend, calc_ema_ribbon
                st = calc_supertrend(kl_highs, kl_lows, kl_closes, atr_period=10, factor=3.0)
                if not st or not st.get('trend_flip'):
                    continue

                ribbon = calc_ema_ribbon(kl_closes, [8, 21, 34])

                st_signal = st.get('signal', '')
                if st_signal == 'BUY' and 35 <= rsi <= 75:
                    if not ribbon or not ribbon.get('bullish_aligned'):
                        continue
                    direction = 'LONG'
                elif st_signal == 'SELL' and 25 <= rsi <= 65:
                    if not ribbon or not ribbon.get('bearish_aligned'):
                        continue
                    direction = 'SHORT'
                else:
                    continue

                base_sl = 2.5
                base_tp = base_sl * 1.5

                if volume_ratio >= 2.5:
                    base_sl = 3.0
                    base_tp = base_sl * 1.5

                tp_percent = base_tp
                sl_percent = base_sl
                deriv_adjustments = []
                derivatives = {}

                tp_percent = min(tp_percent, 4.0)
                sl_percent = min(sl_percent, 4.0)

                tp_percent = max(tp_percent, sl_percent * 1.5)
                tp_percent = min(tp_percent, 4.0)

                if enhanced_ta:
                    from app.services.enhanced_ta import get_atr_based_tp_sl
                    atr_tp, atr_sl = get_atr_based_tp_sl(enhanced_ta, direction, tp_percent, sl_percent)
                    tp_percent = min(atr_tp, 4.0)
                    sl_percent = min(atr_sl, 4.0)
                    tp_percent = max(tp_percent, sl_percent * 1.5)
                    tp_percent = min(tp_percent, 4.0)

                if direction == 'LONG':
                    take_profit = current_price * (1 + tp_percent / 100)
                    stop_loss = current_price * (1 - sl_percent / 100)
                else:
                    take_profit = current_price * (1 - tp_percent / 100)
                    stop_loss = current_price * (1 + sl_percent / 100)

                logger.info(f"📈 SUPERTREND: {symbol} {direction} | Vol {volume_ratio:.1f}x | RSI {rsi:.0f} | TP {tp_percent:.1f}% SL {sl_percent:.1f}%")

                signal_candidate = {
                    'symbol': symbol,
                    'direction': direction,
                    'entry_price': current_price,
                    'stop_loss': stop_loss,
                    'take_profit': take_profit,
                    'take_profit_1': take_profit,
                    'take_profit_2': None,
                    'take_profit_3': None,
                    'tp_percent': tp_percent,
                    'sl_percent': sl_percent,
                    'rsi': rsi,
                    '24h_change': change,
                    '24h_volume': volume_24h,
                    'volume_ratio': volume_ratio,
                    'btc_correlation': price_data.get('btc_correlation', 0.0),
                    'derivatives': derivatives,
                    'deriv_adjustments': deriv_adjustments,
                    'enhanced_ta': enhanced_ta,
                    'trade_type': 'SUPERTREND',
                }

                if is_coin_in_ai_rejection_cooldown(symbol, direction):
                    continue

                ai_result = await ai_analyze_social_signal(signal_candidate)

                if not ai_result.get('approved', True) or ai_result.get('ai_confidence', 5) < 6:
                    _conf = ai_result.get('ai_confidence', 5)
                    _rsn = ai_result.get('reasoning', '')
                    logger.info(f"🤖 AI REJECTED supertrend {symbol}: conf={_conf}/10 | {_rsn}")
                    log_rejection(symbol, 'SUPERTREND', 'AI_REJECTED', direction, _conf, _rsn)
                    add_to_ai_rejection_cooldown(symbol, direction)
                    continue

                add_symbol_cooldown(symbol)

                return {
                    'symbol': symbol,
                    'direction': direction,
                    'entry_price': current_price,
                    'stop_loss': stop_loss,
                    'take_profit': take_profit,
                    'take_profit_1': take_profit,
                    'take_profit_2': None,
                    'take_profit_3': None,
                    'tp_percent': tp_percent,
                    'sl_percent': sl_percent,
                    'confidence': min(int(volume_ratio * 3), 10),
                    'reasoning': ai_result.get('reasoning', ''),
                    'ai_confidence': ai_result.get('ai_confidence', 5),
                    'ai_recommendation': ai_result.get('recommendation', 'BUY'),
                    'trade_explainer': ai_result.get('trade_explainer', ''),
                    'trade_type': 'SUPERTREND',
                    'strategy': 'SUPERTREND',
                    'risk_level': 'SCALP',
                    'galaxy_score': 0,
                    'sentiment': 0.5,
                    'social_volume': 0,
                    'social_interactions': 0,
                    'social_dominance': 0,
                    'alt_rank': 9999,
                    'coin_name': symbol.replace('USDT', ''),
                    'rsi': rsi,
                    '24h_change': change,
                    '24h_volume': volume_24h,
                    'derivatives': derivatives,
                    'deriv_adjustments': deriv_adjustments,
                    'social_strength': 0,
                    'social_vol_change': 0,
                    'volume_ratio': volume_ratio,
                    'btc_correlation': price_data.get('btc_correlation', 0.0),
                    'influencer_consensus': None,
                    'buzz_momentum': None,
                    'enhanced_ta': enhanced_ta,
                    'is_scalp': True,
                }

            logger.info("📈 No supertrend signals passed all checks")
            return None

        except Exception as e:
            logger.error(f"Supertrend scanner error: {e}")
            return None

    async def scan_for_macd_momentum(self) -> Optional[Dict]:
        """
        Scan for MACD crossover momentum trades on 15m candles.
        Fires on BULLISH_CROSS/BEARISH_CROSS with EMA ribbon alignment and volume.
        TP/SL: 2-3%, 1:1 R:R.
        Target: 4 signals per day.
        """
        global _daily_macd_signals, _last_macd_time

        reset_daily_counters_if_needed()
        if _daily_macd_signals >= MAX_DAILY_MACD_SIGNALS:
            logger.debug(f"📊 Daily MACD limit reached ({MAX_DAILY_MACD_SIGNALS})")
            return None

        if _last_macd_time:
            elapsed = (datetime.now() - _last_macd_time).total_seconds() / 60
            if elapsed < MIN_MACD_GAP_MINUTES:
                logger.debug(f"📊 MACD gap: {elapsed:.0f}min (need {MIN_MACD_GAP_MINUTES}min)")
                return None

        await self.init()

        try:
            tickers = None

            try:
                tickers = await self._get_binance_tickers()
            except Exception:
                pass

            if not tickers:
                try:
                    mexc_url = "https://contract.mexc.com/api/v1/contract/ticker"
                    resp = await self.http_client.get(mexc_url, timeout=8)
                    if resp.status_code == 200:
                        mexc_data = resp.json()
                        raw_tickers = mexc_data.get('data', [])
                        tickers = []
                        for t in raw_tickers:
                            sym = t.get('symbol', '')
                            if not sym.endswith('_USDT'):
                                continue
                            tickers.append({
                                'symbol': sym.replace('_USDT', 'USDT'),
                                'priceChangePercent': str(float(t.get('riseFallRate', 0)) * 100),
                                'quoteVolume': str(t.get('amount24', 0) or 0),
                                'lastPrice': str(t.get('lastPrice', 0)),
                                'highPrice': str(t.get('high24Price', 0) or 0),
                                'lowPrice': str(t.get('low24Price', 0) or 0),
                            })
                except Exception:
                    pass

            if not tickers:
                return None

            candidates = []
            for t in tickers:
                sym = t.get('symbol', '')
                if not sym.endswith('USDT') or sym in ('BTCUSDT', 'ETHUSDT', 'USDCUSDT', 'BNBUSDT'):
                    continue

                change = float(t.get('priceChangePercent', 0))
                vol = float(t.get('quoteVolume', 0))
                last_price = float(t.get('lastPrice', 0))

                if vol < 3_000_000:
                    continue
                if abs(change) > 15 or abs(change) < 1:
                    continue
                if last_price <= 0:
                    continue

                candidates.append({
                    'symbol': sym,
                    'change_24h': change,
                    'volume_24h': vol,
                    'price': last_price,
                })

            if not candidates:
                logger.debug("📊 MACD: No candidates with sufficient volume")
                return None

            candidates.sort(key=lambda x: x['volume_24h'], reverse=True)
            candidates = candidates[:30]

            logger.info(f"📊 MACD SCANNER: {len(candidates)} candidates")
            btc_state = await get_btc_state()
            logger.info(f"📊 MACD BTC → {btc_state['summary']}")

            for c in candidates:
                symbol = c['symbol']


                if is_symbol_on_cooldown(symbol) or is_coin_in_signalled_cooldown(symbol):
                    continue

                is_available = await self.check_bitunix_availability(symbol)
                if not is_available:
                    continue

                price_data = await self.fetch_price_data(symbol)
                if not price_data:
                    continue

                rsi = price_data.get('rsi', 50)
                current_price = price_data['price']
                volume_ratio = price_data.get('volume_ratio', 1.0)
                volume_24h = price_data.get('volume_24h', 0)
                enhanced_ta = price_data.get('enhanced_ta', {})
                change = c['change_24h']

                if volume_ratio < 1.0:
                    continue

                try:
                    klines_url = f"https://fapi.binance.com/fapi/v1/klines?symbol={symbol}&interval=15m&limit=50"
                    kl_resp = await self.http_client.get(klines_url, timeout=8)
                    if kl_resp.status_code != 200:
                        continue
                    klines = kl_resp.json()
                    if len(klines) < 30:
                        continue
                    kl_closes = [float(k[4]) for k in klines]
                except Exception:
                    continue

                from app.services.enhanced_ta import calc_macd, calc_ema_ribbon
                macd_data = calc_macd(kl_closes, fast=8, slow=21, signal=5)
                if not macd_data:
                    continue

                crossover = macd_data.get('crossover', '')
                ribbon = calc_ema_ribbon(kl_closes, [8, 21, 34])

                if crossover == 'BULLISH_CROSS' and 40 <= rsi <= 75:
                    if not ribbon or not ribbon.get('bullish_aligned'):
                        continue
                    direction = 'LONG'
                elif crossover == 'BEARISH_CROSS' and 25 <= rsi <= 60:
                    if not ribbon or not ribbon.get('bearish_aligned'):
                        continue
                    direction = 'SHORT'
                else:
                    continue

                base_tp = 2.5
                base_sl = 2.5

                if volume_ratio >= 2.5:
                    base_tp = 3.0
                    base_sl = 3.0

                tp_percent = base_tp
                sl_percent = base_sl
                deriv_adjustments = []
                derivatives = {}

                tp_percent = min(tp_percent, 4.0)
                sl_percent = min(sl_percent, 4.0)

                max_diff = max(tp_percent, sl_percent)
                tp_percent = max_diff
                sl_percent = max_diff

                if enhanced_ta:
                    from app.services.enhanced_ta import get_atr_based_tp_sl
                    atr_tp, atr_sl = get_atr_based_tp_sl(enhanced_ta, direction, tp_percent, sl_percent)
                    avg_target = (atr_tp + atr_sl) / 2
                    tp_percent = min(avg_target, 4.0)
                    sl_percent = tp_percent

                if direction == 'LONG':
                    take_profit = current_price * (1 + tp_percent / 100)
                    stop_loss = current_price * (1 - sl_percent / 100)
                else:
                    take_profit = current_price * (1 - tp_percent / 100)
                    stop_loss = current_price * (1 + sl_percent / 100)

                logger.info(f"📊 MACD: {symbol} {direction} | Vol {volume_ratio:.1f}x | RSI {rsi:.0f} | Cross {crossover} | TP/SL {tp_percent:.1f}%/{sl_percent:.1f}%")

                signal_candidate = {
                    'symbol': symbol,
                    'direction': direction,
                    'entry_price': current_price,
                    'stop_loss': stop_loss,
                    'take_profit': take_profit,
                    'take_profit_1': take_profit,
                    'take_profit_2': None,
                    'take_profit_3': None,
                    'tp_percent': tp_percent,
                    'sl_percent': sl_percent,
                    'rsi': rsi,
                    '24h_change': change,
                    '24h_volume': volume_24h,
                    'volume_ratio': volume_ratio,
                    'btc_correlation': price_data.get('btc_correlation', 0.0),
                    'derivatives': derivatives,
                    'deriv_adjustments': deriv_adjustments,
                    'enhanced_ta': enhanced_ta,
                    'trade_type': 'MACD_MOMENTUM',
                }

                if is_coin_in_ai_rejection_cooldown(symbol, direction):
                    continue

                ai_result = await ai_analyze_social_signal(signal_candidate)

                if not ai_result.get('approved', True) or ai_result.get('ai_confidence', 5) < 6:
                    _conf = ai_result.get('ai_confidence', 5)
                    _rsn = ai_result.get('reasoning', '')
                    logger.info(f"🤖 AI REJECTED MACD {symbol}: conf={_conf}/10 | {_rsn}")
                    log_rejection(symbol, 'MACD', 'AI_REJECTED', direction, _conf, _rsn)
                    add_to_ai_rejection_cooldown(symbol, direction)
                    continue

                add_symbol_cooldown(symbol)

                return {
                    'symbol': symbol,
                    'direction': direction,
                    'entry_price': current_price,
                    'stop_loss': stop_loss,
                    'take_profit': take_profit,
                    'take_profit_1': take_profit,
                    'take_profit_2': None,
                    'take_profit_3': None,
                    'tp_percent': tp_percent,
                    'sl_percent': sl_percent,
                    'confidence': min(int(volume_ratio * 3), 10),
                    'reasoning': ai_result.get('reasoning', ''),
                    'ai_confidence': ai_result.get('ai_confidence', 5),
                    'ai_recommendation': ai_result.get('recommendation', 'BUY'),
                    'trade_explainer': ai_result.get('trade_explainer', ''),
                    'trade_type': 'MACD_MOMENTUM',
                    'strategy': 'MACD_MOMENTUM',
                    'risk_level': 'SCALP',
                    'galaxy_score': 0,
                    'sentiment': 0.5,
                    'social_volume': 0,
                    'social_interactions': 0,
                    'social_dominance': 0,
                    'alt_rank': 9999,
                    'coin_name': symbol.replace('USDT', ''),
                    'rsi': rsi,
                    '24h_change': change,
                    '24h_volume': volume_24h,
                    'derivatives': derivatives,
                    'deriv_adjustments': deriv_adjustments,
                    'social_strength': 0,
                    'social_vol_change': 0,
                    'volume_ratio': volume_ratio,
                    'btc_correlation': price_data.get('btc_correlation', 0.0),
                    'influencer_consensus': None,
                    'buzz_momentum': None,
                    'enhanced_ta': enhanced_ta,
                    'is_scalp': True,
                }

            logger.info("📊 No MACD momentum signals passed all checks")
            return None

        except Exception as e:
            logger.error(f"MACD momentum scanner error: {e}")
            return None

    async def scan_for_range_breakout(self) -> Optional[Dict]:
        """
        RANGE_BREAKOUT: Coin consolidates in tight 4h range (<5%), then breaks out with volume.
        Entry on breakout candle close above range high. TP 3-5%, SL 2%.
        """
        global _daily_range_breakout_signals, _last_range_breakout_time
        reset_daily_counters_if_needed()

        if _daily_range_breakout_signals >= MAX_DAILY_RANGE_BREAKOUT_SIGNALS:
            logger.debug(f"📦 Daily range breakout limit reached ({MAX_DAILY_RANGE_BREAKOUT_SIGNALS})")
            return None
        if _last_range_breakout_time:
            elapsed = (datetime.now() - _last_range_breakout_time).total_seconds() / 60
            if elapsed < MIN_RANGE_BREAKOUT_GAP_MINUTES:
                logger.debug(f"📦 Range breakout gap: {elapsed:.0f}min (need {MIN_RANGE_BREAKOUT_GAP_MINUTES}min)")
                return None

        await self.init()
        try:
            tickers = None
            try:
                tickers = await self._get_binance_tickers()
            except Exception:
                pass
            if not tickers:
                return None

            candidates = []
            for t in tickers:
                sym = t.get('symbol', '')
                if not sym.endswith('USDT') or sym in ('BTCUSDT', 'ETHUSDT', 'USDCUSDT', 'BNBUSDT'):
                    continue
                vol = float(t.get('quoteVolume', 0))
                last_price = float(t.get('lastPrice', 0))
                change = float(t.get('priceChangePercent', 0))
                if vol < 5_000_000 or last_price <= 0:
                    continue
                if abs(change) < 2 or abs(change) > 20:
                    continue
                candidates.append({'symbol': sym, 'volume_24h': vol, 'price': last_price, 'change': change})

            candidates.sort(key=lambda x: x['volume_24h'], reverse=True)
            candidates = candidates[:40]

            btc_state = await get_btc_state()
            logger.debug(f"📦 RANGE_BREAKOUT BTC → {btc_state['summary']}")

            for c in candidates:
                symbol = c['symbol']
                if is_symbol_on_cooldown(symbol) or is_coin_in_signalled_cooldown(symbol):
                    continue
                is_available = await self.check_bitunix_availability(symbol)
                if not is_available:
                    continue

                try:
                    klines_url = f"https://fapi.binance.com/fapi/v1/klines?symbol={symbol}&interval=1h&limit=60"
                    kl_resp = await self.http_client.get(klines_url, timeout=8)
                    if kl_resp.status_code != 200:
                        continue
                    klines = kl_resp.json()
                    if len(klines) < 20:
                        continue
                    closes = [float(k[4]) for k in klines]
                    highs  = [float(k[2]) for k in klines]
                    lows   = [float(k[3]) for k in klines]
                    volumes = [float(k[5]) for k in klines]
                except Exception:
                    continue

                # Consolidation: look at the 8-16 candles before the last one
                consol_highs = highs[-17:-1]
                consol_lows  = lows[-17:-1]
                range_high = max(consol_highs)
                range_low  = min(consol_lows)
                range_pct  = (range_high - range_low) / range_low * 100 if range_low > 0 else 99

                if range_pct > 5.0:
                    continue  # Not tight enough

                current_price = closes[-1]
                current_high  = highs[-1]
                avg_vol = sum(volumes[-20:-1]) / 19 if len(volumes) >= 20 else 1
                current_vol = volumes[-1]
                vol_ratio = current_vol / avg_vol if avg_vol > 0 else 1

                # Must break above range high with volume
                if current_high <= range_high * 1.002:
                    continue
                if vol_ratio < 1.5:
                    continue

                # RSI check
                def _rsi(c_list, period=14):
                    if len(c_list) < period + 1:
                        return 50.0
                    gains, losses = [], []
                    for i in range(1, len(c_list)):
                        d = c_list[i] - c_list[i-1]
                        gains.append(max(d, 0))
                        losses.append(max(-d, 0))
                    ag = sum(gains[-period:]) / period
                    al = sum(losses[-period:]) / period
                    if al == 0:
                        return 100.0
                    rs = ag / al
                    return 100 - 100 / (1 + rs)

                rsi = _rsi(closes)
                if rsi < 40 or rsi > 76:
                    continue

                tp_percent = 4.0
                sl_percent = 4.0
                take_profit = current_price * (1 + tp_percent / 100)
                stop_loss   = current_price * (1 - sl_percent / 100)

                price_data = await self.fetch_price_data(symbol)
                volume_24h = price_data.get('volume_24h', c['volume_24h']) if price_data else c['volume_24h']
                enhanced_ta = price_data.get('enhanced_ta', {}) if price_data else {}

                logger.info(f"📦 RANGE_BREAKOUT: {symbol} LONG | Range {range_pct:.1f}% | Vol {vol_ratio:.1f}x | RSI {rsi:.0f}")

                signal_candidate = {
                    'symbol': symbol, 'direction': 'LONG',
                    'entry_price': current_price, 'stop_loss': stop_loss,
                    'take_profit': take_profit, 'take_profit_1': take_profit,
                    'take_profit_2': None, 'take_profit_3': None,
                    'tp_percent': tp_percent, 'sl_percent': sl_percent,
                    'rsi': rsi, '24h_change': c['change'], '24h_volume': volume_24h,
                    'volume_ratio': vol_ratio, 'btc_correlation': 0.0,
                    'derivatives': {}, 'deriv_adjustments': [], 'enhanced_ta': enhanced_ta,
                    'trade_type': 'RANGE_BREAKOUT',
                }

                if is_coin_in_ai_rejection_cooldown(symbol, 'LONG'):
                    continue
                ai_result = await ai_analyze_social_signal(signal_candidate)
                if not ai_result.get('approved', True) or ai_result.get('ai_confidence', 5) < 6:
                    _conf = ai_result.get('ai_confidence', 5)
                    _rsn = ai_result.get('reasoning', '')
                    logger.info(f"🤖 AI REJECTED range breakout {symbol}: conf={_conf}/10")
                    log_rejection(symbol, 'RANGE_BREAKOUT', 'AI_REJECTED', 'LONG', _conf, _rsn)
                    add_to_ai_rejection_cooldown(symbol, 'LONG')
                    continue

                add_symbol_cooldown(symbol)

                return {
                    'symbol': symbol, 'direction': 'LONG',
                    'entry_price': current_price, 'stop_loss': stop_loss,
                    'take_profit': take_profit, 'take_profit_1': take_profit,
                    'take_profit_2': None, 'take_profit_3': None,
                    'tp_percent': tp_percent, 'sl_percent': sl_percent,
                    'confidence': min(int(vol_ratio * 25), 85),
                    'reasoning': ai_result.get('reasoning', ''),
                    'ai_confidence': ai_result.get('ai_confidence', 6),
                    'ai_recommendation': ai_result.get('recommendation', 'BUY'),
                    'trade_explainer': ai_result.get('trade_explainer', ''),
                    'trade_type': 'RANGE_BREAKOUT', 'strategy': 'RANGE_BREAKOUT',
                    'risk_level': 'MEDIUM', 'galaxy_score': 0, 'sentiment': 0.5,
                    'social_volume': 0, 'social_interactions': 0, 'social_dominance': 0,
                    'alt_rank': 9999, 'coin_name': symbol.replace('USDT', ''),
                    'rsi': rsi, '24h_change': c['change'], '24h_volume': volume_24h,
                    'volume_ratio': vol_ratio, 'btc_correlation': 0.0,
                    'derivatives': {}, 'deriv_adjustments': [], 'enhanced_ta': enhanced_ta,
                }

            logger.info("📦 No RANGE_BREAKOUT signals passed all checks")
            return None
        except Exception as e:
            logger.error(f"Range breakout scanner error: {e}")
            return None

    async def scan_for_ema_pullback(self) -> Optional[Dict]:
        """
        EMA_PULLBACK: Strong uptrend (EMA8>EMA21>EMA50 on 15m), price pulls back to EMA21,
        then bounces. Entry at EMA21 touch with RSI 40-60. TP 3-4%, SL 1.5%.
        """
        global _daily_ema_pullback_signals, _last_ema_pullback_time
        reset_daily_counters_if_needed()

        if _daily_ema_pullback_signals >= MAX_DAILY_EMA_PULLBACK_SIGNALS:
            logger.debug(f"📉 Daily EMA pullback limit reached ({MAX_DAILY_EMA_PULLBACK_SIGNALS})")
            return None
        if _last_ema_pullback_time:
            elapsed = (datetime.now() - _last_ema_pullback_time).total_seconds() / 60
            if elapsed < MIN_EMA_PULLBACK_GAP_MINUTES:
                logger.debug(f"📉 EMA pullback gap: {elapsed:.0f}min (need {MIN_EMA_PULLBACK_GAP_MINUTES}min)")
                return None

        await self.init()
        try:
            tickers = None
            try:
                tickers = await self._get_binance_tickers()
            except Exception:
                pass
            if not tickers:
                return None

            candidates = []
            for t in tickers:
                sym = t.get('symbol', '')
                if not sym.endswith('USDT') or sym in ('BTCUSDT', 'ETHUSDT', 'USDCUSDT', 'BNBUSDT'):
                    continue
                vol = float(t.get('quoteVolume', 0))
                last_price = float(t.get('lastPrice', 0))
                change = float(t.get('priceChangePercent', 0))
                if vol < 5_000_000 or last_price <= 0:
                    continue
                if change < 3 or change > 25:
                    continue  # Want uptrending coins only
                candidates.append({'symbol': sym, 'volume_24h': vol, 'price': last_price, 'change': change})

            candidates.sort(key=lambda x: x['volume_24h'], reverse=True)
            candidates = candidates[:40]

            btc_state = await get_btc_state()
            logger.debug(f"📈 EMA_PULLBACK BTC → {btc_state['summary']}")

            def _ema(data, period):
                if len(data) < period:
                    return data[-1] if data else 0
                k = 2 / (period + 1)
                ema = sum(data[:period]) / period
                for v in data[period:]:
                    ema = v * k + ema * (1 - k)
                return ema

            def _rsi(c_list, period=14):
                if len(c_list) < period + 1:
                    return 50.0
                gains, losses = [], []
                for i in range(1, len(c_list)):
                    d = c_list[i] - c_list[i-1]
                    gains.append(max(d, 0))
                    losses.append(max(-d, 0))
                ag = sum(gains[-period:]) / period
                al = sum(losses[-period:]) / period
                if al == 0:
                    return 100.0
                return 100 - 100 / (1 + ag / al)

            for c in candidates:
                symbol = c['symbol']
                if is_symbol_on_cooldown(symbol) or is_coin_in_signalled_cooldown(symbol):
                    continue
                is_available = await self.check_bitunix_availability(symbol)
                if not is_available:
                    continue

                try:
                    klines_url = f"https://fapi.binance.com/fapi/v1/klines?symbol={symbol}&interval=15m&limit=100"
                    kl_resp = await self.http_client.get(klines_url, timeout=8)
                    if kl_resp.status_code != 200:
                        continue
                    klines = kl_resp.json()
                    if len(klines) < 55:
                        continue
                    closes = [float(k[4]) for k in klines]
                    lows   = [float(k[3]) for k in klines]
                except Exception:
                    continue

                ema8  = _ema(closes, 8)
                ema21 = _ema(closes, 21)
                ema50 = _ema(closes, 50)
                current_price = closes[-1]
                current_low   = lows[-1]
                rsi = _rsi(closes)

                # Uptrend confirmation
                if not (ema8 > ema21 > ema50):
                    continue

                # Price must be pulling back — current price near EMA21 (within 1.5%)
                dist_to_ema21 = abs(current_price - ema21) / ema21 * 100
                if dist_to_ema21 > 1.5:
                    continue

                # Candle low must have touched EMA21 (bounce in progress)
                if current_low > ema21 * 1.005:
                    continue

                # RSI in neutral zone (not oversold, confirming uptrend)
                if rsi < 33 or rsi > 67:
                    continue

                tp_percent = 4.0
                sl_percent = 4.0
                take_profit = current_price * (1 + tp_percent / 100)
                stop_loss   = current_price * (1 - sl_percent / 100)

                price_data = await self.fetch_price_data(symbol)
                vol_ratio = price_data.get('volume_ratio', 1.0) if price_data else 1.0
                volume_24h = price_data.get('volume_24h', c['volume_24h']) if price_data else c['volume_24h']
                enhanced_ta = price_data.get('enhanced_ta', {}) if price_data else {}

                logger.info(f"📉 EMA_PULLBACK: {symbol} LONG | EMA21 dist {dist_to_ema21:.2f}% | RSI {rsi:.0f}")

                signal_candidate = {
                    'symbol': symbol, 'direction': 'LONG',
                    'entry_price': current_price, 'stop_loss': stop_loss,
                    'take_profit': take_profit, 'take_profit_1': take_profit,
                    'take_profit_2': None, 'take_profit_3': None,
                    'tp_percent': tp_percent, 'sl_percent': sl_percent,
                    'rsi': rsi, '24h_change': c['change'], '24h_volume': volume_24h,
                    'volume_ratio': vol_ratio, 'btc_correlation': 0.0,
                    'derivatives': {}, 'deriv_adjustments': [], 'enhanced_ta': enhanced_ta,
                    'trade_type': 'EMA_PULLBACK',
                }

                if is_coin_in_ai_rejection_cooldown(symbol, 'LONG'):
                    continue
                ai_result = await ai_analyze_social_signal(signal_candidate)
                if not ai_result.get('approved', True) or ai_result.get('ai_confidence', 5) < 6:
                    _conf = ai_result.get('ai_confidence', 5)
                    _rsn = ai_result.get('reasoning', '')
                    logger.info(f"🤖 AI REJECTED EMA pullback {symbol}: conf={_conf}/10")
                    log_rejection(symbol, 'EMA_PULLBACK', 'AI_REJECTED', 'LONG', _conf, _rsn)
                    add_to_ai_rejection_cooldown(symbol, 'LONG')
                    continue

                add_symbol_cooldown(symbol)

                return {
                    'symbol': symbol, 'direction': 'LONG',
                    'entry_price': current_price, 'stop_loss': stop_loss,
                    'take_profit': take_profit, 'take_profit_1': take_profit,
                    'take_profit_2': None, 'take_profit_3': None,
                    'tp_percent': tp_percent, 'sl_percent': sl_percent,
                    'confidence': 70, 'reasoning': ai_result.get('reasoning', ''),
                    'ai_confidence': ai_result.get('ai_confidence', 6),
                    'ai_recommendation': ai_result.get('recommendation', 'BUY'),
                    'trade_explainer': ai_result.get('trade_explainer', ''),
                    'trade_type': 'EMA_PULLBACK', 'strategy': 'EMA_PULLBACK',
                    'risk_level': 'LOW', 'galaxy_score': 0, 'sentiment': 0.5,
                    'social_volume': 0, 'social_interactions': 0, 'social_dominance': 0,
                    'alt_rank': 9999, 'coin_name': symbol.replace('USDT', ''),
                    'rsi': rsi, '24h_change': c['change'], '24h_volume': volume_24h,
                    'volume_ratio': vol_ratio, 'btc_correlation': 0.0,
                    'derivatives': {}, 'deriv_adjustments': [], 'enhanced_ta': enhanced_ta,
                }

            logger.info("📉 No EMA_PULLBACK signals passed all checks")
            return None
        except Exception as e:
            logger.error(f"EMA pullback scanner error: {e}")
            return None

    async def scan_for_half_back(self) -> Optional[Dict]:
        """
        HALF_BACK: Coin makes a significant move on 1h, then retraces 45-55% of that move.
        Entry at the 50% level. Resumes original direction. TP 3-5%, SL 2%.
        """
        global _daily_half_back_signals, _last_half_back_time
        reset_daily_counters_if_needed()

        if _daily_half_back_signals >= MAX_DAILY_HALF_BACK_SIGNALS:
            logger.debug(f"↩️ Daily half-back limit reached ({MAX_DAILY_HALF_BACK_SIGNALS})")
            return None
        if _last_half_back_time:
            elapsed = (datetime.now() - _last_half_back_time).total_seconds() / 60
            if elapsed < MIN_HALF_BACK_GAP_MINUTES:
                logger.debug(f"↩️ Half-back gap: {elapsed:.0f}min (need {MIN_HALF_BACK_GAP_MINUTES}min)")
                return None

        await self.init()
        try:
            tickers = None
            try:
                tickers = await self._get_binance_tickers()
            except Exception:
                pass
            if not tickers:
                return None

            candidates = []
            for t in tickers:
                sym = t.get('symbol', '')
                if not sym.endswith('USDT') or sym in ('BTCUSDT', 'ETHUSDT', 'USDCUSDT', 'BNBUSDT'):
                    continue
                vol = float(t.get('quoteVolume', 0))
                last_price = float(t.get('lastPrice', 0))
                change = float(t.get('priceChangePercent', 0))
                if vol < 5_000_000 or last_price <= 0:
                    continue
                if abs(change) < 4 or abs(change) > 30:
                    continue
                candidates.append({'symbol': sym, 'volume_24h': vol, 'price': last_price, 'change': change})

            candidates.sort(key=lambda x: x['volume_24h'], reverse=True)
            candidates = candidates[:40]

            btc_state = await get_btc_state()

            def _rsi(c_list, period=14):
                if len(c_list) < period + 1:
                    return 50.0
                gains, losses = [], []
                for i in range(1, len(c_list)):
                    d = c_list[i] - c_list[i-1]
                    gains.append(max(d, 0))
                    losses.append(max(-d, 0))
                ag = sum(gains[-period:]) / period
                al = sum(losses[-period:]) / period
                if al == 0:
                    return 100.0
                return 100 - 100 / (1 + ag / al)

            for c in candidates:
                symbol = c['symbol']
                if is_symbol_on_cooldown(symbol) or is_coin_in_signalled_cooldown(symbol):
                    continue
                is_available = await self.check_bitunix_availability(symbol)
                if not is_available:
                    continue

                try:
                    klines_url = f"https://fapi.binance.com/fapi/v1/klines?symbol={symbol}&interval=1h&limit=60"
                    kl_resp = await self.http_client.get(klines_url, timeout=8)
                    if kl_resp.status_code != 200:
                        continue
                    klines = kl_resp.json()
                    if len(klines) < 20:
                        continue
                    closes = [float(k[4]) for k in klines]
                    highs  = [float(k[2]) for k in klines]
                    lows   = [float(k[3]) for k in klines]
                except Exception:
                    continue

                # Find the most recent swing: look at candles -20 to -2
                swing_window = closes[-20:-1]
                if len(swing_window) < 10:
                    continue

                swing_high = max(swing_window)
                swing_low  = min(swing_window)
                move_size  = swing_high - swing_low
                move_pct   = move_size / swing_low * 100 if swing_low > 0 else 0

                if move_pct < 5:
                    continue  # Move not significant enough

                current_price = closes[-1]
                rsi = _rsi(closes)

                # Determine move direction and check 50% retrace
                mid_point = swing_low + move_size * 0.5
                dist_to_mid = abs(current_price - mid_point) / mid_point * 100

                if dist_to_mid > 2.0:
                    continue  # Not near the 50% level

                # Bullish move (swing high at end) → expect continuation up
                # Bearish move (swing low at end) → expect continuation down
                last_swing_high_idx = swing_window.index(swing_high)
                last_swing_low_idx  = swing_window.index(swing_low)

                if last_swing_high_idx > last_swing_low_idx:
                    direction = 'LONG'
                    if rsi < 37 or rsi > 70:
                        continue
                else:
                    direction = 'SHORT'
                    if rsi < 30 or rsi > 63:
                        continue

                tp_percent = 4.0
                sl_percent = 4.0
                if direction == 'LONG':
                    take_profit = current_price * (1 + tp_percent / 100)
                    stop_loss   = current_price * (1 - sl_percent / 100)
                else:
                    take_profit = current_price * (1 - tp_percent / 100)
                    stop_loss   = current_price * (1 + sl_percent / 100)

                price_data = await self.fetch_price_data(symbol)
                vol_ratio  = price_data.get('volume_ratio', 1.0) if price_data else 1.0
                volume_24h = price_data.get('volume_24h', c['volume_24h']) if price_data else c['volume_24h']
                enhanced_ta = price_data.get('enhanced_ta', {}) if price_data else {}

                logger.info(f"↩️ HALF_BACK: {symbol} {direction} | Move {move_pct:.1f}% | Mid dist {dist_to_mid:.2f}% | RSI {rsi:.0f}")

                signal_candidate = {
                    'symbol': symbol, 'direction': direction,
                    'entry_price': current_price, 'stop_loss': stop_loss,
                    'take_profit': take_profit, 'take_profit_1': take_profit,
                    'take_profit_2': None, 'take_profit_3': None,
                    'tp_percent': tp_percent, 'sl_percent': sl_percent,
                    'rsi': rsi, '24h_change': c['change'], '24h_volume': volume_24h,
                    'volume_ratio': vol_ratio, 'btc_correlation': 0.0,
                    'derivatives': {}, 'deriv_adjustments': [], 'enhanced_ta': enhanced_ta,
                    'trade_type': 'HALF_BACK',
                }

                if is_coin_in_ai_rejection_cooldown(symbol, direction):
                    continue
                ai_result = await ai_analyze_social_signal(signal_candidate)
                if not ai_result.get('approved', True) or ai_result.get('ai_confidence', 5) < 6:
                    _conf = ai_result.get('ai_confidence', 5)
                    _rsn = ai_result.get('reasoning', '')
                    logger.info(f"🤖 AI REJECTED half-back {symbol}: conf={_conf}/10")
                    log_rejection(symbol, 'HALF_BACK', 'AI_REJECTED', direction, _conf, _rsn)
                    add_to_ai_rejection_cooldown(symbol, direction)
                    continue

                add_symbol_cooldown(symbol)

                return {
                    'symbol': symbol, 'direction': direction,
                    'entry_price': current_price, 'stop_loss': stop_loss,
                    'take_profit': take_profit, 'take_profit_1': take_profit,
                    'take_profit_2': None, 'take_profit_3': None,
                    'tp_percent': tp_percent, 'sl_percent': sl_percent,
                    'confidence': 72, 'reasoning': ai_result.get('reasoning', ''),
                    'ai_confidence': ai_result.get('ai_confidence', 6),
                    'ai_recommendation': ai_result.get('recommendation', 'BUY'),
                    'trade_explainer': ai_result.get('trade_explainer', ''),
                    'trade_type': 'HALF_BACK', 'strategy': 'HALF_BACK',
                    'risk_level': 'MEDIUM', 'galaxy_score': 0, 'sentiment': 0.5,
                    'social_volume': 0, 'social_interactions': 0, 'social_dominance': 0,
                    'alt_rank': 9999, 'coin_name': symbol.replace('USDT', ''),
                    'rsi': rsi, '24h_change': c['change'], '24h_volume': volume_24h,
                    'volume_ratio': vol_ratio, 'btc_correlation': 0.0,
                    'derivatives': {}, 'deriv_adjustments': [], 'enhanced_ta': enhanced_ta,
                }

            logger.info("↩️ No HALF_BACK signals passed all checks")
            return None
        except Exception as e:
            logger.error(f"Half-back scanner error: {e}")
            return None

    async def scan_for_oversold_reversal(self) -> Optional[Dict]:
        """
        OVERSOLD_REVERSAL: RSI < 28 on 1h + price at lower Bollinger Band + positive RSI divergence.
        Contrarian long entry after extended sell-off. TP 4-6%, SL 2.5%.
        """
        global _daily_oversold_reversal_signals, _last_oversold_reversal_time
        reset_daily_counters_if_needed()

        if _daily_oversold_reversal_signals >= MAX_DAILY_OVERSOLD_REVERSAL_SIGNALS:
            logger.debug(f"🔄 Daily oversold reversal limit reached ({MAX_DAILY_OVERSOLD_REVERSAL_SIGNALS})")
            return None
        if _last_oversold_reversal_time:
            elapsed = (datetime.now() - _last_oversold_reversal_time).total_seconds() / 60
            if elapsed < MIN_OVERSOLD_REVERSAL_GAP_MINUTES:
                logger.debug(f"🔄 Oversold reversal gap: {elapsed:.0f}min (need {MIN_OVERSOLD_REVERSAL_GAP_MINUTES}min)")
                return None

        await self.init()
        try:
            tickers = None
            try:
                tickers = await self._get_binance_tickers()
            except Exception:
                pass
            if not tickers:
                return None

            candidates = []
            for t in tickers:
                sym = t.get('symbol', '')
                if not sym.endswith('USDT') or sym in ('BTCUSDT', 'ETHUSDT', 'USDCUSDT', 'BNBUSDT'):
                    continue
                vol = float(t.get('quoteVolume', 0))
                last_price = float(t.get('lastPrice', 0))
                change = float(t.get('priceChangePercent', 0))
                if vol < 5_000_000 or last_price <= 0:
                    continue
                if change > -4:
                    continue  # Only want coins that have sold off
                candidates.append({'symbol': sym, 'volume_24h': vol, 'price': last_price, 'change': change})

            candidates.sort(key=lambda x: x['change'])  # Most oversold first
            candidates = candidates[:40]

            btc_state = await get_btc_state()
            logger.debug(f"📉 OVERSOLD_REVERSAL BTC → {btc_state['summary']}")

            def _rsi_series(c_list, period=14):
                """Returns list of RSI values (one per candle after warmup)."""
                if len(c_list) < period + 2:
                    return [50.0]
                rsi_vals = []
                gains, losses = [], []
                for i in range(1, len(c_list)):
                    d = c_list[i] - c_list[i-1]
                    gains.append(max(d, 0))
                    losses.append(max(-d, 0))
                ag = sum(gains[:period]) / period
                al = sum(losses[:period]) / period
                if al == 0:
                    rsi_vals.append(100.0)
                else:
                    rsi_vals.append(100 - 100 / (1 + ag / al))
                for i in range(period, len(gains)):
                    ag = (ag * (period - 1) + gains[i]) / period
                    al = (al * (period - 1) + losses[i]) / period
                    if al == 0:
                        rsi_vals.append(100.0)
                    else:
                        rsi_vals.append(100 - 100 / (1 + ag / al))
                return rsi_vals

            def _bollinger(c_list, period=20, mult=2.0):
                if len(c_list) < period:
                    return None, None, None
                window = c_list[-period:]
                mid = sum(window) / period
                std = (sum((x - mid) ** 2 for x in window) / period) ** 0.5
                return mid, mid + mult * std, mid - mult * std

            for c in candidates:
                symbol = c['symbol']
                if is_symbol_on_cooldown(symbol) or is_coin_in_signalled_cooldown(symbol):
                    continue
                is_available = await self.check_bitunix_availability(symbol)
                if not is_available:
                    continue

                try:
                    klines_url = f"https://fapi.binance.com/fapi/v1/klines?symbol={symbol}&interval=1h&limit=60"
                    kl_resp = await self.http_client.get(klines_url, timeout=8)
                    if kl_resp.status_code != 200:
                        continue
                    klines = kl_resp.json()
                    if len(klines) < 25:
                        continue
                    closes = [float(k[4]) for k in klines]
                    lows   = [float(k[3]) for k in klines]
                    volumes = [float(k[5]) for k in klines]
                except Exception:
                    continue

                rsi_series = _rsi_series(closes)
                current_rsi = rsi_series[-1] if rsi_series else 50

                # Must be genuinely oversold
                if current_rsi >= 28:
                    continue

                # Bollinger Band: price at or below lower band
                bb_mid, bb_upper, bb_lower = _bollinger(closes)
                if bb_lower is None:
                    continue
                current_price = closes[-1]
                if current_price > bb_lower * 1.01:
                    continue  # Not at lower band

                # Positive RSI divergence: price made a lower low but RSI made higher low
                # Compare last 2 local RSI troughs (simple check: last 5 vs prev 5)
                if len(rsi_series) >= 10:
                    recent_rsi_min = min(rsi_series[-5:])
                    prev_rsi_min   = min(rsi_series[-10:-5])
                    recent_price_min = min(lows[-5:])
                    prev_price_min   = min(lows[-10:-5])
                    # Divergence: price lower but RSI higher
                    has_divergence = (recent_price_min < prev_price_min) and (recent_rsi_min > prev_rsi_min)
                else:
                    has_divergence = False

                # Volume uptick on recent candles (buyers stepping in)
                avg_vol = sum(volumes[-20:-3]) / 17 if len(volumes) >= 20 else 1
                recent_vol = sum(volumes[-3:]) / 3
                vol_ratio = recent_vol / avg_vol if avg_vol > 0 else 1

                # At minimum need RSI < 28; divergence adds conviction but not required
                if not has_divergence and vol_ratio < 1.3:
                    continue  # Need at least one confirming factor

                tp_percent = 4.0
                sl_percent = 4.0
                take_profit = current_price * (1 + tp_percent / 100)
                stop_loss   = current_price * (1 - sl_percent / 100)

                price_data = await self.fetch_price_data(symbol)
                volume_24h = price_data.get('volume_24h', c['volume_24h']) if price_data else c['volume_24h']
                enhanced_ta = price_data.get('enhanced_ta', {}) if price_data else {}

                logger.info(f"🔄 OVERSOLD_REVERSAL: {symbol} LONG | RSI {current_rsi:.0f} | BB lower | Divergence: {has_divergence} | Vol {vol_ratio:.1f}x")

                signal_candidate = {
                    'symbol': symbol, 'direction': 'LONG',
                    'entry_price': current_price, 'stop_loss': stop_loss,
                    'take_profit': take_profit, 'take_profit_1': take_profit,
                    'take_profit_2': None, 'take_profit_3': None,
                    'tp_percent': tp_percent, 'sl_percent': sl_percent,
                    'rsi': current_rsi, '24h_change': c['change'], '24h_volume': volume_24h,
                    'volume_ratio': vol_ratio, 'btc_correlation': 0.0,
                    'derivatives': {}, 'deriv_adjustments': [], 'enhanced_ta': enhanced_ta,
                    'trade_type': 'OVERSOLD_REVERSAL',
                }

                if is_coin_in_ai_rejection_cooldown(symbol, 'LONG'):
                    continue
                ai_result = await ai_analyze_social_signal(signal_candidate)
                if not ai_result.get('approved', True) or ai_result.get('ai_confidence', 5) < 6:
                    _conf = ai_result.get('ai_confidence', 5)
                    _rsn = ai_result.get('reasoning', '')
                    logger.info(f"🤖 AI REJECTED oversold reversal {symbol}: conf={_conf}/10")
                    log_rejection(symbol, 'OVERSOLD_REVERSAL', 'AI_REJECTED', 'LONG', _conf, _rsn)
                    add_to_ai_rejection_cooldown(symbol, 'LONG')
                    continue

                add_symbol_cooldown(symbol)

                return {
                    'symbol': symbol, 'direction': 'LONG',
                    'entry_price': current_price, 'stop_loss': stop_loss,
                    'take_profit': take_profit, 'take_profit_1': take_profit,
                    'take_profit_2': None, 'take_profit_3': None,
                    'tp_percent': tp_percent, 'sl_percent': sl_percent,
                    'confidence': 68, 'reasoning': ai_result.get('reasoning', ''),
                    'ai_confidence': ai_result.get('ai_confidence', 6),
                    'ai_recommendation': ai_result.get('recommendation', 'BUY'),
                    'trade_explainer': ai_result.get('trade_explainer', ''),
                    'trade_type': 'OVERSOLD_REVERSAL', 'strategy': 'OVERSOLD_REVERSAL',
                    'risk_level': 'LOW', 'galaxy_score': 0, 'sentiment': 0.5,
                    'social_volume': 0, 'social_interactions': 0, 'social_dominance': 0,
                    'alt_rank': 9999, 'coin_name': symbol.replace('USDT', ''),
                    'rsi': current_rsi, '24h_change': c['change'], '24h_volume': volume_24h,
                    'volume_ratio': vol_ratio, 'btc_correlation': 0.0,
                    'derivatives': {}, 'deriv_adjustments': [], 'enhanced_ta': enhanced_ta,
                }

            logger.info("🔄 No OVERSOLD_REVERSAL signals passed all checks")
            return None
        except Exception as e:
            logger.error(f"Oversold reversal scanner error: {e}")
            return None

    async def scan_for_short_signal(
        self,
        risk_level: str = "MEDIUM",
        min_galaxy_score: int = 8
    ) -> Optional[Dict]:
        """
        Scan for SHORT signals based on negative social sentiment and bearish indicators.
        Triggers: FUD, negative news, sentiment crash, failed pumps, distribution patterns.
        """
        global _daily_social_signals
        
        reset_daily_counters_if_needed()
        
        if _daily_social_signals >= MAX_DAILY_SOCIAL_SIGNALS:
            logger.info("📱 Daily social signal limit reached")
            return None
        
        await self.init()
        
        # RISK = CONFIDENCE FILTER for shorts
        # TP/SL = ALWAYS DYNAMIC based on signal strength
        
        if risk_level == "LOW":
            min_score = 14
            max_score = 13
            rsi_range = (72, 85)
            require_negative_change = False
            max_sentiment = 0.2
            max_dump_pct = -5
            max_positive_pct = 3
        elif risk_level == "MEDIUM":
            min_score = 12
            max_score = 14
            rsi_range = (68, 85)
            require_negative_change = False
            max_sentiment = 0.25
            max_dump_pct = -5
            max_positive_pct = 5
        elif risk_level == "HIGH":
            min_score = 10
            max_score = 16
            rsi_range = (65, 88)
            require_negative_change = False
            max_sentiment = 0.35
            max_dump_pct = -5
            max_positive_pct = 8
        else:  # ALL
            min_score = 8
            max_score = 18
            rsi_range = (62, 90)
            require_negative_change = False
            max_sentiment = 0.4
            max_dump_pct = -5
            max_positive_pct = 10
        
        logger.info(f"📉 SOCIAL SHORT SCANNER | Risk: {risk_level} | Galaxy Score: {min_score}-{max_score} | Max Sentiment: {max_sentiment}")
        
        bitunix_symbols = await self._get_bitunix_symbols()
        raw_tickers = await self._get_binance_tickers()
        if not raw_tickers:
            logger.warning("📉 Short momentum scan: no Binance ticker data available")
            return None

        tradeable = []
        for t in raw_tickers:
            sym = t.get('symbol', '')
            if not sym.endswith('USDT'):
                continue
            if sym.upper() not in bitunix_symbols:
                continue
            chg = float(t.get('priceChangePercent', 0))
            vol = float(t.get('quoteVolume', 0))
            if chg < 5.0 or chg > 25.0:
                continue
            if vol < 100_000:
                continue
            tradeable.append({
                'symbol': sym,
                'change_24h': chg,
                'volume_24h': vol,
                'high_24h': float(t.get('highPrice', 0)),
                'low_24h': float(t.get('lowPrice', 0)),
                'last_price': float(t.get('lastPrice', 0)),
            })

        tradeable.sort(key=lambda x: x['change_24h'], reverse=True)
        logger.info(f"📉 SHORT scan: {len(raw_tickers)} Binance tickers → {len(tradeable)} overextended candidates on Bitunix (5-25% up, $100K+ vol)")

        if not tradeable:
            return None

        btc_state = await get_btc_state()
        logger.info(f"📉 SOCIAL SHORT BTC → {btc_state['summary']}")

        for coin in tradeable:
            symbol = coin['symbol']
            price_change = coin['change_24h']
            galaxy_score = 0
            sentiment = 0.5
            social_volume = 0
            social_interactions = 0
            social_dominance = 0
            alt_rank = 9999
            social_vol_change = 0

            if is_symbol_on_cooldown(symbol):
                continue
            
            price_data = await self.fetch_price_data(symbol)
            if not price_data:
                continue
            
            binance_change = price_data.get('change_24h', 0)
            if binance_change < -10:
                logger.info(f"  📉 {symbol} - ⛔ REJECTED SHORT: Binance 24h={binance_change:+.1f}% (coin already dumping hard, not a short candidate)")
                continue
            
            current_price = price_data['price']
            rsi = price_data['rsi']
            volume_24h = price_data['volume_24h']
            volume_ratio = price_data.get('volume_ratio', 1.0)
            btc_corr = price_data.get('btc_correlation', 0.0)

            high_24h = coin.get('high_24h', 0)
            if high_24h > 0 and current_price > 0:
                pullback_from_high = (high_24h - current_price) / high_24h * 100
                if pullback_from_high > 10:
                    logger.info(f"  📉 {symbol} - ❌ Already -{pullback_from_high:.1f}% from 24h high — shorting the bottom, move already played out")
                    continue
                logger.info(f"  📉 {symbol} - 📍 {pullback_from_high:.1f}% from 24h high — still near top, valid short zone")

            if volume_24h < 100_000:
                logger.info(f"  📉 {symbol} - ❌ Low volume ${volume_24h/1e6:.3f}M (need $100K+)")
                continue

            if btc_corr > 0.90:
                logger.info(f"  📉 {symbol} - ❌ Moves identical to BTC ({btc_corr:.2f})")
                continue
            
            if not (rsi_range[0] <= rsi <= rsi_range[1]):
                logger.debug(f"  {symbol} - RSI {rsi:.0f} not in short range {rsi_range}")
                continue
            
            social_strength = min(100, price_change * 3.0 + volume_ratio * 15.0)

            logger.info(f"✅ MOMENTUM SHORT: {symbol} | Chg: {price_change:.1f}% | Strength: {social_strength:.0f}/100 | RSI: {rsi:.0f} | Vol: {volume_ratio:.1f}x | BTC corr: {btc_corr:.2f}")

            if price_change >= 15:
                base_tp = 8.0
                base_sl = 4.0
            elif price_change >= 8:
                base_tp = 5.0
                base_sl = 3.0
            else:
                base_tp = 3.5
                base_sl = 2.0

            enhanced_ta = price_data.get('enhanced_ta', {})
            tp_percent = base_tp
            sl_percent = base_sl
            deriv_adjustments = []
            derivatives = {}

            if enhanced_ta:
                from app.services.enhanced_ta import get_atr_based_tp_sl
                tp_percent, sl_percent = get_atr_based_tp_sl(enhanced_ta, 'SHORT', tp_percent, sl_percent)

            sl_cap = tp_percent * 0.70
            if sl_percent > sl_cap:
                logger.info(f"📊 {symbol} SHORT SL capped at 70% of TP: {sl_percent:.1f}%→{sl_cap:.1f}% (TP={tp_percent:.1f}%)")
                sl_percent = sl_cap

            take_profit = current_price * (1 - tp_percent / 100)
            stop_loss = current_price * (1 + sl_percent / 100)
            tp2 = current_price * (1 - (tp_percent * 1.5) / 100)
            tp3 = current_price * (1 - (tp_percent * 2.0) / 100)

            influencer_data = None
            buzz_momentum = None

            signal_candidate = {
                'symbol': symbol,
                'direction': 'SHORT',
                'entry_price': current_price,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'take_profit_1': take_profit,
                'take_profit_2': tp2,
                'take_profit_3': tp3,
                'tp_percent': tp_percent,
                'sl_percent': sl_percent,
                'confidence': min(10, max(5, int(price_change / 3))),
                'galaxy_score': 0,
                'sentiment': 0.5,
                'social_volume': 0,
                'social_strength': social_strength,
                'social_vol_change': 0,
                'is_social_spike': False,
                'rsi': rsi,
                'volume_ratio': volume_ratio,
                'btc_correlation': btc_corr,
                '24h_change': price_change,
                '24h_volume': volume_24h,
                'derivatives': {},
                'deriv_adjustments': [],
                'influencer_consensus': None,
                'buzz_momentum': None,
                'enhanced_ta': enhanced_ta,
            }

            try:
                from app.services.order_flow import analyze_order_flow
                of = await analyze_order_flow(symbol=symbol, price_change_1h=0, price_change_24h=price_change, current_price=current_price, volume_24h=volume_24h, ta_data=enhanced_ta)
                if of and abs(of.get('flow_score', 0)) >= 15:
                    signal_candidate['order_flow'] = of
            except Exception as e:
                logger.debug(f"Order flow failed for short {symbol}: {e}")

            if is_coin_in_ai_rejection_cooldown(symbol, 'SHORT'):
                logger.info(f"⏳ Skipping AI for {symbol} SHORT - in 15min rejection cooldown")
                continue
            
            ai_result = await ai_analyze_social_signal(signal_candidate)
            
            if not ai_result.get('approved', True) or ai_result.get('ai_confidence', 5) < 6:
                logger.info(f"🤖 AI REJECTED {symbol} SHORT: conf={ai_result.get('ai_confidence', 5)}/10 | {ai_result.get('reasoning', 'No reason')}")
                add_to_ai_rejection_cooldown(symbol, 'SHORT')
                continue
            
            ai_reasoning = ai_result.get('reasoning', '')
            ai_confidence = ai_result.get('ai_confidence', 5)
            ai_recommendation = ai_result.get('recommendation', 'BUY')
            trade_explainer = ai_result.get('trade_explainer', '')
            
            add_symbol_cooldown(symbol)
            
            return {
                'symbol': symbol,
                'direction': 'SHORT',
                'entry_price': current_price,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'take_profit_1': take_profit,
                'take_profit_2': tp2,
                'take_profit_3': tp3,
                'tp_percent': tp_percent,
                'sl_percent': sl_percent,
                'confidence': min(10, max(5, int(price_change / 3))),
                'reasoning': ai_reasoning,
                'ai_confidence': ai_confidence,
                'ai_recommendation': ai_recommendation,
                'trade_explainer': trade_explainer,
                'trade_type': 'SOCIAL_SHORT',
                'strategy': 'SOCIAL_SHORT',
                'risk_level': risk_level,
                'galaxy_score': 0,
                'sentiment': 0.5,
                'social_volume': 0,
                'social_interactions': 0,
                'social_dominance': 0,
                'alt_rank': 9999,
                'coin_name': symbol.replace('USDT', ''),
                'rsi': rsi,
                '24h_change': price_change,
                '24h_volume': volume_24h,
                'derivatives': {},
                'deriv_adjustments': [],
                'social_strength': social_strength,
                'social_vol_change': 0,
                'volume_ratio': volume_ratio,
                'btc_correlation': btc_corr,
                'influencer_consensus': None,
                'buzz_momentum': None,
                'enhanced_ta': enhanced_ta,
            }
        
        logger.info("📉 No valid social SHORT signals found")
        return None


async def broadcast_social_signal(db_session: Session, bot):
    """
    Main function to scan for social signals and broadcast to enabled users.
    Runs independently of Top Gainers mode.
    """
    global _social_scanning_active, _daily_social_signals
    global _daily_scalp_signals, _last_scalp_time
    global _daily_squeeze_signals, _last_squeeze_time
    global _daily_supertrend_signals, _last_supertrend_time
    global _daily_macd_signals, _last_macd_time
    global _daily_range_breakout_signals, _last_range_breakout_time
    global _daily_ema_pullback_signals, _last_ema_pullback_time
    global _daily_half_back_signals, _last_half_back_time
    global _daily_oversold_reversal_signals, _last_oversold_reversal_time
    
    if not SOCIAL_SCANNING_ENABLED:
        logger.debug("📱 Social scanning disabled")
        return
    
    if is_circuit_breaker_active():
        logger.warning("🛑 Circuit breaker active - skipping all signal scanning")
        return
    
    if not check_global_signal_limit():
        logger.info(f"📱 Global daily signal limit reached ({MAX_GLOBAL_DAILY_SIGNALS}) - skipping social scan")
        return
    
    if not check_signal_gap():
        return
    
    if _social_scanning_active:
        logger.debug("📱 Social scan already in progress")
        return
    
    
    _social_scanning_active = True
    
    try:
        from app.models import User, UserPreference, Signal
        
        from sqlalchemy import or_
        users_with_social = db_session.query(User).join(UserPreference).filter(
            UserPreference.social_mode_enabled == True,
            or_(
                User.is_admin == True,
                User.grandfathered == True,
                (User.subscription_end != None) & (User.subscription_end > datetime.now())
            )
        ).all()
        
        if not users_with_social:
            logger.debug("📱 No authorized users with social mode enabled")
            return
        
        logger.info(f"📱 ═══════════════════════════════════════════════════════")
        logger.info(f"📱 SOCIAL SIGNALS SCANNER - {len(users_with_social)} users enabled")
        logger.info(f"📱 ═══════════════════════════════════════════════════════")
        
        service = SocialSignalService()
        await service.init()
        
        risk_counts = {}
        for u in users_with_social:
            p = u.preferences
            r = getattr(p, 'social_risk_level', 'MEDIUM') or 'MEDIUM' if p else 'MEDIUM'
            risk_counts[r] = risk_counts.get(r, 0) + 1
        most_common_risk = max(risk_counts, key=risk_counts.get) if risk_counts else "MEDIUM"
        
        galaxy_scores = []
        for u in users_with_social:
            p = u.preferences
            g = getattr(p, 'social_min_galaxy_score', 8) or 8 if p else 8
            galaxy_scores.append(g)
        min_galaxy = min(galaxy_scores) if galaxy_scores else 8
        
        logger.info(f"📱 Risk profile: {most_common_risk} (from {risk_counts}) | Min galaxy: {min_galaxy}")
        
        news_users = users_with_social
        
        # Liquidation cascade alerts disabled
        
        # Run all scanners in priority order.
        # Key design: if a scanner returns a signal but it fails cooldown/confidence,
        # we continue to the next scanner instead of giving up entirely.
        MIN_SCANNER_CONFIDENCE = 6  # Flat minimum across all scanner types

        async def _try_scanner(label: str, coro):
            """Run a scanner coroutine and validate the result. Returns signal or None."""
            try:
                result = await coro
                if not result:
                    return None
                sym = result.get('symbol', '')
                if is_coin_in_signalled_cooldown(sym):
                    logger.info(f"🔇 [{label}] {sym} in 24h cooldown — trying next scanner")
                    return None
                ai_conf = result.get('ai_confidence', 0)
                if ai_conf < MIN_SCANNER_CONFIDENCE:
                    logger.info(f"🚫 [{label}] {sym} confidence {ai_conf}/{MIN_SCANNER_CONFIDENCE} — trying next scanner")
                    return None
                logger.info(f"✅ [{label}] {sym} {result.get('direction','LONG')} cleared all checks (conf {ai_conf}/10)")
                return result
            except Exception as e:
                logger.error(f"{label} scanner error: {e}")
                return None

        signal = None

        # 1. MOMENTUM RUNNERS — disabled (early mover long/short scanning off)
        # if not signal:
        #     signal = await _try_scanner("MOMENTUM", service.scan_for_momentum_runners())

        # 2. BREAKING NEWS — fast-moving event-driven trades
        if not signal and news_users:
            try:
                from app.services.realtime_news import scan_for_breaking_news_signal
                signal = await _try_scanner("NEWS", scan_for_breaking_news_signal(
                    check_bitunix_func=service.check_bitunix_availability,
                    fetch_price_func=service.fetch_price_data
                ))
            except Exception as e:
                logger.error(f"Breaking news scan error: {e}")
        elif not signal:
            logger.debug("📰 News trading disabled for all users, skipping news scan")

        # 3. SOCIAL LONG — disabled (LunarCrush removed)
        # if not signal:
        #     signal = await _try_scanner("SOCIAL_LONG", service.generate_social_signal(
        #         risk_level=most_common_risk,
        #         min_galaxy_score=min_galaxy
        #     ))

        # 4. VOLUME SCALP — sudden volume surges 1.5x+ normal
        if not signal:
            signal = await _try_scanner("VOL_SCALP", service.scan_for_volume_scalps())

        # 5. SQUEEZE BREAKOUT — BB/Keltner squeeze release
        if not signal:
            signal = await _try_scanner("SQUEEZE", service.scan_for_squeeze_breakout())

        # 6. SUPERTREND — 15m trend flips with EMA ribbon
        if not signal:
            signal = await _try_scanner("SUPERTREND", service.scan_for_supertrend())

        # 7. MACD MOMENTUM — fast 8/21/5 crossover
        if not signal:
            signal = await _try_scanner("MACD", service.scan_for_macd_momentum())

        # 8. RANGE BREAKOUT — tight consolidation then volume breakout
        if not signal:
            signal = await _try_scanner("RANGE_BREAKOUT", service.scan_for_range_breakout())

        # 9. EMA PULLBACK — uptrend pullback to 21 EMA bounce
        if not signal:
            signal = await _try_scanner("EMA_PULLBACK", service.scan_for_ema_pullback())

        # 10. HALF BACK — 50% retracement then trend resumes
        if not signal:
            signal = await _try_scanner("HALF_BACK", service.scan_for_half_back())

        # 11. OVERSOLD REVERSAL — RSI<28 + lower BB + positive divergence
        if not signal:
            signal = await _try_scanner("OVERSOLD_REVERSAL", service.scan_for_oversold_reversal())

        # 12. SHORT SIGNALS — disabled
        # if not signal:
        #     signal = await _try_scanner("SHORT", service.scan_for_short_signal(
        #         risk_level=most_common_risk,
        #         min_galaxy_score=min_galaxy
        #     ))

        # 9. RELIEF BOUNCE — top losers -20%+ bouncing from lows
        if not signal:
            signal = await _try_scanner("RELIEF", service.scan_for_relief_bounce())

        if not signal:
            logger.info("📱 No qualifying signal this cycle — all scanners returned nothing")
        
        if signal:
            direction = signal.get('direction', 'LONG')
            symbol = signal.get('symbol', 'UNKNOWN')
            
            # Check for crowded trades (Long/Short ratio extreme)
            deriv_data = signal.get('derivatives', {})
            if deriv_data and deriv_data.get('has_data'):
                ls_ratio = deriv_data.get('ls_ratio_value', 1.0) # Usually 1.0 is balanced
                funding = deriv_data.get('funding_rate', 0) or 0
                
                if direction == 'LONG' and funding > 0.05:
                    logger.info(f"🚫 {symbol} blocked - Extreme positive funding {funding:.4f}% (longs paying heavily)")
                    signal = None
                elif direction == 'SHORT' and funding < -0.05:
                    logger.info(f"🚫 {symbol} blocked - Extreme negative funding {funding:.4f}% (shorts paying heavily)")
                    signal = None
                elif abs(funding) > 0.03:
                    logger.info(f"⚠️ {symbol} - Elevated funding rate {funding:.4f}%, proceeding with caution")
                
                if signal:
                    if direction == 'LONG' and ls_ratio > 2.5:
                        logger.info(f"🚫 {symbol} blocked - Trade too crowded ({ls_ratio:.2f} L/S ratio, >70% longs)")
                        signal = None
                    elif direction == 'SHORT' and ls_ratio < 0.4:
                        logger.info(f"🚫 {symbol} blocked - Trade too crowded ({ls_ratio:.2f} L/S ratio, >70% shorts)")
                        signal = None
                    elif (direction == 'LONG' and ls_ratio > 1.8) or (direction == 'SHORT' and ls_ratio < 0.55):
                        logger.info(f"⚠️ {symbol} - Somewhat crowded ({ls_ratio:.2f} L/S ratio), proceeding")
            
            if signal:
                entry = signal['entry_price']
                sl = signal['stop_loss']
                tp = signal['take_profit']
                galaxy = signal.get('galaxy_score', 0)
                sentiment = signal.get('sentiment', 0)
            
            # Determine leverage based on coin type
            is_top = is_top_coin(symbol)
            
            rating = interpret_signal_score(galaxy)
            
            def fmt_price(p):
                if p >= 1000:
                    return f"${p:,.2f}"
                elif p >= 1:
                    return f"${p:.4f}"
                elif p >= 0.01:
                    return f"${p:.6f}"
                elif p >= 0.0001:
                    return f"${p:.8f}"
                else:
                    return f"${p:.10f}"
            
            tp_pct = signal.get('tp_percent', 0)
            sl_pct = signal.get('sl_percent', 0)

            regime_tag = ""
            try:
                from app.services.ai_market_intelligence import get_regime_tp_sl_multipliers
                regime_adj = get_regime_tp_sl_multipliers()
                tp_mult = regime_adj.get('tp_mult', 1.0)
                sl_mult = regime_adj.get('sl_mult', 1.0)
                regime_name = regime_adj.get('regime', 'UNKNOWN')

                if tp_mult != 1.0 or sl_mult != 1.0:
                    old_tp, old_sl = tp_pct, sl_pct
                    tp_pct = round(tp_pct * tp_mult, 2)
                    sl_pct = round(sl_pct * sl_mult, 2)

                    entry = signal.get('entry_price', 0)
                    direction = signal.get('direction', 'LONG')
                    if direction == 'LONG':
                        signal['take_profit'] = entry * (1 + tp_pct / 100)
                        signal['take_profit_1'] = signal['take_profit']
                        signal['stop_loss'] = entry * (1 - sl_pct / 100)
                        if signal.get('take_profit_2'):
                            signal['take_profit_2'] = entry * (1 + (tp_pct * 1.5) / 100)
                        if signal.get('take_profit_3'):
                            signal['take_profit_3'] = entry * (1 + (tp_pct * 2.0) / 100)
                    else:
                        signal['take_profit'] = entry * (1 - tp_pct / 100)
                        signal['take_profit_1'] = signal['take_profit']
                        signal['stop_loss'] = entry * (1 + sl_pct / 100)
                        if signal.get('take_profit_2'):
                            signal['take_profit_2'] = entry * (1 - (tp_pct * 1.5) / 100)
                        if signal.get('take_profit_3'):
                            signal['take_profit_3'] = entry * (1 - (tp_pct * 2.0) / 100)

                    signal['tp_percent'] = tp_pct
                    signal['sl_percent'] = sl_pct

                    regime_icons = {
                        'TRENDING_UP': '📈', 'TRENDING_DOWN': '📉',
                        'VOLATILE_BREAKOUT': '💥', 'CHOPPY': '🔀', 'RANGING': '↔️'
                    }
                    r_icon = regime_icons.get(regime_name, '🔮')
                    regime_tag = f"\n{r_icon} <i>Regime: {regime_name} (TP {tp_mult}x, SL {sl_mult}x)</i>"
                    logger.info(f"🔮 Regime {regime_name}: TP {old_tp:.1f}%→{tp_pct:.1f}% | SL {old_sl:.1f}%→{sl_pct:.1f}%")
            except Exception as regime_err:
                logger.debug(f"Regime adaptive TP/SL skipped: {regime_err}")

            tp2 = None  # Single TP only — TP2/TP3 disabled
            tp3 = None  # Single TP only — TP2/TP3 disabled
            tp = signal.get('take_profit', signal.get('take_profit_1', 0))
            sl = signal.get('stop_loss', 0)
            entry = signal.get('entry_price', 0)
            risk_level = signal.get('risk_level', 'MEDIUM')
            social_vol = signal.get('social_volume', 0)
            social_strength = signal.get('social_strength', 0)
            social_vol_change = signal.get('social_vol_change', 0)
            is_spike = signal.get('is_social_spike', False)
            rsi_val = signal.get('rsi', 50)
            vol_ratio = signal.get('volume_ratio', 1.0)
            btc_corr = signal.get('btc_correlation', 0.0)
            volume_24h = signal.get('24h_volume', 0)
            change_24h = signal.get('24h_change', 0)
            
            display_lev = 25
            
            risk_badges = {
                'LOW': '🟢 LOW RISK',
                'MEDIUM': '🟡 MEDIUM RISK',
                'HIGH': '🔴 HIGH RISK',
                'MOMENTUM': '🚀 MOMENTUM',
                'ALL': '🌐 ALL',
                'RELIEF': '📉 RELIEF BOUNCE',
            }
            risk_badge = risk_badges.get(risk_level, f'🟡 {risk_level}')
            
            dir_icon = "🟢" if direction == 'LONG' else "🔴"
            sign = "+" if direction == 'LONG' else "-"
            
            tp1_roi = tp_pct * display_lev
            tp_lines = f"  ├  TP1  <code>{fmt_price(tp)}</code>  <b>{sign}{tp_pct:.1f}%</b>  ·  <b>{sign}{tp1_roi:.0f}% ROI</b>"
            if tp2:
                tp2_pct = tp_pct * 1.5
                tp2_roi = tp2_pct * display_lev
                tp_lines += f"\n  ├  TP2  <code>{fmt_price(tp2)}</code>  <b>{sign}{tp2_pct:.1f}%</b>  ·  <b>{sign}{tp2_roi:.0f}% ROI</b>"
            if tp3:
                tp3_pct = tp_pct * 2.0
                tp3_roi = tp3_pct * display_lev
                tp_lines += f"\n  ├  TP3  <code>{fmt_price(tp3)}</code>  <b>{sign}{tp3_pct:.1f}%</b>  ·  <b>{sign}{tp3_roi:.0f}% ROI</b>"
            
            sl_roi = sl_pct * display_lev
            
            vol_display = f"${volume_24h/1e6:.1f}M" if volume_24h >= 1e6 else f"${volume_24h/1e3:.0f}K"

            separator = "━━━━━━━━━━━━━━━━━━━━"
            
            is_news_signal = signal.get('trade_type') == 'NEWS_SIGNAL'
            is_momentum_runner = signal.get('trade_type') == 'MOMENTUM_RUNNER'
            is_relief_bounce = signal.get('trade_type') == 'RELIEF_BOUNCE'
            is_volume_scalp = signal.get('trade_type') == 'VOLUME_SCALP'
            is_fast_trade = signal.get('trade_type') in ('VOLUME_SCALP', 'SQUEEZE_BREAKOUT', 'SUPERTREND', 'MACD_MOMENTUM')
            news_title = signal.get('news_title', '')
            
            strength = calculate_signal_strength(signal)

            try:
                from app.services.order_flow import get_flow_score_modifier as flow_mod
                flow_data = signal.get('order_flow')
                if flow_data and strength:
                    mod = flow_mod(flow_data, direction)
                    if abs(mod) > 0:
                        strength['score'] = max(1, min(10, round(strength.get('score', 5) + mod)))
                        logger.info(f"  📊 Order flow modifier: {mod:+.1f} → score {strength['score']}")
            except Exception as e:
                logger.debug(f"Score modifier error: {e}")

            strength_line = format_signal_strength_detail(strength)
            
            signal_score = strength.get('score', 5) if strength else 5
            if signal_score < 6:
                logger.info(f"🚫 {symbol} blocked - Signal strength too low ({signal_score}/10, minimum 6)")
                signal = None
        
        if signal:
            ai_reasoning = signal.get('reasoning', '')[:200] if signal.get('reasoning') else ''
            ai_rec = signal.get('ai_recommendation', '')
            ai_conf = signal.get('ai_confidence', 0)
            rec_emoji = {"STRONG BUY": "🚀", "BUY": "✅", "STRONG SELL": "🔻", "SELL": "📉", "HOLD": "⏸️", "AVOID": "🚫"}.get(ai_rec, "📊")

            base_ticker_clean = symbol.replace('USDT', '').replace('/USDT:USDT', '')

            if is_momentum_runner:
                galaxy_m = signal.get('galaxy_score', 0)
                sent_m = signal.get('sentiment', 0)
                type_label = "MOMENTUM"
                type_icon = "🚀"
                context_line = f"24h Move <b>{change_24h:+.1f}%</b>"

                social_line = ""
                if galaxy_m > 0:
                    social_line = f"\n📡  Social <b>{galaxy_m}/16</b>  ·  Sentiment <b>{sent_m:.0%}</b>"

                btc_corr_line_m = "" if base_ticker_clean in ('BTC', 'BTCUSDT') else f"\nBTC Corr <b>{btc_corr:.0%}</b>"

                message = (
                    f"{type_icon} <b>{type_label} {direction}</b>  {dir_icon}\n"
                    f"{separator}\n\n"
                    f"<b>${base_ticker_clean}</b>  ·  {context_line}\n"
                    f"Grade: <b>{risk_badge}</b>\n\n"
                    f"{strength_line}\n\n"
                    f"<b>TRADE SETUP</b>\n"
                    f"  ▸  Entry  <code>{fmt_price(entry)}</code>\n"
                    f"{tp_lines}\n"
                    f"  └  SL  <code>{fmt_price(sl)}</code>  <b>-{sl_pct:.1f}%</b>  ·  <b>-{sl_roi:.0f}% ROI</b>\n\n"
                    f"<b>MARKET CONTEXT</b>\n"
                    f"RSI <b>{rsi_val:.0f}</b>  ·  24h <b>{change_24h:+.1f}%</b>  ·  Vol <b>{vol_display}</b>"
                    f"{social_line}"
                    f"{btc_corr_line_m}"
                )

            elif is_volume_scalp:
                vol_ratio_display = signal.get('volume_ratio', 0)
                btc_corr_line_s = "" if base_ticker_clean in ('BTC', 'BTCUSDT') else f"\nBTC Corr <b>{btc_corr:.0%}</b>"

                message = (
                    f"⚡ <b>VOLUME SCALP {direction}</b>  {dir_icon}\n"
                    f"{separator}\n\n"
                    f"<b>${base_ticker_clean}</b>  ·  Vol Surge <b>{vol_ratio_display:.1f}x</b>  ·  24h <b>{change_24h:+.1f}%</b>\n"
                    f"Grade: <b>{risk_badge}</b>\n\n"
                    f"{strength_line}\n\n"
                    f"<b>SCALP SETUP</b>  (1:1 R:R)\n"
                    f"  ▸  Entry  <code>{fmt_price(entry)}</code>\n"
                    f"{tp_lines}\n"
                    f"  └  SL  <code>{fmt_price(sl)}</code>  <b>-{sl_pct:.1f}%</b>  ·  <b>-{sl_roi:.0f}% ROI</b>\n\n"
                    f"<b>MARKET CONTEXT</b>\n"
                    f"RSI <b>{rsi_val:.0f}</b>  ·  Vol <b>{vol_display}</b>  ·  Surge <b>{vol_ratio_display:.1f}x</b>"
                    f"{btc_corr_line_s}"
                )

            elif signal.get('trade_type') == 'SQUEEZE_BREAKOUT':
                btc_corr_line_sq = "" if base_ticker_clean in ('BTC', 'BTCUSDT') else f"\nBTC Corr <b>{btc_corr:.0%}</b>"
                bb_bw = signal.get('bb_bandwidth', 0)

                message = (
                    f"🔥 <b>SQUEEZE BREAKOUT {direction}</b>  {dir_icon}\n"
                    f"{separator}\n\n"
                    f"<b>${base_ticker_clean}</b>  ·  Squeeze Released  ·  24h <b>{change_24h:+.1f}%</b>\n"
                    f"Grade: <b>{risk_badge}</b>\n\n"
                    f"{strength_line}\n\n"
                    f"<b>TRADE SETUP</b>  (Squeeze Release)\n"
                    f"  ▸  Entry  <code>{fmt_price(entry)}</code>\n"
                    f"{tp_lines}\n"
                    f"  └  SL  <code>{fmt_price(sl)}</code>  <b>-{sl_pct:.1f}%</b>  ·  <b>-{sl_roi:.0f}% ROI</b>\n\n"
                    f"<b>MARKET CONTEXT</b>\n"
                    f"RSI <b>{rsi_val:.0f}</b>  ·  BB Width <b>{bb_bw:.1f}</b>  ·  Vol <b>{vol_display}</b>"
                    f"{btc_corr_line_sq}"
                )

            elif signal.get('trade_type') == 'SUPERTREND':
                btc_corr_line_st = "" if base_ticker_clean in ('BTC', 'BTCUSDT') else f"\nBTC Corr <b>{btc_corr:.0%}</b>"
                trend_str = signal.get('trend_strength', 0)

                message = (
                    f"📈 <b>SUPERTREND {direction}</b>  {dir_icon}\n"
                    f"{separator}\n\n"
                    f"<b>${base_ticker_clean}</b>  ·  Trend Flip  ·  24h <b>{change_24h:+.1f}%</b>\n"
                    f"Grade: <b>{risk_badge}</b>\n\n"
                    f"{strength_line}\n\n"
                    f"<b>TRADE SETUP</b>  (SuperTrend)\n"
                    f"  ▸  Entry  <code>{fmt_price(entry)}</code>\n"
                    f"{tp_lines}\n"
                    f"  └  SL  <code>{fmt_price(sl)}</code>  <b>-{sl_pct:.1f}%</b>  ·  <b>-{sl_roi:.0f}% ROI</b>\n\n"
                    f"<b>MARKET CONTEXT</b>\n"
                    f"RSI <b>{rsi_val:.0f}</b>  ·  Strength <b>{trend_str}/5</b>  ·  Vol <b>{vol_display}</b>"
                    f"{btc_corr_line_st}"
                )

            elif signal.get('trade_type') == 'MACD_MOMENTUM':
                btc_corr_line_mc = "" if base_ticker_clean in ('BTC', 'BTCUSDT') else f"\nBTC Corr <b>{btc_corr:.0%}</b>"
                vol_ratio_mc = signal.get('volume_ratio', 0)

                message = (
                    f"⚡ <b>MACD MOMENTUM {direction}</b>  {dir_icon}\n"
                    f"{separator}\n\n"
                    f"<b>${base_ticker_clean}</b>  ·  MACD Cross  ·  24h <b>{change_24h:+.1f}%</b>\n"
                    f"Grade: <b>{risk_badge}</b>\n\n"
                    f"{strength_line}\n\n"
                    f"<b>TRADE SETUP</b>  (MACD 8/21/5)\n"
                    f"  ▸  Entry  <code>{fmt_price(entry)}</code>\n"
                    f"{tp_lines}\n"
                    f"  └  SL  <code>{fmt_price(sl)}</code>  <b>-{sl_pct:.1f}%</b>  ·  <b>-{sl_roi:.0f}% ROI</b>\n\n"
                    f"<b>MARKET CONTEXT</b>\n"
                    f"RSI <b>{rsi_val:.0f}</b>  ·  Vol <b>{vol_display}</b>  ·  Surge <b>{vol_ratio_mc:.1f}x</b>"
                    f"{btc_corr_line_mc}"
                )

            elif is_relief_bounce:
                bounce_pct = signal.get('bounce_from_low', 0)

                message = (
                    f"📉 <b>RELIEF BOUNCE</b>  {dir_icon}\n"
                    f"{separator}\n\n"
                    f"<b>${base_ticker_clean}</b>  ·  Dumped <b>{change_24h:.1f}%</b>  ·  Bouncing <b>+{bounce_pct:.1f}%</b>\n"
                    f"Grade: <b>{risk_badge}</b>\n\n"
                    f"{strength_line}\n\n"
                    f"<b>TRADE SETUP</b>\n"
                    f"  ▸  Entry  <code>{fmt_price(entry)}</code>\n"
                    f"{tp_lines}\n"
                    f"  └  SL  <code>{fmt_price(sl)}</code>  <b>-{sl_pct:.1f}%</b>  ·  <b>-{sl_roi:.0f}% ROI</b>\n\n"
                    f"<b>MARKET CONTEXT</b>\n"
                    f"RSI <b>{rsi_val:.0f}</b>  ·  Vol <b>{vol_display}</b>\n"
                    f"⬆️ Bounce <b>+{bounce_pct:.1f}%</b> from 24h low"
                )

            elif is_news_signal:
                short_title = news_title[:70] + '...' if len(news_title) > 70 else news_title
                news_impact = signal.get('confidence', 0) or galaxy
                btc_corr_line = "" if base_ticker_clean in ('BTC', 'BTCUSDT') else f"  ·  BTC Corr <b>{btc_corr:.0%}</b>"

                message = (
                    f"📰 <b>NEWS {direction}</b>  {dir_icon}\n"
                    f"{separator}\n\n"
                    f"<b>${base_ticker_clean}</b>\n"
                    f"<i>{short_title}</i>\n"
                    f"Grade: <b>{risk_badge}</b>\n\n"
                    f"{strength_line}\n\n"
                    f"<b>TRADE SETUP</b>\n"
                    f"  ▸  Entry  <code>{fmt_price(entry)}</code>\n"
                    f"{tp_lines}\n"
                    f"  └  SL  <code>{fmt_price(sl)}</code>  <b>-{sl_pct:.1f}%</b>  ·  <b>-{sl_roi:.0f}% ROI</b>\n\n"
                    f"<b>MARKET CONTEXT</b>\n"
                    f"RSI <b>{rsi_val:.0f}</b>  ·  24h <b>{change_24h:+.1f}%</b>  ·  Vol <b>{vol_display}</b>\n"
                    f"Impact <b>{news_impact}/100</b>{btc_corr_line}"
                )

            else:
                sentiment_pct = int(sentiment * 100)
                social_interactions = signal.get('social_interactions', 0)
                social_dominance = signal.get('social_dominance', 0)
                alt_rank = signal.get('alt_rank', 9999)
                coin_name = signal.get('coin_name', '')

                interactions_display = f"{social_interactions/1e6:.1f}M" if social_interactions >= 1e6 else f"{social_interactions/1e3:.1f}K" if social_interactions >= 1000 else f"{social_interactions:,}"

                name_display = f" ({coin_name})" if coin_name else ""
                spike_label = "SPIKE " if is_spike else ""

                has_social_data = galaxy > 0 or social_vol > 0 or social_interactions > 0
                vol_ratio_s = signal.get('volume_ratio', 0)

                if has_social_data:
                    intel_block = (
                        f"<b>SOCIAL INTEL</b>\n"
                        f"Strength <b>{social_strength:.0f}/100</b>  ·  Score <b>{galaxy}/16</b> {rating}\n"
                        f"Sentiment <b>{sentiment_pct}%</b>  ·  Posts <b>{social_vol:,}</b>  ·  Reach <b>{interactions_display}</b>"
                    )
                else:
                    near_high_pct = signal.get('near_high_pct', 0) or 0
                    vol_label = f"Vol <b>{vol_ratio_s:.1f}x</b> avg" if vol_ratio_s > 0 else f"Vol <b>{vol_display}</b>"
                    high_label = f"  ·  <b>{near_high_pct:.1f}%</b> from high" if near_high_pct > 0 else ""
                    intel_block = (
                        f"<b>MOMENTUM INTEL</b>\n"
                        f"24h <b>{change_24h:+.1f}%</b>  ·  {vol_label}{high_label}\n"
                        f"Sentiment <b>{sentiment_pct}%</b>"
                    )

                message = (
                    f"{dir_icon} <b>{spike_label}SOCIAL {direction}</b>\n"
                    f"{separator}\n\n"
                    f"<b>${base_ticker_clean}</b>{name_display}\n"
                    f"Grade: <b>{risk_badge}</b>\n\n"
                    f"{strength_line}\n\n"
                    f"<b>TRADE SETUP</b>\n"
                    f"  ▸  Entry  <code>{fmt_price(entry)}</code>\n"
                    f"{tp_lines}\n"
                    f"  └  SL  <code>{fmt_price(sl)}</code>  <b>-{sl_pct:.1f}%</b>  ·  <b>-{sl_roi:.0f}% ROI</b>\n\n"
                    f"{intel_block}"
                )

                if social_vol_change > 0:
                    message += f"\n🔥 Buzz <b>+{social_vol_change:.0f}%</b> surge"

                if social_dominance > 0:
                    dom_line = f"\nDominance <b>{social_dominance:.2f}%</b>"
                    if alt_rank < 9999:
                        dom_line += f"  ·  Rank <b>#{alt_rank}</b>"
                    message += dom_line

                influencer = signal.get('influencer_consensus')
                if influencer and isinstance(influencer, dict) and influencer.get('num_creators', 0) > 0:
                    cons = influencer.get('consensus', 'MIXED')
                    cons_icon = {"BULLISH": "🟢", "LEAN BULLISH": "🟢", "BEARISH": "🔴", "LEAN BEARISH": "🔴", "MIXED": "⚖️"}.get(cons, "⚖️")
                    total_fol = influencer.get('total_followers', 0)
                    followers_display = f"{total_fol/1e6:.1f}M" if total_fol >= 1e6 else f"{total_fol/1e3:.0f}K"
                    whale_tag = f"  ·  <b>{influencer.get('big_accounts', 0)}</b> whales" if influencer.get('big_accounts', 0) > 0 else ""
                    message += (
                        f"\n\n<b>INFLUENCER CONSENSUS</b>\n"
                        f"{cons_icon} <b>{cons}</b>  ({influencer.get('bullish_count', 0)}🟢 {influencer.get('bearish_count', 0)}🔴 {influencer.get('neutral_count', 0)}⚪)\n"
                        f"Reach <b>{followers_display}</b>{whale_tag}"
                    )

                buzz = signal.get('buzz_momentum')
                if buzz and isinstance(buzz, dict) and buzz.get('trend'):
                    trend_icon = {"RISING": "📈", "FALLING": "📉", "STABLE": "➡️"}.get(buzz.get('trend', ''), "➡️")
                    message += (
                        f"\n\n<b>MOMENTUM</b>\n"
                        f"{trend_icon} <b>{buzz.get('trend', 'UNKNOWN')}</b> ({buzz.get('buzz_change_pct', 0):+.0f}%)"
                    )

                btc_corr_line = "" if base_ticker_clean in ('BTC', 'BTCUSDT') else f"\nBTC Corr <b>{btc_corr:.0%}</b>"

                message += (
                    f"\n\n<b>MARKET CONTEXT</b>\n"
                    f"RSI <b>{rsi_val:.0f}</b>  ·  24h <b>{change_24h:+.1f}%</b>  ·  Vol <b>{vol_display}</b>"
                    f"{btc_corr_line}"
                )

            enhanced_ta_data = signal.get('enhanced_ta', {})
            if enhanced_ta_data:
                from app.services.enhanced_ta import format_ta_for_message
                ta_msg = format_ta_for_message(enhanced_ta_data)
                if ta_msg:
                    message += f"\n\n<b>TECHNICALS</b>\n{ta_msg}"

            deriv_data = signal.get('derivatives', {})
            if deriv_data and deriv_data.get('has_data'):
                deriv_msg = format_derivatives_for_message(deriv_data)
                if deriv_msg:
                    message += f"\n\n{deriv_msg}"

                deriv_adj_list = signal.get('deriv_adjustments', [])
                if deriv_adj_list:
                    message += f"\n<i>TP/SL optimized by {len(deriv_adj_list)} factor{'s' if len(deriv_adj_list) > 1 else ''}</i>"

            try:
                flow_data = signal.get('order_flow')
                if flow_data and abs(flow_data.get('flow_score', 0)) >= 15:
                    from app.services.order_flow import format_order_flow_for_message
                    flow_msg = format_order_flow_for_message(flow_data)
                    if flow_msg:
                        message += f"\n\n<b>ORDER FLOW</b>\n{flow_msg}"
            except Exception as e:
                logger.debug(f"Order flow message format error: {e}")

            trade_explainer = signal.get('trade_explainer', '')
            if trade_explainer:
                message += f"\n\n💡 <b>WHY THIS TRADE</b>\n<i>{trade_explainer[:400]}</i>"

            if ai_reasoning:
                message += f"\n\n{rec_emoji} <b>AI: {ai_rec}</b>  ({ai_conf}/10)\n<i>{ai_reasoning}</i>"

            if regime_tag:
                message += regime_tag

            message += f"\n\n{separator}"
            
            # Record signal in database FIRST (needed for trade execution)
            default_lev = 25 if is_top else 10
            if is_news_signal:
                sig_type = 'NEWS_SIGNAL'
            elif is_volume_scalp:
                sig_type = 'VOLUME_SCALP'
            elif is_relief_bounce:
                sig_type = 'RELIEF_BOUNCE'
            elif is_momentum_runner:
                sig_type = 'MOMENTUM_RUNNER'
            else:
                sig_type = 'SOCIAL_SIGNAL'
            ai_conf_val = signal.get('ai_confidence', 5)
            scaled_confidence = ai_conf_val * 10
            new_signal = Signal(
                symbol=symbol,
                direction=direction,
                entry_price=entry,
                stop_loss=sl,
                take_profit=tp,
                take_profit_1=tp,
                take_profit_2=signal.get('take_profit_2'),
                take_profit_3=signal.get('take_profit_3'),
                confidence=scaled_confidence,
                signal_type=sig_type,
                timeframe='15m',
                rsi=rsi_val,
                volume=volume_24h,
                reasoning=signal['reasoning']
            )
            db_session.add(new_signal)
            db_session.commit()
            db_session.refresh(new_signal)
            
            add_to_signalled_cooldown(symbol)
            _daily_social_signals += 1
            increment_global_signal_count()
            record_signal_broadcast()

            # Update per-scanner cooldown timers — only after signal is actually broadcast
            _now = datetime.now()
            _tt = signal.get('trade_type', '')
            if _tt == 'VOLUME_SCALP':
                _daily_scalp_signals += 1
                _last_scalp_time = _now
            elif _tt == 'SQUEEZE_BREAKOUT':
                _daily_squeeze_signals += 1
                _last_squeeze_time = _now
            elif _tt == 'SUPERTREND':
                _daily_supertrend_signals += 1
                _last_supertrend_time = _now
            elif _tt == 'MACD_MOMENTUM':
                _daily_macd_signals += 1
                _last_macd_time = _now
            elif _tt == 'RANGE_BREAKOUT':
                _daily_range_breakout_signals += 1
                _last_range_breakout_time = _now
            elif _tt == 'EMA_PULLBACK':
                _daily_ema_pullback_signals += 1
                _last_ema_pullback_time = _now
            elif _tt == 'HALF_BACK':
                _daily_half_back_signals += 1
                _last_half_back_time = _now
            elif _tt == 'OVERSOLD_REVERSAL':
                _daily_oversold_reversal_signals += 1
                _last_oversold_reversal_time = _now

            # Collect auto-trade eligible users — execution queued for sweep entry
            _sweep_trade_users: list = []

            # Send message + execute trade for each user
            for user in users_with_social:
                try:
                    prefs = user.preferences
                    
                    trade_type = signal.get('trade_type', '')
                    if trade_type == 'SQUEEZE_BREAKOUT' and prefs and not getattr(prefs, 'squeeze_mode_enabled', True):
                        continue
                    if trade_type == 'SUPERTREND' and prefs and not getattr(prefs, 'supertrend_mode_enabled', True):
                        continue
                    if trade_type == 'MACD_MOMENTUM' and prefs and not getattr(prefs, 'macd_mode_enabled', True):
                        continue
                    
                    # Use signal-type-specific leverage
                    if is_volume_scalp and prefs:
                        user_lev = getattr(prefs, 'scalp_leverage', 20) or 20
                        coin_type = "⚡"
                    elif is_news_signal and prefs:
                        if is_top:
                            user_lev = getattr(prefs, 'news_top_coin_leverage', 50) or 50
                            coin_type = "🏆"
                        else:
                            user_lev = getattr(prefs, 'news_leverage', 50) or 50
                            coin_type = "📊"
                    elif is_top:
                        user_lev = getattr(prefs, 'social_top_coin_leverage', 25) or 25 if prefs else 25
                        coin_type = "🏆"
                    else:
                        user_lev = getattr(prefs, 'social_leverage', 25) or 25 if prefs else 25
                        coin_type = "📊"
                    
                    signal_str = strength.get('total_score', 5) if strength else 5
                    if signal_str <= 3:
                        max_lev = 10
                    elif signal_str <= 5:
                        max_lev = 25
                    else:
                        max_lev = 50
                    user_lev = min(user_lev, max_lev)
                    
                    lev_line = f"\n\n{coin_type} {user_lev}x"
                    user_message = message + lev_line
                    
                    await bot.send_message(
                        user.telegram_id,
                        user_message,
                        parse_mode="HTML"
                    )
                    logger.info(f"📱 Sent social {direction} signal {symbol} to user {user.telegram_id} @ {user_lev}x")
                    
                    # AUTO-TRADE: Queue for sweep-entry watcher (better entry + tighter SL)
                    has_keys = prefs and prefs.bitunix_api_key and prefs.bitunix_api_secret
                    if has_keys:
                        if not prefs.auto_trading_enabled:
                            logger.info(f"📱 User {user.telegram_id} (ID {user.id}) - Auto-trading DISABLED, signal only")
                        else:
                            _sweep_trade_users.append((user.id, user.telegram_id, user_lev, sig_type))
                            logger.info(f"🎯 Queuing sweep-entry trade for {user.telegram_id} — {symbol} {direction} @ {user_lev}x")
                    else:
                        logger.info(f"📱 User {user.telegram_id} (ID {user.id}) - No Bitunix API keys, signal only")
                except Exception as e:
                    logger.error(f"Failed to send/execute social signal for {user.telegram_id}: {e}")

            # Queue sweep-entry watcher for auto-trade execution
            if _sweep_trade_users:
                from app.services.sweep_watcher import queue_sweep_entry
                _signal_id   = new_signal.id
                _symbol      = symbol
                _direction   = direction
                _entry       = entry
                _sl          = sl
                _users_snap  = list(_sweep_trade_users)
                _bot_ref     = bot

                async def _sweep_trade_callback(
                    sweep_entry: float, sweep_sl: float, sweep_hit: bool,
                    _sid=_signal_id, _sym=_symbol, _dir=_direction,
                    _users=_users_snap, _b=_bot_ref,
                ):
                    from app.database import SessionLocal as _SL
                    from app.models import User as _U, Signal as _Sig
                    from app.services.bitunix_trader import execute_bitunix_trade as _exec
                    _db = _SL()
                    try:
                        _sig_obj = _db.query(_Sig).filter(_Sig.id == _sid).first()
                        if _sig_obj and sweep_hit:
                            _sig_obj.entry_price = sweep_entry
                            _sig_obj.stop_loss   = sweep_sl
                            _db.commit()
                            _db.refresh(_sig_obj)
                        if not _sig_obj:
                            logger.error(f"❌ Sweep callback: signal {_sid} not found in DB")
                            return
                        for uid, tg_id, lev, trade_t in _users:
                            try:
                                _u = _db.query(_U).filter(_U.id == uid).first()
                                if not _u:
                                    continue
                                sweep_tag = " 🎯 Sweep entry" if sweep_hit else ""
                                logger.info(
                                    f"🔄 EXECUTING TRADE{sweep_tag}: {_sym} {_dir} "
                                    f"for user {tg_id} (ID {uid}) @ {lev}x"
                                )
                                trade_result = await _exec(
                                    signal=_sig_obj, user=_u, db=_db,
                                    trade_type=trade_t, leverage_override=lev,
                                )
                                if trade_result:
                                    logger.info(
                                        f"✅ Auto-traded{sweep_tag} {_sym} {_dir} "
                                        f"for user {tg_id} @ {lev}x"
                                    )
                                    await _b.send_message(
                                        tg_id,
                                        f"✅ <b>Trade Executed on Bitunix{sweep_tag}</b>\n"
                                        f"<b>{_sym}</b> {_dir} @ {lev}x",
                                        parse_mode="HTML",
                                    )
                                else:
                                    logger.warning(
                                        f"⚠️ Auto-trade BLOCKED for {_sym} user {tg_id}"
                                    )
                                    try:
                                        from app.services.bitunix_trader import notify_admin_trade_failure
                                        await notify_admin_trade_failure(
                                            _u, _sym,
                                            f"{trade_t} signal blocked at execution — "
                                            "check position limits, balance, or API keys",
                                        )
                                    except Exception:
                                        pass
                            except Exception as _te:
                                logger.error(
                                    f"❌ Sweep auto-trade FAILED for {_sym} user {tg_id}: {_te}"
                                )
                                try:
                                    await _b.send_message(
                                        tg_id,
                                        f"⚠️ <b>Auto-Trade Failed</b>\n"
                                        f"<b>{_sym}</b> {_dir}\n"
                                        f"<i>{str(_te)[:100]}</i>",
                                        parse_mode="HTML",
                                    )
                                except Exception:
                                    pass
                    finally:
                        _db.close()

                await queue_sweep_entry(_symbol, _direction, _entry, _sl, _sweep_trade_callback)

        await service.close()
        
    except Exception as e:
        logger.error(f"Error in social signal broadcast: {e}")
    finally:
        _social_scanning_active = False
