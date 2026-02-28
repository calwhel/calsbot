"""
AI Signal Filter - Uses Claude to validate trading signals before broadcast.

Reviews each signal candidate with market data and approves/rejects based on:
- Technical analysis quality
- Risk factors (BTC correlation, overextension)
- Market conditions
- Entry timing
"""

import os
import logging
import json
import asyncio
from typing import Dict, Optional
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

# ── Grok Macro Cache ─────────────────────────────────────────────────────────
_grok_macro_cache: Dict = {}
_grok_macro_last_refresh: Optional[datetime] = None
GROK_MACRO_CACHE_MINUTES = 45


def _get_grok_client():
    """Return an AsyncOpenAI client pointed at xAI, or None."""
    xai_key = os.getenv('XAI_API_KEY')
    if not xai_key:
        return None
    from openai import AsyncOpenAI
    return AsyncOpenAI(api_key=xai_key, base_url="https://api.x.ai/v1")


async def refresh_grok_macro_context() -> Dict:
    """
    Ask Grok for a global crypto macro briefing and cache it for 45 minutes.
    Called automatically before every signal evaluation cycle.
    """
    global _grok_macro_cache, _grok_macro_last_refresh
    grok = _get_grok_client()
    if not grok:
        return {}
    try:
        prompt = (
            "You are a real-time macro and geopolitical intelligence analyst for crypto traders. "
            "In the last 2-6 hours, what is driving crypto markets? Cover ALL of the following if relevant:\n"
            "- GEOPOLITICAL: Wars, conflicts, trade wars, tariffs, sanctions, diplomatic tensions, "
            "elections, government seizures of crypto, or any nation-state moves affecting risk sentiment.\n"
            "- MACRO/ECONOMIC: Fed decisions or commentary, CPI/PPI/jobs data, DXY strength, "
            "interest rate expectations, recession fears, commodity moves (oil, gold).\n"
            "- CRYPTO-SPECIFIC: BTC key levels and trend, major protocol events, exchange issues, "
            "regulatory actions, ETF flows, large liquidations, whale moves on X/chain.\n"
            "- SENTIMENT: Fear vs greed shift, major influencer narratives on X/Twitter, "
            "institutional positioning changes.\n"
            "Give a sharp 3-4 sentence briefing that tells a trader whether to lean long or short right now and WHY. "
            "End your response with exactly one of these tags on its own line: "
            "MACRO_BIAS: BULLISH  or  MACRO_BIAS: BEARISH  or  MACRO_BIAS: NEUTRAL"
        )
        response = await asyncio.wait_for(
            grok.chat.completions.create(
                model="grok-3-beta",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=300,
                temperature=0.3,
            ),
            timeout=20.0,
        )
        text = (response.choices[0].message.content or "").strip()
        bias = "NEUTRAL"
        for line in text.splitlines():
            if line.strip().startswith("MACRO_BIAS:"):
                tag = line.split(":", 1)[1].strip().upper()
                if tag in ("BULLISH", "BEARISH", "NEUTRAL"):
                    bias = tag
                break
        summary = text.replace(f"MACRO_BIAS: {bias}", "").strip()
        _grok_macro_cache = {"summary": summary, "bias": bias}
        _grok_macro_last_refresh = datetime.utcnow()
        logger.info(f"🌍 Grok macro refresh → bias={bias} | {summary[:120]}...")
        return _grok_macro_cache
    except asyncio.TimeoutError:
        logger.warning("Grok macro context timed out — using cached/empty")
        return _grok_macro_cache
    except Exception as e:
        logger.warning(f"Grok macro context error: {e}")
        return _grok_macro_cache


async def get_cached_grok_macro() -> Dict:
    """Return cached macro context, auto-refreshing if older than GROK_MACRO_CACHE_MINUTES."""
    if (
        not _grok_macro_last_refresh
        or datetime.utcnow() - _grok_macro_last_refresh > timedelta(minutes=GROK_MACRO_CACHE_MINUTES)
    ):
        return await refresh_grok_macro_context()
    return _grok_macro_cache


async def get_grok_coin_intelligence(symbol: str, direction: str) -> Dict:
    """
    Deep per-coin intelligence from Grok — protocol news, unlocks, whale moves,
    influencer calls, hack/rug risks. Returns a dict with:
      summary    : str   — 2-3 sentence context
      hard_no    : bool  — True if Grok flags a serious red flag
      hard_no_reason : str — reason if hard_no is True
    Times out in 15 seconds so it never blocks signal generation.
    """
    result = {"summary": "", "hard_no": False, "hard_no_reason": ""}
    grok = _get_grok_client()
    if not grok:
        return result
    try:
        coin = symbol.replace("USDT", "").replace("PERP", "").replace("-", "")
        prompt = (
            f"Analyze ${coin} for a {direction} scalp trade right now. "
            f"What do you know about: "
            f"(1) Recent protocol news or updates (last 24-48h)? "
            f"(2) Token unlock schedules or supply events? "
            f"(3) Exchange listings or delistings? "
            f"(4) Whale wallet activity or large transfers on X/chain? "
            f"(5) Influencer calls or viral X posts about this coin? "
            f"(6) Any hack, exploit, or rug pull risks? "
            f"(7) Current social momentum — rising or fading? "
            f"If there are SERIOUS red flags (hack, exploit, delisting, rug, massive token unlock) "
            f"that make a {direction} dangerous, start your response with: HARD_NO: [reason] "
            f"Otherwise give a 2-3 sentence factual summary. Be direct."
        )
        response = await asyncio.wait_for(
            grok.chat.completions.create(
                model="grok-3-beta",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=180,
                temperature=0.3,
            ),
            timeout=15.0,
        )
        text = (response.choices[0].message.content or "").strip()
        if text.upper().startswith("HARD_NO:"):
            reason = text.split(":", 1)[1].strip() if ":" in text else text
            result["hard_no"] = True
            result["hard_no_reason"] = reason[:200]
            logger.warning(f"🚨 Grok HARD VETO on {symbol} {direction}: {reason[:100]}")
        else:
            result["summary"] = text
            logger.info(f"🔍 Grok coin intel for {symbol}: {text[:120]}...")
        return result
    except asyncio.TimeoutError:
        logger.warning(f"Grok coin intelligence timed out for {symbol} — skipping")
        return result
    except Exception as e:
        logger.warning(f"Grok coin intelligence error for {symbol}: {e}")
        return result




def get_anthropic_client():
    """Get Anthropic Claude client - checks Replit AI Integrations first, then standalone key."""
    try:
        import anthropic
        
        # Check for Replit AI Integrations first
        base_url = os.environ.get("AI_INTEGRATIONS_ANTHROPIC_BASE_URL")
        api_key = os.environ.get("AI_INTEGRATIONS_ANTHROPIC_API_KEY")
        
        # Fall back to standalone Anthropic API key (for Railway)
        if not api_key:
            api_key = os.environ.get("ANTHROPIC_API_KEY")
            base_url = None
        
        if not api_key:
            logger.warning("No Anthropic API key found")
            return None
        
        if base_url:
            return anthropic.Anthropic(base_url=base_url, api_key=api_key)
        else:
            return anthropic.Anthropic(api_key=api_key)
    except Exception as e:
        logger.error(f"Failed to create Anthropic client: {e}")
        return None


def build_signal_prompt(
    signal_data: Dict,
    market_context: Optional[Dict] = None,
    grok_context: Optional[str] = None,
    grok_macro: Optional[Dict] = None,
) -> str:
    """Build the analysis prompt for Claude."""
    symbol = signal_data.get('symbol', 'UNKNOWN')
    direction = signal_data.get('direction', 'LONG')
    entry_price = signal_data.get('entry_price', 0)
    stop_loss = signal_data.get('stop_loss', 0)
    take_profit = signal_data.get('take_profit_1', signal_data.get('take_profit', 0))
    confidence = signal_data.get('confidence', 0)
    reasoning = signal_data.get('reasoning', '')
    change_24h = signal_data.get('24h_change', 0)
    volume_24h = signal_data.get('24h_volume', 0)
    is_parabolic = signal_data.get('is_parabolic_reversal', False)
    leverage = signal_data.get('leverage', 10)
    trade_type_label = signal_data.get('trade_type', 'STANDARD')

    # Calculate risk metrics
    if direction == 'LONG':
        sl_pct = ((entry_price - stop_loss) / entry_price) * 100 if entry_price > 0 else 0
        tp_pct = ((take_profit - entry_price) / entry_price) * 100 if entry_price > 0 else 0
    else:
        sl_pct = ((stop_loss - entry_price) / entry_price) * 100 if entry_price > 0 else 0
        tp_pct = ((entry_price - take_profit) / entry_price) * 100 if entry_price > 0 else 0

    rr_ratio = tp_pct / sl_pct if sl_pct > 0 else 0

    # Build BTC short-term context
    btc_context = ""
    if market_context:
        btc_summary = market_context.get('btc_summary', '')
        btc_verdict = market_context.get('btc_verdict', 'NEUTRAL')
        if btc_summary:
            btc_context = btc_summary
            if direction == 'LONG' and btc_verdict in ('BEARISH', 'OVERBOUGHT'):
                btc_context += f"\n⚠️ WARNING: BTC 5m is {btc_verdict} — be very selective with LONG entries."
            elif direction == 'SHORT' and btc_verdict in ('BULLISH', 'OVERSOLD'):
                btc_context += f"\n⚠️ WARNING: BTC 5m is {btc_verdict} — be very selective with SHORT entries."

    # Build Grok macro context string
    macro_bias = ""
    macro_summary = ""
    if grok_macro:
        macro_bias = grok_macro.get('bias', 'NEUTRAL')
        macro_summary = grok_macro.get('summary', '')
        if macro_bias == 'BEARISH' and direction == 'LONG':
            btc_context += f"\n⚠️ GROK MACRO ALERT: Current macro is BEARISH — extra caution on LONG entries."
        elif macro_bias == 'BULLISH' and direction == 'SHORT':
            btc_context += f"\n⚠️ GROK MACRO ALERT: Current macro is BULLISH — extra caution on SHORT entries."

    trade_type = "PARABOLIC REVERSAL SHORT" if is_parabolic else f"{direction}"

    # Append past trade lessons for this direction/type to inform decision
    lessons_context = ""
    try:
        from app.services.ai_trade_learner import format_lessons_for_ai_prompt
        lessons_context = format_lessons_for_ai_prompt(
            trade_type=trade_type_label,
            direction=direction,
            symbol=symbol
        )
    except Exception:
        pass

    # Inject live system performance context
    live_context = ""
    try:
        from app.services.ai_trade_learner import get_live_trading_context
        live_context = get_live_trading_context()
    except Exception:
        pass

    return f"""You are a professional crypto trading analyst. Analyze this trade signal and decide if it should be executed.

SIGNAL DETAILS:
- Symbol: {symbol}
- Direction: {trade_type}
- Entry: ${entry_price:.6f}
- Stop Loss: ${stop_loss:.6f} ({sl_pct:.2f}% risk)
- Take Profit: ${take_profit:.6f} ({tp_pct:.2f}% target)
- Risk/Reward: {rr_ratio:.2f}:1
- Leverage: {leverage}x
- 24h Change: {change_24h:+.1f}%
- 24h Volume: ${volume_24h:,.0f}
- Signal Confidence: {confidence}%
- Technical Reasoning: {reasoning}

MARKET CONTEXT:
{btc_context if btc_context else "No BTC context available."}
{live_context}
{lessons_context}

GROK INTELLIGENCE (real-time world & crypto awareness):
Macro environment ({macro_bias if macro_bias else "UNKNOWN"}): {macro_summary if macro_summary else "No macro data available."}
Coin-specific intel: {grok_context if grok_context else "No coin-specific intelligence available."}

STRATEGY RULES:
- LONGS: Enter early momentum (0-12% pumps), TP at 67%, SL at 65% @ 20x
- SHORTS: Mean reversion on 35%+ gainers, target pullback
- PARABOLIC: Aggressive shorts on 50%+ exhausted pumps, 200% TP @ 20x

Analyze this signal considering:
1. Is the entry timing good? (not chasing, not too early)
2. Is the risk/reward acceptable?
3. Does the technical setup support this trade?
4. Is BTC's short-term momentum ALIGNED with this trade direction? A LONG during a BTC bearish 15m phase or a SHORT during a BTC bullish 15m phase is a significant red flag — lower confidence or reject unless the coin's setup is exceptionally strong.
5. Does the X/Twitter sentiment support or contradict this trade?
6. Would you take this trade with real money?
7. What is the single most important reason to take or skip this trade?

Respond in JSON format only:
{{
    "approved": true or false,
    "confidence": 1-10 (how confident you are in this trade),
    "recommendation": "STRONG BUY" or "BUY" or "HOLD" or "AVOID",
    "reasoning": "2-3 sentence plain English explanation for traders",
    "why_this_trade": "1-2 sentence plain English explanation of WHY this specific trade is being taken right now - focus on the key edge/catalyst. Make it actionable and easy for non-technical traders to understand.",
    "risks": ["list", "of", "key", "risks"],
    "entry_quality": "EXCELLENT" or "GOOD" or "FAIR" or "POOR"
}}"""


async def analyze_signal_with_ai(
    signal_data: Dict,
    market_context: Optional[Dict] = None
) -> Dict:
    """
    Use Claude to analyze a trading signal and decide if it should be broadcast.
    
    Args:
        signal_data: The signal candidate with all technical data
        market_context: Optional BTC/market data for correlation analysis
    
    Returns:
        {
            'approved': True/False,
            'confidence': 1-10,
            'reasoning': 'Plain English explanation',
            'risks': ['risk1', 'risk2'],
            'recommendation': 'STRONG BUY / BUY / HOLD / AVOID'
        }
    """
    try:
        client = get_anthropic_client()
        if not client:
            raise ValueError("Claude client not available")

        symbol = signal_data.get('symbol', 'UNKNOWN')
        direction = signal_data.get('direction', 'LONG')

        # Run Grok macro cache refresh + coin deep intelligence in parallel
        # Grok is the prime AI for world events, news, and sentiment
        grok_macro_result, coin_intel = await asyncio.gather(
            get_cached_grok_macro(),
            get_grok_coin_intelligence(symbol, direction),
            return_exceptions=True,
        )
        if isinstance(grok_macro_result, Exception):
            grok_macro_result = {}
        if isinstance(coin_intel, Exception):
            coin_intel = {"summary": "", "hard_no": False, "hard_no_reason": ""}

        # Grok hard veto — block immediately without spending Claude tokens
        if isinstance(coin_intel, dict) and coin_intel.get('hard_no'):
            reason = coin_intel.get('hard_no_reason', 'Serious red flag detected by Grok')
            logger.warning(f"🚨 GROK HARD VETO: {symbol} {direction} blocked — {reason}")
            return {
                'approved': False,
                'confidence': 1,
                'recommendation': 'AVOID',
                'reasoning': f"Grok intelligence flagged a serious risk: {reason}",
                'why_this_trade': '',
                'risks': [reason],
                'entry_quality': 'POOR',
            }

        coin_summary = coin_intel.get('summary', '') if isinstance(coin_intel, dict) else ''

        # Build the prompt with Grok macro + coin intel injected
        prompt = build_signal_prompt(
            signal_data,
            market_context,
            grok_context=coin_summary,
            grok_macro=grok_macro_result if isinstance(grok_macro_result, dict) else {},
        )

        macro_bias = grok_macro_result.get('bias', '?') if isinstance(grok_macro_result, dict) else '?'
        logger.info(
            f"🧠 Claude analyzing {symbol} {direction} | "
            f"Grok macro={macro_bias} | coin_intel={'✅' if coin_summary else '⬜'}"
        )
        
        # Run sync client in executor
        def call_claude():
            response = client.messages.create(
                model="claude-sonnet-4-5-20250929",  # Latest Claude Sonnet 4.5
                max_tokens=300,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                system="You are a professional crypto trading analyst. Be concise and decisive. Always respond in valid JSON only, no other text."
            )
            # Get text from first text block
            for block in response.content:
                if hasattr(block, 'text'):
                    return block.text
            return "{}"
        
        loop = asyncio.get_event_loop()
        result_text = await loop.run_in_executor(None, call_claude)
        
        # Parse JSON from response
        try:
            # Try to extract JSON if wrapped in markdown
            if "```json" in result_text:
                result_text = result_text.split("```json")[1].split("```")[0].strip()
            elif "```" in result_text:
                result_text = result_text.split("```")[1].split("```")[0].strip()
            result = json.loads(result_text)
        except json.JSONDecodeError:
            logger.warning(f"Claude returned non-JSON: {result_text[:200]}")
            raise
        
        # Ensure all required fields exist
        result.setdefault('approved', False)
        result.setdefault('confidence', 5)
        result.setdefault('recommendation', 'HOLD')
        result.setdefault('reasoning', 'Unable to analyze signal.')
        result.setdefault('why_this_trade', '')
        result.setdefault('risks', [])
        result.setdefault('entry_quality', 'FAIR')

        logger.info(f"🧠 Claude Analysis for {symbol} {direction}: {'✅ APPROVED' if result['approved'] else '❌ REJECTED'} ({result['recommendation']})")
        logger.info(f"   Reasoning: {result['reasoning']}")
        if result.get('why_this_trade'):
            logger.info(f"   Why: {result['why_this_trade']}")

        return result

    except Exception as e:
        logger.error(f"Claude Signal Filter error: {e}")
        # On error, approve signal to not block trading
        return {
            'approved': True,
            'confidence': 5,
            'recommendation': 'BUY',
            'reasoning': f'AI analysis unavailable, proceeding with technical signals. (Error: {str(e)[:50]})',
            'why_this_trade': '',
            'risks': ['AI analysis failed'],
            'entry_quality': 'UNKNOWN'
        }


def _ema(values: list, period: int) -> float:
    """Calculate EMA from a list of floats."""
    if len(values) < period:
        return values[-1] if values else 0.0
    k = 2.0 / (period + 1)
    ema = sum(values[:period]) / period
    for v in values[period:]:
        ema = v * k + ema * (1 - k)
    return ema


def _rsi(closes: list, period: int = 14) -> float:
    """Calculate RSI from a list of close prices."""
    if len(closes) < period + 1:
        return 50.0
    gains, losses = [], []
    for i in range(1, len(closes)):
        diff = closes[i] - closes[i - 1]
        gains.append(max(diff, 0))
        losses.append(max(-diff, 0))
    avg_gain = sum(gains[-period:]) / period
    avg_loss = sum(losses[-period:]) / period
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


async def get_btc_context() -> Optional[Dict]:
    """
    Get BTC short-term context: 15m RSI, 8/21 EMA trend, recent candle direction.
    Returns a structured dict with a human-readable 'summary' and a 'verdict'.
    """
    try:
        import aiohttp
        url = "https://fapi.binance.com/fapi/v1/klines?symbol=BTCUSDT&interval=5m&limit=50"
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=6)) as resp:
                klines = await resp.json()

        if not klines or len(klines) < 5:
            return None

        closes = [float(k[4]) for k in klines]
        current_price = closes[-1]

        rsi_val = _rsi(closes)
        ema8 = _ema(closes, 8)
        ema21 = _ema(closes, 21)

        last3 = closes[-4:]
        candle_changes = [(last3[i] - last3[i - 1]) / last3[i - 1] * 100 for i in range(1, 4)]
        recent_direction = sum(1 if c > 0 else -1 for c in candle_changes)

        above_ema8 = current_price > ema8
        above_ema21 = current_price > ema21
        ema_bullish = above_ema8 and above_ema21
        ema_bearish = not above_ema8 and not above_ema21

        if ema_bullish and rsi_val > 50 and recent_direction >= 1:
            verdict = "BULLISH"
        elif ema_bearish and rsi_val < 50 and recent_direction <= -1:
            verdict = "BEARISH"
        elif rsi_val > 70:
            verdict = "OVERBOUGHT"
        elif rsi_val < 30:
            verdict = "OVERSOLD"
        else:
            verdict = "NEUTRAL"

        change_str = ", ".join(f"{c:+.2f}%" for c in candle_changes)
        summary = (
            f"BTC 5m: RSI {rsi_val:.0f} | "
            f"{'above' if above_ema8 else 'below'} 8EMA, "
            f"{'above' if above_ema21 else 'below'} 21EMA | "
            f"Last 3 candles: {change_str} | "
            f"Verdict: {verdict}"
        )

        logger.info(f"📊 BTC short-term → {summary}")
        return {
            'btc_price': current_price,
            'btc_rsi_15m': rsi_val,
            'btc_ema8': ema8,
            'btc_ema21': ema21,
            'btc_verdict': verdict,
            'btc_summary': summary,
        }

    except Exception as e:
        logger.warning(f"Could not fetch BTC short-term context: {e}")
        return None


def format_ai_analysis_for_signal(ai_result: Dict) -> str:
    """Format AI analysis for inclusion in signal message (hidden - returns empty)."""
    # AI analysis is internal only - don't show in signal output
    return ""


# Minimum confidence to approve a signal
MIN_AI_CONFIDENCE = 8


async def should_broadcast_signal(signal_data: Dict) -> tuple[bool, str]:
    """
    Main entry point - check if signal should be broadcast.

    Returns:
        (should_broadcast: bool, ai_analysis_text: str)
        ai_analysis_text contains the WHY THIS TRADE explanation when approved.
    """
    # Short-circuit: very high confidence signals skip Claude to reduce API costs
    pre_score = signal_data.get('confidence', 0)
    if isinstance(pre_score, (int, float)) and pre_score >= 90:
        logger.info(f"⚡ Signal pre-score {pre_score}% — auto-approved, skipping Claude")
        return True, ""

    # Get market context
    btc_context = await get_btc_context()

    # Analyze with Claude
    ai_result = await analyze_signal_with_ai(signal_data, btc_context)

    # Decision logic
    approved = ai_result.get('approved', False)
    confidence = ai_result.get('confidence', 0)

    # Require both approval AND minimum confidence
    should_broadcast = approved and confidence >= MIN_AI_CONFIDENCE

    if should_broadcast:
        why = ai_result.get('why_this_trade', '').strip()
        if why:
            analysis_text = f"\n<b>💡 Why this trade:</b> <i>{why}</i>\n"
        else:
            analysis_text = ""
    else:
        rejection_reason = ai_result.get('reasoning', 'Did not meet quality standards')
        risks = ai_result.get('risks', [])
        logger.info(f"🚫 Signal REJECTED by Claude: {rejection_reason}")
        if risks:
            logger.info(f"   Risks: {', '.join(risks)}")
        analysis_text = ""

    return should_broadcast, analysis_text
