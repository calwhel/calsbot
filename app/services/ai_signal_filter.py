"""
AI Signal Filter - Uses GPT to validate trading signals before broadcast.

Reviews each signal candidate with market data and approves/rejects based on:
- Technical analysis quality
- Risk factors (BTC correlation, overextension)
- Market conditions
- Entry timing
"""

import os
import logging
import json
from typing import Dict, Optional
from datetime import datetime

from openai import OpenAI

logger = logging.getLogger(__name__)

def get_openai_client():
    """Get or create OpenAI client - always reads fresh API key."""
    api_key = os.environ.get("AI_INTEGRATIONS_OPENAI_API_KEY")
    if not api_key:
        raise ValueError("AI_INTEGRATIONS_OPENAI_API_KEY not set in environment")
    return OpenAI(api_key=api_key)


async def analyze_signal_with_ai(
    signal_data: Dict,
    market_context: Optional[Dict] = None
) -> Dict:
    """
    Use AI to analyze a trading signal and decide if it should be broadcast.
    
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
        client = get_openai_client()
        
        # Extract key data from signal
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
        
        # Calculate risk metrics
        if direction == 'LONG':
            sl_pct = ((entry_price - stop_loss) / entry_price) * 100 if entry_price > 0 else 0
            tp_pct = ((take_profit - entry_price) / entry_price) * 100 if entry_price > 0 else 0
        else:
            sl_pct = ((stop_loss - entry_price) / entry_price) * 100 if entry_price > 0 else 0
            tp_pct = ((entry_price - take_profit) / entry_price) * 100 if entry_price > 0 else 0
        
        rr_ratio = tp_pct / sl_pct if sl_pct > 0 else 0
        
        # Build context for AI
        btc_context = ""
        if market_context:
            btc_change = market_context.get('btc_24h_change', 0)
            btc_trend = "bullish" if btc_change > 0 else "bearish"
            btc_context = f"BTC is currently {btc_trend} ({btc_change:+.1f}% in 24h)."
        
        trade_type = "PARABOLIC REVERSAL SHORT" if is_parabolic else f"{direction}"
        
        prompt = f"""You are a professional crypto trading analyst. Analyze this trade signal and decide if it should be executed.

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

STRATEGY RULES:
- LONGS: Enter early momentum (0-12% pumps), TP at 67%, SL at 65% @ 20x
- SHORTS: Mean reversion on 35%+ gainers, target pullback
- PARABOLIC: Aggressive shorts on 50%+ exhausted pumps, 200% TP @ 20x

Analyze this signal considering:
1. Is the entry timing good? (not chasing, not too early)
2. Is the risk/reward acceptable?
3. Does the technical setup support this trade?
4. Are there any red flags (overextension, low volume, BTC correlation risk)?
5. Would you take this trade with real money?

Respond in JSON format:
{{
    "approved": true or false,
    "confidence": 1-10 (how confident you are in this trade),
    "recommendation": "STRONG BUY" or "BUY" or "HOLD" or "AVOID",
    "reasoning": "2-3 sentence plain English explanation for traders",
    "risks": ["list", "of", "key", "risks"],
    "entry_quality": "EXCELLENT" or "GOOD" or "FAIR" or "POOR"
}}"""

        # the newest OpenAI model is "gpt-5" which was released August 7, 2025.
        # do not change this unless explicitly requested by the user
        # Retry up to 2 times on connection errors
        import asyncio
        last_error = None
        for attempt in range(3):
            try:
                response = client.chat.completions.create(
                    model="gpt-4o-mini",  # Using faster model for signal filtering
                    messages=[
                        {"role": "system", "content": "You are a professional crypto trading analyst. Be concise and decisive. Always respond in valid JSON."},
                        {"role": "user", "content": prompt}
                    ],
                    response_format={"type": "json_object"},
                    max_completion_tokens=500,
                    timeout=15.0  # 15 second timeout
                )
                break  # Success, exit retry loop
            except Exception as retry_error:
                last_error = retry_error
                if attempt < 2:
                    await asyncio.sleep(1)  # Wait 1 second before retry
                    continue
                raise last_error
        
        result_text = response.choices[0].message.content or "{}"
        result = json.loads(result_text)
        
        # Ensure all required fields exist
        result.setdefault('approved', False)
        result.setdefault('confidence', 5)
        result.setdefault('recommendation', 'HOLD')
        result.setdefault('reasoning', 'Unable to analyze signal.')
        result.setdefault('risks', [])
        result.setdefault('entry_quality', 'FAIR')
        
        logger.info(f"🤖 AI Analysis for {symbol} {direction}: {'✅ APPROVED' if result['approved'] else '❌ REJECTED'} ({result['recommendation']})")
        logger.info(f"   Reasoning: {result['reasoning']}")
        
        return result
        
    except Exception as e:
        logger.error(f"AI Signal Filter error: {e}")
        # On error, approve signal to not block trading
        return {
            'approved': True,
            'confidence': 5,
            'recommendation': 'BUY',
            'reasoning': f'AI analysis unavailable, proceeding with technical signals. (Error: {str(e)[:50]})',
            'risks': ['AI analysis failed'],
            'entry_quality': 'UNKNOWN'
        }


async def get_btc_context() -> Optional[Dict]:
    """Get BTC market context for correlation analysis."""
    try:
        import ccxt.async_support as ccxt
        
        exchange = ccxt.binance({'enableRateLimit': True})
        try:
            ticker = await exchange.fetch_ticker('BTC/USDT')
            return {
                'btc_price': ticker['last'],
                'btc_24h_change': ticker['percentage'] or 0,
                'btc_volume': ticker['quoteVolume'] or 0
            }
        finally:
            await exchange.close()
    except Exception as e:
        logger.warning(f"Could not fetch BTC context: {e}")
        return None


def format_ai_analysis_for_signal(ai_result: Dict) -> str:
    """Format AI analysis for inclusion in signal message."""
    if not ai_result.get('approved'):
        return ""
    
    recommendation = ai_result.get('recommendation', 'BUY')
    confidence = ai_result.get('confidence', 5)
    reasoning = ai_result.get('reasoning', '')
    entry_quality = ai_result.get('entry_quality', 'FAIR')
    
    # Emoji based on recommendation
    rec_emoji = {
        'STRONG BUY': '🟢🟢',
        'BUY': '🟢',
        'HOLD': '🟡',
        'AVOID': '🔴'
    }.get(recommendation, '⚪')
    
    # Confidence bar
    conf_bar = '█' * (confidence // 2) + '░' * (5 - confidence // 2)
    
    return f"""
<b>🤖 AI Analysis</b>
{rec_emoji} <b>{recommendation}</b> | Entry: {entry_quality}
Confidence: [{conf_bar}] {confidence}/10

<i>{reasoning}</i>
"""


# Minimum confidence to approve a signal
MIN_AI_CONFIDENCE = 5


async def should_broadcast_signal(signal_data: Dict) -> tuple[bool, str]:
    """
    Main entry point - check if signal should be broadcast.
    
    Returns:
        (should_broadcast: bool, ai_analysis_text: str)
    """
    # Get market context
    btc_context = await get_btc_context()
    
    # Analyze with AI
    ai_result = await analyze_signal_with_ai(signal_data, btc_context)
    
    # Decision logic
    approved = ai_result.get('approved', False)
    confidence = ai_result.get('confidence', 0)
    
    # Require both approval AND minimum confidence
    should_broadcast = approved and confidence >= MIN_AI_CONFIDENCE
    
    if should_broadcast:
        analysis_text = format_ai_analysis_for_signal(ai_result)
    else:
        rejection_reason = ai_result.get('reasoning', 'Did not meet quality standards')
        risks = ai_result.get('risks', [])
        logger.info(f"🚫 Signal REJECTED by AI: {rejection_reason}")
        if risks:
            logger.info(f"   Risks: {', '.join(risks)}")
        analysis_text = ""
    
    return should_broadcast, analysis_text
