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

logger = logging.getLogger(__name__)


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


def build_signal_prompt(signal_data: Dict, market_context: Optional[Dict] = None) -> str:
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

    # Build context for AI
    btc_context = ""
    if market_context:
        btc_change = market_context.get('btc_24h_change', 0)
        btc_trend = "bullish" if btc_change > 0 else "bearish"
        btc_context = f"BTC is currently {btc_trend} ({btc_change:+.1f}% in 24h)."

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
6. What is the single most important reason to take or skip this trade?

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
        
        # Build the prompt
        prompt = build_signal_prompt(signal_data, market_context)
        symbol = signal_data.get('symbol', 'UNKNOWN')
        direction = signal_data.get('direction', 'LONG')
        
        logger.info(f"🧠 Using Claude Sonnet 4.5 to analyze {symbol} {direction}...")
        
        # Run sync client in executor
        def call_claude():
            response = client.messages.create(
                model="claude-sonnet-4-5-20250929",  # Latest Claude Sonnet 4.5
                max_tokens=500,
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
