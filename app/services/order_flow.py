import logging
import asyncio
import os
from typing import Dict, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

try:
    from app.services.coinglass import get_derivatives_summary
except ImportError:
    get_derivatives_summary = None


async def analyze_order_flow(
    symbol: str,
    price_change_1h: float = 0,
    price_change_24h: float = 0,
    current_price: float = 0,
    volume_24h: float = 0,
    ta_data: Dict = None,
) -> Optional[Dict]:
    """Analyze order flow using derivatives data + price action.
    
    Computes a directional flow_score (-100 to +100):
    - Positive = aggressive buying / bullish flow
    - Negative = aggressive selling / bearish flow
    
    Also detects special conditions: squeeze risk, liquidation cascade, divergence.
    """
    try:
        if get_derivatives_summary is None:
            logger.warning("Derivatives API not available for order flow analysis")
            return None

        deriv = await get_derivatives_summary(symbol)
        if not deriv:
            logger.debug(f"No derivatives data available for {symbol}")
            return None

        flow_score = 0
        flow_signals = []
        special_conditions = []

        funding_rate = deriv.get('funding_rate')
        if funding_rate is not None:
            if funding_rate > 0.1:
                flow_score -= 15
                flow_signals.append(f"Extreme positive funding ({funding_rate:.4f}%) = longs paying, overcrowded")
                special_conditions.append('OVERCROWDED_LONGS')
            elif funding_rate > 0.05:
                flow_score -= 8
                flow_signals.append(f"High positive funding ({funding_rate:.4f}%) = long bias")
            elif funding_rate < -0.05:
                flow_score += 12
                flow_signals.append(f"Negative funding ({funding_rate:.4f}%) = shorts paying, squeeze potential")
                special_conditions.append('SHORT_SQUEEZE_RISK')
            elif funding_rate < -0.03:
                flow_score += 6
                flow_signals.append(f"Slightly negative funding ({funding_rate:.4f}%) = mild short bias")

        oi_change = deriv.get('oi_change_pct')
        if oi_change is not None:
            if price_change_1h > 2 and oi_change > 5:
                flow_score += 20
                flow_signals.append(f"Price up {price_change_1h:+.1f}% + OI up {oi_change:+.1f}% = aggressive buying, new positions entering")
            elif price_change_1h > 2 and oi_change < -3:
                flow_score += 8
                flow_signals.append(f"Price up {price_change_1h:+.1f}% + OI down {oi_change:+.1f}% = short covering rally")
            elif price_change_1h < -2 and oi_change > 5:
                flow_score -= 20
                flow_signals.append(f"Price down {price_change_1h:+.1f}% + OI up {oi_change:+.1f}% = aggressive shorting, new short positions")
            elif price_change_1h < -2 and oi_change < -3:
                flow_score -= 8
                flow_signals.append(f"Price down {price_change_1h:+.1f}% + OI down {oi_change:+.1f}% = long liquidation cascade")
                special_conditions.append('LONG_LIQUIDATION_CASCADE')

            if abs(oi_change) > 15:
                flow_signals.append(f"Extreme OI change {oi_change:+.1f}% = major position shifts")
                special_conditions.append('EXTREME_OI_CHANGE')

        ls_ratio = deriv.get('long_short_ratio')
        long_pct = deriv.get('long_pct')
        short_pct = deriv.get('short_pct')
        ls_bias = deriv.get('ls_bias', 'NEUTRAL')

        if ls_ratio is not None:
            if ls_ratio > 3.0:
                flow_score -= 12
                flow_signals.append(f"L/S ratio {ls_ratio:.2f} (extreme long bias {long_pct:.0f}%/{short_pct:.0f}%) = contrarian short setup")
                special_conditions.append('EXTREME_LONG_CROWDING')
            elif ls_ratio > 2.0:
                flow_score -= 5
                flow_signals.append(f"L/S ratio {ls_ratio:.2f} (long heavy) = moderate crowding risk")
            elif ls_ratio < 0.5:
                flow_score += 12
                flow_signals.append(f"L/S ratio {ls_ratio:.2f} (extreme short bias {long_pct:.0f}%/{short_pct:.0f}%) = contrarian long / squeeze")
                special_conditions.append('EXTREME_SHORT_CROWDING')
            elif ls_ratio < 0.7:
                flow_score += 5
                flow_signals.append(f"L/S ratio {ls_ratio:.2f} (short heavy) = potential squeeze")

        long_liq = deriv.get('long_liquidation_usd', 0)
        short_liq = deriv.get('short_liquidation_usd', 0)
        total_liq = long_liq + short_liq

        if total_liq > 0:
            if long_liq > short_liq * 3 and long_liq > 1_000_000:
                flow_score -= 10
                flow_signals.append(f"Heavy long liquidations ${long_liq/1e6:.1f}M vs short ${short_liq/1e6:.1f}M = bearish pressure")
                special_conditions.append('LONG_LIQUIDATION_DOMINANT')
            elif short_liq > long_liq * 3 and short_liq > 1_000_000:
                flow_score += 10
                flow_signals.append(f"Heavy short liquidations ${short_liq/1e6:.1f}M vs long ${long_liq/1e6:.1f}M = bullish short squeeze")
                special_conditions.append('SHORT_LIQUIDATION_DOMINANT')

            if total_liq > 10_000_000:
                flow_signals.append(f"Total liquidations ${total_liq/1e6:.1f}M = extreme volatility event")
                special_conditions.append('MASSIVE_LIQUIDATION_EVENT')

        if ta_data:
            rsi = ta_data.get('rsi_15m')
            vwap = ta_data.get('vwap', {})
            bb = ta_data.get('bollinger', {})

            if rsi is not None:
                if rsi > 75 and flow_score > 10:
                    flow_signals.append(f"RSI overbought ({rsi:.0f}) confirms aggressive buying but exhaustion risk")
                elif rsi < 25 and flow_score < -10:
                    flow_signals.append(f"RSI oversold ({rsi:.0f}) confirms aggressive selling but bounce risk")

            vwap_pos = vwap.get('position', '') if vwap else ''
            vwap_dev = vwap.get('deviation_pct', 0) if vwap else 0
            if vwap_pos == 'ABOVE' and vwap_dev > 3 and flow_score > 0:
                flow_signals.append(f"Price {vwap_dev:+.1f}% above VWAP with bullish flow = institutional buying above value")
            elif vwap_pos == 'BELOW' and vwap_dev < -3 and flow_score < 0:
                flow_signals.append(f"Price {vwap_dev:+.1f}% below VWAP with bearish flow = distribution below value")

            bb_squeeze = bb.get('squeeze', '') if bb else ''
            if bb_squeeze == 'TIGHT' and abs(flow_score) > 15:
                special_conditions.append('SQUEEZE_BREAKOUT_IMMINENT')
                flow_signals.append("Tight Bollinger squeeze + strong directional flow = explosive move building")

        flow_score = max(-100, min(100, flow_score))

        if flow_score > 20:
            flow_direction = 'AGGRESSIVE_BUYING'
        elif flow_score > 10:
            flow_direction = 'MILD_BUYING'
        elif flow_score < -20:
            flow_direction = 'AGGRESSIVE_SELLING'
        elif flow_score < -10:
            flow_direction = 'MILD_SELLING'
        else:
            flow_direction = 'NEUTRAL'

        return {
            'flow_score': flow_score,
            'flow_direction': flow_direction,
            'flow_signals': flow_signals,
            'special_conditions': special_conditions,
            'derivatives': deriv,
            'symbol': symbol,
            'timestamp': datetime.utcnow().isoformat(),
        }

    except Exception as e:
        logger.error(f"Order flow analysis error for {symbol}: {e}", exc_info=True)
        return None


async def ai_analyze_order_flow(flow_data: Dict) -> Optional[Dict]:
    """Use Gemini to analyze order flow data and generate actionable signal."""
    if not flow_data or abs(flow_data.get('flow_score', 0)) < 15:
        return None

    try:
        import google.generativeai as genai

        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("AI_INTEGRATIONS_GEMINI_API_KEY")
        if not api_key:
            logger.warning("No Gemini API key for order flow AI analysis")
            return None

        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.5-flash')

        symbol = flow_data['symbol']
        score = flow_data['flow_score']
        direction = flow_data['flow_direction']
        signals = flow_data['flow_signals']
        conditions = flow_data['special_conditions']

        prompt = f"""Analyze this crypto order flow data for ${symbol.replace('USDT', '').replace('/USDT', '')} and provide a trading recommendation.

ORDER FLOW ANALYSIS:
- Flow Score: {score}/100 ({'bullish' if score > 0 else 'bearish'})
- Direction: {direction}
- Special Conditions: {', '.join(conditions) if conditions else 'None'}

DETAILED SIGNALS:
{chr(10).join(f'• {s}' for s in signals)}

Based on this order flow data, provide:
1. DIRECTION: LONG or SHORT or SKIP
2. CONFIDENCE: 1-10
3. REASONING: 1-2 sentences explaining the flow dynamics
4. RISK_LEVEL: LOW/MEDIUM/HIGH
5. FLOW_QUALITY: How clean/reliable is this flow signal (1-10)

Respond in exactly this format:
DIRECTION: [LONG/SHORT/SKIP]
CONFIDENCE: [1-10]
REASONING: [explanation]
RISK_LEVEL: [LOW/MEDIUM/HIGH]
FLOW_QUALITY: [1-10]"""

        response = await asyncio.to_thread(
            model.generate_content,
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.3,
                max_output_tokens=300,
            )
        )

        text = response.text.strip()
        result = _parse_ai_flow_response(text)
        result['raw_response'] = text
        return result

    except Exception as e:
        logger.error(f"AI order flow analysis error: {e}", exc_info=True)
        return None


def _parse_ai_flow_response(text: str) -> Dict:
    """Parse structured AI response for order flow."""
    result = {
        'direction': 'SKIP',
        'confidence': 5,
        'reasoning': '',
        'risk_level': 'MEDIUM',
        'flow_quality': 5,
    }

    for line in text.split('\n'):
        line = line.strip()
        if line.startswith('DIRECTION:'):
            val = line.split(':', 1)[1].strip().upper()
            if val in ('LONG', 'SHORT', 'SKIP'):
                result['direction'] = val
        elif line.startswith('CONFIDENCE:'):
            try:
                result['confidence'] = min(10, max(1, int(line.split(':', 1)[1].strip().split('/')[0].strip())))
            except (ValueError, IndexError):
                pass
        elif line.startswith('REASONING:'):
            result['reasoning'] = line.split(':', 1)[1].strip()
        elif line.startswith('RISK_LEVEL:'):
            val = line.split(':', 1)[1].strip().upper()
            if val in ('LOW', 'MEDIUM', 'HIGH'):
                result['risk_level'] = val
        elif line.startswith('FLOW_QUALITY:'):
            try:
                result['flow_quality'] = min(10, max(1, int(line.split(':', 1)[1].strip().split('/')[0].strip())))
            except (ValueError, IndexError):
                pass

    return result


def format_order_flow_for_ai(flow_data: Dict) -> str:
    """Format order flow analysis for inclusion in signal AI prompts."""
    if not flow_data:
        return ""

    lines = ["\n--- ORDER FLOW ANALYSIS ---"]
    lines.append(f"Flow Score: {flow_data['flow_score']}/100 ({flow_data['flow_direction']})")

    conditions = flow_data.get('special_conditions', [])
    if conditions:
        lines.append(f"Special Conditions: {', '.join(conditions)}")

    for sig in flow_data.get('flow_signals', [])[:5]:
        lines.append(f"  • {sig}")

    ai_result = flow_data.get('ai_analysis')
    if ai_result:
        lines.append(f"AI Flow Assessment: {ai_result.get('direction', 'N/A')} (confidence {ai_result.get('confidence', 'N/A')}/10, quality {ai_result.get('flow_quality', 'N/A')}/10)")
        if ai_result.get('reasoning'):
            lines.append(f"  Reasoning: {ai_result['reasoning']}")

    return "\n".join(lines)


def format_order_flow_for_message(flow_data: Dict) -> str:
    """Format order flow data for Telegram signal message."""
    if not flow_data:
        return ""

    score = flow_data['flow_score']
    direction = flow_data['flow_direction']

    if 'BUYING' in direction:
        icon = "🟢"
    elif 'SELLING' in direction:
        icon = "🔴"
    else:
        icon = "⚪"

    parts = [f"{icon} <b>Order Flow: {direction.replace('_', ' ')}</b> ({score:+d}/100)"]

    conditions = flow_data.get('special_conditions', [])
    if conditions:
        cond_icons = {
            'SHORT_SQUEEZE_RISK': '⚡',
            'LONG_LIQUIDATION_CASCADE': '💥',
            'EXTREME_OI_CHANGE': '📊',
            'OVERCROWDED_LONGS': '⚠️',
            'EXTREME_LONG_CROWDING': '⚠️',
            'EXTREME_SHORT_CROWDING': '⚡',
            'SHORT_LIQUIDATION_DOMINANT': '🚀',
            'LONG_LIQUIDATION_DOMINANT': '📉',
            'MASSIVE_LIQUIDATION_EVENT': '🌊',
            'SQUEEZE_BREAKOUT_IMMINENT': '💣',
        }
        for cond in conditions[:3]:
            c_icon = cond_icons.get(cond, '🔔')
            parts.append(f"  {c_icon} {cond.replace('_', ' ')}")

    top_signals = flow_data.get('flow_signals', [])[:2]
    for sig in top_signals:
        parts.append(f"  📋 {sig}")

    return "\n".join(parts)


def get_flow_score_modifier(flow_data: Dict, direction: str) -> float:
    """Return score modifier (-2 to +2) based on order flow alignment with trade direction."""
    if not flow_data:
        return 0.0

    score = flow_data.get('flow_score', 0)
    if abs(score) < 10:
        return 0.0

    if direction == 'LONG':
        if score > 20:
            return min(score / 40, 2.0)
        elif score < -20:
            return -min(abs(score) / 40, 2.0)
    elif direction == 'SHORT':
        if score < -20:
            return min(abs(score) / 40, 2.0)
        elif score > 20:
            return -min(score / 40, 2.0)

    return 0.0
