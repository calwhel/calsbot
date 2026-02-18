import logging
from typing import Dict, Optional, List

logger = logging.getLogger(__name__)


def score_sentiment_divergence(
    social_data: Dict,
    ta_data: Dict,
    derivatives: Dict = None
) -> Dict:
    """Score contradictions between social sentiment and technical analysis.
    
    Returns a divergence assessment:
    - divergence_type: BULLISH_DIVERGENCE, BEARISH_DIVERGENCE, ALIGNED, or NEUTRAL
    - divergence_score: 0-100 (higher = stronger divergence)
    - signals: list of detected contradictions
    - recommendation: AI-readable summary
    """
    signals = []
    bullish_points = 0
    bearish_points = 0

    social_sentiment = _extract_social_sentiment(social_data)
    ta_sentiment = _extract_ta_sentiment(ta_data)

    if social_sentiment['direction'] == 'BULLISH' and ta_sentiment['direction'] == 'BEARISH':
        signals.append({
            'type': 'SOCIAL_BULL_TA_BEAR',
            'severity': 'HIGH',
            'detail': f"Social is bullish (score {social_sentiment['strength']}/100) but TA shows bearish setup (RSI: {ta_sentiment.get('rsi', 'N/A')}, MACD: {ta_sentiment.get('macd_signal', 'N/A')})",
        })
        bearish_points += 30

    elif social_sentiment['direction'] == 'BEARISH' and ta_sentiment['direction'] == 'BULLISH':
        signals.append({
            'type': 'SOCIAL_BEAR_TA_BULL',
            'severity': 'HIGH',
            'detail': f"Social is fearful/bearish (score {social_sentiment['strength']}/100) but TA shows bullish setup (RSI: {ta_sentiment.get('rsi', 'N/A')}, MACD: {ta_sentiment.get('macd_signal', 'N/A')})",
        })
        bullish_points += 30

    rsi = ta_sentiment.get('rsi')
    if rsi is not None:
        if rsi > 70 and social_sentiment['direction'] == 'BULLISH' and social_sentiment['strength'] > 70:
            signals.append({
                'type': 'OVERBOUGHT_EUPHORIA',
                'severity': 'HIGH',
                'detail': f"RSI overbought ({rsi:.0f}) + extreme social bullishness ({social_sentiment['strength']}/100) = potential reversal",
            })
            bearish_points += 25

        elif rsi < 30 and social_sentiment['direction'] == 'BEARISH' and social_sentiment['strength'] > 60:
            signals.append({
                'type': 'OVERSOLD_FEAR',
                'severity': 'HIGH',
                'detail': f"RSI oversold ({rsi:.0f}) + social fear ({social_sentiment['strength']}/100) = contrarian buy setup",
            })
            bullish_points += 25

    macd_signal = ta_sentiment.get('macd_signal', '')
    if 'BULLISH_CROSS' in macd_signal and social_sentiment['direction'] == 'BEARISH':
        signals.append({
            'type': 'MACD_BULL_CROSS_SOCIAL_FEAR',
            'severity': 'MEDIUM',
            'detail': "MACD bullish crossover while social sentiment is negative = early reversal signal",
        })
        bullish_points += 15

    elif 'BEARISH_CROSS' in macd_signal and social_sentiment['direction'] == 'BULLISH':
        signals.append({
            'type': 'MACD_BEAR_CROSS_SOCIAL_HYPE',
            'severity': 'MEDIUM',
            'detail': "MACD bearish crossover while social is hyped = distribution/top signal",
        })
        bearish_points += 15

    bb_position = ta_sentiment.get('bb_position', '')
    if bb_position == 'ABOVE_UPPER' and social_sentiment['direction'] == 'BULLISH' and social_sentiment['strength'] > 65:
        signals.append({
            'type': 'BB_UPPER_EUPHORIA',
            'severity': 'MEDIUM',
            'detail': "Price above upper Bollinger Band + social euphoria = mean reversion risk",
        })
        bearish_points += 15

    elif bb_position == 'BELOW_LOWER' and social_sentiment['direction'] == 'BEARISH':
        signals.append({
            'type': 'BB_LOWER_PANIC',
            'severity': 'MEDIUM',
            'detail': "Price below lower Bollinger Band + social panic = bounce candidate",
        })
        bullish_points += 15

    if derivatives:
        funding = derivatives.get('funding_rate')
        if funding is not None:
            if funding > 0.05 and social_sentiment['direction'] == 'BULLISH':
                signals.append({
                    'type': 'HIGH_FUNDING_EUPHORIA',
                    'severity': 'MEDIUM',
                    'detail': f"High funding rate ({funding:.4f}%) + bullish social = overcrowded long trade",
                })
                bearish_points += 10

            elif funding < -0.03 and social_sentiment['direction'] == 'BEARISH':
                signals.append({
                    'type': 'NEGATIVE_FUNDING_FEAR',
                    'severity': 'MEDIUM',
                    'detail': f"Negative funding ({funding:.4f}%) + bearish social = short squeeze setup",
                })
                bullish_points += 10

        ls_ratio = derivatives.get('long_short_ratio')
        if ls_ratio is not None:
            if ls_ratio > 2.5 and social_sentiment['direction'] == 'BULLISH':
                signals.append({
                    'type': 'EXTREME_LONG_BIAS',
                    'severity': 'LOW',
                    'detail': f"L/S ratio {ls_ratio:.2f} (extreme long bias) + bullish social = crowded trade risk",
                })
                bearish_points += 8

    divergence_score = max(bullish_points, bearish_points)
    divergence_score = min(divergence_score, 100)

    if divergence_score < 15:
        divergence_type = 'NEUTRAL'
    elif bullish_points > bearish_points:
        divergence_type = 'BULLISH_DIVERGENCE'
    elif bearish_points > bullish_points:
        divergence_type = 'BEARISH_DIVERGENCE'
    else:
        divergence_type = 'ALIGNED'

    recommendation = _build_recommendation(divergence_type, divergence_score, signals)

    return {
        'divergence_type': divergence_type,
        'divergence_score': divergence_score,
        'bullish_points': bullish_points,
        'bearish_points': bearish_points,
        'signals': signals,
        'recommendation': recommendation,
        'social_sentiment': social_sentiment,
        'ta_sentiment': ta_sentiment,
    }


def _extract_social_sentiment(social_data: Dict) -> Dict:
    """Extract normalized sentiment from LunarCrush / social data."""
    galaxy_score = social_data.get('galaxy_score', 50)
    sentiment = social_data.get('sentiment', 50)
    social_score = social_data.get('social_score', 50)
    alt_rank = social_data.get('alt_rank', 500)
    price_change_24h = social_data.get('price_change_24h', 0)
    social_volume = social_data.get('social_volume', 0)

    strength = 50
    if galaxy_score and sentiment:
        strength = (galaxy_score * 0.4 + sentiment * 0.4 + min(social_score, 100) * 0.2) if social_score else (galaxy_score * 0.5 + sentiment * 0.5)
        strength = min(100, max(0, strength))

    if strength > 60 or price_change_24h > 5:
        direction = 'BULLISH'
    elif strength < 40 or price_change_24h < -5:
        direction = 'BEARISH'
    else:
        direction = 'NEUTRAL'

    return {
        'direction': direction,
        'strength': round(strength, 1),
        'galaxy_score': galaxy_score,
        'sentiment': sentiment,
        'price_change': price_change_24h,
    }


def _extract_ta_sentiment(ta_data: Dict) -> Dict:
    """Extract directional sentiment from technical analysis data."""
    bullish_count = 0
    bearish_count = 0
    total_signals = 0

    rsi = ta_data.get('rsi_15m')
    macd = ta_data.get('macd', {})
    ema = ta_data.get('ema_cross', {})
    bb = ta_data.get('bollinger', {})
    alignment = ta_data.get('trend_alignment', '')

    macd_signal = macd.get('crossover', '') if macd else ''
    bb_position = bb.get('position', '') if bb else ''

    if rsi is not None:
        total_signals += 1
        if rsi > 60:
            bullish_count += 1
        elif rsi < 40:
            bearish_count += 1

    if macd_signal:
        total_signals += 1
        if 'BULLISH' in macd_signal:
            bullish_count += 1
        elif 'BEARISH' in macd_signal:
            bearish_count += 1

    ema_signal = ema.get('signal', '') if ema else ''
    if ema_signal:
        total_signals += 1
        if ema_signal in ('GOLDEN_CROSS', 'BULLISH'):
            bullish_count += 1
        elif ema_signal in ('DEATH_CROSS', 'BEARISH'):
            bearish_count += 1

    if alignment:
        total_signals += 1
        if 'BULLISH' in alignment:
            bullish_count += 1
        elif 'BEARISH' in alignment:
            bearish_count += 1

    if bullish_count > bearish_count:
        direction = 'BULLISH'
    elif bearish_count > bullish_count:
        direction = 'BEARISH'
    else:
        direction = 'NEUTRAL'

    return {
        'direction': direction,
        'bullish_signals': bullish_count,
        'bearish_signals': bearish_count,
        'total_signals': total_signals,
        'rsi': rsi,
        'macd_signal': macd_signal,
        'bb_position': bb_position,
    }


def _build_recommendation(divergence_type: str, score: int, signals: List[Dict]) -> str:
    """Build human-readable recommendation for AI prompt."""
    if divergence_type == 'NEUTRAL':
        return "No significant sentiment divergence detected. Social and technical indicators are roughly aligned."

    severity_map = {'HIGH': 3, 'MEDIUM': 2, 'LOW': 1}
    top_signals = sorted(signals, key=lambda x: severity_map.get(x['severity'], 0), reverse=True)[:3]

    parts = []
    if divergence_type == 'BULLISH_DIVERGENCE':
        parts.append(f"BULLISH DIVERGENCE (score: {score}/100): Social fear contradicts bullish TA setup.")
        parts.append("This is a contrarian BUY signal - crowd is fearful but technicals suggest upside.")
    elif divergence_type == 'BEARISH_DIVERGENCE':
        parts.append(f"BEARISH DIVERGENCE (score: {score}/100): Social euphoria contradicts bearish TA.")
        parts.append("This is a contrarian SELL/SHORT signal - crowd is greedy but technicals warn of reversal.")
    else:
        parts.append(f"ALIGNED (score: {score}/100): Social and TA are in agreement.")

    for sig in top_signals:
        parts.append(f"  [{sig['severity']}] {sig['detail']}")

    return "\n".join(parts)


def format_divergence_for_ai(divergence: Dict) -> str:
    """Format divergence data for inclusion in AI analysis prompts."""
    if not divergence or divergence.get('divergence_type') == 'NEUTRAL':
        return ""

    lines = [
        f"\n--- SENTIMENT DIVERGENCE ANALYSIS ---",
        divergence.get('recommendation', ''),
    ]
    return "\n".join(lines)


def format_divergence_for_message(divergence: Dict) -> str:
    """Format divergence for Telegram signal message."""
    if not divergence or divergence.get('divergence_type') == 'NEUTRAL':
        return ""

    div_type = divergence['divergence_type']
    score = divergence['divergence_score']

    if div_type == 'BULLISH_DIVERGENCE':
        icon = "🟢"
        label = "BULLISH DIVERGENCE"
    elif div_type == 'BEARISH_DIVERGENCE':
        icon = "🔴"
        label = "BEARISH DIVERGENCE"
    else:
        icon = "⚪"
        label = "ALIGNED"

    parts = [f"{icon} <b>{label}</b> ({score}/100)"]

    top_signals = divergence.get('signals', [])[:2]
    for sig in top_signals:
        sev_icon = '🔴' if sig['severity'] == 'HIGH' else ('🟡' if sig['severity'] == 'MEDIUM' else '⚪')
        parts.append(f"  {sev_icon} {sig['detail']}")

    return "\n".join(parts)


def get_signal_score_modifier(divergence: Dict, direction: str) -> float:
    """Return a score modifier (-2 to +2) to add to signal strength score based on divergence.
    
    Positive = boosts signal, Negative = penalizes signal.
    """
    if not divergence:
        return 0.0

    div_type = divergence['divergence_type']
    score = divergence.get('divergence_score', 0)

    if score < 20:
        return 0.0

    if direction == 'LONG':
        if div_type == 'BULLISH_DIVERGENCE':
            return min(score / 50, 2.0)
        elif div_type == 'BEARISH_DIVERGENCE':
            return -min(score / 50, 2.0)
    elif direction == 'SHORT':
        if div_type == 'BEARISH_DIVERGENCE':
            return min(score / 50, 2.0)
        elif div_type == 'BULLISH_DIVERGENCE':
            return -min(score / 50, 2.0)

    return 0.0
