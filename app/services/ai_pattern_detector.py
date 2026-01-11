"""
AI Pattern & Prediction Engine
Detects chart patterns and predicts liquidation zones
"""
import asyncio
import logging
import json
import os
import httpx
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import ccxt.async_support as ccxt

logger = logging.getLogger(__name__)

_pattern_cache = {}
_liquidation_cache = {}
PATTERN_CACHE_TTL = 300
LIQUIDATION_CACHE_TTL = 300


def get_gemini_client():
    """Get Gemini client for AI analysis."""
    try:
        from google import genai
        api_key = os.environ.get("AI_INTEGRATIONS_GEMINI_API_KEY") or os.environ.get("GEMINI_API_KEY")
        base_url = os.environ.get("AI_INTEGRATIONS_GEMINI_BASE_URL")
        if not api_key:
            return None
        if base_url:
            return genai.Client(api_key=api_key, http_options={"api_endpoint": base_url})
        return genai.Client(api_key=api_key)
    except Exception as e:
        logger.error(f"Failed to initialize Gemini client: {e}")
        return None


async def fetch_candles(symbol: str, timeframe: str = '1h', limit: int = 100) -> List:
    """Fetch OHLCV candles from Binance."""
    exchange = None
    try:
        exchange = ccxt.binance({'enableRateLimit': True})
        formatted_symbol = symbol.upper()
        if not formatted_symbol.endswith('/USDT'):
            formatted_symbol = f"{formatted_symbol}/USDT"
        
        candles = await exchange.fetch_ohlcv(formatted_symbol, timeframe, limit=limit)
        return candles
    except Exception as e:
        logger.error(f"Failed to fetch candles for {symbol}: {e}")
        return []
    finally:
        if exchange:
            await exchange.close()


async def detect_chart_patterns(symbol: str) -> Dict:
    """
    🔍 AI CHART PATTERN DETECTOR
    Analyzes price action to detect classic chart patterns.
    
    Detects:
    - Head & Shoulders (bearish reversal)
    - Inverse Head & Shoulders (bullish reversal)
    - Double Top / Double Bottom
    - Triangles (ascending, descending, symmetric)
    - Wedges (rising, falling)
    - Flags and Pennants
    - Cup and Handle
    """
    global _pattern_cache
    
    cache_key = f"{symbol}_patterns"
    if cache_key in _pattern_cache:
        cached = _pattern_cache[cache_key]
        if datetime.utcnow().timestamp() - cached['timestamp'] < PATTERN_CACHE_TTL:
            return {**cached['data'], 'cached': True}
    
    base_symbol = symbol.upper().replace('/USDT', '').replace('USDT', '')
    
    candles_1h = await fetch_candles(base_symbol, '1h', 100)
    candles_4h = await fetch_candles(base_symbol, '4h', 50)
    candles_15m = await fetch_candles(base_symbol, '15m', 100)
    
    if not candles_1h or len(candles_1h) < 50:
        return {'error': 'Insufficient data', 'patterns': []}
    
    def format_candles(candles, limit=50):
        formatted = []
        for c in candles[-limit:]:
            formatted.append({
                'time': datetime.fromtimestamp(c[0]/1000).strftime('%m-%d %H:%M'),
                'O': round(c[1], 6),
                'H': round(c[2], 6),
                'L': round(c[3], 6),
                'C': round(c[4], 6),
            })
        return formatted
    
    candles_1h_data = format_candles(candles_1h, 50)
    candles_4h_data = format_candles(candles_4h, 30) if candles_4h else []
    candles_15m_data = format_candles(candles_15m, 40) if candles_15m else []
    
    current_price = candles_1h[-1][4]
    high_24h = max(c[2] for c in candles_1h[-24:])
    low_24h = min(c[3] for c in candles_1h[-24:])
    
    client = get_gemini_client()
    if not client:
        return {'error': 'AI not available', 'patterns': []}
    
    prompt = f"""You are a professional chart pattern analyst. Analyze this {base_symbol}/USDT price data and identify ANY chart patterns forming.

CURRENT: ${current_price:.6f} | 24h Range: ${low_24h:.6f} - ${high_24h:.6f}

1H CANDLES (last 50):
{json.dumps(candles_1h_data, indent=1)}

4H CANDLES (last 30):
{json.dumps(candles_4h_data, indent=1)}

15M CANDLES (last 40):
{json.dumps(candles_15m_data, indent=1)}

PATTERNS TO LOOK FOR:
1. Head & Shoulders / Inverse H&S (3 peaks, middle highest/lowest)
2. Double Top / Double Bottom (2 peaks at similar level)
3. Triangles: Ascending (flat top, rising lows), Descending (flat bottom, falling highs), Symmetric
4. Wedges: Rising (both lines up, bearish), Falling (both lines down, bullish)
5. Flags/Pennants (sharp move then tight consolidation)
6. Cup & Handle (rounded bottom, small pullback)
7. Range/Consolidation (sideways movement between levels)

ANALYSIS RULES:
- Use ACTUAL price levels from the data
- State pattern completion % (e.g., "70% complete")
- Give breakout/breakdown levels
- Specify if pattern is BULLISH or BEARISH
- Rate confidence 1-10

Respond JSON only:
{{
    "patterns": [
        {{
            "name": "Pattern Name",
            "type": "BULLISH" or "BEARISH",
            "timeframe": "15m" or "1h" or "4h",
            "completion": 75,
            "key_levels": {{
                "neckline": 0.0,
                "breakout_target": 0.0,
                "breakdown_target": 0.0,
                "stop_loss": 0.0
            }},
            "confidence": 8,
            "description": "Brief description with specific price points",
            "trade_idea": "Specific trade setup if pattern confirms"
        }}
    ],
    "overall_bias": "BULLISH" or "BEARISH" or "NEUTRAL",
    "summary": "1-2 sentence summary of what you see"
}}

If NO clear patterns, return empty patterns array with summary explaining price action."""

    try:
        from app.services.openai_limiter import get_rate_limiter
        limiter = await get_rate_limiter()
        await limiter.acquire("pattern_detector")
        
        try:
            def _sync_call():
                response = client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=prompt,
                    config={"temperature": 0.3, "max_output_tokens": 1200}
                )
                return response.text
            
            result_text = await asyncio.to_thread(_sync_call)
        finally:
            limiter.release()
        
        result_text = result_text.strip()
        if result_text.startswith('```'):
            result_text = result_text.split('```')[1]
            if result_text.startswith('json'):
                result_text = result_text[4:]
        
        result = json.loads(result_text)
        result['symbol'] = base_symbol
        result['current_price'] = current_price
        result['timestamp'] = datetime.utcnow().isoformat()
        
        _pattern_cache[cache_key] = {
            'data': result,
            'timestamp': datetime.utcnow().timestamp()
        }
        
        logger.info(f"🔍 Pattern analysis for {base_symbol}: {len(result.get('patterns', []))} patterns found")
        return result
        
    except Exception as e:
        logger.error(f"Pattern detection error: {e}")
        return {'error': str(e), 'patterns': []}


async def analyze_liquidation_zones(symbol: str) -> Dict:
    """
    💀 AI LIQUIDATION ZONE PREDICTOR
    Identifies where liquidation clusters are likely based on:
    - Open interest data
    - Common leverage levels (10x, 20x, 50x, 100x)
    - Recent price action and support/resistance
    
    Predicts cascade zones where mass liquidations could trigger.
    """
    global _liquidation_cache
    
    cache_key = f"{symbol}_liquidations"
    if cache_key in _liquidation_cache:
        cached = _liquidation_cache[cache_key]
        if datetime.utcnow().timestamp() - cached['timestamp'] < LIQUIDATION_CACHE_TTL:
            return {**cached['data'], 'cached': True}
    
    base_symbol = symbol.upper().replace('/USDT', '').replace('USDT', '')
    
    candles = await fetch_candles(base_symbol, '1h', 48)
    if not candles or len(candles) < 24:
        return {'error': 'Insufficient data', 'zones': []}
    
    current_price = candles[-1][4]
    high_24h = max(c[2] for c in candles[-24:])
    low_24h = min(c[3] for c in candles[-24:])
    high_48h = max(c[2] for c in candles)
    low_48h = min(c[3] for c in candles)
    
    oi_data = await fetch_open_interest(base_symbol)
    funding_rate = await fetch_funding_rate(base_symbol)
    
    leverage_levels = [5, 10, 20, 25, 50, 100]
    long_liq_levels = []
    short_liq_levels = []
    
    for lev in leverage_levels:
        long_liq = current_price * (1 - 1/lev)
        short_liq = current_price * (1 + 1/lev)
        if long_liq > 0:
            long_liq_levels.append({'leverage': lev, 'price': round(long_liq, 6)})
        short_liq_levels.append({'leverage': lev, 'price': round(short_liq, 6)})
    
    client = get_gemini_client()
    if not client:
        return {
            'symbol': base_symbol,
            'current_price': current_price,
            'long_liquidation_levels': long_liq_levels,
            'short_liquidation_levels': short_liq_levels,
            'error': 'AI analysis unavailable - showing basic levels'
        }
    
    prompt = f"""You are a derivatives market analyst specializing in liquidation cascades.

SYMBOL: {base_symbol}/USDT
CURRENT PRICE: ${current_price:.6f}

PRICE DATA:
- 24h High: ${high_24h:.6f}
- 24h Low: ${low_24h:.6f}
- 48h High: ${high_48h:.6f}
- 48h Low: ${low_48h:.6f}

OPEN INTEREST: {json.dumps(oi_data) if oi_data else 'Data unavailable'}
FUNDING RATE: {funding_rate if funding_rate else 'Data unavailable'}

CALCULATED LIQUIDATION LEVELS (at current price):
Long Liquidations (price drops to): {json.dumps(long_liq_levels)}
Short Liquidations (price rises to): {json.dumps(short_liq_levels)}

ANALYZE:
1. Where are the HIGHEST RISK liquidation clusters?
2. Which direction is more likely to see a cascade?
3. What price levels would trigger mass liquidations?
4. How does funding rate indicate positioning?

Respond JSON only:
{{
    "high_risk_zones": [
        {{
            "type": "LONG_LIQUIDATION" or "SHORT_LIQUIDATION",
            "price_range": {{"from": 0.0, "to": 0.0}},
            "risk_level": "CRITICAL" or "HIGH" or "MEDIUM",
            "estimated_liquidations": "$X million",
            "trigger_probability": "HIGH" or "MEDIUM" or "LOW",
            "explanation": "Why this zone is dangerous"
        }}
    ],
    "current_positioning": {{
        "bias": "LONG_HEAVY" or "SHORT_HEAVY" or "BALANCED",
        "funding_signal": "Longs paying shorts" or "Shorts paying longs" or "Neutral",
        "squeeze_risk": "HIGH" or "MEDIUM" or "LOW",
        "squeeze_direction": "LONG_SQUEEZE" or "SHORT_SQUEEZE" or "NONE"
    }},
    "cascade_prediction": {{
        "most_likely_direction": "UP" or "DOWN",
        "trigger_level": 0.0,
        "target_after_cascade": 0.0,
        "probability": "HIGH" or "MEDIUM" or "LOW",
        "reasoning": "Brief explanation"
    }},
    "trading_recommendation": "Specific actionable advice based on liquidation analysis"
}}"""

    try:
        from app.services.openai_limiter import get_rate_limiter
        limiter = await get_rate_limiter()
        await limiter.acquire("liquidation_analyzer")
        
        try:
            def _sync_call():
                response = client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=prompt,
                    config={"temperature": 0.3, "max_output_tokens": 1000}
                )
                return response.text
            
            result_text = await asyncio.to_thread(_sync_call)
        finally:
            limiter.release()
        
        result_text = result_text.strip()
        if result_text.startswith('```'):
            result_text = result_text.split('```')[1]
            if result_text.startswith('json'):
                result_text = result_text[4:]
        
        result = json.loads(result_text)
        result['symbol'] = base_symbol
        result['current_price'] = current_price
        result['long_liquidation_levels'] = long_liq_levels
        result['short_liquidation_levels'] = short_liq_levels
        result['timestamp'] = datetime.utcnow().isoformat()
        
        _liquidation_cache[cache_key] = {
            'data': result,
            'timestamp': datetime.utcnow().timestamp()
        }
        
        logger.info(f"💀 Liquidation analysis for {base_symbol} complete")
        return result
        
    except Exception as e:
        logger.error(f"Liquidation analysis error: {e}")
        return {
            'symbol': base_symbol,
            'current_price': current_price,
            'long_liquidation_levels': long_liq_levels,
            'short_liquidation_levels': short_liq_levels,
            'error': str(e)
        }


async def fetch_open_interest(symbol: str) -> Optional[Dict]:
    """Fetch open interest data from Binance Futures."""
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            url = f"https://fapi.binance.com/fapi/v1/openInterest?symbol={symbol}USDT"
            resp = await client.get(url)
            if resp.status_code == 200:
                data = resp.json()
                return {
                    'open_interest': float(data.get('openInterest', 0)),
                    'symbol': data.get('symbol', '')
                }
            
            url2 = f"https://fapi.binance.com/futures/data/openInterestHist?symbol={symbol}USDT&period=1h&limit=24"
            resp2 = await client.get(url2)
            if resp2.status_code == 200:
                hist_data = resp2.json()
                if hist_data:
                    latest = hist_data[-1]
                    oldest = hist_data[0]
                    return {
                        'open_interest': float(latest.get('sumOpenInterest', 0)),
                        'open_interest_value': float(latest.get('sumOpenInterestValue', 0)),
                        'change_24h': ((float(latest.get('sumOpenInterest', 1)) / float(oldest.get('sumOpenInterest', 1))) - 1) * 100
                    }
    except Exception as e:
        logger.warning(f"Failed to fetch OI for {symbol}: {e}")
    return None


async def fetch_funding_rate(symbol: str) -> Optional[str]:
    """Fetch current funding rate from Binance Futures."""
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            url = f"https://fapi.binance.com/fapi/v1/fundingRate?symbol={symbol}USDT&limit=1"
            resp = await client.get(url)
            if resp.status_code == 200:
                data = resp.json()
                if data:
                    rate = float(data[0].get('fundingRate', 0)) * 100
                    if rate > 0.05:
                        return f"{rate:.4f}% (Longs paying shorts - LONG HEAVY)"
                    elif rate < -0.05:
                        return f"{rate:.4f}% (Shorts paying longs - SHORT HEAVY)"
                    else:
                        return f"{rate:.4f}% (Neutral)"
    except Exception as e:
        logger.warning(f"Failed to fetch funding rate for {symbol}: {e}")
    return None


def format_patterns_message(result: Dict) -> str:
    """Format pattern detection result for Telegram."""
    if result.get('error'):
        return f"❌ {result['error']}"
    
    if result.get('cached'):
        cache_note = "📋 <i>Cached (5 min cooldown)</i>\n\n"
    else:
        cache_note = ""
    
    symbol = result.get('symbol', 'UNKNOWN')
    price = result.get('current_price', 0)
    patterns = result.get('patterns', [])
    bias = result.get('overall_bias', 'NEUTRAL')
    summary = result.get('summary', '')
    
    bias_emoji = {'BULLISH': '🟢', 'BEARISH': '🔴', 'NEUTRAL': '⚪'}.get(bias, '⚪')
    
    msg = f"""{cache_note}🔍 <b>CHART PATTERNS: {symbol}/USDT</b>

💰 Current: <code>${price:,.6f}</code>
{bias_emoji} Bias: <b>{bias}</b>

"""
    
    if patterns:
        msg += "<b>📊 PATTERNS DETECTED:</b>\n\n"
        for i, p in enumerate(patterns[:5], 1):
            p_type = p.get('type', 'NEUTRAL')
            p_emoji = '🟢' if p_type == 'BULLISH' else '🔴' if p_type == 'BEARISH' else '⚪'
            confidence = p.get('confidence', 5)
            conf_bars = '█' * (confidence // 2) + '░' * (5 - confidence // 2)
            
            msg += f"""{p_emoji} <b>{p.get('name', 'Pattern')}</b>
├ Timeframe: {p.get('timeframe', '1h')}
├ Completion: {p.get('completion', 0)}%
├ Confidence: [{conf_bars}] {confidence}/10
"""
            
            levels = p.get('key_levels', {})
            if levels.get('breakout_target'):
                msg += f"├ Breakout Target: <code>${levels['breakout_target']:,.6f}</code>\n"
            if levels.get('breakdown_target'):
                msg += f"├ Breakdown Target: <code>${levels['breakdown_target']:,.6f}</code>\n"
            if levels.get('stop_loss'):
                msg += f"├ Stop Loss: <code>${levels['stop_loss']:,.6f}</code>\n"
            
            if p.get('trade_idea'):
                msg += f"└ 💡 {p['trade_idea']}\n"
            
            msg += "\n"
    else:
        msg += "📊 <i>No clear patterns detected at this time.</i>\n\n"
    
    if summary:
        msg += f"<b>📝 Summary:</b> {summary}"
    
    return msg


def format_liquidation_message(result: Dict) -> str:
    """Format liquidation analysis for Telegram."""
    if result.get('error') and not result.get('long_liquidation_levels'):
        return f"❌ {result['error']}"
    
    if result.get('cached'):
        cache_note = "📋 <i>Cached (5 min cooldown)</i>\n\n"
    else:
        cache_note = ""
    
    symbol = result.get('symbol', 'UNKNOWN')
    price = result.get('current_price', 0)
    
    msg = f"""{cache_note}💀 <b>LIQUIDATION ZONES: {symbol}/USDT</b>

💰 Current: <code>${price:,.6f}</code>

"""
    
    zones = result.get('high_risk_zones', [])
    if zones:
        msg += "<b>⚠️ HIGH RISK ZONES:</b>\n\n"
        for zone in zones[:4]:
            risk = zone.get('risk_level', 'MEDIUM')
            risk_emoji = {'CRITICAL': '🔴', 'HIGH': '🟠', 'MEDIUM': '🟡'}.get(risk, '⚪')
            z_type = zone.get('type', '')
            type_emoji = '📉' if 'LONG' in z_type else '📈'
            
            price_range = zone.get('price_range', {})
            from_price = price_range.get('from', 0)
            to_price = price_range.get('to', 0)
            
            msg += f"""{risk_emoji} {type_emoji} <b>{z_type.replace('_', ' ')}</b>
├ Range: <code>${from_price:,.6f}</code> - <code>${to_price:,.6f}</code>
├ Risk: {risk} | Trigger: {zone.get('trigger_probability', 'MEDIUM')}
├ Est. Liqs: {zone.get('estimated_liquidations', 'Unknown')}
└ {zone.get('explanation', '')}

"""
    
    positioning = result.get('current_positioning', {})
    if positioning:
        bias = positioning.get('bias', 'BALANCED')
        bias_emoji = {'LONG_HEAVY': '🟢', 'SHORT_HEAVY': '🔴', 'BALANCED': '⚪'}.get(bias, '⚪')
        squeeze = positioning.get('squeeze_risk', 'LOW')
        squeeze_emoji = {'HIGH': '🔴', 'MEDIUM': '🟠', 'LOW': '🟢'}.get(squeeze, '⚪')
        
        msg += f"""<b>📊 MARKET POSITIONING:</b>
├ Bias: {bias_emoji} {bias.replace('_', ' ')}
├ Funding: {positioning.get('funding_signal', 'Unknown')}
├ Squeeze Risk: {squeeze_emoji} {squeeze}
└ Direction: {positioning.get('squeeze_direction', 'NONE').replace('_', ' ')}

"""
    
    cascade = result.get('cascade_prediction', {})
    if cascade:
        direction = cascade.get('most_likely_direction', 'UNKNOWN')
        dir_emoji = '📈' if direction == 'UP' else '📉'
        
        msg += f"""<b>🎯 CASCADE PREDICTION:</b>
├ Direction: {dir_emoji} {direction}
├ Trigger: <code>${cascade.get('trigger_level', 0):,.6f}</code>
├ Target: <code>${cascade.get('target_after_cascade', 0):,.6f}</code>
├ Probability: {cascade.get('probability', 'MEDIUM')}
└ {cascade.get('reasoning', '')}

"""
    
    rec = result.get('trading_recommendation', '')
    if rec:
        msg += f"<b>💡 RECOMMENDATION:</b>\n{rec}"
    
    long_levels = result.get('long_liquidation_levels', [])
    short_levels = result.get('short_liquidation_levels', [])
    
    if long_levels or short_levels:
        msg += "\n\n<b>📐 LIQUIDATION LEVELS (at current price):</b>\n"
        msg += "<i>Long liqs (price drops to):</i>\n"
        for l in long_levels[:3]:
            pct_drop = ((price - l['price']) / price) * 100
            msg += f"  • {l['leverage']}x: <code>${l['price']:,.6f}</code> (-{pct_drop:.1f}%)\n"
        msg += "<i>Short liqs (price rises to):</i>\n"
        for s in short_levels[:3]:
            pct_rise = ((s['price'] - price) / price) * 100
            msg += f"  • {s['leverage']}x: <code>${s['price']:,.6f}</code> (+{pct_rise:.1f}%)\n"
    
    return msg
