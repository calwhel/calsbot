"""
AI Market Intelligence - News Impact Scanner & Market Regime Detector
Uses Gemini 2.5 Flash for real-time market analysis
"""
import asyncio
import logging
import httpx
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

_last_news_scan = None
_last_regime_check = None
_current_market_regime = None
_news_cache = {'alerts': [], 'market_sentiment': 'NEUTRAL', 'key_themes': []}
NEWS_SCAN_INTERVAL_MINUTES = 30
REGIME_CHECK_INTERVAL_MINUTES = 15


def get_gemini_client():
    """Get Gemini client for AI analysis."""
    try:
        from google import genai
        api_key = os.environ.get("AI_INTEGRATIONS_GEMINI_API_KEY") or os.environ.get("GEMINI_API_KEY")
        if not api_key:
            return None
        return genai.Client(api_key=api_key)
    except Exception as e:
        logger.error(f"Failed to initialize Gemini client: {e}")
        return None


async def fetch_crypto_news() -> List[Dict]:
    """Fetch latest crypto news from multiple sources."""
    news_items = []
    
    async with httpx.AsyncClient(timeout=15) as client:
        try:
            url = "https://min-api.cryptocompare.com/data/v2/news/?lang=EN&sortOrder=latest"
            resp = await client.get(url)
            if resp.status_code == 200:
                data = resp.json()
                for item in data.get('Data', [])[:15]:
                    news_items.append({
                        'title': item.get('title', ''),
                        'body': item.get('body', '')[:500],
                        'source': item.get('source', ''),
                        'published': item.get('published_on', 0),
                        'categories': item.get('categories', ''),
                        'tags': item.get('tags', '')
                    })
                logger.info(f"📰 Fetched {len(news_items)} news articles from CryptoCompare")
        except Exception as e:
            logger.warning(f"CryptoCompare news fetch failed: {e}")
        
        try:
            alt_url = "https://cryptopanic.com/api/v1/posts/?auth_token=free&public=true&kind=news"
            resp = await client.get(alt_url)
            if resp.status_code == 200:
                data = resp.json()
                for item in data.get('results', [])[:10]:
                    news_items.append({
                        'title': item.get('title', ''),
                        'body': '',
                        'source': item.get('source', {}).get('title', ''),
                        'published': 0,
                        'categories': ','.join(item.get('currencies', [])) if item.get('currencies') else '',
                        'tags': ''
                    })
        except Exception as e:
            pass
    
    return news_items


async def analyze_news_impact() -> Dict:
    """
    📰 NEWS IMPACT SCANNER
    Analyzes recent crypto news and identifies potential market-moving events.
    Returns bullish/bearish signals with affected coins.
    """
    global _last_news_scan, _news_cache
    
    now = datetime.utcnow()
    if _last_news_scan and (now - _last_news_scan).total_seconds() < NEWS_SCAN_INTERVAL_MINUTES * 60:
        logger.info("📰 News scan on cooldown, using cached results")
        return {'cached': True, **_news_cache}
    
    _last_news_scan = now
    
    news_items = await fetch_crypto_news()
    if not news_items:
        logger.warning("No news articles fetched")
        return {'alerts': [], 'error': 'No news available'}
    
    client = get_gemini_client()
    if not client:
        logger.warning("Gemini client not available for news analysis")
        return {'alerts': [], 'error': 'AI not available'}
    
    headlines = "\n".join([f"- {item['title']}" for item in news_items[:12]])
    
    prompt = f"""Analyze these crypto news headlines for TRADING SIGNALS.
Identify news that could cause significant price movements in the next 1-24 hours.

NEWS:
{headlines}

For each impactful headline, determine:
1. Affected coin(s) - use symbol format like BTC, ETH, SOL
2. Impact direction - BULLISH or BEARISH
3. Impact strength - HIGH, MEDIUM, or LOW
4. Brief reasoning (10 words max)

Respond JSON only:
{{"alerts": [
  {{"headline": "short headline", "coins": ["BTC", "ETH"], "direction": "BULLISH", "strength": "HIGH", "reason": "brief reason"}},
  ...
], "market_sentiment": "BULLISH" or "BEARISH" or "NEUTRAL", "key_themes": ["theme1", "theme2"]}}

Only include news with actual trading impact. Skip generic or low-impact news.
If no impactful news, return empty alerts array."""

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config={
                "temperature": 0.3,
                "max_output_tokens": 1024
            }
        )
        
        result_text = response.text.strip()
        
        if "```json" in result_text:
            import re
            match = re.search(r'```json\s*(.*?)\s*```', result_text, re.DOTALL)
            if match:
                result_text = match.group(1)
        elif "```" in result_text:
            import re
            match = re.search(r'```\s*(.*?)\s*```', result_text, re.DOTALL)
            if match:
                result_text = match.group(1)
        
        first_brace = result_text.find("{")
        last_brace = result_text.rfind("}")
        if first_brace >= 0 and last_brace > first_brace:
            result_text = result_text[first_brace:last_brace + 1]
        
        result = json.loads(result_text)
        _news_cache = {
            'alerts': result.get('alerts', []),
            'market_sentiment': result.get('market_sentiment', 'NEUTRAL'),
            'key_themes': result.get('key_themes', [])
        }
        
        logger.info(f"📰 News Analysis: {len(_news_cache['alerts'])} alerts | Sentiment: {_news_cache['market_sentiment']}")
        
        return result
        
    except Exception as e:
        logger.error(f"News analysis error: {e}")
        return {'alerts': [], 'error': str(e)}


async def detect_market_regime() -> Dict:
    """
    🔮 MARKET REGIME DETECTOR
    Analyzes BTC and overall market to identify current trading conditions:
    - TRENDING_UP: Strong bullish momentum, good for longs
    - TRENDING_DOWN: Strong bearish momentum, good for shorts
    - RANGING: Sideways action, good for mean reversion
    - CHOPPY: High volatility, no clear direction - reduce position size
    - VOLATILE_BREAKOUT: Big moves expected, use wider stops
    """
    global _last_regime_check, _current_market_regime
    
    now = datetime.utcnow()
    if _last_regime_check and (now - _last_regime_check).total_seconds() < REGIME_CHECK_INTERVAL_MINUTES * 60:
        if _current_market_regime:
            logger.info(f"🔮 Regime check on cooldown, using cached: {_current_market_regime.get('regime', 'UNKNOWN')}")
            return _current_market_regime
    
    _last_regime_check = now
    
    async with httpx.AsyncClient(timeout=15) as client:
        btc_data = {}
        eth_data = {}
        total_data = {}
        
        try:
            btc_url = "https://fapi.binance.com/fapi/v1/ticker/24hr?symbol=BTCUSDT"
            resp = await client.get(btc_url)
            if resp.status_code == 200:
                data = resp.json()
                btc_data = {
                    'price': float(data.get('lastPrice', 0)),
                    'change_24h': float(data.get('priceChangePercent', 0)),
                    'high_24h': float(data.get('highPrice', 0)),
                    'low_24h': float(data.get('lowPrice', 0)),
                    'volume_24h': float(data.get('quoteVolume', 0))
                }
        except Exception as e:
            logger.warning(f"BTC data fetch failed: {e}")
        
        try:
            eth_url = "https://fapi.binance.com/fapi/v1/ticker/24hr?symbol=ETHUSDT"
            resp = await client.get(eth_url)
            if resp.status_code == 200:
                data = resp.json()
                eth_data = {
                    'price': float(data.get('lastPrice', 0)),
                    'change_24h': float(data.get('priceChangePercent', 0))
                }
        except:
            pass
        
        try:
            klines_url = "https://fapi.binance.com/fapi/v1/klines?symbol=BTCUSDT&interval=1h&limit=24"
            resp = await client.get(klines_url)
            if resp.status_code == 200:
                klines = resp.json()
                closes = [float(k[4]) for k in klines]
                highs = [float(k[2]) for k in klines]
                lows = [float(k[3]) for k in klines]
                
                if len(closes) >= 14:
                    deltas = [closes[i] - closes[i-1] for i in range(1, len(closes))]
                    gains = [d if d > 0 else 0 for d in deltas]
                    losses = [-d if d < 0 else 0 for d in deltas]
                    avg_gain = sum(gains[-14:]) / 14
                    avg_loss = sum(losses[-14:]) / 14
                    if avg_loss > 0:
                        rs = avg_gain / avg_loss
                        btc_data['rsi_1h'] = 100 - (100 / (1 + rs))
                    else:
                        btc_data['rsi_1h'] = 100
                
                if len(closes) >= 20:
                    sma20 = sum(closes[-20:]) / 20
                    btc_data['sma20_1h'] = sma20
                    btc_data['price_to_sma20'] = ((closes[-1] - sma20) / sma20) * 100
                
                avg_range = sum([highs[i] - lows[i] for i in range(len(highs))]) / len(highs)
                btc_data['avg_hourly_range'] = avg_range
                btc_data['volatility_pct'] = (avg_range / closes[-1]) * 100 if closes[-1] > 0 else 0
                
                higher_highs = sum(1 for i in range(1, min(6, len(highs))) if highs[-i] > highs[-i-1])
                lower_lows = sum(1 for i in range(1, min(6, len(lows))) if lows[-i] < lows[-i-1])
                btc_data['trend_strength'] = higher_highs - lower_lows
                
        except Exception as e:
            logger.warning(f"BTC klines fetch failed: {e}")
        
        try:
            fear_greed_url = "https://api.alternative.me/fng/?limit=1"
            resp = await client.get(fear_greed_url)
            if resp.status_code == 200:
                data = resp.json()
                if data.get('data'):
                    total_data['fear_greed'] = int(data['data'][0].get('value', 50))
                    total_data['fear_greed_text'] = data['data'][0].get('value_classification', 'Neutral')
        except:
            pass
    
    if not btc_data.get('price'):
        logger.warning("No BTC data available for regime detection")
        return {'regime': 'UNKNOWN', 'confidence': 0, 'recommendation': 'Wait for data'}
    
    client = get_gemini_client()
    if not client:
        regime = analyze_regime_from_data(btc_data, eth_data, total_data)
        _current_market_regime = regime
        return regime
    
    prompt = f"""Analyze current crypto market conditions and identify the TRADING REGIME.

BTC DATA:
- Price: ${btc_data.get('price', 0):,.0f}
- 24h Change: {btc_data.get('change_24h', 0):+.1f}%
- RSI (1h): {btc_data.get('rsi_1h', 50):.0f}
- Price vs SMA20: {btc_data.get('price_to_sma20', 0):+.1f}%
- Hourly Volatility: {btc_data.get('volatility_pct', 0):.2f}%
- Trend Strength: {btc_data.get('trend_strength', 0)} (positive=bullish, negative=bearish)

ETH: {eth_data.get('change_24h', 0):+.1f}%
Fear & Greed: {total_data.get('fear_greed', 50)} ({total_data.get('fear_greed_text', 'Neutral')})

Determine the current market REGIME:
- TRENDING_UP: Clear bullish momentum, RSI 55-75, making higher highs
- TRENDING_DOWN: Clear bearish momentum, RSI 25-45, making lower lows
- RANGING: RSI 40-60, price oscillating around SMA, low volatility
- CHOPPY: High volatility, no clear direction, frequent reversals
- VOLATILE_BREAKOUT: Extreme moves (>3% hourly), prepare for continuation

Respond JSON only:
{{"regime": "TRENDING_UP/TRENDING_DOWN/RANGING/CHOPPY/VOLATILE_BREAKOUT", "confidence": 1-10, "btc_bias": "BULLISH/BEARISH/NEUTRAL", "recommendation": "one sentence trading advice", "position_size_modifier": 0.5-1.5, "preferred_direction": "LONG/SHORT/BOTH/NONE"}}"""

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config={
                "temperature": 0.3,
                "max_output_tokens": 512
            }
        )
        
        result_text = response.text.strip()
        
        if "```json" in result_text:
            import re
            match = re.search(r'```json\s*(.*?)\s*```', result_text, re.DOTALL)
            if match:
                result_text = match.group(1)
        
        first_brace = result_text.find("{")
        last_brace = result_text.rfind("}")
        if first_brace >= 0 and last_brace > first_brace:
            result_text = result_text[first_brace:last_brace + 1]
        
        result = json.loads(result_text)
        
        result['btc_price'] = btc_data.get('price', 0)
        result['btc_change'] = btc_data.get('change_24h', 0)
        result['fear_greed'] = total_data.get('fear_greed', 50)
        result['timestamp'] = datetime.utcnow().isoformat()
        
        _current_market_regime = result
        
        logger.info(f"🔮 Market Regime: {result.get('regime', 'UNKNOWN')} | BTC Bias: {result.get('btc_bias', 'NEUTRAL')} | Confidence: {result.get('confidence', 0)}/10")
        
        return result
        
    except Exception as e:
        logger.error(f"Market regime analysis error: {e}")
        regime = analyze_regime_from_data(btc_data, eth_data, total_data)
        _current_market_regime = regime
        return regime


def analyze_regime_from_data(btc_data: Dict, eth_data: Dict, total_data: Dict) -> Dict:
    """Fallback regime detection without AI."""
    change = btc_data.get('change_24h', 0)
    rsi = btc_data.get('rsi_1h', 50)
    volatility = btc_data.get('volatility_pct', 0)
    trend = btc_data.get('trend_strength', 0)
    
    if abs(change) > 5 or volatility > 1.5:
        regime = 'VOLATILE_BREAKOUT'
        recommendation = 'Use wider stops, reduce size'
    elif change > 2 and rsi > 55 and trend > 2:
        regime = 'TRENDING_UP'
        recommendation = 'Favor LONG positions'
    elif change < -2 and rsi < 45 and trend < -2:
        regime = 'TRENDING_DOWN'
        recommendation = 'Favor SHORT positions'
    elif abs(change) < 1.5 and 40 <= rsi <= 60:
        regime = 'RANGING'
        recommendation = 'Mean reversion plays work well'
    else:
        regime = 'CHOPPY'
        recommendation = 'Reduce position size, wait for clarity'
    
    return {
        'regime': regime,
        'confidence': 6,
        'btc_bias': 'BULLISH' if change > 1 else 'BEARISH' if change < -1 else 'NEUTRAL',
        'recommendation': recommendation,
        'position_size_modifier': 0.7 if regime in ['CHOPPY', 'VOLATILE_BREAKOUT'] else 1.0,
        'preferred_direction': 'LONG' if regime == 'TRENDING_UP' else 'SHORT' if regime == 'TRENDING_DOWN' else 'BOTH',
        'btc_price': btc_data.get('price', 0),
        'btc_change': btc_data.get('change_24h', 0),
        'fear_greed': total_data.get('fear_greed', 50),
        'timestamp': datetime.utcnow().isoformat()
    }


def get_current_regime() -> Optional[Dict]:
    """Get the cached current market regime."""
    return _current_market_regime


def format_news_alert_message(alert: Dict) -> str:
    """Format a news alert for Telegram."""
    direction_emoji = "🟢" if alert.get('direction') == 'BULLISH' else "🔴"
    strength_emoji = "🔥" if alert.get('strength') == 'HIGH' else "⚡" if alert.get('strength') == 'MEDIUM' else "💡"
    
    coins = ", ".join(alert.get('coins', []))
    
    return f"""{direction_emoji} <b>NEWS ALERT</b> {strength_emoji}

📰 {alert.get('headline', 'Unknown')}

💰 <b>Coins:</b> {coins}
📊 <b>Impact:</b> {alert.get('direction', 'NEUTRAL')} ({alert.get('strength', 'LOW')})
💡 <b>Reason:</b> {alert.get('reason', 'N/A')}"""


def format_regime_message(regime: Dict) -> str:
    """Format market regime for Telegram."""
    regime_type = regime.get('regime', 'UNKNOWN')
    
    regime_emojis = {
        'TRENDING_UP': '🚀',
        'TRENDING_DOWN': '📉',
        'RANGING': '↔️',
        'CHOPPY': '🌊',
        'VOLATILE_BREAKOUT': '💥'
    }
    
    emoji = regime_emojis.get(regime_type, '❓')
    
    return f"""{emoji} <b>MARKET REGIME: {regime_type}</b>

💰 <b>BTC:</b> ${regime.get('btc_price', 0):,.0f} ({regime.get('btc_change', 0):+.1f}%)
📊 <b>Bias:</b> {regime.get('btc_bias', 'NEUTRAL')}
😱 <b>Fear & Greed:</b> {regime.get('fear_greed', 50)}/100
🎯 <b>Confidence:</b> {regime.get('confidence', 0)}/10

💡 <b>Recommendation:</b> {regime.get('recommendation', 'N/A')}
📍 <b>Preferred Direction:</b> {regime.get('preferred_direction', 'BOTH')}
📏 <b>Position Size:</b> {regime.get('position_size_modifier', 1.0):.0%} of normal"""
