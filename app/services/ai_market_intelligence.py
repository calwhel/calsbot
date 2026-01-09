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
    🔮 ENHANCED MARKET REGIME DETECTOR
    Analyzes BTC and overall market with derivatives data, funding rates,
    open interest, market breadth, and Fear & Greed to identify trading conditions.
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
        derivatives_data = {}
        breadth_data = {}
        
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
            btc_funding_url = "https://fapi.binance.com/fapi/v1/fundingRate?symbol=BTCUSDT&limit=1"
            resp = await client.get(btc_funding_url)
            if resp.status_code == 200:
                data = resp.json()
                if data:
                    derivatives_data['btc_funding'] = float(data[0].get('fundingRate', 0)) * 100
        except:
            pass
        
        try:
            eth_funding_url = "https://fapi.binance.com/fapi/v1/fundingRate?symbol=ETHUSDT&limit=1"
            resp = await client.get(eth_funding_url)
            if resp.status_code == 200:
                data = resp.json()
                if data:
                    derivatives_data['eth_funding'] = float(data[0].get('fundingRate', 0)) * 100
        except:
            pass
        
        try:
            oi_url = "https://fapi.binance.com/fapi/v1/openInterest?symbol=BTCUSDT"
            resp = await client.get(oi_url)
            if resp.status_code == 200:
                data = resp.json()
                derivatives_data['btc_oi'] = float(data.get('openInterest', 0))
        except:
            pass
        
        try:
            all_tickers_url = "https://fapi.binance.com/fapi/v1/ticker/24hr"
            resp = await client.get(all_tickers_url)
            if resp.status_code == 200:
                tickers = resp.json()
                usdt_tickers = [t for t in tickers if t.get('symbol', '').endswith('USDT')]
                
                gainers = 0
                losers = 0
                big_movers_up = []
                big_movers_down = []
                
                for t in usdt_tickers:
                    change = float(t.get('priceChangePercent', 0))
                    symbol = t.get('symbol', '').replace('USDT', '')
                    volume = float(t.get('quoteVolume', 0))
                    
                    if volume < 1000000:
                        continue
                    
                    if change > 0:
                        gainers += 1
                        if change >= 5:
                            big_movers_up.append({'symbol': symbol, 'change': change})
                    elif change < 0:
                        losers += 1
                        if change <= -5:
                            big_movers_down.append({'symbol': symbol, 'change': change})
                
                breadth_data['gainers'] = gainers
                breadth_data['losers'] = losers
                breadth_data['breadth_ratio'] = gainers / max(losers, 1)
                breadth_data['big_movers_up'] = sorted(big_movers_up, key=lambda x: x['change'], reverse=True)[:5]
                breadth_data['big_movers_down'] = sorted(big_movers_down, key=lambda x: x['change'])[:5]
                breadth_data['total_analyzed'] = gainers + losers
        except Exception as e:
            logger.warning(f"Market breadth fetch failed: {e}")
        
        try:
            klines_url = "https://fapi.binance.com/fapi/v1/klines?symbol=BTCUSDT&interval=1h&limit=24"
            resp = await client.get(klines_url)
            if resp.status_code == 200:
                klines = resp.json()
                closes = [float(k[4]) for k in klines]
                highs = [float(k[2]) for k in klines]
                lows = [float(k[3]) for k in klines]
                volumes = [float(k[5]) for k in klines]
                
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
                
                if len(volumes) >= 6:
                    recent_vol = sum(volumes[-6:]) / 6
                    older_vol = sum(volumes[-12:-6]) / 6 if len(volumes) >= 12 else recent_vol
                    btc_data['volume_trend'] = ((recent_vol - older_vol) / older_vol * 100) if older_vol > 0 else 0
                
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
        
        try:
            btc_dom_url = "https://api.coingecko.com/api/v3/global"
            resp = await client.get(btc_dom_url)
            if resp.status_code == 200:
                data = resp.json()
                total_data['btc_dominance'] = data.get('data', {}).get('market_cap_percentage', {}).get('btc', 0)
                total_data['total_market_cap'] = data.get('data', {}).get('total_market_cap', {}).get('usd', 0)
        except:
            pass
    
    if not btc_data.get('price'):
        logger.warning("No BTC data available for regime detection")
        return {'regime': 'UNKNOWN', 'confidence': 0, 'recommendation': 'Wait for data'}
    
    client = get_gemini_client()
    if not client:
        regime = analyze_regime_from_data(btc_data, eth_data, total_data, derivatives_data, breadth_data)
        _current_market_regime = regime
        return regime
    
    funding_signal = "NEUTRAL"
    btc_funding = derivatives_data.get('btc_funding', 0)
    if btc_funding > 0.03:
        funding_signal = "OVERLEVERAGED_LONG (short squeeze risk low)"
    elif btc_funding < -0.01:
        funding_signal = "OVERLEVERAGED_SHORT (squeeze risk high)"
    
    breadth_signal = "NEUTRAL"
    breadth_ratio = breadth_data.get('breadth_ratio', 1)
    if breadth_ratio > 1.5:
        breadth_signal = "BROAD_STRENGTH (alts outperforming)"
    elif breadth_ratio < 0.7:
        breadth_signal = "BROAD_WEAKNESS (risk-off)"
    
    prompt = f"""Analyze current crypto market conditions and identify the TRADING REGIME.

=== BTC CORE DATA ===
- Price: ${btc_data.get('price', 0):,.0f}
- 24h Change: {btc_data.get('change_24h', 0):+.1f}%
- RSI (1h): {btc_data.get('rsi_1h', 50):.0f}
- Price vs SMA20: {btc_data.get('price_to_sma20', 0):+.1f}%
- Hourly Volatility: {btc_data.get('volatility_pct', 0):.2f}%
- Trend Strength: {btc_data.get('trend_strength', 0)} (positive=bullish, negative=bearish)
- Volume Trend: {btc_data.get('volume_trend', 0):+.1f}% (recent vs older)

=== DERIVATIVES DATA ===
- BTC Funding Rate: {btc_funding:+.4f}% ({funding_signal})
- ETH Funding Rate: {derivatives_data.get('eth_funding', 0):+.4f}%
- BTC Open Interest: {derivatives_data.get('btc_oi', 0):,.0f} BTC

=== MARKET BREADTH ===
- Gainers: {breadth_data.get('gainers', 0)} | Losers: {breadth_data.get('losers', 0)}
- Breadth Ratio: {breadth_ratio:.2f} ({breadth_signal})
- Top Movers Up: {', '.join([f"{m['symbol']} +{m['change']:.0f}%" for m in breadth_data.get('big_movers_up', [])[:3]])}
- Top Movers Down: {', '.join([f"{m['symbol']} {m['change']:.0f}%" for m in breadth_data.get('big_movers_down', [])[:3]])}

=== SENTIMENT ===
- Fear & Greed: {total_data.get('fear_greed', 50)} ({total_data.get('fear_greed_text', 'Neutral')})
- BTC Dominance: {total_data.get('btc_dominance', 0):.1f}%
- ETH 24h: {eth_data.get('change_24h', 0):+.1f}%

Determine the current market REGIME:
- TRENDING_UP: Clear bullish momentum, RSI 55-75, making higher highs, positive funding
- TRENDING_DOWN: Clear bearish momentum, RSI 25-45, making lower lows, negative funding
- RANGING: RSI 40-60, price oscillating around SMA, low volatility, mixed breadth
- CHOPPY: High volatility, no clear direction, frequent reversals, breadth diverging
- VOLATILE_BREAKOUT: Extreme moves (>3% hourly), high volume, prepare for continuation

Provide tactical playbook for this regime and risk assessment.

Respond JSON only:
{{"regime": "TRENDING_UP/TRENDING_DOWN/RANGING/CHOPPY/VOLATILE_BREAKOUT", "confidence": 1-10, "btc_bias": "BULLISH/BEARISH/NEUTRAL", "recommendation": "one sentence trading advice", "position_size_modifier": 0.5-1.5, "preferred_direction": "LONG/SHORT/BOTH/NONE", "risk_level": "LOW/MEDIUM/HIGH/EXTREME", "tactical_playbook": "2-3 sentence specific trading strategy for this regime", "key_levels": {{"support": 0, "resistance": 0}}, "watch_for": "what could change this regime"}}"""

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
        result['btc_rsi'] = btc_data.get('rsi_1h', 50)
        result['btc_volatility'] = btc_data.get('volatility_pct', 0)
        result['fear_greed'] = total_data.get('fear_greed', 50)
        result['fear_greed_text'] = total_data.get('fear_greed_text', 'Neutral')
        result['btc_funding'] = derivatives_data.get('btc_funding', 0)
        result['eth_funding'] = derivatives_data.get('eth_funding', 0)
        result['gainers'] = breadth_data.get('gainers', 0)
        result['losers'] = breadth_data.get('losers', 0)
        result['breadth_ratio'] = breadth_data.get('breadth_ratio', 1)
        result['big_movers_up'] = breadth_data.get('big_movers_up', [])
        result['big_movers_down'] = breadth_data.get('big_movers_down', [])
        result['btc_dominance'] = total_data.get('btc_dominance', 0)
        result['timestamp'] = datetime.utcnow().isoformat()
        
        _current_market_regime = result
        
        logger.info(f"🔮 Market Regime: {result.get('regime', 'UNKNOWN')} | Risk: {result.get('risk_level', 'MEDIUM')} | Confidence: {result.get('confidence', 0)}/10")
        
        return result
        
    except Exception as e:
        logger.error(f"Market regime analysis error: {e}")
        regime = analyze_regime_from_data(btc_data, eth_data, total_data, derivatives_data, breadth_data)
        _current_market_regime = regime
        return regime


def analyze_regime_from_data(btc_data: Dict, eth_data: Dict, total_data: Dict, derivatives_data: Dict = None, breadth_data: Dict = None) -> Dict:
    """Fallback regime detection without AI."""
    derivatives_data = derivatives_data or {}
    breadth_data = breadth_data or {}
    
    change = btc_data.get('change_24h', 0)
    rsi = btc_data.get('rsi_1h', 50)
    volatility = btc_data.get('volatility_pct', 0)
    trend = btc_data.get('trend_strength', 0)
    
    if abs(change) > 5 or volatility > 1.5:
        regime = 'VOLATILE_BREAKOUT'
        recommendation = 'Use wider stops, reduce size'
        risk_level = 'HIGH'
        playbook = 'Wait for volatility to settle before entering. If already in position, tighten stops.'
    elif change > 2 and rsi > 55 and trend > 2:
        regime = 'TRENDING_UP'
        recommendation = 'Favor LONG positions'
        risk_level = 'MEDIUM'
        playbook = 'Look for pullbacks to SMA20 for long entries. Trail stops below recent swing lows.'
    elif change < -2 and rsi < 45 and trend < -2:
        regime = 'TRENDING_DOWN'
        recommendation = 'Favor SHORT positions'
        risk_level = 'MEDIUM'
        playbook = 'Short rallies into resistance. Watch for oversold bounces to add positions.'
    elif abs(change) < 1.5 and 40 <= rsi <= 60:
        regime = 'RANGING'
        recommendation = 'Mean reversion plays work well'
        risk_level = 'LOW'
        playbook = 'Fade extremes, buy support, sell resistance. Keep tight stops.'
    else:
        regime = 'CHOPPY'
        recommendation = 'Reduce position size, wait for clarity'
        risk_level = 'HIGH'
        playbook = 'Best to stay flat or use very small size. Wait for clear direction.'
    
    return {
        'regime': regime,
        'confidence': 6,
        'btc_bias': 'BULLISH' if change > 1 else 'BEARISH' if change < -1 else 'NEUTRAL',
        'recommendation': recommendation,
        'risk_level': risk_level,
        'tactical_playbook': playbook,
        'position_size_modifier': 0.7 if regime in ['CHOPPY', 'VOLATILE_BREAKOUT'] else 1.0,
        'preferred_direction': 'LONG' if regime == 'TRENDING_UP' else 'SHORT' if regime == 'TRENDING_DOWN' else 'BOTH',
        'btc_price': btc_data.get('price', 0),
        'btc_change': btc_data.get('change_24h', 0),
        'btc_rsi': btc_data.get('rsi_1h', 50),
        'btc_volatility': volatility,
        'fear_greed': total_data.get('fear_greed', 50),
        'fear_greed_text': total_data.get('fear_greed_text', 'Neutral'),
        'btc_funding': derivatives_data.get('btc_funding', 0),
        'eth_funding': derivatives_data.get('eth_funding', 0),
        'gainers': breadth_data.get('gainers', 0),
        'losers': breadth_data.get('losers', 0),
        'breadth_ratio': breadth_data.get('breadth_ratio', 1),
        'big_movers_up': breadth_data.get('big_movers_up', []),
        'big_movers_down': breadth_data.get('big_movers_down', []),
        'btc_dominance': total_data.get('btc_dominance', 0),
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
    """Format market regime for Telegram with enhanced presentation."""
    regime_type = regime.get('regime', 'UNKNOWN')
    
    regime_emojis = {
        'TRENDING_UP': '🚀',
        'TRENDING_DOWN': '📉',
        'RANGING': '↔️',
        'CHOPPY': '🌊',
        'VOLATILE_BREAKOUT': '💥'
    }
    
    risk_meters = {
        'LOW': '🟢🟢🟢⚪⚪',
        'MEDIUM': '🟡🟡🟡⚪⚪',
        'HIGH': '🟠🟠🟠🟠⚪',
        'EXTREME': '🔴🔴🔴🔴🔴'
    }
    
    emoji = regime_emojis.get(regime_type, '❓')
    risk_level = str(regime.get('risk_level', 'MEDIUM'))
    risk_meter = risk_meters.get(risk_level, '⚪⚪⚪⚪⚪')
    
    bias_emoji = "🟢" if regime.get('btc_bias') == 'BULLISH' else "🔴" if regime.get('btc_bias') == 'BEARISH' else "⚪"
    
    try:
        funding = float(regime.get('btc_funding', 0))
    except (ValueError, TypeError):
        funding = 0.0
    funding_emoji = "🔥" if funding > 0.03 else "❄️" if funding < -0.01 else "➖"
    
    try:
        fear_greed = int(float(regime.get('fear_greed', 50)))
    except (ValueError, TypeError):
        fear_greed = 50
    fg_emoji = "😨" if fear_greed < 25 else "😰" if fear_greed < 40 else "😐" if fear_greed < 60 else "😊" if fear_greed < 75 else "🤑"
    
    try:
        confidence = int(float(regime.get('confidence', 0)))
    except (ValueError, TypeError):
        confidence = 0
    
    try:
        position_mod = float(regime.get('position_size_modifier', 1.0))
    except (ValueError, TypeError):
        position_mod = 1.0
    
    try:
        btc_price = float(regime.get('btc_price', 0))
    except (ValueError, TypeError):
        btc_price = 0.0
    
    try:
        btc_change = float(regime.get('btc_change', 0))
    except (ValueError, TypeError):
        btc_change = 0.0
    
    try:
        btc_rsi = float(regime.get('btc_rsi', 50))
    except (ValueError, TypeError):
        btc_rsi = 50.0
    
    try:
        btc_volatility = float(regime.get('btc_volatility', 0))
    except (ValueError, TypeError):
        btc_volatility = 0.0
    
    try:
        eth_funding = float(regime.get('eth_funding', 0))
    except (ValueError, TypeError):
        eth_funding = 0.0
    
    try:
        btc_dominance = float(regime.get('btc_dominance', 0))
    except (ValueError, TypeError):
        btc_dominance = 0.0
    
    try:
        breadth_ratio = float(regime.get('breadth_ratio', 1))
    except (ValueError, TypeError):
        breadth_ratio = 1.0
    
    gainers = regime.get('gainers', 0)
    losers = regime.get('losers', 0)
    
    message = f"""{emoji} <b>MARKET REGIME: {regime_type}</b>
━━━━━━━━━━━━━━━━━━━━

<b>💰 BTC ANALYSIS</b>
├ Price: ${btc_price:,.0f} ({btc_change:+.1f}%)
├ RSI (1H): {btc_rsi:.0f}
├ Volatility: {btc_volatility:.2f}%
└ Bias: {bias_emoji} {regime.get('btc_bias', 'NEUTRAL')}

<b>📊 DERIVATIVES</b>
├ BTC Funding: {funding_emoji} {funding:+.4f}%
├ ETH Funding: {eth_funding:+.4f}%
└ Dom: {btc_dominance:.1f}%

<b>📈 MARKET BREADTH</b>
├ Gainers: {gainers} | Losers: {losers}
└ Ratio: {breadth_ratio:.2f}x"""

    big_up = regime.get('big_movers_up', [])
    big_down = regime.get('big_movers_down', [])
    
    if big_up:
        movers_up = ", ".join([f"{m['symbol']} +{m['change']:.0f}%" for m in big_up[:3]])
        message += f"\n🔥 <b>Hot:</b> {movers_up}"
    
    if big_down:
        movers_down = ", ".join([f"{m['symbol']} {m['change']:.0f}%" for m in big_down[:3]])
        message += f"\n❄️ <b>Cold:</b> {movers_down}"
    
    message += f"""

<b>🎭 SENTIMENT</b>
├ Fear & Greed: {fg_emoji} {fear_greed} ({regime.get('fear_greed_text', 'Neutral')})
└ Confidence: {'⭐' * min(max(confidence // 2, 0), 5)} {confidence}/10

<b>⚠️ RISK LEVEL: {risk_level}</b>
{risk_meter}

<b>🎯 TRADING GUIDANCE</b>
├ Direction: {regime.get('preferred_direction', 'BOTH')}
├ Position Size: {position_mod:.0%} of normal
└ {regime.get('recommendation', 'N/A')}"""

    playbook = regime.get('tactical_playbook', '')
    if playbook:
        message += f"""

<b>📋 TACTICAL PLAYBOOK</b>
{playbook}"""
    
    watch_for = regime.get('watch_for', '')
    if watch_for:
        message += f"""

<b>👀 WATCH FOR:</b> {watch_for}"""
    
    message += """

━━━━━━━━━━━━━━━━━━━━
<i>Updated every 15 minutes</i>"""
    
    return message
