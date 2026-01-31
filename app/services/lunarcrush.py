"""
LunarCrush API Service - Social Intelligence for Crypto Trading
Provides Galaxy Score, Social Sentiment, and Social Volume data
"""
import os
import logging
import httpx
from typing import Dict, List, Optional
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

LUNARCRUSH_API_BASE = "https://lunarcrush.com/api4"

_cache: Dict[str, dict] = {}
CACHE_TTL_SECONDS = 300  # 5 minute cache


def get_lunarcrush_api_key() -> Optional[str]:
    """Get LunarCrush API key from environment."""
    return os.environ.get("LUNARCRUSH_API_KEY")


def _get_cached(key: str) -> Optional[dict]:
    """Get cached data if still valid."""
    if key in _cache:
        cached = _cache[key]
        if datetime.now() - cached['timestamp'] < timedelta(seconds=CACHE_TTL_SECONDS):
            return cached['data']
        del _cache[key]
    return None


def _set_cache(key: str, data: dict):
    """Cache data with timestamp."""
    _cache[key] = {
        'data': data,
        'timestamp': datetime.now()
    }


async def get_coin_metrics(symbol: str) -> Optional[Dict]:
    """
    Get social metrics for a specific coin.
    
    Returns:
        {
            'symbol': str,
            'galaxy_score': float (0-100),
            'alt_rank': int,
            'sentiment': float (-1 to 1),
            'social_volume': int,
            'social_dominance': float,
            'market_dominance': float,
            'percent_change_24h': float,
            'interactions_24h': int
        }
    """
    api_key = get_lunarcrush_api_key()
    if not api_key:
        logger.warning("No LUNARCRUSH_API_KEY configured")
        return None
    
    # Normalize symbol (remove USDT suffix)
    coin_name = symbol.replace('USDT', '').replace('/', '').lower()
    
    cache_key = f"coin_{coin_name}"
    cached = _get_cached(cache_key)
    if cached:
        return cached
    
    try:
        async with httpx.AsyncClient() as client:
            headers = {"Authorization": f"Bearer {api_key}"}
            url = f"{LUNARCRUSH_API_BASE}/public/topic/{coin_name}/v1"
            
            response = await client.get(url, headers=headers, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                topic_data = data.get('data', {})
                
                result = {
                    'symbol': symbol,
                    'galaxy_score': topic_data.get('galaxy_score', 0),
                    'alt_rank': topic_data.get('alt_rank', 9999),
                    'sentiment': topic_data.get('sentiment', 0),
                    'social_volume': topic_data.get('social_volume', 0),
                    'social_dominance': topic_data.get('social_dominance', 0),
                    'market_dominance': topic_data.get('market_dominance', 0),
                    'percent_change_24h': topic_data.get('percent_change_24h', 0),
                    'interactions_24h': topic_data.get('interactions_24h', 0),
                    'num_contributors': topic_data.get('num_contributors', 0),
                    'posts_24h': topic_data.get('posts_24h', 0)
                }
                
                _set_cache(cache_key, result)
                logger.debug(f"LunarCrush {symbol}: Galaxy={result['galaxy_score']}, Sentiment={result['sentiment']:.2f}")
                return result
            
            elif response.status_code == 404:
                logger.debug(f"LunarCrush: {coin_name} not found")
                return None
            else:
                logger.warning(f"LunarCrush API error {response.status_code}: {response.text[:100]}")
                return None
                
    except Exception as e:
        logger.error(f"LunarCrush API error for {symbol}: {e}")
        return None


async def get_trending_coins(limit: int = 20) -> List[Dict]:
    """
    Get trending coins by social momentum.
    
    Returns list of coins sorted by Galaxy Score with high social activity.
    """
    api_key = get_lunarcrush_api_key()
    if not api_key:
        logger.warning("No LUNARCRUSH_API_KEY configured")
        return []
    
    cache_key = "trending_coins"
    cached = _get_cached(cache_key)
    if cached:
        return cached[:limit]
    
    try:
        async with httpx.AsyncClient() as client:
            headers = {"Authorization": f"Bearer {api_key}"}
            url = f"{LUNARCRUSH_API_BASE}/public/coins/list/v2"
            
            response = await client.get(url, headers=headers, timeout=15)
            
            if response.status_code == 200:
                data = response.json()
                coins = data.get('data', [])
                
                # Filter and sort by Galaxy Score
                trending = []
                for coin in coins:
                    galaxy_score = coin.get('galaxy_score', 0)
                    social_volume = coin.get('social_volume', 0)
                    
                    # Only include coins with meaningful social activity
                    if galaxy_score >= 50 and social_volume >= 100:
                        trending.append({
                            'symbol': coin.get('symbol', '').upper() + 'USDT',
                            'name': coin.get('name', ''),
                            'galaxy_score': galaxy_score,
                            'alt_rank': coin.get('alt_rank', 9999),
                            'sentiment': coin.get('sentiment', 0),
                            'social_volume': social_volume,
                            'percent_change_24h': coin.get('percent_change_24h', 0),
                            'market_cap': coin.get('market_cap', 0),
                            'interactions_24h': coin.get('interactions_24h', 0)
                        })
                
                # Sort by Galaxy Score descending
                trending.sort(key=lambda x: x['galaxy_score'], reverse=True)
                
                _set_cache(cache_key, trending)
                logger.info(f"LunarCrush: Found {len(trending)} trending coins")
                return trending[:limit]
            else:
                logger.warning(f"LunarCrush trending API error: {response.status_code}")
                return []
                
    except Exception as e:
        logger.error(f"LunarCrush trending error: {e}")
        return []


async def get_social_spikes(min_volume_change: float = 50.0, limit: int = 10) -> List[Dict]:
    """
    Find coins with unusual social volume spikes.
    
    Args:
        min_volume_change: Minimum % increase in social volume (24h)
        limit: Max results to return
    
    Returns list of coins with social volume spikes.
    """
    api_key = get_lunarcrush_api_key()
    if not api_key:
        return []
    
    cache_key = f"social_spikes_{min_volume_change}"
    cached = _get_cached(cache_key)
    if cached:
        return cached[:limit]
    
    try:
        async with httpx.AsyncClient() as client:
            headers = {"Authorization": f"Bearer {api_key}"}
            url = f"{LUNARCRUSH_API_BASE}/public/coins/list/v2"
            
            response = await client.get(url, headers=headers, timeout=15)
            
            if response.status_code == 200:
                data = response.json()
                coins = data.get('data', [])
                
                spikes = []
                for coin in coins:
                    social_volume_change = coin.get('social_volume_24h_percent_change', 0)
                    galaxy_score = coin.get('galaxy_score', 0)
                    
                    if social_volume_change >= min_volume_change and galaxy_score >= 40:
                        spikes.append({
                            'symbol': coin.get('symbol', '').upper() + 'USDT',
                            'name': coin.get('name', ''),
                            'galaxy_score': galaxy_score,
                            'sentiment': coin.get('sentiment', 0),
                            'social_volume': coin.get('social_volume', 0),
                            'social_volume_change_24h': social_volume_change,
                            'percent_change_24h': coin.get('percent_change_24h', 0)
                        })
                
                # Sort by social volume change
                spikes.sort(key=lambda x: x['social_volume_change_24h'], reverse=True)
                
                _set_cache(cache_key, spikes)
                logger.info(f"LunarCrush: Found {len(spikes)} social volume spikes")
                return spikes[:limit]
            
            return []
            
    except Exception as e:
        logger.error(f"LunarCrush spikes error: {e}")
        return []


def interpret_galaxy_score(score: float) -> str:
    """Interpret signal score into human-readable rating (legacy name)."""
    return interpret_signal_score(score)


def interpret_signal_score(score: float) -> str:
    """Interpret Signal Score (0-100) into human-readable rating."""
    if score >= 80:
        return "🚀 VERY BULLISH"
    elif score >= 70:
        return "🟢 BULLISH"
    elif score >= 60:
        return "📈 POSITIVE"
    elif score >= 50:
        return "⚖️ NEUTRAL"
    elif score >= 40:
        return "📉 NEGATIVE"
    elif score >= 30:
        return "🔴 BEARISH"
    else:
        return "⚠️ VERY BEARISH"


def interpret_sentiment(sentiment: float) -> str:
    """Interpret sentiment score (-1 to 1)."""
    if sentiment >= 0.6:
        return "Very Positive"
    elif sentiment >= 0.3:
        return "Positive"
    elif sentiment >= -0.3:
        return "Neutral"
    elif sentiment >= -0.6:
        return "Negative"
    else:
        return "Very Negative"
