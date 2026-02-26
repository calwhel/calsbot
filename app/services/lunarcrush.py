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
CACHE_TTL_SECONDS = 90  # 90 second cache — keep social data fresh for fast signals


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
            all_coins = []
            
            for page in range(10):
                url = f"{LUNARCRUSH_API_BASE}/public/coins/list/v2"
                params = {
                    "sort": "galaxy_score",
                    "desc": "true",
                    "limit": 200,
                    "page": page
                }
                
                response = await client.get(url, headers=headers, params=params, timeout=15)
                
                if response.status_code != 200:
                    logger.warning(f"LunarCrush page {page} error: {response.status_code}")
                    break
                    
                data = response.json()
                page_coins = data.get('data', [])
                if not page_coins:
                    break
                all_coins.extend(page_coins)
                
                has_scores = sum(1 for c in page_coins if (c.get('galaxy_score') or 0) > 0)
                logger.info(f"LunarCrush page {page}: {len(page_coins)} coins, {has_scores} with galaxy_score > 0")
                
                if has_scores == 0:
                    break
            
            logger.info(f"LunarCrush total fetched: {len(all_coins)} coins")
            
            with_scores = [c for c in all_coins if (c.get('galaxy_score') or 0) > 0]
            if with_scores:
                top3 = sorted(with_scores, key=lambda x: x.get('galaxy_score', 0), reverse=True)[:3]
                for c in top3:
                    logger.info(f"Top coin: {c.get('symbol')} gs={c.get('galaxy_score')} sv={c.get('social_volume_24h')} int={c.get('interactions_24h')} sent={c.get('sentiment')}")
            
            trending = []
            for coin in all_coins:
                galaxy_score = coin.get('galaxy_score', 0) or 0
                social_vol = coin.get('social_volume_24h', 0) or 0
                interactions = coin.get('interactions_24h', 0) or 0
                
                if galaxy_score >= 6 and (social_vol >= 3 or interactions >= 20):
                    raw_sentiment = coin.get('sentiment', 0) or 0
                    normalized_sentiment = raw_sentiment / 100.0 if raw_sentiment > 1 else raw_sentiment
                    
                    raw_symbol = coin.get('symbol', '').upper()
                    symbol = raw_symbol + 'USDT' if not raw_symbol.endswith('USDT') else raw_symbol
                    
                    trending.append({
                        'symbol': symbol,
                        'name': coin.get('name', ''),
                        'galaxy_score': galaxy_score,
                        'alt_rank': coin.get('alt_rank', 9999) or 9999,
                        'sentiment': normalized_sentiment,
                        'social_volume': social_vol,
                        'social_interactions': interactions,
                        'social_dominance': coin.get('social_dominance', 0) or 0,
                        'percent_change_24h': coin.get('percent_change_24h', 0) or 0,
                        'market_cap': coin.get('market_cap', 0) or 0,
                        'interactions_24h': interactions
                    })
            
            trending.sort(key=lambda x: x['galaxy_score'], reverse=True)
            
            _set_cache(cache_key, trending)
            logger.info(f"LunarCrush: Found {len(trending)} trending coins (from {len(all_coins)} total)")
            return trending[:limit]
                
    except Exception as e:
        logger.error(f"LunarCrush trending error: {e}")
        return []


async def get_social_spikes(min_volume_change: float = 50.0, limit: int = 10) -> List[Dict]:
    """
    Find coins with unusual social volume spikes.
    These are coins where social buzz is surging -- often leads price moves.
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
            params = {"sort": "galaxy_score", "desc": "true", "limit": 200}
            
            response = await client.get(url, headers=headers, params=params, timeout=15)
            
            if response.status_code == 200:
                data = response.json()
                coins = data.get('data', [])
                
                spikes = []
                for coin in coins:
                    social_vol_change = coin.get('social_volume_24h_percent_change', 0) or 0
                    galaxy_score = coin.get('galaxy_score', 0) or 0
                    interactions = coin.get('interactions_24h', 0) or 0
                    social_vol = coin.get('social_volume_24h', 0) or 0
                    
                    if social_vol_change >= min_volume_change and galaxy_score >= 6:
                        raw_sentiment = coin.get('sentiment', 0) or 0
                        normalized_sentiment = raw_sentiment / 100.0 if raw_sentiment > 1 else raw_sentiment
                        
                        raw_symbol = coin.get('symbol', '').upper()
                        symbol = raw_symbol + 'USDT' if not raw_symbol.endswith('USDT') else raw_symbol
                        
                        spikes.append({
                            'symbol': symbol,
                            'name': coin.get('name', ''),
                            'galaxy_score': galaxy_score,
                            'alt_rank': coin.get('alt_rank', 9999) or 9999,
                            'sentiment': normalized_sentiment,
                            'social_volume': social_vol,
                            'social_volume_change_24h': social_vol_change,
                            'social_interactions': interactions,
                            'social_dominance': coin.get('social_dominance', 0) or 0,
                            'percent_change_24h': coin.get('percent_change_24h', 0) or 0,
                            'market_cap': coin.get('market_cap', 0) or 0,
                            'interactions_24h': interactions,
                            'is_social_spike': True,
                        })
                
                spikes.sort(key=lambda x: x['social_volume_change_24h'], reverse=True)
                
                _set_cache(cache_key, spikes)
                logger.info(f"LunarCrush: Found {len(spikes)} social volume spikes (>{min_volume_change}% change)")
                return spikes[:limit]
            
            return []
            
    except Exception as e:
        logger.error(f"LunarCrush spikes error: {e}")
        return []


async def get_coin_creators(symbol: str, limit: int = 5) -> List[Dict]:
    """
    Get top influencers/creators talking about a specific coin.
    Returns list of top creators with their engagement metrics.
    """
    api_key = get_lunarcrush_api_key()
    if not api_key:
        return []
    
    coin_name = symbol.replace('USDT', '').replace('/', '').lower()
    cache_key = f"creators_{coin_name}"
    cached = _get_cached(cache_key)
    if cached:
        return cached[:limit]
    
    try:
        async with httpx.AsyncClient() as client:
            headers = {"Authorization": f"Bearer {api_key}"}
            url = f"{LUNARCRUSH_API_BASE}/public/topic/{coin_name}/creators/v1"
            
            response = await client.get(url, headers=headers, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                raw_creators = data.get('data', [])
                
                creators = []
                for c in raw_creators[:limit]:
                    creators.append({
                        'name': c.get('display_name') or c.get('name', 'Unknown'),
                        'handle': c.get('identifier', ''),
                        'network': c.get('network', 'twitter'),
                        'followers': c.get('followers_count', 0) or 0,
                        'interactions': c.get('interactions_24h', 0) or 0,
                        'posts': c.get('posts_24h', 0) or 0,
                        'sentiment': c.get('sentiment', 50) or 50,
                    })
                
                _set_cache(cache_key, creators)
                logger.info(f"LunarCrush creators for {symbol}: {len(creators)} found")
                return creators
            
            logger.debug(f"LunarCrush creators {response.status_code} for {coin_name}")
            return []
            
    except Exception as e:
        logger.error(f"LunarCrush creators error for {symbol}: {e}")
        return []


async def get_coin_top_posts(symbol: str, limit: int = 5) -> List[Dict]:
    """
    Get top social posts about a specific coin.
    Returns viral/high-engagement posts mentioning the coin.
    """
    api_key = get_lunarcrush_api_key()
    if not api_key:
        return []
    
    coin_name = symbol.replace('USDT', '').replace('/', '').lower()
    cache_key = f"posts_{coin_name}"
    cached = _get_cached(cache_key)
    if cached:
        return cached[:limit]
    
    try:
        async with httpx.AsyncClient() as client:
            headers = {"Authorization": f"Bearer {api_key}"}
            url = f"{LUNARCRUSH_API_BASE}/public/topic/{coin_name}/posts/v1"
            
            response = await client.get(url, headers=headers, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                raw_posts = data.get('data', [])
                
                posts = []
                for p in raw_posts[:limit]:
                    body = p.get('body') or p.get('title') or ''
                    if len(body) > 200:
                        body = body[:197] + '...'
                    
                    posts.append({
                        'body': body,
                        'network': p.get('post_type', 'tweet'),
                        'creator_name': p.get('creator_display_name') or p.get('creator_name', ''),
                        'creator_handle': p.get('creator_identifier', ''),
                        'followers': p.get('creator_followers_count', 0) or 0,
                        'interactions': p.get('interactions_total', 0) or 0,
                        'sentiment': p.get('sentiment', 50) or 50,
                        'created_at': p.get('post_created', ''),
                    })
                
                _set_cache(cache_key, posts)
                logger.info(f"LunarCrush posts for {symbol}: {len(posts)} found")
                return posts
            
            logger.debug(f"LunarCrush posts {response.status_code} for {coin_name}")
            return []
            
    except Exception as e:
        logger.error(f"LunarCrush posts error for {symbol}: {e}")
        return []


async def get_coin_news(symbol: str, limit: int = 5) -> List[Dict]:
    """
    Get top news posts about a specific coin from LunarCrush.
    Different aggregation from CryptoNews - cross-reference both for stronger signals.
    """
    api_key = get_lunarcrush_api_key()
    if not api_key:
        return []
    
    coin_name = symbol.replace('USDT', '').replace('/', '').lower()
    cache_key = f"news_{coin_name}"
    cached = _get_cached(cache_key)
    if cached:
        return cached[:limit]
    
    try:
        async with httpx.AsyncClient() as client:
            headers = {"Authorization": f"Bearer {api_key}"}
            url = f"{LUNARCRUSH_API_BASE}/public/topic/{coin_name}/news/v1"
            
            response = await client.get(url, headers=headers, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                raw_news = data.get('data', [])
                
                news_items = []
                for n in raw_news[:limit]:
                    title = n.get('post_title') or n.get('title') or n.get('body', '')
                    if len(title) > 150:
                        title = title[:147] + '...'
                    
                    news_items.append({
                        'title': title,
                        'source': n.get('creator_display_name') or n.get('post_type', 'news'),
                        'sentiment': n.get('sentiment', 50) or 50,
                        'interactions': n.get('interactions_total', 0) or 0,
                        'created_at': n.get('post_created', ''),
                    })
                
                _set_cache(cache_key, news_items)
                logger.info(f"LunarCrush news for {symbol}: {len(news_items)} found")
                return news_items
            
            logger.debug(f"LunarCrush news {response.status_code} for {coin_name}")
            return []
            
    except Exception as e:
        logger.error(f"LunarCrush news error for {symbol}: {e}")
        return []


async def get_social_time_series(symbol: str, interval: str = '1h', points: int = 24) -> Optional[Dict]:
    """
    Get social time series data for a coin to detect buzz momentum.
    
    Returns trend analysis:
    {
        'trend': 'RISING' | 'FALLING' | 'STABLE',
        'momentum_score': float (-100 to 100),
        'buzz_change_pct': float,
        'sentiment_trend': 'IMPROVING' | 'DECLINING' | 'STABLE',
        'data_points': int,
        'recent_avg_interactions': float,
        'prior_avg_interactions': float
    }
    """
    api_key = get_lunarcrush_api_key()
    if not api_key:
        return None
    
    coin_name = symbol.replace('USDT', '').replace('/', '').lower()
    cache_key = f"timeseries_{coin_name}_{interval}"
    cached = _get_cached(cache_key)
    if cached:
        return cached
    
    try:
        async with httpx.AsyncClient() as client:
            headers = {"Authorization": f"Bearer {api_key}"}
            url = f"{LUNARCRUSH_API_BASE}/public/topic/{coin_name}/time-series/v1"
            
            import time
            end_time = int(time.time())
            if interval == '1h':
                start_time = end_time - (points * 3600)
            else:
                start_time = end_time - (points * 86400)
            
            params = {
                "start": start_time,
                "end": end_time,
            }
            if interval == '1d':
                params["interval"] = "1d"
            
            response = await client.get(url, headers=headers, params=params, timeout=15)
            
            if response.status_code == 200:
                data = response.json()
                time_data = data.get('data', [])
                
                if len(time_data) < 4:
                    logger.debug(f"LunarCrush time series: insufficient data for {symbol} ({len(time_data)} points)")
                    return None
                
                midpoint = len(time_data) // 2
                recent_half = time_data[midpoint:]
                prior_half = time_data[:midpoint]
                
                def avg_metric(data_slice, key):
                    vals = [d.get(key, 0) or 0 for d in data_slice]
                    return sum(vals) / len(vals) if vals else 0
                
                recent_interactions = avg_metric(recent_half, 'interactions')
                prior_interactions = avg_metric(prior_half, 'interactions')
                recent_posts = avg_metric(recent_half, 'posts_created')
                prior_posts = avg_metric(prior_half, 'posts_created')
                recent_sentiment = avg_metric(recent_half, 'sentiment')
                prior_sentiment = avg_metric(prior_half, 'sentiment')
                
                if prior_interactions > 0:
                    interaction_change = ((recent_interactions - prior_interactions) / prior_interactions) * 100
                else:
                    interaction_change = 100 if recent_interactions > 0 else 0
                
                if prior_posts > 0:
                    posts_change = ((recent_posts - prior_posts) / prior_posts) * 100
                else:
                    posts_change = 100 if recent_posts > 0 else 0
                
                buzz_change = (interaction_change * 0.6) + (posts_change * 0.4)
                
                momentum_score = max(-100, min(100, buzz_change))
                
                if momentum_score > 15:
                    trend = 'RISING'
                elif momentum_score < -15:
                    trend = 'FALLING'
                else:
                    trend = 'STABLE'
                
                sentiment_diff = recent_sentiment - prior_sentiment
                if sentiment_diff > 5:
                    sentiment_trend = 'IMPROVING'
                elif sentiment_diff < -5:
                    sentiment_trend = 'DECLINING'
                else:
                    sentiment_trend = 'STABLE'
                
                result = {
                    'trend': trend,
                    'momentum_score': round(momentum_score, 1),
                    'buzz_change_pct': round(buzz_change, 1),
                    'sentiment_trend': sentiment_trend,
                    'data_points': len(time_data),
                    'recent_avg_interactions': round(recent_interactions, 0),
                    'prior_avg_interactions': round(prior_interactions, 0),
                    'recent_avg_posts': round(recent_posts, 0),
                    'prior_avg_posts': round(prior_posts, 0),
                    'sentiment_change': round(sentiment_diff, 1),
                }
                
                _set_cache(cache_key, result)
                logger.info(f"LunarCrush time series {symbol}: trend={trend}, momentum={momentum_score:.1f}, buzz_change={buzz_change:.1f}%")
                return result
            
            logger.debug(f"LunarCrush time series {response.status_code} for {coin_name}")
            return None
            
    except Exception as e:
        logger.error(f"LunarCrush time series error for {symbol}: {e}")
        return None


async def get_influencer_consensus(symbol: str) -> Optional[Dict]:
    """
    Analyze top influencers talking about a coin and determine consensus.
    Returns a summary of influencer sentiment and activity.
    """
    creators = await get_coin_creators(symbol, limit=10)
    if not creators:
        return None
    
    total_followers = sum(c['followers'] for c in creators)
    total_interactions = sum(c['interactions'] for c in creators)
    avg_sentiment = sum(c['sentiment'] for c in creators) / len(creators) if creators else 50
    
    bullish = sum(1 for c in creators if c['sentiment'] > 60)
    bearish = sum(1 for c in creators if c['sentiment'] < 40)
    neutral = len(creators) - bullish - bearish
    
    if bullish > bearish * 2:
        consensus = 'BULLISH'
    elif bearish > bullish * 2:
        consensus = 'BEARISH'
    elif bullish > bearish:
        consensus = 'LEAN BULLISH'
    elif bearish > bullish:
        consensus = 'LEAN BEARISH'
    else:
        consensus = 'MIXED'
    
    big_accounts = [c for c in creators if c['followers'] >= 50000]
    big_account_sentiment = sum(c['sentiment'] for c in big_accounts) / len(big_accounts) if big_accounts else 50
    
    return {
        'consensus': consensus,
        'num_creators': len(creators),
        'bullish_count': bullish,
        'bearish_count': bearish,
        'neutral_count': neutral,
        'avg_sentiment': round(avg_sentiment, 1),
        'total_followers': total_followers,
        'total_interactions': total_interactions,
        'big_accounts': len(big_accounts),
        'big_account_sentiment': round(big_account_sentiment, 1),
        'top_creators': creators[:3],
    }


def interpret_galaxy_score(score: float) -> str:
    """Interpret signal score into human-readable rating (legacy name)."""
    return interpret_signal_score(score)


def interpret_signal_score(score: float) -> str:
    """Interpret Galaxy Score (LunarCrush v4: 0-16 scale)."""
    if score >= 15:
        return "🚀 VERY BULLISH"
    elif score >= 14:
        return "🟢 BULLISH"
    elif score >= 13:
        return "📈 POSITIVE"
    elif score >= 12:
        return "⚖️ NEUTRAL"
    elif score >= 10:
        return "📉 WEAK"
    else:
        return "⚠️ LOW SCORE"


def interpret_sentiment(sentiment: float) -> str:
    """Interpret sentiment score (0-1 normalized from LunarCrush 0-100)."""
    if sentiment >= 0.8:
        return "Very Positive"
    elif sentiment >= 0.6:
        return "Positive"
    elif sentiment >= 0.4:
        return "Neutral"
    elif sentiment >= 0.2:
        return "Negative"
    else:
        return "Very Negative"
