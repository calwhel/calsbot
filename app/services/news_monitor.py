import httpx
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from app.config import settings

logger = logging.getLogger(__name__)

class NewsMonitor:
    def __init__(self):
        self.api_key = settings.CRYPTOPANIC_API_KEY
        self.base_url = "https://cryptopanic.com/api/developer/v2"
        self.seen_news_ids = set()
        self.news_cache = []
        self.last_fetch_time = None
        self.cache_duration_minutes = 10  # Cache for 10 minutes
        self.rate_limit_cooldown = None  # Track rate limit cooldown
        
    async def fetch_recent_news(
        self, 
        currencies: Optional[List[str]] = None,
        filter_type: str = "important",
        kind: str = "news"
    ) -> List[Dict]:
        """
        Fetch recent crypto news from CryptoPanic API
        
        Args:
            currencies: List of coin symbols (e.g., ['BTC', 'ETH'])
            filter_type: 'rising', 'hot', 'bullish', 'bearish', 'important', 'saved', 'lol'
            kind: 'news', 'media', 'all'
        """
        try:
            params = {
                'auth_token': self.api_key,
                'public': 'true',
                'kind': kind,
                'filter': filter_type
            }
            
            if currencies:
                params['currencies'] = ','.join(currencies)
            
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(f"{self.base_url}/posts/", params=params)
                response.raise_for_status()
                data = response.json()
                
                # Filter out already seen news
                new_articles = []
                for article in data.get('results', []):
                    article_id = article.get('id')
                    if article_id and article_id not in self.seen_news_ids:
                        new_articles.append(article)
                        self.seen_news_ids.add(article_id)
                
                logger.info(f"Fetched {len(new_articles)} new articles from CryptoPanic")
                return new_articles
                
        except Exception as e:
            logger.error(f"Error fetching news from CryptoPanic: {e}")
            return []
    
    def clean_old_seen_ids(self, hours: int = 24):
        """Clean up seen IDs older than specified hours to prevent memory bloat"""
        # For production, you'd want to track timestamps
        # For now, clear if set gets too large
        if len(self.seen_news_ids) > 1000:
            self.seen_news_ids.clear()
            logger.info("Cleared seen news IDs cache")
    
    async def get_important_news(self, symbols: List[str]) -> List[Dict]:
        """
        Get important/breaking news for specific symbols with caching and rate limit handling
        Returns only high-impact news items
        """
        # Check if we're in rate limit cooldown
        if self.rate_limit_cooldown:
            if datetime.utcnow() < self.rate_limit_cooldown:
                logger.warning(f"In rate limit cooldown until {self.rate_limit_cooldown}")
                return self.news_cache
            else:
                self.rate_limit_cooldown = None
        
        # Check cache freshness
        if self.last_fetch_time and self.news_cache:
            time_since_fetch = (datetime.utcnow() - self.last_fetch_time).total_seconds() / 60
            if time_since_fetch < self.cache_duration_minutes:
                logger.info(f"Using cached news (fetched {time_since_fetch:.1f} min ago)")
                return self.news_cache
        
        # Extract coin symbols from trading pairs (e.g., BTC/USDT -> BTC)
        currencies = [symbol.split('/')[0] for symbol in symbols]
        
        try:
            # Only fetch important news to reduce API calls from 3 to 1
            important = await self.fetch_recent_news(
                currencies=currencies,
                filter_type='important',
                kind='news'
            )
            
            # Update cache
            self.news_cache = important
            self.last_fetch_time = datetime.utcnow()
            
            return important
            
        except Exception as e:
            error_msg = str(e)
            if '429' in error_msg or 'Too Many Requests' in error_msg:
                # Set cooldown for 15 minutes on rate limit
                self.rate_limit_cooldown = datetime.utcnow() + timedelta(minutes=15)
                logger.warning(f"Rate limited! Cooldown until {self.rate_limit_cooldown}")
            
            # Return cached news if available
            return self.news_cache if self.news_cache else []
    
    def extract_coins_from_news(self, article: Dict) -> List[str]:
        """Extract mentioned coins from news article"""
        currencies = []
        
        # CryptoPanic provides currencies in the article
        if 'currencies' in article:
            for currency in article['currencies']:
                if 'code' in currency:
                    currencies.append(currency['code'])
        
        return currencies
    
    def get_news_metadata(self, article: Dict) -> Dict:
        """Extract key metadata from news article"""
        return {
            'id': article.get('id'),
            'title': article.get('title', ''),
            'url': article.get('url', ''),
            'source': article.get('source', {}).get('title', 'Unknown'),
            'published_at': article.get('published_at'),
            'currencies': self.extract_coins_from_news(article),
            'votes': {
                'positive': article.get('votes', {}).get('positive', 0),
                'negative': article.get('votes', {}).get('negative', 0),
                'important': article.get('votes', {}).get('important', 0),
                'liked': article.get('votes', {}).get('liked', 0),
                'disliked': article.get('votes', {}).get('disliked', 0),
                'lol': article.get('votes', {}).get('lol', 0),
                'toxic': article.get('votes', {}).get('toxic', 0),
                'saved': article.get('votes', {}).get('saved', 0)
            },
            'domain': article.get('domain', ''),
            'kind': article.get('kind', 'news')
        }
