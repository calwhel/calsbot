import httpx
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from app.config import settings

logger = logging.getLogger(__name__)

class NewsMonitor:
    def __init__(self):
        self.api_key = settings.CRYPTONEWS_API_KEY
        self.base_url = "https://cryptonews-api.com/api/v1"
        self.seen_news_ids = set()
        self.news_cache = []
        self.last_fetch_time = None
        self.cache_duration_minutes = 10
        self.rate_limit_cooldown = None
        
    async def fetch_recent_news(
        self, 
        currencies: Optional[List[str]] = None,
        items: int = 50,
        date_filter: str = "last24hours",
        sentiment: Optional[str] = None
    ) -> List[Dict]:
        """
        Fetch recent crypto news from CryptoNews API
        
        Args:
            currencies: List of coin symbols (e.g., ['BTC', 'ETH'])
            items: Number of news items (1-100)
            date_filter: Time range - last5min, last15min, last30min, last60min, today, yesterday, last7days, etc.
            sentiment: Filter by sentiment - 'positive', 'negative', 'neutral' (optional)
        """
        try:
            if not self.api_key:
                logger.error("CRYPTONEWS_API_KEY not configured")
                return []
            
            params = {
                'token': self.api_key,
                'items': min(items, 100),
                'date': date_filter,
                'sortby': 'rank'
            }
            
            if sentiment:
                params['sentiment'] = sentiment
            
            if currencies:
                params['tickers'] = ','.join(currencies)
            
            endpoint = f"{self.base_url}/news" if currencies else f"{self.base_url}/all"
            
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(endpoint, params=params)
                response.raise_for_status()
                data = response.json()
                
                articles = data.get('data', [])
                
                new_articles = []
                for article in articles:
                    article_id = article.get('news_url', '') + article.get('date', '')
                    if article_id and article_id not in self.seen_news_ids:
                        new_articles.append(article)
                        self.seen_news_ids.add(article_id)
                
                logger.info(f"Fetched {len(new_articles)} new articles from CryptoNews API")
                return new_articles
                
        except Exception as e:
            logger.error(f"Error fetching news from CryptoNews API: {e}")
            return []
    
    def clean_old_seen_ids(self, hours: int = 24):
        """Clean up seen IDs older than specified hours to prevent memory bloat"""
        if len(self.seen_news_ids) > 1000:
            self.seen_news_ids.clear()
            logger.info("Cleared seen news IDs cache")
    
    async def get_important_news(self, symbols: List[str]) -> List[Dict]:
        """
        Get important/breaking news for specific symbols with caching and rate limit handling
        Returns high-impact news items sorted by importance (rank)
        """
        if self.rate_limit_cooldown:
            if datetime.utcnow() < self.rate_limit_cooldown:
                logger.warning(f"In rate limit cooldown until {self.rate_limit_cooldown}")
                return self.news_cache
            else:
                self.rate_limit_cooldown = None
        
        if self.last_fetch_time and self.news_cache:
            time_since_fetch = (datetime.utcnow() - self.last_fetch_time).total_seconds() / 60
            if time_since_fetch < self.cache_duration_minutes:
                logger.info(f"Using cached news (fetched {time_since_fetch:.1f} min ago)")
                return self.news_cache
        
        currencies = [symbol.split('/')[0] for symbol in symbols]
        
        try:
            important = await self.fetch_recent_news(
                currencies=currencies,
                items=50,
                date_filter='last60min'
            )
            
            self.news_cache = important
            self.last_fetch_time = datetime.utcnow()
            
            return important
            
        except Exception as e:
            error_msg = str(e)
            if '429' in error_msg or 'Too Many Requests' in error_msg:
                self.rate_limit_cooldown = datetime.utcnow() + timedelta(minutes=15)
                logger.warning(f"Rate limited! Cooldown until {self.rate_limit_cooldown}")
            
            return self.news_cache if self.news_cache else []
    
    def extract_coins_from_news(self, article: Dict) -> List[str]:
        """Extract mentioned coins from news article"""
        tickers = article.get('tickers', '')
        if tickers:
            return [ticker.strip().upper() for ticker in tickers.split(',')]
        return []
    
    def get_news_metadata(self, article: Dict) -> Dict:
        """Extract key metadata from news article"""
        return {
            'id': article.get('news_url', '') + article.get('date', ''),
            'title': article.get('title', ''),
            'url': article.get('news_url', ''),
            'source': article.get('source_name', 'Unknown'),
            'published_at': article.get('date'),
            'currencies': self.extract_coins_from_news(article),
            'text': article.get('text', ''),
            'image_url': article.get('image_url', ''),
            'sentiment': article.get('sentiment', 'neutral'),
            'type': article.get('type', 'article')
        }

news_monitor = NewsMonitor()
