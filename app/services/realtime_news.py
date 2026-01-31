"""
Real-Time News Scanner for Social & News Trading
Checks for breaking news every 1-2 minutes and triggers immediate signals
"""
import asyncio
import logging
import httpx
import os
import re
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

HIGH_IMPACT_KEYWORDS = {
    'LONG': [
        'listed on binance', 'binance listing', 'coinbase listing', 'listed on coinbase',
        'partnership with', 'partners with', 'major partnership',
        'mainnet launch', 'mainnet live', 'v2 launch', 'v3 launch',
        'upgrade complete', 'major upgrade', 'network upgrade',
        'institutional investment', 'venture capital', 'funding round',
        'etf approved', 'etf approval', 'sec approval',
        'all-time high', 'new ath', 'breaks record',
        'major adoption', 'mass adoption', 'government adoption',
        'burns tokens', 'token burn', 'supply reduction',
        'staking rewards', 'airdrop announced', 'major airdrop'
    ],
    'SHORT': [
        'hacked', 'exploit', 'security breach', 'funds stolen',
        'rug pull', 'exit scam', 'ponzi', 'fraud',
        'sec lawsuit', 'sued by sec', 'regulatory action', 'lawsuit filed',
        'delisted', 'delisting', 'removed from', 'suspended trading',
        'team dumps', 'insider selling', 'whale dump',
        'network down', 'chain halted', 'major outage',
        'ceo arrested', 'founder arrested', 'investigation',
        'hack confirmed', 'bridge exploit', 'contract vulnerability',
        'bankruptcy', 'insolvent', 'freezes withdrawals'
    ]
}

_news_cache: Dict[str, datetime] = {}
NEWS_COOLDOWN_MINUTES = 30

_last_scan_time: Optional[datetime] = None
SCAN_INTERVAL_SECONDS = 60


class RealtimeNewsScanner:
    def __init__(self):
        self.api_key = os.environ.get("CRYPTONEWS_API_KEY")
        self.base_url = "https://cryptonews-api.com/api/v1"
        self.seen_news: set = set()
        
    async def fetch_breaking_news(self) -> List[Dict]:
        """Fetch news from the last 15 minutes"""
        if not self.api_key:
            logger.debug("No CRYPTONEWS_API_KEY configured")
            return []
        
        try:
            async with httpx.AsyncClient(timeout=15) as client:
                params = {
                    'token': self.api_key,
                    'items': 30,
                    'date': 'last15min',
                    'sortby': 'rank'
                }
                
                response = await client.get(f"{self.base_url}/all", params=params)
                
                if response.status_code == 200:
                    data = response.json()
                    articles = data.get('data', [])
                    
                    new_articles = []
                    for article in articles:
                        news_id = article.get('news_url', '')
                        if news_id and news_id not in self.seen_news:
                            self.seen_news.add(news_id)
                            new_articles.append(article)
                    
                    if len(self.seen_news) > 500:
                        self.seen_news = set(list(self.seen_news)[-200:])
                    
                    logger.info(f"📰 Found {len(new_articles)} new breaking news articles")
                    return new_articles
                    
        except Exception as e:
            logger.error(f"Error fetching breaking news: {e}")
        
        return []
    
    def extract_coins_from_news(self, article: Dict) -> List[str]:
        """Extract coin symbols mentioned in the article"""
        coins = []
        
        tickers = article.get('tickers', [])
        if tickers:
            coins.extend([t.upper() for t in tickers if len(t) <= 10])
        
        title = article.get('title', '').upper()
        text = article.get('text', '').upper()
        content = title + ' ' + text
        
        common_coins = ['BTC', 'ETH', 'SOL', 'XRP', 'DOGE', 'ADA', 'AVAX', 'DOT', 'MATIC', 
                       'LINK', 'UNI', 'ATOM', 'LTC', 'NEAR', 'APT', 'ARB', 'OP', 'INJ',
                       'SUI', 'SEI', 'TIA', 'JUP', 'PEPE', 'WIF', 'BONK', 'SHIB', 'FLOKI']
        
        for coin in common_coins:
            if coin in content and coin not in coins:
                coins.append(coin)
        
        return coins[:5]
    
    def analyze_news_impact(self, article: Dict) -> Tuple[str, int, str]:
        """
        Analyze news for trading direction and impact.
        Returns: (direction, impact_score 0-100, trigger_reason)
        """
        title = article.get('title', '').lower()
        text = article.get('text', '').lower()
        content = title + ' ' + text
        
        long_score = 0
        short_score = 0
        trigger_reason = ""
        
        for keyword in HIGH_IMPACT_KEYWORDS['LONG']:
            if keyword in content:
                long_score += 20
                if not trigger_reason:
                    trigger_reason = keyword.title()
        
        for keyword in HIGH_IMPACT_KEYWORDS['SHORT']:
            if keyword in content:
                short_score += 25
                if not trigger_reason:
                    trigger_reason = keyword.title()
        
        sentiment = article.get('sentiment', 'neutral')
        if sentiment == 'positive':
            long_score += 15
        elif sentiment == 'negative':
            short_score += 15
        
        votes = article.get('votes', 0)
        if isinstance(votes, dict):
            votes = votes.get('liked', 0) + votes.get('positive', 0)
        if votes > 100:
            long_score += 10 if long_score > short_score else 0
            short_score += 10 if short_score > long_score else 0
        
        if long_score > short_score and long_score >= 20:
            return 'LONG', min(long_score, 100), trigger_reason or 'Positive News'
        elif short_score > long_score and short_score >= 20:
            return 'SHORT', min(short_score, 100), trigger_reason or 'Negative News'
        
        return 'NONE', 0, ''
    
    def is_coin_on_cooldown(self, symbol: str) -> bool:
        """Check if we recently signaled this coin"""
        if symbol in _news_cache:
            if datetime.now() - _news_cache[symbol] < timedelta(minutes=NEWS_COOLDOWN_MINUTES):
                return True
        return False
    
    def add_cooldown(self, symbol: str):
        """Add cooldown for a coin"""
        _news_cache[symbol] = datetime.now()


async def scan_for_breaking_news_signal(
    check_bitunix_func,
    fetch_price_func
) -> Optional[Dict]:
    """
    Scan for breaking news and generate immediate trading signals.
    Called frequently (every 1-2 minutes) to catch news fast.
    """
    global _last_scan_time
    
    if _last_scan_time:
        elapsed = (datetime.now() - _last_scan_time).total_seconds()
        if elapsed < SCAN_INTERVAL_SECONDS:
            return None
    
    _last_scan_time = datetime.now()
    
    scanner = RealtimeNewsScanner()
    
    articles = await scanner.fetch_breaking_news()
    
    if not articles:
        return None
    
    for article in articles:
        coins = scanner.extract_coins_from_news(article)
        
        if not coins:
            continue
        
        direction, impact_score, trigger = scanner.analyze_news_impact(article)
        
        if direction == 'NONE' or impact_score < 30:
            continue
        
        logger.info(f"📰 HIGH IMPACT NEWS: {article.get('title', '')[:60]}...")
        logger.info(f"   Direction: {direction} | Impact: {impact_score} | Coins: {coins}")
        
        for coin in coins:
            symbol = f"{coin}USDT"
            
            if scanner.is_coin_on_cooldown(symbol):
                continue
            
            is_available = await check_bitunix_func(symbol)
            if not is_available:
                continue
            
            price_data = await fetch_price_func(symbol)
            if not price_data:
                continue
            
            current_price = price_data['price']
            rsi = price_data.get('rsi', 50)
            
            if direction == 'LONG':
                if rsi > 80:
                    continue
                
                tp_percent = 8.0 if impact_score >= 60 else 5.0
                if impact_score >= 80:
                    tp_percent = 15.0
                
                sl_percent = 4.0
                
                take_profit = current_price * (1 + tp_percent / 100)
                stop_loss = current_price * (1 - sl_percent / 100)
                
            else:
                if rsi < 30:
                    continue
                
                tp_percent = 10.0 if impact_score >= 60 else 6.0
                if impact_score >= 80:
                    tp_percent = 18.0
                
                sl_percent = 5.0
                
                take_profit = current_price * (1 - tp_percent / 100)
                stop_loss = current_price * (1 + sl_percent / 100)
            
            scanner.add_cooldown(symbol)
            
            news_title = article.get('title', 'Breaking News')[:100]
            news_url = article.get('news_url', '')
            
            return {
                'symbol': symbol,
                'direction': direction,
                'entry_price': current_price,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'take_profit_1': take_profit,
                'take_profit_2': None,
                'take_profit_3': None,
                'tp_percent': tp_percent,
                'sl_percent': sl_percent,
                'confidence': impact_score,
                'reasoning': f"📰 BREAKING: {trigger} | {news_title}",
                'trade_type': 'NEWS_SIGNAL',
                'strategy': 'BREAKING_NEWS',
                'news_title': news_title,
                'news_url': news_url,
                'trigger_reason': trigger,
                'rsi': rsi
            }
    
    return None
