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
        # Crypto-specific bullish
        'listed on binance', 'binance listing', 'coinbase listing', 'listed on coinbase',
        'partnership with', 'partners with', 'major partnership',
        'mainnet launch', 'mainnet live', 'v2 launch', 'v3 launch',
        'upgrade complete', 'major upgrade', 'network upgrade',
        'institutional investment', 'venture capital', 'funding round',
        'etf approved', 'etf approval', 'sec approval',
        'all-time high', 'new ath', 'breaks record',
        'major adoption', 'mass adoption', 'government adoption',
        'burns tokens', 'token burn', 'supply reduction',
        'staking rewards', 'airdrop announced', 'major airdrop',
        # Macro/World bullish for crypto
        'fed cuts rates', 'rate cut', 'interest rate cut', 'dovish fed',
        'inflation falls', 'inflation drops', 'cpi lower', 'inflation cooling',
        'dollar weakens', 'dxy falls', 'usd drops',
        'money printing', 'quantitative easing', 'qe announced',
        'china stimulus', 'economic stimulus', 'fiscal stimulus',
        'bank collapse', 'bank failure', 'banking crisis',
        'gold rallies', 'gold surges', 'safe haven demand',
        'blackrock bitcoin', 'fidelity crypto', 'institutional adoption',
        'el salvador', 'bitcoin legal tender', 'nation adopts',
        'war fears ease', 'peace talks', 'tensions ease',
        'trump crypto', 'pro-crypto regulation', 'crypto friendly'
    ],
    'SHORT': [
        # Crypto-specific bearish
        'hacked', 'exploit', 'security breach', 'funds stolen',
        'rug pull', 'exit scam', 'ponzi', 'fraud',
        'sec lawsuit', 'sued by sec', 'regulatory action', 'lawsuit filed',
        'delisted', 'delisting', 'removed from', 'suspended trading',
        'team dumps', 'insider selling', 'whale dump',
        'network down', 'chain halted', 'major outage',
        'ceo arrested', 'founder arrested', 'investigation',
        'hack confirmed', 'bridge exploit', 'contract vulnerability',
        'bankruptcy', 'insolvent', 'freezes withdrawals',
        # Macro/World bearish for crypto
        'fed raises rates', 'rate hike', 'interest rate hike', 'hawkish fed',
        'inflation rises', 'inflation surges', 'cpi higher', 'hot inflation',
        'dollar strengthens', 'dxy rallies', 'usd surges',
        'quantitative tightening', 'qt continues', 'balance sheet reduction',
        'crypto ban', 'bitcoin ban', 'mining ban', 'trading ban',
        'china crackdown', 'regulatory crackdown', 'sec crackdown',
        'recession fears', 'recession warning', 'economic downturn',
        'stock market crash', 'markets plunge', 'risk off',
        'war escalates', 'military conflict', 'geopolitical crisis',
        'mt gox distribution', 'government sells', 'whale sells',
        'tether fud', 'stablecoin concerns', 'usdt depeg'
    ]
}

# Macro news impacts BTC primarily, which affects all alts
MACRO_AFFECTS_BTC = [
    'fed', 'interest rate', 'inflation', 'cpi', 'fomc', 'powell',
    'dollar', 'dxy', 'recession', 'gdp', 'unemployment', 'jobs report',
    'treasury', 'bonds', 'yields', 'quantitative', 'stimulus',
    'china', 'russia', 'war', 'sanctions', 'geopolitical',
    'bank', 'banking', 'financial crisis', 'liquidity'
]

# 🔥 GEOPOLITICAL RISK-OFF TRIGGERS (Auto-short BTC/top coins)
RISK_OFF_TRIGGERS = {
    'SHORT': [
        # War/Military
        'bomb', 'bombs', 'bombing', 'airstrike', 'airstrikes', 'missile', 'missiles',
        'attack', 'attacks', 'strike', 'strikes', 'military action',
        'war begins', 'war declared', 'invasion', 'invades', 'invaded',
        'troops deployed', 'military operation', 'combat', 'conflict escalates',
        'iran', 'israel', 'russia', 'ukraine', 'taiwan', 'china', 'north korea',
        'nuclear', 'wmd', 'chemical weapons', 'biological weapons',
        # Terror/Crisis
        'terrorist', 'terrorism', 'explosion', 'attack on', 'assassination',
        'emergency declared', 'martial law', 'coup', 'government collapse',
        # Financial Crisis
        'bank run', 'bank collapse', 'systemic risk', 'contagion',
        'market crash', 'flash crash', 'circuit breaker', 'trading halted',
        'liquidity crisis', 'credit crisis', 'debt default', 'sovereign default',
        # Regulatory Shock
        'crypto banned', 'exchange shutdown', 'major hack', 'billions stolen'
    ],
    'LONG': [
        # De-escalation
        'ceasefire', 'peace deal', 'peace agreement', 'war ends',
        'tensions ease', 'troops withdraw', 'diplomatic solution',
        'sanctions lifted', 'trade deal', 'resolution reached'
    ]
}

# Top 10 coins to trade on macro news
TOP_COINS_FOR_MACRO = ['BTC', 'ETH', 'SOL', 'XRP', 'DOGE', 'ADA', 'AVAX', 'DOT', 'LINK', 'LTC']

_news_cache: Dict[str, datetime] = {}
NEWS_COOLDOWN_MINUTES = 30

_last_scan_time: Optional[datetime] = None
SCAN_INTERVAL_SECONDS = 30  # 30 seconds for breaking news speed


class RealtimeNewsScanner:
    def __init__(self):
        self.api_key = os.environ.get("CRYPTONEWS_API_KEY")
        self.base_url = "https://cryptonews-api.com/api/v1"
        self.seen_news: set = set()
        
    async def fetch_breaking_news(self) -> List[Dict]:
        """Fetch news from the last 15 minutes"""
        if not self.api_key:
            logger.warning("📰 NEWS SCANNER: No CRYPTONEWS_API_KEY configured - skipping")
            return []
        
        try:
            logger.info("📰 NEWS SCANNER: Fetching breaking news from CryptoNews API...")
            
            async with httpx.AsyncClient(timeout=15) as client:
                params = {
                    'token': self.api_key,
                    'items': 30,
                    'sortby': 'rank'
                }
                
                response = await client.get(self.base_url, params=params)
                
                if response.status_code == 200:
                    data = response.json()
                    articles = data.get('data', [])
                    
                    logger.info(f"📰 NEWS SCANNER: API returned {len(articles)} articles from last 15 min")
                    
                    new_articles = []
                    for article in articles:
                        news_id = article.get('news_url', '')
                        if news_id and news_id not in self.seen_news:
                            self.seen_news.add(news_id)
                            new_articles.append(article)
                            title = article.get('title', '')[:80]
                            logger.info(f"📰 NEW ARTICLE: {title}...")
                    
                    if len(self.seen_news) > 500:
                        self.seen_news = set(list(self.seen_news)[-200:])
                    
                    logger.info(f"📰 NEWS SCANNER: {len(new_articles)} NEW articles (not seen before)")
                    return new_articles
                else:
                    logger.error(f"📰 NEWS SCANNER: API error - status {response.status_code}")
                    
        except Exception as e:
            logger.error(f"📰 NEWS SCANNER: Error fetching news - {e}")
        
        return []
    
    def extract_coins_from_news(self, article: Dict) -> List[str]:
        """Extract coin symbols mentioned in the article"""
        coins = []
        
        tickers = article.get('tickers', [])
        if tickers:
            coins.extend([t.upper() for t in tickers if len(t) <= 10])
        
        title = article.get('title', '').lower()
        text = article.get('text', '').lower()
        content = title + ' ' + text
        content_upper = content.upper()
        
        # Check if this is macro/world news (affects BTC primarily)
        is_macro_news = any(keyword in content for keyword in MACRO_AFFECTS_BTC)
        
        # 🔥 NEW: Check for geopolitical RISK-OFF triggers (bombs, war, attacks)
        is_geopolitical_risk = any(trigger in content for trigger in RISK_OFF_TRIGGERS['SHORT'][:20])  # Top war/crisis keywords
        
        if is_macro_news and 'BTC' not in coins:
            coins.insert(0, 'BTC')  # BTC first for macro news
        
        # 🔥 For geopolitical risk news, auto-add top coins (don't need to mention crypto)
        if is_geopolitical_risk:
            logger.info(f"🌍 GEOPOLITICAL RISK DETECTED - Adding top coins for risk-off trade")
            for coin in TOP_COINS_FOR_MACRO[:3]:  # BTC, ETH, SOL for geo risk
                if coin not in coins:
                    coins.append(coin)
        
        common_coins = ['BTC', 'ETH', 'SOL', 'XRP', 'DOGE', 'ADA', 'AVAX', 'DOT', 'MATIC', 
                       'LINK', 'UNI', 'ATOM', 'LTC', 'NEAR', 'APT', 'ARB', 'OP', 'INJ',
                       'SUI', 'SEI', 'TIA', 'JUP', 'PEPE', 'WIF', 'BONK', 'SHIB', 'FLOKI']
        
        for coin in common_coins:
            if coin in content_upper and coin not in coins:
                coins.append(coin)
        
        # For macro/geopolitical news with no specific coins, default to BTC
        if not coins and (is_macro_news or is_geopolitical_risk):
            coins = ['BTC']
        
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
        
        # 🔥 CHECK GEOPOLITICAL RISK-OFF TRIGGERS FIRST (highest priority)
        for trigger in RISK_OFF_TRIGGERS['SHORT']:
            if trigger in content:
                short_score += 40  # High impact for geo risk
                if not trigger_reason:
                    trigger_reason = f"🌍 RISK-OFF: {trigger.title()}"
                logger.info(f"🌍 GEOPOLITICAL TRIGGER: '{trigger}' detected")
        
        for trigger in RISK_OFF_TRIGGERS['LONG']:
            if trigger in content:
                long_score += 35  # De-escalation is bullish
                if not trigger_reason:
                    trigger_reason = f"🕊️ RISK-ON: {trigger.title()}"
        
        # Check crypto-specific keywords
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
            logger.debug(f"📰 NEWS SCANNER: Skipping, {int(SCAN_INTERVAL_SECONDS - elapsed)}s until next scan")
            return None
    
    logger.info(f"📰 ═══════════════════════════════════════════════════════")
    logger.info(f"📰 NEWS SCANNER RUNNING")
    
    _last_scan_time = datetime.now()
    
    scanner = RealtimeNewsScanner()
    
    logger.info("📰 NEWS SCANNER: Starting breaking news check...")
    
    articles = await scanner.fetch_breaking_news()
    
    if not articles:
        logger.info("📰 NEWS SCANNER: No articles returned (API key missing or no recent news)")
        return None
    
    logger.info(f"📰 NEWS SCANNER: Found {len(articles)} articles to analyze")
    
    for article in articles:
        title = article.get('title', '')[:80]
        coins = scanner.extract_coins_from_news(article)
        
        if not coins:
            logger.debug(f"📰 SKIP (no coins): {title}")
            continue
        
        direction, impact_score, trigger = scanner.analyze_news_impact(article)
        
        logger.info(f"📰 ANALYZING: {title}")
        logger.info(f"   → Coins: {coins} | Direction: {direction} | Score: {impact_score}")
        
        if direction == 'NONE':
            logger.info(f"   → SKIP: No clear direction detected")
            continue
            
        if impact_score < 30:
            logger.info(f"   → SKIP: Impact score {impact_score} < 30 threshold")
            continue
        
        logger.info(f"📰 ✅ HIGH IMPACT NEWS DETECTED!")
        logger.info(f"   → Trigger: {trigger}")
        
        for coin in coins:
            symbol = f"{coin}USDT"
            logger.info(f"📰 Checking {symbol}...")
            
            if scanner.is_coin_on_cooldown(symbol):
                logger.info(f"   → SKIP {symbol}: On cooldown (signaled in last 30 min)")
                continue
            
            is_available = await check_bitunix_func(symbol)
            if not is_available:
                logger.info(f"   → SKIP {symbol}: Not available on Bitunix")
                continue
            
            price_data = await fetch_price_func(symbol)
            if not price_data:
                logger.info(f"   → SKIP {symbol}: Could not fetch price data")
                continue
            
            current_price = price_data['price']
            rsi = price_data.get('rsi', 50)
            logger.info(f"   → {symbol}: Price ${current_price:.4f} | RSI {rsi:.1f}")
            
            if direction == 'LONG':
                if rsi > 80:
                    logger.info(f"   → SKIP {symbol}: RSI {rsi:.1f} > 80 (overbought for LONG)")
                    continue
                
                tp_percent = 8.0 if impact_score >= 60 else 5.0
                if impact_score >= 80:
                    tp_percent = 15.0
                
                sl_percent = 4.0
                
                take_profit = current_price * (1 + tp_percent / 100)
                stop_loss = current_price * (1 - sl_percent / 100)
                
            else:
                if rsi < 30:
                    logger.info(f"   → SKIP {symbol}: RSI {rsi:.1f} < 30 (oversold for SHORT)")
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
            
            logger.info(f"📰 🚀 NEWS SIGNAL GENERATED!")
            logger.info(f"   → {symbol} {direction} | Entry: ${current_price:.4f}")
            logger.info(f"   → TP: {tp_percent}% | SL: {sl_percent}% | Score: {impact_score}")
            logger.info(f"   → News: {news_title}")
            
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
