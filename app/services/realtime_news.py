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
NEWS_COOLDOWN_MINUTES = 60

_last_scan_time: Optional[datetime] = None
SCAN_INTERVAL_SECONDS = 120  # 2 minutes between news scans

_global_seen_news: set = set()


class RealtimeNewsScanner:
    def __init__(self):
        self.api_key = os.environ.get("CRYPTONEWS_API_KEY")
        self.base_url = "https://cryptonews-api.com/api/v1"
        self.seen_news = _global_seen_news
        
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
                    'tickers': 'BTC,ETH,SOL,XRP,DOGE,ADA,AVAX,DOT,LINK,LTC',
                    'items': 30,
                    'date': 'last60min',
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
                        keep = set(list(self.seen_news)[-200:])
                        self.seen_news.clear()
                        self.seen_news.update(keep)
                    
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
        
        geo_short_hit = False
        geo_long_hit = False
        
        # 🔥 CHECK GEOPOLITICAL RISK-OFF TRIGGERS FIRST (highest priority)
        for trigger in RISK_OFF_TRIGGERS['SHORT']:
            if trigger in content:
                short_score += 40
                geo_short_hit = True
                if not trigger_reason:
                    trigger_reason = f"🌍 RISK-OFF: {trigger.title()}"
                logger.info(f"🌍 GEOPOLITICAL TRIGGER: '{trigger}' detected")
        
        for trigger in RISK_OFF_TRIGGERS['LONG']:
            if trigger in content:
                long_score += 35
                geo_long_hit = True
                if not trigger_reason:
                    trigger_reason = f"🕊️ RISK-ON: {trigger.title()}"
        
        # If geopolitical risk-off detected, force SHORT immediately
        # Don't let other keywords cancel out the direction
        if geo_short_hit and not geo_long_hit:
            score = min(short_score, 100)
            logger.info(f"🌍 FORCED SHORT from geopolitical risk-off (score: {score})")
            return 'SHORT', max(score, 50), trigger_reason
        
        if geo_long_hit and not geo_short_hit:
            score = min(long_score, 100)
            return 'LONG', max(score, 50), trigger_reason
        
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
            
        if impact_score < 20:
            logger.info(f"   → SKIP: Impact score {impact_score} < 20 threshold")
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
            volume_24h = price_data.get('volume_24h', 0)
            volume_ratio = price_data.get('volume_ratio', 1.0)
            btc_corr = price_data.get('btc_correlation', 0.0)
            change_24h = price_data.get('change_24h', 0)
            logger.info(f"   → {symbol}: Price ${current_price:.4f} | RSI {rsi:.1f} | VolRatio {volume_ratio:.1f}x | BTC corr {btc_corr:.2f}")
            
            if direction == 'LONG':
                if rsi > 85:
                    logger.info(f"   → SKIP {symbol}: RSI {rsi:.1f} > 85 (extremely overbought)")
                    continue
                
                if impact_score >= 80:
                    base_tp = 8.0
                    base_sl = 3.0
                elif impact_score >= 60:
                    base_tp = 5.0
                    base_sl = 2.0
                elif impact_score >= 40:
                    base_tp = 3.0
                    base_sl = 1.5
                else:
                    base_tp = 2.0
                    base_sl = 1.0
                
            else:
                if rsi < 15:
                    logger.info(f"   → SKIP {symbol}: RSI {rsi:.1f} < 15 (extremely oversold)")
                    continue
                
                if impact_score >= 80:
                    base_tp = 8.0
                    base_sl = 3.0
                elif impact_score >= 60:
                    base_tp = 5.0
                    base_sl = 2.0
                elif impact_score >= 40:
                    base_tp = 3.5
                    base_sl = 1.5
                else:
                    base_tp = 2.0
                    base_sl = 1.0
            
            try:
                from app.services.coinglass import get_derivatives_summary, adjust_tp_sl_from_derivatives
                derivatives = await get_derivatives_summary(symbol)
                adj = adjust_tp_sl_from_derivatives(direction, base_tp, base_sl, derivatives)
                tp_percent = adj['tp_pct']
                sl_percent = adj['sl_pct']
                deriv_adjustments = adj['adjustments']
                if deriv_adjustments:
                    logger.info(f"   → 📊 Derivatives adjusted TP/SL: TP {base_tp:.1f}%→{tp_percent:.1f}% | SL {base_sl:.1f}%→{sl_percent:.1f}%")
            except Exception as e:
                logger.warning(f"   → Derivatives fetch failed: {e}")
                derivatives = {}
                tp_percent = base_tp
                sl_percent = base_sl
                deriv_adjustments = []
            
            if direction == 'LONG':
                take_profit = current_price * (1 + tp_percent / 100)
                stop_loss = current_price * (1 - sl_percent / 100)
            else:
                take_profit = current_price * (1 - tp_percent / 100)
                stop_loss = current_price * (1 + sl_percent / 100)
            
            news_title = article.get('title', 'Breaking News')[:100]
            news_url = article.get('news_url', '')
            
            try:
                lc_news = []
                lc_metrics = None
                influencer_data = None
                buzz_momentum = None
                try:
                    from app.services.lunarcrush import get_coin_news, get_coin_metrics, get_influencer_consensus, get_social_time_series
                    import asyncio
                    lc_news, lc_metrics, influencer_data, buzz_momentum = await asyncio.gather(
                        get_coin_news(symbol, limit=3),
                        get_coin_metrics(symbol),
                        get_influencer_consensus(symbol),
                        get_social_time_series(symbol),
                        return_exceptions=True
                    )
                    if isinstance(lc_news, Exception): lc_news = []
                    if isinstance(lc_metrics, Exception): lc_metrics = None
                    if isinstance(influencer_data, Exception): influencer_data = None
                    if isinstance(buzz_momentum, Exception): buzz_momentum = None
                except Exception as e:
                    logger.debug(f"LunarCrush cross-ref failed for {symbol}: {e}")
                
                lc_galaxy = lc_metrics.get('galaxy_score', 0) if lc_metrics else 0
                lc_sentiment = lc_metrics.get('sentiment', 0) if lc_metrics else 0
                
                news_cross_ref = ""
                if lc_news:
                    lc_titles = [n['title'][:80] for n in lc_news[:3]]
                    news_cross_ref = f"\nLunarCrush Cross-Reference ({len(lc_news)} related articles):\n" + "\n".join(f"  - {t}" for t in lc_titles)
                
                from app.services.social_signals import ai_analyze_social_signal
                ai_candidate = {
                    'symbol': symbol,
                    'direction': direction,
                    'entry_price': current_price,
                    'stop_loss': stop_loss,
                    'take_profit': take_profit,
                    'tp_percent': tp_percent,
                    'sl_percent': sl_percent,
                    'rsi': rsi,
                    'volume_ratio': volume_ratio,
                    'btc_correlation': btc_corr,
                    '24h_change': change_24h,
                    '24h_volume': volume_24h,
                    'galaxy_score': lc_galaxy,
                    'sentiment': lc_sentiment,
                    'derivatives': derivatives,
                    'deriv_adjustments': deriv_adjustments,
                    'trade_type': 'NEWS_SIGNAL',
                    'news_context': f"BREAKING NEWS: {news_title} | Trigger: {trigger} | Impact Score: {impact_score}/100{news_cross_ref}",
                    'influencer_consensus': influencer_data,
                    'buzz_momentum': buzz_momentum,
                }
                from app.services.social_signals import is_coin_in_ai_rejection_cooldown, add_to_ai_rejection_cooldown
                if is_coin_in_ai_rejection_cooldown(symbol, direction):
                    logger.info(f"   → ⏳ Skipping AI for {symbol} {direction} - in 15min rejection cooldown")
                    continue
                
                ai_result = await ai_analyze_social_signal(ai_candidate)
                
                if not ai_result.get('approved', False):
                    ai_reason = ai_result.get('reasoning', 'No reason')
                    logger.info(f"   → 🤖 AI REJECTED {symbol} {direction}: {ai_reason}")
                    add_to_ai_rejection_cooldown(symbol, direction)
                    continue
                
                logger.info(f"   → 🤖 AI APPROVED {symbol} {direction}: {ai_result.get('recommendation', '')} (conf: {ai_result.get('confidence', 0)})")
                ai_reasoning = ai_result.get('reasoning', '')
                ai_recommendation = ai_result.get('recommendation', '')
                ai_confidence = ai_result.get('confidence', 0)
            except Exception as e:
                logger.warning(f"   → AI validation failed, proceeding with news signal: {e}")
                ai_reasoning = ''
                ai_recommendation = ''
                ai_confidence = 0
            
            scanner.add_cooldown(symbol)
            
            logger.info(f"📰 🚀 NEWS SIGNAL GENERATED!")
            logger.info(f"   → {symbol} {direction} | Entry: ${current_price:.4f}")
            logger.info(f"   → TP: {tp_percent:.1f}% | SL: {sl_percent:.1f}% | Score: {impact_score}")
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
                'reasoning': ai_reasoning if ai_reasoning else f"📰 BREAKING: {trigger} | {news_title}",
                'ai_recommendation': ai_recommendation,
                'ai_confidence': ai_confidence,
                'trade_type': 'NEWS_SIGNAL',
                'strategy': 'BREAKING_NEWS',
                'news_title': news_title,
                'news_url': news_url,
                'trigger_reason': trigger,
                'rsi': rsi,
                '24h_volume': volume_24h,
                '24h_change': change_24h,
                'btc_correlation': btc_corr,
                'galaxy_score': lc_galaxy,
                'sentiment': lc_sentiment,
                'derivatives': derivatives,
                'deriv_adjustments': deriv_adjustments
            }
    
    return None
