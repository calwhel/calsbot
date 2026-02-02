"""
Twitter/X Auto-Posting Service for TradeHub AI

Features:
- Posts top gainers every few hours
- Posts market summaries
- Posts trading signals (optional)
- Generates chart images for visual posts
- Rate limited to stay within Twitter limits
"""

import asyncio
import logging
import os
import io
import random
from datetime import datetime, timedelta
from typing import Optional, Dict, List
import tweepy
import ccxt.async_support as ccxt

logger = logging.getLogger(__name__)

# Twitter API credentials
TWITTER_CONSUMER_KEY = os.getenv("TWITTER_CONSUMER_KEY")
TWITTER_CONSUMER_SECRET = os.getenv("TWITTER_CONSUMER_SECRET")
TWITTER_ACCESS_TOKEN = os.getenv("TWITTER_ACCESS_TOKEN")
TWITTER_ACCESS_TOKEN_SECRET = os.getenv("TWITTER_ACCESS_TOKEN_SECRET")
TWITTER_BEARER_TOKEN = os.getenv("TWITTER_BEARER_TOKEN")

# Posting limits - 15 posts per day
MAX_POSTS_PER_DAY = 15
POSTS_TODAY = 0
LAST_RESET = datetime.utcnow().date()

# Global coin cooldown tracking (shared across all accounts)
GLOBAL_COIN_POSTS = {}  # {symbol: count}
GLOBAL_COIN_RESET_DATE = None


def check_global_coin_cooldown(symbol: str, max_per_day: int = 2) -> bool:
    """Check if coin can be posted (global cooldown across all accounts)"""
    global GLOBAL_COIN_POSTS, GLOBAL_COIN_RESET_DATE
    
    today = datetime.utcnow().date()
    
    # Reset at midnight
    if GLOBAL_COIN_RESET_DATE != today:
        GLOBAL_COIN_POSTS = {}
        GLOBAL_COIN_RESET_DATE = today
    
    count = GLOBAL_COIN_POSTS.get(symbol, 0)
    return count < max_per_day


def record_global_coin_post(symbol: str):
    """Record that a coin was posted (global tracking)"""
    global GLOBAL_COIN_POSTS, GLOBAL_COIN_RESET_DATE
    
    today = datetime.utcnow().date()
    if GLOBAL_COIN_RESET_DATE != today:
        GLOBAL_COIN_POSTS = {}
        GLOBAL_COIN_RESET_DATE = today
    
    GLOBAL_COIN_POSTS[symbol] = GLOBAL_COIN_POSTS.get(symbol, 0) + 1
    logger.info(f"📊 GLOBAL: {symbol} posted {GLOBAL_COIN_POSTS[symbol]}x today")

# Auto-posting enabled
AUTO_POST_ENABLED = True


class TwitterPoster:
    """Handles automated posting to Twitter/X"""
    
    def __init__(self):
        self.client = None
        self.api_v1 = None  # For media uploads
        self.posts_today = 0
        self.last_reset = datetime.utcnow().date()
        self.last_post_time = None
        self.coin_post_count = {}  # Track per-coin posts: {symbol: count}
        self.coin_post_reset = datetime.utcnow().date()
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize Twitter API clients"""
        try:
            # Re-read env vars in case they were set after module load
            consumer_key = os.getenv("TWITTER_CONSUMER_KEY")
            consumer_secret = os.getenv("TWITTER_CONSUMER_SECRET")
            access_token = os.getenv("TWITTER_ACCESS_TOKEN")
            access_token_secret = os.getenv("TWITTER_ACCESS_TOKEN_SECRET")
            bearer_token = os.getenv("TWITTER_BEARER_TOKEN")
            
            missing = []
            if not consumer_key: missing.append("TWITTER_CONSUMER_KEY")
            if not consumer_secret: missing.append("TWITTER_CONSUMER_SECRET")
            if not access_token: missing.append("TWITTER_ACCESS_TOKEN")
            if not access_token_secret: missing.append("TWITTER_ACCESS_TOKEN_SECRET")
            
            if missing:
                logger.warning(f"Twitter credentials missing: {', '.join(missing)}")
                return
            
            # v2 Client for posting tweets (bearer_token is optional)
            self.client = tweepy.Client(
                consumer_key=consumer_key,
                consumer_secret=consumer_secret,
                access_token=access_token,
                access_token_secret=access_token_secret,
                bearer_token=bearer_token if bearer_token else None
            )
            
            # v1.1 API for media uploads
            auth = tweepy.OAuth1UserHandler(
                consumer_key,
                consumer_secret,
                access_token,
                access_token_secret
            )
            self.api_v1 = tweepy.API(auth)
            
            logger.info("✅ Twitter API initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Twitter API: {e}")
    
    def _check_rate_limit(self) -> bool:
        """Check if we can post (daily limit)"""
        today = datetime.utcnow().date()
        
        # Reset counter at midnight UTC
        if today != self.last_reset:
            self.posts_today = 0
            self.last_reset = today
        
        if self.posts_today >= MAX_POSTS_PER_DAY:
            logger.warning(f"Daily post limit reached: {self.posts_today}/{MAX_POSTS_PER_DAY}")
            return False
        
        return True
    
    async def post_tweet(self, text: str, media_ids: List[str] = None) -> Optional[Dict]:
        """Post a tweet with optional media"""
        if not self.client:
            logger.error("Twitter client not initialized")
            return None
        
        if not self._check_rate_limit():
            return None
        
        try:
            # Use v2 API (required for Free tier)
            if media_ids:
                response = self.client.create_tweet(text=text, media_ids=media_ids)
            else:
                response = self.client.create_tweet(text=text)
            
            self.posts_today += 1
            self.last_post_time = datetime.utcnow()
            
            tweet_id = response.data['id']
            logger.info(f"✅ Tweet posted: {tweet_id} | Posts today: {self.posts_today}/{MAX_POSTS_PER_DAY}")
            
            return {
                'success': True,
                'tweet_id': tweet_id,
                'text': text[:50] + '...' if len(text) > 50 else text
            }
            
        except tweepy.Forbidden as e:
            error_msg = f"403 Forbidden - {str(e)}"
            logger.error(f"Twitter 403 error: {e}")
            return {'success': False, 'error': error_msg}
        except tweepy.Unauthorized as e:
            error_msg = f"401 Unauthorized - {str(e)}"
            logger.error(f"Twitter 401 error: {e}")
            return {'success': False, 'error': error_msg}
        except tweepy.TweepyException as e:
            logger.error(f"Twitter API error: {e}")
            return {'success': False, 'error': str(e)}
        except Exception as e:
            logger.error(f"Failed to post tweet: {e}")
            return {'success': False, 'error': str(e)}
    
    async def upload_media(self, image_bytes: bytes, alt_text: str = None) -> Optional[str]:
        """Upload media and return media_id"""
        if not self.api_v1:
            logger.error("Twitter v1 API not initialized for media upload")
            return None
        
        try:
            # Upload media using v1.1 API
            logger.info(f"Uploading {len(image_bytes)} bytes to Twitter media endpoint...")
            media = self.api_v1.media_upload(filename="chart.png", file=io.BytesIO(image_bytes))
            
            if alt_text:
                try:
                    self.api_v1.create_media_metadata(media.media_id, alt_text=alt_text)
                except Exception as e:
                    logger.warning(f"Failed to set alt text (non-fatal): {e}")
            
            logger.info(f"✅ Media uploaded successfully: {media.media_id}")
            return str(media.media_id)
            
        except tweepy.Forbidden as e:
            logger.error(f"❌ Media upload 403 Forbidden - Free tier may not support media: {e}")
            return None
        except tweepy.Unauthorized as e:
            logger.error(f"❌ Media upload 401 Unauthorized: {e}")
            return None
        except Exception as e:
            logger.error(f"❌ Failed to upload media: {type(e).__name__}: {e}")
            return None
    
    async def get_top_gainers_data(self, limit: int = 5) -> List[Dict]:
        """Fetch top gaining coins from Binance"""
        try:
            exchange = ccxt.binance({'enableRateLimit': True})
            tickers = await exchange.fetch_tickers()
            await exchange.close()
            
            # Filter USDT pairs and sort by % change
            usdt_tickers = []
            for symbol, data in tickers.items():
                if symbol.endswith('/USDT') and data.get('percentage'):
                    # Skip stablecoins
                    base = symbol.replace('/USDT', '')
                    if base in ['USDC', 'BUSD', 'DAI', 'TUSD', 'USDP']:
                        continue
                    
                    usdt_tickers.append({
                        'symbol': base,
                        'price': data['last'],
                        'change': data['percentage'],
                        'volume': data.get('quoteVolume', 0)
                    })
            
            # Sort by change and return top gainers
            usdt_tickers.sort(key=lambda x: x['change'], reverse=True)
            return usdt_tickers[:limit]
            
        except Exception as e:
            logger.error(f"Failed to fetch top gainers: {e}")
            return []
    
    async def get_market_summary(self) -> Dict:
        """Get BTC and overall market summary"""
        try:
            exchange = ccxt.binance({'enableRateLimit': True})
            
            # Get BTC and ETH
            btc = await exchange.fetch_ticker('BTC/USDT')
            eth = await exchange.fetch_ticker('ETH/USDT')
            
            await exchange.close()
            
            return {
                'btc_price': btc['last'],
                'btc_change': btc['percentage'] or 0,
                'eth_price': eth['last'],
                'eth_change': eth['percentage'] or 0,
                'timestamp': datetime.utcnow()
            }
            
        except Exception as e:
            logger.error(f"Failed to fetch market summary: {e}")
            return {}
    
    async def post_top_gainers(self) -> Optional[Dict]:
        """Post top gainers update"""
        gainers = await self.get_top_gainers_data(5)
        
        if not gainers:
            return None
        
        # Massive variety for human-like posts
        headers = [
            "🚀 TOP 5 GAINERS RIGHT NOW",
            "📈 BIGGEST MOVERS TODAY",
            "🔥 HOT COINS ALERT",
            "💹 TODAY'S TOP PERFORMERS",
            "⚡ COINS PUMPING NOW",
            "🎯 Who's winning today?",
            "💰 Money is flowing here",
            "📊 Check out these movers",
            "🏆 Today's champions",
            "🔥 These coins are cooking",
            "👀 What everyone's watching",
            "💎 Green across the board",
            "⬆️ Up only mode activated",
            "🎢 Riding the wave",
            "✨ Shining bright today",
            "🌊 The tide is rising",
            "💪 Strength in these coins",
            "🚨 Alert: Major movers",
            "📍 Where the action is",
            "🎲 Today's hot picks"
        ]
        
        lines = [f"{random.choice(headers)}\n"]
        
        for i, coin in enumerate(gainers, 1):
            emoji = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else "🔥" if i == 4 else "📈"
            change_sign = "+" if coin['change'] >= 0 else ""
            price = coin.get('price', 0)
            price_str = f"${price:,.4f}" if price < 1 else f"${price:,.2f}"
            vol = coin.get('volume', 0)
            vol_str = f"${vol/1e6:.1f}M" if vol < 1e9 else f"${vol/1e9:.1f}B"
            lines.append(f"{emoji} ${coin['symbol']} {change_sign}{coin['change']:.1f}% @ {price_str} ({vol_str} vol)")
        
        # Non-question tips (80%) vs question tips (20%)
        if random.random() < 0.8:
            tips = [
                "\n💡 High volume = more conviction",
                "\n📊 Watch for continuation patterns",
                "\n🎯 Set your targets wisely",
                "\n⚠️ Always manage your risk",
                "\n🔍 DYOR before entering",
                "\n📈 Momentum is everything",
                "\n💪 Bulls in control today",
                "\n⚡ Fast movers need fast decisions",
                "\n🔥 The heat is real",
                "\n💎 Diamond hands prevail",
                "\n🎯 Pick your entries carefully",
                "\n📍 Mark these on your charts",
                "\n🧠 Trade smart, not hard",
                "\n⏰ Timing is everything",
                "\n🚀 Room to run"
            ]
        else:
            tips = [
                "\n💭 What's on your watchlist?",
                "\n🤔 Any of these catching your eye?",
                "\n🎲 Which one are you playing?",
                "\n🌊 Ride the wave or wait?"
            ]
        lines.append(random.choice(tips))
        lines.append("\n#Crypto #Trading #TopGainers #Altcoins")
        
        tweet_text = "\n".join(lines)
        return await self.post_tweet(tweet_text)
    
    async def post_market_summary(self) -> Optional[Dict]:
        """Post market summary update with variety"""
        market = await self.get_market_summary()
        
        if not market:
            return None
        
        btc_emoji = "🟢" if market['btc_change'] >= 0 else "🔴"
        eth_emoji = "🟢" if market['eth_change'] >= 0 else "🔴"
        btc_sign = "+" if market['btc_change'] >= 0 else ""
        eth_sign = "+" if market['eth_change'] >= 0 else ""
        
        # Pick random style (1-6), style 4 is question - only 10% chance
        style = random.randint(1, 6)
        if style == 4 and random.random() > 0.1:
            style = random.choice([1, 2, 3, 5, 6])
        
        # Mood based on BTC
        if market['btc_change'] >= 5:
            mood = random.choice(["Euphoria mode 🚀", "Bulls running wild", "Green everywhere", "Party in crypto land"])
        elif market['btc_change'] >= 2:
            mood = random.choice(["Looking good today", "Bulls in control", "Positive vibes", "Green candles printing"])
        elif market['btc_change'] >= 0:
            mood = random.choice(["Slow grind up", "Quiet day so far", "Steady as she goes", "Nothing crazy"])
        elif market['btc_change'] >= -3:
            mood = random.choice(["Little pullback", "Some red today", "Bears nibbling", "Slight dip"])
        else:
            mood = random.choice(["Pain in the markets", "Bears in control", "Red day", "Rough out there"])
        
        # ETH/BTC ratio comment
        if market['eth_change'] > market['btc_change'] + 2:
            eth_note = random.choice(["ETH outperforming today", "ETH leading the charge", "ETH looking strong vs BTC"])
        elif market['btc_change'] > market['eth_change'] + 2:
            eth_note = random.choice(["BTC dominance rising", "BTC leading today", "ETH lagging BTC"])
        else:
            eth_note = random.choice(["Moving together", "Correlated moves", "BTC and ETH in sync"])
        
        if style == 1:
            tweet_text = f"""Quick market check 👀

BTC: ${market['btc_price']:,.0f} ({btc_sign}{market['btc_change']:.1f}%)
ETH: ${market['eth_price']:,.0f} ({eth_sign}{market['eth_change']:.1f}%)

{mood}

#Crypto #Bitcoin"""
        
        elif style == 2:
            tweet_text = f"""{btc_emoji} {eth_emoji} Market Pulse

₿ {btc_sign}{market['btc_change']:.1f}%
⟠ {eth_sign}{market['eth_change']:.1f}%

{eth_note}

{random.choice(["How are you positioned?", "What's your read?", "Thoughts?", "Trading this?"])} 💬

#Bitcoin #Ethereum"""
        
        elif style == 3:
            # Just vibes, minimal numbers
            tweet_text = f"""{mood}

BTC @ ${market['btc_price']:,.0f}
ETH @ ${market['eth_price']:,.0f}

{random.choice(["NFA", "DYOR", "Stay sharp", "Eyes on the charts"])} 👁️

#Crypto"""
        
        elif style == 4:
            # Question format
            questions = [
                "How's everyone feeling about the market today?",
                "What's your strategy in this market?",
                "Accumulating or waiting?",
                "Bull or bear from here?",
                "What's your next move?"
            ]
            tweet_text = f"""{random.choice(questions)} 🤔

{btc_emoji} BTC {btc_sign}{market['btc_change']:.1f}%
{eth_emoji} ETH {eth_sign}{market['eth_change']:.1f}%

Drop your thoughts 👇

#Bitcoin #Crypto"""
        
        elif style == 5:
            # Story format
            if market['btc_change'] >= 0 and market['eth_change'] >= 0:
                story = "Both majors green ✅"
            elif market['btc_change'] < 0 and market['eth_change'] < 0:
                story = "Both majors red today"
            else:
                story = "Mixed signals in the market"
            
            tweet_text = f"""{story}

₿ ${market['btc_price']:,.0f}
⟠ ${market['eth_price']:,.0f}

{mood}

#Crypto #Trading"""
        
        else:
            # Classic but varied
            headers = ["📊 Market Check", "🌐 Crypto Update", "📈 How we lookin?", "💹 Market Snapshot", "👀 Quick Update"]
            tweet_text = f"""{random.choice(headers)}

{btc_emoji} BTC ${market['btc_price']:,.0f} ({btc_sign}{market['btc_change']:.1f}%)
{eth_emoji} ETH ${market['eth_price']:,.0f} ({eth_sign}{market['eth_change']:.1f}%)

{random.choice([mood, eth_note])}

#Bitcoin #Ethereum #Crypto"""
        
        return await self.post_tweet(tweet_text)
    
    async def post_top_losers(self) -> Optional[Dict]:
        """Post top losing coins"""
        try:
            exchange = ccxt.binance({'enableRateLimit': True})
            tickers = await exchange.fetch_tickers()
            await exchange.close()
            
            usdt_tickers = []
            for symbol, data in tickers.items():
                if symbol.endswith('/USDT') and data.get('percentage'):
                    base = symbol.replace('/USDT', '')
                    if base in ['USDC', 'BUSD', 'DAI', 'TUSD', 'USDP']:
                        continue
                    usdt_tickers.append({
                        'symbol': base,
                        'price': data['last'],
                        'change': data['percentage'],
                    })
            
            usdt_tickers.sort(key=lambda x: x['change'])
            losers = usdt_tickers[:5]
            
            headers = [
                "📉 BIGGEST LOSERS (24H)",
                "🩸 COINS BLEEDING TODAY",
                "💀 TOP 5 DUMPS",
                "⚠️ RED ALERT: BIGGEST DROPS",
                "📊 WORST PERFORMERS TODAY",
                "😬 Ouch... these are hurting",
                "🔴 Red day for these coins",
                "📉 Pain across the board",
                "💔 Holders not happy today",
                "⬇️ Who's catching the knife?",
                "🎢 Down bad today",
                "😰 Brutal day for these",
                "🚨 Major dumps happening",
                "📊 Blood in the streets",
                "🥶 Cold day in crypto",
                "💸 Selling pressure intense",
                "⚠️ These are getting hit hard",
                "🔻 Down only mode",
                "😵 Getting rekt today",
                "📉 Bearish on these"
            ]
            
            lines = [f"{random.choice(headers)}\n"]
            for i, coin in enumerate(losers, 1):
                emoji = "💀" if i == 1 else "🩸" if i == 2 else "📉"
                price = coin.get('price', 0)
                price_str = f"${price:,.4f}" if price < 1 else f"${price:,.2f}"
                lines.append(f"{emoji} ${coin['symbol']} {coin['change']:.1f}% @ {price_str}")
            
            # Non-question tips (80%) vs question tips (20%)
            if random.random() < 0.8:
                tips = [
                    "\n💡 Dead cat bounce incoming?",
                    "\n📊 Check support levels before buying",
                    "\n🎯 Patience is key in these moments",
                    "\n⏰ Wait for confirmation",
                    "\n🧠 Don't FOMO into falling coins",
                    "\n💪 Only strong hands survive this",
                    "\n🌊 Waiting for the reversal",
                    "\n📍 Key support levels to watch",
                    "\n🎯 Be patient, be smart",
                    "\n⚠️ Catching knives is risky",
                    "\n📉 Blood in the streets"
                ]
            else:
                tips = [
                    "\n🤔 Dip or dead?",
                    "\n💭 Would you buy any of these?",
                    "\n🔮 Where's the bottom?"
                ]
            lines.append(random.choice(tips))
            lines.append("\n#Crypto #CryptoNews #Altcoins #Trading")
            tweet_text = "\n".join(lines)
            return await self.post_tweet(tweet_text)
            
        except Exception as e:
            logger.error(f"Failed to post top losers: {e}")
            return None
    
    async def post_btc_update(self) -> Optional[Dict]:
        """Post detailed BTC update with variety"""
        try:
            exchange = ccxt.binance({'enableRateLimit': True})
            btc = await exchange.fetch_ticker('BTC/USDT')
            
            # Get some chart analysis
            ohlcv = await exchange.fetch_ohlcv('BTC/USDT', '1h', limit=24)
            await exchange.close()
            
            price = btc['last']
            change = btc['percentage'] or 0
            high = btc['high']
            low = btc['low']
            volume = btc['quoteVolume'] or 0
            
            emoji = "🟢" if change >= 0 else "🔴"
            sign = "+" if change >= 0 else ""
            
            # Calculate simple RSI
            if ohlcv and len(ohlcv) >= 14:
                closes = [c[4] for c in ohlcv]
                gains = [max(0, closes[i] - closes[i-1]) for i in range(1, len(closes))]
                losses = [max(0, closes[i-1] - closes[i]) for i in range(1, len(closes))]
                avg_gain = sum(gains[-14:]) / 14
                avg_loss = sum(losses[-14:]) / 14 or 0.0001
                rsi = 100 - (100 / (1 + avg_gain / avg_loss))
            else:
                rsi = 50
            
            # Distance from high/low
            range_size = high - low if high > low else 1
            position_in_range = (price - low) / range_size * 100
            
            # Pick random style (1-6), style 5 is question - only 10% chance
            style = random.randint(1, 6)
            if style == 5 and random.random() > 0.1:
                style = random.choice([1, 2, 3, 4, 6])
            
            # Mood commentary
            if change >= 5:
                mood = random.choice(["Bitcoin is PUMPING 🚀", "BTC on fire today", "Bulls absolutely ripping", "Major move happening"])
            elif change >= 2:
                mood = random.choice(["Solid day for Bitcoin", "BTC looking healthy", "Green candles printing", "Bulls in control"])
            elif change >= 0:
                mood = random.choice(["Quiet grind up", "Steady day", "Calm before the storm?", "Consolidating nicely"])
            elif change >= -3:
                mood = random.choice(["Small pullback", "Healthy dip", "Bears testing", "Nothing major"])
            else:
                mood = random.choice(["Rough day for BTC", "Bears winning today", "Ouch", "Pain in the market"])
            
            # RSI commentary
            if rsi >= 70:
                rsi_note = random.choice(["RSI getting hot", "Overbought territory", "Extended here"])
            elif rsi <= 30:
                rsi_note = random.choice(["Oversold levels", "Could bounce from here", "RSI bottoming"])
            else:
                rsi_note = random.choice(["RSI looks balanced", "Room to move either way", "Healthy RSI"])
            
            if style == 1:
                tweet_text = f"""₿ Bitcoin Check

${price:,.0f} ({sign}{change:.1f}%)

{mood}

{random.choice(["Where to from here?", "What's your target?", "Accumulating?", "Thoughts?"])} 🤔

#Bitcoin #BTC"""
            
            elif style == 2:
                tweet_text = f"""{emoji} BTC @ ${price:,.0f}

24h Range: ${low:,.0f} - ${high:,.0f}
Currently: {position_in_range:.0f}% of range

{rsi_note}

#Bitcoin"""
            
            elif style == 3:
                # Casual
                if change >= 0:
                    opener = random.choice(["BTC vibing today", "Bitcoin doing its thing", "Another day, another candle"])
                else:
                    opener = random.choice(["BTC taking a breather", "Bitcoin cooling off", "Red day for King BTC"])
                
                tweet_text = f"""{opener} 👀

${price:,.0f}

{random.choice([mood, rsi_note])}

#BTC #Crypto"""
            
            elif style == 4:
                # Just price and observation
                observations = []
                if position_in_range > 80:
                    observations.append("Near daily high")
                elif position_in_range < 20:
                    observations.append("Near daily low")
                if rsi >= 65:
                    observations.append(f"RSI elevated ({rsi:.0f})")
                elif rsi <= 35:
                    observations.append(f"RSI low ({rsi:.0f})")
                if abs(change) >= 3:
                    observations.append(f"Big move: {sign}{change:.1f}%")
                
                if not observations:
                    observations = [mood]
                
                tweet_text = f"""Bitcoin ${price:,.0f}

{chr(10).join(['• ' + o for o in observations[:3]])}

#Bitcoin #BTC"""
            
            elif style == 5:
                # Question
                questions = [
                    f"BTC at ${price:,.0f} - what's the play?",
                    f"${price:,.0f} - loading or waiting?",
                    f"Bitcoin {sign}{change:.1f}% - bullish or bearish?",
                    f"Where's BTC heading from ${price:,.0f}?",
                    f"${price:,.0f} - fair value or nah?"
                ]
                tweet_text = f"""{random.choice(questions)} 🤔

{random.choice([mood, rsi_note])}

Drop your take 👇

#Bitcoin"""
            
            else:
                # Volume focus
                vol_str = f"${volume/1e9:.1f}B" if volume >= 1e9 else f"${volume/1e6:.0f}M"
                
                tweet_text = f"""₿ BTC Update

{emoji} ${price:,.0f} ({sign}{change:.1f}%)
📊 Volume: {vol_str}

{mood}

#Bitcoin #BTC #Crypto"""
            
            return await self.post_tweet(tweet_text)
            
        except Exception as e:
            logger.error(f"Failed to post BTC update: {e}")
            return None
    
    async def post_altcoin_movers(self) -> Optional[Dict]:
        """Post notable altcoin movements with variety"""
        try:
            exchange = ccxt.binance({'enableRateLimit': True})
            tickers = await exchange.fetch_tickers()
            await exchange.close()
            
            alts = []
            exclude = ['BTC', 'ETH', 'USDC', 'BUSD', 'DAI', 'TUSD', 'USDP', 'USDT']
            
            for symbol, data in tickers.items():
                if symbol.endswith('/USDT') and data.get('percentage'):
                    base = symbol.replace('/USDT', '')
                    if base in exclude:
                        continue
                    vol = data.get('quoteVolume', 0) or 0
                    if vol >= 10_000_000:
                        alts.append({
                            'symbol': base,
                            'change': data['percentage'],
                            'volume': vol
                        })
            
            alts.sort(key=lambda x: abs(x['change']), reverse=True)
            top_movers = alts[:6]
            
            if not top_movers:
                return None
            
            # Count green vs red
            green_count = sum(1 for c in top_movers if c['change'] >= 0)
            red_count = len(top_movers) - green_count
            
            # Pick random style (1-5), style 3 is question - only 10% chance
            style = random.randint(1, 5)
            if style == 3 and random.random() > 0.1:
                style = random.choice([1, 2, 4, 5])
            
            if style == 1:
                # Simple list with casual header
                headers = [
                    "Altcoins doing things 👀",
                    "What the alts are up to",
                    "Alt action today",
                    "Eyes on these alts",
                    "Altcoin check-in"
                ]
                lines = [f"{random.choice(headers)}\n"]
                for coin in top_movers[:5]:
                    emoji = "🟢" if coin['change'] >= 0 else "🔴"
                    sign = "+" if coin['change'] >= 0 else ""
                    lines.append(f"{emoji} ${coin['symbol']} {sign}{coin['change']:.1f}%")
                lines.append("\n#Altcoins #Crypto")
                tweet_text = "\n".join(lines)
            
            elif style == 2:
                # Just top 3 with commentary
                top3 = top_movers[:3]
                if all(c['change'] >= 0 for c in top3):
                    mood = random.choice(["Alts pumping today 🚀", "Green across the board", "Altseason vibes"])
                elif all(c['change'] < 0 for c in top3):
                    mood = random.choice(["Alts struggling today", "Red day for alts", "Bears attacking alts"])
                else:
                    mood = random.choice(["Mixed bag in altland", "Some winners, some losers", "Split action today"])
                
                lines = [f"{mood}\n"]
                for coin in top3:
                    sign = "+" if coin['change'] >= 0 else ""
                    lines.append(f"${coin['symbol']} {sign}{coin['change']:.1f}%")
                lines.append("\n#Altcoins #Trading")
                tweet_text = "\n".join(lines)
            
            elif style == 3:
                # Question format
                biggest = top_movers[0]
                sign = "+" if biggest['change'] >= 0 else ""
                
                questions = [
                    f"${biggest['symbol']} leading the alts with {sign}{biggest['change']:.1f}%",
                    f"Big move on ${biggest['symbol']} today ({sign}{biggest['change']:.1f}%)",
                    f"${biggest['symbol']} making noise - {sign}{biggest['change']:.1f}%"
                ]
                
                tweet_text = f"""{random.choice(questions)}

Anyone else watching this? 👀

Other movers:
{chr(10).join([f"• ${c['symbol']} {'+' if c['change'] >= 0 else ''}{c['change']:.1f}%" for c in top_movers[1:4]])}

#Altcoins #Crypto"""
            
            elif style == 4:
                # Sentiment summary
                if green_count > red_count:
                    sentiment = random.choice(["More green than red today ✅", "Bulls winning in altland", "Alts looking healthy"])
                elif red_count > green_count:
                    sentiment = random.choice(["More red than green today", "Bears in control of alts", "Tough day for alts"])
                else:
                    sentiment = random.choice(["Split market today", "50/50 in altland", "Balanced action"])
                
                tweet_text = f"""{sentiment}

Top movers:
{chr(10).join([f"{'🟢' if c['change'] >= 0 else '🔴'} ${c['symbol']} {'+' if c['change'] >= 0 else ''}{c['change']:.1f}%" for c in top_movers[:4]])}

#Altcoins #Trading"""
            
            else:
                # Classic with tip
                headers = ["🔥 ALTCOIN MOVERS", "💹 ALT ACTION", "📊 ALTCOIN UPDATE", "⚡ ALT PULSE"]
                tips = [
                    "Volume confirms conviction",
                    "Follow the momentum",
                    "Watch for continuation",
                    "Set your levels",
                    "Trade what you see"
                ]
                lines = [f"{random.choice(headers)}\n"]
                for coin in top_movers[:5]:
                    emoji = "🟢" if coin['change'] >= 0 else "🔴"
                    sign = "+" if coin['change'] >= 0 else ""
                    lines.append(f"{emoji} ${coin['symbol']} {sign}{coin['change']:.1f}%")
                lines.append(f"\n💡 {random.choice(tips)}")
                lines.append("\n#Altcoins #CryptoTrading")
                tweet_text = "\n".join(lines)
            
            return await self.post_tweet(tweet_text)
            
        except Exception as e:
            logger.error(f"Failed to post altcoin movers: {e}")
            return None
    
    async def post_signal_alert(self, symbol: str, direction: str, entry: float, 
                                 tp: float, sl: float, confidence: int) -> Optional[Dict]:
        """Post a trading signal alert"""
        direction_emoji = "🟢 LONG" if direction == "LONG" else "🔴 SHORT"
        
        # Calculate TP/SL percentages
        if direction == "LONG":
            tp_pct = ((tp - entry) / entry) * 100
            sl_pct = ((entry - sl) / entry) * 100
        else:
            tp_pct = ((entry - tp) / entry) * 100
            sl_pct = ((sl - entry) / entry) * 100
        
        tweet_text = f"""⚡ SIGNAL ALERT

{direction_emoji} ${symbol.replace('/USDT', '').replace('USDT', '')}

📍 Entry: ${entry:.4f}
🎯 TP: ${tp:.4f} (+{tp_pct:.1f}%)
🛑 SL: ${sl:.4f} (-{sl_pct:.1f}%)
📊 Confidence: {confidence}/10

⚠️ NFA - DYOR

#CryptoSignals #Trading #TradeHub"""
        
        return await self.post_tweet(tweet_text)
    
    def _check_coin_cooldown(self, symbol: str, max_per_day: int = 2) -> bool:
        """Check if coin has been posted too many times today (uses GLOBAL tracking)"""
        return check_global_coin_cooldown(symbol, max_per_day)
    
    def _record_coin_post(self, symbol: str):
        """Record that a coin was posted (uses GLOBAL tracking)"""
        record_global_coin_post(symbol)
    
    async def _get_chart_analysis(self, symbol: str) -> Dict:
        """Get technical analysis data for more insightful posts"""
        try:
            exchange = ccxt.binance({'enableRateLimit': True})
            
            # Get OHLCV data for analysis
            ohlcv = await exchange.fetch_ohlcv(f"{symbol}/USDT", '1h', limit=48)
            await exchange.close()
            
            if not ohlcv or len(ohlcv) < 20:
                return {}
            
            closes = [c[4] for c in ohlcv]
            highs = [c[2] for c in ohlcv]
            lows = [c[3] for c in ohlcv]
            volumes = [c[5] for c in ohlcv]
            
            current_price = closes[-1]
            
            # Calculate RSI
            gains = []
            losses = []
            for i in range(1, min(15, len(closes))):
                diff = closes[i] - closes[i-1]
                if diff > 0:
                    gains.append(diff)
                    losses.append(0)
                else:
                    gains.append(0)
                    losses.append(abs(diff))
            
            avg_gain = sum(gains) / len(gains) if gains else 0
            avg_loss = sum(losses) / len(losses) if losses else 0.0001
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            
            # Simple EMAs
            ema9 = sum(closes[-9:]) / 9 if len(closes) >= 9 else current_price
            ema21 = sum(closes[-21:]) / 21 if len(closes) >= 21 else current_price
            
            # Support/Resistance (recent high/low)
            recent_high = max(highs[-24:]) if len(highs) >= 24 else max(highs)
            recent_low = min(lows[-24:]) if len(lows) >= 24 else min(lows)
            
            # Trend direction
            trend = "bullish" if ema9 > ema21 else "bearish" if ema9 < ema21 else "neutral"
            
            # Volume comparison
            avg_vol = sum(volumes[-24:]) / 24 if len(volumes) >= 24 else sum(volumes) / len(volumes)
            current_vol = volumes[-1]
            vol_ratio = current_vol / avg_vol if avg_vol > 0 else 1
            
            # Distance from key levels
            dist_from_high = ((recent_high - current_price) / current_price) * 100
            dist_from_low = ((current_price - recent_low) / current_price) * 100
            
            return {
                'rsi': round(rsi, 1),
                'trend': trend,
                'ema9': ema9,
                'ema21': ema21,
                'recent_high': recent_high,
                'recent_low': recent_low,
                'vol_ratio': round(vol_ratio, 1),
                'dist_from_high': round(dist_from_high, 1),
                'dist_from_low': round(dist_from_low, 1),
                'above_ema': current_price > ema21
            }
            
        except Exception as e:
            logger.debug(f"Chart analysis failed for {symbol}: {e}")
            return {}
    
    async def _generate_varied_featured_tweet(self, symbol: str, price: float, price_str: str, 
                                               change: float, sign: str, vol_str: str, 
                                               analysis: Dict) -> str:
        """Generate highly varied tweets with different formats and analysis"""
        
        # Pick a random format style (1-8)
        # Style 6 is question format - only 10% chance
        style = random.randint(1, 8)
        if style == 6 and random.random() > 0.1:
            style = random.choice([1, 2, 3, 4, 5, 7, 8])
        
        rsi = analysis.get('rsi', 50)
        trend = analysis.get('trend', 'neutral')
        vol_ratio = analysis.get('vol_ratio', 1)
        dist_high = analysis.get('dist_from_high', 0)
        dist_low = analysis.get('dist_from_low', 0)
        above_ema = analysis.get('above_ema', True)
        
        # RSI description
        if rsi >= 70:
            rsi_text = random.choice(["RSI is heated", "Getting overbought", "RSI running hot", "Momentum stretched"])
        elif rsi >= 60:
            rsi_text = random.choice(["RSI healthy", "Good momentum", "RSI looks solid", "Strength showing"])
        elif rsi <= 30:
            rsi_text = random.choice(["RSI oversold", "Could be a bounce setup", "RSI bottoming", "Oversold territory"])
        else:
            rsi_text = random.choice(["RSI neutral", "RSI in range", "Normal RSI levels", "RSI looks balanced"])
        
        # Trend description
        if trend == "bullish":
            trend_text = random.choice(["Trend is UP", "Bulls in control", "Uptrend intact", "Riding the wave up"])
        elif trend == "bearish":
            trend_text = random.choice(["Trend is DOWN", "Bears pushing", "Downtrend active", "Sellers in control"])
        else:
            trend_text = random.choice(["Trend unclear", "Ranging price action", "Consolidating", "No clear direction"])
        
        # Volume description
        if vol_ratio >= 2:
            vol_text = random.choice(["Volume exploding", "Massive volume spike", "Huge interest", "Volume is insane"])
        elif vol_ratio >= 1.3:
            vol_text = random.choice(["Above average volume", "Good volume", "Volume picking up", "Buyers stepping in"])
        else:
            vol_text = random.choice(["Normal volume", "Average volume", "Steady flow", "Nothing unusual"])
        
        # Style 1: Pure analysis, no numbers
        if style == 1:
            openers = [
                f"👀 ${symbol} looking interesting here",
                f"📊 Quick ${symbol} check",
                f"🔍 ${symbol} on my radar",
                f"💭 Thoughts on ${symbol}",
                f"⚡ ${symbol} caught my eye"
            ]
            tweet = f"""{random.choice(openers)}

{trend_text}
{rsi_text}
{vol_text}

Chart speaks for itself 📈

#Crypto #{symbol}"""
        
        # Style 2: Casual observation
        elif style == 2:
            if change >= 15:
                mood = random.choice(["sheesh", "okay then", "alright alright", "damn", "well well well"])
            elif change >= 8:
                mood = random.choice(["nice", "not bad", "looking good", "okay okay", "solid"])
            else:
                mood = random.choice(["hmm", "interesting", "watching this", "curious", "let's see"])
            
            tweet = f"""${symbol}... {mood} 👀

{random.choice([trend_text, rsi_text, vol_text])}

What do you think? 💬

#Crypto #{symbol}"""
        
        # Style 3: Quick stats with analysis
        elif style == 3:
            tweet = f"""${symbol} @ {price_str}

📈 {sign}{change:.1f}% today
📊 {rsi_text}
⚡ {trend_text}

{random.choice(["Your move 🎯", "What's the play? 🤔", "Thoughts? 💭", "Trading this? 👀"])}

#{symbol} #Crypto"""
        
        # Style 4: Just the chart talking
        elif style == 4:
            observations = []
            if above_ema:
                observations.append("Trading above key MAs")
            else:
                observations.append("Below moving averages")
            
            if dist_high < 5:
                observations.append(f"Near 24h high")
            elif dist_low < 5:
                observations.append(f"Near 24h low")
            
            observations.append(rsi_text)
            
            tweet = f"""${symbol} 📊

{chr(10).join(['• ' + o for o in observations[:3]])}

The chart tells the story 👁️

#{symbol} #Trading"""
        
        # Style 5: Price action focus
        elif style == 5:
            if change >= 10:
                action = random.choice(["Ripping higher", "Pushing up", "Breaking out", "Sending it"])
            elif change >= 5:
                action = random.choice(["Moving nicely", "Grinding up", "Steady climb", "Building"])
            else:
                action = random.choice(["Some movement", "Showing signs", "Stirring", "Waking up"])
            
            tweet = f"""{action} on ${symbol} 📈

{price_str}

{random.choice([rsi_text, trend_text])}

#{symbol} #Crypto #Trading"""
        
        # Style 6: Question format
        elif style == 6:
            questions = [
                f"What are we thinking about ${symbol}? 🤔",
                f"${symbol} - bullish or bearish here? 🎯",
                f"Anyone playing ${symbol}? 👀",
                f"${symbol} setup - what's your read? 📊",
                f"How are you trading ${symbol}? 💡"
            ]
            
            tweet = f"""{random.choice(questions)}

{price_str} | {sign}{change:.1f}%

{random.choice([rsi_text, trend_text, vol_text])}

Drop your take 👇

#{symbol}"""
        
        # Style 7: Simple with emoji story
        elif style == 7:
            if change >= 15:
                story = "🚀📈💪"
            elif change >= 8:
                story = "📈✅👀"
            elif change >= 3:
                story = "⬆️💹🔍"
            else:
                story = "👁️📊🎯"
            
            tweet = f"""${symbol} {story}

{random.choice([f"RSI: {rsi:.0f}", trend_text, f"{sign}{change:.1f}%"])}

{random.choice(["NFA", "DYOR", "Chart attached", "What do you see?"])} 👀

#{symbol} #Crypto"""
        
        # Style 8: Technical callout
        else:
            tech_points = []
            if rsi >= 65:
                tech_points.append(f"RSI at {rsi:.0f}")
            if vol_ratio >= 1.5:
                tech_points.append("Volume spiking")
            if above_ema:
                tech_points.append("Above key EMAs")
            if dist_high < 3:
                tech_points.append("Testing resistance")
            elif dist_low < 3:
                tech_points.append("At support")
            
            if not tech_points:
                tech_points = [trend_text, rsi_text]
            
            tweet = f"""${symbol} Technical Check 📊

{chr(10).join(['• ' + p for p in tech_points[:3]])}

{random.choice(["Your analysis?", "Agree?", "What's your view?", "Thoughts?"])} 💬

#{symbol} #Trading"""
        
        return tweet
    
    async def post_featured_coin(self) -> Optional[Dict]:
        """Post featured top gainer with professional chart"""
        try:
            gainers = await self.get_top_gainers_data(15)  # Get more to have options
            if not gainers:
                logger.warning("No gainers data available for featured coin")
                return None
            
            # Pick the best performing coin with good volume AND not posted too much
            featured = None
            for coin in gainers:
                symbol = coin['symbol']
                has_volume = coin.get('volume', 0) >= 5_000_000
                not_overposted = self._check_coin_cooldown(symbol, max_per_day=2)
                
                if has_volume and not_overposted:
                    featured = coin
                    logger.info(f"Selected {symbol} (not overposted, good volume)")
                    break
                elif not not_overposted:
                    logger.info(f"Skipping {symbol} - already posted 2x today")
            
            # Fallback to any coin not overposted
            if not featured:
                for coin in gainers:
                    if self._check_coin_cooldown(coin['symbol'], max_per_day=2):
                        featured = coin
                        break
            
            if not featured:
                logger.warning("All top coins already posted 2x today")
                featured = gainers[0]  # Last resort
            
            symbol = featured['symbol']
            change = featured['change']
            price = featured['price']
            
            logger.info(f"Generating chart for featured coin: {symbol}")
            
            # Generate chart
            from app.services.chart_generator import generate_coin_chart
            chart_bytes = await generate_coin_chart(symbol, change, price)
            
            if chart_bytes:
                logger.info(f"✅ Chart generated: {len(chart_bytes)} bytes")
            else:
                logger.warning(f"⚠️ Chart generation returned None for {symbol}")
            
            # Upload chart if available
            media_id = None
            if chart_bytes:
                logger.info("Uploading chart to Twitter...")
                media_id = await self.upload_media(chart_bytes, f"{symbol} 48h price chart")
                if media_id:
                    logger.info(f"✅ Media uploaded with ID: {media_id}")
                else:
                    logger.warning("⚠️ Media upload returned None")
            
            sign = '+' if change >= 0 else ''
            volume = featured.get('volume', 0)
            vol_str = f"${volume/1e6:.1f}M" if volume < 1e9 else f"${volume/1e9:.1f}B"
            price_str = f"${price:,.4f}" if price < 1 else f"${price:,.2f}"
            
            # Get chart analysis for more interesting posts
            chart_analysis = await self._get_chart_analysis(symbol)
            
            # Generate varied tweet - randomly pick a format style
            tweet_text = await self._generate_varied_featured_tweet(
                symbol, price, price_str, change, sign, vol_str, chart_analysis
            )
            
            if media_id:
                result = await self.post_tweet(tweet_text, media_ids=[media_id])
            else:
                result = await self.post_tweet(tweet_text)
            
            # Record this coin was posted
            if result and result.get('success'):
                self._record_coin_post(symbol)
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to post featured coin: {e}")
            return None
    
    async def post_daily_recap(self) -> Optional[Dict]:
        """Post daily market recap"""
        try:
            market = await self.get_market_summary()
            gainers = await self.get_top_gainers_data(3)
            
            if not market:
                return None
            
            btc_sign = '+' if market['btc_change'] >= 0 else ''
            eth_sign = '+' if market['eth_change'] >= 0 else ''
            
            if market['btc_change'] >= 3:
                day_emoji = "🟢"
                day_texts = ["BULLISH DAY", "GREEN DAY", "BULLS WINNING", "PUMP DAY", "SEND IT DAY", "GAINS DAY"]
                moods = [
                    "Bulls in control 🐂", "Green candles everywhere 💚", "Longs eating good today 🍽️",
                    "What a day to be bullish 🚀", "The bulls showed up today", "Money printer going brrrr",
                    "Portfolios looking healthy 💰", "This is what we like to see", "Up only vibes today",
                    "Bears got cooked today 🔥", "Green dildos for everyone", "Bullish momentum strong"
                ]
            elif market['btc_change'] <= -3:
                day_emoji = "🔴"
                day_texts = ["BEARISH DAY", "RED DAY", "BEARS WINNING", "DUMP DAY", "PAIN DAY", "BLOOD DAY"]
                moods = [
                    "Bears taking over 🐻", "Pain across the board 😬", "Shorts having a field day 📉",
                    "Rough day for hodlers", "Blood in the streets", "Longs got liquidated",
                    "The dump was real 💀", "RIP to those longs", "Bears absolutely feasting",
                    "Down bad today 😵", "Wallets crying today", "Brutal day in crypto"
                ]
            else:
                day_emoji = "⚪"
                day_texts = ["CHOPPY DAY", "SIDEWAYS ACTION", "CONSOLIDATION", "CRAB MARKET", "BORING DAY", "RANGE DAY"]
                moods = [
                    "Sideways action 📊", "Range-bound trading 📈📉", "Waiting for direction 🔍",
                    "Crabbing along 🦀", "Neither bulls nor bears winning", "Patience is key right now",
                    "Calm before the storm?", "Low volatility day", "Market taking a breather",
                    "Just chilling today", "Nothing crazy happening", "Waiting for a catalyst"
                ]
            
            day_text = random.choice(day_texts)
            mood = random.choice(moods)
            
            headers = [
                f"{day_emoji} DAILY RECAP: {day_text}",
                f"{day_emoji} END OF DAY: {day_text}",
                f"{day_emoji} MARKET CLOSE: {day_text}",
                f"{day_emoji} TODAY'S SUMMARY: {day_text}",
                f"{day_emoji} That's a wrap: {day_text}",
                f"{day_emoji} Market update: {day_text}",
                f"{day_emoji} How'd we do? {day_text}",
                f"{day_emoji} Day in review: {day_text}",
                f"{day_emoji} Closing thoughts: {day_text}",
                f"{day_emoji} EOD update: {day_text}"
            ]
            
            tweet_text = f"""{random.choice(headers)}

₿ BTC: ${market['btc_price']:,.0f} ({btc_sign}{market['btc_change']:.1f}%)
⟠ ETH: ${market['eth_price']:,.0f} ({eth_sign}{market['eth_change']:.1f}%)
"""
            if gainers:
                gainer_intros = ["🏆 Top Gainer:", "🥇 Winner:", "📈 Best performer:", "💰 Star of the day:"]
                tweet_text += f"\n{random.choice(gainer_intros)} ${gainers[0]['symbol']} +{gainers[0]['change']:.1f}%"
                if len(gainers) > 1:
                    runner_intros = ["🥈 Runner Up:", "📊 Second place:", "✨ Also pumping:"]
                    tweet_text += f"\n{random.choice(runner_intros)} ${gainers[1]['symbol']} +{gainers[1]['change']:.1f}%"
            
            tweet_text += f"\n\n{mood}"
            
            # Only 15% chance of adding a question CTA
            if random.random() < 0.15:
                ctas = [
                    "\n\n📈 How did YOUR bags perform? 👇",
                    "\n\n💬 Share your wins below 👇",
                    "\n\n💰 Green or red for you? 👇",
                    "\n\n🔥 What are you watching tomorrow? 👇",
                ]
                tweet_text += random.choice(ctas)
            
            tweet_text += "\n\n#Crypto #Bitcoin #DailyRecap"
            
            return await self.post_tweet(tweet_text)
            
        except Exception as e:
            logger.error(f"Failed to post daily recap: {e}")
            return None
    
    def get_status(self) -> Dict:
        """Get current posting status"""
        return {
            'initialized': self.client is not None,
            'posts_today': self.posts_today,
            'max_posts': MAX_POSTS_PER_DAY,
            'remaining': MAX_POSTS_PER_DAY - self.posts_today,
            'last_post': self.last_post_time.isoformat() if self.last_post_time else None,
            'last_reset': self.last_reset.isoformat() if self.last_reset else None
        }


# Global instance
twitter_poster = None


def get_twitter_poster() -> TwitterPoster:
    """Get or create the Twitter poster instance"""
    global twitter_poster
    if twitter_poster is None:
        twitter_poster = TwitterPoster()
    # Reinitialize if client wasn't set (env vars might be available now)
    elif twitter_poster.client is None:
        twitter_poster._initialize_client()
    return twitter_poster


# Multi-account poster cache
_account_posters = {}  # {account_id: TwitterPoster}


class MultiAccountPoster:
    """Handles posting for a specific Twitter account from database"""
    
    def __init__(self, account):
        self.account = account
        self.account_id = account.id
        self.name = account.name
        self.client = None
        self.api_v1 = None
        self._initialize_from_account(account)
    
    def _initialize_from_account(self, account):
        """Initialize Twitter client from database account"""
        from app.utils.encryption import decrypt_api_key
        
        try:
            consumer_key = decrypt_api_key(account.consumer_key)
            consumer_secret = decrypt_api_key(account.consumer_secret)
            access_token = decrypt_api_key(account.access_token)
            access_token_secret = decrypt_api_key(account.access_token_secret)
            bearer_token = decrypt_api_key(account.bearer_token) if account.bearer_token else None
            
            self.client = tweepy.Client(
                consumer_key=consumer_key,
                consumer_secret=consumer_secret,
                access_token=access_token,
                access_token_secret=access_token_secret,
                bearer_token=bearer_token
            )
            
            auth = tweepy.OAuth1UserHandler(
                consumer_key,
                consumer_secret,
                access_token,
                access_token_secret
            )
            self.api_v1 = tweepy.API(auth)
            
            logger.info(f"✅ Initialized Twitter account: {account.name}")
            
        except Exception as e:
            logger.error(f"Failed to initialize Twitter account {account.name}: {e}")
    
    def post_tweet(self, text: str, media_ids: List[str] = None) -> Optional[Dict]:
        """Post tweet using this account"""
        if not self.client:
            return None
        
        try:
            if media_ids:
                response = self.client.create_tweet(text=text, media_ids=media_ids)
            else:
                response = self.client.create_tweet(text=text)
            
            tweet_id = response.data['id']
            logger.info(f"✅ [{self.name}] Tweet posted: {tweet_id}")
            
            return {
                'success': True,
                'tweet_id': tweet_id,
                'account': self.name
            }
            
        except Exception as e:
            logger.error(f"[{self.name}] Twitter error: {e}")
            return {'success': False, 'error': str(e), 'account': self.name}
    
    def upload_media(self, image_bytes: bytes) -> Optional[str]:
        """Upload media for this account"""
        if not self.api_v1:
            return None
        
        try:
            media = self.api_v1.media_upload(filename="chart.png", file=io.BytesIO(image_bytes))
            return str(media.media_id)
        except Exception as e:
            logger.error(f"[{self.name}] Media upload error: {e}")
            return None


def get_account_poster(account) -> MultiAccountPoster:
    """Get or create a poster for a specific account"""
    if account.id not in _account_posters:
        _account_posters[account.id] = MultiAccountPoster(account)
    return _account_posters[account.id]


def clear_account_cache():
    """Clear the account poster cache (call when accounts change)"""
    global _account_posters
    _account_posters = {}


def get_all_twitter_accounts():
    """Get all active Twitter accounts from database"""
    from app.database import SessionLocal
    from app.models import TwitterAccount
    
    db = SessionLocal()
    try:
        accounts = db.query(TwitterAccount).filter(TwitterAccount.is_active == True).all()
        return accounts
    finally:
        db.close()


def get_account_for_post_type(post_type: str):
    """Get the account assigned to a specific post type"""
    from app.database import SessionLocal
    from app.models import TwitterAccount
    
    db = SessionLocal()
    try:
        accounts = db.query(TwitterAccount).filter(TwitterAccount.is_active == True).all()
        
        for account in accounts:
            if post_type in account.get_post_types():
                return account
        
        # If no account assigned, return the first active one (fallback)
        if accounts:
            return accounts[0]
        
        return None
    finally:
        db.close()


def add_twitter_account(name: str, handle: str, consumer_key: str, consumer_secret: str,
                       access_token: str, access_token_secret: str, bearer_token: str = None) -> Dict:
    """Add a new Twitter account to the database"""
    from app.database import SessionLocal
    from app.models import TwitterAccount
    from app.utils.encryption import encrypt_api_key
    
    db = SessionLocal()
    try:
        # Check if name already exists
        existing = db.query(TwitterAccount).filter(TwitterAccount.name == name).first()
        if existing:
            return {'success': False, 'error': f'Account "{name}" already exists'}
        
        account = TwitterAccount(
            name=name,
            handle=handle,
            consumer_key=encrypt_api_key(consumer_key),
            consumer_secret=encrypt_api_key(consumer_secret),
            access_token=encrypt_api_key(access_token),
            access_token_secret=encrypt_api_key(access_token_secret),
            bearer_token=encrypt_api_key(bearer_token) if bearer_token else None
        )
        
        db.add(account)
        db.commit()
        
        clear_account_cache()
        
        return {'success': True, 'account_id': account.id, 'name': name}
        
    except Exception as e:
        db.rollback()
        return {'success': False, 'error': str(e)}
    finally:
        db.close()


def remove_twitter_account(name: str) -> Dict:
    """Remove a Twitter account from the database"""
    from app.database import SessionLocal
    from app.models import TwitterAccount
    
    db = SessionLocal()
    try:
        account = db.query(TwitterAccount).filter(TwitterAccount.name == name).first()
        if not account:
            return {'success': False, 'error': f'Account "{name}" not found'}
        
        db.delete(account)
        db.commit()
        
        clear_account_cache()
        
        return {'success': True, 'name': name}
        
    except Exception as e:
        db.rollback()
        return {'success': False, 'error': str(e)}
    finally:
        db.close()


def assign_post_types(name: str, post_types: List[str]) -> Dict:
    """Assign post types to a Twitter account"""
    from app.database import SessionLocal
    from app.models import TwitterAccount
    
    valid_types = ['featured_coin', 'market_summary', 'top_gainers', 'btc_update', 
                   'altcoin_movers', 'daily_recap', 'top_losers']
    
    # Validate post types
    invalid = [t for t in post_types if t not in valid_types]
    if invalid:
        return {'success': False, 'error': f'Invalid post types: {invalid}. Valid: {valid_types}'}
    
    db = SessionLocal()
    try:
        account = db.query(TwitterAccount).filter(TwitterAccount.name == name).first()
        if not account:
            return {'success': False, 'error': f'Account "{name}" not found'}
        
        account.set_post_types(post_types)
        db.commit()
        
        return {'success': True, 'name': name, 'post_types': post_types}
        
    except Exception as e:
        db.rollback()
        return {'success': False, 'error': str(e)}
    finally:
        db.close()


POST_SCHEDULE = [
    # (hour_utc, minute, post_type)
    (0, 30, 'featured_coin'),      # 12:30 AM - Featured coin with chart
    (2, 0, 'market_summary'),      # 2:00 AM
    (4, 0, 'btc_update'),          # 4:00 AM
    (6, 0, 'featured_coin'),       # 6:00 AM - Featured coin with chart
    (8, 0, 'top_gainers'),         # 8:00 AM - Peak hours start
    (9, 30, 'altcoin_movers'),     # 9:30 AM
    (11, 0, 'featured_coin'),      # 11:00 AM - Featured coin with chart
    (12, 30, 'market_summary'),    # 12:30 PM
    (14, 0, 'top_gainers'),        # 2:00 PM
    (15, 30, 'featured_coin'),     # 3:30 PM - Featured coin with chart
    (17, 0, 'btc_update'),         # 5:00 PM
    (18, 30, 'top_gainers'),       # 6:30 PM - Peak engagement
    (20, 0, 'featured_coin'),      # 8:00 PM - Featured coin with chart
    (21, 30, 'altcoin_movers'),    # 9:30 PM
    (23, 0, 'daily_recap'),        # 11:00 PM - Daily recap
]

POSTED_SLOTS = set()
LAST_POSTED_DAY = None
SLOT_OFFSETS = {}  # Store fixed random offsets per slot per day


def get_twitter_schedule() -> Dict:
    """Get the full posting schedule and next post info"""
    now = datetime.utcnow()
    
    post_type_labels = {
        'featured_coin': '🌟 Featured Coin + Chart',
        'market_summary': '📊 Market Summary',
        'top_gainers': '🚀 Top Gainers',
        'btc_update': '₿ BTC Update',
        'altcoin_movers': '💹 Altcoin Movers',
        'daily_recap': '📈 Daily Recap'
    }
    
    schedule_info = []
    next_post = None
    next_post_time = None
    
    for hour, minute, post_type in POST_SCHEDULE:
        slot_time = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
        
        # If slot is in the past today, it's either done or moves to tomorrow for display
        is_past = slot_time <= now
        
        label = post_type_labels.get(post_type, post_type)
        
        schedule_info.append({
            'time': slot_time if not is_past else slot_time + timedelta(days=1),
            'time_str': slot_time.strftime('%H:%M UTC'),
            'type': label,
            'posted': is_past,
            'hour': hour,
            'minute': minute
        })
        
        # Find next upcoming slot (not in the past)
        if not is_past:
            if next_post_time is None or slot_time < next_post_time:
                next_post = label
                next_post_time = slot_time
    
    # Sort by time
    schedule_info.sort(key=lambda x: (x['hour'], x['minute']))
    
    # Calculate time until next post
    time_until = None
    if next_post_time:
        delta = next_post_time - now
        total_seconds = delta.total_seconds()
        if total_seconds > 0:
            hours = int(total_seconds // 3600)
            minutes = int((total_seconds % 3600) // 60)
            time_until = f"{hours}h {minutes}m" if hours > 0 else f"{minutes}m"
    
    poster = get_twitter_poster()
    
    return {
        'enabled': AUTO_POST_ENABLED,
        'posts_today': poster.posts_today,
        'max_posts': MAX_POSTS_PER_DAY,
        'posts_remaining': MAX_POSTS_PER_DAY - poster.posts_today,
        'next_post_type': next_post,
        'next_post_time': next_post_time.strftime('%H:%M UTC') if next_post_time else None,
        'time_until_next': time_until,
        'last_post': poster.last_post_time.strftime('%H:%M UTC') if poster.last_post_time else None,
        'schedule': schedule_info
    }


async def auto_post_loop():
    """Background loop for automated posting - 15 posts per day per account"""
    global POSTED_SLOTS, LAST_POSTED_DAY, SLOT_OFFSETS
    
    # Check for database accounts first
    db_accounts = get_all_twitter_accounts()
    
    if db_accounts:
        logger.info(f"🐦 Starting Twitter auto-post loop with {len(db_accounts)} database accounts")
    else:
        # Fall back to environment variable poster
        poster = get_twitter_poster()
        if not poster.client:
            logger.error("Twitter not configured - auto posting disabled")
            return
        logger.info("🐦 Starting Twitter auto-post loop (15 posts/day) with env var account")
    
    while True:
        try:
            if not AUTO_POST_ENABLED:
                await asyncio.sleep(60)
                continue
            
            now = datetime.utcnow()
            current_day = now.date()
            
            # Reset posted slots and offsets at midnight
            if LAST_POSTED_DAY is not None and LAST_POSTED_DAY != current_day:
                POSTED_SLOTS.clear()
                SLOT_OFFSETS.clear()
                LAST_POSTED_DAY = current_day
            
            if LAST_POSTED_DAY is None:
                LAST_POSTED_DAY = current_day
            
            # Refresh database accounts periodically
            db_accounts = get_all_twitter_accounts()
            
            # Check each scheduled slot
            for hour, minute, post_type in POST_SCHEDULE:
                slot_key = f"{hour}:{minute}"
                
                # Get or create fixed random offset for this slot (persists for the day)
                if slot_key not in SLOT_OFFSETS:
                    SLOT_OFFSETS[slot_key] = random.randint(-5, 10)
                
                random_offset = SLOT_OFFSETS[slot_key]
                adjusted_minute = minute + random_offset
                adjusted_hour = hour
                
                # Handle minute overflow/underflow
                if adjusted_minute >= 60:
                    adjusted_minute -= 60
                    adjusted_hour = (hour + 1) % 24
                elif adjusted_minute < 0:
                    adjusted_minute += 60
                    adjusted_hour = (hour - 1) % 24
                
                # Check if we're within 3 minutes of the adjusted slot
                slot_time = now.replace(hour=adjusted_hour, minute=adjusted_minute, second=0, microsecond=0)
                time_diff = abs((now - slot_time).total_seconds())
                
                if time_diff <= 180 and slot_key not in POSTED_SLOTS:
                    result = None
                    account_name = "default"
                    
                    # Find the account for this post type (or use fallback)
                    account = get_account_for_post_type(post_type)
                    
                    if account:
                        # Use database account
                        account_poster = get_account_poster(account)
                        account_name = account.name
                        
                        # Generate content from the main poster (for data fetching)
                        main_poster = get_twitter_poster()
                        
                        if post_type == 'featured_coin':
                            result = await post_with_account(account_poster, main_poster, 'featured_coin')
                        elif post_type == 'market_summary':
                            result = await post_with_account(account_poster, main_poster, 'market_summary')
                        elif post_type == 'top_gainers':
                            result = await post_with_account(account_poster, main_poster, 'top_gainers')
                        elif post_type == 'btc_update':
                            result = await post_with_account(account_poster, main_poster, 'btc_update')
                        elif post_type == 'altcoin_movers':
                            result = await post_with_account(account_poster, main_poster, 'altcoin_movers')
                        elif post_type == 'daily_recap':
                            result = await post_with_account(account_poster, main_poster, 'daily_recap')
                    else:
                        # Use fallback env var account
                        poster = get_twitter_poster()
                        if post_type == 'featured_coin':
                            result = await poster.post_featured_coin()
                        elif post_type == 'market_summary':
                            result = await poster.post_market_summary()
                        elif post_type == 'top_gainers':
                            result = await poster.post_top_gainers()
                        elif post_type == 'btc_update':
                            result = await poster.post_btc_update()
                        elif post_type == 'altcoin_movers':
                            result = await poster.post_altcoin_movers()
                        elif post_type == 'daily_recap':
                            result = await poster.post_daily_recap()
                    
                    if result and result.get('success'):
                        logger.info(f"✅ [{account_name}] Auto-posted {post_type} at {slot_key}")
                        POSTED_SLOTS.add(slot_key)
                    else:
                        logger.warning(f"⚠️ [{account_name}] Failed to auto-post {post_type}")
                    
                    break
            
            # Check every 2 minutes
            await asyncio.sleep(120)
            
        except Exception as e:
            logger.error(f"Error in auto-post loop: {e}")
            await asyncio.sleep(60)


async def post_with_account(account_poster: MultiAccountPoster, main_poster, post_type: str) -> Optional[Dict]:
    """Post using a specific account - generates content from main poster, posts with account"""
    try:
        if post_type == 'featured_coin':
            # Get gainers and pick one that's not overposted
            gainers = await main_poster.get_top_gainers_data(15)
            if not gainers:
                return None
            
            # Find a coin that hasn't been posted too much today
            featured = None
            for coin in gainers:
                symbol = coin['symbol']
                has_volume = coin.get('volume', 0) >= 5_000_000
                not_overposted = check_global_coin_cooldown(symbol, max_per_day=2)
                
                if has_volume and not_overposted:
                    featured = coin
                    logger.info(f"[MultiAccount] Selected {symbol} (not overposted, good volume)")
                    break
                elif not not_overposted:
                    logger.info(f"[MultiAccount] Skipping {symbol} - already posted 2x today")
            
            # Fallback to any coin not overposted
            if not featured:
                for coin in gainers:
                    if check_global_coin_cooldown(coin['symbol'], max_per_day=2):
                        featured = coin
                        break
            
            if not featured:
                logger.warning("[MultiAccount] All top coins already posted 2x today")
                return None  # Don't post duplicates
            
            # Generate chart and get analysis
            from app.services.chart_generator import generate_coin_chart
            symbol = featured['symbol']
            change = featured.get('change', 0)
            price = featured.get('price', 0)
            volume = featured.get('volume', 0)
            
            chart_bytes = await generate_coin_chart(symbol, change, price)
            
            sign = '+' if change >= 0 else ''
            price_str = f"${price:,.4f}" if price < 1 else f"${price:,.2f}"
            vol_str = f"${volume/1e6:.1f}M" if volume < 1e9 else f"${volume/1e9:.1f}B"
            
            # Get chart analysis for varied posts
            chart_analysis = await main_poster._get_chart_analysis(symbol)
            
            # Generate varied tweet
            tweet_text = await main_poster._generate_varied_featured_tweet(
                symbol, price, price_str, change, sign, vol_str, chart_analysis
            )
            
            # Upload media and post
            result = None
            if chart_bytes:
                media_id = account_poster.upload_media(chart_bytes)
                if media_id:
                    result = account_poster.post_tweet(tweet_text, media_ids=[media_id])
            
            if not result:
                result = account_poster.post_tweet(tweet_text)
            
            # Record the coin post on success
            if result and result.get('success'):
                record_global_coin_post(symbol)
            
            return result
        
        elif post_type == 'top_gainers':
            gainers = await main_poster.get_top_gainers_data(5)
            if not gainers:
                return None
            
            lines = ["🚀 TOP 5 GAINERS\n"]
            for i, coin in enumerate(gainers, 1):
                emoji = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else "📈"
                change_sign = "+" if coin['change'] >= 0 else ""
                price = coin.get('price', 0)
                price_str = f"${price:,.4f}" if price < 1 else f"${price:,.2f}"
                lines.append(f"{emoji} ${coin['symbol']} {change_sign}{coin['change']:.1f}% @ {price_str}")
            
            lines.append("\n#Crypto #Trading #TopGainers")
            return account_poster.post_tweet("\n".join(lines))
        
        elif post_type == 'market_summary':
            market = await main_poster.get_market_summary()
            if not market:
                return None
            
            btc_sign = "+" if market['btc_change'] >= 0 else ""
            eth_sign = "+" if market['eth_change'] >= 0 else ""
            
            tweet = f"""📊 MARKET UPDATE

₿ BTC: ${market['btc_price']:,.0f} ({btc_sign}{market['btc_change']:.1f}%)
⟠ ETH: ${market['eth_price']:,.0f} ({eth_sign}{market['eth_change']:.1f}%)

#Bitcoin #Ethereum #Crypto"""
            return account_poster.post_tweet(tweet)
        
        elif post_type == 'btc_update':
            market = await main_poster.get_market_summary()
            if not market:
                return None
            
            btc_sign = "+" if market['btc_change'] >= 0 else ""
            emoji = "🟢" if market['btc_change'] >= 0 else "🔴"
            
            tweet = f"""{emoji} BITCOIN UPDATE

₿ ${market['btc_price']:,.0f}
24h: {btc_sign}{market['btc_change']:.1f}%

#Bitcoin #BTC #Crypto"""
            return account_poster.post_tweet(tweet)
        
        elif post_type == 'altcoin_movers':
            gainers = await main_poster.get_top_gainers_data(5)
            if not gainers:
                return None
            
            alts = [g for g in gainers if g['symbol'] not in ['BTC', 'ETH']][:3]
            if not alts:
                return None
            
            lines = ["💹 ALTCOIN MOVERS\n"]
            for coin in alts:
                sign = "+" if coin['change'] >= 0 else ""
                lines.append(f"• ${coin['symbol']} {sign}{coin['change']:.1f}%")
            
            lines.append("\n#Altcoins #Crypto #Trading")
            return account_poster.post_tweet("\n".join(lines))
        
        elif post_type == 'daily_recap':
            market = await main_poster.get_market_summary()
            gainers = await main_poster.get_top_gainers_data(3)
            
            if not market:
                return None
            
            btc_sign = "+" if market['btc_change'] >= 0 else ""
            eth_sign = "+" if market['eth_change'] >= 0 else ""
            
            tweet = f"""📈 DAILY RECAP

₿ BTC: ${market['btc_price']:,.0f} ({btc_sign}{market['btc_change']:.1f}%)
⟠ ETH: ${market['eth_price']:,.0f} ({eth_sign}{market['eth_change']:.1f}%)"""
            
            if gainers:
                tweet += f"\n\n🏆 Top: ${gainers[0]['symbol']} +{gainers[0]['change']:.1f}%"
            
            tweet += "\n\n#Crypto #Bitcoin #DailyRecap"
            return account_poster.post_tweet(tweet)
        
        return None
        
    except Exception as e:
        logger.error(f"Error posting with account: {e}")
        return None
