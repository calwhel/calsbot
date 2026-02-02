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

# Telegram bot token for notifications
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")

async def notify_admin_post_result(account_name: str, post_type: str, success: bool, error: str = None):
    """Send Telegram notification to admins about post result"""
    try:
        if not TELEGRAM_BOT_TOKEN:
            return
        
        from app.database import SessionLocal
        from app.models import User
        import httpx
        
        db = SessionLocal()
        try:
            admins = db.query(User).filter(User.is_admin == True).all()
            
            if success:
                msg = f"✅ <b>Twitter Post Success</b>\n\n📱 Account: {account_name}\n📝 Type: {post_type}\n⏰ {datetime.utcnow().strftime('%H:%M UTC')}"
            else:
                msg = f"❌ <b>Twitter Post FAILED</b>\n\n📱 Account: {account_name}\n📝 Type: {post_type}\n⚠️ Error: {error[:200] if error else 'Unknown'}\n⏰ {datetime.utcnow().strftime('%H:%M UTC')}"
            
            async with httpx.AsyncClient() as client:
                for admin in admins:
                    try:
                        await client.post(
                            f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
                            json={
                                "chat_id": int(admin.telegram_id),
                                "text": msg,
                                "parse_mode": "HTML"
                            },
                            timeout=10
                        )
                    except Exception as e:
                        logger.error(f"Failed to notify admin {admin.telegram_id}: {e}")
        finally:
            db.close()
            
    except Exception as e:
        logger.error(f"Error sending post notification: {e}")


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
                not_overposted = self._check_coin_cooldown(symbol, max_per_day=1)
                
                if has_volume and not_overposted:
                    featured = coin
                    logger.info(f"Selected {symbol} (not overposted, good volume)")
                    break
                elif not not_overposted:
                    logger.info(f"Skipping {symbol} - already posted 2x today")
            
            # Fallback to any coin not overposted
            if not featured:
                for coin in gainers:
                    if self._check_coin_cooldown(coin['symbol'], max_per_day=1):
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
        # Expunge all to keep them accessible after session closes
        for account in accounts:
            db.expunge(account)
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
        
        selected_account = None
        for account in accounts:
            if post_type in account.get_post_types():
                selected_account = account
                break
        
        # If no account assigned, return the first active one (fallback)
        if not selected_account and accounts:
            selected_account = accounts[0]
        
        # Expunge to keep object accessible after session closes
        if selected_account:
            db.expunge(selected_account)
        
        return selected_account
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


def toggle_account_active(name: str, active: bool) -> Dict:
    """Enable or disable auto-posting for a Twitter account"""
    from app.database import SessionLocal
    from app.models import TwitterAccount
    
    db = SessionLocal()
    try:
        account = db.query(TwitterAccount).filter(TwitterAccount.name == name).first()
        if not account:
            return {'success': False, 'error': f'Account "{name}" not found'}
        
        account.is_active = active
        db.commit()
        
        status = "enabled" if active else "disabled"
        logger.info(f"Twitter account {name} auto-posting {status}")
        return {'success': True, 'name': name, 'is_active': active}
        
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
    # (hour_utc, minute, post_type) - 20 posts per day for more engagement
    (0, 30, 'featured_coin'),      # 12:30 AM - Featured coin with chart
    (1, 45, 'early_gainer'),       # 1:45 AM - Early mover
    (3, 0, 'market_summary'),      # 3:00 AM
    (4, 30, 'whale_alert'),        # 4:30 AM - Whale activity
    (6, 0, 'featured_coin'),       # 6:00 AM - Featured coin with chart
    (7, 30, 'early_gainer'),       # 7:30 AM - Early mover
    (8, 30, 'top_gainers'),        # 8:30 AM - Peak hours start
    (9, 45, 'quick_ta'),           # 9:45 AM - Technical analysis
    (11, 0, 'featured_coin'),      # 11:00 AM - Featured coin with chart
    (12, 15, 'funding_extreme'),   # 12:15 PM - Funding rates
    (13, 30, 'early_gainer'),      # 1:30 PM - Early mover
    (14, 45, 'whale_alert'),       # 2:45 PM - Whale activity
    (15, 30, 'featured_coin'),     # 3:30 PM - Featured coin with chart
    (16, 45, 'quick_ta'),          # 4:45 PM - Technical analysis
    (17, 30, 'top_gainers'),       # 5:30 PM - Top gainers
    (18, 45, 'early_gainer'),      # 6:45 PM - Peak engagement
    (20, 0, 'featured_coin'),      # 8:00 PM - Featured coin with chart
    (21, 15, 'whale_alert'),       # 9:15 PM - Whale activity
    (22, 30, 'quick_ta'),          # 10:30 PM - Technical analysis
    (23, 30, 'daily_recap'),       # 11:30 PM - Daily recap
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
        'daily_recap': '📈 Daily Recap',
        'early_gainer': '🎯 Early Mover',
        'whale_alert': '🐋 Whale Alert',
        'quick_ta': '📊 Quick TA',
        'funding_extreme': '⚠️ Funding Alert'
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
    """Background loop for automated posting - 20 posts per day per account"""
    global POSTED_SLOTS, LAST_POSTED_DAY, SLOT_OFFSETS
    
    logger.info("=" * 50)
    logger.info("🐦 AUTO-POST LOOP INITIALIZING...")
    logger.info("=" * 50)
    
    # Wait a bit for database to be ready
    await asyncio.sleep(5)
    
    try:
        # Check for database accounts first
        db_accounts = get_all_twitter_accounts()
        
        if db_accounts:
            logger.info(f"🐦 AUTO-POST: Found {len(db_accounts)} database accounts")
            for acc in db_accounts:
                logger.info(f"  - Account: {acc.name} (@{acc.handle}), active: {acc.is_active}")
        else:
            logger.warning("🐦 AUTO-POST: No database accounts found, checking env vars...")
            # Fall back to environment variable poster
            poster = get_twitter_poster()
            if not poster.client:
                logger.error("❌ Twitter not configured - no accounts in DB and no env vars")
                logger.error("❌ Auto posting disabled - add accounts via /twitter command")
                return
            logger.info("🐦 AUTO-POST: Using env var account as fallback")
        
        logger.info(f"🐦 AUTO-POST: Schedule has {len(POST_SCHEDULE)} slots per day")
        logger.info("🐦 AUTO-POST LOOP STARTED SUCCESSFULLY!")
    except Exception as e:
        logger.error(f"❌ AUTO-POST INIT ERROR: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return
    
    loop_count = 0
    while True:
        try:
            loop_count += 1
            
            # Log every 30 iterations (~1 hour) to show loop is alive
            if loop_count % 30 == 1:
                logger.info(f"🐦 AUTO-POST: Loop check #{loop_count} at {datetime.utcnow().strftime('%H:%M:%S')} UTC")
            
            if not AUTO_POST_ENABLED:
                if loop_count % 30 == 1:
                    logger.info("🐦 AUTO-POST: Disabled, waiting...")
                await asyncio.sleep(60)
                continue
            
            now = datetime.utcnow()
            current_day = now.date()
            
            # Reset posted slots and offsets at midnight
            if LAST_POSTED_DAY is not None and LAST_POSTED_DAY != current_day:
                logger.info("🐦 AUTO-POST: Midnight reset - clearing posted slots")
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
                # ±15 minutes random offset for natural posting
                if slot_key not in SLOT_OFFSETS:
                    SLOT_OFFSETS[slot_key] = random.randint(-15, 15)
                
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
                        
                        # All post types go through post_with_account which handles routing
                        result = await post_with_account(account_poster, main_poster, post_type)
                    else:
                        # Use fallback env var account
                        poster = get_twitter_poster()
                        main_poster = poster
                        # Create a fallback account poster using env credentials
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
                        elif post_type == 'early_gainer':
                            # For fallback, use featured coin logic
                            result = await poster.post_featured_coin()
                        elif post_type == 'whale_alert':
                            result = await poster.post_top_gainers()
                        elif post_type == 'quick_ta':
                            result = await poster.post_featured_coin()
                        elif post_type == 'funding_extreme':
                            result = await poster.post_market_summary()
                    
                    if result and result.get('success'):
                        logger.info(f"✅ [{account_name}] Auto-posted {post_type} at {slot_key}")
                        POSTED_SLOTS.add(slot_key)
                        # Notify admin of success
                        await notify_admin_post_result(account_name, post_type, True)
                    else:
                        error_msg = result.get('error', 'Unknown error') if result else 'No result returned'
                        logger.warning(f"⚠️ [{account_name}] Failed to auto-post {post_type}: {error_msg}")
                        # Notify admin of failure
                        await notify_admin_post_result(account_name, post_type, False, error_msg)
                    
                    break
            
            # Check every 2 minutes
            await asyncio.sleep(120)
            
        except Exception as e:
            logger.error(f"Error in auto-post loop: {e}")
            # Notify admin of exception
            await notify_admin_post_result("system", "auto_post_loop", False, str(e))
            await asyncio.sleep(60)


async def post_with_account(account_poster: MultiAccountPoster, main_poster, post_type: str) -> Optional[Dict]:
    """Post using a specific account - generates content from main poster, posts with account"""
    try:
        # Check if this is the Crypto Social account - use LunarCrush posts
        if is_social_account(account_poster.name):
            logger.info(f"[CryptoSocial] Using LunarCrush-powered posts for {account_poster.name}")
            return await post_for_social_account(account_poster, post_type)
        
        if post_type == 'featured_coin':
            # Get gainers and pick one randomly that's not overposted
            gainers = await main_poster.get_top_gainers_data(20)
            if not gainers:
                return None
            
            # Collect ALL valid coins first, then pick randomly
            # Max 1 post per coin per day for maximum variety
            valid_coins = []
            for coin in gainers:
                symbol = coin['symbol']
                has_volume = coin.get('volume', 0) >= 5_000_000
                not_overposted = check_global_coin_cooldown(symbol, max_per_day=1)
                
                if has_volume and not_overposted:
                    valid_coins.append(coin)
                elif not not_overposted:
                    logger.info(f"[MultiAccount] Skipping {symbol} - already posted today (1x max)")
            
            # Pick RANDOMLY from valid coins (not first one!)
            if valid_coins:
                featured = random.choice(valid_coins)
                logger.info(f"[MultiAccount] Randomly selected {featured['symbol']} from {len(valid_coins)} valid coins")
            else:
                # Fallback: collect any coins not overposted (allow 2 if we run out)
                fallback_coins = [c for c in gainers if check_global_coin_cooldown(c['symbol'], max_per_day=1)]
                if fallback_coins:
                    featured = random.choice(fallback_coins)
                    logger.info(f"[MultiAccount] Fallback: randomly selected {featured['symbol']}")
                else:
                    logger.warning("[MultiAccount] All top coins already posted 2x today")
                    return None
            
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
            
            style = random.randint(1, 6)
            if style == 1:
                lines = ["Today's big movers catching my attention:\n"]
            elif style == 2:
                lines = ["These coins are ripping today:\n"]
            elif style == 3:
                lines = ["Here's what's running right now:\n"]
            elif style == 4:
                lines = ["The market's showing some life:\n"]
            elif style == 5:
                lines = ["Worth watching today:\n"]
            else:
                lines = ["Scanning the top movers:\n"]
            
            for i, coin in enumerate(gainers, 1):
                emoji = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else "•"
                change_sign = "+" if coin['change'] >= 0 else ""
                price = coin.get('price', 0)
                price_str = f"${price:,.4f}" if price < 1 else f"${price:,.2f}"
                lines.append(f"{emoji} ${coin['symbol']} {change_sign}{coin['change']:.1f}% @ {price_str}")
            
            closings = [
                "\n\nAlways do your own research",
                "\n\nKeeping these on my radar",
                "\n\nVolume looking healthy across the board",
                "\n\nInteresting setups forming",
                "\n\nMomentum is real today",
                ""
            ]
            lines.append(random.choice(closings))
            return account_poster.post_tweet("\n".join(lines))
        
        elif post_type == 'market_summary':
            market = await main_poster.get_market_summary()
            if not market:
                return None
            
            btc_sign = "+" if market['btc_change'] >= 0 else ""
            eth_sign = "+" if market['eth_change'] >= 0 else ""
            
            # Mood based on BTC
            if market['btc_change'] >= 3:
                mood = random.choice(["Bulls are eating today", "Green across the board", "Risk on mode"])
            elif market['btc_change'] >= 0:
                mood = random.choice(["Quiet day so far", "Steady grind", "Holding up well"])
            elif market['btc_change'] >= -3:
                mood = random.choice(["Little pullback happening", "Some red on the screens", "Bears testing support"])
            else:
                mood = random.choice(["Tough day in the markets", "Bears in control", "Blood in the streets"])
            
            style = random.randint(1, 5)
            if style == 1:
                tweet = f"""Quick market check:

BTC: ${market['btc_price']:,.0f} ({btc_sign}{market['btc_change']:.1f}%)
ETH: ${market['eth_price']:,.0f} ({eth_sign}{market['eth_change']:.1f}%)

{mood}"""
            elif style == 2:
                tweet = f"""{mood}

BTC sitting at ${market['btc_price']:,.0f} ({btc_sign}{market['btc_change']:.1f}%)
ETH at ${market['eth_price']:,.0f} ({eth_sign}{market['eth_change']:.1f}%)

Watching closely from here"""
            elif style == 3:
                tweet = f"""Where we're at right now:

BTC: ${market['btc_price']:,.0f}
ETH: ${market['eth_price']:,.0f}

{btc_sign}{market['btc_change']:.1f}% and {eth_sign}{market['eth_change']:.1f}% respectively

{mood}"""
            elif style == 4:
                tweet = f"""Market update

Bitcoin: ${market['btc_price']:,.0f} | {btc_sign}{market['btc_change']:.1f}%
Ethereum: ${market['eth_price']:,.0f} | {eth_sign}{market['eth_change']:.1f}%

{mood}"""
            else:
                tweet = f"""Checking in on the markets

BTC {btc_sign}{market['btc_change']:.1f}% at ${market['btc_price']:,.0f}
ETH {eth_sign}{market['eth_change']:.1f}% at ${market['eth_price']:,.0f}

{mood}"""
            return account_poster.post_tweet(tweet)
        
        elif post_type == 'btc_update':
            market = await main_poster.get_market_summary()
            if not market:
                return None
            
            btc_sign = "+" if market['btc_change'] >= 0 else ""
            
            # Contextual commentary
            if market['btc_change'] >= 5:
                comment = random.choice(["Bitcoin is absolutely ripping", "Major move happening", "Bulls are in control"])
            elif market['btc_change'] >= 2:
                comment = random.choice(["Solid day for Bitcoin", "Steady climb today", "Looking healthy"])
            elif market['btc_change'] >= 0:
                comment = random.choice(["Quiet consolidation", "Holding steady", "Calm before the storm?"])
            elif market['btc_change'] >= -3:
                comment = random.choice(["Small pullback", "Testing support", "Nothing to panic about"])
            else:
                comment = random.choice(["Rough day for BTC", "Bears winning today", "Finding a bottom?"])
            
            style = random.randint(1, 5)
            if style == 1:
                tweet = f"""Bitcoin at ${market['btc_price']:,.0f}

{btc_sign}{market['btc_change']:.1f}% on the day

{comment}"""
            elif style == 2:
                tweet = f"""{comment}

BTC: ${market['btc_price']:,.0f} ({btc_sign}{market['btc_change']:.1f}%)

Always interesting to see where we go from here"""
            elif style == 3:
                tweet = f"""BTC update

${market['btc_price']:,.0f}
24h: {btc_sign}{market['btc_change']:.1f}%

{comment}"""
            elif style == 4:
                tweet = f"""Checking in on Bitcoin

Trading at ${market['btc_price']:,.0f}
{btc_sign}{market['btc_change']:.1f}% today

{comment}"""
            else:
                tweet = f"""Bitcoin sitting at ${market['btc_price']:,.0f} right now

{btc_sign}{market['btc_change']:.1f}%

{comment}"""
            return account_poster.post_tweet(tweet)
        
        elif post_type == 'altcoin_movers':
            gainers = await main_poster.get_top_gainers_data(8)
            if not gainers:
                return None
            
            alts = [g for g in gainers if g['symbol'] not in ['BTC', 'ETH']][:4]
            if not alts:
                return None
            
            style = random.randint(1, 5)
            if style == 1:
                lines = ["Altcoins making moves today:\n"]
            elif style == 2:
                lines = ["While BTC chops, alts are running:\n"]
            elif style == 3:
                lines = ["Interesting action in the altcoin market:\n"]
            elif style == 4:
                lines = ["Some alts catching my eye:\n"]
            else:
                lines = ["Alt season vibes:\n"]
            
            for coin in alts:
                sign = "+" if coin['change'] >= 0 else ""
                price = coin.get('price', 0)
                price_str = f"${price:,.4f}" if price < 1 else f"${price:,.2f}"
                vol = coin.get('volume', 0)
                vol_str = f"${vol/1e6:.0f}M" if vol >= 1e6 else ""
                lines.append(f"${coin['symbol']} {sign}{coin['change']:.1f}% @ {price_str}" + (f" ({vol_str} vol)" if vol_str else ""))
            
            closings = [
                "\n\nVolume looking solid",
                "\n\nWorth keeping on the watchlist",
                "\n\nMomentum is building",
                "\n\nDYOR as always",
                ""
            ]
            lines.append(random.choice(closings))
            return account_poster.post_tweet("\n".join(lines))
        
        elif post_type == 'early_gainer':
            # Early gainer post for standard accounts (same as social but branded)
            return await post_early_gainer_standard(account_poster, main_poster)
        
        elif post_type == 'whale_alert':
            # Whale alert - coins with unusual volume
            return await post_whale_alert(account_poster, main_poster)
        
        elif post_type == 'funding_extreme':
            # Funding rate extremes - potential liquidation zones
            return await post_funding_extreme(account_poster)
        
        elif post_type == 'quick_ta':
            # Quick TA setup
            return await post_quick_ta(account_poster, main_poster)
        
        elif post_type == 'daily_recap':
            market = await main_poster.get_market_summary()
            gainers = await main_poster.get_top_gainers_data(3)
            
            if not market:
                return None
            
            btc_sign = "+" if market['btc_change'] >= 0 else ""
            eth_sign = "+" if market['eth_change'] >= 0 else ""
            
            # Summary of the day's vibe
            if market['btc_change'] >= 3:
                vibe = random.choice(["Great day for the bulls", "Solid green day", "The bulls ate well today"])
            elif market['btc_change'] >= 0:
                vibe = random.choice(["Quiet day overall", "Choppy but we closed green", "Nothing crazy but we'll take it"])
            elif market['btc_change'] >= -3:
                vibe = random.choice(["Bit of red today", "Bears took a nibble", "Small pullback, nothing major"])
            else:
                vibe = random.choice(["Rough day out there", "Bears won today", "Time to zoom out"])
            
            style = random.randint(1, 5)
            if style == 1:
                tweet = f"""That's a wrap on today's trading

BTC: ${market['btc_price']:,.0f} ({btc_sign}{market['btc_change']:.1f}%)
ETH: ${market['eth_price']:,.0f} ({eth_sign}{market['eth_change']:.1f}%)

{vibe}"""
            elif style == 2:
                tweet = f"""End of day check in:

Bitcoin at ${market['btc_price']:,.0f}
Ethereum at ${market['eth_price']:,.0f}

{vibe}"""
            elif style == 3:
                tweet = f"""Closing out the day

BTC {btc_sign}{market['btc_change']:.1f}% | ETH {eth_sign}{market['eth_change']:.1f}%

{vibe}

Tomorrow's another day"""
            elif style == 4:
                tweet = f"""How'd we do today?

BTC: ${market['btc_price']:,.0f} ({btc_sign}{market['btc_change']:.1f}%)
ETH: ${market['eth_price']:,.0f} ({eth_sign}{market['eth_change']:.1f}%)

{vibe}"""
            else:
                tweet = f"""Daily wrap up

{vibe}

BTC {btc_sign}{market['btc_change']:.1f}% @ ${market['btc_price']:,.0f}
ETH {eth_sign}{market['eth_change']:.1f}% @ ${market['eth_price']:,.0f}"""
            
            if gainers and random.random() < 0.6:
                top = gainers[0]
                sign = "+" if top['change'] >= 0 else ""
                tweet += f"\n\nToday's biggest mover: ${top['symbol']} {sign}{top['change']:.1f}%"
            
            return account_poster.post_tweet(tweet)
        
        return None
        
    except Exception as e:
        logger.error(f"Error posting with account: {e}")
        return None


# ============== CRYPTO SOCIAL ACCOUNT - NEWS & EARLY GAINERS ==============

async def post_social_news(account_poster: MultiAccountPoster) -> Optional[Dict]:
    """Post breaking crypto news - for Crypto Social account"""
    try:
        from app.services.news_monitor import NewsMonitor
        
        news_monitor = NewsMonitor()
        articles = await news_monitor.fetch_recent_news(items=20, date_filter="last24hours")
        
        if not articles:
            logger.info("No recent news for social post")
            return None
        
        # Pick a random article
        article = random.choice(articles[:10])
        title = article.get('title', '')[:200]
        source = article.get('source_name', 'Crypto News')
        sentiment = article.get('sentiment', 'neutral')
        tickers = article.get('tickers', [])
        
        # Get main coin mentioned
        main_coin = tickers[0] if tickers else None
        
        style = random.randint(1, 5)
        
        if style == 1:
            # Breaking news style
            emoji = "🚨" if 'hack' in title.lower() or 'crash' in title.lower() else "📰"
            tweet_text = f"""{emoji} BREAKING

{title}

Source: {source}
{f'$' + main_coin if main_coin else ''}

#CryptoNews #Breaking"""
        
        elif style == 2:
            # Quick headline
            sentiment_emoji = "🟢" if sentiment == 'positive' else "🔴" if sentiment == 'negative' else "⚪"
            tweet_text = f"""{sentiment_emoji} {title}

{f'📊 ${main_coin}' if main_coin else ''}

#CryptoNews"""
        
        elif style == 3:
            # Casual share
            intros = [
                "Interesting development 👀",
                "This just in",
                "Worth watching",
                "News alert",
                "Heads up"
            ]
            tweet_text = f"""{random.choice(intros)}

{title}

#Crypto #News"""
        
        elif style == 4:
            # Coin-focused (if ticker available)
            if main_coin:
                tweet_text = f"""${main_coin} in the news 📰

{title}

#Crypto #{main_coin}"""
            else:
                tweet_text = f"""📰 {title}

#CryptoNews"""
        
        else:
            # Simple share
            tweet_text = f"""{title}

Via {source}

#Crypto #News"""
        
        return account_poster.post_tweet(tweet_text)
        
    except Exception as e:
        logger.error(f"Error posting social news: {e}")
        return None


async def post_early_gainers(account_poster: MultiAccountPoster) -> Optional[Dict]:
    """Post coins gaining traction early - before they pump big"""
    try:
        exchange = ccxt.binance({'enableRateLimit': True})
        tickers = await exchange.fetch_tickers()
        await exchange.close()
        
        # Find coins in the "sweet spot" - gaining but not yet massive
        early_movers = []
        for symbol, data in tickers.items():
            if not symbol.endswith('/USDT') or not data.get('percentage'):
                continue
            
            base = symbol.replace('/USDT', '')
            if base in ['USDC', 'BUSD', 'DAI', 'TUSD', 'USDP', 'USDD']:
                continue
            
            change = data['percentage']
            volume = data.get('quoteVolume', 0)
            
            # Sweet spot: 3-12% gain with decent volume (gaining traction, not yet FOMO)
            # Also check global cooldown to avoid posting same coins as other accounts
            if 3 <= change <= 12 and volume >= 5_000_000 and check_global_coin_cooldown(base, max_per_day=1):
                early_movers.append({
                    'symbol': base,
                    'change': change,
                    'volume': volume,
                    'price': data.get('last', 0)
                })
        
        if not early_movers:
            return None
        
        # Sort by change (ascending) - we want the ones just starting to move
        early_movers.sort(key=lambda x: x['change'])
        
        style = random.randint(1, 5)
        
        # Pick the coin we'll feature (for recording)
        featured_coin = early_movers[0]
        
        if style == 1:
            # Early alert
            coin = featured_coin
            vol_str = f"${coin['volume']/1e6:.1f}M"
            tweet_text = f"""👀 EARLY MOVER ALERT

${coin['symbol']} +{coin['change']:.1f}%

Starting to build momentum
Volume: {vol_str}

Watching this one closely

#Crypto #EarlyGainer"""
        
        elif style == 2:
            # List of early movers
            lines = ["🌱 COINS GAINING TRACTION\n"]
            for coin in early_movers[:4]:
                emoji = "🔹" if coin['change'] < 6 else "📈"
                lines.append(f"{emoji} ${coin['symbol']} +{coin['change']:.1f}%")
            lines.append("\nEarly stages - watching closely")
            lines.append("\n#Crypto #Altcoins")
            tweet_text = "\n".join(lines)
        
        elif style == 3:
            # Casual observation
            coin = early_movers[0]
            casuals = [
                f"${coin['symbol']} quietly up {coin['change']:.1f}% - something brewing?",
                f"Noticing ${coin['symbol']} starting to move (+{coin['change']:.1f}%)",
                f"${coin['symbol']} gaining steam. +{coin['change']:.1f}% so far",
                f"Eyes on ${coin['symbol']} - up {coin['change']:.1f}% and building"
            ]
            tweet_text = f"{random.choice(casuals)}\n\n#Crypto"
        
        elif style == 4:
            # Volume focus
            coin = early_movers[0]
            vol_str = f"${coin['volume']/1e6:.1f}M" if coin['volume'] < 1e9 else f"${coin['volume']/1e9:.1f}B"
            tweet_text = f"""📊 VOLUME BUILDING

${coin['symbol']}
📈 +{coin['change']:.1f}%
💰 {vol_str} volume

Early signs of momentum

#Crypto #Trading"""
        
        else:
            # Under the radar
            coin = random.choice(early_movers[:3])
            tweet_text = f"""👁️ UNDER THE RADAR

${coin['symbol']} quietly gaining +{coin['change']:.1f}%

Not making headlines yet...

#Crypto"""
        
        result = account_poster.post_tweet(tweet_text)
        
        # Record the featured coin to prevent other accounts posting same ticker
        if result and result.get('success') and featured_coin:
            record_global_coin_post(featured_coin['symbol'])
            logger.info(f"[CryptoSocial] Recorded {featured_coin['symbol']} to global cooldown")
        
        return result
        
    except Exception as e:
        logger.error(f"Error posting early gainers: {e}")
        return None


async def post_momentum_shift(account_poster: MultiAccountPoster) -> Optional[Dict]:
    """Post coins showing momentum shifts - starting to move after consolidation"""
    try:
        exchange = ccxt.binance({'enableRateLimit': True})
        tickers = await exchange.fetch_tickers()
        await exchange.close()
        
        # Find strong movers (check global cooldown to avoid same tickers as other accounts)
        movers = []
        for symbol, data in tickers.items():
            if not symbol.endswith('/USDT') or not data.get('percentage'):
                continue
            
            base = symbol.replace('/USDT', '')
            if base in ['USDC', 'BUSD', 'DAI', 'TUSD', 'USDP']:
                continue
            
            change = data['percentage']
            volume = data.get('quoteVolume', 0)
            
            if change >= 5 and volume >= 10_000_000 and check_global_coin_cooldown(base, max_per_day=1):
                movers.append({
                    'symbol': base,
                    'change': change,
                    'volume': volume,
                    'price': data.get('last', 0),
                    'high': data.get('high', 0),
                    'low': data.get('low', 0)
                })
        
        if not movers:
            return None
        
        movers.sort(key=lambda x: x['change'], reverse=True)
        featured_coin = movers[0]  # Track the main coin for cooldown
        
        style = random.randint(1, 4)
        
        if style == 1:
            # Momentum alert
            coin = movers[0]
            range_size = coin['high'] - coin['low'] if coin['high'] > coin['low'] else 1
            position = (coin['price'] - coin['low']) / range_size * 100
            
            tweet_text = f"""🚀 MOMENTUM DETECTED

${coin['symbol']} +{coin['change']:.1f}%

{'Near highs' if position > 80 else 'Mid-range' if position > 40 else 'Breaking out from lows'}

#Crypto #Momentum"""
        
        elif style == 2:
            # Top movers
            lines = ["📈 TOP MOMENTUM TODAY\n"]
            for coin in movers[:3]:
                vol_str = f"${coin['volume']/1e6:.0f}M"
                lines.append(f"🔥 ${coin['symbol']} +{coin['change']:.1f}% ({vol_str})")
            lines.append("\n#Crypto #Altcoins")
            tweet_text = "\n".join(lines)
        
        elif style == 3:
            # Single focus
            coin = movers[0]
            vol_str = f"${coin['volume']/1e6:.1f}M" if coin['volume'] < 1e9 else f"${coin['volume']/1e9:.1f}B"
            tweet_text = f"""${coin['symbol']} is MOVING 🔥

+{coin['change']:.1f}%
{vol_str} volume

Strong momentum today

#Crypto #{coin['symbol']}"""
        
        else:
            # Casual
            coin = movers[0]
            casuals = [
                f"${coin['symbol']} up {coin['change']:.1f}% - momentum is real",
                f"Strong move on ${coin['symbol']} (+{coin['change']:.1f}%)",
                f"${coin['symbol']} gaining serious traction today"
            ]
            tweet_text = f"{random.choice(casuals)}\n\n#Crypto"
        
        result = account_poster.post_tweet(tweet_text)
        
        # Record the featured coin to prevent other accounts posting same ticker
        if result and result.get('success') and featured_coin:
            record_global_coin_post(featured_coin['symbol'])
            logger.info(f"[CryptoSocial] Recorded {featured_coin['symbol']} to global cooldown")
        
        return result
        
    except Exception as e:
        logger.error(f"Error posting momentum shift: {e}")
        return None


async def post_volume_surge(account_poster: MultiAccountPoster) -> Optional[Dict]:
    """Post coins with unusual volume - often precedes big moves"""
    try:
        exchange = ccxt.binance({'enableRateLimit': True})
        tickers = await exchange.fetch_tickers()
        await exchange.close()
        
        # Find high volume coins (check global cooldown)
        high_volume = []
        for symbol, data in tickers.items():
            if not symbol.endswith('/USDT'):
                continue
            
            base = symbol.replace('/USDT', '')
            if base in ['USDC', 'BUSD', 'DAI', 'TUSD', 'BTC', 'ETH']:
                continue
            
            volume = data.get('quoteVolume', 0)
            change = data.get('percentage', 0) or 0
            
            # High volume altcoins (check cooldown to avoid same tickers)
            if volume >= 50_000_000 and check_global_coin_cooldown(base, max_per_day=1):
                high_volume.append({
                    'symbol': base,
                    'change': change,
                    'volume': volume,
                    'price': data.get('last', 0)
                })
        
        if not high_volume:
            return None
        
        high_volume.sort(key=lambda x: x['volume'], reverse=True)
        featured_coin = high_volume[0]  # Track the main coin for cooldown
        
        style = random.randint(1, 4)
        
        if style == 1:
            # Volume leader
            coin = high_volume[0]
            vol_str = f"${coin['volume']/1e6:.0f}M" if coin['volume'] < 1e9 else f"${coin['volume']/1e9:.1f}B"
            sign = "+" if coin['change'] >= 0 else ""
            
            tweet_text = f"""💰 MASSIVE VOLUME

${coin['symbol']}
Volume: {vol_str}
Price: {sign}{coin['change']:.1f}%

Big money moving

#Crypto #Volume"""
        
        elif style == 2:
            # Volume leaders list
            lines = ["📊 HIGHEST VOLUME ALTS\n"]
            for coin in high_volume[:4]:
                vol_str = f"${coin['volume']/1e6:.0f}M" if coin['volume'] < 1e9 else f"${coin['volume']/1e9:.1f}B"
                emoji = "🟢" if coin['change'] >= 0 else "🔴"
                lines.append(f"{emoji} ${coin['symbol']} - {vol_str}")
            lines.append("\n#Crypto #Trading")
            tweet_text = "\n".join(lines)
        
        elif style == 3:
            # Observation
            coin = high_volume[0]
            vol_str = f"${coin['volume']/1e6:.0f}M" if coin['volume'] < 1e9 else f"${coin['volume']/1e9:.1f}B"
            
            observations = [
                f"${coin['symbol']} doing {vol_str} in volume today. Something's happening",
                f"Serious volume on ${coin['symbol']} ({vol_str}). Worth watching",
                f"${coin['symbol']} volume is massive today - {vol_str}"
            ]
            tweet_text = f"{random.choice(observations)}\n\n#Crypto"
        
        else:
            # Simple
            coin = high_volume[0]
            vol_str = f"${coin['volume']/1e9:.1f}B" if coin['volume'] >= 1e9 else f"${coin['volume']/1e6:.0f}M"
            tweet_text = f"👀 ${coin['symbol']} - {vol_str} volume\n\nSomething brewing\n\n#Crypto"
        
        result = account_poster.post_tweet(tweet_text)
        
        # Record the featured coin to prevent other accounts posting same ticker
        if result and result.get('success') and featured_coin:
            record_global_coin_post(featured_coin['symbol'])
            logger.info(f"[CryptoSocial] Recorded {featured_coin['symbol']} to global cooldown")
        
        return result
        
    except Exception as e:
        logger.error(f"Error posting volume surge: {e}")
        return None


async def post_market_pulse(account_poster: MultiAccountPoster) -> Optional[Dict]:
    """Post overall market pulse with BTC + top alts"""
    try:
        exchange = ccxt.binance({'enableRateLimit': True})
        btc = await exchange.fetch_ticker('BTC/USDT')
        eth = await exchange.fetch_ticker('ETH/USDT')
        tickers = await exchange.fetch_tickers()
        await exchange.close()
        
        btc_change = btc.get('percentage', 0) or 0
        eth_change = eth.get('percentage', 0) or 0
        
        # Count green vs red
        green = 0
        red = 0
        for symbol, data in tickers.items():
            if symbol.endswith('/USDT') and data.get('percentage'):
                if data['percentage'] > 0:
                    green += 1
                else:
                    red += 1
        
        total = green + red
        green_pct = (green / total * 100) if total > 0 else 50
        
        style = random.randint(1, 4)
        
        if style == 1:
            # Market mood
            if green_pct >= 70:
                mood = "🟢 BULLISH - Most coins are green"
            elif green_pct >= 55:
                mood = "📈 LEANING BULLISH"
            elif green_pct >= 45:
                mood = "⚪ MIXED - No clear direction"
            elif green_pct >= 30:
                mood = "📉 LEANING BEARISH"
            else:
                mood = "🔴 BEARISH - Sea of red"
            
            btc_sign = "+" if btc_change >= 0 else ""
            tweet_text = f"""📊 MARKET PULSE

BTC: {btc_sign}{btc_change:.1f}%
Market: {mood}
{green_pct:.0f}% of coins green

#Crypto #Bitcoin"""
        
        elif style == 2:
            # BTC + ETH focus
            btc_sign = "+" if btc_change >= 0 else ""
            eth_sign = "+" if eth_change >= 0 else ""
            
            tweet_text = f"""Market Check ✅

₿ BTC: {btc_sign}{btc_change:.1f}%
⟠ ETH: {eth_sign}{eth_change:.1f}%

{green}/{green+red} coins green ({green_pct:.0f}%)

#Crypto #Bitcoin #Ethereum"""
        
        elif style == 3:
            # Casual
            if green_pct >= 60:
                mood = "Green vibes today"
            elif green_pct <= 40:
                mood = "Red day in crypto"
            else:
                mood = "Mixed market today"
            
            btc_sign = "+" if btc_change >= 0 else ""
            tweet_text = f"""{mood}

BTC {btc_sign}{btc_change:.1f}%
{green_pct:.0f}% of market is green

#Crypto"""
        
        else:
            # Simple stats
            btc_sign = "+" if btc_change >= 0 else ""
            emoji = "🟢" if btc_change >= 0 else "🔴"
            tweet_text = f"""{emoji} BTC {btc_sign}{btc_change:.1f}%

{green} coins green
{red} coins red

#Crypto #Market"""
        
        return account_poster.post_tweet(tweet_text)
        
    except Exception as e:
        logger.error(f"Error posting market pulse: {e}")
        return None


async def post_early_gainer_standard(account_poster: MultiAccountPoster, main_poster) -> Optional[Dict]:
    """Post early gainer for standard accounts with chart - human-like varied posts"""
    try:
        gainers = await main_poster.get_top_gainers_data(20)
        if not gainers:
            return {'success': False, 'error': 'No gainers data'}
        
        # Find early gainers (3-15% range with good volume)
        early_gainers = [
            g for g in gainers 
            if 3 <= g.get('change', 0) <= 15 
            and g.get('volume', 0) >= 3_000_000
            and check_global_coin_cooldown(g['symbol'], max_per_day=1)
        ]
        
        if not early_gainers:
            early_gainers = [g for g in gainers if g.get('change', 0) >= 3][:3]
        
        if not early_gainers:
            return {'success': False, 'error': 'No suitable early gainers'}
        
        coin = random.choice(early_gainers[:5])
        symbol = coin['symbol']
        change = coin.get('change', 0)
        price = coin.get('price', 0)
        volume = coin.get('volume', 0)
        
        price_str = f"${price:,.4f}" if price < 1 else f"${price:,.2f}"
        vol_str = f"${volume/1e6:.1f}M" if volume < 1e9 else f"${volume/1e9:.1f}B"
        
        # Generate chart
        chart_bytes = None
        try:
            from app.services.chart_generator import ChartGenerator
            chart_gen = ChartGenerator()
            chart_bytes = await chart_gen.generate_chart(symbol)
        except Exception as e:
            logger.warning(f"Chart generation failed: {e}")
        
        # Human-like varied tweet styles (10 options)
        style = random.randint(1, 10)
        if style == 1:
            tweet_text = f"""Spotted ${symbol} early today - up {change:.1f}% with solid volume ({vol_str})

Currently trading at {price_str}

The kind of move I like to see before bigger runs

#Crypto"""
        elif style == 2:
            tweet_text = f"""${symbol} catching my attention this morning

+{change:.1f}% on {vol_str} volume
Price: {price_str}

Still early if this momentum holds"""
        elif style == 3:
            tweet_text = f"""Been watching ${symbol} for a bit now

Finally making its move - {change:.1f}% with {vol_str} in volume

{price_str} current price

Chart looking clean"""
        elif style == 4:
            tweet_text = f"""${symbol} waking up

{change:.1f}% move on decent volume ({vol_str})
Trading at {price_str}

These early movers are worth tracking"""
        elif style == 5:
            tweet_text = f"""Not financial advice but ${symbol} is showing some life

Up {change:.1f}% today
{vol_str} volume
{price_str}

Always do your own research"""
        elif style == 6:
            tweet_text = f"""Early mover alert: ${symbol}

Quietly up {change:.1f}% while most aren't watching
Volume: {vol_str}
Price: {price_str}"""
        elif style == 7:
            tweet_text = f"""${symbol} +{change:.1f}%

This is exactly the kind of early momentum I scan for
{vol_str} volume backing the move
Currently {price_str}"""
        elif style == 8:
            tweet_text = f"""Interesting price action on ${symbol}

Up {change:.1f}% with {vol_str} volume
{price_str}

Keeping this one on my watchlist today"""
        elif style == 9:
            tweet_text = f"""${symbol} starting to trend

{change:.1f}% gain so far
Volume looking healthy at {vol_str}
Price sitting at {price_str}

Could be worth watching"""
        else:
            tweet_text = f"""Just noticed ${symbol} is moving

+{change:.1f}% | {vol_str} volume | {price_str}

Not a lot of people talking about this one yet"""
        
        result = None
        if chart_bytes:
            media_id = account_poster.upload_media(chart_bytes)
            if media_id:
                result = account_poster.post_tweet(tweet_text, media_ids=[media_id])
        
        if not result:
            result = account_poster.post_tweet(tweet_text)
        
        if result and result.get('success'):
            record_global_coin_post(symbol)
        
        return result
        
    except Exception as e:
        logger.error(f"Error posting early gainer standard: {e}")
        return {'success': False, 'error': str(e)}


async def post_whale_alert(account_poster: MultiAccountPoster, main_poster) -> Optional[Dict]:
    """Post about coins with unusual volume spikes - human-like"""
    try:
        gainers = await main_poster.get_top_gainers_data(30)
        if not gainers:
            return {'success': False, 'error': 'No data'}
        
        # Find coins with massive volume
        whale_coins = [
            g for g in gainers 
            if g.get('volume', 0) >= 50_000_000
            and check_global_coin_cooldown(g['symbol'], max_per_day=1)
        ]
        
        if not whale_coins:
            whale_coins = [g for g in gainers if g.get('volume', 0) >= 20_000_000][:3]
        
        if not whale_coins:
            return {'success': False, 'error': 'No whale activity detected'}
        
        coin = whale_coins[0]
        symbol = coin['symbol']
        change = coin.get('change', 0)
        volume = coin.get('volume', 0)
        price = coin.get('price', 0)
        
        vol_str = f"${volume/1e6:.0f}M" if volume < 1e9 else f"${volume/1e9:.1f}B"
        price_str = f"${price:,.4f}" if price < 1 else f"${price:,.2f}"
        sign = "+" if change >= 0 else ""
        
        style = random.randint(1, 8)
        if style == 1:
            tweet_text = f"""Interesting... ${symbol} just did {vol_str} in volume

That's not normal activity for this coin

Price {sign}{change:.1f}% at {price_str}

When volume spikes like this, someone usually knows something"""
        elif style == 2:
            tweet_text = f"""${symbol} volume is through the roof today

{vol_str} traded in 24h
Currently {sign}{change:.1f}%

This kind of volume doesn't happen randomly

Worth keeping an eye on"""
        elif style == 3:
            tweet_text = f"""Big money moving into ${symbol}

{vol_str} volume (way above average)
Price: {price_str} ({sign}{change:.1f}%)

Either whales know something or someone's accumulating"""
        elif style == 4:
            tweet_text = f"""The volume on ${symbol} right now is insane

{vol_str} in 24 hours
{sign}{change:.1f}% on the day

I always pay attention when volume spikes like this"""
        elif style == 5:
            tweet_text = f"""${symbol} showing serious accumulation signs

Volume: {vol_str}
Price: {price_str}
24h: {sign}{change:.1f}%

Smart money tends to be early"""
        elif style == 6:
            tweet_text = f"""Can't ignore this ${symbol} volume

{vol_str} traded today - that's massive for this coin

Currently trading at {price_str}

Something's brewing here"""
        elif style == 7:
            tweet_text = f"""${symbol} whale activity detected

{vol_str} volume isn't retail buying
{sign}{change:.1f}% move

When you see volume like this, pay attention"""
        else:
            tweet_text = f"""Spotted unusual activity on ${symbol}

Volume just hit {vol_str}
Price sitting at {price_str} ({sign}{change:.1f}%)

This is the kind of thing I watch for"""
        
        result = account_poster.post_tweet(tweet_text)
        
        if result and result.get('success'):
            record_global_coin_post(symbol)
        
        return result
        
    except Exception as e:
        logger.error(f"Error posting whale alert: {e}")
        return {'success': False, 'error': str(e)}


async def post_funding_extreme(account_poster: MultiAccountPoster) -> Optional[Dict]:
    """Post about extreme funding rates - human-like"""
    try:
        import ccxt.async_support as ccxt_async
        exchange = ccxt_async.binance({'options': {'defaultType': 'future'}})
        
        funding = await exchange.fetch_funding_rates()
        await exchange.close()
        
        if not funding:
            return {'success': False, 'error': 'No funding data'}
        
        # Find extreme funding rates
        extremes = []
        for symbol, data in funding.items():
            rate = data.get('fundingRate', 0) or 0
            if abs(rate) >= 0.0005:  # 0.05% or higher
                base = symbol.replace('/USDT:USDT', '').replace('USDT', '')
                extremes.append({
                    'symbol': base,
                    'rate': rate * 100,  # Convert to percentage
                    'direction': 'LONG' if rate > 0 else 'SHORT'
                })
        
        if not extremes:
            return {'success': False, 'error': 'No extreme funding rates'}
        
        # Sort by absolute rate
        extremes.sort(key=lambda x: abs(x['rate']), reverse=True)
        top = extremes[0]
        
        if not check_global_coin_cooldown(top['symbol'], max_per_day=1):
            if len(extremes) > 1:
                top = extremes[1]
            else:
                return {'success': False, 'error': 'Coin on cooldown'}
        
        is_long_crowded = top['rate'] > 0
        rate_str = f"{abs(top['rate']):.3f}%"
        
        style = random.randint(1, 6)
        if style == 1:
            if is_long_crowded:
                tweet_text = f"""${top['symbol']} funding rate just hit {rate_str}

That's a lot of longs paying shorts right now

When everyone's on one side of the trade... you know what usually happens

Be careful out there"""
            else:
                tweet_text = f"""${top['symbol']} funding rate at -{rate_str}

Shorts are paying longs heavily

Could be setting up for a squeeze if price starts moving up

Interesting setup to watch"""
        elif style == 2:
            crowd = "long" if is_long_crowded else "short"
            tweet_text = f"""Everyone's {crowd} on ${top['symbol']} right now

Funding: {'+' if is_long_crowded else '-'}{rate_str}

These crowded trades have a way of reversing when you least expect it"""
        elif style == 3:
            tweet_text = f"""${top['symbol']} funding is getting extreme

{'+' if is_long_crowded else '-'}{rate_str} per 8 hours

The market has a way of punishing overcrowded positions

Not saying it'll happen now but worth noting"""
        elif style == 4:
            direction = "bullish" if is_long_crowded else "bearish"
            opposite = "longs" if is_long_crowded else "shorts"
            tweet_text = f"""Traders are extremely {direction} on ${top['symbol']}

Funding at {'+' if is_long_crowded else '-'}{rate_str}

If this reverses, a lot of {opposite} could get caught"""
        elif style == 5:
            tweet_text = f"""Funding rate alert: ${top['symbol']}

Currently at {'+' if is_long_crowded else '-'}{rate_str}

This is one of the highest I've seen in a while

Usually means the crowd is about to be wrong"""
        else:
            squeeze_type = "short squeeze" if not is_long_crowded else "long squeeze"
            tweet_text = f"""${top['symbol']} looking ripe for a {squeeze_type}

Funding: {'+' if is_long_crowded else '-'}{rate_str}

When funding gets this extreme, reversals tend to be violent"""
        
        result = account_poster.post_tweet(tweet_text)
        
        if result and result.get('success'):
            record_global_coin_post(top['symbol'])
        
        return result
        
    except Exception as e:
        logger.error(f"Error posting funding extreme: {e}")
        return {'success': False, 'error': str(e)}


async def post_quick_ta(account_poster: MultiAccountPoster, main_poster) -> Optional[Dict]:
    """Post quick technical analysis setup - human-like"""
    try:
        gainers = await main_poster.get_top_gainers_data(15)
        if not gainers:
            return {'success': False, 'error': 'No data'}
        
        # Pick a coin with good movement
        candidates = [
            g for g in gainers 
            if 2 <= abs(g.get('change', 0)) <= 20
            and g.get('volume', 0) >= 5_000_000
            and check_global_coin_cooldown(g['symbol'], max_per_day=1)
        ]
        
        if not candidates:
            return {'success': False, 'error': 'No suitable coins for TA'}
        
        coin = random.choice(candidates[:5])
        symbol = coin['symbol']
        change = coin.get('change', 0)
        price = coin.get('price', 0)
        
        price_str = f"${price:,.4f}" if price < 1 else f"${price:,.2f}"
        sign = "+" if change >= 0 else ""
        
        # Generate chart with TA
        chart_bytes = None
        chart_analysis = None
        try:
            from app.services.chart_generator import ChartGenerator
            chart_gen = ChartGenerator()
            chart_bytes, chart_analysis = await chart_gen.generate_chart_with_analysis(symbol)
        except Exception as e:
            logger.warning(f"Chart generation failed: {e}")
        
        # Build human-like TA tweet
        if chart_analysis:
            rsi = chart_analysis.get('rsi', 50)
            trend = chart_analysis.get('trend', 'neutral')
            
            style = random.randint(1, 8)
            
            if style == 1:
                if rsi > 70:
                    tweet_text = f"""${symbol} is looking stretched on the RSI ({rsi:.0f})

Price at {price_str} after a {sign}{change:.1f}% move

Usually when RSI gets this high, a pullback follows

Not saying sell, just be aware"""
                elif rsi < 30:
                    tweet_text = f"""${symbol} RSI just hit {rsi:.0f} - that's oversold territory

Currently {price_str} ({sign}{change:.1f}%)

Oversold doesn't mean buy immediately but worth watching for a bounce"""
                else:
                    tweet_text = f"""Looking at ${symbol} chart

RSI sitting at {rsi:.0f} - room to move either direction
Price: {price_str} ({sign}{change:.1f}%)

Clean setup forming here"""
            elif style == 2:
                trend_word = "bullish" if trend == 'bullish' else "bearish" if trend == 'bearish' else "choppy"
                tweet_text = f"""${symbol} chart analysis

Trend: {trend_word}
RSI: {rsi:.0f}
Price: {price_str}

The technicals are {'' if trend == 'neutral' else 'looking '}{'interesting' if trend == 'neutral' else 'pretty clear'} on this one"""
            elif style == 3:
                tweet_text = f"""Pulled up the ${symbol} chart

{sign}{change:.1f}% today at {price_str}
RSI: {rsi:.0f}

{'Momentum is there' if trend == 'bullish' else 'Sellers in control' if trend == 'bearish' else 'Waiting for direction'}"""
            elif style == 4:
                tweet_text = f"""Technical check on ${symbol}

Price: {price_str} ({sign}{change:.1f}%)
RSI: {rsi:.0f}
Trend: {trend}

I like what I'm seeing on the chart here"""
            elif style == 5:
                if trend == 'bullish':
                    tweet_text = f"""${symbol} holding its uptrend nicely

{price_str} | {sign}{change:.1f}%
RSI at {rsi:.0f}

As long as the structure holds, bulls are in control"""
                else:
                    tweet_text = f"""${symbol} technical breakdown

{price_str} ({sign}{change:.1f}%)
RSI: {rsi:.0f}

Chart is telling a story here"""
            elif style == 6:
                tweet_text = f"""Quick TA on ${symbol}

RSI: {rsi:.0f}
Trend: {trend.title()}
{sign}{change:.1f}% at {price_str}

One to keep on the watchlist"""
            elif style == 7:
                tweet_text = f"""${symbol} chart update

Currently trading at {price_str}
24h: {sign}{change:.1f}%
RSI reading: {rsi:.0f}

{'Looking healthy' if rsi < 65 and rsi > 35 else 'Extended but could run more'}"""
            else:
                tweet_text = f"""Been studying the ${symbol} chart

{price_str} right now ({sign}{change:.1f}%)
RSI: {rsi:.0f}

Technicals suggesting {'more upside possible' if trend == 'bullish' else 'caution here' if trend == 'bearish' else 'patience needed'}"""
        else:
            style = random.randint(1, 4)
            if style == 1:
                tweet_text = f"""${symbol} chart looking interesting today

{price_str} ({sign}{change:.1f}%)

Worth a closer look if you have time"""
            elif style == 2:
                tweet_text = f"""Checking in on ${symbol}

{sign}{change:.1f}% move
Currently at {price_str}

The chart has my attention"""
            elif style == 3:
                tweet_text = f"""${symbol} on my radar

Price: {price_str}
Move: {sign}{change:.1f}%

Keeping an eye on this one"""
            else:
                tweet_text = f"""Quick look at ${symbol}

Trading at {price_str} after {sign}{change:.1f}%

Something brewing here"""
        
        result = None
        if chart_bytes:
            media_id = account_poster.upload_media(chart_bytes)
            if media_id:
                result = account_poster.post_tweet(tweet_text, media_ids=[media_id])
        
        if not result:
            result = account_poster.post_tweet(tweet_text)
        
        if result and result.get('success'):
            record_global_coin_post(symbol)
        
        return result
        
    except Exception as e:
        logger.error(f"Error posting quick TA: {e}")
        return {'success': False, 'error': str(e)}


def is_social_account(account_name: str) -> bool:
    """Check if this is the Crypto Social account"""
    name_lower = account_name.lower()
    return 'social' in name_lower or 'cryptosocial' in name_lower


async def post_for_social_account(account_poster: MultiAccountPoster, post_type: str) -> Optional[Dict]:
    """Handle posting for Crypto Social account - NEWS & EARLY GAINERS focus"""
    # Direct mapping for manual post buttons
    if post_type == 'breaking_news':
        return await post_social_news(account_poster)
    
    elif post_type == 'early_gainer':
        return await post_early_gainers(account_poster)
    
    elif post_type == 'momentum_shift':
        return await post_momentum_shift(account_poster)
    
    elif post_type == 'volume_surge':
        return await post_volume_surge(account_poster)
    
    elif post_type == 'market_pulse':
        return await post_market_pulse(account_poster)
    
    # Map standard post types to social-specific functions
    elif post_type == 'featured_coin':
        # Rotate between news and early gainers
        social_posts = [
            post_social_news,
            post_early_gainers,
            post_momentum_shift,
        ]
        return await random.choice(social_posts)(account_poster)
    
    elif post_type == 'market_summary':
        return await post_market_pulse(account_poster)
    
    elif post_type == 'top_gainers':
        return await post_early_gainers(account_poster)
    
    elif post_type == 'btc_update':
        return await post_social_news(account_poster)
    
    elif post_type == 'altcoin_movers':
        return await post_momentum_shift(account_poster)
    
    elif post_type == 'daily_recap':
        return await post_volume_surge(account_poster)
    
    else:
        # Random pick for any other type
        funcs = [post_social_news, post_early_gainers, post_momentum_shift, post_volume_surge]
        return await random.choice(funcs)(account_poster)
