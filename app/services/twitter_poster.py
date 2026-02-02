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
        
        tips = [
            "\n💡 High volume = more conviction",
            "\n📊 Watch for continuation patterns",
            "\n🎯 Set your targets wisely",
            "\n⚠️ Always manage your risk",
            "\n🔍 DYOR before entering",
            "\n💭 What's on your watchlist?",
            "\n🤔 Any of these catching your eye?",
            "\n📈 Momentum is everything",
            "\n💪 Bulls in control today",
            "\n🎲 Which one are you playing?",
            "\n⚡ Fast movers need fast decisions",
            "\n🔥 The heat is real",
            "\n💎 Diamond hands prevail",
            "\n🌊 Ride the wave or wait?",
            "\n🎯 Pick your entries carefully",
            "\n📍 Mark these on your charts",
            "\n💰 Where are you putting your chips?",
            "\n🧠 Trade smart, not hard",
            "\n⏰ Timing is everything",
            "\n🚀 Room to run?"
        ]
        lines.append(random.choice(tips))
        lines.append("\n#Crypto #Trading #TopGainers #Altcoins")
        
        tweet_text = "\n".join(lines)
        return await self.post_tweet(tweet_text)
    
    async def post_market_summary(self) -> Optional[Dict]:
        """Post market summary update"""
        market = await self.get_market_summary()
        
        if not market:
            return None
        
        btc_emoji = "🟢" if market['btc_change'] >= 0 else "🔴"
        eth_emoji = "🟢" if market['eth_change'] >= 0 else "🔴"
        
        btc_sign = "+" if market['btc_change'] >= 0 else ""
        eth_sign = "+" if market['eth_change'] >= 0 else ""
        
        # Determine market sentiment
        if market['btc_change'] >= 3:
            sentiment = "🚀 BULLISH"
        elif market['btc_change'] <= -3:
            sentiment = "🐻 BEARISH"
        else:
            sentiment = "😐 NEUTRAL"
        
        tweet_text = f"""📊 MARKET UPDATE

{btc_emoji} BTC ${market['btc_price']:,.0f} ({btc_sign}{market['btc_change']:.1f}%)
{eth_emoji} ETH ${market['eth_price']:,.0f} ({eth_sign}{market['eth_change']:.1f}%)

Market Sentiment: {sentiment}

#Bitcoin #Ethereum #Crypto #CryptoMarket"""
        
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
            
            tips = [
                "\n⚠️ Dip or dead? Watch the volume!",
                "\n🔍 Opportunity or trap? DYOR!",
                "\n💡 Dead cat bounce incoming?",
                "\n📊 Check support levels before buying",
                "\n🎯 Patience is key in these moments",
                "\n🤔 Catching knives is risky",
                "\n💭 Would you buy any of these?",
                "\n⏰ Wait for confirmation",
                "\n🧠 Don't FOMO into falling coins",
                "\n📈 Or is this the opportunity?",
                "\n💎 Diamond hands or cut losses?",
                "\n🎲 Risk vs reward...",
                "\n⚡ Volatility = opportunity?",
                "\n🔮 Where's the bottom?",
                "\n💪 Only strong hands survive this",
                "\n🌊 Waiting for the reversal",
                "\n📍 Key support levels to watch",
                "\n🎯 Be patient, be smart",
                "\n⚠️ Remember: scared money don't make money",
                "\n🏃 Running or staying?"
            ]
            lines.append(random.choice(tips))
            lines.append("\n#Crypto #CryptoNews #Altcoins #Trading")
            tweet_text = "\n".join(lines)
            return await self.post_tweet(tweet_text)
            
        except Exception as e:
            logger.error(f"Failed to post top losers: {e}")
            return None
    
    async def post_btc_update(self) -> Optional[Dict]:
        """Post detailed BTC update"""
        try:
            exchange = ccxt.binance({'enableRateLimit': True})
            btc = await exchange.fetch_ticker('BTC/USDT')
            await exchange.close()
            
            price = btc['last']
            change = btc['percentage'] or 0
            high = btc['high']
            low = btc['low']
            volume = btc['quoteVolume'] or 0
            
            emoji = "🟢" if change >= 0 else "🔴"
            sign = "+" if change >= 0 else ""
            
            # Price level commentary
            if price >= 100000:
                level = "🚀 Above $100K!"
            elif price >= 90000:
                level = "💪 Holding strong"
            elif price >= 80000:
                level = "📊 Key support zone"
            else:
                level = "⚠️ Watch this level"
            
            tweet_text = f"""₿ BITCOIN UPDATE

{emoji} ${price:,.0f} ({sign}{change:.1f}%)

📈 24h High: ${high:,.0f}
📉 24h Low: ${low:,.0f}
💰 Volume: ${volume/1e9:.1f}B

{level}

#Bitcoin #BTC #Crypto"""
            
            return await self.post_tweet(tweet_text)
            
        except Exception as e:
            logger.error(f"Failed to post BTC update: {e}")
            return None
    
    async def post_altcoin_movers(self) -> Optional[Dict]:
        """Post notable altcoin movements"""
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
                    if vol >= 10_000_000:  # Min $10M volume
                        alts.append({
                            'symbol': base,
                            'change': data['percentage'],
                            'volume': vol
                        })
            
            # Sort by absolute change
            alts.sort(key=lambda x: abs(x['change']), reverse=True)
            top_movers = alts[:6]
            
            if not top_movers:
                return None
            
            lines = ["🔥 ALTCOIN MOVERS\n"]
            for coin in top_movers:
                emoji = "🟢" if coin['change'] >= 0 else "🔴"
                sign = "+" if coin['change'] >= 0 else ""
                lines.append(f"{emoji} ${coin['symbol']} {sign}{coin['change']:.1f}%")
            
            lines.append("\n#Altcoins #CryptoTrading #Altseason")
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
        """Check if coin has been posted too many times today"""
        today = datetime.utcnow().date()
        
        # Reset daily counts
        if self.coin_post_reset != today:
            self.coin_post_count = {}
            self.coin_post_reset = today
        
        count = self.coin_post_count.get(symbol, 0)
        return count < max_per_day
    
    def _record_coin_post(self, symbol: str):
        """Record that a coin was posted"""
        self.coin_post_count[symbol] = self.coin_post_count.get(symbol, 0) + 1
        logger.info(f"📊 {symbol} posted {self.coin_post_count[symbol]}x today")
    
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
            
            # Massive variety for human-like posts
            if change >= 20:
                headlines = [
                    f"🚀 ${symbol} IS ON FIRE!",
                    f"🔥 ${symbol} EXPLODING RIGHT NOW",
                    f"💥 ${symbol} GOING PARABOLIC",
                    f"⚡ ${symbol} CAN'T BE STOPPED",
                    f"🎢 ${symbol} TO THE MOON",
                    f"💎 ${symbol} DIAMONDS FORMING",
                    f"🏆 ${symbol} ABSOLUTELY SENDING IT",
                    f"⚡ Holy... ${symbol} is flying",
                    f"👀 ${symbol} woke up and chose violence",
                    f"🚨 ${symbol} ALERT - This is nuts",
                    f"📈 ${symbol} said 'watch this'",
                    f"💪 ${symbol} showing everyone how it's done",
                    f"🔥 Who else is watching ${symbol}?",
                    f"⬆️ ${symbol} just keeps going",
                    f"🎯 ${symbol} hitting different today"
                ]
                subtexts = [
                    "Massive momentum building",
                    "Volume is insane right now",
                    "Bulls have taken full control",
                    "This move is just getting started",
                    "Shorts getting absolutely rekt",
                    "The chart looks beautiful",
                    "Momentum traders eating good",
                    "This is what we wait for",
                    "Whales are definitely involved here",
                    "Pure strength on display",
                    "Nothing stopping this train",
                    "Technical analysis working perfectly",
                    "Called it. Just saying 😏",
                    "Imagine not being in this",
                    "The breakout everyone was waiting for"
                ]
            elif change >= 10:
                headlines = [
                    f"📈 ${symbol} BREAKING OUT",
                    f"🎯 ${symbol} HITTING TARGETS",
                    f"💪 ${symbol} SHOWING STRENGTH",
                    f"📊 ${symbol} ON THE MOVE",
                    f"✅ ${symbol} Looking really good",
                    f"🔥 ${symbol} heating up",
                    f"⬆️ ${symbol} pushing higher",
                    f"💹 ${symbol} is cooking",
                    f"👀 ${symbol} catching attention",
                    f"📈 Nice move on ${symbol}",
                    f"🎢 ${symbol} gaining traction",
                    f"💎 ${symbol} holders winning today",
                    f"⚡ ${symbol} waking up",
                    f"🚀 ${symbol} building steam",
                    f"🔍 Interesting ${symbol} action"
                ]
                subtexts = [
                    "Breaking key resistance levels",
                    "Smart money loading up",
                    "Technical breakout confirmed",
                    "Buyers stepping in hard",
                    "Structure looking bullish",
                    "Higher lows forming nicely",
                    "Momentum picking up fast",
                    "Volume confirming the move",
                    "This could run further",
                    "Breaking above the noise",
                    "Clean price action",
                    "Bulls taking control here",
                    "Nice setup developing",
                    "Worth keeping an eye on",
                    "Could be early still"
                ]
            elif change >= 5:
                headlines = [
                    f"💹 ${symbol} Looking Strong",
                    f"📊 ${symbol} Building Momentum",
                    f"✅ ${symbol} Holding Well",
                    f"🔍 ${symbol} Worth Watching",
                    f"📈 ${symbol} quietly moving",
                    f"👀 Keeping an eye on ${symbol}",
                    f"💪 ${symbol} showing life",
                    f"🎯 ${symbol} on my radar",
                    f"⬆️ ${symbol} ticking higher",
                    f"🔥 ${symbol} looking decent",
                    f"💎 ${symbol} building a base",
                    f"📍 ${symbol} at an interesting level",
                    f"⚡ ${symbol} perking up",
                    f"✨ ${symbol} catching bids",
                    f"🌱 ${symbol} growing steady"
                ]
                subtexts = [
                    "Steady gains with volume",
                    "Accumulation phase looks solid",
                    "Setting up for a bigger move?",
                    "Patient holders being rewarded",
                    "Slow and steady wins the race",
                    "Building a nice foundation",
                    "Healthy price action",
                    "No need to rush, let it develop",
                    "Structure looking constructive",
                    "Could be just the beginning",
                    "Consolidation looking healthy",
                    "Buyers showing interest",
                    "Nice steady climb",
                    "Support holding strong",
                    "Textbook accumulation pattern"
                ]
            else:
                headlines = [
                    f"👀 ${symbol} Making Moves",
                    f"🔎 Watching ${symbol} Closely",
                    f"📍 ${symbol} At Key Level",
                    f"💡 ${symbol} On The Radar",
                    f"🎯 ${symbol} worth a look",
                    f"📊 Checking out ${symbol}",
                    f"🔍 ${symbol} caught my attention",
                    f"💭 Thoughts on ${symbol}?",
                    f"👁️ Eyes on ${symbol}",
                    f"📈 ${symbol} showing some movement",
                    f"⚡ ${symbol} starting to move",
                    f"🌊 ${symbol} making waves",
                    f"💫 ${symbol} looking interesting",
                    f"🎲 ${symbol} could be one to watch",
                    f"🔮 ${symbol} setup forming"
                ]
                subtexts = [
                    "One to watch closely",
                    "Could be setting up something",
                    "Interesting price action here",
                    "Keep this one on your list",
                    "Early stages, watching closely",
                    "Something brewing here",
                    "Let's see how this develops",
                    "Adding to watchlist",
                    "Might be worth a deeper look",
                    "Chart pattern forming",
                    "Could go either way from here",
                    "Waiting for confirmation",
                    "Keeping tabs on this one",
                    "Potential opportunity brewing",
                    "Worth monitoring imo"
                ]
            
            headline = random.choice(headlines)
            subtext = random.choice(subtexts)
            
            # Massive variety of CTAs
            ctas = [
                "🤔 Where's it heading? Drop your prediction 👇",
                "💬 What's your take? Comment below 👇",
                "📊 Bullish or bearish? Let us know 👇",
                "🎯 What's your target? Share below 👇",
                "🔮 Where do YOU think it's going? 👇",
                "💭 Thoughts? Drop them below 👇",
                "🗣️ What are you seeing? 👇",
                "📈 Long or short from here? 👇",
                "🎲 Taking a position? Let us know 👇",
                "💡 Your analysis? Share it 👇",
                "🔥 Who's trading this? 👇",
                "⚡ What's your play? 👇",
                "🎯 Entry or wait? 👇",
                "💎 Holding or selling? 👇",
                "🚀 Moon or doom? 👇",
                "📉 Buying the dip or nah? 👇",
                "🤷 What would you do here? 👇",
                "💬 Agree or disagree? 👇",
                "👀 Anyone else watching this? 👇",
                "🎢 Riding this wave? 👇"
            ]
            cta = random.choice(ctas)
            
            price_str = f"${price:,.4f}" if price < 1 else f"${price:,.2f}"
            
            tweet_text = f"""{headline}

💰 Price: {price_str}
📊 24h Change: {sign}{change:.1f}%
📈 Volume: {vol_str}

{subtext}

{cta}

#Crypto #{symbol} #Trading #Altcoins"""
            
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
            
            ctas = [
                "\n\n📈 How did YOUR bags perform today? 👇",
                "\n\n💬 Share your wins (or losses) below 👇",
                "\n\n🤔 Did you make money today? Let us know 👇",
                "\n\n📊 What was your best trade today? 👇",
                "\n\n💰 Green or red for you? 👇",
                "\n\n🎯 Hit your targets today? 👇",
                "\n\n🔥 What are you watching tomorrow? 👇",
                "\n\n💭 Any regrets today? 👇",
                "\n\n🚀 Ready for tomorrow? 👇",
                "\n\n📍 What's your game plan? 👇",
                "\n\n🎲 Taking profits or holding? 👇",
                "\n\n⚡ Best play of the day? 👇",
                "\n\n🏆 Who made gains today? 👇",
                "\n\n💎 Diamond hands check! 👇",
                "\n\n🤷 How did you play it today? 👇"
            ]
            tweet_text += random.choice(ctas)
            tweet_text += "\n\n#Crypto #Bitcoin #CryptoTrading #DailyRecap"
            
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


def get_twitter_schedule() -> Dict:
    """Get the full posting schedule and next post info"""
    global POSTED_SLOTS, LAST_POSTED_DAY
    
    now = datetime.utcnow()
    current_day = now.date()
    
    # Reset at midnight
    if LAST_POSTED_DAY != current_day:
        POSTED_SLOTS = set()
        LAST_POSTED_DAY = current_day
    
    # Find next scheduled post
    next_post = None
    next_post_time = None
    
    post_type_labels = {
        'featured_coin': '🌟 Featured Coin + Chart',
        'market_summary': '📊 Market Summary',
        'top_gainers': '🚀 Top Gainers',
        'btc_update': '₿ BTC Update',
        'altcoin_movers': '💹 Altcoin Movers',
        'daily_recap': '📈 Daily Recap'
    }
    
    schedule_info = []
    
    for hour, minute, post_type in POST_SCHEDULE:
        slot_key = f"{hour}:{minute}"
        slot_time = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
        
        # If slot is in the past today, move to tomorrow
        if slot_time < now:
            slot_time += timedelta(days=1)
        
        posted = slot_key in POSTED_SLOTS
        label = post_type_labels.get(post_type, post_type)
        
        schedule_info.append({
            'time': slot_time,
            'time_str': slot_time.strftime('%H:%M UTC'),
            'type': label,
            'posted': posted,
            'slot_key': slot_key
        })
        
        # Find next unposted slot
        if not posted and (next_post_time is None or slot_time < next_post_time):
            next_post = label
            next_post_time = slot_time
    
    # Sort by time
    schedule_info.sort(key=lambda x: x['time'])
    
    # Calculate time until next post
    time_until = None
    if next_post_time:
        delta = next_post_time - now
        hours = int(delta.total_seconds() // 3600)
        minutes = int((delta.total_seconds() % 3600) // 60)
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
    """Background loop for automated posting - 15 posts per day"""
    global POSTED_SLOTS, LAST_POSTED_DAY
    
    poster = get_twitter_poster()
    
    if not poster.client:
        logger.error("Twitter not configured - auto posting disabled")
        return
    
    logger.info("🐦 Starting Twitter auto-post loop (15 posts/day)")
    
    while True:
        try:
            if not AUTO_POST_ENABLED:
                await asyncio.sleep(60)
                continue
            
            now = datetime.utcnow()
            current_day = now.date()
            
            # Reset posted slots at midnight
            if LAST_POSTED_DAY is not None and LAST_POSTED_DAY != current_day:
                POSTED_SLOTS.clear()
            
            # Check each scheduled slot
            for hour, minute, post_type in POST_SCHEDULE:
                slot_key = f"{hour}:{minute}"
                
                # Add random offset of -10 to +15 minutes for natural timing
                random_offset = random.randint(-10, 15)
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
                    # Post based on type
                    result = None
                    
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
                        logger.info(f"✅ Auto-posted {post_type} at {slot_key}")
                        POSTED_SLOTS.add(slot_key)
                    else:
                        logger.warning(f"⚠️ Failed to auto-post {post_type}")
                    
                    break
            
            # Check every 2 minutes
            await asyncio.sleep(120)
            
        except Exception as e:
            logger.error(f"Error in auto-post loop: {e}")
            await asyncio.sleep(60)
