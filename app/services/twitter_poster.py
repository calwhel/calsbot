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
        
        # Build tweet text with more detail
        lines = ["🚀 TOP 5 GAINERS RIGHT NOW\n"]
        
        for i, coin in enumerate(gainers, 1):
            emoji = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else "🔥" if i == 4 else "📈"
            change_sign = "+" if coin['change'] >= 0 else ""
            price = coin.get('price', 0)
            price_str = f"${price:,.4f}" if price < 1 else f"${price:,.2f}"
            vol = coin.get('volume', 0)
            vol_str = f"${vol/1e6:.1f}M" if vol < 1e9 else f"${vol/1e9:.1f}B"
            lines.append(f"{emoji} ${coin['symbol']} {change_sign}{coin['change']:.1f}% @ {price_str} ({vol_str} vol)")
        
        lines.append("\n💡 High volume = more conviction")
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
            
            lines = ["📉 BIGGEST LOSERS (24H)\n"]
            for i, coin in enumerate(losers, 1):
                emoji = "💀" if i == 1 else "🩸" if i == 2 else "📉"
                price = coin.get('price', 0)
                price_str = f"${price:,.4f}" if price < 1 else f"${price:,.2f}"
                lines.append(f"{emoji} ${coin['symbol']} {coin['change']:.1f}% @ {price_str}")
            
            lines.append("\n⚠️ Dip or dead? Watch the volume!")
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
    
    async def post_featured_coin(self) -> Optional[Dict]:
        """Post featured top gainer with professional chart"""
        try:
            gainers = await self.get_top_gainers_data(10)
            if not gainers:
                logger.warning("No gainers data available for featured coin")
                return None
            
            # Pick the best performing coin with good volume
            featured = None
            for coin in gainers:
                if coin.get('volume', 0) >= 5_000_000:  # Min $5M volume
                    featured = coin
                    break
            
            if not featured:
                featured = gainers[0]
            
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
            
            # High engagement tweet format
            if change >= 20:
                headline = f"🚀 ${symbol} IS ON FIRE!"
                subtext = "Massive momentum building"
            elif change >= 10:
                headline = f"📈 ${symbol} BREAKING OUT"
                subtext = "Breaking key resistance levels"
            elif change >= 5:
                headline = f"💹 ${symbol} Looking Strong"
                subtext = "Steady gains with volume"
            else:
                headline = f"👀 ${symbol} Making Moves"
                subtext = "One to watch closely"
            
            price_str = f"${price:,.4f}" if price < 1 else f"${price:,.2f}"
            
            tweet_text = f"""{headline}

💰 Price: {price_str}
📊 24h Change: {sign}{change:.1f}%
📈 Volume: {vol_str}

{subtext}

🤔 Where's it heading? Drop your prediction 👇

#Crypto #{symbol} #Trading #Altcoins"""
            
            if media_id:
                return await self.post_tweet(tweet_text, media_ids=[media_id])
            else:
                return await self.post_tweet(tweet_text)
            
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
                day_text = "BULLISH DAY"
                mood = "Bulls in control 🐂"
            elif market['btc_change'] <= -3:
                day_emoji = "🔴"
                day_text = "BEARISH DAY"
                mood = "Bears taking over 🐻"
            else:
                day_emoji = "⚪"
                day_text = "CHOPPY DAY"
                mood = "Sideways action 📊"
            
            tweet_text = f"""{day_emoji} DAILY RECAP: {day_text}

₿ BTC: ${market['btc_price']:,.0f} ({btc_sign}{market['btc_change']:.1f}%)
⟠ ETH: ${market['eth_price']:,.0f} ({eth_sign}{market['eth_change']:.1f}%)
"""
            if gainers:
                tweet_text += f"\n🏆 Top Gainer: ${gainers[0]['symbol']} +{gainers[0]['change']:.1f}%"
                if len(gainers) > 1:
                    tweet_text += f"\n🥈 Runner Up: ${gainers[1]['symbol']} +{gainers[1]['change']:.1f}%"
            
            tweet_text += f"\n\n{mood}"
            tweet_text += "\n\n📈 How did YOUR bags perform today? 👇\n\n#Crypto #Bitcoin #CryptoTrading #DailyRecap"
            
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


async def auto_post_loop():
    """Background loop for automated posting - 15 posts per day"""
    poster = get_twitter_poster()
    
    if not poster.client:
        logger.error("Twitter not configured - auto posting disabled")
        return
    
    logger.info("🐦 Starting Twitter auto-post loop (15 posts/day)")
    
    # 15 posts spread across 24 hours (~every 96 minutes)
    # Mix of content types for high engagement
    post_schedule = [
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
    
    posted_slots = set()
    
    while True:
        try:
            if not AUTO_POST_ENABLED:
                await asyncio.sleep(60)
                continue
            
            now = datetime.utcnow()
            current_day = now.date()
            
            # Reset posted slots at midnight
            if hasattr(auto_post_loop, 'last_day') and auto_post_loop.last_day != current_day:
                posted_slots.clear()
            auto_post_loop.last_day = current_day
            
            # Check each scheduled slot
            for hour, minute, post_type in post_schedule:
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
                
                if time_diff <= 180 and slot_key not in posted_slots:
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
                        posted_slots.add(slot_key)
                    else:
                        logger.warning(f"⚠️ Failed to auto-post {post_type}")
                    
                    break
            
            # Check every 2 minutes
            await asyncio.sleep(120)
            
        except Exception as e:
            logger.error(f"Error in auto-post loop: {e}")
            await asyncio.sleep(60)
