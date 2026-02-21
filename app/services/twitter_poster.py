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

# Global cache for top gainers data to handle API failures
_gainers_cache = {
    'data': [],
    'timestamp': None,
    'ttl': 300  # 5 minutes cache
}

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

# Posting limits - 25 posts per day (20 regular + 4 campaign + 1 buffer)
MAX_POSTS_PER_DAY = 25
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


def get_time_context() -> Dict[str, str]:
    """Get time-of-day context for more natural posts"""
    from datetime import datetime
    hour = datetime.utcnow().hour
    
    if 5 <= hour < 12:
        return {
            'period': 'morning',
            'greeting': random.choice(['Morning scan.', 'Early look at the charts.', 'Coffee and charts.', '']),
            'energy': random.choice(['Still waking up but', 'Starting the day and', 'Morning check shows', ''])
        }
    elif 12 <= hour < 17:
        return {
            'period': 'afternoon', 
            'greeting': random.choice(['Afternoon update.', 'Midday check.', 'Quick afternoon scan.', '']),
            'energy': random.choice(['Been watching and', 'Afternoon showing', 'Checking in and', ''])
        }
    elif 17 <= hour < 22:
        return {
            'period': 'evening',
            'greeting': random.choice(['Evening charts.', 'End of day look.', 'Evening scan.', '']),
            'energy': random.choice(['Wrapping up the day and', 'Evening showing', 'Late day check shows', ''])
        }
    else:
        return {
            'period': 'night',
            'greeting': random.choice(['Late night charts.', 'Cant sleep, checking charts.', 'Night owl hours.', '']),
            'energy': random.choice(['Late night and', 'Should be sleeping but', 'Night session showing', ''])
        }


def get_random_mood() -> str:
    """Get a random mood/personality modifier"""
    moods = [
        '',  # No mood modifier (most common)
        '',
        '',
        'Feeling good about this one. ',
        'Not gonna lie, ',
        'Honestly, ',
        'Gotta say, ',
        'Real talk, ',
        'For what its worth, ',
        'Just my take but ',
        'Lowkey think ',
        'Ngl ',
        'Tbh ',
    ]
    return random.choice(moods)


def get_day_context() -> Dict[str, str]:
    """Get day-of-week context for natural posts"""
    from datetime import datetime
    day = datetime.utcnow().weekday()  # 0=Monday, 6=Sunday
    
    if day == 0:  # Monday
        return {
            'day': 'monday',
            'vibe': random.choice(['Back to the charts after the weekend.', 'Monday grind begins.', 'New week, new opportunities.', '']),
            'energy': random.choice(['Starting the week and', 'Monday check shows', ''])
        }
    elif day == 4:  # Friday
        return {
            'day': 'friday',
            'vibe': random.choice(['Ending the week on a good note.', 'Friday vibes.', 'Weekend is close.', '']),
            'energy': random.choice(['Wrapping up the week and', 'Friday showing', ''])
        }
    elif day in [5, 6]:  # Weekend
        return {
            'day': 'weekend',
            'vibe': random.choice(['Weekend trading hits different.', 'Charts dont sleep.', 'Weekend warriors.', '']),
            'energy': random.choice(['Weekend session and', 'Even on weekends', ''])
        }
    else:  # Midweek
        return {
            'day': 'midweek',
            'vibe': random.choice(['', '', '']),  # Usually no special vibe midweek
            'energy': ''
        }


def get_trading_context() -> str:
    """Get random personal trading context - use sparingly"""
    contexts = [
        '',  # Most common - no context
        '',
        '',
        '',
        '',
        'Added to my bag on this one. ',
        'Been holding this since way lower. ',
        'Almost sold yesterday, glad I didnt. ',
        'This ones been on my radar for a while. ',
        'Finally seeing some movement on this. ',
        'Trimmed a little here but still holding. ',
        'One of my higher conviction plays. ',
    ]
    return random.choice(contexts)


def get_contrarian_take() -> str:
    """Get occasional contrarian/opinion modifier"""
    takes = [
        '',  # Most common - no contrarian take
        '',
        '',
        '',
        '',
        '',
        'Everyone seems bearish on this but I like it. ',
        'Unpopular opinion but ',
        'Going against the grain here but ',
        'Not sure why people are sleeping on this. ',
        'Contrarian take: ',
    ]
    return random.choice(takes)


def get_uncertainty_phrase() -> str:
    """Get occasional admission of uncertainty - keeps it real"""
    phrases = [
        '',  # Most common - no uncertainty
        '',
        '',
        '',
        '',
        '',
        'Could be wrong but ',
        'Not 100% sure but ',
        'Might regret this but ',
        'Could go either way from here but ',
        'Take this with a grain of salt but ',
        'Just my read on it, ',
    ]
    return random.choice(phrases)


def get_followup_style() -> str:
    """Get occasional update/follow-up style intro"""
    styles = [
        '',  # Most common - not a follow-up
        '',
        '',
        '',
        '',
        '',
        '',
        'Update: ',
        'Quick update on this one - ',
        'Following up on this - ',
        'Still watching this and ',
    ]
    return random.choice(styles)


async def get_market_sentiment() -> Dict[str, str]:
    """Get current market sentiment based on BTC for tone adjustment"""
    try:
        import ccxt
        exchange = ccxt.binance({'enableRateLimit': True})
        btc = await exchange.fetch_ticker('BTC/USDT')
        await exchange.close()
        
        change = btc.get('percentage', 0) or 0
        
        if change >= 5:
            return {
                'condition': 'pumping',
                'tone': 'confident',
                'phrases': ['Market is cooking today.', 'Bulls in full control.', 'Good day to be in crypto.', '']
            }
        elif change >= 2:
            return {
                'condition': 'bullish',
                'tone': 'optimistic',
                'phrases': ['Solid day for the market.', 'Green across the board.', '']
            }
        elif change >= -2:
            return {
                'condition': 'neutral',
                'tone': 'balanced',
                'phrases': ['', '', '']  # No special phrase for neutral
            }
        elif change >= -5:
            return {
                'condition': 'bearish',
                'tone': 'cautious',
                'phrases': ['Choppy day.', 'Markets taking a breather.', 'Red day but weve seen worse.', '']
            }
        else:
            return {
                'condition': 'dumping',
                'tone': 'measured',
                'phrases': ['Rough day out there.', 'Blood in the streets.', 'This too shall pass.', '']
            }
    except:
        return {'condition': 'neutral', 'tone': 'balanced', 'phrases': ['']}


def _pick_tweet_length() -> str:
    """Pick a random tweet length category for variety"""
    weights = [25, 40, 25, 10]
    return random.choices(['short', 'medium', 'long', 'ultra_short'], weights=weights, k=1)[0]


def _pick_personality() -> Dict:
    """Pick a random writing personality for tweet generation"""
    personalities = [
        {
            'name': 'chill_trader',
            'voice': 'Relaxed, unbothered. Uses lowercase energy, casual punctuation. Talks like texting a friend about markets. Sometimes drops words. Never excited.',
            'examples': [
                "honestly ${symbol} been on my screen all day. {change:.1f}% up and I keep going back and forth on adding. probably will. probably shouldnt",
                "${symbol} vibing at {price_str}. no drama no fomo just steady green",
                "${symbol}. chill",
            ],
        },
        {
            'name': 'dry_wit',
            'voice': 'Sarcastic, self-deprecating, deadpan. Makes fun of themselves AND crypto culture. Dark humor about losses. Never uses exclamation marks.',
            'examples': [
                "me: im done checking charts for today\nalso me at 2am: hmm ${symbol} looks interesting at {price_str}\n\nthis is a disease",
                "sold ${symbol} last week. naturally",
                "my portfolio management strategy is basically just vibes and ${symbol} is giving good vibes rn. up {change:.1f}%. scientific method",
            ],
        },
        {
            'name': 'chart_nerd',
            'voice': 'Technical analysis focused but conversational, not robotic. Talks structure, levels, confluences. Gets genuinely excited about clean setups. Doesnt use jargon to show off, uses it because thats how they think.',
            'examples': [
                "${symbol} just reclaimed the 21 EMA on the hourly after a clean retest. RSI resetting from {rsi:.0f}. this is textbook and I rarely say that. watching {price_str} as the level",
                "structure on ${symbol} is actually really clean right now. higher lows, volume expanding on the pushes, contracting on pullbacks. {price_str}. thats what you want to see",
                "${symbol} RSI at {rsi:.0f}. noted",
            ],
        },
        {
            'name': 'old_head',
            'voice': 'Experienced. Has seen multiple cycles. Measured, patient. References past experience without being preachy. Slightly tired of the noise but still loves the game. Sometimes gives genuine wisdom.',
            'examples': [
                "been doing this since 2017 and ${symbol} at {price_str} reminds me of those early days when a {change:.1f}% move was just tuesday. now everyone screenshots it. different times",
                "${symbol} up {change:.1f}% and my timeline is acting like this has never happened before. it has. many times. relax",
                "the ones who survive this game are the ones who size properly. ${symbol} looks great but I still wouldnt put more than 5% on it",
            ],
        },
        {
            'name': 'night_owl',
            'voice': 'Late night energy. Contemplative, slightly unhinged from sleep deprivation. Vulnerable about the lifestyle. Mix of humor and genuine reflection.',
            'examples': [
                "its 3am and im looking at ${symbol} again\n\nup {change:.1f}%\n\nI should sleep\n\nbut what if it keeps going",
                "theres something about checking charts at night that hits different. ${symbol} at {price_str} under dim lighting feels like insider info even tho its public data",
                "${symbol} pumping while normal people sleep. story of my life",
            ],
        },
        {
            'name': 'minimalist',
            'voice': 'Ultra brief. Maximum impact minimum words. Period after short phrases. Cool detachment. Sometimes just a ticker and a number.',
            'examples': [
                "${symbol}. interesting",
                "{change:.1f}%",
                "${symbol} at {price_str}. thats it. thats the tweet",
            ],
        },
        {
            'name': 'storyteller',
            'voice': 'Narrative driven. Sets scenes, creates tension, resolves. Writes longer posts that read like a mini blog entry. Personal anecdotes mixed with market data. The kind of post people screenshot and share.',
            'examples': [
                "three weeks ago I added ${symbol} to a watchlist I have on my phone that I check every morning before coffee. it was doing nothing. flat. boring. the kind of coin you almost delete from the list.\n\ntoday its up {change:.1f}% at {price_str}.\n\nthe boring ones always surprise you",
                "I remember telling someone in a group chat that ${symbol} was dead money. they said give it time. I laughed. {price_str} now. I dont laugh anymore. I listen",
            ],
        },
        {
            'name': 'pragmatist',
            'voice': 'Risk-first thinker. Always mentions position sizing, risk reward, what could go wrong. Not bearish, just realistic. The person in the group chat who says "whats your stop" when everyone is celebrating.',
            'examples': [
                "${symbol} at {price_str} is interesting but heres the thing - {change:.1f}% move already happened. if youre entering now your risk reward is different than the people who caught it early. doesnt mean its bad. just means size accordingly",
                "good setup on ${symbol}. but I still wouldnt risk more than 2% of my account on it. the best traders I know are the ones who survive, not the ones who hit home runs",
                "${symbol} looks solid. managing risk tho",
            ],
        },
        {
            'name': 'confessional',
            'voice': 'Brutally honest about their own trading. Admits mistakes, celebrates small wins without pretending theyre big. Real emotions. The trader who makes you feel less alone.',
            'examples': [
                "gonna be honest. I almost sold ${symbol} during that dip last week. had my finger on the button. closed the app instead. up {change:.1f}% now. sometimes doing nothing is the hardest trade",
                "I was wrong about ${symbol}. said it was overextended at {price_str} and it kept going. added {change:.1f}% since then. wrong is wrong. moving on",
                "my win rate this month is probably like 40%. but the wins are bigger than the losses. thats the game. ${symbol} helping today",
            ],
        },
        {
            'name': 'hype_contrarian',
            'voice': 'Goes against consensus. Called it when nobody believed. Not obnoxious about it, more amused than arrogant. Finds opportunity where others see nothing.',
            'examples': [
                "funny how ${symbol} was \"dead\" according to ct last month\n\n{price_str} now\n\nthe market doesnt care about your timeline takes",
                "while everyone was debating BTC dominance I was quietly accumulating ${symbol}. up {change:.1f}%. sometimes the play is just not doing what everyone else is doing",
            ],
        },
        {
            'name': 'stream_of_consciousness',
            'voice': 'Unfiltered thoughts as they come. Uses dashes, ellipses, run-on sentences. Like reading someones inner monologue while they scroll charts. Relatable chaos.',
            'examples': [
                "${symbol} at {price_str} - ok so do I add here or wait for a pullback - but what if the pullback never comes - last time I waited I missed a {change:.1f}% move - ok maybe small size - yeah small size feels right - maybe",
                "thinking about ${symbol}... up {change:.1f}%... chart looks decent... but I said that about the last three coins and two of them dumped... but this one has volume... but volume can lie... idk man markets are hard",
            ],
        },
        {
            'name': 'zen_trader',
            'voice': 'Philosophical and calm. Uses metaphors about patience, flow, nature. Never panics. Makes trading sound like meditation. Longer reflective posts.',
            'examples': [
                "the market will do what the market does. ${symbol} up {change:.1f}% today could be down tomorrow. the only thing you control is your risk and your reaction. {price_str} is just a number on a screen until you make it mean something",
                "${symbol} moving. no rush. the trade will present itself or it wont",
            ],
        },
        {
            'name': 'degen_reformed',
            'voice': 'Used to be reckless, now trades with discipline but still has that degen energy lurking. Funny tension between old habits and new rules.',
            'examples': [
                "old me wouldve 50x leveraged ${symbol} at {price_str} right now. new me is taking a 2% position with a stop loss. character development is boring but profitable",
                "the urge to ape ${symbol} after seeing {change:.1f}% is strong. but I have rules now. rules I made at 4am after a liquidation. those rules stay",
                "${symbol} up {change:.1f}%. the old me is screaming. the new me is journaling",
            ],
        },
        {
            'name': 'data_head',
            'voice': 'Obsessed with numbers and data patterns. Notices correlations others miss. Talks about funding rates, OI, volume profiles. But not in a boring way - genuinely fascinated.',
            'examples': [
                "interesting thing about ${symbol} right now - the {change:.1f}% move is happening on relatively low open interest expansion. that usually means theres room for more. price at {price_str}. watching",
                "${symbol} volume to price ratio is the best its looked in weeks. when I see divergence like this I pay attention. not always right but the data is the data",
            ],
        },
        {
            'name': 'weekend_trader',
            'voice': 'Casual weekend energy even on weekdays. Talks about trading as a hobby alongside normal life. Mentions coffee, walks, regular stuff. Grounded and relatable.',
            'examples': [
                "making coffee and checking ${symbol}. up {change:.1f}%. nice way to start the day honestly. {price_str}. gonna walk the dog and check again later",
                "between meetings and ${symbol} quietly up {change:.1f}% in the background. love when the portfolio works while I work. {price_str}",
            ],
        },
    ]
    return random.choice(personalities)


def _get_hashtag_style() -> str:
    """Get varied hashtag usage - sometimes none at all"""
    styles = [
        '',
        '',
        '',
        f"\n\n#Crypto",
        f"\n\n#{random.choice(['BTC', 'Altcoins', 'Trading', 'Crypto'])}",
    ]
    return random.choice(styles)


async def generate_ai_tweet(coin_data: Dict, post_type: str = "featured") -> Optional[str]:
    """Use AI to generate a unique, human-like tweet with diverse personalities and lengths"""
    try:
        symbol = coin_data.get('symbol', 'UNKNOWN')
        change = coin_data.get('change', 0)
        price = coin_data.get('price', 0)
        volume = coin_data.get('volume', 0)
        rsi = coin_data.get('rsi', 50)
        trend = coin_data.get('trend', 'neutral')
        vol_ratio = coin_data.get('vol_ratio', 1)
        
        vol_str = f"${volume/1e6:.1f}M" if volume < 1e9 else f"${volume/1e9:.1f}B" if volume else "solid"
        price_str = f"${price:,.4f}" if price < 1 else f"${price:,.2f}" if price else "unknown"
        sign = "+" if change >= 0 else ""
        
        time_ctx = get_time_context()
        day_ctx = get_day_context()
        personality = _pick_personality()
        tweet_length = _pick_tweet_length()

        tech_notes = []
        if rsi > 70: tech_notes.append("RSI overbought above 70")
        elif rsi < 30: tech_notes.append("RSI oversold below 30")
        elif rsi > 55: tech_notes.append(f"RSI at {rsi:.0f}")
        if trend == 'bullish': tech_notes.append("uptrend")
        elif trend == 'bearish': tech_notes.append("downtrend")
        if vol_ratio > 2: tech_notes.append(f"volume {vol_ratio:.1f}x avg")
        tech_context = ", ".join(tech_notes[:2]) if tech_notes else "consolidating"

        examples_str = "\n".join([f'"{e}"' for e in personality['examples']])

        if tweet_length == 'ultra_short':
            length_instruction = "Write 1 very short sentence. Under 50 characters. Bare minimum words. Like a passing thought you barely bothered to type."
            max_chars = 80
        elif tweet_length == 'short':
            length_instruction = "Write 1-2 short sentences. Under 120 characters total. Casual and brief, like a quick thought between doing other things."
            max_chars = 140
        elif tweet_length == 'long':
            length_instruction = "Write 3-5 sentences. Use line breaks between thoughts. 180-270 characters. Tell a mini story or share a real observation with some depth. This should feel like a post someone spent 30 seconds thinking about."
            max_chars = 280
        else:
            length_instruction = "Write 2-3 sentences. 100-180 characters. A complete thought but not overdone."
            max_chars = 200

        sometimes_include_data = random.random() < 0.6
        data_instruction = ""
        if not sometimes_include_data:
            data_instruction = "You do NOT need to mention the exact price or percentage. You can speak generally about the coin's movement or your feelings about it. Real traders dont always quote exact numbers."
        else:
            data_instruction = "You can mention price and/or percentage but dont force both in. Use whichever feels natural for this personality."

        opening_instruction = random.choice([
            "Start with a personal observation or feeling",
            "Start with a time reference or what you were doing",
            "Start with a general market thought that leads into the coin",
            "Start mid-thought like youre continuing a conversation",
            "Start with the ticker if it feels natural for this personality",
            "Start with 'honestly' or 'ngl' or another casual opener",
            "Start with a reference to something you did wrong or right recently",
        ])

        prompt = f"""You are a real crypto trader posting on Twitter/X. You are NOT a bot, NOT a news feed, NOT an influencer. You are a normal person who trades and sometimes shares thoughts.

YOUR PERSONALITY: {personality['name']}
{personality['voice']}

EXAMPLES OF YOUR VOICE (match this energy):
{examples_str}

COIN DATA (use naturally, dont dump all of it):
${symbol} is at {price_str}, {sign}{change:.1f}% today. Volume: {vol_str}. {tech_context}. Its {time_ctx['period']} on {day_ctx['day']}.

LENGTH: {length_instruction}
OPENING: {opening_instruction}
DATA USAGE: {data_instruction}

CRITICAL RULES - BREAK ANY OF THESE AND THE TWEET IS REJECTED:
1. NO emojis whatsoever. Zero. None. Not even one
2. NO hashtags
3. NO bullet points or lists
4. NO "not financial advice" or "NFA" or "DYOR"  
5. NO ALL CAPS words (except $TICKER format)
6. NO exclamation marks
7. NO questions asking followers to engage
8. NO promotional language ("check out", "dont miss", "huge")
9. Use $TICKER format for coin mentions
10. lowercase is fine and often preferred. dont capitalize unnecessarily
11. Imperfect grammar is fine - real people dont proofread tweets
12. Can use line breaks for longer posts to create breathing room
13. Can express ANY emotion: boredom, doubt, regret, quiet satisfaction, mild interest, exhaustion, humor
14. Sometimes dont even mention the percentage - just talk about the coin or your position
15. Never sound like youre trying to get people to buy

Write ONLY the tweet. No quotes around it. No explanation:"""

        try:
            from google import genai
            
            gemini_key = os.getenv('AI_INTEGRATIONS_GEMINI_API_KEY') or os.getenv('GEMINI_API_KEY')
            if gemini_key:
                client = genai.Client(api_key=gemini_key)
                response = await asyncio.to_thread(
                    lambda: client.models.generate_content(
                        model="gemini-2.0-flash",
                        contents=prompt
                    )
                )
                tweet = response.text.strip().strip('"').strip("'").strip('```').strip()
                tweet = tweet.replace('**', '')
                if tweet and 5 < len(tweet) <= max_chars:
                    logger.info(f"AI [{personality['name']}/{tweet_length}] for ${symbol}: {tweet[:60]}...")
                    return tweet
        except Exception as e:
            logger.warning(f"Gemini tweet generation failed: {e}")
        
        try:
            import anthropic
            
            claude_key = os.getenv('ANTHROPIC_API_KEY')
            if claude_key:
                client = anthropic.Anthropic(api_key=claude_key)
                max_tokens = 40 if tweet_length == 'ultra_short' else 80 if tweet_length == 'short' else 200 if tweet_length == 'long' else 120
                response = await asyncio.to_thread(
                    lambda: client.messages.create(
                        model="claude-sonnet-4-20250514",
                        max_tokens=max_tokens,
                        messages=[{"role": "user", "content": prompt}]
                    )
                )
                tweet = response.content[0].text.strip().strip('"').strip("'").strip('```').strip()
                tweet = tweet.replace('**', '')
                if tweet and 5 < len(tweet) <= max_chars:
                    logger.info(f"Claude [{personality['name']}/{tweet_length}] for ${symbol}: {tweet[:60]}...")
                    return tweet
        except Exception as e:
            logger.warning(f"Claude tweet generation failed: {e}")
        
        return None
        
    except Exception as e:
        logger.error(f"AI tweet generation error: {e}")
        return None


def record_global_coin_post(symbol: str):
    """Record that a coin was posted (global tracking)"""
    global GLOBAL_COIN_POSTS, GLOBAL_COIN_RESET_DATE
    
    today = datetime.utcnow().date()
    if GLOBAL_COIN_RESET_DATE != today:
        GLOBAL_COIN_POSTS = {}
        GLOBAL_COIN_RESET_DATE = today
    
    GLOBAL_COIN_POSTS[symbol] = GLOBAL_COIN_POSTS.get(symbol, 0) + 1
    logger.info(f"📊 GLOBAL: ${symbol} posted {GLOBAL_COIN_POSTS[symbol]}x today")

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
        """Fetch top gaining coins from Binance with caching fallback"""
        global _gainers_cache
        
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
            
            # Update cache on success
            if usdt_tickers:
                _gainers_cache['data'] = usdt_tickers[:100]  # Cache top 100
                _gainers_cache['timestamp'] = datetime.utcnow()
                logger.debug(f"Updated gainers cache with {len(usdt_tickers)} coins")
            
            return usdt_tickers[:limit]
            
        except Exception as e:
            logger.error(f"Failed to fetch top gainers: {e}")
            
            # Fallback to cache if available and not too old
            if _gainers_cache['data'] and _gainers_cache['timestamp']:
                cache_age = (datetime.utcnow() - _gainers_cache['timestamp']).total_seconds()
                if cache_age < 600:  # Use cache up to 10 minutes old
                    logger.info(f"Using cached gainers data ({cache_age:.0f}s old)")
                    return _gainers_cache['data'][:limit]
            
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
        """Post top gainers update with diverse formats"""
        gainers = await self.get_top_gainers_data(5)
        
        if not gainers:
            return None
        
        style = random.randint(1, 8)
        
        if style == 1:
            lines = [random.choice(["whats moving today:", "todays runners:", "green on the board:", "the movers rn:"])]
            lines.append("")
            for coin in gainers:
                lines.append(f"${coin['symbol']} +{coin['change']:.1f}%")
            tweet_text = "\n".join(lines)
        
        elif style == 2:
            top = gainers[0]
            lines = [f"${top['symbol']} leading the pack today at +{top['change']:.1f}%"]
            lines.append("")
            for coin in gainers[1:4]:
                lines.append(f"${coin['symbol']} +{coin['change']:.1f}%")
            lines.append("")
            lines.append(random.choice(["volume confirming on most of these", "momentum is real today", "some names I didnt expect in here"]))
            tweet_text = "\n".join(lines)
        
        elif style == 3:
            tweet_text = f"top 5 gainers and im only in one of them. classic\n\n"
            for coin in gainers:
                tweet_text += f"${coin['symbol']} +{coin['change']:.1f}%\n"
        
        elif style == 4:
            lines = []
            for i, coin in enumerate(gainers):
                price = coin.get('price', 0)
                p = f"${price:,.4f}" if price < 1 else f"${price:,.2f}"
                lines.append(f"${coin['symbol']}  +{coin['change']:.1f}%  {p}")
            tweet_text = "\n".join(lines)
            tweet_text += random.choice(["\n\nthe chart dont lie", "\n\nnot bad for a random tuesday", "\n\nvolume matters", ""])
        
        elif style == 5:
            tweet_text = f"five coins having a good day:\n\n"
            for coin in gainers:
                vol = coin.get('volume', 0)
                v = f"{vol/1e6:.0f}M" if vol >= 1e6 else f"{vol/1e3:.0f}K"
                tweet_text += f"${coin['symbol']} +{coin['change']:.1f}% (${v} vol)\n"
        
        elif style == 6:
            top3 = gainers[:3]
            tweet_text = f"top movers rn. ${top3[0]['symbol']} +{top3[0]['change']:.1f}%, ${top3[1]['symbol']} +{top3[1]['change']:.1f}%, ${top3[2]['symbol']} +{top3[2]['change']:.1f}%. green across the board"
        
        elif style == 7:
            tweet_text = random.choice(["scanning the charts this morning. heres whats popping:", "daily scan results:", "what caught my eye today:"]) + "\n\n"
            for coin in gainers:
                tweet_text += f"${coin['symbol']} {coin['change']:.1f}%\n"
            tweet_text += random.choice(["\nvolume is there on most of these", "\nkeeping an eye on the top 2", "\nsome interesting setups in here"])
        
        else:
            tweet_text = f"${gainers[0]['symbol']} up {gainers[0]['change']:.1f}% leading everything today. "
            tweet_text += f"also watching ${gainers[1]['symbol']} (+{gainers[1]['change']:.1f}%) and ${gainers[2]['symbol']} (+{gainers[2]['change']:.1f}%). "
            tweet_text += random.choice(["good day to be in crypto", "momentum day", "bulls eating", "charts looking clean"])
        
        tweet_text += _get_hashtag_style()
        return await self.post_tweet(tweet_text)
    
    async def post_market_summary(self) -> Optional[Dict]:
        """Post market summary with human-like variety"""
        market = await self.get_market_summary()
        
        if not market:
            return None
        
        btc_sign = "+" if market['btc_change'] >= 0 else ""
        eth_sign = "+" if market['eth_change'] >= 0 else ""
        btc_p = f"${market['btc_price']:,.0f}"
        eth_p = f"${market['eth_price']:,.0f}"
        
        if market['btc_change'] >= 5: mood = random.choice(["bulls running wild", "green everywhere", "euphoria vibes"])
        elif market['btc_change'] >= 2: mood = random.choice(["looking good", "bulls in control", "green candles"])
        elif market['btc_change'] >= 0: mood = random.choice(["quiet day", "slow grind", "nothing crazy"])
        elif market['btc_change'] >= -3: mood = random.choice(["little pullback", "some red", "slight dip"])
        else: mood = random.choice(["pain", "bears in control", "rough out there"])
        
        templates = [
            f"$BTC {btc_p} ({btc_sign}{market['btc_change']:.1f}%)\n$ETH {eth_p} ({eth_sign}{market['eth_change']:.1f}%)\n\n{mood}",
            f"market check. $BTC at {btc_p}, {btc_sign}{market['btc_change']:.1f}%. $ETH at {eth_p}, {eth_sign}{market['eth_change']:.1f}%. {mood}",
            f"{mood}. $BTC {btc_sign}{market['btc_change']:.1f}% at {btc_p}. $ETH {eth_sign}{market['eth_change']:.1f}% at {eth_p}",
            f"$BTC {btc_sign}{market['btc_change']:.1f}%\n$ETH {eth_sign}{market['eth_change']:.1f}%\n\nthats the vibe today",
            f"quick update. bitcoin at {btc_p} ({btc_sign}{market['btc_change']:.1f}%), eth at {eth_p} ({eth_sign}{market['eth_change']:.1f}%). {mood}",
            f"checked the charts. $BTC sitting at {btc_p}. $ETH at {eth_p}. {mood}",
            f"$BTC {btc_p}\n$ETH {eth_p}\n\n{random.choice(['not much else to report', 'the numbers speak', 'keeping it simple', 'let the chart talk'])}",
            f"woke up to $BTC at {btc_p} ({btc_sign}{market['btc_change']:.1f}%). $ETH at {eth_p}. {mood}",
        ]
        
        tweet_text = random.choice(templates) + _get_hashtag_style()
        return await self.post_tweet(tweet_text)
    
    async def post_top_losers(self) -> Optional[Dict]:
        """Post top losing coins with human-like variety"""
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
            
            style = random.randint(1, 6)
            
            if style == 1:
                tweet_text = random.choice(["biggest dumps today:", "pain across the board:", "red everywhere:", "who got rekt:"]) + "\n\n"
                for coin in losers:
                    tweet_text += f"${coin['symbol']} {coin['change']:.1f}%\n"
            
            elif style == 2:
                tweet_text = f"${losers[0]['symbol']} leading the dump at {losers[0]['change']:.1f}%. "
                tweet_text += f"${losers[1]['symbol']} ({losers[1]['change']:.1f}%) and ${losers[2]['symbol']} ({losers[2]['change']:.1f}%) not far behind. "
                tweet_text += random.choice(["rough day", "catching knives is a hobby", "patience", "blood in the streets"])
            
            elif style == 3:
                tweet_text = f"coins having a terrible day:\n\n"
                for coin in losers:
                    p = f"${coin['price']:,.4f}" if coin['price'] < 1 else f"${coin['price']:,.2f}"
                    tweet_text += f"${coin['symbol']} {coin['change']:.1f}% ({p})\n"
                tweet_text += random.choice(["\nnot catching any of these yet", "\nwaiting for capitulation", "\npatience > fomo"])
            
            elif style == 4:
                tweet_text = f"${losers[0]['symbol']} down {abs(losers[0]['change']):.1f}% and everyone wants to buy the dip. maybe wait for it to stop dipping first"
            
            elif style == 5:
                lines = []
                for coin in losers:
                    lines.append(f"${coin['symbol']}  {coin['change']:.1f}%")
                tweet_text = "\n".join(lines)
                tweet_text += "\n\n" + random.choice(["the market gives and the market takes", "sometimes the chart just says no", "red days build character"])
            
            else:
                tweet_text = f"today in pain:\n\n"
                for coin in losers[:3]:
                    tweet_text += f"${coin['symbol']} {coin['change']:.1f}%\n"
                tweet_text += random.choice(["\nnot great", "\ncould be worse I guess", "\nthe chart will recover or it wont"])
            
            tweet_text += _get_hashtag_style()
            return await self.post_tweet(tweet_text)
            
        except Exception as e:
            logger.error(f"Failed to post top losers: {e}")
            return None
    
    async def post_btc_update(self) -> Optional[Dict]:
        """Post BTC update with human-like variety"""
        try:
            exchange = ccxt.binance({'enableRateLimit': True})
            btc = await exchange.fetch_ticker('BTC/USDT')
            ohlcv = await exchange.fetch_ohlcv('BTC/USDT', '1h', limit=24)
            await exchange.close()
            
            price = btc['last']
            change = btc['percentage'] or 0
            high = btc['high']
            low = btc['low']
            volume = btc['quoteVolume'] or 0
            sign = "+" if change >= 0 else ""
            
            if ohlcv and len(ohlcv) >= 14:
                closes = [c[4] for c in ohlcv]
                gains = [max(0, closes[i] - closes[i-1]) for i in range(1, len(closes))]
                losses = [max(0, closes[i-1] - closes[i]) for i in range(1, len(closes))]
                avg_gain = sum(gains[-14:]) / 14
                avg_loss = sum(losses[-14:]) / 14 or 0.0001
                rsi = 100 - (100 / (1 + avg_gain / avg_loss))
            else:
                rsi = 50
            
            range_size = high - low if high > low else 1
            position_in_range = (price - low) / range_size * 100
            
            if change >= 5: mood = random.choice(["$BTC woke up and chose violence", "bulls not playing today", "not a drill"])
            elif change >= 2: mood = random.choice(["solid day for the king", "green candles doing their thing", "bulls quietly taking control"])
            elif change >= 0: mood = random.choice(["quiet grind up", "nothing crazy just steady", "boring day is a good day"])
            elif change >= -3: mood = random.choice(["small dip not panicking", "healthy pullback tbh", "bears trying something"])
            else: mood = random.choice(["rough one ngl", "oof", "this too shall pass", "pain but weve seen worse"])
            
            if rsi >= 70: rsi_note = random.choice(["RSI running hot", "overbought territory", "extended here"])
            elif rsi <= 30: rsi_note = random.choice(["oversold levels", "could bounce from here", "RSI bottoming"])
            else: rsi_note = random.choice(["RSI looks balanced", "room to move either way", "healthy RSI"])
            
            vol_str = f"${volume/1e9:.1f}B" if volume >= 1e9 else f"${volume/1e6:.0f}M"
            
            templates = [
                f"$BTC at ${price:,.0f}. {sign}{change:.1f}% on the day. {mood}. {rsi_note}",
                f"bitcoin sitting at ${price:,.0f} right now. {sign}{change:.1f}%. {mood}",
                f"$BTC ranging between ${low:,.0f} and ${high:,.0f} today. currently at ${price:,.0f}. {rsi_note}",
                f"$BTC {sign}{change:.1f}% at ${price:,.0f}. {mood}. not much else to add",
                f"checked $BTC. ${price:,.0f}. {sign}{change:.1f}%. {rsi_note}. the chart speaks for itself",
                f"ngl I like where $BTC is sitting at ${price:,.0f}. {sign}{change:.1f}%. {rsi_note}",
                f"$BTC ${price:,.0f} with about {vol_str} in volume today. {mood}",
                f"bitcoin at ${price:,.0f}, {sign}{change:.1f}% today. {rsi_note}. {mood}",
                f"$BTC doing $BTC things at ${price:,.0f}. {sign}{change:.1f}% move. {rsi_note}",
                f"the king sits at ${price:,.0f}. {sign}{change:.1f}%. {mood}. patience",
            ]
            
            tweet_text = random.choice(templates) + _get_hashtag_style()
            return await self.post_tweet(tweet_text)
            
        except Exception as e:
            logger.error(f"Failed to post BTC update: {e}")
            return None
    
    async def post_altcoin_movers(self) -> Optional[Dict]:
        """Post altcoin movements with human-like variety"""
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
            
            green_count = sum(1 for c in top_movers if c['change'] >= 0)
            
            style = random.randint(1, 6)
            
            if style == 1:
                tweet_text = random.choice(["altcoins doing things today:", "what the alts are up to:", "alt check-in:"]) + "\n\n"
                for coin in top_movers[:5]:
                    s = "+" if coin['change'] >= 0 else ""
                    tweet_text += f"${coin['symbol']} {s}{coin['change']:.1f}%\n"
            
            elif style == 2:
                top3 = top_movers[:3]
                tweet_text = f"${top3[0]['symbol']} leading the alts at {'+' if top3[0]['change'] >= 0 else ''}{top3[0]['change']:.1f}%. "
                tweet_text += f"also seeing ${top3[1]['symbol']} ({'+' if top3[1]['change'] >= 0 else ''}{top3[1]['change']:.1f}%) "
                tweet_text += f"and ${top3[2]['symbol']} ({'+' if top3[2]['change'] >= 0 else ''}{top3[2]['change']:.1f}%)"
            
            elif style == 3:
                if green_count > 3:
                    tweet_text = "green across the altcoin board today\n\n"
                elif green_count < 2:
                    tweet_text = "rough day for alts\n\n"
                else:
                    tweet_text = "mixed bag in altland\n\n"
                for coin in top_movers[:4]:
                    s = "+" if coin['change'] >= 0 else ""
                    tweet_text += f"${coin['symbol']} {s}{coin['change']:.1f}%\n"
            
            elif style == 4:
                biggest = top_movers[0]
                s = "+" if biggest['change'] >= 0 else ""
                tweet_text = f"${biggest['symbol']} making noise at {s}{biggest['change']:.1f}% today. "
                tweet_text += random.choice(["rest of the alts trying to keep up", "leading the pack", "momentum is real"])
            
            elif style == 5:
                tweet_text = "alt scan:\n\n"
                for coin in top_movers[:5]:
                    s = "+" if coin['change'] >= 0 else ""
                    v = f"${coin['volume']/1e6:.0f}M"
                    tweet_text += f"${coin['symbol']}  {s}{coin['change']:.1f}%  ({v})\n"
                tweet_text += random.choice(["\nvolume matters", "\nfollow the money", "\nmomentum day"])
            
            else:
                tweet_text = f"top alt movers rn. "
                for coin in top_movers[:3]:
                    s = "+" if coin['change'] >= 0 else ""
                    tweet_text += f"${coin['symbol']} {s}{coin['change']:.1f}%, "
                tweet_text = tweet_text.rstrip(", ") + ". " + random.choice(["interesting day", "keeping watch", "something brewing"])
            
            tweet_text += _get_hashtag_style()
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
        
        clean_symbol = symbol.replace('/USDT', '').replace('USDT', '')
        tweet_text = f"""{direction_emoji} ${clean_symbol} signal

Entry around ${entry:.4f}, targeting ${tp:.4f} for about {tp_pct:.1f}% upside. Stop at ${sl:.4f}. Confidence: {confidence}/10. Not financial advice."""
        
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
            logger.debug(f"Chart analysis failed for ${symbol}: {e}")
            return {}
    
    async def _generate_varied_featured_tweet(self, symbol: str, price: float, price_str: str, 
                                               change: float, sign: str, vol_str: str, 
                                               analysis: Dict) -> str:
        """Generate highly varied tweets with AI personalities and diverse templates"""
        
        if random.random() < 0.85:
            try:
                vol_num = 0
                if vol_str:
                    vol_clean = vol_str.replace('$', '').replace(',', '')
                    if 'B' in vol_clean:
                        vol_num = float(vol_clean.replace('B', '')) * 1e9
                    elif 'M' in vol_clean:
                        vol_num = float(vol_clean.replace('M', '')) * 1e6
                    else:
                        try: vol_num = float(vol_clean)
                        except: pass
                
                coin_data = {
                    'symbol': symbol, 'change': change, 'price': price, 'volume': vol_num,
                    'rsi': analysis.get('rsi', 50), 'trend': analysis.get('trend', 'neutral'),
                    'vol_ratio': analysis.get('vol_ratio', 1),
                }
                ai_tweet = await generate_ai_tweet(coin_data, "featured")
                if ai_tweet:
                    return ai_tweet + _get_hashtag_style()
            except Exception as e:
                logger.debug(f"AI tweet generation failed, using template: {e}")
        
        rsi = analysis.get('rsi', 50)
        trend = analysis.get('trend', 'neutral')
        vol_ratio = analysis.get('vol_ratio', 1)
        
        tweet_length = _pick_tweet_length()

        ultra_short = [
            f"${symbol}. interesting",
            f"${symbol} up {change:.1f}%. noted",
            f"${symbol} moving",
            f"${symbol} at {price_str}. hm",
            f"watching ${symbol}",
            f"${symbol}. {change:.1f}%. ok then",
            f"${symbol} doing things",
        ]

        short = [
            f"${symbol} up {change:.1f}%. chart looks decent",
            f"been holding ${symbol} and for once its working. {price_str}",
            f"${symbol} quietly doing its thing. {sign}{change:.1f}%",
            f"woke up to ${symbol} at {price_str}. pleasant surprise",
            f"${symbol} {sign}{change:.1f}% while nobody was looking",
            f"small wins. ${symbol} at {price_str}",
            f"${symbol} finally moving. about time",
            f"added to my ${symbol} watchlist early. {sign}{change:.1f}% now",
            f"${symbol} at {price_str} looking interesting ngl",
            f"cant complain about ${symbol} today. {sign}{change:.1f}%",
        ]

        medium = [
            f"${symbol} up {change:.1f}% and the structure actually looks clean. {price_str} with momentum behind it. not chasing but watching",
            f"checked ${symbol} expecting nothing and got a {change:.1f}% candle. {price_str} now. sometimes the boring ones surprise you",
            f"not gonna pretend I called this ${symbol} move. {sign}{change:.1f}% at {price_str}. just happy I was positioned",
            f"${symbol} doing {change:.1f}% while I was focused on other charts. classic case of overcomplicating things. {price_str}",
            f"pulled up ${symbol} between meetings. {sign}{change:.1f}% at {price_str}. the one chart I almost didnt check today",
            f"${symbol} at {price_str} with volume confirming. {sign}{change:.1f}% and it doesnt look extended yet. keeping my position",
            f"my patience with ${symbol} might actually be paying off. {sign}{change:.1f}% today at {price_str}. still early imo",
            f"everyone sleeping on ${symbol} and its quietly up {change:.1f}%. {price_str}. the best trades are the ones nobody talks about",
            f"sold ${symbol} too early last time. watching it do {change:.1f}% from the sidelines. lesson learned. again. {price_str}",
            f"three weeks ago nobody mentioned ${symbol} in any group chat. now its up {change:.1f}% at {price_str}. funny how that works",
        ]

        long = [
            f"been watching ${symbol} for a while now and today it finally did what I was waiting for. {sign}{change:.1f}% move to {price_str}.\n\nthe thing about this one is the volume is real. not the kind of pump that fades by lunch. keeping my position and seeing where it goes",
            f"gonna be honest about ${symbol}. almost sold it during that dip last week. had the order ready. decided to walk away from the screen instead.\n\nnow its up {change:.1f}% at {price_str}. sometimes the best trade is the one you dont make",
            f"${symbol} at {price_str} after a {sign}{change:.1f}% day.\n\nI know everyone wants to hear \"this is going to 10x\" but honestly I just like the setup. clean chart, volume there, not overextended. thats all you can ask for",
            f"something about ${symbol} today. up {change:.1f}% and theres this energy in the chart that reminds me of early 2024 setups.\n\nnot saying its the same thing. just saying I recognize the pattern. {price_str} is where we are. watching closely",
            f"interesting day for ${symbol}. the move to {price_str} was clean - {sign}{change:.1f}% on volume that actually matters.\n\nive been wrong before and ill be wrong again but this is the kind of trade where the risk reward makes sense. small size, tight stop, let it work",
        ]

        if tweet_length == 'ultra_short':
            tweet = random.choice(ultra_short)
        elif tweet_length == 'short':
            tweet = random.choice(short)
        elif tweet_length == 'long':
            tweet = random.choice(long)
        else:
            tweet = random.choice(medium)
        
        return tweet + _get_hashtag_style()
    
    async def post_featured_coin(self) -> Optional[Dict]:
        """Post featured top gainer with professional chart"""
        try:
            gainers = await self.get_top_gainers_data(20)  # Get top 20 for variety
            if not gainers:
                logger.warning("No gainers data available for featured coin")
                return None
            
            # RANDOMIZE: Shuffle top 20 gainers and pick first valid one
            shuffled_gainers = gainers.copy()
            random.shuffle(shuffled_gainers)
            
            # Pick randomly from shuffled list with good volume AND not posted today
            featured = None
            for coin in shuffled_gainers:
                symbol = coin['symbol']
                has_volume = coin.get('volume', 0) >= 5_000_000
                not_overposted = self._check_coin_cooldown(symbol, max_per_day=1)
                
                if has_volume and not_overposted:
                    featured = coin
                    logger.info(f"Selected ${symbol} randomly (not posted today, good volume)")
                    break
                elif not not_overposted:
                    logger.info(f"Skipping ${symbol} - already posted today")
            
            # Fallback to any coin not posted today (still randomized)
            if not featured:
                for coin in shuffled_gainers:
                    if self._check_coin_cooldown(coin['symbol'], max_per_day=1):
                        featured = coin
                        break
            
            if not featured:
                logger.warning("All top 20 coins already posted today")
                featured = random.choice(gainers)  # Random last resort
            
            symbol = featured['symbol']
            change = featured['change']
            price = featured['price']
            
            sign = '+' if change >= 0 else ''
            volume = featured.get('volume', 0)
            vol_str = f"${volume/1e6:.1f}M" if volume < 1e9 else f"${volume/1e9:.1f}B"
            price_str = f"${price:,.4f}" if price < 1 else f"${price:,.2f}"
            
            # Get chart analysis for more interesting posts
            chart_analysis = await self._get_chart_analysis(symbol)
            
            tweet_text = await self._generate_varied_featured_tweet(
                symbol, price, price_str, change, sign, vol_str, chart_analysis
            )
            
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
            btc_chg = market['btc_change']
            eth_chg = market['eth_change']
            btc_price = market['btc_price']
            eth_price = market['eth_price']
            
            if btc_chg >= 3:
                mood = random.choice(["good day for the bulls", "solid green across the board", "bulls showed up today", "strong day. no complaints"])
            elif btc_chg >= 0:
                mood = random.choice(["quiet day overall", "nothing crazy but we closed green", "small wins count", "steady. ill take it"])
            elif btc_chg >= -3:
                mood = random.choice(["bit of red today", "bears took a nibble", "small pullback nothing major", "light red. not worried"])
            else:
                mood = random.choice(["rough day out there", "bears won today", "pain across the board", "the kind of day you close the app"])
            
            tl = _pick_tweet_length()
            
            if tl == 'ultra_short':
                templates = [
                    f"$BTC {btc_sign}{btc_chg:.1f}%. {mood}",
                    f"end of day. $BTC {btc_sign}{btc_chg:.1f}%",
                    f"{mood}. $BTC {btc_sign}{btc_chg:.1f}%",
                ]
            elif tl == 'short':
                templates = [
                    f"thats a wrap. $BTC {btc_sign}{btc_chg:.1f}% $ETH {eth_sign}{eth_chg:.1f}%. {mood}",
                    f"end of day check. $BTC at ${btc_price:,.0f} ({btc_sign}{btc_chg:.1f}%). {mood}",
                    f"closing out the day. $BTC {btc_sign}{btc_chg:.1f}%. $ETH {eth_sign}{eth_chg:.1f}%. {mood}",
                ]
            elif tl == 'long':
                gainer_line = ""
                if gainers:
                    top = gainers[0]
                    gainer_line = f"\n\nbiggest mover today was ${top['symbol']} at +{top['change']:.1f}%. " + random.choice(["interesting one to watch tomorrow", "wonder if it holds", "late to that one but noted"])
                
                templates = [
                    f"end of day thoughts.\n\n$BTC closed at ${btc_price:,.0f} ({btc_sign}{btc_chg:.1f}%). $ETH at ${eth_price:,.0f} ({eth_sign}{eth_chg:.1f}%).\n\n{mood}. {'honestly these are the days I enjoy. no panic, no fomo, just watching the market do its thing' if abs(btc_chg) < 2 else 'volatile day but nothing that changes the thesis for me. zoom out' if abs(btc_chg) < 5 else 'these are the days that separate the tourists from the traders. stay disciplined'}{gainer_line}",
                    f"daily wrap up.\n\n$BTC {btc_sign}{btc_chg:.1f}% | $ETH {eth_sign}{eth_chg:.1f}%\n\n{mood}. tomorrows another day{gainer_line}",
                ]
            else:
                templates = [
                    f"daily recap. $BTC {btc_sign}{btc_chg:.1f}% at ${btc_price:,.0f}. $ETH {eth_sign}{eth_chg:.1f}%. {mood}",
                    f"how'd we do today. $BTC {btc_sign}{btc_chg:.1f}% $ETH {eth_sign}{eth_chg:.1f}%. {mood}. tomorrows another day",
                    f"closing thoughts. $BTC at ${btc_price:,.0f} ({btc_sign}{btc_chg:.1f}%). {mood}",
                ]
            
            tweet_text = random.choice(templates) + _get_hashtag_style()
            
            return await self.post_tweet(tweet_text)
            
        except Exception as e:
            logger.error(f"Failed to post daily recap: {e}")
            return None
    
    async def post_high_viewing(self) -> Optional[Dict]:
        """Post a high-viewing viral coin with AI-generated detailed analysis"""
        try:
            gainers = await self.get_top_gainers_data(30)
            if not gainers:
                return None
            
            # High-viewing coins: meme coins, extreme movers, high volume
            MEME_COINS = ['DOGE', 'SHIB', 'PEPE', 'FLOKI', 'BONK', 'WIF', 'MEME', 'TURBO', 
                          'NEIRO', 'BOME', 'BRETT', 'MOG', 'POPCAT', 'BABYDOGE', 'ELON', 
                          'WOJAK', 'LADYS', 'MONG', 'BOB', 'TOSHI', 'SPX', 'GROK']
            
            # Priority 1: Meme coins in the gainers list (always viral)
            meme_gainers = [g for g in gainers if g['symbol'] in MEME_COINS and g['change'] > 5]
            
            # Priority 2: Extreme movers (+20% or more) - always get attention
            extreme_movers = [g for g in gainers if g['change'] >= 20]
            
            # Priority 3: High volume coins (lots of interest)
            high_volume = [g for g in gainers if g.get('volume', 0) >= 50_000_000]
            
            # Pick the best viral coin
            viral_coin = None
            category = ""
            
            if meme_gainers:
                viral_coin = random.choice(meme_gainers)
                category = "meme"
            elif extreme_movers:
                viral_coin = random.choice(extreme_movers[:5])
                category = "extreme"
            elif high_volume:
                viral_coin = random.choice(high_volume[:5])
                category = "volume"
            else:
                # Fallback to random top 10 gainer
                viral_coin = random.choice(gainers[:10])
                category = "gainer"
            
            symbol = viral_coin['symbol']
            change = viral_coin['change']
            price = viral_coin['price']
            volume = viral_coin.get('volume', 0)
            
            # Get chart analysis for AI context
            analysis = await self._get_chart_analysis(symbol)
            
            # Try AI generation with full technical context
            tweet = None
            try:
                coin_data = {
                    'symbol': symbol,
                    'change': change,
                    'price': price,
                    'volume': volume,
                    'rsi': analysis.get('rsi', 50),
                    'trend': analysis.get('trend', 'neutral'),
                    'ema_diff': analysis.get('ema_diff', 0),
                    'vwap_diff': analysis.get('vwap_diff', 0),
                    'vol_ratio': analysis.get('vol_ratio', 1),
                }
                post_type = "meme" if category == "meme" else "high_viewing"
                ai_tweet = await generate_ai_tweet(coin_data, post_type)
                if ai_tweet:
                    tweet = ai_tweet + f"\n\n#${symbol} #Crypto"
            except Exception as e:
                logger.debug(f"AI high viewing tweet failed: {e}")
            
            if not tweet:
                price_str = f"${price:,.4f}" if price < 1 else f"${price:,.2f}"
                tl = _pick_tweet_length()
                
                if tl == 'ultra_short':
                    templates = [
                        f"${symbol}. wild",
                        f"${symbol} again. ok",
                        f"${symbol} doing ${symbol} things",
                        f"well ${symbol} is moving",
                    ]
                elif tl == 'short':
                    if category == "meme":
                        templates = [
                            f"${symbol} up {change:.1f}% because of course it is",
                            f"${symbol} outperforming my serious picks. classic meme coin behavior",
                            f"${symbol} pumping and I cant even pretend to be surprised anymore",
                            f"meme coins man. ${symbol} up {change:.1f}%",
                        ]
                    else:
                        templates = [
                            f"${symbol} up {change:.1f}%. chart looks real",
                            f"noticed ${symbol} today. {sign}{change:.1f}% at {price_str}",
                            f"${symbol} finally doing something. {sign}{change:.1f}%",
                            f"woke up to ${symbol} at {price_str}. ok then",
                        ]
                elif tl == 'long':
                    if category == "meme":
                        templates = [
                            f"I keep saying im done with meme coins and then ${symbol} goes and does {change:.1f}% in a day at {price_str}.\n\nmy portfolio is basically a comedy show at this point. the memes are outperforming the research plays. maybe thats the lesson. probably not but maybe",
                            f"${symbol} up {change:.1f}% and honestly at this point I just accept that meme coins will forever be a part of my trading life.\n\nI used to judge people who traded these. now im one of them. {price_str}. no regrets",
                        ]
                    elif category == "extreme":
                        templates = [
                            f"${symbol} just printed a {change:.1f}% day and im sitting here trying to figure out if I missed it or if theres more.\n\nthis is the hardest part of trading honestly. the move already happened. chasing feels wrong. watching from the sidelines also feels wrong.\n\n{price_str} right now",
                            f"one of those days where ${symbol} reminds you why you trade crypto.\n\n{change:.1f}% move to {price_str}. the kind of candle that makes you close your laptop, take a walk, and come back to make sure it was real",
                        ]
                    else:
                        templates = [
                            f"been keeping ${symbol} on my watchlist for a while and today it finally showed up. {sign}{change:.1f}% move to {price_str}.\n\nnothing crazy but the kind of steady move I actually trust more than the 50% pumps that dump by tomorrow",
                            f"${symbol} at {price_str} today, up {sign}{change:.1f}%.\n\nI know nobody asked but this is the kind of setup I was looking for. volume behind it, clean move, not some random spike. sometimes the market gives you what you want if you wait long enough",
                        ]
                else:
                    if category == "meme":
                        templates = [
                            f"${symbol} casually up {change:.1f}% at {price_str}. meme coins gonna meme I guess",
                            f"${symbol} doing {change:.1f}% while the \"serious\" coins consolidate. I dont make the rules",
                            f"woke up and ${symbol} decided today was the day. {sign}{change:.1f}% at {price_str}. degen plays sometimes work",
                        ]
                    elif category == "extreme":
                        templates = [
                            f"${symbol} chose chaos today. {sign}{change:.1f}% at {price_str}. extended but when its moving like this you just watch",
                            f"didnt have ${symbol} doing {change:.1f}% on my bingo card today. {price_str} with volume to back it up",
                            f"checked ${symbol} expecting nothing and got a {change:.1f}% candle instead. life comes at you fast. {price_str}",
                        ]
                    else:
                        templates = [
                            f"${symbol} at {price_str} quietly doing its thing. up {change:.1f}% while I wasnt looking",
                            f"${symbol} finally moving at {change:.1f}%. {price_str} now. patience sometimes works",
                            f"something about ${symbol} today. {sign}{change:.1f}% and the volume looks real. {price_str}",
                        ]
                tweet = random.choice(templates)
            
            return await self.post_tweet(tweet)
            
        except Exception as e:
            logger.error(f"Failed to post high viewing: {e}")
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


def migrate_env_account_to_database():
    """Migrate ccally account from environment variables to database if not exists"""
    from app.database import SessionLocal
    from app.models import TwitterAccount
    
    consumer_key = os.getenv('TWITTER_CONSUMER_KEY')
    consumer_secret = os.getenv('TWITTER_CONSUMER_SECRET')
    access_token = os.getenv('TWITTER_ACCESS_TOKEN')
    access_token_secret = os.getenv('TWITTER_ACCESS_TOKEN_SECRET')
    bearer_token = os.getenv('TWITTER_BEARER_TOKEN')
    
    if not all([consumer_key, consumer_secret, access_token, access_token_secret]):
        logger.info("No environment Twitter credentials found, skipping migration")
        return
    
    db = SessionLocal()
    try:
        existing = db.query(TwitterAccount).filter(TwitterAccount.name == 'ccally').first()
        if existing:
            logger.info("ccally account already exists in database")
            return
        
        result = add_twitter_account(
            name='ccally',
            handle='@ccally_crypto',
            consumer_key=consumer_key,
            consumer_secret=consumer_secret,
            access_token=access_token,
            access_token_secret=access_token_secret,
            bearer_token=bearer_token
        )
        
        if result.get('success'):
            logger.info("✅ Successfully migrated ccally account to database!")
            account = db.query(TwitterAccount).filter(TwitterAccount.name == 'ccally').first()
            if account:
                account.set_post_types(['featured_coin', 'market_summary', 'top_gainers', 'btc_update', 'altcoin_movers', 'quick_ta', 'daily_recap'])
                db.commit()
                logger.info("✅ Set default post types for ccally")
        else:
            logger.error(f"Failed to migrate ccally: {result.get('error')}")
            
    except Exception as e:
        logger.error(f"Error migrating env account: {e}")
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
    # (hour_utc, minute, post_type) - 20 regular posts + 4 campaign posts per day
    (0, 30, 'featured_coin'),
    (1, 45, 'early_gainer'),
    (3, 0, 'quick_ta'),
    (4, 30, 'memecoin'),
    (5, 15, 'bitunix_campaign'),    # Campaign post 1 - Asia morning
    (6, 0, 'featured_coin'),
    (7, 30, 'early_gainer'),
    (8, 30, 'quick_ta'),
    (9, 45, 'featured_coin'),
    (10, 30, 'bitunix_campaign'),   # Campaign post 2 - EU morning
    (11, 0, 'early_gainer'),
    (12, 15, 'memecoin'),
    (13, 30, 'featured_coin'),
    (14, 45, 'quick_ta'),
    (15, 30, 'early_gainer'),
    (16, 0, 'bitunix_campaign'),    # Campaign post 3 - US morning
    (16, 45, 'featured_coin'),
    (17, 30, 'memecoin'),
    (18, 45, 'early_gainer'),
    (20, 0, 'featured_coin'),
    (21, 15, 'quick_ta'),
    (22, 0, 'bitunix_campaign'),    # Campaign post 4 - US evening
    (22, 30, 'early_gainer'),
    (23, 30, 'featured_coin'),
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
        'memecoin': '🐸 Trending Memecoin',
        'quick_ta': '📊 Quick TA',
        'high_viewing': '🔥 High Viewing',
        'bitunix_campaign': '💰 Bitunix Campaign'
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
                    main_poster = get_twitter_poster()
                    posted_any = False
                    
                    # Loop through ALL accounts and post for each that has this post type
                    for account in db_accounts:
                        if not account.is_active:
                            continue
                        
                        account_post_types = account.get_post_types()
                        if post_type not in account_post_types:
                            continue
                        
                        # Create unique slot key per account to track independently
                        account_slot_key = f"{slot_key}_{account.name}"
                        if account_slot_key in POSTED_SLOTS:
                            continue
                        
                        # Add per-account random offset (±5 mins) so they don't post at exact same time
                        account_offset_key = f"{slot_key}_{account.name}_offset"
                        if account_offset_key not in SLOT_OFFSETS:
                            SLOT_OFFSETS[account_offset_key] = random.randint(0, 300)  # 0-5 min delay
                        
                        # Small delay between accounts
                        await asyncio.sleep(SLOT_OFFSETS[account_offset_key] / 60)  # Convert to fractional minutes
                        
                        result = None
                        try:
                            account_poster = get_account_poster(account)
                            result = await post_with_account(account_poster, main_poster, post_type)
                        except Exception as e:
                            logger.error(f"Error posting for {account.name}: {e}")
                            result = {'success': False, 'error': str(e)}
                        
                        if result and result.get('success'):
                            logger.info(f"✅ [{account.name}] Auto-posted {post_type} at {slot_key}")
                            POSTED_SLOTS.add(account_slot_key)
                            posted_any = True
                            await notify_admin_post_result(account.name, post_type, True)
                        else:
                            error_msg = result.get('error', 'Unknown error') if result else 'No result returned'
                            logger.warning(f"⚠️ [{account.name}] Failed to auto-post {post_type}: {error_msg}")
                            POSTED_SLOTS.add(account_slot_key)  # Mark as attempted
                            await notify_admin_post_result(account.name, post_type, False, error_msg)
                    
                    # Mark the base slot as done if any account posted
                    if posted_any:
                        POSTED_SLOTS.add(slot_key)
                    
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
        if is_social_account(account_poster.name):
            logger.info(f"[CryptoSocial] Using social-powered posts for {account_poster.name}")
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
                    logger.info(f"[MultiAccount] Skipping ${symbol} - already posted today (1x max)")
            
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
            
            symbol = featured['symbol']
            change = featured.get('change', 0)
            price = featured.get('price', 0)
            volume = featured.get('volume', 0)
            
            sign = '+' if change >= 0 else ''
            price_str = f"${price:,.4f}" if price < 1 else f"${price:,.2f}"
            vol_str = f"${volume/1e6:.1f}M" if volume < 1e9 else f"${volume/1e9:.1f}B"
            
            chart_analysis = await main_poster._get_chart_analysis(symbol)
            
            tweet_text = await main_poster._generate_varied_featured_tweet(
                symbol, price, price_str, change, sign, vol_str, chart_analysis
            )
            
            result = account_poster.post_tweet(tweet_text)
            
            # Record the coin post on success
            if result and result.get('success'):
                record_global_coin_post(symbol)
            
            return result
        
        elif post_type == 'top_gainers':
            gainers = await main_poster.get_top_gainers_data(5)
            if not gainers:
                return None
            
            # Try AI for unique intro
            ai_intro = None
            try:
                if random.random() < 0.7:
                    top_coin = gainers[0]
                    ai_tweet = await generate_ai_tweet({
                        'symbol': top_coin['symbol'],
                        'change': top_coin['change'],
                        'price': top_coin.get('price', 0),
                        'volume': top_coin.get('volume', 0)
                    }, "featured")
                    if ai_tweet and len(ai_tweet) < 100:
                        ai_intro = ai_tweet.split('\n')[0]  # Just the first line
            except:
                pass
            
            if ai_intro:
                lines = [ai_intro + "\n"]
            else:
                intros = ["Few coins caught my eye this morning.\n", "The market actually doing something for once.\n",
                          "Some names showing strength while I drink my coffee.\n", "Heres what Im looking at today.\n",
                          "Portfolio finally has some green. Heres what.\n", "Quick scan of whats moving.\n"]
                lines = [random.choice(intros)]
            
            for i, coin in enumerate(gainers, 1):
                change_sign = "+" if coin['change'] >= 0 else ""
                price = coin.get('price', 0)
                price_str = f"${price:,.4f}" if price < 1 else f"${price:,.2f}"
                lines.append(f"${coin['symbol']} {change_sign}{coin['change']:.1f}% at {price_str}")
            
            closings = ["\n\nDoing my own research on these.", "\n\nKeeping these on the watchlist.",
                        "\n\nVolume looks decent across the board.", "", "\n\nNot financial advice obviously."]
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

$BTC: ${market['btc_price']:,.0f} ({btc_sign}{market['btc_change']:.1f}%)
$ETH: ${market['eth_price']:,.0f} ({eth_sign}{market['eth_change']:.1f}%)

{mood}"""
            elif style == 2:
                tweet = f"""{mood}

$BTC sitting at ${market['btc_price']:,.0f} ({btc_sign}{market['btc_change']:.1f}%)
$ETH at ${market['eth_price']:,.0f} ({eth_sign}{market['eth_change']:.1f}%)

Watching closely from here"""
            elif style == 3:
                tweet = f"""Where we're at right now:

$BTC: ${market['btc_price']:,.0f}
$ETH: ${market['eth_price']:,.0f}

{btc_sign}{market['btc_change']:.1f}% and {eth_sign}{market['eth_change']:.1f}% respectively

{mood}"""
            elif style == 4:
                tweet = f"""Market update

$BTC: ${market['btc_price']:,.0f} | {btc_sign}{market['btc_change']:.1f}%
$ETH: ${market['eth_price']:,.0f} | {eth_sign}{market['eth_change']:.1f}%

{mood}"""
            else:
                tweet = f"""Checking in on the markets

$BTC {btc_sign}{market['btc_change']:.1f}% at ${market['btc_price']:,.0f}
$ETH {eth_sign}{market['eth_change']:.1f}% at ${market['eth_price']:,.0f}

{mood}"""
            return account_poster.post_tweet(tweet)
        
        elif post_type == 'btc_update':
            market = await main_poster.get_market_summary()
            if not market:
                return None
            
            btc_sign = "+" if market['btc_change'] >= 0 else ""
            
            # Try AI generation first
            ai_tweet = None
            try:
                if random.random() < 0.7:
                    ai_tweet = await generate_ai_tweet({
                        'symbol': 'BTC',
                        'change': market['btc_change'],
                        'price': market['btc_price'],
                        'volume': 0,
                        'trend': 'bullish' if market['btc_change'] > 0 else 'bearish'
                    }, "featured")
            except:
                pass
            
            if ai_tweet:
                return account_poster.post_tweet(ai_tweet)
            
            # Fallback to natural templates
            if market['btc_change'] >= 5:
                comment = random.choice(["Strong day for $BTC.", "Bulls showed up today.", "Solid move higher."])
            elif market['btc_change'] >= 2:
                comment = random.choice(["Decent day for $BTC.", "Grinding higher.", "Looking healthy."])
            elif market['btc_change'] >= 0:
                comment = random.choice(["Quiet day so far.", "Holding steady.", "Consolidating."])
            elif market['btc_change'] >= -3:
                comment = random.choice(["Small pullback happening.", "Testing some support here.", "Nothing too concerning."])
            else:
                comment = random.choice(["Rough day for $BTC.", "Bears in control today.", "Looking for a bounce."])
            
            templates = [
                f"$BTC trading at ${market['btc_price']:,.0f} today, {btc_sign}{market['btc_change']:.1f}% on the day. {comment}",
                f"$BTC at ${market['btc_price']:,.0f} with a {btc_sign}{market['btc_change']:.1f}% move. {comment}",
                f"Checking in on $BTC. Currently ${market['btc_price']:,.0f}, {btc_sign}{market['btc_change']:.1f}% today. {comment}",
                f"$BTC sitting at ${market['btc_price']:,.0f} right now. {btc_sign}{market['btc_change']:.1f}% change. {comment}",
            ]
            return account_poster.post_tweet(random.choice(templates))
        
        elif post_type == 'altcoin_movers':
            gainers = await main_poster.get_top_gainers_data(8)
            if not gainers:
                return None
            
            alts = [g for g in gainers if g['symbol'] not in ['BTC', 'ETH']][:4]
            if not alts:
                return None
            
            # Try AI for intro
            ai_intro = None
            try:
                if random.random() < 0.7:
                    best_alt = alts[0]
                    ai_tweet = await generate_ai_tweet({
                        'symbol': best_alt['symbol'],
                        'change': best_alt['change'],
                        'price': best_alt.get('price', 0),
                        'volume': best_alt.get('volume', 0)
                    }, "featured")
                    if ai_tweet and len(ai_tweet) < 80:
                        ai_intro = ai_tweet.split('\n')[0]
            except:
                pass
            
            if ai_intro:
                lines = [ai_intro + "\n"]
            else:
                intros = ["Few altcoins catching my attention today.\n", "Some alts showing strength while $BTC consolidates.\n",
                          "Interesting moves in the altcoin space.\n", "Keeping an eye on a few alts.\n",
                          "Altcoins worth watching today.\n"]
                lines = [random.choice(intros)]
            
            for coin in alts:
                sign = "+" if coin['change'] >= 0 else ""
                price = coin.get('price', 0)
                price_str = f"${price:,.4f}" if price < 1 else f"${price:,.2f}"
                lines.append(f"${coin['symbol']} {sign}{coin['change']:.1f}% at {price_str}")
            
            closings = ["\n\nVolume looks decent on these.", "\n\nAdding to the watchlist.", "", "\n\nNot financial advice."]
            lines.append(random.choice(closings))
            return account_poster.post_tweet("\n".join(lines))
        
        elif post_type == 'early_gainer':
            # Early gainer post for standard accounts (same as social but branded)
            return await post_early_gainer_standard(account_poster, main_poster)
        
        elif post_type == 'memecoin':
            # Trending memecoin with contract address
            return await post_memecoin(account_poster)
        
        elif post_type == 'quick_ta':
            return await post_quick_ta(account_poster, main_poster)
        
        elif post_type == 'bitunix_campaign':
            return await post_bitunix_campaign(account_poster)
        
        elif post_type == 'daily_recap':
            market = await main_poster.get_market_summary()
            gainers = await main_poster.get_top_gainers_data(3)
            
            if not market:
                return None
            
            btc_sign = "+" if market['btc_change'] >= 0 else ""
            eth_sign = "+" if market['eth_change'] >= 0 else ""
            btc_chg = market['btc_change']
            eth_chg = market['eth_change']
            btc_price = market['btc_price']
            
            if btc_chg >= 3:
                vibe = random.choice(["good day for the bulls", "solid green day", "bulls ate well today"])
            elif btc_chg >= 0:
                vibe = random.choice(["quiet day overall", "we closed green. ill take it", "nothing crazy but green is green"])
            elif btc_chg >= -3:
                vibe = random.choice(["bit of red today", "bears took a nibble", "small pullback nothing major"])
            else:
                vibe = random.choice(["rough day out there", "bears won today", "time to zoom out"])
            
            tl = _pick_tweet_length()
            
            if tl == 'ultra_short':
                templates = [
                    f"$BTC {btc_sign}{btc_chg:.1f}%. {vibe}",
                    f"{vibe}. $BTC {btc_sign}{btc_chg:.1f}%",
                ]
            elif tl == 'short':
                templates = [
                    f"thats a wrap. $BTC {btc_sign}{btc_chg:.1f}% $ETH {eth_sign}{eth_chg:.1f}%. {vibe}",
                    f"end of day. $BTC at ${btc_price:,.0f} ({btc_sign}{btc_chg:.1f}%). {vibe}",
                    f"{vibe}. $BTC {btc_sign}{btc_chg:.1f}% $ETH {eth_sign}{eth_chg:.1f}%. tomorrows another day",
                ]
            elif tl == 'long':
                gainer_bit = ""
                if gainers:
                    top = random.choice(gainers[:5])
                    gainer_bit = f"\n\nbiggest mover: ${top['symbol']} +{top['change']:.1f}%"
                templates = [
                    f"wrapping up the day.\n\n$BTC at ${btc_price:,.0f} ({btc_sign}{btc_chg:.1f}%)\n$ETH {eth_sign}{eth_chg:.1f}%\n\n{vibe}. {'no complaints here. clean price action and the structure held' if btc_chg >= 0 else 'not ideal but nothing that changes the bigger picture. patience'}{gainer_bit}",
                ]
            else:
                templates = [
                    f"daily recap. $BTC {btc_sign}{btc_chg:.1f}% $ETH {eth_sign}{eth_chg:.1f}%. {vibe}",
                    f"closing out the day. $BTC at ${btc_price:,.0f} ({btc_sign}{btc_chg:.1f}%). {vibe}",
                ]
            
            tweet = random.choice(templates) + _get_hashtag_style()
            return account_poster.post_tweet(tweet)
        
        elif post_type == 'high_viewing':
            # High viewing post - viral/trending coins
            gainers = await main_poster.get_top_gainers_data(30)
            if not gainers:
                return None
            
            MEME_COINS = ['DOGE', 'SHIB', 'PEPE', 'FLOKI', 'BONK', 'WIF', 'MEME', 'TURBO', 
                          'NEIRO', 'BOME', 'BRETT', 'MOG', 'POPCAT', 'BABYDOGE', 'ELON', 
                          'WOJAK', 'LADYS', 'MONG', 'BOB', 'TOSHI', 'SPX', 'GROK']
            
            meme_gainers = [g for g in gainers if g['symbol'] in MEME_COINS and g['change'] > 5]
            extreme_movers = [g for g in gainers if g['change'] >= 20]
            high_volume = [g for g in gainers if g.get('volume', 0) >= 50_000_000]
            
            viral_coin = None
            category = ""
            
            if meme_gainers:
                viral_coin = random.choice(meme_gainers)
                category = "meme"
            elif extreme_movers:
                viral_coin = random.choice(extreme_movers[:5])
                category = "extreme"
            elif high_volume:
                viral_coin = random.choice(high_volume[:5])
                category = "volume"
            else:
                viral_coin = random.choice(gainers[:10])
                category = "gainer"
            
            symbol = viral_coin['symbol']
            change = viral_coin['change']
            price = viral_coin['price']
            
            # Try AI generation first for unique human-like tweets
            ai_tweet = None
            ai_type = "meme" if category == "meme" else "high_viewing"
            try:
                ai_tweet = await generate_ai_tweet(viral_coin, ai_type)
            except Exception as e:
                logger.warning(f"AI tweet generation failed: {e}")
            
            if ai_tweet:
                tweet = ai_tweet + "\n\n#Crypto #Trading"
                return await account_poster.post_tweet(tweet)
            
            tl = _pick_tweet_length()
            price_str = f"${price:,.4f}" if price < 1 else f"${price:,.2f}"
            
            if tl == 'ultra_short':
                templates = [
                    f"${symbol}. yep",
                    f"${symbol} again",
                    f"well. ${symbol} is moving",
                    f"${symbol} doing its thing",
                ]
            elif tl == 'long':
                templates = [
                    f"${symbol} printed {change:.1f}% today and im genuinely unsure what to do with this information.\n\non one hand the move already happened. on the other hand volume is still flowing and the chart doesnt look exhausted yet.\n\n{price_str}. thinking out loud",
                    f"the thing about ${symbol} is everyone has an opinion now that its up {change:.1f}%. nobody was talking about it a week ago.\n\nthats how it always goes. the best trades happen in silence and the worst trades happen in hype. {price_str}",
                    f"been in ${symbol} for a minute now. watching it do {change:.1f}% today while everyone else discovers it is a weird feeling.\n\nhappy about the move but also know that this is usually when the late fomo starts. staying disciplined at {price_str}",
                ]
            elif tl == 'short':
                templates = [
                    f"${symbol} up {change:.1f}%. classic",
                    f"${symbol} at {price_str}. I see it",
                    f"ngl ${symbol} has my attention",
                    f"${symbol} doing {change:.1f}% while I was looking elsewhere",
                    f"${symbol} moving. watching",
                    f"${symbol} {sign}{change:.1f}% today. not bad at all",
                ]
            else:
                if category == "meme":
                    templates = [
                        f"${symbol} casually up {change:.1f}% outperforming everything I researched for hours. meme coins are humbling",
                        f"woke up and ${symbol} decided today was its day. {sign}{change:.1f}%. degen plays occasionally work I guess",
                        f"${symbol} doing {change:.1f}% while the \"fundamentals\" crowd sleeps. I dont make the rules",
                    ]
                elif category == "extreme":
                    templates = [
                        f"didnt have ${symbol} doing {change:.1f}% on my bingo card. {price_str} with volume to back it up. respect",
                        f"${symbol} chose chaos today. {sign}{change:.1f}% at {price_str}. extended but when its moving like this you just watch",
                        f"checked ${symbol} expecting nothing and got a {change:.1f}% candle. life is funny sometimes",
                    ]
                else:
                    templates = [
                        f"${symbol} quietly up {change:.1f}% while I wasnt paying attention. {price_str}. sometimes the boring ones win",
                        f"${symbol} finally moving at {change:.1f}%. been on my list for a while. {price_str} now",
                        f"something about ${symbol} today. {sign}{change:.1f}% and the volume actually looks real. {price_str}",
                    ]
            
            tweet = random.choice(templates)
            return await account_poster.post_tweet(tweet)
        
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
    """Post coins gaining traction early with human-like variety"""
    try:
        exchange = ccxt.binance({'enableRateLimit': True})
        tickers = await exchange.fetch_tickers()
        await exchange.close()
        
        early_movers = []
        for symbol, data in tickers.items():
            if not symbol.endswith('/USDT') or not data.get('percentage'):
                continue
            
            base = symbol.replace('/USDT', '')
            if base in ['USDC', 'BUSD', 'DAI', 'TUSD', 'USDP', 'USDD']:
                continue
            
            change = data['percentage']
            volume = data.get('quoteVolume', 0)
            
            if 3 <= change <= 12 and volume >= 5_000_000 and check_global_coin_cooldown(base, max_per_day=1):
                early_movers.append({
                    'symbol': base,
                    'change': change,
                    'volume': volume,
                    'price': data.get('last', 0)
                })
        
        if not early_movers:
            return None
        
        early_movers.sort(key=lambda x: x['change'])
        featured_coin = early_movers[0]
        coin = featured_coin
        vol_str = f"${coin['volume']/1e6:.1f}M" if coin['volume'] < 1e9 else f"${coin['volume']/1e9:.1f}B"
        
        tl = _pick_tweet_length()
        sym = coin['symbol']
        chg = coin['change']
        
        if tl == 'ultra_short':
            templates = [
                f"${sym} early. watching",
                f"${sym} starting to move",
                f"${sym} on my radar",
                f"${sym}. early stages",
            ]
        elif tl == 'short':
            templates = [
                f"${sym} quietly up {chg:.1f}%. something might be brewing",
                f"noticing ${sym} starting to move. +{chg:.1f}% on decent volume",
                f"${sym} +{chg:.1f}% and nobody is talking about it yet",
                f"found ${sym} early. +{chg:.1f}% with {vol_str} volume behind it",
                f"${sym} waking up. {chg:.1f}% so far",
            ]
        elif tl == 'long':
            templates = [
                f"spotted ${sym} on my scanner this morning at +{chg:.1f}% with {vol_str} volume.\n\nthe thing about early movers is you never know if its the start of something or a one day pop. but the volume profile on this one looks different. not the usual pump and dump pattern.\n\nkeeping a close eye",
                f"${sym} is one of those coins that hasnt made anyones list yet. +{chg:.1f}% today on real volume.\n\nI like finding these before twitter starts talking about them. sometimes they fade. sometimes they 5x from here. either way im watching",
            ]
        else:
            templates = [
                f"${sym} gaining steam at +{chg:.1f}%. early stages but {vol_str} volume looks real. the kind of move I like to catch before it trends",
                f"${sym} +{chg:.1f}% and building momentum. not making headlines yet which is usually a good sign",
                f"under the radar today: ${sym} up {chg:.1f}% on {vol_str} volume. watching this one closely",
                f"found ${sym} on my morning scan. +{chg:.1f}% with volume confirming. early but the setup is there",
            ]
        
        if len(early_movers) >= 3 and random.random() < 0.3:
            list_templates = [
                f"early movers catching my eye:\n\n" + "\n".join([f"${m['symbol']} +{m['change']:.1f}%" for m in early_movers[:4]]) + "\n\nstill early on all of these",
                f"coins starting to move this morning:\n\n" + "\n".join([f"${m['symbol']} +{m['change']:.1f}%" for m in early_movers[:3]]) + "\n\nvolume looks real",
            ]
            templates = list_templates
        
        tweet_text = random.choice(templates) + _get_hashtag_style()
        result = account_poster.post_tweet(tweet_text)
        
        if result and result.get('success') and featured_coin:
            record_global_coin_post(featured_coin['symbol'])
            logger.info(f"[CryptoSocial] Recorded {featured_coin['symbol']} to global cooldown")
        
        return result
        
    except Exception as e:
        logger.error(f"Error posting early gainers: {e}")
        return None


async def post_momentum_shift(account_poster: MultiAccountPoster) -> Optional[Dict]:
    """Post coins showing momentum shifts with human-like variety"""
    try:
        exchange = ccxt.binance({'enableRateLimit': True})
        tickers = await exchange.fetch_tickers()
        await exchange.close()
        
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
        featured_coin = movers[0]
        coin = featured_coin
        vol_str = f"${coin['volume']/1e6:.1f}M" if coin['volume'] < 1e9 else f"${coin['volume']/1e9:.1f}B"
        
        tl = _pick_tweet_length()
        sym = coin['symbol']
        chg = coin['change']
        
        if tl == 'ultra_short':
            templates = [
                f"${sym} has momentum",
                f"${sym} is going",
                f"${sym}. legit move",
            ]
        elif tl == 'short':
            templates = [
                f"${sym} up {chg:.1f}% and the momentum looks real",
                f"cant ignore ${sym} anymore. +{chg:.1f}%",
                f"${sym} woke up. {chg:.1f}% on serious volume",
                f"momentum shift on ${sym}. +{chg:.1f}% move",
                f"${sym} gaining traction. {chg:.1f}% and still going",
            ]
        elif tl == 'long':
            templates = [
                f"${sym} up {chg:.1f}% and this is the kind of momentum thats hard to fake.\n\n{vol_str} volume behind it. not a random pump on no volume. when you see this kind of conviction from the market it usually means someone knows something or at least thinks they do.\n\nwatching for continuation",
                f"been tracking momentum shifts all day and ${sym} stands out. {chg:.1f}% move on {vol_str} volume.\n\nmost of the other movers are fading by now. this one is holding. thats the difference between a pump and a real move",
            ]
        else:
            templates = [
                f"${sym} up {chg:.1f}% and the momentum is real. {vol_str} volume backing it up. not the kind of move that fades easily",
                f"momentum on ${sym} looking legit. +{chg:.1f}% with volume confirming at {vol_str}. keeping this one on my screen",
                f"${sym} building momentum. {chg:.1f}% so far on {vol_str} volume. these are the moves I pay attention to",
            ]
        
        if len(movers) >= 3 and random.random() < 0.25:
            templates = [
                f"momentum today:\n\n" + "\n".join([f"${m['symbol']} +{m['change']:.1f}%" for m in movers[:3]]) + "\n\nvolume is there on all of them",
                f"these are actually moving:\n\n" + "\n".join([f"${m['symbol']} +{m['change']:.1f}%" for m in movers[:3]]) + "\n\nnot just pump and dumps either. volume is real",
            ]
        
        tweet_text = random.choice(templates) + _get_hashtag_style()
        
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
        
        coin = high_volume[0]
        vol_str = f"${coin['volume']/1e6:.0f}M" if coin['volume'] < 1e9 else f"${coin['volume']/1e9:.1f}B"
        sign = "+" if coin['change'] >= 0 else ""
        sym = coin['symbol']
        chg = coin['change']
        
        tl = _pick_tweet_length()
        
        if tl == 'ultra_short':
            templates = [
                f"${sym} volume is wild today",
                f"${sym}. {vol_str} volume. ok",
                f"money flowing into ${sym}",
            ]
        elif tl == 'short':
            templates = [
                f"${sym} doing {vol_str} in volume today. something is happening",
                f"serious volume on ${sym}. {vol_str}. worth watching",
                f"${sym} volume spiking. {vol_str}. either smart money or fomo. either way paying attention",
                f"${sym} {vol_str} volume today. {sign}{chg:.1f}%. the volume tells the story",
            ]
        elif tl == 'long':
            templates = [
                f"the volume on ${sym} today is what got my attention. {vol_str} flowing through while most people are focused elsewhere.\n\nvol precedes price. doesnt always mean up, but it means something is happening. {sign}{chg:.1f}% so far. this is the kind of activity I watch for",
                f"${sym} at {vol_str} in daily volume.\n\nfor context thats significantly above its average. when you see this kind of volume expansion it usually means either institutions are positioning or retail found a narrative.\n\neither way its {sign}{chg:.1f}% and I have my eye on it",
            ]
        else:
            templates = [
                f"${sym} doing {vol_str} in volume with a {sign}{chg:.1f}% move. the kind of activity that catches my attention",
                f"volume on ${sym} is elevated today. {vol_str}. {sign}{chg:.1f}%. keeping this on my screen",
                f"when volume spikes like this on ${sym} it usually means something. {vol_str} today. watching closely",
            ]
        
        if len(high_volume) >= 3 and random.random() < 0.2:
            templates = [
                f"biggest volume today:\n\n" + "\n".join([f"${c['symbol']} - {c['volume']/1e6:.0f}M vol ({'+' if c['change']>=0 else ''}{c['change']:.1f}%)" for c in high_volume[:4]]) + "\n\nwhen this much money moves its worth paying attention",
            ]
        
        tweet_text = random.choice(templates) + _get_hashtag_style()
        
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
        
        btc_sign = "+" if btc_change >= 0 else ""
        eth_sign = "+" if eth_change >= 0 else ""
        
        if green_pct >= 70: mood = random.choice(["green everywhere today", "bulls running the show", "good day across the board"])
        elif green_pct >= 55: mood = random.choice(["leaning green today", "more green than red", "bulls have a slight edge"])
        elif green_pct >= 45: mood = random.choice(["mixed signals today", "no clear direction", "choppy across the board"])
        elif green_pct >= 30: mood = random.choice(["more red than green", "bears have the edge today", "leaning red"])
        else: mood = random.choice(["sea of red today", "rough day out there", "bears in full control"])
        
        tl = _pick_tweet_length()
        
        if tl == 'ultra_short':
            templates = [
                f"$BTC {btc_sign}{btc_change:.1f}%. {mood}",
                f"market is {'green' if green_pct >= 55 else 'red' if green_pct < 45 else 'mixed'}. $BTC {btc_sign}{btc_change:.1f}%",
                f"$BTC {btc_sign}{btc_change:.1f}%. moving on",
            ]
        elif tl == 'short':
            templates = [
                f"$BTC {btc_sign}{btc_change:.1f}%. $ETH {eth_sign}{eth_change:.1f}%. {mood}",
                f"market check: {mood}. $BTC at {btc_sign}{btc_change:.1f}%. {green_pct:.0f}% of coins green",
                f"$BTC {btc_sign}{btc_change:.1f}% today. {mood}. not much else to say",
                f"{mood}. $BTC {btc_sign}{btc_change:.1f}% $ETH {eth_sign}{eth_change:.1f}%",
            ]
        elif tl == 'long':
            templates = [
                f"end of day look at the market.\n\n$BTC {btc_sign}{btc_change:.1f}%\n$ETH {eth_sign}{eth_change:.1f}%\n\n{green_pct:.0f}% of coins closed green today. {mood}.\n\n{'the kind of day where you do nothing and feel good about it' if abs(btc_change) < 2 else 'interesting moves but nothing that changes the bigger picture' if abs(btc_change) < 5 else 'volatile day. these are the days that test your conviction'}",
                f"market update and some thoughts.\n\n$BTC sitting at {btc_sign}{btc_change:.1f}% on the day. $ETH at {eth_sign}{eth_change:.1f}%. {mood} with {green_pct:.0f}% green.\n\nhonestly these are the days I just observe. no need to force trades when the market is {'this choppy' if 40 <= green_pct <= 60 else 'this one-sided' if green_pct > 70 or green_pct < 30 else 'doing its thing'}",
            ]
        else:
            templates = [
                f"quick market check. $BTC {btc_sign}{btc_change:.1f}%, $ETH {eth_sign}{eth_change:.1f}%. {mood}. {green_pct:.0f}% of coins green",
                f"$BTC {btc_sign}{btc_change:.1f}% $ETH {eth_sign}{eth_change:.1f}%. {mood}. the market does what it wants",
                f"market vibes: {mood}. $BTC {btc_sign}{btc_change:.1f}%. about {green_pct:.0f}% of alts in the green",
            ]
        
        tweet_text = random.choice(templates) + _get_hashtag_style()
        
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
        
        tl = _pick_tweet_length()
        
        if tl == 'ultra_short':
            templates = [
                f"${symbol}. early",
                f"${symbol} starting to move",
                f"${symbol} on my scanner",
                f"${symbol}. {change:.1f}%. watching",
            ]
        elif tl == 'short':
            templates = [
                f"${symbol} catching my attention. +{change:.1f}% on real volume",
                f"${symbol} waking up. {change:.1f}% so far at {price_str}",
                f"${symbol} quietly up {change:.1f}%. not a lot of people talking about this yet",
                f"spotted ${symbol} early. +{change:.1f}% with {vol_str} behind it",
                f"${symbol} +{change:.1f}% at {price_str}. interesting",
            ]
        elif tl == 'long':
            templates = [
                f"${symbol} showed up on my scanner this morning and I almost scrolled past it. +{change:.1f}% on {vol_str} volume at {price_str}.\n\nbut something about the price action made me stop. the move is steady, not spiky. volume is distributed evenly not concentrated in one candle. thats usually what real accumulation looks like.\n\nnot saying its a lock but its on my list now",
                f"found ${symbol} early today at {price_str}. up {change:.1f}% with {vol_str} volume.\n\nthe thing I like about early movers like this is the risk reward is still favorable. you haven't missed the bulk of the move yet and if it fails you know quickly. thats the kind of trade I want",
            ]
        else:
            templates = [
                f"spotted ${symbol} early today. up {change:.1f}% with {vol_str} volume at {price_str}. the kind of move I like to catch before it trends",
                f"${symbol} catching my attention. +{change:.1f}% on {vol_str} volume. still early if momentum holds. {price_str}",
                f"been watching ${symbol} for a bit. finally making its move at {change:.1f}% with volume confirming. {price_str}",
                f"interesting price action on ${symbol}. up {change:.1f}% on real volume. {price_str}. on the watchlist now",
            ]
        
        tweet_text = random.choice(templates) + _get_hashtag_style()
        
        result = account_poster.post_tweet(tweet_text)
        
        if result and result.get('success'):
            record_global_coin_post(symbol)
        
        return result
        
    except Exception as e:
        logger.error(f"Error posting early gainer standard: {e}")
        return {'success': False, 'error': str(e)}


async def post_memecoin(account_poster: MultiAccountPoster) -> Optional[Dict]:
    """Post Solana memecoin using DexScreener's trending data"""
    import httpx
    
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'application/json',
        }
        
        candidates = []
        
        async with httpx.AsyncClient(timeout=15.0, headers=headers) as client:
            # Method 1: Get boosted/trending tokens from DexScreener (Solana only)
            try:
                resp = await client.get('https://api.dexscreener.com/token-boosts/top/v1')
                if resp.status_code == 200:
                    data = resp.json()
                    for token in data[:30]:
                        if token.get('chainId') == 'solana':
                            ca = token.get('tokenAddress', '')
                            if ca:
                                is_pump = 'pump' in ca.lower() or 'pump.fun' in str(token.get('links', []))
                                candidates.append({
                                    'ca': ca,
                                    'name': '',
                                    'symbol': '',
                                    'market_cap': 0,
                                    'complete': True,
                                    'source': 'boosted',
                                    'is_pump': is_pump
                                })
            except Exception as e:
                logger.debug(f"DexScreener boosts fetch failed: {e}")
            
            # Method 2: Search for pump.fun tokens specifically
            try:
                resp = await client.get('https://api.dexscreener.com/latest/dex/search?q=solana%20pump')
                if resp.status_code == 200:
                    data = resp.json()
                    pairs = data.get('pairs', [])
                    for pair in pairs[:20]:
                        if pair.get('chainId') == 'solana':
                            ca = pair.get('baseToken', {}).get('address', '')
                            fdv = float(pair.get('fdv', 0) or 0)
                            volume = float(pair.get('volume', {}).get('h24', 0) or 0)
                            if ca and fdv >= 50000 and volume >= 10000 and ca not in [c['ca'] for c in candidates]:
                                candidates.append({
                                    'ca': ca,
                                    'name': pair.get('baseToken', {}).get('name', ''),
                                    'symbol': pair.get('baseToken', {}).get('symbol', ''),
                                    'market_cap': fdv,
                                    'complete': True,
                                    'source': 'search',
                                    'is_pump': 'pump' in ca.lower()
                                })
            except Exception as e:
                logger.debug(f"DexScreener search failed: {e}")
            
            # Method 3: Get trending Solana pairs
            try:
                resp = await client.get('https://api.dexscreener.com/latest/dex/pairs/solana')
                if resp.status_code == 200:
                    data = resp.json()
                    pairs = data.get('pairs', [])
                    for pair in pairs[:30]:
                        ca = pair.get('baseToken', {}).get('address', '')
                        fdv = float(pair.get('fdv', 0) or 0)
                        volume = float(pair.get('volume', {}).get('h24', 0) or 0)
                        if ca and fdv >= 50000 and volume >= 20000 and ca not in [c['ca'] for c in candidates]:
                            candidates.append({
                                'ca': ca,
                                'name': pair.get('baseToken', {}).get('name', ''),
                                'symbol': pair.get('baseToken', {}).get('symbol', ''),
                                'market_cap': fdv,
                                'complete': True,
                                'source': 'trending',
                                'is_pump': 'pump' in ca.lower()
                            })
            except Exception as e:
                logger.debug(f"DexScreener trending failed: {e}")
        
        if not candidates:
            return {'success': False, 'error': 'No Solana memecoins found'}
        
        # Score each candidate by traction metrics using DexScreener
        scored_candidates = []
        
        async with httpx.AsyncClient(timeout=10.0) as client:
            for coin in candidates[:25]:  # Check top 25 candidates
                try:
                    resp = await client.get(f'https://api.dexscreener.com/latest/dex/tokens/{coin["ca"]}')
                    if resp.status_code != 200:
                        continue
                    
                    token_data = resp.json()
                    pairs = token_data.get('pairs', [])
                    if not pairs:
                        continue
                    
                    pair = pairs[0]
                    
                    # Extract traction metrics
                    volume_24h = float(pair.get('volume', {}).get('h24', 0) or 0)
                    volume_6h = float(pair.get('volume', {}).get('h6', 0) or 0)
                    volume_1h = float(pair.get('volume', {}).get('h1', 0) or 0)
                    txns = pair.get('txns', {})
                    buys_24h = int(txns.get('h24', {}).get('buys', 0) or 0)
                    sells_24h = int(txns.get('h24', {}).get('sells', 0) or 0)
                    buys_1h = int(txns.get('h1', {}).get('buys', 0) or 0)
                    sells_1h = int(txns.get('h1', {}).get('sells', 0) or 0)
                    change_5m = float(pair.get('priceChange', {}).get('m5', 0) or 0)
                    change_1h = float(pair.get('priceChange', {}).get('h1', 0) or 0)
                    liquidity = float(pair.get('liquidity', {}).get('usd', 0) or 0)
                    fdv = float(pair.get('fdv', 0) or 0)
                    price = float(pair.get('priceUsd', 0) or 0)
                    
                    # Calculate traction score (higher = more engagement/community)
                    traction_score = 0
                    
                    # Volume velocity - recent volume vs older (momentum)
                    if volume_6h > 0:
                        vol_velocity = (volume_1h * 6) / volume_6h  # 1h extrapolated vs 6h
                        traction_score += min(vol_velocity * 20, 40)  # Max 40 pts for accelerating volume
                    
                    # Buy pressure - more buys than sells = community accumulating
                    total_txns_1h = buys_1h + sells_1h
                    if total_txns_1h > 0:
                        buy_ratio = buys_1h / total_txns_1h
                        traction_score += buy_ratio * 30  # Max 30 pts for buy pressure
                    
                    # Transaction count - more txns = more engagement
                    traction_score += min(total_txns_1h / 10, 20)  # Max 20 pts for txn count
                    
                    # Volume size bonus
                    if volume_1h >= 50000:
                        traction_score += 15
                    elif volume_1h >= 20000:
                        traction_score += 10
                    elif volume_1h >= 5000:
                        traction_score += 5
                    
                    # Positive momentum bonus
                    if change_1h > 10:
                        traction_score += 10
                    elif change_1h > 5:
                        traction_score += 5
                    
                    # Graduated bonus (made it through the gauntlet)
                    if coin.get('complete'):
                        traction_score += 15
                    
                    # Close to bonding bonus (FOMO territory)
                    elif coin['market_cap'] >= 50000:
                        traction_score += 10
                    
                    # Get symbol/name from pair data if not already set
                    base_token = pair.get('baseToken', {})
                    if not coin.get('symbol') or coin['symbol'] == '':
                        coin['symbol'] = base_token.get('symbol', 'UNKNOWN')
                    if not coin.get('name') or coin['name'] == '':
                        coin['name'] = base_token.get('name', 'Unknown')
                    
                    # Store scored candidate
                    coin['traction_score'] = traction_score
                    coin['volume_24h'] = volume_24h
                    coin['volume_1h'] = volume_1h
                    coin['buys_1h'] = buys_1h
                    coin['sells_1h'] = sells_1h
                    coin['change_5m'] = change_5m
                    coin['change_1h'] = change_1h
                    coin['price'] = price
                    coin['liquidity'] = liquidity
                    coin['fdv'] = fdv
                    coin['market_cap'] = fdv
                    
                    scored_candidates.append(coin)
                    
                except Exception as e:
                    logger.debug(f"Failed to score {coin['ca'][:8]}: {e}")
                    continue
        
        if not scored_candidates:
            return {'success': False, 'error': 'Failed to score any candidates'}
        
        # Sort by traction score (highest first)
        scored_candidates.sort(key=lambda x: x.get('traction_score', 0), reverse=True)
        
        # Pick from top 3 highest traction (some randomness for variety)
        top_picks = scored_candidates[:3]
        chosen = random.choice(top_picks)
        
        # Determine status based on coin state
        if chosen.get('complete'):
            status = "just graduated"
        elif chosen['market_cap'] >= 50000:
            bonding_pct = min(99, int((chosen['market_cap'] / 69000) * 100))
            status = f"~{bonding_pct}% to bonding"
        else:
            status = "gaining traction"
        
        ca = chosen['ca']
        symbol = chosen['symbol']
        name = chosen['name']
        market_cap = chosen['market_cap']
        price = chosen.get('price', 0)
        volume_24h = chosen.get('volume_24h', 0)
        volume_1h = chosen.get('volume_1h', 0)
        buys_1h = chosen.get('buys_1h', 0)
        sells_1h = chosen.get('sells_1h', 0)
        change_5m = chosen.get('change_5m', 0)
        change_1h = chosen.get('change_1h', 0)
        traction_score = chosen.get('traction_score', 0)
        
        logger.info(f"Selected memecoin ${symbol} with traction score {traction_score:.1f} (vol_1h: ${volume_1h:.0f}, buys: {buys_1h}, sells: {sells_1h})")
        
        # Format values
        price_str = f"${price:.10f}" if price < 0.0001 else f"${price:.6f}" if price < 0.01 else f"${price:.4f}"
        mc_str = f"${market_cap/1e6:.2f}M" if market_cap >= 1e6 else f"${market_cap/1e3:.1f}K"
        vol_str = f"${volume_24h/1e6:.1f}M" if volume_24h >= 1e6 else f"${volume_24h/1e3:.0f}K" if volume_24h >= 1000 else "low"
        
        sign_5m = "+" if change_5m >= 0 else ""
        sign_1h = "+" if change_1h >= 0 else ""
        
        # Super human-like Solana memecoin tweets - casual degen style
        is_pump = chosen.get('is_pump', False)
        
        # Trending/boosted coin templates
        trending_templates = [
            f"yo ${symbol} popping off rn\n\n{ca}\n\nmight be nothing might be something idk",
            f"${symbol} looking interesting ngl\n\nchart actually looks clean for once\n\n{ca}",
            f"ok ${symbol} hasnt rugged yet thats already better than most\n\n{ca}",
            f"this ${symbol} thing keeps showing up\n\n{ca}\n\ncould be early could be exit liquidity who knows",
            f"${symbol} getting attention lately\n\n{ca}",
            f"another one on the radar\n\n${symbol}\n\n{ca}\n\nnot in yet just watching",
            f"${symbol} volume looking real\n\n{ca}\n\nnfa obviously",
        ]
        
        # General discovery templates  
        general_templates = [
            f"found ${symbol} while doomscrolling at 2am as one does\n\n{ca}",
            f"${symbol} popped up on my feed\n\nidk if im early or late but the chart looks decent\n\n{ca}",
            f"someone in the gc mentioned ${symbol}\n\n{ca}\n\ndoing my own research now",
            f"${symbol}\n\n{ca}\n\nnot advice just what im looking at",
            f"this ${symbol} thing keeps showing up\n\n{ca}\n\nmight throw a small bag at it idk",
            f"${symbol}\n\n{ca}\n\nchart doesnt look terrible which is rare",
            f"saw someone ape ${symbol} so naturally i had to look\n\n{ca}\n\nits giving early vibes but ive been fooled before",
            f"${symbol} caught my attention\n\n{ca}\n\nthe volume is actually real for once",
            f"ok hear me out\n\n${symbol}\n\n{ca}\n\ncould be the one or could be nothing",
            f"${symbol} looking like it might do something\n\n{ca}\n\nnot financial advice im literally just a guy on the internet",
            f"${symbol} on solana\n\n{ca}\n\nwatching this one",
            f"stumbled onto ${symbol}\n\n{ca}\n\ncommunity seems active idk",
        ]
        
        # Pick random template
        all_templates = trending_templates + general_templates
        tweet = random.choice(all_templates)
        
        import asyncio
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, account_poster.post_tweet, tweet)
        
    except httpx.TimeoutException:
        logger.error("Pump.fun API timeout")
        return {'success': False, 'error': 'API timeout'}
    except Exception as e:
        logger.error(f"Error posting memecoin: {e}")
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
    """Post quick technical analysis with human-like personality"""
    try:
        gainers = await main_poster.get_top_gainers_data(15)
        if not gainers:
            return {'success': False, 'error': 'No data'}
        
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
        
        chart_analysis = None
        try:
            exchange = ccxt.binance({'enableRateLimit': True})
            ohlcv = await exchange.fetch_ohlcv(f"{symbol}/USDT", '1h', limit=48)
            await exchange.close()
            if ohlcv and len(ohlcv) >= 20:
                closes = [c[4] for c in ohlcv]
                ema9 = sum(closes[-9:]) / 9
                ema21 = sum(closes[-21:]) / 21
                gains, losses = [], []
                for i in range(1, min(15, len(closes))):
                    diff = closes[i] - closes[i-1]
                    gains.append(max(diff, 0))
                    losses.append(max(-diff, 0))
                avg_gain = sum(gains) / len(gains) if gains else 0
                avg_loss = sum(losses) / len(losses) if losses else 0.0001
                rsi_val = 100 - (100 / (1 + avg_gain / avg_loss))
                trend_val = "bullish" if ema9 > ema21 else "bearish" if ema9 < ema21 else "neutral"
                chart_analysis = {'rsi': round(rsi_val, 1), 'trend': trend_val}
        except Exception as e:
            logger.warning(f"TA analysis failed: {e}")
        
        tl = _pick_tweet_length()
        
        if chart_analysis:
            rsi = chart_analysis.get('rsi', 50)
            trend = chart_analysis.get('trend', 'neutral')
            
            if rsi > 70: rsi_note = random.choice([f"RSI at {rsi:.0f} stretched", f"RSI running hot at {rsi:.0f}", f"overbought territory RSI {rsi:.0f}"])
            elif rsi < 30: rsi_note = random.choice([f"RSI oversold at {rsi:.0f}", f"RSI compressed at {rsi:.0f}", f"bounce territory RSI at {rsi:.0f}"])
            else: rsi_note = random.choice([f"RSI at {rsi:.0f}", f"RSI sitting at {rsi:.0f}", f"RSI neutral at {rsi:.0f}"])
            
            if trend == 'bullish': trend_note = random.choice(["uptrend intact", "buyers in control", "structure looks good", "higher lows holding"])
            elif trend == 'bearish': trend_note = random.choice(["sellers in control", "downtrend active", "lower highs forming", "bears have it"])
            else: trend_note = random.choice(["no clear direction yet", "ranging", "choppy", "waiting for a break"])
            
            if tl == 'ultra_short':
                templates = [
                    f"${symbol}. {rsi_note}",
                    f"${symbol} looking {'clean' if trend == 'bullish' else 'choppy' if trend == 'neutral' else 'heavy'}",
                    f"${symbol} chart. interesting",
                ]
            elif tl == 'short':
                templates = [
                    f"${symbol} at {price_str}. {rsi_note}. {trend_note}",
                    f"pulled up ${symbol}. {sign}{change:.1f}%. {rsi_note}",
                    f"${symbol} {sign}{change:.1f}% at {price_str}. {trend_note}",
                    f"quick look at ${symbol}. {rsi_note}. {trend_note}. nothing more to add",
                ]
            elif tl == 'long':
                templates = [
                    f"spent some time looking at ${symbol} tonight. {price_str} after a {sign}{change:.1f}% day.\n\n{rsi_note}. {trend_note}. the thing that stands out to me is how {'clean the structure is right now. higher lows, volume on the pushes' if trend == 'bullish' else 'indecisive the price action is. no clear direction yet which usually resolves with a big move either way' if trend == 'neutral' else 'methodically the sellers are pushing it down. lower highs on every bounce'}.\n\nnot making any moves tonight but this is on my list for tomorrow",
                    f"${symbol} technical breakdown:\n\n{rsi_note} - {'careful here, momentum is extended' if rsi > 70 else 'could see a bounce from these levels' if rsi < 30 else 'nothing extreme which is actually what you want'}\n\n{trend_note}. sitting at {price_str} ({sign}{change:.1f}%).\n\nthe numbers dont lie. {'setup looks solid' if trend == 'bullish' and rsi < 65 else 'waiting for more confirmation' if trend == 'neutral' else 'patience is the play here'}",
                ]
            else:
                templates = [
                    f"pulled up ${symbol}. {sign}{change:.1f}% at {price_str}. {rsi_note}. {trend_note}. {'I like what im seeing' if trend == 'bullish' and rsi < 65 else 'watching for a setup' if trend == 'neutral' else 'not my favorite chart but keeping an eye on it'}",
                    f"${symbol} at {price_str}. {sign}{change:.1f}% today. {rsi_note} and {trend_note}. chart tells the story",
                    f"been studying ${symbol}. {price_str} with {rsi_note}. {trend_note}. {'clean setup' if trend == 'bullish' else 'needs more time' if trend == 'neutral' else 'caution here'}",
                    f"${symbol} {sign}{change:.1f}% at {price_str}. {rsi_note}. {trend_note}. the data speaks for itself",
                ]
        else:
            if tl == 'ultra_short':
                templates = [
                    f"${symbol}. watching",
                    f"${symbol} has my eye",
                ]
            elif tl == 'long':
                templates = [
                    f"${symbol} at {price_str} after {sign}{change:.1f}% today.\n\ndont have the full technical picture on this one yet but the price action alone caught my attention. gonna pull up the chart properly later but wanted to note this while its fresh. sometimes the best trades start as a screenshot on your phone",
                ]
            else:
                templates = [
                    f"${symbol} looking interesting at {price_str}. {sign}{change:.1f}% today. keeping an eye on this",
                    f"checking in on ${symbol}. {sign}{change:.1f}% at {price_str}. something about this chart",
                    f"${symbol} on my radar. {price_str} after {sign}{change:.1f}%",
                ]
        
        tweet_text = random.choice(templates) + _get_hashtag_style()
        
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
    
    elif post_type == 'memecoin':
        return await post_memecoin(account_poster)
    
    elif post_type == 'bitunix_campaign':
        return await post_bitunix_campaign(account_poster)
    
    else:
        funcs = [post_social_news, post_early_gainers, post_momentum_shift, post_volume_surge]
        return await random.choice(funcs)(account_poster)


BITUNIX_CAMPAIGN_IMAGE = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 
                                       "attached_assets", "IMG_0697_1771526385489.jpeg")
BITUNIX_CAMPAIGN_LINK = "https://www.bitunix.com/activity/basic/1771470735?vipCode=fgq7"
BITUNIX_CAMPAIGN_START = datetime(2026, 2, 20)
BITUNIX_CAMPAIGN_END = datetime(2026, 3, 19, 23, 59, 59)

CAMPAIGN_TEMPLATES = [
    {
        'id': 'launch_fomo',
        'text': """Bitunix x TradeHub Markets just went LIVE and slots are already disappearing

While {ticker1} {ticker2} {ticker3} are pumping, smart traders are also grabbing deposit bonuses

Up to $400 deposit bonus + $6,000 volume rewards

Only 50 slots at the low tier. 20 at the top. No restock.

{link}

{hashtags}"""
    },
    {
        'id': 'urgency_tickers',
        'text': """People longing {ticker1} and {ticker2} on Bitunix right now are also stacking deposit bonuses

$400 free on deposits + $6,000 in volume rewards

Bitunix x TradeHub Markets
Slots filling fast. Don't find out when it's too late.

{link}

{hashtags}"""
    },
    {
        'id': 'fomo_math',
        'text': """Quick math while {ticker1} pumps:

Deposit $1,000 on Bitunix = $200 bonus
Trade $5M volume = another $300

That's $500 FREE on top of your {ticker1} gains

Bitunix x TradeHub Markets
Deposit slots almost gone. Ends Mar 19.

{link}

{hashtags}"""
    },
    {
        'id': 'volume_flex',
        'text': """If you're already doing volume on {ticker1} {ticker2} {ticker3} perps you're literally leaving money on the table

250K vol = $20
5M vol = $300
50M vol = $3,000
100M vol = $6,000

Bitunix x TradeHub Markets
Stop trading for free.

{link}

{hashtags}"""
    },
    {
        'id': 'social_proof',
        'text': """Traders already claimed 60%+ of the deposit bonus slots on Bitunix

$400 deposit match + $6,000 volume rewards

While you're watching {ticker1} charts, others are getting paid to trade it.

Bitunix x TradeHub Markets
Ends Mar 19. Top tiers almost full.

{link}

{hashtags}"""
    },
    {
        'id': 'perps_angle',
        'text': """You're already trading {ticker1} and {ticker2} perps

Why not get paid an extra $400 deposit bonus + $6,000 volume rewards for doing the same thing

Bitunix x TradeHub Markets
Limited slots. First come first served.

{link}

{hashtags}"""
    },
    {
        'id': 'regret_fomo',
        'text': """Every day you wait is a day someone else takes your deposit bonus slot

$2,000 tier = $400 free (only 20 slots left)
$1,000 tier = $200 free (going fast)

Plus up to $6,000 volume rewards on top

Bitunix x TradeHub Markets
Don't be the one saying I should've signed up

{link}

{hashtags}"""
    },
    {
        'id': 'whale_degen',
        'text': """Degen or whale doesn't matter

$100 deposit = $20 bonus
$2,000 deposit = $400 bonus
Volume rewards scale to $6,000

{ticker1} {ticker2} {ticker3} all tradeable on Bitunix with up to 125x leverage

Bitunix x TradeHub Markets
Slots are first come. No extensions.

{link}

{hashtags}"""
    },
    {
        'id': 'night_grind',
        'text': """3am degen session trading {ticker1}?

Might as well collect $400 in deposit bonuses and $6,000 in volume rewards while you're at it

Bitunix x TradeHub Markets
Campaign ends Mar 19. Slots won't last.

{link}

{hashtags}"""
    },
    {
        'id': 'countdown_panic',
        'text': """Campaign clock is ticking and deposit slots keep disappearing

$400 bonus slots = almost gone
$200 bonus slots = going fast
Volume rewards up to $6,000 = still available

{ticker1} traders are already in. Are you?

Bitunix x TradeHub Markets
Ends March 19. No extensions.

{link}

{hashtags}"""
    },
    {
        'id': 'comparison_brutal',
        'text': """Your current exchange charges you fees to trade {ticker1}

Bitunix is PAYING you to trade it

$400 deposit bonus + $6,000 volume rewards

Bitunix x TradeHub Markets
Switch now before slots run out.

{link}

{hashtags}"""
    },
    {
        'id': 'stack_rewards',
        'text': """The rewards stack and it's insane

Deposit $2,000 = $400 bonus
Trade your way to $6,000 more

That's $6,400 FREE on top of your {ticker1} and {ticker2} PnL

Bitunix x TradeHub Markets
Limited time. Limited slots. No second chances.

{link}

{hashtags}"""
    },
    {
        'id': 'scarcity',
        'text': """Slot count update:

$2,000 tier ($400 bonus) = only 20 slots total
$1,000 tier ($200 bonus) = 30 slots
$500 tier ($100 bonus) = 50 slots

These are filling every day. Not a drill.

Trade {ticker1} {ticker2} {ticker3} on Bitunix and get rewarded

{link}

{hashtags}"""
    },
    {
        'id': 'gains_angle',
        'text': """{ticker1} up {pct1}% today
{ticker2} up {pct2}% today

Imagine catching those moves AND getting a $400 deposit bonus on top

Bitunix x TradeHub Markets
$6,000 volume rewards still up for grabs

Slots running out. Don't sleep.

{link}

{hashtags}"""
    },
    {
        'id': 'no_brainer_v2',
        'text': """This shouldn't even be a question

Deposit on Bitunix = free money ($400 max)
Trade on Bitunix = more free money ($6,000 max)

{ticker1} {ticker2} {ticker3} all available with leverage

Bitunix x TradeHub Markets
Campaign ends Mar 19. Claim your slot NOW.

{link}

{hashtags}"""
    },
    {
        'id': 'last_call',
        'text': """Last call for the good tiers

$400 deposit bonus slots almost gone
$6,000 volume rewards still available

If you're trading {ticker1} or {ticker2} anywhere else you're doing it wrong

Bitunix x TradeHub Markets

{link}

{hashtags}"""
    },
]

_campaign_post_index = 0
_campaign_posted_today = set()


async def get_trending_hashtags(main_poster=None) -> str:
    """Fetch top gainers and build trending cashtags ($TICKER format for blue links)"""
    try:
        poster = main_poster or get_twitter_poster()
        gainers = await poster.get_top_gainers_data(8)
        
        if not gainers:
            return "$BTC $ETH $SOL"
        
        tags = []
        for coin in gainers[:5]:
            symbol = coin['symbol'].replace('USDT', '').replace('/', '')
            tags.append(f"${symbol}")
        
        if not any(t == "$BTC" for t in tags):
            tags.append("$BTC")
        
        return " ".join(tags)
        
    except Exception as e:
        logger.error(f"Error getting trending hashtags: {e}")
        return "$BTC $ETH $SOL"


async def get_live_tickers_for_campaign() -> Dict:
    """Fetch live top gainers for campaign ticker placeholders using CCXT (handles geo-blocks)"""
    fallback = {
        'ticker1': '$BTC', 'ticker2': '$ETH', 'ticker3': '$SOL',
        'pct1': '3.2', 'pct2': '2.8',
    }
    try:
        poster = get_twitter_poster()
        gainers = await poster.get_top_gainers_data(10)
        
        if gainers and len(gainers) >= 3:
            results = []
            for g in gainers:
                sym = g.get('symbol', '')
                change = g.get('change', 0)
                if sym and change and change > 0:
                    results.append({'symbol': f'${sym}', 'pct': round(abs(change), 1)})
            
            if len(results) >= 3:
                return {
                    'ticker1': results[0]['symbol'],
                    'ticker2': results[1]['symbol'],
                    'ticker3': results[2]['symbol'],
                    'pct1': str(results[0]['pct']),
                    'pct2': str(results[1]['pct']),
                }
    except Exception as e:
        logger.error(f"Error fetching live tickers for campaign: {e}")
    
    try:
        import httpx
        async with httpx.AsyncClient() as client:
            resp = await client.get("https://contract.mexc.com/api/v1/contract/ticker", timeout=8)
            if resp.status_code == 200:
                data = resp.json().get('data', [])
                mexc_gainers = []
                for t in data:
                    sym = t.get('symbol', '')
                    if not sym.endswith('_USDT'):
                        continue
                    change = float(t.get('riseFallRate', 0)) * 100
                    vol = float(t.get('amount24', 0) or 0)
                    if vol > 1_000_000 and change > 0.5:
                        clean = sym.replace('_USDT', '')
                        mexc_gainers.append({'symbol': f'${clean}', 'pct': round(change, 1)})
                mexc_gainers.sort(key=lambda x: x['pct'], reverse=True)
                if len(mexc_gainers) >= 3:
                    return {
                        'ticker1': mexc_gainers[0]['symbol'],
                        'ticker2': mexc_gainers[1]['symbol'],
                        'ticker3': mexc_gainers[2]['symbol'],
                        'pct1': str(mexc_gainers[0]['pct']),
                        'pct2': str(mexc_gainers[1]['pct']),
                    }
    except Exception as e:
        logger.error(f"MEXC fallback also failed for campaign tickers: {e}")
    
    return fallback


async def post_bitunix_campaign(account_poster) -> Optional[Dict]:
    """Post a Bitunix campaign tweet with the campaign image, live tickers, and FOMO"""
    global _campaign_post_index
    
    now = datetime.utcnow()
    if now < BITUNIX_CAMPAIGN_START or now > BITUNIX_CAMPAIGN_END:
        logger.info("Bitunix campaign not active, skipping campaign post")
        return None
    
    try:
        template = CAMPAIGN_TEMPLATES[_campaign_post_index % len(CAMPAIGN_TEMPLATES)]
        _campaign_post_index += 1
        
        hashtags = await get_trending_hashtags()
        live_tickers = await get_live_tickers_for_campaign()
        
        tweet_text = template['text'].format(
            link=BITUNIX_CAMPAIGN_LINK,
            hashtags=hashtags,
            **live_tickers
        )
        
        if len(tweet_text) > 280:
            lines = tweet_text.split('\n')
            while len('\n'.join(lines)) > 280 and len(lines) > 3:
                removed = False
                for i in range(len(lines) - 1, -1, -1):
                    line = lines[i].strip()
                    if not line:
                        lines.pop(i)
                        removed = True
                        break
                if not removed:
                    for i in range(len(lines) - 1, 0, -1):
                        line = lines[i].strip()
                        if line and not line.startswith('$') and not line.startswith('http') and 'Bitunix' not in line:
                            lines.pop(i)
                            break
                    else:
                        break
            tweet_text = '\n'.join(lines)
        
        media_id = None
        if os.path.exists(BITUNIX_CAMPAIGN_IMAGE):
            try:
                with open(BITUNIX_CAMPAIGN_IMAGE, 'rb') as f:
                    image_bytes = f.read()
                
                if hasattr(account_poster, 'upload_media'):
                    media_id = account_poster.upload_media(image_bytes)
                elif hasattr(account_poster, 'api_v1') and account_poster.api_v1:
                    media = account_poster.api_v1.media_upload(
                        filename="bitunix_campaign.jpeg", 
                        file=io.BytesIO(image_bytes)
                    )
                    media_id = str(media.media_id)
                
                if media_id:
                    logger.info(f"Campaign image uploaded: {media_id}")
                else:
                    logger.warning("Campaign image upload failed, posting without image")
            except Exception as e:
                logger.error(f"Error uploading campaign image: {e}")
        else:
            logger.warning(f"Campaign image not found: {BITUNIX_CAMPAIGN_IMAGE}")
        
        media_ids = [media_id] if media_id else None
        result = account_poster.post_tweet(tweet_text, media_ids=media_ids)
        
        if result and result.get('success'):
            logger.info(f"Campaign tweet posted (template: {template['id']})")
        
        return result
        
    except Exception as e:
        logger.error(f"Error posting campaign tweet: {e}")
        return {'success': False, 'error': str(e)}
