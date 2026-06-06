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

logger = logging.getLogger(__name__)

# Global cache for top gainers data to handle API failures
_gainers_cache = {
    'data': [],
    'timestamp': None,
    'ttl': 300  # 5 minutes cache
}

# ── Daily gainers ticker cache (injected into every post for discoverability) ─
_DAILY_GAINERS_TICKERS: List[str] = []
_DAILY_GAINERS_RESET = None


def _update_daily_gainers(tickers: List[Dict]):
    """Update the daily top-gainer tickers used to append to all posts."""
    global _DAILY_GAINERS_TICKERS, _DAILY_GAINERS_RESET
    today = datetime.utcnow().date()
    if _DAILY_GAINERS_RESET != today:
        _DAILY_GAINERS_TICKERS = []
        _DAILY_GAINERS_RESET = today
    _DAILY_GAINERS_TICKERS = [t['symbol'] for t in tickers[:10] if t.get('volume', 0) >= 1_000_000]


def get_daily_gainers_str(max_tickers: int = 5, exclude: Optional[str] = None) -> str:
    """Return a space-separated string of today's top gainer tickers."""
    syms = [s for s in _DAILY_GAINERS_TICKERS if s != exclude][:max_tickers]
    return " ".join(f"${s}" for s in syms) if syms else ""


async def _fetch_mexc_tickers() -> List[Dict]:
    """
    Fetch all USDT spot tickers from MEXC (not Binance — geoblocked on Replit).
    Returns list sorted by 24h % change descending: {symbol, price, change, volume}
    """
    import httpx as _httpx
    async with _httpx.AsyncClient(timeout=15.0) as client:
        r = await client.get("https://api.mexc.com/api/v3/ticker/24hr")
        r.raise_for_status()
        raw = r.json()

    stables = {'USDC', 'BUSD', 'DAI', 'TUSD', 'USDP', 'FDUSD', 'USDE', 'PYUSD', 'USDD', 'USDJ'}
    result = []
    for t in raw:
        sym = t.get('symbol', '')
        if not sym.endswith('USDT'):
            continue
        base = sym[:-4]
        if base in stables or 'USD' in base or len(base) < 2:
            continue
        try:
            change = float(t.get('priceChangePercent', 0))
            last   = float(t.get('lastPrice', 0))
            vol    = float(t.get('quoteVolume', 0))
        except (ValueError, TypeError):
            continue
        if last <= 0 or vol < 100_000:
            continue
        result.append({'symbol': base, 'price': last, 'change': change, 'volume': vol})

    result.sort(key=lambda x: x['change'], reverse=True)
    return result


async def _fetch_mexc_ohlcv(symbol: str, interval: str = '1h', limit: int = 48) -> List:
    """
    Fetch OHLCV candles from MEXC spot API.
    Returns [[openTime, open, high, low, close, volume, closeTime, quoteVolume, ...], ...]

    MEXC interval map: 1m, 5m, 15m, 30m, 60m, 4h, 1d, 1W, 1M
    Accepts standard aliases (1h→60m, 2h→120m, 1d stays 1d).
    """
    _interval_map = {'1h': '60m', '2h': '120m', '3h': '180m', '6h': '360m', '12h': '720m'}
    mexc_interval = _interval_map.get(interval, interval)
    import httpx as _httpx
    mexc_sym = f"{symbol.upper()}USDT" if not symbol.upper().endswith('USDT') else symbol.upper()
    try:
        async with _httpx.AsyncClient(timeout=12.0) as client:
            r = await client.get(
                "https://api.mexc.com/api/v3/klines",
                params={"symbol": mexc_sym, "interval": mexc_interval, "limit": limit}
            )
            r.raise_for_status()
            return r.json()
    except Exception as e:
        logger.debug(f"MEXC OHLCV fetch failed for {symbol}: {e}")
        return []


# Telegram bot token for notifications
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")

async def notify_admin_post_result(account_name: str, post_type: str, success: bool, error: str = None):
    """Send Telegram notification to owner only about post result"""
    try:
        if not TELEGRAM_BOT_TOKEN:
            return

        import os
        owner_id = os.environ.get("OWNER_TELEGRAM_ID", "")
        if not owner_id:
            return

        if success:
            msg = f"✅ <b>Twitter Post Success</b>\n\n📱 Account: {account_name}\n📝 Type: {post_type}\n⏰ {datetime.utcnow().strftime('%H:%M UTC')}"
        else:
            msg = f"❌ <b>Twitter Post FAILED</b>\n\n📱 Account: {account_name}\n📝 Type: {post_type}\n⚠️ Error: {error[:200] if error else 'Unknown'}\n⏰ {datetime.utcnow().strftime('%H:%M UTC')}"

        import httpx
        async with httpx.AsyncClient() as client:
            await client.post(
                f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
                json={
                    "chat_id": int(owner_id),
                    "text": msg,
                    "parse_mode": "HTML"
                },
                timeout=10
            )

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
    """Get current market sentiment based on BTC (via MEXC — Binance is geoblocked)"""
    try:
        import httpx as _httpx
        async with _httpx.AsyncClient(timeout=8.0) as client:
            r = await client.get("https://api.mexc.com/api/v3/ticker/24hr", params={"symbol": "BTCUSDT"})
            btc = r.json()
        change = float(btc.get('priceChangePercent', 0) or 0)
        
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
        {
            'name': 'alpha_hunter',
            'voice': 'Always searching for the next thing before the crowd. Talks about liquidity, narrative shifts, rotation. Quietly confident. Never hypes — just states what they see. Respects the process.',
            'examples': [
                "rotation out of large caps into alts is real. ${symbol} at {price_str} is one of three names I wrote down this week. not touching yet but it's in the stack",
                "the liquidity on ${symbol} improved significantly this week. before, the spread was wide enough to park a car in. now it's tight. that's a setup forming",
                "narratives move money. ${symbol} is sitting right at the intersection of two that are heating up. {change:.1f}%. watching the volume closely",
            ],
        },
        {
            'name': 'honest_loser',
            'voice': 'Talks openly about bad trades with zero ego. Finds lessons everywhere, even in L\'s. Dry and self-aware. Makes readers feel seen because every trader has been here.',
            'examples': [
                "got stopped out of ${symbol} at {price_str} this morning. two hours later it's up {change:.1f}%. the stop was right. the timing was off. these things happen",
                "I keep a trade journal and the ${symbol} entry from last week is painful to reread. I knew the setup was marginal. did it anyway. now I know again",
                "down on the week. ${symbol} went the wrong way. $ROBO too. when multiple positions all go against you it's almost never all bad luck — there's usually something to learn",
            ],
        },
        {
            'name': 'size_caller',
            'voice': 'Position sizing is everything to them. Talks about risk in terms of portfolio percentage, never absolutes. Calm and methodical. The person you want advising you during a volatile session.',
            'examples': [
                "${symbol} at {price_str} is a 2% position for me. could go to 4% if it holds the level. but I don't go heavier than that on anything moving {change:.1f}% in a day",
                "adding to ${symbol} here in small clips. not a full position yet. the chart needs to prove itself. {price_str} is the level I'm watching",
                "people ask how much I trade. the answer is always the same — whatever size lets me sleep at night. ${symbol} qualifies right now",
            ],
        },
        {
            'name': 'thread_brain',
            'voice': 'Thinks in threads but posts as singles. The tweet feels like a condensed 10-tweet thread. Sharp observations that make you want to reread. Structured thinking, loose delivery.',
            'examples': [
                "the ${symbol} thesis was: narrative + volume expansion + low float. two of three hit. {change:.1f}% in. I only half-sized it. this is why you don't partial-conviction trade",
                "pattern I've been tracking: ${symbol} puts in a quiet consolidation for 3-4 days then breaks with volume. this is day 4. {price_str}. I have my levels",
                "three things I look for before entering: liquidity, narrative alignment, and confirmation. ${symbol} has all three today at {price_str}",
            ],
        },
    ]
    return random.choice(personalities)


def _strip_extra_cashtags(text: str, max_cashtags: int = 1) -> str:
    """Twitter API limits posts to 1 cashtag ($SYMBOL). Strip extras beyond the first."""
    import re as _re
    matches = list(_re.finditer(r'\$[A-Z]{1,10}\b', text))
    if len(matches) <= max_cashtags:
        return text
    result = text
    for m in reversed(matches[max_cashtags:]):
        s, e = m.start(), m.end()
        # If the cashtag is surrounded by spaces (tag-cloud style), remove it + one space
        if s > 0 and result[s - 1] == ' ':
            result = result[:s - 1] + result[e:]
        else:
            result = result[:s] + result[e:]
    result = _re.sub(r'\n{3,}', '\n\n', result).strip()
    return result


async def _ai_review_tweet(tweet_text: str, post_type: str, context: dict = None) -> str:
    """
    AI pre-post review: reads the tweet and either approves it or rewrites it.
    Uses Claude Haiku for speed. Times out after 6s and returns original on failure.
    Never blocks a post — always falls back to original on any error.
    """
    try:
        import anthropic as _anthropic
        ctx_lines = "\n".join(f"- {k}: {v}" for k, v in (context or {}).items())
        prompt = (
            f"You are reviewing a crypto tweet before it goes live on X (Twitter).\n"
            f"Post type: {post_type}\n"
            f"Context about this post:\n{ctx_lines}\n\n"
            f"Tweet to review:\n---\n{tweet_text}\n---\n\n"
            "Check ONLY these issues:\n"
            "1. Does it contradict the context? (e.g., claims coin is pumping when context says it's flat/down)\n"
            "2. Does it sound like a corporate promo or a bot? (Should read like a real trader's personal take)\n"
            "3. Any obvious internal contradiction within the text itself?\n\n"
            "If the tweet is good as-is, reply with exactly: APPROVED\n"
            "If it needs fixing, reply with ONLY the rewritten tweet text — no explanation, no prefix.\n"
            "Rewrite must stay under 280 chars. Keep the same vibe and length. "
            "Do not add links, CTAs, or mentions not in the original."
        )
        _api_key = os.environ.get("AI_INTEGRATIONS_ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_API_KEY")
        _base_url = os.environ.get("AI_INTEGRATIONS_ANTHROPIC_BASE_URL")
        client = _anthropic.Anthropic(base_url=_base_url, api_key=_api_key) if _base_url else _anthropic.Anthropic(api_key=_api_key)

        async def _call():
            return await asyncio.to_thread(
                client.messages.create,
                model="claude-haiku-4-5",
                max_tokens=300,
                messages=[{"role": "user", "content": prompt}],
            )

        response = await asyncio.wait_for(_call(), timeout=6.0)
        result = (response.content[0].text or "").strip()

        if not result or result == "APPROVED":
            return tweet_text
        if len(result) > 350 or len(result) < 10:
            return tweet_text

        logger.info(f"[AI Reviewer] ✏️ Rewrote {post_type} post ({len(tweet_text)}→{len(result)} chars)")
        return result
    except Exception as e:
        logger.debug(f"[AI Reviewer] Skipped ({type(e).__name__}): {e}")
        return tweet_text


def _get_hashtag_style() -> str:
    """Append today's actual top-gainer cashtags for discoverability."""
    tickers = get_daily_gainers_str(max_tickers=3)
    return f"\n\n{tickers}" if tickers else ""


async def _get_ticker_suffix() -> str:
    """
    Async version — fetches fresh MEXC top gainers when the cache is cold
    (e.g. right after a bot restart). Always returns real coin cashtags, never generic hashtags.
    """
    if not _DAILY_GAINERS_TICKERS:
        try:
            fresh = await _fetch_mexc_tickers()
            _update_daily_gainers(fresh)
        except Exception:
            pass
    tickers = get_daily_gainers_str(max_tickers=3)
    return f"\n\n{tickers}" if tickers else ""


def _sanitize_tickers(text: str, expected_symbol: str) -> str:
    """
    Strip cashtags that don't belong — remove $ prefix from any ticker
    that isn't the coin we're actually posting about.
    Prevents AI hallucinating e.g. $TRADOOR when posting about $PEPE.
    """
    import re as _re
    expected = expected_symbol.upper()
    def _fix(m):
        ticker = m.group(1).upper()
        if ticker == expected:
            return f"${ticker}"
        # Keep as plain text — readable but not a cashtag link
        return ticker.lower()
    return _re.sub(r'\$([A-Za-z]{2,10})\b', _fix, text)


def _maybe_yubit_drop() -> str:
    """
    Casually mention the Yubit rewards campaign in ~1-in-4 posts.
    Only fires while the campaign is live (until April 30 2026).
    Returns an empty string most of the time so posts feel organic.
    """
    if datetime.utcnow() > datetime(2026, 4, 30, 23, 59):
        return ""
    if random.random() > 0.25:
        return ""
    link = "https://www.yubit.com/en-US/rewards-hub?inviteCode=TZQL"
    lines = [
        f"\n\nalso yubit has a rewards campaign running rn if you trade there → {link}",
        f"\n\nbtw yubit's rewards hub is live atm → {link}",
        f"\n\noff topic but yubit running a rewards thing worth checking → {link}",
        f"\n\ntrade on yubit? rewards hub is active → {link}",
        f"\n\nyubit rewards campaign still going if anyone's on there → {link}",
        f"\n\nnot related but yubit has something going on their rewards hub → {link}",
    ]
    return random.choice(lines)


async def _call_grok_tweet(prompt: str, max_chars: int, label: str = "",
                           system: str = "") -> Optional[str]:
    """
    Generate tweets using Claude. Primary AI for all Twitter content.
    """
    try:
        import anthropic as _anthropic
        _base_url = os.environ.get("AI_INTEGRATIONS_ANTHROPIC_BASE_URL")
        _api_key = os.environ.get("AI_INTEGRATIONS_ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_API_KEY")
        if not _api_key:
            return None
        client = _anthropic.Anthropic(base_url=_base_url, api_key=_api_key) if _base_url else _anthropic.Anthropic(api_key=_api_key)
        # Allow enough tokens to generate a full tweet — we'll trim cleanly if needed
        max_tokens = max(80, min(400, max_chars * 2))
        full_prompt = f"{system}\n\n{prompt}".strip() if system else prompt
        response = await asyncio.to_thread(
            client.messages.create,
            model="claude-haiku-4-5",
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": full_prompt}],
        )
        candidate = (response.content[0].text or "").strip().strip('"').strip("'").strip('```').strip()
        candidate = candidate.replace('**', '').replace('*', '')
        if not candidate or len(candidate) < 5:
            return None
        # Trim to max_chars at a clean boundary — never mid-word
        if len(candidate) > max_chars:
            trimmed = candidate[:max_chars]
            # Prefer cutting at a sentence boundary
            for sep in ('. ', '.\n', '! ', '? ', '\n\n', '\n'):
                idx = trimmed.rfind(sep)
                if idx > max_chars * 0.55:
                    candidate = trimmed[:idx + 1].rstrip()
                    break
            else:
                # Fall back to last word boundary
                candidate = trimmed.rsplit(' ', 1)[0].rstrip()
        logger.info(f"🐦 Claude tweet{' [' + label + ']' if label else ''}: {candidate[:70]}...")
        return candidate
    except Exception as e:
        logger.warning(f"Claude tweet generation failed: {e}")
        return None


async def generate_ai_tweet(coin_data: Dict, post_type: str = "featured") -> Optional[str]:
    """Use AI to generate a unique, human-like tweet with diverse personalities and lengths"""
    try:
        symbol      = coin_data.get('symbol', 'UNKNOWN')
        change      = coin_data.get('change', 0)
        price       = coin_data.get('price', 0)
        volume      = coin_data.get('volume', 0)
        rsi         = coin_data.get('rsi', 50)
        trend       = coin_data.get('trend', 'neutral')
        vol_ratio   = coin_data.get('vol_ratio', 1)
        change_7d   = coin_data.get('change_7d', None)     # 7-day % change if available
        from_ath    = coin_data.get('from_ath', None)      # % below ATH if available
        market_cap  = coin_data.get('market_cap', None)    # raw USD market cap

        price_str = f"${price:,.4f}" if price < 1 else f"${price:,.2f}" if price else "unknown"
        sign      = "+" if change >= 0 else ""
        vol_str   = f"${volume/1e9:.1f}B" if volume >= 1e9 else (f"${volume/1e6:.0f}M" if volume >= 1e6 else "low")

        time_ctx    = get_time_context()
        day_ctx     = get_day_context()
        personality = _pick_personality()
        tweet_length = _pick_tweet_length()

        # ── Build a richer context string ───────────────────────────────────
        ctx_parts = []
        if rsi >= 72:
            ctx_parts.append(f"RSI overbought at {rsi:.0f}")
        elif rsi <= 30:
            ctx_parts.append(f"RSI oversold at {rsi:.0f}")
        elif rsi >= 58:
            ctx_parts.append(f"RSI healthy at {rsi:.0f}")
        if trend == 'bullish':
            ctx_parts.append("uptrend on the hourly")
        elif trend == 'bearish':
            ctx_parts.append("downtrend on the hourly")
        if vol_ratio >= 2.5:
            ctx_parts.append(f"volume surging {vol_ratio:.1f}x average")
        elif vol_ratio >= 1.5:
            ctx_parts.append(f"volume {vol_ratio:.1f}x average")
        if change_7d is not None:
            wk_sign = "+" if change_7d >= 0 else ""
            ctx_parts.append(f"{wk_sign}{change_7d:.1f}% on the week")
        if from_ath is not None and from_ath <= -70:
            ctx_parts.append(f"still {abs(from_ath):.0f}% below ATH")
        elif from_ath is not None and from_ath >= -10:
            ctx_parts.append("near all-time high")
        if market_cap is not None:
            if market_cap >= 10e9:
                ctx_parts.append("large cap")
            elif market_cap >= 1e9:
                ctx_parts.append("mid cap")
            else:
                ctx_parts.append("small cap")
        tech_context = ", ".join(ctx_parts[:3]) if ctx_parts else "quiet market"

        examples_str = "\n".join([f'  "{e}"' for e in personality['examples']])

        if tweet_length == 'ultra_short':
            length_instruction = "1 fragment or bare sentence. Under 60 chars. Like a thought you typed mid-scroll."
            max_chars = 80
        elif tweet_length == 'short':
            length_instruction = "1-2 sentences. 60-140 chars. Tap it out, hit post."
            max_chars = 155
        elif tweet_length == 'long':
            length_instruction = "3-5 sentences with 1-2 line breaks. 190-270 chars. The kind of post people screenshot."
            max_chars = 285
        else:
            length_instruction = "2-3 sentences. 110-185 chars. One complete thought."
            max_chars = 205

        show_data = random.random() < 0.55
        data_note = (
            "Use the price or % if it flows naturally. Skip it if it would feel forced."
            if show_data else
            "Skip exact numbers. Talk about the move in feel, not data. Real people don't always quote prices."
        )

        # Randomised human situation — primes the AI with actual context instead of rules
        situation = random.choice([
            f"You noticed ${symbol} while checking charts between other things.",
            f"You almost bought ${symbol} two days ago and didn't. Now you're looking again.",
            f"Someone in a group chat mentioned ${symbol}. You pulled up the chart.",
            f"You've had a rough week trading. You're looking at ${symbol} with more caution than usual.",
            f"You're up on other positions. More relaxed, more willing to sit with uncertainty.",
            f"You've had ${symbol} on a watchlist for a week with nothing happening. Then this move.",
            f"It's late. You're tired. You're still checking charts.",
            f"You closed a ${symbol} trade too early last week. Watching it now with different eyes.",
            f"You're talking yourself into or out of a ${symbol} position. Haven't decided.",
            f"You saw the ${symbol} move on your lock screen. Opened the app to check.",
        ])

        _gainers_ctx = get_daily_gainers_str(max_tickers=5, exclude=symbol)

        system_msg = (
            "You are a crypto trader who has been in markets since 2019 and posts on X. "
            "You're not a brand, not a bot, not a marketer. You type fast and post when something catches your attention. "
            "Your voice is real — sometimes uncertain, sometimes direct, occasionally dark. "
            "No emojis. No hashtags. No exclamation marks. No NFA or DYOR. Lowercase when it feels right. "
            "Write like you're texting, not presenting. "
            f"IMPORTANT: The ONLY coin you may reference with a $ cashtag is ${symbol}. "
            "Do not write any other coin name as a cashtag (no $BTC, $ETH, $SOL etc). "
            "You may mention other coin names in plain text without a $ prefix if truly natural, but the main cashtag must be the coin above."
        )

        prompt = f"""{personality['name']} voice: {personality['voice']}

Examples — match this register exactly, do not copy:
{examples_str}

---
Coin: ${symbol} | {sign}{change:.1f}% today | {tech_context}
Volume: {vol_str} | {time_ctx['period']}, {day_ctx['day']}
{f"Other movers: {_gainers_ctx}" if _gainers_ctx else ""}

Situation: {situation}

Length: {length_instruction}
Numbers: {data_note}

Write just the tweet. No quotes, no labels, no explanation."""

        # Primary: Gemini 2.0 Flash — most natural casual voice
        try:
            from google import genai as _genai
            _gemini_key = os.getenv('AI_INTEGRATIONS_GEMINI_API_KEY') or os.getenv('GEMINI_API_KEY')
            if _gemini_key:
                _gclient = _genai.Client(api_key=_gemini_key)
                _full = system_msg + "\n\n" + prompt
                _resp = await asyncio.to_thread(
                    lambda: _gclient.models.generate_content(model="gemini-2.0-flash", contents=_full)
                )
                _tweet = (_resp.text or "").strip().strip('"').strip("'").strip('```').strip().replace('**', '').replace('*', '')
                if _tweet and 5 < len(_tweet) <= max_chars:
                    _tweet = _sanitize_tickers(_tweet, symbol)
                    logger.info(f"🐦 Gemini tweet [{personality['name']}/{tweet_length}] ${symbol}: {_tweet[:70]}...")
                    return _tweet
                # Trim if slightly over
                if _tweet and len(_tweet) > max_chars:
                    for sep in ('. ', '.\n', '! ', '? ', '\n\n', '\n'):
                        idx = _tweet[:max_chars].rfind(sep)
                        if idx > max_chars * 0.5:
                            _tweet = _tweet[:idx + 1].rstrip()
                            break
                    else:
                        _tweet = _tweet[:max_chars].rsplit(' ', 1)[0].rstrip()
                    if len(_tweet) > 10:
                        _tweet = _sanitize_tickers(_tweet, symbol)
                        logger.info(f"🐦 Gemini tweet (trimmed) [{personality['name']}/{tweet_length}] ${symbol}: {_tweet[:70]}...")
                        return _tweet
        except Exception as _e:
            logger.warning(f"Gemini tweet failed: {_e}")

        # Fallback 1: Claude Sonnet — higher quality than Haiku
        try:
            import anthropic as _anthropic
            _sonnet_key = os.getenv('AI_INTEGRATIONS_ANTHROPIC_API_KEY') or os.getenv('ANTHROPIC_API_KEY')
            _base_url = os.environ.get("AI_INTEGRATIONS_ANTHROPIC_BASE_URL")
            if _sonnet_key:
                _sclient = _anthropic.Anthropic(base_url=_base_url, api_key=_sonnet_key) if _base_url else _anthropic.Anthropic(api_key=_sonnet_key)
                mtok = max(80, min(400, max_chars * 2))
                _sresp = await asyncio.to_thread(
                    lambda: _sclient.messages.create(
                        model="claude-sonnet-4-5",
                        max_tokens=mtok,
                        system=system_msg,
                        messages=[{"role": "user", "content": prompt}],
                    )
                )
                _tweet = (_sresp.content[0].text or "").strip().strip('"').strip("'").strip('```').strip().replace('**', '').replace('*', '')
                if _tweet and len(_tweet) > 5:
                    if len(_tweet) > max_chars:
                        for sep in ('. ', '.\n', '? ', '\n\n', '\n'):
                            idx = _tweet[:max_chars].rfind(sep)
                            if idx > max_chars * 0.5:
                                _tweet = _tweet[:idx + 1].rstrip()
                                break
                        else:
                            _tweet = _tweet[:max_chars].rsplit(' ', 1)[0].rstrip()
                    _tweet = _sanitize_tickers(_tweet, symbol)
                    logger.info(f"🐦 Sonnet tweet [{personality['name']}/{tweet_length}] ${symbol}: {_tweet[:70]}...")
                    return _tweet
        except Exception as _e:
            logger.warning(f"Claude Sonnet tweet failed: {_e}")

        # Fallback 2: Claude Haiku (fast, cheap last resort)
        grok_tweet = await _call_grok_tweet(
            prompt, max_chars,
            label=f"haiku/{personality['name']}/{tweet_length}",
            system=system_msg,
        )
        if grok_tweet:
            return _sanitize_tickers(grok_tweet, symbol)

        return None

    except Exception as e:
        logger.error(f"AI tweet generation error: {e}")
        return None


async def generate_market_reflection_tweet(market_data: Dict) -> Optional[str]:
    """
    Generate a tweet about general market conditions — no specific coin focus.
    Real traders post these all the time: BTC thoughts, altseason vibes,
    market structure, general mood. Uses same personality system as coin tweets.
    """
    try:
        btc_change  = market_data.get('btc_change', 0)
        eth_change  = market_data.get('eth_change', 0)
        btc_price   = market_data.get('btc_price', 0)
        gainers_pct = market_data.get('gainers_pct', 50)   # % of coins green today
        fear_greed  = market_data.get('fear_greed', None)  # 0-100 index if available

        personality  = _pick_personality()
        tweet_length = _pick_tweet_length()
        time_ctx     = get_time_context()

        btc_sign = "+" if btc_change >= 0 else ""
        eth_sign = "+" if eth_change >= 0 else ""

        if btc_change >= 5:
            market_mood = "strong bull day"
        elif btc_change >= 2:
            market_mood = "decent green day"
        elif btc_change >= 0:
            market_mood = "flat to slightly up"
        elif btc_change >= -3:
            market_mood = "mild pullback"
        else:
            market_mood = "red across the board"

        breadth_note = ""
        if gainers_pct >= 70:
            breadth_note = "almost everything is green"
        elif gainers_pct >= 55:
            breadth_note = "more coins up than down"
        elif gainers_pct <= 30:
            breadth_note = "most coins bleeding"
        elif gainers_pct <= 45:
            breadth_note = "more red than green"

        fg_note = ""
        if fear_greed is not None:
            if fear_greed >= 75:
                fg_note = "fear and greed is in extreme greed territory"
            elif fear_greed >= 60:
                fg_note = "greed is creeping in"
            elif fear_greed <= 25:
                fg_note = "extreme fear on the index"
            elif fear_greed <= 40:
                fg_note = "fear index still elevated"

        context_lines = [f"$BTC {btc_sign}{btc_change:.1f}%, $ETH {eth_sign}{eth_change:.1f}%"]
        if breadth_note:
            context_lines.append(breadth_note)
        if fg_note:
            context_lines.append(fg_note)
        context_str = " | ".join(context_lines)

        examples_str = "\n".join([f'  "{e}"' for e in personality['examples']])

        if tweet_length == 'ultra_short':
            length_instruction = "1 sentence or fragment. Under 60 characters. Passing thought."
            max_chars = 85
        elif tweet_length == 'short':
            length_instruction = "1-2 sentences, 60-130 characters. Quick market read."
            max_chars = 150
        elif tweet_length == 'long':
            length_instruction = "3-5 sentences, 190-265 characters with 1-2 line breaks. A genuine market reflection."
            max_chars = 280
        else:
            length_instruction = "2-3 sentences, 110-180 characters. Complete thought about the market."
            max_chars = 200

        topic_angle = random.choice([
            "Your general read on today's market vibe or energy",
            "Something you noticed about how Bitcoin is behaving",
            "Your mental state as a trader today — patience, boredom, focus",
            "An observation about altcoins relative to BTC",
            "A market pattern or behavior you've seen before",
            "Something counterintuitive or ironic about today's market",
            "A reflection on risk management given today's conditions",
            "Your plan or approach for the rest of the session",
        ])

        system_msg = (
            "You are a real person who trades crypto and sometimes posts on Twitter/X. "
            "Not a bot, not an influencer. You post genuine market thoughts. "
            "No emojis, no hashtags, no exclamation marks, no promotional language. "
            "Lowercase, casual, real."
        )

        prompt = f"""PERSONALITY: {personality['name']}
{personality['voice']}

VOICE EXAMPLES:
{examples_str}

MARKET TODAY: {context_str}
OVERALL VIBE: {market_mood} | time: {time_ctx['period']}

WHAT TO WRITE ABOUT: {topic_angle}
LENGTH: {length_instruction}

HARD RULES:
- no emojis, no hashtags, no exclamation marks
- no NFA / DYOR / "not financial advice"
- no engagement bait
- no promotional words (huge, massive, moon, gem)
- use $BTC / $ETH if mentioning specific coins
- lowercase preferred
- no bullet points

FORBIDDEN PATTERNS (bot tells — never use):
- "The market is sending signals"
- "Keep an eye on..."
- "This could be a good entry"
- "Interesting times in crypto"
- Any generic hype phrase

Write ONLY the tweet. No quotes. No label:"""

        grok_tweet = await _call_grok_tweet(
            prompt, max_chars,
            label=f"market_reflection/{personality['name']}/{tweet_length}",
            system=system_msg,
        )
        if grok_tweet:
            return grok_tweet

        # Gemini fallback
        try:
            from google import genai
            gemini_key = os.getenv('AI_INTEGRATIONS_GEMINI_API_KEY') or os.getenv('GEMINI_API_KEY')
            if gemini_key:
                client = genai.Client(api_key=gemini_key)
                response = await asyncio.to_thread(
                    lambda: client.models.generate_content(
                        model="gemini-2.0-flash",
                        contents=system_msg + "\n\n" + prompt,
                    )
                )
                tweet = response.text.strip().strip('"').strip("'").strip('```').strip().replace('**', '').replace('*', '')
                if tweet and 5 < len(tweet) <= max_chars:
                    return tweet
        except Exception as e:
            logger.warning(f"Gemini market reflection fallback: {e}")

        return None

    except Exception as e:
        logger.error(f"Market reflection tweet error: {e}")
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
        """Fetch top gaining coins from MEXC (Binance is geoblocked on Replit)"""
        global _gainers_cache

        try:
            all_tickers = await _fetch_mexc_tickers()

            if all_tickers:
                _gainers_cache['data']      = all_tickers[:100]
                _gainers_cache['timestamp'] = datetime.utcnow()
                _update_daily_gainers(all_tickers)
                logger.debug(f"MEXC tickers: {len(all_tickers)} coins fetched")

            top = all_tickers[:limit]

            # ── Enhance with multi-day OHLCV change so tweets reflect the
            # full move, not just the rolling 24h window which misses pumps
            # that started 25+ hours ago.
            import asyncio as _aio
            async def _enrich(coin: Dict) -> Dict:
                try:
                    klines = await _fetch_mexc_ohlcv(coin['symbol'], interval='4h', limit=18)
                    if klines and len(klines) >= 4:
                        first_open = float(klines[0][1])
                        last_close = float(klines[-1][4])
                        if first_open > 0:
                            multi_day = (last_close - first_open) / first_open * 100
                            coin['change_7d'] = round(multi_day, 1)
                            if multi_day > coin['change']:
                                coin['change'] = round(multi_day, 1)
                except Exception:
                    pass
                return coin

            top = list(await _aio.gather(*[_enrich(c) for c in top], return_exceptions=False))

            return top

        except Exception as e:
            logger.error(f"Failed to fetch top gainers: {e}")

            if _gainers_cache['data'] and _gainers_cache['timestamp']:
                cache_age = (datetime.utcnow() - _gainers_cache['timestamp']).total_seconds()
                if cache_age < 1800:
                    logger.info(f"Using cached gainers data ({cache_age:.0f}s old)")
                    return _gainers_cache['data'][:limit]

            return []
    
    async def get_market_summary(self) -> Dict:
        """Get BTC and ETH market summary via MEXC (Binance geoblocked on Replit)"""
        try:
            import httpx as _httpx
            async with _httpx.AsyncClient(timeout=10.0) as client:
                rb = await client.get("https://api.mexc.com/api/v3/ticker/24hr", params={"symbol": "BTCUSDT"})
                re = await client.get("https://api.mexc.com/api/v3/ticker/24hr", params={"symbol": "ETHUSDT"})
            btc = rb.json()
            eth = re.json()
            return {
                'btc_price':  float(btc.get('lastPrice', 0)),
                'btc_change': float(btc.get('priceChangePercent', 0) or 0),
                'eth_price':  float(eth.get('lastPrice', 0)),
                'eth_change': float(eth.get('priceChangePercent', 0) or 0),
                'timestamp':  datetime.utcnow()
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
        """Post market summary — 80% AI-generated, 20% template fallback"""
        market = await self.get_market_summary()
        if not market:
            return None

        btc_sign = "+" if market['btc_change'] >= 0 else ""
        eth_sign = "+" if market['eth_change'] >= 0 else ""
        btc_p = f"${market['btc_price']:,.0f}"
        eth_p = f"${market['eth_price']:,.0f}"

        # 80% chance: AI-generated market reflection
        if random.random() < 0.80:
            try:
                ai_tweet = await generate_market_reflection_tweet({
                    'btc_change':  market['btc_change'],
                    'eth_change':  market['eth_change'],
                    'btc_price':   market['btc_price'],
                    'gainers_pct': market.get('gainers_pct', 50),
                    'fear_greed':  market.get('fear_greed'),
                })
                if ai_tweet:
                    return await self.post_tweet(ai_tweet + _get_hashtag_style())
            except Exception as e:
                logger.debug(f"Market summary AI failed, using template: {e}")

        # Template fallback
        if market['btc_change'] >= 5:   mood = random.choice(["bulls running wild", "green everywhere", "euphoria vibes"])
        elif market['btc_change'] >= 2:  mood = random.choice(["looking good", "bulls in control", "green candles"])
        elif market['btc_change'] >= 0:  mood = random.choice(["quiet day", "slow grind", "nothing crazy"])
        elif market['btc_change'] >= -3: mood = random.choice(["little pullback", "some red", "slight dip"])
        else:                            mood = random.choice(["pain", "bears in control", "rough out there"])

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
            all_tickers = await _fetch_mexc_tickers()
            # Losers = sort ascending by change, filter reasonable volume
            losers_all = [t for t in all_tickers if t.get('volume', 0) >= 1_000_000]
            losers_all.sort(key=lambda x: x['change'])
            losers = losers_all[:5]
            
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
            import httpx as _httpx
            async with _httpx.AsyncClient(timeout=10.0) as _cl:
                rb = await _cl.get("https://api.mexc.com/api/v3/ticker/24hr", params={"symbol": "BTCUSDT"})
            btc_t = rb.json()
            ohlcv = await _fetch_mexc_ohlcv('BTC', interval='1h', limit=24)

            price  = float(btc_t.get('lastPrice', 0))
            change = float(btc_t.get('priceChangePercent', 0) or 0)
            high   = float(btc_t.get('highPrice', 0))
            low    = float(btc_t.get('lowPrice', 0))
            volume = float(btc_t.get('quoteVolume', 0))
            sign   = "+" if change >= 0 else ""

            if ohlcv and len(ohlcv) >= 14:
                closes = [float(c[4]) for c in ohlcv]
                gains = [max(0, closes[i] - closes[i-1]) for i in range(1, len(closes))]
                losses = [max(0, closes[i-1] - closes[i]) for i in range(1, len(closes))]
                avg_gain = sum(gains[-14:]) / 14
                avg_loss = sum(losses[-14:]) / 14 or 0.0001
                rsi = 100 - (100 / (1 + avg_gain / avg_loss))
            else:
                rsi = 50
            
            vol_str = f"${volume/1e9:.1f}B" if volume >= 1e9 else f"${volume/1e6:.0f}M"

            # 80%: AI-generated BTC-focused market reflection
            if random.random() < 0.80:
                try:
                    trend_hint = 'bullish' if change >= 1 else ('bearish' if change <= -1 else 'neutral')
                    ai_tweet = await generate_ai_tweet({
                        'symbol':    'BTC',
                        'change':    change,
                        'price':     price,
                        'volume':    volume,
                        'rsi':       rsi,
                        'trend':     trend_hint,
                        'vol_ratio': 1.0,
                    }, post_type="btc_update")
                    if ai_tweet:
                        return await self.post_tweet(ai_tweet + _get_hashtag_style())
                except Exception as e:
                    logger.debug(f"BTC update AI failed, using template: {e}")

            # Template fallback
            if change >= 5:   mood = random.choice(["$BTC woke up and chose violence", "bulls not playing today", "not a drill"])
            elif change >= 2:  mood = random.choice(["solid day for the king", "green candles doing their thing", "bulls quietly taking control"])
            elif change >= 0:  mood = random.choice(["quiet grind up", "nothing crazy just steady", "boring day is a good day"])
            elif change >= -3: mood = random.choice(["small dip not panicking", "healthy pullback tbh", "bears trying something"])
            else:              mood = random.choice(["rough one ngl", "oof", "this too shall pass", "pain but weve seen worse"])

            if rsi >= 70:   rsi_note = random.choice(["RSI running hot", "overbought territory", "extended here"])
            elif rsi <= 30: rsi_note = random.choice(["oversold levels", "could bounce from here", "RSI bottoming"])
            else:           rsi_note = random.choice(["RSI looks balanced", "room to move either way", "healthy RSI"])

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
        """Post altcoin movements with human-like variety (via MEXC)"""
        try:
            all_tickers = await _fetch_mexc_tickers()

            exclude = {'BTC', 'ETH', 'USDC', 'BUSD', 'DAI', 'TUSD', 'USDP', 'USDT'}
            alts = [t for t in all_tickers if t['symbol'] not in exclude and t.get('volume', 0) >= 10_000_000]
            
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
        """Get technical analysis data via MEXC OHLCV (Binance geoblocked on Replit)"""
        try:
            ohlcv = await _fetch_mexc_ohlcv(symbol, interval='1h', limit=48)

            if not ohlcv or len(ohlcv) < 20:
                return {}

            # MEXC klines: [openTime, open, high, low, close, volume, ...]
            closes  = [float(c[4]) for c in ohlcv]
            highs   = [float(c[2]) for c in ohlcv]
            lows    = [float(c[3]) for c in ohlcv]
            volumes = [float(c[5]) for c in ohlcv]
            
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

        # Enforce Twitter's 1-cashtag limit globally — prevents 403 on all accounts
        text = _strip_extra_cashtags(text)

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
                'account': self.name,
                'text': text[:280] if text else ''
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


def _query_twitter_accounts_once():
    """Single DB attempt — returns list of active TwitterAccount objects."""
    from app.database import SessionLocal
    from app.models import TwitterAccount
    db = SessionLocal()
    try:
        accounts = db.query(TwitterAccount).filter(TwitterAccount.is_active == True).all()
        for account in accounts:
            db.expunge(account)
        return accounts
    finally:
        db.close()


def get_all_twitter_accounts():
    """Get all active Twitter accounts from database (retries once on SSL drop)."""
    try:
        return _query_twitter_accounts_once()
    except Exception as e:
        if "SSL" in str(e) or "connection" in str(e).lower() or "closed" in str(e).lower():
            logger.warning(f"DB connection error getting twitter accounts, retrying: {e}")
            try:
                from app.database import engine
                engine.dispose()
            except Exception:
                pass
            try:
                return _query_twitter_accounts_once()
            except Exception as e2:
                logger.error(f"DB retry also failed: {e2}")
                return []
        raise


def get_account_for_post_type(post_type: str):
    """Get the account assigned to a specific post type (retries once on SSL drop)."""
    def _query():
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
            if not selected_account and accounts:
                selected_account = accounts[0]
            if selected_account:
                db.expunge(selected_account)
            return selected_account
        finally:
            db.close()

    try:
        return _query()
    except Exception as e:
        if "SSL" in str(e) or "connection" in str(e).lower() or "closed" in str(e).lower():
            logger.warning(f"DB connection error getting account for post_type, retrying: {e}")
            try:
                from app.database import engine
                engine.dispose()
            except Exception:
                pass
            try:
                return _query()
            except Exception as e2:
                logger.error(f"DB retry also failed: {e2}")
                return None
        raise


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
                account.set_post_types([
                    'featured_coin', 'market_summary', 'top_gainers', 'btc_update',
                    'altcoin_movers', 'quick_ta', 'daily_recap', 'bitunix_campaign',
                    'yubit_campaign', 'bydfi_campaign', 'tradehub_promo', 'early_gainer', 'memecoin',
                    'market_take', 'free_telegram',
                ])
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
                   'altcoin_movers', 'daily_recap', 'top_losers', 'early_gainer',
                   'memecoin', 'quick_ta', 'tradehub_promo', 'market_take',
                   'bitunix_campaign', 'yubit_campaign', 'bydfi_campaign', 'free_telegram']
    
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
    (7, 0,   'top_gainer_ta'),      # Asia morning — top gainer TA
    (9, 15,  'bydfi_campaign'),     # BYDFi $2000 bonus — EU morning slot
    (12, 30, 'top_gainer_ta'),      # EU midday — top gainer TA
    (14, 0,  'bydfi_campaign'),     # BYDFi $2000 bonus — EU/US overlap
    (17, 45, 'top_gainer_ta'),      # US open — top gainer TA
    (19, 0,  'bydfi_campaign'),     # BYDFi $2000 bonus — US afternoon slot
]

POSTED_SLOTS = set()
LAST_POSTED_DAY = None
SLOT_OFFSETS = {}  # Store fixed random offsets per slot per day

# ── Persist posted slots across restarts so a reboot never causes duplicates ──
_SLOTS_STATE_FILE = '/tmp/twitter_posted_slots.json'


def _load_posted_slots_from_disk() -> set:
    """Load today's posted slots from disk. Returns empty set if file is stale or missing."""
    import json as _json
    try:
        with open(_SLOTS_STATE_FILE) as _f:
            _data = _json.load(_f)
        if _data.get('date') == datetime.utcnow().date().isoformat():
            _slots = set(_data.get('slots', []))
            _offsets = _data.get('offsets', {})
            logger.info(f"🐦 Loaded {len(_slots)} posted slots from disk (restart-safe)")
            return _slots, _offsets
    except Exception:
        pass
    return set(), {}


def _save_posted_slots_to_disk(slots: set, offsets: dict):
    """Save today's posted slots to disk so restarts don't cause duplicate posts."""
    import json as _json
    try:
        with open(_SLOTS_STATE_FILE, 'w') as _f:
            _json.dump({
                'date': datetime.utcnow().date().isoformat(),
                'slots': list(slots),
                'offsets': offsets,
            }, _f)
    except Exception as _e:
        logger.warning(f"Could not save posted slots to disk: {_e}")


# ── DB-backed slot persistence (survives restarts even if /tmp is cleared) ────
_TWITTER_SLOTS_TABLE_READY = False

def _ensure_twitter_slots_table():
    """Create the twitter_schedule_slots table if it doesn't exist."""
    global _TWITTER_SLOTS_TABLE_READY
    if _TWITTER_SLOTS_TABLE_READY:
        return
    try:
        import os, psycopg2
        url = os.environ.get("NEON_DATABASE_URL") or os.environ.get("DATABASE_URL")
        if not url:
            return
        conn = psycopg2.connect(url)
        conn.autocommit = True
        cur = conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS twitter_schedule_slots (
                id          SERIAL PRIMARY KEY,
                slot_date   DATE NOT NULL,
                slot_key    TEXT NOT NULL,
                created_at  TIMESTAMPTZ DEFAULT NOW(),
                UNIQUE (slot_date, slot_key)
            )
        """)
        cur.close()
        conn.close()
        _TWITTER_SLOTS_TABLE_READY = True
    except Exception as _e:
        logger.warning(f"Could not ensure twitter_schedule_slots table: {_e}")


def _load_posted_slots_from_db() -> set:
    """Load today's posted slot keys from the database. Restart-safe."""
    _ensure_twitter_slots_table()
    try:
        import os, psycopg2
        url = os.environ.get("NEON_DATABASE_URL") or os.environ.get("DATABASE_URL")
        if not url:
            return set()
        today = datetime.utcnow().date().isoformat()
        conn = psycopg2.connect(url)
        cur = conn.cursor()
        cur.execute("SELECT slot_key FROM twitter_schedule_slots WHERE slot_date = %s", (today,))
        rows = cur.fetchall()
        cur.close()
        conn.close()
        slots = {r[0] for r in rows}
        if slots:
            logger.info(f"🐦 Loaded {len(slots)} posted slots from DB (restart-safe)")
        return slots
    except Exception as _e:
        logger.warning(f"Could not load posted slots from DB: {_e}")
        return set()


def _save_slot_to_db(slot_key: str):
    """Persist a posted slot key to the database."""
    _ensure_twitter_slots_table()
    try:
        import os, psycopg2
        url = os.environ.get("NEON_DATABASE_URL") or os.environ.get("DATABASE_URL")
        if not url:
            return
        today = datetime.utcnow().date().isoformat()
        conn = psycopg2.connect(url)
        conn.autocommit = True
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO twitter_schedule_slots (slot_date, slot_key) VALUES (%s, %s) ON CONFLICT DO NOTHING",
            (today, slot_key)
        )
        cur.close()
        conn.close()
    except Exception as _e:
        logger.warning(f"Could not save slot to DB: {_e}")


# ═══════════════════════════════════════════════════════════════════
# ENGAGEMENT TRACKING + X TRENDING SYSTEM
# ═══════════════════════════════════════════════════════════════════

_POST_METRICS_TABLE_READY = False

def _ensure_post_metrics_table():
    """Create twitter_post_metrics table if it doesn't exist."""
    global _POST_METRICS_TABLE_READY
    if _POST_METRICS_TABLE_READY:
        return
    try:
        import os, psycopg2
        url = os.environ.get("NEON_DATABASE_URL") or os.environ.get("DATABASE_URL")
        if not url:
            return
        conn = psycopg2.connect(url)
        conn.autocommit = True
        cur = conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS twitter_post_metrics (
                id               SERIAL PRIMARY KEY,
                tweet_id         TEXT UNIQUE NOT NULL,
                account_name     TEXT NOT NULL,
                post_type        TEXT NOT NULL,
                tweet_text       TEXT,
                impressions      INTEGER,
                likes            INTEGER,
                retweets         INTEGER,
                replies          INTEGER,
                quotes           INTEGER,
                metrics_fetched  BOOLEAN DEFAULT FALSE,
                fetch_attempts   INTEGER DEFAULT 0,
                posted_at        TIMESTAMPTZ DEFAULT NOW(),
                metrics_fetched_at TIMESTAMPTZ
            )
        """)
        cur.close()
        conn.close()
        _POST_METRICS_TABLE_READY = True
        logger.info("✅ twitter_post_metrics table ready")
    except Exception as _e:
        logger.warning(f"Could not ensure twitter_post_metrics table: {_e}")


def _save_post_to_metrics_db(tweet_id: str, account_name: str, post_type: str, tweet_text: str = ""):
    """Save a newly posted tweet to the metrics table (metrics fetched later)."""
    _ensure_post_metrics_table()
    try:
        import os, psycopg2
        url = os.environ.get("NEON_DATABASE_URL") or os.environ.get("DATABASE_URL")
        if not url:
            return
        conn = psycopg2.connect(url)
        conn.autocommit = True
        cur = conn.cursor()
        cur.execute(
            """INSERT INTO twitter_post_metrics (tweet_id, account_name, post_type, tweet_text)
               VALUES (%s, %s, %s, %s) ON CONFLICT (tweet_id) DO NOTHING""",
            (tweet_id, account_name, post_type, tweet_text[:500] if tweet_text else "")
        )
        cur.close()
        conn.close()
        logger.info(f"📊 Saved tweet {tweet_id} ({post_type}) for metrics tracking")
    except Exception as _e:
        logger.warning(f"Could not save post to metrics DB: {_e}")


def _fetch_pending_tweet_metrics():
    """
    Fetch engagement metrics from X API for tweets older than 24h that haven't been
    measured yet. Uses batch GET /2/tweets endpoint — up to 100 per request.
    Returns count of tweets updated.
    """
    _ensure_post_metrics_table()
    try:
        import os, psycopg2, tweepy
        url = os.environ.get("NEON_DATABASE_URL") or os.environ.get("DATABASE_URL")
        if not url:
            return 0
        conn = psycopg2.connect(url)
        cur = conn.cursor()
        # Get tweets older than 24h, not yet fetched, and attempted fewer than 3 times
        cur.execute("""
            SELECT tweet_id FROM twitter_post_metrics
            WHERE metrics_fetched = FALSE
              AND fetch_attempts < 3
              AND posted_at < NOW() - INTERVAL '24 hours'
            ORDER BY posted_at DESC
            LIMIT 100
        """)
        rows = cur.fetchall()
        if not rows:
            cur.close()
            conn.close()
            return 0

        tweet_ids = [r[0] for r in rows]
        logger.info(f"📊 Fetching X metrics for {len(tweet_ids)} tweets...")

        # Increment attempt counter for all
        cur.execute(
            "UPDATE twitter_post_metrics SET fetch_attempts = fetch_attempts + 1 WHERE tweet_id = ANY(%s)",
            (tweet_ids,)
        )
        conn.commit()

        # X API v2 lookup — bearer token is fine for public_metrics
        bearer = os.environ.get("TWITTER_BEARER_TOKEN")
        if not bearer:
            cur.close()
            conn.close()
            return 0

        client = tweepy.Client(bearer_token=bearer, wait_on_rate_limit=False)
        response = client.get_tweets(
            ids=tweet_ids,
            tweet_fields=["public_metrics", "created_at"]
        )

        if not response or not response.data:
            cur.close()
            conn.close()
            return 0

        updated = 0
        for tweet in response.data:
            pm = tweet.public_metrics or {}
            cur.execute("""
                UPDATE twitter_post_metrics SET
                    impressions        = %s,
                    likes              = %s,
                    retweets           = %s,
                    replies            = %s,
                    quotes             = %s,
                    metrics_fetched    = TRUE,
                    metrics_fetched_at = NOW()
                WHERE tweet_id = %s
            """, (
                pm.get("impression_count"),
                pm.get("like_count", 0),
                pm.get("retweet_count", 0),
                pm.get("reply_count", 0),
                pm.get("quote_count", 0),
                str(tweet.id)
            ))
            updated += 1

        conn.commit()
        cur.close()
        conn.close()
        logger.info(f"📊 Updated metrics for {updated} tweets")
        return updated
    except Exception as _e:
        logger.warning(f"Could not fetch tweet metrics: {_e}")
        return 0


_TRENDING_CACHE: dict = {"topics": [], "ts": 0.0}
_TRENDING_TTL = 1800  # 30 minutes


async def _fetch_x_trending_crypto() -> list:
    """
    Search X for high-engagement recent crypto tweets to find what topics are
    trending and getting views right now. Returns list of topic dicts:
      {"topic": "$BTC", "avg_likes": 142, "avg_impressions": 8400, "sample": "...tweet text..."}
    Results cached for 30 minutes.
    """
    import time, os, tweepy, asyncio
    from collections import defaultdict

    now = time.time()
    if now - _TRENDING_CACHE["ts"] < _TRENDING_TTL and _TRENDING_CACHE["topics"]:
        return _TRENDING_CACHE["topics"]

    try:
        bearer = os.environ.get("TWITTER_BEARER_TOKEN")
        if not bearer:
            return []

        client = tweepy.Client(bearer_token=bearer, wait_on_rate_limit=False)

        # Search for high-quality crypto tweets — recent, no retweets, English
        query = (
            '(crypto OR bitcoin OR ethereum OR altcoin OR "crypto trading" OR "on-chain") '
            'lang:en -is:retweet -is:reply has:cashtags min_faves:20'
        )
        response = await asyncio.to_thread(
            client.search_recent_tweets,
            query=query,
            max_results=50,
            tweet_fields=["public_metrics", "text", "created_at"],
            sort_order="relevancy"
        )

        if not response or not response.data:
            return []

        # Aggregate by cashtag/topic
        topic_data = defaultdict(lambda: {"likes": [], "impressions": [], "samples": []})

        for tweet in response.data:
            pm = tweet.public_metrics or {}
            text = tweet.text or ""
            likes = pm.get("like_count", 0)
            impressions = pm.get("impression_count") or 0

            # Extract $TICKER mentions
            import re
            tickers = re.findall(r'\$([A-Z]{2,8})\b', text)
            stablecoins = {"USDT", "USDC", "BUSD", "DAI", "USD", "EUR"}
            tickers = [t for t in tickers if t not in stablecoins]

            if not tickers:
                tickers = ["CRYPTO"]

            for ticker in tickers[:3]:
                topic_data[f"${ticker}"]["likes"].append(likes)
                if impressions:
                    topic_data[f"${ticker}"]["impressions"].append(impressions)
                if len(topic_data[f"${ticker}"]["samples"]) < 2:
                    topic_data[f"${ticker}"]["samples"].append(text[:120])

        # Build sorted results
        results = []
        for topic, data in topic_data.items():
            if len(data["likes"]) < 2:
                continue
            avg_likes = sum(data["likes"]) / len(data["likes"])
            avg_imp = sum(data["impressions"]) / max(len(data["impressions"]), 1)
            results.append({
                "topic": topic,
                "avg_likes": round(avg_likes, 1),
                "avg_impressions": round(avg_imp),
                "mentions": len(data["likes"]),
                "sample": data["samples"][0] if data["samples"] else ""
            })

        results.sort(key=lambda x: x["avg_likes"], reverse=True)
        results = results[:10]

        _TRENDING_CACHE["topics"] = results
        _TRENDING_CACHE["ts"] = now

        if results:
            logger.info(f"🔥 X trending topics fetched: {[r['topic'] for r in results[:5]]}")
        return results

    except Exception as _e:
        logger.warning(f"Could not fetch X trending topics: {_e}")
        return []


def _get_engagement_insights() -> str:
    """
    Query twitter_post_metrics to return a formatted summary of which post types
    are getting the most views and engagement. Used to inform AI post generation.
    """
    _ensure_post_metrics_table()
    try:
        import os, psycopg2
        url = os.environ.get("NEON_DATABASE_URL") or os.environ.get("DATABASE_URL")
        if not url:
            return ""
        conn = psycopg2.connect(url)
        cur = conn.cursor()
        # Last 30 days, only fetched metrics
        cur.execute("""
            SELECT post_type,
                   COUNT(*) AS posts,
                   ROUND(AVG(impressions))   AS avg_impressions,
                   ROUND(AVG(likes))         AS avg_likes,
                   ROUND(AVG(retweets))      AS avg_retweets,
                   MAX(impressions)          AS best_impressions
            FROM twitter_post_metrics
            WHERE metrics_fetched = TRUE
              AND posted_at > NOW() - INTERVAL '30 days'
              AND impressions IS NOT NULL
            GROUP BY post_type
            ORDER BY avg_impressions DESC NULLS LAST
        """)
        rows = cur.fetchall()
        cur.close()
        conn.close()

        if not rows:
            return ""

        lines = ["YOUR PAST 30-DAY POST PERFORMANCE (use this to write posts that actually get views):"]
        for pt, posts, avg_imp, avg_likes, avg_rt, best_imp in rows:
            imp_str = f"{avg_imp:,}" if avg_imp else "n/a"
            lines.append(
                f"  {pt}: {posts} posts — avg {imp_str} impressions, {avg_likes or 0} likes, "
                f"{avg_rt or 0} retweets | best post: {best_imp:,} impressions"
            )
        # Identify top performer
        best = rows[0]
        lines.append(
            f"\nTop performer: '{best[0]}' at {best[2]:,} avg impressions. "
            f"Write posts that match the tone and specificity of what's working."
        )
        return "\n".join(lines)
    except Exception as _e:
        logger.warning(f"Could not get engagement insights: {_e}")
        return ""


async def tweet_metrics_fetcher_loop():
    """Background loop — every 2 hours, fetch X API engagement metrics for posts older than 24h."""
    _ensure_post_metrics_table()
    await asyncio.sleep(300)  # 5 min delay on startup
    while True:
        try:
            updated = await asyncio.to_thread(_fetch_pending_tweet_metrics)
            if updated:
                logger.info(f"📊 Tweet metrics loop: updated {updated} posts")
        except Exception as _e:
            logger.warning(f"Tweet metrics loop error: {_e}")
        await asyncio.sleep(7200)  # every 2 hours


# ═══════════════════════════════════════════════════════════════════
# ACCOUNT GROWTH TRACKER
# Snapshots follower/following/tweet counts every 6 hours via X API
# ═══════════════════════════════════════════════════════════════════

_GROWTH_TABLE_READY = False


def _ensure_growth_table():
    """Create twitter_account_growth table if it doesn't exist."""
    global _GROWTH_TABLE_READY
    if _GROWTH_TABLE_READY:
        return
    try:
        import os, psycopg2
        url = os.environ.get("NEON_DATABASE_URL") or os.environ.get("DATABASE_URL")
        if not url:
            return
        conn = psycopg2.connect(url)
        conn.autocommit = True
        cur = conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS twitter_account_growth (
                id              SERIAL PRIMARY KEY,
                account_name    TEXT NOT NULL,
                handle          TEXT NOT NULL,
                followers       INTEGER,
                following       INTEGER,
                tweet_count     INTEGER,
                listed_count    INTEGER,
                checked_at      TIMESTAMPTZ DEFAULT NOW()
            )
        """)
        # Index for fast time-series queries
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_growth_account_time
            ON twitter_account_growth (account_name, checked_at DESC)
        """)
        cur.close()
        conn.close()
        _GROWTH_TABLE_READY = True
        logger.info("✅ twitter_account_growth table ready")
    except Exception as _e:
        logger.warning(f"Could not ensure twitter_account_growth table: {_e}")


def _take_account_growth_snapshot():
    """
    Fetch follower/following/tweet counts for all active accounts from X API v2
    and store a snapshot in the DB. Called every 6 hours.
    """
    _ensure_growth_table()
    try:
        import os, psycopg2, tweepy
        url = os.environ.get("NEON_DATABASE_URL") or os.environ.get("DATABASE_URL")
        bearer = os.environ.get("TWITTER_BEARER_TOKEN")
        if not url or not bearer:
            return

        accounts = get_all_twitter_accounts()
        if not accounts:
            return

        client = tweepy.Client(bearer_token=bearer, wait_on_rate_limit=False)
        conn = psycopg2.connect(url)
        conn.autocommit = True
        cur = conn.cursor()

        snapped = 0
        for acc in accounts:
            if not acc.is_active:
                continue
            handle = (acc.handle or "").lstrip("@").strip()
            if not handle:
                continue
            try:
                resp = client.get_user(
                    username=handle,
                    user_fields=["public_metrics"]
                )
                if not resp or not resp.data:
                    continue
                pm = resp.data.public_metrics or {}
                cur.execute("""
                    INSERT INTO twitter_account_growth
                        (account_name, handle, followers, following, tweet_count, listed_count)
                    VALUES (%s, %s, %s, %s, %s, %s)
                """, (
                    acc.name,
                    handle,
                    pm.get("followers_count"),
                    pm.get("following_count"),
                    pm.get("tweet_count"),
                    pm.get("listed_count", 0),
                ))
                snapped += 1
                logger.info(
                    f"📈 Growth snapshot: @{handle} — "
                    f"{pm.get('followers_count', '?')} followers, "
                    f"{pm.get('tweet_count', '?')} tweets"
                )
            except Exception as _ae:
                logger.warning(f"Could not snapshot @{handle}: {_ae}")

        cur.close()
        conn.close()
        logger.info(f"📈 Growth tracker: snapped {snapped} accounts")
    except Exception as _e:
        logger.warning(f"Growth snapshot error: {_e}")


def get_account_growth_summary(days: int = 7) -> list:
    """
    Return a growth summary for all tracked accounts over the last N days.
    Each item: {account_name, handle, current_followers, followers_N_days_ago,
                 gained, pct_change, current_tweets, avg_followers_per_day}
    """
    _ensure_growth_table()
    try:
        import os, psycopg2
        url = os.environ.get("NEON_DATABASE_URL") or os.environ.get("DATABASE_URL")
        if not url:
            return []
        conn = psycopg2.connect(url, connect_timeout=5, options="-c statement_timeout=8000")
        cur = conn.cursor()
        cur.execute("""
            WITH latest AS (
                SELECT DISTINCT ON (account_name)
                    account_name, handle, followers, tweet_count, checked_at
                FROM twitter_account_growth
                ORDER BY account_name, checked_at DESC
            ),
            earliest AS (
                SELECT DISTINCT ON (account_name)
                    account_name, followers AS old_followers, checked_at AS old_ts
                FROM twitter_account_growth
                WHERE checked_at >= NOW() - INTERVAL '1 day' * %s
                ORDER BY account_name, checked_at ASC
            )
            SELECT
                l.account_name,
                l.handle,
                l.followers          AS current_followers,
                e.old_followers      AS followers_then,
                l.tweet_count        AS current_tweets,
                l.checked_at         AS last_checked
            FROM latest l
            LEFT JOIN earliest e USING (account_name)
            ORDER BY l.followers DESC NULLS LAST
        """, (days,))
        rows = cur.fetchall()
        cur.close()
        conn.close()

        results = []
        for row in rows:
            acc_name, handle, cur_f, old_f, cur_t, last_chk = row
            gained = (cur_f - old_f) if (cur_f is not None and old_f is not None) else None
            pct = round((gained / old_f * 100), 2) if (gained is not None and old_f and old_f > 0) else None
            apd = round(gained / days, 1) if gained is not None else None
            results.append({
                "account_name":        acc_name,
                "handle":              handle,
                "current_followers":   cur_f,
                "followers_then":      old_f,
                "gained":              gained,
                "pct_change":          pct,
                "current_tweets":      cur_t,
                "avg_per_day":         apd,
                "last_checked":        str(last_chk) if last_chk else None,
            })
        return results
    except Exception as _e:
        logger.warning(f"Could not get growth summary: {_e}")
        return []


async def account_growth_tracker_loop():
    """Background loop — takes follower/growth snapshots every 6 hours."""
    _ensure_growth_table()
    await asyncio.sleep(120)  # 2 min delay on startup — let accounts init first
    while True:
        try:
            await asyncio.to_thread(_take_account_growth_snapshot)
        except Exception as _e:
            logger.warning(f"Account growth tracker loop error: {_e}")
        await asyncio.sleep(21600)  # every 6 hours


# ═══════════════════════════════════════════════════════════════════
# DAILY TREND DISCOVERY + AUTO-OPTIMISATION ENGINE
#
# Every 4 hours, searches X for what crypto/finance topics are
# actually getting engagement RIGHT NOW. Posts are then generated
# about those specific trending coins/topics rather than random picks.
# After enough engagement data is collected the posting schedule
# automatically shifts slots toward the best-performing formats.
# ═══════════════════════════════════════════════════════════════════

_DAILY_TRENDS_CACHE: dict = {"date": "", "coins": [], "topics": [], "ts": 0.0}
_TRENDS_TTL = 14400  # 4 hours


def _ensure_daily_trends_table():
    try:
        import os, psycopg2
        url = os.environ.get("NEON_DATABASE_URL") or os.environ.get("DATABASE_URL")
        if not url:
            return
        conn = psycopg2.connect(url)
        conn.autocommit = True
        cur = conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS twitter_daily_trends (
                id           SERIAL PRIMARY KEY,
                trend_date   DATE NOT NULL,
                symbol       TEXT,
                topic        TEXT,
                kind         TEXT NOT NULL DEFAULT 'coin',   -- 'coin' | 'topic'
                trend_score  FLOAT DEFAULT 0,
                avg_likes    FLOAT DEFAULT 0,
                mentions     INTEGER DEFAULT 0,
                sample_tweet TEXT,
                discovered_at TIMESTAMPTZ DEFAULT NOW()
            )
        """)
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_daily_trends_date
            ON twitter_daily_trends (trend_date DESC)
        """)
        cur.close()
        conn.close()
    except Exception as _e:
        logger.warning(f"Could not ensure daily_trends table: {_e}")


async def _discover_daily_trends() -> dict:
    """
    Search X for what's actually getting engagement in crypto/finance TODAY.
    Returns {'coins': [...], 'topics': [...]} sorted by trend_score desc.

    coins  — list of {'symbol', 'trend_score', 'avg_likes', 'mentions', 'sample'}
    topics — list of {'topic', 'trend_score', 'avg_likes', 'mentions', 'sample'}

    Refreshed every 4 hours, cached in memory and DB.
    """
    import time, os, tweepy, re, asyncio
    from collections import defaultdict

    now = time.time()
    today_str = datetime.utcnow().date().isoformat()

    # Return cached result if fresh
    if (now - _DAILY_TRENDS_CACHE["ts"] < _TRENDS_TTL
            and _DAILY_TRENDS_CACHE["date"] == today_str
            and _DAILY_TRENDS_CACHE["coins"]):
        return {"coins": _DAILY_TRENDS_CACHE["coins"], "topics": _DAILY_TRENDS_CACHE["topics"]}

    bearer = os.environ.get("TWITTER_BEARER_TOKEN")
    if not bearer:
        return {"coins": [], "topics": []}

    logger.info("🔍 Discovering today's trending crypto topics on X...")
    _ensure_daily_trends_table()

    try:
        client = tweepy.Client(bearer_token=bearer, wait_on_rate_limit=False)

        # ── Search queries focused on low-cap / memecoin viral content ──────
        # We deliberately skip BTC/ETH dominant searches — they always appear
        # and drown out the small caps that are actually going viral.
        # min_faves thresholds kept at 15-25 to catch emerging moves early.
        searches = [
            # Memecoins and gems going viral right now
            '(memecoin OR meme OR "low cap" OR gem OR "hidden gem") lang:en -is:retweet has:cashtags min_faves:15',
            # Altcoin season / rotate / new coin narratives
            '(altcoin OR "alt season" OR "rotate" OR narrative OR "new listing" OR "just launched") lang:en -is:retweet has:cashtags min_faves:15',
            # Price action viral posts — 10x / 100x / mooning / pumping
            '("10x" OR "100x" OR mooning OR pumping OR "up only" OR parabolic) lang:en -is:retweet has:cashtags min_faves:20',
            # On-chain / degen / ape narratives (low cap discovery ground)
            '(defi OR "on-chain" OR degen OR aping OR "ape in" OR "aped") lang:en -is:retweet has:cashtags min_faves:15',
            # KOL-style alpha sharing (where low caps get discovered)
            '("early" OR "alpha" OR "next" OR "sleeping" OR "undervalued") lang:en -is:retweet has:cashtags min_faves:25',
        ]

        coin_data:  dict = defaultdict(lambda: {"likes": [], "impressions": [], "samples": []})
        topic_data: dict = defaultdict(lambda: {"likes": [], "impressions": [], "samples": []})

        # Coins to exclude from trending output — too dominant to be useful signals
        LARGE_CAPS = {"BTC", "ETH", "BNB", "SOL", "XRP", "ADA", "AVAX", "LINK",
                      "DOT", "MATIC", "LTC", "BCH", "ATOM", "UNI", "NEAR", "FIL"}
        STABLES = {"USDT", "USDC", "BUSD", "DAI", "USD", "EUR", "GBP", "USDS", "FDUSD", "PYUSD"}
        TOPIC_KEYWORDS = {
            "memecoin": "Memecoins", "meme coin": "Memecoins", "meme": "Memecoins",
            "altcoin": "Altcoins", "alt season": "Alt Season", "alt szn": "Alt Season",
            "gem": "Gems", "hidden gem": "Gems", "low cap": "Low Caps",
            "100x": "100x Plays", "10x": "10x Plays", "parabolic": "Parabolic Moves",
            "mooning": "Mooning Coins", "pumping": "Pumping Coins",
            "alpha": "Alpha Calls", "early": "Early Plays", "undervalued": "Undervalued",
            "degen": "Degen Plays", "aping": "Ape Plays", "narrative": "Narratives",
            "defi": "DeFi", "on-chain": "On-Chain", "nft": "NFTs",
            "whale": "Whale Alert", "regulation": "Regulation", "etf": "ETF",
            "breakout": "Breakouts", "liquidation": "Liquidations",
            "new listing": "New Listings", "just launched": "New Launches",
            "rotate": "Rotation Plays", "sleeping": "Sleeping Giants",
        }

        for query in searches:
            try:
                resp = await asyncio.to_thread(
                    client.search_recent_tweets,
                    query=query,
                    max_results=50,
                    tweet_fields=["public_metrics", "text"],
                    sort_order="relevancy"
                )
                if not resp or not resp.data:
                    continue

                for tweet in resp.data:
                    pm    = tweet.public_metrics or {}
                    text  = tweet.text or ""
                    likes = pm.get("like_count", 0)
                    imps  = pm.get("impression_count") or 0

                    # Extract $TICKER coins — skip stables AND large caps
                    tickers = re.findall(r'\$([A-Z]{2,10})\b', text)
                    for t in tickers:
                        if t in STABLES or t in LARGE_CAPS or len(t) > 8:
                            continue
                        coin_data[t]["likes"].append(likes)
                        if imps:
                            coin_data[t]["impressions"].append(imps)
                        if len(coin_data[t]["samples"]) < 2:
                            coin_data[t]["samples"].append(text[:150])

                    # Extract topic keywords
                    text_lower = text.lower()
                    for kw, canonical in TOPIC_KEYWORDS.items():
                        if kw in text_lower:
                            topic_data[canonical]["likes"].append(likes)
                            if imps:
                                topic_data[canonical]["impressions"].append(imps)
                            if len(topic_data[canonical]["samples"]) < 2:
                                topic_data[canonical]["samples"].append(text[:150])

            except Exception as _se:
                logger.warning(f"Trend search error: {_se}")
                continue

        # ── Score and rank coins (low caps only — large caps already filtered above) ──
        coins_ranked = []
        for sym, d in coin_data.items():
            if len(d["likes"]) < 2:
                continue
            avg_l = sum(d["likes"]) / len(d["likes"])
            avg_i = sum(d["impressions"]) / max(len(d["impressions"]), 1)
            mentions = len(d["likes"])
            # trend_score = engagement density × reach
            # Low caps (ticker ≤ 5 chars) get a 1.3× boost — they're rarer in tweets
            # so even modest mentions signal real interest
            low_cap_boost = 1.3 if len(sym) <= 5 else 1.0
            score = round((avg_l * (1 + mentions / 10) + avg_i / 1000) * low_cap_boost, 2)
            coins_ranked.append({
                "symbol":       sym,
                "trend_score":  score,
                "avg_likes":    round(avg_l, 1),
                "avg_impressions": round(avg_i),
                "mentions":     mentions,
                "sample":       d["samples"][0] if d["samples"] else "",
            })
        coins_ranked.sort(key=lambda x: x["trend_score"], reverse=True)
        coins_ranked = coins_ranked[:20]

        # ── Score and rank topics ───────────────────────────────────────
        topics_ranked = []
        for topic, d in topic_data.items():
            if len(d["likes"]) < 3:
                continue
            avg_l = sum(d["likes"]) / len(d["likes"])
            avg_i = sum(d["impressions"]) / max(len(d["impressions"]), 1)
            mentions = len(d["likes"])
            score = round(avg_l * (1 + mentions / 15) + avg_i / 1000, 2)
            topics_ranked.append({
                "topic":        topic,
                "trend_score":  score,
                "avg_likes":    round(avg_l, 1),
                "avg_impressions": round(avg_i),
                "mentions":     mentions,
                "sample":       d["samples"][0] if d["samples"] else "",
            })
        topics_ranked.sort(key=lambda x: x["trend_score"], reverse=True)
        topics_ranked = topics_ranked[:10]

        # ── Persist to DB ───────────────────────────────────────────────
        try:
            import psycopg2
            url = os.environ.get("NEON_DATABASE_URL") or os.environ.get("DATABASE_URL")
            conn = psycopg2.connect(url)
            conn.autocommit = True
            cur = conn.cursor()
            today = datetime.utcnow().date()
            # Clear old today entries (refresh)
            cur.execute("DELETE FROM twitter_daily_trends WHERE trend_date = %s", (today,))
            for c in coins_ranked:
                cur.execute("""
                    INSERT INTO twitter_daily_trends
                        (trend_date, symbol, kind, trend_score, avg_likes, mentions, sample_tweet)
                    VALUES (%s, %s, 'coin', %s, %s, %s, %s)
                """, (today, c["symbol"], c["trend_score"], c["avg_likes"], c["mentions"], c["sample"][:300]))
            for t in topics_ranked:
                cur.execute("""
                    INSERT INTO twitter_daily_trends
                        (trend_date, topic, kind, trend_score, avg_likes, mentions, sample_tweet)
                    VALUES (%s, %s, 'topic', %s, %s, %s, %s)
                """, (today, t["topic"], t["trend_score"], t["avg_likes"], t["mentions"], t["sample"][:300]))
            cur.close()
            conn.close()
        except Exception as _de:
            logger.warning(f"Could not persist daily trends: {_de}")

        # ── Update memory cache ─────────────────────────────────────────
        _DAILY_TRENDS_CACHE.update({
            "date":   today_str,
            "coins":  coins_ranked,
            "topics": topics_ranked,
            "ts":     now,
        })

        top_coins = [c["symbol"] for c in coins_ranked[:8]]
        top_topics = [t["topic"] for t in topics_ranked[:5]]
        logger.info(f"🔥 Today's trending coins on X: {top_coins}")
        logger.info(f"🔥 Today's trending topics on X: {top_topics}")
        return {"coins": coins_ranked, "topics": topics_ranked}

    except Exception as _e:
        logger.warning(f"Daily trend discovery failed: {_e}")
        return {"coins": [], "topics": []}


def get_todays_trending_coins(top_n: int = 15) -> list:
    """
    Return today's cached trending coins list (non-async).
    Falls back to DB if memory cache is empty.
    """
    today_str = datetime.utcnow().date().isoformat()
    if _DAILY_TRENDS_CACHE["date"] == today_str and _DAILY_TRENDS_CACHE["coins"]:
        return _DAILY_TRENDS_CACHE["coins"][:top_n]

    # Try DB fallback
    try:
        import os, psycopg2
        url = os.environ.get("NEON_DATABASE_URL") or os.environ.get("DATABASE_URL")
        if not url:
            return []
        conn = psycopg2.connect(url)
        cur = conn.cursor()
        today = datetime.utcnow().date()
        cur.execute("""
            SELECT symbol, trend_score, avg_likes, mentions, sample_tweet
            FROM twitter_daily_trends
            WHERE trend_date = %s AND kind = 'coin'
            ORDER BY trend_score DESC LIMIT %s
        """, (today, top_n))
        rows = cur.fetchall()
        cur.close()
        conn.close()
        return [{"symbol": r[0], "trend_score": r[1], "avg_likes": r[2],
                 "mentions": r[3], "sample": r[4] or ""} for r in rows]
    except Exception:
        return []


def get_todays_trending_topics(top_n: int = 5) -> list:
    """Return today's cached trending topic list (non-async)."""
    today_str = datetime.utcnow().date().isoformat()
    if _DAILY_TRENDS_CACHE["date"] == today_str and _DAILY_TRENDS_CACHE["topics"]:
        return _DAILY_TRENDS_CACHE["topics"][:top_n]

    try:
        import os, psycopg2
        url = os.environ.get("NEON_DATABASE_URL") or os.environ.get("DATABASE_URL")
        if not url:
            return []
        conn = psycopg2.connect(url)
        cur = conn.cursor()
        today = datetime.utcnow().date()
        cur.execute("""
            SELECT topic, trend_score, avg_likes, mentions, sample_tweet
            FROM twitter_daily_trends
            WHERE trend_date = %s AND kind = 'topic'
            ORDER BY trend_score DESC LIMIT %s
        """, (today, top_n))
        rows = cur.fetchall()
        cur.close()
        conn.close()
        return [{"topic": r[0], "trend_score": r[1], "avg_likes": r[2],
                 "mentions": r[3], "sample": r[4] or ""} for r in rows]
    except Exception:
        return []


def _get_auto_weighted_post_type(original_type: str) -> str:
    """
    Based on actual engagement data from the last 14 days, optionally swap
    a low-performing post type for a high-performing one.

    Only activates when there's enough data (7+ measured posts per type).
    Keeps the schedule diverse — won't replace every slot with one type.
    """
    try:
        import os, psycopg2
        url = os.environ.get("NEON_DATABASE_URL") or os.environ.get("DATABASE_URL")
        if not url:
            return original_type
        conn = psycopg2.connect(url)
        cur = conn.cursor()
        cur.execute("""
            SELECT post_type, AVG(impressions) AS avg_imp, COUNT(*) AS posts
            FROM twitter_post_metrics
            WHERE metrics_fetched = TRUE
              AND posted_at > NOW() - INTERVAL '14 days'
              AND impressions IS NOT NULL
            GROUP BY post_type
            HAVING COUNT(*) >= 7
            ORDER BY avg_imp DESC
        """)
        rows = cur.fetchall()
        cur.close()
        conn.close()

        if len(rows) < 3:
            return original_type  # not enough data yet

        # Build perf map
        perf = {r[0]: float(r[1]) for r in rows}
        if original_type not in perf:
            return original_type

        orig_imp = perf[original_type]
        best_type, best_imp = rows[0][0], float(rows[0][1])

        # Only swap if: original is underperforming by >50%, there's a clear winner,
        # and the swap type is in the market content group (no promos)
        market_types = {"featured_coin", "quick_ta", "market_take", "memecoin",
                        "early_gainer", "top_gainers"}
        if (original_type in market_types
                and best_type in market_types
                and best_type != original_type
                and best_imp > orig_imp * 1.5):
            # 30% chance to upgrade to best-performing type
            if random.random() < 0.30:
                logger.info(
                    f"⚡ Auto-weight: swapping {original_type} → {best_type} "
                    f"({orig_imp:.0f} → {best_imp:.0f} avg impressions)"
                )
                return best_type

        return original_type
    except Exception:
        return original_type


async def trend_discovery_loop():
    """
    Background loop — discovers trending coins/topics on X every 4 hours.
    First run is immediate (after a 60s startup delay).
    """
    _ensure_daily_trends_table()
    await asyncio.sleep(60)  # brief startup delay
    while True:
        try:
            result = await _discover_daily_trends()
            top = [c["symbol"] for c in result.get("coins", [])[:6]]
            if top:
                logger.info(f"🔥 Trend discovery complete — top coins today: {top}")
        except Exception as _e:
            logger.warning(f"Trend discovery loop error: {_e}")
        await asyncio.sleep(_TRENDS_TTL)  # every 4 hours


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
        'bitunix_campaign': '💰 Bitunix Campaign',
        'yubit_campaign': '💰 Yubit Campaign',
        'bydfi_campaign': '💰 BYDFi Campaign',
        'top_gainer_ta': '📈 Top Gainer TA',
        'tradehub_promo': '🏆 TradeHub Leaderboard',
        'market_take': '💭 Market Hot Take',
        'free_telegram': '📲 Free Telegram Promo',
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

    # Restore today's posted slots — DB is authoritative, /tmp/ is fast cache
    _restored_slots, _restored_offsets = _load_posted_slots_from_disk()
    POSTED_SLOTS.update(_restored_slots)
    SLOT_OFFSETS.update(_restored_offsets)
    # Also load from DB (survives /tmp/ being cleared on restarts)
    _db_slots = _load_posted_slots_from_db()
    POSTED_SLOTS.update(_db_slots)

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
    
    # Start background engagement metrics fetcher
    asyncio.create_task(tweet_metrics_fetcher_loop())
    logger.info("📊 Tweet engagement metrics fetcher started")

    # Start account growth tracker (follower snapshots every 6h)
    asyncio.create_task(account_growth_tracker_loop())
    logger.info("📈 Account growth tracker started")

    # Start daily trend discovery (runs every 4h — finds what's trending on X)
    asyncio.create_task(trend_discovery_loop())
    logger.info("🔥 Daily trend discovery loop started")

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
                _save_posted_slots_to_disk(POSTED_SLOTS, SLOT_OFFSETS)
            
            if LAST_POSTED_DAY is None:
                LAST_POSTED_DAY = current_day
            
            # Refresh database accounts periodically
            db_accounts = get_all_twitter_accounts()
            
            # Check each scheduled slot
            for hour, minute, post_type in POST_SCHEDULE:
                slot_key = f"{hour}:{minute}"
                
                # Get or create fixed random offset for this slot (persists for the day)
                # ±5 minutes — small enough that no two adjacent slots ever collide
                if slot_key not in SLOT_OFFSETS:
                    SLOT_OFFSETS[slot_key] = random.randint(-5, 5)
                
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
                
                # Campaign slots get a 30-min catch-up window (survive restarts).
                # Regular slots fire within 5 minutes of the adjusted time.
                _campaign_types = {'yubit_campaign', 'bitunix_campaign', 'bydfi_campaign'}
                _window_secs = 1800 if post_type in _campaign_types else 300

                slot_time = now.replace(hour=adjusted_hour, minute=adjusted_minute, second=0, microsecond=0)
                time_diff = (now - slot_time).total_seconds()  # positive = we're past the slot

                if 0 <= time_diff <= _window_secs and slot_key not in POSTED_SLOTS:
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
                            # Auto-weight: swap underperforming types for better ones
                            effective_type = _get_auto_weighted_post_type(post_type)
                            result = await post_with_account(account_poster, main_poster, effective_type)
                        except Exception as e:
                            logger.error(f"Error posting for {account.name}: {e}")
                            result = {'success': False, 'error': str(e)}
                        
                        if result and result.get('success'):
                            logger.info(f"✅ [{account.name}] Auto-posted {post_type} at {slot_key}")
                            POSTED_SLOTS.add(account_slot_key)
                            posted_any = True
                            _save_posted_slots_to_disk(POSTED_SLOTS, SLOT_OFFSETS)
                            # Save tweet_id for engagement tracking
                            _tid = result.get('tweet_id')
                            if _tid:
                                _save_post_to_metrics_db(
                                    str(_tid), account.name, post_type,
                                    result.get('text', '')
                                )
                            await notify_admin_post_result(account.name, post_type, True)
                        else:
                            error_msg = result.get('error', 'Unknown error') if result else 'No result returned'
                            logger.warning(f"⚠️ [{account.name}] Failed to auto-post {post_type}: {error_msg}")
                            # NOTE: Do NOT add account_slot_key to POSTED_SLOTS here —
                            # we want the loop to retry on the next 2-min cycle while still
                            # inside the slot's posting window. The slot window naturally
                            # expires after `_window_secs`, so retries are time-bounded.
                            await notify_admin_post_result(account.name, post_type, False, error_msg)
                    
                    # Only mark the base slot as "done" if at least one account posted
                    # successfully. If everyone failed, leave it open so the next loop
                    # iteration (~2 min later) can try again within the slot window.
                    if posted_any:
                        POSTED_SLOTS.add(slot_key)
                        _save_slot_to_db(slot_key)          # DB — survives restarts
                        _save_posted_slots_to_disk(POSTED_SLOTS, SLOT_OFFSETS)  # /tmp/ cache

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
        if post_type == 'featured_coin':
            # Get gainers and pick one — prioritise coins trending on X today
            gainers = await main_poster.get_top_gainers_data(30)
            if not gainers:
                return None

            # Today's X trending coins (high-engagement on X right now)
            trending_symbols = {c["symbol"] for c in get_todays_trending_coins(15)}

            valid_coins = []
            for coin in gainers:
                symbol = coin['symbol']
                has_volume = coin.get('volume', 0) >= 5_000_000
                not_overposted = check_global_coin_cooldown(symbol, max_per_day=1)
                if has_volume and not_overposted:
                    valid_coins.append(coin)
                elif not not_overposted:
                    logger.info(f"[MultiAccount] Skipping ${symbol} - already posted today (1x max)")

            # ── Tier 1: trending on X AND pumping in price ──────────────
            tier1 = [c for c in valid_coins if c['symbol'] in trending_symbols]
            # ── Tier 2: valid price gainers not in X trends ─────────────
            tier2 = [c for c in valid_coins if c['symbol'] not in trending_symbols]

            if tier1:
                featured = random.choice(tier1)
                logger.info(f"[MultiAccount] 🔥 Trend+price pick: ${featured['symbol']} (trending on X today)")
            elif tier2:
                featured = random.choice(tier2)
                logger.info(f"[MultiAccount] Price pick: ${featured['symbol']} from {len(tier2)} valid coins")
            elif valid_coins:
                featured = random.choice(valid_coins)
                logger.info(f"[MultiAccount] Fallback pick: ${featured['symbol']}")
            else:
                fallback_coins = [c for c in gainers if check_global_coin_cooldown(c['symbol'], max_per_day=1)]
                featured = random.choice(fallback_coins) if fallback_coins else random.choice(gainers)
                logger.info(f"[MultiAccount] Last resort: ${featured['symbol']}")
            
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
        
        elif post_type == 'tradehub_promo':
            # While the Bitunix campaign is active, ~1-in-4 promo slots becomes
            # a campaign tweet. Keeps coverage without flooding the timeline.
            _now = datetime.utcnow()
            _campaign_active = BITUNIX_CAMPAIGN_START <= _now <= BITUNIX_CAMPAIGN_END
            if _campaign_active and random.random() < 0.25:
                return await post_bitunix_campaign(account_poster)
            return await post_tradehub_promo(account_poster)

        elif post_type == 'market_take':
            return await post_market_take(account_poster)

        elif post_type == 'bitunix_campaign':
            return await post_bitunix_campaign(account_poster)

        elif post_type == 'yubit_campaign':
            return await post_yubit_campaign(account_poster)

        elif post_type == 'bydfi_campaign':
            return await post_bydfi_campaign(account_poster)

        elif post_type == 'top_gainer_ta':
            return await post_top_gainer_ta(account_poster, main_poster)

        elif post_type == 'free_telegram':
            return await post_free_telegram_promo(account_poster)

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
    """Post a human-like AI reaction to a real crypto news article"""
    try:
        from app.services.news_monitor import NewsMonitor

        news_monitor = NewsMonitor()
        articles = await news_monitor.fetch_recent_news(items=20, date_filter="last24hours")

        if not articles:
            logger.info("No recent news for social post")
            return None

        with_tickers = [a for a in articles[:15] if a.get('tickers')]
        pool = with_tickers[:8] if with_tickers else articles[:8]
        article = random.choice(pool)

        title = article.get('title', '')[:200]
        text_snippet = (article.get('text', '') or '')[:300]
        tickers = article.get('tickers', [])
        sentiment = (article.get('sentiment', '') or '').lower()

        coin_refs = ' '.join([f'${t}' for t in tickers[:3]]) if tickers else ''

        personality = _pick_personality()
        tweet_length = _pick_tweet_length()
        examples_str = "\n".join([f'"{e}"' for e in personality['examples']])

        if tweet_length == 'ultra_short':
            length_instruction = "1 very short sentence. Under 60 characters. Knee-jerk reaction, barely a thought."
            max_chars = 90
        elif tweet_length == 'short':
            length_instruction = "1-2 sentences. Under 130 characters. Quick personal take."
            max_chars = 150
        elif tweet_length == 'long':
            length_instruction = "3-5 sentences with line breaks between thoughts. 180-270 characters. Genuine reaction with some real depth."
            max_chars = 280
        else:
            length_instruction = "2-3 sentences. 100-180 characters. A complete thought, not overdone."
            max_chars = 200

        reaction_angle = random.choice([
            "what this means for price action or your position",
            "something ironic or obvious about this news",
            "how this changes your view or what you're doing because of it",
            "what most people are missing about this story",
            "your honest gut reaction as a trader",
            "whether this is actually news or just noise",
        ])

        sentiment_hint = ""
        if sentiment == 'positive':
            sentiment_hint = "The news is broadly positive/bullish."
        elif sentiment == 'negative':
            sentiment_hint = "The news is broadly negative/bearish."

        prompt = f"""You are a real crypto trader reacting to news on Twitter/X. You are NOT a news bot. You are a person with opinions.

YOUR PERSONALITY: {personality['name']}
{personality['voice']}

EXAMPLES OF YOUR VOICE (match this energy exactly):
{examples_str}

NEWS YOU JUST READ:
Headline: {title}
{f"Context: {text_snippet}" if text_snippet else ""}
{sentiment_hint}
{f"Coins involved: {coin_refs}" if coin_refs else ""}

YOUR TASK: React to this news from angle: {reaction_angle}

Dont just restate the headline. Give your genuine reaction as this personality. You might agree, disagree, find it obvious, find it alarming, find it funny, or find it irrelevant.

LENGTH: {length_instruction}

CRITICAL RULES - BREAK ANY AND THE TWEET IS REJECTED:
1. NO emojis whatsoever. Zero
2. NO hashtags
3. NO bullet points or lists
4. NO "not financial advice" or "NFA" or "DYOR"
5. NO ALL CAPS (except $TICKER format)
6. NO exclamation marks
7. NO news-anchor language (breaking, alert, just in, heads up)
8. NO restating the headline word for word
9. Use $TICKER format when mentioning coins
10. lowercase is preferred, imperfect grammar is fine
11. Sound like a person talking, not a content creator

Write ONLY the tweet. No quotes, no explanation:"""

        tweet = None

        # Grok first — native X/Twitter training makes it the best for news reactions
        tweet = await _call_grok_tweet(prompt, max_chars, label="news_social")

        if not tweet:
            try:
                from google import genai as _genai
                gemini_key = os.getenv('AI_INTEGRATIONS_GEMINI_API_KEY') or os.getenv('GEMINI_API_KEY')
                if gemini_key:
                    _gc = _genai.Client(api_key=gemini_key)
                    resp = await asyncio.to_thread(
                        lambda: _gc.models.generate_content(model="gemini-2.0-flash", contents=prompt)
                    )
                    candidate = (resp.text or "").strip().strip('"').strip("'").strip('```').strip()
                    candidate = candidate.replace('**', '')
                    if candidate and 5 < len(candidate) <= max_chars:
                        tweet = candidate
                        logger.info(f"📰 News tweet [Gemini/{personality['name']}]: {tweet[:70]}...")
            except Exception as e:
                logger.warning(f"Gemini news tweet failed: {e}")

        if not tweet:
            try:
                import anthropic as _anthropic
                claude_key = os.getenv('ANTHROPIC_API_KEY')
                if claude_key:
                    _cc = _anthropic.Anthropic(api_key=claude_key)
                    max_tokens = 40 if tweet_length == 'ultra_short' else 80 if tweet_length == 'short' else 200 if tweet_length == 'long' else 120
                    resp = await asyncio.to_thread(
                        lambda: _cc.messages.create(
                            model="claude-sonnet-4-5",
                            max_tokens=max_tokens,
                            messages=[{"role": "user", "content": prompt}]
                        )
                    )
                    candidate = resp.content[0].text.strip().strip('"').strip("'").strip('```').strip()
                    candidate = candidate.replace('**', '')
                    if candidate and 5 < len(candidate) <= max_chars:
                        tweet = candidate
                        logger.info(f"📰 News tweet [Claude/{personality['name']}]: {tweet[:70]}...")
            except Exception as e:
                logger.warning(f"Claude news tweet failed: {e}")

        if not tweet:
            logger.info("📰 News tweet generation produced no usable output")
            return None

        return account_poster.post_tweet(tweet)

    except Exception as e:
        logger.error(f"Error posting social news: {e}")
        return None


async def tweet_news_event_reaction(headline: str, trigger: str, direction: str, symbol: str) -> Optional[Dict]:
    """
    Claude writes and posts a reactive tweet when a breaking news event fires a trading signal.
    Called automatically from the news scanner when a signal is confirmed.
    Uses Claude first (better at nuanced geopolitical/macro commentary), Gemini fallback.
    """
    try:
        poster = get_twitter_poster()
        if not poster or not poster.client:
            logger.info("📰 NEWS TWEET: No Twitter client, skipping")
            return None

        personality = _pick_personality()
        tweet_length = _pick_tweet_length()
        examples_str = "\n".join([f'"{e}"' for e in personality['examples']])
        coin = symbol.replace('USDT', '').replace('/USDT:USDT', '')
        direction_context = "bearish/risk-off (expecting price to fall)" if direction == 'SHORT' else "bullish/risk-on (expecting price to rise)"

        if tweet_length == 'ultra_short':
            length_instruction = "1 very short sentence. Under 60 characters. Knee-jerk reaction."
            max_chars = 90
        elif tweet_length == 'short':
            length_instruction = "1-2 sentences. Under 130 characters. Quick personal take."
            max_chars = 150
        elif tweet_length == 'long':
            length_instruction = "3-4 sentences with line breaks. 180-260 characters. Real depth, genuine reaction."
            max_chars = 280
        else:
            length_instruction = "2-3 sentences. 100-180 characters. Complete thought, not overdone."
            max_chars = 200

        reaction_angle = random.choice([
            f"what this means for ${coin} and crypto broadly",
            "whether this is actually news or noise that gets priced in fast",
            "your honest gut reaction as a trader who just read this",
            "what most people are missing about how this hits markets",
            f"your view on the {direction_context} trade setup this creates",
            "how quickly you think this gets fully priced in",
        ])

        prompt = f"""You are a real crypto trader reacting to breaking news on Twitter/X. You are NOT a news bot.

YOUR PERSONALITY: {personality['name']}
{personality['voice']}

EXAMPLES OF YOUR VOICE (match this energy exactly):
{examples_str}

BREAKING NEWS YOU JUST SAW:
Headline: {headline}
Key trigger: {trigger}
Your read: {direction_context} for crypto

YOUR TASK: React to this from angle: {reaction_angle}

Do NOT restate the headline. Give your genuine take as a trader who sees the market implication immediately.

LENGTH: {length_instruction}

CRITICAL RULES - BREAK ANY AND THE TWEET IS REJECTED:
1. NO emojis whatsoever. Zero
2. NO hashtags
3. NO bullet points or lists
4. NO "not financial advice" or "NFA" or "DYOR"
5. NO ALL CAPS (except $TICKER)
6. NO exclamation marks
7. NO news-anchor phrases (breaking, alert, just in)
8. lowercase preferred, imperfect grammar is fine
9. Sound like a person, not a content creator

Write ONLY the tweet. No quotes, no explanation:"""

        tweet = None

        # Grok first — native X/Twitter training, best for breaking news reactions
        tweet = await _call_grok_tweet(prompt, max_chars, label="news_reaction")

        # Claude second — strong on geopolitical/macro nuance
        if not tweet:
            try:
                import anthropic as _anthropic
                claude_key = os.getenv('AI_INTEGRATIONS_ANTHROPIC_API_KEY') or os.getenv('ANTHROPIC_API_KEY')
                if claude_key:
                    _cc = _anthropic.Anthropic(api_key=claude_key)
                    max_tokens = 40 if tweet_length == 'ultra_short' else 80 if tweet_length == 'short' else 200 if tweet_length == 'long' else 120
                    resp = await asyncio.to_thread(
                        lambda: _cc.messages.create(
                            model="claude-sonnet-4-5-20250929",
                            max_tokens=max_tokens,
                            messages=[{"role": "user", "content": prompt}]
                        )
                    )
                    candidate = resp.content[0].text.strip().strip('"').strip("'").strip('```').strip()
                    candidate = candidate.replace('**', '')
                    if candidate and 5 < len(candidate) <= max_chars:
                        tweet = candidate
                        logger.info(f"📰 NEWS REACTION TWEET [Claude/{personality['name']}]: {tweet[:70]}...")
            except Exception as e:
                logger.warning(f"Claude news reaction tweet failed: {e}")

        # Gemini fallback
        if not tweet:
            try:
                from google import genai as _genai
                gemini_key = os.getenv('AI_INTEGRATIONS_GEMINI_API_KEY') or os.getenv('GEMINI_API_KEY')
                if gemini_key:
                    _gc = _genai.Client(api_key=gemini_key)
                    resp = await asyncio.to_thread(
                        lambda: _gc.models.generate_content(model="gemini-2.0-flash", contents=prompt)
                    )
                    candidate = (resp.text or "").strip().strip('"').strip("'").strip('```').strip()
                    candidate = candidate.replace('**', '')
                    if candidate and 5 < len(candidate) <= max_chars:
                        tweet = candidate
                        logger.info(f"📰 NEWS REACTION TWEET [Gemini/{personality['name']}]: {tweet[:70]}...")
            except Exception as e:
                logger.warning(f"Gemini news reaction tweet failed: {e}")

        if not tweet:
            logger.info("📰 NEWS REACTION TWEET: No usable output generated")
            return None

        result = await asyncio.to_thread(lambda: poster.post_tweet(tweet))
        if result and result.get('success'):
            logger.info(f"📰 NEWS REACTION TWEET posted: {tweet[:80]}...")
        return result

    except Exception as e:
        logger.error(f"Error posting news reaction tweet: {e}")
        return None


async def post_early_gainers(account_poster: MultiAccountPoster) -> Optional[Dict]:
    """Post coins gaining traction early with human-like variety (via MEXC)"""
    try:
        all_tickers = await _fetch_mexc_tickers()

        _eg_excl = {'USDC', 'BUSD', 'DAI', 'TUSD', 'USDP', 'USDD'}
        early_movers = [
            t for t in all_tickers
            if 3 <= t['change'] <= 12
            and t.get('volume', 0) >= 5_000_000
            and t['symbol'] not in _eg_excl
            and check_global_coin_cooldown(t['symbol'], max_per_day=1)
        ]
        
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
                f"${sym} early. I see it.",
                f"${sym} +{chg:.1f}%. quiet, for now.",
                f"${sym} on the radar. +{chg:.1f}%.",
                f"${sym}. early stages. watching.",
            ]
        elif tl == 'short':
            templates = [
                f"${sym} up {chg:.1f}% and nobody's talking about it. that's usually when I'm most interested.",
                f"${sym} +{chg:.1f}% on {vol_str} volume. flagged this one before it was loud.",
                f"${sym} +{chg:.1f}%. the ones nobody's tweeting about are always the ones worth watching.",
                f"caught ${sym} early. +{chg:.1f}%. {vol_str} volume. I don't wait for confirmation from twitter.",
                f"${sym} waking up. +{chg:.1f}%. I was already watching.",
            ]
        elif tl == 'long':
            templates = [
                f"${sym} +{chg:.1f}% on {vol_str} volume and twitter hasn't noticed yet.\n\nvolume profile looks different to a normal pump — steady accumulation across multiple candles.\n\nI've been here a while. this is what early looks like.",
                f"${sym} up {chg:.1f}% today.\n\nthis is the one that hasn't made anyone's list. I like finding these before the crowd does. sometimes they fade, sometimes they run 5x. either way — I was early.\n\n{vol_str} volume. real.",
            ]
        else:
            templates = [
                f"${sym} +{chg:.1f}% on {vol_str} volume. still early. the kind of move I catch before it's a thread.",
                f"${sym} +{chg:.1f}% and gaining. not making headlines yet — that's the point.",
                f"under the radar: ${sym} up {chg:.1f}% on {vol_str}. found it before the tweet storm.",
                f"${sym} on my scanner. +{chg:.1f}%. volume confirming. early but the setup is there.",
            ]

        if len(early_movers) >= 3 and random.random() < 0.3:
            list_templates = [
                f"things catching my eye right now:\n\n" + "\n".join([f"${m['symbol']} +{m['change']:.1f}%" for m in early_movers[:4]]) + "\n\nstill early on all of these. I'm watching.",
                f"early movers before twitter notices:\n\n" + "\n".join([f"${m['symbol']} +{m['change']:.1f}%" for m in early_movers[:3]]) + "\n\nvolume is real on all three.",
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
    """Post coins showing momentum shifts with human-like variety (via MEXC)"""
    try:
        all_tickers = await _fetch_mexc_tickers()

        _ms_excl = {'USDC', 'BUSD', 'DAI', 'TUSD', 'USDP'}
        movers = [
            t for t in all_tickers
            if t['change'] >= 5
            and t.get('volume', 0) >= 10_000_000
            and t['symbol'] not in _ms_excl
            and check_global_coin_cooldown(t['symbol'], max_per_day=1)
        ]
        
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
    """Post coins with unusual volume (via MEXC — Binance geoblocked on Replit)"""
    try:
        all_tickers = await _fetch_mexc_tickers()

        _vs_excl = {'USDC', 'BUSD', 'DAI', 'TUSD', 'BTC', 'ETH'}
        high_volume = [
            t for t in all_tickers
            if t.get('volume', 0) >= 50_000_000
            and t['symbol'] not in _vs_excl
            and check_global_coin_cooldown(t['symbol'], max_per_day=1)
        ]
        
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
        import httpx as _httpx
        async with _httpx.AsyncClient(timeout=10.0) as _cl:
            rb = await _cl.get("https://api.mexc.com/api/v3/ticker/24hr", params={"symbol": "BTCUSDT"})
            re = await _cl.get("https://api.mexc.com/api/v3/ticker/24hr", params={"symbol": "ETHUSDT"})
        btc_t = rb.json(); eth_t = re.json()
        btc_change = float(btc_t.get('priceChangePercent', 0) or 0)
        eth_change = float(eth_t.get('priceChangePercent', 0) or 0)

        all_tickers = await _fetch_mexc_tickers()

        # Count green vs red
        green = sum(1 for t in all_tickers if t.get('change', 0) > 0)
        red   = sum(1 for t in all_tickers if t.get('change', 0) < 0)
        
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


def _try_attach_card(account_poster: MultiAccountPoster, image_bytes: bytes) -> Optional[list]:
    """Upload image_bytes to Twitter v1 and return [media_id], or None on failure."""
    try:
        if hasattr(account_poster, 'api_v1') and account_poster.api_v1:
            media = account_poster.api_v1.media_upload(
                filename="tradehub_card.png", file=io.BytesIO(image_bytes)
            )
            return [str(media.media_id)]
    except Exception as e:
        logger.warning(f"Card image upload failed (non-fatal): {e}")
    return None


async def post_early_gainer_standard(account_poster: MultiAccountPoster, main_poster) -> Optional[Dict]:
    """Post early gainer for standard accounts with chart - human-like varied posts"""
    try:
        gainers = await main_poster.get_top_gainers_data(20)
        if not gainers:
            return {'success': False, 'error': 'No gainers data'}

        # Find early gainers (3–20% range with real volume)
        early_gainers = [
            g for g in gainers
            if 3 <= g.get('change', 0) <= 20
            and g.get('volume', 0) >= 3_000_000
            and check_global_coin_cooldown(g['symbol'], max_per_day=1)
        ]
        if not early_gainers:
            early_gainers = [g for g in gainers if g.get('change', 0) >= 3][:3]
        if not early_gainers:
            return {'success': False, 'error': 'No suitable early gainers'}

        coin   = random.choice(early_gainers[:5])
        symbol = coin['symbol']
        change = coin.get('change', 0)
        price  = coin.get('price', 0)
        volume = coin.get('volume', 0)
        mcap   = coin.get('market_cap', 0)
        rank   = early_gainers.index(coin) + 1

        if price < 0.01:
            price_str = f"${price:.6f}"
        elif price < 1:
            price_str = f"${price:.4f}"
        elif price < 100:
            price_str = f"${price:,.3f}"
        else:
            price_str = f"${price:,.2f}"

        vol_str  = f"${volume/1e6:.1f}M" if volume < 1e9 else f"${volume/1e9:.2f}B"
        mcap_str = f"${mcap/1e9:.2f}B"   if mcap >= 1e9 else (f"${mcap/1e6:.0f}M" if mcap > 0 else "")
        cap_note = f" | mcap {mcap_str}" if mcap_str else ""

        # AI-generated card images disabled — text-only posts
        media_ids = None

        # Fetch 1h OHLCV for TA enrichment (~50% of posts)
        ta = None
        if random.random() < 0.5:
            try:
                ohlcv = await _fetch_mexc_ohlcv(symbol, interval='1h', limit=48)
                if ohlcv and len(ohlcv) >= 20:
                    closes = [float(c[4]) for c in ohlcv]
                    highs  = [float(c[2]) for c in ohlcv]
                    lows   = [float(c[3]) for c in ohlcv]
                    vols   = [float(c[5]) for c in ohlcv]
                    # True exponential EMA: seed from SMA then walk forward
                    def _ema(series: list, period: int) -> float:
                        if len(series) < period:
                            return sum(series) / len(series)
                        k = 2.0 / (period + 1)
                        val = sum(series[:period]) / period  # SMA seed
                        for price in series[period:]:
                            val = price * k + val * (1 - k)
                        return val
                    ema9   = _ema(closes, 9)
                    ema21  = _ema(closes, 21)
                    # RSI-14 on the most recent 15 closes (→ 14 deltas)
                    recent_closes = closes[-15:]
                    gains, losses = [], []
                    for i in range(1, len(recent_closes)):
                        diff = recent_closes[i] - recent_closes[i - 1]
                        gains.append(max(diff, 0))
                        losses.append(max(-diff, 0))
                    avg_gain = sum(gains) / len(gains) if gains else 0
                    avg_loss = sum(losses) / len(losses) if losses else 0.0001
                    rsi_val  = round(100 - (100 / (1 + avg_gain / avg_loss)), 1)
                    trend_val = "bullish" if ema9 > ema21 else "bearish"
                    avg_vol10 = sum(vols[-10:]) / 10 if len(vols) >= 10 else 0
                    vol_above_avg = vols[-1] > avg_vol10 * 1.2 if avg_vol10 > 0 else False
                    h24 = max(highs[-24:]) if len(highs) >= 24 else max(highs)
                    l24 = min(lows[-24:])  if len(lows)  >= 24 else min(lows)
                    ta = {
                        'rsi': rsi_val,
                        'trend': trend_val,
                        'vol_above_avg': vol_above_avg,
                        'h24': h24,
                        'l24': l24,
                    }
            except Exception:
                pass

        # Build coin data for AI generation — pass TA if available
        coin_data_for_ai = {
            'symbol': symbol, 'change': change, 'price': price,
            'volume': volume, 'market_cap': mcap if mcap > 0 else None,
        }
        if ta:
            coin_data_for_ai['rsi']   = ta['rsi']
            coin_data_for_ai['trend'] = ta['trend']

        tweet_text = await generate_ai_tweet(coin_data_for_ai, 'early_gainer')
        if not tweet_text:
            _sign = "+" if change >= 0 else ""
            tweet_text = f"${symbol} {_sign}{change:.1f}% at {price_str}. watching this one."

        tweet_text = tweet_text + _get_hashtag_style()
        _yd = _maybe_yubit_drop()
        if _yd and (len(tweet_text + _yd) - 32) <= 280:
            tweet_text = tweet_text + _yd
        tweet_text = await _ai_review_tweet(tweet_text, 'early_gainer', {
            'symbol': symbol, 'change': f'{change:+.1f}%', 'price': price_str,
            'volume': vol_str, 'ta_available': ta is not None,
        })
        result = account_poster.post_tweet(tweet_text, media_ids=media_ids)

        if result and result.get('success'):
            record_global_coin_post(symbol)

        return result

    except Exception as e:
        logger.error(f"Error posting early gainer standard: {e}")
        return {'success': False, 'error': str(e)}


async def post_memecoin(account_poster: MultiAccountPoster) -> Optional[Dict]:
    """Post about a trending meme coin from CoinGecko top gainers in KOL voice."""
    MEME_COINS = {
        'DOGE', 'SHIB', 'PEPE', 'FLOKI', 'BONK', 'WIF', 'MEME', 'TURBO',
        'NEIRO', 'BOME', 'BRETT', 'MOG', 'POPCAT', 'BABYDOGE', 'ELON',
        'WOJAK', 'LADYS', 'MONG', 'BOB', 'TOSHI', 'SPX', 'GROK', 'COQ',
        'MYRO', 'SLERF', 'WEN', 'BODEN', 'PORK', 'BILLY', 'MAGA', 'TRUMP',
        'MELANIA', 'FARTCOIN', 'PONKE', 'MICHI', 'GOAT', 'ACT', 'MOODENG',
        'PNUT', 'CHILLGUY', 'VIRTUAL', 'GRIFFAIN',
    }

    try:
        import httpx as _httpx
        # Fetch top 100 coins by 24h gain — then filter for known memes
        async with _httpx.AsyncClient(timeout=10.0) as client:
            r = await client.get(
                "https://api.coingecko.com/api/v3/coins/markets",
                params={
                    "vs_currency": "usd",
                    "order": "percent_change_24h_desc",
                    "per_page": 100,
                    "page": 1,
                    "sparkline": False,
                }
            )

        if r.status_code != 200:
            return {'success': False, 'error': f'CoinGecko error {r.status_code}'}

        all_coins = r.json()
        # Prefer known meme coins that are actually moving; fall back to any top gainer
        meme_gainers = [
            c for c in all_coins
            if c.get("symbol", "").upper() in MEME_COINS
            and (c.get("price_change_percentage_24h") or 0) >= 3
            and (c.get("total_volume") or 0) >= 5_000_000
            and check_global_coin_cooldown(c["symbol"].upper(), max_per_day=1)
        ]
        if not meme_gainers:
            meme_gainers = [
                c for c in all_coins
                if c.get("symbol", "").upper() in MEME_COINS
                and check_global_coin_cooldown(c["symbol"].upper(), max_per_day=1)
            ][:5]
        if not meme_gainers:
            meme_gainers = [all_coins[0]] if all_coins else []

        if not meme_gainers:
            return {'success': False, 'error': 'No meme coin data'}

        coin    = random.choice(meme_gainers[:5])
        symbol  = coin.get("symbol", "").upper()
        change  = coin.get("price_change_percentage_24h") or 0
        price   = coin.get("current_price") or 0
        volume  = coin.get("total_volume") or 0
        mcap    = coin.get("market_cap") or 0
        sign    = "+" if change >= 0 else ""

        if price < 0.000001:
            price_str = f"${price:.10f}"
        elif price < 0.01:
            price_str = f"${price:.8f}"
        elif price < 1:
            price_str = f"${price:.6f}"
        else:
            price_str = f"${price:,.4f}"

        vol_str  = f"${volume/1e6:.1f}M" if volume < 1e9 else f"${volume/1e9:.1f}B"
        mcap_str = f"${mcap/1e9:.2f}B" if mcap >= 1e9 else (f"${mcap/1e6:.0f}M" if mcap > 0 else "")
        cap_note = f" (mcap {mcap_str})" if mcap_str else ""

        # AI-generated card images disabled — text-only posts
        meme_media_ids = None

        coin_data_for_ai = {
            'symbol': symbol, 'change': change, 'price': price,
            'volume': volume, 'market_cap': mcap if mcap > 0 else None,
        }
        tweet_text = await generate_ai_tweet(coin_data_for_ai, 'memecoin')
        if not tweet_text:
            tweet_text = f"${symbol} {sign}{change:.1f}% at {price_str}. {vol_str} behind it."

        tweet_text = tweet_text + _get_hashtag_style()
        tweet_text = await _ai_review_tweet(tweet_text, 'memecoin', {
            'symbol': symbol, 'change': f'{sign}{change:.1f}%', 'price': price_str,
            'volume': vol_str, 'mcap': mcap_str,
        })
        result = account_poster.post_tweet(tweet_text, media_ids=meme_media_ids)

        if result and result.get('success'):
            record_global_coin_post(symbol)

        return result

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
        
        templates = [
            f"something unusual happening with ${symbol}. {vol_str} in volume today — that's not normal activity for this coin. price is {sign}{change:.1f}% at {price_str}. when volume spikes like this, someone usually knows something.",
            f"the volume on ${symbol} caught me off guard. {vol_str} in 24h and the price is {sign}{change:.1f}% at {price_str}. this kind of volume doesn't show up randomly.",
            f"${symbol} is trading {vol_str} in volume today and I can't ignore it. price is sitting at {price_str}, {sign}{change:.1f}% on the day. either there's positioning happening or news I haven't seen yet.",
            f"been watching ${symbol} and the volume just told me something. {vol_str} in 24 hours — that's real size moving. price is {sign}{change:.1f}% at {price_str}. I always pay attention when volume does this.",
            f"${symbol} volume alert. {vol_str} traded today. price at {price_str}, {sign}{change:.1f}% on the day. smart money tends to be early — and this much volume doesn't come from retail.\n\nsomething is happening here.",
            f"can't ignore {vol_str} in volume on ${symbol}. the price is {sign}{change:.1f}% at {price_str} and the volume profile looks nothing like a normal day.\n\nwhen I see this pattern I stop and pay attention.",
            f"{vol_str} in volume on ${symbol} today. that's not retail buying.\n\nprice is {sign}{change:.1f}% at {price_str} and the move came with enough size that someone with serious capital was involved. watching closely.",
            f"spotted unusual activity on ${symbol} — {vol_str} in 24h volume, price at {price_str} ({sign}{change:.1f}%). this is exactly the kind of setup I track. nothing moves like this without a reason.",
        ]

        tweet_text = random.choice(templates)
        tweet_text = await _ai_review_tweet(tweet_text, 'whale_alert', {
            'symbol': symbol, 'change': f'{sign}{change:.1f}%', 'price': price_str,
            'volume_24h': vol_str, 'signal': 'unusual volume spike',
        })
        result = account_poster.post_tweet(tweet_text)
        
        if result and result.get('success'):
            record_global_coin_post(symbol)
        
        return result
        
    except Exception as e:
        logger.error(f"Error posting whale alert: {e}")
        return {'success': False, 'error': str(e)}


async def post_funding_extreme(account_poster: MultiAccountPoster) -> Optional[Dict]:
    """Post about extreme funding rates via MEXC (Binance geoblocked on Replit)"""
    try:
        import httpx as _httpx
        async with _httpx.AsyncClient(timeout=12.0) as _cl:
            resp = await _cl.get("https://contract.mexc.com/api/v1/contract/funding_rate")
        raw = resp.json()
        funding_list = raw.get('data', []) if isinstance(raw, dict) else raw
        if not funding_list:
            return {'success': False, 'error': 'No funding data'}

        # Find extreme funding rates
        extremes = []
        for item in funding_list:
            rate = float(item.get('fundingRate', 0) or 0)
            if abs(rate) >= 0.0005:
                base = item.get('symbol', '').replace('_USDT', '')
                extremes.append({
                    'symbol': base,
                    'rate': rate * 100,
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
        
        funding_sign = '+' if is_long_crowded else '-'
        crowd_side   = "longs" if is_long_crowded else "shorts"
        other_side   = "shorts" if is_long_crowded else "longs"
        squeeze_type = "short squeeze" if not is_long_crowded else "long squeeze"

        if is_long_crowded:
            templates = [
                f"${top['symbol']} funding rate hit {rate_str}. that's a lot of longs paying shorts every 8 hours. when everyone's on one side of the trade, you know what usually happens — the market finds a way to embarrass the crowd.",
                f"everyone's long on ${top['symbol']} right now. funding at {funding_sign}{rate_str}. these crowded positions have a way of reversing when you least expect it. not a prediction, just an observation worth tracking.",
                f"${top['symbol']} funding at {rate_str} is getting uncomfortable for the longs. that cost adds up fast. if buyers don't follow through soon, the unwind can be brutal.",
                f"${top['symbol']} funding is at {rate_str}. one of the higher readings I've seen recently. when the crowd is this unanimously long, the market usually finds a reason to prove them wrong.",
                f"been watching the funding on ${top['symbol']} and it's at {rate_str} now. longs are paying heavily. the market has a way of punishing overcrowded positions — not always immediately, but eventually.",
                f"funding rate on ${top['symbol']}: {rate_str}. the {crowd_side} are paying up to hold their positions. historically this kind of reading tends to precede some turbulence. watching carefully.",
            ]
        else:
            templates = [
                f"${top['symbol']} funding is at -{rate_str} — shorts are paying longs every 8 hours. that's a real cost for holding short. if price starts moving up, the unwind could be fast.",
                f"everyone's short on ${top['symbol']} right now. funding at {funding_sign}{rate_str}. short squeezes from these setups can be violent because the cost of staying short keeps climbing.",
                f"${top['symbol']} funding at -{rate_str}. shorts are extremely crowded here. it's not always wrong to be in the crowd — but it's always worth knowing when you are.",
                f"noticed ${top['symbol']} funding is at -{rate_str}. that's significant short-side crowding. the crowd might be right about direction but the funding cost makes it expensive to stay in this trade.",
                f"funding rate alert on ${top['symbol']}: -{rate_str}. {crowd_side} are paying to hold. if buyers step in, the {other_side} who've been holding here could see a nasty reversal.",
                f"${top['symbol']} is looking like a potential {squeeze_type} setup. funding at -{rate_str} means {crowd_side} are paying up. one good catalyst and this gets violent.",
            ]

        tweet_text = random.choice(templates)
        tweet_text = await _ai_review_tweet(tweet_text, 'funding_extreme', {
            'symbol': top['symbol'], 'funding_rate': rate_str,
            'bias': 'short-heavy' if top.get('fundingRate', 0) < 0 else 'long-heavy',
            'signal': 'extreme funding rate / potential squeeze setup',
        })
        result = account_poster.post_tweet(tweet_text)
        
        if result and result.get('success'):
            record_global_coin_post(top['symbol'])
        
        return result
        
    except Exception as e:
        logger.error(f"Error posting funding extreme: {e}")
        return {'success': False, 'error': str(e)}


async def post_quick_ta(account_poster: MultiAccountPoster, main_poster) -> Optional[Dict]:
    """Post quick technical analysis — picks coins trending on X today first"""
    try:
        gainers = await main_poster.get_top_gainers_data(30)
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

        # Prioritise coins that are also trending on X today
        trending_symbols = {c["symbol"] for c in get_todays_trending_coins(15)}
        trending_candidates = [c for c in candidates if c['symbol'] in trending_symbols]
        if trending_candidates:
            coin = random.choice(trending_candidates)
            logger.info(f"[QuickTA] 🔥 Picked ${coin['symbol']} — trending on X today")
        else:
            coin = random.choice(candidates[:5])
            logger.info(f"[QuickTA] Picked ${coin['symbol']} from top movers")
        symbol = coin['symbol']
        change = coin.get('change', 0)
        price = coin.get('price', 0)
        
        price_str = f"${price:,.4f}" if price < 1 else f"${price:,.2f}"
        sign = "+" if change >= 0 else ""
        
        chart_analysis = None
        ohlcv_closes = []
        try:
            ohlcv = await _fetch_mexc_ohlcv(symbol, interval='1h', limit=48)
            if ohlcv and len(ohlcv) >= 20:
                closes = [float(c[4]) for c in ohlcv]
                ohlcv_closes = closes
                highs  = [float(c[2]) for c in ohlcv]
                lows   = [float(c[3]) for c in ohlcv]
                vols   = [float(c[5]) for c in ohlcv]
                ema9   = sum(closes[-9:])  / 9
                ema21  = sum(closes[-21:]) / 21
                ema50  = sum(closes[-50:]) / min(50, len(closes))
                gains, losses = [], []
                for i in range(1, min(15, len(closes))):
                    diff = closes[i] - closes[i-1]
                    gains.append(max(diff, 0))
                    losses.append(max(-diff, 0))
                avg_gain = sum(gains) / len(gains) if gains else 0
                avg_loss = sum(losses) / len(losses) if losses else 0.0001
                rsi_val = 100 - (100 / (1 + avg_gain / avg_loss))
                trend_val = "bullish" if ema9 > ema21 else "bearish" if ema9 < ema21 else "neutral"
                # Volume vs 10-period average
                avg_vol = sum(vols[-10:]) / 10 if len(vols) >= 10 else 0
                vol_surge = vols[-1] > avg_vol * 1.3 if avg_vol > 0 else False
                # 24h high/low range
                h24 = max(highs[-24:]) if len(highs) >= 24 else max(highs)
                l24 = min(lows[-24:])  if len(lows)  >= 24 else min(lows)
                chart_analysis = {
                    'rsi': round(rsi_val, 1),
                    'trend': trend_val,
                    'ema9': round(ema9, 6),
                    'ema21': round(ema21, 6),
                    'vol_surge': vol_surge,
                    'h24': h24,
                    'l24': l24,
                }
        except Exception as e:
            logger.warning(f"TA analysis failed: {e}")

        volume = coin.get('volume', 0)

        # AI-generated card images disabled — text-only posts
        ta_media_ids = None

        # Build coin data for AI generation — pass TA if available
        coin_data_for_ai = {
            'symbol': symbol, 'change': change, 'price': price, 'volume': volume,
        }
        if chart_analysis:
            coin_data_for_ai['rsi']   = chart_analysis['rsi']
            coin_data_for_ai['trend'] = chart_analysis['trend']

        tweet_text = await generate_ai_tweet(coin_data_for_ai, 'quick_ta')
        if not tweet_text:
            tweet_text = f"${symbol} {sign}{change:.1f}% at {price_str}. chart caught my eye."

        tweet_text = tweet_text + _get_hashtag_style()
        _yd = _maybe_yubit_drop()
        if _yd and (len(tweet_text + _yd) - 32) <= 280:  # -32 corrects for t.co URL shortening
            tweet_text = tweet_text + _yd
        tweet_text = await _ai_review_tweet(tweet_text, 'quick_ta', {
            'symbol': symbol, 'change': f'{sign}{change:.1f}%', 'price': price_str,
            'rsi': chart_analysis['rsi'] if chart_analysis else 'not computed',
            'trend': chart_analysis['trend'] if chart_analysis else 'neutral',
        })

        result = account_poster.post_tweet(tweet_text, media_ids=ta_media_ids)

        if result and result.get('success'):
            record_global_coin_post(symbol)

        return result

    except Exception as e:
        logger.error(f"Error posting quick TA: {e}")
        return {'success': False, 'error': str(e)}




_TOP_GAINER_ANGLE_IDX = 0

_TOP_GAINER_ANGLES = [
    {
        'id': 'breakout',
        'instruction': (
            "Focus on whether this is a genuine breakout or a potential fakeout. "
            "Reference the price structure and whether volume is confirming the move. "
            "Be direct and analytical — give your honest read on the setup."
        ),
    },
    {
        'id': 'rsi_volume',
        'instruction': (
            "Lead with the RSI and volume story. Is the move overextended or is there "
            "still room? What does the volume tell you about conviction behind this move? "
            "Keep it punchy — one clear read."
        ),
    },
    {
        'id': 'key_levels',
        'instruction': (
            "Give specific price levels. Where is current support, where is resistance, "
            "what's the level to hold for this move to stay valid. "
            "Use the 24h range and EMA data. Be precise."
        ),
    },
    {
        'id': 'momentum',
        'instruction': (
            "Talk about momentum — is it accelerating or slowing? Is the EMA structure "
            "supporting continuation? What's the realistic target if momentum holds? "
            "Short, confident, technical."
        ),
    },
    {
        'id': 'context',
        'instruction': (
            "Put the move in context — is this a continuation of a trend or a sudden spike? "
            "Is the volume unusual for this coin? What changed today? "
            "Read it like someone who's watched this market for years."
        ),
    },
    {
        'id': 'trade_idea',
        'instruction': (
            "Give a clean trade perspective. What the setup looks like, what level you'd "
            "watch for an entry, where the trade is invalidated. "
            "Not financial advice framing — just your honest technical read on the chart."
        ),
    },
]


async def post_top_gainer_ta(account_poster, main_poster) -> Optional[Dict]:
    """Post a technical analysis on today's top gainer. 6 rotating analytical angles.
    No cooldown gate — intentionally posts about the top mover multiple times/day
    from different analytical angles. Different angle = different post every time."""
    global _TOP_GAINER_ANGLE_IDX
    try:
        # ── Pick the top gainer (no cooldown — angle rotation handles variety) ─
        gainers = await main_poster.get_top_gainers_data(50)
        if not gainers:
            return {'success': False, 'error': 'No gainers data'}

        _SKIP = {
            'USDT', 'USDC', 'BUSD', 'DAI', 'TUSD', 'USDP', 'FDUSD',
            'WBTC', 'WETH', 'STETH', 'RETH',
        }
        candidates = [
            g for g in gainers
            if g.get('change', 0) >= 3
            and g.get('volume', 0) >= 1_500_000
            and g.get('symbol', '') not in _SKIP
        ]
        # Progressive volume fallback so we always find a coin
        if not candidates:
            candidates = [g for g in gainers if g.get('change', 0) >= 2 and g.get('symbol', '') not in _SKIP]
        if not candidates:
            candidates = [g for g in gainers if g.get('symbol', '') not in _SKIP][:5]
        if not candidates:
            return {'success': False, 'error': 'No suitable top gainer'}

        coin   = candidates[0]
        symbol = coin['symbol']
        change = coin.get('change', 0)
        price  = coin.get('price', 0)
        volume = coin.get('volume', 0)

        if price < 0.001:
            price_str = f"${price:.8f}"
        elif price < 0.01:
            price_str = f"${price:.6f}"
        elif price < 1:
            price_str = f"${price:.4f}"
        elif price < 100:
            price_str = f"${price:,.3f}"
        else:
            price_str = f"${price:,.2f}"

        vol_str = f"${volume/1e9:.1f}B" if volume >= 1e9 else f"${volume/1e6:.1f}M"

        # ── Fetch OHLCV and compute TA ────────────────────────────────────────
        rsi_val   = 50.0
        trend     = "neutral"
        vol_ratio = 1.0
        h24 = l24 = price
        ema9 = ema21 = price
        ema_gap_pct = 0.0
        got_ta = False

        try:
            ohlcv = await _fetch_mexc_ohlcv(symbol, interval='1h', limit=50)
            if ohlcv and len(ohlcv) >= 20:
                closes = [float(c[4]) for c in ohlcv]
                highs  = [float(c[2]) for c in ohlcv]
                lows   = [float(c[3]) for c in ohlcv]
                vols   = [float(c[5]) for c in ohlcv]

                def _ema(series, period):
                    if len(series) < period:
                        return sum(series) / len(series)
                    k = 2.0 / (period + 1)
                    val = sum(series[:period]) / period
                    for p in series[period:]:
                        val = p * k + val * (1 - k)
                    return val

                ema9  = _ema(closes, 9)
                ema21 = _ema(closes, 21)
                trend = "bullish" if ema9 > ema21 else "bearish" if ema9 < ema21 else "neutral"
                ema_gap_pct = round(abs(ema9 - ema21) / ema21 * 100, 2) if ema21 else 0

                gains_, losses_ = [], []
                for i in range(1, min(15, len(closes))):
                    d = closes[i] - closes[i-1]
                    gains_.append(max(d, 0))
                    losses_.append(max(-d, 0))
                ag = sum(gains_) / len(gains_) if gains_ else 0
                al = sum(losses_) / len(losses_) if losses_ else 0.0001
                rsi_val = round(100 - (100 / (1 + ag / al)), 1)

                avg_vol   = sum(vols[-10:]) / 10 if len(vols) >= 10 else 0
                vol_ratio = round(vols[-1] / avg_vol, 1) if avg_vol > 0 else 1.0

                h24 = max(highs[-24:]) if len(highs) >= 24 else max(highs)
                l24 = min(lows[-24:])  if len(lows)  >= 24 else min(lows)
                got_ta = True
        except Exception as te:
            logger.warning(f"[TopGainerTA] TA calc failed for {symbol}: {te}")

        # ── Build concise TA fact-sheet for the AI ───────────────────────────
        sign = "+" if change >= 0 else ""

        rsi_label = (
            "overbought" if rsi_val >= 72
            else "extended" if rsi_val >= 65
            else "healthy" if rsi_val >= 50
            else "oversold" if rsi_val <= 30
            else "neutral"
        )

        # Precise price levels for AI to pick from
        def _fmt_level(p):
            if p < 0.001:   return f"${p:.8f}"
            if p < 0.01:    return f"${p:.6f}"
            if p < 1:       return f"${p:.4f}"
            if p < 100:     return f"${p:,.3f}"
            return f"${p:,.2f}"

        ta_facts = [f"{sign}{change:.1f}% move", f"price {price_str}", f"vol {vol_str}"]
        if got_ta:
            ta_facts += [
                f"RSI {rsi_val} ({rsi_label})",
                f"EMA9 {'above' if trend == 'bullish' else 'below'} EMA21 ({trend}, gap {ema_gap_pct}%)",
                f"volume {vol_ratio}x 10-bar avg",
                f"24h range: {_fmt_level(l24)} – {_fmt_level(h24)}",
            ]

        ta_block = "\n".join(f"• {f}" for f in ta_facts)

        # ── Pick rotating angle ───────────────────────────────────────────────
        angle = _TOP_GAINER_ANGLES[_TOP_GAINER_ANGLE_IDX % len(_TOP_GAINER_ANGLES)]
        _TOP_GAINER_ANGLE_IDX += 1

        # ── Craft AI prompt — specificity beats vague "sound human" rules ────
        system_prompt = f"""You are a crypto trader writing a quick tweet about a chart you're looking at.

HARD RULES (break any = fail):
- Output ONLY the tweet. No quotes, no labels, no explanation.
- All lowercase (ticker ${symbol} stays uppercase cashtag). No sentence-case.
- Max 240 chars. Count carefully.
- No hashtags. No "NFA". No "DYOR". No "not financial advice".
- No em dashes (—). Use a comma or period instead.
- Do NOT start with: "just", "looks like", "interesting", "it's worth noting", "worth keeping an eye", "keep an eye on", "this is", "the key", "what's", "notably", "i noticed", "noticed that"

HOW TO WRITE LIKE A REAL TRADER:
- Use fragments. Real people don't write full sentences every time.
- Use actual numbers from the TA. Specificity = credibility.
- Lowercase everything except the cashtag.
- One clear thought. Don't over-explain.
- End mid-thought sometimes, no filler conclusion needed.
- You can ask a short question or just state an observation.
- Mix: sometimes 1 sentence, sometimes 2-3 short lines, sometimes a fragment with a reaction.

GOOD EXAMPLES (style only, don't copy):
- "$XYZ up 18%, rsi 71 and volume 3.4x avg. extended but still going"
- "ema9 crossed ema21 on the 1h. $XYZ watching to see if this holds above support"
- "$XYZ sitting right at the 24h high. either breaks out here or fades hard"
- "volume 4x on $XYZ. someone knows something"
- "$XYZ: rsi 68, ema trending up, support held twice. this setup i like"

Analytical angle for this tweet: {angle['instruction']}"""

        user_prompt = (
            f"Write a tweet about ${symbol}.\n\n"
            f"TA facts:\n{ta_block}\n\n"
            f"Write one tweet. Lowercase. Use the real numbers above. Under 240 chars."
        )

        tweet_text = None

        # Try Anthropic (Sonnet — best for natural voice)
        try:
            import anthropic as _ac
            _api_key = os.environ.get("AI_INTEGRATIONS_ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_API_KEY")
            _base_url = os.environ.get("AI_INTEGRATIONS_ANTHROPIC_BASE_URL")
            _aclient = _ac.Anthropic(base_url=_base_url, api_key=_api_key) if _base_url else _ac.Anthropic(api_key=_api_key)
            resp = await asyncio.to_thread(
                _aclient.messages.create,
                model="claude-sonnet-4-5",
                max_tokens=150,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
            )
            tweet_text = resp.content[0].text.strip().strip('"').strip("'")
        except Exception as ae:
            logger.warning(f"[TopGainerTA] Anthropic failed: {ae}")

        # Gemini fallback
        if not tweet_text:
            try:
                import google.generativeai as genai
                _gmodel = genai.GenerativeModel("gemini-2.0-flash")
                resp = await asyncio.to_thread(
                    _gmodel.generate_content,
                    system_prompt + "\n\n" + user_prompt
                )
                tweet_text = resp.text.strip().strip('"').strip("'")
            except Exception as ge:
                logger.warning(f"[TopGainerTA] Gemini failed: {ge}")

        # Hard fallback — raw but reads naturally
        if not tweet_text:
            if got_ta:
                tweet_text = f"${symbol} {sign}{change:.1f}% at {price_str}. rsi {rsi_val}, volume {vol_ratio}x avg. {trend} on the hourly"
            else:
                tweet_text = f"${symbol} {sign}{change:.1f}% — top gainer right now at {price_str}. vol {vol_str}"

        tweet_text = _sanitize_tickers(tweet_text, symbol)

        # Quick post-process: strip any accidental AI prefix like "tweet:" or "here's..."
        for _bad_prefix in ("tweet:", "here's a tweet:", "here is", "tweet\n"):
            if tweet_text.lower().startswith(_bad_prefix):
                tweet_text = tweet_text[len(_bad_prefix):].strip()

        tweet_text = await _ai_review_tweet(tweet_text, 'top_gainer_ta', {
            'symbol': symbol, 'change': f'{sign}{change:.1f}%',
            'angle': angle['id'], 'rsi': rsi_val, 'trend': trend,
        })

        result = account_poster.post_tweet(tweet_text)
        if result and result.get('success'):
            logger.info(f"✅ TopGainerTA [{angle['id']}] ${symbol} {sign}{change:.1f}% posted")

        return result

    except Exception as e:
        logger.error(f"Error posting top gainer TA: {e}")
        return {'success': False, 'error': str(e)}


async def post_free_telegram_promo(account_poster) -> Optional[Dict]:
    """
    Promote the free Telegram signals group using today's top gainer tickers.
    Posts like a KOL sharing what they're watching, then pointing to the free group.
    No images. Links to tradehubmarkets.com/start.
    """
    try:
        gainers = await _fetch_mexc_tickers()
        top = [g for g in gainers if g.get('volume', 0) >= 2_000_000][:8]

        if len(top) >= 3:
            picks = random.sample(top[:8], min(4, len(top)))
        elif top:
            picks = top
        else:
            picks = []

        tickers = " ".join(f"${g['symbol']}" for g in picks) if picks else "$BTC $ETH $SOL"

        tl = _pick_tweet_length()

        if tl == 'ultra_short':
            templates = [
                f"trading {tickers} today. signals are free in my Telegram → tradehubmarkets.com/start",
                f"on {tickers} right now. free signals in the group → tradehubmarkets.com/start",
                f"watching {tickers}. every trade I take goes to the free group → tradehubmarkets.com/start",
            ]
        elif tl == 'short':
            templates = [
                f"current watchlist: {tickers}\n\nevery signal I take gets posted in my free Telegram group — entry, TP, SL. nothing held back → tradehubmarkets.com/start",
                f"these are moving today: {tickers}\n\nI share the exact trades I'm taking in the free group. join before the next signal → tradehubmarkets.com/start",
                f"{tickers} on the radar right now\n\nif you want the actual trades with entries — free Telegram group → tradehubmarkets.com/start",
            ]
        elif tl == 'long':
            templates = [
                f"watchlist for today:\n{tickers}\n\nI post every trade I take in my free Telegram group — exact entry, TP, SL. no paid tier, no upsell. just the signals.\n\njoin free → tradehubmarkets.com/start",
                f"currently watching {tickers} for entries\n\nI run a free Telegram group where I share every signal I take myself — entry price, TP levels, stop loss. all live.\n\ntradehubmarkets.com/start",
            ]
        else:
            templates = [
                f"watching {tickers} today\n\nsharing every trade in the free Telegram group — entry, TP, SL posted live → tradehubmarkets.com/start",
                f"{tickers} are moving. sharing live trades in the free group → tradehubmarkets.com/start",
                f"on {tickers} right now. all my signals are free in Telegram — tradehubmarkets.com/start",
            ]

        tweet_text = random.choice(templates)
        result = account_poster.post_tweet(tweet_text)
        if result and result.get('success'):
            logger.info(f"Free Telegram promo posted with tickers: {tickers}")
        return result

    except Exception as e:
        logger.error(f"Error posting free telegram promo: {e}")
        return {'success': False, 'error': str(e)}


BITUNIX_CAMPAIGN_IMAGE = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                                       "attached_assets", "IMG_1209_1775372269740.jpeg")
BITUNIX_CAMPAIGN_LINK = "https://www.bitunix.com/activity/basic/1774508484?vipCode=fgq74890"
BITUNIX_CAMPAIGN_START = datetime(2026, 3, 27)
BITUNIX_CAMPAIGN_END = datetime(2026, 4, 26, 23, 59, 59)

CAMPAIGN_TEMPLATES = [
    {
        'id': 'low_barrier_entry',
        'text': """$100 deposit on Bitunix = $100 USDT position voucher free

100 slots. First come first served.

Bitunix x TradeHub Markets running until April 26
Trading {ticker1} {ticker2} anyway — might as well claim it

{link}"""
    },
    {
        'id': 'top_tier_math',
        'text': """Deposit $2,000 on Bitunix = $1,000 USDT in position vouchers

That's a 50% bonus just for depositing and holding 9 days

Only 20 slots at the top tier. No restock.

Bitunix x TradeHub Markets — ends April 26

{link}"""
    },
    {
        'id': 'volume_rewards_breakdown',
        'text': """If you're already running volume on {ticker1} {ticker2} {ticker3} perps you're leaving free money on the table

$10K vol = 200 USDT BTC voucher
$50K vol = 300 USDT BTC voucher
$100K vol = 15 USDT futures bonus

Bitunix x TradeHub Markets. Ends April 26.

{link}"""
    },
    {
        'id': 'stack_both',
        'text': """Deposit bonus + volume rewards on Bitunix stack independently

Put in $1,000 = 2×250 USDT vouchers
Run $50K volume = another 300 USDT BTC voucher

You're already trading {ticker1}. Both rewards apply at once.

Bitunix x TradeHub Markets

{link}"""
    },
    {
        'id': 'scarcity_slots',
        'text': """Slot count for the Bitunix x TradeHub campaign:

$100 tier = 100 slots
$500 tier = 50 slots
$1,000 tier = 30 slots
$2,000 tier = 20 slots

Slots are the constraint here, not the end date.
{ticker1} {ticker2} traders already moving in.

{link}"""
    },
    {
        'id': 'fomo_ticker_hook',
        'text': """{ticker1} up {pct1}% and {ticker2} up {pct2}% today

Traders catching those moves on Bitunix are also collecting deposit bonuses on top

Up to $1,000 USDT in vouchers for new deposits
Volume rewards on top of that

Bitunix x TradeHub Markets — April 26

{link}"""
    },
    {
        'id': 'perps_trader_angle',
        'text': """You're already longing {ticker1} and {ticker2} perps somewhere

If it's Bitunix, you qualify for up to $1,000 USDT in deposit vouchers this month

New user deposit campaign live now — 27 March to 26 April

Bitunix x TradeHub Markets

{link}"""
    },
    {
        'id': 'low_barrier_v2',
        'text': """Entry level on this campaign is $100

Hold it 3 days. Get a 100 USDT BTC position voucher back.

There are 100 slots at that tier. Not "limited" as marketing. Literally 100.

Bitunix x TradeHub Markets
{ticker1} {ticker2} both tradeable on there.

{link}"""
    },
    {
        'id': 'btc_voucher_angle',
        'text': """The volume rewards on this Bitunix campaign pay out in BTC position vouchers

$10K in volume = 200 USDT BTC voucher
$50K in volume = 300 USDT BTC voucher

Meaning you get leveraged BTC exposure for free just by trading {ticker1} {ticker2} {ticker3}

Ends April 26. Bitunix x TradeHub Markets

{link}"""
    },
    {
        'id': 'regret_frame',
        'text': """The 20 slots at the $2,000 deposit tier go first every campaign

$1,000 USDT in position vouchers for one deposit and a 9 day hold

When those slots go they're gone. No waitlist. No second round.

Bitunix x TradeHub Markets — runs until April 26

{link}"""
    },
    {
        'id': 'night_grind',
        'text': """grinding {ticker1} positions at 3am anyway

might as well be collecting deposit bonuses and volume rewards at the same time

Bitunix x TradeHub Markets. $100 minimum. 100 slots at that tier. ends April 26

{link}"""
    },
    {
        'id': 'comparison_exchange',
        'text': """Other exchanges take fees to trade {ticker1}

Bitunix is paying new users to trade it

$100 minimum deposit. Volume rewards on top. Campaign runs through April 26.

Bitunix x TradeHub Markets — limited slots across all tiers

{link}"""
    },
    {
        'id': 'honest_observation',
        'text': """I usually ignore exchange campaigns

This one is different because the volume rewards apply to trades you'd be doing anyway

$10K vol on {ticker1} {ticker2} perps = 200 USDT BTC voucher. No extra steps.

Bitunix x TradeHub Markets — ends April 26

{link}"""
    },
    {
        'id': 'whale_tier',
        'text': """For the bigger accounts running volume on Bitunix:

$500K volume = 30 USDT futures bonus
$1M volume = 60 USDT futures bonus
$5M volume = 300 USDT futures bonus

Plus deposit rewards on top if you haven't claimed those

Bitunix x TradeHub Markets. {ticker1} {ticker2} {ticker3} all there.

{link}"""
    },
    {
        'id': 'simple_cta',
        'text': """Bitunix x TradeHub Markets campaign is live

Deposit rewards up to $1,000 USDT in vouchers
Volume rewards up to $300 USDT BTC position

{ticker1} {ticker2} {ticker3} all available on there
Runs until April 26. Slots fill by tier.

{link}"""
    },
]

_campaign_post_index = 0
_campaign_posted_today = set()


YUBIT_CAMPAIGN_IMAGE = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                                    "attached_assets", "IMG_1461_1775736677070.jpeg")
YUBIT_CAMPAIGN_LINK  = "https://www.yubit.com/en-US/rewards-hub?inviteCode=TZQL"
YUBIT_CAMPAIGN_START = datetime(2026, 4, 9)
YUBIT_CAMPAIGN_END   = datetime(2026, 4, 30, 23, 59, 59)

# ── BYDFi x TradeHub $2000 deposit bonus campaign ─────────────────────────────
# Period: April 22 – April 30, 2026
BYDFI_CAMPAIGN_IMAGE = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                                    "attached_assets", "IMG_1885_1776865573041.jpeg")
BYDFI_CAMPAIGN_LINK  = "https://www.bydfi.com/en/activities/view?id=1253770817526616065&ru=tradehub&p=L2VuL2FjdGl2aXRpZXMvdmlldw%3D%3D"
BYDFI_CAMPAIGN_START = datetime(2026, 4, 22)
BYDFI_CAMPAIGN_END   = datetime(2026, 4, 30, 23, 59, 59)

YUBIT_CAMPAIGN_TEMPLATES = [
    {
        'id': 'double_deal',
        'text': """🚨 two things happening right now — {ticker1} {ticker2} moving and Yubit paying people to trade them

① 20,000 USDT campaign — ends april 30
   $100 deposit + 30k volume → 50 USDT back
   $2,000 deposit + 900k volume → 200 USDT back

② sign up via my link + DM me on telegram (link in bio)
   → FREE access to my live trading signals

one link. two rewards. 100 slots left.

{link}"""
    },
    {
        'id': 'fomo_ticker',
        'text': """{ticker1} +{pct1}% ↑  {ticker2} +{pct2}% ↑

Yubit is paying traders to trade moves like this — 20,000 USDT campaign, ends april 30

💰 $100 deposit + 30k vol → 50 USDT
💰 $2,000 deposit + 900k vol → 200 USDT

📲 sign up via my link + DM me on telegram (link in bio)
   → i'll get you into my free signals group on top

limited slots. don't sleep on it.

{link}"""
    },
    {
        'id': 'stack_both',
        'text': """here's how to stack two things at once while {ticker1} and {ticker2} are running

🔹 Yubit rewards (ends april 30)
   • $100 in + 30k vol = 50 USDT reward
   • $2,000 in + 900k vol = 200 USDT reward
   • 100 entry slots left

🔹 free signals from me
   • sign up via my link
   • DM me on telegram (link in bio) after
   • i'll add you to my live signals — free

{link}"""
    },
    {
        'id': 'straight_offer',
        'text': """making this simple — {ticker1} {ticker2} {ticker3} all on Yubit

sign up via my link
→ trade 30k volume before april 30
→ earn 50 USDT from their campaign

DM me on telegram (link in bio) after
→ free access to my trading signals

USDT in your pocket + my signals. both free to claim.

{link}"""
    },
    {
        'id': 'countdown',
        'text': """⏰ april 30 — two things close

❶ Yubit's 20,000 USDT campaign
   $100 + 30k vol → 50 USDT
   $1,000 + 300k vol → 120 USDT
   $2,000 + 900k vol → 200 USDT

❷ my free signals for new Yubit signups
   sign up via my link + DM me on telegram (link in bio) and i'll add you

{ticker1} {ticker2} are moving — don't leave both on the table.

{link}"""
    },
    {
        'id': 'value_stack',
        'text': """🔥 if you're trading {ticker1} {ticker2} perps this month this is for you

Yubit campaign (ends april 30):
deposit + trade → earn up to 200 USDT cash
20,000 USDT pool. entry starts at $100.

PLUS — sign up through my link + DM me on telegram (link in bio):
i'll get you free access to my live trading signals

rewards AND signals. both included.

{link}"""
    },
    {
        'id': 'perps_trader',
        'text': """trading {ticker1} and {ticker2} perps already?

then this is literally free money — move that volume to Yubit

📈 30k vol = 50 USDT reward
📈 900k vol = 200 USDT reward
📲 sign up via my link + DM me on telegram (link in bio) = free signals on top

one move. three wins. april 30 cutoff, 100 slots left.

{link}"""
    },
    {
        'id': 'urgency_close',
        'text': """last call — {ticker1} {ticker2} pumping and april 30 is the hard deadline

✅ Yubit campaign — 20,000 USDT pool
   $100 deposit + 30k vol → 50 USDT
   $2,000 deposit + 900k vol → 200 USDT

✅ free signals — DM me on telegram (link in bio) after signing up via my link

both end april 30. slots filling.

claim it → {link}"""
    },
]

_yubit_post_index = 0

# ── BYDFi campaign templates — written human/casual, lowercase, light emoji ──
# Style ref: low-key crypto trader voice, mentions a few real moving tickers as
# context, casual filler ("rn", "btw", "tbh"), no em dashes, no marketing-speak.
BYDFI_CAMPAIGN_TEMPLATES = [
    {
        'id': 'casual_intro',
        'text': """💊 $2000 deposit bonus live with BYDFi x TradeHub 💊

been seeing more people trading coins like {ticker1} {ticker2} {ticker3} lately and using these bonuses as extra margin to catch the moves 👀📈

offers right now:
$100 deposit = $50 bonus
$500 deposit = $100 bonus
$1000 deposit = $200 bonus

whale side:
$5000 = $500
$10,000 = $1000
$20,000 = $2000 🐋🔥

a few in the group already redeeming and putting it to work while low caps are active

link if you want in:
{link}"""
    },
    {
        'id': 'momentum_tie',
        'text': """market's moving again and BYDFi just dropped a $2000 deposit bonus campaign with us

basically free margin if you were planning to trade {ticker1} {ticker2} this week anyway

how it stacks:
• $100 in → $50 back
• $500 in → $100 back
• $1000 in → $200 back
• $5k → $500 / $10k → $1000 / $20k → $2000

ends april 30. first deposits only. 72hr hold.

{link}"""
    },
    {
        'id': 'low_cap_play',
        'text': """if you're chasing low caps like {ticker1} {ticker2} {ticker3} the BYDFi bonus is honestly the move rn

extra margin, no strings beyond a 3 day hold

$100 → $50 bonus
$500 → $100 bonus
$1000 → $200 bonus
whale tier maxes at $2000 on a $20k deposit

period: april 22 to april 30 only

{link}"""
    },
    {
        'id': 'simple_breakdown',
        'text': """quick one — BYDFi x TradeHub bonus is live til april 30

deposit → bonus paid in USDT-M futures
$100 = $50
$500 = $100
$1000 = $200
$5000 = $500
$10k = $1000
$20k = $2000

{ticker1} {ticker2} are the ones people are putting it into atm

first deposits only. 72hr hold to qualify

{link}"""
    },
    {
        'id': 'group_proof',
        'text': """few people in the group already redeemed the BYDFi bonus today and used it to size up on {ticker1} and {ticker2}

if you missed it:
$100 deposit → $50 free margin
$1000 → $200
$10k → $1000 (whale tier)
$20k → $2000 max payout

ends april 30, first deposits only

{link}"""
    },
    {
        'id': 'short_hook',
        'text': """$2000 in bonus money on BYDFi if you size up

$100 → $50
$1000 → $200
$20k → $2000

low caps like {ticker1} {ticker2} running rn so the extra margin actually means something this week

deadline april 30

{link}"""
    },
    {
        'id': 'casual_question',
        'text': """anyone else stacking the BYDFi bonus this week?

$100 deposit gets $50 back, $1000 gets $200, $20k whale tier maxes at $2000

planning to put mine into {ticker1} or {ticker2} depending on which one breaks first

ends april 30 btw, first deposits only

{link}"""
    },
    {
        'id': 'final_call',
        'text': """⏰ BYDFi x TradeHub bonus closes april 30

if you trade {ticker1} {ticker2} or any of the low caps moving rn this is basically free margin

$100 → $50
$500 → $100
$1000 → $200
$5000 → $500
$10,000 → $1000
$20,000 → $2000 🐋

USDT-M futures only, 72hr hold, first deposits

{link}"""
    },
]

_bydfi_post_index = 0


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


MEME_COINS = {
    'DOGE', 'SHIB', 'PEPE', 'FLOKI', 'BONK', 'WIF', 'MEME', 'TURBO',
    'NEIRO', 'BOME', 'BRETT', 'MOG', 'POPCAT', 'BABYDOGE', 'SPX', 'GROK',
    'TRUMP', 'WOJAK', 'LADYS', 'MONG', 'BOB', 'TOSHI',
}

HIGH_VIEWING_COINS = MEME_COINS | {
    'BTC', 'ETH', 'SOL', 'BNB', 'XRP', 'ADA', 'AVAX', 'DOT', 'LINK',
    'UNI', 'ATOM', 'LTC', 'APT', 'ARB', 'OP', 'SUI', 'SEI',
    'NEAR', 'INJ', 'TIA', 'FET', 'RENDER', 'WLD', 'JUP',
    'AAVE', 'MKR', 'STX', 'IMX', 'GALA', 'SAND',
    'FTM', 'RUNE', 'TAO', 'ONDO', 'ENA', 'PENDLE', 'DYDX',
    'ORDI', 'BLUR', 'NOT', 'TON', 'PI', 'LAYER', 'MATIC',
    'ICP', 'HBAR', 'TRX', 'ETC', 'BCH',
}

# Coins confirmed/highly likely to be listed on Yubit USDT perpetuals.
# Used to filter top-gainer picks so campaign posts only reference
# coins traders can actually open on Yubit.
YUBIT_LISTED_SYMBOLS = {
    # Layer 1 majors
    'BTC', 'ETH', 'SOL', 'BNB', 'XRP', 'ADA', 'AVAX', 'DOT', 'TRX',
    'LTC', 'BCH', 'ETC', 'ATOM', 'NEAR', 'ICP', 'HBAR', 'FIL', 'VET',
    'ALGO', 'XLM', 'EGLD', 'FLOW', 'THETA', 'XTZ', 'EOS', 'ZIL',
    # Layer 2 / Ethereum ecosystem
    'ARB', 'OP', 'MATIC', 'IMX', 'LRC', 'METIS', 'BOBA', 'ZKJ',
    # DeFi blue chips
    'UNI', 'AAVE', 'MKR', 'CRV', 'SNX', 'COMP', 'SUSHI', 'BAL',
    'YFI', '1INCH', 'RUNE', 'CAKE', 'DYDX', 'GMX', 'PENDLE', 'ENA',
    # AI / infra
    'FET', 'RENDER', 'WLD', 'TAO', 'OCEAN', 'GRT', 'LINK', 'API3',
    'BAND', 'ONDO', 'IO',
    # Move / new L1
    'SUI', 'APT', 'SEI', 'INJ', 'TIA', 'LAYER', 'STRK',
    # Popular alts
    'STX', 'GALA', 'SAND', 'MANA', 'AXS', 'ENJ', 'CHZ', 'AUDIO',
    'BLUR', 'IMX', 'MAGIC', 'LDO', 'RPL', 'SSV', 'ETHFI',
    'JUP', 'JTO', 'WEN', 'RAY', 'PYTH', 'DRIFT', 'W',
    'ORDI', 'SATS', 'RATS', 'MEME', 'NOT', 'TON',
    # Meme coins commonly listed on perp DEXes/CEXes
    'DOGE', 'SHIB', 'PEPE', 'FLOKI', 'BONK', 'WIF', 'BOME', 'POPCAT',
    'TURBO', 'BRETT', 'NEIRO', 'MOG', 'DOGS',
    # Other high-volume alts
    'FTM', 'KAVA', 'ZEC', 'DASH', 'IOTA', 'NEO', 'ONT', 'QTUM',
    'ICX', 'ZRX', 'BAT', 'KNC', 'RSR', 'ANKR', 'SKL', 'NMR',
    'CELR', 'CELO', 'ACH', 'CTSI', 'ORN', 'LQTY', 'SPELL', 'RNDR',
}


async def get_live_tickers_for_campaign() -> Dict:
    """Fetch actual top gainers of the day for campaign posts.
    Picks real movers by % change, filtered to coins listed on Bitunix
    (used as a live proxy for Yubit's near-identical futures listings).
    Minimum $2M 24h volume to exclude illiquid/scam tokens.
    """
    fallback = {
        'ticker1': '$BTC', 'ticker2': '$ETH', 'ticker3': '$SOL',
        'pct1': '3.2', 'pct2': '2.8',
    }

    # Symbols to skip — stablecoins, wrapped tokens, leveraged tokens, garbage
    _SKIP = {
        'USDT', 'USDC', 'BUSD', 'DAI', 'TUSD', 'USDP', 'FDUSD', 'USDD',
        'WBTC', 'WETH', 'WBNB', 'STETH', 'RETH', 'CBETH',
        'UP', 'DOWN', 'BULL', 'BEAR', '2L', '2S', '3L', '3S',
    }
    MIN_VOLUME_USD = 2_000_000  # $2M minimum 24h volume

    import re as _re

    def pick_actual_gainers(raw: list, listed: set) -> list:
        """Sort by % gain, filter by volume + exchange listing, skip garbage."""
        candidates = []
        for g in raw:
            sym = g.get('symbol', '').upper()
            change = g.get('change', 0)
            vol = g.get('volume', 0)
            if change <= 0:
                continue
            if vol < MIN_VOLUME_USD:
                continue
            if sym in _SKIP:
                continue
            if _re.search(r'\d[LS]$|^[LS]\d', sym):
                continue
            if len(sym) > 12:
                continue
            # Only include coins confirmed listed on the exchange
            if listed and sym not in listed:
                continue
            candidates.append({'symbol': f'${sym}', 'pct': round(change, 1), 'volume': vol})
        candidates.sort(key=lambda x: x['pct'], reverse=True)
        return candidates[:5]

    try:
        import httpx
        async with httpx.AsyncClient() as client:
            # Step 1: fetch Bitunix symbol list as live proxy for Yubit listings
            listed_syms: set = set()
            try:
                br = await client.get(
                    "https://fapi.bitunix.com/api/v1/futures/market/tickers", timeout=6
                )
                if br.status_code == 200:
                    bdata = br.json().get('data') or []
                    for t in bdata:
                        s = t.get('symbol', '')
                        if s.endswith('USDT'):
                            listed_syms.add(s.replace('USDT', ''))
                    if listed_syms:
                        logger.debug(f"Yubit proxy: {len(listed_syms)} Bitunix symbols loaded")
            except Exception as be:
                logger.warning(f"Bitunix symbol fetch for campaign tickers failed: {be}")
            # If Bitunix fetch failed, fall back to static list
            if not listed_syms:
                listed_syms = YUBIT_LISTED_SYMBOLS

            # Step 2: fetch MEXC top gainers and filter against confirmed listings
            resp = await client.get("https://contract.mexc.com/api/v1/contract/ticker", timeout=8)
            if resp.status_code == 200:
                data = resp.json().get('data', [])
                raw = []
                for t in data:
                    sym = t.get('symbol', '')
                    if not sym.endswith('_USDT'):
                        continue
                    change = float(t.get('riseFallRate', 0) or 0) * 100
                    vol = float(t.get('amount24', 0) or 0)
                    raw.append({'symbol': sym.replace('_USDT', ''), 'change': change, 'volume': vol})
                picks = pick_actual_gainers(raw, listed_syms)
                if len(picks) >= 2:
                    return {
                        'ticker1': picks[0]['symbol'],
                        'ticker2': picks[1]['symbol'],
                        'ticker3': picks[2]['symbol'] if len(picks) > 2 else picks[0]['symbol'],
                        'pct1': str(picks[0]['pct']),
                        'pct2': str(picks[1]['pct']),
                    }
    except Exception as e:
        logger.error(f"Campaign tickers fetch failed: {e}")

    return fallback


def generate_tradehub_card_image(strategies: List[Dict]) -> Optional[bytes]:
    """Generate a leaderboard card that looks exactly like the TradeHub website.

    Primary: cairosvg SVG card (no browser required).
    Fallback: PIL-drawn promo card.
    """
    # ── Primary: cairosvg SVG card ───────────────────────────────────────────
    try:
        from app.services.screenshot_card import screenshot_leaderboard_card_sync
        png = screenshot_leaderboard_card_sync(strategies)
        if png:
            logger.info(f"[generate_tradehub_card_image] cairosvg SVG card OK — {len(png):,} bytes")
            return png
    except Exception as e:
        logger.warning(f"[generate_tradehub_card_image] cairosvg card failed: {e}")

    # ── Fallback: PIL card ────────────────────────────────────────────────────
    try:
        from app.services.tweet_card_generator import make_promo_card
        # Build stats from top strategies
        stats = []
        if strategies:
            top = strategies[0]
            pnl_sign = "+" if top.get("total_pnl", 0) >= 0 else ""
            stats = [
                ("Top P&L", f"{pnl_sign}{top.get('total_pnl', 0):.1f}%"),
                ("Win Rate", f"{top.get('win_rate', 0):.0f}%"),
                ("Active Strats", str(len(strategies))),
            ]
        return make_promo_card(
            feature_headline="TradeHub Strategy Leaderboard",
            sub="Automated strategies · Live P&L · 80% revenue share",
            stats=stats,
            cta="Build & automate your strategy free — tradehubmarkets.com",
        )
    except Exception as e:
        logger.error(f"Failed to generate TradeHub card image (PIL fallback): {e}")
        return None


async def _fetch_leaderboard_strategies(limit: int = 3) -> List[Dict]:
    """Fetch top strategies from DB for the promo card."""
    try:
        from app.database import SessionLocal
        from app.strategy_models import StrategyPerformance, UserStrategy, StrategyExecution
        import json as _json

        db = SessionLocal()
        try:
            rows = (
                db.query(StrategyPerformance, UserStrategy)
                .join(UserStrategy, UserStrategy.id == StrategyPerformance.strategy_id)
                .filter(StrategyPerformance.total_trades >= 3)
                .order_by(StrategyPerformance.total_pnl_pct.desc())
                .limit(limit)
                .all()
            )

            KNOWN_TICKERS = [
                "BTC","ETH","SOL","BNB","XRP","ADA","DOGE","AVAX","DOT","MATIC",
                "LINK","UNI","ATOM","LTC","BCH","NEAR","APT","ARB","OP","SUI",
                "PEPE","WIF","BONK","FLOKI","SHIB","INJ","TIA","SEI","JUP","PYTH",
                "FTM","ALGO","EGLD","FIL","ICP","SAND","MANA","APE","LDO","CRV",
                "RUNE","STX","CFX","GMX","DYDX","AAVE","SNX","COMP","MKR","YFI",
            ]

            result = []
            for perf, strat in rows:
                name = strat.name or "Unnamed"
                name_upper = name.upper()
                found = [t for t in KNOWN_TICKERS if t in name_upper]
                if not found:
                    found = ["CRYPTO"]

                # Pull direction / leverage / tp / sl from config JSON
                direction, leverage, tp_pct, sl_pct = "", "", "", ""
                try:
                    cfg = _json.loads(strat.config_json or "{}")
                    direction = cfg.get("direction", "")
                    leverage  = cfg.get("leverage", "")
                    risk      = cfg.get("risk_management", {})
                    tp_pct    = risk.get("tp_pct", cfg.get("tp_pct", ""))
                    sl_pct    = risk.get("sl_pct", cfg.get("sl_pct", ""))
                except Exception:
                    pass

                # Recent WIN/LOSS tags — last 5 closed executions
                recent_tags = []
                try:
                    recent_execs = (
                        db.query(StrategyExecution)
                        .filter(
                            StrategyExecution.strategy_id == strat.id,
                            StrategyExecution.outcome.in_(["WIN", "LOSS"]),
                            StrategyExecution.pnl_pct.isnot(None),
                        )
                        .order_by(StrategyExecution.closed_at.desc())
                        .limit(5)
                        .all()
                    )
                    for ex in recent_execs:
                        ticker = ex.symbol.replace("USDT", "") if ex.symbol else "?"
                        pnl    = ex.pnl_pct or 0
                        sign   = "+" if pnl >= 0 else ""
                        recent_tags.append(f"{ticker} {sign}{pnl:.0f}%")
                except Exception:
                    pass

                result.append({
                    "name":         name,
                    "tickers":      found[:3],
                    "total_trades": perf.total_trades,
                    "win_rate":     round(perf.win_rate, 1),
                    "total_pnl":    round(perf.total_pnl_pct, 2),
                    "direction":    direction,
                    "leverage":     leverage,
                    "tp_pct":       tp_pct,
                    "sl_pct":       sl_pct,
                    "recent_tags":  recent_tags,
                })
            return result
        finally:
            db.close()
    except Exception as e:
        logger.error(f"Error fetching leaderboard strategies: {e}")
        return []


async def post_market_take(account_poster) -> Optional[Dict]:
    """
    Post a pure KOL hot take about the current market — no specific coin required.
    Covers macro observations, trading psychology, market structure, cycle sentiment.
    These are the posts KOLs are known for: opinionated, conversational, screenshot-worthy.
    """
    try:
        # Fetch BTC price and 24h change for context
        btc_price = None
        btc_change = None
        dominance = None
        try:
            import httpx as _httpx
            async with _httpx.AsyncClient(timeout=8.0) as _cl:
                r = await _cl.get(
                    "https://api.coingecko.com/api/v3/coins/markets",
                    params={"vs_currency": "usd", "ids": "bitcoin", "sparkline": False}
                )
                if r.status_code == 200:
                    data = r.json()
                    if data:
                        btc_price  = data[0].get("current_price", 0)
                        btc_change = data[0].get("price_change_percentage_24h", 0)
        except Exception:
            pass

        # Pick a topic angle
        topic_angles = [
            "market_cycle_take",    # where are we in the cycle
            "trading_psychology",   # a real lesson about trader behavior
            "risk_reality",         # hard truth about risk / sizing
            "crypto_vs_stock",      # contrast with tradfi
            "narrative_shift",      # which narratives are changing
            "discipline_take",      # something about rules vs feelings
            "market_structure",     # observation about current price action
            "patience_lesson",      # waiting for the setup
            "contrarian_view",      # against the crowd take
            "late_night_thought",   # philosophical / reflective
        ]
        angle = random.choice(topic_angles)

        btc_ctx = ""
        if btc_price and btc_change is not None:
            sign = "+" if btc_change >= 0 else ""
            btc_ctx = f"BTC is at ${btc_price:,.0f}, {sign}{btc_change:.1f}% in the last 24h. "

        time_ctx = get_time_context()
        day_ctx  = get_day_context()

        _mt_gainers = get_daily_gainers_str(max_tickers=4)
        _mt_gainers_ctx = f"Top movers today: {_mt_gainers}. " if _mt_gainers else ""

        # Fetch today's trending data — coins AND topics getting engagement on X
        _daily_topics  = get_todays_trending_topics(5)   # from daily discovery (4h refresh)
        _daily_coins   = get_todays_trending_coins(8)    # from daily discovery
        # Supplement with real-time X search (30-min cache)
        _rt_trending   = await _fetch_x_trending_crypto()

        _trending_ctx = ""
        if _daily_topics or _daily_coins or _rt_trending:
            parts = []
            if _daily_topics:
                parts.append(
                    "TODAY'S MOST-DISCUSSED TOPICS ON X: "
                    + ", ".join(f"{t['topic']} ({t['avg_likes']:.0f} avg likes, {t['mentions']} posts)"
                                for t in _daily_topics[:4])
                )
            if _daily_coins:
                parts.append(
                    "COINS PEOPLE ARE TALKING ABOUT TODAY: "
                    + ", ".join(f"${c['symbol']} ({c['avg_likes']:.0f} avg likes)"
                                for c in _daily_coins[:5])
                )
            if _rt_trending and not _daily_topics:
                parts.append(
                    "RIGHT NOW ON X: "
                    + ", ".join(f"{t['topic']} ({t['avg_likes']:.0f} avg likes)"
                                for t in _rt_trending[:4])
                )
            _trending_ctx = (
                "\n\n".join(parts)
                + "\n\nWrite your take on something FROM THIS LIST — "
                  "these are the conversations already happening on X. "
                  "Your angle should be different from the obvious consensus take."
            )

        _engagement_insights = await asyncio.to_thread(_get_engagement_insights)

        angle_prompts = {
            "market_cycle_take": (
                f"{btc_ctx}{_mt_gainers_ctx}Write a tweet with your current read on where we are in the market cycle. "
                f"Not a prediction — an observation. Something you've noticed in the data, the sentiment, "
                f"or the behavior of other market participants. Time: {time_ctx['period']} on a {day_ctx['day']}."
            ),
            "trading_psychology": (
                f"{btc_ctx}Write a tweet about a specific pattern in trader psychology you've observed — "
                f"how people behave near tops, near bottoms, or during sideways chop. "
                f"Something real, not a motivational quote. Make it specific and a little uncomfortable."
            ),
            "risk_reality": (
                f"{btc_ctx}Write a tweet with a hard, specific truth about risk management that most people "
                f"either don't know or refuse to apply. Not generic advice. An actual position on something."
            ),
            "crypto_vs_stock": (
                f"{btc_ctx}Write a tweet contrasting something about crypto markets vs traditional finance. "
                f"Something you've genuinely noticed — in volatility, behavior, timing, or sentiment. "
                f"Not preachy. Just an observation that makes people think."
            ),
            "narrative_shift": (
                f"{btc_ctx}{_mt_gainers_ctx}Write a tweet about a narrative in crypto that is either gaining or losing momentum right now. "
                f"What changed? Who's wrong? What does the market actually seem to care about vs what CT says it cares about?"
            ),
            "discipline_take": (
                f"{btc_ctx}Write a tweet about trading discipline — what it actually means in practice, "
                f"not what it sounds like in productivity content. A real moment or rule you've come to rely on."
            ),
            "market_structure": (
                f"{btc_ctx}Write a tweet about something you're seeing in the current market structure. "
                f"Could be liquidity, ranging, compression, or a breakout setup in the macro. "
                f"Technical but conversational — like texting someone who trades."
            ),
            "patience_lesson": (
                f"{btc_ctx}Write a tweet about the cost of impatience in trading. "
                f"Not a lesson — a specific observation about what happens when you don't wait for the setup. "
                f"Make it personal or relatable. Time: {time_ctx['period']}."
            ),
            "contrarian_view": (
                f"{btc_ctx}Write a tweet with a view that goes against the current consensus in crypto Twitter. "
                f"Not contrarian for the sake of it — something you actually believe that most people disagree with. "
                f"Back it up in one sentence."
            ),
            "late_night_thought": (
                f"{btc_ctx}It's {time_ctx['period']}. Write a reflective, honest tweet about trading — "
                f"what it actually takes, what you got wrong, what surprised you. "
                f"The kind of thought that comes when the market is quiet and you have time to think. "
                f"Not inspirational. Just real."
            ),
        }

        market_take_system = (
            "You are a confident, slightly arrogant crypto trader with a strong track record. "
            "You post on X/Twitter because you have something to say, not because you're trying to build an audience. "
            "You are not a coach, not a signals provider, not a news account. "
            "You've seen multiple bull and bear cycles. You know when others are wrong and you say so — "
            "not rudely, but with the quiet certainty of someone who has been right before. "
            "You sometimes brag, but only about things you can back up. "
            "You are direct, occasionally dismissive of the crowd, and always specific. "
            "Your posts are lowercase, no hashtags, no emojis, no engagement bait. "
            "You write the way you'd text a fellow trader — fast, honest, no fluff. "
            "A great post: one specific take, delivered with authority, ends when the point is made."
        )

        _trending_block = f"\n\n{_trending_ctx}" if _trending_ctx else ""
        _insights_block = f"\n\n{_engagement_insights}" if _engagement_insights else ""

        ai_prompt = f"""{angle_prompts[angle]}{_trending_block}{_insights_block}

HARD RULES:
- No hashtags, no emojis, no exclamation marks
- No "NFA", "DYOR", "LFG", "gm", "ngmi"
- No engagement bait ("what do you think?", "agree?", "follow for more")
- No generic motivational language ("stay disciplined", "trust the process")
- Lowercase except $BTC and other coin tickers
- Max 240 characters — stop when the thought is complete
- One clear opinion, not a list of observations
- Sound like a person with a point of view, not a content creator

WHAT SEPARATES GREAT FROM MEDIOCRE:
- Great: says something specific and slightly uncomfortable
- Mediocre: says something vague that could apply to any market at any time
- Great: the reader feels seen or challenged
- Mediocre: the reader nods and scrolls

Write ONLY the tweet text."""

        tweet_text = await _call_grok_tweet(
            ai_prompt, max_chars=240,
            label=f"market_take/{angle}",
            system=market_take_system,
        )

        # Fallback takes — confident, arrogant, specific
        if not tweet_text:
            fallbacks = [
                f"{btc_ctx or ''}I've been early on more moves than most accounts posting charts right now. the difference isn't tools. it's patience and position sizing.",
                f"the most expensive trade isn't the one you lost. it's the one you sized too big on and had to close too early because you couldn't breathe.",
                f"crypto twitter has a consensus opinion on every coin at any given time. the consensus is wrong at inflection points. every time. I trade against it.",
                f"I don't have 'research'. I have a process. I run it every day whether or not the market is moving. that's the edge.",
                f"the market doesn't care about your cost basis. this is obvious and yet it shapes almost every bad decision people make. I cut early and I sleep fine.",
                f"most people enter positions based on conviction and exit based on emotion. I enter based on process and exit based on plan. the difference is the P&L.",
                f"{btc_ctx or ''}the alts that lead the next leg almost never come from the names everyone is already talking about. I'm not looking where you're looking.",
                f"patience in a ranging market is underrated. I've made more money waiting than trading. most people can't sit still long enough to find out.",
                f"I've been wrong before. the difference is I was wrong in small size and right in large size. that's not luck. that's risk management.",
                f"people ask me how I stay calm when the market drops 15% in a day. position sizing. always position sizing. nothing else matters as much.",
            ]
            tweet_text = random.choice(fallbacks)

        # Append top-gainer tickers for discoverability
        tickers = get_daily_gainers_str(max_tickers=3)
        if tickers and len(tweet_text) + len(tickers) + 2 <= 278:
            tweet_text = tweet_text + "\n\n" + tickers

        _yd = _maybe_yubit_drop()
        if _yd and (len(tweet_text + _yd) - 32) <= 280:
            tweet_text = tweet_text + _yd

        result = account_poster.post_tweet(tweet_text)

        if result and result.get('success'):
            logger.info(f"✅ Market take [{angle}] posted: {tweet_text[:60]}...")
        else:
            logger.warning(f"Market take post failed: {result}")

        return result

    except Exception as e:
        logger.error(f"Error posting market take: {e}")
        return {'success': False, 'error': str(e)}


async def post_tradehub_promo(account_poster) -> Optional[Dict]:
    """
    Educational market/Bitcoin post with top gainer tickers.
    Sharp market observations, BTC reads, and trading insights — no product promotion.
    """
    try:
        # Fetch BTC context
        btc_price = None
        btc_change = None
        try:
            import httpx as _httpx
            async with _httpx.AsyncClient(timeout=8.0) as _cl:
                r = await _cl.get(
                    "https://api.coingecko.com/api/v3/coins/markets",
                    params={"vs_currency": "usd", "ids": "bitcoin", "sparkline": False}
                )
                if r.status_code == 200:
                    data = r.json()
                    if data:
                        btc_price  = data[0].get("current_price", 0)
                        btc_change = data[0].get("price_change_percentage_24h", 0)
        except Exception:
            pass

        # Fresh top gainers for context + appending
        if not _DAILY_GAINERS_TICKERS:
            try:
                _fresh = await _fetch_mexc_tickers()
                _update_daily_gainers(_fresh)
            except Exception:
                pass
        ticker_str = get_daily_gainers_str(max_tickers=4)

        btc_ctx = ""
        if btc_price and btc_change is not None:
            _sign = "+" if btc_change >= 0 else ""
            btc_ctx = f"BTC is at ${btc_price:,.0f}, {_sign}{btc_change:.1f}% in 24h. "

        angles = [
            "btc_level",
            "market_structure",
            "alt_cycle",
            "volume_read",
            "dominance",
            "risk_lesson",
            "timing_take",
            "macro_watch",
        ]
        angle = random.choice(angles)
        time_ctx = get_time_context()

        angle_prompts = {
            "btc_level": (
                f"{btc_ctx}Write a tweet about what BTC's current level means right now. "
                f"Is it at support, resistance, or in no-man's-land? What should a trader be watching? "
                f"Concrete — not 'BTC could go up or down'."
            ),
            "market_structure": (
                f"{btc_ctx}Write a tweet about something specific in crypto market structure right now. "
                f"Could be consolidation, a squeeze, a range, or a breakout setup. Technical but conversational."
            ),
            "alt_cycle": (
                f"{btc_ctx}Write a tweet about where we are in the alt cycle relative to BTC. "
                f"Top movers today: {ticker_str or 'mixed'}. What does current BTC dominance mean for alts?"
            ),
            "volume_read": (
                f"{btc_ctx}Write a tweet about what volume is telling you right now. "
                f"A specific observation about current price action and participation, not a generic lesson."
            ),
            "dominance": (
                f"{btc_ctx}Write a tweet about BTC dominance and what it means for altcoin traders right now. "
                f"One concrete implication — rising or falling, what's the trade?"
            ),
            "risk_lesson": (
                f"{btc_ctx}Write a tweet with one real insight about risk management in the current market. "
                f"Not generic — something specific to what the market is doing right now."
            ),
            "timing_take": (
                f"{btc_ctx}Write a tweet about timing in this market — "
                f"when to wait vs when to act. Based on what the chart is showing, not theory."
            ),
            "macro_watch": (
                f"{btc_ctx}Write a tweet about a macro factor (DXY, rates, equities, geopolitics) that's relevant "
                f"for crypto right now. One sentence of context, one concrete implication."
            ),
        }

        system_msg = (
            "You are a sharp crypto market analyst who posts on X. "
            "You've traded through multiple cycles and know how to read market structure. "
            "Post educational market observations — not advice, not hype, not product mentions. "
            "Lowercase, no hashtags, no emojis, no NFA/DYOR. Write like a sharp trader texting a colleague. "
            "Be specific and concrete. Vague market commentary is useless. Max 220 characters."
        )

        tweet_text = await _call_grok_tweet(
            angle_prompts[angle], max_chars=220,
            label=f"market_edu/{angle}",
            system=system_msg,
        )

        if not tweet_text:
            fallbacks = [
                f"{btc_ctx or ''}market structure matters more than price. BTC can be at 80k and still look weak. learn to read structure.",
                f"volume doesn't lie. price can be manipulated short term but volume tells you who's actually participating.",
                f"{btc_ctx or ''}the alts that lead a recovery are rarely the ones that led the previous run. new cycle, new leaders.",
                f"most traders lose not because they're wrong about direction but because they're wrong about timing.",
                f"{btc_ctx or ''}BTC dominance rising = alts underperform. dominance falling = alt season. simple model that still works.",
                f"the best entries come from waiting. the worst losses from chasing. discipline in entries is worth more than any indicator.",
                f"{btc_ctx or ''}if you can't explain your position sizing logic in one sentence, you probably don't have one.",
                f"ranges resolve. the question is always direction and timing. both are harder than people admit.",
            ]
            tweet_text = random.choice(fallbacks)

        # Append top gainer tickers for discoverability
        if ticker_str and (len(tweet_text) + len(ticker_str) + 2) <= 278:
            tweet_text = tweet_text + "\n\n" + ticker_str

        result = account_poster.post_tweet(tweet_text)

        if result and result.get('success'):
            logger.info(f"✅ Market edu [{angle}] posted")
        else:
            logger.warning(f"Market edu post failed: {result}")

        return result

    except Exception as e:
        logger.error(f"Error posting market edu: {e}")
        return {'success': False, 'error': str(e)}

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
        
        # Twitter shortens all URLs to 23 chars — use that count when checking length
        import re as _re
        def _tw_len(t: str) -> int:
            count = 0
            last = 0
            for m in _re.finditer(r'https?://\S+', t):
                count += m.start() - last
                count += 23
                last = m.end()
            count += len(t) - last
            return count

        if _tw_len(tweet_text) > 280:
            lines = tweet_text.split('\n')
            while _tw_len('\n'.join(lines)) > 280 and len(lines) > 3:
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

        # Strip any accidentally injected strategy CTA — campaign link must be the final word
        _cta_markers = [
            "automating moves like this at tradehubmarkets",
            "running a strategy on exactly this",
            "build a strategy to catch these",
            "my algo already flagged this",
            "this is why I automate",
            "automated plays on moves like this",
            "strategy bot caught this early",
            "have a strategy running on setups",
            "been catching these with an algo",
            "strategy leaderboard is tracking",
            "this is exactly what I automate",
            "catching these moves automatically",
        ]
        for _marker in _cta_markers:
            _idx = tweet_text.lower().find(_marker.lower())
            if _idx != -1:
                tweet_text = tweet_text[:_idx].rstrip()
        
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
        
        # Append today's actual top-gainer cashtags — fetch fresh if cache is cold
        _bt_tickers = await _get_ticker_suffix()
        if _bt_tickers and len(tweet_text + _bt_tickers) <= 280:
            tweet_text = tweet_text + _bt_tickers

        media_ids = [media_id] if media_id else None
        result = account_poster.post_tweet(tweet_text, media_ids=media_ids)
        
        if result and result.get('success'):
            logger.info(f"Campaign tweet posted (template: {template['id']})")
        
        return result
        
    except Exception as e:
        logger.error(f"Error posting campaign tweet: {e}")
        return {'success': False, 'error': str(e)}


async def post_yubit_campaign(account_poster) -> Optional[Dict]:
    """Post a Yubit campaign tweet with the campaign image, live tickers"""
    global _yubit_post_index

    now = datetime.utcnow()
    if now < YUBIT_CAMPAIGN_START or now > YUBIT_CAMPAIGN_END:
        logger.info("Yubit campaign not active, skipping")
        return None

    try:
        template = YUBIT_CAMPAIGN_TEMPLATES[_yubit_post_index % len(YUBIT_CAMPAIGN_TEMPLATES)]
        _yubit_post_index += 1

        live_tickers = await get_live_tickers_for_campaign()

        tweet_text = template['text'].format(
            link=YUBIT_CAMPAIGN_LINK,
            **live_tickers
        )

        # Twitter shortens all URLs to 23 chars — use that count when checking length
        import re as _re
        def _tw_len(t: str) -> int:
            count = 0
            last = 0
            for m in _re.finditer(r'https?://\S+', t):
                count += m.start() - last
                count += 23
                last = m.end()
            count += len(t) - last
            return count

        if _tw_len(tweet_text) > 280:
            lines = tweet_text.split('\n')
            while _tw_len('\n'.join(lines)) > 280 and len(lines) > 3:
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
                        if line and not line.startswith('$') and not line.startswith('http') and 'Yubit' not in line:
                            lines.pop(i)
                            break
                    else:
                        break
            tweet_text = '\n'.join(lines)

        media_id = None
        if os.path.exists(YUBIT_CAMPAIGN_IMAGE):
            try:
                with open(YUBIT_CAMPAIGN_IMAGE, 'rb') as f:
                    image_bytes = f.read()

                if hasattr(account_poster, 'upload_media'):
                    media_id = account_poster.upload_media(image_bytes)
                elif hasattr(account_poster, 'api_v1') and account_poster.api_v1:
                    media = account_poster.api_v1.media_upload(
                        filename="yubit_campaign.jpeg",
                        file=io.BytesIO(image_bytes)
                    )
                    media_id = str(media.media_id)

                if media_id:
                    logger.info(f"Yubit campaign image uploaded: {media_id}")
                else:
                    logger.warning("Yubit campaign image upload failed, posting without image")
            except Exception as e:
                logger.error(f"Error uploading Yubit campaign image: {e}")
        else:
            logger.warning(f"Yubit campaign image not found: {YUBIT_CAMPAIGN_IMAGE}")

        # Append today's actual top-gainer cashtags — fetch fresh if cache is cold
        _tickers_suffix = await _get_ticker_suffix()
        if _tickers_suffix and _tw_len(tweet_text + _tickers_suffix) <= 280:
            tweet_text = tweet_text + _tickers_suffix

        media_ids = [media_id] if media_id else None
        tweet_text = await _ai_review_tweet(tweet_text, 'yubit_campaign', {
            'template': template['id'],
            'exchange': 'Yubit',
            'ticker1': live_tickers.get('ticker1', ''), 'ticker2': live_tickers.get('ticker2', ''),
            'campaign': 'deposit + trade volume rewards, ends April 30 2026',
        })
        result = account_poster.post_tweet(tweet_text, media_ids=media_ids)

        if result and result.get('success'):
            logger.info(f"Yubit campaign tweet posted (template: {template['id']})")

        return result

    except Exception as e:
        logger.error(f"Error posting Yubit campaign tweet: {e}")
        return {'success': False, 'error': str(e)}


async def post_bydfi_campaign(account_poster) -> Optional[Dict]:
    """Post a BYDFi x TradeHub $2000 deposit bonus campaign tweet — human style."""
    global _bydfi_post_index

    now = datetime.utcnow()
    if now < BYDFI_CAMPAIGN_START or now > BYDFI_CAMPAIGN_END:
        logger.info("BYDFi campaign not active, skipping")
        return None

    try:
        template = BYDFI_CAMPAIGN_TEMPLATES[_bydfi_post_index % len(BYDFI_CAMPAIGN_TEMPLATES)]
        _bydfi_post_index += 1

        live_tickers = await get_live_tickers_for_campaign()

        tweet_text = template['text'].format(
            link=BYDFI_CAMPAIGN_LINK,
            **live_tickers
        )

        # Twitter shortens all URLs to 23 chars — use that count when checking length
        import re as _re
        def _tw_len(t: str) -> int:
            count = 0
            last = 0
            for m in _re.finditer(r'https?://\S+', t):
                count += m.start() - last
                count += 23
                last = m.end()
            count += len(t) - last
            return count

        if _tw_len(tweet_text) > 280:
            lines = tweet_text.split('\n')
            while _tw_len('\n'.join(lines)) > 280 and len(lines) > 3:
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
                        if line and not line.startswith('$') and not line.startswith('http') and 'BYDFi' not in line:
                            lines.pop(i)
                            break
                    else:
                        break
            tweet_text = '\n'.join(lines)

        media_id = None
        if os.path.exists(BYDFI_CAMPAIGN_IMAGE):
            try:
                with open(BYDFI_CAMPAIGN_IMAGE, 'rb') as f:
                    image_bytes = f.read()

                if hasattr(account_poster, 'upload_media'):
                    media_id = account_poster.upload_media(image_bytes)
                elif hasattr(account_poster, 'api_v1') and account_poster.api_v1:
                    media = account_poster.api_v1.media_upload(
                        filename="bydfi_campaign.jpeg",
                        file=io.BytesIO(image_bytes)
                    )
                    media_id = str(media.media_id)

                if media_id:
                    logger.info(f"BYDFi campaign image uploaded: {media_id}")
                else:
                    logger.warning("BYDFi campaign image upload failed, posting without image")
            except Exception as e:
                logger.error(f"Error uploading BYDFi campaign image: {e}")
        else:
            logger.warning(f"BYDFi campaign image not found: {BYDFI_CAMPAIGN_IMAGE}")

        # Append today's actual top-gainer cashtags — fetch fresh if cache is cold
        _tickers_suffix = await _get_ticker_suffix()
        if _tickers_suffix and _tw_len(tweet_text + _tickers_suffix) <= 280:
            tweet_text = tweet_text + _tickers_suffix

        media_ids = [media_id] if media_id else None
        tweet_text = await _ai_review_tweet(tweet_text, 'bydfi_campaign', {
            'template': template['id'],
            'exchange': 'BYDFi',
            'ticker1': live_tickers.get('ticker1', ''),
            'ticker2': live_tickers.get('ticker2', ''),
            'ticker3': live_tickers.get('ticker3', ''),
            'campaign': '$2000 deposit bonus, USDT-M futures, ends April 30 2026',
            'style_note': 'Keep it lowercase and casual like a real trader. No em dashes. No marketing-speak. Keep all dollar amounts and the link intact.',
        })
        result = account_poster.post_tweet(tweet_text, media_ids=media_ids)

        if result and result.get('success'):
            logger.info(f"BYDFi campaign tweet posted (template: {template['id']})")

        return result

    except Exception as e:
        logger.error(f"Error posting BYDFi campaign tweet: {e}")
        return {'success': False, 'error': str(e)}
