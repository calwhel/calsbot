"""
AI Chat Assistant - Natural language trading assistant
Answers user questions about markets, coins, and trading decisions
"""
import os
import re
import logging
import time
from typing import Dict, Optional, List
from collections import deque
import asyncio

logger = logging.getLogger(__name__)

# Conversation memory - stores last N messages per user
# Format: {user_id: deque([(role, content, timestamp), ...])}
_conversation_memory: Dict[int, deque] = {}
MAX_MEMORY_MESSAGES = 10  # Remember last 10 exchanges
MEMORY_EXPIRY_SECONDS = 3600  # Clear after 1 hour of inactivity


def get_conversation_history(user_id: int) -> List[Dict]:
    """Get conversation history for a user"""
    if user_id not in _conversation_memory:
        return []
    
    # Clean expired messages
    current_time = time.time()
    history = _conversation_memory[user_id]
    
    # Filter out expired messages
    valid_messages = [(role, content, ts) for role, content, ts in history 
                      if current_time - ts < MEMORY_EXPIRY_SECONDS]
    
    if len(valid_messages) != len(history):
        _conversation_memory[user_id] = deque(valid_messages, maxlen=MAX_MEMORY_MESSAGES)
    
    # Convert to OpenAI format
    return [{"role": role, "content": content} for role, content, _ in valid_messages]


def add_to_conversation(user_id: int, role: str, content: str):
    """Add a message to conversation history"""
    if user_id not in _conversation_memory:
        _conversation_memory[user_id] = deque(maxlen=MAX_MEMORY_MESSAGES)
    
    _conversation_memory[user_id].append((role, content, time.time()))


def clear_conversation(user_id: int):
    """Clear conversation history for a user"""
    if user_id in _conversation_memory:
        del _conversation_memory[user_id]

COIN_PATTERNS = [
    r'\b(BTC|BITCOIN)\b',
    r'\b(ETH|ETHEREUM)\b',
    r'\b(SOL|SOLANA)\b',
    r'\b(BNB)\b',
    r'\b(XRP|RIPPLE)\b',
    r'\b(DOGE|DOGECOIN)\b',
    r'\b(ADA|CARDANO)\b',
    r'\b(AVAX|AVALANCHE)\b',
    r'\b(LINK|CHAINLINK)\b',
    r'\b(DOT|POLKADOT)\b',
    r'\b(MATIC|POLYGON)\b',
    r'\b(SHIB)\b',
    r'\b(LTC|LITECOIN)\b',
    r'\b(UNI|UNISWAP)\b',
    r'\b(ATOM|COSMOS)\b',
    r'\b(APT|APTOS)\b',
    r'\b(ARB|ARBITRUM)\b',
    r'\b(OP|OPTIMISM)\b',
    r'\b(SUI)\b',
    r'\b(SEI)\b',
    r'\b(TIA|CELESTIA)\b',
    r'\b(INJ|INJECTIVE)\b',
    r'\b(FET)\b',
    r'\b(RENDER|RNDR)\b',
    r'\b(WIF)\b',
    r'\b(PEPE)\b',
    r'\b(BONK)\b',
    r'\b(FLOKI)\b',
]

COIN_MAPPING = {
    'BITCOIN': 'BTC',
    'ETHEREUM': 'ETH',
    'SOLANA': 'SOL',
    'RIPPLE': 'XRP',
    'DOGECOIN': 'DOGE',
    'CARDANO': 'ADA',
    'AVALANCHE': 'AVAX',
    'CHAINLINK': 'LINK',
    'POLKADOT': 'DOT',
    'POLYGON': 'MATIC',
    'LITECOIN': 'LTC',
    'UNISWAP': 'UNI',
    'COSMOS': 'ATOM',
    'APTOS': 'APT',
    'ARBITRUM': 'ARB',
    'OPTIMISM': 'OP',
    'CELESTIA': 'TIA',
    'INJECTIVE': 'INJ',
    'RENDER': 'RNDR',
}

TRADING_KEYWORDS = [
    'should i', 'is it', 'what about', 'how is', "what's happening",
    'bullish', 'bearish', 'long', 'short', 'buy', 'sell',
    'trade', 'entry', 'exit', 'target', 'stop loss', 'tp', 'sl',
    'price', 'pump', 'dump', 'moon', 'crash', 'dip',
    'market', 'trend', 'momentum', 'volume', 'rsi',
    'support', 'resistance', 'breakout', 'breakdown',
    'good time', 'bad time', 'worth', 'risky',
    'analysis', 'prediction', 'forecast', 'outlook',
    'when', 'why', 'explain', 'tell me',
]

# Commands to clear conversation
CLEAR_COMMANDS = ['new chat', 'clear chat', 'reset chat', 'start over', 'forget', 'new conversation']

# Commands for market scanner
SCANNER_TRIGGERS = [
    "what's moving", "whats moving", "what is moving",
    "find opportunities", "find me opportunities", "any opportunities",
    "what should i trade", "what to trade", "best trades",
    "scan the market", "market scan", "scan market",
    "top movers", "what's hot", "whats hot", "what is hot",
    "any setups", "good setups", "best setups",
    "what looks good", "anything interesting"
]


def is_clear_command(text: str) -> bool:
    """Check if user wants to clear conversation"""
    text_lower = text.lower().strip()
    return any(cmd in text_lower for cmd in CLEAR_COMMANDS)


def is_scanner_request(text: str) -> bool:
    """Check if user wants a market scan"""
    text_lower = text.lower().strip()
    return any(trigger in text_lower for trigger in SCANNER_TRIGGERS)


def is_trading_question(text: str) -> bool:
    """Check if message should trigger AI - any non-command message"""
    if text.startswith('/'):
        return False
    
    if len(text.strip()) < 2:
        return False
    
    # Reply to any message that's not a command
    return True


def extract_coins(text: str) -> List[str]:
    """Extract coin symbols from text"""
    coins = set()
    text_upper = text.upper()
    
    for pattern in COIN_PATTERNS:
        matches = re.findall(pattern, text_upper)
        for match in matches:
            coin = COIN_MAPPING.get(match, match)
            coins.add(coin)
    
    return list(coins)


async def get_coin_context(symbol: str) -> Dict:
    """Fetch real-time market data for a coin"""
    try:
        import ccxt.async_support as ccxt
        
        exchange = ccxt.binance({
            'enableRateLimit': True,
            'options': {'defaultType': 'future'}
        })
        
        try:
            pair = f"{symbol}/USDT"
            
            ticker = await exchange.fetch_ticker(pair)
            ohlcv = await exchange.fetch_ohlcv(pair, '15m', limit=50)
            
            if not ohlcv:
                return {'error': f'No data for {symbol}'}
            
            closes = [c[4] for c in ohlcv]
            volumes = [c[5] for c in ohlcv]
            
            delta = [closes[i] - closes[i-1] for i in range(1, len(closes))]
            gains = [d if d > 0 else 0 for d in delta]
            losses = [-d if d < 0 else 0 for d in delta]
            avg_gain = sum(gains[-14:]) / 14 if len(gains) >= 14 else 0
            avg_loss = sum(losses[-14:]) / 14 if len(losses) >= 14 else 0
            rs = avg_gain / avg_loss if avg_loss > 0 else 100
            rsi = 100 - (100 / (1 + rs))
            
            avg_vol = sum(volumes[:-1]) / len(volumes[:-1]) if len(volumes) > 1 else 1
            vol_ratio = volumes[-1] / avg_vol if avg_vol > 0 else 1
            
            price = ticker['last']
            change_24h = ticker.get('percentage', 0) or 0
            high_24h = ticker.get('high', price)
            low_24h = ticker.get('low', price)
            
            ema_20 = sum(closes[-20:]) / 20 if len(closes) >= 20 else price
            trend = "bullish" if price > ema_20 else "bearish"
            
            recent_high = max(closes[-10:])
            recent_low = min(closes[-10:])
            
            return {
                'symbol': symbol,
                'price': price,
                'change_24h': change_24h,
                'high_24h': high_24h,
                'low_24h': low_24h,
                'rsi': rsi,
                'volume_ratio': vol_ratio,
                'trend': trend,
                'ema_20': ema_20,
                'recent_high': recent_high,
                'recent_low': recent_low,
            }
            
        finally:
            await exchange.close()
            
    except Exception as e:
        logger.error(f"Error fetching {symbol} data: {e}")
        return {'error': str(e)}


async def get_market_overview() -> Dict:
    """Get general market overview (BTC dominance, sentiment)"""
    try:
        btc_data = await get_coin_context('BTC')
        eth_data = await get_coin_context('ETH')
        
        return {
            'btc_price': btc_data.get('price', 0),
            'btc_change': btc_data.get('change_24h', 0),
            'btc_rsi': btc_data.get('rsi', 50),
            'btc_trend': btc_data.get('trend', 'neutral'),
            'eth_price': eth_data.get('price', 0),
            'eth_change': eth_data.get('change_24h', 0),
        }
    except Exception as e:
        logger.error(f"Error getting market overview: {e}")
        return {}


async def scan_market_opportunities() -> Optional[str]:
    """Scan top coins and find best trading opportunities using AI"""
    try:
        import ccxt.async_support as ccxt
        from openai import OpenAI
        
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            return "I need an API key to scan. Please set up your OpenAI API key."
        
        exchange = ccxt.binance({
            'enableRateLimit': True,
            'options': {'defaultType': 'future'}
        })
        
        try:
            # Fetch top movers
            tickers = await exchange.fetch_tickers()
            
            # Filter USDT pairs and sort by volume
            usdt_pairs = []
            for symbol, ticker in tickers.items():
                if symbol.endswith('/USDT') and ticker.get('quoteVolume', 0) > 10000000:
                    change = ticker.get('percentage', 0) or 0
                    usdt_pairs.append({
                        'symbol': symbol.replace('/USDT', ''),
                        'price': ticker.get('last', 0),
                        'change': change,
                        'volume': ticker.get('quoteVolume', 0),
                        'high': ticker.get('high', 0),
                        'low': ticker.get('low', 0),
                    })
            
            # Get top gainers, losers, and volume leaders
            top_gainers = sorted(usdt_pairs, key=lambda x: x['change'], reverse=True)[:5]
            top_losers = sorted(usdt_pairs, key=lambda x: x['change'])[:5]
            top_volume = sorted(usdt_pairs, key=lambda x: x['volume'], reverse=True)[:5]
            
            # Get detailed data for interesting coins
            interesting_coins = list(set(
                [c['symbol'] for c in top_gainers[:3]] + 
                [c['symbol'] for c in top_losers[:2]] +
                [c['symbol'] for c in top_volume[:2]]
            ))[:6]
            
            detailed_data = []
            for coin in interesting_coins:
                data = await get_coin_context(coin)
                if not data.get('error'):
                    detailed_data.append(data)
            
            # Build market summary
            market_summary = f"""
TOP GAINERS (24h):
{chr(10).join([f"• {c['symbol']}: {c['change']:+.1f}% @ ${c['price']:,.4f}" for c in top_gainers])}

TOP LOSERS (24h):
{chr(10).join([f"• {c['symbol']}: {c['change']:+.1f}% @ ${c['price']:,.4f}" for c in top_losers])}

HIGHEST VOLUME:
{chr(10).join([f"• {c['symbol']}: ${c['volume']/1e6:.0f}M volume, {c['change']:+.1f}%" for c in top_volume])}

DETAILED ANALYSIS:
"""
            for data in detailed_data:
                market_summary += f"""
{data['symbol']}:
- Price: ${data['price']:,.6f} | 24h: {data['change_24h']:+.1f}%
- RSI: {data['rsi']:.0f} | Volume: {data['volume_ratio']:.1f}x avg
- Trend: {data['trend']} | Range: ${data['recent_low']:,.6f} - ${data['recent_high']:,.6f}
"""
            
        finally:
            await exchange.close()
        
        # Ask AI to analyze
        client = OpenAI(api_key=api_key, timeout=30.0)
        
        system_prompt = """You are an expert crypto trader scanning for opportunities.

RULES:
1. Identify 2-3 BEST opportunities from the data
2. For each opportunity, give: direction (LONG/SHORT), entry zone, stop loss, take profit
3. Explain WHY in 1 sentence
4. Rate confidence (1-10)
5. Consider: RSI extremes, volume spikes, trend alignment, 24h momentum
6. Prefer: oversold bounces, overbought shorts, breakout volume
7. Be specific with price levels
8. Format cleanly with emojis

OUTPUT FORMAT:
🎯 TOP OPPORTUNITIES

1. [COIN] - [LONG/SHORT]
   Entry: $X.XX - $X.XX
   Stop: $X.XX | TP: $X.XX
   Why: [reason]
   Confidence: X/10

2. ...

💡 MARKET VIBE: [1 sentence summary]"""

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Analyze this market data and find the best 2-3 trading opportunities:\n\n{market_summary}"}
            ],
            max_tokens=500,
            temperature=0.2  # Low temperature for consistent trade ideas
        )
        
        return response.choices[0].message.content.strip()
        
    except Exception as e:
        logger.error(f"Market scanner error: {e}", exc_info=True)
        return None


# Position coach triggers
POSITION_TRIGGERS = [
    "should i close", "close my", "exit my", "take profit", 
    "hold my", "keep my", "my position", "my trade", "my long", "my short",
    "what about my", "how's my position", "hows my position"
]


def is_position_question(text: str) -> bool:
    """Check if user is asking about their positions"""
    text_lower = text.lower().strip()
    return any(trigger in text_lower for trigger in POSITION_TRIGGERS)


async def get_user_positions(user_id: int) -> List[Dict]:
    """Fetch user's open positions from Bitunix"""
    try:
        from app.database import SessionLocal
        from app.models import User
        from app.services.bitunix_trader import BitunixTrader
        
        db = SessionLocal()
        try:
            user = db.query(User).filter(User.telegram_id == str(user_id)).first()
            logger.info(f"🔍 Position Coach: Looking up user {user_id}")
            
            if not user:
                logger.info(f"🔍 Position Coach: User {user_id} not found in database")
                return []
            
            if not user.bitunix_api_key:
                logger.info(f"🔍 Position Coach: User {user_id} has no Bitunix API key")
                return []
            
            logger.info(f"🔍 Position Coach: Creating trader for user {user_id}")
            trader = BitunixTrader(user.bitunix_api_key, user.bitunix_api_secret)
            positions = await trader.get_open_positions()
            await trader.close()
            
            logger.info(f"🔍 Position Coach: Found {len(positions) if positions else 0} positions for user {user_id}")
            if positions:
                logger.info(f"🔍 Position Coach: Raw positions: {positions}")
            
            return positions if positions else []
        finally:
            db.close()
    except Exception as e:
        logger.error(f"Error fetching positions for user {user_id}: {e}", exc_info=True)
        return []


async def analyze_positions(user_id: int, question: str) -> Optional[str]:
    """AI analysis of user's open positions"""
    try:
        from openai import OpenAI
        
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            return "I need an API key to analyze positions."
        
        positions = await get_user_positions(user_id)
        
        if not positions:
            return "You don't have any open positions right now. Would you like me to scan for opportunities?"
        
        # Get current market data for each position
        position_data = []
        for pos in positions:
            symbol = pos.get('symbol', '').replace('USDT', '').replace('_', '')
            if symbol:
                market_data = await get_coin_context(symbol)
                # Map Bitunix field names correctly
                position_data.append({
                    'symbol': symbol,
                    'side': pos.get('hold_side', 'unknown'),  # Bitunix uses 'hold_side'
                    'size': pos.get('total', 0),  # Bitunix uses 'total'
                    'entry': pos.get('entry_price', 0),  # Bitunix uses 'entry_price' (mapped from openPriceAvg)
                    'current_price': pos.get('mark_price', 0) or market_data.get('price', 0),  # Bitunix provides mark_price
                    'unrealized_pnl': pos.get('unrealized_pl', 0),  # Bitunix uses 'unrealized_pl'
                    'leverage': pos.get('leverage', 1),
                    'rsi': market_data.get('rsi', 50),
                    'trend': market_data.get('trend', 'neutral'),
                    'volume_ratio': market_data.get('volume_ratio', 1),
                })
        
        # Build position summary
        position_summary = "YOUR OPEN POSITIONS:\n"
        for p in position_data:
            pnl_pct = ((p['current_price'] - p['entry']) / p['entry'] * 100) if p['entry'] > 0 else 0
            if p['side'].lower() == 'short':
                pnl_pct = -pnl_pct
            position_summary += f"""
{p['symbol']} {p['side'].upper()} @ {p['leverage']}x
- Entry: ${p['entry']:.6f} | Now: ${p['current_price']:.6f}
- PnL: {pnl_pct:+.2f}%
- RSI: {p['rsi']:.0f} | Trend: {p['trend']} | Volume: {p['volume_ratio']:.1f}x
"""
        
        client = OpenAI(api_key=api_key, timeout=20.0)
        
        system_prompt = """You are a trading coach helping a user manage their open positions.

RULES:
1. Be direct and actionable
2. Consider: current PnL, RSI, trend, volume
3. Give specific advice: HOLD, CLOSE, PARTIAL CLOSE, or MOVE STOP
4. If closing, suggest where to take profit
5. If holding, suggest where to set stop loss
6. Consider the user's specific question
7. Keep response concise (3-5 sentences max)
8. Use emojis sparingly"""

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"{position_summary}\n\nUser question: {question}"}
            ],
            max_tokens=300,
            temperature=0.3  # Lower temperature for consistent trade advice
        )
        
        return response.choices[0].message.content.strip()
        
    except Exception as e:
        logger.error(f"Position analysis error: {e}", exc_info=True)
        return None


async def generate_daily_digest() -> Optional[str]:
    """Generate daily market digest with top opportunities"""
    try:
        import ccxt.async_support as ccxt
        from openai import OpenAI
        
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            return None
        
        exchange = ccxt.binance({
            'enableRateLimit': True,
            'options': {'defaultType': 'future'}
        })
        
        try:
            tickers = await exchange.fetch_tickers()
            
            usdt_pairs = []
            for symbol, ticker in tickers.items():
                if symbol.endswith('/USDT') and ticker.get('quoteVolume', 0) > 50000000:
                    change = ticker.get('percentage', 0) or 0
                    usdt_pairs.append({
                        'symbol': symbol.replace('/USDT', ''),
                        'price': ticker.get('last', 0),
                        'change': change,
                        'volume': ticker.get('quoteVolume', 0),
                    })
            
            top_gainers = sorted(usdt_pairs, key=lambda x: x['change'], reverse=True)[:5]
            top_losers = sorted(usdt_pairs, key=lambda x: x['change'])[:5]
            
            # Get detailed data for top 4 coins
            detailed = []
            for coin in (top_gainers[:2] + top_losers[:2]):
                data = await get_coin_context(coin['symbol'])
                if not data.get('error'):
                    detailed.append(data)
            
        finally:
            await exchange.close()
        
        market_data = f"""
TOP GAINERS:
{chr(10).join([f"• {c['symbol']}: {c['change']:+.1f}%" for c in top_gainers])}

TOP LOSERS:
{chr(10).join([f"• {c['symbol']}: {c['change']:+.1f}%" for c in top_losers])}

DETAILED:
"""
        for d in detailed:
            market_data += f"{d['symbol']}: RSI {d['rsi']:.0f}, {d['trend']}, {d['volume_ratio']:.1f}x vol\n"
        
        client = OpenAI(api_key=api_key, timeout=20.0)
        
        system_prompt = """Create a brief daily crypto trading digest.

FORMAT:
☀️ DAILY DIGEST

📊 MARKET MOOD: [1 sentence - bullish/bearish/mixed]

🎯 TOP OPPORTUNITIES:
1. [COIN] [LONG/SHORT] - [1 sentence why]
2. [COIN] [LONG/SHORT] - [1 sentence why]

⚠️ WATCH OUT: [1 coin to avoid and why]

💡 TIP: [Quick actionable advice for today]

Keep it SHORT and punchy - traders are busy!"""

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Create today's digest based on:\n{market_data}"}
            ],
            max_tokens=400,
            temperature=0.7
        )
        
        return response.choices[0].message.content.strip()
        
    except Exception as e:
        logger.error(f"Daily digest error: {e}", exc_info=True)
        return None


async def ask_ai_assistant(
    question: str,
    coins: List[str] = None,
    user_context: str = "",
    user_id: int = None
) -> Optional[str]:
    """
    Get AI response to user's trading question
    
    Args:
        question: User's natural language question
        coins: List of coin symbols mentioned
        user_context: Additional context about the user
        user_id: Telegram user ID for conversation memory
        
    Returns:
        AI-generated response or None if error
    """
    try:
        from openai import OpenAI
        
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            return "I need an API key to answer questions. Please set up your OpenAI API key."
        
        client = OpenAI(api_key=api_key, timeout=20.0)
        
        # Get conversation history
        conversation_history = []
        if user_id:
            conversation_history = get_conversation_history(user_id)
            # Add current question to memory
            add_to_conversation(user_id, "user", question)
        
        coin_data = []
        if coins:
            tasks = [get_coin_context(coin) for coin in coins[:3]]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for result in results:
                if isinstance(result, dict) and not result.get('error'):
                    coin_data.append(result)
        
        market = await get_market_overview()
        
        market_context = f"""
CURRENT MARKET CONDITIONS:
- BTC: ${market.get('btc_price', 0):,.2f} ({market.get('btc_change', 0):+.2f}% 24h) | RSI: {market.get('btc_rsi', 50):.0f} | Trend: {market.get('btc_trend', 'neutral')}
- ETH: ${market.get('eth_price', 0):,.2f} ({market.get('eth_change', 0):+.2f}% 24h)
"""
        
        if coin_data:
            market_context += "\nCOIN DATA:\n"
            for data in coin_data:
                market_context += f"""
{data['symbol']}/USDT:
- Price: ${data['price']:,.6f}
- 24h Change: {data['change_24h']:+.2f}%
- RSI (15m): {data['rsi']:.0f}
- Volume: {data['volume_ratio']:.1f}x average
- Trend: {data['trend']}
- EMA20: ${data['ema_20']:,.6f}
- Recent Range: ${data['recent_low']:,.6f} - ${data['recent_high']:,.6f}
"""
        
        system_prompt = """You are an expert crypto trading assistant. You help traders make informed decisions.

RULES:
1. Be concise but helpful - aim for 2-4 sentences max
2. Give actionable insights, not generic advice
3. Reference the real market data provided
4. Be honest about uncertainty - crypto is volatile
5. For trade recommendations, mention key levels (entry, SL, TP)
6. Use simple language, avoid jargon
7. Include relevant emojis sparingly
8. NEVER guarantee profits - always mention risk
9. If asked about timing, be specific (now vs wait for pullback)
10. Consider BTC's trend when discussing altcoins

RESPONSE STYLE:
- Direct and conversational
- Focus on what matters NOW
- Include 1-2 specific price levels when relevant
- End with a clear takeaway or action point"""

        user_prompt = f"""{market_context}

{user_context}

USER QUESTION: {question}

Provide a helpful, concise response based on the current market data."""

        # Build messages with conversation history
        messages = [{"role": "system", "content": system_prompt}]
        
        # Add conversation history (last few exchanges for context)
        if conversation_history:
            messages.extend(conversation_history[-6:])  # Last 3 exchanges
        
        # Add current question with market context
        messages.append({"role": "user", "content": user_prompt})
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            max_tokens=300,
            temperature=0.7
        )
        
        answer = response.choices[0].message.content.strip()
        
        # Save assistant response to memory
        if user_id:
            add_to_conversation(user_id, "assistant", answer)
        
        return answer
        
    except Exception as e:
        logger.error(f"AI assistant error: {e}")
        return None
