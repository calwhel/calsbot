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


def is_clear_command(text: str) -> bool:
    """Check if user wants to clear conversation"""
    text_lower = text.lower().strip()
    return any(cmd in text_lower for cmd in CLEAR_COMMANDS)


def is_trading_question(text: str) -> bool:
    """Check if message is a trading-related question"""
    text_lower = text.lower()
    
    if text.startswith('/'):
        return False
    
    if len(text) < 5:
        return False
    
    has_question_mark = '?' in text
    has_trading_keyword = any(kw in text_lower for kw in TRADING_KEYWORDS)
    has_coin_mention = any(re.search(pattern, text.upper()) for pattern in COIN_PATTERNS)
    
    # Any message with a coin mention triggers AI
    if has_coin_mention:
        return True
    
    # Any question with trading keywords
    if has_question_mark and has_trading_keyword:
        return True
    
    # Question starters with any trading context
    question_starters = ['should', 'is ', 'what', 'how', 'when', 'why', 'can', 'will', 'would', 'tell', 'any']
    starts_with_question = any(text_lower.strip().startswith(q) for q in question_starters)
    
    if starts_with_question:
        return True
    
    # Trading keywords alone are enough
    if has_trading_keyword:
        return True
    
    return False


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
        
        # Calculate approximate cost
        input_tokens = response.usage.prompt_tokens if response.usage else 0
        output_tokens = response.usage.completion_tokens if response.usage else 0
        # GPT-4o-mini pricing: $0.15/1M input, $0.60/1M output
        cost = (input_tokens * 0.00000015) + (output_tokens * 0.0000006)
        
        # Add cost footer
        answer += f"\n\n<i>Cost: ${cost:.4f}</i>"
        
        return answer
        
    except Exception as e:
        logger.error(f"AI assistant error: {e}")
        return None
