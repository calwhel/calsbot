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
import random

logger = logging.getLogger(__name__)


def get_openai_api_key():
    """Get OpenAI API key - checks both Railway and Replit sources."""
    key = os.environ.get("OPENAI_API_KEY") or os.environ.get("AI_INTEGRATIONS_OPENAI_API_KEY")
    if key and "DUMMY" in key.upper():
        return None
    return key


# Per-user request throttling
_user_last_request: Dict[int, float] = {}
MIN_REQUEST_INTERVAL = 2.0  # Minimum seconds between requests per user

# Global request semaphore to limit concurrent API calls
_api_semaphore = asyncio.Semaphore(5)  # Max 5 concurrent OpenAI calls

# Fallback responses when AI is unavailable
FALLBACK_RESPONSES = [
    "I'm analyzing a lot of market data right now. Give me a moment and try again!",
    "The market's moving fast! Let me catch my breath - try again in a few seconds.",
    "Processing multiple requests. Please try again shortly!",
    "I'm a bit busy at the moment. Try again in a few seconds!",
]


def get_fallback_response() -> str:
    """Get a random friendly fallback response"""
    return random.choice(FALLBACK_RESPONSES)


def check_user_throttle(user_id: int) -> tuple[bool, float]:
    """Check if user should be throttled. Returns (should_wait, wait_time)"""
    if user_id not in _user_last_request:
        return False, 0
    
    elapsed = time.time() - _user_last_request[user_id]
    if elapsed < MIN_REQUEST_INTERVAL:
        return True, MIN_REQUEST_INTERVAL - elapsed
    return False, 0


def update_user_throttle(user_id: int):
    """Update user's last request time"""
    _user_last_request[user_id] = time.time()


async def call_openai_with_retry(client, messages, model="gpt-4o-mini", max_retries=4, timeout=30.0):
    """Call OpenAI API with exponential backoff retry on rate limits"""
    last_error = None
    
    async with _api_semaphore:  # Limit concurrent API calls
        for attempt in range(max_retries):
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=0.7,
                    max_tokens=800,
                    timeout=timeout
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                last_error = e
                error_str = str(e)
                
                # Check if it's a rate limit error
                if "429" in error_str or "rate limit" in error_str.lower():
                    # Exponential backoff with jitter - longer waits for more attempts
                    wait_time = (2 ** attempt) + random.uniform(1.0, 3.0)
                    logger.warning(f"OpenAI rate limit hit, retrying in {wait_time:.1f}s (attempt {attempt+1}/{max_retries})")
                    await asyncio.sleep(wait_time)
                elif "timeout" in error_str.lower() or "timed out" in error_str.lower():
                    # Timeout - quick retry
                    wait_time = 1.0 + random.uniform(0.5, 1.5)
                    logger.warning(f"OpenAI timeout, retrying in {wait_time:.1f}s (attempt {attempt+1}/{max_retries})")
                    await asyncio.sleep(wait_time)
                else:
                    # Non-retryable error
                    raise e
    
    # All retries exhausted
    raise last_error

# Conversation memory - stores last N messages per user
# Format: {user_id: deque([(role, content, timestamp), ...])}
_conversation_memory: Dict[int, deque] = {}
MAX_MEMORY_MESSAGES = 10  # Remember last 10 exchanges
MEMORY_EXPIRY_SECONDS = 3600  # Clear after 1 hour of inactivity

# Price cache to avoid Binance rate limits
# Format: {symbol: (data_dict, timestamp)}
_price_cache: Dict[str, tuple] = {}
PRICE_CACHE_TTL = 30  # Cache prices for 30 seconds


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
    # Major coins
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
    # Memecoins
    r'\b(WIF)\b',
    r'\b(PEPE)\b',
    r'\b(BONK)\b',
    r'\b(FLOKI)\b',
    r'\b(MEME)\b',
    r'\b(TURBO)\b',
    r'\b(POPCAT)\b',
    r'\b(MOG)\b',
    r'\b(NEIRO)\b',
    r'\b(PNUT)\b',
    r'\b(ACT)\b',
    r'\b(GOAT)\b',
    # AI coins
    r'\b(TAO|BITTENSOR)\b',
    r'\b(NEAR)\b',
    r'\b(ICP)\b',
    r'\b(GRT|GRAPH)\b',
    r'\b(AGIX)\b',
    r'\b(OCEAN)\b',
    r'\b(AKT|AKASH)\b',
    r'\b(OLAS)\b',
    r'\b(VIRTUAL)\b',
    r'\b(AI16Z)\b',
    # Layer 2 / Infrastructure
    r'\b(STRK|STARKNET)\b',
    r'\b(ZK|ZKSYNC)\b',
    r'\b(MANTA)\b',
    r'\b(BLAST)\b',
    r'\b(MODE)\b',
    r'\b(SCROLL)\b',
    r'\b(BASE)\b',
    # DeFi
    r'\b(AAVE)\b',
    r'\b(MKR|MAKER)\b',
    r'\b(CRV|CURVE)\b',
    r'\b(SNX|SYNTHETIX)\b',
    r'\b(COMP|COMPOUND)\b',
    r'\b(SUSHI)\b',
    r'\b(1INCH)\b',
    r'\b(PENDLE)\b',
    r'\b(DYDX)\b',
    r'\b(GMX)\b',
    r'\b(JUP|JUPITER)\b',
    r'\b(RAY|RAYDIUM)\b',
    # Gaming / Metaverse
    r'\b(AXS|AXIE)\b',
    r'\b(SAND|SANDBOX)\b',
    r'\b(MANA|DECENTRALAND)\b',
    r'\b(IMX|IMMUTABLE)\b',
    r'\b(GALA)\b',
    r'\b(ENJ|ENJIN)\b',
    r'\b(PRIME)\b',
    r'\b(BEAM)\b',
    r'\b(PIXEL)\b',
    r'\b(PORTAL)\b',
    # Other popular
    r'\b(FIL|FILECOIN)\b',
    r'\b(HBAR|HEDERA)\b',
    r'\b(VET|VECHAIN)\b',
    r'\b(ALGO|ALGORAND)\b',
    r'\b(XLM|STELLAR)\b',
    r'\b(EOS)\b',
    r'\b(THETA)\b',
    r'\b(FTM|FANTOM)\b',
    r'\b(EGLD|ELROND|MULTIVERSX)\b',
    r'\b(KAVA)\b',
    r'\b(ROSE|OASIS)\b',
    r'\b(JASMY)\b',
    r'\b(CHZ|CHILIZ)\b',
    r'\b(SAND|SANDBOX)\b',
    r'\b(CKB|NERVOS)\b',
    r'\b(STX|STACKS)\b',
    r'\b(KAS|KASPA)\b',
    r'\b(TON|TONCOIN)\b',
    r'\b(TRX|TRON)\b',
    r'\b(RUNE|THORCHAIN)\b',
    r'\b(ENS)\b',
    r'\b(LDO|LIDO)\b',
    r'\b(RPL|ROCKETPOOL)\b',
    r'\b(SSV)\b',
    r'\b(EIGEN)\b',
    r'\b(ENA|ETHENA)\b',
    r'\b(W|WORMHOLE)\b',
    r'\b(JTO|JITO)\b',
    r'\b(PYTH)\b',
    r'\b(WLD|WORLDCOIN)\b',
    r'\b(ZRO|LAYERZERO)\b',
    r'\b(IO)\b',
    r'\b(NOT|NOTCOIN)\b',
    r'\b(DOGS)\b',
    r'\b(HMSTR|HAMSTER)\b',
    r'\b(CATI)\b',
    r'\b(MAJOR)\b',
    r'\b(HYPE|HYPERLIQUID)\b',
    r'\b(VIRTUAL)\b',
    r'\b(AI16Z)\b',
    r'\b(FARTCOIN|FART)\b',
    r'\b(GRIFFAIN)\b',
    r'\b(ZEREBRO)\b',
    r'\b(GOAT)\b',
    r'\b(PNUT)\b',
    r'\b(ACT)\b',
    r'\b(MOODENG)\b',
    r'\b(SPX)\b',
    r'\b(PENGU|PUDGY)\b',
    r'\b(ME|MAGIC EDEN)\b',
    r'\b(MOVE)\b',
    r'\b(USUAL)\b',
    r'\b(VANA)\b',
    r'\b(BIO)\b',
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
    'BITTENSOR': 'TAO',
    'GRAPH': 'GRT',
    'AKASH': 'AKT',
    'STARKNET': 'STRK',
    'ZKSYNC': 'ZK',
    'MAKER': 'MKR',
    'CURVE': 'CRV',
    'SYNTHETIX': 'SNX',
    'COMPOUND': 'COMP',
    'JUPITER': 'JUP',
    'RAYDIUM': 'RAY',
    'AXIE': 'AXS',
    'SANDBOX': 'SAND',
    'DECENTRALAND': 'MANA',
    'IMMUTABLE': 'IMX',
    'ENJIN': 'ENJ',
    'FILECOIN': 'FIL',
    'HEDERA': 'HBAR',
    'VECHAIN': 'VET',
    'ALGORAND': 'ALGO',
    'STELLAR': 'XLM',
    'FANTOM': 'FTM',
    'ELROND': 'EGLD',
    'MULTIVERSX': 'EGLD',
    'OASIS': 'ROSE',
    'CHILIZ': 'CHZ',
    'NERVOS': 'CKB',
    'STACKS': 'STX',
    'KASPA': 'KAS',
    'TONCOIN': 'TON',
    'TRON': 'TRX',
    'THORCHAIN': 'RUNE',
    'LIDO': 'LDO',
    'ROCKETPOOL': 'RPL',
    'ETHENA': 'ENA',
    'WORMHOLE': 'W',
    'JITO': 'JTO',
    'WORLDCOIN': 'WLD',
    'LAYERZERO': 'ZRO',
    'NOTCOIN': 'NOT',
    'HAMSTER': 'HMSTR',
    'HYPERLIQUID': 'HYPE',
    'PUDGY': 'PENGU',
    'MAGIC EDEN': 'ME',
    'FART': 'FARTCOIN',
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
    """Fetch real-time market data for a coin - tries multiple exchanges with caching"""
    global _price_cache
    
    try:
        import ccxt.async_support as ccxt
        
        # Clean symbol
        symbol = symbol.upper().replace('USDT', '').replace('/USDT', '').replace('-USDT', '').strip()
        
        # Check cache first
        if symbol in _price_cache:
            cached_data, cached_time = _price_cache[symbol]
            if time.time() - cached_time < PRICE_CACHE_TTL:
                logger.debug(f"📊 {symbol} from cache (age: {time.time() - cached_time:.0f}s)")
                return cached_data
        
        pair = f"{symbol}/USDT"
        
        # Try exchanges in order - prioritize those with more listings
        # MEXC/Bybit/OKX/Bitget have newer coins, Binance last due to rate limits
        exchanges_to_try = [
            ('mexc', {'enableRateLimit': True}),
            ('bybit', {'enableRateLimit': True}),
            ('okx', {'enableRateLimit': True}),
            ('bitget', {'enableRateLimit': True}),
            ('binance', {'enableRateLimit': True, 'options': {'defaultType': 'spot'}}),
            ('kucoin', {'enableRateLimit': True}),
            ('gate', {'enableRateLimit': True}),
        ]
        
        for exchange_id, config in exchanges_to_try:
            exchange = None
            try:
                exchange = getattr(ccxt, exchange_id)(config)
                
                ticker = await exchange.fetch_ticker(pair)
                ohlcv = await exchange.fetch_ohlcv(pair, '15m', limit=50)
                
                # Close exchange immediately after fetching
                await exchange.close()
                exchange = None
                
                if not ohlcv or not ticker:
                    continue
                
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
                
                # Validate price is reasonable (not 0, not None)
                if not price or price <= 0:
                    logger.warning(f"Invalid price from {exchange_id} for {symbol}: {price}")
                    continue
                
                change_24h = ticker.get('percentage', 0) or 0
                high_24h = ticker.get('high', price)
                low_24h = ticker.get('low', price)
                
                ema_20 = sum(closes[-20:]) / 20 if len(closes) >= 20 else price
                trend = "bullish" if price > ema_20 else "bearish"
                
                recent_high = max(closes[-10:])
                recent_low = min(closes[-10:])
                
                logger.info(f"📊 {symbol} price from {exchange_id}: ${price:.4f}")
                
                result = {
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
                    'source': exchange_id,
                }
                
                # Cache the result
                _price_cache[symbol] = (result, time.time())
                
                return result
                
            except Exception as e:
                logger.debug(f"Failed to get {symbol} from {exchange_id}: {e}")
                if exchange:
                    try:
                        await exchange.close()
                    except:
                        pass
                continue
        
        return {'error': f'No data for {symbol} on any exchange'}
            
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


async def assess_trade_risk(symbol: str, direction: str = 'LONG') -> Dict:
    """
    Assess the risk level for a potential trade
    
    Returns:
        Dict with risk_score (1-10), risk_level, factors, and recommendation
    """
    try:
        coin_data = await get_coin_context(symbol)
        btc_data = await get_coin_context('BTC')
        
        if coin_data.get('error'):
            return {'error': f"Couldn't fetch data for {symbol}"}
        
        risk_factors = []
        risk_score = 5  # Start neutral
        
        rsi = coin_data.get('rsi', 50)
        change_24h = coin_data.get('change_24h', 0)
        volume_ratio = coin_data.get('volume_ratio', 1)
        trend = coin_data.get('trend', 'neutral')
        btc_trend = btc_data.get('trend', 'neutral')
        btc_change = btc_data.get('change_24h', 0)
        
        # RSI extremes
        if direction == 'LONG':
            if rsi > 80:
                risk_score += 3
                risk_factors.append(f"RSI extremely overbought ({rsi:.0f})")
            elif rsi > 70:
                risk_score += 2
                risk_factors.append(f"RSI overbought ({rsi:.0f})")
            elif rsi < 30:
                risk_score -= 1
                risk_factors.append(f"RSI oversold - good for longs ({rsi:.0f})")
        else:  # SHORT
            if rsi < 20:
                risk_score += 3
                risk_factors.append(f"RSI extremely oversold ({rsi:.0f})")
            elif rsi < 30:
                risk_score += 2
                risk_factors.append(f"RSI oversold ({rsi:.0f})")
            elif rsi > 70:
                risk_score -= 1
                risk_factors.append(f"RSI overbought - good for shorts ({rsi:.0f})")
        
        # 24h momentum
        if direction == 'LONG':
            if change_24h > 30:
                risk_score += 2
                risk_factors.append(f"Already pumped +{change_24h:.1f}% - chasing")
            elif change_24h > 15:
                risk_score += 1
                risk_factors.append(f"Extended move +{change_24h:.1f}%")
            elif change_24h < -10:
                risk_factors.append(f"Catching falling knife ({change_24h:.1f}%)")
                risk_score += 1
        else:  # SHORT
            if change_24h < -20:
                risk_score += 2
                risk_factors.append(f"Already dumped {change_24h:.1f}% - chasing")
            elif change_24h > 20:
                risk_factors.append(f"Strong pump - risky to short ({change_24h:+.1f}%)")
                risk_score += 1
        
        # Volume analysis
        if volume_ratio < 0.5:
            risk_score += 1
            risk_factors.append("Low volume - poor liquidity")
        elif volume_ratio > 3:
            risk_factors.append(f"High volume ({volume_ratio:.1f}x) - volatile")
        
        # Trend alignment
        if direction == 'LONG' and trend == 'bearish':
            risk_score += 1
            risk_factors.append("Counter-trend trade (bearish trend)")
        elif direction == 'SHORT' and trend == 'bullish':
            risk_score += 1
            risk_factors.append("Counter-trend trade (bullish trend)")
        
        # BTC correlation
        if btc_trend == 'bearish' and direction == 'LONG':
            risk_score += 1
            risk_factors.append("BTC in downtrend - risky for alts")
        elif btc_change < -3 and direction == 'LONG':
            risk_score += 1
            risk_factors.append(f"BTC dumping ({btc_change:.1f}%)")
        
        # Cap risk score
        risk_score = max(1, min(10, risk_score))
        
        # Determine risk level
        if risk_score <= 3:
            risk_level = "LOW"
            emoji = "🟢"
        elif risk_score <= 5:
            risk_level = "MODERATE"
            emoji = "🟡"
        elif risk_score <= 7:
            risk_level = "HIGH"
            emoji = "🟠"
        else:
            risk_level = "VERY HIGH"
            emoji = "🔴"
        
        # Generate recommendation
        if risk_score <= 4:
            recommendation = "Conditions look favorable. Standard position size recommended."
        elif risk_score <= 6:
            recommendation = "Proceed with caution. Consider reduced position size."
        elif risk_score <= 8:
            recommendation = "High risk setup. Only trade with strict stop-loss and small size."
        else:
            recommendation = "Very risky. Consider waiting for better conditions."
        
        return {
            'symbol': symbol,
            'direction': direction,
            'risk_score': risk_score,
            'risk_level': risk_level,
            'emoji': emoji,
            'factors': risk_factors,
            'recommendation': recommendation,
            'coin_data': coin_data,
            'btc_data': btc_data
        }
        
    except Exception as e:
        logger.error(f"Error assessing risk for {symbol}: {e}")
        return {'error': str(e)}


def is_risk_question(text: str) -> bool:
    """Check if user is asking about trade risk"""
    text_lower = text.lower()
    risk_keywords = ['risk', 'risky', 'safe', 'dangerous', 'how risky', 'is it safe', 'should i']
    direction_keywords = ['long', 'short', 'buy', 'sell', 'trade', 'enter', 'position']
    
    has_risk = any(kw in text_lower for kw in risk_keywords)
    has_direction = any(kw in text_lower for kw in direction_keywords)
    
    return has_risk and has_direction


async def scan_market_opportunities() -> Optional[str]:
    """Scan top coins and find best trading opportunities using AI"""
    try:
        import ccxt.async_support as ccxt
        from openai import OpenAI
        
        api_key = get_openai_api_key()
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

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Analyze this market data and find the best 2-3 trading opportunities:\n\n{market_summary}"}
        ]
        
        try:
            return await call_openai_with_retry(client, messages, max_retries=4, timeout=30.0)
        except Exception as api_error:
            logger.warning(f"Scan API error: {api_error}")
            return get_fallback_response()
        
    except Exception as e:
        logger.error(f"Market scanner error: {e}", exc_info=True)
        return get_fallback_response()


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
        
        api_key = get_openai_api_key()
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

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"{position_summary}\n\nUser question: {question}"}
        ]
        
        try:
            return await call_openai_with_retry(client, messages, max_retries=4, timeout=20.0)
        except Exception as api_error:
            logger.warning(f"Position analysis API error: {api_error}")
            return get_fallback_response()
        
    except Exception as e:
        logger.error(f"Position analysis error: {e}", exc_info=True)
        return get_fallback_response()


async def generate_daily_digest() -> Optional[str]:
    """Generate daily market digest with top opportunities"""
    try:
        import ccxt.async_support as ccxt
        from openai import OpenAI
        
        api_key = get_openai_api_key()
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

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Create today's digest based on:\n{market_data}"}
        ]
        
        try:
            return await call_openai_with_retry(client, messages, max_retries=3, timeout=20.0)
        except Exception as api_error:
            error_str = str(api_error)
            if "429" in error_str or "rate limit" in error_str.lower():
                logger.warning("Daily digest hit rate limit")
            raise api_error
        
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
        AI-generated response or friendly fallback message (never None for users)
    """
    try:
        from openai import OpenAI
        
        # Apply per-user throttling to prevent spam
        if user_id:
            should_wait, wait_time = check_user_throttle(user_id)
            if should_wait:
                await asyncio.sleep(wait_time)  # Small wait instead of rejecting
            update_user_throttle(user_id)
        
        api_key = get_openai_api_key()
        if not api_key:
            logger.error("No OpenAI API key set")
            return "I'm having trouble connecting right now. Please try again in a moment!"
        
        client = OpenAI(api_key=api_key, timeout=30.0)
        
        # Get conversation history
        conversation_history = []
        if user_id:
            conversation_history = get_conversation_history(user_id)
            # Add current question to memory
            add_to_conversation(user_id, "user", question)
        
        # Fetch coin data with timeout protection
        coin_data = []
        if coins:
            try:
                tasks = [get_coin_context(coin) for coin in coins[:3]]
                results = await asyncio.wait_for(
                    asyncio.gather(*tasks, return_exceptions=True),
                    timeout=15.0
                )
                for result in results:
                    if isinstance(result, dict) and not result.get('error'):
                        coin_data.append(result)
            except asyncio.TimeoutError:
                logger.warning("Coin data fetch timed out, continuing without")
        
        # Get market overview with fallback
        market = {}
        try:
            market = await asyncio.wait_for(get_market_overview(), timeout=10.0)
        except asyncio.TimeoutError:
            logger.warning("Market overview timed out, using defaults")
        except Exception as e:
            logger.warning(f"Market overview failed: {e}, using defaults")
        
        # Build market context - handle missing data gracefully
        if market and market.get('btc_price'):
            market_context = f"""
CURRENT MARKET CONDITIONS:
- BTC: ${market.get('btc_price', 0):,.2f} ({market.get('btc_change', 0):+.2f}% 24h) | RSI: {market.get('btc_rsi', 50):.0f} | Trend: {market.get('btc_trend', 'neutral')}
- ETH: ${market.get('eth_price', 0):,.2f} ({market.get('eth_change', 0):+.2f}% 24h)
"""
        else:
            market_context = "(Market data temporarily unavailable - answer based on general knowledge)"
        
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
3. Reference the real market data if provided, otherwise use general trading wisdom
4. Be honest about uncertainty - crypto is volatile
5. For trade recommendations, mention key levels (entry, SL, TP) when you have price data
6. Use simple language, avoid jargon
7. Include relevant emojis sparingly
8. NEVER guarantee profits - always mention risk
9. If asked about timing, be specific (now vs wait for pullback)
10. Consider BTC's trend when discussing altcoins

RESPONSE STYLE:
- Direct and conversational
- Focus on what matters NOW
- Include 1-2 specific price levels when relevant
- End with a clear takeaway or action point
- If market data is unavailable, still provide helpful general guidance"""

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
        
        logger.info(f"Calling OpenAI for question: {question[:50]}...")
        
        try:
            answer = await call_openai_with_retry(
                client=client,
                messages=messages,
                model="gpt-4o-mini",
                max_retries=4,
                timeout=30.0
            )
        except Exception as api_error:
            error_str = str(api_error)
            if "429" in error_str or "rate limit" in error_str.lower():
                logger.warning(f"OpenAI rate limit persisted after retries: {api_error}")
                return get_fallback_response()
            raise api_error
        
        if not answer:
            logger.error("OpenAI returned empty response")
            return get_fallback_response()
        
        # Save assistant response to memory
        if user_id:
            add_to_conversation(user_id, "assistant", answer)
        
        logger.info(f"AI response generated: {len(answer)} chars")
        return answer
        
    except Exception as e:
        logger.error(f"AI assistant error: {e}", exc_info=True)
        # Always return a friendly message - never show technical errors to users
        return get_fallback_response()
