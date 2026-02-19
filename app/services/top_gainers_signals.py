"""
Top Gainers Trading Mode - Generates signals from Bitunix top movers
Focuses on momentum plays with 5x leverage and 15% TP/SL
Includes 48h watchlist to catch delayed reversals
"""
import asyncio
import logging
import httpx
import os
import json
import random
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from tenacity import retry, stop_after_attempt, wait_exponential_jitter, retry_if_exception_type

logger = logging.getLogger(__name__)


def get_openai_api_key() -> Optional[str]:
    """Get OpenAI API key from environment - checks both Railway and Replit sources."""
    railway_key = os.environ.get("OPENAI_API_KEY")
    replit_key = os.environ.get("AI_INTEGRATIONS_OPENAI_API_KEY")
    
    key = railway_key or replit_key
    
    if key and "DUMMY" in key.upper():
        return None
    
    return key


_ai_rejection_cache: Dict[str, datetime] = {}
AI_REJECTION_COOLDOWN_MINUTES = 15

# Global dump mode state
_dump_mode_cache: Dict[str, any] = {
    'is_dump': False,
    'btc_change': 0,
    'btc_rsi': 50,
    'last_check': None
}
DUMP_MODE_CACHE_TTL = 60  # Check every 60 seconds


def is_coin_in_ai_cooldown(symbol: str, signal_type: str) -> bool:
    """Check if a coin was recently rejected by AI and is still in cooldown."""
    cache_key = f"{symbol}_{signal_type}"
    if cache_key in _ai_rejection_cache:
        rejected_at = _ai_rejection_cache[cache_key]
        if datetime.now() - rejected_at < timedelta(minutes=AI_REJECTION_COOLDOWN_MINUTES):
            return True
        else:
            del _ai_rejection_cache[cache_key]
    return False


def add_to_ai_rejection_cache(symbol: str, signal_type: str):
    """Add a coin to the AI rejection cache."""
    cache_key = f"{symbol}_{signal_type}"
    _ai_rejection_cache[cache_key] = datetime.now()
    logger.info(f"📝 Added {symbol} ({signal_type}) to AI rejection cache for {AI_REJECTION_COOLDOWN_MINUTES}min")


async def check_dump_mode() -> Dict:
    """
    🔴 DUMP MODE DETECTOR
    Detects when BTC is dumping hard to relax SHORT filters.
    
    Triggers when:
    - BTC 24h change ≤ -2% OR
    - BTC RSI < 40 (oversold territory)
    
    Returns cached result for 60 seconds to avoid API spam.
    """
    global _dump_mode_cache
    
    now = datetime.now()
    
    # Return cached result if fresh
    if _dump_mode_cache['last_check']:
        age = (now - _dump_mode_cache['last_check']).total_seconds()
        if age < DUMP_MODE_CACHE_TTL:
            return _dump_mode_cache
    
    try:
        import httpx
        
        async with httpx.AsyncClient() as client:
            # Get BTC 24h change
            btc_url = "https://fapi.binance.com/fapi/v1/ticker/24hr?symbol=BTCUSDT"
            resp = await client.get(btc_url, timeout=5)
            
            if resp.status_code == 200:
                data = resp.json()
                btc_change = float(data.get('priceChangePercent', 0))
            else:
                btc_change = 0
            
            # Get BTC RSI from recent candles
            klines_url = "https://fapi.binance.com/fapi/v1/klines?symbol=BTCUSDT&interval=15m&limit=20"
            resp = await client.get(klines_url, timeout=5)
            
            if resp.status_code == 200:
                klines = resp.json()
                closes = [float(k[4]) for k in klines]
                
                # Calculate RSI
                if len(closes) >= 14:
                    deltas = [closes[i] - closes[i-1] for i in range(1, len(closes))]
                    gains = [d if d > 0 else 0 for d in deltas]
                    losses = [-d if d < 0 else 0 for d in deltas]
                    avg_gain = sum(gains[-14:]) / 14
                    avg_loss = sum(losses[-14:]) / 14
                    if avg_loss > 0:
                        rs = avg_gain / avg_loss
                        btc_rsi = 100 - (100 / (1 + rs))
                    else:
                        btc_rsi = 100
                else:
                    btc_rsi = 50
            else:
                btc_rsi = 50
        
        # Determine dump mode
        is_dump = btc_change <= -2.0 or btc_rsi < 40
        
        _dump_mode_cache = {
            'is_dump': is_dump,
            'btc_change': btc_change,
            'btc_rsi': btc_rsi,
            'last_check': now
        }
        
        if is_dump:
            logger.info(f"🔴 DUMP MODE ACTIVE | BTC: {btc_change:+.1f}% | RSI: {btc_rsi:.0f}")
        
        return _dump_mode_cache
        
    except Exception as e:
        logger.warning(f"Dump mode check failed: {e}")
        _dump_mode_cache['last_check'] = now
        return _dump_mode_cache


_market_regime_cache = {
    'regime': 'NEUTRAL',  # BULLISH, BEARISH, NEUTRAL, EXTREME_BULLISH, EXTREME_BEARISH
    'focus': 'BOTH',  # LONGS, SHORTS, BOTH
    'disable_longs': False,
    'disable_shorts': False,
    'btc_change': 0,
    'btc_rsi': 50,
    'btc_ema_bullish': True,
    'reasoning': '',
    'last_check': None
}
MARKET_REGIME_CACHE_TTL = 120  # Check every 2 minutes


async def detect_market_regime() -> Dict:
    """
    🎯 AUTOMATIC MARKET REGIME DETECTOR
    
    Analyzes BTC to determine if market favors LONGS or SHORTS.
    
    EXTREME BEARISH (LONGS OFF):
    - BTC 24h change ≤ -3%
    - BTC RSI ≤ 35
    - BTC EMA9 < EMA21
    (2+ extreme signs = disable longs)
    
    EXTREME BULLISH (SHORTS OFF):
    - BTC 24h change ≥ +3%
    - BTC RSI ≥ 65
    - BTC EMA9 > EMA21
    (2+ extreme signs = disable shorts)
    
    BULLISH (Focus on LONGS):
    - BTC 24h change > +1%
    - BTC RSI > 55
    - BTC EMA9 > EMA21
    
    BEARISH (Focus on SHORTS):
    - BTC 24h change < -1%
    - BTC RSI < 45
    - BTC EMA9 < EMA21
    
    NEUTRAL (Both active, no priority change):
    - Mixed signals
    
    Returns cached result for 2 minutes.
    """
    global _market_regime_cache
    
    now = datetime.now()
    
    if _market_regime_cache['last_check']:
        age = (now - _market_regime_cache['last_check']).total_seconds()
        if age < MARKET_REGIME_CACHE_TTL:
            return _market_regime_cache
    
    try:
        import httpx
        
        async with httpx.AsyncClient() as client:
            btc_url = "https://fapi.binance.com/fapi/v1/ticker/24hr?symbol=BTCUSDT"
            resp = await client.get(btc_url, timeout=5)
            
            if resp.status_code == 200:
                data = resp.json()
                btc_change = float(data.get('priceChangePercent', 0))
            else:
                btc_change = 0
            
            klines_url = "https://fapi.binance.com/fapi/v1/klines?symbol=BTCUSDT&interval=15m&limit=30"
            resp = await client.get(klines_url, timeout=5)
            
            btc_rsi = 50
            btc_ema_bullish = True
            
            if resp.status_code == 200:
                klines = resp.json()
                closes = [float(k[4]) for k in klines]
                
                if len(closes) >= 21:
                    ema9 = sum(closes[-9:]) / 9
                    ema21 = sum(closes[-21:]) / 21
                    btc_ema_bullish = ema9 > ema21
                
                if len(closes) >= 14:
                    deltas = [closes[i] - closes[i-1] for i in range(1, len(closes))]
                    gains = [d if d > 0 else 0 for d in deltas]
                    losses = [-d if d < 0 else 0 for d in deltas]
                    avg_gain = sum(gains[-14:]) / 14
                    avg_loss = sum(losses[-14:]) / 14
                    if avg_loss > 0:
                        rs = avg_gain / avg_loss
                        btc_rsi = 100 - (100 / (1 + rs))
                    else:
                        btc_rsi = 100
        
        # EXTREME requires the price move + 1 other sign (stricter)
        # Must have the 3% move as a hard requirement
        extreme_bullish = btc_change >= 3.0 and (btc_rsi >= 65 or btc_ema_bullish)
        extreme_bearish = btc_change <= -3.0 and (btc_rsi <= 35 or not btc_ema_bullish)
        
        bullish_signs = sum([
            btc_change > 1.0,
            btc_rsi > 55,
            btc_ema_bullish
        ])
        
        bearish_signs = sum([
            btc_change < -1.0,
            btc_rsi < 45,
            not btc_ema_bullish
        ])
        
        disable_longs = False
        disable_shorts = False
        
        if extreme_bearish:
            regime = 'EXTREME_BEARISH'
            focus = 'SHORTS'
            disable_longs = True
            reasoning = f"🔴 SUPER BEARISH: {btc_change:+.1f}% | RSI {btc_rsi:.0f} | EMA {'↗' if btc_ema_bullish else '↘'} | LONGS OFF"
        elif extreme_bullish:
            regime = 'EXTREME_BULLISH'
            focus = 'LONGS'
            disable_shorts = True
            reasoning = f"🟢 SUPER BULLISH: {btc_change:+.1f}% | RSI {btc_rsi:.0f} | EMA {'↗' if btc_ema_bullish else '↘'} | SHORTS OFF"
        elif bearish_signs >= 2:
            regime = 'BEARISH'
            focus = 'SHORTS'
            reasoning = f"BTC bearish: {btc_change:+.1f}% | RSI {btc_rsi:.0f} | EMA {'↗' if btc_ema_bullish else '↘'}"
        elif bullish_signs >= 2:
            regime = 'BULLISH'
            focus = 'LONGS'
            reasoning = f"BTC bullish: {btc_change:+.1f}% | RSI {btc_rsi:.0f} | EMA {'↗' if btc_ema_bullish else '↘'}"
        else:
            regime = 'NEUTRAL'
            focus = 'BOTH'
            reasoning = f"BTC mixed: {btc_change:+.1f}% | RSI {btc_rsi:.0f} | EMA {'↗' if btc_ema_bullish else '↘'}"
        
        _market_regime_cache = {
            'regime': regime,
            'focus': focus,
            'disable_longs': disable_longs,
            'disable_shorts': disable_shorts,
            'btc_change': btc_change,
            'btc_rsi': btc_rsi,
            'btc_ema_bullish': btc_ema_bullish,
            'reasoning': reasoning,
            'last_check': now
        }
        
        if disable_longs:
            logger.warning(f"🔴 EXTREME BEARISH: LONGS DISABLED | {reasoning}")
        elif disable_shorts:
            logger.warning(f"🟢 EXTREME BULLISH: SHORTS DISABLED | {reasoning}")
        else:
            logger.info(f"🎯 MARKET REGIME: {regime} | Focus: {focus} | {reasoning}")
        
        return _market_regime_cache
        
    except Exception as e:
        logger.warning(f"Market regime check failed: {e}")
        _market_regime_cache['last_check'] = now
        return _market_regime_cache


def clean_json_response(response_text: str) -> str:
    """Clean JSON response from AI - handles markdown code blocks, thinking, and truncation."""
    import re
    
    if not response_text:
        return "{}"
    
    text = response_text.strip()
    
    # Log raw response for debugging (first 200 chars)
    logger.debug(f"Raw AI response (first 200 chars): {text[:200]}")
    
    # Remove markdown code blocks
    if "```json" in text:
        match = re.search(r'```json\s*(.*?)\s*```', text, re.DOTALL)
        if match:
            text = match.group(1)
    elif "```" in text:
        match = re.search(r'```\s*(.*?)\s*```', text, re.DOTALL)
        if match:
            text = match.group(1)
    
    text = text.strip()
    
    # Find JSON object - look for { ... }
    first_brace = text.find("{")
    last_brace = text.rfind("}")
    
    if first_brace >= 0 and last_brace > first_brace:
        text = text[first_brace:last_brace + 1]
    elif first_brace >= 0:
        # No closing brace - try to repair truncated JSON
        text = text[first_brace:]
        text = _repair_truncated_json(text)
    
    # If still no valid JSON structure, return empty
    if not text.startswith("{"):
        logger.warning(f"Could not extract JSON from response: {response_text[:100]}...")
        return "{}"
    
    # Attempt parse and repair if needed
    try:
        json.loads(text)
        return text
    except json.JSONDecodeError as e:
        logger.debug(f"JSON parse failed, attempting repair: {e}")
        repaired = _repair_truncated_json(text)
        try:
            json.loads(repaired)
            return repaired
        except:
            logger.warning(f"Could not repair JSON: {text[:100]}...")
            return "{}"


def _repair_truncated_json(text: str) -> str:
    """Repair truncated JSON by fixing unterminated strings and missing braces."""
    if not text:
        return "{}"
    
    # Check for unterminated strings by counting quotes
    quote_count = 0
    in_string = False
    escaped = False
    
    for i, char in enumerate(text):
        if escaped:
            escaped = False
            continue
        if char == '\\':
            escaped = True
            continue
        if char == '"':
            in_string = not in_string
            quote_count += 1
    
    # If odd number of quotes, we have an unterminated string
    if in_string:
        # Find the last quote and truncate cleanly after it, or close the string
        # Try to find a reasonable truncation point
        last_quote = text.rfind('"')
        if last_quote > 0:
            # Check if this is the start of a string value
            colon_pos = text.rfind(':', 0, last_quote)
            if colon_pos > 0:
                # Truncate to before this incomplete key-value pair
                # Find the comma or brace before the colon
                prev_delimiter = max(text.rfind(',', 0, colon_pos), text.rfind('{', 0, colon_pos))
                if prev_delimiter > 0:
                    text = text[:prev_delimiter + 1]
                else:
                    # Just close the string with a placeholder
                    text = text + '..."'
            else:
                text = text + '"'
    
    # Count and fix brace imbalance
    open_braces = text.count("{") - text.count("}")
    open_brackets = text.count("[") - text.count("]")
    
    # Remove trailing commas before closing
    text = text.rstrip()
    if text.endswith(','):
        text = text[:-1]
    
    # Add missing closers
    if open_brackets > 0:
        text = text + "]" * open_brackets
    if open_braces > 0:
        text = text + "}" * open_braces
    
    return text


def get_claude_client():
    """Get Claude client - checks Replit AI Integrations first, then standalone ANTHROPIC_API_KEY."""
    try:
        import anthropic
        
        # Check for Replit AI Integrations first
        base_url = os.environ.get("AI_INTEGRATIONS_ANTHROPIC_BASE_URL")
        api_key = os.environ.get("AI_INTEGRATIONS_ANTHROPIC_API_KEY")
        
        # Fall back to standalone Anthropic API key (for Railway)
        if not api_key:
            api_key = os.environ.get("ANTHROPIC_API_KEY")
            base_url = None  # Use default Anthropic endpoint
        
        if not api_key:
            logger.warning("🔑 No Anthropic API key found (checked AI_INTEGRATIONS and ANTHROPIC_API_KEY)")
            return None
        
        # Configure with base_url if provided (Replit proxy)
        if base_url:
            client = anthropic.Anthropic(base_url=base_url, api_key=api_key)
        else:
            client = anthropic.Anthropic(api_key=api_key)
            
        logger.debug("✅ Claude client initialized")
        return client
    except Exception as e:
        logger.error(f"Failed to create Claude client: {e}")
        return None


def get_gemini_client():
    """Get Gemini client - checks Replit AI Integrations first, then standalone GEMINI_API_KEY."""
    try:
        from google import genai
        
        # Check Replit AI Integrations first, then standalone key (for Railway)
        api_key = os.environ.get("AI_INTEGRATIONS_GEMINI_API_KEY") or os.environ.get("GEMINI_API_KEY")
        base_url = os.environ.get("AI_INTEGRATIONS_GEMINI_BASE_URL")
        
        if not api_key or "DUMMY" in api_key.upper():
            logger.warning("🔑 No Gemini API key found (checked AI_INTEGRATIONS_GEMINI_API_KEY and GEMINI_API_KEY)")
            return None
        
        # Configure with base_url if provided (Replit proxy)
        if base_url:
            client = genai.Client(api_key=api_key, http_options={"api_endpoint": base_url})
        else:
            client = genai.Client(api_key=api_key)
        
        logger.debug("✅ Gemini client initialized")
        return client
    except Exception as e:
        logger.error(f"Failed to create Gemini client: {e}")
        return None


async def call_gemini_signal(prompt: str, feature: str = "signal") -> Optional[str]:
    """Call Gemini API for signal generation with rate limiting.
    
    HYBRID APPROACH: Uses Gemini for initial scanning (cheap/free).
    Claude is reserved for final signal approval in ai_signal_filter.py.
    """
    from app.services.openai_limiter import get_rate_limiter
    
    limiter = await get_rate_limiter()
    allowed = await limiter.acquire(feature)
    
    # If rate limit blocked us (max signal calls reached), skip
    if not allowed:
        logger.info(f"🚫 Rate limiter blocked {feature} - max calls per cycle reached")
        return None
    
    try:
        # Use Gemini for initial scanning (cost-effective)
        gemini_client = get_gemini_client()
        if not gemini_client:
            logger.warning("Gemini client not available")
            return None
        
        def _gemini_call():
            response = gemini_client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
                config={
                    "temperature": 0.3,
                    "max_output_tokens": 1024,
                    "response_mime_type": "application/json",
                    "thinking_config": {"thinking_budget": 0}
                }
            )
            return response.text
        
        return await asyncio.to_thread(_gemini_call)
    except Exception as e:
        logger.warning(f"Gemini call failed: {e}")
        limiter.record_rate_limit()
        return None
    finally:
        limiter.release()


async def call_openai_signal_with_retry(client, messages, max_retries=4, timeout=25.0, response_format=None, use_premium=False, feature="signal"):
    """Call OpenAI API with global rate limiting and exponential backoff retry.
    
    DEPRECATED: Prefer call_gemini_signal for better rate limits.
    Falls back to this if Gemini unavailable.
    """
    import openai
    from app.services.openai_limiter import get_rate_limiter
    
    limiter = await get_rate_limiter()
    await limiter.acquire(feature)
    
    try:
        model = "gpt-4o" if use_premium else "gpt-4o-mini"
        effective_timeout = 35.0 if use_premium else timeout
        
        @retry(
            stop=stop_after_attempt(max_retries),
            wait=wait_exponential_jitter(initial=15, max=180, jitter=10),
            retry=retry_if_exception_type((openai.RateLimitError, openai.APITimeoutError, openai.APIConnectionError)),
            before_sleep=lambda retry_state: (
                limiter.record_rate_limit(),
                logger.warning(f"OpenAI retry {retry_state.attempt_number}/{max_retries} after {retry_state.outcome.exception().__class__.__name__}, waiting ~{15 * (2 ** (retry_state.attempt_number - 1))}s...")
            )[-1]
        )
        def _sync_call():
            kwargs = {
                "model": model,
                "messages": messages,
                "max_tokens": 400 if use_premium else 300,
                "temperature": 0.3,
                "timeout": effective_timeout
            }
            if response_format:
                kwargs["response_format"] = response_format
            
            response = client.with_options(max_retries=0).chat.completions.create(**kwargs)
            return response.choices[0].message.content
        
        return await asyncio.to_thread(_sync_call)
    finally:
        limiter.release()


async def enhance_signal_with_ai(signal_data: Dict) -> Dict:
    """
    Use AI to optimize signal levels (entry, SL, TP) based on market analysis.
    Returns enhanced signal data or original if AI fails.
    """
    try:
        from openai import OpenAI
        
        api_key = get_openai_api_key()
        if not api_key:
            logger.debug("No AI API key - using original signal levels")
            return signal_data
        
        client = OpenAI(api_key=api_key)
        
        symbol = signal_data.get('symbol', 'UNKNOWN')
        direction = signal_data.get('direction', 'LONG')
        entry_price = signal_data.get('entry_price', 0)
        stop_loss = signal_data.get('stop_loss', 0)
        take_profit_1 = signal_data.get('take_profit_1', 0)
        change_24h = signal_data.get('24h_change', 0)
        volume_24h = signal_data.get('24h_volume', 0)
        reasoning = signal_data.get('reasoning', '')
        leverage = signal_data.get('leverage', 20)
        
        base_symbol = symbol.replace('/USDT', '').replace('USDT', '').upper()
        is_major = base_symbol in ['BTC', 'ETH', 'SOL', 'XRP', 'BNB']
        
        # Calculate current SL/TP percentages
        if direction == 'LONG':
            sl_pct = ((entry_price - stop_loss) / entry_price) * 100 if entry_price > 0 else 0
            tp_pct = ((take_profit_1 - entry_price) / entry_price) * 100 if entry_price > 0 else 0
        else:
            sl_pct = ((stop_loss - entry_price) / entry_price) * 100 if entry_price > 0 else 0
            tp_pct = ((entry_price - take_profit_1) / entry_price) * 100 if entry_price > 0 else 0
        
        prompt = f"""You are an expert crypto trader. Review and optimize this trade signal.

CURRENT SIGNAL:
- Symbol: {symbol} ({"Major coin" if is_major else "Altcoin"})
- Direction: {direction}
- Entry: ${entry_price:.6f}
- Stop Loss: ${stop_loss:.6f} ({sl_pct:.2f}%)
- Take Profit: ${take_profit_1:.6f} ({tp_pct:.2f}%)
- Leverage: {leverage}x
- 24h Change: {change_24h:+.1f}%
- 24h Volume: ${volume_24h:,.0f}
- Signal Reasoning: {reasoning}

RULES:
1. For {direction}s at {leverage}x leverage, optimize SL/TP for realistic targets
2. LONGS: Should capture 67% profit (3.35% move) with 65% max loss (3.25% move) at 20x
3. SHORTS: Mean reversion targets, capped at 150% max profit / 80% max loss
4. Consider if the entry timing is optimal or if we should wait
5. R:R should be at least 1:1

Should we OPTIMIZE the levels or KEEP them as-is?

Respond in JSON:
{{
    "action": "OPTIMIZE" or "KEEP",
    "optimized_entry": {entry_price},
    "optimized_sl": <price or {stop_loss}>,
    "optimized_tp": <price or {take_profit_1}>,
    "sl_pct": <percentage>,
    "tp_pct": <percentage>,
    "reasoning": "Brief explanation of optimization or why levels are good",
    "confidence_boost": -2 to +2 (adjust signal confidence)
}}"""

        messages = [
            {"role": "system", "content": "You are an expert crypto trader. Optimize trade levels for maximum profit with controlled risk. Always respond with valid JSON."},
            {"role": "user", "content": prompt}
        ]
        
        # Use retry helper to handle rate limits
        response_content = await call_openai_signal_with_retry(
            client,
            messages,
            max_retries=4,
            timeout=15.0,
            response_format={"type": "json_object"}
        )
        
        cleaned_json = clean_json_response(response_content)
        result = json.loads(cleaned_json)
        
        if result.get('action') == 'OPTIMIZE':
            # Apply AI optimizations
            if result.get('optimized_sl'):
                signal_data['stop_loss'] = result['optimized_sl']
            if result.get('optimized_tp'):
                signal_data['take_profit_1'] = result['optimized_tp']
            if result.get('optimized_entry'):
                signal_data['entry_price'] = result['optimized_entry']
            
            # Boost/reduce confidence
            boost = result.get('confidence_boost', 0)
            current_conf = signal_data.get('confidence', 70)
            signal_data['confidence'] = min(100, max(50, current_conf + (boost * 10)))
            
            # Add AI reasoning to signal
            ai_reason = result.get('reasoning', '')
            if ai_reason:
                signal_data['ai_enhancement'] = ai_reason
            
            logger.info(f"🤖 AI optimized {symbol} {direction}: {result.get('reasoning', 'No reason')}")
        else:
            signal_data['ai_enhancement'] = result.get('reasoning', 'Levels confirmed by AI')
            logger.info(f"🤖 AI confirmed {symbol} {direction} levels are good")
        
        signal_data['ai_enhanced'] = True
        return signal_data
        
    except Exception as e:
        logger.warning(f"AI signal enhancement failed: {e}")
        return signal_data


async def ai_validate_long_signal(coin_data: Dict, candle_data: Dict) -> Optional[Dict]:
    """
    🤖 AI-POWERED LONG SIGNAL VALIDATION (Using Gemini)
    
    Uses Gemini 2.5 Flash for analysis with rate limiting.
    Focus: High win-rate entries with optimal timing.
    """
    try:
        symbol = coin_data.get('symbol', 'UNKNOWN')
        
        if is_coin_in_ai_cooldown(symbol, "LONG"):
            logger.info(f"⏳ {symbol} LONG in AI rejection cooldown - skipping")
            return {'approved': False, 'reasoning': 'In cooldown from recent AI rejection', 'symbol': symbol}
        
        change_24h = coin_data.get('change_24h', 0)
        volume_24h = coin_data.get('volume_24h', 0)
        current_price = coin_data.get('price', 0)
        
        rsi = candle_data.get('rsi', 50)
        ema9 = candle_data.get('ema9', 0)
        ema21 = candle_data.get('ema21', 0)
        volume_ratio = candle_data.get('volume_ratio', 1)
        trend_5m = candle_data.get('trend_5m', 'neutral')
        trend_15m = candle_data.get('trend_15m', 'neutral')
        funding_rate = candle_data.get('funding_rate', 0)
        price_to_ema9 = candle_data.get('price_to_ema9', 0)
        recent_high = candle_data.get('recent_high', 0)
        recent_low = candle_data.get('recent_low', 0)
        last_3_candles = candle_data.get('last_3_candles', 'unknown')
        btc_change = candle_data.get('btc_change', 0)
        
        price_range_pct = ((recent_high - recent_low) / recent_low * 100) if recent_low > 0 else 5.0
        
        sr_info = ""
        sr_data = candle_data.get('support_resistance', {})
        if sr_data:
            sr_lines = []
            if sr_data.get('nearest_support'):
                sr_lines.append(f"• Nearest Support: ${sr_data['nearest_support']:.6f} ({sr_data.get('support_distance_pct', 0):.2f}% below)")
            if sr_data.get('nearest_resistance'):
                sr_lines.append(f"• Nearest Resistance: ${sr_data['nearest_resistance']:.6f} ({sr_data.get('resistance_distance_pct', 0):.2f}% above)")
            supports = sr_data.get('supports', [])
            if len(supports) > 1:
                sr_lines.append(f"• Support Levels: {', '.join([f'${s:.6f}' for s in supports[:3]])}")
            resistances = sr_data.get('resistances', [])
            if len(resistances) > 1:
                sr_lines.append(f"• Resistance Levels: {', '.join([f'${r:.6f}' for r in resistances[:3]])}")
            if sr_lines:
                sr_info = "\n\nCHART KEY LEVELS:\n" + "\n".join(sr_lines)

        prompt = f"""You are a crypto futures trading analyst. Evaluate this LONG setup.

📊 {symbol} @ ${current_price:.6f}

PRICE: 24h {change_24h:+.1f}% | Range ${recent_low:.6f}-${recent_high:.6f} | Vol {price_range_pct:.1f}%
TREND: EMA9 ${ema9:.6f} ({price_to_ema9:+.1f}%) | EMA21 ${ema21:.6f} | 5m={trend_5m} 15m={trend_15m}
MOMENTUM: RSI {rsi:.0f} | Volume {volume_ratio:.1f}x | Funding {funding_rate:+.4f}%
CONTEXT: BTC {btc_change:+.1f}% | Last 3: {last_3_candles}
{sr_info}

CRITERIA:
✅ APPROVE: RSI 35-65, at least one timeframe bullish, volume 1.2x+, momentum visible
❌ REJECT: RSI extreme (<30 or >70), both TFs bearish, weak volume, clear downtrend

CRITICAL TP/SL RULES:
- Set TP at or near resistance levels from the chart (where price is likely to face selling)
- Set SL just below nearest support (where breakdown would invalidate the long)
- Use chart levels as anchors for TP/SL, not arbitrary percentages
- Be reasonable - approve solid setups

Respond JSON only:
{{"action": "LONG" or "SKIP", "confidence": 5-10, "reasoning": "one sentence including chart level targets", "entry_quality": "A+" or "A" or "B" or "C", "tp_percent": 3.0-6.0, "sl_percent": 2.0-4.0, "risk_reward": number}}"""

        response_content = await call_gemini_signal(prompt, feature="long_validation")
        
        if not response_content:
            logger.warning(f"No AI response for {symbol} LONG - auto-approving with defaults")
            return {
                'approved': True, 'recommendation': 'BUY', 'confidence': 7,
                'reasoning': 'AI unavailable - approved based on TA', 'entry_quality': 'B', 'symbol': symbol,
                'entry_price': current_price, 'stop_loss': current_price * 0.97, 'take_profit': current_price * 1.04,
                'tp_percent': 4.0, 'sl_percent': 3.0, 'risk_reward': 1.33, 'leverage': 20
            }
        
        cleaned_json = clean_json_response(response_content)
        result = json.loads(cleaned_json)
        
        action = result.get('action', 'SKIP')
        confidence = result.get('confidence', 0)
        reasoning = result.get('reasoning', 'No analysis available')
        entry_quality = result.get('entry_quality', 'C')
        tp_percent = result.get('tp_percent', 3.35)
        sl_percent = result.get('sl_percent', 3.25)
        risk_reward = result.get('risk_reward', 1.0)
        
        # Set recommendation based on AI action
        if action == 'LONG':
            if confidence >= 9:
                recommendation = 'STRONG BUY'
            elif confidence >= 8:
                recommendation = 'BUY'
            else:
                recommendation = 'WEAK BUY'
        else:
            recommendation = 'SKIP'
        
        # Only approve if AI said LONG with good quality (minimum 8/10 confidence)
        approved = action == 'LONG' and entry_quality in ['A+', 'A', 'B'] and confidence >= 8
        
        logger.info(f"🤖 AI LONGS: {symbol} → {action} ({confidence}/10) [{entry_quality}] | R:R {risk_reward:.1f}")
        
        if not approved:
            add_to_ai_rejection_cache(symbol, "LONG")
            return {'approved': False, 'recommendation': recommendation, 'confidence': confidence, 'reasoning': reasoning, 'symbol': symbol}
        
        tp_percent = max(2.0, min(6.0, tp_percent))
        sl_percent = max(1.5, min(4.0, sl_percent))
        
        take_profit = current_price * (1 + tp_percent / 100)
        stop_loss = current_price * (1 - sl_percent / 100)
        
        logger.info(f"✅ AI APPROVED LONG: {symbol} | TP +{tp_percent:.1f}% | SL -{sl_percent:.1f}%")
        
        return {
            'approved': True, 'recommendation': recommendation, 'confidence': confidence,
            'reasoning': reasoning, 'entry_quality': entry_quality, 'symbol': symbol,
            'entry_price': current_price, 'stop_loss': stop_loss, 'take_profit': take_profit,
            'tp_percent': tp_percent, 'sl_percent': sl_percent, 'risk_reward': risk_reward, 'leverage': 20
        }
        
    except Exception as e:
        logger.error(f"AI LONG validation error for {coin_data.get('symbol', 'unknown')}: {e}")
        return None


async def ai_validate_scalp_signal(scalp_data: Dict) -> Optional[Dict]:
    """
    🤖 AI-POWERED SCALP VALIDATION (Using Gemini)
    Focus: High probability 0.8-1.5% moves.
    """
    try:
        symbol = scalp_data.get('symbol', 'UNKNOWN')
        current_price = scalp_data.get('price', 0)
        vwap = scalp_data.get('vwap', 0)
        rsi = scalp_data.get('rsi', 0)
        
        prompt = f"""You are a Scalp Trading Specialist. Evaluate this VWAP BOUNCE setup for a 1% target.
        
        SYMBOL: {symbol}
        PRICE: ${current_price:.6f}
        VWAP: ${vwap:.6f} (Dist: {scalp_data.get('dist_vwap', 0):.2f}%)
        RSI: {rsi:.1f}
        TREND (1H): {scalp_data.get('trend_1h')}
        
        STRATEGY: Entry at VWAP touch in strong 1H uptrend. 
        GOAL: 0.8% - 1.5% quick bounce.
        
        Respond JSON only:
        {{"action": "LONG" or "SKIP", "confidence": 1-10, "reasoning": "brief", "tp_percent": 1.2, "sl_percent": 0.5, "entry_quality": "A" or "B" or "C"}}"""
        
        response_content = await call_gemini_signal(prompt, feature="scalp_validation")
        if not response_content:
            return None
            
        cleaned_json = clean_json_response(response_content)
        result = json.loads(cleaned_json)
        
        if result.get('action') == 'LONG' and result.get('confidence', 0) >= 7:
            tp_pct = result.get('tp_percent', 1.2)
            sl_pct = result.get('sl_percent', 0.5)
            
            return {
                'approved': True,
                'symbol': symbol,
                'entry_price': current_price,
                'take_profit': current_price * (1 + tp_pct / 100),
                'stop_loss': current_price * (1 - sl_pct / 100),
                'tp_percent': tp_pct,
                'sl_percent': sl_pct,
                'leverage': 20,
                'confidence': result.get('confidence'),
                'reasoning': result.get('reasoning'),
                'signal_type': 'VWAP_SCALP'
            }
        return None
    except Exception as e:
        logger.error(f"Scalp AI validation error: {e}")
        return None


async def ai_validate_short_signal(coin_data: Dict, candle_data: Dict) -> Optional[Dict]:
    """
    🤖 AI-POWERED SHORT SIGNAL VALIDATION (Using Gemini)
    
    Uses Gemini 2.5 Flash for analysis - higher rate limits than OpenAI.
    Focus: Catching overextended moves ready to reverse.
    
    Args:
        coin_data: {symbol, change_24h, volume_24h, price}
        candle_data: {rsi, ema9, price_to_ema9, wick_size, is_bearish, volume_ratio, etc}
    
    Returns:
        Dict with AI decision including dynamic TP/SL or None if rejected
    """
    try:
        symbol = coin_data.get('symbol', 'UNKNOWN')
        signal_type = coin_data.get('signal_type', 'SHORT')
        
        if is_coin_in_ai_cooldown(symbol, signal_type):
            logger.info(f"⏳ {symbol} {signal_type} in AI rejection cooldown - skipping")
            return {'approved': False, 'reasoning': 'In cooldown from recent AI rejection', 'symbol': symbol}
        
        change_24h = coin_data.get('change_24h', 0)
        volume_24h = coin_data.get('volume_24h', 0)
        current_price = coin_data.get('price', 0)
        
        rsi = candle_data.get('rsi', 50)
        ema9 = candle_data.get('ema9', 0)
        price_to_ema9 = candle_data.get('price_to_ema9', 0)
        volume_ratio = candle_data.get('volume_ratio', 1)
        wick_size = candle_data.get('wick_size', 0)
        is_bearish = candle_data.get('is_bearish', False)
        recent_high = candle_data.get('recent_high', 0)
        recent_low = candle_data.get('recent_low', 0)
        btc_change = candle_data.get('btc_change', 0)
        exhaustion_count = candle_data.get('exhaustion_count', 0)
        slowing_momentum = candle_data.get('slowing_momentum', False)
        
        # Calculate volatility and overextension
        price_range_pct = ((recent_high - recent_low) / recent_low * 100) if recent_low > 0 else 5.0
        
        sr_info = ""
        sr_data = candle_data.get('support_resistance', {})
        if sr_data:
            sr_lines = []
            if sr_data.get('nearest_support'):
                sr_lines.append(f"• Nearest Support: ${sr_data['nearest_support']:.6f} ({sr_data.get('support_distance_pct', 0):.2f}% below)")
            if sr_data.get('nearest_resistance'):
                sr_lines.append(f"• Nearest Resistance: ${sr_data['nearest_resistance']:.6f} ({sr_data.get('resistance_distance_pct', 0):.2f}% above)")
            supports = sr_data.get('supports', [])
            if len(supports) > 1:
                sr_lines.append(f"• Support Levels: {', '.join([f'${s:.6f}' for s in supports[:3]])}")
            resistances = sr_data.get('resistances', [])
            if len(resistances) > 1:
                sr_lines.append(f"• Resistance Levels: {', '.join([f'${r:.6f}' for r in resistances[:3]])}")
            if sr_data.get('recent_high'):
                sr_lines.append(f"• Recent High: ${sr_data['recent_high']:.6f} | Recent Low: ${sr_data['recent_low']:.6f}")
            if sr_lines:
                sr_info = "\n\nCHART KEY LEVELS (use these for optimal TP/SL):\n" + "\n".join(sr_lines)

        prompt = f"""You are a crypto futures trader. Your job: Find SHORT opportunities on coins showing weakness.

═══════════════════════════════════════════════════════════
📊 {symbol} @ ${current_price:.6f} | 24h: +{change_24h:.1f}%
═══════════════════════════════════════════════════════════

CURRENT STATE:
• 24h Gain: +{change_24h:.1f}%
• Price vs EMA9: {price_to_ema9:+.1f}%
• RSI: {rsi:.0f}/100
• Upper Wick: {wick_size:.1f}%
• Candle: {"🔴 Bearish" if is_bearish else "🟢 Bullish"}
• Momentum: {"⚠️ SLOWING" if slowing_momentum else "Strong"}
• Volume: {volume_ratio:.1f}x average
• BTC: {btc_change:+.1f}%
{sr_info}

═══════════════════════════════════════════════════════════
🎯 SHORT DECISION (20x Leverage) - BE AGGRESSIVE
═══════════════════════════════════════════════════════════

✅ APPROVE SHORT IF ANY OF THESE (just need 1-2):
• RSI > 60 (elevated, room to drop)
• Pulled back from high (not at peak)
• Red candles showing OR momentum slowing
• BTC weak/neutral (not pumping hard)
• Any sign of exhaustion or weakness

❌ ONLY REJECT IF:
• RSI < 50 (already oversold area)
• BTC pumping +3%+ (strong sympathy rally)
• Price making aggressive new highs right now

BE AGGRESSIVE - we want to catch moves early. Small gainers (+3-10%) can still dump hard.

CRITICAL TP/SL RULES:
- Set TP at or near support levels from the chart (where price is likely to bounce)
- Set SL just above nearest resistance (where breakout would invalidate the short)
- If support is 4% below entry, set TP there. If resistance is 2% above, set SL there
- Use chart levels as anchors, not arbitrary percentages

Respond JSON:
{{
    "action": "SHORT" or "SKIP",
    "confidence": 6-10,
    "reasoning": "Brief reason including which chart level TP targets",
    "entry_quality": "A+" or "A" or "B",
    "tp_percent": 3.0-6.0,
    "sl_percent": 2.5-4.0 (MAX 4%),
    "risk_reward": R:R ratio
}}

Rules:
- Default to SHORT unless clear reason not to
- tp_percent = 1.5x sl_percent minimum
- Align TP/SL with chart support/resistance levels when available
- We want trades, not perfect setups"""

        # Use Gemini for better rate limits (via Replit AI Integrations)
        full_prompt = f"""You are an AGGRESSIVE crypto trader. Your bias is to SHORT.
Default action is SHORT unless there's a clear reason not to. We want trades. Respond with valid JSON only.

{prompt}"""
        
        response_content = await call_gemini_signal(full_prompt, feature="short_validation")
        
        # Fallback to OpenAI if Gemini fails
        if not response_content:
            logger.warning("Gemini failed, trying OpenAI fallback...")
            api_key = get_openai_api_key()
            if api_key:
                from openai import OpenAI
                client = OpenAI(api_key=api_key, timeout=20.0)
                messages = [
                    {"role": "system", "content": "You are a consistently profitable crypto trader specializing in shorting parabolic pumps. You have 60%+ win rate on reversal trades. Be decisive - SHORT or SKIP. Respond with valid JSON only."},
                    {"role": "user", "content": prompt}
                ]
                response_content = await call_openai_signal_with_retry(
                    client, messages, max_retries=2, timeout=25.0,
                    response_format={"type": "json_object"}, use_premium=False
                )
        
        # If all AI fails, auto-approve based on TA
        if not response_content:
            logger.warning(f"All AI failed for {symbol} SHORT - auto-approving with defaults")
            return {
                'approved': True, 'recommendation': 'SELL', 'confidence': 7,
                'reasoning': 'AI unavailable - approved based on TA exhaustion signs', 'entry_quality': 'B', 'symbol': symbol,
                'entry_price': current_price, 'stop_loss': current_price * 1.035, 'take_profit': current_price * 0.95,
                'tp_percent': 5.0, 'sl_percent': 3.5, 'risk_reward': 1.43, 'leverage': 20
            }
        
        cleaned_json = clean_json_response(response_content)
        result = json.loads(cleaned_json)
        
        action = result.get('action', 'SKIP')
        confidence = result.get('confidence', 0)
        reasoning = result.get('reasoning', 'No analysis available')
        entry_quality = result.get('entry_quality', 'C')
        tp_percent = result.get('tp_percent', 5.0)
        sl_percent = result.get('sl_percent', 3.5)
        risk_reward = result.get('risk_reward', 1.4)
        reversal_confidence = result.get('reversal_confidence', 5)
        
        # Map to recommendation format
        if action == 'SHORT' and confidence >= 9:
            recommendation = 'STRONG SELL'
        elif action == 'SHORT' and confidence >= 8:
            recommendation = 'SELL'
        else:
            recommendation = 'SKIP'
        
        # Require minimum 8/10 confidence for all SHORT signals
        approved = action == 'SHORT' and entry_quality in ['A+', 'A', 'B'] and confidence >= 8
        
        logger.info(f"🤖 AI SHORTS: {symbol} → {action} ({confidence}/10) [{entry_quality}] | Reversal: {reversal_confidence}/10 | {reasoning[:50]}...")
        
        if not approved:
            add_to_ai_rejection_cache(symbol, signal_type)
            logger.info(f"🤖 AI REJECTED {signal_type}: {symbol} - Grade: {entry_quality}, Conf: {confidence} - {reasoning}")
            return {
                'approved': False,
                'recommendation': recommendation,
                'confidence': confidence,
                'reasoning': reasoning,
                'symbol': symbol
            }
        
        # Calculate entry levels with AI-suggested TP/SL (reversed for SHORT)
        entry_price = current_price
        
        # Clamp TP/SL to reasonable ranges for shorts
        # CRITICAL: Max 3% SL = 60% max loss at 20x leverage (our hard limit)
        tp_percent = max(3.0, min(10.0, tp_percent))
        sl_percent = max(2.0, min(3.0, sl_percent))  # Hard cap at 3%
        
        # SHORT: TP is below entry, SL is above entry
        take_profit = entry_price * (1 - tp_percent / 100)
        stop_loss = entry_price * (1 + sl_percent / 100)
        
        logger.info(f"✅ AI APPROVED SHORT: {symbol} | TP: -{tp_percent:.1f}% | SL: +{sl_percent:.1f}% | R:R {risk_reward:.1f}")
        
        return {
            'approved': True,
            'recommendation': recommendation,
            'confidence': confidence,
            'reasoning': reasoning,
            'entry_quality': entry_quality,
            'reversal_confidence': reversal_confidence,
            'symbol': symbol,
            'entry_price': entry_price,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'tp_percent': tp_percent,
            'sl_percent': sl_percent,
            'risk_reward': risk_reward,
            'leverage': 20
        }
        
    except Exception as e:
        logger.error(f"AI SHORT validation error for {coin_data.get('symbol', 'unknown')}: {e}")
        return None


# 🛑 MASTER KILL SWITCH - Set to True to disable all scanning
SCANNING_DISABLED = False  # Toggle this to enable/disable scanning - SCANNING ON

# 🔴 SHORT STRATEGY CONTROLS
# Both strategies enabled with STRICT quality filters (max 2/day total)
SHORTS_DISABLED = True  # Master switch for all shorts - DISABLED
PARABOLIC_DISABLED = True  # Disable 50%+ exhausted pump shorts
NORMAL_SHORTS_ENABLED = False  # Disable AI-powered normal shorts

# 🟢 LONG STRATEGY CONTROLS
LONGS_DISABLED = True  # Master switch for all longs - DISABLED

# 🚫 BLACKLISTED SYMBOLS - These coins will never generate signals
BLACKLISTED_SYMBOLS = ['FHE', 'FHEUSDT', 'FHE/USDT', 'BAS', 'BASUSDT', 'BAS/USDT', 'BEAT', 'BEATUSDT', 'BEAT/USDT', 'PTB', 'PTBUSDT', 'PTB/USDT', 'ICNT', 'ICNTUSDT', 'ICNT/USDT', 'TA', 'TAUSDT', 'TA/USDT', 'LIGHT', 'LIGHTUSDT', 'LIGHT/USDT', 'TRADOOR', 'TRADOORUSDT', 'TRADOOR/USDT', 'LAB', 'LABUSDT', 'LAB/USDT', 'RIVER', 'RIVERUSDT', 'RIVER/USDT', 'ARPA', 'ARPAUSDT', 'ARPA/USDT', 'ALPHA', 'ALPHAUSDT', 'ALPHA/USDT', 'NAORIS', 'NAORISUSDT', 'NAORIS/USDT', 'AGIX', 'AGIXUSDT', 'AGIX/USDT']

# Track SHORTS that lost to prevent re-shorting the same pump
# Format: {symbol: datetime_when_cooldown_expires}
shorts_cooldown = {}

# 🔄 BINANCE FAILURE TRACKING - Switch to MEXC-only mode after repeated failures
binance_failure_count = 0
binance_blocked_until = None  # datetime when to retry Binance
BINANCE_FAILURE_THRESHOLD = 3  # Switch to MEXC after 3 failures
BINANCE_BLOCK_DURATION_MINUTES = 10  # Stay on MEXC for 10 minutes

# 🚀 PARALLEL FETCHING - Limit concurrent API requests to avoid rate limits
API_SEMAPHORE = asyncio.Semaphore(5)  # Max 5 concurrent API calls
PARALLEL_BATCH_SIZE = 10  # Process symbols in batches of 10

# 🔴 SHORT COOLDOWNS - Prevent signal spam
last_short_signal_time = None
SHORT_GLOBAL_COOLDOWN_HOURS = 1  # 1 hour cooldown between SHORT signals

# 🟢 LONG COOLDOWNS - Prevent signal spam
# Global cooldown: 30 mins between ANY long signals
last_long_signal_time = None
LONG_GLOBAL_COOLDOWN_HOURS = 2  # 2 hour cooldown between LONG signals

# Per-symbol 24-HOUR cooldown: Block coin for 24h after trading
# Format: {symbol: datetime_when_traded} - tracks when the symbol was last traded
longs_symbol_cooldown = {}
LONGS_COOLDOWN_HOURS = 0  # No cooldown

def is_symbol_in_next_window_cooldown(symbol: str) -> bool:
    """Check if symbol is still in 24-hour cooldown"""
    if symbol not in longs_symbol_cooldown:
        return False
    
    last_traded = longs_symbol_cooldown[symbol]
    now = datetime.utcnow()
    cooldown_expires = last_traded + timedelta(hours=LONGS_COOLDOWN_HOURS)
    
    if now < cooldown_expires:
        hours_remaining = (cooldown_expires - now).total_seconds() / 3600
        logger.info(f"    ⏳ {symbol} in 24h cooldown ({hours_remaining:.1f}h remaining)")
        return True
    
    # Cooldown expired, clean up
    del longs_symbol_cooldown[symbol]
    return False

def add_symbol_window_cooldown(symbol: str):
    """Mark symbol as traded (starts 24h cooldown)"""
    longs_symbol_cooldown[symbol] = datetime.utcnow()
    logger.info(f"📝 Added {symbol} to 24h cooldown (expires {datetime.utcnow() + timedelta(hours=LONGS_COOLDOWN_HOURS)})")

# 🔥 BREAKOUT TRACKING CACHE - Track candidates waiting for pullback
# Format: {symbol: {'detected_at': datetime, 'breakout_data': {...}, 'checks': int}}
# Candidates are re-evaluated on each scan until pullback occurs or timeout (10 min)
pending_breakout_candidates = {}
BREAKOUT_CANDIDATE_TIMEOUT_MINUTES = 10

# 🔥 SIGNAL LIMITS - prevents over-trading
MAX_DAILY_SIGNALS = 4  # Cap at 4 signals per day for quality over quantity
MAX_DAILY_SHORTS = 2  # Cap shorts at 2 per day
daily_signal_count = 0
daily_short_count = 0
last_signal_date = None

def check_and_increment_daily_signals(direction: str = None) -> bool:
    """
    Check if we can send another signal.
    - Max 2 EXECUTED trades per 4-hour FIXED window (window starts from first trade, expires 4h later)
    - FIXED: Now counts actual Trade entries, not Signal entries (prevents false counting when AI rejects)
    Returns True if allowed, False if limit reached.
    """
    global daily_signal_count, daily_short_count, last_signal_date
    
    # 1. Check 4-hour FIXED window (starts from first EXECUTED trade in window)
    from app.database import SessionLocal
    from app.models import Trade
    from datetime import datetime, timedelta
    
    db = SessionLocal()
    try:
        now = datetime.utcnow()
        four_hours_ago = now - timedelta(hours=4)
        
        # Get the FIRST automated scanner trade within the last 4 hours (this defines window start)
        # Count UNIQUE SIGNALS (not individual user trades), exclude SCALP trades
        from sqlalchemy import func
        valid_statuses = ['open', 'closed', 'tp_hit', 'sl_hit', 'breakeven']
        first_trade_in_window = db.query(Trade).filter(
            Trade.opened_at >= four_hours_ago,
            Trade.signal_id.isnot(None),  # Must have signal_id (from scanner)
            Trade.trade_type != 'SCALP',  # Exclude scalp trades (run independently)
            Trade.status.in_(valid_statuses)  # Only successful trades
        ).order_by(Trade.opened_at.asc()).first()
        
        if first_trade_in_window is None:
            # No automated trades in last 4h, window is empty - allow new trade
            logger.info(f"✅ No automated scanner trades in last 4h - starting new window")
            return _increment_and_allow(direction)
        
        # Calculate when the current window expires (4h after first trade)
        window_start = first_trade_in_window.opened_at
        window_end = window_start + timedelta(hours=4)
        
        if now >= window_end:
            # Window has expired - allow new trade (this starts a new window)
            logger.info(f"✅ Previous window expired at {window_end.strftime('%H:%M')} - starting new window")
            return _increment_and_allow(direction)
        
        # Window is still active - count UNIQUE SIGNALS within THIS window
        # Each signal can create multiple trades (one per user), but we count unique signals
        # Scalps run independently and DON'T count toward this limit
        recent_signals_count = db.query(func.count(func.distinct(Trade.signal_id))).filter(
            Trade.opened_at >= window_start,
            Trade.opened_at < window_end,
            Trade.signal_id.isnot(None),  # Must have signal_id (from scanner)
            Trade.trade_type != 'SCALP',  # Exclude scalp trades
            Trade.status.in_(valid_statuses)  # Only successful trades
        ).scalar() or 0
        
        if recent_signals_count >= 2:
            time_remaining = window_end - now
            mins_remaining = int(time_remaining.total_seconds() / 60)
            logger.warning(f"⏳ 4-HOUR LIMIT REACHED: {recent_signals_count}/2 unique signals in window (resets in {mins_remaining} mins at {window_end.strftime('%H:%M')} UTC)")
            return False

        # Window has space - allow and log
        time_remaining = window_end - now
        mins_remaining = int(time_remaining.total_seconds() / 60)
        logger.info(f"✅ Window slot available: {recent_signals_count}/2 unique signals (window resets in {mins_remaining} mins)")
        return _increment_and_allow(direction)
        
    finally:
        db.close()


def _increment_and_allow(direction: str = None) -> bool:
    """Helper to increment daily counters and return True"""
    global daily_signal_count, daily_short_count, last_signal_date
    
    today = datetime.utcnow().date()
    
    # Reset counter if new day
    if last_signal_date != today:
        daily_signal_count = 0
        daily_short_count = 0
        last_signal_date = today
        logger.info(f"📅 New day - daily signal counters reset to 0")
    
    # Check total limit
    if daily_signal_count >= MAX_DAILY_SIGNALS:
        logger.warning(f"⚠️ DAILY LIMIT REACHED: {daily_signal_count}/{MAX_DAILY_SIGNALS} signals today")
        return False
    
    # Check SHORT limit
    if direction == 'SHORT' and daily_short_count >= MAX_DAILY_SHORTS:
        logger.warning(f"⚠️ DAILY SHORT LIMIT REACHED: {daily_short_count}/{MAX_DAILY_SHORTS} shorts today")
        return False
    
    # Increment and allow
    daily_signal_count += 1
    if direction == 'SHORT':
        daily_short_count += 1
        logger.info(f"📊 Daily signals: {daily_signal_count}/{MAX_DAILY_SIGNALS} (Shorts: {daily_short_count}/{MAX_DAILY_SHORTS})")
    else:
        logger.info(f"📊 Daily signals: {daily_signal_count}/{MAX_DAILY_SIGNALS}")
    return True

def get_daily_signal_count() -> int:
    """Get current daily signal count"""
    global daily_signal_count, last_signal_date
    today = datetime.utcnow().date()
    if last_signal_date != today:
        return 0
    return daily_signal_count

def get_daily_short_count() -> int:
    """Get current daily SHORT signal count"""
    global daily_short_count, last_signal_date
    today = datetime.utcnow().date()
    if last_signal_date != today:
        return 0
    return daily_short_count


def decrement_daily_signals(direction: str = None) -> None:
    """
    Decrement daily signal count (used when AI rejects a signal after increment).
    """
    global daily_signal_count, daily_short_count
    
    if daily_signal_count > 0:
        daily_signal_count -= 1
    if direction == 'SHORT' and daily_short_count > 0:
        daily_short_count -= 1
    logger.info(f"📉 Decremented daily counts: {daily_signal_count}/{MAX_DAILY_SIGNALS} (Shorts: {daily_short_count}/{MAX_DAILY_SHORTS})")


def calculate_leverage_capped_targets(
    entry_price: float,
    direction: str,
    tp_pcts: List[float],  # List of TP percentages [TP1, TP2, ...] or single value
    base_sl_pct: float,
    leverage: int,
    max_profit_cap: float = 150.0,
    max_loss_cap: float = 80.0
) -> Dict:
    """
    Calculate TP/SL prices with profit/loss cap based on leverage
    
    Maintains proportional spacing between multiple TPs when capping is applied.
    Scales the entire TP ladder uniformly to preserve strategy integrity.
    
    Args:
        entry_price: Entry price
        direction: 'LONG' or 'SHORT'
        tp_pcts: List of TP percentages (e.g., [5.0, 10.0] for dual TP LONGS)
        base_sl_pct: Base stop loss percentage (price move, e.g., 4%)
        leverage: User's leverage (5x, 10x, 20x, etc.)
        max_profit_cap: Maximum profit percentage allowed (default: 80%)
        max_loss_cap: Maximum loss percentage allowed (default: 80%)
    
    Returns:
        Dict with tp_prices[], sl_price, scaling_factor, tp_profit_pcts[], sl_loss_pct
    
    Examples:
        LONG with 20x leverage, TPs [5%, 10%]:
        - Max TP (10%) would be 200% profit → exceeds 150% cap
        - Scaling factor: 150% / 200% = 0.75
        - Scaled TPs: [7.5%] → profits: [150%] ✅
        - TP capped at 150% max profit
    """
    # Ensure tp_pcts is a list
    if not isinstance(tp_pcts, list):
        tp_pcts = [tp_pcts]
    
    # Find the maximum TP (furthest profit target)
    max_tp_pct = max(tp_pcts)
    max_tp_profit = max_tp_pct * leverage
    
    # Calculate scaling factor if max TP exceeds cap
    if max_tp_profit > max_profit_cap:
        scaling_factor = max_profit_cap / max_tp_profit
    else:
        scaling_factor = 1.0
    
    # Scale all TPs proportionally
    effective_tp_pcts = [tp * scaling_factor for tp in tp_pcts]
    # SL is NOT scaled - it caps directly at max_loss_cap/leverage (e.g., 80%/20x = 4% price move = 80% loss)
    effective_sl_pct = min(base_sl_pct, max_loss_cap / leverage)
    
    # Calculate actual profit/loss percentages with leverage
    tp_profit_pcts = [tp * leverage for tp in effective_tp_pcts]
    sl_loss_pct = effective_sl_pct * leverage
    
    # Calculate price targets
    tp_prices = []
    for effective_tp in effective_tp_pcts:
        if direction == 'LONG':
            tp_price = entry_price * (1 + effective_tp / 100)
        else:  # SHORT
            tp_price = entry_price * (1 - effective_tp / 100)
        tp_prices.append(tp_price)
    
    # Calculate SL price
    if direction == 'LONG':
        sl_price = entry_price * (1 - effective_sl_pct / 100)
    else:  # SHORT
        sl_price = entry_price * (1 + effective_sl_pct / 100)
    
    return {
        'tp_prices': tp_prices,  # List of TP prices
        'sl_price': sl_price,
        'effective_tp_pcts': effective_tp_pcts,  # List of effective price move %
        'effective_sl_pct': effective_sl_pct,  # Effective SL price move %
        'tp_profit_pcts': tp_profit_pcts,  # List of profit % with leverage
        'sl_loss_pct': sl_loss_pct,  # Loss % with leverage
        'scaling_factor': scaling_factor,  # How much we scaled (1.0 = no cap)
        'is_capped': scaling_factor < 1.0  # True if cap was applied
    }


def add_short_cooldown(symbol: str, cooldown_minutes: int = 30):
    """
    Add a symbol to SHORT cooldown after a losing trade
    
    Args:
        symbol: Trading pair (e.g., 'LSK/USDT')
        cooldown_minutes: How long to block SHORTS (default: 30 min)
    """
    cooldown_until = datetime.utcnow() + timedelta(minutes=cooldown_minutes)
    shorts_cooldown[symbol] = cooldown_until
    logger.info(f"🚫 {symbol} added to SHORT cooldown for {cooldown_minutes} minutes (prevents re-shorting strong pump)")
    return cooldown_until


# 🔥 GLOBAL SIGNAL COOLDOWN - Prevents duplicate signals for same coin
signal_cooldowns = {}  # {symbol: datetime} - When cooldown expires

def is_symbol_on_cooldown(symbol: str) -> bool:
    """Check if symbol was recently signaled (prevents duplicates)"""
    if symbol in signal_cooldowns:
        if datetime.utcnow() < signal_cooldowns[symbol]:
            return True
        else:
            del signal_cooldowns[symbol]
    return False

def add_signal_cooldown(symbol: str, cooldown_minutes: int = 30):
    """Add cooldown after signaling a coin (prevents rapid duplicate signals)"""
    cooldown_until = datetime.utcnow() + timedelta(minutes=cooldown_minutes)
    signal_cooldowns[symbol] = cooldown_until
    logger.info(f"⏰ {symbol} on signal cooldown for {cooldown_minutes} minutes")


class TopGainersSignalService:
    """Service to fetch and analyze top gainers from Bitunix using direct API"""
    
    def __init__(self):
        self.base_url = "https://fapi.bitunix.com"  # For tickers and trading
        self.binance_url = "https://fapi.binance.com"  # For candle data (Binance Futures public API)
        self.client = httpx.AsyncClient(timeout=30.0)
        self.min_volume_usdt = 200000  # $200k minimum - catches lower caps that pump hard!
        self.max_spread_percent = 0.5  # Max 0.5% bid-ask spread for good execution
        self.min_depth_usdt = 50000  # Min $50k liquidity at ±1% price levels
        
    async def initialize(self):
        """Initialize Bitunix API client"""
        try:
            logger.info("TopGainersSignalService initialized with Bitunix direct API")
        except Exception as e:
            logger.error(f"Failed to initialize TopGainersSignalService: {e}")
            raise
    
    async def fetch_candles(self, symbol: str, interval: str, limit: int = 100) -> List:
        """
        Fetch OHLCV candles from Binance Futures with MEXC fallback.
        Smart switching: After 3 Binance failures, use MEXC-only for 10 minutes.
        Uses semaphore to limit concurrent API requests.
        """
        global binance_failure_count, binance_blocked_until
        
        async with API_SEMAPHORE:
            # Check if we should skip Binance (blocked mode)
            use_mexc_only = False
            if binance_blocked_until:
                if datetime.utcnow() < binance_blocked_until:
                    use_mexc_only = True
                else:
                    # Block expired, reset and try Binance again
                    binance_failure_count = 0
                    binance_blocked_until = None
                    logger.info("🔄 Binance block expired, trying Binance again")
            
            # Try Binance first (unless blocked)
            if not use_mexc_only:
                try:
                    binance_symbol = symbol.replace('/', '')
                    url = f"{self.binance_url}/fapi/v1/klines"
                    params = {
                        'symbol': binance_symbol,
                        'interval': interval,
                        'limit': limit
                    }
                    
                    response = await self.client.get(url, params=params, timeout=10)
                    response.raise_for_status()
                    data = response.json()
                    
                    if isinstance(data, list) and data:
                        # Success - reset failure count
                        binance_failure_count = 0
                        binance_blocked_until = None
                        
                        formatted_candles = []
                        for candle in data:
                            if isinstance(candle, list) and len(candle) >= 6:
                                formatted_candles.append([
                                    int(candle[0]),
                                    float(candle[1]),
                                    float(candle[2]),
                                    float(candle[3]),
                                    float(candle[4]),
                                    float(candle[5])
                                ])
                        return formatted_candles
                        
                except Exception as e:
                    binance_failure_count += 1
                    logger.warning(f"Binance failed ({binance_failure_count}/{BINANCE_FAILURE_THRESHOLD}): {e}")
                    
                    if binance_failure_count >= BINANCE_FAILURE_THRESHOLD:
                        binance_blocked_until = datetime.utcnow() + timedelta(minutes=BINANCE_BLOCK_DURATION_MINUTES)
                        logger.warning(f"🚫 BINANCE BLOCKED - Switching to MEXC-only for {BINANCE_BLOCK_DURATION_MINUTES} minutes")
            
            # MEXC fallback (or primary if Binance blocked)
            try:
                mexc_symbol = symbol.replace('/', '_')
                mexc_url = "https://contract.mexc.com/api/v1/contract/kline"
                mexc_params = {
                    'symbol': mexc_symbol,
                    'interval': 'Min5',  # Default
                    'limit': limit
                }
                
                # Fix interval format for MEXC
                if interval == '5m':
                    mexc_params['interval'] = 'Min5'
                elif interval == '15m':
                    mexc_params['interval'] = 'Min15'
                elif interval == '1h':
                    mexc_params['interval'] = 'Hour1'
                elif interval == '4h':
                    mexc_params['interval'] = 'Hour4'
                
                response = await self.client.get(mexc_url, params=mexc_params, timeout=10)
                response.raise_for_status()
                data = response.json()
                
                if data.get('success') and data.get('data'):
                    candles_data = data['data'].get('time', [])
                    opens = data['data'].get('open', [])
                    highs = data['data'].get('high', [])
                    lows = data['data'].get('low', [])
                    closes = data['data'].get('close', [])
                    vols = data['data'].get('vol', [])
                    
                    formatted_candles = []
                    for i in range(len(candles_data)):
                        formatted_candles.append([
                            int(candles_data[i]) * 1000,  # Convert to ms
                            float(opens[i]),
                            float(highs[i]),
                            float(lows[i]),
                            float(closes[i]),
                            float(vols[i])
                        ])
                    
                    if use_mexc_only:
                        logger.debug(f"📊 MEXC-only mode: {symbol} candles fetched")
                    else:
                        logger.info(f"✅ MEXC fallback successful for {symbol} candles")
                    return formatted_candles
                    
            except Exception as mexc_error:
                logger.error(f"MEXC candle fetch failed for {symbol}: {mexc_error}")
            
            return []
    
    async def get_ticker_price(self, symbol: str) -> Optional[float]:
        """
        Fetch LIVE ticker price from Binance Futures.
        Used for anti-top filters to catch mid-candle spikes.
        """
        try:
            binance_symbol = symbol.replace('/', '')
            url = f"{self.binance_url}/fapi/v1/ticker/price"
            params = {'symbol': binance_symbol}
            
            response = await self.client.get(url, params=params, timeout=5)
            response.raise_for_status()
            data = response.json()
            
            price = float(data.get('price', 0))
            if price > 0:
                return price
            return None
        except Exception as e:
            logger.warning(f"Failed to get live price for {symbol}: {e}")
            return None
    
    async def check_liquidity(self, symbol: str) -> Dict:
        """
        Check execution quality via spread and order book depth
        
        Returns quality metrics:
        - is_liquid: bool (passes all checks)
        - spread_percent: float (bid-ask spread %)
        - depth_1pct: float (total liquidity at ±1% levels in USDT)
        - reason: str (failure reason if not liquid)
        """
        try:
            # Convert symbol format: BTC/USDT → BTCUSDT
            bitunix_symbol = symbol.replace('/', '')
            
            # Fetch ticker for current bid/ask
            ticker_url = f"{self.base_url}/api/v1/futures/market/tickers"
            ticker_response = await self.client.get(ticker_url, params={'symbols': bitunix_symbol})
            ticker_data = ticker_response.json()
            
            # Parse ticker response
            tickers = ticker_data.get('data', []) if isinstance(ticker_data, dict) else ticker_data
            if not tickers or not isinstance(tickers, list):
                return {'is_liquid': False, 'reason': 'No ticker data'}
            
            ticker = tickers[0] if isinstance(tickers, list) else tickers
            
            # Bitunix API doesn't return bid/ask, only markPrice and lastPrice
            # Use 24h volume as primary liquidity indicator
            volume_24h = float(ticker.get('quoteVol', 0) or ticker.get('volume24h', 0))
            last_price = float(ticker.get('lastPrice', 0) or ticker.get('last', 0))
            
            if last_price <= 0:
                return {'is_liquid': False, 'reason': 'Invalid price data'}
            
            # Note: No spread check since Bitunix API doesn't provide bid/ask
            # Volume is more important for momentum trades anyway
            
            # If volume is high, assume decent depth
            if volume_24h < self.min_volume_usdt:
                return {
                    'is_liquid': False,
                    'spread_percent': 0,
                    'reason': f'Low 24h volume: ${volume_24h:,.0f} (need ${self.min_volume_usdt:,.0f}+)'
                }
            
            # Passed all checks
            return {
                'is_liquid': True,
                'spread_percent': 0,  # Not available from Bitunix API
                'volume_24h': volume_24h,
                'reason': 'Good liquidity'
            }
            
        except Exception as e:
            logger.error(f"Error checking liquidity for {symbol}: {e}")
            return {'is_liquid': False, 'reason': f'Error: {str(e)}'}
    
    async def get_funding_rate(self, symbol: str) -> Dict:
        """
        Fetch current funding rate from Bitunix Futures API
        
        Funding rate indicates market sentiment:
        - Positive (>0.01%) = Longs paying shorts = Bullish/greedy market
        - Negative (<-0.01%) = Shorts paying longs = Bearish market
        - High positive (>0.1%) = Extremely greedy = Good for SHORTS
        - Negative (<-0.05%) = Shorts underwater = Good for LONGS
        
        Returns:
        - funding_rate: float (e.g., 0.0015 = 0.15%)
        - next_funding_time: int (timestamp)
        - is_extreme: bool (funding > 0.1% or < -0.05%)
        """
        try:
            # Convert symbol format: BTC/USDT → BTCUSDT
            bitunix_symbol = symbol.replace('/', '')
            
            url = f"{self.base_url}/api/v1/futures/market/funding_rate"
            response = await self.client.get(url, params={'symbol': bitunix_symbol})
            response.raise_for_status()
            data = response.json()
            
            # Parse response
            if isinstance(data, dict) and 'data' in data:
                funding_data = data['data']
            else:
                funding_data = data
            
            funding_rate = float(funding_data.get('fundingRate', 0))
            next_funding_time = int(funding_data.get('nextFundingTime', 0))
            
            # Classify funding rate
            is_extreme_positive = funding_rate > 0.001  # >0.1% = very greedy
            is_extreme_negative = funding_rate < -0.0005  # <-0.05% = shorts underwater
            
            return {
                'funding_rate': funding_rate,
                'funding_rate_percent': funding_rate * 100,  # Convert to percentage
                'next_funding_time': next_funding_time,
                'is_extreme_positive': is_extreme_positive,
                'is_extreme_negative': is_extreme_negative,
                'sentiment': 'greedy' if funding_rate > 0.001 else 'fearful' if funding_rate < -0.0005 else 'neutral'
            }
            
        except Exception as e:
            logger.warning(f"Error fetching funding rate for {symbol}: {e}")
            return {
                'funding_rate': 0,
                'funding_rate_percent': 0,
                'next_funding_time': 0,
                'is_extreme_positive': False,
                'is_extreme_negative': False,
                'sentiment': 'unknown'
            }
    
    async def get_order_book_walls(self, symbol: str, entry_price: float, direction: str = 'LONG') -> Dict:
        """
        Analyze order book for massive buy/sell walls that could block price movement
        
        For LONGS:
        - Check for sell walls ABOVE entry price (resistance)
        - Check for buy walls BELOW entry price (support)
        
        For SHORTS:
        - Check for buy walls BELOW entry price (support that prevents dump)
        - Check for sell walls ABOVE entry price (resistance)
        
        Returns:
        - has_blocking_wall: bool (True if massive wall detected in path)
        - wall_price: float (price level of the wall)
        - wall_size_usdt: float (total USDT value of the wall)
        - support_below: float (total buy wall support in USDT)
        - resistance_above: float (total sell wall resistance in USDT)
        """
        try:
            # Convert symbol format: BTC/USDT → BTCUSDT
            bitunix_symbol = symbol.replace('/', '')
            
            # Fetch order book depth (limit=20 gives ±2% price range typically)
            url = f"{self.base_url}/api/v1/futures/market/depth"
            response = await self.client.get(url, params={'symbol': bitunix_symbol, 'limit': 20})
            response.raise_for_status()
            data = response.json()
            
            # Parse response
            if isinstance(data, dict) and 'data' in data:
                book = data['data']
            else:
                book = data
            
            asks = book.get('asks', [])  # [[price, quantity], ...]
            bids = book.get('bids', [])  # [[price, quantity], ...]
            
            if not asks or not bids:
                return {'has_blocking_wall': False, 'reason': 'Empty order book'}
            
            # Calculate wall detection threshold (dynamic based on book depth)
            # A "wall" is an order 5x larger than average order size
            avg_ask_size = sum(float(a[1]) * float(a[0]) for a in asks[:10]) / 10 if asks else 0
            avg_bid_size = sum(float(b[1]) * float(b[0]) for b in bids[:10]) / 10 if bids else 0
            wall_threshold = max(avg_ask_size, avg_bid_size) * 5
            
            # For LONGS: Check for sell walls above entry (resistance)
            if direction == 'LONG':
                resistance_walls = []
                support_walls = []
                
                for ask in asks:
                    price = float(ask[0])
                    quantity = float(ask[1])
                    size_usdt = price * quantity
                    
                    # Only check walls above entry price
                    if price > entry_price and size_usdt > wall_threshold:
                        resistance_walls.append({
                            'price': price,
                            'size_usdt': size_usdt,
                            'distance_percent': ((price - entry_price) / entry_price) * 100
                        })
                
                # Check buy walls below entry (support)
                for bid in bids:
                    price = float(bid[0])
                    quantity = float(bid[1])
                    size_usdt = price * quantity
                    
                    if price < entry_price and size_usdt > wall_threshold:
                        support_walls.append({
                            'price': price,
                            'size_usdt': size_usdt,
                            'distance_percent': ((entry_price - price) / entry_price) * 100
                        })
                
                # Find nearest resistance wall
                if resistance_walls:
                    nearest_wall = min(resistance_walls, key=lambda x: x['distance_percent'])
                    return {
                        'has_blocking_wall': True,
                        'wall_type': 'resistance',
                        'wall_price': nearest_wall['price'],
                        'wall_size_usdt': nearest_wall['size_usdt'],
                        'wall_distance_percent': nearest_wall['distance_percent'],
                        'total_resistance': sum(w['size_usdt'] for w in resistance_walls),
                        'total_support': sum(w['size_usdt'] for w in support_walls)
                    }
                
                return {
                    'has_blocking_wall': False,
                    'total_resistance': sum(float(a[1]) * float(a[0]) for a in asks[:5]),
                    'total_support': sum(float(b[1]) * float(b[0]) for b in bids[:5])
                }
            
            # For SHORTS: Check for buy walls below entry (support that prevents dump)
            else:  # direction == 'SHORT'
                support_walls = []
                resistance_walls = []
                
                for bid in bids:
                    price = float(bid[0])
                    quantity = float(bid[1])
                    size_usdt = price * quantity
                    
                    # Only check walls below entry price (support)
                    if price < entry_price and size_usdt > wall_threshold:
                        support_walls.append({
                            'price': price,
                            'size_usdt': size_usdt,
                            'distance_percent': ((entry_price - price) / entry_price) * 100
                        })
                
                # Check sell walls above entry
                for ask in asks:
                    price = float(ask[0])
                    quantity = float(ask[1])
                    size_usdt = price * quantity
                    
                    if price > entry_price and size_usdt > wall_threshold:
                        resistance_walls.append({
                            'price': price,
                            'size_usdt': size_usdt,
                            'distance_percent': ((price - entry_price) / entry_price) * 100
                        })
                
                # Find nearest support wall (blocks dump)
                if support_walls:
                    nearest_wall = min(support_walls, key=lambda x: x['distance_percent'])
                    return {
                        'has_blocking_wall': True,
                        'wall_type': 'support',
                        'wall_price': nearest_wall['price'],
                        'wall_size_usdt': nearest_wall['size_usdt'],
                        'wall_distance_percent': nearest_wall['distance_percent'],
                        'total_support': sum(w['size_usdt'] for w in support_walls),
                        'total_resistance': sum(w['size_usdt'] for w in resistance_walls)
                    }
                
                return {
                    'has_blocking_wall': False,
                    'total_support': sum(float(b[1]) * float(b[0]) for b in bids[:5]),
                    'total_resistance': sum(float(a[1]) * float(a[0]) for a in asks[:5])
                }
            
        except Exception as e:
            logger.warning(f"Error fetching order book for {symbol}: {e}")
            return {'has_blocking_wall': False, 'reason': f'Error: {str(e)}'}
    
    async def check_manipulation_risk(self, symbol: str, candles_5m: List) -> Dict:
        """
        Detect pump & dump manipulation patterns
        
        Checks:
        1. Volume distribution (not just 1 giant candle)
        2. Wick-to-body ratio (avoid fake pumps with huge wicks)
        3. Listing age (skip coins <48h old)
        
        Returns:
        - is_safe: bool (passes all checks)
        - risk_score: int (0-100, higher = more risky)
        - flags: List[str] (specific red flags)
        """
        try:
            flags = []
            risk_score = 0
            
            if not candles_5m or len(candles_5m) < 10:
                return {'is_safe': False, 'risk_score': 100, 'flags': ['Insufficient candle data']}
            
            # Extract last 10 candles
            recent_candles = candles_5m[-10:]
            volumes = [c[5] for c in recent_candles]
            
            # Check 1: Volume Distribution (not just 1 whale candle)
            max_volume = max(volumes)
            avg_volume = sum(volumes) / len(volumes)
            
            if max_volume > avg_volume * 5:
                flags.append('Single whale candle detected')
                risk_score += 30
            
            # Count elevated volume candles (>1.5x average)
            elevated_count = sum(1 for v in volumes if v > avg_volume * 1.5)
            if elevated_count < 3:
                flags.append('Volume not sustained (need 3+ elevated candles)')
                risk_score += 20
            
            # Check 2: Wick-to-Body Ratio (avoid fake pumps)
            current_candle = candles_5m[-1]
            open_price = current_candle[1]
            high = current_candle[2]
            low = current_candle[3]
            close = current_candle[4]
            
            body_size = abs(close - open_price)
            upper_wick = high - max(close, open_price)
            lower_wick = min(close, open_price) - low
            
            # If wick is >2x body size, it's a fake pump
            if body_size > 0:
                wick_to_body_ratio = max(upper_wick, lower_wick) / body_size
                if wick_to_body_ratio > 2.0:
                    flags.append(f'Excessive wick (fake pump): wick/body={wick_to_body_ratio:.1f}x')
                    risk_score += 40
            
            # Check 3: Listing Age (skip very new coins < 10 candles of history)
            # If we have less than 30 candles of 5m data, coin might be too new
            if len(candles_5m) < 30:
                flags.append('Coin too new (<2.5 hours of data)')
                risk_score += 50
            
            # Determine if safe
            is_safe = risk_score < 50 and len(flags) < 2
            
            return {
                'is_safe': is_safe,
                'risk_score': risk_score,
                'flags': flags if flags else ['No red flags'],
                'elevated_volume_candles': elevated_count
            }
            
        except Exception as e:
            logger.error(f"Error checking manipulation risk for {symbol}: {e}")
            return {'is_safe': False, 'risk_score': 100, 'flags': [f'Error: {str(e)}']}
    
    async def get_top_gainers(self, limit: int = 10, min_change_percent: float = 10.0) -> List[Dict]:
        """
        Fetch top gainers using BINANCE + MEXC FUTURES APIs for accurate 24h data
        Then filter to only coins available on Bitunix for trading
        
        OPTIMIZED FOR SHORTS: Higher min_change (10%+) = better reversal candidates
        Uses Binance as primary source, MEXC as fallback for coins not on Binance
        
        Args:
            limit: Number of top gainers to return
            min_change_percent: Minimum 24h change % to qualify (default 10% for shorts)
            
        Returns:
            List of {symbol, change_percent, volume, price} sorted by change %
        """
        try:
            # 🔥 FETCH FROM MULTIPLE SOURCES for better coverage
            merged_data = {}  # symbol -> ticker data (Binance priority)
            
            # === SOURCE 1: BINANCE FUTURES (Primary - most reliable) ===
            binance_count = 0
            try:
                binance_url = "https://fapi.binance.com/fapi/v1/ticker/24hr"
                response = await self.client.get(binance_url, timeout=10)
                response.raise_for_status()
                binance_tickers = response.json()
                
                for ticker in binance_tickers:
                    symbol = ticker.get('symbol', '')
                    if not symbol.endswith('USDT'):
                        continue
                    try:
                        merged_data[symbol] = {
                            'symbol': symbol,
                            'change_percent': float(ticker.get('priceChangePercent', 0)),
                            'last_price': float(ticker.get('lastPrice', 0)),
                            'volume_usdt': float(ticker.get('quoteVolume', 0)),
                            'high_24h': float(ticker.get('highPrice', 0)),
                            'low_24h': float(ticker.get('lowPrice', 0)),
                            'source': 'binance'
                        }
                        binance_count += 1
                    except (ValueError, TypeError):
                        continue
            except Exception as e:
                logger.warning(f"⚠️ Binance API error (using MEXC only): {e}")
            
            # === SOURCE 2: MEXC FUTURES (Secondary - for coins not on Binance) ===
            mexc_count = 0
            mexc_added = 0
            try:
                mexc_url = "https://contract.mexc.com/api/v1/contract/ticker"
                response = await self.client.get(mexc_url, timeout=10)
                response.raise_for_status()
                mexc_data = response.json()
                mexc_tickers = mexc_data.get('data', []) if isinstance(mexc_data, dict) else mexc_data
                
                for ticker in mexc_tickers:
                    # MEXC uses underscore format: BTC_USDT -> BTCUSDT
                    raw_symbol = ticker.get('symbol', '')
                    symbol = raw_symbol.replace('_', '')
                    if not symbol.endswith('USDT'):
                        continue
                    mexc_count += 1
                    
                    # Only add if NOT already in Binance data (Binance takes priority)
                    if symbol not in merged_data:
                        try:
                            # MEXC fields: lastPrice, riseFallRate (change %), amount24 (USDT volume)
                            change_pct = float(ticker.get('riseFallRate', 0))
                            # MEXC riseFallRate might be decimal (0.35) or percent (35)
                            if abs(change_pct) < 5 and abs(change_pct) > 0:
                                change_pct = change_pct * 100  # Convert decimal to percent
                            
                            merged_data[symbol] = {
                                'symbol': symbol,
                                'change_percent': change_pct,
                                'last_price': float(ticker.get('lastPrice', 0)),
                                'volume_usdt': float(ticker.get('amount24', 0)),  # USDT volume
                                'high_24h': float(ticker.get('high24Price', 0)),
                                'low_24h': float(ticker.get('low24Price', 0)),
                                'source': 'mexc'
                            }
                            mexc_added += 1
                        except (ValueError, TypeError):
                            continue
            except Exception as e:
                logger.warning(f"⚠️ MEXC API error (using Binance only): {e}")
            
            # === GET BITUNIX AVAILABLE SYMBOLS + CHANGE DATA ===
            bitunix_url = f"{self.base_url}/api/v1/futures/market/tickers"
            bitunix_response = await self.client.get(bitunix_url)
            bitunix_data = bitunix_response.json()
            bitunix_symbols = set()
            bitunix_change_map = {}
            if isinstance(bitunix_data, dict) and bitunix_data.get('data'):
                for t in bitunix_data.get('data', []):
                    sym = t.get('symbol', '')
                    bitunix_symbols.add(sym)
                    try:
                        open_price = float(t.get('open', 0))
                        last_price = float(t.get('last', 0) or t.get('lastPrice', 0))
                        if open_price > 0 and last_price > 0:
                            bitunix_change_map[sym] = round(((last_price - open_price) / open_price) * 100, 2)
                    except (ValueError, TypeError):
                        pass
            
            logger.info(f"📊 DATA SOURCES: Binance={binance_count} | MEXC={mexc_count} (+{mexc_added} unique) | Bitunix={len(bitunix_symbols)} tradeable")
            
            gainers = []
            rejected_not_on_bitunix = 0
            
            for symbol, data in merged_data.items():
                if symbol not in bitunix_symbols:
                    rejected_not_on_bitunix += 1
                    continue
                
                normalized = symbol.replace('/USDT', '').replace('USDT', '')
                if normalized in BLACKLISTED_SYMBOLS or symbol in BLACKLISTED_SYMBOLS:
                    logger.info(f"🚫 {symbol} BLACKLISTED at source - excluded from gainers")
                    continue
                
                change_percent = bitunix_change_map.get(symbol, data['change_percent'])
                last_price = data['last_price']
                volume_usdt = data['volume_usdt']
                high_24h = data['high_24h']
                low_24h = data['low_24h']
                
                if (change_percent >= min_change_percent and 
                    volume_usdt >= self.min_volume_usdt):
                    
                    gainers.append({
                        'symbol': symbol.replace('USDT', '/USDT'),
                        'change_percent': round(change_percent, 2),
                        'volume_24h': round(volume_usdt, 0),
                        'price': last_price,
                        'high_24h': high_24h,
                        'low_24h': low_24h
                    })
            
            # Sort by change % descending
            gainers.sort(key=lambda x: x['change_percent'], reverse=True)
            
            if rejected_not_on_bitunix > 0:
                logger.debug(f"Filtered out {rejected_not_on_bitunix} coins not tradeable on Bitunix")
            
            return gainers[:limit]
            
        except Exception as e:
            logger.error(f"Error fetching top gainers: {e}", exc_info=True)
            return []
    
    async def scan_fresh_impulses(self, limit: int = 20) -> List[Dict]:
        """
        🔥 FRESH IMPULSE SCANNER - Find coins BEFORE they hit top gainers
        
        Scans ALL Bitunix coins for:
        - LOW 24h change (<15%) = hasn't pumped yet
        - Recent 5m volume spike (3x+ average)
        - Small recent price increase (2-8% in last 30 min) = just starting
        
        This catches coins at the START of moves, not after they're already up 20%+
        Uses BINANCE for accurate 24h data (Bitunix API is garbage!)
        """
        try:
            logger.info("🔍 FRESH IMPULSE SCAN: Scanning ALL coins for new moves...")
            
            # Get Bitunix tradeable symbols (for filtering)
            bitunix_url = f"{self.base_url}/api/v1/futures/market/tickers"
            bitunix_response = await self.client.get(bitunix_url, timeout=15)
            bitunix_data = bitunix_response.json()
            
            bitunix_symbols = set()
            bitunix_change_map = {}
            if isinstance(bitunix_data, dict) and bitunix_data.get('data'):
                for t in bitunix_data.get('data', []):
                    sym = t.get('symbol', '')
                    bitunix_symbols.add(sym)
                    try:
                        open_price = float(t.get('open', 0))
                        last_price_val = float(t.get('last', 0) or t.get('lastPrice', 0))
                        if open_price > 0 and last_price_val > 0:
                            bitunix_change_map[sym] = round(((last_price_val - open_price) / open_price) * 100, 2)
                    except (ValueError, TypeError):
                        pass
            
            all_symbols = []
            try:
                binance_url = "https://fapi.binance.com/fapi/v1/ticker/24hr"
                response = await self.client.get(binance_url, timeout=10)
                response.raise_for_status()
                binance_tickers = response.json()
                
                for ticker in binance_tickers:
                    symbol = ticker.get('symbol', '')
                    if not symbol.endswith('USDT'):
                        continue
                    if symbol not in bitunix_symbols:
                        continue
                    try:
                        change_24h = bitunix_change_map.get(symbol, float(ticker.get('priceChangePercent', 0)))
                        volume_usdt = float(ticker.get('quoteVolume', 0))
                        price = float(ticker.get('lastPrice', 0))
                        all_symbols.append({
                            'symbol': symbol.replace('USDT', '/USDT'),
                            'change_24h': change_24h,
                            'volume_24h': volume_usdt,
                            'price': price
                        })
                    except (ValueError, TypeError):
                        continue
            except Exception as e:
                logger.warning(f"⚠️ Binance API error: {e}")
                return []
            
            logger.info(f"📊 Scanning {len(all_symbols)} coins (Binance data, Bitunix tradeable)")
            
            # Filter: Low 24h change (hasn't pumped yet)
            fresh_candidates = []
            for coin in all_symbols:
                # Skip blacklisted
                normalized = coin['symbol'].replace('/USDT', '').replace('USDT', '')
                if normalized in BLACKLISTED_SYMBOLS or coin['symbol'] in BLACKLISTED_SYMBOLS:
                    continue
                
                # Must have LOW 24h change - coin hasn't pumped yet
                # ONLY green or flat coins - no calling coins already down!
                if coin['change_24h'] > 25:  # Already up 25%+ = too late
                    continue
                if coin['change_24h'] < -5:  # Down more than 5% = downtrend, skip
                    continue
                
                # Minimum volume for liquidity
                if coin['volume_24h'] < 100000:  # $100K min
                    continue
                
                fresh_candidates.append(coin)
            
            logger.info(f"📊 {len(fresh_candidates)} coins with 24h change -5% to +25% (Binance verified)")
            
            # Now scan each candidate for RECENT impulse (5m volume spike + price move)
            impulse_coins = []
            scan_count = 0
            max_scans = 50  # Limit API calls
            
            for coin in fresh_candidates[:max_scans]:
                scan_count += 1
                symbol = coin['symbol']
                
                try:
                    # Get 5m candles for volume analysis
                    candles_5m = await self.fetch_candles(symbol, '5m', limit=12)  # Last 1 hour
                    if len(candles_5m) < 8:
                        continue
                    
                    # Calculate average volume for last hour (excluding latest 2 candles)
                    volumes = [c[5] for c in candles_5m[:-2]]  # Volume is index 5
                    avg_volume = sum(volumes) / len(volumes) if volumes else 0
                    
                    # Latest 2 candles volume
                    recent_volume = sum(c[5] for c in candles_5m[-2:])
                    
                    if avg_volume <= 0:
                        continue
                    
                    # Check for volume spike (3x+ average)
                    volume_ratio = (recent_volume / 2) / avg_volume
                    
                    if volume_ratio < 3.0:  # Need 3x volume spike
                        continue
                    
                    # Check recent price move (last 30 min = 6 x 5m candles)
                    price_30m_ago = candles_5m[-6][4]  # Close 30 min ago
                    current_price = candles_5m[-1][4]
                    recent_move = ((current_price - price_30m_ago) / price_30m_ago) * 100
                    
                    # Must be up 2-8% in last 30 min (starting to move, not extended)
                    if recent_move < 2 or recent_move > 8:
                        continue
                    
                    logger.info(f"  🚀 FRESH IMPULSE: {symbol} | 30m: +{recent_move:.1f}% | Vol: {volume_ratio:.1f}x | 24h: {coin['change_24h']:.1f}%")
                    
                    impulse_coins.append({
                        'symbol': symbol,
                        'change_percent': coin['change_24h'],
                        'recent_move': recent_move,
                        'volume_ratio': volume_ratio,
                        'volume_24h': coin['volume_24h'],
                        'price': current_price
                    })
                    
                except Exception as e:
                    continue
            
            logger.info(f"✅ Found {len(impulse_coins)} fresh impulses from {scan_count} scanned")
            
            # Sort by volume ratio (strongest signals first)
            impulse_coins.sort(key=lambda x: x['volume_ratio'], reverse=True)
            return impulse_coins[:limit]
            
        except Exception as e:
            logger.error(f"Error in fresh impulse scan: {e}")
            return []
    
    async def get_top_losers(self, limit: int = 10, max_change_percent: float = -10.0, min_change_percent: float = -30.0) -> List[Dict]:
        """
        Fetch TOP LOSERS using BINANCE + MEXC FUTURES APIs
        Filter to only coins available on Bitunix for trading
        
        OPTIMIZED FOR SHORTS: Coins already in downtrend = ride the momentum!
        Short the relief rally bounce instead of trying to call tops
        
        Args:
            limit: Number of top losers to return
            max_change_percent: Maximum 24h change % (e.g., -10% = down at least 10%)
            min_change_percent: Minimum 24h change % (e.g., -30% = not down more than 30%)
            
        Returns:
            List of {symbol, change_percent, volume, price, high_24h, low_24h} sorted by change %
        """
        try:
            merged_data = {}
            
            # === SOURCE 1: BINANCE FUTURES (Primary) ===
            binance_count = 0
            try:
                binance_url = "https://fapi.binance.com/fapi/v1/ticker/24hr"
                response = await self.client.get(binance_url, timeout=10)
                response.raise_for_status()
                binance_tickers = response.json()
                
                for ticker in binance_tickers:
                    symbol = ticker.get('symbol', '')
                    if not symbol.endswith('USDT'):
                        continue
                    try:
                        merged_data[symbol] = {
                            'symbol': symbol,
                            'change_percent': float(ticker.get('priceChangePercent', 0)),
                            'last_price': float(ticker.get('lastPrice', 0)),
                            'volume_usdt': float(ticker.get('quoteVolume', 0)),
                            'high_24h': float(ticker.get('highPrice', 0)),
                            'low_24h': float(ticker.get('lowPrice', 0)),
                            'source': 'binance'
                        }
                        binance_count += 1
                    except (ValueError, TypeError):
                        continue
            except Exception as e:
                logger.warning(f"⚠️ Binance API error for losers: {e}")
            
            # === SOURCE 2: MEXC FUTURES (Secondary) ===
            mexc_added = 0
            try:
                mexc_url = "https://contract.mexc.com/api/v1/contract/ticker"
                response = await self.client.get(mexc_url, timeout=10)
                response.raise_for_status()
                mexc_data = response.json()
                mexc_tickers = mexc_data.get('data', []) if isinstance(mexc_data, dict) else mexc_data
                
                for ticker in mexc_tickers:
                    raw_symbol = ticker.get('symbol', '')
                    symbol = raw_symbol.replace('_', '')
                    if not symbol.endswith('USDT'):
                        continue
                    
                    if symbol not in merged_data:
                        try:
                            change_pct = float(ticker.get('riseFallRate', 0))
                            if abs(change_pct) < 5 and abs(change_pct) > 0:
                                change_pct = change_pct * 100
                            
                            merged_data[symbol] = {
                                'symbol': symbol,
                                'change_percent': change_pct,
                                'last_price': float(ticker.get('lastPrice', 0)),
                                'volume_usdt': float(ticker.get('amount24', 0)),
                                'high_24h': float(ticker.get('high24Price', 0)),
                                'low_24h': float(ticker.get('low24Price', 0)),
                                'source': 'mexc'
                            }
                            mexc_added += 1
                        except (ValueError, TypeError):
                            continue
            except Exception as e:
                logger.warning(f"⚠️ MEXC API error for losers: {e}")
            
            # === GET BITUNIX AVAILABLE SYMBOLS + CHANGE DATA ===
            bitunix_url = f"{self.base_url}/api/v1/futures/market/tickers"
            bitunix_response = await self.client.get(bitunix_url)
            bitunix_data = bitunix_response.json()
            bitunix_symbols = set()
            bitunix_change_map = {}
            if isinstance(bitunix_data, dict) and bitunix_data.get('data'):
                for t in bitunix_data.get('data', []):
                    sym = t.get('symbol', '')
                    bitunix_symbols.add(sym)
                    try:
                        open_price = float(t.get('open', 0))
                        last_price = float(t.get('last', 0) or t.get('lastPrice', 0))
                        if open_price > 0 and last_price > 0:
                            bitunix_change_map[sym] = round(((last_price - open_price) / open_price) * 100, 2)
                    except (ValueError, TypeError):
                        pass
            
            losers = []
            
            for symbol, data in merged_data.items():
                if symbol not in bitunix_symbols:
                    continue
                
                normalized = symbol.replace('/USDT', '').replace('USDT', '')
                if normalized in BLACKLISTED_SYMBOLS or symbol in BLACKLISTED_SYMBOLS:
                    continue
                
                change_percent = bitunix_change_map.get(symbol, data['change_percent'])
                last_price = data['last_price']
                volume_usdt = data['volume_usdt']
                high_24h = data['high_24h']
                low_24h = data['low_24h']
                
                if (change_percent <= max_change_percent and 
                    change_percent >= min_change_percent and
                    volume_usdt >= self.min_volume_usdt):
                    
                    losers.append({
                        'symbol': symbol.replace('USDT', '/USDT'),
                        'change_percent': round(change_percent, 2),
                        'volume_24h': round(volume_usdt, 0),
                        'price': last_price,
                        'high_24h': high_24h,
                        'low_24h': low_24h
                    })
            
            # Sort by change % ascending (most negative first = biggest losers)
            losers.sort(key=lambda x: x['change_percent'])
            
            logger.info(f"📉 TOP LOSERS: Found {len(losers)} coins down {max_change_percent}% to {min_change_percent}%")
            
            return losers[:limit]
            
        except Exception as e:
            logger.error(f"Error fetching top losers: {e}", exc_info=True)
            return []
    
    async def analyze_normal_short(self, symbol: str, coin_data: Dict, current_price: float) -> Optional[Dict]:
        """
        🎯 TREND REVERSAL SHORTS (with DUMP MODE relaxation)
        
        Detects when trend has CHANGED from bullish to bearish, then finds good entries.
        NOT looking for overextension - looking for trend reversal + entry timing.
        
        TREND CHANGE DETECTION:
        - EMA9 < EMA21 (bearish cross)
        - Lower highs forming
        - Lower lows forming
        - EMAs converging with red candles
        
        ENTRY QUALITY:
        - Upper wick rejection
        - Recent selling pressure (red candles)
        - Pulled back from high
        - Not chasing too far above EMA
        
        🔴 DUMP MODE: Only 1 trend sign needed (vs 2 normally)
        
        Returns signal dict or None if TA filters fail
        """
        try:
            # Check for dump mode
            dump_state = await check_dump_mode()
            is_dump_mode = dump_state.get('is_dump', False)
            
            if is_dump_mode:
                logger.info(f"  🔴 {symbol} - DUMP MODE: Relaxed SHORT filters active")
            
            high_24h = coin_data.get('high_24h', 0)
            low_24h = coin_data.get('low_24h', 0)
            change_24h = coin_data.get('change_percent', 0)
            volume_24h = coin_data.get('volume_24h', 0)
            
            if high_24h <= 0 or low_24h <= 0:
                return None
            
            # ═══════════════════════════════════════════════════════
            # PRE-FILTER 1: 24h change range (TIGHTENED)
            # ═══════════════════════════════════════════════════════
            min_change = 3.0 if is_dump_mode else 5.0  # Tightened: 5%+ normally, 3%+ in dump
            max_change = 60.0 if is_dump_mode else 50.0
            
            if not (min_change <= change_24h <= max_change):
                logger.info(f"  ⏭️ {symbol} - Change {change_24h:.1f}% outside {min_change}-{max_change}% range")
                return None
            
            # ═══════════════════════════════════════════════════════
            # PRE-FILTER 2: Liquidity check (TIGHTENED)
            # ═══════════════════════════════════════════════════════
            min_volume = 2_000_000 if is_dump_mode else 3_000_000  # Higher volume required
            if volume_24h < min_volume:
                logger.info(f"  ⏭️ {symbol} - Low volume ${volume_24h:,.0f} (need ${min_volume/1e6:.1f}M+)")
                return None
            
            # ═══════════════════════════════════════════════════════
            # PRE-FILTER 3: Technical analysis
            # ═══════════════════════════════════════════════════════
            candles_5m = await self.fetch_candles(symbol, '5m', limit=20)
            candles_1h = await self.fetch_candles(symbol, '1h', limit=12)
            
            if not candles_5m or len(candles_5m) < 14:
                return None
            
            # Calculate indicators
            closes_5m = [float(c[4]) for c in candles_5m]
            highs_5m = [float(c[2]) for c in candles_5m]
            lows_5m = [float(c[3]) for c in candles_5m]
            volumes = [float(c[5]) for c in candles_5m]
            rsi_5m = self._calculate_rsi(closes_5m, 14)
            
            # Volume ratio
            if len(volumes) >= 6:
                avg_volume = sum(volumes[:-1]) / len(volumes[:-1])
                current_volume = volumes[-1]
                volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0
            else:
                volume_ratio = 1.0
            
            # Calculate EMA structure
            ema9 = self._calculate_ema(closes_5m, 9)
            ema21 = self._calculate_ema(closes_5m, 21)
            current_price = closes_5m[-1]
            price_to_ema9 = ((current_price - ema9) / ema9) * 100 if ema9 > 0 else 0
            ema_spread = ((ema9 - ema21) / ema21) * 100 if ema21 > 0 else 0
            
            # ═══════════════════════════════════════════════════════
            # 🔄 TREND REVERSAL DETECTION (not overextension)
            # ═══════════════════════════════════════════════════════
            
            # Check for lower highs (bearish structure)
            recent_highs = highs_5m[-5:]
            has_lower_highs = len(recent_highs) >= 3 and recent_highs[-1] < recent_highs[-3]
            
            # Check for lower lows (trend changed)
            recent_lows = lows_5m[-5:]
            has_lower_lows = len(recent_lows) >= 3 and recent_lows[-1] < recent_lows[-3]
            
            # EMA bearish cross or structure
            ema_bearish = ema9 < ema21  # EMA9 below EMA21 = bearish
            ema_flattening = abs(ema_spread) < 0.3  # EMAs converging
            
            # Check for bearish momentum (recent red candles)
            red_count = 0
            for c in candles_5m[-5:]:
                if float(c[4]) < float(c[1]):  # Close < Open
                    red_count += 1
            
            # Rejection wick detection (upper wick > body = rejection)
            last_candle = candles_5m[-1]
            c_open, c_high, c_low, c_close = float(last_candle[1]), float(last_candle[2]), float(last_candle[3]), float(last_candle[4])
            body_size = abs(c_close - c_open)
            upper_wick = c_high - max(c_open, c_close)
            wick_ratio = (upper_wick / body_size) if body_size > 0 else 0
            has_rejection_wick = wick_ratio >= 0.5 and upper_wick > 0
            
            # Distance from 24h high
            distance_from_high = ((current_price - high_24h) / high_24h) * 100
            
            # 1H context
            if candles_1h and len(candles_1h) >= 4:
                closes_1h = [float(c[4]) for c in candles_1h]
                rsi_1h = self._calculate_rsi(closes_1h, 14) if len(closes_1h) >= 14 else 50
                ema9_1h = self._calculate_ema(closes_1h, 9)
                ema21_1h = self._calculate_ema(closes_1h, 21) if len(closes_1h) >= 21 else ema9_1h
                ema_bearish_1h = ema9_1h < ema21_1h
            else:
                rsi_1h = 50
                ema_bearish_1h = False
            
            # ═══════════════════════════════════════════════════════
            # 🎯 WEAKNESS DETECTION (not full reversal - too strict)
            # Just need any sign the pump is losing steam
            # ═══════════════════════════════════════════════════════
            
            # RSI overbought or cooling off (TIGHTENED)
            rsi_extended = rsi_5m > 68  # Tightened from 65
            rsi_cooling = rsi_5m < 50 and change_24h > 12  # Tightened: RSI dropped more + higher price
            
            # Price weakening
            ema_converging = abs(ema_spread) < 0.5  # EMAs getting close
            below_recent_high = distance_from_high < -3  # Pulled back 3%+ from high
            
            weakness_signs = sum([
                ema_bearish,  # EMA9 < EMA21 (rare on gainers)
                has_lower_highs,  # Making lower highs
                has_lower_lows,  # Making lower lows
                ema_flattening and red_count >= 2,  # EMAs converging + selling
                rsi_extended,  # RSI overbought
                rsi_cooling,  # RSI dropped significantly
                ema_converging and red_count >= 2,  # Momentum fading
                below_recent_high,  # Already pulled back from high
            ])
            
            entry_quality_signs = sum([
                has_rejection_wick,  # Upper wick rejection
                red_count >= 2,  # Recent selling pressure
                distance_from_high < -2,  # Pulled back from high
                price_to_ema9 < 2.0,  # Not chasing too high above EMA
                rsi_5m < 70,  # Not at absolute peak
            ])
            
            weakness_required = 2 if is_dump_mode else 3
            entry_required = 2 if is_dump_mode else 3
            
            h1_bearish_bonus = 0
            if candles_1h and len(candles_1h) >= 14:
                closes_1h = [float(c[4]) for c in candles_1h]
                rsi_1h = self._calculate_rsi(closes_1h, 14)
                ema9_1h = self._calculate_ema(closes_1h, 9)
                ema21_1h = self._calculate_ema(closes_1h, 21) if len(closes_1h) >= 21 else ema9_1h
                if ema9_1h < ema21_1h:
                    h1_bearish_bonus = 1
                    weakness_signs += 1
                if rsi_1h > 72:
                    h1_bearish_bonus += 1
                    weakness_signs += 1
            
            logger.info(f"  📊 {symbol} analysis: weakness={weakness_signs} entry={entry_quality_signs} 1h_bonus={h1_bearish_bonus} | RSI:{rsi_5m:.0f} EMA:{'↘' if ema_bearish else '↗'} LH:{has_lower_highs} LL:{has_lower_lows} Wick:{has_rejection_wick} Red:{red_count}")
            
            if weakness_signs < weakness_required:
                logger.info(f"  ⏭️ {symbol} - Insufficient weakness signs {weakness_signs}/{weakness_required} (EMA={ema_bearish}, LH={has_lower_highs}, LL={has_lower_lows}, RSI={rsi_5m:.0f})")
                return None
            
            if entry_quality_signs < entry_required:
                logger.info(f"  ⏭️ {symbol} - Insufficient entry quality {entry_quality_signs}/{entry_required} (Wick={has_rejection_wick}, Red={red_count}, Dist={distance_from_high:.1f}%)")
                return None
            
            mode_label = "🔴 DUMP" if is_dump_mode else "📉"
            logger.info(f"  {mode_label} {symbol} - WEAKNESS SHORT: +{change_24h:.1f}% | {weakness_signs} weakness signs | {entry_quality_signs} entry signs")
            logger.info(f"     EMA: {'bearish' if ema_bearish else 'bullish'} | LH: {has_lower_highs} | LL: {has_lower_lows} | Wick: {has_rejection_wick}")
            
            # ═══════════════════════════════════════════════════════
            # AI VALIDATION
            # ═══════════════════════════════════════════════════════
            try:
                btc_change = await self._get_btc_24h_change()
            except:
                btc_change = 0.0
            
            coin_data_for_ai = {
                'symbol': symbol,
                'change_24h': change_24h,
                'rsi_5m': rsi_5m,
                'rsi_1h': rsi_1h,
                'volume_ratio': volume_ratio,
                'ema_bearish': ema_bearish,
                'ema_bearish_1h': ema_bearish_1h,
                'price_to_ema9': price_to_ema9,
                'distance_from_high': distance_from_high,
                'has_lower_highs': has_lower_highs,
                'has_lower_lows': has_lower_lows,
                'has_rejection_wick': has_rejection_wick,
                'red_candles_5': red_count,
                'btc_change': btc_change,
                'volume_24h': volume_24h,
                'weakness_signs': weakness_signs,
                'entry_quality_signs': entry_quality_signs
            }
            
            # Describe last 3 candles
            candle_desc = []
            for c in candles_5m[-3:]:
                o, h, l, cl = float(c[1]), float(c[2]), float(c[3]), float(c[4])
                candle_type = "GREEN" if cl > o else "RED"
                body_pct = abs((cl - o) / o) * 100 if o > 0 else 0
                candle_desc.append(f"{candle_type} {body_pct:.1f}%")
            
            sr_levels = {}
            try:
                from app.services.enhanced_ta import find_support_resistance
                c_highs = [float(c[2]) for c in candles_5m]
                c_lows = [float(c[3]) for c in candles_5m]
                c_closes = [float(c[4]) for c in candles_5m]
                sr_levels = find_support_resistance(c_highs, c_lows, c_closes, current_price)
            except Exception as e:
                logger.warning(f"S/R calculation failed for {symbol}: {e}")
            
            candle_data = {
                'last_3_candles': ', '.join(candle_desc),
                'support_resistance': sr_levels,
            }
            
            ai_result = await ai_validate_short_signal(coin_data_for_ai, candle_data)
            
            if not ai_result or not ai_result.get('approved', False):
                reason = ai_result.get('reasoning', 'No reason') if ai_result else 'AI error'
                logger.info(f"  ❌ {symbol} - AI REJECTED SHORT: {reason}")
                return None
            
            # AI approved - use its levels
            tp_percent = ai_result.get('tp_percent', 5.0)
            sl_percent = min(ai_result.get('sl_percent', 3.0), 3.0)  # Cap at 3% (60% max loss at 20x)
            ai_quality = ai_result.get('entry_quality', 'A')
            ai_confidence = ai_result.get('confidence', 7)
            ai_reasoning = ai_result.get('reasoning', 'AI approved')
            
            confidence = 90 if ai_quality in ['A+', 'A'] else 85
            
            reason_parts = [
                f"🔴 SHORT [{ai_quality}]",
                f"+{change_24h:.1f}% gainer",
                f"RSI {rsi_5m:.0f}",
                f"{weakness_signs} weakness signs"
            ]
            
            logger.info(f"{symbol} ✅ SHORT SIGNAL: {ai_quality} | +{change_24h:.1f}% | RSI {rsi_5m:.0f} | TP {tp_percent}% SL {sl_percent}%")
            
            return {
                'direction': 'SHORT',
                'confidence': confidence,
                'entry_price': current_price,
                'strategy': 'NORMAL_SHORT',
                'tp_percent': tp_percent,
                'sl_percent': sl_percent,
                'ai_quality': ai_quality,
                'ai_confidence': ai_confidence,
                'ai_reasoning': ai_reasoning,
                'rsi': rsi_5m,
                'volume_ratio': volume_ratio,
                'reason': ' | '.join(reason_parts)
            }
            
        except Exception as e:
            logger.warning(f"Error analyzing normal short for {symbol}: {e}")
            return None
    
    async def validate_fresh_5m_pump(self, symbol: str) -> Optional[Dict]:
        """
        Validate ULTRA-EARLY 5-minute pump (5%+ green candle in last 5min)
        
        ⚡ TIER 1 - ULTRA-EARLY DETECTION ⚡
        Catches pumps in the first 5-10 minutes!
        
        Requirements:
        - Most recent 5m candle is 5%+ green (close > open)
        - Volume 2.5x+ average of previous 3 candles (relaxed from 3.0x)
        - Candle is fresh (within 10 minutes, relaxed from 7)
        
        Returns:
            {
                'is_fresh_pump': bool,
                'tier': '5m',
                'candle_change_percent': float,
                'volume_ratio': float,
                'candle_close_time': int
            }
        """
        try:
            from datetime import datetime
            
            # Fetch last 5 5m candles (need 4 for volume average)
            candles_5m = await self.fetch_candles(symbol, '5m', limit=6)
            
            if len(candles_5m) < 4:
                return None
            
            # Most recent candle
            latest_candle = candles_5m[-1]
            timestamp, open_price, high, low, close_price, volume = latest_candle
            
            # Check 1: Is candle fresh? (within 7 minutes)
            # CRITICAL FIX: timestamp is open_time, add 5min interval to get close_time
            candle_close_time = datetime.fromtimestamp((timestamp + 300000) / 1000)  # +5min in ms
            now = datetime.utcnow()
            age_minutes = (now - candle_close_time).total_seconds() / 60
            
            if age_minutes > 10:  # Candle older than 10 minutes = stale (relaxed from 7)
                return {'is_fresh_pump': False, 'reason': 'stale_candle'}
            
            # Check 2: Is it a green candle AND 5%+ gain?
            if close_price <= open_price:
                return {'is_fresh_pump': False, 'reason': 'not_green_candle'}
            
            candle_change_percent = ((close_price - open_price) / open_price) * 100
            
            if candle_change_percent < 3.0:  # 🔥 RELAXED: 3%+ (was 5%+)
                return {'is_fresh_pump': False, 'reason': 'insufficient_pump', 'change': candle_change_percent}
            
            # Check 3: Volume 1.5x+ average of previous 3 candles (RELAXED for more signals!)
            prev_volumes = [candles_5m[-4][5], candles_5m[-3][5], candles_5m[-2][5]]
            avg_volume = sum(prev_volumes) / len(prev_volumes)
            
            volume_ratio = volume / avg_volume if avg_volume > 0 else 0
            
            if volume_ratio < 1.5:  # 1.5x volume required for fresh pump detection
                return {'is_fresh_pump': False, 'reason': 'low_volume', 'volume_ratio': volume_ratio}
            
            # ✅ ULTRA-EARLY PUMP DETECTED!
            logger.info(f"⚡ {symbol} ULTRA-EARLY 5m PUMP: +{candle_change_percent:.1f}% with {volume_ratio:.1f}x volume")
            
            return {
                'is_fresh_pump': True,
                'tier': '5m',
                'candle_change_percent': round(candle_change_percent, 2),
                'volume_ratio': round(volume_ratio, 2),
                'candle_close_time': timestamp,
                'candle_age_minutes': round(age_minutes, 1)
            }
            
        except Exception as e:
            logger.error(f"Error validating 5m pump for {symbol}: {e}")
            return None
    
    async def validate_fresh_15m_pump(self, symbol: str) -> Optional[Dict]:
        """
        Validate EARLY 15-minute pump (7%+ green candle in last 15min)
        
        🔥 TIER 2 - EARLY DETECTION 🔥
        Catches pumps in the first 15-20 minutes!
        
        Requirements:
        - Most recent 15m candle is 7%+ green (close > open)
        - Volume 2.5x+ average of previous 2-3 candles
        - Candle is fresh (within 20 minutes)
        
        Returns:
            {
                'is_fresh_pump': bool,
                'tier': '15m',
                'candle_change_percent': float,
                'volume_ratio': float,
                'candle_close_time': int
            }
        """
        try:
            from datetime import datetime
            
            # Fetch last 4 15m candles (need 3 for volume average)
            candles_15m = await self.fetch_candles(symbol, '15m', limit=5)
            
            if len(candles_15m) < 3:
                return None
            
            # Most recent candle
            latest_candle = candles_15m[-1]
            timestamp, open_price, high, low, close_price, volume = latest_candle
            
            # Check 1: Is candle fresh? (within 20 minutes)
            # CRITICAL FIX: timestamp is open_time, add 15min interval to get close_time
            candle_close_time = datetime.fromtimestamp((timestamp + 900000) / 1000)  # +15min in ms
            now = datetime.utcnow()
            age_minutes = (now - candle_close_time).total_seconds() / 60
            
            if age_minutes > 20:  # Candle older than 20 minutes = stale
                return {'is_fresh_pump': False, 'reason': 'stale_candle'}
            
            # Check 2: Is it a green candle AND 5%+ gain? (RELAXED from 7%)
            if close_price <= open_price:
                return {'is_fresh_pump': False, 'reason': 'not_green_candle'}
            
            candle_change_percent = ((close_price - open_price) / open_price) * 100
            
            if candle_change_percent < 5.0:  # 🔥 RELAXED: 5%+ (was 7%+)
                return {'is_fresh_pump': False, 'reason': 'insufficient_pump', 'change': candle_change_percent}
            
            # Check 3: Volume 1.5x+ average of previous 2 candles (RELAXED from 2.5x)
            prev_volumes = [candles_15m[-3][5], candles_15m[-2][5]]
            avg_volume = sum(prev_volumes) / len(prev_volumes)
            
            volume_ratio = volume / avg_volume if avg_volume > 0 else 0
            
            if volume_ratio < 1.5:  # 1.5x volume required for fresh pump detection
                return {'is_fresh_pump': False, 'reason': 'low_volume', 'volume_ratio': volume_ratio}
            
            # ✅ EARLY PUMP DETECTED!
            logger.info(f"🔥 {symbol} EARLY 15m PUMP: +{candle_change_percent:.1f}% with {volume_ratio:.1f}x volume")
            
            return {
                'is_fresh_pump': True,
                'tier': '15m',
                'candle_change_percent': round(candle_change_percent, 2),
                'volume_ratio': round(volume_ratio, 2),
                'candle_close_time': timestamp,
                'candle_age_minutes': round(age_minutes, 1)
            }
            
        except Exception as e:
            logger.error(f"Error validating 15m pump for {symbol}: {e}")
            return None
    
    async def validate_fresh_30m_pump(self, symbol: str) -> Optional[Dict]:
        """
        Validate FRESH 30-minute pump (5%+ green candle in last 30min)
        
        Requirements:
        - Most recent 30m candle is 5%+ green (close > open)  
        - Volume 1.5x+ average of previous 2-3 candles
        - Candle is fresh (within 35 minutes)
        
        Returns:
            {
                'is_fresh_pump': bool,
                'candle_change_percent': float,
                'volume_ratio': float,
                'candle_close_time': int
            }
        """
        try:
            from datetime import datetime, timedelta
            
            # Fetch last 3 30m candles (need 3 for volume average)
            candles_30m = await self.fetch_candles(symbol, '30m', limit=5)
            
            if len(candles_30m) < 3:
                return None
            
            # Most recent candle
            latest_candle = candles_30m[-1]
            timestamp, open_price, high, low, close_price, volume = latest_candle
            
            # Check 1: Is candle fresh? (within 35 minutes)
            # CRITICAL FIX: timestamp is open_time, add 30min interval to get close_time
            candle_close_time = datetime.fromtimestamp((timestamp + 1800000) / 1000)  # +30min in ms
            now = datetime.utcnow()
            age_minutes = (now - candle_close_time).total_seconds() / 60
            
            if age_minutes > 35:  # Candle older than 35 minutes = stale
                logger.debug(f"{symbol} 30m candle too old: {age_minutes:.1f} min")
                return {'is_fresh_pump': False, 'reason': 'stale_candle'}
            
            # Check 2: Is it a green candle AND 7%+ gain? (RELAXED from 10%)
            if close_price <= open_price:
                return {'is_fresh_pump': False, 'reason': 'not_green_candle'}
            
            candle_change_percent = ((close_price - open_price) / open_price) * 100
            
            if candle_change_percent < 5.0:  # 🔥 ULTRA RELAXED: 5%+ (catch early 5-20% gainers!)
                logger.debug(f"{symbol} 30m candle only +{candle_change_percent:.1f}% (need 5%+)")
                return {'is_fresh_pump': False, 'reason': 'insufficient_pump', 'change': candle_change_percent}
            
            # Check 3: Volume 1.5x+ average of previous 2 candles (RELAXED from 2.0x)
            prev_volumes = [candles_30m[-3][5], candles_30m[-2][5]]  # Previous 2 candles' volume
            avg_volume = sum(prev_volumes) / len(prev_volumes)
            
            volume_ratio = volume / avg_volume if avg_volume > 0 else 0
            
            if volume_ratio < 1.5:  # 🔥 RELAXED: 1.5x (was 2.0x)
                logger.debug(f"{symbol} 30m volume only {volume_ratio:.1f}x (need 1.5x+)")
                return {'is_fresh_pump': False, 'reason': 'low_volume', 'volume_ratio': volume_ratio}
            
            # ✅ All checks passed - FRESH PUMP!
            logger.info(f"✅ {symbol} FRESH 30m PUMP: +{candle_change_percent:.1f}% with {volume_ratio:.1f}x volume")
            
            return {
                'is_fresh_pump': True,
                'tier': '30m',
                'candle_change_percent': round(candle_change_percent, 2),
                'volume_ratio': round(volume_ratio, 2),
                'candle_close_time': timestamp,
                'candle_age_minutes': round(age_minutes, 1)
            }
            
        except Exception as e:
            logger.error(f"Error validating 30m pump for {symbol}: {e}")
            return None
    
    async def detect_realtime_breakouts(self, max_symbols: int = 30) -> List[Dict]:
        """
        🚀 REAL-TIME BREAKOUT DETECTOR - Catches pumps BEFORE they hit top-gainer lists!
        
        This is the key to EARLY entries. Instead of scanning 24h top gainers (already late),
        we scan ALL symbols for FRESH 1m volume/price spikes happening RIGHT NOW.
        
        Detection criteria (on 1m candles):
        1. Volume spike: Current 1m candle has 3x+ volume vs average of last 10 candles
        2. Price velocity: 1%+ price move in last 1-3 minutes (momentum building)
        3. Trend confirmation: Price above EMA9 on 1m (fresh breakout)
        4. Freshness: Candle must be within last 2 minutes (live action!)
        
        Returns:
            List of breakout candidates sorted by volume spike strength:
            {symbol, volume_ratio, price_velocity, current_price, breakout_type}
        """
        try:
            logger.info("🔍 REALTIME BREAKOUT SCAN: Checking ALL symbols for fresh 1m volume spikes...")
            
            # ═══════════════════════════════════════════════════════
            # 🔥 USE BINANCE + MEXC FOR SCANNING (not Bitunix - unreliable!)
            # Same dual data source architecture as SHORT scanner
            # ═══════════════════════════════════════════════════════
            
            merged_symbols = {}  # symbol -> volume_usdt
            
            # === SOURCE 1: BINANCE FUTURES (Primary) ===
            binance_count = 0
            try:
                binance_url = "https://fapi.binance.com/fapi/v1/ticker/24hr"
                response = await self.client.get(binance_url, timeout=10)
                response.raise_for_status()
                binance_tickers = response.json()
                
                for ticker in binance_tickers:
                    symbol = ticker.get('symbol', '')
                    if symbol.endswith('USDT'):
                        volume_usdt = float(ticker.get('quoteVolume', 0))
                        if volume_usdt >= self.min_volume_usdt:
                            merged_symbols[symbol] = volume_usdt
                            binance_count += 1
            except Exception as e:
                logger.warning(f"⚠️ Binance API error in breakout scan: {e}")
            
            # === SOURCE 2: MEXC FUTURES (Secondary - for coins not on Binance) ===
            mexc_added = 0
            try:
                mexc_url = "https://contract.mexc.com/api/v1/contract/ticker"
                response = await self.client.get(mexc_url, timeout=10)
                response.raise_for_status()
                mexc_data = response.json()
                mexc_tickers = mexc_data.get('data', []) if isinstance(mexc_data, dict) else mexc_data
                
                for ticker in mexc_tickers:
                    raw_symbol = ticker.get('symbol', '')
                    symbol = raw_symbol.replace('_', '')
                    if symbol.endswith('USDT') and symbol not in merged_symbols:
                        volume_usdt = float(ticker.get('amount24', 0))
                        if volume_usdt >= self.min_volume_usdt:
                            merged_symbols[symbol] = volume_usdt
                            mexc_added += 1
            except Exception as e:
                logger.warning(f"⚠️ MEXC API error in breakout scan: {e}")
            
            # === FILTER TO BITUNIX TRADEABLE SYMBOLS ===
            bitunix_url = f"{self.base_url}/api/v1/futures/market/tickers"
            bitunix_response = await self.client.get(bitunix_url, timeout=10)
            bitunix_data = bitunix_response.json()
            bitunix_symbols = set()
            if isinstance(bitunix_data, dict) and bitunix_data.get('data'):
                for t in bitunix_data.get('data', []):
                    bitunix_symbols.add(t.get('symbol', ''))
            
            # Only keep symbols tradeable on Bitunix (with volume data)
            all_symbols = []
            symbol_volumes = {}  # symbol -> 24h volume
            for symbol, volume_usdt in merged_symbols.items():
                if symbol in bitunix_symbols:
                    formatted_symbol = symbol.replace('USDT', '/USDT')
                    all_symbols.append(formatted_symbol)
                    symbol_volumes[formatted_symbol] = volume_usdt
            
            logger.info(f"📊 Scanning {len(all_symbols)} symbols for breakouts (Binance={binance_count}, MEXC=+{mexc_added}, Bitunix={len(bitunix_symbols)} tradeable)")
            
            breakout_candidates = []
            
            # Scan ALL symbols (no limit - catch every breakout!)
            batch_size = 15
            for i in range(0, len(all_symbols), batch_size):
                batch = all_symbols[i:i + batch_size]
                
                # Parallel fetch 1m candles for batch
                tasks = [self.fetch_candles(symbol, '1m', limit=15) for symbol in batch]
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                for symbol, candles in zip(batch, results):
                    if isinstance(candles, Exception) or not candles or len(candles) < 12:
                        continue
                    
                    try:
                        # Latest 1m candle
                        latest = candles[-1]
                        timestamp, open_price, high, low, close_price, volume = latest
                        
                        # Check freshness - candle close time = open_time + 60 seconds
                        # For a 1m candle at 12:00, close_time is 12:01
                        # If current time is 12:02, the candle is 1 minute old (fresh!)
                        from datetime import datetime
                        candle_open_time = datetime.fromtimestamp(timestamp / 1000)
                        candle_close_time = datetime.fromtimestamp((timestamp + 60000) / 1000)
                        now = datetime.utcnow()
                        
                        # Age = how long since candle CLOSED
                        age_seconds = (now - candle_close_time).total_seconds()
                        
                        # Accept candles that are still open (age < 0) or closed within 3 min
                        if age_seconds > 180:  # Skip if closed more than 3 minutes ago
                            continue
                        
                        # Volume spike detection (2.5x+ average for quality)
                        prev_volumes = [c[5] for c in candles[-11:-1]]
                        avg_volume = sum(prev_volumes) / len(prev_volumes) if prev_volumes else 0
                        volume_ratio = volume / avg_volume if avg_volume > 0 else 0
                        
                        # Calculate 1m quote volume (volume in USDT terms)
                        current_1m_volume_usdt = volume * close_price
                        avg_1m_volume_usdt = avg_volume * close_price
                        
                        if volume_ratio < 2.5:  # Need 2.5x volume spike (loosened from 3.5x)
                            continue
                        
                        # Price velocity (current candle change %)
                        candle_change = ((close_price - open_price) / open_price) * 100 if open_price > 0 else 0
                        
                        if candle_change < 0.5:  # Need 0.5%+ momentum (loosened from 0.8%)
                            continue
                        
                        # EMA9 check on 1m (price should be above for breakout)
                        closes = [c[4] for c in candles]
                        ema9 = self._calculate_ema(closes, 9)
                        
                        if close_price < ema9:  # Not a breakout
                            continue
                        
                        # Calculate 3-candle momentum (last 3 minutes velocity)
                        price_3m_ago = candles[-3][4] if len(candles) >= 3 else open_price
                        velocity_3m = ((close_price - price_3m_ago) / price_3m_ago) * 100 if price_3m_ago > 0 else 0
                        
                        # Require 0.8%+ 3-minute velocity (loosened from 1.2%)
                        if velocity_3m < 0.8:
                            continue
                        
                        # Determine breakout strength - Include BUILDING for more candidates
                        if volume_ratio >= 5.0 and candle_change >= 1.2:
                            breakout_type = "EXPLOSIVE"
                        elif volume_ratio >= 3.5 and candle_change >= 0.8:
                            breakout_type = "STRONG"
                        elif volume_ratio >= 2.5 and candle_change >= 0.5:
                            breakout_type = "BUILDING"
                        else:
                            continue
                        
                        breakout_candidates.append({
                            'symbol': symbol,
                            'volume_ratio': round(volume_ratio, 1),
                            'candle_change': round(candle_change, 2),
                            'velocity_3m': round(velocity_3m, 2),
                            'current_price': close_price,
                            'ema9_distance': round(((close_price - ema9) / ema9) * 100, 2),
                            'breakout_type': breakout_type,
                            'age_seconds': round(age_seconds, 0),
                            'volume_24h': symbol_volumes.get(symbol, 0),  # 24h volume from Binance/MEXC
                            'current_1m_vol': round(current_1m_volume_usdt, 0),  # Current 1m USDT volume
                            'avg_1m_vol': round(avg_1m_volume_usdt, 0)  # Average 1m USDT volume
                        })
                        
                    except Exception as e:
                        continue
            
            # Sort by volume spike (strongest first)
            breakout_candidates.sort(key=lambda x: x['volume_ratio'], reverse=True)
            
            if breakout_candidates:
                logger.info(f"🚀 FOUND {len(breakout_candidates)} BREAKOUT CANDIDATES!")
                for bc in breakout_candidates[:5]:
                    logger.info(f"  ⚡ {bc['symbol']}: {bc['breakout_type']} | {bc['volume_ratio']}x vol | +{bc['candle_change']}% | vel: +{bc['velocity_3m']}%")
            else:
                logger.info("❌ No breakout candidates found in this scan")
            
            return breakout_candidates[:max_symbols]
            
        except Exception as e:
            logger.error(f"Error in realtime breakout detection: {e}")
            return []
    
    async def analyze_breakout_entry(self, symbol: str, breakout_data: Dict) -> Optional[Dict]:
        """
        🎯 PULLBACK-FIRST ENTRY - NEVER enter at top of green candle!
        
        Core principle: We detect breakouts, but WAIT for pullback THEN enter on resumption.
        This prevents buying tops and getting stopped out on natural retracements.
        
        Entry requirements (ALL must be true):
        1. Prior impulse detected (green candle with volume)
        2. Pullback occurred (at least 1 red candle touching EMA support)
        3. Resumption starting (current green candle after red)
        4. Current price near candle LOW, not HIGH (not buying the top)
        5. RSI cooled down (48-65, not overbought)
        
        Returns:
            Signal dict or None if entry conditions not met
        """
        try:
            logger.info(f"🎯 PULLBACK-FIRST ENTRY: {symbol} ({breakout_data['breakout_type']})")
            
            # Fetch 1m, 5m, and 15m candles for proper trend analysis
            candles_1m = await self.fetch_candles(symbol, '1m', limit=20)
            candles_5m = await self.fetch_candles(symbol, '5m', limit=30)
            candles_15m = await self.fetch_candles(symbol, '15m', limit=30)
            
            if len(candles_1m) < 15 or len(candles_5m) < 20 or len(candles_15m) < 20:
                logger.info(f"  ❌ {symbol} - Insufficient candle data")
                return None
            
            closes_1m = [c[4] for c in candles_1m]
            closes_5m = [c[4] for c in candles_5m]
            closes_15m = [c[4] for c in candles_15m]
            
            # Current candle data
            current_candle = candles_1m[-1]
            current_open = current_candle[1]
            current_high = current_candle[2]
            current_low = current_candle[3]
            current_close = current_candle[4]
            
            # EMAs for all timeframes
            ema9_1m = self._calculate_ema(closes_1m, 9)
            ema21_1m = self._calculate_ema(closes_1m, 21)
            ema9_5m = self._calculate_ema(closes_5m, 9)
            ema21_5m = self._calculate_ema(closes_5m, 21)
            ema9_15m = self._calculate_ema(closes_15m, 9)
            ema21_15m = self._calculate_ema(closes_15m, 21)
            
            # RSI
            rsi_5m = self._calculate_rsi(closes_5m, 14)
            
            # ═══════════════════════════════════════════════════════
            # CRITICAL CHECK 0a: NO LONGING COINS DOWN ON THE DAY
            # If coin is -5% or more on 24h, it's in a downtrend - don't touch!
            # ═══════════════════════════════════════════════════════
            # Calculate 24h change from 15m candles (approx 7.5 hours of data)
            # Use oldest vs newest close as proxy for trend direction
            price_oldest_15m = closes_15m[0]
            price_change_approx = ((current_close - price_oldest_15m) / price_oldest_15m) * 100
            
            if price_change_approx < -5.0:
                logger.info(f"  ❌ {symbol} - DOWN {price_change_approx:.1f}% on day - NOT longing a dump!")
                return None
            
            # ═══════════════════════════════════════════════════════
            # CRITICAL CHECK 0b: 15m TREND MUST BE BULLISH (ANTI-DOWNTREND)
            # This prevents longing small bounces in clear downtrends
            # ═══════════════════════════════════════════════════════
            if not (ema9_15m > ema21_15m):
                logger.info(f"  ❌ {symbol} - 15m DOWNTREND (EMA9 < EMA21) - NOT longing a bounce!")
                return None
            
            # Price must be above 15m EMA21 (not in downtrend territory)
            if current_close < ema21_15m:
                logger.info(f"  ❌ {symbol} - Price BELOW 15m EMA21 - in downtrend territory!")
                return None
            
            # ═══════════════════════════════════════════════════════
            # CRITICAL CHECK 1: Current candle must NOT be at its high
            # If close is in top 40% of candle range = buying the top = SKIP
            # TIGHTENED: Was 0.7, now 0.6 - don't enter in top 40%
            # ═══════════════════════════════════════════════════════
            candle_range = current_high - current_low
            if candle_range > 0:
                close_position = (current_close - current_low) / candle_range
                if close_position > 0.6:  # Close is in top 40% of range (TIGHTENED from 0.7)
                    logger.info(f"  ❌ {symbol} - Price at candle TOP ({close_position:.0%}) - would buy top!")
                    return None
            
            # ═══════════════════════════════════════════════════════
            # CRITICAL CHECK 2: Must have had a RED pullback candle recently
            # Looking for impulse → pullback → resumption pattern
            # ═══════════════════════════════════════════════════════
            candle_m1 = candles_1m[-1]  # Current
            candle_m2 = candles_1m[-2]  # 1 min ago
            candle_m3 = candles_1m[-3]  # 2 min ago
            candle_m4 = candles_1m[-4]  # 3 min ago
            candle_m5 = candles_1m[-5]  # 4 min ago
            
            # Check if candles are bullish (green) or bearish (red)
            c1_green = candle_m1[4] > candle_m1[1]
            c2_green = candle_m2[4] > candle_m2[1]
            c3_green = candle_m3[4] > candle_m3[1]
            c4_green = candle_m4[4] > candle_m4[1]
            c5_green = candle_m5[4] > candle_m5[1]
            
            # Count red (pullback) candles in last 4 candles before current
            red_count = sum([not c2_green, not c3_green, not c4_green, not c5_green])
            
            # Must have at least 2 red pullback candles for REAL pullback (STRICT)
            if red_count < 2:
                logger.info(f"  ❌ {symbol} - Weak pullback (only {red_count} red) - need 2+ red candles")
                return None
            
            # ═══════════════════════════════════════════════════════
            # CRITICAL CHECK 3: Current candle must be GREEN (resumption)
            # We enter on the resumption AFTER pullback, not during pullback
            # ═══════════════════════════════════════════════════════
            if not c1_green:
                logger.info(f"  ⏳ {symbol} - Pullback in progress (current red) - waiting for green")
                return None
            
            # ═══════════════════════════════════════════════════════
            # CRITICAL CHECK 4: Price must be near EMA support (not extended)
            # Entry near EMA = better R:R, natural support
            # ULTRA STRICT: 1.5% max distance from EMA
            # ═══════════════════════════════════════════════════════
            ema_distance = ((current_close - ema9_1m) / ema9_1m) * 100
            if ema_distance > 1.5:  # STRICT: Max 1.5% above EMA
                logger.info(f"  ❌ {symbol} - Extended {ema_distance:.1f}% above EMA (need ≤1.5%)")
                return None
            
            # ═══════════════════════════════════════════════════════
            # CRITICAL CHECK 5: Pullback must have touched EMA support
            # At least one of the red candles should have wicked near EMA
            # ═══════════════════════════════════════════════════════
            pullback_touched_ema = False
            for candle in [candle_m2, candle_m3, candle_m4]:
                candle_low = candle[3]
                # Low within 0.5% of EMA9 or below = touched support
                if candle_low <= ema9_1m * 1.005:
                    pullback_touched_ema = True
                    break
            
            if not pullback_touched_ema:
                logger.info(f"  ❌ {symbol} - Pullback didn't touch EMA support - shallow pullback REJECTED")
                return None  # TIGHTENED: No longer allow shallow pullbacks - must touch EMA
            
            # ═══════════════════════════════════════════════════════
            # CRITICAL CHECK 6: RSI must be in sweet spot
            # ULTRA STRICT: 45-60 only - cooled but with momentum
            # ═══════════════════════════════════════════════════════
            if not (45 <= rsi_5m <= 60):
                logger.info(f"  ❌ {symbol} - RSI {rsi_5m:.0f} out of range (need 45-60)")
                return None
            
            # ═══════════════════════════════════════════════════════
            # CRITICAL CHECK 7: 5m trend must be bullish
            # ═══════════════════════════════════════════════════════
            if not (ema9_5m > ema21_5m):
                logger.info(f"  ❌ {symbol} - 5m trend not bullish")
                return None
            
            # ═══════════════════════════════════════════════════════
            # CRITICAL CHECK 8: Minimum 24h volume for liquidity
            # ULTRA STRICT: $500K minimum volume
            # ═══════════════════════════════════════════════════════
            volume_24h = breakout_data.get('volume_24h', 0)
            if volume_24h < 500000:
                logger.info(f"  ❌ {symbol} - Low liquidity ${volume_24h:,.0f} (need $500K+)")
                return None
            
            # ═══════════════════════════════════════════════════════
            # ENTRY CONFIRMED: Pullback complete, resumption starting
            # ═══════════════════════════════════════════════════════
            
            # Determine entry pattern based on pullback depth
            # Note: EMA touch is now required, so all entries are EMA_BOUNCE or DEEP_PULLBACK
            if red_count >= 3:
                entry_pattern = "DEEP_PULLBACK"
            else:
                entry_pattern = "EMA_BOUNCE"
            
            logger.info(f"  ✅ {symbol} PULLBACK ENTRY CONFIRMED!")
            logger.info(f"     Pattern: {entry_pattern} | RSI: {rsi_5m:.0f} | EMA dist: {ema_distance:.1f}%")
            logger.info(f"     Pullback: {red_count} red candles | EMA touch: {pullback_touched_ema}")
            logger.info(f"     Candle position: {close_position:.0%} (not at top ✓)")
            
            return {
                'direction': 'LONG',
                'entry_price': current_close,
                'confidence': 85 if red_count >= 3 else 80,  # Higher confidence for deeper pullbacks
                'reason': f"QUALITY ENTRY: {entry_pattern} | {red_count} red + EMA touch | RSI {rsi_5m:.0f} | {ema_distance:.1f}% from EMA",
                'breakout_type': breakout_data['breakout_type'],
                'volume_ratio': breakout_data['volume_ratio'],
                'entry_pattern': entry_pattern
            }
            
        except Exception as e:
            logger.error(f"Error analyzing breakout entry for {symbol}: {e}")
            return None
    
    async def generate_breakout_long_signal(self) -> Optional[Dict]:
        """
        🚀 LONG STRATEGY: Real-time breakout detection with pullback tracking
        
        KEY FIX: Tracks breakout candidates and re-evaluates them on subsequent
        scans to catch the pullback entry (can't catch pullback immediately!)
        
        Flow:
        1. Check global LONG cooldown
        2. Clean up expired pending candidates (>10 min old)
        3. Re-evaluate PENDING candidates first (they already had breakout, now check pullback)
        4. Detect NEW breakouts and add to pending cache
        5. Return signal when pullback entry is found
        
        Returns:
            Signal dict matching existing format, or None
        """
        global last_long_signal_time, longs_symbol_cooldown, pending_breakout_candidates
        
        try:
            logger.info("═══════════════════════════════════════════════════════")
            logger.info("🚀 REALTIME BREAKOUT SCANNER - Tracking Pullback Entries!")
            logger.info("═══════════════════════════════════════════════════════")
            
            # 🔒 CHECK GLOBAL LONG COOLDOWN
            if last_long_signal_time:
                hours_since_last = (datetime.utcnow() - last_long_signal_time).total_seconds() / 3600
                if hours_since_last < LONG_GLOBAL_COOLDOWN_HOURS:
                    remaining = LONG_GLOBAL_COOLDOWN_HOURS - hours_since_last
                    logger.info(f"⏳ LONG COOLDOWN: {remaining:.1f}h remaining (last signal {hours_since_last:.1f}h ago)")
                    return None
            
            # ═══════════════════════════════════════════════════════
            # STEP 1: Clean up expired candidates (>10 min old)
            # ═══════════════════════════════════════════════════════
            now = datetime.utcnow()
            expired = [sym for sym, data in pending_breakout_candidates.items() 
                      if (now - data['detected_at']).total_seconds() > BREAKOUT_CANDIDATE_TIMEOUT_MINUTES * 60]
            for sym in expired:
                logger.info(f"🗑️ Removing expired candidate: {sym} (>{BREAKOUT_CANDIDATE_TIMEOUT_MINUTES}min old)")
                del pending_breakout_candidates[sym]
            
            logger.info(f"📋 Pending breakout candidates: {len(pending_breakout_candidates)}")
            for sym, data in pending_breakout_candidates.items():
                age_min = (now - data['detected_at']).total_seconds() / 60
                logger.info(f"   • {sym}: {data['breakout_data']['breakout_type']} | {age_min:.1f}min old | checks: {data['checks']}")
            
            # ═══════════════════════════════════════════════════════
            # STEP 2: Re-evaluate PENDING candidates (they had breakout, check pullback now)
            # ═══════════════════════════════════════════════════════
            for symbol, candidate_data in list(pending_breakout_candidates.items()):
                logger.info(f"  🔄 Re-checking pending: {symbol}...")
                
                # 🚫 Check blacklist
                normalized = symbol.replace('/USDT', '').replace('USDT', '')
                if normalized in BLACKLISTED_SYMBOLS or symbol in BLACKLISTED_SYMBOLS:
                    logger.info(f"🚫 {symbol} BLACKLISTED - removing from pending")
                    del pending_breakout_candidates[symbol]
                    continue
                
                # Check per-symbol WINDOW cooldown (skip next window if traded)
                if is_symbol_in_next_window_cooldown(symbol):
                        del pending_breakout_candidates[symbol]  # Remove from pending
                        continue
                
                # Check if we already have an open position on this symbol
                from app.models import Trade
                from app.database import SessionLocal
                db_check = SessionLocal()
                try:
                    open_pos = db_check.query(Trade).filter(
                        Trade.symbol == symbol,
                        Trade.status == 'open'
                    ).first()
                    if open_pos:
                        logger.info(f"    🚫 {symbol} already has OPEN position - skipping")
                        del pending_breakout_candidates[symbol]
                        continue
                finally:
                    db_check.close()
                
                # Increment check counter
                candidate_data['checks'] += 1
                
                # Analyze for pullback entry
                entry = await self.analyze_breakout_entry(symbol, candidate_data['breakout_data'])
                
                if entry and entry['direction'] == 'LONG':
                    # Found valid pullback entry!
                    entry_price = entry['entry_price']
                    
                    # LONG @ 20x leverage - SINGLE TP
                    # TP: 3.35% = 67% profit at 20x
                    # SL: 3.25% = 65% loss at 20x
                    stop_loss = entry_price * (1 - 3.25 / 100)  # 65% loss at 20x
                    take_profit_1 = entry_price * (1 + 3.35 / 100)  # 67% profit at 20x
                    take_profit_2 = None  # Single TP only
                    
                    breakout_data = candidate_data['breakout_data']
                    signal = {
                        'symbol': symbol,
                        'direction': 'LONG',
                        'entry_price': entry_price,
                        'stop_loss': stop_loss,
                        'take_profit_1': take_profit_1,
                        'take_profit_2': take_profit_2,
                        'take_profit_3': None,
                        'leverage': 20,
                        'confidence': entry['confidence'],
                        'reasoning': entry['reason'],
                        'mode': 'BREAKOUT',
                        'breakout_type': breakout_data['breakout_type'],
                        'volume_ratio': breakout_data['volume_ratio'],
                        '24h_change': breakout_data.get('velocity_3m', 0),
                        '24h_volume': breakout_data.get('volume_24h', 0),
                        'current_1m_vol': breakout_data.get('current_1m_vol', 0),
                        'avg_1m_vol': breakout_data.get('avg_1m_vol', 0)
                    }
                    
                    logger.info(f"✅ PULLBACK ENTRY FOUND: {symbol}")
                    logger.info(f"   Entry: ${entry_price:.6f} | SL: ${stop_loss:.6f} | TP1: ${take_profit_1:.6f}")
                    logger.info(f"   Waited {candidate_data['checks']} checks for pullback")
                    
                    # 🤖 AI Enhancement - optimize signal levels
                    signal = await enhance_signal_with_ai(signal)
                    
                    # Remove from pending and update cooldowns
                    del pending_breakout_candidates[symbol]
                    last_long_signal_time = datetime.utcnow()
                    add_symbol_window_cooldown(symbol)  # Mark as traded in current window
                    
                    return signal
            
            # ═══════════════════════════════════════════════════════
            # STEP 3: Detect NEW breakouts and add to pending cache
            # ═══════════════════════════════════════════════════════
            logger.info("🔍 Scanning for NEW breakouts...")
            breakouts = await self.detect_realtime_breakouts(max_symbols=20)
            
            if breakouts:
                for breakout in breakouts:
                    symbol = breakout['symbol']
                    
                    # 🚫 Check blacklist
                    normalized = symbol.replace('/USDT', '').replace('USDT', '')
                    if normalized in BLACKLISTED_SYMBOLS or symbol in BLACKLISTED_SYMBOLS:
                        logger.info(f"🚫 {symbol} BLACKLISTED - skipping")
                        continue
                    
                    # Skip if already pending or on cooldown
                    if symbol in pending_breakout_candidates:
                        continue
                    if is_symbol_in_next_window_cooldown(symbol):
                        continue
                    
                    # Check liquidity before adding
                    liquidity = await self.check_liquidity(symbol)
                    if not liquidity['is_liquid']:
                        logger.info(f"    ❌ {symbol} - {liquidity['reason']}")
                        continue
                    
                    # ANTI-TOP FILTERS: Don't enter coins that already ran
                    candles_15m = await self.fetch_candles(symbol, '15m', limit=10)
                    candles_4h = await self.fetch_candles(symbol, '4h', limit=21)
                    
                    # Check 15m impulse
                    if len(candles_15m) >= 2:
                        impulse_15m = ((candles_15m[-1][4] - candles_15m[-2][4]) / candles_15m[-2][4]) * 100
                        if impulse_15m > 6:
                            logger.info(f"    ❌ {symbol} TOO LATE: {impulse_15m:.1f}% in 15m")
                            continue
                    
                    # Check 1h impulse
                    if len(candles_15m) >= 5:
                        impulse_1h = ((candles_15m[-1][4] - candles_15m[-5][4]) / candles_15m[-5][4]) * 100
                        if impulse_1h > 12:
                            logger.info(f"    ❌ {symbol} TOO LATE: {impulse_1h:.1f}% in 1h")
                            continue
                    
                    # Check 4h EMA extension - 8% max
                    if len(candles_4h) >= 21:
                        closes_4h = [c[4] for c in candles_4h]
                        ema21_4h = self._calculate_ema(closes_4h, 21)
                        current_price = closes_4h[-1]
                        extension_4h = ((current_price - ema21_4h) / ema21_4h) * 100
                        
                        if extension_4h > 8:
                            logger.info(f"    ❌ {symbol} EXTENDED: {extension_4h:.1f}% above 4h EMA (max 8%)")
                            continue
                        
                        # Check consecutive green 4h candles
                        recent_4h = candles_4h[-6:]
                        green_count = sum(1 for c in recent_4h if c[4] > c[1])
                        if green_count >= 4:
                            logger.info(f"    ❌ {symbol} SUSTAINED PUMP: {green_count}/6 green 4h candles")
                            continue
                    
                    # Add to pending cache
                    pending_breakout_candidates[symbol] = {
                        'detected_at': datetime.utcnow(),
                        'breakout_data': breakout,
                        'checks': 0
                    }
                    logger.info(f"  ➕ Added to pending: {symbol} ({breakout['breakout_type']} | {breakout['volume_ratio']}x vol)")
            else:
                logger.info("❌ No new breakouts detected")
            
            logger.info(f"📊 Total pending candidates: {len(pending_breakout_candidates)}")
            logger.info("⏳ Waiting for pullback entries on next scan...")
            return None
            
        except Exception as e:
            logger.error(f"Error in breakout long signal generation: {e}")
            return None
    
    async def generate_momentum_long_signal(self) -> Optional[Dict]:
        """
        🚀 MOMENTUM LONG: Top gainers showing strong momentum
        
        Targets coins showing EARLY momentum for timely entries.
        Anti-top filters: 12% 15m cap, 20% 1h cap, 22% 4h EMA extension,
        30% max 24h change, RSI <72, EMA distance <3%.
        
        Returns signal dict or None
        """
        global last_long_signal_time, longs_symbol_cooldown
        
        try:
            logger.info("═══════════════════════════════════════════════════════")
            logger.info("🔥 MOMENTUM LONG SCANNER - Top Gainers")
            logger.info("═══════════════════════════════════════════════════════")
            
            # Check global cooldown
            if last_long_signal_time:
                hours_since_last = (datetime.utcnow() - last_long_signal_time).total_seconds() / 3600
                if hours_since_last < LONG_GLOBAL_COOLDOWN_HOURS:
                    remaining = LONG_GLOBAL_COOLDOWN_HOURS - hours_since_last
                    logger.info(f"⏳ LONG COOLDOWN: {remaining:.1f}h remaining")
                    return None
            
            top_gainers = await self.get_early_pumpers(limit=50, min_change=-50.0, max_change=30.0)
            
            if not top_gainers:
                logger.info("❌ No coins found")
                return None
            
            logger.info(f"📊 Scanning {len(top_gainers)} coins for momentum")
            
            for gainer in top_gainers:
                symbol = gainer['symbol']
                change_24h = gainer['change_percent']
                volume_24h = gainer.get('volume_24h', 0)
                
                # Skip blacklisted
                normalized = symbol.replace('/USDT', '').replace('USDT', '')
                if normalized in BLACKLISTED_SYMBOLS or symbol in BLACKLISTED_SYMBOLS:
                    continue
                
                # Check per-symbol WINDOW cooldown (skip next window if traded)
                if is_symbol_in_next_window_cooldown(symbol):
                    continue
                
                # Check if we already traded this symbol TODAY (persists across redeploys)
                from app.models import Trade
                from app.database import SessionLocal
                from sqlalchemy import func
                db_check = SessionLocal()
                try:
                    # Check open position
                    open_pos = db_check.query(Trade).filter(
                        Trade.symbol == symbol,
                        Trade.status == 'open'
                    ).first()
                    if open_pos:
                        logger.info(f"    🚫 {symbol} already has OPEN position - skipping")
                        continue
                    
                    # Check if already called TODAY (once per day limit)
                    today_start = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
                    today_trade = db_check.query(Trade).filter(
                        Trade.symbol == symbol,
                        Trade.direction == 'LONG',
                        Trade.opened_at >= today_start
                    ).first()
                    if today_trade:
                        logger.info(f"    🚫 {symbol} already called TODAY - once per day limit")
                        continue
                finally:
                    db_check.close()
                
                logger.info(f"  🔍 Analyzing {symbol} (+{change_24h:.1f}%)")
                
                # Fetch candles for analysis
                candles_1m = await self.fetch_candles(symbol, '1m', limit=20)
                candles_5m = await self.fetch_candles(symbol, '5m', limit=30)
                candles_15m = await self.fetch_candles(symbol, '15m', limit=30)
                candles_4h = await self.fetch_candles(symbol, '4h', limit=20)
                
                if len(candles_1m) < 15 or len(candles_5m) < 20 or len(candles_15m) < 20:
                    logger.info(f"    ❌ Insufficient candle data")
                    continue
                
                # 🔥 FIX: Get LIVE ticker price for anti-top checks (not stale candle close)
                live_price = await self.get_ticker_price(symbol)
                if not live_price:
                    live_price = candles_1m[-1][4]  # Fallback to 1m close
                    logger.warning(f"    ⚠️ No live ticker, using 1m close: ${live_price}")
                
                # ═══════════════════════════════════════════════════════
                # ANTI-TOP FILTERS: Don't buy coins that already ran hard
                # Uses LIVE price to catch mid-candle spikes!
                # ═══════════════════════════════════════════════════════
                
                # Check 15m impulse - if moved 15%+ in last 15 min, we're too late
                if len(candles_15m) >= 2:
                    price_15m_ago = candles_15m[-2][4]
                    impulse_15m = ((live_price - price_15m_ago) / price_15m_ago) * 100
                    if impulse_15m > 15:
                        logger.info(f"    ❌ TOO LATE: {impulse_15m:.1f}% move in last 15m (live) - already ran!")
                        continue
                
                # Check 1h impulse - if moved 25%+ in last hour, we're too late
                if len(candles_15m) >= 5:
                    price_1h_ago = candles_15m[-5][4]
                    impulse_1h = ((live_price - price_1h_ago) / price_1h_ago) * 100
                    if impulse_1h > 25:
                        logger.info(f"    ❌ TOO LATE: {impulse_1h:.1f}% move in last 1h (live) - already ran!")
                        continue
                
                # Check 4h EMA extension - 22% max above 4h EMA21
                if len(candles_4h) >= 21:
                    closes_4h = [c[4] for c in candles_4h]
                    ema21_4h = self._calculate_ema(closes_4h, 21)
                    extension_4h = ((live_price - ema21_4h) / ema21_4h) * 100
                    
                    if extension_4h > 22:
                        logger.info(f"    ❌ EXTENDED: {extension_4h:.1f}% above 4h EMA21 (max 22%)")
                        continue
                    
                    # Check consecutive green 4h candles (multi-day pump)
                    recent_4h = candles_4h[-5:]
                    green_count = sum(1 for c in recent_4h if c[4] > c[1])
                    if green_count >= 5:
                        logger.info(f"    ❌ SUSTAINED PUMP: {green_count}/5 green 4h candles")
                        continue
                
                closes_1m = [c[4] for c in candles_1m]
                closes_5m = [c[4] for c in candles_5m]
                closes_15m = [c[4] for c in candles_15m]
                current_price = live_price  # Use live ticker price (already fetched above)
                
                # Calculate EMAs
                ema9_5m = self._calculate_ema(closes_5m, 9)
                ema21_5m = self._calculate_ema(closes_5m, 21)
                ema9_15m = self._calculate_ema(closes_15m, 9)
                ema21_15m = self._calculate_ema(closes_15m, 21)
                
                # RSI
                rsi_5m = self._calculate_rsi(closes_5m, 14)
                
                # RELAXED Filter 1: 15m uptrend required (keep this one)
                if not (ema9_15m > ema21_15m):
                    logger.info(f"    ❌ 15m downtrend - skipping")
                    continue
                
                # RELAXED Filter 2: 5m uptrend OR close to crossover (within 0.3%)
                ema_gap_5m = ((ema9_5m - ema21_5m) / ema21_5m) * 100
                if ema9_5m < ema21_5m and ema_gap_5m < -0.3:
                    logger.info(f"    ❌ 5m bearish ({ema_gap_5m:.2f}% gap)")
                    continue
                
                # RELAXED Filter 3: Price above 15m EMA21 OR within 1%
                price_vs_ema21_15m = ((current_price - ema21_15m) / ema21_15m) * 100
                if price_vs_ema21_15m < -1.0:
                    logger.info(f"    ❌ Price {price_vs_ema21_15m:.1f}% below 15m EMA21")
                    continue
                
                # REMOVED: 5m EMA21 check (too strict)
                
                if rsi_5m > 72:
                    logger.info(f"    ❌ RSI {rsi_5m:.0f} too hot (need <72)")
                    continue
                
                if rsi_5m < 30:
                    logger.info(f"    ❌ RSI {rsi_5m:.0f} too cold (need 30+)")
                    continue
                
                # Filter 6: Check for pullback entry on 5m candle (not buying top)
                current_5m_candle = candles_5m[-1]
                candle_5m_high = current_5m_candle[2]
                candle_5m_low = current_5m_candle[3]
                candle_5m_range = candle_5m_high - candle_5m_low
                
                if candle_5m_range > 0:
                    close_position_5m = (current_price - candle_5m_low) / candle_5m_range
                    if close_position_5m > 0.75:  # Relaxed: Allow up to 75% of candle
                        logger.info(f"    ❌ Price at 5m candle top ({close_position_5m:.0%}) - need <75%")
                        continue
                
                # Filter 6b: Also check 1m candle isn't at extreme top
                current_candle = candles_1m[-1]
                candle_high = current_candle[2]
                candle_low = current_candle[3]
                candle_range = candle_high - candle_low
                
                if candle_range > 0:
                    close_position = (current_price - candle_low) / candle_range
                    if close_position > 0.80:  # Relaxed: Allow up to 80% of candle
                        logger.info(f"    ❌ Price at 1m candle top ({close_position:.0%}) - need <80%")
                        continue
                
                # Filter 7: EMA distance - must be close to EMA (not extended)
                ema_distance = ((current_price - ema9_5m) / ema9_5m) * 100
                if ema_distance > 3.0:
                    logger.info(f"    ❌ Extended {ema_distance:.1f}% above EMA (need <3.0%)")
                    continue
                
                # Filter 8: Must be near EMA (pullback to support)
                if ema_distance < -2.0:
                    logger.info(f"    ❌ Below EMA {ema_distance:.1f}% - weak")
                    continue
                
                # Filter 9: Volume - low threshold for low caps
                if volume_24h < 100000:  # $100K+ for low caps
                    logger.info(f"    ❌ Low volume ${volume_24h:,.0f} (need $100K+)")
                    continue
                
                # Filter 10: Must have some pullback (1+ red candles in last 6)
                recent_candles = candles_1m[-7:-1]  # 6 candles before current
                red_count = sum(1 for c in recent_candles if c[4] < c[1])
                if red_count < 1:
                    logger.info(f"    ❌ No pullback ({red_count}/6 red) - need 1+ red candles")
                    continue
                
                # Filter 11: Current candle must be green (resumption)
                if current_price <= current_candle[1]:
                    logger.info(f"    ❌ Current candle red - wait for green resumption")
                    continue
                
                # All filters passed - generate signal!
                logger.info(f"  ✅ MOMENTUM LONG: {symbol} +{change_24h:.1f}% | RSI {rsi_5m:.0f}")
                
                # Calculate TP/SL at 20x leverage - SINGLE TP
                # TP: 3.35% price move = 67% profit at 20x
                # SL: 3.25% price move = 65% loss at 20x
                take_profit_1 = current_price * 1.0335
                take_profit_2 = None  # Single TP only
                stop_loss = current_price * 0.9675
                
                # Update cooldowns
                last_long_signal_time = datetime.utcnow()
                add_symbol_window_cooldown(symbol)  # Mark as traded in current window
                
                return {
                    'symbol': symbol,
                    'direction': 'LONG',
                    'entry_price': current_price,
                    'stop_loss': stop_loss,
                    'take_profit': take_profit_1,
                    'take_profit_1': take_profit_1,
                    'take_profit_2': take_profit_2,
                    'take_profit_3': None,
                    'leverage': 20,
                    '24h_change': change_24h,
                    '24h_volume': volume_24h,
                    'trade_type': 'TOP_GAINER',
                    'strategy': 'MOMENTUM_LONG',
                    'confidence': 80,
                    'reasoning': f"MOMENTUM: +{change_24h:.1f}% gainer | RSI {rsi_5m:.0f} | 15m uptrend | Vol ${volume_24h/1000:.0f}K"
                }
            
            logger.info("❌ No momentum long candidates passed all filters")
            return None
            
        except Exception as e:
            logger.error(f"Error in momentum long signal generation: {e}")
            return None
    
    async def get_early_pumpers(self, limit: int = 10, min_change: float = 5.0, max_change: float = 200.0) -> List[Dict]:
        """
        Fetch FRESH PUMP candidates for LONG entries
        
        🔥 FIXED: Now uses BINANCE + MEXC for accurate 24h data (Bitunix API is garbage!)
        
        ⚡ 3-TIER ULTRA-EARLY DETECTION ⚡
        1. Quick filter: 24h movers with 5%+ gain (Binance/MEXC data)
        2. Multi-tier validation (checks earliest to latest):
           - TIER 1 (5m):  5%+ pump, 3x volume   → Ultra-early (5-10 min)
           - TIER 2 (15m): 5%+ pump, 1.5x volume → Early (15-20 min)
           - TIER 3 (30m): 5%+ pump, 1.5x volume → Fresh (25-30 min)
        
        Returns ONLY fresh pumps (not stale 24h gains!) with tier priority!
        
        Args:
            limit: Number of fresh pumpers to return
            min_change: Minimum 24h change % for pre-filter (default 5%)
            max_change: Maximum 24h change % (default 200% - no cap)
            
        Returns:
            List of {symbol, change_percent, volume_24h, tier, fresh_pump_data} sorted by tier priority then pump %
        """
        try:
            # 🔥 USE BINANCE + MEXC FOR ACCURATE 24H DATA (same as SHORTS!)
            merged_data = {}
            
            # === SOURCE 1: BINANCE FUTURES (Primary - most reliable) ===
            binance_count = 0
            try:
                binance_url = "https://fapi.binance.com/fapi/v1/ticker/24hr"
                response = await self.client.get(binance_url, timeout=10)
                response.raise_for_status()
                binance_tickers = response.json()
                
                for ticker in binance_tickers:
                    symbol = ticker.get('symbol', '')
                    if not symbol.endswith('USDT'):
                        continue
                    try:
                        merged_data[symbol] = {
                            'symbol': symbol,
                            'change_percent': float(ticker.get('priceChangePercent', 0)),
                            'last_price': float(ticker.get('lastPrice', 0)),
                            'volume_usdt': float(ticker.get('quoteVolume', 0)),
                            'high_24h': float(ticker.get('highPrice', 0)),
                            'low_24h': float(ticker.get('lowPrice', 0)),
                            'source': 'binance'
                        }
                        binance_count += 1
                    except (ValueError, TypeError):
                        continue
            except Exception as e:
                logger.warning(f"⚠️ LONGS: Binance API error: {e}")
            
            # === SOURCE 2: MEXC FUTURES (Secondary - for coins not on Binance) ===
            mexc_count = 0
            mexc_added = 0
            try:
                mexc_url = "https://contract.mexc.com/api/v1/contract/ticker"
                response = await self.client.get(mexc_url, timeout=10)
                response.raise_for_status()
                mexc_data = response.json()
                mexc_tickers = mexc_data.get('data', []) if isinstance(mexc_data, dict) else mexc_data
                
                for ticker in mexc_tickers:
                    raw_symbol = ticker.get('symbol', '')
                    symbol = raw_symbol.replace('_', '')
                    if not symbol.endswith('USDT'):
                        continue
                    mexc_count += 1
                    
                    if symbol not in merged_data:
                        try:
                            change_pct = float(ticker.get('riseFallRate', 0))
                            if abs(change_pct) < 5 and abs(change_pct) > 0:
                                change_pct = change_pct * 100
                            
                            merged_data[symbol] = {
                                'symbol': symbol,
                                'change_percent': change_pct,
                                'last_price': float(ticker.get('lastPrice', 0)),
                                'volume_usdt': float(ticker.get('amount24', 0)),
                                'high_24h': float(ticker.get('high24Price', 0)),
                                'low_24h': float(ticker.get('low24Price', 0)),
                                'source': 'mexc'
                            }
                            mexc_added += 1
                        except (ValueError, TypeError):
                            continue
            except Exception as e:
                logger.warning(f"⚠️ LONGS: MEXC API error: {e}")
            
            # === GET BITUNIX AVAILABLE SYMBOLS + CHANGE DATA ===
            bitunix_url = f"{self.base_url}/api/v1/futures/market/tickers"
            bitunix_response = await self.client.get(bitunix_url)
            bitunix_data = bitunix_response.json()
            bitunix_symbols = set()
            bitunix_change_map = {}
            if isinstance(bitunix_data, dict) and bitunix_data.get('data'):
                for t in bitunix_data.get('data', []):
                    sym = t.get('symbol', '')
                    bitunix_symbols.add(sym)
                    try:
                        open_price = float(t.get('open', 0))
                        last_price_val = float(t.get('last', 0) or t.get('lastPrice', 0))
                        if open_price > 0 and last_price_val > 0:
                            bitunix_change_map[sym] = round(((last_price_val - open_price) / open_price) * 100, 2)
                    except (ValueError, TypeError):
                        pass
            
            logger.info(f"📈 LONGS DATA: Binance={binance_count} | MEXC={mexc_count} (+{mexc_added} unique) | Bitunix={len(bitunix_symbols)} tradeable")
            
            # DEBUG: Log top 5 pumpers from merged data
            all_pumpers = [(s, d['change_percent']) for s, d in merged_data.items() if d['change_percent'] > 0]
            all_pumpers.sort(key=lambda x: x[1], reverse=True)
            if all_pumpers[:5]:
                top5 = [(s, f"+{c:.1f}%") for s, c in all_pumpers[:5]]
                logger.info(f"📊 TOP 5 FROM BINANCE/MEXC: {top5}")
            
            # STAGE 1: Filter to pumpers in range AND available on Bitunix
            candidates = []
            rejected_not_bitunix = 0
            rejected_low_volume = 0
            rejected_out_of_range = 0
            
            for symbol, data in merged_data.items():
                change_percent = bitunix_change_map.get(symbol, data['change_percent'])
                
                if symbol not in bitunix_symbols:
                    if min_change <= change_percent <= max_change:
                        rejected_not_bitunix += 1
                    continue
                
                if not (min_change <= change_percent <= max_change):
                    rejected_out_of_range += 1
                    continue
                
                if data['volume_usdt'] < self.min_volume_usdt:
                    rejected_low_volume += 1
                    continue
                
                candidates.append({
                    'symbol': symbol.replace('USDT', '/USDT'),
                    'change_percent_24h': round(change_percent, 2),
                    'volume_24h': round(data['volume_usdt'], 0),
                    'price': data['last_price'],
                    'high_24h': data['high_24h'],
                    'low_24h': data['low_24h']
                })
            
            # Sort by change % for logging
            candidates.sort(key=lambda x: x['change_percent_24h'], reverse=True)
            logger.info(f"Stage 1: {len(candidates)} candidates | Rejected: {rejected_not_bitunix} not-on-Bitunix, {rejected_low_volume} low-vol, {rejected_out_of_range} out-of-range")
            if candidates[:5]:
                top_list = [(c['symbol'], f"+{c['change_percent_24h']}%") for c in candidates[:5]]
                logger.info(f"📈 TOP CANDIDATES: {top_list}")
            
            # 🔥 SIMPLIFIED: Skip tier validation - just return pumping coins
            # Entry quality check happens in analyze_momentum_long / analyze_early_pump_long
            # Those functions check: RSI, EMA position, volume, candle position, etc.
            for candidate in candidates:
                candidate['change_percent'] = candidate['change_percent_24h']  # Use 24h change
                candidate['tier'] = '24h'  # Mark as 24h pumper
                candidate['fresh_pump_data'] = {}
            
            logger.info(f"📈 Returning {min(len(candidates), limit)} pumping coins for LONG analysis")
            return candidates[:limit]
            
        except Exception as e:
            logger.error(f"Error fetching fresh pumpers: {e}")
            return []
    
    async def analyze_momentum(self, symbol: str) -> Optional[Dict]:
        """
        PROFESSIONAL-GRADE ENTRY ANALYSIS for Top Gainers
        
        Filters:
        1. ✅ Pullback entries (price near EMA9, not chasing tops)
        2. ✅ Volume confirmation (>1.3x average = real money flowing)
        3. ✅ Overextension checks (avoid entries >2.5% from EMA)
        4. ✅ RSI confirmation (30-70 range, no extreme overbought/oversold)
        5. ✅ Recent momentum (last 3 candles direction)
        
        Returns:
            {
                'direction': 'LONG' or 'SHORT',
                'confidence': 0-100,
                'entry_price': float,
                'reason': str (detailed entry justification)
            }
        """
        try:
            # 🔥 QUALITY CHECK #1: Liquidity Validation
            liquidity_check = await self.check_liquidity(symbol)
            if not liquidity_check['is_liquid']:
                logger.info(f"{symbol} SHORTS SKIPPED - {liquidity_check['reason']}")
                return None
            
            # 🔥 FRESHNESS CHECK: Skip fresh pumps for SHORTS (let LONGS handle them!)
            # Check all 3 tiers: 5m, 15m, 30m - if ANY are fresh, this is a LONG opportunity
            is_fresh_5m = await self.validate_fresh_5m_pump(symbol)
            is_fresh_15m = await self.validate_fresh_15m_pump(symbol)
            is_fresh_30m = await self.validate_fresh_30m_pump(symbol)
            
            # Check if ANY tier confirms this is a fresh pump (not just truthy dict)
            fresh_5m_confirmed = is_fresh_5m and is_fresh_5m.get('is_fresh_pump') == True
            fresh_15m_confirmed = is_fresh_15m and is_fresh_15m.get('is_fresh_pump') == True
            fresh_30m_confirmed = is_fresh_30m and is_fresh_30m.get('is_fresh_pump') == True
            
            if fresh_5m_confirmed or fresh_15m_confirmed or fresh_30m_confirmed:
                tier = "5m" if fresh_5m_confirmed else ("15m" if fresh_15m_confirmed else "30m")
                logger.info(f"🟢 {symbol} is FRESH PUMP ({tier}) - Skipping SHORTS, will generate LONG signal instead!")
                return None
            
            # Fetch candles with sufficient history for accurate analysis
            candles_5m = await self.fetch_candles(symbol, '5m', limit=50)
            candles_15m = await self.fetch_candles(symbol, '15m', limit=50)
            
            if len(candles_5m) < 30 or len(candles_15m) < 30:
                return None
            
            # 🔥 QUALITY CHECK #2: Anti-Manipulation Filter
            manipulation_check = await self.check_manipulation_risk(symbol, candles_5m)
            if not manipulation_check['is_safe']:
                logger.info(f"{symbol} SHORTS SKIPPED - Manipulation risk: {', '.join(manipulation_check['flags'])}")
                return None
            
            # Convert to DataFrame for candle size analysis
            import pandas as pd
            df_5m = pd.DataFrame(candles_5m, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            
            # 🚨 CRITICAL CHECK: Skip oversized candles (RELAXED: 5% threshold for parabolic moves)
            if self._is_candle_oversized(df_5m, max_body_percent=5.0):
                logger.info(f"{symbol} SKIPPED - Current candle is oversized (prevents poor entries)")
                return None
            
            # Extract price and volume data
            closes_5m = [c[4] for c in candles_5m]
            volumes_5m = [c[5] for c in candles_5m]
            closes_15m = [c[4] for c in candles_15m]
            
            # Current price and previous prices for momentum
            current_price = closes_5m[-1]
            prev_close = closes_5m[-2]
            prev_prev_close = closes_5m[-3] if len(closes_5m) >= 3 else prev_close
            high_5m = candles_5m[-1][2]
            low_5m = candles_5m[-1][3]
            
            # Extract previous candle data for pullback detection
            prev_open = candles_5m[-2][1]
            prev_high = candles_5m[-2][2]
            prev_low = candles_5m[-2][3]
            
            # Calculate previous candle direction
            prev_candle_bullish = prev_close > prev_open
            prev_candle_bearish = prev_close < prev_open
            
            # Current candle direction
            current_open = candles_5m[-1][1]
            current_high = candles_5m[-1][2]
            current_low = candles_5m[-1][3]
            current_candle_bullish = current_price > current_open
            current_candle_bearish = current_price < current_open
            
            # ═══════════════════════════════════════════════════════
            # 🔥 CRITICAL: GLOBAL ENTRY TIMING CHECK FOR ALL SHORTS
            # Prevents shorting at the BOTTOM of dumps (chasing losses)
            # Price must be in UPPER 40% of candle to enter SHORT
            # ═══════════════════════════════════════════════════════
            candle_range = current_high - current_low if current_high > current_low else 0.0001
            price_position_in_candle = (current_price - current_low) / candle_range  # 0 = bottom, 1 = top
            
            # 🚫 REJECT ALL SHORTS if price already dumped to bottom of candle
            if price_position_in_candle < 0.60:
                logger.info(f"🚫 {symbol} SHORT REJECTED - Bad entry timing: price at {price_position_in_candle*100:.0f}% of candle (need 60%+ = near top)")
                return None
            
            logger.info(f"✅ {symbol} Entry timing OK: price at {price_position_in_candle*100:.0f}% of candle (near top)")
            
            # Calculate EMAs (trend identification)
            ema9_5m = self._calculate_ema(closes_5m, 9)
            ema21_5m = self._calculate_ema(closes_5m, 21)
            ema9_15m = self._calculate_ema(closes_15m, 9)
            ema21_15m = self._calculate_ema(closes_15m, 21)
            
            # Calculate RSI (momentum strength)
            rsi_5m = self._calculate_rsi(closes_5m, 14)
            
            # ===== VOLUME ANALYSIS (Critical for top gainers) =====
            avg_volume = sum(volumes_5m[-20:-1]) / 19  # Last 20 candles excluding current
            current_volume = volumes_5m[-1]
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0
            
            # ===== PRICE TO EMA DISTANCE (Pullback detection) =====
            price_to_ema9_dist = ((current_price - ema9_5m) / ema9_5m) * 100
            price_to_ema21_dist = ((current_price - ema21_5m) / ema21_5m) * 100
            
            # Is price near EMA9? (STRICT pullback entry zone)
            is_near_ema9 = abs(price_to_ema9_dist) < 1.0  # Within 1% of EMA9 (TIGHTENED)
            is_near_ema21 = abs(price_to_ema21_dist) < 1.5  # Within 1.5% of EMA21
            
            # ===== TREND ALIGNMENT (Multi-timeframe confirmation) =====
            bullish_5m = ema9_5m > ema21_5m
            bullish_15m = ema9_15m > ema21_15m
            
            # ===== RECENT MOMENTUM (Last 3 candles direction) =====
            recent_candles = closes_5m[-4:]
            bullish_momentum = recent_candles[-1] > recent_candles[-3]  # Higher highs
            bearish_momentum = recent_candles[-1] < recent_candles[-3]  # Lower lows
            
            # ===== OVEREXTENSION CHECK (Avoid buying tops / selling bottoms) =====
            is_overextended_up = price_to_ema9_dist > 2.5  # >2.5% above EMA9
            is_overextended_down = price_to_ema9_dist < -2.5  # >2.5% below EMA9
            
            # ═══════════════════════════════════════════════════════
            # 🔥 BOUNCE DETECTION - Don't short coins already bouncing!
            # If coin dropped 15%+ from high and recovered 30%+ from low = BOUNCING
            # ═══════════════════════════════════════════════════════
            try:
                # Check last 6 hours (72 5m candles) for dump & bounce pattern
                lookback = min(72, len(candles_5m))
                recent_highs = [c[2] for c in candles_5m[-lookback:]]
                recent_lows = [c[3] for c in candles_5m[-lookback:]]
                
                period_high = max(recent_highs)
                period_low = min(recent_lows)
                
                # Calculate drop from high and recovery from low
                drop_from_high_pct = ((period_high - current_price) / period_high) * 100 if period_high > 0 else 0
                recovery_from_low_pct = ((current_price - period_low) / period_low) * 100 if period_low > 0 else 0
                total_range_pct = ((period_high - period_low) / period_low) * 100 if period_low > 0 else 0
                
                # Recovery ratio: how much of the dump has been recovered?
                recovery_ratio = recovery_from_low_pct / total_range_pct if total_range_pct > 0 else 0
                
                # BOUNCE DETECTED: Dropped 15%+ from high but recovered 35%+ of that drop
                is_bouncing = (
                    drop_from_high_pct >= 15 and  # Had significant dump
                    recovery_ratio >= 0.35  # Recovered 35%+ of the drop
                )
                
                if is_bouncing:
                    logger.info(f"  🚫 {symbol} - BOUNCING: Dropped {drop_from_high_pct:.1f}% from high, recovered {recovery_ratio*100:.0f}% - would short the bottom!")
                    return None
                    
            except Exception as e:
                logger.warning(f"{symbol} Bounce detection failed: {e}")
                is_bouncing = False
            
            # ═══════════════════════════════════════════════════════
            # 🔥 EXHAUSTION DETECTION - Find the TOP of chart!
            # ═══════════════════════════════════════════════════════
            try:
                # 1. Price near 24h HIGH (top of chart) - STRICT: within 1% only!
                highs_5m = [c[2] for c in candles_5m]
                high_24h = max(highs_5m[-48:]) if len(highs_5m) >= 48 else max(highs_5m)  # 48 5m candles = 4 hours
                distance_from_high = ((high_24h - current_price) / high_24h) * 100 if high_24h > 0 else 0
                is_near_top = distance_from_high < 1.0  # Within 1% of recent high
                
                # 2. Wick rejection analysis (long upper wick = buyers rejected)
                current_high = candles_5m[-1][2]
                current_low = candles_5m[-1][3]
                candle_body = abs(current_price - current_open)
                upper_wick = current_high - max(current_price, current_open)
                lower_wick = min(current_price, current_open) - current_low
                total_range = current_high - current_low if current_high > current_low else 0.0001
                
                upper_wick_ratio = upper_wick / total_range if total_range > 0 else 0
                has_rejection_wick = upper_wick_ratio > 0.4  # Upper wick is 40%+ of candle = rejection
                
                # 3. Volume exhaustion (declining volume on pumps = buyers drying up)
                recent_volumes = volumes_5m[-5:] if len(volumes_5m) >= 5 else volumes_5m
                older_volumes = volumes_5m[-10:-5] if len(volumes_5m) >= 10 else volumes_5m[:5]
                avg_recent_vol = sum(recent_volumes) / len(recent_volumes) if recent_volumes else 1
                avg_older_vol = sum(older_volumes) / len(older_volumes) if older_volumes else 1
                volume_declining = avg_recent_vol < avg_older_vol * 0.7  # Recent volume 30%+ lower
                
                # 4. Bearish divergence (price making higher high but RSI making lower high)
                has_bearish_divergence = False
                if len(closes_5m) >= 20:
                    try:
                        rsi_range = min(5, len(closes_5m) - 15)
                        if rsi_range > 1:
                            rsi_recent = [self._calculate_rsi(closes_5m[:-i] if i > 0 else closes_5m, 14) for i in range(rsi_range)]
                            price_making_new_high = current_price >= max(closes_5m[-10:-1])
                            rsi_making_lower_high = rsi_5m < max(rsi_recent[1:]) if len(rsi_recent) > 1 else False
                            has_bearish_divergence = price_making_new_high and rsi_making_lower_high and rsi_5m > 55
                    except:
                        pass
                
                # 5. Current candle is RED with body > wick (sellers confirmed)
                candle_body_size = abs(current_price - current_open)
                is_red_candle_confirmed = (
                    current_candle_bearish and 
                    candle_body_size > upper_wick and 
                    candle_body_size > lower_wick
                )
                
                # 6. 15m timeframe showing rejection (upper wick on 15m)
                highs_15m = [c[2] for c in candles_15m]
                current_15m_high = candles_15m[-1][2]
                current_15m_close = closes_15m[-1]
                current_15m_open = candles_15m[-1][1]
                upper_wick_15m = current_15m_high - max(current_15m_close, current_15m_open)
                total_range_15m = candles_15m[-1][2] - candles_15m[-1][3]
                has_15m_rejection = (upper_wick_15m / total_range_15m > 0.3) if total_range_15m > 0 else False
                
                # 7. Consecutive green candles (pump exhaustion - too many green = reversal coming)
                green_streak = 0
                for i in range(-1, -8, -1):
                    if len(candles_5m) >= abs(i):
                        candle = candles_5m[i]
                        if candle[4] > candle[1]:  # close > open = green
                            green_streak += 1
                        else:
                            break
                has_extended_green_streak = green_streak >= 4  # 4+ consecutive green = exhaustion
                
                # 8. 15m RSI overbought (higher timeframe confirmation)
                rsi_15m = self._calculate_rsi(closes_15m, 14)
                is_15m_overbought = rsi_15m >= 67  # Stricter: 67+ for quality
                
                # 9. Slowing momentum (current candle smaller than previous = losing steam)
                current_body = abs(current_price - current_open)
                prev_body = abs(prev_close - prev_open)
                is_momentum_slowing = current_body < prev_body * 0.6  # Current candle 40%+ smaller
                
                # 10. Price far from EMA21 (extended = mean reversion likely)
                is_very_extended = price_to_ema21_dist >= 4.0  # 4%+ above EMA21
                
                # ═══════════════════════════════════════════════════════
                # 🔥 WEIGHTED EXHAUSTION SCORING - Quality over Quantity!
                # Core flags (2 pts each) = high-confidence reversal signs
                # Secondary flags (1 pt each) = supporting confirmation
                # Require: ≥6 total points AND ≥2 core flags for quality
                # ═══════════════════════════════════════════════════════
                
                # CORE FLAGS (2 pts each) - Strong reversal indicators
                core_flags = {
                    'wick_rejection': has_rejection_wick,      # Buyers rejected at top
                    'volume_declining': volume_declining,      # Buyers drying up
                    'bearish_divergence': has_bearish_divergence,  # RSI diverging from price
                    'very_extended': is_very_extended,         # Far from mean (4%+ EMA21)
                    '15m_rejection': has_15m_rejection         # Higher TF rejection
                }
                
                # SECONDARY FLAGS (1 pt each) - Supporting confirmation
                secondary_flags = {
                    'near_top': is_near_top,                   # Within 1% of high
                    'rsi_overbought': rsi_5m >= 70,            # 5m overbought
                    'red_candle': is_red_candle_confirmed,     # Bearish candle
                    'green_streak': has_extended_green_streak, # Exhausted pump
                    '15m_overbought': is_15m_overbought,       # 15m overbought
                    'momentum_slowing': is_momentum_slowing    # Candle shrinking
                }
                
                core_count = sum(core_flags.values())
                secondary_count = sum(secondary_flags.values())
                exhaustion_score = (core_count * 2) + (secondary_count * 1)
                
                # Legacy count for logging
                exhaustion_signs = core_count + secondary_count
                
                logger.info(f"  📊 {symbol} EXHAUSTION SCORE: {exhaustion_score} pts ({core_count} core + {secondary_count} secondary)")
                logger.info(f"     Core: Wick={has_rejection_wick}, VolDown={volume_declining}, Diverg={has_bearish_divergence}, Extended={is_very_extended}, 15mRej={has_15m_rejection}")
                logger.info(f"     Secondary: Top={is_near_top}, RSI={rsi_5m:.0f}, RedCandle={is_red_candle_confirmed}, GreenStreak={green_streak}, 15mRSI={rsi_15m:.0f}, SlowMom={is_momentum_slowing}")
            except Exception as e:
                logger.warning(f"{symbol} Exhaustion detection failed: {e}")
                exhaustion_signs = 0
                exhaustion_score = 0
                core_count = 0
                secondary_count = 0
                is_near_top = False
                has_rejection_wick = False
                volume_declining = False
                has_bearish_divergence = False
                is_red_candle_confirmed = False
                has_15m_rejection = False
                has_extended_green_streak = False
                is_15m_overbought = False
                is_momentum_slowing = False
                is_very_extended = False
                rsi_15m = 50
                green_streak = 0
                highs_5m = [c[2] for c in candles_5m]
                distance_from_high = 0
            
            # ═══════════════════════════════════════════════════════
            # STRATEGY 1: RETEST REJECTION SHORT - Wait for failed retest!
            # ═══════════════════════════════════════════════════════
            # DON'T short on first breakdown - wait for RETEST to fail
            # Pattern: Big drop → Bounce (retest) → Rejection → SHORT
            # This confirms the breakdown wasn't a fake-out
            # ═══════════════════════════════════════════════════════
            if bullish_5m and bullish_15m:
                # Both short timeframes still bullish - look for retest rejection
                
                # 🔥 STEP 1: Get 1H candles for structure analysis
                try:
                    candles_1h = await self.get_candles(symbol, '1h', limit=48)  # Last 48 hours
                    if not candles_1h or len(candles_1h) < 12:
                        logger.info(f"{symbol} SKIP: Not enough 1H data")
                        return None
                except Exception as e:
                    logger.warning(f"{symbol} Could not fetch 1H data: {e}")
                    return None
                
                closed_1h = candles_1h[:-1]  # Skip live candle
                
                # 🔥 STEP 2: Find the PEAK (highest point in last 24h)
                peak_1h = max(c[2] for c in closed_1h[-24:]) if len(closed_1h) >= 24 else max(c[2] for c in closed_1h)
                
                # Find when peak occurred
                peak_1h_idx = None
                for i in range(len(closed_1h)-1, max(len(closed_1h)-25, -1), -1):
                    if closed_1h[i][2] == peak_1h:
                        peak_1h_idx = i
                        break
                
                if not peak_1h_idx:
                    logger.info(f"{symbol} SKIP: Could not find peak")
                    return None
                
                candles_since_peak = len(closed_1h) - 1 - peak_1h_idx
                
                # 🔥 STEP 3: Calculate drop from peak (STRICT: 8%+ drop required)
                current_vs_peak = ((current_price - peak_1h) / peak_1h) * 100
                has_major_drop = current_vs_peak <= -8.0  # Must have dropped 8%+ from peak
                
                # 🔥 STEP 4: Find the LOW after the peak (breakdown point)
                if candles_since_peak >= 2:
                    post_peak_candles = closed_1h[peak_1h_idx+1:]
                    if post_peak_candles:
                        breakdown_low = min(c[3] for c in post_peak_candles)  # Lowest low after peak
                        
                        # Find when breakdown occurred
                        breakdown_idx = None
                        for i, c in enumerate(post_peak_candles):
                            if c[3] == breakdown_low:
                                breakdown_idx = i
                                break
                        
                        # 🔥 STEP 5: Check for RETEST pattern
                        # After breakdown, price should have bounced back UP
                        # Then rejected (current candles should be red, below the bounce high)
                        if breakdown_idx is not None and breakdown_idx < len(post_peak_candles) - 1:
                            post_breakdown_candles = post_peak_candles[breakdown_idx+1:]
                            
                            if len(post_breakdown_candles) >= 2:
                                # Retest high = highest point after breakdown
                                retest_high = max(c[2] for c in post_breakdown_candles)
                                
                                # Retest must have bounced at least 3% from breakdown low (STRICT)
                                retest_bounce = ((retest_high - breakdown_low) / breakdown_low) * 100
                                has_retest_bounce = retest_bounce >= 3.0
                                
                                # Retest must have FAILED HARD (current price 2%+ below retest high)
                                retest_failed = current_price < retest_high * 0.98  # At least 2% below retest high
                                
                                # Retest high must be MUCH LOWER than peak (lower high = trend changed)
                                retest_lower_than_peak = retest_high < peak_1h * 0.95  # At least 5% below peak
                                
                                # 🔥 STEP 6: Current price action must be bearish (2 red candles)
                                last_1h_red = closed_1h[-1][4] < closed_1h[-1][1]
                                prev_1h_red = closed_1h[-2][4] < closed_1h[-2][1] if len(closed_1h) >= 2 else False
                                two_red_candles = last_1h_red and prev_1h_red
                                
                                # 🔥 STEP 7: Get funding rate
                                funding = await self.get_funding_rate(symbol)
                                funding_pct = funding['funding_rate_percent']
                                is_greedy = funding['is_extreme_positive']
                                
                                # 🔥 STEP 8: Check for buy walls
                                orderbook = await self.get_order_book_walls(symbol, current_price, direction='SHORT')
                                has_wall = orderbook.get('has_blocking_wall', False)
                                
                                # ═══════════════════════════════════════════════════════
                                # STRICT RETEST REJECTION SHORT CONDITIONS:
                                # 1. 8%+ drop from peak (major breakdown happened)
                                # 2. Price bounced 3%+ from low (retest occurred)
                                # 3. Retest high is 5%+ below peak (strong lower high)
                                # 4. Current price 2%+ below retest high (retest failed hard)
                                # 5. Last 2 1H candles are red (confirmed selling)
                                # ═══════════════════════════════════════════════════════
                                is_retest_short = (
                                    has_major_drop and  # 8%+ drop from peak
                                    has_retest_bounce and  # 3%+ bounce from low
                                    retest_lower_than_peak and  # 5%+ lower high
                                    retest_failed and  # 2%+ below retest high
                                    two_red_candles and  # 2 red 1H candles
                                    exhaustion_score >= 4
                                )
                                
                                if is_retest_short:
                                    if has_wall:
                                        logger.info(f"  ⚠️ {symbol} - BUY WALL detected - SKIP")
                                        return None
                                    
                                    confidence = 90
                                    if is_greedy:
                                        confidence += 5
                                    
                                    logger.info(f"{symbol} ✅ RETEST SHORT: Peak ${peak_1h:.4f} → Low ${breakdown_low:.4f} → Retest ${retest_high:.4f} → REJECTED @ ${current_price:.4f}")
                                    return {
                                        'direction': 'SHORT',
                                        'confidence': min(confidence, 99),
                                        'entry_price': current_price,
                                        'reason': f'🎯 RETEST SHORT | {current_vs_peak:.1f}% from peak | Bounce rejected | Lower high confirmed | Funding {funding_pct:.2f}%'
                                    }
                                
                                # Log why skipped
                                skip_reasons = []
                                if not has_major_drop:
                                    skip_reasons.append(f"Only {current_vs_peak:.1f}% from peak (need -8%+)")
                                if not has_retest_bounce:
                                    skip_reasons.append(f"Retest bounce only {retest_bounce:.1f}% (need 3%+)")
                                if not retest_lower_than_peak:
                                    skip_reasons.append(f"Retest too close to peak (need 5%+ below)")
                                if not retest_failed:
                                    skip_reasons.append(f"Retest not failed hard (need 2%+ below)")
                                if not two_red_candles:
                                    skip_reasons.append(f"Need 2 red 1H candles")
                                if skip_reasons:
                                    logger.info(f"{symbol} NO RETEST PATTERN: {', '.join(skip_reasons)}")
                                return None
                
                # RARE LONG EXCEPTION: Massive volume breakout (3.5x+) with perfect setup
                if volume_ratio >= 3.5 and rsi_5m > 50 and rsi_5m < 65 and not is_overextended_up and is_near_ema9:
                    logger.info(f"{symbol} ✅ EXCEPTIONAL LONG: Massive volume {volume_ratio:.1f}x + perfect EMA9 entry")
                    return {
                        'direction': 'LONG',
                        'confidence': 95,
                        'entry_price': current_price,
                        'reason': f'🚀 EXCEPTIONAL VOLUME {volume_ratio:.1f}x | RSI: {rsi_5m:.0f} | Perfect EMA9 pullback - RARE LONG!'
                    }
                
                # Not enough structure for retest pattern
                logger.info(f"{symbol} SKIP: No retest pattern found")
                return None
            
            
            # ═══════════════════════════════════════════════════════
            # STRATEGY 2: DOWNTREND SHORT - DISABLED (bounces too often)
            # ═══════════════════════════════════════════════════════
            # These were catching falling knives that bounce back up
            # OVEREXTENDED shorts at the TOP work best - keep only those
            # ═══════════════════════════════════════════════════════
            elif not bullish_5m and not bullish_15m:
                logger.info(f"{symbol} DOWNTREND STRATEGY DISABLED - Trend already flipped, too late for quality entry")
                return None
            
            # OLD DISABLED CODE - keeping for reference but never executed
            elif False:  # DISABLED
                # Calculate current candle size for strong dump detection
                current_candle_size = abs((current_price - current_open) / current_open * 100)
                
                # 🔥 CONFIRM THIS WAS A TOP GAINER THAT PEAKED
                # Check that price was significantly higher recently (had a run)
                highs_5m_local = [c[2] for c in candles_5m]  # Get highs for this strategy
                recent_high = max(highs_5m_local[-12:]) if len(highs_5m_local) >= 12 else max(highs_5m_local)  # 1hr high
                drop_from_high = ((recent_high - current_price) / recent_high) * 100
                had_significant_run = drop_from_high >= 3.0  # Price dropped 3%+ from recent high
                
                # 🔥 CONFIRM DOWNTREND (lower lows forming)
                lows_5m = [c[3] for c in candles_5m]
                recent_lows = lows_5m[-6:]  # Last 30 min
                is_making_lower_lows = len(recent_lows) >= 3 and recent_lows[-1] < recent_lows[0]
                
                # 🔥 CONFIRM SELLERS IN CONTROL (more red candles than green recently)
                recent_directions = [candles_5m[i][4] < candles_5m[i][1] for i in range(-6, 0)]  # Close < Open = red
                red_candle_count = sum(recent_directions)
                sellers_dominant = red_candle_count >= 4  # 4+ of last 6 candles are red
                
                logger.info(f"  📉 {symbol} DOWNTREND CHECK: Drop from high={drop_from_high:.1f}%, Lower lows={is_making_lower_lows}, Sellers dominant={sellers_dominant} ({red_candle_count}/6 red)")
                
                # ═════ ENTRY PATH 1: CONFIRMED DOWNTREND (Best entry) - TIGHTENED ═════
                is_confirmed_downtrend = (
                    had_significant_run and  # Was a top gainer that pumped
                    drop_from_high >= 5.0 and  # 🔥 STRICT: Need 5%+ drop (was 3%)
                    is_making_lower_lows and  # Making lower lows
                    sellers_dominant and  # Red candles dominant
                    red_candle_count >= 5 and  # 🔥 STRICT: 5/6 red candles (was 4)
                    current_candle_bearish and  # Current candle red
                    35 <= rsi_5m <= 55  # RSI tighter range (was 40-60)
                )
                
                if is_confirmed_downtrend:
                    logger.info(f"{symbol} ✅ DOWNTREND CONFIRMED: {drop_from_high:.1f}% off high | Lower lows forming | {red_candle_count}/6 red candles")
                    return {
                        'direction': 'SHORT',
                        'confidence': 92,
                        'entry_price': current_price,
                        'reason': f'📉 DOWNTREND | {drop_from_high:.1f}% off high | Lower lows | {red_candle_count}/6 red | Trend flipped!'
                    }
                
                # ═════ ENTRY PATH 2: STRONG DUMP (Direct Entry - No Pullback Needed) - TIGHTENED ═════
                # For violent dumps with high volume, enter immediately
                is_strong_dump = (
                    current_candle_bearish and 
                    current_candle_size >= 2.5 and  # 🔥 STRICT: Need 2.5%+ dump (was 1.5%)
                    volume_ratio >= 2.0 and  # 🔥 STRICT: 2x volume (was 1.5x)
                    35 <= rsi_5m <= 55  # Tighter RSI range (was 40-60)
                )
                
                if is_strong_dump:
                    logger.info(f"{symbol} ✅ STRONG DUMP DETECTED: {current_candle_size:.2f}% red candle | Vol: {volume_ratio:.1f}x | RSI: {rsi_5m:.0f}")
                    return {
                        'direction': 'SHORT',
                        'confidence': 88,
                        'entry_price': current_price,
                        'reason': f'🔥 STRONG DUMP | {current_candle_size:.1f}% red candle | Vol: {volume_ratio:.1f}x | RSI: {rsi_5m:.0f} | Direct entry'
                    }
                
                # ═════ ENTRY PATH 2: RESUMPTION PATTERN (Safer, After Pullback) ═════
                has_resumption_pattern = False
                
                # Check if we have: prev-prev RED (dump) → prev GREEN (pullback) → current RED (resumption)
                if len(closes_5m) >= 3 and len(candles_5m) >= 3:
                    prev_prev_open = candles_5m[-3][1]
                    prev_prev_close = closes_5m[-3]
                    
                    # Calculate candle sizes
                    prev_prev_bearish = prev_prev_close < prev_prev_open
                    prev_candle_size = abs((prev_close - prev_open) / prev_open * 100)
                    
                    # PERFECT PATTERN: Red dump → Green pullback → Red resumption
                    if prev_prev_bearish and prev_candle_bullish and current_candle_bearish:
                        prev_prev_size = abs((prev_prev_close - prev_prev_open) / prev_prev_open * 100)
                        
                        # Pullback must be smaller than dump, and current is resuming down
                        if prev_prev_size > prev_candle_size * 1.5 and current_candle_bearish:
                            has_resumption_pattern = True
                            logger.info(f"{symbol} ✅ RESUMPTION PATTERN: Dump {prev_prev_size:.2f}% → Pullback {prev_candle_size:.2f}% → Resuming down")
                
                # Resumption entry: More relaxed than before
                if has_resumption_pattern and rsi_5m >= 45 and rsi_5m <= 70 and volume_ratio >= 1.0:
                    return {
                        'direction': 'SHORT',
                        'confidence': 95,
                        'entry_price': current_price,
                        'reason': f'🎯 RESUMPTION SHORT | Entered AFTER pullback | Vol: {volume_ratio:.1f}x | RSI: {rsi_5m:.0f} | Clean entry'
                    }
                
                # SKIP - no valid entry pattern
                else:
                    skip_reason = []
                    if not has_resumption_pattern and not is_strong_dump:
                        skip_reason.append("No entry pattern (need: strong dump OR resumption)")
                    if not current_candle_bearish:
                        skip_reason.append("Current candle not red")
                    if rsi_5m < 40 or rsi_5m > 70:
                        skip_reason.append(f"RSI {rsi_5m:.0f} out of range (need 40-70)")
                    if volume_ratio < 1.0:
                        skip_reason.append(f"Low volume {volume_ratio:.1f}x (need 1.0x+)")
                    
                    logger.info(f"{symbol} SHORT SKIPPED: {', '.join(skip_reason)}")
                    return None
            
            
            # ═══════════════════════════════════════════════════════
            # PARABOLIC REVERSAL - ACTIVE (works well!)
            # ═══════════════════════════════════════════════════════
            # 5m bullish + 15m bearish = 15m turning, catching reversal at the point
            # ═══════════════════════════════════════════════════════
            elif bullish_5m and not bullish_15m:
                # 15m already bearish but 5m lagging - this is the reversal point
                # Perfect for catching top gainer dumps (like CROSS +48% starting to roll over)
                price_extension = price_to_ema9_dist
                
                # 🔥 RELAXED PARABOLIC DETECTION (Catch exhausted pumps earlier!)
                # Check last 6 candles (30 min) instead of just 2 (10 min)
                has_exhaustion_signs = False
                exhaustion_reason = ""
                
                # Check 1: Topping pattern in last 6 candles (30 min window)
                if len(closes_5m) >= 6 and len(candles_5m) >= 6:
                    recent_closes = closes_5m[-6:]
                    recent_candles = candles_5m[-6:]
                    
                    # Look for reversal signs: lower highs in last 3 candles
                    highs = [c[2] for c in recent_candles[-3:]]  # Last 3 highs
                    has_lower_highs = highs[-1] < highs[0]  # Current high < first high
                    
                    # OR: Climax candle with big upper wick (10%+ wick = rejection)
                    current_high = candles_5m[-1][2]
                    current_wick_size = ((current_high - current_price) / current_price) * 100
                    has_big_wick = current_wick_size >= 1.0  # 🔥 RELAXED: 1%+ wick (was 10%+)
                    
                    if has_lower_highs or has_big_wick:
                        has_exhaustion_signs = True
                        if has_lower_highs:
                            exhaustion_reason = "Lower highs forming"
                        else:
                            exhaustion_reason = f"{current_wick_size:.1f}% upper wick rejection"
                        logger.info(f"{symbol} ✅ EXHAUSTION: {exhaustion_reason}")
                
                # 🔥 STRICT ENTRY LOGIC: Require RSI AND exhaustion signs (quality over quantity)
                good_rsi = rsi_5m >= 65  # 🔥 STRICT: 65+ (was 55)
                good_volume = volume_ratio >= 1.5  # 🔥 STRICT: 1.5x (was 1.2x)
                good_extension = price_extension > 3.0  # 🔥 STRICT: 3%+ (was 2%)
                
                # Entry if: RSI good AND exhaustion signs AND volume + extension (strict!)
                if good_extension and good_volume and good_rsi and has_exhaustion_signs:
                    confidence = 92 if (good_rsi and has_exhaustion_signs) else 88
                    logger.info(f"{symbol} ✅ PARABOLIC REVERSAL: Extension {price_extension:+.1f}% | RSI {rsi_5m:.0f} | Vol {volume_ratio:.1f}x | {exhaustion_reason}")
                    return {
                        'direction': 'SHORT',
                        'confidence': confidence,
                        'entry_price': current_price,
                        'reason': f'🎯 PARABOLIC REVERSAL | {price_extension:+.1f}% overextended | {exhaustion_reason if exhaustion_reason else f"RSI {rsi_5m:.0f}"} | Vol: {volume_ratio:.1f}x'
                    }
                else:
                    skip_reason = []
                    if not good_extension:
                        skip_reason.append(f"Not extended enough ({price_extension:+.1f}%, need >2%)")
                    if not good_rsi and not has_exhaustion_signs:
                        skip_reason.append(f"RSI {rsi_5m:.0f} (need 55+) AND no exhaustion signs")
                    if not good_volume:
                        skip_reason.append(f"Low volume {volume_ratio:.1f}x (need 1.2x+)")
                    
                    logger.info(f"{symbol} PARABOLIC SKIPPED: {', '.join(skip_reason)}")
                    return None
            
            
            # ═══════════════════════════════════════════════════════
            # EARLY REVERSAL - DISABLED (bounces too often)
            # ═══════════════════════════════════════════════════════
            # 5m bearish + 15m bullish = too early, often bounces back
            # OVEREXTENDED shorts (both TFs bullish) work best
            # ═══════════════════════════════════════════════════════
            elif not bullish_5m and bullish_15m:
                logger.info(f"{symbol} EARLY REVERSAL DISABLED - Mixed signals (5m bearish, 15m bullish), unreliable")
                return None
            
            # OLD DISABLED EARLY REVERSAL CODE
            elif False:  # DISABLED
                # 5m turned bearish but 15m still bullish = Early reversal signal!
                # This catches dumps BEFORE the 15m confirms (super early entry)
                
                # Check for early reversal pattern - TIGHTENED
                is_early_reversal = (
                    current_candle_bearish and  # Current candle is red
                    bearish_momentum and  # Recent momentum turning down
                    rsi_5m >= 60 and rsi_5m <= 75 and  # 🔥 STRICT: RSI 60-75 (was 50-70)
                    volume_ratio >= 1.8  # 🔥 STRICT: 1.8x volume (was 1.2x)
                )
                
                if is_early_reversal:
                    logger.info(f"{symbol} ✅ EARLY REVERSAL: 5m bearish, 15m still bullish | Vol: {volume_ratio:.1f}x | RSI: {rsi_5m:.0f}")
                    return {
                        'direction': 'SHORT',
                        'confidence': 85,
                        'entry_price': current_price,
                        'reason': f'⚡ EARLY REVERSAL | 5m turning bearish | Vol: {volume_ratio:.1f}x | RSI: {rsi_5m:.0f} | Caught early!'
                    }
                else:
                    logger.info(f"{symbol} MIXED SIGNALS - not clear enough (5m bearish: {not bullish_5m}, 15m bullish: {bullish_15m}, RSI: {rsi_5m:.0f}, Vol: {volume_ratio:.1f}x)")
                    return None
            
            # Other mixed signals - skip (e.g., 5m bullish + 15m bearish = lagging, not early)
            else:
                logger.info(f"{symbol} MIXED SIGNALS - skipping (5m bullish: {bullish_5m}, 15m bullish: {bullish_15m})")
                return None
                
        except Exception as e:
            logger.error(f"Error analyzing momentum for {symbol}: {e}")
            return None
    
    async def analyze_momentum_long(self, symbol: str) -> Optional[Dict]:
        """
        🚀 AGGRESSIVE MOMENTUM LONG - Catch strong pumps WITHOUT waiting for retracement!
        
        This is the "aggressive" version for coins showing STRONG momentum (10-30% pumps).
        Allows entries during the pump (no pullback required) to catch big runners.
        
        Entry criteria:
        1. ✅ Strong volume surge (1.5x+ average)
        2. ✅ Bullish momentum (EMA9 > EMA21, both timeframes)
        3. ✅ NOT OVEREXTENDED - within 10% of EMA9 (relaxed from 5%)
        4. ✅ RSI 45-78 (momentum allowed, not screaming overbought)
        5. ✅ Bullish candle (green, not red)
        6. ✅ No retracement required - direct momentum entry!
        
        Returns signal for AGGRESSIVE LONG entry or None if criteria not met
        """
        try:
            logger.info(f"🚀 MOMENTUM LONG ANALYSIS: {symbol}...")
            
            # Quality checks
            liquidity_check = await self.check_liquidity(symbol)
            if not liquidity_check['is_liquid']:
                logger.info(f"  ❌ {symbol} - {liquidity_check['reason']}")
                return None
            
            candles_5m = await self.fetch_candles(symbol, '5m', limit=50)
            candles_15m = await self.fetch_candles(symbol, '15m', limit=50)
            
            if len(candles_5m) < 30 or len(candles_15m) < 30:
                return None
            
            manipulation_check = await self.check_manipulation_risk(symbol, candles_5m)
            if not manipulation_check['is_safe']:
                logger.info(f"  ❌ {symbol} - Manipulation risk")
                return None
            
            # Technical analysis
            closes_5m = [c[4] for c in candles_5m]
            volumes_5m = [c[5] for c in candles_5m]
            closes_15m = [c[4] for c in candles_15m]
            
            current_price = closes_5m[-1]
            current_open = candles_5m[-1][1]
            current_candle_bullish = current_price > current_open
            
            # EMAs
            ema9_5m = self._calculate_ema(closes_5m, 9)
            ema21_5m = self._calculate_ema(closes_5m, 21)
            ema9_15m = self._calculate_ema(closes_15m, 9)
            ema21_15m = self._calculate_ema(closes_15m, 21)
            
            # RSI
            rsi_5m = self._calculate_rsi(closes_5m, 14)
            
            # Volume
            avg_volume = sum(volumes_5m[-20:-1]) / 19
            current_volume = volumes_5m[-1]
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0
            
            # Price to EMA distance
            price_to_ema9_dist = ((current_price - ema9_5m) / ema9_5m) * 100
            
            # Trend alignment (bullish required)
            bullish_5m = ema9_5m > ema21_5m
            bullish_15m = ema9_15m > ema21_15m
            
            # 🔥 AGGRESSIVE MOMENTUM ENTRY
            # Allows entries during strong pumps WITHOUT waiting for pullback
            
            # 🛡️ CRITICAL: Reject if price is near 24h HIGH (prevents top entries on multi-day pumps!)
            highs_5m = [c[2] for c in candles_5m]
            high_24h = max(highs_5m[-48:]) if len(highs_5m) >= 48 else max(highs_5m)  # 4 hours of 5m candles
            distance_from_24h_high = ((high_24h - current_price) / high_24h) * 100 if high_24h > 0 else 0
            
            if distance_from_24h_high < 1.5:
                logger.info(f"  ❌ {symbol} - Too close to 24h high ({distance_from_24h_high:.1f}% away, need >1.5% pullback)")
                return None
            logger.info(f"  📊 {symbol} - Distance from 24h high: {distance_from_24h_high:.1f}%")
            
            if not (bullish_5m and bullish_15m):
                logger.info(f"  ❌ {symbol} - Not bullish on both timeframes")
                return None
            
            candles_1h_mom = await self.fetch_candles(symbol, '1h', limit=30)
            if candles_1h_mom and len(candles_1h_mom) >= 21:
                closes_1h_mom = [c[4] for c in candles_1h_mom]
                ema9_1h_mom = self._calculate_ema(closes_1h_mom, 9)
                ema21_1h_mom = self._calculate_ema(closes_1h_mom, 21)
                if ema9_1h_mom <= ema21_1h_mom:
                    logger.info(f"  ❌ {symbol} - 1H trend is bearish (EMA9 ≤ EMA21)")
                    return None
            
            if price_to_ema9_dist > 3.0:
                logger.info(f"  ❌ {symbol} - Too extended ({price_to_ema9_dist:+.1f}% from EMA9, need ≤3%)")
                return None
            
            if not (50 <= rsi_5m <= 62):
                logger.info(f"  ❌ {symbol} - RSI {rsi_5m:.0f} out of range (need 50-62)")
                return None
            
            if volume_ratio < 2.0:
                logger.info(f"  ❌ {symbol} - Low volume {volume_ratio:.1f}x (need 2.0x+)")
                return None
            
            if not current_candle_bullish:
                logger.info(f"  ❌ {symbol} - Current 5m candle is bearish (need green)")
                return None
            
            # 🔥 FASTER PULLBACK DETECTION: Use 1m candles to catch dips earlier!
            # Instead of waiting for 5m RED candle, check for ANY 1m pullback in last 3 candles
            candles_1m = await self.fetch_candles(symbol, '1m', limit=10)
            
            if len(candles_1m) >= 5:
                # Check last 5 one-minute candles for ANY red candle (pullback)
                has_1m_pullback = False
                for i in range(-5, -1):  # Check candles -5 to -2 (not current)
                    c = candles_1m[i]
                    if c[4] < c[1]:  # Close < Open = RED candle
                        has_1m_pullback = True
                        break
                
                # Current 1m should be green (bounce)
                current_1m = candles_1m[-1]
                current_1m_green = current_1m[4] > current_1m[1]
                
                if has_1m_pullback and current_1m_green:
                    logger.info(f"  ✅ {symbol} - 1m MICRO-PULLBACK detected → GREEN bounce!")
                elif has_1m_pullback:
                    logger.info(f"  ⚠️ {symbol} - 1m pullback found, waiting for green bounce")
                    return None
                else:
                    # No 1m pullback - check if 5m has pullback as fallback
                    prev_close = closes_5m[-2]
                    prev_open = candles_5m[-2][1]
                    if prev_close < prev_open:
                        logger.info(f"  ✅ {symbol} - 5m RED pullback → GREEN continuation")
                    else:
                        logger.info(f"  ❌ {symbol} - No pullback on 1m or 5m - too risky")
                        return None
            else:
                # Fallback to 5m pullback check
                prev_close = closes_5m[-2]
                prev_open = candles_5m[-2][1]
                if prev_close >= prev_open:
                    logger.info(f"  ❌ {symbol} - No pullback detected - waiting for dip")
                    return None
                logger.info(f"  ✅ {symbol} - 5m RED pullback → GREEN continuation")
            
            # Filter 6: Don't enter at TOP of green candle!
            current_high = candles_5m[-1][2]
            current_low = candles_5m[-1][3]
            candle_range = current_high - current_low
            
            if candle_range > 0:
                price_position_in_candle = ((current_price - current_low) / candle_range) * 100
            else:
                price_position_in_candle = 50
            
            if price_position_in_candle > 60:
                logger.info(f"  ❌ {symbol} - Price at TOP of candle ({price_position_in_candle:.0f}% of range, need <60%)")
                return None
            
            # ✅ ALL CHECKS PASSED - Momentum entry AFTER pullback!
            confidence = 88  # Aggressive = slightly lower confidence than safe pullback entries
            
            logger.info(f"  ✅ {symbol} - MOMENTUM LONG entry!")
            logger.info(f"     • Price: {price_to_ema9_dist:+.1f}% from EMA9")
            logger.info(f"     • RSI: {rsi_5m:.0f}")
            logger.info(f"     • Volume: {volume_ratio:.1f}x")
            
            return {
                'direction': 'LONG',
                'confidence': confidence,
                'entry_price': current_price,
                'reason': f'🚀 MOMENTUM ENTRY | {price_to_ema9_dist:+.1f}% from EMA9 | RSI {rsi_5m:.0f} | Vol {volume_ratio:.1f}x | Riding strong pump!'
            }
            
        except Exception as e:
            logger.error(f"Error in momentum LONG analysis for {symbol}: {e}")
            return None
    
    async def analyze_early_pump_long(self, symbol: str, coin_data: Dict = None) -> Optional[Dict]:
        """
        🎯 TA-FIRST LONG ANALYSIS (6/6 confirmations required before AI)
        
        AI is ONLY called after coin passes ALL 6 TA confirmations.
        AI's job is JUST to set optimal TP/SL levels, not decide whether to trade.
        
        🔒 STRICT TA Confirmations (Jan 2026 - TIGHTENED):
        1. Liquidity OK (spread < threshold) - HARD REQUIREMENT
        2. Anti-manipulation check passed - HARD REQUIREMENT
        3. Both 5m AND 15m bullish with EMA spread (0.30% day / 0.40% night)
        4. RSI in safe zone (40-58 day / 40-55 night)
        5. Strong volume (1.3x day / 1.5x night)
        6. Price not at top (<70% day / <60% night)
        
        🌙 OVERNIGHT MODE (11pm-8am GMT): Extra strict filters applied
        
        Returns signal for LONG entry or None if TA filters fail
        """
        try:
            from datetime import datetime, timezone
            
            current_utc = datetime.now(timezone.utc)
            current_hour_gmt = current_utc.hour
            is_overnight = current_hour_gmt >= 23 or current_hour_gmt < 8
            
            if is_overnight:
                logger.info(f"🌙 OVERNIGHT MODE ({current_hour_gmt}:00 GMT) - Stricter LONG filters active")
            
            logger.info(f"🟢 ANALYZING {symbol} FOR LONGS (TA-first)...")
            
            confirmations = 0
            confirmation_details = []
            
            # ═══════════════════════════════════════════════════════
            # CONFIRMATION #1: Liquidity Validation
            # ═══════════════════════════════════════════════════════
            liquidity_check = await self.check_liquidity(symbol)
            if liquidity_check['is_liquid']:
                confirmations += 1
                confirmation_details.append(f"✅ Liquidity (spread: {liquidity_check.get('spread_percent', 0):.2f}%)")
            else:
                confirmation_details.append(f"❌ Liquidity: {liquidity_check['reason']}")
                logger.info(f"  ❌ {symbol} - {liquidity_check['reason']}")
                return None  # Hard requirement
            
            # Fetch candles
            candles_5m = await self.fetch_candles(symbol, '5m', limit=50)
            candles_15m = await self.fetch_candles(symbol, '15m', limit=50)
            
            if len(candles_5m) < 30 or len(candles_15m) < 30:
                logger.info(f"  ❌ {symbol} - Not enough candle data")
                return None
            
            # ═══════════════════════════════════════════════════════
            # CONFIRMATION #2: Anti-Manipulation Filter
            # ═══════════════════════════════════════════════════════
            manipulation_check = await self.check_manipulation_risk(symbol, candles_5m)
            if manipulation_check['is_safe']:
                confirmations += 1
                confirmation_details.append("✅ Anti-manipulation")
            else:
                confirmation_details.append(f"❌ Manipulation: {', '.join(manipulation_check['flags'])}")
                logger.info(f"  ❌ {symbol} - Manipulation risk: {', '.join(manipulation_check['flags'])}")
                return None  # Hard requirement
            
            # Extract data
            closes_5m = [c[4] for c in candles_5m]
            volumes_5m = [c[5] for c in candles_5m]
            closes_15m = [c[4] for c in candles_15m]
            highs_5m = [c[2] for c in candles_5m]
            lows_5m = [c[3] for c in candles_5m]
            
            current_price = closes_5m[-1]
            
            # Calculate technical indicators
            ema9_5m = self._calculate_ema(closes_5m, 9)
            ema21_5m = self._calculate_ema(closes_5m, 21)
            ema9_15m = self._calculate_ema(closes_15m, 9)
            ema21_15m = self._calculate_ema(closes_15m, 21)
            rsi_5m = self._calculate_rsi(closes_5m, 14)
            
            # Volume analysis
            avg_volume = sum(volumes_5m[-20:-1]) / 19 if len(volumes_5m) >= 20 else sum(volumes_5m) / len(volumes_5m)
            current_volume = volumes_5m[-1]
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0
            
            # Price to EMA distance
            price_to_ema9 = ((current_price - ema9_5m) / ema9_5m) * 100 if ema9_5m > 0 else 0
            
            # Trend alignment
            trend_5m = "bullish" if ema9_5m > ema21_5m else "bearish"
            trend_15m = "bullish" if ema9_15m > ema21_15m else "bearish"
            
            # ═══════════════════════════════════════════════════════
            # 🌙 OVERNIGHT MODE THRESHOLDS (11pm-8am GMT)
            # ═══════════════════════════════════════════════════════
            if is_overnight:
                ema_spread_min = 0.40
                rsi_min, rsi_max = 42, 53
                volume_min = 1.8
                price_pos_max = 55
                confirmations_required = 7
            else:
                ema_spread_min = 0.35
                rsi_min, rsi_max = 42, 56
                volume_min = 1.5
                price_pos_max = 60
                confirmations_required = 7
            
            # ═══════════════════════════════════════════════════════
            # CONFIRMATION #3: Trend Alignment (both TFs bullish)
            # ═══════════════════════════════════════════════════════
            ema_spread = ((ema9_5m - ema21_5m) / ema21_5m * 100) if ema21_5m > 0 else 0
            if trend_5m == "bullish" and ema_spread >= ema_spread_min and trend_15m == "bullish":
                confirmations += 1
                confirmation_details.append(f"✅ Trend (5m: {trend_5m}, 15m: {trend_15m}, spread: {ema_spread:.2f}%)")
            else:
                confirmation_details.append(f"❌ Trend: Need both TFs bullish + {ema_spread_min}% spread (got {ema_spread:.2f}%)")
            
            # ═══════════════════════════════════════════════════════
            # CONFIRMATION #4: RSI in safe zone
            # ═══════════════════════════════════════════════════════
            if rsi_min <= rsi_5m <= rsi_max:
                confirmations += 1
                confirmation_details.append(f"✅ RSI: {rsi_5m:.0f}")
            else:
                confirmation_details.append(f"❌ RSI: {rsi_5m:.0f} (need {rsi_min}-{rsi_max})")
            
            # ═══════════════════════════════════════════════════════
            # CONFIRMATION #5: Volume confirmation
            # ═══════════════════════════════════════════════════════
            if volume_ratio >= volume_min:
                confirmations += 1
                confirmation_details.append(f"✅ Volume: {volume_ratio:.1f}x")
            else:
                confirmation_details.append(f"❌ Volume: {volume_ratio:.1f}x (need ≥{volume_min}x)")
            
            # ═══════════════════════════════════════════════════════
            # CONFIRMATION #6: Price not at top
            # ═══════════════════════════════════════════════════════
            recent_high = max(highs_5m[-10:])
            recent_low = min(lows_5m[-10:])
            price_range = recent_high - recent_low
            price_position = ((current_price - recent_low) / price_range * 100) if price_range > 0 else 50
            
            if price_position < price_pos_max:
                confirmations += 1
                confirmation_details.append(f"✅ Price position: {price_position:.0f}%")
            else:
                confirmation_details.append(f"❌ Price at top: {price_position:.0f}% (need <{price_pos_max}%)")
            
            # ═══════════════════════════════════════════════════════
            # HARD FILTER: VWAP - Price must be near/above VWAP for LONGs
            # ═══════════════════════════════════════════════════════
            vwap = self._calculate_vwap(candles_5m)
            price_to_vwap_pct = ((current_price - vwap) / vwap) * 100 if vwap > 0 else 0
            if price_to_vwap_pct < -1.5:
                logger.info(f"  ❌ {symbol} - Price too far below VWAP ({price_to_vwap_pct:+.1f}%, need >-1.5%)")
                return None
            if price_to_vwap_pct > 4.0:
                logger.info(f"  ❌ {symbol} - Price too far above VWAP ({price_to_vwap_pct:+.1f}%, need <4.0%) - overextended")
                return None
            confirmation_details.append(f"✅ VWAP: {price_to_vwap_pct:+.1f}% from VWAP")
            
            # ═══════════════════════════════════════════════════════
            # CONFIRMATION #7: 1H Trend Alignment (HIGHER TIMEFRAME)
            # ═══════════════════════════════════════════════════════
            candles_1h = await self.fetch_candles(symbol, '1h', limit=30)
            if candles_1h and len(candles_1h) >= 21:
                closes_1h = [c[4] for c in candles_1h]
                ema9_1h = self._calculate_ema(closes_1h, 9)
                ema21_1h = self._calculate_ema(closes_1h, 21)
                rsi_1h = self._calculate_rsi(closes_1h, 14)
                trend_1h_bullish = ema9_1h > ema21_1h
                rsi_1h_ok = rsi_1h < 72
                
                if trend_1h_bullish and rsi_1h_ok:
                    confirmations += 1
                    confirmation_details.append(f"✅ 1H Trend: Bullish (RSI {rsi_1h:.0f})")
                else:
                    reasons = []
                    if not trend_1h_bullish:
                        reasons.append("bearish EMA")
                    if not rsi_1h_ok:
                        reasons.append(f"RSI {rsi_1h:.0f} overbought")
                    confirmation_details.append(f"❌ 1H Trend: {', '.join(reasons)}")
            else:
                confirmation_details.append("❌ 1H Trend: Insufficient data")
            
            # Log confirmation status
            mode_label = "🌙 OVERNIGHT" if is_overnight else "☀️ DAYTIME"
            logger.info(f"  📊 {symbol} [{mode_label}] - {confirmations}/7 confirmations | RSI: {rsi_5m:.0f} | Vol: {volume_ratio:.1f}x | Trend: 5m={trend_5m}")
            for detail in confirmation_details:
                logger.info(f"     {detail}")
            
            # ═══════════════════════════════════════════════════════
            # 🎯 CONFIRMATION REQUIREMENT (ALL 7 required)
            # ═══════════════════════════════════════════════════════
            if confirmations < confirmations_required:
                logger.info(f"  ❌ {symbol} - Only {confirmations}/7 confirmations (need {confirmations_required}/7 {'overnight' if is_overnight else ''})")
                return None
            
            logger.info(f"  ✅ {symbol} - PASSED {confirmations}/7 TA confirmations! Calling AI for levels...")
            
            # Funding rate
            funding = await self.get_funding_rate(symbol)
            funding_pct = funding['funding_rate_percent']
            
            # Last 3 candles pattern
            last_3 = []
            for i in range(-3, 0):
                c_open = candles_5m[i][1]
                c_close = candles_5m[i][4]
                last_3.append("GREEN" if c_close > c_open else "RED")
            last_3_candles = " → ".join(last_3)
            
            # Get BTC 24h change for context
            btc_change = 0
            try:
                btc_url = "https://fapi.binance.com/fapi/v1/ticker/24hr?symbol=BTCUSDT"
                btc_resp = await self.client.get(btc_url, timeout=5)
                if btc_resp.status_code == 200:
                    btc_data = btc_resp.json()
                    btc_change = float(btc_data.get('priceChangePercent', 0))
            except:
                pass
            
            # BTC context logged but not used as filter
            logger.debug(f"  📊 BTC 24h: {btc_change:+.1f}%")
            
            # Structure checks removed - AI will handle quality filtering
            
            # Get 24h change from coin_data if available
            change_24h = coin_data.get('change_percent_24h', 0) if coin_data else 0
            volume_24h = coin_data.get('volume_24h', 0) if coin_data else 0
            
            # 🤖 AI VALIDATION WITH RATE LIMITING
            coin_info = {
                'symbol': symbol, 'change_24h': change_24h,
                'volume_24h': volume_24h, 'price': current_price
            }
            sr_levels = {}
            try:
                from app.services.enhanced_ta import find_support_resistance
                c_highs = [float(c[2]) for c in candles_5m]
                c_lows = [float(c[3]) for c in candles_5m]
                c_closes = [float(c[4]) for c in candles_5m]
                sr_levels = find_support_resistance(c_highs, c_lows, c_closes, current_price)
            except Exception as e:
                logger.warning(f"S/R calculation failed for {symbol}: {e}")
            
            candle_info = {
                'rsi': rsi_5m, 'ema9': ema9_5m, 'ema21': ema21_5m,
                'volume_ratio': volume_ratio, 'trend_5m': trend_5m, 'trend_15m': trend_15m,
                'funding_rate': funding_pct, 'price_to_ema9': price_to_ema9,
                'recent_high': recent_high, 'recent_low': recent_low,
                'last_3_candles': last_3_candles, 'btc_change': btc_change,
                'confirmations': confirmations,
                'support_resistance': sr_levels,
            }
            
            ai_result = await ai_validate_long_signal(coin_info, candle_info)
            
            # AI can reject if it doesn't like the setup
            if not ai_result or not ai_result.get('approved', False):
                reason = ai_result.get('reasoning', 'No reason') if ai_result else 'AI skipped (rate limit)'
                logger.info(f"  ❌ {symbol} - AI REJECTED: {reason}")
                return None
            
            # AI approved - use its levels
            confidence = ai_result.get('confidence', 7)
            recommendation = ai_result.get('recommendation', 'BUY')
            reasoning = ai_result.get('reasoning', 'AI approved')
            entry = current_price
            tp_pct = ai_result.get('tp_percent', 3.35)
            sl_pct = min(ai_result.get('sl_percent', 3.0), 3.0)  # 3% max = 60% loss at 20x
            tp = entry * (1 + tp_pct / 100)
            sl = entry * (1 - sl_pct / 100)
            
            logger.info(f"  ✅ {symbol} - AI APPROVED: TP {tp_pct:.1f}% / SL {sl_pct:.1f}%")
            logger.info(f"  ✅ {symbol} - SIGNAL READY ({confirmations}/7 TA) | TP: ${tp:.6f} | SL: ${sl:.6f}")
            
            return {
                'direction': 'LONG',
                'confidence': confidence * 10,
                'entry_price': entry,
                'stop_loss': sl,
                'take_profit': tp,
                'ai_recommendation': recommendation,
                'ai_reasoning': reasoning,
                'reason': f"🤖 {confirmations}/7 TA | {reasoning}",
                'ta_confirmations': confirmations
            }
            
        except Exception as e:
            logger.error(f"Error analyzing early pump for {symbol}: {e}")
            return None
    
    def _calculate_rsi(self, prices: List[float], period: int = 14) -> float:
        """Calculate Relative Strength Index (RSI)"""
        if len(prices) < period + 1:
            return 50.0  # Neutral if not enough data
        
        deltas = [prices[i] - prices[i-1] for i in range(1, len(prices))]
        gains = [d if d > 0 else 0 for d in deltas]
        losses = [-d if d < 0 else 0 for d in deltas]
        
        avg_gain = sum(gains[-period:]) / period
        avg_loss = sum(losses[-period:]) / period
        
        if avg_loss == 0:
            return 100.0
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def _calculate_ema(self, prices: List[float], period: int) -> float:
        """Calculate Exponential Moving Average"""
        if len(prices) < period:
            return sum(prices) / len(prices)
        
        multiplier = 2 / (period + 1)
        ema = sum(prices[:period]) / period
        
        for price in prices[period:]:
            ema = (price - ema) * multiplier + ema
        
        return ema

    def _calculate_vwap(self, candles: list) -> float:
        """Calculate Volume Weighted Average Price from OHLCV candles"""
        total_tp_vol = 0.0
        total_vol = 0.0
        for c in candles:
            high, low, close, volume = float(c[2]), float(c[3]), float(c[4]), float(c[5])
            typical_price = (high + low + close) / 3
            total_tp_vol += typical_price * volume
            total_vol += volume
        if total_vol == 0:
            return float(candles[-1][4])
        return total_tp_vol / total_vol
    
    def _is_candle_oversized(self, df, max_body_percent: float = 2.5) -> bool:
        """
        Check if current candle is oversized (prevents entering on parabolic dumps/pumps)
        
        Args:
            df: DataFrame with OHLC data
            max_body_percent: Maximum allowed candle body size (default 2.5%)
        
        Returns:
            True if candle is too big (should skip entry), False if safe to enter
        """
        if len(df) < 2:
            return False
        
        current_candle = df.iloc[-1]
        open_price = current_candle['open']
        close_price = current_candle['close']
        
        # Calculate candle body size as percentage
        body_percent = abs((close_price - open_price) / open_price * 100)
        
        # Also check against average candle size (last 10 candles)
        if len(df) >= 10:
            avg_body_sizes = []
            for i in range(-10, 0):
                candle = df.iloc[i]
                candle_body = abs((candle['close'] - candle['open']) / candle['open'] * 100)
                avg_body_sizes.append(candle_body)
            
            avg_body = sum(avg_body_sizes) / len(avg_body_sizes)
            
            # If current candle is > 2.5x average, it's oversized
            if body_percent > avg_body * 2.5:
                logger.info(f"⚠️ Oversized candle: {body_percent:.2f}% vs avg {avg_body:.2f}% (2.5x threshold)")
                return True
        
        # Absolute threshold: reject if candle body > max_body_percent
        if body_percent > max_body_percent:
            logger.info(f"⚠️ Candle too large: {body_percent:.2f}% body (max: {max_body_percent}%)")
            return True
        
        return False
    
    async def generate_top_gainer_signal(
        self, 
        min_change_percent: float = 10.0,
        max_symbols: int = 5
    ) -> Optional[Dict]:
        """
        Generate trading signal from top gainers
        
        OPTIMIZED FOR SHORTS: Prioritizes mean reversion on big pumps (10%+ gains)
        - Scans more symbols (5 vs 3) to find best short setups
        - Higher min_change (10% vs 5%) = better reversal candidates
        - Parabolic reversal detection prioritized for 50%+ pumps
        
        Returns:
            {
                'symbol': str,
                'direction': 'LONG' or 'SHORT',
                'entry_price': float,
                'stop_loss': float,
                'take_profit': float,
                'confidence': int,
                'reasoning': str,
                'trade_type': 'TOP_GAINER'
            }
        """
        try:
            # Get top gainers (optimized for shorts - higher % gains = better reversal candidates)
            gainers = await self.get_top_gainers(limit=max_symbols, min_change_percent=min_change_percent)
            
            if not gainers:
                logger.info("No top gainers found meeting criteria")
                return None
            
            # Clean up expired cooldowns
            now = datetime.utcnow()
            expired = [sym for sym, expires in shorts_cooldown.items() if expires <= now]
            for sym in expired:
                del shorts_cooldown[sym]
            
            # PRIORITY 1: PARABOLIC SHORTS DISABLED - No longer shorting top gainers
            # Only NORMAL_SHORTS and PARABOLIC strategies generate shorts now
            logger.debug("PARABOLIC/TOP-GAINER shorts DISABLED - only NORMAL_SHORTS and PARABOLIC generate shorts")
            
            # Regular analysis - LONGS ONLY (shorts disabled for top gainers)
            for gainer in gainers:
                symbol = gainer['symbol']
                
                # 🚫 Check blacklist
                normalized = symbol.replace('/USDT', '').replace('USDT', '')
                if normalized in BLACKLISTED_SYMBOLS or symbol in BLACKLISTED_SYMBOLS:
                    logger.info(f"🚫 {symbol} BLACKLISTED - skipping")
                    continue
                
                # 🔒 SMALL COIN FILTER: Require 30%+ pump for low-volume coins
                # Small coins (<$500k volume) are riskier, need stronger exhaustion
                volume_24h = gainer.get('volume_24h', 0)
                change_percent = gainer.get('change_percent', 0)
                if volume_24h < 500000 and change_percent < 30.0:
                    logger.info(f"⚠️ {symbol} SKIPPED - Small coin (${volume_24h/1000:.0f}k vol) needs 30%+ pump, only +{change_percent:.1f}%")
                    continue
                
                # Check if symbol is in cooldown (lost SHORT recently)
                if symbol in shorts_cooldown:
                    remaining_min = (shorts_cooldown[symbol] - now).total_seconds() / 60
                    logger.info(f"⏰ {symbol} SKIPPED - SHORT cooldown active ({remaining_min:.0f} min left)")
                    continue
                
                # Analyze momentum
                momentum = await self.analyze_momentum(symbol)
                
                if not momentum:
                    continue
                
                # SHORTS DISABLED for top gainers - only NORMAL_SHORTS scanner generates shorts
                if momentum['direction'] == 'SHORT':
                    logger.debug(f"{symbol} - SHORT signal skipped (top-gainer shorts disabled)")
                    continue
                
                entry_price = momentum['entry_price']
                
                # LONG @ 20x leverage - SINGLE TP
                # TP: 3.35% = 67% profit at 20x
                # SL: 3.25% = 65% loss at 20x
                stop_loss = entry_price * (1 - 3.25 / 100)  # 65% loss at 20x
                take_profit_1 = entry_price * (1 + 3.35 / 100)  # 67% profit at 20x
                take_profit_2 = None  # Single TP only
                take_profit_3 = None
                
                return {
                    'symbol': symbol,
                    'direction': 'LONG',
                    'entry_price': entry_price,
                    'stop_loss': stop_loss,
                    'take_profit': take_profit_1,  # Backward compatible
                    'take_profit_1': take_profit_1,
                    'take_profit_2': take_profit_2,
                    'take_profit_3': take_profit_3,
                    'confidence': momentum['confidence'],
                    'reasoning': f"Top Gainer: {gainer['change_percent']}% in 24h | {momentum['reason']}",
                    'trade_type': 'TOP_GAINER',
                    'leverage': 20,  # 20x leverage
                    '24h_change': gainer['change_percent'],
                    '24h_volume': gainer['volume_24h'],
                    'is_parabolic_reversal': False
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Error generating top gainer signal: {e}")
            return None
    
    async def generate_parabolic_dump_signal(
        self,
        min_change_percent: float = 1.0,
        max_symbols: int = 10
    ) -> Optional[Dict]:
        """
        🚀 DEDICATED PARABOLIC DUMP SCANNER 🚀
        
        Scans for EXHAUSTED parabolic pumps ready to reverse.
        🔓 OPEN: Any coins gaining 1%+ on the day
        AI validation ensures quality - filters happen in analysis!
        
        Strategy:
        - Scans multiple gainers (1%+)
        - Scores each by overextension (RSI, EMA distance, momentum)
        - Returns strongest parabolic reversal candidate
        - Triple TPs: 4%, 8%, 12% (20%, 40%, 60% at 5x leverage)
        
        Returns:
        - Signal dict with PARABOLIC_REVERSAL signal type
        """
        try:
            logger.info("🔥 ═══════════════════════════════════════════════════════")
            logger.info(f"🔥 PARABOLIC DUMP SCANNER - Looking for any exhausted pumps")
            logger.info("🔥 ═══════════════════════════════════════════════════════")
            
            # Get gainers (1%+)
            gainers = await self.get_top_gainers(limit=max_symbols, min_change_percent=min_change_percent)
            
            if not gainers:
                logger.info(f"No {min_change_percent}%+ gainers found")
                return None
            
            logger.info(f"📊 Found {len(gainers)} coins with positive pumps")
            
            # Clean up expired cooldowns
            now = datetime.utcnow()
            expired = [sym for sym, expires in shorts_cooldown.items() if expires <= now]
            for sym in expired:
                del shorts_cooldown[sym]
            
            # Evaluate ALL candidates and score them
            candidates = []
            
            for gainer in gainers:
                symbol = gainer['symbol']
                change_pct = gainer['change_percent']
                
                # 🔥 PARABOLIC SHORTS BYPASS COOLDOWN!
                # Dedicated parabolic scanner doesn't check cooldown - we want ALL 50%+ reversals
                logger.info(f"🎯 Analyzing: {symbol} @ +{change_pct:.1f}% (parabolic scanner - no cooldown)")
                
                # 🚀 PARABOLIC EXHAUSTION LOGIC (balanced - not too strict, not too loose)
                # For 50%+ pumps: check overextension + exhaustion signs + trend context
                try:
                    # Get 5m candles for analysis
                    candles_5m = await self.fetch_candles(symbol, '5m', 20)
                    
                    if not candles_5m or len(candles_5m) < 14:
                        logger.info(f"  ❌ {symbol} - Insufficient 5m candle data")
                        continue
                    
                    # Calculate basic indicators
                    closes_5m = [float(c[4]) for c in candles_5m]
                    current_price = closes_5m[-1]
                    rsi_5m = self._calculate_rsi(closes_5m, 14)
                    
                    # Volume check (proper averaging with minimum sample guard)
                    volumes = [float(c[5]) for c in candles_5m]
                    current_volume = volumes[-1]
                    
                    # Need minimum 5 historical candles for volume context
                    if len(volumes) < 6:
                        logger.info(f"  ❌ {symbol} - Insufficient volume data ({len(volumes)} candles, need 6+)")
                        continue
                    
                    # Average last N candles (excluding current)
                    if len(volumes) >= 10:
                        vol_samples = volumes[-10:-1]  # Last 9 candles
                    else:
                        vol_samples = volumes[:-1]  # All but current
                    avg_volume = sum(vol_samples) / len(vol_samples)
                    volume_ratio = current_volume / avg_volume
                    
                    # Calculate EMA9 for overextension check
                    ema9 = self._calculate_ema(closes_5m, 9)  # Returns float, not list
                    price_to_ema9_dist = ((current_price - ema9) / ema9) * 100
                    
                    # 🔥 TREND VALIDATION: Must be overextended above EMA9
                    is_overextended = price_to_ema9_dist > 3.0  # Increased from 1.5% to 3.0% for much stricter entries
                    
                    if not is_overextended:
                        logger.info(f"  ❌ {symbol} - Not overextended (only {price_to_ema9_dist:+.1f}% above EMA9, need >3.0%)")
                        continue
                    
                    # 🔥 TREND CONFIRMATION: Market structure must turn bearish
                    # Check for Lower Highs or Lower Lows on 5m timeframe
                    highs_5m = [float(c[2]) for c in candles_5m]
                    lows_5m = [float(c[3]) for c in candles_5m]
                    
                    # Trend: Recent high must be lower than previous local high (Lower High)
                    # or Recent low must be lower than previous local low (Lower Low)
                    local_high_1 = max(highs_5m[-5:])  # Last 25 mins
                    local_high_2 = max(highs_5m[-10:-5]) # 25-50 mins ago
                    has_lower_high = local_high_1 < local_high_2
                    
                    local_low_1 = min(lows_5m[-5:])
                    local_low_2 = min(lows_5m[-10:-5])
                    has_lower_low = local_low_1 < local_low_2
                    
                    # Trend must show signs of breaking (Lower High or Lower Low)
                    trend_turning = has_lower_high or has_lower_low
                    
                    if not trend_turning:
                        logger.info(f"  ❌ {symbol} - Trend still bullish (No Lower High/Low detected)")
                        continue
                    
                    # 🔥 REVERSAL DETECTION - Clear sign of trend change
                    # Signal 1: High RSI (overbought)
                    high_rsi = rsi_5m >= 75  # Increased from 65 to 75 for stricter overbought check
                    
                    # Signal 2: Upper wick rejection (selling pressure at top)
                    current_candle = candles_5m[-1]
                    current_open = float(current_candle[1])
                    current_high = float(current_candle[2])
                    current_low = float(current_candle[3])
                    wick_size = ((current_high - current_price) / current_price) * 100
                    has_rejection = wick_size >= 1.0  # Increased from 0.5% to 1.0%
                    
                    # Signal 3: Bearish confirmation (Red Candle) - REQUIRED
                    is_bearish_candle = current_price < current_open
                    
                    # 🎯 BALANCED ENTRY CRITERIA
                    # REQUIRED: Trend turning + Red candle
                    # PLUS: At least one of (RSI overbought OR Wick rejection)
                    has_exhaustion_sign = high_rsi or has_rejection
                    good_volume = volume_ratio >= 1.2  # Relaxed from 1.5
                    
                    # Entry: Trend Change + Bearish Close + (RSI OR Wick)
                    has_strong_signal = trend_turning and is_bearish_candle and has_exhaustion_sign
                    
                    # 🔥 ENTRY TIMING: Price in lower portion of candle
                    candle_range = current_high - current_low if current_high > current_low else 0.0001
                    price_position_in_candle = (current_price - current_low) / candle_range
                    is_good_entry_timing = price_position_in_candle <= 0.8  # Relaxed from 0.7 to 0.8
                    
                    if has_strong_signal and good_volume and is_good_entry_timing:
                        # Build exhaustion reason
                        reasons = []
                        if high_rsi:
                            reasons.append(f"RSI {rsi_5m:.0f}")
                        if has_rejection:
                            reasons.append(f"{wick_size:.1f}% wick")
                        if trend_turning:
                            reasons.append("Lower High/Low")
                        if is_bearish_candle:
                            reasons.append("Red candle")
                        
                        exhaustion_reason = " + ".join(reasons) if reasons else f"RSI {rsi_5m:.0f}"
                        
                        # Confidence based on signal strength
                        exhaustion_count = sum([high_rsi, has_rejection, trend_turning, is_bearish_candle])
                        if exhaustion_count >= 4:
                            confidence = 95
                        elif exhaustion_count >= 3:
                            confidence = 90
                        else:
                            confidence = 85
                        
                        logger.info(f"  ✅ {symbol} - PARABOLIC EXHAUSTION: {exhaustion_reason} | Vol: {volume_ratio:.1f}x")
                        
                        # Store as candidate with full data for AI validation
                        candidates.append({
                            'symbol': symbol,
                            'gainer': gainer,
                            'momentum': {
                                'direction': 'SHORT',
                                'confidence': confidence,
                                'entry_price': current_price,
                                'reason': f'🎯 PARABOLIC REVERSAL | +{change_pct:.1f}% exhausted | {exhaustion_reason} | Vol: {volume_ratio:.1f}x'
                            },
                            'score': change_pct * 0.4 + confidence * 0.6,
                            # AI validation data
                            'rsi': rsi_5m,
                            'ema9': ema9,
                            'price_to_ema9': price_to_ema9_dist,
                            'volume_ratio': volume_ratio,
                            'wick_size': wick_size,
                            'is_bearish': is_bearish_candle,
                            'trend_turning': trend_turning,
                            'exhaustion_count': exhaustion_count,
                            'recent_high': current_high,
                            'recent_low': current_low
                        })
                        continue
                    else:
                        skip_reasons = []
                        if not is_bearish_candle:
                            skip_reasons.append("No red candle")
                        if not has_exhaustion_sign:
                            skip_reasons.append(f"RSI {rsi_5m:.0f} (need ≥65) or wick {wick_size:.1f}% (need ≥0.5%)")
                        if not good_volume:
                            skip_reasons.append(f"Vol {volume_ratio:.1f}x (need ≥1.2x)")
                        if not is_good_entry_timing:
                            skip_reasons.append(f"Price at {price_position_in_candle*100:.0f}% of candle (need ≤80%)")
                        logger.info(f"  ❌ {symbol} - {', '.join(skip_reasons)}")
                        continue
                        
                except Exception as e:
                    logger.error(f"  ❌ {symbol} - Error analyzing: {e}")
                    continue
            
            if not candidates:
                logger.info("No valid parabolic reversal candidates found")
                return None
            
            # Sort by score (highest first) - TA already confirmed, just pick best
            candidates.sort(key=lambda x: x['score'], reverse=True)
            
            # 🎯 TA-FIRST: Take best candidate (already passed all TA filters)
            # AI only sets TP/SL levels - can't reject a TA-confirmed signal
            best = candidates[0]
            symbol = best['symbol']
            entry_price = best['momentum']['entry_price']
            
            logger.info(f"✅ {symbol} PASSED TA filters (score: {best['score']:.1f}) - calling AI for levels...")
            
            # Get BTC change for context
            try:
                btc_change = await self._get_btc_24h_change()
            except:
                btc_change = 0
            
            # Prepare data for AI to set optimal TP/SL
            coin_data = {
                'symbol': symbol,
                'change_24h': best['gainer']['change_percent'],
                'volume_24h': best['gainer'].get('volume_24h', 0),
                'price': entry_price
            }
            
            sr_levels = {}
            try:
                from app.services.enhanced_ta import find_support_resistance
                parabolic_candles = best.get('candles_5m', [])
                if parabolic_candles and len(parabolic_candles) >= 10:
                    c_highs = [float(c[2]) for c in parabolic_candles]
                    c_lows = [float(c[3]) for c in parabolic_candles]
                    c_closes = [float(c[4]) for c in parabolic_candles]
                    sr_levels = find_support_resistance(c_highs, c_lows, c_closes, entry_price)
            except Exception as e:
                logger.warning(f"S/R calculation failed for parabolic {symbol}: {e}")
            
            candle_data = {
                'rsi': best.get('rsi', 75),
                'ema9': best.get('ema9', entry_price),
                'price_to_ema9': best.get('price_to_ema9', 2.0),
                'volume_ratio': best.get('volume_ratio', 1.5),
                'wick_size': best.get('wick_size', 1.0),
                'is_bearish': best.get('is_bearish', True),
                'recent_high': best.get('recent_high', entry_price),
                'recent_low': best.get('recent_low', entry_price * 0.9),
                'btc_change': btc_change,
                'exhaustion_count': best.get('exhaustion_count', 3),
                'trend_turning': best.get('trend_turning', True),
                'support_resistance': sr_levels,
            }
            
            ai_result = await ai_validate_short_signal(coin_data, candle_data)
            
            # Use AI levels if available, otherwise use defaults
            if ai_result:
                tp_percent = ai_result.get('tp_percent', 6.0)
                sl_percent = min(ai_result.get('sl_percent', 3.0), 3.0)  # Cap at 3% = 60% max loss
                ai_reasoning = ai_result.get('reasoning', 'TA confirmed')
                ai_quality = ai_result.get('entry_quality', 'A')
            else:
                # Default levels if AI unavailable
                tp_percent = 6.0  # 6% TP = 120% profit at 20x
                sl_percent = 3.0  # 3% SL = 60% loss at 20x (safe from liquidation)
                ai_reasoning = "TA-confirmed parabolic exhaustion"
                ai_quality = "TA"
                logger.info(f"  ⚠️ {symbol} - AI unavailable, using default levels (TP 6%, SL 3%)")
            
            stop_loss = entry_price * (1 + sl_percent / 100)
            take_profit_1 = entry_price * (1 - tp_percent / 100)
            
            signal = {
                'symbol': symbol,
                'direction': 'SHORT',
                'entry_price': entry_price,
                'stop_loss': stop_loss,
                'take_profit': take_profit_1,
                'take_profit_1': take_profit_1,
                'take_profit_2': None,
                'take_profit_3': None,
                'confidence': best['momentum']['confidence'],
                'reasoning': f"🤖 [{ai_quality}]: {ai_reasoning}",
                'trade_type': 'PARABOLIC_REVERSAL',
                'leverage': 20,
                '24h_change': best['gainer']['change_percent'],
                '24h_volume': best['gainer'].get('volume_24h', 0),
                'is_parabolic_reversal': True,
                'parabolic_score': best['score'],
                'ai_recommendation': 'SELL',
                'ai_reasoning': ai_reasoning,
                'ai_quality': ai_quality,
                'tp_percent': tp_percent,
                'sl_percent': sl_percent,
                'risk_reward': tp_percent / sl_percent if sl_percent > 0 else 1.5
            }
            
            logger.info(f"✅ PARABOLIC SIGNAL: {symbol} | TP: -{tp_percent:.1f}% | SL: +{sl_percent:.1f}%")
            return signal
            
        except Exception as e:
            logger.error(f"Error in parabolic dump scanner: {e}", exc_info=True)
            return None
    
    async def generate_early_pump_long_signal(
        self,
        min_change: float = -50.0,
        max_change: float = 50.0,
        max_symbols: int = 30
    ) -> Optional[Dict]:
        """
        Generate LONG signals from ANY coin showing bullish momentum
        
        🎯 RANGE: -50% to +50% (catches reversals AND fresh pumps!)
        AI validation ensures quality - filters happen in analysis!
        
        Returns:
            Same signal format as generate_top_gainer_signal but for LONGS
        """
        try:
            # Get early pump candidates (5-20% range with high volume)
            pumpers = await self.get_early_pumpers(limit=max_symbols, min_change=min_change, max_change=max_change)
            
            if not pumpers:
                logger.info(f"❌ No coins found in {min_change}% to {max_change}% range")
                return None
            
            logger.info(f"📈 Found {len(pumpers)} pumping coins to analyze:")
            
            # Analyze each early pumper for LONG entry
            for idx, pumper in enumerate(pumpers, 1):
                symbol = pumper['symbol']
                
                # 🚫 BLACKLIST CHECK - Skip banned symbols immediately
                normalized = symbol.replace('/', '').replace('USDT', '')
                if normalized in BLACKLISTED_SYMBOLS or symbol in BLACKLISTED_SYMBOLS:
                    logger.info(f"  ⛔ {symbol} BLACKLISTED - skipping")
                    continue
                
                logger.info(f"  [{idx}/{len(pumpers)}] {symbol}: +{pumper['change_percent']:.2f}%")
                
                # 🤖 AI-POWERED ANALYSIS - Pass coin data for context
                coin_data_for_ai = {
                    'change_percent_24h': pumper.get('change_percent_24h', pumper.get('change_percent', 0)),
                    'volume_24h': pumper.get('volume_24h', 0)
                }
                
                # 🎯 FRESH MOMENTUM FILTERS (Jan 2026 - Entry timing fix)
                # Only long coins with FRESH momentum, not exhausted pumps
                
                # Basic liquidity check (SUPER RELAXED)
                liq_ok = pumper.get('volume_24h', 0) >= 500_000  # $500K+ liquidity (was $2M)
                if not liq_ok:
                    logger.info(f"  ⏭️ {symbol} - Low liquidity (<$500K)")
                    continue
                
                # Fetch candles for freshness analysis
                candles_5m = await self.fetch_candles(symbol, '5m', limit=20)
                candles_15m = await self.fetch_candles(symbol, '15m', limit=8)
                
                if not candles_5m or len(candles_5m) < 14:
                    logger.info(f"  ⏭️ {symbol} - Insufficient candle data")
                    continue
                
                closes_5m = [float(c[4]) for c in candles_5m]
                current_price = closes_5m[-1]
                
                # Calculate RSI
                rsi_5m = self._calculate_rsi(closes_5m, 14)
                
                # 🚫 EXHAUSTION CHECK: RSI too high = already pumped too much (RELAXED)
                if rsi_5m > 85:
                    logger.info(f"  ⏭️ {symbol} - RSI {rsi_5m:.0f} too high (exhausted, need ≤85)")
                    continue
                
                # 🚫 EXHAUSTION CHECK: RSI too low = no momentum (RELAXED)
                if rsi_5m < 20:
                    logger.info(f"  ⏭️ {symbol} - RSI {rsi_5m:.0f} too low (weak momentum, need ≥20)")
                    continue
                
                # Calculate EMA for freshness check
                ema9 = self._calculate_ema(closes_5m, 9)
                ema21 = self._calculate_ema(closes_5m, 21)
                price_to_ema9 = ((current_price - ema9) / ema9) * 100 if ema9 > 0 else 0
                
                # 🚫 EXHAUSTION CHECK: Price WAY too far above EMA (SUPER RELAXED)
                if price_to_ema9 > 15.0:
                    logger.info(f"  ⏭️ {symbol} - Price {price_to_ema9:.1f}% above EMA9 (overextended, need ≤15%)")
                    continue
                
                # ✅ TREND CHECK: Price should be above EMA (RELAXED - allow dips)
                if price_to_ema9 < -8.0:
                    logger.info(f"  ⏭️ {symbol} - Price {price_to_ema9:.1f}% below EMA9 (breaking down)")
                    continue
                
                # ✅ TREND CHECK: EMA9 > EMA21 (RELAXED - allow catching reversals)
                ema_gap = ((ema9 - ema21) / ema21) * 100 if ema21 > 0 else 0
                if ema_gap < -2.0:  # Allow bearish setup for reversal plays
                    logger.info(f"  ⏭️ {symbol} - Too bearish EMA gap {ema_gap:.2f}% (need ≥-2%)")
                    continue
                
                # 🔥 15m ACCELERATION CHECK: Recent momentum should be positive
                if candles_15m and len(candles_15m) >= 4:
                    closes_15m = [float(c[4]) for c in candles_15m]
                    change_15m = ((closes_15m[-1] - closes_15m[-4]) / closes_15m[-4]) * 100  # Last 1h on 15m
                    
                    # Must have some momentum (can be slightly negative if recovering)
                    if change_15m < -1.0:
                        logger.info(f"  ⏭️ {symbol} - 15m change {change_15m:.1f}% too weak (need ≥-1.0%)")
                        continue
                    
                    # But not too much (already pumped)
                    if change_15m > 15.0:
                        logger.info(f"  ⏭️ {symbol} - 15m change {change_15m:.1f}% too high (exhausted)")
                        continue
                else:
                    change_15m = 0
                
                # 🔥 VOLUME SURGE CHECK
                volumes = [float(c[5]) for c in candles_5m]
                if len(volumes) >= 6:
                    avg_vol = sum(volumes[:-1]) / len(volumes[:-1])
                    vol_ratio = volumes[-1] / avg_vol if avg_vol > 0 else 1.0
                else:
                    vol_ratio = 1.0
                
                if vol_ratio < 0.7:
                    logger.info(f"  ⏭️ {symbol} - Volume ratio {vol_ratio:.1f}x too low (need ≥0.7x)")
                    continue
                
                logger.info(f"  ✅ {symbol} FRESH: RSI {rsi_5m:.0f} | EMA dist {price_to_ema9:+.1f}% | 15m {change_15m:+.1f}% | Vol {vol_ratio:.1f}x")
                
                # Use AI-powered analysis (replaces old rule-based logic)
                momentum = await self.analyze_early_pump_long(symbol, coin_data=coin_data_for_ai)
                
                if not momentum or momentum['direction'] != 'LONG':
                    continue
                
                # Found a valid LONG signal! AI has already approved.
                entry_price = momentum['entry_price']
                
                # Use AI's calculated levels (already set by ai_validate_long_signal)
                stop_loss = momentum.get('stop_loss', entry_price * 0.9675)
                take_profit_1 = momentum.get('take_profit', entry_price * 1.0335)
                take_profit_2 = None  # Single TP only
                take_profit_3 = None
                
                # Get tier and pump data
                tier = pumper.get('tier', '30m')  # Default to 30m if not set
                pump_data = pumper.get('fresh_pump_data', {})
                tier_change = pump_data.get('candle_change_percent', pumper['change_percent'])
                volume_ratio = pump_data.get('volume_ratio', 0)
                
                # Tier-specific labels
                tier_labels = {
                    '5m': '⚡ ULTRA-EARLY',
                    '15m': '🔥 EARLY',
                    '30m': '✅ FRESH'
                }
                tier_label = tier_labels.get(tier, '✅ FRESH')
                
                # Get AI analysis info
                ai_recommendation = momentum.get('ai_recommendation', 'BUY')
                ai_reasoning = momentum.get('ai_reasoning', '')
                
                logger.info(f"✅ AI {ai_recommendation} LONG: {symbol} @ +{tier_change}% ({tier})")
                
                # Build reasoning with AI analysis
                reasoning_parts = [f"🤖 AI: {ai_recommendation}"]
                if ai_reasoning:
                    reasoning_parts.append(ai_reasoning)
                reasoning_parts.append(f"{tier_label} | +{tier_change}% pump")
                
                signal = {
                    'symbol': symbol,
                    'direction': 'LONG',
                    'entry_price': entry_price,
                    'stop_loss': stop_loss,
                    'take_profit': take_profit_1,
                    'take_profit_1': take_profit_1,
                    'take_profit_2': take_profit_2,
                    'take_profit_3': take_profit_3,
                    'confidence': momentum['confidence'],
                    'reasoning': " | ".join(reasoning_parts),
                    'trade_type': 'TOP_GAINER',
                    'leverage': 20,  # 20x leverage
                    '24h_change': pumper.get('change_percent_24h', pumper.get('change_percent', 0)),
                    '24h_volume': pumper.get('volume_24h', 0),
                    'is_parabolic_reversal': False,
                    'tier': tier,
                    'tier_label': tier_label,
                    'tier_change': tier_change,
                    'volume_ratio': volume_ratio,
                    'ai_recommendation': ai_recommendation,
                    'ai_reasoning': ai_reasoning
                }
                
                return signal
            
            logger.info("No valid LONG entries found in early pumpers")
            return None
            
        except Exception as e:
            logger.error(f"Error generating early pump LONG signal: {e}")
            return None
    
    async def add_to_watchlist(self, db_session: Session, symbol: str, price: float, change_percent: float):
        """
        Add a symbol to the 48-hour watchlist for delayed reversal monitoring.
        
        Args:
            db_session: Database session
            symbol: Trading pair (e.g., 'AIXBT/USDT')
            price: Current price
            change_percent: Current 24h% change
        """
        from app.models import TopGainerWatchlist
        
        try:
            # Check if already in watchlist
            existing = db_session.query(TopGainerWatchlist).filter(
                TopGainerWatchlist.symbol == symbol
            ).first()
            
            if existing:
                # Update if new peak
                if change_percent > existing.peak_change_percent:
                    existing.peak_price = price
                    existing.peak_change_percent = change_percent
                existing.last_checked = datetime.utcnow()
                db_session.commit()
                logger.info(f"Updated watchlist for {symbol}: {change_percent}% (peak: {existing.peak_change_percent}%)")
            else:
                # Add new entry
                watchlist_entry = TopGainerWatchlist(
                    symbol=symbol,
                    peak_price=price,
                    peak_change_percent=change_percent,
                    first_seen=datetime.utcnow(),
                    last_checked=datetime.utcnow()
                )
                db_session.add(watchlist_entry)
                db_session.commit()
                logger.info(f"Added {symbol} to watchlist: {change_percent}% | Will monitor for 48h")
                
        except Exception as e:
            logger.error(f"Error adding {symbol} to watchlist: {e}")
            db_session.rollback()
    
    async def cleanup_old_watchlist(self, db_session: Session):
        """Remove watchlist entries older than 48 hours"""
        from app.models import TopGainerWatchlist
        
        try:
            cutoff_time = datetime.utcnow() - timedelta(hours=48)
            deleted = db_session.query(TopGainerWatchlist).filter(
                TopGainerWatchlist.first_seen < cutoff_time
            ).delete()
            
            if deleted > 0:
                db_session.commit()
                logger.info(f"Cleaned up {deleted} expired watchlist entries (>48h old)")
                
        except Exception as e:
            logger.error(f"Error cleaning up watchlist: {e}")
            db_session.rollback()
    
    async def get_watchlist_symbols(self, db_session: Session) -> List[Dict]:
        """
        Get all symbols currently on the watchlist.
        
        Returns:
            List of dicts with symbol, peak_change_percent, hours_tracked
        """
        from app.models import TopGainerWatchlist
        
        try:
            watchlist = db_session.query(TopGainerWatchlist).filter(
                TopGainerWatchlist.still_monitoring == True
            ).all()
            
            return [
                {
                    'symbol': entry.symbol,
                    'peak_change_percent': entry.peak_change_percent,
                    'hours_tracked': entry.hours_tracked,
                    'first_seen': entry.first_seen
                }
                for entry in watchlist
            ]
        except Exception as e:
            logger.error(f"Error getting watchlist: {e}")
            return []
    
    async def mark_watchlist_signal_sent(self, db_session: Session, symbol: str):
        """Mark a watchlist symbol as having sent a reversal signal"""
        from app.models import TopGainerWatchlist
        
        try:
            entry = db_session.query(TopGainerWatchlist).filter(
                TopGainerWatchlist.symbol == symbol
            ).first()
            
            if entry:
                entry.still_monitoring = False
                db_session.commit()
                logger.info(f"Marked {symbol} as signal sent (will stop monitoring)")
        except Exception as e:
            logger.error(f"Error marking watchlist entry: {e}")
            db_session.rollback()
    
    async def close(self):
        """Close HTTP client connection"""
        if self.client:
            await self.client.aclose()
            self.client = None


async def broadcast_top_gainer_signal(bot, db_session):
    """
    Scan for signals and broadcast to users with top_gainers_mode_enabled
    Supports 3 modes: SHORTS_ONLY, LONGS_ONLY, or BOTH
    
    - SHORTS: Scan 28%+ gainers for mean reversion (wait for exhausted pumps!)
    - LONGS: Scan 5-20% early pumps for momentum entries (catch pumps early!)
    - BOTH: Try both scans (shorts first, then longs)
    """
    from app.models import User, UserPreference, Signal, Trade
    from app.services.bitunix_trader import execute_bitunix_trade
    from datetime import datetime
    import logging
    
    logger = logging.getLogger(__name__)
    
    # 🛑 KILL SWITCH - Check if scanning is disabled
    if SCANNING_DISABLED:
        logger.info("🛑 SCANNING DISABLED - Skipping scan (set SCANNING_DISABLED=False to re-enable)")
        return
    
    try:
        # 🔥 CHECK DAILY LIMIT FIRST (max 6 signals per day)
        current_count = get_daily_signal_count()
        if current_count >= MAX_DAILY_SIGNALS:
            logger.info(f"⚠️ DAILY LIMIT: {current_count}/{MAX_DAILY_SIGNALS} signals sent today - skipping scan")
            return
        
        remaining = MAX_DAILY_SIGNALS - current_count
        logger.info(f"📊 Daily signals: {current_count}/{MAX_DAILY_SIGNALS} ({remaining} remaining)")
        
        service = TopGainersSignalService()
        await service.initialize()
        
        # Get all users with top gainers mode enabled (regardless of auto-trading status)
        users_with_mode = db_session.query(User).join(UserPreference).filter(
            UserPreference.top_gainers_mode_enabled == True
        ).all()
        
        if not users_with_mode:
            logger.info("No users with top gainers mode enabled")
            await service.close()
            return
        
        # Count users with auto-trading enabled (only active subscribers can auto-trade)
        auto_traders = [u for u in users_with_mode if u.preferences and u.preferences.auto_trading_enabled and (u.is_subscribed or u.is_admin)]
        manual_traders = [u for u in users_with_mode if u not in auto_traders]
        
        logger.info(f"Scanning for signals: {len(users_with_mode)} total ({len(auto_traders)} auto, {len(manual_traders)} manual)")
        
        # 🔥 CRITICAL FIX: Check if ANY user wants SHORTS or LONGS
        # Don't just check first user - check ALL users' preferences!
        wants_shorts = False
        wants_longs = False
        
        for user in users_with_mode:
            prefs = user.preferences
            if prefs:
                user_mode = getattr(prefs, 'top_gainers_trade_mode', 'shorts_only')
                if user_mode in ['shorts_only', 'both']:
                    wants_shorts = True
                if user_mode in ['longs_only', 'both']:
                    wants_longs = True
        
        logger.info(f"User Preferences: SHORTS={wants_shorts}, LONGS={wants_longs}")
        
        # ═══════════════════════════════════════════════════════
        # 🎯 AUTOMATIC MARKET REGIME DETECTION
        # ═══════════════════════════════════════════════════════
        market_regime = await detect_market_regime()
        regime_focus = market_regime.get('focus', 'BOTH')
        
        logger.info(f"🎯 Market Regime: {market_regime['regime']} | Focus: {regime_focus}")
        logger.info(f"   {market_regime['reasoning']}")
        
        # 🔥 CRITICAL: Generate ALL signal types if wanted (don't exit early!)
        parabolic_signal = None
        normal_short_signal = None  # AI-powered normal shorts
        long_signal = None
        
        # ═══════════════════════════════════════════════════════
        # 🔴 SHORTS DISABLED - All short strategies keep losing
        # ═══════════════════════════════════════════════════════
        if SHORTS_DISABLED and wants_shorts:
            logger.info("🔴 SHORTS DISABLED - All short strategies paused until proven edge found")
            wants_shorts = False
        
        # ═══════════════════════════════════════════════════════
        # 🎯 EXTREME REGIME: Disable opposite direction
        # ═══════════════════════════════════════════════════════
        if market_regime.get('disable_longs') and wants_longs:
            logger.warning("🔴 EXTREME BEARISH: LONGS DISABLED - BTC dumping hard")
            wants_longs = False
        
        if market_regime.get('disable_shorts') and wants_shorts:
            logger.warning("🟢 EXTREME BULLISH: SHORTS DISABLED - BTC pumping hard")
            wants_shorts = False
        
        # ═══════════════════════════════════════════════════════
        # 🎯 REGIME-BASED PRIORITY ADJUSTMENT
        # ═══════════════════════════════════════════════════════
        shorts_first = regime_focus == 'SHORTS'
        longs_first = regime_focus == 'LONGS'
        
        if shorts_first:
            logger.info("📉 BEARISH REGIME: Scanning SHORTS first, LONGS second")
        elif longs_first:
            logger.info("📈 BULLISH REGIME: Scanning LONGS first, SHORTS second")
        
        # ═══════════════════════════════════════════════════════
        # CHECK SHORT DAILY LIMIT (max 3 shorts per day)
        # ═══════════════════════════════════════════════════════
        short_count = get_daily_short_count()
        shorts_remaining = MAX_DAILY_SHORTS - short_count
        if wants_shorts and shorts_remaining <= 0:
            logger.info(f"⚠️ SHORT DAILY LIMIT: {short_count}/{MAX_DAILY_SHORTS} - skipping short scans")
            wants_shorts = False  # Skip all short scanning
        elif wants_shorts:
            logger.info(f"📊 Daily shorts: {short_count}/{MAX_DAILY_SHORTS} ({shorts_remaining} remaining)")
        
        # ═══════════════════════════════════════════════════════
        # 🤖 AI-POWERED LONGS - PRIORITY #1 (Best performer!)
        # ═══════════════════════════════════════════════════════
        if LONGS_DISABLED and wants_longs:
            logger.info("🟢 LONGS DISABLED - Skipping long scans")
            wants_longs = False
        
        # 4-hour fixed window check (Starts at first trade)
        from datetime import datetime, timedelta
        from app.models import Signal
        four_hours_ago = datetime.utcnow() - timedelta(hours=4)
        recent_signals = db_session.query(Signal).filter(
            Signal.created_at >= four_hours_ago,
            Signal.outcome.isnot(None)
        ).order_by(Signal.created_at.asc()).all()
        
        recent_signal_count = len(recent_signals)
        
        # 🔥 NEW: Track if we're in "perfect trades only" mode (limit reached but still scanning)
        perfect_trades_only = False
        
        if recent_signal_count >= 2:
            # Check if the 4h window started by the first signal has passed
            first_signal_time = recent_signals[0].created_at
            if datetime.utcnow() < first_signal_time + timedelta(hours=4):
                # 🔥 DON'T STOP SCANNING - just require 100% confidence (10/10 + A+ grade)
                logger.info(f"⚠️ SIGNAL LIMIT: {recent_signal_count}/2 trades in window - switching to PERFECT TRADES ONLY mode (10/10 A+ required)")
                perfect_trades_only = True

        # ═══════════════════════════════════════════════════════
        # 🎯 REGIME-AWARE SCANNING ORDER
        # BEARISH: Shorts first, then longs
        # BULLISH/NEUTRAL: Longs first, then shorts
        # ═══════════════════════════════════════════════════════
        
        async def scan_longs():
            nonlocal long_signal
            if wants_longs:
                logger.info("🤖 ═══════════════════════════════════════════════════════")
                logger.info("🤖 AI-POWERED LONGS SCANNER")
                logger.info("🤖 ═══════════════════════════════════════════════════════")
                long_signal = await service.generate_early_pump_long_signal()
                
                if long_signal and long_signal['direction'] == 'LONG':
                    logger.info(f"✅ AI LONG found: {long_signal['symbol']} | AI: {long_signal.get('ai_recommendation')} | Confidence: {long_signal.get('confidence')}")
        
        async def scan_shorts():
            nonlocal parabolic_signal, normal_short_signal
            global last_short_signal_time
            
            if wants_shorts and last_short_signal_time:
                hours_since_last = (datetime.utcnow() - last_short_signal_time).total_seconds() / 3600
                if hours_since_last < SHORT_GLOBAL_COOLDOWN_HOURS:
                    remaining = SHORT_GLOBAL_COOLDOWN_HOURS - hours_since_last
                    logger.info(f"⏳ SHORT GLOBAL COOLDOWN: {remaining:.1f}h remaining (1h between shorts)")
                    return
            
            if wants_shorts and not PARABOLIC_DISABLED:
                logger.info("🔥 PARABOLIC SCANNER (50%+ exhausted pumps)")
                parabolic_signal = await service.generate_parabolic_dump_signal(min_change_percent=1.0, max_symbols=10)
                
                if parabolic_signal and parabolic_signal['direction'] == 'SHORT':
                    logger.info(f"✅ PARABOLIC signal found: {parabolic_signal['symbol']} @ +{parabolic_signal.get('24h_change')}%")
                    return
            elif wants_shorts and PARABOLIC_DISABLED:
                logger.info("🔥 PARABOLIC DISABLED - skipping 50%+ dump scan")
            
            if wants_shorts and not parabolic_signal and NORMAL_SHORTS_ENABLED:
                logger.info("📉 ═══════════════════════════════════════════════════════")
                logger.info("📉 NORMAL SHORTS SCANNER - Weakness detection")
                logger.info("📉 ═══════════════════════════════════════════════════════")
                
                # Check dump mode - lower requirements during dumps
                dump_state = await check_dump_mode()
                is_dump = dump_state.get('is_dump', False)
                
                # TIGHTENED: Higher threshold for shorts (5%+ normally, 3%+ in dump)
                min_change = 3.0 if is_dump else 5.0
                
                short_candidates = await service.get_top_gainers(limit=20, min_change_percent=min_change)
                logger.info(f"📉 Raw candidates fetched: {len(short_candidates)} (min {min_change}%)")
                
                # Log top 5 candidates for debugging
                for i, c in enumerate(short_candidates[:5]):
                    logger.info(f"   #{i+1}: {c['symbol']} +{c.get('change_percent', 0):.1f}% vol ${c.get('volume_24h', 0)/1e6:.1f}M")
                
                short_candidates = [g for g in short_candidates if min_change <= g.get('change_percent', 0) <= 50.0]
                
                if short_candidates:
                    logger.info(f"📉 Filtered to {len(short_candidates)} candidates ({min_change}-50% range)")
                    
                    ai_attempts = 0
                    for candidate in short_candidates[:5]:
                        symbol = candidate['symbol']
                        current_price = candidate['price']
                        
                        if is_symbol_on_cooldown(symbol):
                            continue
                        
                        if ai_attempts > 0:
                            await asyncio.sleep(10.0)
                        ai_attempts += 1
                        
                        analysis = await service.analyze_normal_short(symbol, candidate, current_price)
                        
                        if analysis and analysis['direction'] == 'SHORT':
                            tp_pct = analysis.get('tp_percent', 5.0)
                            sl_pct = analysis.get('sl_percent', 3.5)
                            tp_price = current_price * (1 - tp_pct / 100)
                            sl_price = current_price * (1 + sl_pct / 100)
                            
                            normal_short_signal = {
                                'symbol': symbol,
                                'direction': 'SHORT',
                                'confidence': analysis['confidence'],
                                'entry_price': current_price,
                                'stop_loss': sl_price,
                                'take_profit': tp_price,
                                'take_profit_1': tp_price,
                                'take_profit_2': None,
                                'take_profit_3': None,
                                '24h_change': candidate['change_percent'],
                                '24h_volume': candidate['volume_24h'],
                                'trade_type': 'TOP_GAINER',
                                'strategy': 'NORMAL_SHORT',
                                'leverage': 20,
                                'reasoning': f"🤖 AI SHORT [{analysis.get('ai_quality', 'A')}]: {analysis.get('ai_reasoning', analysis['reason'])}"
                            }
                            logger.info(f"✅ AI APPROVED SHORT: {symbol} @ +{candidate['change_percent']:.1f}% | TP {tp_pct}% / SL {sl_pct}%")
                            break
                else:
                    logger.info("📉 No candidates found in 3-50% range")
        
        # 🎯 EXECUTE IN REGIME-BASED ORDER
        if shorts_first:
            await scan_shorts()
            if not parabolic_signal and not normal_short_signal:
                await scan_longs()
        else:
            await scan_longs()
            if not long_signal:
                await scan_shorts()
        
        # If no signals at all, exit
        if not parabolic_signal and not normal_short_signal and not long_signal:
            mode_str = []
            if wants_shorts:
                mode_str.append("SHORTS (PARABOLIC/NORMAL)")
            if wants_longs:
                mode_str.append("LONGS")
            logger.info(f"No signals found for {' and '.join(mode_str) if mode_str else 'any mode'}")
            await service.close()
            return
        
        # 🔥 PERFECT TRADES FILTER: If limit reached, only allow 10/10 A+ signals
        def is_perfect_trade(signal):
            """Check if signal has 100% AI confidence (10/10 + A+ grade)"""
            if not signal:
                return False
            ai_confidence = signal.get('ai_confidence', signal.get('confidence', 0))
            ai_quality = signal.get('ai_quality', signal.get('entry_quality', ''))
            # Allow 10/10 confidence OR 100 confidence (different scales)
            is_max_confidence = ai_confidence >= 10 or ai_confidence >= 100
            is_top_grade = ai_quality in ['A+', 'A']  # Also allow 'A' since A+ is rare
            return is_max_confidence and is_top_grade
        
        # Process AI-POWERED LONG signal first (PRIORITY #1 - Best performer!)
        if long_signal:
            if perfect_trades_only:
                if is_perfect_trade(long_signal):
                    logger.info(f"🌟 PERFECT TRADE: {long_signal['symbol']} passes limit override (Confidence: {long_signal.get('ai_confidence', long_signal.get('confidence'))})")
                    await process_and_broadcast_signal(long_signal, users_with_mode, db_session, bot, service)
                else:
                    logger.info(f"⏳ LIMIT MODE: Skipping {long_signal['symbol']} - not 100% confidence (Conf: {long_signal.get('ai_confidence', long_signal.get('confidence'))}, Grade: {long_signal.get('ai_quality', 'N/A')})")
            else:
                await process_and_broadcast_signal(long_signal, users_with_mode, db_session, bot, service)
        
        # Process PARABOLIC signal (Priority #2 - 50%+ exhausted dumps)
        if parabolic_signal:
            if perfect_trades_only:
                if is_perfect_trade(parabolic_signal):
                    logger.info(f"🌟 PERFECT TRADE: {parabolic_signal['symbol']} passes limit override")
                    await process_and_broadcast_signal(parabolic_signal, users_with_mode, db_session, bot, service)
                    last_short_signal_time = datetime.utcnow()
                else:
                    logger.info(f"⏳ LIMIT MODE: Skipping {parabolic_signal['symbol']} - not 100% confidence")
            else:
                await process_and_broadcast_signal(parabolic_signal, users_with_mode, db_session, bot, service)
                last_short_signal_time = datetime.utcnow()
        
        # Process NORMAL SHORT signal (Priority #3 - AI overbought reversals)
        elif normal_short_signal:
            if perfect_trades_only:
                if is_perfect_trade(normal_short_signal):
                    logger.info(f"🌟 PERFECT TRADE: {normal_short_signal['symbol']} passes limit override")
                    await process_and_broadcast_signal(normal_short_signal, users_with_mode, db_session, bot, service)
                    last_short_signal_time = datetime.utcnow()
                else:
                    logger.info(f"⏳ LIMIT MODE: Skipping {normal_short_signal['symbol']} - not 100% confidence")
            else:
                await process_and_broadcast_signal(normal_short_signal, users_with_mode, db_session, bot, service)
                last_short_signal_time = datetime.utcnow()
        
        # 📊 SCAN SUMMARY - Log what happened this cycle
        signals_found = []
        if long_signal:
            signals_found.append(f"LONG: {long_signal['symbol']}")
        if parabolic_signal:
            signals_found.append(f"PARABOLIC: {parabolic_signal['symbol']}")
        if normal_short_signal:
            signals_found.append(f"SHORT: {normal_short_signal['symbol']}")
        
        if signals_found:
            logger.info(f"✅ SCAN COMPLETE - Found: {', '.join(signals_found)}")
        else:
            logger.info(f"⚪ SCAN COMPLETE - No signals found this cycle (LONGS={wants_longs}, SHORTS={wants_shorts}, Regime={regime_focus})")
        
        await service.close()
    
    except Exception as e:
        logger.error(f"Error in broadcast_top_gainer_signal: {e}", exc_info=True)


async def process_and_broadcast_signal(signal_data, users_with_mode, db_session, bot, service):
    """Helper function to process and broadcast a single signal with parallel execution
    
    🔒 CRITICAL: Uses PostgreSQL Advisory Locks to prevent duplicate signals
    - Normalizes symbol format before hashing (XAN/USDT → XAN)
    - Acquires advisory lock on {symbol}:{direction} key
    - Checks for duplicates within 5-minute window
    - Creates and broadcasts signal if no duplicate exists
    - Always releases lock in finally block
    """
    from app.models import Signal, Trade
    from app.services.bitunix_trader import execute_bitunix_trade
    from datetime import datetime, timedelta
    import logging
    import asyncio
    import random
    import hashlib
    from sqlalchemy import text
    
    logger = logging.getLogger(__name__)
    
    # 🔒 CRITICAL: PostgreSQL Advisory Lock for Duplicate Prevention
    lock_acquired = False
    lock_id = None
    signal = None
    
    try:
        # Normalize symbol (XAN/USDT → XAN, BTCUSDT → BTC) before hashing
        # This ensures "XAN/USDT" and "XANUSDT" map to the same lock
        normalized_symbol = signal_data['symbol'].replace('/USDT', '').replace('USDT', '')
        lock_key = f"{normalized_symbol}:{signal_data['direction']}"
        lock_id = int(hashlib.md5(lock_key.encode()).hexdigest()[:16], 16) % (2**63 - 1)
        
        # Acquire PostgreSQL advisory lock with timeout (NON-BLOCKING with retry)
        # Use pg_try_advisory_lock to avoid deadlocks from stuck locks
        result = db_session.execute(text(f"SELECT pg_try_advisory_lock({lock_id})"))
        lock_acquired = result.scalar()
        
        if not lock_acquired:
            logger.warning(f"⚠️ Could not acquire lock for {lock_key} - another process is holding it")
            # Try to force-release if lock seems stuck (older than 2 min)
            db_session.execute(text(f"SELECT pg_advisory_unlock({lock_id})"))
            # Retry once
            result = db_session.execute(text(f"SELECT pg_try_advisory_lock({lock_id})"))
            lock_acquired = result.scalar()
            if not lock_acquired:
                logger.error(f"❌ Lock still held after force-release attempt: {lock_key}")
                return
        
        logger.info(f"🔒 Advisory lock acquired: {lock_key} (ID: {lock_id})")
        
        # 🔥 CHECK 1: Recent signal duplicate (within 24 HOURS for SAME direction)
        # If LONG was called, same coin can be called for SHORT next round (and vice versa)
        recent_cutoff = datetime.utcnow() - timedelta(hours=24)
        existing_signal = db_session.query(Signal).filter(
            Signal.symbol == signal_data['symbol'],
            Signal.direction == signal_data['direction'],  # Only block SAME direction
            Signal.created_at >= recent_cutoff
        ).first()
        
        if existing_signal:
            hours_ago = (datetime.utcnow() - existing_signal.created_at).total_seconds() / 3600
            logger.warning(f"🚫 DUPLICATE PREVENTED (24h cooldown): {signal_data['symbol']} {signal_data['direction']} (Signal #{existing_signal.id}, {hours_ago:.1f}h ago)")
            return
        
        # 🔥 CHECK 2: ANY open positions in this symbol (across ALL users!)
        open_positions = db_session.query(Trade).filter(
            Trade.symbol == signal_data['symbol'],
            Trade.status == 'open'
        ).count()
        
        if open_positions > 0:
            logger.warning(f"🚫 DUPLICATE PREVENTED (open positions): {signal_data['symbol']} has {open_positions} open position(s) - SKIPPING!")
            return
        
        # 🔥 CHECK 3: GLOBAL + MODULE DAILY LIMIT
        from app.services.social_signals import check_global_signal_limit, increment_global_signal_count, check_signal_gap, record_signal_broadcast
        if not check_global_signal_limit():
            logger.warning(f"⚠️ GLOBAL DAILY LIMIT REACHED - Cannot broadcast {signal_data['symbol']} {signal_data['direction']}")
            return
        if not check_signal_gap():
            logger.warning(f"⚠️ SIGNAL GAP NOT MET - Too soon since last signal, skipping {signal_data['symbol']}")
            return
        if not check_and_increment_daily_signals(direction=signal_data['direction']):
            logger.warning(f"⚠️ DAILY LIMIT REACHED - Cannot broadcast {signal_data['symbol']} {signal_data['direction']}")
            return
        
        # 🤖 CHECK 4: AI SIGNAL FILTER - Get AI approval before broadcasting
        from app.services.ai_signal_filter import should_broadcast_signal
        ai_approved, ai_analysis_text = await should_broadcast_signal(signal_data)
        
        if not ai_approved:
            logger.warning(f"🤖 AI REJECTED: {signal_data['symbol']} {signal_data['direction']} - Signal quality insufficient")
            # Decrement daily count since we're not broadcasting
            decrement_daily_signals(direction=signal_data['direction'])
            return
        
        logger.info(f"🤖 AI APPROVED: {signal_data['symbol']} {signal_data['direction']}")
        
        # Create signal (protected by advisory lock - NO race condition!)
        signal_type = signal_data.get('trade_type', 'TOP_GAINER')
        signal = Signal(
            symbol=signal_data['symbol'],
            direction=signal_data['direction'],
            entry_price=signal_data['entry_price'],
            stop_loss=signal_data['stop_loss'],
            take_profit=signal_data.get('take_profit'),
            take_profit_1=signal_data.get('take_profit_1'),
            take_profit_2=signal_data.get('take_profit_2'),
            take_profit_3=signal_data.get('take_profit_3'),
            confidence=signal_data['confidence'],
            reasoning=signal_data['reasoning'],
            signal_type=signal_type,
            timeframe='5m',
            created_at=datetime.utcnow()
        )
        db_session.add(signal)
        db_session.flush()
        db_session.commit()
        db_session.refresh(signal)
        
        logger.info(f"✅ SIGNAL CREATED: {signal.symbol} {signal.direction} @ ${signal.entry_price} (24h: {signal_data.get('24h_change')}%)")
        
        # 🔥 ADD COOLDOWN - Prevent same coin/direction being signaled again for 24 hours
        add_signal_cooldown(signal.symbol, cooldown_minutes=1440)  # 24 hours
        increment_global_signal_count()
        record_signal_broadcast()
        
        # 📣 BROADCAST & EXECUTE SIGNAL (lock is still held throughout)
        # Check if parabolic reversal (aggressive 20x leverage)
        is_parabolic = signal_data.get('is_parabolic_reversal', False)
        
        # Build TP text - Calculate actual profit % from price levels
        leverage = 20
        entry = signal.entry_price
        tp1 = signal.take_profit_1 or signal.take_profit
        sl = signal.stop_loss
        
        if signal.direction == 'LONG':
            tp_price_pct = ((tp1 - entry) / entry) * 100 if entry > 0 else 0
            sl_price_pct = ((entry - sl) / entry) * 100 if entry > 0 else 0
            tp_profit_pct = tp_price_pct * leverage
            sl_loss_pct = sl_price_pct * leverage
        else:  # SHORT
            tp_price_pct = ((entry - tp1) / entry) * 100 if entry > 0 else 0
            sl_price_pct = ((sl - entry) / entry) * 100 if entry > 0 else 0
            tp_profit_pct = tp_price_pct * leverage
            sl_loss_pct = sl_price_pct * leverage
        
        rr_ratio = tp_profit_pct / sl_loss_pct if sl_loss_pct > 0 else 1.0
        
        if is_parabolic and signal.direction == 'SHORT':
            tp_text = f"<b>TP:</b> ${tp1:.6f} (+{tp_profit_pct:.0f}% @ {leverage}x) 🚀💥"
            sl_text = f"(-{sl_loss_pct:.0f}% @ {leverage}x)"
            rr_text = f"{rr_ratio:.1f}:1 risk-to-reward (PARABOLIC REVERSAL)"
        elif signal.direction == 'LONG':
            tp_text = f"<b>TP:</b> ${tp1:.6f} (+{tp_profit_pct:.0f}% @ {leverage}x) 🎯"
            sl_text = f"(-{sl_loss_pct:.0f}% @ {leverage}x)"
            rr_text = f"{rr_ratio:.1f}:1 risk-to-reward"
        elif signal.direction == 'SHORT':
            tp_text = f"<b>TP:</b> ${tp1:.6f} (+{tp_profit_pct:.0f}% @ {leverage}x) 🎯"
            sl_text = f"(-{sl_loss_pct:.0f}% @ {leverage}x)"
            rr_text = f"{rr_ratio:.1f}:1 risk-to-reward"
        else:
            tp_text = f"<b>TP:</b> ${signal.take_profit:.6f} (+{tp_profit_pct:.0f}% @ {leverage}x)"
            sl_text = f"(-{sl_loss_pct:.0f}% @ {leverage}x)"
            rr_text = f"{rr_ratio:.1f}:1 risk-to-reward"
        
        # Broadcast to users
        direction_emoji = "🟢 LONG" if signal.direction == 'LONG' else "🔴 SHORT"
        
        # Add tier badge for LONGS
        tier_badge = ""
        if signal.direction == 'LONG' and signal_data.get('tier'):
            tier_label = signal_data.get('tier_label', '')
            tier = signal_data.get('tier', '')
            tier_change = signal_data.get('tier_change', 0)
            tier_badge = f"\n🎯 <b>{tier_label}</b> detection ({tier} pump: +{tier_change}%)\n"
        
        # Volume display - different for LONGS (early breakout) vs SHORTS (24h volume)
        if signal.direction == 'LONG' and signal_data.get('volume_ratio'):
            # LONGS: Show spike ratio + 24h liquidity baseline
            vol_ratio = signal_data.get('volume_ratio', 0)
            current_1m = signal_data.get('current_1m_vol', 0)
            vol_24h = signal_data.get('24h_volume', 0)
            volume_display = f"""├ 1m Spike: <b>{vol_ratio}x</b> (${current_1m:,.0f} vs avg)
└ 24h Liquidity: ${vol_24h:,.0f}"""
        else:
            # SHORTS: Standard 24h volume
            volume_display = f"└ Volume: ${signal_data.get('24h_volume', 0):,.0f}"
        
        signal_text = f"""
┏━━━━━━━━━━━━━━━━━━━━━━━┓
  🔥 <b>TOP GAINER ALERT</b> 🔥
┗━━━━━━━━━━━━━━━━━━━━━━━┛
{tier_badge}
{direction_emoji} <b>${signal.symbol.replace('/USDT', '').replace('USDT', '')}</b>
━━━━━━━━━━━━━━━━━━━━━━━━━━

<b>📊 Market Data</b>
├ 24h Change: <b>+{signal_data.get('24h_change')}%</b>
{volume_display}

<b>🎯 Trade Setup</b>
├ Entry: <b>${signal.entry_price:.6f}</b>
├ {tp_text.replace(chr(10), chr(10) + '├ ')}
└ SL: ${signal.stop_loss:.6f} {sl_text}

<b>⚡ Risk Management</b>
├ Leverage: <b>20x</b>
└ Risk/Reward: <b>{rr_text}</b>

<b>💡 Analysis</b>
{signal.reasoning}
{ai_analysis_text}
⚠️ <b>HIGH VOLATILITY MODE</b>
<i>Auto-executing for enabled users...</i>
"""
        
        # 🚀 PARALLEL EXECUTION with controlled concurrency
        # Use Semaphore to limit concurrent trades (prevents API rate limit issues)
        # Increased from 3 to 10 for faster execution across all users
        semaphore = asyncio.Semaphore(10)  # Max 10 concurrent trades
        
        async def execute_user_trade(user, user_idx):
            """Execute trade for a single user with controlled concurrency"""
            from app.database import SessionLocal
            from app.models import User, UserPreference, Trade, TradeAttempt
            
            # Each task gets its own DB session (sessions are not thread-safe)
            user_db = SessionLocal()
            start_time = asyncio.get_event_loop().time()
            executed = False
            
            def log_attempt(status: str, reason: str, balance: float = None, pos_size: float = None):
                """Helper to log trade attempt to database"""
                try:
                    attempt = TradeAttempt(
                        signal_id=signal.id if signal else None,
                        user_id=user.id,
                        symbol=signal.symbol,
                        direction=signal.direction,
                        status=status,
                        reason=reason,
                        balance_at_attempt=balance,
                        position_size=pos_size
                    )
                    user_db.add(attempt)
                    user_db.commit()
                except Exception as e:
                    logger.error(f"Failed to log trade attempt: {e}")
            
            try:
                async with semaphore:
                    # Minimal jitter (50-100ms) - fast execution is priority
                    jitter = random.uniform(0.05, 0.1)
                    await asyncio.sleep(jitter)
                    
                    logger.info(f"⚡ Starting trade execution for user {user.id} ({user_idx+1}/{len(users_with_mode)})")
                    
                    # 🔐 SUBSCRIPTION CHECK: Skip expired/unsubscribed users
                    # Re-fetch user to get fresh subscription status
                    fresh_user = user_db.query(User).filter_by(id=user.id).first()
                    if not fresh_user:
                        log_attempt('skipped', 'User not found')
                        return executed
                    
                    if not fresh_user.is_subscribed and not fresh_user.is_admin:
                        logger.info(f"⏭️ User {user.id} - Subscription expired, skipping signal")
                        log_attempt('skipped', 'Subscription expired')
                        return executed
                    
                    prefs = user_db.query(UserPreference).filter_by(user_id=user.id).first()
                    
                    # ALL users get ALL signals - no per-user trade mode filtering
                    # Everyone receives the same trades as the owner
                    
                    # Check if user has auto-trading enabled
                    has_auto_trading = prefs and prefs.auto_trading_enabled
                    
                    # Send manual signal notification for users without auto-trading
                    if not has_auto_trading:
                        try:
                            direction_emoji = "🟢 LONG" if signal.direction == 'LONG' else "🔴 SHORT"
                    
                            # Add tier badge for LONGS
                            tier_badge_manual = ""
                            if signal.direction == 'LONG' and signal_data.get('tier'):
                                tier_label = signal_data.get('tier_label', '')
                                tier = signal_data.get('tier', '')
                                tier_change = signal_data.get('tier_change', 0)
                                tier_badge_manual = f"\n🎯 <b>{tier_label}</b> detection ({tier} pump: +{tier_change}%)\n"
                            
                            # Calculate TP text - Direction-specific
                            if signal.direction == 'LONG':
                                # LONGS: Single TP at 67% with 65% SL @ 20x
                                tp_manual = f"<b>TP:</b> ${signal.take_profit_1:.6f} (+67% @ 20x) 🎯"
                                sl_manual = "(-65% @ 20x)"
                            elif signal.direction == 'SHORT':
                                # SHORTS: Single TP at 8% (CAPPED at 80% max for display)
                                tp_manual = f"<b>TP:</b> ${signal.take_profit_1:.6f} (up to +150% max) 🎯"
                                sl_manual = "(up to -80% max)"  # SHORTS: SL capped at 80%
                            else:
                                # Fallback
                                profit_pct_manual = 25 if signal.direction == 'LONG' else 40
                                tp_manual = f"<b>TP:</b> ${signal.take_profit:.6f} (+{profit_pct_manual}% @ 5x)"
                                sl_manual = f"(-{profit_pct_manual}% @ 5x)"
                            
                            manual_signal_msg = f"""
┏━━━━━━━━━━━━━━━━━━━━━━━┓
  🔥 <b>TOP GAINER SIGNAL</b> 🔥
┗━━━━━━━━━━━━━━━━━━━━━━━┛
{tier_badge_manual}
{direction_emoji} <b>${signal.symbol.replace('/USDT', '').replace('USDT', '')}</b>
━━━━━━━━━━━━━━━━━━━━━━━━━━

<b>📊 Market Data</b>
├ 24h Change: <b>+{signal_data.get('24h_change')}%</b>
└ Volume: ${signal_data.get('24h_volume'):,.0f}

<b>🎯 Trade Setup</b>
├ Entry: <b>${signal.entry_price:.6f}</b>
├ {tp_manual.replace(chr(10), chr(10) + '├ ')}
└ SL: ${signal.stop_loss:.6f} {sl_manual}

<b>⚡ Recommended Settings</b>
├ Leverage: <b>20x</b>
└ Risk/Reward: <b>{'1:1' if signal.direction == 'SHORT' else '1:1'}</b>

<b>💡 Analysis</b>
{signal.reasoning}

⚠️ <b>MANUAL SIGNAL</b>
<i>Enable auto-trading to execute automatically!</i>
"""
                            
                            await bot.send_message(
                                user.telegram_id,
                                manual_signal_msg,
                                parse_mode='HTML'
                            )
                            logger.info(f"✅ Sent manual signal to user {user.id}")
                            
                        except Exception as e:
                            logger.error(f"Error sending manual signal to user {user.id}: {e}")
                        
                        log_attempt('skipped', 'Manual trader - auto-trading disabled')
                        return executed  # Skip auto-execution for manual traders
            
                    # 🔥 CRITICAL FIX: Check if user already has position in this SPECIFIC symbol
                    existing_symbol_position = user_db.query(Trade).filter(
                        Trade.user_id == user.id,
                        Trade.status == 'open',
                        Trade.symbol == signal.symbol  # Same symbol check
                    ).first()
                    
                    if existing_symbol_position:
                        logger.info(f"⚠️ DUPLICATE PREVENTED: User {user.id} already has open position in {signal.symbol} (Trade ID: {existing_symbol_position.id})")
                        log_attempt('skipped', f'Already has open position in {signal.symbol}')
                        return executed
                    
                    # Check if user has space for more top gainer positions
                    current_top_gainer_positions = user_db.query(Trade).filter(
                        Trade.user_id == user.id,
                        Trade.status == 'open',
                        Trade.trade_type == 'TOP_GAINER'
                    ).count()
                    
                    max_allowed = prefs.top_gainers_max_symbols if prefs else 3
                    
                    if current_top_gainer_positions >= max_allowed:
                        logger.info(f"⏭️ User {user.id} ({user.username}) - MAX POSITIONS: {current_top_gainer_positions}/{max_allowed}")
                        log_attempt('skipped', f'Max positions reached: {current_top_gainer_positions}/{max_allowed}')
                        return executed
            
                    # Execute trade with user's custom leverage for top gainers
                    user_leverage = prefs.top_gainers_leverage if prefs and prefs.top_gainers_leverage else 5
                    logger.info(f"🚀 EXECUTING: User {user.id} ({user.username or user.first_name}) - {signal.symbol} {signal.direction} @ {user_leverage}x")
                    
                    trade = await execute_bitunix_trade(
                        signal=signal,
                        user=user,
                        db=user_db,
                        trade_type='TOP_GAINER',
                        leverage_override=user_leverage  # Use user's custom top gainer leverage
                    )
                    
                    if trade:
                        executed = True
                        logger.info(f"✅ SUCCESS: User {user.id} ({user.username or user.first_name}) - {signal.symbol} {signal.direction} EXECUTED")
                        log_attempt('success', 'Trade executed successfully', pos_size=trade.position_size)
                
                        # Send personalized notification with user's actual leverage
                        try:
                            user_leverage = prefs.top_gainers_leverage if prefs and prefs.top_gainers_leverage else 5
                            
                            # Calculate profit percentages from ACTUAL signal levels
                            entry = signal.entry_price
                            tp1 = signal.take_profit_1 or signal.take_profit
                            sl = signal.stop_loss
                            display_leverage = user_leverage
                            
                            if signal.direction == 'LONG':
                                # Use actual signal TP/SL
                                tp_price_pct = ((tp1 - entry) / entry) * 100 if entry > 0 else 3.35
                                sl_price_pct = ((entry - sl) / entry) * 100 if entry > 0 else 3.25
                                tp1_profit_pct = tp_price_pct * user_leverage
                                sl_loss_pct = sl_price_pct * user_leverage
                                targets = {'tp_prices': [tp1], 'sl_price': sl}
                            else:  # SHORT
                                # Use actual signal TP/SL
                                tp_price_pct = ((entry - tp1) / entry) * 100 if entry > 0 else 4.0
                                sl_price_pct = ((sl - entry) / entry) * 100 if entry > 0 else 4.0
                                tp1_profit_pct = tp_price_pct * user_leverage
                                sl_loss_pct = sl_price_pct * user_leverage
                                targets = {'tp_prices': [tp1], 'sl_price': sl}
                    
                            # Rebuild TP/SL text with leverage-capped percentages
                            # Calculate R:R based on capped values
                            if signal.direction == 'LONG':
                                # LONGS: Single TP at 67% with 65% SL @ 20x
                                user_tp_text = f"<b>TP:</b> ${targets['tp_prices'][0]:.6f} (+{tp1_profit_pct:.0f}% @ {display_leverage}x) 🎯"
                                user_sl_text = f"${targets['sl_price']:.6f} (-{sl_loss_pct:.0f}% @ {display_leverage}x)"
                                rr_ratio = tp1_profit_pct / sl_loss_pct if sl_loss_pct > 0 else 1.03
                                rr_text = f"{rr_ratio:.2f}:1 risk-to-reward"
                            elif signal.direction == 'SHORT':
                                # SHORTS: Show capped profit percentage
                                user_tp_text = f"<b>TP:</b> ${targets['tp_prices'][0]:.6f} (+{tp1_profit_pct:.0f}% @ {display_leverage}x) 🎯"
                                user_sl_text = f"${targets['sl_price']:.6f} (-{sl_loss_pct:.0f}% @ {display_leverage}x)"
                                # Calculate actual R:R
                                rr_ratio = tp1_profit_pct / sl_loss_pct if sl_loss_pct > 0 else 0
                                rr_text = f"1:{rr_ratio:.1f} risk-to-reward"
                            else:
                                # Fallback
                                user_tp_text = f"<b>TP:</b> ${signal.take_profit:.6f} (+{tp1_profit_pct:.0f}% @ {display_leverage}x)"
                                user_sl_text = f"${signal.stop_loss:.6f} (-{sl_loss_pct:.0f}% @ {display_leverage}x)"
                                rr_ratio = tp1_profit_pct / sl_loss_pct if sl_loss_pct > 0 else 0
                                rr_text = f"1:{rr_ratio:.1f} risk-to-reward"
                            
                            # Personalized signal message
                            direction_emoji = "🟢 LONG" if signal.direction == 'LONG' else "🔴 SHORT"
                            
                            # Add tier badge for LONGS (same as broadcast)
                            tier_badge_personalized = ""
                            if signal.direction == 'LONG' and signal_data.get('tier'):
                                tier_label = signal_data.get('tier_label', '')
                                tier = signal_data.get('tier', '')
                                tier_change = signal_data.get('tier_change', 0)
                                tier_badge_personalized = f"\n🎯 <b>{tier_label}</b> detection ({tier} pump: +{tier_change}%)\n"
                            
                            personalized_signal = f"""
┏━━━━━━━━━━━━━━━━━━━━━━━┓
  🔥 <b>TOP GAINER ALERT</b> 🔥
┗━━━━━━━━━━━━━━━━━━━━━━━┛
{tier_badge_personalized}
{direction_emoji} <b>${signal.symbol.replace('/USDT', '').replace('USDT', '')}</b>
━━━━━━━━━━━━━━━━━━━━━━━━━━

<b>📊 Market Data</b>
├ 24h Change: <b>+{signal_data.get('24h_change')}%</b>
└ Volume: ${signal_data.get('24h_volume'):,.0f}

<b>🎯 Your Trade</b>
├ Entry: <b>${signal.entry_price:.6f}</b>
├ {user_tp_text.replace(chr(10), chr(10) + '├ ')}
└ SL: {user_sl_text}

<b>⚡ Your Settings</b>
├ Leverage: <b>{user_leverage}x</b>
└ Risk/Reward: <b>{rr_text}</b>

<b>💡 Analysis</b>
{signal.reasoning}

⚠️ <b>HIGH VOLATILITY MODE</b>
"""
                            
                            await bot.send_message(
                                user.telegram_id,
                                f"{personalized_signal}\n✅ <b>Trade Executed!</b>\n"
                                f"Position Size: ${trade.position_size:.2f}",
                                parse_mode="HTML"
                            )
                        except Exception as e:
                            logger.error(f"Failed to send notification to user {user.id}: {e}")
                    else:
                        # Trade execution failed - log clearly for debugging
                        logger.error(f"❌ FAILED: User {user.id} ({user.username or user.first_name}) - {signal.symbol} {signal.direction} - execute_bitunix_trade returned None")
                        log_attempt('failed', 'execute_bitunix_trade returned None - check balance/API keys')
                
                return executed
            except Exception as e:
                logger.exception(f"Error executing trade for user {user.id}: {e}")
                log_attempt('error', f'Exception: {str(e)[:200]}')
                return False
            finally:
                user_db.close()
        
        # Execute trades in parallel with controlled concurrency
        tasks = [execute_user_trade(user, idx) for idx, user in enumerate(users_with_mode)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 🔥 IMPROVED LOGGING: Track and report execution results
        executed_count = 0
        exception_count = 0
        skipped_count = 0
        
        for idx, (result, user) in enumerate(zip(results, users_with_mode)):
            if result is True:
                executed_count += 1
            elif isinstance(result, Exception):
                exception_count += 1
                logger.error(f"❌ EXCEPTION for user {user.id} ({user.username}): {result}")
            else:
                skipped_count += 1
                # Only log first few skips to avoid spam
                if skipped_count <= 5:
                    logger.info(f"⏭️ Skipped user {user.id} ({user.username}): result={result}")
        
        logger.info(f"📊 EXECUTION SUMMARY: {executed_count} executed, {skipped_count} skipped, {exception_count} errors out of {len(users_with_mode)} users")
        
    except Exception as e:
        db_session.rollback()
        logger.error(f"❌ Error in process_and_broadcast_signal: {e}", exc_info=True)
    
    finally:
        # 🔓 CRITICAL: Always release advisory lock (even on error/early return!)
        # This ensures the lock is ALWAYS released, preventing deadlocks
        if lock_acquired and lock_id is not None:
            try:
                # Normalize symbol again for logging (in case early return happened)
                norm_sym = signal_data['symbol'].replace('/USDT', '').replace('USDT', '')
                db_session.execute(text(f"SELECT pg_advisory_unlock({lock_id})"))
                logger.info(f"🔓 Lock released: {norm_sym}:{signal_data['direction']}")
            except Exception as unlock_error:
                logger.error(f"⚠️ Failed to release lock: {unlock_error}")
