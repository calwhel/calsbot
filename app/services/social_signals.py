"""
Social & News Signals Trading Mode - AI-powered trading
Completely separate from Top Gainers mode
"""
import asyncio
import json
import logging
import os
import httpx
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from sqlalchemy.orm import Session

from app.services.lunarcrush import (
    get_coin_metrics, 
    get_trending_coins, 
    get_social_spikes,
    interpret_signal_score,
    get_lunarcrush_api_key
)

logger = logging.getLogger(__name__)


async def ai_analyze_social_signal(signal_data: Dict) -> Dict:
    """
    Use Gemini for fast scan + Claude for final approval of social signals.
    Returns dict with 'approved', 'reasoning', 'ai_confidence', 'recommendation'.
    """
    symbol = signal_data['symbol']
    direction = signal_data.get('direction', 'LONG')
    
    data_summary = (
        f"Coin: {symbol}\n"
        f"Direction: {direction}\n"
        f"Entry: ${signal_data['entry_price']}\n"
        f"Galaxy Score: {signal_data['galaxy_score']}/16\n"
        f"Sentiment: {signal_data['sentiment']*100:.0f}%\n"
        f"RSI (15m): {signal_data['rsi']:.0f}\n"
        f"24h Change: {signal_data['24h_change']:+.1f}%\n"
        f"24h Volume: ${signal_data['24h_volume']/1e6:.1f}M\n"
        f"Social Volume: {signal_data['social_volume']:,}\n"
        f"TP: +{signal_data['tp_percent']:.1f}%\n"
        f"SL: -{signal_data['sl_percent']:.1f}%"
    )
    
    # STEP 1: Gemini fast scan
    gemini_reasoning = None
    try:
        from app.services.ai_market_intelligence import get_gemini_client
        gemini = get_gemini_client()
        if gemini:
            prompt = f"""You are a crypto perps trader analyzing a social sentiment signal. Give a brief, sharp trading analysis.

{data_summary}

Analyze this {direction} signal. Consider:
1. Is the RSI supporting the direction? (oversold for longs, overbought for shorts)
2. Does the social sentiment and Galaxy Score justify the trade?
3. What's the risk/reward ratio look like?
4. Any concerns about the 24h price action?

Respond in JSON:
{{
    "scan_pass": true/false,
    "reasoning": "2-3 sentence sharp analysis for traders. Be specific about why this is a good or bad entry.",
    "confidence": 1-10,
    "key_risk": "one sentence main risk"
}}"""
            
            def _gemini_call():
                return gemini.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=prompt,
                    config={"temperature": 0.3, "max_output_tokens": 300}
                )
            
            response = await asyncio.get_event_loop().run_in_executor(None, _gemini_call)
            result_text = response.text.strip()
            
            if "```json" in result_text:
                import re
                match = re.search(r'```json\s*(.*?)\s*```', result_text, re.DOTALL)
                if match:
                    result_text = match.group(1)
            
            first_brace = result_text.find("{")
            last_brace = result_text.rfind("}")
            if first_brace >= 0 and last_brace > first_brace:
                result_text = result_text[first_brace:last_brace + 1]
            
            gemini_result = json.loads(result_text)
            gemini_reasoning = gemini_result.get('reasoning', '')
            
            if not gemini_result.get('scan_pass', True):
                logger.info(f"🤖 Gemini REJECTED {symbol}: {gemini_reasoning}")
                return {
                    'approved': False,
                    'reasoning': gemini_reasoning,
                    'ai_confidence': gemini_result.get('confidence', 3),
                    'recommendation': 'AVOID',
                    'key_risk': gemini_result.get('key_risk', '')
                }
            
            logger.info(f"🤖 Gemini PASSED {symbol}: confidence {gemini_result.get('confidence', 5)}")
    except Exception as e:
        logger.warning(f"Gemini analysis failed for {symbol}: {e}")
    
    # STEP 2: Claude final approval
    claude_reasoning = None
    try:
        import anthropic
        api_key = os.environ.get("AI_INTEGRATIONS_ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_API_KEY")
        if api_key:
            client = anthropic.Anthropic(api_key=api_key)
            
            gemini_context = f"\nGemini Initial Scan: {gemini_reasoning}" if gemini_reasoning else ""
            
            claude_prompt = f"""You are an expert crypto perpetual futures trader. Analyze this social sentiment signal and give your final verdict.

{data_summary}
{gemini_context}

As a final quality gate, determine:
1. Should this trade be executed? Consider the full picture.
2. Is the entry timing right based on RSI and 24h change?
3. Are the TP/SL levels reasonable for the setup?

Respond in JSON:
{{
    "approved": true/false,
    "confidence": 1-10,
    "recommendation": "STRONG BUY" or "BUY" or "HOLD" or "AVOID",
    "reasoning": "2-3 sentence concise analysis. Be direct and actionable. Mention specific numbers.",
    "entry_quality": "EXCELLENT" or "GOOD" or "FAIR" or "POOR"
}}"""
            
            def _claude_call():
                return client.messages.create(
                    model="claude-sonnet-4-20250514",
                    max_tokens=300,
                    messages=[{"role": "user", "content": claude_prompt}]
                )
            
            response = await asyncio.get_event_loop().run_in_executor(None, _claude_call)
            result_text = response.content[0].text.strip()
            
            if "```json" in result_text:
                import re
                match = re.search(r'```json\s*(.*?)\s*```', result_text, re.DOTALL)
                if match:
                    result_text = match.group(1)
            
            first_brace = result_text.find("{")
            last_brace = result_text.rfind("}")
            if first_brace >= 0 and last_brace > first_brace:
                result_text = result_text[first_brace:last_brace + 1]
            
            claude_result = json.loads(result_text)
            claude_reasoning = claude_result.get('reasoning', '')
            
            logger.info(f"🧠 Claude verdict {symbol}: {claude_result.get('recommendation')} (conf: {claude_result.get('confidence')})")
            
            return {
                'approved': claude_result.get('approved', True),
                'reasoning': claude_reasoning,
                'ai_confidence': claude_result.get('confidence', 5),
                'recommendation': claude_result.get('recommendation', 'BUY'),
                'entry_quality': claude_result.get('entry_quality', 'FAIR'),
                'key_risk': ''
            }
    except Exception as e:
        logger.warning(f"Claude analysis failed for {symbol}: {e}")
    
    if gemini_reasoning:
        return {
            'approved': True,
            'reasoning': gemini_reasoning,
            'ai_confidence': 5,
            'recommendation': 'BUY',
            'key_risk': ''
        }
    
    return {
        'approved': True,
        'reasoning': f"Social momentum signal - Galaxy Score {signal_data['galaxy_score']}/16 with {signal_data['sentiment']*100:.0f}% sentiment",
        'ai_confidence': 4,
        'recommendation': 'BUY',
        'key_risk': ''
    }

# Top 10 coins get higher leverage (more stable)
TOP_10_COINS = {'BTC', 'ETH', 'SOL', 'XRP', 'DOGE', 'ADA', 'AVAX', 'DOT', 'LINK', 'LTC'}

def is_top_coin(symbol: str) -> bool:
    """Check if a symbol is a top 10 coin"""
    base = symbol.replace('USDT', '').replace('PERP', '').upper()
    return base in TOP_10_COINS

# Scanning control
SOCIAL_SCANNING_ENABLED = True
_social_scanning_active = False

# Cooldowns to prevent over-trading
_symbol_cooldowns: Dict[str, datetime] = {}
SYMBOL_COOLDOWN_MINUTES = 60

# Signal tracking
_daily_social_signals = 0
_daily_reset_date: Optional[datetime] = None
MAX_DAILY_SOCIAL_SIGNALS = 6


def is_social_scanning_enabled() -> bool:
    return SOCIAL_SCANNING_ENABLED


def enable_social_scanning():
    global SOCIAL_SCANNING_ENABLED
    SOCIAL_SCANNING_ENABLED = True
    logger.info("📱 Social scanning ENABLED")


def disable_social_scanning():
    global SOCIAL_SCANNING_ENABLED
    SOCIAL_SCANNING_ENABLED = False
    logger.info("📱 Social scanning DISABLED")


def is_symbol_on_cooldown(symbol: str) -> bool:
    """Check if symbol is on cooldown."""
    if symbol in _symbol_cooldowns:
        cooldown_end = _symbol_cooldowns[symbol]
        if datetime.now() < cooldown_end:
            return True
        del _symbol_cooldowns[symbol]
    return False


def add_symbol_cooldown(symbol: str):
    """Add symbol to cooldown."""
    _symbol_cooldowns[symbol] = datetime.now() + timedelta(minutes=SYMBOL_COOLDOWN_MINUTES)


def reset_daily_counters_if_needed():
    """Reset daily counters at midnight UTC."""
    global _daily_social_signals, _daily_reset_date
    
    today = datetime.utcnow().date()
    if _daily_reset_date != today:
        _daily_social_signals = 0
        _daily_reset_date = today
        logger.info("📱 Daily social signal counters reset")


class SocialSignalService:
    """Service for generating trading signals from social data."""
    
    def __init__(self):
        self.http_client: Optional[httpx.AsyncClient] = None
        
    async def init(self):
        """Initialize HTTP client."""
        if not self.http_client:
            self.http_client = httpx.AsyncClient(timeout=15)
    
    async def close(self):
        """Close HTTP client."""
        if self.http_client:
            await self.http_client.aclose()
            self.http_client = None
    
    async def fetch_price_data(self, symbol: str) -> Optional[Dict]:
        """Fetch current price and technical data from Binance, fallback to Bitunix."""
        try:
            await self.init()
            
            result = await self._fetch_binance_price(symbol)
            if result:
                return result
            
            result = await self._fetch_bitunix_price(symbol)
            if result:
                return result
            
            return None
            
        except Exception as e:
            logger.error(f"Error fetching price data for {symbol}: {e}")
            return None
    
    async def _fetch_binance_price(self, symbol: str) -> Optional[Dict]:
        """Try fetching from Binance Futures."""
        try:
            ticker_url = f"https://fapi.binance.com/fapi/v1/ticker/24hr?symbol={symbol}"
            resp = await self.http_client.get(ticker_url, timeout=5)
            
            if resp.status_code != 200:
                return None
            
            ticker = resp.json()
            
            klines_url = f"https://fapi.binance.com/fapi/v1/klines?symbol={symbol}&interval=15m&limit=20"
            klines_resp = await self.http_client.get(klines_url, timeout=5)
            
            closes = []
            volumes = []
            if klines_resp.status_code == 200:
                klines = klines_resp.json()
                closes = [float(k[4]) for k in klines]
                volumes = [float(k[5]) for k in klines]
            
            rsi = self._calc_rsi(closes)
            volume_ratio = self._calc_volume_ratio(volumes)
            
            return {
                'price': float(ticker.get('lastPrice', 0)),
                'change_24h': float(ticker.get('priceChangePercent', 0)),
                'volume_24h': float(ticker.get('quoteVolume', 0)),
                'high_24h': float(ticker.get('highPrice', 0)),
                'low_24h': float(ticker.get('lowPrice', 0)),
                'rsi': rsi,
                'volume_ratio': volume_ratio
            }
        except Exception:
            return None
    
    async def _fetch_bitunix_price(self, symbol: str) -> Optional[Dict]:
        """Fallback: fetch from Bitunix tickers + Binance spot klines for RSI."""
        try:
            url = f"https://fapi.bitunix.com/api/v1/futures/market/tickers?symbols={symbol}"
            resp = await self.http_client.get(url, timeout=5)
            
            if resp.status_code != 200:
                return None
            
            data = resp.json()
            tickers = data.get('data', [])
            if not tickers or not isinstance(tickers, list):
                return None
            
            ticker = tickers[0]
            price = float(ticker.get('lastPrice', 0) or ticker.get('last', 0) or 0)
            if price <= 0:
                return None
            
            open_price = float(ticker.get('open', 0) or 0)
            volume_24h = float(ticker.get('quoteVol', 0) or 0)
            change_24h = ((price - open_price) / open_price * 100) if open_price > 0 else 0
            
            rsi = 50
            volume_ratio = 1.0
            try:
                klines_url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval=15m&limit=20"
                klines_resp = await self.http_client.get(klines_url, timeout=5)
                if klines_resp.status_code == 200:
                    klines = klines_resp.json()
                    closes = [float(k[4]) for k in klines]
                    volumes = [float(k[5]) for k in klines]
                    rsi = self._calc_rsi(closes)
                    volume_ratio = self._calc_volume_ratio(volumes)
                    logger.info(f"  📱 {symbol} - Bitunix price + Binance spot RSI={rsi:.0f}")
                else:
                    logger.info(f"  📱 {symbol} - Bitunix only, RSI unavailable (defaulting 50)")
            except Exception:
                pass
            
            return {
                'price': price,
                'change_24h': change_24h,
                'volume_24h': volume_24h,
                'high_24h': float(ticker.get('high', 0) or 0),
                'low_24h': float(ticker.get('low', 0) or 0),
                'rsi': rsi,
                'volume_ratio': volume_ratio
            }
        except Exception:
            return None
    
    def _calc_rsi(self, closes: list) -> float:
        """Calculate RSI from close prices."""
        if len(closes) < 14:
            return 50
        deltas = [closes[i] - closes[i-1] for i in range(1, len(closes))]
        gains = [d if d > 0 else 0 for d in deltas]
        losses = [-d if d < 0 else 0 for d in deltas]
        avg_gain = sum(gains[-14:]) / 14
        avg_loss = sum(losses[-14:]) / 14
        if avg_loss > 0:
            rs = avg_gain / avg_loss
            return 100 - (100 / (1 + rs))
        return 50
    
    def _calc_volume_ratio(self, volumes: list) -> float:
        """Calculate volume ratio."""
        if len(volumes) < 5:
            return 1.0
        avg_vol = sum(volumes[:-1]) / len(volumes[:-1])
        return volumes[-1] / avg_vol if avg_vol > 0 else 1.0
    
    _bitunix_symbols_cache: Optional[set] = None
    _bitunix_cache_time: Optional[datetime] = None
    
    async def _get_bitunix_symbols(self) -> set:
        """Get cached set of Bitunix symbols."""
        now = datetime.now()
        if self._bitunix_symbols_cache and self._bitunix_cache_time and (now - self._bitunix_cache_time).seconds < 300:
            return self._bitunix_symbols_cache
        
        try:
            await self.init()
            url = "https://fapi.bitunix.com/api/v1/futures/market/trading_pairs"
            resp = await self.http_client.get(url, timeout=10)
            
            if resp.status_code == 200:
                data = resp.json()
                contracts = data.get('data', [])
                symbols = {c.get('symbol', '').upper() for c in contracts if c.get('symbol')}
                SocialSignalService._bitunix_symbols_cache = symbols
                SocialSignalService._bitunix_cache_time = now
                logger.info(f"📱 Cached {len(symbols)} Bitunix symbols")
                return symbols
        except Exception as e:
            logger.error(f"Error fetching Bitunix symbols: {e}")
        
        return self._bitunix_symbols_cache or set()
    
    async def check_bitunix_availability(self, symbol: str) -> bool:
        """Check if symbol is tradeable on Bitunix."""
        try:
            symbols = await self._get_bitunix_symbols()
            return symbol.upper() in symbols
            
        except Exception as e:
            logger.debug(f"Bitunix check failed for {symbol}: {e}")
            return False
    
    async def generate_social_signal(
        self,
        risk_level: str = "MEDIUM",
        min_galaxy_score: int = 8
    ) -> Optional[Dict]:
        """
        Generate a trading signal based on social metrics.
        
        Risk levels affect filters:
        - SAFE: Signal Score ≥70, RSI 40-65, bullish price action only
        - BALANCED: Signal Score ≥60, RSI 35-70, some flexibility
        - AGGRESSIVE: Signal Score ≥50, RSI 30-75, more aggressive
        - NEWS RUNNER: Signal Score ≥80, catch big pumps (+15-30%)
        
        Returns signal dict or None.
        """
        global _daily_social_signals
        
        reset_daily_counters_if_needed()
        
        if _daily_social_signals >= MAX_DAILY_SOCIAL_SIGNALS:
            logger.info(f"📱 Daily social signal limit reached ({MAX_DAILY_SOCIAL_SIGNALS})")
            return None
        
        # Get risk-based filters
        # ALL mode: Smart mode - accepts more signals, TP/SL adapts dynamically
        # RISK = CONFIDENCE FILTER (how strong must the signal be?)
        # TP/SL = ALWAYS DYNAMIC based on actual signal strength
        # 
        # LOW = Only high-confidence signals (can still get big TPs on strong signals!)
        # MEDIUM = Moderate confidence signals
        # HIGH = Accept lower confidence signals
        # ALL = Accept everything, dynamic TPs
        
        if risk_level == "LOW":
            min_score = max(15, min_galaxy_score)
            rsi_range = (40, 65)
            require_positive_change = True
            min_sentiment = 0.4
        elif risk_level == "MEDIUM":
            min_score = max(13, min_galaxy_score)
            rsi_range = (35, 70)
            require_positive_change = False
            min_sentiment = 0.2
        elif risk_level == "HIGH":
            min_score = max(12, min_galaxy_score)
            rsi_range = (30, 75)
            require_positive_change = False
            min_sentiment = 0.1
        else:  # ALL or MOMENTUM
            min_score = max(10, min_galaxy_score)
            rsi_range = (25, 80)
            require_positive_change = False
            min_sentiment = 0.0
        
        logger.info(f"📱 SOCIAL SCANNER | Risk: {risk_level} | Min Score: {min_score}")
        
        # Get trending coins from social data
        trending = await get_trending_coins(limit=30)
        
        if not trending:
            logger.warning("📱 No trending coins from social data")
            return None
        
        logger.info(f"📱 Found {len(trending)} trending coins to analyze")
        
        passed_filters = 0
        for coin in trending:
            symbol = coin['symbol']
            galaxy_score = coin['galaxy_score']
            sentiment = coin.get('sentiment', 0)
            social_volume = coin.get('social_volume', 0)
            social_interactions = coin.get('social_interactions', 0) or coin.get('interactions_24h', 0) or 0
            social_dominance = coin.get('social_dominance', 0) or 0
            alt_rank = coin.get('alt_rank', 9999) or 9999
            coin_name = coin.get('name', '')
            price_change = coin.get('percent_change_24h', 0)
            
            if is_symbol_on_cooldown(symbol):
                logger.debug(f"  📱 {symbol} - On cooldown, skipping")
                continue
            
            if galaxy_score < min_score:
                logger.debug(f"  📱 {symbol} - Galaxy {galaxy_score} < {min_score}")
                continue
            
            if sentiment < min_sentiment:
                logger.info(f"  📱 {symbol} - Sentiment {sentiment:.2f} < {min_sentiment}")
                continue
            
            if require_positive_change and price_change < 0:
                logger.info(f"  📱 {symbol} - Negative 24h change {price_change:.1f}% (need positive)")
                continue
            
            passed_filters += 1
            logger.info(f"  📱 {symbol} - gs={galaxy_score} sent={sentiment:.2f} chg={price_change:.1f}% - checking availability...")
            
            is_available = await self.check_bitunix_availability(symbol)
            if not is_available:
                logger.info(f"  📱 {symbol} - ❌ Not on Bitunix")
                continue
            
            price_data = await self.fetch_price_data(symbol)
            if not price_data:
                logger.info(f"  📱 {symbol} - ❌ No price data from any source")
                continue
            
            current_price = price_data['price']
            rsi = price_data['rsi']
            volume_24h = price_data['volume_24h']
            
            min_vol = 1_000_000
            if volume_24h < min_vol:
                logger.info(f"  📱 {symbol} - ❌ Low volume ${volume_24h/1e6:.1f}M (need $1M+)")
                continue
            
            if not (rsi_range[0] <= rsi <= rsi_range[1]):
                logger.info(f"  📱 {symbol} - ❌ RSI {rsi:.0f} outside range {rsi_range}")
                continue
            
            # 🎉 SIGNAL FOUND!
            logger.info(f"✅ SOCIAL SIGNAL: {symbol} | Score: {galaxy_score} | Sentiment: {sentiment:.2f} | RSI: {rsi:.0f}")
            
            # 🚀 DYNAMIC TP/SL - ALWAYS based on signal score
            # Higher score = stronger signal = bigger TP potential
            
            if galaxy_score >= 15:
                tp_percent = 30.0 + (sentiment * 20)  # 30-50%
                sl_percent = 12.0
            elif galaxy_score >= 14:
                tp_percent = 20.0 + (sentiment * 15)  # 20-35%
                sl_percent = 10.0
            elif galaxy_score >= 13:
                tp_percent = 15.0 + (sentiment * 10)  # 15-25%
                sl_percent = 8.0
            elif galaxy_score >= 12:
                tp_percent = 12.0 + (sentiment * 8)  # 12-20%
                sl_percent = 7.0
            elif galaxy_score >= 10:
                tp_percent = 10.0 + (sentiment * 5)  # 10-15%
                sl_percent = 6.0
            else:
                tp_percent = 8.0 + (sentiment * 4)  # 8-12%
                sl_percent = 5.0
            
            take_profit = current_price * (1 + tp_percent / 100)
            stop_loss = current_price * (1 - sl_percent / 100)
            
            tp2 = current_price * (1 + (tp_percent * 1.5) / 100)
            tp3 = current_price * (1 + (tp_percent * 2.0) / 100)
            
            signal_candidate = {
                'symbol': symbol,
                'direction': 'LONG',
                'entry_price': current_price,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'take_profit_1': take_profit,
                'take_profit_2': tp2,
                'take_profit_3': tp3,
                'tp_percent': tp_percent,
                'sl_percent': sl_percent,
                'confidence': int(galaxy_score),
                'galaxy_score': galaxy_score,
                'sentiment': sentiment,
                'social_volume': social_volume,
                'rsi': rsi,
                '24h_change': price_change,
                '24h_volume': volume_24h,
            }
            
            ai_result = await ai_analyze_social_signal(signal_candidate)
            
            if not ai_result.get('approved', True):
                logger.info(f"🤖 AI REJECTED {symbol} LONG: {ai_result.get('reasoning', 'No reason')}")
                continue
            
            ai_reasoning = ai_result.get('reasoning', '')
            ai_confidence = ai_result.get('ai_confidence', 5)
            ai_recommendation = ai_result.get('recommendation', 'BUY')
            
            add_symbol_cooldown(symbol)
            _daily_social_signals += 1
            
            return {
                'symbol': symbol,
                'direction': 'LONG',
                'entry_price': current_price,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'take_profit_1': take_profit,
                'take_profit_2': tp2,
                'take_profit_3': tp3,
                'tp_percent': tp_percent,
                'sl_percent': sl_percent,
                'confidence': int(galaxy_score),
                'reasoning': ai_reasoning,
                'ai_confidence': ai_confidence,
                'ai_recommendation': ai_recommendation,
                'trade_type': 'SOCIAL_SIGNAL',
                'strategy': 'NEWS_MOMENTUM' if risk_level == "MOMENTUM" else 'SOCIAL_SIGNAL',
                'risk_level': risk_level,
                'galaxy_score': galaxy_score,
                'sentiment': sentiment,
                'social_volume': social_volume,
                'social_interactions': social_interactions,
                'social_dominance': social_dominance,
                'alt_rank': alt_rank,
                'coin_name': coin_name,
                'rsi': rsi,
                '24h_change': price_change,
                '24h_volume': volume_24h
            }
        
        logger.info(f"📱 No valid social LONG signals found ({passed_filters} passed initial filters)")
        return None
    
    async def scan_for_short_signal(
        self,
        risk_level: str = "MEDIUM",
        min_galaxy_score: int = 8
    ) -> Optional[Dict]:
        """
        Scan for SHORT signals based on negative social sentiment and bearish indicators.
        Triggers: FUD, negative news, sentiment crash, failed pumps, distribution patterns.
        """
        global _daily_social_signals
        
        reset_daily_counters_if_needed()
        
        if _daily_social_signals >= MAX_DAILY_SOCIAL_SIGNALS:
            logger.info("📱 Daily social signal limit reached")
            return None
        
        await self.init()
        
        # RISK = CONFIDENCE FILTER for shorts
        # TP/SL = ALWAYS DYNAMIC based on signal strength
        
        if risk_level == "LOW":
            min_score = 15
            rsi_range = (65, 85)
            require_negative_change = True
            max_sentiment = 0.3
        elif risk_level == "MEDIUM":
            min_score = 13
            rsi_range = (60, 85)
            require_negative_change = True
            max_sentiment = 0.4
        elif risk_level == "HIGH":
            min_score = 12
            rsi_range = (55, 90)
            require_negative_change = True
            max_sentiment = 0.5
        else:  # ALL
            min_score = 10
            rsi_range = (50, 95)
            require_negative_change = False
            max_sentiment = 0.6
        
        logger.info(f"📉 SOCIAL SHORT SCANNER | Risk: {risk_level} | Max Sentiment: {max_sentiment}")
        
        # Get trending coins (even bearish ones get attention)
        trending = await get_trending_coins(limit=30)
        
        if not trending:
            logger.warning("📉 No trending coins for short scan")
            return None
        
        for coin in trending:
            symbol = coin['symbol']
            galaxy_score = coin['galaxy_score']
            sentiment = coin.get('sentiment', 0)
            social_volume = coin.get('social_volume', 0)
            price_change = coin.get('percent_change_24h', 0)
            
            # Skip if on cooldown
            if is_symbol_on_cooldown(symbol):
                continue
            
            # Need social attention but BEARISH sentiment
            if galaxy_score < min_score:
                continue
            
            # Key filter: sentiment must be bearish or neutral (for shorts)
            if sentiment > max_sentiment:
                logger.debug(f"  {symbol} - Sentiment {sentiment:.2f} too bullish for short")
                continue
            
            # For safer shorts, require coin to be dropping
            if require_negative_change and price_change > 0:
                continue
            
            # Check Bitunix availability
            is_available = await self.check_bitunix_availability(symbol)
            if not is_available:
                continue
            
            # Get price data
            price_data = await self.fetch_price_data(symbol)
            if not price_data:
                continue
            
            current_price = price_data['price']
            rsi = price_data['rsi']
            volume_24h = price_data['volume_24h']
            
            # Liquidity check
            if volume_24h < 1_000_000:
                logger.info(f"  📉 {symbol} - ❌ Low volume ${volume_24h/1e6:.1f}M (need $1M+)")
                continue
            
            # RSI filter - want overbought or topping
            if not (rsi_range[0] <= rsi <= rsi_range[1]):
                logger.debug(f"  {symbol} - RSI {rsi:.0f} not in short range {rsi_range}")
                continue
            
            # 🎉 SHORT SIGNAL FOUND!
            logger.info(f"✅ SOCIAL SHORT: {symbol} | Score: {galaxy_score} | Sentiment: {sentiment:.2f} | RSI: {rsi:.0f}")
            
            bearish_strength = max(0, 1.0 - sentiment)  # Lower sentiment = more bearish (0-1 scale)
            
            if galaxy_score >= 15:
                tp_percent = 25.0 + (bearish_strength * 15)  # 25-40%
                sl_percent = 12.0
            elif galaxy_score >= 14:
                tp_percent = 18.0 + (bearish_strength * 12)  # 18-30%
                sl_percent = 10.0
            elif galaxy_score >= 13:
                tp_percent = 14.0 + (bearish_strength * 8)  # 14-22%
                sl_percent = 8.0
            elif galaxy_score >= 12:
                tp_percent = 10.0 + (bearish_strength * 6)  # 10-16%
                sl_percent = 7.0
            elif galaxy_score >= 10:
                tp_percent = 8.0 + (bearish_strength * 4)  # 8-12%
                sl_percent = 6.0
            else:
                tp_percent = 6.0 + (bearish_strength * 3)  # 6-9%
                sl_percent = 5.0
            
            # For SHORTS: TP is below entry, SL is above entry
            take_profit = current_price * (1 - tp_percent / 100)
            stop_loss = current_price * (1 + sl_percent / 100)
            
            signal_candidate = {
                'symbol': symbol,
                'direction': 'SHORT',
                'entry_price': current_price,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'take_profit_1': take_profit,
                'take_profit_2': None,
                'take_profit_3': None,
                'tp_percent': tp_percent,
                'sl_percent': sl_percent,
                'confidence': int(galaxy_score),
                'galaxy_score': galaxy_score,
                'sentiment': sentiment,
                'social_volume': social_volume,
                'rsi': rsi,
                '24h_change': price_change,
                '24h_volume': volume_24h,
            }
            
            ai_result = await ai_analyze_social_signal(signal_candidate)
            
            if not ai_result.get('approved', True):
                logger.info(f"🤖 AI REJECTED {symbol} SHORT: {ai_result.get('reasoning', 'No reason')}")
                continue
            
            ai_reasoning = ai_result.get('reasoning', '')
            ai_confidence = ai_result.get('ai_confidence', 5)
            ai_recommendation = ai_result.get('recommendation', 'BUY')
            
            add_symbol_cooldown(symbol)
            _daily_social_signals += 1
            
            return {
                'symbol': symbol,
                'direction': 'SHORT',
                'entry_price': current_price,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'take_profit_1': take_profit,
                'take_profit_2': None,
                'take_profit_3': None,
                'tp_percent': tp_percent,
                'sl_percent': sl_percent,
                'confidence': int(galaxy_score),
                'reasoning': ai_reasoning,
                'ai_confidence': ai_confidence,
                'ai_recommendation': ai_recommendation,
                'trade_type': 'SOCIAL_SHORT',
                'strategy': 'SOCIAL_SHORT',
                'risk_level': risk_level,
                'galaxy_score': galaxy_score,
                'sentiment': sentiment,
                'social_volume': social_volume,
                'rsi': rsi,
                '24h_change': price_change,
                '24h_volume': volume_24h
            }
        
        logger.info("📉 No valid social SHORT signals found")
        return None


async def broadcast_social_signal(db_session: Session, bot):
    """
    Main function to scan for social signals and broadcast to enabled users.
    Runs independently of Top Gainers mode.
    """
    global _social_scanning_active
    
    if not SOCIAL_SCANNING_ENABLED:
        logger.debug("📱 Social scanning disabled")
        return
    
    if _social_scanning_active:
        logger.debug("📱 Social scan already in progress")
        return
    
    # Check API key
    if not get_lunarcrush_api_key():
        logger.warning("📱 No API key configured - skipping social scan")
        return
    
    _social_scanning_active = True
    
    try:
        from app.models import User, UserPreference, Signal
        
        # Get users with social mode enabled
        users_with_social = db_session.query(User).join(UserPreference).filter(
            UserPreference.social_mode_enabled == True
        ).all()
        
        if not users_with_social:
            logger.debug("📱 No users with social mode enabled")
            return
        
        logger.info(f"📱 ═══════════════════════════════════════════════════════")
        logger.info(f"📱 SOCIAL SIGNALS SCANNER - {len(users_with_social)} users enabled")
        logger.info(f"📱 ═══════════════════════════════════════════════════════")
        
        service = SocialSignalService()
        await service.init()
        
        # Use the most common risk level among users (or default to MEDIUM)
        risk_levels = [u.preferences.social_risk_level or "MEDIUM" for u in users_with_social if u.preferences]
        most_common_risk = max(set(risk_levels), key=risk_levels.count) if risk_levels else "MEDIUM"
        
        # Use lowest min galaxy score to catch more signals
        min_scores = [u.preferences.social_min_galaxy_score or 8 for u in users_with_social if u.preferences]
        min_galaxy = min(min_scores) if min_scores else 8
        
        # 1. PRIORITY: Check for BREAKING NEWS first (fastest signals)
        try:
            from app.services.realtime_news import scan_for_breaking_news_signal
            signal = await scan_for_breaking_news_signal(
                check_bitunix_func=service.check_bitunix_availability,
                fetch_price_func=service.fetch_price_data
            )
            if signal:
                logger.info(f"📰 BREAKING NEWS SIGNAL: {signal['symbol']} {signal['direction']}")
        except Exception as e:
            logger.error(f"Breaking news scan error: {e}")
            signal = None
        
        # 2. Scan for social LONG signals
        if not signal:
            signal = await service.generate_social_signal(
                risk_level=most_common_risk,
                min_galaxy_score=min_galaxy
            )
        
        # 3. If no LONG, try SHORT signals
        if not signal:
            signal = await service.scan_for_short_signal(
                risk_level=most_common_risk,
                min_galaxy_score=min_galaxy
            )
        
        if signal:
            # Format and broadcast signal
            symbol = signal['symbol']
            entry = signal['entry_price']
            sl = signal['stop_loss']
            tp = signal['take_profit']
            galaxy = signal['galaxy_score']
            sentiment = signal['sentiment']
            direction = signal.get('direction', 'LONG')
            
            # Determine leverage based on coin type
            is_top = is_top_coin(symbol)
            
            rating = interpret_signal_score(galaxy)
            
            def fmt_price(p):
                if p >= 1000:
                    return f"${p:,.2f}"
                elif p >= 1:
                    return f"${p:.4f}"
                elif p >= 0.01:
                    return f"${p:.6f}"
                elif p >= 0.0001:
                    return f"${p:.8f}"
                else:
                    return f"${p:.10f}"
            
            tp_pct = signal.get('tp_percent', 0)
            sl_pct = signal.get('sl_percent', 0)
            tp2 = signal.get('take_profit_2')
            tp3 = signal.get('take_profit_3')
            risk_level = signal.get('risk_level', 'MEDIUM')
            social_vol = signal.get('social_volume', 0)
            rsi_val = signal.get('rsi', 50)
            volume_24h = signal.get('24h_volume', 0)
            change_24h = signal.get('24h_change', 0)
            
            dir_icon = "🟢" if direction == 'LONG' else "🔴"
            
            tp_lines = f"🎯 TP1  <code>{fmt_price(tp)}</code>  <b>+{tp_pct:.1f}%</b>"
            if tp2:
                tp2_pct = tp_pct * 1.5
                tp_lines += f"\n🎯 TP2  <code>{fmt_price(tp2)}</code>  <b>+{tp2_pct:.1f}%</b>"
            if tp3:
                tp3_pct = tp_pct * 2.0
                tp_lines += f"\n🎯 TP3  <code>{fmt_price(tp3)}</code>  <b>+{tp3_pct:.1f}%</b>"
            
            if direction == 'SHORT':
                tp_lines = f"🎯 TP1  <code>{fmt_price(tp)}</code>  <b>-{tp_pct:.1f}%</b>"
            
            vol_display = f"${volume_24h/1e6:.1f}M" if volume_24h >= 1e6 else f"${volume_24h/1e3:.0f}K"
            
            is_news_signal = signal.get('trade_type') == 'NEWS_SIGNAL'
            news_title = signal.get('news_title', '')
            
            if is_news_signal:
                trigger = signal.get('trigger_reason', 'Breaking News')
                short_title = news_title[:70] + '...' if len(news_title) > 70 else news_title
                reasoning = signal.get('reasoning', '')[:200] if signal.get('reasoning') else ''
                
                message = (
                    f"{dir_icon} <b>NEWS {direction}</b>\n\n"
                    f"<b>{symbol}</b>\n"
                    f"<i>{short_title}</i>\n\n"
                    f"💵  Entry  <code>{fmt_price(entry)}</code>\n"
                    f"{tp_lines}\n"
                    f"🛑  SL  <code>{fmt_price(sl)}</code>  <b>-{sl_pct:.1f}%</b>\n\n"
                    f"⚡ Score {galaxy}  ·  {trigger}"
                )
                if reasoning:
                    message += f"\n\n💡 <i>{reasoning}</i>"
            else:
                sentiment_pct = int(sentiment * 100)
                ai_reasoning = signal.get('reasoning', '')
                ai_rec = signal.get('ai_recommendation', '')
                ai_conf = signal.get('ai_confidence', 0)
                social_interactions = signal.get('social_interactions', 0)
                social_dominance = signal.get('social_dominance', 0)
                alt_rank = signal.get('alt_rank', 9999)
                coin_name = signal.get('coin_name', '')
                
                rec_emoji = {"STRONG BUY": "🚀", "BUY": "✅", "HOLD": "⏸️", "AVOID": "🚫"}.get(ai_rec, "📊")
                
                interactions_display = f"{social_interactions/1e6:.1f}M" if social_interactions >= 1e6 else f"{social_interactions/1e3:.1f}K" if social_interactions >= 1000 else f"{social_interactions:,}"
                
                name_display = f" ({coin_name})" if coin_name else ""
                
                message = (
                    f"{dir_icon} <b>SOCIAL {direction}</b>\n\n"
                    f"<b>{symbol}</b>{name_display}\n\n"
                    f"💵  Entry  <code>{fmt_price(entry)}</code>\n"
                    f"{tp_lines}\n"
                    f"🛑  SL  <code>{fmt_price(sl)}</code>  <b>-{sl_pct:.1f}%</b>\n\n"
                    f"<b>📊 Social Data (LunarCrush)</b>\n"
                    f"🌙 Galaxy Score <b>{galaxy}/16</b>  ·  {rating}\n"
                    f"💬 Sentiment <b>{sentiment_pct}%</b> bullish across social media\n"
                    f"🔊 Posts <b>{social_vol:,}</b>  ·  Interactions <b>{interactions_display}</b>\n"
                )
                
                if social_dominance > 0:
                    message += f"📡 Social Dominance <b>{social_dominance:.2f}%</b>"
                    if alt_rank < 9999:
                        message += f"  ·  AltRank <b>#{alt_rank}</b>"
                    message += "\n"
                
                message += (
                    f"\n<b>📈 Market Data</b>\n"
                    f"RSI <b>{rsi_val:.0f}</b>  ·  24h <b>{change_24h:+.1f}%</b>  ·  Vol <b>{vol_display}</b>"
                )
                
                if ai_reasoning:
                    message += f"\n\n{rec_emoji} <b>AI: {ai_rec}</b> (Confidence {ai_conf}/10)\n💡 <i>{ai_reasoning}</i>"
            
            # Record signal in database FIRST (needed for trade execution)
            default_lev = 25 if is_top else 10
            new_signal = Signal(
                user_id=users_with_social[0].id if users_with_social else None,
                symbol=symbol,
                direction=direction,
                entry_price=entry,
                stop_loss=sl,
                take_profit=tp,
                take_profit_1=tp,
                take_profit_2=signal.get('take_profit_2'),
                take_profit_3=signal.get('take_profit_3'),
                leverage=default_lev,
                confidence=galaxy,
                trade_type='SOCIAL_SIGNAL',
                signal_type='SOCIAL_SIGNAL',
                timeframe='15m',
                rsi=rsi_val,
                volume=volume_24h,
                reasoning=signal['reasoning']
            )
            db_session.add(new_signal)
            db_session.commit()
            db_session.refresh(new_signal)
            
            # Send message + execute trade for each user
            for user in users_with_social:
                try:
                    prefs = user.preferences
                    if is_top:
                        user_lev = getattr(prefs, 'social_top_coin_leverage', 25) or 25 if prefs else 25
                        coin_type = "🏆"
                    else:
                        user_lev = getattr(prefs, 'social_leverage', 10) or 10 if prefs else 10
                        coin_type = "📊"
                    
                    lev_line = f"\n\n{coin_type} {user_lev}x"
                    user_message = message + lev_line
                    
                    await bot.send_message(
                        user.telegram_id,
                        user_message,
                        parse_mode="HTML"
                    )
                    logger.info(f"📱 Sent social {direction} signal {symbol} to user {user.telegram_id} @ {user_lev}x")
                    
                    # AUTO-TRADE: Execute on Bitunix if user has API keys configured
                    if prefs and prefs.bitunix_api_key and prefs.bitunix_api_secret:
                        try:
                            from app.services.bitunix_trader import execute_bitunix_trade
                            trade_result = await execute_bitunix_trade(
                                signal=new_signal,
                                user=user,
                                db=db_session,
                                trade_type='SOCIAL_SIGNAL',
                                leverage_override=user_lev
                            )
                            if trade_result:
                                logger.info(f"✅ Auto-traded {symbol} {direction} for user {user.telegram_id} @ {user_lev}x")
                                await bot.send_message(
                                    user.telegram_id,
                                    f"✅ <b>Trade Executed on Bitunix</b>\n"
                                    f"<b>{symbol}</b> {direction} @ {user_lev}x",
                                    parse_mode="HTML"
                                )
                            else:
                                logger.warning(f"⚠️ Auto-trade returned None for {symbol} user {user.telegram_id}")
                        except Exception as trade_err:
                            logger.error(f"❌ Auto-trade failed for {symbol} user {user.telegram_id}: {trade_err}")
                            await bot.send_message(
                                user.telegram_id,
                                f"⚠️ <b>Auto-Trade Failed</b>\n"
                                f"<b>{symbol}</b> {direction}\n"
                                f"<i>{str(trade_err)[:100]}</i>",
                                parse_mode="HTML"
                            )
                    else:
                        logger.info(f"📱 User {user.telegram_id} - No Bitunix API keys, signal only")
                except Exception as e:
                    logger.error(f"Failed to send/execute social signal for {user.telegram_id}: {e}")
        
        await service.close()
        
    except Exception as e:
        logger.error(f"Error in social signal broadcast: {e}")
    finally:
        _social_scanning_active = False
