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
from app.services.coinglass import calculate_signal_strength, format_signal_strength_detail
from app.services.coinglass import (
    get_derivatives_summary,
    format_derivatives_for_ai,
    format_derivatives_for_message,
    adjust_tp_sl_from_derivatives
)

logger = logging.getLogger(__name__)

MAJOR_COINS = {'BTC', 'ETH', 'SOL', 'BNB', 'XRP', 'ADA', 'DOGE', 'AVAX', 'DOT', 'LINK'}


async def ai_analyze_social_signal(signal_data: Dict) -> Dict:
    """
    Use Gemini for fast scan + Claude for final approval of social signals.
    Returns dict with 'approved', 'reasoning', 'ai_confidence', 'recommendation'.
    """
    symbol = signal_data['symbol']
    direction = signal_data.get('direction', 'LONG')
    
    deriv_data = signal_data.get('derivatives', {})
    deriv_summary = format_derivatives_for_ai(deriv_data) if deriv_data else ""
    
    deriv_adj = signal_data.get('deriv_adjustments', [])
    adj_note = ""
    if deriv_adj:
        adj_note = "\nTP/SL Adjusted by Derivatives:\n" + "\n".join(f"  • {a}" for a in deriv_adj)
    
    is_news = signal_data.get('trade_type') == 'NEWS_SIGNAL'
    news_context = signal_data.get('news_context', '')
    
    volume_raw = signal_data.get('24h_volume', 0)
    vol_str = f"${volume_raw/1e6:.1f}M" if volume_raw >= 1e6 else f"${volume_raw/1e3:.0f}K"
    
    data_summary = (
        f"Coin: {symbol}\n"
        f"Direction: {direction}\n"
        f"Entry: ${signal_data['entry_price']}\n"
    )
    
    if is_news and news_context:
        data_summary += f"Signal Type: NEWS-DRIVEN TRADE\n{news_context}\n"
    else:
        galaxy = signal_data.get('galaxy_score', 0)
        sentiment = signal_data.get('sentiment', 0)
        social_vol = signal_data.get('social_volume', 0)
        social_strength = signal_data.get('social_strength', 0)
        social_interactions = signal_data.get('social_interactions', 0) or 0
        social_dominance = signal_data.get('social_dominance', 0) or 0
        alt_rank = signal_data.get('alt_rank', 9999)
        social_vol_change = signal_data.get('social_vol_change', 0)
        is_spike = signal_data.get('is_social_spike', False)
        
        data_summary += (
            f"Galaxy Score: {galaxy}/16\n"
            f"Social Strength: {social_strength:.0f}/100 (composite)\n"
            f"Sentiment: {sentiment*100:.0f}%\n"
            f"Social Volume: {social_vol:,}\n"
            f"Social Interactions: {social_interactions:,}\n"
            f"Social Dominance: {social_dominance:.2f}%\n"
            f"AltRank: #{alt_rank}\n"
        )
        if social_vol_change > 0:
            data_summary += f"Social Volume 24h Change: +{social_vol_change:.0f}%\n"
        if is_spike:
            data_summary += "⚠️ SOCIAL SPIKE DETECTED - social buzz surging rapidly\n"
    
    btc_corr = signal_data.get('btc_correlation', 0)
    vol_ratio = signal_data.get('volume_ratio', 1.0)
    
    data_summary += (
        f"RSI (15m): {signal_data.get('rsi', 50):.0f}\n"
        f"24h Change: {signal_data.get('24h_change', 0):+.1f}%\n"
        f"24h Volume: {vol_str}\n"
        f"BTC Correlation: {btc_corr:.0%}\n"
        f"TP: +{signal_data['tp_percent']:.1f}%\n"
        f"SL: -{signal_data['sl_percent']:.1f}%"
        f"{adj_note}"
    )
    
    influencer_data = signal_data.get('influencer_consensus')
    if influencer_data and isinstance(influencer_data, dict) and influencer_data.get('num_creators', 0) > 0:
        data_summary += (
            f"\n\n--- INFLUENCER INTELLIGENCE ---\n"
            f"Influencer Consensus: {influencer_data.get('consensus', 'UNKNOWN')}\n"
            f"Creators Talking: {influencer_data.get('num_creators', 0)} "
            f"(Bullish: {influencer_data.get('bullish_count', 0)}, Bearish: {influencer_data.get('bearish_count', 0)}, Neutral: {influencer_data.get('neutral_count', 0)})\n"
            f"Avg Influencer Sentiment: {influencer_data.get('avg_sentiment', 50):.0f}/100\n"
            f"Combined Followers: {influencer_data.get('total_followers', 0):,}\n"
            f"Combined Interactions: {influencer_data.get('total_interactions', 0):,}\n"
        )
        if influencer_data.get('big_accounts', 0) > 0:
            data_summary += f"Big Accounts (50K+ followers): {influencer_data.get('big_accounts', 0)} (sentiment: {influencer_data.get('big_account_sentiment', 50):.0f}/100)\n"
        top_names = [c.get('name', '') for c in influencer_data.get('top_creators', []) if c.get('name')]
        if top_names:
            data_summary += f"Top Creators: {', '.join(top_names)}\n"
    
    buzz_data = signal_data.get('buzz_momentum')
    if buzz_data and isinstance(buzz_data, dict) and buzz_data.get('trend'):
        data_summary += (
            f"\n--- SOCIAL BUZZ MOMENTUM ---\n"
            f"Buzz Trend: {buzz_data.get('trend', 'UNKNOWN')} (momentum score: {buzz_data.get('momentum_score', 0):.0f})\n"
            f"Buzz Change: {buzz_data.get('buzz_change_pct', 0):+.1f}%\n"
            f"Sentiment Trend: {buzz_data.get('sentiment_trend', 'UNKNOWN')} ({buzz_data.get('sentiment_change', 0):+.1f} pts)\n"
            f"Recent vs Prior Interactions: {buzz_data.get('recent_avg_interactions', 0):.0f} vs {buzz_data.get('prior_avg_interactions', 0):.0f}\n"
        )
    
    if deriv_summary:
        data_summary += f"\n\n{deriv_summary}"
    
    # STEP 1: Gemini fast scan
    gemini_reasoning = None
    try:
        from app.services.ai_market_intelligence import get_gemini_client
        gemini = get_gemini_client()
        if gemini:
            signal_type_desc = "a NEWS-DRIVEN trading signal based on breaking crypto news" if is_news else "a social sentiment signal with LunarCrush data"
            if is_news:
                social_instruction = "3. Does the news headline justify immediate entry? Is the impact significant enough to move the price?"
            else:
                social_instruction = """3. SOCIAL ANALYSIS (critical):
   - Is the Social Strength score (composite of galaxy score, sentiment, interactions, dominance, alt rank) high enough?
   - If this is a SOCIAL SPIKE, is the buzz surge likely to drive price action or is it just noise?
   - Does the social volume and interaction count suggest real interest or just bots/spam?
   - Is the BTC correlation low enough that this coin can move independently?
   - INFLUENCER CHECK: Are top influencers aligned with the trade direction? Big accounts (50K+ followers) carry more weight.
   - BUZZ MOMENTUM: Is social buzz RISING, STABLE, or FALLING? Rising buzz with bullish sentiment = strong LONG confirmation. Falling buzz = weakening conviction."""
            
            prompt = f"""You are an aggressive crypto perps scalp trader. Your job is to FIND TRADES, not avoid them. You make money by taking quick positions with tight stops.

{data_summary}

Analyze this {direction} signal. Your DEFAULT is to APPROVE unless there's a clear dealbreaker:
- RSI >90 for longs or RSI <10 for shorts = reject
- Extremely against the trend with no reversal signs = reject
- Everything else = APPROVE and set confidence level

{social_instruction}

IMPORTANT: You are a TRADER, not a risk manager. Tight stop losses protect us. APPROVE most signals and let the stops do the work.

Only reject if there's an OBVIOUS reason NOT to trade. RSI ranges, moderate correlations, mixed sentiment = NOT reasons to reject.

Respond in JSON:
{{
    "scan_pass": true/false,
    "reasoning": "2-3 sentence sharp analysis. Focus on the OPPORTUNITY not the risks.",
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
            
            signal_desc = "news-driven trading signal" if is_news else "social sentiment signal"
            news_extra = "\n- News catalyst assessment: Is this significant enough to move price?" if is_news else ""
            if direction == 'SHORT':
                rec_options = '"STRONG SELL" or "SELL" or "HOLD" or "AVOID"'
            else:
                rec_options = '"STRONG BUY" or "BUY" or "HOLD" or "AVOID"'
            
            claude_prompt = f"""You are an aggressive crypto perpetual futures scalp trader reviewing a {signal_desc}. You WANT to take trades. Tight stop losses protect your downside.

{data_summary}
{gemini_context}

TRADING RULES:
- Your DEFAULT answer is APPROVED. You need a strong reason to REJECT.
- Tight SL means limited risk. APPROVE and let stops work.
- Only REJECT for: extreme RSI (>90 long, <10 short), clearly exhausted moves, or obvious manipulation.
- Mixed signals, moderate risk, imperfect setups = STILL APPROVE with adjusted confidence.
- Derivatives data is supplementary context, NOT a veto. Funding rates and OI can lag.{news_extra}
- CRITICAL: This is a {direction} signal. Your recommendation MUST match the direction.
  For SHORT signals use STRONG SELL/SELL. For LONG signals use STRONG BUY/BUY. Never say BUY on a SHORT.

Respond in JSON:
{{
    "approved": true/false,
    "confidence": 1-10,
    "recommendation": {rec_options},
    "reasoning": "2-3 sentence concise analysis. Focus on opportunity. Be direct and actionable.",
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
            
            rec = claude_result.get('recommendation', '')
            if direction == 'SHORT' and rec in ('STRONG BUY', 'BUY'):
                rec = rec.replace('BUY', 'SELL')
            elif direction == 'LONG' and rec in ('STRONG SELL', 'SELL'):
                rec = rec.replace('SELL', 'BUY')
            if not rec:
                rec = 'SELL' if direction == 'SHORT' else 'BUY'
            
            return {
                'approved': claude_result.get('approved', True),
                'reasoning': claude_reasoning,
                'ai_confidence': claude_result.get('confidence', 5),
                'recommendation': rec,
                'entry_quality': claude_result.get('entry_quality', 'FAIR'),
                'key_risk': ''
            }
    except Exception as e:
        logger.warning(f"Claude analysis failed for {symbol}: {e}")
    
    default_rec = 'SELL' if direction == 'SHORT' else 'BUY'
    
    if gemini_reasoning:
        return {
            'approved': True,
            'reasoning': gemini_reasoning,
            'ai_confidence': 5,
            'recommendation': default_rec,
            'key_risk': ''
        }
    
    if is_news:
        return {
            'approved': True,
            'reasoning': f"News-driven signal - AI unavailable, proceeding with impact score {signal_data.get('confidence', 0)}/100",
            'ai_confidence': 4,
            'recommendation': default_rec,
            'key_risk': ''
        }
    
    return {
        'approved': True,
        'reasoning': f"Social momentum signal - Galaxy Score {signal_data.get('galaxy_score', 0)}/16 with {signal_data.get('sentiment', 0)*100:.0f}% sentiment",
        'ai_confidence': 4,
        'recommendation': default_rec,
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
SYMBOL_COOLDOWN_MINUTES = 15

# AI rejection cooldown before re-analyzing a rejected coin
_ai_rejection_cache: Dict[str, datetime] = {}
AI_REJECTION_COOLDOWN_MINUTES = 15

_signalled_cooldown: Dict[str, datetime] = {}
SIGNALLED_COOLDOWN_HOURS = 6


def is_coin_in_signalled_cooldown(symbol: str) -> bool:
    if symbol in _signalled_cooldown:
        signalled_at = _signalled_cooldown[symbol]
        elapsed = datetime.now() - signalled_at
        if elapsed < timedelta(hours=SIGNALLED_COOLDOWN_HOURS):
            remaining_hrs = (SIGNALLED_COOLDOWN_HOURS * 3600 - elapsed.total_seconds()) / 3600
            logger.info(f"🔇 {symbol} already signalled - {remaining_hrs:.1f}h cooldown remaining")
            return True
        else:
            del _signalled_cooldown[symbol]
    return False


def add_to_signalled_cooldown(symbol: str):
    _signalled_cooldown[symbol] = datetime.now()
    logger.info(f"⏰ {symbol} added to {SIGNALLED_COOLDOWN_HOURS}h signal cooldown")


def is_coin_in_ai_rejection_cooldown(symbol: str, direction: str) -> bool:
    cache_key = f"{symbol}_{direction}"
    if cache_key in _ai_rejection_cache:
        rejected_at = _ai_rejection_cache[cache_key]
        if datetime.now() - rejected_at < timedelta(minutes=AI_REJECTION_COOLDOWN_MINUTES):
            remaining = AI_REJECTION_COOLDOWN_MINUTES - (datetime.now() - rejected_at).total_seconds() / 60
            logger.debug(f"⏳ {symbol} {direction} in AI rejection cooldown ({remaining:.0f}min left)")
            return True
        else:
            del _ai_rejection_cache[cache_key]
    return False


def add_to_ai_rejection_cooldown(symbol: str, direction: str):
    cache_key = f"{symbol}_{direction}"
    _ai_rejection_cache[cache_key] = datetime.now()
    logger.info(f"📝 {symbol} {direction} added to AI rejection cooldown for {AI_REJECTION_COOLDOWN_MINUTES}min")

# Signal tracking
_daily_social_signals = 0
_daily_reset_date: Optional[datetime] = None
MAX_DAILY_SOCIAL_SIGNALS = 8

_global_daily_signals = 0
_global_daily_reset_date: Optional[datetime] = None
MAX_GLOBAL_DAILY_SIGNALS = 8

def check_global_signal_limit() -> bool:
    global _global_daily_signals, _global_daily_reset_date
    today = datetime.now().date()
    if _global_daily_reset_date != today:
        _global_daily_signals = 0
        _global_daily_reset_date = today
    return _global_daily_signals < MAX_GLOBAL_DAILY_SIGNALS

def increment_global_signal_count():
    global _global_daily_signals, _global_daily_reset_date
    today = datetime.now().date()
    if _global_daily_reset_date != today:
        _global_daily_signals = 0
        _global_daily_reset_date = today
    _global_daily_signals += 1
    logger.info(f"📊 Global daily signals: {_global_daily_signals}/{MAX_GLOBAL_DAILY_SIGNALS}")

def get_global_signal_count() -> int:
    global _global_daily_signals, _global_daily_reset_date
    today = datetime.now().date()
    if _global_daily_reset_date != today:
        return 0
    return _global_daily_signals


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
            
            btc_closes = await self._get_btc_closes()
            btc_corr = self._calc_correlation(closes, btc_closes) if closes and btc_closes else 0.0
            
            return {
                'price': float(ticker.get('lastPrice', 0)),
                'change_24h': float(ticker.get('priceChangePercent', 0)),
                'volume_24h': float(ticker.get('quoteVolume', 0)),
                'high_24h': float(ticker.get('highPrice', 0)),
                'low_24h': float(ticker.get('lowPrice', 0)),
                'rsi': rsi,
                'volume_ratio': volume_ratio,
                'btc_correlation': btc_corr
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
                    btc_closes = await self._get_btc_closes()
                    btc_corr = self._calc_correlation(closes, btc_closes) if closes and btc_closes else 0.0
                    logger.info(f"  📱 {symbol} - Bitunix price + Binance spot RSI={rsi:.0f}")
                else:
                    btc_corr = 0.0
                    logger.info(f"  📱 {symbol} - Bitunix only, RSI unavailable (defaulting 50)")
            except Exception:
                btc_corr = 0.0
            
            return {
                'price': price,
                'change_24h': change_24h,
                'volume_24h': volume_24h,
                'high_24h': float(ticker.get('high', 0) or 0),
                'low_24h': float(ticker.get('low', 0) or 0),
                'rsi': rsi,
                'volume_ratio': volume_ratio,
                'btc_correlation': btc_corr
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
    
    def _calc_social_strength(self, galaxy_score: float, sentiment: float,
                              social_volume: int, social_interactions: int,
                              social_dominance: float, alt_rank: int,
                              social_vol_change: float = 0, is_spike: bool = False) -> float:
        """
        Composite social strength score (0-100) using multiple LunarCrush metrics.
        Weights: Galaxy Score 25%, Sentiment 15%, Social Volume 15%, 
                 Interactions 15%, Dominance 10%, AltRank 10%, Spike Bonus 10%
        """
        gs_score = min(galaxy_score / 16, 1.0) * 25
        
        sent_score = min(max(sentiment, 0), 1.0) * 15
        
        if social_volume >= 1000:
            vol_score = 15
        elif social_volume >= 500:
            vol_score = 12
        elif social_volume >= 100:
            vol_score = 8
        elif social_volume >= 20:
            vol_score = 5
        else:
            vol_score = 2
        
        if social_interactions >= 100000:
            int_score = 15
        elif social_interactions >= 50000:
            int_score = 12
        elif social_interactions >= 10000:
            int_score = 9
        elif social_interactions >= 1000:
            int_score = 6
        else:
            int_score = 2
        
        if social_dominance >= 1.0:
            dom_score = 10
        elif social_dominance >= 0.5:
            dom_score = 8
        elif social_dominance >= 0.1:
            dom_score = 5
        else:
            dom_score = 1
        
        if alt_rank <= 10:
            rank_score = 10
        elif alt_rank <= 50:
            rank_score = 8
        elif alt_rank <= 100:
            rank_score = 6
        elif alt_rank <= 300:
            rank_score = 3
        else:
            rank_score = 1
        
        spike_score = 0
        if is_spike:
            if social_vol_change >= 200:
                spike_score = 10
            elif social_vol_change >= 100:
                spike_score = 8
            elif social_vol_change >= 50:
                spike_score = 5
            else:
                spike_score = 3
        
        total = gs_score + sent_score + vol_score + int_score + dom_score + rank_score + spike_score
        return min(total, 100)
    
    _btc_klines_cache: Optional[list] = None
    _btc_klines_time: Optional[datetime] = None
    
    async def _get_btc_closes(self) -> list:
        """Get cached BTC 15m closes for correlation calc. Cache 5 min."""
        now = datetime.utcnow()
        if self._btc_klines_cache and self._btc_klines_time and (now - self._btc_klines_time).seconds < 300:
            return self._btc_klines_cache
        try:
            url = "https://fapi.binance.com/fapi/v1/klines?symbol=BTCUSDT&interval=15m&limit=20"
            resp = await self.http_client.get(url, timeout=5)
            if resp.status_code == 200:
                klines = resp.json()
                closes = [float(k[4]) for k in klines]
                self._btc_klines_cache = closes
                self._btc_klines_time = now
                return closes
        except Exception:
            pass
        return self._btc_klines_cache or []
    
    def _calc_correlation(self, coin_closes: list, btc_closes: list) -> float:
        """Calculate Pearson correlation of returns between coin and BTC."""
        n = min(len(coin_closes), len(btc_closes))
        if n < 6:
            return 0.0
        coin_closes = coin_closes[-n:]
        btc_closes = btc_closes[-n:]
        coin_returns = [(coin_closes[i] - coin_closes[i-1]) / coin_closes[i-1] for i in range(1, n) if coin_closes[i-1] != 0]
        btc_returns = [(btc_closes[i] - btc_closes[i-1]) / btc_closes[i-1] for i in range(1, n) if btc_closes[i-1] != 0]
        m = min(len(coin_returns), len(btc_returns))
        if m < 5:
            return 0.0
        coin_returns = coin_returns[-m:]
        btc_returns = btc_returns[-m:]
        mean_c = sum(coin_returns) / m
        mean_b = sum(btc_returns) / m
        cov = sum((coin_returns[i] - mean_c) * (btc_returns[i] - mean_b) for i in range(m))
        var_c = sum((x - mean_c) ** 2 for x in coin_returns)
        var_b = sum((x - mean_b) ** 2 for x in btc_returns)
        if var_c == 0 or var_b == 0:
            return 0.0
        return cov / (var_c ** 0.5 * var_b ** 0.5)
    
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
            min_score = 10
            rsi_range = (25, 75)
            require_positive_change = True
            min_sentiment = 0.2
        elif risk_level == "MEDIUM":
            min_score = 8
            rsi_range = (20, 80)
            require_positive_change = False
            min_sentiment = 0.0
        elif risk_level == "HIGH":
            min_score = 6
            rsi_range = (15, 85)
            require_positive_change = False
            min_sentiment = 0.0
        else:  # ALL or MOMENTUM
            min_score = 4
            rsi_range = (10, 90)
            require_positive_change = False
            min_sentiment = 0.0
        
        logger.info(f"📱 SOCIAL SCANNER | Risk: {risk_level} | Min Score: {min_score}")
        
        bitunix_symbols = await self._get_bitunix_symbols()
        logger.info(f"📱 Pre-loaded {len(bitunix_symbols)} Bitunix symbols for filtering")
        
        trending = await get_trending_coins(limit=200)
        
        spikes = await get_social_spikes(min_volume_change=25.0, limit=50)
        
        seen_symbols = set()
        all_social = []
        
        for coin in spikes:
            sym = coin['symbol']
            if sym not in seen_symbols:
                seen_symbols.add(sym)
                all_social.append(coin)
        
        for coin in trending:
            sym = coin['symbol']
            if sym not in seen_symbols:
                seen_symbols.add(sym)
                all_social.append(coin)
        
        if not all_social:
            logger.warning("📱 No trending or spiking coins from social data - LunarCrush returned 0 coins")
            return None
        
        combined = []
        not_on_bitunix = 0
        for coin in all_social:
            sym = coin['symbol']
            if sym.upper() in bitunix_symbols:
                combined.append(coin)
            else:
                not_on_bitunix += 1
        
        spike_count = sum(1 for c in combined if c.get('is_social_spike'))
        logger.info(f"📱 LunarCrush: {len(all_social)} total | {len(combined)} on Bitunix ({spike_count} spikes) | {not_on_bitunix} filtered out (not tradeable)")
        
        if not combined:
            logger.warning(f"📱 None of {len(all_social)} LunarCrush coins are on Bitunix!")
            return None
        
        combined.sort(key=lambda x: x.get('galaxy_score', 0), reverse=True)
        
        rejected_reasons = {'cooldown': 0, 'signal_cooldown': 0, 'galaxy_low': 0, 'sentiment_low': 0, 'negative_change': 0, 'no_price_data': 0, 'low_volume': 0, 'btc_corr': 0, 'rsi_range': 0, 'ai_cooldown': 0, 'ai_rejected': 0}
        
        passed_filters = 0
        for coin in combined:
            symbol = coin['symbol']
            galaxy_score = coin['galaxy_score']
            sentiment = coin.get('sentiment', 0)
            social_volume = coin.get('social_volume', 0)
            social_interactions = coin.get('social_interactions', 0) or coin.get('interactions_24h', 0) or 0
            social_dominance = coin.get('social_dominance', 0) or 0
            alt_rank = coin.get('alt_rank', 9999) or 9999
            coin_name = coin.get('name', '')
            price_change = coin.get('percent_change_24h', 0)
            social_vol_change = coin.get('social_volume_change_24h', 0) or 0
            is_spike = coin.get('is_social_spike', False)
            
            if is_symbol_on_cooldown(symbol):
                logger.debug(f"  📱 {symbol} - On cooldown, skipping")
                rejected_reasons['cooldown'] += 1
                continue
            
            if is_coin_in_signalled_cooldown(symbol):
                rejected_reasons['signal_cooldown'] += 1
                continue
            
            normalized_sym = symbol.replace('USDT', '').replace('/USDT', '')
            is_major = normalized_sym in MAJOR_COINS
            
            effective_min_score = max(4, min_score - 4) if is_major else min_score
            
            if galaxy_score < effective_min_score:
                logger.debug(f"  📱 {symbol} - Galaxy {galaxy_score} < {effective_min_score}")
                rejected_reasons['galaxy_low'] += 1
                continue
            
            effective_min_sentiment = max(0.0, min_sentiment - 0.15) if is_major else min_sentiment
            if sentiment < effective_min_sentiment:
                logger.info(f"  📱 {symbol} - Sentiment {sentiment:.2f} < {effective_min_sentiment}")
                rejected_reasons['sentiment_low'] += 1
                continue
            
            if require_positive_change and price_change < 0:
                logger.info(f"  📱 {symbol} - Negative 24h change {price_change:.1f}% (need positive)")
                rejected_reasons['negative_change'] += 1
                continue
            
            passed_filters += 1
            logger.info(f"  📱 ✅ {symbol} - gs={galaxy_score} sent={sentiment:.2f} chg={price_change:.1f}% - ON BITUNIX, checking price...")
            
            price_data = await self.fetch_price_data(symbol)
            if not price_data:
                logger.info(f"  📱 {symbol} - ❌ No price data from any source")
                rejected_reasons['no_price_data'] += 1
                continue
            
            current_price = price_data['price']
            rsi = price_data['rsi']
            volume_24h = price_data['volume_24h']
            volume_ratio = price_data.get('volume_ratio', 1.0)
            btc_corr = price_data.get('btc_correlation', 0.0)
            
            min_vol = 100_000
            if volume_24h < min_vol:
                logger.info(f"  📱 {symbol} - ❌ Low volume ${volume_24h/1e6:.1f}M (need $100K+)")
                rejected_reasons['low_volume'] += 1
                continue
            
            
            if btc_corr > 0.95:
                logger.info(f"  📱 {symbol} - ❌ Moves identical to BTC ({btc_corr:.2f})")
                rejected_reasons['btc_corr'] += 1
                continue
            
            if not (rsi_range[0] <= rsi <= rsi_range[1]):
                logger.info(f"  📱 {symbol} - ❌ RSI {rsi:.0f} outside range {rsi_range}")
                rejected_reasons['rsi_range'] += 1
                continue
            
            social_strength = self._calc_social_strength(
                galaxy_score=galaxy_score,
                sentiment=sentiment,
                social_volume=social_volume,
                social_interactions=social_interactions,
                social_dominance=social_dominance,
                alt_rank=alt_rank,
                social_vol_change=social_vol_change,
                is_spike=is_spike
            )
            
            spike_tag = " 🔥SPIKE" if is_spike else ""
            major_tag = " 🏛️MAJOR" if is_major else ""
            logger.info(f"✅ SOCIAL SIGNAL: {symbol}{spike_tag}{major_tag} | Galaxy: {galaxy_score} | Strength: {social_strength:.0f}/100 | Sent: {sentiment:.2f} | RSI: {rsi:.0f} | Vol: {volume_ratio:.1f}x | BTC corr: {btc_corr:.2f}")
            
            if is_major:
                base_tp = 1.2 + (sentiment * 0.3)
                base_sl = 0.8
                logger.info(f"  🏛️ MAJOR COIN {symbol} - tight TP/SL: TP {base_tp:.1f}% SL {base_sl:.1f}%")
            elif galaxy_score >= 18:
                base_tp = 5.0 + (sentiment * 2)
                base_sl = 3.0
            elif galaxy_score >= 15:
                base_tp = 4.0 + (sentiment * 1.5)
                base_sl = 2.5
            elif galaxy_score >= 13:
                base_tp = 3.0 + (sentiment * 1.5)
                base_sl = 2.0
            elif galaxy_score >= 11:
                base_tp = 2.5 + (sentiment * 1)
                base_sl = 1.5
            elif galaxy_score >= 9:
                base_tp = 2.0 + (sentiment * 0.5)
                base_sl = 1.2
            else:
                base_tp = 1.5 + (sentiment * 0.5)
                base_sl = 1.0
            
            derivatives = await get_derivatives_summary(symbol)
            
            adj = adjust_tp_sl_from_derivatives('LONG', base_tp, base_sl, derivatives)
            tp_percent = adj['tp_pct']
            sl_percent = adj['sl_pct']
            deriv_adjustments = adj['adjustments']
            
            if deriv_adjustments:
                logger.info(f"📊 {symbol} LONG TP/SL adjusted by derivatives: TP {base_tp:.1f}%→{tp_percent:.1f}% | SL {base_sl:.1f}%→{sl_percent:.1f}%")
            
            take_profit = current_price * (1 + tp_percent / 100)
            stop_loss = current_price * (1 - sl_percent / 100)
            
            tp2 = current_price * (1 + (tp_percent * 1.5) / 100)
            tp3 = current_price * (1 + (tp_percent * 2.0) / 100)
            
            from app.services.lunarcrush import get_influencer_consensus, get_social_time_series
            influencer_data = None
            buzz_momentum = None
            try:
                influencer_data = await get_influencer_consensus(symbol)
                buzz_momentum = await get_social_time_series(symbol)
            except Exception as e:
                logger.debug(f"Influencer/time-series fetch failed for {symbol}: {e}")
            
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
                'social_strength': social_strength,
                'social_vol_change': social_vol_change,
                'is_social_spike': is_spike,
                'rsi': rsi,
                'volume_ratio': volume_ratio,
                'btc_correlation': btc_corr,
                '24h_change': price_change,
                '24h_volume': volume_24h,
                'derivatives': derivatives,
                'deriv_adjustments': deriv_adjustments,
                'influencer_consensus': influencer_data,
                'buzz_momentum': buzz_momentum,
            }
            
            if is_coin_in_ai_rejection_cooldown(symbol, 'LONG'):
                logger.info(f"⏳ Skipping AI for {symbol} LONG - in 15min rejection cooldown")
                rejected_reasons['ai_cooldown'] += 1
                continue
            
            ai_result = await ai_analyze_social_signal(signal_candidate)
            
            if not ai_result.get('approved', True):
                logger.info(f"🤖 AI REJECTED {symbol} LONG: {ai_result.get('reasoning', 'No reason')}")
                add_to_ai_rejection_cooldown(symbol, 'LONG')
                rejected_reasons['ai_rejected'] += 1
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
                '24h_volume': volume_24h,
                'derivatives': derivatives,
                'deriv_adjustments': deriv_adjustments,
                'social_strength': social_strength,
                'social_vol_change': social_vol_change,
                'is_social_spike': is_spike,
                'volume_ratio': volume_ratio,
                'btc_correlation': btc_corr,
                'influencer_consensus': influencer_data,
                'buzz_momentum': buzz_momentum,
            }
        
        active_rejections = {k: v for k, v in rejected_reasons.items() if v > 0}
        logger.info(f"📱 No social LONG signals found | {len(combined)} scanned | {passed_filters} passed filters | Rejections: {active_rejections}")
        return None
    
    async def scan_for_momentum_runners(self) -> Optional[Dict]:
        """
        Scan Binance Futures for coins with big moves RIGHT NOW.
        Catches runners that social/news scanners might miss.
        Looks for: ±3% to ±50% 24h change with $500K+ volume.
        Widened filters to catch more PIPPIN-style runners early.
        """
        global _daily_social_signals
        
        reset_daily_counters_if_needed()
        if _daily_social_signals >= MAX_DAILY_SOCIAL_SIGNALS:
            return None
        
        await self.init()
        
        try:
            url = "https://fapi.binance.com/fapi/v1/ticker/24hr"
            resp = await self.http_client.get(url, timeout=8)
            if resp.status_code != 200:
                return None
            
            tickers = resp.json()
            
            runners = []
            for t in tickers:
                sym = t.get('symbol', '')
                if not sym.endswith('USDT') or sym in ('BTCUSDT', 'ETHUSDT', 'USDCUSDT'):
                    continue
                change = float(t.get('priceChangePercent', 0))
                vol = float(t.get('quoteVolume', 0))
                
                if (change >= 3 or change <= -3) and vol >= 500_000:
                    runners.append({
                        'symbol': sym,
                        'change_24h': change,
                        'volume_24h': vol,
                        'price': float(t.get('lastPrice', 0)),
                        'high': float(t.get('highPrice', 0)),
                        'low': float(t.get('lowPrice', 0)),
                    })
            
            runners.sort(key=lambda x: abs(x['change_24h']), reverse=True)
            runners = runners[:25]
            
            early_movers = []
            try:
                for t in tickers:
                    sym = t.get('symbol', '')
                    if not sym.endswith('USDT') or sym in ('BTCUSDT', 'ETHUSDT', 'USDCUSDT'):
                        continue
                    change_24h = float(t.get('priceChangePercent', 0))
                    vol = float(t.get('quoteVolume', 0))
                    if abs(change_24h) < 3 and vol >= 300_000:
                        open_price = float(t.get('openPrice', 0))
                        last_price = float(t.get('lastPrice', 0))
                        weighted_avg = float(t.get('weightedAvgPrice', 0))
                        if open_price > 0 and weighted_avg > 0:
                            price_vs_vwap = ((last_price - weighted_avg) / weighted_avg) * 100
                            if abs(price_vs_vwap) >= 1.5:
                                already_in = any(r['symbol'] == sym for r in runners)
                                if not already_in:
                                    early_movers.append({
                                        'symbol': sym,
                                        'change_24h': change_24h,
                                        'volume_24h': vol,
                                        'price': last_price,
                                        'high': float(t.get('highPrice', 0)),
                                        'low': float(t.get('lowPrice', 0)),
                                        'is_early_mover': True,
                                        'vwap_deviation': price_vs_vwap,
                                    })
                early_movers.sort(key=lambda x: abs(x.get('vwap_deviation', 0)), reverse=True)
                early_movers = early_movers[:10]
                if early_movers:
                    logger.info(f"🔍 EARLY MOVERS: Found {len(early_movers)} coins deviating from VWAP (starting to move)")
            except Exception as em_err:
                logger.debug(f"Early mover scan error: {em_err}")
            
            all_candidates = runners + early_movers
            
            if not all_candidates:
                logger.info("🚀 MOMENTUM: No runners or early movers found")
                return None
            
            runner_count = len(runners)
            early_count = len(early_movers)
            logger.info(f"🚀 MOMENTUM SCANNER: {runner_count} runners + {early_count} early movers = {len(all_candidates)} candidates")
            
            for r in all_candidates:
                symbol = r['symbol']
                is_early = r.get('is_early_mover', False)
                
                if is_symbol_on_cooldown(symbol) or is_coin_in_signalled_cooldown(symbol):
                    continue
                
                is_available = await self.check_bitunix_availability(symbol)
                if not is_available:
                    continue
                
                price_data = await self.fetch_price_data(symbol)
                if not price_data:
                    continue
                
                rsi = price_data.get('rsi', 50)
                current_price = price_data['price']
                change = r['change_24h']
                vol = r['volume_24h']
                
                abs_change = abs(change)
                vwap_dev = r.get('vwap_deviation', 0)
                
                if is_early:
                    if vwap_dev > 0:
                        direction = 'LONG'
                        if rsi > 80:
                            continue
                    else:
                        direction = 'SHORT'
                        if rsi < 20:
                            continue
                    abs_change = max(abs(vwap_dev), abs_change)
                    logger.info(f"  🔍 EARLY MOVER {symbol} | 24h {change:+.1f}% | VWAP dev {vwap_dev:+.1f}% | RSI {rsi:.0f}")
                elif change >= 3:
                    direction = 'LONG'
                    if rsi > 85:
                        logger.info(f"  🚀 {symbol} +{change:.1f}% - RSI {rsi:.0f} extremely overbought, skip long")
                        continue
                    if change > 50:
                        logger.info(f"  🚀 {symbol} +{change:.1f}% - Already pumped too much (>50%), skip long")
                        continue
                elif change <= -3:
                    direction = 'SHORT'
                    if rsi < 15:
                        logger.info(f"  🚀 {symbol} {change:.1f}% - RSI {rsi:.0f} extremely oversold, skip short")
                        continue
                    if change < -50:
                        logger.info(f"  🚀 {symbol} {change:.1f}% - Already dumped too much (>50%), skip short")
                        continue
                else:
                    continue
                
                if abs_change >= 20:
                    base_tp = 8.0 + min(abs_change * 0.2, 12.0)
                    base_sl = 3.5
                elif abs_change >= 15:
                    base_tp = 6.0 + min(abs_change * 0.15, 8.0)
                    base_sl = 3.0
                elif abs_change >= 10:
                    base_tp = 4.0 + min(abs_change * 0.1, 4.0)
                    base_sl = 2.0
                elif abs_change >= 5:
                    base_tp = 2.5 + min(abs_change * 0.08, 3.0)
                    base_sl = 1.5
                else:
                    base_tp = 1.5 + min(abs_change * 0.06, 1.5)
                    base_sl = 1.0
                
                from app.services.coinglass import get_derivatives_summary, adjust_tp_sl_from_derivatives
                derivatives = await get_derivatives_summary(symbol)
                adj = adjust_tp_sl_from_derivatives(direction, base_tp, base_sl, derivatives)
                tp_percent = adj['tp_pct']
                sl_percent = adj['sl_pct']
                deriv_adjustments = adj['adjustments']
                
                if direction == 'LONG':
                    take_profit = current_price * (1 + tp_percent / 100)
                    stop_loss = current_price * (1 - sl_percent / 100)
                    tp2 = current_price * (1 + (tp_percent * 1.5) / 100)
                    tp3 = current_price * (1 + (tp_percent * 2.0) / 100)
                else:
                    take_profit = current_price * (1 - tp_percent / 100)
                    stop_loss = current_price * (1 + sl_percent / 100)
                    tp2 = current_price * (1 - (tp_percent * 1.5) / 100)
                    tp3 = current_price * (1 - (tp_percent * 2.0) / 100)
                
                if derivatives and derivatives.get('has_data'):
                    funding = derivatives.get('funding_rate', 0) or 0
                    if direction == 'LONG' and funding > 0.05:
                        logger.info(f"  🚀 {symbol} - Extreme positive funding {funding:.4f}% (longs paying heavily), skip long")
                        continue
                    if direction == 'SHORT' and funding < -0.05:
                        logger.info(f"  🚀 {symbol} - Extreme negative funding {funding:.4f}% (shorts paying heavily), skip short")
                        continue
                
                logger.info(f"🚀 RUNNER: {symbol} +{change:.1f}% | Vol ${vol/1e6:.1f}M | RSI {rsi:.0f} | TP {tp_percent:.1f}% SL {sl_percent:.1f}%")
                
                from app.services.lunarcrush import get_influencer_consensus, get_social_time_series, get_coin_metrics
                lunar_galaxy = 0
                lunar_sentiment = 0.5
                lunar_social_vol = 0
                lunar_interactions = 0
                lunar_dominance = 0
                lunar_alt_rank = 9999
                lunar_social_vol_change = 0
                influencer_data = None
                buzz_momentum = None
                try:
                    social_data = await get_coin_metrics(symbol)
                    if social_data:
                        lunar_galaxy = social_data.get('galaxy_score', 0) or 0
                        lunar_sentiment = social_data.get('sentiment', 0.5) or 0.5
                        lunar_social_vol = social_data.get('social_volume', 0) or 0
                        lunar_interactions = social_data.get('interactions_24h', 0) or social_data.get('social_interactions', 0) or 0
                        lunar_dominance = social_data.get('social_dominance', 0) or 0
                        lunar_alt_rank = social_data.get('alt_rank', 9999) or 9999
                        lunar_social_vol_change = social_data.get('social_volume_change_24h', 0) or 0
                        logger.info(f"  🌙 {symbol} LunarCrush: Galaxy {lunar_galaxy} | Sent {lunar_sentiment:.2f} | SocVol {lunar_social_vol}")
                    influencer_data = await get_influencer_consensus(symbol)
                    buzz_momentum = await get_social_time_series(symbol)
                except Exception as e:
                    logger.debug(f"LunarCrush fetch failed for {symbol}: {e}")
                
                social_strength = self._calc_social_strength(
                    galaxy_score=lunar_galaxy,
                    sentiment=lunar_sentiment,
                    social_volume=lunar_social_vol,
                    social_interactions=lunar_interactions,
                    social_dominance=lunar_dominance,
                    alt_rank=lunar_alt_rank,
                    social_vol_change=lunar_social_vol_change,
                    is_spike=lunar_social_vol_change > 30
                )
                
                signal_candidate = {
                    'symbol': symbol,
                    'direction': direction,
                    'entry_price': current_price,
                    'stop_loss': stop_loss,
                    'take_profit': take_profit,
                    'take_profit_1': take_profit,
                    'take_profit_2': tp2,
                    'take_profit_3': tp3,
                    'tp_percent': tp_percent,
                    'sl_percent': sl_percent,
                    'rsi': rsi,
                    '24h_change': change,
                    '24h_volume': vol,
                    'volume_ratio': price_data.get('volume_ratio', 1.0),
                    'btc_correlation': price_data.get('btc_correlation', 0.0),
                    'derivatives': derivatives,
                    'deriv_adjustments': deriv_adjustments,
                    'galaxy_score': lunar_galaxy,
                    'sentiment': lunar_sentiment,
                    'social_strength': social_strength,
                    'social_vol_change': lunar_social_vol_change,
                    'is_social_spike': lunar_social_vol_change > 30,
                    'influencer_consensus': influencer_data,
                    'buzz_momentum': buzz_momentum,
                }
                
                if is_coin_in_ai_rejection_cooldown(symbol, direction):
                    continue
                
                ai_result = await ai_analyze_social_signal(signal_candidate)
                
                if not ai_result.get('approved', True):
                    logger.info(f"🤖 AI REJECTED runner {symbol}: {ai_result.get('reasoning', '')}")
                    add_to_ai_rejection_cooldown(symbol, direction)
                    continue
                
                add_symbol_cooldown(symbol)
                _daily_social_signals += 1
                
                effective_change = max(abs(change), abs(vwap_dev)) if is_early else abs(change)
                
                return {
                    'symbol': symbol,
                    'direction': direction,
                    'entry_price': current_price,
                    'stop_loss': stop_loss,
                    'take_profit': take_profit,
                    'take_profit_1': take_profit,
                    'take_profit_2': tp2,
                    'take_profit_3': tp3,
                    'tp_percent': tp_percent,
                    'sl_percent': sl_percent,
                    'confidence': min(int(effective_change), 10),
                    'reasoning': ai_result.get('reasoning', ''),
                    'ai_confidence': ai_result.get('ai_confidence', 5),
                    'ai_recommendation': ai_result.get('recommendation', 'BUY'),
                    'trade_type': 'EARLY_MOVER' if is_early else 'MOMENTUM_RUNNER',
                    'strategy': 'EARLY_MOVER' if is_early else 'MOMENTUM_RUNNER',
                    'risk_level': 'MOMENTUM',
                    'galaxy_score': lunar_galaxy,
                    'sentiment': lunar_sentiment,
                    'social_volume': lunar_social_vol,
                    'social_interactions': lunar_interactions,
                    'social_dominance': lunar_dominance,
                    'alt_rank': lunar_alt_rank,
                    'coin_name': symbol.replace('USDT', ''),
                    'rsi': rsi,
                    '24h_change': change,
                    '24h_volume': vol,
                    'derivatives': derivatives,
                    'deriv_adjustments': deriv_adjustments,
                    'social_strength': social_strength,
                    'social_vol_change': lunar_social_vol_change,
                    'volume_ratio': price_data.get('volume_ratio', 1.0),
                    'btc_correlation': price_data.get('btc_correlation', 0.0),
                    'influencer_consensus': influencer_data,
                    'buzz_momentum': buzz_momentum,
                    'is_early_mover': is_early,
                    'vwap_deviation': vwap_dev,
                }
            
            logger.info("🚀 No momentum runners passed all checks")
            return None
            
        except Exception as e:
            logger.error(f"Momentum scanner error: {e}")
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
            min_score = 6
            max_score = 13
            rsi_range = (60, 90)
            require_negative_change = True
            max_sentiment = 0.4
            max_dump_pct = -10
        elif risk_level == "MEDIUM":
            min_score = 5
            max_score = 14
            rsi_range = (55, 90)
            require_negative_change = False
            max_sentiment = 0.5
            max_dump_pct = -15
        elif risk_level == "HIGH":
            min_score = 4
            max_score = 16
            rsi_range = (50, 95)
            require_negative_change = False
            max_sentiment = 0.6
            max_dump_pct = -25
        else:  # ALL
            min_score = 3
            max_score = 18
            rsi_range = (45, 95)
            require_negative_change = False
            max_sentiment = 0.7
            max_dump_pct = -30
        
        logger.info(f"📉 SOCIAL SHORT SCANNER | Risk: {risk_level} | Galaxy Score: {min_score}-{max_score} | Max Sentiment: {max_sentiment}")
        
        bitunix_symbols = await self._get_bitunix_symbols()
        trending = await get_trending_coins(limit=200)
        
        if not trending:
            logger.warning("📉 No trending coins for short scan")
            return None
        
        tradeable = [c for c in trending if c['symbol'].upper() in bitunix_symbols]
        logger.info(f"📉 SHORT scan: {len(trending)} trending, {len(tradeable)} on Bitunix")
        
        if not tradeable:
            return None
        
        for coin in tradeable:
            symbol = coin['symbol']
            galaxy_score = coin['galaxy_score']
            sentiment = coin.get('sentiment', 0)
            social_volume = coin.get('social_volume', 0)
            social_interactions = coin.get('social_interactions', 0) or coin.get('interactions_24h', 0) or 0
            social_dominance = coin.get('social_dominance', 0) or 0
            alt_rank = coin.get('alt_rank', 9999) or 9999
            price_change = coin.get('percent_change_24h', 0)
            social_vol_change = coin.get('social_volume_change_24h', 0) or 0
            
            if is_symbol_on_cooldown(symbol):
                continue
            
            normalized_sym = symbol.replace('USDT', '').replace('/USDT', '')
            is_major = normalized_sym in MAJOR_COINS
            
            effective_min_score = max(2, min_score - 4) if is_major else min_score
            effective_max_score = max_score + 6 if is_major else max_score
            
            if galaxy_score < effective_min_score:
                continue
            
            if galaxy_score > effective_max_score:
                logger.debug(f"  {symbol} - Galaxy Score {galaxy_score} too bullish for short (max {effective_max_score})")
                continue
            
            effective_max_sentiment = min(0.8, max_sentiment + 0.15) if is_major else max_sentiment
            if sentiment > effective_max_sentiment:
                logger.debug(f"  {symbol} - Sentiment {sentiment:.2f} too bullish for short (max {effective_max_sentiment:.2f})")
                continue
            
            if require_negative_change and price_change > 0:
                continue
            
            if price_change < max_dump_pct:
                logger.info(f"  📉 {symbol} - ❌ Already dumped {price_change:.1f}% (max {max_dump_pct}%) - chasing the dump")
                continue
            
            price_data = await self.fetch_price_data(symbol)
            if not price_data:
                continue
            
            current_price = price_data['price']
            rsi = price_data['rsi']
            volume_24h = price_data['volume_24h']
            volume_ratio = price_data.get('volume_ratio', 1.0)
            btc_corr = price_data.get('btc_correlation', 0.0)
            
            if volume_24h < 100_000:
                logger.info(f"  📉 {symbol} - ❌ Low volume ${volume_24h/1e6:.1f}M (need $100K+)")
                continue
            
            
            if btc_corr > 0.95:
                logger.info(f"  📉 {symbol} - ❌ Moves identical to BTC ({btc_corr:.2f})")
                continue
            
            if not (rsi_range[0] <= rsi <= rsi_range[1]):
                logger.debug(f"  {symbol} - RSI {rsi:.0f} not in short range {rsi_range}")
                continue
            
            social_strength = self._calc_social_strength(
                galaxy_score=galaxy_score,
                sentiment=(1.0 - sentiment),
                social_volume=social_volume,
                social_interactions=social_interactions,
                social_dominance=social_dominance,
                alt_rank=alt_rank,
                social_vol_change=social_vol_change,
                is_spike=False
            )
            
            major_tag = " 🏛️MAJOR" if is_major else ""
            logger.info(f"✅ SOCIAL SHORT: {symbol}{major_tag} | Galaxy: {galaxy_score} | Strength: {social_strength:.0f}/100 | Sent: {sentiment:.2f} | RSI: {rsi:.0f} | Vol: {volume_ratio:.1f}x | BTC corr: {btc_corr:.2f}")
            
            bearish_strength = max(0, 1.0 - sentiment)
            
            if is_major:
                base_tp = 1.0 + (bearish_strength * 0.3)
                base_sl = 0.7
                logger.info(f"  🏛️ MAJOR COIN SHORT {symbol} - tight TP/SL: TP {base_tp:.1f}% SL {base_sl:.1f}%")
            elif galaxy_score <= 4:
                base_tp = 4.0 + (bearish_strength * 2)
                base_sl = 2.5
            elif galaxy_score <= 6:
                base_tp = 3.0 + (bearish_strength * 1.5)
                base_sl = 2.0
            elif galaxy_score <= 8:
                base_tp = 2.5 + (bearish_strength * 1)
                base_sl = 1.5
            elif galaxy_score <= 10:
                base_tp = 2.0 + (bearish_strength * 0.5)
                base_sl = 1.2
            else:
                base_tp = 1.5 + (bearish_strength * 0.5)
                base_sl = 1.0
            
            derivatives = await get_derivatives_summary(symbol)
            
            adj = adjust_tp_sl_from_derivatives('SHORT', base_tp, base_sl, derivatives)
            tp_percent = adj['tp_pct']
            sl_percent = adj['sl_pct']
            deriv_adjustments = adj['adjustments']
            
            if deriv_adjustments:
                logger.info(f"📊 {symbol} SHORT TP/SL adjusted by derivatives: TP {base_tp:.1f}%→{tp_percent:.1f}% | SL {base_sl:.1f}%→{sl_percent:.1f}%")
            
            take_profit = current_price * (1 - tp_percent / 100)
            stop_loss = current_price * (1 + sl_percent / 100)
            
            from app.services.lunarcrush import get_influencer_consensus, get_social_time_series
            influencer_data = None
            buzz_momentum = None
            try:
                influencer_data = await get_influencer_consensus(symbol)
                buzz_momentum = await get_social_time_series(symbol)
            except Exception as e:
                logger.debug(f"Influencer/time-series fetch failed for {symbol}: {e}")
            
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
                'social_strength': social_strength,
                'social_vol_change': social_vol_change,
                'is_social_spike': False,
                'rsi': rsi,
                'volume_ratio': volume_ratio,
                'btc_correlation': btc_corr,
                '24h_change': price_change,
                '24h_volume': volume_24h,
                'derivatives': derivatives,
                'deriv_adjustments': deriv_adjustments,
                'influencer_consensus': influencer_data,
                'buzz_momentum': buzz_momentum,
            }
            
            if is_coin_in_ai_rejection_cooldown(symbol, 'SHORT'):
                logger.info(f"⏳ Skipping AI for {symbol} SHORT - in 15min rejection cooldown")
                continue
            
            ai_result = await ai_analyze_social_signal(signal_candidate)
            
            if not ai_result.get('approved', True):
                logger.info(f"🤖 AI REJECTED {symbol} SHORT: {ai_result.get('reasoning', 'No reason')}")
                add_to_ai_rejection_cooldown(symbol, 'SHORT')
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
                'social_interactions': social_interactions,
                'social_dominance': social_dominance,
                'alt_rank': alt_rank,
                'coin_name': coin_name,
                'rsi': rsi,
                '24h_change': price_change,
                '24h_volume': volume_24h,
                'derivatives': derivatives,
                'deriv_adjustments': deriv_adjustments,
                'social_strength': social_strength,
                'social_vol_change': social_vol_change,
                'volume_ratio': volume_ratio,
                'btc_correlation': btc_corr,
                'influencer_consensus': influencer_data,
                'buzz_momentum': buzz_momentum,
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
    
    if not check_global_signal_limit():
        logger.info(f"📱 Global daily signal limit reached ({MAX_GLOBAL_DAILY_SIGNALS}) - skipping social scan")
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
        
        SOCIAL_TRADING_ALLOWED_IDS = {1, 6}
        
        users_with_social = db_session.query(User).join(UserPreference).filter(
            UserPreference.social_mode_enabled == True,
            User.id.in_(SOCIAL_TRADING_ALLOWED_IDS)
        ).all()
        
        if not users_with_social:
            logger.debug("📱 No authorized users with social mode enabled")
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
        
        # Check if any user has news trading enabled
        news_users = [u for u in users_with_social if u.preferences and getattr(u.preferences, 'news_trading_enabled', True)]
        
        # 0. CHECK FOR LIQUIDATION CASCADE ALERTS (throttled: every 10 min)
        try:
            from app.services.coinglass import detect_liquidation_cascade, format_cascade_alert_message
            from app.services.lunarcrush import get_social_time_series
            
            now_ts = datetime.now()
            cascade_cooldown_ok = (not hasattr(broadcast_social_signal, '_last_cascade') or 
                                   (now_ts - broadcast_social_signal._last_cascade).total_seconds() >= 600)
            
            if cascade_cooldown_ok:
                broadcast_social_signal._last_cascade = now_ts
                trending = await get_trending_coins(limit=10)
                cascade_coins = [c.get('symbol', '') for c in (trending or []) if c.get('symbol')][:5]
                
                trending_map = {c.get('symbol', ''): c for c in (trending or []) if c.get('symbol')}
                
                for cascade_symbol in cascade_coins:
                    try:
                        coin_data = trending_map.get(cascade_symbol, {})
                        coin_price_change = coin_data.get('percent_change_24h', 0) or 0
                        buzz = await get_social_time_series(cascade_symbol)
                        cascade_alert = await detect_liquidation_cascade(cascade_symbol, social_buzz=buzz, price_change_24h=coin_price_change)
                        if cascade_alert:
                            alert_msg = format_cascade_alert_message(cascade_alert)
                            sent_count = 0
                            for user in users_with_social:
                                try:
                                    await bot.send_message(user.telegram_id, alert_msg, parse_mode="HTML")
                                    sent_count += 1
                                    if sent_count % 5 == 0:
                                        await asyncio.sleep(0.5)
                                except Exception:
                                    pass
                            logger.warning(f"⚠️ Sent cascade alert for {cascade_symbol} to {sent_count} users")
                            break
                    except Exception as ce:
                        logger.debug(f"Cascade check error for {cascade_symbol}: {ce}")
        except Exception as cascade_err:
            logger.error(f"Cascade detection error: {cascade_err}")
        
        # 1. PRIORITY: Scan for MOMENTUM RUNNERS first (best signals - PIPPIN-style runners)
        signal = None
        try:
            signal = await service.scan_for_momentum_runners()
            if signal:
                logger.info(f"🚀 MOMENTUM RUNNER: {signal['symbol']} {signal.get('direction', 'LONG')} {signal.get('24h_change', 0):+.1f}%")
        except Exception as e:
            logger.error(f"Momentum scanner error: {e}")
        
        # 2. Check for BREAKING NEWS (fast-moving events)
        if not signal and news_users:
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
        elif not signal:
            logger.debug("📰 News trading disabled for all users, skipping news scan")
        
        # 3. Scan for social LONG signals
        if not signal:
            signal = await service.generate_social_signal(
                risk_level=most_common_risk,
                min_galaxy_score=min_galaxy
            )
        
        # 4. If no LONG, try SHORT signals
        if not signal:
            signal = await service.scan_for_short_signal(
                risk_level=most_common_risk,
                min_galaxy_score=min_galaxy
            )
        
        if signal:
            symbol = signal['symbol']
            
            if is_coin_in_signalled_cooldown(symbol):
                logger.info(f"🔇 {symbol} blocked by 24h signal cooldown - skipping broadcast")
                signal = None
        
        if signal:
            ai_conf_check = signal.get('ai_confidence', 0)
            if ai_conf_check is not None and ai_conf_check < 5:
                logger.info(f"🚫 {signal['symbol']} blocked - AI confidence too low ({ai_conf_check}/10, minimum 5)")
                signal = None
            elif ai_conf_check is not None and ai_conf_check < 7:
                logger.info(f"⚠️ {signal['symbol']} - AI confidence moderate ({ai_conf_check}/10), proceeding with caution")
        
        if signal:
            direction = signal.get('direction', 'LONG')
            symbol = signal.get('symbol', 'UNKNOWN')
            
            # Check for crowded trades (Long/Short ratio extreme)
            deriv_data = signal.get('derivatives', {})
            if deriv_data and deriv_data.get('has_data'):
                ls_ratio = deriv_data.get('ls_ratio_value', 1.0) # Usually 1.0 is balanced
                funding = deriv_data.get('funding_rate', 0) or 0
                
                if direction == 'LONG' and funding > 0.05:
                    logger.info(f"🚫 {symbol} blocked - Extreme positive funding {funding:.4f}% (longs paying heavily)")
                    signal = None
                elif direction == 'SHORT' and funding < -0.05:
                    logger.info(f"🚫 {symbol} blocked - Extreme negative funding {funding:.4f}% (shorts paying heavily)")
                    signal = None
                elif abs(funding) > 0.03:
                    logger.info(f"⚠️ {symbol} - Elevated funding rate {funding:.4f}%, proceeding with caution")
                
                if signal:
                    if direction == 'LONG' and ls_ratio > 2.5:
                        logger.info(f"🚫 {symbol} blocked - Trade too crowded ({ls_ratio:.2f} L/S ratio, >70% longs)")
                        signal = None
                    elif direction == 'SHORT' and ls_ratio < 0.4:
                        logger.info(f"🚫 {symbol} blocked - Trade too crowded ({ls_ratio:.2f} L/S ratio, >70% shorts)")
                        signal = None
                    elif (direction == 'LONG' and ls_ratio > 1.8) or (direction == 'SHORT' and ls_ratio < 0.55):
                        logger.info(f"⚠️ {symbol} - Somewhat crowded ({ls_ratio:.2f} L/S ratio), proceeding")
            
            if signal:
                entry = signal['entry_price']
                sl = signal['stop_loss']
                tp = signal['take_profit']
                galaxy = signal.get('galaxy_score', 0)
                sentiment = signal.get('sentiment', 0)
            
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
            social_strength = signal.get('social_strength', 0)
            social_vol_change = signal.get('social_vol_change', 0)
            is_spike = signal.get('is_social_spike', False)
            rsi_val = signal.get('rsi', 50)
            vol_ratio = signal.get('volume_ratio', 1.0)
            btc_corr = signal.get('btc_correlation', 0.0)
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
            is_momentum_runner = signal.get('trade_type') in ('MOMENTUM_RUNNER', 'EARLY_MOVER')
            news_title = signal.get('news_title', '')
            
            strength = calculate_signal_strength(signal)
            strength_line = format_signal_strength_detail(strength)
            
            signal_score = strength.get('total_score', 5) if strength else 5
            if signal_score <= 3:
                logger.info(f"🚫 {symbol} blocked - Signal strength too low ({signal_score}/10, minimum 4)")
                signal = None
        
        if signal:
            if is_momentum_runner:
                ai_reasoning = signal.get('reasoning', '')[:200] if signal.get('reasoning') else ''
                ai_rec = signal.get('ai_recommendation', '')
                ai_conf = signal.get('ai_confidence', 0)
                rec_emoji = {"STRONG BUY": "🚀", "BUY": "✅", "STRONG SELL": "🔻", "SELL": "📉", "HOLD": "⏸️", "AVOID": "🚫"}.get(ai_rec, "📊")
                
                is_early = signal.get('is_early_mover', False)
                vwap_dev = signal.get('vwap_deviation', 0)
                
                if is_early:
                    header = f"🔍 <b>EARLY MOVER {direction}</b>"
                    subtitle = f"<b>${symbol.replace('USDT', '')}</b> breaking from VWAP <b>{vwap_dev:+.1f}%</b> (24h {change_24h:+.1f}%)"
                else:
                    header = f"🚀 <b>MOMENTUM RUNNER {direction}</b>"
                    subtitle = f"<b>${symbol.replace('USDT', '')}</b> is running <b>{change_24h:+.1f}%</b> in 24h"
                
                galaxy = signal.get('galaxy_score', 0)
                sent = signal.get('sentiment', 0)
                lunar_line = ""
                if galaxy > 0:
                    lunar_line = f"\n🌙 Galaxy <b>{galaxy}</b>  ·  Sentiment <b>{sent:.0%}</b>"
                
                message = (
                    f"{header}\n\n"
                    f"{subtitle}\n\n"
                    f"{strength_line}\n\n"
                    f"💵  Entry  <code>{fmt_price(entry)}</code>\n"
                    f"{tp_lines}\n"
                    f"🛑  SL  <code>{fmt_price(sl)}</code>  <b>-{sl_pct:.1f}%</b>\n\n"
                    f"<b>📈 Market Data</b>\n"
                    f"RSI <b>{rsi_val:.0f}</b>  ·  24h <b>{change_24h:+.1f}%</b>  ·  Vol <b>{vol_display}</b>"
                    f"{lunar_line}"
                )
                
                deriv_data = signal.get('derivatives', {})
                if deriv_data and deriv_data.get('has_data'):
                    deriv_msg = format_derivatives_for_message(deriv_data)
                    if deriv_msg:
                        message += f"\n\n{deriv_msg}"
                
                if ai_reasoning:
                    message += f"\n\n{rec_emoji} <b>AI: {ai_rec}</b> (Confidence {ai_conf}/10)\n💡 <i>{ai_reasoning}</i>"
            
            elif is_news_signal:
                trigger = signal.get('trigger_reason', 'Breaking News')
                short_title = news_title[:70] + '...' if len(news_title) > 70 else news_title
                ai_reasoning = signal.get('reasoning', '')[:200] if signal.get('reasoning') else ''
                ai_rec = signal.get('ai_recommendation', '')
                ai_conf = signal.get('ai_confidence', 0)
                
                rec_emoji = {"STRONG BUY": "🚀", "BUY": "✅", "STRONG SELL": "🔻", "SELL": "📉", "HOLD": "⏸️", "AVOID": "🚫"}.get(ai_rec, "📊")
                
                news_impact = signal.get('confidence', 0) or galaxy
                base_ticker = symbol.replace('USDT', '').replace('/USDT:USDT', '')
                btc_corr_line = "" if base_ticker in ('BTC', 'BTCUSDT') else f"\n🔗 BTC Corr <b>{btc_corr:.0%}</b>"
                
                message = (
                    f"📰 <b>NEWS {direction}</b>\n\n"
                    f"<b>${symbol.replace('USDT', '')}</b>\n"
                    f"<i>{short_title}</i>\n\n"
                    f"{strength_line}\n\n"
                    f"💵  Entry  <code>{fmt_price(entry)}</code>\n"
                    f"{tp_lines}\n"
                    f"🛑  SL  <code>{fmt_price(sl)}</code>  <b>-{sl_pct:.1f}%</b>\n\n"
                    f"<b>📈 Market Data</b>\n"
                    f"RSI <b>{rsi_val:.0f}</b>  ·  24h <b>{change_24h:+.1f}%</b>  ·  Vol <b>{vol_display}</b>\n"
                    f"⚡ News Impact <b>{news_impact}/100</b>{btc_corr_line}"
                )
                
                deriv_data = signal.get('derivatives', {})
                if deriv_data and deriv_data.get('has_data'):
                    deriv_msg = format_derivatives_for_message(deriv_data)
                    if deriv_msg:
                        message += f"\n\n{deriv_msg}"
                    
                    deriv_adj_list = signal.get('deriv_adjustments', [])
                    if deriv_adj_list:
                        message += f"\n⚙️ <i>TP/SL adjusted by {len(deriv_adj_list)} derivatives factor{'s' if len(deriv_adj_list) > 1 else ''}</i>"
                
                if ai_reasoning:
                    message += f"\n\n{rec_emoji} <b>AI: {ai_rec}</b> (Confidence {ai_conf}/10)\n💡 <i>{ai_reasoning}</i>"
            else:
                sentiment_pct = int(sentiment * 100)
                ai_reasoning = signal.get('reasoning', '')
                ai_rec = signal.get('ai_recommendation', '')
                ai_conf = signal.get('ai_confidence', 0)
                social_interactions = signal.get('social_interactions', 0)
                social_dominance = signal.get('social_dominance', 0)
                alt_rank = signal.get('alt_rank', 9999)
                coin_name = signal.get('coin_name', '')
                
                rec_emoji = {"STRONG BUY": "🚀", "BUY": "✅", "STRONG SELL": "🔻", "SELL": "📉", "HOLD": "⏸️", "AVOID": "🚫"}.get(ai_rec, "📊")
                
                interactions_display = f"{social_interactions/1e6:.1f}M" if social_interactions >= 1e6 else f"{social_interactions/1e3:.1f}K" if social_interactions >= 1000 else f"{social_interactions:,}"
                
                name_display = f" ({coin_name})" if coin_name else ""
                spike_label = "🔥 SOCIAL SPIKE " if is_spike else ""
                
                strength_bar = "🟢" if social_strength >= 70 else "🟡" if social_strength >= 45 else "🟠"
                
                message = (
                    f"{dir_icon} <b>{spike_label}SOCIAL {direction}</b>\n\n"
                    f"<b>${symbol.replace('USDT', '')}</b>{name_display}\n\n"
                    f"{strength_line}\n\n"
                    f"💵  Entry  <code>{fmt_price(entry)}</code>\n"
                    f"{tp_lines}\n"
                    f"🛑  SL  <code>{fmt_price(sl)}</code>  <b>-{sl_pct:.1f}%</b>\n\n"
                    f"<b>📊 Social Intelligence (LunarCrush)</b>\n"
                    f"{strength_bar} Social Strength <b>{social_strength:.0f}/100</b>\n"
                    f"🌙 Galaxy <b>{galaxy}/16</b> {rating}  ·  💬 Sentiment <b>{sentiment_pct}%</b>\n"
                    f"🔊 Posts <b>{social_vol:,}</b>  ·  Interactions <b>{interactions_display}</b>\n"
                )
                
                if social_vol_change > 0:
                    message += f"📈 Social Buzz <b>+{social_vol_change:.0f}%</b> (24h surge)\n"
                
                if social_dominance > 0:
                    message += f"📡 Dominance <b>{social_dominance:.2f}%</b>"
                    if alt_rank < 9999:
                        message += f"  ·  AltRank <b>#{alt_rank}</b>"
                    message += "\n"
                
                influencer = signal.get('influencer_consensus')
                if influencer and isinstance(influencer, dict) and influencer.get('num_creators', 0) > 0:
                    cons = influencer.get('consensus', 'MIXED')
                    cons_icon = {"BULLISH": "🟢", "LEAN BULLISH": "🟢", "BEARISH": "🔴", "LEAN BEARISH": "🔴", "MIXED": "⚖️"}.get(cons, "⚖️")
                    total_fol = influencer.get('total_followers', 0)
                    followers_display = f"{total_fol/1e6:.1f}M" if total_fol >= 1e6 else f"{total_fol/1e3:.0f}K"
                    message += (
                        f"\n<b>👥 Influencer Intel</b>\n"
                        f"{cons_icon} Consensus <b>{cons}</b> ({influencer.get('bullish_count', 0)}🟢 {influencer.get('bearish_count', 0)}🔴 {influencer.get('neutral_count', 0)}⚪)\n"
                        f"Reach <b>{followers_display}</b> followers"
                    )
                    if influencer.get('big_accounts', 0) > 0:
                        message += f"  ·  <b>{influencer.get('big_accounts', 0)}</b> whale accounts"
                    message += "\n"
                
                buzz = signal.get('buzz_momentum')
                if buzz and isinstance(buzz, dict) and buzz.get('trend'):
                    trend_icon = {"RISING": "📈", "FALLING": "📉", "STABLE": "➡️"}.get(buzz.get('trend', ''), "➡️")
                    sent_icon = {"IMPROVING": "😀", "DECLINING": "😟", "STABLE": "😐"}.get(buzz.get('sentiment_trend', ''), "😐")
                    message += (
                        f"\n<b>📊 Buzz Momentum</b>\n"
                        f"{trend_icon} Trend <b>{buzz.get('trend', 'UNKNOWN')}</b> ({buzz.get('buzz_change_pct', 0):+.0f}%)  ·  "
                        f"{sent_icon} Sentiment <b>{buzz.get('sentiment_trend', 'UNKNOWN')}</b>\n"
                    )
                
                base_ticker_social = symbol.replace('USDT', '').replace('/USDT:USDT', '')
                btc_corr_social = "" if base_ticker_social in ('BTC', 'BTCUSDT') else f"\n🔗 BTC Corr <b>{btc_corr:.0%}</b>"
                
                message += (
                    f"\n<b>📈 Market Data</b>\n"
                    f"RSI <b>{rsi_val:.0f}</b>  ·  24h <b>{change_24h:+.1f}%</b>  ·  Vol <b>{vol_display}</b>"
                    f"{btc_corr_social}"
                )
                
                deriv_data = signal.get('derivatives', {})
                if deriv_data and deriv_data.get('has_data'):
                    deriv_msg = format_derivatives_for_message(deriv_data)
                    if deriv_msg:
                        message += f"\n\n{deriv_msg}"
                    
                    deriv_adj_list = signal.get('deriv_adjustments', [])
                    if deriv_adj_list:
                        message += f"\n⚙️ <i>TP/SL adjusted by {len(deriv_adj_list)} derivatives factor{'s' if len(deriv_adj_list) > 1 else ''}</i>"
                
                if ai_reasoning:
                    message += f"\n\n{rec_emoji} <b>AI: {ai_rec}</b> (Confidence {ai_conf}/10)\n💡 <i>{ai_reasoning}</i>"
            
            # Record signal in database FIRST (needed for trade execution)
            default_lev = 25 if is_top else 10
            is_early_mover = signal.get('trade_type') == 'EARLY_MOVER'
            if is_news_signal:
                sig_type = 'NEWS_SIGNAL'
            elif is_early_mover:
                sig_type = 'EARLY_MOVER'
            elif is_momentum_runner:
                sig_type = 'MOMENTUM_RUNNER'
            else:
                sig_type = 'SOCIAL_SIGNAL'
            ai_conf_val = signal.get('ai_confidence', 5)
            scaled_confidence = ai_conf_val * 10
            new_signal = Signal(
                symbol=symbol,
                direction=direction,
                entry_price=entry,
                stop_loss=sl,
                take_profit=tp,
                take_profit_1=tp,
                take_profit_2=signal.get('take_profit_2'),
                take_profit_3=signal.get('take_profit_3'),
                confidence=scaled_confidence,
                signal_type=sig_type,
                timeframe='15m',
                rsi=rsi_val,
                volume=volume_24h,
                reasoning=signal['reasoning']
            )
            db_session.add(new_signal)
            db_session.commit()
            db_session.refresh(new_signal)
            
            add_to_signalled_cooldown(symbol)
            increment_global_signal_count()
            
            # Send message + execute trade for each user
            for user in users_with_social:
                try:
                    prefs = user.preferences
                    
                    # Skip users who have news trading disabled for news signals
                    if is_news_signal and prefs and not getattr(prefs, 'news_trading_enabled', True):
                        logger.info(f"📰 Skipping user {user.telegram_id} - news trading disabled")
                        continue
                    
                    # Use news-specific leverage for news signals, social leverage otherwise
                    if is_news_signal and prefs:
                        if is_top:
                            user_lev = getattr(prefs, 'news_top_coin_leverage', 50) or 50
                            coin_type = "🏆"
                        else:
                            user_lev = getattr(prefs, 'news_leverage', 50) or 50
                            coin_type = "📊"
                    elif is_top:
                        user_lev = getattr(prefs, 'social_top_coin_leverage', 25) or 25 if prefs else 25
                        coin_type = "🏆"
                    else:
                        user_lev = getattr(prefs, 'social_leverage', 25) or 25 if prefs else 25
                        coin_type = "📊"
                    
                    signal_str = strength.get('total_score', 5) if strength else 5
                    if signal_str <= 3:
                        max_lev = 10
                    elif signal_str <= 5:
                        max_lev = 25
                    else:
                        max_lev = 50
                    user_lev = min(user_lev, max_lev)
                    
                    lev_line = f"\n\n{coin_type} {user_lev}x"
                    user_message = message + lev_line
                    
                    await bot.send_message(
                        user.telegram_id,
                        user_message,
                        parse_mode="HTML"
                    )
                    logger.info(f"📱 Sent social {direction} signal {symbol} to user {user.telegram_id} @ {user_lev}x")
                    
                    # AUTO-TRADE: Execute on Bitunix if auto-trading enabled AND API keys configured
                    if prefs and prefs.auto_trading_enabled and prefs.bitunix_api_key and prefs.bitunix_api_secret:
                        try:
                            from app.services.bitunix_trader import execute_bitunix_trade
                            logger.info(f"🔄 EXECUTING TRADE: {symbol} {direction} for user {user.telegram_id} (auto_trading=ON)")
                            trade_result = await execute_bitunix_trade(
                                signal=new_signal,
                                user=user,
                                db=db_session,
                                trade_type=sig_type,
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
                    elif prefs and not prefs.auto_trading_enabled:
                        logger.info(f"📱 User {user.telegram_id} - Auto-trading DISABLED, signal only")
                    else:
                        logger.info(f"📱 User {user.telegram_id} - No Bitunix API keys, signal only")
                except Exception as e:
                    logger.error(f"Failed to send/execute social signal for {user.telegram_id}: {e}")
        
        await service.close()
        
    except Exception as e:
        logger.error(f"Error in social signal broadcast: {e}")
    finally:
        _social_scanning_active = False
