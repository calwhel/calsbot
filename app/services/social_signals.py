"""
Social & News Signals Trading Mode - AI-powered trading
Completely separate from Top Gainers mode
"""
import asyncio
import logging
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
            min_score = max(14, min_galaxy_score)
            rsi_range = (40, 65)
            require_positive_change = True
            min_sentiment = 0.3
        elif risk_level == "MEDIUM":
            min_score = max(12, min_galaxy_score)
            rsi_range = (35, 70)
            require_positive_change = False
            min_sentiment = 0.1
        elif risk_level == "HIGH":
            min_score = max(10, min_galaxy_score)
            rsi_range = (30, 75)
            require_positive_change = False
            min_sentiment = 0.0
        else:  # ALL or MOMENTUM
            min_score = max(8, min_galaxy_score)
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
            
            if volume_24h < 5_000_000:
                logger.info(f"  📱 {symbol} - ❌ Low volume ${volume_24h/1e6:.1f}M (need $5M+)")
                continue
            
            if not (rsi_range[0] <= rsi <= rsi_range[1]):
                logger.info(f"  📱 {symbol} - ❌ RSI {rsi:.0f} outside range {rsi_range}")
                continue
            
            # 🎉 SIGNAL FOUND!
            logger.info(f"✅ SOCIAL SIGNAL: {symbol} | Score: {galaxy_score} | Sentiment: {sentiment:.2f} | RSI: {rsi:.0f}")
            
            # 🚀 DYNAMIC TP/SL - ALWAYS based on signal score
            # Higher score = stronger signal = bigger TP potential
            
            if galaxy_score >= 16:
                tp_percent = 18.0 + (sentiment * 12)  # 18-30%
                sl_percent = 6.0
            elif galaxy_score >= 14:
                tp_percent = 10.0 + (sentiment * 5)  # 10-15%
                sl_percent = 4.5
            elif galaxy_score >= 12:
                tp_percent = 6.0 + (sentiment * 3)  # 6-9%
                sl_percent = 3.5
            elif galaxy_score >= 10:
                tp_percent = 4.0 + (sentiment * 2)  # 4-6%
                sl_percent = 2.5
            else:
                tp_percent = 3.0 + (sentiment * 1)  # 3-4%
                sl_percent = 2.0
            
            take_profit = current_price * (1 + tp_percent / 100)
            stop_loss = current_price * (1 - sl_percent / 100)
            
            tp2 = None
            tp3 = None
            if galaxy_score >= 15 and tp_percent >= 12:
                tp2 = current_price * (1 + (tp_percent * 1.5) / 100)
                tp3 = current_price * (1 + (tp_percent * 2.0) / 100)
            
            # Add cooldown
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
                'reasoning': f"🌙 AI Social | Score: {galaxy_score} | Sentiment: {sentiment:.2f} | Social Vol: {social_volume:,}",
                'trade_type': 'SOCIAL_SIGNAL',
                'strategy': 'NEWS_MOMENTUM' if risk_level == "MOMENTUM" else 'SOCIAL_SIGNAL',
                'risk_level': risk_level,
                'galaxy_score': galaxy_score,
                'sentiment': sentiment,
                'social_volume': social_volume,
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
            min_score = 14
            rsi_range = (65, 85)
            require_negative_change = True
            max_sentiment = 0.3
        elif risk_level == "MEDIUM":
            min_score = 12
            rsi_range = (60, 85)
            require_negative_change = True
            max_sentiment = 0.4
        elif risk_level == "HIGH":
            min_score = 10
            rsi_range = (55, 90)
            require_negative_change = True
            max_sentiment = 0.5
        else:  # ALL
            min_score = 8
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
            if volume_24h < 5_000_000:
                continue
            
            # RSI filter - want overbought or topping
            if not (rsi_range[0] <= rsi <= rsi_range[1]):
                logger.debug(f"  {symbol} - RSI {rsi:.0f} not in short range {rsi_range}")
                continue
            
            # 🎉 SHORT SIGNAL FOUND!
            logger.info(f"✅ SOCIAL SHORT: {symbol} | Score: {galaxy_score} | Sentiment: {sentiment:.2f} | RSI: {rsi:.0f}")
            
            bearish_strength = max(0, 1.0 - sentiment)  # Lower sentiment = more bearish (0-1 scale)
            
            if galaxy_score >= 16:
                tp_percent = 15.0 + (bearish_strength * 10)  # 15-25%
                sl_percent = 6.0
            elif galaxy_score >= 14:
                tp_percent = 10.0 + (bearish_strength * 5)  # 10-15%
                sl_percent = 4.5
            elif galaxy_score >= 12:
                tp_percent = 6.0 + (bearish_strength * 3)  # 6-9%
                sl_percent = 3.5
            elif galaxy_score >= 10:
                tp_percent = 4.0 + (bearish_strength * 2)  # 4-6%
                sl_percent = 2.5
            else:
                tp_percent = 3.0 + (bearish_strength * 1)  # 3-4%
                sl_percent = 2.0
            
            # For SHORTS: TP is below entry, SL is above entry
            take_profit = current_price * (1 - tp_percent / 100)
            stop_loss = current_price * (1 + sl_percent / 100)
            
            # Add cooldown
            add_symbol_cooldown(symbol)
            _daily_social_signals += 1
            
            # Determine short trigger reason
            if sentiment <= -0.4:
                trigger_reason = "🔴 Strong FUD/negative sentiment detected"
            elif price_change <= -5:
                trigger_reason = "📉 Sharp price drop with social attention"
            elif rsi >= 75:
                trigger_reason = "⚠️ Overbought + negative sentiment shift"
            else:
                trigger_reason = "🌙 Bearish social signals detected"
            
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
                'reasoning': f"📉 AI Social SHORT | {trigger_reason} | Score: {galaxy_score} | Sentiment: {sentiment:.2f}",
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
            
            # Check if this is a breaking news signal
            is_news_signal = signal.get('trade_type') == 'NEWS_SIGNAL'
            news_title = signal.get('news_title', '')
            
            # Format based on direction
            if direction == 'SHORT':
                dir_emoji = "📉"
                if is_news_signal:
                    signal_title = "🚨 <b>BREAKING NEWS - SHORT</b>"
                else:
                    signal_title = "🔴 <b>SOCIAL SIGNAL - SHORT</b>"
                tp_pct = ((entry - tp) / entry) * 100
                sl_pct = ((sl - entry) / entry) * 100
                tp_display = f"🎯 Take Profit: ${tp:,.4f} (-{tp_pct:.1f}%)"
                sl_display = f"🛑 Stop Loss: ${sl:,.4f} (+{sl_pct:.1f}%)"
            else:
                dir_emoji = "📈"
                if is_news_signal:
                    signal_title = "🚨 <b>BREAKING NEWS - LONG</b>"
                else:
                    signal_title = "🟢 <b>SOCIAL SIGNAL - LONG</b>"
                tp_pct = ((tp - entry) / entry) * 100
                sl_pct = ((entry - sl) / entry) * 100
                tp_display = f"🎯 Take Profit: ${tp:,.4f} (+{tp_pct:.1f}%)"
                sl_display = f"🛑 Stop Loss: ${sl:,.4f} (-{sl_pct:.1f}%)"
            
            # Build message based on signal type
            if is_news_signal:
                trigger = signal.get('trigger_reason', 'Breaking News')
                short_title = news_title[:70] + '...' if len(news_title) > 70 else news_title
                
                dir_icon = "🟢" if direction == 'LONG' else "🔴"
                reasoning = signal.get('reasoning', '')[:200] if signal.get('reasoning') else ''
                
                message = (
                    f"{dir_icon} <b>NEWS {direction}</b>\n\n"
                    f"<b>{symbol}</b>\n"
                    f"<i>{short_title}</i>\n\n"
                    f"💵  Entry  <code>${entry:,.2f}</code>\n"
                    f"🎯  Target  <code>${tp:,.2f}</code>  <b>+{tp_pct:.1f}%</b>\n"
                    f"🛑  Stop  <code>${sl:,.2f}</code>  <b>-{sl_pct:.1f}%</b>\n\n"
                    f"⚡ Score {galaxy}  ·  {trigger}\n\n"
                    f"💡 <i>{reasoning}</i>" if reasoning else f"⚡ Score {galaxy}  ·  {trigger}"
                )
            else:
                risk_level = signal.get('risk_level', 'MEDIUM')
                social_vol = signal.get('social_volume', 0)
                rsi_val = signal.get('rsi', 50)
                
                dir_icon = "🟢" if direction == 'LONG' else "🔴"
                reasoning = signal.get('reasoning', '')[:200] if signal.get('reasoning') else ''
                
                message = (
                    f"{dir_icon} <b>SOCIAL {direction}</b>\n\n"
                    f"<b>{symbol}</b>\n\n"
                    f"💵  Entry  <code>${entry:,.2f}</code>\n"
                    f"🎯  Target  <code>${tp:,.2f}</code>  <b>+{tp_pct:.1f}%</b>\n"
                    f"🛑  Stop  <code>${sl:,.2f}</code>  <b>-{sl_pct:.1f}%</b>\n\n"
                    f"📊 Score {galaxy}  ·  RSI {rsi_val:.0f}  ·  {risk_level}\n"
                    f"💬 Sentiment {sentiment:+.2f}  ·  Vol {social_vol:,}\n\n"
                    f"💡 <i>{reasoning}</i>" if reasoning else 
                    f"📊 Score {galaxy}  ·  RSI {rsi_val:.0f}  ·  {risk_level}\n"
                    f"💬 Sentiment {sentiment:+.2f}  ·  Vol {social_vol:,}"
                )
            
            # Send to each user with their specific leverage
            for user in users_with_social:
                try:
                    # Get user-specific leverage based on coin type
                    prefs = user.preferences
                    if is_top:
                        user_lev = getattr(prefs, 'social_top_coin_leverage', 25) or 25 if prefs else 25
                        coin_type = "🏆"
                    else:
                        user_lev = getattr(prefs, 'social_leverage', 10) or 10 if prefs else 10
                        coin_type = "📊"
                    
                    # Add leverage to the message
                    lev_line = f"\n\n{coin_type} {user_lev}x"
                    user_message = message + lev_line
                    
                    await bot.send_message(
                        user.telegram_id,
                        user_message,
                        parse_mode="HTML"
                    )
                    logger.info(f"📱 Sent social {direction} signal {symbol} to user {user.telegram_id} @ {user_lev}x")
                except Exception as e:
                    logger.error(f"Failed to send social signal to {user.telegram_id}: {e}")
            
            # Record signal in database (use default leverage for record)
            default_lev = 25 if is_top else 10
            new_signal = Signal(
                user_id=users_with_social[0].id if users_with_social else None,
                symbol=symbol,
                direction=direction,
                entry_price=entry,
                stop_loss=sl,
                take_profit=tp,
                leverage=default_lev,
                confidence=galaxy,
                trade_type='SOCIAL_SIGNAL',
                reasoning=signal['reasoning']
            )
            db_session.add(new_signal)
            db_session.commit()
        
        await service.close()
        
    except Exception as e:
        logger.error(f"Error in social signal broadcast: {e}")
    finally:
        _social_scanning_active = False
