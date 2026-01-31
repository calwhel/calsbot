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
        """Fetch current price and technical data from Binance."""
        try:
            await self.init()
            
            # Get ticker data
            ticker_url = f"https://fapi.binance.com/fapi/v1/ticker/24hr?symbol={symbol}"
            resp = await self.http_client.get(ticker_url)
            
            if resp.status_code != 200:
                return None
            
            ticker = resp.json()
            
            # Get recent candles for RSI
            klines_url = f"https://fapi.binance.com/fapi/v1/klines?symbol={symbol}&interval=15m&limit=20"
            klines_resp = await self.http_client.get(klines_url)
            
            closes = []
            volumes = []
            if klines_resp.status_code == 200:
                klines = klines_resp.json()
                closes = [float(k[4]) for k in klines]
                volumes = [float(k[5]) for k in klines]
            
            # Calculate RSI
            rsi = 50
            if len(closes) >= 14:
                deltas = [closes[i] - closes[i-1] for i in range(1, len(closes))]
                gains = [d if d > 0 else 0 for d in deltas]
                losses = [-d if d < 0 else 0 for d in deltas]
                avg_gain = sum(gains[-14:]) / 14
                avg_loss = sum(losses[-14:]) / 14
                if avg_loss > 0:
                    rs = avg_gain / avg_loss
                    rsi = 100 - (100 / (1 + rs))
            
            # Calculate volume ratio
            volume_ratio = 1.0
            if len(volumes) >= 5:
                avg_vol = sum(volumes[:-1]) / len(volumes[:-1])
                volume_ratio = volumes[-1] / avg_vol if avg_vol > 0 else 1.0
            
            return {
                'price': float(ticker.get('lastPrice', 0)),
                'change_24h': float(ticker.get('priceChangePercent', 0)),
                'volume_24h': float(ticker.get('quoteVolume', 0)),
                'high_24h': float(ticker.get('highPrice', 0)),
                'low_24h': float(ticker.get('lowPrice', 0)),
                'rsi': rsi,
                'volume_ratio': volume_ratio
            }
            
        except Exception as e:
            logger.error(f"Error fetching price data for {symbol}: {e}")
            return None
    
    async def check_bitunix_availability(self, symbol: str) -> bool:
        """Check if symbol is tradeable on Bitunix."""
        try:
            await self.init()
            
            # Query Bitunix contracts
            url = "https://fapi.bitunix.com/api/v1/futures/market/list"
            resp = await self.http_client.get(url)
            
            if resp.status_code == 200:
                data = resp.json()
                contracts = data.get('data', [])
                
                # Check if symbol exists
                for contract in contracts:
                    if contract.get('symbol', '').upper() == symbol.upper():
                        return True
            
            return False
            
        except Exception as e:
            logger.debug(f"Bitunix check failed for {symbol}: {e}")
            return False
    
    async def generate_social_signal(
        self,
        risk_level: str = "MEDIUM",
        min_galaxy_score: int = 60
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
        # MOMENTUM mode: For catching big news runners with high TPs (15-30%+)
        # HIGH mode: Aggressive with decent TPs (8-15%)
        # MEDIUM mode: Balanced approach (5-10%)
        # LOW mode: Conservative, quick profits (3-5%)
        
        if risk_level == "LOW":
            min_score = max(70, min_galaxy_score)
            rsi_range = (40, 65)
            require_positive_change = True
            min_sentiment = 0.3
            base_tp = 3.0
            base_sl = 2.0
        elif risk_level == "MOMENTUM":
            # 🚀 NEWS RUNNERS MODE - Very high TPs for catching big moves
            min_score = max(80, min_galaxy_score)  # Only top scoring coins
            rsi_range = (30, 80)  # Wide RSI range for momentum
            require_positive_change = True  # Must be pumping
            min_sentiment = 0.5  # Strong bullish sentiment required
            base_tp = 15.0  # Base 15% TP - can go higher
            base_sl = 5.0   # Wider SL for volatility
        elif risk_level == "HIGH":
            min_score = max(50, min_galaxy_score)
            rsi_range = (30, 75)
            require_positive_change = False
            min_sentiment = 0.0
            base_tp = 8.0
            base_sl = 4.0
        else:  # MEDIUM
            min_score = max(60, min_galaxy_score)
            rsi_range = (35, 70)
            require_positive_change = False
            min_sentiment = 0.1
            base_tp = 5.0
            base_sl = 3.0
        
        logger.info(f"📱 SOCIAL SCANNER | Risk: {risk_level} | Min Score: {min_score}")
        
        # Get trending coins from social data
        trending = await get_trending_coins(limit=30)
        
        if not trending:
            logger.warning("📱 No trending coins from social data")
            return None
        
        logger.info(f"📱 Found {len(trending)} trending coins to analyze")
        
        for coin in trending:
            symbol = coin['symbol']
            galaxy_score = coin['galaxy_score']
            sentiment = coin.get('sentiment', 0)
            social_volume = coin.get('social_volume', 0)
            price_change = coin.get('percent_change_24h', 0)
            
            # Skip if on cooldown
            if is_symbol_on_cooldown(symbol):
                continue
            
            # Apply risk filters
            if galaxy_score < min_score:
                continue
            
            if sentiment < min_sentiment:
                logger.debug(f"  {symbol} - Sentiment {sentiment:.2f} too low")
                continue
            
            if require_positive_change and price_change < 0:
                continue
            
            # Check Bitunix availability
            is_available = await self.check_bitunix_availability(symbol)
            if not is_available:
                logger.debug(f"  {symbol} - Not on Bitunix")
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
                logger.debug(f"  {symbol} - Low volume ${volume_24h/1e6:.1f}M")
                continue
            
            # RSI filter
            if not (rsi_range[0] <= rsi <= rsi_range[1]):
                logger.debug(f"  {symbol} - RSI {rsi:.0f} outside range {rsi_range}")
                continue
            
            # 🎉 SIGNAL FOUND!
            logger.info(f"✅ SOCIAL SIGNAL: {symbol} | Score: {galaxy_score} | Sentiment: {sentiment:.2f} | RSI: {rsi:.0f}")
            
            # 🚀 DYNAMIC TP/SL based on Signal Score strength + risk level
            # Higher Score = stronger social/news momentum = can hold for bigger moves
            
            tp_percent = base_tp
            sl_percent = base_sl
            
            # Scale TP based on Signal Score strength (for MOMENTUM and HIGH modes)
            if risk_level == "MOMENTUM":
                # Score 80-85: 15% TP, 85-90: 20% TP, 90-95: 25% TP, 95+: 30% TP
                if galaxy_score >= 95:
                    tp_percent = 30.0
                    sl_percent = 8.0
                elif galaxy_score >= 90:
                    tp_percent = 25.0
                    sl_percent = 7.0
                elif galaxy_score >= 85:
                    tp_percent = 20.0
                    sl_percent = 6.0
                else:
                    tp_percent = 15.0
                    sl_percent = 5.0
                    
                # Boost TP if sentiment is extremely bullish
                if sentiment >= 0.7:
                    tp_percent *= 1.2  # 20% bonus
                    
            elif risk_level == "HIGH":
                # Scale 8-15% based on Signal Score
                score_bonus = (galaxy_score - 50) / 50  # 0 to 1 scale
                tp_percent = 8.0 + (score_bonus * 7.0)  # 8% to 15%
                sl_percent = 4.0 + (score_bonus * 2.0)  # 4% to 6%
            
            take_profit = current_price * (1 + tp_percent / 100)
            stop_loss = current_price * (1 - sl_percent / 100)
            
            # For MOMENTUM mode, add multiple TP targets
            tp2 = None
            tp3 = None
            if risk_level == "MOMENTUM" and tp_percent >= 15:
                tp2 = current_price * (1 + (tp_percent * 1.5) / 100)  # 1.5x main TP
                tp3 = current_price * (1 + (tp_percent * 2.0) / 100)  # 2x main TP (moon shot)
            
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
        
        logger.info("📱 No valid social signals found this scan")
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
        min_scores = [u.preferences.social_min_galaxy_score or 60 for u in users_with_social if u.preferences]
        min_galaxy = min(min_scores) if min_scores else 60
        
        signal = await service.generate_social_signal(
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
            
            rating = interpret_galaxy_score(galaxy)
            
            message = (
                f"🌙 <b>SOCIAL SIGNAL - LONG</b>\n"
                f"━━━━━━━━━━━━━━━━━━━━\n\n"
                f"📊 <b>{symbol}</b>\n\n"
                f"📈 Direction: LONG\n"
                f"💰 Entry: ${entry:,.4f}\n"
                f"🎯 Take Profit: ${tp:,.4f} (+{((tp-entry)/entry)*100:.1f}%)\n"
                f"🛑 Stop Loss: ${sl:,.4f} (-{((entry-sl)/entry)*100:.1f}%)\n\n"
                f"<b>📱 AI Signal Analysis:</b>\n"
                f"• Signal Score: {galaxy}/100 {rating}\n"
                f"• Sentiment: {sentiment:.2f}\n"
                f"• Social Volume: {signal.get('social_volume', 0):,}\n"
                f"• RSI: {signal.get('rsi', 50):.0f}\n\n"
                f"⚙️ Risk Level: {signal['risk_level']}\n"
                f"<i>Powered by AI Tech | Social + News</i>"
            )
            
            # Send to each user
            for user in users_with_social:
                try:
                    await bot.send_message(
                        user.telegram_id,
                        message,
                        parse_mode="HTML"
                    )
                    logger.info(f"📱 Sent social signal {symbol} to user {user.telegram_id}")
                except Exception as e:
                    logger.error(f"Failed to send social signal to {user.telegram_id}: {e}")
            
            # Record signal in database
            new_signal = Signal(
                user_id=users_with_social[0].id if users_with_social else None,
                symbol=symbol,
                direction='LONG',
                entry_price=entry,
                stop_loss=sl,
                take_profit=tp,
                leverage=10,
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
