"""
Metals Trading Signals - Gold (XAU) and Silver (XAG) trading on Bitunix.

Uses MarketAux news sentiment + AI analysis to generate trading signals.
Admin-only feature for now.
"""

import os
import logging
import asyncio
import ccxt.async_support as ccxt
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Tuple
from sqlalchemy.orm import Session

from app.models import User, Signal, Trade
from app.services.marketaux import get_metals_sentiment, MarketAuxClient
from app.database import SessionLocal

logger = logging.getLogger(__name__)

METALS_SYMBOLS = {
    "gold": "XAUUSDT",
    "silver": "XAGUSDT"
}

METALS_LEVERAGE = 10
METALS_TP_PERCENT = 2.0
METALS_SL_PERCENT = 1.0

_metals_scanning_enabled = False


def is_metals_scanning_enabled() -> bool:
    """Check if metals scanning is enabled."""
    return _metals_scanning_enabled


def toggle_metals_scanning() -> bool:
    """Toggle metals scanning on/off. Returns new state."""
    global _metals_scanning_enabled
    _metals_scanning_enabled = not _metals_scanning_enabled
    logger.info(f"Metals scanning {'ENABLED' if _metals_scanning_enabled else 'DISABLED'}")
    return _metals_scanning_enabled


def set_metals_scanning(enabled: bool):
    """Set metals scanning state."""
    global _metals_scanning_enabled
    _metals_scanning_enabled = enabled
    logger.info(f"Metals scanning set to {'ENABLED' if enabled else 'DISABLED'}")


class MetalsSignalService:
    """Service for generating gold/silver trading signals."""
    
    def __init__(self):
        self.exchange = None
        self.marketaux = MarketAuxClient()
    
    async def initialize(self):
        """Initialize exchange connection.
        
        Note: Using Binance for price data since Bitunix may not have 
        XAU/XAG pairs listed yet. Can switch to Bitunix when available.
        """
        try:
            self.exchange = ccxt.binance({
                'enableRateLimit': True,
                'options': {'defaultType': 'future'}
            })
            logger.info("Metals signal service initialized (using Binance for price data)")
        except Exception as e:
            logger.error(f"Failed to initialize metals service: {e}")
    
    async def close(self):
        """Close exchange connection."""
        if self.exchange:
            await self.exchange.close()
    
    async def get_metal_price(self, metal: str) -> Optional[float]:
        """Get current price for gold or silver.
        
        Args:
            metal: "gold" or "silver"
        """
        symbol = METALS_SYMBOLS.get(metal)
        if not symbol:
            return None
        
        try:
            ticker = await self.exchange.fetch_ticker(symbol)
            return ticker.get('last')
        except Exception as e:
            logger.warning(f"Could not fetch {metal} price from Binance: {e}")
            return None
    
    async def get_metal_ohlcv(self, metal: str, timeframe: str = '1h', limit: int = 50) -> Optional[List]:
        """Get OHLCV data for technical analysis."""
        symbol = METALS_SYMBOLS.get(metal)
        if not symbol:
            return None
        
        try:
            ohlcv = await self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            return ohlcv
        except Exception as e:
            logger.warning(f"Could not fetch {metal} OHLCV: {e}")
            return None
    
    def calculate_rsi(self, closes: List[float], period: int = 14) -> float:
        """Calculate RSI from close prices."""
        if len(closes) < period + 1:
            return 50.0
        
        deltas = [closes[i] - closes[i-1] for i in range(1, len(closes))]
        gains = [d if d > 0 else 0 for d in deltas[-period:]]
        losses = [-d if d < 0 else 0 for d in deltas[-period:]]
        
        avg_gain = sum(gains) / period
        avg_loss = sum(losses) / period
        
        if avg_loss == 0:
            return 100.0
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return round(rsi, 2)
    
    def calculate_ema(self, prices: List[float], period: int) -> float:
        """Calculate EMA."""
        if len(prices) < period:
            return prices[-1] if prices else 0
        
        multiplier = 2 / (period + 1)
        ema = sum(prices[:period]) / period
        
        for price in prices[period:]:
            ema = (price - ema) * multiplier + ema
        
        return round(ema, 4)
    
    async def analyze_metal(self, metal: str, sentiment_data: Dict) -> Optional[Dict]:
        """Analyze a metal and generate signal if conditions met.
        
        Args:
            metal: "gold" or "silver"
            sentiment_data: From MarketAux
            
        Returns:
            Signal data dict or None
        """
        price = await self.get_metal_price(metal)
        if not price:
            logger.warning(f"Could not get {metal} price")
            return None
        
        ohlcv = await self.get_metal_ohlcv(metal, '1h', 50)
        if not ohlcv or len(ohlcv) < 20:
            logger.warning(f"Insufficient OHLCV data for {metal}")
            return None
        
        closes = [candle[4] for candle in ohlcv]
        rsi = self.calculate_rsi(closes)
        ema9 = self.calculate_ema(closes, 9)
        ema21 = self.calculate_ema(closes, 21)
        
        metal_sentiment = sentiment_data.get(metal, {})
        sentiment_score = metal_sentiment.get("score", 0)
        sentiment_label = metal_sentiment.get("sentiment", "neutral")
        news_recommendation = sentiment_data.get("recommendation", "SKIP")
        
        direction = None
        confidence = 0
        reasoning = []
        
        if sentiment_label == "bullish" and sentiment_score > 0.2:
            reasoning.append(f"Bullish news sentiment ({sentiment_score:.2f})")
            confidence += 30
            
            if rsi < 70:
                reasoning.append(f"RSI not overbought ({rsi})")
                confidence += 20
                
                if ema9 > ema21:
                    reasoning.append("EMA9 > EMA21 (uptrend)")
                    confidence += 20
                    direction = "LONG"
                elif rsi < 40:
                    reasoning.append(f"RSI oversold ({rsi}) - potential bounce")
                    confidence += 15
                    direction = "LONG"
        
        elif sentiment_label == "bearish" and sentiment_score < -0.2:
            reasoning.append(f"Bearish news sentiment ({sentiment_score:.2f})")
            confidence += 30
            
            if rsi > 30:
                reasoning.append(f"RSI not oversold ({rsi})")
                confidence += 20
                
                if ema9 < ema21:
                    reasoning.append("EMA9 < EMA21 (downtrend)")
                    confidence += 20
                    direction = "SHORT"
                elif rsi > 60:
                    reasoning.append(f"RSI overbought ({rsi}) - potential drop")
                    confidence += 15
                    direction = "SHORT"
        
        if not direction or confidence < 50:
            logger.info(f"{metal.upper()}: No signal - confidence {confidence}%, direction: {direction}")
            return None
        
        if direction == "LONG":
            stop_loss = price * (1 - METALS_SL_PERCENT / 100)
            take_profit = price * (1 + METALS_TP_PERCENT / 100)
        else:
            stop_loss = price * (1 + METALS_SL_PERCENT / 100)
            take_profit = price * (1 - METALS_TP_PERCENT / 100)
        
        symbol = METALS_SYMBOLS[metal]
        
        return {
            "symbol": symbol,
            "metal": metal.upper(),
            "direction": direction,
            "entry_price": price,
            "stop_loss": round(stop_loss, 2),
            "take_profit_1": round(take_profit, 2),
            "leverage": METALS_LEVERAGE,
            "confidence": confidence,
            "reasoning": " | ".join(reasoning),
            "rsi": rsi,
            "ema9": ema9,
            "ema21": ema21,
            "sentiment_score": sentiment_score,
            "headlines": sentiment_data.get("headlines", [])[:3],
            "signal_type": "METALS_NEWS"
        }
    
    async def scan_for_signals(self, force: bool = False) -> List[Dict]:
        """Scan gold and silver for trading signals.
        
        Args:
            force: If True, scan even if scanning is disabled (for manual scans)
        
        Returns:
            List of signal dicts
        """
        if not force and not is_metals_scanning_enabled():
            logger.debug("Metals scanning is disabled")
            return []
        
        logger.info("🥇 Scanning metals (Gold/Silver) for signals...")
        
        sentiment = await get_metals_sentiment()
        
        if not sentiment or not sentiment.get('gold') or not sentiment.get('silver'):
            logger.warning("Failed to get metals sentiment from MarketAux")
            return []
        
        logger.info(f"MarketAux sentiment: Gold={sentiment['gold']['sentiment']}, Silver={sentiment['silver']['sentiment']}")
        logger.info(f"Combined score: {sentiment.get('combined_score', 0)}, Recommendation: {sentiment['recommendation']}")
        
        if sentiment["recommendation"] == "SKIP":
            logger.info(f"Skipping metals scan: {sentiment['reason']}")
            return []
        
        signals = []
        
        for metal in ["gold", "silver"]:
            try:
                signal = await self.analyze_metal(metal, sentiment)
                if signal:
                    signals.append(signal)
                    logger.info(f"✅ {metal.upper()} signal: {signal['direction']} @ {signal['entry_price']}")
            except Exception as e:
                logger.error(f"Error analyzing {metal}: {e}")
        
        return signals


async def run_metals_scanner(bot=None):
    """Run the metals scanner once.
    
    Called periodically by the scheduler.
    """
    if not is_metals_scanning_enabled():
        return
    
    service = MetalsSignalService()
    await service.initialize()
    
    try:
        signals = await service.scan_for_signals()
        
        if signals and bot:
            db = SessionLocal()
            try:
                admins = db.query(User).filter(User.is_admin == True).all()
                
                for signal in signals:
                    message = format_metals_signal(signal)
                    
                    for admin in admins:
                        try:
                            await bot.send_message(
                                chat_id=int(admin.telegram_id),
                                text=message,
                                parse_mode="HTML"
                            )
                        except Exception as e:
                            logger.error(f"Failed to send metals signal to admin {admin.id}: {e}")
            finally:
                db.close()
    finally:
        await service.close()


def format_metals_signal(signal: Dict) -> str:
    """Format a metals signal for Telegram."""
    direction_emoji = "🟢" if signal["direction"] == "LONG" else "🔴"
    metal_emoji = "🥇" if signal["metal"] == "GOLD" else "🥈"
    
    headlines_text = ""
    if signal.get("headlines"):
        headlines_text = "\n\n📰 <b>Key Headlines:</b>\n" + "\n".join([f"• {h[:80]}..." for h in signal["headlines"][:3]])
    
    return f"""
{metal_emoji} <b>METALS SIGNAL - {signal['metal']}</b> {metal_emoji}

{direction_emoji} <b>{signal['direction']}</b> {signal['symbol']}

💰 Entry: <code>${signal['entry_price']:,.2f}</code>
🎯 Take Profit: <code>${signal['take_profit_1']:,.2f}</code>
🛑 Stop Loss: <code>${signal['stop_loss']:,.2f}</code>
⚡ Leverage: {signal['leverage']}x

📊 <b>Technical:</b>
• RSI: {signal['rsi']}
• EMA9/21: {signal['ema9']:.2f} / {signal['ema21']:.2f}
• News Sentiment: {signal['sentiment_score']:.2f}

📝 <b>Reasoning:</b>
{signal['reasoning']}{headlines_text}

⚠️ <i>Admin-only test signal</i>
"""
