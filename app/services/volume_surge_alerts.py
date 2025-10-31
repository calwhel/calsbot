import logging
import httpx
from typing import List, Dict, Optional
from datetime import datetime, timedelta
from collections import defaultdict

logger = logging.getLogger(__name__)


class VolumeSurgeDetector:
    """
    Detects volume surges BEFORE coins hit 25% gains
    Catches pumps early (5-15% gains) for early entry opportunities
    
    Strategy: Volume leads price - catch the pump as it starts!
    """
    
    def __init__(self):
        self.bitunix_api = "https://fapi.bitunix.com/api/v1/futures/market/tickers"
        self.client = None
        self.volume_baseline = {}  # Track average volume per coin
        self.last_alerts = {}  # Prevent spam alerts
        
    async def initialize(self):
        """Initialize HTTP client and establish volume baselines"""
        self.client = httpx.AsyncClient(timeout=30.0)
        
        # Build initial volume baseline
        await self._build_volume_baseline()
        
    async def _build_volume_baseline(self):
        """
        Establish baseline volume for all coins
        Uses current volume as starting point, then updates with rolling average
        """
        try:
            response = await self.client.get(self.bitunix_api)
            if response.status_code == 200:
                data = response.json()
                if data.get('code') == 0 and data.get('data'):
                    for item in data['data']:
                        symbol = item['symbol']
                        try:
                            volume_usdt = float(item.get('quoteVol', 0))
                            if volume_usdt > 0:
                                self.volume_baseline[symbol] = volume_usdt
                        except (ValueError, TypeError):
                            continue
                    
                    logger.info(f"Built volume baseline for {len(self.volume_baseline)} pairs")
        except Exception as e:
            logger.error(f"Error building volume baseline: {e}")
    
    async def scan_for_volume_surges(self) -> List[Dict]:
        """
        Scan for volume surges with early price movement
        Catches coins BEFORE they hit 25% threshold
        """
        try:
            response = await self.client.get(self.bitunix_api)
            
            if response.status_code != 200:
                logger.error(f"Bitunix API error: {response.status_code}")
                return []
            
            data = response.json()
            
            if data.get('code') != 0:
                logger.error(f"Bitunix API error: {data.get('msg')}")
                return []
            
            surges = []
            now = datetime.utcnow()
            
            for item in data['data']:
                symbol = item['symbol']
                
                try:
                    current_volume = float(item.get('quoteVol', 0))
                    price = float(item.get('last', 0))
                    change_24h = float(item.get('priceChangePercent', 0))
                except (ValueError, TypeError):
                    continue
                
                # Skip if no baseline or too low volume
                if symbol not in self.volume_baseline or current_volume < 50000:
                    continue
                
                baseline_volume = self.volume_baseline[symbol]
                volume_ratio = current_volume / baseline_volume if baseline_volume > 0 else 1.0
                
                # VOLUME SURGE CRITERIA:
                # 1. Volume 2x+ normal (surge detected)
                # 2. Price 5-20% up (early pump, not yet 25% threshold)
                # 3. Minimum $50K volume (real activity)
                # 4. Not alerted in last 2 hours (prevent spam)
                
                is_volume_surge = volume_ratio >= 2.0
                is_early_pump = 5.0 <= change_24h < 20.0  # Sweet spot: early but confirmed
                not_recently_alerted = (
                    symbol not in self.last_alerts or 
                    (now - self.last_alerts[symbol]).total_seconds() > 7200  # 2 hours
                )
                
                if is_volume_surge and is_early_pump and not_recently_alerted:
                    
                    # Analyze trend to confirm it's not a fake pump
                    trend_quality = await self._check_trend_quality(symbol)
                    
                    if not trend_quality['valid']:
                        logger.info(f"{symbol} volume surge detected but trend quality poor - skipping")
                        continue
                    
                    surge_data = {
                        'symbol': symbol.replace('USDT', ''),
                        'pair': symbol,
                        'price': price,
                        'change_24h': change_24h,
                        'volume_current': current_volume,
                        'volume_baseline': baseline_volume,
                        'volume_ratio': volume_ratio,
                        'trend_quality': trend_quality,
                        'detected_at': now,
                        'confidence': self._calculate_confidence(volume_ratio, change_24h, trend_quality)
                    }
                    
                    surges.append(surge_data)
                    self.last_alerts[symbol] = now
                    
                    logger.info(
                        f"⚡ VOLUME SURGE: {symbol} | "
                        f"Vol: {volume_ratio:.1f}x | "
                        f"Price: +{change_24h:.1f}% | "
                        f"Trend: {trend_quality['description']}"
                    )
                
                # Update baseline with rolling average (70% old, 30% new)
                self.volume_baseline[symbol] = (baseline_volume * 0.7) + (current_volume * 0.3)
            
            return surges
            
        except Exception as e:
            logger.error(f"Error scanning for volume surges: {e}")
            return []
    
    async def _check_trend_quality(self, symbol: str) -> Dict:
        """
        Quick trend quality check to filter fake pumps
        Returns trend strength and description
        """
        try:
            # Get 5m candles for quick trend check
            binance_symbol = symbol.replace('/', '')
            candles_url = f"https://fapi.binance.com/fapi/v1/klines?symbol={binance_symbol}&interval=5m&limit=20"
            
            response = await self.client.get(candles_url)
            if response.status_code != 200:
                return {'valid': False, 'description': 'Unable to verify'}
            
            candles = response.json()
            if len(candles) < 10:
                return {'valid': False, 'description': 'Insufficient data'}
            
            # Simple trend check: count green candles in last 10
            green_candles = 0
            for candle in candles[-10:]:
                close = float(candle[4])
                open_price = float(candle[1])
                if close > open_price:
                    green_candles += 1
            
            # Check volume trend (increasing)
            recent_volume = sum(float(c[5]) for c in candles[-5:])
            older_volume = sum(float(c[5]) for c in candles[-10:-5])
            volume_increasing = recent_volume > older_volume * 1.2
            
            # Quality assessment
            if green_candles >= 7 and volume_increasing:
                return {
                    'valid': True,
                    'description': 'Strong uptrend',
                    'strength': 'high',
                    'green_candles': green_candles
                }
            elif green_candles >= 5 and volume_increasing:
                return {
                    'valid': True,
                    'description': 'Moderate uptrend',
                    'strength': 'medium',
                    'green_candles': green_candles
                }
            else:
                return {
                    'valid': False,
                    'description': 'Weak/choppy trend',
                    'strength': 'low',
                    'green_candles': green_candles
                }
                
        except Exception as e:
            logger.warning(f"Error checking trend for {symbol}: {e}")
            return {'valid': False, 'description': 'Error checking trend'}
    
    def _calculate_confidence(self, volume_ratio: float, price_change: float, trend_quality: Dict) -> int:
        """
        Calculate confidence score (0-100) for the surge
        Higher = better opportunity
        """
        score = 0
        
        # Volume contribution (max 40 points)
        if volume_ratio >= 5.0:
            score += 40
        elif volume_ratio >= 3.0:
            score += 30
        elif volume_ratio >= 2.0:
            score += 20
        
        # Price movement contribution (max 30 points)
        if 10.0 <= price_change <= 15.0:  # Sweet spot
            score += 30
        elif 8.0 <= price_change < 20.0:
            score += 20
        else:
            score += 10
        
        # Trend quality contribution (max 30 points)
        if trend_quality.get('strength') == 'high':
            score += 30
        elif trend_quality.get('strength') == 'medium':
            score += 20
        elif trend_quality.get('strength') == 'low':
            score += 10
        
        return min(score, 100)
    
    async def close(self):
        """Close HTTP client"""
        if self.client:
            await self.client.aclose()


async def scan_and_broadcast_volume_surges(bot, db_session):
    """
    Scan for volume surges and broadcast alerts
    Called every 3 minutes (faster than Top Gainers to catch early pumps)
    """
    from app.models import User, UserPreference
    
    try:
        detector = VolumeSurgeDetector()
        await detector.initialize()
        
        # Scan for surges
        surges = await detector.scan_for_volume_surges()
        
        if not surges:
            logger.debug("No volume surges detected")
            await detector.close()
            return
        
        logger.info(f"Found {len(surges)} volume surges to alert users about")
        
        # Get subscribed users (premium feature)
        users = db_session.query(User).join(UserPreference).filter(
            User.is_subscribed == True
        ).all()
        
        if not users:
            logger.info("No users to send volume surge alerts to")
            await detector.close()
            return
        
        # Send alerts
        for surge in surges:
            # Confidence emoji
            confidence = surge['confidence']
            if confidence >= 80:
                confidence_emoji = "🔥"
                confidence_text = "Very High"
            elif confidence >= 60:
                confidence_emoji = "⚡"
                confidence_text = "High"
            else:
                confidence_emoji = "💫"
                confidence_text = "Moderate"
            
            alert_message = f"""
⚡ <b>VOLUME SURGE ALERT</b>

{confidence_emoji} <b>{surge['symbol']}/USDT</b> - Early Pump Detected!

<b>📊 Current Stats:</b>
├ Price: ${surge['price']:.6f} (<b>+{surge['change_24h']:.1f}%</b>)
├ Volume Surge: <b>{surge['volume_ratio']:.1f}x normal</b>
└ 24h Volume: ${surge['volume_current']:,.0f}

<b>💡 Why This Matters:</b>
Volume is leading price - pump just starting!
Currently at +{surge['change_24h']:.1f}% (below 25% threshold)

<b>📈 Trend Analysis:</b>
├ Strength: {surge['trend_quality']['description']}
└ Confidence: {confidence_text} ({confidence}%)

<b>🎯 Opportunity:</b>
This coin hasn't hit Top Gainers threshold yet
(needs 25%+). Volume surge suggests it could pump
to 25-50%+ soon.

<b>⏰ Timing:</b>
Detected at: {surge['detected_at'].strftime('%H:%M UTC')}

<i>💎 Early bird gets the worm - volume leads price!</i>
<i>⚠️ Alert only - not a trade signal. DYOR!</i>
"""
            
            for user in users:
                try:
                    await bot.send_message(
                        user.telegram_id,
                        alert_message,
                        parse_mode="HTML"
                    )
                    logger.info(f"Sent volume surge alert for {surge['symbol']} to user {user.telegram_id}")
                except Exception as e:
                    logger.error(f"Error sending volume surge alert to user {user.telegram_id}: {e}")
        
        await detector.close()
        
    except Exception as e:
        logger.error(f"Error in volume surge detector: {e}", exc_info=True)
