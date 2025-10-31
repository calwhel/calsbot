import logging
import httpx
from typing import List, Dict, Optional
from datetime import datetime, timedelta
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)


class NewCoinAlertService:
    """
    Detects newly listed coins on Bitunix with high volume
    Sends alerts (not trade signals) about new opportunities
    
    Examples: COAI, ASTER, XPL - catches these early before they pump
    """
    
    def __init__(self):
        self.bitunix_api = "https://fapi.bitunix.com/api/v1/futures/market/tickers"
        self.coingecko_api = "https://api.coingecko.com/api/v3"
        self.client = None
        self.known_pairs = set()  # Track pairs we've already seen
        
    async def initialize(self):
        """Initialize HTTP client and load existing pairs"""
        self.client = httpx.AsyncClient(timeout=30.0)
        
        # Get current pairs to establish baseline
        await self._load_existing_pairs()
        
    async def _load_existing_pairs(self):
        """Load all current trading pairs as baseline"""
        try:
            response = await self.client.get(self.bitunix_api)
            if response.status_code == 200:
                data = response.json()
                if data.get('code') == 0 and data.get('data'):
                    self.known_pairs = {item['symbol'] for item in data['data']}
                    logger.info(f"Loaded {len(self.known_pairs)} existing Bitunix pairs as baseline")
        except Exception as e:
            logger.error(f"Error loading existing pairs: {e}")
    
    async def scan_for_new_listings(self) -> List[Dict]:
        """
        Scan for newly listed coins on Bitunix
        Returns list of new coins with high volume
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
            
            current_pairs = {item['symbol']: item for item in data['data']}
            current_pair_names = set(current_pairs.keys())
            
            # Find NEW pairs that weren't in our baseline
            new_pairs = current_pair_names - self.known_pairs
            
            if not new_pairs:
                logger.info("No new listings detected")
                return []
            
            logger.info(f"🆕 DETECTED {len(new_pairs)} NEW LISTINGS: {new_pairs}")
            
            # Analyze new listings
            new_coins = []
            for pair in new_pairs:
                ticker_data = current_pairs[pair]
                
                # Extract data
                symbol = pair.replace('USDT', '')
                try:
                    price = float(ticker_data.get('last', 0))
                    volume_usdt = float(ticker_data.get('quoteVol', 0))
                    change_24h = float(ticker_data.get('priceChangePercent', 0))
                except (ValueError, TypeError):
                    continue
                
                # Only alert if significant volume (new listings typically have high volume)
                if volume_usdt < 100000:  # Minimum $100K volume
                    continue
                
                # Get coin info from CoinGecko (description, why it's pumping)
                coin_info = await self._get_coin_info(symbol)
                
                new_coin = {
                    'symbol': symbol,
                    'pair': pair,
                    'price': price,
                    'volume_24h': volume_usdt,
                    'change_24h': change_24h,
                    'description': coin_info.get('description', 'New listing on Bitunix'),
                    'categories': coin_info.get('categories', []),
                    'listed_at': datetime.utcnow(),
                    'pump_reason': self._analyze_pump_reason(change_24h, volume_usdt, coin_info)
                }
                
                new_coins.append(new_coin)
                
                # Add to known pairs
                self.known_pairs.add(pair)
            
            return new_coins
            
        except Exception as e:
            logger.error(f"Error scanning for new listings: {e}")
            return []
    
    async def _get_coin_info(self, symbol: str) -> Dict:
        """
        Get coin information from CoinGecko
        Returns description, categories, etc.
        """
        try:
            # Search for coin on CoinGecko
            search_url = f"{self.coingecko_api}/search?query={symbol}"
            response = await self.client.get(search_url)
            
            if response.status_code != 200:
                return {}
            
            search_data = response.json()
            coins = search_data.get('coins', [])
            
            if not coins:
                return {}
            
            # Get first match (usually correct)
            coin_id = coins[0].get('id')
            
            # Get detailed info
            detail_url = f"{self.coingecko_api}/coins/{coin_id}"
            detail_response = await self.client.get(detail_url)
            
            if detail_response.status_code != 200:
                return {}
            
            coin_data = detail_response.json()
            
            # Extract description (first 200 chars)
            description = coin_data.get('description', {}).get('en', '')
            if description:
                description = description[:200] + '...' if len(description) > 200 else description
            
            return {
                'description': description or f'New {symbol} listing on Bitunix',
                'categories': coin_data.get('categories', [])[:3],  # Top 3 categories
                'homepage': coin_data.get('links', {}).get('homepage', [])[0] if coin_data.get('links', {}).get('homepage') else None
            }
            
        except Exception as e:
            logger.warning(f"Could not fetch CoinGecko info for {symbol}: {e}")
            return {
                'description': f'New {symbol} listing detected on Bitunix',
                'categories': []
            }
    
    def _analyze_pump_reason(self, change_24h: float, volume_usdt: float, coin_info: Dict) -> str:
        """
        Analyze why the coin is pumping
        Returns human-readable reason
        """
        reasons = []
        
        # New listing hype
        reasons.append("🆕 New listing")
        
        # Volume analysis
        if volume_usdt > 10_000_000:
            reasons.append("🔥 Massive volume ($10M+)")
        elif volume_usdt > 1_000_000:
            reasons.append("📈 High volume ($1M+)")
        
        # Price action
        if change_24h > 100:
            reasons.append(f"🚀 Parabolic pump (+{change_24h:.0f}%)")
        elif change_24h > 50:
            reasons.append(f"⚡ Strong pump (+{change_24h:.0f}%)")
        elif change_24h > 20:
            reasons.append(f"📊 Pumping (+{change_24h:.0f}%)")
        elif change_24h < -20:
            reasons.append(f"📉 Dumping ({change_24h:.0f}%)")
        
        # Category-based insights
        categories = coin_info.get('categories', [])
        if 'Meme' in categories or 'Memes' in categories:
            reasons.append("🐸 Meme coin hype")
        if 'AI' in ' '.join(categories) or 'Artificial Intelligence' in categories:
            reasons.append("🤖 AI narrative")
        if 'DeFi' in categories:
            reasons.append("💰 DeFi sector")
        if 'Gaming' in categories or 'GameFi' in categories:
            reasons.append("🎮 Gaming sector")
        
        return " | ".join(reasons)
    
    async def close(self):
        """Close HTTP client"""
        if self.client:
            await self.client.aclose()


async def scan_and_broadcast_new_coins(bot, db_session: Session):
    """
    Scan for new coin listings and broadcast alerts to users
    Called periodically by scheduler (every 5 minutes)
    """
    from app.models import User, UserPreference
    
    try:
        service = NewCoinAlertService()
        await service.initialize()
        
        # Scan for new listings
        new_coins = await service.scan_for_new_listings()
        
        if not new_coins:
            logger.info("No new high-volume listings to alert")
            await service.close()
            return
        
        logger.info(f"Found {len(new_coins)} new coins to alert users about")
        
        # Get users who want new coin alerts (you can add a preference for this)
        users = db_session.query(User).join(UserPreference).filter(
            User.is_subscribed == True  # Only premium users get new coin alerts
        ).all()
        
        if not users:
            logger.info("No users to send alerts to")
            await service.close()
            return
        
        # Send alerts to users
        for coin in new_coins:
            alert_message = f"""
🆕 <b>NEW COIN ALERT</b>

<b>💎 {coin['symbol']}/USDT</b> just listed on Bitunix!

<b>📊 Current Stats:</b>
├ Price: ${coin['price']:.6f}
├ 24h Volume: ${coin['volume_24h']:,.0f}
└ 24h Change: {coin['change_24h']:+.1f}%

<b>🔍 About {coin['symbol']}:</b>
{coin['description']}

<b>💡 Why it's moving:</b>
{coin['pump_reason']}

{f"<b>🏷️ Categories:</b> {', '.join(coin['categories'])}" if coin['categories'] else ""}

<i>⚠️ This is an informational alert, not a trade signal. DYOR!</i>
"""
            
            for user in users:
                try:
                    await bot.send_message(
                        user.telegram_id,
                        alert_message,
                        parse_mode="HTML"
                    )
                    logger.info(f"Sent new coin alert for {coin['symbol']} to user {user.telegram_id}")
                except Exception as e:
                    logger.error(f"Error sending alert to user {user.telegram_id}: {e}")
        
        await service.close()
        
    except Exception as e:
        logger.error(f"Error in new coin alert service: {e}", exc_info=True)
