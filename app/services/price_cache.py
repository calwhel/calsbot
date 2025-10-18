import asyncio
import ccxt.async_support as ccxt
import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, Tuple

logger = logging.getLogger(__name__)


class PriceCache:
    """
    Global price cache to reduce API calls to exchanges
    Caches prices for 30 seconds to avoid hitting rate limits with multiple users
    Thread-safe with per-symbol locking to prevent race conditions
    """
    
    def __init__(self, cache_duration_seconds: int = 30):
        self.cache: Dict[str, Tuple[float, datetime]] = {}  # {symbol: (price, timestamp)}
        self.cache_duration = timedelta(seconds=cache_duration_seconds)
        self.exchange = None
        self.locks: Dict[str, asyncio.Lock] = {}  # Per-symbol locks for concurrent safety
        self.global_lock = asyncio.Lock()  # For lock dictionary access
        
    async def get_price(self, symbol: str, exchange_name: str = 'kucoin') -> Optional[float]:
        """
        Get price from cache if available and fresh, otherwise fetch from exchange
        Thread-safe with per-symbol locking to prevent concurrent fetches
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTC/USDT')
            exchange_name: Exchange to use (default: 'kucoin' - works in UK)
            
        Returns:
            Current price or None if error (does NOT return stale data on failure)
        """
        now = datetime.utcnow()
        
        # Quick check for fresh cache (no lock needed for read)
        if symbol in self.cache:
            cached_price, cached_time = self.cache[symbol]
            age = now - cached_time
            
            if age < self.cache_duration:
                logger.debug(f"Cache HIT for {symbol}: ${cached_price:.4f} (age: {age.total_seconds():.1f}s)")
                return cached_price
        
        # Get or create lock for this symbol
        async with self.global_lock:
            if symbol not in self.locks:
                self.locks[symbol] = asyncio.Lock()
            symbol_lock = self.locks[symbol]
        
        # Acquire symbol-specific lock to prevent concurrent fetches
        async with symbol_lock:
            # Double-check cache after acquiring lock (another coroutine might have fetched it)
            if symbol in self.cache:
                cached_price, cached_time = self.cache[symbol]
                age = now - cached_time
                
                if age < self.cache_duration:
                    logger.debug(f"Cache HIT (after lock) for {symbol}: ${cached_price:.4f}")
                    return cached_price
            
            # Cache miss or stale - fetch fresh price
            try:
                # Initialize exchange if not already done
                if self.exchange is None:
                    if exchange_name == 'kucoin':
                        self.exchange = ccxt.kucoin()
                    elif exchange_name == 'binance':
                        self.exchange = ccxt.binance()
                    elif exchange_name == 'okx':
                        self.exchange = ccxt.okx()
                    else:
                        self.exchange = ccxt.kucoin()  # Default to KuCoin
                
                # Fetch ticker
                ticker = await self.exchange.fetch_ticker(symbol)
                price = ticker['last']
                
                # Update cache
                self.cache[symbol] = (price, datetime.utcnow())
                logger.debug(f"Cache MISS - Fetched fresh price for {symbol}: ${price:.4f}")
                
                return price
                
            except Exception as e:
                logger.error(f"Error fetching price for {symbol}: {e}")
                
                # CRITICAL FIX: Do NOT return stale data on failure
                # This prevents incorrect TP/SL triggers with outdated prices
                # Better to skip the update than use wrong data
                return None
    
    async def get_multiple_prices(self, symbols: list[str], exchange_name: str = 'kucoin') -> Dict[str, float]:
        """
        Get multiple prices efficiently (batch fetch with caching)
        
        Args:
            symbols: List of trading pairs
            exchange_name: Exchange to use
            
        Returns:
            Dictionary of {symbol: price}
        """
        prices = {}
        
        for symbol in symbols:
            price = await self.get_price(symbol, exchange_name)
            if price:
                prices[symbol] = price
        
        return prices
    
    def clear_cache(self):
        """Clear all cached prices"""
        self.cache.clear()
        logger.info("Price cache cleared")
    
    def get_cache_stats(self) -> Dict:
        """Get cache statistics"""
        now = datetime.utcnow()
        fresh_count = 0
        stale_count = 0
        
        for symbol, (price, timestamp) in self.cache.items():
            age = now - timestamp
            if age < self.cache_duration:
                fresh_count += 1
            else:
                stale_count += 1
        
        return {
            'total_cached': len(self.cache),
            'fresh': fresh_count,
            'stale': stale_count,
            'cache_duration_seconds': self.cache_duration.total_seconds()
        }
    
    async def close(self):
        """Close exchange connection"""
        if self.exchange:
            await self.exchange.close()


# Global singleton instance
_global_price_cache = PriceCache(cache_duration_seconds=30)


async def get_cached_price(symbol: str, exchange_name: str = 'kucoin') -> Optional[float]:
    """
    Global function to get cached price
    This is the main entry point for the rest of the application
    """
    return await _global_price_cache.get_price(symbol, exchange_name)


async def get_multiple_cached_prices(symbols: list[str], exchange_name: str = 'kucoin') -> Dict[str, float]:
    """Get multiple cached prices at once"""
    return await _global_price_cache.get_multiple_prices(symbols, exchange_name)


def get_cache_stats() -> Dict:
    """Get cache statistics"""
    return _global_price_cache.get_cache_stats()
