"""
Global OpenAI Rate Limiter

Coordinates all OpenAI API calls across the application to prevent 429 errors.
Uses an async semaphore with minimum delay between requests.
"""

import asyncio
import time
import logging
from typing import Optional
from functools import wraps

logger = logging.getLogger(__name__)

class OpenAIRateLimiter:
    """
    Global rate limiter for OpenAI API calls.
    
    Enforces:
    - Max 1 concurrent request at a time (semaphore)
    - Minimum 2 second gap between requests
    - Tracks calls per minute for observability
    """
    
    _instance: Optional['OpenAIRateLimiter'] = None
    _lock = asyncio.Lock()
    
    def __init__(self):
        self._semaphore = asyncio.Semaphore(1)  # Only 1 concurrent request
        self._last_request_time = 0.0
        self._min_delay = 2.0  # Minimum seconds between requests
        self._calls_this_minute = 0
        self._minute_start = time.time()
        self._total_calls = 0
        self._rate_limited_calls = 0
    
    @classmethod
    async def get_instance(cls) -> 'OpenAIRateLimiter':
        """Get or create the singleton instance."""
        if cls._instance is None:
            async with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
                    logger.info("🔒 OpenAI Rate Limiter initialized")
        return cls._instance
    
    async def acquire(self, feature: str = "unknown") -> None:
        """
        Acquire permission to make an OpenAI request.
        Blocks until it's safe to proceed.
        """
        await self._semaphore.acquire()
        
        try:
            # Enforce minimum delay between requests
            now = time.time()
            elapsed = now - self._last_request_time
            if elapsed < self._min_delay:
                wait_time = self._min_delay - elapsed
                logger.debug(f"⏳ Rate limiter: waiting {wait_time:.1f}s before {feature} request")
                await asyncio.sleep(wait_time)
            
            # Update tracking
            self._last_request_time = time.time()
            self._total_calls += 1
            
            # Track per-minute stats
            if time.time() - self._minute_start > 60:
                if self._calls_this_minute > 0:
                    logger.info(f"📊 OpenAI calls last minute: {self._calls_this_minute}")
                self._calls_this_minute = 0
                self._minute_start = time.time()
            self._calls_this_minute += 1
            
            logger.debug(f"✅ Rate limiter: allowing {feature} request (#{self._total_calls})")
            
        except Exception as e:
            self._semaphore.release()
            raise e
    
    def release(self) -> None:
        """Release the semaphore after request completes."""
        self._semaphore.release()
    
    async def __aenter__(self):
        await self.acquire()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self.release()
        return False
    
    def record_rate_limit(self) -> None:
        """Record that we hit a rate limit."""
        self._rate_limited_calls += 1
        # Increase delay temporarily when we hit limits
        self._min_delay = min(self._min_delay + 1.0, 10.0)
        logger.warning(f"⚠️ Rate limit hit! Increasing delay to {self._min_delay}s")
    
    def get_stats(self) -> dict:
        """Get rate limiter statistics."""
        return {
            'total_calls': self._total_calls,
            'rate_limited_calls': self._rate_limited_calls,
            'calls_this_minute': self._calls_this_minute,
            'current_delay': self._min_delay
        }


# Global instance getter
async def get_rate_limiter() -> OpenAIRateLimiter:
    """Get the global OpenAI rate limiter instance."""
    return await OpenAIRateLimiter.get_instance()


# Convenience decorator for functions that call OpenAI
def rate_limited(feature: str = "unknown"):
    """
    Decorator that automatically acquires/releases the rate limiter.
    
    Usage:
        @rate_limited("signal_validation")
        async def my_openai_function():
            ...
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            limiter = await get_rate_limiter()
            await limiter.acquire(feature)
            try:
                return await func(*args, **kwargs)
            finally:
                limiter.release()
        return wrapper
    return decorator
