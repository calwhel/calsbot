import asyncio
import logging
from datetime import datetime, timedelta
from typing import Optional
import os
import signal

logger = logging.getLogger(__name__)


class HealthMonitor:
    """
    Monitors bot health and automatically restarts if frozen or unresponsive
    """
    
    def __init__(self):
        self.last_heartbeat: Optional[datetime] = None
        self.last_message_processed: Optional[datetime] = None
        self.check_interval_seconds = 60
        self.heartbeat_timeout_seconds = 180
        self.is_monitoring = False
        
    def update_heartbeat(self):
        """Update the heartbeat timestamp"""
        self.last_heartbeat = datetime.utcnow()
        
    def update_message_timestamp(self):
        """Update when last message was processed"""
        self.last_message_processed = datetime.utcnow()
        
    async def check_health(self) -> bool:
        """
        Check if bot is healthy
        
        Returns:
            True if healthy, False if frozen/unresponsive
        """
        now = datetime.utcnow()
        
        if self.last_heartbeat is None:
            self.last_heartbeat = now
            return True
            
        time_since_heartbeat = (now - self.last_heartbeat).total_seconds()
        
        if time_since_heartbeat > self.heartbeat_timeout_seconds:
            logger.error(f"🚨 Bot appears FROZEN! No heartbeat for {time_since_heartbeat:.0f}s")
            return False
            
        return True
    
    async def auto_recovery(self):
        """
        Attempt automatic recovery by restarting the process
        """
        logger.warning("🔄 Attempting automatic recovery...")
        
        pid = os.getpid()
        logger.info(f"Sending SIGTERM to process {pid} for clean restart...")
        
        os.kill(pid, signal.SIGTERM)
        
        await asyncio.sleep(2)
        
        os.kill(pid, signal.SIGKILL)
        
    async def start_monitoring(self):
        """
        Start health monitoring loop
        """
        if self.is_monitoring:
            logger.warning("Health monitor already running")
            return
            
        self.is_monitoring = True
        logger.info("🏥 Health monitor started")
        
        while self.is_monitoring:
            try:
                await asyncio.sleep(self.check_interval_seconds)
                
                is_healthy = await self.check_health()
                
                if not is_healthy:
                    logger.error("Health check FAILED - initiating auto-recovery")
                    await self.auto_recovery()
                else:
                    logger.debug("✅ Health check passed")
                    
            except asyncio.CancelledError:
                logger.info("Health monitor cancelled")
                break
            except Exception as e:
                logger.error(f"Error in health monitor: {e}")
                await asyncio.sleep(10)
    
    def stop_monitoring(self):
        """Stop health monitoring"""
        self.is_monitoring = False
        logger.info("Health monitor stopped")


_global_health_monitor = HealthMonitor()


def get_health_monitor() -> HealthMonitor:
    """Get global health monitor instance"""
    return _global_health_monitor


async def update_heartbeat():
    """Convenience function to update heartbeat"""
    _global_health_monitor.update_heartbeat()


async def update_message_timestamp():
    """Convenience function to update message timestamp"""
    _global_health_monitor.update_message_timestamp()
