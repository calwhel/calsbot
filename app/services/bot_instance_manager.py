"""
Bot Instance Manager - Prevents and manages multiple bot instance conflicts
"""
import asyncio
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from aiogram import Bot
from app.config import settings

logger = logging.getLogger(__name__)

LOCK_FILE = "/tmp/telegram_bot.lock"
INSTANCE_ID = str(os.getpid())  # Use process ID as instance identifier


class BotInstanceManager:
    """Manages single bot instance and prevents conflicts"""
    
    def __init__(self, bot: Bot):
        self.bot = bot
        self.is_locked = False
        self.monitor_task = None
        
    async def acquire_lock(self) -> bool:
        """
        Acquire single instance lock
        Returns True if lock acquired, False if another instance running
        """
        try:
            lock_path = Path(LOCK_FILE)
            
            # Check if lock file exists
            if lock_path.exists():
                # Read existing lock info
                with open(LOCK_FILE, 'r') as f:
                    lock_data = f.read().strip().split('|')
                    if len(lock_data) >= 2:
                        old_pid = lock_data[0]
                        lock_time = lock_data[1]
                        
                        # Check if process is still running
                        try:
                            os.kill(int(old_pid), 0)  # Signal 0 checks if process exists
                            logger.warning(f"Another bot instance is running (PID: {old_pid}, started: {lock_time})")
                            return False
                        except OSError:
                            # Process doesn't exist, remove stale lock
                            logger.info(f"Removing stale lock file from PID {old_pid}")
                            lock_path.unlink()
            
            # Create lock file
            with open(LOCK_FILE, 'w') as f:
                f.write(f"{INSTANCE_ID}|{datetime.utcnow().isoformat()}")
            
            self.is_locked = True
            logger.info(f"✅ Bot instance lock acquired (PID: {INSTANCE_ID})")
            return True
            
        except Exception as e:
            logger.error(f"Error acquiring instance lock: {e}")
            return False
    
    async def release_lock(self):
        """Release the instance lock"""
        try:
            if self.is_locked:
                lock_path = Path(LOCK_FILE)
                if lock_path.exists():
                    lock_path.unlink()
                    logger.info(f"🔓 Bot instance lock released (PID: {INSTANCE_ID})")
                self.is_locked = False
        except Exception as e:
            logger.error(f"Error releasing lock: {e}")
    
    async def force_stop_other_instances(self) -> bool:
        """
        Force stop other bot instances by dropping Telegram webhook
        This terminates any other connection to the bot
        """
        try:
            logger.info("🛑 Force stopping other bot instances...")
            
            # Delete webhook (if exists) - this kicks out any other instances
            await self.bot.delete_webhook(drop_pending_updates=True)
            logger.info("✅ Webhook deleted, pending updates dropped")
            
            # Wait a moment for other instances to disconnect
            await asyncio.sleep(2)
            
            # Remove any stale lock files
            lock_path = Path(LOCK_FILE)
            if lock_path.exists():
                lock_path.unlink()
                logger.info("🗑️  Removed stale lock file")
            
            return True
            
        except Exception as e:
            logger.error(f"Error force stopping instances: {e}")
            return False
    
    async def start_conflict_monitor(self):
        """Monitor for conflicts and alert if another instance starts"""
        logger.info("👀 Starting conflict monitor...")
        
        while True:
            try:
                # Check if our lock file still exists
                lock_path = Path(LOCK_FILE)
                if not lock_path.exists() and self.is_locked:
                    # Someone deleted our lock - another instance took over!
                    logger.error("⚠️ CONFLICT DETECTED: Another bot instance has taken over!")
                    self.is_locked = False
                    
                    # Try to notify admins via Telegram
                    try:
                        from app.database import SessionLocal
                        from app.models import User
                        
                        db = SessionLocal()
                        admins = db.query(User).filter(User.is_admin == True).all()
                        
                        for admin in admins:
                            try:
                                await self.bot.send_message(
                                    admin.telegram_id,
                                    "⚠️ <b>BOT INSTANCE CONFLICT DETECTED!</b>\n\n"
                                    "Another bot instance has started and taken over. "
                                    "This instance will shut down to prevent conflicts.\n\n"
                                    f"<i>Instance PID: {INSTANCE_ID}</i>",
                                    parse_mode="HTML"
                                )
                            except:
                                pass
                        
                        db.close()
                    except:
                        pass
                    
                    # Exit this instance gracefully
                    logger.critical("🛑 Shutting down this instance due to conflict")
                    await self.release_lock()
                    os._exit(1)  # Force exit
                
                # Check if lock file has our PID
                if lock_path.exists():
                    with open(LOCK_FILE, 'r') as f:
                        lock_data = f.read().strip().split('|')
                        if lock_data[0] != INSTANCE_ID:
                            logger.error(f"⚠️ Lock file hijacked by PID {lock_data[0]}!")
                            self.is_locked = False
                
                await asyncio.sleep(5)  # Check every 5 seconds
                
            except Exception as e:
                logger.error(f"Error in conflict monitor: {e}")
                await asyncio.sleep(10)
    
    async def check_bot_health(self) -> dict:
        """Check bot connection health and detect conflicts"""
        try:
            # Try to get bot info
            me = await self.bot.get_me()
            
            return {
                'healthy': True,
                'bot_username': me.username,
                'bot_id': me.id,
                'instance_pid': INSTANCE_ID,
                'has_lock': self.is_locked
            }
        except Exception as e:
            return {
                'healthy': False,
                'error': str(e),
                'instance_pid': INSTANCE_ID,
                'has_lock': self.is_locked
            }


# Global instance manager
_instance_manager = None

def get_instance_manager(bot: Bot = None) -> BotInstanceManager:
    """Get or create instance manager singleton"""
    global _instance_manager
    if _instance_manager is None and bot:
        _instance_manager = BotInstanceManager(bot)
    return _instance_manager
