"""
Bot Instance Manager - Prevents and manages multiple bot instance conflicts
"""
import asyncio
import logging
import os
import fcntl
import time
from datetime import datetime
from pathlib import Path
from aiogram import Bot
from app.config import settings

logger = logging.getLogger(__name__)

LOCK_FILE = "/tmp/telegram_bot.lock"
INSTANCE_ID = str(os.getpid())  # Use process ID as instance identifier
_lock_fd = None  # Global lock file descriptor


class BotInstanceManager:
    """Manages single bot instance and prevents conflicts"""
    
    def __init__(self, bot: Bot):
        self.bot = bot
        self.is_locked = False
        self.monitor_task = None
        self.telegram_conflict_count = 0
        self.last_conflict_time = None
        
    async def acquire_lock(self) -> bool:
        """
        Acquire single instance lock using fcntl.flock (truly atomic)
        Returns True if lock acquired, False if another instance running
        """
        global _lock_fd
        
        try:
            # Open/create lock file
            _lock_fd = open(LOCK_FILE, 'w')
            
            # Try to acquire exclusive lock (non-blocking)
            try:
                fcntl.flock(_lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                # Another instance holds the lock
                logger.warning("Another bot instance is running (lock held)")
                _lock_fd.close()
                _lock_fd = None
                return False
            
            # Write instance info to lock file
            _lock_fd.write(f"{INSTANCE_ID}|{datetime.utcnow().isoformat()}\n")
            _lock_fd.flush()
            
            self.is_locked = True
            logger.info(f"✅ Bot instance lock acquired (PID: {INSTANCE_ID})")
            return True
            
        except Exception as e:
            logger.error(f"Error acquiring instance lock: {e}")
            if _lock_fd:
                _lock_fd.close()
                _lock_fd = None
            return False
    
    async def release_lock(self):
        """Release the instance lock"""
        global _lock_fd
        
        try:
            if self.is_locked and _lock_fd:
                # Release the lock and close file descriptor
                fcntl.flock(_lock_fd, fcntl.LOCK_UN)
                _lock_fd.close()
                _lock_fd = None
                logger.info(f"🔓 Bot instance lock released (PID: {INSTANCE_ID})")
                self.is_locked = False
        except Exception as e:
            logger.error(f"Error releasing lock: {e}")
    
    def register_telegram_conflict(self):
        """Register a Telegram conflict error occurred"""
        import time as time_module
        current_time = time_module.time()
        
        # Reset counter if last conflict was > 60 seconds ago
        if self.last_conflict_time and (current_time - self.last_conflict_time) > 60:
            self.telegram_conflict_count = 0
        
        self.telegram_conflict_count += 1
        self.last_conflict_time = current_time
        
        logger.warning(f"Telegram conflict #{self.telegram_conflict_count} detected")
    
    async def check_telegram_conflicts(self):
        """Check if we should shut down due to sustained Telegram conflicts"""
        # On first conflict: actively kick out the old instance via delete_webhook
        if self.telegram_conflict_count == 1:
            try:
                logger.warning("🔄 Conflict detected — re-deleting webhook to kick out old instance...")
                await self.bot.delete_webhook(drop_pending_updates=True)
            except Exception:
                pass

        # If we have 20+ conflicts in last 60 seconds, another persistent instance is active.
        # We log and alert but DO NOT shut down — this instance should win once the old one stops.
        if self.telegram_conflict_count >= 20:
            logger.error("⚠️ SUSTAINED TELEGRAM CONFLICT: Another remote instance is polling!")
            
            # Alert admins once (not on every cycle)
            if self.telegram_conflict_count == 20:
                try:
                    from app.database import SessionLocal
                    from app.models import User
                    
                    db = SessionLocal()
                    admins = db.query(User).filter(User.is_admin == True).all()
                    
                    for admin in admins:
                        try:
                            await self.bot.send_message(
                                str(admin.telegram_id),
                                "⚠️ <b>TELEGRAM CONFLICT DETECTED!</b>\n\n"
                                "Another bot instance is polling. This instance will keep retrying — "
                                "stop the other deployment (e.g. Railway) to resolve.\n\n"
                                f"<i>PID: {INSTANCE_ID}</i>",
                                parse_mode="HTML"
                            )
                        except:
                            pass
                    
                    db.close()
                except:
                    pass

            # Reset counter so we don't alert every single tick, and keep retrying
            self.telegram_conflict_count = 0
            logger.warning("🔁 Conflict counter reset — will keep polling and retry")
    
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
            
            # Wait for other instances to disconnect and stop polling
            await asyncio.sleep(10)
            
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
                # Check for sustained Telegram conflicts
                await self.check_telegram_conflicts()
                
                # Check if our lock file still exists
                lock_path = Path(LOCK_FILE)
                if not lock_path.exists() and self.is_locked:
                    # Lock file gone — re-acquire it and keep running
                    logger.warning("⚠️ Lock file gone — re-acquiring and continuing")
                    self.is_locked = False
                    try:
                        await self.acquire_lock()
                    except Exception:
                        pass
                
                # Check if lock file has our PID
                if lock_path.exists():
                    with open(LOCK_FILE, 'r') as f:
                        lock_data = f.read().strip().split('|')
                        if lock_data[0] != INSTANCE_ID:
                            # Lock hijacked by another instance!
                            logger.error(f"⚠️ CONFLICT: Lock file hijacked by PID {lock_data[0]}!")
                            self.is_locked = False
                            
                            # Alert admins
                            try:
                                from app.database import SessionLocal
                                from app.models import User
                                
                                db = SessionLocal()
                                admins = db.query(User).filter(User.is_admin == True).all()
                                
                                for admin in admins:
                                    try:
                                        await self.bot.send_message(
                                            str(admin.telegram_id),
                                            "⚠️ <b>BOT INSTANCE CONFLICT!</b>\n\n"
                                            f"Another instance (PID: {lock_data[0]}) has hijacked the lock. "
                                            "This instance will shut down to prevent conflicts.\n\n"
                                            f"<i>Shutting down PID: {INSTANCE_ID}</i>",
                                            parse_mode="HTML"
                                        )
                                    except:
                                        pass
                                
                                db.close()
                            except:
                                pass
                            
                            # Re-acquire our lock rather than shutting down
                            logger.warning("🔁 Lock appears hijacked — re-acquiring and continuing")
                            try:
                                await self.acquire_lock()
                            except Exception:
                                pass
                
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

def get_instance_manager(bot: Bot | None = None) -> BotInstanceManager | None:
    """Get or create instance manager singleton"""
    global _instance_manager
    if _instance_manager is None and bot:
        _instance_manager = BotInstanceManager(bot)
    return _instance_manager
