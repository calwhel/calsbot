"""
Bot Instance Manager — PostgreSQL advisory lock prevents Telegram conflicts
across multiple servers (dev + deployed) sharing the same Neon database.

How it works:
  - On startup each instance calls pg_try_advisory_lock(LOCK_ID)
  - Only ONE instance can hold the lock at a time (DB-level, cross-server)
  - The winner polls Telegram normally
  - The loser waits 30 s and retries in a loop
  - The lock is automatically released when the holder's DB connection closes
    (process crash, restart, normal shutdown)
"""
import asyncio
import logging
import os
import threading
import time
from aiogram import Bot
from app.config import settings

logger = logging.getLogger(__name__)

ADVISORY_LOCK_ID = 5_432_109_876          # Arbitrary stable integer — unique to this app
RETRY_INTERVAL   = 30                     # Seconds loser waits before re-trying
KEEPALIVE_INTERVAL = 20                   # Seconds between keepalive pings on the lock conn

_lock_conn   = None    # The psycopg2 connection holding the lock
_lock_held   = False
_instance_id = str(os.getpid())


def _get_db_url() -> str:
    """Return Neon database URL from environment."""
    url = os.environ.get("NEON_DATABASE_URL") or os.environ.get("DATABASE_URL", "")
    return url


def _try_acquire_advisory_lock() -> bool:
    """
    Try to acquire a PostgreSQL advisory lock.
    Returns True if acquired, False if another instance holds it.
    Keeps the connection open so the lock persists.
    """
    global _lock_conn, _lock_held
    import psycopg2

    db_url = _get_db_url()
    if not db_url:
        logger.warning("No DB URL — skipping advisory lock, proceeding anyway")
        _lock_held = True
        return True

    try:
        if _lock_conn:
            try:
                _lock_conn.close()
            except Exception:
                pass
            _lock_conn = None

        conn = psycopg2.connect(db_url, connect_timeout=10,
                                keepalives=1, keepalives_idle=30,
                                keepalives_interval=10, keepalives_count=5)
        conn.autocommit = True

        with conn.cursor() as cur:
            cur.execute("SELECT pg_try_advisory_lock(%s)", (ADVISORY_LOCK_ID,))
            acquired = cur.fetchone()[0]

        if acquired:
            _lock_conn = conn
            _lock_held = True
            logger.info(f"✅ DB advisory lock acquired — this instance is the Telegram poller (PID {_instance_id})")
            return True
        else:
            conn.close()
            _lock_held = False
            logger.info("⏳ Another instance holds the advisory lock — this instance will wait")
            return False

    except Exception as e:
        logger.error(f"Advisory lock error: {e}")
        _lock_held = True      # Fail-open: if DB is unreachable, try to poll anyway
        return True


def _release_advisory_lock():
    """Release the advisory lock by closing the connection."""
    global _lock_conn, _lock_held
    if _lock_conn:
        try:
            _lock_conn.close()
        except Exception:
            pass
        _lock_conn = None
    _lock_held = False
    logger.info("🔓 Advisory lock released")


def _keepalive_loop():
    """Background thread: ping the lock connection to keep it alive."""
    import psycopg2
    while True:
        time.sleep(KEEPALIVE_INTERVAL)
        global _lock_conn
        if _lock_conn is None:
            break
        try:
            with _lock_conn.cursor() as cur:
                cur.execute("SELECT 1")
        except Exception as e:
            logger.warning(f"Advisory lock keepalive failed: {e}")
            break


# ─── Public interface ────────────────────────────────────────────────────────

class BotInstanceManager:
    """Manages single bot instance via PostgreSQL advisory lock."""

    def __init__(self, bot: Bot):
        self.bot = bot
        self.telegram_conflict_count = 0
        self.last_conflict_time      = None
        self._keepalive_thread       = None

    @property
    def is_locked(self) -> bool:
        return _lock_held

    async def acquire_lock(self) -> bool:
        """
        Attempt to become the active polling instance.
        Blocks (with async sleep) until the lock is acquired.
        Returns True once this instance is the winner.
        """
        while True:
            acquired = await asyncio.to_thread(_try_acquire_advisory_lock)
            if acquired:
                self._start_keepalive()
                return True
            logger.info(f"🔁 Lock not available — retrying in {RETRY_INTERVAL}s...")
            await asyncio.sleep(RETRY_INTERVAL)

    def _start_keepalive(self):
        """Start the background keepalive thread."""
        if self._keepalive_thread is None or not self._keepalive_thread.is_alive():
            t = threading.Thread(target=_keepalive_loop, daemon=True)
            t.start()
            self._keepalive_thread = t

    async def release_lock(self):
        await asyncio.to_thread(_release_advisory_lock)

    def register_telegram_conflict(self):
        """Called whenever aiogram receives a TelegramConflictError."""
        current = time.time()
        if self.last_conflict_time and (current - self.last_conflict_time) > 60:
            self.telegram_conflict_count = 0
        self.telegram_conflict_count += 1
        self.last_conflict_time = current
        logger.warning(f"Telegram conflict #{self.telegram_conflict_count} detected")

    async def check_telegram_conflicts(self):
        """Handle sustained conflicts: try to kick out the ghost instance."""
        if self.telegram_conflict_count == 1:
            try:
                logger.warning("🔄 Conflict — deleting webhook to kick out ghost instance...")
                await self.bot.delete_webhook(drop_pending_updates=True)
            except Exception:
                pass

        if self.telegram_conflict_count >= 20:
            logger.error("⚠️ SUSTAINED TELEGRAM CONFLICT: another instance is still polling!")
            if self.telegram_conflict_count == 20:
                await self._alert_admins()
            self.telegram_conflict_count = 0
            logger.warning("🔁 Conflict counter reset — continuing to retry")

    async def _alert_admins(self):
        try:
            from app.database import SessionLocal
            from app.models import User
            db = SessionLocal()
            admins = db.query(User).filter(User.is_admin == True).all()
            for admin in admins:
                try:
                    await self.bot.send_message(
                        str(admin.telegram_id),
                        "⚠️ <b>TELEGRAM CONFLICT</b>\n\n"
                        "Two bot instances are polling. "
                        "Check your deployed app — one environment should win shortly.\n\n"
                        f"<i>PID: {_instance_id}</i>",
                        parse_mode="HTML"
                    )
                except Exception:
                    pass
            db.close()
        except Exception:
            pass

    async def force_stop_other_instances(self) -> bool:
        try:
            await self.bot.delete_webhook(drop_pending_updates=True)
            await asyncio.sleep(5)
            return True
        except Exception as e:
            logger.error(f"force_stop error: {e}")
            return False

    async def start_conflict_monitor(self):
        """Lightweight monitor — main conflict prevention is the DB lock."""
        logger.info("👀 Conflict monitor started (DB advisory lock is primary guard)")
        while True:
            try:
                await self.check_telegram_conflicts()
                await asyncio.sleep(5)
            except Exception as e:
                logger.error(f"Conflict monitor error: {e}")
                await asyncio.sleep(10)

    async def check_bot_health(self) -> dict:
        try:
            me = await self.bot.get_me()
            return {
                'healthy': True,
                'bot_username': me.username,
                'bot_id': me.id,
                'instance_pid': _instance_id,
                'has_lock': self.is_locked,
            }
        except Exception as e:
            return {
                'healthy': False,
                'error': str(e),
                'instance_pid': _instance_id,
                'has_lock': self.is_locked,
            }


# Backward-compatible constants (some handlers still import these)
LOCK_FILE   = "/tmp/telegram_bot.lock"   # kept for compatibility — not used for locking
INSTANCE_ID = _instance_id

# Global singleton
_instance_manager: BotInstanceManager | None = None


def get_instance_manager(bot: Bot | None = None) -> BotInstanceManager | None:
    global _instance_manager
    if _instance_manager is None and bot:
        _instance_manager = BotInstanceManager(bot)
    return _instance_manager
