print("=" * 50, flush=True)
print("🚀 CRYPTO BOT STARTING...", flush=True)
print("=" * 50, flush=True)

import asyncio
import os
import uvicorn
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI

print("✅ Core imports loaded", flush=True)

# In deployed production REPLIT_DEPLOYMENT is set to "1".
# In the Replit IDE (dev) it is absent.
# Dev should NOT poll Telegram — it would conflict with the live production bot.
# Set FORCE_BOT_POLL=1 in the IDE environment to override this for local testing.
_IS_PRODUCTION = bool(os.environ.get("REPLIT_DEPLOYMENT"))
_FORCE_BOT_POLL = bool(os.environ.get("FORCE_BOT_POLL"))
_START_BOT_POLLING = _IS_PRODUCTION or _FORCE_BOT_POLL

if _IS_PRODUCTION:
    print("🌐 PRODUCTION environment detected — bot polling ENABLED", flush=True)
elif _FORCE_BOT_POLL:
    print("⚠️  FORCE_BOT_POLL override — bot polling ENABLED (dev mode)", flush=True)
else:
    print("🛠️  DEV environment — bot polling DISABLED (set FORCE_BOT_POLL=1 to override)", flush=True)

from app.services.bot import start_bot
print("✅ Bot module loaded", flush=True)
from app.config import settings
from app.database import init_db_minimal as init_db
from app.services.subscriptions import api as subscription_api

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


def run_migrations():
    """Run database migrations automatically on startup"""
    try:
        from sqlalchemy import create_engine, text
        
        logging.info("🔧 Running database migrations...")
        engine = create_engine(settings.get_database_url())
        
        with engine.connect() as conn:
            # Add referral system columns if they don't exist
            conn.execute(text("ALTER TABLE users ADD COLUMN IF NOT EXISTS referral_earnings FLOAT DEFAULT 0.0"))
            conn.execute(text("ALTER TABLE users ADD COLUMN IF NOT EXISTS paid_referrals TEXT DEFAULT ''"))
            conn.execute(text("ALTER TABLE users ADD COLUMN IF NOT EXISTS crypto_wallet VARCHAR"))
            conn.commit()
            logging.info("✅ Database migrations completed successfully (added referral_earnings, paid_referrals, crypto_wallet)")
        
        engine.dispose()
    except Exception as e:
        logging.warning(f"Migration warning (may already exist): {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    run_migrations()  # Run migrations after DB init
    
    # Migrate ccally account from env vars to database
    from app.services.twitter_poster import migrate_env_account_to_database
    migrate_env_account_to_database()
    
    # Only poll Telegram in production (or when FORCE_BOT_POLL=1 override is set).
    # This prevents the dev IDE instance from conflicting with the live production bot.
    if _START_BOT_POLLING:
        bot_task = asyncio.create_task(start_bot())
        from app.services.forex_bot import start_forex_bot
        forex_bot_task = asyncio.create_task(start_forex_bot())
    else:
        bot_task = None
        forex_bot_task = None
        logging.info("🛠️  DEV mode — Telegram polling skipped (production bot handles it)")
    
    # Start OxaPay automatic payment verification
    from app.services.oxapay_poller import poll_oxapay_payments
    poller_task = asyncio.create_task(poll_oxapay_payments())
    
    # Twitter auto-posting — strategy leaderboard + market content
    from app.services.twitter_poster import auto_post_loop
    twitter_task = asyncio.create_task(auto_post_loop())

    # Wall intel — schema init + background watch alerts
    from app.services.wall_intel import init_wall_intel_schema, watch_loop
    init_wall_intel_schema()
    wall_watch_task = asyncio.create_task(watch_loop())
    
    yield
    
    # Cleanup
    if bot_task:
        bot_task.cancel()
    if forex_bot_task:
        forex_bot_task.cancel()
        try:
            await forex_bot_task
        except asyncio.CancelledError:
            pass
    poller_task.cancel()
    twitter_task.cancel()
    wall_watch_task.cancel()
    try:
        await wall_watch_task
    except asyncio.CancelledError:
        pass
    if bot_task:
        try:
            await bot_task
        except asyncio.CancelledError:
            pass
    try:
        await poller_task
    except asyncio.CancelledError:
        pass
    try:
        await twitter_task
    except asyncio.CancelledError:
        pass


app = FastAPI(lifespan=lifespan)
app.mount("/", subscription_api)


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=settings.PORT,
        log_level="info"
    )
