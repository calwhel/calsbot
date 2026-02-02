print("=" * 50, flush=True)
print("🚀 CRYPTO BOT STARTING...", flush=True)
print("=" * 50, flush=True)

import asyncio
import uvicorn
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI

print("✅ Core imports loaded", flush=True)

from app.services.bot import start_bot
print("✅ Bot module loaded", flush=True)
from app.config import settings
from app.database import init_db
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
        engine = create_engine(settings.DATABASE_URL)
        
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
    
    # Start bot and OxaPay payment poller
    bot_task = asyncio.create_task(start_bot())
    
    # Start OxaPay automatic payment verification
    from app.services.oxapay_poller import poll_oxapay_payments
    poller_task = asyncio.create_task(poll_oxapay_payments())
    
    # Start Twitter auto-posting loop
    from app.services.twitter_poster import auto_post_loop
    twitter_task = asyncio.create_task(auto_post_loop())
    logging.info("🐦 Twitter auto-post loop started")
    
    yield
    
    # Cleanup
    bot_task.cancel()
    poller_task.cancel()
    twitter_task.cancel()
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
