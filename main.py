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

from app.deployment import should_poll_telegram, payments_enabled

# Dev should NOT poll Telegram — it would conflict with the live production bot.
# Railway/Replit set this automatically; use FORCE_BOT_POLL=1 to override locally.
_START_BOT_POLLING = should_poll_telegram()

if _START_BOT_POLLING:
    print("🌐 Production/Railway — Telegram bot polling ENABLED", flush=True)
else:
    print("🛠️  DEV environment — bot polling DISABLED (set FORCE_BOT_POLL=1 to override)", flush=True)

from app.services.bot import start_bot
print("✅ Bot module loaded", flush=True)
from app.config import settings
from app.database import init_db_minimal as init_db
from app.services.subscriptions import api as subscription_api

from app.logging_safe import configure_safe_logging

configure_safe_logging(logging.INFO)


def run_migrations():
    """Run database migrations automatically on startup"""
    try:
        from sqlalchemy import text
        from app.database import engine
        
        logging.info("🔧 Running database migrations...")
        
        with engine.connect() as conn:
            # Add referral system columns if they don't exist
            # lock_timeout prevents holding ACCESS EXCLUSIVE lock on users table
            # indefinitely; migration retries on next restart if lock is busy.
            for _ddl in (
                "ALTER TABLE users ADD COLUMN IF NOT EXISTS referral_earnings FLOAT DEFAULT 0.0",
                "ALTER TABLE users ADD COLUMN IF NOT EXISTS paid_referrals TEXT DEFAULT ''",
                "ALTER TABLE users ADD COLUMN IF NOT EXISTS crypto_wallet VARCHAR",
            ):
                try:
                    conn.execute(text("SET LOCAL lock_timeout = '2s'"))
                    conn.execute(text(_ddl))
                    conn.commit()
                except Exception:
                    try:
                        conn.rollback()
                    except Exception:
                        pass
            logging.info("✅ Database migrations completed successfully (added referral_earnings, paid_referrals, crypto_wallet)")
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
        try:
            from app.services.telegram_poller_lock import describe_bot_token
            from app.services.telegram_tokens import forex_bot_token, main_bot_token

            _main_info = await describe_bot_token(main_bot_token(), "main")
            _fx_info = await describe_bot_token(forex_bot_token(), "forex")
            logging.info(f"[tg] poller identity: main={_main_info} forex={_fx_info}")
        except Exception as _id_err:
            logging.warning(f"[tg] bot identity probe failed: {_id_err}")

        bot_task = asyncio.create_task(start_bot())
        from app.services.telegram_tokens import should_run_forex_poller, tokens_are_same
        from app.services.telegram_tokens import forex_bot_token, main_bot_token
        if should_run_forex_poller():
            from app.services.forex_bot import start_forex_bot
            forex_bot_task = asyncio.create_task(start_forex_bot())
        elif forex_bot_token() and tokens_are_same(forex_bot_token(), main_bot_token()):
            logging.warning(
                "[tg] FOREX_BOT_TOKEN == TELEGRAM_BOT_TOKEN — skipping second "
                "poller (prevents TelegramConflictError)"
            )
    else:
        bot_task = None
        forex_bot_task = None
        logging.info("🛠️  DEV mode — Telegram polling skipped (production bot handles it)")
    
    # OxaPay poller only when payments are configured
    poller_task = None
    if payments_enabled():
        from app.services.oxapay_poller import poll_oxapay_payments
        poller_task = asyncio.create_task(poll_oxapay_payments())
    else:
        logging.info("OxaPay poller skipped (free portal / no merchant key)")
    
    # Twitter auto-posting — strategy leaderboard + market content.
    # Gated by an advisory lock (run_auto_post_loop_singleton) so this companion
    # and the always-up web worker never both post; whichever wins the lock runs,
    # the other stands by and takes over automatically on failover.
    from app.services.twitter_poster import run_auto_post_loop_singleton
    twitter_task = asyncio.create_task(run_auto_post_loop_singleton())

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
    if poller_task:
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
    if poller_task:
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
