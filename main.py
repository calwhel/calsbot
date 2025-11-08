import asyncio
import uvicorn
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI

from app.services.bot import start_bot
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
            # Add crypto_wallet column if it doesn't exist
            conn.execute(text("ALTER TABLE users ADD COLUMN IF NOT EXISTS crypto_wallet VARCHAR"))
            conn.commit()
            logging.info("✅ Database migrations completed successfully")
        
        engine.dispose()
    except Exception as e:
        logging.warning(f"Migration warning (may already exist): {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    run_migrations()  # Run migrations after DB init
    bot_task = asyncio.create_task(start_bot())
    yield
    bot_task.cancel()
    try:
        await bot_task
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
