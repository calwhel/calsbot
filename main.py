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


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
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
