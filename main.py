import asyncio
import uvicorn
from contextlib import asynccontextmanager
from fastapi import FastAPI

from app.services.subscriptions import api
from app.services.bot import start_bot
from app.config import settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    bot_task = asyncio.create_task(start_bot())
    yield
    bot_task.cancel()
    try:
        await bot_task
    except asyncio.CancelledError:
        pass


api.router.lifespan_context = lifespan


if __name__ == "__main__":
    uvicorn.run(
        "main:api",
        host="0.0.0.0",
        port=settings.PORT,
        log_level="info"
    )
