from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.database import init_db
from app.services.trade_tracker import router as tracker_router

app = FastAPI(title="Trade Tracker")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(tracker_router)


@app.get("/")
async def root():
    return {"status": "ok", "service": "trade-tracker"}


@app.get("/health")
async def health():
    return {"ok": True}


@app.on_event("startup")
async def startup():
    import asyncio
    from app.services.trade_tracker import run_trade_monitor
    asyncio.create_task(_startup_bg())
    asyncio.create_task(run_trade_monitor())


async def _startup_bg():
    """Run blocking DB init off the event loop so the tracker is live instantly."""
    import asyncio as _aio
    loop = _aio.get_event_loop()
    try:
        await loop.run_in_executor(None, init_db)
    except Exception as e:
        import logging
        logging.getLogger(__name__).warning(f"Tracker init_db error: {e}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("tracker_server:app", host="0.0.0.0", port=8000, log_level="info")
