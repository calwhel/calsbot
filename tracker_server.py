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
    init_db()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("tracker_server:app", host="0.0.0.0", port=5000, log_level="info")
