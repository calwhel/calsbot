"""
Strategy Portal Server — Build Your Own Strategy Portal
Standalone FastAPI server on port 8080.
Run alongside the main bot (separate workflow).
"""
import os
import logging
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, Request, HTTPException, Query
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Strategy Portal", docs_url=None, redoc_url=None)
templates = Jinja2Templates(directory="app/templates")


def get_db():
    from app.database import SessionLocal
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def _get_user_by_uid(uid: str, db: Session):
    from app.models import User
    return db.query(User).filter(User.uid == uid).first()


def _ensure_tables():
    from app.database import engine
    from app.strategy_models import init_strategy_tables
    init_strategy_tables(engine)


@app.on_event("startup")
async def startup():
    _ensure_tables()
    logger.info("Strategy portal started on port 8080")


# ─────────────────────────────────────────────────────────────────────────────
# Main portal page
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/strategies", response_class=HTMLResponse)
async def portal_page(request: Request, uid: str = Query(...)):
    from app.database import SessionLocal
    db = SessionLocal()
    try:
        user = _get_user_by_uid(uid, db)
        if not user:
            raise HTTPException(status_code=403, detail="Invalid access link")
        if user.banned:
            raise HTTPException(status_code=403, detail="Account banned")

        from app.strategy_models import UserStrategy, StrategyPerformance

        strategies = (
            db.query(UserStrategy)
            .filter(UserStrategy.user_id == user.id)
            .order_by(UserStrategy.updated_at.desc())
            .all()
        )

        strategy_data = []
        for s in strategies:
            perf = db.query(StrategyPerformance).filter(
                StrategyPerformance.strategy_id == s.id
            ).first()
            strategy_data.append({
                "id":           s.id,
                "name":         s.name,
                "description":  s.description,
                "status":       s.status,
                "is_public":    s.is_public,
                "config":       s.config,
                "created_at":   s.created_at.strftime("%Y-%m-%d") if s.created_at else "",
                "total_trades": perf.total_trades if perf else 0,
                "win_rate":     round(perf.win_rate, 1) if perf else 0,
                "total_pnl":    round(perf.total_pnl_pct, 2) if perf else 0,
                "open_trades":  perf.open_trades if perf else 0,
            })

        return templates.TemplateResponse("strategy_portal.html", {
            "request":    request,
            "user":       user,
            "uid":        uid,
            "strategies": strategy_data,
        })
    finally:
        db.close()


# ─────────────────────────────────────────────────────────────────────────────
# API endpoints (used by portal JS)
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/api/strategies")
async def api_strategies(uid: str = Query(...)):
    from app.database import SessionLocal
    db = SessionLocal()
    try:
        user = _get_user_by_uid(uid, db)
        if not user:
            raise HTTPException(status_code=403)

        from app.strategy_models import UserStrategy, StrategyPerformance, StrategyExecution
        strategies = (
            db.query(UserStrategy)
            .filter(UserStrategy.user_id == user.id)
            .order_by(UserStrategy.updated_at.desc())
            .all()
        )

        result = []
        for s in strategies:
            perf = db.query(StrategyPerformance).filter(
                StrategyPerformance.strategy_id == s.id
            ).first()
            recent_execs = (
                db.query(StrategyExecution)
                .filter(StrategyExecution.strategy_id == s.id)
                .order_by(StrategyExecution.fired_at.desc())
                .limit(10)
                .all()
            )
            result.append({
                "id":           s.id,
                "name":         s.name,
                "description":  s.description,
                "status":       s.status,
                "config":       s.config,
                "is_public":    s.is_public,
                "created_at":   s.created_at.isoformat() if s.created_at else None,
                "performance": {
                    "total_trades": perf.total_trades if perf else 0,
                    "wins":         perf.wins if perf else 0,
                    "losses":       perf.losses if perf else 0,
                    "win_rate":     round(perf.win_rate, 1) if perf else 0,
                    "total_pnl":    round(perf.total_pnl_pct, 2) if perf else 0,
                    "open_trades":  perf.open_trades if perf else 0,
                    "best_trade":   round(perf.best_trade, 2) if perf else 0,
                    "worst_trade":  round(perf.worst_trade, 2) if perf else 0,
                } if perf else {},
                "recent_trades": [{
                    "symbol":    ex.symbol,
                    "direction": ex.direction,
                    "outcome":   ex.outcome,
                    "pnl_pct":   round(ex.pnl_pct, 2) if ex.pnl_pct else None,
                    "fired_at":  ex.fired_at.isoformat() if ex.fired_at else None,
                } for ex in recent_execs],
            })
        return JSONResponse(result)
    finally:
        db.close()


@app.post("/api/strategies/{strategy_id}/toggle")
async def api_toggle_strategy(strategy_id: int, uid: str = Query(...)):
    from app.database import SessionLocal
    db = SessionLocal()
    try:
        user = _get_user_by_uid(uid, db)
        if not user:
            raise HTTPException(status_code=403)

        from app.strategy_models import UserStrategy
        strategy = db.query(UserStrategy).filter(
            UserStrategy.id == strategy_id,
            UserStrategy.user_id == user.id,
        ).first()
        if not strategy:
            raise HTTPException(status_code=404)

        strategy.status = "active" if strategy.status != "active" else "paused"
        db.commit()
        return {"status": strategy.status}
    finally:
        db.close()


@app.delete("/api/strategies/{strategy_id}")
async def api_delete_strategy(strategy_id: int, uid: str = Query(...)):
    from app.database import SessionLocal
    db = SessionLocal()
    try:
        user = _get_user_by_uid(uid, db)
        if not user:
            raise HTTPException(status_code=403)

        from app.strategy_models import UserStrategy
        strategy = db.query(UserStrategy).filter(
            UserStrategy.id == strategy_id,
            UserStrategy.user_id == user.id,
        ).first()
        if not strategy:
            raise HTTPException(status_code=404)

        db.delete(strategy)
        db.commit()
        return {"deleted": True}
    finally:
        db.close()


@app.get("/api/marketplace")
async def api_marketplace(uid: str = Query(...)):
    from app.database import SessionLocal
    db = SessionLocal()
    try:
        user = _get_user_by_uid(uid, db)
        if not user:
            raise HTTPException(status_code=403)

        from app.strategy_models import StrategyMarketplace, UserStrategy, StrategyPerformance
        listings = (
            db.query(StrategyMarketplace)
            .order_by(StrategyMarketplace.clone_count.desc())
            .limit(20)
            .all()
        )

        result = []
        for m in listings:
            strat = db.query(UserStrategy).filter(UserStrategy.id == m.strategy_id).first()
            perf  = db.query(StrategyPerformance).filter(
                StrategyPerformance.strategy_id == m.strategy_id
            ).first()
            result.append({
                "id":          m.id,
                "strategy_id": m.strategy_id,
                "title":       m.title,
                "summary":     m.summary,
                "tags":        m.tags,
                "clone_count": m.clone_count,
                "is_featured": m.is_featured,
                "win_rate":    round(perf.win_rate, 1) if perf and perf.total_trades > 0 else None,
                "total_pnl":   round(perf.total_pnl_pct, 2) if perf and perf.total_trades > 0 else None,
                "config":      strat.config if strat else {},
            })
        return JSONResponse(result)
    finally:
        db.close()


@app.post("/api/marketplace/{listing_id}/clone")
async def api_clone_strategy(listing_id: int, uid: str = Query(...)):
    from app.database import SessionLocal
    db = SessionLocal()
    try:
        user = _get_user_by_uid(uid, db)
        if not user:
            raise HTTPException(status_code=403)

        from app.strategy_models import (
            StrategyMarketplace, UserStrategy, StrategyPerformance, init_strategy_tables
        )
        from app.database import engine
        init_strategy_tables(engine)

        listing = db.query(StrategyMarketplace).filter(
            StrategyMarketplace.id == listing_id
        ).first()
        if not listing:
            raise HTTPException(status_code=404)

        original = db.query(UserStrategy).filter(
            UserStrategy.id == listing.strategy_id
        ).first()
        if not original:
            raise HTTPException(status_code=404)

        import copy
        cloned_config = copy.deepcopy(original.config)
        cloned_config["name"] = f"{original.name} (Clone)"

        new_strategy = UserStrategy(
            user_id     = user.id,
            name        = cloned_config["name"],
            description = original.description,
            config      = cloned_config,
            status      = "draft",
        )
        db.add(new_strategy)
        listing.clone_count += 1
        db.commit()
        db.refresh(new_strategy)

        perf = StrategyPerformance(strategy_id=new_strategy.id)
        db.add(perf)
        db.commit()

        return {"id": new_strategy.id, "name": new_strategy.name}
    finally:
        db.close()


@app.post("/api/build-strategy")
async def api_build_strategy(request: Request):
    """Compile a strategy from plain-English description using AI."""
    body = await request.json()
    uid  = body.get("uid")
    if not uid:
        raise HTTPException(status_code=400, detail="uid required")

    from app.database import SessionLocal
    db = SessionLocal()
    try:
        user = _get_user_by_uid(uid, db)
        if not user or user.banned:
            raise HTTPException(status_code=403)
    finally:
        db.close()

    name = body.get("name", "My Strategy")
    desc = body.get("description", "")
    if not desc:
        raise HTTPException(status_code=400, detail="description required")

    from app.services.strategy_builder import (
        compile_strategy_from_conversation,
        validate_strategy,
    )

    config = await compile_strategy_from_conversation([], f"Strategy name: {name}\n\n{desc}")
    if not config:
        return JSONResponse({"error": "Could not parse strategy. Try adding more detail about entry conditions and TP/SL."}, status_code=422)

    config["name"]        = name
    config["description"] = desc

    validation = await validate_strategy(config)

    return JSONResponse({
        "config":      config,
        "warnings":    validation.get("warnings", []),
        "suggestions": validation.get("suggestions", []),
        "risk_rating": validation.get("risk_rating", "MEDIUM"),
        "summary":     validation.get("summary", desc),
    })


@app.post("/api/save-strategy")
async def api_save_strategy(request: Request):
    """Save a compiled strategy config to the database."""
    body = await request.json()
    uid  = body.get("uid")
    if not uid:
        raise HTTPException(status_code=400, detail="uid required")

    config = body.get("config")
    if not config:
        raise HTTPException(status_code=400, detail="config required")

    from app.database import SessionLocal, engine
    from app.strategy_models import UserStrategy, StrategyPerformance, init_strategy_tables
    init_strategy_tables(engine)

    db = SessionLocal()
    try:
        user = _get_user_by_uid(uid, db)
        if not user or user.banned:
            raise HTTPException(status_code=403)

        strategy = UserStrategy(
            user_id     = user.id,
            name        = config.get("name", "My Strategy"),
            description = config.get("description", ""),
            config      = config,
            status      = "draft",
        )
        db.add(strategy)
        db.commit()
        db.refresh(strategy)

        perf = StrategyPerformance(strategy_id=strategy.id)
        db.add(perf)
        db.commit()

        return JSONResponse({"id": strategy.id, "name": strategy.name, "status": "draft"})
    finally:
        db.close()


@app.get("/health")
async def health():
    return {"status": "ok", "service": "strategy-portal"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
