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
from app.models import User

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
            # Fast inline health score
            wr  = perf.win_rate if perf else 0
            tot = perf.total_trades if perf else 0
            pf  = (perf.avg_win_pct * max(perf.wins,1)) / (abs(perf.avg_loss_pct) * max(perf.losses,1)) if perf and perf.losses > 0 and perf.avg_loss_pct else 0
            health = 0.0
            if tot >= 3:
                health += min(wr / 100, 1.0) * 4.0
                health += min(pf / 2.0, 1.0) * 3.0
                health += min(tot / 30.0, 1.0) * 2.0
                health += 1.0  # base point for having trades
            health_score = round(min(health, 10.0), 1)

            result.append({
                "id":           s.id,
                "name":         s.name,
                "description":  s.description,
                "status":       s.status,
                "config":       s.config,
                "is_public":    s.is_public,
                "created_at":   s.created_at.isoformat() if s.created_at else None,
                "health_score": health_score,
                "performance": {
                    "total_trades": perf.total_trades if perf else 0,
                    "wins":         perf.wins if perf else 0,
                    "losses":       perf.losses if perf else 0,
                    "win_rate":     round(perf.win_rate, 1) if perf else 0,
                    "total_pnl":    round(perf.total_pnl_pct, 2) if perf else 0,
                    "open_trades":  perf.open_trades if perf else 0,
                    "best_trade":   round(perf.best_trade, 2) if perf else 0,
                    "worst_trade":  round(perf.worst_trade, 2) if perf else 0,
                    "avg_win_pct":  round(perf.avg_win_pct, 2) if perf else 0,
                    "avg_loss_pct": round(perf.avg_loss_pct, 2) if perf else 0,
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
async def api_marketplace(
    uid:      str = Query(...),
    sort:     str = Query("top"),
    category: str = Query("all"),
    pricing:  str = Query("all"),
    search:   str = Query(""),
):
    from app.database import SessionLocal, engine
    db = SessionLocal()
    try:
        user = _get_user_by_uid(uid, db)
        if not user:
            raise HTTPException(status_code=403)

        from app.strategy_models import StrategyMarketplace, UserStrategy, StrategyPerformance
        from app.strategy_marketplace_ext import StrategyPurchase, init_marketplace_ext_tables
        init_marketplace_ext_tables(engine)

        q = db.query(StrategyMarketplace)
        if category != "all":
            q = q.filter(StrategyMarketplace.category == category)
        if pricing == "free":
            q = q.filter(StrategyMarketplace.pricing_model == "free")
        elif pricing == "paid":
            q = q.filter(StrategyMarketplace.pricing_model != "free")
        if sort == "new":
            q = q.order_by(StrategyMarketplace.published_at.desc())
        elif sort == "trending":
            q = q.order_by(StrategyMarketplace.clone_count.desc())
        elif sort == "verified":
            q = q.filter(StrategyMarketplace.is_verified == True).order_by(StrategyMarketplace.verified_win_rate.desc())
        else:
            q = q.order_by(StrategyMarketplace.avg_rating.desc(), StrategyMarketplace.clone_count.desc())

        listings = q.limit(50).all()
        if search:
            s = search.lower()
            listings = [m for m in listings if s in (m.title or "").lower() or s in (m.summary or "").lower()]

        my_purchases = {
            p.listing_id for p in db.query(StrategyPurchase)
            .filter(StrategyPurchase.buyer_id == user.id, StrategyPurchase.status == "active").all()
        }

        result = []
        for m in listings:
            perf   = db.query(StrategyPerformance).filter(StrategyPerformance.strategy_id == m.strategy_id).first()
            author = db.query(User).filter(User.id == m.author_id).first()
            result.append({
                "id":               m.id,
                "strategy_id":      m.strategy_id,
                "title":            m.title,
                "summary":          m.summary,
                "tags":             m.tags or [],
                "category":         m.category or "general",
                "pricing_model":    m.pricing_model or "free",
                "price_usdt":       m.price_usdt or 0,
                "clone_count":      m.clone_count or 0,
                "subscriber_count": m.subscriber_count or 0,
                "avg_rating":       round(m.avg_rating or 0, 1),
                "rating_count":     m.rating_count or 0,
                "is_featured":      m.is_featured,
                "is_trending":      getattr(m, "is_trending", False),
                "is_verified":      m.is_verified,
                "verified_win_rate": round(m.verified_win_rate, 1) if m.is_verified and m.verified_win_rate else None,
                "verified_pnl":     round(m.verified_pnl, 2) if m.is_verified and m.verified_pnl else None,
                "live_win_rate":    round(perf.win_rate, 1) if perf and perf.total_trades >= 3 else None,
                "live_pnl":         round(perf.total_pnl_pct, 2) if perf and perf.total_trades >= 3 else None,
                "live_trades":      perf.total_trades if perf else 0,
                "author_name":      (author.first_name or author.username or "Anonymous") if author else "Anonymous",
                "author_uid":       author.uid if author else None,
                "is_owned":         m.id in my_purchases or (m.pricing_model or "free") == "free",
            })
        return JSONResponse(result)
    finally:
        db.close()


@app.get("/api/marketplace/{listing_id}")
async def api_marketplace_detail(listing_id: int, uid: str = Query(...)):
    from app.database import SessionLocal, engine
    db = SessionLocal()
    try:
        user = _get_user_by_uid(uid, db)
        if not user:
            raise HTTPException(status_code=403)

        from app.strategy_models import StrategyMarketplace, UserStrategy, StrategyPerformance, StrategyExecution
        from app.strategy_marketplace_ext import StrategyRating, StrategyPurchase, init_marketplace_ext_tables
        init_marketplace_ext_tables(engine)

        m = db.query(StrategyMarketplace).filter(StrategyMarketplace.id == listing_id).first()
        if not m:
            raise HTTPException(status_code=404)
        m.view_count = (m.view_count or 0) + 1
        db.commit()

        perf   = db.query(StrategyPerformance).filter(StrategyPerformance.strategy_id == m.strategy_id).first()
        author = db.query(User).filter(User.id == m.author_id).first()
        recent_trades = (
            db.query(StrategyExecution)
            .filter(StrategyExecution.strategy_id == m.strategy_id)
            .order_by(StrategyExecution.fired_at.desc()).limit(10).all()
        )
        ratings = (
            db.query(StrategyRating)
            .filter(StrategyRating.listing_id == listing_id)
            .order_by(StrategyRating.created_at.desc()).limit(20).all()
        )
        is_owned = (m.pricing_model or "free") == "free" or db.query(StrategyPurchase).filter(
            StrategyPurchase.buyer_id == user.id,
            StrategyPurchase.listing_id == listing_id,
            StrategyPurchase.status == "active",
        ).first() is not None
        my_rating = db.query(StrategyRating).filter(
            StrategyRating.rater_id == user.id, StrategyRating.listing_id == listing_id
        ).first()

        return JSONResponse({
            "id": m.id, "title": m.title, "summary": m.summary,
            "tags": m.tags or [], "category": m.category or "general",
            "pricing_model": m.pricing_model or "free", "price_usdt": m.price_usdt or 0,
            "is_verified": m.is_verified, "verified_trades": m.verified_trades or 0,
            "verified_win_rate": round(m.verified_win_rate or 0, 1),
            "avg_rating": round(m.avg_rating or 0, 1), "rating_count": m.rating_count or 0,
            "clone_count": m.clone_count or 0, "subscriber_count": m.subscriber_count or 0,
            "view_count": m.view_count or 0, "is_owned": is_owned,
            "author_name": (author.first_name or author.username or "Anonymous") if author else "Anonymous",
            "author_uid": author.uid if author else None,
            "live_performance": {
                "total_trades": perf.total_trades if perf else 0,
                "win_rate": round(perf.win_rate, 1) if perf else 0,
                "total_pnl": round(perf.total_pnl_pct, 2) if perf else 0,
            },
            "recent_trades": [{"symbol": ex.symbol, "direction": ex.direction, "outcome": ex.outcome,
                "pnl_pct": round(ex.pnl_pct, 2) if ex.pnl_pct else None} for ex in recent_trades],
            "ratings": [{"stars": r.stars, "review": r.review, "is_verified": r.is_verified} for r in ratings],
            "my_rating": {"stars": my_rating.stars, "review": my_rating.review} if my_rating else None,
        })
    finally:
        db.close()


@app.post("/api/marketplace/{listing_id}/purchase")
async def api_purchase_strategy(listing_id: int, uid: str = Query(...)):
    from app.database import SessionLocal, engine
    db = SessionLocal()
    try:
        user = _get_user_by_uid(uid, db)
        if not user:
            raise HTTPException(status_code=403)

        from app.strategy_models import StrategyMarketplace, UserStrategy, StrategyPerformance, init_strategy_tables
        from app.strategy_marketplace_ext import StrategyPurchase, CreatorEarnings, EarningsTransaction, init_marketplace_ext_tables, calculate_creator_cut, calculate_platform_cut
        init_strategy_tables(engine)
        init_marketplace_ext_tables(engine)

        listing = db.query(StrategyMarketplace).filter(StrategyMarketplace.id == listing_id).first()
        if not listing:
            raise HTTPException(status_code=404)

        existing = db.query(StrategyPurchase).filter(
            StrategyPurchase.buyer_id == user.id,
            StrategyPurchase.listing_id == listing_id,
            StrategyPurchase.status == "active",
        ).first()
        if existing:
            return JSONResponse({"already_owned": True, "cloned_strategy_id": existing.cloned_strategy_id})

        if (listing.pricing_model or "free") != "free" and (listing.price_usdt or 0) > 0:
            return JSONResponse({
                "requires_payment": True,
                "price_usdt": listing.price_usdt,
                "pricing_model": listing.pricing_model,
                "message": f"This strategy costs ${listing.price_usdt:.2f}. Pay via Telegram bot to unlock.",
            })

        # Free — clone immediately
        original = db.query(UserStrategy).filter(UserStrategy.id == listing.strategy_id).first()
        if not original:
            raise HTTPException(status_code=404)

        import copy
        cloned_config = copy.deepcopy(original.config)
        cloned_config["name"] = f"{original.name} (Clone)"
        new_strategy = UserStrategy(
            user_id=user.id, name=cloned_config["name"],
            description=original.description, config=cloned_config, status="draft"
        )
        db.add(new_strategy)
        db.commit()
        db.refresh(new_strategy)

        perf     = StrategyPerformance(strategy_id=new_strategy.id)
        purchase = StrategyPurchase(
            buyer_id=user.id, listing_id=listing_id, strategy_id=listing.strategy_id,
            pricing_model="free", amount_paid_usd=0.0, status="active",
            cloned_strategy_id=new_strategy.id,
        )
        db.add(perf)
        db.add(purchase)
        listing.clone_count = (listing.clone_count or 0) + 1
        db.commit()

        return JSONResponse({"success": True, "cloned_strategy_id": new_strategy.id, "name": new_strategy.name})
    finally:
        db.close()


@app.post("/api/marketplace/{listing_id}/rate")
async def api_rate_strategy(listing_id: int, uid: str = Query(...), stars: int = Query(...), review: str = Query("")):
    from app.database import SessionLocal, engine
    db = SessionLocal()
    try:
        user = _get_user_by_uid(uid, db)
        if not user:
            raise HTTPException(status_code=403)
        if not 1 <= stars <= 5:
            raise HTTPException(status_code=400, detail="Stars must be 1-5")

        from app.strategy_models import StrategyMarketplace
        from app.strategy_marketplace_ext import StrategyRating, StrategyPurchase, init_marketplace_ext_tables
        init_marketplace_ext_tables(engine)

        listing = db.query(StrategyMarketplace).filter(StrategyMarketplace.id == listing_id).first()
        if not listing:
            raise HTTPException(status_code=404)

        is_buyer = db.query(StrategyPurchase).filter(
            StrategyPurchase.buyer_id == user.id,
            StrategyPurchase.listing_id == listing_id,
            StrategyPurchase.status == "active",
        ).first() is not None

        existing = db.query(StrategyRating).filter(
            StrategyRating.rater_id == user.id, StrategyRating.listing_id == listing_id
        ).first()

        if existing:
            existing.stars = stars
            existing.review = review or None
            existing.is_verified = is_buyer
        else:
            db.add(StrategyRating(
                rater_id=user.id, listing_id=listing_id,
                stars=stars, review=review or None, is_verified=is_buyer
            ))

        db.commit()
        all_ratings = db.query(StrategyRating).filter(StrategyRating.listing_id == listing_id).all()
        if all_ratings:
            listing.avg_rating   = sum(r.stars for r in all_ratings) / len(all_ratings)
            listing.rating_count = len(all_ratings)
        db.commit()
        return JSONResponse({"success": True, "new_avg": round(listing.avg_rating, 1)})
    finally:
        db.close()


@app.get("/api/leaderboard")
async def api_leaderboard(uid: str = Query(...), metric: str = Query("win_rate")):
    from app.database import SessionLocal
    db = SessionLocal()
    try:
        user = _get_user_by_uid(uid, db)
        if not user:
            raise HTTPException(status_code=403)

        from app.strategy_models import StrategyMarketplace, StrategyPerformance
        listings = db.query(StrategyMarketplace).all()
        results  = []
        for m in listings:
            perf  = db.query(StrategyPerformance).filter(StrategyPerformance.strategy_id == m.strategy_id).first()
            author = db.query(User).filter(User.id == m.author_id).first()
            if not perf or perf.total_trades < 3:
                continue
            results.append({
                "listing_id": m.id, "title": m.title,
                "author": (author.first_name or author.username) if author else "Anonymous",
                "author_uid": author.uid if author else None,
                "win_rate": round(perf.win_rate, 1),
                "total_pnl": round(perf.total_pnl_pct, 2),
                "total_trades": perf.total_trades,
                "best_trade": round(perf.best_trade, 2),
                "avg_rating": round(m.avg_rating or 0, 1),
                "pricing_model": m.pricing_model or "free",
                "price_usdt": m.price_usdt or 0,
            })

        results.sort(key=lambda x: x.get(metric if metric in x else "win_rate", 0), reverse=True)
        return JSONResponse(results[:20])
    finally:
        db.close()


@app.get("/api/creator/{creator_uid}")
async def api_creator_profile(creator_uid: str, uid: str = Query(...)):
    from app.database import SessionLocal, engine
    db = SessionLocal()
    try:
        _get_user_by_uid(uid, db)
        creator = db.query(User).filter(User.uid == creator_uid).first()
        if not creator:
            raise HTTPException(status_code=404)

        from app.strategy_models import StrategyMarketplace, StrategyPerformance
        from app.strategy_marketplace_ext import CreatorEarnings, init_marketplace_ext_tables
        init_marketplace_ext_tables(engine)

        listings = db.query(StrategyMarketplace).filter(StrategyMarketplace.author_id == creator.id).all()
        earnings = db.query(CreatorEarnings).filter(CreatorEarnings.creator_id == creator.id).first()

        return JSONResponse({
            "name": creator.first_name or creator.username or "Anonymous",
            "uid": creator.uid,
            "joined": creator.created_at.strftime("%B %Y") if creator.created_at else "Unknown",
            "strategy_count": len(listings),
            "total_subscribers": earnings.total_subscribers if earnings else 0,
            "total_sales": earnings.total_sales if earnings else 0,
            "strategies": [{
                "id": m.id, "title": m.title, "summary": m.summary,
                "pricing_model": m.pricing_model or "free", "price_usdt": m.price_usdt or 0,
                "clone_count": m.clone_count or 0, "avg_rating": round(m.avg_rating or 0, 1),
                "is_verified": m.is_verified,
            } for m in listings],
        })
    finally:
        db.close()


@app.get("/api/my-earnings")
async def api_my_earnings(uid: str = Query(...)):
    from app.database import SessionLocal, engine
    db = SessionLocal()
    try:
        user = _get_user_by_uid(uid, db)
        if not user:
            raise HTTPException(status_code=403)

        from app.strategy_marketplace_ext import CreatorEarnings, EarningsTransaction, init_marketplace_ext_tables
        init_marketplace_ext_tables(engine)

        earnings   = db.query(CreatorEarnings).filter(CreatorEarnings.creator_id == user.id).first()
        recent_txs = (
            db.query(EarningsTransaction).filter(EarningsTransaction.creator_id == user.id)
            .order_by(EarningsTransaction.created_at.desc()).limit(20).all()
        )
        return JSONResponse({
            "total_earned": earnings.total_earned if earnings else 0,
            "pending_payout": earnings.pending_payout if earnings else 0,
            "total_paid_out": earnings.total_paid_out if earnings else 0,
            "total_sales": earnings.total_sales if earnings else 0,
            "platform_cut_pct": 20, "creator_cut_pct": 80,
            "recent_sales": [{"gross": tx.gross_amount, "your_cut": tx.creator_cut,
                "created_at": tx.created_at.isoformat() if tx.created_at else None} for tx in recent_txs],
        })
    finally:
        db.close()


@app.get("/api/my-purchases")
async def api_my_purchases(uid: str = Query(...)):
    from app.database import SessionLocal, engine
    db = SessionLocal()
    try:
        user = _get_user_by_uid(uid, db)
        if not user:
            raise HTTPException(status_code=403)

        from app.strategy_models import StrategyMarketplace, UserStrategy
        from app.strategy_marketplace_ext import StrategyPurchase, init_marketplace_ext_tables
        init_marketplace_ext_tables(engine)

        purchases = (
            db.query(StrategyPurchase)
            .filter(StrategyPurchase.buyer_id == user.id, StrategyPurchase.status == "active")
            .order_by(StrategyPurchase.purchased_at.desc()).all()
        )
        result = []
        for p in purchases:
            listing = db.query(StrategyMarketplace).filter(StrategyMarketplace.id == p.listing_id).first()
            strat   = db.query(UserStrategy).filter(UserStrategy.id == p.cloned_strategy_id).first() if p.cloned_strategy_id else None
            result.append({
                "listing_id": p.listing_id, "title": listing.title if listing else "Unknown",
                "pricing_model": p.pricing_model, "amount_paid": p.amount_paid_usd,
                "purchased_at": p.purchased_at.isoformat() if p.purchased_at else None,
                "strategy_id": strat.id if strat else None,
                "strategy_status": strat.status if strat else None,
            })
        return JSONResponse(result)
    finally:
        db.close()


@app.post("/api/marketplace/{listing_id}/clone")
async def api_clone_strategy(listing_id: int, uid: str = Query(...)):
    return await api_purchase_strategy(listing_id, uid)


@app.post("/api/strategies/{strategy_id}/share")
async def api_share_strategy(strategy_id: int, uid: str = Query(...)):
    """Publish a strategy to the marketplace from the web portal."""
    from app.database import SessionLocal, engine
    db = SessionLocal()
    try:
        user = _get_user_by_uid(uid, db)
        if not user:
            raise HTTPException(status_code=403)

        from app.strategy_models import UserStrategy, StrategyMarketplace, init_strategy_tables
        init_strategy_tables(engine)

        strategy = db.query(UserStrategy).filter(
            UserStrategy.id == strategy_id,
            UserStrategy.user_id == user.id,
        ).first()
        if not strategy:
            raise HTTPException(status_code=404)

        existing = db.query(StrategyMarketplace).filter(
            StrategyMarketplace.strategy_id == strategy_id
        ).first()
        if existing:
            raise HTTPException(status_code=409, detail="Already published")

        from app.services.strategy_builder import generate_strategy_summary
        summary = await generate_strategy_summary(strategy.config)

        listing = StrategyMarketplace(
            strategy_id   = strategy_id,
            author_id     = user.id,
            title         = strategy.name,
            summary       = summary,
            tags          = [],
            category      = "general",
            pricing_model = "free",
            price_usdt    = 0.0,
        )
        db.add(listing)
        strategy.is_public = True
        db.commit()
        return JSONResponse({"success": True, "listing_id": listing.id})
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


@app.get("/api/strategies/templates")
async def api_strategy_templates():
    """Pre-built strategy templates the user can start from."""
    templates = [
        {
            "id": "fvg_bounce",
            "name": "FVG Bounce",
            "emoji": "🧲",
            "category": "smc",
            "tagline": "Smart Money Concepts — enter on fair value gap fills",
            "description": "Long when price pulls back into a bullish Fair Value Gap (FVG) created on the 15m chart. Entry confirmed when price enters the gap zone with RSI between 40-60, signaling momentum reset rather than breakdown.",
            "direction": "LONG",
            "leverage": 20,
            "position_size_pct": 5,
            "take_profit_pct": 5,
            "stop_loss_pct": 2.5,
            "take_profit2_pct": 8,
            "trailing_stop": False,
            "max_trades_per_day": 3,
            "cooldown_minutes": 90,
            "daily_loss_limit_pct": 8,
            "max_open_positions": 2,
            "difficulty": "Intermediate",
            "style": "Swing",
        },
        {
            "id": "rsi_reversal",
            "name": "RSI Oversold Reversal",
            "emoji": "📉",
            "category": "reversal",
            "tagline": "Buy extreme fear, ride the recovery",
            "description": "Long when RSI drops below 28 on the 15m timeframe and starts turning up. Requires the 1h RSI to also be below 45 to confirm the broader oversold context. Volume must be above average to confirm accumulation.",
            "direction": "LONG",
            "leverage": 10,
            "position_size_pct": 6,
            "take_profit_pct": 4,
            "stop_loss_pct": 2,
            "take_profit2_pct": 7,
            "trailing_stop": False,
            "max_trades_per_day": 4,
            "cooldown_minutes": 60,
            "daily_loss_limit_pct": 6,
            "max_open_positions": 3,
            "difficulty": "Beginner",
            "style": "Reversal",
        },
        {
            "id": "volume_scalp",
            "name": "Volume Spike Scalp",
            "emoji": "⚡",
            "category": "scalp",
            "tagline": "Ride the volume surge for quick 2-3%",
            "description": "Long when volume spikes 2x above the 20-period average on the 5m chart with a bullish candle body (close > open). RSI must be between 45-65 to avoid overbought entries. Tight TP and SL for clean R:R.",
            "direction": "LONG",
            "leverage": 15,
            "position_size_pct": 4,
            "take_profit_pct": 2,
            "stop_loss_pct": 1,
            "take_profit2_pct": None,
            "trailing_stop": False,
            "max_trades_per_day": 6,
            "cooldown_minutes": 30,
            "daily_loss_limit_pct": 5,
            "max_open_positions": 2,
            "difficulty": "Beginner",
            "style": "Scalp",
        },
        {
            "id": "macd_momentum",
            "name": "MACD Momentum Cross",
            "emoji": "📊",
            "category": "momentum",
            "tagline": "Classic crossover with trend confirmation",
            "description": "Enter long when MACD (8,21,5) crosses bullish on the 15m chart with the signal line turning up. EMA 21 must be above EMA 50 for trend alignment. Avoid entries if RSI is above 72 (overbought). Short the same setup inverted.",
            "direction": "BOTH",
            "leverage": 10,
            "position_size_pct": 5,
            "take_profit_pct": 3,
            "stop_loss_pct": 1.5,
            "take_profit2_pct": 5,
            "trailing_stop": False,
            "max_trades_per_day": 4,
            "cooldown_minutes": 60,
            "daily_loss_limit_pct": 6,
            "max_open_positions": 2,
            "difficulty": "Beginner",
            "style": "Momentum",
        },
        {
            "id": "bb_squeeze",
            "name": "Bollinger Squeeze Breakout",
            "emoji": "🔥",
            "category": "breakout",
            "tagline": "Low volatility squeezes lead to big moves",
            "description": "Enter when Bollinger Bands (20,2) squeeze tight for 10+ candles on the 15m chart and then price breaks out with volume 1.5x above average. Long when price breaks above upper band, short when it breaks below lower band.",
            "direction": "BOTH",
            "leverage": 15,
            "position_size_pct": 5,
            "take_profit_pct": 4,
            "stop_loss_pct": 2,
            "take_profit2_pct": 7,
            "trailing_stop": True,
            "trailing_stop_pct": 1.5,
            "max_trades_per_day": 3,
            "cooldown_minutes": 60,
            "daily_loss_limit_pct": 6,
            "max_open_positions": 2,
            "difficulty": "Intermediate",
            "style": "Breakout",
        },
        {
            "id": "pump_fade",
            "name": "Pump Fader",
            "emoji": "🩸",
            "category": "reversal",
            "tagline": "Short the over-extension, ride the dump",
            "description": "Short when a coin pumps 8% or more in 15 minutes on high volume with RSI above 80 on the 5m chart. The EMA 8 must be sharply above EMA 21 showing parabolic extension. Wait for the first 5m red candle to confirm reversal before entry.",
            "direction": "SHORT",
            "leverage": 10,
            "position_size_pct": 4,
            "take_profit_pct": 3,
            "stop_loss_pct": 1.5,
            "take_profit2_pct": 5,
            "trailing_stop": False,
            "max_trades_per_day": 4,
            "cooldown_minutes": 45,
            "daily_loss_limit_pct": 6,
            "max_open_positions": 2,
            "difficulty": "Intermediate",
            "style": "Reversal",
        },
        {
            "id": "ema_ribbon",
            "name": "EMA Ribbon Long",
            "emoji": "🎯",
            "category": "momentum",
            "tagline": "Trend-following with multi-EMA confluence",
            "description": "Long when the EMA 8, 21, and 50 are aligned in bullish order (8 > 21 > 50) on the 15m chart. Price must pull back to touch the EMA 21 and bounce with RSI between 45-62. This is a trend-continuation entry after a healthy pullback.",
            "direction": "LONG",
            "leverage": 12,
            "position_size_pct": 5,
            "take_profit_pct": 4,
            "stop_loss_pct": 2,
            "take_profit2_pct": 7,
            "trailing_stop": True,
            "trailing_stop_pct": 1,
            "max_trades_per_day": 4,
            "cooldown_minutes": 60,
            "daily_loss_limit_pct": 6,
            "max_open_positions": 3,
            "difficulty": "Beginner",
            "style": "Swing",
        },
        {
            "id": "support_bounce",
            "name": "Support Zone Bounce",
            "emoji": "🪃",
            "category": "reversal",
            "tagline": "Buy key support, tight SL below the zone",
            "description": "Long when price tests a major support level (previous swing low within ±1% on 1h chart) with RSI showing bullish divergence (price makes lower low but RSI makes higher low). Volume must confirm with a spike on the support candle.",
            "direction": "LONG",
            "leverage": 8,
            "position_size_pct": 7,
            "take_profit_pct": 6,
            "stop_loss_pct": 3,
            "take_profit2_pct": 10,
            "trailing_stop": False,
            "max_trades_per_day": 2,
            "cooldown_minutes": 120,
            "daily_loss_limit_pct": 6,
            "max_open_positions": 2,
            "difficulty": "Advanced",
            "style": "Swing",
        },
    ]
    return JSONResponse(templates)


@app.get("/api/strategies/{strategy_id}/analytics")
async def api_strategy_analytics(strategy_id: int, uid: str = Query(...)):
    """Advanced analytics: Sharpe, drawdown, profit factor, equity curve, health score."""
    from app.database import SessionLocal
    db = SessionLocal()
    try:
        user = _get_user_by_uid(uid, db)
        if not user:
            raise HTTPException(status_code=403)

        from app.strategy_models import UserStrategy, StrategyExecution, StrategyPerformance
        s = db.query(UserStrategy).filter(
            UserStrategy.id == strategy_id, UserStrategy.user_id == user.id
        ).first()
        if not s:
            raise HTTPException(status_code=404)

        execs = (
            db.query(StrategyExecution)
            .filter(StrategyExecution.strategy_id == strategy_id)
            .order_by(StrategyExecution.fired_at.asc())
            .all()
        )
        perf = db.query(StrategyPerformance).filter(StrategyPerformance.strategy_id == strategy_id).first()

        closed = [e for e in execs if e.outcome in ("WIN", "LOSS", "BREAKEVEN") and e.pnl_pct is not None]
        wins   = [e.pnl_pct for e in closed if e.outcome == "WIN"]
        losses = [e.pnl_pct for e in closed if e.outcome == "LOSS"]

        # Equity curve (cumulative P&L %)
        cumulative = 0.0
        equity_labels = []
        equity_values = []
        for e in closed:
            cumulative += (e.pnl_pct or 0)
            dt = (e.closed_at or e.fired_at)
            equity_labels.append(dt.strftime("%m/%d") if dt else "")
            equity_values.append(round(cumulative, 2))

        # Max drawdown
        peak = 0.0
        max_dd = 0.0
        for v in equity_values:
            if v > peak: peak = v
            dd = peak - v
            if dd > max_dd: max_dd = dd

        # Profit factor
        gross_win  = sum(wins)  if wins   else 0.0
        gross_loss = abs(sum(losses)) if losses else 0.0
        profit_factor = round(gross_win / gross_loss, 2) if gross_loss > 0 else (99.0 if gross_win > 0 else 0.0)

        # Sharpe ratio (simplified — treat each trade as 1 period)
        sharpe = 0.0
        if len(closed) >= 5:
            import statistics as _st
            pnls = [e.pnl_pct for e in closed]
            mean_r = _st.mean(pnls)
            std_r  = _st.stdev(pnls) if len(pnls) > 1 else 0
            sharpe = round((mean_r / std_r) * (252 ** 0.5), 2) if std_r > 0 else 0

        # Streak
        best_streak = worst_streak = cur_w = cur_l = 0
        for e in closed:
            if e.outcome == "WIN":
                cur_w += 1; cur_l = 0
                best_streak = max(best_streak, cur_w)
            elif e.outcome == "LOSS":
                cur_l += 1; cur_w = 0
                worst_streak = max(worst_streak, cur_l)

        # Per-coin breakdown
        coin_pnl = {}
        coin_trades = {}
        for e in closed:
            coin_pnl[e.symbol]    = round(coin_pnl.get(e.symbol, 0) + (e.pnl_pct or 0), 2)
            coin_trades[e.symbol] = coin_trades.get(e.symbol, 0) + 1
        top_coins = sorted(coin_pnl.items(), key=lambda x: -x[1])

        # Win rate by direction
        long_closed  = [e for e in closed if e.direction == "LONG"]
        short_closed = [e for e in closed if e.direction == "SHORT"]
        long_wr  = round(len([e for e in long_closed  if e.outcome == "WIN"]) / len(long_closed)  * 100, 1) if long_closed  else None
        short_wr = round(len([e for e in short_closed if e.outcome == "WIN"]) / len(short_closed) * 100, 1) if short_closed else None

        # Health score (0–10)
        wr_pct = perf.win_rate if perf else 0
        health = 0.0
        if len(closed) >= 3:
            health += min(wr_pct / 100, 1.0) * 4.0
            health += min(profit_factor / 2.0, 1.0) * 3.0
            health += min(max(sharpe, 0) / 2.0, 1.0) * 2.0
            health += min(len(closed) / 30.0, 1.0) * 1.0
        health = round(health, 1)

        return JSONResponse({
            "equity_curve":  {"labels": equity_labels, "values": equity_values},
            "profit_factor": profit_factor,
            "max_drawdown":  round(max_dd, 2),
            "sharpe_ratio":  sharpe,
            "best_streak":   best_streak,
            "worst_streak":  worst_streak,
            "avg_win_pct":   round(sum(wins) / len(wins), 2)   if wins   else 0,
            "avg_loss_pct":  round(sum(losses) / len(losses), 2) if losses else 0,
            "long_win_rate": long_wr,
            "short_win_rate": short_wr,
            "coin_breakdown": [{"symbol": s, "pnl": p, "trades": coin_trades[s]} for s, p in top_coins[:10]],
            "health_score":  health,
            "total_closed":  len(closed),
        })
    finally:
        db.close()


@app.get("/api/strategies/{strategy_id}/export")
async def api_export_trades(strategy_id: int, uid: str = Query(...)):
    """Download all strategy trades as CSV."""
    import csv, io
    from fastapi.responses import StreamingResponse
    from app.database import SessionLocal
    db = SessionLocal()
    try:
        user = _get_user_by_uid(uid, db)
        if not user:
            raise HTTPException(status_code=403)
        from app.strategy_models import UserStrategy, StrategyExecution
        s = db.query(UserStrategy).filter(
            UserStrategy.id == strategy_id, UserStrategy.user_id == user.id
        ).first()
        if not s:
            raise HTTPException(status_code=404)
        execs = (
            db.query(StrategyExecution)
            .filter(StrategyExecution.strategy_id == strategy_id)
            .order_by(StrategyExecution.fired_at.desc())
            .all()
        )
        buf = io.StringIO()
        w   = csv.writer(buf)
        w.writerow(["date", "symbol", "direction", "leverage", "entry_price", "exit_price", "outcome", "pnl_pct", "pnl_usd"])
        for e in execs:
            w.writerow([
                (e.fired_at.strftime("%Y-%m-%d %H:%M") if e.fired_at else ""),
                e.symbol, e.direction, e.leverage,
                e.entry_price or "", e.exit_price or "",
                e.outcome, e.pnl_pct or "", e.pnl_usd or "",
            ])
        buf.seek(0)
        filename = f"strategy_{strategy_id}_{s.name.replace(' ','_')}.csv"
        return StreamingResponse(
            iter([buf.getvalue()]),
            media_type="text/csv",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )
    finally:
        db.close()


@app.get("/api/portfolio")
async def api_portfolio(uid: str = Query(...)):
    """Portfolio-level metrics across all strategies."""
    from app.database import SessionLocal
    db = SessionLocal()
    try:
        user = _get_user_by_uid(uid, db)
        if not user:
            raise HTTPException(status_code=403)
        from app.strategy_models import UserStrategy, StrategyExecution, StrategyPerformance
        strategies = db.query(UserStrategy).filter(UserStrategy.user_id == user.id).all()
        all_execs  = []
        for strat in strategies:
            execs = db.query(StrategyExecution).filter(
                StrategyExecution.strategy_id == strat.id
            ).all()
            all_execs.extend(execs)

        closed = [e for e in all_execs if e.outcome in ("WIN", "LOSS", "BREAKEVEN") and e.pnl_pct is not None]
        wins   = len([e for e in closed if e.outcome == "WIN"])
        total  = len(closed)

        # Rolling 7-day P&L
        from datetime import datetime, timedelta
        cutoff_7d  = datetime.utcnow() - timedelta(days=7)
        cutoff_30d = datetime.utcnow() - timedelta(days=30)
        pnl_7d  = sum(e.pnl_pct for e in closed if (e.closed_at or e.fired_at) > cutoff_7d)
        pnl_30d = sum(e.pnl_pct for e in closed if (e.closed_at or e.fired_at) > cutoff_30d)
        pnl_all = sum(e.pnl_pct for e in closed)

        # Daily P&L breakdown (last 30 days for chart)
        from collections import defaultdict
        daily = defaultdict(float)
        for e in closed:
            if (e.closed_at or e.fired_at) > cutoff_30d:
                day = (e.closed_at or e.fired_at).strftime("%m/%d")
                daily[day] += e.pnl_pct or 0
        # Sort by date
        sorted_daily = sorted(daily.items())
        cumulative   = 0.0
        port_labels  = []
        port_values  = []
        for day, pnl in sorted_daily:
            cumulative += pnl
            port_labels.append(day)
            port_values.append(round(cumulative, 2))

        # Active strategies and exposure
        active = [s for s in strategies if s.status == "active"]
        open_trades = [e for e in all_execs if e.outcome == "OPEN"]

        return JSONResponse({
            "total_strategies": len(strategies),
            "active_count":     len(active),
            "open_trades":      len(open_trades),
            "total_trades":     total,
            "win_rate":         round(wins / total * 100, 1) if total > 0 else 0,
            "pnl_7d":           round(pnl_7d, 2),
            "pnl_30d":          round(pnl_30d, 2),
            "pnl_all":          round(pnl_all, 2),
            "equity_30d":       {"labels": port_labels, "values": port_values},
        })
    finally:
        db.close()


@app.get("/api/strategies/{strategy_id}/detail")
async def api_strategy_detail(strategy_id: int, uid: str = Query(...)):
    """Get full config for one strategy (for configure screen)."""
    from app.database import SessionLocal
    db = SessionLocal()
    try:
        user = _get_user_by_uid(uid, db)
        if not user:
            raise HTTPException(status_code=403)
        from app.strategy_models import UserStrategy, StrategyPerformance
        s = db.query(UserStrategy).filter(
            UserStrategy.id == strategy_id, UserStrategy.user_id == user.id
        ).first()
        if not s:
            raise HTTPException(status_code=404)
        perf = db.query(StrategyPerformance).filter(StrategyPerformance.strategy_id == s.id).first()
        return JSONResponse({
            "id":          s.id,
            "name":        s.name,
            "description": s.description,
            "status":      s.status,
            "config":      s.config or {},
            "performance": {
                "total_trades": perf.total_trades if perf else 0,
                "win_rate":     round(perf.win_rate, 1) if perf else 0,
                "total_pnl":    round(perf.total_pnl_pct, 2) if perf else 0,
                "best_trade":   round(perf.best_trade, 2) if perf else 0,
                "worst_trade":  round(perf.worst_trade, 2) if perf else 0,
                "wins":         perf.wins if perf else 0,
                "losses":       perf.losses if perf else 0,
            },
        })
    finally:
        db.close()


@app.put("/api/strategies/{strategy_id}")
async def api_update_strategy(strategy_id: int, request: Request):
    """Update a strategy's config (name, risk params, conditions, etc.)."""
    body = await request.json()
    uid  = body.get("uid")
    if not uid:
        raise HTTPException(status_code=400)

    from app.database import SessionLocal
    db = SessionLocal()
    try:
        user = _get_user_by_uid(uid, db)
        if not user:
            raise HTTPException(status_code=403)

        from app.strategy_models import UserStrategy
        s = db.query(UserStrategy).filter(
            UserStrategy.id == strategy_id, UserStrategy.user_id == user.id
        ).first()
        if not s:
            raise HTTPException(status_code=404)

        # Merge in top-level overrides
        config = dict(s.config or {})

        if "name" in body:
            s.name       = body["name"]
            config["name"] = body["name"]
        if "description" in body:
            s.description         = body["description"]
            config["description"] = body["description"]

        # Risk block
        risk = dict(config.get("risk", {}))
        for k in ("leverage", "position_size_pct", "max_trades_per_day", "cooldown_minutes",
                  "max_open_positions", "daily_loss_limit_pct"):
            if k in body:
                risk[k] = body[k]
        config["risk"] = risk

        # Exit block
        exit_ = dict(config.get("exit", {}))
        for k in ("take_profit_pct", "take_profit2_pct", "stop_loss_pct",
                  "trailing_stop", "trailing_stop_pct", "breakeven_at_pct"):
            if k in body:
                exit_[k] = body[k]
        config["exit"] = exit_

        # Direction / universe
        if "direction" in body:
            config["direction"] = body["direction"]
        if "universe" in body:
            config["universe"] = body["universe"]

        # Status change
        if "status" in body and body["status"] in ("draft", "active", "paused"):
            s.status = body["status"]

        s.config = config
        db.commit()
        return JSONResponse({"success": True, "id": s.id, "status": s.status})
    finally:
        db.close()


@app.get("/health")
async def health():
    return {"status": "ok", "service": "strategy-portal"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
