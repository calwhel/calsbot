"""FastAPI routes + page for Gold AI Trader."""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates

from app.database import SessionLocal
from app.gold_ai_trader.config import env_defaults
from app.gold_ai_trader.guardrails import (
    calls_today,
    trades_today,
    cost_today_usd,
    merge_config,
)
from app.gold_ai_trader.models import GoldAiConfig, GoldAiDecision, GoldAiLesson
from app.gold_ai_trader.schema import ensure_gold_ai_trader_schema, seed_config_if_missing
from app.gold_ai_trader import state as runtime_state

logger = logging.getLogger(__name__)

router = APIRouter()
templates = Jinja2Templates(directory="app/templates")


def _normalize_uid(uid: str) -> str:
    """Match Strategy Portal UID format (TH-XXXXXXXX)."""
    uid = (uid or "").strip().upper()
    if uid and not uid.startswith("TH-"):
        uid = f"TH-{uid}"
    return uid


def _resolve_user(uid: str, db):
    from app.models import User

    uid = _normalize_uid(uid)
    u = db.query(User).filter(User.uid == uid).first()
    if not u:
        raise HTTPException(status_code=403, detail="Invalid UID")
    is_admin = bool(getattr(u, "is_admin", False)) or u.uid == "TH-YP0BADA8"
    if not is_admin:
        raise HTTPException(status_code=403, detail="Admin access required")
    return u


@router.get("/gold-ai-trader", response_class=HTMLResponse)
async def gold_ai_trader_page(request: Request, uid: str = Query(...)):
    db = SessionLocal()
    try:
        _resolve_user(uid, db)
    finally:
        db.close()
    return templates.TemplateResponse(
        "gold_ai_trader.html",
        {"request": request, "uid": uid},
    )


@router.get("/api/gold-ai-trader/status")
async def api_status(uid: str = Query(...)):
    ensure_gold_ai_trader_schema()
    db = SessionLocal()
    try:
        _resolve_user(uid, db)
        cfg_row = seed_config_if_missing(db)
        cfg = merge_config(cfg_row, env_defaults())
        lessons = (
            db.query(GoldAiLesson)
            .order_by(GoldAiLesson.ts.desc())
            .limit(3)
            .all()
        )
        decisions = (
            db.query(GoldAiDecision)
            .order_by(GoldAiDecision.ts.desc())
            .limit(40)
            .all()
        )
        feed = []
        for d in decisions:
            feed.append(
                {
                    "id": d.id,
                    "ts": d.ts.isoformat() if d.ts else None,
                    "session": d.session,
                    "candidate_type": d.candidate_type,
                    "action": d.action,
                    "confidence": d.confidence,
                    "executed": d.executed,
                    "execution_id": d.execution_id,
                    "cost_usd": d.cost_usd,
                    "rationale": (d.decision or {}).get("rationale", ""),
                    "reasoning_preview": (d.reasoning or "")[:300],
                }
            )
        return {
            "ok": True,
            "demo_label": "DEMO ACCOUNT ONLY",
            "runtime": runtime_state.get_status(),
            "config": {
                "enabled": cfg.enabled,
                "kill_switch": cfg.kill_switch,
                "london_start_hour": cfg.london_start_hour,
                "london_end_hour": cfg.london_end_hour,
                "ny_start_hour": cfg.ny_start_hour,
                "ny_end_hour": cfg.ny_end_hour,
                "max_calls_day": cfg.max_calls_day,
                "max_trades_day": cfg.max_trades_day,
                "no_overnight": cfg.no_overnight,
                "model": cfg.model,
                "demo_ctrader_account_id": cfg.demo_ctrader_account_id,
            },
            "stats_today": {
                "calls": calls_today(db),
                "trades": trades_today(db),
                "cost_usd": round(cost_today_usd(db), 4),
            },
            "lessons": [
                {"session": x.session, "ts": x.ts.isoformat(), "digest": x.digest}
                for x in lessons
            ],
            "decision_feed": feed,
        }
    finally:
        db.close()


@router.post("/api/gold-ai-trader/config")
async def api_update_config(request: Request, uid: str = Query(...)):
    db = SessionLocal()
    try:
        _resolve_user(uid, db)
        body = await request.json()
        row = seed_config_if_missing(db)
        for field in (
            "enabled",
            "kill_switch",
            "london_start_hour",
            "london_end_hour",
            "ny_start_hour",
            "ny_end_hour",
            "max_calls_day",
            "max_trades_day",
            "no_overnight",
            "model",
            "demo_ctrader_account_id",
        ):
            if field in body:
                setattr(row, field, body[field])
        row.updated_at = datetime.utcnow()
        db.commit()
        return {"ok": True}
    finally:
        db.close()


@router.post("/api/gold-ai-trader/kill-switch")
async def api_kill_switch(request: Request, uid: str = Query(...)):
    db = SessionLocal()
    try:
        _resolve_user(uid, db)
        body = await request.json()
        on = bool(body.get("on", True))
        row = seed_config_if_missing(db)
        row.kill_switch = on
        db.commit()
        runtime_state.set_status(status="killed" if on else "dormant")
        return {"ok": True, "kill_switch": on}
    finally:
        db.close()
