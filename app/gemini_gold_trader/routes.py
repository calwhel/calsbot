"""FastAPI routes + page for Gemini Vision Gold Trader."""
from __future__ import annotations

import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

from app.database import SessionLocal
from app.gemini_gold_trader import state as runtime_state
from app.gemini_gold_trader.accounts import (
    demo_accounts_for_user_id,
    find_demo_account,
    validate_demo_ctid_allowed,
)
from app.gemini_gold_trader.config import env_defaults, gemini_gold_enabled
from app.gemini_gold_trader.guardrails import (
    calls_today,
    cost_today_usd,
    demo_account_configured,
    merge_config,
    trades_today,
)
from app.gemini_gold_trader.models import GeminiGoldDecision
from app.gemini_gold_trader.schema import ensure_gemini_gold_trader_schema, seed_config_if_missing
from app.services.forex_sessions import gold_ai_session_hours

logger = logging.getLogger(__name__)

router = APIRouter()
templates = Jinja2Templates(directory="app/templates")


def _normalize_uid(uid: str) -> str:
    uid = (uid or "").strip().upper()
    if uid and not uid.startswith("TH-"):
        uid = f"TH-{uid}"
    return uid


def _gemini_gold_admin_uids() -> frozenset[str]:
    uids = {"TH-YP0BADA8", "TH-ZKJO6YKX"}
    raw = os.environ.get("GEMINI_GOLD_ADMIN_UIDS", "").strip()
    if not raw:
        raw = os.environ.get("GOLD_AI_ADMIN_UIDS", "").strip()
    for part in raw.split(","):
        part = part.strip().upper()
        if not part:
            continue
        if not part.startswith("TH-"):
            part = f"TH-{part}"
        uids.add(part)
    return frozenset(uids)


def _resolve_user(uid: str, db):
    from app.models import User

    uid = _normalize_uid(uid)
    u = db.query(User).filter(User.uid == uid).first()
    if not u:
        raise HTTPException(status_code=403, detail="Invalid UID")
    is_admin = bool(getattr(u, "is_admin", False)) or u.uid in _gemini_gold_admin_uids()
    if not is_admin:
        raise HTTPException(status_code=403, detail="Admin access required")
    return u


def _trader_user_id(cfg, admin_user) -> Optional[int]:
    return cfg.demo_user_id or getattr(admin_user, "id", None)


def _persist_demo_user_from_admin(row, admin) -> None:
    uid = getattr(admin, "id", None)
    if uid is not None:
        row.demo_user_id = int(uid)


def _config_payload(cfg, cfg_row) -> Dict[str, Any]:
    return {
        "enabled": cfg.enabled,
        "kill_switch": cfg.kill_switch,
        "dry_run": cfg.dry_run,
        "max_calls_day": cfg.max_calls_day,
        "max_trades_day": cfg.max_trades_day,
        "scan_interval_s": cfg.scan_interval_s,
        "model": cfg.model,
        "demo_ctrader_account_id": cfg.demo_ctrader_account_id,
        "demo_lot_size": cfg.demo_lot_size,
        "confidence_threshold": cfg.confidence_threshold,
        "env_enabled": gemini_gold_enabled(),
        "calls_reset_at": (
            cfg_row.calls_reset_at.isoformat()
            if cfg_row and getattr(cfg_row, "calls_reset_at", None)
            else None
        ),
    }


def _decision_feed(db, *, limit: int = 30) -> List[Dict[str, Any]]:
    rows = (
        db.query(GeminiGoldDecision)
        .order_by(GeminiGoldDecision.ts.desc())
        .limit(limit)
        .all()
    )
    out: List[Dict[str, Any]] = []
    for row in rows:
        d = row.decision if isinstance(row.decision, dict) else {}
        out.append(
            {
                "id": row.id,
                "ts": row.ts.isoformat() if row.ts else None,
                "session": row.session,
                "action": row.action,
                "direction": row.direction or d.get("direction"),
                "confidence": row.confidence,
                "rationale": row.rationale or d.get("rationale"),
                "executed": bool(row.executed),
                "execution_id": row.execution_id,
                "dry_run": bool(row.dry_run),
                "skip_reason": row.skip_reason,
                "cost_usd": float(row.cost_usd or 0),
                "entry": d.get("entry"),
                "stop_loss": d.get("stop_loss"),
                "take_profit": d.get("take_profit"),
            }
        )
    return out


def _session_token_for_page(request: Request, uid: str) -> Optional[str]:
    from app.portal_auth import session_token_from_request

    return session_token_from_request(request, uid)


def _page_auth_redirect(request: Request, uid: str):
    from app.portal_auth import login_redirect, normalize_portal_uid, resolve_session_uid_with_user

    norm_uid = normalize_portal_uid(uid)
    session_uid = resolve_session_uid_with_user(request)
    if not session_uid:
        return login_redirect(f"/gemini-gold-trader?uid={norm_uid}", msg="session_expired")
    if session_uid != norm_uid:
        return RedirectResponse(url=f"/gemini-gold-trader?uid={session_uid}", status_code=302)
    return None


@router.get("/gemini-gold-trader", response_class=HTMLResponse)
async def gemini_gold_trader_page(request: Request, uid: Optional[str] = Query(None)):
    from app.portal_auth import login_redirect, normalize_portal_uid, resolve_session_uid_with_user
    from app.portal_session import set_session_cookie

    session_uid = resolve_session_uid_with_user(request)
    if not uid:
        if session_uid:
            return RedirectResponse(url=f"/gemini-gold-trader?uid={session_uid}", status_code=302)
        return login_redirect("/gemini-gold-trader", msg="session_expired")

    norm_uid = normalize_portal_uid(uid)
    auth_redirect = _page_auth_redirect(request, norm_uid)
    if auth_redirect is not None:
        return auth_redirect

    session_token = _session_token_for_page(request, norm_uid) or ""
    resp = templates.TemplateResponse(
        "gemini_gold_trader.html",
        {
            "request": request,
            "uid": norm_uid,
            "session_token": session_token,
        },
    )
    try:
        set_session_cookie(resp, norm_uid, request)
    except Exception as exc:
        logger.warning("[gemini-gold] refresh session cookie failed: %s", exc)
    resp.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    return resp


@router.get("/api/gemini-gold-trader/status")
async def api_status(uid: str = Query(...)):
    norm_uid = _normalize_uid(uid)
    db = SessionLocal()
    try:
        admin = _resolve_user(norm_uid, db)
        ensure_gemini_gold_trader_schema()
        row = seed_config_if_missing(db)
        if not row.demo_user_id:
            _persist_demo_user_from_admin(row, admin)
            db.commit()
            db.refresh(row)
        env = env_defaults()
        cfg = merge_config(row, env)
        trader_uid = _trader_user_id(cfg, admin)

        demo_accounts = demo_accounts_for_user_id(db, trader_uid)
        selected = find_demo_account(demo_accounts, cfg.demo_ctrader_account_id)

        return {
            "ok": True,
            "demo_label": "[Gemini Gold] Demo",
            "runtime": runtime_state.get_status(),
            "shared_session_hours": gold_ai_session_hours(),
            "config": _config_payload(cfg, row),
            "demo_accounts": demo_accounts,
            "demo_account_selected": selected,
            "demo_account_ready": demo_account_configured(cfg),
            "stats_today": {
                "calls": calls_today(db),
                "trades": trades_today(db),
                "cost_usd": round(cost_today_usd(db), 6),
            },
            "decision_feed": _decision_feed(db),
        }
    finally:
        db.close()


@router.post("/api/gemini-gold-trader/config")
async def api_update_config(request: Request, uid: str = Query(...)):
    db = SessionLocal()
    try:
        admin = _resolve_user(uid, db)
        body = await request.json()
        row = seed_config_if_missing(db)
        env = env_defaults()
        trader_uid = _trader_user_id(merge_config(row, env), admin)

        if "demo_ctrader_account_id" in body:
            raw = body.get("demo_ctrader_account_id")
            ctid = str(raw).strip() if raw not in (None, "") else ""
            if ctid:
                demo_list = demo_accounts_for_user_id(db, trader_uid)
                try:
                    validate_demo_ctid_allowed(demo_list, ctid)
                except ValueError as e:
                    raise HTTPException(status_code=400, detail=str(e)) from e
                row.demo_ctrader_account_id = ctid
            else:
                row.demo_ctrader_account_id = None

        for field in (
            "enabled",
            "kill_switch",
            "dry_run",
            "max_calls_day",
            "max_trades_day",
            "demo_lot_size",
            "confidence_threshold",
        ):
            if field in body:
                setattr(row, field, body[field])

        cfg_after = merge_config(row, env)
        if body.get("enabled") is True and not demo_account_configured(cfg_after):
            raise HTTPException(
                status_code=400,
                detail="Select a demo cTrader account before enabling the trader",
            )
        if body.get("enabled") is True and not gemini_gold_enabled():
            raise HTTPException(
                status_code=400,
                detail="Set GEMINI_GOLD_ENABLED=true in Railway to start the scan process",
            )

        if (
            body.get("enabled") is True
            or "demo_ctrader_account_id" in body
            or not row.demo_user_id
        ):
            _persist_demo_user_from_admin(row, admin)

        row.updated_at = datetime.utcnow()
        db.commit()
        return {"ok": True}
    finally:
        db.close()


@router.post("/api/gemini-gold-trader/kill-switch")
async def api_kill_switch(request: Request, uid: str = Query(...)):
    db = SessionLocal()
    try:
        _resolve_user(uid, db)
        body = await request.json()
        on = bool(body.get("on", True))
        row = seed_config_if_missing(db)
        row.kill_switch = on
        db.commit()
        runtime_state.note_dormant("kill_switch" if on else "outside_session")
        return {"ok": True, "kill_switch": on}
    finally:
        db.close()
