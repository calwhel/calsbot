"""FastAPI routes + page for Gold AI Trader."""
from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from app.database import SessionLocal
from app.gold_ai_trader.accounts import (
    demo_accounts_for_user_id,
    find_demo_account,
    validate_demo_ctid_allowed,
)
from app.gold_ai_trader.config import env_defaults
from app.services.forex_sessions import gold_ai_session_hours
from app.gold_ai_trader.guardrails import (
    calls_today,
    trades_today,
    cost_today_usd,
    demo_account_configured,
    demo_pnl_today_usd,
    live_pnl_today_usd,
    live_trades_today,
    merge_config,
    reset_daily_claude_credits,
    resolve_live_mirror_status,
)
from app.gold_ai_trader.learning import get_setup_stats
from app.gold_ai_trader.call_gates import call_stats_today
from app.gold_ai_trader.funnel_persist import recent_funnel_events
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


def _trader_user_id(cfg, admin_user) -> Optional[int]:
    return cfg.demo_user_id or getattr(admin_user, "id", None)


def _persist_demo_user_from_admin(row, admin) -> None:
    """Bind background loop to logged-in admin — no GOLD_AI_TRADER_USER_ID env required."""
    uid = getattr(admin, "id", None)
    if uid is not None:
        row.demo_user_id = int(uid)


def _live_accounts_for_user(db, user_id: Optional[int]) -> List[Dict[str, Any]]:
    if not user_id:
        return []
    from app.models import UserPreference

    prefs = db.query(UserPreference).filter(UserPreference.user_id == user_id).first()
    if not prefs or not prefs.ctrader_accounts:
        return []
    try:
        accounts = json.loads(prefs.ctrader_accounts)
    except (TypeError, json.JSONDecodeError):
        return []
    out = []
    for acct in accounts:
        if not acct.get("isLive"):
            continue
        ctid = acct.get("ctidTraderAccountId")
        if ctid is None:
            continue
        out.append(
            {
                "ctid": str(ctid),
                "trader_login": acct.get("traderLogin"),
                "balance": acct.get("balance"),
            }
        )
    return out


def _coerce_decision_dict(raw: Any) -> Dict[str, Any]:
    """GoldAiDecision.decision may be dict or legacy JSON string."""
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str) and raw.strip():
        try:
            parsed = json.loads(raw)
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            return {}
    return {}


def _safe_funnel_events(db, *, limit: int = 50) -> list:
    try:
        return recent_funnel_events(db, limit=limit)
    except Exception as exc:
        logger.warning("[gold-ai-trader] funnel events load failed: %s", exc)
        return []


def _sync_live_mirror_fields(db, decision: GoldAiDecision) -> None:
    if not decision.live_mirror_execution_id:
        return
    from app.strategy_models import StrategyExecution

    ex = db.query(StrategyExecution).filter_by(id=decision.live_mirror_execution_id).first()
    status, err = resolve_live_mirror_status(ex)
    if status != decision.live_mirror_status or err != decision.live_mirror_error:
        decision.live_mirror_status = status
        decision.live_mirror_error = err
        db.commit()


def _session_token_for_page(request: Request, uid: str) -> Optional[str]:
    """Session token for same-origin / WebView API calls (must match requested UID)."""
    from app.portal_session import make_session_token, session_uid_from_request

    uid = _normalize_uid(uid)
    session_uid = session_uid_from_request(request)
    if session_uid == uid:
        auth = request.headers.get("Authorization", "")
        if auth.lower().startswith("bearer "):
            return auth[7:].strip()
        hdr = request.headers.get("X-TradeHub-Session", "").strip()
        if hdr:
            return hdr
        cookie = request.cookies.get("th_session")
        if cookie:
            return cookie
    return make_session_token(uid)


@router.get("/gold-ai-trader", response_class=HTMLResponse)
async def gold_ai_trader_page(request: Request, uid: str = Query(...)):
    """Serve dashboard shell — admin auth enforced on /api/gold-ai-trader/* only."""
    from app.portal_session import make_session_token, set_session_cookie

    norm_uid = _normalize_uid(uid)
    session_token = _session_token_for_page(request, norm_uid) or make_session_token(norm_uid)
    resp = templates.TemplateResponse(
        "gold_ai_trader.html",
        {
            "request": request,
            "uid": norm_uid,
            "session_token": session_token,
        },
    )
    try:
        set_session_cookie(resp, norm_uid, request)
    except Exception as exc:
        logger.warning("[gold-ai-trader] set session cookie failed: %s", exc)
    return resp


@router.get("/api/gold-ai-trader/status")
async def api_status(uid: str = Query(...)):
    try:
        ensure_gold_ai_trader_schema()
    except Exception as exc:
        logger.warning("[gold-ai-trader] schema ensure on status: %s", exc)

    db = SessionLocal()
    try:
        admin = _resolve_user(uid, db)
        cfg_row = seed_config_if_missing(db)
        cfg = merge_config(cfg_row, env_defaults())
        trader_uid = _trader_user_id(cfg, admin)
        demo_accounts = demo_accounts_for_user_id(db, trader_uid)
        selected = find_demo_account(demo_accounts, cfg.demo_ctrader_account_id)
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
            try:
                _sync_live_mirror_fields(db, d)
            except Exception as exc:
                logger.debug("[gold-ai-trader] live mirror sync skip id=%s: %s", d.id, exc)
            dec = _coerce_decision_dict(d.decision)
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
                    "live_mirror_execution_id": d.live_mirror_execution_id,
                    "live_mirror_status": d.live_mirror_status,
                    "live_mirror_error": d.live_mirror_error,
                    "cost_usd": d.cost_usd,
                    "rationale": dec.get("rationale", ""),
                    "reasoning_preview": (d.reasoning or "")[:300],
                }
            )
        user_id = cfg.demo_user_id or getattr(admin, "id", None) or 0

        stats_today: Dict[str, Any] = {
            "calls": 0,
            "trades": 0,
            "cost_usd": 0.0,
            "demo_pnl_usd": 0.0,
            "live_pnl_usd": 0.0,
            "live_trades": 0,
        }
        try:
            stats_today = {
                "calls": calls_today(db),
                "trades": trades_today(db),
                "cost_usd": round(cost_today_usd(db), 4),
                "demo_pnl_usd": demo_pnl_today_usd(db, int(user_id)),
                "live_pnl_usd": live_pnl_today_usd(db, int(user_id)),
                "live_trades": live_trades_today(db),
            }
        except Exception as exc:
            logger.warning("[gold-ai-trader] stats_today failed: %s", exc)

        setup_stats: List[Dict[str, Any]] = []
        try:
            setup_stats = get_setup_stats(db)
        except Exception as exc:
            logger.warning("[gold-ai-trader] setup_stats failed: %s", exc)

        call_stats: Dict[str, int] = {}
        try:
            call_stats = call_stats_today(db)
        except Exception as exc:
            logger.warning("[gold-ai-trader] call_stats_today failed: %s", exc)

        return {
            "ok": True,
            "demo_label": "DEMO ACCOUNT ONLY",
            "runtime": runtime_state.get_status(),
            "shared_session_hours": gold_ai_session_hours(),
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
                "live_mirror_enabled": cfg.live_mirror_enabled,
                "live_ctrader_account_id": cfg.live_ctrader_account_id,
                "live_lot_size": cfg.live_lot_size,
                "demo_lot_size": cfg.demo_lot_size,
                "max_live_trades_day": cfg.max_live_trades_day,
                "use_limit_entry": cfg.use_limit_entry,
                "pending_entry_timeout_min": cfg.pending_entry_timeout_min,
                "learning_daily_at_ny_end": cfg.learning_daily_at_ny_end,
                "calls_reset_at": (
                    cfg_row.calls_reset_at.isoformat()
                    if getattr(cfg_row, "calls_reset_at", None)
                    else None
                ),
                "live_mirror_confirmed_at": (
                    cfg_row.live_mirror_confirmed_at.isoformat()
                    if getattr(cfg_row, "live_mirror_confirmed_at", None)
                    else None
                ),
            },
            "demo_accounts": demo_accounts,
            "demo_account_selected": selected,
            "demo_account_ready": demo_account_configured(cfg),
            "live_accounts": _live_accounts_for_user(db, trader_uid),
            "stats_today": stats_today,
            "lessons": [
                {"session": x.session, "ts": x.ts.isoformat(), "digest": x.digest}
                for x in lessons
            ],
            "setup_stats": setup_stats,
            "call_stats_today": call_stats,
            "recent_funnel_events": _safe_funnel_events(db, limit=40),
            "decision_feed": feed,
        }
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("[gold-ai-trader] status API failed uid=%s", uid)
        raise HTTPException(status_code=503, detail="Trader status temporarily unavailable") from exc
    finally:
        db.close()


@router.get("/api/gold-ai-trader/setup-stats")
async def api_setup_stats(uid: str = Query(...), days: int = Query(14)):
    ensure_gold_ai_trader_schema()
    db = SessionLocal()
    try:
        _resolve_user(uid, db)
        return {"ok": True, "days": days, "stats": get_setup_stats(db, days=days)}
    finally:
        db.close()


@router.get("/api/gold-ai-trader/funnel-events")
async def api_funnel_events(uid: str = Query(...), limit: int = Query(50)):
    ensure_gold_ai_trader_schema()
    db = SessionLocal()
    try:
        _resolve_user(uid, db)
        from app.gold_ai_trader.funnel import snapshot as funnel_snapshot

        return {
            "ok": True,
            "funnel": funnel_snapshot(),
            "events": _safe_funnel_events(db, limit=min(limit, 200)),
        }
    finally:
        db.close()


@router.get("/api/gold-ai-trader/calibration")
async def api_calibration(uid: str = Query(...), days: int = Query(14)):
    """Funnel + setup stats for pipeline tuning."""
    ensure_gold_ai_trader_schema()
    db = SessionLocal()
    try:
        _resolve_user(uid, db)
        from app.gold_ai_trader.funnel import snapshot as funnel_snapshot

        return {
            "ok": True,
            "funnel": funnel_snapshot(),
            "setup_stats": get_setup_stats(db, days=days),
            "call_stats_today": call_stats_today(db),
            "recent_funnel_events": _safe_funnel_events(db, limit=60),
        }
    finally:
        db.close()


@router.post("/api/gold-ai-trader/config")
async def api_update_config(request: Request, uid: str = Query(...)):
    db = SessionLocal()
    try:
        admin = _resolve_user(uid, db)
        body = await request.json()
        row = seed_config_if_missing(db)
        env = env_defaults()
        trader_uid = _trader_user_id(merge_config(row, env), admin)

        enabling_live = (
            body.get("live_mirror_enabled") is True
            and not bool(getattr(row, "live_mirror_enabled", False))
        )
        if enabling_live:
            if not body.get("confirm_real_money"):
                raise HTTPException(
                    status_code=400,
                    detail="confirm_real_money required to enable live mirror",
                )
            if not body.get("live_ctrader_account_id") and not row.live_ctrader_account_id:
                raise HTTPException(
                    status_code=400,
                    detail="live_ctrader_account_id required to enable live mirror",
                )

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
            "london_start_hour",
            "london_end_hour",
            "ny_start_hour",
            "ny_end_hour",
            "max_calls_day",
            "max_trades_day",
            "no_overnight",
            "model",
            "live_mirror_enabled",
            "live_ctrader_account_id",
            "live_lot_size",
            "demo_lot_size",
            "max_live_trades_day",
            "use_limit_entry",
            "pending_entry_timeout_min",
            "learning_daily_at_ny_end",
        ):
            if field in body:
                setattr(row, field, body[field])

        cfg_after = merge_config(row, env)
        if body.get("enabled") is True and not demo_account_configured(cfg_after):
            raise HTTPException(
                status_code=400,
                detail="Select a demo cTrader account before enabling the trader",
            )

        if enabling_live:
            row.live_mirror_confirmed_at = datetime.utcnow()
        if body.get("live_mirror_enabled") is False:
            row.live_mirror_confirmed_at = None

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


@router.post("/api/gold-ai-trader/disconnect-live")
async def api_disconnect_live(uid: str = Query(...)):
    """Disable live mirror only — demo trader keeps running."""
    db = SessionLocal()
    try:
        _resolve_user(uid, db)
        row = seed_config_if_missing(db)
        row.live_mirror_enabled = False
        row.live_mirror_confirmed_at = None
        row.updated_at = datetime.utcnow()
        db.commit()
        return {"ok": True, "live_mirror_enabled": False}
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


@router.post("/api/gold-ai-trader/reset-daily-credits")
async def api_reset_daily_credits(uid: str = Query(...)):
    """Reset today's Claude call/cost counters (decision history is kept)."""
    ensure_gold_ai_trader_schema()
    db = SessionLocal()
    try:
        _resolve_user(uid, db)
        reset_at = reset_daily_claude_credits(db)
        return {
            "ok": True,
            "calls_reset_at": reset_at.isoformat(),
            "stats_today": {
                "calls": calls_today(db),
                "cost_usd": round(cost_today_usd(db), 4),
            },
        }
    finally:
        db.close()
