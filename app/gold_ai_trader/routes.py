"""FastAPI routes + page for Gold AI Trader."""
from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from app.database import SessionLocal
from app.db_resilience import is_db_connection_error
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
_STATUS_RECONCILE_LAST_RUN: Dict[int, float] = {}
_STATUS_RECONCILE_INTERVAL_S = max(
    5.0,
    float(os.environ.get("GOLD_AI_STATUS_RECONCILE_INTERVAL_S", "12")),
)
_STATUS_RECONCILE_TIMEOUT_S = max(
    2.0,
    float(os.environ.get("GOLD_AI_STATUS_RECONCILE_TIMEOUT_S", "6")),
)


def _should_run_status_reconcile(user_id: int) -> bool:
    now = time.monotonic()
    last = _STATUS_RECONCILE_LAST_RUN.get(int(user_id), 0.0)
    if now - last < _STATUS_RECONCILE_INTERVAL_S:
        return False
    _STATUS_RECONCILE_LAST_RUN[int(user_id)] = now
    if len(_STATUS_RECONCILE_LAST_RUN) > 512:
        cutoff = now - (_STATUS_RECONCILE_INTERVAL_S * 4.0)
        for uid in list(_STATUS_RECONCILE_LAST_RUN):
            if _STATUS_RECONCILE_LAST_RUN[uid] < cutoff:
                _STATUS_RECONCILE_LAST_RUN.pop(uid, None)
    return True


def _normalize_uid(uid: str) -> str:
    """Match Strategy Portal UID format (TH-XXXXXXXX)."""
    uid = (uid or "").strip().upper()
    if uid and not uid.startswith("TH-"):
        uid = f"TH-{uid}"
    return uid


def _gold_ai_admin_uids() -> frozenset[str]:
    # Built-in owner + current operator UID; can be extended via env var.
    uids = {"TH-YP0BADA8", "TH-ZKJO6YKX"}
    raw = os.environ.get("GOLD_AI_ADMIN_UIDS", "").strip()
    for part in raw.split(","):
        part = part.strip().upper()
        if not part:
            continue
        if not part.startswith("TH-"):
            part = f"TH-{part}"
        uids.add(part)
    return frozenset(uids)


def _offline_admin(uid: str):
    return type("_OfflineAdmin", (), {"id": None, "uid": uid, "is_admin": True})()


def _session_admin_fallback(request: Request, uid: str):
    """When Neon is down, trust HMAC session token for allowlisted admin UIDs."""
    from app.portal_session import session_uid_from_request

    uid = _normalize_uid(uid)
    if session_uid_from_request(request) != uid:
        return None
    if uid not in _gold_ai_admin_uids():
        return None
    logger.warning("[gold-ai-trader] DB unavailable — session auth fallback uid=%s", uid)
    return _offline_admin(uid)


def _resolve_user(uid: str, db, *, request: Optional[Request] = None):
    from app.models import User

    uid = _normalize_uid(uid)
    try:
        u = db.query(User).filter(User.uid == uid).first()
        if not u:
            raise HTTPException(status_code=403, detail="Invalid UID")
        is_admin = bool(getattr(u, "is_admin", False)) or u.uid in _gold_ai_admin_uids()
        if not is_admin:
            raise HTTPException(status_code=403, detail="Admin access required")
        return u
    except HTTPException:
        raise
    except Exception as exc:
        if request is not None and is_db_connection_error(exc):
            admin = _session_admin_fallback(request, uid)
            if admin is not None:
                return admin
        raise


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


def _safe_lessons(db, *, limit: int = 3) -> list:
    try:
        return (
            db.query(GoldAiLesson)
            .order_by(GoldAiLesson.ts.desc())
            .limit(limit)
            .all()
        )
    except Exception as exc:
        logger.warning("[gold-ai-trader] lessons load failed: %s", exc)
        try:
            db.rollback()
        except Exception:
            pass
        return []


def _safe_decisions(db, *, limit: int = 40) -> list:
    try:
        return (
            db.query(GoldAiDecision)
            .order_by(GoldAiDecision.ts.desc())
            .limit(limit)
            .all()
        )
    except Exception as exc:
        logger.warning("[gold-ai-trader] decisions load failed: %s", exc)
        try:
            db.rollback()
        except Exception:
            pass
        return []


def _load_config_for_status(db, admin) -> tuple[Any, Any, Optional[Any]]:
    """Return (cfg_row, merged_cfg, trader_user_id). Falls back to env defaults."""
    env = env_defaults()
    try:
        row = seed_config_if_missing(db)
        if not row.demo_user_id:
            _persist_demo_user_from_admin(row, admin)
            db.commit()
            db.refresh(row)
        cfg = merge_config(row, env)
        return row, cfg, _trader_user_id(cfg, admin)
    except Exception as exc:
        logger.warning("[gold-ai-trader] config load failed: %s", exc)
        try:
            db.rollback()
        except Exception:
            pass
        try:
            ensure_gold_ai_trader_schema(force=True)
            row = seed_config_if_missing(db)
            if not row.demo_user_id:
                _persist_demo_user_from_admin(row, admin)
                db.commit()
                db.refresh(row)
            cfg = merge_config(row, env)
            return row, cfg, _trader_user_id(cfg, admin)
        except Exception as retry_exc:
            logger.warning("[gold-ai-trader] config retry failed: %s", retry_exc)
            try:
                db.rollback()
            except Exception:
                pass
            cfg = env
            cfg.demo_user_id = cfg.demo_user_id or getattr(admin, "id", None)
            return None, cfg, cfg.demo_user_id or getattr(admin, "id", None)


def _config_payload(cfg, cfg_row) -> Dict[str, Any]:
    return {
        "enabled": cfg.enabled,
        "kill_switch": cfg.kill_switch,
        "london_start_hour": cfg.london_start_hour,
        "london_end_hour": cfg.london_end_hour,
        "ny_start_hour": cfg.ny_start_hour,
        "ny_end_hour": cfg.ny_end_hour,
        "max_calls_day": cfg.max_calls_day,
        "max_trades_day": cfg.max_trades_day,
        "scan_interval_s": cfg.scan_interval_s,
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
        "orb_enabled": cfg.orb_enabled,
        "orb_range_minutes": cfg.orb_range_minutes,
        "orb_trade_window_minutes": cfg.orb_trade_window_minutes,
        "orb_timeframe": cfg.orb_timeframe,
        "orb_confirmation": cfg.orb_confirmation,
        "orb_require_retest": cfg.orb_require_retest,
        "orb_retest_max_bars": cfg.orb_retest_max_bars,
        "orb_retest_tol_atr": cfg.orb_retest_tol_atr,
        "orb_fakeout_filter": cfg.orb_fakeout_filter,
        "orb_break_buffer_atr": cfg.orb_break_buffer_atr,
        "orb_break_buffer_range_pct": cfg.orb_break_buffer_range_pct,
        "orb_min_break_body_atr": cfg.orb_min_break_body_atr,
        "orb_min_range_atr": cfg.orb_min_range_atr,
        "orb_max_range_atr": cfg.orb_max_range_atr,
        "orb_sl_mode": cfg.orb_sl_mode,
        "orb_sl_atr_mult": cfg.orb_sl_atr_mult,
        "orb_sl_range_buffer_atr": cfg.orb_sl_range_buffer_atr,
        "orb_tp_mode": cfg.orb_tp_mode,
        "orb_tp_range_mult": cfg.orb_tp_range_mult,
        "orb_tp_rr": cfg.orb_tp_rr,
        "orb_confidence_threshold": cfg.orb_confidence_threshold,
        "orb_max_calls_day": cfg.orb_max_calls_day,
        "orb_min_global_calls_left": cfg.orb_min_global_calls_left,
        "orb_max_trades_per_session": cfg.orb_max_trades_per_session,
        "orb_entry_max_break_atr": cfg.orb_entry_max_break_atr,
        "orb_entry_max_break_range_pct": cfg.orb_entry_max_break_range_pct,
        "calls_reset_at": (
            cfg_row.calls_reset_at.isoformat()
            if cfg_row and getattr(cfg_row, "calls_reset_at", None)
            else None
        ),
        "live_mirror_confirmed_at": (
            cfg_row.live_mirror_confirmed_at.isoformat()
            if cfg_row and getattr(cfg_row, "live_mirror_confirmed_at", None)
            else None
        ),
    }


def _empty_stats_today() -> Dict[str, Any]:
    return {
        "calls": 0,
        "trades": 0,
        "cost_usd": 0.0,
        "demo_pnl_usd": 0.0,
        "live_pnl_usd": 0.0,
        "live_trades": 0,
    }


def _offline_status_payload(
    admin,
    *,
    degraded: Optional[List[str]] = None,
    db_message: Optional[str] = None,
) -> Dict[str, Any]:
    """Minimal dashboard payload when Neon is waking or unreachable."""
    env = env_defaults()
    env.demo_user_id = env.demo_user_id or getattr(admin, "id", None)
    parts = list(degraded or [])
    if "database" not in parts:
        parts.insert(0, "database")
    payload: Dict[str, Any] = {
        "ok": True,
        "db_unavailable": True,
        "db_message": db_message or "Database waking up — tap refresh in a moment.",
        "degraded": parts,
        "demo_label": "DEMO ACCOUNT ONLY",
        "runtime": runtime_state.get_status(),
        "shared_session_hours": gold_ai_session_hours(),
        "config": _config_payload(env, None),
        "demo_accounts": [],
        "demo_account_selected": None,
        "demo_account_ready": demo_account_configured(env),
        "live_accounts": [],
        "stats_today": _empty_stats_today(),
        "lessons": [],
        "setup_stats": [],
        "call_stats_today": {},
        "recent_funnel_events": [],
        "decision_feed": [],
    }
    return payload


def _build_decision_feed(db, decisions: list) -> list:
    execution_map: Dict[int, Dict[str, Optional[str]]] = {}
    execution_ids = [int(d.execution_id) for d in decisions if getattr(d, "execution_id", None)]
    if execution_ids:
        try:
            from app.strategy_models import StrategyExecution

            rows = (
                db.query(
                    StrategyExecution.id,
                    StrategyExecution.outcome,
                    StrategyExecution.closed_at,
                )
                .filter(StrategyExecution.id.in_(execution_ids))
                .all()
            )
            for r in rows:
                execution_map[int(r.id)] = {
                    "outcome": ((r.outcome or "").strip().upper() or None),
                    "closed_ts": r.closed_at.isoformat() if r.closed_at else None,
                }
        except Exception as exc:
            logger.debug("[gold-ai-trader] execution map load failed: %s", exc)

    feed = []
    for d in decisions:
        try:
            _sync_live_mirror_fields(db, d)
        except Exception as exc:
            logger.debug("[gold-ai-trader] live mirror sync skip id=%s: %s", d.id, exc)
        dec = _coerce_decision_dict(d.decision)
        execution_meta = execution_map.get(int(d.execution_id)) if d.execution_id else None
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
                "execution_outcome": (execution_meta or {}).get("outcome"),
                "execution_closed_ts": (execution_meta or {}).get("closed_ts"),
                "live_mirror_execution_id": getattr(d, "live_mirror_execution_id", None),
                "live_mirror_status": getattr(d, "live_mirror_status", None),
                "live_mirror_error": getattr(d, "live_mirror_error", None),
                "cost_usd": d.cost_usd,
                "rationale": dec.get("rationale", ""),
                "reasoning_preview": (d.reasoning or "")[:300],
            }
        )
    return feed


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
    from app.portal_auth import session_token_from_request

    return session_token_from_request(request, uid)


def _gold_ai_page_auth_redirect(request: Request, uid: str):
    """Require a valid portal session matching uid and admin access."""
    from app.portal_auth import login_redirect, normalize_portal_uid, resolve_session_uid_with_user

    norm_uid = normalize_portal_uid(uid)
    session_uid = resolve_session_uid_with_user(request)
    if not session_uid or session_uid != norm_uid:
        return login_redirect(f"/gold-ai-trader?uid={norm_uid}", msg="session_expired")

    db = SessionLocal()
    try:
        try:
            _resolve_user(norm_uid, db, request=request)
        except HTTPException as exc:
            if exc.status_code == 403:
                return login_redirect(f"/gold-ai-trader?uid={norm_uid}", msg="admin_required")
            raise
    except Exception as exc:
        if is_db_connection_error(exc) and _session_admin_fallback(request, norm_uid) is not None:
            return None
        raise
    finally:
        db.close()
    return None


@router.get("/gold-ai-trader", response_class=HTMLResponse)
async def gold_ai_trader_page(request: Request, uid: str = Query(...)):
    """Serve dashboard shell — requires portal login + admin access."""
    from app.portal_auth import normalize_portal_uid

    norm_uid = normalize_portal_uid(uid)
    auth_redirect = _gold_ai_page_auth_redirect(request, norm_uid)
    if auth_redirect is not None:
        return auth_redirect

    session_token = _session_token_for_page(request, norm_uid) or ""
    resp = templates.TemplateResponse(
        "gold_ai_trader.html",
        {
            "request": request,
            "uid": norm_uid,
            "session_token": session_token,
        },
    )
    resp.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    resp.headers["Pragma"] = "no-cache"
    resp.headers["Expires"] = "0"
    return resp


@router.get("/api/gold-ai-trader/status")
async def api_status(request: Request, uid: str = Query(...)):
    norm_uid = _normalize_uid(uid)
    db = SessionLocal()
    degraded: List[str] = []
    try:
        try:
            admin = _resolve_user(norm_uid, db, request=request)
        except Exception as exc:
            if is_db_connection_error(exc):
                admin = _session_admin_fallback(request, norm_uid)
                if admin is None:
                    raise HTTPException(
                        status_code=503,
                        detail="Database temporarily unavailable — retry shortly",
                    ) from exc
                degraded.append("database")
                logger.warning(
                    "[gold-ai-trader] status using offline admin uid=%s: %s",
                    norm_uid,
                    exc,
                )
            else:
                raise

        # Fast path: avoid force-running DDL on every status poll (mobile/web
        # refresh every few seconds). Forced schema repair is only needed on
        # genuine schema errors and is already handled by retry paths below.
        try:
            ensure_gold_ai_trader_schema()
        except Exception as exc:
            logger.warning("[gold-ai-trader] schema ensure on status: %s", exc)

        # Self-heal: ensure this worker schedules the background loop on demand.
        try:
            from app.gold_ai_trader.loop import (
                maybe_start_background_loop,
                ensure_scan_liveness,
            )

            await maybe_start_background_loop()
            scan_kick = await ensure_scan_liveness()
            if scan_kick not in ("healthy", "throttled"):
                logger.info("[gold-ai-trader] status scan liveness kick: %s", scan_kick)
        except Exception as exc:
            logger.warning("[gold-ai-trader] status loop kick failed: %s", exc)

        cfg_row, cfg, trader_uid = _load_config_for_status(db, admin)
        if cfg_row is None:
            degraded.append("config")

        # Keep dashboard "IN TRADE" state in sync with broker truth. This
        # catches missed/delayed close events quickly and avoids stale OPEN rows
        # blocking new scans for extended periods.
        if trader_uid:
            try:
                _uid = int(trader_uid)
            except (TypeError, ValueError):
                _uid = 0
            if _uid > 0 and _should_run_status_reconcile(_uid):
                try:
                    from app.services.strategy_executor import _reconcile_forex_closes

                    await asyncio.wait_for(
                        _reconcile_forex_closes(user_id=_uid),
                        timeout=_STATUS_RECONCILE_TIMEOUT_S,
                    )
                except asyncio.TimeoutError:
                    logger.warning(
                        "[gold-ai-trader] status reconcile timed out uid=%s",
                        _uid,
                    )
                except Exception as exc:
                    logger.warning(
                        "[gold-ai-trader] status reconcile failed uid=%s: %s",
                        _uid,
                        exc,
                    )

        demo_accounts: List[Dict[str, Any]] = []
        selected = None
        try:
            demo_accounts = demo_accounts_for_user_id(db, trader_uid)
            selected = find_demo_account(demo_accounts, cfg.demo_ctrader_account_id)
        except Exception as exc:
            logger.warning("[gold-ai-trader] demo accounts load failed: %s", exc)
            degraded.append("demo_accounts")
            if is_db_connection_error(exc):
                degraded.append("database")
            try:
                db.rollback()
            except Exception:
                pass

        lessons = _safe_lessons(db, limit=3)
        decisions = _safe_decisions(db, limit=40)
        feed = _build_decision_feed(db, decisions)
        user_id = cfg.demo_user_id or getattr(admin, "id", None) or 0

        stats_today = _empty_stats_today()
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
            degraded.append("stats")
            if is_db_connection_error(exc):
                degraded.append("database")

        setup_stats: List[Dict[str, Any]] = []
        try:
            setup_stats = get_setup_stats(db)
        except Exception as exc:
            logger.warning("[gold-ai-trader] setup_stats failed: %s", exc)
            if is_db_connection_error(exc):
                degraded.append("database")

        call_stats: Dict[str, int] = {}
        try:
            call_stats = call_stats_today(db)
        except Exception as exc:
            logger.warning("[gold-ai-trader] call_stats_today failed: %s", exc)
            if is_db_connection_error(exc):
                degraded.append("database")

        live_accounts: List[Dict[str, Any]] = []
        try:
            live_accounts = _live_accounts_for_user(db, trader_uid)
        except Exception as exc:
            logger.warning("[gold-ai-trader] live accounts load failed: %s", exc)
            degraded.append("live_accounts")
            if is_db_connection_error(exc):
                degraded.append("database")

        db_down = "database" in degraded
        payload: Dict[str, Any] = {
            "ok": True,
            "demo_label": "DEMO ACCOUNT ONLY",
            "runtime": runtime_state.get_status(),
            "shared_session_hours": gold_ai_session_hours(),
            "config": _config_payload(cfg, cfg_row),
            "demo_accounts": demo_accounts,
            "demo_account_selected": selected,
            "demo_account_ready": demo_account_configured(cfg),
            "live_accounts": live_accounts,
            "stats_today": stats_today,
            "lessons": [
                {
                    "session": x.session,
                    "ts": x.ts.isoformat() if x.ts else None,
                    "digest": x.digest,
                }
                for x in lessons
            ],
            "setup_stats": setup_stats,
            "call_stats_today": call_stats,
            "recent_funnel_events": _safe_funnel_events(db, limit=40),
            "decision_feed": feed,
        }
        if db_down:
            payload["db_unavailable"] = True
            payload["db_message"] = "Database waking up — live data will appear when Neon reconnects."
        if degraded:
            payload["degraded"] = sorted(set(degraded))
            logger.warning("[gold-ai-trader] status degraded uid=%s parts=%s", norm_uid, payload["degraded"])
        return payload
    except HTTPException:
        raise
    except Exception as exc:
        if is_db_connection_error(exc):
            admin = _session_admin_fallback(request, norm_uid)
            if admin is not None:
                logger.warning(
                    "[gold-ai-trader] status offline fallback uid=%s: %s",
                    norm_uid,
                    exc,
                )
                return _offline_status_payload(admin, db_message=str(exc)[:160])
        logger.exception("[gold-ai-trader] status API failed uid=%s", norm_uid)
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
