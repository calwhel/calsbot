"""Isolated background scan loop for Gemini Vision Gold Trader."""
from __future__ import annotations

import asyncio
import logging
import os
import time
from datetime import datetime
from typing import Any, Dict, Optional

from app.gemini_gold_trader import state as runtime_state
from app.gemini_gold_trader.chart_renderer import render_candlestick_chart
from app.gemini_gold_trader.config import (
    env_defaults,
    gemini_gold_enabled,
    gemini_gold_loop_disabled_in_gunicorn,
    is_standalone_gemini_gold,
)
from app.gemini_gold_trader.db_thread import run_in_db_thread, run_with_db, with_db_session
from app.gemini_gold_trader.block_reason import format_block_reason
from app.gemini_gold_trader.executor import execute_take_market
from app.gemini_gold_trader.gemini import decide_from_charts
from app.gemini_gold_trader.guardrails import check_can_call_gemini, try_reserve_execution, merge_config
from app.gemini_gold_trader.schema import seed_config_if_missing
from app.gemini_gold_trader.klines import get_chart_klines, klines_ready
from app.gemini_gold_trader.models import GeminiGoldDecision
from app.gemini_gold_trader.outcomes import record_outcome_from_execution, sync_closed_outcomes
from app.gemini_gold_trader.telegram_notify import (
    maybe_notify_call_cap_reached,
    notify_decision,
)
from app.gemini_gold_trader.validator import validate_take_decision

logger = logging.getLogger(__name__)

_loop_task: asyncio.Task | None = None
_watchdog_task: asyncio.Task | None = None
_scan_cycle_lock: asyncio.Lock | None = None
_loop_task_started_mono = 0.0
_watchdog_last_restart_mono = 0.0
_restart_lock: asyncio.Lock | None = None


def _get_restart_lock() -> asyncio.Lock:
    global _restart_lock
    if _restart_lock is None:
        _restart_lock = asyncio.Lock()
    return _restart_lock


def _get_scan_cycle_lock() -> asyncio.Lock:
    global _scan_cycle_lock
    if _scan_cycle_lock is None:
        _scan_cycle_lock = asyncio.Lock()
    return _scan_cycle_lock


def active_session(now: datetime) -> Optional[str]:
    from app.services.forex_sessions import is_named_session_active

    if is_named_session_active("asia", now):
        return "asia"
    if is_named_session_active("new_york", now):
        return "new_york"
    if is_named_session_active("london", now):
        return "london"
    return None


def _load_merged_config(db, env):
    row = seed_config_if_missing(db)
    return merge_config(row, env)


def _parse_last_scan_at(raw: Optional[str]) -> Optional[datetime]:
    if not raw:
        return None
    try:
        s = str(raw).replace("Z", "")
        return datetime.fromisoformat(s)
    except (TypeError, ValueError):
        return None


def scan_heartbeat_age_seconds() -> float | None:
    last = _parse_last_scan_at(runtime_state.get_status().get("last_scan_at"))
    if last is None:
        return None
    return max(0.0, (datetime.utcnow() - last).total_seconds())


def _persist_decision_db(
    db,
    *,
    session: str,
    decision: Optional[Dict[str, Any]],
    action: str,
    confidence: int,
    rationale: str,
    chart_meta: Dict[str, Any],
    tokens_in: int,
    tokens_out: int,
    cost_usd: float,
    dry_run: bool,
    skip_reason: Optional[str] = None,
) -> GeminiGoldDecision:
    direction = None
    if decision:
        direction = (decision.get("direction") or None)
        if direction:
            direction = str(direction).upper()
    row = GeminiGoldDecision(
        session=session,
        decision=decision,
        action=action,
        direction=direction,
        confidence=confidence,
        rationale=rationale,
        chart_meta=chart_meta,
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        cost_usd=cost_usd,
        dry_run=dry_run,
        skip_reason=skip_reason,
    )
    db.add(row)
    db.commit()
    db.refresh(row)
    return row


async def run_gemini_gold_trader_loop() -> None:
    """One scan cycle."""
    now = datetime.utcnow()
    env = env_defaults()
    cfg = await run_with_db(_load_merged_config, env)

    session = active_session(now)
    runtime_state.note_scan(session)

    if not cfg.enabled:
        runtime_state.note_dormant("disabled")
        return
    if cfg.kill_switch:
        runtime_state.note_dormant("kill_switch")
        return
    if not session:
        runtime_state.note_dormant("outside_session")
        return

    can_call, call_reason = await run_with_db(check_can_call_gemini, cfg)
    if not can_call:
        runtime_state.note_dormant(call_reason)
        if call_reason == "max_calls_day":
            await maybe_notify_call_cap_reached()
        return

    bars_15m, meta_15m = await get_chart_klines(
        "15m", cfg.chart_bars, user_id=cfg.demo_user_id,
    )
    bars_1h, meta_1h = await get_chart_klines(
        "1h", cfg.chart_bars, user_id=cfg.demo_user_id,
    )
    chart_meta = {"15m": meta_15m, "1h": meta_1h}

    if not klines_ready(bars_15m, bars_1h):
        logger.info(
            "[gemini-gold] skipping scan — stale/missing klines "
            "(15m=%s/%s bars source=%s, 1h=%s/%s bars source=%s)",
            len(bars_15m),
            meta_15m.get("status"),
            meta_15m.get("source"),
            len(bars_1h),
            meta_1h.get("status"),
            meta_1h.get("source"),
        )
        runtime_state.note_dormant("stale_klines")
        return

    spot = float(bars_15m[-1][4])
    try:
        from app.services.tradfi_prices import get_price_fresh

        live = await get_price_fresh(
            "XAUUSD", "forex", paper_ok=False, user_id=cfg.demo_user_id
        )
        if live and live > 0:
            spot = float(live)
    except Exception:
        pass

    png_15m = await asyncio.to_thread(
        render_candlestick_chart,
        bars_15m,
        timeframe="15m",
        session=session,
    )
    png_1h = await asyncio.to_thread(
        render_candlestick_chart,
        bars_1h,
        timeframe="1h",
        session=session,
    )
    if not png_15m or not png_1h:
        logger.warning("[gemini-gold] chart render failed")
        runtime_state.note_error("chart_render_failed")
        return

    decision, tokens_in, tokens_out, cost_usd, api_error = await decide_from_charts(
        cfg=cfg,
        session=session,
        spot=spot,
        png_15m=png_15m,
        png_1h=png_1h,
        bars_15m=len(bars_15m),
        bars_1h=len(bars_1h),
    )

    if api_error or not decision:
        row = await run_with_db(
            _persist_decision_db,
            session=session,
            decision=None,
            action="ERROR",
            confidence=0,
            rationale="",
            chart_meta=chart_meta,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            cost_usd=cost_usd,
            dry_run=cfg.dry_run,
            skip_reason=api_error or "no_decision",
        )
        logger.warning("[gemini-gold] Gemini call failed: %s (decision_id=%s)", api_error, row.id)
        return

    action = str(decision.get("action") or "SKIP").upper()
    confidence = int(decision.get("confidence") or 0)
    rationale = str(decision.get("rationale") or "")

    row = await run_with_db(
        _persist_decision_db,
        session=session,
        decision=decision,
        action=action,
        confidence=confidence,
        rationale=rationale,
        chart_meta=chart_meta,
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        cost_usd=cost_usd,
        dry_run=cfg.dry_run,
    )
    runtime_state.note_decision(decision)
    logger.info(
        "[gemini-gold] decision_id=%s action=%s confidence=%s%% session=%s cost=$%.4f",
        row.id,
        action,
        confidence,
        session,
        cost_usd,
    )

    executed = False
    execution_id: Optional[int] = None
    block_reason: Optional[str] = None

    if action == "SKIP":
        await notify_decision(
            session=session,
            decision=decision,
            action=action,
            confidence=confidence,
            dry_run=cfg.dry_run,
        )
        return

    if action != "TAKE":
        block_reason = f"unknown_action:{action}"
        await notify_decision(
            session=session,
            decision=decision,
            action=action,
            confidence=confidence,
            block_reason=block_reason,
            dry_run=cfg.dry_run,
        )
        return

    if confidence < cfg.confidence_threshold:
        block_reason = (
            f"confidence {confidence}% below {cfg.confidence_threshold}% threshold"
        )
        await notify_decision(
            session=session,
            decision=decision,
            action=action,
            confidence=confidence,
            block_reason=block_reason,
            dry_run=cfg.dry_run,
        )
        return

    ok, val_reason, decision = validate_take_decision(decision, cfg=cfg, spot=spot)
    if not ok:
        block_reason = val_reason
        await notify_decision(
            session=session,
            decision=decision,
            action=action,
            confidence=confidence,
            block_reason=block_reason,
            dry_run=cfg.dry_run,
        )
        return

    await _preflight_execution_caps(cfg)

    can_exec, exec_reason = await run_with_db(
        try_reserve_execution, cfg, cfg.demo_user_id or 0, row.id
    )
    if not can_exec:
        block_reason = format_block_reason(exec_reason)
        await notify_decision(
            session=session,
            decision=decision,
            action=action,
            confidence=confidence,
            block_reason=block_reason,
            dry_run=cfg.dry_run,
        )
        return

    from app.database import SessionLocal

    order_ctx: Dict[str, Any] = {}
    db = SessionLocal()
    try:
        execution_id = await execute_take_market(
            db=db,
            cfg=cfg,
            decision=decision,
            decision_id=row.id,
            spot_hint=spot,
            order_ctx=order_ctx,
        )
    finally:
        db.close()

    if execution_id:
        executed = True

        def _mark_executed(db):
            r = db.query(GeminiGoldDecision).filter(GeminiGoldDecision.id == row.id).first()
            if r:
                r.executed = True
                r.execution_id = execution_id
                db.commit()

        await run_with_db(_mark_executed)
    else:
        block_reason = format_block_reason(
            order_ctx.get("block_reason")
            or order_ctx.get("broker_error")
            or block_reason
            or "demo order rejected"
        )

        def _clear_reservation(db):
            from app.gemini_gold_trader.guardrails import clear_execution_reservation

            clear_execution_reservation(db, row.id)

        await run_with_db(_clear_reservation)

    await notify_decision(
        session=session,
        decision=decision,
        action=action,
        confidence=confidence,
        executed=executed,
        execution_id=execution_id,
        block_reason=block_reason,
        dry_run=cfg.dry_run,
    )

    if execution_id:
        from app.database import SessionLocal
        from app.strategy_models import StrategyExecution

        db = SessionLocal()
        try:
            ex = db.query(StrategyExecution).filter(StrategyExecution.id == execution_id).first()
            if ex:
                await run_with_db(record_outcome_from_execution, row.id, ex)
        finally:
            db.close()


async def _call_with_db_session(async_fn, /, *args, **kwargs):
    """Open a short-lived session for one async callee, then close."""
    from app.database import SessionLocal

    holder: list = []

    def _open():
        holder.append(SessionLocal())

    await run_in_db_thread(_open)
    db = holder[0]
    try:
        return await async_fn(*args, db=db, **kwargs)
    finally:
        await run_in_db_thread(db.close)


async def _preflight_execution_caps(cfg) -> None:
    """Clear phantom OPEN rows and stale reservations before reserving a new slot."""
    demo_uid = int(getattr(cfg, "demo_user_id", 0) or 0)
    if demo_uid <= 0:
        return
    try:
        from app.gemini_gold_trader.guardrails import clear_stale_execution_reservations
        from app.gemini_gold_trader.reconcile import reconcile_orphan_open_executions

        cleared = await run_with_db(clear_stale_execution_reservations)
        if cleared:
            logger.info("[gemini-gold] cleared %s stale execution reservation(s)", cleared)
        orphan_result = await _call_with_db_session(
            reconcile_orphan_open_executions,
            cfg=cfg,
            user_id=demo_uid,
        )
        closed = (orphan_result or {}).get("orphans_closed") or []
        if closed:
            logger.warning(
                "[gemini-gold] preflight cancelled %s phantom OPEN row(s): %s",
                len(closed),
                [c.get("execution_id") for c in closed],
            )
    except Exception as exc:
        logger.warning("[gemini-gold] preflight cap cleanup failed: %s", exc)


async def _sync_closed_outcomes_pass() -> None:
    """Broker close reconcile + orphan OPEN cleanup + outcome sync every loop cycle."""

    def _load_demo_uid(db):
        row = seed_config_if_missing(db)
        cfg = merge_config(row, env_defaults())
        return int(getattr(cfg, "demo_user_id", 0) or 0), cfg

    try:
        demo_uid, cfg = await run_with_db(_load_demo_uid)
        if demo_uid > 0:
            try:
                from app.gemini_gold_trader.guardrails import clear_stale_execution_reservations

                cleared = await run_with_db(clear_stale_execution_reservations)
                if cleared:
                    logger.info(
                        "[gemini-gold] cleared %s stale execution reservation(s)",
                        cleared,
                    )
            except Exception as exc:
                logger.warning("[gemini-gold] stale reservation cleanup failed: %s", exc)

            try:
                from app.gemini_gold_trader.reconcile import reconcile_orphan_open_executions

                orphan_result = await _call_with_db_session(
                    reconcile_orphan_open_executions,
                    cfg=cfg,
                    user_id=demo_uid,
                )
                closed = (orphan_result or {}).get("orphans_closed") or []
                if closed:
                    logger.warning(
                        "[gemini-gold] orphan reconcile closed %s phantom OPEN row(s): %s",
                        len(closed),
                        [c.get("execution_id") for c in closed],
                    )
            except Exception as exc:
                logger.warning(
                    "[gemini-gold] orphan OPEN reconcile failed uid=%s: %s",
                    demo_uid,
                    exc,
                )

            try:
                from app.services.strategy_executor import _reconcile_forex_closes

                timeout_s = max(
                    8.0,
                    min(float(getattr(cfg, "scan_interval_s", 60.0) or 60.0), 45.0),
                )
                await asyncio.wait_for(
                    _reconcile_forex_closes(user_id=demo_uid),
                    timeout=timeout_s,
                )
            except asyncio.TimeoutError:
                logger.warning(
                    "[gemini-gold] broker close reconcile timed out uid=%s",
                    demo_uid,
                )
            except Exception as exc:
                logger.warning(
                    "[gemini-gold] broker close reconcile failed uid=%s: %s",
                    demo_uid,
                    exc,
                )

        def _sync(db):
            return sync_closed_outcomes(db, cfg.demo_user_id)

        recorded = await run_with_db(_sync)
        if recorded:
            logger.info("[gemini-gold] synced %s closed outcomes", recorded)

        from app.gemini_gold_trader.telegram_notify import sync_closed_trade_notifications

        notified = await _call_with_db_session(sync_closed_trade_notifications, cfg=cfg)
        if notified:
            logger.info("[gemini-gold] sent %s close notifications", notified)
    except Exception as exc:
        logger.warning("[gemini-gold] closed-outcome sync pass failed: %s", exc)


def _loop_task_done(task: asyncio.Task) -> None:
    if task.cancelled():
        logger.warning("[gemini-gold] background loop task cancelled")
        return
    exc = task.exception()
    if exc is None:
        logger.warning("[gemini-gold] background loop task exited unexpectedly")
    else:
        logger.error("[gemini-gold] background loop task crashed: %s", exc, exc_info=exc)


def _watchdog_task_done(task: asyncio.Task) -> None:
    if task.cancelled():
        logger.warning("[gemini-gold] watchdog task cancelled")
        return
    exc = task.exception()
    if exc is None:
        logger.warning("[gemini-gold] watchdog task exited unexpectedly")
    else:
        logger.error("[gemini-gold] watchdog task crashed: %s", exc, exc_info=exc)


def _schedule_loop_task() -> None:
    global _loop_task, _loop_task_started_mono
    _loop_task = asyncio.create_task(_scan_loop_forever())
    _loop_task_started_mono = time.monotonic()
    _loop_task.add_done_callback(_loop_task_done)
    logger.info("[gemini-gold] background task scheduled")


def _schedule_watchdog_task() -> None:
    global _watchdog_task
    _watchdog_task = asyncio.create_task(_watchdog_loop_forever())
    _watchdog_task.add_done_callback(_watchdog_task_done)
    logger.info("[gemini-gold] watchdog task scheduled")


def _watchdog_snapshot() -> tuple[bool, bool, str | None, float | None]:
    env = env_defaults()
    cfg = with_db_session(_load_merged_config)(env)
    if not cfg.enabled:
        return False, bool(cfg.kill_switch), None, None
    session = active_session(datetime.utcnow())
    age_s = scan_heartbeat_age_seconds()
    return True, bool(cfg.kill_switch), session, age_s


async def _stop_loop_task(reason: str) -> None:
    global _loop_task
    task = _loop_task
    if task and not task.done():
        task.cancel()
        try:
            await asyncio.wait_for(task, timeout=8.0)
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            logger.warning("[gemini-gold] loop cancel wait failed (%s): %s", reason, exc)
    _loop_task = None


async def _restart_background_loop(reason: str) -> str:
    global _watchdog_last_restart_mono
    min_restart_interval_s = max(
        30.0,
        float(os.environ.get("GEMINI_GOLD_WATCHDOG_MIN_RESTART_INTERVAL_S", "90")),
    )
    now_m = time.monotonic()
    if now_m - _watchdog_last_restart_mono < min_restart_interval_s:
        return "restart_throttled"
    async with _get_restart_lock():
        now_m = time.monotonic()
        if now_m - _watchdog_last_restart_mono < min_restart_interval_s:
            return "restart_throttled"
        _watchdog_last_restart_mono = now_m
        logger.error("[gemini-gold] watchdog restarting loop: %s", reason)
        await _stop_loop_task(reason)
        _schedule_loop_task()
        return "restarted"


async def _watchdog_loop_forever() -> None:
    interval_s = max(15.0, float(os.environ.get("GEMINI_GOLD_WATCHDOG_INTERVAL_S", "30")))
    stale_after_s = max(60.0, float(os.environ.get("GEMINI_GOLD_WATCHDOG_STALE_AFTER_S", "240")))
    startup_grace_s = max(30.0, float(os.environ.get("GEMINI_GOLD_WATCHDOG_STARTUP_GRACE_S", "90")))
    logger.info("[gemini-gold] watchdog starting (interval=%ss)", interval_s)
    try:
        while True:
            await asyncio.sleep(interval_s)
            try:
                enabled, kill_switch, session, age_s = await run_in_db_thread(
                    _watchdog_snapshot
                )
            except Exception as exc:
                logger.warning("[gemini-gold] watchdog snapshot failed: %s", exc)
                continue
            if not enabled or kill_switch or not session:
                continue
            if _loop_task is None or _loop_task.done():
                await _restart_background_loop("loop_task_missing")
                continue
            if _scan_cycle_lock is not None and _scan_cycle_lock.locked():
                logger.warning(
                    "[gemini-gold] watchdog: scan cycle still running — skip stale restart"
                )
                continue
            if age_s is None:
                if time.monotonic() - _loop_task_started_mono < startup_grace_s:
                    continue
                await _restart_background_loop("no_heartbeat")
                continue
            if age_s > stale_after_s:
                await _restart_background_loop(f"stale_heartbeat_{int(age_s)}s")
    except asyncio.CancelledError:
        raise


async def _scan_loop_forever() -> None:
    env = env_defaults()
    try:
        cfg = await run_with_db(_load_merged_config, env)
        delay = max(60.0, float(cfg.scan_interval_s))
    except Exception as exc:
        delay = max(60.0, float(env.scan_interval_s))
        logger.error(
            "[gemini-gold] initial config load failed, using env scan interval (%.0fs): %s",
            delay,
            exc,
            exc_info=True,
        )
    cycle_timeout_s = max(delay * 2.0, float(os.environ.get("GEMINI_GOLD_LOOP_CYCLE_TIMEOUT_S", "180")))
    reconcile_timeout_s = max(10.0, float(os.environ.get("GEMINI_GOLD_LOOP_RECON_TIMEOUT_S", "45")))
    logger.info("[gemini-gold] background loop starting (interval=%ss)", delay)
    try:
        try:
            await asyncio.wait_for(_sync_closed_outcomes_pass(), timeout=reconcile_timeout_s)
            logger.info("[gemini-gold] startup broker reconcile pass complete")
        except Exception as exc:
            logger.warning("[gemini-gold] startup reconcile pass failed: %s", exc)
        while True:
            async with _get_scan_cycle_lock():
                try:
                    await asyncio.wait_for(run_gemini_gold_trader_loop(), timeout=cycle_timeout_s)
                except asyncio.TimeoutError:
                    runtime_state.note_error(f"scan_cycle_timeout>{int(cycle_timeout_s)}s")
                    logger.error(
                        "[gemini-gold] scan cycle timeout after %.1fs — continuing",
                        cycle_timeout_s,
                    )
                except Exception as exc:
                    logger.error("[gemini-gold] scan loop cycle error: %s", exc, exc_info=True)
                try:
                    await asyncio.wait_for(_sync_closed_outcomes_pass(), timeout=reconcile_timeout_s)
                except asyncio.TimeoutError:
                    logger.error("[gemini-gold] reconcile pass timeout after %.1fs", reconcile_timeout_s)
                except Exception as exc:
                    logger.error("[gemini-gold] reconcile pass error: %s", exc)
            await asyncio.sleep(delay)
    except asyncio.CancelledError:
        logger.warning("[gemini-gold] background loop cancelled")
        raise
    finally:
        logger.warning("[gemini-gold] background loop stopped")


async def start_gemini_gold_trader_loop() -> None:
    global _loop_task, _watchdog_task
    if not gemini_gold_enabled():
        logger.info("[gemini-gold] disabled (GEMINI_GOLD_ENABLED=false)")
        return
    if gemini_gold_loop_disabled_in_gunicorn() and not is_standalone_gemini_gold():
        logger.info("[gemini-gold] loop disabled in gunicorn")
        return
    if _loop_task is None or _loop_task.done():
        _schedule_loop_task()
    if _watchdog_task is None or _watchdog_task.done():
        _schedule_watchdog_task()
