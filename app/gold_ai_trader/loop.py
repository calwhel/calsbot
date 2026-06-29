"""Isolated background loop with advisory lock."""
from __future__ import annotations

import asyncio
import logging
import os
import time
from datetime import datetime

import httpx

from app.database import SessionLocal
from app.gold_ai_trader.config import env_defaults, gold_ai_trader_enabled
from app.gold_ai_trader.data_quality import (
    assess_gold_market_data,
    format_data_source,
    gold_data_ok_for_claude,
)
from app.gold_ai_trader.schema import seed_config_if_missing
from app.gold_ai_trader.guardrails import (
    merge_config,
    check_can_call_claude,
    check_can_call_orb,
    check_can_execute,
    check_can_execute_live_mirror,
)
from app.gold_ai_trader.scanner import (
    active_session,
    scan_candidates,
    pick_top_candidates,
    record_claude_invocation,
    _setup_cooldown_s,
)
from app.gold_ai_trader.call_gates import (
    atr_from_klines,
    collect_key_levels,
    in_killzone,
    killzone_only_enabled,
    killzone_override_enabled,
    killzone_override_min_confluence,
    candidate_confluence_counts,
    candidate_meets_killzone_override,
    should_invoke_claude,
    call_stats_today,
)
from app.gold_ai_trader.decision_validator import validate_take_decision
from app.gold_ai_trader.funnel import record as funnel_record, snapshot as funnel_snapshot
from app.gold_ai_trader.setup_toggles import max_candidates_per_scan
from app.gold_ai_trader.klines import get_gold_ai_klines
from app.services.tradfi_prices import get_klines, confirm_entry_price
from app.gold_ai_trader.config import SYMBOL, ASSET_CLASS
from app.gold_ai_trader.context import build_context_snapshot
from app.gold_ai_trader.claude import decide
from app.gold_ai_trader.claude import decide_orb
from app.gold_ai_trader.executor import execute_take, execute_live_mirror_take, flatten_open_demo_positions
from app.gold_ai_trader.fire_time_validation import refresh_spot_after_claude
from app.gold_ai_trader.learning import maybe_run_learning_review, record_outcome_from_execution, get_setup_stats
from app.gold_ai_trader.orb import build_orb_context, detect_orb_signal
from app.gold_ai_trader.pending_entry import sync_pending_entries
from app.gold_ai_trader.telegram_notify import (
    maybe_send_daily_summary,
    notify_take_decision,
    sync_closed_trade_notifications,
    maybe_notify_call_cap_reached,
)
from app.gold_ai_trader.models import GoldAiConfig, GoldAiDecision
from app.gold_ai_trader import state as runtime_state

logger = logging.getLogger(__name__)

_LOCK_ID = 42_424_250
_LOCK_APP_NAME = "gold-ai-trader"
_loop_task: asyncio.Task | None = None
_watchdog_task: asyncio.Task | None = None
_prev_session: str | None = None
_lock_conn = None
_loop_task_started_mono = 0.0
_watchdog_last_restart_mono = 0.0
_restart_lock: asyncio.Lock | None = None
_on_demand_lock: asyncio.Lock | None = None
_on_demand_last_mono = 0.0


def _utc_iso_now() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def _fire_context(
    *,
    setup_type: str,
    candidate_direction: str,
    setup_detail: str,
    atr: float,
    key_levels,
) -> dict:
    return {
        "setup_type": setup_type,
        "candidate_direction": candidate_direction,
        "setup_detail": setup_detail,
        "atr": atr,
        "key_levels": key_levels,
    }


async def _stale_entry_recheck(
    *,
    decision: dict,
    cfg,
    decision_ts: datetime | None,
    decision_id: int,
    setup_type: str,
) -> tuple[bool, str]:
    """
    If execution is delayed beyond threshold, reconfirm entry against live spot.
    """
    if decision_ts is None:
        return True, "no_decision_ts"
    max_delay_s = max(
        0.0,
        float(os.environ.get("GOLD_AI_MAX_EXEC_DELAY_S", "90")),
    )
    if max_delay_s <= 0:
        return True, "delay_guard_disabled"
    age_s = max(0.0, (datetime.utcnow() - decision_ts).total_seconds())
    if age_s <= max_delay_s:
        return True, f"delay_ok({age_s:.1f}s)"
    try:
        proposed = float(decision.get("entry") or 0.0)
    except (TypeError, ValueError):
        proposed = 0.0
    if proposed <= 0:
        reason = f"stale_guard_block:no_entry delay={age_s:.1f}s>{max_delay_s:.1f}s"
        logger.warning("[gold-ai] %s decision_id=%s setup=%s", reason, decision_id, setup_type)
        return False, reason
    confirmed, confirm_reason = await confirm_entry_price(
        SYMBOL,
        ASSET_CLASS,
        proposed,
        paper_ok=False,
        user_id=cfg.demo_user_id,
    )
    if confirmed is None or confirmed <= 0:
        reason = (
            f"stale_guard_block:{confirm_reason} "
            f"delay={age_s:.1f}s>{max_delay_s:.1f}s"
        )
        logger.warning("[gold-ai] %s decision_id=%s setup=%s", reason, decision_id, setup_type)
        return False, reason
    decision["entry"] = float(confirmed)
    note = f"stale_guard_reconfirmed({confirm_reason}) delay={age_s:.1f}s"
    decision["stale_guard_note"] = note
    logger.info(
        "[gold-ai] stale guard recheck decision_id=%s setup=%s old_entry=%.4f new_entry=%.4f age_s=%.1f",
        decision_id,
        setup_type,
        proposed,
        float(confirmed),
        age_s,
    )
    return True, note


def _ping_lock_connection(conn) -> bool:
    try:
        if conn is None or getattr(conn, "closed", 0):
            return False
        with conn.cursor() as cur:
            cur.execute("SELECT 1")
        return True
    except Exception:
        return False


def _acquire_gold_ai_lock():
    from app.executor_lock import build_lock_connection, try_acquire_lock

    conn = build_lock_connection(_LOCK_APP_NAME)
    if try_acquire_lock(conn, _LOCK_ID):
        return conn
    try:
        conn.close()
    except Exception:
        pass
    return None


def _reconnect_gold_ai_lock(old_conn):
    from app.executor_lock import close_lock_connection, reconnect_lock_connection

    return reconnect_lock_connection(
        old_conn,
        lock_id=_LOCK_ID,
        application_name=_LOCK_APP_NAME,
        max_attempts=5,
        retry_delay=2.0,
        silent=False,
    )


def _reclaim_stale_gold_ai_locks(*, min_idle_seconds: float) -> int:
    from app.executor_lock import terminate_lock_holders

    return int(
        terminate_lock_holders(
            _LOCK_ID,
            min_idle_seconds=max(0.0, float(min_idle_seconds)),
            owner_app=_LOCK_APP_NAME,
            log_prefix="[gold-ai-trader-lock]",
        )
        or 0
    )


def _release_gold_ai_lock(conn) -> None:
    if not conn:
        return
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT pg_advisory_unlock(%s)", (_LOCK_ID,))
    except Exception:
        pass
    try:
        conn.close()
    except Exception:
        pass


def _parse_last_scan_at(raw: str | None) -> datetime | None:
    if not raw:
        return None
    txt = str(raw).strip()
    if not txt:
        return None
    if txt.endswith("Z"):
        txt = txt[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(txt)
    except Exception:
        return None
    if dt.tzinfo is not None:
        try:
            return dt.astimezone().replace(tzinfo=None)
        except Exception:
            return dt.replace(tzinfo=None)
    return dt


def _get_on_demand_lock() -> asyncio.Lock:
    global _on_demand_lock
    if _on_demand_lock is None:
        _on_demand_lock = asyncio.Lock()
    return _on_demand_lock


def _get_restart_lock() -> asyncio.Lock:
    global _restart_lock
    if _restart_lock is None:
        _restart_lock = asyncio.Lock()
    return _restart_lock


def _loop_task_done(task: asyncio.Task) -> None:
    if task.cancelled():
        logger.warning("[gold-ai-trader] background loop task cancelled")
        return
    exc = task.exception()
    if exc is None:
        logger.warning("[gold-ai-trader] background loop task exited unexpectedly")
    else:
        logger.error("[gold-ai-trader] background loop task crashed: %s", exc, exc_info=exc)


def _watchdog_task_done(task: asyncio.Task) -> None:
    if task.cancelled():
        logger.warning("[gold-ai-trader] watchdog task cancelled")
        return
    exc = task.exception()
    if exc is None:
        logger.warning("[gold-ai-trader] watchdog task exited unexpectedly")
    else:
        logger.error("[gold-ai-trader] watchdog task crashed: %s", exc, exc_info=exc)


def _schedule_loop_task() -> None:
    global _loop_task, _loop_task_started_mono
    _loop_task = asyncio.create_task(_locked_loop_forever())
    _loop_task_started_mono = time.monotonic()
    _loop_task.add_done_callback(_loop_task_done)
    logger.info("[gold-ai-trader] background task scheduled")


def _schedule_watchdog_task() -> None:
    global _watchdog_task
    _watchdog_task = asyncio.create_task(_watchdog_loop_forever())
    _watchdog_task.add_done_callback(_watchdog_task_done)
    logger.info("[gold-ai-trader] watchdog task scheduled")


def _watchdog_snapshot() -> tuple[bool, bool, str | None, float | None]:
    """Return (enabled, kill_switch, active_session, heartbeat_age_s)."""
    env = env_defaults()
    cfg = env
    db = SessionLocal()
    try:
        row = seed_config_if_missing(db)
        cfg = merge_config(row, env)
    except Exception:
        cfg = env
    finally:
        db.close()
    if not cfg.enabled:
        return False, bool(cfg.kill_switch), None, None
    session = active_session(datetime.utcnow(), cfg)
    last = _freshest_scan_heartbeat_utc()
    age_s = (datetime.utcnow() - last).total_seconds() if last else None
    return True, bool(cfg.kill_switch), session, age_s


async def _stop_loop_task(reason: str) -> None:
    global _loop_task, _lock_conn
    task = _loop_task
    if task and not task.done():
        task.cancel()
        try:
            await asyncio.wait_for(task, timeout=8.0)
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            logger.warning("[gold-ai-trader] loop cancel wait failed (%s): %s", reason, exc)
    _loop_task = None
    conn = _lock_conn
    _lock_conn = None
    if conn is not None:
        await asyncio.to_thread(_release_gold_ai_lock, conn)


async def _restart_background_loop(reason: str, *, force_reclaim: bool) -> str:
    global _watchdog_last_restart_mono
    min_restart_interval_s = max(
        30.0,
        float(os.environ.get("GOLD_AI_WATCHDOG_MIN_RESTART_INTERVAL_S", "90")),
    )
    now_m = time.monotonic()
    if now_m - _watchdog_last_restart_mono < min_restart_interval_s:
        return "restart_throttled"
    async with _get_restart_lock():
        now_m = time.monotonic()
        if now_m - _watchdog_last_restart_mono < min_restart_interval_s:
            return "restart_throttled"
        _watchdog_last_restart_mono = now_m
        logger.error("[gold-ai-trader] watchdog restarting loop: %s", reason)
        if force_reclaim:
            try:
                await asyncio.to_thread(_reclaim_stale_gold_ai_locks, min_idle_seconds=0.0)
            except Exception as exc:
                logger.warning("[gold-ai-trader] watchdog force reclaim failed: %s", exc)
        await _stop_loop_task(reason)
        _schedule_loop_task()
        return "restarted"


def _freshest_scan_heartbeat_utc() -> datetime | None:
    """Best scan heartbeat across local runtime + persisted funnel events."""
    best = _parse_last_scan_at(runtime_state.get_status().get("last_scan_at"))
    db = SessionLocal()
    try:
        from app.gold_ai_trader.models import GoldAiFunnelEvent

        row = (
            db.query(GoldAiFunnelEvent.ts)
            .filter(GoldAiFunnelEvent.event == "scan")
            .order_by(GoldAiFunnelEvent.ts.desc())
            .limit(1)
            .first()
        )
        ts = row[0] if row else None
        if ts is not None and getattr(ts, "tzinfo", None) is not None:
            try:
                ts = ts.astimezone().replace(tzinfo=None)
            except Exception:
                ts = ts.replace(tzinfo=None)
        if ts is not None and (best is None or ts > best):
            best = ts
    except Exception as exc:
        logger.debug("[gold-ai-trader] heartbeat query failed: %s", exc)
    finally:
        db.close()
    return best


async def ensure_scan_liveness() -> str:
    """On-demand one-shot scan recovery for stale in-session runtimes."""
    global _on_demand_last_mono
    if not gold_ai_trader_enabled():
        return "disabled"
    min_interval_s = max(
        10.0,
        float(os.environ.get("GOLD_AI_ON_DEMAND_SCAN_MIN_INTERVAL_S", "20")),
    )
    stale_after_s = max(
        60.0,
        float(os.environ.get("GOLD_AI_ON_DEMAND_SCAN_STALE_AFTER_S", "120")),
    )
    timeout_s = max(
        10.0,
        float(os.environ.get("GOLD_AI_ON_DEMAND_SCAN_TIMEOUT_S", "35")),
    )
    force_reclaim_after_s = max(
        stale_after_s,
        float(os.environ.get("GOLD_AI_ON_DEMAND_FORCE_RECLAIM_AFTER_S", "180")),
    )
    now_m = time.monotonic()
    if now_m - _on_demand_last_mono < min_interval_s:
        return "throttled"

    last = await asyncio.to_thread(_freshest_scan_heartbeat_utc)
    age_s = (datetime.utcnow() - last).total_seconds() if last else None
    if age_s is not None and age_s <= stale_after_s:
        return "healthy"

    lock = _get_on_demand_lock()
    async with lock:
        now_m = time.monotonic()
        if now_m - _on_demand_last_mono < min_interval_s:
            return "throttled"
        _on_demand_last_mono = now_m

        conn = await asyncio.to_thread(_acquire_gold_ai_lock)
        if conn is None:
            reclaim_idle_s = max(stale_after_s, 120.0)
            if age_s is not None and age_s >= force_reclaim_after_s:
                # Heartbeat is stale cluster-wide; force-takeover even if holder
                # looks "live" from pg_stat_activity idle timers.
                reclaim_idle_s = 0.0
            try:
                reclaimed = await asyncio.to_thread(
                    _reclaim_stale_gold_ai_locks,
                    min_idle_seconds=reclaim_idle_s,
                )
                if reclaimed and reclaim_idle_s <= 0.0:
                    logger.warning(
                        "[gold-ai-trader] forced lock reclaim after stale heartbeat age=%.0fs",
                        age_s or -1.0,
                    )
            except Exception as exc:
                logger.warning("[gold-ai-trader] on-demand stale lock reclaim failed: %s", exc)
            conn = await asyncio.to_thread(_acquire_gold_ai_lock)
        if conn is None:
            return "lock_busy"
        try:
            await asyncio.wait_for(run_gold_ai_trader_loop(), timeout=timeout_s)
            return "ran"
        except asyncio.TimeoutError:
            logger.warning("[gold-ai-trader] on-demand scan timeout after %.1fs", timeout_s)
            return "timeout"
        except Exception as exc:
            logger.warning("[gold-ai-trader] on-demand scan failed: %s", exc)
            return "error"
        finally:
            await asyncio.to_thread(_release_gold_ai_lock, conn)


async def _watchdog_loop_forever() -> None:
    interval_s = max(
        15.0,
        float(os.environ.get("GOLD_AI_WATCHDOG_INTERVAL_S", "30")),
    )
    stale_after_s = max(
        120.0,
        float(os.environ.get("GOLD_AI_WATCHDOG_STALE_AFTER_S", "240")),
    )
    startup_grace_s = max(
        60.0,
        float(os.environ.get("GOLD_AI_WATCHDOG_STARTUP_GRACE_S", "90")),
    )
    logger.info(
        "[gold-ai-trader] watchdog active (interval=%ss stale_after=%ss)",
        interval_s,
        stale_after_s,
    )
    while True:
        try:
            if not gold_ai_trader_enabled():
                await asyncio.sleep(interval_s)
                continue
            if _loop_task is None or _loop_task.done():
                await _restart_background_loop("loop_task_missing_or_done", force_reclaim=False)
                await asyncio.sleep(interval_s)
                continue
            enabled, kill_switch, session, age_s = await asyncio.to_thread(_watchdog_snapshot)
            if not enabled or kill_switch or not session:
                await asyncio.sleep(interval_s)
                continue
            has_local_lock = False
            if _lock_conn is not None:
                try:
                    has_local_lock = await asyncio.to_thread(_ping_lock_connection, _lock_conn)
                except Exception:
                    has_local_lock = False
            # Prevent multi-worker watchdog herding: only the worker currently
            # holding the advisory lock is allowed to force stale-heartbeat
            # restarts/reclaims. Non-owners keep trying normal lock acquire.
            if not has_local_lock:
                await asyncio.sleep(interval_s)
                continue
            loop_age_s = max(0.0, time.monotonic() - _loop_task_started_mono)
            if loop_age_s < startup_grace_s:
                await asyncio.sleep(interval_s)
                continue
            if age_s is None or age_s > stale_after_s:
                reason = (
                    f"stale_heartbeat session={session} age_s="
                    f"{int(age_s) if age_s is not None else -1}"
                )
                logger.error("[gold-ai-trader] watchdog detected %s", reason)
                await _restart_background_loop(reason, force_reclaim=True)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.error("[gold-ai-trader] watchdog error: %s", exc, exc_info=True)
        await asyncio.sleep(interval_s)


async def _sync_closed_outcomes_pass() -> None:
    """Run close reconciliation every loop cycle, independent of scan gating."""
    db = SessionLocal()
    try:
        cfg_row = seed_config_if_missing(db)
        cfg = merge_config(cfg_row, env_defaults())
        demo_uid = int(getattr(cfg, "demo_user_id", 0) or 0)
        if demo_uid > 0:
            try:
                # Gold AI depends on this OPEN→closed transition for hero/feed state.
                # Run a targeted broker reconcile pass every cycle so missed stream
                # events do not leave executions stuck OPEN in the dashboard.
                from app.services.strategy_executor import _reconcile_forex_closes

                timeout_s = max(
                    8.0,
                    min(float(getattr(cfg, "scan_interval_s", 20.0) or 20.0), 25.0),
                )
                await asyncio.wait_for(
                    _reconcile_forex_closes(user_id=demo_uid),
                    timeout=timeout_s,
                )
            except asyncio.TimeoutError:
                logger.warning(
                    "[gold-ai-trader] broker close reconcile timed out uid=%s",
                    demo_uid,
                )
            except Exception as exc:
                logger.warning(
                    "[gold-ai-trader] broker close reconcile failed uid=%s: %s",
                    demo_uid,
                    exc,
                )
        await sync_closed_trade_notifications(db, cfg)
    except Exception as exc:
        logger.warning("[gold-ai-trader] closed-outcome sync pass failed: %s", exc)
    finally:
        db.close()


async def _maybe_run_orb_strategy(
    *,
    db,
    cfg,
    session: str,
    now: datetime,
    price: float,
    source_tag: str,
) -> bool:
    """Run additive ORB detector + dedicated Claude call path."""
    if not bool(getattr(cfg, "orb_enabled", False)):
        return False
    try:
        signal, orb_state, detect_reason = await detect_orb_signal(
            db=db,
            cfg=cfg,
            session=session,
            now=now,
            user_id=cfg.demo_user_id,
        )
    except Exception as exc:
        logger.warning(
            "[gold-ai-trader] setup detector error setup=%s kind=%s: %s",
            "orb",
            "orb_breakout",
            exc,
            exc_info=True,
        )
        return False
    if not signal:
        if detect_reason not in {
            "forming_range",
            "trade_window_expired",
            "no_breakout",
            "range_unavailable",
            "breakout_already_processed",
            "waiting_new_bar_for_retest",
            "retest_not_confirmed",
            "breakout_seen_wait_retest",
            "no_post_range_bars",
        }:
            logger.info("[gold-ai-orb] skipped reason=%s session=%s", detect_reason, session)
        return False

    ok_orb, orb_reason = check_can_call_orb(db, cfg)
    if not ok_orb:
        logger.info("[gold-ai-orb] claude blocked reason=%s setup=%s", orb_reason, signal.setup_type)
        return False

    recent_tf = (getattr(cfg, "orb_timeframe", "5m") or "5m").strip().lower()
    recent_bars = await get_gold_ai_klines(recent_tf, 64, user_id=cfg.demo_user_id) or []
    context = build_orb_context(
        signal,
        session=session,
        cfg=cfg,
        now=now,
        recent_bars=recent_bars,
    )

    claude_timeout_s = max(
        15.0,
        float(os.environ.get("GOLD_AI_CLAUDE_DECIDE_TIMEOUT_S", "60")),
    )
    try:
        decision, reasoning, usage = await asyncio.wait_for(
            decide_orb(
                context,
                model=cfg.model,
                confidence_threshold=int(getattr(cfg, "orb_confidence_threshold", 55)),
            ),
            timeout=claude_timeout_s,
        )
    except asyncio.TimeoutError:
        logger.warning(
            "[gold-ai-orb] Claude decision timeout after %.1fs (setup=%s)",
            claude_timeout_s,
            signal.setup_type,
        )
        decision = {
            "action": "skip",
            "confidence": 0,
            "rationale": "Claude timeout",
        }
        reasoning = "Claude decision timeout"
        usage = {
            "tokens_in": 0,
            "tokens_out": 0,
            "cache_read_tokens": 0,
            "cache_write_tokens": 0,
            "cost_usd": 0.0,
        }

    action = (decision.get("action") or "skip").lower()
    conf = int(decision.get("confidence") or 0)
    price = await refresh_spot_after_claude(float(price), user_id=cfg.demo_user_id)
    row = GoldAiDecision(
        session=session,
        candidate_type=signal.setup_type,
        context_snapshot=context,
        reasoning=reasoning,
        decision=decision,
        action=action,
        confidence=conf,
        executed=False,
        tokens_in=int(usage.get("tokens_in", 0)),
        tokens_out=int(usage.get("tokens_out", 0)),
        cache_read_tokens=int(usage.get("cache_read_tokens", 0)),
        cache_write_tokens=int(usage.get("cache_write_tokens", 0)),
        cost_usd=float(usage.get("cost_usd", 0)),
    )
    db.add(row)
    db.commit()
    db.refresh(row)

    if orb_state is not None:
        orb_state.decision_id = row.id
        db.commit()

    runtime_state.note_decision(
        {
            "id": row.id,
            "action": action,
            "confidence": conf,
            "rationale": decision.get("rationale", ""),
        }
    )

    execution_id = None
    executed = False
    block_reason = None
    timing_ctx = {
        "decision_ts": row.ts.isoformat() + "Z" if getattr(row, "ts", None) else None,
        "validated_ts": None,
        "enqueued_ts": None,
        "broker_ack_ts": None,
    }
    if action == "take":
        k5 = await get_klines(SYMBOL, ASSET_CLASS, "5m", 60) or []
        atr = atr_from_klines(k5)
        k1h = await get_gold_ai_klines("1h", 50, user_id=cfg.demo_user_id) or []
        k_daily = await get_gold_ai_klines("1d", 5, user_id=cfg.demo_user_id) or []
        key_levels = collect_key_levels(float(price), session, cfg, now, k_daily, k1h, k5)
        decision["validator_profile"] = "orb"
        decision["orb_break_level"] = signal.break_level
        decision["orb_range_height"] = signal.range_height
        val_ok, val_reason, decision = validate_take_decision(
            decision,
            candidate_direction=signal.side,
            spot=float(price),
            atr=atr,
            setup_detail=(
                f"orb breakout level={signal.break_level:.2f} "
                f"range_high={signal.range_high:.2f} range_low={signal.range_low:.2f}"
            ),
            key_levels=key_levels,
        )
        row.decision = decision
        db.commit()
        timing_ctx["validated_ts"] = _utc_iso_now()
        if not val_ok:
            block_reason = val_reason
            logger.info(
                "[gold-ai-orb] validator rejected decision_id=%s setup=%s reason=%s",
                row.id,
                signal.setup_type,
                val_reason,
            )
        elif conf >= int(getattr(cfg, "orb_confidence_threshold", 55)):
            stale_ok, stale_reason = await _stale_entry_recheck(
                decision=decision,
                cfg=cfg,
                decision_ts=getattr(row, "ts", None),
                decision_id=row.id,
                setup_type=signal.setup_type,
            )
            if not stale_ok:
                block_reason = stale_reason
            else:
                row.decision = decision
                db.commit()
            ok_exec, exec_reason = check_can_execute(db, cfg, cfg.demo_user_id or 0)
            if ok_exec and stale_ok:
                timing_ctx["enqueued_ts"] = _utc_iso_now()
                orb_detail = (
                    f"orb breakout level={signal.break_level:.2f} "
                    f"range_high={signal.range_high:.2f} range_low={signal.range_low:.2f}"
                )
                exec_id = await execute_take(
                    db=db,
                    cfg=cfg,
                    decision=decision,
                    decision_id=row.id,
                    session=session,
                    setup_type=signal.setup_type,
                    timing_ctx=timing_ctx,
                    fire_context=_fire_context(
                        setup_type=signal.setup_type,
                        candidate_direction=signal.side,
                        setup_detail=orb_detail,
                        atr=atr,
                        key_levels=key_levels,
                    ),
                )
                if exec_id and exec_id > 0:
                    executed = True
                    execution_id = exec_id
                    row.executed = True
                    row.execution_id = exec_id
                    if orb_state is not None:
                        orb_state.execution_id = exec_id
                        orb_state.trades_taken = int(orb_state.trades_taken or 0) + 1
                        if orb_state.trades_taken >= max(
                            1, int(getattr(cfg, "orb_max_trades_per_session", 1))
                        ):
                            orb_state.status = "traded"
                    db.commit()
                elif exec_id and exec_id < 0:
                    block_reason = f"entry pending #{-exec_id} (limit/entry-watch)"
                    if orb_state is not None:
                        orb_state.trades_taken = int(orb_state.trades_taken or 0) + 1
                        if orb_state.trades_taken >= max(
                            1, int(getattr(cfg, "orb_max_trades_per_session", 1))
                        ):
                            orb_state.status = "traded"
                    db.commit()
                else:
                    block_reason = timing_ctx.get("block_reason") or "demo order rejected"
            else:
                block_reason = block_reason or exec_reason
        else:
            block_reason = (
                f"confidence {conf}% below "
                f"{int(getattr(cfg, 'orb_confidence_threshold', 55))}% threshold"
            )

    if action == "take":
        await notify_take_decision(
            candidate_type=signal.setup_type,
            session=session,
            decision=decision,
            confidence=conf,
            executed=executed,
            execution_id=execution_id,
            block_reason=block_reason,
        )

    logger.info(
        "[gold-ai] decision_id=%s confidence=%s%% setup=%s source=%s action=%s",
        row.id,
        conf,
        signal.setup_type,
        source_tag,
        action,
    )
    logger.info(
        "[gold-ai] calibration decision_id=%s confidence=%s%% setup=%s source=%s "
        "action=%s exec_id=%s session=%s",
        row.id,
        conf,
        signal.setup_type,
        source_tag,
        action,
        row.execution_id or execution_id or "none",
        session,
    )
    if action == "take":
        logger.info(
            "[gold-ai-latency] decision_id=%s setup=%s decision_ts=%s validated_ts=%s "
            "enqueued_ts=%s broker_ack_ts=%s exec_id=%s block_reason=%s",
            row.id,
            signal.setup_type,
            timing_ctx.get("decision_ts"),
            timing_ctx.get("validated_ts"),
            timing_ctx.get("enqueued_ts"),
            timing_ctx.get("broker_ack_ts"),
            row.execution_id or execution_id or "none",
            block_reason or "",
        )
        await sync_pending_entries(db, cfg, float(price))

    if row.execution_id:
        from app.strategy_models import StrategyExecution

        ex = db.query(StrategyExecution).filter_by(id=row.execution_id).first()
        record_outcome_from_execution(db, row.id, ex)
    return True


async def run_gold_ai_trader_loop() -> None:
    """Main scan → candidate → Claude → optional execute cycle."""
    global _prev_session
    env = env_defaults()
    if not env.enabled:
        runtime_state.note_dormant("disabled")
        return

    db = SessionLocal()
    try:
        cfg_row = seed_config_if_missing(db)
        cfg = merge_config(cfg_row, env)
    finally:
        db.close()

    if cfg.kill_switch or not cfg.enabled:
        runtime_state.note_dormant("killed" if cfg.kill_switch else "disabled")
        await asyncio.sleep(max(cfg.scan_interval_s, 15))
        return

    now = datetime.utcnow()
    session = active_session(now, cfg)
    if not session:
        if _prev_session == "new_york" and cfg.no_overnight:
            db = SessionLocal()
            try:
                await flatten_open_demo_positions(db, cfg)
            finally:
                db.close()
        _prev_session = None
        runtime_state.note_dormant("outside_session")
        await asyncio.sleep(max(cfg.scan_interval_s, 15))
        return

    _prev_session = session
    runtime_state.note_scan(session)

    orb_enabled = bool(getattr(cfg, "orb_enabled", False))
    killzone_blocked = killzone_only_enabled() and not in_killzone(now, session, cfg)
    if killzone_blocked and not orb_enabled and not killzone_override_enabled():
        runtime_state.note_dormant("outside_killzone")
        await asyncio.sleep(max(cfg.scan_interval_s, 15))
        return

    db = SessionLocal()
    try:
        funnel_record("scan", db=db, session=session)
        cfg_row = db.query(GoldAiConfig).filter_by(id=1).first()
        if cfg_row:
            cfg = merge_config(cfg_row, env)
        orb_enabled = bool(getattr(cfg, "orb_enabled", False))
        killzone_blocked = killzone_only_enabled() and not in_killzone(now, session, cfg)
        killzone_override_scan = killzone_blocked and killzone_override_enabled()

        if not killzone_blocked or killzone_override_scan:
            ok_call, reason = check_can_call_claude(db, cfg)
            if not ok_call:
                runtime_state.note_dormant(reason)
                if reason == "max_calls_day":
                    await maybe_notify_call_cap_reached(db, cfg)
                return

        market_data = await assess_gold_market_data(user_id=cfg.demo_user_id)
        data_ok, data_block = gold_data_ok_for_claude(market_data)
        source_tag = format_data_source(market_data)
        if not data_ok:
            if str(data_block).startswith("stale_klines:"):
                bar_age_s = market_data.get("kline_bar_age_s")
                fetch_age_s = market_data.get("kline_fetch_age_s")
                logger.warning(
                    "[gold-ai] stale skip source=%s bar_age_s=%s fetch_age_s=%s "
                    "trendbar_blocked=%s trendbar_reason=%s detail=%s",
                    source_tag,
                    f"{float(bar_age_s):.1f}" if bar_age_s is not None else "n/a",
                    f"{float(fetch_age_s):.1f}" if fetch_age_s is not None else "n/a",
                    bool(market_data.get("ctrader_trendbar_blocked")),
                    market_data.get("ctrader_trendbar_block_reason") or "",
                    market_data.get("stale_reason") or data_block,
                )
            logger.info(
                "[gold-ai] confidence=N/A source=%s decision=skip reason=%s",
                source_tag,
                data_block,
            )
            funnel_record(
                "data_blocked", reason=data_block, db=db, session=session,
            )
            runtime_state.note_dormant(f"data_quality:{data_block}")
            return

        news_gate = os.environ.get("GOLD_AI_NEWS_GATE", "true").strip().lower() in (
            "1", "true", "yes", "on",
        )
        if news_gate:
            from app.services.strategy_ta import eval_forex_news_avoidance

            news_ok, news_reason = await eval_forex_news_avoidance(
                {"minutes_before": 30, "minutes_after": 30, "min_impact": "high"},
                SYMBOL,
            )
            if not news_ok:
                logger.info(
                    "[gold-ai] confidence=N/A source=%s decision=skip reason=%s",
                    source_tag,
                    news_reason,
                )
                funnel_record(
                    "news_blocked", reason=news_reason, db=db, session=session,
                )
                runtime_state.note_dormant(f"news:{news_reason[:80]}")
                return

        orb_logged = await _maybe_run_orb_strategy(
            db=db,
            cfg=cfg,
            session=session,
            now=now,
            price=float(market_data["price"]),
            source_tag=source_tag,
        )

        if killzone_blocked and not killzone_override_enabled():
            runtime_state.note_dormant("outside_killzone")
            if orb_logged:
                await sync_closed_trade_notifications(db, cfg)
                await maybe_send_daily_summary(db, cfg)
                await maybe_run_learning_review(db, session, cfg)
            runtime_state.set_funnel(funnel_snapshot())
            return

        async with httpx.AsyncClient(timeout=15) as http:
            price, candidates = await scan_candidates(
                http,
                session=session,
                cfg=cfg,
                price=float(market_data["price"]),
                user_id=cfg.demo_user_id,
                db=db,
            )
        if not candidates or price is None:
            if orb_logged:
                await sync_closed_trade_notifications(db, cfg)
                await maybe_send_daily_summary(db, cfg)
                await maybe_run_learning_review(db, session, cfg)
            if killzone_override_scan:
                runtime_state.note_dormant("outside_killzone_low_confluence")
            runtime_state.set_funnel(funnel_snapshot())
            return

        await sync_pending_entries(db, cfg, float(price))

        k5 = await get_klines(SYMBOL, ASSET_CLASS, "5m", 60) or []
        atr = atr_from_klines(k5)

        candidate = None
        dedupe_reason = ""
        override_min_conf = killzone_override_min_confluence() if killzone_override_scan else 0
        for cand in pick_top_candidates(candidates, max_candidates_per_scan()):
            if killzone_override_scan:
                passed_n, total_n = candidate_confluence_counts(cand)
                if not candidate_meets_killzone_override(cand):
                    skip_reason = f"confluence_{passed_n}/{total_n}<{override_min_conf}"
                    funnel_record(
                        "override_confluence_skipped",
                        setup=cand.type,
                        reason=skip_reason,
                        db=db,
                        session=session,
                    )
                    logger.debug(
                        "[gold-ai-trader] killzone override skip %s: %s",
                        cand.type,
                        skip_reason,
                    )
                    continue
            ok_dedupe, dedupe_reason = should_invoke_claude(
                db, cand, float(price), atr, setup_cooldown_s=_setup_cooldown_s()
            )
            if ok_dedupe:
                candidate = cand
                break
            funnel_record(
                "dedupe_skipped",
                setup=cand.type,
                reason=dedupe_reason,
                db=db,
                session=session,
            )
            logger.debug("[gold-ai-trader] claude dedupe skip %s: %s", cand.type, dedupe_reason)

        if not candidate:
            if killzone_override_scan:
                runtime_state.note_dormant("outside_killzone_low_confluence")
            else:
                runtime_state.note_dormant(dedupe_reason or "all_candidates_deduped")
            runtime_state.set_funnel(funnel_snapshot())
            return

        runtime_state.note_candidate(
            {
                "type": candidate.type,
                "direction": candidate.direction,
                "detail": candidate.detail,
                "price": price,
            }
        )

        context = await build_context_snapshot(
            candidate=candidate,
            price=price,
            session=session,
            db=db,
            cfg=cfg,
            user_id=cfg.demo_user_id,
            market_data=market_data,
            smt=candidate.raw.get("smt"),
            cisd=candidate.raw.get("cisd"),
        )

        ok_call, reason = check_can_call_claude(db, cfg)
        if not ok_call:
            runtime_state.note_dormant(reason)
            if reason == "max_calls_day":
                await maybe_notify_call_cap_reached(db, cfg)
            return

        claude_timeout_s = max(
            15.0,
            float(os.environ.get("GOLD_AI_CLAUDE_DECIDE_TIMEOUT_S", "60")),
        )
        try:
            decision, reasoning, usage = await asyncio.wait_for(
                decide(
                    context,
                    model=cfg.model,
                    confidence_threshold=cfg.confidence_threshold,
                ),
                timeout=claude_timeout_s,
            )
        except asyncio.TimeoutError:
            logger.warning(
                "[gold-ai-trader] Claude decision timeout after %.1fs (setup=%s)",
                claude_timeout_s,
                candidate.type,
            )
            decision = {
                "action": "skip",
                "confidence": 0,
                "rationale": "Claude timeout",
            }
            reasoning = "Claude decision timeout"
            usage = {
                "tokens_in": 0,
                "tokens_out": 0,
                "cache_read_tokens": 0,
                "cache_write_tokens": 0,
                "cost_usd": 0.0,
            }
        record_claude_invocation(candidate)
        funnel_record("claude_called", setup=candidate.type, db=db, session=session)
        action = (decision.get("action") or "skip").lower()
        conf = int(decision.get("confidence") or 0)
        price = await refresh_spot_after_claude(float(price), user_id=cfg.demo_user_id)
        if action == "take":
            funnel_record("claude_take", setup=candidate.type, db=db, session=session)
        else:
            funnel_record("claude_skip", setup=candidate.type, db=db, session=session)

        row = GoldAiDecision(
            session=session,
            candidate_type=candidate.type,
            context_snapshot=context,
            reasoning=reasoning,
            decision=decision,
            action=action,
            confidence=conf,
            executed=False,
            tokens_in=int(usage.get("tokens_in", 0)),
            tokens_out=int(usage.get("tokens_out", 0)),
            cache_read_tokens=int(usage.get("cache_read_tokens", 0)),
            cache_write_tokens=int(usage.get("cache_write_tokens", 0)),
            cost_usd=float(usage.get("cost_usd", 0)),
        )
        db.add(row)
        db.commit()
        db.refresh(row)

        logger.info(
            "[gold-ai] decision_id=%s confidence=%s%% setup=%s source=%s action=%s",
            row.id,
            conf,
            candidate.type,
            source_tag,
            action,
        )

        runtime_state.note_decision(
            {
                "id": row.id,
                "action": action,
                "confidence": conf,
                "rationale": decision.get("rationale", ""),
            }
        )

        execution_id = None
        if action == "take":
            executed = False
            block_reason = None
            validator_block = None
            timing_ctx = {
                "decision_ts": row.ts.isoformat() + "Z" if getattr(row, "ts", None) else None,
                "validated_ts": None,
                "enqueued_ts": None,
                "broker_ack_ts": None,
            }
            k1h = await get_gold_ai_klines("1h", 50, user_id=cfg.demo_user_id) or []
            k_daily = await get_gold_ai_klines("1d", 5, user_id=cfg.demo_user_id) or []
            key_levels = collect_key_levels(
                float(price), session, cfg, now, k_daily, k1h, k5,
            )
            if candidate.type.startswith("momentum_flag_break_"):
                decision["validator_profile"] = "momentum_flag"
                momentum_break_level = candidate.raw.get("momentum_break_level")
                try:
                    if momentum_break_level is not None:
                        decision["momentum_break_level"] = float(momentum_break_level)
                except (TypeError, ValueError):
                    pass
                decision["momentum_used_retest"] = bool(
                    candidate.raw.get("momentum_used_retest")
                )
            elif candidate.type.startswith("liquidity_grab_"):
                decision["validator_profile"] = "liquidity_grab"
                liq_grab_mss_level = candidate.raw.get("liq_grab_mss_level")
                try:
                    if liq_grab_mss_level is not None:
                        decision["liq_grab_mss_level"] = float(liq_grab_mss_level)
                except (TypeError, ValueError):
                    pass
            val_ok, val_reason, decision = validate_take_decision(
                decision,
                candidate_direction=candidate.direction,
                spot=float(price),
                atr=atr,
                setup_detail=candidate.detail,
                key_levels=key_levels,
            )
            row.decision = decision
            db.commit()
            timing_ctx["validated_ts"] = _utc_iso_now()
            if not val_ok:
                validator_block = val_reason
                funnel_record(
                    "validator_rejected",
                    setup=candidate.type,
                    reason=val_reason,
                    db=db,
                    session=session,
                    decision_id=row.id,
                )
                logger.info(
                    "[gold-ai] validator rejected decision_id=%s setup=%s reason=%s",
                    row.id,
                    candidate.type,
                    val_reason,
                )

            if validator_block:
                block_reason = validator_block
            elif conf >= cfg.confidence_threshold:
                stale_ok, stale_reason = await _stale_entry_recheck(
                    decision=decision,
                    cfg=cfg,
                    decision_ts=getattr(row, "ts", None),
                    decision_id=row.id,
                    setup_type=candidate.type,
                )
                if not stale_ok:
                    block_reason = stale_reason
                else:
                    row.decision = decision
                    db.commit()
                ok_exec, exec_reason = check_can_execute(db, cfg, cfg.demo_user_id or 0)
                if ok_exec and stale_ok:
                    timing_ctx["enqueued_ts"] = _utc_iso_now()
                    exec_id = await execute_take(
                        db=db,
                        cfg=cfg,
                        decision=decision,
                        decision_id=row.id,
                        session=session,
                        setup_type=candidate.type,
                        timing_ctx=timing_ctx,
                        fire_context=_fire_context(
                            setup_type=candidate.type,
                            candidate_direction=candidate.direction,
                            setup_detail=candidate.detail,
                            atr=atr,
                            key_levels=key_levels,
                        ),
                    )
                    if exec_id and exec_id > 0:
                        executed = True
                        execution_id = exec_id
                        row.executed = True
                        row.execution_id = exec_id
                        db.commit()
                        funnel_record(
                            "executed",
                            setup=candidate.type,
                            db=db,
                            session=session,
                            decision_id=row.id,
                        )

                        ok_live, live_reason = check_can_execute_live_mirror(
                            db, cfg, cfg.demo_user_id or 0
                        )
                        if ok_live:
                            live_exec_id = await execute_live_mirror_take(
                                db=db,
                                cfg=cfg,
                                decision=decision,
                                decision_id=row.id,
                                demo_execution_id=exec_id,
                            )
                            if live_exec_id:
                                row.live_mirror_execution_id = live_exec_id
                                row.live_mirror_status = "pending"
                                db.commit()
                            else:
                                row.live_mirror_status = "failed"
                                row.live_mirror_error = "live mirror enqueue rejected"
                                db.commit()
                        elif cfg.live_mirror_enabled:
                            row.live_mirror_status = "skipped"
                            row.live_mirror_error = live_reason
                            db.commit()
                    elif exec_id and exec_id < 0:
                        block_reason = f"entry pending #{-exec_id} (limit/entry-watch)"
                        funnel_record(
                            "pending_entry",
                            setup=candidate.type,
                            db=db,
                            session=session,
                            decision_id=row.id,
                        )
                    else:
                        block_reason = timing_ctx.get("block_reason") or "demo order rejected"
                else:
                    block_reason = block_reason or exec_reason
                    logger.info("[gold-ai-trader] execute blocked: %s", exec_reason)
            else:
                block_reason = (
                    f"confidence {conf}% below {cfg.confidence_threshold}% threshold"
                )

            await notify_take_decision(
                candidate_type=candidate.type,
                session=session,
                decision=decision,
                confidence=conf,
                executed=executed,
                execution_id=execution_id,
                block_reason=block_reason,
            )
            logger.info(
                "[gold-ai-latency] decision_id=%s setup=%s decision_ts=%s validated_ts=%s "
                "enqueued_ts=%s broker_ack_ts=%s exec_id=%s block_reason=%s",
                row.id,
                candidate.type,
                timing_ctx.get("decision_ts"),
                timing_ctx.get("validated_ts"),
                timing_ctx.get("enqueued_ts"),
                timing_ctx.get("broker_ack_ts"),
                row.execution_id or execution_id or "none",
                block_reason or "",
            )
            await sync_pending_entries(db, cfg, float(price))

        logger.info(
            "[gold-ai] calibration decision_id=%s confidence=%s%% setup=%s source=%s "
            "action=%s exec_id=%s session=%s",
            row.id,
            conf,
            candidate.type,
            source_tag,
            action,
            row.execution_id or execution_id or "none",
            session,
        )

        # Sync outcomes for closed trades + Telegram close alerts
        if row.execution_id:
            from app.strategy_models import StrategyExecution

            ex = db.query(StrategyExecution).filter_by(id=row.execution_id).first()
            record_outcome_from_execution(db, row.id, ex)

        await sync_closed_trade_notifications(db, cfg)
        await maybe_send_daily_summary(db, cfg)
        await maybe_run_learning_review(db, session, cfg)
        runtime_state.set_funnel(funnel_snapshot())
    except Exception as e:
        logger.error("[gold-ai-trader] loop error: %s", e, exc_info=True)
        runtime_state.note_error(str(e))
    finally:
        db.close()


async def _locked_loop_forever() -> None:
    """Advisory lock with Neon keepalive — hold lock across scan cycles."""
    from app.executor_lock import log_executor_lock_keepalive_config

    global _lock_conn
    delay = max(15.0, float(os.environ.get("GOLD_AI_TRADER_SCAN_INTERVAL_S", "20")))
    cycle_timeout_s = max(
        delay * 2.0,
        float(os.environ.get("GOLD_AI_LOOP_CYCLE_TIMEOUT_S", "120")),
    )
    reconcile_timeout_s = max(
        10.0,
        float(os.environ.get("GOLD_AI_LOOP_RECON_TIMEOUT_S", "45")),
    )
    reclaim_after_misses = max(
        3,
        int(os.environ.get("GOLD_AI_LOCK_RECLAIM_AFTER_MISSES", "6")),
    )
    reclaim_min_idle_s = max(
        30.0,
        float(os.environ.get("GOLD_AI_LOCK_RECLAIM_MIN_IDLE_S", "120")),
    )
    miss_streak = 0
    log_executor_lock_keepalive_config("gold-ai-trader:loop")
    logger.info("[gold-ai-trader] background loop starting (interval=%ss)", delay)
    try:
        while True:
            if _lock_conn is None:
                try:
                    _lock_conn = await asyncio.to_thread(_acquire_gold_ai_lock)
                except Exception as exc:
                    logger.error("[gold-ai-trader] lock acquire error: %s", exc, exc_info=True)
                    _lock_conn = None
                if _lock_conn is None:
                    miss_streak += 1
                    if miss_streak % reclaim_after_misses == 0:
                        try:
                            reclaimed = await asyncio.to_thread(
                                _reclaim_stale_gold_ai_locks,
                                min_idle_seconds=reclaim_min_idle_s,
                            )
                            if reclaimed:
                                logger.warning(
                                    "[gold-ai-trader] reclaimed %s stale lock holder(s) after %s misses",
                                    reclaimed,
                                    miss_streak,
                                )
                        except Exception as exc:
                            logger.warning("[gold-ai-trader] stale lock reclaim failed: %s", exc)
                    await asyncio.sleep(delay)
                    continue
                miss_streak = 0
                logger.info("[gold-ai-trader] advisory lock acquired")

            if not await asyncio.to_thread(_ping_lock_connection, _lock_conn):
                logger.warning("[gold-ai-trader] advisory lock connection lost — reconnecting")
                _lock_conn = await asyncio.to_thread(_reconnect_gold_ai_lock, _lock_conn)
                if _lock_conn is None:
                    logger.error(
                        "[gold-ai-trader] advisory lock re-claim failed — retry in %ss",
                        delay,
                    )
                    await asyncio.sleep(delay)
                    continue

            try:
                await asyncio.wait_for(run_gold_ai_trader_loop(), timeout=cycle_timeout_s)
            except asyncio.TimeoutError:
                runtime_state.note_error(f"scan_cycle_timeout>{int(cycle_timeout_s)}s")
                logger.error(
                    "[gold-ai-trader] scan cycle timeout after %.1fs — continuing",
                    cycle_timeout_s,
                )
            except Exception as e:
                logger.error("[gold-ai-trader] lock loop cycle error: %s", e, exc_info=True)
            try:
                await asyncio.wait_for(
                    _sync_closed_outcomes_pass(), timeout=reconcile_timeout_s
                )
            except asyncio.TimeoutError:
                logger.error(
                    "[gold-ai-trader] reconcile pass timeout after %.1fs — continuing",
                    reconcile_timeout_s,
                )
            except Exception as e:
                logger.error("[gold-ai-trader] lock loop reconcile pass: %s", e, exc_info=True)

            await asyncio.sleep(delay)
    except asyncio.CancelledError:
        logger.warning("[gold-ai-trader] background loop cancelled")
        raise
    finally:
        conn = _lock_conn
        _lock_conn = None
        if conn is not None:
            await asyncio.to_thread(_release_gold_ai_lock, conn)
        logger.warning("[gold-ai-trader] background loop stopped")


async def maybe_start_background_loop() -> None:
    global _loop_task, _watchdog_task
    if not gold_ai_trader_enabled():
        logger.info("[gold-ai-trader] disabled (GOLD_AI_TRADER_ENABLED=false)")
        return
    if _loop_task is None or _loop_task.done():
        _schedule_loop_task()
    if _watchdog_task is None or _watchdog_task.done():
        _schedule_watchdog_task()
