"""Isolated background scan loop (standalone process)."""
from __future__ import annotations

import asyncio
import logging
import os
import time
from datetime import datetime

import httpx

from app.gold_ai_trader.config import env_defaults, gold_ai_trader_enabled
from app.gold_ai_trader.data_quality import (
    assess_gold_market_data,
    format_data_source,
    gold_data_ok_for_claude,
)
from app.gold_ai_trader.db_thread import db_commit, run_in_db_thread, run_with_db, with_db_session
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
    meets_min_confluence_for_take,
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
from app.gold_ai_trader.pending_entry import sync_pending_entries, pending_status_label
from app.gold_ai_trader.telegram_notify import (
    maybe_send_daily_summary,
    notify_take_decision,
    sync_closed_trade_notifications,
    maybe_notify_call_cap_reached,
    maybe_notify_fallback_klines_blocked,
)
from app.gold_ai_trader.models import GoldAiConfig, GoldAiDecision
from app.gold_ai_trader import state as runtime_state

logger = logging.getLogger(__name__)

_loop_task: asyncio.Task | None = None
_watchdog_task: asyncio.Task | None = None
_prev_session: str | None = None
_loop_task_started_mono = 0.0
_watchdog_last_restart_mono = 0.0
_restart_lock: asyncio.Lock | None = None


def gold_ai_loop_disabled_in_gunicorn() -> bool:
    return os.environ.get("DISABLE_GOLD_AI_IN_GUNICORN", "").lower() in (
        "1",
        "true",
        "yes",
    )


def is_standalone_gold_ai() -> bool:
    return os.environ.get("GOLD_AI_STANDALONE", "").lower() in ("1", "true", "yes")


def _load_merged_config(db, env):
    cfg_row = seed_config_if_missing(db)
    return merge_config(cfg_row, env)


def _persist_scan_and_reload_cfg(db, session: str, env):
    funnel_record("scan", db=db, session=session)
    cfg_row = db.query(GoldAiConfig).filter_by(id=1).first()
    if cfg_row:
        return merge_config(cfg_row, env)
    return None


def _check_can_call_claude_db(db, cfg):
    return check_can_call_claude(db, cfg)


def _check_can_call_orb_db(db, cfg):
    return check_can_call_orb(db, cfg)


def _check_can_execute_db(db, cfg, user_id: int):
    return check_can_execute(db, cfg, user_id)


def _check_can_execute_live_mirror_db(db, cfg, user_id: int):
    return check_can_execute_live_mirror(db, cfg, user_id)


def _record_funnel_db(db, event: str, **kwargs):
    funnel_record(event, db=db, **kwargs)


def _should_invoke_claude_db(db, cand, price: float, atr: float, setup_cooldown_s: float):
    return should_invoke_claude(db, cand, price, atr, setup_cooldown_s=setup_cooldown_s)


def _persist_orb_state_link(db, orb_state):
    from app.gold_ai_trader.orb import _persist_state

    _persist_state(db, orb_state)


def _update_decision_row(db, row_id: int, decision: dict, *, executed: bool | None = None, execution_id: int | None = None):
    row = db.query(GoldAiDecision).filter_by(id=row_id).first()
    if not row:
        return
    row.decision = decision
    if executed is not None:
        row.executed = executed
    if execution_id is not None:
        row.execution_id = execution_id
    db.commit()


def _finalize_orb_execution_state(db, row_id: int, exec_id: int, orb_state):
    row = db.query(GoldAiDecision).filter_by(id=row_id).first()
    if row:
        row.executed = True
        row.execution_id = exec_id
    if orb_state is not None:
        from app.gold_ai_trader.orb import _persist_state

        orb_state.execution_id = exec_id
        orb_state.trades_taken = int(orb_state.trades_taken or 0) + 1
        if orb_state.trades_taken >= 1:
            orb_state.status = "traded"
        _persist_state(db, orb_state)
    db.commit()


def _record_outcome_for_execution(db, decision_id: int, execution_id: int):
    from app.strategy_models import StrategyExecution

    ex = db.query(StrategyExecution).filter_by(id=execution_id).first()
    record_outcome_from_execution(db, decision_id, ex)


async def _post_scan_tail(cfg, session: str) -> None:
    await _call_with_db_session(sync_closed_trade_notifications, cfg=cfg)
    await _call_with_db_session(maybe_send_daily_summary, cfg=cfg)
    await _call_with_db_session(maybe_run_learning_review, session=session, cfg=cfg)
    runtime_state.set_funnel(funnel_snapshot())


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
    _loop_task = asyncio.create_task(_scan_loop_forever())
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
    cfg = with_db_session(_load_merged_config)(env)
    if not cfg.enabled:
        return False, bool(cfg.kill_switch), None, None
    session = active_session(datetime.utcnow(), cfg)
    last = _freshest_scan_heartbeat_utc()
    age_s = (datetime.utcnow() - last).total_seconds() if last else None
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
            logger.warning("[gold-ai-trader] loop cancel wait failed (%s): %s", reason, exc)
    _loop_task = None


async def _restart_background_loop(reason: str) -> str:
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
        await _stop_loop_task(reason)
        _schedule_loop_task()
        return "restarted"


def _freshest_scan_heartbeat_utc() -> datetime | None:
    """Best scan heartbeat across local runtime + persisted funnel events."""
    from app.database import SessionLocal

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


def scan_heartbeat_age_seconds() -> float | None:
    last = _freshest_scan_heartbeat_utc()
    if last is None:
        return None
    return max(0.0, (datetime.utcnow() - last).total_seconds())


async def ensure_scan_liveness() -> str:
    """Read-only scan health for status API (recovery is watchdog-owned)."""
    if not gold_ai_trader_enabled():
        return "disabled"
    stale_after_s = max(
        60.0,
        float(os.environ.get("GOLD_AI_ON_DEMAND_SCAN_STALE_AFTER_S", "120")),
    )
    last = await run_in_db_thread(_freshest_scan_heartbeat_utc)
    age_s = (datetime.utcnow() - last).total_seconds() if last else None
    if age_s is not None and age_s <= stale_after_s:
        return "healthy"
    return "stale"


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
                await _restart_background_loop("loop_task_missing_or_done")
                await asyncio.sleep(interval_s)
                continue
            enabled, kill_switch, session, age_s = await run_in_db_thread(_watchdog_snapshot)
            if not enabled or kill_switch or not session:
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
                await _restart_background_loop(reason)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.error("[gold-ai-trader] watchdog error: %s", exc, exc_info=True)
        await asyncio.sleep(interval_s)


async def _sync_closed_outcomes_pass() -> None:
    """Run close reconciliation every loop cycle, independent of scan gating."""

    def _load_demo_uid(db):
        cfg_row = seed_config_if_missing(db)
        cfg = merge_config(cfg_row, env_defaults())
        return int(getattr(cfg, "demo_user_id", 0) or 0), cfg

    try:
        demo_uid, cfg = await run_with_db(_load_demo_uid)
        if demo_uid > 0:
            try:
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
        await _call_with_db_session(sync_closed_trade_notifications, cfg=cfg)
    except Exception as exc:
        logger.warning("[gold-ai-trader] closed-outcome sync pass failed: %s", exc)


async def _maybe_run_orb_strategy(
    *,
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

    ok_orb, orb_reason = await run_with_db(_check_can_call_orb_db, cfg)
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

    row = await run_with_db(
        _save_orb_decision,
        session_name=session,
        signal=signal,
        context=context,
        reasoning=reasoning,
        decision=decision,
        action=action,
        conf=conf,
        usage=usage,
    )

    if orb_state is not None:
        orb_state.decision_id = row.id
        await run_with_db(_persist_orb_state_link, orb_state)

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
        await run_with_db(_update_decision_row, row.id, decision)
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
                await run_with_db(_update_decision_row, row.id, decision)
            ok_exec, exec_reason = await run_with_db(
                _check_can_execute_db, cfg, cfg.demo_user_id or 0
            )
            if ok_exec and stale_ok:
                timing_ctx["enqueued_ts"] = _utc_iso_now()
                orb_detail = (
                    f"orb breakout level={signal.break_level:.2f} "
                    f"range_high={signal.range_high:.2f} range_low={signal.range_low:.2f}"
                )
                exec_id = await _call_with_db_session(
                    execute_take,
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
                    await run_with_db(
                        _finalize_orb_execution_state,
                        row.id,
                        exec_id,
                        orb_state,
                    )
                elif exec_id and exec_id < 0:
                    block_reason = await run_with_db(pending_status_label, -exec_id)
                    if orb_state is not None:
                        orb_state.trades_taken = int(orb_state.trades_taken or 0) + 1
                        if orb_state.trades_taken >= max(
                            1, int(getattr(cfg, "orb_max_trades_per_session", 1))
                        ):
                            orb_state.status = "traded"
                        await run_with_db(_persist_orb_state_link, orb_state)
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
        await _call_with_db_session(sync_pending_entries, cfg=cfg, spot=float(price))

    if row.execution_id:
        await run_with_db(_record_outcome_for_execution, row.id, row.execution_id)
    return True


def _save_orb_decision(
    db,
    *,
    session_name: str,
    signal,
    context: str,
    reasoning: str,
    decision: dict,
    action: str,
    conf: int,
    usage: dict,
):
    row = GoldAiDecision(
        session=session_name,
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
    return row


def _save_scan_decision(
    db,
    *,
    session_name: str,
    candidate,
    context: str,
    reasoning: str,
    decision: dict,
    action: str,
    conf: int,
    usage: dict,
):
    row = GoldAiDecision(
        session=session_name,
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
    return row


def _update_live_mirror_row(
    db,
    row_id: int,
    *,
    live_exec_id: int | None,
    status: str,
    error: str | None = None,
):
    row = db.query(GoldAiDecision).filter_by(id=row_id).first()
    if not row:
        return
    if live_exec_id:
        row.live_mirror_execution_id = live_exec_id
    row.live_mirror_status = status
    if error is not None:
        row.live_mirror_error = error
    db.commit()


async def run_gold_ai_trader_loop() -> None:
    """Main scan → candidate → Claude → optional execute cycle."""
    global _prev_session
    env = env_defaults()
    if not env.enabled:
        runtime_state.note_dormant("disabled")
        return

    cfg = await run_with_db(_load_merged_config, env)

    if cfg.kill_switch or not cfg.enabled:
        runtime_state.note_dormant("killed" if cfg.kill_switch else "disabled")
        await asyncio.sleep(max(cfg.scan_interval_s, 15))
        return

    now = datetime.utcnow()
    session = active_session(now, cfg)
    if not session:
        if _prev_session == "new_york" and cfg.no_overnight:
            await _call_with_db_session(flatten_open_demo_positions, cfg=cfg)
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

    try:
        reloaded = await run_with_db(_persist_scan_and_reload_cfg, session, env)
        if reloaded is not None:
            cfg = reloaded
        orb_enabled = bool(getattr(cfg, "orb_enabled", False))
        killzone_blocked = killzone_only_enabled() and not in_killzone(now, session, cfg)
        killzone_override_scan = killzone_blocked and killzone_override_enabled()

        if not killzone_blocked or killzone_override_scan:
            ok_call, reason = await run_with_db(_check_can_call_claude_db, cfg)
            if not ok_call:
                runtime_state.note_dormant(reason)
                if reason == "max_calls_day":
                    await _call_with_db_session(maybe_notify_call_cap_reached, cfg=cfg)
                return

        market_data = await assess_gold_market_data(user_id=cfg.demo_user_id)
        data_ok, data_block = gold_data_ok_for_claude(market_data)
        source_tag = format_data_source(market_data)
        if not data_ok:
            if str(data_block).startswith("fallback_klines:"):
                await maybe_notify_fallback_klines_blocked(
                    str(data_block), source_tag=source_tag,
                )
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
            await run_with_db(
                _record_funnel_db,
                "data_blocked",
                reason=data_block,
                session=session,
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
                await run_with_db(
                    _record_funnel_db,
                    "news_blocked",
                    reason=news_reason,
                    session=session,
                )
                runtime_state.note_dormant(f"news:{news_reason[:80]}")
                return

        orb_logged = await _maybe_run_orb_strategy(
            cfg=cfg,
            session=session,
            now=now,
            price=float(market_data["price"]),
            source_tag=source_tag,
        )

        if killzone_blocked and not killzone_override_enabled():
            runtime_state.note_dormant("outside_killzone")
            if orb_logged:
                await _post_scan_tail(cfg, session)
            else:
                runtime_state.set_funnel(funnel_snapshot())
            return

        async with httpx.AsyncClient(timeout=15) as http:
            price, candidates = await _call_with_db_session(
                scan_candidates,
                http,
                session=session,
                cfg=cfg,
                price=float(market_data["price"]),
                user_id=cfg.demo_user_id,
            )
        if not candidates or price is None:
            if orb_logged:
                await _post_scan_tail(cfg, session)
            else:
                if killzone_override_scan:
                    runtime_state.note_dormant("outside_killzone_low_confluence")
                runtime_state.set_funnel(funnel_snapshot())
            return

        await _call_with_db_session(sync_pending_entries, cfg=cfg, spot=float(price))

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
                    await run_with_db(
                        _record_funnel_db,
                        "override_confluence_skipped",
                        setup=cand.type,
                        reason=skip_reason,
                        session=session,
                    )
                    logger.debug(
                        "[gold-ai-trader] killzone override skip %s: %s",
                        cand.type,
                        skip_reason,
                    )
                    continue
            ok_dedupe, dedupe_reason = await run_with_db(
                _should_invoke_claude_db,
                cand,
                float(price),
                atr,
                _setup_cooldown_s(),
            )
            if ok_dedupe:
                candidate = cand
                break
            await run_with_db(
                _record_funnel_db,
                "dedupe_skipped",
                setup=cand.type,
                reason=dedupe_reason,
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
            cfg=cfg,
            user_id=cfg.demo_user_id,
            market_data=market_data,
            smt=candidate.raw.get("smt"),
            cisd=candidate.raw.get("cisd"),
        )

        ok_call, reason = await run_with_db(_check_can_call_claude_db, cfg)
        if not ok_call:
            runtime_state.note_dormant(reason)
            if reason == "max_calls_day":
                await _call_with_db_session(maybe_notify_call_cap_reached, cfg=cfg)
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
        await run_with_db(
            _record_funnel_db,
            "claude_called",
            setup=candidate.type,
            session=session,
        )
        action = (decision.get("action") or "skip").lower()
        conf = int(decision.get("confidence") or 0)
        price = await refresh_spot_after_claude(float(price), user_id=cfg.demo_user_id)
        if action == "take":
            await run_with_db(
                _record_funnel_db,
                "claude_take",
                setup=candidate.type,
                session=session,
            )
        else:
            await run_with_db(
                _record_funnel_db,
                "claude_skip",
                setup=candidate.type,
                session=session,
            )

        row = await run_with_db(
            _save_scan_decision,
            session_name=session,
            candidate=candidate,
            context=context,
            reasoning=reasoning,
            decision=decision,
            action=action,
            conf=conf,
            usage=usage,
        )

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
            await run_with_db(_update_decision_row, row.id, decision)
            timing_ctx["validated_ts"] = _utc_iso_now()
            if not val_ok:
                validator_block = val_reason
                await run_with_db(
                    _record_funnel_db,
                    "validator_rejected",
                    setup=candidate.type,
                    reason=val_reason,
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
                    await run_with_db(_update_decision_row, row.id, decision)
                ok_exec, exec_reason = await run_with_db(
                    _check_can_execute_db, cfg, cfg.demo_user_id or 0
                )
                conf_ok, conf_reason = meets_min_confluence_for_take(candidate)
                if not conf_ok:
                    block_reason = conf_reason
                    await run_with_db(
                        _record_funnel_db,
                        "confluence_blocked",
                        setup=candidate.type,
                        reason=conf_reason,
                        session=session,
                        decision_id=row.id,
                    )
                if ok_exec and stale_ok and conf_ok:
                    timing_ctx["enqueued_ts"] = _utc_iso_now()
                    exec_id = await _call_with_db_session(
                        execute_take,
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
                        await run_with_db(
                            _update_decision_row,
                            row.id,
                            decision,
                            executed=True,
                            execution_id=exec_id,
                        )
                        await run_with_db(
                            _record_funnel_db,
                            "executed",
                            setup=candidate.type,
                            session=session,
                            decision_id=row.id,
                        )

                        ok_live, live_reason = await run_with_db(
                            _check_can_execute_live_mirror_db,
                            cfg,
                            cfg.demo_user_id or 0,
                        )
                        if ok_live:
                            live_exec_id = await _call_with_db_session(
                                execute_live_mirror_take,
                                cfg=cfg,
                                decision=decision,
                                decision_id=row.id,
                                demo_execution_id=exec_id,
                            )
                            if live_exec_id:
                                await run_with_db(
                                    _update_live_mirror_row,
                                    row.id,
                                    live_exec_id=live_exec_id,
                                    status="pending",
                                )
                            else:
                                await run_with_db(
                                    _update_live_mirror_row,
                                    row.id,
                                    live_exec_id=None,
                                    status="failed",
                                    error="live mirror enqueue rejected",
                                )
                        elif cfg.live_mirror_enabled:
                            await run_with_db(
                                _update_live_mirror_row,
                                row.id,
                                live_exec_id=None,
                                status="skipped",
                                error=live_reason,
                            )
                    elif exec_id and exec_id < 0:
                        block_reason = await run_with_db(pending_status_label, -exec_id)
                        await run_with_db(
                            _record_funnel_db,
                            "pending_entry",
                            setup=candidate.type,
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
            await _call_with_db_session(sync_pending_entries, cfg=cfg, spot=float(price))

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

        if row.execution_id:
            await run_with_db(_record_outcome_for_execution, row.id, row.execution_id)

        await _post_scan_tail(cfg, session)
    except Exception as e:
        logger.error("[gold-ai-trader] loop error: %s", e, exc_info=True)
        runtime_state.note_error(str(e))


async def _scan_loop_forever() -> None:
    """Dedicated-process scan driver — no advisory lock."""
    delay = max(15.0, float(os.environ.get("GOLD_AI_TRADER_SCAN_INTERVAL_S", "20")))
    cycle_timeout_s = max(
        delay * 2.0,
        float(os.environ.get("GOLD_AI_LOOP_CYCLE_TIMEOUT_S", "120")),
    )
    reconcile_timeout_s = max(
        10.0,
        float(os.environ.get("GOLD_AI_LOOP_RECON_TIMEOUT_S", "45")),
    )
    logger.info("[gold-ai-trader] background loop starting (interval=%ss)", delay)
    try:
        while True:
            try:
                await asyncio.wait_for(run_gold_ai_trader_loop(), timeout=cycle_timeout_s)
            except asyncio.TimeoutError:
                runtime_state.note_error(f"scan_cycle_timeout>{int(cycle_timeout_s)}s")
                logger.error(
                    "[gold-ai-trader] scan cycle timeout after %.1fs — continuing",
                    cycle_timeout_s,
                )
            except Exception as e:
                logger.error("[gold-ai-trader] scan loop cycle error: %s", e, exc_info=True)
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
                logger.error("[gold-ai-trader] reconcile pass error: %s", e, exc_info=True)

            await asyncio.sleep(delay)
    except asyncio.CancelledError:
        logger.warning("[gold-ai-trader] background loop cancelled")
        raise
    finally:
        logger.warning("[gold-ai-trader] background loop stopped")


async def start_gold_ai_trader_loop() -> None:
    """Start scan loop + watchdog (standalone runner only)."""
    global _loop_task, _watchdog_task
    if not gold_ai_trader_enabled():
        logger.info("[gold-ai-trader] disabled (GOLD_AI_TRADER_ENABLED=false)")
        return
    if gold_ai_loop_disabled_in_gunicorn() and not is_standalone_gold_ai():
        logger.info(
            "[gold-ai-trader] loop disabled in gunicorn (DISABLE_GOLD_AI_IN_GUNICORN=1)"
        )
        return
    if _loop_task is None or _loop_task.done():
        _schedule_loop_task()
    if _watchdog_task is None or _watchdog_task.done():
        _schedule_watchdog_task()


async def maybe_start_background_loop() -> None:
    """Backward-compatible alias — prefer start_gold_ai_trader_loop."""
    await start_gold_ai_trader_loop()
