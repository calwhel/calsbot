"""Isolated background loop with advisory lock."""
from __future__ import annotations

import asyncio
import logging
import os
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
    should_invoke_claude,
    call_stats_today,
)
from app.gold_ai_trader.decision_validator import validate_take_decision
from app.gold_ai_trader.funnel import record as funnel_record, snapshot as funnel_snapshot
from app.gold_ai_trader.setup_toggles import max_candidates_per_scan
from app.gold_ai_trader.klines import get_gold_ai_klines
from app.services.tradfi_prices import get_klines
from app.gold_ai_trader.config import SYMBOL, ASSET_CLASS
from app.gold_ai_trader.context import build_context_snapshot
from app.gold_ai_trader.claude import decide
from app.gold_ai_trader.executor import execute_take, execute_live_mirror_take, flatten_open_demo_positions
from app.gold_ai_trader.learning import maybe_run_learning_review, record_outcome_from_execution, get_setup_stats
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
_prev_session: str | None = None
_lock_conn = None


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

    if killzone_only_enabled() and not in_killzone(now, session, cfg):
        runtime_state.note_dormant("outside_killzone")
        await asyncio.sleep(max(cfg.scan_interval_s, 15))
        return

    db = SessionLocal()
    try:
        funnel_record("scan", db=db, session=session)
        cfg_row = db.query(GoldAiConfig).filter_by(id=1).first()
        if cfg_row:
            cfg = merge_config(cfg_row, env)

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
            runtime_state.set_funnel(funnel_snapshot())
            return

        await sync_pending_entries(db, cfg, float(price))

        k5 = await get_klines(SYMBOL, ASSET_CLASS, "5m", 60) or []
        atr = atr_from_klines(k5)

        candidate = None
        dedupe_reason = ""
        for cand in pick_top_candidates(candidates, max_candidates_per_scan()):
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

        decision, reasoning, usage = await decide(
            context,
            model=cfg.model,
            confidence_threshold=cfg.confidence_threshold,
        )
        record_claude_invocation(candidate)
        funnel_record("claude_called", setup=candidate.type, db=db, session=session)
        action = (decision.get("action") or "skip").lower()
        conf = int(decision.get("confidence") or 0)
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
            k1h = await get_gold_ai_klines("1h", 50, user_id=cfg.demo_user_id) or []
            k_daily = await get_gold_ai_klines("1d", 5, user_id=cfg.demo_user_id) or []
            key_levels = collect_key_levels(
                float(price), session, cfg, now, k_daily, k1h, k5,
            )
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
                ok_exec, exec_reason = check_can_execute(db, cfg, cfg.demo_user_id or 0)
                if ok_exec:
                    exec_id = await execute_take(
                        db=db,
                        cfg=cfg,
                        decision=decision,
                        decision_id=row.id,
                        session=session,
                        setup_type=candidate.type,
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
                        block_reason = "demo order rejected"
                else:
                    block_reason = exec_reason
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
    log_executor_lock_keepalive_config("gold-ai-trader:loop")
    logger.info("[gold-ai-trader] background loop starting (interval=%ss)", delay)

    while True:
        if _lock_conn is None:
            _lock_conn = await asyncio.to_thread(_acquire_gold_ai_lock)
            if _lock_conn is None:
                await asyncio.sleep(delay)
                continue
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
            await run_gold_ai_trader_loop()
        except Exception as e:
            logger.error("[gold-ai-trader] lock loop: %s", e)

        await asyncio.sleep(delay)


async def maybe_start_background_loop() -> None:
    global _loop_task
    if not gold_ai_trader_enabled():
        logger.info("[gold-ai-trader] disabled (GOLD_AI_TRADER_ENABLED=false)")
        return
    if _loop_task and not _loop_task.done():
        return
    _loop_task = asyncio.create_task(_locked_loop_forever())
    logger.info("[gold-ai-trader] background task scheduled")
