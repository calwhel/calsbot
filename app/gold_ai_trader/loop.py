"""Isolated background loop with advisory lock."""
from __future__ import annotations

import asyncio
import logging
import os
from datetime import datetime

import httpx

from app.database import SessionLocal
from app.gold_ai_trader.config import env_defaults, gold_ai_trader_enabled
from app.gold_ai_trader.schema import ensure_gold_ai_trader_schema, seed_config_if_missing
from app.gold_ai_trader.guardrails import (
    merge_config,
    check_can_call_claude,
    check_can_execute,
    check_can_execute_live_mirror,
)
from app.gold_ai_trader.scanner import (
    active_session,
    scan_candidates,
    pick_best,
    record_claude_invocation,
    _setup_cooldown_s,
)
from app.gold_ai_trader.call_gates import (
    atr_from_klines,
    should_invoke_claude,
    call_stats_today,
)
from app.services.tradfi_prices import get_klines
from app.gold_ai_trader.config import SYMBOL, ASSET_CLASS
from app.gold_ai_trader.context import build_context_snapshot
from app.gold_ai_trader.funnel import (
    funnel_stats_today,
    persist_false_reject,
    run_funnel_pipeline,
)
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
_loop_task: asyncio.Task | None = None
_prev_session: str | None = None


async def run_gold_ai_trader_loop() -> None:
    """Main scan → candidate → Claude → optional execute cycle."""
    global _prev_session
    ensure_gold_ai_trader_schema()
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

    db = SessionLocal()
    try:
        cfg_row = db.query(GoldAiConfig).filter_by(id=1).first()
        if cfg_row:
            cfg = merge_config(cfg_row, env)

        ok_call, reason = check_can_call_claude(db, cfg)
        if not ok_call:
            runtime_state.note_dormant(reason)
            if reason == "max_calls_day":
                await maybe_notify_call_cap_reached(db, cfg)
            return

        async with httpx.AsyncClient(timeout=15) as http:
            price, candidates = await scan_candidates(http, session=session, cfg=cfg)
        if not candidates or price is None:
            return

        await sync_pending_entries(db, cfg, float(price))

        candidate = pick_best(candidates)
        if not candidate:
            return

        k5 = await get_klines(SYMBOL, ASSET_CLASS, "5m", 60) or []
        atr = atr_from_klines(k5)
        rvol = 1.0
        vols = [float(r[5]) for r in k5 if r and len(r) >= 6]
        if len(vols) >= 20:
            avg = sum(vols[-21:-1]) / 20
            rvol = (vols[-1] / avg) if avg else 1.0
        ok_dedupe, dedupe_reason = should_invoke_claude(
            db, candidate, float(price), atr, setup_cooldown_s=_setup_cooldown_s()
        )
        if not ok_dedupe:
            logger.debug("[gold-ai-trader] claude dedupe skip: %s", dedupe_reason)
            runtime_state.note_dormant(dedupe_reason)
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
        )

        funnel = await run_funnel_pipeline(
            candidate=candidate,
            price=float(price),
            atr=float(atr),
            rvol=float(rvol),
            session=session,
            full_context=context,
            cfg=cfg,
            db=db,
        )
        decision = funnel.decision
        reasoning = funnel.reasoning
        usage = funnel.opus_usage
        record_claude_invocation(candidate)
        action = (decision.get("action") or "skip").lower()
        conf = int(decision.get("confidence") or 0)

        screen_cost = float(funnel.screen_usage.get("cost_usd", 0) or 0)
        opus_cost = float(usage.get("cost_usd", 0) or 0)
        total_cost = round(screen_cost + opus_cost, 6)

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
            cost_usd=opus_cost,
            screen_action=funnel.screen_action,
            screen_reason=funnel.screen_reason,
            screen_model=getattr(cfg, "screen_model", "claude-haiku-4-5"),
            screen_tokens_in=int(funnel.screen_usage.get("tokens_in", 0)),
            screen_tokens_out=int(funnel.screen_usage.get("tokens_out", 0)),
            screen_cache_read_tokens=int(funnel.screen_usage.get("cache_read_tokens", 0)),
            screen_cache_write_tokens=int(funnel.screen_usage.get("cache_write_tokens", 0)),
            screen_cost_usd=screen_cost,
            funnel_mode=funnel.funnel_mode,
            opus_called=bool(funnel.opus_called),
            total_cost_usd=total_cost,
        )
        db.add(row)
        db.commit()
        db.refresh(row)

        if funnel.false_reject:
            persist_false_reject(
                db,
                decision_id=row.id,
                session=session,
                candidate_type=candidate.type,
                screen_reason=funnel.screen_reason,
                opus_action=action,
                opus_confidence=conf,
                opus_rationale=str(decision.get("rationale", "") or ""),
            )

        funnel_stats_today(db)

        runtime_state.note_decision(
            {
                "id": row.id,
                "action": action,
                "confidence": conf,
                "rationale": decision.get("rationale", ""),
            }
        )

        if action == "take":
            executed = False
            execution_id = None
            block_reason = None
            if conf >= 70:
                ok_exec, exec_reason = check_can_execute(db, cfg, cfg.demo_user_id or 0)
                if ok_exec:
                    exec_id = await execute_take(
                        db=db, cfg=cfg, decision=decision, decision_id=row.id, session=session
                    )
                    if exec_id and exec_id > 0:
                        executed = True
                        execution_id = exec_id
                        row.executed = True
                        row.execution_id = exec_id
                        db.commit()

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
                    else:
                        block_reason = "demo order rejected"
                else:
                    block_reason = exec_reason
                    logger.info("[gold-ai-trader] execute blocked: %s", exec_reason)
            else:
                block_reason = f"confidence {conf}% below 70% threshold"

            await notify_take_decision(
                candidate_type=candidate.type,
                session=session,
                decision=decision,
                confidence=conf,
                executed=executed,
                execution_id=execution_id,
                block_reason=block_reason,
            )

        # Sync outcomes for closed trades + Telegram close alerts
        if row.execution_id:
            from app.strategy_models import StrategyExecution

            ex = db.query(StrategyExecution).filter_by(id=row.execution_id).first()
            record_outcome_from_execution(db, row.id, ex)

        await sync_closed_trade_notifications(db, cfg)
        await maybe_send_daily_summary(db, cfg)
        await maybe_run_learning_review(db, session, cfg)
    except Exception as e:
        logger.error("[gold-ai-trader] loop error: %s", e, exc_info=True)
        runtime_state.note_error(str(e))
    finally:
        db.close()


async def _locked_loop_forever() -> None:
    """Advisory lock so only one worker runs the trader."""
    from app.database import bg_engine
    from sqlalchemy import text

    delay = max(15.0, float(os.environ.get("GOLD_AI_TRADER_SCAN_INTERVAL_S", "20")))
    logger.info("[gold-ai-trader] background loop starting (interval=%ss)", delay)
    while True:
        conn = None
        try:
            conn = bg_engine.raw_connection()
            cur = conn.cursor()
            cur.execute("SELECT pg_try_advisory_lock(%s)", (_LOCK_ID,))
            got = cur.fetchone()[0]
            cur.close()
            if not got:
                await asyncio.sleep(delay)
                continue
            try:
                await run_gold_ai_trader_loop()
            finally:
                cur = conn.cursor()
                cur.execute("SELECT pg_advisory_unlock(%s)", (_LOCK_ID,))
                cur.close()
                conn.commit()
        except Exception as e:
            logger.error("[gold-ai-trader] lock loop: %s", e)
        finally:
            if conn:
                conn.close()
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
