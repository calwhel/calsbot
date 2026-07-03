"""ORB parallel path for Gemini Gold."""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, Optional

from app.gemini_gold_trader import state as runtime_state
from app.gemini_gold_trader.block_reason import format_block_reason
from app.gemini_gold_trader.db_thread import run_with_db
from app.gemini_gold_trader.fire_validation import refresh_spot_after_gemini, stale_entry_recheck
from app.gemini_gold_trader.funnel import record as funnel_record
from app.gemini_gold_trader.gemini import decide_orb_text
from app.gemini_gold_trader.guardrails import check_can_call_orb, check_can_execute, try_reserve_execution
from app.gemini_gold_trader.orb import (
    build_orb_context,
    detect_orb_signal,
    suggested_orb_levels,
)
from app.gemini_gold_trader.validator import validate_take_decision
from app.gold_ai_trader.call_gates import atr_from_klines

logger = logging.getLogger(__name__)


async def maybe_run_orb_strategy(
    *,
    cfg,
    session: str,
    now: datetime,
    price: float,
    run_with_db_fn,
    call_with_db_session,
    persist_decision_fn,
    execute_take_fn,
    notify_fn,
    mark_executed_fn,
    live_mirror_fn,
) -> bool:
    """Run ORB detector + text Gemini confirm. Returns True if ORB path handled the cycle."""
    if not cfg.orb_enabled:
        return False

    try:
        signal, orb_state, detect_reason = await detect_orb_signal(
            cfg=cfg,
            session=session,
            now=now,
            user_id=cfg.demo_user_id,
        )
    except Exception as exc:
        logger.warning("[gemini-gold-orb] detector error: %s", exc, exc_info=True)
        return False

    if not signal:
        return False

    funnel_record("orb_detected", setup=signal.setup_type, session=session)

    ok_orb, orb_reason = await run_with_db_fn(check_can_call_orb, cfg)
    if not ok_orb:
        logger.info("[gemini-gold-orb] blocked: %s", orb_reason)
        return False

    from app.gemini_gold_trader.orb import _get_orb_klines

    recent_bars = await _get_orb_klines(
        getattr(cfg, "orb_timeframe", "5m") if hasattr(cfg, "orb_timeframe") else "5m",
        64,
        user_id=cfg.demo_user_id,
    )
    context = build_orb_context(signal, session=session, cfg=cfg, now=now, recent_bars=recent_bars)

    decision, tokens_in, tokens_out, cost_usd, api_error = await decide_orb_text(
        context,
        cfg=cfg,
        confidence_threshold=int(cfg.orb_confidence_threshold),
    )
    if api_error or not decision:
        logger.warning("[gemini-gold-orb] gemini failed: %s", api_error)
        return False

    funnel_record("gemini_called", setup=signal.setup_type, session=session)
    price = await refresh_spot_after_gemini(float(price), user_id=cfg.demo_user_id)

    entry, sl, tp = suggested_orb_levels(signal, cfg)
    direction = "LONG" if signal.side == "long" else "SHORT"
    decision.setdefault("action", "TAKE")
    decision["setup_type"] = signal.setup_type
    decision["direction"] = direction
    decision["entry"] = decision.get("entry") or entry
    decision["stop_loss"] = decision.get("stop_loss") or sl
    decision["take_profit"] = decision.get("take_profit") or tp
    decision["orb_break_level"] = signal.break_level
    decision["orb_range_height"] = signal.range_height
    decision["validator_profile"] = "orb"

    action = str(decision.get("action") or "SKIP").upper()
    confidence = int(decision.get("confidence") or 0)

    row = await run_with_db_fn(
        persist_decision_fn,
        session=session,
        decision=decision,
        action=action,
        confidence=confidence,
        rationale=str(decision.get("rationale") or ""),
        chart_meta={"orb": True, "detect_reason": detect_reason},
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        cost_usd=cost_usd,
        dry_run=cfg.dry_run,
        setup_type=signal.setup_type,
    )

    if action != "TAKE" or confidence < cfg.orb_confidence_threshold:
        funnel_record("gemini_skip", setup=signal.setup_type, session=session)
        await notify_fn(session=session, decision=decision, action=action, confidence=confidence, dry_run=cfg.dry_run)
        return True

    funnel_record("gemini_take", setup=signal.setup_type, session=session, decision_id=row.id)
    atr = float(atr_from_klines(recent_bars) or 0.0)
    ok, val_reason, decision = validate_take_decision(decision, cfg=cfg, spot=price, atr=atr)
    if not ok:
        funnel_record("validator_rejected", setup=signal.setup_type, reason=val_reason, session=session, decision_id=row.id)
        await notify_fn(session=session, decision=decision, action=action, confidence=confidence, block_reason=val_reason, dry_run=cfg.dry_run)
        return True

    stale_ok, stale_reason = await stale_entry_recheck(
        decision=decision,
        cfg=cfg,
        decision_ts=getattr(row, "ts", None),
        decision_id=row.id,
        setup_type=signal.setup_type,
    )
    if not stale_ok:
        funnel_record("stale_entry_blocked", setup=signal.setup_type, reason=stale_reason, session=session, decision_id=row.id)
        await notify_fn(session=session, decision=decision, action=action, confidence=confidence, block_reason=stale_reason, dry_run=cfg.dry_run)
        return True

    can_exec, exec_reason = await run_with_db_fn(try_reserve_execution, cfg, cfg.demo_user_id or 0, row.id)
    if not can_exec:
        await notify_fn(session=session, decision=decision, action=action, confidence=confidence, block_reason=format_block_reason(exec_reason), dry_run=cfg.dry_run)
        return True

    from app.database import SessionLocal

    order_ctx: Dict[str, Any] = {}
    db = SessionLocal()
    try:
        exec_id = await execute_take_fn(
            db=db,
            cfg=cfg,
            decision=decision,
            decision_id=row.id,
            spot_hint=price,
            session=session,
            order_ctx=order_ctx,
            atr=atr,
        )
    finally:
        db.close()

    executed = False
    execution_id = None
    block_reason = None
    if exec_id and exec_id > 0:
        executed = True
        execution_id = exec_id
        funnel_record("executed", setup=signal.setup_type, session=session, decision_id=row.id)
        funnel_record("orb_executed", setup=signal.setup_type, session=session, decision_id=row.id)
        await mark_executed_fn(row.id, exec_id)
        await live_mirror_fn(cfg, decision, row.id, exec_id)
        if orb_state is not None:
            orb_state.trades_taken = int(getattr(orb_state, "trades_taken", 0) or 0) + 1
            orb_state.decision_id = row.id
            orb_state.execution_id = exec_id
    elif exec_id and exec_id < 0:
        funnel_record("pending_entry", setup=signal.setup_type, session=session, decision_id=row.id)
        block_reason = f"pending entry watch #{-exec_id}"
    else:
        block_reason = format_block_reason(order_ctx.get("block_reason") or "orb order rejected")

    await notify_fn(
        session=session,
        decision=decision,
        action=action,
        confidence=confidence,
        executed=executed,
        execution_id=execution_id,
        block_reason=block_reason,
        dry_run=cfg.dry_run,
    )
    runtime_state.note_decision(decision)
    return True
