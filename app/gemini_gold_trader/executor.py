"""Demo and live order execution for Gemini Gold Trader."""
from __future__ import annotations

import logging
import time
from datetime import datetime
from typing import Any, Dict, Optional, Tuple

from app.gemini_gold_trader.config import ASSET_CLASS, SYMBOL, GeminiGoldRuntimeConfig
from app.gemini_gold_trader.db_thread import db_commit, run_in_db_thread
from app.gemini_gold_trader.fire_validation import revalidate_before_fire
from app.gemini_gold_trader.guardrails import (
    DemoAccountRequired,
    LiveAccountRequired,
    active_ctrader_account_id,
    active_lot_size,
    assert_execution_account,
    assert_live_account,
    clear_execution_reservation,
    is_live_execution_mode,
)

logger = logging.getLogger(__name__)

GEMINI_GOLD_STRATEGY_NAME_DEMO = "Gemini Gold Trader (Demo)"
GEMINI_GOLD_STRATEGY_NAME_LIVE = "Gemini Gold Trader (Live)"
GEMINI_GOLD_STRATEGY_NAME_LIVE_MIRROR = "Gemini Gold Trader (Live Mirror)"


def _parse_direction(decision: Dict[str, Any]) -> Optional[str]:
    direction = (decision.get("direction") or "").upper()
    if direction in ("LONG", "SHORT"):
        return direction
    return None


def _parse_prices(decision: Dict[str, Any]) -> Optional[Tuple[str, float, float, float]]:
    from app.gemini_gold_trader.trade_invert import invert_take_decision

    d = invert_take_decision(decision)
    direction = _parse_direction(d)
    if not direction:
        return None
    try:
        entry = float(d.get("entry") or 0)
        sl = float(d.get("stop_loss") or 0)
        tp = float(d.get("take_profit") or 0)
    except (TypeError, ValueError):
        return None
    if entry <= 0 or sl <= 0 or tp <= 0:
        return None
    return direction, entry, sl, tp


def _pct_from_prices(direction: str, entry: float, sl: float, tp: float) -> Tuple[float, float]:
    if direction == "LONG":
        sl_pct = abs(entry - sl) / entry * 100.0
        tp_pct = abs(tp - entry) / entry * 100.0
    else:
        sl_pct = abs(sl - entry) / entry * 100.0
        tp_pct = abs(entry - tp) / entry * 100.0
    return max(sl_pct, 0.01), max(tp_pct, 0.01)


def ensure_system_strategy(
    db,
    user_id: int,
    *,
    live: bool = False,
    live_mirror: bool = False,
) -> int:
    from app.strategy_models import UserStrategy

    if live_mirror:
        name = GEMINI_GOLD_STRATEGY_NAME_LIVE_MIRROR
        description = "Gemini Vision XAUUSD live mirror (isolated module)"
        config_extra = {"gemini_gold_live_mirror": True}
    elif live:
        name = GEMINI_GOLD_STRATEGY_NAME_LIVE
        description = "Gemini Vision XAUUSD live trader (isolated module)"
        config_extra = {"gemini_gold_live": True}
    else:
        name = GEMINI_GOLD_STRATEGY_NAME_DEMO
        description = "Gemini Vision XAUUSD demo trader (isolated module)"
        config_extra = {"gemini_gold_live": False}
    row = (
        db.query(UserStrategy)
        .filter(
            UserStrategy.user_id == user_id,
            UserStrategy.name == name,
        )
        .first()
    )
    if row:
        return row.id
    row = UserStrategy(
        user_id=user_id,
        name=name,
        description=description,
        config={
            "asset_class": ASSET_CLASS,
            "symbol": SYMBOL,
            "gemini_gold_trader": True,
            **config_extra,
        },
        status="paused",
        asset_class=ASSET_CLASS,
        is_public=False,
    )
    db.add(row)
    db.commit()
    db.refresh(row)
    return row.id


def _resolve_trader(db, cfg: GeminiGoldRuntimeConfig):
    from app.models import User, UserPreference
    from app.services.ctrader_client import resolve_ctrader_ctid

    if not cfg.demo_user_id:
        return None, None, None
    user = db.query(User).filter(User.id == cfg.demo_user_id).first()
    if not user:
        return None, None, None
    prefs = db.query(UserPreference).filter(UserPreference.user_id == user.id).first()
    if not prefs or not prefs.ctrader_access_token:
        return None, None, None
    ctid_str = resolve_ctrader_ctid(
        execution_account_id=active_ctrader_account_id(cfg),
        prefs_default=prefs.ctrader_account_id,
    )
    if not ctid_str:
        return None, None, None
    try:
        assert_execution_account(prefs, int(ctid_str), cfg)
    except (DemoAccountRequired, LiveAccountRequired):
        return None, None, None
    return user, prefs, int(ctid_str)


async def execute_take(
    *,
    db,
    cfg: GeminiGoldRuntimeConfig,
    decision: Dict[str, Any],
    decision_id: int,
    spot_hint: float,
    session: str = "",
    order_ctx: Optional[Dict[str, Any]] = None,
    atr: float = 0.0,
) -> Optional[int]:
    """Route market vs limit entry based on setup_type."""
    from app.gemini_gold_trader.entry_routing import use_limit_entry_for_setup
    from app.gemini_gold_trader.pending_entry import (
        create_entry_watch_pending,
        try_place_broker_limit,
    )

    setup_type = str(decision.get("setup_type") or "")
    if not use_limit_entry_for_setup(setup_type, cfg):
        return await execute_take_market(
            db=db,
            cfg=cfg,
            decision=decision,
            decision_id=decision_id,
            spot_hint=spot_hint,
            order_ctx=order_ctx,
            atr=atr,
        )

    user, prefs, ctid = await run_in_db_thread(_resolve_trader, db, cfg)
    if not user or not prefs or not ctid:
        if order_ctx is not None:
            order_ctx["block_reason"] = "blocked: trader_resolution_failed"
        return None

    lots = active_lot_size(cfg)
    result, err = await try_place_broker_limit(
        user=user,
        prefs=prefs,
        ctid=ctid,
        cfg=cfg,
        decision=decision,
        decision_id=decision_id,
        volume_lots=lots,
    )
    if result and result.get("actual_fill"):
        return await execute_take_market(
            db=db,
            cfg=cfg,
            decision=decision,
            decision_id=decision_id,
            spot_hint=spot_hint,
            order_ctx=order_ctx,
            atr=atr,
        )

    pending_id = await create_entry_watch_pending(
        db,
        cfg=cfg,
        decision=decision,
        decision_id=decision_id,
        session=session,
    )
    if pending_id:
        logger.info(
            "[gemini-gold] limit entry-watch pending id=%s decision=%s err=%s",
            pending_id,
            decision_id,
            err,
        )
        return -int(pending_id)
    return await execute_take_market(
        db=db,
        cfg=cfg,
        decision=decision,
        decision_id=decision_id,
        spot_hint=spot_hint,
        order_ctx=order_ctx,
        atr=atr,
    )


async def execute_take_market(
    *,
    db,
    cfg: GeminiGoldRuntimeConfig,
    decision: Dict[str, Any],
    decision_id: int,
    spot_hint: float,
    order_ctx: Optional[Dict[str, Any]] = None,
    atr: float = 0.0,
) -> Optional[int]:
    """Place market order on configured demo or live account."""

    async def _fail_async(reason: str, *, clear_reserve: bool = True) -> None:
        if order_ctx is not None:
            order_ctx["block_reason"] = reason
        if clear_reserve:
            await run_in_db_thread(clear_execution_reservation, db, decision_id)

    if not cfg.demo_user_id:
        logger.warning("[gemini-gold] GEMINI_GOLD_USER_ID not set")
        await _fail_async("blocked: no_demo_user")
        return None

    user, prefs, ctid = await run_in_db_thread(_resolve_trader, db, cfg)
    if not user or not prefs or not ctid:
        logger.warning("[gemini-gold] trader resolution failed mode=%s", cfg.execution_mode)
        await _fail_async("blocked: trader_resolution_failed")
        return None

    fire_ok, fire_reason, decision = await revalidate_before_fire(
        decision=decision,
        cfg=cfg,
        user_id=cfg.demo_user_id,
        spot_hint=spot_hint,
        decision_id=decision_id,
        atr=atr,
    )
    if not fire_ok:
        logger.info("[gemini-gold] fire blocked: %s", fire_reason)
        await _fail_async(fire_reason)
        return None

    parsed = _parse_prices(decision)
    if not parsed:
        logger.warning("[gemini-gold] invalid prices entry/sl/tp")
        await _fail_async("blocked: invalid_entry_sl_tp")
        return None
    direction, entry, sl, tp = parsed

    from app.services.ctrader_client import place_market_order_resilient
    from app.services.order_latency import new_order_latency
    from app.strategy_models import StrategyExecution

    token = prefs.ctrader_access_token
    latency = new_order_latency(decision_id, signal_mono=time.monotonic())
    latency.mark_queued()
    latency.mark_dequeue()
    live_mode = is_live_execution_mode(cfg)
    result = await place_market_order_resilient(
        user_id=user.id,
        access_token=token,
        ctid=ctid,
        prefs=prefs,
        symbol_name=SYMBOL,
        direction=direction,
        volume_lots=active_lot_size(cfg),
        stop_loss_price=sl,
        take_profit_price=tp,
        entry_price=entry,
        label="GeminiGold",
        latency=latency,
        execution_id=decision_id,
    )
    try:
        latency.log_summary(outcome="fill" if result and result.get("actual_fill") else "fail")
    except Exception:
        pass
    if not result or not result.get("actual_fill"):
        broker_err = (result or {}).get("error")
        if order_ctx is not None:
            if broker_err:
                order_ctx["broker_error"] = str(broker_err)[:240]
                order_ctx["block_reason"] = str(broker_err)[:240]
            elif not order_ctx.get("block_reason"):
                order_ctx["block_reason"] = "order rejected"
        logger.warning(
            "[gemini-gold] order failed decision_id=%s mode=%s broker_error=%s result=%s",
            decision_id,
            cfg.execution_mode,
            broker_err,
            result,
        )
        await run_in_db_thread(clear_execution_reservation, db, decision_id)
        return None

    position_id = result.get("position_id")
    if not position_id or not str(position_id).strip():
        reason = "broker fill without position_id — not recording OPEN execution"
        if order_ctx is not None:
            order_ctx["block_reason"] = reason
        logger.warning(
            "[gemini-gold] order fill missing position_id decision_id=%s order_id=%s",
            decision_id,
            result.get("order_id"),
        )
        await run_in_db_thread(clear_execution_reservation, db, decision_id)
        return None

    fill = float(result["actual_fill"])
    broker_units = result.get("volume")
    broker_units_i: Optional[int] = None
    try:
        if broker_units is not None:
            broker_units_i = int(broker_units)
    except Exception:
        broker_units_i = None

    strategy_id = await run_in_db_thread(ensure_system_strategy, db, user.id, live=live_mode)
    note = f"gemini_gold_trader decision_id={decision_id}"
    if live_mode:
        note += " | live"
    order_id = result.get("order_id")
    position_id = result.get("position_id")
    if order_id:
        note += f" | ord={order_id}"
    if position_id:
        note += f" | pos={position_id}"
    if broker_units_i and broker_units_i > 0:
        note += f" | vol={broker_units_i}"

    ex = StrategyExecution(
        strategy_id=strategy_id,
        user_id=user.id,
        symbol=SYMBOL,
        direction=direction,
        entry_price=fill,
        tp_price=tp,
        sl_price=sl,
        current_sl=sl,
        outcome="OPEN",
        fired_at=datetime.utcnow(),
        is_paper=False,
        asset_class=ASSET_CLASS,
        ctrader_account_id=str(ctid),
        ctrader_order_id=str(result.get("order_id") or ""),
        ctrader_position_id=str(result.get("position_id") or ""),
        broker_volume_units=broker_units_i,
        remaining_volume=float(broker_units_i) if broker_units_i and broker_units_i > 0 else None,
        notes=note,
        conditions_met={"gemini_gold_decision_id": decision_id},
    )
    db.add(ex)
    await db_commit(db)
    await run_in_db_thread(db.refresh, ex)
    logger.info(
        "[gemini-gold] %s order placed exec=%s %s %s @ %s ctid=%s",
        "LIVE" if live_mode else "DEMO",
        ex.id,
        direction,
        SYMBOL,
        fill,
        ctid,
    )
    return ex.id


async def maybe_live_mirror_after_demo(
    *,
    db,
    cfg: GeminiGoldRuntimeConfig,
    decision: Dict[str, Any],
    decision_id: int,
    demo_execution_id: int,
    session: str = "",
) -> None:
    """Copy a successful demo TAKE to the configured live account when enabled."""
    from app.gemini_gold_trader.guardrails import (
        check_can_execute_live_mirror,
        is_live_execution_mode,
    )
    from app.gemini_gold_trader.models import GeminiGoldDecision

    if is_live_execution_mode(cfg) or not cfg.live_mirror_enabled:
        return

    def _update(*, live_exec_id: Optional[int], status: str, error: Optional[str] = None):
        row = db.query(GeminiGoldDecision).filter_by(id=int(decision_id)).first()
        if not row:
            return
        if live_exec_id:
            row.live_mirror_execution_id = live_exec_id
        row.live_mirror_status = status
        if error is not None:
            row.live_mirror_error = error

    ok_live, live_reason = check_can_execute_live_mirror(db, cfg, cfg.demo_user_id or 0)
    if not ok_live:
        _update(live_exec_id=None, status="skipped", error=live_reason)
        await db_commit(db)
        return

    live_exec_id = await execute_live_mirror_take(
        db=db,
        cfg=cfg,
        decision=decision,
        decision_id=decision_id,
        demo_execution_id=demo_execution_id,
    )
    if live_exec_id:
        _update(live_exec_id=live_exec_id, status="filled")
        await db_commit(db)
        try:
            from app.gemini_gold_trader.telegram_notify import notify_live_mirror_filled

            await notify_live_mirror_filled(
                session=str(session or decision.get("session") or ""),
                decision=decision,
                confidence=int(decision.get("confidence") or 0),
                decision_id=decision_id,
                live_execution_id=live_exec_id,
                demo_execution_id=demo_execution_id,
            )
        except Exception as exc:
            logger.warning(
                "[gemini-gold] live mirror notify failed decision=%s: %s",
                decision_id,
                exc,
            )
    else:
        _update(
            live_exec_id=None,
            status="failed",
            error="live mirror order rejected",
        )
        await db_commit(db)


def _resolve_live_mirror_trader(db, cfg: GeminiGoldRuntimeConfig):
    from app.models import User, UserPreference

    if not cfg.demo_user_id:
        return None, None
    user = db.query(User).filter(User.id == cfg.demo_user_id).first()
    if not user:
        return None, None
    prefs = db.query(UserPreference).filter(UserPreference.user_id == user.id).first()
    return user, prefs


async def execute_live_mirror_take(
    *,
    db,
    cfg: GeminiGoldRuntimeConfig,
    decision: Dict[str, Any],
    decision_id: int,
    demo_execution_id: int,
) -> Optional[int]:
    """Place a live market order mirroring a successful demo TAKE.

    Places directly via place_market_order_resilient (same path as demo).
    The shared CtraderOrderJob queue is gated on the main executor advisory
    lock, which the standalone gemini_gold_runner never holds — so queue-based
    live mirror always failed with "executor lock unconfirmed".
    """
    if not cfg.live_mirror_enabled or not cfg.demo_user_id:
        return None
    if is_live_execution_mode(cfg):
        return None

    from app.services.ctrader_client import place_market_order_resilient
    from app.services.order_latency import new_order_latency
    from app.strategy_models import StrategyExecution

    user, prefs = await run_in_db_thread(_resolve_live_mirror_trader, db, cfg)
    if not user:
        return None
    if not prefs or not prefs.ctrader_access_token:
        logger.warning("[gemini-gold] live mirror: missing cTrader token")
        return None

    if not cfg.live_ctrader_account_id:
        logger.warning("[gemini-gold] live mirror: no live ctid configured")
        return None
    ctid = int(cfg.live_ctrader_account_id)
    try:
        assert_live_account(prefs, ctid, cfg)
    except LiveAccountRequired as e:
        logger.error("[gemini-gold] LIVE MIRROR LOCK: %s", e)
        return None

    parsed = _parse_prices(decision)
    if not parsed:
        return None
    direction, entry, sl, tp = parsed

    # Prefer the demo fill as the signal price so relative SL/TP match what
    # just filled on demo (avoids entry drift vs chart mid).
    def _load_demo_ex(db_sess):
        return (
            db_sess.query(StrategyExecution)
            .filter(StrategyExecution.id == int(demo_execution_id))
            .first()
        )

    demo_ex = await run_in_db_thread(_load_demo_ex, db)
    if demo_ex and demo_ex.entry_price:
        try:
            entry = float(demo_ex.entry_price)
        except (TypeError, ValueError):
            pass

    lots = max(0.01, float(cfg.live_lot_size or 0.01))
    strategy_id = await run_in_db_thread(
        ensure_system_strategy, db, user.id, live_mirror=True
    )

    latency = new_order_latency(decision_id, signal_mono=time.monotonic())
    latency.mark_queued()
    latency.mark_dequeue()
    result = await place_market_order_resilient(
        user_id=user.id,
        access_token=prefs.ctrader_access_token,
        ctid=ctid,
        prefs=prefs,
        symbol_name=SYMBOL,
        direction=direction,
        volume_lots=lots,
        stop_loss_price=sl,
        take_profit_price=tp,
        entry_price=entry,
        label="GeminiGoldLiveMirror",
        latency=latency,
        execution_id=decision_id,
    )
    try:
        latency.log_summary(outcome="fill" if result and result.get("actual_fill") else "fail")
    except Exception:
        pass

    if not result or not result.get("actual_fill"):
        broker_err = (result or {}).get("error")
        logger.warning(
            "[gemini-gold] live mirror order failed decision=%s demo_exec=%s err=%s",
            decision_id,
            demo_execution_id,
            broker_err,
        )
        return None

    position_id = result.get("position_id")
    if not position_id or not str(position_id).strip():
        logger.warning(
            "[gemini-gold] live mirror fill missing position_id decision=%s order_id=%s",
            decision_id,
            result.get("order_id"),
        )
        return None

    fill = float(result["actual_fill"])
    broker_units = result.get("volume")
    broker_units_i: Optional[int] = None
    try:
        if broker_units is not None:
            broker_units_i = int(broker_units)
    except Exception:
        broker_units_i = None

    note = (
        f"gemini_gold_trader_live_mirror decision_id={decision_id} "
        f"demo_exec={demo_execution_id}"
    )
    order_id = result.get("order_id")
    if order_id:
        note += f" | ord={order_id}"
    note += f" | pos={position_id}"
    if broker_units_i and broker_units_i > 0:
        note += f" | vol={broker_units_i}"

    ex = StrategyExecution(
        strategy_id=strategy_id,
        user_id=user.id,
        symbol=SYMBOL,
        direction=direction,
        entry_price=fill,
        tp_price=tp,
        sl_price=sl,
        current_sl=sl,
        outcome="OPEN",
        fired_at=datetime.utcnow(),
        is_paper=False,
        asset_class=ASSET_CLASS,
        ctrader_account_id=str(ctid),
        ctrader_order_id=str(order_id or ""),
        ctrader_position_id=str(position_id),
        broker_volume_units=broker_units_i,
        remaining_volume=float(broker_units_i) if broker_units_i and broker_units_i > 0 else None,
        notes=note,
        conditions_met={
            "gemini_gold_decision_id": decision_id,
            "gemini_gold_live_mirror": True,
            "demo_execution_id": demo_execution_id,
        },
    )
    db.add(ex)
    await db_commit(db)
    await run_in_db_thread(db.refresh, ex)
    logger.info(
        "[gemini-gold] LIVE MIRROR filled exec=%s ctid=%s %s %s @ %s lots=%s decision=%s",
        ex.id,
        ctid,
        direction,
        SYMBOL,
        fill,
        lots,
        decision_id,
    )
    return ex.id
