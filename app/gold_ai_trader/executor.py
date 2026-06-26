"""Demo execution via existing cTrader client + StrategyExecution row."""
from __future__ import annotations

import logging
import time
from datetime import datetime
from typing import Any, Dict, Optional, Tuple

from app.gold_ai_trader.config import GoldAiRuntimeConfig, SYMBOL, ASSET_CLASS
from app.gold_ai_trader.guardrails import (
    assert_demo_account,
    assert_live_account,
    DemoAccountRequired,
    LiveAccountRequired,
)

logger = logging.getLogger(__name__)

GOLD_AI_STRATEGY_NAME = "Gold AI Trader (Demo)"
GOLD_AI_LIVE_STRATEGY_NAME = "Gold AI Trader (Live Mirror)"


def _parse_direction(decision: Dict[str, Any]) -> Optional[str]:
    direction = (decision.get("direction") or "").upper()
    if direction not in ("LONG", "SHORT"):
        direction = "LONG" if direction == "long" else "SHORT" if direction == "short" else None
    return direction


def _parse_prices(decision: Dict[str, Any]) -> Optional[Tuple[str, float, float, float]]:
    direction = _parse_direction(decision)
    if not direction:
        return None
    entry = float(decision.get("entry") or 0)
    sl = float(decision.get("stop_loss") or 0)
    tp = float(decision.get("take_profit") or 0)
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


def ensure_system_strategy(db, user_id: int, *, live_mirror: bool = False) -> int:
    from app.strategy_models import UserStrategy

    name = GOLD_AI_LIVE_STRATEGY_NAME if live_mirror else GOLD_AI_STRATEGY_NAME
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
        description=(
            "Autonomous Claude-driven XAUUSD live mirror (isolated module)"
            if live_mirror
            else "Autonomous Claude-driven XAUUSD demo trader (isolated module)"
        ),
        config={
            "asset_class": "forex",
            "symbol": SYMBOL,
            "gold_ai_trader": True,
            "gold_ai_live_mirror": live_mirror,
        },
        status="paused",
        asset_class="forex",
        is_public=False,
    )
    db.add(row)
    db.commit()
    db.refresh(row)
    return row.id


async def execute_take(
    *,
    db,
    cfg: GoldAiRuntimeConfig,
    decision: Dict[str, Any],
    decision_id: int,
    session: str = "",
    setup_type: str = "",
    timing_ctx: Optional[Dict[str, Any]] = None,
) -> Optional[int]:
    """Route TAKE to limit/entry-watch pending or immediate market."""
    from app.gold_ai_trader.entry_routing import use_limit_entry_for_setup

    use_limit = use_limit_entry_for_setup(setup_type, cfg)
    if not use_limit:
        return await execute_take_market(
            db=db, cfg=cfg, decision=decision, decision_id=decision_id,
            entry_note="use_limit_entry=false; market entry",
            timing_ctx=timing_ctx,
        )

    user, prefs, ctid = _resolve_demo_trader(db, cfg)
    if not user or not prefs or not ctid:
        return None

    from app.gold_ai_trader.pending_entry import (
        broker_limit_supported,
        create_entry_watch_pending,
        try_place_broker_limit,
    )

    if broker_limit_supported():
        result, err = await try_place_broker_limit(
            user=user,
            prefs=prefs,
            ctid=ctid,
            cfg=cfg,
            decision=decision,
            decision_id=decision_id,
        )
        if result and result.get("order_id"):
            from app.gold_ai_trader.models import GoldAiPendingOrder
            from app.gold_ai_trader.pending_entry import compute_pending_expiry

            parsed = _parse_prices(decision)
            if not parsed:
                return None
            direction, entry, sl, tp = parsed
            now = datetime.utcnow()
            row = GoldAiPendingOrder(
                decision_id=decision_id,
                session=session,
                direction=direction,
                entry_price=entry,
                stop_loss=sl,
                take_profit=tp,
                status="pending",
                method="broker_limit",
                broker_order_id=str(result.get("order_id")),
                created_at=now,
                expires_at=compute_pending_expiry(
                    now, session, cfg, getattr(cfg, "pending_entry_timeout_min", 30)
                ),
                notes="broker LIMIT placed",
            )
            db.add(row)
            db.commit()
            logger.info(
                "[gold-ai-trader] broker LIMIT pending id=%s order=%s entry=%s",
                row.id,
                result.get("order_id"),
                entry,
            )
            return -row.id  # negative signals pending (not filled exec yet)

    pending_id = await create_entry_watch_pending(
        db,
        cfg=cfg,
        decision=decision,
        decision_id=decision_id,
        session=session,
    )
    return -pending_id if pending_id else await execute_take_market(
        db=db,
        cfg=cfg,
        decision=decision,
        decision_id=decision_id,
        entry_note="pending unsupported, used market",
        timing_ctx=timing_ctx,
    )


def _resolve_demo_trader(db, cfg: GoldAiRuntimeConfig):
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
        execution_account_id=cfg.demo_ctrader_account_id,
        prefs_default=prefs.ctrader_account_id,
    )
    if not ctid_str:
        return None, None, None
    try:
        assert_demo_account(prefs, int(ctid_str), cfg)
    except DemoAccountRequired:
        return None, None, None
    return user, prefs, int(ctid_str)


async def execute_take_market(
    *,
    db,
    cfg: GoldAiRuntimeConfig,
    decision: Dict[str, Any],
    decision_id: int,
    entry_note: str = "",
    timing_ctx: Optional[Dict[str, Any]] = None,
) -> Optional[int]:
    """Place demo market order; return StrategyExecution.id."""
    if not cfg.demo_user_id:
        logger.warning("[gold-ai-trader] GOLD_AI_TRADER_USER_ID not set")
        return None

    user, prefs, ctid = _resolve_demo_trader(db, cfg)
    if not user or not prefs or not ctid:
        if not user:
            logger.warning("[gold-ai-trader] demo user missing")
        elif not prefs or not prefs.ctrader_access_token:
            logger.warning("[gold-ai-trader] demo user missing cTrader token")
        else:
            logger.warning("[gold-ai-trader] no demo ctid configured")
        return None

    from app.strategy_models import StrategyExecution
    from app.services.ctrader_client import place_market_order_resilient
    from app.services.order_latency import new_order_latency

    direction = _parse_direction(decision)
    if not direction:
        return None

    parsed = _parse_prices(decision)
    if not parsed:
        logger.warning("[gold-ai-trader] invalid prices entry/sl/tp")
        return None
    direction, entry, sl, tp = parsed

    token = prefs.ctrader_access_token
    latency = new_order_latency(
        decision_id,
        signal_mono=time.monotonic(),
    )
    latency.mark_queued()
    latency.mark_dequeue()
    result = await place_market_order_resilient(
        user_id=user.id,
        access_token=token,
        ctid=ctid,
        prefs=prefs,
        symbol_name=SYMBOL,
        direction=direction,
        volume_lots=max(0.01, float(cfg.demo_lot_size or cfg.min_lot or 0.01)),
        stop_loss_price=sl,
        take_profit_price=tp,
        entry_price=entry,
        label="GoldAITrader",
        latency=latency,
        execution_id=decision_id,
    )
    if timing_ctx is not None:
        timing_ctx["broker_ack_ts"] = datetime.utcnow().replace(microsecond=0).isoformat() + "Z"
    try:
        latency.log_summary(
            outcome="fill" if result and result.get("actual_fill") else "fail"
        )
    except Exception:
        pass
    if not result or not result.get("actual_fill"):
        logger.warning("[gold-ai-trader] order failed: %s", result)
        return None

    fill = float(result["actual_fill"])
    broker_units = result.get("volume")
    broker_units_i: Optional[int] = None
    try:
        if broker_units is not None:
            broker_units_i = int(broker_units)
    except Exception:
        broker_units_i = None
    strategy_id = ensure_system_strategy(db, user.id)
    note = f"gold_ai_trader decision_id={decision_id}"
    order_id = result.get("order_id")
    position_id = result.get("position_id")
    if entry_note:
        note += f" | {entry_note}"
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
        conditions_met={"gold_ai_decision_id": decision_id},
    )
    db.add(ex)
    db.commit()
    db.refresh(ex)
    logger.info(
        "[gold-ai-trader] DEMO order placed exec=%s %s %s @ %s",
        ex.id,
        direction,
        SYMBOL,
        fill,
    )
    return ex.id


async def execute_live_mirror_take(
    *,
    db,
    cfg: GoldAiRuntimeConfig,
    decision: Dict[str, Any],
    decision_id: int,
    demo_execution_id: int,
) -> Optional[int]:
    """Queue live mirror order via the standard live forex order path."""
    if not cfg.live_mirror_enabled or not cfg.demo_user_id:
        return None

    from app.models import User, UserPreference
    from app.strategy_models import StrategyExecution
    from app.services.ctrader_order_queue import CtraderOrderJob, enqueue_ctrader_order

    user = db.query(User).filter(User.id == cfg.demo_user_id).first()
    if not user:
        return None
    prefs = db.query(UserPreference).filter(UserPreference.user_id == user.id).first()
    if not prefs or not prefs.ctrader_access_token:
        logger.warning("[gold-ai-trader] live mirror: missing cTrader token")
        return None

    if not cfg.live_ctrader_account_id:
        logger.warning("[gold-ai-trader] live mirror: no live ctid configured")
        return None
    ctid = int(cfg.live_ctrader_account_id)
    try:
        assert_live_account(prefs, ctid, cfg)
    except LiveAccountRequired as e:
        logger.error("[gold-ai-trader] LIVE LOCK: %s", e)
        return None

    parsed = _parse_prices(decision)
    if not parsed:
        return None
    direction, entry, sl, tp = parsed
    sl_pct, tp_pct = _pct_from_prices(direction, entry, sl, tp)
    lots = max(0.01, float(cfg.live_lot_size or 0.01))

    strategy_id = ensure_system_strategy(db, user.id, live_mirror=True)
    ex = StrategyExecution(
        strategy_id=strategy_id,
        user_id=user.id,
        symbol=SYMBOL,
        direction=direction,
        entry_price=entry,
        tp_price=tp,
        sl_price=sl,
        current_sl=sl,
        outcome="OPEN",
        fired_at=datetime.utcnow(),
        is_paper=False,
        asset_class=ASSET_CLASS,
        ctrader_account_id=str(ctid),
        notes=f"gold_ai_trader_live_mirror decision_id={decision_id} demo_exec={demo_execution_id}",
        conditions_met={
            "gold_ai_decision_id": decision_id,
            "gold_ai_live_mirror": True,
            "demo_execution_id": demo_execution_id,
        },
    )
    db.add(ex)
    db.commit()
    db.refresh(ex)

    signal_mono = time.monotonic()
    job = CtraderOrderJob(
        user_id=user.id,
        strategy_id=strategy_id,
        execution_id=ex.id,
        symbol=SYMBOL,
        direction=direction,
        entry_price=entry,
        tp_pct=tp_pct,
        sl_pct=sl_pct,
        fixed_lots=lots,
        asset_class=ASSET_CLASS,
        ctrader_account_id=str(ctid),
        signal_mono=signal_mono,
        signal_generated_at=time.time(),
        signal_price_source="gold_ai_trader",
    )
    ok = await enqueue_ctrader_order(job)
    if not ok:
        ex.notes = (ex.notes or "") + " | enqueue_failed"
        db.commit()
        logger.warning("[gold-ai-trader] live mirror enqueue failed exec=%s", ex.id)
        return ex.id

    logger.info(
        "[gold-ai-trader] LIVE MIRROR queued exec=%s ctid=%s %s %s lots=%s decision=%s",
        ex.id,
        ctid,
        direction,
        SYMBOL,
        lots,
        decision_id,
    )
    return ex.id


async def flatten_open_demo_positions(db, cfg: GoldAiRuntimeConfig) -> int:
    """Mark no-overnight intent — open positions rely on existing SL/TP + forex reconcile."""
    if not cfg.demo_user_id or not cfg.no_overnight:
        return 0
    from app.strategy_models import StrategyExecution

    count = (
        db.query(StrategyExecution)
        .filter(
            StrategyExecution.user_id == cfg.demo_user_id,
            StrategyExecution.symbol == SYMBOL,
            StrategyExecution.outcome == "OPEN",
            StrategyExecution.notes.like("%gold_ai_trader%"),
            ~StrategyExecution.notes.like("%live_mirror%"),
        )
        .count()
    )
    if count:
        logger.info(
            "[gold-ai-trader] NY window ended — %s open demo position(s); "
            "managed by existing forex SL/TP/reconcile",
            count,
        )
    return count
