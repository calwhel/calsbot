"""Demo order execution for Gemini Gold Trader."""
from __future__ import annotations

import logging
import time
from datetime import datetime
from typing import Any, Dict, Optional, Tuple

from app.gemini_gold_trader.config import ASSET_CLASS, SYMBOL, GeminiGoldRuntimeConfig
from app.gemini_gold_trader.db_thread import db_commit, run_in_db_thread
from app.gemini_gold_trader.fire_validation import revalidate_before_fire
from app.gemini_gold_trader.guardrails import DemoAccountRequired, assert_demo_account

logger = logging.getLogger(__name__)

GEMINI_GOLD_STRATEGY_NAME = "Gemini Gold Trader (Demo)"


def _parse_direction(decision: Dict[str, Any]) -> Optional[str]:
    direction = (decision.get("direction") or "").upper()
    if direction in ("LONG", "SHORT"):
        return direction
    return None


def _parse_prices(decision: Dict[str, Any]) -> Optional[Tuple[str, float, float, float]]:
    direction = _parse_direction(decision)
    if not direction:
        return None
    try:
        entry = float(decision.get("entry") or 0)
        sl = float(decision.get("stop_loss") or 0)
        tp = float(decision.get("take_profit") or 0)
    except (TypeError, ValueError):
        return None
    if entry <= 0 or sl <= 0 or tp <= 0:
        return None
    return direction, entry, sl, tp


def ensure_system_strategy(db, user_id: int) -> int:
    from app.strategy_models import UserStrategy

    row = (
        db.query(UserStrategy)
        .filter(
            UserStrategy.user_id == user_id,
            UserStrategy.name == GEMINI_GOLD_STRATEGY_NAME,
        )
        .first()
    )
    if row:
        return row.id
    row = UserStrategy(
        user_id=user_id,
        name=GEMINI_GOLD_STRATEGY_NAME,
        description="Gemini Vision XAUUSD demo trader (isolated module)",
        config={
            "asset_class": ASSET_CLASS,
            "symbol": SYMBOL,
            "gemini_gold_trader": True,
        },
        is_active=True,
    )
    db.add(row)
    db.commit()
    db.refresh(row)
    return row.id


def _resolve_demo_trader(db, cfg: GeminiGoldRuntimeConfig):
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
        prefs,
        preferred_ctid=cfg.demo_ctrader_account_id,
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
    cfg: GeminiGoldRuntimeConfig,
    decision: Dict[str, Any],
    decision_id: int,
    spot_hint: float,
) -> Optional[int]:
    """Place demo market order; return StrategyExecution.id."""
    if not cfg.demo_user_id:
        logger.warning("[gemini-gold] GEMINI_GOLD_USER_ID not set")
        return None

    user, prefs, ctid = await run_in_db_thread(_resolve_demo_trader, db, cfg)
    if not user or not prefs or not ctid:
        logger.warning("[gemini-gold] demo trader resolution failed")
        return None

    fire_ok, fire_reason, decision = await revalidate_before_fire(
        decision=decision,
        cfg=cfg,
        user_id=cfg.demo_user_id,
        spot_hint=spot_hint,
    )
    if not fire_ok:
        logger.info("[gemini-gold] fire blocked: %s", fire_reason)
        return None

    parsed = _parse_prices(decision)
    if not parsed:
        logger.warning("[gemini-gold] invalid prices entry/sl/tp")
        return None
    direction, entry, sl, tp = parsed

    from app.services.ctrader_client import place_market_order_resilient
    from app.services.order_latency import new_order_latency
    from app.strategy_models import StrategyExecution

    token = prefs.ctrader_access_token
    latency = new_order_latency(decision_id, signal_mono=time.monotonic())
    latency.mark_queued()
    latency.mark_dequeue()
    result = await place_market_order_resilient(
        user_id=user.id,
        access_token=token,
        ctid=ctid,
        prefs=prefs,
        symbol_name=SYMBOL,
        direction=direction,
        volume_lots=max(0.01, float(cfg.demo_lot_size or 0.1)),
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
        logger.warning("[gemini-gold] order failed: %s", result)
        return None

    fill = float(result["actual_fill"])
    broker_units = result.get("volume")
    broker_units_i: Optional[int] = None
    try:
        if broker_units is not None:
            broker_units_i = int(broker_units)
    except Exception:
        broker_units_i = None

    strategy_id = await run_in_db_thread(ensure_system_strategy, db, user.id)
    note = f"gemini_gold_trader decision_id={decision_id}"
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
        "[gemini-gold] DEMO order placed exec=%s %s %s @ %s",
        ex.id,
        direction,
        SYMBOL,
        fill,
    )
    return ex.id
