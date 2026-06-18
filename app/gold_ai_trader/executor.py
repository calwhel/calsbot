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
) -> Optional[int]:
    """Place demo market order; return StrategyExecution.id."""
    if not cfg.demo_user_id:
        logger.warning("[gold-ai-trader] GOLD_AI_TRADER_USER_ID not set")
        return None

    from app.models import User, UserPreference
    from app.strategy_models import StrategyExecution
    from app.services.ctrader_client import (
        place_market_order_resilient,
        refresh_user_ctrader_token,
        resolve_ctrader_ctid,
    )

    user = db.query(User).filter(User.id == cfg.demo_user_id).first()
    if not user:
        return None
    prefs = db.query(UserPreference).filter(UserPreference.user_id == user.id).first()
    if not prefs or not prefs.ctrader_access_token:
        logger.warning("[gold-ai-trader] demo user missing cTrader token")
        return None

    ctid_str = resolve_ctrader_ctid(
        execution_account_id=cfg.demo_ctrader_account_id,
        prefs_default=prefs.ctrader_account_id,
    )
    if not ctid_str:
        logger.warning("[gold-ai-trader] no demo ctid configured")
        return None
    ctid = int(ctid_str)
    try:
        assert_demo_account(prefs, ctid, cfg)
    except DemoAccountRequired as e:
        logger.error("[gold-ai-trader] DEMO LOCK: %s", e)
        return None

    direction = _parse_direction(decision)
    if not direction:
        return None

    parsed = _parse_prices(decision)
    if not parsed:
        logger.warning("[gold-ai-trader] invalid prices entry/sl/tp")
        return None
    direction, entry, sl, tp = parsed

    token = prefs.ctrader_access_token
    result = await place_market_order_resilient(
        user_id=user.id,
        access_token=token,
        ctid=ctid,
        prefs=prefs,
        symbol_name=SYMBOL,
        direction=direction,
        volume_lots=cfg.min_lot,
        stop_loss_price=sl,
        take_profit_price=tp,
        entry_price=entry,
        label="GoldAITrader",
        execution_id=decision_id,
    )
    if not result or not result.get("actual_fill"):
        logger.warning("[gold-ai-trader] order failed: %s", result)
        return None

    fill = float(result["actual_fill"])
    strategy_id = ensure_system_strategy(db, user.id)
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
        notes=f"gold_ai_trader decision_id={decision_id}",
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
