"""Demo execution via existing cTrader client + StrategyExecution row."""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, Optional

from app.gold_ai_trader.config import GoldAiRuntimeConfig, SYMBOL, ASSET_CLASS
from app.gold_ai_trader.guardrails import assert_demo_account, DemoAccountRequired

logger = logging.getLogger(__name__)

GOLD_AI_STRATEGY_NAME = "Gold AI Trader (Demo)"


def ensure_system_strategy(db, user_id: int) -> int:
    from app.strategy_models import UserStrategy

    row = (
        db.query(UserStrategy)
        .filter(
            UserStrategy.user_id == user_id,
            UserStrategy.name == GOLD_AI_STRATEGY_NAME,
        )
        .first()
    )
    if row:
        return row.id
    row = UserStrategy(
        user_id=user_id,
        name=GOLD_AI_STRATEGY_NAME,
        description="Autonomous Claude-driven XAUUSD demo trader (isolated module)",
        config={"asset_class": "forex", "symbol": SYMBOL, "gold_ai_trader": True},
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

    direction = (decision.get("direction") or "").upper()
    if direction not in ("LONG", "SHORT"):
        direction = "LONG" if direction == "long" else "SHORT" if direction == "short" else None
    if not direction:
        return None

    entry = float(decision.get("entry") or 0)
    sl = float(decision.get("stop_loss") or 0)
    tp = float(decision.get("take_profit") or 0)
    if entry <= 0 or sl <= 0 or tp <= 0:
        logger.warning("[gold-ai-trader] invalid prices entry/sl/tp")
        return None

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
