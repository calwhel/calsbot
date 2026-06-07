"""
Async cTrader order queue — decouples signal evaluation from broker placement.

The forex executor cycle stays ~5s even when many users fire at once; orders
are placed sequentially per account (via per-account locks in ctrader_client).
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

_queue: Optional[asyncio.Queue] = None
_worker_started = False


@dataclass
class CtraderOrderJob:
  user_id: int
  strategy_id: int
  execution_id: int
  symbol: str
  direction: str
  entry_price: float
  tp_pct: float
  sl_pct: float
  risk_pct: float = 1.0
  risk_usd: Optional[float] = None
  use_risk_pct: bool = False
  sl_pips: Optional[float] = None
  fixed_lots: Optional[float] = None
  asset_class: str = "forex"
  tp2_pct: Optional[float] = None
  partial_close_pct: Optional[float] = None
  broker: str = "ctrader"


def _get_queue() -> asyncio.Queue:
    global _queue
    if _queue is None:
        _queue = asyncio.Queue(maxsize=500)
    return _queue


def start_ctrader_order_worker() -> None:
    global _worker_started
    if _worker_started:
        return
    _worker_started = True
    try:
        asyncio.get_running_loop().create_task(_ctrader_order_worker())
    except RuntimeError:
        asyncio.get_event_loop().create_task(_ctrader_order_worker())
    logger.info("[ctrader-queue] order worker started")


async def enqueue_ctrader_order(job: CtraderOrderJob) -> bool:
    """Queue a live cTrader order. Returns False if queue is full."""
    start_ctrader_order_worker()
    try:
        _get_queue().put_nowait(job)
        return True
    except asyncio.QueueFull:
        logger.error(f"[ctrader-queue] full — dropping exec #{job.execution_id}")
        return False


async def _apply_order_result(job: CtraderOrderJob, order_result: Optional[dict]) -> None:
    from app.database import SessionLocal
    from app.strategy_models import StrategyExecution, StrategyPerformance

    db = SessionLocal()
    try:
        from app.models import User
        user = db.query(User).filter(User.id == job.user_id).first()
        execution = db.query(StrategyExecution).filter(
            StrategyExecution.id == job.execution_id
        ).first()
        if not execution or not user:
            return

        order_id = None
        actual_fill = None
        position_id = None
        account_id = None
        order_err = None
        volume = None

        if order_result:
            order_id = order_result.get("order_id")
            actual_fill = order_result.get("actual_fill")
            position_id = order_result.get("position_id")
            account_id = order_result.get("account_id")
            order_err = order_result.get("error")
            volume = order_result.get("volume")

        from app.strategy_models import UserStrategy, StrategyPortalSettings

        strategy = db.query(UserStrategy).filter(
            UserStrategy.id == job.strategy_id
        ).first()
        portal_settings = db.query(StrategyPortalSettings).filter(
            StrategyPortalSettings.user_id == job.user_id
        ).first()

        if not order_id:
            execution.is_paper = True
            execution.notes = (
                f"Live→Paper fallback (queued order): {(order_err or 'no order id')[:180]}"
            )
            db.commit()
            if user and strategy and (not portal_settings or portal_settings.dm_live_alerts):
                try:
                    from app.services.strategy_executor import _telegram_int_id, _tg_send
                    tg_id = _telegram_int_id(user)
                    if tg_id:
                        asyncio.create_task(_tg_send(
                            tg_id,
                            f"⚠️ <b>cTrader order failed — paper trade started</b>\n"
                            f"Strategy: <b>{strategy.name}</b>\n"
                            f"Signal: {job.symbol} {job.direction}\n"
                            f"Error: <code>{(order_err or 'no order id')[:120]}</code>",
                        ))
                except Exception:
                    pass
            return

        execution.ctrader_order_id = str(order_id)
        if position_id:
            execution.ctrader_position_id = str(position_id)
        if account_id:
            execution.ctrader_account_id = str(account_id)
        if volume:
            try:
                execution.broker_volume_units = int(volume)
            except Exception:
                pass

        _n = (execution.notes or "").strip()
        if position_id and f"pos={position_id}" not in _n:
            _acct_tok = f" | acct={account_id}" if account_id else ""
            _vol_tok = f" | vol={volume}" if volume else ""
            execution.notes = (f"{_n} | pos={position_id}{_acct_tok}{_vol_tok}".strip(" |"))

        if (
            actual_fill and actual_fill > 0 and execution.entry_price
            and abs(actual_fill - execution.entry_price) > execution.entry_price * 1e-7
        ):
            _delta = actual_fill - execution.entry_price
            execution.entry_price = actual_fill
            if execution.sl_price:
                execution.sl_price += _delta
            if execution.tp_price:
                execution.tp_price += _delta
            if execution.tp2_price:
                execution.tp2_price += _delta

        db.commit()

        if user and strategy and (not portal_settings or portal_settings.dm_live_alerts):
            try:
                from app.services.strategy_executor import (
                    _fmt_open_card, _telegram_int_id, _tg_send,
                )
                tg_id = _telegram_int_id(user)
                if tg_id and execution.entry_price and execution.tp_price and execution.sl_price:
                    entry = actual_fill if actual_fill and actual_fill > 0 else execution.entry_price
                    tp_pct = abs(execution.tp_price - entry) / entry * 100 if entry else 0
                    sl_pct = abs(execution.sl_price - entry) / entry * 100 if entry else 0
                    tp2_pct = None
                    if execution.tp2_price and entry:
                        tp2_pct = abs(execution.tp2_price - entry) / entry * 100
                    asyncio.create_task(_tg_send(
                        tg_id,
                        _fmt_open_card(
                            strategy_name=strategy.name or "Your Strategy",
                            symbol=job.symbol,
                            direction=job.direction,
                            entry=entry,
                            tp_price=execution.tp_price,
                            tp_pct=tp_pct,
                            tp2_price=execution.tp2_price,
                            tp2_pct=tp2_pct,
                            sl_price=execution.sl_price,
                            sl_pct=sl_pct,
                            leverage=execution.leverage or 1,
                            conditions=execution.conditions_met or [],
                            is_paper=False,
                            order_id=str(order_id),
                            asset_class=job.asset_class,
                        ),
                    ))
            except Exception as notify_err:
                logger.warning(f"[ctrader-queue] fill notify exec #{job.execution_id}: {notify_err}")
    except Exception as e:
        logger.exception(f"[ctrader-queue] apply result exec #{job.execution_id}: {e}")
        try:
            db.rollback()
        except Exception:
            pass
    finally:
        db.close()


async def _ctrader_order_worker() -> None:
    while True:
        job: CtraderOrderJob = await _get_queue().get()
        try:
            from app.database import SessionLocal
            from app.models import User
            from app.services.ctrader_client import place_ctrader_order_for_user

            db = SessionLocal()
            try:
                user = db.query(User).filter(User.id == job.user_id).first()
            finally:
                db.close()

            if not user:
                await _apply_order_result(job, None)
                continue

            order_result = await place_ctrader_order_for_user(
                user=user,
                symbol=job.symbol,
                direction=job.direction,
                entry_price=job.entry_price,
                tp_pct=job.tp_pct,
                sl_pct=job.sl_pct,
                risk_pct=job.risk_pct,
                risk_usd=job.risk_usd,
                use_risk_pct=job.use_risk_pct,
                sl_pips=job.sl_pips,
                fixed_lots=job.fixed_lots,
            )
            if order_result and not order_result.get("account_id"):
                from app.models import UserPreference
                db2 = SessionLocal()
                try:
                    prefs = db2.query(UserPreference).filter(
                        UserPreference.user_id == user.id
                    ).first()
                    if prefs and prefs.ctrader_account_id:
                        order_result["account_id"] = prefs.ctrader_account_id
                finally:
                    db2.close()
            await _apply_order_result(job, order_result)
        except Exception as e:
            logger.exception(f"[ctrader-queue] job exec #{job.execution_id} failed: {e}")
            await _apply_order_result(job, {"error": str(e)})
        finally:
            _get_queue().task_done()


# Per-strategy gate stats from last executor cycle (in-memory, executor worker only)
_GATE_STATS: Dict[int, Dict[str, int]] = {}
_GATE_STATS_AT: Dict[int, str] = {}


def record_gate_stats(strategy_id: int, stats: Dict[str, int]) -> None:
    _GATE_STATS[strategy_id] = dict(stats)
    _GATE_STATS_AT[strategy_id] = datetime.utcnow().isoformat() + "Z"


def get_gate_stats(strategy_id: int) -> Dict[str, Any]:
    return {
        "stats": _GATE_STATS.get(strategy_id, {}),
        "updated_at": _GATE_STATS_AT.get(strategy_id),
    }
