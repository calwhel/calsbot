"""
Async cTrader order queue — decouples signal evaluation from broker placement.

Orders run on a DEDICATED asyncio event-loop thread so executor scan cycles
(blocking the main loop for 30–90s) cannot delay live submission.
"""
from __future__ import annotations

import asyncio
import concurrent.futures
import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

_queue: Optional[asyncio.Queue] = None
_priority_queue: Optional[asyncio.Queue] = None
_worker_started = False
_order_loop: Optional[asyncio.AbstractEventLoop] = None
_order_thread: Optional[threading.Thread] = None
_order_loop_ready = threading.Event()


@dataclass
class CtraderSlAmendJob:
    """Priority SL amend — processed before new order placement."""
    user_id: int
    exec_id: int
    position_id: int
    new_sl: float
    keep_tp: Optional[float] = None


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
    signal_mono: float = field(default_factory=time.monotonic)
    latency: Any = None


def _get_queue() -> asyncio.Queue:
    global _queue
    if _queue is None:
        _queue = asyncio.Queue(maxsize=500)
    return _queue


def _get_priority_queue() -> asyncio.Queue:
    global _priority_queue
    if _priority_queue is None:
        _priority_queue = asyncio.Queue(maxsize=200)
    return _priority_queue


async def enqueue_ctrader_sl_amend(
    *,
    user_id: int,
    exec_id: int,
    position_id: int,
    new_sl: float,
    keep_tp: Optional[float] = None,
) -> dict:
    """Queue SL amend on the priority lane; await broker-confirmed result."""
    start_ctrader_order_worker()
    loop = _ensure_order_event_loop()
    job = CtraderSlAmendJob(
        user_id=user_id,
        exec_id=exec_id,
        position_id=position_id,
        new_sl=float(new_sl),
        keep_tp=keep_tp,
    )
    result_fut: concurrent.futures.Future = concurrent.futures.Future()
    job._result_fut = result_fut  # type: ignore[attr-defined]
    put_fut = asyncio.run_coroutine_threadsafe(_get_priority_queue().put(job), loop)
    await asyncio.wrap_future(put_fut)
    return await asyncio.wait_for(asyncio.wrap_future(result_fut), timeout=20.0)


def _ensure_order_event_loop() -> asyncio.AbstractEventLoop:
    global _order_loop, _order_thread
    if _order_loop is not None and _order_thread and _order_thread.is_alive():
        return _order_loop

    def _run() -> None:
        global _order_loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        _order_loop = loop
        _order_loop_ready.set()
        logger.info(
            "[ctrader-queue] dedicated order event loop started "
            "(isolated from executor scan / advisory-lock recovery)"
        )
        try:
            loop.run_forever()
        finally:
            try:
                loop.close()
            except Exception:
                pass

    _order_loop_ready.clear()
    _order_thread = threading.Thread(
        target=_run, daemon=True, name="ctrader-order-loop",
    )
    _order_thread.start()
    if not _order_loop_ready.wait(timeout=10.0):
        raise RuntimeError("cTrader order event loop failed to start within 10s")
    return _order_loop


def ctrader_order_worker_running() -> bool:
    return _worker_started


def start_ctrader_order_worker() -> None:
    global _worker_started
    loop = _ensure_order_event_loop()
    if _worker_started:
        return

    async def _boot() -> None:
        global _worker_started
        if _worker_started:
            return
        _worker_started = True
        asyncio.create_task(_ctrader_order_worker())
        logger.info("[ctrader-queue] order worker started")

    asyncio.run_coroutine_threadsafe(_boot(), loop)


async def enqueue_ctrader_order(job: CtraderOrderJob) -> bool:
    """Queue a live cTrader order on the isolated order loop."""
    from app.services.order_latency import new_order_latency

    start_ctrader_order_worker()
    if job.latency is None:
        job.latency = new_order_latency(job.execution_id, job.signal_mono)
    job.latency.mark_queued()
    loop = _ensure_order_event_loop()
    try:
        fut = asyncio.run_coroutine_threadsafe(_get_queue().put(job), loop)
        await asyncio.wrap_future(fut)
        return True
    except Exception as exc:
        logger.error(
            "[ctrader-queue] enqueue failed exec#%s: %s",
            job.execution_id,
            type(exc).__name__,
        )
        return False


async def _try_reconcile_ambiguous(user, job: CtraderOrderJob, order_result: dict) -> Optional[dict]:
    from app.database import SessionLocal
    from app.models import UserPreference
    from app.services.ctrader_client import (
        _host_for_account,
        is_ambiguous_order_error,
        reconcile_order_fill_after_miss,
    )

    if not is_ambiguous_order_error(order_result.get("error")):
        return None
    db = SessionLocal()
    try:
        prefs = db.query(UserPreference).filter(UserPreference.user_id == user.id).first()
        if not prefs or not prefs.ctrader_access_token or not prefs.ctrader_account_id:
            return None
        at = prefs.ctrader_access_token
        ctid = int(prefs.ctrader_account_id)
        host = _host_for_account(prefs, ctid)
    finally:
        db.close()
    return await reconcile_order_fill_after_miss(
        access_token=at,
        ctid=ctid,
        host=host,
        symbol_name=job.symbol,
        direction=job.direction,
        entry_hint=job.entry_price,
    )


async def _abort_stale_order(job: CtraderOrderJob, reason: str, age_s: float, slip_pips: float) -> None:
    from app.database import SessionLocal
    from app.strategy_models import StrategyExecution

    logger.warning(
        "[ctrader-queue] exec#%s %s %s — signal stale: %s",
        job.execution_id,
        job.symbol,
        job.direction,
        reason,
    )
    if job.latency is not None:
        job.latency.log_summary(outcome="stale")

    db = SessionLocal()
    try:
        execution = db.query(StrategyExecution).filter(
            StrategyExecution.id == job.execution_id
        ).first()
        if execution:
            execution.outcome = "CANCELLED"
            execution.notes = f"Live skip: {reason}"
            db.commit()
        from app.models import User
        from app.strategy_models import StrategyPortalSettings, UserStrategy

        user = db.query(User).filter(User.id == job.user_id).first()
        strategy = db.query(UserStrategy).filter(UserStrategy.id == job.strategy_id).first()
        portal_settings = db.query(StrategyPortalSettings).filter(
            StrategyPortalSettings.user_id == job.user_id
        ).first()
        if user and strategy and (not portal_settings or portal_settings.dm_live_alerts):
            try:
                from app.services.strategy_executor import _telegram_int_id, _tg_send
                tg_id = _telegram_int_id(user)
                if tg_id:
                    asyncio.create_task(_tg_send(
                        tg_id,
                        "⏭️ <b>Live order skipped — signal stale</b>\n"
                        f"Strategy: <b>{strategy.name}</b>\n"
                        f"Signal: {job.symbol} {job.direction}\n"
                        f"<code>{reason}</code>",
                        asset_class=job.asset_class or "forex",
                    ))
            except Exception:
                pass
    except Exception as exc:
        logger.warning("[ctrader-queue] stale abort persist failed: %s", exc)
        try:
            db.rollback()
        except Exception:
            pass
    finally:
        db.close()


async def _apply_order_result(job: CtraderOrderJob, order_result: Optional[dict]) -> None:
    from app.database import SessionLocal
    from app.strategy_models import StrategyExecution, StrategyPerformance

    if job.latency is not None:
        outcome = "fill" if order_result and order_result.get("actual_fill") else "fail"
        job.latency.log_summary(outcome=outcome)

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

        _has_fill = bool(actual_fill and actual_fill > 0)
        _has_broker_ref = bool(order_id or position_id)
        if not _has_broker_ref or not _has_fill:
            if position_id and _has_fill:
                order_id = order_id or "reconciled"
            else:
                execution.is_paper = True
                _err_txt = (order_err or "no order id")[:180]
                if order_id and not (actual_fill and actual_fill > 0):
                    _err_txt = (_err_txt + " (no broker fill confirmation)")[:180]
                execution.notes = f"Live→Paper fallback (queued order): {_err_txt}"
                db.commit()
                try:
                    from app.services.strategy_executor import _release_tg_open_notify
                    _release_tg_open_notify(db, execution.id)
                except Exception:
                    pass
                logger.warning(
                    "[ctrader-queue] exec#%s %s %s — broker order failed: %s",
                    job.execution_id,
                    job.symbol,
                    job.direction,
                    _err_txt,
                )
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
                                asset_class=job.asset_class or "forex",
                            ))
                    except Exception:
                        pass
                return

        execution.ctrader_order_id = str(order_id) if order_id else None
        if position_id:
            execution.ctrader_position_id = str(position_id)
        if account_id:
            execution.ctrader_account_id = str(account_id)
        if volume:
            try:
                execution.broker_volume_units = int(volume)
                execution.remaining_volume = float(int(volume))
            except Exception:
                pass
        if execution.sl_price is not None and execution.current_sl is None:
            execution.current_sl = float(execution.sl_price)

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
            if execution.current_sl:
                execution.current_sl += _delta
            if execution.tp_price:
                execution.tp_price += _delta
            if execution.tp2_price:
                execution.tp2_price += _delta

        db.commit()

        if position_id and execution.outcome == "OPEN":
            try:
                from app.services.forex_tick_manager import register_live_position

                register_live_position({
                    "exec_id": execution.id,
                    "symbol": execution.symbol,
                    "strategy_id": execution.strategy_id,
                    "user_id": execution.user_id,
                    "sl_price": float(execution.sl_price) if execution.sl_price else None,
                    "direction": execution.direction,
                })
            except Exception:
                pass

        from datetime import datetime as _dt
        _fill_px = actual_fill if actual_fill and actual_fill > 0 else execution.entry_price
        logger.info(
            f"[{_dt.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}] "
            f"[ctrader-queue] live fill exec#{job.execution_id} "
            f"{job.symbol} {job.direction} @ {_fill_px}"
        )

        if (
            user and strategy
            and (not portal_settings or portal_settings.dm_live_alerts)
            and actual_fill and actual_fill > 0
        ):
            try:
                from app.services.strategy_executor import (
                    _claim_tg_open_notify,
                    _fmt_open_card, _telegram_int_id, _schedule_tg_open_notify,
                )
                if not _claim_tg_open_notify(db, execution.id):
                    return
                tg_id = _telegram_int_id(user)
                if tg_id and execution.entry_price and execution.tp_price and execution.sl_price:
                    entry = float(actual_fill)
                    tp_pct = abs(execution.tp_price - entry) / entry * 100 if entry else 0
                    sl_pct = abs(execution.sl_price - entry) / entry * 100 if entry else 0
                    tp2_pct = None
                    if execution.tp2_price and entry:
                        tp2_pct = abs(execution.tp2_price - entry) / entry * 100
                    _ac = job.asset_class or "forex"
                    _schedule_tg_open_notify(
                        execution.id,
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
                            asset_class=_ac,
                        ),
                        asset_class=_ac,
                    )
            except Exception as notify_err:
                from datetime import datetime as _dt
                logger.warning(
                    f"[{_dt.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}] "
                    f"[ctrader-queue] fill notify exec #{job.execution_id}: {notify_err}"
                )
    except Exception as e:
        logger.exception(f"[ctrader-queue] apply result exec #{job.execution_id}: {e}")
        try:
            db.rollback()
        except Exception:
            pass
    finally:
        db.close()


async def _process_sl_amend_job(job: CtraderSlAmendJob) -> None:
    from app.services.ctrader_client import amend_position_sl_result

    res = await amend_position_sl_result(
        job.user_id,
        job.position_id,
        job.new_sl,
        keep_tp=job.keep_tp,
        exec_id=job.exec_id,
    )
    fut = getattr(job, "_result_fut", None)
    if fut and not fut.done():
        fut.set_result(res)


async def _dequeue_next_job():
    """Priority lane (SL amends) always ahead of new order jobs."""
    try:
        return _get_priority_queue().get_nowait()
    except asyncio.QueueEmpty:
        pass
    get_fut = asyncio.ensure_future(_get_queue().get())
    pri_fut = asyncio.ensure_future(_get_priority_queue().get())
    done, pending = await asyncio.wait(
        {get_fut, pri_fut},
        return_when=asyncio.FIRST_COMPLETED,
    )
    for p in pending:
        p.cancel()
    return next(iter(done)).result()


async def _ctrader_order_worker() -> None:
    while True:
        job = await _dequeue_next_job()
        if isinstance(job, CtraderSlAmendJob):
            try:
                await _process_sl_amend_job(job)
            except Exception as exc:
                fut = getattr(job, "_result_fut", None)
                if fut and not fut.done():
                    fut.set_result({
                        "ok": False,
                        "result": "failed",
                        "error": str(exc),
                        "broker_reply": {},
                    })
            finally:
                _get_priority_queue().task_done()
            continue

        if job.latency is not None:
            job.latency.mark_dequeue()
        queue_wait_ms = -1
        if job.latency and job.latency.queued_mono and job.latency.dequeue_mono:
            queue_wait_ms = int((job.latency.dequeue_mono - job.latency.queued_mono) * 1000)
        logger.info(
            "[ctrader-queue] placing exec#%s %s %s user=%s queue_wait=%sms",
            job.execution_id,
            job.symbol,
            job.direction,
            job.user_id,
            queue_wait_ms,
        )
        try:
            from app.services.order_stale_guard import check_signal_stale

            stale = check_signal_stale(
                symbol=job.symbol,
                direction=job.direction,
                signal_price=job.entry_price,
                signal_mono=job.signal_mono,
            )
            if stale:
                reason, age_s, slip_pips = stale
                await _abort_stale_order(job, reason, age_s, slip_pips)
                continue

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
                latency=job.latency,
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

            if order_result:
                recovered = await _try_reconcile_ambiguous(user, job, order_result)
                if recovered:
                    order_result = recovered

            await _apply_order_result(job, order_result)
        except Exception as e:
            logger.exception(f"[ctrader-queue] job exec #{job.execution_id} failed: {e}")
            await _apply_order_result(job, {"error": str(e)})
        finally:
            _get_queue().task_done()


# Per-strategy gate stats from last executor cycle (in-memory, executor worker only)
_GATE_STATS: Dict[int, Dict[str, int]] = {}
_GATE_STATS_AT: Dict[int, str] = {}


def record_gate_stats(
    strategy_id: int,
    stats: Dict[str, int],
    *,
    persist_db: bool = True,
) -> None:
    """Record per-strategy gate blockers from the last evaluate_and_fire pass."""
    _GATE_STATS[strategy_id] = dict(stats)
    at = datetime.utcnow().isoformat() + "Z"
    _GATE_STATS_AT[strategy_id] = at
    if persist_db:
        _persist_gate_stats_db(strategy_id, dict(stats), at)


def _persist_gate_stats_db(strategy_id: int, stats: Dict[str, int], at: str) -> None:
    try:
        from app.services.discovery_jobs import _job_key
        from app.strategy_models import DiscoveryScanJob
        from app.services.discovery_jobs import _db_session

        key = _job_key("executor_gate", str(strategy_id))
        payload = {"stats": dict(stats), "updated_at": at}
        db = _db_session()
        try:
            row = db.query(DiscoveryScanJob).filter(DiscoveryScanJob.job_key == key).first()
            if not row:
                row = DiscoveryScanJob(
                    job_key=key,
                    scan_type="executor_gate",
                    uid="_shared",
                    status="done",
                    message="Executor gate stats",
                )
                db.add(row)
            row.status = "done"
            row.result_json = payload
            row.updated_at = datetime.utcnow()
            db.commit()
        finally:
            db.close()
    except Exception:
        pass


def flush_gate_stats_to_db(strategy_ids: list) -> None:
    """Batch-persist in-memory gate stats once per executor cycle (single commit)."""
    if not strategy_ids:
        return
    try:
        from app.services.discovery_jobs import _job_key, _db_session
        from app.strategy_models import DiscoveryScanJob

        pending = [
            sid for sid in strategy_ids if _GATE_STATS.get(sid) is not None
        ]
        if not pending:
            return
        db = _db_session()
        try:
            keys = [_job_key("executor_gate", str(sid)) for sid in pending]
            rows = (
                db.query(DiscoveryScanJob)
                .filter(DiscoveryScanJob.job_key.in_(keys))
                .all()
            )
            by_key = {r.job_key: r for r in rows}
            now = datetime.utcnow()
            for sid in pending:
                stats = _GATE_STATS.get(sid)
                if stats is None:
                    continue
                key = _job_key("executor_gate", str(sid))
                at = _GATE_STATS_AT.get(sid) or (now.isoformat() + "Z")
                payload = {"stats": dict(stats), "updated_at": at}
                row = by_key.get(key)
                if not row:
                    row = DiscoveryScanJob(
                        job_key=key,
                        scan_type="executor_gate",
                        uid="_shared",
                        status="done",
                        message="Executor gate stats",
                    )
                    db.add(row)
                    by_key[key] = row
                row.status = "done"
                row.result_json = payload
                row.updated_at = now
            db.commit()
        finally:
            db.close()
    except Exception:
        for sid in strategy_ids:
            stats = _GATE_STATS.get(sid)
            if stats is None:
                continue
            at = _GATE_STATS_AT.get(sid) or (datetime.utcnow().isoformat() + "Z")
            _persist_gate_stats_db(sid, stats, at)


def get_gate_stats(strategy_id: int) -> Dict[str, Any]:
    mem_stats = _GATE_STATS.get(strategy_id, {})
    mem_at = _GATE_STATS_AT.get(strategy_id)
    if mem_stats:
        return {"stats": mem_stats, "updated_at": mem_at}
    try:
        from app.services.discovery_jobs import get_job
        row = get_job("executor_gate", str(strategy_id))
        if row and row.result_json:
            return {
                "stats": row.result_json.get("stats") or {},
                "updated_at": row.result_json.get("updated_at"),
            }
    except Exception:
        pass
    return {"stats": {}, "updated_at": None}


def get_gate_stats_bulk(strategy_ids: list) -> Dict[int, Dict[str, Any]]:
    """One DB round-trip for diagnostics — avoids N× get_job() on large accounts."""
    out: Dict[int, Dict[str, Any]] = {}
    if not strategy_ids:
        return out
    for sid in strategy_ids:
        mem = _GATE_STATS.get(sid)
        if mem:
            out[sid] = {"stats": mem, "updated_at": _GATE_STATS_AT.get(sid)}
    missing = [sid for sid in strategy_ids if sid not in out]
    if not missing:
        return out
    try:
        from app.services.discovery_jobs import _job_key, _db_session
        from app.strategy_models import DiscoveryScanJob

        keys = [_job_key("executor_gate", str(sid)) for sid in missing]
        db = _db_session()
        try:
            rows = (
                db.query(DiscoveryScanJob)
                .filter(DiscoveryScanJob.job_key.in_(keys))
                .all()
            )
            by_key = {r.job_key: r for r in rows}
            for sid in missing:
                row = by_key.get(_job_key("executor_gate", str(sid)))
                if row and row.result_json:
                    out[sid] = {
                        "stats": row.result_json.get("stats") or {},
                        "updated_at": row.result_json.get("updated_at"),
                    }
                else:
                    out[sid] = {"stats": {}, "updated_at": None}
        finally:
            db.close()
    except Exception:
        for sid in missing:
            out.setdefault(sid, {"stats": {}, "updated_at": None})
    return out
