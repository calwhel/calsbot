"""
Tick-driven live forex management + paper SL/TP checks.

Invoked from the cTrader spot-stream on each ProtoOASpotEvent. Debounced to at
most one manage pass per second per open position on that symbol.
"""
from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

_TICK_DEBOUNCE_S = float(__import__("os").environ.get("EXECUTOR_TICK_DEBOUNCE_S", "1.0"))
_REGISTRY_BACKSTOP_S = 60.0
_last_manage_mono: Dict[Tuple[str, int], float] = {}
_live_registry: Dict[str, List[dict]] = {}
_paper_registry: Dict[str, List[dict]] = {}
_registry_mono: float = 0.0
_threshold_cross_mono: Dict[int, float] = {}


def register_live_position(work: dict) -> None:
    """In-memory registry — call on open/fill."""
    sym = (work.get("symbol") or "").upper()
    eid = int(work.get("exec_id") or 0)
    if not sym or not eid:
        return
    items = _live_registry.setdefault(sym, [])
    for i, w in enumerate(items):
        if int(w.get("exec_id") or 0) == eid:
            items[i] = dict(work)
            return
    items.append(dict(work))


def unregister_position(exec_id: int, symbol: str = "") -> None:
    """Remove from registry on close."""
    eid = int(exec_id)
    syms = [symbol.upper()] if symbol else list(_live_registry.keys())
    for sym in syms:
        _live_registry[sym] = [
            w for w in _live_registry.get(sym, [])
            if int(w.get("exec_id") or 0) != eid
        ]
    for sym in list(_paper_registry.keys()):
        _paper_registry[sym] = [
            w for w in _paper_registry.get(sym, [])
            if int(w.get("exec_id") or 0) != eid
        ]
    _threshold_cross_mono.pop(eid, None)


def _should_run(symbol: str, exec_id: int) -> bool:
    key = (symbol.upper(), int(exec_id))
    now = time.monotonic()
    last = _last_manage_mono.get(key, 0.0)
    if now - last < _TICK_DEBOUNCE_S:
        return False
    _last_manage_mono[key] = now
    if len(_last_manage_mono) > 2000:
        cutoff = now - _TICK_DEBOUNCE_S * 2
        for k in list(_last_manage_mono):
            if _last_manage_mono[k] < cutoff:
                del _last_manage_mono[k]
    return True


def _registry_backstop_refresh() -> None:
    """DB backstop — refresh in-memory registry at most every 60s."""
    global _registry_mono, _live_registry, _paper_registry
    now = time.monotonic()
    if now - _registry_mono < _REGISTRY_BACKSTOP_S:
        return
    _registry_mono = now

    try:
        from app.services.strategy_executor import _build_forex_worklist

        fresh: Dict[str, List[dict]] = {}
        for w in _build_forex_worklist():
            sym = (w.get("symbol") or "").upper()
            if sym:
                fresh.setdefault(sym, []).append(dict(w))
        _live_registry = fresh
    except Exception as exc:
        logger.debug("[tick-manage] live registry backstop failed: %s", exc)

    try:
        from app.database import BgSessionLocal as SessionLocal
        from app.strategy_models import StrategyExecution

        fresh_paper: Dict[str, List[dict]] = {}
        db = SessionLocal()
        try:
            rows = (
                db.query(StrategyExecution)
                .filter(
                    StrategyExecution.outcome == "OPEN",
                    StrategyExecution.is_paper == True,  # noqa: E712
                    StrategyExecution.asset_class.in_(("forex", "index")),
                )
                .all()
            )
            for ex in rows:
                sym = (ex.symbol or "").upper()
                if not sym:
                    continue
                fresh_paper.setdefault(sym, []).append({
                    "exec_id": ex.id,
                    "symbol": ex.symbol,
                    "strategy_id": ex.strategy_id,
                })
        finally:
            db.close()
        _paper_registry = fresh_paper
    except Exception as exc:
        logger.debug("[tick-manage] paper registry backstop failed: %s", exc)


async def on_ctrader_tick(symbol: str, mid: float) -> None:
    """Run trade management for open positions on this symbol (debounced, non-blocking)."""
    if not symbol or not mid or mid <= 0:
        return
    sym = symbol.upper()
    await asyncio.to_thread(_registry_backstop_refresh)

    live_items = list(_live_registry.get(sym, []))
    for w in live_items:
        eid = int(w.get("exec_id") or 0)
        if not eid or not _should_run(sym, eid):
            continue
        t_cross = time.monotonic()
        _threshold_cross_mono[eid] = t_cross
        try:
            from app.services.strategy_executor import _amend_forex_position_tick

            asyncio.create_task(
                _amend_forex_position_tick(w, tick_mono=t_cross, mid=float(mid)),
                name=f"tick-manage-{eid}",
            )
        except Exception as exc:
            logger.debug("[tick-manage] live exec#%s dispatch: %s", eid, exc)

    paper_items = list(_paper_registry.get(sym, []))
    for p in paper_items:
        eid = int(p.get("exec_id") or 0)
        if not eid or not _should_run(sym, eid):
            continue
        try:
            asyncio.create_task(
                _manage_paper_on_tick(eid, sym, float(mid)),
                name=f"tick-paper-{eid}",
            )
        except Exception as exc:
            logger.debug("[tick-manage] paper exec#%s dispatch: %s", eid, exc)


async def _manage_paper_on_tick(exec_id: int, symbol: str, mid: float) -> None:
    from app.database import BgSessionLocal as SessionLocal
    from app.strategy_models import StrategyExecution, UserStrategy
    from app.services.trade_management import (
        check_directional_exit_hit,
        manage_open_position,
        validate_close_sanity,
    )
    from app.services.strategy_executor import (
        _close_paper_execution,
        _evaluate_paper_position_against_candles,
    )

    now_ms = int(datetime.utcnow().timestamp() * 1000)
    candles = [[now_ms, mid, mid, mid, mid]]

    db = SessionLocal()
    try:
        ex = db.query(StrategyExecution).filter(StrategyExecution.id == exec_id).first()
        if not ex or ex.outcome != "OPEN" or not ex.is_paper:
            return
        hit = check_directional_exit_hit(ex, mid, mid)
        if hit:
            outcome, exit_px, kind = hit
            outcome, label = validate_close_sanity(ex, outcome, exit_px, kind)
            _close_paper_execution(ex, outcome, exit_px, db, close_label=label)
            unregister_position(exec_id, symbol)
            return
        strat = db.query(UserStrategy).filter(UserStrategy.id == ex.strategy_id).first()
        if strat:
            await manage_open_position(ex, strat.config or {}, mid, db)
        _evaluate_paper_position_against_candles(ex, candles, db)
    finally:
        db.close()
