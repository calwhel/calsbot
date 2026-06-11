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
_last_manage_mono: Dict[Tuple[str, int], float] = {}
_live_by_symbol: Dict[str, List[dict]] = {}
_paper_by_symbol: Dict[str, List[dict]] = {}
_cache_mono: float = 0.0
_CACHE_TTL_S = 1.0


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


def _refresh_symbol_cache() -> None:
    global _cache_mono, _live_by_symbol, _paper_by_symbol
    now = time.monotonic()
    if now - _cache_mono < _CACHE_TTL_S:
        return
    _cache_mono = now
    _live_by_symbol = {}
    _paper_by_symbol = {}

    try:
        from app.services.strategy_executor import _build_forex_worklist

        for w in _build_forex_worklist():
            sym = (w.get("symbol") or "").upper()
            if sym:
                _live_by_symbol.setdefault(sym, []).append(w)
    except Exception as exc:
        logger.debug("[tick-manage] live worklist refresh failed: %s", exc)

    try:
        from app.database import BgSessionLocal as SessionLocal
        from app.strategy_models import StrategyExecution

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
                _paper_by_symbol.setdefault(sym, []).append({
                    "exec_id": ex.id,
                    "symbol": ex.symbol,
                    "strategy_id": ex.strategy_id,
                })
        finally:
            db.close()
    except Exception as exc:
        logger.debug("[tick-manage] paper cache refresh failed: %s", exc)


async def on_ctrader_tick(symbol: str, mid: float) -> None:
    """Run trade management for open positions on this symbol (debounced)."""
    if not symbol or not mid or mid <= 0:
        return
    sym = symbol.upper()
    await asyncio.to_thread(_refresh_symbol_cache)

    live_items = list(_live_by_symbol.get(sym, []))
    for w in live_items:
        eid = int(w.get("exec_id") or 0)
        if not eid or not _should_run(sym, eid):
            continue
        try:
            from app.services.strategy_executor import _amend_forex_position

            await _amend_forex_position(w)
        except Exception as exc:
            logger.debug("[tick-manage] live exec#%s: %s", eid, exc)

    paper_items = list(_paper_by_symbol.get(sym, []))
    for p in paper_items:
        eid = int(p.get("exec_id") or 0)
        if not eid or not _should_run(sym, eid):
            continue
        try:
            await _manage_paper_on_tick(eid, sym, float(mid))
        except Exception as exc:
            logger.debug("[tick-manage] paper exec#%s: %s", eid, exc)


async def _manage_paper_on_tick(exec_id: int, symbol: str, mid: float) -> None:
    from app.database import BgSessionLocal as SessionLocal
    from app.strategy_models import StrategyExecution, UserStrategy
    from app.services.trade_management import manage_open_position
    from app.services.strategy_executor import _evaluate_paper_position_against_candles

    now_ms = int(datetime.utcnow().timestamp() * 1000)
    candles = [[now_ms, mid, mid, mid, mid]]

    db = SessionLocal()
    try:
        ex = db.query(StrategyExecution).filter(StrategyExecution.id == exec_id).first()
        if not ex or ex.outcome != "OPEN" or not ex.is_paper:
            return
        strat = db.query(UserStrategy).filter(UserStrategy.id == ex.strategy_id).first()
        if strat:
            await manage_open_position(ex, strat.config or {}, mid, db)
        _evaluate_paper_position_against_candles(ex, candles, db)
    finally:
        db.close()
