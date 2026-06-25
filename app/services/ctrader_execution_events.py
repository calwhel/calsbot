"""
Event-driven live forex close detection via ProtoOAExecutionEvent on the
persistent cTrader spot-stream connection.

When the broker closes a position (SL/TP/manual), we update the execution row
and fire Telegram immediately (~1s). The periodic FX-reconcile loop remains as
backup for missed events while disconnected.
"""
from __future__ import annotations

import asyncio
import logging
import re
import time
from typing import Dict, Optional

logger = logging.getLogger(__name__)

_PT_EXECUTION_EVENT = 2126
_RECENT_CLOSE: Dict[int, float] = {}  # position_id → monotonic
_RECENT_CLOSE_TTL_S = 45.0


def _recently_closed(position_id: int) -> bool:
    t = _RECENT_CLOSE.get(int(position_id))
    return t is not None and (time.monotonic() - t) < _RECENT_CLOSE_TTL_S


def _mark_closed(position_id: int) -> None:
    _RECENT_CLOSE[int(position_id)] = time.monotonic()
    if len(_RECENT_CLOSE) > 500:
        cutoff = time.monotonic() - _RECENT_CLOSE_TTL_S
        for pid in list(_RECENT_CLOSE):
            if _RECENT_CLOSE[pid] < cutoff:
                del _RECENT_CLOSE[pid]


def _position_id_from_execution(ex) -> Optional[int]:
    if getattr(ex, "ctrader_position_id", None):
        try:
            return int(ex.ctrader_position_id)
        except Exception:
            pass
    m = re.search(r"pos=(\d+)", ex.notes or "")
    return int(m.group(1)) if m else None


def _find_open_exec(position_id: int, user_id: Optional[int] = None) -> Optional[dict]:
    from app.database import BgSessionLocal as SessionLocal
    from app.strategy_models import StrategyExecution
    from app.services.trade_management import CTRADER_LIVE_ASSET_CLASSES

    db = SessionLocal()
    try:
        q = (
            db.query(StrategyExecution)
            .filter(
                StrategyExecution.outcome == "OPEN",
                StrategyExecution.is_paper == False,  # noqa: E712
                StrategyExecution.asset_class.in_(CTRADER_LIVE_ASSET_CLASSES),
            )
        )
        if user_id is not None:
            q = q.filter(StrategyExecution.user_id == int(user_id))
        for ex in q.all():
            pid = _position_id_from_execution(ex)
            if pid is not None and int(pid) == int(position_id):
                return {
                    "exec_id": ex.id,
                    "user_id": ex.user_id,
                    "symbol": ex.symbol,
                    "direction": ex.direction,
                    "entry": float(ex.entry_price or 0),
                }
    finally:
        db.close()
    return None


def _classify_outcome(
    gross_profit: int,
    entry: float,
    exit_price: float,
    direction: str,
) -> str:
    if gross_profit > 0:
        return "WIN"
    if gross_profit < 0:
        return "LOSS"
    if entry > 0 and exit_price > 0:
        if (direction or "").upper() == "LONG":
            if abs(exit_price - entry) / entry < 0.00005:
                return "BREAKEVEN"
            return "WIN" if exit_price >= entry else "LOSS"
        if abs(exit_price - entry) / entry < 0.00005:
            return "BREAKEVEN"
        return "WIN" if exit_price <= entry else "LOSS"
    return "LOSS"


async def handle_execution_event(
    payload: bytes,
    *,
    ctid: int,
    user_id: int,
) -> None:
    """Process one ProtoOAExecutionEvent from the live spot stream."""
    try:
        from ctrader_open_api.messages.OpenApiMessages_pb2 import ProtoOAExecutionEvent
        from app.services.ctrader_client import _normalize_deal_price
    except ImportError:
        return

    try:
        ev = ProtoOAExecutionEvent()
        ev.ParseFromString(payload)
    except Exception as exc:
        logger.debug("[ctrader-event] parse failed: %s", exc)
        return

    if not ev.HasField("deal"):
        return
    deal = ev.deal
    if not deal.HasField("closePositionDetail"):
        return

    position_id = int(deal.positionId) if deal.positionId else 0
    if position_id <= 0:
        return
    if _recently_closed(position_id):
        return

    detail = deal.closePositionDetail
    entry_hint = None
    direction = "LONG"
    match = _find_open_exec(position_id, user_id=user_id)
    if not match:
        return

    entry_hint = match["entry"]
    direction = match["direction"] or "LONG"
    exit_price = _normalize_deal_price(float(detail.entryPrice), entry_hint)
    gross = int(detail.grossProfit) if detail.HasField("grossProfit") else 0
    outcome = _classify_outcome(gross, entry_hint, exit_price, direction)

    _mark_closed(position_id)
    logger.info(
        "[ctrader-event] position close pos=%s exec#%s %s @ %s → %s (uid=%s ctid=%s)",
        position_id,
        match["exec_id"],
        match["symbol"],
        exit_price,
        outcome,
        user_id,
        ctid,
    )

    from app.services.strategy_executor import _close_live_forex_execution_with_db_retry

    await _close_live_forex_execution_with_db_retry(
        match["exec_id"],
        outcome,
        float(exit_price),
        source="ctrader-execution-event",
        pnl_usd=round(float(gross) / 100.0, 2),
    )


def schedule_execution_event(payload: bytes, *, ctid: int, user_id: int) -> None:
    """Fire-and-forget handler safe from the feed read loop."""
    try:
        loop = asyncio.get_running_loop()
        loop.create_task(
            handle_execution_event(payload, ctid=ctid, user_id=user_id),
            name=f"ctrader-exec-{user_id}",
        )
    except RuntimeError:
        pass
