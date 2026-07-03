"""Server-side guardrails — demo lock, caps, kill switch."""
from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Optional, Tuple

from sqlalchemy import func, or_

from app.gemini_gold_trader.config import (
    EXECUTION_MODE_DEMO,
    EXECUTION_MODE_LIVE,
    GeminiGoldRuntimeConfig,
    env_defaults,
)
from app.gemini_gold_trader.models import GeminiGoldConfig, GeminiGoldDecision

logger = logging.getLogger(__name__)

_IN_FLIGHT_TTL_MIN = 30


class DemoAccountRequired(Exception):
    """Raised when order routing would not use the configured demo account."""


class LiveAccountRequired(Exception):
    """Raised when live execution routing would not use the configured live account."""


def is_live_execution_mode(cfg: GeminiGoldRuntimeConfig) -> bool:
    return (cfg.execution_mode or EXECUTION_MODE_DEMO).strip().lower() == EXECUTION_MODE_LIVE


def active_ctrader_account_id(cfg: GeminiGoldRuntimeConfig) -> Optional[str]:
    if is_live_execution_mode(cfg):
        raw = cfg.live_ctrader_account_id
    else:
        raw = cfg.demo_ctrader_account_id
    return str(raw).strip() if raw and str(raw).strip() else None


def active_lot_size(cfg: GeminiGoldRuntimeConfig) -> float:
    if is_live_execution_mode(cfg):
        return max(0.01, float(cfg.live_lot_size or 0.01))
    return max(0.01, float(cfg.demo_lot_size or 0.01))


def merge_config(db_row: GeminiGoldConfig, env: GeminiGoldRuntimeConfig) -> GeminiGoldRuntimeConfig:
    mode = str(getattr(db_row, "execution_mode", None) or env.execution_mode or EXECUTION_MODE_DEMO)
    if mode not in (EXECUTION_MODE_DEMO, EXECUTION_MODE_LIVE):
        mode = EXECUTION_MODE_DEMO
    return GeminiGoldRuntimeConfig(
        enabled=bool(db_row.enabled) and env.enabled,
        kill_switch=bool(db_row.kill_switch) or env.kill_switch,
        dry_run=bool(db_row.dry_run) if db_row.dry_run is not None else env.dry_run,
        max_calls_day=int(db_row.max_calls_day or env.max_calls_day),
        max_trades_day=int(db_row.max_trades_day or env.max_trades_day),
        scan_interval_s=env.scan_interval_s,
        model=str(db_row.model or env.model),
        demo_user_id=db_row.demo_user_id or env.demo_user_id,
        demo_ctrader_account_id=db_row.demo_ctrader_account_id or env.demo_ctrader_account_id,
        demo_lot_size=float(db_row.demo_lot_size or env.demo_lot_size),
        execution_mode=mode,
        live_ctrader_account_id=getattr(db_row, "live_ctrader_account_id", None) or env.live_ctrader_account_id,
        live_lot_size=float(getattr(db_row, "live_lot_size", None) or env.live_lot_size or 0.01),
        confidence_threshold=int(db_row.confidence_threshold or env.confidence_threshold),
        chart_bars=env.chart_bars,
        chart_bars_1m=env.chart_bars_1m,
        min_sl_pips=env.min_sl_pips,
        max_sl_pips=env.max_sl_pips,
        min_rr=env.min_rr,
        max_rr=env.max_rr,
        entry_max_drift_pct=env.entry_max_drift_pct,
        min_trade_gap_min=env.min_trade_gap_min,
    )


def demo_account_configured(cfg: GeminiGoldRuntimeConfig) -> bool:
    return bool(cfg.demo_ctrader_account_id and str(cfg.demo_ctrader_account_id).strip())


def live_account_configured(cfg: GeminiGoldRuntimeConfig) -> bool:
    return bool(cfg.live_ctrader_account_id and str(cfg.live_ctrader_account_id).strip())


def trading_account_configured(cfg: GeminiGoldRuntimeConfig) -> bool:
    if is_live_execution_mode(cfg):
        return live_account_configured(cfg)
    return demo_account_configured(cfg)


def assert_demo_account(prefs, ctid: int, cfg: GeminiGoldRuntimeConfig) -> None:
    if cfg.demo_ctrader_account_id and str(ctid) != str(cfg.demo_ctrader_account_id):
        raise DemoAccountRequired(
            f"ctid {ctid} != configured demo GEMINI_GOLD_DEMO_ACCOUNT_ID"
        )
    from app.services.ctrader_client import _account_is_live

    live = _account_is_live(prefs, ctid)
    if live is not False:
        raise DemoAccountRequired(
            f"Account {ctid} is not confirmed demo (isLive={live}) — order blocked"
        )


def assert_live_account(prefs, ctid: int, cfg: GeminiGoldRuntimeConfig) -> None:
    if cfg.live_ctrader_account_id and str(ctid) != str(cfg.live_ctrader_account_id):
        raise LiveAccountRequired(
            f"ctid {ctid} != configured live GEMINI_GOLD_LIVE_ACCOUNT_ID"
        )
    from app.services.ctrader_client import _account_is_live

    live = _account_is_live(prefs, ctid)
    if live is not True:
        raise LiveAccountRequired(
            f"Account {ctid} is not confirmed live (isLive={live}) — order blocked"
        )


def assert_execution_account(prefs, ctid: int, cfg: GeminiGoldRuntimeConfig) -> None:
    if is_live_execution_mode(cfg):
        assert_live_account(prefs, ctid, cfg)
    else:
        assert_demo_account(prefs, ctid, cfg)


def _gemini_execution_filter(q):
    from app.strategy_models import StrategyExecution

    return q.filter(StrategyExecution.notes.like("%gemini_gold_trader%"))


def _today_start() -> datetime:
    now = datetime.utcnow()
    return now.replace(hour=0, minute=0, second=0, microsecond=0)


def _in_flight_cutoff() -> datetime:
    return datetime.utcnow() - timedelta(minutes=_IN_FLIGHT_TTL_MIN)


def _calls_cutoff(db) -> datetime:
    today = _today_start()
    row = db.query(GeminiGoldConfig).filter(GeminiGoldConfig.id == 1).first()
    reset_at = getattr(row, "calls_reset_at", None) if row else None
    if reset_at is not None:
        return max(today, reset_at)
    return today


def calls_today(db) -> int:
    return (
        db.query(func.count(GeminiGoldDecision.id))
        .filter(GeminiGoldDecision.ts >= _calls_cutoff(db))
        .scalar()
        or 0
    )


def in_flight_execution_count(db) -> int:
    """Reserved slots not yet marked executed (counts toward open + daily caps)."""
    return (
        db.query(func.count(GeminiGoldDecision.id))
        .filter(
            GeminiGoldDecision.execution_reserved_at.isnot(None),
            GeminiGoldDecision.executed.is_(False),
            GeminiGoldDecision.execution_reserved_at >= _in_flight_cutoff(),
        )
        .scalar()
        or 0
    )


def trades_today(db) -> int:
    """Executed trades since UTC midnight."""
    return (
        db.query(func.count(GeminiGoldDecision.id))
        .filter(
            GeminiGoldDecision.ts >= _today_start(),
            GeminiGoldDecision.executed.is_(True),
        )
        .scalar()
        or 0
    )


def trades_today_effective(db) -> int:
    """Executed + in-flight reservations (prevents cap races)."""
    return (
        db.query(func.count(GeminiGoldDecision.id))
        .filter(
            GeminiGoldDecision.ts >= _today_start(),
            or_(
                GeminiGoldDecision.executed.is_(True),
                (
                    (GeminiGoldDecision.execution_reserved_at.isnot(None))
                    & (GeminiGoldDecision.execution_reserved_at >= _in_flight_cutoff())
                ),
            ),
        )
        .scalar()
        or 0
    )


def last_executed_trade_at(db) -> Optional[datetime]:
    return (
        db.query(func.max(GeminiGoldDecision.ts))
        .filter(GeminiGoldDecision.executed.is_(True))
        .scalar()
    )


def minutes_since_last_executed_trade(db) -> Optional[float]:
    last = last_executed_trade_at(db)
    if last is None:
        return None
    return max(0.0, (datetime.utcnow() - last).total_seconds() / 60.0)


def cost_today_usd(db) -> float:
    val = (
        db.query(func.coalesce(func.sum(GeminiGoldDecision.cost_usd), 0.0))
        .filter(GeminiGoldDecision.ts >= _calls_cutoff(db))
        .scalar()
    )
    return float(val or 0.0)


def open_position_count(
    db,
    user_id: int,
    *,
    demo_ctid: Optional[str] = None,
) -> int:
    from app.strategy_models import StrategyExecution

    q = db.query(func.count(StrategyExecution.id)).filter(
        StrategyExecution.user_id == user_id,
        StrategyExecution.symbol == "XAUUSD",
        StrategyExecution.outcome == "OPEN",
    )
    q = _gemini_execution_filter(q)
    if demo_ctid:
        ctid = str(demo_ctid).strip()
        q = q.filter(
            or_(
                StrategyExecution.ctrader_account_id == ctid,
                StrategyExecution.ctrader_account_id.is_(None),
                StrategyExecution.ctrader_account_id == "",
            )
        )
    return int(q.scalar() or 0)


def clear_stale_execution_reservations(db) -> int:
    """Drop expired in-flight slots so they stop counting toward caps."""
    cutoff = _in_flight_cutoff()
    rows = (
        db.query(GeminiGoldDecision)
        .filter(
            GeminiGoldDecision.execution_reserved_at.isnot(None),
            GeminiGoldDecision.executed.is_(False),
            GeminiGoldDecision.execution_reserved_at < cutoff,
        )
        .all()
    )
    if not rows:
        return 0
    for row in rows:
        row.execution_reserved_at = None
    db.commit()
    return len(rows)


def describe_open_cap_block(
    db,
    user_id: int,
    cfg: GeminiGoldRuntimeConfig,
) -> str:
    """Human-readable cap block for Telegram/logs."""
    from app.gemini_gold_trader.reconcile import list_open_executions

    demo_ctid = active_ctrader_account_id(cfg)
    opens = open_position_count(db, user_id, demo_ctid=demo_ctid)
    inflight = in_flight_execution_count(db)
    ids: list[str] = []
    try:
        rows = list_open_executions(db, user_id, demo_ctid=demo_ctid)
        ids = [str(r.get("execution_id")) for r in rows[:3] if r.get("execution_id")]
    except Exception:
        rows = []
    detail = f"{opens} open row(s), {inflight} in-flight"
    if ids:
        detail += f" (exec #{', #'.join(ids)}"
        if len(rows) > 3:
            detail += f", +{len(rows) - 3} more"
        detail += ")"
    return f"blocked: max_open_position — {detail}"


def effective_open_slots_used(
    db,
    user_id: int,
    cfg: Optional[GeminiGoldRuntimeConfig] = None,
) -> int:
    """Broker-open positions plus in-flight reservations."""
    ctid = active_ctrader_account_id(cfg) if cfg else None
    return int(open_position_count(db, user_id, demo_ctid=ctid)) + int(
        in_flight_execution_count(db)
    )


def check_can_call_gemini(db, cfg: GeminiGoldRuntimeConfig) -> Tuple[bool, str]:
    if cfg.kill_switch:
        return False, "kill_switch"
    if not cfg.enabled:
        return False, "disabled"
    if calls_today(db) >= cfg.max_calls_day:
        return False, "max_calls_day"
    return True, "ok"


def check_can_execute(db, cfg: GeminiGoldRuntimeConfig, user_id: int) -> Tuple[bool, str]:
    if cfg.kill_switch:
        return False, "kill_switch"
    if cfg.dry_run:
        return False, "dry_run"
    if not trading_account_configured(cfg):
        return False, "no_trading_account" if is_live_execution_mode(cfg) else "no_demo_account"
    if not cfg.demo_user_id:
        return False, "no_demo_user"
    if trades_today_effective(db) >= cfg.max_trades_day:
        return False, "max_trades_day"
    if effective_open_slots_used(db, user_id, cfg) >= 1:
        return False, describe_open_cap_block(db, user_id, cfg)
    gap_min = max(0, int(cfg.min_trade_gap_min or 0))
    if gap_min > 0:
        since = minutes_since_last_executed_trade(db)
        if since is not None and since < float(gap_min):
            return False, "min_trade_gap"
    return True, "ok"


def try_reserve_execution(
    db,
    cfg: GeminiGoldRuntimeConfig,
    user_id: int,
    decision_id: int,
) -> Tuple[bool, str]:
    """
    Atomically re-check caps and reserve one execution slot on the decision row.
    Uses FOR UPDATE on config + decision to serialize concurrent scan cycles.
    """
    row = (
        db.query(GeminiGoldDecision)
        .filter(GeminiGoldDecision.id == decision_id)
        .with_for_update()
        .first()
    )
    if not row:
        db.rollback()
        return False, "decision_not_found"
    if row.executed:
        db.rollback()
        return False, "already_executed"
    if row.execution_reserved_at is not None:
        db.rollback()
        return False, "already_reserved"

    _ = (
        db.query(GeminiGoldConfig)
        .filter(GeminiGoldConfig.id == 1)
        .with_for_update()
        .first()
    )
    can, reason = check_can_execute(db, cfg, user_id)
    if not can:
        db.rollback()
        return False, reason

    row.execution_reserved_at = datetime.utcnow()
    db.commit()
    return True, "ok"


def clear_execution_reservation(db, decision_id: int) -> None:
    row = db.query(GeminiGoldDecision).filter(GeminiGoldDecision.id == decision_id).first()
    if not row or row.executed:
        return
    row.execution_reserved_at = None
    db.commit()
