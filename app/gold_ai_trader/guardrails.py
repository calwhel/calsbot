"""Server-side guardrails — demo lock, caps, kill switch."""
from __future__ import annotations

import logging
from datetime import datetime, date
from typing import Optional, Tuple

from sqlalchemy import func

from app.gold_ai_trader.config import GoldAiRuntimeConfig, env_defaults
from app.gold_ai_trader.models import GoldAiConfig, GoldAiDecision

logger = logging.getLogger(__name__)


class DemoAccountRequired(Exception):
    """Raised when order routing would not use the configured demo account."""


class LiveAccountRequired(Exception):
    """Raised when live mirror routing would not use a confirmed live account."""


class GuardrailBlocked(Exception):
    def __init__(self, reason: str):
        self.reason = reason
        super().__init__(reason)


def merge_config(db_row: GoldAiConfig, env: GoldAiRuntimeConfig) -> GoldAiRuntimeConfig:
    """DB row overrides env defaults when present (session hours always from shared windows)."""
    from app.services.forex_sessions import gold_ai_session_hours

    shared = gold_ai_session_hours()
    return GoldAiRuntimeConfig(
        enabled=bool(db_row.enabled) and env.enabled,
        kill_switch=bool(db_row.kill_switch) or env.kill_switch,
        london_start_hour=shared["london"]["start_hour"],
        london_end_hour=shared["london"]["end_hour"],
        ny_start_hour=shared["new_york"]["start_hour"],
        ny_end_hour=shared["new_york"]["end_hour"],
        max_calls_day=int(db_row.max_calls_day),
        max_trades_day=int(db_row.max_trades_day),
        no_overnight=bool(db_row.no_overnight),
        scan_interval_s=env.scan_interval_s,
        model=str(db_row.model or env.model),
        demo_user_id=db_row.demo_user_id or env.demo_user_id,
        demo_ctrader_account_id=db_row.demo_ctrader_account_id or env.demo_ctrader_account_id,
        live_mirror_enabled=bool(getattr(db_row, "live_mirror_enabled", False)),
        live_ctrader_account_id=getattr(db_row, "live_ctrader_account_id", None) or env.live_ctrader_account_id,
        live_lot_size=float(getattr(db_row, "live_lot_size", None) or env.live_lot_size or 0.01),
        demo_lot_size=float(getattr(db_row, "demo_lot_size", None) or env.demo_lot_size or 0.01),
        max_live_trades_day=int(getattr(db_row, "max_live_trades_day", None) or env.max_live_trades_day or 3),
        learning_every_n_closes=env.learning_every_n_closes,
        min_lot=float(getattr(db_row, "demo_lot_size", None) or env.demo_lot_size or env.min_lot or 0.01),
        use_limit_entry=bool(getattr(db_row, "use_limit_entry", env.use_limit_entry)),
        pending_entry_timeout_min=int(
            getattr(db_row, "pending_entry_timeout_min", None) or env.pending_entry_timeout_min
        ),
        learning_daily_at_ny_end=bool(
            getattr(db_row, "learning_daily_at_ny_end", env.learning_daily_at_ny_end)
        ),
        confidence_threshold=env.confidence_threshold,
        include_history_in_decisions=env.include_history_in_decisions,
    )


def demo_account_configured(cfg: GoldAiRuntimeConfig) -> bool:
    return bool(cfg.demo_ctrader_account_id and str(cfg.demo_ctrader_account_id).strip())


def assert_demo_account(prefs, ctid: int, cfg: GoldAiRuntimeConfig) -> None:
    from app.services.ctrader_client import _account_is_live

    if cfg.demo_ctrader_account_id and str(ctid) != str(cfg.demo_ctrader_account_id):
        raise DemoAccountRequired(
            f"ctid {ctid} != configured demo GOLD_AI_TRADER_DEMO_ACCOUNT_ID"
        )
    live = _account_is_live(prefs, ctid)
    if live is not False:
        raise DemoAccountRequired(
            f"Account {ctid} is not confirmed demo (isLive={live}) — order blocked"
        )


def assert_live_account(prefs, ctid: int, cfg: GoldAiRuntimeConfig) -> None:
    from app.services.ctrader_client import _account_is_live

    if not cfg.live_ctrader_account_id or str(ctid) != str(cfg.live_ctrader_account_id):
        raise LiveAccountRequired(
            f"ctid {ctid} != configured live mirror account"
        )
    live = _account_is_live(prefs, ctid)
    if live is not True:
        raise LiveAccountRequired(
            f"Account {ctid} is not confirmed live (isLive={live}) — mirror blocked"
        )


def _demo_execution_filter(q):
    """Demo-only gold AI executions (exclude live mirror rows)."""
    from app.strategy_models import StrategyExecution

    return q.filter(
        StrategyExecution.notes.like("%gold_ai_trader%"),
        ~StrategyExecution.notes.like("%live_mirror%"),
    )


def _live_mirror_execution_filter(q):
    from app.strategy_models import StrategyExecution

    return q.filter(StrategyExecution.notes.like("%gold_ai_trader_live_mirror%"))


def _today_start() -> datetime:
    now = datetime.utcnow()
    return now.replace(hour=0, minute=0, second=0, microsecond=0)


def _calls_cutoff(db) -> datetime:
    """Count Claude calls from midnight UTC or last manual reset, whichever is later."""
    today = _today_start()
    row = db.query(GoldAiConfig).filter(GoldAiConfig.id == 1).first()
    reset_at = getattr(row, "calls_reset_at", None) if row else None
    if reset_at is not None:
        return max(today, reset_at)
    return today


def reset_daily_claude_credits(db) -> datetime:
    """Zero today's Claude call/cost counters without deleting decision history."""
    row = db.query(GoldAiConfig).filter(GoldAiConfig.id == 1).first()
    if not row:
        from app.gold_ai_trader.schema import seed_config_if_missing

        row = seed_config_if_missing(db)
    now = datetime.utcnow()
    row.calls_reset_at = now
    row.updated_at = now
    db.commit()
    db.refresh(row)
    try:
        from app.gold_ai_trader.telegram_notify import clear_call_cap_notify_state

        clear_call_cap_notify_state()
    except Exception:
        pass
    logger.info("[gold-ai-trader] daily Claude credits reset at %s", now.isoformat())
    return now


def _raw_calls_since_midnight(db) -> int:
    return (
        db.query(func.count(GoldAiDecision.id))
        .filter(GoldAiDecision.ts >= _today_start())
        .scalar()
        or 0
    )


def maybe_reset_daily_claude_credits(db=None) -> bool:
    """Reset today's Claude counters when blocked or on first deploy after the cost fix."""
    import os

    owns_session = db is None
    if owns_session:
        from app.database import SessionLocal

        db = SessionLocal()
    try:
        force = os.environ.get("GOLD_AI_TRADER_RESET_DAILY_CREDITS", "").strip().lower() in (
            "1",
            "true",
            "yes",
            "on",
        )
        row = db.query(GoldAiConfig).filter(GoldAiConfig.id == 1).first()
        if not row:
            return False

        if force:
            reset_daily_claude_credits(db)
            logger.info("[gold-ai-trader] forced daily Claude credits reset (env)")
            return True

        if getattr(row, "calls_reset_at", None) is not None:
            return False

        raw = _raw_calls_since_midnight(db)
        if raw <= 0:
            return False

        cfg = merge_config(row, env_defaults())
        pre_fix_burst = raw >= 22
        blocked = raw >= cfg.max_calls_day
        if blocked or pre_fix_burst:
            reset_daily_claude_credits(db)
            logger.info(
                "[gold-ai-trader] auto daily Claude credits reset (%s calls today, cap %s)",
                raw,
                cfg.max_calls_day,
            )
            return True
        return False
    finally:
        if owns_session:
            db.close()


def calls_today(db) -> int:
    return (
        db.query(func.count(GoldAiDecision.id))
        .filter(GoldAiDecision.ts >= _calls_cutoff(db))
        .scalar()
        or 0
    )


def trades_today(db) -> int:
    return (
        db.query(func.count(GoldAiDecision.id))
        .filter(
            GoldAiDecision.ts >= _today_start(),
            GoldAiDecision.executed.is_(True),
        )
        .scalar()
        or 0
    )


def cost_today_usd(db) -> float:
    val = (
        db.query(func.coalesce(func.sum(GoldAiDecision.cost_usd), 0.0))
        .filter(GoldAiDecision.ts >= _calls_cutoff(db))
        .scalar()
    )
    return float(val or 0.0)


def open_position_count(db, user_id: int) -> int:
    from app.strategy_models import StrategyExecution
    from app.gold_ai_trader.pending_entry import pending_entry_count

    q = db.query(func.count(StrategyExecution.id)).filter(
        StrategyExecution.user_id == user_id,
        StrategyExecution.symbol == "XAUUSD",
        StrategyExecution.outcome == "OPEN",
    )
    open_exec = _demo_execution_filter(q).scalar() or 0
    return int(open_exec) + pending_entry_count(db, user_id)


def live_trades_today(db) -> int:
    return (
        db.query(func.count(GoldAiDecision.id))
        .filter(
            GoldAiDecision.ts >= _today_start(),
            GoldAiDecision.live_mirror_execution_id.isnot(None),
        )
        .scalar()
        or 0
    )


def live_open_position_count(db, user_id: int) -> int:
    from app.strategy_models import StrategyExecution

    q = db.query(func.count(StrategyExecution.id)).filter(
        StrategyExecution.user_id == user_id,
        StrategyExecution.symbol == "XAUUSD",
        StrategyExecution.outcome == "OPEN",
    )
    return _live_mirror_execution_filter(q).scalar() or 0


def demo_pnl_today_usd(db, user_id: int) -> float:
    from app.strategy_models import StrategyExecution

    q = db.query(StrategyExecution).filter(
        StrategyExecution.user_id == user_id,
        StrategyExecution.closed_at.isnot(None),
        StrategyExecution.closed_at >= _today_start(),
        StrategyExecution.outcome.in_(("WIN", "LOSS", "BREAKEVEN")),
    )
    rows = _demo_execution_filter(q).all()
    total = 0.0
    for ex in rows:
        if ex.pnl_usd is not None:
            total += float(ex.pnl_usd)
        elif (
            ex.entry_price is not None
            and ex.exit_price is not None
            and getattr(ex, "broker_volume_units", None) is not None
        ):
            try:
                units = float(ex.broker_volume_units)
                move = float(ex.exit_price) - float(ex.entry_price)
                sign = 1.0 if (ex.direction or "").upper() == "LONG" else -1.0
                total += move * units * sign
            except Exception:
                pass
        elif ex.pips_pnl is not None:
            total += float(ex.pips_pnl) * 0.1  # rough USD proxy for XAUUSD demo display
    return round(total, 2)


def live_pnl_today_usd(db, user_id: int) -> float:
    from app.strategy_models import StrategyExecution

    q = db.query(StrategyExecution).filter(
        StrategyExecution.user_id == user_id,
        StrategyExecution.closed_at.isnot(None),
        StrategyExecution.closed_at >= _today_start(),
        StrategyExecution.outcome.in_(("WIN", "LOSS", "BREAKEVEN")),
    )
    rows = _live_mirror_execution_filter(q).all()
    total = 0.0
    for ex in rows:
        if ex.pnl_usd is not None:
            total += float(ex.pnl_usd)
        elif (
            ex.entry_price is not None
            and ex.exit_price is not None
            and getattr(ex, "broker_volume_units", None) is not None
        ):
            try:
                units = float(ex.broker_volume_units)
                move = float(ex.exit_price) - float(ex.entry_price)
                sign = 1.0 if (ex.direction or "").upper() == "LONG" else -1.0
                total += move * units * sign
            except Exception:
                pass
        elif ex.pips_pnl is not None:
            total += float(ex.pips_pnl) * 0.1
    return round(total, 2)


def resolve_live_mirror_status(execution) -> tuple[str, Optional[str]]:
    """Map StrategyExecution row to mirror status + error for UI/log."""
    if not execution:
        return "skipped", None
    notes = execution.notes or ""
    if execution.outcome == "OPEN" and execution.ctrader_position_id:
        return "filled", None
    if execution.outcome == "OPEN" and not execution.ctrader_position_id:
        return "pending", None
    if "Live→Paper fallback" in notes or "live failed" in notes.lower():
        err = notes.split(":", 1)[-1].strip() if ":" in notes else notes
        return "failed", err[:500]
    if execution.outcome in ("WIN", "LOSS", "BREAKEVEN"):
        return "filled", None
    return "failed", notes[:500] if notes else "order failed"


def check_can_call_claude(db, cfg: GoldAiRuntimeConfig) -> Tuple[bool, str]:
    if cfg.kill_switch:
        return False, "kill_switch"
    if not cfg.enabled:
        return False, "disabled"
    if not demo_account_configured(cfg):
        return False, "no_demo_account"
    if calls_today(db) >= cfg.max_calls_day:
        return False, "max_calls_day"
    return True, "ok"


def check_can_execute(db, cfg: GoldAiRuntimeConfig, user_id: int) -> Tuple[bool, str]:
    if cfg.kill_switch:
        return False, "kill_switch"
    if not demo_account_configured(cfg):
        return False, "no_demo_account"
    if trades_today(db) >= cfg.max_trades_day:
        return False, "max_trades_day"
    if open_position_count(db, user_id) >= 1:
        return False, "max_open_position"
    return True, "ok"


def check_can_execute_live_mirror(db, cfg: GoldAiRuntimeConfig, user_id: int) -> Tuple[bool, str]:
    if cfg.kill_switch:
        return False, "kill_switch"
    if not cfg.live_mirror_enabled:
        return False, "live_mirror_disabled"
    if not cfg.live_ctrader_account_id:
        return False, "live_account_not_configured"
    if live_trades_today(db) >= cfg.max_live_trades_day:
        return False, "max_live_trades_day"
    if live_open_position_count(db, user_id) >= 1:
        return False, "max_live_open_position"
    return True, "ok"
