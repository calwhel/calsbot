"""Server-side guardrails — demo lock, caps, kill switch."""
from __future__ import annotations

import logging
from datetime import datetime, date
from typing import Optional, Tuple

from sqlalchemy import func

from app.gold_ai_trader.config import GoldAiRuntimeConfig
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
    """DB row overrides env defaults when present."""
    return GoldAiRuntimeConfig(
        enabled=bool(db_row.enabled) and env.enabled,
        kill_switch=bool(db_row.kill_switch) or env.kill_switch,
        london_start_hour=int(db_row.london_start_hour),
        london_end_hour=int(db_row.london_end_hour),
        ny_start_hour=int(db_row.ny_start_hour),
        ny_end_hour=int(db_row.ny_end_hour),
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
        max_live_trades_day=int(getattr(db_row, "max_live_trades_day", None) or env.max_live_trades_day or 3),
        learning_every_n_closes=env.learning_every_n_closes,
        min_lot=env.min_lot,
        use_limit_entry=bool(getattr(db_row, "use_limit_entry", env.use_limit_entry)),
        pending_entry_timeout_min=int(
            getattr(db_row, "pending_entry_timeout_min", None) or env.pending_entry_timeout_min
        ),
        learning_daily_at_ny_end=bool(
            getattr(db_row, "learning_daily_at_ny_end", env.learning_daily_at_ny_end)
        ),
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


def calls_today(db) -> int:
    return (
        db.query(func.count(GoldAiDecision.id))
        .filter(GoldAiDecision.ts >= _today_start())
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
        .filter(GoldAiDecision.ts >= _today_start())
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
