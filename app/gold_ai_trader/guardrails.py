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
        learning_every_n_closes=env.learning_every_n_closes,
        min_lot=env.min_lot,
    )


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

    return (
        db.query(func.count(StrategyExecution.id))
        .filter(
            StrategyExecution.user_id == user_id,
            StrategyExecution.symbol == "XAUUSD",
            StrategyExecution.outcome == "OPEN",
            StrategyExecution.notes.like("%gold_ai_trader%"),
        )
        .scalar()
        or 0
    )


def check_can_call_claude(db, cfg: GoldAiRuntimeConfig) -> Tuple[bool, str]:
    if cfg.kill_switch:
        return False, "kill_switch"
    if not cfg.enabled:
        return False, "disabled"
    if calls_today(db) >= cfg.max_calls_day:
        return False, "max_calls_day"
    return True, "ok"


def check_can_execute(db, cfg: GoldAiRuntimeConfig, user_id: int) -> Tuple[bool, str]:
    if cfg.kill_switch:
        return False, "kill_switch"
    if trades_today(db) >= cfg.max_trades_day:
        return False, "max_trades_day"
    if open_position_count(db, user_id) >= 1:
        return False, "max_open_position"
    return True, "ok"
