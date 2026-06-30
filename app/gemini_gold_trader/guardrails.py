"""Server-side guardrails — demo lock, caps, kill switch."""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Optional, Tuple

from sqlalchemy import func

from app.gemini_gold_trader.config import GeminiGoldRuntimeConfig, env_defaults
from app.gemini_gold_trader.models import GeminiGoldConfig, GeminiGoldDecision

logger = logging.getLogger(__name__)


class DemoAccountRequired(Exception):
    """Raised when order routing would not use the configured demo account."""


def merge_config(db_row: GeminiGoldConfig, env: GeminiGoldRuntimeConfig) -> GeminiGoldRuntimeConfig:
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
        confidence_threshold=int(db_row.confidence_threshold or env.confidence_threshold),
        chart_bars=env.chart_bars,
        min_sl_pips=env.min_sl_pips,
        entry_max_drift_pct=env.entry_max_drift_pct,
    )


def demo_account_configured(cfg: GeminiGoldRuntimeConfig) -> bool:
    return bool(cfg.demo_ctrader_account_id and str(cfg.demo_ctrader_account_id).strip())


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


def _gemini_execution_filter(q):
    from app.strategy_models import StrategyExecution

    return q.filter(StrategyExecution.notes.like("%gemini_gold_trader%"))


def _today_start() -> datetime:
    now = datetime.utcnow()
    return now.replace(hour=0, minute=0, second=0, microsecond=0)


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


def trades_today(db) -> int:
    return (
        db.query(func.count(GeminiGoldDecision.id))
        .filter(
            GeminiGoldDecision.ts >= _today_start(),
            GeminiGoldDecision.executed.is_(True),
        )
        .scalar()
        or 0
    )


def cost_today_usd(db) -> float:
    val = (
        db.query(func.coalesce(func.sum(GeminiGoldDecision.cost_usd), 0.0))
        .filter(GeminiGoldDecision.ts >= _calls_cutoff(db))
        .scalar()
    )
    return float(val or 0.0)


def open_position_count(db, user_id: int) -> int:
    from app.strategy_models import StrategyExecution

    q = db.query(func.count(StrategyExecution.id)).filter(
        StrategyExecution.user_id == user_id,
        StrategyExecution.symbol == "XAUUSD",
        StrategyExecution.outcome == "OPEN",
    )
    return int(_gemini_execution_filter(q).scalar() or 0)


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
    if not demo_account_configured(cfg):
        return False, "no_demo_account"
    if not cfg.demo_user_id:
        return False, "no_demo_user"
    if trades_today(db) >= cfg.max_trades_day:
        return False, "max_trades_day"
    if open_position_count(db, user_id) >= 1:
        return False, "max_open_position"
    return True, "ok"


