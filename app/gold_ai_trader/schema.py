"""Create Gold AI Trader tables (idempotent)."""
from __future__ import annotations

import logging

from sqlalchemy import inspect, text

from app.database import engine, Base
from app.gold_ai_trader.models import (
    GoldAiConfig,
    GoldAiDecision,
    GoldAiLesson,
    GoldAiOutcome,
)

logger = logging.getLogger(__name__)

_GOLD_AI_CONFIG_ALTERS = (
    "ALTER TABLE gold_ai_config ADD COLUMN IF NOT EXISTS live_mirror_enabled BOOLEAN DEFAULT FALSE NOT NULL",
    "ALTER TABLE gold_ai_config ADD COLUMN IF NOT EXISTS live_ctrader_account_id VARCHAR(40)",
    "ALTER TABLE gold_ai_config ADD COLUMN IF NOT EXISTS live_lot_size FLOAT DEFAULT 0.01 NOT NULL",
    "ALTER TABLE gold_ai_config ADD COLUMN IF NOT EXISTS max_live_trades_day INTEGER DEFAULT 3 NOT NULL",
    "ALTER TABLE gold_ai_config ADD COLUMN IF NOT EXISTS live_mirror_confirmed_at TIMESTAMP",
)

_GOLD_AI_DECISION_ALTERS = (
    "ALTER TABLE gold_ai_decisions ADD COLUMN IF NOT EXISTS live_mirror_execution_id INTEGER",
    "ALTER TABLE gold_ai_decisions ADD COLUMN IF NOT EXISTS live_mirror_status VARCHAR(24)",
    "ALTER TABLE gold_ai_decisions ADD COLUMN IF NOT EXISTS live_mirror_error TEXT",
)


def _apply_alters() -> None:
    with engine.begin() as conn:
        for sql in _GOLD_AI_CONFIG_ALTERS + _GOLD_AI_DECISION_ALTERS:
            try:
                conn.execute(text(sql))
            except Exception as exc:
                logger.debug("[gold-ai-trader] alter skipped/failed: %s (%s)", sql[:60], exc)


def ensure_gold_ai_trader_schema() -> None:
    Base.metadata.create_all(
        bind=engine,
        tables=[
            GoldAiConfig.__table__,
            GoldAiDecision.__table__,
            GoldAiOutcome.__table__,
            GoldAiLesson.__table__,
        ],
    )
    _apply_alters()
    insp = inspect(engine)
    if insp.has_table("gold_ai_config"):
        logger.info("[gold-ai-trader] schema ready")


def seed_config_if_missing(db) -> GoldAiConfig:
    row = db.query(GoldAiConfig).filter(GoldAiConfig.id == 1).first()
    if row:
        return row
    from app.gold_ai_trader.config import env_defaults

    d = env_defaults()
    row = GoldAiConfig(
        id=1,
        enabled=d.enabled,
        kill_switch=d.kill_switch,
        london_start_hour=d.london_start_hour,
        london_end_hour=d.london_end_hour,
        ny_start_hour=d.ny_start_hour,
        ny_end_hour=d.ny_end_hour,
        max_calls_day=d.max_calls_day,
        max_trades_day=d.max_trades_day,
        no_overnight=d.no_overnight,
        model=d.model,
        demo_ctrader_account_id=d.demo_ctrader_account_id,
        demo_user_id=d.demo_user_id,
    )
    db.add(row)
    db.commit()
    db.refresh(row)
    return row
