"""Create Gemini Gold Trader tables (idempotent)."""
from __future__ import annotations

import logging
import os

from sqlalchemy import inspect

from app.database import Base, engine
from app.db_resilience import run_with_db_retry
from app.gemini_gold_trader.config import env_defaults
from app.gemini_gold_trader.models import (
    GeminiGoldConfig,
    GeminiGoldDecision,
    GeminiGoldOutcome,
)

logger = logging.getLogger(__name__)

_schema_ready = False
_DDL_RETRY_ATTEMPTS = max(1, int(os.environ.get("GEMINI_GOLD_SCHEMA_DDL_RETRY_ATTEMPTS", "4")))
_DDL_RETRY_DELAY_S = max(0.1, float(os.environ.get("GEMINI_GOLD_SCHEMA_DDL_RETRY_DELAY_S", "0.75")))


def ensure_gemini_gold_trader_schema(*, force: bool = False) -> None:
    global _schema_ready
    if _schema_ready and not force:
        return
    run_with_db_retry(
        lambda: Base.metadata.create_all(
            bind=engine,
            tables=[
                GeminiGoldConfig.__table__,
                GeminiGoldDecision.__table__,
                GeminiGoldOutcome.__table__,
            ],
        ),
        max_attempts=_DDL_RETRY_ATTEMPTS,
        retry_delay=_DDL_RETRY_DELAY_S,
        label="gemini-gold-schema-create-all",
    )
    insp = inspect(engine)
    if not insp.has_table("gemini_gold_config"):
        raise RuntimeError("Gemini Gold schema repair incomplete: gemini_gold_config missing")
    if not _schema_ready or force:
        logger.info("[gemini-gold] schema ready")
    _schema_ready = True


def seed_config_if_missing(db) -> GeminiGoldConfig:
    row = db.query(GeminiGoldConfig).filter(GeminiGoldConfig.id == 1).first()
    if row:
        return row
    d = env_defaults()
    row = GeminiGoldConfig(
        id=1,
        enabled=d.enabled,
        kill_switch=d.kill_switch,
        dry_run=d.dry_run,
        max_calls_day=d.max_calls_day,
        max_trades_day=d.max_trades_day,
        model=d.model,
        demo_ctrader_account_id=d.demo_ctrader_account_id,
        demo_user_id=d.demo_user_id,
        demo_lot_size=d.demo_lot_size,
        confidence_threshold=d.confidence_threshold,
    )
    db.add(row)
    db.commit()
    db.refresh(row)
    return row
