"""Create Gemini Gold Trader tables (idempotent)."""
from __future__ import annotations

import logging
import os

from sqlalchemy import inspect, text

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

_GEMINI_GOLD_COLUMN_ALTERS: tuple[tuple[str, str, str], ...] = (
    ("gemini_gold_decisions", "execution_reserved_at", "TIMESTAMP"),
)

_REQUIRED_COLUMNS: dict[str, tuple[str, ...]] = {
    "gemini_gold_decisions": ("execution_reserved_at",),
}


def _missing_columns() -> dict[str, list[str]]:
    def _scan() -> dict[str, list[str]]:
        insp = inspect(engine)
        missing: dict[str, list[str]] = {}
        for table, cols in _REQUIRED_COLUMNS.items():
            if not insp.has_table(table):
                continue
            existing = {c["name"] for c in insp.get_columns(table)}
            need = [c for c in cols if c not in existing]
            if need:
                missing[table] = need
        return missing

    return run_with_db_retry(_scan, label="gemini-gold-schema-inspect")


def _apply_column_alters() -> None:
    missing = _missing_columns()
    if not missing:
        return

    def _run() -> None:
        with engine.begin() as conn:
            for table, column, typedef in _GEMINI_GOLD_COLUMN_ALTERS:
                if table not in missing or column not in missing[table]:
                    continue
                conn.execute(text(f"ALTER TABLE {table} ADD COLUMN {column} {typedef}"))
                logger.info("[gemini-gold] schema alter: %s.%s", table, column)

    run_with_db_retry(_run, label="gemini-gold-schema-alter")


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
    _apply_column_alters()
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
