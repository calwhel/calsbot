"""Create Gemini Gold Trader tables (idempotent)."""
from __future__ import annotations

import logging
import os
from typing import Optional

from sqlalchemy import inspect, text

from app.database import Base, engine
from app.db_resilience import run_with_db_retry
from app.gemini_gold_trader.config import EXECUTION_MODE_DEMO, env_defaults
from app.gemini_gold_trader.models import (
    GeminiGoldConfig,
    GeminiGoldDecision,
    GeminiGoldFunnelEvent,
    GeminiGoldOrbState,
    GeminiGoldOutcome,
    GeminiGoldPendingOrder,
)

logger = logging.getLogger(__name__)

_schema_ready = False
_DDL_RETRY_ATTEMPTS = max(1, int(os.environ.get("GEMINI_GOLD_SCHEMA_DDL_RETRY_ATTEMPTS", "4")))
_DDL_RETRY_DELAY_S = max(0.1, float(os.environ.get("GEMINI_GOLD_SCHEMA_DDL_RETRY_DELAY_S", "0.75")))

_GEMINI_GOLD_COLUMN_ALTERS: tuple[tuple[str, str, str], ...] = (
    ("gemini_gold_decisions", "execution_reserved_at", "TIMESTAMP"),
    ("gemini_gold_config", "execution_mode", "VARCHAR(16) DEFAULT 'demo' NOT NULL"),
    ("gemini_gold_config", "live_ctrader_account_id", "VARCHAR(40)"),
    ("gemini_gold_config", "live_lot_size", "FLOAT DEFAULT 0.01 NOT NULL"),
    ("gemini_gold_config", "live_confirmed_at", "TIMESTAMP"),
    ("gemini_gold_config", "live_mirror_enabled", "BOOLEAN DEFAULT FALSE NOT NULL"),
    ("gemini_gold_config", "max_live_trades_day", "INTEGER DEFAULT 3 NOT NULL"),
    ("gemini_gold_config", "live_mirror_confirmed_at", "TIMESTAMP"),
    ("gemini_gold_decisions", "live_mirror_execution_id", "INTEGER"),
    ("gemini_gold_decisions", "live_mirror_status", "VARCHAR(24)"),
    ("gemini_gold_decisions", "live_mirror_error", "TEXT"),
    ("gemini_gold_config", "use_limit_entry", "BOOLEAN DEFAULT TRUE NOT NULL"),
    ("gemini_gold_config", "pending_entry_timeout_min", "INTEGER DEFAULT 30 NOT NULL"),
    ("gemini_gold_config", "orb_enabled", "BOOLEAN DEFAULT FALSE NOT NULL"),
    ("gemini_gold_config", "orb_confidence_threshold", "INTEGER DEFAULT 65 NOT NULL"),
    ("gemini_gold_config", "orb_max_calls_day", "INTEGER DEFAULT 20 NOT NULL"),
    ("gemini_gold_config", "orb_max_trades_per_session", "INTEGER DEFAULT 1 NOT NULL"),
    ("gemini_gold_decisions", "setup_type", "VARCHAR(64)"),
)

_REQUIRED_COLUMNS: dict[str, tuple[str, ...]] = {
    "gemini_gold_decisions": (
        "execution_reserved_at",
        "live_mirror_execution_id",
        "live_mirror_status",
        "live_mirror_error",
        "setup_type",
    ),
    "gemini_gold_config": (
        "execution_mode",
        "live_ctrader_account_id",
        "live_lot_size",
        "live_confirmed_at",
        "live_mirror_enabled",
        "max_live_trades_day",
        "live_mirror_confirmed_at",
        "use_limit_entry",
        "pending_entry_timeout_min",
        "orb_enabled",
        "orb_confidence_threshold",
        "orb_max_calls_day",
        "orb_max_trades_per_session",
    ),
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
                GeminiGoldFunnelEvent.__table__,
                GeminiGoldOrbState.__table__,
                GeminiGoldPendingOrder.__table__,
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


def _migrate_legacy_demo_lot_size(db, row: Optional[GeminiGoldConfig] = None) -> None:
    """Align stored demo lot with gold-ai default (0.1 was 10× oversized for demo margin)."""
    if row is None:
        row = db.query(GeminiGoldConfig).filter(GeminiGoldConfig.id == 1).first()
    if not row:
        return
    try:
        lots = float(row.demo_lot_size or 0)
    except (TypeError, ValueError):
        return
    if abs(lots - 0.1) < 1e-9:
        row.demo_lot_size = 0.01
        db.commit()
        logger.info("[gemini-gold] migrated demo_lot_size 0.1 → 0.01")


def _migrate_legacy_confidence_threshold(db, row: Optional[GeminiGoldConfig] = None) -> None:
    if row is None:
        row = db.query(GeminiGoldConfig).filter(GeminiGoldConfig.id == 1).first()
    if not row:
        return
    try:
        threshold = int(row.confidence_threshold or 0)
    except (TypeError, ValueError):
        return
    if threshold == 60:
        row.confidence_threshold = 85
        db.commit()
        logger.info("[gemini-gold] migrated confidence_threshold 60 → 85")
        return
    if threshold == 90:
        row.confidence_threshold = 85
        db.commit()
        logger.info("[gemini-gold] migrated confidence_threshold 90 → 85")


def seed_config_if_missing(db) -> GeminiGoldConfig:
    row = db.query(GeminiGoldConfig).filter(GeminiGoldConfig.id == 1).first()
    if row:
        _migrate_legacy_demo_lot_size(db, row=row)
        _migrate_legacy_confidence_threshold(db, row=row)
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
        execution_mode=EXECUTION_MODE_DEMO,
        live_ctrader_account_id=d.live_ctrader_account_id,
        live_lot_size=d.live_lot_size,
        live_mirror_enabled=False,
        max_live_trades_day=d.max_live_trades_day,
        use_limit_entry=d.use_limit_entry,
        pending_entry_timeout_min=d.pending_entry_timeout_min,
        orb_enabled=d.orb_enabled,
        orb_confidence_threshold=d.orb_confidence_threshold,
        orb_max_calls_day=d.orb_max_calls_day,
        orb_max_trades_per_session=d.orb_max_trades_per_session,
        confidence_threshold=d.confidence_threshold,
    )
    db.add(row)
    db.commit()
    db.refresh(row)
    return row
