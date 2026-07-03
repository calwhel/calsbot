"""Create Gold AI Trader tables (idempotent)."""
from __future__ import annotations

import logging
import os

from sqlalchemy import inspect, text

from app.database import engine, Base
from app.db_resilience import run_with_db_retry
from app.gold_ai_trader.models import (
    GoldAiConfig,
    GoldAiDecision,
    GoldAiFunnelEvent,
    GoldAiLesson,
    GoldAiOrbState,
    GoldAiOutcome,
    GoldAiPendingOrder,
)

logger = logging.getLogger(__name__)

_schema_ready = False
_DDL_RETRY_ATTEMPTS = max(
    1, int(os.environ.get("GOLD_AI_SCHEMA_DDL_RETRY_ATTEMPTS", "4"))
)
_DDL_RETRY_DELAY_S = max(
    0.1, float(os.environ.get("GOLD_AI_SCHEMA_DDL_RETRY_DELAY_S", "0.75"))
)

# (table, column, type/default fragment after column name)
_GOLD_AI_COLUMN_ALTERS: tuple[tuple[str, str, str], ...] = (
    ("gold_ai_config", "live_mirror_enabled", "BOOLEAN DEFAULT FALSE NOT NULL"),
    ("gold_ai_config", "live_ctrader_account_id", "VARCHAR(40)"),
    ("gold_ai_config", "live_lot_size", "FLOAT DEFAULT 0.01 NOT NULL"),
    ("gold_ai_config", "demo_lot_size", "FLOAT DEFAULT 0.01 NOT NULL"),
    ("gold_ai_config", "max_live_trades_day", "INTEGER DEFAULT 3 NOT NULL"),
    ("gold_ai_config", "live_mirror_confirmed_at", "TIMESTAMP"),
    ("gold_ai_config", "use_limit_entry", "BOOLEAN DEFAULT TRUE NOT NULL"),
    ("gold_ai_config", "pending_entry_timeout_min", "INTEGER DEFAULT 30 NOT NULL"),
    ("gold_ai_config", "learning_daily_at_ny_end", "BOOLEAN DEFAULT TRUE NOT NULL"),
    ("gold_ai_config", "calls_reset_at", "TIMESTAMP"),
    ("gold_ai_decisions", "live_mirror_execution_id", "INTEGER"),
    ("gold_ai_decisions", "live_mirror_status", "VARCHAR(24)"),
    ("gold_ai_decisions", "live_mirror_error", "TEXT"),
    ("gold_ai_outcomes", "setup_type", "VARCHAR(64)"),
    ("gold_ai_outcomes", "session", "VARCHAR(16)"),
    ("gold_ai_outcomes", "r_multiple", "FLOAT"),
    ("gold_ai_lessons", "tokens_in", "INTEGER DEFAULT 0"),
    ("gold_ai_lessons", "tokens_out", "INTEGER DEFAULT 0"),
    ("gold_ai_lessons", "cost_usd", "FLOAT DEFAULT 0"),
)

_REQUIRED_COLUMNS: dict[str, tuple[str, ...]] = {
    "gold_ai_config": (
        "live_mirror_enabled",
        "live_ctrader_account_id",
        "live_lot_size",
        "demo_lot_size",
        "max_live_trades_day",
        "live_mirror_confirmed_at",
        "use_limit_entry",
        "pending_entry_timeout_min",
        "learning_daily_at_ny_end",
        "calls_reset_at",
    ),
    "gold_ai_decisions": (
        "live_mirror_execution_id",
        "live_mirror_status",
        "live_mirror_error",
    ),
    "gold_ai_outcomes": ("setup_type", "session", "r_multiple"),
    "gold_ai_lessons": ("tokens_in", "tokens_out", "cost_usd"),
}


def _missing_columns() -> dict[str, list[str]]:
    """Return table -> missing column names (empty when schema matches ORM)."""
    def _scan() -> dict[str, list[str]]:
        insp = inspect(engine)
        missing: dict[str, list[str]] = {}
        for table, cols in _REQUIRED_COLUMNS.items():
            if not insp.has_table(table):
                continue
            existing = {c["name"] for c in insp.get_columns(table)}
            gap = [c for c in cols if c not in existing]
            if gap:
                missing[table] = gap
        return missing

    return run_with_db_retry(
        _scan,
        max_attempts=_DDL_RETRY_ATTEMPTS,
        retry_delay=_DDL_RETRY_DELAY_S,
        label="gold-ai-schema-missing-columns",
    )


def _alter_sql(dialect: str, table: str, column: str, col_def: str) -> str:
    if dialect == "sqlite":
        return f"ALTER TABLE {table} ADD COLUMN {column} {col_def}"
    return f"ALTER TABLE {table} ADD COLUMN IF NOT EXISTS {column} {col_def}"


def _column_exists(conn, table: str, column: str) -> bool:
    dialect = engine.dialect.name
    if dialect == "sqlite":
        existing = {c["name"] for c in inspect(conn).get_columns(table)}
        return column in existing
    row = conn.execute(
        text(
            "SELECT 1 FROM information_schema.columns "
            "WHERE table_schema = 'public' AND table_name = :table_name "
            "AND column_name = :column_name LIMIT 1"
        ),
        {"table_name": table, "column_name": column},
    ).first()
    return row is not None


def _apply_alters() -> None:
    """Run each ALTER in its own transaction — PG aborts the whole txn on one failure."""
    dialect = engine.dialect.name
    for table, column, col_def in _GOLD_AI_COLUMN_ALTERS:
        try:
            def _run_one_alter() -> None:
                with engine.begin() as conn:
                    if not inspect(conn).has_table(table):
                        return
                    if _column_exists(conn, table, column):
                        return
                    sql = _alter_sql(dialect, table, column, col_def)
                    conn.execute(text(sql))

            run_with_db_retry(
                _run_one_alter,
                max_attempts=_DDL_RETRY_ATTEMPTS,
                retry_delay=_DDL_RETRY_DELAY_S,
                label=f"gold-ai-schema-alter:{table}.{column}",
            )
        except Exception as exc:
            logger.warning(
                "[gold-ai-trader] alter skipped/failed: %s.%s (%s)",
                table,
                column,
                exc,
            )


def ensure_gold_ai_trader_schema(*, force: bool = False) -> None:
    global _schema_ready
    if _schema_ready and not force:
        return
    run_with_db_retry(
        lambda: Base.metadata.create_all(
            bind=engine,
            tables=[
                GoldAiConfig.__table__,
                GoldAiDecision.__table__,
                GoldAiOutcome.__table__,
                GoldAiLesson.__table__,
                GoldAiPendingOrder.__table__,
                GoldAiFunnelEvent.__table__,
                GoldAiOrbState.__table__,
            ],
        ),
        max_attempts=_DDL_RETRY_ATTEMPTS,
        retry_delay=_DDL_RETRY_DELAY_S,
        label="gold-ai-schema-create-all",
    )
    _apply_alters()
    missing = _missing_columns()
    if missing:
        logger.warning(
            "[gold-ai-trader] schema still missing columns after alters: %s",
            missing,
        )
        _apply_alters()
        missing = _missing_columns()
    if missing:
        _schema_ready = False
        raise RuntimeError(f"Gold AI schema repair incomplete: {missing}")
    insp = inspect(engine)
    if insp.has_table("gold_ai_config"):
        if not _schema_ready or force:
            logger.info("[gold-ai-trader] schema ready")
        try:
            from app.gold_ai_trader.guardrails import maybe_reset_daily_claude_credits

            maybe_reset_daily_claude_credits()
        except Exception as exc:
            logger.warning("[gold-ai-trader] credits reset on schema ensure failed: %s", exc)
    _schema_ready = True


def seed_config_if_missing(db) -> GoldAiConfig:
    row = db.query(GoldAiConfig).filter(GoldAiConfig.id == 1).first()
    if row:
        updated = False
        # Upgrade legacy defaults in-place while preserving user custom values.
        if int(getattr(row, "max_calls_day", 0) or 0) == 22:
            row.max_calls_day = 70
            updated = True
        if (getattr(row, "model", "") or "").strip() == "claude-haiku-4-5":
            row.model = "claude-sonnet-4-6"
            updated = True
        if updated:
            db.commit()
            db.refresh(row)
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
