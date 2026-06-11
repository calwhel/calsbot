"""Idempotent strategy_executions trade-management column migrations."""
from __future__ import annotations

import logging
import time
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)

from app.advisory_lock_ids import APP_NAME_SCHEMA_MIGRATION, SCHEMA_MIGRATION_LOCK_ID

_SCHEMA_MIGRATION_LOCK_ID = SCHEMA_MIGRATION_LOCK_ID

TRADE_MGMT_COLUMN_MIGRATIONS: List[Tuple[str, str, str]] = [
    (
        "strategy_executions",
        "breakeven_applied",
        "ALTER TABLE strategy_executions ADD COLUMN IF NOT EXISTS breakeven_applied BOOLEAN DEFAULT FALSE",
    ),
    (
        "strategy_executions",
        "tp1_done",
        "ALTER TABLE strategy_executions ADD COLUMN IF NOT EXISTS tp1_done BOOLEAN DEFAULT FALSE",
    ),
    (
        "strategy_executions",
        "tp1_closed_volume",
        "ALTER TABLE strategy_executions ADD COLUMN IF NOT EXISTS tp1_closed_volume NUMERIC",
    ),
    (
        "strategy_executions",
        "tp1_realized_pips",
        "ALTER TABLE strategy_executions ADD COLUMN IF NOT EXISTS tp1_realized_pips NUMERIC",
    ),
    (
        "strategy_executions",
        "current_sl",
        "ALTER TABLE strategy_executions ADD COLUMN IF NOT EXISTS current_sl NUMERIC",
    ),
    (
        "strategy_executions",
        "remaining_volume",
        "ALTER TABLE strategy_executions ADD COLUMN IF NOT EXISTS remaining_volume NUMERIC",
    ),
    (
        "strategy_executions",
        "mfe_pips",
        "ALTER TABLE strategy_executions ADD COLUMN IF NOT EXISTS mfe_pips NUMERIC",
    ),
    (
        "strategy_executions",
        "mae_pips",
        "ALTER TABLE strategy_executions ADD COLUMN IF NOT EXISTS mae_pips NUMERIC",
    ),
]

TRADE_MGMT_REQUIRED_COLUMNS = [col for _t, col, _ddl in TRADE_MGMT_COLUMN_MIGRATIONS]


def is_trade_mgmt_schema_error(exc: BaseException) -> bool:
    """True when psycopg2/SQLAlchemy reports a missing trade-mgmt column."""
    es = str(exc).lower()
    if "undefinedcolumn" not in es and "does not exist" not in es:
        return False
    return any(c in es for c in TRADE_MGMT_REQUIRED_COLUMNS)


def _column_exists(conn, table: str, col: str) -> bool:
    import sqlalchemy as sa

    return bool(
        conn.execute(
            sa.text(
                "SELECT 1 FROM information_schema.columns "
                "WHERE table_schema = 'public' "
                "AND table_name = :t AND column_name = :c LIMIT 1"
            ),
            {"t": table, "c": col},
        ).scalar()
    )


def trade_mgmt_columns_ready(engine) -> bool:
    with engine.connect() as conn:
        return all(
            _column_exists(conn, "strategy_executions", col)
            for col in TRADE_MGMT_REQUIRED_COLUMNS
        )


def _run_migration_locked(engine) -> bool:
    """Apply ADD COLUMN DDL under advisory lock. Returns True if lock acquired."""
    import sqlalchemy as sa
    from app.db_resilience import is_transient_db_error

    conn = engine.connect()
    got_lock = False
    try:
        conn.execute(
            sa.text("SET application_name TO :n"),
            {"n": APP_NAME_SCHEMA_MIGRATION},
        )
        got_lock = bool(
            conn.execute(
                sa.text("SELECT pg_try_advisory_lock(:lid)"),
                {"lid": _SCHEMA_MIGRATION_LOCK_ID},
            ).scalar()
        )
        if not got_lock:
            logger.info(
                "[schema] trade_mgmt_columns: another worker migrating — skip ALTER"
            )
            return False
        conn.execute(sa.text("SET lock_timeout = '8s'"))
        conn.execute(sa.text("SET statement_timeout = '30000'"))
        for table, col, ddl in TRADE_MGMT_COLUMN_MIGRATIONS:
            if _column_exists(conn, table, col):
                continue
            try:
                conn.execute(sa.text(ddl))
                conn.commit()
                logger.info("[schema] trade_mgmt_columns: added %s.%s", table, col)
            except Exception as exc:
                try:
                    conn.rollback()
                except Exception:
                    pass
                es = str(exc).lower()
                if "already exists" in es or "duplicate" in es:
                    logger.info(
                        "[schema] trade_mgmt_columns: %s.%s already present", table, col,
                    )
                elif is_transient_db_error(exc):
                    logger.warning(
                        "[schema] trade_mgmt_columns: transient error on %s.%s: %s",
                        table, col, exc,
                    )
                else:
                    logger.warning(
                        "[schema] trade_mgmt_columns: %s.%s failed: %s", table, col, exc,
                    )
        return True
    finally:
        if got_lock:
            try:
                conn.execute(
                    sa.text("SELECT pg_advisory_unlock(:lid)"),
                    {"lid": _SCHEMA_MIGRATION_LOCK_ID},
                )
            except Exception:
                pass
        try:
            conn.close()
        except Exception:
            pass


def _backfill_open_rows(engine) -> None:
    import sqlalchemy as sa

    try:
        with engine.begin() as conn:
            r1 = conn.execute(
                sa.text(
                    "UPDATE strategy_executions SET current_sl = sl_price "
                    "WHERE current_sl IS NULL AND outcome = 'OPEN' "
                    "AND sl_price IS NOT NULL"
                )
            )
            r2 = conn.execute(
                sa.text(
                    "UPDATE strategy_executions SET remaining_volume = broker_volume_units "
                    "WHERE remaining_volume IS NULL AND outcome = 'OPEN' "
                    "AND broker_volume_units IS NOT NULL"
                )
            )
            r3 = conn.execute(
                sa.text(
                    "UPDATE strategy_executions SET remaining_volume = 1 "
                    "WHERE remaining_volume IS NULL AND outcome = 'OPEN'"
                )
            )
        logger.info(
            "[schema] trade_mgmt backfill: current_sl=%s remaining_volume=%s+%s",
            getattr(r1, "rowcount", 0),
            getattr(r2, "rowcount", 0),
            getattr(r3, "rowcount", 0),
        )
    except Exception as exc:
        logger.warning("[schema] trade_mgmt backfill failed (non-fatal): %s", exc)


def ensure_trade_mgmt_columns(engine, wait_seconds: float = 15.0) -> bool:
    """Ensure trade-mgmt columns exist; wait if another worker is migrating."""
    if trade_mgmt_columns_ready(engine):
        _backfill_open_rows(engine)
        return True

    _run_migration_locked(engine)

    deadline = time.monotonic() + wait_seconds
    while time.monotonic() < deadline:
        if trade_mgmt_columns_ready(engine):
            _backfill_open_rows(engine)
            logger.info("[schema] trade_mgmt columns ready")
            return True
        _run_migration_locked(engine)
        time.sleep(0.5)

    logger.error(
        "[schema] trade_mgmt columns still missing after %.0fs — "
        "executor may hit UndefinedColumn until migration completes",
        wait_seconds,
    )
    return False
