"""Idempotent full-schema bootstrap for fresh or half-built Postgres databases.

Uses SQLAlchemy ``Base.metadata.sorted_tables`` (after importing all ORM modules)
so FK order is automatic — no hand-maintained tier lists.

Gated behind ``SCHEMA_MIGRATION_LOCK_ID`` so only one gunicorn worker runs DDL.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Callable, Iterable, List, Optional, Sequence, Set

from sqlalchemy import inspect, text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import DBAPIError, OperationalError, ProgrammingError

from app.advisory_lock_ids import SCHEMA_MIGRATION_LOCK_ID
from app.database import Base, engine

logger = logging.getLogger(__name__)

# Raw-DDL tables created outside ORM metadata (lazy / legacy SQL).
LAZY_DDL_TABLES: frozenset[str] = frozenset(
    {
        "affiliate_applications",
        "executor_loop_heartbeats",
        "executor_runtime_heartbeat",
        "fmp_rate_events",
        "market_spot_ticks",
        "owner_notification_dedup",
        "twitter_account_growth",
        "twitter_daily_trends",
        "twitter_post_metrics",
        "twitter_schedule_slots",
        "wall_snapshots",
        "wall_watches",
        "weekly_coach_reports",
    }
)


SCHEMA_RESET_ONCE_ENV = "SCHEMA_RESET_ONCE"

CRITICAL_TABLES: tuple[str, ...] = (
    "users",
    "strategy_executions",
    "user_strategies",
    "gold_ai_config",
    "gold_ai_decisions",
    "gold_ai_outcomes",
)


@dataclass
class BootstrapResult:
    ran: bool = False
    ok: bool = False
    orm_expected: int = 0
    orm_present: int = 0
    orm_missing: List[str] = field(default_factory=list)
    lazy_missing: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


def register_all_models() -> None:
    """Import every ORM module so tables register on shared ``Base.metadata``."""
    import app.models  # noqa: F401
    import app.strategy_models  # noqa: F401
    import app.strategy_marketplace_ext  # noqa: F401
    import app.social_models  # noqa: F401
    import app.gold_ai_trader.models  # noqa: F401
    import app.services.error_handler  # noqa: F401


def orm_table_names() -> List[str]:
    register_all_models()
    return [t.name for t in Base.metadata.sorted_tables]


def all_expected_table_names() -> Set[str]:
    return set(orm_table_names()) | set(LAZY_DDL_TABLES)


def _is_benign_ddl_error(exc: BaseException) -> bool:
    msg = str(exc).lower()
    if any(
        token in msg
        for token in (
            "already exists",
            "duplicate key",
            "duplicate table",
            "pg_type_typname_nsp_index",
            "unique constraint",
        )
    ):
        return True
    orig = getattr(exc, "orig", None)
    if orig is not None and orig is not exc:
        return _is_benign_ddl_error(orig)
    return False


def create_orm_tables(bind: Engine) -> None:
    """``CREATE TABLE`` every ORM model in FK order (``checkfirst=True`` per table)."""
    register_all_models()
    tables = list(Base.metadata.sorted_tables)
    insp = inspect(bind)
    logger.info("[schema-bootstrap] creating %s ORM tables (sorted_tables order)", len(tables))
    for table in tables:
        try:
            table.create(bind=bind, checkfirst=True)
        except (OperationalError, ProgrammingError, DBAPIError) as exc:
            if _is_benign_ddl_error(exc):
                if insp.has_table(table.name):
                    logger.info("[schema-bootstrap] skip existing/raced table %s", table.name)
                    continue
                msg = (
                    f"CREATE TABLE {table.name} hit benign-classified error but table "
                    f"is still missing (likely orphaned pg_type): {exc}"
                )
                logger.error("[schema-bootstrap] %s", msg)
                raise RuntimeError(msg) from exc
            raise
        if not insp.has_table(table.name):
            raise RuntimeError(
                f"CREATE TABLE {table.name} returned without error but table is still missing"
            )


def _run_step(label: str, fn: Callable[[], None], errors: List[str]) -> None:
    try:
        fn()
    except Exception as exc:
        if _is_benign_ddl_error(exc):
            logger.info("[schema-bootstrap] %s benign race: %s", label, exc)
            return
        logger.warning("[schema-bootstrap] %s failed: %s", label, exc)
        errors.append(f"{label}: {exc}")


def _ensure_affiliate_applications(bind: Engine) -> None:
    with bind.begin() as conn:
        conn.execute(
            text(
                """
                CREATE TABLE IF NOT EXISTS affiliate_applications (
                    id SERIAL PRIMARY KEY,
                    user_id INTEGER NOT NULL,
                    telegram VARCHAR NOT NULL,
                    twitter VARCHAR,
                    instagram VARCHAR,
                    youtube VARCHAR,
                    tiktok VARCHAR,
                    website VARCHAR,
                    bio TEXT NOT NULL,
                    plan TEXT NOT NULL,
                    status VARCHAR NOT NULL DEFAULT 'pending',
                    sub_share_pct DOUBLE PRECISION DEFAULT 30,
                    fee_share_pct DOUBLE PRECISION DEFAULT 20,
                    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
                    reviewed_at TIMESTAMP,
                    reviewer_note TEXT,
                    UNIQUE(user_id)
                )
                """
            )
        )
        conn.execute(
            text(
                "CREATE INDEX IF NOT EXISTS idx_affiliate_apps_status "
                "ON affiliate_applications(status)"
            )
        )


def _ensure_owner_notification_dedup(bind: Engine) -> None:
    with bind.begin() as conn:
        conn.execute(
            text(
                """
                CREATE TABLE IF NOT EXISTS owner_notification_dedup (
                    dedupe_key VARCHAR(128) PRIMARY KEY,
                    sent_at TIMESTAMP NOT NULL DEFAULT NOW()
                )
                """
            )
        )


def _ensure_weekly_coach_reports(bind: Engine) -> None:
    with bind.begin() as conn:
        conn.execute(
            text(
                """
                CREATE TABLE IF NOT EXISTS weekly_coach_reports (
                    id           SERIAL PRIMARY KEY,
                    user_id      INTEGER NOT NULL,
                    week_start   DATE    NOT NULL,
                    report_json  JSONB   NOT NULL,
                    generated_at TIMESTAMP DEFAULT NOW(),
                    UNIQUE (user_id, week_start)
                )
                """
            )
        )
        conn.execute(
            text(
                "CREATE INDEX IF NOT EXISTS idx_wcr_user_week "
                "ON weekly_coach_reports(user_id, week_start DESC)"
            )
        )


def _ensure_executor_tables(bind: Engine) -> None:
    with bind.begin() as conn:
        conn.execute(
            text(
                """
                CREATE TABLE IF NOT EXISTS executor_loop_heartbeats (
                    loop_name VARCHAR(64) PRIMARY KEY,
                    last_seen_at TIMESTAMP NOT NULL DEFAULT (NOW() AT TIME ZONE 'utc'),
                    host_pid INTEGER
                )
                """
            )
        )
        conn.execute(
            text(
                """
                CREATE TABLE IF NOT EXISTS executor_runtime_heartbeat (
                    process_key VARCHAR(32) PRIMARY KEY,
                    last_seen_at TIMESTAMP NOT NULL DEFAULT (NOW() AT TIME ZONE 'utc'),
                    lock_id INTEGER,
                    host_pid INTEGER
                )
                """
            )
        )


def _ensure_twitter_tables(bind: Engine) -> None:
    stmts = (
        """
        CREATE TABLE IF NOT EXISTS twitter_schedule_slots (
            id          SERIAL PRIMARY KEY,
            slot_date   DATE NOT NULL,
            slot_key    TEXT NOT NULL,
            created_at  TIMESTAMPTZ DEFAULT NOW(),
            UNIQUE (slot_date, slot_key)
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS twitter_post_metrics (
            id               SERIAL PRIMARY KEY,
            tweet_id         TEXT UNIQUE NOT NULL,
            account_name     TEXT NOT NULL,
            post_type        TEXT NOT NULL,
            tweet_text       TEXT,
            impressions      INTEGER,
            likes            INTEGER,
            retweets         INTEGER,
            replies          INTEGER,
            quotes           INTEGER,
            metrics_fetched  BOOLEAN DEFAULT FALSE,
            fetch_attempts   INTEGER DEFAULT 0,
            posted_at        TIMESTAMPTZ DEFAULT NOW(),
            metrics_fetched_at TIMESTAMPTZ
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS twitter_account_growth (
            id              SERIAL PRIMARY KEY,
            account_name    TEXT NOT NULL,
            handle          TEXT NOT NULL,
            followers       INTEGER,
            following       INTEGER,
            tweet_count     INTEGER,
            listed_count    INTEGER,
            checked_at      TIMESTAMPTZ DEFAULT NOW()
        )
        """,
        """
        CREATE INDEX IF NOT EXISTS idx_growth_account_time
        ON twitter_account_growth (account_name, checked_at DESC)
        """,
        """
        CREATE TABLE IF NOT EXISTS twitter_daily_trends (
            id           SERIAL PRIMARY KEY,
            trend_date   DATE NOT NULL,
            symbol       TEXT,
            topic        TEXT,
            kind         TEXT NOT NULL DEFAULT 'coin',
            trend_score  FLOAT DEFAULT 0,
            avg_likes    FLOAT DEFAULT 0,
            mentions     INTEGER DEFAULT 0,
            sample_tweet TEXT,
            discovered_at TIMESTAMPTZ DEFAULT NOW()
        )
        """,
        """
        CREATE INDEX IF NOT EXISTS idx_daily_trends_date
        ON twitter_daily_trends (trend_date DESC)
        """,
    )
    with bind.begin() as conn:
        for sql in stmts:
            conn.execute(text(sql))


def ensure_lazy_ddl_tables(bind: Engine, errors: Optional[List[str]] = None) -> None:
    """Create Tier-4 / raw-SQL tables (IF NOT EXISTS)."""
    err = errors if errors is not None else []

    def _spot() -> None:
        from app.services.spot_price_store import _ensure_table

        _ensure_table()

    def _wall() -> None:
        from app.services.wall_intel import init_wall_intel_schema

        init_wall_intel_schema()

    def _assignments() -> None:
        from app.services.strategy_account_assignments import (
            ensure_strategy_account_assignments_table,
        )

        ensure_strategy_account_assignments_table(bind)

    def _fmp() -> None:
        from app.services.fmp_price_feed import _ensure_fmp_rate_table

        _ensure_fmp_rate_table()

    def _live_fire() -> None:
        from app.services.live_order_failure import ensure_live_fire_failures_table

        ensure_live_fire_failures_table(bind)

    steps: Sequence[tuple[str, Callable[[], None]]] = (
        ("market_spot_ticks", _spot),
        ("wall_intel", _wall),
        ("strategy_account_assignments_ddl", _assignments),
        ("fmp_rate_events", _fmp),
        ("live_fire_failures_ddl", _live_fire),
        ("affiliate_applications", lambda: _ensure_affiliate_applications(bind)),
        ("owner_notification_dedup", lambda: _ensure_owner_notification_dedup(bind)),
        ("weekly_coach_reports", lambda: _ensure_weekly_coach_reports(bind)),
        ("executor_heartbeats", lambda: _ensure_executor_tables(bind)),
        ("twitter_tables", lambda: _ensure_twitter_tables(bind)),
    )
    for label, fn in steps:
        _run_step(label, fn, err)


def _present_tables(bind: Engine, names: Iterable[str]) -> Set[str]:
    insp = inspect(bind)
    present: Set[str] = set()
    for name in names:
        if insp.has_table(name):
            present.add(name)
    return present


def verify_orm_tables(bind: Engine) -> List[str]:
    expected = orm_table_names()
    present = _present_tables(bind, expected)
    return sorted(set(expected) - present)


def verify_lazy_tables(bind: Engine) -> List[str]:
    present = _present_tables(bind, LAZY_DDL_TABLES)
    return sorted(LAZY_DDL_TABLES - present)


def verify_all_tables(bind: Engine) -> tuple[List[str], List[str]]:
    return verify_orm_tables(bind), verify_lazy_tables(bind)


def schema_is_complete(bind: Engine) -> bool:
    orm_missing, lazy_missing = verify_all_tables(bind)
    return not orm_missing and not lazy_missing


def reset_public_schema(bind: Engine) -> None:
    """Destructive: drop and recreate ``public`` schema (PostgreSQL only)."""
    if bind.dialect.name != "postgresql":
        raise RuntimeError("reset_public_schema requires PostgreSQL")
    logger.warning("[schema-bootstrap] DROP SCHEMA public CASCADE — all data will be lost")
    with bind.begin() as conn:
        conn.execute(text("DROP SCHEMA IF EXISTS public CASCADE"))
        conn.execute(text("CREATE SCHEMA public"))
        conn.execute(text("GRANT ALL ON SCHEMA public TO public"))
        conn.execute(text("GRANT ALL ON SCHEMA public TO CURRENT_USER"))


def wait_for_schema(
    bind: Engine,
    *,
    timeout_seconds: float = 120.0,
    poll_seconds: float = 2.0,
) -> bool:
    """Poll until all expected tables exist (another worker may be bootstrapping)."""
    import time

    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if schema_is_complete(bind):
            return True
        time.sleep(poll_seconds)
    return schema_is_complete(bind)


def _try_advisory_lock(conn) -> bool:
    if conn.dialect.name != "postgresql":
        return True
    return bool(
        conn.execute(
            text("SELECT pg_try_advisory_lock(:lid)"),
            {"lid": SCHEMA_MIGRATION_LOCK_ID},
        ).scalar()
    )


def _advisory_unlock(conn) -> None:
    if conn.dialect.name != "postgresql":
        return
    try:
        conn.execute(
            text("SELECT pg_advisory_unlock(:lid)"),
            {"lid": SCHEMA_MIGRATION_LOCK_ID},
        )
    except Exception:
        pass


def bootstrap_schema(
    bind: Engine,
    *,
    force: bool = False,
    skip_lock: bool = False,
) -> BootstrapResult:
    """Full idempotent bootstrap under a single-worker advisory lock."""
    result = BootstrapResult()
    expected = orm_table_names()
    result.orm_expected = len(expected)
    errors: List[str] = []

    conn = bind.connect()
    lock_held = skip_lock
    try:
        if not skip_lock:
            lock_held = _try_advisory_lock(conn)
            if not lock_held:
                logger.info(
                    "[schema-bootstrap] skipped — advisory lock %s held by another worker",
                    SCHEMA_MIGRATION_LOCK_ID,
                )
                orm_missing, lazy_missing = verify_all_tables(bind)
                result.orm_missing = orm_missing
                result.lazy_missing = lazy_missing
                result.orm_present = result.orm_expected - len(orm_missing)
                result.ok = not orm_missing and not lazy_missing
                return result

        result.ran = True
        logger.info("[schema-bootstrap] starting (lock=%s)", SCHEMA_MIGRATION_LOCK_ID)

        create_orm_tables(bind)

        try:
            from app.gold_ai_trader.schema import ensure_gold_ai_trader_schema

            ensure_gold_ai_trader_schema(force=force)
        except Exception as exc:
            _run_step("gold_ai_schema", lambda: (_ for _ in ()).throw(exc), errors)

        ensure_lazy_ddl_tables(bind, errors)

        orm_missing, lazy_missing = verify_all_tables(bind)
        result.orm_missing = orm_missing
        result.lazy_missing = lazy_missing
        result.orm_present = result.orm_expected - len(orm_missing)
        result.errors = errors
        result.ok = not orm_missing and not lazy_missing and not errors

        if result.ok:
            logger.info(
                "[schema-bootstrap] complete — %s ORM + %s lazy tables present",
                result.orm_present,
                len(LAZY_DDL_TABLES) - len(lazy_missing),
            )
        else:
            msg = (
                f"Schema bootstrap incomplete — orm_missing={orm_missing} "
                f"lazy_missing={lazy_missing} errors={errors}"
            )
            logger.error("[schema-bootstrap] %s", msg)
            raise RuntimeError(msg)
        return result
    finally:
        if lock_held and not skip_lock:
            _advisory_unlock(conn)
        conn.close()


def _print_verify(bind: Engine) -> int:
    return log_verify_summary(bind, label="schema-bootstrap")


def schema_reset_once_enabled() -> bool:
    """True when ``SCHEMA_RESET_ONCE`` env is set (one-shot destructive rebuild)."""
    value = os.environ.get(SCHEMA_RESET_ONCE_ENV, "").strip().lower()
    return value in ("1", "true", "yes", "on")


def verify_summary(bind: Engine) -> tuple[int, dict]:
    orm_missing, lazy_missing = verify_all_tables(bind)
    expected = orm_table_names()
    present = _present_tables(bind, expected)
    lazy_present = _present_tables(bind, LAZY_DDL_TABLES)
    insp = inspect(bind)
    missing_critical = [t for t in CRITICAL_TABLES if not insp.has_table(t)]
    ok = not orm_missing and not lazy_missing and not missing_critical
    return (0 if ok else 1), {
        "orm_expected": len(expected),
        "orm_present": len(present),
        "orm_missing": orm_missing,
        "lazy_expected": len(LAZY_DDL_TABLES),
        "lazy_present": len(lazy_present),
        "lazy_missing": lazy_missing,
        "missing_critical": missing_critical,
        "critical_present": [t for t in CRITICAL_TABLES if t not in missing_critical],
    }


def log_verify_summary(bind: Engine, *, label: str = "schema-bootstrap") -> int:
    """Log and print ORM/lazy/critical table verification (for deploy logs)."""
    code, summary = verify_summary(bind)
    lines = [
        f"[{label}] ════════════════════════════════════════════════════════",
        f"[{label}] ORM tables: {summary['orm_present']}/{summary['orm_expected']} present",
        f"[{label}] Lazy DDL tables: {summary['lazy_present']}/{summary['lazy_expected']} present",
        f"[{label}] Critical tables present: {', '.join(summary['critical_present'])}",
    ]
    if summary["orm_missing"]:
        lines.append(f"[{label}] ORM missing: {', '.join(summary['orm_missing'])}")
    if summary["lazy_missing"]:
        lines.append(f"[{label}] Lazy missing: {', '.join(summary['lazy_missing'])}")
    if summary["missing_critical"]:
        lines.append(
            f"[{label}] CRITICAL missing: {', '.join(summary['missing_critical'])}"
        )
    status = "SUCCESS" if code == 0 else "FAILED"
    lines.append(f"[{label}] Verify: {status}")
    lines.append(f"[{label}] ════════════════════════════════════════════════════════")
    for line in lines:
        logger.info(line)
        print(line, flush=True)
    return code


def run_schema_reset_once_if_env(bind: Optional[Engine] = None) -> bool:
    """Run destructive reset + bootstrap when ``SCHEMA_RESET_ONCE`` is set.

    Intended to run once from ``start.sh`` before gunicorn workers fork.
    Uses the schema migration advisory lock so only one process resets when
    several Railway replicas start simultaneously.
    """
    if not schema_reset_once_enabled():
        return False

    bind = bind or engine
    tag = "SCHEMA_RESET_ONCE"
    logger.warning("[%s] env flag set — DROP SCHEMA public CASCADE + full bootstrap", tag)
    print(f"[{tag}] env flag set — starting destructive schema rebuild", flush=True)

    conn = bind.connect()
    lock_held = False
    try:
        lock_held = _try_advisory_lock(conn)
        if not lock_held:
            logger.info("[%s] another replica holds lock — waiting for schema", tag)
            print(f"[{tag}] waiting for another replica to finish reset…", flush=True)
            if wait_for_schema(bind, timeout_seconds=300.0):
                code = log_verify_summary(bind, label=tag)
                if code != 0:
                    raise SystemExit(code)
                print(
                    f"[{tag}] schema ready (built by another replica) — "
                    f"remove {SCHEMA_RESET_ONCE_ENV} from Railway vars",
                    flush=True,
                )
                return True
            raise SystemExit(
                f"{tag}: lock held by another process but schema still incomplete after 300s"
            )

        reset_public_schema(bind)
        bootstrap_schema(bind, force=True, skip_lock=True)
        code = log_verify_summary(bind, label=tag)
        if code != 0:
            raise SystemExit(code)
        print(
            f"[{tag}] SUCCESS — remove {SCHEMA_RESET_ONCE_ENV} from Railway vars "
            f"after confirming these log lines",
            flush=True,
        )
        return True
    finally:
        if lock_held:
            _advisory_unlock(conn)
        conn.close()


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Bootstrap TradeHub Postgres schema")
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Only verify tables exist; do not run DDL",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force gold_ai column alters (ensure_gold_ai_trader_schema force=True)",
    )
    parser.add_argument(
        "--skip-lock",
        action="store_true",
        help="Skip advisory lock (use for one-shot Railway shell runs)",
    )
    parser.add_argument(
        "--reset-schema",
        action="store_true",
        help="DROP SCHEMA public CASCADE then CREATE SCHEMA public (PostgreSQL, destructive)",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.reset_schema:
        reset_public_schema(engine)

    if args.verify_only:
        return _print_verify(engine)

    result = bootstrap_schema(engine, force=args.force, skip_lock=args.skip_lock)
    if not result.ran and not args.skip_lock:
        print("Bootstrap skipped (another worker holds the lock). Re-run with --skip-lock in shell.")
    elif result.ran:
        print(f"Bootstrap ran: ok={result.ok}")
    return log_verify_summary(engine, label="schema-bootstrap")


if __name__ == "__main__":
    sys.exit(main())
