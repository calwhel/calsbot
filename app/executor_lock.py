"""Executor advisory-lock helpers — importable without loading strategy_portal_server."""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Optional

from app.advisory_lock_ids import (
    APP_NAME_EXECUTOR,
    APP_NAME_FOREX_EXECUTOR,
    EXECUTOR_LOCK_ID,
    FOREX_EXECUTOR_LOCK_ID,
    application_name_for_lock,
)

logger = logging.getLogger(__name__)

# Re-export for existing imports (strategy_portal_server, tests, etc.).
__all__ = [
    "EXECUTOR_LOCK_ID",
    "FOREX_EXECUTOR_LOCK_ID",
    "get_executor_lock_id",
    "get_executor_application_name",
    "create_lock_connection",
    "try_acquire_lock",
    "close_lock_connection",
    "reconnect_lock_connection",
    "reclaim_executor_lock",
    "is_standalone_executor",
    "NEON_LOCK_CONNECT_KWARGS",
    "LOCK_RECONNECT_ATTEMPTS",
    "LOCK_RECONNECT_DELAY_SECS",
    "LOCK_ZOMBIE_TERMINATE_AFTER",
]


def get_executor_lock_id() -> int:
    """Advisory lock id for this process (portal combined vs forex-only replica)."""
    if os.getenv("EXECUTOR_ONLY", "").lower() in ("1", "true", "yes"):
        return FOREX_EXECUTOR_LOCK_ID
    return EXECUTOR_LOCK_ID


def get_executor_application_name() -> str:
    if os.getenv("EXECUTOR_ONLY", "").lower() in ("1", "true", "yes"):
        return APP_NAME_FOREX_EXECUTOR
    return APP_NAME_EXECUTOR


# Neon SSL blips can drop the dedicated lock session. Session-scoped advisory
# locks are released when the SSL session dies — re-claim on a FRESH connection
# (never revive the dead psycopg2 session).
LOCK_RECONNECT_ATTEMPTS = 15
LOCK_RECONNECT_DELAY_SECS = 5.0
LOCK_ZOMBIE_TERMINATE_AFTER = 5  # re-claim attempts before pg_terminate_backend on stale holder

# libpq TCP keepalive — prevents Neon from closing the idle SSL session at ~1–2 min.
NEON_LOCK_CONNECT_KWARGS = {
    "connect_timeout": 10,
    "keepalives": 1,
    "keepalives_idle": 30,
    "keepalives_interval": 10,
    "keepalives_count": 5,
    "sslmode": "require",
    "options": (
        "-c tcp_keepalives_idle=30 "
        "-c statement_timeout=0 "
        "-c idle_in_transaction_session_timeout=0"
    ),
}


def is_standalone_executor() -> bool:
    return os.getenv("EXECUTOR_STANDALONE", "").lower() in ("1", "true", "yes")


def create_lock_connection(application_name: Optional[str] = None):
    """Dedicated psycopg2 session for pg_advisory_lock (must stay open)."""
    import psycopg2
    from app.config import settings

    kwargs = dict(NEON_LOCK_CONNECT_KWARGS)
    if application_name:
        kwargs["application_name"] = application_name
    return psycopg2.connect(settings.get_database_url(), **kwargs)


def try_acquire_lock(conn, lock_id: Optional[int] = None) -> bool:
    """Try to acquire a session-level advisory lock on an open connection."""
    if lock_id is None:
        lock_id = get_executor_lock_id()
    conn.autocommit = True
    with conn.cursor() as cur:
        cur.execute("SELECT pg_try_advisory_lock(%s)", (lock_id,))
        return bool(cur.fetchone()[0])


def close_lock_connection(conn) -> None:
    if conn is None:
        return
    try:
        if getattr(conn, "closed", 0):
            return
        conn.close()
    except Exception:
        pass


def reconnect_lock_connection(
    old_conn,
    *,
    lock_id: Optional[int] = None,
    max_attempts: int = LOCK_RECONNECT_ATTEMPTS,
    retry_delay: float = LOCK_RECONNECT_DELAY_SECS,
    silent: bool = False,
    application_name: Optional[str] = None,
) -> Optional[Any]:
    """Close a dead lock session and re-claim the advisory lock on a fresh session.

    pg advisory locks are session-scoped: when Neon drops the SSL connection the
    lock is already released server-side. Opening a new connection and running
    pg_try_advisory_lock is the correct recovery path — not reconnecting the dead
    socket. Only returns None when another backend genuinely holds the lock or
    the DB is unreachable after all attempts.
    """
    if lock_id is None:
        lock_id = get_executor_lock_id()
    if application_name is None:
        application_name = application_name_for_lock(lock_id)
    _log = logger.debug if silent else logger.warning

    close_lock_connection(old_conn)
    for attempt in range(1, max_attempts + 1):
        conn = None
        try:
            conn = create_lock_connection(application_name)
            if try_acquire_lock(conn, lock_id):
                if not silent:
                    logger.info(
                        "Executor lock re-claimed on fresh DB session (attempt %s/%s)",
                        attempt,
                        max_attempts,
                    )
                return conn
            close_lock_connection(conn)
            _log(
                "Executor lock re-claim attempt %s/%s: lock held by another backend",
                attempt,
                max_attempts,
            )
            if (
                not silent
                and attempt >= LOCK_ZOMBIE_TERMINATE_AFTER
                and attempt % LOCK_ZOMBIE_TERMINATE_AFTER == 0
            ):
                try:
                    from app.services.telegram_poller_lock import (
                        terminate_advisory_lock_holders,
                    )

                    n = terminate_advisory_lock_holders(
                        lock_id,
                        min_idle_seconds=0.0,
                        owner_app_prefix=application_name,
                        log_prefix="[executor_lock]",
                    )
                    if n:
                        logger.warning(
                            "Executor lock re-claim: terminated %s zombie holder(s) "
                            "for lock %s (attempt %s)",
                            n,
                            lock_id,
                            attempt,
                        )
                except Exception as term_exc:
                    logger.warning(
                        "Executor lock zombie terminate failed (attempt %s): %s",
                        attempt,
                        term_exc,
                    )
        except Exception as exc:
            close_lock_connection(conn)
            _log(
                "Executor lock re-claim attempt %s/%s failed: %s",
                attempt,
                max_attempts,
                exc,
            )
        if attempt < max_attempts:
            time.sleep(retry_delay)
    return None


def reclaim_executor_lock(*, force: bool = False, lock_id: Optional[int] = None) -> int:
    """Terminate other backends holding the executor advisory lock.

    ``force=True`` (standalone process) reclaims any holder — including a live
    connection from an old deploy or dev machine on the shared Neon DB. Gunicorn
    siblings must not use force or they thrash each other.
    """
    from app.services.telegram_poller_lock import terminate_advisory_lock_holders

    lid = lock_id if lock_id is not None else get_executor_lock_id()
    app_name = application_name_for_lock(lid)
    min_idle = 0.0 if force else 120.0
    n = terminate_advisory_lock_holders(
        lid,
        min_idle_seconds=min_idle,
        owner_app_prefix=app_name,
        log_prefix="[executor_lock]",
    )
    if n:
        logger.warning(
            "Reclaimed executor lock %s — terminated %s holder(s) (force=%s)",
            lid,
            n,
            force,
        )
    return n
