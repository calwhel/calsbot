"""Executor advisory-lock helpers — importable without loading strategy_portal_server."""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Optional

logger = logging.getLogger(__name__)

EXECUTOR_LOCK_ID = 708_110_004
# Dedicated forex/tradfi executor replica (EXECUTOR_ONLY=1) — separate from portal+crypto.
FOREX_EXECUTOR_LOCK_ID = 708_110_006


def get_executor_lock_id() -> int:
    """Advisory lock id for this process (portal combined vs forex-only replica)."""
    if os.getenv("EXECUTOR_ONLY", "").lower() in ("1", "true", "yes"):
        return FOREX_EXECUTOR_LOCK_ID
    return EXECUTOR_LOCK_ID

# Neon SSL blips can drop the dedicated lock session. Session-scoped advisory
# locks are released when the SSL session dies — re-claim on a FRESH connection
# (never revive the dead psycopg2 session).
LOCK_RECONNECT_ATTEMPTS = 15
LOCK_RECONNECT_DELAY_SECS = 1.5


def is_standalone_executor() -> bool:
    return os.getenv("EXECUTOR_STANDALONE", "").lower() in ("1", "true", "yes")


def create_lock_connection():
    """Dedicated psycopg2 session for pg_advisory_lock (must stay open)."""
    import psycopg2
    from app.config import settings

    return psycopg2.connect(
        settings.get_database_url(),
        connect_timeout=10,
        keepalives=1,
        keepalives_idle=10,
        keepalives_interval=5,
        keepalives_count=5,
    )


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
    close_lock_connection(old_conn)
    for attempt in range(1, max_attempts + 1):
        conn = None
        try:
            conn = create_lock_connection()
            if try_acquire_lock(conn, lock_id):
                logger.info(
                    "Executor lock re-claimed on fresh DB session (attempt %s/%s)",
                    attempt,
                    max_attempts,
                )
                return conn
            close_lock_connection(conn)
            logger.warning(
                "Executor lock re-claim attempt %s/%s: lock held by another backend",
                attempt,
                max_attempts,
            )
        except Exception as exc:
            close_lock_connection(conn)
            logger.warning(
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
    min_idle = 0.0 if force else 120.0
    n = terminate_advisory_lock_holders(lid, min_idle_seconds=min_idle)
    if n:
        logger.warning(
            "Reclaimed executor lock %s — terminated %s holder(s) (force=%s)",
            lid,
            n,
            force,
        )
    return n
