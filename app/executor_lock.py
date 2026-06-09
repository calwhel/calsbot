"""Executor advisory-lock helpers — importable without loading strategy_portal_server."""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Optional

logger = logging.getLogger(__name__)

EXECUTOR_LOCK_ID = 708_110_004

# Neon SSL blips can drop the dedicated lock session; reconnect before tearing
# down the whole executor (which restarts prefetch and duplicates scan loops).
LOCK_RECONNECT_ATTEMPTS = 8
LOCK_RECONNECT_DELAY_SECS = 2.0


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


def try_acquire_lock(conn, lock_id: int = EXECUTOR_LOCK_ID) -> bool:
    """Try to acquire a session-level advisory lock on an open connection."""
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
    lock_id: int = EXECUTOR_LOCK_ID,
    max_attempts: int = LOCK_RECONNECT_ATTEMPTS,
    retry_delay: float = LOCK_RECONNECT_DELAY_SECS,
) -> Optional[Any]:
    """Close a dead lock session and race to re-acquire on a fresh connection."""
    close_lock_connection(old_conn)
    for attempt in range(1, max_attempts + 1):
        conn = None
        try:
            conn = create_lock_connection()
            if try_acquire_lock(conn, lock_id):
                logger.info(
                    "Executor lock re-acquired on new DB session (attempt %s/%s)",
                    attempt,
                    max_attempts,
                )
                return conn
            close_lock_connection(conn)
        except Exception as exc:
            close_lock_connection(conn)
            logger.warning(
                "Executor lock reconnect attempt %s/%s failed: %s",
                attempt,
                max_attempts,
                exc,
            )
        if attempt < max_attempts:
            time.sleep(retry_delay)
    return None


def reclaim_executor_lock(*, force: bool = False) -> int:
    """Terminate other backends holding the executor advisory lock.

    ``force=True`` (standalone process) reclaims any holder — including a live
    connection from an old deploy or dev machine on the shared Neon DB. Gunicorn
    siblings must not use force or they thrash each other.
    """
    from app.services.telegram_poller_lock import terminate_advisory_lock_holders

    min_idle = 0.0 if force else 120.0
    n = terminate_advisory_lock_holders(EXECUTOR_LOCK_ID, min_idle_seconds=min_idle)
    if n:
        logger.warning(
            "Reclaimed executor lock %s — terminated %s holder(s) (force=%s)",
            EXECUTOR_LOCK_ID,
            n,
            force,
        )
    return n
