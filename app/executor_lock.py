"""Executor advisory-lock helpers — importable without loading strategy_portal_server."""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Optional
from urllib.parse import urlparse, urlunparse

from app.advisory_lock_ids import (
    APP_NAME_EXECUTOR,
    APP_NAME_FOREX_EXECUTOR,
    FOREX_EXECUTOR_LOCK_ID,
    application_name_for_lock,
)
from app.lock_ids import EXECUTOR_LOCK_ID

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

KEEPALIVE_PING_SECS = 30
_KEEPALIVE_CFG_LOGGED = False


def get_lock_database_url() -> str:
    """Direct Neon endpoint for session advisory locks (not the pooler)."""
    from app.config import settings

    url = settings.get_database_url()
    parsed = urlparse(url)
    host = parsed.hostname or ""
    if "-pooler" not in host:
        return url
    direct_host = host.replace("-pooler", "")
    netloc = parsed.netloc.replace(host, direct_host, 1)
    return urlunparse(parsed._replace(netloc=netloc))


def log_executor_lock_keepalive_config() -> None:
    """Once-per-process startup line so deploy logs prove keepalive settings."""
    global _KEEPALIVE_CFG_LOGGED
    if _KEEPALIVE_CFG_LOGGED:
        return
    _KEEPALIVE_CFG_LOGGED = True
    logger.info(
        "[executor_lock] keepalive cfg: idle=%s interval=%s count=%s ping=%ss",
        NEON_LOCK_CONNECT_KWARGS["keepalives_idle"],
        NEON_LOCK_CONNECT_KWARGS["keepalives_interval"],
        NEON_LOCK_CONNECT_KWARGS["keepalives_count"],
        KEEPALIVE_PING_SECS,
    )


def is_standalone_executor() -> bool:
    return os.getenv("EXECUTOR_STANDALONE", "").lower() in ("1", "true", "yes")


def create_lock_connection(application_name: Optional[str] = None):
    """Dedicated psycopg2 session for pg_advisory_lock (must stay open)."""
    import psycopg2

    log_executor_lock_keepalive_config()
    kwargs = dict(NEON_LOCK_CONNECT_KWARGS)
    if application_name:
        kwargs["application_name"] = application_name
    return psycopg2.connect(get_lock_database_url(), **kwargs)


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


def _pid_holds_lock(cur, pid: int, lock_id: int) -> bool:
    cur.execute(
        """
        SELECT 1
        FROM pg_locks l
        WHERE l.locktype = 'advisory'
          AND l.objid = %s
          AND l.granted = true
          AND l.pid = %s
        LIMIT 1
        """,
        (lock_id, pid),
    )
    return cur.fetchone() is not None


def terminate_lock_holders(
    lock_id: int,
    min_idle_seconds: float = 0.0,
    *,
    owner_app: str,
    log_prefix: str = "[advisory-lock]",
) -> int:
    """Terminate stale holders of *lock_id* only when app_name exactly matches *owner_app*."""
    import psycopg2
    db_url = get_lock_database_url()
    if not db_url:
        return 0

    terminated = 0
    try:
        conn = psycopg2.connect(db_url, **NEON_LOCK_CONNECT_KWARGS)
        conn.autocommit = True
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT l.pid,
                       COALESCE(a.state, 'gone'),
                       a.application_name,
                       EXTRACT(EPOCH FROM (now() - a.state_change))
                FROM pg_locks l
                LEFT JOIN pg_stat_activity a ON a.pid = l.pid
                WHERE l.locktype = 'advisory'
                  AND l.objid = %s
                  AND l.granted = true
                  AND l.pid != pg_backend_pid()
                """,
                (lock_id,),
            )
            rows = cur.fetchall()
            for pid, state, app, idle_secs in rows:
                app_str = (app or "").strip()
                if app_str != owner_app:
                    logger.info(
                        f"{log_prefix} skipping pid={pid} app={app!r} "
                        f"(not {owner_app!r}) for lock {lock_id}"
                    )
                    continue
                if not _pid_holds_lock(cur, pid, lock_id):
                    logger.info(
                        f"{log_prefix} skipping pid={pid} — no longer holds lock {lock_id}"
                    )
                    continue
                if min_idle_seconds > 0 and state != "gone" and idle_secs is not None:
                    if float(idle_secs) < float(min_idle_seconds):
                        logger.info(
                            f"{log_prefix} keeping live holder pid={pid} state={state} "
                            f"idle={float(idle_secs):.0f}s < {min_idle_seconds:.0f}s "
                            f"for lock {lock_id} (sibling worker, not stale)"
                        )
                        continue
                try:
                    cur.execute("SELECT pg_terminate_backend(%s)", (pid,))
                    ok = cur.fetchone()[0]
                    if ok:
                        terminated += 1
                        logger.warning(
                            f"{log_prefix} terminated stale holder pid={pid} "
                            f"state={state} app={app!r} idle="
                            f"{(f'{float(idle_secs):.0f}s' if idle_secs is not None else 'n/a')} "
                            f"for lock {lock_id}"
                        )
                except Exception as te:
                    logger.warning(f"{log_prefix} could not terminate pid={pid}: {te}")
        conn.close()
    except Exception as e:
        logger.error(f"{log_prefix} terminate_lock_holders failed: {e}")
    return terminated


def _terminate_executor_lock_holders(
    lock_id: int,
    min_idle_seconds: float = 0.0,
    *,
    owner_app: str,
    log_prefix: str = "[executor_lock]",
) -> int:
    return terminate_lock_holders(
        lock_id,
        min_idle_seconds,
        owner_app=owner_app,
        log_prefix=log_prefix,
    )


def reconnect_lock_connection(
    old_conn,
    *,
    lock_id: Optional[int] = None,
    max_attempts: int = LOCK_RECONNECT_ATTEMPTS,
    retry_delay: float = LOCK_RECONNECT_DELAY_SECS,
    silent: bool = False,
    application_name: Optional[str] = None,
) -> Optional[Any]:
    """Close a dead lock session and re-claim the advisory lock on a fresh session."""
    if lock_id is None:
        lock_id = get_executor_lock_id()
    if application_name is None:
        application_name = get_executor_application_name()
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
                n = _terminate_executor_lock_holders(
                    lock_id,
                    min_idle_seconds=0.0,
                    owner_app=application_name,
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
    """Terminate other backends holding the executor advisory lock."""
    lid = lock_id if lock_id is not None else get_executor_lock_id()
    app_name = application_name_for_lock(lid)
    min_idle = 0.0 if force else 120.0
    n = _terminate_executor_lock_holders(
        lid,
        min_idle_seconds=min_idle,
        owner_app=app_name,
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
