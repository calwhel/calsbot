"""In-process executor lock leadership state + fail-closed fire gate checks."""

from __future__ import annotations

import logging
import threading
from typing import Dict, Optional

logger = logging.getLogger(__name__)

_STATE_LOCK = threading.Lock()
_HAS_LEADERSHIP = False
_LEADERSHIP_UNCERTAIN = True
_LOCK_ID: Optional[int] = None
_LOCK_BACKEND_PID: Optional[int] = None
_APPLICATION_NAME: Optional[str] = None
_LAST_REASON = "startup"


def mark_executor_lock_lost(reason: str = "lost") -> None:
    """Fail closed: pause eval/fire until lock ownership is re-confirmed."""
    global _HAS_LEADERSHIP, _LEADERSHIP_UNCERTAIN
    global _LOCK_ID, _LOCK_BACKEND_PID, _APPLICATION_NAME, _LAST_REASON
    with _STATE_LOCK:
        _HAS_LEADERSHIP = False
        _LEADERSHIP_UNCERTAIN = True
        _LOCK_ID = None
        _LOCK_BACKEND_PID = None
        _APPLICATION_NAME = None
        _LAST_REASON = reason


def mark_executor_lock_uncertain(reason: str = "uncertain") -> None:
    """Fail closed while reconnect/re-claim is in flight."""
    mark_executor_lock_lost(reason)


def _mark_executor_lock_acquired_pid(
    *,
    lock_id: int,
    backend_pid: int,
    application_name: str,
    reason: str,
) -> None:
    global _HAS_LEADERSHIP, _LEADERSHIP_UNCERTAIN
    global _LOCK_ID, _LOCK_BACKEND_PID, _APPLICATION_NAME, _LAST_REASON
    with _STATE_LOCK:
        _HAS_LEADERSHIP = True
        _LEADERSHIP_UNCERTAIN = False
        _LOCK_ID = int(lock_id)
        _LOCK_BACKEND_PID = int(backend_pid)
        _APPLICATION_NAME = str(application_name)
        _LAST_REASON = reason


def mark_executor_lock_acquired(
    conn,
    *,
    lock_id: int,
    application_name: str,
    reason: str = "acquired",
) -> bool:
    """Record ownership from the live lock-holding psycopg2 connection."""
    pid = None
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT pg_backend_pid()")
            row = cur.fetchone()
            if row and row[0]:
                pid = int(row[0])
    except Exception as exc:
        logger.warning(
            "[executor-lock-gate] could not read backend pid on acquire: %s",
            exc,
        )
    if not pid:
        mark_executor_lock_uncertain(f"acquired-without-backend-pid:{reason}")
        return False
    _mark_executor_lock_acquired_pid(
        lock_id=lock_id,
        backend_pid=pid,
        application_name=application_name,
        reason=reason,
    )
    return True


def executor_can_run() -> bool:
    """True only while this process has confidently confirmed lock ownership."""
    with _STATE_LOCK:
        return bool(_HAS_LEADERSHIP and not _LEADERSHIP_UNCERTAIN)


def executor_lock_snapshot() -> Dict[str, object]:
    with _STATE_LOCK:
        return {
            "has_leadership": bool(_HAS_LEADERSHIP),
            "uncertain": bool(_LEADERSHIP_UNCERTAIN),
            "lock_id": _LOCK_ID,
            "lock_backend_pid": _LOCK_BACKEND_PID,
            "application_name": _APPLICATION_NAME,
            "reason": _LAST_REASON,
        }


def verify_executor_lock_live() -> bool:
    """Cross-check lock ownership from a fresh DB session right before broker send."""
    from app.executor_lock import build_lock_connection, close_lock_connection

    snap = executor_lock_snapshot()
    if (
        not snap.get("has_leadership")
        or snap.get("uncertain")
        or not snap.get("lock_id")
        or not snap.get("lock_backend_pid")
        or not snap.get("application_name")
    ):
        return False

    lock_id = int(snap["lock_id"])
    backend_pid = int(snap["lock_backend_pid"])
    app_name = str(snap["application_name"])
    conn = None
    try:
        conn = build_lock_connection(app_name)
        conn.autocommit = True
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT 1
                FROM pg_locks l
                LEFT JOIN pg_stat_activity a ON a.pid = l.pid
                WHERE l.locktype='advisory'
                  AND l.objid = %s
                  AND l.granted = true
                  AND l.pid = %s
                  AND COALESCE(a.application_name, '') = %s
                LIMIT 1
                """,
                (lock_id, backend_pid, app_name),
            )
            ok = cur.fetchone() is not None
        if not ok:
            mark_executor_lock_uncertain("live-check-miss")
        return bool(ok)
    except Exception as exc:
        logger.warning(
            "[executor-lock-gate] live lock check failed: %s",
            exc,
        )
        mark_executor_lock_uncertain(f"live-check-error:{type(exc).__name__}")
        return False
    finally:
        close_lock_connection(conn)
