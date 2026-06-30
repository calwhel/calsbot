"""Cross-replica single-runner lock for Gemini Gold Trader."""
from __future__ import annotations

import logging
from typing import Any, Optional

from app.advisory_lock_ids import APP_NAME_GEMINI_GOLD_TRADER, GEMINI_GOLD_TRADER_LOCK_ID

logger = logging.getLogger(__name__)

_lock_connection: Any = None


def holds_gemini_gold_trader_lock() -> bool:
    conn = _lock_connection
    return conn is not None and getattr(conn, "closed", 1) == 0


def try_acquire_gemini_gold_trader_lock() -> bool:
    """Try once to become the sole gemini-gold scan runner (session advisory lock)."""
    global _lock_connection
    if holds_gemini_gold_trader_lock():
        return True
    try:
        from app.executor_lock import build_lock_connection, try_acquire_lock

        conn = build_lock_connection(APP_NAME_GEMINI_GOLD_TRADER)
        if not try_acquire_lock(conn, GEMINI_GOLD_TRADER_LOCK_ID):
            conn.close()
            return False
        _lock_connection = conn
        logger.info(
            "[gemini-gold] advisory lock %s acquired — this process owns the scan loop",
            GEMINI_GOLD_TRADER_LOCK_ID,
        )
        return True
    except Exception as exc:
        logger.warning("[gemini-gold] advisory lock acquire failed: %s", exc)
        return False


def release_gemini_gold_trader_lock() -> None:
    global _lock_connection
    conn = _lock_connection
    _lock_connection = None
    if not conn:
        return
    try:
        if getattr(conn, "closed", 0):
            return
        with conn.cursor() as cur:
            cur.execute("SELECT pg_advisory_unlock(%s)", (GEMINI_GOLD_TRADER_LOCK_ID,))
        conn.close()
        logger.info("[gemini-gold] advisory lock %s released", GEMINI_GOLD_TRADER_LOCK_ID)
    except Exception as exc:
        logger.warning("[gemini-gold] advisory lock release failed: %s", exc)


def lock_holder_hint() -> Optional[str]:
    """Best-effort pid of current lock holder for diagnostics."""
    try:
        from app.executor_lock import build_lock_connection

        conn = build_lock_connection(APP_NAME_GEMINI_GOLD_TRADER)
        conn.autocommit = True
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT a.pid, a.application_name, a.state
                    FROM pg_locks l
                    JOIN pg_stat_activity a ON a.pid = l.pid
                    WHERE l.locktype = 'advisory'
                      AND l.objid = %s
                      AND l.granted = true
                    LIMIT 1
                    """,
                    (GEMINI_GOLD_TRADER_LOCK_ID,),
                )
                row = cur.fetchone()
                if not row:
                    return None
                return f"pid={row[0]} app={row[1]} state={row[2]}"
        finally:
            conn.close()
    except Exception:
        return None
