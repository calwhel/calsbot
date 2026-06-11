"""
PostgreSQL advisory locks — one poller per deployment across all hosts/replicas.

Uses TG_POLLER_LOCK_ID exclusively so the Telegram poller never contends with
the strategy executor lock (EXECUTOR_LOCK_ID).
"""

from __future__ import annotations

import logging
import os
import threading
import time
from typing import Dict, Optional

from app.advisory_lock_ids import (
    APP_NAME_TG_POLLER,
    TWITTER_POSTER_LOCK_ID,
    application_name_for_lock,
)
from app.lock_ids import TG_POLLER_LOCK_ID

logger = logging.getLogger(__name__)

# Re-export for existing imports.
__all__ = [
    "TG_POLLER_LOCK_ID",
    "MAIN_POLLER_LOCK_ID",
    "FOREX_POLLER_LOCK_ID",
    "TWITTER_POSTER_LOCK_ID",
    "wait_for_poller_lock",
    "release_poller_lock",
    "holds_lock",
    "try_acquire_persistent_lock",
    "describe_bot_token",
]

# Backward-compatible aliases — both bots share the single poller lock.
MAIN_POLLER_LOCK_ID = TG_POLLER_LOCK_ID
FOREX_POLLER_LOCK_ID = TG_POLLER_LOCK_ID

KEEPALIVE_INTERVAL = 30
_instance_id = str(os.getpid())

_lock_conns: Dict[int, object] = {}
_keepalive_threads: Dict[int, threading.Thread] = {}


def _get_db_url() -> str:
    return os.environ.get("NEON_DATABASE_URL") or os.environ.get("DATABASE_URL", "") or ""


def _try_acquire(lock_id: int) -> bool:
    """Try to acquire lock_id. Fail-closed — never poll without a real lock."""
    if lock_id != TG_POLLER_LOCK_ID:
        logger.error(
            "[tg-lock] refused acquire lock %s — poller may only use %s",
            lock_id,
            TG_POLLER_LOCK_ID,
        )
        return False

    import psycopg2

    db_url = _get_db_url()
    if not db_url:
        logger.error(
            "[tg-lock] NEON_DATABASE_URL missing — refusing to poll (fail-closed)"
        )
        return False

    try:
        old = _lock_conns.pop(lock_id, None)
        if old:
            try:
                old.close()
            except Exception:
                pass

        from app.executor_lock import NEON_LOCK_CONNECT_KWARGS

        conn_kw = dict(NEON_LOCK_CONNECT_KWARGS)
        conn_kw["application_name"] = APP_NAME_TG_POLLER
        conn = psycopg2.connect(db_url, **conn_kw)
        conn.autocommit = True
        with conn.cursor() as cur:
            cur.execute("SELECT pg_try_advisory_lock(%s)", (TG_POLLER_LOCK_ID,))
            acquired = bool(cur.fetchone()[0])

        if acquired:
            _lock_conns[lock_id] = conn
            logger.info(
                f"[tg-lock] acquired lock {TG_POLLER_LOCK_ID} — PID {_instance_id} may poll"
            )
            return True

        conn.close()
        logger.info(f"[tg-lock] lock {TG_POLLER_LOCK_ID} held elsewhere — waiting")
        return False
    except Exception as e:
        logger.error(
            f"[tg-lock] acquire {TG_POLLER_LOCK_ID} failed: {e} — will retry (fail-closed)"
        )
        return False


def _release(lock_id: int) -> None:
    if lock_id != TG_POLLER_LOCK_ID:
        return
    conn = _lock_conns.pop(lock_id, None)
    if conn:
        try:
            conn.close()
        except Exception:
            pass
        logger.info(f"[tg-lock] released lock {TG_POLLER_LOCK_ID} (PID {_instance_id})")


def _keepalive_loop(lock_id: int) -> None:
    while lock_id in _lock_conns:
        time.sleep(KEEPALIVE_INTERVAL)
        conn = _lock_conns.get(lock_id)
        if not conn:
            break
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
        except Exception as e:
            logger.warning(f"[tg-lock] keepalive failed for {TG_POLLER_LOCK_ID}: {e}")
            break


def _start_keepalive(lock_id: int) -> None:
    t = _keepalive_threads.get(lock_id)
    if t and t.is_alive():
        return
    t = threading.Thread(target=_keepalive_loop, args=(lock_id,), daemon=True)
    _keepalive_threads[lock_id] = t
    t.start()


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


def _terminate_poller_lock_holders(
    min_idle_seconds: float = 0.0,
    *,
    log_prefix: str = "[tg-lock]",
) -> int:
    """Terminate stale holders of TG_POLLER_LOCK_ID owned by th-tgpoller only."""
    import psycopg2

    db_url = _get_db_url()
    if not db_url:
        return 0

    lock_id = TG_POLLER_LOCK_ID
    owner_app = APP_NAME_TG_POLLER
    terminated = 0
    try:
        conn = psycopg2.connect(db_url, connect_timeout=10)
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
                            f"for lock {lock_id}"
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
        logger.error(f"{log_prefix} terminate_poller_lock_holders failed: {e}")
    return terminated


async def wait_for_poller_lock(lock_id: int, retry_seconds: int = 15) -> None:
    """Block until this process holds TG_POLLER_LOCK_ID."""
    import asyncio

    from app.deployment import is_railway

    if lock_id != TG_POLLER_LOCK_ID:
        raise ValueError(
            f"Telegram poller must use TG_POLLER_LOCK_ID ({TG_POLLER_LOCK_ID}), not {lock_id}"
        )

    attempts = 0
    break_after = 3 if is_railway() else 6

    while True:
        if await asyncio.to_thread(_try_acquire, TG_POLLER_LOCK_ID):
            _start_keepalive(TG_POLLER_LOCK_ID)
            return
        attempts += 1
        if attempts >= break_after:
            n = await asyncio.to_thread(
                _terminate_poller_lock_holders,
                log_prefix="[tg-lock]",
            )
            if n:
                logger.warning(
                    f"[tg-lock] reclaimed lock {TG_POLLER_LOCK_ID} — terminated {n} stale holder(s)"
                )
                attempts = 0
                continue
        await asyncio.sleep(retry_seconds)


def release_poller_lock(lock_id: int) -> None:
    _release(lock_id)


def holds_lock(lock_id: int) -> bool:
    return lock_id in _lock_conns


def try_acquire_persistent_lock(lock_id: int) -> bool:
    """Non-blocking acquire for single-runner jobs (e.g. Twitter poster).

    Telegram polling must use wait_for_poller_lock / TG_POLLER_LOCK_ID only.
    """
    if lock_id == TG_POLLER_LOCK_ID:
        if holds_lock(TG_POLLER_LOCK_ID):
            return True
        if _try_acquire(TG_POLLER_LOCK_ID):
            _start_keepalive(TG_POLLER_LOCK_ID)
            return True
        return False
    # Non-TG locks (e.g. TWITTER_POSTER_LOCK_ID) use generic acquire path.
    import psycopg2

    from app.executor_lock import NEON_LOCK_CONNECT_KWARGS

    db_url = _get_db_url()
    if not db_url:
        return False
    if holds_lock(lock_id):
        return True
    try:
        conn_kw = dict(NEON_LOCK_CONNECT_KWARGS)
        conn_kw["application_name"] = application_name_for_lock(lock_id)
        conn = psycopg2.connect(db_url, **conn_kw)
        conn.autocommit = True
        with conn.cursor() as cur:
            cur.execute("SELECT pg_try_advisory_lock(%s)", (lock_id,))
            acquired = bool(cur.fetchone()[0])
        if acquired:
            _lock_conns[lock_id] = conn
            _start_keepalive(lock_id)
            return True
        conn.close()
    except Exception:
        pass
    return False


async def describe_bot_token(token: str, label: str) -> dict:
    """getMe probe for admin diagnostics."""
    import httpx

    if not token:
        return {"label": label, "configured": False}
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.get(f"https://api.telegram.org/bot{token}/getMe")
            body = r.json()
        if r.status_code == 200 and body.get("ok"):
            me = body["result"]
            return {
                "label": label,
                "configured": True,
                "id": me.get("id"),
                "username": me.get("username"),
            }
        return {"label": label, "configured": True, "error": body.get("description", r.text[:200])}
    except Exception as e:
        return {"label": label, "configured": True, "error": str(e)}
