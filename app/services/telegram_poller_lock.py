"""
PostgreSQL advisory locks — one poller per bot token across all hosts/replicas.

Main crypto bot and forex bot use separate lock IDs so two different bots can
poll concurrently without fighting each other.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from typing import Dict, Optional

logger = logging.getLogger(__name__)

# Must fit PostgreSQL int32 — pg_locks.objid is 32-bit. Values >2^31-1 break
# lock queries (OID out of range) and prevented the bot from ever polling.
MAIN_POLLER_LOCK_ID = 708_110_002
FOREX_POLLER_LOCK_ID = 708_110_003
# X / Twitter auto-poster — single-runner across the web workers AND the bot
# companion process so exactly ONE process posts (no double tweets / rate bans).
TWITTER_POSTER_LOCK_ID = 708_110_005

KEEPALIVE_INTERVAL = 20
_instance_id = str(os.getpid())

_lock_conns: Dict[int, object] = {}
_keepalive_threads: Dict[int, threading.Thread] = {}


def _get_db_url() -> str:
    return os.environ.get("NEON_DATABASE_URL") or os.environ.get("DATABASE_URL", "") or ""


def _try_acquire(lock_id: int) -> bool:
    """Try to acquire lock_id. Fail-closed — never poll without a real lock."""
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

        conn = psycopg2.connect(
            db_url,
            connect_timeout=10,
            keepalives=1,
            keepalives_idle=30,
            keepalives_interval=10,
            keepalives_count=5,
        )
        conn.autocommit = True
        with conn.cursor() as cur:
            cur.execute("SELECT pg_try_advisory_lock(%s)", (lock_id,))
            acquired = bool(cur.fetchone()[0])

        if acquired:
            _lock_conns[lock_id] = conn
            logger.info(
                f"[tg-lock] acquired lock {lock_id} — PID {_instance_id} may poll"
            )
            return True

        conn.close()
        logger.info(f"[tg-lock] lock {lock_id} held elsewhere — waiting")
        return False
    except Exception as e:
        logger.error(f"[tg-lock] acquire {lock_id} failed: {e} — will retry (fail-closed)")
        return False


def _release(lock_id: int) -> None:
    conn = _lock_conns.pop(lock_id, None)
    if conn:
        try:
            conn.close()
        except Exception:
            pass
        logger.info(f"[tg-lock] released lock {lock_id} (PID {_instance_id})")


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
            logger.warning(f"[tg-lock] keepalive failed for {lock_id}: {e}")
            break


def _start_keepalive(lock_id: int) -> None:
    t = _keepalive_threads.get(lock_id)
    if t and t.is_alive():
        return
    t = threading.Thread(target=_keepalive_loop, args=(lock_id,), daemon=True)
    _keepalive_threads[lock_id] = t
    t.start()


def _terminate_other_lock_holders(lock_id: int, min_idle_seconds: float = 0.0) -> int:
    """Drop advisory-lock DB sessions held by other backends (e.g. stale Replit VM).

    ``min_idle_seconds`` guards against killing a *live sibling* holder: when set
    (> 0), a holder is only terminated if it has been idle (no query) for at least
    that long. A genuinely stale/zombie connection from a dead deploy stops being
    pinged, so its idle age grows without bound and crosses the threshold; a live
    holder that keeps its lock connection warm (the executor pings every few tens
    of seconds) never does. This is essential with ``GUNICORN_WORKERS>1`` — the
    HTTP worker would otherwise terminate the executor worker's lock connection on
    every claim attempt, thrashing the executor and halting all trade firing.
    """
    import psycopg2

    db_url = _get_db_url()
    if not db_url:
        return 0
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
                # Protect a live sibling: a holder that is an active client
                # session and was recently pinged (idle < threshold) is the
                # legitimate current lock owner — never terminate it.
                if min_idle_seconds > 0 and state != "gone" and idle_secs is not None:
                    if float(idle_secs) < float(min_idle_seconds):
                        logger.info(
                            f"[tg-lock] keeping live holder pid={pid} state={state} "
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
                            f"[tg-lock] terminated stale holder pid={pid} "
                            f"state={state} app={app!r} idle="
                            f"{(f'{float(idle_secs):.0f}s' if idle_secs is not None else 'n/a')} "
                            f"for lock {lock_id}"
                        )
                except Exception as te:
                    logger.warning(f"[tg-lock] could not terminate pid={pid}: {te}")
        conn.close()
    except Exception as e:
        logger.error(f"[tg-lock] terminate_other_lock_holders failed: {e}")
    return terminated


# Public alias for executor / other advisory-lock reclaim paths.
terminate_advisory_lock_holders = _terminate_other_lock_holders


async def wait_for_poller_lock(lock_id: int, retry_seconds: int = 15) -> None:
    """Block until this process holds lock_id.

    If another host (e.g. a zombie Replit deploy) holds the lock without polling,
    we terminate its DB session after a few waits so Railway can take over.
    """
    import asyncio

    from app.deployment import is_railway

    attempts = 0
    # Railway is production — reclaim faster when a ghost Replit holds the lock.
    break_after = 3 if is_railway() else 6

    while True:
        if await asyncio.to_thread(_try_acquire, lock_id):
            _start_keepalive(lock_id)
            return
        attempts += 1
        if attempts >= break_after:
            n = await asyncio.to_thread(_terminate_other_lock_holders, lock_id)
            if n:
                logger.warning(
                    f"[tg-lock] reclaimed lock {lock_id} — terminated {n} stale holder(s)"
                )
                attempts = 0
                continue
        await asyncio.sleep(retry_seconds)


def release_poller_lock(lock_id: int) -> None:
    _release(lock_id)


def holds_lock(lock_id: int) -> bool:
    return lock_id in _lock_conns


def try_acquire_persistent_lock(lock_id: int) -> bool:
    """Non-blocking advisory-lock acquire + keepalive — NEVER terminates others.

    Returns True if this process now holds ``lock_id`` (or already held it). Use
    for single-runner background jobs (e.g. the X auto-poster) where several
    processes contend and we just want one winner; losers retry later. Unlike
    ``wait_for_poller_lock``, this never kills the current holder, so a live
    sibling is never disrupted. The session-level lock auto-releases if this
    process dies, letting another contender take over.
    """
    if holds_lock(lock_id):
        return True
    if _try_acquire(lock_id):
        _start_keepalive(lock_id)
        return True
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
