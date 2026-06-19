"""Single-owner scheduled cTrader OAuth refresh (all workers/shards are consumers).

Exactly one process holds CTRADER_TOKEN_REFRESH_OWNER_LOCK_ID and runs the refresh
cycle. Per-user OAuth calls are serialised via pg_try_advisory_xact_lock in
refresh_user_ctrader_token — this module is the only scheduled initiator.
"""
from __future__ import annotations

import asyncio
import logging
import os
import threading
import time
from typing import Dict, Optional

from app.advisory_lock_ids import (
    APP_NAME_CTRADER_TOKEN_REFRESH,
    CTRADER_TOKEN_REFRESH_OWNER_LOCK_ID,
)
from app.executor_lock import build_lock_connection, log_executor_lock_keepalive_config

logger = logging.getLogger(__name__)

_POLL_LOCK_ID = CTRADER_TOKEN_REFRESH_OWNER_LOCK_ID
_LOOP_INTERVAL_S = float(os.environ.get("CTRADER_TOKEN_SCHEDULER_INTERVAL_S", str(3600)))
_owner_lock_conn = None
_owner_thread_id: Optional[int] = None
_scheduler_task: Optional[asyncio.Task] = None
_wake_event: Optional[asyncio.Event] = None


def is_token_refresh_owner() -> bool:
    return _owner_thread_id == threading.get_ident()


def wake_token_scheduler() -> None:
    ev = _wake_event
    if ev is not None:
        try:
            ev.call_soon_threadsafe(ev.set)
        except RuntimeError:
            ev.set()


def _release_owner_lock() -> None:
    global _owner_lock_conn, _owner_thread_id
    conn = _owner_lock_conn
    _owner_lock_conn = None
    _owner_thread_id = None
    if not conn:
        return
    try:
        cur = conn.cursor()
        cur.execute("SELECT pg_advisory_unlock(%s)", (_POLL_LOCK_ID,))
        cur.close()
    except Exception:
        pass
    try:
        conn.close()
    except Exception:
        pass


def _try_acquire_owner_lock() -> bool:
    global _owner_lock_conn, _owner_thread_id
    if _owner_lock_conn is not None:
        return _owner_thread_id == threading.get_ident()
    try:
        log_executor_lock_keepalive_config("ctrader_token_scheduler:owner_lock")
        conn = build_lock_connection(application_name=APP_NAME_CTRADER_TOKEN_REFRESH)
        cur = conn.cursor()
        cur.execute("SELECT pg_try_advisory_lock(%s)", (_POLL_LOCK_ID,))
        ok = bool(cur.fetchone()[0])
        cur.close()
        if not ok:
            conn.close()
            return False
        _owner_lock_conn = conn
        _owner_thread_id = threading.get_ident()
        logger.info("[ctrader-token] refresh owner lock acquired")
        return True
    except Exception as exc:
        logger.warning("[ctrader-token] owner lock acquire failed: %s", exc)
        return False


async def run_token_refresh_cycle(*, reason: str = "scheduled") -> dict:
    """Refresh tokens for linked users when expiry is within the early window."""
    from app.database import SessionLocal
    from app.models import UserPreference
    from app.services.ctrader_client import (
        _SCHEDULED_REFRESH_WHEN_REMAINING_S,
        _WARN_WHEN_REMAINING_S,
        _log_ctrader_token_startup,
        _token_seconds_remaining_from_prefs,
        audit_ctrader_credentials,
        is_refresh_denied,
        refresh_user_ctrader_token,
    )

    stats = {"linked_users": 0, "refreshed": 0, "failed": 0, "denied": 0, "reason": reason}
    db = SessionLocal()
    try:
        rows = (
            db.query(UserPreference)
            .filter(UserPreference.ctrader_refresh_token.isnot(None))
            .all()
        )
        stats["linked_users"] = len(rows)
        for prefs in rows:
            uid = int(prefs.user_id)
            audit = audit_ctrader_credentials(uid, prefs)
            if not audit.get("ok"):
                logger.warning(
                    "[ctrader-token] audit user=%s: %s",
                    uid,
                    audit.get("reason"),
                )
                stats["failed"] += 1
                continue
            if is_refresh_denied(uid):
                stats["denied"] += 1
                continue

            remaining = _token_seconds_remaining_from_prefs(prefs)
            at = (prefs.ctrader_access_token or "").strip()
            if at and remaining is not None:
                if remaining <= _WARN_WHEN_REMAINING_S:
                    logger.warning(
                        "[ctrader-token] access token expires in %.0fh user=%s — refresh scheduled",
                        remaining / 3600.0,
                        uid,
                    )
                elif remaining > _SCHEDULED_REFRESH_WHEN_REMAINING_S:
                    _log_ctrader_token_startup(uid, at)
                    continue
            elif at:
                _log_ctrader_token_startup(uid, at)

            new_at = await refresh_user_ctrader_token(uid, force=False)
            if new_at:
                stats["refreshed"] += 1
                try:
                    from app.services.ctrader_price_feed import notify_account_linked

                    notify_account_linked(uid)
                except Exception:
                    pass
            else:
                stats["failed"] += 1
    finally:
        db.close()

    logger.info("[ctrader-token] refresh cycle (%s): %s", reason, stats)
    return stats


async def _scheduler_loop() -> None:
    global _wake_event
    _wake_event = asyncio.Event()
    # Immediate cycle on startup so deploy picks up persisted tokens quickly.
    try:
        await run_token_refresh_cycle(reason="scheduler_start")
    except Exception as exc:
        logger.warning("[ctrader-token] startup cycle failed: %s", type(exc).__name__)

    while True:
        wake = _wake_event
        try:
            await asyncio.wait_for(wake.wait(), timeout=_LOOP_INTERVAL_S)
            wake.clear()
            reason = "scheduler_wake"
        except asyncio.TimeoutError:
            reason = "scheduled"
        try:
            await run_token_refresh_cycle(reason=reason)
        except Exception as exc:
            logger.warning("[ctrader-token] cycle failed: %s", type(exc).__name__)


def start_ctrader_token_scheduler() -> None:
    """Start the refresh-owner loop on this process (no-op if lock not acquired)."""
    global _scheduler_task

    if not _try_acquire_owner_lock():
        logger.info("[ctrader-token] refresh owner lock held elsewhere — consumer mode")
        return
    if _scheduler_task and not _scheduler_task.done():
        return

    async def _runner():
        try:
            await _scheduler_loop()
        finally:
            _release_owner_lock()

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        logger.warning("[ctrader-token] no running loop — scheduler not started")
        _release_owner_lock()
        return
    _scheduler_task = loop.create_task(_runner(), name="ctrader-token-scheduler")
    logger.info("[ctrader-token] scheduled refresh owner started (interval %.0fs)", _LOOP_INTERVAL_S)
