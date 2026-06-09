"""Executor advisory-lock helpers — importable without loading strategy_portal_server."""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

EXECUTOR_LOCK_ID = 708_110_004


def is_standalone_executor() -> bool:
    return os.getenv("EXECUTOR_STANDALONE", "").lower() in ("1", "true", "yes")


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
