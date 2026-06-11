"""Advisory-lock helpers for the dedicated cTrader feed process."""

from __future__ import annotations

import logging
import os

from app.advisory_lock_ids import APP_NAME_CTRADER_FEED, CTRADER_FEED_LOCK_ID
from app.executor_lock import (
    LOCK_RECONNECT_ATTEMPTS,
    LOCK_RECONNECT_DELAY_SECS,
    close_lock_connection,
    create_lock_connection,
    reconnect_lock_connection,
    try_acquire_lock,
)

logger = logging.getLogger(__name__)


def is_feed_only_process() -> bool:
    return os.getenv("CTRADER_FEED_ONLY", "").lower() in ("1", "true", "yes")


def remote_feed_enabled() -> bool:
    """Main portal reads ticks from Postgres; spot stream runs elsewhere."""
    return os.getenv("CTRADER_REMOTE_FEED", "").lower() in ("1", "true", "yes")


def _remote_ctrader_ticks_fresh(max_age_s: float = 60.0) -> bool:
    """Postgres tick store only — avoids importing ctrader_price_feed at startup."""
    try:
        from app.services.spot_price_store import snapshot

        snap = snapshot(max_age_s=max_age_s)
        return bool((snap.get("by_source") or {}).get("ctrader"))
    except Exception:
        return False


def feed_disabled_in_executor() -> bool:
    """Skip local cTrader socket in executor when remote ticks are healthy."""
    if remote_feed_enabled():
        if _remote_ctrader_ticks_fresh(max_age_s=60.0):
            return True
        # Remote feed configured but stale — allow local fallback in executor.
        return False
    return os.getenv("DISABLE_CTRADER_FEED_IN_EXECUTOR", "").lower() in (
        "1",
        "true",
        "yes",
    )


def reclaim_feed_lock(*, force: bool = False) -> bool:
    """Terminate stale feed lock holders (deploy handoff)."""
    if not force:
        return False
    try:
        from app.services.telegram_poller_lock import terminate_advisory_lock_holders

        n = terminate_advisory_lock_holders(
            CTRADER_FEED_LOCK_ID,
            min_idle_seconds=0.0,
            owner_app_prefix=APP_NAME_CTRADER_FEED,
            log_prefix="[ctrader-feed-lock]",
        )
        if n:
            logger.warning(
                "Reclaimed cTrader feed lock %s — terminated %s holder(s)",
                CTRADER_FEED_LOCK_ID,
                n,
            )
        return n > 0
    except Exception as exc:
        logger.debug("feed lock reclaim skipped: %s", exc)
        return False


def try_acquire_feed_lock():
    conn = create_lock_connection(APP_NAME_CTRADER_FEED)
    if try_acquire_lock(conn, CTRADER_FEED_LOCK_ID):
        return conn
    close_lock_connection(conn)
    return None


def reconnect_feed_lock(old_conn):
    return reconnect_lock_connection(
        old_conn,
        lock_id=CTRADER_FEED_LOCK_ID,
        max_attempts=LOCK_RECONNECT_ATTEMPTS,
        retry_delay=LOCK_RECONNECT_DELAY_SECS,
        application_name=APP_NAME_CTRADER_FEED,
    )
