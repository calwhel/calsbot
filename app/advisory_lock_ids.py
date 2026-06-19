"""Central registry of PostgreSQL advisory lock IDs and connection application names.

Every subsystem that uses pg_advisory_lock must import its lock ID from here so
IDs cannot collide silently across modules. Pair each lock with a distinct
``application_name`` so zombie-reclaim logic only terminates backends from the
same subsystem.
"""

from __future__ import annotations

from app.lock_ids import EXECUTOR_LOCK_ID, TG_POLLER_LOCK_ID

# Portal misc one-off (strategy_portal_server startup probe).
PORTAL_MISC_LOCK_ID = 708_110_001

# Schema migration — must not share an ID with any long-lived poller/executor.
SCHEMA_MIGRATION_LOCK_ID = 708_110_002

# Dedicated forex/tradfi executor replica (EXECUTOR_ONLY=1).
FOREX_EXECUTOR_LOCK_ID = 708_110_012

# Telegram poller — single lock for main + forex bots (see app.lock_ids).
MAIN_POLLER_LOCK_ID = TG_POLLER_LOCK_ID
FOREX_POLLER_LOCK_ID = TG_POLLER_LOCK_ID

# X / Twitter auto-poster single-runner.
TWITTER_POSTER_LOCK_ID = 708_110_005

# Background price feeds (each its own lock).
FMP_POLL_LOCK_ID = 708_110_013
CTRADER_FEED_LOCK_ID = 708_110_014
METALS_SPOT_POLL_LOCK_ID = 708_110_015
CTRADER_TOKEN_REFRESH_OWNER_LOCK_ID = 708_110_022

# All lock IDs in use — handy for uniqueness audits / diagnostics.
ALL_ADVISORY_LOCK_IDS = frozenset(
    {
        PORTAL_MISC_LOCK_ID,
        SCHEMA_MIGRATION_LOCK_ID,
        EXECUTOR_LOCK_ID,
        FOREX_EXECUTOR_LOCK_ID,
        TG_POLLER_LOCK_ID,
        TWITTER_POSTER_LOCK_ID,
        FMP_POLL_LOCK_ID,
        CTRADER_FEED_LOCK_ID,
        METALS_SPOT_POLL_LOCK_ID,
        CTRADER_TOKEN_REFRESH_OWNER_LOCK_ID,
    }
)

# libpq application_name per subsystem (shown in pg_stat_activity).
APP_NAME_EXECUTOR = "th-executor"
APP_NAME_FOREX_EXECUTOR = "th-forex-executor"
APP_NAME_WEB = "th-web"
APP_NAME_TG_POLLER = "th-tgpoller"
APP_NAME_TG_POLLER_MAIN = APP_NAME_TG_POLLER
APP_NAME_TG_POLLER_FOREX = APP_NAME_TG_POLLER
APP_NAME_TWITTER_POSTER = "th-twitter-poster"
APP_NAME_SCHEMA_MIGRATION = "th-schema-migration"
APP_NAME_FMP_POLL = "th-fmp-poll"
APP_NAME_CTRADER_FEED = "th-ctrader-feed"
APP_NAME_METALS_SPOT = "th-metals-spot"
APP_NAME_CTRADER_TOKEN_REFRESH = "th-ctrader-token-refresh"

_LOCK_APP_NAMES: dict[int, str] = {
    EXECUTOR_LOCK_ID: APP_NAME_EXECUTOR,
    FOREX_EXECUTOR_LOCK_ID: APP_NAME_FOREX_EXECUTOR,
    TG_POLLER_LOCK_ID: APP_NAME_TG_POLLER,
    TWITTER_POSTER_LOCK_ID: APP_NAME_TWITTER_POSTER,
    SCHEMA_MIGRATION_LOCK_ID: APP_NAME_SCHEMA_MIGRATION,
    FMP_POLL_LOCK_ID: APP_NAME_FMP_POLL,
    CTRADER_FEED_LOCK_ID: APP_NAME_CTRADER_FEED,
    METALS_SPOT_POLL_LOCK_ID: APP_NAME_METALS_SPOT,
    CTRADER_TOKEN_REFRESH_OWNER_LOCK_ID: APP_NAME_CTRADER_TOKEN_REFRESH,
}


def application_name_for_lock(lock_id: int) -> str:
    """Return the canonical application_name for a lock-holding connection."""
    return _LOCK_APP_NAMES.get(lock_id, f"th-advisory-{lock_id}")
