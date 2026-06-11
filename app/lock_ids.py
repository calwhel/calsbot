"""Canonical PostgreSQL advisory lock IDs — import from here; never hardcode."""

from __future__ import annotations

# Strategy executor (portal combined crypto+forex worker).
EXECUTOR_LOCK_ID = 708_110_004

# Telegram long-polling — one holder across all hosts/replicas.
TG_POLLER_LOCK_ID = 708_110_020

# cTrader OAuth refresh — pg_try_advisory_xact_lock(namespace, user_id).
# Serialises rotating refresh_token updates per user across gunicorn workers.
CTRADER_TOKEN_REFRESH_LOCK_NS = 708_110_021
