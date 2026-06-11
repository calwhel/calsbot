"""Canonical PostgreSQL advisory lock IDs — import from here; never hardcode."""

from __future__ import annotations

# Strategy executor (portal combined crypto+forex worker).
EXECUTOR_LOCK_ID = 708_110_004

# Telegram long-polling — one holder across all hosts/replicas.
TG_POLLER_LOCK_ID = 708_110_020
