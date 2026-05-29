"""Thread-safe in-memory TTL cache.

Each Gunicorn worker has its own in-process cache (no cross-worker sharing).
This is intentional — the cache is cheap to repopulate from Neon and
per-worker caching avoids any IPC complexity. On a 2-worker setup the
worst case is two slightly-divergent caches for one TTL window, which is
acceptable for leaderboard / marketplace data.
"""
import threading
import time
from typing import Any, Optional

_store: dict[str, tuple[Any, float]] = {}   # key → (value, expires_at monotonic)
_lock = threading.Lock()


def get_cache(key: str) -> Optional[Any]:
    """Return cached value if present and not expired, else None."""
    with _lock:
        entry = _store.get(key)
    if entry is None:
        return None
    value, expires_at = entry
    if time.monotonic() > expires_at:
        with _lock:
            _store.pop(key, None)
        return None
    return value


def set_cache(key: str, value: Any, ttl_seconds: int) -> None:
    """Store value with a TTL (seconds from now)."""
    with _lock:
        _store[key] = (value, time.monotonic() + ttl_seconds)


def invalidate_cache(key: str) -> None:
    """Remove a single key from the cache."""
    with _lock:
        _store.pop(key, None)


def invalidate_prefix(prefix: str) -> None:
    """Remove all keys that start with *prefix*."""
    with _lock:
        keys = [k for k in _store if k.startswith(prefix)]
        for k in keys:
            del _store[k]


def cache_size() -> int:
    """Return number of (possibly expired) entries — for diagnostics."""
    with _lock:
        return len(_store)
