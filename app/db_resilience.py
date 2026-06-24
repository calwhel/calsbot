"""Neon/Postgres transient connection helpers (no SQLAlchemy engine import)."""
from __future__ import annotations

import logging
import time
from typing import Callable, TypeVar

logger = logging.getLogger(__name__)

_TRANSIENT_DB_MARKERS = (
    "ssl connection has been closed",
    "ssl syscall error",
    "eof detected",
    "server closed the connection unexpectedly",
    "connection reset by peer",
    "connection not open",
    "could not receive data from server",
    "terminating connection",
    "connection timed out",
    "broken pipe",
)

_CONNECTION_ERROR_MARKERS = _TRANSIENT_DB_MARKERS + (
    "endpoint has been disabled",
    "connection to server",
    "could not connect",
    "connection refused",
    "database_waking",
    "password authentication failed",
)

T = TypeVar("T")


def is_transient_db_error(exc: BaseException | None) -> bool:
    """True when Neon/Postgres dropped an SSL session — safe to retry on a fresh conn."""
    if exc is None:
        return False
    try:
        from sqlalchemy.exc import InterfaceError, OperationalError, TimeoutError

        if isinstance(exc, (OperationalError, InterfaceError, TimeoutError)):
            exc = getattr(exc, "orig", exc) or exc
    except Exception:
        pass
    msg = str(exc).lower()
    return any(marker in msg for marker in _TRANSIENT_DB_MARKERS)


def is_db_connection_error(exc: BaseException | None) -> bool:
    """True when the database is unreachable (pool/endpoint down), not a logic error."""
    if exc is None:
        return False
    try:
        from sqlalchemy.exc import InterfaceError, OperationalError, TimeoutError

        if isinstance(exc, (OperationalError, InterfaceError, TimeoutError)):
            return True
    except Exception:
        pass
    msg = str(exc).lower()
    return any(marker in msg for marker in _CONNECTION_ERROR_MARKERS)


def run_with_db_retry(
    fn: Callable[[], T],
    *,
    max_attempts: int = 3,
    retry_delay: float = 0.5,
    label: str = "db",
) -> T:
    """Run a DB callable; retry on transient Neon SSL / connection drops."""
    last_exc: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            return fn()
        except Exception as exc:
            last_exc = exc
            if is_transient_db_error(exc) and attempt < max_attempts:
                logger.warning(
                    "%s: transient DB error (attempt %s/%s): %s",
                    label,
                    attempt,
                    max_attempts,
                    exc,
                )
                time.sleep(retry_delay * attempt)
                continue
            raise
    if last_exc is not None:
        raise last_exc
    raise RuntimeError(f"{label}: run_with_db_retry exhausted without result")
