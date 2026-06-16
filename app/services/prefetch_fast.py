"""Fast prefetch mode — hard timeouts, no provider backoff sleeps on hot paths."""
from __future__ import annotations

import contextvars
import os
from contextlib import asynccontextmanager
from typing import AsyncIterator

_PREFETCH_FAST = contextvars.ContextVar("prefetch_fast", default=False)

# Hard per-symbol prefetch ceiling — one slow provider must not hold the cycle.
PROVIDER_TIMEOUT_S = float(os.environ.get("EXECUTOR_PREFETCH_PROVIDER_TIMEOUT_S", "2.5"))
SYMBOL_BUDGET_S = float(os.environ.get("EXECUTOR_PREFETCH_SYMBOL_BUDGET_S", "2.5"))


def prefetch_fast_active() -> bool:
    return bool(_PREFETCH_FAST.get())


def provider_timeout_s(default: float) -> float:
    """Cap per-provider timeout during prefetch (5s default)."""
    if prefetch_fast_active():
        return min(default, PROVIDER_TIMEOUT_S)
    return default


@asynccontextmanager
async def prefetch_fast_context() -> AsyncIterator[None]:
    token = _PREFETCH_FAST.set(True)
    try:
        yield
    finally:
        _PREFETCH_FAST.reset(token)
