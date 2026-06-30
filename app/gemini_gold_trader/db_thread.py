"""Run sync SQLAlchemy work off the gemini-gold asyncio event loop."""
from __future__ import annotations

import asyncio
from typing import Callable, TypeVar

from app.database import BgSessionLocal
from app.db_resilience import run_with_db_retry

T = TypeVar("T")


async def run_in_db_thread(fn: Callable[..., T], *args, **kwargs) -> T:
    return await asyncio.to_thread(fn, *args, **kwargs)


def with_db_session(fn: Callable[..., T]) -> Callable[..., T]:
    """Wrap ``fn(db, *args)`` with BgSessionLocal + transient Neon SSL retry."""

    def wrapper(*args, **kwargs):
        def _run() -> T:
            db = BgSessionLocal()
            try:
                return fn(db, *args, **kwargs)
            finally:
                db.close()

        return run_with_db_retry(_run, label="gemini-gold-db")

    return wrapper


async def run_with_db(fn: Callable[..., T], *args, **kwargs) -> T:
    """Open session, run sync ``fn(db, ...)``, close — with SSL retry.

    ``fn`` must accept ``db`` as its first parameter and must NOT already be
    wrapped with ``@with_db_session`` (that would double-inject ``db``).
    """
    return await run_in_db_thread(with_db_session(fn), *args, **kwargs)


async def db_commit(db) -> None:
    await run_in_db_thread(db.commit)


async def db_refresh(db, instance) -> None:
    await run_in_db_thread(db.refresh, instance)
