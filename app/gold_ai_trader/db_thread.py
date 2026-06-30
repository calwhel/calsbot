"""Run sync SQLAlchemy work off the gold-ai asyncio event loop."""
from __future__ import annotations

import asyncio
from typing import Callable, TypeVar

from app.database import SessionLocal

T = TypeVar("T")


async def run_in_db_thread(fn: Callable[..., T], *args, **kwargs) -> T:
    """Execute a sync callable in a worker thread."""
    return await asyncio.to_thread(fn, *args, **kwargs)


def with_db_session(fn: Callable[..., T]) -> Callable[..., T]:
    """Wrap ``fn(db, *args)`` with a fresh SessionLocal that is always closed."""

    def wrapper(*args, **kwargs):
        db = SessionLocal()
        try:
            return fn(db, *args, **kwargs)
        finally:
            db.close()

    return wrapper


async def run_with_db(fn: Callable[..., T], *args, **kwargs) -> T:
    """``await``-able helper: open session, run sync ``fn(db, ...)``, close.

    ``fn`` must accept ``db`` as its first parameter and must NOT already be
    wrapped with ``@with_db_session`` (that would double-inject ``db``).
    """
    return await run_in_db_thread(with_db_session(fn), *args, **kwargs)


async def db_commit(db) -> None:
    """Commit on a worker thread so a slow Neon round-trip cannot block the loop."""
    await run_in_db_thread(db.commit)


async def db_refresh(db, instance) -> None:
    await run_in_db_thread(db.refresh, instance)
