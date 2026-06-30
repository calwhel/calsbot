"""Run sync SQLAlchemy work off the gemini-gold asyncio event loop."""
from __future__ import annotations

import asyncio
from typing import Callable, TypeVar

from app.database import SessionLocal

T = TypeVar("T")


async def run_in_db_thread(fn: Callable[..., T], *args, **kwargs) -> T:
    return await asyncio.to_thread(fn, *args, **kwargs)


def with_db_session(fn: Callable[..., T]) -> Callable[..., T]:
    def wrapper(*args, **kwargs):
        db = SessionLocal()
        try:
            return fn(db, *args, **kwargs)
        finally:
            db.close()

    return wrapper


async def run_with_db(fn: Callable[..., T], *args, **kwargs) -> T:
    return await run_in_db_thread(with_db_session(fn), *args, **kwargs)


async def db_commit(db) -> None:
    await run_in_db_thread(db.commit)


async def db_refresh(db, instance) -> None:
    await run_in_db_thread(db.refresh, instance)
