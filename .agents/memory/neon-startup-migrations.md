---
name: Neon startup migrations
description: How to safely add ORM-mapped columns at startup on Neon PostgreSQL without blocking uvicorn or hanging on table locks.
---

## The rule
Always check `information_schema.columns` first (lock-free) and skip ALTER TABLE entirely if the column already exists. Only run ALTER TABLE on the first-deployment path, with `lock_timeout = '3s'` + retry loop.

**Why:** Three failure modes collide at startup:
1. Neon's default `statement_timeout` (~5s) cancels ALTER TABLE when the statement runs as a plain synchronous call inside an async uvicorn startup event.
2. `strategy_executions` is constantly read/written by the live executor. `ALTER TABLE ADD COLUMN` needs ACCESS EXCLUSIVE — even with `IF NOT EXISTS` — so it waits for all shared locks to clear. On a busy table this can take many seconds or fail with `LockNotAvailable`.
3. Running the synchronous `engine.connect()` + DDL directly in `@app.on_event("startup")` blocks the uvicorn event loop. Even with `asyncio.to_thread()`, uvicorn waits for the startup coroutine to finish before accepting requests — so a slow/retrying migration delays the entire server.
4. `SET statement_timeout = 0` in the session "fixes" the cancellation but then a waiting ALTER TABLE hangs forever if locks are never released, causing the portal to never start.

## How to apply
In `strategy_portal_server.py` `startup()`:
1. Wrap the migration in `asyncio.to_thread()` + `asyncio.wait_for()` so it runs in a thread (non-blocking event loop).
2. Inside the thread: query `information_schema.columns` first (no locks needed — always instant).
3. If all target columns exist → return immediately (fast path, <100ms).
4. If any column is missing (first deployment only): retry loop with `SET lock_timeout = '3s'` + `SET statement_timeout = '10s'` + `time.sleep(2)` between attempts, each attempt using its own connection.

In `app/strategy_models.py` `init_strategy_tables()`:
- Include all three target tables in the `existing_cols` query scope (not just `strategy_executions` and `strategy_marketplace`).
- Use `ADD COLUMN IF NOT EXISTS` (not bare `ADD COLUMN`) to make reruns idempotent.
- Add `SET statement_timeout = 0` at the start of the migration connection (this function runs in the background task, not in the startup event, so hanging is less catastrophic — it just delays background init).

## Files
- `strategy_portal_server.py` — `startup()` function, `_run_critical_migs()` inner function
- `app/strategy_models.py` — `init_strategy_tables()` function
