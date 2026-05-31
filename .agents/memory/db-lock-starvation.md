---
name: DB lock-queue starvation from startup DDL
description: Why portal API calls all hang after a restart, and the rules that prevent it
---

# DB lock-queue starvation on startup DDL

Startup migrations (`_ensure_tables` / `init_strategy_tables`) that run `ALTER TABLE`
or `CREATE INDEX` on a large, actively-read table (e.g. `strategy_executions`)
WITHOUT a `lock_timeout` can hang **every** read on that table — not just the DDL.

**Why:** PostgreSQL queues lock requests. A blocked `ALTER TABLE` waiting for an
`ACCESS EXCLUSIVE` lock sits at the head of the queue; every subsequent `SELECT`
(which only needs `ACCESS SHARE`) queues *behind* it and blocks until the DDL
either acquires+finishes or gives up. On a multi-worker portal sharing one Neon DB,
one worker's waiting migration freezes all API endpoints that touch that table →
the whole portal "won't load". Workers can then OOM/SIGKILL mid-ALTER, leaving the
lock held even longer.

**How to apply:**
- Every startup DDL statement must set a short wait cap. Use session-level
  `SET lock_timeout = '2s'` on the migration connection (applies to all following
  DDL) or `SET LOCAL lock_timeout = '2s'` inside each per-statement transaction.
- Never use `SET statement_timeout = 0` (unbounded) on the migration connection —
  bound it (e.g. `'30000'`).
- Guard every ALTER with existence checks (information_schema) + `ADD COLUMN IF NOT
  EXISTS` so a timed-out/skipped DDL is harmless and simply retries next restart.
- For request handlers that read these hot tables, also set a per-request
  `SET LOCAL statement_timeout` and run the sync DB work via
  `asyncio.wait_for(asyncio.to_thread(...))` so a slow query returns a clean 503
  instead of pinning the worker's event loop and stalling every other endpoint.
- Verify the flag name with `gunicorn --help` / PG docs before adding — it's
  `lock_timeout`, value as a quoted interval string.
