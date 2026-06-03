---
name: App DB is Neon, not the Replit built-in
description: The app's real data lives in an external Neon Postgres; the executeSql replit_database target queries a different, stale, unused DB. How to inspect the real data.
---

# App DB is Neon, not the Replit built-in Postgres

`app/config.py::get_database_url()` resolves to `NEON_DATABASE_URL` first ("Neon is
the single source of truth for ALL environments — all trade history, UIDs,
strategies, performance live in Neon"). There is effectively ONE database shared by
dev and prod (no separate prod replica for this app's data).

## Trap when debugging data
- The `executeSql` tool's `replit_database` target (both `development` AND
  `production`) hits the **Replit built-in Postgres**, which this app does NOT use.
  That DB is stale/legacy — querying it shows old schema (missing columns like
  `asset_class`, `ctrader_order_id`, `pips_pnl`) and frozen old data. Misleading.
- **To inspect the REAL data**, query Neon through the app itself:
  `python3 -c "from app.database import SessionLocal; ..."` (or `BgSessionLocal`)
  with `sqlalchemy.text(...)` and bound params. This uses the same connection the
  app uses, so schema + rows are the live truth.

**Why:** an investigation of a "missing notification" almost went wrong because the
Replit built-in DB showed zero live trades + an old schema; the real Neon DB had the
actual execution row that revealed the bug.

## Strategy status values when querying
`UserStrategy.status` ∈ draft | active | paper | paused | archived. The executor
runs the **active + paper (+ paused for live-position management)** set — and FOREX
strategies live almost entirely under `status='paper'`, NOT 'active'. Filtering
forex on just `active`/`paused` silently drops ~90% of them (found 5 instead of 40).
Always include `'paper'`. Strategy symbols are inside `config['universe']['symbols']`
(JSON), there is no `symbol` column on `user_strategies`.
