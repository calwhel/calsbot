---
name: Strategy-id ownership / IDOR
description: Helpers that take a raw strategy_id and query by id alone must be ownership-gated before their output reaches an LLM prompt or response.
---

Several portal helpers (e.g. the per-strategy trade-stats analytics) query `StrategyExecution`/perf by `strategy_id` ALONE — no `user_id` filter. They are safe inside endpoints that already loaded the user's OWN strategy, but become an IDOR the moment a caller can pass an arbitrary `existing_strategy_id` (e.g. the chat-builder improve mode, webhook lookups, advisor).

**Rule:** before calling any id-only helper with a client-supplied strategy id, verify ownership:
`db.query(UserStrategy.id).filter(UserStrategy.id == sid, UserStrategy.user_id == user.id).first()` and skip/stub when not owned.

**Why:** the helper output is injected into the AI prompt and can be echoed back to the caller — an unguarded id leaks another user's trade performance cross-tenant.

**How to apply:** treat every endpoint that accepts a raw `strategy_id`/`existing_strategy_id` from the request body as untrusted; gate on `user_id` ownership before any id-keyed analytics read. Keep this pattern for future helpers that accept raw strategy ids.
