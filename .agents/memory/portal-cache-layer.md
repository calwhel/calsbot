---
name: Portal cache layer
description: How the in-memory cache is structured in strategy_portal_server.py and app/cache.py.
---

## Rule
All NEW caching in `strategy_portal_server.py` must use `from app.cache import get_cache, set_cache, invalidate_cache, invalidate_prefix`. The old `_CACHE` dict (still present) is used only by legacy endpoints — don't extend it.

**Why:** `app/cache.py` is thread-safe (uses a lock), handles TTL expiry correctly, and provides prefix-based invalidation. The old `_CACHE` dict has no locking.

**How to apply:**
- Cache check: `hit = get_cache(key); if hit is not None: return JSONResponse(hit)`
- Cache set: `set_cache(key, payload, ttl_seconds)`
- Invalidate on write: call `invalidate_prefix(f"api_strats_{uid}")` and/or `invalidate_prefix(f"api_mkt:{uid}")` after `db.commit()` in any write endpoint.
- Key schema: `api_mkt:{uid}:{sort}:{cat}:{pricing}:{search}:{ai}` (TTL 180s), `api_mkt_lb:{uid}:{period}:{limit}` (TTL 120s), `api_ldr:{uid}:{metric}` (TTL 120s), `analytics:{strategy_id}` (TTL 300s).
