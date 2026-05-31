---
name: Response-object caching bug
description: Why caching a JSONResponse/Response object breaks under BaseHTTPMiddleware, and the safe pattern.
---

# Never cache a Response object — cache the payload

In `strategy_portal_server.py` the app uses Starlette `BaseHTTPMiddleware` for
`?uid=` auth. That middleware consumes a Response's `body_iterator` exactly once
when it streams the body downstream.

**Rule:** an endpoint must NOT store a `JSONResponse`/`Response` object in a cache
and re-return the same object. Cache the plain `dict`/`list` payload and build a
fresh `JSONResponse(payload)` on every return path (cache-hit, every stale
fallback, and the normal build).

**Why:** re-returning the same cached Response works on the first send but on
later sends the already-consumed `body_iterator` yields an EMPTY body while the
`Content-Length` header is still set → browser `ERR_CONTENT_LENGTH_MISMATCH`,
curl shows declared length but 0 bytes downloaded. This presented as the whole
portal "not loading data" intermittently (cache misses returned full body, cache
hits returned 0 bytes).

**How to apply:** when adding/reviewing any cached endpoint, confirm the cached
value is a dict/list/primitive (FastAPI re-serializes it fresh each return) and
never a Response object. Grep `_CACHE[...] =` / `set_cache(` for `JSONResponse`.
