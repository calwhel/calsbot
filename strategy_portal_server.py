"""
Strategy Portal Server — Build Your Own Strategy Portal
Standalone FastAPI server on port 8080.
Run alongside the main bot (separate workflow).
"""
import os
import asyncio
import hmac
import hashlib
import secrets
import json
import time
import logging
import urllib.parse
import httpx
from datetime import datetime, timedelta
from typing import Optional, Dict, Tuple, List

from fastapi import FastAPI, Request, HTTPException, Query, Depends, BackgroundTasks
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, FileResponse, PlainTextResponse
from fastapi.templating import Jinja2Templates
from starlette.middleware.gzip import GZipMiddleware
from starlette.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from sqlalchemy import func
from app.models import User
from app.database import SessionLocal, engine
from app.strategy_models import (
    UserStrategy, StrategyExecution, StrategyPerformance,
    StrategyMarketplace, StrategyPortalSettings, PortalSubscription,
    PortalPayment, StrategyOffer, init_strategy_tables,
)
from app.strategy_marketplace_ext import (
    StrategyPurchase, StrategyRating, CreatorEarnings,
    EarningsTransaction, init_marketplace_ext_tables,
    calculate_creator_cut, calculate_platform_cut,
)
from app.social_models import init_social_tables
from app.deployment import (
    is_production_deploy,
    is_replit,
    is_railway,
    portal_features_free,
    payments_enabled,
    request_is_https,
    public_base_url,
    railway_app_hostname,
    railway_service_base_url,
    deploy_commit,
    google_auth_enabled,
)

# ─── Simple in-process TTL cache for public/read-heavy endpoints ─────────────
# Format: { key: (payload, expiry_timestamp) }
_CACHE: Dict[str, Tuple] = {}
# Structured cache module (thread-safe, used for new/fixed endpoints)
from app.cache import get_cache, set_cache, invalidate_cache, invalidate_prefix  # noqa: E402

from app.logging_safe import configure_safe_logging

configure_safe_logging(logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Strategy Portal", docs_url=None, redoc_url=None)
app.add_middleware(GZipMiddleware, minimum_size=500)


def _cached_json(payload, hit: bool, ttl_seconds: int | None = None, status_code: int = 200) -> JSONResponse:
    """Build a JSON response with cache diagnostics for Railway latency debugging."""
    headers = {"X-Cache": "HIT" if hit else "MISS"}
    if ttl_seconds is not None:
        headers["X-Cache-TTL"] = str(ttl_seconds)
    return JSONResponse(payload, status_code=status_code, headers=headers)


def _slim_strategy_config(cfg: dict) -> dict:
    """Card-view config slice — drops heavy entry_conditions/universe blobs."""
    if not cfg:
        return {}
    risk = cfg.get("risk") or {}
    exit_ = cfg.get("exit") or {}
    slim_risk = {
        k: risk[k] for k in (
            "leverage", "position_size_type", "position_size_pct",
            "position_size_usd", "position_size_lots",
        ) if risk.get(k) is not None
    }
    slim_exit = {
        k: exit_[k] for k in ("take_profit_pct", "stop_loss_pct")
        if exit_.get(k) is not None
    }
    return {
        "asset_class": cfg.get("asset_class") or cfg.get("_asset_class") or "crypto",
        "direction": cfg.get("direction"),
        "_locked": cfg.get("_locked"),
        "risk": slim_risk,
        "exit": slim_exit,
    }


def _html_error_page(status: int, title: str, message: str) -> str:
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{title} — TradeHub Markets</title>
  <style>
    *{{margin:0;padding:0;box-sizing:border-box}}
    body{{background:#0E0F11;color:#E8EAF0;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
         display:flex;align-items:center;justify-content:center;min-height:100vh;padding:24px}}
    .card{{background:#1A1C20;border:1px solid rgba(255,255,255,0.08);border-radius:16px;
           padding:40px 32px;max-width:440px;width:100%;text-align:center}}
    .logo{{font-size:28px;font-weight:700;color:#fff;letter-spacing:-0.5px;margin-bottom:4px}}
    .logo span{{color:#3FB68B}}
    .code{{font-size:48px;font-weight:800;color:rgba(255,255,255,0.12);margin:24px 0 8px}}
    h1{{font-size:20px;font-weight:600;margin-bottom:10px;color:#fff}}
    p{{font-size:14px;color:#8B909A;line-height:1.6;margin-bottom:28px}}
    a{{display:inline-block;background:#3FB68B;color:#0E0F11;font-weight:600;font-size:14px;
       padding:12px 28px;border-radius:10px;text-decoration:none}}
    a:hover{{opacity:.88}}
  </style>
</head>
<body>
  <div class="card">
    <div class="logo">TradeHub<span>.</span></div>
    <div class="code">{status}</div>
    <h1>{title}</h1>
    <p>{message}</p>
    <a href="/app">Back to app</a>
  </div>
</body>
</html>"""


from fastapi import Request as _Request
from fastapi.responses import HTMLResponse as _HTMLResponse
from fastapi.exceptions import HTTPException as _HTTPException
from starlette.exceptions import HTTPException as _StarletteHTTPException


@app.exception_handler(_StarletteHTTPException)
async def _http_exception_handler(request: _Request, exc: _StarletteHTTPException):
    # API routes always get JSON
    if request.url.path.startswith("/api/") or request.url.path.startswith("/webhook"):
        from fastapi.responses import JSONResponse
        return JSONResponse(status_code=exc.status_code, content={"detail": str(exc.detail)})
    # Page routes get a friendly HTML error
    if exc.status_code == 503:
        page = _html_error_page(503, "Back in a moment",
                                "Loading your account — this page will refresh automatically.")
        # Inject auto-retry script before </body>
        page = page.replace("</body>", "<script>setTimeout(()=>location.reload(),4000)</script></body>")
        return _HTMLResponse(page, status_code=503)
    if exc.status_code == 403:
        return _HTMLResponse(
            _html_error_page(403, "Access denied",
                             str(exc.detail) or "You don't have permission to view this page."),
            status_code=403,
        )
    if exc.status_code == 404:
        return _HTMLResponse(
            _html_error_page(404, "Page not found",
                             "The page you're looking for doesn't exist or has moved."),
            status_code=404,
        )
    return _HTMLResponse(
        _html_error_page(exc.status_code, "Something went wrong", str(exc.detail or "An unexpected error occurred.")),
        status_code=exc.status_code,
    )
# Mobile app (Expo Go) makes cross-origin requests. Open CORS for /api/*, but
# protected UID endpoints below must still present a signed session token.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"],
    allow_headers=["*"],
)


@app.middleware("http")
async def _mobile_uid_header_to_query(request: Request, call_next):
    """Translate the mobile-app `X-TradeHub-UID` header into a `?uid=` query
    param so all existing endpoints (which read `request.query_params.get("uid")`)
    keep working without per-route changes. Header is preferred over the legacy
    query param — UIDs in URLs leak into access logs and browser history.
    The web portal (cookie session) is unaffected.
    """
    header_uid = request.headers.get("X-TradeHub-UID") or request.headers.get("x-tradehub-uid")
    if header_uid and not request.query_params.get("uid"):
        existing_qs = request.scope.get("query_string", b"") or b""
        prefix = f"uid={header_uid.strip()}".encode()
        request.scope["query_string"] = (prefix + b"&" + existing_qs) if existing_qs else prefix
    return await call_next(request)


@app.on_event("startup")
async def _warm_neon_db():
    """Wake the Neon serverless DB on worker boot + keep it warm with a 4-min
    heartbeat. Neon scales to zero after ~5 min idle, and the cold-wake takes
    15-30 s — long enough that the mobile app's fetch times out and shows an
    infinite spinner on the very first login of a session. A `SELECT 1` ping
    every 4 minutes keeps the compute online with negligible cost.
    """
    async def _ping():
        try:
            start = time.monotonic()
            from sqlalchemy import text
            from app.database import SessionLocal as _SL
            def _exec():
                db = _SL()
                try:
                    db.execute(text("SELECT 1"))
                    db.commit()
                finally:
                    db.close()
            await asyncio.to_thread(_exec)
            elapsed_ms = int((time.monotonic() - start) * 1000)
            logger.info("[neon-keepwarm] ping ok in %sms; next ping in 240s", elapsed_ms)
        except Exception as exc:
            logger.warning(f"[neon-keepwarm] ping failed: {exc}")

    # Initial wake (fire-and-forget so startup doesn't block worker readiness)
    asyncio.create_task(_ping())

    async def _loop():
        while True:
            await asyncio.sleep(240)  # 4 min — under Neon's 5-min idle cutoff
            await _ping()
    asyncio.create_task(_loop())
    logger.info("[neon-keepwarm] heartbeat scheduled every 240s")

    async def _backfill_pips():
        """One-time backfill: compute pips_pnl for closed forex/metals executions
        that pre-date the pips_pnl column addition. Uses a single bulk SQL UPDATE
        (no Python loop) to avoid DB contention at startup."""
        await asyncio.sleep(60)  # wait until all workers are stable
        try:
            from app.database import SessionLocal as _SL
            from sqlalchemy import text as _t
            def _run():
                db = _SL()
                try:
                    # Check if there's anything to backfill first
                    count = db.execute(_t(
                        "SELECT COUNT(*) FROM strategy_executions "
                        "WHERE asset_class IN ('forex','metals','commodity','index') "
                        "  AND outcome <> 'OPEN' AND pips_pnl IS NULL "
                        "  AND entry_price IS NOT NULL AND exit_price IS NOT NULL"
                    )).scalar()
                    if not count:
                        logger.info("[pips-backfill] nothing to backfill")
                        return
                    # Single bulk UPDATE using SQL CASE for pip_size logic
                    result = db.execute(_t("""
                        UPDATE strategy_executions
                        SET pips_pnl = ROUND(CAST(
                            CASE
                                -- Historical backfill ONLY (rows with pips_pnl IS NULL,
                                -- pre-dating the column). Intentionally kept at 0.01: the
                                -- era these gold trades closed in used pip=$0.01, so this
                                -- keeps old gold rows consistent with each other. LIVE
                                -- closes now use $0.10 (forex_engine.pip_size) — do NOT
                                -- "align" this to 0.10 or historical stats shift 10x.
                                WHEN UPPER(symbol) IN ('XAUUSD') THEN
                                    CASE WHEN direction='LONG'
                                         THEN (exit_price - entry_price) / 0.01
                                         ELSE (entry_price - exit_price) / 0.01 END
                                WHEN UPPER(symbol) IN ('XAGUSD','CLUSD','NGUSD') THEN
                                    CASE WHEN direction='LONG'
                                         THEN (exit_price - entry_price) / 0.001
                                         ELSE (entry_price - exit_price) / 0.001 END
                                WHEN UPPER(symbol) LIKE '%JPY%' THEN
                                    CASE WHEN direction='LONG'
                                         THEN (exit_price - entry_price) / 0.01
                                         ELSE (entry_price - exit_price) / 0.01 END
                                ELSE
                                    CASE WHEN direction='LONG'
                                         THEN (exit_price - entry_price) / 0.0001
                                         ELSE (entry_price - exit_price) / 0.0001 END
                            END AS NUMERIC), 1)
                        WHERE asset_class IN ('forex','metals','commodity','index')
                          AND outcome <> 'OPEN'
                          AND pips_pnl IS NULL
                          AND entry_price IS NOT NULL
                          AND exit_price IS NOT NULL
                    """))
                    db.commit()
                    updated = result.rowcount
                    logger.info(f"[pips-backfill] bulk updated {updated} rows")
                    if updated == 0:
                        return
                    # Get affected strategy ids and recompute performance
                    sids = db.execute(_t(
                        "SELECT DISTINCT strategy_id FROM strategy_executions "
                        "WHERE asset_class IN ('forex','metals','commodity','index') "
                        "  AND outcome <> 'OPEN' AND pips_pnl IS NOT NULL"
                    )).scalars().all()
                    db.close()
                    from app.services.strategy_executor import _update_performance
                    for sid in sids:
                        try:
                            db2 = _SL()
                            _update_performance(sid, db2)
                            db2.close()
                        except Exception as pe:
                            logger.warning(f"[pips-backfill] perf update sid={sid}: {pe}")
                    logger.info(f"[pips-backfill] recomputed perf for {len(sids)} strategies")
                except Exception as inner:
                    db.close()
                    raise inner
            await asyncio.to_thread(_run)
        except Exception as e:
            logger.warning(f"[pips-backfill] failed: {e}")

    asyncio.create_task(_backfill_pips())


templates = Jinja2Templates(directory="app/templates")

@app.middleware("http")
async def redirect_www_and_log_500(request: Request, call_next):
    # Do NOT redirect www → apex. Apex DNS often still points at a dead host
    # (Replit/GCP) while www CNAME targets Railway — that redirect broke the site.
    # Optional CANONICAL_HOST=www.example.com redirects bare apex → www when both
    # domains are attached in Railway.
    host = (request.headers.get("host") or "").split(":")[0].lower()
    canonical = (os.getenv("CANONICAL_HOST") or "").strip().lower()
    if canonical and canonical.startswith("www.") and host == canonical[4:]:
        url = str(request.url).replace(f"://{host}", f"://{canonical}", 1)
        return RedirectResponse(url=url, status_code=301)
    try:
        return await call_next(request)
    except Exception as exc:
        import traceback as _tb
        logger.error(
            "[500] %s %s — %s: %s\n%s",
            request.method,
            request.url.path,
            type(exc).__name__,
            exc,
            _tb.format_exc(),
        )
        raise

# ─── Session cookie helpers (HMAC-signed, no extra deps) ──────────────────────
def _load_cookie_secret() -> str:
    """Load the HMAC secret without killing Railway workers at import time.

    Railway deployments migrated from Replit commonly have SESSION_SECRET
    rather than SECRET_KEY. Prefer SECRET_KEY, accept SESSION_SECRET, and keep
    an emergency boot-only fallback so the app serves traffic while operators
    add a proper stable secret.
    """
    value = (os.getenv("SECRET_KEY") or os.getenv("SESSION_SECRET") or "").strip()
    if value:
        if not os.getenv("SECRET_KEY") and os.getenv("SESSION_SECRET"):
            logger.warning("SECRET_KEY missing; using SESSION_SECRET for portal session signing")
        return value

    db_seed = (
        os.getenv("NEON_DATABASE_URL")
        or os.getenv("DATABASE_URL")
        or os.getenv("RAILWAY_DATABASE_URL")
        or ""
    ).strip()
    if db_seed:
        logger.critical(
            "SECRET_KEY/SESSION_SECRET missing; deriving temporary session signer from DB URL. "
            "Set SECRET_KEY in Railway for stable sessions."
        )
        return hashlib.sha256(f"tradehub-session:{db_seed}".encode()).hexdigest()

    logger.critical(
        "SECRET_KEY/SESSION_SECRET missing and no DB URL fallback is available; "
        "using per-process session signer. Set SECRET_KEY in Railway immediately."
    )
    return secrets.token_urlsafe(32)


_COOKIE_SECRET = _load_cookie_secret()
_COOKIE_NAME   = "th_session"
_COOKIE_MAX_AGE = 60 * 60 * 24 * 30  # 30 days


def _make_token(uid: str) -> str:
    sig = hmac.new(_COOKIE_SECRET.encode(), uid.encode(), hashlib.sha256).hexdigest()[:20]
    return f"{uid}:{sig}"


def _verify_token(token: str) -> Optional[str]:
    if not token or ":" not in token:
        return None
    uid, sig = token.rsplit(":", 1)
    expected = hmac.new(_COOKIE_SECRET.encode(), uid.encode(), hashlib.sha256).hexdigest()[:20]
    if hmac.compare_digest(sig, expected):
        return (uid or "").strip().upper() or None
    return None


def _get_session_uid(request: Request) -> Optional[str]:
    token = request.cookies.get(_COOKIE_NAME)
    return _verify_token(token) if token else None


def _get_request_token_uid(request: Request) -> Optional[str]:
    """Return the UID bound to a mobile/API token, if one was supplied."""
    auth = request.headers.get("Authorization", "")
    token = ""
    if auth.lower().startswith("bearer "):
        token = auth[7:].strip()
    token = token or request.headers.get("X-TradeHub-Session", "").strip()
    return _verify_token(token) if token else None


def _requested_uid(request: Request) -> Optional[str]:
    uid = request.query_params.get("uid") or request.headers.get("X-TradeHub-UID") or request.headers.get("x-tradehub-uid")
    uid = (uid or "").strip().upper()
    return uid or None


def _uid_auth_is_legacy_allowed() -> bool:
    return os.getenv("ALLOW_LEGACY_UID_API_AUTH", "").lower() in ("1", "true", "yes")


def _required_env_secret(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        logger.error("%s is not configured; protected endpoint is unavailable", name)
        raise HTTPException(status_code=503, detail=f"{name} is not configured")
    return value


def _require_admin_secret(secret: str):
    expected = _required_env_secret("ADMIN_SECRET")
    if not hmac.compare_digest(secret or "", expected):
        raise HTTPException(status_code=403, detail="Forbidden")


def _require_admin_bearer(request: Request):
    secret = request.headers.get("Authorization", "").replace("Bearer ", "", 1).strip()
    _require_admin_secret(secret)


@app.middleware("http")
async def _require_bound_uid_for_api(request: Request, call_next):
    """Require UID-bearing API calls to prove ownership of that UID.

    Legacy clients can temporarily opt back into UID-only auth with
    ALLOW_LEGACY_UID_API_AUTH=1 while they roll out Bearer/X-TradeHub-Session.
    """
    if request.method == "OPTIONS":
        return await call_next(request)

    path = request.url.path
    uid = _requested_uid(request)
    public_prefixes = (
        "/api/public/",
        "/api/mobile/login",
        "/api/webhooks/tv/",
        "/api/ctrader/callback",
        "/api/ctrader/oauth-progress",
        "/api/markets/catalog",
        "/api/ping",
    )
    if path.startswith("/api/") and uid and not path.startswith(public_prefixes):
        if not _uid_auth_is_legacy_allowed():
            session_uid = _get_session_uid(request)
            token_uid = _get_request_token_uid(request)
            allowed = {
                (session_uid or "").strip().upper(),
                (token_uid or "").strip().upper(),
            }
            allowed.discard("")
            if uid not in allowed:
                return JSONResponse(
                    {"detail": "A signed session token is required for this UID."},
                    status_code=401,
                )
    return await call_next(request)


def _set_session(response, uid: str, request: Request = None):
    uid = (uid or "").strip().upper()
    _secure = request_is_https(request)
    response.set_cookie(
        key=_COOKIE_NAME,
        value=_make_token(uid),
        max_age=_COOKIE_MAX_AGE,
        httponly=True,
        samesite="lax",
        secure=_secure,
    )


def _safe_next(value: Optional[str], default: str = "/app") -> str:
    """Return ``value`` if it's a safe internal path, else ``default``.

    A safe path is a relative URL that begins with a single ``/``, doesn't
    start with ``//`` (which would be protocol-relative → external host),
    contains no scheme, no whitespace/CR/LF, and is reasonably short. This
    keeps post-login redirects from being abused as an open-redirect vector.
    """
    if not value or not isinstance(value, str):
        return default
    v = value.strip()
    if not v.startswith("/") or v.startswith("//") or v.startswith("/\\"):
        return default
    if len(v) > 512:
        return default
    if any(c in v for c in ("\r", "\n", "\t", " ")):
        return default
    # Disallow scheme injection like "/http://..." paths
    low = v.lower()
    if "://" in low or low.startswith("/javascript:") or low.startswith("/data:"):
        return default
    return v


def get_db():
    from app.database import SessionLocal
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ── User-by-UID cache ────────────────────────────────────────────────────
# In-memory cache of (user.id, telegram_id, banned, is_admin, …) keyed on uid
# with a 30 s TTL. This lookup fires on EVERY authenticated request — under
# Neon load it was the #1 source of "Internal Server Error" 500s because the
# 20s statement_timeout was being hit on cold-wake. Caching the row in-memory
# means subsequent page loads bypass the DB entirely for 30s, so the user
# never sees a 500 from a transient DB blip while their session is hot.
# The cached row is a thin `_CachedUser` namespace, NOT the SQLAlchemy
# instance (instances can't safely cross sessions). Callers that need to
# mutate the user must do a fresh fetch.
import threading as _thr
import time as _time

_USER_CACHE: dict[str, tuple[float, "object"]] = {}
_USER_CACHE_LOCK = _thr.Lock()
_USER_CACHE_TTL = 120.0  # seconds — keep for 2 min to cut DB pressure

class _CachedUser:
    """Detached snapshot of a User row safe to share across requests."""
    __slots__ = ("id", "uid", "telegram_id", "email", "username", "first_name",
                 "is_admin", "banned", "approved", "grandfathered",
                 "subscription_end", "subscription_type", "trial_started_at",
                 "trial_ends_at", "trial_used", "referral_code", "referred_by",
                 "auth_provider", "email_verified", "google_id", "apple_id",
                 "created_at", "admin_notes", "nowpayments_subscription_id",
                 "password_hash")
    def __init__(self, u):
        for f in self.__slots__:
            setattr(self, f, getattr(u, f, None))


def _cache_user(u) -> "_CachedUser":
    cu = _CachedUser(u)
    with _USER_CACHE_LOCK:
        _USER_CACHE[u.uid] = (_time.time(), cu)
    return cu


def _user_from_cache(uid: str, allow_stale: bool = False):
    """Return cached user. allow_stale=True returns expired entries too (stale-while-revalidate)."""
    with _USER_CACHE_LOCK:
        entry = _USER_CACHE.get(uid)
        if not entry:
            return None
        ts, cu = entry
        if _time.time() - ts > _USER_CACHE_TTL:
            if allow_stale:
                return cu  # stale but usable when DB is down
            _USER_CACHE.pop(uid, None)
            return None
        return cu


def _invalidate_user_cache(uid: str | None = None):
    with _USER_CACHE_LOCK:
        if uid:
            _USER_CACHE.pop(uid, None)
        else:
            _USER_CACHE.clear()


def _get_user_by_uid(uid: str, db: Session):
    # Delegate to the resilient version — retries on statement_timeout so
    # every call site gets the retry fix without individual changes.
    return _get_user_by_uid_safe(uid, db)


def _get_user_by_uid_safe(uid: str, db: Session):
    """Resilient user lookup that survives transient `QueryCanceled` /
    `OperationalError` failures from Postgres' statement_timeout.

    The user-fetch query is on the heavy-traffic path of every authenticated
    endpoint and occasionally times out under load. This helper retries once
    with a fresh session before giving up, mirroring the pattern used in
    ``app/services/strategy_executor.py``.

    Returns the user row, or ``None`` if the UID is unknown after retry.
    Raises ``HTTPException(503)`` only when the database is genuinely
    unreachable so the client gets a clean error instead of a hung worker.
    """
    uid = (uid or "").strip().upper()
    from app.models import User
    from app.database import SessionLocal as _SL
    from sqlalchemy.exc import OperationalError as _SAOperationalError
    # Hot-path cache hit — bypasses DB entirely for up to 2 min after first load.
    cached = _user_from_cache(uid)
    if cached is not None:
        return cached
    try:
        u = db.query(User).filter(User.uid == uid).first()
        if u is not None:
            return _cache_user(u)
        return None
    except _SAOperationalError as exc:
        # DB timeout — roll back poisoned session, try once on a fresh connection.
        try:
            db.rollback()
        except Exception:
            pass
        logger.warning(f"[uid lookup] transient DB error, retrying on fresh session: {exc}")
        fresh = _SL()
        try:
            u = fresh.query(User).filter(User.uid == uid).first()
            if u is not None:
                return _cache_user(u)
            return None
        except _SAOperationalError as exc_retry:
            logger.warning(f"[uid lookup] retry also timed out: {exc_retry}")
            # Stale-while-revalidate: return cached user (even if expired) so
            # the worker doesn't block and the user gets a fast response.
            stale = _user_from_cache(uid, allow_stale=True)
            if stale is not None:
                logger.info(f"[uid lookup] serving stale cache for {uid}")
                return stale
            raise HTTPException(
                status_code=503,
                detail="Database is busy — please try again in a moment.",
            )
        finally:
            try:
                fresh.close()
            except Exception:
                pass


# ── Portal subscription helpers ────────────────────────────
FREE_CHAT_LIMIT = 10   # AI messages per month on free tier


def _get_portal_sub(user_id: int, db: Session):
    """Return (and auto-create) the PortalSubscription row for a user."""
    from app.strategy_models import PortalSubscription
    sub = db.query(PortalSubscription).filter(PortalSubscription.user_id == user_id).first()
    if not sub:
        sub = PortalSubscription(user_id=user_id, tier="free")
        db.add(sub)
        db.commit()
        db.refresh(sub)
    return sub


def _is_portal_pro(sub) -> bool:
    if portal_features_free():
        return True
    if not sub or (getattr(sub, "tier", "") or "").lower() != "pro":
        return False
    end = getattr(sub, "subscription_end", None)
    return end is None or end > datetime.utcnow()


def _chat_calls_info(sub, db: Session, user=None):
    """Check / reset monthly counter. Returns (allowed, used, limit, is_pro).

    Portal admins (``user.is_admin``) are treated as Pro for quota purposes —
    same intent as every other Pro-gated endpoint in this file."""
    now = datetime.utcnow()
    if sub.chat_calls_reset_at is None or (now - sub.chat_calls_reset_at).days >= 30:
        sub.chat_calls_used = 0
        sub.chat_calls_reset_at = now
        db.commit()
    pro = _is_portal_pro(sub) or bool(getattr(user, "is_admin", False))
    if pro:
        return True, sub.chat_calls_used, -1, True
    allowed = sub.chat_calls_used < FREE_CHAT_LIMIT
    return allowed, sub.chat_calls_used, FREE_CHAT_LIMIT, False


# ── Password helpers (no external deps) ──────────────────────────────────────
def _hash_password(password: str) -> str:
    salt = secrets.token_hex(16)
    h = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), 200_000)
    return f"{salt}:{h.hex()}"


def _verify_password(password: str, stored: str) -> bool:
    try:
        salt, h = stored.split(":", 1)
        h2 = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), 200_000)
        return secrets.compare_digest(h2.hex(), h)
    except Exception:
        return False


# ── Google OAuth config ───────────────────────────────────────────────────────
_GOOGLE_CLIENT_ID     = os.getenv("GOOGLE_CLIENT_ID", "")
_GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET", "")
_GOOGLE_AUTH_URL      = "https://accounts.google.com/o/oauth2/v2/auth"
_GOOGLE_TOKEN_URL     = "https://oauth2.googleapis.com/token"
_GOOGLE_USERINFO_URL  = "https://www.googleapis.com/oauth2/v2/userinfo"
_GOOGLE_SCOPE         = "openid email profile"


def _google_enabled() -> bool:
    return google_auth_enabled()


def _google_redirect_uri(request: Request) -> str:
    base = str(request.base_url).rstrip("/")
    return f"{base}/auth/google/callback"


# ── In-memory OTP store: email → (otp_code, expires_at) ─────────────────────
_otp_store: Dict[str, Tuple[str, datetime]] = {}
_OTP_TTL_MINUTES = 10


def _generate_otp() -> str:
    return str(secrets.randbelow(900000) + 100000)  # 6-digit


def _otp_valid(email: str, code: str) -> bool:
    entry = _otp_store.get(email.lower())
    if not entry:
        return False
    stored_code, expires_at = entry
    if datetime.utcnow() > expires_at:
        _otp_store.pop(email.lower(), None)
        return False
    return secrets.compare_digest(stored_code, code.strip())


async def _tg_send_msg(chat_id: str, text: str):
    """Generic helper — send any HTML message to a Telegram chat."""
    import httpx
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not token:
        return
    async with httpx.AsyncClient(timeout=10.0) as client:
        await client.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json={"chat_id": chat_id, "text": text, "parse_mode": "HTML"},
        )


async def _notify_admin_forex_connect(user_id: int, name: str, uname: str, tg_id: str, account_id: str):
    """Send admin (@bu11dogg) a Telegram message with Approve/Deny buttons when a user connects cTrader.
    Uses FOREX_BOT_TOKEN so the inline-button callbacks route to the dedicated forex bot."""
    from app.config import settings
    admin_chat = getattr(settings, "OWNER_TELEGRAM_ID", None)
    # Prefer the dedicated forex bot token so callbacks go to @TradehubStrategyBot
    token = os.getenv("FOREX_BOT_TOKEN") or os.getenv("TELEGRAM_BOT_TOKEN")
    if not admin_chat or not token:
        return
    mention = f"@{uname}" if uname else (f"TG {tg_id}" if tg_id else f"uid={user_id}")
    text = (
        f"<b>🔗 cTrader Connected — Forex Approval Needed</b>\n\n"
        f"User:        {name} ({mention})\n"
        f"cTrader Acct: <code>{account_id}</code>\n"
        f"DB user_id:  <code>{user_id}</code>\n\n"
        f"<i>Approve to enable live forex trading for this user.</i>"
    )
    keyboard = {
        "inline_keyboard": [[
            {"text": "✅ Approve", "callback_data": f"forex_approve:{user_id}"},
            {"text": "❌ Deny",    "callback_data": f"forex_deny:{user_id}"},
        ]]
    }
    try:
        import json as _json
        async with httpx.AsyncClient(timeout=10.0) as client:
            await client.post(
                f"https://api.telegram.org/bot{token}/sendMessage",
                json={
                    "chat_id": admin_chat,
                    "text": text,
                    "parse_mode": "HTML",
                    "reply_markup": keyboard,
                },
            )
    except Exception as e:
        logger.warning(f"[forex-connect notify] {e}")


async def _notify_admin_go_live(tg_id: str, name: str, uname: str, uid: str, cfg: dict):
    """Alert admin via Telegram when a user promotes a strategy to live."""
    from app.config import settings
    from app.database import SessionLocal
    from app.models import UserPreference
    admin_id = getattr(settings, "OWNER_TELEGRAM_ID", None)
    if not admin_id:
        return
    mention = f"@{uname}" if uname else f"ID {tg_id}"
    desc    = cfg.get("description", "")[:200]

    # Fetch Bitunix UID from preferences
    bitunix_uid = "NOT SET"
    try:
        db2 = SessionLocal()
        prefs = db2.query(UserPreference).filter(UserPreference.user_id == user.id).first()
        if prefs and getattr(prefs, "bitunix_uid", None):
            bitunix_uid = prefs.bitunix_uid
        db2.close()
    except Exception:
        pass

    text = (
        f"<b>Strategy Portal — Go Live Request</b>\n\n"
        f"User:        {name} ({mention})\n"
        f"TradeHub UID: <code>{uid}</code>\n"
        f"Telegram ID: <code>{tg_id}</code>\n"
        f"Bitunix UID: <code>{bitunix_uid}</code>\n\n"
        f"Strategy:  <b>{strategy.name}</b>\n"
        f"Details:   {desc}\n\n"
        f"<i>User has promoted their strategy to LIVE mode. "
        f"Add them as a copy trader on Bitunix using the UID above.</i>"
    )
    try:
        await _tg_send_msg(str(admin_id), text)
    except Exception as e:
        import logging
        logging.getLogger(__name__).warning(f"Admin go-live notify failed: {e}")


async def _notify_admin_referral(new_name: str, new_uid: str, referrer_name: str, referrer_uid: str):
    """Alert admin via Telegram when a new user signs up through a referral link."""
    from app.config import settings
    admin_id = getattr(settings, "OWNER_TELEGRAM_ID", None)
    if not admin_id:
        return
    text = (
        f"<b>🎉 New Referral Sign-Up!</b>\n\n"
        f"New user:    <b>{new_name}</b>\n"
        f"UID:         <code>{new_uid}</code>\n\n"
        f"Referred by: <b>{referrer_name}</b>\n"
        f"Referrer UID: <code>{referrer_uid}</code>\n\n"
        f"<i>Tier will show as Free until they upgrade to Pro.</i>"
    )
    try:
        await _tg_send_msg(str(admin_id), text)
    except Exception as e:
        _log.warning(f"Admin referral notify failed: {e}")


async def _notify_admin_uid_connected(user_name: str, user_uid: str, bitunix_uid: str):
    """Alert admin via Telegram when a user saves their Bitunix UID."""
    from app.config import settings
    admin_id = getattr(settings, "OWNER_TELEGRAM_ID", None)
    if not admin_id:
        return
    text = (
        f"<b>🔗 Bitunix UID Connected</b>\n\n"
        f"User:        <b>{user_name}</b>\n"
        f"TradeHub UID: <code>{user_uid}</code>\n"
        f"Bitunix UID: <code>{bitunix_uid}</code>\n\n"
        f"<i>Add them as a copy trader on Bitunix using the UID above.</i>"
    )
    try:
        await _tg_send_msg(str(admin_id), text)
    except Exception as e:
        _log.warning(f"Admin UID notify failed: {e}")


async def _notify_admin_pro_referral(new_name: str, new_uid: str, referrer_name: str, referrer_uid: str):
    """Alert admin when a referred user upgrades to Pro ($10 credited)."""
    from app.config import settings
    admin_id = getattr(settings, "OWNER_TELEGRAM_ID", None)
    if not admin_id:
        return
    text = (
        f"<b>💰 Referral Pro Upgrade — $10 Earned!</b>\n\n"
        f"Pro user:    <b>{new_name}</b> (<code>{new_uid}</code>)\n"
        f"Referred by: <b>{referrer_name}</b> (<code>{referrer_uid}</code>)\n\n"
        f"<i>$10 added to referrer's pending payout balance.</i>"
    )
    try:
        await _tg_send_msg(str(admin_id), text)
    except Exception as e:
        _log.warning(f"Admin pro-referral notify failed: {e}")


async def _send_otp_via_telegram(telegram_id: str, otp: str, name: str):
    """Send OTP code to user's Telegram chat."""
    import httpx
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not token:
        raise ValueError("Bot token not configured")
    display_name = name or "there"
    text = (
        f"🔐 <b>TradeHub Login Code</b>\n\n"
        f"Hi {display_name}! Here's your one-time sign-in code:\n\n"
        f"<code>{otp}</code>\n\n"
        f"⏱ Expires in <b>{_OTP_TTL_MINUTES} minutes</b>. "
        f"Don't share this with anyone."
    )
    async with httpx.AsyncClient(timeout=10.0) as client:
        r = await client.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json={"chat_id": telegram_id, "text": text, "parse_mode": "HTML"},
        )
        if r.status_code != 200:
            raise ValueError(f"Telegram API error {r.status_code}: {r.text[:200]}")


_SCHEMA_MIGRATION_LOCK_ID = 708110002


def _ensure_additive_columns(
    engine,
    *,
    lock_id: int,
    migrations: list,
    label: str,
) -> None:
    """
    Run ADD COLUMN migrations under a single-worker advisory lock.

    Skips ALTER when the column already exists (information_schema) so hot
    strategy_executions traffic does not stall on repeated ACCESS EXCLUSIVE
    attempts from every gunicorn worker at boot.
    """
    import time as _time

    import sqlalchemy as sa
    from app.db_resilience import is_transient_db_error

    def _benign(exc: Exception) -> bool:
        es = str(exc).lower()
        return (
            "already exists" in es
            or "duplicate" in es
            or "locknotavailable" in es
            or "lock timeout" in es
            or is_transient_db_error(exc)
        )

    for _attempt in range(1, 4):
        try:
            _run_ensure_additive_columns_inner(
                engine, lock_id=lock_id, migrations=migrations, label=label,
                sa=sa, _benign=_benign,
            )
            return
        except Exception as e:
            if is_transient_db_error(e) and _attempt < 3:
                logger.warning(
                    f"_ensure_additive_columns({label}): transient DB error "
                    f"(attempt {_attempt}/3) — retrying: {e}"
                )
                _time.sleep(0.5 * _attempt)
                continue
            if _benign(e):
                logger.info(f"_ensure_additive_columns({label}) outer skip: {type(e).__name__}")
            else:
                logger.warning(f"_ensure_additive_columns({label}) outer: {e}")
            return


def _run_ensure_additive_columns_inner(
    engine,
    *,
    lock_id: int,
    migrations: list,
    label: str,
    sa,
    _benign,
) -> None:
    conn = engine.connect()
    try:
        got = conn.execute(
            sa.text("SELECT pg_try_advisory_lock(:lid)"),
            {"lid": lock_id},
        ).scalar()
        if not got:
            logger.info(
                f"_ensure_additive_columns({label}): another worker migrating — skip"
            )
            return
        try:
            conn.execute(sa.text("SET lock_timeout = '8s'"))
            conn.execute(sa.text("SET statement_timeout = '30000'"))
            for table, col, ddl in migrations:
                exists = conn.execute(
                    sa.text(
                        "SELECT 1 FROM information_schema.columns "
                        "WHERE table_schema = 'public' "
                        "AND table_name = :t AND column_name = :c LIMIT 1"
                    ),
                    {"t": table, "c": col},
                ).scalar()
                if exists:
                    continue
                try:
                    conn.execute(sa.text(ddl))
                    conn.commit()
                    logger.info(f"_ensure_additive_columns({label}): added {table}.{col}")
                except Exception as e:
                    try:
                        conn.rollback()
                    except Exception:
                        pass
                    if _benign(e):
                        logger.info(
                            f"_ensure_additive_columns({label}): {table}.{col} "
                            f"skipped ({type(e).__name__})"
                        )
                    else:
                        logger.warning(
                            f"_ensure_additive_columns({label}): {table}.{col}: {e}"
                        )
        finally:
            try:
                conn.execute(
                    sa.text("SELECT pg_advisory_unlock(:lid)"),
                    {"lid": lock_id},
                )
            except Exception:
                pass
    finally:
        try:
            conn.close()
        except Exception:
            pass


_PIP_COLUMN_MIGRATIONS = [
    (
        "strategy_executions",
        "pips_pnl",
        "ALTER TABLE strategy_executions ADD COLUMN IF NOT EXISTS pips_pnl FLOAT",
    ),
    (
        "strategy_executions",
        "spread_pips_applied",
        "ALTER TABLE strategy_executions ADD COLUMN IF NOT EXISTS spread_pips_applied FLOAT",
    ),
    (
        "strategy_performance",
        "total_pips_pnl",
        "ALTER TABLE strategy_performance ADD COLUMN IF NOT EXISTS total_pips_pnl FLOAT",
    ),
    (
        "strategy_performance",
        "avg_pips_per_trade",
        "ALTER TABLE strategy_performance ADD COLUMN IF NOT EXISTS avg_pips_per_trade FLOAT",
    ),
]

_CTRADER_COLUMN_MIGRATIONS = [
    (
        "strategy_executions",
        "ctrader_order_id",
        "ALTER TABLE strategy_executions ADD COLUMN IF NOT EXISTS ctrader_order_id VARCHAR(80)",
    ),
    (
        "user_preferences",
        "ctrader_access_token",
        "ALTER TABLE user_preferences ADD COLUMN IF NOT EXISTS ctrader_access_token VARCHAR",
    ),
    (
        "user_preferences",
        "ctrader_refresh_token",
        "ALTER TABLE user_preferences ADD COLUMN IF NOT EXISTS ctrader_refresh_token VARCHAR",
    ),
    (
        "user_preferences",
        "ctrader_account_id",
        "ALTER TABLE user_preferences ADD COLUMN IF NOT EXISTS ctrader_account_id VARCHAR",
    ),
    (
        "user_preferences",
        "ctrader_accounts",
        "ALTER TABLE user_preferences ADD COLUMN IF NOT EXISTS ctrader_accounts TEXT",
    ),
    (
        "user_preferences",
        "forex_approved",
        "ALTER TABLE user_preferences ADD COLUMN IF NOT EXISTS forex_approved BOOLEAN DEFAULT FALSE",
    ),
]


def _ensure_tables():
    import sqlalchemy as sa
    import time as _time
    from app.database import Base
    # Make sure new model classes are imported BEFORE create_all so their
    # tables are registered with the metadata.
    from app.models import (  # noqa: F401
        IndicatorAlert, TradeIndicatorSetup,
        AutoTradeStrategy, AutoTradePaperTrade, TradeDrawing,
    )

    # Each legacy initializer runs in its own try/except so an error in one
    # cannot prevent the create_all calls below from running.
    for label, fn in (
        ("init_strategy_tables", init_strategy_tables),
        ("init_marketplace_ext_tables", init_marketplace_ext_tables),
        ("init_social_tables", init_social_tables),
    ):
        try:
            fn(engine)
        except Exception as e:
            logger.error(f"_ensure_tables({label}): {e}", exc_info=True)

    def _create_with_retry(label: str, tables: list, attempts: int = 3) -> bool:
        """Idempotent CREATE TABLE IF NOT EXISTS with retry for transient errors
        (Neon connection drops, multi-worker startup races, etc.)."""
        last_err = None
        for i in range(1, attempts + 1):
            try:
                Base.metadata.create_all(bind=engine, tables=tables)
                return True
            except Exception as e:
                last_err = e
                logger.warning(
                    f"_ensure_tables({label}) attempt {i}/{attempts}: {type(e).__name__}: {e}"
                )
                if i < attempts:
                    _time.sleep(0.7 * i)
        logger.error(f"_ensure_tables({label}) FAILED after {attempts} attempts: {last_err}")
        return False

    _create_with_retry("IndicatorAlert", [IndicatorAlert.__table__])
    _create_with_retry("TradeIndicatorSetup", [TradeIndicatorSetup.__table__])
    _create_with_retry("AutoTrade*", [
        AutoTradeStrategy.__table__,
        AutoTradePaperTrade.__table__,
    ])
    _create_with_retry("TradeDrawing", [TradeDrawing.__table__])

    # ── Hot-path indexes ──────────────────────────────────────────────────────
    # SQLAlchemy `index=True` on a Column only creates the index when the
    # TABLE is initially created via create_all. If the column was added to
    # an existing table later (or the table was created before the index
    # decoration), the index is never backfilled. On Neon Postgres, missing
    # indexes on these hot paths cause /app and every authed request to time
    # out with `statement_timeout` because each query degenerates into a full
    # table scan over users / strategy_executions.
    # CREATE INDEX CONCURRENTLY: does not lock the table, can survive on a
    # saturated Neon instance, and MUST run outside any transaction. We use
    # an autocommit connection and bump statement_timeout to 5 minutes for
    # this one connection so the initial backfill of a large index never
    # gets killed by the default 8 s timeout. Each statement is run in its
    # own try/except so one slow index doesn't block the others.
    _index_ddl = [
        ("idx_users_uid", "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_users_uid ON users(uid)"),
        ("idx_users_telegram_id", "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_users_telegram_id ON users(telegram_id)"),
        ("idx_strategy_executions_strategy_id", "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_strategy_executions_strategy_id ON strategy_executions(strategy_id)"),
        ("idx_strategy_executions_user_id", "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_strategy_executions_user_id ON strategy_executions(user_id)"),
        ("idx_strategy_executions_outcome_paper", "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_strategy_executions_outcome_paper ON strategy_executions(outcome, is_paper)"),
        ("idx_user_strategies_user_id", "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_user_strategies_user_id ON user_strategies(user_id)"),
        # Composite indexes for the executor's per-strategy hot queries
        # (cooldown / daily-cap / last-fired lookups). Without these, each
        # WHERE strategy_id=? AND fired_at>=today and ORDER BY fired_at DESC
        # query degenerates into a sort over every row matching strategy_id,
        # which on busy strategies pushes the 60s bg statement_timeout.
        ("idx_strategy_executions_sid_fired", "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_strategy_executions_sid_fired ON strategy_executions(strategy_id, fired_at DESC)"),
        ("idx_strategy_executions_sid_symbol_fired", "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_strategy_executions_sid_symbol_fired ON strategy_executions(strategy_id, symbol, fired_at DESC)"),
        ("idx_strategy_executions_sid_outcome", "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_strategy_executions_sid_outcome ON strategy_executions(strategy_id, outcome)"),
        # close_stale_open_executions: WHERE outcome='OPEN' AND fired_at<=?
        ("idx_strategy_executions_outcome_fired", "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_strategy_executions_outcome_fired ON strategy_executions(outcome, fired_at)"),
        # /api/portfolio/trades ORDER BY coalesce(closed_at, fired_at) DESC
        ("idx_strategy_executions_closed_fired", "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_strategy_executions_closed_fired ON strategy_executions(closed_at DESC NULLS LAST, fired_at DESC)"),
        # /api/portfolio JOIN user_strategies WHERE user_id=? (covers the JOIN filter)
        ("idx_strategy_executions_uid_outcome", "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_strategy_executions_uid_outcome ON strategy_executions(user_id, outcome)"),
        # /api/trade/auto/list batched DISTINCT ON (strategy_id) ORDER BY
        # strategy_id, opened_at DESC — without this the lookup degrades to a
        # full table scan + sort and hits Neon's statement_timeout once the
        # paper-trade history grows.
        ("idx_auto_trade_paper_sid_opened", "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_auto_trade_paper_sid_opened ON auto_trade_paper_trades(strategy_id, opened_at DESC)"),
    ]
    try:
        # AUTOCOMMIT is required for CREATE INDEX CONCURRENTLY.
        ac_conn = engine.connect().execution_options(isolation_level="AUTOCOMMIT")
        try:
            ac_conn.execute(sa.text("SET statement_timeout = '300s'"))
            # Postgres advisory lock so concurrent gunicorn workers don't race
            # each other into ShareUpdateExclusiveLock deadlocks on the same
            # CREATE INDEX CONCURRENTLY (which we hit in production logs).
            # Non-blocking try-lock: if another worker holds it, skip — the
            # winner will create the indexes for everyone.
            _got_lock = ac_conn.execute(
                sa.text("SELECT pg_try_advisory_lock(708110001)")
            ).scalar()
            if not _got_lock:
                logger.info("_ensure_tables(indexes): another worker holds the lock — skipping")
                return
            for label, ddl in _index_ddl:
                try:
                    ac_conn.execute(sa.text(ddl))
                    logger.info(f"_ensure_tables(indexes): {label} ready")
                except Exception as ie:
                    iemsg = str(ie)
                    if "already exists" in iemsg or "duplicate key" in iemsg:
                        logger.info(f"_ensure_tables(indexes): {label} already exists")
                    else:
                        logger.warning(f"_ensure_tables(indexes) {label}: {ie}")
        finally:
            try:
                ac_conn.execute(sa.text("SELECT pg_advisory_unlock(708110001)"))
            except Exception:
                pass
            ac_conn.close()
    except Exception as e:
        logger.warning(f"_ensure_tables(indexes outer): {e}")

    # Forex pip + cTrader columns — one worker only; skip when column already exists.
    _ensure_additive_columns(
        engine,
        lock_id=_SCHEMA_MIGRATION_LOCK_ID,
        migrations=_PIP_COLUMN_MIGRATIONS,
        label="pip_columns",
    )
    logger.info("_ensure_tables: forex pip columns ready")
    _ensure_additive_columns(
        engine,
        lock_id=_SCHEMA_MIGRATION_LOCK_ID,
        migrations=_CTRADER_COLUMN_MIGRATIONS,
        label="ctrader_columns",
    )
    logger.info("_ensure_tables: ctrader columns ready")
    from app.trade_mgmt_schema import ensure_trade_mgmt_columns
    if not ensure_trade_mgmt_columns(engine, wait_seconds=15.0):
        logger.error("_ensure_tables: trade management columns NOT ready after wait")
    else:
        logger.info("_ensure_tables: trade management columns ready")

    # Auto-promote admin users to lifetime Pro + forex-approved so they always
    # bypass every Pro gate without needing the in-memory admin-bypass path.
    # Idempotent — safe to run on every worker boot.
    try:
        with engine.begin() as conn:
            conn.execute(sa.text("""
                INSERT INTO portal_subscriptions (user_id, tier, subscription_end)
                SELECT id, 'pro', '2099-12-31 23:59:59'::timestamp
                FROM users WHERE is_admin = TRUE
                ON CONFLICT (user_id) DO UPDATE
                    SET tier = 'pro',
                        subscription_end = '2099-12-31 23:59:59'::timestamp
            """))
            conn.execute(sa.text("""
                UPDATE user_preferences
                SET forex_approved = TRUE
                WHERE user_id IN (SELECT id FROM users WHERE is_admin = TRUE)
            """))
        logger.info("_ensure_tables: admin users set to lifetime Pro + forex approved")
    except Exception as e:
        logger.warning(f"_ensure_tables(admin-lifetime-pro): {e}")

    # Kubera (TH-UAI2DR1J) — 3-month Pro subscription (ends 2026-08-31).
    # Idempotent — safe to run on every worker boot.
    try:
        with engine.begin() as conn:
            conn.execute(sa.text("""
                INSERT INTO portal_subscriptions (user_id, tier, subscription_end)
                SELECT id, 'pro', '2026-08-31 23:59:59'::timestamp
                FROM users WHERE uid = 'TH-UAI2DR1J'
                ON CONFLICT (user_id) DO UPDATE
                    SET tier = 'pro',
                        subscription_end = GREATEST(
                            portal_subscriptions.subscription_end,
                            '2026-08-31 23:59:59'::timestamp
                        )
            """))
        logger.info("_ensure_tables: Kubera (TH-UAI2DR1J) pro subscription ensured")
    except Exception as e:
        logger.warning(f"_ensure_tables(kubera-pro): {e}")

    # Affiliate program — applications + per-user affiliate state. Raw CREATE
    # to avoid registering yet another SQLAlchemy model just for a simple,
    # mostly-write-once table.
    try:
        with engine.begin() as conn:
            conn.execute(sa.text("""
                CREATE TABLE IF NOT EXISTS affiliate_applications (
                    id SERIAL PRIMARY KEY,
                    user_id INTEGER NOT NULL,
                    telegram VARCHAR NOT NULL,
                    twitter VARCHAR,
                    instagram VARCHAR,
                    youtube VARCHAR,
                    tiktok VARCHAR,
                    website VARCHAR,
                    bio TEXT NOT NULL,
                    plan TEXT NOT NULL,
                    status VARCHAR NOT NULL DEFAULT 'pending',
                    sub_share_pct DOUBLE PRECISION DEFAULT 30,
                    fee_share_pct DOUBLE PRECISION DEFAULT 20,
                    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
                    reviewed_at TIMESTAMP,
                    reviewer_note TEXT,
                    UNIQUE(user_id)
                )
            """))
            conn.execute(sa.text(
                "CREATE INDEX IF NOT EXISTS idx_affiliate_apps_status ON affiliate_applications(status)"
            ))
        logger.info("_ensure_tables: affiliate_applications ready")
    except Exception as e:
        # Multi-worker startup race on `CREATE TABLE IF NOT EXISTS` can still
        # raise DuplicateTable / pg_type unique violation when two workers run
        # the DDL simultaneously. The table ends up created either way — the
        # loser's exception is harmless, so log it at info level only.
        emsg = str(e)
        if "already exists" in emsg or "duplicate key" in emsg:
            logger.info("_ensure_tables(affiliate_applications): lost startup race (harmless)")
        else:
            logger.error(f"_ensure_tables(affiliate_applications): {e}")

    # Verification — query pg_tables for the names we expect and shout loudly
    # if any are still missing so we never silently end up with broken save APIs.
    try:
        expected = {
            "indicator_alerts", "trade_indicator_setups",
            "auto_trade_strategies", "auto_trade_paper_trades",
            "trade_drawings",
        }
        with engine.connect() as conn:
            present = {
                row[0] for row in conn.execute(sa.text(
                    "SELECT tablename FROM pg_tables WHERE schemaname='public' "
                    "AND tablename = ANY(:names)"
                ), {"names": list(expected)})
            }
        missing = expected - present
        if missing:
            logger.error(
                f"_ensure_tables VERIFICATION FAILED — tables still missing after "
                f"create_all: {sorted(missing)}. Save endpoints WILL 500 until this "
                f"is resolved."
            )
        else:
            logger.info(f"_ensure_tables: all expected tables present ({len(expected)})")
    except Exception as e:
        logger.error(f"_ensure_tables verification step error: {e}")
    # Only ALTER if column is genuinely missing — avoids table locks when both
    # dev and production portals start against the same shared Neon database.
    user_cols = {
        "email":          "ALTER TABLE users ADD COLUMN email VARCHAR UNIQUE",
        "email_verified": "ALTER TABLE users ADD COLUMN email_verified BOOLEAN DEFAULT FALSE",
        "password_hash":  "ALTER TABLE users ADD COLUMN password_hash VARCHAR",
        "google_id":      "ALTER TABLE users ADD COLUMN google_id VARCHAR UNIQUE",
        "apple_id":       "ALTER TABLE users ADD COLUMN apple_id VARCHAR UNIQUE",
        "auth_provider":  "ALTER TABLE users ADD COLUMN auth_provider VARCHAR DEFAULT 'telegram'",
        "uid":            "ALTER TABLE users ADD COLUMN uid VARCHAR UNIQUE",
    }

    # New columns on the auto-trader strategy/trade tables (Phase: do them all).
    # Each entry is column_name → ALTER SQL. Defaults must be set so existing
    # rows pick up sensible values without a backfill step.
    auto_strategy_cols = {
        "max_concurrent_trades":       "ALTER TABLE auto_trade_strategies ADD COLUMN max_concurrent_trades INTEGER DEFAULT 1",
        "max_daily_loss_usd":          "ALTER TABLE auto_trade_strategies ADD COLUMN max_daily_loss_usd DOUBLE PRECISION",
        "max_consecutive_losses":      "ALTER TABLE auto_trade_strategies ADD COLUMN max_consecutive_losses INTEGER DEFAULT 0",
        "position_sizing_mode":        "ALTER TABLE auto_trade_strategies ADD COLUMN position_sizing_mode VARCHAR DEFAULT 'fixed'",
        "risk_pct":                    "ALTER TABLE auto_trade_strategies ADD COLUMN risk_pct DOUBLE PRECISION",
        "account_size_usd":            "ALTER TABLE auto_trade_strategies ADD COLUMN account_size_usd DOUBLE PRECISION DEFAULT 10000",
        "enable_partial_tp1":          "ALTER TABLE auto_trade_strategies ADD COLUMN enable_partial_tp1 BOOLEAN DEFAULT FALSE",
        "partial_tp1_pct":             "ALTER TABLE auto_trade_strategies ADD COLUMN partial_tp1_pct DOUBLE PRECISION DEFAULT 50",
        "move_stop_to_be_after_tp1":   "ALTER TABLE auto_trade_strategies ADD COLUMN move_stop_to_be_after_tp1 BOOLEAN DEFAULT FALSE",
        "session_start_utc":           "ALTER TABLE auto_trade_strategies ADD COLUMN session_start_utc VARCHAR",
        "session_end_utc":             "ALTER TABLE auto_trade_strategies ADD COLUMN session_end_utc VARCHAR",
        "cooldown_minutes_after_loss": "ALTER TABLE auto_trade_strategies ADD COLUMN cooldown_minutes_after_loss INTEGER DEFAULT 0",
        "consecutive_losses":          "ALTER TABLE auto_trade_strategies ADD COLUMN consecutive_losses INTEGER DEFAULT 0",
        "paused_until":                "ALTER TABLE auto_trade_strategies ADD COLUMN paused_until TIMESTAMP",
        "daily_loss_today_usd":        "ALTER TABLE auto_trade_strategies ADD COLUMN daily_loss_today_usd DOUBLE PRECISION DEFAULT 0",
        "daily_loss_date":             "ALTER TABLE auto_trade_strategies ADD COLUMN daily_loss_date VARCHAR",
        # Risk profile (N) — picky vs loose entry behaviour
        "risk_profile":                "ALTER TABLE auto_trade_strategies ADD COLUMN risk_profile VARCHAR DEFAULT 'medium'",
        "min_minutes_between_trades":  "ALTER TABLE auto_trade_strategies ADD COLUMN min_minutes_between_trades INTEGER DEFAULT 0",
        "last_trade_closed_at":        "ALTER TABLE auto_trade_strategies ADD COLUMN last_trade_closed_at TIMESTAMP",
    }
    indicator_alerts_cols = {
        "fire_mode":         "ALTER TABLE indicator_alerts ADD COLUMN fire_mode VARCHAR NOT NULL DEFAULT 'once'",
        "cooldown_minutes":  "ALTER TABLE indicator_alerts ADD COLUMN cooldown_minutes INTEGER NOT NULL DEFAULT 0",
        "daily_cap":         "ALTER TABLE indicator_alerts ADD COLUMN daily_cap INTEGER",
        "fire_count":        "ALTER TABLE indicator_alerts ADD COLUMN fire_count INTEGER NOT NULL DEFAULT 0",
        "last_fired_at":     "ALTER TABLE indicator_alerts ADD COLUMN last_fired_at TIMESTAMP",
        "fired_today_count": "ALTER TABLE indicator_alerts ADD COLUMN fired_today_count INTEGER NOT NULL DEFAULT 0",
        "fired_today_date":  "ALTER TABLE indicator_alerts ADD COLUMN fired_today_date VARCHAR",
    }

    auto_trade_cols = {
        "tp1_hit":                "ALTER TABLE auto_trade_paper_trades ADD COLUMN tp1_hit BOOLEAN DEFAULT FALSE",
        "tp1_hit_at":             "ALTER TABLE auto_trade_paper_trades ADD COLUMN tp1_hit_at TIMESTAMP",
        "partial_pnl_usd":        "ALTER TABLE auto_trade_paper_trades ADD COLUMN partial_pnl_usd DOUBLE PRECISION DEFAULT 0",
        "stop_moved_to_be":       "ALTER TABLE auto_trade_paper_trades ADD COLUMN stop_moved_to_be BOOLEAN DEFAULT FALSE",
        "original_notional_usd":  "ALTER TABLE auto_trade_paper_trades ADD COLUMN original_notional_usd DOUBLE PRECISION",
        "remaining_notional_usd": "ALTER TABLE auto_trade_paper_trades ADD COLUMN remaining_notional_usd DOUBLE PRECISION",
    }

    def _migrate_columns(table: str, needed: dict):
        try:
            with engine.connect() as conn:
                conn.execute(sa.text("SET statement_timeout = '15s'"))
                existing = {
                    row[0] for row in conn.execute(sa.text(
                        "SELECT column_name FROM information_schema.columns "
                        "WHERE table_name=:t AND table_schema='public'"
                    ), {"t": table})
                }
                for col, sql in needed.items():
                    if col not in existing:
                        try:
                            conn.execute(sa.text(sql))
                            logger.info(f"Migration: added {table}.{col}")
                        except Exception as ce:
                            logger.warning(f"Column migration {table}.{col}: {ce}")
                conn.commit()
        except Exception as e:
            logger.warning(f"_ensure_tables({table}): {e}")

    user_preference_cols = {
        "push_notify_paper":      "ALTER TABLE user_preferences ADD COLUMN push_notify_paper BOOLEAN DEFAULT TRUE",
        "push_notify_live":       "ALTER TABLE user_preferences ADD COLUMN push_notify_live BOOLEAN DEFAULT TRUE",
        "push_min_position_usd":  "ALTER TABLE user_preferences ADD COLUMN push_min_position_usd DOUBLE PRECISION DEFAULT 0",
        "account_balance":        "ALTER TABLE user_preferences ADD COLUMN account_balance DOUBLE PRECISION DEFAULT 10000",
        "lot_size":               "ALTER TABLE user_preferences ADD COLUMN lot_size DOUBLE PRECISION DEFAULT 0.1",
    }

    strategy_cols = {
        "webhook_token": "ALTER TABLE user_strategies ADD COLUMN IF NOT EXISTS webhook_token VARCHAR(64)",
        "asset_class":   "ALTER TABLE user_strategies ADD COLUMN IF NOT EXISTS asset_class VARCHAR(16) NOT NULL DEFAULT 'crypto'",
    }

    execution_cols = {
        "asset_class": "ALTER TABLE strategy_executions ADD COLUMN IF NOT EXISTS asset_class VARCHAR(16) NOT NULL DEFAULT 'crypto'",
    }

    _migrate_columns("users", user_cols)
    _migrate_columns("auto_trade_strategies", auto_strategy_cols)
    _migrate_columns("auto_trade_paper_trades", auto_trade_cols)
    _migrate_columns("indicator_alerts", indicator_alerts_cols)
    _migrate_columns("user_preferences", user_preference_cols)
    _migrate_columns("user_strategies", strategy_cols)
    _migrate_columns("strategy_executions", execution_cols)

    # ── Backfill asset_class column from config JSON ───────────────────────────
    # Strategies created before this column was properly set (e.g. forex/index
    # strategies that got the DEFAULT 'crypto' from the migration) will have
    # asset_class='crypto' in the column but 'forex'/'stock'/'index' inside the
    # config JSON blob. The executor reads the column first; if it says 'crypto'
    # (truthy) the config fallback is never reached, so forex strategies scan as
    # crypto and find no price → silently skip every cycle.
    try:
        from sqlalchemy import text as _text
        with engine.begin() as _conn:
            _result = _conn.execute(_text("""
                UPDATE user_strategies
                   SET asset_class = config->>'asset_class'
                 WHERE config->>'asset_class' IN ('crypto','stock','forex','index','commodity')
                   AND config->>'asset_class' IS DISTINCT FROM asset_class
                RETURNING id, name, config->>'asset_class' AS new_class
            """))
            _fixed = _result.fetchall()
            if _fixed:
                for _row in _fixed:
                    logger.info(
                        f"[migration] Fixed asset_class for strategy {_row[0]} "
                        f"'{_row[1]}' → {_row[2]}"
                    )
            else:
                logger.info("[migration] asset_class backfill: all rows already consistent")
    except Exception as _e:
        logger.warning(f"[migration] asset_class backfill failed (non-fatal): {_e}")


@app.on_event("startup")
async def startup():
    # Never await DB work here — each gunicorn worker must accept HTTP immediately.
    # Schema ADD COLUMN migrations run once inside _startup_background → _ensure_tables
    # (advisory-locked; duplicate startup DDL removed to avoid lock-timeout noise).
    asyncio.create_task(_startup_background())
    asyncio.create_task(_refresh_heavy_caches())
    if is_replit() or is_railway():
        asyncio.create_task(_keepalive_ping_loop())

    async def _prime_spot_store():
        await asyncio.sleep(3)
        try:
            from app.services.spot_price_store import _ensure_table
            await asyncio.to_thread(_ensure_table)
        except Exception:
            pass

    asyncio.create_task(_prime_spot_store())
    logger.info("Strategy portal ready — migrations running in background")


def _do_refresh_heavy_caches_sync():
    """Sync DB work for cache pre-warming — called via asyncio.to_thread so it
    never blocks the event loop."""
    from app.database import SessionLocal
    from sqlalchemy import text as _text
    from datetime import datetime as _dt, timedelta as _td
    from app.strategy_models import StrategyMarketplace as _SM, StrategyPerformance as _SP
    db = SessionLocal()
    try:
        _listings = db.query(_SM).all()
        _mkt_sids = [m.strategy_id for m in _listings]
        _perf_map = {p.strategy_id: p for p in db.query(_SP).filter(_SP.strategy_id.in_(_mkt_sids)).all()} if _mkt_sids else {}
        _auth_ids = list({m.author_id for m in _listings if m.author_id})
        _auth_map = {u.id: u for u in db.query(User).filter(User.id.in_(_auth_ids)).all()} if _auth_ids else {}
        _ldr_rows = []
        for m in _listings:
            perf = _perf_map.get(m.strategy_id)
            author = _auth_map.get(m.author_id)
            if not perf or perf.total_trades < 3 or perf.total_pnl_pct <= 0:
                continue
            _ldr_rows.append({
                "strategy_id": m.strategy_id, "listing_id": m.id,
                "title": m.title, "mode": "live",
                "author": (author.first_name or author.username) if author else "Anonymous",
                "author_uid": author.uid if author else None,
                "is_own": False,
                "win_rate": round(perf.win_rate, 1),
                "total_pnl": round(perf.total_pnl_pct, 2),
                "total_trades": perf.total_trades,
                "avg_rating": round(m.avg_rating or 0, 1),
                "pricing_model": m.pricing_model or "free",
                "price_usdt": m.price_usdt or 0,
            })
        _ldr_rows.sort(key=lambda x: x.get("win_rate", 0), reverse=True)
        set_cache("_warmup:ldr:win_rate", _ldr_rows[:30], 130)

        cutoff = _dt.utcnow() - _td(days=30)
        rows = db.execute(_text("""
            SELECT m.id AS listing_id, m.title, m.author_id,
                   m.pricing_model, m.price_usdt, m.is_verified, m.is_ai_generated,
                   COALESCE(SUM(e.pnl_pct), 0) AS pnl_sum,
                   COUNT(e.id) AS trades,
                   SUM(CASE WHEN e.outcome='WIN' THEN 1 ELSE 0 END) AS wins,
                   SUM(CASE WHEN e.outcome IN ('WIN','LOSS') THEN 1 ELSE 0 END) AS decisive
            FROM strategy_marketplace m
            JOIN strategy_executions e ON e.strategy_id = m.strategy_id
            WHERE e.outcome IN ('WIN','LOSS','BREAKEVEN')
              AND e.pnl_pct IS NOT NULL
              AND COALESCE(e.closed_at, e.fired_at) > :cutoff
            GROUP BY m.id, m.title, m.author_id, m.pricing_model,
                     m.price_usdt, m.is_verified, m.is_ai_generated
            HAVING COUNT(e.id) >= 1
            ORDER BY pnl_sum DESC LIMIT 10
        """), {"cutoff": cutoff}).fetchall()
        _lb_author_ids = list({r.author_id for r in rows})
        _lb_names: dict = {}
        if _lb_author_ids:
            for u in db.query(User).filter(User.id.in_(_lb_author_ids)).all():
                _lb_names[u.id] = u.first_name or u.username or "Anonymous"
        entries = []
        for rank, r in enumerate(rows, 1):
            decisive = int(getattr(r, "decisive", 0) or 0)
            wins = int(r.wins or 0)
            wr = round((wins / decisive) * 100, 1) if decisive > 0 else 0.0
            entries.append({
                "rank": rank, "listing_id": r.listing_id, "title": r.title,
                "author_name": _lb_names.get(r.author_id, "Anonymous"),
                "pricing_model": r.pricing_model or "free",
                "price_usdt": float(r.price_usdt or 0),
                "is_verified": bool(r.is_verified),
                "is_ai_generated": bool(r.is_ai_generated),
                "pnl_pct": round(float(r.pnl_sum or 0), 2),
                "trades": int(r.trades or 0), "win_rate": wr,
            })
        set_cache("_warmup:mkt_lb:30d:10", {"period": "30d", "entries": entries}, 130)
    finally:
        db.close()


async def _refresh_heavy_caches():
    """Pre-warm leaderboard and marketplace-leaderboard caches every 5 minutes.

    DB work runs in asyncio.to_thread so it never blocks the event loop.
    Sleep is at the BOTTOM of the loop so caches are populated soon after
    the initial 60 s startup delay.
    """
    await asyncio.sleep(60)  # let workers settle after startup
    while True:
        try:
            await asyncio.to_thread(_do_refresh_heavy_caches_sync)
            logger.debug("[cache-refresh] heavy caches pre-warmed")
        except Exception as _e:
            logger.debug(f"[cache-refresh] error: {_e}")
        await asyncio.sleep(300)


def _do_cancel_ghost_executions_sync():
    """Sync DB work for ghost cleanup — called via asyncio.to_thread."""
    from app.database import SessionLocal
    from sqlalchemy import text
    db = SessionLocal()
    try:
        result = db.execute(text("""
            UPDATE strategy_executions
            SET outcome = 'CANCELLED',
                notes   = 'Auto-cancelled: no broker order_id (ghost execution)'
            WHERE is_paper = false
              AND outcome  = 'OPEN'
              AND bitunix_order_id IS NULL
              AND ctrader_order_id IS NULL
              AND fired_at < NOW() - INTERVAL '5 minutes'
            RETURNING id, symbol, direction
        """))
        cancelled = result.fetchall()
        db.commit()
        return cancelled
    finally:
        db.close()


async def _cancel_ghost_executions():
    """
    Cancel any live (is_paper=False) OPEN strategy executions that have NO
    broker order ID at all (neither Bitunix for crypto NOR cTrader for
    forex/index).  These are 'ghost' records created when the broker API call
    failed — they were never real positions, but old code left them OPEN so the
    live monitor would fire false SL/TP alerts.
    CRITICAL: a real cTrader forex/index trade has ctrader_order_id set but
    bitunix_order_id NULL; it must NOT be treated as a ghost (doing so silently
    cancelled live forex stop-outs and suppressed their close notifications).
    Run once on startup and then every 5 minutes as a safety net.
    DB work runs in asyncio.to_thread to avoid blocking the event loop.
    """
    try:
        cancelled = await asyncio.to_thread(_do_cancel_ghost_executions_sync)
        if cancelled:
            for row in cancelled:
                logger.warning(
                    f"[ghost-cleanup] Cancelled ghost execution id={row[0]} "
                    f"{row[1]} {row[2]} — no broker order_id"
                )
        else:
            logger.debug("[ghost-cleanup] No ghost executions found")
    except Exception as e:
        logger.error(f"[ghost-cleanup] Error: {e}")


async def _ghost_cleanup_loop():
    """
    Runs _cancel_ghost_executions every 5 minutes indefinitely.
    Only started in the worker that holds the executor advisory lock.
    """
    while True:
        await asyncio.sleep(300)
        await _cancel_ghost_executions()


# ── PostgreSQL advisory lock — ensures only ONE uvicorn worker runs the
#    executor, monitors, and ghost-cleanup (not all 4 workers). ─────────────
# Must fit PostgreSQL int32 for pg_locks.objid queries. Old id 42424242 was
# held by a zombie Replit/Railway session (pid ~10481) that blocked the executor.
from app.executor_lock import (
    EXECUTOR_LOCK_ID,
    FOREX_EXECUTOR_LOCK_ID,
    get_executor_lock_id,
)
# Keepalive cadence for the lock-holding connection. Pinged from a DEDICATED
# DAEMON THREAD (not the asyncio loop, which the executor's scan cycles block for
# tens of seconds), so it must stay comfortably BELOW _EXECUTOR_LOCK_STALE_IDLE_SECS.
_EXECUTOR_LOCK_KEEPALIVE_SECS = 30
# A waiting worker only force-terminates the current lock holder if it has been
# idle (un-pinged) at least this long — i.e. it is a genuine zombie from a dead
# deploy, NOT the live sibling worker that legitimately holds the lock. With
# GUNICORN_WORKERS=2 the HTTP worker would otherwise kill the executor worker's
# lock connection, thrashing the executor so NO trades fire. The margin over the
# keepalive cadence (6x) absorbs event-loop / GIL pauses on the busy executor
# worker; genuine zombies are still reclaimed (and Postgres TCP keepalives reap
# a truly-dead connection in ~80s regardless).
_EXECUTOR_LOCK_STALE_IDLE_SECS = 120

def _try_acquire_executor_lock():
    """
    Try to acquire a PostgreSQL session-level advisory lock using a raw
    psycopg2 connection.  Returns the open connection (which must be kept
    alive to hold the lock) or None if another worker already holds it.
    Session-level advisory locks are automatically released when the
    connection (i.e. the process) closes.
    """
    try:
        from app.executor_lock import (
            close_lock_connection,
            create_lock_connection,
            get_executor_lock_id,
            try_acquire_lock,
        )

        conn = create_lock_connection()
        if try_acquire_lock(conn, get_executor_lock_id()):
            return conn
        close_lock_connection(conn)
        return None
    except Exception as e:
        logger.warning(f"Advisory lock attempt failed: {e}")
        return None

# Tracks whether THIS uvicorn worker is currently running the executor.
# Used by the claim loop to avoid double-starting.
_executor_running_in_this_worker = False
# Set after _start_executor_tasks() runs once — prevents duplicate scan loops
# when Neon SSL drops the lock session and we re-acquire without a full restart.
_executor_tasks_started = False
# Standalone force-reclaim runs once at process boot (executor_runner), not on
# every lock reconnect — otherwise we terminate our own stale session and
# restart prefetch from zero in a thrash loop.
_initial_executor_reclaim_done = False


def _advisory_lock_keepalive_thread(conn_holder, stop_event, lost_event):
    """Ping the advisory-lock connection on a fixed cadence from a DEDICATED
    THREAD — independent of the asyncio event loop, which the executor's scan
    cycles can block for tens of seconds at a time. Keeping the connection's
    `state_change` fresh is what stops a sibling worker from mistaking this live
    holder for a stale/zombie connection and terminating it.

    On Neon SSL blips, reconnect in-place and keep holding the executor lock
    without tearing down scan loops. Only sets `lost_event` when reconnect is
    exhausted so the async side can re-enter the claim loop."""
    import time as _t
    from app.executor_lock import reconnect_lock_connection

    while not stop_event.is_set():
        # Sleep in 1s steps so shutdown is prompt.
        for _ in range(_EXECUTOR_LOCK_KEEPALIVE_SECS):
            if stop_event.is_set():
                return
            _t.sleep(1)
        conn = conn_holder.get("conn")
        try:
            cur = conn.cursor()
            cur.execute("SELECT 1")
            cur.close()
        except Exception as e:
            # One silent reconnect on the same tick — avoids a full re-claim loop
            # for a single transient Neon SSL close.
            new_conn = reconnect_lock_connection(conn, max_attempts=1, silent=True)
            if new_conn:
                conn_holder["conn"] = new_conn
                logger.debug(
                    "Advisory lock keepalive: silent reconnect succeeded"
                )
                continue
            logger.warning(
                f"Advisory lock keepalive failed: {e} — re-claiming on fresh session"
            )
            new_conn = reconnect_lock_connection(conn)
            if new_conn:
                conn_holder["conn"] = new_conn
                logger.info(
                    "Advisory lock re-claimed on fresh session — executor continues"
                )
                continue
            logger.error(
                "Advisory lock re-claim exhausted (another holder or DB down) "
                "— will re-enter claim loop"
            )
            lost_event.set()
            return


async def _maintain_advisory_lock(conn):
    """
    Keep the psycopg2 advisory-lock connection alive via a dedicated daemon
    thread (NOT the event loop — the executor blocks it for 30s+ during scans,
    which previously let the connection go idle long enough that a sibling worker
    falsely reclaimed the lock and thrashed the executor). Returns when the
    connection dies and in-thread reconnect failed (signal to claim loop).
    """
    import threading
    stop_event = threading.Event()
    lost_event = threading.Event()
    conn_holder = {"conn": conn}
    t = threading.Thread(
        target=_advisory_lock_keepalive_thread,
        args=(conn_holder, stop_event, lost_event),
        daemon=True,
    )
    t.start()
    try:
        # Loss detection may lag if the loop is blocked, but that only delays the
        # (harmless) re-claim — the THREAD keeps the connection warm regardless.
        while not lost_event.is_set():
            await asyncio.sleep(5)
    finally:
        stop_event.set()


def _launch_executor_price_feeds():
    """Start FMP + cTrader + metals feeds on the executor-winning worker."""
    try:
        from app.services.fmp_price_feed import start as _fmp_feed_start
        _fmp_feed_start()
        logger.info("FMP real-time price feed task scheduled (executor worker)")
    except Exception as _fmp_err:
        logger.warning(f"FMP price feed start error (non-fatal): {_fmp_err}")
    try:
        from app.services.ctrader_price_feed import launch_ctrader_feed
        launch_ctrader_feed()
    except Exception as _ctf_err:
        logger.error(
            "[CTraderFeed] executor feed launch error: %s", _ctf_err, exc_info=True,
        )
    try:
        from app.services.metals_spot_feed import start as _metals_feed_start
        _metals_feed_start()
        logger.info("Metals spot feed (XAUUSD/XAGUSD) scheduled (executor worker)")
    except Exception as _met_err:
        logger.warning(f"Metals spot feed start error (non-fatal): {_met_err}")
    try:
        from app.services.economic_calendar import start as _econ_cal_start
        _econ_cal_start()
        logger.info("Economic calendar refresh scheduled (executor worker)")
    except Exception as _cal_err:
        logger.warning(f"Economic calendar start error (non-fatal): {_cal_err}")


def _ensure_executor_feeds_running():
    """Restart price feeds after lock loss without duplicating scan loops."""
    _launch_executor_price_feeds()


def _lock_loss_should_stop_feeds() -> bool:
    """Standalone executor has no sibling workers — keep feeds warm during re-claim."""
    import os as _os
    return _os.environ.get("EXECUTOR_STANDALONE", "").lower() not in (
        "1",
        "true",
        "yes",
    )


async def _restart_executor_subsystems_after_lock_reacquire():
    """After Neon SSL re-claim: restart feeds, order worker, and broker reconcile."""
    logger.info(
        "Executor lock re-acquired — restarting feeds, order worker, reconcile catch-up"
    )
    _ensure_executor_feeds_running()
    try:
        from app.services.ctrader_order_queue import start_ctrader_order_worker
        start_ctrader_order_worker()
    except Exception as _oq_err:
        logger.warning(f"cTrader order queue restart error (non-fatal): {_oq_err}")
    try:
        from app.services.strategy_executor import (
            _reconcile_forex_closes,
            mark_heartbeat,
        )
        mark_heartbeat("executor_lock_reacquire")
        await _reconcile_forex_closes()
    except Exception as _rc_err:
        logger.warning(f"Forex reconcile catch-up after re-claim failed: {_rc_err}")


async def _keepalive_ping_loop():
    """
    Ping the app every 4 minutes so idle hosts stay warm.

    Replit Autoscale scales to zero without traffic; Railway free/low tiers
    can sleep or take 20–40s to accept the first request after idle. A local
    /ping every 4 min keeps gunicorn workers alive and Neon warm.
    """
    if not (is_replit() or is_railway()):
        return
    import aiohttp as _aiohttp
    import os as _os

    if is_railway():
        port = int(_os.environ.get("PORT", "5000"))
        url = f"http://127.0.0.1:{port}/ping"
    else:
        domain = _os.environ.get("PUBLIC_DOMAIN", "tradehubmarkets.com")
        url = f"https://{domain}/health"

    await asyncio.sleep(45)   # let workers bind before first ping
    while True:
        try:
            async with _aiohttp.ClientSession() as _sess:
                async with _sess.get(url, timeout=_aiohttp.ClientTimeout(total=8)) as r:
                    logger.debug(f"[Keepalive] ping {url} → {r.status}")
        except Exception as _e:
            logger.debug(f"[Keepalive] ping failed (non-critical): {_e}")
        await asyncio.sleep(240)   # 4-minute interval — under Neon 5-min idle cutoff


async def _resilient_task(name: str, coro_fn, restart_delay: int = 30):
    """
    Wraps a background coroutine with auto-restart on crash or clean exit.
    Without this, a transient DB SSL error silently kills the executor forever.
    """
    while True:
        try:
            await coro_fn()
            logger.warning(f"⚠️ {name} exited cleanly — restarting in {restart_delay}s")
        except Exception as exc:
            logger.error(f"🔴 {name} crashed ({exc!r}) — restarting in {restart_delay}s")
        await asyncio.sleep(restart_delay)


async def _start_executor_tasks():
    """Import and launch the executor + monitor tasks in this worker."""
    from app.database import bg_engine
    from app.trade_mgmt_schema import ensure_trade_mgmt_columns

    ok = await asyncio.to_thread(ensure_trade_mgmt_columns, bg_engine, 15.0)
    if not ok:
        logger.error(
            "[executor] trade_mgmt schema not ready — delaying executor tasks 5s"
        )
        await asyncio.sleep(5)
        ok = await asyncio.to_thread(ensure_trade_mgmt_columns, bg_engine, 15.0)
    if not ok:
        logger.error(
            "[executor] trade_mgmt columns still missing — starting tasks anyway "
            "(FX-fast will retry migration on UndefinedColumn)"
        )

    # ── Price feeds FIRST (before ghost cleanup / scan loops) ─────────────────
    # Started HERE (not in _startup_background) so each runs in EXACTLY ONE worker —
    # the same one that runs the executor (their only consumer).
    logger.info("[executor] launching price feeds (FMP + cTrader + metals)")
    _launch_executor_price_feeds()

    await _cancel_ghost_executions()
    asyncio.create_task(_ghost_cleanup_loop())
    # Keepalive runs on every worker via startup(); executor worker no longer sole pinger.

    async def _spot_price_primer_loop():
        """Keep shared spot store warm — real-time ticks for metals + majors."""
        _KEYS = [
            ("XAUUSD", "forex"), ("XAGUSD", "forex"),
            ("EURUSD", "forex"), ("GBPUSD", "forex"), ("USDJPY", "forex"),
            ("AUDUSD", "forex"), ("USDCAD", "forex"),
            ("NAS100", "index"), ("US30", "index"), ("SPX500", "index"),
        ]
        _primer_s = max(3, int(os.environ.get("SPOT_PRIMER_INTERVAL_SECONDS", "5")))
        await asyncio.sleep(8)
        while True:
            try:
                from app.services.realtime_spot import prime_symbols
                n = await prime_symbols(_KEYS)
                if n:
                    logger.debug(f"[spot-primer] refreshed {n} symbol(s)")
            except Exception:
                pass
            await asyncio.sleep(_primer_s)

    asyncio.create_task(_spot_price_primer_loop())

    async def _executor_startup_heal():
        try:
            from app.services.strategy_heal import heal_on_executor_startup
            stats = await heal_on_executor_startup()
            logger.info("[executor] strategy heal on startup: %s", stats)
        except Exception as _heal_err:
            logger.warning("[executor] strategy heal failed (non-fatal): %s", _heal_err)

    asyncio.create_task(_executor_startup_heal())

    async def _executor_startup_reconcile():
        """Catch up broker-side closes that happened while executor was down."""
        await asyncio.sleep(3)
        try:
            from app.services.strategy_executor import _reconcile_forex_closes
            await _reconcile_forex_closes()
            logger.info("[executor] startup forex broker reconcile complete")
        except Exception as _sr_err:
            logger.warning("[executor] startup forex reconcile failed: %s", _sr_err)

    asyncio.create_task(_executor_startup_reconcile())

    async def _feed_startup_diagnostics():
        await asyncio.sleep(12)
        try:
            from app.services.feed_diagnostics import run_startup_diagnostics
            await run_startup_diagnostics()
        except Exception as _diag_err:
            logger.warning("[executor] feed diagnostics failed (non-fatal): %s", _diag_err)

    asyncio.create_task(_feed_startup_diagnostics())

    from app.services.strategy_executor import (
        run_strategy_executor, run_live_position_monitor,
        run_forex_executor, run_forex_live_manager_fast,
        run_paper_position_monitor, run_session_alert_loop,
        backfill_cancelled_paper_trades,
        backfill_ghost_cancelled_executions,
        close_stale_open_executions,
        crypto_executor_disabled, forex_executor_disabled,
        executor_runtime_profile,
    )
    _prof = executor_runtime_profile()
    logger.info(
        "[executor] runtime profile: executor_only=%s crypto_disabled=%s "
        "forex_disabled=%s crypto_cycle=%ss forex_cycle=%ss",
        _prof["executor_only"],
        _prof["crypto_disabled"],
        _prof["forex_disabled"],
        _prof["crypto_scan_interval_s"],
        _prof["forex_scan_interval_s"],
    )
    # Stale-position cleanup before scan loops — fire-and-forget so HTTP workers
    # are not blocked if Neon is slow on the first connection after deploy.
    asyncio.create_task(close_stale_open_executions(stale_after_hours=48))
    asyncio.create_task(
        _resilient_task("run_paper_position_monitor", run_paper_position_monitor, restart_delay=20)
    )
    if not forex_executor_disabled():
        asyncio.create_task(
            _resilient_task("run_session_alert_loop", run_session_alert_loop, restart_delay=30)
        )
    # Wrapped in _resilient_task so a transient crash (e.g. DB SSL drop)
    # auto-restarts instead of silently killing the executor permanently.
    if not crypto_executor_disabled():
        asyncio.create_task(_resilient_task("run_strategy_executor", run_strategy_executor, restart_delay=20))
    if not forex_executor_disabled():
        try:
            from app.services.ctrader_order_queue import start_ctrader_order_worker
            start_ctrader_order_worker()
            logger.info("cTrader async order queue worker started (executor worker)")
        except Exception as _oq_err:
            logger.warning(f"cTrader order queue start error (non-fatal): {_oq_err}")
        asyncio.create_task(_resilient_task("run_forex_executor", run_forex_executor, restart_delay=20))
        # Dedicated fast (~1s) loop that pushes live forex breakeven/trailing SL
        # amendments to cTrader off the real-time spot feed — so gold reaches
        # breakeven in well under a second instead of on the 5s scan cadence.
        asyncio.create_task(
            _resilient_task("run_forex_live_manager_fast", run_forex_live_manager_fast, restart_delay=15)
        )
    if not crypto_executor_disabled():
        asyncio.create_task(
            _resilient_task("run_live_position_monitor", run_live_position_monitor, restart_delay=15)
        )
    asyncio.create_task(backfill_cancelled_paper_trades(lookback_days=30))
    asyncio.create_task(backfill_ghost_cancelled_executions(lookback_days=7))
    # Indicator alerts loop — same advisory-lock-protected worker so a
    # multi-worker gunicorn deploy never double-fires alert DMs.
    try:
        from app.services.alerts_engine import run_alerts_engine
        asyncio.create_task(_resilient_task("run_alerts_engine", run_alerts_engine, restart_delay=30))
        logger.info("✅ Alerts engine task launched")
    except Exception as e:
        logger.error(f"Failed to launch alerts engine: {e}")

    # Periodic system health monitor (default every 4h) — verifies website +
    # data/broker connections + executor loops. Trade alerts stay instant.
    try:
        from app.services.system_health_check import run_system_health_monitor
        asyncio.create_task(_resilient_task("run_system_health_monitor", run_system_health_monitor, restart_delay=60))
        logger.info("✅ System health monitor task launched")
    except Exception as e:
        logger.error(f"Failed to launch system health monitor: {e}")

    # Optional one-shot owner ping on deploy — OFF by default (lock reclaim was
    # sending duplicate "executor online" DMs; trade alerts stay instant).
    async def _executor_startup_ping():
        import os as _os_ping
        if _os_ping.environ.get("EXECUTOR_STARTUP_PING", "").lower() not in (
            "1", "true", "yes",
        ):
            return
        try:
            from sqlalchemy import text as _txt
            from app.database import SessionLocal
            from app.services.telegram_dm import owner_chat_id, send_dm
            from app.deployment import deploy_commit

            owner = owner_chat_id()
            if not owner:
                logger.warning("[executor] OWNER_TELEGRAM_ID not set — startup ping skipped")
                return
            commit = (deploy_commit() or "unknown")[:12]
            dedup_key = f"startup_ping:{commit}"
            db = SessionLocal()
            try:
                db.execute(_txt("""
                    CREATE TABLE IF NOT EXISTS owner_notification_dedup (
                        key        VARCHAR(64) PRIMARY KEY,
                        sent_at    TIMESTAMP NOT NULL DEFAULT (NOW() AT TIME ZONE 'utc')
                    )
                """))
                exists = db.execute(
                    _txt("SELECT 1 FROM owner_notification_dedup WHERE key = :k"),
                    {"k": dedup_key},
                ).fetchone()
                if exists:
                    logger.info("[executor] startup ping already sent for this commit — skipped")
                    return
            finally:
                db.close()

            msg = (
                "🟢 <b>TradeHub executor online</b>\n\n"
                f"Commit: <code>{commit}</code>\n"
                "Live trade alerts are active. Periodic health reports run every few hours."
            )
            if await send_dm(owner, msg):
                db2 = SessionLocal()
                try:
                    db2.execute(_txt("""
                        INSERT INTO owner_notification_dedup (key, sent_at)
                        VALUES (:k, NOW() AT TIME ZONE 'utc')
                        ON CONFLICT (key) DO NOTHING
                    """), {"k": dedup_key})
                    db2.commit()
                finally:
                    db2.close()
                logger.info("[executor] startup Telegram ping delivered to owner")
            else:
                logger.warning("[executor] startup Telegram ping NOT delivered — check OWNER_TELEGRAM_ID + bot token")
        except Exception as e:
            logger.warning(f"[executor] startup ping error: {e}")

    asyncio.create_task(_executor_startup_ping())

    # Legacy Trade-table monitor (TP/SL/breakeven on `trades` rows from signal bot).
    # Previously only ran via tracker_server.py on Replit — was missing on Railway.
    try:
        from app.services.trade_tracker import run_trade_monitor
        asyncio.create_task(_resilient_task("run_trade_monitor", run_trade_monitor, restart_delay=30))
        logger.info("✅ Trade tracker monitor launched (legacy trades table)")
    except Exception as e:
        logger.error(f"Failed to launch trade tracker monitor: {e}")

    # X / Twitter auto-posting — portal worker only (skip on EXECUTOR_ONLY replica).
    if os.environ.get("EXECUTOR_ONLY", "").lower() not in ("1", "true", "yes"):
        try:
            from app.services.twitter_poster import run_auto_post_loop_singleton
            asyncio.create_task(
                _resilient_task(
                    "twitter_auto_post", run_auto_post_loop_singleton, restart_delay=120
                )
            )
            logger.info("✅ X auto-poster launched (advisory-lock single-runner)")
        except Exception as e:
            logger.error(f"Failed to launch X auto-poster: {e}")


async def _executor_claim_loop(first_attempt_delay: int = 0):
    """
    Continuously tries to acquire the executor advisory lock.
    - On startup, one worker wins immediately; others loop here.
    - If the winning worker's DB connection drops, its keepalive returns and
      that worker calls this function again — racing all other workers.
    - Safe: pg_try_advisory_lock is non-blocking and only one worker wins.
    """
    global _executor_running_in_this_worker, _executor_tasks_started
    global _initial_executor_reclaim_done
    loop = asyncio.get_event_loop()

    if first_attempt_delay:
        await asyncio.sleep(first_attempt_delay)

    import os as _os
    _standalone = _os.environ.get("EXECUTOR_STANDALONE", "").lower() in ("1", "true", "yes")

    _wait_attempts = 0
    while True:
        if _executor_running_in_this_worker:
            return  # Already running in this worker — nothing to do

        # Standalone: force-reclaim external ghosts once per process boot only.
        # Re-running on every SSL blip was killing our own session and spawning
        # duplicate executor loops (prefetch 80s → 800s, zero cycle-done lines).
        if _standalone and not _initial_executor_reclaim_done:
            from app.executor_lock import reclaim_executor_lock
            await loop.run_in_executor(None, lambda: reclaim_executor_lock(force=True))
            _initial_executor_reclaim_done = True

        lock_conn = await loop.run_in_executor(None, _try_acquire_executor_lock)
        if lock_conn:
            _executor_running_in_this_worker = True
            logger.info("✅ Executor lock acquired — this worker runs executor + monitors")
            # Start keepalive; when it returns the connection died → re-enter claim loop
            asyncio.create_task(_keepalive_then_reclaim(lock_conn))
            try:
                if not _executor_tasks_started:
                    await _start_executor_tasks()
                    _executor_tasks_started = True
                else:
                    logger.info(
                        "Executor lock re-acquired — restarting feeds + reconcile "
                        "(scan loops still running via _resilient_task)"
                    )
                    await _restart_executor_subsystems_after_lock_reacquire()
            except Exception as e:
                logger.error(f"Failed to start executor tasks: {e}")
            return  # This worker is now the executor; stop trying
        else:
            _wait_attempts += 1
            # Once a worker is the steady-state executor, the sibling sits here
            # forever — log/poll only occasionally (first attempt, then ~every
            # 60s) so the healthy standby state doesn't drown out executor logs.
            _STATUS_EVERY = 12  # 12 × 5s ≈ 60s
            if _wait_attempts == 1 or _wait_attempts % _STATUS_EVERY == 0:
                try:
                    from app.database import SessionLocal
                    from sqlalchemy import text
                    db = SessionLocal()
                    try:
                        row = db.execute(
                            text(
                                "SELECT l.pid, COALESCE(a.state, 'gone') "
                                "FROM pg_locks l "
                                "LEFT JOIN pg_stat_activity a ON a.pid = l.pid "
                                "WHERE l.locktype='advisory' AND l.objid=:lid "
                                "AND l.granted=true LIMIT 1"
                            ),
                            {"lid": get_executor_lock_id()},
                        ).fetchone()
                        if row:
                            # INFO, not WARNING: a sibling holding the lock and
                            # running the executor is the healthy steady state.
                            # Report standby uptime (not a raw retry counter) so a
                            # climbing number never reads as repeated failures.
                            _standby_min = (_wait_attempts * 5) // 60
                            logger.info(
                                f"[executor] healthy — executor active on pid={row[0]} "
                                f"(state={row[1]}); this is the HTTP/standby worker "
                                f"(standby {_standby_min}m, normal)"
                            )
                        else:
                            logger.warning(
                                f"[executor] lock {get_executor_lock_id()} busy but holder unknown "
                                f"— retrying (attempt {_wait_attempts})"
                            )
                    finally:
                        db.close()
                except Exception as _le:
                    logger.warning(f"[executor] lock status check failed: {_le}")
            if _wait_attempts >= _STATUS_EVERY and _wait_attempts % _STATUS_EVERY == 0:
                from app.executor_lock import reclaim_executor_lock
                # Standalone process: force-reclaim any holder (external ghost).
                # Gunicorn siblings: only reclaim genuinely idle zombies so the
                # live executor worker is never terminated.
                n = await loop.run_in_executor(
                    None,
                    lambda: reclaim_executor_lock(force=_standalone),
                )
                if n:
                    logger.warning(
                        f"[executor] reclaimed lock {get_executor_lock_id()} — "
                        f"terminated {n} stale holder(s); retrying acquire"
                    )
            await asyncio.sleep(5)  # Retry every 5 s until lock is free


async def _keepalive_then_reclaim(conn):
    """Keeps the advisory-lock connection alive; when it dies, re-enters the claim loop."""
    global _executor_running_in_this_worker
    await _maintain_advisory_lock(conn)      # blocks until in-thread re-claim exhausted
    _executor_running_in_this_worker = False
    # Gunicorn siblings: stop feeds so the new lock-holder doesn't open duplicate
    # broker sessions. Standalone executor keeps feeds running during re-claim.
    if _lock_loss_should_stop_feeds():
        try:
            from app.services.fmp_price_feed import stop as _fmp_feed_stop
            _fmp_feed_stop()
        except Exception as _fe:
            logger.warning(f"FMP feed stop on lock-loss error (non-fatal): {_fe}")
        try:
            from app.services.ctrader_price_feed import stop as _ctrader_feed_stop
            _ctrader_feed_stop()
        except Exception as _fe:
            logger.warning(f"cTrader feed stop on lock-loss error (non-fatal): {_fe}")
    logger.warning("Advisory lock connection lost — re-entering claim loop")
    asyncio.create_task(_executor_claim_loop(first_attempt_delay=5))


# ── AI Strategy Generator advisory lock (mirror of executor pattern) ───────
_AIGEN_LOCK_ID = 42_424_243
_aigen_running_in_this_worker = False


def _try_acquire_aigen_lock():
    """Same shape as _try_acquire_executor_lock but for the AI generator."""
    try:
        from app.executor_lock import (
            close_lock_connection,
            create_lock_connection,
            try_acquire_lock,
        )

        conn = create_lock_connection()
        if try_acquire_lock(conn, _AIGEN_LOCK_ID):
            return conn
        close_lock_connection(conn)
        return None
    except Exception as e:
        logger.warning(f"[AIGen] advisory lock attempt failed: {e}")
        return None


async def _aigen_keepalive_then_reclaim(conn):
    global _aigen_running_in_this_worker
    await _maintain_advisory_lock(conn)
    _aigen_running_in_this_worker = False
    logger.warning("[AIGen] advisory lock connection lost — re-entering claim loop")
    asyncio.create_task(_aigen_claim_loop(first_attempt_delay=5))


async def _aigen_claim_loop(first_attempt_delay: int = 0):
    """Same pattern as _executor_claim_loop. Only ONE worker wins."""
    global _aigen_running_in_this_worker
    loop = asyncio.get_event_loop()
    if first_attempt_delay:
        await asyncio.sleep(first_attempt_delay)
    while True:
        if _aigen_running_in_this_worker:
            return
        lock_conn = await loop.run_in_executor(None, _try_acquire_aigen_lock)
        if lock_conn:
            _aigen_running_in_this_worker = True
            logger.info("✅ AI Strategy Generator lock acquired — this worker runs the generator")
            asyncio.create_task(_aigen_keepalive_then_reclaim(lock_conn))
            try:
                from app.services.ai_strategy_generator import run_loop as _aigen_loop
                asyncio.create_task(_aigen_loop())
            except Exception as e:
                logger.error(f"[AIGen] failed to start loop: {e}")
                _aigen_running_in_this_worker = False
            return
        else:
            await asyncio.sleep(60)   # retry every minute (slower than executor — less time-critical)


async def _startup_background():
    """Non-critical startup work that runs after the server is already live."""
    import asyncio as _aio
    loop = _aio.get_event_loop()
    try:
        await loop.run_in_executor(None, _ensure_tables)
        logger.info("Schema migrations complete")
    except Exception as e:
        logger.warning(f"Background _ensure_tables error: {e}")

    # Expire any positions stuck OPEN > 48h — these silently block the
    # max_open gate and prevent strategies from ever firing again.
    try:
        from app.services.strategy_executor import close_stale_open_executions
        await close_stale_open_executions(stale_after_hours=48)
    except Exception as e:
        logger.warning(f"close_stale_open_executions error (non-fatal): {e}")

    # ── FMP + cTrader price feeds ─────────────────────────────────────────────
    # NOTE: started in _start_executor_tasks() (the single executor-winning
    # worker), NOT here — per-worker feeds doubled FMP 429s and duplicate
    # cTrader sessions. See _start_executor_tasks for details.

    # Only run the strategy executor in production (Railway/Replit) or FORCE_EXECUTOR=1.
    # In dev, both the dev portal and production share the same Neon DB, so
    # running the executor in dev doubles all API calls and causes confusion.
    import os as _os
    _is_production = is_production_deploy()
    _executor_disabled = _os.environ.get("DISABLE_EXECUTOR", "").lower() in ("1", "true", "yes")
    _gunicorn_executor_disabled = _os.environ.get(
        "DISABLE_EXECUTOR_IN_GUNICORN", ""
    ).lower() in ("1", "true", "yes")
    if _is_production and not _executor_disabled and not _gunicorn_executor_disabled:
        # Each worker enters the claim loop.  The first to win acquires the lock
        # and starts the executor; the rest keep retrying every 30 s so they
        # can take over automatically if the current holder's connection drops.
        # Brief delay so at least one worker is serving HTTP before executor tasks spin up.
        _exec_delay = int(os.getenv("EXECUTOR_START_DELAY", "8"))
        asyncio.create_task(_executor_claim_loop(first_attempt_delay=_exec_delay))
    elif _gunicorn_executor_disabled and _is_production:
        logger.info(
            "Strategy executor in gunicorn SKIPPED (DISABLE_EXECUTOR_IN_GUNICORN=1 — "
            "standalone executor process runs scans)"
        )
    else:
        logger.info("Strategy executor DISABLED (not a production deploy)")

    # AI Strategy Generator — autonomous content engine for the marketplace.
    # Runs only in production (or with ENABLE_AI_GENERATOR=1) so dev never
    # double-runs against the shared Neon DB. Uses its own advisory lock so
    # only ONE gunicorn worker runs the loop (LLM spend + dedupe sanity).
    _aigen_enabled = (
        _is_production
        or _os.environ.get("ENABLE_AI_GENERATOR", "").lower() in ("1", "true", "yes")
    )
    _aigen_disabled = _os.environ.get("DISABLE_AI_GENERATOR", "").lower() in ("1", "true", "yes")
    if _aigen_enabled and not _aigen_disabled:
        asyncio.create_task(_aigen_claim_loop(first_attempt_delay=15))
    else:
        logger.info("AI Strategy Generator DISABLED (dev environment)")


# ─────────────────────────────────────────────────────────────────────────────
# Health check — /health is instant (Railway probe). Deep checks → /health/deep.
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/ping")
async def ping():
    """Lightest possible liveness probe for Railway."""
    return PlainTextResponse("ok")


@app.get("/health")
async def health():
    """Instant liveness probe — no DB or Telegram calls (those blocked workers 30–80s)."""
    _commit = deploy_commit()
    return {
        "status": "ok",
        "ts": int(time.time()),
        "commit": _commit[:12] if _commit else "unknown",
    }


@app.get("/health/deep")
async def health_deep():
    """Full diagnostics: executor locks, Telegram poller, heartbeats."""
    _commit = deploy_commit()
    _tg_lock = {}
    _tg_status = {"configured": False, "poll_disabled": False, "bot_username": None, "bot_id": None}
    try:
        from app.services.telegram_tokens import main_bot_token
        from app.services.telegram_poller_lock import (
            MAIN_POLLER_LOCK_ID,
            FOREX_POLLER_LOCK_ID,
            describe_bot_token,
        )

        _tok = main_bot_token()
        _tg_status["configured"] = bool(_tok)
        _tg_status["poll_disabled"] = os.getenv("DISABLE_TELEGRAM_POLL", "").lower() in (
            "1", "true", "yes",
        )
        if _tok:
            _me = await describe_bot_token(_tok, "main")
            if _me.get("username"):
                _tg_status["bot_username"] = _me["username"]
                _tg_status["bot_id"] = _me.get("id")
            elif _me.get("error"):
                _tg_status["token_error"] = str(_me["error"])[:120]

        from app.database import SessionLocal
        from sqlalchemy import text
        db = SessionLocal()
        try:
            rows = db.execute(
                text(
                    "SELECT l.objid, l.pid, COALESCE(a.state, 'gone') "
                    "FROM pg_locks l "
                    "LEFT JOIN pg_stat_activity a ON a.pid = l.pid "
                    "WHERE l.locktype = 'advisory' AND l.granted = true "
                    "AND l.objid IN (:m, :f)"
                ),
                {"m": MAIN_POLLER_LOCK_ID, "f": FOREX_POLLER_LOCK_ID},
            ).fetchall()
            _tg_lock = {str(r[0]): {"pid": r[1], "state": r[2]} for r in rows}
        finally:
            db.close()
        _tg_status["polling_active"] = str(MAIN_POLLER_LOCK_ID) in _tg_lock
    except Exception as _tg_err:
        _tg_status["error"] = str(_tg_err)[:120]
    _exec_status = {
        "enabled": False,
        "this_process_lock_id": get_executor_lock_id(),
        "locks": {},
    }
    try:
        from app.services.strategy_executor import executor_runtime_profile

        _exec_status["runtime_profile"] = executor_runtime_profile()
        _exec_status["enabled"] = (
            is_production_deploy()
            and os.getenv("DISABLE_EXECUTOR", "").lower() not in ("1", "true", "yes")
        )
        from app.database import SessionLocal
        from sqlalchemy import text
        db = SessionLocal()
        try:
            rows = db.execute(
                text(
                    "SELECT l.objid, l.pid, COALESCE(a.state, 'gone') "
                    "FROM pg_locks l "
                    "LEFT JOIN pg_stat_activity a ON a.pid = l.pid "
                    "WHERE l.locktype='advisory' AND l.granted=true "
                    "AND l.objid IN (:portal, :forex)"
                ),
                {"portal": EXECUTOR_LOCK_ID, "forex": FOREX_EXECUTOR_LOCK_ID},
            ).fetchall()
            for objid, pid, state in rows:
                label = "forex_only" if objid == FOREX_EXECUTOR_LOCK_ID else "portal"
                _exec_status["locks"][label] = {
                    "lock_id": objid,
                    "pid": pid,
                    "state": state,
                }
            _mine = get_executor_lock_id()
            _exec_status["lock_held"] = any(
                r[0] == _mine for r in rows
            )
        finally:
            db.close()
    except Exception as _ex_err:
        _exec_status["error"] = str(_ex_err)[:120]
    if _executor_running_in_this_worker:
        _exec_status["running_in_this_worker"] = True
        try:
            from app.services.strategy_executor import get_heartbeats
            _now = time.time()
            _hb = get_heartbeats()
            _exec_status["heartbeats"] = {
                k: {"age_s": round(_now - ts, 1)} for k, ts in sorted(_hb.items())
            }
        except Exception as _hb_err:
            _exec_status["heartbeat_error"] = str(_hb_err)[:80]
    else:
        _exec_status["running_in_this_worker"] = False
    return {
        "status": "ok",
        "ts": int(time.time()),
        "v": "railway-free-v2-live-forex-feed",
        "commit": _commit[:12] if _commit else "unknown",
        "gold_scan": "yahoo-chart-v3",
        "features_free": portal_features_free(),
        "executor": _exec_status["enabled"],
        "executor_detail": _exec_status,
        "telegram": _tg_status,
        "telegram_poller_locks": _tg_lock,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Public website routes
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def landing_page(request: Request):
    """Marketing landing page — served as static HTML (no Jinja2 processing)."""
    uid = _get_session_uid(request)
    if uid:
        return RedirectResponse(url="/app", status_code=302)
    return FileResponse("app/templates/website.html", media_type="text/html")


@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    uid = _get_session_uid(request)
    if uid:
        # Honour ?next=/some/path so links from /trade etc. round-trip the user
        # back to the page they came from.
        target = _safe_next(request.query_params.get("next"))
        return RedirectResponse(url=target, status_code=302)
    return FileResponse("app/templates/login.html", media_type="text/html")


@app.get("/terms", response_class=HTMLResponse)
async def terms_page():
    return FileResponse("app/templates/terms.html", media_type="text/html")


@app.get("/privacy", response_class=HTMLResponse)
async def privacy_page():
    return FileResponse("app/templates/privacy.html", media_type="text/html")


# ── CallyX brand landing page ─────────────────────────────────────────────────
# Isolated: only accessible via /start or /callyx — no impact on any other route.
_callyx_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "app", "templates", "callyx.html")

@app.get("/start", response_class=HTMLResponse)
@app.get("/callyx", response_class=HTMLResponse)
async def callyx_page():
    return FileResponse(_callyx_file, media_type="text/html")


_CRYPTODICTATOR_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0, viewport-fit=cover">
<meta name="theme-color" content="#efefef">
<title>CryptoDictator</title>
<meta name="description" content="Follow my live trades. Systematic crypto futures on Bitunix.">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap" rel="stylesheet">
<style>
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
body{background:#efefef;font-family:'Inter',-apple-system,BlinkMacSystemFont,sans-serif;-webkit-font-smoothing:antialiased;min-height:100vh;display:flex;justify-content:center}
.page{width:100%;max-width:480px;padding:44px 16px 64px;display:flex;flex-direction:column;align-items:center}
.avatar{width:88px;height:88px;border-radius:50%;background:#fff;border:3px solid #fff;box-shadow:0 2px 12px rgba(0,0,0,0.10);display:flex;align-items:center;justify-content:center;margin-bottom:14px;flex-shrink:0}
.name{font-size:26px;font-weight:800;color:#111;letter-spacing:-0.6px;text-align:center;margin-bottom:10px;line-height:1.2}
.bio{font-size:14px;font-weight:400;color:#555;text-align:center;line-height:1.6;max-width:300px;margin-bottom:18px}
.live{display:inline-flex;align-items:center;gap:6px;font-size:13px;font-weight:500;color:#555;margin-bottom:28px}
.live-dot{width:7px;height:7px;border-radius:50%;background:#22c55e;flex-shrink:0;animation:blink 2.4s ease-in-out infinite}
@keyframes blink{0%,100%{opacity:1}50%{opacity:.4}}
.live strong{color:#16a34a;font-weight:700}
.socials{display:flex;gap:18px;margin-bottom:32px}
.soc-icon{width:36px;height:36px;border-radius:50%;background:#fff;box-shadow:0 1px 4px rgba(0,0,0,0.10);display:flex;align-items:center;justify-content:center;color:#333;text-decoration:none;transition:box-shadow 0.15s,transform 0.12s}
.soc-icon:active{transform:scale(0.92)}
@media(hover:hover){.soc-icon:hover{box-shadow:0 3px 10px rgba(0,0,0,0.16)}}
.links{width:100%;display:flex;flex-direction:column;gap:12px}
.link{width:100%;display:flex;align-items:center;background:#fff;border-radius:16px;box-shadow:0 1px 4px rgba(0,0,0,0.08),0 1px 2px rgba(0,0,0,0.04);text-decoration:none;color:inherit;cursor:pointer;-webkit-tap-highlight-color:transparent;overflow:hidden;transition:box-shadow 0.15s,transform 0.12s;min-height:82px}
.link:active{transform:scale(0.977)}
@media(hover:hover){.link:hover{box-shadow:0 4px 14px rgba(0,0,0,0.13),0 1px 4px rgba(0,0,0,0.06)}}
.thumb{width:72px;height:72px;flex-shrink:0;display:flex;align-items:center;justify-content:center;border-radius:14px 0 0 14px}
.link-body{flex:1;padding:0 12px;display:flex;flex-direction:column;align-items:center;justify-content:center;gap:3px}
.link-title{font-size:15px;font-weight:700;color:#111;text-align:center;letter-spacing:-0.2px;line-height:1.3}
.link-sub{font-size:12px;font-weight:400;color:#888;text-align:center;line-height:1.4}
.dots{width:44px;flex-shrink:0;display:flex;align-items:center;justify-content:center;color:#bbb}
.thumb-green{background:#dcfce7}.thumb-dark{background:#1e293b}.thumb-blue{background:#eff6ff}.thumb-amber{background:#fef9c3}
.foot{margin-top:36px;font-size:11.5px;color:#aaa;text-align:center;line-height:1.9}
.foot a{color:#888;text-decoration:none}.foot a:hover{color:#555}
</style>
</head>
<body>
<div class="page">
  <div class="avatar" style="padding:0;overflow:hidden;">
    <img src="data:image/jpeg;base64,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" alt="CryptoDictator" style="width:100%;height:100%;object-fit:cover;border-radius:50%;display:block;">
  </div>
  <h1 class="name">CryptoDictator</h1>
  <p class="bio">Systematic futures trading on Bitunix.<br>Real trades. No noise.</p>
  <div class="live"><span class="live-dot"></span><span><strong>Live signals</strong> &mdash; every trade posted</span></div>
  <div class="socials">
    <a class="soc-icon" href="https://x.com/cryptodictator" target="_blank" rel="noopener" aria-label="X / Twitter">
      <svg width="16" height="16" viewBox="0 0 16 16" fill="currentColor"><path d="M12.6 1h2.4L9.8 6.8 16 15h-4.6l-3.6-4.7L3.5 15H1l5.6-6.4L0 1h4.7l3.3 4.3L12.6 1Zm-.8 12.6h1.3L4.2 2.3H2.8l9 11.3Z"/></svg>
    </a>
    <a class="soc-icon" href="https://t.me/bnickl88" target="_blank" rel="noopener" aria-label="Telegram">
      <svg width="16" height="16" viewBox="0 0 16 16" fill="currentColor"><path d="M8 0C3.58 0 0 3.58 0 8s3.58 8 8 8 8-3.58 8-8-3.58-8-8-8Zm3.92 5.4-1.36 6.4c-.1.45-.37.56-.74.35L8.1 10.6l-.88.85c-.1.1-.18.18-.37.18l.13-1.87 3.4-3.07c.15-.13-.03-.2-.23-.07L4.33 9.94 2.68 9.43c-.36-.11-.37-.36.08-.53l7.67-2.96c.3-.11.56.07.49.46Z"/></svg>
    </a>
  </div>
  <div class="links">
    <a class="link" href="https://t.me/bnickl88?text=Hey%2C%20I%20found%20you%20through%20your%20link.%20I%27d%20like%20to%20follow%20your%20live%20trades%20%E2%80%94%20how%20do%20I%20get%20access%20to%20the%20free%20group%3F" target="_blank" rel="noopener">
      <div class="thumb thumb-green">
        <svg width="28" height="28" viewBox="0 0 28 28" fill="none"><path d="M4 21L9 12L13.5 18L19 8L24 18" stroke="#16a34a" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"/><circle cx="24" cy="18" r="2.2" fill="#16a34a"/></svg>
      </div>
      <div class="link-body">
        <span class="link-title">Follow My Trades (Free)</span>
        <span class="link-sub">Every signal I take myself &mdash; exact entry, TP &amp; SL shared live</span>
      </div>
      <div class="dots"><svg width="18" height="18" viewBox="0 0 18 18" fill="currentColor"><circle cx="9" cy="4" r="1.5"/><circle cx="9" cy="9" r="1.5"/><circle cx="9" cy="14" r="1.5"/></svg></div>
    </a>
    <a class="link" href="https://t.me/bnickl88?text=Hey%2C%20I%20came%20across%20your%20page%20and%20want%20to%20get%20my%20trades%20running%20on%20automation.%20What%20do%20I%20need%20to%20get%20started%3F" target="_blank" rel="noopener">
      <div class="thumb thumb-dark">
        <svg width="28" height="28" viewBox="0 0 28 28" fill="none"><circle cx="14" cy="14" r="9" stroke="#94a3b8" stroke-width="2"/><path d="M11.5 10.5L19 14L11.5 17.5V10.5Z" fill="#e2e8f0"/></svg>
      </div>
      <div class="link-body">
        <span class="link-title">Start Automation</span>
        <span class="link-sub">AI strategies run 24/7 &mdash; no manual trading needed</span>
      </div>
      <div class="dots"><svg width="18" height="18" viewBox="0 0 18 18" fill="currentColor"><circle cx="9" cy="4" r="1.5"/><circle cx="9" cy="9" r="1.5"/><circle cx="9" cy="14" r="1.5"/></svg></div>
    </a>
    <a class="link" href="https://tradehubmarkets.com" target="_blank" rel="noopener">
      <div class="thumb thumb-blue">
        <svg width="28" height="28" viewBox="0 0 28 28" fill="none"><rect x="3" y="17" width="5" height="8" rx="1.2" stroke="#3b82f6" stroke-width="1.8"/><rect x="11.5" y="11" width="5" height="14" rx="1.2" stroke="#3b82f6" stroke-width="1.8"/><rect x="20" y="4" width="5" height="21" rx="1.2" stroke="#3b82f6" stroke-width="1.8"/></svg>
      </div>
      <div class="link-body">
        <span class="link-title">Strategy Builder</span>
        <span class="link-sub">Build, backtest &amp; automate your own strategies</span>
      </div>
      <div class="dots"><svg width="18" height="18" viewBox="0 0 18 18" fill="currentColor"><circle cx="9" cy="4" r="1.5"/><circle cx="9" cy="9" r="1.5"/><circle cx="9" cy="14" r="1.5"/></svg></div>
    </a>
    <a class="link" href="https://www.bitunix.com/partners/fnJr" target="_blank" rel="noopener">
      <div class="thumb thumb-amber">
        <svg width="28" height="28" viewBox="0 0 28 28" fill="none"><path d="M14 2.5L24 7.5v13L14 25.5 4 20.5v-13L14 2.5Z" stroke="#d97706" stroke-width="1.8" stroke-linejoin="round"/><path d="M9.5 14h9M14 9.5V18.5" stroke="#d97706" stroke-width="1.8" stroke-linecap="round"/></svg>
      </div>
      <div class="link-body">
        <span class="link-title">Open Bitunix &mdash; Save 70% on Fees</span>
        <span class="link-sub">Exclusive affiliate link &mdash; lower fees, instant signup</span>
      </div>
      <div class="dots"><svg width="18" height="18" viewBox="0 0 18 18" fill="currentColor"><circle cx="9" cy="4" r="1.5"/><circle cx="9" cy="9" r="1.5"/><circle cx="9" cy="14" r="1.5"/></svg></div>
    </a>
  </div>
  <p class="foot">CryptoDictator &middot; <a href="https://tradehubmarkets.com">TradeHub Markets</a><br><a href="https://tradehubmarkets.com/terms">Terms &amp; Conditions</a></p>
</div>
</body>
</html>"""

_cd_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "app", "templates", "cryptodictator.html")
try:
    os.makedirs(os.path.dirname(_cd_file), exist_ok=True)
    with open(_cd_file, "w", encoding="utf-8") as _fh:
        _fh.write(_CRYPTODICTATOR_HTML)
except Exception as _e:
    logger.warning(f"Could not write cryptodictator.html: {_e}")

@app.get("/cryptodictator", response_class=HTMLResponse)
@app.get("/cd", response_class=HTMLResponse)
async def cryptodictator_page():
    return FileResponse(_cd_file, media_type="text/html")


# ── LarkNexus affiliate landing page ──────────────────────────────────────────
_ln_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "app", "templates", "larknexus.html")

@app.get("/larknexus", response_class=HTMLResponse)
@app.get("/ln", response_class=HTMLResponse)
@app.get("/lark", response_class=HTMLResponse)
async def larknexus_page():
    return FileResponse(_ln_file, media_type="text/html")


_ASSET_LANDING_PAGES = {
    "forex": {
        "slug": "forex",
        "page_title": "TradeHub — Forex Strategy Builder & Live cTrader Execution",
        "meta_description": "Build forex strategies in plain English, paper-test on real ticks, go live on FP Markets via cTrader. Structure scanner + gold-grade execution.",
        "hero_emoji": "💱",
        "hero_label": "Forex · cTrader live · Paper-first",
        "hero_title": "Build forex strategies.<br>Execute on <span style=\"color:var(--green)\">cTrader</span> in minutes.",
        "hero_sub": "Describe entries in plain English, scan London/NY sessions with the live structure finder, paper-test on broker-matched prices, then connect FP Markets and go live — no Pine Script, no VPS.",
        "accent": "#22c55e",
        "lb_asset_class": "forex",
        "lb_market": "",
        "features": [
            ("Structure scanner", "BOS, CHoCH, FVG, order blocks — scanned across major pairs every few seconds."),
            ("cTrader live link", "OAuth to FP Markets demo or live. Breakeven & trailing SL in under a second."),
            ("Session filters", "London, NY, Asia — only trade when your edge is statistically active."),
            ("Paper realism", "1-minute OHLC + live spot append so TP/SL hits match broker behaviour."),
        ],
        "build_cta": "/login?next=/app%23build",
    },
    "gold": {
        "slug": "gold",
        "page_title": "TradeHub — Gold (XAUUSD) Strategy Builder",
        "meta_description": "Discover and automate XAUUSD strategies. Gold Strategy Finder, paper testing, live cTrader execution.",
        "hero_emoji": "🥇",
        "hero_label": "XAUUSD · Metals · Sub-second SL management",
        "hero_title": "Automate <span style=\"color:#f59e0b\">gold</span> strategies<br>without writing code.",
        "hero_sub": "Run the Gold Strategy Finder across timeframes, backtest on real XAU candles, paper-trade with live spot detection, then push orders to cTrader when you're ready.",
        "accent": "#f59e0b",
        "lb_asset_class": "",
        "lb_market": "gold",
        "features": [
            ("Gold Strategy Finder", "Scans XAUUSD across TFs and ranks setups by backtested edge."),
            ("Live spot TP/SL", "Synthetic intra-candle prices from cTrader — no waiting for candle close."),
            ("Fast SL manager", "Dedicated 1s loop amends breakeven/trailing stops on the broker."),
            ("Risk presets", "Pips-based TP/SL, lot sizing, Friday close & no-overnight guards."),
        ],
        "build_cta": "/login?next=/app%23build",
    },
    "indices": {
        "slug": "indices",
        "page_title": "TradeHub — Index CFD Strategy Builder (NAS100, US30)",
        "meta_description": "Build and automate index strategies — NASDAQ, US30, and more. Paper test then trade via cTrader.",
        "hero_emoji": "📊",
        "hero_label": "NAS100 · US30 · Index CFDs",
        "hero_title": "Index strategies.<br><span style=\"color:#a855f7\">Backtested. Automated. Tracked.</span>",
        "hero_sub": "Use the Index Strategy Finder on NAS100 and US30, paper-test with points-based P&L, and connect cTrader for live index CFD execution.",
        "accent": "#a855f7",
        "lb_asset_class": "index",
        "lb_market": "",
        "features": [
            ("Index Strategy Finder", "Discovers edge on NAS100, US30 and major indices from live data."),
            ("Points-based risk", "TP/SL in index points — matches how CFD books quote indices."),
            ("Multi-source prices", "cTrader ticks with FMP/Yahoo fallbacks when markets are closed."),
            ("Same portal", "One dashboard for crypto, forex, gold and indices — no separate tools."),
        ],
        "build_cta": "/login?next=/app%23build",
    },
}


def _asset_landing_response(request: Request, key: str):
    ctx = {"request": request, **_ASSET_LANDING_PAGES[key]}
    return templates.TemplateResponse("asset_landing.html", ctx)


@app.get("/forex", response_class=HTMLResponse)
async def forex_landing_page(request: Request):
    return _asset_landing_response(request, "forex")


@app.get("/gold", response_class=HTMLResponse)
@app.get("/xau", response_class=HTMLResponse)
async def gold_landing_page(request: Request):
    return _asset_landing_response(request, "gold")


@app.get("/indices", response_class=HTMLResponse)
@app.get("/index", response_class=HTMLResponse)
async def indices_landing_page(request: Request):
    return _asset_landing_response(request, "indices")


@app.post("/login")
async def login_submit(request: Request):
    """Verify UID. Two-step: first call validates UID and signals whether a password
    is needed. Second call (uid + password) verifies the password and logs in."""
    from app.database import SessionLocal
    body = await request.json()
    uid      = (body.get("uid")      or "").strip().upper()
    password = (body.get("password") or "").strip()

    if not uid.startswith("TH-") or len(uid) < 6:
        raise HTTPException(status_code=400, detail="Invalid access code format")

    def _do_lookup():
        db = SessionLocal()
        try:
            u = _get_user_by_uid(uid, db)
            if not u:
                return None, False, False
            return u.uid, u.banned, bool(u.password_hash)
        finally:
            db.close()

    uid_found, banned, has_pw = await asyncio.to_thread(_do_lookup)
    if not uid_found:
        raise HTTPException(status_code=403, detail="Access code not found. Send /start to the Telegram bot to get your code — it appears at the top of the bot dashboard.")
    if banned:
        raise HTTPException(status_code=403, detail="This account has been suspended.")

    # Step 1 — no password supplied: tell the frontend what to show next
    if not password:
        if not has_pw:
            return JSONResponse({"needs_password": True})
        else:
            return JSONResponse({"enter_password": True})

    # Step 2 — password supplied
    if not has_pw:
        # They shouldn't reach here without using uid-set-password, but handle gracefully
        raise HTTPException(status_code=400, detail="Please use the Create Password form.")

    def _verify():
        db = SessionLocal()
        try:
            u = _get_user_by_uid(uid, db)
            return u and _verify_password(password, u.password_hash)
        finally:
            db.close()

    ok = await asyncio.to_thread(_verify)
    if not ok:
        raise HTTPException(status_code=403, detail="Incorrect password.")

    resp = JSONResponse({"redirect": _safe_next(body.get("next"))})
    _set_session(resp, uid, request)
    return resp


@app.post("/login/uid-set-password")
async def login_uid_set_password(request: Request):
    """Set a password for a UID-only account (first time), then log them in."""
    from app.database import SessionLocal
    body     = await request.json()
    uid      = (body.get("uid")      or "").strip().upper()
    password = (body.get("password") or "").strip()

    if not uid.startswith("TH-") or len(uid) < 6:
        raise HTTPException(status_code=400, detail="Invalid access code format")
    if len(password) < 8:
        raise HTTPException(status_code=400, detail="Password must be at least 8 characters.")

    def _set_pw():
        db = SessionLocal()
        try:
            u = _get_user_by_uid(uid, db)
            if not u:
                return "not_found"
            if u.banned:
                return "banned"
            if u.password_hash:
                return "already_set"
            u.password_hash = _hash_password(password)
            db.commit()
            return "ok"
        finally:
            db.close()

    result = await asyncio.to_thread(_set_pw)
    if result == "not_found":
        raise HTTPException(status_code=403, detail="Access code not found.")
    if result == "banned":
        raise HTTPException(status_code=403, detail="This account has been suspended.")
    if result == "already_set":
        raise HTTPException(status_code=400, detail="Password already set — use Change Password in Settings.")

    resp = JSONResponse({"redirect": _safe_next(body.get("next"))})
    _set_session(resp, uid, request)
    return resp


@app.post("/login/email-request")
async def login_email_request(request: Request):
    """Step 1: user submits email — if found, send OTP via Telegram."""
    from app.database import SessionLocal
    body = await request.json()
    email = (body.get("email") or "").strip().lower()
    if not email or "@" not in email:
        raise HTTPException(status_code=400, detail="Please enter a valid email address.")

    def _find_user():
        db = SessionLocal()
        try:
            u = db.query(User).filter(User.email == email).first()
            if not u:
                return None
            return {"banned": u.banned, "telegram_id": u.telegram_id,
                    "first_name": u.first_name, "username": u.username}
        finally:
            db.close()

    udata = await asyncio.to_thread(_find_user)
    if not udata:
        raise HTTPException(
            status_code=404,
            detail="No account found with that email. Link your email via the Telegram bot first — type /setemail your@email.com"
        )
    if udata["banned"]:
        raise HTTPException(status_code=403, detail="This account has been suspended.")
    otp = _generate_otp()
    _otp_store[email] = (otp, datetime.utcnow() + timedelta(minutes=_OTP_TTL_MINUTES))
    await _send_otp_via_telegram(
        udata["telegram_id"],
        otp,
        udata["first_name"] or udata["username"] or ""
    )
    return JSONResponse({"ok": True, "message": "Code sent to your Telegram. Check your messages!"})


@app.post("/login/email-verify")
async def login_email_verify(request: Request):
    """Step 2: user submits email + OTP — verify and set session."""
    from app.database import SessionLocal
    body = await request.json()
    email = (body.get("email") or "").strip().lower()
    code = (body.get("code") or "").strip()
    if not email or not code:
        raise HTTPException(status_code=400, detail="Email and code are required.")
    if not _otp_valid(email, code):
        raise HTTPException(status_code=403, detail="Invalid or expired code. Request a new one.")
    _otp_store.pop(email, None)

    def _find_user():
        db = SessionLocal()
        try:
            u = db.query(User).filter(User.email == email).first()
            if not u:
                return None
            return {"uid": u.uid, "banned": u.banned}
        finally:
            db.close()

    udata = await asyncio.to_thread(_find_user)
    if not udata or not udata["uid"]:
        raise HTTPException(status_code=403, detail="Account not found.")
    if udata["banned"]:
        raise HTTPException(status_code=403, detail="This account has been suspended.")
    resp = JSONResponse({"redirect": _safe_next(body.get("next"))})
    _set_session(resp, udata["uid"], request)
    return resp


@app.get("/logout")
async def logout():
    resp = RedirectResponse(url="/login", status_code=302)
    resp.delete_cookie(_COOKIE_NAME)
    return resp


# ── Email/password registration ────────────────────────────────────────────────
@app.post("/register")
async def register_submit(request: Request):
    """Create a new account with email + password."""
    from app.database import SessionLocal
    body = await request.json()
    email = (body.get("email") or "").strip().lower()
    password = (body.get("password") or "").strip()
    name = (body.get("name") or "").strip()
    ref_code = (body.get("ref_code") or "").strip().upper()

    if not email or "@" not in email:
        raise HTTPException(status_code=400, detail="Please enter a valid email address.")
    if len(password) < 8:
        raise HTTPException(status_code=400, detail="Password must be at least 8 characters.")
    if not name:
        raise HTTPException(status_code=400, detail="Please enter your name.")

    password_hash = _hash_password(password)

    def _do_register():
        import random as _rand, string as _str
        db = SessionLocal()
        try:
            if db.query(User).filter(User.email == email).first():
                return {"error": "exists"}
            uid = _generate_web_uid(db)
            web_tid = f"WEB-{secrets.token_hex(8).upper()}"
            user = User(
                telegram_id=web_tid,
                uid=uid,
                email=email,
                email_verified=False,
                password_hash=password_hash,
                first_name=name,
                auth_provider="email",
                approved=True,
            )
            db.add(user)
            db.commit()
            db.refresh(user)
            if not user.referral_code:
                _chars = _str.ascii_uppercase + _str.digits
                for _ in range(20):
                    _code = "REF-" + "".join(_rand.choices(_chars, k=6))
                    if not db.query(User).filter(User.referral_code == _code).first():
                        user.referral_code = _code
                        break
                db.commit()
            referrer_info = None
            if ref_code:
                referrer = db.query(User).filter(User.referral_code == ref_code).first()
                if referrer and referrer.id != user.id:
                    user.referred_by = ref_code
                    db.commit()
                    referrer_info = {
                        "name": referrer.first_name or referrer.username or "User",
                        "uid": referrer.uid or "",
                    }
            return {"uid": user.uid, "name": user.first_name or name, "referrer": referrer_info}
        finally:
            db.close()

    result = await asyncio.to_thread(_do_register)
    if result.get("error") == "exists":
        raise HTTPException(status_code=409, detail="An account with this email already exists. Please sign in.")
    if result.get("referrer"):
        import asyncio as _aio
        _aio.create_task(_notify_admin_referral(
            new_name=result.get("name", "User"),
            new_uid=result.get("uid", ""),
            referrer_name=result["referrer"]["name"],
            referrer_uid=result["referrer"]["uid"],
        ))
    resp = JSONResponse({"redirect": _safe_next(body.get("next"))})
    _set_session(resp, result["uid"], request)
    return resp


def _generate_web_uid(db) -> str:
    """Generate a unique TH- UID that doesn't conflict with existing ones."""
    import string, random
    chars = string.ascii_uppercase + string.digits
    for _ in range(20):
        uid = "TH-" + "".join(random.choices(chars, k=8))
        if not db.query(User).filter(User.uid == uid).first():
            return uid
    raise ValueError("Could not generate unique UID")


# ── Email/password sign-in ─────────────────────────────────────────────────────
@app.post("/login/password")
async def login_password(request: Request):
    """Sign in with email and password."""
    from app.database import SessionLocal
    body = await request.json()
    email = (body.get("email") or "").strip().lower()
    password = (body.get("password") or "").strip()

    if not email or not password:
        raise HTTPException(status_code=400, detail="Email and password are required.")

    def _check_password():
        db = SessionLocal()
        try:
            u = db.query(User).filter(User.email == email).first()
            if not u or not u.password_hash:
                return {"error": "no_account"}
            if not _verify_password(password, u.password_hash):
                return {"error": "bad_password"}
            if u.banned:
                return {"error": "banned"}
            if not u.uid:
                return {"error": "incomplete"}
            return {"uid": u.uid}
        finally:
            db.close()

    result = await asyncio.to_thread(_check_password)
    err = result.get("error")
    if err == "no_account":
        raise HTTPException(status_code=403, detail="No account found with that email. Did you sign up with Google?")
    if err == "bad_password":
        raise HTTPException(status_code=403, detail="Incorrect password.")
    if err == "banned":
        raise HTTPException(status_code=403, detail="This account has been suspended.")
    if err == "incomplete":
        raise HTTPException(status_code=403, detail="Account setup incomplete. Please contact support.")
    resp = JSONResponse({"redirect": _safe_next(body.get("next"))})
    _set_session(resp, result["uid"], request)
    return resp


# ── Telegram Login Widget ──────────────────────────────────────────────────────
@app.post("/login/telegram")
async def login_telegram(request: Request):
    """Verify Telegram Login Widget payload and create a session."""
    from app.database import SessionLocal
    from app.config import settings

    data = await request.json()
    check_hash = data.get("hash", "")
    if not check_hash:
        raise HTTPException(status_code=400, detail="Missing authentication data.")

    # Build the data-check string (all fields except 'hash', sorted, key=value joined by \n)
    data_check = {k: str(v) for k, v in data.items() if k != "hash"}
    data_check_string = "\n".join(f"{k}={v}" for k, v in sorted(data_check.items()))

    # Verify HMAC: secret = SHA256(bot_token), hash = HMAC-SHA256(secret, data_check_string)
    secret_key = hashlib.sha256(settings.TELEGRAM_BOT_TOKEN.encode()).digest()
    expected = hmac.new(secret_key, data_check_string.encode(), hashlib.sha256).hexdigest()
    if not hmac.compare_digest(expected, check_hash):
        raise HTTPException(status_code=403, detail="Invalid Telegram authentication signature.")

    # Reject stale auth (older than 24 hours)
    auth_date = int(data.get("auth_date", 0))
    if time.time() - auth_date > 86400:
        raise HTTPException(status_code=403, detail="Authentication expired — please try again.")

    # Look up user by their Telegram ID
    telegram_id = str(data.get("id", ""))

    def _find_tg_user():
        db = SessionLocal()
        try:
            u = db.query(User).filter(User.telegram_id == telegram_id).first()
            if not u:
                return None
            if not u.uid:
                u.uid = _generate_web_uid(db)
                db.commit()
                logger.info(f"Auto-assigned UID {u.uid} to existing user tg_id={telegram_id}")
            return {"uid": u.uid, "banned": u.banned, "approved": u.approved,
                    "username": u.username}
        finally:
            db.close()

    udata = await asyncio.to_thread(_find_tg_user)
    if not udata:
        raise HTTPException(
            status_code=404,
            detail="No TradeHub account found for this Telegram account. "
                   "Open the bot and send /start first, then try again."
        )
    if udata["banned"]:
        raise HTTPException(status_code=403, detail="This account has been suspended.")
    if not udata["approved"]:
        raise HTTPException(status_code=403, detail="Your account is pending approval.")

    logger.info(f"Telegram login: uid={udata['uid']} tg_id={telegram_id} username=@{udata['username']}")
    resp = JSONResponse({"redirect": _safe_next(data.get("next"))})
    _set_session(resp, udata["uid"], request)
    return resp


# ── Google OAuth ───────────────────────────────────────────────────────────────
@app.get("/auth/google")
async def google_auth_start(request: Request, ref: str = ""):
    """Redirect user to Google's OAuth consent screen."""
    if not _google_enabled():
        return RedirectResponse(url="/login?error=google_not_configured", status_code=302)
    state = secrets.token_urlsafe(16)
    params = {
        "client_id": _GOOGLE_CLIENT_ID,
        "redirect_uri": _google_redirect_uri(request),
        "response_type": "code",
        "scope": _GOOGLE_SCOPE,
        "state": state,
        "access_type": "online",
        "prompt": "select_account",
    }
    url = _GOOGLE_AUTH_URL + "?" + urllib.parse.urlencode(params)
    resp = RedirectResponse(url=url, status_code=302)
    resp.set_cookie("google_oauth_state", state, max_age=600, httponly=True, samesite="lax")
    # Persist referral code through the OAuth round-trip
    if ref:
        resp.set_cookie("google_oauth_ref", ref.strip().upper(), max_age=600, httponly=True, samesite="lax")
    return resp


@app.get("/auth/google/callback")
async def google_auth_callback(request: Request, code: str = "", state: str = "", error: str = ""):
    """Handle Google's redirect back after user approves."""
    if error:
        return RedirectResponse(url="/login?error=google_denied", status_code=302)
    stored_state = request.cookies.get("google_oauth_state", "")
    if not state or not secrets.compare_digest(state, stored_state):
        return RedirectResponse(url="/login?error=invalid_state", status_code=302)

    import httpx
    from app.database import SessionLocal

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            # Exchange code for tokens
            token_resp = await client.post(_GOOGLE_TOKEN_URL, data={
                "code": code,
                "client_id": _GOOGLE_CLIENT_ID,
                "client_secret": _GOOGLE_CLIENT_SECRET,
                "redirect_uri": _google_redirect_uri(request),
                "grant_type": "authorization_code",
            })
            if token_resp.status_code != 200:
                logger.error(f"Google token exchange failed: {token_resp.text[:300]}")
                return RedirectResponse(url="/login?error=google_token_failed", status_code=302)
            tokens = token_resp.json()
            access_token = tokens.get("access_token")

            # Fetch user info from Google
            info_resp = await client.get(
                _GOOGLE_USERINFO_URL,
                headers={"Authorization": f"Bearer {access_token}"}
            )
            if info_resp.status_code != 200:
                return RedirectResponse(url="/login?error=google_info_failed", status_code=302)
            ginfo = info_resp.json()

        google_id = str(ginfo.get("id", ""))
        email = (ginfo.get("email") or "").lower()
        name = ginfo.get("name") or ginfo.get("given_name") or ""

        if not google_id or not email:
            return RedirectResponse(url="/login?error=google_no_email", status_code=302)

        # Read referral code stored before the OAuth round-trip
        oauth_ref = (request.cookies.get("google_oauth_ref") or "").strip().upper()

        db = SessionLocal()
        try:
            # Find by google_id or email
            user = db.query(User).filter(User.google_id == google_id).first()
            if not user:
                user = db.query(User).filter(User.email == email).first()

            if user:
                # Update google_id if not set
                if not user.google_id:
                    user.google_id = google_id
                    user.email_verified = True
                    db.commit()
            else:
                # Apply referral if a valid referrer exists
                referred_by_code = None
                if oauth_ref:
                    _referrer = db.query(User).filter(User.referral_code == oauth_ref).first()
                    if _referrer:
                        referred_by_code = oauth_ref

                # Create new account
                uid = _generate_web_uid(db)
                web_tid = f"WEB-{secrets.token_hex(8).upper()}"
                user = User(
                    telegram_id=web_tid,
                    uid=uid,
                    email=email,
                    email_verified=True,
                    google_id=google_id,
                    first_name=name,
                    auth_provider="google",
                    approved=True,
                    referred_by=referred_by_code,
                )
                db.add(user)
                db.commit()
                db.refresh(user)

            if user.banned:
                return RedirectResponse(url="/login?error=banned", status_code=302)

            resp = RedirectResponse(url="/app", status_code=302)
            _set_session(resp, user.uid, request)
            resp.delete_cookie("google_oauth_state")
            return resp
        finally:
            db.close()

    except Exception as e:
        logger.error(f"Google OAuth callback error: {e}")
        return RedirectResponse(url="/login?error=google_error", status_code=302)


# ─────────────────────────────────────────────────────────────────────────────
# /app  — cookie-session authenticated portal (same as /strategies but web)
# ─────────────────────────────────────────────────────────────────────────────

def _load_portal_data(uid: str):
    """Sync helper — runs DB work in a thread pool thread."""
    from app.database import SessionLocal
    from sqlalchemy import text
    from datetime import datetime, timedelta
    db = SessionLocal()
    try:
        # Use the resilient + cached lookup — this is the #1 source of /app
        # 500s under Neon load, and the cache means a logged-in user's hot
        # path doesn't even touch the DB for their user row.
        user = _get_user_by_uid_safe(uid, db)
        if not user:
            return None
        if user.banned:
            return "banned"

        # Strategy counts for SSR stats strip (lightweight — no per-strategy data needed)
        strat_counts = db.execute(text("""
            SELECT
                COUNT(*) AS total,
                COUNT(*) FILTER (WHERE status = 'active') AS active_count,
                COUNT(*) FILTER (WHERE status = 'paper')  AS paper_count
            FROM user_strategies WHERE user_id = :uid
        """), {"uid": user.id}).fetchone()

        strategy_data = []  # Cards loaded via JS — no SSR card rendering needed

        # ── Portfolio stats — single aggregate SQL query ──
        cutoff_7d = datetime.utcnow() - timedelta(days=7)
        agg = db.execute(text("""
            SELECT
                COALESCE(SUM(CASE WHEN e.outcome = 'OPEN' THEN 1 ELSE 0 END), 0)                                              AS open_trades,
                COALESCE(SUM(CASE WHEN e.outcome IN ('WIN','LOSS','BREAKEVEN') AND e.pnl_pct IS NOT NULL THEN 1 ELSE 0 END), 0) AS total_trades,
                COALESCE(SUM(CASE WHEN e.outcome IN ('WIN','LOSS')             AND e.pnl_pct IS NOT NULL THEN 1 ELSE 0 END), 0) AS decisive_trades,
                COALESCE(SUM(CASE WHEN e.outcome = 'WIN' THEN 1 ELSE 0 END), 0)                                               AS wins,
                COALESCE(SUM(CASE WHEN e.outcome IN ('WIN','LOSS','BREAKEVEN') AND e.pnl_pct IS NOT NULL
                                  AND COALESCE(e.closed_at, e.fired_at) > :cutoff
                             THEN e.pnl_pct ELSE 0 END), 0)                                                                   AS pnl_7d,
                COALESCE(SUM(CASE WHEN e.outcome IN ('WIN','LOSS','BREAKEVEN') AND e.pnl_pct IS NOT NULL
                             THEN e.pnl_pct ELSE 0 END), 0)                                                                   AS pnl_all
            FROM strategy_executions e
            JOIN user_strategies s ON s.id = e.strategy_id
            WHERE s.user_id = :uid
        """), {"uid": user.id, "cutoff": cutoff_7d}).fetchone()

        open_trades = int(agg.open_trades)
        total       = int(agg.total_trades)
        decisive    = int(agg.decisive_trades)
        wins        = int(agg.wins)
        pnl_7d      = float(agg.pnl_7d)
        pnl_all     = float(agg.pnl_all)
        # Win rate denominator excludes BREAKEVEN — breakevens are neutral
        # outcomes and shouldn't drag the win rate down (matches executor).
        win_rate    = round(wins / decisive * 100, 1) if decisive > 0 else 0

        total_strats  = int(strat_counts.total)   if strat_counts else 0
        active_count  = int(strat_counts.active_count) if strat_counts else 0
        paper_count   = int(strat_counts.paper_count)  if strat_counts else 0

        def _fmt_pnl(v):
            r = round(v)
            return ('+' if r > 0 else '') + str(r) + '%'

        portfolio = {
            "total_strategies": total_strats,
            "active_count":     active_count,
            "paper_count":      paper_count,
            "open_trades":      open_trades,
            "total_trades":     total,
            "win_rate":         win_rate,
            "pnl_7d_fmt":       _fmt_pnl(pnl_7d),
            "pnl_all_fmt":      _fmt_pnl(pnl_all),
            "pnl_7d_pos":       pnl_7d >= 0,
            "pnl_all_pos":      pnl_all >= 0,
            "pnl_7d":           round(pnl_7d, 2),
            "pnl_all":          round(pnl_all, 2),
        }

        _psub = _get_portal_sub(user.id, db)
        _is_pro = _is_portal_pro(_psub) or bool(getattr(user, "is_admin", False))
        is_web_user = str(getattr(user, "telegram_id", "") or "").startswith("WEB-")

        return {
            "user":        user,
            "strategies":  strategy_data,
            "portfolio":   portfolio,
            "is_web_user": is_web_user,
            "is_pro":      _is_pro,
        }
    finally:
        db.close()


async def _render_portal(request: Request, uid: str):
    """Serves the portal HTML shell INSTANTLY — zero DB calls.
    All personalised data (name, plan, stats) loads client-side via /api/me + /api/portfolio."""
    _default_user = {"first_name": "", "username": "", "uid": uid}
    _default_portfolio = {
        "total_strategies": 0, "active_count": 0, "paper_count": 0,
        "open_trades": 0, "pnl_7d_fmt": "—", "pnl_all_fmt": "—",
        "pnl_7d_pos": True, "pnl_all_pos": True, "win_rate": 0, "total_trades": 0,
    }
    _free = portal_features_free()
    uid = (uid or "").strip().upper()
    response = templates.TemplateResponse(request, "strategy_portal.html", {
        "user":          _default_user,
        "uid":           uid,
        "strategies":    [],
        "portfolio":     _default_portfolio,
        "is_web_user":   False,
        "is_pro":        _free,
        "features_free": _free,
        "payments_enabled": payments_enabled(),
    })
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response


@app.get("/app", response_class=HTMLResponse)
async def app_page(request: Request):
    """Cookie-session authenticated app entry point."""
    uid = _get_session_uid(request)
    if not uid:
        return RedirectResponse(url="/login", status_code=302)
    return await _render_portal(request, uid)


@app.get("/trades", response_class=HTMLResponse)
async def trades_page(request: Request):
    """Direct entry point that opens the portal on the global Trades feed.

    Mirrors the mobile app's "Trades" bottom-tab. Reuses the same template +
    auth gate as /app — the template reads ?p=trades on init and switches
    pages without a round-trip.
    """
    uid = _get_session_uid(request)
    if not uid:
        return RedirectResponse(url="/login?next=/trades", status_code=302)
    return RedirectResponse(url="/app?p=trades", status_code=302)


# ─────────────────────────────────────────────────────────────────────────────
# Day Trading page — public BTC chart with order block overlays
# ─────────────────────────────────────────────────────────────────────────────

TRADE_SYMBOL_WHITELIST = {
    # Top-50 by market-cap (stablecoins & wrapped tokens excluded).
    # All pairs are MEXC USDT spot.
    "BTC":   "BTCUSDT",
    "ETH":   "ETHUSDT",
    "BNB":   "BNBUSDT",
    "SOL":   "SOLUSDT",
    "XRP":   "XRPUSDT",
    "DOGE":  "DOGEUSDT",
    "ADA":   "ADAUSDT",
    "TRX":   "TRXUSDT",
    "TON":   "TONUSDT",
    "AVAX":  "AVAXUSDT",
    "SHIB":  "SHIBUSDT",
    "DOT":   "DOTUSDT",
    "LINK":  "LINKUSDT",
    "BCH":   "BCHUSDT",
    "LTC":   "LTCUSDT",
    "NEAR":  "NEARUSDT",
    "UNI":   "UNIUSDT",
    "APT":   "APTUSDT",
    "OP":    "OPUSDT",
    "ICP":   "ICPUSDT",
    "XLM":   "XLMUSDT",
    "ATOM":  "ATOMUSDT",
    "FIL":   "FILUSDT",
    "ETC":   "ETCUSDT",
    "HBAR":  "HBARUSDT",
    "ARB":   "ARBUSDT",
    "VET":   "VETUSDT",
    "ALGO":  "ALGOUSDT",
    "MATIC": "MATICUSDT",
    "GRT":   "GRTUSDT",
    "FTM":   "FTMUSDT",
    "AAVE":  "AAVEUSDT",
    "STX":   "STXUSDT",
    "THETA": "THETAUSDT",
    "FLOW":  "FLOWUSDT",
    "CRV":   "CRVUSDT",
    "LDO":   "LDOUSDT",
    "RUNE":  "RUNEUSDT",
    "INJ":   "INJUSDT",
    "SUI":   "SUIUSDT",
    "SEI":   "SEIUSDT",
    "JUP":   "JUPUSDT",
    "ENA":   "ENAUSDT",
    "WIF":   "WIFUSDT",
    "PEPE":  "PEPEUSDT",
    "BONK":  "BONKUSDT",
    "AR":    "ARUSDT",
    "KAS":   "KASUSDT",
    "MKR":   "MKRUSDT",
    "RENDER":"RENDERUSDT",
}
_TRADE_TF_MAP = {"1m": "1m", "5m": "5m", "15m": "15m", "1h": "60m"}  # MEXC uses '60m' not '1h'


@app.get("/trade", response_class=HTMLResponse)
async def trade_page_default(request: Request):
    """Public day-trading page — defaults to BTC."""
    return FileResponse("app/templates/trade.html", media_type="text/html")


@app.get("/trade/{slug}", response_class=HTMLResponse)
async def trade_page_symbol(slug: str, request: Request):
    """Public day-trading page for a specific symbol (e.g. /trade/eth)."""
    sym = (slug or "").upper().strip()
    if sym not in TRADE_SYMBOL_WHITELIST:
        return RedirectResponse(url="/trade", status_code=302)
    return FileResponse("app/templates/trade.html", media_type="text/html")


def _build_mobile_login_response(user, db) -> dict:
    """Shape the user record into the canonical mobile login payload.
    Shared between UID and email login paths so they return identical fields.

    Uses the SAME pro-tier check as the web portal (`_is_portal_pro` against
    the row returned by `_get_portal_sub`) so mobile + web can never disagree
    about a user's plan. Previously this filtered by `status == "active"`
    which incorrectly classified users with NULL/missing status as free even
    when their tier was `pro` and subscription_end was in the future.
    """
    display = (user.username or user.first_name or user.uid or "").strip() or user.uid
    is_admin = (user.uid == "TH-YP0BADA8") or bool(getattr(user, "is_admin", False))
    entitled = is_admin   # admins are always treated as Pro for entitlement
    try:
        sub = _get_portal_sub(user.id, db)
        if _is_portal_pro(sub):
            entitled = True
    except Exception as exc:
        logger.warning(f"[mobile login] portal-sub lookup failed for {user.uid}: {exc}")
    # `plan` is purely informational — never derive entitlement from it.
    # An expired pro sub keeps tier="pro" but should report as free here so
    # the UI cannot accidentally unlock paid features off the plan string.
    plan = "pro" if entitled else "free"
    session_token = _make_token(user.uid)
    return {
        "uid":        user.uid,
        "display":    display,
        "username":   user.username,
        "first_name": user.first_name,
        "plan":       plan,
        "is_pro":     entitled,
        "is_admin":   is_admin,
        "session_token": session_token,
        "auth_token":    session_token,
    }


@app.post("/api/mobile/login")
async def api_mobile_login(request: Request):
    """UID-based login for the native mobile app (Expo Go).

    Body: {"uid": "TH-XXXXXXXX"}
    Returns the resolved user profile or 403 if the UID doesn't match.
    The mobile client persists the UID in SecureStore and re-validates on launch.
    """
    if not _uid_auth_is_legacy_allowed():
        raise HTTPException(
            status_code=410,
            detail="UID-only mobile login is disabled. Use email/password or Apple login to receive a signed session token.",
        )
    try:
        body = await request.json()
    except Exception:
        body = {}
    uid = (body.get("uid") or "").strip().upper()
    if not uid:
        raise HTTPException(status_code=400, detail="uid required")
    db = SessionLocal()
    try:
        user = _get_user_by_uid(uid, db)
        if not user:
            raise HTTPException(status_code=403, detail="invalid UID")
        if getattr(user, "banned", False):
            raise HTTPException(status_code=403, detail="This account has been suspended.")
        return JSONResponse(_build_mobile_login_response(user, db))
    finally:
        db.close()


@app.post("/api/mobile/login/email")
async def api_mobile_login_email(request: Request):
    """Email + password login for the native mobile app.

    Body: {"email": "user@example.com", "password": "..."}
    Returns the same payload as /api/mobile/login (with the resolved UID inside)
    so the client can persist the UID and use it for all subsequent calls.
    """
    try:
        body = await request.json()
    except Exception:
        body = {}
    email = (body.get("email") or "").strip().lower()
    password = (body.get("password") or "").strip()
    if not email or not password:
        raise HTTPException(status_code=400, detail="email and password required")

    def _check():
        db = SessionLocal()
        try:
            u = db.query(User).filter(User.email == email).first()
            if not u or not u.password_hash:
                return ("no_account", None, None)
            if not _verify_password(password, u.password_hash):
                return ("bad_password", None, None)
            if getattr(u, "banned", False):
                return ("banned", None, None)
            if not u.uid:
                return ("incomplete", None, None)
            payload = _build_mobile_login_response(u, db)
            return ("ok", payload, None)
        finally:
            db.close()

    status, payload, _ = await asyncio.to_thread(_check)
    if status == "no_account":
        raise HTTPException(status_code=403, detail="No account found for that email.")
    if status == "bad_password":
        raise HTTPException(status_code=403, detail="Incorrect password.")
    if status == "banned":
        raise HTTPException(status_code=403, detail="This account has been suspended.")
    if status == "incomplete":
        raise HTTPException(status_code=403, detail="Account is missing a UID. Finish setup at tradehub.markets.")
    return JSONResponse(payload)


@app.post("/api/mobile/push/register")
async def api_mobile_push_register(request: Request):
    """Register an Expo push token for the signed-in mobile user.

    Body: {"token": "ExponentPushToken[...]", "platform": "ios" | "android"}
    Auth: X-TradeHub-UID header (translated to ?uid= by middleware).
    Idempotent — re-registering an existing token transfers it to whoever
    is signed in on this device now.
    """
    from app.models import MobilePushToken
    uid = request.query_params.get("uid", "").strip().upper()
    if not uid:
        raise HTTPException(status_code=401, detail="auth required")
    try:
        body = await request.json()
    except Exception:
        body = {}
    token = (body.get("token") or "").strip()
    platform = (body.get("platform") or "ios").strip().lower()
    if not token:
        raise HTTPException(status_code=400, detail="token required")
    if platform not in ("ios", "android"):
        platform = "ios"

    db = SessionLocal()
    try:
        user = _get_user_by_uid(uid, db)
        if not user:
            raise HTTPException(status_code=403, detail="invalid UID")
        existing = db.query(MobilePushToken).filter(
            MobilePushToken.token == token
        ).first()
        if existing:
            existing.user_id = user.id
            existing.platform = platform
            existing.updated_at = datetime.utcnow()
        else:
            db.add(MobilePushToken(
                user_id=user.id,
                token=token,
                platform=platform,
            ))
        db.commit()
        return JSONResponse({"ok": True})
    finally:
        db.close()


@app.post("/api/mobile/push/unregister")
async def api_mobile_push_unregister(request: Request):
    """Remove a push token (called on sign-out so the device stops receiving
    pushes for the previously-signed-in user)."""
    from app.models import MobilePushToken
    try:
        body = await request.json()
    except Exception:
        body = {}
    token = (body.get("token") or "").strip()
    if not token:
        return JSONResponse({"ok": True})  # nothing to do
    db = SessionLocal()
    try:
        db.query(MobilePushToken).filter(
            MobilePushToken.token == token
        ).delete()
        db.commit()
        return JSONResponse({"ok": True})
    finally:
        db.close()


_apple_jwks_cache: dict = {"keys": [], "fetched_at": 0}

def _get_apple_public_keys():
    """Fetch and cache Apple's JWKS (public keys for verifying identity tokens).
    Cached for 1 hour to avoid hitting Apple on every login.
    """
    import time as _time
    now = _time.time()
    if _apple_jwks_cache["keys"] and (now - _apple_jwks_cache["fetched_at"]) < 3600:
        return _apple_jwks_cache["keys"]
    try:
        import urllib.request
        req = urllib.request.Request("https://appleid.apple.com/auth/keys")
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())
        _apple_jwks_cache["keys"] = data.get("keys", [])
        _apple_jwks_cache["fetched_at"] = now
        return _apple_jwks_cache["keys"]
    except Exception as e:
        logger.warning(f"[apple-auth] Failed to fetch Apple JWKS: {e}")
        return _apple_jwks_cache["keys"]


@app.post("/api/mobile/login/apple")
async def api_mobile_login_apple(request: Request):
    """Sign in with Apple for the native mobile app.

    Body: {"identity_token": "<JWT from Apple>", "full_name": "...", "email": "..."}
    The identity_token is a JWT signed by Apple. We verify its signature against
    Apple's JWKS, validate iss/exp claims, then find-or-create the user.
    """
    import jwt as _jwt
    from jwt.algorithms import RSAAlgorithm as _RSAAlgorithm

    try:
        body = await request.json()
    except Exception:
        body = {}
    id_token = (body.get("identity_token") or "").strip()
    if not id_token:
        raise HTTPException(status_code=400, detail="identity_token required")

    # Step 1: Decode the JWT header to find which Apple key was used
    try:
        unverified_header = _jwt.get_unverified_header(id_token)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid Apple identity token format")

    kid = unverified_header.get("kid")
    if not kid:
        raise HTTPException(status_code=400, detail="Apple token missing kid in header")

    # Step 2: Fetch Apple's public keys and find the matching key
    apple_keys = await asyncio.to_thread(_get_apple_public_keys)
    matching_key = None
    for k in apple_keys:
        if k.get("kid") == kid:
            matching_key = k
            break

    if not matching_key:
        raise HTTPException(status_code=400, detail="Apple token signed with unknown key")

    # Step 3: Verify the JWT signature and decode claims
    try:
        public_key = _RSAAlgorithm.from_jwk(json.dumps(matching_key))
        claims = _jwt.decode(
            id_token,
            key=public_key,
            algorithms=["RS256"],
            issuer="https://appleid.apple.com",
            options={
                "verify_aud": False,
                "verify_exp": True,
                "verify_iss": True,
            },
        )
    except _jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Apple token has expired")
    except _jwt.InvalidIssuerError:
        raise HTTPException(status_code=400, detail="Apple token has invalid issuer")
    except Exception as e:
        logger.warning(f"[apple-auth] JWT verification failed: {e}")
        raise HTTPException(status_code=400, detail="Could not verify Apple identity token")

    apple_sub = claims.get("sub", "").strip()
    if not apple_sub:
        raise HTTPException(status_code=400, detail="Apple token missing sub claim")

    # Trust the email from the verified token, NOT from the client body
    apple_email = claims.get("email", "").strip().lower()
    # Full name is only provided on first sign-in; Apple strips it after that.
    full_name = (body.get("full_name") or "").strip()

    def _find_or_create():
        db = SessionLocal()
        try:
            # First try to find by apple_sub stored in User.apple_id
            user = db.query(User).filter(User.apple_id == apple_sub).first()
            if user:
                if getattr(user, "banned", False):
                    return ("banned", None)
                return ("ok", _build_mobile_login_response(user, db))

            # Try by email match (link Apple ID to existing account)
            if apple_email:
                user = db.query(User).filter(User.email == apple_email).first()
                if user:
                    if getattr(user, "banned", False):
                        return ("banned", None)
                    user.apple_id = apple_sub
                    if not user.auth_provider or user.auth_provider == "telegram":
                        user.auth_provider = "apple"
                    db.commit()
                    return ("ok", _build_mobile_login_response(user, db))

            # Create a new account
            import secrets as _secrets
            new_uid = "TH-" + _secrets.token_hex(4).upper()
            while db.query(User).filter(User.uid == new_uid).first():
                new_uid = "TH-" + _secrets.token_hex(4).upper()
            # telegram_id is NOT NULL in the schema — use a placeholder for
            # Apple-auth users who have no Telegram account.
            tg_placeholder = f"apple_{apple_sub[:32]}"
            user = User(
                telegram_id=tg_placeholder,
                uid=new_uid,
                email=apple_email or None,
                first_name=full_name or None,
                apple_id=apple_sub,
                auth_provider="apple",
            )
            db.add(user)
            db.commit()
            db.refresh(user)
            return ("ok", _build_mobile_login_response(user, db))
        finally:
            db.close()

    status, payload = await asyncio.to_thread(_find_or_create)
    if status == "banned":
        raise HTTPException(status_code=403, detail="This account has been suspended.")
    return JSONResponse(payload)


@app.delete("/api/mobile/account")
async def api_mobile_delete_account(request: Request):
    """Delete the authenticated user's account (Apple/Google requirement).

    Soft-deletes: deactivates all strategies, removes push tokens,
    clears PII, and marks the account as banned so the UID cannot be reused
    for login. Trade history is anonymized.
    """
    from app.models import MobilePushToken
    uid = request.query_params.get("uid", "").strip().upper()
    if not uid:
        uid_header = request.headers.get("X-TradeHub-UID", "").strip().upper()
        uid = uid_header or ""
    if not uid:
        raise HTTPException(status_code=401, detail="auth required")

    def _do_delete():
        db = SessionLocal()
        try:
            user = _get_user_by_uid(uid, db)
            if not user:
                raise HTTPException(status_code=403, detail="invalid UID")

            # Deactivate all strategies
            db.query(UserStrategy).filter(
                UserStrategy.user_id == user.id
            ).update({"status": "deleted"})

            # Remove push tokens
            db.query(MobilePushToken).filter(
                MobilePushToken.user_id == user.id
            ).delete()

            # Clear PII
            user.email = None
            user.password_hash = None
            user.first_name = f"Deleted User"
            user.username = None
            user.apple_id = None
            user.banned = True

            db.commit()
            return {"ok": True, "message": "Account deleted successfully"}
        finally:
            db.close()

    result = await asyncio.to_thread(_do_delete)
    return JSONResponse(result)


@app.get("/api/me")
async def api_me(request: Request, uid: str = Query(None)):
    """Session + profile probe for the portal shell and auth-aware nav.

    Always returns 200 — anonymous visitors get {"logged_in": false}.
  When ?uid= is supplied it must match the signed session (middleware).
    """
    no_cache = {"Cache-Control": "private, no-store", "Pragma": "no-cache"}
    session_uid = _get_session_uid(request)
    if not session_uid:
        return JSONResponse({"logged_in": False}, headers=no_cache)

    _me_key = f"api_me:{session_uid}"
    _me_cached = get_cache(_me_key)
    if isinstance(_me_cached, dict):
        return JSONResponse(_me_cached, headers={**no_cache, "X-Cache": "HIT"})

    def _load_me():
        from app.database import SessionLocal
        db = SessionLocal()
        try:
            user = _get_user_by_uid(session_uid, db)
            if not user:
                return None
            display = (user.username or user.first_name or user.uid or "").strip() or user.uid
            avatar = (user.first_name or user.username or "T")[0].upper()
            sub = _get_portal_sub(user.id, db)
            is_pro = _is_portal_pro(sub) or bool(getattr(user, "is_admin", False))
            is_web = str(getattr(user, "telegram_id", "") or "").startswith("WEB-")
            return {
                "logged_in": True,
                "uid": user.uid or session_uid,
                "username": user.username or "",
                "first_name": user.first_name,
                "display": display,
                "name": user.first_name or user.username or "Trader",
                "avatar": avatar,
                "is_pro": is_pro,
                "is_web_user": is_web,
            }
        finally:
            db.close()

    try:
        payload = await asyncio.wait_for(asyncio.to_thread(_load_me), timeout=8.0)
    except asyncio.TimeoutError:
        raise HTTPException(status_code=503, detail="Database busy")
    if not payload:
        return JSONResponse({"logged_in": False}, headers=no_cache)
    set_cache(_me_key, payload, ttl_seconds=15)
    return JSONResponse(payload, headers=no_cache)


@app.get("/api/walls/{symbol}")
@app.get("/api/trade/walls/{symbol}")
async def trade_walls(symbol: str, ai: int = 0):
    """Public wall report for the trade page. Cached ~25s."""
    sym = (symbol or "").upper().strip()
    if sym not in TRADE_SYMBOL_WHITELIST:
        return JSONResponse({"error": "symbol not supported"}, status_code=404)

    pair = TRADE_SYMBOL_WHITELIST[sym]
    cache_key = f"trade_walls_{pair}_{1 if ai else 0}"
    cached = _CACHE.get(cache_key)
    if cached and time.time() < cached[1]:
        return cached[0]

    try:
        from dataclasses import asdict
        from app.services.liquidity_walls import scan_walls
        report = await scan_walls(pair, use_ai=bool(ai))
        if not report:
            return JSONResponse({"error": "no order book data"}, status_code=503)
        payload = asdict(report)
        # behavior dict has float keys — JSON needs strings
        wb = payload.get("wall_behavior") or {}
        payload["wall_behavior"] = {str(k): v for k, v in wb.items()}
        _CACHE[cache_key] = (payload, time.time() + 10)  # was 25 — match frontend 12s poll
        return payload
    except Exception as e:
        logger.warning(f"trade_walls({sym}) failed: {e}")
        return JSONResponse({"error": "scan failed"}, status_code=500)


@app.get("/api/candles/{symbol}")
@app.get("/api/trade/candles/{symbol}")
async def trade_candles(symbol: str, tf: str = "5m", limit: int = 300):
    """Public candle feed for the trade chart. MEXC source.
    Cache TTL scales with limit — small (≤5) requests cache ~1.5s for live ticks,
    big (≥50) requests cache ~10s since they're only used for initial chart load."""
    sym = (symbol or "").upper().strip()
    if sym not in TRADE_SYMBOL_WHITELIST:
        return JSONResponse({"error": "symbol not supported"}, status_code=404)
    interval = _TRADE_TF_MAP.get(tf)
    if not interval:
        return JSONResponse({"error": "bad timeframe"}, status_code=400)
    limit = max(2, min(int(limit or 300), 500))

    pair = TRADE_SYMBOL_WHITELIST[sym]
    cache_key = f"trade_candles_{pair}_{interval}_{limit}"
    cached = _CACHE.get(cache_key)
    if cached and time.time() < cached[1]:
        return cached[0]

    try:
        import httpx
        async with httpx.AsyncClient(timeout=4.0) as client:
            r = await client.get(
                "https://api.mexc.com/api/v3/klines",
                params={"symbol": pair, "interval": interval, "limit": limit},
            )
            r.raise_for_status()
            rows = r.json() or []
        candles = []
        for k in rows:
            try:
                # Lightweight Charts wants UTC seconds for time.
                # MEXC kline shape: [openTime, o, h, l, c, baseVol, closeTime, quoteVol(USD), ...]
                candles.append({
                    "time":   int(k[0]) // 1000,
                    "open":   float(k[1]),
                    "high":   float(k[2]),
                    "low":    float(k[3]),
                    "close":  float(k[4]),
                    "volume": float(k[5]) if len(k) > 5 else 0.0,
                    "quote_volume": float(k[7]) if len(k) > 7 else 0.0,
                })
            except (TypeError, ValueError, IndexError):
                continue
        payload = {"symbol": pair, "tf": interval, "candles": candles}
        # Frontend polls candles every 3s — keep cache short so each poll
        # mostly returns fresh data. 1.5s for tick (limit≤5), 4s for full
        # candle history (was 10s).
        ttl = 1.5 if limit <= 5 else 4.0
        _CACHE[cache_key] = (payload, time.time() + ttl)
        return payload
    except Exception as e:
        logger.warning(f"trade_candles({sym},{tf}) failed: {e}")
        return JSONResponse({"error": "candle fetch failed"}, status_code=502)


@app.get("/api/ticker/{symbol}")
@app.get("/api/trade/ticker/{symbol}")
async def trade_ticker(symbol: str):
    """Live price + 24h stats for the trade page header. MEXC 24hr endpoint.
    Cached ~2s — keeps header live without hammering MEXC."""
    sym = (symbol or "").upper().strip()
    if sym not in TRADE_SYMBOL_WHITELIST:
        return JSONResponse({"error": "symbol not supported"}, status_code=404)
    pair = TRADE_SYMBOL_WHITELIST[sym]
    cache_key = f"trade_ticker24_{pair}"
    cached = _CACHE.get(cache_key)
    if cached and time.time() < cached[1]:
        return cached[0]
    try:
        import httpx
        async with httpx.AsyncClient(timeout=3.0) as client:
            r = await client.get(
                "https://api.mexc.com/api/v3/ticker/24hr",
                params={"symbol": pair},
            )
            r.raise_for_status()
            d = r.json() or {}

        def _f(k):
            try: return float(d.get(k) or 0)
            except (TypeError, ValueError): return 0.0

        last = _f("lastPrice")
        bid  = _f("bidPrice")
        ask  = _f("askPrice")
        # Prefer last trade price; fall back to mid if MEXC ever returns 0
        price = last or ((bid + ask) / 2 if (bid > 0 and ask > 0) else (bid or ask))
        payload = {
            "symbol": pair,
            "price": price,
            "bid": bid, "ask": ask,
            "change_abs": _f("priceChange"),
            "change_pct": _f("priceChangePercent"),
            "open_24h":  _f("openPrice"),
            "high_24h":  _f("highPrice"),
            "low_24h":   _f("lowPrice"),
            "vol_24h_base":  _f("volume"),
            "vol_24h_quote": _f("quoteVolume"),
            "ts": int(time.time() * 1000),
        }
        _CACHE[cache_key] = (payload, time.time() + 2.0)
        return payload
    except Exception as e:
        logger.debug(f"trade_ticker({sym}) failed: {e}")
        return JSONResponse({"error": "ticker fetch failed"}, status_code=502)


# ─────────────────────────────────────────────────────────────────────────────
# Live trade tape — recent big prints from MEXC public trades feed.
# Powers the scrolling tape panel + chart markers on /trade.
# ─────────────────────────────────────────────────────────────────────────────

# Per-symbol rolling buffer of recent big trades. MEXC's /api/v3/trades returns
# id=null, so we dedup by (time_ms, price, qty) and use time as the cursor.
_TAPE_BUF: dict[str, list[dict]] = {}        # values are dicts with synthetic "id" = sequence number
_TAPE_KEYS: dict[str, set] = {}              # set of (ts, price, qty) for dedup
_TAPE_SEQ: dict[str, int] = {}               # monotonic sequence so client can use "since"

@app.get("/api/trade/tape/{symbol}")
async def trade_tape(symbol: str, since: int = 0, min_usd: float = 25_000.0, limit: int = 50):
    """Return recent BIG trades (>= min_usd) for the live tape on /trade.
    Pulls MEXC `/api/v3/trades` (id is null on this endpoint, so we dedup by
    composite key and assign our own sequence id for client-side cursoring).
    """
    sym = (symbol or "").upper().strip()
    if sym not in TRADE_SYMBOL_WHITELIST:
        return JSONResponse({"error": "symbol not supported"}, status_code=404)
    pair = TRADE_SYMBOL_WHITELIST[sym]
    min_usd = max(1_000.0, float(min_usd or 25_000.0))
    limit = max(1, min(int(limit or 50), 200))

    # 1.5s cached fetch from MEXC so 5-tab spam doesn't hammer them
    cache_key = f"tape_raw_{pair}"
    cached = _CACHE.get(cache_key)
    rows: list = []
    if cached and time.time() < cached[1]:
        rows = cached[0]
    else:
        try:
            import httpx
            async with httpx.AsyncClient(timeout=3.0) as client:
                r = await client.get(
                    "https://api.mexc.com/api/v3/trades",
                    params={"symbol": pair, "limit": 500},
                )
                r.raise_for_status()
                rows = r.json() or []
            _CACHE[cache_key] = (rows, time.time() + 1.5)
        except Exception as e:
            logger.debug(f"trade_tape({sym}) fetch failed: {e}")
            rows = cached[0] if cached else []

    buf  = _TAPE_BUF.setdefault(pair, [])
    keys = _TAPE_KEYS.setdefault(pair, set())
    seq  = _TAPE_SEQ.get(pair, 0)

    # MEXC returns newest first — process in chronological order so seq numbers ascend with time
    for t in reversed(rows):
        try:
            ts    = int(t.get("time") or 0)
            price = float(t.get("price") or 0)
            qty   = float(t.get("qty") or 0)
            if ts <= 0 or price <= 0 or qty <= 0:
                continue
            usd = float(t.get("quoteQty") or (price * qty))
            if usd < 1_000:
                continue
            key = (ts, price, qty)
            if key in keys:
                continue
            keys.add(key)
            seq += 1
            # MEXC: isBuyerMaker=True → an aggressive SELL (taker hit the bid).
            side = "sell" if bool(t.get("isBuyerMaker")) else "buy"
            buf.append({
                "id": seq,
                "ts": ts,
                "price": price,
                "qty": qty,
                "usd": usd,
                "side": side,
            })
        except (TypeError, ValueError):
            continue
    _TAPE_SEQ[pair] = seq

    # Trim buffer + key set in lockstep
    if len(buf) > 400:
        for old in buf[:-400]:
            keys.discard((old["ts"], old["price"], old["qty"]))
        del buf[:-400]

    out = [t for t in buf if t["usd"] >= min_usd and t["id"] > int(since or 0)]
    out.sort(key=lambda x: x["id"], reverse=True)
    out = out[:limit]
    return {
        "symbol": pair,
        "trades": out,
        "last_id": seq,
        "min_usd": min_usd,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Chart-aware AI trade read — POST endpoint that consumes the user's full
# chart context (visible indicators, overlays, recent flow) and returns a
# structured trade plan from Claude.
# ─────────────────────────────────────────────────────────────────────────────

def _hash_ai_read_request(payload: dict) -> str:
    """Stable cache key for identical chart-context requests."""
    try:
        canonical = json.dumps(payload, sort_keys=True, default=str)
    except Exception:
        canonical = str(payload)
    return hashlib.md5(canonical.encode("utf-8")).hexdigest()[:12]


@app.post("/api/trade/ai_read/{symbol}")
async def trade_ai_read(symbol: str, request: Request):
    """Generate a chart-aware trade plan. Body shape:
        {
          "tf": "5m",
          "indicators": [{"type":"ema","src":"close","params":{"period":20}}, ...],
          "toggles": {"order_blocks": true, "liq_heatmap": true, "big_prints": true},
          "tape": {"buy_count": 12, "sell_count": 8, "buy_usd": 1.2e6, "sell_usd": 0.8e6}
        }
    Cached 25 s per (symbol, tf, full-payload-hash) to throttle Claude calls.
    """
    sym = (symbol or "").upper().strip()
    if sym not in TRADE_SYMBOL_WHITELIST:
        return JSONResponse({"error": "symbol not supported"}, status_code=404)
    pair = TRADE_SYMBOL_WHITELIST[sym]

    try:
        body = await request.json()
    except Exception:
        body = {}
    if not isinstance(body, dict):
        body = {}

    tf = str(body.get("tf") or "5m").lower()
    if tf not in ("1m", "5m", "15m", "1h"):
        tf = "5m"
    raw_indicators = body.get("indicators") or []
    if not isinstance(raw_indicators, list):
        raw_indicators = []
    # Strict whitelist — type/src must match known values; params is a flat dict
    # of numeric values. Prevents prompt injection via free-form indicator names.
    _ALLOWED_TYPES = {"ema", "sma", "rsi", "macd", "bb", "vwap", "atr", "supertrend", "stochrsi"}
    _ALLOWED_SRCS  = {"close", "open", "high", "low", "hl2", "hlc3", "ohlc4"}
    _ALLOWED_PARAM_KEYS = {"period", "fast", "slow", "signal", "mult", "stoch"}
    indicators: list = []
    for raw in raw_indicators[:12]:
        if not isinstance(raw, dict):
            continue
        t = str(raw.get("type", "")).lower().strip()
        if t not in _ALLOWED_TYPES:
            continue
        s = str(raw.get("src", "close")).lower().strip()
        if s not in _ALLOWED_SRCS:
            s = "close"
        rp = raw.get("params") or {}
        if not isinstance(rp, dict):
            rp = {}
        clean_params: dict = {}
        for k, v in rp.items():
            if k not in _ALLOWED_PARAM_KEYS:
                continue
            try:
                fv = float(v)
            except (TypeError, ValueError):
                continue
            # Cap to sane TA bounds — protects against absurd compute
            if fv < 0 or fv > 500:
                continue
            clean_params[k] = fv
        indicators.append({"type": t, "src": s, "params": clean_params})
    toggles = body.get("toggles") or {}
    if not isinstance(toggles, dict):
        toggles = {}
    tape = body.get("tape") or {}
    if not isinstance(tape, dict):
        tape = {}

    # Cache identical chart contexts to throttle Claude usage.
    cache_payload = {"sym": sym, "tf": tf, "ind": indicators, "tg": toggles, "tp": tape}
    cache_key = f"trade_ai_read_{_hash_ai_read_request(cache_payload)}"
    cached = _CACHE.get(cache_key)
    if cached and time.time() < cached[1]:
        c = dict(cached[0])
        c["age_secs"] = int(time.time() - c.get("generated_at_ts", time.time()))
        c["cached"] = True
        return c

    # Pull candles (server-side fetch; same cached source used by /candles).
    interval = _TRADE_TF_MAP.get(tf)
    candles: list = []
    try:
        ck_key = f"trade_candles_{pair}_{interval}_300"
        ck = _CACHE.get(ck_key)
        if ck and time.time() < ck[1]:
            candles = (ck[0].get("candles") or [])[:]
        else:
            async with httpx.AsyncClient(timeout=4.0) as client:
                r = await client.get(
                    "https://api.mexc.com/api/v3/klines",
                    params={"symbol": pair, "interval": interval, "limit": 300},
                )
                r.raise_for_status()
                rows = r.json() or []
            for k in rows:
                try:
                    candles.append({
                        "time":   int(k[0]) // 1000,
                        "open":   float(k[1]),
                        "high":   float(k[2]),
                        "low":    float(k[3]),
                        "close":  float(k[4]),
                        "volume": float(k[5]) if len(k) > 5 else 0.0,
                    })
                except (TypeError, ValueError, IndexError):
                    continue
    except Exception as e:
        logger.warning(f"ai_read candle fetch failed for {sym} {tf}: {e}")

    # Pull cached wall report (no AI — we make our own). Tolerate failure.
    wall_report: Optional[dict] = None
    try:
        from dataclasses import asdict
        from app.services.liquidity_walls import scan_walls
        wkey = f"trade_walls_{pair}_0"
        wcache = _CACHE.get(wkey)
        if wcache and time.time() < wcache[1]:
            wall_report = wcache[0]
        else:
            rep = await scan_walls(pair, use_ai=False)
            if rep:
                wall_report = asdict(rep)
                wb = wall_report.get("wall_behavior") or {}
                wall_report["wall_behavior"] = {str(k): v for k, v in wb.items()}
                _CACHE[wkey] = (wall_report, time.time() + 25)
    except Exception as e:
        logger.debug(f"ai_read wall fetch failed for {sym}: {e}")

    # If client didn't send tape stats, derive from server buffer (last ~10 min of big prints).
    if not tape and pair in _TAPE_BUF:
        cutoff_ms = int(time.time() * 1000) - 10 * 60 * 1000
        recent = [t for t in _TAPE_BUF[pair] if t["ts"] >= cutoff_ms and t["usd"] >= 25_000]
        if recent:
            buys  = [t for t in recent if t["side"] == "buy"]
            sells = [t for t in recent if t["side"] == "sell"]
            tape = {
                "buy_count":  len(buys),
                "sell_count": len(sells),
                "buy_usd":    sum(t["usd"] for t in buys),
                "sell_usd":   sum(t["usd"] for t in sells),
            }

    # Fetch HTF candles (1H + 4H) so the AI can weigh higher-timeframe trend.
    # Uses the alerts_engine cache so concurrent reads share one HTTP call.
    htf_1h: list = []
    htf_4h: list = []
    try:
        from app.services.alerts_engine import _fetch_candles as _fc_htf
        htf_1h = await _fc_htf(sym, "1h", limit=120) or []
        htf_4h = await _fc_htf(sym, "4h", limit=120) or []
    except Exception as e:
        logger.debug(f"ai_read HTF fetch failed for {sym}: {e}")

    # Funding/OI snapshot via the shared auto_trader cache (60s TTL) so the
    # /trade button and the AI auto-trader fleet share one Coinglass call.
    funding_data: Optional[dict] = None
    try:
        from app.services.auto_trader import _fetch_funding_oi
        funding_data = await _fetch_funding_oi(sym)
    except Exception as e:
        logger.debug(f"ai_read funding fetch failed for {sym}: {e}")

    try:
        from app.services.ai_trade_read import generate_ai_trade_read
        result = await generate_ai_trade_read(
            symbol=sym, tf=tf, candles=candles,
            indicators=indicators, toggles=toggles, tape=tape,
            wall_report=wall_report,
            htf_1h_candles=htf_1h, htf_4h_candles=htf_4h,
            funding_data=funding_data,
        )
    except Exception as e:
        logger.warning(f"ai_read generation failed for {sym}: {e}")
        return JSONResponse({"error": "ai read failed"}, status_code=502)

    payload = {
        "symbol": sym,
        "tf": tf,
        "summary": result.get("summary", ""),
        "fallback": bool(result.get("fallback", False)),
        "sources_used": result.get("sources_used", []),
        "generated_at_ts": time.time(),
        "generated_at": int(time.time()),
        "age_secs": 0,
        "cached": False,
    }
    _CACHE[cache_key] = (payload, time.time() + 25)
    return payload


# ─────────────────────────────────────────────────────────────────────────────
# Indicator alerts API — fires Telegram DM when condition hits
# ─────────────────────────────────────────────────────────────────────────────

# Limits + validation
_MAX_ALERTS_PER_USER = 25
_ALERT_KINDS = {"price", "rsi", "ema_cross", "macd_cross_zero",
                "supertrend_flip", "fvg_retest"}
# "bull" / "bear" are the two sides of an FVG retest — they don't fit the
# generic above/below verbs since they encode an entire pattern, not a
# single threshold cross.
_ALERT_CONDITIONS = {"above", "below", "crossover", "crossunder", "flip",
                     "bull", "bear"}
_ALERT_TFS = {"1m", "5m", "15m", "1h"}
_ALERT_FIRE_MODES = {"once", "every_cross", "every_cross_with_cooldown"}
_MAX_ALERT_COOLDOWN_MIN = 24 * 60       # 1 day
_MAX_ALERT_DAILY_CAP    = 500           # absolute spam ceiling


def _alert_to_dict(a) -> dict:
    return {
        "id": a.id,
        "symbol": a.symbol,
        "timeframe": a.timeframe,
        "kind": a.kind,
        "condition": a.condition,
        "target": a.target,
        "params": json.loads(a.params or "{}"),
        "label": a.label,
        "status": a.status,
        "created_at": a.created_at.isoformat() if a.created_at else None,
        "triggered_at": a.triggered_at.isoformat() if a.triggered_at else None,
        "triggered_message": a.triggered_message,
        "triggered_price": a.triggered_price,
        # Repeating-alert fields (Task #10)
        "fire_mode": getattr(a, "fire_mode", None) or "once",
        "cooldown_minutes": int(getattr(a, "cooldown_minutes", 0) or 0),
        "daily_cap": getattr(a, "daily_cap", None),
        "fire_count": int(getattr(a, "fire_count", 0) or 0),
        "last_fired_at": a.last_fired_at.isoformat() if getattr(a, "last_fired_at", None) else None,
        "fired_today_count": int(getattr(a, "fired_today_count", 0) or 0),
        "fired_today_date": getattr(a, "fired_today_date", None),
    }


def _build_alert_label(kind: str, params: dict, condition: str, target, symbol: str) -> str:
    p = params or {}
    side = {"above": "above", "below": "below", "crossover": "crosses up",
            "crossunder": "crosses down", "flip": "flips"}.get(condition, condition)
    if kind == "price":
        return f"Price {side} ${float(target):,.2f} · {symbol}"
    if kind == "rsi":
        return f"RSI({int(p.get('period', 14))}) {side} {float(target):.1f} · {symbol}"
    if kind == "ema_cross":
        return f"Price {side} EMA({int(p.get('period', 50))}) · {symbol}"
    if kind == "macd_cross_zero":
        return f"MACD({int(p.get('fast', 12))},{int(p.get('slow', 26))}) {side} zero · {symbol}"
    if kind == "supertrend_flip":
        return f"SuperTrend({int(p.get('period', 10))},{p.get('mult', 3)}) flips · {symbol}"
    if kind == "fvg_retest":
        word = "bull" if condition == "bull" else "bear"
        return (f"Price retests unfilled {word.upper()} FVG "
                f"(≥{float(p.get('min_gap_pct', 0.05)):g}%, "
                f"≤{int(p.get('max_age_bars', 100))} bars) · {symbol}")
    return f"{kind} · {symbol}"


@app.get("/api/trade/alerts/me")
async def trade_alerts_me(request: Request, db: Session = Depends(get_db)):
    """Returns logged-in state + active count. Used by /trade UI to decide
    whether to show 'Log in to create alerts' or the manage panel."""
    from app.models import IndicatorAlert
    uid = _get_session_uid(request)
    if not uid:
        return {"logged_in": False, "telegram_linked": False, "active_count": 0, "max": _MAX_ALERTS_PER_USER}
    user = _get_user_by_uid(uid, db)
    if not user:
        return {"logged_in": False, "telegram_linked": False, "active_count": 0, "max": _MAX_ALERTS_PER_USER}
    active_count = (db.query(IndicatorAlert)
                      .filter(IndicatorAlert.user_id == user.id,
                              IndicatorAlert.status == "active")
                      .count())
    tg_linked = bool(user.telegram_id) and not str(user.telegram_id).startswith("WEB-")
    return {
        "logged_in": True,
        "telegram_linked": tg_linked,
        "telegram_id": user.telegram_id if tg_linked else None,
        "active_count": active_count,
        "max": _MAX_ALERTS_PER_USER,
    }


@app.get("/api/trade/alerts/list")
async def trade_alerts_list(request: Request, db: Session = Depends(get_db)):
    """List the logged-in user's alerts (active + recently triggered)."""
    from app.models import IndicatorAlert
    uid = _get_session_uid(request)
    if not uid:
        raise HTTPException(status_code=401, detail="login required")
    user = _get_user_by_uid(uid, db)
    if not user:
        raise HTTPException(status_code=401, detail="login required")
    rows = (db.query(IndicatorAlert)
              .filter(IndicatorAlert.user_id == user.id,
                      IndicatorAlert.status.in_(["active", "triggered"]))
              .order_by(IndicatorAlert.id.desc())
              .limit(60).all())
    return {"alerts": [_alert_to_dict(a) for a in rows]}


@app.post("/api/trade/alerts/create")
async def trade_alerts_create(request: Request, db: Session = Depends(get_db)):
    """Create a new alert. Body: {kind, condition, symbol, timeframe, target, params, label?}"""
    from app.models import IndicatorAlert
    uid = _get_session_uid(request)
    if not uid:
        raise HTTPException(status_code=401, detail="login required")
    user = _get_user_by_uid(uid, db)
    if not user:
        raise HTTPException(status_code=401, detail="login required")

    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="invalid JSON body")

    kind = (body.get("kind") or "").lower().strip()
    condition = (body.get("condition") or "").lower().strip()
    symbol = (body.get("symbol") or "BTC").upper().strip()
    timeframe = (body.get("timeframe") or "5m").lower().strip()
    target = body.get("target")
    params = body.get("params") or {}
    if not isinstance(params, dict):
        raise HTTPException(status_code=400, detail="params must be object")

    if kind not in _ALERT_KINDS:
        raise HTTPException(status_code=400, detail=f"unknown kind '{kind}'")
    if condition not in _ALERT_CONDITIONS:
        raise HTTPException(status_code=400, detail=f"unknown condition '{condition}'")
    if symbol not in TRADE_SYMBOL_WHITELIST:
        raise HTTPException(status_code=400, detail=f"unsupported symbol '{symbol}'")
    if timeframe not in _ALERT_TFS:
        raise HTTPException(status_code=400, detail=f"unsupported timeframe '{timeframe}'")

    # Per-kind validation
    if kind in ("price", "rsi"):
        if target is None:
            raise HTTPException(status_code=400, detail=f"{kind} alert requires target value")
        try:
            target = float(target)
        except (TypeError, ValueError):
            raise HTTPException(status_code=400, detail="target must be a number")
        if kind == "rsi" and not (0 <= target <= 100):
            raise HTTPException(status_code=400, detail="RSI target must be 0–100")
    else:
        target = None

    # Sane numeric bounds on params
    def _clamp_int(v, lo, hi, default):
        try:
            n = int(v)
        except (TypeError, ValueError):
            n = default
        return max(lo, min(hi, n))

    if kind == "rsi":
        params = {"period": _clamp_int(params.get("period", 14), 2, 200, 14)}
    elif kind == "ema_cross":
        params = {"period": _clamp_int(params.get("period", 50), 2, 500, 50)}
    elif kind == "macd_cross_zero":
        params = {
            "fast":   _clamp_int(params.get("fast", 12),   2, 100, 12),
            "slow":   _clamp_int(params.get("slow", 26),   3, 200, 26),
            "signal": _clamp_int(params.get("signal", 9),  2, 100, 9),
        }
        if params["fast"] >= params["slow"]:
            raise HTTPException(status_code=400, detail="MACD fast must be < slow")
    elif kind == "supertrend_flip":
        try:
            mult = float(params.get("mult", 3))
        except (TypeError, ValueError):
            mult = 3.0
        params = {
            "period": _clamp_int(params.get("period", 10), 2, 100, 10),
            "mult":   max(0.5, min(20.0, mult)),
        }
    elif kind == "fvg_retest":
        try:
            mgp = float(params.get("min_gap_pct", 0.05))
        except (TypeError, ValueError):
            mgp = 0.05
        params = {
            "min_gap_pct":  max(0.01, min(5.0, mgp)),
            "max_age_bars": _clamp_int(params.get("max_age_bars", 100), 5, 500, 100),
        }
        if condition not in ("bull", "bear"):
            raise HTTPException(status_code=400,
                                detail="FVG retest condition must be 'bull' or 'bear'")

    # Fire mode (Task #10) — backwards compatible: missing/blank → 'once'
    fire_mode = (body.get("fire_mode") or "once").lower().strip()
    if fire_mode not in _ALERT_FIRE_MODES:
        raise HTTPException(status_code=400, detail=f"unknown fire_mode '{fire_mode}'")
    cooldown_minutes = 0
    daily_cap = None
    if fire_mode == "every_cross_with_cooldown":
        try:
            cooldown_minutes = int(body.get("cooldown_minutes") or 0)
        except (TypeError, ValueError):
            raise HTTPException(status_code=400, detail="cooldown_minutes must be an integer")
        if cooldown_minutes < 0:
            cooldown_minutes = 0
        if cooldown_minutes > _MAX_ALERT_COOLDOWN_MIN:
            cooldown_minutes = _MAX_ALERT_COOLDOWN_MIN
        raw_cap = body.get("daily_cap")
        if raw_cap not in (None, "", 0, "0"):
            try:
                daily_cap = int(raw_cap)
            except (TypeError, ValueError):
                raise HTTPException(status_code=400, detail="daily_cap must be an integer")
            if daily_cap < 1:
                daily_cap = None
            elif daily_cap > _MAX_ALERT_DAILY_CAP:
                daily_cap = _MAX_ALERT_DAILY_CAP
        # Require at least one rate-limit knob so users can't pick the rate-limited
        # mode then leave both fields blank — that's identical to 'every_cross'.
        if cooldown_minutes <= 0 and daily_cap is None:
            raise HTTPException(
                status_code=400,
                detail="Set a cooldown (minutes) or a daily cap for rate-limited alerts.",
            )

    # Quota
    active_count = (db.query(IndicatorAlert)
                      .filter(IndicatorAlert.user_id == user.id,
                              IndicatorAlert.status == "active")
                      .count())
    if active_count >= _MAX_ALERTS_PER_USER:
        raise HTTPException(status_code=400,
                            detail=f"Max {_MAX_ALERTS_PER_USER} active alerts. Cancel one first.")

    label = (body.get("label") or "").strip()[:200] or _build_alert_label(kind, params, condition, target, symbol)

    alert = IndicatorAlert(
        user_id=user.id,
        symbol=symbol,
        timeframe=timeframe,
        kind=kind,
        params=json.dumps(params),
        condition=condition,
        target=target,
        label=label,
        status="active",
        fire_mode=fire_mode,
        cooldown_minutes=cooldown_minutes,
        daily_cap=daily_cap,
    )
    db.add(alert)
    db.commit()
    db.refresh(alert)
    return {"ok": True, "alert": _alert_to_dict(alert)}


@app.post("/api/trade/alerts/{alert_id}/cancel")
async def trade_alerts_cancel(alert_id: int, request: Request, db: Session = Depends(get_db)):
    """Cancel (kill) an alert."""
    from app.models import IndicatorAlert
    uid = _get_session_uid(request)
    if not uid:
        raise HTTPException(status_code=401, detail="login required")
    user = _get_user_by_uid(uid, db)
    if not user:
        raise HTTPException(status_code=401, detail="login required")
    alert = db.query(IndicatorAlert).filter(IndicatorAlert.id == alert_id).first()
    if not alert or alert.user_id != user.id:
        raise HTTPException(status_code=404, detail="alert not found")
    alert.status = "cancelled"
    db.commit()
    return {"ok": True}


# ─────────────────────────────────────────────────────────────────────────────
# Auto-trader — turn the user's chart setup into a paper-trading strategy
# ─────────────────────────────────────────────────────────────────────────────
_AUTO_MAX_PER_USER = 30
_AUTO_VALID_TFS    = {"1m", "5m", "15m", "1h"}


def _auto_strategy_to_dict(s, last_trade=None):
    win_rate = None
    closed = (s.wins or 0) + (s.losses or 0)
    if closed > 0:
        win_rate = round((s.wins or 0) * 100.0 / closed, 1)
    return {
        "id": s.id,
        "name": s.name,
        "symbol": s.symbol,
        "tf": s.timeframe,
        "mode": s.mode,
        "rules_summary": s.rules_summary,
        "cadence_min": s.cadence_min,
        "min_odds": s.min_odds,
        "notional_usd": s.notional_usd,
        "leverage": s.leverage,
        "tp_sl_source": s.tp_sl_source,
        "status": s.status,
        "notify_telegram": bool(s.notify_telegram),
        "created_at": s.created_at.isoformat() if s.created_at else None,
        "last_evaluated_at": s.last_evaluated_at.isoformat() if s.last_evaluated_at else None,
        "last_signal_at":    s.last_signal_at.isoformat()    if s.last_signal_at    else None,
        "stats": {
            "total_signals": s.total_signals or 0,
            "wins":   s.wins or 0,
            "losses": s.losses or 0,
            "win_rate": win_rate,
            "pnl_usd_total": round(s.pnl_usd_total or 0.0, 2),
        },
        "last_trade": last_trade,
        # Risk caps + sizing + partials + session + cooldown (new)
        "risk": {
            "max_concurrent_trades":       int(getattr(s, "max_concurrent_trades", 1) or 1),
            "max_daily_loss_usd":          float(s.max_daily_loss_usd) if s.max_daily_loss_usd else None,
            "max_consecutive_losses":      int(getattr(s, "max_consecutive_losses", 0) or 0),
            "position_sizing_mode":        getattr(s, "position_sizing_mode", "fixed") or "fixed",
            "risk_pct":                    float(s.risk_pct) if s.risk_pct else None,
            "account_size_usd":            float(getattr(s, "account_size_usd", 10000) or 10000),
            "enable_partial_tp1":          bool(getattr(s, "enable_partial_tp1", False)),
            "partial_tp1_pct":             float(getattr(s, "partial_tp1_pct", 50) or 50),
            "move_stop_to_be_after_tp1":   bool(getattr(s, "move_stop_to_be_after_tp1", False)),
            "session_start_utc":           getattr(s, "session_start_utc", None),
            "session_end_utc":             getattr(s, "session_end_utc", None),
            "cooldown_minutes_after_loss": int(getattr(s, "cooldown_minutes_after_loss", 0) or 0),
            # N — risk profile + between-trade cooldown
            "risk_profile":                (getattr(s, "risk_profile", None) or "medium"),
            "min_minutes_between_trades":  int(getattr(s, "min_minutes_between_trades", 0) or 0),
        },
        "runtime": (lambda _cap=float(getattr(s, "max_daily_loss_usd", 0) or 0),
                           _today=float(getattr(s, "daily_loss_today_usd", 0) or 0): {
            "consecutive_losses":   int(getattr(s, "consecutive_losses", 0) or 0),
            # Frontend-friendly aliases — keep both names so callers don't need
            # to know about the "paused_until" / "daily_loss_today" internals.
            "paused_until":         s.paused_until.isoformat() if getattr(s, "paused_until", None) else None,
            "cooldown_until":       s.paused_until.isoformat() if getattr(s, "paused_until", None) else None,
            "daily_loss_today_usd": round(_today, 2),
            "daily_loss_date":      getattr(s, "daily_loss_date", None),
            "daily_loss_hit":       bool(_cap > 0 and _today <= -abs(_cap)),
        })(),
    }


def _read_risk_fields(body: dict) -> dict:
    """Pluck and clamp the new risk/sizing/partial/session/cooldown fields out
    of a create-or-update request body. Centralised so create + update agree."""
    risk = (body.get("risk") if isinstance(body.get("risk"), dict) else body) or {}

    def _opt_float(v, lo=None, hi=None):
        try:
            f = float(v)
        except (TypeError, ValueError):
            return None
        if lo is not None: f = max(lo, f)
        if hi is not None: f = min(hi, f)
        return f

    def _opt_int(v, lo=0, hi=10_000):
        try:
            return max(lo, min(hi, int(float(v))))
        except (TypeError, ValueError):
            return None

    def _hhmm(v):
        if v is None: return None
        s = str(v).strip()
        if not s: return None
        try:
            h, m = s.split(":", 1)
            h = int(h); m = int(m)
            if 0 <= h < 24 and 0 <= m < 60:
                return f"{h:02d}:{m:02d}"
        except Exception:
            pass
        return None

    out: dict = {}
    if "max_concurrent_trades" in risk:
        v = _opt_int(risk.get("max_concurrent_trades"), 1, 5)
        if v is not None: out["max_concurrent_trades"] = v
    if "max_daily_loss_usd" in risk:
        v = risk.get("max_daily_loss_usd")
        out["max_daily_loss_usd"] = (None if v in (None, "", 0)
                                     else _opt_float(v, 1, 1_000_000))
    if "max_consecutive_losses" in risk:
        v = _opt_int(risk.get("max_consecutive_losses"), 0, 50)
        if v is not None: out["max_consecutive_losses"] = v
    if "position_sizing_mode" in risk:
        m = str(risk.get("position_sizing_mode") or "fixed").lower()
        out["position_sizing_mode"] = "risk_pct" if m == "risk_pct" else "fixed"
    if "risk_pct" in risk:
        v = risk.get("risk_pct")
        out["risk_pct"] = (None if v in (None, "", 0) else _opt_float(v, 0.1, 25))
    if "account_size_usd" in risk:
        v = _opt_float(risk.get("account_size_usd"), 100, 10_000_000)
        if v is not None: out["account_size_usd"] = v
    if "enable_partial_tp1" in risk:
        out["enable_partial_tp1"] = bool(risk.get("enable_partial_tp1"))
    if "partial_tp1_pct" in risk:
        v = _opt_float(risk.get("partial_tp1_pct"), 1, 99)
        if v is not None: out["partial_tp1_pct"] = v
    if "move_stop_to_be_after_tp1" in risk:
        out["move_stop_to_be_after_tp1"] = bool(risk.get("move_stop_to_be_after_tp1"))
    # Session window — strict: if the caller sent a non-empty value but it
    # didn't parse as HH:MM (UTC), reject the whole save instead of silently
    # storing None. Silent-None is what made users think their session
    # filter was on when it actually wasn't, so the bot kept trading 24/7.
    if "session_start_utc" in risk:
        raw = risk.get("session_start_utc")
        if raw in (None, ""):
            out["session_start_utc"] = None
        else:
            parsed = _hhmm(raw)
            if parsed is None:
                raise HTTPException(
                    status_code=400,
                    detail=f"session_start_utc must be HH:MM in 24h UTC (got '{raw}')",
                )
            out["session_start_utc"] = parsed
    if "session_end_utc" in risk:
        raw = risk.get("session_end_utc")
        if raw in (None, ""):
            out["session_end_utc"] = None
        else:
            parsed = _hhmm(raw)
            if parsed is None:
                raise HTTPException(
                    status_code=400,
                    detail=f"session_end_utc must be HH:MM in 24h UTC (got '{raw}')",
                )
            out["session_end_utc"] = parsed
    if "cooldown_minutes_after_loss" in risk:
        v = _opt_int(risk.get("cooldown_minutes_after_loss"), 0, 24 * 60)
        if v is not None: out["cooldown_minutes_after_loss"] = v
    # N — risk profile + between-trade cooldown. The profile lifts engine
    # floors; the explicit minute value is a per-strategy override (max() of
    # the two wins).
    if "risk_profile" in risk:
        rp = str(risk.get("risk_profile") or "medium").lower().strip()
        out["risk_profile"] = rp if rp in ("low", "medium", "high") else "medium"
    if "min_minutes_between_trades" in risk:
        v = _opt_int(risk.get("min_minutes_between_trades"), 0, 24 * 60)
        if v is not None: out["min_minutes_between_trades"] = v
    return out


def _auto_trade_to_dict(t):
    return {
        "id": t.id,
        "strategy_id": t.strategy_id,
        "symbol": t.symbol,
        "tf": t.timeframe,
        "side": t.side,
        "source": t.source,
        "entry": t.entry_price,
        "stop":  t.stop_price,
        "tp1":   t.tp1_price,
        "tp2":   t.tp2_price,
        "exit":  t.exit_price,
        "pnl_pct": t.pnl_pct,
        "pnl_usd": t.pnl_usd,
        "status":  t.status,
        "notional_usd": t.notional_usd,
        "leverage":     t.leverage,
        "plan_text":    (t.plan_text or "")[:400],
        "opened_at": t.opened_at.isoformat() if t.opened_at else None,
        "closed_at": t.closed_at.isoformat() if t.closed_at else None,
    }


@app.post("/api/trade/auto/create")
async def trade_auto_create(request: Request, db: Session = Depends(get_db)):
    """Create an auto-trade strategy from the current chart setup.

    Body shape (mirrors /api/trade/ai_read body):
        {
          "name": "BTC 5m EMA cross",
          "symbol": "BTC",
          "tf": "5m",
          "mode": "ai" | "rules",
          "indicators": [...], "toggles": {...}, "tape": {...},
          "cadence_min": 5,
          "min_odds": 60,
          "notional_usd": 1000,
          "leverage": 10,
          "tp_sl_source": "ai" | "walls" | "atr",
          "notify_telegram": true
        }
    """
    from app.models import AutoTradeStrategy
    from app.services.auto_trader import compile_rules_from_chart_state

    uid = _get_session_uid(request)
    if not uid:
        raise HTTPException(status_code=401, detail="login required")
    user = _get_user_by_uid(uid, db)
    if not user:
        raise HTTPException(status_code=401, detail="login required")
    # Auto Trader is open to all logged-in users (no subscription gate).

    try:
        body = await request.json()
    except Exception:
        body = {}
    if not isinstance(body, dict):
        body = {}

    name = (str(body.get("name") or "Auto strategy")).strip()[:60]
    symbol = (str(body.get("symbol") or "BTC")).upper().strip()
    if symbol not in TRADE_SYMBOL_WHITELIST:
        raise HTTPException(status_code=400, detail="symbol not supported")
    tf = (str(body.get("tf") or "5m")).lower().strip()
    if tf not in _AUTO_VALID_TFS:
        raise HTTPException(status_code=400, detail="invalid timeframe")
    mode = (str(body.get("mode") or "ai")).lower().strip()
    if mode not in ("ai", "rules"):
        raise HTTPException(status_code=400, detail="mode must be 'ai' or 'rules'")

    # Quota
    active_count = (db.query(AutoTradeStrategy)
                      .filter(AutoTradeStrategy.user_id == user.id,
                              AutoTradeStrategy.status != "archived")
                      .count())
    if active_count >= _AUTO_MAX_PER_USER:
        raise HTTPException(status_code=400, detail=f"Max {_AUTO_MAX_PER_USER} auto strategies. Pause or delete one first.")

    # Sanitize the chart_state — same shape as the ai_read body
    raw_indicators = body.get("indicators") or []
    if not isinstance(raw_indicators, list):
        raw_indicators = []
    _ALLOWED_TYPES = {"ema", "sma", "rsi", "macd", "bb", "vwap", "atr", "supertrend", "stochrsi", "fvg"}
    _ALLOWED_SRCS  = {"close", "open", "high", "low", "hl2", "hlc3", "ohlc4"}
    _ALLOWED_PARAM_KEYS = {"period", "fast", "slow", "signal", "mult", "stoch"}
    # FVG indicator carries its own param vocabulary (gap floors, instant
    # entry, direction filter, age cap). Numeric bounds clamp to safe ranges
    # so users can't blow up the detector with absurd values.
    _ALLOWED_FVG_PARAM_KEYS = {
        "side": ("str", {"long", "short", "either"}),
        "min_gap_pct":            ("float", 0.0, 100.0),
        "min_gap_atr_mult":       ("float", 0.0, 50.0),
        "min_gap_usd":            ("float", 0.0, 1_000_000.0),
        "disp_atr_mult":          ("float", 0.0, 50.0),
        "max_age_bars":           ("int",   1,   10000),
        "instant_entry_atr_mult": ("float", 0.0, 50.0),
        "instant_entry_max_age":  ("int",   1,   1000),
    }
    indicators: list = []
    for raw in raw_indicators[:12]:
        if not isinstance(raw, dict):
            continue
        t = str(raw.get("type", "")).lower().strip()
        if t not in _ALLOWED_TYPES:
            continue
        s = str(raw.get("src", "close")).lower().strip()
        if s not in _ALLOWED_SRCS:
            s = "close"
        rp = raw.get("params") or {}
        if not isinstance(rp, dict):
            rp = {}
        clean_params: dict = {}
        if t == "fvg":
            # Per-key validation with FVG-specific bounds + a string side enum.
            for k, v in rp.items():
                spec = _ALLOWED_FVG_PARAM_KEYS.get(k)
                if not spec:
                    continue
                if spec[0] == "str":
                    sv = str(v or "").lower().strip()
                    if sv in spec[1]:
                        clean_params[k] = sv
                else:
                    try:
                        fv = float(v)
                    except (TypeError, ValueError):
                        continue
                    lo, hi = spec[1], spec[2]
                    if fv < lo or fv > hi:
                        continue
                    clean_params[k] = int(fv) if spec[0] == "int" else fv
        else:
            for k, v in rp.items():
                if k not in _ALLOWED_PARAM_KEYS:
                    continue
                try:
                    fv = float(v)
                except (TypeError, ValueError):
                    continue
                if fv < 0 or fv > 500:
                    continue
                clean_params[k] = fv
        indicators.append({"type": t, "src": s, "params": clean_params})
    toggles = body.get("toggles") or {}
    if not isinstance(toggles, dict):
        toggles = {}
    # Coerce toggle values to bool
    toggles = {k: bool(v) for k, v in toggles.items() if isinstance(k, str)}
    tape = body.get("tape") or {}
    if not isinstance(tape, dict):
        tape = {}

    # Strategy-level modifiers (TP/SL floors + trend filter). Sanitised here
    # rather than blindly trusted so a malformed client can't poke arbitrary
    # keys into compile_rules. Each is opt-in (zero = disabled).
    raw_mods = body.get("mods") or {}
    if not isinstance(raw_mods, dict):
        raw_mods = {}
    mods: dict = {}
    def _bounded_float(name: str, lo: float, hi: float) -> None:
        try:
            v = float(raw_mods.get(name, 0) or 0)
        except (TypeError, ValueError):
            return
        if lo <= v <= hi:
            mods[name] = v
    def _bounded_int(name: str, lo: int, hi: int) -> None:
        try:
            v = int(float(raw_mods.get(name, 0) or 0))
        except (TypeError, ValueError):
            return
        if lo <= v <= hi:
            mods[name] = v
    _bounded_float("min_rr_ratio",    0.0, 20.0)
    _bounded_float("min_stop_pct",    0.0, 50.0)
    _bounded_int(  "trend_ema_period", 0,   1000)

    chart_state = {"indicators": indicators, "toggles": toggles, "tape": tape, "mods": mods}

    # Numeric knobs (clamped)
    cadence_min  = max(1,  min(60,   int(float(body.get("cadence_min")  or 5))))
    min_odds     = max(40, min(95,   int(float(body.get("min_odds")     or 60))))
    notional_usd = max(50.0, min(100000.0, float(body.get("notional_usd") or 1000.0)))
    leverage     = max(1,  min(200,  int(float(body.get("leverage")     or 10))))
    tp_sl_source = (str(body.get("tp_sl_source") or "ai")).lower().strip()
    if tp_sl_source not in ("ai", "walls", "atr"):
        tp_sl_source = "ai" if mode == "ai" else "walls"

    # N — pull risk profile early so it can flow into the rules compiler
    # (which lifts mods floors based on profile). Default 'medium' = legacy.
    raw_profile = str(body.get("risk_profile") or (body.get("risk") or {}).get("risk_profile") or "medium").lower().strip()
    risk_profile = raw_profile if raw_profile in ("low", "medium", "high") else "medium"

    rules_json: Optional[str] = None
    rules_summary: Optional[str] = None
    if mode == "rules":
        compiled, summary = compile_rules_from_chart_state(chart_state, risk_profile=risk_profile)
        if not compiled:
            # `summary` carries the error reason
            raise HTTPException(status_code=400, detail=summary)
        rules_json    = json.dumps(compiled)
        rules_summary = summary
    else:
        rules_summary = "AI reads the chart every "  + f"{cadence_min} min"

    s = AutoTradeStrategy(
        user_id          = user.id,
        name             = name,
        symbol           = symbol,
        timeframe        = tf,
        mode             = mode,
        chart_state_json = json.dumps(chart_state),
        rules_json       = rules_json,
        rules_summary    = rules_summary,
        cadence_min      = cadence_min,
        min_odds         = min_odds,
        notional_usd     = notional_usd,
        leverage         = leverage,
        tp_sl_source     = tp_sl_source,
        status           = "active",
        notify_telegram  = bool(body.get("notify_telegram", True)),
    )
    # Apply optional risk caps / sizing / partial / session / cooldown fields.
    for k, v in _read_risk_fields(body).items():
        setattr(s, k, v)
    db.add(s)
    db.commit()
    db.refresh(s)
    return {"ok": True, "strategy": _auto_strategy_to_dict(s)}


@app.post("/api/trade/auto/{sid}/update")
async def trade_auto_update(sid: int, request: Request, db: Session = Depends(get_db)):
    """Edit a subset of an existing strategy's settings without re-creating it.

    Only the small "knobs" the user typically tunes are mutable here — name,
    notify, cadence/odds, sizing, risk caps, partial TP, session, cooldown.
    The chart state, mode, symbol, timeframe are immutable; clone instead.
    """
    from app.models import AutoTradeStrategy
    uid = _get_session_uid(request)
    if not uid:
        raise HTTPException(status_code=401, detail="login required")
    user = _get_user_by_uid(uid, db)
    if not user:
        raise HTTPException(status_code=401, detail="login required")
    s = db.query(AutoTradeStrategy).filter(AutoTradeStrategy.id == sid).first()
    if not s or s.user_id != user.id:
        raise HTTPException(status_code=404, detail="strategy not found")

    try:
        body = await request.json()
    except Exception:
        body = {}
    if not isinstance(body, dict):
        body = {}

    if "name" in body:
        s.name = (str(body["name"]) or "").strip()[:60] or s.name
    if "notify_telegram" in body:
        s.notify_telegram = bool(body["notify_telegram"])
    if "cadence_min" in body:
        try:
            s.cadence_min = max(1, min(60, int(float(body["cadence_min"]))))
        except (TypeError, ValueError): pass
    if "min_odds" in body:
        try:
            s.min_odds = max(40, min(95, int(float(body["min_odds"]))))
        except (TypeError, ValueError): pass
    if "notional_usd" in body:
        try:
            s.notional_usd = max(50.0, min(100000.0, float(body["notional_usd"])))
        except (TypeError, ValueError): pass
    if "leverage" in body:
        try:
            s.leverage = max(1, min(200, int(float(body["leverage"]))))
        except (TypeError, ValueError): pass
    if "tp_sl_source" in body:
        v = (str(body["tp_sl_source"]) or "").lower().strip()
        if v in ("ai", "walls", "atr"):
            s.tp_sl_source = v

    # Risk caps + sizing + partial + session + cooldown fields
    risk_fields = _read_risk_fields(body)
    profile_changed = (
        "risk_profile" in risk_fields
        and risk_fields["risk_profile"] != (getattr(s, "risk_profile", None) or "medium")
    )
    for k, v in risk_fields.items():
        setattr(s, k, v)

    # N — when a rules-mode strategy's risk_profile changes, recompile its
    # rules so the new profile floors (R:R, stop, trend EMA) actually take
    # effect. Without this, a user flipping medium→low would only get the
    # min_odds + between-trade gates without the modifier lifts.
    if profile_changed and (s.mode or "") == "rules":
        try:
            from app.services.auto_trader import compile_rules_from_chart_state
            cs = json.loads(s.chart_state_json or "{}")
            recompiled, summary = compile_rules_from_chart_state(
                cs, risk_profile=s.risk_profile,
            )
            if recompiled:
                s.rules_json    = json.dumps(recompiled)
                s.rules_summary = summary
        except Exception as _e:
            logger.warning(f"Rules recompile after profile change failed for #{s.id}: {_e}")

    # Allow user to manually clear an active cooldown ("resume now")
    if body.get("clear_cooldown") is True:
        s.paused_until = None
        s.consecutive_losses = 0

    db.commit()
    db.refresh(s)
    return {"ok": True, "strategy": _auto_strategy_to_dict(s)}


@app.get("/api/trade/auto/list")
async def trade_auto_list(request: Request, db: Session = Depends(get_db)):
    """List the current user's auto strategies (active + paused).

    Open to any logged-in user — no subscription gate."""
    from app.models import AutoTradeStrategy, AutoTradePaperTrade
    uid = _get_session_uid(request)
    if not uid:
        raise HTTPException(status_code=401, detail="login required")
    user = _get_user_by_uid(uid, db)
    if not user:
        raise HTTPException(status_code=401, detail="login required")
    rows = (db.query(AutoTradeStrategy)
              .filter(AutoTradeStrategy.user_id == user.id,
                      AutoTradeStrategy.status != "archived")
              .order_by(AutoTradeStrategy.created_at.desc())
              .all())

    # Batch the "last trade per strategy" lookup into a single round-trip —
    # previously this was N+1 (one SELECT per strategy), which on Neon was
    # blowing past the statement_timeout once a user had many strategies and
    # many historical paper trades. DISTINCT ON keeps just the newest row per
    # strategy_id in one indexed scan.
    last_by_sid: dict[int, "AutoTradePaperTrade"] = {}
    sids = [s.id for s in rows]
    if sids:
        from sqlalchemy import text as _sql_text
        latest_ids = [
            r[0] for r in db.execute(
                _sql_text(
                    "SELECT DISTINCT ON (strategy_id) id "
                    "FROM auto_trade_paper_trades "
                    "WHERE strategy_id = ANY(:sids) "
                    "ORDER BY strategy_id, opened_at DESC NULLS LAST"
                ),
                {"sids": sids},
            ).fetchall()
        ]
        if latest_ids:
            for t in (db.query(AutoTradePaperTrade)
                        .filter(AutoTradePaperTrade.id.in_(latest_ids))
                        .all()):
                last_by_sid[t.strategy_id] = t

    out = []
    for s in rows:
        last = last_by_sid.get(s.id)
        out.append(_auto_strategy_to_dict(s, last_trade=_auto_trade_to_dict(last) if last else None))
    return {"ok": True, "strategies": out}


@app.get("/api/trade/auto/{sid}/trades")
async def trade_auto_trades(sid: int, request: Request, db: Session = Depends(get_db)):
    """List paper trades for one strategy (newest first, last 100)."""
    from app.models import AutoTradeStrategy, AutoTradePaperTrade
    uid = _get_session_uid(request)
    if not uid:
        raise HTTPException(status_code=401, detail="login required")
    user = _get_user_by_uid(uid, db)
    if not user:
        raise HTTPException(status_code=401, detail="login required")
    s = db.query(AutoTradeStrategy).filter(AutoTradeStrategy.id == sid).first()
    if not s or s.user_id != user.id:
        raise HTTPException(status_code=404, detail="strategy not found")
    trades = (db.query(AutoTradePaperTrade)
                .filter(AutoTradePaperTrade.strategy_id == sid)
                .order_by(AutoTradePaperTrade.opened_at.desc())
                .limit(100)
                .all())
    return {
        "ok": True,
        "strategy": _auto_strategy_to_dict(s),
        "trades": [_auto_trade_to_dict(t) for t in trades],
    }


@app.post("/api/trade/auto/{sid}/pause")
async def trade_auto_pause(sid: int, request: Request, db: Session = Depends(get_db)):
    from app.models import AutoTradeStrategy
    uid = _get_session_uid(request)
    if not uid:
        raise HTTPException(status_code=401, detail="login required")
    user = _get_user_by_uid(uid, db)
    if not user:
        raise HTTPException(status_code=401, detail="login required")
    s = db.query(AutoTradeStrategy).filter(AutoTradeStrategy.id == sid).first()
    if not s or s.user_id != user.id:
        raise HTTPException(status_code=404, detail="strategy not found")
    s.status = "paused"
    s.paused_at = datetime.utcnow()
    db.commit()
    return {"ok": True, "status": "paused"}


@app.post("/api/trade/auto/{sid}/resume")
async def trade_auto_resume(sid: int, request: Request, db: Session = Depends(get_db)):
    from app.models import AutoTradeStrategy
    uid = _get_session_uid(request)
    if not uid:
        raise HTTPException(status_code=401, detail="login required")
    user = _get_user_by_uid(uid, db)
    if not user:
        raise HTTPException(status_code=401, detail="login required")
    s = db.query(AutoTradeStrategy).filter(AutoTradeStrategy.id == sid).first()
    if not s or s.user_id != user.id:
        raise HTTPException(status_code=404, detail="strategy not found")
    s.status = "active"
    s.paused_at = None
    db.commit()
    return {"ok": True, "status": "active"}


@app.delete("/api/trade/auto/{sid}")
async def trade_auto_delete(sid: int, request: Request, db: Session = Depends(get_db)):
    from app.models import AutoTradeStrategy
    uid = _get_session_uid(request)
    if not uid:
        raise HTTPException(status_code=401, detail="login required")
    user = _get_user_by_uid(uid, db)
    if not user:
        raise HTTPException(status_code=401, detail="login required")
    s = db.query(AutoTradeStrategy).filter(AutoTradeStrategy.id == sid).first()
    if not s or s.user_id != user.id:
        raise HTTPException(status_code=404, detail="strategy not found")
    s.status = "archived"
    db.commit()
    return {"ok": True}


# ─────────────────────────────────────────────────────────────────────────────
# Per-strategy performance dashboard (L)
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/api/trade/auto/{sid}/stats")
async def trade_auto_stats(sid: int, request: Request, db: Session = Depends(get_db)):
    """Return rich performance analytics for one strategy.

    Includes the equity curve (cumulative P&L by close-time), running
    drawdown vs the running peak, win-rate broken out by side and by
    UTC hour-of-day, and basic R:R achieved. The frontend renders this
    as a small dashboard modal — no chart library needed (inline SVG).
    """
    from app.models import AutoTradeStrategy, AutoTradePaperTrade
    uid = _get_session_uid(request)
    if not uid:
        raise HTTPException(status_code=401, detail="login required")
    user = _get_user_by_uid(uid, db)
    if not user:
        raise HTTPException(status_code=401, detail="login required")
    s = db.query(AutoTradeStrategy).filter(AutoTradeStrategy.id == sid).first()
    if not s or s.user_id != user.id:
        raise HTTPException(status_code=404, detail="strategy not found")

    closed = (db.query(AutoTradePaperTrade)
                .filter(AutoTradePaperTrade.strategy_id == sid,
                        AutoTradePaperTrade.status.in_(("tp1_hit", "stop_hit",
                                                        "tp2_hit", "manual_close",
                                                        "expired")),
                        AutoTradePaperTrade.closed_at.isnot(None))
                .order_by(AutoTradePaperTrade.closed_at.asc())
                .all())

    equity_curve  = []      # list of [iso_ts, cumulative_pnl_usd]
    drawdown      = []      # list of [iso_ts, dd_usd_from_peak]
    by_hour       = {}      # hour_int -> {wins, losses, pnl}
    by_side       = {"long": {"n": 0, "wins": 0, "pnl": 0.0},
                     "short": {"n": 0, "wins": 0, "pnl": 0.0}}
    rr_planned: list = []   # planned reward:risk per trade
    rr_achieved: list = []  # achieved R (price move ÷ risk)

    cum  = 0.0
    peak = 0.0
    win_count  = 0
    loss_count = 0
    pnl_total  = 0.0
    for t in closed:
        pnl = float(t.pnl_usd or 0)
        cum += pnl
        peak = max(peak, cum)
        equity_curve.append([t.closed_at.isoformat(), round(cum, 2)])
        drawdown.append([t.closed_at.isoformat(), round(cum - peak, 2)])
        is_win = pnl >= 0
        if is_win: win_count += 1
        else:      loss_count += 1
        pnl_total += pnl

        h = t.closed_at.hour
        b = by_hour.setdefault(h, {"wins": 0, "losses": 0, "pnl": 0.0})
        if is_win: b["wins"]   += 1
        else:      b["losses"] += 1
        b["pnl"] += pnl

        sd = by_side.get(t.side) or by_side.setdefault(t.side, {"n": 0, "wins": 0, "pnl": 0.0})
        sd["n"]   += 1
        sd["pnl"] += pnl
        if is_win: sd["wins"] += 1

        # R:R math — risk = entry → stop, reward_planned = entry → tp1
        if t.entry_price and t.stop_price and t.tp1_price:
            risk    = abs(t.entry_price - t.stop_price)
            reward  = abs(t.tp1_price - t.entry_price)
            if risk > 0:
                rr_planned.append(round(reward / risk, 2))
                if t.exit_price:
                    achieved = abs(t.exit_price - t.entry_price)
                    sign = 1 if (
                        (t.side == "long"  and t.exit_price >= t.entry_price) or
                        (t.side == "short" and t.exit_price <= t.entry_price)
                    ) else -1
                    rr_achieved.append(round(sign * achieved / risk, 2))

    n = len(closed)
    avg_win  = round(sum(float(t.pnl_usd or 0) for t in closed if (t.pnl_usd or 0) >  0) / max(1, win_count),  2)
    avg_loss = round(sum(float(t.pnl_usd or 0) for t in closed if (t.pnl_usd or 0) <= 0) / max(1, loss_count), 2)

    return {
        "ok": True,
        "strategy": _auto_strategy_to_dict(s),
        "stats": {
            "n_closed":       n,
            "wins":           win_count,
            "losses":         loss_count,
            "win_rate":       round(100.0 * win_count / n, 1) if n else None,
            "pnl_total":      round(pnl_total, 2),
            "avg_win":        avg_win,
            "avg_loss":       avg_loss,
            "max_drawdown":   round(min((d[1] for d in drawdown), default=0.0), 2),
            "current_drawdown": round((cum - peak), 2),
            "peak_equity":    round(peak, 2),
            "equity_curve":   equity_curve,
            "drawdown":       drawdown,
            "by_hour":        {str(k): v for k, v in sorted(by_hour.items())},
            "by_side":        by_side,
            "rr_planned_avg": round(sum(rr_planned)  / len(rr_planned),  2) if rr_planned  else None,
            "rr_achieved_avg":round(sum(rr_achieved) / len(rr_achieved), 2) if rr_achieved else None,
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# Strategy backtest replay (M)
# ─────────────────────────────────────────────────────────────────────────────
@app.post("/api/trade/auto/{sid}/backtest")
async def trade_auto_backtest(sid: int, request: Request, db: Session = Depends(get_db)):
    """Replay a strategy against the last N candles and return a synthetic
    equity curve + summary. Only Rules-mode strategies are replayed live —
    AI-mode strategies are skipped because each scan would cost an LLM call.

    Body:
        {"bars": 500}   # how many recent candles to walk over (max 1500)
    """
    from app.models import AutoTradeStrategy
    from app.services.auto_trader import (
        evaluate_rules, derive_stop_tp_from_chart, check_position_close, compute_pnl,
        _fetch_candles,
    )

    uid = _get_session_uid(request)
    if not uid:
        raise HTTPException(status_code=401, detail="login required")
    user = _get_user_by_uid(uid, db)
    if not user:
        raise HTTPException(status_code=401, detail="login required")
    s = db.query(AutoTradeStrategy).filter(AutoTradeStrategy.id == sid).first()
    if not s or s.user_id != user.id:
        raise HTTPException(status_code=404, detail="strategy not found")
    if s.mode != "rules":
        return {"ok": False, "error": "Backtest replay currently supports rules-mode strategies only. AI-mode would re-run the LLM on every bar."}

    try:
        body = await request.json()
    except Exception:
        body = {}
    bars = max(50, min(1500, int(body.get("bars") or 500)))

    # Pull the same candles the live engine would see — reuse its helper so
    # we don't duplicate the MEXC request shape or caching logic.
    try:
        candles = await _fetch_candles(s.symbol, s.timeframe, limit=bars)
    except Exception as e:
        logger.warning(f"backtest candle fetch failed: {e}")
        raise HTTPException(status_code=503, detail="candle fetch failed")
    if not candles or len(candles) < 60:
        return {"ok": False, "error": f"only {len(candles)} candles available — need ≥ 60"}

    rules = json.loads(s.rules_json or "{}")
    if not rules.get("entry"):
        return {"ok": False, "error": "strategy has no compiled rules"}

    fills: list = []        # closed simulated trades
    open_trade            = None
    cum                   = 0.0
    peak                  = 0.0
    equity_curve          = []
    drawdown              = []

    # Walk forward — at each bar, the engine would have known about candles[:i+1].
    # Try to open on the close of bar i (no lookahead), then check fill on bars i+1…
    for i in range(40, len(candles) - 1):
        window = candles[: i + 1]
        if open_trade is None:
            hit = evaluate_rules(rules, window, walls=[], tape={})
            if hit:
                side  = hit["side"]
                entry = hit["entry_price"]
                stop, tp1, tp2 = derive_stop_tp_from_chart(side, entry, window,
                                                           walls=None, source="atr")
                # Sanity: SL/TP must straddle entry for the side
                if side == "long" and not (stop < entry < tp1):
                    continue
                if side == "short" and not (tp1 < entry < stop):
                    continue
                open_trade = {
                    "side": side, "entry": entry, "stop": stop,
                    "tp1": tp1, "tp2": tp2, "open_idx": i,
                }
        else:
            # Simulate close on the next bar's range
            class _T:
                opened_at = datetime.utcfromtimestamp(int(candles[open_trade["open_idx"]]["time"]))
                side = open_trade["side"]
                stop_price = open_trade["stop"]
                tp1_price  = open_trade["tp1"]
                entry_price= open_trade["entry"]
            res = check_position_close(_T(), [candles[i + 1]])
            if res:
                reason, exit_p = res
                pnl_pct, pnl_usd = compute_pnl(_T.side, _T.entry_price, exit_p,
                                               s.notional_usd or 1000, s.leverage or 10)
                fills.append({
                    "open_time":  candles[open_trade["open_idx"]]["time"],
                    "close_time": candles[i + 1]["time"],
                    "side": _T.side, "entry": _T.entry_price, "exit": exit_p,
                    "reason": reason, "pnl_usd": round(pnl_usd, 2),
                })
                cum  += pnl_usd
                peak  = max(peak, cum)
                equity_curve.append([candles[i + 1]["time"], round(cum, 2)])
                drawdown    .append([candles[i + 1]["time"], round(cum - peak, 2)])
                open_trade = None

    wins  = sum(1 for f in fills if f["pnl_usd"] >= 0)
    losses= len(fills) - wins
    return {
        "ok": True,
        "summary": {
            "bars_replayed": len(candles),
            "n_trades":      len(fills),
            "wins":          wins,
            "losses":        losses,
            "win_rate":      round(100.0 * wins / len(fills), 1) if fills else None,
            "pnl_usd":       round(cum, 2),
            "max_drawdown":  round(min((d[1] for d in drawdown), default=0.0), 2),
        },
        "fills":        fills,
        "equity_curve": equity_curve,
        "drawdown":     drawdown,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Funding rate + open interest overlay (C) — OKX public APIs (no key required)
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/api/trade/funding/{symbol}")
async def trade_funding(symbol: str):
    """Pull funding rate + open interest summary for the chart sidebar.

    Uses the shared `_fetch_funding_oi` helper in `auto_trader` so the chart
    widget, the AI Trade Read button, and the AI-mode auto-trader fleet all
    share one upstream call (60s in-process cache). Backed by OKX's free
    public APIs — no key, no subscription, not geoblocked from Replit.
    """
    sym = (symbol or "").upper().strip()
    if sym not in TRADE_SYMBOL_WHITELIST:
        return JSONResponse({"error": "symbol not supported"}, status_code=404)
    try:
        from app.services.auto_trader import _fetch_funding_oi
        out = await _fetch_funding_oi(sym)
    except Exception as e:
        logger.warning(f"trade_funding({sym}) fetch failed: {e}")
        return JSONResponse({"error": f"upstream failed: {e}"}, status_code=502)
    if not out:
        return JSONResponse({"error": "funding data unavailable"}, status_code=503)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Quick Trade — one-tap manual market order on the mobile Trade tab.
# Routes through BitunixTrader.place_trade so it reuses the same affiliate gate,
# margin/leverage/limits pre-flight, and post-fill TP/SL correction as auto trades.
# Body: {symbol, side: "LONG"|"SHORT", leverage:int, position_usd:float,
#        tp_pct?:float, sl_pct?:float}
# Auth: X-TradeHub-UID. Live orders gated by the same env flag + per-user
# affiliate roster check as every other live trade in the system.
# ─────────────────────────────────────────────────────────────────────────────
@app.post("/api/trade/quick")
async def trade_quick(request: Request):
    uid = request.query_params.get("uid", "").strip().upper()
    if not uid:
        raise HTTPException(status_code=401, detail="auth required")
    try:
        body = await request.json()
    except Exception:
        body = {}

    import math as _math
    sym  = (body.get("symbol") or "").upper().strip()
    side = (body.get("side") or "").upper().strip()
    try:
        leverage     = int(body.get("leverage") or 10)
        position_usd = float(body.get("position_usd") or 0)
    except (TypeError, ValueError):
        raise HTTPException(status_code=400, detail="leverage and position_usd must be numeric")
    if not _math.isfinite(position_usd):
        raise HTTPException(status_code=400, detail="position_usd must be a finite number")
    tp_pct = body.get("tp_pct")
    sl_pct = body.get("sl_pct")
    try:
        tp_pct = float(tp_pct) if tp_pct not in (None, "") else None
        sl_pct = float(sl_pct) if sl_pct not in (None, "") else None
    except (TypeError, ValueError):
        raise HTTPException(status_code=400, detail="tp_pct/sl_pct must be numeric or null")
    if tp_pct is not None and not _math.isfinite(tp_pct):
        raise HTTPException(status_code=400, detail="tp_pct must be finite")
    if sl_pct is not None and not _math.isfinite(sl_pct):
        raise HTTPException(status_code=400, detail="sl_pct must be finite")

    if sym not in TRADE_SYMBOL_WHITELIST:
        raise HTTPException(status_code=400, detail="symbol not supported")
    if side not in ("LONG", "SHORT"):
        raise HTTPException(status_code=400, detail="side must be LONG or SHORT")
    if leverage < 1 or leverage > 125:
        raise HTTPException(status_code=400, detail="leverage out of range (1-125)")
    if position_usd < 5 or position_usd > 100_000:
        raise HTTPException(status_code=400, detail="position_usd out of range ($5-$100k)")
    if tp_pct is not None and (tp_pct <= 0 or tp_pct > 1000):
        raise HTTPException(status_code=400, detail="tp_pct out of range")
    if sl_pct is not None and (sl_pct <= 0 or sl_pct >= 100):
        raise HTTPException(status_code=400, detail="sl_pct out of range (0-100 exclusive)")

    db = SessionLocal()
    try:
        user = _get_user_by_uid(uid, db)
        if not user:
            raise HTTPException(status_code=403, detail="invalid UID")

        # Bitunix creds + UID live on UserPreference, NOT User. Stored encrypted —
        # decrypt before passing to BitunixTrader (matches strategy_executor pattern).
        from app.models import UserPreference
        from app.utils.encryption import decrypt_api_key
        prefs = db.query(UserPreference).filter(UserPreference.user_id == user.id).first()
        enc_key = (getattr(prefs, "bitunix_api_key", None) if prefs else None) or ""
        enc_sec = (getattr(prefs, "bitunix_api_secret", None) if prefs else None) or ""
        if not enc_key or not enc_sec:
            return JSONResponse({
                "ok": False,
                "error": "NO_API_KEY",
                "message": "Add your Bitunix API key in Settings before placing live trades.",
            }, status_code=400)
        try:
            api_key = decrypt_api_key(enc_key)
            api_sec = decrypt_api_key(enc_sec)
        except Exception as e:
            logger.warning(f"[quick_trade] decrypt failed uid={uid}: {e}")
            return JSONResponse({
                "ok": False,
                "error": "KEY_DECRYPT_FAILED",
                "message": "Could not decrypt your Bitunix API key. Re-enter it in Settings.",
            }, status_code=500)
        bitunix_uid_pref = (getattr(prefs, "bitunix_uid", None) if prefs else None) or None

        from app.services.bitunix_trader import BitunixTrader
        trader = BitunixTrader(api_key=api_key, api_secret=api_sec)
        # Tag with the user's bitunix_uid so the affiliate gate inside place_trade
        # can verify this user is under our master account (fail-closed).
        try:
            setattr(trader, "bitunix_uid", bitunix_uid_pref)
        except Exception:
            pass

        try:
            live_price = await trader.get_current_price(f"{sym}/USDT")
        except Exception as e:
            logger.warning(f"[quick_trade] get_current_price failed: {e}")
            live_price = None
        if not live_price or live_price <= 0:
            return JSONResponse({
                "ok": False,
                "error": "PRICE_UNAVAILABLE",
                "message": f"Could not fetch live price for {sym}.",
            }, status_code=502)

        # Convert TP/SL pcts → absolute prices using the live price as entry hint.
        # Direction-aware: LONG TP above / SL below; SHORT inverted.
        tp_price = None
        sl_price = None
        if tp_pct is not None:
            tp_price = live_price * (1 + tp_pct / 100.0) if side == "LONG" else live_price * (1 - tp_pct / 100.0)
        if sl_pct is not None:
            sl_price = live_price * (1 - sl_pct / 100.0) if side == "LONG" else live_price * (1 + sl_pct / 100.0)

        try:
            result = await trader.place_trade(
                symbol=f"{sym}/USDT",
                direction=side,
                entry_price=live_price,
                stop_loss=sl_price if sl_price is not None else (
                    live_price * 0.95 if side == "LONG" else live_price * 1.05
                ),
                take_profit=tp_price if tp_price is not None else (
                    live_price * 1.10 if side == "LONG" else live_price * 0.90
                ),
                position_size_usdt=position_usd,
                leverage=leverage,
                live_price_hint=live_price,
            )
        except Exception as e:
            logger.error(f"[quick_trade] place_trade exception for uid={uid}: {e}")
            return JSONResponse({
                "ok": False,
                "error": "PLACE_FAILED",
                "message": str(e)[:240],
            }, status_code=502)
        finally:
            try:
                await trader.close()
            except Exception:
                pass

        if not result or not result.get("success"):
            err = (result or {}).get("error", "unknown") if result else "no_response"
            # AFFILIATE_GATE → user not under our master; tell them how to fix it
            if isinstance(err, str) and err.startswith("AFFILIATE_GATE"):
                return JSONResponse({
                    "ok": False,
                    "error": "AFFILIATE_GATE",
                    "message": "Your Bitunix UID isn't linked to TradeHub yet. Sign up via the affiliate link in Settings.",
                }, status_code=403)
            return JSONResponse({
                "ok": False,
                "error": "ORDER_REJECTED",
                "message": str(err)[:240],
            }, status_code=502)

        # Success — fire push notification (best-effort).
        try:
            from app.services.expo_push import notify_user_bg
            notify_user_bg(
                user.id,
                title=f"⚡ Quick {side} {sym}",
                body=f"{leverage}× · ${position_usd:.0f} @ ${live_price:,.4f}",
                data={"type": "quick_trade", "symbol": sym, "side": side},
                kind="manual",
                position_usd=float(position_usd),
            )
        except Exception as _push_err:
            logger.debug(f"[quick_trade] push send skipped: {_push_err}")

        return JSONResponse({
            "ok": True,
            "order_id": result.get("order_id"),
            "symbol": sym,
            "side": side,
            "leverage": leverage,
            "entry_price": live_price,
            "tp_price": tp_price,
            "sl_price": sl_price,
            "position_usd": position_usd,
        })
    finally:
        db.close()


# ─────────────────────────────────────────────────────────────────────────────
# Fair Value Gap (FVG) overlay — ICT 3-candle gap detection for /trade
# ─────────────────────────────────────────────────────────────────────────────
# Re-uses the candle source the chart already loads, runs them through the
# detector in app/services/auto_trader.py (single source of truth so the
# overlay you SEE matches the rule the Auto Trader EVALUATES), and caches per
# (pair, tf) for ~25 s to absorb the 30 s frontend poll without melting MEXC.

_FVG_TTL = 8  # seconds — was 25; user wants FVG scanned as fast as possible (frontend polls every 10s)

@app.get("/api/trade/fvg/{symbol}")
async def trade_fvg(symbol: str, tf: str = "5m",
                    min_gap_pct: float = 0.0,
                    min_gap_atr_mult: float = 0.10,
                    disp_atr_mult: float = 0.5,
                    max_age_bars: int = 200,
                    only_unfilled: int = 1,
                    limit: int = 30):
    """Return active fair-value gaps for the requested symbol/timeframe.

    Gaps are ICT-style 3-bar formations and are computed live from the same
    MEXC candles the chart renders. ``only_unfilled=1`` (the default) hides
    any gap a later candle has already traded through.

    Quality filters (default ON):
      * ``min_gap_atr_mult`` — width must be ≥ this × ATR(14). Volatility-aware,
        so a single setting works across symbols + timeframes.
      * ``disp_atr_mult`` — formation candle body must be ≥ this × ATR(14).
        ICT "displacement" filter — keeps strong-expansion gaps, rejects dojis.
      * ``min_gap_pct`` — legacy fixed % filter; default 0 (off) since the
        ATR filters handle width quality.
    """
    sym = (symbol or "").upper()
    if sym not in TRADE_SYMBOL_WHITELIST:
        return JSONResponse({"error": f"unknown symbol {sym}"}, status_code=400)
    pair = TRADE_SYMBOL_WHITELIST[sym]
    if tf not in _TRADE_TF_MAP:
        return JSONResponse({"error": f"bad tf {tf}"}, status_code=400)

    # Clamp inputs.
    min_gap_pct      = max(0.0, min(5.0, float(min_gap_pct or 0.0)))
    min_gap_atr_mult = max(0.0, min(5.0, float(min_gap_atr_mult or 0.0)))
    disp_atr_mult    = max(0.0, min(5.0, float(disp_atr_mult or 0.0)))
    max_age_bars     = max(10, min(500, int(max_age_bars or 200)))
    limit            = max(1,  min(100, int(limit or 30)))
    only_unfilled    = bool(only_unfilled)

    cache_key = (
        f"trade_fvg_{pair}_{tf}_{min_gap_pct}_{min_gap_atr_mult}_"
        f"{disp_atr_mult}_{max_age_bars}_{int(only_unfilled)}_{limit}"
    )
    hit = _CACHE.get(cache_key)
    if hit and hit[1] > time.time():
        return hit[0]

    interval = _TRADE_TF_MAP[tf]

    # Pull a generous window of MEXC klines (same source as /api/trade/candles
    # so the overlay is byte-identical to what the chart already drew).
    candles: List[dict] = []
    try:
        import httpx
        async with httpx.AsyncClient(timeout=5.0) as client:
            r = await client.get(
                "https://api.mexc.com/api/v3/klines",
                params={"symbol": pair, "interval": interval, "limit": 500},
            )
            r.raise_for_status()
            rows = r.json() or []
        for k in rows:
            try:
                candles.append({
                    "time":  int(k[0]) // 1000,
                    "open":  float(k[1]),
                    "high":  float(k[2]),
                    "low":   float(k[3]),
                    "close": float(k[4]),
                })
            except (TypeError, ValueError, IndexError):
                continue
    except Exception as e:
        logger.warning(f"trade_fvg({sym}) candles fetch failed: {e}")
        return JSONResponse({"error": f"upstream failed: {e}"}, status_code=502)

    if not candles:
        out = {"symbol": sym, "tf": tf, "gaps": [], "count": 0,
               "ts": int(time.time())}
        _CACHE[cache_key] = (out, time.time() + _FVG_TTL)
        return out

    try:
        from app.services.auto_trader import detect_fvgs
        gaps = detect_fvgs(
            candles,
            min_gap_pct=min_gap_pct,
            min_gap_atr_mult=min_gap_atr_mult,
            disp_atr_mult=disp_atr_mult,
            only_unfilled=only_unfilled,
            max_age_bars=max_age_bars,
            max_results=limit,
        )
    except Exception as e:
        logger.warning(f"trade_fvg({sym}) detect failed: {e}")
        return JSONResponse({"error": f"detect failed: {e}"}, status_code=500)

    out = {
        "symbol": sym,
        "tf":     tf,
        "gaps":   gaps,
        "count":  len(gaps),
        "ts":     int(time.time()),
    }
    _CACHE[cache_key] = (out, time.time() + _FVG_TTL)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Multi-symbol watchlist (F) — light ticker poll for the sidebar
# ─────────────────────────────────────────────────────────────────────────────
_WATCHLIST_PAIRS = [
    (sym, pair) for sym, pair in TRADE_SYMBOL_WHITELIST.items()
]

@app.get("/api/trade/watchlist")
async def trade_watchlist():
    """Return a small rotating list of futures tickers (last + 24h % change).

    Cached 6s — short enough to look live, long enough to share a single MEXC
    round-trip across the gunicorn workers.
    """
    cache_key = "trade_watchlist_v1"
    cached = _CACHE.get(cache_key)
    if cached and time.time() < cached[1]:
        return cached[0]

    out = []
    try:
        import httpx
        async with httpx.AsyncClient(timeout=8.0) as cl:
            r = await cl.get("https://api.mexc.com/api/v3/ticker/24hr")
            r.raise_for_status()
            rows = r.json() or []
        idx = {x.get("symbol"): x for x in rows if isinstance(x, dict)}
        for sym, pair in _WATCHLIST_PAIRS:
            t = idx.get(pair)
            if not t:
                continue
            try:
                out.append({
                    "symbol": sym,
                    "pair":   pair,
                    "price":  float(t.get("lastPrice", 0) or 0),
                    "pct":    float(t.get("priceChangePercent", 0) or 0),
                    "vol_quote_usd": float(t.get("quoteVolume", 0) or 0),
                })
            except (TypeError, ValueError):
                continue
    except Exception as e:
        logger.warning(f"trade_watchlist failed: {e}")
        return JSONResponse({"error": "ticker fetch failed"}, status_code=502)

    payload = {"ok": True, "tickers": out, "fetched_at": int(time.time())}
    _CACHE[cache_key] = (payload, time.time() + 6)
    return payload


# ─────────────────────────────────────────────────────────────────────────────
# Persistent chart drawings (D)
# ─────────────────────────────────────────────────────────────────────────────
_DRAWING_KINDS = {"trendline", "hline", "fib", "rect", "note"}
_MAX_DRAWINGS_PER_SYMBOL = 50

def _drawing_to_dict(d):
    return {
        "id": d.id,
        "symbol": d.symbol,
        "tf": d.timeframe,
        "kind": d.kind,
        "points": json.loads(d.points_json or "[]"),
        "color": d.color,
        "label": d.label,
        "created_at": d.created_at.isoformat() if d.created_at else None,
        "updated_at": d.updated_at.isoformat() if d.updated_at else None,
    }


@app.get("/api/trade/drawings")
async def trade_drawings_list(request: Request, symbol: str = "BTC",
                              db: Session = Depends(get_db)):
    """List the user's saved drawings for a symbol."""
    from app.models import TradeDrawing
    uid = _get_session_uid(request)
    if not uid:
        return {"ok": True, "drawings": []}      # anonymous → no drawings, no error
    user = _get_user_by_uid(uid, db)
    if not user:
        return {"ok": True, "drawings": []}
    sym = (symbol or "BTC").upper().strip()
    rows = (db.query(TradeDrawing)
              .filter(TradeDrawing.user_id == user.id,
                      TradeDrawing.symbol == sym)
              .order_by(TradeDrawing.created_at.desc())
              .limit(_MAX_DRAWINGS_PER_SYMBOL)
              .all())
    return {"ok": True, "drawings": [_drawing_to_dict(d) for d in rows]}


@app.post("/api/trade/drawings")
async def trade_drawings_create(request: Request, db: Session = Depends(get_db)):
    """Save a drawing. Body: {symbol, tf?, kind, points:[{time,price}], color?, label?}"""
    from app.models import TradeDrawing
    uid = _get_session_uid(request)
    if not uid:
        raise HTTPException(status_code=401, detail="login required")
    user = _get_user_by_uid(uid, db)
    if not user:
        raise HTTPException(status_code=401, detail="login required")
    try:
        body = await request.json()
    except Exception:
        body = {}
    if not isinstance(body, dict):
        body = {}
    sym  = (str(body.get("symbol") or "BTC")).upper().strip()
    kind = (str(body.get("kind") or "")).lower().strip()
    if kind not in _DRAWING_KINDS:
        raise HTTPException(status_code=400, detail=f"kind must be one of {sorted(_DRAWING_KINDS)}")
    pts  = body.get("points") or []
    if not isinstance(pts, list) or not pts:
        raise HTTPException(status_code=400, detail="points required (non-empty list)")
    # Sanitize points — keep only numeric {time, price}
    clean: list = []
    for p in pts[:8]:
        if not isinstance(p, dict): continue
        try:
            clean.append({"time": int(p.get("time")), "price": float(p.get("price"))})
        except (TypeError, ValueError):
            continue
    if not clean:
        raise HTTPException(status_code=400, detail="all points were invalid")

    # Quota: at most _MAX_DRAWINGS_PER_SYMBOL per symbol
    cur = (db.query(TradeDrawing)
             .filter(TradeDrawing.user_id == user.id,
                     TradeDrawing.symbol == sym)
             .count())
    if cur >= _MAX_DRAWINGS_PER_SYMBOL:
        raise HTTPException(status_code=400, detail=f"max {_MAX_DRAWINGS_PER_SYMBOL} drawings per symbol")

    d = TradeDrawing(
        user_id     = user.id,
        symbol      = sym,
        timeframe   = (str(body.get("tf") or "") or None) and str(body.get("tf"))[:8],
        kind        = kind,
        points_json = json.dumps(clean),
        color       = (str(body.get("color") or "") or None),
        label       = (str(body.get("label") or "") or None),
    )
    db.add(d); db.commit(); db.refresh(d)
    return {"ok": True, "drawing": _drawing_to_dict(d)}


@app.delete("/api/trade/drawings/{did}")
async def trade_drawings_delete(did: int, request: Request, db: Session = Depends(get_db)):
    from app.models import TradeDrawing
    uid = _get_session_uid(request)
    if not uid:
        raise HTTPException(status_code=401, detail="login required")
    user = _get_user_by_uid(uid, db)
    if not user:
        raise HTTPException(status_code=401, detail="login required")
    d = db.query(TradeDrawing).filter(TradeDrawing.id == did,
                                      TradeDrawing.user_id == user.id).first()
    if not d:
        raise HTTPException(status_code=404, detail="drawing not found")
    db.delete(d); db.commit()
    return {"ok": True}


# ─────────────────────────────────────────────────────────────────────────────
# Quick alert templates (E) — frontend hits this to populate a dropdown
# ─────────────────────────────────────────────────────────────────────────────
_ALERT_TEMPLATES = [
    {"id": "px_above",   "label": "Price crosses ABOVE level",
     "expr": "price > {level}", "params": [{"name": "level", "type": "number", "placeholder": "70000"}]},
    {"id": "px_below",   "label": "Price crosses BELOW level",
     "expr": "price < {level}", "params": [{"name": "level", "type": "number", "placeholder": "60000"}]},
    {"id": "rsi_over",   "label": "RSI(14) crosses ABOVE 70 (overbought)",
     "expr": "rsi(14) > 70", "params": []},
    {"id": "rsi_under",  "label": "RSI(14) crosses BELOW 30 (oversold)",
     "expr": "rsi(14) < 30", "params": []},
    {"id": "vwap_reclm", "label": "Price reclaims VWAP",
     "expr": "price > vwap", "params": []},
    {"id": "vwap_lost",  "label": "Price loses VWAP",
     "expr": "price < vwap", "params": []},
    {"id": "ema_cross_up",   "label": "Fast EMA(9) crosses ABOVE slow EMA(21)",
     "expr": "ema(9) > ema(21)", "params": []},
    {"id": "ema_cross_down", "label": "Fast EMA(9) crosses BELOW slow EMA(21)",
     "expr": "ema(9) < ema(21)", "params": []},
    {"id": "macd_bull",  "label": "MACD line crosses ABOVE signal (bullish)",
     "expr": "macd_line > macd_signal", "params": []},
    {"id": "macd_bear",  "label": "MACD line crosses BELOW signal (bearish)",
     "expr": "macd_line < macd_signal", "params": []},
    {"id": "supertrend_flip_up",   "label": "SuperTrend flips bullish",
     "expr": "supertrend_dir == 1", "params": []},
    {"id": "supertrend_flip_down", "label": "SuperTrend flips bearish",
     "expr": "supertrend_dir == -1", "params": []},
    # FVG (fair-value-gap) retests — server side resolves these by reading
    # the unfilled-FVG list and checking whether the latest candle wicked
    # back into the nearest active gap on the side selected.
    {"id": "fvg_bull_retest", "label": "Price retests an unfilled BULL FVG (long-bias support)",
     "expr": "fvg_retest_long", "params": []},
    {"id": "fvg_bear_retest", "label": "Price retests an unfilled BEAR FVG (short-bias resistance)",
     "expr": "fvg_retest_short", "params": []},
]

@app.get("/api/trade/alert_templates")
async def trade_alert_templates():
    """Return the list of one-click alert templates the UI offers."""
    return {"ok": True, "templates": _ALERT_TEMPLATES}


# ─────────────────────────────────────────────────────────────────────────────
# Indicator setup sync — save/load/share /trade chart layouts across devices
# ─────────────────────────────────────────────────────────────────────────────

# Allowed values mirror the JS PRESETS / VALID_SRC table in trade.html.
_INDICATOR_TYPES = {"ema", "sma", "bb", "vwap", "supertrend",
                    "rsi", "macd", "stochrsi", "atr", "fvg"}
_INDICATOR_SRC   = {"close", "open", "high", "low", "hl2", "hlc3", "ohlc4"}
_MAX_INDICATORS_PER_SETUP = 30
_MAX_SETUPS_PER_USER      = 20
_MAX_SETUP_NAME           = 60


def _sanitize_indicator_spec(spec) -> list:
    """Strip a client-supplied indicator list down to safe, persistable data.

    Rejects bad shapes outright but silently drops individual unknown items so
    a single malformed entry can't poison the whole save."""
    if not isinstance(spec, list):
        raise HTTPException(status_code=400, detail="spec must be a list")
    if len(spec) > _MAX_INDICATORS_PER_SETUP:
        raise HTTPException(
            status_code=400,
            detail=f"too many indicators (max {_MAX_INDICATORS_PER_SETUP})",
        )
    out = []
    for item in spec:
        if not isinstance(item, dict):
            continue
        t = (item.get("type") or "").lower().strip()
        if t not in _INDICATOR_TYPES:
            continue
        raw_params = item.get("params") or {}
        if not isinstance(raw_params, dict):
            raw_params = {}
        clean_params: dict = {}
        for k, v in list(raw_params.items())[:8]:
            if not isinstance(k, str) or not k or len(k) > 32:
                continue
            try:
                fv = float(v)
            except (TypeError, ValueError):
                continue
            # Cap to a sane range — UI clamps these too but defence in depth.
            if not (-1e6 < fv < 1e6):
                continue
            clean_params[k] = fv
        color = item.get("color") or "#42a5f5"
        if (not isinstance(color, str) or len(color) > 16
                or not color.startswith("#")):
            color = "#42a5f5"
        src = item.get("src") or "close"
        if src not in _INDICATOR_SRC:
            src = "close"
        visible = item.get("visible", True)
        out.append({
            "type":    t,
            "params":  clean_params,
            "color":   color,
            "src":     src,
            "visible": bool(visible),
        })
    return out


def _new_share_token() -> str:
    """Short, URL-safe, hard-to-guess token for share links."""
    return secrets.token_urlsafe(9)


def _setup_to_dict(s, include_spec: bool = False) -> dict:
    out = {
        "id":         s.id,
        "name":       s.name,
        "is_default": bool(s.is_default),
        "share_token": s.share_token,
        "updated_at": s.updated_at.isoformat() if s.updated_at else None,
    }
    if include_spec:
        try:
            out["spec"] = json.loads(s.json_spec or "[]")
        except Exception:
            out["spec"] = []
    return out


def _ensure_share_token(db: Session, setup) -> str:
    """Lazily mint a share token if one isn't set yet."""
    from app.models import TradeIndicatorSetup
    if setup.share_token:
        return setup.share_token
    # Try a few times in the (extremely unlikely) event of a collision.
    for _ in range(5):
        tok = _new_share_token()
        clash = (db.query(TradeIndicatorSetup)
                   .filter(TradeIndicatorSetup.share_token == tok)
                   .first())
        if not clash:
            setup.share_token = tok
            db.commit()
            return tok
    raise HTTPException(status_code=500, detail="could not allocate share token")


@app.get("/api/trade/indicators")
async def trade_indicators_list(request: Request, db: Session = Depends(get_db)):
    """Return all of the logged-in user's saved chart setups + the default spec.

    Logged-out users get {logged_in: false} so the front-end can fall back to
    the LocalStorage-only flow."""
    from app.models import TradeIndicatorSetup
    uid = _get_session_uid(request)
    if not uid:
        return {"logged_in": False, "setups": [], "default_spec": None,
                "max": _MAX_SETUPS_PER_USER}
    user = _get_user_by_uid(uid, db)
    if not user:
        return {"logged_in": False, "setups": [], "default_spec": None,
                "max": _MAX_SETUPS_PER_USER}

    rows = (db.query(TradeIndicatorSetup)
              .filter(TradeIndicatorSetup.user_id == user.id)
              .order_by(TradeIndicatorSetup.is_default.desc(),
                        TradeIndicatorSetup.updated_at.desc())
              .limit(_MAX_SETUPS_PER_USER + 5)
              .all())

    # Lazy-mint share tokens for any legacy rows missing one so every saved
    # setup is shareable from the UI without an extra round-trip.
    minted_any = False
    for r in rows:
        if not r.share_token:
            for _ in range(5):
                tok = _new_share_token()
                clash = (db.query(TradeIndicatorSetup)
                           .filter(TradeIndicatorSetup.share_token == tok)
                           .first())
                if not clash:
                    r.share_token = tok
                    minted_any = True
                    break
    if minted_any:
        db.commit()

    # Default spec = whichever row is currently the working copy. Prefer the
    # 'Auto-saved' buffer (always reflects the user's last on-screen state);
    # fall back to the row marked is_default for legacy data.
    default_spec = None
    autosave_row = next((r for r in rows if r.name == _AUTO_SAVE_NAME), None)
    fallback_row = autosave_row or next((r for r in rows if r.is_default), None)
    if fallback_row:
        try:
            default_spec = json.loads(fallback_row.json_spec or "[]")
        except Exception:
            default_spec = []

    return {
        "logged_in":    True,
        "setups":       [_setup_to_dict(r) for r in rows],
        "default_spec": default_spec,
        "max":          _MAX_SETUPS_PER_USER,
    }


_AUTO_SAVE_NAME = "Auto-saved"


def _get_or_create_autosave(db: Session, user_id: int):
    """Return the user's dedicated 'Auto-saved' working-copy row, creating it
    (with a share token) on first use. Always exists separately from named
    presets so /sync writes never clobber 'Scalp'/'Swing'/etc."""
    from app.models import TradeIndicatorSetup
    row = (db.query(TradeIndicatorSetup)
             .filter(TradeIndicatorSetup.user_id == user_id,
                     TradeIndicatorSetup.name == _AUTO_SAVE_NAME)
             .first())
    if row:
        if not row.share_token:
            _ensure_share_token(db, row)
        return row
    # Promote auto-save to default only if no named preset is already default.
    has_default = (db.query(TradeIndicatorSetup)
                     .filter(TradeIndicatorSetup.user_id == user_id,
                             TradeIndicatorSetup.is_default == True)  # noqa: E712
                     .first())
    row = TradeIndicatorSetup(
        user_id=user_id,
        name=_AUTO_SAVE_NAME,
        json_spec="[]",
        is_default=(not has_default),
        share_token=_new_share_token(),
    )
    db.add(row)
    db.flush()
    return row


@app.post("/api/trade/indicators/sync")
async def trade_indicators_sync(request: Request, db: Session = Depends(get_db)):
    """Auto-save the user's current working layout.

    Body: {spec: [...]}. Always writes to the dedicated 'Auto-saved' row
    (created on first use with a share token). Named presets like 'Scalp'
    are never touched here, so tweaking the chart can't overwrite a saved
    preset the user is treating as immutable."""
    uid = _get_session_uid(request)
    if not uid:
        raise HTTPException(status_code=401, detail="login required")
    user = _get_user_by_uid(uid, db)
    if not user:
        raise HTTPException(status_code=401, detail="login required")
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="invalid JSON body")

    spec = _sanitize_indicator_spec(body.get("spec") or [])

    setup = _get_or_create_autosave(db, user.id)
    setup.json_spec = json.dumps(spec)
    db.commit()
    return {"ok": True, "count": len(spec)}


@app.post("/api/trade/indicators/save")
async def trade_indicators_save(request: Request, db: Session = Depends(get_db)):
    """Save the current layout under a name (e.g. 'Scalp', 'Swing').

    Body: {name, spec, set_default?}. If a setup with the same name already
    exists for this user we overwrite it (so users can hit 'Save' again to
    update an existing preset). A share token is minted on creation."""
    from app.models import TradeIndicatorSetup
    uid = _get_session_uid(request)
    if not uid:
        raise HTTPException(status_code=401, detail="login required")
    user = _get_user_by_uid(uid, db)
    if not user:
        raise HTTPException(status_code=401, detail="login required")
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="invalid JSON body")

    name = (body.get("name") or "").strip()[:_MAX_SETUP_NAME]
    if not name:
        raise HTTPException(status_code=400, detail="name is required")
    spec = _sanitize_indicator_spec(body.get("spec") or [])
    set_default = bool(body.get("set_default", False))

    # Update-in-place if a setup with this exact name exists.
    existing = (db.query(TradeIndicatorSetup)
                  .filter(TradeIndicatorSetup.user_id == user.id,
                          TradeIndicatorSetup.name == name)
                  .first())
    if existing:
        existing.json_spec = json.dumps(spec)
        if set_default:
            (db.query(TradeIndicatorSetup)
                .filter(TradeIndicatorSetup.user_id == user.id,
                        TradeIndicatorSetup.id != existing.id,
                        TradeIndicatorSetup.is_default == True)  # noqa: E712
                .update({"is_default": False}))
            existing.is_default = True
        if not existing.share_token:
            _ensure_share_token(db, existing)
        # Mirror into Auto-saved buffer so cross-device GET always returns the
        # spec the user just saved, even before the next debounced /sync runs.
        if set_default and existing.name != _AUTO_SAVE_NAME:
            autosave = _get_or_create_autosave(db, user.id)
            autosave.json_spec = json.dumps(spec)
            autosave.is_default = False
        db.commit()
        db.refresh(existing)
        return {"ok": True, "setup": _setup_to_dict(existing, include_spec=True)}

    # Quota check — count only user-named presets, never the auto-save buffer.
    named_count = (db.query(TradeIndicatorSetup)
                     .filter(TradeIndicatorSetup.user_id == user.id,
                             TradeIndicatorSetup.name != _AUTO_SAVE_NAME)
                     .count())
    if named_count >= _MAX_SETUPS_PER_USER:
        raise HTTPException(
            status_code=400,
            detail=f"Max {_MAX_SETUPS_PER_USER} saved setups. Delete one first.",
        )

    setup = TradeIndicatorSetup(
        user_id=user.id,
        name=name,
        json_spec=json.dumps(spec),
        is_default=False,
    )
    db.add(setup)
    db.flush()
    _ensure_share_token(db, setup)
    if set_default:
        (db.query(TradeIndicatorSetup)
            .filter(TradeIndicatorSetup.user_id == user.id,
                    TradeIndicatorSetup.id != setup.id,
                    TradeIndicatorSetup.is_default == True)  # noqa: E712
            .update({"is_default": False}))
        setup.is_default = True
        # Mirror into Auto-saved so other devices see this layout immediately
        # on next page load (no wait for the debounced sync).
        autosave = _get_or_create_autosave(db, user.id)
        autosave.json_spec = json.dumps(spec)
        autosave.is_default = False
    db.commit()
    db.refresh(setup)
    return {"ok": True, "setup": _setup_to_dict(setup, include_spec=True)}


@app.post("/api/trade/indicators/{setup_id}/load")
async def trade_indicators_load(setup_id: int, request: Request,
                                db: Session = Depends(get_db)):
    """Mark a saved setup as the active default + return its spec.

    The chart polls /api/trade/indicators on next load and other devices will
    pick up the same setup."""
    from app.models import TradeIndicatorSetup
    uid = _get_session_uid(request)
    if not uid:
        raise HTTPException(status_code=401, detail="login required")
    user = _get_user_by_uid(uid, db)
    if not user:
        raise HTTPException(status_code=401, detail="login required")

    setup = (db.query(TradeIndicatorSetup)
               .filter(TradeIndicatorSetup.id == setup_id,
                       TradeIndicatorSetup.user_id == user.id)
               .first())
    if not setup:
        raise HTTPException(status_code=404, detail="setup not found")

    (db.query(TradeIndicatorSetup)
        .filter(TradeIndicatorSetup.user_id == user.id,
                TradeIndicatorSetup.id != setup.id,
                TradeIndicatorSetup.is_default == True)  # noqa: E712
        .update({"is_default": False}))
    setup.is_default = True

    # Mirror the loaded spec into the Auto-saved buffer (unless the user is
    # loading Auto-saved itself) so subsequent tweaks accumulate there
    # instead of accidentally writing back into the named preset on next
    # device open.
    if setup.name != _AUTO_SAVE_NAME:
        autosave = _get_or_create_autosave(db, user.id)
        autosave.json_spec = setup.json_spec or "[]"

    db.commit()
    db.refresh(setup)
    return {"ok": True, "setup": _setup_to_dict(setup, include_spec=True)}


@app.delete("/api/trade/indicators/{setup_id}")
async def trade_indicators_delete(setup_id: int, request: Request,
                                  db: Session = Depends(get_db)):
    """Delete a saved setup. If it was the default, the most recently updated
    remaining setup is promoted in its place (or none if there are none left)."""
    from app.models import TradeIndicatorSetup
    uid = _get_session_uid(request)
    if not uid:
        raise HTTPException(status_code=401, detail="login required")
    user = _get_user_by_uid(uid, db)
    if not user:
        raise HTTPException(status_code=401, detail="login required")

    setup = (db.query(TradeIndicatorSetup)
               .filter(TradeIndicatorSetup.id == setup_id,
                       TradeIndicatorSetup.user_id == user.id)
               .first())
    if not setup:
        raise HTTPException(status_code=404, detail="setup not found")

    was_default = bool(setup.is_default)
    db.delete(setup)
    db.flush()

    if was_default:
        successor = (db.query(TradeIndicatorSetup)
                       .filter(TradeIndicatorSetup.user_id == user.id)
                       .order_by(TradeIndicatorSetup.updated_at.desc())
                       .first())
        if successor:
            successor.is_default = True
    db.commit()
    return {"ok": True}


@app.get("/api/trade/indicators/shared/{token}")
async def trade_indicators_shared(token: str, db: Session = Depends(get_db)):
    """Public — anyone with the link can read a shared setup's indicator list.

    Returns just the name + spec (no owner identity, no setup id). The viewer
    can then hit /api/trade/indicators/save to keep their own copy if they want."""
    from app.models import TradeIndicatorSetup
    tok = (token or "").strip()
    if not tok or len(tok) > 64:
        raise HTTPException(status_code=404, detail="setup not found")
    setup = (db.query(TradeIndicatorSetup)
               .filter(TradeIndicatorSetup.share_token == tok)
               .first())
    if not setup:
        raise HTTPException(status_code=404, detail="setup not found")
    try:
        spec = json.loads(setup.json_spec or "[]")
    except Exception:
        spec = []
    return {"name": setup.name, "spec": spec}


# ─────────────────────────────────────────────────────────────────────────────
# Public data APIs (no auth — used by landing page JS)
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/api/public/stats")
async def public_stats():
    """Aggregate stats for the landing page hero section. Cached 2 min."""
    cached = _CACHE.get("public_stats")
    if cached and time.time() < cached[1]:
        return cached[0]

    def _load():
        db = SessionLocal()
        try:
            total_strategies = db.query(func.count(UserStrategy.id)).scalar() or 0
            win_rates = db.query(StrategyPerformance.win_rate).filter(
                StrategyPerformance.win_rate > 0
            ).all()
            avg_win_rate = round(
                sum(r.win_rate for r in win_rates) / len(win_rates), 1
            ) if win_rates else 0
            try:
                total_paid = db.query(func.sum(StrategyPurchase.amount_paid)).scalar() or 0
                creator_payout = float(total_paid) * 0.8
            except Exception:
                creator_payout = 0
            return {
                "total_strategies": total_strategies,
                "avg_win_rate":     avg_win_rate,
                "total_paid_out":   round(creator_payout, 2),
            }
        except Exception as e:
            logger.warning(f"public_stats error: {e}")
            return {"total_strategies": 0, "avg_win_rate": 0, "total_paid_out": 0}
        finally:
            db.close()

    payload = await asyncio.to_thread(_load)
    _CACHE["public_stats"] = (payload, time.time() + 120)
    return payload


@app.get("/api/public/marketplace")
async def public_marketplace(limit: int = Query(6, ge=1, le=20)):
    """Top marketplace listings — no auth required. Cached 2 min.
    Uses batch queries instead of N+1 per-listing lookups."""
    cache_key = f"pub_mkt_{limit}"
    cached = _CACHE.get(cache_key)
    if cached and time.time() < cached[1]:
        return cached[0]
    db = SessionLocal()
    try:
        listings = (
            db.query(StrategyMarketplace)
            .order_by(StrategyMarketplace.avg_rating.desc(),
                      StrategyMarketplace.clone_count.desc())
            .limit(limit)
            .all()
        )
        if not listings:
            return []

        # Batch-load strategies, performances, and authors in 3 queries total
        strat_ids  = [m.strategy_id for m in listings if m.strategy_id]
        author_ids = [m.author_id   for m in listings if m.author_id]

        strats_map = {
            s.id: s for s in db.query(UserStrategy).filter(UserStrategy.id.in_(strat_ids)).all()
        }
        perfs_map = {
            p.strategy_id: p
            for p in db.query(StrategyPerformance).filter(
                StrategyPerformance.strategy_id.in_(strat_ids)
            ).all()
        }
        # Collect user_ids from strategies so we can look up names
        strat_user_ids = list({s.user_id for s in strats_map.values()})
        users_map = {
            u.id: u for u in db.query(User).filter(
                User.id.in_(list(set(author_ids + strat_user_ids)))
            ).all()
        }

        result = []
        for m in listings:
            strat = strats_map.get(m.strategy_id)
            perf  = perfs_map.get(m.strategy_id)
            creator_user = users_map.get(strat.user_id) if strat else None
            creator_name = (
                (creator_user.first_name or creator_user.username or creator_user.uid)
                if creator_user else "Anonymous"
            )
            result.append({
                "id":            m.id,
                "title":         m.title,
                "summary":       m.summary,
                "category":      m.category,
                "pricing_model": m.pricing_model,
                "price_usdt":    float(m.price_usdt) if m.price_usdt else None,
                "avg_rating":    round(float(m.avg_rating or 0), 1),
                "clone_count":   m.clone_count or 0,
                "is_verified":   m.is_verified,
                "creator_name":  creator_name,
                "win_rate":      round(perf.win_rate, 1)      if perf else None,
                "total_pnl":     round(perf.total_pnl_pct, 2) if perf else None,
                "total_trades":  perf.total_trades             if perf else 0,
            })
        _CACHE[cache_key] = (result, time.time() + 120)
        return result
    except Exception as e:
        logger.warning(f"public_marketplace error: {e}")
        return []
    finally:
        db.close()


@app.get("/api/public/leaderboard")
async def public_leaderboard(
    limit: int = Query(5, ge=1, le=10),
    asset_class: str = Query("", description="crypto | forex | index | stock"),
    market: str = Query("", description="gold — forex strategies focused on XAU/gold"),
):
    """Top strategies by P&L — no auth required. Cached 2 min."""
    ac = (asset_class or "").strip().lower()
    mkt = (market or "").strip().lower()
    cache_key = f"pub_lb_{limit}_{ac}_{mkt}"
    cached = _CACHE.get(cache_key)
    if cached and time.time() < cached[1]:
        return _cached_json(cached[0], True, 120)

    def _load():
        db = SessionLocal()
        try:
            from sqlalchemy import or_, func
            q = (
                db.query(StrategyPerformance, UserStrategy)
                .join(UserStrategy, UserStrategy.id == StrategyPerformance.strategy_id)
                .filter(StrategyPerformance.total_trades >= 5)
            )
            if mkt == "gold":
                q = q.filter(
                    UserStrategy.asset_class == "forex",
                    or_(
                        func.lower(UserStrategy.name).like("%gold%"),
                        func.lower(UserStrategy.name).like("%xau%"),
                    ),
                )
            elif ac in ("crypto", "forex", "index", "stock"):
                q = q.filter(UserStrategy.asset_class == ac)
            rows = q.order_by(StrategyPerformance.total_pnl_pct.desc()).limit(limit).all()
            return [
                {
                    "name":         r.UserStrategy.name or "Unnamed",
                    "asset_class":  getattr(r.UserStrategy, "asset_class", None) or "crypto",
                    "total_trades": r.StrategyPerformance.total_trades,
                    "win_rate":     round(r.StrategyPerformance.win_rate, 1),
                    "total_pnl":    round(r.StrategyPerformance.total_pnl_pct, 2),
                }
                for r in rows
            ]
        except Exception as e:
            logger.warning(f"public_leaderboard error: {e}")
            return []
        finally:
            db.close()

    payload = await asyncio.to_thread(_load)
    _CACHE[cache_key] = (payload, time.time() + 120)
    return _cached_json(payload, False, 120)


# ─────────────────────────────────────────────────────────────────────────────
# Main portal page (legacy URL — keeps existing Telegram bot links working)
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/strategies", response_class=HTMLResponse)
async def portal_page(request: Request, uid: str = Query(...)):
    """Legacy URL — keeps existing Telegram bot links working."""
    return await _render_portal(request, uid)


# ─────────────────────────────────────────────────────────────────────────────
# API endpoints (used by portal JS)
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/api/ping")
async def api_ping(lite: int = Query(0)):
    """Lightweight liveness + DB wake-up before heavier portal API calls.

    ?lite=1 returns instantly (no DB) — use /ping for Railway cold-start detection.
    """
    if lite:
        return {"ok": True, "lite": True}
    t0 = time.monotonic()
    try:
        from sqlalchemy import text as _pt
        from app.database import SessionLocal as _PSL

        def _db_ping():
            db = _PSL()
            try:
                db.execute(_pt("SELECT 1"))
                db.commit()
            finally:
                db.close()

        await asyncio.wait_for(asyncio.to_thread(_db_ping), timeout=28.0)
        return {"ok": True, "db_ms": int((time.monotonic() - t0) * 1000)}
    except Exception as exc:
        logger.warning(f"[api/ping] db wake failed: {exc}")
        return JSONResponse(
            {"ok": False, "error": "database_waking", "detail": str(exc)[:120]},
            status_code=503,
        )


_HEAL_USER_AT: dict = {}


@app.get("/api/portal/bootstrap")
async def api_portal_bootstrap(uid: str = Query(...)):
    """Strategies + portfolio in one round-trip (one Neon wake, shared cold-start)."""
    import json as _json

    strat_key = f"api_strats_{uid}"
    port_key = f"portfolio_{uid}"
    now = time.time()
    heal_info = None
    if uid and now - _HEAL_USER_AT.get(uid, 0) >= 300:
        _HEAL_USER_AT[uid] = now

        def _heal_sync():
            from app.database import SessionLocal
            from app.services.strategy_heal import heal_user_account
            _hdb = SessionLocal()
            try:
                _huser = _get_user_by_uid(uid, _hdb)
                if _huser:
                    return heal_user_account(_hdb, _huser.id)
            finally:
                _hdb.close()
            return None

        try:
            heal_info = await asyncio.wait_for(asyncio.to_thread(_heal_sync), timeout=4.0)
            if heal_info and (heal_info.get("rows_touched") or heal_info.get("stale_expired")):
                _CACHE.pop(strat_key, None)
                _CACHE.pop(port_key, None)
        except Exception as _he:
            logger.warning("[bootstrap] strategy heal uid=%s: %s", uid, _he)

    sc = _CACHE.get(strat_key)
    pc = _CACHE.get(port_key)
    if sc and pc and now < sc[1] and now < pc[1]:
        out = {"strategies": sc[0], "portfolio": pc[0], "cached": True}
        if heal_info:
            out["heal"] = heal_info
        return JSONResponse(out)

    try:
        strats_r, port_r = await asyncio.gather(
            api_strategies(uid),
            api_portfolio(uid),
            return_exceptions=True,
        )
    except Exception as exc:
        logger.warning(f"[bootstrap] gather failed uid={uid}: {exc}")
        raise HTTPException(status_code=503, detail="Portal busy — retry in a moment")

    for item in (strats_r, port_r):
        if isinstance(item, HTTPException):
            raise item

    def _body(resp):
        if isinstance(resp, JSONResponse):
            return _json.loads(resp.body)
        if isinstance(resp, dict):
            return resp
        return _json.loads(getattr(resp, "body", b"{}") or b"{}")

    try:
        strats = _body(strats_r)
        port = _body(port_r)
    except Exception as exc:
        logger.warning(f"[bootstrap] parse failed uid={uid}: {exc}")
        if sc and pc:
            return JSONResponse({"strategies": sc[0], "portfolio": pc[0], "cached": True, "stale": True})
        raise HTTPException(status_code=503, detail="Portal busy — retry in a moment")

    if isinstance(strats, dict) and strats.get("detail"):
        raise HTTPException(status_code=503, detail=str(strats.get("detail")))
    if isinstance(port, dict) and port.get("error"):
        raise HTTPException(status_code=503, detail=str(port.get("detail") or "Portfolio busy"))

    payload = {"strategies": strats, "portfolio": port, "cached": False}
    if heal_info:
        payload["heal"] = heal_info
    return JSONResponse(payload)


@app.post("/api/strategies/raise-daily-cap")
async def api_strategies_raise_daily_cap(request: Request):
    """
    Raise max_trades_per_day on every strategy for this user (default → 10).
    Only increases values below the target — never lowers a higher cap.
    Takes effect immediately; strategies that already hit today's old cap can
    fire again if the new cap is higher than today's fire count.
    """
    body = await request.json()
    uid = (body.get("uid") or "").strip()
    if not uid:
        raise HTTPException(status_code=400, detail="uid required")
    try:
        min_cap = int(body.get("max_trades_per_day", 10))
    except (TypeError, ValueError):
        raise HTTPException(status_code=400, detail="max_trades_per_day must be an integer")
    if min_cap < 1 or min_cap > 100:
        raise HTTPException(status_code=400, detail="max_trades_per_day must be between 1 and 100")

    from app.database import SessionLocal
    from app.services.strategy_heal import raise_daily_trade_caps

    db = SessionLocal()
    try:
        user = _get_user_by_uid(uid, db)
        if not user or user.banned:
            raise HTTPException(status_code=403)
        stats = raise_daily_trade_caps(db, user.id, min_cap=min_cap)
        _CACHE.pop(f"api_strats_{uid}", None)
        _CACHE.pop(f"portfolio_{uid}", None)
        return JSONResponse({"ok": True, **stats})
    finally:
        db.close()


@app.post("/api/strategies/ensure-firing")
async def api_strategies_ensure_firing(request: Request):
    """
    One-tap repair: promote draft/paused → paper, fix asset_class/universe,
    expire stale OPEN ghosts blocking max_open. Safe to call repeatedly.
    """
    body = await request.json()
    uid = (body.get("uid") or "").strip()
    if not uid:
        raise HTTPException(status_code=400, detail="uid required")
    from app.database import SessionLocal
    from app.services.strategy_heal import (
        heal_user_account,
        expire_untracked_forex_opens_when_broker_empty,
    )
    db = SessionLocal()
    try:
        user = _get_user_by_uid(uid, db)
        if not user or user.banned:
            raise HTTPException(status_code=403)
        stats = heal_user_account(db, user.id)
        stats["orphan_forex_expired"] = await expire_untracked_forex_opens_when_broker_empty(
            min_age_minutes=30,
            user_id=user.id,
        )
        _CACHE.pop(f"api_strats_{uid}", None)
        _CACHE.pop(f"portfolio_{uid}", None)
        return JSONResponse({"ok": True, **stats})
    finally:
        db.close()


@app.get("/api/strategies")
async def api_strategies(uid: str = Query(...)):
    cache_key = f"api_strats_{uid}"
    cached = _CACHE.get(cache_key)
    if cached and time.time() < cached[1]:
        return _cached_json(cached[0], True, 60)

    from app.database import SessionLocal
    from app.strategy_models import UserStrategy, StrategyPerformance, StrategyExecution

    def _load():
        db = SessionLocal()
        try:
            from sqlalchemy import text as _t2
            db.execute(_t2("SET LOCAL statement_timeout = '8000'"))
            user = _get_user_by_uid(uid, db)
            if not user:
                return None

            strategies = (
                db.query(UserStrategy)
                .filter(UserStrategy.user_id == user.id)
                .order_by(UserStrategy.updated_at.desc())
                .all()
            )

            strategy_ids = [s.id for s in strategies]
            perf_map: dict = {}
            exec_map: dict = {}
            if strategy_ids:
                perfs = db.query(StrategyPerformance).filter(
                    StrategyPerformance.strategy_id.in_(strategy_ids)
                ).all()
                perf_map = {p.strategy_id: p for p in perfs}

                # LATERAL join — one index-range-scan per strategy_id using
                # the (strategy_id, fired_at DESC) composite index, avoids the
                # slow global IN + ORDER BY that times out under DB load.
                ids_str = ",".join(str(i) for i in strategy_ids)
                try:
                    raw_rows = db.execute(_t2(f"""
                        SELECT e.strategy_id, e.symbol, e.direction, e.outcome,
                               e.pnl_pct, e.entry_price, e.exit_price, e.fired_at
                        FROM unnest(ARRAY[{ids_str}]::int[]) AS s(sid)
                        CROSS JOIN LATERAL (
                            SELECT strategy_id, symbol, direction, outcome,
                                   pnl_pct, entry_price, exit_price, fired_at
                            FROM strategy_executions
                            WHERE strategy_id = s.sid
                            ORDER BY fired_at DESC
                            LIMIT 5
                        ) e
                    """)).fetchall()
                    for row in raw_rows:
                        exec_map.setdefault(row.strategy_id, []).append(row)
                except Exception:
                    pass

            result = []
            for s in strategies:
                perf = perf_map.get(s.id)
                recent_execs = exec_map.get(s.id, [])

                # Fast inline health score
                wr  = (perf.win_rate or 0) if perf else 0
                tot = (perf.total_trades or 0) if perf else 0
                _wins   = (perf.wins or 0) if perf else 0
                _losses = (perf.losses or 0) if perf else 0
                _avg_w  = (perf.avg_win_pct or 0) if perf else 0
                _avg_l  = (perf.avg_loss_pct or 0) if perf else 0
                pf = (_avg_w * max(_wins, 1)) / (abs(_avg_l) * max(_losses, 1)) if (_losses > 0 and _avg_l) else 0
                health = 0.0
                if tot >= 3:
                    health += min(wr / 100, 1.0) * 4.0
                    health += min(pf / 2.0, 1.0) * 3.0
                    health += min(tot / 30.0, 1.0) * 2.0
                    health += 1.0
                health_score = round(min(health, 10.0), 1)

                cfg = s.config or {}
                result.append({
                    "id":           s.id,
                    "name":         s.name,
                    "description":  s.description,
                    "status":       s.status,
                    "config":       _slim_strategy_config(cfg),
                    "is_locked":    bool(cfg.get("_locked")),
                    "is_public":    s.is_public,
                    "created_at":   s.created_at.isoformat() if s.created_at else None,
                    "health_score": health_score,
                    "performance": {
                        "total_trades":      int(perf.total_trades or 0),
                        "wins":              int(perf.wins or 0),
                        "losses":            int(perf.losses or 0),
                        "win_rate":          round(perf.win_rate or 0, 1),
                        "total_pnl":         round(perf.total_pnl_pct or 0, 2),
                        "total_pips_pnl":    round(perf.total_pips_pnl, 1) if perf.total_pips_pnl is not None else None,
                        "avg_pips_per_trade": round(perf.avg_pips_per_trade, 1) if perf.avg_pips_per_trade is not None else None,
                        "open_trades":       int(perf.open_trades or 0),
                        "best_trade":        round(perf.best_trade or 0, 2),
                        "worst_trade":       round(perf.worst_trade or 0, 2),
                        "avg_win_pct":       round(perf.avg_win_pct or 0, 2),
                        "avg_loss_pct":      round(perf.avg_loss_pct or 0, 2),
                    } if perf else {},
                    "recent_trades": [{
                        "symbol":      ex.symbol,
                        "direction":   ex.direction,
                        "outcome":     ex.outcome,
                        "pnl_pct":     round(ex.pnl_pct, 4) if ex.pnl_pct is not None else None,
                        "entry_price": ex.entry_price,
                        "close_price": ex.exit_price,
                        "fired_at":    ex.fired_at.isoformat() if ex.fired_at else None,
                    } for ex in recent_execs],
                })
            return result
        finally:
            db.close()

    try:
        data = await asyncio.wait_for(asyncio.to_thread(_load), timeout=25.0)
    except (asyncio.TimeoutError, Exception) as _e:
        logger.warning(f"api_strategies timeout/error for {uid}: {_e}")
        stale = _CACHE.get(cache_key)
        if stale:
            return _cached_json(stale[0], True, 60)
        raise HTTPException(status_code=503, detail="Database busy — please retry in a moment")
    if data is None:
        raise HTTPException(status_code=403)
    _CACHE[cache_key] = (data, time.time() + 60)
    return _cached_json(data, False, 60)


@app.post("/api/strategies/{strategy_id}/toggle")
async def api_toggle_strategy(strategy_id: int, uid: str = Query(...)):
    from app.database import SessionLocal
    db = SessionLocal()
    try:
        user = _get_user_by_uid(uid, db)
        if not user:
            raise HTTPException(status_code=403)

        from app.strategy_models import UserStrategy
        strategy = db.query(UserStrategy).filter(
            UserStrategy.id == strategy_id,
            UserStrategy.user_id == user.id,
        ).first()
        if not strategy:
            raise HTTPException(status_code=404)

        # Paper strategies stay paper — promotion to live is done via PUT /status=active
        if strategy.status == "paper":
            pass  # already running in paper mode
        elif strategy.status == "active":
            strategy.status = "paused"
        else:
            strategy.status = "active"
        db.commit()
        invalidate_prefix(f"api_strats_{uid}")
        invalidate_prefix(f"api_mkt:{uid}")
        invalidate_cache(f"portfolio_{uid}")
        invalidate_prefix(f"portfolio_trades:{uid}:")
        invalidate_cache(f"live_forex_acct:{(uid or '').strip().upper()}")
        return {"status": strategy.status}
    finally:
        db.close()


@app.post("/api/strategies/{strategy_id}/sizing")
async def api_update_strategy_sizing(strategy_id: int, request: Request):
    """Lightweight inline position-size update from the Live Forex page.
    Body: {uid, size_type: 'lots'|'pct'|'fixed'|'risk_pct', value}.
    Allowed even for locked (subscribed) strategies — sizing is a user choice,
    not creator IP. Writes a fresh config dict (SQLAlchemy JSON mutation safety)."""
    body = await request.json()
    uid = body.get("uid")
    if not uid:
        raise HTTPException(status_code=400)
    size_type = str(body.get("size_type") or "").lower()
    if size_type not in ("lots", "pct", "fixed", "risk_pct"):
        raise HTTPException(status_code=400, detail="Invalid size_type")
    try:
        value = float(body.get("value"))
    except (TypeError, ValueError):
        raise HTTPException(status_code=400, detail="Invalid value")

    from app.database import SessionLocal
    db = SessionLocal()
    try:
        user = _get_user_by_uid(uid, db)
        if not user:
            raise HTTPException(status_code=403)
        from app.strategy_models import UserStrategy
        s = db.query(UserStrategy).filter(
            UserStrategy.id == strategy_id, UserStrategy.user_id == user.id
        ).first()
        if not s:
            raise HTTPException(status_code=404)

        config = dict(s.config or {})
        risk = dict(config.get("risk", {}))
        risk["position_size_type"] = size_type if size_type != "risk_pct" else "pct"
        # Clamp per type, write the canonical value field, and clear stale ones.
        if size_type == "lots":
            risk["position_size_lots"] = round(max(0.01, min(50.0, value)), 2)
            risk["use_risk_pct"] = False
        elif size_type == "fixed":
            risk["position_size_usd"] = round(max(1.0, min(1_000_000.0, value)), 2)
            risk["use_risk_pct"] = False
        elif size_type == "risk_pct":
            _rp = round(max(0.1, min(100.0, value)), 2)
            risk["risk_pct_per_trade"] = _rp
            # Mirror into position_size_pct: the subscriber-propagation execution
            # path reads position_size_pct (not risk_pct_per_trade) as its risk%,
            # so keep both in sync or auto-lot sizing uses a stale value.
            risk["position_size_pct"] = _rp
            risk["use_risk_pct"] = True
        else:  # pct of balance
            risk["position_size_pct"] = round(max(0.1, min(100.0, value)), 2)
            risk["use_risk_pct"] = False
        config["risk"] = risk
        s.config = config
        db.commit()
        invalidate_prefix(f"api_strats_{uid}")
        invalidate_prefix(f"api_mkt:{uid}")
        invalidate_cache(f"analytics:{uid}:{s.id}")

        pst = risk["position_size_type"]
        if size_type == "lots":
            size_label = f"{risk['position_size_lots']:g} lots"
        elif size_type == "fixed":
            size_label = f"${risk['position_size_usd']:g} fixed"
        elif size_type == "risk_pct":
            size_label = f"{risk['risk_pct_per_trade']:g}% risk"
        else:
            size_label = f"{risk['position_size_pct']:g}% balance"
        return JSONResponse({"success": True, "id": s.id,
                             "size_type": size_type, "size_label": size_label})
    finally:
        db.close()


def _delete_strategy_marketplace_fks(db, strategy_id: int) -> None:
    """Marketplace / purchase FK cleanup before removing a user_strategies row."""
    from sqlalchemy import text

    try:
        db.execute(text("""
            DELETE FROM earnings_transactions
            WHERE purchase_id IN (
                SELECT id FROM strategy_purchases WHERE strategy_id = :sid
            )
        """), {"sid": strategy_id})
        db.execute(text(
            "DELETE FROM strategy_purchases WHERE strategy_id = :sid"
        ), {"sid": strategy_id})
        db.execute(text("""
            UPDATE strategy_purchases
            SET cloned_strategy_id = NULL
            WHERE cloned_strategy_id = :sid
        """), {"sid": strategy_id})
        db.execute(text("""
            DELETE FROM strategy_ratings
            WHERE listing_id IN (
                SELECT id FROM strategy_marketplace WHERE strategy_id = :sid
            )
        """), {"sid": strategy_id})
        db.execute(text(
            "DELETE FROM strategy_marketplace WHERE strategy_id = :sid"
        ), {"sid": strategy_id})
        db.execute(text(
            "DELETE FROM strategy_offers WHERE strategy_id = :sid"
        ), {"sid": strategy_id})
    except Exception as _clean_err:
        logger.warning(f"[DeleteStrategy] Cleanup warning (non-fatal): {_clean_err}")


def _delete_owned_strategy(db, strategy) -> None:
    """Remove FK children then delete a UserStrategy row (caller commits)."""
    _delete_strategy_marketplace_fks(db, strategy.id)
    db.delete(strategy)


def _bulk_delete_owned_strategies(db, user_id: int, strategy_ids: list) -> tuple:
    """
    Fast bulk delete — one SQL sweep for executions/performance, then marketplace
    cleanup per row. Avoids ORM cascade N×M round-trips that time out on Railway.
    Returns (deleted_ids, not_found_ids).
    """
    from sqlalchemy import text
    from app.strategy_models import UserStrategy

    if not strategy_ids:
        return [], []

    rows = (
        db.query(UserStrategy)
        .filter(
            UserStrategy.user_id == user_id,
            UserStrategy.id.in_(strategy_ids),
        )
        .all()
    )
    found_ids = sorted({int(s.id) for s in rows})
    not_found = [i for i in strategy_ids if i not in found_ids]
    if not found_ids:
        return [], not_found

    try:
        db.execute(
            text("DELETE FROM strategy_executions WHERE strategy_id = ANY(:ids)"),
            {"ids": found_ids},
        )
        db.execute(
            text("DELETE FROM strategy_performance WHERE strategy_id = ANY(:ids)"),
            {"ids": found_ids},
        )
        db.execute(
            text("DELETE FROM competition_entries WHERE strategy_id = ANY(:ids)"),
            {"ids": found_ids},
        )
        db.execute(
            text(
                "UPDATE feed_activities SET strategy_id = NULL "
                "WHERE strategy_id = ANY(:ids)"
            ),
            {"ids": found_ids},
        )
    except Exception as _child_err:
        logger.warning(f"[BulkDelete] child-row cleanup: {_child_err}")

    for sid in found_ids:
        _delete_strategy_marketplace_fks(db, sid)

    db.execute(
        text(
            "DELETE FROM user_strategies "
            "WHERE user_id = :uid AND id = ANY(:ids)"
        ),
        {"uid": user_id, "ids": found_ids},
    )
    return found_ids, not_found


@app.delete("/api/strategies/{strategy_id}")
async def api_delete_strategy(strategy_id: int, uid: str = Query(...)):
    from app.database import SessionLocal
    db = SessionLocal()
    try:
        user = _get_user_by_uid(uid, db)
        if not user:
            raise HTTPException(status_code=403)

        from app.strategy_models import UserStrategy
        strategy = db.query(UserStrategy).filter(
            UserStrategy.id == strategy_id,
            UserStrategy.user_id == user.id,
        ).first()
        if not strategy:
            raise HTTPException(status_code=404)

        _delete_owned_strategy(db, strategy)
        db.commit()
        return {"deleted": True}
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()


@app.post("/api/strategies/bulk-delete")
async def api_bulk_delete_strategies(
    request: Request,
    uid: str = Query(""),
):
    """Delete multiple owned strategies in one request (portal bulk-select UI)."""
    body = await request.json()
    uid = (uid or body.get("uid") or "").strip()
    raw_ids = body.get("strategy_ids") or body.get("ids") or []
    if not uid:
        raise HTTPException(status_code=400, detail="uid required")
    try:
        strategy_ids = sorted({int(i) for i in raw_ids})
    except (TypeError, ValueError):
        raise HTTPException(status_code=400, detail="strategy_ids must be integers")
    if not strategy_ids:
        raise HTTPException(status_code=400, detail="strategy_ids required")
    if len(strategy_ids) > 200:
        raise HTTPException(status_code=400, detail="Maximum 200 strategies per request")

    from app.database import SessionLocal

    def _run_bulk():
        db = SessionLocal()
        try:
            user = _get_user_by_uid(uid, db)
            if not user:
                raise HTTPException(status_code=403)
            deleted, not_found = _bulk_delete_owned_strategies(db, user.id, strategy_ids)
            db.commit()
            return deleted, not_found
        except HTTPException:
            raise
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()

    try:
        deleted, not_found = await asyncio.to_thread(_run_bulk)
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("[BulkDelete] failed for uid=%s count=%s", uid, len(strategy_ids))
        raise HTTPException(status_code=500, detail=str(e)[:300])

    _CACHE.pop(f"api_strats_{uid}", None)
    return {
        "deleted": deleted,
        "deleted_count": len(deleted),
        "not_found": not_found,
    }


@app.get("/api/marketplace")
async def api_marketplace(
    uid:      str = Query(...),
    sort:     str = Query("top"),
    category: str = Query("all"),
    pricing:  str = Query("all"),
    search:   str = Query(""),
    ai_only:  int = Query(0),     # 1 → only AI-generated listings
):
    from app.database import SessionLocal, engine
    db = SessionLocal()
    try:
        user = _get_user_by_uid(uid, db)
        if not user:
            raise HTTPException(status_code=403)
        _mkt_key = f"api_mkt:{uid}:{sort}:{category}:{pricing}:{search}:{ai_only}"
        _mkt_cached = get_cache(_mkt_key)
        if _mkt_cached is not None:
            return JSONResponse(_mkt_cached)

        from app.strategy_models import StrategyMarketplace, UserStrategy, StrategyPerformance
        from app.strategy_marketplace_ext import StrategyPurchase, init_marketplace_ext_tables

        q = db.query(StrategyMarketplace)
        # "ai" is a sentinel category that filters by is_ai_generated rather
        # than by the category column (which holds scalp/swing/etc).
        if category == "ai" or ai_only:
            q = q.filter(StrategyMarketplace.is_ai_generated == True)
        elif category != "all":
            q = q.filter(StrategyMarketplace.category == category)
        if pricing == "free":
            q = q.filter(StrategyMarketplace.pricing_model == "free")
        elif pricing == "paid":
            q = q.filter(StrategyMarketplace.pricing_model != "free")
        if sort == "new":
            q = q.order_by(StrategyMarketplace.published_at.desc())
        elif sort == "trending":
            q = q.order_by(StrategyMarketplace.clone_count.desc())
        elif sort == "verified":
            q = q.filter(StrategyMarketplace.is_verified == True).order_by(StrategyMarketplace.verified_win_rate.desc())
        elif sort == "sharpe":
            # Sort by backtest Sharpe — only listings that have a Sharpe score
            q = q.filter(StrategyMarketplace.backtest_sharpe.isnot(None))\
                 .order_by(StrategyMarketplace.backtest_sharpe.desc())
        elif sort == "winrate":
            # Best win rate — prefer verified over backtest, but show both
            from sqlalchemy import func
            q = q.order_by(
                func.coalesce(StrategyMarketplace.verified_win_rate,
                              StrategyMarketplace.backtest_win_rate, 0).desc()
            )
        elif sort == "followers":
            q = q.order_by(StrategyMarketplace.subscriber_count.desc(),
                           StrategyMarketplace.clone_count.desc())
        else:
            q = q.order_by(StrategyMarketplace.avg_rating.desc(), StrategyMarketplace.clone_count.desc())

        listings = q.limit(50).all()
        if search:
            s = search.lower()
            listings = [m for m in listings if s in (m.title or "").lower() or s in (m.summary or "").lower()]

        my_purchases = {
            p.listing_id for p in db.query(StrategyPurchase)
            .filter(StrategyPurchase.buyer_id == user.id, StrategyPurchase.status == "active").all()
        }

        from app.strategy_models import StrategyExecution
        # Batched asset_class lookup (avoid N+1)
        _strat_ids = [m.strategy_id for m in listings]
        _asset_map: dict = {}
        if _strat_ids:
            for sid, ac in db.query(UserStrategy.id, UserStrategy.asset_class).filter(UserStrategy.id.in_(_strat_ids)).all():
                _asset_map[sid] = ac or "crypto"

        # Batch-load performances and authors (avoid N+1 per listing)
        _perf_map2 = {
            p.strategy_id: p
            for p in db.query(StrategyPerformance).filter(
                StrategyPerformance.strategy_id.in_(_strat_ids)
            ).all()
        } if _strat_ids else {}
        _mkt_author_ids = list({m.author_id for m in listings if m.author_id})
        _mkt_author_map = {
            u.id: u for u in db.query(User).filter(User.id.in_(_mkt_author_ids)).all()
        } if _mkt_author_ids else {}

        # Bulk-load equity curves in ONE query (was N+1 — one per listing)
        _equity_raw: dict[int, list] = {}
        if _strat_ids:
            from sqlalchemy import text as _eqt
            _ids_str = ",".join(str(i) for i in _strat_ids)
            _eq_rows = db.execute(_eqt(f"""
                SELECT strategy_id, pnl_pct FROM (
                    SELECT strategy_id, pnl_pct, fired_at,
                           ROW_NUMBER() OVER (PARTITION BY strategy_id ORDER BY fired_at ASC) AS rn
                    FROM strategy_executions
                    WHERE strategy_id = ANY(ARRAY[{_ids_str}]::int[])
                      AND outcome IN ('WIN','LOSS','BREAKEVEN')
                      AND pnl_pct IS NOT NULL
                ) t WHERE rn <= 30
                ORDER BY strategy_id, fired_at ASC
            """)).fetchall()
            for row in _eq_rows:
                _equity_raw.setdefault(row.strategy_id, []).append(float(row.pnl_pct))

        result = []
        for m in listings:
            perf   = _perf_map2.get(m.strategy_id)
            author = _mkt_author_map.get(m.author_id)
            # Build cumulative equity curve from pre-loaded data
            equity = []
            cum = 0.0
            for pnl in _equity_raw.get(m.strategy_id, []):
                cum += pnl
                equity.append(round(cum, 2))
            result.append({
                "id":               m.id,
                "strategy_id":      m.strategy_id,
                "title":            m.title,
                "summary":          m.summary,
                "tags":             m.tags or [],
                "category":         m.category or "general",
                "pricing_model":    m.pricing_model or "free",
                "price_usdt":       m.price_usdt or 0,
                "clone_count":      m.clone_count or 0,
                "subscriber_count": m.subscriber_count or 0,
                "avg_rating":       round(m.avg_rating or 0, 1),
                "rating_count":     m.rating_count or 0,
                "is_featured":      m.is_featured,
                "is_trending":      getattr(m, "is_trending", False),
                "is_verified":      m.is_verified,
                "is_ai_generated":  bool(getattr(m, "is_ai_generated", False)),
                "asset_class":      _asset_map.get(m.strategy_id, "crypto"),
                "backtest_sharpe":  round(float(getattr(m, "backtest_sharpe", 0) or 0), 2),
                "backtest_pnl":     round(float(getattr(m, "backtest_pnl_pct", 0) or 0), 2),
                "backtest_trades":  int(getattr(m, "backtest_trades", 0) or 0),
                "backtest_win_rate": round(float(getattr(m, "backtest_win_rate", 0) or 0), 1),
                "verified_win_rate": round(m.verified_win_rate, 1) if m.is_verified and m.verified_win_rate else None,
                "verified_pnl":     round(m.verified_pnl, 2) if m.is_verified and m.verified_pnl else None,
                "live_win_rate":    round(perf.win_rate, 1) if perf and perf.total_trades >= 3 else None,
                "live_pnl":         round(perf.total_pnl_pct, 2) if perf and perf.total_trades >= 3 else None,
                "live_trades":      perf.total_trades if perf else 0,
                "equity_curve":     equity,
                "author_name":      (author.first_name or author.username or "Anonymous") if author else "Anonymous",
                "author_uid":       author.uid if author else None,
                "is_owned":         m.id in my_purchases or (m.pricing_model or "free") == "free",
            })
        set_cache(_mkt_key, result, 180)
        return JSONResponse(result)
    finally:
        db.close()


@app.get("/api/marketplace/leaderboard")
async def api_marketplace_leaderboard(
    uid: str = Query(...),
    period: str = Query("30d"),       # 7d | 30d | all
    limit: int  = Query(10, ge=1, le=50),
):
    """Top public marketplace strategies ranked by trailing summed pnl_pct.

    Single SQL aggregation over strategy_executions JOIN strategy_marketplace —
    no N+1 over listings. Powers the mobile leaderboard rail at the top of the
    Market tab and is cheap enough to call on every refresh."""
    from app.database import SessionLocal
    from sqlalchemy import text
    from datetime import datetime, timedelta
    _lb_ck = f"api_mkt_lb:{uid}:{period}:{limit}"
    _lb_hit = get_cache(_lb_ck)
    if _lb_hit is not None:
        return _cached_json(_lb_hit, True, 120)

    def _load():
        db = SessionLocal()
        try:
            user = _get_user_by_uid(uid, db)
            if not user:
                return None

            _period = period
            if _period == "7d":
                cutoff = datetime.utcnow() - timedelta(days=7)
            elif _period == "all":
                cutoff = None
            else:
                _period = "30d"
                cutoff = datetime.utcnow() - timedelta(days=30)

            cutoff_clause = "AND COALESCE(e.closed_at, e.fired_at) > :cutoff" if cutoff else ""
            params: dict = {"limit": int(limit)}
            if cutoff is not None:
                params["cutoff"] = cutoff

            rows = db.execute(text(f"""
                SELECT m.id           AS listing_id,
                       m.title        AS title,
                       m.author_id    AS author_id,
                       m.pricing_model AS pricing_model,
                       m.price_usdt   AS price_usdt,
                       m.is_verified  AS is_verified,
                       m.is_ai_generated AS is_ai_generated,
                       COALESCE(SUM(e.pnl_pct), 0) AS pnl_sum,
                       COUNT(e.id)              AS trades,
                       SUM(CASE WHEN e.outcome = 'WIN' THEN 1 ELSE 0 END) AS wins,
                       SUM(CASE WHEN e.outcome IN ('WIN','LOSS') THEN 1 ELSE 0 END) AS decisive
                FROM strategy_marketplace m
                JOIN strategy_executions e ON e.strategy_id = m.strategy_id
                WHERE e.outcome IN ('WIN', 'LOSS', 'BREAKEVEN')
                  AND e.pnl_pct IS NOT NULL
                  {cutoff_clause}
                GROUP BY m.id, m.title, m.author_id, m.pricing_model, m.price_usdt,
                         m.is_verified, m.is_ai_generated
                HAVING COUNT(e.id) >= 1
                ORDER BY pnl_sum DESC
                LIMIT :limit
            """), params).fetchall()

            # Resolve author display names in one shot
            author_ids = list({r.author_id for r in rows})
            names: dict = {}
            if author_ids:
                for u in db.query(User).filter(User.id.in_(author_ids)).all():
                    names[u.id] = u.first_name or u.username or "Anonymous"

            entries = []
            for rank, r in enumerate(rows, 1):
                trades   = int(r.trades or 0)
                wins     = int(r.wins or 0)
                decisive = int(getattr(r, "decisive", 0) or 0)
                wr       = round((wins / decisive) * 100, 1) if decisive > 0 else 0.0
                entries.append({
                    "rank":          rank,
                    "listing_id":    r.listing_id,
                    "title":         r.title,
                    "author_name":   names.get(r.author_id, "Anonymous"),
                    "pricing_model": r.pricing_model or "free",
                    "price_usdt":    float(r.price_usdt or 0),
                    "is_verified":   bool(r.is_verified),
                    "is_ai_generated": bool(r.is_ai_generated),
                    "pnl_pct":       round(float(r.pnl_sum or 0), 2),
                    "trades":        trades,
                    "win_rate":      wr,
                })
            return {"period": _period, "entries": entries}
        finally:
            db.close()

    _lb_payload = await asyncio.to_thread(_load)
    if _lb_payload is None:
        raise HTTPException(status_code=403)
    set_cache(_lb_ck, _lb_payload, 120)
    return _cached_json(_lb_payload, False, 120)


@app.get("/api/marketplace/{listing_id}")
async def api_marketplace_detail(listing_id: int, uid: str = Query(...)):
    from app.database import SessionLocal, engine
    db = SessionLocal()
    try:
        user = _get_user_by_uid(uid, db)
        if not user:
            raise HTTPException(status_code=403)

        from app.strategy_models import StrategyMarketplace, UserStrategy, StrategyPerformance, StrategyExecution
        from app.strategy_marketplace_ext import StrategyRating, StrategyPurchase, init_marketplace_ext_tables

        m = db.query(StrategyMarketplace).filter(StrategyMarketplace.id == listing_id).first()
        if not m:
            raise HTTPException(status_code=404)
        m.view_count = (m.view_count or 0) + 1
        db.commit()

        perf   = db.query(StrategyPerformance).filter(StrategyPerformance.strategy_id == m.strategy_id).first()
        author = db.query(User).filter(User.id == m.author_id).first()
        recent_trades = (
            db.query(StrategyExecution)
            .filter(StrategyExecution.strategy_id == m.strategy_id)
            .order_by(StrategyExecution.fired_at.desc()).limit(10).all()
        )
        ratings = (
            db.query(StrategyRating)
            .filter(StrategyRating.listing_id == listing_id)
            .order_by(StrategyRating.created_at.desc()).limit(20).all()
        )
        # is_owned = user has an active purchase/subscription record
        # (free strategies are NOT auto-owned — user must subscribe)
        is_owned = (
            m.author_id == user.id or
            db.query(StrategyPurchase).filter(
                StrategyPurchase.buyer_id == user.id,
                StrategyPurchase.listing_id == listing_id,
                StrategyPurchase.status == "active",
            ).first() is not None
        )
        my_rating = db.query(StrategyRating).filter(
            StrategyRating.rater_id == user.id, StrategyRating.listing_id == listing_id
        ).first()

        return JSONResponse({
            "id": m.id, "title": m.title, "summary": m.summary,
            "tags": m.tags or [], "category": m.category or "general",
            "pricing_model": m.pricing_model or "free", "price_usdt": m.price_usdt or 0,
            "is_verified": m.is_verified, "verified_trades": m.verified_trades or 0,
            "verified_win_rate": round(m.verified_win_rate or 0, 1),
            "avg_rating": round(m.avg_rating or 0, 1), "rating_count": m.rating_count or 0,
            "clone_count": m.clone_count or 0, "subscriber_count": m.subscriber_count or 0,
            "view_count": m.view_count or 0, "is_owned": is_owned,
            "author_name": (author.first_name or author.username or "Anonymous") if author else "Anonymous",
            "author_uid": author.uid if author else None,
            "live_performance": {
                "total_trades": perf.total_trades if perf else 0,
                "win_rate": round(perf.win_rate, 1) if perf else 0,
                "total_pnl": round(perf.total_pnl_pct, 2) if perf else 0,
            },
            "recent_trades": [{"symbol": ex.symbol, "direction": ex.direction, "outcome": ex.outcome,
                "pnl_pct": round(ex.pnl_pct, 4) if ex.pnl_pct else None,
                "entry_price": ex.entry_price, "close_price": ex.exit_price} for ex in recent_trades],
            "ratings": [{"stars": r.stars, "review": r.review, "is_verified": r.is_verified} for r in ratings],
            "my_rating": {"stars": my_rating.stars, "review": my_rating.review} if my_rating else None,
        })
    finally:
        db.close()


@app.post("/api/marketplace/{listing_id}/purchase")
async def api_purchase_strategy(listing_id: int, uid: str = Query(...), risk_scale: float = Query(1.0)):
    """Clone or purchase a marketplace listing.

    `risk_scale` (0.25 - 2.0) scales the cloned strategy's `position_size_pct`
    so a follower can copy a strategy at a fraction (or multiple) of the
    author's intended risk. Leverage is intentionally NOT scaled — that would
    change the strategy's character (different liquidation distances, fees).
    """
    # Clamp to safe bounds — silent clip is fine, no point 400-ing the user.
    try:
        risk_scale = max(0.25, min(2.0, float(risk_scale)))
    except (TypeError, ValueError):
        risk_scale = 1.0

    from app.database import SessionLocal, engine
    db = SessionLocal()
    try:
        user = _get_user_by_uid(uid, db)
        if not user:
            raise HTTPException(status_code=403)

        from app.strategy_models import StrategyMarketplace, UserStrategy, StrategyPerformance, init_strategy_tables
        from app.strategy_marketplace_ext import StrategyPurchase, CreatorEarnings, EarningsTransaction, init_marketplace_ext_tables, calculate_creator_cut, calculate_platform_cut

        # Pro subscription required to copy marketplace strategies
        # (admins bypass — same intent as every other Pro-gated endpoint).
        _psub = _get_portal_sub(user.id, db)
        if not _is_portal_pro(_psub) and not getattr(user, "is_admin", False):
            return JSONResponse(
                {"error": "PRO_REQUIRED",
                 "message": "A Pro subscription ($50/month) is required to copy strategies from the marketplace."},
                status_code=403
            )

        listing = db.query(StrategyMarketplace).filter(StrategyMarketplace.id == listing_id).first()
        if not listing:
            raise HTTPException(status_code=404)

        existing = db.query(StrategyPurchase).filter(
            StrategyPurchase.buyer_id == user.id,
            StrategyPurchase.listing_id == listing_id,
            StrategyPurchase.status == "active",
        ).first()
        if existing:
            # If the purchase exists but the cloned strategy was never created
            # (e.g. old purchases from before this flow was added, or DB errors),
            # create the strategy now so the user can actually use it.
            if not existing.cloned_strategy_id:
                original = db.query(UserStrategy).filter(UserStrategy.id == listing.strategy_id).first()
                if original:
                    _orig_risk = dict(original.config.get("risk", {}) or {})
                    if risk_scale != 1.0 and "position_size_pct" in _orig_risk:
                        try:
                            _scaled = float(_orig_risk["position_size_pct"]) * risk_scale
                            # Cap at 50% — anything higher is unsafe execution-side
                            # regardless of what the original strategy specified.
                            _orig_risk["position_size_pct"] = round(max(0.01, min(50.0, _scaled)), 3)
                        except (TypeError, ValueError):
                            pass  # leave original value if it's non-numeric
                    locked_config = {
                        "name":                listing.title or original.name,
                        "direction":           original.config.get("direction", "LONG"),
                        "risk":                _orig_risk,
                        "exit":                original.config.get("exit", {}),
                        "filters":             original.config.get("filters", {}),
                        "universe":            original.config.get("universe", {}),
                        "_locked":             True,
                        "_source_strategy_id": listing.strategy_id,
                        "_listing_id":         listing_id,
                        "_risk_scale":         risk_scale,
                    }
                    recovered = UserStrategy(
                        user_id=user.id,
                        name=listing.title or original.name,
                        description=original.description,
                        config=locked_config,
                        status="paper",
                    )
                    db.add(recovered)
                    db.commit()
                    db.refresh(recovered)
                    from app.strategy_models import StrategyPerformance
                    db.add(StrategyPerformance(strategy_id=recovered.id))
                    existing.cloned_strategy_id = recovered.id
                    db.commit()
            return JSONResponse({"already_owned": True, "cloned_strategy_id": existing.cloned_strategy_id})

        if (listing.pricing_model or "free") != "free" and (listing.price_usdt or 0) > 0:
            return JSONResponse({
                "requires_payment": True,
                "price_usdt": listing.price_usdt,
                "pricing_model": listing.pricing_model,
                "message": f"This strategy costs ${listing.price_usdt:.2f}. Pay via Telegram bot to unlock.",
            })

        # Free — subscribe (lock-linked, no config copy so IP is protected)
        original = db.query(UserStrategy).filter(UserStrategy.id == listing.strategy_id).first()
        if not original:
            raise HTTPException(status_code=404)

        _orig_risk = dict(original.config.get("risk", {}) or {})
        if risk_scale != 1.0 and "position_size_pct" in _orig_risk:
            try:
                _scaled = float(_orig_risk["position_size_pct"]) * risk_scale
                # Cap at 50% — anything higher is unsafe execution-side
                # regardless of what the original strategy specified.
                _orig_risk["position_size_pct"] = round(max(0.01, min(50.0, _scaled)), 3)
            except (TypeError, ValueError):
                pass  # leave original value if it's non-numeric
        # Build a locked stub — no entry_conditions or strategy logic exposed
        locked_config = {
            "name":             listing.title or original.name,
            "direction":        original.config.get("direction", "LONG"),
            "risk":             _orig_risk,
            "exit":             original.config.get("exit", {}),
            "filters":          original.config.get("filters", {}),
            "universe":         original.config.get("universe", {}),
            "_locked":          True,
            "_source_strategy_id": listing.strategy_id,
            "_listing_id":      listing_id,
            "_risk_scale":      risk_scale,
        }
        new_strategy = UserStrategy(
            user_id=user.id,
            name=listing.title or original.name,
            description=original.description,
            config=locked_config,
            status="paper",
        )
        db.add(new_strategy)
        db.commit()
        db.refresh(new_strategy)

        perf     = StrategyPerformance(strategy_id=new_strategy.id)
        purchase = StrategyPurchase(
            buyer_id=user.id, listing_id=listing_id, strategy_id=listing.strategy_id,
            pricing_model="free", amount_paid_usd=0.0, status="active",
            cloned_strategy_id=new_strategy.id,
        )
        db.add(perf)
        db.add(purchase)
        listing.clone_count = (listing.clone_count or 0) + 1
        db.commit()

        orig_risk = original.config.get("risk", {})
        return JSONResponse({
            "success": True,
            "cloned_strategy_id": new_strategy.id,
            "name": new_strategy.name,
            "default_leverage": orig_risk.get("leverage", 10),
            "default_position_size_pct": orig_risk.get("position_size_pct", 5),
        })
    finally:
        db.close()


@app.post("/api/marketplace/{listing_id}/rate")
async def api_rate_strategy(listing_id: int, uid: str = Query(...), stars: int = Query(...), review: str = Query("")):
    from app.database import SessionLocal, engine
    db = SessionLocal()
    try:
        user = _get_user_by_uid(uid, db)
        if not user:
            raise HTTPException(status_code=403)
        if not 1 <= stars <= 5:
            raise HTTPException(status_code=400, detail="Stars must be 1-5")

        from app.strategy_models import StrategyMarketplace
        from app.strategy_marketplace_ext import StrategyRating, StrategyPurchase, init_marketplace_ext_tables

        listing = db.query(StrategyMarketplace).filter(StrategyMarketplace.id == listing_id).first()
        if not listing:
            raise HTTPException(status_code=404)

        is_buyer = db.query(StrategyPurchase).filter(
            StrategyPurchase.buyer_id == user.id,
            StrategyPurchase.listing_id == listing_id,
            StrategyPurchase.status == "active",
        ).first() is not None

        existing = db.query(StrategyRating).filter(
            StrategyRating.rater_id == user.id, StrategyRating.listing_id == listing_id
        ).first()

        if existing:
            existing.stars = stars
            existing.review = review or None
            existing.is_verified = is_buyer
        else:
            db.add(StrategyRating(
                rater_id=user.id, listing_id=listing_id,
                stars=stars, review=review or None, is_verified=is_buyer
            ))

        db.commit()
        all_ratings = db.query(StrategyRating).filter(StrategyRating.listing_id == listing_id).all()
        if all_ratings:
            listing.avg_rating   = sum(r.stars for r in all_ratings) / len(all_ratings)
            listing.rating_count = len(all_ratings)
        db.commit()
        return JSONResponse({"success": True, "new_avg": round(listing.avg_rating, 1)})
    finally:
        db.close()


@app.get("/api/leaderboard")
async def api_leaderboard(uid: str = Query(...), metric: str = Query("win_rate")):
    from app.database import SessionLocal
    _ldr_key = f"api_ldr:{uid}:{metric}"
    _ldr_cached = get_cache(_ldr_key)
    if _ldr_cached is not None:
        return _cached_json(_ldr_cached, True, 120)

    def _load():
        db = SessionLocal()
        try:
            user = _get_user_by_uid(uid, db)
            if not user:
                return None

            from app.strategy_models import StrategyMarketplace, StrategyPerformance, UserStrategy, StrategyOffer
            from app.strategy_models import StrategyExecution
            results = []
            seen_strategy_ids = set()

            # 1. Marketplace (live-published) strategies
            listings = db.query(StrategyMarketplace).all()

            # Batch-load perfs and authors for marketplace listings (avoid N+1)
            _mkt_sids   = [m.strategy_id for m in listings]
            _mkt_auids  = list({m.author_id for m in listings if m.author_id})
            _ldr_perf1  = {p.strategy_id: p for p in db.query(StrategyPerformance).filter(StrategyPerformance.strategy_id.in_(_mkt_sids)).all()} if _mkt_sids else {}
            _ldr_auth1  = {u.id: u for u in db.query(User).filter(User.id.in_(_mkt_auids)).all()} if _mkt_auids else {}

            for m in listings:
                perf   = _ldr_perf1.get(m.strategy_id)
                author = _ldr_auth1.get(m.author_id)
                if not perf or perf.total_trades < 3 or perf.total_pnl_pct <= 0:
                    continue
                seen_strategy_ids.add(m.strategy_id)
                results.append({
                    "strategy_id": m.strategy_id,
                    "listing_id": m.id,
                    "title": m.title,
                    "author": (author.first_name or author.username) if author else "Anonymous",
                    "author_uid": author.uid if author else None,
                    "is_own": (author.id == user.id) if author else False,
                    "mode": "live",
                    "win_rate": round(perf.win_rate, 1),
                    "total_pnl": round(perf.total_pnl_pct, 2),
                    "total_trades": perf.total_trades,
                    "avg_rating": round(m.avg_rating or 0, 1),
                    "pricing_model": m.pricing_model or "free",
                    "price_usdt": m.price_usdt or 0,
                })

            # 2. Non-marketplace strategies (paper or live) with enough trade history
            all_strategies = db.query(UserStrategy).filter(
                UserStrategy.status.in_(["active", "paused", "draft", "paper"])
            ).all()

            # Batch-load perfs and authors for all strategies (avoid N+1)
            _strat_ids2  = [s.id for s in all_strategies]
            _strat_uids2 = list({s.user_id for s in all_strategies if s.user_id})
            _ldr_perf2   = {p.strategy_id: p for p in db.query(StrategyPerformance).filter(StrategyPerformance.strategy_id.in_(_strat_ids2)).all()} if _strat_ids2 else {}
            _ldr_auth2   = {u.id: u for u in db.query(User).filter(User.id.in_(_strat_uids2)).all()} if _strat_uids2 else {}

            for s in all_strategies:
                if s.id in seen_strategy_ids:
                    continue
                perf   = _ldr_perf2.get(s.id)
                author = _ldr_auth2.get(s.user_id)
                if not perf or perf.total_trades < 3 or perf.total_pnl_pct <= 0:
                    continue
                # Determine paper vs live from most recent executions
                recent = db.query(StrategyExecution).filter(
                    StrategyExecution.strategy_id == s.id
                ).order_by(StrategyExecution.fired_at.desc()).limit(5).all()
                paper_count = sum(1 for e in recent if getattr(e, 'is_paper', True))
                mode = "paper" if paper_count >= len(recent) / 2 else "live"
                # Check if current user already sent an offer
                offer_sent = db.query(StrategyOffer).filter(
                    StrategyOffer.strategy_id == s.id,
                    StrategyOffer.requester_id == user.id
                ).first()
                results.append({
                    "strategy_id": s.id,
                    "listing_id": None,
                    "title": s.name,
                    "author": (author.first_name or author.username) if author else "Anonymous",
                    "author_uid": author.uid if author else None,
                    "is_own": (author.id == user.id) if author else False,
                    "mode": mode,
                    "win_rate": round(perf.win_rate, 1),
                    "total_pnl": round(perf.total_pnl_pct, 2),
                    "total_trades": perf.total_trades,
                    "avg_rating": 0.0,
                    "pricing_model": None,
                    "price_usdt": 0,
                    "offer_sent": offer_sent is not None,
                    "offer_status": offer_sent.status if offer_sent else None,
                })

            results.sort(key=lambda x: x.get(metric if metric in x else "win_rate", 0), reverse=True)
            return results[:30]
        finally:
            db.close()

    top = await asyncio.to_thread(_load)
    if top is None:
        raise HTTPException(status_code=403)
    set_cache(_ldr_key, top, 120)
    return _cached_json(top, False, 120)


@app.post("/api/leaderboard/offer")
async def send_strategy_offer(request: Request):
    from app.database import SessionLocal
    db = SessionLocal()
    try:
        body = await request.json()
        uid         = body.get("uid", "")
        strategy_id = body.get("strategy_id")
        message     = (body.get("message") or "").strip()[:400]

        user = _get_user_by_uid(uid, db)
        if not user:
            raise HTTPException(status_code=403)

        from app.strategy_models import UserStrategy, StrategyOffer
        strategy = db.query(UserStrategy).filter(UserStrategy.id == strategy_id).first()
        if not strategy:
            raise HTTPException(status_code=404, detail="Strategy not found")
        if strategy.user_id == user.id:
            raise HTTPException(status_code=400, detail="Cannot send offer for your own strategy")

        # Prevent duplicate offers
        existing = db.query(StrategyOffer).filter(
            StrategyOffer.strategy_id == strategy_id,
            StrategyOffer.requester_id == user.id
        ).first()
        if existing:
            return JSONResponse({"ok": True, "duplicate": True})

        offer = StrategyOffer(
            strategy_id  = strategy_id,
            author_id    = strategy.user_id,
            requester_id = user.id,
            message      = message or None,
            status       = "pending",
        )
        db.add(offer)
        db.commit()

        # Notify the strategy owner via Telegram
        author = db.query(User).filter(User.id == strategy.user_id).first()
        requester_name = user.first_name or user.username or user.uid
        msg_parts = [
            "💼 <b>Strategy Access Offer</b>\n\n",
            f"<b>{requester_name}</b> ({user.uid}) wants access to your strategy:\n",
            f"<b>{strategy.name}</b> (paper testing)\n\n",
        ]
        if message:
            msg_parts.append(f"📝 {message}\n\n")
        msg_parts.append("Reply to them directly or contact them via TradeHub.")
        msg = "".join(msg_parts)
        if author and author.telegram_id and not author.telegram_id.startswith("WEB-"):
            try:
                await _tg_send_msg(author.telegram_id, msg)
            except Exception:
                pass

        return JSONResponse({"ok": True})
    finally:
        db.close()


@app.get("/api/creator/{creator_uid}")
async def api_creator_profile(creator_uid: str, uid: str = Query(...)):
    from app.database import SessionLocal, engine
    db = SessionLocal()
    try:
        viewer = _get_user_by_uid(uid, db)
        creator = db.query(User).filter(User.uid == creator_uid).first()
        if not creator:
            raise HTTPException(status_code=404)

        from app.strategy_models import StrategyMarketplace, StrategyPerformance
        from app.strategy_marketplace_ext import CreatorEarnings, init_marketplace_ext_tables
        from app.social_models import UserFollow, init_social_tables

        listings  = db.query(StrategyMarketplace).filter(StrategyMarketplace.author_id == creator.id).all()
        earnings  = db.query(CreatorEarnings).filter(CreatorEarnings.creator_id == creator.id).first()
        followers = db.query(UserFollow).filter(UserFollow.following_id == creator.id).count()
        following = db.query(UserFollow).filter(UserFollow.follower_id == creator.id).count()
        is_following = False
        if viewer:
            is_following = db.query(UserFollow).filter(
                UserFollow.follower_id == viewer.id,
                UserFollow.following_id == creator.id,
            ).first() is not None

        strat_details = []
        for m in listings:
            perf = db.query(StrategyPerformance).filter(StrategyPerformance.strategy_id == m.strategy_id).first()
            strat_details.append({
                "id": m.id, "title": m.title, "summary": m.summary,
                "pricing_model": m.pricing_model or "free", "price_usdt": m.price_usdt or 0,
                "clone_count": m.clone_count or 0, "avg_rating": round(m.avg_rating or 0, 1),
                "is_verified": m.is_verified,
                "win_rate":    round(perf.win_rate, 1) if perf else None,
                "total_pnl":   round(perf.total_pnl_pct, 2) if perf else None,
                "total_trades": perf.total_trades if perf else 0,
            })

        return JSONResponse({
            "name": creator.first_name or creator.username or "Anonymous",
            "uid": creator.uid,
            "joined": creator.created_at.strftime("%B %Y") if creator.created_at else "Unknown",
            "strategy_count": len(listings),
            "follower_count": followers,
            "following_count": following,
            "is_following": is_following,
            "total_subscribers": earnings.total_subscribers if earnings else 0,
            "total_sales": earnings.total_sales if earnings else 0,
            "strategies": strat_details,
        })
    finally:
        db.close()


@app.get("/api/my-earnings")
async def api_my_earnings(uid: str = Query(...)):
    from app.database import SessionLocal, engine
    db = SessionLocal()
    try:
        user = _get_user_by_uid(uid, db)
        if not user:
            raise HTTPException(status_code=403)

        from app.strategy_marketplace_ext import CreatorEarnings, EarningsTransaction, init_marketplace_ext_tables

        earnings   = db.query(CreatorEarnings).filter(CreatorEarnings.creator_id == user.id).first()
        recent_txs = (
            db.query(EarningsTransaction).filter(EarningsTransaction.creator_id == user.id)
            .order_by(EarningsTransaction.created_at.desc()).limit(20).all()
        )
        return JSONResponse({
            "total_earned": earnings.total_earned if earnings else 0,
            "pending_payout": earnings.pending_payout if earnings else 0,
            "total_paid_out": earnings.total_paid_out if earnings else 0,
            "total_sales": earnings.total_sales if earnings else 0,
            "platform_cut_pct": 20, "creator_cut_pct": 80,
            "recent_sales": [{"gross": tx.gross_amount, "your_cut": tx.creator_cut,
                "created_at": tx.created_at.isoformat() if tx.created_at else None} for tx in recent_txs],
        })
    finally:
        db.close()


@app.get("/api/my-purchases")
async def api_my_purchases(uid: str = Query(...)):
    from app.database import SessionLocal, engine
    db = SessionLocal()
    try:
        user = _get_user_by_uid(uid, db)
        if not user:
            raise HTTPException(status_code=403)

        from app.strategy_models import StrategyMarketplace, UserStrategy
        from app.strategy_marketplace_ext import StrategyPurchase, init_marketplace_ext_tables

        purchases = (
            db.query(StrategyPurchase)
            .filter(StrategyPurchase.buyer_id == user.id, StrategyPurchase.status == "active")
            .order_by(StrategyPurchase.purchased_at.desc()).all()
        )
        result = []
        for p in purchases:
            listing = db.query(StrategyMarketplace).filter(StrategyMarketplace.id == p.listing_id).first()
            strat   = db.query(UserStrategy).filter(UserStrategy.id == p.cloned_strategy_id).first() if p.cloned_strategy_id else None
            result.append({
                "listing_id": p.listing_id, "title": listing.title if listing else "Unknown",
                "pricing_model": p.pricing_model, "amount_paid": p.amount_paid_usd,
                "purchased_at": p.purchased_at.isoformat() if p.purchased_at else None,
                "strategy_id": strat.id if strat else None,
                "strategy_status": strat.status if strat else None,
            })
        return JSONResponse(result)
    finally:
        db.close()


@app.post("/api/marketplace/{listing_id}/clone")
async def api_clone_strategy(listing_id: int, uid: str = Query(...), risk_scale: float = Query(1.0)):
    return await api_purchase_strategy(listing_id, uid, risk_scale)


@app.post("/api/strategies/{strategy_id}/share")
async def api_share_strategy(strategy_id: int, uid: str = Query(...)):
    """Publish a strategy to the marketplace from the web portal."""
    from app.database import SessionLocal, engine
    db = SessionLocal()
    try:
        user = _get_user_by_uid(uid, db)
        if not user:
            raise HTTPException(status_code=403)

        from app.strategy_models import UserStrategy, StrategyMarketplace, init_strategy_tables

        strategy = db.query(UserStrategy).filter(
            UserStrategy.id == strategy_id,
            UserStrategy.user_id == user.id,
        ).first()
        if not strategy:
            raise HTTPException(status_code=404)

        # Locked (subscribed from marketplace) — cannot re-publish someone else's strategy
        if (strategy.config or {}).get("_locked"):
            raise HTTPException(status_code=403, detail="Marketplace subscriptions cannot be re-published")

        existing = db.query(StrategyMarketplace).filter(
            StrategyMarketplace.strategy_id == strategy_id
        ).first()
        if existing:
            raise HTTPException(status_code=409, detail="Already published")

        from app.services.strategy_builder import generate_strategy_summary
        summary = await generate_strategy_summary(strategy.config)

        listing = StrategyMarketplace(
            strategy_id   = strategy_id,
            author_id     = user.id,
            title         = strategy.name,
            summary       = summary,
            tags          = [],
            category      = "general",
            pricing_model = "free",
            price_usdt    = 0.0,
        )
        db.add(listing)
        strategy.is_public = True
        db.commit()

        # Emit a feed activity so followers see this publish
        try:
            from app.social_models import FeedActivity, init_social_tables
            db.add(FeedActivity(
                user_id       = user.id,
                activity_type = "strategy_published",
                title         = f"Published a new strategy: {strategy.name}",
                subtitle      = (summary or "")[:200],
                strategy_id   = strategy_id,
                listing_id    = listing.id,
            ))
            db.commit()
        except Exception as fe:
            logger.warning(f"feed activity create failed: {fe}")

        return JSONResponse({"success": True, "listing_id": listing.id})
    finally:
        db.close()


@app.post("/api/build-strategy")
async def api_build_strategy(request: Request):
    """Compile a strategy from plain-English description using AI."""
    body = await request.json()
    uid  = body.get("uid")
    if not uid:
        raise HTTPException(status_code=400, detail="uid required")

    from app.database import SessionLocal
    db = SessionLocal()
    try:
        user = _get_user_by_uid(uid, db)
        if not user or user.banned:
            raise HTTPException(status_code=403)
    finally:
        db.close()

    name = body.get("name", "My Strategy")
    desc = body.get("description", "")
    if not desc:
        raise HTTPException(status_code=400, detail="description required")

    from app.services.strategy_builder import (
        compile_strategy_from_conversation,
        validate_strategy,
    )
    import asyncio

    config = await compile_strategy_from_conversation([], f"Strategy name: {name}\n\n{desc}")
    if not config:
        return JSONResponse({"error": "Could not parse strategy. Try being more specific — e.g. 'SuperTrend bullish flip on 1m, LONG only, 10× leverage, 2% TP, 1% SL'."}, status_code=422)

    config["name"]        = name
    config["description"] = desc

    # Run validation in parallel with a short timeout so a slow AI call doesn't block
    try:
        validation = await asyncio.wait_for(validate_strategy(config), timeout=20.0)
    except asyncio.TimeoutError:
        validation = {
            "valid": True, "warnings": [], "suggestions": [],
            "summary": desc, "risk_rating": "MEDIUM",
        }
    except Exception:
        validation = {
            "valid": True, "warnings": [], "suggestions": [],
            "summary": desc, "risk_rating": "MEDIUM",
        }

    return JSONResponse({
        "config":      config,
        "warnings":    validation.get("warnings", []),
        "suggestions": validation.get("suggestions", []),
        "risk_rating": validation.get("risk_rating", "MEDIUM"),
        "summary":     validation.get("summary", desc),
    })


@app.post("/api/build-from-scan")
async def api_build_from_scan(request: Request):
    """
    Deterministic compile for Index/Gold finder leaderboard rows — no AI call.
    Used by 'Build all (paper)' so 15 strategies save reliably in one batch.
    """
    body = await request.json()
    uid = body.get("uid")
    if not uid:
        raise HTTPException(status_code=400, detail="uid required")
    entry = body.get("entry")
    if not isinstance(entry, dict):
        raise HTTPException(status_code=400, detail="entry object required")

    from app.database import SessionLocal
    db = SessionLocal()
    try:
        user = _get_user_by_uid(uid, db)
        if not user or user.banned:
            raise HTTPException(status_code=403)
    finally:
        db.close()

    symbol = (body.get("symbol") or entry.get("symbol") or "NAS100").upper()
    asset_class = (body.get("asset_class") or entry.get("asset_class") or "index").lower()

    from app.services.gold_strategy_scanner import compile_scan_entry_to_config
    try:
        config = compile_scan_entry_to_config(
            entry,
            symbol=symbol,
            asset_class=asset_class,
            name=body.get("name") or entry.get("build_name"),
        )
    except Exception as e:
        logger.warning(f"build-from-scan compile failed: {e}")
        raise HTTPException(status_code=422, detail=f"Could not compile scan entry: {e}")

    return JSONResponse({"config": config, "compiled": "deterministic"})


@app.post("/api/build-from-live-scan")
async def api_build_from_live_scan(request: Request):
    """
    Compile a live structure scanner hit into a paper strategy and save it.
    Body: { uid, signal: <scanner row>, name? }
    """
    body = await request.json()
    uid = body.get("uid")
    if not uid:
        raise HTTPException(status_code=400, detail="uid required")
    signal = body.get("signal")
    if not isinstance(signal, dict):
        raise HTTPException(status_code=400, detail="signal object required")

    from app.database import SessionLocal
    db = SessionLocal()
    try:
        user = _get_user_by_uid(uid, db)
        if not user or user.banned:
            raise HTTPException(status_code=403)
    finally:
        db.close()

    from app.services.gold_strategy_scanner import compile_live_scan_to_config
    try:
        config = compile_live_scan_to_config(signal, name=body.get("name"))
    except Exception as e:
        logger.warning(f"build-from-live-scan compile failed: {e}")
        raise HTTPException(status_code=422, detail=f"Could not compile live signal: {e}")

    config["_build_mode"] = "paper"
    return await api_save_strategy(_JsonBodyRequest({"uid": uid, "config": config}))


class _JsonBodyRequest:
    """Minimal request shim so build-from-live-scan can reuse api_save_strategy."""
    def __init__(self, body: dict):
        self._body = body
        self.headers: dict = {}

    async def json(self):
        return self._body


# ─── Forex multi-pair market structure scanner ───────────────────────────────
_scanner_mod = None


def _scanner():
    global _scanner_mod
    if _scanner_mod is None:
        from app.services import live_structure_scanner as _scanner_mod
    return _scanner_mod


def _auth_scanner_uid(uid: str):
    from app.database import SessionLocal
    db = SessionLocal()
    try:
        user = _get_user_by_uid(uid, db)
        if not user or user.banned:
            raise HTTPException(status_code=403)
        return user
    finally:
        db.close()


@app.post("/api/forex/scanner/start")
async def api_forex_scanner_start(request: Request):
    """Start a streaming live scan — poll GET /api/forex/scanner/progress for results."""
    body = await request.json()
    uid = (body.get("uid") or "").strip()
    if not uid:
        raise HTTPException(status_code=400, detail="uid required")
    _auth_scanner_uid(uid)
    mode = (body.get("mode") or "fast").lower()
    if mode not in ("fast", "full"):
        mode = "fast"
    return JSONResponse(content=_scanner().start_live_scan(uid, mode=mode))


@app.get("/api/forex/scanner/progress")
async def api_forex_scanner_progress(uid: str = Query(...)):
    """Poll streaming scan — signals grow until status=done."""
    return JSONResponse(content=_scanner().live_scan_progress(uid))


@app.get("/api/forex/scanner")
async def api_forex_scanner(request: Request):
    """Sync fast scan (legacy/mobile). Prefer POST start + progress for streaming."""
    uid = request.query_params.get("uid", "")
    if not uid:
        raise HTTPException(status_code=400, detail="uid required")
    _auth_scanner_uid(uid)
    mode = (request.query_params.get("mode") or "fast").lower()
    if mode not in ("fast", "full"):
        mode = "fast"
    sc = _scanner()
    try:
        signals, partial = await sc.run_forex_scanner(mode=mode)
    except Exception as e:
        logger.warning(f"forex scanner error: {e}")
        signals, partial = [], False
    return JSONResponse(content=sc.build_scan_response(signals, mode=mode, partial=partial))


@app.get("/api/forex/scanner/best")
async def api_forex_scanner_best(request: Request):
    """Top structure signals right now — uses shared 60s cache when available."""
    uid = request.query_params.get("uid", "")
    if uid:
        _auth_scanner_uid(uid)
    mode = (request.query_params.get("mode") or "fast").lower()
    if mode not in ("fast", "full"):
        mode = "fast"
    limit = min(20, max(1, int(request.query_params.get("limit") or 8)))
    sc = _scanner()
    cached = sc.get_shared_scan(mode)
    if cached and cached.get("signals"):
        signals = cached["signals"][:limit]
        out = dict(cached)
        out["signals"] = signals
        out["count"] = len(signals)
        out["source"] = "cache"
        return JSONResponse(out)
    try:
        signals, partial = await sc.run_forex_scanner(mode=mode)
    except Exception as e:
        logger.warning(f"forex scanner best error: {e}")
        signals, partial = [], False
    resp = sc.build_scan_response(signals[:limit], mode=mode, partial=partial)
    resp["source"] = "live"
    return JSONResponse(resp)


@app.get("/api/strategies/{strategy_id}/gate-stats")
async def api_strategy_gate_stats(strategy_id: int, uid: str = Query(...)):
    """Why a strategy did or didn't fire on the last executor cycle."""
    from app.database import SessionLocal
    from app.strategy_models import UserStrategy
    db = SessionLocal()
    try:
        user = _get_user_by_uid(uid, db)
        if not user:
            raise HTTPException(status_code=403)
        strat = db.query(UserStrategy).filter(
            UserStrategy.id == strategy_id,
            UserStrategy.user_id == user.id,
        ).first()
        if not strat:
            raise HTTPException(status_code=404)
        strat_name = strat.name
        strat_status = strat.status
    finally:
        db.close()
    from app.services.ctrader_order_queue import get_gate_stats
    return JSONResponse({
        "strategy_id": strategy_id,
        "name": strat_name,
        "status": strat_status,
        **get_gate_stats(strategy_id),
    })


@app.get("/api/markets/catalog")
async def api_markets_catalog():
    """
    Return the per-asset-class symbol catalog used by the wizard symbol picker.
    Shape: { "stock": [{symbol, name}, ...], "forex": [...], "index": [...] }
    Crypto is omitted on purpose — its universe is dynamic (top gainers etc.)
    and the existing /api/coins-by-volume endpoint handles its listing.
    """
    from app.services.asset_classes import catalog_for_api, is_market_open, ASSET_CLASSES
    cat = catalog_for_api()
    status = {cls: is_market_open(cls) for cls in ASSET_CLASSES}
    return JSONResponse({"catalog": cat, "market_open": status})


@app.post("/api/save-strategy")
async def api_save_strategy(request: Request):
    """Save a compiled strategy config to the database."""
    body = await request.json()
    uid  = body.get("uid")
    if not uid:
        raise HTTPException(status_code=400, detail="uid required")

    config = body.get("config")
    if not config:
        raise HTTPException(status_code=400, detail="config required")

    from app.database import SessionLocal, engine
    from app.strategy_models import UserStrategy, StrategyPerformance, init_strategy_tables

    db = SessionLocal()
    try:
        user = _get_user_by_uid(uid, db)
        if not user or user.banned:
            raise HTTPException(status_code=403)

        # Sanity-check the universe spec — a strategy with type='specific' but no
        # symbols listed will silently never fire (the executor's eligibility
        # filter excludes every symbol). This is the single most common bug
        # class for AI-compiled chat-builder strategies.
        _uni = config.get("universe") or {}
        if _uni.get("type") == "specific":
            _syms = [
                s.strip().upper()
                for s in (_uni.get("symbols") or [])
                if isinstance(s, str) and s.strip()
            ]
            if not _syms:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        "Universe is set to 'specific coins' but no symbols were "
                        "listed. Either add at least one coin (e.g. BTCUSDT, ETHUSDT) "
                        "or change the universe type to 'all coins'."
                    ),
                )
            # Normalize so the executor's case/whitespace-sensitive matching
            # always sees a clean list.
            _uni["symbols"] = _syms
            config["universe"] = _uni

        # Determine initial status from build mode flag and portal settings
        build_mode = config.get("_build_mode", "live")
        from app.strategy_models import StrategyPortalSettings
        portal = db.query(StrategyPortalSettings).filter(StrategyPortalSettings.user_id == user.id).first()

        _psub2 = _get_portal_sub(user.id, db)
        _user_is_pro = _is_portal_pro(_psub2) or bool(getattr(user, "is_admin", False))

        if build_mode == "paper":
            initial_status = "paper"
        elif portal and portal.paper_mode_default:
            initial_status = "paper"
        elif not _user_is_pro:
            # Free users can never start a live strategy — silently downgrade to paper
            initial_status = "paper"
            config["_build_mode"] = "paper"
        elif portal and portal.auto_activate:
            initial_status = "active"
        else:
            initial_status = "draft"

        import secrets as _secrets
        _entry = config.get("entry_conditions", {})
        _is_webhook = (
            _entry.get("entry_type") == "tradingview_webhook"
            or any(
                c.get("type") == "tradingview_webhook"
                for c in (_entry.get("conditions") or [])
            )
        )
        _wh_token = _secrets.token_urlsafe(32) if _is_webhook else None

        # Asset class — crypto (default) / stock / forex / index.
        # Forex + indices go live via cTrader (FP Markets); stocks are paper-only.
        from app.services.asset_classes import (
            normalize_asset_class, PAPER_ONLY_CLASSES, is_supported,
        )
        _asset_class = normalize_asset_class(config.get("asset_class"))
        # cTrader live gate — covers both forex and index CFDs.
        _ctrader_live_ok = False
        if _asset_class in ("forex", "index"):
            try:
                from app.models import UserPreference as _UP_save
                _ps = db.query(_UP_save).filter(_UP_save.user_id == user.id).first()
                _ready, _ = _user_ctrader_live_ready(_ps, user)
                _ctrader_live_ok = _ready
            except Exception:
                _ctrader_live_ok = False
        # Tradfi strategies always start as paper so the executor scans them
        # immediately — never leave forex/index on draft waiting for manual promote.
        if _asset_class in PAPER_ONLY_CLASSES:
            initial_status = "paper"
            config["_build_mode"] = "paper"
        if _asset_class in PAPER_ONLY_CLASSES:
            # Require an explicit, non-empty, catalog-valid symbol list — non-crypto
            # strategies have no dynamic universe (no "all gainers" etc.) and would
            # otherwise silently never fire.
            _uni2 = config.get("universe") or {}
            if (_uni2.get("type") or "all") != "specific":
                raise HTTPException(
                    status_code=400,
                    detail=f"{_asset_class.title()} strategies must pick specific symbols "
                           f"(no dynamic universe).",
                )
            _syms = [s for s in (_uni2.get("symbols") or []) if isinstance(s, str) and s.strip()]
            if not _syms:
                raise HTTPException(
                    status_code=400,
                    detail=f"Pick at least one {_asset_class} symbol from the wizard catalog.",
                )
            for _s in _syms:
                if not is_supported(_asset_class, _s):
                    raise HTTPException(
                        status_code=400,
                        detail=f"Symbol '{_s}' is not in our {_asset_class} catalog. "
                               f"Pick one from the list shown in the wizard.",
                    )
        config["asset_class"] = _asset_class

        # ── R:R guardrail — TP must be > SL (minimum 1.5:1) ──────────────────
        # Catches AI-builder outputs that set equal TP/SL (e.g. 0.5%/0.5%).
        # Auto-corrects silently so the save still succeeds; the UI shows the
        # corrected values.  Applies to both new strategies and improve-mode updates.
        _ex_guard = config.get("exit") or {}
        _tp_pct = _ex_guard.get("take_profit_pct")
        _sl_pct = _ex_guard.get("stop_loss_pct")
        if (
            isinstance(_tp_pct, (int, float)) and _tp_pct > 0
            and isinstance(_sl_pct, (int, float)) and _sl_pct > 0
            and _tp_pct < _sl_pct * 1.5
        ):
            _ex_guard = dict(_ex_guard)
            _ex_guard["take_profit_pct"] = round(_sl_pct * 2.0, 2)  # push to 2:1 minimum
            config["exit"] = _ex_guard
        # Also handle top-level tp1/sl keys (older config shape)
        _tp1_top = config.get("tp1")
        _sl_top  = config.get("sl")
        if (
            isinstance(_tp1_top, (int, float)) and _tp1_top > 0
            and isinstance(_sl_top,  (int, float)) and _sl_top  > 0
            and _tp1_top < _sl_top * 1.5
        ):
            config["tp1"] = round(_sl_top * 2.0, 2)

        # Defense-in-depth: paper-only asset classes are forced to 1× leverage.
        # Forex/index with cTrader connected can use real leverage.
        if _asset_class in PAPER_ONLY_CLASSES and not _ctrader_live_ok:
            _risk = config.get("risk") or {}
            if int(_risk.get("leverage", 1) or 1) != 1:
                _risk["leverage"] = 1
                config["risk"] = _risk

        # Strip crypto-only signal types from non-crypto strategies — these
        # signals (funding rate, OI, liquidations) only exist on perp markets
        # and would silently never fire on stocks/forex/indices.
        if _asset_class != "crypto":
            _CRYPTO_ONLY_SIGS = {"funding_rate", "open_interest", "liquidation"}
            _entry2 = config.get("entry_conditions") or {}
            _conds  = _entry2.get("conditions") or []
            _filtered = [
                c for c in _conds
                if not (isinstance(c, dict) and c.get("type") in _CRYPTO_ONLY_SIGS)
            ]
            if len(_filtered) != len(_conds):
                _entry2["conditions"] = _filtered
                config["entry_conditions"] = _entry2

        existing_strategy_id = body.get("existing_strategy_id")
        if existing_strategy_id:
            # ── IMPROVE MODE: update existing strategy in place ──────────────
            strategy = db.query(UserStrategy).filter(
                UserStrategy.id == existing_strategy_id,
                UserStrategy.user_id == user.id,
            ).first()
            if not strategy:
                raise HTTPException(status_code=404, detail="Strategy not found or not yours.")
            if getattr(strategy, "is_locked", False):
                raise HTTPException(status_code=400, detail="LOCKED — this strategy cannot be edited.")
            strategy.name        = config.get("name", strategy.name)
            strategy.description = config.get("description", strategy.description or "")
            strategy.config      = config
            strategy.asset_class = _asset_class
            if _wh_token:
                strategy.webhook_token = _wh_token
            db.commit()
            db.refresh(strategy)
        else:
            # ── CREATE MODE: new strategy ─────────────────────────────────────
            strategy = UserStrategy(
                user_id       = user.id,
                name          = config.get("name", "My Strategy"),
                description   = config.get("description", ""),
                config        = config,
                status        = initial_status,
                webhook_token = _wh_token,
                asset_class   = _asset_class,
            )
            db.add(strategy)
            db.commit()
            db.refresh(strategy)

            perf = StrategyPerformance(strategy_id=strategy.id)
            db.add(perf)
            db.commit()

        _resp: dict = {"id": strategy.id, "name": strategy.name, "status": strategy.status}
        if _wh_token:
            _resp["webhook_url"] = f"https://{os.environ.get('REPLIT_DEV_DOMAIN', request.headers.get('host', ''))}/api/webhooks/tv/{_wh_token}"
            _resp["webhook_token"] = _wh_token
        return JSONResponse(_resp)
    finally:
        db.close()


@app.post("/api/webhooks/tv/{token}")
async def api_tradingview_webhook(token: str, request: Request):
    """Receive a TradingView alert and fire a trade for the linked strategy."""
    from app.database import SessionLocal
    from app.strategy_models import (
        UserStrategy, StrategyExecution, StrategyPerformance,
        StrategyPortalSettings, PortalSubscription,
    )
    from app.models import User
    import json as _json

    try:
        raw = await request.body()
        body = {}
        if raw:
            try:
                body = _json.loads(raw)
            except Exception:
                text = raw.decode("utf-8", errors="replace").strip()
                body = {"symbol": text.upper().replace(" ", "")} if text else {}
    except Exception:
        body = {}

    db = SessionLocal()
    try:
        strategy = db.query(UserStrategy).filter(
            UserStrategy.webhook_token == token
        ).first()
        if not strategy:
            raise HTTPException(status_code=404, detail="Unknown webhook token")

        if strategy.status not in ("active", "paper", "draft"):
            return JSONResponse({"ok": False, "reason": "strategy_paused"})

        user = db.query(User).filter(User.id == strategy.user_id).first()
        if not user or user.banned:
            raise HTTPException(status_code=403, detail="User unavailable")

        # Subscription gates removed — all features are free.

        config = dict(strategy.config or {})
        risk = config.get("risk", {})
        ex_config = config.get("exit", {})
        direction_pref = config.get("direction", "LONG")

        symbol_raw = body.get("symbol") or body.get("ticker") or ""
        symbol = symbol_raw.strip().upper().replace("/", "").replace("-", "")
        if ":" in symbol:
            symbol = symbol.split(":")[-1]
        import re as _re
        symbol = _re.sub(r"[^A-Z0-9]", "", symbol)
        if not symbol:
            uni = config.get("universe", {})
            syms = uni.get("symbols", [])
            if syms:
                symbol = syms[0]
            else:
                raise HTTPException(status_code=400, detail="No symbol in payload or strategy universe")

        if not symbol.endswith("USDT"):
            symbol = symbol + "USDT"

        alert_dir = (body.get("direction") or body.get("side") or body.get("action") or "").upper()
        if alert_dir in ("LONG", "BUY"):
            direction = "LONG"
        elif alert_dir in ("SHORT", "SELL"):
            direction = "SHORT"
        elif direction_pref == "BOTH":
            direction = "LONG"
        else:
            direction = direction_pref

        import httpx as _httpx
        async with _httpx.AsyncClient(timeout=5) as hc:
            try:
                r = await hc.get("https://api.mexc.com/api/v3/ticker/price", params={"symbol": symbol})
                current_price = float(r.json().get("price", 0))
            except Exception:
                try:
                    r = await hc.get("https://api.binance.com/api/v3/ticker/price", params={"symbol": symbol})
                    current_price = float(r.json().get("price", 0))
                except Exception:
                    raise HTTPException(status_code=502, detail=f"Could not fetch price for {symbol}")

        if not current_price:
            raise HTTPException(status_code=502, detail=f"Price is 0 for {symbol}")

        tp_pct = float(ex_config.get("take_profit_pct", 3.0))
        tp2_pct = ex_config.get("take_profit2_pct")
        sl_pct = float(ex_config.get("stop_loss_pct", 1.5))
        leverage = int(risk.get("leverage", 10))

        if direction == "LONG":
            tp_price = current_price * (1 + tp_pct / 100)
            sl_price = current_price * (1 - sl_pct / 100)
            tp2_price = current_price * (1 + float(tp2_pct) / 100) if tp2_pct else None
        else:
            tp_price = current_price * (1 - tp_pct / 100)
            sl_price = current_price * (1 + sl_pct / 100)
            tp2_price = current_price * (1 - float(tp2_pct) / 100) if tp2_pct else None

        from app.services.strategy_executor import (
            _daily_execution_count, _open_execution_count,
            _fired_today_for_symbol, _last_any_fired_time,
        )
        max_per_day = int(risk.get("max_trades_per_day", 3))
        max_open = int(risk.get("max_open_positions", 1))
        cooldown_mins = int(risk.get("cooldown_minutes", 30))
        no_dup = bool(risk.get("no_duplicate_symbol", False))

        if _daily_execution_count(strategy.id, db) >= max_per_day:
            return JSONResponse({"ok": False, "reason": "daily_trade_limit"})
        if _open_execution_count(strategy.id, db) >= max_open:
            return JSONResponse({"ok": False, "reason": "max_open_positions"})
        last_global = _last_any_fired_time(strategy.id, db)
        if last_global:
            elapsed = (_now - last_global).total_seconds() / 60
            if elapsed < cooldown_mins:
                return JSONResponse({"ok": False, "reason": "cooldown", "retry_after_sec": int((cooldown_mins - elapsed) * 60)})
        if no_dup and _fired_today_for_symbol(strategy.id, symbol, db):
            return JSONResponse({"ok": False, "reason": "duplicate_symbol_today"})

        is_paper = strategy.status != "active"

        execution = StrategyExecution(
            strategy_id=strategy.id,
            user_id=user.id,
            symbol=symbol,
            direction=direction,
            entry_price=current_price,
            tp_price=tp_price,
            tp2_price=tp2_price,
            sl_price=sl_price,
            leverage=leverage,
            outcome="OPEN",
            conditions_met=["📺 TradingView webhook alert"],
            fired_at=_now,
            is_paper=is_paper,
            asset_class=getattr(strategy, "asset_class", None) or "crypto",
        )
        db.add(execution)
        db.commit()
        db.refresh(execution)

        _perf = db.query(StrategyPerformance).filter(
            StrategyPerformance.strategy_id == strategy.id
        ).first()
        if _perf:
            _perf.open_trades = (_perf.open_trades or 0) + 1
        else:
            _perf = StrategyPerformance(strategy_id=strategy.id, open_trades=1)
            db.add(_perf)
        db.commit()

        try:
            from app.services.strategy_executor import (
                _claim_tg_open_notify, _telegram_int_id, _tg_send, _fmt_open_card,
            )
            tg_id = _telegram_int_id(user)
            if tg_id and _claim_tg_open_notify(db, execution.id):
                portal_settings = db.query(StrategyPortalSettings).filter(
                    StrategyPortalSettings.user_id == user.id
                ).first()
                if not portal_settings or portal_settings.dm_paper_alerts or not is_paper:
                    import asyncio as _aio
                    _aio.create_task(_tg_send(
                        tg_id,
                        _fmt_open_card(
                            strategy_name=strategy.name,
                            symbol=symbol,
                            direction=direction,
                            entry=current_price,
                            tp_price=tp_price,
                            tp_pct=tp_pct,
                            tp2_price=tp2_price,
                            tp2_pct=float(tp2_pct) if tp2_pct else None,
                            sl_price=sl_price,
                            sl_pct=sl_pct,
                            leverage=leverage,
                            conditions=["📺 TradingView webhook alert"],
                            is_paper=is_paper,
                        ),
                    ))
        except Exception as _tg_err:
            logger.warning(f"Webhook TG notify failed: {_tg_err}")

        try:
            from app.services.expo_push import notify_user_bg
            _coin = symbol.replace("USDT", "")
            notify_user_bg(
                user.id,
                title=f"{'📝' if is_paper else '🚀'} {strategy.name}",
                body=f"{'Paper' if is_paper else 'Live'}: {_coin} {direction} {leverage}x @ ${current_price:,.4f}",
                data={"type": "trade_open", "strategy_id": strategy.id, "kind": "paper" if is_paper else "live"},
                kind="paper" if is_paper else "live",
            )
        except Exception:
            pass

        if not is_paper:
            try:
                from app.services.strategy_trader import place_bitunix_order_for_user
                order_result = await place_bitunix_order_for_user(
                    user=user,
                    symbol=symbol,
                    direction=direction,
                    leverage=leverage,
                    entry_price=current_price,
                    tp_pct=tp_pct,
                    sl_pct=sl_pct,
                    risk_pct=float(risk.get("position_size_pct", 5)),
                    risk_usd=float(risk["position_size_usd"]) if risk.get("position_size_type") == "fixed" and risk.get("position_size_usd") else None,
                )
                if order_result:
                    _oid = order_result.get("order_id")
                    if _oid:
                        execution.bitunix_order_id = str(_oid)
                    _fill = order_result.get("actual_fill")
                    if _fill:
                        execution.entry_price = float(_fill)
                    db.commit()
                else:
                    execution.is_paper = True
                    execution.notes = "Live->Paper fallback (webhook): Bitunix returned no result"
                    db.commit()
            except Exception as _ord_err:
                execution.is_paper = True
                execution.notes = f"Live->Paper fallback (webhook): {str(_ord_err)[:200]}"
                db.commit()
                logger.warning(f"Webhook live order failed for strategy {strategy.id}: {_ord_err}")

        mode = "paper" if execution.is_paper else "live"
        logger.info(
            f"[Webhook] Strategy {strategy.id} '{strategy.name}' — "
            f"{symbol} {direction} {leverage}x @ {current_price} ({mode})"
        )

        return JSONResponse({
            "ok": True,
            "execution_id": execution.id,
            "symbol": symbol,
            "direction": direction,
            "entry_price": current_price,
            "mode": mode,
        })
    finally:
        db.close()


@app.get("/api/strategies/{strategy_id}/webhook-url")
async def api_get_webhook_url(strategy_id: int, request: Request, uid: str = Query(...)):
    """Return the webhook URL for a strategy (if it has one)."""
    from app.database import SessionLocal
    db = SessionLocal()
    try:
        user = _get_user_by_uid(uid, db)
        if not user:
            raise HTTPException(status_code=403)
        strategy = db.query(UserStrategy).filter(
            UserStrategy.id == strategy_id,
            UserStrategy.user_id == user.id,
        ).first()
        if not strategy:
            raise HTTPException(status_code=404)
        if not strategy.webhook_token:
            return JSONResponse({"webhook_url": None})
        base = public_base_url(request)
        return JSONResponse({
            "webhook_url": f"{base}/api/webhooks/tv/{strategy.webhook_token}",
            "webhook_token": strategy.webhook_token,
        })
    finally:
        db.close()


@app.get("/api/strategies/templates")
async def api_strategy_templates():
    """Pre-built strategy templates the user can start from."""
    templates = [
        {
            "id": "fvg_bounce",
            "name": "FVG Bounce",
            "emoji": "🧲",
            "category": "smc",
            "tagline": "Smart Money Concepts — enter on fair value gap fills",
            "description": "Long when price pulls back into a bullish Fair Value Gap (FVG) created on the 15m chart. Entry confirmed when price enters the gap zone with RSI between 40-60, signaling momentum reset rather than breakdown.",
            "direction": "LONG",
            "leverage": 20,
            "position_size_pct": 5,
            "take_profit_pct": 5,
            "stop_loss_pct": 2.5,
            "take_profit2_pct": 8,
            "trailing_stop": False,
            "max_trades_per_day": 3,
            "cooldown_minutes": 90,
            "daily_loss_limit_pct": 8,
            "max_open_positions": 2,
            "difficulty": "Intermediate",
            "style": "Swing",
        },
        {
            "id": "rsi_reversal",
            "name": "RSI Oversold Reversal",
            "emoji": "📉",
            "category": "reversal",
            "tagline": "Buy extreme fear, ride the recovery",
            "description": "Long when RSI drops below 28 on the 15m timeframe and starts turning up. Requires the 1h RSI to also be below 45 to confirm the broader oversold context. Volume must be above average to confirm accumulation.",
            "direction": "LONG",
            "leverage": 10,
            "position_size_pct": 6,
            "take_profit_pct": 4,
            "stop_loss_pct": 2,
            "take_profit2_pct": 7,
            "trailing_stop": False,
            "max_trades_per_day": 4,
            "cooldown_minutes": 60,
            "daily_loss_limit_pct": 6,
            "max_open_positions": 3,
            "difficulty": "Beginner",
            "style": "Reversal",
        },
        {
            "id": "volume_scalp",
            "name": "Volume Spike Scalp",
            "emoji": "⚡",
            "category": "scalp",
            "tagline": "Ride the volume surge for quick 2-3%",
            "description": "Long when volume spikes 2x above the 20-period average on the 5m chart with a bullish candle body (close > open). RSI must be between 45-65 to avoid overbought entries. Tight TP and SL for clean R:R.",
            "direction": "LONG",
            "leverage": 15,
            "position_size_pct": 4,
            "take_profit_pct": 2,
            "stop_loss_pct": 1,
            "take_profit2_pct": None,
            "trailing_stop": False,
            "max_trades_per_day": 6,
            "cooldown_minutes": 30,
            "daily_loss_limit_pct": 5,
            "max_open_positions": 2,
            "difficulty": "Beginner",
            "style": "Scalp",
        },
        {
            "id": "macd_momentum",
            "name": "MACD Momentum Cross",
            "emoji": "📊",
            "category": "momentum",
            "tagline": "Classic crossover with trend confirmation",
            "description": "Enter long when MACD (8,21,5) crosses bullish on the 15m chart with the signal line turning up. EMA 21 must be above EMA 50 for trend alignment. Avoid entries if RSI is above 72 (overbought). Short the same setup inverted.",
            "direction": "BOTH",
            "leverage": 10,
            "position_size_pct": 5,
            "take_profit_pct": 3,
            "stop_loss_pct": 1.5,
            "take_profit2_pct": 5,
            "trailing_stop": False,
            "max_trades_per_day": 4,
            "cooldown_minutes": 60,
            "daily_loss_limit_pct": 6,
            "max_open_positions": 2,
            "difficulty": "Beginner",
            "style": "Momentum",
        },
        {
            "id": "bb_squeeze",
            "name": "Bollinger Squeeze Breakout",
            "emoji": "🔥",
            "category": "breakout",
            "tagline": "Low volatility squeezes lead to big moves",
            "description": "Enter when Bollinger Bands (20,2) squeeze tight for 10+ candles on the 15m chart and then price breaks out with volume 1.5x above average. Long when price breaks above upper band, short when it breaks below lower band.",
            "direction": "BOTH",
            "leverage": 15,
            "position_size_pct": 5,
            "take_profit_pct": 4,
            "stop_loss_pct": 2,
            "take_profit2_pct": 7,
            "trailing_stop": True,
            "trailing_stop_pct": 1.5,
            "max_trades_per_day": 3,
            "cooldown_minutes": 60,
            "daily_loss_limit_pct": 6,
            "max_open_positions": 2,
            "difficulty": "Intermediate",
            "style": "Breakout",
        },
        {
            "id": "pump_fade",
            "name": "Pump Fader",
            "emoji": "🩸",
            "category": "reversal",
            "tagline": "Short the over-extension, ride the dump",
            "description": "Short when a coin pumps 8% or more in 15 minutes on high volume with RSI above 80 on the 5m chart. The EMA 8 must be sharply above EMA 21 showing parabolic extension. Wait for the first 5m red candle to confirm reversal before entry.",
            "direction": "SHORT",
            "leverage": 10,
            "position_size_pct": 4,
            "take_profit_pct": 3,
            "stop_loss_pct": 1.5,
            "take_profit2_pct": 5,
            "trailing_stop": False,
            "max_trades_per_day": 4,
            "cooldown_minutes": 45,
            "daily_loss_limit_pct": 6,
            "max_open_positions": 2,
            "difficulty": "Intermediate",
            "style": "Reversal",
        },
        {
            "id": "ema_ribbon",
            "name": "EMA Ribbon Long",
            "emoji": "🎯",
            "category": "momentum",
            "tagline": "Trend-following with multi-EMA confluence",
            "description": "Long when the EMA 8, 21, and 50 are aligned in bullish order (8 > 21 > 50) on the 15m chart. Price must pull back to touch the EMA 21 and bounce with RSI between 45-62. This is a trend-continuation entry after a healthy pullback.",
            "direction": "LONG",
            "leverage": 12,
            "position_size_pct": 5,
            "take_profit_pct": 4,
            "stop_loss_pct": 2,
            "take_profit2_pct": 7,
            "trailing_stop": True,
            "trailing_stop_pct": 1,
            "max_trades_per_day": 4,
            "cooldown_minutes": 60,
            "daily_loss_limit_pct": 6,
            "max_open_positions": 3,
            "difficulty": "Beginner",
            "style": "Swing",
        },
        {
            "id": "support_bounce",
            "name": "Support Zone Bounce",
            "emoji": "🪃",
            "category": "reversal",
            "tagline": "Buy key support, tight SL below the zone",
            "description": "Long when price tests a major support level (previous swing low within ±1% on 1h chart) with RSI showing bullish divergence (price makes lower low but RSI makes higher low). Volume must confirm with a spike on the support candle.",
            "direction": "LONG",
            "leverage": 8,
            "position_size_pct": 7,
            "take_profit_pct": 6,
            "stop_loss_pct": 3,
            "take_profit2_pct": 10,
            "trailing_stop": False,
            "max_trades_per_day": 2,
            "cooldown_minutes": 120,
            "daily_loss_limit_pct": 6,
            "max_open_positions": 2,
            "difficulty": "Advanced",
            "style": "Swing",
        },
        # ── Single-coin strategies ─────────────────────────────────────────
        {
            "id": "btc_precision_scalp",
            "name": "BTC Precision Scalp",
            "emoji": "₿",
            "category": "scalp",
            "tagline": "Bitcoin only — EMA golden cross on 5m with RSI filter",
            "description": "Trades Bitcoin exclusively. Enter long when the 5m EMA 8 crosses above EMA 21 and RSI is between 50–70 (momentum confirmed but not overbought). SuperTrend must be bullish on the same timeframe. Tight TP/SL for disciplined R:R. Best during London/NY overlap hours.",
            "direction": "LONG",
            "single_coin": "BTCUSDT",
            "leverage": 15,
            "position_size_pct": 5,
            "take_profit_pct": 1.5,
            "stop_loss_pct": 0.8,
            "take_profit2_pct": 2.5,
            "trailing_stop": False,
            "max_trades_per_day": 6,
            "cooldown_minutes": 45,
            "daily_loss_limit_pct": 5,
            "max_open_positions": 1,
            "difficulty": "Intermediate",
            "style": "Scalp",
        },
        {
            "id": "eth_golden_ratio",
            "name": "ETH Golden Ratio",
            "emoji": "Ξ",
            "category": "swing",
            "tagline": "Ethereum only — buy the 0.618 Fibonacci retracement",
            "description": "Trades Ethereum exclusively on the 4h timeframe. Enter long when price retraces to the 0.618 Fibonacci level from the most recent swing high. Requires RSI below 50 and a bullish candlestick pattern (hammer or engulfing) to confirm the bounce. Wide TP for full swing capture.",
            "direction": "LONG",
            "single_coin": "ETHUSDT",
            "leverage": 8,
            "position_size_pct": 7,
            "take_profit_pct": 8,
            "stop_loss_pct": 4,
            "take_profit2_pct": 14,
            "trailing_stop": True,
            "trailing_stop_pct": 2,
            "max_trades_per_day": 2,
            "cooldown_minutes": 180,
            "daily_loss_limit_pct": 6,
            "max_open_positions": 1,
            "difficulty": "Intermediate",
            "style": "Swing",
        },
        {
            "id": "sol_momentum_burst",
            "name": "SOL Chain Momentum",
            "emoji": "◎",
            "category": "momentum",
            "tagline": "Solana only — catches fast momentum bursts with volume confirmation",
            "description": "Trades Solana exclusively. Enter long after 3 consecutive green 5m candles where each candle closes higher than the previous, combined with volume spiking 1.5× above the 20-period average. RSI must be between 55–75. SOL is highly reactive to volume — this strategy catches chain-specific momentum surges.",
            "direction": "LONG",
            "single_coin": "SOLUSDT",
            "leverage": 20,
            "position_size_pct": 4,
            "take_profit_pct": 2,
            "stop_loss_pct": 1.5,
            "take_profit2_pct": None,
            "trailing_stop": False,
            "max_trades_per_day": 5,
            "cooldown_minutes": 30,
            "daily_loss_limit_pct": 8,
            "max_open_positions": 1,
            "difficulty": "Beginner",
            "style": "Scalp",
        },
        # ── Session-based strategies ───────────────────────────────────────
        {
            "id": "london_power_fade",
            "name": "London Power Fade",
            "emoji": "🏛️",
            "category": "reversal",
            "tagline": "Short overextended morning pumps during London session",
            "description": "Active only during London + overlap session (07:00–16:00 UTC). Short when a coin pumps 5%+ in 30 minutes during the London open with RSI above 75. The London session frequently reverses morning Asia pumps as European traders take profit. Wait for the first 5m red candle before entry.",
            "direction": "SHORT",
            "leverage": 10,
            "position_size_pct": 5,
            "take_profit_pct": 3,
            "stop_loss_pct": 1.5,
            "take_profit2_pct": 5,
            "trailing_stop": False,
            "max_trades_per_day": 3,
            "cooldown_minutes": 60,
            "daily_loss_limit_pct": 5,
            "max_open_positions": 2,
            "difficulty": "Intermediate",
            "style": "Reversal",
            "sessions": ["london", "overlap"],
        },
        {
            "id": "asia_open_breakout",
            "name": "Asia Open Breakout",
            "emoji": "🌏",
            "category": "scalp",
            "tagline": "Catch the Asian session open momentum (00:00–08:00 UTC)",
            "description": "Active only during the Asian session (00:00–08:00 UTC). Enter long when price moves 1.5% upward in 30 minutes with volume 1.5× average. The Asian session often sets the directional bias for the day with low-resistance moves. Uses VWAP confirmation to avoid chasing overextended entries.",
            "direction": "LONG",
            "leverage": 12,
            "position_size_pct": 5,
            "take_profit_pct": 3,
            "stop_loss_pct": 2,
            "take_profit2_pct": None,
            "trailing_stop": False,
            "max_trades_per_day": 4,
            "cooldown_minutes": 45,
            "daily_loss_limit_pct": 6,
            "max_open_positions": 2,
            "difficulty": "Beginner",
            "style": "Scalp",
            "sessions": ["asian"],
        },
        # ── Advanced indicator strategies ──────────────────────────────────
        {
            "id": "supertrend_flip",
            "name": "SuperTrend Flip",
            "emoji": "🔄",
            "category": "momentum",
            "tagline": "Trade the exact moment SuperTrend flips direction on 1h",
            "description": "Enter when SuperTrend (10, 3.0) flips from bearish to bullish on the 1h chart. ADX must be above 25 to confirm the market is trending (not choppy). RSI must be above 50 at the time of flip. Closes when SuperTrend flips back or TP/SL is hit. Works on both sides — long on bullish flip, short on bearish flip.",
            "direction": "BOTH",
            "leverage": 12,
            "position_size_pct": 5,
            "take_profit_pct": 4,
            "stop_loss_pct": 2,
            "take_profit2_pct": 7,
            "trailing_stop": True,
            "trailing_stop_pct": 1.5,
            "max_trades_per_day": 4,
            "cooldown_minutes": 90,
            "daily_loss_limit_pct": 6,
            "max_open_positions": 2,
            "difficulty": "Intermediate",
            "style": "Momentum",
        },
        {
            "id": "stochrsi_exhaustion",
            "name": "StochRSI Exhaustion",
            "emoji": "🎚️",
            "category": "reversal",
            "tagline": "Buy when StochRSI crosses up from oversold after 3 red candles",
            "description": "Long when StochRSI K line crosses above the D line from below 20 (oversold) on the 15m chart. Requires 3 consecutive red candles immediately before the cross — the exhaustion pattern. Volume must be above average on the signal candle. This catches the exact moment selling pressure dries up.",
            "direction": "LONG",
            "leverage": 15,
            "position_size_pct": 5,
            "take_profit_pct": 3,
            "stop_loss_pct": 1.5,
            "take_profit2_pct": 5,
            "trailing_stop": False,
            "max_trades_per_day": 5,
            "cooldown_minutes": 45,
            "daily_loss_limit_pct": 6,
            "max_open_positions": 2,
            "difficulty": "Intermediate",
            "style": "Reversal",
        },
        {
            "id": "choch_structure_hunt",
            "name": "CHoCH Structure Hunt",
            "emoji": "📐",
            "category": "smc",
            "tagline": "Smart Money — enter the moment bearish structure flips bullish",
            "description": "Long when a Change of Character (CHoCH) bullish flip is detected on the 15m chart — price was making lower highs then broke above the previous swing high. This is the earliest possible SMC entry. RSI must be below 60 (not already overbought). Best combined with a higher-timeframe bullish bias.",
            "direction": "LONG",
            "leverage": 15,
            "position_size_pct": 5,
            "take_profit_pct": 4,
            "stop_loss_pct": 2,
            "take_profit2_pct": 7,
            "trailing_stop": False,
            "max_trades_per_day": 4,
            "cooldown_minutes": 60,
            "daily_loss_limit_pct": 6,
            "max_open_positions": 2,
            "difficulty": "Advanced",
            "style": "Smart Money",
        },
        # ── Exotic condition types ─────────────────────────────────────────
        {
            "id": "rsi_divergence_long",
            "name": "RSI Divergence Long",
            "emoji": "↔️",
            "category": "reversal",
            "tagline": "Classic hidden bullish divergence — price lower, RSI higher",
            "description": "Long when bullish RSI divergence is detected on the 15m chart: price makes a lower low but RSI makes a higher low. This classic setup signals momentum is shifting before price confirms. Volume must spike on the divergence candle. RSI must be below 45 for the setup to be valid.",
            "direction": "LONG",
            "leverage": 10,
            "position_size_pct": 6,
            "take_profit_pct": 5,
            "stop_loss_pct": 2.5,
            "take_profit2_pct": 8,
            "trailing_stop": False,
            "max_trades_per_day": 3,
            "cooldown_minutes": 90,
            "daily_loss_limit_pct": 5,
            "max_open_positions": 2,
            "difficulty": "Intermediate",
            "style": "Reversal",
        },
        {
            "id": "funding_rate_fade",
            "name": "Funding Rate Fade",
            "emoji": "💰",
            "category": "reversal",
            "tagline": "Short when funding is extreme — over-leveraged longs get squeezed",
            "description": "Short when the perpetual funding rate exceeds 0.05% (extremely positive, meaning too many longs). This signals the market is over-leveraged to one side and a flush is coming. Enter short when funding > 0.05% AND RSI is above 65. The funding squeeze creates sharp, fast moves. Exit quickly at first TP.",
            "direction": "SHORT",
            "leverage": 10,
            "position_size_pct": 5,
            "take_profit_pct": 3,
            "stop_loss_pct": 2,
            "take_profit2_pct": None,
            "trailing_stop": False,
            "max_trades_per_day": 3,
            "cooldown_minutes": 120,
            "daily_loss_limit_pct": 5,
            "max_open_positions": 2,
            "difficulty": "Advanced",
            "style": "Reversal",
        },
        {
            "id": "oi_surge_breakout",
            "name": "OI Surge Breakout",
            "emoji": "📡",
            "category": "momentum",
            "tagline": "Trade the breakout when open interest spikes — big money is moving",
            "description": "Enter when open interest increases 10%+ in 30 minutes combined with a volume spike 2× above average and a 2%+ price move. Rising OI with rising price confirms new longs are entering (not just short covers). Direction follows the price move — long if up, short if down.",
            "direction": "BOTH",
            "leverage": 12,
            "position_size_pct": 5,
            "take_profit_pct": 4,
            "stop_loss_pct": 2,
            "take_profit2_pct": 7,
            "trailing_stop": True,
            "trailing_stop_pct": 1.5,
            "max_trades_per_day": 4,
            "cooldown_minutes": 60,
            "daily_loss_limit_pct": 6,
            "max_open_positions": 2,
            "difficulty": "Advanced",
            "style": "Momentum",
        },
        {
            "id": "fibonacci_sniper",
            "name": "Fibonacci Sniper",
            "emoji": "🎯",
            "category": "smc",
            "tagline": "2 trades/day max — enter only at the golden ratio (0.618)",
            "description": "Sniper-mode strategy with max 2 trades per day. Enter long when price pulls back to the exact 0.618 Fibonacci retracement of the most recent swing move on the 4h chart. RSI must be between 35–55 (reset, not oversold). A bullish engulfing or hammer pattern on the 15m confirms the entry. Wide TP for full retracement capture.",
            "direction": "LONG",
            "leverage": 10,
            "position_size_pct": 6,
            "take_profit_pct": 5,
            "stop_loss_pct": 2.5,
            "take_profit2_pct": 9,
            "trailing_stop": False,
            "max_trades_per_day": 2,
            "cooldown_minutes": 180,
            "daily_loss_limit_pct": 5,
            "max_open_positions": 1,
            "difficulty": "Advanced",
            "style": "Sniper",
        },
        # ── BTC playbooks — popular setups traders actually use ────────────
        {
            "id": "btc_daily_trend_hold",
            "name": "BTC Daily Trend Hold",
            "emoji": "₿",
            "category": "swing",
            "tagline": "Hold BTC longs while the daily 50/200 SMA stays bullish",
            "description": "Bitcoin only, daily timeframe. Enters long when price closes above the 200-day SMA AND the 50-day SMA is above the 200-day SMA (golden-cross context). Adds a momentum filter — RSI between 50 and 70 to avoid blow-off tops. Designed for multi-day swings during BTC bull legs. Wide TP, wide SL.",
            "direction": "LONG",
            "single_coin": "BTCUSDT",
            "leverage": 5,
            "position_size_pct": 8,
            "take_profit_pct": 12,
            "stop_loss_pct": 5,
            "take_profit2_pct": 25,
            "trailing_stop": True,
            "trailing_stop_pct": 4,
            "max_trades_per_day": 1,
            "cooldown_minutes": 1440,
            "daily_loss_limit_pct": 6,
            "max_open_positions": 1,
            "difficulty": "Beginner",
            "style": "Swing",
        },
        {
            "id": "btc_supertrend_1h",
            "name": "BTC SuperTrend 1H Flip",
            "emoji": "🔄",
            "category": "momentum",
            "tagline": "BTC only — flip with the 1h SuperTrend, only when ADX is hot",
            "description": "Bitcoin only, 1h timeframe. Enters at the exact candle the SuperTrend (10, 3.0) flips bullish OR bearish. Requires ADX above 25 — no entries during chop. Trails the stop tight to lock in the trend's move. One of the simplest, most-used trend systems on BTC.",
            "direction": "BOTH",
            "single_coin": "BTCUSDT",
            "leverage": 10,
            "position_size_pct": 6,
            "take_profit_pct": 3,
            "stop_loss_pct": 1.5,
            "take_profit2_pct": 6,
            "trailing_stop": True,
            "trailing_stop_pct": 1.2,
            "max_trades_per_day": 4,
            "cooldown_minutes": 60,
            "daily_loss_limit_pct": 6,
            "max_open_positions": 1,
            "difficulty": "Intermediate",
            "style": "Momentum",
        },
        {
            "id": "btc_4h_mean_revert",
            "name": "BTC 4H Mean Reversion",
            "emoji": "🪃",
            "category": "reversal",
            "tagline": "BTC only — buy the dip when 4H RSI is washed out",
            "description": "Bitcoin only, 4h timeframe. Enters long when RSI dips below 32 AND price closes below the lower Bollinger Band (20, 2). Confirmed by a bullish engulfing candle to time the bounce. Tends to fire 1–3 times a month and catches the meatiest swing lows. Wider SL because reversals chop before reversing.",
            "direction": "LONG",
            "single_coin": "BTCUSDT",
            "leverage": 5,
            "position_size_pct": 8,
            "take_profit_pct": 5,
            "stop_loss_pct": 3,
            "take_profit2_pct": 9,
            "trailing_stop": False,
            "max_trades_per_day": 1,
            "cooldown_minutes": 720,
            "daily_loss_limit_pct": 5,
            "max_open_positions": 1,
            "difficulty": "Intermediate",
            "style": "Reversal",
        },
        {
            "id": "btc_liquidity_sweep",
            "name": "BTC Liquidity Sweep + Reclaim",
            "emoji": "💧",
            "category": "smc",
            "tagline": "BTC only — fade the stop hunt, ride the reclaim",
            "description": "Bitcoin only, 15m timeframe. Enters long after a Break of Structure bullish (BoS) — meaning price wicked below a recent swing low and closed back above it (classic liquidity sweep). Confirmed by a bullish engulfing on the reclaim candle. RSI must be below 55 so we're not chasing strength.",
            "direction": "LONG",
            "single_coin": "BTCUSDT",
            "leverage": 12,
            "position_size_pct": 5,
            "take_profit_pct": 2.5,
            "stop_loss_pct": 1.2,
            "take_profit2_pct": 4.5,
            "trailing_stop": False,
            "max_trades_per_day": 4,
            "cooldown_minutes": 60,
            "daily_loss_limit_pct": 5,
            "max_open_positions": 1,
            "difficulty": "Advanced",
            "style": "Smart Money",
        },
        {
            "id": "btc_asia_breakout",
            "name": "BTC Asia Range Breakout",
            "emoji": "🌏",
            "category": "scalp",
            "tagline": "BTC only — long the break of the Asia session range",
            "description": "Bitcoin only, 5m timeframe. Active during the Asia session (00:00–08:00 UTC). Enters long when BTC breaks above its rolling 30-min high with volume 1.5× average AND VWAP confirms (price above VWAP). The Asian range break often sets the day's directional bias.",
            "direction": "LONG",
            "single_coin": "BTCUSDT",
            "leverage": 12,
            "position_size_pct": 5,
            "take_profit_pct": 1.8,
            "stop_loss_pct": 1.0,
            "take_profit2_pct": 3.0,
            "trailing_stop": False,
            "max_trades_per_day": 3,
            "cooldown_minutes": 60,
            "daily_loss_limit_pct": 5,
            "max_open_positions": 1,
            "difficulty": "Beginner",
            "style": "Scalp",
            "sessions": ["asian"],
        },
        {
            "id": "btc_squeeze_pop",
            "name": "BTC Squeeze Pop",
            "emoji": "🎯",
            "category": "breakout",
            "tagline": "BTC only — Bollinger squeeze inside Keltner, then trade the fire",
            "description": "Bitcoin only, 1h timeframe. Enters when the Bollinger Band squeeze fires (LazyBear's classic squeeze) — Bollinger Bands had been contracted inside the Keltner Channel and now release. Direction is whichever side breaks first (long if price closes above the upper Keltner). ADX above 22 confirms the move has legs.",
            "direction": "LONG",
            "single_coin": "BTCUSDT",
            "leverage": 10,
            "position_size_pct": 6,
            "take_profit_pct": 3.5,
            "stop_loss_pct": 1.8,
            "take_profit2_pct": 6,
            "trailing_stop": True,
            "trailing_stop_pct": 1.5,
            "max_trades_per_day": 2,
            "cooldown_minutes": 240,
            "daily_loss_limit_pct": 5,
            "max_open_positions": 1,
            "difficulty": "Intermediate",
            "style": "Breakout",
        },
        {
            "id": "btc_vwap_reclaim",
            "name": "BTC VWAP Reclaim Scalp",
            "emoji": "📍",
            "category": "scalp",
            "tagline": "BTC only — long the moment price reclaims the daily VWAP",
            "description": "Bitcoin only, 5m timeframe. Enters long when price closes back above the daily VWAP after a dip below it, confirmed by a bullish candle and volume above average. Institutional flow pivot — when VWAP is reclaimed it tends to act as support for the rest of the session. Scalp-style, tight SL.",
            "direction": "LONG",
            "single_coin": "BTCUSDT",
            "leverage": 15,
            "position_size_pct": 5,
            "take_profit_pct": 1.2,
            "stop_loss_pct": 0.7,
            "take_profit2_pct": 2.0,
            "trailing_stop": False,
            "max_trades_per_day": 6,
            "cooldown_minutes": 30,
            "daily_loss_limit_pct": 4,
            "max_open_positions": 1,
            "difficulty": "Beginner",
            "style": "Scalp",
        },
        {
            "id": "btc_rsi_div_4h",
            "name": "BTC RSI Bullish Divergence (4H)",
            "emoji": "↗️",
            "category": "reversal",
            "tagline": "BTC only — catch swing lows when 4H RSI prints higher low vs price",
            "description": "Bitcoin only, 4h timeframe. Enters long when bullish RSI divergence appears: BTC prints a lower low but RSI prints a higher low. RSI must be under 45 at the divergence candle. Confirmed by a bullish engulfing or hammer to time entry. Catches the highest-quality reversals — they're rare but they pay.",
            "direction": "LONG",
            "single_coin": "BTCUSDT",
            "leverage": 6,
            "position_size_pct": 8,
            "take_profit_pct": 6,
            "stop_loss_pct": 3,
            "take_profit2_pct": 12,
            "trailing_stop": True,
            "trailing_stop_pct": 2.5,
            "max_trades_per_day": 1,
            "cooldown_minutes": 720,
            "daily_loss_limit_pct": 5,
            "max_open_positions": 1,
            "difficulty": "Advanced",
            "style": "Reversal",
        },
        # ── ETH playbooks ──────────────────────────────────────────────────
        {
            "id": "eth_4h_trend_continuation",
            "name": "ETH 4H Trend Continuation",
            "emoji": "Ξ",
            "category": "swing",
            "tagline": "ETH only — pyramid into the 4H uptrend on EMA pullbacks",
            "description": "Ethereum only, 4h timeframe. Enters long when EMA 8/21/50/100 are aligned bullish AND price pulls back to touch the EMA 21. RSI must be 45–62 (healthy reset, not overbought). MACD histogram rising confirms momentum is returning. Designed to capture multi-day trend legs.",
            "direction": "LONG",
            "single_coin": "ETHUSDT",
            "leverage": 8,
            "position_size_pct": 7,
            "take_profit_pct": 7,
            "stop_loss_pct": 3,
            "take_profit2_pct": 13,
            "trailing_stop": True,
            "trailing_stop_pct": 2,
            "max_trades_per_day": 1,
            "cooldown_minutes": 480,
            "daily_loss_limit_pct": 5,
            "max_open_positions": 1,
            "difficulty": "Intermediate",
            "style": "Swing",
        },
        {
            "id": "eth_macd_cross_15m",
            "name": "ETH MACD Cross + Volume",
            "emoji": "📊",
            "category": "momentum",
            "tagline": "ETH only — 15m MACD bullish cross confirmed by volume",
            "description": "Ethereum only, 15m timeframe. Enters long on the candle a MACD (12, 26, 9) bullish cross prints, when volume is 1.4× the 20-period average and EMA 21 is above EMA 50. RSI must be between 50 and 65 — momentum confirmed but not overbought. ETH responds cleanly to MACD crosses on liquid timeframes.",
            "direction": "LONG",
            "single_coin": "ETHUSDT",
            "leverage": 10,
            "position_size_pct": 6,
            "take_profit_pct": 2.5,
            "stop_loss_pct": 1.2,
            "take_profit2_pct": 4.5,
            "trailing_stop": False,
            "max_trades_per_day": 4,
            "cooldown_minutes": 60,
            "daily_loss_limit_pct": 5,
            "max_open_positions": 1,
            "difficulty": "Beginner",
            "style": "Momentum",
        },
        {
            "id": "eth_supertrend_1h",
            "name": "ETH SuperTrend 1H Flip",
            "emoji": "🔄",
            "category": "momentum",
            "tagline": "ETH only — flip with the 1h SuperTrend when ADX confirms",
            "description": "Ethereum only, 1h timeframe. Mirrors the BTC SuperTrend system but tuned for ETH's higher volatility. Enters at the SuperTrend (10, 3.0) flip with ADX above 22. Trails the stop to ride extended legs. Works long and short — fires more often than the BTC version because ETH trends harder.",
            "direction": "BOTH",
            "single_coin": "ETHUSDT",
            "leverage": 10,
            "position_size_pct": 6,
            "take_profit_pct": 3.5,
            "stop_loss_pct": 1.8,
            "take_profit2_pct": 6,
            "trailing_stop": True,
            "trailing_stop_pct": 1.5,
            "max_trades_per_day": 4,
            "cooldown_minutes": 60,
            "daily_loss_limit_pct": 6,
            "max_open_positions": 1,
            "difficulty": "Intermediate",
            "style": "Momentum",
        },
    ]
    return JSONResponse(templates)


def _mode_analytics_slice(closed: list) -> dict:
    """Aggregate stats for a paper or live subset of closed executions."""
    if not closed:
        return {"trades": 0, "wins": 0, "win_rate": None, "total_pnl_pct": 0.0, "total_pnl_usd": 0.0}
    decisive = [e for e in closed if e.outcome in ("WIN", "LOSS")]
    wins = sum(1 for e in decisive if e.outcome == "WIN")
    return {
        "trades": len(closed),
        "wins": wins,
        "win_rate": round(wins / len(decisive) * 100, 1) if decisive else None,
        "total_pnl_pct": round(sum(e.pnl_pct or 0 for e in closed), 2),
        "total_pnl_usd": round(sum(float(e.pnl_usd or 0) for e in closed), 2),
    }


def _equity_and_drawdown(closed: list) -> tuple:
    """Build cumulative equity + drawdown series from closed executions."""
    cumulative = 0.0
    equity_labels, equity_values = [], []
    drawdown_labels, drawdown_values = [], []
    peak = 0.0
    for e in closed:
        cumulative += (e.pnl_pct or 0)
        dt = e.closed_at or e.fired_at
        label = dt.strftime("%m/%d") if dt else ""
        equity_labels.append(label)
        equity_values.append(round(cumulative, 2))
        peak = max(peak, cumulative)
        drawdown_labels.append(label)
        drawdown_values.append(round(cumulative - peak, 2))
    max_dd = round(abs(min(drawdown_values)), 2) if drawdown_values else 0.0
    return equity_labels, equity_values, drawdown_labels, drawdown_values, max_dd


def _compute_execution_analytics(execs: list, perf=None) -> dict:
    """Rich analytics payload from StrategyExecution rows (oldest → newest)."""
    closed = [e for e in execs if e.outcome in ("WIN", "LOSS", "BREAKEVEN") and e.pnl_pct is not None]
    wins   = [e.pnl_pct for e in closed if e.outcome == "WIN"]
    losses = [e.pnl_pct for e in closed if e.outcome == "LOSS"]

    eq_labels, eq_values, dd_labels, dd_values, max_dd = _equity_and_drawdown(closed)

    gross_win  = sum(wins) if wins else 0.0
    gross_loss = abs(sum(losses)) if losses else 0.0
    profit_factor = round(gross_win / gross_loss, 2) if gross_loss > 0 else (99.0 if gross_win > 0 else 0.0)

    sharpe = 0.0
    if len(closed) >= 5:
        import statistics as _st
        pnls = [e.pnl_pct for e in closed]
        mean_r = _st.mean(pnls)
        std_r  = _st.stdev(pnls) if len(pnls) > 1 else 0
        sharpe = round((mean_r / std_r) * (252 ** 0.5), 2) if std_r > 0 else 0

    best_streak = worst_streak = cur_w = cur_l = 0
    for e in closed:
        if e.outcome == "WIN":
            cur_w += 1
            cur_l = 0
            best_streak = max(best_streak, cur_w)
        elif e.outcome == "LOSS":
            cur_l += 1
            cur_w = 0
            worst_streak = max(worst_streak, cur_l)

    hold_mins = []
    for e in closed:
        if e.closed_at and e.fired_at:
            mins = (e.closed_at - e.fired_at).total_seconds() / 60.0
            if mins >= 0:
                hold_mins.append(mins)
    avg_hold = round(sum(hold_mins) / len(hold_mins), 1) if hold_mins else None
    median_hold = None
    if hold_mins:
        import statistics as _st2
        median_hold = round(_st2.median(hold_mins), 1)

    paper_closed = [e for e in closed if e.is_paper]
    live_closed  = [e for e in closed if not e.is_paper]

    def _curve_for(subset):
        if len(subset) < 2:
            return {"labels": [], "values": []}
        _l, _v, _, _, _ = _equity_and_drawdown(subset)
        return {"labels": _l, "values": _v}

    coin_pnl, coin_trades = {}, {}
    for e in closed:
        coin_pnl[e.symbol] = round(coin_pnl.get(e.symbol, 0) + (e.pnl_pct or 0), 2)
        coin_trades[e.symbol] = coin_trades.get(e.symbol, 0) + 1
    top_coins = sorted(coin_pnl.items(), key=lambda x: -x[1])

    long_dec  = [e for e in closed if e.direction == "LONG"  and e.outcome in ("WIN", "LOSS")]
    short_dec = [e for e in closed if e.direction == "SHORT" and e.outcome in ("WIN", "LOSS")]
    long_wr  = round(len([e for e in long_dec  if e.outcome == "WIN"]) / len(long_dec)  * 100, 1) if long_dec  else None
    short_wr = round(len([e for e in short_dec if e.outcome == "WIN"]) / len(short_dec) * 100, 1) if short_dec else None

    wr_pct = perf.win_rate if perf else 0
    health = 0.0
    if len(closed) >= 3:
        health += min(wr_pct / 100, 1.0) * 4.0
        health += min(profit_factor / 2.0, 1.0) * 3.0
        health += min(max(sharpe, 0) / 2.0, 1.0) * 2.0
        health += min(len(closed) / 30.0, 1.0) * 1.0

    return {
        "equity_curve":       {"labels": eq_labels, "values": eq_values},
        "drawdown_curve":     {"labels": dd_labels, "values": dd_values},
        "equity_curve_paper": _curve_for(paper_closed),
        "equity_curve_live":  _curve_for(live_closed),
        "paper_vs_live": {
            "paper": _mode_analytics_slice(paper_closed),
            "live":  _mode_analytics_slice(live_closed),
        },
        "profit_factor":      profit_factor,
        "max_drawdown":       max_dd,
        "current_drawdown":   dd_values[-1] if dd_values else 0.0,
        "sharpe_ratio":       sharpe,
        "best_streak":        best_streak,
        "worst_streak":       worst_streak,
        "avg_hold_minutes":   avg_hold,
        "median_hold_minutes": median_hold,
        "avg_win_pct":        round(sum(wins) / len(wins), 2) if wins else 0,
        "avg_loss_pct":       round(sum(losses) / len(losses), 2) if losses else 0,
        "long_win_rate":      long_wr,
        "short_win_rate":     short_wr,
        "coin_breakdown":     [{"symbol": s, "pnl": p, "trades": coin_trades[s]} for s, p in top_coins[:10]],
        "health_score":       round(health, 1),
        "total_closed":       len(closed),
        "total_pnl_usd":      round(sum(float(e.pnl_usd or 0) for e in closed), 2),
    }


@app.get("/api/strategies/{strategy_id}/analytics")
def api_strategy_analytics(strategy_id: int, uid: str = Query(...)):
    """Advanced analytics: Sharpe, drawdown, profit factor, equity curve, health score."""
    _an_key = f"analytics:{uid}:{strategy_id}"
    _an_hit = get_cache(_an_key)
    if _an_hit is not None:
        return _cached_json(_an_hit, True, 300)

    from app.database import SessionLocal
    db = SessionLocal()
    try:
        user = _get_user_by_uid(uid, db)
        if not user:
            raise HTTPException(status_code=403)

        from app.strategy_models import UserStrategy, StrategyExecution, StrategyPerformance
        s = db.query(UserStrategy).filter(
            UserStrategy.id == strategy_id, UserStrategy.user_id == user.id
        ).first()
        if not s:
            raise HTTPException(status_code=404)

        execs = (
            db.query(StrategyExecution)
            .filter(StrategyExecution.strategy_id == strategy_id)
            .order_by(StrategyExecution.fired_at.asc())
            .all()
        )
        perf = db.query(StrategyPerformance).filter(StrategyPerformance.strategy_id == strategy_id).first()
        _an_payload = _compute_execution_analytics(execs, perf)
        set_cache(_an_key, _an_payload, 300)
        return _cached_json(_an_payload, False, 300)
    finally:
        db.close()


@app.get("/api/portfolio/analytics")
def api_portfolio_analytics(uid: str = Query(...)):
    """Portfolio-wide performance analytics across all strategies."""
    _key = f"portfolio_analytics:{uid}"
    hit = get_cache(_key)
    if hit is not None:
        return _cached_json(hit, True, 300)

    from app.database import SessionLocal
    from app.strategy_models import StrategyExecution, UserStrategy
    db = SessionLocal()
    try:
        user = _get_user_by_uid(uid, db)
        if not user:
            raise HTTPException(status_code=403)
        execs = (
            db.query(StrategyExecution)
            .join(UserStrategy, UserStrategy.id == StrategyExecution.strategy_id)
            .filter(UserStrategy.user_id == user.id)
            .order_by(StrategyExecution.fired_at.asc())
            .all()
        )
        payload = _compute_execution_analytics(execs)
        set_cache(_key, payload, 300)
        return _cached_json(payload, False, 300)
    finally:
        db.close()


def _tax_csv_rows(execs, strategy_name: str = "", year: Optional[int] = None):
    """Build tax-report CSV rows from execution objects."""
    import csv, io
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow([
        "opened_at", "closed_at", "strategy", "symbol", "asset_class", "direction",
        "mode", "outcome", "pnl_pct", "pnl_usd", "pips_pnl", "hold_minutes",
    ])
    for e in execs:
        if e.outcome not in ("WIN", "LOSS", "BREAKEVEN"):
            continue
        closed_at = e.closed_at or e.fired_at
        if year and closed_at and closed_at.year != year:
            continue
        hold_m = ""
        if e.closed_at and e.fired_at:
            hold_m = round((e.closed_at - e.fired_at).total_seconds() / 60.0, 1)
        w.writerow([
            e.fired_at.strftime("%Y-%m-%d %H:%M") if e.fired_at else "",
            e.closed_at.strftime("%Y-%m-%d %H:%M") if e.closed_at else "",
            strategy_name,
            e.symbol,
            getattr(e, "asset_class", None) or "crypto",
            e.direction,
            "paper" if e.is_paper else "live",
            e.outcome,
            e.pnl_pct if e.pnl_pct is not None else "",
            e.pnl_usd if e.pnl_usd is not None else "",
            e.pips_pnl if getattr(e, "pips_pnl", None) is not None else "",
            hold_m,
        ])
    buf.seek(0)
    return buf.getvalue()


@app.get("/api/strategies/{strategy_id}/export/tax")
async def api_export_strategy_tax(
    strategy_id: int,
    uid: str = Query(...),
    year: int = Query(None),
):
    """Annual tax summary CSV for one strategy."""
    from fastapi.responses import StreamingResponse
    from app.database import SessionLocal
    from app.strategy_models import UserStrategy, StrategyExecution
    db = SessionLocal()
    try:
        user = _get_user_by_uid(uid, db)
        if not user:
            raise HTTPException(status_code=403)
        s = db.query(UserStrategy).filter(
            UserStrategy.id == strategy_id, UserStrategy.user_id == user.id
        ).first()
        if not s:
            raise HTTPException(status_code=404)
        execs = (
            db.query(StrategyExecution)
            .filter(StrategyExecution.strategy_id == strategy_id)
            .order_by(StrategyExecution.fired_at.asc())
            .all()
        )
        yr = year or datetime.utcnow().year
        body = _tax_csv_rows(execs, strategy_name=s.name or f"strategy_{strategy_id}", year=yr)
        fname = f"tax_{strategy_id}_{yr}.csv"
        return StreamingResponse(
            iter([body]),
            media_type="text/csv",
            headers={"Content-Disposition": f'attachment; filename="{fname}"'},
        )
    finally:
        db.close()


@app.get("/api/portfolio/export/tax")
async def api_export_portfolio_tax(
    uid: str = Query(...),
    year: int = Query(None),
):
    """Annual tax summary CSV across all strategies for the user."""
    from fastapi.responses import StreamingResponse
    from app.database import SessionLocal
    from app.strategy_models import StrategyExecution, UserStrategy
    db = SessionLocal()
    try:
        user = _get_user_by_uid(uid, db)
        if not user:
            raise HTTPException(status_code=403)
        rows = (
            db.query(StrategyExecution, UserStrategy.name)
            .join(UserStrategy, UserStrategy.id == StrategyExecution.strategy_id)
            .filter(UserStrategy.user_id == user.id)
            .order_by(StrategyExecution.fired_at.asc())
            .all()
        )
        yr = year or datetime.utcnow().year
        import csv, io
        buf = io.StringIO()
        w = csv.writer(buf)
        w.writerow([
            "opened_at", "closed_at", "strategy", "symbol", "asset_class", "direction",
            "mode", "outcome", "pnl_pct", "pnl_usd", "pips_pnl", "hold_minutes",
        ])
        for e, strat_name in rows:
            if e.outcome not in ("WIN", "LOSS", "BREAKEVEN"):
                continue
            closed_at = e.closed_at or e.fired_at
            if closed_at and closed_at.year != yr:
                continue
            hold_m = ""
            if e.closed_at and e.fired_at:
                hold_m = round((e.closed_at - e.fired_at).total_seconds() / 60.0, 1)
            w.writerow([
                e.fired_at.strftime("%Y-%m-%d %H:%M") if e.fired_at else "",
                e.closed_at.strftime("%Y-%m-%d %H:%M") if e.closed_at else "",
                strat_name or "",
                e.symbol, getattr(e, "asset_class", None) or "crypto",
                e.direction, "paper" if e.is_paper else "live", e.outcome,
                e.pnl_pct if e.pnl_pct is not None else "",
                e.pnl_usd if e.pnl_usd is not None else "",
                e.pips_pnl if getattr(e, "pips_pnl", None) is not None else "",
                hold_m,
            ])
        buf.seek(0)
        fname = f"tradehub_tax_{user.uid or uid}_{yr}.csv"
        return StreamingResponse(
            iter([buf.getvalue()]),
            media_type="text/csv",
            headers={"Content-Disposition": f'attachment; filename="{fname}"'},
        )
    finally:
        db.close()


@app.get("/api/strategies/{strategy_id}/trades")
async def api_strategy_trades(strategy_id: int, uid: str = Query(...)):
    """Return all trade executions for a strategy (newest first)."""
    from app.database import SessionLocal
    db = SessionLocal()
    try:
        user = _get_user_by_uid(uid, db)
        if not user:
            raise HTTPException(status_code=403)
        from app.strategy_models import UserStrategy, StrategyExecution
        s = db.query(UserStrategy).filter(
            UserStrategy.id == strategy_id, UserStrategy.user_id == user.id
        ).first()
        if not s:
            raise HTTPException(status_code=404)
        execs = (
            db.query(StrategyExecution)
            .filter(StrategyExecution.strategy_id == strategy_id)
            .order_by(StrategyExecution.fired_at.desc())
            .limit(200)
            .all()
        )

        # Fetch live prices for open positions
        open_symbols = list({e.symbol for e in execs if e.outcome == "OPEN"})
        live_prices: dict = {}
        if open_symbols:
            try:
                async with httpx.AsyncClient() as hc:
                    from app.services.strategy_executor import _fetch_live_price_batch
                    live_prices = await _fetch_live_price_batch(open_symbols, hc)
            except Exception as _lpe:
                logger.debug(f"live price fetch skipped: {_lpe}")

        trades = []
        for e in execs:
            dur = None
            if e.fired_at and e.closed_at:
                dur = int((e.closed_at - e.fired_at).total_seconds() / 60)

            live_px = live_prices.get(e.symbol) if e.outcome == "OPEN" else None
            unrealised = None
            if live_px and e.entry_price and e.outcome == "OPEN":
                lev = e.leverage or 10
                if e.direction == "LONG":
                    unrealised = round((live_px - e.entry_price) / e.entry_price * 100 * lev, 2)
                else:
                    unrealised = round((e.entry_price - live_px) / e.entry_price * 100 * lev, 2)

            trades.append({
                "id":             e.id,
                "symbol":         e.symbol,
                "direction":      e.direction,
                "entry_price":    e.entry_price,
                "exit_price":     e.exit_price,
                "tp_price":       e.tp_price,
                "sl_price":       e.sl_price,
                "leverage":       e.leverage,
                "outcome":        e.outcome,
                "pnl_pct":        e.pnl_pct,
                "is_paper":       e.is_paper,
                "fired_at":       e.fired_at.isoformat() if e.fired_at else None,
                "closed_at":      e.closed_at.isoformat() if e.closed_at else None,
                "duration_mins":  dur,
                "conditions_met": e.conditions_met,
                "notes":          e.notes,
                "live_price":     live_px,
                "unrealised_pnl": unrealised,
            })
        return JSONResponse({"trades": trades, "total": len(trades)})
    finally:
        db.close()


@app.get("/api/portfolio/trades")
async def api_portfolio_trades(
    uid: str = Query(...),
    limit: int = Query(200, ge=1, le=500),
    offset: int = Query(0, ge=0),
    filter: str = Query("all"),  # all | live | paper | wins | losses | open
    live: bool = Query(False),   # fetch mark prices for OPEN rows (slower)
):
    """Aggregate trade executions across ALL of the caller's strategies.

    Used by the mobile "Trades" tab. Newest first. Each row carries the
    parent strategy's id + name so the client can deep-link to the strategy
    detail screen without an extra round-trip.
    """
    _ptrades_key = f"portfolio_trades:{uid}:{filter}:{offset}:{limit}:{int(live)}"
    _ptrades_cached = _CACHE.get(_ptrades_key)
    if _ptrades_cached and time.time() < _ptrades_cached[1]:
        return JSONResponse(_ptrades_cached[0])

    from app.database import SessionLocal
    from app.strategy_models import UserStrategy, StrategyExecution

    def _load_rows_sync():
        db = SessionLocal()
        try:
            from sqlalchemy import text as _t
            db.execute(_t("SET LOCAL statement_timeout = '10000'"))
            user = _get_user_by_uid_safe(uid, db)
            if not user:
                return None

            q = (
                db.query(StrategyExecution, UserStrategy.name, UserStrategy.id, UserStrategy.asset_class)
                .join(UserStrategy, StrategyExecution.strategy_id == UserStrategy.id)
                .filter(UserStrategy.user_id == user.id)
            )

            f = (filter or "all").lower()
            if f == "live":
                q = q.filter(StrategyExecution.is_paper == False)  # noqa: E712
            elif f == "paper":
                q = q.filter(StrategyExecution.is_paper == True)   # noqa: E712
            elif f == "wins":
                q = q.filter(StrategyExecution.outcome == "WIN")
            elif f == "losses":
                q = q.filter(StrategyExecution.outcome == "LOSS")
            elif f == "open":
                q = q.filter(StrategyExecution.outcome == "OPEN")

            q = q.order_by(
                func.coalesce(StrategyExecution.closed_at, StrategyExecution.fired_at).desc()
            )
            # Skip q.count() — full-table count was blocking the worker 30s+ under load.
            probe = q.offset(offset).limit(limit + 1).all()
            has_more = len(probe) > limit
            rows = probe[:limit]

            parsed = []
            for e, sname, sid, sasset in rows:
                dur = None
                if e.fired_at and e.closed_at:
                    dur = int((e.closed_at - e.fired_at).total_seconds() / 60)
                parsed.append({
                    "execution": e,
                    "strategy_name": sname,
                    "strategy_id": sid,
                    "asset_class": sasset or "crypto",
                    "duration_mins": dur,
                })
            return {"rows": parsed, "filter": f, "has_more": has_more}
        finally:
            db.close()

    try:
        loaded = await asyncio.wait_for(asyncio.to_thread(_load_rows_sync), timeout=25.0)
    except asyncio.TimeoutError:
        stale = _CACHE.get(_ptrades_key)
        if stale:
            return JSONResponse(stale[0])
        raise HTTPException(status_code=503, detail="Trades are taking too long to load — please retry.")
    if loaded is None:
        raise HTTPException(status_code=403, detail="Invalid UID")

    rows = loaded["rows"]
    f = loaded["filter"]
    has_more = loaded["has_more"]

    open_symbols = list({
        item["execution"].symbol
        for item in rows
        if item["execution"].outcome == "OPEN"
    })
    live_prices: dict = {}
    if live and open_symbols:
        async def _quick_marks() -> dict:
            out: dict = {}
            from app.services.asset_classes import get_symbol as _ac_get
            tradfi_syms = [s for s in open_symbols
                           if any(_ac_get(c, s) for c in ("stock", "forex", "index"))]
            crypto_syms = [s for s in open_symbols if s not in set(tradfi_syms)]
            if crypto_syms:
                try:
                    from app.services.price_cache import get_multiple_cached_prices
                    out.update(await get_multiple_cached_prices(crypto_syms[:20]))
                except Exception:
                    pass
            if tradfi_syms:
                try:
                    from app.services.strategy_executor import _fetch_live_price_batch_tradfi
                    out.update(await _fetch_live_price_batch_tradfi(tradfi_syms[:12]))
                except Exception:
                    pass
            missing = [s for s in open_symbols if s not in out][:8]
            if missing:
                try:
                    from app.services.ctrader_price_feed import get_price as _ctf_px
                    for sym in missing:
                        px = _ctf_px(sym)
                        if px and px > 0:
                            out[sym] = px
                except Exception:
                    pass
            return out

        try:
            live_prices = await asyncio.wait_for(_quick_marks(), timeout=2.5)
        except asyncio.TimeoutError:
            logger.debug("[portfolio/trades] live marks timed out — skipped")
        except Exception as _lpe:
            logger.debug(f"live price fetch skipped: {_lpe}")

    out = []
    for item in rows:
        e = item["execution"]
        live_px = live_prices.get(e.symbol) if e.outcome == "OPEN" else None
        unrealised = None
        if live_px and e.entry_price and e.outcome == "OPEN":
            lev = e.leverage or 10
            if e.direction == "LONG":
                unrealised = round((live_px - e.entry_price) / e.entry_price * 100 * lev, 2)
            else:
                unrealised = round((e.entry_price - live_px) / e.entry_price * 100 * lev, 2)

        out.append({
            "id":             e.id,
            "strategy_id":    item["strategy_id"],
            "strategy_name":  item["strategy_name"],
            "asset_class":    item["asset_class"],
            "symbol":         e.symbol,
            "direction":      e.direction,
            "entry_price":    e.entry_price,
            "exit_price":     e.exit_price,
            "leverage":       e.leverage,
            "outcome":        e.outcome,
            "pnl_pct":        e.pnl_pct,
            "is_paper":       e.is_paper,
            "fired_at":       e.fired_at.isoformat()  if e.fired_at  else None,
            "closed_at":      e.closed_at.isoformat() if e.closed_at else None,
            "duration_mins":  item["duration_mins"],
            "live_price":     live_px,
            "unrealised_pnl": unrealised,
        })

    _resp = {
        "trades":   out,
        "total":    offset + len(out) + (1 if has_more else 0),
        "offset":   offset,
        "limit":    limit,
        "filter":   f,
        "has_more": has_more,
    }
    _CACHE[_ptrades_key] = (_resp, time.time() + 60)
    return JSONResponse(_resp)


@app.get("/api/strategies/{strategy_id}/export")
async def api_export_trades(strategy_id: int, uid: str = Query(...)):
    """Download all strategy trades as CSV."""
    import csv, io
    from fastapi.responses import StreamingResponse
    from app.database import SessionLocal
    db = SessionLocal()
    try:
        user = _get_user_by_uid(uid, db)
        if not user:
            raise HTTPException(status_code=403)
        from app.strategy_models import UserStrategy, StrategyExecution
        s = db.query(UserStrategy).filter(
            UserStrategy.id == strategy_id, UserStrategy.user_id == user.id
        ).first()
        if not s:
            raise HTTPException(status_code=404)
        execs = (
            db.query(StrategyExecution)
            .filter(StrategyExecution.strategy_id == strategy_id)
            .order_by(StrategyExecution.fired_at.desc())
            .all()
        )
        buf = io.StringIO()
        w   = csv.writer(buf)
        w.writerow(["date", "symbol", "direction", "leverage", "entry_price", "exit_price", "outcome", "pnl_pct", "pnl_usd"])
        for e in execs:
            w.writerow([
                (e.fired_at.strftime("%Y-%m-%d %H:%M") if e.fired_at else ""),
                e.symbol, e.direction, e.leverage,
                e.entry_price or "", e.exit_price or "",
                e.outcome, e.pnl_pct or "", e.pnl_usd or "",
            ])
        buf.seek(0)
        filename = f"strategy_{strategy_id}_{s.name.replace(' ','_')}.csv"
        return StreamingResponse(
            iter([buf.getvalue()]),
            media_type="text/csv",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )
    finally:
        db.close()


@app.get("/api/portfolio")
async def api_portfolio(uid: str = Query(...)):
    """Portfolio-level metrics across all strategies."""
    cache_key = f"portfolio_{uid}"
    cached = _CACHE.get(cache_key)
    if cached and time.time() < cached[1]:
        # Cache the dict, not the Response — a JSONResponse body is consumed by
        # the auth middleware on first send, so re-returning the same object
        # yields an empty body (ERR_CONTENT_LENGTH_MISMATCH). Rebuild each time.
        return _cached_json(cached[0], True, 60)

    from app.database import SessionLocal
    from sqlalchemy import text
    from datetime import datetime, timedelta
    from collections import defaultdict

    # All DB-heavy work runs in a worker thread bounded by wait_for, so a slow
    # query under DB pressure can never pin this request's event loop (which
    # would stall every other portal API call served by the same worker).
    def _load_portfolio_sync():
        db = SessionLocal()
        try:
            db.execute(text("SET LOCAL statement_timeout = '8000'"))
            user = _get_user_by_uid(uid, db)
            if not user:
                raise HTTPException(status_code=403)

            cutoff_30d = datetime.utcnow() - timedelta(days=30)
            cutoff_7d  = datetime.utcnow() - timedelta(days=7)

            # Single query: strategy counts
            strat_row = db.execute(text("""
                SELECT
                    COUNT(*) AS total,
                    COUNT(*) FILTER (WHERE status = 'active') AS active_count
                FROM user_strategies WHERE user_id = :uid
            """), {"uid": user.id}).fetchone()
            total_strategies = strat_row.total if strat_row else 0
            active_count     = strat_row.active_count if strat_row else 0

            cutoff_today = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)

            # Single query: executions via JOIN — OPEN trades unbounded,
            # closed trades capped at 90 days to prevent full-table scan.
            cutoff_90d = datetime.utcnow() - timedelta(days=90)
            exec_rows = db.execute(text("""
                SELECT e.outcome, e.pnl_pct, e.is_paper,
                       COALESCE(e.closed_at, e.fired_at) AS ts
                FROM strategy_executions e
                JOIN user_strategies s ON s.id = e.strategy_id
                WHERE s.user_id = :uid
                  AND (e.outcome = 'OPEN' OR COALESCE(e.closed_at, e.fired_at) >= :cutoff)
            """), {"uid": user.id, "cutoff": cutoff_90d}).fetchall()

            open_trades  = sum(1 for r in exec_rows if r.outcome == "OPEN")
            live_open    = sum(1 for r in exec_rows if r.outcome == "OPEN" and not r.is_paper)
            paper_open   = sum(1 for r in exec_rows if r.outcome == "OPEN" and r.is_paper)
            closed = [(r.pnl_pct, r.ts, r.outcome, bool(r.is_paper)) for r in exec_rows
                      if r.outcome in ("WIN", "LOSS", "BREAKEVEN") and r.pnl_pct is not None]
            total = len(closed)
            wins  = sum(1 for _, _, o, _ in closed if o == "WIN")
            # Win rate denominator excludes BREAKEVEN — neutral shouldn't drag it down.
            decisive = sum(1 for _, _, o, _ in closed if o in ("WIN", "LOSS"))

            closed_30d = [(p, ts, o, ip) for p, ts, o, ip in closed if ts and ts > cutoff_30d]

            pnl_today    = sum(p for p, ts, _, _  in closed if ts and ts > cutoff_today)
            pnl_7d       = sum(p for p, ts, _, _  in closed if ts and ts > cutoff_7d)
            pnl_30d      = sum(p for p, _, _, _   in closed_30d)
            pnl_all      = sum(p for p, _, _, _   in closed)
            live_pnl_30d  = sum(p for p, _, _, ip in closed_30d if not ip)
            paper_pnl_30d = sum(p for p, _, _, ip in closed_30d if ip)
            live_closed_30d  = sum(1 for _, _, _, ip in closed_30d if not ip)
            paper_closed_30d = sum(1 for _, _, _, ip in closed_30d if ip)

            daily = defaultdict(float)
            for pnl, ts, _, _ in closed_30d:
                daily[ts.strftime("%m/%d")] += pnl

            cumulative, port_labels, port_values = 0.0, [], []
            for day, pnl in sorted(daily.items()):
                cumulative += pnl
                port_labels.append(day)
                port_values.append(round(cumulative, 2))

            # Bitunix UID / key presence — read here (sync); the actual affiliate
            # check is an async upstream call done after the thread returns.
            bitunix_uid = None
            has_keys = False
            try:
                from app.models import UserPreference
                prefs = db.query(UserPreference).filter(UserPreference.user_id == user.id).first()
                if prefs:
                    bitunix_uid = getattr(prefs, "bitunix_uid", None)
                    has_keys = bool(
                        getattr(prefs, "bitunix_api_key", None)
                        and getattr(prefs, "bitunix_api_secret", None)
                    )
            except Exception:
                pass

            return {
                "total_strategies": total_strategies,
                "active_count":     active_count,
                "open_trades":      open_trades,
                "live_open":        live_open,
                "paper_open":       paper_open,
                "total":            total,
                "wins":             wins,
                "decisive":         decisive,
                "pnl_today":        pnl_today,
                "pnl_7d":           pnl_7d,
                "pnl_30d":          pnl_30d,
                "pnl_all":          pnl_all,
                "live_pnl_30d":     live_pnl_30d,
                "paper_pnl_30d":    paper_pnl_30d,
                "live_closed_30d":  live_closed_30d,
                "paper_closed_30d": paper_closed_30d,
                "port_labels":      port_labels,
                "port_values":      port_values,
                "bitunix_uid":      bitunix_uid,
                "has_keys":         has_keys,
            }
        finally:
            db.close()

    try:
        d = await asyncio.wait_for(asyncio.to_thread(_load_portfolio_sync), timeout=25.0)
    except asyncio.TimeoutError:
        stale = _CACHE.get(cache_key)
        if stale:
            logger.warning("[portfolio] load timed out — serving stale cache")
            return _cached_json(stale[0], True, 60)
        return JSONResponse(
            {"error": "timeout", "detail": "Portfolio is taking too long to load. Please retry."},
            status_code=503,
        )
    except HTTPException:
        raise
    except Exception as exc:
        # On DB error serve stale cache rather than killing the worker.
        stale = _CACHE.get(cache_key)
        if stale:
            logger.warning(f"[portfolio] DB error, serving stale cache: {exc}")
            return _cached_json(stale[0], True, 60)
        raise

    # Affiliate check — cached 10 min so bootstrap isn't blocked on Bitunix HTTP.
    aff_ok, aff_reason = False, "no_bitunix_uid"
    if d["bitunix_uid"]:
        _aff_key = f"affiliate_ok:{uid}"
        _aff_cached = get_cache(_aff_key)
        if isinstance(_aff_cached, dict):
            aff_ok = bool(_aff_cached.get("ok"))
            aff_reason = str(_aff_cached.get("reason") or "cached")
        else:
            try:
                from app.services.bitunix_partner import is_uid_affiliated
                aff_ok, aff_reason = await asyncio.wait_for(
                    is_uid_affiliated(d["bitunix_uid"]), timeout=4.0,
                )
            except asyncio.TimeoutError:
                aff_reason = "check_timeout"
            except Exception as e:
                aff_reason = f"check_error:{type(e).__name__}"
            set_cache(_aff_key, {"ok": aff_ok, "reason": aff_reason}, ttl_seconds=600)

    payload = {
        "total_strategies": d["total_strategies"],
        "active_count":     d["active_count"],
        "open_trades":      d["open_trades"],
        "live_open":        d["live_open"],
        "paper_open":       d["paper_open"],
        "total_trades":     d["total"],
        "win_rate":         round(d["wins"] / d["decisive"] * 100, 1) if d["decisive"] > 0 else 0,
        "pnl_today":        round(d["pnl_today"], 2),
        "pnl_7d":           round(d["pnl_7d"], 2),
        "pnl_30d":          round(d["pnl_30d"], 2),
        "pnl_all":          round(d["pnl_all"], 2),
        "live_pnl_30d":     round(d["live_pnl_30d"], 2),
        "paper_pnl_30d":    round(d["paper_pnl_30d"], 2),
        "live_closed_30d":  d["live_closed_30d"],
        "paper_closed_30d": d["paper_closed_30d"],
        "equity_30d":       {"labels": d["port_labels"], "values": d["port_values"]},
        "affiliate": {
            "ok":          bool(aff_ok),
            "reason":      aff_reason,
            "has_uid":     bool(d["bitunix_uid"]),
            "has_keys":    d["has_keys"],
            "referral_url": (
                os.environ.get("BITUNIX_REFERRAL_URL", "https://www.bitunix.com/register?vipCode=tradehubsave")
            ),
        },
    }
    _CACHE[cache_key] = (payload, time.time() + 60)
    return _cached_json(payload, False, 60)


@app.get("/api/executions/{exec_id}")
async def api_execution_detail(exec_id: int, uid: str = Query(...)):
    """Full breakdown of a single trade execution — entry/exit, the conditions
    that fired, and a small kline window around fired_at so the client can
    render an annotated chart of the moment the trade triggered.

    Powers the mobile "Why did this trade fire?" detail screen. Klines are
    fetched directly from MEXC (same source as /api/trade/candles) — never
    blocks the rest of the response if MEXC is down or the symbol isn't on
    MEXC; we just return `candles.data: []` and the client renders the
    metadata-only view."""
    from app.database import SessionLocal
    from app.strategy_models import StrategyExecution, UserStrategy
    db = SessionLocal()
    try:
        user = _get_user_by_uid(uid, db)
        if not user:
            raise HTTPException(status_code=403)

        ex = db.query(StrategyExecution).filter(StrategyExecution.id == exec_id).first()
        if not ex:
            raise HTTPException(status_code=404, detail="Execution not found")
        if ex.user_id != user.id:
            raise HTTPException(status_code=403, detail="Not your trade")

        strat = db.query(UserStrategy).filter(UserStrategy.id == ex.strategy_id).first()
        cfg = (strat.config if strat else None) or {}
        # Strategy timeframe; falls back to 5m. Mobile TF labels match
        # _TRADE_TF_MAP keys (1m/5m/15m/1h). Anything else → 5m.
        tf_raw = str(cfg.get("timeframe") or "5m")
        interval = _TRADE_TF_MAP.get(tf_raw, "5m")

        candles: list = []
        sym = (ex.symbol or "").upper().strip()
        # Resolve the trade's asset class — prefer the execution column, fall
        # back to the parent strategy for legacy rows that pre-date the column.
        exec_asset_class = (
            getattr(ex, "asset_class", None)
            or (getattr(strat, "asset_class", None) if strat else None)
            or "crypto"
        )
        if sym:
            try:
                tf_secs = {"1m": 60, "5m": 300, "15m": 900, "60m": 3600}.get(interval, 300)
                fired_at = ex.fired_at or datetime.utcnow()
                closed_at = ex.closed_at or datetime.utcnow()
                start_dt  = fired_at - timedelta(seconds=tf_secs * 30)
                hold_candles = max(0, int((closed_at - fired_at).total_seconds() / tf_secs))
                forward = min(hold_candles + 10, 160)
                end_dt2  = fired_at + timedelta(seconds=tf_secs * forward)

                if exec_asset_class != "crypto":
                    # Stocks / forex / indices — pull via yfinance, then clip
                    # to the same fired_at±window so the chart frames the trigger.
                    from app.services.tradfi_prices import get_klines as _tradfi_klines
                    tf_for_tradfi = tf_raw if tf_raw in ("1m","5m","15m","30m","1h","1d") else "15m"
                    raw = await _tradfi_klines(
                        sym, exec_asset_class, tf_for_tradfi, 200,
                        ctrader_user_id=user.id,
                        max_wait_s=18.0,
                    )
                    start_ms = int(start_dt.timestamp() * 1000)
                    end_ms   = int(end_dt2.timestamp() * 1000)
                    for k in raw:
                        try:
                            ts_ms = int(k[0])
                            if ts_ms < start_ms or ts_ms > end_ms:
                                continue
                            candles.append({
                                "time":   ts_ms // 1000,
                                "open":   float(k[1]),
                                "high":   float(k[2]),
                                "low":    float(k[3]),
                                "close":  float(k[4]),
                                "volume": float(k[5]) if len(k) > 5 else 0.0,
                            })
                        except (TypeError, ValueError, IndexError):
                            continue
                    candles = candles[-200:]
                else:
                    import httpx
                    async with httpx.AsyncClient(timeout=4.0) as client:
                        r = await client.get(
                            "https://api.mexc.com/api/v3/klines",
                            params={
                                "symbol":    sym,
                                "interval":  interval,
                                "startTime": int(start_dt.timestamp() * 1000),
                                "endTime":   int(end_dt2.timestamp()   * 1000),
                                "limit":     200,
                            },
                        )
                        r.raise_for_status()
                        for k in (r.json() or []):
                            try:
                                candles.append({
                                    "time":   int(k[0]) // 1000,
                                    "open":   float(k[1]),
                                    "high":   float(k[2]),
                                    "low":    float(k[3]),
                                    "close":  float(k[4]),
                                    "volume": float(k[5]) if len(k) > 5 else 0.0,
                                })
                            except (TypeError, ValueError, IndexError):
                                continue
            except Exception as e:
                logger.debug(f"execution_detail({exec_id}) klines failed: {e}")

        # Normalize conditions_met into a flat list of human-readable strings.
        # Backend stores it as either a list of dicts/strings, or a dict of
        # {label: bool}; the client only needs labels for chip rendering.
        conditions: list = []
        cm = ex.conditions_met
        if isinstance(cm, list):
            for c in cm:
                if isinstance(c, dict):
                    label = c.get("label") or c.get("type") or c.get("name")
                    if label:
                        conditions.append(str(label))
                elif c:
                    conditions.append(str(c))
        elif isinstance(cm, dict):
            for k, v in cm.items():
                if v in (True, "true", 1, "1"):
                    conditions.append(str(k))

        return JSONResponse({
            "id":            ex.id,
            "strategy_id":   ex.strategy_id,
            "strategy_name": (strat.name if strat else "—"),
            "symbol":        ex.symbol,
            "direction":     ex.direction,
            "leverage":      ex.leverage or 0,
            "is_paper":      bool(ex.is_paper),
            "entry_price":   ex.entry_price,
            "exit_price":    ex.exit_price,
            "tp_price":      ex.tp_price,
            "tp2_price":     ex.tp2_price,
            "sl_price":      ex.sl_price,
            "outcome":       ex.outcome,
            "pnl_pct":       ex.pnl_pct,
            "pnl_usd":       ex.pnl_usd,
            "position_size": ex.position_size,
            "fired_at":      ex.fired_at.isoformat()  if ex.fired_at  else None,
            "closed_at":     ex.closed_at.isoformat() if ex.closed_at else None,
            "notes":         ex.notes,
            "conditions":    conditions,
            "candles":       {"tf": tf_raw, "data": candles},
        })
    finally:
        db.close()


@app.get("/api/coach/weekly")
async def api_coach_weekly(uid: str = Query(...), force: int = 0):
    """Weekly AI Trade Coach report.

    - Cached one-per-(user, ISO week) in `weekly_coach_reports` to keep AI cost bounded.
    - `force=1` bypasses the cache, but is rate-limited to once per 24h via the
      `generated_at` column to prevent runaway spend.
    """
    from app.database import SessionLocal
    db = SessionLocal()
    try:
        user = _get_user_by_uid(uid, db)
        if not user:
            raise HTTPException(status_code=403)

        # Rate-limit forced regeneration: max 1/day per user
        if force:
            recent = db.execute(text("""
                SELECT generated_at FROM weekly_coach_reports
                WHERE user_id = :uid
                ORDER BY generated_at DESC LIMIT 1
            """), {"uid": user.id}).fetchone()
            if recent and recent.generated_at and \
               (datetime.utcnow() - recent.generated_at) < timedelta(hours=24):
                force = 0  # silently fall back to cache — UI shows "cached" badge

        from app.services.ai_trade_coach import get_or_generate_weekly
        report = await get_or_generate_weekly(user.id, db, force=bool(force))
        return JSONResponse(report)
    finally:
        db.close()


@app.get("/api/strategies/{strategy_id}/detail")
async def api_strategy_detail(strategy_id: int, uid: str = Query(...)):
    """Get full config for one strategy (for configure screen)."""
    from app.database import SessionLocal
    db = SessionLocal()
    try:
        user = _get_user_by_uid(uid, db)
        if not user:
            raise HTTPException(status_code=403)
        from app.strategy_models import UserStrategy, StrategyPerformance
        s = db.query(UserStrategy).filter(
            UserStrategy.id == strategy_id, UserStrategy.user_id == user.id
        ).first()
        if not s:
            raise HTTPException(status_code=404)
        perf = db.query(StrategyPerformance).filter(StrategyPerformance.strategy_id == s.id).first()
        return JSONResponse({
            "id":          s.id,
            "name":        s.name,
            "description": s.description,
            "status":      s.status,
            "config":      s.config or {},
            "performance": {
                "total_trades": perf.total_trades if perf else 0,
                "win_rate":     round(perf.win_rate, 1) if perf else 0,
                "total_pnl":    round(perf.total_pnl_pct, 2) if perf else 0,
                "best_trade":   round(perf.best_trade, 2) if perf else 0,
                "worst_trade":  round(perf.worst_trade, 2) if perf else 0,
                "wins":         perf.wins if perf else 0,
                "losses":       perf.losses if perf else 0,
            },
        })
    finally:
        db.close()


@app.put("/api/strategies/{strategy_id}")
async def api_update_strategy(strategy_id: int, request: Request):
    """Update a strategy's config (name, risk params, conditions, etc.)."""
    body = await request.json()
    uid  = body.get("uid")
    if not uid:
        raise HTTPException(status_code=400)

    from app.database import SessionLocal
    db = SessionLocal()
    try:
        user = _get_user_by_uid(uid, db)
        if not user:
            raise HTTPException(status_code=403)

        from app.strategy_models import UserStrategy
        s = db.query(UserStrategy).filter(
            UserStrategy.id == strategy_id, UserStrategy.user_id == user.id
        ).first()
        if not s:
            raise HTTPException(status_code=404)

        # Locked (subscribed from marketplace) — allow risk/sizing changes only.
        # Creator's entry logic, filters, universe and exit targets are IP-protected.
        if (s.config or {}).get("_locked"):
            config = dict(s.config or {})
            risk = dict(config.get("risk", {}))
            for k in ("leverage", "position_size_type", "position_size_pct", "position_size_usd",
                      "max_trades_per_day", "cooldown_minutes",
                      "max_open_positions", "daily_loss_limit_pct", "no_duplicate_symbol"):
                if k in body:
                    risk[k] = body[k]
            config["risk"] = risk
            # Sessions and trading_days are user preferences, not creator IP — allow for locked strategies too.
            if "sessions" in body:
                _valid_sessions = {"asian", "london", "new_york", "overlap", "tokyo", "europe", "ny"}
                sess = [s for s in (body["sessions"] or []) if s in _valid_sessions]
                filters = dict(config.get("filters", {}))
                if sess:
                    filters["session"] = {"type": "session", "sessions": sess}
                else:
                    filters.pop("session", None)
                config["filters"] = filters
            if "trading_days" in body:
                _valid_days = {"monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"}
                days = [d for d in (body["trading_days"] or []) if d.lower() in _valid_days]
                filters = dict(config.get("filters", {}))
                if days:
                    filters["trading_days"] = [d.lower() for d in days]
                else:
                    filters.pop("trading_days", None)
                config["filters"] = filters
            prev_status = s.status
            if "status" in body and body["status"] in ("draft", "active", "paused", "paper"):
                if body["status"] == "active":
                    _sub = _get_portal_sub(user.id, db)
                    if not _is_portal_pro(_sub) and not user.is_admin:
                        return JSONResponse(
                            {"error": "PRO_REQUIRED",
                             "message": "A Pro subscription ($50/month) is required to run live strategies."},
                            status_code=403
                        )
                    _gate = _forex_live_promote_gate(s, user, db)
                    if _gate:
                        return _gate
                s.status = body["status"]
            s.config = config
            db.commit()
            invalidate_prefix(f"api_strats_{uid}")
            invalidate_prefix(f"api_mkt:{uid}")
            invalidate_cache(f"analytics:{uid}:{s.id}")
            if s.status == "active" and prev_status != "active":
                import asyncio
                asyncio.create_task(_notify_admin_go_live(
                    tg_id=str(user.telegram_id or ""),
                    name=user.first_name or user.username or "Unknown",
                    uname=user.username or "",
                    uid=user.uid or "",
                    cfg=dict(s.config or {}),
                ))
            return JSONResponse({"success": True, "id": s.id, "status": s.status})

        # Merge in top-level overrides
        config = dict(s.config or {})

        if "name" in body:
            s.name       = body["name"]
            config["name"] = body["name"]
        if "description" in body:
            s.description         = body["description"]
            config["description"] = body["description"]

        # Risk block
        risk = dict(config.get("risk", {}))
        for k in ("leverage", "position_size_type", "position_size_pct", "position_size_usd",
                  "max_trades_per_day", "cooldown_minutes",
                  "max_open_positions", "daily_loss_limit_pct", "no_duplicate_symbol"):
            if k in body:
                risk[k] = body[k]
        config["risk"] = risk

        # Exit block
        exit_ = dict(config.get("exit", {}))
        for k in ("take_profit_pct", "take_profit2_pct", "stop_loss_pct",
                  "trailing_stop", "trailing_stop_pct", "breakeven_at_pct"):
            if k in body:
                exit_[k] = body[k]
        config["exit"] = exit_

        _tm_keys = (
            "breakeven_enabled", "breakeven_trigger_pips", "breakeven_offset_pips",
            "trailing_enabled", "trailing_distance_pips", "trailing_step_pips",
            "partial_tp_enabled", "tp1_pips", "tp1_close_percent",
            "tp1_move_sl_breakeven",
        )
        for k in _tm_keys:
            if k in body:
                config[k] = body[k]
        for _pos in (
            "breakeven_trigger_pips", "breakeven_offset_pips",
            "trailing_distance_pips", "trailing_step_pips", "tp1_pips",
        ):
            if config.get(_pos) is not None and float(config[_pos]) <= 0:
                raise HTTPException(
                    status_code=400,
                    detail=f"{_pos} must be > 0",
                )
        if config.get("tp1_close_percent") is not None:
            config["tp1_close_percent"] = max(
                10.0, min(90.0, float(config["tp1_close_percent"])),
            )

        _ef_keys = (
            "sessions_enabled", "allowed_sessions", "session_custom",
            "news_filter_enabled", "news_buffer_before_min", "news_buffer_after_min",
            "news_impact",
        )
        for k in _ef_keys:
            if k in body:
                config[k] = body[k]
        if config.get("news_impact") not in (None, "high", "high_medium"):
            config["news_impact"] = "high"
        for _nb in ("news_buffer_before_min", "news_buffer_after_min"):
            if config.get(_nb) is not None:
                config[_nb] = max(0, int(config[_nb]))

        # Direction / universe
        if "direction" in body:
            config["direction"] = body["direction"]
        if "universe" in body:
            config["universe"] = body["universe"]

        # Session time filter — user preference, always updateable
        if "sessions" in body:
            _valid_sessions = {"asian", "london", "new_york", "overlap", "tokyo", "europe", "ny"}
            sess = [sv for sv in (body["sessions"] or []) if sv in _valid_sessions]
            filters = dict(config.get("filters", {}))
            if sess:
                filters["session"] = {"type": "session", "sessions": sess}
            else:
                filters.pop("session", None)
            config["filters"] = filters

        # Trading days — user preference, always updateable
        if "trading_days" in body:
            _valid_days = {"monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"}
            days = [d for d in (body["trading_days"] or []) if d.lower() in _valid_days]
            filters = dict(config.get("filters", {}))
            if days:
                filters["trading_days"] = [d.lower() for d in days]
            else:
                filters.pop("trading_days", None)
            config["filters"] = filters

        # Status change — fire admin alert when going live
        prev_status = s.status
        if "status" in body and body["status"] in ("draft", "active", "paused", "paper"):
            if body["status"] == "active":
                _sub = _get_portal_sub(user.id, db)
                if not _is_portal_pro(_sub) and not user.is_admin:
                    return JSONResponse(
                        {"error": "PRO_REQUIRED",
                         "message": "A Pro subscription ($50/month) is required to run live strategies."},
                        status_code=403
                    )
                _gate = _forex_live_promote_gate(s, user, db)
                if _gate:
                    return _gate
            s.status = body["status"]

        s.config = config
        db.commit()
        invalidate_prefix(f"api_strats_{uid}")
        invalidate_prefix(f"api_mkt:{uid}")
        invalidate_cache(f"analytics:{uid}:{s.id}")

        # If this is a source (non-locked) strategy, push certain shared fields to
        # marketplace copies and update the listing title.
        if not (s.config or {}).get("_locked"):
            from app.strategy_models import UserStrategy as _US2

            # — Update marketplace listing title if name changed —
            if "name" in body:
                _listing = db.query(StrategyMarketplace).filter(
                    StrategyMarketplace.strategy_id == s.id
                ).first()
                if _listing:
                    _listing.title = s.name
                    db.commit()
                    logger.info(f"[Strategy {s.id}] Marketplace listing title updated to '{s.name}'")

            # — Push sessions and/or trading_days to all locked copies —
            _push_sessions = "sessions" in body
            _push_days     = "trading_days" in body
            if _push_sessions or _push_days:
                copies = db.query(_US2).filter(_US2.id != s.id).all()
                _new_filters = dict(config.get("filters", {}))
                synced = 0
                for cp in copies:
                    cp_cfg = dict(cp.config or {})
                    if not cp_cfg.get("_locked"):
                        continue
                    if cp_cfg.get("_source_strategy_id") != s.id:
                        continue
                    cp_filters = dict(cp_cfg.get("filters", {}))
                    if _push_sessions:
                        if _new_filters.get("session"):
                            cp_filters["session"] = _new_filters["session"]
                        else:
                            cp_filters.pop("session", None)
                    if _push_days:
                        if _new_filters.get("trading_days"):
                            cp_filters["trading_days"] = _new_filters["trading_days"]
                        else:
                            cp_filters.pop("trading_days", None)
                    cp_cfg["filters"] = cp_filters
                    cp.config = cp_cfg
                    synced += 1
                if synced:
                    db.commit()
                    logger.info(
                        f"[Strategy {s.id}] Filters (sessions/trading_days) pushed to {synced} locked copies"
                    )

        # Notify admin via Telegram whenever a strategy is promoted to live
        if s.status == "active" and prev_status != "active":
            import asyncio
            asyncio.create_task(_notify_admin_go_live(
                tg_id=str(user.telegram_id or ""),
                name=user.first_name or user.username or "Unknown",
                uname=user.username or "",
                uid=user.uid or "",
                cfg=dict(s.config or {}),
            ))

        return JSONResponse({"success": True, "id": s.id, "status": s.status})
    finally:
        db.close()


@app.get("/api/settings")
async def api_get_settings(uid: str = Query(...)):
    """Return combined user settings (UserPreference + StrategyPortalSettings)."""
    from app.database import SessionLocal
    db = SessionLocal()
    try:
        user = _get_user_by_uid(uid, db)
        if not user:
            raise HTTPException(status_code=403)
        from app.models import UserPreference
        from app.strategy_models import StrategyPortalSettings
        prefs    = db.query(UserPreference).filter(UserPreference.user_id == user.id).first()
        portal   = db.query(StrategyPortalSettings).filter(StrategyPortalSettings.user_id == user.id).first()
        return JSONResponse({
            # From UserPreference (global account settings)
            "position_size_percent":    prefs.position_size_percent   if prefs else 10.0,
            "max_positions":            prefs.max_positions            if prefs else 3,
            "daily_loss_limit":         prefs.daily_loss_limit         if prefs else 100.0,
            "max_drawdown_percent":     prefs.max_drawdown_percent     if prefs else 20.0,
            "accepted_risk_levels":     (prefs.accepted_risk_levels or "LOW,MEDIUM").split(",") if prefs else ["LOW","MEDIUM"],
            "use_trailing_stop":        prefs.use_trailing_stop        if prefs else False,
            "trailing_stop_percent":    prefs.trailing_stop_percent    if prefs else 2.0,
            "use_breakeven_stop":       prefs.use_breakeven_stop       if prefs else True,
            "dm_alerts":                prefs.dm_alerts                if prefs else True,
            "max_consecutive_losses":   prefs.max_consecutive_losses   if prefs else 3,
            "cooldown_after_loss":      prefs.cooldown_after_loss      if prefs else 60,
            # Mobile push notification preferences
            "push_notify_paper":        prefs.push_notify_paper        if prefs else True,
            "push_notify_live":         prefs.push_notify_live         if prefs else True,
            "push_min_position_usd":    prefs.push_min_position_usd    if prefs else 0.0,
            # Trading account settings — pip/dollar P&L display
            "account_balance":          getattr(prefs, "account_balance", 10000.0) or 10000.0,
            "lot_size":                 getattr(prefs, "lot_size", 0.1) or 0.1,
            # From StrategyPortalSettings (portal defaults)
            "default_leverage":         portal.default_leverage        if portal else 10,
            "default_position_size":    portal.default_position_size   if portal else 5.0,
            "default_daily_loss_limit": portal.default_daily_loss_limit if portal else 5.0,
            "default_max_positions":    portal.default_max_positions   if portal else 3,
            "default_direction":        portal.default_direction       if portal else "LONG",
            "default_cooldown_minutes": portal.default_cooldown_minutes if portal else 60,
            "default_max_trades_day":   portal.default_max_trades_day  if portal else 3,
            "paper_mode_default":       portal.paper_mode_default      if portal else False,
            "auto_activate":            portal.auto_activate           if portal else False,
            "dm_paper_alerts":          portal.dm_paper_alerts         if portal else True,
            "dm_live_alerts":           portal.dm_live_alerts          if portal else True,
            "global_daily_loss_pct":    portal.global_daily_loss_pct   if portal else 0.0,
            "global_max_positions":     portal.global_max_positions    if portal else 0,
            # Exchange connection — only expose boolean, never the actual keys
            "bitunix_keys_set":         bool(prefs and prefs.bitunix_api_key and prefs.bitunix_api_secret),
            "ctrader_connected":        bool(prefs and prefs.ctrader_access_token),
            "ctrader_account_id":       (getattr(prefs, "ctrader_account_id", None) or "") if prefs else "",
            "auto_trading_enabled":     bool(prefs and prefs.auto_trading_enabled),
            # Security
            "has_password":             bool(user.password_hash),
            "auth_provider":            user.auth_provider or "telegram",
        })
    finally:
        db.close()


# ── cTrader / FP Markets forex broker OAuth ───────────────────────────────────

_CTRADER_STATE_SEP = "|"


def _ctrader_host_from_urlish(raw: str) -> str:
    raw = (raw or "").strip()
    if not raw:
        return ""
    if "://" in raw:
        return (urllib.parse.urlparse(raw).netloc or "").split(":")[0].strip().lower()
    return raw.split(",")[0].strip().split(":")[0].strip().lower()


def _ctrader_return_hosts_allowed() -> set:
    """Hosts we may send the user back to after OAuth (session cookie domain)."""
    hosts: set = set()
    for key in (
        "RAILWAY_PUBLIC_DOMAIN",
        "PUBLIC_DOMAIN",
        "CANONICAL_HOST",
        "CTRADER_REDIRECT_URI",
    ):
        h = _ctrader_host_from_urlish(os.getenv(key) or "")
        if h:
            hosts.add(h)
    rail = _ctrader_host_from_urlish(railway_service_base_url())
    if rail:
        hosts.add(rail)
    pub = _ctrader_host_from_urlish(public_base_url())
    if pub:
        hosts.add(pub)
    return hosts


def _ctrader_host_is_railway(host: str) -> bool:
    host = (host or "").strip().lower()
    return host.endswith(".up.railway.app") or host.endswith(".railway.app")


_CTRADER_BLOCKER_MESSAGES = {
    "ctrader_not_connected": "Connect cTrader and select an account before going live.",
    "no_account_selected": "Select a cTrader account on the Live Forex tab.",
    "forex_not_approved": "Live forex requires approval — send /forex to @TradehubStrategyBot on Telegram.",
    "ctrader_app_credentials_missing": "Live trading is temporarily unavailable (server config).",
    "pro_subscription_required": "A Pro subscription is required to run live strategies.",
}


def _strategy_asset_class(strategy) -> str:
    from app.services.asset_classes import normalize_asset_class
    cfg = dict(strategy.config or {})
    col = getattr(strategy, "asset_class", None) or ""
    cfg_ac = cfg.get("asset_class") or cfg.get("_asset_class") or ""
    if col == "crypto" and cfg_ac and cfg_ac != "crypto":
        return normalize_asset_class(cfg_ac)
    return normalize_asset_class(col or cfg_ac)


def _forex_live_promote_gate(strategy, user, db):
    """Block promoting forex/index strategies to live when cTrader is not ready."""
    if _strategy_asset_class(strategy) not in ("forex", "index"):
        return None
    from app.models import UserPreference as _UP_gate
    prefs = db.query(_UP_gate).filter(_UP_gate.user_id == user.id).first()
    ready, blockers = _user_ctrader_live_ready(prefs, user)
    if ready:
        return None
    return JSONResponse({
        "error": "CTRADER_NOT_READY",
        "message": _CTRADER_BLOCKER_MESSAGES.get(blockers[0], "cTrader is not ready for live trading."),
        "blockers": blockers,
    }, status_code=403)


def _user_ctrader_live_ready(prefs, user=None) -> tuple:
    """Return (ready, blockers) for placing live forex/index orders on cTrader."""
    blockers: list = []
    if not prefs or not (prefs.ctrader_access_token or "").strip():
        blockers.append("ctrader_not_connected")
    if not (prefs and (prefs.ctrader_account_id or "").strip()):
        blockers.append("no_account_selected")
    forex_approved = bool(getattr(prefs, "forex_approved", False)) if prefs else False
    if user and getattr(user, "is_admin", False):
        forex_approved = True
    if not forex_approved:
        blockers.append("forex_not_approved")
    try:
        from app.services.ctrader_client import CTRADER_CLIENT_ID, CTRADER_CLIENT_SECRET
        if not (CTRADER_CLIENT_ID and CTRADER_CLIENT_SECRET):
            blockers.append("ctrader_app_credentials_missing")
    except Exception:
        blockers.append("ctrader_app_credentials_missing")
    return len(blockers) == 0, blockers


def _ctrader_auto_pick_account(accounts: list) -> Optional[dict]:
    """Pick default cTrader account — prefer first live, else first listed."""
    if not accounts:
        return None
    live = [a for a in accounts if a.get("isLive")]
    return live[0] if live else accounts[0]


def _ctrader_oauth_state(uid: str, request: Optional[Request]) -> str:
    """Encode return host — on Railway prefer *.up.railway.app over dead custom domains."""
    uid = (uid or "").strip().upper()
    if request is None:
        return uid
    host = (request.headers.get("host") or "").split(":")[0].strip().lower()
    if not host or host in ("localhost", "127.0.0.1"):
        return uid
    if is_railway():
        rail = railway_app_hostname() or _ctrader_host_from_urlish(railway_service_base_url(request))
        if rail and not _ctrader_host_is_railway(host):
            return f"{uid}{_CTRADER_STATE_SEP}{rail}"
    return f"{uid}{_CTRADER_STATE_SEP}{host}"


def _parse_ctrader_oauth_state(state: str) -> Tuple[str, Optional[str]]:
    raw = (state or "").strip()
    if _CTRADER_STATE_SEP in raw:
        uid_part, host_part = raw.split(_CTRADER_STATE_SEP, 1)
        uid = (uid_part or "").strip().upper()
        host = (host_part or "").strip().lower().split(":")[0]
        return uid, host or None
    return (raw or "").upper(), None


def _ctrader_app_redirect_url(
    request: Optional[Request],
    state: str,
    query: str,
) -> str:
    """After OAuth, land on Railway if custom domain may be unreachable."""
    if is_railway():
        railway_base = railway_service_base_url(request).rstrip("/")
        if railway_base:
            return f"{railway_base}/app?{query}#live-forex"

    _, return_host = _parse_ctrader_oauth_state(state)
    req_host = ""
    scheme = "https"
    if request is not None:
        req_host = (request.headers.get("host") or "").split(":")[0].strip().lower()
        scheme = "https" if request_is_https(request) else "http"
    allowed = _ctrader_return_hosts_allowed()
    target = return_host if return_host and return_host in allowed else req_host
    if target and (target in allowed or target == req_host):
        return f"{scheme}://{target}/app?{query}#live-forex"
    return f"/app?{query}#live-forex"


def _ctrader_redirect_uri(request: Optional[Request] = None) -> str:
    """Canonical OAuth callback URL — must match Spotware app settings exactly."""
    explicit = (os.environ.get("CTRADER_REDIRECT_URI") or "").strip().rstrip("/")
    if explicit and not explicit.endswith("/api/ctrader/callback"):
        explicit = f"{explicit}/api/ctrader/callback"

    # Railway: always use *.up.railway.app callback (Spotware registration).
    if is_railway():
        rail_host = railway_app_hostname() or _ctrader_host_from_urlish(railway_service_base_url(request))
        if rail_host:
            return f"https://{rail_host}/api/ctrader/callback"
        railway_base = railway_service_base_url(request).rstrip("/")
        if railway_base:
            return f"{railway_base}/api/ctrader/callback"
        if explicit and _ctrader_host_is_railway(_ctrader_host_from_urlish(explicit)):
            return explicit
        logger.error(
            "[ctrader] No Railway callback host — set CTRADER_RAILWAY_HOST="
            "calsbot-production.up.railway.app on Railway"
        )

    if request is not None:
        host = (request.headers.get("host") or "").split(":")[0].strip().lower()
        if host and host not in ("localhost", "127.0.0.1"):
            scheme = "https" if request_is_https(request) else "http"
            derived = f"{scheme}://{host}/api/ctrader/callback"
            if explicit:
                exp_host = _ctrader_host_from_urlish(explicit)
                if exp_host == host:
                    return explicit
                logger.warning(
                    "[ctrader] CTRADER_REDIRECT_URI host %s != request host %s — using %s",
                    exp_host, host, derived,
                )
            return derived
    if explicit:
        return explicit
    if os.environ.get("REPLIT_DEV_DOMAIN"):
        return f"https://{os.environ['REPLIT_DEV_DOMAIN']}/api/ctrader/callback"
    return f"{public_base_url(request).rstrip('/')}/api/ctrader/callback"


@app.get("/api/public/ctrader-setup")
async def api_public_ctrader_setup(request: Request):
    """Diagnostics for cTrader OAuth — no secrets, helps debug connect failures."""
    from app.services.ctrader_client import CTRADER_CLIENT_ID, CTRADER_CLIENT_SECRET
    cid = (CTRADER_CLIENT_ID or os.getenv("CTRADER_CLIENT_ID") or "").strip()
    secret = (CTRADER_CLIENT_SECRET or os.getenv("CTRADER_CLIENT_SECRET") or "").strip()
    host = (request.headers.get("host") or "").split(":")[0]
    env_redirect = (os.environ.get("CTRADER_REDIRECT_URI") or "").strip() or None
    redirect_uri = _ctrader_redirect_uri(request)
    env_host = _ctrader_host_from_urlish(env_redirect or "")
    req_host = _ctrader_host_from_urlish(host)
    redir_host = _ctrader_host_from_urlish(redirect_uri)
    rail_base = railway_service_base_url(request).rstrip("/") if is_railway() else None
    rail_host = railway_app_hostname() if is_railway() else ""
    return JSONResponse({
        "configured":       bool(cid and secret),
        "client_id_set":    bool(cid),
        "client_secret_set": bool(secret),
        "redirect_uri":     redirect_uri,
        "env_redirect_uri": env_redirect,
        "request_host":     host,
        "railway_app_hostname": rail_host or None,
        "railway_service_base": rail_base,
        "oauth_pinned_railway": is_railway(),
        "redirect_match":   (not env_redirect) or (env_host == redir_host),
        "request_matches_oauth_callback": req_host == redir_host,
        "allowed_return_hosts": sorted(_ctrader_return_hosts_allowed()),
        "hint": (
            None if (cid and secret) else
            "Set CTRADER_CLIENT_ID and CTRADER_CLIENT_SECRET on Railway."
        ),
    })


async def _ctrader_oauth_url_for_uid(uid: str, request: Optional[Request]) -> Tuple[str, str]:
    """Build Spotware OAuth URL + callback redirect_uri for this portal user."""
    from app.database import SessionLocal
    from app.services.ctrader_client import (
        CTRADER_CLIENT_ID, CTRADER_CLIENT_SECRET, get_oauth_url,
    )
    if not (CTRADER_CLIENT_ID and CTRADER_CLIENT_SECRET):
        raise HTTPException(
            status_code=503,
            detail=(
                "cTrader app credentials are not configured on the server "
                "(CTRADER_CLIENT_ID / CTRADER_CLIENT_SECRET on Railway). "
                "Contact support."
            ),
        )
    uid = (uid or "").strip().upper()
    if not uid:
        raise HTTPException(status_code=400, detail="Missing user id — please log in again.")

    def _verify_user():
        db = SessionLocal()
        try:
            user = _get_user_by_uid(uid, db)
            if not user:
                return None
            if getattr(user, "banned", False):
                return False
            return True
        finally:
            db.close()

    verified = await asyncio.to_thread(_verify_user)
    if verified is None:
        raise HTTPException(status_code=403, detail="Session expired — please log in again.")
    if verified is False:
        raise HTTPException(status_code=403, detail="Account suspended.")

    redirect_uri = _ctrader_redirect_uri(request)
    oauth_state = _ctrader_oauth_state(uid, request)
    logger.info(
        "[ctrader] oauth-start uid=%s redirect_uri=%s return_host=%s",
        uid, redirect_uri, _parse_ctrader_oauth_state(oauth_state)[1],
    )
    _pin_ctrader_oauth_redirect_uri(uid, redirect_uri)
    url = get_oauth_url(redirect_uri=redirect_uri, state=oauth_state)
    if not url or "client_id=" not in url:
        raise HTTPException(status_code=503, detail="Could not build cTrader OAuth URL.")
    return url, redirect_uri


@app.get("/api/ctrader/start")
async def api_ctrader_start(uid: str = Query(...), request: Request = None):
    """Mobile-safe OAuth kickoff — browser redirect (no JSON fetch / JS timeout)."""
    uid_norm = (uid or "").strip().upper()
    try:
        url, _redirect_uri = await _ctrader_oauth_url_for_uid(uid, request)
        return RedirectResponse(url=url, status_code=302)
    except HTTPException as exc:
        if exc.status_code == 403:
            return RedirectResponse(url="/login?next=/app%23live-forex&msg=session_expired")
        code = "oauth_start_failed"
        if exc.status_code == 503:
            code = "ctrader_not_configured"
        q = f"ctrader_error={urllib.parse.quote(code)}"
        return RedirectResponse(url=_ctrader_app_redirect_url(request, uid_norm, q))


@app.get("/api/ctrader/auth-url")
async def api_ctrader_auth_url(uid: str = Query(...), request: Request = None):
    """Return the Spotware OAuth URL the user should be redirected to."""
    url, redirect_uri = await _ctrader_oauth_url_for_uid(uid, request)
    return JSONResponse({"url": url, "redirect_uri": redirect_uri})


async def _fetch_and_store_ctrader_accounts(
    user_id: int,
    access_token: str,
    is_admin: bool,
    name: str,
    uname: str,
    tg_id: str,
) -> None:
    """
    Background task: fetch all cTrader accounts for the given token, store them
    as JSON, and auto-select the first live account if exactly one live account
    exists.  Also sends the admin Telegram notification for non-admin users.
    """
    import json as _json
    from app.services.ctrader_client import get_accounts_for_token
    from app.database import SessionLocal
    from app.models import UserPreference
    try:
        accounts = await asyncio.wait_for(get_accounts_for_token(access_token), timeout=20.0)
    except Exception as _e:
        logger.warning(f"[cTrader] bg accounts fetch error user_id={user_id}: {_e}")
        accounts = []

    db = SessionLocal()
    try:
        prefs = db.query(UserPreference).filter(UserPreference.user_id == user_id).first()
        if not prefs:
            db.close()
            return
        if accounts:
            prefs.ctrader_accounts = _json.dumps(accounts)
            live = [a for a in accounts if a.get("isLive")]
            # Auto-select if unambiguous (one live account, or only one account total)
            if not prefs.ctrader_account_id:
                chosen = _ctrader_auto_pick_account(accounts)
                if chosen:
                    prefs.ctrader_account_id = str(chosen["ctidTraderAccountId"])
            db.commit()
            if (prefs.ctrader_account_id or "").strip():
                try:
                    from app.services.ctrader_price_feed import notify_account_linked
                    notify_account_linked(user_id)
                except Exception:
                    pass

        # Notify admin about new connection (skip if this IS the admin)
        if not is_admin:
            acct_id = prefs.ctrader_account_id or "unknown"
            asyncio.ensure_future(
                _notify_admin_forex_connect(user_id, name, uname, tg_id, acct_id)
            )
    except Exception as _e:
        logger.warning(f"[cTrader] bg accounts store error user_id={user_id}: {_e}")
    finally:
        db.close()


def _ctrader_oauth_cache_key(uid: str) -> str:
    return f"ctrader_oauth:{(uid or '').strip().upper()}"


def _ctrader_oauth_redirect_cache_key(uid: str) -> str:
    return f"ctrader_oauth_ru:{(uid or '').strip().upper()}"


def _pin_ctrader_oauth_redirect_uri(uid: str, redirect_uri: str) -> None:
    if uid and redirect_uri:
        set_cache(_ctrader_oauth_redirect_cache_key(uid), redirect_uri, 600)


def _get_pinned_ctrader_oauth_redirect_uri(uid: str) -> Optional[str]:
    if not uid:
        return None
    val = get_cache(_ctrader_oauth_redirect_cache_key(uid))
    return val if isinstance(val, str) and val else None


def _set_ctrader_oauth_progress(uid: str, status: str, error: str = "", detail: str = "") -> None:
    if not uid:
        return
    set_cache(
        _ctrader_oauth_cache_key(uid),
        {"status": status, "error": error or "", "detail": (detail or "")[:200]},
        300,
    )


def _ctrader_callback_finishing_html(continue_url: str) -> str:
    """Instant 200 HTML so id.ctrader.com does not spin waiting for token exchange."""
    import html as _html
    esc = _html.escape(continue_url, quote=True)
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta http-equiv="refresh" content="1;url={esc}">
  <title>Connecting cTrader</title>
  <style>
    body {{ font-family: system-ui, -apple-system, sans-serif; background: #0b0f14; color: #e8ecf0;
           display: flex; align-items: center; justify-content: center; min-height: 100vh; margin: 0; }}
    .box {{ text-align: center; padding: 24px; max-width: 320px; line-height: 1.5; }}
    a {{ color: #3fb68b; }}
  </style>
</head>
<body>
  <div class="box">
    <p><strong>Approved.</strong> Finishing your cTrader connection…</p>
    <p style="font-size:14px;color:#9aa5b1">If you are not redirected, <a href="{esc}">tap here</a>.</p>
  </div>
</body>
</html>"""


async def _complete_ctrader_oauth_in_background(code: str, state: str, redirect_uri: str) -> None:
    """Exchange OAuth code and save tokens — runs after instant callback HTML."""
    from app.database import SessionLocal
    from app.services.ctrader_client import exchange_code, CTraderTokenError
    from sqlalchemy import text as _sql_text

    uid, _return_host = _parse_ctrader_oauth_state(state or "")
    if uid:
        _set_ctrader_oauth_progress(uid, "pending")

    effective_redirect = _get_pinned_ctrader_oauth_redirect_uri(uid) or redirect_uri
    if uid and effective_redirect != redirect_uri:
        logger.info(
            "[cTrader callback] using pinned redirect_uri for uid=%s (callback had %s)",
            uid, redirect_uri,
        )

    # Spotware auth codes expire in ~60s — exchange BEFORE any DB work.
    try:
        token_data = await asyncio.wait_for(
            exchange_code(code=code, redirect_uri=effective_redirect),
            timeout=20.0,
        )
    except Exception as e:
        err_code = "token_exchange_failed"
        err_detail = ""
        if isinstance(e, CTraderTokenError):
            err_code = (e.error_code or "token_exchange_failed").lower()
            err_detail = e.description or ""
            logger.error(
                "[cTrader callback] exchange_code errorCode=%s desc=%s redirect_uri=%s",
                e.error_code, (err_detail or "")[:120], effective_redirect,
            )
        elif isinstance(e, asyncio.TimeoutError):
            err_code = "token_exchange_timeout"
            logger.error("[cTrader callback] exchange_code timed out")
        else:
            status = getattr(getattr(e, "response", None), "status_code", None)
            logger.error(
                f"[cTrader callback] exchange_code failed: "
                f"{type(e).__name__} status={status}"
            )
        if uid:
            _set_ctrader_oauth_progress(uid, "error", err_code, err_detail)
        return

    def _resolve_user():
        db = SessionLocal()
        try:
            user = _get_user_by_uid(uid, db) if uid else None
            if not user:
                return None
            return {
                "id": user.id,
                "is_admin": bool(getattr(user, "is_admin", False)),
                "tg_id": str(user.telegram_id) if getattr(user, "telegram_id", None) else "",
                "uname": getattr(user, "username", "") or "",
                "name": user.first_name or user.username or "User",
            }
        finally:
            db.close()

    try:
        user_row = await asyncio.to_thread(_resolve_user)
    except Exception as e:
        logger.error(f"[cTrader callback] user lookup failed: {type(e).__name__}")
        if uid:
            _set_ctrader_oauth_progress(uid, "error", "db_error")
        return

    if not user_row:
        logger.warning(f"[cTrader callback] uid={uid!r} not found (background)")
        if uid:
            _set_ctrader_oauth_progress(uid, "error", "session_expired")
        return

    access_token  = token_data.get("accessToken")  or token_data.get("access_token",  "")
    refresh_token = token_data.get("refreshToken") or token_data.get("refresh_token", "")
    if not access_token:
        logger.error("[cTrader callback] no access_token in response")
        if uid:
            _set_ctrader_oauth_progress(uid, "error", "no_token")
        return

    def _save_tokens():
        db = SessionLocal()
        try:
            db.execute(
                _sql_text("""
                    INSERT INTO user_preferences
                        (user_id, ctrader_access_token, ctrader_refresh_token, forex_approved)
                    VALUES (:uid, :tok, :ref, :fa)
                    ON CONFLICT (user_id) DO UPDATE SET
                        ctrader_access_token  = EXCLUDED.ctrader_access_token,
                        ctrader_refresh_token = EXCLUDED.ctrader_refresh_token,
                        forex_approved        = CASE WHEN :fa THEN TRUE
                                                    ELSE user_preferences.forex_approved END
                """),
                {"uid": user_row["id"], "tok": access_token,
                 "ref": refresh_token, "fa": user_row["is_admin"]},
            )
            db.commit()
            row = db.execute(
                _sql_text(
                    "SELECT ctrader_access_token IS NOT NULL AS has_token "
                    "FROM user_preferences WHERE user_id = :uid"
                ),
                {"uid": user_row["id"]},
            ).fetchone()
            return bool(row and row[0])
        except Exception:
            try:
                db.rollback()
            except Exception:
                pass
            raise
        finally:
            db.close()

    try:
        ok = await asyncio.to_thread(_save_tokens)
        logger.info(
            f"[cTrader callback] token UPSERT ok user_id={user_row['id']} "
            f"token_len={len(access_token)} verify_has_token={ok}"
        )
    except Exception as e:
        logger.error(f"[cTrader callback] UPSERT failed: {type(e).__name__}")
        if uid:
            _set_ctrader_oauth_progress(uid, "error", "db_error")
        return

    if uid:
        _set_ctrader_oauth_progress(uid, "ok")

    try:
        await _fetch_and_store_ctrader_accounts(
            user_row["id"], access_token, user_row["is_admin"],
            user_row["name"], user_row["uname"], user_row["tg_id"],
        )
    except Exception as e:
        logger.warning(f"[cTrader callback] accounts fetch failed: {type(e).__name__}")

    # Wake feed AFTER account_id is persisted — _list_connected_accounts needs both.
    try:
        from app.services.ctrader_price_feed import notify_account_linked
        notify_account_linked(user_row["id"])
        logger.info(
            "[cTrader callback] feed reconnect triggered uid=%s after token+account save",
            user_row["id"],
        )
    except Exception:
        pass


@app.get("/api/ctrader/oauth-progress")
async def api_ctrader_oauth_progress(uid: str = Query(...)):
    """Poll while /api/ctrader/callback finishes token exchange in background."""
    uid = (uid or "").strip().upper()
    if not uid:
        raise HTTPException(status_code=400, detail="Missing uid")
    prog = get_cache(_ctrader_oauth_cache_key(uid))
    if not prog:
        return JSONResponse({"status": "unknown"})
    return JSONResponse(prog)


@app.get("/api/ctrader/callback")
async def api_ctrader_callback(
    code:  str = Query(None),
    state: str = Query(None),
    error: str = Query(None),
    request: Request = None,
    background_tasks: BackgroundTasks = None,
):
    """
    Spotware OAuth callback. Return HTML immediately (id.ctrader.com waits for
    this response) and finish token exchange in a background task.
    """
    if error:
        q = f"ctrader_error={urllib.parse.quote(str(error))}"
        return RedirectResponse(url=_ctrader_app_redirect_url(request, state or "", q))

    if not code or not state:
        q = "ctrader_error=missing_code"
        return RedirectResponse(url=_ctrader_app_redirect_url(request, state or "", q))

    redirect_uri = _ctrader_redirect_uri(request)
    uid, _ = _parse_ctrader_oauth_state(state or "")
    if uid:
        _set_ctrader_oauth_progress(uid, "pending")
        _pin_ctrader_oauth_redirect_uri(uid, redirect_uri)

    # Start token exchange immediately (auth codes expire in ~60s).
    asyncio.create_task(
        _complete_ctrader_oauth_in_background(code, state, redirect_uri)
    )

    continue_url = _ctrader_app_redirect_url(request, state or "", "ctrader=linking")
    return HTMLResponse(_ctrader_callback_finishing_html(continue_url), status_code=200)




@app.get("/api/ctrader/status")
async def api_ctrader_status(uid: str = Query(...)):
    from app.database import SessionLocal
    from app.models import UserPreference
    db = SessionLocal()
    try:
        user = _get_user_by_uid(uid, db)
        if not user:
            raise HTTPException(status_code=403)
        import json as _json
        prefs = db.query(UserPreference).filter(UserPreference.user_id == user.id).first()
        connected = bool(prefs and prefs.ctrader_access_token)
        accounts = _json.loads(prefs.ctrader_accounts or "[]") if prefs else []
        return JSONResponse({
            "connected":  connected,
            "account_id": (prefs.ctrader_account_id or "") if prefs else "",
            "accounts":   accounts,
        })
    finally:
        db.close()


@app.get("/api/ctrader/accounts")
async def api_ctrader_accounts_list(uid: str = Query(...)):
    """Return all cTrader accounts for the user so the UI can show a picker.

    Re-fetches the LIVE account list from cTrader using the stored access token so
    that accounts created AFTER the original link (e.g. a new demo account) show up
    — previously this only returned the JSON snapshot captured at link time, so new
    accounts never appeared. Falls back to the cached snapshot if the broker is
    unreachable or the token is stale (never returns an empty picker on a transient
    failure)."""
    from app.database import SessionLocal
    from app.models import UserPreference
    import json as _json
    db = SessionLocal()
    try:
        user = _get_user_by_uid(uid, db)
        if not user:
            raise HTTPException(status_code=403)
        prefs = db.query(UserPreference).filter(UserPreference.user_id == user.id).first()
        cached = _json.loads(prefs.ctrader_accounts or "[]") if prefs else []
        selected = (prefs.ctrader_account_id or "") if prefs else ""

        fresh = None
        if prefs and prefs.ctrader_access_token:
            try:
                from app.services.ctrader_client import get_accounts_for_token
                fetched = await asyncio.wait_for(
                    get_accounts_for_token(prefs.ctrader_access_token), timeout=12.0
                )
                if fetched:
                    fresh = fetched
            except Exception as _e:
                logger.warning(f"[cTrader] live accounts refresh failed uid={uid}: {type(_e).__name__}")

        accounts = fresh if fresh is not None else cached
        if fresh is not None and _json.dumps(fresh) != (prefs.ctrader_accounts or ""):
            try:
                prefs.ctrader_accounts = _json.dumps(fresh)
                db.commit()
            except Exception:
                db.rollback()

        return JSONResponse({
            "accounts":            accounts,
            "selected_account_id": selected,
        })
    finally:
        db.close()


@app.post("/api/ctrader/select-account")
async def api_ctrader_select_account(uid: str = Query(...), request: Request = None):
    """Save which cTrader account the user wants to trade on."""
    import json as _json
    from app.database import SessionLocal
    from app.models import UserPreference
    body = await request.json()
    account_id = str(body.get("account_id", "")).strip()
    if not account_id:
        raise HTTPException(status_code=400, detail="account_id required")
    db = SessionLocal()
    try:
        user = _get_user_by_uid(uid, db)
        if not user:
            raise HTTPException(status_code=403)
        prefs = db.query(UserPreference).filter(UserPreference.user_id == user.id).first()
        if not prefs or not prefs.ctrader_access_token:
            raise HTTPException(status_code=400, detail="Not connected")
        try:
            from app.services.ctrader_client import (
                get_accounts_for_token, _invalidate_persistent_connection,
                CTRADER_HOST_LIVE, CTRADER_HOST_DEMO,
            )
            accounts = await asyncio.wait_for(
                get_accounts_for_token(prefs.ctrader_access_token), timeout=15.0,
            )
            if accounts:
                prefs.ctrader_accounts = _json.dumps(accounts)
                valid = {str(a.get("ctidTraderAccountId")) for a in accounts}
                if account_id not in valid:
                    raise HTTPException(
                        status_code=400,
                        detail="That account is not linked to your cTrader token",
                    )
        except HTTPException:
            raise
        except Exception as _ae:
            logger.warning(f"[ctrader] account refresh on select uid={uid}: {_ae}")
        prefs.ctrader_account_id = account_id
        db.commit()
        try:
            from app.services.ctrader_client import (
                _invalidate_persistent_connection, CTRADER_HOST_LIVE, CTRADER_HOST_DEMO,
            )
            for _h in (CTRADER_HOST_LIVE, CTRADER_HOST_DEMO):
                _invalidate_persistent_connection(_h, int(account_id))
        except Exception:
            pass
        try:
            from app.services.ctrader_price_feed import notify_account_linked
            notify_account_linked(user.id)
        except Exception:
            pass
        _invalidate_user_cache(uid)
        return JSONResponse({"ok": True, "account_id": account_id})
    finally:
        db.close()


@app.delete("/api/ctrader/disconnect")
async def api_ctrader_disconnect(uid: str = Query(...)):
    from app.database import SessionLocal
    from app.models import UserPreference
    db = SessionLocal()
    try:
        user = _get_user_by_uid(uid, db)
        if not user:
            raise HTTPException(status_code=403)
        prefs = db.query(UserPreference).filter(UserPreference.user_id == user.id).first()
        if prefs:
            prefs.ctrader_access_token  = None
            prefs.ctrader_refresh_token = None
            prefs.ctrader_account_id    = None
            db.commit()
        return JSONResponse({"ok": True})
    finally:
        db.close()


@app.get("/api/ctrader/feed-status")
async def api_ctrader_feed_status():
    """
    Price-feed diagnostics: cTrader live ticks (broker-matched) + FMP poll cache.
    No auth required — read-only.
    """
    out = {
        "ctrader": {"live": False, "symbol_count": 0, "cached_symbols": []},
        "fmp":     {"live": False, "symbol_count": 0, "cached_symbols": []},
        "primary_source": "yfinance_fallback",
    }
    try:
        from app.services import ctrader_price_feed as _ctf
        _st = _ctf.feed_status()
        out["ctrader"] = {
            "live":              bool(_st.get("live")),
            "subscribed":        bool(_st.get("subscribed")),
            "remote_feed":       bool(_st.get("remote_feed")),
            "remote_live":       bool(_st.get("remote_live")),
            "local_subscribed":  bool(_st.get("local_subscribed")),
            "symbol_count":      int(_st.get("symbol_count") or 0),
            "cached_symbols":    list(_st.get("cached_symbols") or [])[:30],
            "symbols_seen":      int(_st.get("symbols_seen") or 0),
            "last_tick_age_s":   _st.get("last_tick_age_s"),
            "forex_market_open": _st.get("forex_market_open"),
            "last_auth_error":   _st.get("last_auth_error"),
            "needs_relink":      bool(_st.get("needs_relink")),
            "auth_backoff_s":    _st.get("auth_backoff_s"),
            "note":              _st.get("note"),
        }
    except Exception as e:
        out["ctrader"]["error"] = str(e)

    try:
        from app.services.fmp_price_feed import (
            is_live as _fmp_live,
            cached_symbols as _fmp_syms,
            symbol_count as _fmp_count,
            fmp_in_backoff as _fmp_backoff,
            fmp_backoff_remaining_seconds as _fmp_backoff_s,
            _fmp_api_key,
        )
        _f_syms = _fmp_syms()
        out["fmp"] = {
            "live":           bool(_fmp_live()),
            "symbol_count":   _fmp_count(),
            "cached_symbols": _f_syms[:30],
            "rate_limited":   bool(_fmp_backoff()),
            "backoff_seconds": _fmp_backoff_s(),
            "api_key_set":    bool(_fmp_api_key()),
        }
    except Exception as e:
        out["fmp"]["error"] = str(e)

    try:
        from app.services.spot_price_store import snapshot as _spot_snap, get_mid as _spot_mid
        shared = _spot_snap(max_age_s=20.0)
        out["shared_ticks"] = shared
        out["samples"] = {
            s: _spot_mid(s)
            for s in ("EURUSD", "GBPUSD", "XAUUSD", "NAS100", "US30")
        }
        shared_n = int(shared.get("symbol_count") or 0)
        if shared_n > 0:
            out["symbol_count"] = shared_n
            out["cached_symbols"] = list(shared.get("symbols") or [])[:30]
            by_src = shared.get("by_source") or {}
            if by_src.get("ctrader"):
                out["primary_source"] = "ctrader_realtime"
            elif by_src.get("fmp"):
                out["primary_source"] = "fmp_realtime"
            elif by_src.get("binance") or by_src.get("coinbase") or by_src.get("kraken"):
                out["primary_source"] = "metals_realtime"
            out["live"] = True
    except Exception as e:
        out["shared_ticks"] = {"error": str(e)[:120]}

    try:
        from app.services.metals_spot_feed import (
            is_live as _metals_live,
            cached_symbols as _metals_syms,
            symbol_count as _metals_count,
            get_price as _metals_px,
        )
        out["metals"] = {
            "live": bool(_metals_live()),
            "symbol_count": _metals_count(),
            "cached_symbols": _metals_syms(),
            "xauusd": _metals_px("XAUUSD"),
            "xagusd": _metals_px("XAGUSD"),
        }
    except Exception as e:
        out["metals"] = {"error": str(e)[:120]}

    if out.get("primary_source") == "yfinance_fallback":
        if out["ctrader"]["symbol_count"] > 0:
            out["primary_source"] = "ctrader_realtime"
        elif out["ctrader"].get("subscribed"):
            out["primary_source"] = (
                "ctrader_subscribed" if out["ctrader"].get("forex_market_open") else "ctrader_market_closed"
            )
        elif out["fmp"]["symbol_count"] > 0:
            out["primary_source"] = "fmp_realtime"

    if "symbol_count" not in out or not out.get("symbol_count"):
        out["live"] = out["fmp"]["live"] or out["ctrader"]["live"]
        out["cached_symbols"] = out["ctrader"]["cached_symbols"] or out["fmp"]["cached_symbols"]
        out["symbol_count"] = out["ctrader"]["symbol_count"] + out["fmp"]["symbol_count"]
    out["source"] = out.get("primary_source", "yfinance_fallback")
    return JSONResponse(out)


@app.get("/api/live-forex/account")
async def api_live_forex_account(uid: str = Query(...), refresh: bool = Query(False)):
    """
    Returns live cTrader account data (balance, equity, open positions)
    plus connection/approval state for the Live Forex tab.
    """
    import json as _json
    _uid_key = (uid or "").strip().upper()
    _cache_key = f"live_forex_acct:{_uid_key}"
    if not refresh:
        _cached = get_cache(_cache_key)
        if isinstance(_cached, dict):
            return JSONResponse({**_cached, "cached": True})

    def _load_db_snapshot():
        from app.database import SessionLocal
        from app.models import UserPreference
        from sqlalchemy import text as _t
        db = SessionLocal()
        try:
            db.execute(_t("SET LOCAL statement_timeout = '8000'"))
            user = _get_user_by_uid(uid, db)
            if not user:
                return None
            prefs = db.query(UserPreference).filter(UserPreference.user_id == user.id).first()
            connected = bool(prefs and prefs.ctrader_access_token)
            forex_approved = bool(getattr(prefs, "forex_approved", False)) if prefs else False
            if not forex_approved and getattr(user, "is_admin", False):
                forex_approved = True
                if prefs:
                    prefs.forex_approved = True
                    db.commit()
            accounts = _json.loads(prefs.ctrader_accounts or "[]") if prefs else []
            positions = db.execute(_t("""
                SELECT e.id, e.symbol, e.direction, e.entry_price,
                       e.tp_price AS tp1_price, e.sl_price, e.pnl_pct,
                       e.fired_at, s.name AS strategy_name, e.asset_class
                FROM strategy_executions e
                JOIN user_strategies s ON s.id = e.strategy_id
                WHERE s.user_id = :uid
                  AND e.outcome = 'OPEN' AND e.is_paper = false
                  AND e.asset_class IN ('forex', 'index', 'metals', 'commodity')
                ORDER BY e.fired_at DESC LIMIT 20
            """), {"uid": user.id}).fetchall()
            live_rows = db.execute(_t("""
                SELECT s.id, s.name, s.status, s.asset_class, s.config,
                       COUNT(e.id) FILTER (WHERE e.outcome='OPEN' AND e.is_paper=false) AS open_count,
                       COUNT(e.id) FILTER (
                           WHERE e.outcome IN ('WIN','LOSS','BREAKEVEN') AND e.is_paper=false
                       ) AS closed_count,
                       COUNT(e.id) FILTER (WHERE e.outcome='WIN' AND e.is_paper=false) AS win_count,
                       COALESCE(SUM(e.pnl_pct) FILTER (
                           WHERE e.outcome IN ('WIN','LOSS','BREAKEVEN') AND e.is_paper=false
                       ), 0) AS total_pnl_pct
                FROM user_strategies s
                LEFT JOIN strategy_executions e ON e.strategy_id = s.id
                WHERE s.user_id = :uid
                  AND s.status IN ('active','paused')
                  AND s.asset_class IN ('forex','index','metals','commodity')
                GROUP BY s.id
                ORDER BY (s.status='active') DESC, s.updated_at DESC
                LIMIT 30
            """), {"uid": user.id}).fetchall()
            return {
                "user": user,
                "prefs": prefs,
                "connected": connected,
                "forex_approved": forex_approved,
                "accounts": accounts,
                "positions": [dict(r._mapping) for r in positions],
                "live_rows": [dict(r._mapping) for r in live_rows],
            }
        finally:
            db.close()

    try:
        snap = await asyncio.wait_for(asyncio.to_thread(_load_db_snapshot), timeout=12.0)
    except asyncio.TimeoutError:
        stale = get_cache(_cache_key)
        if isinstance(stale, dict):
            return JSONResponse({**stale, "cached": True, "stale": True})
        raise HTTPException(status_code=503, detail="Database busy — retry in a moment")
    if snap is None:
        raise HTTPException(status_code=403, detail="Invalid UID")

    user = snap["user"]
    prefs = snap["prefs"]
    if not snap["connected"]:
        payload = {
            "connected": False,
            "forex_approved": snap["forex_approved"],
            "accounts": snap["accounts"],
            "balance": None,
            "equity": None,
        }
        set_cache(_cache_key, payload, ttl_seconds=15)
        return JSONResponse(payload)

    balance = None
    accounts = snap["accounts"]

    async def _sync_ctrader_accounts():
        nonlocal accounts
        if not (prefs and prefs.ctrader_access_token):
            return
        need = (not accounts) or not (prefs.ctrader_account_id or "").strip()
        if not need:
            return
        from app.database import SessionLocal
        from app.models import UserPreference
        try:
            from app.services.ctrader_client import get_accounts_for_token
            fetched = await asyncio.wait_for(
                get_accounts_for_token(prefs.ctrader_access_token), timeout=6.0,
            )
            if not fetched:
                return
            accounts = fetched
            db2 = SessionLocal()
            try:
                p2 = db2.query(UserPreference).filter(UserPreference.user_id == user.id).first()
                if p2:
                    p2.ctrader_accounts = _json.dumps(fetched)
                    if not (p2.ctrader_account_id or "").strip():
                        chosen = _ctrader_auto_pick_account(fetched)
                        if chosen:
                            p2.ctrader_account_id = str(chosen["ctidTraderAccountId"])
                    db2.commit()
                    prefs.ctrader_account_id = p2.ctrader_account_id
            finally:
                db2.close()
        except Exception as _ae:
            logger.warning(f"[live-forex] account sync failed uid={uid}: {type(_ae).__name__}")

    async def _fetch_balance():
        nonlocal balance
        _acct_id = (prefs.ctrader_account_id or "").strip() if prefs else ""
        if not _acct_id:
            return
        _bal_key = f"ctrader_balance:{user.id}"
        _cached = get_cache(_bal_key)
        if _cached == "__miss__":
            return
        if _cached is not None:
            balance = _cached
            return
        try:
            from app.services.ctrader_client import get_account_balance_resilient
            balance = await asyncio.wait_for(
                get_account_balance_resilient(
                    prefs.ctrader_access_token,
                    int(_acct_id),
                    prefs=prefs,
                    user_id=user.id,
                ),
                timeout=8.0,
            )
            if balance is not None:
                set_cache(_bal_key, balance, ttl_seconds=60)
            else:
                set_cache(_bal_key, "__miss__", ttl_seconds=30)
        except Exception as _be:
            logger.warning(
                f"[live-forex] balance fetch uid={uid} ctid={_acct_id}: "
                f"{type(_be).__name__}"
            )
            set_cache(_bal_key, "__miss__", ttl_seconds=30)

    try:
        await asyncio.wait_for(
            asyncio.gather(_sync_ctrader_accounts(), _fetch_balance(), return_exceptions=True),
            timeout=9.0,
        )
    except asyncio.TimeoutError:
        _stale_bal = get_cache(f"ctrader_balance:{user.id}")
        if isinstance(_stale_bal, (int, float)):
            balance = _stale_bal

    positions = snap["positions"]
    for p in positions:
        if p.get("fired_at"):
            p["fired_at"] = p["fired_at"].isoformat()
        try:
            from app.services.ctrader_price_feed import get_price as _mkt_px
            from app.services.forex_engine import pip_size as _pip_sz
            sym = (p.get("symbol") or "").upper()
            entry = float(p.get("entry_price") or 0)
            if sym and entry > 0:
                mark = _mkt_px(sym)
                if mark:
                    p["mark_price"] = round(mark, 6)
                    pip = max(_pip_sz(sym), 1e-10)
                    if p.get("direction") == "LONG":
                        p["unrealized_pips"] = round((mark - entry) / pip, 1)
                    else:
                        p["unrealized_pips"] = round((entry - mark) / pip, 1)
        except Exception:
            pass

    live_strategies = []
    for s in snap["live_rows"]:
        cfg = s.get("config") or {}
        if isinstance(cfg, str):
            try:
                cfg = _json.loads(cfg)
            except Exception:
                cfg = {}
        risk = cfg.get("risk") or {}
        uni = cfg.get("universe") or {}
        syms = (uni.get("symbols") or cfg.get("tradfi_symbols") or cfg.get("symbols") or [])
        if isinstance(syms, str):
            syms = [syms]
        _tf = cfg.get("timeframe") or cfg.get("_timeframe")
        if not _tf:
            _conds = (cfg.get("entry_conditions") or {}).get("conditions") or []
            if _conds and isinstance(_conds[0], dict):
                _tf = _conds[0].get("timeframe")
        pst = (risk.get("position_size_type") or "pct").lower()
        if pst == "lots":
            size_value = float(risk.get("position_size_lots", 0.1) or 0.1)
            size_label = f"{size_value:g} lots"
        elif pst == "fixed":
            size_value = float(risk.get("position_size_usd", 50) or 50)
            size_label = f"${size_value:g} fixed"
        elif risk.get("use_risk_pct"):
            size_value = float(risk.get("risk_pct_per_trade", 1) or 1)
            size_label = f"{size_value:g}% risk"
            pst = "risk_pct"
        else:
            size_value = float(risk.get("position_size_pct", 5) or 5)
            size_label = f"{size_value:g}% balance"
            pst = "pct"
        _closed = int(s["closed_count"] or 0)
        _wins = int(s["win_count"] or 0)
        live_strategies.append({
            "id": s["id"], "name": s["name"], "status": s["status"],
            "asset_class": s["asset_class"],
            "symbols": [str(x).upper() for x in syms][:8],
            "open_count": int(s["open_count"] or 0),
            "closed_count": _closed, "win_count": _wins,
            "win_rate": round(_wins / _closed * 100, 1) if _closed else None,
            "total_pnl_pct": round(float(s["total_pnl_pct"] or 0), 2),
            "size_type": pst, "size_value": round(size_value, 4),
            "size_label": size_label, "timeframe": _tf or "—",
        })

    feed_info: Dict[str, object] = {}
    try:
        from app.services import ctrader_price_feed as _ctf
        feed_info = _ctf.feed_status()
    except Exception:
        pass

    exec_ready, exec_blockers = _user_ctrader_live_ready(prefs, user)
    payload = {
        "connected": True,
        "forex_approved": snap["forex_approved"],
        "account_id": (prefs.ctrader_account_id or "") if prefs else "",
        "accounts": accounts,
        "balance": round(balance, 2) if balance is not None else None,
        "equity": round(balance, 2) if balance is not None else None,
        "open_positions": positions,
        "live_strategies": live_strategies,
        "price_feed": feed_info,
        "execution_ready": exec_ready,
        "execution_blockers": exec_blockers,
    }
    set_cache(_cache_key, payload, ttl_seconds=12)
    return JSONResponse(payload)


async def _build_executor_diagnostics(
    db,
    user,
    *,
    run_heal: bool = False,
    run_deep: bool = False,
) -> dict:
    """
    Executor diagnostics payload. Default is fast (<3s) for large accounts.
    Heavy work (heal, klines, TA probe) is opt-in — the full endpoint used to
    run heal + N× DB gate lookups per strategy and caused Cloudflare 524 timeouts.
    """
    import time as _time
    from datetime import datetime, timedelta
    from app.deployment import is_production_deploy
    from app.strategy_models import UserStrategy, StrategyExecution
    from app.services.strategy_heal import resolve_strategy_asset_class as _resolve_ac

    heal_stats: dict = {"skipped": True, "hint": "POST /api/strategies/ensure-firing"}
    if run_heal:
        heal_stats = {}
        try:
            from app.services.strategy_heal import (
                heal_user_account,
                expire_untracked_forex_opens_when_broker_empty,
            )
            heal_stats = await asyncio.wait_for(
                asyncio.to_thread(heal_user_account, db, user.id),
                timeout=5.0,
            )
            heal_stats["orphan_forex_expired"] = await asyncio.wait_for(
                expire_untracked_forex_opens_when_broker_empty(
                    min_age_minutes=30,
                    user_id=user.id,
                ),
                timeout=8.0,
            )
        except Exception as _he:
            heal_stats = {"error": str(_he)[:200]}

    strategies = db.query(UserStrategy).filter(UserStrategy.user_id == user.id).all()
    by_status: dict = {}
    by_class: dict = {}
    tradfi_scanning = 0
    draft_count = 0
    active_ids: list = []
    for s in strategies:
        by_status[s.status] = by_status.get(s.status, 0) + 1
        ac = _resolve_ac(s).lower()
        by_class[ac] = by_class.get(ac, 0) + 1
        if s.status in ("paper", "active") and ac in ("forex", "index", "stock"):
            tradfi_scanning += 1
        if s.status in ("draft", "paused"):
            draft_count += 1
        if s.status in ("paper", "active"):
            active_ids.append(s.id)

    since = datetime.utcnow() - timedelta(hours=24)
    fires_24h = db.query(StrategyExecution).filter(
        StrategyExecution.user_id == user.id,
        StrategyExecution.fired_at >= since,
    ).count()
    open_rows = db.query(StrategyExecution).filter(
        StrategyExecution.user_id == user.id,
        StrategyExecution.outcome == "OPEN",
    ).count()

    from app.services.ctrader_order_queue import get_gate_stats_bulk
    gate_by_id = get_gate_stats_bulk(active_ids)
    gate_agg: dict = {}
    strategy_rows = []
    for s in strategies:
        if s.id not in gate_by_id:
            continue
        gs_entry = gate_by_id[s.id]
        gs = gs_entry.get("stats") or {}
        for k, v in gs.items():
            gate_agg[k] = gate_agg.get(k, 0) + int(v)
        if gs:
            top = max(gs.items(), key=lambda kv: kv[1])
            strategy_rows.append({
                "id": s.id,
                "name": s.name,
                "status": s.status,
                "asset_class": s.asset_class,
                "top_blocker": top[0].replace("blk_", ""),
                "top_count": top[1],
                "updated_at": gs_entry.get("updated_at"),
            })
    strategy_rows.sort(key=lambda r: r.get("top_count", 0), reverse=True)

    top_blockers = sorted(
        [{"gate": k.replace("blk_", ""), "count": v} for k, v in gate_agg.items()],
        key=lambda x: -x["count"],
    )[:12]

    hb: dict = {}
    try:
        from app.services.strategy_executor import get_heartbeats
        hb = get_heartbeats()
    except Exception:
        pass
    _now = _time.time()
    fx_age = (_now - hb["forex_executor"]) if hb.get("forex_executor") else None

    klines_probe: dict = {"skipped": True}
    probe: dict = {"skipped": True}
    if run_deep:
        klines_probe = {}
        try:
            from app.services.tradfi_prices import get_klines as _gk
            _bars = await asyncio.wait_for(
                _gk("EURUSD", "forex", "15m", 50, ctrader_user_id=user.id, max_wait_s=6.0),
                timeout=8.0,
            )
            klines_probe = {
                "eurusd_15m_bars": len(_bars or []),
                "ok": len(_bars or []) >= 20,
            }
        except Exception as _ke:
            klines_probe = {"ok": False, "error": str(_ke)[:160]}

        probe = {}
        try:
            import httpx
            from app.services.strategy_executor import _fetch_price_and_ta, _primary_timeframe
            from app.services.strategy_ta import evaluate_strategy_conditions
            from app.services.strategy_heal import resolve_strategy_asset_class

            pick = None
            for s in strategies:
                if s.status not in ("paper", "active"):
                    continue
                ac = resolve_strategy_asset_class(s)
                if ac not in ("forex", "index", "stock"):
                    continue
                syms = ((s.config or {}).get("universe") or {}).get("symbols") or []
                if syms:
                    pick = s
                    break
            if pick:
                cfg = dict(pick.config or {})
                ac = resolve_strategy_asset_class(pick)
                _syms = ((cfg.get("universe") or {}).get("symbols") or [])
                sym = str(_syms[0]).upper()
                async with httpx.AsyncClient(timeout=8.0) as _hc:
                    px = await _fetch_price_and_ta(
                        sym, _hc, ac, user_id=user.id,
                        timeframe=_primary_timeframe(cfg),
                    )
                if px:
                    passed, details = await evaluate_strategy_conditions(
                        cfg, sym, px, px.get("enhanced_ta", {}), _hc,
                        ctrader_user_id=user.id,
                    )
                    probe = {
                        "strategy_id": pick.id,
                        "strategy_name": pick.name,
                        "symbol": sym,
                        "would_fire": passed,
                        "conditions": details[:8],
                    }
                else:
                    probe = {
                        "strategy_id": pick.id,
                        "strategy_name": pick.name,
                        "symbol": sym,
                        "error": "no_price_data",
                    }
        except Exception as _pe:
            probe = {"error": str(_pe)[:200]}

    telegram_status: dict = {}
    if run_deep:
        try:
            from app.services.telegram_alert_status import build_telegram_alert_status
            telegram_status = await asyncio.wait_for(
                build_telegram_alert_status(user, db),
                timeout=4.0,
            )
        except Exception:
            pass

    notes = []
    if draft_count:
        notes.append(
            f"{draft_count} strateg{'y' if draft_count == 1 else 'ies'} still draft/paused "
            "— use Settings → Repair strategies or POST /api/strategies/ensure-firing."
        )
    if open_rows:
        notes.append(
            f"{open_rows} OPEN execution(s) — may block max_open_positions until TP/SL or auto-expire."
        )
    if not fires_24h and tradfi_scanning:
        top = top_blockers[0]["gate"] if top_blockers else "unknown"
        notes.append(
            f"Executor is scanning {tradfi_scanning} tradfi strateg"
            f"{'y' if tradfi_scanning == 1 else 'ies'} but 0 fires in 24h. "
            f"Top blocker: {top}."
        )
    if telegram_status.get("blockers"):
        notes.append("Telegram: " + "; ".join(telegram_status["blockers"][:3]))
    if not run_deep:
        notes.append(
            "Fast diagnostics — add ?deep=1 for klines/condition probe, "
            "?heal=1 to run repair (or use Settings → Repair strategies)."
        )

    return {
        "ok": True,
        "heal": heal_stats,
        "executor": {
            "production": is_production_deploy(),
            "forex_executor_age_sec": round(fx_age, 1) if fx_age is not None else None,
            "forex_executor_alive": fx_age is not None and fx_age < 120,
        },
        "strategies": {
            "total": len(strategies),
            "by_status": by_status,
            "by_asset_class": by_class,
            "tradfi_scanning": tradfi_scanning,
        },
        "executions": {"fires_24h": fires_24h, "open": open_rows},
        "top_blockers": top_blockers,
        "strategies_with_gates": strategy_rows[:15],
        "klines_probe": klines_probe,
        "condition_probe": probe,
        "telegram": telegram_status,
        "notes": notes,
        "ensure_firing_url": "/api/strategies/ensure-firing",
    }


@app.get("/api/executor/diagnostics/quick")
async def api_executor_diagnostics_quick(uid: str = Query(...)):
    """Lightweight diagnostics — safe on large accounts; avoids Cloudflare 524."""
    from app.database import SessionLocal

    db = SessionLocal()
    try:
        user = _get_user_by_uid(uid, db)
        if not user:
            raise HTTPException(status_code=403, detail="Invalid UID")
        payload = await asyncio.wait_for(
            _build_executor_diagnostics(db, user, run_heal=False, run_deep=False),
            timeout=12.0,
        )
        return JSONResponse(payload)
    except asyncio.TimeoutError:
        return JSONResponse(
            {"ok": False, "error": "diagnostics timeout — try again in 30s"},
            status_code=504,
        )
    finally:
        db.close()


@app.get("/api/executor/diagnostics")
async def api_executor_diagnostics(
    uid: str = Query(...),
    heal: bool = Query(False, description="Run strategy heal (slow on large accounts)"),
    deep: bool = Query(False, description="Run klines + live condition probe"),
):
    """
    Why strategies are or aren't firing — executor health and gate blockers.
    Default is fast. Use ?deep=1 for klines/TA probe; ?heal=1 or ensure-firing for repair.
    """
    from app.database import SessionLocal

    db = SessionLocal()
    try:
        user = _get_user_by_uid(uid, db)
        if not user:
            raise HTTPException(status_code=403, detail="Invalid UID")
        _budget = 25.0 if deep or heal else 12.0
        payload = await asyncio.wait_for(
            _build_executor_diagnostics(db, user, run_heal=heal, run_deep=deep),
            timeout=_budget,
        )
        return JSONResponse(payload)
    except asyncio.TimeoutError:
        return JSONResponse(
            {
                "ok": False,
                "error": "diagnostics timeout",
                "quick_url": f"/api/executor/diagnostics/quick?uid={uid}",
            },
            status_code=504,
        )
    finally:
        db.close()


@app.get("/api/live-forex/readiness")
async def api_live_forex_readiness(uid: str = Query(...)):
    """Diagnostics for the live forex signal → cTrader order pipeline."""
    import time as _time
    from app.database import SessionLocal
    from app.models import UserPreference
    from app.deployment import is_production_deploy

    db = SessionLocal()
    try:
        user = _get_user_by_uid(uid, db)
        if not user:
            raise HTTPException(status_code=403, detail="Invalid UID")
        prefs = db.query(UserPreference).filter(UserPreference.user_id == user.id).first()
        ready, blockers = _user_ctrader_live_ready(prefs, user)

        from app.strategy_models import UserStrategy
        active_forex = db.query(UserStrategy).filter(
            UserStrategy.user_id == user.id,
            UserStrategy.status == "active",
            UserStrategy.asset_class.in_(["forex", "index", "metals", "commodity"]),
        ).count()

        _sub = _get_portal_sub(user.id, db)
        pro_ok = _is_portal_pro(_sub) or bool(getattr(user, "is_admin", False))

        hb: dict = {}
        try:
            from app.services.strategy_executor import get_heartbeats
            hb = get_heartbeats()
        except Exception:
            pass
        _now = _time.time()
        _fx_age = (_now - hb["forex_executor"]) if hb.get("forex_executor") else None
        _mgr_age = (_now - hb["forex_live_manager"]) if hb.get("forex_live_manager") else None

        feed_info: dict = {}
        try:
            from app.services import ctrader_price_feed as _ctf
            feed_info = _ctf.feed_status()
        except Exception:
            feed_info = {}

        try:
            from app.services.ctrader_order_queue import ctrader_order_worker_running
            queue_worker = ctrader_order_worker_running()
        except Exception:
            queue_worker = False

        pipeline_notes = [
            "Structure scanner (/api/forex/scanner) is discovery-only — live orders "
            "fire when your active strategy's entry conditions match in the forex executor.",
        ]
        if not pro_ok:
            blockers = list(dict.fromkeys(blockers + ["pro_subscription_required"]))
        if not is_production_deploy():
            pipeline_notes.append(
                "Executor runs on production only — dev/staging will not place live orders."
            )

        paper_scanning = db.query(UserStrategy).filter(
            UserStrategy.user_id == user.id,
            UserStrategy.status.in_(["paper", "active"]),
        ).count()

        telegram_status: dict = {}
        try:
            from app.services.telegram_alert_status import build_telegram_alert_status
            telegram_status = await build_telegram_alert_status(user, db)
        except Exception:
            pass

        return JSONResponse({
            "execution_ready": ready and pro_ok,
            "blockers": blockers + ([] if pro_ok else ["pro_subscription_required"]),
            "blocker_messages": {
                k: _CTRADER_BLOCKER_MESSAGES.get(k, k)
                for k in (blockers + ([] if pro_ok else ["pro_subscription_required"]))
            },
            "connected": bool(prefs and prefs.ctrader_access_token),
            "forex_approved": bool(getattr(prefs, "forex_approved", False)) or bool(getattr(user, "is_admin", False)),
            "account_id": (prefs.ctrader_account_id or "") if prefs else "",
            "pro_subscription": pro_ok,
            "active_live_strategies": active_forex,
            "scanning_strategies": paper_scanning,
            "telegram": telegram_status,
            "executor": {
                "production_deploy": is_production_deploy(),
                "forex_executor_age_sec": round(_fx_age, 1) if _fx_age is not None else None,
                "forex_live_manager_age_sec": round(_mgr_age, 1) if _mgr_age is not None else None,
                "forex_executor_alive": _fx_age is not None and _fx_age < 120,
            },
            "order_queue_worker_started": queue_worker,
            "price_feed": feed_info,
            "pipeline_notes": pipeline_notes,
            "test_trade_url": f"/api/ctrader/test-trade?uid={uid}&symbol=EURUSD",
        })
    finally:
        db.close()


@app.get("/api/telegram/alert-status")
async def api_telegram_alert_status(uid: str = Query(...)):
    """Why trade Telegram alerts may or may not be arriving."""
    from app.database import SessionLocal
    from app.services.telegram_alert_status import build_telegram_alert_status

    db = SessionLocal()
    try:
        user = _get_user_by_uid(uid, db)
        if not user:
            raise HTTPException(status_code=403)
        return JSONResponse(await build_telegram_alert_status(user, db))
    finally:
        db.close()


@app.post("/api/telegram/test-alert")
async def api_telegram_test_alert(request: Request):
    """Send a sample paper-trade alert DM to the logged-in user."""
    body = await request.json()
    uid = (body.get("uid") or "").strip()
    if not uid:
        raise HTTPException(status_code=400, detail="uid required")
    from app.database import SessionLocal
    from app.services.strategy_executor import _telegram_int_id, _tg_send

    db = SessionLocal()
    try:
        user = _get_user_by_uid(uid, db)
        if not user or user.banned:
            raise HTTPException(status_code=403)
        tg_id = _telegram_int_id(user)
        if not tg_id:
            raise HTTPException(
                status_code=400,
                detail="Telegram not linked — log in with Telegram or link your account in Settings.",
            )
        ok = await _tg_send(
            tg_id,
            "🧪 <b>TradeHub test alert</b>\n\n"
            "If you see this, paper + live trade notifications can reach you.\n"
            "Make sure <b>Paper alerts</b> and <b>Live alerts</b> are on in Settings.",
            asset_class="forex",
        )
        if not ok:
            raise HTTPException(
                status_code=502,
                detail=(
                    "Telegram rejected the message. Open @AISIGNALPERPBOT and "
                    "@TradehubStrategyBot and tap /start on each, then retry."
                ),
            )
        return JSONResponse({"ok": True, "delivered": True, "telegram_id": tg_id})
    finally:
        db.close()


@app.post("/api/ctrader/test-trade")
async def api_ctrader_test_trade(
    uid: str = Query(...),
    symbol: str = Query("EURUSD"),
):
    """Place a tiny REAL test order on the user's cTrader account and immediately
    close it — a one-tap end-to-end check that live trading actually works
    (auth → symbol resolution → volume sizing → fill → close) without waiting for
    a strategy signal.

    Query ``symbol``: EURUSD (default), NAS100, SPX500, US30 — forex uses 0.01
    lots; indices use 1 contract. No SL/TP on the test round-trip.
    """
    from app.database import SessionLocal
    from app.models import UserPreference
    db = SessionLocal()
    try:
        user = _get_user_by_uid(uid, db)
        if not user:
            raise HTTPException(status_code=403, detail="Invalid UID")
        prefs = db.query(UserPreference).filter(UserPreference.user_id == user.id).first()
        if not prefs or not prefs.ctrader_access_token or not prefs.ctrader_account_id:
            return JSONResponse(
                {"success": False, "error": "cTrader account not connected"},
                status_code=400,
            )
        access_token = prefs.ctrader_access_token
        ctid = int(prefs.ctrader_account_id)
        _prefs_snap = prefs
        _user_id = user.id
    finally:
        db.close()

    from app.services.ctrader_client import (
        place_market_order_resilient, close_position, _host_for_account,
    )
    from app.services.index_symbols import normalize_index_symbol, is_index_symbol

    TEST_SYMBOL = (symbol or "EURUSD").upper().strip()
    is_index = is_index_symbol(TEST_SYMBOL)
    if is_index:
        TEST_SYMBOL = normalize_index_symbol(TEST_SYMBOL)
    TEST_LOTS = 0.01  # forex minimum
    TEST_CONTRACTS = 1
    host = _host_for_account(_prefs_snap, ctid)

    try:
        placed = await asyncio.wait_for(
            place_market_order_resilient(
                user_id=_user_id,
                access_token=access_token,
                ctid=ctid,
                prefs=_prefs_snap,
                symbol_name=TEST_SYMBOL,
                direction="LONG",
                volume_units=TEST_CONTRACTS if is_index else None,
                volume_lots=TEST_LOTS if not is_index else None,
            ),
            timeout=30.0,
        )
    except asyncio.TimeoutError:
        # AMBIGUOUS: the order may have reached the broker even though we timed out
        # waiting for the response — tell the user to verify in cTrader.
        return JSONResponse(
            {"success": False, "check_open_positions": True,
             "error": "cTrader did not respond in time — the order MAY have gone through"},
            status_code=504,
        )
    except Exception as e:
        logger.error(f"[test-trade] place failed uid={uid}: {type(e).__name__}: {e}")
        return JSONResponse(
            {"success": False, "check_open_positions": True, "error": "order placement failed"},
            status_code=500,
        )

    if not placed or placed.get("error") or not placed.get("order_id"):
        _perr = (placed or {}).get("error") or "order was not filled"
        _hint = None
        if "account auth failed" in _perr.lower():
            _hint = (
                "Try switching to the correct account in the dropdown above "
                "(demo vs live use different servers), then run the test again. "
                "If it still fails, disconnect and reconnect cTrader."
            )
        # A clean broker rejection (e.g. NOT_ENOUGH_MONEY) means NO position exists.
        # But "no execution event"/"unexpected exit" are ambiguous — the order could
        # have filled while we lost the response; ask the user to check cTrader.
        _ambiguous = any(t in _perr.lower() for t in ("no execution event", "unexpected exit", "timeout"))
        return JSONResponse(
            {"success": False, "stage": "place", "error": _perr, "hint": _hint,
             "check_open_positions": _ambiguous},
            status_code=200,
        )

    fill        = placed.get("actual_fill")
    position_id = placed.get("position_id")
    volume      = placed.get("volume")

    # Round-trip: close the test position so nothing is left open.
    closed = False
    close_error = None
    if position_id and volume:
        try:
            closed = await asyncio.wait_for(
                close_position(access_token, ctid, int(position_id), int(volume), host=host),
                timeout=20.0,
            )
        except Exception as e:
            close_error = f"{type(e).__name__}"
            logger.warning(f"[test-trade] close failed uid={uid}: {e}")
        if not closed and not close_error:
            close_error = "close request not confirmed"

    return JSONResponse({
        "success": True,
        "symbol": TEST_SYMBOL,
        "asset_class": "index" if is_index else "forex",
        "lots": TEST_LOTS if not is_index else None,
        "contracts": TEST_CONTRACTS if is_index else None,
        "order_id": placed.get("order_id"),
        "fill": fill,
        "closed": closed,
        "close_error": close_error,
    })


@app.post("/api/live-forex/approve")
async def api_live_forex_approve(uid: str = Query(...), request: Request = None):
    """Admin-only endpoint to approve a user for live forex trading."""
    from app.database import SessionLocal
    from app.models import UserPreference
    body = await request.json()
    admin_uid = (body.get("admin_uid") or "").strip()

    def _do():
        db = SessionLocal()
        try:
            admin = _get_user_by_uid(admin_uid, db)
            if not admin or not getattr(admin, "is_admin", False):
                return None
            target = _get_user_by_uid(uid, db)
            if not target:
                return False
            prefs = db.query(UserPreference).filter(UserPreference.user_id == target.id).first()
            if not prefs:
                prefs = UserPreference(user_id=target.id)
                db.add(prefs)
            prefs.forex_approved = True
            db.commit()
            return True
        finally:
            db.close()

    result = await asyncio.to_thread(_do)
    if result is None:
        raise HTTPException(status_code=403, detail="Admin access required")
    if result is False:
        raise HTTPException(status_code=404, detail="User not found")
    return JSONResponse({"ok": True})


@app.put("/api/settings")
async def api_put_settings(request: Request, uid: str = Query(...)):
    """Update user settings across UserPreference + StrategyPortalSettings."""
    from app.database import SessionLocal
    db = SessionLocal()
    try:
        user = _get_user_by_uid(uid, db)
        if not user:
            raise HTTPException(status_code=403)
        body = await request.json()
        from app.models import UserPreference
        from app.strategy_models import StrategyPortalSettings

        prefs = db.query(UserPreference).filter(UserPreference.user_id == user.id).first()
        if not prefs:
            prefs = UserPreference(user_id=user.id)
            db.add(prefs)

        portal = db.query(StrategyPortalSettings).filter(StrategyPortalSettings.user_id == user.id).first()
        if not portal:
            portal = StrategyPortalSettings(user_id=user.id)
            db.add(portal)

        # Capture Bitunix UID before the pref_map loop mutates it
        _old_buid = prefs.bitunix_uid or ""

        # UserPreference fields
        pref_map = {
            "position_size_percent":  "position_size_percent",
            "max_positions":          "max_positions",
            "daily_loss_limit":       "daily_loss_limit",
            "max_drawdown_percent":   "max_drawdown_percent",
            "use_trailing_stop":      "use_trailing_stop",
            "trailing_stop_percent":  "trailing_stop_percent",
            "use_breakeven_stop":     "use_breakeven_stop",
            "dm_alerts":              "dm_alerts",
            "max_consecutive_losses": "max_consecutive_losses",
            "cooldown_after_loss":    "cooldown_after_loss",
            "bitunix_uid":            "bitunix_uid",
        }
        for k, attr in pref_map.items():
            if k in body:
                setattr(prefs, attr, body[k])

        # Push notification prefs — strict coercion + range guards.
        # Booleans accept actual bools; numeric threshold must be finite & >= 0.
        import math as _math2
        if "push_notify_paper" in body:
            prefs.push_notify_paper = bool(body["push_notify_paper"])
        if "push_notify_live" in body:
            prefs.push_notify_live = bool(body["push_notify_live"])
        if "push_min_position_usd" in body:
            try:
                _v = float(body["push_min_position_usd"])
            except (TypeError, ValueError):
                raise HTTPException(status_code=400, detail="push_min_position_usd must be numeric")
            if not _math2.isfinite(_v) or _v < 0 or _v > 1_000_000:
                raise HTTPException(status_code=400, detail="push_min_position_usd must be a finite value 0-1_000_000")
            prefs.push_min_position_usd = _v

        # API keys — only overwrite if non-empty strings are provided
        api_key    = body.get("bitunix_api_key", "").strip()
        api_secret = body.get("bitunix_api_secret", "").strip()
        if api_key or api_secret:
            from app.utils.encryption import encrypt_api_key
        if api_key:
            prefs.bitunix_api_key = encrypt_api_key(api_key)
        if api_secret:
            prefs.bitunix_api_secret = encrypt_api_key(api_secret)
        # Auto-enable live trading once both keys are present
        if prefs.bitunix_api_key and prefs.bitunix_api_secret:
            prefs.auto_trading_enabled = True

        # Trading account settings — pip/dollar P&L display in mobile app
        if "account_balance" in body:
            try:
                _bal = float(body["account_balance"])
            except (TypeError, ValueError):
                raise HTTPException(status_code=400, detail="account_balance must be numeric")
            if not _math2.isfinite(_bal) or _bal <= 0 or _bal > 100_000_000:
                raise HTTPException(status_code=400, detail="account_balance must be a positive value up to 100,000,000")
            prefs.account_balance = _bal

        if "lot_size" in body:
            try:
                _ls = float(body["lot_size"])
            except (TypeError, ValueError):
                raise HTTPException(status_code=400, detail="lot_size must be numeric")
            if not _math2.isfinite(_ls) or _ls <= 0 or _ls > 1000:
                raise HTTPException(status_code=400, detail="lot_size must be between 0 and 1000")
            prefs.lot_size = _ls

        if "accepted_risk_levels" in body:
            val = body["accepted_risk_levels"]
            prefs.accepted_risk_levels = ",".join(val) if isinstance(val, list) else val

        # StrategyPortalSettings fields
        portal_map = {
            "default_leverage":          "default_leverage",
            "default_position_size":     "default_position_size",
            "default_daily_loss_limit":  "default_daily_loss_limit",
            "default_max_positions":     "default_max_positions",
            "default_direction":         "default_direction",
            "default_cooldown_minutes":  "default_cooldown_minutes",
            "default_max_trades_day":    "default_max_trades_day",
            "paper_mode_default":        "paper_mode_default",
            "auto_activate":             "auto_activate",
            "dm_paper_alerts":           "dm_paper_alerts",
            "dm_live_alerts":            "dm_live_alerts",
            "global_daily_loss_pct":     "global_daily_loss_pct",
            "global_max_positions":      "global_max_positions",
        }
        for k, attr in portal_map.items():
            if k in body:
                setattr(portal, attr, body[k])

        # Detect Bitunix UID change → notify admin
        new_buid = body.get("bitunix_uid", "").strip() if "bitunix_uid" in body else None
        db.commit()

        if new_buid and new_buid != _old_buid:
            import asyncio as _aio3
            _aio3.create_task(_notify_admin_uid_connected(
                user_name=user.first_name or user.username or "User",
                user_uid=user.uid or "",
                bitunix_uid=new_buid,
            ))

        return JSONResponse({"success": True})
    finally:
        db.close()


@app.get("/api/settings/test-exchange")
async def api_test_exchange(uid: str = Query(...)):
    """Test Bitunix API keys by fetching account balance.
    Returns {ok, balance, currency} on success or {ok, error} on failure.
    """
    from app.database import SessionLocal
    db = SessionLocal()
    try:
        user = _get_user_by_uid(uid, db)
        if not user:
            raise HTTPException(status_code=403)
        from app.models import UserPreference
        prefs = db.query(UserPreference).filter(UserPreference.user_id == user.id).first()
        if not prefs or not prefs.bitunix_api_key or not prefs.bitunix_api_secret:
            return JSONResponse({"ok": False, "error": "No API keys saved. Enter and save your Bitunix API keys first."})
        from app.services.bitunix_trader import BitunixTrader
        from app.utils.encryption import decrypt_api_key
        import httpx, asyncio
        async with httpx.AsyncClient(timeout=10) as client:
            trader = BitunixTrader(
                decrypt_api_key(prefs.bitunix_api_key),
                decrypt_api_key(prefs.bitunix_api_secret),
            )
            trader.client = client
            try:
                balance = await asyncio.wait_for(trader.get_account_balance(), timeout=9)
                return JSONResponse({"ok": True, "balance": round(float(balance), 2), "currency": "USDT"})
            except asyncio.TimeoutError:
                return JSONResponse({"ok": False, "error": "Connection timed out — check your API keys and Bitunix account."})
            except Exception as e:
                err = str(e)
                if "401" in err or "403" in err or "invalid" in err.lower() or "auth" in err.lower():
                    return JSONResponse({"ok": False, "error": "Authentication failed — double-check your API key and secret."})
                return JSONResponse({"ok": False, "error": f"Connection failed: {err[:120]}"})
    finally:
        db.close()


@app.put("/api/settings/password")
async def api_set_password(request: Request, uid: str = Query(...)):
    """Set or change the account password.
    Body: { current_password?: str, new_password: str }
    - UID-only users with no password: no current_password required.
    - Users who already have a password: current_password must be provided and correct.
    """
    from app.database import SessionLocal
    body             = await request.json()
    current_password = (body.get("current_password") or "").strip()
    new_password     = (body.get("new_password")     or "").strip()

    if len(new_password) < 8:
        raise HTTPException(status_code=400, detail="Password must be at least 8 characters.")

    def _do():
        db = SessionLocal()
        try:
            u = _get_user_by_uid(uid, db)
            if not u:
                return "not_found"
            if u.password_hash:
                if not current_password:
                    return "need_current"
                if not _verify_password(current_password, u.password_hash):
                    return "wrong_current"
            u.password_hash = _hash_password(new_password)
            db.commit()
            return "ok"
        finally:
            db.close()

    result = await asyncio.to_thread(_do)
    if result == "not_found":
        raise HTTPException(status_code=403, detail="Unknown user.")
    if result == "need_current":
        raise HTTPException(status_code=400, detail="Please enter your current password.")
    if result == "wrong_current":
        raise HTTPException(status_code=403, detail="Current password is incorrect.")
    return JSONResponse({"success": True})


def _describe_existing_config(config: dict) -> str:
    """Render a strategy config dict as a compact human-readable block for Claude."""
    lines = []
    lines.append(f'Name: "{config.get("name", "Unnamed")}"')
    lines.append(f'Direction: {config.get("direction", "BOTH")}')

    entry = config.get("entry_conditions") or {}
    conds = entry.get("conditions") or []
    for i, c in enumerate(conds[:4]):
        ctype = c.get("type", "?")
        ctf   = c.get("timeframe", "")
        extra = {k: v for k, v in c.items() if k not in ("type", "timeframe", "condition", "entry_type")}
        label = f"{ctype} on {ctf}" if ctf else ctype
        if extra:
            bits = ", ".join(f"{k}={v}" for k, v in list(extra.items())[:3])
            label += f" ({bits})"
        tag = "Primary Signal" if i == 0 else f"Confirmation {i}"
        lines.append(f"{tag}: {label}")

    ex  = config.get("exit") or {}
    tp1 = ex.get("take_profit_pct") or ex.get("take_profit_pips")
    tp2 = ex.get("take_profit2_pct") or ex.get("take_profit2_pips")
    sl  = ex.get("stop_loss_pct") or ex.get("stop_loss_pips")
    tp_unit = "pips" if ex.get("stop_loss_pips") else "%"
    parts = [f"TP1: {tp1}{tp_unit}" if tp1 else None,
             f"TP2: {tp2}{tp_unit}" if tp2 else None,
             f"SL: {sl}{tp_unit}"  if sl  else None,
             "trailing stop" if ex.get("trailing_stop") else None,
             f"breakeven at {ex['breakeven_at_pct']}%" if ex.get("breakeven_at_pct") else None]
    lines.append(" | ".join(p for p in parts if p))

    risk = config.get("risk") or {}
    _risk_line = (
        f"Leverage: {risk.get('leverage', 1)}x | "
        f"Position Size: {risk.get('position_size_pct', 5)}% | "
        f"Max Trades/Day: {risk.get('max_trades_per_day', 5)}"
    )
    if risk.get("max_open_positions"):
        _risk_line += f" | Max Open Positions: {risk['max_open_positions']}"
    if risk.get("cooldown_minutes"):
        _risk_line += f" | Cooldown: {risk['cooldown_minutes']}min"
    if risk.get("daily_loss_limit_pct"):
        _risk_line += f" | Daily Loss Limit: {risk['daily_loss_limit_pct']}%"
    lines.append(_risk_line)

    # Session / day filters — round-trip so an "improve" edit never silently drops them.
    # Tolerate legacy non-dict shapes (a bare list or string) so this never raises.
    _filters = config.get("filters") or {}
    _sraw = _filters.get("session")
    if isinstance(_sraw, dict):
        _sess = _sraw.get("sessions") or []
    elif isinstance(_sraw, (list, tuple)):
        _sess = list(_sraw)
    elif isinstance(_sraw, str) and _sraw.strip():
        _sess = [_sraw.strip()]
    else:
        _sess = []
    _draw = _filters.get("trading_days")
    if isinstance(_draw, (list, tuple)):
        _days = list(_draw)
    elif isinstance(_draw, str) and _draw.strip():
        _days = [_draw.strip()]
    else:
        _days = []
    if _sess or _days:
        _bits = []
        if _sess:
            _bits.append("Sessions: " + ", ".join(str(s) for s in _sess))
        if _days:
            _bits.append("Trading Days: " + ", ".join(str(d) for d in _days))
        lines.append(" | ".join(_bits))

    ac  = config.get("asset_class", "crypto")
    uni = config.get("universe") or {}
    if uni.get("type") == "specific":
        syms = ", ".join((uni.get("symbols") or [])[:6])
        lines.append(f"Asset Class: {ac} | Symbols: {syms}")
    else:
        lines.append(f"Asset Class: {ac} | Universe: {uni.get('type', 'all')}")

    return "\n  ".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# AI-builder personalization — feed the user's REAL strategies + performance
# into the chat builder, improve-mode, and portfolio review.
# ─────────────────────────────────────────────────────────────────────────────
_SESSION_ORDER = ["Night (00-06 UTC)", "Morning (06-12 UTC)",
                  "Afternoon (12-18 UTC)", "Evening (18-24 UTC)"]


def _utc_session_name(h: int) -> str:
    if h < 6:  return "Night (00-06 UTC)"
    if h < 12: return "Morning (06-12 UTC)"
    if h < 18: return "Afternoon (12-18 UTC)"
    return "Evening (18-24 UTC)"


def _compute_strategy_trade_stats(db, strategy_id, limit: int = 100) -> dict:
    """
    Shared trade-analytics computation over a strategy's closed executions.
    Returns advisor-style text blocks plus a compact one-line summary and the
    raw per-bucket dicts. Used by /api/strategy-advisor, chat-builder improve
    mode, and /api/portfolio-review so the analytics math lives in one place.
    """
    from app.strategy_models import StrategyExecution

    out = {
        "session_block": "  No session data yet",
        "dow_block":     "  No day-of-week data yet",
        "best_coins":    "not enough data",
        "worst_coins":   "none negative yet",
        "trade_log":     "  No trades yet",
        "compact":       "no closed trades yet",
        "n_closed":      0,
        "best_session":  None,
        "worst_session": None,
    }
    try:
        execs = (
            db.query(StrategyExecution)
            .filter(
                StrategyExecution.strategy_id == strategy_id,
                StrategyExecution.outcome.in_(["WIN", "LOSS", "BREAKEVEN"]),
            )
            .order_by(StrategyExecution.fired_at.desc())
            .limit(limit)
            .all()
        )
    except Exception as e:
        logger.warning(f"_compute_strategy_trade_stats query error (non-fatal): {e}")
        return out

    if not execs:
        return out

    DOW = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

    def _wr(s):
        tot = s["win"] + s["loss"]
        return f"{round(s['win']/tot*100)}% ({s['win']}W/{s['loss']}L)" if tot else "no data"

    hour_stats, dow_stats, sess_stats, coin_stats = {}, {}, {}, {}
    try:
        for ex in execs:
            is_win  = ex.outcome == "WIN"
            is_loss = ex.outcome == "LOSS"
            coin = (ex.symbol or "").replace("USDT", "")
            if ex.fired_at:
                h  = ex.fired_at.hour
                d  = ex.fired_at.weekday()
                sn = _utc_session_name(h)
                hour_stats.setdefault(h, {"win": 0, "loss": 0})
                dow_stats.setdefault(d, {"win": 0, "loss": 0})
                sess_stats.setdefault(sn, {"win": 0, "loss": 0})
                if is_win:
                    hour_stats[h]["win"] += 1; dow_stats[d]["win"] += 1; sess_stats[sn]["win"] += 1
                if is_loss:
                    hour_stats[h]["loss"] += 1; dow_stats[d]["loss"] += 1; sess_stats[sn]["loss"] += 1
            if coin:
                coin_stats.setdefault(coin, {"win": 0, "loss": 0, "pnl": 0.0})
                if is_win:  coin_stats[coin]["win"]  += 1
                if is_loss: coin_stats[coin]["loss"] += 1
                if ex.pnl_pct is not None:
                    coin_stats[coin]["pnl"] += float(ex.pnl_pct)

        sess_lines = [f"  {sn}: {_wr(sess_stats[sn])}" for sn in _SESSION_ORDER if sn in sess_stats]
        if sess_lines:
            out["session_block"] = "\n".join(sess_lines)

        dow_lines = [f"  {DOW[d]}: {_wr(dow_stats[d])}" for d in range(7) if d in dow_stats]
        if dow_lines:
            out["dow_block"] = "\n".join(dow_lines)

        coin_sorted = sorted(coin_stats.items(), key=lambda x: x[1]["pnl"], reverse=True)
        if coin_sorted:
            out["best_coins"] = ", ".join(
                f"{c} ({v['win']}W/{v['loss']}L, {v['pnl']:+.0f}%)" for c, v in coin_sorted[:5]
            )
            negatives = [(c, v) for c, v in coin_sorted[-5:] if v["pnl"] < 0]
            if negatives:
                out["worst_coins"] = ", ".join(
                    f"{c} ({v['win']}W/{v['loss']}L, {v['pnl']:+.0f}%)" for c, v in negatives
                )

        recent_lines = []
        for ex in reversed(execs[:25]):
            try:
                dt   = ex.fired_at.strftime("%m/%d %H:%M") if ex.fired_at else "?"
                coin = (ex.symbol or "?").replace("USDT", "")
                pnl  = f"{ex.pnl_pct:+.1f}%" if ex.pnl_pct is not None else "?"
                dur  = ""
                if ex.fired_at and ex.closed_at:
                    fa = ex.fired_at.replace(tzinfo=None)
                    ca = ex.closed_at.replace(tzinfo=None)
                    dur = f" {int((ca - fa).total_seconds() / 60)}m"
                recent_lines.append(f"  {dt} | {coin} {ex.direction or ''} | {ex.outcome}{dur} | {pnl}")
            except Exception:
                continue
        if recent_lines:
            out["trade_log"] = "\n".join(recent_lines)

        # Best/worst session by win rate (min 4 decided trades to be meaningful)
        ranked = []
        for sn, s in sess_stats.items():
            tot = s["win"] + s["loss"]
            if tot >= 4:
                ranked.append((sn, s["win"] / tot, tot))
        if ranked:
            ranked.sort(key=lambda x: x[1], reverse=True)
            out["best_session"]  = f"{ranked[0][0]} ({round(ranked[0][1]*100)}%)"
            out["worst_session"] = f"{ranked[-1][0]} ({round(ranked[-1][1]*100)}%)"

        n = len(execs)
        out["n_closed"] = n
        bits = [f"{n} closed trades"]
        if out["best_session"]:
            bits.append(f"best session {out['best_session']}")
        if out["worst_session"] and out["worst_session"] != out["best_session"]:
            bits.append(f"worst session {out['worst_session']}")
        if out["worst_coins"] != "none negative yet":
            bits.append(f"losing symbols: {out['worst_coins']}")
        out["compact"] = "; ".join(bits)
    except Exception as e:
        logger.warning(f"_compute_strategy_trade_stats analytics error (non-fatal): {e}")

    return out


def _build_user_trading_context(db, user, asset_class: str = None, max_strategies: int = 8) -> str:
    """
    Compact summary of the user's saved strategies + real performance, formatted
    for an LLM system prompt. Lets the AI builder reference what the user already
    owns, build on winners, avoid duplicates, and warn on past-loser patterns.
    Cached per (user, asset_class) for 60s. Returns "" when the user has nothing.
    """
    try:
        uid = getattr(user, "id", 0) or 0
        if not uid:
            return ""
        from app.cache import get_cache, set_cache
        ckey = f"trading_ctx:{uid}:{asset_class or 'all'}"
        cached = get_cache(ckey)
        if cached is not None:
            return cached

        from app.strategy_models import UserStrategy, StrategyPerformance

        q = (
            db.query(UserStrategy, StrategyPerformance)
            .outerjoin(StrategyPerformance, StrategyPerformance.strategy_id == UserStrategy.id)
            .filter(
                UserStrategy.user_id == uid,
                UserStrategy.status.in_(["active", "paused", "draft"]),
            )
        )
        rows = q.all()
        if not rows:
            set_cache(ckey, "", 60)
            return ""

        def _signal_types(cfg):
            try:
                conds = (cfg.get("entry_conditions") or {}).get("conditions") or []
                types = [c.get("type", "?") for c in conds if isinstance(c, dict)]
                return "+".join(types[:3]) if types else "?"
            except Exception:
                return "?"

        def _sort_key(row):
            _s, perf = row
            return (perf.total_trades if perf and perf.total_trades else 0)

        rows.sort(key=_sort_key, reverse=True)

        lines, winners, bleeders = [], [], []
        total = len(rows)
        for strat, perf in rows[:max_strategies]:
            cfg = strat.config or {}
            sig = _signal_types(cfg)
            ac  = strat.asset_class or cfg.get("asset_class", "crypto")
            tf  = "?"
            try:
                conds = (cfg.get("entry_conditions") or {}).get("conditions") or []
                if conds and isinstance(conds[0], dict):
                    tf = conds[0].get("timeframe", "?")
            except Exception:
                pass
            if perf and perf.total_trades:
                wr   = round(perf.win_rate or 0)
                pnl  = round(perf.total_pnl_pct or 0, 1)
                perf_s = f"{perf.total_trades} trades, {wr}% WR, {pnl:+}%"
                if perf.total_trades >= 10 and wr >= 55 and pnl > 0:
                    winners.append(f"{strat.name} ({wr}% WR, {pnl:+}%)")
                if perf.total_trades >= 10 and pnl < 0:
                    bleeders.append(f"{strat.name} ({pnl:+}% over {perf.total_trades})")
            else:
                perf_s = "no closed trades yet"
            lines.append(f'  • "{strat.name}" [{ac}/{sig} on {tf}, {strat.status}] — {perf_s}')

        block = [
            "\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501",
            f"THE USER ALREADY HAS {total} SAVED STRATEG{'Y' if total == 1 else 'IES'} — USE THIS:",
            "\n".join(lines),
        ]
        if winners:
            block.append("Their best performers (lean into these styles): " + "; ".join(winners[:4]))
        if bleeders:
            block.append("Underperformers (avoid repeating these patterns): " + "; ".join(bleeders[:4]))
        block.append(
            "Use this to: reference what they own, AVOID building a near-duplicate of an existing "
            "strategy (suggest a genuinely different angle if their ask overlaps), build on what's "
            "winning, and gently flag if their idea resembles one of their underperformers."
        )
        block.append("\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501")
        text = "\n".join(block)
        set_cache(ckey, text, 60)
        return text
    except Exception as e:
        logger.warning(f"_build_user_trading_context error (non-fatal): {e}")
        return ""


@app.post("/api/chat-builder")
async def chat_builder_api(request: Request):
    """
    Conversational AI strategy builder.
    Body: { uid, messages: [{role, content}] }
    Returns: { reply, complete, description }
    """
    import anthropic

    body = await request.json()
    uid = (body.get("uid") or "").strip()
    messages = body.get("messages") or []

    # Auth + tier check
    from app.database import SessionLocal
    db = SessionLocal()
    user = None
    allowed, used, limit, is_pro = True, 0, None, False
    try:
        try:
            user = _get_user_by_uid(uid, db)
            if not user:
                raise HTTPException(status_code=403, detail="Invalid UID")
            sub = _get_portal_sub(user.id, db)
            allowed, used, limit, is_pro = _chat_calls_info(sub, db, user)
        except HTTPException:
            raise
        except Exception as _db_err:
            logger.warning(f"Chat builder auth DB error: {_db_err} — continuing without quota check")
            # DB is overloaded; allow the request through with default limits rather
            # than surfacing a "Connection error" to the user.
            user = type("_FallbackUser", (), {"id": 0})()
        if not messages and not body.get("existing_config"):
            return {
                "reply": "Hey! 👋 Tell me what you want to trade — long, short, or both? And what kind of signal are you thinking? (e.g. RSI scalp, MACD swing, order block, etc.)",
                "complete": False,
                "description": None,
                "calls_used": used,
                "calls_limit": limit,
                "is_pro": is_pro,
            }
        if not allowed:
            return {
                "reply": None,
                "complete": False,
                "description": None,
                "limit_reached": True,
                "calls_used": used,
                "calls_limit": limit,
                "is_pro": False,
            }
    finally:
        db.close()

    asset_class          = body.get("asset_class", "crypto") or "crypto"
    existing_strategy_id = body.get("existing_strategy_id")
    existing_config      = body.get("existing_config") or {}

    market_ctx = {
        "crypto":  "crypto perpetual futures (BTC, ETH, SOL, altcoins)",
        "forex":   "forex (EURUSD, GBPUSD, USDJPY, XAUUSD/Gold and other pairs)",
        "stock":   "stocks and equities (AAPL, TSLA, NVDA, SPY etc.)",
        "index":   "indices (S&P 500, Nasdaq, FTSE, DAX etc.)",
    }.get(asset_class, "crypto perpetual futures")

    tp_sl_example = "TP Pips: 30 | SL Pips: 15" if asset_class == "forex" else "TP1: 2% | SL: 1%"
    symbols_example = "Symbols: EURUSD,GBPUSD" if asset_class == "forex" \
        else ("Symbols: AAPL,TSLA,NVDA" if asset_class == "stock" \
        else ("Symbols: NAS100,SPX500" if asset_class == "index" else "Coins: all"))

    # Randomise the example primary+confirmation signals so Claude doesn't anchor
    # on RSI+EMA as the default every single time.
    import random as _random
    _EXAMPLE_POOL = {
        "crypto": [
            ("FVG bullish tap on 15m", "BOS bullish on 1h", "StochRSI bullish cross on 5m"),
            ("SuperTrend bullish flip on 1h", "ADX > 25 on 1h", "EMA ribbon bullish on 4h"),
            ("BB lower touch on 15m", "StochRSI bullish cross on 5m", "EMA 50 bullish on 1h"),
            ("MACD bullish cross on 1h", "price above 200 SMA on 4h", "volume spike 1.5x on 1h"),
            ("RSI below 30 on 15m", "EMA 50 bullish on 1h", "StochRSI bullish cross on 15m"),
            ("order_block bullish touch on 15m", "CHoCH bullish on 1h", "StochRSI oversold on 5m"),
            ("price momentum +3% on 30m", "volume spike 2x on 30m", "SuperTrend bullish on 4h"),
            ("Ichimoku TK cross bullish on 4h", "ADX > 25 on 1h", "OI rising"),
            ("funding rate < -0.08% on 1h", "BB lower touch on 4h", "RSI < 35 on 1h"),
            ("IFVG bullish on 15m", "CHoCH bullish on 1h", "StochRSI oversold on 15m"),
        ],
        "forex": [
            ("london_kz killzone on 15m", "OTE retracement on 15m", "EMA ribbon bullish on 1h"),
            ("fx_displacement bullish on 15m", "FVG bullish tap on 15m", "EMA 50 bullish on 1h"),
            ("fx_po3 Asian low swept on 15m", "ny_kz timing", "stochastic oversold on 15m"),
            ("wyckoff spring on 1h", "stochastic bullish_cross on 15m", "EMA 50 bullish on 4h"),
            ("fx_silver_bullet 10 AM on 5m", "FVG bullish tap on 5m", "PD array discount zone"),
            ("fx_judas_swing on 5m", "fx_breaker bullish on 5m", "EMA 50 bullish on 1h"),
            ("stochastic bullish_cross on 15m", "PDH/PDL confluence on 15m", "EMA ribbon bullish on 1h"),
            ("ifvg bullish on 15m", "fx_displacement bullish on 1h", "RSI < 50 on 15m"),
            ("forex_currency_strength bullish on 4h", "EMA ribbon bullish on 4h", "ADX > 25 on 1h"),
            ("fx_breaker bullish on 15m", "london_kz timing", "RSI < 45 on 15m"),
        ],
        "stock": [
            ("ORB breakout 30m", "volume spike 1.5x on 30m", "price above VWAP"),
            ("RSI < 28 on 4h", "BB lower touch on 4h", "stochastic oversold on 1h"),
            ("SuperTrend bullish flip on 1h", "MACD bullish cross on 1h", "price above VWAP"),
            ("EMA ribbon bullish on 1h", "ADX > 28 on 1h", "price above 200 SMA on daily"),
            ("BB/Keltner squeeze fires on 1h", "volume spike 2x on 1h", "SuperTrend bullish on 4h"),
            ("price_momentum +4% gap up", "RSI > 72 on 15m", "EMA 50 bearish on 1h"),
        ],
        "index": [
            ("RSI < 40 on 15m", "price below VWAP", "BB lower touch on 15m"),
            ("ORB breakout 30m", "volume spike 1.8x on 30m", "SuperTrend bullish on 1h"),
            ("EMA ribbon bullish on 1h", "ADX > 25 on 1h", "price above 200 SMA on daily"),
            ("BB/Keltner squeeze fires on 1h", "volume spike 2x on 1h", "SuperTrend bullish on 4h"),
            ("Ichimoku TK cross bullish on 4h", "ADX > 25 on 1h", "price above cloud on 4h"),
        ],
    }
    _ex_primary, _ex_conf1, _ex_conf2 = _random.choice(
        _EXAMPLE_POOL.get(asset_class, _EXAMPLE_POOL["crypto"])
    )

    # Pick a signal family to nudge Claude toward variety without overriding user intent
    _VARIETY_NUDGES = {
        "crypto": [
            "SMC (FVG / order block / BOS / CHoCH / IFVG)",
            "volatility compression (BB/Keltner squeeze, ATR expansion)",
            "funding rate + open interest contrarian setups",
            "multi-timeframe EMA ribbon + SuperTrend trend-following",
            "oscillator mean-reversion (RSI, StochRSI, BB lower)",
            "price momentum / volume spike breakouts",
            "Ichimoku cloud breakout or TK cross",
            "MACD divergence or zero-cross setups",
            "intraday VWAP bands / VWAP bias + RVOL + ATR volatility filters",
        ],
        "forex": [
            "ICT killzone (London / NY / Asian KZ, Silver Bullet, Power of 3)",
            "SMC structure (BOS, CHoCH, FVG, IFVG, order block)",
            "liquidity grabs and displacement moves",
            "Wyckoff (spring, upthrust, distribution phases)",
            "currency strength divergence",
            "stochastic or oscillator setups at key levels",
            "session break / PDH-PDL / previous level plays",
            "breaker blocks and mitigation block reversals",
        ],
        "stock": [
            "opening range breakout (ORB) + volume confirmation",
            "mean reversion from oversold extremes (RSI, BB, StochRSI)",
            "momentum continuation (EMA ribbon + ADX + SuperTrend)",
            "gap fade or gap follow strategy",
            "squeeze / compression breakout (BB inside Keltner)",
            "VWAP bias + VWAP band fades with RVOL confirmation",
        ],
        "index": [
            "VWAP mean reversion intraday (VWAP bands + RVOL)",
            "VWAP bias trend-day rider with ATR volatility filter",
            "ORB breakout on session open",
            "Ichimoku cloud breakout or continuation",
            "trend-following with EMA ribbon + ADX",
            "oversold reversal at major support (RSI + BB)",
        ],
    }
    _nudge = _random.choice(_VARIETY_NUDGES.get(asset_class, _VARIETY_NUDGES["crypto"]))

    # ── Per-asset-class expertise block ──────────────────────────────────────
    if asset_class == "forex":
        asset_rules = """
FOREX RULES:
- TP and SL in PIPS only — never percentages.
  Scalp: TP 15–25 / SL 8–15. Intraday ICT: TP 30–50 / SL 15–25. Swing: TP 60–120 / SL 30–50.
  Gold (XAUUSD): minimum viable TP is 50 pips — spreads are wider and each pip = $1 per 0.1 lot.
- Never mention BTC regime. Never mention leverage.
- Pair aliases: "Gold" → XAUUSD · "Silver" → XAGUSD · "Cable" → GBPUSD · "Fiber" → EURUSD · "Yen" → USDJPY · "Aussie" → AUDUSD · "Loonie" → USDCAD · "Kiwi" → NZDUSD.
- In ###STRATEGY### use "TP Pips" and "SL Pips".

FP MARKETS BROKER CONTEXT (this is the live broker — know it):
- Platform: cTrader. Execution: 29ms average — reliable for scalp strategies.
- Tight spreads on majors. Minimum viable TP on majors is 15 pips for scalps — anything tighter is not worth the spread cost.
- Available instruments: 70+ forex pairs (majors, minors, exotics), Gold/Silver/Platinum vs USD, 19 indices (SPX, NDX, DAX, FTSE, NKY…), crypto CFDs.
- Never mention commissions, spreads, or account types to the user — just use this to inform your TP/SL recommendations silently.

SIGNAL RECOGNITION:
  killzone / London KZ / NY KZ / Asian KZ → fx_killzone
  OTE / optimal trade entry / golden zone / 61.8 / 78.6 fib → fx_ote
  displacement / impulse candle / institutional move → fx_displacement
  equal highs / EQH / equal lows / EQL / BSL / SSL → fx_equal_hl
  CISD / change in state of delivery / delivery flip / change of delivery → fx_cisd
  SDP / sweep displacement pullback / sweep then displacement then retrace / sweep + FVG + pullback → fx_sdp
  breaker block / failed OB / mitigation block → fx_breaker
  premium zone / discount / PD array / equilibrium → fx_pd_array
  Judas swing / fake move / manipulation leg / stop hunt then reverse → fx_judas_swing
  silver bullet / 3 AM / 10 AM / 3 PM setup → fx_silver_bullet
  only trade/fire DURING London/NY session / active all of the session / restrict to session hours → set the top-level Sessions field (e.g. "Sessions: London, New York") — this is the preferred cross-asset session restriction; reserve a forex_session in_session CONDITION for ICT signal combos or session_open/session_close sub-windows only (never both for the same sessions)
  London breakout / Asian range break / session break → forex_session_break
  PDH sweep / previous day high / PDL → forex_prev_level
  currency strength / strong USD / weak GBP → forex_currency_strength
  liquidity grab / stop hunt / equal highs sweep → forex_liquidity_pa
  COT / commitment of traders / positioning → forex_cot
  stochastic / stoch cross / %K %D / stoch oversold → stochastic
  Power of 3 / PO3 / AMD cycle / Asian range swept → fx_po3
  Wyckoff / spring / shakeout / upthrust / markup / distribution → wyckoff
  IFVG / inverse FVG / inverse fair value gap / re-entry gap / mitigated gap → ifvg
  FVG / fair value gap / imbalance / unmitigated gap / price gap → fvg
  CONFIDENCE/QUALITY tier for fvg & ifvg → add min_confidence:"low"|"medium"|"high" (omit/"any"=off).
    Graded from ATR-relative quality (displacement strength + gap size + freshness); a tier accepts
    that grade and above. Use for "high-confidence FVG", "only the strong/quality gaps", "A+ setups only",
    "skip the weak FVGs" → min_confidence:"high"; "medium-confidence and up" → min_confidence:"medium".

IDEA BANK — pitch these when user is open to suggestions:
  "London ICT sniper" — london_kz killzone + OTE retracement + FVG bullish tap on 15m, confirmed by EMA ribbon bullish on 1h. LONG, TP1 35 / TP2 55 / SL 15, breakeven at 70%. R:R 2.3:1 on TP1, 3.7:1 to TP2. Fires 1–2× per London session. EURUSD or GBPUSD.
  "Asian PO3 fade" — fx_po3 (Asian low swept → distribution up) confirmed by ny_kz killzone timing + stochastic oversold on 15m. LONG, TP1 30 / TP2 50 / SL 12, breakeven at 65%. Catches the AMD reversal right as NY opens. Any major pair.
  "Gold Wyckoff spring" — wyckoff spring on 1h XAUUSD + stochastic bullish_cross on 15m + EMA 50 bullish on 4h. LONG, TP1 80 / TP2 140 / SL 35. Swing play, trailing stop after TP1. Gold respects Wyckoff structure exceptionally well — 1–2 trades/week.
  "Silver bullet scalp" — fx_silver_bullet (10 AM NY window) + bullish FVG tap + PD array discount zone. LONG/SHORT, TP1 20 / SL 10, no TP2 (tight scalp window). ICT's highest-probability 90-minute window.
  "Judas + breaker fade" — fx_judas_swing (session open fake move) + fx_breaker (former supply flipped support) on 5m, EMA 50 direction on 1h. BOTH, TP1 30 / TP2 50 / SL 15, breakeven at 70%. Catches the reversal after stop hunts at open.
  "Gold IFVG retest" — IFVG bullish (mitigated gap retest) on 15m XAUUSD + fx_displacement bullish on 1h as higher TF confirmation. LONG, TP1 60 / TP2 100 / SL 25. 2.4:1 R:R. Gold fills IFVG gaps with high precision — institutional re-entry after a displacement leg.
  "Displacement + EQH liquidity" — fx_displacement bullish on 15m + fx_equal_hl (EQL swept) + FVG bullish tap. LONG, TP1 40 / TP2 70 / SL 18, breakeven at 70%. Institutional move creating imbalance after liquidity grab below equal lows.
  "Currency strength swing" — forex_currency_strength (base strong vs quote weak) + EMA ribbon aligned bullish on 4h + ADX > 25 on 1h. LONG, TP1 100 / TP2 180 / SL 45, trailing stop after TP1. Best on slow macro trends — 3–5 day holds.
  "Breaker block killzone" — fx_breaker bullish on 15m + london_kz or ny_kz timing + RSI < 45 on 15m (not overbought). LONG, TP1 35 / TP2 60 / SL 15, breakeven at 65%. Former supply turned demand with session timing.

CONFIRMATION PAIRINGS (add to make entries sharper):
  Primary: OTE or displacement → add: FVG bullish tap on same TF + EMA ribbon bullish on next TF up
  Primary: killzone → add: OTE retracement or EQH/EQL sweep + stochastic oversold
  Primary: PO3/AMD → add: session timing (ny_kz after Asian range sets) + ADX > 20
  Primary: Wyckoff spring → add: stochastic bullish_cross on lower TF + volume confirmation
  Primary: stochastic cross → add: key S/R level or PDH/PDL confluence + EMA direction filter
  Primary: silver bullet → add: FVG or PD array for precision entry
  Primary: IFVG → add: fx_displacement on higher TF + PDL/PDH not yet swept"""

    elif asset_class == "stock":
        asset_rules = f"""
STOCK RULES:
- TP/SL in percentages. Leverage 1–5 max. No BTC regime.
- Always ask which tickers (e.g. AAPL, TSLA, NVDA, MSFT, META, AMZN, SPY, QQQ).
- Best timeframes: 5m–1h scalp, 1h–4h swing. Daily for macro position.
- ORB (opening range breakout) is the highest-probability intraday stock signal — 30-min or 1h range.
- RSI and MACD work best on 1h+ for stocks. Avoid 1m–5m indicator noise.

IDEA BANK — pitch these when user is open:
  "Mega-cap momentum" — EMA ribbon aligned bullish on 1h + ADX > 28 on 1h + price above 200 SMA on daily. LONG, TP1 4% / TP2 7% / SL 2%, trailing stop after TP1. Rides institutional accumulation on AAPL/MSFT/NVDA. Swing, 1–2 trades/week.
  "Tech ORB breakout" — opening range breakout (30m) + volume spike 1.5× + market structure BOS bullish on 5m, confirmed by price above VWAP. LONG/SHORT, TP1 3% / TP2 5% / SL 1.5%, breakeven at 70%. Catches the directional move after the first 30 minutes resolve.
  "Oversold quality reversal" — RSI < 28 on 4h + BB lower touch on 4h + stochastic oversold on 1h (multi-TF confluence). LONG, TP1 5% / TP2 9% / SL 2.5%, breakeven at 65%. Beaten-down quality names bouncing from major support. AAPL, NVDA, MSFT best.
  "SuperTrend continuation" — SuperTrend bullish flip on 1h + MACD bullish cross on 1h + price above VWAP. LONG, TP1 4% / TP2 7% / SL 2%, trailing stop after TP1 hit. Clean trend entry after pullback. Works well on high-beta names.
  "Squeeze breakout" — BB inside Keltner squeeze on 1h + volume spike 2×, confirmed by SuperTrend direction on 4h. LONG/SHORT, TP1 5% / TP2 9% / SL 2%. Plays the compression → explosion setup. Best after tight consolidation.
  "Gap fade" — price_momentum 4%+ gap up at open + RSI > 72 on 15m + EMA 50 bearish on 1h. SHORT, TP1 3% / SL 1.5%. Fades overextended gap opens — no TP2 since gap fades are fast moves. Works on earnings-adjacent days.
  "VWAP reclaim" — vwap_bias above (price reclaims session VWAP) on 5m + rvol high (≥1.5×) + market structure BOS bullish on 5m. LONG, TP1 2.5% / SL 1.2%, breakeven at 70%. The opening-drive reclaim — strongest with real relative volume behind it. AAPL/NVDA/SPY.
  "Band fade to VWAP" — vwap_bands above_upper (price stretched to +2 SD) on 5m + RSI > 72 on 5m. SHORT, TP1 1.8% / SL 1%. Fades intraday overextension back to the VWAP mean. Best on high-beta names mid-session.

CONFIRMATION PAIRINGS:
  Breakout → add: volume spike 1.5×+ and ATR expanding (or atr_filter expanding)
  Mean reversion → add: candlestick reversal (hammer/pin bar) and RSI divergence
  Trend follow → add: price above 200 SMA and ADX trending + vwap_bias above
  ORB → add: vwap_bias above (long) or below (short) + rvol high
  Any intraday → add: rvol high (real participation) + atr_filter volatile (enough range)"""

    elif asset_class == "index":
        asset_rules = f"""
INDEX RULES:
- TP/SL in percentages. Leverage 1–10. No BTC regime.
- Symbols: NAS100 (Nasdaq), SPX500 (S&P), US30 (Dow), GER40 (DAX), UK100 (FTSE).
- Indices trend well — SuperTrend and EMA ribbon setups work exceptionally here.
- VWAP is the key intraday level for index scalps. Session opens are critical.

IDEA BANK — pitch these when user is open:
  "VWAP bounce scalp" — price pulls back below VWAP + RSI < 40 + BB lower touch. LONG, TP 1.5% / SL 0.7%. Classic intraday mean reversion on SPX. 5m–15m, high frequency.
  "ORB index breakout" — opening range breakout (30m) + volume spike 1.8×. LONG/SHORT, TP 2.5% / SL 1%. Indices have the clearest ORB setups — institutional desks set their bias in the first 30 minutes.
  "Macro trend ride" — EMA ribbon aligned bullish (50/100/200) + ADX > 25 + weekly close above 200 SMA. LONG, TP 4% / SL 2%. Catches multi-week trending phases. NDX and SPX best. 2–3 trades/month.
  "Keltner expansion" — BB/Keltner squeeze fires + volume spike 2× + SuperTrend bullish flip. BOTH, TP 3% / SL 1.5%. Post-compression volatility explosion. Works perfectly on index futures.
  "Ichimoku cloud ride" — price above Ichimoku cloud + TK cross bullish + ADX trending. LONG, TP 5% / SL 2.5%. Ichimoku is unusually reliable on daily/4h index charts. Longer hold, low noise.
  "Oversold index reversal" — RSI < 32 on 4h + price at major support + bullish divergence (MACD). LONG, TP 3.5% / SL 1.5%. Index dips are for buying — institutions accumulate at oversold extremes.
  "VWAP band scalp" — vwap_bands below_lower (price tags −2 SD VWAP band) on 5m + rvol high (≥1.5×). LONG, TP 1.2% / SL 0.6%. The purest intraday index mean-reversion — SPX/NDX respect VWAP bands tightly. High frequency.
  "VWAP trend hold" — vwap_bias above (price holding above VWAP) on 5m + atr_filter volatile + EMA ribbon bullish on 15m. LONG, TP 2.5% / SL 1.2%, breakeven at 70%. Rides trend days from the long side only while price respects VWAP.

CONFIRMATION PAIRINGS:
  Breakout → add: volume spike and ATR expanding (or atr_filter expanding)
  Trend → add: price above 200 EMA and ADX > 25 + vwap_bias above
  Reversal → add: RSI divergence and candlestick reversal + vwap_bands tag
  VWAP scalp → add: rvol high (real volume) + price at key S/R level"""

    else:  # crypto
        asset_rules = """
CRYPTO RULES:
- TP/SL in percentages. Leverage: scalp 10–20×, swing 5–10×, never exceed 25× unless asked.
- "All coins" / altcoins: use min_volume_usd 500k and exclude slow high-caps.
- BTC regime filter: use "bullish" for long-only strategies, "bearish" for shorts, null for BOTH.
- Funding rate signals are uniquely powerful for crypto — negative funding = hidden long opportunity.
- Open interest (OI) rising + price rising = strong institutional accumulation.

SIGNAL RECOGNITION (day-trader filters):
  relative volume / RVOL / unusual volume / volume vs average → rvol
  ATR filter / enough volatility / not dead tape / volatility gate / expanding ATR → atr_filter
  VWAP bands / VWAP standard deviation / ±2 SD VWAP / fade VWAP band → vwap_bands
  above VWAP / below VWAP / VWAP bias / longs only above VWAP → vwap_bias
  volume profile / POC / point of control / value area (VAH/VAL) / high volume node / HVN / LVN / low volume node → volume_profile

QUICK START RECIPES — use these when the user wants something simple, fast-firing, or says "quick reversal / RSI / EMA / fires often / active / scalp / beginner":
  "RSI scalp" — RSI < 30 on 5m, NO confirmations (none/none). BOTH, TP1 2% / SL 1%, 10×, max 8 trades/day. Fires 5-10× per day on active alts. Dead simple — RSI oversold is the single entry trigger.
  "EMA cross" — EMA 9/21 golden cross on 15m, NO confirmations (none/none). LONG, TP1 3% / SL 1.5%, 10×, max 6 trades/day. Clean trend entry — one signal, nothing else.
  "Chart reversal" — bullish_engulfing candlestick on 15m, NO confirmations (none/none). BOTH, TP1 2.5% / SL 1%, 8×, max 6 trades/day. Pure price action — candle pattern tells the whole story.
  "MACD cross" — MACD bullish cross on 15m, NO confirmations (none/none). BOTH, TP1 3% / SL 1.5%, 10×, max 6 trades/day. Classic momentum signal. No filter needed at this frequency.
  "Stoch RSI bounce" — StochRSI oversold on 5m, NO confirmations (none/none). BOTH, TP1 2% / SL 1%, 12×, max 8 trades/day. High-frequency oscillator — fires constantly on anything volatile.
  "Breakout" — range breakout above 20-bar high on 15m, NO confirmations (none/none). LONG, TP1 4% / SL 2%, 10×, max 5 trades/day. Price breaks out → ride the momentum.
  "Supertrend flip" — SuperTrend bullish flip on 15m, NO confirmations (none/none). LONG, TP1 4% / SL 2%, 10×, max 4 trades/day. Clean trend change signal — one of the most reliable single-signal setups.
  RULE for all Quick Start: set Confirmation 1 = none, Confirmation 2 = none. These are designed to fire often — confirmations defeat the purpose.

IDEA BANK — pitch these when user is open:
  "Altcoin oversold sniper" — RSI < 28 on 15m + BB lower touch on 15m + StochRSI bullish cross on 5m (lower TF trigger). LONG, TP1 4% / TP2 7% / SL 2%, 10×, breakeven at 70%. Catches violent altcoin recoveries from oversold extremes. Best on mid-cap alts.
  "EMA ribbon momentum" — EMA ribbon aligned bullish (9/21/55/100/200) on 1h + ADX > 28 on 1h + SuperTrend bullish on 4h (higher TF trend filter). LONG, TP1 6% / TP2 10% / SL 3%, 8×, trailing stop after TP1. The cleanest trend-following stack — waits for full MTF alignment.
  "FVG sniper SMC" — FVG bullish tap on 15m + BOS bullish on 1h + OI rising. LONG, TP1 4% / TP2 7% / SL 2%, 12×, breakeven at 65%. Smart money: trade the imbalance left by institutional displacement, confirmed by 1h structure break.
  "IFVG re-entry" — IFVG bullish (mitigated gap retest) on 15m + CHoCH bullish on 1h + StochRSI oversold on 15m. LONG, TP1 5% / TP2 9% / SL 2%, 10×, breakeven at 70%. Highest R:R SMC setup: price returns to a filled gap that now acts as institutional demand — strongest when 1h structure just changed character.
  "London pump fade" — price momentum +5% in 30m + RSI > 75 on 15m + funding rate > 0.08% (overheated) + EMA 50 bearish on 1h. SHORT, TP1 3% / SL 1.5%, 15×. No TP2 — momentum fades are fast. Fades over-leveraged longs after a parabolic move.
  "Squeeze momentum fire" — BB inside Keltner squeeze on 1h + squeeze momentum bull_mom + volume spike 2× + SuperTrend direction on 4h. LONG/SHORT, TP1 5% / TP2 9% / SL 2%, 12×, breakeven at 70%. Compression → explosion with higher TF direction bias.
  "Order block reversal" — bullish order block touch on 15m + CHoCH bullish on 15m + StochRSI oversold on 5m + price below EMA 50 on 1h (not extended). LONG, TP1 5% / TP2 9% / SL 2.5%, 10×, breakeven at 65%. SMC: institutional demand zone + structure change + oversold trigger.
  "Funding rate reversal" — funding < -0.08% (shorts paying longs) + RSI < 35 on 1h + BB lower touch on 4h. LONG, TP1 6% / TP2 10% / SL 3%, 8×. MTF oversold with contrarian funding — when everyone is short and paying for it, fade them.
  "Ichimoku cloud breakout" — price breaks above Ichimoku cloud on 4h + TK cross bullish on 4h + OI rising + EMA ribbon aligned bullish on 1h. LONG, TP1 8% / TP2 13% / SL 4%, 6×, trailing stop after TP1. Ichimoku on 4h filters 80% of noise — only fires on real institutional breakouts.
  "BTC breakout altcoin chase" — market structure BOS bullish on BTC 1h → scan all alts for EMA ribbon aligning bullish on 1h + volume spike 1.5×. LONG, TP1 7% / TP2 12% / SL 3%, 10×, breakeven at 65%. BTC sets the trend; alts follow with leverage.
  "VWAP band fade" — vwap_bands below_lower (price tags −2 SD VWAP band) on 5m + rvol high (≥1.5× relative volume) confirming the flush. LONG, TP1 2% / SL 1%, 12×, max 8 trades/day. Classic intraday mean reversion back to VWAP — RVOL filters fake low-liquidity wicks.
  "Trend-day VWAP rider" — vwap_bias above (price holding above session VWAP) on 5m + atr_filter volatile (enough range) + EMA ribbon bullish on 15m. LONG, TP1 3% / TP2 5% / SL 1.5%, 10×, breakeven at 70%. Only longs above VWAP on volatile days — the #1 intraday discipline rule.

CONFIRMATION PAIRINGS:
  RSI/MACD → add: EMA direction filter (price above 50 EMA for longs) on higher TF + ADX trending
  FVG / order block / IFVG → add: BOS or CHoCH on 1h as structural confirmation
  Breakout (price momentum, Keltner) → add: volume spike 1.5×+ and ATR expanding
  Trend (EMA ribbon, SuperTrend) → add: ADX > 25 on 1h + SuperTrend aligned on 4h + BTC regime = bullish
  Mean reversion (BB lower, RSI<30) → add: StochRSI bullish cross on lower TF + candlestick reversal
  Funding fade → add: BB lower touch on 4h and RSI divergence on 1h
  SMC (OB, CHoCH, BOS, IFVG) → add: OI rising + session timing (London/NY open)
  Any intraday entry → add: rvol high (real volume) + vwap_bias (trade with the VWAP side)
  Breakout/momentum → add: atr_filter volatile or expanding (skip dead, low-range tape)
  Mean reversion scalp → add: vwap_bands below_lower/above_upper for the precise band tag"""

    # Pre-compute improve-mode block BEFORE the f-string — nested f"""..."""
    # inside an outer f"""...""" is a SyntaxError in Python 3.11.
    if existing_config:
        # Proactive improvement: pull the strategy's REAL trade performance so the
        # AI's assessment is data-driven (losing sessions/symbols, actual win rate)
        # instead of guessing from the config alone. Best-effort + bounded.
        _perf_block_txt = ""
        if existing_strategy_id and getattr(user, "id", 0):
            try:
                _perf_db = SessionLocal()
                try:
                    # Ownership guard: only inject perf for a strategy this user
                    # owns — _compute_strategy_trade_stats queries by id alone, so
                    # without this check a caller could pass another user's id (IDOR).
                    from app.strategy_models import UserStrategy as _UStrat
                    _owned = (
                        _perf_db.query(_UStrat.id)
                        .filter(_UStrat.id == existing_strategy_id,
                                _UStrat.user_id == user.id)
                        .first()
                    )
                    _st = _compute_strategy_trade_stats(_perf_db, existing_strategy_id) if _owned else {"n_closed": 0}
                finally:
                    _perf_db.close()
                if _st.get("n_closed", 0) > 0:
                    _perf_block_txt = (
                        "\nREAL PERFORMANCE OF THIS STRATEGY (use these actual numbers, not guesses):\n"
                        f"  Summary: {_st['compact']}\n"
                        "  Win rate by session (UTC):\n" + _st["session_block"] + "\n"
                        "  Win rate by day of week:\n" + _st["dow_block"] + "\n"
                        f"  Best symbols: {_st['best_coins']}\n"
                        f"  Losing symbols: {_st['worst_coins']}\n"
                        "Ground EVERY suggestion in this data — name the exact losing sessions/days/"
                        "symbols and propose concrete filters (restrict timing, drop a bleeding symbol, "
                        "tighten SL where it's leaking). Do NOT give generic advice when you have real numbers.\n"
                    )
                else:
                    _perf_block_txt = (
                        "\nThis strategy has no closed trades yet — base your assessment on the config "
                        "structure and best practices; suggest running a backtest to validate.\n"
                    )
            except Exception as _perf_err:
                logger.warning(f"Improve-mode perf fetch failed (non-fatal): {_perf_err}")

        _improve_block = (
            "\n\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\n"
            "IMPROVE MODE \u2014 REFINING AN EXISTING STRATEGY\n"
            "You are NOT building from scratch. The user wants to improve their existing strategy.\n\n"
            "Current config:\n"
            "  " + _describe_existing_config(existing_config) + "\n"
            + _perf_block_txt + "\n"
            "Your job:\n"
            "1. Open with a brief honest assessment (3 sentences max): what\u2019s solid, what\u2019s the single biggest weakness (cite the real data above if present), and one specific change you\u2019d try first.\n"
            "2. Let the user steer \u2014 they might want to change signals, tighten stops, add confirmations, or just tweak R:R.\n"
            "3. Make targeted suggestions that ADDRESS the specific weaknesses, not generic advice.\n"
            "4. When ready, compile the FULL revised ###STRATEGY### line \u2014 this will UPDATE the existing strategy (not create a new one), so output all fields even the unchanged ones.\n"
            "\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\n"
        )
    else:
        _improve_block = ""

    system_prompt = f"""You are a veteran algorithmic trading strategist — the user's personal strategy architect inside TradeHub. You trade {market_ctx}. You've spent years building automated strategies and you have strong, specific opinions on what works.

PERSONALITY & STYLE:
- Direct, confident, and opinionated. You give real recommendations, not generic options.
- When a user is vague, you don't ask empty questions — you pitch 2–3 specific named strategy ideas and let them pick.
- You mix signals in creative, logical ways and explain briefly WHY a combo works.
- Reply in 2–5 short sentences max. Conversational, never bullet-pointed. Use trading language naturally: "confluence", "sniper entry", "sweep and reverse", "structure", "imbalance", "distribution", "scale out", "partial target".
- Sound like a mentor, not a form. Use their exact words back.
- If they say something risky (80× leverage, 0.1% SL), flag it once — then still help build it.
- Never ask about paper/live mode — strategies always start paper. Never mention it.
- After compiling, briefly suggest the user run a 30-day backtest to validate before going live.

SIGNAL VARIETY — THIS IS CRITICAL:
- You have a rich IDEA BANK below. Use it. Do NOT default to RSI + EMA every time — that is the most generic, overused combination and users notice when they get the same suggestion repeatedly.
- This session, lean toward: **{_nudge}** — unless the user explicitly asks for something else.
- When pitching 2–3 ideas unprompted, span different signal families (e.g. one SMC, one oscillator, one trend-follow — never three RSI variants). Each idea should feel meaningfully different from the others.
- If the conversation already discussed a signal type, suggest something from a different family next time. Rotate across the IDEA BANK, not the same 2–3 strategies every session.

HANDLING COMPLEX / MULTI-PART REQUESTS:
- When a user describes a detailed, multi-condition setup (e.g. a full ICT stack: killzone + liquidity sweep + market-structure shift + FVG + CISD), treat every named concept as a REQUIRED ingredient. Acknowledge each part briefly so they know you caught all of it — don't silently drop the hard ones.
- Build the FULL stack they asked for rather than trimming to 2–3 confirmations; only push back if two requirements genuinely conflict, and if so name the conflict and propose a fix.
- For sequenced setups ("first a sweep, then structure breaks, then enter on the retrace"), keep the order in how you describe and compile the conditions.
{_improve_block}
WHAT YOU COLLECT (naturally, any order):
- Direction: LONG / SHORT / BOTH
- Primary signal + timeframe
- TP1 and SL (required — see asset rules below for units)
- TP2 (optional scale-out target — ALWAYS suggest it for swing/position trades, strongly recommended for any R:R ≥ 2:1)
  Example: "Want a partial exit at TP1 and let the rest ride to TP2? That's how professionals lock in profit while keeping upside open."
- Trailing stop (optional — suggest it for trend-following setups and breakouts)
- Breakeven (optional — suggest moving SL to entry after TP1 hit; especially important for forex and leveraged crypto)
  Example: "Once price hits TP1, should I automatically move the stop to breakeven? Protects the trade for free."
- Position size % (optional — default 5%; suggest reducing to 2-3% for high-leverage or volatile setups, up to 10% for low-risk swing with SL < 2%)
- Confirmations: every confirmation you add makes the signal RARER — tune this to the goal:
  - Scalp / high-frequency (5m, "fires often", "more trades", "active", "lots of signals"): 0 confirmations — set BOTH to "none". Extra filters kill trade frequency. This is the most common mistake beginners make.
  - Day-trade (15m, 1h, standard): suggest exactly 1 confirmation max.
  - Swing / position (4h, daily, "high conviction", "few good trades"): suggest 2 confirmations.
  - RULE: if the user asks for high frequency, active trading, scalp, or mentions wanting more trades — use zero confirmations. The raw signal alone is the edge.

GREAT STRATEGY PRINCIPLES — apply these silently when building:
- Multi-timeframe confluence is the biggest edge: primary signal on a lower TF, confirmed by trend on a higher TF. Pick the right signal family for the market condition — oscillators for ranging, trend-followers for momentum, SMC for precision.
- At least one structural confirmation (BOS/CHoCH, EMA alignment, SuperTrend direction) prevents counter-trend entries.
- R:R ≥ 2:1 is the minimum for consistency. 3:1 is professional grade.
- Breakeven protection on leveraged trades is not optional — it converts would-be losses into free trades.
- TP2 + trailing stop together is the most profitable exit pattern for trend strategies.
- Tight cooldown (5–15 min) prevents re-entry after a stop out on volatile moves.{asset_rules}

LEVERAGE — never ask about this. Rules by asset:
  FOREX: do NOT mention leverage at all. It is a broker account setting (FP Markets sets it at the account level). The strategy uses pip-based TP/SL and position sizing — leverage is irrelevant here.
  Crypto: pick silently — scalp 15×, momentum 10×, swing 8×, reversal 8×, SMC 10×.
  Stocks: 2× default silently. Indices: 5× default silently.
  Only bring up leverage if the user explicitly asks about it.

COMPILE when you have direction + signal + TP + SL. Don't keep asking once you have enough. You MAY compile with defaults for optional fields if the user doesn't specify (TP2: none, trailing: false, breakeven: 70%, position size: 5%).

When ready to compile, say something natural like "Perfect, locking that in — compiling now." then output EXACTLY on its own line after a blank line:
###STRATEGY###
Asset Class: {asset_class} | Direction: LONG | Style: SCALPER | Primary Signal: {_ex_primary} | Confirmation 1: {_ex_conf1} | Confirmation 2: {_ex_conf2} | {tp_sl_example} | TP2: none | Trailing Stop: false | Breakeven: 70% | Position Size: 5% | Max Trades/Day: 6 | Sessions: none | Trading Days: none | Daily Loss Limit: none | Max Open Positions: 1 | Cooldown: 30min | Leverage: 10x | {symbols_example}

FIELD RULES for the ###STRATEGY### line:
- Sessions: "none" (trade 24/7) or a comma list of asian / london / new_york / overlap (e.g. "Sessions: New York" or "Sessions: London, New York"). ONLY set when the user asks to restrict trading to a session window — otherwise "none". This applies to every asset class and is editable later in the wizard.
- Trading Days: "none" (every day) or specific weekdays (e.g. "Trading Days: Mon-Fri" or "Trading Days: Monday, Wednesday"). ONLY set when the user names days.
- Daily Loss Limit: "none" or a percentage of account that halts trading for the day (e.g. "Daily Loss Limit: 5%").
- Max Open Positions: integer, how many trades may be open at once (default 1).
- Cooldown: minutes to wait after a trade before re-entering (e.g. "Cooldown: 30min" or "Cooldown: 1h").
- When the user gives a configuration instruction (session, days, risk limits, sizing, breakeven, trailing) about an EXISTING strategy, change ONLY what they asked and keep every other field at its current value.
- TP1 MUST be at least 1.5× the SL value (minimum 1.5:1 R:R). Aim for 2:1 or better. NEVER set TP equal to SL — that is a losing setup over time. If a user asks for 1:1, explain why it's bad and suggest 2:1 instead before compiling.
- TP2: "none" or a value using same units as TP1 (e.g. "TP2: 4%" or "TP2 Pips: 60")
- Trailing Stop: "true" or "false"
- Breakeven: "none" or a percentage of TP1 at which to move SL to entry (e.g. "Breakeven: 60%" means move SL to entry when 60% of TP1 distance is covered)
- Position Size: percentage of account per trade (e.g. "Position Size: 3%")
- Max Trades/Day: integer, typically 3–20 depending on style (scalp: 10–20, swing: 2–5)
- Confirmation 1 / Confirmation 2: can be "none" if user doesn't want confirmations (discourage this)
- Use actual values from the conversation. Output ###STRATEGY### exactly once, only when you have enough real info."""

    # ── Personalization: inject the user's REAL saved strategies + performance ──
    # so the AI references what they own, builds on winners, avoids duplicates,
    # and warns on past-loser patterns. Skipped in improve mode (that path gets
    # the focused single-strategy performance block instead). Best-effort: never
    # let a context-build failure break the chat.
    if not existing_config and getattr(user, "id", 0):
        try:
            _ctx_db = SessionLocal()
            try:
                _user_ctx = _build_user_trading_context(_ctx_db, user, asset_class)
            finally:
                _ctx_db.close()
            if _user_ctx:
                system_prompt += "\n" + _user_ctx
        except Exception as _ctx_err:
            logger.warning(f"Chat builder context inject failed (non-fatal): {_ctx_err}")

    api_messages = [{"role": m["role"], "content": m["content"]} for m in messages]

    try:
        client = anthropic.AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        resp = await asyncio.wait_for(
            client.messages.create(
                model="claude-opus-4-8",
                max_tokens=700,
                system=system_prompt,
                messages=api_messages,
            ),
            timeout=30.0,
        )
        raw = resp.content[0].text
    except asyncio.TimeoutError:
        logger.warning("Chat builder timed out after 28s")
        return {
            "reply": "That took too long on my end — try sending your message again, I'm ready.",
            "complete": False,
            "description": None,
        }
    except Exception as e:
        logger.error(f"Chat builder AI error: {e}")
        return {
            "reply": "Sorry, I hit a snag there. Could you repeat that?",
            "complete": False,
            "description": None,
        }

    # Increment usage counter (re-open DB briefly)
    db2 = SessionLocal()
    try:
        sub2 = _get_portal_sub(user.id, db2)
        sub2.chat_calls_used = (sub2.chat_calls_used or 0) + 1
        db2.commit()
        new_used = sub2.chat_calls_used
    finally:
        db2.close()

    complete = "###STRATEGY###" in raw
    description = None
    reply = raw

    if complete:
        parts = raw.split("###STRATEGY###")
        reply = parts[0].strip()
        description = parts[1].strip() if len(parts) > 1 else ""

    return {
        "reply": reply,
        "complete": complete,
        "description": description,
        "calls_used": new_used,
        "calls_limit": limit,
        "is_pro": is_pro,
    }


@app.post("/api/generate-indicator")
async def api_generate_indicator(request: Request):
    """
    Generate strategy entry conditions from a plain-English indicator description.
    Body: { prompt: str, timeframe: str, direction: str, uid: str }
    Returns: { conditions: list, explanation: str }
    No Pro gate — available to all users.
    """
    body = await request.json()
    uid    = (body.get("uid") or "").strip()
    prompt = (body.get("prompt") or "").strip()

    if not prompt:
        raise HTTPException(status_code=400, detail="prompt is required")

    from app.database import SessionLocal
    db = SessionLocal()
    try:
        user = _get_user_by_uid(uid, db)
        if not user:
            raise HTTPException(status_code=403, detail="Unknown UID")
    finally:
        db.close()

    # Build the AI prompt
    system_prompt = """You are an expert algorithmic trading engineer. Your job is to faithfully translate a trading indicator description into precise, executable strategy conditions for a live trading platform.

ANALYSIS INSTRUCTIONS:
- Read the description carefully. Extract every specific parameter mentioned: periods, lengths, multipliers, thresholds, source inputs (close/high/low/hl2), MA types (SMA/EMA/SMMA/WMA).
- If the description or code mentions specific numbers (e.g. CCI period 20, ATR period 1, EMA 50), use those exact values.
- Infer timeframe from context (e.g. "15m chart", "1h RSI", "daily candle"). Default to "15m".
- Infer direction: LONG for bullish/bounce setups, SHORT for bearish/breakdown, BOTH for trend-following both ways.
- Generate 2–5 conditions that capture the CORE logic of the indicator, not generic placeholders.

⚠️ CRITICAL — DIRECTION RULES (the strategy engine evaluates ALL conditions with AND, so they must ALL be true at the same time):
- If the indicator naturally fires in only ONE direction (oversold long, breakdown short, etc.), set direction to "LONG" or "SHORT" and ONLY include conditions that confirm that side. Never mix bullish AND bearish triggers in the same condition list.
- If the indicator can fire BOTH ways (e.g. an SMA cross, SuperTrend flip, EMA ribbon flip), pick the SINGLE direction that is most natural for the description and ONLY return conditions for that side. Set direction = "LONG" or "SHORT", NEVER "BOTH". The user can clone the strategy and invert it to cover the other side.
- Forbidden combinations in the same conditions list: "price_above" + "price_below" of the same MA · "bullish_engulfing" + "bearish_engulfing" · "oversold" + "overbought" · "bullish" + "bearish" of the same indicator · any pair where one rule contradicts another. If you ever feel the urge to include opposing rules, you have picked the wrong direction — pick one side and stop.
- Direction "BOTH" is reserved for trend-following systems where every condition is direction-neutral (e.g. ATR expanding, ADX > 25, volume > 1.5×). Do NOT use BOTH if any condition has a direction-specific operator or pattern.

CONDITION REFERENCE (use EXACT type/name/field names from this list):

━━━ TYPE: "indicator" ━━━ (most technical indicators go here — set "name" to one of:)
• rsi        → { type:"indicator", name:"rsi", timeframe, operator:"gt"|"lt"|"gte"|"lte", value:NUMBER, label }
• macd       → { type:"indicator", name:"macd", timeframe, condition:"bullish_cross"|"bearish_cross"|"positive"|"negative"|"histogram_rising", label }
• macd_hist  → { type:"indicator", name:"macd_hist", timeframe, operator:"gt"|"lt", value:NUMBER, label }
• ema        → { type:"indicator", name:"ema", timeframe, condition:"bullish"|"golden_cross"|"bearish"|"death_cross", label }
• ema_ribbon → { type:"indicator", name:"ema_ribbon", timeframe, periods:[9,21,55,100,200], condition:"aligned_bullish"|"aligned_bearish", label }
• sma        → { type:"indicator", name:"sma", timeframe, period:INT, source:"close"|"high"|"low", condition:"price_above"|"price_below"|"bullish_cross"|"bearish_cross"|"above_high"|"below_low"|"inside_band", period2:INT(opt), label }
• sma_ribbon → { type:"indicator", name:"sma_ribbon", timeframe, periods:[20,50,100,200], condition:"aligned_bullish"|"aligned_bearish", label }
• bb         → { type:"indicator", name:"bb", timeframe, condition:"squeeze"|"above_upper"|"below_lower"|"upper_touch"|"lower_touch"|"overbought"|"oversold"|"mean_reversion", label }
• vwap       → { type:"indicator", name:"vwap", timeframe, condition:"above"|"below", label }
• volume     → { type:"indicator", name:"volume", timeframe, operator:"gt"|"lt", value:NUMBER, label }   (value = ratio vs average, e.g. 1.5 = 50% above average)
• stoch_rsi  → { type:"indicator", name:"stoch_rsi", timeframe, condition:"oversold"|"overbought"|"bullish_cross"|"bearish_cross", label }
• supertrend → { type:"indicator", name:"supertrend", timeframe, period:INT, multiplier:FLOAT, condition:"bullish"|"bearish"|"bullish_flip"|"bearish_flip", label }

━━━ TYPE: "stochastic" ━━━  (full Stochastic Oscillator — %K/%D — distinct from stoch_rsi above)
{ type:"stochastic", timeframe, condition:"oversold"|"overbought"|"bullish_cross"|"bearish_cross", k_period:INT, d_period:INT, label }
  oversold = %K < 20 · overbought = %K > 80 · bullish_cross = %K crosses above %D · bearish_cross = %K crosses below %D

━━━ TYPE: "fx_po3" ━━━  (ICT Power of 3 — accumulation → manipulation sweep → distribution reversal; forex/index)
{ type:"fx_po3", direction:"bullish"|"bearish", sweep_pips:FLOAT, timeframe, label }
  bullish = Asian low swept then price reverses up · bearish = Asian high swept then price reverses down

━━━ TYPE: "wyckoff" ━━━  (Wyckoff phases — all asset classes)
{ type:"wyckoff", phase:"spring"|"shakeout"|"upthrust"|"test"|"markup"|"markdown", lookback:INT, timeframe, label }
  spring/shakeout = bullish wick below support · upthrust = bearish wick above resistance
  test = low-volume re-test after spring/upthrust · markup/markdown = trend phase
• adx        → { type:"indicator", name:"adx", timeframe, condition:"trending"|"strong_trend"|"weak"|"ranging", label }
              OR { type:"indicator", name:"adx", timeframe, operator:"gt"|"lt", value:NUMBER, label }
• atr        → { type:"indicator", name:"atr", timeframe, condition:"expanding"|"contracting", multiplier:FLOAT, label }
• williams_r → { type:"indicator", name:"williams_r", timeframe, period:INT, condition:"oversold"|"overbought", label }
              OR { type:"indicator", name:"williams_r", timeframe, period:INT, operator:"lt"|"gt", value:NUMBER, label }
• cci        → { type:"indicator", name:"cci", timeframe, period:INT, ma_type:"sma"|"ema"|"smma"|"wma"|""|"none", ma_period:INT, condition:"overbought"|"oversold"|"bullish"|"bearish", label }
              OR { type:"indicator", name:"cci", timeframe, period:INT, operator:"gt"|"lt", value:NUMBER, label }
• obv        → { type:"indicator", name:"obv", timeframe, condition:"bullish"|"bearish"|"divergence_bullish"|"divergence_bearish", lookback:INT, label }
• heikin_ashi→ { type:"indicator", name:"heikin_ashi", timeframe, condition:"bullish"|"bearish"|"bullish_flip"|"bearish_flip"|"strong_bull"|"strong_bear", label }
• ichimoku   → { type:"indicator", name:"ichimoku", timeframe, condition:"above_cloud"|"below_cloud"|"in_cloud"|"tk_cross_bullish"|"tk_cross_bearish"|"bullish_cloud"|"bearish_cloud", label }
• keltner    → { type:"indicator", name:"keltner", timeframe, period:INT, multiplier:FLOAT, condition:"squeeze"|"above_upper"|"below_lower"|"inside_bands", label }
• squeeze    → { type:"indicator", name:"squeeze", timeframe, condition:"firing"|"on"|"off"|"bull_mom"|"bear_mom", label }

━━━ TYPE: "candlestick" ━━━
{ type:"candlestick", timeframe, pattern:"bullish_engulfing"|"bearish_engulfing"|"hammer"|"shooting_star"|"pin_bar"|"doji"|"dragonfly_doji"|"gravestone_doji"|"morning_star"|"evening_star"|"three_white_soldiers"|"three_black_crows"|"tweezer_bottom"|"tweezer_top"|"inside_bar"|"outside_bar"|"marubozu", label }

━━━ TYPE: "market_structure" ━━━
{ type:"market_structure", timeframe, condition:"bos_bullish"|"bos_bearish"|"choch_bullish"|"choch_bearish", label }

━━━ TYPE: "fx_cisd" ━━━  (ICT Change in State of Delivery — delivery flips direction)
{ type:"fx_cisd", direction:"bullish"|"bearish", max_run:INT, timeframe, label }
  bullish = latest candle closes ABOVE the open of the last bearish run (sellers done → buyers in)
  bearish = latest candle closes BELOW the open of the last bullish run (buyers done → sellers in)
  max_run = max length of the opposing delivery run to scan (default 10). Pairs with liquidity sweeps + IFVG retests.

━━━ TYPE: "fx_sdp" ━━━  (ICT Sweep → Displacement → Pullback — sequenced entry model)
{ type:"fx_sdp", direction:"bullish"|"bearish", swing_lookback:INT, sweep_window:INT, min_body_ratio:FLOAT, max_age:INT, timeframe, label }
  ONE sequenced setup (order enforced, not 3 separate conditions): liquidity sweep → displacement candle that leaves an FVG → price pulls back into that FVG (fires at entry).
  bullish = sweep lows → up displacement → pull back into bullish FVG (long)
  bearish = sweep highs → down displacement → pull back into bearish FVG (short)
  swing_lookback = bars defining the swept swing extreme (default 20); sweep_window = max bars between sweep & displacement (default 5)
  min_body_ratio = displacement body ≥ N× avg body (default 2.0); max_age = recent bars scanned (default 20)

━━━ TYPE: "price_momentum" ━━━
{ type:"price_momentum", window_minutes:INT, operator:"gt"|"lt", value:FLOAT, direction:"up"|"down"|"any", label }

━━━ TYPE: "volume_spike" ━━━
{ type:"volume_spike", multiplier:FLOAT, label }    (e.g. multiplier:1.5 = volume is 1.5× its average)

━━━ TYPE: "support_resistance" ━━━
{ type:"support_resistance", condition:"at_support"|"at_resistance"|"breakout_above"|"breakout_below"|"between", tolerance_pct:FLOAT, label }

━━━ TYPE: "order_block" ━━━
{ type:"order_block", ob_type:"bullish"|"bearish", timeframe, tolerance_pct:FLOAT, label }

━━━ TYPE: "fvg" ━━━ (Fair Value Gap)
{ type:"fvg", timeframe, condition:"gap_exists"|"price_in_gap"|"approaching", direction:"bullish"|"bearish"|"any", lookback:INT, label }

━━━ TYPE: "divergence" ━━━
{ type:"divergence", indicator:"rsi"|"macd", timeframe, condition:"bullish"|"bearish", label }

━━━ TYPE: "atr_filter" ━━━  (intraday volatility gate — direction-neutral confirmation)
{ type:"atr_filter", condition:"volatile"|"expanding", min_atr_pct:FLOAT, period:INT, lookback:INT, timeframe, label }
  volatile = ATR ≥ min_atr_pct % of price (default 0.3) · expanding = ATR rising vs lookback bars ago (default 5)
  → Use to avoid dead, low-range tape. Pairs with breakouts/momentum.

━━━ TYPE: "rvol" ━━━  (relative volume — current bar volume vs recent average; direction-neutral)
{ type:"rvol", condition:"high"|"low", threshold:FLOAT, period:INT, timeframe, label }
  high = RVOL ≥ threshold (default 1.5) · low = RVOL < threshold · period = bars to average (default 20)
  → "high" confirms real participation behind a move.

━━━ TYPE: "vwap_bands" ━━━  (session VWAP ± standard-deviation bands)
{ type:"vwap_bands", condition:"below_lower"|"above_upper"|"inside", num_std:FLOAT, timeframe, label }
  below_lower = price ≤ VWAP−N·SD (oversold/LONG) · above_upper = price ≥ VWAP+N·SD (overbought/SHORT) · inside = between bands
  num_std = band width in standard deviations (default 2.0). Classic intraday mean-reversion scalp.

━━━ TYPE: "vwap_bias" ━━━  (directional filter — price above/below session VWAP)
{ type:"vwap_bias", condition:"above"|"below", timeframe, label }
  above = price > VWAP (LONG bias) · below = price < VWAP (SHORT bias). The core intraday discipline filter.

━━━ TYPE: "volume_profile" ━━━  (POC / Value Area / high & low volume nodes — FX & gold via broker tick volume)
{ type:"volume_profile", condition:"at_poc"|"in_value"|"above_value"|"below_value"|"at_hvn"|"at_lvn", lookback:INT, bins:INT, value_area_pct:FLOAT, tolerance_pct:FLOAT, timeframe, label }
  at_poc = price back at the Point of Control (biggest-volume magnet) · in_value = inside value area
  above_value = broke above value-area high (VAH) · below_value = broke below value-area low (VAL)
  at_hvn = in a high-volume node (S/R / reversal magnet) · at_lvn = in a low-volume node (fast move / rejection)
  lookback default 120 · bins default 24 · value_area_pct default 70. Needs real volume — won't fire on a flat-volume feed.

━━━ TYPE: "consecutive_candles" ━━━
{ type:"consecutive_candles", timeframe, count:INT, direction:"green"|"red", label }

━━━ OPERATOR values ━━━
"gt" = greater than, "lt" = less than, "gte" = >=, "lte" = <=

IMPORTANT INDICATOR MAPPINGS:
- Trend Magic (CCI + ATR): Use cci condition "bullish"/"bearish" for the trend direction signal, plus atr "expanding" to confirm momentum
- Squeeze Momentum (LazyBear): Use squeeze "firing"/"bull_mom"/"bear_mom"
- SuperTrend (ATR-based): Use supertrend with period/multiplier from description
- Keltner Channel breakout: Use keltner condition "above_upper"/"below_lower"
- Market structure breaks: Use market_structure "bos_bullish"/"choch_bullish"
- CCI zero-cross: Use cci with condition "bullish" (cci > 0) or "bearish" (cci < 0)

━━━ RECOMMENDED RISK LEVELS ━━━
You MUST also recommend take-profit, stop-loss, leverage, and trade frequency that *fit the indicator's nature*.
Use these reference profiles as anchors:
- Scalp (1m–5m, fast oscillators / volume spikes): TP 1.0–2.0%, SL 0.6–1.2%, leverage 10–20x, max 8/day
- Intraday momentum (15m, RSI/MACD/SuperTrend): TP 2.5–4.0%, SL 1.2–2.0%, leverage 8–15x, max 4/day
- Swing trend (1h, EMA ribbon / Ichimoku / SMA cross): TP 5–10%, SL 2.5–4%, leverage 5–10x, max 2/day
- Mean reversion / oversold (BB lower / RSI<30 / Stoch oversold): TP 2–4%, SL 1.5–2.5% (slightly wider than R:R suggests, since reversals chop), leverage 5–10x, max 3/day
- Breakout (Keltner/BB squeeze fire, BoS): TP 4–8%, SL 1.5–2.5%, leverage 8–15x, max 3/day
Always keep R:R ≥ 1.3:1 unless the strategy is explicitly mean-reversion.
If a 2nd TP makes sense (trend following), include "recommended_tp2_pct" at 1.5–2× the first TP.

Return ONLY valid JSON (no markdown, no explanation outside the JSON):
{
  "timeframe": "15m",
  "direction": "LONG"|"SHORT"|"BOTH",
  "conditions": [ <2-5 condition objects from the list above> ],
  "explanation": "2-3 sentences explaining what this indicator measures, the specific parameters it uses, and what market condition triggers an entry.",
  "recommended_tp_pct": 2.5,
  "recommended_sl_pct": 1.5,
  "recommended_tp2_pct": 5.0,
  "recommended_leverage": 10,
  "recommended_max_trades_per_day": 4,
  "recommended_cooldown_minutes": 30,
  "risk_rationale": "1 sentence on why these levels fit this indicator's volatility / hold-time."
}"""

    user_msg = f"Indicator description:\n{prompt}"

    result = None
    # Try Anthropic first
    try:
        import anthropic as _anthropic
        _ac = _anthropic.Anthropic()
        resp = _ac.messages.create(
            model="claude-opus-4-8",
            max_tokens=2000,
            system=system_prompt,
            messages=[{"role": "user", "content": user_msg}],
        )
        result = resp.content[0].text.strip()
    except Exception as e:
        _log.warning(f"[generate-indicator] Anthropic failed: {e}")

    # Fallback to Gemini
    if not result:
        try:
            import google.generativeai as _genai
            _genai.configure(api_key=__import__("os").environ.get("GEMINI_API_KEY", ""))
            m = _genai.GenerativeModel("gemini-2.0-flash")
            resp = m.generate_content(system_prompt + "\n\n" + user_msg)
            result = resp.text.strip()
        except Exception as e:
            _log.warning(f"[generate-indicator] Gemini failed: {e}")

    if not result:
        raise HTTPException(status_code=503, detail="AI service temporarily unavailable. Please try again.")

    # Parse JSON from result
    import json as _json, re as _re
    try:
        # Strip markdown code fences if present
        clean = _re.sub(r"^```(?:json)?\s*|\s*```$", "", result, flags=_re.MULTILINE).strip()
        data = _json.loads(clean)
        return data
    except Exception:
        raise HTTPException(status_code=500, detail="Could not parse AI response. Please try rephrasing your description.")


@app.post("/api/pinescript/import")
async def api_pinescript_import(request: Request):
    """
    Translate a PineScript indicator/strategy into a platform strategy config.
    Body: { pine_code: str, name: str, uid: str }
    Returns: { config, pine_notes, warnings }
    Requires Pro subscription.
    """
    body = await request.json()
    uid = (body.get("uid") or "").strip()
    pine_code = (body.get("pine_code") or "").strip()
    name = (body.get("name") or "PineScript Import").strip()

    if not uid:
        raise HTTPException(status_code=400, detail="uid required")
    if not pine_code:
        raise HTTPException(status_code=400, detail="pine_code required")

    from app.database import SessionLocal
    db = SessionLocal()
    try:
        user = _get_user_by_uid(uid, db)
        if not user or user.banned:
            raise HTTPException(status_code=403, detail="Invalid or banned user")
        sub = _get_portal_sub(user.id, db)
        is_admin = bool(getattr(user, "is_admin", False))
        if not _is_portal_pro(sub) and not is_admin:
            raise HTTPException(status_code=403, detail="Pro subscription required to use PineScript Import")
    finally:
        db.close()

    from app.services.strategy_builder import compile_from_pinescript, validate_strategy

    # ── Compile step — hard 90 s ceiling so the endpoint never hangs ──────────
    try:
        config = await asyncio.wait_for(compile_from_pinescript(pine_code), timeout=90.0)
    except asyncio.TimeoutError:
        return JSONResponse(
            {"error": "Analysis timed out. The script may be too complex — try trimming it to just the signal logic."},
            status_code=504,
        )

    if not config:
        return JSONResponse(
            {"error": "Could not parse this PineScript. Make sure it uses standard indicators (RSI, MACD, EMA, BB, VWAP, CCI, SuperTrend) and try again."},
            status_code=422,
        )

    config["name"] = name
    config["_pine_source"] = pine_code
    config["_build_mode"] = "paper"

    pine_notes    = config.pop("_pine_notes", [])
    pine_warnings = config.pop("_pine_warnings", [])

    # ── Validate concurrently (best-effort, 15 s max — not blocking) ──────────
    try:
        validation = await asyncio.wait_for(validate_strategy(config), timeout=15.0)
    except Exception:
        validation = {"valid": True, "warnings": [], "suggestions": [], "summary": "", "risk_rating": "MEDIUM"}

    all_warnings = pine_warnings + validation.get("warnings", [])

    return JSONResponse({
        "config":      config,
        "pine_notes":  pine_notes,
        "warnings":    all_warnings,
        "suggestions": validation.get("suggestions", []),
        "risk_rating": validation.get("risk_rating", "MEDIUM"),
    })


# ── One-time admin strategy seed (removes itself after first use) ──────────────
@app.get("/admin/seed-strategies")
async def seed_admin_strategies(secret: str = Query(...), fix: bool = Query(False)):
    """Seed dev strategies into production for admin. Protected by admin telegram_id."""
    _require_admin_secret(secret)
    from app.database import SessionLocal
    from app.strategy_models import UserStrategy, StrategyPerformance
    import json as _json
    STRATEGIES = [
        {"name":"test 1","description":"Build a swing trading strategy. Trade direction: both LONG and SHORT.\n\nPrimary entry signal: Price pumped 2% in 10 minutes.\nUniverse: Top gainers only (coins up 5%+ in 24h).\n\nExit targets: Take Profit 1 = 3%. Stop Loss = 2.5%.\nRisk: 10× leverage, 5% position size per trade. Max 9 trades per day, 5 minute cooldown between trades. Max 2 simultaneous positions. Daily loss limit: 10%.","status":"paper","config":{"version":"1.0","name":"test 1","universe":{"type":"all"},"direction":"BOTH","entry_conditions":{"operator":"AND","conditions":[{"type":"price_momentum","window_minutes":10,"operator":"gt","value":2,"direction":"any"}]},"exit":{"take_profit_pct":3,"take_profit2_pct":None,"stop_loss_pct":2.5,"trailing_stop":False},"risk":{"leverage":10,"position_size_pct":5,"max_trades_per_day":9,"max_open_positions":2,"cooldown_minutes":5,"daily_loss_limit_pct":10,"no_duplicate_symbol":True},"filters":{"time_filter":None,"btc_regime":None},"_build_mode":"paper","_category":"swing"},"perf":{"total_trades":9,"wins":3,"losses":6,"win_rate":33.3,"total_pnl_pct":-60.0}},
        {"name":"test2","description":"Build a scalp trading strategy. Trade direction: LONG only.\n\nPrimary entry signal: Volume spike of 1.2× the normal level.\nUniverse: All eligible altcoins.\n\nExit targets: Take Profit 1 = 1.5%. Stop Loss = 1.5%.\nRisk: 15× leverage, 3% position size per trade. Max 10 trades per day, 15 minute cooldown between trades. Max 3 simultaneous positions. Daily loss limit: 5%.","status":"paper","config":{"version":"1.0","name":"test2","universe":{"type":"all","exclude_slow_highcap":True,"min_volume_usd":500000},"direction":"LONG","entry_conditions":{"operator":"AND","conditions":[{"type":"volume_spike","multiplier":1.2}]},"exit":{"take_profit_pct":1.5,"take_profit2_pct":None,"stop_loss_pct":1.5,"trailing_stop":False},"risk":{"leverage":15,"position_size_pct":3,"max_trades_per_day":10,"max_open_positions":3,"cooldown_minutes":15,"daily_loss_limit_pct":5},"filters":{"time_filter":None,"btc_regime":None},"_build_mode":"paper","_category":"scalp"},"perf":{"total_trades":9,"wins":7,"losses":2,"win_rate":77.8,"total_pnl_pct":112.5}},
        {"name":"parabolic pump/dump","description":"Build a reversal trading strategy. Trade direction: SHORT only.\n\nPrimary entry signal: Price pumped 6% in 12 minutes.\nUniverse: All eligible altcoins.\n\nExit targets: Take Profit 1 = 2.5%. Stop Loss = 2%.\nRisk: 20× leverage, 6% position size per trade. Max 15 trades per day, 5 minute cooldown between trades. Max 3 simultaneous positions. Daily loss limit: 5%.","status":"paper","config":{"version":"1.0","name":"parabolic pump/dump","universe":{"type":"all"},"direction":"SHORT","entry_conditions":{"operator":"AND","conditions":[{"type":"price_momentum","window_minutes":12,"operator":"gt","value":6,"direction":"up"}]},"exit":{"take_profit_pct":2.5,"take_profit2_pct":None,"stop_loss_pct":2.0,"trailing_stop":False},"risk":{"leverage":20,"position_size_pct":6,"max_trades_per_day":15,"max_open_positions":3,"cooldown_minutes":5,"daily_loss_limit_pct":10,"no_duplicate_symbol":True},"filters":{"time_filter":None,"btc_regime":None},"_build_mode":"paper","_category":"reversal"},"perf":{"total_trades":11,"wins":5,"losses":6,"win_rate":45.5,"total_pnl_pct":10.0}},
        {"name":"super 1H dump","description":"Build a reversal trading strategy. Trade direction: SHORT only.\n\nPrimary entry signal: Price pumped 12% in 60 minutes.\nUniverse: All eligible altcoins.\n\nExit targets: Take Profit 1 = 3%. Stop Loss = 2.5%.\nRisk: 20× leverage, 6% position size per trade. Max 20 trades per day, 5 minute cooldown between trades. Max 3 simultaneous positions. Daily loss limit: 20%.","status":"paper","config":{"version":"1.0","name":"super 1H dump","universe":{"type":"all"},"direction":"SHORT","entry_conditions":{"operator":"AND","conditions":[{"type":"price_momentum","window_minutes":60,"operator":"gt","value":12,"direction":"up"}]},"exit":{"take_profit_pct":3,"take_profit2_pct":None,"stop_loss_pct":2.5,"trailing_stop":False},"risk":{"leverage":20,"position_size_pct":6,"max_trades_per_day":20,"max_open_positions":3,"cooldown_minutes":5,"daily_loss_limit_pct":20,"no_duplicate_symbol":True},"filters":{"time_filter":None,"btc_regime":None},"_build_mode":"paper","_category":"reversal"},"perf":{"total_trades":9,"wins":6,"losses":3,"win_rate":66.7,"total_pnl_pct":210.0}},
        {"name":"scalptest Long","description":"Build a scalp trading strategy. Trade direction: LONG only.\n\nPrimary entry signal: Price pumped 1% in 10 minutes.\nUniverse: All eligible altcoins.\n\nExit targets: Take Profit 1 = 2%. Stop Loss = 2%.\nRisk: 20× leverage, 4% position size per trade. Max 20 trades per day, 5 minute cooldown between trades. Max 5 simultaneous positions. Daily loss limit: 20%.","status":"paper","config":{"version":"1.0","name":"scalptest Long","universe":{"type":"all"},"direction":"LONG","entry_conditions":{"operator":"AND","conditions":[{"type":"price_momentum","window_minutes":10,"operator":"gte","value":1,"direction":"up"}]},"exit":{"take_profit_pct":2,"take_profit2_pct":None,"stop_loss_pct":2,"trailing_stop":False},"risk":{"leverage":20,"position_size_pct":4,"max_trades_per_day":20,"max_open_positions":5,"cooldown_minutes":5,"daily_loss_limit_pct":20,"no_duplicate_symbol":True},"filters":{"time_filter":None,"btc_regime":None},"_build_mode":"paper","_category":"scalp"},"perf":{"total_trades":10,"wins":9,"losses":1,"win_rate":90.0,"total_pnl_pct":320.0}},
        {"name":"shiiii","description":"Build a smc trading strategy. Trade direction: both LONG and SHORT.\n\nPrimary entry signal: bullish order block touch on 15m.\nUniverse: All eligible altcoins.\n\nExit targets: Take Profit 1 = 3%. Stop Loss = 2%.\nRisk: 25× leverage, 5% position size per trade. Max 10 trades per day, 5 minute cooldown between trades. Max 5 simultaneous positions. Daily loss limit: 20%.","status":"paper","config":{"version":"1.0","name":"shiiii","universe":{"type":"all"},"direction":"BOTH","entry_conditions":{"operator":"AND","conditions":[{"type":"order_block","ob_type":"bullish","timeframe":"15m","tolerance_pct":1.5}]},"exit":{"take_profit_pct":3,"take_profit2_pct":None,"stop_loss_pct":2,"trailing_stop":False},"risk":{"leverage":25,"position_size_pct":5,"max_trades_per_day":10,"max_open_positions":5,"cooldown_minutes":5,"daily_loss_limit_pct":20,"no_duplicate_symbol":True},"filters":{"time_filter":None,"btc_regime":None},"_build_mode":"paper","_category":"smc"},"perf":{"total_trades":2,"wins":1,"losses":1,"win_rate":50.0,"total_pnl_pct":25.0}},
        {"name":"pumpydumpy","description":"Build a reversal trading strategy. Trade direction: SHORT only.\n\nPrimary entry signal: Price pumped 20% in 30 minutes.\nUniverse: All eligible altcoins.\n\nExit targets: Take Profit 1 = 4%. Stop Loss = 3.5%.\nRisk: 20× leverage, 6% position size per trade. Max 20 trades per day, 5 minute cooldown between trades. Max 4 simultaneous positions. Daily loss limit: 20%.","status":"paper","config":{"version":"1.0","name":"pumpydumpy","universe":{"type":"all","exclude_slow_highcap":True,"min_volume_usd":500000},"direction":"SHORT","entry_conditions":{"operator":"AND","conditions":[{"type":"price_momentum","window_minutes":30,"operator":"gt","value":20,"direction":"up"}]},"exit":{"take_profit_pct":4,"take_profit2_pct":None,"stop_loss_pct":3.5,"trailing_stop":False},"risk":{"leverage":20,"position_size_pct":6,"max_trades_per_day":20,"max_open_positions":4,"cooldown_minutes":5,"daily_loss_limit_pct":20,"no_duplicate_symbol":True},"filters":{"time_filter":None,"btc_regime":None},"_build_mode":"paper","_category":"reversal"},"perf":{"total_trades":1,"wins":1,"losses":0,"win_rate":100.0,"total_pnl_pct":80.0}},
    ]
    db = SessionLocal()
    created = []
    try:
        admin_telegram_id = os.getenv("OWNER_TELEGRAM_ID", "").strip()
        if not admin_telegram_id:
            raise HTTPException(status_code=503, detail="OWNER_TELEGRAM_ID is not configured")
        user = db.query(User).filter(User.telegram_id == admin_telegram_id).first()
        if not user:
            raise HTTPException(status_code=404, detail="Admin user not found in this database")
        existing = {s.name: s for s in db.query(UserStrategy).filter(UserStrategy.user_id == user.id).all()}
        fixed = []
        for s in STRATEGIES:
            p = s["perf"]
            if s["name"] in existing:
                if fix:
                    # Patch zeroed performance records back to correct values
                    strat_id = existing[s["name"]].id
                    perf = db.query(StrategyPerformance).filter(
                        StrategyPerformance.strategy_id == strat_id
                    ).first()
                    if perf and perf.total_trades == 0 and p["total_trades"] > 0:
                        perf.total_trades  = p["total_trades"]
                        perf.wins          = p["wins"]
                        perf.losses        = p["losses"]
                        perf.win_rate      = p["win_rate"]
                        perf.total_pnl_pct = p["total_pnl_pct"]
                        fixed.append(s["name"])
                continue
            strat = UserStrategy(
                user_id     = user.id,
                name        = s["name"],
                description = s["description"],
                status      = s["status"],
                config      = s["config"],
            )
            db.add(strat)
            db.flush()
            perf = StrategyPerformance(
                strategy_id   = strat.id,
                total_trades  = p["total_trades"],
                wins          = p["wins"],
                losses        = p["losses"],
                win_rate      = p["win_rate"],
                total_pnl_pct = p["total_pnl_pct"],
            )
            db.add(perf)
            created.append(s["name"])
        db.commit()
        return JSONResponse({"seeded": created, "fixed": fixed, "skipped": [n for n in existing if n not in fixed]})
    finally:
        db.close()


# ── Portal public config (auth/payments UI) ─────────────────
@app.get("/api/portal/config")
async def portal_config():
    return {
        "features_free": portal_features_free(),
        "payments_enabled": payments_enabled(),
        "google_auth_enabled": _google_enabled(),
    }


# ── Portal subscription status ─────────────────────────────
@app.get("/api/portal/subscription")
async def portal_subscription(request: Request):
    uid = request.query_params.get("uid", "")
    from app.database import SessionLocal
    db = SessionLocal()
    try:
        user = _get_user_by_uid(uid, db)
        if not user:
            raise HTTPException(status_code=403)
        sub = _get_portal_sub(user.id, db)
        _, used, lim, is_pro = _chat_calls_info(sub, db, user)
        return {
            "tier": "pro" if is_pro else "free",
            "is_pro": is_pro,
            "chat_calls_used": used,
            "chat_calls_limit": lim,
            "subscription_end": sub.subscription_end.isoformat() if sub.subscription_end else None,
        }
    finally:
        db.close()


# ── Executor health & force-start (admin) ───────────────────
@app.get("/api/admin/executor/status")
async def executor_status(request: Request):
    """Returns whether the strategy executor is running in this worker process."""
    import os as _os
    is_prod = is_production_deploy()
    # Query DB for the advisory lock
    from app.database import SessionLocal
    from sqlalchemy import text
    db = SessionLocal()
    try:
        row = db.execute(
            text(
                "SELECT l.pid, COALESCE(a.state,'gone') AS state "
                "FROM pg_locks l "
                "LEFT JOIN pg_stat_activity a ON a.pid = l.pid "
                "WHERE l.locktype='advisory' AND l.objid=:lid AND l.granted=true "
                "LIMIT 1"
            ),
            {"lid": get_executor_lock_id()},
        ).fetchone()
        lock_info = {"pid": row[0], "state": row[1]} if row else None
    except Exception as e:
        lock_info = {"error": str(e)}
    finally:
        db.close()

    try:
        from app.services.strategy_executor import executor_runtime_profile as _erp
        _profile = _erp()
    except Exception:
        _profile = {}

    return {
        "executor_running_in_this_worker": _executor_running_in_this_worker,
        "executor_tasks_started": _executor_tasks_started,
        "is_production": is_prod,
        "advisory_lock_holder": lock_info,
        "lock_id": get_executor_lock_id(),
        "runtime_profile": _profile,
    }


@app.get("/api/admin/telegram/status")
async def admin_telegram_status(request: Request):
    """Show which bots each token maps to + DB poller lock holders."""
    _require_admin_bearer(request)
    from app.services.telegram_poller_lock import (
        MAIN_POLLER_LOCK_ID,
        FOREX_POLLER_LOCK_ID,
        describe_bot_token,
    )
    from app.services.telegram_tokens import forex_bot_token, main_bot_token

    locks = {}
    from app.database import SessionLocal
    from sqlalchemy import text
    db = SessionLocal()
    try:
        rows = db.execute(
            text(
                "SELECT l.objid, l.pid, COALESCE(a.state, 'gone') AS state "
                "FROM pg_locks l "
                "LEFT JOIN pg_stat_activity a ON a.pid = l.pid "
                "WHERE l.locktype = 'advisory' AND l.granted = true "
                "AND l.objid IN (:main, :forex)"
            ),
            {"main": MAIN_POLLER_LOCK_ID, "forex": FOREX_POLLER_LOCK_ID},
        ).fetchall()
        for objid, pid, state in rows:
            locks[str(objid)] = {"pid": pid, "state": state}
    except Exception as e:
        locks = {"error": str(e)}
    finally:
        db.close()

    return {
        "main_bot": await describe_bot_token(main_bot_token(), "TELEGRAM_BOT_TOKEN"),
        "forex_bot": await describe_bot_token(forex_bot_token(), "FOREX_BOT_TOKEN"),
        "poller_locks": locks,
        "lock_ids": {"main": MAIN_POLLER_LOCK_ID, "forex": FOREX_POLLER_LOCK_ID},
        "hint": (
            "TelegramConflictError on the main bot means two processes poll "
            "TELEGRAM_BOT_TOKEN. Check Railway replicas=1 and remove the token "
            "from Replit. Main and forex bots may both poll if they are different bots."
        ),
    }


@app.post("/api/admin/telegram/test")
async def admin_telegram_test(request: Request):
    """Send a test DM to OWNER_TELEGRAM_ID — verifies trade/health notification path."""
    _require_admin_bearer(request)
    from app.services.telegram_dm import owner_chat_id, send_dm, bot_tokens

    owner = owner_chat_id()
    if not owner:
        raise HTTPException(status_code=503, detail="OWNER_TELEGRAM_ID is not configured")
    if not bot_tokens():
        raise HTTPException(status_code=503, detail="No TELEGRAM_BOT_TOKEN configured")

    ok = await send_dm(
        owner,
        "🧪 <b>TradeHub notification test</b>\n\n"
        "If you see this, trade TP/SL alerts and hourly health reports can reach you.",
    )
    if not ok:
        raise HTTPException(
            status_code=502,
            detail="Telegram rejected the message — open a chat with the bot (/start) and retry",
        )
    return {"status": "delivered", "owner_telegram_id": owner}


@app.post("/api/admin/executor/force-start")
async def executor_force_start(request: Request):
    """
    Admin endpoint — forces this worker to acquire the executor lock and start
    the executor right now.  Use when production shows both workers as HTTP-only.
    Requires the dev secret in the Authorization header.
    """
    import os as _os
    _require_admin_bearer(request)

    global _executor_running_in_this_worker, _executor_tasks_started
    if _executor_running_in_this_worker:
        return {"status": "already_running", "message": "Executor already active in this worker"}

    # Try to force-release stuck lock then re-acquire
    loop = asyncio.get_event_loop()

    def _force_acquire():
        try:
            import psycopg2
            from app.config import settings
            conn = psycopg2.connect(settings.get_database_url())
            conn.autocommit = True
            cur = conn.cursor()
            # Forcefully terminate any backend holding our lock
            cur.execute(
                """
                SELECT pid FROM pg_locks l
                JOIN pg_stat_activity a ON a.pid = l.pid
                WHERE l.locktype='advisory' AND l.objid=%s AND l.granted=true
                """,
                (get_executor_lock_id(),),
            )
            holders = cur.fetchall()
            for (pid,) in holders:
                cur.execute("SELECT pg_terminate_backend(%s)", (pid,))
                logger.info(f"[force-start] Terminated stale lock holder PID={pid}")
            import time; time.sleep(0.5)  # brief pause for PG to release
            cur.execute("SELECT pg_try_advisory_lock(%s)", (get_executor_lock_id(),))
            acquired = cur.fetchone()[0]
            cur.close()
            if acquired:
                return conn
            conn.close()
            return None
        except Exception as e:
            logger.error(f"[force-start] Error: {e}")
            return None

    lock_conn = await loop.run_in_executor(None, _force_acquire)
    if not lock_conn:
        return {"status": "failed", "message": "Could not acquire advisory lock — check DB logs"}

    _executor_running_in_this_worker = True
    asyncio.create_task(_keepalive_then_reclaim(lock_conn))
    try:
        if not _executor_tasks_started:
            await _start_executor_tasks()
            _executor_tasks_started = True
        else:
            await _restart_executor_subsystems_after_lock_reacquire()
        return {"status": "started", "message": "Executor started successfully in this worker"}
    except Exception as e:
        logger.error(f"[force-start] Failed to start tasks: {e}")
        return {"status": "error", "message": str(e)}


@app.get("/api/admin/twitter/growth")
async def twitter_growth_stats(request: Request, days: int = 7):
    """
    Admin endpoint — returns account growth snapshots from X API tracking.
    Query param: days (default 7, max 90)
    """
    _require_admin_bearer(request)

    days = max(1, min(90, days))
    try:
        from app.services.twitter_poster import get_account_growth_summary, get_all_twitter_accounts
        summary = get_account_growth_summary(days=days)
        return {
            "status": "ok",
            "days": days,
            "accounts": summary,
            "total_accounts": len(summary),
        }
    except Exception as e:
        logger.error(f"[twitter/growth] Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/admin/twitter/post-metrics")
async def twitter_post_metrics_stats(request: Request, days: int = 30):
    """
    Admin endpoint — returns aggregated tweet performance by post type.
    Query param: days (default 30, max 90)
    """
    _require_admin_bearer(request)

    days = max(1, min(90, days))
    try:
        import psycopg2, os
        url = os.environ.get("NEON_DATABASE_URL") or os.environ.get("DATABASE_URL")
        conn = psycopg2.connect(url)
        cur = conn.cursor()
        cur.execute("""
            SELECT post_type,
                   COUNT(*) AS posts,
                   ROUND(AVG(impressions))   AS avg_impressions,
                   ROUND(AVG(likes))         AS avg_likes,
                   ROUND(AVG(retweets))      AS avg_retweets,
                   MAX(impressions)          AS best_impressions,
                   SUM(impressions)          AS total_impressions,
                   SUM(likes)                AS total_likes
            FROM twitter_post_metrics
            WHERE metrics_fetched = TRUE
              AND posted_at > NOW() - INTERVAL '1 day' * %s
              AND impressions IS NOT NULL
            GROUP BY post_type
            ORDER BY avg_impressions DESC NULLS LAST
        """, (days,))
        rows = cur.fetchall()
        cur.close()
        conn.close()

        result = []
        for row in rows:
            pt, posts, avg_imp, avg_likes, avg_rt, best_imp, total_imp, total_likes = row
            result.append({
                "post_type":         pt,
                "posts":             posts,
                "avg_impressions":   int(avg_imp) if avg_imp else 0,
                "avg_likes":         int(avg_likes) if avg_likes else 0,
                "avg_retweets":      int(avg_rt) if avg_rt else 0,
                "best_impressions":  int(best_imp) if best_imp else 0,
                "total_impressions": int(total_imp) if total_imp else 0,
                "total_likes":       int(total_likes) if total_likes else 0,
            })
        return {"status": "ok", "days": days, "by_post_type": result}
    except Exception as e:
        logger.error(f"[twitter/post-metrics] Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ── Upgrade to Pro (admin grant or payment webhook) ─────────
@app.post("/api/portal/upgrade")
async def portal_upgrade(request: Request):
    """
    Admin-callable endpoint to grant Pro access.
    Body: { uid, months } — or called from payment webhook in future.
    """
    body = await request.json()
    uid = (body.get("uid") or "").strip()
    months = int(body.get("months", 1))

    from app.database import SessionLocal
    from datetime import timedelta
    db = SessionLocal()
    try:
        user = _get_user_by_uid(uid, db)
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        sub = _get_portal_sub(user.id, db)
        now = datetime.utcnow()
        start = max(now, sub.subscription_end) if sub.subscription_end and sub.subscription_end > now else now
        sub.tier = "pro"
        sub.subscription_start = sub.subscription_start or now
        sub.subscription_end = start + timedelta(days=30 * months)
        db.commit()

        # Award $10 to the user who referred this person (first upgrade only)
        if user.referred_by and not getattr(user, "_referral_credited", False):
            referrer = db.query(User).filter(User.referral_code == user.referred_by).first()
            if referrer:
                referrer.referral_earnings = (referrer.referral_earnings or 0.0) + 10.0
                db.commit()

        return {"success": True, "tier": "pro", "subscription_end": sub.subscription_end.isoformat()}
    finally:
        db.close()


# ── OxaPay checkout ──────────────────────────────────────────────────────────

PRO_PRICE_USD = 50.0   # $ per month

@app.post("/api/portal/checkout")
async def portal_checkout(request: Request):
    """
    Create an OxaPay invoice for a Pro subscription.
    Returns { payLink, trackId } so the frontend can redirect the user.
    """
    uid = _get_session_uid(request)
    if not uid:
        raise HTTPException(status_code=401, detail="Not logged in")

    from app.database import SessionLocal
    from app.strategy_models import PortalPayment
    from app.services.oxapay import OxaPayService
    from app.config import settings
    import asyncio, time, os

    db = SessionLocal()
    try:
        user = _get_user_by_uid(uid, db)
        if not user:
            raise HTTPException(status_code=403, detail="User not found")

        if not payments_enabled():
            raise HTTPException(status_code=503, detail="Payments disabled — portal is free to use")

        if not settings.OXAPAY_MERCHANT_API_KEY:
            raise HTTPException(status_code=503, detail="Payment not configured")

        callback_url = f"{public_base_url(request)}/api/portal/oxapay-webhook"
        order_id     = f"portal-{uid}-{int(time.time())}"

        oxapay  = OxaPayService(settings.OXAPAY_MERCHANT_API_KEY)
        invoice = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: oxapay.create_invoice(
                amount       = PRO_PRICE_USD,
                currency     = "USD",
                description  = "TradeHub Pro — 1 Month",
                order_id     = order_id,
                callback_url = callback_url,
                metadata     = {"uid": uid, "user_id": user.id},
            )
        )

        if not invoice or not invoice.get("trackId"):
            raise HTTPException(status_code=502, detail="Failed to create payment invoice")

        payment = PortalPayment(
            user_id  = user.id,
            track_id = invoice["trackId"],
            amount   = PRO_PRICE_USD,
            months   = 1,
            status   = "pending",
        )
        db.add(payment)
        db.commit()

        return {"payLink": invoice["payLink"], "trackId": invoice["trackId"]}
    finally:
        db.close()


@app.get("/api/portal/payment-status")
async def portal_payment_status(track_id: str = Query(...)):
    """Poll endpoint — frontend checks this every 10 s after opening OxaPay link."""
    from app.database import SessionLocal
    from app.strategy_models import PortalPayment
    db = SessionLocal()
    try:
        payment = db.query(PortalPayment).filter(PortalPayment.track_id == track_id).first()
        if not payment:
            return {"status": "not_found"}
        return {"status": payment.status}
    finally:
        db.close()


@app.post("/api/portal/oxapay-webhook")
async def portal_oxapay_webhook(request: Request):
    """
    OxaPay sends a POST here when a payment status changes.
    On 'Paid', upgrade the matching user to Pro and send them a Telegram DM.
    """
    import json as _json
    from app.database import SessionLocal
    from app.strategy_models import PortalPayment
    from app.services.oxapay import OxaPayService
    from app.config import settings
    from datetime import timedelta

    raw  = await request.body()
    body = raw.decode("utf-8")

    # Verify OxaPay HMAC-SHA512 signature
    merchant_key = (settings.OXAPAY_MERCHANT_API_KEY or "").strip()
    if not merchant_key:
        logger.error("[oxapay-webhook] OXAPAY_MERCHANT_API_KEY is not configured")
        raise HTTPException(status_code=503, detail="Payment webhook is not configured")
    sig    = request.headers.get("hmac", "").strip()
    oxapay = OxaPayService(merchant_key)
    if not sig:
        logger.warning("[oxapay-webhook] Missing HMAC signature")
        raise HTTPException(status_code=400, detail="Missing signature")
    if not oxapay.verify_webhook_signature(sig, body, merchant_key):
        logger.warning("[oxapay-webhook] Invalid HMAC signature — ignoring")
        raise HTTPException(status_code=400, detail="Invalid signature")

    try:
        data = _json.loads(body)
    except Exception:
        raise HTTPException(status_code=400, detail="Bad JSON")

    status   = data.get("status", "")
    track_id = str(data.get("trackId", ""))
    logger.info(f"[oxapay-webhook] status={status} trackId={track_id}")

    if status != "Paid":
        return {"ok": True}   # nothing to do for Waiting/Expired/Canceled

    db = SessionLocal()
    try:
        payment = db.query(PortalPayment).filter(PortalPayment.track_id == track_id).first()
        if not payment:
            logger.warning(f"[oxapay-webhook] No PortalPayment for trackId={track_id}")
            return {"ok": True}
        if payment.status == "paid":
            return {"ok": True}   # idempotent

        inquiry = await asyncio.to_thread(oxapay.check_payment_status, track_id)
        if not inquiry or str(inquiry.get("status", "")).lower() != "paid":
            logger.warning(f"[oxapay-webhook] Inquiry did not confirm Paid for trackId={track_id}: {inquiry}")
            raise HTTPException(status_code=502, detail="Payment status could not be verified")

        from decimal import Decimal, InvalidOperation
        def _decimal_value(*values):
            for value in values:
                if value is None or value == "":
                    continue
                try:
                    return Decimal(str(value))
                except (InvalidOperation, ValueError):
                    continue
            return None

        paid_amount = _decimal_value(inquiry.get("amount"), inquiry.get("payAmount"), data.get("amount"))
        paid_currency = str(inquiry.get("currency") or data.get("currency") or "").upper()
        expected_amount = Decimal(str(payment.amount))
        if paid_amount is None or paid_amount < expected_amount or paid_currency != "USD":
            logger.warning(
                "[oxapay-webhook] Payment mismatch trackId=%s paid=%s %s expected=%s USD",
                track_id, paid_amount, paid_currency, expected_amount,
            )
            raise HTTPException(status_code=400, detail="Payment amount or currency mismatch")

        payment.status  = "paid"
        payment.paid_at = datetime.utcnow()

        sub   = _get_portal_sub(payment.user_id, db)
        now   = datetime.utcnow()
        start = max(now, sub.subscription_end) if sub.subscription_end and sub.subscription_end > now else now
        sub.tier               = "pro"
        sub.subscription_start = sub.subscription_start or now
        sub.subscription_end   = start + timedelta(days=30 * payment.months)
        db.commit()

        # Referral credit — $10 to referrer on first upgrade
        user = db.query(User).filter(User.id == payment.user_id).first()
        if user and user.referred_by:
            referrer = db.query(User).filter(User.referral_code == user.referred_by).first()
            if referrer:
                referrer.referral_earnings = (referrer.referral_earnings or 0.0) + 10.0
                db.commit()
                import asyncio as _aio2
                _aio2.create_task(_notify_admin_pro_referral(
                    new_name=user.first_name or user.username or "User",
                    new_uid=user.uid or "",
                    referrer_name=referrer.first_name or referrer.username or "User",
                    referrer_uid=referrer.uid or "",
                ))

        # Telegram DM to subscriber
        if user:
            tg_id = _telegram_int_id(user)
            if tg_id:
                ends = sub.subscription_end.strftime("%d %b %Y")
                asyncio.create_task(_tg_send(
                    tg_id,
                    f"🎉 <b>TradeHub Pro activated!</b>\n"
                    f"Payment confirmed (${payment.amount:.0f}). "
                    f"Your Pro subscription runs until <b>{ends}</b>.\n\n"
                    f"Enjoy unlimited AI chat, live strategy automation, and marketplace access!"
                ))

        logger.info(f"[oxapay-webhook] Upgraded user_id={payment.user_id} to Pro until {sub.subscription_end}")
        return {"ok": True}
    finally:
        db.close()


@app.get("/api/referral-info")
async def api_referral_info(uid: str = Query(...)):
    """Return the user's referral code, link, count of referrals, and earnings."""
    from app.database import SessionLocal
    db = SessionLocal()
    try:
        user = _get_user_by_uid(uid, db)
        if not user:
            raise HTTPException(status_code=403)

        # Auto-generate referral code if missing
        if not user.referral_code:
            import random as _r, string as _s
            _ch = _s.ascii_uppercase + _s.digits
            for _ in range(20):
                _c = "REF-" + "".join(_r.choices(_ch, k=6))
                if not db.query(User).filter(User.referral_code == _c).first():
                    user.referral_code = _c
                    break
            db.commit()
            db.refresh(user)

        # Build referred-users list with tier info
        import json as _json2
        from app.strategy_models import PortalSubscription
        referred_users_raw = db.query(User).filter(User.referred_by == user.referral_code).all()
        referred_list = []
        for ru in referred_users_raw:
            sub = db.query(PortalSubscription).filter(PortalSubscription.user_id == ru.id).first()
            tier = "pro" if (sub and sub.tier == "pro" and
                             (not sub.subscription_end or datetime.utcnow() < sub.subscription_end)) else "free"
            referred_list.append({
                "name":   ru.first_name or ru.username or "User",
                "uid":    ru.uid or "",
                "tier":   tier,
                "joined": ru.created_at.strftime("%d %b %Y") if ru.created_at else "",
            })

        # Parse payout wallet
        payout_address = ""
        payout_network = "SOL"
        if user.crypto_wallet:
            try:
                pw = _json2.loads(user.crypto_wallet)
                payout_address = pw.get("address", "")
                payout_network = pw.get("network", "SOL")
            except Exception:
                payout_address = user.crypto_wallet

        return JSONResponse({
            "referral_code":      user.referral_code,
            "referral_earnings":  float(user.referral_earnings or 0.0),
            "referred_count":     len(referred_list),
            "referred_users":     referred_list,
            "payout_address":     payout_address,
            "payout_network":     payout_network,
        })
    finally:
        db.close()


@app.put("/api/referral/payout")
async def api_put_referral_payout(request: Request, uid: str = Query(...)):
    """Save a user's payout wallet address and network."""
    import json as _json3
    from app.database import SessionLocal
    db = SessionLocal()
    try:
        user = _get_user_by_uid(uid, db)
        if not user:
            raise HTTPException(status_code=403)
        body = await request.json()
        network = (body.get("network") or "SOL").strip()
        address = (body.get("address") or "").strip()
        if network not in ("SOL", "USDT-SOL"):
            raise HTTPException(status_code=400, detail="Network must be SOL or USDT-SOL")
        user.crypto_wallet = _json3.dumps({"network": network, "address": address})
        db.commit()
        return JSONResponse({"success": True})
    finally:
        db.close()


# ── Wizard: AI Name Generator ───────────────────────────────
@app.post("/api/wizard/suggest-name")
async def wizard_suggest_name(request: Request):
    """Free for all users — lightweight Claude Haiku call to suggest strategy names."""
    import anthropic, json as _json
    body = await request.json()
    uid = (body.get("uid") or "").strip()

    from app.database import SessionLocal
    db = SessionLocal()
    try:
        user = _get_user_by_uid(uid, db)
        if not user:
            raise HTTPException(status_code=403, detail="Invalid UID")
    finally:
        db.close()

    style      = body.get("style") or "general"
    direction  = body.get("direction") or "LONG"
    signal     = body.get("primaryType") or "rsi"
    confirms   = body.get("confirms") or []
    tp         = body.get("tp1", 3)
    sl         = body.get("sl", 1.5)
    leverage   = body.get("leverage", 10)
    coins      = body.get("coins") or "all"
    timeframe  = body.get("timeframe") or "5m"
    conf_labels = ", ".join(c.get("type","") for c in confirms) if confirms else "none"

    prompt = f"""You are naming a crypto perpetuals trading strategy. Generate exactly 3 short, creative, memorable names.

Strategy config:
- Style: {style}
- Direction: {direction}
- Entry signal: {signal}
- Confirmations: {conf_labels}
- Timeframe: {timeframe}
- Take Profit: {tp}%, Stop Loss: {sl}%, Leverage: {leverage}x
- Coin universe: {coins}

Rules:
- Each name: 2-4 words, professional, reflects strategy character
- Tagline: max 55 characters, describes what the strategy does
- No generic names like "Crypto Strategy" or "My Strategy"
- Be creative: use trading terminology, market vibes, or poetic refs

Return ONLY this JSON (no markdown, no extra text):
{{"names":[{{"name":"...","tagline":"..."}},{{"name":"...","tagline":"..."}},{{"name":"...","tagline":"..."}}]}}"""

    try:
        client = anthropic.AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        msg = await client.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=300,
            messages=[{"role": "user", "content": prompt}]
        )
        result = _json.loads(msg.content[0].text)
        return result
    except Exception as e:
        logger.error(f"Wizard name suggestion error: {e}")
        return {"names": [
            {"name": f"{style.title()} {signal.replace('_',' ').title()}", "tagline": f"{direction} · {timeframe} · TP {tp}% / SL {sl}%"},
            {"name": f"Alpha {timeframe} {direction.title()}", "tagline": f"Precision {signal.replace('_',' ')} entries"},
            {"name": f"Signal {leverage}X Runner", "tagline": f"{style.title()} strategy with {leverage}x leverage"},
        ]}


# ── AI Strategy Advisor (Pro only) ─────────────────────────
@app.post("/api/strategy-advisor")
async def strategy_advisor(request: Request):
    """
    Pro-only: Ask AI to review or improve a specific strategy.
    Body: { uid, strategy_id, messages: [{role, content}] }
    """
    import anthropic
    body = await request.json()
    uid = (body.get("uid") or "").strip()
    strategy_id = body.get("strategy_id")
    messages = body.get("messages") or []

    from app.database import SessionLocal
    db = SessionLocal()
    try:
        user = _get_user_by_uid(uid, db)
        if not user:
            raise HTTPException(status_code=403, detail="Invalid UID")
        sub = _get_portal_sub(user.id, db)
        if not _is_portal_pro(sub) and not getattr(user, "is_admin", False):
            return {"reply": None, "pro_required": True}

        from app.strategy_models import UserStrategy, StrategyPerformance, StrategyExecution
        strategy = db.query(UserStrategy).filter(
            UserStrategy.id == strategy_id,
            UserStrategy.user_id == user.id
        ).first()
        if not strategy:
            raise HTTPException(status_code=404, detail="Strategy not found")

        perf = db.query(StrategyPerformance).filter(
            StrategyPerformance.strategy_id == strategy_id
        ).first()

        # ── Trade analytics (shared helper — same math used by improve mode
        #    and portfolio review) ──────────────────────────────────────────────
        _adv_stats    = _compute_strategy_trade_stats(db, strategy_id, limit=100)
        session_block = _adv_stats["session_block"]
        dow_block     = _adv_stats["dow_block"]
        best_coins    = _adv_stats["best_coins"]
        worst_coins   = _adv_stats["worst_coins"]
        trade_log     = _adv_stats["trade_log"]

        # Strategy config summary
        try:
            cfg  = strategy.config or {}
            strat_dir  = cfg.get("direction", "BOTH")
            strat_tf   = cfg.get("timeframe", "?")
            tp_pct     = cfg.get("take_profit_pct", "?")
            sl_pct     = cfg.get("stop_loss_pct", "?")
            lev        = cfg.get("leverage", "?")
            max_trades = cfg.get("max_trades_per_day", "?")
            conditions = cfg.get("entry_conditions", [])
            cond_lines = []
            for c in (conditions or []):
                ctype = c.get("type", "?") if isinstance(c, dict) else str(c)
                cond_lines.append(f"  - {ctype}: {c}")
            cond_block = "\n".join(cond_lines) if cond_lines else "  (no conditions set)"
        except Exception:
            strat_dir = strat_tf = tp_pct = sl_pct = lev = max_trades = "?"
            cond_block = "  (error reading config)"

        # Overall perf summary
        try:
            if perf and perf.total_trades:
                wr_overall = round(perf.win_rate or 0, 1)
                pnl_total  = round(perf.total_pnl_pct or 0, 2)
                avg_win    = round(perf.avg_win_pct or 0, 1)
                avg_loss   = round(perf.avg_loss_pct or 0, 1)
                perf_block = (
                    f"Total trades: {perf.total_trades} | Win rate: {wr_overall}% | "
                    f"Total P&L: {pnl_total:+}% | Avg win: +{avg_win}% | Avg loss: {avg_loss}%"
                )
            else:
                perf_block = "No closed trades yet."
        except Exception:
            perf_block = "Performance data unavailable."

        system_prompt = f"""You are an expert crypto trading strategy analyst embedded in TradeHub Markets.

The user is asking about their strategy: "{strategy.name}"
Description: {strategy.description or "none"}

━━ STRATEGY CONFIG ━━
Direction: {strat_dir} | Timeframe: {strat_tf} | TP: {tp_pct}% | SL: {sl_pct}% | Leverage: {lev}x | Max trades/day: {max_trades}
Entry conditions:
{cond_block}

━━ OVERALL PERFORMANCE ━━
{perf_block}

━━ WIN RATE BY SESSION (UTC) ━━
{session_block}

━━ WIN RATE BY DAY OF WEEK ━━
{dow_block}

━━ BEST COINS ━━
{best_coins or "not enough data"}

━━ WORST COINS ━━
{worst_coins or "none negative yet"}

━━ LAST 25 TRADES (oldest→newest) ━━
{trade_log}

━━ YOUR JOB ━━
- You have the REAL trade data above — use it to give specific, data-driven answers
- Call out exactly which sessions / days / coins are underperforming with the actual numbers
- Give concrete suggestions: add a filter for a specific time window, avoid certain coins, tighten SL on weekends, etc.
- Be direct and concise — 2-5 sentences, no generic advice
- Never say you don't have access to trade data — you have it all above"""

        api_messages = [{"role": m["role"], "content": m["content"]} for m in messages]

    finally:
        db.close()

    try:
        client = anthropic.AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        resp = await client.messages.create(
            model="claude-opus-4-8",
            max_tokens=500,
            system=system_prompt,
            messages=api_messages,
        )
        return {"reply": resp.content[0].text, "pro_required": False}
    except Exception as e:
        logger.error(f"Strategy advisor AI error: {e}")
        return {"reply": "Sorry, I hit an issue — please try again.", "pro_required": False}


@app.post("/api/portfolio-review")
async def portfolio_review(request: Request):
    """
    Pro-only: AI review of the user's ENTIRE strategy portfolio at once —
    what's working, what's bleeding, where they overlap, and what's missing.
    Body: { uid }
    Returns: { review: str|None, pro_required: bool, n_strategies: int }
    """
    import anthropic
    body = await request.json()
    uid  = (body.get("uid") or "").strip()

    from app.database import SessionLocal
    db = SessionLocal()
    try:
        user = _get_user_by_uid(uid, db)
        if not user:
            raise HTTPException(status_code=403, detail="Invalid UID")
        sub = _get_portal_sub(user.id, db)
        if not _is_portal_pro(sub) and not getattr(user, "is_admin", False):
            return {"review": None, "pro_required": True, "n_strategies": 0}

        from app.strategy_models import UserStrategy, StrategyPerformance

        rows = (
            db.query(UserStrategy, StrategyPerformance)
            .outerjoin(StrategyPerformance, StrategyPerformance.strategy_id == UserStrategy.id)
            .filter(
                UserStrategy.user_id == user.id,
                UserStrategy.status.in_(["active", "paused", "draft"]),
            )
            .all()
        )
        if not rows:
            return {
                "review": "You don't have any saved strategies yet. Build your first one with the "
                          "AI builder or the wizard, paper-test it for ~30 days, then come back and "
                          "I'll review your whole portfolio.",
                "pro_required": False,
                "n_strategies": 0,
            }

        # ── Build a detailed portfolio digest (config + real perf + per-strategy
        #    session/symbol leaks) for the AI to reason over ───────────────────
        def _sig_types(cfg):
            try:
                conds = (cfg.get("entry_conditions") or {}).get("conditions") or []
                t = [c.get("type", "?") for c in conds if isinstance(c, dict)]
                return "+".join(t[:4]) if t else "?"
            except Exception:
                return "?"

        digest_lines = []
        active_n = paused_n = draft_n = 0
        total_trades = 0
        for strat, perf in rows:
            cfg = strat.config or {}
            st  = (strat.status or "active")
            if   st == "active": active_n += 1
            elif st == "paused": paused_n += 1
            elif st == "draft":  draft_n  += 1
            ac  = strat.asset_class or cfg.get("asset_class", "crypto")
            sig = _sig_types(cfg)
            tf  = "?"
            try:
                conds = (cfg.get("entry_conditions") or {}).get("conditions") or []
                if conds and isinstance(conds[0], dict):
                    tf = conds[0].get("timeframe", "?")
            except Exception:
                pass
            direction = cfg.get("direction", "BOTH")

            line = f'• "{strat.name}" [{ac}/{direction}/{sig} on {tf}, {st}]'
            if perf and perf.total_trades:
                total_trades += perf.total_trades
                wr  = round(perf.win_rate or 0)
                pnl = round(perf.total_pnl_pct or 0, 1)
                line += f" — {perf.total_trades} trades, {wr}% WR, {pnl:+}% total"
                # add session/symbol leaks for strategies with real history
                stats = _compute_strategy_trade_stats(db, strat.id, limit=100)
                extras = []
                if stats.get("worst_session"):
                    extras.append(f"weak session {stats['worst_session']}")
                if stats.get("worst_coins") and stats["worst_coins"] != "none negative yet":
                    extras.append(f"losing symbols {stats['worst_coins']}")
                if extras:
                    line += " | " + "; ".join(extras)
            else:
                line += " — no closed trades yet"
            digest_lines.append(line)

        digest = "\n".join(digest_lines)
        n_strat = len(rows)
        overview = (
            f"{n_strat} strategies total ({active_n} active, {paused_n} paused, {draft_n} draft); "
            f"{total_trades} closed trades across the portfolio."
        )

        system_prompt = f"""You are a veteran portfolio strategist inside TradeHub Markets, reviewing a Pro trader's ENTIRE strategy book at once.

PORTFOLIO OVERVIEW:
{overview}

ALL STRATEGIES (config + real performance + known leaks):
{digest}

Write a sharp, specific portfolio review in plain prose (NOT JSON, NOT code). Cover, in this order, using short labelled paragraphs:
1. WORKING — name the 1–3 strategies that are genuinely performing and why (cite their real numbers).
2. BLEEDING — name the strategies losing money or with weak win rates; for each, give ONE concrete fix grounded in the data (e.g. "cut its worst session", "drop the losing symbol", "tighten SL").
3. OVERLAP — call out strategies that are too similar (same asset/signal/timeframe) and are effectively competing for the same trades; suggest consolidating or differentiating.
4. GAPS — what's missing from the book (e.g. no short-side strategy, no forex exposure, everything on one timeframe, no mean-reversion to balance the trend-following) and suggest 1–2 specific additions.
5. NEXT MOVE — one clear, prioritized recommendation for what to do this week.

Rules:
- Use the REAL numbers above. Never say you lack data.
- Be direct and concrete. Reference strategies by their exact names in quotes.
- No generic platitudes. Every point must be actionable.
- Keep it tight — aim for ~250–350 words total."""

    finally:
        db.close()

    try:
        client = anthropic.AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        resp = await asyncio.wait_for(
            client.messages.create(
                model="claude-opus-4-8",
                max_tokens=1100,
                system=system_prompt,
                messages=[{"role": "user", "content": "Review my entire strategy portfolio."}],
            ),
            timeout=45.0,
        )
        return {"review": resp.content[0].text, "pro_required": False, "n_strategies": n_strat}
    except asyncio.TimeoutError:
        logger.warning("Portfolio review timed out")
        return {"review": "The review took too long to generate — please try again.",
                "pro_required": False, "n_strategies": n_strat}
    except Exception as e:
        logger.error(f"Portfolio review AI error: {e}")
        return {"review": "Sorry, I hit an issue generating your portfolio review — please try again.",
                "pro_required": False, "n_strategies": n_strat}


@app.post("/api/backtest/scan")
async def backtest_scan(request: Request):
    """
    AI Strategy Scanner — run 40 strategy templates against historical data
    across one-or-more coins and one-or-more timeframes in parallel, rank
    every (strategy, coin, TF) combination by composite score, return the
    top winners with full trade history & equity curves for charting.
    Pro subscribers only.

    Body: {
      uid, days, direction,
      coin                — legacy single coin (used if `coins` not given)
      coins               — optional list of tickers, max 5 (multi-coin scan)
      timeframes          — optional list, subset of ["15m","1h","4h"] (multi-TF scan)
      risk_mode           — "optimised" (default) runs the 2-stage TP/SL search across
                            4 profiles and returns the historically best variant per
                            combo. "fixed" pins a single user-supplied TP/SL pair, runs
                            ONLY stage 1, and skips the optimisation pass entirely.
      fixed_tp, fixed_sl  — only used when risk_mode=="fixed". Both are percentages
                            (e.g. 3.0 for 3%). Clamped to sane ranges server-side.
    }
    """
    body      = await request.json()
    uid       = (body.get("uid") or "").strip()
    days      = int(body.get("days", 30))
    direction = (body.get("direction") or "LONG").upper()
    risk_mode = (body.get("risk_mode") or "optimised").lower()
    # Asset-class routing for stocks / forex / indices. Defaults to crypto so
    # legacy callers (mobile pre-multi-asset, web wizard) keep working.
    try:
        from app.services.asset_classes import normalize_asset_class as _norm_ac
        asset_class = _norm_ac(body.get("asset_class"))
    except Exception:
        asset_class = (body.get("asset_class") or "crypto").lower().strip() or "crypto"
    if risk_mode not in ("optimised", "fixed"):
        risk_mode = "optimised"

    # Parse + clamp fixed TP/SL when in fixed mode. We accept any positive
    # number but bound it so a fat-finger 9999% can't break the engine.
    def _clamp_pct(v, lo, hi, default):
        try:
            f = float(v)
            if not (f > 0): return default
            return max(lo, min(hi, f))
        except (TypeError, ValueError):
            return default
    fixed_tp = _clamp_pct(body.get("fixed_tp"), 0.1, 50.0, 3.0)
    fixed_sl = _clamp_pct(body.get("fixed_sl"), 0.1, 25.0, 1.5)

    # ── Coin universe — accept either single `coin` or list `coins` ────────────
    coins_raw = body.get("coins")
    if isinstance(coins_raw, list) and coins_raw:
        coin_list = [str(c).upper().strip().replace("USDT", "") for c in coins_raw if str(c).strip()]
    else:
        coin_list = [(body.get("coin") or "BTC").upper().strip().replace("USDT", "")]
    # Dedupe + cap at 12 to keep total backtest count reasonable. At 12 coins ×
    # 3 TFs × 40 strategies = 1,440 stage-1 backtests + ~96 stage-2 variants;
    # bounded by Semaphore(20) + 120s/90s stage timeouts. Going higher than 12
    # risks tripping the stage-1 timeout on slow candle providers.
    seen = set(); coin_list = [c for c in coin_list if not (c in seen or seen.add(c))][:12]
    if not coin_list:
        coin_list = ["BTC"]

    # ── Timeframes — subset of supported list, default ["1h"] ──────────────────
    tfs_raw = body.get("timeframes")
    SUPPORTED_TFS = ["15m", "1h", "4h"]
    if isinstance(tfs_raw, list) and tfs_raw:
        tf_list = [t.lower() for t in tfs_raw if t.lower() in SUPPORTED_TFS]
    else:
        tf_list = ["1h"]
    if not tf_list:
        tf_list = ["1h"]
    tf_list = list(dict.fromkeys(tf_list))[:3]  # dedupe, max 3

    if days not in (30, 90):
        days = 30
    if direction not in ("LONG", "SHORT"):
        direction = "LONG"

    # Auth + Pro check — wrapped in a retry-safe block because the User /
    # PortalSubscription queries occasionally hit Postgres `statement_timeout`
    # on busy workers, and the previous `[500] OperationalError` would surface
    # to the UI as a generic "Unknown error" with no recourse.
    from app.database import SessionLocal
    from sqlalchemy.exc import OperationalError as _SAOperationalError

    def _auth_check():
        for attempt in (0, 1):
            db = SessionLocal()
            try:
                user = _get_user_by_uid_safe(uid, db)
                if not user:
                    return ("invalid", None)
                sub = _get_portal_sub(user.id, db)
                is_pro = _is_portal_pro(sub) or bool(getattr(user, "is_admin", False))
                return ("ok" if is_pro else "not_pro", None)
            except _SAOperationalError as exc:
                try: db.rollback()
                except Exception: pass
                logger.warning(f"[scan] auth-check DB timeout (attempt {attempt + 1}): {exc}")
                if attempt == 0:
                    import time as _t; _t.sleep(0.25)
                    continue
                return ("db_busy", str(exc))
            except HTTPException:
                # _get_user_by_uid_safe raises 503 on persistent DB failure.
                return ("db_busy", None)
            finally:
                try: db.close()
                except Exception: pass
        return ("db_busy", None)

    auth_status, _ = await asyncio.to_thread(_auth_check)
    if auth_status == "invalid":
        raise HTTPException(status_code=403, detail="Invalid UID")
    if auth_status == "not_pro":
        return JSONResponse(
            status_code=402,
            content={"error": "PRO_REQUIRED",
                     "message": "A Pro subscription is required to use the Strategy Scanner."},
        )
    if auth_status == "db_busy":
        return JSONResponse(
            status_code=503,
            content={"error": "DB_BUSY",
                     "message": "Our database is busy right now — please try again in a few seconds."},
        )

    # Normalised symbols (engine input) + display tickers. For crypto we
    # auto-append USDT; for tradfi (stocks/forex/indices) the symbol IS the
    # display ticker (AAPL, EURUSD, SPX) so the two lists are identical.
    if asset_class == "crypto":
        symbols  = [(c if c.endswith("USDT") else c + "USDT") for c in coin_list]
    else:
        symbols  = coin_list[:]
    tickers  = coin_list[:]
    primary_ticker = tickers[0]   # used for AI insight / legacy `coin` field

    # ── Strategy templates to scan ─────────────────────────────────────────────
    # All use the same coin/direction/days; TP/SL/leverage kept conservative so
    # the ranking reflects signal quality rather than raw risk settings.
    _tp, _sl, _lev = (3.0, 1.5, 5) if direction == "LONG" else (3.0, 1.5, 5)

    def _C(cond: dict) -> dict:
        """Flip a confirm/primary-cfg condition dict for SHORT direction.

        Indicators that are direction-agnostic (adx_filter, atr_volatility,
        keltner squeeze, volume_spike) are returned unchanged.
        """
        c = dict(cond)
        t = c.get("type", "")
        if t == "rsi":
            if c.get("operator") == "lt":   c["operator"] = "gt"; c["value"] = 70
            elif c.get("operator") == "gt" and float(c.get("value", 0)) >= 50:
                c["operator"] = "lt"; c["value"] = 50
            else:                           c["operator"] = "gt"; c["value"] = 70
        elif t == "macd":
            c["condition"] = c.get("condition", "").replace("bullish", "bearish")
        elif t == "ema":
            sub = c.get("condition", "")
            c["condition"] = (sub.replace("bullish_cross", "bearish_cross")
                                 .replace("above", "below")
                                 .replace("crosses_above", "crosses_below"))
        elif t == "sma":
            c["condition"] = c.get("condition", "").replace("price_above", "price_below")
        elif t == "supertrend":   c["condition"] = "bearish"
        elif t == "bb":
            c["condition"] = (c.get("condition", "")
                                .replace("below_lower", "above_upper")
                                .replace("price_below_lower", "price_above_upper")
                                .replace("price_near_lower", "price_near_upper"))
        elif t == "stochrsi":
            if c.get("operator") == "lt":  c["operator"] = "gt"; c["value"] = 80
        elif t == "williams_r":   c["condition"] = "overbought" if c.get("condition") == "oversold" else c.get("condition", "overbought")
        elif t == "breakout":     c["bo_dir"] = "down" if c.get("bo_dir", "up") == "up" else "up"
        elif t == "support_resistance":
            c["condition"] = {"at_support":"at_resistance", "breakout_above":"breakout_below",
                              "at_resistance":"at_support", "breakout_below":"breakout_above"}.get(c.get("condition","at_support"), "at_resistance")
        elif t == "candlestick":
            p = c.get("pattern", c.get("condition", ""))
            new_p = {"bullish_engulfing":"bearish_engulfing", "hammer":"shooting_star",
                     "bearish_engulfing":"bullish_engulfing", "shooting_star":"hammer"}.get(p, p)
            if "pattern" in c: c["pattern"] = new_p
            else:              c["condition"] = new_p
        elif t == "divergence":   c["direction"] = "bearish" if c.get("direction") == "bullish" else "bullish"
        elif t == "order_block":  c["ob_type"]  = "bearish" if c.get("ob_type")  == "bullish" else "bullish"
        elif t == "fvg":          c["fvg_dir"]  = "bearish" if c.get("fvg_dir")  == "bullish" else "bullish"
        elif t == "market_structure":
            c["condition"] = c.get("condition", "").replace("bullish", "bearish")
        elif t == "consecutive_candles":
            d = c.get("cc_dir", c.get("direction", "green"))
            c["cc_dir"] = "red" if d in ("green","bullish","up") else "green"
        elif t == "vwap_deviation":
            c["vwap_side"] = "above" if c.get("vwap_side", "below") == "below" else "below"
        elif t in ("ifvg", "breaker_block", "mss", "choch", "liquidity_sweep",
                   "mitigation_block", "supply_demand", "pin_bar", "engulfing",
                   "inside_bar", "equilibrium", "fib_retracement", "vwap_bounce"):
            c["direction"] = "bearish" if c.get("direction") == "bullish" else "bullish"
        elif t == "premium_discount":
            c["zone"] = "premium" if c.get("zone") == "discount" else "discount"
        # adx_filter, atr_volatility, volume_spike, keltner squeeze, hh_hl, lh_ll: direction-agnostic
        return c

    # ── Singles — diverse indicator families with parameter variants ───────────
    singles = [
        # RSI variants (different periods + thresholds)
        {"label": "RSI 7 Aggressive Oversold",   "category": "Momentum",   "primaryType": "rsi",          "primaryCfg": {"period": 7,  "operator": "lt", "value": 25},              "confirms": []},
        {"label": "RSI 14 Classic Oversold",     "category": "Momentum",   "primaryType": "rsi",          "primaryCfg": {"period": 14, "operator": "lt", "value": 30},              "confirms": []},
        {"label": "RSI 21 Swing Oversold",       "category": "Momentum",   "primaryType": "rsi",          "primaryCfg": {"period": 21, "operator": "lt", "value": 35},              "confirms": []},
        {"label": "RSI Trend Continuation >55",  "category": "Momentum",   "primaryType": "rsi",          "primaryCfg": {"period": 14, "operator": "gt", "value": 55},              "confirms": []},
        # MACD
        {"label": "MACD Bullish Cross",          "category": "Momentum",   "primaryType": "macd",         "primaryCfg": {"condition": "bullish_cross"},                             "confirms": []},
        # EMA crosses (fast/medium/long)
        {"label": "EMA Fast Cross 9/21",         "category": "Trend",      "primaryType": "ema",          "primaryCfg": {"period": 9,  "period2": 21,  "condition": "bullish_cross"}, "confirms": []},
        {"label": "EMA Medium Cross 20/50",      "category": "Trend",      "primaryType": "ema",          "primaryCfg": {"period": 20, "period2": 50,  "condition": "bullish_cross"}, "confirms": []},
        {"label": "EMA Golden Cross 50/200",     "category": "Trend",      "primaryType": "ema",          "primaryCfg": {"period": 50, "period2": 200, "condition": "bullish_cross"}, "confirms": []},
        # SMA trend filters
        {"label": "Price Above SMA 50",          "category": "Trend",      "primaryType": "sma",          "primaryCfg": {"period": 50,  "condition": "price_above"},                "confirms": []},
        {"label": "Price Above SMA 200",         "category": "Trend",      "primaryType": "sma",          "primaryCfg": {"period": 200, "condition": "price_above"},                "confirms": []},
        # SuperTrend
        {"label": "SuperTrend Bullish Flip",     "category": "Trend",      "primaryType": "supertrend",   "primaryCfg": {"condition": "bullish"},                                   "confirms": []},
        # Bollinger
        {"label": "BB Lower Bounce",             "category": "Mean Rev",   "primaryType": "bb",           "primaryCfg": {"condition": "price_below_lower"},                         "confirms": []},
        {"label": "BB Squeeze Breakout",         "category": "Volatility", "primaryType": "bb",           "primaryCfg": {"condition": "squeeze"},                                   "confirms": []},
        # StochRSI
        {"label": "StochRSI Oversold <20",       "category": "Momentum",   "primaryType": "stochrsi",     "primaryCfg": {"operator": "lt", "value": 20},                            "confirms": []},
        # Volume
        {"label": "Volume Spike 2×",             "category": "Volume",     "primaryType": "volume_spike", "primaryCfg": {"multiplier": 2.0},                                        "confirms": []},
        # Williams %R
        {"label": "Williams %R Oversold",        "category": "Momentum",   "primaryType": "williams_r",   "primaryCfg": {"condition": "oversold"},                                  "confirms": []},
        # ADX (trend filter)
        {"label": "ADX Strong Trend >25",        "category": "Trend",      "primaryType": "adx_filter",   "primaryCfg": {"condition": "trending"},                                  "confirms": []},
        # Breakouts
        {"label": "Breakout 20-bar High",        "category": "Breakout",   "primaryType": "breakout",     "primaryCfg": {"bo_lookback": 20, "bo_pct": 1.0, "bo_dir": "up"},          "confirms": []},
        {"label": "Breakout 50-bar High",        "category": "Breakout",   "primaryType": "breakout",     "primaryCfg": {"bo_lookback": 50, "bo_pct": 0.5, "bo_dir": "up"},          "confirms": []},
        # S/R bounce
        {"label": "Support Bounce",              "category": "Price Action","primaryType": "support_resistance", "primaryCfg": {"condition": "at_support"},                          "confirms": []},
        # Candlestick patterns
        {"label": "Bullish Engulfing Pattern",   "category": "Price Action","primaryType": "candlestick", "primaryCfg": {"pattern": "bullish_engulfing"},                            "confirms": []},
        # Divergence
        {"label": "RSI Bullish Divergence",      "category": "Divergence", "primaryType": "divergence",   "primaryCfg": {"indicator": "rsi", "direction": "bullish"},                "confirms": []},
        # ── ICT / Smart Money ──────────────────────────────────────────────────────
        {"label": "FVG — Fair Value Gap",        "category": "ICT",        "primaryType": "fvg",           "primaryCfg": {"fvg_dir": "bullish", "min_gap_pct": 0.15},                  "confirms": []},
        {"label": "IFVG — Inverted Fair Value Gap","category": "ICT",      "primaryType": "ifvg",          "primaryCfg": {"direction": "bullish", "min_gap_pct": 0.2},                  "confirms": []},
        {"label": "OB — Order Block",            "category": "ICT",        "primaryType": "order_block",   "primaryCfg": {"ob_type": "bullish"},                                       "confirms": []},
        {"label": "BB — Breaker Block",          "category": "ICT",        "primaryType": "breaker_block", "primaryCfg": {"direction": "bullish"},                                      "confirms": []},
        {"label": "MSS — Market Structure Shift","category": "ICT",        "primaryType": "mss",           "primaryCfg": {"direction": "bullish"},                                      "confirms": []},
        {"label": "CHoCH — Change of Character", "category": "ICT",        "primaryType": "choch",         "primaryCfg": {"direction": "bullish"},                                      "confirms": []},
        {"label": "LQ — Liquidity Sweep",        "category": "ICT",        "primaryType": "liquidity_sweep","primaryCfg": {"direction": "bullish"},                                     "confirms": []},
        {"label": "MIT — Mitigation Block",      "category": "ICT",        "primaryType": "mitigation_block","primaryCfg": {"direction": "bullish"},                                    "confirms": []},
        # ── Supply & Demand ────────────────────────────────────────────────────────
        {"label": "SDP — Supply/Demand Zone",    "category": "Supply/Demand","primaryType": "supply_demand","primaryCfg": {"direction": "bullish"},                                     "confirms": []},
        {"label": "PD — Premium/Discount",       "category": "Supply/Demand","primaryType": "premium_discount","primaryCfg": {"zone": "discount"},                                     "confirms": []},
        {"label": "EQ — Equilibrium Entry",      "category": "Supply/Demand","primaryType": "equilibrium",  "primaryCfg": {"direction": "bullish"},                                     "confirms": []},
        # ── Price Action (forex) ───────────────────────────────────────────────────
        {"label": "PIN — Pin Bar",               "category": "Price Action","primaryType": "pin_bar",       "primaryCfg": {"direction": "bullish"},                                     "confirms": []},
        {"label": "ENG — Engulfing Candle",      "category": "Price Action","primaryType": "engulfing",     "primaryCfg": {"direction": "bullish"},                                     "confirms": []},
        {"label": "IB — Inside Bar Breakout",    "category": "Price Action","primaryType": "inside_bar",    "primaryCfg": {"direction": "bullish"},                                     "confirms": []},
        # ── Structure ──────────────────────────────────────────────────────────────
        {"label": "HH/HL — Bullish Structure",   "category": "Structure",  "primaryType": "hh_hl",         "primaryCfg": {},                                                           "confirms": []},
        {"label": "LH/LL — Bearish Structure",   "category": "Structure",  "primaryType": "lh_ll",         "primaryCfg": {},                                                           "confirms": []},
        {"label": "FIB — Fibonacci Retracement", "category": "Structure",  "primaryType": "fib_retracement","primaryCfg": {"direction": "bullish"},                                    "confirms": []},
        {"label": "VWAP — Volume Weighted Average","category": "Structure", "primaryType": "vwap_bounce",   "primaryCfg": {"direction": "bullish"},                                     "confirms": []},
    ]

    # ── Combos — primary + one or two confirmation layers ──────────────────────
    combos = [
        # Oscillator + trend confirm
        {"label": "RSI Oversold + MACD Confirm",          "category": "Combo",      "primaryType": "rsi",        "primaryCfg": {"period": 14, "operator": "lt", "value": 30},              "confirms": [{"type": "macd",         "condition": "bullish_cross"}]},
        {"label": "RSI Oversold + SuperTrend Confirm",    "category": "Combo",      "primaryType": "rsi",        "primaryCfg": {"period": 14, "operator": "lt", "value": 30},              "confirms": [{"type": "supertrend",   "condition": "bullish"}]},
        # EMA cross + volume/momentum confirm
        {"label": "EMA Cross + Volume Spike",             "category": "Combo",      "primaryType": "ema",        "primaryCfg": {"period": 9, "period2": 21, "condition": "bullish_cross"}, "confirms": [{"type": "volume_spike", "multiplier": 1.5}]},
        {"label": "EMA Cross + RSI > 50 Strength",        "category": "Combo",      "primaryType": "ema",        "primaryCfg": {"period": 9, "period2": 21, "condition": "bullish_cross"}, "confirms": [{"type": "rsi",          "period": 14, "operator": "gt", "value": 50}]},
        # MACD + trend filter
        {"label": "MACD Cross + EMA20 Bullish",           "category": "Combo",      "primaryType": "macd",       "primaryCfg": {"condition": "bullish_cross"},                             "confirms": [{"type": "ema",          "period": 20, "condition": "above"}]},
        {"label": "MACD Cross + Above SMA 200",           "category": "Combo",      "primaryType": "macd",       "primaryCfg": {"condition": "bullish_cross"},                             "confirms": [{"type": "sma",          "period": 200, "condition": "price_above"}]},
        # SuperTrend + oscillator
        {"label": "SuperTrend + MACD Confirm",            "category": "Combo",      "primaryType": "supertrend", "primaryCfg": {"condition": "bullish"},                                   "confirms": [{"type": "macd",         "condition": "bullish_cross"}]},
        {"label": "SuperTrend + RSI > 50",                "category": "Combo",      "primaryType": "supertrend", "primaryCfg": {"condition": "bullish"},                                   "confirms": [{"type": "rsi",          "period": 14, "operator": "gt", "value": 50}]},
        # BB + RSI
        {"label": "BB Bounce + RSI Oversold Confirm",     "category": "Combo",      "primaryType": "bb",         "primaryCfg": {"condition": "price_below_lower"},                         "confirms": [{"type": "rsi",          "period": 14, "operator": "lt", "value": 35}]},
        # StochRSI + MACD
        {"label": "StochRSI + MACD Confirm",              "category": "Combo",      "primaryType": "stochrsi",   "primaryCfg": {"operator": "lt", "value": 20},                            "confirms": [{"type": "macd",         "condition": "bullish_cross"}]},
        # Breakout + volume/trend
        {"label": "Breakout + Volume Spike (Classic)",    "category": "Combo",      "primaryType": "breakout",   "primaryCfg": {"bo_lookback": 20, "bo_pct": 1.0, "bo_dir": "up"},          "confirms": [{"type": "volume_spike", "multiplier": 1.8}]},
        {"label": "Breakout + ADX Trending",              "category": "Combo",      "primaryType": "breakout",   "primaryCfg": {"bo_lookback": 20, "bo_pct": 1.0, "bo_dir": "up"},          "confirms": [{"type": "adx_filter",   "condition": "trending"}]},
        # ADX trend + EMA momentum
        {"label": "ADX Trending + EMA Cross",             "category": "Combo",      "primaryType": "adx_filter", "primaryCfg": {"condition": "trending"},                                  "confirms": [{"type": "ema",          "period": 9, "period2": 21, "condition": "bullish_cross"}]},
        # Reversal pattern + oscillator
        {"label": "RSI Oversold + Bullish Engulfing",     "category": "Combo",      "primaryType": "rsi",        "primaryCfg": {"period": 14, "operator": "lt", "value": 35},              "confirms": [{"type": "candlestick",  "pattern": "bullish_engulfing"}]},
        # Pullback in uptrend
        {"label": "Above SMA 200 + RSI Oversold Pullback","category": "Combo",      "primaryType": "sma",        "primaryCfg": {"period": 200, "condition": "price_above"},                "confirms": [{"type": "rsi",          "period": 14, "operator": "lt", "value": 35}]},
        # Williams %R + MACD oscillator combo
        {"label": "Williams %R + MACD Confirm",           "category": "Combo",      "primaryType": "williams_r", "primaryCfg": {"condition": "oversold"},                                  "confirms": [{"type": "macd",         "condition": "bullish_cross"}]},
        # Divergence + MACD confirmed
        {"label": "RSI Divergence + MACD Cross",          "category": "Combo",      "primaryType": "divergence", "primaryCfg": {"indicator": "rsi", "direction": "bullish"},                "confirms": [{"type": "macd",         "condition": "bullish_cross"}]},
        # BB squeeze breakout + volume
        {"label": "BB Squeeze + Volume Spike",            "category": "Combo",      "primaryType": "bb",         "primaryCfg": {"condition": "squeeze"},                                   "confirms": [{"type": "volume_spike", "multiplier": 1.5}]},
        # ── ICT / Forex combos ─────────────────────────────────────────────────────
        {"label": "MSS + FVG Retest",                     "category": "ICT Combo",  "primaryType": "mss",        "primaryCfg": {"direction": "bullish"},                                   "confirms": [{"type": "fvg", "fvg_dir": "bullish", "min_gap_pct": 0.1}]},
        {"label": "CHoCH + Order Block Retest",           "category": "ICT Combo",  "primaryType": "choch",      "primaryCfg": {"direction": "bullish"},                                   "confirms": [{"type": "order_block", "ob_type": "bullish"}]},
        {"label": "Liquidity Sweep + Pin Bar",            "category": "ICT Combo",  "primaryType": "liquidity_sweep","primaryCfg": {"direction": "bullish"},                              "confirms": [{"type": "pin_bar", "direction": "bullish"}]},
        {"label": "FVG + Premium/Discount Zone",          "category": "ICT Combo",  "primaryType": "fvg",        "primaryCfg": {"fvg_dir": "bullish", "min_gap_pct": 0.1},                 "confirms": [{"type": "premium_discount", "zone": "discount"}]},
        {"label": "Engulfing + Fibonacci Zone",           "category": "ICT Combo",  "primaryType": "engulfing",  "primaryCfg": {"direction": "bullish"},                                   "confirms": [{"type": "fib_retracement", "direction": "bullish"}]},
        {"label": "Supply/Demand + RSI Oversold",         "category": "ICT Combo",  "primaryType": "supply_demand","primaryCfg": {"direction": "bullish"},                                "confirms": [{"type": "rsi", "period": 14, "operator": "lt", "value": 35}]},
        {"label": "Breaker Block + MACD Confirm",         "category": "ICT Combo",  "primaryType": "breaker_block","primaryCfg": {"direction": "bullish"},                                "confirms": [{"type": "macd", "condition": "bullish_cross"}]},
        {"label": "VWAP Bounce + EMA Trend",              "category": "ICT Combo",  "primaryType": "vwap_bounce","primaryCfg": {"direction": "bullish"},                                   "confirms": [{"type": "ema", "period": 20, "condition": "above"}]},
    ]

    # ── Signal category filtering ─────────────────────────────────────────────
    # The frontend can optionally pass `signal_categories` to limit which signal
    # families are scanned. Map frontend category ids to backend category names.
    _sig_cats_raw = body.get("signal_categories")
    _CAT_MAP = {
        "classic": {"Momentum", "Trend", "Mean Rev", "Volatility", "Volume", "Breakout", "Divergence"},
        "ict":     {"ICT"},
        "sd":      {"Supply/Demand"},
        "pa":      {"Price Action"},
        "structure": {"Structure"},
    }
    _allowed_cats = None
    if isinstance(_sig_cats_raw, list) and _sig_cats_raw:
        _allowed_cats = set()
        for sc in _sig_cats_raw:
            _allowed_cats.update(_CAT_MAP.get(sc, set()))
        _allowed_cats.update({"Combo", "ICT Combo"})

    templates = []
    for base_list in (singles, combos):
        for t in base_list:
            if _allowed_cats and t.get("category", "Other") not in _allowed_cats:
                continue
            entry = {
                "label":       t["label"],
                "category":    t.get("category", "Other"),
                "primaryType": t["primaryType"],
                "primaryCfg":  dict(t["primaryCfg"]),
                "confirms":    [dict(c) for c in t.get("confirms", [])],
                "timeframe":   t.get("timeframe", "1h"),
                "tp1": _tp, "sl": _sl, "leverage": _lev,
            }
            if direction == "SHORT":
                # Flip primary by wrapping it through _C()
                primary_as_cond = {"type": entry["primaryType"], **entry["primaryCfg"]}
                flipped_primary = _C(primary_as_cond)
                flipped_primary.pop("type", None)
                entry["primaryCfg"] = flipped_primary
                # Flip each confirm
                entry["confirms"] = [_C(c) for c in entry["confirms"]]
                # Cosmetic label adjustments
                entry["label"] = (entry["label"]
                                  .replace("Oversold", "Overbought")
                                  .replace("Bullish", "Bearish")
                                  .replace("Bounce", "Rejection")
                                  .replace("Above SMA", "Below SMA")
                                  .replace("Price Above", "Price Below")
                                  .replace("Golden Cross", "Death Cross")
                                  .replace("Engulfing", "Engulfing")
                                  .replace("Breakout", "Breakdown")
                                  .replace("High", "Low")
                                  .replace("Support", "Resistance"))
            templates.append(entry)

    from app.services.backtest_engine import run_backtest, _fetch_historical
    import httpx

    # ── Pre-fetch candles for every (coin, timeframe) combo in parallel ────────
    # The engine accepts precomputed_candles, so we fetch each (coin,TF) once
    # and reuse the candle array across all 40 strategies × 4 risk profiles.
    # This converts ~600 redundant HTTP fetches into ~15 fetches.
    _candle_cache: dict = {}   # (symbol, tf) → (candles, source_label)
    _missing_pairs: list = []  # (ticker, tf) we couldn't fetch candles for

    async def _prefetch_one(symbol: str, ticker: str, tf: str, client: httpx.AsyncClient):
        try:
            candles, label, _ = await _fetch_historical(symbol, days, client, tf, asset_class=asset_class)
            if len(candles) >= 60:
                _candle_cache[(symbol, tf)] = (candles, label)
            else:
                _missing_pairs.append((ticker, tf))
        except Exception as exc:
            logger.warning(f"[Scanner] prefetch failed {ticker} {tf}: {exc}")
            _missing_pairs.append((ticker, tf))

    async with httpx.AsyncClient(timeout=20) as _client:
        await asyncio.gather(*[
            _prefetch_one(sym, tk, tf, _client)
            for sym, tk in zip(symbols, tickers)
            for tf in tf_list
        ])

    if not _candle_cache:
        return {"ok": False, "error": "NO_DATA",
                "message": "Couldn't fetch candle data for any of the requested coins/timeframes."}

    # ── Risk profiles tested on the top strategies in stage 2 ──────────────────
    # Each strategy is first scored at the "Balanced" profile (stage 1), then
    # the top 8 are re-tested with all 4 profiles in stage 2 to find the
    # optimal TP/SL for that specific (strategy, coin, TF) combo.
    if risk_mode == "fixed":
        # User pinned an exact TP/SL — single profile, stage 2 becomes a no-op
        # because the optimisation loop iterates over `risk_profiles` and skips
        # DEFAULT_PROFILE, leaving zero variants to test.
        _rr_ratio = round(fixed_tp / max(fixed_sl, 0.01), 2)
        risk_profiles = [{
            "name":     f"Fixed {fixed_tp:g}% / {fixed_sl:g}%",
            "tp1":      fixed_tp,
            "sl":       fixed_sl,
            "leverage": 5,
            "rr":       f"{_rr_ratio}:1",
        }]
        DEFAULT_PROFILE = risk_profiles[0]
    else:
        # Optimised mode — historical 4-profile search picks the best variant
        # per (strategy, coin, TF) combo (stage 2 below).
        risk_profiles = [
            {"name": "Tight Scalp", "tp1": 1.5, "sl": 0.75, "leverage": 5, "rr": "2:1"},
            {"name": "Balanced",    "tp1": 3.0, "sl": 1.5,  "leverage": 5, "rr": "2:1"},
            {"name": "Wide Swing",  "tp1": 5.0, "sl": 2.0,  "leverage": 5, "rr": "2.5:1"},
            {"name": "Runner",      "tp1": 8.0, "sl": 2.0,  "leverage": 5, "rr": "4:1"},
        ]
        DEFAULT_PROFILE = risk_profiles[1]  # Balanced

    # Bound concurrency so we don't blow up CPU when running 600+ backtests.
    # Backtests are synchronous CPU work; async only helps with I/O. We pre-
    # fetched all candles, so this is mostly compute. A small semaphore lets
    # the event loop interleave but keeps memory bounded.
    _bt_sem = asyncio.Semaphore(20)

    async def _run_one(tpl: dict, ticker: str, symbol: str, tf: str,
                       risk: dict | None = None) -> dict:
        r = risk or DEFAULT_PROFILE
        cached = _candle_cache.get((symbol, tf))
        if cached is None:
            return {
                "label": tpl["label"], "category": tpl.get("category", "Other"),
                "tpl": tpl, "risk": r, "coin": ticker, "timeframe": tf,
                "config": {}, "result": {"error": "no candles"},
            }
        candles, label = cached
        cfg = {
            "direction":   direction,
            "primaryType": tpl["primaryType"],
            "primaryCfg":  tpl["primaryCfg"],
            "confirms":    tpl.get("confirms", []),
            "tp1":         r["tp1"],
            "sl":          r["sl"],
            "leverage":    r["leverage"],
            "timeframe":   tf,
            "singleCoin":  symbol,
            "asset_class": asset_class,
        }
        async with _bt_sem:
            try:
                result = await asyncio.wait_for(
                    run_backtest(cfg, days,
                                 precomputed_candles=candles,
                                 precomputed_source_label=label),
                    timeout=30,
                )
            except asyncio.TimeoutError:
                result = {"error": "timeout"}
            except Exception as exc:
                result = {"error": str(exc)}
        return {
            "label":    tpl["label"],
            "category": tpl.get("category", "Other"),
            "tpl":      tpl,
            "risk":     r,
            "coin":     ticker,
            "timeframe": tf,
            "config":   cfg,
            "result":   result,
        }

    # ── Stage 1: Run all templates × coins × TFs at the Balanced profile ──────
    stage1_jobs = []
    for sym, tk in zip(symbols, tickers):
        for tf in tf_list:
            if (sym, tf) not in _candle_cache:
                continue
            for tpl in templates:
                stage1_jobs.append(_run_one(tpl, tk, sym, tf, DEFAULT_PROFILE))

    try:
        outcomes = await asyncio.wait_for(
            asyncio.gather(*stage1_jobs),
            timeout=120,
        )
    except asyncio.TimeoutError:
        return JSONResponse(
            status_code=408,
            content={"error": "TIMEOUT",
                     "message": "Scan timed out — try fewer coins, fewer timeframes, or a 30-day window."},
        )
    except Exception as e:
        logger.exception("backtest scan stage1 failed")
        return JSONResponse(
            status_code=500,
            content={"error": "BACKTEST_FAILED",
                     "message": str(e) or "Scan failed unexpectedly. Please try again."},
        )

    # ── Score and rank ─────────────────────────────────────────────────────────
    # Stricter trade-count floor prevents sparse-sample strategies (e.g. 1 lucky
    # trade with PF=99) from dominating the ranking. Long windows scale up the
    # floor; SLOW timeframes (1h, 4h) scale it DOWN so a clean 4h trend strategy
    # making 4 quality trades over 30d isn't unfairly discarded vs noisy 15m
    # strategies. Floor is computed per (TF, days) below in _min_trades_for_tf().
    def _min_trades_for_tf(tf: str, n_days: int) -> int:
        # Roughly proportional to the number of candles available in the window.
        # A strategy that fires once every ~12 hours will hit these floors:
        #   15m × 30d ≈ 8 trades · 15m × 90d ≈ 14 trades
        #   1h  × 30d ≈ 6 trades · 1h  × 90d ≈ 10 trades
        #   4h  × 30d ≈ 4 trades · 4h  × 90d ≈ 7 trades
        base = {"5m": 10, "15m": 8, "1h": 6, "4h": 4}.get(tf, 6)
        if n_days >= 90:
            base = int(round(base * 1.7))
        return base

    def _score(stats: dict, tf: str = "1h") -> float:
        trades = stats.get("closed_trades", 0)
        if trades < _min_trades_for_tf(tf, days):
            return -1.0   # too few trades to trust the result
        pf = float(stats.get("profit_factor", 0) or 0)
        wr = float(stats.get("win_rate", 0) or 0) / 100
        pnl = float(stats.get("total_pnl", 0) or 0)
        # Composite: profit factor × win rate × log(trades) so strategies with
        # MORE trades at the same quality outrank sparse ones; tie-break by P&L.
        import math
        sample_weight = math.log(trades + 1) / math.log(20)  # ~1.0 at 19 trades
        return round(pf * wr * sample_weight + pnl * 0.001, 4)

    def _forex_score(stats: dict) -> int:
        """Forex-specific composite score out of 100.
        Win rate (30%) + Avg pips per winning trade (40%) + Consistency (30%)."""
        wr = float(stats.get("win_rate", 0) or 0) / 100
        avg_pips = float(stats.get("avg_pips_per_trade", 0) or 0)
        consistency = float(stats.get("consistency", 50) or 50) / 100
        wr_score = min(wr * 100, 100) * 0.30
        pips_score = min(max(avg_pips, 0) / 50 * 100, 100) * 0.40
        cons_score = min(consistency * 100, 100) * 0.30
        return max(0, min(100, int(round(wr_score + pips_score + cons_score))))

    def _compute_forex_pips(stats: dict, trades: list, ac: str) -> dict:
        """Add forex-specific pips metrics to stats for forex asset class."""
        if ac != "forex" or not trades:
            return stats
        from app.services.forex_engine import pip_size as _pip_size
        try:
            psize = _pip_size(symbols[0] if symbols else "XAUUSD")
        except Exception:
            psize = 0.01
        total_pips = 0.0
        win_pips = []
        all_pips = []
        for t in trades:
            entry_p = float(t.get("entry_price", 0) or 0)
            exit_p = float(t.get("exit_price", 0) or 0)
            direction_t = t.get("direction", "LONG")
            if psize > 0 and entry_p > 0:
                if direction_t == "LONG":
                    pips = (exit_p - entry_p) / psize
                else:
                    pips = (entry_p - exit_p) / psize
                total_pips += pips
                all_pips.append(pips)
                if pips > 0:
                    win_pips.append(pips)
        stats = dict(stats)
        stats["total_pips"] = round(total_pips, 1)
        stats["avg_pips_per_trade"] = round(total_pips / len(trades), 1) if trades else 0
        stats["avg_pips_per_win"] = round(sum(win_pips) / len(win_pips), 1) if win_pips else 0
        if len(all_pips) > 1:
            import statistics
            std = statistics.stdev(all_pips)
            mean = abs(statistics.mean(all_pips)) if statistics.mean(all_pips) != 0 else 1
            stats["consistency"] = max(0, min(100, int(round(100 - (std / mean * 50))))) if mean > 0 else 50
        else:
            stats["consistency"] = 50
        stats["forex_score"] = _forex_score(stats)
        return stats

    def _outcome_to_row(o: dict) -> dict:
        res   = o.get("result") or {}
        err   = res.get("error")
        stats = res.get("stats") or {}
        if not err and asset_class == "forex":
            stats = _compute_forex_pips(stats, res.get("trades", []), asset_class)
        score = _score(stats, o.get("timeframe", "1h")) if not err else -1.0
        return {
            "label":     o["label"],
            "category":  o.get("category", "Other"),
            "tpl":       o["tpl"],
            "risk":      o["risk"],
            "coin":      o["coin"],
            "timeframe": o["timeframe"],
            "config":    o["config"],
            "stats":     stats,
            "trades":    res.get("trades", []),
            "equity_curve": res.get("equity_curve", []),
            "score":     score,
            "error":     err,
        }

    ranked_s1 = sorted([_outcome_to_row(o) for o in outcomes],
                       key=lambda x: x["score"], reverse=True)
    valid_s1   = [r for r in ranked_s1 if r["score"] >= 0]
    invalid    = [r for r in ranked_s1 if r["score"] < 0]

    # ── Stage 2: Optimise TP/SL on the top 8 (strategy, coin, TF) combos ─────
    # For each surviving combo, re-test it with the OTHER 3 risk profiles
    # (Balanced is already done in stage 1) and keep the highest-scoring
    # variant. The combo's coin + TF stay locked — only TP/SL/leverage change.
    top_for_optimisation = valid_s1[:8]
    optimisation_jobs = []
    # Use a stable key per combo so we can group variants back together.
    # Includes `category` defensively — labels are expected to be unique
    # within `templates`, but the extra field guarantees no future collision
    # between two strategies that happen to share a name.
    def _combo_key(row: dict) -> tuple:
        return (row["label"], row.get("category", ""), row["coin"], row["timeframe"])

    for row in top_for_optimisation:
        for profile in risk_profiles:
            if profile["name"] == DEFAULT_PROFILE["name"]:
                continue
            symbol_for_row = row["config"].get("singleCoin")
            optimisation_jobs.append((
                _combo_key(row),
                _run_one(row["tpl"], row["coin"], symbol_for_row, row["timeframe"], profile),
            ))

    try:
        opt_results = await asyncio.wait_for(
            asyncio.gather(*[job for _, job in optimisation_jobs]),
            timeout=90,
        )
    except asyncio.TimeoutError:
        # Stage 2 is optimisation — if it times out we still have valid stage 1
        # results, so degrade gracefully instead of failing the whole scan.
        logger.warning("backtest scan stage2 (TP/SL optimisation) timed out — returning stage1 results")
        opt_results = []
    except Exception as e:
        logger.exception("backtest scan stage2 failed")
        opt_results = []

    # Group all variants per (label, coin, TF) combo, then pick the best
    by_combo: dict = {}
    for row in top_for_optimisation:
        by_combo.setdefault(_combo_key(row), []).append(row)
    for (key, _job), o in zip(optimisation_jobs, opt_results):
        by_combo.setdefault(key, []).append(_outcome_to_row(o))

    # For each combo, pick the variant with the best score
    valid = []
    for row in top_for_optimisation:
        variants = by_combo[_combo_key(row)]
        best = max(variants, key=lambda v: v["score"])
        # Attach all variant stats so the UI can show the comparison
        best["all_risk_variants"] = [
            {
                "risk_name": v["risk"]["name"],
                "tp1":       v["risk"]["tp1"],
                "sl":        v["risk"]["sl"],
                "rr":        v["risk"]["rr"],
                "win_rate":  v["stats"].get("win_rate", 0),
                "total_pnl": v["stats"].get("total_pnl", 0),
                "profit_factor": v["stats"].get("profit_factor", 0),
                "closed_trades": v["stats"].get("closed_trades", 0),
                "score":     v["score"],
                "is_best":   v["score"] == best["score"],
            }
            for v in sorted(variants, key=lambda v: v["risk"]["tp1"])
        ]
        valid.append(best)

    # Re-sort by the best-variant score so the leaderboard reflects optimisation
    valid.sort(key=lambda x: x["score"], reverse=True)

    # ── AI one-liner for the winner ────────────────────────────────────────────
    winner_insight = ""
    if valid:
        w  = valid[0]
        ws = w["stats"]
        wr = w["risk"]
        try:
            import anthropic as _ant
            _ac = _ant.AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
            _msg = await asyncio.wait_for(
                _ac.messages.create(
                    model="claude-sonnet-4-5",
                    max_tokens=140,
                    system="You are a crypto quant analyst. Respond with exactly 1 short sentence (max 25 words) explaining why this strategy + coin + timeframe + risk combo wins. No markdown.",
                    messages=[{"role": "user", "content":
                        f"Scan winner: '{w['label']}' on {w['coin']} {w['timeframe']} with '{wr['name']}' risk "
                        f"(TP {wr['tp1']}% / SL {wr['sl']}%, R:R {wr['rr']}) over {days} days: "
                        f"win rate {ws.get('win_rate')}%, P&L {ws.get('total_pnl')}%, "
                        f"profit factor {ws.get('profit_factor')}, {ws.get('closed_trades')} trades. "
                        f"Why does this combo work?"}],
                ),
                timeout=12,
            )
            winner_insight = _msg.content[0].text.strip()
        except Exception:
            pass

    # ── Strip non-serialisable bits + cap trade lists for response size ───────
    # We send full equity curves + trades for the top 5 winners (so the UI
    # can chart them). For ranks 6-12 we drop those to keep the payload small.
    def _clean_row(r: dict, include_full: bool = False) -> dict:
        out = {
            "label":            r["label"],
            "category":         r["category"],
            "coin":             r["coin"],
            "timeframe":        r["timeframe"],
            "config":           r["config"],
            "stats":            r["stats"],
            "score":            r["score"],
            "risk":             r["risk"],
            "all_risk_variants": r.get("all_risk_variants", []),
        }
        if include_full:
            # Cap trades at 200 to bound payload (most scans produce 10-50)
            out["trades"]       = (r.get("trades") or [])[:200]
            out["equity_curve"] = r.get("equity_curve") or []
        return out

    return {
        "ok":      True,
        "coin":    primary_ticker,           # legacy single-coin field
        "coins":   tickers,                  # full coin universe scanned
        "timeframes": tf_list,               # all TFs scanned
        "days":    days,
        "direction": direction,
        "risk_mode": risk_mode,
        "fixed_tp":  fixed_tp if risk_mode == "fixed" else None,
        "fixed_sl":  fixed_sl if risk_mode == "fixed" else None,
        "ranked":  [_clean_row(r, include_full=(i < 5)) for i, r in enumerate(valid[:12])],
        "tested":  len(stage1_jobs),
        "skipped": len(invalid),
        "optimisation_jobs": len(optimisation_jobs),
        "missing_pairs": [{"coin": c, "timeframe": tf} for c, tf in _missing_pairs],
        "winner_insight": winner_insight,
    }


async def _discovery_auth(uid: str) -> tuple:
    """Single DB round-trip: (status, user_id) where status is ok|not_pro|bad_uid."""
    from app.database import SessionLocal
    import asyncio as _asyncio
    for _attempt in range(3):
        try:
            db = SessionLocal()
            try:
                user = _get_user_by_uid_safe(uid, db)
                if not user:
                    return "bad_uid", None
                sub = _get_portal_sub(user.id, db)
                is_pro = _is_portal_pro(sub) or bool(getattr(user, "is_admin", False))
                return ("ok" if is_pro else "not_pro"), int(user.id)
            finally:
                db.close()
        except HTTPException:
            await _asyncio.sleep(0.25)
        except Exception:
            await _asyncio.sleep(0.25)
    return "bad_uid", None


def _discovery_scan_options(body: dict) -> tuple:
    """Parse quality_cfg + signal categories from discovery POST body."""
    from app.services.setup_quality import normalize_quality_cfg

    quality_cfg = dict(body.get("quality_cfg") or {})
    if body.get("strict_quality") is not None:
        quality_cfg["quality_mode"] = "strict" if body.get("strict_quality", True) else "all"
    quality_cfg = normalize_quality_cfg(quality_cfg)

    categories = body.get("categories") or body.get("signal_categories")
    if isinstance(categories, list):
        categories = [str(c).strip() for c in categories if str(c).strip()]
    else:
        categories = None
    return quality_cfg, categories


def _discovery_progress_response(scan_type: str, uid: str) -> dict:
    from app.services.discovery_jobs import job_progress
    prog = job_progress(scan_type, uid.strip())
    # Back-compat: legacy clients read `message` only
    if "message" not in prog:
        prog["message"] = ""
    return prog


@app.get("/api/backtest/gold-discovery/progress")
async def gold_discovery_progress(uid: str = Query(...)):
    """Poll gold discovery job — status idle|running|done|error + result when done."""
    return _discovery_progress_response("gold", uid)


@app.post("/api/backtest/gold-discovery")
async def backtest_gold_discovery(request: Request):
    """
    Start Claude-driven GOLD (XAUUSD) discovery in the background.
    Poll GET /api/backtest/gold-discovery/progress until status=done.
    Body: { uid, days, direction, strict_quality?, quality_cfg?, categories? }
    """
    body = await request.json()
    uid  = (body.get("uid") or "").strip()
    days = int(body.get("days", 90))
    direction_mode = (body.get("direction") or "BOTH").upper()
    quality_cfg, categories = _discovery_scan_options(body)

    auth, user_id = await _discovery_auth(uid)
    if auth == "bad_uid":
        raise HTTPException(status_code=403, detail="Invalid UID")
    if auth == "not_pro":
        return JSONResponse(status_code=403, content={
            "ok": False,
            "message": "A Pro subscription is required to use the Gold Strategy Finder."})

    from app.services.discovery_jobs import start_discovery_job

    async def _runner(progress_cb):
        from app.services.gold_strategy_scanner import run_gold_discovery
        return await run_gold_discovery(
            days=days, direction_mode=direction_mode,
            user_id=user_id, progress_cb=progress_cb,
            quality_cfg=quality_cfg, categories=categories,
        )

    return JSONResponse(content=start_discovery_job("gold", uid, _runner))


@app.get("/api/backtest/index-discovery/progress")
async def index_discovery_progress(uid: str = Query(...)):
    """Poll index discovery job — status idle|running|done|error + result when done."""
    return _discovery_progress_response("index", uid)


@app.post("/api/backtest/index-discovery")
async def backtest_index_discovery(request: Request):
    """
    Start index CFD discovery in the background.
    Body: { uid, symbol, days, direction }
    """
    body = await request.json()
    uid  = (body.get("uid") or "").strip()
    days = int(body.get("days", 90))
    direction_mode = (body.get("direction") or "BOTH").upper()
    symbol = (body.get("symbol") or "NAS100").strip()
    quality_cfg, categories = _discovery_scan_options(body)

    auth, user_id = await _discovery_auth(uid)
    if auth == "bad_uid":
        raise HTTPException(status_code=403, detail="Invalid UID")
    if auth == "not_pro":
        return JSONResponse(status_code=403, content={
            "ok": False,
            "message": "A Pro subscription is required to use the Index Strategy Finder."})

    from app.services.discovery_jobs import start_discovery_job

    async def _runner(progress_cb):
        from app.services.index_strategy_scanner import run_index_discovery
        return await run_index_discovery(
            symbol=symbol, days=days, direction_mode=direction_mode,
            user_id=user_id, progress_cb=progress_cb,
            quality_cfg=quality_cfg, categories=categories,
        )

    return JSONResponse(content=start_discovery_job("index", uid, _runner))


@app.get("/api/backtest/forex-discovery/progress")
async def forex_discovery_progress(uid: str = Query(...)):
    """Poll forex pair discovery job."""
    return _discovery_progress_response("forex", uid)


@app.post("/api/backtest/forex-discovery")
async def backtest_forex_discovery(request: Request):
    """
    Start major FX pair discovery in the background.
    Body: { uid, symbol, days, direction }
    """
    body = await request.json()
    uid  = (body.get("uid") or "").strip()
    days = int(body.get("days", 90))
    direction_mode = (body.get("direction") or "BOTH").upper()
    symbol = (body.get("symbol") or "EURUSD").strip()
    quality_cfg, categories = _discovery_scan_options(body)

    auth, user_id = await _discovery_auth(uid)
    if auth == "bad_uid":
        raise HTTPException(status_code=403, detail="Invalid UID")
    if auth == "not_pro":
        return JSONResponse(status_code=403, content={
            "ok": False,
            "message": "A Pro subscription is required to use the Forex Strategy Finder."})

    from app.services.discovery_jobs import start_discovery_job

    async def _runner(progress_cb):
        from app.services.forex_pair_strategy_scanner import run_forex_discovery
        return await run_forex_discovery(
            symbol=symbol, days=days, direction_mode=direction_mode,
            user_id=user_id, progress_cb=progress_cb,
            quality_cfg=quality_cfg, categories=categories,
        )

    return JSONResponse(content=start_discovery_job("forex", uid, _runner))


@app.post("/api/backtest/run")
async def run_backtest_endpoint(request: Request):
    """
    Run a strategy backtest against historical OHLCV data.
    Pro subscribers only.
    Body: { uid, config: <wizard WZ state>, days: 30|90 }
    """
    body = await request.json()
    uid  = (body.get("uid") or "").strip()

    from app.database import SessionLocal
    def _auth_backtest_user():
        db = SessionLocal()
        try:
            user = _get_user_by_uid_safe(uid, db)
            if not user:
                return "invalid"
            _sub = _get_portal_sub(user.id, db)
            if not _is_portal_pro(_sub) and not getattr(user, "is_admin", False):
                return "pro_required"
            return "ok"
        finally:
            db.close()

    auth_status = await asyncio.to_thread(_auth_backtest_user)
    if auth_status == "invalid":
        raise HTTPException(status_code=403, detail="Invalid UID")
    if auth_status == "pro_required":
        return JSONResponse(
            status_code=402,
            content={"error": "PRO_REQUIRED",
                     "message": "A Pro subscription ($50/month) is required to run backtests."},
        )

    config = body.get("config") or {}
    days   = int(body.get("days", 30))
    if days not in (30, 90):
        days = 30

    try:
        config_fingerprint = hashlib.sha256(
            json.dumps(config, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest()[:24]
    except Exception:
        config_fingerprint = hashlib.sha256(str(config).encode("utf-8")).hexdigest()[:24]
    cache_key = f"backtest_run:{uid}:{days}:{config_fingerprint}"
    cached_result = get_cache(cache_key)
    if cached_result is not None:
        return _cached_json(cached_result, True, 600)

    try:
        from app.services.backtest_engine import run_backtest
        # Engine has its own ~60s budget; we wrap in 90s as a hard ceiling so
        # the request can never outlive the worker timeout.
        result = await asyncio.wait_for(run_backtest(config, days), timeout=90)
        set_cache(cache_key, result, 600)
        return _cached_json(result, False, 600)
    except asyncio.TimeoutError:
        # Use 408 so mobile clients can detect the timeout at the HTTP layer
        # instead of parsing a 200-with-error-key blob.
        return JSONResponse(
            status_code=408,
            content={"error": "TIMEOUT",
                     "message": "Backtest timed out. Try a shorter window (30 days) or a faster timeframe."},
        )
    except Exception as exc:
        logger.error(f"[Backtest] error uid={uid}: {exc}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"error": "BACKTEST_FAILED",
                     "message": f"Backtest failed: {exc}"},
        )


@app.post("/api/backtest/suggest")
async def backtest_suggest(request: Request):
    """
    Analyze backtest results with Claude and return up to 3 concrete
    parameter changes that are likely to improve ROI.
    Body: { uid, config, stats, days }
    """
    body = await request.json()
    uid  = (body.get("uid") or "").strip()

    from app.database import SessionLocal
    db = SessionLocal()
    try:
        user = _get_user_by_uid(uid, db)
        if not user:
            raise HTTPException(status_code=403, detail="Invalid UID")
    finally:
        db.close()

    config       = body.get("config") or {}
    stats        = body.get("stats")  or {}
    days         = int(body.get("days", 30))
    raw_trades   = body.get("trades") or []
    equity_curve = body.get("equity_curve") or []

    direction    = config.get("direction", "LONG")
    primary_type = config.get("primaryType", "rsi")
    primary_cfg  = config.get("primaryCfg") or {}
    confirms     = config.get("confirms") or []
    tp1          = config.get("tp1", 3)
    sl           = config.get("sl", 2)
    leverage     = config.get("leverage", 5)
    coin         = config.get("singleCoin", "BTCUSDT").replace("USDT", "")

    win_rate      = stats.get("win_rate", 0)
    total_pnl     = stats.get("total_pnl", 0)
    profit_factor = stats.get("profit_factor", 0)
    max_drawdown  = stats.get("max_drawdown", 0)
    closed_trades = stats.get("closed_trades", 0)
    avg_hold_mins = stats.get("avg_hold_minutes", 0)
    trades_per_day = round(closed_trades / max(days, 1), 2)

    import json as _json

    # ── Pre-analyse trade log for patterns ────────────────────────────────────
    closed = [t for t in raw_trades if t.get("outcome") in ("WIN", "LOSS")]
    wins   = [t for t in closed if t["outcome"] == "WIN"]
    losses = [t for t in closed if t["outcome"] == "LOSS"]

    # Hold time: wins vs losses (in hours)
    avg_win_hold  = round(sum(t.get("hold_candles", 0) for t in wins)   / len(wins),   1) if wins   else 0
    avg_loss_hold = round(sum(t.get("hold_candles", 0) for t in losses) / len(losses), 1) if losses else 0

    # Consecutive loss streak
    max_consec_loss = 0
    cur_streak = 0
    for t in closed:
        if t["outcome"] == "LOSS":
            cur_streak += 1
            max_consec_loss = max(max_consec_loss, cur_streak)
        else:
            cur_streak = 0

    # Equity curve: at what % of trades did peak equity occur?
    peak_idx = 0
    peak_val = 0.0
    for idx, pt in enumerate(equity_curve):
        if pt.get("y", 0) > peak_val:
            peak_val = pt["y"]
            peak_idx = idx
    peak_pct = round(peak_idx / max(len(equity_curve) - 1, 1) * 100) if equity_curve else 50

    # P&L distribution: biggest single win / loss
    biggest_win  = max((t["pnl_pct"] for t in wins),   default=0)
    biggest_loss = min((t["pnl_pct"] for t in losses), default=0)

    # Hour-of-day clustering: parse "Feb 15 14:00" → hour
    win_hours  = []
    loss_hours = []
    for t in closed:
        try:
            hr = int(t.get("entry_date", "").split(" ")[-1].split(":")[0])
            (win_hours if t["outcome"] == "WIN" else loss_hours).append(hr)
        except Exception:
            pass
    # Find 6-hour bucket with most losses (0, 6, 12, 18)
    loss_hour_dist = {}
    for h in loss_hours:
        b = (h // 6) * 6
        loss_hour_dist[b] = loss_hour_dist.get(b, 0) + 1
    worst_loss_bucket = max(loss_hour_dist, key=loss_hour_dist.get) if loss_hour_dist else None
    bucket_labels = {0: "00:00–06:00 UTC", 6: "06:00–12:00 UTC", 12: "12:00–18:00 UTC", 18: "18:00–00:00 UTC"}
    worst_loss_window = bucket_labels.get(worst_loss_bucket, "unknown") if worst_loss_bucket is not None else "no clear cluster"
    loss_in_worst = loss_hour_dist.get(worst_loss_bucket, 0) if worst_loss_bucket is not None else 0
    loss_pct_in_worst = round(loss_in_worst / max(len(losses), 1) * 100) if losses else 0

    # First-half vs second-half performance
    mid = len(closed) // 2
    first_half_wr  = round(sum(1 for t in closed[:mid]  if t["outcome"] == "WIN") / max(mid, 1) * 100, 1)
    second_half_wr = round(sum(1 for t in closed[mid:]  if t["outcome"] == "WIN") / max(len(closed) - mid, 1) * 100, 1)

    # Trade log snippet (max 40 trades for token budget)
    trade_snippet = "\n".join(
        f"  {t.get('entry_date','?')} → {t.get('exit_date','?')}  {t.get('outcome','?'):4s}  {'+' if t.get('pnl_pct',0)>=0 else ''}{t.get('pnl_pct',0)}%  held {t.get('hold_candles',0)}h"
        for t in closed[:40]
    )

    pattern_block = f"""
TRADE-BY-TRADE PATTERNS (derived from actual backtest data):
- Avg hold time — winners: {avg_win_hold}h  |  losers: {avg_loss_hold}h
- Worst consecutive loss streak: {max_consec_loss}
- Biggest single win: +{biggest_win}%  |  Biggest single loss: {biggest_loss}%
- Equity curve peaked at {peak_pct}% through the test period ({"strategy degraded in 2nd half" if peak_pct < 55 else "strategy improved over time" if peak_pct > 75 else "fairly consistent"})
- First-half win rate: {first_half_wr}%  |  Second-half win rate: {second_half_wr}%
- Loss clustering: {loss_pct_in_worst}% of all losses occurred in {worst_loss_window}
- Total trades logged: {len(closed)}

FULL TRADE LOG:
{trade_snippet if trade_snippet else "  (no closed trades)"}
"""

    system_prompt = (
        "You are a quantitative trading strategy optimizer. "
        "You analyze actual backtested trade-by-trade data and equity curve patterns to find specific, data-backed parameter changes that improve ROI. "
        "Every suggestion must reference a specific pattern you found in the trade log or equity curve. "
        "Respond with valid JSON only — no markdown, no explanation outside the JSON."
    )
    user_prompt = f"""Analyze this backtested {direction} strategy on {coin} and return EXACTLY 3 improvement suggestions grounded in the actual trade data below.

CURRENT CONFIG:
- Primary indicator: {primary_type} with params {_json.dumps(primary_cfg)}
- Confirmations: {_json.dumps(confirms)}
- Take Profit: {tp1}%  |  Stop Loss: {sl}%  |  Leverage: {leverage}x
- Backtest window: {days} days
{pattern_block}
AGGREGATE RESULTS:
- Win Rate: {win_rate}%  |  Total P&L: {total_pnl}%  |  Profit Factor: {profit_factor}
- Max Drawdown: {max_drawdown}%  |  Trades: {closed_trades} ({trades_per_day}/day)  |  Avg Hold: {round(avg_hold_mins/60,1)}h

RULES:
1. Each suggestion must be DIRECTLY justified by a specific pattern in the trade log above (e.g. "losers held 3× longer than winners" → cap hold time).
2. Each suggestion must target a DIFFERENT lever: (a) indicator threshold/period, (b) TP/SL/hold-time, (c) confirm filter or leverage.
3. Changes must be specific numbers.
4. "changes" keys must exactly match: primaryCfg (sub-object with only changed sub-keys), tp1, sl, leverage, confirms (full array).
5. Only include keys in "changes" that differ from current values.
6. No leverage above 10x, no SL below 0.5%.

Respond with ONLY this JSON:
{{
  "suggestions": [
    {{
      "title": "short verb-first title under 8 words",
      "why": "cite the specific trade pattern that justifies this change",
      "changes": {{ ... }},
      "expected": "quantified estimate e.g. win rate +8-12%, P&L ~+35%"
    }},
    {{...}},
    {{...}}
  ]
}}"""

    def _parse_suggestions(raw: str) -> list:
        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        return _json.loads(raw.strip()).get("suggestions", [])

    # ── Try Claude Haiku first ──────────────────────────────────────────────
    try:
        import anthropic as _anthropic
        _ac = _anthropic.AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        _msg = await asyncio.wait_for(
            _ac.messages.create(
                model="claude-sonnet-4-5",
                max_tokens=900,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
            ),
            timeout=28,
        )
        suggestions = _parse_suggestions(_msg.content[0].text)
        return {"suggestions": suggestions[:3]}
    except asyncio.TimeoutError:
        return {"error": "AI took too long — please try again."}
    except Exception as _claude_err:
        logger.warning(f"[BtSuggest] Claude failed ({_claude_err}), trying xAI fallback…")

    # ── Fallback: xAI / Grok (OpenAI-compatible) ────────────────────────────
    try:
        from openai import AsyncOpenAI as _AsyncOpenAI
        _xc = _AsyncOpenAI(api_key=os.getenv("XAI_API_KEY"), base_url="https://api.x.ai/v1")
        _xmsg = await asyncio.wait_for(
            _xc.chat.completions.create(
                model="grok-3-mini",
                max_tokens=900,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": user_prompt},
                ],
                response_format={"type": "json_object"},
            ),
            timeout=28,
        )
        suggestions = _parse_suggestions(_xmsg.choices[0].message.content)
        return {"suggestions": suggestions[:3]}
    except Exception as exc:
        logger.error(f"[BtSuggest] xAI fallback also failed uid={uid}: {exc}", exc_info=True)
        return {"error": "AI suggestion service is temporarily unavailable. Try again in a moment."}


# ── AI Strategy Tuner ───────────────────────────────────────────────────────────
# Generates ~9 parameter variants of an existing strategy, backtests them all in
# parallel, ranks by a (pnl − 0.5×drawdown) score, and lets the user one-tap
# apply the winning patch. Pro-only because it fans out into many backtests.

# Allowlist of config keys the optimizer is permitted to mutate. Anything else
# (entry signals, universe, filters, etc.) is left untouched so we never silently
# rewrite the user's strategy logic — only its risk/exit knobs.
_OPTIMIZER_PATCH_ALLOWED_KEYS = {"tp1", "sl", "leverage", "timeframe", "primaryCfg"}
_OPTIMIZER_TF_ORDER = ["5m", "15m", "1h", "4h"]


def _signal_param_variants(primary_type: str, primary_cfg: dict) -> list[tuple[str, str, dict]]:
    """Return (label, tagline, primary_cfg_patch) variants for the strategy's
    primary signal. The primary signal is usually the single biggest lever on
    trade-count and win-rate — way bigger than TP/SL knobs — so we MUST try a
    few values of its main numeric parameter.

    Each entry's patch is a partial dict to merge into the existing primaryCfg
    (NOT a replacement). Returns at most 4 variants per signal type to keep
    the total budget bounded.
    """
    cfg = dict(primary_cfg or {})
    out: list[tuple[str, str, dict]] = []
    pt = (primary_type or "").lower()

    if pt == "rsi":
        base = float(cfg.get("value") or 30)
        for new in (15, 25, 35, 45, 55, 65, 75, 85):
            if abs(new - base) >= 5 and 5 <= new <= 95:
                out.append((f"RSI level {int(new)}", f"RSI {int(base)} → {int(new)}", {"value": new}))
    elif pt == "ema":
        cur = str(cfg.get("periods") or "9/21")
        for p in ("5/13", "9/21", "12/26", "20/50", "50/200"):
            if p != cur:
                try:
                    fast_s, slow_s = p.split("/")
                    fast, slow = int(fast_s), int(slow_s)
                except Exception:
                    continue
                # Backtest engine's indicator/ema branch reads `period` and
                # `period2` — NOT the wizard's `periods` string. So we emit
                # both: `periods` keeps the wizard label valid for the UI,
                # and `period`/`period2` actually drive the backtest.
                out.append((f"EMA {p}", f"Periods {cur} → {p}",
                            {"periods": p, "period": fast, "period2": slow}))
    elif pt == "volume_spike":
        base = float(cfg.get("multiplier") or 2)
        for new in (1.5, 2.5, 3, 4, 5):
            if abs(new - base) >= 0.4:
                out.append((f"Volume ×{new}", f"Multiplier ×{base} → ×{new}", {"multiplier": new}))
    elif pt == "price_momentum":
        base = float(cfg.get("pm_pct") or 10)
        for new in (3, 5, 7, 15, 20):
            if abs(new - base) >= 1:
                out.append((f"Momentum {new}%", f"Threshold {base}% → {new}%", {"pm_pct": new}))
    elif pt == "breakout":
        base = int(cfg.get("bo_lookback") or 20)
        for new in (10, 30, 50, 100):
            if new != base:
                out.append((f"Breakout {new}b", f"Lookback {base} → {new}", {"bo_lookback": new}))
    elif pt == "sma":
        base = int(cfg.get("period") or 200)
        for new in (20, 50, 100, 200):
            if new != base:
                out.append((f"SMA {new}", f"Period {base} → {new}", {"period": new}))
    elif pt == "donchian":
        base = int(cfg.get("period") or 20)
        for new in (10, 20, 30, 50, 100):
            if new != base:
                out.append((f"Donchian {new}", f"Period {base} → {new}", {"period": new}))
    elif pt in ("cci", "mfi", "roc"):
        default_p = 20 if pt == "cci" else 14
        base = int(cfg.get("period") or default_p)
        for new in (7, 14, 21, 28):
            if new != base:
                out.append((f"{pt.upper()} period {new}", f"Period {base} → {new}", {"period": new}))
    # Other signal types (macd, bb, supertrend, ichimoku, divergence, stoch_rsi)
    # are condition-only in the wizard schema — there's no obvious single
    # numeric knob to sweep, so we leave the signal unchanged and rely on
    # risk/exit/leverage/timeframe variants instead.

    # Cap per-signal sweeps at 6 — combined with 8 risk/exit + 2 leverage +
    # 2 timeframe variants, this gives us ~18 total backtests under the
    # 5-concurrent budget (≈3-4 batches at 50s each = 150-220s worst case).
    return out[:6]


def _generate_optimizer_variants(base_config: dict) -> list[dict]:
    """Return a broad set of parameter tweaks to test against the baseline.

    v2 (May 2026) — was 9 risk/exit-only tweaks, now ~16-22 variants including:
      • 8 risk/exit combos (3 SL × 3 TP grid, minus baseline duplicates)
      • 2 leverage tweaks
      • 2 timeframe shifts (one step each direction)
      • Up to 4 primary-signal parameter sweeps (RSI level, EMA periods,
        SMA period, Donchian period, etc — the actual entry knob, which is
        the biggest performance lever on most strategies)
    """
    base_tp  = float(base_config.get("tp1") or 3.0)
    base_sl  = float(base_config.get("sl")  or 2.0)
    base_lev = int(base_config.get("leverage") or 5)
    base_tf  = str(base_config.get("timeframe") or "1h")
    primary_type = str(base_config.get("primaryType") or "")
    primary_cfg  = dict(base_config.get("primaryCfg") or {})

    def _step_tf(direction: int):
        try:
            i = _OPTIMIZER_TF_ORDER.index(base_tf)
            j = i + direction
            if 0 <= j < len(_OPTIMIZER_TF_ORDER):
                return _OPTIMIZER_TF_ORDER[j]
        except ValueError:
            pass
        return None

    out: list[dict] = []
    seen_patches: set[str] = set()  # dedupe identical patches

    def _add(label: str, tagline: str, patch: dict):
        if not patch:
            return
        # Stable-key dedupe — two variants with the same effective patch waste
        # a backtest slot. JSON sort-keys gives a deterministic key.
        try:
            key = json.dumps(patch, sort_keys=True, default=str)
        except Exception:
            key = repr(sorted(patch.items()))
        if key in seen_patches:
            return
        seen_patches.add(key)
        out.append({"label": label, "tagline": tagline, "patch": patch})

    # ── Risk/exit grid: 3 SL multipliers × 3 TP multipliers minus identity ──
    sl_mults = [0.6, 1.0, 1.5]
    tp_mults = [0.6, 1.0, 1.6]
    for sm in sl_mults:
        for tm in tp_mults:
            if sm == 1.0 and tm == 1.0:
                continue  # baseline
            patch: dict = {}
            if sm != 1.0:
                patch["sl"] = round(base_sl * sm, 2)
            if tm != 1.0:
                patch["tp1"] = round(base_tp * tm, 2)
            # Friendly label for the 4 corner/notable combos
            if sm == 0.6 and tm == 1.6:
                lbl, tag = "Better risk/reward", f"TP ×1.6, SL ×0.6"
            elif sm == 0.6 and tm == 0.6:
                lbl, tag = "Tight scalper", f"TP ×0.6, SL ×0.6"
            elif sm == 1.5 and tm == 1.6:
                lbl, tag = "Wide swing", f"TP ×1.6, SL ×1.5"
            elif sm == 1.5 and tm == 0.6:
                lbl, tag = "Mean-revert risk", f"TP ×0.6, SL ×1.5"
            elif sm == 1.0:
                lbl = "Tighter take-profit" if tm < 1 else "Wider take-profit"
                tag = f"TP {base_tp:.1f}% → {patch.get('tp1', base_tp):.1f}%"
            elif tm == 1.0:
                lbl = "Tighter stop-loss" if sm < 1 else "Wider stop-loss"
                tag = f"SL {base_sl:.1f}% → {patch.get('sl', base_sl):.1f}%"
            else:
                lbl = f"SL ×{sm}, TP ×{tm}"
                tag = f"SL → {patch.get('sl', base_sl):.1f}%, TP → {patch.get('tp1', base_tp):.1f}%"
            _add(lbl, tag, patch)

    # ── Leverage ──
    if base_lev > 2:
        new_lev = max(1, base_lev - 3)
        _add("Lower leverage", f"{base_lev}× → {new_lev}×", {"leverage": new_lev})
    if base_lev < 25:
        new_lev = min(25, base_lev + 5)
        _add("Higher leverage", f"{base_lev}× → {new_lev}×", {"leverage": new_lev})

    # ── Timeframe ──
    nxt = _step_tf(1)
    if nxt:
        _add("Slower timeframe", f"{base_tf} → {nxt}", {"timeframe": nxt})
    prv = _step_tf(-1)
    if prv:
        _add("Faster timeframe", f"{base_tf} → {prv}", {"timeframe": prv})

    # ── Primary signal parameter sweeps (THE big lever) ──
    for label, tagline, sub_patch in _signal_param_variants(primary_type, primary_cfg):
        merged_cfg = dict(primary_cfg)
        merged_cfg.update(sub_patch)
        _add(label, tagline, {"primaryCfg": merged_cfg})

    return out


def _generate_replay_variants(base_config: dict) -> list[dict]:
    """Variants for trade-replay mode. Only knobs that affect EXIT logic are
    sweepable here — entries are fixed (we're replaying real fills), so
    timeframe / primaryCfg / signal tweaks would be no-ops. We sweep TP, SL,
    and leverage in a wider grid than the backtest mode because replay is
    nearly free (no per-variant candle fetch — we share the same candles).
    """
    base_tp  = float(base_config.get("tp1") or 3.0)
    base_sl  = float(base_config.get("sl")  or 2.0)
    base_lev = int(base_config.get("leverage") or 5)

    out: list[dict] = []
    seen: set[str] = set()

    def _add(label: str, tagline: str, patch: dict):
        if not patch:
            return
        try:
            key = json.dumps(patch, sort_keys=True, default=str)
        except Exception:
            key = repr(sorted(patch.items()))
        if key in seen:
            return
        seen.add(key)
        out.append({"label": label, "tagline": tagline, "patch": patch})

    # Wider 4×4 SL/TP grid since replay is nearly free
    sl_mults = [0.5, 0.75, 1.25, 1.75]
    tp_mults = [0.5, 0.75, 1.25, 2.0]
    for sm in sl_mults:
        new_sl = round(base_sl * sm, 2)
        if new_sl <= 0:
            continue
        _add(f"SL ×{sm}", f"SL {base_sl:.1f}% → {new_sl:.1f}%", {"sl": new_sl})
    for tm in tp_mults:
        new_tp = round(base_tp * tm, 2)
        if new_tp <= 0:
            continue
        _add(f"TP ×{tm}", f"TP {base_tp:.1f}% → {new_tp:.1f}%", {"tp1": new_tp})

    # Combo cells worth highlighting
    combos = [
        (0.5, 1.5, "Better risk/reward",  "Tight SL ×0.5, wide TP ×1.5"),
        (0.75, 2.0, "Aggressive R:R",     "SL ×0.75, TP ×2.0"),
        (1.5, 0.75, "Wider stop, quick TP", "Patient SL, fast TP"),
        (0.75, 0.75, "Tight scalper",     "Both ×0.75"),
    ]
    for sm, tm, lbl, tag in combos:
        patch = {"sl": round(base_sl * sm, 2), "tp1": round(base_tp * tm, 2)}
        if patch["sl"] > 0 and patch["tp1"] > 0:
            _add(lbl, tag, patch)

    # Leverage sweeps
    for delta in (-3, -2, +3, +5):
        new_lev = max(1, min(25, base_lev + delta))
        if new_lev != base_lev:
            _add(f"{new_lev}× leverage",
                 f"{base_lev}× → {new_lev}×",
                 {"leverage": new_lev})

    return out


def _replay_trade(
    entry_price: float,
    direction: str,
    candles: list,
    tp_pct: float,
    sl_pct: float,
    leverage: int,
) -> dict:
    """Replay a single execution against new TP/SL/leverage. Walks the post-entry
    candles minute-by-minute and decides whether the variant's TP or SL would
    have hit first.

    Conservative tie-break: when a single candle's high+low both straddle the
    levels (we can't tell from OHLC which touched first), we assume the SL
    fired — this avoids over-promising winners. Industry-standard for
    backtesting against OHLC bars.

    Returns: {outcome, pnl_pct} where pnl_pct already includes leverage.
    """
    if entry_price <= 0 or not candles:
        return {"outcome": "OPEN", "pnl_pct": 0.0}

    is_long = (direction or "LONG").upper() == "LONG"
    if is_long:
        tp_price = entry_price * (1 + tp_pct / 100.0)
        sl_price = entry_price * (1 - sl_pct / 100.0)
    else:
        tp_price = entry_price * (1 - tp_pct / 100.0)
        sl_price = entry_price * (1 + sl_pct / 100.0)

    for c in candles:
        # candles are (ts, open, high, low, close) tuples
        try:
            high = float(c[2]); low = float(c[3])
        except (IndexError, TypeError, ValueError):
            continue

        if is_long:
            sl_hit = low  <= sl_price
            tp_hit = high >= tp_price
        else:
            sl_hit = high >= sl_price
            tp_hit = low  <= tp_price

        if sl_hit and tp_hit:
            return {"outcome": "LOSS", "pnl_pct": round(-sl_pct * leverage, 2)}
        if sl_hit:
            return {"outcome": "LOSS", "pnl_pct": round(-sl_pct * leverage, 2)}
        if tp_hit:
            return {"outcome": "WIN",  "pnl_pct": round( tp_pct * leverage, 2)}

    # Window expired with neither level hit — mark with mark-to-market PnL on
    # last close so unfinished trades still affect the variant's scoring (a
    # variant that leaves trades open isn't necessarily neutral — it has
    # opportunity cost).
    try:
        last_close = float(candles[-1][4])
        raw_pct = ((last_close - entry_price) / entry_price) * 100.0
        if not is_long:
            raw_pct = -raw_pct
        return {"outcome": "OPEN", "pnl_pct": round(raw_pct * leverage, 2)}
    except Exception:
        return {"outcome": "OPEN", "pnl_pct": 0.0}


def _aggregate_replay(outcomes: list[dict]) -> dict:
    """Roll up replay results into the same stats shape as run_backtest()."""
    if not outcomes:
        return {"total_pnl": 0.0, "win_rate": 0.0, "max_drawdown": 0.0,
                "closed_trades": 0, "trades": 0, "profit_factor": 0.0,
                "liquidations": 0}

    pnls   = [o["pnl_pct"] for o in outcomes]
    closed = [o for o in outcomes if o["outcome"] in ("WIN", "LOSS")]
    wins   = [o for o in closed if o["outcome"] == "WIN"]
    losses = [o for o in closed if o["outcome"] == "LOSS"]

    win_rate = (len(wins) / len(closed) * 100.0) if closed else 0.0
    total_pnl = sum(pnls)

    # Max drawdown via equity-curve walk
    eq = 0.0; peak = 0.0; max_dd = 0.0
    for p in pnls:
        eq += p
        if eq > peak:
            peak = eq
        dd = peak - eq
        if dd > max_dd:
            max_dd = dd

    # Profit factor = sum(wins) / |sum(losses)|
    win_sum  = sum(o["pnl_pct"] for o in wins)
    loss_sum = abs(sum(o["pnl_pct"] for o in losses))
    profit_factor = (win_sum / loss_sum) if loss_sum > 0 else (win_sum if win_sum > 0 else 0.0)

    # Liquidations = trades whose loss exceeded ~80% of nominal (proxy)
    liquidations = sum(1 for p in pnls if p <= -80.0)

    return {
        "total_pnl":     round(total_pnl, 2),
        "win_rate":      round(win_rate, 2),
        # Positive convention to match run_backtest()'s stats — UI's delta_dd
        # logic in `VariantCard` already inverts the comparison (smaller is
        # better), so emitting a negative number here would double-invert and
        # mis-rank variants.
        "max_drawdown":  round(max_dd, 2),
        "closed_trades": len(closed),
        "trades":        len(outcomes),
        "profit_factor": round(profit_factor, 2),
        "liquidations":  liquidations,
    }


async def _run_trade_replay_optimizer(
    base_config: dict,
    executions: list,
    strategy_id: int,
) -> dict | None:
    """Replay every variant against the user's actual past fills. Returns the
    same-shape response as the backtest path, or None if we couldn't fetch
    enough candle data to make the result trustworthy.
    """
    import httpx
    variants = _generate_replay_variants(base_config)
    if not variants:
        return None

    # Asset-class routing: tradfi (stocks/forex/indices) goes through yfinance
    # via the shared executor helper. Crypto retains the hand-rolled MEXC →
    # Binance paging below for max throughput.
    _replay_ac = ((base_config.get("asset_class") or "crypto") or "crypto").lower().strip()

    # Pre-fetch candles ONCE per execution (shared across all variants — this
    # is the whole performance unlock vs the backtest path).
    fetch_concurrency = asyncio.Semaphore(15)

    async def _fetch_for(ex):
        """Bounded 1m OHLC fetch from `fired_at` to `end_at`.

        Architect-flagged: the upstream `_fetch_candles_since_entry` always
        fetches up to NOW with no endTime arg, so a 60-day-old execution would
        page through ~86k candles before we trimmed — multiplied across all
        executions that blew the 90s replay budget. This bounded version
        passes endTime to MEXC/Binance directly so we only fetch what we need.
        """
        async with fetch_concurrency:
            try:
                fired = ex.fired_at or datetime.utcnow()
                # Window ends at min(closed_at, fired+14d, now).
                hard_cap = fired + timedelta(days=14)
                end_at = ex.closed_at or hard_cap
                if end_at > hard_cap:
                    end_at = hard_cap
                now = datetime.utcnow()
                if end_at > now:
                    end_at = now
                if end_at <= fired:
                    return None

                start_ms = int(fired.timestamp()  * 1000)
                end_ms   = int(end_at.timestamp() * 1000)

                # ── Tradfi path: yfinance 1m candles via the shared helper ─
                # yfinance 1m data is capped at 5d upstream, so trades older
                # than that simply return None and we fall back to backtest.
                _ex_ac = (getattr(ex, "asset_class", None) or _replay_ac or "crypto")
                if _ex_ac != "crypto":
                    from app.services.strategy_executor import _fetch_candles_since_entry as _exec_fetch
                    async with httpx.AsyncClient(timeout=8) as client:
                        tup_candles = await _exec_fetch(ex.symbol, fired, client, _ex_ac)
                    # Trim to end_at window
                    filtered = [c for c in tup_candles if c[0] <= end_ms]
                    return filtered or None
                # Page through in 900-candle chunks (MEXC safe)
                all_candles: list = []
                cursor = start_ms
                async with httpx.AsyncClient(timeout=8) as client:
                    while cursor < end_ms:
                        # 1 candle per minute ⇒ minutes-needed = (end-cursor)/60s
                        needed = min(900, max(1, int((end_ms - cursor) / 60000) + 1))
                        sources = [
                            ("https://api.mexc.com/api/v3/klines",
                             {"symbol": ex.symbol, "interval": "1m",
                              "startTime": cursor, "endTime": end_ms, "limit": needed}),
                            ("https://api.binance.com/api/v3/klines",
                             {"symbol": ex.symbol, "interval": "1m",
                              "startTime": cursor, "endTime": end_ms, "limit": needed}),
                            ("https://fapi.binance.com/fapi/v1/klines",
                             {"symbol": ex.symbol, "interval": "1m",
                              "startTime": cursor, "endTime": end_ms, "limit": needed}),
                        ]
                        page: list = []
                        for url, params in sources:
                            try:
                                resp = await client.get(url, params=params)
                                if resp.status_code != 200:
                                    continue
                                klines = resp.json()
                                if not klines:
                                    continue
                                for k in klines:
                                    page.append((int(k[0]), float(k[1]), float(k[2]),
                                                 float(k[3]), float(k[4])))
                                break
                            except Exception:
                                continue
                        if not page:
                            break  # no source returned data — give up gracefully
                        all_candles.extend(page)
                        # Advance cursor past the last candle we got
                        last_ts = page[-1][0]
                        if last_ts <= cursor:
                            break  # no progress, prevent infinite loop
                        cursor = last_ts + 60_000
                return all_candles or None
            except Exception as exc:
                logger.warning(f"[Replay sid={strategy_id}] candles failed for ex={ex.id}: {exc}")
                return None

    candles_lists = await asyncio.gather(*[_fetch_for(ex) for ex in executions])
    paired = [(ex, c) for ex, c in zip(executions, candles_lists) if c]
    if len(paired) < 5:
        # Couldn't fetch candles for enough trades — fall back to backtest mode.
        return None

    base_tp  = float(base_config.get("tp1") or 3.0)
    base_sl  = float(base_config.get("sl")  or 2.0)
    base_lev = int(base_config.get("leverage") or 5)

    def _replay_variant(patch: dict) -> dict:
        """Replay all paired trades with the patched TP/SL/lev — returns stats."""
        tp = float(patch.get("tp1",      base_tp))
        sl = float(patch.get("sl",       base_sl))
        lv = int(  patch.get("leverage", base_lev))
        outs = [_replay_trade(ex.entry_price, ex.direction, candles, tp, sl, lv)
                for ex, candles in paired]
        return _aggregate_replay(outs)

    # Run baseline + all variants concurrently in threads (CPU-bound, no I/O).
    all_patches = [{}] + [v["patch"] for v in variants]
    all_stats = await asyncio.gather(*[
        asyncio.to_thread(_replay_variant, p) for p in all_patches
    ])
    baseline_stats = all_stats[0]
    base_score = _optimizer_score(baseline_stats)
    base_pnl   = float(baseline_stats.get("total_pnl") or 0.0)
    base_dd    = float(baseline_stats.get("max_drawdown") or 0.0)
    base_wr    = float(baseline_stats.get("win_rate") or 0.0)

    out_variants: list[dict] = []
    for v, stats in zip(variants, all_stats[1:]):
        score = _optimizer_score(stats)
        out_variants.append({
            "label":          v["label"],
            "tagline":        v["tagline"],
            "patch":          v["patch"],
            "stats":          stats,
            "score":          round(score, 2),
            "delta_pnl":      round(float(stats.get("total_pnl") or 0.0) - base_pnl, 2),
            "delta_dd":       round(float(stats.get("max_drawdown") or 0.0) - base_dd, 2),
            "delta_win_rate": round(float(stats.get("win_rate") or 0.0) - base_wr, 2),
            "improved":       score > base_score,
        })

    out_variants.sort(key=lambda v: v["score"], reverse=True)
    any_improved = any(v["improved"] for v in out_variants)

    return {
        "mode":                 "replay",
        "trades_replayed":      len(paired),
        "trades_total":         len(executions),
        "days":                 None,
        "baseline": {
            "label": "Current settings",
            "stats": baseline_stats,
            "score": round(base_score, 2),
        },
        "variants":             out_variants,
        "any_improved":         any_improved,
        "baseline_zero_trades": False,  # by definition — we have real trades
        "primary_tunable":      True,
        "total_tested":         len(out_variants),
        "total_attempted":      len(out_variants),
        "ran_at":               datetime.utcnow().isoformat() + "Z",
    }


def _optimizer_score(stats: dict) -> float:
    """Risk-adjusted score: PnL minus half the max drawdown.

    Penalising drawdown stops the ranker from promoting a "+200% but blew up
    twice" variant over a steadier "+80% with 12% DD" one.
    """
    try:
        pnl = float(stats.get("total_pnl") or 0.0)
        dd  = float(stats.get("max_drawdown") or 0.0)
        return pnl - 0.5 * abs(dd)
    except Exception:
        return -1e9


@app.post("/api/strategies/{strategy_id}/optimize")
async def api_strategy_optimize(strategy_id: int, request: Request):
    """Run a fan-out backtest sweep over parameter variants of a strategy.

    Body: { uid, days?: 30 | 90 }
    Returns: { baseline: {label, stats, score}, variants: [{label, tagline, patch, stats, score, delta_pnl, delta_dd, delta_win_rate, improved}], days, ran_at }
    Pro-gated. Each backtest runs with a 60s budget; whole call capped at 120s.
    """
    body = await request.json()
    uid  = (body.get("uid") or "").strip()
    days = int(body.get("days") or 30)
    if days not in (30, 90):
        days = 30

    from app.database import SessionLocal
    db = SessionLocal()
    try:
        user = _get_user_by_uid_safe(uid, db)
        if not user:
            raise HTTPException(status_code=403, detail="Invalid UID")
        _sub = _get_portal_sub(user.id, db)
        if not _is_portal_pro(_sub) and not getattr(user, "is_admin", False):
            return JSONResponse(
                status_code=402,
                content={"error": "PRO_REQUIRED",
                         "message": "A Pro subscription is required to run the AI tuner."},
            )

        from app.strategy_models import UserStrategy
        s = db.query(UserStrategy).filter(
            UserStrategy.id == strategy_id, UserStrategy.user_id == user.id
        ).first()
        if not s:
            raise HTTPException(status_code=404, detail="strategy not found")
        # Locked (subscribed-from-marketplace) strategies have IP-protected entry
        # logic, so we'd be spinning the optimizer over knobs the user can't keep
        # afterwards. Block early with a clear message.
        if (s.config or {}).get("_locked"):
            return JSONResponse(
                status_code=400,
                content={"error": "LOCKED",
                         "message": "Subscribed strategies can't be optimized — clone it first to make it editable."},
            )
        base_config = dict(s.config or {})

        # ── Try TRADE REPLAY mode first ──────────────────────────────────
        # Pull the strategy's recent closed executions (paper or live). If
        # there are enough of them, we can replay the user's actual fills
        # against each variant's TP/SL/leverage — this works even when the
        # synthetic backtest fires zero trades, and the answers are concrete:
        # "if your SL had been 1.5% these specific 8 historical trades would
        # have been wins instead of losses."
        from app.strategy_models import StrategyExecution
        cutoff = datetime.utcnow() - timedelta(days=120)
        executions = (
            db.query(StrategyExecution)
            .filter(
                StrategyExecution.strategy_id == strategy_id,
                StrategyExecution.user_id == user.id,
                StrategyExecution.outcome.in_(("WIN", "LOSS", "BREAKEVEN", "OPEN")),
                StrategyExecution.entry_price.isnot(None),
                StrategyExecution.fired_at >= cutoff,
            )
            .order_by(StrategyExecution.fired_at.desc())
            .limit(60)
            .all()
        )
        # Detach so we can use them after closing the session
        executions_detached = [
            type("ExSnap", (), {
                "id":          ex.id,
                "symbol":      ex.symbol,
                "direction":   ex.direction,
                "entry_price": ex.entry_price,
                "leverage":    ex.leverage or 5,
                "fired_at":    ex.fired_at,
                "closed_at":   ex.closed_at,
                "asset_class": (getattr(ex, "asset_class", None) or "crypto"),
            })()
            for ex in executions
        ]
    finally:
        db.close()

    if len(executions_detached) >= 5:
        try:
            replay = await asyncio.wait_for(
                _run_trade_replay_optimizer(base_config, executions_detached, strategy_id),
                timeout=55,
            )
            if replay:
                return replay
        except asyncio.TimeoutError:
            logger.warning(f"[Optimize sid={strategy_id}] replay timed out, falling back to backtest")
        except Exception as exc:
            logger.warning(f"[Optimize sid={strategy_id}] replay failed: {exc}, falling back to backtest")

    variants = _generate_optimizer_variants(base_config)
    if not variants:
        return JSONResponse(
            status_code=400,
            content={"error": "NO_VARIANTS",
                     "message": "Couldn't generate any parameter variants for this strategy."},
        )

    from app.services.backtest_engine import run_backtest

    # Cap concurrency so we don't blow up worker memory / hit Gate.io rate
    # limits when many backtests all request candles at once. Bumped from 5 → 8
    # to reduce wall time; per-job timeout tightened from 50 → 25s (Gate.io
    # candles typically arrive in <5s — 25s is still very generous headroom).
    concurrency = 8
    sem = asyncio.Semaphore(concurrency)

    async def _run_one(cfg: dict) -> dict | None:
        async with sem:
            try:
                # run_backtest returns {symbol, days, trades, stats:{...}, equity_curve, ...}
                # We only care about the stats sub-dict for ranking + the wire payload.
                full = await asyncio.wait_for(run_backtest(cfg, days), timeout=25)
                if not isinstance(full, dict):
                    return None
                if full.get("error"):
                    return None
                inner = full.get("stats")
                if not isinstance(inner, dict):
                    return None
                return inner
            except Exception as exc:
                logger.warning(f"[Optimize sid={strategy_id}] variant failed: {exc}")
                return None

    # Build the full job list: baseline + each variant config
    jobs: list[tuple[str, dict, dict | None]] = [("__baseline__", dict(base_config), None)]
    for v in variants:
        merged = dict(base_config)
        merged.update(v["patch"])
        jobs.append((v["label"], merged, v))

    # Worst-case wall time = ceil(N / concurrency) * per_job_timeout + buffer.
    import math
    outer_timeout = math.ceil(len(jobs) / concurrency) * 25 + 15

    try:
        results = await asyncio.wait_for(
            asyncio.gather(*[_run_one(cfg) for _, cfg, _ in jobs]),
            timeout=outer_timeout,
        )
    except asyncio.TimeoutError:
        return JSONResponse(
            status_code=408,
            content={"error": "TIMEOUT",
                     "message": "Optimizer timed out. Try again or use a 30-day window."},
        )

    baseline_stats = results[0]
    if not baseline_stats:
        return JSONResponse(
            status_code=500,
            content={"error": "BASELINE_FAILED",
                     "message": "Couldn't backtest the current strategy. The optimizer can't compare without a baseline."},
        )
    base_score = _optimizer_score(baseline_stats)
    base_pnl   = float(baseline_stats.get("total_pnl") or 0.0)
    base_dd    = float(baseline_stats.get("max_drawdown") or 0.0)
    base_wr    = float(baseline_stats.get("win_rate") or 0.0)

    out_variants: list[dict] = []
    for (label, _cfg, vmeta), stats in zip(jobs[1:], results[1:]):
        if not stats or vmeta is None:
            continue
        score = _optimizer_score(stats)
        out_variants.append({
            "label":          label,
            "tagline":        vmeta["tagline"],
            "patch":          vmeta["patch"],
            "stats":          stats,
            "score":          round(score, 2),
            "delta_pnl":      round(float(stats.get("total_pnl") or 0.0) - base_pnl, 2),
            "delta_dd":       round(float(stats.get("max_drawdown") or 0.0) - base_dd, 2),
            "delta_win_rate": round(float(stats.get("win_rate") or 0.0) - base_wr, 2),
            "improved":       score > base_score,
        })

    # Best variant first
    out_variants.sort(key=lambda v: v["score"], reverse=True)

    # Convenience flag for the mobile UI — lets us show a clear "every variant
    # was worse" banner without the client recomputing it.
    any_improved = any(v["improved"] for v in out_variants)

    # Diagnostic: if the baseline fired zero trades, the strategy's entry
    # conditions are too restrictive — tweaking SL/TP can't help. Mobile shows
    # a different (more useful) empty state in that case.
    base_trade_count = int(baseline_stats.get("closed_trades") or 0)
    baseline_zero_trades = base_trade_count == 0
    # Whether this strategy's primary signal has tunable numeric parameters in
    # the optimizer. macd / bb / supertrend / stoch_rsi / ichimoku are all
    # hardcoded inside the backtest engine (no period/multiplier args), so
    # there's literally nothing to sweep on the entry side. The mobile uses
    # this to show a stronger "switch your signal" message instead of a vague
    # "well-tuned" one when the strategy isn't firing.
    _TUNABLE_PRIMARIES = {"rsi", "ema", "sma", "donchian", "cci", "mfi", "roc",
                          "volume_spike", "price_momentum", "breakout"}
    primary_type_lc = str(base_config.get("primaryType") or "").lower()
    primary_tunable = primary_type_lc in _TUNABLE_PRIMARIES
    # How many variants returned actual results (i.e. backtest didn't error
    # out or time out). Useful for the "Tested 18 of 22 variations" copy.
    completed = sum(1 for r in results[1:] if r)

    return {
        "days":     days,
        "baseline": {
            "label": "Current settings",
            "stats": baseline_stats,
            "score": round(base_score, 2),
        },
        "mode":                 "backtest",
        "trades_replayed":      0,
        "variants":             out_variants,
        "any_improved":         any_improved,
        "baseline_zero_trades": baseline_zero_trades,
        "primary_tunable":      primary_tunable,
        "primary_type":         primary_type_lc,
        "total_tested":         completed,
        "total_attempted":      len(jobs) - 1,
        "ran_at":               datetime.utcnow().isoformat() + "Z",
    }


@app.post("/api/strategies/{strategy_id}/optimize/apply")
async def api_strategy_optimize_apply(strategy_id: int, request: Request):
    """Apply a single optimizer-suggested patch to the strategy's config.

    Body: { uid, patch: { tp1?, sl?, leverage?, timeframe? } }
    Only allowlisted keys are written; everything else is ignored. Locked
    (subscribed) strategies cannot be optimized.
    """
    body  = await request.json()
    uid   = (body.get("uid") or "").strip()
    patch = body.get("patch") or {}
    if not isinstance(patch, dict) or not patch:
        raise HTTPException(status_code=400, detail="patch required")

    # Strip anything outside the allowlist BEFORE we touch the DB so a bad
    # client can't sneak in arbitrary config edits via this endpoint.
    clean: dict = {}
    for k, v in patch.items():
        if k not in _OPTIMIZER_PATCH_ALLOWED_KEYS:
            continue
        if k == "leverage":
            try:
                v = max(1, min(50, int(v)))
            except Exception:
                continue
        elif k in ("tp1", "sl"):
            try:
                v = round(float(v), 2)
                if v <= 0:
                    continue
            except Exception:
                continue
        elif k == "timeframe":
            if v not in _OPTIMIZER_TF_ORDER:
                continue
        elif k == "primaryCfg":
            # Must be a dict; we replace the whole primaryCfg block since
            # different signal types have different keys and a partial merge
            # could leave stale keys from the old type.
            if not isinstance(v, dict):
                continue
            # Sanity-cap: prevent payload bombs.
            if len(v) > 32:
                continue
        clean[k] = v
    if not clean:
        raise HTTPException(status_code=400, detail="no valid patch keys")

    from app.database import SessionLocal
    db = SessionLocal()
    try:
        user = _get_user_by_uid_safe(uid, db)
        if not user:
            raise HTTPException(status_code=403, detail="Invalid UID")
        # Apply must be Pro-gated for parity with the optimize endpoint —
        # otherwise a free user could synthesize their own patch and bypass the
        # paywall by hitting apply directly.
        _sub = _get_portal_sub(user.id, db)
        if not _is_portal_pro(_sub) and not getattr(user, "is_admin", False):
            return JSONResponse(
                status_code=402,
                content={"error": "PRO_REQUIRED",
                         "message": "A Pro subscription is required to apply optimizer suggestions."},
            )

        from app.strategy_models import UserStrategy
        s = db.query(UserStrategy).filter(
            UserStrategy.id == strategy_id, UserStrategy.user_id == user.id
        ).first()
        if not s:
            raise HTTPException(status_code=404, detail="strategy not found")
        if (s.config or {}).get("_locked"):
            raise HTTPException(status_code=400, detail="locked strategy — clone first")

        # SQLAlchemy's JSON column doesn't track in-place mutations, so we
        # reassign a fresh dict to force the UPDATE.
        new_config = dict(s.config or {})
        new_config.update(clean)
        s.config = new_config
        db.commit()
        return {"ok": True, "applied": clean, "config": new_config}
    finally:
        db.close()


@app.get("/admin/test-bitunix")
async def admin_test_bitunix(secret: str = Query(...), user_id: int = Query(...)):
    """
    Admin diagnostic: test a subscriber's Bitunix connection.
    Returns balance, open positions, and any errors so you can see exactly
    why a user's live orders are failing.
    """
    _require_admin_secret(secret)

    from app.database import SessionLocal
    from app.models import User, UserPreference
    from app.utils.encryption import decrypt_api_key

    db = SessionLocal()
    result: dict = {}
    try:
        user = db.query(User).filter(User.id == user_id).first()
        if not user:
            return JSONResponse({"error": f"User {user_id} not found"}, status_code=404)

        prefs = db.query(UserPreference).filter_by(user_id=user_id).first()
        result["user_id"]    = user_id
        result["username"]   = user.username
        result["uid"]        = user.uid
        result["is_admin"]   = user.is_admin
        result["auto_trading_enabled"] = getattr(prefs, "auto_trading_enabled", False) if prefs else False
        result["has_api_key"]    = bool(getattr(prefs, "bitunix_api_key", None)) if prefs else False
        result["has_api_secret"] = bool(getattr(prefs, "bitunix_api_secret", None)) if prefs else False
        result["bitunix_uid"]    = getattr(prefs, "bitunix_uid", None) if prefs else None

        if not prefs or not prefs.auto_trading_enabled:
            result["verdict"] = "❌ auto_trading_enabled is False — live trading disabled"
            return JSONResponse(result)

        if not getattr(prefs, "bitunix_api_key", None) or not getattr(prefs, "bitunix_api_secret", None):
            result["verdict"] = "❌ API keys not saved"
            return JSONResponse(result)

        try:
            raw_key = decrypt_api_key(prefs.bitunix_api_key)
            raw_sec = decrypt_api_key(prefs.bitunix_api_secret)
        except Exception as dec_err:
            result["verdict"] = f"❌ Key decryption failed: {dec_err}"
            return JSONResponse(result)

        if not raw_key or len(raw_key) < 10:
            result["verdict"] = "❌ Decrypted API key is too short / invalid"
            return JSONResponse(result)

        result["key_length"] = len(raw_key)

        from app.services.bitunix_trader import BitunixTrader
        trader = BitunixTrader(api_key=raw_key, api_secret=raw_sec)

        # Test 1: account balance
        try:
            balance = await trader.get_account_balance()
            result["futures_balance_usdt"] = balance
            if balance is None or balance < 0:
                result["balance_error"] = "API returned None/-1 — likely wrong key permissions (needs Futures read)"
            elif balance == 0:
                result["balance_warn"] = "Balance is $0 — user needs to deposit USDT to Bitunix Futures wallet"
        except Exception as bal_err:
            result["balance_error"] = str(bal_err)

        # Test 2: open positions
        try:
            positions = await trader.get_open_positions()
            result["open_positions"] = positions if isinstance(positions, list) else []
            result["open_positions_count"] = len(result["open_positions"])
        except Exception as pos_err:
            result["positions_error"] = str(pos_err)

        # Overall verdict
        bal = result.get("futures_balance_usdt", -1)
        if "balance_error" in result:
            result["verdict"] = f"❌ Bitunix API error fetching balance — check API key has Futures READ permission. Error: {result['balance_error']}"
        elif bal == 0:
            result["verdict"] = "⚠️ Connected OK but Futures balance is $0 — user must deposit to Bitunix Futures wallet"
        elif bal and bal > 0:
            result["verdict"] = f"✅ Connected. Futures balance: ${bal:.2f} USDT. Orders should be going live."
        else:
            result["verdict"] = "❓ Unknown state — check errors above"

        # Recent executions for this user
        from app.strategy_models import StrategyExecution, UserStrategy
        recent_execs = (
            db.query(StrategyExecution, UserStrategy.name)
            .join(UserStrategy, UserStrategy.id == StrategyExecution.strategy_id)
            .filter(StrategyExecution.user_id == user_id)
            .order_by(StrategyExecution.fired_at.desc())
            .limit(10)
            .all()
        )
        result["recent_executions"] = [
            {
                "id": e.StrategyExecution.id,
                "strategy": e.name,
                "symbol": e.StrategyExecution.symbol,
                "direction": e.StrategyExecution.direction,
                "outcome": e.StrategyExecution.outcome,
                "is_paper": e.StrategyExecution.is_paper,
                "has_order_id": bool(e.StrategyExecution.bitunix_order_id),
                "notes": e.StrategyExecution.notes,
                "fired_at": e.StrategyExecution.fired_at.isoformat() if e.StrategyExecution.fired_at else None,
            }
            for e in recent_execs
        ]

    finally:
        db.close()

    return JSONResponse(result)


# ═══════════════════════════════════════════════════════════
# ─── FOLLOW / FEED ─────────────────────────────────────────
# ═══════════════════════════════════════════════════════════

@app.post("/api/follow")
async def api_follow(request: Request):
    """Toggle follow/unfollow a creator. Returns new is_following state."""
    from app.database import SessionLocal, engine
    db = SessionLocal()
    try:
        body = await request.json()
        uid         = body.get("uid", "")
        target_uid  = body.get("target_uid", "")

        viewer  = _get_user_by_uid(uid, db)
        if not viewer:
            raise HTTPException(status_code=403)
        creator = db.query(User).filter(User.uid == target_uid).first()
        if not creator:
            raise HTTPException(status_code=404)
        if viewer.id == creator.id:
            raise HTTPException(status_code=400, detail="Cannot follow yourself")

        from app.social_models import UserFollow, FeedActivity, init_social_tables

        existing = db.query(UserFollow).filter(
            UserFollow.follower_id == viewer.id,
            UserFollow.following_id == creator.id,
        ).first()

        if existing:
            db.delete(existing)
            db.commit()
            return JSONResponse({"is_following": False})
        else:
            db.add(UserFollow(follower_id=viewer.id, following_id=creator.id))
            db.commit()
            follower_count = db.query(UserFollow).filter(UserFollow.following_id == creator.id).count()
            return JSONResponse({"is_following": True, "follower_count": follower_count})
    finally:
        db.close()


@app.get("/api/feed")
async def api_feed(uid: str = Query(...), limit: int = Query(30, ge=1, le=100)):
    """Return recent feed activities from creators the viewer follows."""
    from app.database import SessionLocal, engine
    db = SessionLocal()
    try:
        viewer = _get_user_by_uid(uid, db)
        if not viewer:
            raise HTTPException(status_code=403)

        from app.social_models import UserFollow, FeedActivity, init_social_tables

        following_ids = [
            row.following_id for row in
            db.query(UserFollow).filter(UserFollow.follower_id == viewer.id).all()
        ]
        if not following_ids:
            return JSONResponse([])

        activities = (
            db.query(FeedActivity)
            .filter(FeedActivity.user_id.in_(following_ids))
            .order_by(FeedActivity.created_at.desc())
            .limit(limit)
            .all()
        )

        results = []
        for a in activities:
            creator = db.query(User).filter(User.id == a.user_id).first()
            results.append({
                "id":            a.id,
                "activity_type": a.activity_type,
                "title":         a.title,
                "subtitle":      a.subtitle,
                "strategy_id":   a.strategy_id,
                "listing_id":    a.listing_id,
                "creator_name":  (creator.first_name or creator.username or "Anonymous") if creator else "Unknown",
                "creator_uid":   creator.uid if creator else None,
                "created_at":    a.created_at.isoformat() if a.created_at else None,
            })
        return JSONResponse(results)
    finally:
        db.close()


# ═══════════════════════════════════════════════════════════
# ─── MONTHLY COMPETITIONS ──────────────────────────────────
# ═══════════════════════════════════════════════════════════

@app.get("/api/competition/current")
async def api_competition_current(uid: str = Query(...)):
    """Return the current active competition + live leaderboard rankings."""
    from app.database import SessionLocal, engine
    db = SessionLocal()
    try:
        viewer = _get_user_by_uid(uid, db)
        if not viewer:
            raise HTTPException(status_code=403)

        from app.social_models import Competition, CompetitionEntry, init_social_tables
        from app.strategy_models import StrategyExecution, UserStrategy

        now  = datetime.utcnow()
        comp = (
            db.query(Competition)
            .filter(Competition.status == "active", Competition.ends_at >= now)
            .order_by(Competition.starts_at.desc())
            .first()
        )
        if not comp:
            return JSONResponse({"active": False})

        entries = db.query(CompetitionEntry).filter(
            CompetitionEntry.competition_id == comp.id
        ).all()

        my_entry = next((e for e in entries if e.user_id == viewer.id), None)

        rankings = []
        for entry in entries:
            strategy = db.query(UserStrategy).filter(UserStrategy.id == entry.strategy_id).first()
            author   = db.query(User).filter(User.id == entry.user_id).first()
            if not strategy:
                continue
            # Score = sum of pnl_pct from trades fired during competition window
            trades = db.query(StrategyExecution).filter(
                StrategyExecution.strategy_id == entry.strategy_id,
                StrategyExecution.fired_at >= comp.starts_at,
                StrategyExecution.fired_at <= comp.ends_at,
                StrategyExecution.outcome.in_(["WIN", "LOSS", "BREAKEVEN"]),
            ).all()
            score       = round(sum(t.pnl_pct for t in trades if t.pnl_pct), 2)
            trade_count = len(trades)
            wins        = sum(1 for t in trades if t.outcome == "WIN")
            # Exclude BREAKEVEN from win-rate denominator — neutral outcome.
            decisive    = sum(1 for t in trades if t.outcome in ("WIN", "LOSS"))
            win_rate    = round(wins / decisive * 100, 1) if decisive else 0
            rankings.append({
                "user_id":      entry.user_id,
                "user_name":    (author.first_name or author.username or "Anonymous") if author else "Unknown",
                "user_uid":     author.uid if author else None,
                "strategy_id":  entry.strategy_id,
                "strategy_name": strategy.name,
                "score":        score,
                "trade_count":  trade_count,
                "win_rate":     win_rate,
                "is_me":        entry.user_id == viewer.id,
            })

        rankings.sort(key=lambda x: x["score"], reverse=True)
        for i, r in enumerate(rankings):
            r["rank"] = i + 1

        days_left = max(0, (comp.ends_at - now).days)

        return JSONResponse({
            "active":       True,
            "id":           comp.id,
            "title":        comp.title,
            "description":  comp.description,
            "prize_text":   comp.prize_text,
            "starts_at":    comp.starts_at.isoformat(),
            "ends_at":      comp.ends_at.isoformat(),
            "days_left":    days_left,
            "entry_count":  len(entries),
            "rankings":     rankings[:50],
            "my_entry":     {"strategy_id": my_entry.strategy_id, "entered_at": my_entry.entered_at.isoformat()} if my_entry else None,
        })
    finally:
        db.close()


@app.post("/api/competition/enter")
async def api_competition_enter(request: Request):
    """Enter a strategy into the current active competition."""
    from app.database import SessionLocal, engine
    db = SessionLocal()
    try:
        body        = await request.json()
        uid         = body.get("uid", "")
        strategy_id = body.get("strategy_id")

        viewer = _get_user_by_uid(uid, db)
        if not viewer:
            raise HTTPException(status_code=403)

        from app.social_models import Competition, CompetitionEntry, FeedActivity, init_social_tables
        from app.strategy_models import UserStrategy

        now  = datetime.utcnow()
        comp = (
            db.query(Competition)
            .filter(Competition.status == "active", Competition.ends_at >= now)
            .order_by(Competition.starts_at.desc())
            .first()
        )
        if not comp:
            raise HTTPException(status_code=404, detail="No active competition")

        strategy = db.query(UserStrategy).filter(
            UserStrategy.id == strategy_id,
            UserStrategy.user_id == viewer.id,
        ).first()
        if not strategy:
            raise HTTPException(status_code=404, detail="Strategy not found")

        existing = db.query(CompetitionEntry).filter(
            CompetitionEntry.competition_id == comp.id,
            CompetitionEntry.user_id == viewer.id,
        ).first()
        if existing:
            return JSONResponse({"success": False, "error": "Already entered"})

        db.add(CompetitionEntry(
            competition_id = comp.id,
            user_id        = viewer.id,
            strategy_id    = strategy_id,
        ))
        # Feed activity
        db.add(FeedActivity(
            user_id       = viewer.id,
            activity_type = "competition_entered",
            title         = f"Entered '{strategy.name}' in {comp.title}",
            subtitle      = comp.prize_text,
            strategy_id   = strategy_id,
        ))
        db.commit()
        return JSONResponse({"success": True})
    finally:
        db.close()


@app.post("/admin/auto_trader/archive_by_cadence")
@app.get("/admin/auto_trader/archive_by_cadence")
async def admin_archive_fast_auto_strategies(
    secret: str = Query(...),
    max_cadence_min: int = Query(1, ge=1, le=60),
    user_uid: Optional[str] = Query(None),
):
    """Admin one-click: bulk-archive AI-mode auto-trader strategies whose
    cadence is at or below `max_cadence_min`. Used to clean up runaway
    1-minute strategies that burn Anthropic credits.

    Defaults to `max_cadence_min=1` (only the 1m strategies). Pass
    `user_uid=TH-XXXXXXXX` to scope to one user, or omit to apply globally.
    Soft-archives (status='archived') — same as the per-strategy delete UI,
    so trades already open are unaffected and the row is preserved for stats.
    """
    _require_admin_secret(secret)

    from app.database import SessionLocal
    from app.models import AutoTradeStrategy, User
    db = SessionLocal()
    try:
        q = db.query(AutoTradeStrategy).filter(
            AutoTradeStrategy.status == "active",
            AutoTradeStrategy.cadence_min <= max_cadence_min,
        )
        if user_uid:
            u = db.query(User).filter(User.uid == user_uid).first()
            if not u:
                raise HTTPException(status_code=404, detail=f"user {user_uid} not found")
            q = q.filter(AutoTradeStrategy.user_id == u.id)

        rows = q.all()
        archived = []
        now = datetime.utcnow()
        for s in rows:
            s.status = "archived"
            s.paused_at = now
            archived.append({
                "id": s.id, "user_id": s.user_id, "symbol": s.symbol,
                "timeframe": s.timeframe, "mode": s.mode,
                "cadence_min": s.cadence_min,
                "total_signals": s.total_signals,
                "wins": s.wins, "losses": s.losses,
                "pnl_usd_total": float(s.pnl_usd_total or 0),
            })
        db.commit()
        logger.warning(
            f"[admin] archived {len(archived)} auto-trader strategies "
            f"(max_cadence_min={max_cadence_min}, user_uid={user_uid or 'all'})"
        )
        return {"ok": True, "archived_count": len(archived), "archived": archived}
    finally:
        db.close()


@app.post("/admin/competition/create")
async def admin_competition_create(request: Request, secret: str = Query(...)):
    """Admin-only: create a new monthly competition."""
    _require_admin_secret(secret)

    from app.database import SessionLocal, engine
    db = SessionLocal()
    try:
        body = await request.json()
        from app.social_models import Competition, init_social_tables

        starts = datetime.fromisoformat(body["starts_at"])
        ends   = datetime.fromisoformat(body["ends_at"])

        comp = Competition(
            title       = body.get("title", f"Monthly Competition"),
            description = body.get("description"),
            prize_text  = body.get("prize_text"),
            starts_at   = starts,
            ends_at     = ends,
            status      = "active",
        )
        db.add(comp)
        db.commit()
        return JSONResponse({"success": True, "id": comp.id})
    finally:
        db.close()


# ─────────────────────────────────────────────────────────────────────────────
# AI Strategy Generator — admin endpoints (stats + manual trigger)
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/api/admin/ai-generator/stats")
async def admin_ai_generator_stats(uid: str = Query(...)):
    from app.database import SessionLocal
    db = SessionLocal()
    try:
        user = _get_user_by_uid(uid, db)
        if not user or not getattr(user, "is_admin", False):
            raise HTTPException(status_code=403, detail="Admin only")
    finally:
        db.close()
    try:
        from app.services.ai_strategy_generator import get_stats
        return JSONResponse(get_stats())
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/admin/bitunix/status")
async def admin_bitunix_status(uid: str = Query(...)):
    """Combined snapshot: trading executor gates + partner/affiliate API state + per-user-key coverage."""
    from app.database import SessionLocal
    from app.models import User
    db = SessionLocal()
    try:
        user = _get_user_by_uid(uid, db)
        if not user or not getattr(user, "is_admin", False):
            raise HTTPException(status_code=403, detail="Admin only")
        try:
            with_keys = (
                db.query(User)
                .filter(User.bitunix_api_key.isnot(None))
                .filter(User.bitunix_api_secret.isnot(None))
                .count()
            )
            with_uid = (
                db.query(User)
                .filter(User.bitunix_uid.isnot(None))
                .count()
            )
        except Exception:
            with_keys, with_uid = -1, -1
    finally:
        db.close()
    try:
        from app.services.bitunix_executor import status as _bx_status
        from app.services.bitunix_partner  import status as _bp_status
        return JSONResponse({
            "executor": {**_bx_status(), "users_with_keys_on_file": with_keys, "users_with_bitunix_uid": with_uid},
            "partner":  _bp_status(),
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/admin/bitunix/affiliates/refresh")
async def admin_bitunix_affiliates_refresh(uid: str = Query(...)):
    """Force-refresh the cached affiliate-UID roster from the partner API."""
    from app.database import SessionLocal
    db = SessionLocal()
    try:
        user = _get_user_by_uid(uid, db)
        if not user or not getattr(user, "is_admin", False):
            raise HTTPException(status_code=403, detail="Admin only")
    finally:
        db.close()
    try:
        from app.services.bitunix_partner import refresh_affiliate_uids
        uids = await refresh_affiliate_uids(force=True)
        return JSONResponse({"count": len(uids), "sample": list(uids)[:10]})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/admin/ai-generator/promote")
async def admin_ai_generator_promote(uid: str = Query(...)):
    """Manually trigger one promotion / unpublish pass over AI Curator strategies."""
    from app.database import SessionLocal
    db = SessionLocal()
    try:
        user = _get_user_by_uid(uid, db)
        if not user or not getattr(user, "is_admin", False):
            raise HTTPException(status_code=403, detail="Admin only")
    finally:
        db.close()
    try:
        from app.services.ai_strategy_generator import run_promotion_cycle
        loop = asyncio.get_event_loop()
        summary = await loop.run_in_executor(None, run_promotion_cycle)
        return JSONResponse(summary)
    except Exception as e:
        logger.error(f"[AIGen] manual promote failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/admin/ai-generator/run")
async def admin_ai_generator_run(uid: str = Query(...), n: int = Query(3)):
    """Manually trigger one cycle of N specs. Returns when the cycle finishes."""
    from app.database import SessionLocal
    db = SessionLocal()
    try:
        user = _get_user_by_uid(uid, db)
        if not user or not getattr(user, "is_admin", False):
            raise HTTPException(status_code=403, detail="Admin only")
    finally:
        db.close()
    try:
        from app.services.ai_strategy_generator import run_one_cycle
        n_clamped = max(1, min(int(n), 12))
        summary = await run_one_cycle(per_cycle=n_clamped)
        return JSONResponse(summary)
    except Exception as e:
        logger.error(f"[AIGen] manual run failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ── Affiliate program endpoints ─────────────────────────────────────────────
# Mobile users can apply to become an affiliate, then earn a share of every
# subscription + trading-fee referral they bring in. Affiliate state is keyed
# off the existing `users.referral_code` column (auto-minted on first request
# if missing). Application metadata lives in `affiliate_applications`.

REFERRAL_BASE_URL = os.environ.get("REFERRAL_BASE_URL", "https://tradehub.markets")
DEFAULT_SUB_SHARE_PCT = 30.0
DEFAULT_FEE_SHARE_PCT = 20.0


def _ensure_referral_code(user, db) -> str:
    """Mint a referral code for the user if they don't have one yet."""
    if user.referral_code:
        return user.referral_code
    import random as _r
    chars = "ABCDEFGHJKMNPQRSTUVWXYZ23456789"
    for _ in range(40):
        code = "".join(_r.choices(chars, k=6))
        if not db.query(User).filter(User.referral_code == code).first():
            user.referral_code = code
            db.commit()
            return code
    raise HTTPException(status_code=500, detail="Could not mint referral code")


@app.get("/api/affiliates/me")
async def api_affiliates_me(request: Request):
    """Return the signed-in user's affiliate status, application, and link.

    Response shape:
      {
        "is_affiliate": bool,
        "status": "none" | "pending" | "approved" | "rejected",
        "referral_code": str | null,
        "referral_url": str | null,
        "sub_share_pct": float,
        "fee_share_pct": float,
        "application": { ...latest application... } | null,
        "stats": { "referrals": int, "earnings_usd": float }
      }
    """
    import sqlalchemy as sa
    uid = request.query_params.get("uid", "").strip().upper()
    if not uid:
        raise HTTPException(status_code=401, detail="auth required")
    db = SessionLocal()
    try:
        user = _get_user_by_uid(uid, db)
        if not user:
            raise HTTPException(status_code=403, detail="invalid UID")

        row = None
        with engine.connect() as conn:
            r = conn.execute(sa.text(
                "SELECT id, telegram, twitter, instagram, youtube, tiktok, website, "
                "bio, plan, status, sub_share_pct, fee_share_pct, created_at, "
                "reviewed_at, reviewer_note "
                "FROM affiliate_applications WHERE user_id = :uid_pk"
            ), {"uid_pk": user.id}).first()
            if r:
                row = dict(r._mapping)
            # Count how many users used this referral code (if any)
            ref_count = 0
            if user.referral_code:
                cnt = conn.execute(sa.text(
                    "SELECT COUNT(*) FROM users WHERE referred_by = :code"
                ), {"code": user.referral_code}).scalar() or 0
                ref_count = int(cnt)

        is_approved = bool(row and row["status"] == "approved")
        # Anyone with an application (even pending) gets a shareable referral
        # link — full commission only kicks in on approval, but the link is
        # live so creators can start warming up their audience.
        ref_code = user.referral_code if row else None
        ref_url = f"{REFERRAL_BASE_URL}/r/{ref_code}" if (row and ref_code) else None
        sub_pct = float((row or {}).get("sub_share_pct") or DEFAULT_SUB_SHARE_PCT)
        fee_pct = float((row or {}).get("fee_share_pct") or DEFAULT_FEE_SHARE_PCT)

        # Serialise app row for JSON
        app_payload = None
        if row:
            app_payload = {
                k: (v.isoformat() if hasattr(v, "isoformat") else v)
                for k, v in row.items()
            }

        return JSONResponse({
            "is_affiliate": is_approved,
            "status": (row["status"] if row else "none"),
            "referral_code": ref_code,
            "referral_url": ref_url,
            "sub_share_pct": sub_pct,
            "fee_share_pct": fee_pct,
            "application": app_payload,
            "stats": {
                "referrals": ref_count,
                "earnings_usd": 0.0,  # placeholder until commission ledger exists
            },
        })
    finally:
        db.close()


@app.post("/api/affiliates/apply")
async def api_affiliates_apply(request: Request):
    """Submit (or re-submit) an affiliate application.

    Body fields: telegram (required), twitter, instagram, youtube, tiktok,
    website, bio (required), plan (required) — bio is a short "about you"
    blurb, plan is "how do you intend to promote".

    On success the user receives a referral_code immediately so they can
    start sharing while their application is being reviewed; full affiliate
    earnings only kick in once status flips to 'approved' by an admin.
    """
    import sqlalchemy as sa
    uid = request.query_params.get("uid", "").strip().upper()
    if not uid:
        raise HTTPException(status_code=401, detail="auth required")
    try:
        body = await request.json()
    except Exception:
        body = {}

    def _clean(s, max_len=200):
        v = (body.get(s) or "")
        if not isinstance(v, str): v = str(v)
        return v.strip()[:max_len]

    telegram = _clean("telegram", 64)
    bio      = _clean("bio", 1000)
    plan     = _clean("plan", 1500)
    if not telegram:
        raise HTTPException(status_code=400, detail="telegram handle is required")
    if len(bio) < 20:
        raise HTTPException(status_code=400, detail="please write at least a short bio (20+ chars)")
    if len(plan) < 30:
        raise HTTPException(status_code=400, detail="please describe your promotion plan in a bit more detail (30+ chars)")

    twitter   = _clean("twitter", 200)
    instagram = _clean("instagram", 200)
    youtube   = _clean("youtube", 200)
    tiktok    = _clean("tiktok", 200)
    website   = _clean("website", 200)

    db = SessionLocal()
    try:
        user = _get_user_by_uid(uid, db)
        if not user:
            raise HTTPException(status_code=403, detail="invalid UID")
        # Mint referral code so the user can start sharing right away.
        ref_code = _ensure_referral_code(user, db)

        # UPSERT application (one-per-user). Re-submitting before approval
        # overwrites the previous application; once approved we lock it.
        with engine.begin() as conn:
            # Atomic UPSERT with a WHERE-guard on the UPDATE branch — we never
            # touch a row that's already been approved, even if an admin flips
            # the status between this request and a concurrent one.
            res = conn.execute(sa.text("""
                INSERT INTO affiliate_applications
                  (user_id, telegram, twitter, instagram, youtube, tiktok,
                   website, bio, plan, status, sub_share_pct, fee_share_pct,
                   created_at)
                VALUES
                  (:u, :tg, :tw, :ig, :yt, :tt, :web, :bio, :plan,
                   'pending', :sp, :fp, NOW())
                ON CONFLICT (user_id) DO UPDATE SET
                  telegram = EXCLUDED.telegram,
                  twitter = EXCLUDED.twitter,
                  instagram = EXCLUDED.instagram,
                  youtube = EXCLUDED.youtube,
                  tiktok = EXCLUDED.tiktok,
                  website = EXCLUDED.website,
                  bio = EXCLUDED.bio,
                  plan = EXCLUDED.plan,
                  status = 'pending',
                  created_at = NOW(),
                  reviewed_at = NULL,
                  reviewer_note = NULL
                WHERE affiliate_applications.status <> 'approved'
                RETURNING id
            """), {
                "u": user.id, "tg": telegram, "tw": twitter, "ig": instagram,
                "yt": youtube, "tt": tiktok, "web": website, "bio": bio,
                "plan": plan, "sp": DEFAULT_SUB_SHARE_PCT, "fp": DEFAULT_FEE_SHARE_PCT,
            })
            # If RETURNING produced no row, the conflict target hit an already-
            # approved application and the WHERE-guard skipped the update.
            if res.first() is None:
                raise HTTPException(
                    status_code=409,
                    detail="You're already an approved affiliate.",
                )

        return JSONResponse({
            "ok": True,
            "status": "pending",
            "referral_code": ref_code,
            "referral_url": f"{REFERRAL_BASE_URL}/r/{ref_code}",
            "message": "Application received! We'll review and reach out on Telegram within 48h.",
        })
    finally:
        db.close()


if __name__ == "__main__":
    import uvicorn, time, logging
    _log = logging.getLogger("startup")
    while True:
        try:
            _log.info("Starting uvicorn on port 5000 with 2 workers...")
            uvicorn.run(
                "strategy_portal_server:app",
                host="0.0.0.0",
                port=5000,
                workers=2,
                loop="asyncio",
            )
        except Exception as _e:
            _log.error(f"Server crashed: {_e} — restarting in 3s")
            time.sleep(3)
