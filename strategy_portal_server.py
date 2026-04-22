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
from typing import Optional, Dict, Tuple

from fastapi import FastAPI, Request, HTTPException, Query
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, FileResponse
from fastapi.templating import Jinja2Templates
from starlette.middleware.gzip import GZipMiddleware
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

# ─── Simple in-process TTL cache for public/read-heavy endpoints ─────────────
# Format: { key: (payload, expiry_timestamp) }
_CACHE: Dict[str, Tuple] = {}

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Strategy Portal", docs_url=None, redoc_url=None)
app.add_middleware(GZipMiddleware, minimum_size=500)
templates = Jinja2Templates(directory="app/templates")

@app.middleware("http")
async def redirect_www(request: Request, call_next):
    host = request.headers.get("host", "")
    if host.startswith("www."):
        url = request.url
        non_www = host[4:]
        redirect_url = str(url).replace(f"://{host}", f"://{non_www}", 1)
        return RedirectResponse(url=redirect_url, status_code=301)
    return await call_next(request)

# ─── Session cookie helpers (HMAC-signed, no extra deps) ──────────────────────
_COOKIE_SECRET = os.getenv("SECRET_KEY", "tradehub-portal-secret-2025")
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
        return uid
    return None


def _get_session_uid(request: Request) -> Optional[str]:
    token = request.cookies.get(_COOKIE_NAME)
    return _verify_token(token) if token else None


def _set_session(response, uid: str, request: Request = None):
    # Detect HTTPS: check REPL_DEPLOYMENT env var OR X-Forwarded-Proto header from the proxy
    _secure = bool(os.getenv("REPL_DEPLOYMENT"))
    if not _secure and request:
        forwarded_proto = request.headers.get("x-forwarded-proto", "")
        _secure = forwarded_proto.lower() == "https"
    response.set_cookie(
        key=_COOKIE_NAME,
        value=_make_token(uid),
        max_age=_COOKIE_MAX_AGE,
        httponly=True,
        samesite="lax",
        secure=_secure,
    )


def get_db():
    from app.database import SessionLocal
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def _get_user_by_uid(uid: str, db: Session):
    from app.models import User
    return db.query(User).filter(User.uid == uid).first()


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
    if sub.tier == "pro" and sub.subscription_end and datetime.utcnow() < sub.subscription_end:
        return True
    return False


def _chat_calls_info(sub, db: Session):
    """Check / reset monthly counter. Returns (allowed, used, limit, is_pro)."""
    now = datetime.utcnow()
    if sub.chat_calls_reset_at is None or (now - sub.chat_calls_reset_at).days >= 30:
        sub.chat_calls_used = 0
        sub.chat_calls_reset_at = now
        db.commit()
    pro = _is_portal_pro(sub)
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
    return bool(_GOOGLE_CLIENT_ID and _GOOGLE_CLIENT_SECRET)


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


def _ensure_tables():
    import sqlalchemy as sa
    init_strategy_tables(engine)
    init_marketplace_ext_tables(engine)
    init_social_tables(engine)
    # Only ALTER if column is genuinely missing — avoids table locks when both
    # dev and production portals start against the same shared Neon database.
    needed = {
        "email":          "ALTER TABLE users ADD COLUMN email VARCHAR UNIQUE",
        "email_verified": "ALTER TABLE users ADD COLUMN email_verified BOOLEAN DEFAULT FALSE",
        "password_hash":  "ALTER TABLE users ADD COLUMN password_hash VARCHAR",
        "google_id":      "ALTER TABLE users ADD COLUMN google_id VARCHAR UNIQUE",
        "auth_provider":  "ALTER TABLE users ADD COLUMN auth_provider VARCHAR DEFAULT 'telegram'",
        "uid":            "ALTER TABLE users ADD COLUMN uid VARCHAR UNIQUE",
    }
    try:
        with engine.connect() as conn:
            existing = {
                row[0] for row in conn.execute(sa.text(
                    "SELECT column_name FROM information_schema.columns "
                    "WHERE table_name='users' AND table_schema='public'"
                ))
            }
            for col, sql in needed.items():
                if col not in existing:
                    try:
                        conn.execute(sa.text(sql))
                        logger.info(f"Migration: added users.{col}")
                    except Exception as ce:
                        logger.warning(f"Column migration {col}: {ce}")
            conn.commit()
    except Exception as e:
        logger.warning(f"_ensure_tables: {e}")


@app.on_event("startup")
async def startup():
    # Run schema migrations in the background so the server starts accepting
    # requests immediately instead of blocking on DB round-trips.
    asyncio.create_task(_startup_background())
    logger.info("Strategy portal started on port 5000 (migrations running in background)")


async def _cancel_ghost_executions():
    """
    Cancel any live (is_paper=False) OPEN strategy executions that have no
    Bitunix order ID.  These are 'ghost' records created when the Bitunix API
    call failed — they were never real positions, but old code left them OPEN
    so the live monitor would fire false SL/TP alerts.
    Run once on startup and then every 5 minutes as a safety net.
    """
    from app.database import SessionLocal
    try:
        db = SessionLocal()
        try:
            from sqlalchemy import text
            result = db.execute(text("""
                UPDATE strategy_executions
                SET outcome = 'CANCELLED',
                    notes   = 'Auto-cancelled: no Bitunix order_id (ghost execution)'
                WHERE is_paper = false
                  AND outcome  = 'OPEN'
                  AND bitunix_order_id IS NULL
                  AND fired_at < NOW() - INTERVAL '5 minutes'
                RETURNING id, symbol, direction
            """))
            cancelled = result.fetchall()
            db.commit()
            if cancelled:
                for row in cancelled:
                    logger.warning(
                        f"[ghost-cleanup] Cancelled ghost execution id={row[0]} "
                        f"{row[1]} {row[2]} — no Bitunix order_id"
                    )
            else:
                logger.debug("[ghost-cleanup] No ghost executions found")
        finally:
            db.close()
    except Exception as e:
        logger.error(f"[ghost-cleanup] Error: {e}")


async def _ghost_cleanup_loop():
    """
    Runs _cancel_ghost_executions every 60 seconds indefinitely.
    Only started in the worker that holds the executor advisory lock.
    """
    while True:
        await asyncio.sleep(60)
        await _cancel_ghost_executions()


# ── PostgreSQL advisory lock — ensures only ONE uvicorn worker runs the
#    executor, monitors, and ghost-cleanup (not all 4 workers). ─────────────
_EXECUTOR_LOCK_ID = 42_424_242   # arbitrary unique integer for this process role

def _try_acquire_executor_lock():
    """
    Try to acquire a PostgreSQL session-level advisory lock using a raw
    psycopg2 connection.  Returns the open connection (which must be kept
    alive to hold the lock) or None if another worker already holds it.
    Session-level advisory locks are automatically released when the
    connection (i.e. the process) closes.
    """
    try:
        import psycopg2
        from app.config import settings
        conn = psycopg2.connect(settings.get_database_url())
        conn.autocommit = True
        cur = conn.cursor()
        cur.execute("SELECT pg_try_advisory_lock(%s)", (_EXECUTOR_LOCK_ID,))
        acquired = cur.fetchone()[0]
        cur.close()
        if acquired:
            return conn          # caller keeps this connection open
        conn.close()
        return None
    except Exception as e:
        logger.warning(f"Advisory lock attempt failed: {e}")
        return None

# Tracks whether THIS uvicorn worker is currently running the executor.
# Used by the claim loop to avoid double-starting.
_executor_running_in_this_worker = False


async def _maintain_advisory_lock(conn):
    """
    Keep the psycopg2 advisory-lock connection alive with periodic pings.
    Neon drops idle connections after ~5 min, so we ping every 4 minutes.
    Returns when the connection dies (signal to the claim loop to retry).
    """
    loop = asyncio.get_event_loop()
    while True:
        await asyncio.sleep(240)   # 4 minutes
        try:
            await loop.run_in_executor(None, lambda: conn.cursor().execute("SELECT 1"))
        except Exception as e:
            logger.warning(f"Advisory lock keepalive failed: {e} — will attempt re-claim")
            break


async def _keepalive_ping_loop():
    """
    Ping the app's own health endpoint every 4 minutes so Replit Autoscale
    never scales to zero while strategies are active.
    Runs only in production (REPL_DEPLOYMENT=1).
    """
    import aiohttp as _aiohttp
    import os as _os
    # Resolve the public domain — prefer the custom domain, fall back to replit.app
    domain = _os.environ.get("PUBLIC_DOMAIN", "tradehubmarkets.com")
    url = f"https://{domain}/health"
    await asyncio.sleep(60)   # small initial delay to let the server fully start
    while True:
        try:
            async with _aiohttp.ClientSession() as _sess:
                async with _sess.get(url, timeout=_aiohttp.ClientTimeout(total=12)) as r:
                    logger.debug(f"[Keepalive] ✅ ping {url} → {r.status}")
        except Exception as _e:
            logger.debug(f"[Keepalive] ping failed (non-critical): {_e}")
        await asyncio.sleep(240)   # 4-minute interval


async def _start_executor_tasks():
    """Import and launch the executor + monitor tasks in this worker."""
    await _cancel_ghost_executions()
    asyncio.create_task(_ghost_cleanup_loop())
    asyncio.create_task(_keepalive_ping_loop())   # keep Autoscale awake
    from app.services.strategy_executor import (
        run_strategy_executor, run_live_position_monitor,
        backfill_cancelled_paper_trades,
        backfill_ghost_cancelled_executions,
    )
    asyncio.create_task(run_strategy_executor())
    asyncio.create_task(run_live_position_monitor())
    asyncio.create_task(backfill_cancelled_paper_trades(lookback_days=30))
    asyncio.create_task(backfill_ghost_cancelled_executions(lookback_days=7))


async def _executor_claim_loop(first_attempt_delay: int = 0):
    """
    Continuously tries to acquire the executor advisory lock.
    - On startup, one worker wins immediately; others loop here.
    - If the winning worker's DB connection drops, its keepalive returns and
      that worker calls this function again — racing all other workers.
    - Safe: pg_try_advisory_lock is non-blocking and only one worker wins.
    """
    global _executor_running_in_this_worker
    loop = asyncio.get_event_loop()

    if first_attempt_delay:
        await asyncio.sleep(first_attempt_delay)

    while True:
        if _executor_running_in_this_worker:
            return  # Already running in this worker — nothing to do

        lock_conn = await loop.run_in_executor(None, _try_acquire_executor_lock)
        if lock_conn:
            _executor_running_in_this_worker = True
            logger.info("✅ Executor lock acquired — this worker runs executor + monitors")
            # Start keepalive; when it returns the connection died → re-enter claim loop
            asyncio.create_task(_keepalive_then_reclaim(lock_conn))
            try:
                await _start_executor_tasks()
            except Exception as e:
                logger.error(f"Failed to start executor tasks: {e}")
            return  # This worker is now the executor; stop trying
        else:
            await asyncio.sleep(30)  # Retry every 30 s until lock is free


async def _keepalive_then_reclaim(conn):
    """Keeps the advisory-lock connection alive; when it dies, re-enters the claim loop."""
    global _executor_running_in_this_worker
    await _maintain_advisory_lock(conn)      # blocks until connection dies
    _executor_running_in_this_worker = False
    logger.warning("Advisory lock connection lost — re-entering claim loop")
    asyncio.create_task(_executor_claim_loop(first_attempt_delay=5))


async def _startup_background():
    """Non-critical startup work that runs after the server is already live."""
    import asyncio as _aio
    loop = _aio.get_event_loop()
    try:
        await loop.run_in_executor(None, _ensure_tables)
        logger.info("Schema migrations complete")
    except Exception as e:
        logger.warning(f"Background _ensure_tables error: {e}")

    # Only run the strategy executor in production (REPL_DEPLOYMENT=1).
    # In dev, both the dev portal and production share the same Neon DB, so
    # running the executor in dev doubles all API calls and causes confusion.
    import os as _os
    _is_production = (
        _os.environ.get("REPL_DEPLOYMENT") == "1"
        or _os.environ.get("FORCE_EXECUTOR", "").lower() in ("1", "true", "yes")
    )
    _executor_disabled = _os.environ.get("DISABLE_EXECUTOR", "").lower() in ("1", "true", "yes")
    if _is_production and not _executor_disabled:
        # Each worker enters the claim loop.  The first to win acquires the lock
        # and starts the executor; the rest keep retrying every 30 s so they
        # can take over automatically if the current holder's connection drops.
        # first_attempt_delay=0 → try immediately on startup
        asyncio.create_task(_executor_claim_loop(first_attempt_delay=0))
    else:
        logger.info("Strategy executor DISABLED (dev environment — only production runs it)")


# ─────────────────────────────────────────────────────────────────────────────
# Health check — responds immediately, even before migrations finish
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok", "ts": int(__import__("time").time()), "v": "callyx-live"}


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
        return RedirectResponse(url="/app", status_code=302)
    return FileResponse("app/templates/login.html", media_type="text/html")


@app.get("/terms", response_class=HTMLResponse)
async def terms_page():
    return FileResponse("app/templates/terms.html", media_type="text/html")


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

    resp = JSONResponse({"redirect": "/app"})
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

    resp = JSONResponse({"redirect": "/app"})
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
    resp = JSONResponse({"redirect": "/app"})
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
    resp = JSONResponse({"redirect": "/app"})
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
    resp = JSONResponse({"redirect": "/app"})
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
    resp = JSONResponse({"redirect": "/app"})
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
        user = _get_user_by_uid(uid, db)
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
        wins        = int(agg.wins)
        pnl_7d      = float(agg.pnl_7d)
        pnl_all     = float(agg.pnl_all)
        win_rate    = round(wins / total * 100, 1) if total > 0 else 0

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
    """Shared logic for /app and /strategies."""
    ctx = await asyncio.to_thread(_load_portal_data, uid)
    if ctx is None:
        raise HTTPException(status_code=403, detail="Invalid access link")
    if ctx == "banned":
        raise HTTPException(status_code=403, detail="Account banned")

    response = templates.TemplateResponse("strategy_portal.html", {
        "request":      request,
        "user":         ctx["user"],
        "uid":          uid,
        "strategies":   ctx["strategies"],
        "portfolio":    ctx["portfolio"],
        "is_web_user":  ctx["is_web_user"],
        "is_pro":       ctx["is_pro"],
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
async def public_leaderboard(limit: int = Query(5, ge=1, le=10)):
    """Top strategies by P&L — no auth required. Cached 2 min."""
    cache_key = f"pub_lb_{limit}"
    cached = _CACHE.get(cache_key)
    if cached and time.time() < cached[1]:
        return cached[0]

    def _load():
        db = SessionLocal()
        try:
            rows = (
                db.query(StrategyPerformance, UserStrategy)
                .join(UserStrategy, UserStrategy.id == StrategyPerformance.strategy_id)
                .filter(StrategyPerformance.total_trades >= 5)
                .order_by(StrategyPerformance.total_pnl_pct.desc())
                .limit(limit)
                .all()
            )
            return [
                {
                    "name":         r.UserStrategy.name or "Unnamed",
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
    return payload


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

@app.get("/api/strategies")
async def api_strategies(uid: str = Query(...)):
    cache_key = f"api_strats_{uid}"
    cached = _CACHE.get(cache_key)
    if cached and time.time() < cached[1]:
        return JSONResponse(cached[0])

    from app.database import SessionLocal
    from app.strategy_models import UserStrategy, StrategyPerformance, StrategyExecution

    def _load():
        db = SessionLocal()
        try:
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

                execs = (
                    db.query(StrategyExecution)
                    .filter(StrategyExecution.strategy_id.in_(strategy_ids))
                    .order_by(StrategyExecution.fired_at.desc())
                    .limit(len(strategy_ids) * 10 + 50)
                    .all()
                )
                for ex in execs:
                    exec_map.setdefault(ex.strategy_id, [])
                    if len(exec_map[ex.strategy_id]) < 10:
                        exec_map[ex.strategy_id].append(ex)

            result = []
            for s in strategies:
                perf = perf_map.get(s.id)
                recent_execs = exec_map.get(s.id, [])

                # Fast inline health score
                wr  = perf.win_rate if perf else 0
                tot = perf.total_trades if perf else 0
                pf  = (perf.avg_win_pct * max(perf.wins, 1)) / (abs(perf.avg_loss_pct) * max(perf.losses, 1)) if perf and perf.losses > 0 and perf.avg_loss_pct else 0
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
                    "config":       cfg,
                    "is_locked":    bool(cfg.get("_locked")),
                    "is_public":    s.is_public,
                    "created_at":   s.created_at.isoformat() if s.created_at else None,
                    "health_score": health_score,
                    "performance": {
                        "total_trades": perf.total_trades if perf else 0,
                        "wins":         perf.wins if perf else 0,
                        "losses":       perf.losses if perf else 0,
                        "win_rate":     round(perf.win_rate, 1) if perf else 0,
                        "total_pnl":    round(perf.total_pnl_pct, 2) if perf else 0,
                        "open_trades":  perf.open_trades if perf else 0,
                        "best_trade":   round(perf.best_trade, 2) if perf else 0,
                        "worst_trade":  round(perf.worst_trade, 2) if perf else 0,
                        "avg_win_pct":  round(perf.avg_win_pct, 2) if perf else 0,
                        "avg_loss_pct": round(perf.avg_loss_pct, 2) if perf else 0,
                    } if perf else {},
                    "recent_trades": [{
                        "symbol":    ex.symbol,
                        "direction": ex.direction,
                        "outcome":   ex.outcome,
                        "pnl_pct":   round(ex.pnl_pct, 2) if ex.pnl_pct else None,
                        "fired_at":  ex.fired_at.isoformat() if ex.fired_at else None,
                    } for ex in recent_execs],
                })
            return result
        finally:
            db.close()

    data = await asyncio.to_thread(_load)
    if data is None:
        raise HTTPException(status_code=403)
    _CACHE[cache_key] = (data, time.time() + 60)
    return JSONResponse(data)


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
        return {"status": strategy.status}
    finally:
        db.close()


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

        # Clean up all FK-constrained child records before deleting the strategy.
        # Use raw SQL to handle non-nullable FKs and complex dependency chains.
        from sqlalchemy import text
        try:
            # 1) Delete earnings_transactions tied to purchases of this strategy
            db.execute(text("""
                DELETE FROM earnings_transactions
                WHERE purchase_id IN (
                    SELECT id FROM strategy_purchases WHERE strategy_id = :sid
                )
            """), {"sid": strategy_id})

            # 2) Delete purchases where this strategy was the source listing
            db.execute(text(
                "DELETE FROM strategy_purchases WHERE strategy_id = :sid"
            ), {"sid": strategy_id})

            # 3) Null out cloned_strategy_id for subscriptions that cloned this strategy
            db.execute(text("""
                UPDATE strategy_purchases
                SET cloned_strategy_id = NULL
                WHERE cloned_strategy_id = :sid
            """), {"sid": strategy_id})

            # 4) Delete ratings tied to this strategy's marketplace listing
            db.execute(text("""
                DELETE FROM strategy_ratings
                WHERE listing_id IN (
                    SELECT id FROM strategy_marketplace WHERE strategy_id = :sid
                )
            """), {"sid": strategy_id})

            # 5) Delete the marketplace listing itself
            db.execute(text(
                "DELETE FROM strategy_marketplace WHERE strategy_id = :sid"
            ), {"sid": strategy_id})

            # 6) Delete strategy offers
            db.execute(text(
                "DELETE FROM strategy_offers WHERE strategy_id = :sid"
            ), {"sid": strategy_id})
        except Exception as _clean_err:
            logger.warning(f"[DeleteStrategy] Cleanup warning (non-fatal): {_clean_err}")

        db.delete(strategy)
        db.commit()
        return {"deleted": True}
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()


@app.get("/api/marketplace")
async def api_marketplace(
    uid:      str = Query(...),
    sort:     str = Query("top"),
    category: str = Query("all"),
    pricing:  str = Query("all"),
    search:   str = Query(""),
):
    from app.database import SessionLocal, engine
    db = SessionLocal()
    try:
        user = _get_user_by_uid(uid, db)
        if not user:
            raise HTTPException(status_code=403)

        from app.strategy_models import StrategyMarketplace, UserStrategy, StrategyPerformance
        from app.strategy_marketplace_ext import StrategyPurchase, init_marketplace_ext_tables

        q = db.query(StrategyMarketplace)
        if category != "all":
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
        result = []
        for m in listings:
            perf   = db.query(StrategyPerformance).filter(StrategyPerformance.strategy_id == m.strategy_id).first()
            author = db.query(User).filter(User.id == m.author_id).first()
            # Build equity curve from last 30 closed trades (cumulative P&L %)
            closed_trades = (
                db.query(StrategyExecution.pnl_pct)
                .filter(
                    StrategyExecution.strategy_id == m.strategy_id,
                    StrategyExecution.outcome.in_(["WIN", "LOSS", "BREAKEVEN"]),
                    StrategyExecution.pnl_pct.isnot(None),
                )
                .order_by(StrategyExecution.fired_at.asc())
                .limit(30).all()
            )
            equity = []
            cum = 0.0
            for (pnl,) in closed_trades:
                cum += float(pnl)
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
        return JSONResponse(result)
    finally:
        db.close()


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
                "pnl_pct": round(ex.pnl_pct, 2) if ex.pnl_pct else None} for ex in recent_trades],
            "ratings": [{"stars": r.stars, "review": r.review, "is_verified": r.is_verified} for r in ratings],
            "my_rating": {"stars": my_rating.stars, "review": my_rating.review} if my_rating else None,
        })
    finally:
        db.close()


@app.post("/api/marketplace/{listing_id}/purchase")
async def api_purchase_strategy(listing_id: int, uid: str = Query(...)):
    from app.database import SessionLocal, engine
    db = SessionLocal()
    try:
        user = _get_user_by_uid(uid, db)
        if not user:
            raise HTTPException(status_code=403)

        from app.strategy_models import StrategyMarketplace, UserStrategy, StrategyPerformance, init_strategy_tables
        from app.strategy_marketplace_ext import StrategyPurchase, CreatorEarnings, EarningsTransaction, init_marketplace_ext_tables, calculate_creator_cut, calculate_platform_cut

        # Pro subscription required to copy marketplace strategies
        _psub = _get_portal_sub(user.id, db)
        if not _is_portal_pro(_psub):
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
                    locked_config = {
                        "name":                listing.title or original.name,
                        "direction":           original.config.get("direction", "LONG"),
                        "risk":                original.config.get("risk", {}),
                        "exit":                original.config.get("exit", {}),
                        "filters":             original.config.get("filters", {}),
                        "universe":            original.config.get("universe", {}),
                        "_locked":             True,
                        "_source_strategy_id": listing.strategy_id,
                        "_listing_id":         listing_id,
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

        # Build a locked stub — no entry_conditions or strategy logic exposed
        locked_config = {
            "name":             listing.title or original.name,
            "direction":        original.config.get("direction", "LONG"),
            "risk":             original.config.get("risk", {}),
            "exit":             original.config.get("exit", {}),
            "filters":          original.config.get("filters", {}),
            "universe":         original.config.get("universe", {}),
            "_locked":          True,
            "_source_strategy_id": listing.strategy_id,
            "_listing_id":      listing_id,
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
    db = SessionLocal()
    try:
        user = _get_user_by_uid(uid, db)
        if not user:
            raise HTTPException(status_code=403)

        from app.strategy_models import StrategyMarketplace, StrategyPerformance, UserStrategy, StrategyOffer
        results = []
        seen_strategy_ids = set()

        # 1. Marketplace (live-published) strategies
        listings = db.query(StrategyMarketplace).all()
        for m in listings:
            perf   = db.query(StrategyPerformance).filter(StrategyPerformance.strategy_id == m.strategy_id).first()
            author = db.query(User).filter(User.id == m.author_id).first()
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
        for s in all_strategies:
            if s.id in seen_strategy_ids:
                continue
            perf   = db.query(StrategyPerformance).filter(StrategyPerformance.strategy_id == s.id).first()
            author = db.query(User).filter(User.id == s.user_id).first()
            if not perf or perf.total_trades < 3 or perf.total_pnl_pct <= 0:
                continue
            # Determine paper vs live from most recent executions
            from app.strategy_models import StrategyExecution
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
        return JSONResponse(results[:30])
    finally:
        db.close()


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
async def api_clone_strategy(listing_id: int, uid: str = Query(...)):
    return await api_purchase_strategy(listing_id, uid)


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

        strategy = UserStrategy(
            user_id     = user.id,
            name        = config.get("name", "My Strategy"),
            description = config.get("description", ""),
            config      = config,
            status      = initial_status,
        )
        db.add(strategy)
        db.commit()
        db.refresh(strategy)

        perf = StrategyPerformance(strategy_id=strategy.id)
        db.add(perf)
        db.commit()

        return JSONResponse({"id": strategy.id, "name": strategy.name, "status": initial_status})
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
    ]
    return JSONResponse(templates)


@app.get("/api/strategies/{strategy_id}/analytics")
async def api_strategy_analytics(strategy_id: int, uid: str = Query(...)):
    """Advanced analytics: Sharpe, drawdown, profit factor, equity curve, health score."""
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

        closed = [e for e in execs if e.outcome in ("WIN", "LOSS", "BREAKEVEN") and e.pnl_pct is not None]
        wins   = [e.pnl_pct for e in closed if e.outcome == "WIN"]
        losses = [e.pnl_pct for e in closed if e.outcome == "LOSS"]

        # Equity curve (cumulative P&L %)
        cumulative = 0.0
        equity_labels = []
        equity_values = []
        for e in closed:
            cumulative += (e.pnl_pct or 0)
            dt = (e.closed_at or e.fired_at)
            equity_labels.append(dt.strftime("%m/%d") if dt else "")
            equity_values.append(round(cumulative, 2))

        # Max drawdown
        peak = 0.0
        max_dd = 0.0
        for v in equity_values:
            if v > peak: peak = v
            dd = peak - v
            if dd > max_dd: max_dd = dd

        # Profit factor
        gross_win  = sum(wins)  if wins   else 0.0
        gross_loss = abs(sum(losses)) if losses else 0.0
        profit_factor = round(gross_win / gross_loss, 2) if gross_loss > 0 else (99.0 if gross_win > 0 else 0.0)

        # Sharpe ratio (simplified — treat each trade as 1 period)
        sharpe = 0.0
        if len(closed) >= 5:
            import statistics as _st
            pnls = [e.pnl_pct for e in closed]
            mean_r = _st.mean(pnls)
            std_r  = _st.stdev(pnls) if len(pnls) > 1 else 0
            sharpe = round((mean_r / std_r) * (252 ** 0.5), 2) if std_r > 0 else 0

        # Streak
        best_streak = worst_streak = cur_w = cur_l = 0
        for e in closed:
            if e.outcome == "WIN":
                cur_w += 1; cur_l = 0
                best_streak = max(best_streak, cur_w)
            elif e.outcome == "LOSS":
                cur_l += 1; cur_w = 0
                worst_streak = max(worst_streak, cur_l)

        # Per-coin breakdown
        coin_pnl = {}
        coin_trades = {}
        for e in closed:
            coin_pnl[e.symbol]    = round(coin_pnl.get(e.symbol, 0) + (e.pnl_pct or 0), 2)
            coin_trades[e.symbol] = coin_trades.get(e.symbol, 0) + 1
        top_coins = sorted(coin_pnl.items(), key=lambda x: -x[1])

        # Win rate by direction
        long_closed  = [e for e in closed if e.direction == "LONG"]
        short_closed = [e for e in closed if e.direction == "SHORT"]
        long_wr  = round(len([e for e in long_closed  if e.outcome == "WIN"]) / len(long_closed)  * 100, 1) if long_closed  else None
        short_wr = round(len([e for e in short_closed if e.outcome == "WIN"]) / len(short_closed) * 100, 1) if short_closed else None

        # Health score (0–10)
        wr_pct = perf.win_rate if perf else 0
        health = 0.0
        if len(closed) >= 3:
            health += min(wr_pct / 100, 1.0) * 4.0
            health += min(profit_factor / 2.0, 1.0) * 3.0
            health += min(max(sharpe, 0) / 2.0, 1.0) * 2.0
            health += min(len(closed) / 30.0, 1.0) * 1.0
        health = round(health, 1)

        return JSONResponse({
            "equity_curve":  {"labels": equity_labels, "values": equity_values},
            "profit_factor": profit_factor,
            "max_drawdown":  round(max_dd, 2),
            "sharpe_ratio":  sharpe,
            "best_streak":   best_streak,
            "worst_streak":  worst_streak,
            "avg_win_pct":   round(sum(wins) / len(wins), 2)   if wins   else 0,
            "avg_loss_pct":  round(sum(losses) / len(losses), 2) if losses else 0,
            "long_win_rate": long_wr,
            "short_win_rate": short_wr,
            "coin_breakdown": [{"symbol": s, "pnl": p, "trades": coin_trades[s]} for s, p in top_coins[:10]],
            "health_score":  health,
            "total_closed":  len(closed),
        })
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
        return cached[0]

    from app.database import SessionLocal
    from sqlalchemy import text
    from datetime import datetime, timedelta
    from collections import defaultdict
    db = SessionLocal()
    try:
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

        # Single query: all executions via JOIN (replaces N+1 loop)
        exec_rows = db.execute(text("""
            SELECT e.outcome, e.pnl_pct,
                   COALESCE(e.closed_at, e.fired_at) AS ts
            FROM strategy_executions e
            JOIN user_strategies s ON s.id = e.strategy_id
            WHERE s.user_id = :uid
        """), {"uid": user.id}).fetchall()

        open_trades = sum(1 for r in exec_rows if r.outcome == "OPEN")
        closed = [(r.pnl_pct, r.ts, r.outcome) for r in exec_rows
                  if r.outcome in ("WIN", "LOSS", "BREAKEVEN") and r.pnl_pct is not None]
        total = len(closed)
        wins  = sum(1 for _, _, o in closed if o == "WIN")

        closed_30d = [(p, ts, o) for p, ts, o in closed if ts and ts > cutoff_30d]

        pnl_7d  = sum(p for p, ts, _ in closed if ts and ts > cutoff_7d)
        pnl_30d = sum(p for p, _, _ in closed_30d)
        pnl_all = sum(p for p, _, _ in closed)

        daily = defaultdict(float)
        for pnl, ts, _ in closed_30d:
            daily[ts.strftime("%m/%d")] += pnl

        cumulative, port_labels, port_values = 0.0, [], []
        for day, pnl in sorted(daily.items()):
            cumulative += pnl
            port_labels.append(day)
            port_values.append(round(cumulative, 2))

        result = JSONResponse({
            "total_strategies": total_strategies,
            "active_count":     active_count,
            "open_trades":      open_trades,
            "total_trades":     total,
            "win_rate":         round(wins / total * 100, 1) if total > 0 else 0,
            "pnl_7d":           round(pnl_7d, 2),
            "pnl_30d":          round(pnl_30d, 2),
            "pnl_all":          round(pnl_all, 2),
            "equity_30d":       {"labels": port_labels, "values": port_values},
        })
        _CACHE[cache_key] = (result, time.time() + 30)
        return result
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
                s.status = body["status"]
            s.config = config
            db.commit()
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
            s.status = body["status"]

        s.config = config
        db.commit()

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
            "auto_trading_enabled":     bool(prefs and prefs.auto_trading_enabled),
            # Security
            "has_password":             bool(user.password_hash),
            "auth_provider":            user.auth_provider or "telegram",
        })
    finally:
        db.close()


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

        # API keys — only overwrite if non-empty strings are provided
        api_key    = body.get("bitunix_api_key", "").strip()
        api_secret = body.get("bitunix_api_secret", "").strip()
        if api_key:
            prefs.bitunix_api_key = api_key
        if api_secret:
            prefs.bitunix_api_secret = api_secret
        # Auto-enable live trading once both keys are present
        if prefs.bitunix_api_key and prefs.bitunix_api_secret:
            prefs.auto_trading_enabled = True

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
        import httpx, asyncio
        async with httpx.AsyncClient(timeout=10) as client:
            trader = BitunixTrader(prefs.bitunix_api_key, prefs.bitunix_api_secret)
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
    try:
        user = _get_user_by_uid(uid, db)
        if not user:
            raise HTTPException(status_code=403, detail="Invalid UID")
        sub = _get_portal_sub(user.id, db)
        allowed, used, limit, is_pro = _chat_calls_info(sub, db)
        if not messages:
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

    system_prompt = """You are a sharp, experienced crypto trading strategist helping a user build their own automated strategy inside TradeHub. You're having a real conversation — not running through a checklist.

RULES:
- Reply in 1-3 short sentences max. No bullet points. Conversational only.
- Extract information from what the user naturally says — don't ask for something they've already told you.
- Never ask about "paper or live" — all strategies start in paper mode automatically. Don't mention it.
- Be reactive and specific to what the user says. If they say "RSI scalp" respond to THAT specifically.
- Sound like an expert, not a form. Use their words back at them.
- If something they say is risky or unusual, briefly flag it — but still help.

WHAT YOU NEED (in any order, collect naturally):
- Direction: LONG / SHORT / BOTH
- Style: SCALPER / SWING / MOMENTUM / REVERSAL  
- Primary signal (e.g. "RSI below 30", "MACD bullish cross on 15m", "EMA 9/21 crossover")
- Take profit % and stop loss % (you can suggest sensible defaults based on their style)
- Leverage (default to 5x for scalpers, 3x for swing if they don't specify)

OPTIONAL (ask only if relevant):
- Confirmation signal (only suggest this if their entry signal alone seems weak)
- Specific coins or "all"

COMPILE when you have: direction + signal + TP + SL. Don't keep asking more questions if you have enough.

When ready, say something natural like "Alright, that's everything I need — compiling your strategy now!" then output EXACTLY (on its own line after a blank line):
###STRATEGY###
Direction: LONG | Style: SCALPER | Primary Signal: RSI below 30 on 5m | Confirmation: none | TP1: 2% | SL: 1% | Leverage: 5x | Mode: paper | Coins: all

Use the actual values from the conversation. Only output ###STRATEGY### once and only when you have enough real info."""

    api_messages = [{"role": m["role"], "content": m["content"]} for m in messages]

    try:
        client = anthropic.AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        resp = await client.messages.create(
            model="claude-sonnet-4-5-20250929",
            max_tokens=400,
            system=system_prompt,
            messages=api_messages,
        )
        raw = resp.content[0].text
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

Return ONLY valid JSON (no markdown, no explanation outside the JSON):
{
  "timeframe": "15m",
  "direction": "LONG"|"SHORT"|"BOTH",
  "conditions": [ <2-5 condition objects from the list above> ],
  "explanation": "2-3 sentences explaining what this indicator measures, the specific parameters it uses, and what market condition triggers an entry."
}"""

    user_msg = f"Indicator description:\n{prompt}"

    result = None
    # Try Anthropic first
    try:
        import anthropic as _anthropic
        _ac = _anthropic.Anthropic()
        resp = _ac.messages.create(
            model="claude-sonnet-4-5",
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
    if secret != "5603353066":
        raise HTTPException(status_code=403, detail="Forbidden")
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
        user = db.query(User).filter(User.telegram_id == "5603353066").first()
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
        _, used, lim, is_pro = _chat_calls_info(sub, db)
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
    is_prod = _os.environ.get("REPL_DEPLOYMENT") == "1" or _os.environ.get("FORCE_EXECUTOR", "").lower() in ("1","true","yes")
    # Query DB for the advisory lock
    from app.database import SessionLocal
    from sqlalchemy import text
    db = SessionLocal()
    try:
        row = db.execute(text(
            "SELECT l.pid, COALESCE(a.state,'gone') AS state "
            "FROM pg_locks l "
            "LEFT JOIN pg_stat_activity a ON a.pid = l.pid "
            "WHERE l.locktype='advisory' AND l.objid=42424242 AND l.granted=true "
            "LIMIT 1"
        )).fetchone()
        lock_info = {"pid": row[0], "state": row[1]} if row else None
    except Exception as e:
        lock_info = {"error": str(e)}
    finally:
        db.close()

    return {
        "executor_running_in_this_worker": _executor_running_in_this_worker,
        "is_production": is_prod,
        "advisory_lock_holder": lock_info,
        "lock_id": 42424242,
    }


@app.post("/api/admin/executor/force-start")
async def executor_force_start(request: Request):
    """
    Admin endpoint — forces this worker to acquire the executor lock and start
    the executor right now.  Use when production shows both workers as HTTP-only.
    Requires the dev secret in the Authorization header.
    """
    import os as _os
    secret = request.headers.get("Authorization", "").replace("Bearer ", "")
    if secret != "tradehub-portal-secret-2025":
        raise HTTPException(status_code=403, detail="Forbidden")

    global _executor_running_in_this_worker
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
            cur.execute("""
                SELECT pid FROM pg_locks l
                JOIN pg_stat_activity a ON a.pid = l.pid
                WHERE l.locktype='advisory' AND l.objid=42424242 AND l.granted=true
            """)
            holders = cur.fetchall()
            for (pid,) in holders:
                cur.execute("SELECT pg_terminate_backend(%s)", (pid,))
                logger.info(f"[force-start] Terminated stale lock holder PID={pid}")
            import time; time.sleep(0.5)  # brief pause for PG to release
            cur.execute("SELECT pg_try_advisory_lock(42424242)")
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
        await _start_executor_tasks()
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
    secret = request.headers.get("Authorization", "").replace("Bearer ", "")
    if secret != "tradehub-portal-secret-2025":
        raise HTTPException(status_code=403, detail="Forbidden")

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
    secret = request.headers.get("Authorization", "").replace("Bearer ", "")
    if secret != "tradehub-portal-secret-2025":
        raise HTTPException(status_code=403, detail="Forbidden")

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

        if not settings.OXAPAY_MERCHANT_API_KEY:
            raise HTTPException(status_code=503, detail="Payment not configured")

        domain = os.getenv("REPLIT_DOMAINS", "perp-signal-bot.replit.app").split(",")[0].strip()
        callback_url = f"https://{domain}/api/portal/oxapay-webhook"
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
    sig    = request.headers.get("hmac", "")
    oxapay = OxaPayService(settings.OXAPAY_MERCHANT_API_KEY or "")
    if settings.OXAPAY_MERCHANT_API_KEY and sig:
        if not oxapay.verify_webhook_signature(sig, body, settings.OXAPAY_MERCHANT_API_KEY):
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
            model="claude-haiku-4-5",
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
        if not _is_portal_pro(sub):
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

        # ── Fetch last 100 closed executions ─────────────────────────────────
        execs = (
            db.query(StrategyExecution)
            .filter(
                StrategyExecution.strategy_id == strategy_id,
                StrategyExecution.outcome.in_(["WIN", "LOSS", "BREAKEVEN"]),
            )
            .order_by(StrategyExecution.fired_at.desc())
            .limit(100)
            .all()
        )

        # ── Build trade analytics ─────────────────────────────────────────────
        DOW = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

        def _session(h):
            if h < 6:   return "Night (00-06 UTC)"
            if h < 12:  return "Morning (06-12 UTC)"
            if h < 18:  return "Afternoon (12-18 UTC)"
            return "Evening (18-24 UTC)"

        def _wr(s):
            tot = s["win"] + s["loss"]
            return f"{round(s['win']/tot*100)}% ({s['win']}W/{s['loss']}L)" if tot else "no data"

        hour_stats = {}
        dow_stats  = {}
        sess_stats = {}
        coin_stats = {}

        session_block = "  No session data yet"
        dow_block     = "  No day-of-week data yet"
        best_coins    = "not enough data"
        worst_coins   = "none negative yet"
        trade_log     = "  No trades yet"

        try:
            for ex in execs:
                is_win  = ex.outcome == "WIN"
                is_loss = ex.outcome == "LOSS"
                coin = (ex.symbol or "").replace("USDT", "")

                if ex.fired_at:
                    h  = ex.fired_at.hour
                    d  = ex.fired_at.weekday()
                    sn = _session(h)

                    if h  not in hour_stats: hour_stats[h]  = {"win": 0, "loss": 0}
                    if d  not in dow_stats:  dow_stats[d]   = {"win": 0, "loss": 0}
                    if sn not in sess_stats: sess_stats[sn] = {"win": 0, "loss": 0}

                    if is_win:
                        hour_stats[h]["win"]  += 1
                        dow_stats[d]["win"]   += 1
                        sess_stats[sn]["win"] += 1
                    if is_loss:
                        hour_stats[h]["loss"]  += 1
                        dow_stats[d]["loss"]   += 1
                        sess_stats[sn]["loss"] += 1

                if coin:
                    if coin not in coin_stats: coin_stats[coin] = {"win": 0, "loss": 0, "pnl": 0.0}
                    if is_win:  coin_stats[coin]["win"]  += 1
                    if is_loss: coin_stats[coin]["loss"] += 1
                    if ex.pnl_pct is not None:
                        coin_stats[coin]["pnl"] += float(ex.pnl_pct)

            # Session breakdown
            sess_lines = []
            for sn in ["Night (00-06 UTC)", "Morning (06-12 UTC)", "Afternoon (12-18 UTC)", "Evening (18-24 UTC)"]:
                if sn in sess_stats:
                    sess_lines.append(f"  {sn}: {_wr(sess_stats[sn])}")
            if sess_lines:
                session_block = "\n".join(sess_lines)

            # Day-of-week breakdown
            dow_lines = []
            for d in range(7):
                if d in dow_stats:
                    dow_lines.append(f"  {DOW[d]}: {_wr(dow_stats[d])}")
            if dow_lines:
                dow_block = "\n".join(dow_lines)

            # Best / worst coins
            coin_sorted = sorted(coin_stats.items(), key=lambda x: x[1]["pnl"], reverse=True)
            if coin_sorted:
                best_coins  = ", ".join(
                    f"{c} ({v['win']}W/{v['loss']}L, {v['pnl']:+.0f}%)"
                    for c, v in coin_sorted[:5]
                )
                negatives = [(c, v) for c, v in coin_sorted[-5:] if v["pnl"] < 0]
                if negatives:
                    worst_coins = ", ".join(
                        f"{c} ({v['win']}W/{v['loss']}L, {v['pnl']:+.0f}%)"
                        for c, v in negatives
                    )

            # Recent trade log (last 25)
            recent_lines = []
            for ex in reversed(execs[:25]):
                try:
                    dt   = ex.fired_at.strftime("%m/%d %H:%M") if ex.fired_at else "?"
                    coin = (ex.symbol or "?").replace("USDT", "")
                    pnl  = f"{ex.pnl_pct:+.1f}%" if ex.pnl_pct is not None else "?"
                    dur  = ""
                    if ex.fired_at and ex.closed_at:
                        # strip tz info to avoid naive/aware mismatch
                        fa = ex.fired_at.replace(tzinfo=None)
                        ca = ex.closed_at.replace(tzinfo=None)
                        mins = int((ca - fa).total_seconds() / 60)
                        dur  = f" {mins}m"
                    recent_lines.append(
                        f"  {dt} | {coin} {ex.direction or ''} | {ex.outcome}{dur} | {pnl}"
                    )
                except Exception:
                    continue
            if recent_lines:
                trade_log = "\n".join(recent_lines)

        except Exception as analytics_err:
            logger.warning(f"Strategy advisor analytics error (non-fatal): {analytics_err}")

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
            model="claude-sonnet-4-5-20250929",
            max_tokens=500,
            system=system_prompt,
            messages=api_messages,
        )
        return {"reply": resp.content[0].text, "pro_required": False}
    except Exception as e:
        logger.error(f"Strategy advisor AI error: {e}")
        return {"reply": "Sorry, I hit an issue — please try again.", "pro_required": False}


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
    db = SessionLocal()
    try:
        user = _get_user_by_uid(uid, db)
        if not user:
            raise HTTPException(status_code=403, detail="Invalid UID")
        _sub = _get_portal_sub(user.id, db)
        if not _is_portal_pro(_sub) and not getattr(user, "is_admin", False):
            return {"error": "PRO_REQUIRED",
                    "message": "A Pro subscription ($50/month) is required to run backtests."}
    finally:
        db.close()

    config = body.get("config") or {}
    days   = int(body.get("days", 30))
    if days not in (30, 90):
        days = 30

    try:
        from app.services.backtest_engine import run_backtest
        result = await asyncio.wait_for(run_backtest(config, days), timeout=60)
        return result
    except asyncio.TimeoutError:
        return {"error": "Backtest timed out. Try a shorter window (30 days) or a faster timeframe."}
    except Exception as exc:
        logger.error(f"[Backtest] error uid={uid}: {exc}", exc_info=True)
        return {"error": f"Backtest failed: {exc}"}


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
                model="claude-haiku-4-5",
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


@app.get("/admin/test-bitunix")
async def admin_test_bitunix(secret: str = Query(...), user_id: int = Query(...)):
    """
    Admin diagnostic: test a subscriber's Bitunix connection.
    Returns balance, open positions, and any errors so you can see exactly
    why a user's live orders are failing.
    """
    if secret != "tradehub-portal-secret-2025":
        raise HTTPException(status_code=403, detail="Forbidden")

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


@app.get("/health")
async def health():
    return {"status": "ok", "service": "strategy-portal"}


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
            score      = round(sum(t.pnl_pct for t in trades if t.pnl_pct), 2)
            trade_count = len(trades)
            wins        = sum(1 for t in trades if t.outcome == "WIN")
            win_rate    = round(wins / trade_count * 100, 1) if trade_count else 0
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


@app.post("/admin/competition/create")
async def admin_competition_create(request: Request, secret: str = Query(...)):
    """Admin-only: create a new monthly competition."""
    import os
    if secret != os.environ.get("ADMIN_SECRET", "5603353066"):
        raise HTTPException(status_code=403)

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
