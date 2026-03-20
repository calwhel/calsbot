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

async def _maintain_advisory_lock(conn):
    """
    Keep the psycopg2 advisory-lock connection alive with periodic pings.
    Neon drops idle connections after ~5 min, so we ping every 4 minutes.
    If the connection dies we simply stop pinging — the lock is gone but
    the executor continues (the next sweep will just be a bit heavier).
    """
    loop = asyncio.get_event_loop()
    while True:
        await asyncio.sleep(240)   # 4 minutes
        try:
            await loop.run_in_executor(None, lambda: conn.cursor().execute("SELECT 1"))
        except Exception as e:
            logger.warning(f"Advisory lock keepalive failed: {e}")
            break


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
        # ── Advisory lock: only ONE worker runs the executor + monitors ────────
        # With 4 uvicorn workers, without this lock ALL FOUR would start the
        # executor, live monitor, paper monitor and ghost cleanup — 4× the DB
        # connections and 4× duplicate trade signals.
        lock_conn = await loop.run_in_executor(None, _try_acquire_executor_lock)
        if lock_conn:
            trigger = "REPL_DEPLOYMENT" if _os.environ.get("REPL_DEPLOYMENT") == "1" else "FORCE_EXECUTOR"
            logger.info(f"✅ Executor lock acquired — this worker runs executor + monitors ({trigger})")
            # Keep the advisory-lock connection alive
            asyncio.create_task(_maintain_advisory_lock(lock_conn))
            # Ghost cleanup runs only in this worker
            await _cancel_ghost_executions()
            asyncio.create_task(_ghost_cleanup_loop())
            try:
                from app.services.strategy_executor import (
                    run_strategy_executor, run_live_position_monitor,
                    backfill_cancelled_paper_trades,
                    backfill_ghost_cancelled_executions,
                )
                asyncio.create_task(run_strategy_executor())
                asyncio.create_task(run_live_position_monitor())
                asyncio.create_task(backfill_cancelled_paper_trades(lookback_days=30))
                asyncio.create_task(backfill_ghost_cancelled_executions(lookback_days=7))
            except Exception as e:
                logger.error(f"Failed to start strategy executor: {e}")
        else:
            logger.info("⏭️  Executor lock held by another worker — this worker handles HTTP only")
    else:
        logger.info("Strategy executor DISABLED (dev environment — only production runs it)")


# ─────────────────────────────────────────────────────────────────────────────
# Health check — responds immediately, even before migrations finish
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok", "ts": int(__import__("time").time())}


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


@app.post("/login")
async def login_submit(request: Request):
    """Verify UID, set session cookie, return redirect URL as JSON."""
    from app.database import SessionLocal
    body = await request.json()
    uid = (body.get("uid") or "").strip().upper()
    if not uid.startswith("TH-") or len(uid) < 6:
        raise HTTPException(status_code=400, detail="Invalid access code format")

    def _do_lookup():
        db = SessionLocal()
        try:
            return _get_user_by_uid(uid, db), getattr(db.query(User).filter(User.uid == uid).first(), "banned", False)
        finally:
            db.close()

    user_result = await asyncio.to_thread(_do_lookup)
    user, banned = user_result
    if not user:
        raise HTTPException(status_code=403, detail="Access code not found. Send /start to the Telegram bot to get your code — it appears at the top of the bot dashboard.")
    if banned:
        raise HTTPException(status_code=403, detail="This account has been suspended.")
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
    resp = RedirectResponse(url="/", status_code=302)
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
            if ref_code:
                referrer = db.query(User).filter(User.referral_code == ref_code).first()
                if referrer and referrer.id != user.id:
                    user.referred_by = ref_code
                    db.commit()
            return {"uid": user.uid}
        finally:
            db.close()

    result = await asyncio.to_thread(_do_register)
    if result.get("error") == "exists":
        raise HTTPException(status_code=409, detail="An account with this email already exists. Please sign in.")
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
async def google_auth_start(request: Request):
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
    from app.strategy_models import UserStrategy, StrategyPerformance
    db = SessionLocal()
    try:
        user = _get_user_by_uid(uid, db)
        if not user:
            return None
        if user.banned:
            return "banned"

        strategies = (
            db.query(UserStrategy)
            .filter(UserStrategy.user_id == user.id)
            .order_by(UserStrategy.updated_at.desc())
            .all()
        )

        strategy_ids = [s.id for s in strategies]
        perf_map: dict = {}
        if strategy_ids:
            perfs = db.query(StrategyPerformance).filter(
                StrategyPerformance.strategy_id.in_(strategy_ids)
            ).all()
            perf_map = {p.strategy_id: p for p in perfs}

        strategy_data = []
        for s in strategies:
            perf = perf_map.get(s.id)
            strategy_data.append({
                "id":           s.id,
                "name":         s.name,
                "description":  s.description,
                "status":       s.status,
                "is_public":    s.is_public,
                "config":       s.config,
                "created_at":   s.created_at.strftime("%Y-%m-%d") if s.created_at else "",
                "total_trades": perf.total_trades if perf else 0,
                "win_rate":     round(perf.win_rate, 1) if perf else 0,
                "total_pnl":    round(perf.total_pnl_pct, 2) if perf else 0,
                "open_trades":  perf.open_trades if perf else 0,
            })

        _psub = _get_portal_sub(user.id, db)
        _is_pro = _is_portal_pro(_psub) or bool(getattr(user, "is_admin", False))
        is_web_user = str(getattr(user, "telegram_id", "") or "").startswith("WEB-")

        return {
            "user":        user,
            "strategies":  strategy_data,
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
        return cached[0]

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
    resp = JSONResponse(data)
    _CACHE[cache_key] = (resp, time.time() + 15)
    return resp


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

        # Null out FK references in strategy_purchases so the delete doesn't fail
        # when this strategy was previously listed on the marketplace and purchased.
        try:
            from app.strategy_marketplace_ext import StrategyPurchase, StrategyListing
            db.query(StrategyPurchase).filter(
                StrategyPurchase.cloned_strategy_id == strategy_id
            ).update({"cloned_strategy_id": None}, synchronize_session=False)
            # Delist from marketplace if still active
            db.query(StrategyListing).filter(
                StrategyListing.strategy_id == strategy_id
            ).update({"is_active": False}, synchronize_session=False)
        except Exception:
            pass  # marketplace tables may not exist in all environments

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

        result = []
        for m in listings:
            perf   = db.query(StrategyPerformance).filter(StrategyPerformance.strategy_id == m.strategy_id).first()
            author = db.query(User).filter(User.id == m.author_id).first()
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

        return JSONResponse({"success": True, "cloned_strategy_id": new_strategy.id, "name": new_strategy.name})
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

        pnl_7d  = sum(p for p, ts, _ in closed if ts and ts > cutoff_7d)
        pnl_30d = sum(p for p, ts, _ in closed if ts and ts > cutoff_30d)
        pnl_all = sum(p for p, _, _ in closed)

        daily = defaultdict(float)
        for pnl, ts, _ in closed:
            if ts and ts > cutoff_30d:
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

        db.commit()
        return JSONResponse({"success": True})
    finally:
        db.close()


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
    uid  = (body.get("uid") or "").strip()
    prompt    = (body.get("prompt") or "").strip()
    timeframe = (body.get("timeframe") or "15m").strip()
    direction = (body.get("direction") or "LONG").strip()

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
    system_prompt = """You are an expert algorithmic trading assistant.
The user will describe a trading indicator or set of conditions in plain English.
Your job is to convert it into a JSON list of structured entry conditions that our platform understands.

Each condition must be one of these types (use exact type names):
- rsi: { type:"rsi", timeframe, operator:"<"|">"|"crosses_above"|"crosses_below", value:number }
- ema: { type:"ema", timeframe, period:number, operator:"price_above"|"price_below"|"crosses_above"|"crosses_below" }
- macd: { type:"macd", timeframe, condition:"bullish_cross"|"bearish_cross"|"positive"|"negative"|"histogram_rising" }
- volume: { type:"volume", timeframe, condition:"above_average"|"spike"|"below_average", multiplier:number }
- bb: { type:"bb", timeframe, condition:"price_above_upper"|"price_below_lower"|"price_near_middle"|"squeeze" }
- vwap: { type:"vwap", timeframe, condition:"price_above"|"price_below"|"bounce" }
- stoch: { type:"stoch", timeframe, operator:"<"|">"|"crosses_above"|"crosses_below", value:number }
- adx: { type:"adx", timeframe, operator:">"|"<", value:number }
- price_action: { type:"price_action", timeframe, condition:"higher_high"|"higher_low"|"lower_high"|"lower_low"|"inside_bar" }
- supertrend: { type:"supertrend", timeframe, condition:"bullish"|"bearish"|"flip_bullish"|"flip_bearish" }

Also add a human-readable "label" field to each condition for display.

Return ONLY valid JSON in this format:
{
  "conditions": [ ... ],
  "explanation": "One sentence explaining what this indicator setup looks for."
}
Do not include any text outside the JSON."""

    user_msg = f"Indicator description: {prompt}\nTimeframe: {timeframe}\nDirection preference: {direction}"

    result = None
    # Try Anthropic first
    try:
        import anthropic as _anthropic
        _ac = _anthropic.Anthropic()
        resp = _ac.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=800,
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

        referred_count = db.query(User).filter(User.referred_by == user.referral_code).count()
        return JSONResponse({
            "referral_code": user.referral_code,
            "referral_earnings": float(user.referral_earnings or 0.0),
            "referred_count": referred_count,
        })
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

        from app.strategy_models import UserStrategy, StrategyPerformance
        strategy = db.query(UserStrategy).filter(
            UserStrategy.id == strategy_id,
            UserStrategy.user_id == user.id
        ).first()
        if not strategy:
            raise HTTPException(status_code=404, detail="Strategy not found")

        perf = db.query(StrategyPerformance).filter(
            StrategyPerformance.strategy_id == strategy_id
        ).first()

        perf_summary = "No trades run yet."
        if perf and perf.total_trades:
            wr = round(perf.win_rate or 0, 1)
            pnl = round(perf.total_pnl_pct or 0, 2)
            perf_summary = f"{perf.total_trades} trades | {wr}% win rate | {pnl}% total P&L"

        system_prompt = f"""You are an expert crypto trading strategy analyst inside TradeHub.

The user is asking about their strategy called "{strategy.name}".

Strategy description:
{strategy.description or "No description provided."}

Performance:
{perf_summary}

Your job:
- Give honest, specific, actionable advice
- Point out real weaknesses if they exist — don't be generic
- Suggest concrete improvements: better confirmation signals, tighter/wider SL, TP targets, market conditions it works best in
- Keep responses concise (2-4 sentences max)
- Be conversational, not bullet-list heavy"""

        api_messages = [{"role": m["role"], "content": m["content"]} for m in messages]

    finally:
        db.close()

    try:
        client = anthropic.AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        resp = await client.messages.create(
            model="claude-sonnet-4-5-20250929",
            max_tokens=350,
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
            _log.info("Starting uvicorn on port 5000 with 4 workers...")
            uvicorn.run(
                "strategy_portal_server:app",
                host="0.0.0.0",
                port=5000,
                workers=4,
                loop="asyncio",
            )
        except Exception as _e:
            _log.error(f"Server crashed: {_e} — restarting in 3s")
            time.sleep(3)
