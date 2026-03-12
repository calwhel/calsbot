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
from datetime import datetime, timedelta
from typing import Optional, Dict, Tuple

from fastapi import FastAPI, Request, HTTPException, Query
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, FileResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session
from app.models import User

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Strategy Portal", docs_url=None, redoc_url=None)
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


def _set_session(response, uid: str):
    response.set_cookie(
        key=_COOKIE_NAME,
        value=_make_token(uid),
        max_age=_COOKIE_MAX_AGE,
        httponly=True,
        samesite="lax",
        secure=False,  # set True if behind HTTPS proxy
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


async def _notify_admin_go_live(user, strategy):
    """Alert admin via Telegram when a user promotes a strategy to live."""
    from app.config import settings
    from app.database import SessionLocal
    from app.models import UserPreference
    admin_id = getattr(settings, "OWNER_TELEGRAM_ID", None)
    if not admin_id:
        return
    tg_id   = getattr(user, "telegram_id", "unknown")
    name    = getattr(user, "first_name", "") or getattr(user, "username", "") or "Unknown"
    uname   = getattr(user, "username", "")
    uid     = getattr(user, "uid", "")
    mention = f"@{uname}" if uname else f"ID {tg_id}"
    cfg     = strategy.config or {}
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
    from app.database import engine
    from app.strategy_models import init_strategy_tables
    import sqlalchemy as sa
    init_strategy_tables(engine)
    # Ensure all portal/auth columns exist on users table
    migrations = [
        "ALTER TABLE users ADD COLUMN IF NOT EXISTS email VARCHAR UNIQUE",
        "ALTER TABLE users ADD COLUMN IF NOT EXISTS email_verified BOOLEAN DEFAULT FALSE",
        "ALTER TABLE users ADD COLUMN IF NOT EXISTS password_hash VARCHAR",
        "ALTER TABLE users ADD COLUMN IF NOT EXISTS google_id VARCHAR UNIQUE",
        "ALTER TABLE users ADD COLUMN IF NOT EXISTS auth_provider VARCHAR DEFAULT 'telegram'",
    ]
    try:
        with engine.connect() as conn:
            for sql in migrations:
                conn.execute(sa.text(sql))
            conn.commit()
    except Exception as e:
        logger.warning(f"user column migration: {e}")


@app.on_event("startup")
async def startup():
    _ensure_tables()
    logger.info("Strategy portal started on port 5000")
    # Launch strategy executor in the background — reads from this same DB
    # and sends Telegram DMs directly via Bot API (no Railway dependency)
    try:
        from app.services.strategy_executor import run_strategy_executor
        asyncio.create_task(run_strategy_executor())
        logger.info("Strategy executor started")
    except Exception as e:
        logger.error(f"Failed to start strategy executor: {e}")


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


@app.post("/login")
async def login_submit(request: Request):
    """Verify UID, set session cookie, return redirect URL as JSON."""
    from app.database import SessionLocal
    body = await request.json()
    uid = (body.get("uid") or "").strip().upper()
    if not uid.startswith("TH-") or len(uid) < 6:
        raise HTTPException(status_code=400, detail="Invalid access code format")
    db = SessionLocal()
    try:
        user = _get_user_by_uid(uid, db)
        if not user:
            raise HTTPException(status_code=403, detail="Access code not found. Check the Telegram bot for your TH-XXXXXXXX code.")
        if user.banned:
            raise HTTPException(status_code=403, detail="This account has been suspended.")
        resp = JSONResponse({"redirect": "/app"})
        _set_session(resp, uid)
        return resp
    finally:
        db.close()


@app.post("/login/email-request")
async def login_email_request(request: Request):
    """Step 1: user submits email — if found, send OTP via Telegram."""
    from app.database import SessionLocal
    body = await request.json()
    email = (body.get("email") or "").strip().lower()
    if not email or "@" not in email:
        raise HTTPException(status_code=400, detail="Please enter a valid email address.")
    db = SessionLocal()
    try:
        user = db.query(User).filter(User.email == email).first()
        if not user:
            raise HTTPException(
                status_code=404,
                detail="No account found with that email. Link your email via the Telegram bot first — type /setemail your@email.com"
            )
        if user.banned:
            raise HTTPException(status_code=403, detail="This account has been suspended.")
        otp = _generate_otp()
        _otp_store[email] = (otp, datetime.utcnow() + timedelta(minutes=_OTP_TTL_MINUTES))
        await _send_otp_via_telegram(
            user.telegram_id,
            otp,
            user.first_name or user.username or ""
        )
        return JSONResponse({"ok": True, "message": "Code sent to your Telegram. Check your messages!"})
    finally:
        db.close()


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
    db = SessionLocal()
    try:
        user = db.query(User).filter(User.email == email).first()
        if not user or not user.uid:
            raise HTTPException(status_code=403, detail="Account not found.")
        if user.banned:
            raise HTTPException(status_code=403, detail="This account has been suspended.")
        resp = JSONResponse({"redirect": "/app"})
        _set_session(resp, user.uid)
        return resp
    finally:
        db.close()


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

    if not email or "@" not in email:
        raise HTTPException(status_code=400, detail="Please enter a valid email address.")
    if len(password) < 8:
        raise HTTPException(status_code=400, detail="Password must be at least 8 characters.")
    if not name:
        raise HTTPException(status_code=400, detail="Please enter your name.")

    db = SessionLocal()
    try:
        existing = db.query(User).filter(User.email == email).first()
        if existing:
            raise HTTPException(status_code=409, detail="An account with this email already exists. Please sign in.")

        uid = _generate_web_uid(db)
        web_tid = f"WEB-{secrets.token_hex(8).upper()}"
        user = User(
            telegram_id=web_tid,
            uid=uid,
            email=email,
            email_verified=False,
            password_hash=_hash_password(password),
            first_name=name,
            auth_provider="email",
            approved=True,
        )
        db.add(user)
        db.commit()
        db.refresh(user)

        resp = JSONResponse({"redirect": "/app"})
        _set_session(resp, user.uid)
        return resp
    finally:
        db.close()


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

    db = SessionLocal()
    try:
        user = db.query(User).filter(User.email == email).first()
        if not user or not user.password_hash:
            raise HTTPException(status_code=403, detail="No account found with that email. Did you sign up with Google?")
        if not _verify_password(password, user.password_hash):
            raise HTTPException(status_code=403, detail="Incorrect password.")
        if user.banned:
            raise HTTPException(status_code=403, detail="This account has been suspended.")
        if not user.uid:
            raise HTTPException(status_code=403, detail="Account setup incomplete. Please contact support.")
        resp = JSONResponse({"redirect": "/app"})
        _set_session(resp, user.uid)
        return resp
    finally:
        db.close()


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
    db = SessionLocal()
    try:
        user = db.query(User).filter(User.telegram_id == telegram_id).first()
        if not user:
            raise HTTPException(
                status_code=404,
                detail="No TradeHub account found for this Telegram account. "
                       "Open the bot and send /start first, then try again."
            )
        if user.banned:
            raise HTTPException(status_code=403, detail="This account has been suspended.")
        if not user.approved:
            raise HTTPException(status_code=403, detail="Your account is pending approval.")
        if not user.uid:
            raise HTTPException(status_code=403, detail="Account setup incomplete — please contact support.")

        logger.info(f"Telegram login: uid={user.uid} tg_id={telegram_id} username=@{user.username}")
        resp = JSONResponse({"redirect": "/app"})
        _set_session(resp, user.uid)
        return resp
    finally:
        db.close()


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
            _set_session(resp, user.uid)
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

async def _render_portal(request: Request, uid: str):
    """Shared logic for /app and /strategies."""
    from app.database import SessionLocal
    db = SessionLocal()
    try:
        user = _get_user_by_uid(uid, db)
        if not user:
            raise HTTPException(status_code=403, detail="Invalid access link")
        if user.banned:
            raise HTTPException(status_code=403, detail="Account banned")

        from app.strategy_models import UserStrategy, StrategyPerformance

        strategies = (
            db.query(UserStrategy)
            .filter(UserStrategy.user_id == user.id)
            .order_by(UserStrategy.updated_at.desc())
            .all()
        )

        strategy_data = []
        for s in strategies:
            perf = db.query(StrategyPerformance).filter(
                StrategyPerformance.strategy_id == s.id
            ).first()
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

        return templates.TemplateResponse("strategy_portal.html", {
            "request":    request,
            "user":       user,
            "uid":        uid,
            "strategies": strategy_data,
        })
    finally:
        db.close()


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
    """Aggregate stats for the landing page hero section."""
    from app.database import SessionLocal
    db = SessionLocal()
    try:
        from app.strategy_models import UserStrategy, StrategyPerformance, StrategyMarketplace
        from app.strategy_marketplace_ext import StrategyPurchase
        from sqlalchemy import func

        total_strategies = db.query(func.count(UserStrategy.id)).scalar() or 0
        perf_rows = db.query(StrategyPerformance.win_rate).filter(
            StrategyPerformance.win_rate > 0
        ).all()
        avg_win_rate = round(
            sum(r.win_rate for r in perf_rows) / len(perf_rows), 1
        ) if perf_rows else 0

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


@app.get("/api/public/marketplace")
async def public_marketplace(limit: int = Query(6, ge=1, le=20)):
    """Top marketplace listings — no auth required."""
    from app.database import SessionLocal, engine
    db = SessionLocal()
    try:
        from app.strategy_models import StrategyMarketplace, StrategyPerformance, UserStrategy
        from app.strategy_marketplace_ext import init_marketplace_ext_tables
        from app.models import User as UserModel
        init_marketplace_ext_tables(engine)

        listings = (
            db.query(StrategyMarketplace)
            .order_by(
                StrategyMarketplace.avg_rating.desc(),
                StrategyMarketplace.clone_count.desc(),
            )
            .limit(limit)
            .all()
        )

        result = []
        for m in listings:
            # get creator name
            creator_name = "Anonymous"
            try:
                if m.strategy_id:
                    us = db.query(UserStrategy).filter(UserStrategy.id == m.strategy_id).first()
                    if us:
                        u = db.query(UserModel).filter(UserModel.id == us.user_id).first()
                        if u:
                            creator_name = (u.first_name or u.username or u.uid or "Anonymous")
            except Exception:
                pass

            perf = None
            if m.strategy_id:
                perf = db.query(StrategyPerformance).filter(
                    StrategyPerformance.strategy_id == m.strategy_id
                ).first()

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
                "win_rate":      round(perf.win_rate, 1) if perf else None,
                "total_pnl":     round(perf.total_pnl_pct, 2) if perf else None,
                "total_trades":  perf.total_trades if perf else 0,
            })
        return result
    except Exception as e:
        logger.warning(f"public_marketplace error: {e}")
        return []
    finally:
        db.close()


@app.get("/api/public/leaderboard")
async def public_leaderboard(limit: int = Query(5, ge=1, le=10)):
    """Top strategies by P&L — no auth required."""
    from app.database import SessionLocal
    db = SessionLocal()
    try:
        from app.strategy_models import StrategyPerformance, UserStrategy
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
    from app.database import SessionLocal
    db = SessionLocal()
    try:
        user = _get_user_by_uid(uid, db)
        if not user:
            raise HTTPException(status_code=403)

        from app.strategy_models import UserStrategy, StrategyPerformance, StrategyExecution
        strategies = (
            db.query(UserStrategy)
            .filter(UserStrategy.user_id == user.id)
            .order_by(UserStrategy.updated_at.desc())
            .all()
        )

        result = []
        for s in strategies:
            perf = db.query(StrategyPerformance).filter(
                StrategyPerformance.strategy_id == s.id
            ).first()
            recent_execs = (
                db.query(StrategyExecution)
                .filter(StrategyExecution.strategy_id == s.id)
                .order_by(StrategyExecution.fired_at.desc())
                .limit(10)
                .all()
            )
            # Fast inline health score
            wr  = perf.win_rate if perf else 0
            tot = perf.total_trades if perf else 0
            pf  = (perf.avg_win_pct * max(perf.wins,1)) / (abs(perf.avg_loss_pct) * max(perf.losses,1)) if perf and perf.losses > 0 and perf.avg_loss_pct else 0
            health = 0.0
            if tot >= 3:
                health += min(wr / 100, 1.0) * 4.0
                health += min(pf / 2.0, 1.0) * 3.0
                health += min(tot / 30.0, 1.0) * 2.0
                health += 1.0  # base point for having trades
            health_score = round(min(health, 10.0), 1)

            result.append({
                "id":           s.id,
                "name":         s.name,
                "description":  s.description,
                "status":       s.status,
                "config":       s.config,
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
        return JSONResponse(result)
    finally:
        db.close()


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

        db.delete(strategy)
        db.commit()
        return {"deleted": True}
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
        init_marketplace_ext_tables(engine)

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
        init_marketplace_ext_tables(engine)

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
        is_owned = (m.pricing_model or "free") == "free" or db.query(StrategyPurchase).filter(
            StrategyPurchase.buyer_id == user.id,
            StrategyPurchase.listing_id == listing_id,
            StrategyPurchase.status == "active",
        ).first() is not None
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
        init_strategy_tables(engine)
        init_marketplace_ext_tables(engine)

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

        # Free — clone immediately
        original = db.query(UserStrategy).filter(UserStrategy.id == listing.strategy_id).first()
        if not original:
            raise HTTPException(status_code=404)

        import copy
        cloned_config = copy.deepcopy(original.config)
        cloned_config["name"] = f"{original.name} (Clone)"
        new_strategy = UserStrategy(
            user_id=user.id, name=cloned_config["name"],
            description=original.description, config=cloned_config, status="draft"
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
        init_marketplace_ext_tables(engine)

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
            if not perf or perf.total_trades < 3:
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
            UserStrategy.status.in_(["active", "paused", "draft"])
        ).all()
        for s in all_strategies:
            if s.id in seen_strategy_ids:
                continue
            perf   = db.query(StrategyPerformance).filter(StrategyPerformance.strategy_id == s.id).first()
            author = db.query(User).filter(User.id == s.user_id).first()
            if not perf or perf.total_trades < 3:
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
        _get_user_by_uid(uid, db)
        creator = db.query(User).filter(User.uid == creator_uid).first()
        if not creator:
            raise HTTPException(status_code=404)

        from app.strategy_models import StrategyMarketplace, StrategyPerformance
        from app.strategy_marketplace_ext import CreatorEarnings, init_marketplace_ext_tables
        init_marketplace_ext_tables(engine)

        listings = db.query(StrategyMarketplace).filter(StrategyMarketplace.author_id == creator.id).all()
        earnings = db.query(CreatorEarnings).filter(CreatorEarnings.creator_id == creator.id).first()

        return JSONResponse({
            "name": creator.first_name or creator.username or "Anonymous",
            "uid": creator.uid,
            "joined": creator.created_at.strftime("%B %Y") if creator.created_at else "Unknown",
            "strategy_count": len(listings),
            "total_subscribers": earnings.total_subscribers if earnings else 0,
            "total_sales": earnings.total_sales if earnings else 0,
            "strategies": [{
                "id": m.id, "title": m.title, "summary": m.summary,
                "pricing_model": m.pricing_model or "free", "price_usdt": m.price_usdt or 0,
                "clone_count": m.clone_count or 0, "avg_rating": round(m.avg_rating or 0, 1),
                "is_verified": m.is_verified,
            } for m in listings],
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
        init_marketplace_ext_tables(engine)

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
        init_marketplace_ext_tables(engine)

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
        init_strategy_tables(engine)

        strategy = db.query(UserStrategy).filter(
            UserStrategy.id == strategy_id,
            UserStrategy.user_id == user.id,
        ).first()
        if not strategy:
            raise HTTPException(status_code=404)

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
    init_strategy_tables(engine)

    db = SessionLocal()
    try:
        user = _get_user_by_uid(uid, db)
        if not user or user.banned:
            raise HTTPException(status_code=403)

        # Determine initial status from build mode flag and portal settings
        build_mode = config.get("_build_mode", "live")
        from app.strategy_models import StrategyPortalSettings
        portal = db.query(StrategyPortalSettings).filter(StrategyPortalSettings.user_id == user.id).first()

        if build_mode == "paper":
            initial_status = "paper"
        elif portal and portal.paper_mode_default:
            initial_status = "paper"
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
        trades = []
        for e in execs:
            dur = None
            if e.fired_at and e.closed_at:
                dur = int((e.closed_at - e.fired_at).total_seconds() / 60)
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
    from app.database import SessionLocal
    db = SessionLocal()
    try:
        user = _get_user_by_uid(uid, db)
        if not user:
            raise HTTPException(status_code=403)
        from app.strategy_models import UserStrategy, StrategyExecution, StrategyPerformance
        strategies = db.query(UserStrategy).filter(UserStrategy.user_id == user.id).all()
        all_execs  = []
        for strat in strategies:
            execs = db.query(StrategyExecution).filter(
                StrategyExecution.strategy_id == strat.id
            ).all()
            all_execs.extend(execs)

        closed = [e for e in all_execs if e.outcome in ("WIN", "LOSS", "BREAKEVEN") and e.pnl_pct is not None]
        wins   = len([e for e in closed if e.outcome == "WIN"])
        total  = len(closed)

        # Rolling 7-day P&L
        from datetime import datetime, timedelta
        cutoff_7d  = datetime.utcnow() - timedelta(days=7)
        cutoff_30d = datetime.utcnow() - timedelta(days=30)
        pnl_7d  = sum(e.pnl_pct for e in closed if (e.closed_at or e.fired_at) > cutoff_7d)
        pnl_30d = sum(e.pnl_pct for e in closed if (e.closed_at or e.fired_at) > cutoff_30d)
        pnl_all = sum(e.pnl_pct for e in closed)

        # Daily P&L breakdown (last 30 days for chart)
        from collections import defaultdict
        daily = defaultdict(float)
        for e in closed:
            if (e.closed_at or e.fired_at) > cutoff_30d:
                day = (e.closed_at or e.fired_at).strftime("%m/%d")
                daily[day] += e.pnl_pct or 0
        # Sort by date
        sorted_daily = sorted(daily.items())
        cumulative   = 0.0
        port_labels  = []
        port_values  = []
        for day, pnl in sorted_daily:
            cumulative += pnl
            port_labels.append(day)
            port_values.append(round(cumulative, 2))

        # Active strategies and exposure
        active = [s for s in strategies if s.status == "active"]
        open_trades = [e for e in all_execs if e.outcome == "OPEN"]

        return JSONResponse({
            "total_strategies": len(strategies),
            "active_count":     len(active),
            "open_trades":      len(open_trades),
            "total_trades":     total,
            "win_rate":         round(wins / total * 100, 1) if total > 0 else 0,
            "pnl_7d":           round(pnl_7d, 2),
            "pnl_30d":          round(pnl_30d, 2),
            "pnl_all":          round(pnl_all, 2),
            "equity_30d":       {"labels": port_labels, "values": port_values},
        })
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
        for k in ("leverage", "position_size_pct", "max_trades_per_day", "cooldown_minutes",
                  "max_open_positions", "daily_loss_limit_pct"):
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
            s.status = body["status"]

        s.config = config
        db.commit()

        # Notify admin via Telegram whenever a strategy is promoted to live
        if s.status == "active" and prev_status != "active":
            import asyncio
            asyncio.create_task(_notify_admin_go_live(user, s))

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
        client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        resp = client.messages.create(
            model="claude-sonnet-4-5",
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
        return {"success": True, "tier": "pro", "subscription_end": sub.subscription_end.isoformat()}
    finally:
        db.close()


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
        client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        resp = client.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=350,
            system=system_prompt,
            messages=api_messages,
        )
        return {"reply": resp.content[0].text, "pro_required": False}
    except Exception as e:
        logger.error(f"Strategy advisor AI error: {e}")
        return {"reply": "Sorry, I hit an issue — please try again.", "pro_required": False}


@app.get("/health")
async def health():
    return {"status": "ok", "service": "strategy-portal"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)
