"""HMAC session tokens shared by portal and Gold AI Trader (no circular imports)."""
from __future__ import annotations

import hashlib
import hmac
import os
import secrets
from typing import Optional

from fastapi import Request

from app.deployment import request_is_https

_COOKIE_NAME = "th_session"
_COOKIE_MAX_AGE = 60 * 60 * 24 * 30
_cached_secret: Optional[str] = None


def cookie_secret() -> str:
    global _cached_secret
    if _cached_secret is None:
        value = (os.getenv("SECRET_KEY") or os.getenv("SESSION_SECRET") or "").strip()
        if value:
            _cached_secret = value
        else:
            db_seed = (
                os.getenv("NEON_DATABASE_URL")
                or os.getenv("DATABASE_URL")
                or os.getenv("RAILWAY_DATABASE_URL")
                or ""
            ).strip()
            if db_seed:
                _cached_secret = hashlib.sha256(f"tradehub-session:{db_seed}".encode()).hexdigest()
            else:
                _cached_secret = secrets.token_urlsafe(32)
    return _cached_secret


def make_session_token(uid: str) -> str:
    uid = (uid or "").strip().upper()
    sig = hmac.new(cookie_secret().encode(), uid.encode(), hashlib.sha256).hexdigest()[:20]
    return f"{uid}:{sig}"


def verify_session_token(token: str) -> Optional[str]:
    if not token or ":" not in token:
        return None
    uid, sig = token.rsplit(":", 1)
    expected = hmac.new(cookie_secret().encode(), uid.encode(), hashlib.sha256).hexdigest()[:20]
    if hmac.compare_digest(sig, expected):
        return (uid or "").strip().upper() or None
    return None


def session_uid_from_request(request: Request) -> Optional[str]:
    cookie = request.cookies.get(_COOKIE_NAME)
    if cookie:
        uid = verify_session_token(cookie)
        if uid:
            return uid
    auth = request.headers.get("Authorization", "")
    token = auth[7:].strip() if auth.lower().startswith("bearer ") else ""
    token = token or request.headers.get("X-TradeHub-Session", "").strip()
    return verify_session_token(token) if token else None


def set_session_cookie(response, uid: str, request: Request = None) -> None:
    uid = (uid or "").strip().upper()
    response.set_cookie(
        key=_COOKIE_NAME,
        value=make_session_token(uid),
        max_age=_COOKIE_MAX_AGE,
        httponly=True,
        samesite="lax",
        secure=request_is_https(request),
    )
