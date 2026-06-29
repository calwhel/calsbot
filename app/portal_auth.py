"""Shared portal session auth (no imports from strategy_portal_server)."""
from __future__ import annotations

from typing import Optional
from urllib.parse import quote

from fastapi import Request
from fastapi.responses import RedirectResponse

from app.portal_session import COOKIE_NAME, delete_session_cookie, session_uid_from_request


def normalize_portal_uid(uid: str) -> str:
    uid = (uid or "").strip().upper()
    if uid and not uid.startswith("TH-"):
        uid = f"TH-{uid}"
    return uid


def resolve_session_uid_with_user(request: Request) -> Optional[str]:
    """Return session UID only when the token is valid and the user row exists."""
    uid = session_uid_from_request(request)
    if not uid:
        return None
    uid = uid.strip().upper()
    from app.database import SessionLocal
    from app.models import User

    db = SessionLocal()
    try:
        user = db.query(User).filter(User.uid == uid).first()
        return uid if user else None
    finally:
        db.close()


def login_redirect(next_path: str, *, msg: str = "session_expired") -> RedirectResponse:
    safe_next = (next_path or "/app").strip()
    if not safe_next.startswith("/"):
        safe_next = "/app"
    return RedirectResponse(
        url=f"/login?next={quote(safe_next, safe='')}&msg={msg}",
        status_code=302,
    )


def stale_session_redirect(request: Request | None = None) -> RedirectResponse:
    resp = RedirectResponse(url="/login?msg=session_stale", status_code=302)
    delete_session_cookie(resp, request)
    return resp


def session_token_from_request(request: Request, uid: str) -> Optional[str]:
    """Return the client's existing session token when it matches ``uid``."""
    uid = normalize_portal_uid(uid)
    if session_uid_from_request(request) != uid:
        return None
    auth = request.headers.get("Authorization", "")
    if auth.lower().startswith("bearer "):
        return auth[7:].strip()
    hdr = request.headers.get("X-TradeHub-Session", "").strip()
    if hdr:
        return hdr
    return request.cookies.get(COOKIE_NAME) or None
