"""Tests for shared portal session tokens."""
from __future__ import annotations

import os

os.environ.setdefault("DATABASE_URL", "sqlite:///:memory:")

from app.portal_session import make_session_token, verify_session_token


def test_make_and_verify_session_token():
    os.environ["SECRET_KEY"] = "test-secret-for-session"
    try:
        token = make_session_token("TH-YP0BADA8")
        assert token.startswith("TH-YP0BADA8:")
        assert verify_session_token(token) == "TH-YP0BADA8"
        assert verify_session_token("TH-YP0BADA8:bad") is None
    finally:
        os.environ.pop("SECRET_KEY", None)
