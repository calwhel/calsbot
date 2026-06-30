"""Gemini Gold Trader routes."""
import os

os.environ.setdefault("DATABASE_URL", "sqlite:///:memory:")

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.gemini_gold_trader.routes import router

app = FastAPI()
app.include_router(router)
client = TestClient(app)


def test_gemini_gold_page_requires_session():
    r = client.get("/gemini-gold-trader?uid=TH-YP0BADA8", follow_redirects=False)
    assert r.status_code == 302
    assert "/login" in r.headers.get("location", "")
