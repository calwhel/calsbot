"""Gemini Gold Trader routes."""
import os

os.environ.setdefault("DATABASE_URL", "sqlite:///:memory:")

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.gemini_gold_trader.config import env_defaults
from app.gemini_gold_trader.guardrails import merge_config
from app.gemini_gold_trader.models import GeminiGoldConfig
from app.gemini_gold_trader.routes import _config_payload, _decision_feed, router

app = FastAPI()
app.include_router(router)
client = TestClient(app)


def test_gemini_gold_page_requires_session():
    r = client.get("/gemini-gold-trader?uid=TH-YP0BADA8", follow_redirects=False)
    assert r.status_code == 302
    assert "/login" in r.headers.get("location", "")


def test_config_payload_exposes_kill_switch_sources():
    env = env_defaults()
    env.kill_switch = True
    row = GeminiGoldConfig(id=1, kill_switch=False, enabled=True, dry_run=True)
    cfg = merge_config(row, env)
    payload = _config_payload(cfg, row, env)
    assert payload["kill_switch"] is True
    assert payload["kill_switch_db"] is False
    assert payload["env_kill_switch"] is True
    assert payload["kill_switch_env_locked"] is True


def test_decision_feed_imports_model():
    """Regression: _decision_feed must not NameError on GeminiGoldDecision."""
    from unittest.mock import MagicMock

    db = MagicMock()
    db.query.return_value.order_by.return_value.limit.return_value.all.return_value = []
    assert _decision_feed(db) == []
