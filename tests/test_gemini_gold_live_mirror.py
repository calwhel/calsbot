"""Gemini Gold live mirror config + guardrail tests."""
from __future__ import annotations

import os
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

os.environ.setdefault("DATABASE_URL", "sqlite:///:memory:")

from app.gemini_gold_trader.config import EXECUTION_MODE_DEMO, EXECUTION_MODE_LIVE
from app.gemini_gold_trader.models import GeminiGoldConfig
from app.gemini_gold_trader.routes import router


@pytest.fixture()
def api_client(monkeypatch):
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    GeminiGoldConfig.__table__.create(bind=engine)
    Session = sessionmaker(bind=engine)
    bootstrap = Session()
    bootstrap.add(
        GeminiGoldConfig(
            id=1,
            enabled=True,
            dry_run=False,
            demo_user_id=42,
            demo_ctrader_account_id="111",
            execution_mode=EXECUTION_MODE_DEMO,
            live_ctrader_account_id="222",
            live_mirror_enabled=False,
            max_live_trades_day=3,
        )
    )
    bootstrap.commit()
    bootstrap.close()

    admin = MagicMock(id=42, uid="TH-TEST", is_admin=True)

    monkeypatch.setattr("app.gemini_gold_trader.routes.SessionLocal", Session)
    monkeypatch.setattr(
        "app.gemini_gold_trader.routes.ensure_gemini_gold_trader_schema",
        lambda *a, **k: None,
    )
    monkeypatch.setattr("app.gemini_gold_trader.routes._resolve_user", lambda uid, db: admin)
    monkeypatch.setattr(
        "app.gemini_gold_trader.routes.live_accounts_for_user_id",
        lambda db, uid: [{"ctid": "222", "label": "#222 Live"}],
    )

    app = FastAPI()
    app.include_router(router)
    yield TestClient(app), Session, engine


def test_enable_live_mirror_persists(api_client):
    client, Session, engine = api_client
    r = client.post(
        "/api/gemini-gold-trader/config?uid=TH-TEST",
        json={
            "live_mirror_enabled": True,
            "live_ctrader_account_id": "222",
            "confirm_real_money": True,
        },
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["ok"] is True
    assert body["live_mirror_enabled"] is True
    assert body["config"]["live_mirror_enabled"] is True
    check = Session(bind=engine)
    row = check.query(GeminiGoldConfig).filter_by(id=1).first()
    assert row.live_mirror_enabled is True
    assert row.live_mirror_confirmed_at is not None
    check.close()


def test_enable_live_mirror_requires_confirm(api_client):
    client, _Session, _engine = api_client
    r = client.post(
        "/api/gemini-gold-trader/config?uid=TH-TEST",
        json={"live_mirror_enabled": True, "live_ctrader_account_id": "222"},
    )
    assert r.status_code == 400
    assert "confirm_real_money" in r.json()["detail"]


def test_enable_live_mirror_blocks_live_execution_mode(api_client):
    client, Session, engine = api_client
    check = Session(bind=engine)
    row = check.query(GeminiGoldConfig).filter_by(id=1).first()
    row.execution_mode = EXECUTION_MODE_LIVE
    check.commit()
    check.close()
    r = client.post(
        "/api/gemini-gold-trader/config?uid=TH-TEST",
        json={
            "live_mirror_enabled": True,
            "live_ctrader_account_id": "222",
            "confirm_real_money": True,
        },
    )
    assert r.status_code == 400
    assert "demo execution mode" in r.json()["detail"].lower()


def test_disconnect_live_mirror(api_client):
    client, Session, engine = api_client
    check = Session(bind=engine)
    row = check.query(GeminiGoldConfig).filter_by(id=1).first()
    row.live_mirror_enabled = True
    check.commit()
    r = client.post("/api/gemini-gold-trader/disconnect-live?uid=TH-TEST")
    assert r.status_code == 200
    assert r.json()["live_mirror_enabled"] is False
    check.refresh(row)
    assert row.live_mirror_enabled is False
    assert row.live_mirror_confirmed_at is None
    check.close()


def _demo_cfg():
    from app.gemini_gold_trader.config import env_defaults

    cfg = env_defaults()
    return cfg.__class__(
        **{
            **cfg.__dict__,
            "execution_mode": EXECUTION_MODE_DEMO,
            "live_mirror_enabled": True,
            "demo_user_id": 42,
            "live_ctrader_account_id": "222",
        }
    )


def test_live_mirror_skip_sends_telegram_reason():
    """A blocked live mirror (enabled) must surface the reason on Telegram."""
    import asyncio
    from unittest.mock import AsyncMock, patch

    from app.gemini_gold_trader import executor

    db = MagicMock()
    notify = AsyncMock()

    async def _run():
        with patch.object(
            executor, "is_live_execution_mode", return_value=False
        ), patch(
            "app.gemini_gold_trader.guardrails.check_can_execute_live_mirror",
            return_value=(False, "max_live_open_position"),
        ), patch.object(
            executor, "db_commit", new=AsyncMock()
        ), patch(
            "app.gemini_gold_trader.telegram_notify.notify_live_mirror_skipped",
            new=notify,
        ), patch.object(
            executor, "execute_live_mirror_take", new=AsyncMock(return_value=None)
        ):
            await executor.maybe_live_mirror_after_demo(
                db=db,
                cfg=_demo_cfg(),
                decision={"direction": "LONG", "confidence": 80},
                decision_id=99,
                demo_execution_id=10,
                session="new_york",
            )
        notify.assert_awaited_once()
        kwargs = notify.await_args.kwargs
        assert kwargs["reason"] == "max_live_open_position"
        assert kwargs["status"] == "skipped"

    asyncio.run(_run())
